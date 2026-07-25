"""Memory-aware auto-sizing for the two likelihood block-size knobs.

The hierarchical likelihood has two static-jit block-size knobs that trade
throughput for peak device memory:

* ``--sel_batch_size`` — injections per selection-integral chunk (``N_sel`` on
  real data is ~1.07e6; a single pass is fastest but the widest working set);
* ``--pe_event_block`` — events per PE-reduction chunk (``N_events`` ~259, each
  with ``n_samp`` ~4096 posterior samples).

Historically both defaulted to ``None`` (single pass) and had to be hand-tuned
to avoid an out-of-memory abort on dense dark-siren catalogs or small GPUs.
This module resolves them automatically from a probed free-memory budget and a
small **linear peak-memory model** whose slopes are calibrated once per device
class by ``scripts/benchmark_block_sizes.py`` (see ``docs/source/performance.md``).

The resolution is deliberately conservative and *falls back to today's exact
behavior* — a single pass (``None``) — whenever the budget suffices, so a run on
a large idle GPU is bit-for-bit unchanged.  Explicit values always win; ``auto``
only fills in a value the user did not pin.

Design notes
------------
* ``resolve_block_sizes`` is **pure** (no JAX, no device access): the caller
  passes a probed ``free_bytes`` and ``backend`` so the whole policy is unit
  testable on CPU with injected budgets.  It emits a single **loud diagnostic**
  only on the rare edge cases where the measured static state forces a block
  below the usual floor, or leaves no residual budget at all (still deterministic
  in its returned plan).  ``probe_device_memory_bytes`` (moved here from
  ``cli/analyze.py`` so both share one probe) is the only device-touching
  function and imports JAX lazily.
* ``measure_static_state_bytes`` sums the size of the loaded device-resident
  data-constants (duck-typed ``.nbytes``, no JAX import) and adds an analytic
  estimate for the factory-built KDE caches / ``base_miss`` curves, so the peak
  model sees the allocations that dominate real dark-siren runs.
* The selection axis is blocked **first** (it dominates: ~1.07e6 injections),
  the PE axis only if a floored selection block still overflows.  Blocks are an
  **even split** (``k`` equal chunks) rounded up to a multiple of 256 rather
  than a power-of-two, because ``N_sel`` pads badly against power-of-two blocks
  (1,067,946 → a 1,048,576 block wastes a whole near-empty second chunk).
"""

from __future__ import annotations

import math
from argparse import ArgumentTypeError
from dataclasses import dataclass

# ── Calibrated peak-memory model ────────────────────────────────────────────────
#
# Peak device bytes during one likelihood value+grad are modelled as
#
#     peak ≈ TRUE_FIXED_BYTES                       (JAX/XLA runtime + workspace)
#            + static_state_bytes                   (resident data-constants; MEASURED)
#            + sel_batch  * sel_bpi(dims)           (transient selection working set)
#            + pe_block   * n_samp * pe_bps(dims)   (transient PE-reduction working set)
#
# where ``sel_batch`` defaults to the full ``N_sel`` and ``pe_block`` to the full
# ``N_events`` (a single pass).  ``static_state_bytes`` is the summed size of the
# device-resident data-constants for THIS run (compact catalog views, KDE caches,
# logQ member tables, marks, ...); the CLI measures the loaded arrays and adds an
# analytic estimate for factory-built state (see :func:`measure_static_state_bytes`).
# The two transient slopes scale with the dominant per-block dimensions — grid
# nodes ``n_grid`` and, for catalog runs, galaxies-per-row and catalog count ``K`` —
# RELATIVE to the calibration config, so the calibrated point is preserved (every
# scale factor is 1 there).  The ``_CAT`` variants apply when a galaxy catalog is
# loaded (dark sirens), whose per-injection / per-sample redshift-prior state is
# heavier than the catalog-free spectral path.
#
# ── Why the split (PERF-4) ──
# The original model folded ALL static state into a single ``FIXED_OVERHEAD``
# calibrated on ONE spectral (catalog-free) config, so the OOM-avoidance feature
# was blind to the KDE caches / compact catalog views / logQ member tables that
# DOMINATE real dark-siren runs.  ``FIXED_OVERHEAD`` is now DECOMPOSED into a
# true-fixed runtime term plus that config's (tiny) static state, and the
# configuration-dependent state is made explicit and measured.  The MEASURED
# spectral slopes are UNCHANGED: this is an additive refinement, not a recalibration.
#
# MEASURED slopes (NVIDIA H100-80GB, scripts/benchmark_block_sizes.py --repeats 10,
# 2026-07-21; scripts/benchmarks/block_sizes_h100_80gb.json) for the real spectral
# likelihood value+grad (N_sel=1,067,946; N_events=259; n_samp=4096, BFC allocator,
# PREALLOCATE=false).  Two findings drove these numbers (see a8ba5e7):
#   1. A full single pass peaks at 78.8 GB — NOT the ~40 GB the old placeholder
#      predicted; the calibrated model now predicts ~80 GB so auto blocks on any
#      <~115 GB card (0.7*free rule).
#   2. Peak is a NARROW BAND (min 70.4 .. max 78.8 GB), and mid-range sel blocks can
#      peak *above* the single pass (XLA fuses the single pass better).  So the two
#      slopes are small: blocking buys little, the real lever is card size.
# The slopes are the reducible (min-block -> single-pass) marginals rounded up
# (SEL 7,865 -> 8,000 /inj; PE 8,102 -> 9,000 /samp).  The _CAT (dark-siren) path
# was NOT measured — its slopes stay a 2x scaled estimate, now normalised to a
# reference catalog config (CAL_MAX_GALS_PER_ROW below).  Policy tests inject
# budgets and assert structural properties, independent of these exact numbers.
CONSTANTS_VERSION = "measured-h100-80gb-decomposed-2026-07-22"

SEL_BYTES_PER_INJECTION = 8_000        # spectral / catalog-free selection integral (measured 7,865)
SEL_BYTES_PER_INJECTION_CAT = 16_000   # dark-siren selection integral (catalog state; 2x, UNMEASURED)
PE_BYTES_PER_SAMPLE = 9_000            # spectral per-event PE reduction, per sample (measured 8,102)
PE_BYTES_PER_SAMPLE_CAT = 18_000       # dark-siren per-event PE reduction, per sample (2x, UNMEASURED)

# ── FIXED-overhead decomposition (calibration-point preserving) ─────────────────
# a8ba5e7 calibrated a single FIXED_OVERHEAD of 58 GiB on the spectral config so
# the single-pass prediction sat just above the measured 78.8 GB peak:
#     58 GiB (62.277 GB) + N_sel*8000 + N_events*n_samp*9000
#       = 62.277 + 8.544 + 9.548  ≈ 80.37 GB   (err-high of the 78.8 GB measurement).
# We keep that EXACT anchor and split it into a true-fixed runtime term and the
# spectral config's own static state:
#
#     FIXED_OVERHEAD_BYTES  =  TRUE_FIXED_BYTES        +  STATIC_STATE_CAL_BYTES
#     58 GiB                =  (JAX/XLA + workspace)    +  (spectral config inputs)
#
# The spectral calibration config has NO catalog, so its only static state is the
# sample fields it loads: 5 float64 PE fields (m1det, m2det, dL, chieff, p_pe) of
# shape (N_events, n_samp) and 5 float64 selection fields (m1detsels, m2detsels,
# dLsels, chieffsels, p_draw) of shape (N_sel,).  With static_state_bytes set to
# STATIC_STATE_CAL_BYTES and n_grid == CAL_N_GRID the new budget reproduces the old
# one BIT-FOR-BIT:
#     budget = f*free - TRUE_FIXED - STATIC_STATE_CAL == f*free - FIXED_OVERHEAD.
FIXED_OVERHEAD_BYTES = 58 * 1024**3    # a8ba5e7 calibration anchor (retained; = TRUE_FIXED + STATIC_STATE_CAL)

# Calibration-config dimensions — the reference the slopes and static state
# normalise to (every scale factor below is 1.0 at these values).
CAL_N_SEL = 1_067_946
CAL_N_EVENTS = 259
CAL_N_SAMP = 4096
CAL_N_GRID = 1000                      # zgrid nodes at the default DARKSIRENS_ZMAX=5
CAL_MAX_GALS_PER_ROW = 2113            # reference max galaxies / unique pixel for the _CAT slopes

# Static state of the spectral calibration config: 5 f64 PE fields + 5 f64 sel fields.
STATIC_STATE_CAL_BYTES = (
    5 * 8 * CAL_N_EVENTS * CAL_N_SAMP      # PE samples: m1det, m2det, dL, chieff, p_pe
    + 5 * 8 * CAL_N_SEL                    # selection:  m1detsels, m2detsels, dLsels, chieffsels, p_draw
)                                          # ≈ 85.15 MB (0.0793 GiB)
TRUE_FIXED_BYTES = FIXED_OVERHEAD_BYTES - STATIC_STATE_CAL_BYTES   # ≈ 57.92 GiB JAX/XLA + workspace

# Floors: never chunk below these (below them the launch/padding overhead and
# recompile churn cost more than the memory they save).
SEL_MIN_BATCH = 32768
PE_MIN_BLOCK = 8
# Hard floor when even ``SEL_MIN_BATCH`` / ``PE_MIN_BLOCK`` overflow the residual
# budget under the measured static state: we keep blocking (with a loud warning)
# down to this many units rather than silently exceed the budget / OOM.
BLOCK_HARD_MIN = 1

# Fraction of probed free memory the working set may occupy (headroom for
# fragmentation, transient copies, and — on a shared box — another process's
# growth).  Per-process peak is what we bound; see the contention note in
# docs/source/performance.md.
SAFETY_FACTOR = 0.7

# Even-split blocks are rounded up to a multiple of this (keeps XLA tile shapes
# friendly and minimises the padding of the final chunk for N_sel ~1e6).
BLOCK_ROUND_TO = 256

_GIB = 1024**3  # for human-readable diagnostics only

# GPU-class backends that actually benefit from blocking; on everything else a
# single pass is both correct and simplest (host RAM is large / not the bottleneck).
_GPU_BACKENDS = frozenset({"gpu", "cuda", "rocm"})

# Sentinel stored on ``opts`` for an unresolved "auto" knob.  A plain string so
# it round-trips through ``vars(opts)`` → settings.json without a custom encoder;
# ``resolve_block_sizes`` replaces it with a concrete int / None before the
# factory ever reads it (the factory raises if it sees the sentinel).
BLOCK_AUTO = "auto"


def block_size_arg(value):
    """argparse ``type`` for the block-size knobs.

    ``"auto"`` → :data:`BLOCK_AUTO` (resolved post-data-load); ``"off"``/``"none"``
    /``"0"`` → ``None`` (an explicit single pass, today's behavior); a positive
    integer → that block size.  Anything else raises ``ArgumentTypeError``.
    """
    if value is None:
        return BLOCK_AUTO
    text = str(value).strip().lower()
    if text == BLOCK_AUTO:
        return BLOCK_AUTO
    if text in ("off", "none", "0"):
        return None
    try:
        n = int(text)
    except ValueError:
        raise ArgumentTypeError(
            f"block size must be 'auto', 'off'/'none'/'0', or a positive integer; "
            f"got {value!r}."
        )
    if n <= 0:
        raise ArgumentTypeError(f"block size must be a positive integer; got {n}.")
    return n


def format_block_size_request(value) -> str:
    """Human-readable form of an *unresolved* block-size request for the run-config
    table (printed before the data load, so ``value`` may be the ``auto`` sentinel)."""
    if value is BLOCK_AUTO or value == BLOCK_AUTO:
        return "auto (resolved after data load)"
    if value is None:
        return "off (single pass)"
    return f"{int(value):,} (pinned)"


def require_resolved_block_size(name: str, value):
    """Raise if a block-size knob still holds the ``auto`` sentinel string.

    The factory reads plain ``int``/``None`` block sizes; the CLI must call
    :func:`resolve_block_sizes` (post data load) to replace the ``auto`` sentinel
    first.  A late, clear error beats an opaque failure deep in a jit trace.
    """
    if isinstance(value, str):
        raise TypeError(
            f"{name}={value!r} is still the 'auto' sentinel; resolve_block_sizes() "
            "must run (after data load) before make_likelihood()."
        )
    return value


@dataclass(frozen=True)
class BlockSizePlan:
    """Resolved block sizes plus a one-word provenance tag for the log/settings."""
    sel_batch_size: int | None
    pe_event_block: int | None
    # "explicit" | "auto" | "auto-single-pass" | "auto-floor-reduced" | "cpu" | "flow"
    source: str


#: Environment override for the device memory budget, in GB.  Set this to pin
#: the budget without touching the allocator (mirrors the analyze CLI's
#: ``DARKSIRENS_ANALYZE_MAX_MEM_GB``).
DEVICE_MEM_ENV_VAR = "DARKSIRENS_DEVICE_MEM_GB"


def _nvidia_smi_free_bytes(device_index):
    """Free bytes on GPU ``device_index`` via ``nvidia-smi``, or ``None``.

    ``memory_stats()`` returns ``None`` under ``XLA_PYTHON_CLIENT_ALLOCATOR=
    platform`` (which ``core.jax_config`` sets), so on a production CUDA run it
    is the only probe that reports anything at all.  Deliberately subprocess and
    not NVML: pynvml is not a declared dependency.

    ``nvidia-smi`` indexes GPUs the same way CUDA does *after*
    ``CUDA_VISIBLE_DEVICES`` filtering is undone, so the visible-device list is
    applied here to map a JAX device index back to a physical one.
    """
    import os
    import subprocess

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        entries = [e.strip() for e in visible.split(",") if e.strip()]
        # A UUID-based list cannot be indexed positionally against nvidia-smi's
        # integer indices; query by that UUID instead.
        if device_index >= len(entries):
            return None
        target = entries[device_index]
    else:
        target = str(device_index)

    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={target}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except Exception:
        return None

    line = out.strip().splitlines()[0].strip() if out.strip() else ""
    try:
        mib = float(line)
    except ValueError:
        return None
    if not (mib > 0):
        return None
    return int(mib * 1024 * 1024)


def probe_device_memory_bytes(default_gb=4.0):
    """Best-effort free-memory probe for the first JAX device.

    Returns ``(bytes, source)``, trying in order:

    1. ``$DARKSIRENS_DEVICE_MEM_GB`` — explicit override, always wins.
    2. ``device.memory_stats()`` (GPU/TPU): ``bytes_limit - bytes_in_use`` if
       both present, else ``bytes_limit`` / ``bytes_reservable_limit``.
    3. ``nvidia-smi`` free memory, for CUDA devices.
    4. ``default_gb``.

    Step 3 exists because step 2 is INERT in production: ``core.jax_config``
    sets ``XLA_PYTHON_CLIENT_ALLOCATOR=platform``, under which
    ``memory_stats()`` returns ``None``.  Without it every CLI run fell through
    to the 4 GB default on machines with far more — measured 4 GB reported
    against 92.4 GB actually free on an H100 NVL — so memory-aware sizing never
    engaged and the calibrated single-pass constants were never used.

    Note: with ``memory_stats`` available and ``XLA_PYTHON_CLIENT_PREALLOCATE``
    unset, JAX preallocates a large fraction of the GPU and ``bytes_limit``
    reports that *pool* (still a safe ceiling for sizing — we stay within it).
    """
    import os

    override = os.environ.get(DEVICE_MEM_ENV_VAR)
    if override:
        try:
            gb = float(override)
            if gb > 0:
                return int(gb * 1e9), f"env:{DEVICE_MEM_ENV_VAR}"
        except ValueError:
            pass

    try:
        import jax  # lazy: this module is imported by the (JAX-free) argparse layer
        dev = jax.devices()[0]
        stats = dev.memory_stats() or {}
        limit = stats.get("bytes_limit")
        if limit:
            in_use = int(stats.get("bytes_in_use", 0) or 0)
            free = max(int(limit) - in_use, int(0.1 * int(limit)))
            return int(free), f"device:{dev.platform} bytes_limit-in_use"
        res = stats.get("bytes_reservable_limit")
        if res:
            return int(res), f"device:{dev.platform} bytes_reservable_limit"
        if str(dev.platform).lower() in _GPU_BACKENDS:
            free = _nvidia_smi_free_bytes(int(getattr(dev, "id", 0) or 0))
            if free:
                return int(free), "nvidia-smi memory.free"
    except Exception:
        pass
    return int(default_gb * 1e9), "fallback-default"


def _even_split_block(n_total: int, budget_bytes: float, bytes_per_unit: float,
                      floor: int, round_to: int = BLOCK_ROUND_TO) -> int:
    """Largest even-split block of ``n_total`` units fitting ``budget_bytes``.

    Splits into ``k = ceil(n_total / B_fit)`` equal chunks (``B_fit`` = units that
    fit the budget), then rounds the resulting ``ceil(n_total / k)`` up to a
    multiple of ``round_to`` and applies ``floor``.  Even chunks minimise the
    padding of the final block versus a fixed power-of-two size.
    """
    bytes_per_unit = max(1.0, float(bytes_per_unit))
    b_fit = max(1, int(budget_bytes // bytes_per_unit))
    if b_fit >= n_total:
        return n_total
    k = max(1, math.ceil(n_total / b_fit))
    block = math.ceil(n_total / k)
    block = int(math.ceil(block / round_to) * round_to)
    block = max(floor, block)
    return min(block, n_total)


def _loud(message: str) -> None:
    """Print a single, hard-to-miss diagnostic for a block-sizing edge case."""
    print(f"  [!] block-sizing: {message}")


def _floored_block(n_total: int, place_budget: float, guard_room: float,
                   bytes_per_unit: float, floor: int, name: str,
                   static_state_bytes: float,
                   round_to: int = BLOCK_ROUND_TO) -> tuple[int, bool]:
    """Size a block of ``n_total`` units, reducing the floor only on true
    static-state dominance.

    Returns ``(block, floor_reduced)``.  Two distinct budgets are used, and the
    distinction matters:

    * ``place_budget`` = ``safety_factor*free - TRUE_FIXED - static`` chooses the
      block size via :func:`_even_split_block`, floored at ``floor``.  This is the
      calibration-preserving, deliberately conservative placement: when it is
      negative (common for a catalog run whose spectral-calibrated ``TRUE_FIXED``
      over-predicts the transient floor) the even split simply returns ``floor`` —
      exactly the historical behavior.
    * ``guard_room`` = ``safety_factor*free - static`` is the LAST-RESORT floor
      guard, where ``static`` is the PENDING (not-yet-resident) static state the
      probe is blind to (factory KDE caches / base_miss; the loaded arrays are
      already reflected in ``free``).  It intentionally EXCLUDES ``TRUE_FIXED`` —
      that term is the spectral value+grad's transient workspace floor (measured
      without a catalog) and a known over-estimate of the catalog path's floor, so
      letting it gate the minimum block would needlessly shrink blocks on runs that
      fit fine.  The floor is only dropped when the pending static plus a floor-sized
      block's working set will not fit the remaining device memory — genuine
      static-state dominance — in which case we reduce to the largest feasible block
      (down to :data:`BLOCK_HARD_MIN`) and warn loudly.  Erring toward a
      smaller-but-feasible block never OOMs.
    """
    bytes_per_unit = max(1.0, float(bytes_per_unit))
    floor = max(BLOCK_HARD_MIN, min(int(floor), int(n_total)))
    if floor * bytes_per_unit <= guard_room:
        # The floor's working set fits alongside the resident static state: honor
        # the floor and size by the (safety-discounted) placement budget.
        return _even_split_block(int(n_total), max(1.0, place_budget),
                                 bytes_per_unit, floor=floor, round_to=round_to), False
    # Even a floor-sized block will not fit next to the measured static state.
    b_fit = int(max(0.0, guard_room) // bytes_per_unit)
    reduced = max(BLOCK_HARD_MIN, min(floor, b_fit))
    detail = (
        "no device room after the measured static state"
        if guard_room <= 0
        else f"only {guard_room / _GIB:.2f} GiB left after static state"
    )
    _loud(
        f"{name} floor {floor:,} × {bytes_per_unit:,.0f} B/unit does not fit — "
        f"{detail} (static ≈ {static_state_bytes / _GIB:.1f} GiB dominates the "
        f"device). Dropping the floor to {reduced:,}. The peak-memory model may be "
        f"infeasible on this device — the run may still OOM."
    )
    return reduced, True


def estimate_pending_static_bytes(data, *, n_grid: int, has_catalog: bool,
                                  catalog_memory=None) -> int:
    """Analytic estimate of the static state the FACTORY will allocate *after* the
    block-size resolver runs — i.e. the piece the device probe is blind to.

    The resolver runs AFTER the data load, so the loaded arrays are ALREADY
    device-resident and hence already reflected in the probed ``free_bytes`` (they
    dropped it).  What is NOT yet allocated is what the likelihood factory builds:

    * **KDE cache** — ``build_pixel_kde_cache`` produces a ``(n_unique, n_grid)``
      float64 table (8 B) per view; ``(unique_pe + unique_sel) * n_grid * 8``
      (conservatively counts both even when the two views share one table).
    * **base_miss** — the completion ensemble carries a ``(N_rows, n_grid)`` f64
      curve per view (PE + sel); estimated from the loaded
      ``lss_completion_logq_members`` row count when a completion ensemble is
      present (the ``(M, N_rows, n_grid)`` member table itself is a LOADED array,
      already resident — it is not counted here).

    This is the quantity to subtract from the memory budget / reserve in the floor
    guard; subtracting the already-resident loaded arrays too would double-count
    them against ``free_bytes``.  Pure Python (no JAX import).
    """
    if not has_catalog:
        return 0
    cm = catalog_memory or (data.get("catalog_memory") if isinstance(data, dict) else None) or {}
    n_unique = int(cm.get("unique_pe_pixels", 0)) + int(cm.get("unique_sel_pixels", 0))
    if n_unique <= 0 and isinstance(data, dict):
        # Best-effort fallback: derive union-row counts from the compact views.
        for k in ("zgals_pe", "zgals_sel"):
            shape = getattr(data.get(k), "shape", None)
            if shape:
                n_unique += int(shape[0])
    pending = n_unique * int(n_grid) * 8         # KDE cache(s), f64

    # base_miss (N_rows, n_grid) f64 per view (PE + sel) for a completion ensemble.
    members = data.get("lss_completion_logq_members") if isinstance(data, dict) else None
    shape = getattr(members, "shape", None)
    if shape and len(shape) >= 2:
        pending += 2 * int(shape[-2]) * int(n_grid) * 8
    return int(pending)


def measure_static_state_bytes(data, *, n_grid: int, has_catalog: bool,
                               n_catalogs: int = 1, catalog_memory=None,
                               drop_full_catalog: bool = False) -> int:
    """Total static data-constant bytes for a loaded run (for the run-config report).

    Sums the exact ``nbytes`` of every loaded device-resident array — deduplicated
    by object identity (``data['zgals']`` and ``data['zgals_catalog']`` alias one
    buffer; a bundle's PE and selection views alias one galaxy table) — PLUS the
    analytic :func:`estimate_pending_static_bytes` for the factory-built KDE caches
    / ``base_miss`` curves.  This is the human-facing "how big is the static state"
    number; the resolver subtracts only the *pending* portion (see that function's
    note on double-counting).  Pure Python: duck-types ``.nbytes``, no JAX import.
    """
    seen: set[int] = set()
    total = 0

    def _add(value):
        nonlocal total
        nbytes = getattr(value, "nbytes", None)
        if nbytes is None:
            return
        key = id(value)
        if key in seen:
            return
        seen.add(key)
        total += int(nbytes)

    def _walk(container):
        if container is None:
            return
        values = container.values() if isinstance(container, dict) else container
        for value in values:
            if isinstance(value, dict) or isinstance(value, (list, tuple)):
                _walk(value)
            else:
                _add(value)

    if isinstance(data, dict):
        # Top-level arrays (dedup by identity handles the zgals/zgals_catalog and
        # PE/sel view aliasing).  Per-catalog bundles (K >= 2) are walked too.
        for key, value in data.items():
            if key == "catalogs":
                _walk(value)  # list of per-catalog bundle dicts
            elif isinstance(value, (dict, list, tuple)):
                _walk(value)
            else:
                _add(value)

    total += estimate_pending_static_bytes(
        data, n_grid=n_grid, has_catalog=has_catalog, catalog_memory=catalog_memory)
    return int(total)


def resolve_block_sizes(*, n_events: int, n_samp: int, n_sel: int,
                        sel_requested, pe_requested, has_catalog: bool,
                        flow_path: bool, n_grid: int = CAL_N_GRID,
                        max_gals_per_row: int = CAL_MAX_GALS_PER_ROW,
                        n_catalogs: int = 1, static_state_bytes: float = 0.0,
                        free_bytes: int | None = None,
                        free_bytes_reliable: bool = True,
                        backend: str | None = None,
                        safety_factor: float = SAFETY_FACTOR) -> BlockSizePlan:
    """Resolve ``(sel_batch_size, pe_event_block)`` from a memory budget.

    ``sel_requested`` / ``pe_requested`` are the parsed CLI values: an ``int``
    (explicit, passes through), ``None`` (an explicit single pass), or
    :data:`BLOCK_AUTO` (resolve here).  Explicit values always win per knob.

    The peak model is
    ``TRUE_FIXED_BYTES + static_state_bytes + sel_batch*sel_bpi + pe_block*n_samp*pe_bps``;
    the transient slopes ``sel_bpi`` / ``pe_bps`` scale with the dominant per-block
    dimensions relative to the calibration config:

    * ``n_grid / CAL_N_GRID`` on both slopes (more grid nodes → more work/bytes);
    * for catalog runs, ``(max_gals_per_row / CAL_MAX_GALS_PER_ROW) * n_catalogs``
      on the heavier ``_CAT`` slopes (more galaxies per row and a K-catalog
      mixture both multiply the redshift-prior work).

    Auto policy: on a non-GPU backend keep a single pass; else block the selection
    axis first and the PE axis only if a floored selection block still overflows
    ``safety_factor * free_bytes - TRUE_FIXED_BYTES - static_state_bytes``.  A
    single pass that already fits resolves to ``(None, None)`` — bit-identical to
    the historical default.  With ``static_state_bytes == STATIC_STATE_CAL_BYTES``,
    ``n_grid == CAL_N_GRID`` and no catalog the budget equals the pre-decomposition
    ``safety_factor*free - FIXED_OVERHEAD_BYTES`` exactly (calibration preserved).
    """
    sel_auto = sel_requested is BLOCK_AUTO
    pe_auto = pe_requested is BLOCK_AUTO

    # Explicit passthrough (int or an explicit-off None) is honoured verbatim.
    sel = None if sel_auto else sel_requested
    pe = None if pe_auto else pe_requested

    # The flow-surrogate spectral path has no per-event PE reduction to chunk.
    if flow_path:
        pe = None
        pe_auto = False

    if not (sel_auto or pe_auto):
        return BlockSizePlan(sel, pe, "explicit")

    # Blocking only helps on a GPU-class device; elsewhere a single pass stands.
    if backend is not None and str(backend).lower() not in _GPU_BACKENDS:
        return BlockSizePlan(
            None if sel_auto else sel,
            None if pe_auto else pe,
            "cpu",
        )

    if free_bytes is None:
        free_bytes, _probe_src = probe_device_memory_bytes()
        # A failed probe (no device memory_stats — common unless
        # XLA_PYTHON_CLIENT_PREALLOCATE=false) returns a small default; it is NOT
        # trustworthy evidence of memory pressure, so it must not trigger floor
        # reduction (see below).  The placement budget still uses it and simply
        # falls back to the floor — the historical behavior.
        free_bytes_reliable = _probe_src != "fallback-default"
    # ``static_state_bytes`` is the PENDING static state — what the factory will
    # allocate after this probe (KDE caches / base_miss).  The already-loaded arrays
    # are device-resident by now and thus already reflected in ``free_bytes``;
    # subtracting them again would double-count.  (The CLI passes the pending
    # estimate; unit tests may pass any reserve.)
    static_state_bytes = max(0.0, float(static_state_bytes))
    # Placement budget: safety-discounted, minus the transient value+grad floor and
    # the pending static.  Drives even-split block sizing; when it goes negative
    # (a catalog run whose spectral-calibrated TRUE_FIXED over-predicts the floor)
    # the even split falls back to the floor — the historical behavior.
    budget = max(
        1.0,
        float(safety_factor) * float(free_bytes) - TRUE_FIXED_BYTES - static_state_bytes,
    )
    # Floor-reduction guard: room for block working sets AFTER the pending static,
    # EXCLUDING the (spectral-over-estimated) transient TRUE_FIXED.  Only a genuinely
    # device-dominating static state reduces the minimum block (see _floored_block);
    # otherwise the floor stands even when ``budget`` is negative.  An UNRELIABLE
    # free-memory reading (failed probe) gives no basis to reduce, so the guard is
    # disabled (guard_room = ∞) and the floor stands — matching the historical model.
    guard_room = (
        float(safety_factor) * float(free_bytes) - static_state_bytes
        if free_bytes_reliable else math.inf
    )

    # Dimension scaling of the transient slopes, relative to the calibration config.
    grid_scale = max(1e-9, float(n_grid) / float(CAL_N_GRID))
    if has_catalog:
        cat_scale = (
            max(1e-9, float(max_gals_per_row) / float(CAL_MAX_GALS_PER_ROW))
            * max(1, int(n_catalogs))
        )
        sel_bpi = SEL_BYTES_PER_INJECTION_CAT * grid_scale * cat_scale
        pe_bps = PE_BYTES_PER_SAMPLE_CAT * grid_scale * cat_scale
    else:
        sel_bpi = SEL_BYTES_PER_INJECTION * grid_scale
        pe_bps = PE_BYTES_PER_SAMPLE * grid_scale

    # PE single-pass footprint (fixed unless we end up blocking PE below).
    pe_full_bytes = float(n_events) * float(n_samp) * pe_bps if not flow_path else 0.0
    sel_full_bytes = float(n_sel) * sel_bpi

    source = "auto-single-pass"

    # ── Selection axis (blocked first; it dominates) ──
    if sel_auto:
        sel_budget = budget - pe_full_bytes
        if sel_full_bytes <= sel_budget:
            sel = None                       # single pass fits → unchanged behavior
        else:
            sel, reduced = _floored_block(
                int(n_sel), sel_budget, guard_room, sel_bpi,
                floor=min(SEL_MIN_BATCH, int(n_sel)), name="selection",
                static_state_bytes=static_state_bytes)
            source = "auto-floor-reduced" if reduced else "auto"

    # ── PE axis (only if a floored selection block still overflows) ──
    if pe_auto and not flow_path:
        sel_bytes_now = (sel_full_bytes if sel is None
                         else float(sel) * sel_bpi)
        if sel_bytes_now + pe_full_bytes <= budget:
            pe = None                        # PE single pass fits alongside sel
        else:
            pe_budget = budget - sel_bytes_now
            pe, reduced = _floored_block(
                int(n_events), pe_budget, guard_room, float(n_samp) * pe_bps,
                floor=min(PE_MIN_BLOCK, int(n_events)), name="PE",
                static_state_bytes=static_state_bytes, round_to=1)
            if reduced:
                source = "auto-floor-reduced"
            elif source != "auto-floor-reduced":
                source = "auto"

    return BlockSizePlan(sel, pe, source)
