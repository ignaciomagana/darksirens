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
#
# ── Value-only constant set (PERF-P4) ──
# The set above is the value+GRAD peak.  ``dynesty`` and ``tinyns`` NEVER
# differentiate the likelihood, and the gradient-free peak is 14x smaller: the
# same calibration run records peak_value_bytes = 5.587 GB for the single pass
# against peak_bytes = 78.77 GB for value+grad.  Applying the grad model to a
# gradient-free run therefore blocks a plan that needs no blocking at all —
# MEASURED on the H100 NVL (88.1 GiB probed free, the production probe path):
# the auto plan (32768, 87) costs 49.3 ms/call for a 2.27 GB peak, versus
# 27.5 ms/call and 5.65 GB for the single pass, i.e. 1.79x slower for memory a
# 95 GB card has 17x spare of.  ``needs_grad=False`` selects the value-only set
# below (calibrated in scripts/benchmarks/block_sizes_h100_80gb_value_only.json).
CONSTANTS_VERSION = "measured-h100-80gb-decomposed+valueonly+nq-2026-07-26"

SEL_BYTES_PER_INJECTION = 8_000        # spectral / catalog-free selection integral (measured 7,865)
SEL_BYTES_PER_INJECTION_CAT = 16_000   # dark-siren selection integral (catalog state; 2x, UNMEASURED)
PE_BYTES_PER_SAMPLE = 9_000            # spectral per-event PE reduction, per sample (measured 8,102)
PE_BYTES_PER_SAMPLE_CAT = 18_000       # dark-siren per-event PE reduction, per sample (2x, UNMEASURED)

# Value-only (gradient-free) slopes.  MEASURED on the H100 NVL, spectral config,
# value pass only (scripts/benchmark_block_sizes.py --sampler dynesty --mode value):
#     (off, off)      -> 5.648 GB      (32768, off) -> 5.695 GB
#     (off, 8)        -> 5.648 GB      (32768, 8)   -> 0.703 GB
#     (131072, 32)    -> 1.187 GB      (1048576, off) -> 5.774 GB
# Blocking ONE axis alone barely moves the peak, so the value peak is the MAX of
# the two per-axis working sets, not their sum (see resolve_block_sizes).  Solving
# the two-point system F + B*c = 0.703 GB / F + N_sel*c = 5.648 GB gives
# c = 4,777 B/unit and F = 0.51 GiB; both are rounded UP below.
SEL_BYTES_PER_INJECTION_VALUE = 5_000       # measured 4,777
SEL_BYTES_PER_INJECTION_VALUE_CAT = 10_000  # dark-siren (2x, UNMEASURED)
PE_BYTES_PER_SAMPLE_VALUE = 5_000           # measured 4,777 (the two axes coincide)
PE_BYTES_PER_SAMPLE_VALUE_CAT = 10_000      # dark-siren (2x, UNMEASURED)
TRUE_FIXED_VALUE_BYTES = 1 * 1024**3        # measured intercept 0.51 GiB, rounded up

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
# Population q-quadrature nodes (``--norm_nq`` / DARKSIRENS_GW_N_Q, default 200).
# This is the axis that MULTIPLIES the per-injection / per-sample working set: the
# default pairing model normalises p(q|m1) by an exact per-sample q-integral
# (gw/populations/base.py), so the dominant intermediates of the whole call are
# (N_sel, n_q) / (N_events*n_samp, n_q) tensors — 1.71 GB each at the production
# shapes.  MEASURED value-only peak vs --norm_nq at sel/pe = off/off on the H100
# NVL: 1.34 GB (32), 2.16 (64), 3.09 (100), 5.65 (200), 10.78 (400), 21.03 (800) —
# dead linear at ~26 MB per q-node, i.e. ~93% of the peak is the n_q term.
# ``n_chi`` deliberately does NOT enter: SpinModel._norm integrates the spin grid
# over theta only, producing a (n_chi,) vector, never a per-sample tensor.
CAL_N_Q = 200

# Static state of the spectral calibration config: 5 f64 PE fields + 5 f64 sel fields.
STATIC_STATE_CAL_BYTES = (
    5 * 8 * CAL_N_EVENTS * CAL_N_SAMP      # PE samples: m1det, m2det, dL, chieff, p_pe
    + 5 * 8 * CAL_N_SEL                    # selection:  m1detsels, m2detsels, dLsels, chieffsels, p_draw
)                                          # ≈ 85.15 MB (0.0793 GiB)
TRUE_FIXED_BYTES = FIXED_OVERHEAD_BYTES - STATIC_STATE_CAL_BYTES   # ≈ 57.92 GiB JAX/XLA + workspace

# ── TRUE_FIXED is mostly WORKING SET, not runtime (measured 2026-07-26) ─────────
# ``TRUE_FIXED_BYTES`` was described as "JAX/XLA runtime + workspace", i.e. a term
# INDEPENDENT of the problem dimensions, and only the (small, reducible) slopes were
# scaled by them.  A second n_q point shows that is wrong by a factor of ~25:
#
#     value+grad single-pass peak, spectral config, H100 NVL, MEASURED
#         n_q = 64   ->  26.89 GB      (scripts/.../grad_nq64, this repo)
#         n_q = 200  ->  78.77 GB      (scripts/benchmarks/block_sizes_h100_80gb.json)
#
# Solving ``F + s*W = peak`` with s = n_q/200 gives W = 78.7 GB of DIMENSION-SCALED
# working set and only F = 1.7 GB of genuinely fixed runtime — the peak is ~97 %
# working set at the default n_q, not 77 % fixed.  Left unsplit, the model
# under-predicts badly the moment any working-set dimension grows: at ``--norm_nq
# 400`` on a 141 GB card it would still promise a full single pass against a real
# ~155 GB peak.
#
# So TRUE_FIXED is split here, and the WORKSET part is scaled by the same factor as
# the slopes.  The split is calibration-preserving BY CONSTRUCTION: at the
# calibration dimensions every scale is 1.0, so
# ``GRAD_RUNTIME_FIXED + 1.0*GRAD_WORKSET_FLOOR == TRUE_FIXED_BYTES`` exactly and
# every block plan is bit-for-bit what it was.  It is also self-consistent with the
# measurement: 1.7 + (60.6 + 8.54 + 9.55) = 80.4 GB at n_q=200 (the retained a8ba5e7
# anchor) and 1.7 + 0.32*(60.6 + 18.09) = 26.9 GB at n_q=64 (the new point).
#
# NOTE the VALUE-only fixed term is NOT split: its measured n_q intercept really is
# small and flat (0.52 GB from the 32/64/200-node sweep), which is exactly why
# TRUE_FIXED_VALUE_BYTES is 1 GiB rather than tens of GB.
GRAD_RUNTIME_FIXED_BYTES = 1_700_000_000
GRAD_WORKSET_FLOOR_BYTES = TRUE_FIXED_BYTES - GRAD_RUNTIME_FIXED_BYTES

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


# Samplers that DIFFERENTIATE the likelihood, and therefore incur the value+grad
# peak the default constant set is calibrated on.  Everything else (dynesty,
# tinyns — rwalk / prior replacement only, no HMC kernel) evaluates the value
# alone; ``resolve_block_sizes(needs_grad=False)`` then uses the value-only set.
_GRADIENT_SAMPLERS = frozenset({"numpyro"})


def sampler_block_sizing_profile(opts) -> tuple[bool, int]:
    """``(needs_grad, concurrent_evals)`` for the run's sampler.

    * ``needs_grad`` — whether the sampler differentiates the likelihood.  Only
      NumPyro NUTS does; dynesty and tinyns are gradient-free (this is the same
      distinction ``factory._resolve_redshift_prior_materialization`` already
      draws for the non-differentiable ``optimization_barrier``).
    * ``concurrent_evals`` — how many likelihood evaluations are LIVE AT ONCE, so
      the transient working set is multiplied by it.  tinyns' JAX rwalk kernel
      ``jax.vmap``s the loglike over ``replacement_chains`` proposals (and over
      the largest entry of ``replacement_chain_schedule`` when a schedule is set),
      which batches every intermediate.  ``jax_block_size`` does NOT count: that
      path ``lax.scan``s whole nested-sampling iterations, which are sequential.

    Pure Python (reads plain attributes off ``opts``); no JAX, no device access.
    """
    sampler = str(getattr(opts, "sampler", "") or "").strip().lower()
    needs_grad = sampler in _GRADIENT_SAMPLERS
    concurrent = 1
    if sampler == "tinyns":
        cfg = getattr(opts, "tinyns_resolved_config", None) or {}
        try:
            chains = int(cfg.get("replacement_chains", 1) or 1)
        except (TypeError, ValueError):
            chains = 1
        schedule = cfg.get("replacement_chain_schedule") or ()
        try:
            sched_max = max((int(v) for v in schedule), default=0)
        except (TypeError, ValueError):
            sched_max = 0
        concurrent = max(1, chains, sched_max)
    return needs_grad, concurrent


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


def _is_device_resident(value) -> bool:
    """Duck-typed "this array already lives on the accelerator".

    A ``jax.Array`` exposes ``.devices()``; ``numpy.ndarray`` does not (NumPy >= 2
    grew a scalar ``.device`` ATTRIBUTE, hence the plural method is the test).  Kept
    duck-typed so this module stays JAX-free and unit-testable on CPU.
    """
    return hasattr(value, "devices")


def _rows_and_width(value):
    """``(n_rows, row_width)`` of a 2-D-ish array, or ``(0, 0)``."""
    shape = getattr(value, "shape", None)
    if not shape:
        return 0, 0
    rows = int(shape[0])
    width = int(shape[1]) if len(shape) >= 2 else 1
    return rows, width


def _pending_for_catalog_source(src, *, n_grid: int, catalog_memory=None,
                                catalog_sky_weighting: str = "conditional",
                                is_bundle: bool = False) -> int:
    """Pending factory allocations for ONE catalog source (the flat top-level
    ``data`` dict, or one ``data['catalogs']`` bundle).  See
    :func:`estimate_pending_static_bytes` for the term-by-term rationale."""
    cm = catalog_memory or (src.get("catalog_memory") if isinstance(src, dict) else None) or {}
    rows_pe, width_pe = _rows_and_width(src.get("zgals_pe"))
    rows_sel, width_sel = _rows_and_width(src.get("zgals_sel"))
    n_unique = int(cm.get("unique_pe_pixels", 0)) + int(cm.get("unique_sel_pixels", 0))
    if n_unique <= 0:
        n_unique = rows_pe + rows_sel
    max_gals = int(cm.get("max_galaxies_per_unique_pixel", 0) or 0)
    max_gals = max(max_gals, width_pe, width_sel)

    pending = n_unique * int(n_grid) * 8         # KDE cache(s), f64

    # ── Compact UNION galaxy tables (flat path only) ──
    # ``prepare_catalog_views`` rebuilds the PE-union-selection galaxy tables from
    # the FULL-sky rows at FACTORY time (catalog_views.py: full_z[union_pixels],
    # ...), so a flat run that still carries the full-sky arrays pays 3 f64 tables
    # plus an int32 count vector that the probe never saw.  A per-catalog bundle
    # carries no full-sky rows (the loader compacted them), and
    # ``--drop_full_catalog`` removes them, so neither rebuilds anything.
    if not is_bundle and src.get("zgals") is not None and max_gals > 0:
        union_rows = n_unique                    # |PE ∪ sel| <= unique_pe + unique_sel
        pending += union_rows * max_gals * 3 * 8 + union_rows * 4

    members = src.get("lss_completion_logq_members")
    shape = getattr(members, "shape", None)
    if shape and len(shape) >= 2:
        n_members = int(shape[0]) if len(shape) >= 3 else 1
        table_rows = int(shape[-2])
        # base_miss (N_rows, n_grid) f64 per view (PE + sel).
        pending += 2 * table_rows * int(n_grid) * 8
        # ── Q ENSEMBLE device copies ──
        # ``catalogs/lss.py`` loads ``logq_members`` as HOST numpy; the transfer
        # happens in the factory (``jnp.asarray(full)`` then ``full_j[:, up]``),
        # which holds the FULL table and the compact slice live at once.  Only
        # count it while the table is still on the host — a caller that hands the
        # factory a device array has already paid for it in ``free_bytes``.
        # (An already-compact table — ``lss_completion_indexing == 1`` — is
        # transferred whole with no slice; counting a slice anyway over-reserves,
        # which is the safe direction.)
        if not _is_device_resident(members):
            pending += n_members * table_rows * int(n_grid) * 8      # full copy
            n_slices = 1 if is_bundle else 2   # bundles alias one compact slice
            pending += n_slices * n_members * max(1, n_unique) * int(n_grid) * 8
        # ── field_lss_q_members (M, n_occupied, n_grid) float32 ──
        # Built by ``prepare_catalog_views`` (factory time) for the flat path under
        # the FIELD convention — now the K=1 dark-siren DEFAULT.  For a bundle the
        # LOADER builds it, so it is already resident.  n_occupied <= n_pix, and
        # the member table's row axis IS n_pix for a global ensemble.
        if (not is_bundle and str(catalog_sky_weighting) == "field"
                and src.get("field_lss_q_members") is None):
            pending += n_members * table_rows * int(n_grid) * 4
    return int(pending)


def estimate_pending_static_bytes(data, *, n_grid: int, has_catalog: bool,
                                  catalog_memory=None,
                                  catalog_sky_weighting: str = "conditional") -> int:
    """Analytic estimate of the static state the FACTORY will allocate *after* the
    block-size resolver runs — i.e. the piece the device probe is blind to.

    The resolver runs AFTER the data load, so the loaded arrays are ALREADY
    device-resident and hence already reflected in the probed ``free_bytes`` (they
    dropped it).  What is NOT yet allocated is what the likelihood factory builds:

    * **KDE cache** — ``build_pixel_kde_cache`` produces a ``(n_unique, n_grid)``
      float64 table (8 B) per view; ``(unique_pe + unique_sel) * n_grid * 8``
      (conservatively counts both even when the two views share one table).
    * **base_miss** — the completion ensemble carries a ``(N_rows, n_grid)`` f64
      curve per view (PE + sel).
    * **compact union galaxy tables** — ``prepare_catalog_views`` gathers the
      full-sky rows to the PE-union-selection pixels at factory time.
    * **Q-ensemble device copies** — ``lss_completion_logq_members`` is loaded as
      HOST numpy, so both the full ``(M, n_pix, n_grid)`` transfer and the compact
      per-view slices are pending, not resident.
    * **field_lss_q_members** — the ``(M, n_occupied, n_grid)`` float32 rows the
      field-convention normaliser needs, also built at factory time on the flat path.

    The last three were previously ASSUMED already-resident ("a LOADED array"),
    which under-reserved a DESI-wide ``--lss_marginalize`` field run by ~34 GB and
    inflated both the placement budget and the last-resort floor guard by that much.

    A K >= 2 multitracer run keeps nothing at the top level: the per-catalog bundles
    in ``data['catalogs']`` are walked instead (each carries its own compact rows and
    its own Q ensemble), so the estimate is no longer silently 0 for every K >= 2 run.

    This is the quantity to subtract from the memory budget / reserve in the floor
    guard; subtracting the already-resident loaded arrays too would double-count
    them against ``free_bytes``.  Pure Python (no JAX import).
    """
    if not has_catalog or not isinstance(data, dict):
        return 0
    bundles = data.get("catalogs")
    if bundles:
        return int(sum(
            _pending_for_catalog_source(
                bundle, n_grid=n_grid, catalog_memory=None,
                catalog_sky_weighting=catalog_sky_weighting, is_bundle=True)
            for bundle in bundles if isinstance(bundle, dict)
        ))
    return _pending_for_catalog_source(
        data, n_grid=n_grid, catalog_memory=catalog_memory,
        catalog_sky_weighting=catalog_sky_weighting, is_bundle=False)


def measure_static_state_bytes(data, *, n_grid: int, has_catalog: bool,
                               n_catalogs: int = 1, catalog_memory=None,
                               drop_full_catalog: bool = False,
                               catalog_sky_weighting: str = "conditional") -> int:
    """Total static data-constant bytes for a loaded run (for the run-config report).

    Sums the exact ``nbytes`` of every loaded array — deduplicated by object
    identity (``data['zgals']`` and ``data['zgals_catalog']`` alias one buffer; a
    bundle's PE and selection views alias one galaxy table) — PLUS the analytic
    :func:`estimate_pending_static_bytes` for what the factory has yet to build.
    This is the human-facing "how big is the static state" number; the resolver
    subtracts only the *pending* portion (see that function's note on
    double-counting).  Note the loaded sum spans HOST and DEVICE arrays alike (the
    Q ensemble is loaded as host numpy), so it is a total-footprint figure, not a
    device-only one; the pending term separately accounts for the device copies the
    factory makes of the host arrays.  Pure Python: duck-types ``.nbytes``, no JAX
    import.
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
        data, n_grid=n_grid, has_catalog=has_catalog, catalog_memory=catalog_memory,
        catalog_sky_weighting=catalog_sky_weighting)
    return int(total)


def _slopes_and_fixed(*, has_catalog: bool, needs_grad: bool, n_grid: int,
                      n_q: int, max_gals_per_row, n_catalogs: int,
                      concurrent_evals: int, warn: bool = True):
    """``(sel_bpi, pe_bps, fixed_bytes)`` of the peak model for these dimensions.

    Factored out of :func:`resolve_block_sizes` so :func:`predicted_peak_bytes` and
    the resolver cannot drift apart.  See ``resolve_block_sizes``' docstring for the
    meaning of every scale factor.
    """
    grid_scale = max(1e-9, float(n_grid) / float(CAL_N_GRID))
    # The q-quadrature is evaluated PER SAMPLE by the default (exact) pairing
    # normaliser, so n_q multiplies both working sets.  With the opt-in
    # ``--pairing_m1_grid`` the quadrature moves off the sample axis and this scale
    # merely over-predicts (the safe direction).
    q_scale = max(1e-9, float(n_q) / float(CAL_N_Q))
    # A sampler that vmaps several proposals at once (tinyns replacement_chains)
    # holds that many copies of every intermediate live simultaneously.
    batch_scale = float(max(1, int(concurrent_evals)))
    if has_catalog:
        if max_gals_per_row is None or int(max_gals_per_row) <= 0:
            # UNKNOWN row width on a catalog run: the caller could not determine
            # the dimension (the K >= 2 stubbed top-level ``catalog_memory`` was
            # the historical case, and it silently produced max_gals_per_row = 1).
            # Scaling the heavier _CAT slopes BELOW the catalog-free ones would
            # claim a dark-siren likelihood is ~1000x lighter per injection than
            # the spectral config they were calibrated on, so fall back to the
            # calibration reference (ratio 1.0) and say so.  An explicit small
            # integer is TRUSTED — a genuinely sparse catalog really is lighter.
            if warn:
                _loud(
                    f"has_catalog=True but max_gals_per_row={max_gals_per_row!r}: "
                    "the caller could not supply the galaxies-per-unique-pixel "
                    "dimension. Falling back to the calibration reference "
                    f"({CAL_MAX_GALS_PER_ROW:,} galaxies/row) rather than scaling "
                    "the _CAT slopes below the catalog-free ones."
                )
            gals_ratio = 1.0
        else:
            gals_ratio = max(
                1e-9, float(max_gals_per_row) / float(CAL_MAX_GALS_PER_ROW))
        cat_scale = gals_ratio * max(1, int(n_catalogs))
        base_sel = (SEL_BYTES_PER_INJECTION_CAT if needs_grad
                    else SEL_BYTES_PER_INJECTION_VALUE_CAT)
        base_pe = (PE_BYTES_PER_SAMPLE_CAT if needs_grad
                   else PE_BYTES_PER_SAMPLE_VALUE_CAT)
    else:
        cat_scale = 1.0
        base_sel = (SEL_BYTES_PER_INJECTION if needs_grad
                    else SEL_BYTES_PER_INJECTION_VALUE)
        base_pe = (PE_BYTES_PER_SAMPLE if needs_grad
                   else PE_BYTES_PER_SAMPLE_VALUE)
    unit_scale = grid_scale * q_scale * cat_scale * batch_scale
    # Fixed term of the peak model.  On the GRADIENT path only ~1.7 GB of the 57.9
    # GiB is genuinely fixed runtime; the rest is the IRREDUCIBLE working-set floor
    # and scales with the same dimensions as the slopes (MEASURED: the single-pass
    # peak is 26.89 GB at n_q=64 and 78.77 GB at n_q=200).  At the calibration
    # dimensions unit_scale == 1.0, so this is TRUE_FIXED_BYTES exactly and every
    # historical plan is preserved bit-for-bit.  The gradient-free fixed term is a
    # genuinely small, genuinely flat measured intercept and is NOT scaled.
    if needs_grad:
        fixed_bytes = (GRAD_RUNTIME_FIXED_BYTES
                       + unit_scale * GRAD_WORKSET_FLOOR_BYTES)
    else:
        fixed_bytes = float(TRUE_FIXED_VALUE_BYTES)
    return base_sel * unit_scale, base_pe * unit_scale, fixed_bytes


def predicted_peak_bytes(*, n_events: int, n_samp: int, n_sel: int,
                         sel_batch_size=None, pe_event_block=None,
                         has_catalog: bool = False, needs_grad: bool = True,
                         n_grid: int = CAL_N_GRID, n_q: int = CAL_N_Q,
                         max_gals_per_row=CAL_MAX_GALS_PER_ROW,
                         n_catalogs: int = 1, concurrent_evals: int = 1,
                         static_state_bytes: float = 0.0,
                         flow_path: bool = False) -> float:
    """Peak device bytes the model predicts for a given plan.

    ``sel_batch_size``/``pe_event_block`` of ``None`` mean a single pass over that
    axis.  The two axes ADD on the gradient path (value + reverse-mode residuals of
    both are live) and take the MAX on the gradient-free one (measured: blocking one
    axis alone does not move the value peak).  Used by the resolver's calibration
    tests; the resolver itself compares a budget rather than a peak, but both go
    through :func:`_slopes_and_fixed` so they cannot drift.
    """
    sel_bpi, pe_bps, fixed_bytes = _slopes_and_fixed(
        has_catalog=has_catalog, needs_grad=needs_grad, n_grid=n_grid, n_q=n_q,
        max_gals_per_row=max_gals_per_row, n_catalogs=n_catalogs,
        concurrent_evals=concurrent_evals, warn=False)
    sel = float(n_sel if sel_batch_size is None else sel_batch_size) * sel_bpi
    pe = (0.0 if flow_path else
          float(n_events if pe_event_block is None else pe_event_block)
          * float(n_samp) * pe_bps)
    transient = sel + pe if needs_grad else max(sel, pe)
    return float(fixed_bytes) + max(0.0, float(static_state_bytes)) + transient


def resolve_block_sizes(*, n_events: int, n_samp: int, n_sel: int,
                        sel_requested, pe_requested, has_catalog: bool,
                        flow_path: bool, n_grid: int = CAL_N_GRID,
                        max_gals_per_row: int | None = CAL_MAX_GALS_PER_ROW,
                        n_catalogs: int = 1, n_q: int = CAL_N_Q,
                        static_state_bytes: float = 0.0,
                        needs_grad: bool = True, concurrent_evals: int = 1,
                        free_bytes: int | None = None,
                        free_bytes_reliable: bool = True,
                        backend: str | None = None,
                        safety_factor: float = SAFETY_FACTOR) -> BlockSizePlan:
    """Resolve ``(sel_batch_size, pe_event_block)`` from a memory budget.

    ``sel_requested`` / ``pe_requested`` are the parsed CLI values: an ``int``
    (explicit, passes through), ``None`` (an explicit single pass), or
    :data:`BLOCK_AUTO` (resolve here).  Explicit values always win per knob.

    The peak model (see :func:`predicted_peak_bytes`) is

    * ``needs_grad=True`` (NumPyro NUTS) — ``fixed + static + sel_batch*sel_bpi +
      pe_block*n_samp*pe_bps``: the value and the reverse-mode residuals of BOTH
      axes are live simultaneously, so the two transient working sets ADD.  ``fixed``
      is ``GRAD_RUNTIME_FIXED_BYTES + scale*GRAD_WORKSET_FLOOR_BYTES``, i.e. the
      irreducible part of the working set scales with the dimensions too; at the
      calibration dimensions it equals ``TRUE_FIXED_BYTES`` exactly, so every
      historical plan is preserved.
    * ``needs_grad=False`` (dynesty / tinyns) — ``TRUE_FIXED_VALUE_BYTES + static +
      max(sel_batch*sel_bpi, pe_block*n_samp*pe_bps)`` with the value-only slopes.
      MEASURED: blocking one axis alone leaves the value peak unchanged (5.648 →
      5.695 GB), because the unblocked axis's ``(N, n_q)`` intermediate still sets
      it; blocking BOTH drops it to 0.703 GB.  A max, not a sum.

    The slopes ``sel_bpi`` / ``pe_bps`` (and, on the gradient path, the working-set
    floor) scale with the dominant dimensions relative to the calibration config:

    * ``n_grid / CAL_N_GRID`` on both slopes (more grid nodes → more work/bytes);
    * ``n_q / CAL_N_Q`` on both slopes — the population q-quadrature is evaluated
      per SAMPLE by the default pairing model, so it multiplies both working sets
      (measured ~26 MB per q-node at N_sel = 1.07e6);
    * for catalog runs, ``(max_gals_per_row / CAL_MAX_GALS_PER_ROW) * n_catalogs``
      on the heavier ``_CAT`` slopes (more galaxies per row and a K-catalog
      mixture both multiply the redshift-prior work).  ``max_gals_per_row=None``
      means "the caller could not determine it": the ratio then falls back to 1.0
      with a loud warning instead of silently scaling the ``_CAT`` slopes below the
      catalog-free ones;
    * ``concurrent_evals`` on both slopes, for a sampler that ``vmap``s several
      likelihood evaluations at once (tinyns ``replacement_chains``).

    Auto policy: on a non-GPU backend keep a single pass; else size each axis
    against ``safety_factor * free_bytes - true_fixed - static_state_bytes`` — the
    selection axis first, and (in the additive value+grad model only) the PE axis
    just if a floored selection block still overflows.  A single pass that already
    fits resolves to ``(None, None)`` — bit-identical to the historical default.
    With ``needs_grad=True``, ``static_state_bytes == STATIC_STATE_CAL_BYTES``,
    ``n_grid == CAL_N_GRID``, ``n_q == CAL_N_Q`` and no catalog the budget equals
    the pre-decomposition ``safety_factor*free - FIXED_OVERHEAD_BYTES`` exactly
    (calibration preserved).
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

    sel_bpi, pe_bps, fixed_bytes = _slopes_and_fixed(
        has_catalog=has_catalog, needs_grad=needs_grad, n_grid=n_grid, n_q=n_q,
        max_gals_per_row=max_gals_per_row, n_catalogs=n_catalogs,
        concurrent_evals=concurrent_evals)

    # Placement budget: safety-discounted, minus that fixed term and the pending
    # static.  Drives even-split block sizing; when it goes negative (a catalog run
    # whose spectral-calibrated floor over-predicts) the even split falls back to the
    # floor — the historical behavior.
    budget = max(
        1.0,
        float(safety_factor) * float(free_bytes) - fixed_bytes - static_state_bytes,
    )
    # Floor-reduction guard: room for block working sets AFTER the pending static,
    # EXCLUDING the (spectral-over-estimated) fixed term.  Only a genuinely
    # device-dominating static state reduces the minimum block (see _floored_block);
    # otherwise the floor stands even when ``budget`` is negative.  An UNRELIABLE
    # free-memory reading (failed probe) gives no basis to reduce, so the guard is
    # disabled (guard_room = ∞) and the floor stands — matching the historical model.
    guard_room = (
        float(safety_factor) * float(free_bytes) - static_state_bytes
        if free_bytes_reliable else math.inf
    )

    # PE single-pass footprint (fixed unless we end up blocking PE below).
    pe_full_bytes = float(n_events) * float(n_samp) * pe_bps if not flow_path else 0.0
    sel_full_bytes = float(n_sel) * sel_bpi

    source = "auto-single-pass"

    if not needs_grad:
        # ── Gradient-free (value-only) peak: fixed + static + MAX(sel, pe) ──
        # Each axis is sized against the WHOLE budget, independently: blocking one
        # axis alone was measured to leave the value peak unchanged, so there is no
        # sense in charging the other axis's footprint against this one's budget
        # (which is what made 'auto' block the selection axis for a 0.0 GB saving
        # at a 1.79x throughput cost).
        if sel_auto:
            if sel_full_bytes <= budget:
                sel = None
            else:
                sel, reduced = _floored_block(
                    int(n_sel), budget, guard_room, sel_bpi,
                    floor=min(SEL_MIN_BATCH, int(n_sel)), name="selection",
                    static_state_bytes=static_state_bytes)
                source = "auto-floor-reduced" if reduced else "auto"
        if pe_auto and not flow_path:
            if pe_full_bytes <= budget:
                pe = None
            else:
                pe, reduced = _floored_block(
                    int(n_events), budget, guard_room, float(n_samp) * pe_bps,
                    floor=min(PE_MIN_BLOCK, int(n_events)), name="PE",
                    static_state_bytes=static_state_bytes, round_to=1)
                if reduced:
                    source = "auto-floor-reduced"
                elif source != "auto-floor-reduced":
                    source = "auto"
        return BlockSizePlan(sel, pe, source)

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
