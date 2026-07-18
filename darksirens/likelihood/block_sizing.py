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
  testable on CPU with injected budgets.  ``probe_device_memory_bytes`` (moved
  here from ``cli/analyze.py`` so both share one probe) is the only
  device-touching function and imports JAX lazily.
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
#     peak ≈ FIXED_OVERHEAD_BYTES
#            + sel_batch  * SEL_BYTES_PER_INJECTION[_CAT]
#            + pe_block   * n_samp * PE_BYTES_PER_SAMPLE[_CAT]
#
# where ``sel_batch`` defaults to the full ``N_sel`` and ``pe_block`` to the full
# ``N_events`` (a single pass).  The ``_CAT`` variants apply when a galaxy
# catalog is loaded (dark sirens), whose per-injection / per-sample redshift-prior
# state is heavier than the catalog-free spectral path.
#
# CONSTANTS_VERSION and the numbers below are PLACEHOLDERS until
# scripts/benchmark_block_sizes.py fits them on real data; each is re-stamped
# with the measured slope and the device it was measured on.  The policy logic
# and its unit tests do NOT depend on the exact values (tests inject budgets and
# assert structural properties), so placeholders keep the module importable and
# the wiring testable ahead of the calibration run.
CONSTANTS_VERSION = "uncalibrated-placeholder-2026-07-18"

SEL_BYTES_PER_INJECTION = 512          # spectral / catalog-free selection integral
SEL_BYTES_PER_INJECTION_CAT = 2048     # dark-siren selection integral (catalog state)
PE_BYTES_PER_SAMPLE = 512              # spectral per-event PE reduction, per sample
PE_BYTES_PER_SAMPLE_CAT = 2048         # dark-siren per-event PE reduction, per sample
FIXED_OVERHEAD_BYTES = 2 * 1024**3     # jit constants, population grids, params

# Floors: never chunk below these (below them the launch/padding overhead and
# recompile churn cost more than the memory they save).
SEL_MIN_BATCH = 32768
PE_MIN_BLOCK = 8

# Fraction of probed free memory the working set may occupy (headroom for
# fragmentation, transient copies, and — on a shared box — another process's
# growth).  Per-process peak is what we bound; see the contention note in
# docs/source/performance.md.
SAFETY_FACTOR = 0.7

# Even-split blocks are rounded up to a multiple of this (keeps XLA tile shapes
# friendly and minimises the padding of the final chunk for N_sel ~1e6).
BLOCK_ROUND_TO = 256

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
    source: str  # "explicit" | "auto" | "auto-single-pass" | "cpu" | "flow"


def probe_device_memory_bytes(default_gb=4.0):
    """Best-effort free-memory probe for the first JAX device.

    Returns ``(bytes, source)``.  Uses ``device.memory_stats()`` when available
    (GPU/TPU): ``bytes_limit - bytes_in_use`` if both present, else
    ``bytes_limit`` / ``bytes_reservable_limit``.  Falls back to ``default_gb``
    on the CPU backend or older jaxlib (where ``memory_stats`` returns ``None``
    or lacks these keys).

    Note: unless ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` is exported, JAX
    preallocates a large fraction of the GPU and ``bytes_limit`` reports that
    *pool* (still a safe ceiling for sizing — we stay within it).
    """
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


def resolve_block_sizes(*, n_events: int, n_samp: int, n_sel: int,
                        sel_requested, pe_requested, has_catalog: bool,
                        flow_path: bool, n_grid: int = 1000,
                        free_bytes: int | None = None, backend: str | None = None,
                        safety_factor: float = SAFETY_FACTOR) -> BlockSizePlan:
    """Resolve ``(sel_batch_size, pe_event_block)`` from a memory budget.

    ``sel_requested`` / ``pe_requested`` are the parsed CLI values: an ``int``
    (explicit, passes through), ``None`` (an explicit single pass), or
    :data:`BLOCK_AUTO` (resolve here).  Explicit values always win per knob.

    Auto policy: on a non-GPU backend keep a single pass; else block the
    selection axis first and the PE axis only if a floored selection block still
    overflows ``safety_factor * free_bytes - FIXED_OVERHEAD_BYTES``.  A single
    pass that already fits resolves to ``(None, None)`` — bit-identical to the
    historical default.
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
        free_bytes, _ = probe_device_memory_bytes()
    budget = max(1.0, float(safety_factor) * float(free_bytes) - FIXED_OVERHEAD_BYTES)

    sel_bpi = SEL_BYTES_PER_INJECTION_CAT if has_catalog else SEL_BYTES_PER_INJECTION
    pe_bps = PE_BYTES_PER_SAMPLE_CAT if has_catalog else PE_BYTES_PER_SAMPLE

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
            sel = _even_split_block(int(n_sel), max(1.0, sel_budget), sel_bpi,
                                    floor=min(SEL_MIN_BATCH, int(n_sel)))
            source = "auto"

    # ── PE axis (only if a floored selection block still overflows) ──
    if pe_auto and not flow_path:
        sel_bytes_now = (sel_full_bytes if sel is None
                         else float(sel) * sel_bpi)
        if sel_bytes_now + pe_full_bytes <= budget:
            pe = None                        # PE single pass fits alongside sel
        else:
            pe_budget = budget - sel_bytes_now
            pe = _even_split_block(int(n_events), max(1.0, pe_budget),
                                   float(n_samp) * pe_bps,
                                   floor=min(PE_MIN_BLOCK, int(n_events)),
                                   round_to=1)
            source = "auto"

    return BlockSizePlan(sel, pe, source)
