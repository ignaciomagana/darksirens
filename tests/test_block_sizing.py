"""Block-size auto-sizer (PR 3): argparse type, pure resolution policy, and the
downstream contracts (parser defaults, factory string-guard, analyze re-export).

The policy tests inject ``free_bytes``/``backend`` so they run on CPU with no
JAX device and assert *structural* properties (fits/floors/monotonicity/even
split), independent of the calibrated byte-per-unit constants.
"""
import math
from argparse import ArgumentTypeError

import pytest

from darksirens.likelihood.block_sizing import (
    BLOCK_AUTO,
    BlockSizePlan,
    SEL_MIN_BATCH,
    PE_MIN_BLOCK,
    BLOCK_ROUND_TO,
    block_size_arg,
    resolve_block_sizes,
)

# Real-data shapes (gwsamples_bbh_whitelist_all_events + selection_o3o4ab_allsky).
N_EVENTS = 259
N_SAMP = 4096
N_SEL = 1_067_946

GB = 1024**3


def _auto(free_bytes, *, has_catalog=False, flow_path=False, backend="gpu",
          sel=BLOCK_AUTO, pe=BLOCK_AUTO):
    return resolve_block_sizes(
        n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL,
        sel_requested=sel, pe_requested=pe,
        has_catalog=has_catalog, flow_path=flow_path,
        free_bytes=free_bytes, backend=backend,
    )


# ── block_size_arg ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (None, BLOCK_AUTO),
    ("auto", BLOCK_AUTO),
    ("AUTO", BLOCK_AUTO),
    ("off", None),
    ("none", None),
    ("0", None),
    ("65536", 65536),
    (" 131072 ", 131072),
])
def test_block_size_arg_valid(value, expected):
    assert block_size_arg(value) == expected


@pytest.mark.parametrize("value", ["-1", "banana", "3.5", "1e5"])
def test_block_size_arg_invalid(value):
    with pytest.raises(ArgumentTypeError):
        block_size_arg(value)


# ── explicit passthrough ─────────────────────────────────────────────────────────

def test_explicit_int_passes_through():
    plan = _auto(1 * GB, sel=65536, pe=32)
    assert plan == BlockSizePlan(65536, 32, "explicit")


def test_explicit_off_stays_none():
    # Explicit None (from --sel_batch_size off) is an explicit single pass.
    plan = _auto(1 * GB, sel=None, pe=None)
    assert plan == BlockSizePlan(None, None, "explicit")


def test_mixed_explicit_sel_auto_pe():
    # Budget well above the calibrated single-pass footprint (~80 GB) so pe fits.
    plan = _auto(400 * GB, sel=65536, pe=BLOCK_AUTO)
    assert plan.sel_batch_size == 65536      # explicit sel honoured
    assert plan.pe_event_block is None       # pe auto-resolves to single pass here
    assert plan.source in ("auto", "auto-single-pass")


# ── backend fallback ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend", ["cpu", "CPU", "tpu", "metal"])
def test_non_gpu_backend_single_pass(backend):
    plan = _auto(1 * GB, backend=backend)
    assert plan == BlockSizePlan(None, None, "cpu")


# ── budget extremes ──────────────────────────────────────────────────────────────

def test_huge_budget_single_pass():
    # 400 GB is comfortably above the calibrated single-pass working set (~80 GB on
    # the real spectral problem), so auto keeps the historical single pass.
    plan = _auto(400 * GB)
    assert plan.sel_batch_size is None
    assert plan.pe_event_block is None
    assert plan.source == "auto-single-pass"


def test_small_budget_blocks_and_respects_floor():
    plan = _auto(3 * GB)  # forces the selection axis to chunk
    assert plan.sel_batch_size is not None
    assert plan.sel_batch_size >= SEL_MIN_BATCH
    assert plan.sel_batch_size <= N_SEL
    assert plan.source == "auto"


def test_tiny_budget_hits_pe_floor_not_below():
    plan = _auto(1 * GB, has_catalog=True)
    if plan.pe_event_block is not None:
        assert plan.pe_event_block >= min(PE_MIN_BLOCK, N_EVENTS)
    if plan.sel_batch_size is not None:
        assert plan.sel_batch_size >= min(SEL_MIN_BATCH, N_SEL)


# ── structural properties ────────────────────────────────────────────────────────

def test_even_split_rounds_to_block_multiple():
    plan = _auto(3 * GB)
    b = plan.sel_batch_size
    assert b is not None
    # even split → ceil(N_SEL / b) chunks with a small final remainder, and the
    # block is a multiple of BLOCK_ROUND_TO (unless clamped to N_SEL).
    if b < N_SEL:
        assert b % BLOCK_ROUND_TO == 0
        k = math.ceil(N_SEL / b)
        # the split is genuinely even: the last chunk is never more than one
        # full block short (no near-empty trailing chunk).
        assert (k - 1) * b < N_SEL <= k * b


def test_monotonic_in_free_bytes():
    # More memory never produces a smaller selection block (None == single pass
    # is treated as the largest, i.e. == N_SEL).
    def _sel(free):
        s = _auto(free).sel_batch_size
        return N_SEL if s is None else s
    # Span the floor→single-pass transition: the low budgets floor the selection
    # block; the top ones (above ~80 GB / 0.7) restore the single pass (== N_SEL).
    budgets = [1 * GB, 4 * GB, 20 * GB, 80 * GB, 200 * GB, 400 * GB]
    sels = [_sel(b) for b in budgets]
    assert sels == sorted(sels)


def test_catalog_block_le_no_catalog():
    # Catalog runs have a heavier per-injection footprint → smaller (or equal)
    # selection block at the same budget.  Use a budget in the blocking regime
    # (below the ~80 GB single-pass footprint) so the difference is exercised.
    free = 105 * GB
    def _sel(has_cat):
        s = _auto(free, has_catalog=has_cat).sel_batch_size
        return N_SEL if s is None else s
    assert _sel(True) <= _sel(False)


# ── flow surrogate ───────────────────────────────────────────────────────────────

def test_flow_forces_pe_none():
    # The flow-surrogate path has no per-event PE reduction to chunk.
    plan = _auto(1 * GB, flow_path=True, pe=BLOCK_AUTO)
    assert plan.pe_event_block is None
    plan2 = _auto(1 * GB, flow_path=True, pe=64)  # even an explicit pe is dropped
    assert plan2.pe_event_block is None


# ── provenance ───────────────────────────────────────────────────────────────────

def test_source_tags():
    assert _auto(1 * GB, sel=100, pe=10).source == "explicit"
    assert _auto(1 * GB, backend="cpu").source == "cpu"
    assert _auto(400 * GB).source == "auto-single-pass"
    assert _auto(3 * GB).source == "auto"


# ── downstream contracts: parser defaults, factory guard, analyze re-export ──────

def test_factory_rejects_unresolved_sentinel():
    from darksirens.likelihood.block_sizing import require_resolved_block_size
    with pytest.raises(TypeError):
        require_resolved_block_size("sel_batch_size", BLOCK_AUTO)
    # concrete values pass through unchanged
    assert require_resolved_block_size("sel_batch_size", 65536) == 65536
    assert require_resolved_block_size("pe_event_block", None) is None


def test_parser_defaults_auto_both_clis():
    pytest.importorskip("jax")
    from darksirens.cli.inference import build_parser as inf
    from darksirens.cli.inference_lensing import build_parser as lens
    o = inf().parse_args(["--sampler", "tinyns"])
    assert o.sel_batch_size == BLOCK_AUTO
    assert o.pe_event_block == BLOCK_AUTO
    lo = lens().parse_args(
        ["--gw_path", "g.h5", "--gwselection_path", "s.h5", "--sampler", "tinyns"])
    assert lo.sel_batch_size == BLOCK_AUTO
    # 'off' and an integer parse through the shared block_size_arg on both CLIs.
    assert inf().parse_args(["--sampler", "tinyns", "--sel_batch_size", "off"]).sel_batch_size is None
    assert lens().parse_args(
        ["--gw_path", "g.h5", "--gwselection_path", "s.h5", "--sampler", "tinyns",
         "--sel_batch_size", "65536"]).sel_batch_size == 65536


def test_analyze_reexports_probe():
    pytest.importorskip("jax")
    from darksirens.cli.analyze import probe_device_memory_bytes as a
    from darksirens.likelihood.block_sizing import probe_device_memory_bytes as b
    assert a is b
