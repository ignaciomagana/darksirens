"""Block-size auto-sizer (PR 3): argparse type, pure resolution policy, and the
downstream contracts (parser defaults, factory string-guard, analyze re-export).

The policy tests inject ``free_bytes``/``backend`` so they run on CPU with no
JAX device and assert *structural* properties (fits/floors/monotonicity/even
split), independent of the calibrated byte-per-unit constants.
"""
import math
from argparse import ArgumentTypeError

import pytest

import numpy as np

from darksirens.likelihood.block_sizing import (
    BLOCK_AUTO,
    BlockSizePlan,
    SEL_MIN_BATCH,
    PE_MIN_BLOCK,
    BLOCK_ROUND_TO,
    CAL_N_GRID,
    CAL_N_SEL,
    CAL_N_EVENTS,
    CAL_N_SAMP,
    CAL_MAX_GALS_PER_ROW,
    CAL_N_Q,
    FIXED_OVERHEAD_BYTES,
    TRUE_FIXED_BYTES,
    TRUE_FIXED_VALUE_BYTES,
    STATIC_STATE_CAL_BYTES,
    SEL_BYTES_PER_INJECTION_VALUE,
    PE_BYTES_PER_SAMPLE_VALUE,
    SEL_BYTES_PER_INJECTION_CAT,
    PE_BYTES_PER_SAMPLE_CAT,
    SEL_BYTES_PER_INJECTION_VALUE_CAT,
    PE_BYTES_PER_SAMPLE_VALUE_CAT,
    GRAD_RUNTIME_FIXED_BYTES,
    GRAD_WORKSET_FLOOR_BYTES,
    SAFETY_FACTOR,
    block_size_arg,
    resolve_block_sizes,
    predicted_peak_bytes,
    measure_static_state_bytes,
    estimate_pending_static_bytes,
    sampler_block_sizing_profile,
)

# Real-data shapes (gwsamples_bbh_whitelist_all_events + selection_o3o4ab_allsky).
N_EVENTS = 259
N_SAMP = 4096
N_SEL = 1_067_946

GB = 1024**3


def _auto(free_bytes, *, has_catalog=False, flow_path=False, backend="gpu",
          sel=BLOCK_AUTO, pe=BLOCK_AUTO, static_state_bytes=0):
    return resolve_block_sizes(
        n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL,
        sel_requested=sel, pe_requested=pe,
        has_catalog=has_catalog, flow_path=flow_path,
        static_state_bytes=static_state_bytes,
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


def test_moderate_budget_blocks_and_respects_floor():
    # A budget just below the ~80 GB single-pass working set (0.7*free minus the
    # decomposed fixed overhead) forces the selection axis to chunk while leaving
    # room to respect the SEL_MIN_BATCH floor.
    plan = _auto(100 * GB)
    assert plan.sel_batch_size is not None
    assert plan.sel_batch_size >= SEL_MIN_BATCH
    assert plan.sel_batch_size <= N_SEL
    assert plan.source == "auto"


def test_low_budget_keeps_floor_when_static_is_small():
    # A small *free* budget alone does NOT reduce the floor: the transient
    # TRUE_FIXED over-predicts (especially for catalog runs) and must not gate the
    # minimum block.  With negligible static state the floor stands (historical
    # behavior — the run fits in the safety margin).
    plan = _auto(1 * GB, has_catalog=True, static_state_bytes=0)
    assert plan.sel_batch_size == min(SEL_MIN_BATCH, N_SEL)
    assert plan.source == "auto"


def test_static_dominance_reduces_floor_below_min_batch():
    # Only a device-DOMINATING resident static state drops the floor: here static
    # leaves no room for even a SEL_MIN_BATCH block's working set.
    plan = _auto(80 * GB, has_catalog=True, static_state_bytes=78 * GB)
    assert plan.source == "auto-floor-reduced"
    assert plan.sel_batch_size is not None and plan.sel_batch_size >= 1
    assert plan.sel_batch_size < min(SEL_MIN_BATCH, N_SEL)
    if plan.pe_event_block is not None:
        assert plan.pe_event_block >= 1


def test_unreliable_probe_never_reduces_floor():
    # A failed memory probe (no reliable free reading) must NOT reduce the floor —
    # the floor stands (historical behavior), even under a dominating static state.
    plan = resolve_block_sizes(
        n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL,
        sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
        has_catalog=True, flow_path=False,
        n_grid=1000, static_state_bytes=78 * GB,
        free_bytes=80 * GB, free_bytes_reliable=False, backend="gpu",
    )
    assert plan.source == "auto"                  # blocked to the floor, not reduced
    assert plan.sel_batch_size == min(SEL_MIN_BATCH, N_SEL)


# ── structural properties ────────────────────────────────────────────────────────

def test_even_split_rounds_to_block_multiple():
    plan = _auto(100 * GB)
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
    # Span reduced-floor → blocking → single-pass: the low budgets can't fit even
    # a floored block (fixed overhead alone exceeds them → smallest blocks); the
    # mid ones chunk the selection axis; the top ones restore the single pass
    # (== N_SEL).  With the ~58 GiB decomposed fixed overhead the interesting
    # transition sits around 95-110 GB.
    budgets = [80 * GB, 95 * GB, 100 * GB, 106 * GB, 108 * GB, 400 * GB]
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
    assert _auto(100 * GB).source == "auto"
    # Floor reduction is driven by static-state dominance, not a small free budget.
    assert _auto(80 * GB, static_state_bytes=78 * GB).source == "auto-floor-reduced"


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


# ── FIXED-overhead decomposition: calibration-point preservation (no regression) ──

def _legacy_blocks(free_bytes):
    """Faithful re-implementation of the pre-decomposition (a8ba5e7) resolver:
    ``budget = 0.7*free - FIXED_OVERHEAD_BYTES``, fixed spectral slopes, floors
    NEVER reduced.  The no-regression reference for the spectral calibration config.
    """
    from darksirens.likelihood.block_sizing import _even_split_block
    budget = max(1.0, SAFETY_FACTOR * free_bytes - FIXED_OVERHEAD_BYTES)
    sel_bpi, pe_bps = 8000, 9000
    pe_full = N_EVENTS * N_SAMP * pe_bps
    sel_full = N_SEL * sel_bpi
    sel_budget = budget - pe_full
    if sel_full <= sel_budget:
        sel = None
    else:
        sel = _even_split_block(N_SEL, max(1.0, sel_budget), sel_bpi,
                                floor=min(SEL_MIN_BATCH, N_SEL))
    sel_bytes = sel_full if sel is None else sel * sel_bpi
    if sel_bytes + pe_full <= budget:
        pe = None
    else:
        pe = _even_split_block(N_EVENTS, max(1.0, budget - sel_bytes),
                               N_SAMP * pe_bps, floor=min(PE_MIN_BLOCK, N_EVENTS),
                               round_to=1)
    return sel, pe


def test_decomposition_arithmetic_reconstructs_fixed_overhead():
    # TRUE_FIXED + (this config's static state) must equal the a8ba5e7 anchor.
    assert TRUE_FIXED_BYTES + STATIC_STATE_CAL_BYTES == FIXED_OVERHEAD_BYTES
    # And STATIC_STATE_CAL is exactly the spectral config's 5 PE + 5 sel f64 fields.
    assert STATIC_STATE_CAL_BYTES == (
        5 * 8 * CAL_N_EVENTS * CAL_N_SAMP + 5 * 8 * CAL_N_SEL
    )


# The spectral calibration config (CAL_N_GRID, static == the config's measured
# static state) must reproduce the pre-change blocks BIT-FOR-BIT across the whole
# feasible range.  The floor-reduction guard keys on measured static state, not on
# the (transient, over-estimated) TRUE_FIXED, so it never fires for the spectral
# config — the new blocks equal the a8ba5e7 blocks at every budget below.
@pytest.mark.parametrize("free_gb", [80, 90, 95, 100, 106, 108, 120, 150, 200, 400])
def test_calibration_point_preserved(free_gb):
    free = free_gb * GB
    plan = resolve_block_sizes(
        n_events=CAL_N_EVENTS, n_samp=CAL_N_SAMP, n_sel=CAL_N_SEL,
        sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
        has_catalog=False, flow_path=False,
        n_grid=CAL_N_GRID, static_state_bytes=STATIC_STATE_CAL_BYTES,
        free_bytes=free, backend="gpu",
    )
    assert (plan.sel_batch_size, plan.pe_event_block) == _legacy_blocks(free)


# ── dominant per-block dimensions: monotonicity ──────────────────────────────────

def _sel_of(**kw):
    base = dict(
        n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL,
        sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
        flow_path=False, backend="gpu",
    )
    base.update(kw)
    s = resolve_block_sizes(**base).sel_batch_size
    return N_SEL if s is None else s


def test_monotonic_in_n_grid():
    # More grid nodes → heavier per-unit work → never a larger selection block.
    sels = [_sel_of(has_catalog=False, free_bytes=110 * GB, n_grid=g)
            for g in (1000, 1500, 2000, 3000)]
    assert sels == sorted(sels, reverse=True)


def test_monotonic_in_max_gals():
    sels = [_sel_of(has_catalog=True, free_bytes=110 * GB, max_gals_per_row=m)
            for m in (300, 1000, CAL_MAX_GALS_PER_ROW, 4000)]
    assert sels == sorted(sels, reverse=True)


def test_monotonic_in_catalog_count_K():
    sels = [_sel_of(has_catalog=True, free_bytes=110 * GB, n_catalogs=k)
            for k in (1, 2, 4, 8)]
    assert sels == sorted(sels, reverse=True)


def test_monotonic_in_static_state():
    sels = [_sel_of(has_catalog=False, free_bytes=110 * GB, static_state_bytes=s)
            for s in (0, 5 * GB, 15 * GB, 30 * GB)]
    assert sels == sorted(sels, reverse=True)


def test_static_state_dominance_yields_smallest_blocks(capsys):
    # Static state approaching device memory leaves no residual budget: the model
    # drops to the smallest feasible block and tags/warns loudly.
    plan = resolve_block_sizes(
        n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL,
        sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
        has_catalog=True, flow_path=False,
        n_grid=1000, static_state_bytes=78 * GB,
        free_bytes=80 * GB, backend="gpu",
    )
    assert plan.source == "auto-floor-reduced"
    assert plan.sel_batch_size == 1               # smallest feasible block
    out = capsys.readouterr().out
    assert "block-sizing" in out and "may still OOM" in out


# ── measure_static_state_bytes ───────────────────────────────────────────────────

def test_measure_static_state_no_catalog_is_arrays_only():
    data = {"m1detsels": np.zeros(100, np.float64)}  # 100 * 8 = 800 B
    assert measure_static_state_bytes(data, n_grid=1000, has_catalog=False) == 800


def test_measure_static_state_dedup_and_kde_estimate():
    z = np.zeros((4, 30), np.float64)                 # 4*30*8 = 960 B
    data = {
        "zgals": z, "zgals_catalog": z,               # ALIAS: counted once
        "wgals_sel": np.zeros((4, 30), np.float32),   # 4*30*4 = 480 B
        "catalog_memory": {"unique_pe_pixels": 4, "unique_sel_pixels": 4},
        "meta_string": "not an array", "a_scalar": 3,  # ignored (no .nbytes)
    }
    total = measure_static_state_bytes(
        data, n_grid=30, has_catalog=True, catalog_memory=data["catalog_memory"])
    # arrays 960 + 480, plus KDE (4+4)*30*8 = 1920
    assert total == (960 + 480) + 1920


def test_estimate_pending_static_is_factory_only():
    # Pending = factory KDE (+ base_miss); it must NOT include the loaded arrays
    # (already device-resident, already in free_bytes).
    data = {
        "zgals_pe": np.zeros((7, 40), np.float64),    # loaded, resident -> excluded
        "catalog_memory": {"unique_pe_pixels": 3, "unique_sel_pixels": 5},
    }
    pending = estimate_pending_static_bytes(
        data, n_grid=40, has_catalog=True, catalog_memory=data["catalog_memory"])
    assert pending == (3 + 5) * 40 * 8            # KDE only; no loaded-array bytes
    # No catalog -> nothing pending.
    assert estimate_pending_static_bytes({}, n_grid=1000, has_catalog=False) == 0


def test_measure_static_state_base_miss_from_members():
    members = np.zeros((3, 5, 30), np.float64)        # M=3, N_rows=5, n_grid=30
    data = {
        "lss_completion_logq_members": members,       # 3*5*30*8 = 3600 B (counted)
        "catalog_memory": {"unique_pe_pixels": 2, "unique_sel_pixels": 2},
    }
    total = measure_static_state_bytes(
        data, n_grid=30, has_catalog=True, catalog_memory=data["catalog_memory"])
    # loaded members 3600
    #  + KDE (2+2)*30*8 = 960
    #  + base_miss 2*(5*30*8) = 2400
    #  + the ensemble's DEVICE copies, which the factory (not the loader) makes
    #    because catalogs/lss.py loads logq_members as HOST numpy: the full
    #    (M, N_rows, n_grid) transfer 3*5*30*8 = 3600, plus one compact
    #    (M, n_union, n_grid) slice per view 2*3*4*30*8 = 5760.
    assert total == 3600 + 960 + 2400 + 3600 + 5760


# ── P-3: pending static counts what the FACTORY (not the loader) allocates ───────

def test_pending_static_counts_union_galaxy_tables():
    """``prepare_catalog_views`` rebuilds the union galaxy tables from the FULL-sky
    rows at factory time, so a run that still carries them owes those bytes."""
    cm = {"unique_pe_pixels": 3, "unique_sel_pixels": 5,
          "max_galaxies_per_unique_pixel": 40}
    base = {"zgals_pe": np.zeros((3, 40), np.float64), "catalog_memory": cm}
    without_full = estimate_pending_static_bytes(
        base, n_grid=20, has_catalog=True, catalog_memory=cm)
    with_full = estimate_pending_static_bytes(
        {**base, "zgals": np.zeros((100, 40), np.float64)},
        n_grid=20, has_catalog=True, catalog_memory=cm)
    kde = (3 + 5) * 20 * 8
    assert without_full == kde                     # nothing to rebuild
    # union rows (<= unique_pe + unique_sel = 8) x 40 galaxies x 3 f64 tables,
    # plus the int32 ngals count vector.
    assert with_full == kde + 8 * 40 * 3 * 8 + 8 * 4


def test_pending_static_skips_union_tables_for_bundles():
    """A K >= 2 bundle carries no full-sky rows (the loader compacted them), so the
    factory rebuilds nothing for it."""
    bundle = {
        "zgals_pe": np.zeros((6, 30), np.float64),
        "zgals_sel": np.zeros((6, 30), np.float64),
        "zgals": np.zeros((100, 30), np.float64),  # must be IGNORED for a bundle
    }
    pending = estimate_pending_static_bytes(
        {"catalogs": [bundle]}, n_grid=20, has_catalog=True)
    assert pending == (6 + 6) * 20 * 8             # KDE caches only


def test_pending_static_skips_device_resident_member_table():
    """A member table already on the device is reflected in ``free_bytes``; only a
    HOST table costs the factory a transfer + a compact slice."""
    class _FakeDeviceArray:
        def __init__(self, shape):
            self.shape = shape
            self.nbytes = int(np.prod(shape)) * 8

        def devices(self):                          # the duck-typed device marker
            return ("gpu:0",)

    cm = {"unique_pe_pixels": 2, "unique_sel_pixels": 2}
    host = estimate_pending_static_bytes(
        {"lss_completion_logq_members": np.zeros((3, 5, 30), np.float64),
         "catalog_memory": cm},
        n_grid=30, has_catalog=True, catalog_memory=cm)
    device = estimate_pending_static_bytes(
        {"lss_completion_logq_members": _FakeDeviceArray((3, 5, 30)),
         "catalog_memory": cm},
        n_grid=30, has_catalog=True, catalog_memory=cm)
    kde_and_base_miss = (2 + 2) * 30 * 8 + 2 * 5 * 30 * 8
    assert device == kde_and_base_miss
    assert host == kde_and_base_miss + 3 * 5 * 30 * 8 + 2 * 3 * 4 * 30 * 8
    assert host > device


def test_pending_static_drops_the_full_member_copy_when_pixels_are_compacted():
    """The factory gathers a HOST ensemble in numpy BEFORE the transfer
    (``factory._gather_pixel_rows``), so only the compact ``(M, n_union, n_grid)``
    slices reach the device -- the full ``(M, n_pix, n_grid)`` copy is no longer
    pending.  A source with no union-pixel map has nothing to gather to and is
    still transferred whole, so it keeps owing the full term."""
    cm = {"unique_pe_pixels": 2, "unique_sel_pixels": 2}
    members = np.zeros((3, 5, 30), np.float64)
    uncompacted = estimate_pending_static_bytes(
        {"lss_completion_logq_members": members, "catalog_memory": cm},
        n_grid=30, has_catalog=True, catalog_memory=cm)
    compacted = estimate_pending_static_bytes(
        {"lss_completion_logq_members": members, "catalog_memory": cm,
         "unique_pixels_pe": np.arange(2, dtype=np.int32),
         "unique_pixels_sel": np.arange(2, dtype=np.int32)},
        n_grid=30, has_catalog=True, catalog_memory=cm)
    full_copy = 3 * 5 * 30 * 8
    assert uncompacted - compacted == full_copy
    # What remains is the KDE caches, base_miss and the two compact slices.
    assert compacted == (2 + 2) * 30 * 8 + 2 * 5 * 30 * 8 + 2 * 3 * 4 * 30 * 8


def test_pending_static_counts_field_member_rows():
    """The field convention adds an (M, n_occupied, n_grid) float32 table, also
    built at factory time (K=1 flat path)."""
    cm = {"unique_pe_pixels": 2, "unique_sel_pixels": 2}
    data = {"lss_completion_logq_members": np.zeros((3, 5, 30), np.float64),
            "catalog_memory": cm}
    conditional = estimate_pending_static_bytes(
        data, n_grid=30, has_catalog=True, catalog_memory=cm,
        catalog_sky_weighting="conditional")
    field = estimate_pending_static_bytes(
        data, n_grid=30, has_catalog=True, catalog_memory=cm,
        catalog_sky_weighting="field")
    assert field - conditional == 3 * 5 * 30 * 4
    # Already built by the loader (bundle path) -> not pending again.
    assert estimate_pending_static_bytes(
        {**data, "field_lss_q_members": np.zeros((3, 5, 30), np.float32)},
        n_grid=30, has_catalog=True, catalog_memory=cm,
        catalog_sky_weighting="field") == conditional


def test_pending_static_walks_k2_bundles():
    """K >= 2 keeps its catalogs per-bundle; the top level is empty by design, so a
    bundle-blind estimate was silently 0 for every multitracer run."""
    bundles = [
        {"zgals_pe": np.zeros((4, 20), np.float64),
         "zgals_sel": np.zeros((4, 20), np.float64)},
        {"zgals_pe": np.zeros((7, 20), np.float64),
         "zgals_sel": np.zeros((7, 20), np.float64)},
    ]
    data = {"catalog_memory": None, "zgals_pe": None, "catalogs": bundles}
    pending = estimate_pending_static_bytes(data, n_grid=50, has_catalog=True)
    assert pending == ((4 + 4) + (7 + 7)) * 50 * 8
    assert pending > 0


class _ShapeOnly:
    """Array stand-in carrying only ``.shape``/``.nbytes`` (no allocation), so a
    DESI-scale scenario can be exercised in a unit test."""

    def __init__(self, shape, itemsize=8, device=False):
        self.shape = tuple(shape)
        self.nbytes = int(np.prod(shape)) * itemsize
        if device:
            self.devices = lambda: ("gpu:0",)


def test_desi_scale_field_run_is_blocked_down_not_single_passed():
    """P-3 end to end: a DESI-wide K=1 ``--lss_marginalize`` field run on a 141 GB
    card.  With the old KDE+base_miss-only reserve (1.18 GB) the resolver believed a
    full single pass fitted, while the factory actually allocates ~48 GB before the
    first evaluation; with the corrected reserve the same call is blocked down."""
    n_rows, n_grid, n_members, max_gals = 49152, 1000, 32, 2113
    cm = {"unique_pe_pixels": n_rows // 2, "unique_sel_pixels": n_rows // 2,
          "max_galaxies_per_unique_pixel": max_gals}
    data = {
        "zgals": _ShapeOnly((786432, max_gals), device=True),   # full-sky, retained
        "zgals_pe": _ShapeOnly((n_rows, max_gals), device=True),
        "zgals_sel": _ShapeOnly((n_rows, max_gals), device=True),
        "lss_completion_logq_members": _ShapeOnly((n_members, n_rows, n_grid)),
        "catalog_memory": cm,
    }
    pending = estimate_pending_static_bytes(
        data, n_grid=n_grid, has_catalog=True, catalog_memory=cm,
        catalog_sky_weighting="field")
    legacy = 3 * n_rows * n_grid * 8               # KDE + base_miss only
    assert legacy < 2e9 < 40e9 < pending < 60e9    # ~1.2 GB -> ~47.7 GB

    def _plan(static):
        return resolve_block_sizes(
            n_events=259, n_samp=4096, n_sel=1_067_946,
            sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
            has_catalog=True, flow_path=False, n_grid=n_grid,
            max_gals_per_row=max_gals, n_catalogs=1,
            static_state_bytes=static, needs_grad=True,
            free_bytes=141 * GB, backend="gpu")

    assert _plan(legacy) == BlockSizePlan(None, None, "auto-single-pass")
    blocked = _plan(pending)
    assert blocked.sel_batch_size == SEL_MIN_BATCH
    assert blocked.pe_event_block == PE_MIN_BLOCK


def test_desi_scale_field_run_still_single_passes_for_dynesty():
    """... and the two fixes compose: the same corrected ~48 GB reserve still leaves
    a gradient-free run its single pass, because the value peak is ~10 GB not ~80."""
    n_rows, n_grid, n_members, max_gals = 49152, 1000, 32, 2113
    cm = {"unique_pe_pixels": n_rows // 2, "unique_sel_pixels": n_rows // 2,
          "max_galaxies_per_unique_pixel": max_gals}
    pending = estimate_pending_static_bytes(
        {"zgals": _ShapeOnly((786432, max_gals), device=True),
         "zgals_pe": _ShapeOnly((n_rows, max_gals), device=True),
         "zgals_sel": _ShapeOnly((n_rows, max_gals), device=True),
         "lss_completion_logq_members": _ShapeOnly((n_members, n_rows, n_grid)),
         "catalog_memory": cm},
        n_grid=n_grid, has_catalog=True, catalog_memory=cm,
        catalog_sky_weighting="field")
    plan = resolve_block_sizes(
        n_events=259, n_samp=4096, n_sel=1_067_946,
        sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
        has_catalog=True, flow_path=False, n_grid=n_grid,
        max_gals_per_row=max_gals, n_catalogs=1,
        static_state_bytes=pending, needs_grad=False,
        free_bytes=141 * GB, backend="gpu")
    assert plan == BlockSizePlan(None, None, "auto-single-pass")


# ── P-1/P-3 regression: the catalog slope must never fall below catalog-free ─────

def test_unknown_max_gals_falls_back_to_calibration_reference(capsys):
    """``max_gals_per_row=None`` (caller could not determine it) must NOT scale the
    _CAT slopes below the catalog-free ones — the K >= 2 blindness that made a
    2-catalog dark-siren run look ~1000x lighter per injection than spectral."""
    free = 110 * GB
    unknown = _sel_of(has_catalog=True, free_bytes=free, max_gals_per_row=None,
                      n_catalogs=2)
    out = capsys.readouterr().out
    assert "block-sizing" in out and "max_gals_per_row=None" in out
    reference = _sel_of(has_catalog=True, free_bytes=free,
                        max_gals_per_row=CAL_MAX_GALS_PER_ROW, n_catalogs=2)
    no_catalog = _sel_of(has_catalog=False, free_bytes=free)
    assert unknown == reference             # falls back to the calibration point
    assert unknown <= no_catalog            # never LIGHTER than the spectral path


def test_k2_degenerate_inputs_would_single_pass_where_true_dims_block():
    """P-1 end to end: the K=2 DESI+DES plan.  Fed the degenerate inputs the stubbed
    top-level ``catalog_memory`` used to produce (max_gals/row = 1, pending static =
    0) the resolver claims a full single pass fits; fed the real per-bundle
    dimensions the identical call blocks.  The CLI-side fix is that
    ``_block_sizing_inputs`` can no longer produce the degenerate pair (see
    ``test_cli_block_sizing_inputs_k2_bundles``)."""
    base = dict(n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL,
                sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
                has_catalog=True, flow_path=False, n_grid=1000, n_catalogs=2,
                needs_grad=True, free_bytes=141 * GB, backend="gpu")
    degenerate = resolve_block_sizes(max_gals_per_row=1, static_state_bytes=0, **base)
    assert degenerate == BlockSizePlan(None, None, "auto-single-pass")
    real = resolve_block_sizes(max_gals_per_row=CAL_MAX_GALS_PER_ROW,
                               static_state_bytes=786_000_000, **base)
    assert real.sel_batch_size is not None and real.sel_batch_size < N_SEL
    assert real.source == "auto"


def test_explicit_small_max_gals_is_trusted():
    # A genuinely sparse catalog really is lighter: an explicit int is honoured
    # (and no warning is emitted) — only None triggers the fallback.
    sparse = _sel_of(has_catalog=True, free_bytes=110 * GB, max_gals_per_row=1)
    dense = _sel_of(has_catalog=True, free_bytes=110 * GB,
                    max_gals_per_row=CAL_MAX_GALS_PER_ROW)
    assert sparse >= dense


# ── JAX-03: the catalog estimate is spectral COMMON + catalog INCREMENT ─────────
#
# The old model multiplied the WHOLE per-unit cost — and the gradient
# working-set floor — by ``gals_ratio * n_catalogs``.  At the calibration shapes
# a one-galaxy-per-row catalog therefore predicted a 1.746 GB gradient peak
# against the 80.283 GB the equivalent catalog-free path predicts: ~46x below a
# working set the two paths SHARE (same population log density, same exact q
# normalisation, same per-sample selection/PE weights).  ``auto`` would promise
# a single pass on a 40-60 GB card for a graph needing ~80 GB.

_CAL_SHAPES = dict(n_events=CAL_N_EVENTS, n_samp=CAL_N_SAMP, n_sel=CAL_N_SEL)

#: Catalog-free predictions at the calibration shapes, pinned so a remodel of
#: the catalog path cannot move the spectral path it is measured against.
_SPECTRAL_GRAD_PEAK = 80_283_217_392.0
_SPECTRAL_VALUE_PEAK = 6_413_471_824.0


@pytest.mark.parametrize("needs_grad,expected",
                         [(True, _SPECTRAL_GRAD_PEAK), (False, _SPECTRAL_VALUE_PEAK)])
def test_spectral_path_prediction_is_unchanged(needs_grad, expected):
    """The catalog remodel must not move the catalog-free path by one byte."""
    assert predicted_peak_bytes(
        **_CAL_SHAPES, has_catalog=False, needs_grad=needs_grad) == expected


@pytest.mark.parametrize("needs_grad", [True, False])
@pytest.mark.parametrize("max_gals", [1, 2, 10, 300, 1000, CAL_MAX_GALS_PER_ROW,
                                      2 * CAL_MAX_GALS_PER_ROW])
@pytest.mark.parametrize("n_catalogs", [1, 2, 3])
def test_catalog_never_predicts_below_the_catalog_free_baseline(
        needs_grad, max_gals, n_catalogs):
    """Adding catalog work can only ADD memory.  No density empties the shared set."""
    spectral = predicted_peak_bytes(
        **_CAL_SHAPES, has_catalog=False, needs_grad=needs_grad)
    catalog = predicted_peak_bytes(
        **_CAL_SHAPES, has_catalog=True, needs_grad=needs_grad,
        max_gals_per_row=max_gals, n_catalogs=n_catalogs)
    assert catalog >= spectral


def test_sparse_catalog_no_longer_underpredicts_by_46x():
    """The exact reproduction: max row 1 predicted 1.746 GB against 80.283 GB."""
    sparse = predicted_peak_bytes(
        **_CAL_SHAPES, has_catalog=True, needs_grad=True, max_gals_per_row=1)
    assert sparse >= _SPECTRAL_GRAD_PEAK
    # ... and it is an INCREMENT on that baseline, not a rescaling of it.
    assert sparse < 1.01 * _SPECTRAL_GRAD_PEAK


@pytest.mark.parametrize("needs_grad", [True, False])
def test_catalog_peak_is_monotonic_in_density_and_count(needs_grad):
    """More galaxies per row, or more catalogs, never predicts LESS memory."""
    widths = [1, 2, 10, 300, 1000, CAL_MAX_GALS_PER_ROW, 2 * CAL_MAX_GALS_PER_ROW]
    peaks = [predicted_peak_bytes(**_CAL_SHAPES, has_catalog=True,
                                  needs_grad=needs_grad, max_gals_per_row=m)
             for m in widths]
    assert peaks == sorted(peaks)
    assert peaks[-1] > peaks[0]                      # the term is not inert
    by_k = [predicted_peak_bytes(**_CAL_SHAPES, has_catalog=True,
                                 needs_grad=needs_grad,
                                 max_gals_per_row=CAL_MAX_GALS_PER_ROW,
                                 n_catalogs=k)
            for k in (1, 2, 3, 4)]
    assert by_k == sorted(by_k)
    assert by_k[-1] > by_k[0]


def _legacy_catalog_peak(cat_scale, needs_grad):
    """The shipped multiplicative model, recomputed from the same constants."""
    if needs_grad:
        fixed = GRAD_RUNTIME_FIXED_BYTES + cat_scale * GRAD_WORKSET_FLOOR_BYTES
        sel_b, pe_b = SEL_BYTES_PER_INJECTION_CAT, PE_BYTES_PER_SAMPLE_CAT
    else:
        fixed = float(TRUE_FIXED_VALUE_BYTES)
        sel_b, pe_b = SEL_BYTES_PER_INJECTION_VALUE_CAT, PE_BYTES_PER_SAMPLE_VALUE_CAT
    sel = CAL_N_SEL * sel_b * cat_scale
    pe = CAL_N_EVENTS * CAL_N_SAMP * pe_b * cat_scale
    return fixed + (sel + pe if needs_grad else max(sel, pe))


@pytest.mark.parametrize("needs_grad", [True, False])
@pytest.mark.parametrize("max_gals,n_catalogs,cat_scale", [
    (CAL_MAX_GALS_PER_ROW, 1, 1.0),
    (2 * CAL_MAX_GALS_PER_ROW, 1, 2.0),
    (CAL_MAX_GALS_PER_ROW, 3, 3.0),
])
def test_dense_catalog_predictions_are_bit_identical_to_the_shipped_model(
        needs_grad, max_gals, n_catalogs, cat_scale):
    """At and above the calibration density nothing moves: the remodel only lifts
    the SPARSE end.  The _CAT constants are still unmeasured 2x estimates, so
    lowering a dense-catalog prediction is the OOM-risky direction."""
    assert predicted_peak_bytes(
        **_CAL_SHAPES, has_catalog=True, needs_grad=needs_grad,
        max_gals_per_row=max_gals, n_catalogs=n_catalogs
    ) == _legacy_catalog_peak(cat_scale, needs_grad)


def test_sparse_catalog_cannot_single_pass_a_card_the_spectral_path_blocks():
    """The operational consequence: a 60 GB card.  The shared gradient graph needs
    ~80 GB, so neither path may promise a single pass — the sparse catalog used to."""
    base = dict(n_events=CAL_N_EVENTS, n_samp=CAL_N_SAMP, n_sel=CAL_N_SEL,
                sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
                flow_path=False, needs_grad=True, free_bytes=60 * GB,
                backend="gpu")
    spectral = resolve_block_sizes(has_catalog=False, **base)
    sparse = resolve_block_sizes(has_catalog=True, max_gals_per_row=1, **base)
    assert spectral.sel_batch_size is not None
    assert sparse.sel_batch_size is not None
    assert sparse.sel_batch_size <= spectral.sel_batch_size


# ── P-6: the population q-quadrature grid enters the peak model ──────────────────

def test_monotonic_in_n_q():
    # n_q multiplies the per-injection / per-sample working set (the default
    # pairing normaliser integrates over q PER SAMPLE), so more nodes must never
    # yield a larger selection block.  Before this term, --norm_nq 800 got the
    # IDENTICAL plan to --norm_nq 200 against a 4x larger working set.
    sels = [_sel_of(has_catalog=False, free_bytes=110 * GB, n_q=q)
            for q in (32, 64, CAL_N_Q, 400, 800)]
    assert sels == sorted(sels, reverse=True)
    assert sels[0] > sels[-1]               # the term is not inert


def test_n_q_scale_is_unity_at_the_calibration_point():
    kw = dict(n_events=CAL_N_EVENTS, n_samp=CAL_N_SAMP, n_sel=CAL_N_SEL,
              sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
              has_catalog=False, flow_path=False, n_grid=CAL_N_GRID,
              static_state_bytes=STATIC_STATE_CAL_BYTES,
              free_bytes=100 * GB, backend="gpu")
    assert resolve_block_sizes(**kw) == resolve_block_sizes(n_q=CAL_N_Q, **kw)


# ── P-4: gradient-free samplers get the value-only peak model ────────────────────

def test_sampler_block_sizing_profile():
    from types import SimpleNamespace as NS
    assert sampler_block_sizing_profile(NS(sampler="numpyro")) == (True, 1)
    assert sampler_block_sizing_profile(NS(sampler="dynesty")) == (False, 1)
    assert sampler_block_sizing_profile(NS(sampler="tinyns")) == (False, 1)
    # tinyns vmaps the loglike over replacement_chains (and over the largest entry
    # of a chain schedule), so that many evaluations are live at once.
    assert sampler_block_sizing_profile(NS(
        sampler="tinyns",
        tinyns_resolved_config={"replacement_chains": 16},
    )) == (False, 16)
    assert sampler_block_sizing_profile(NS(
        sampler="tinyns",
        tinyns_resolved_config={"replacement_chains": 1,
                                "replacement_chain_schedule": (1, 4, 64)},
    )) == (False, 64)
    # jax_block_size scans whole ITERATIONS (sequential) -> not a memory multiplier.
    assert sampler_block_sizing_profile(NS(
        sampler="tinyns", tinyns_resolved_config={"jax_block_size": 32},
    )) == (False, 1)
    # Unknown / missing sampler is treated as gradient-free but never batched.
    assert sampler_block_sizing_profile(NS()) == (False, 1)


def test_needs_grad_defaults_to_true():
    # The default MUST stay the (calibrated) value+grad model so an un-updated
    # caller keeps the conservative behaviour.
    kw = dict(n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL,
              sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
              has_catalog=False, flow_path=False,
              free_bytes=95 * GB, backend="gpu")
    assert resolve_block_sizes(**kw) == resolve_block_sizes(needs_grad=True, **kw)


def _plan(free, **kw):
    base = dict(n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL,
                sel_requested=BLOCK_AUTO, pe_requested=BLOCK_AUTO,
                has_catalog=False, flow_path=False,
                free_bytes=free, backend="gpu")
    base.update(kw)
    return resolve_block_sizes(**base)


def test_value_only_single_pass_where_grad_model_blocks():
    """The production defect: on the H100 NVL (88.1 GiB probed free) the value+grad
    model blocks a dynesty run into (32768, 87) — MEASURED 49.3 ms/call for a
    2.27 GB peak — where the single pass is 27.5 ms/call at 5.65 GB."""
    free = int(88.1 * GB)
    grad = _plan(free, needs_grad=True)
    assert grad.sel_batch_size == SEL_MIN_BATCH and grad.pe_event_block is not None
    value = _plan(free, needs_grad=False)
    assert (value.sel_batch_size, value.pe_event_block) == (None, None)
    assert value.source == "auto-single-pass"


def test_value_only_single_pass_on_a_small_gpu():
    # The measured value peak of the single pass is 5.65 GB, so any GPU with more
    # than ~10 GB free needs no blocking at all for a gradient-free run.
    assert _plan(16 * GB, needs_grad=False).sel_batch_size is None
    assert _plan(16 * GB, needs_grad=False).pe_event_block is None
    # ... while the value+grad model (rightly) blocks the same card hard.
    assert _plan(16 * GB, needs_grad=True).sel_batch_size == SEL_MIN_BATCH


def _predicted(**kw):
    base = dict(n_events=N_EVENTS, n_samp=N_SAMP, n_sel=N_SEL, has_catalog=False)
    base.update(kw)
    return predicted_peak_bytes(**base)


def test_value_only_model_brackets_the_measured_peaks():
    """Formula-level calibration check against the H100 NVL value-only sweep
    (scripts/benchmarks/block_sizes_h100_80gb_value_only.json): the model must be
    conservative (predict >= measured) but not wildly so."""
    for (sel, pe), measured in [((None, None), 5.648e9),
                                ((32768, None), 5.695e9),
                                ((None, 8), 5.648e9),
                                ((32768, 8), 0.703e9),
                                ((65536, 16), 0.990e9),
                                ((131072, 32), 1.187e9)]:
        p = _predicted(needs_grad=False, sel_batch_size=sel, pe_event_block=pe)
        assert p >= measured, (sel, pe, p, measured)
        assert p <= 2.0 * measured, (sel, pe, p, measured)
    # Sanity: the components are the ones documented, not an accident.
    assert TRUE_FIXED_VALUE_BYTES < 1.1e9
    assert SEL_BYTES_PER_INJECTION_VALUE == PE_BYTES_PER_SAMPLE_VALUE == 5_000


def test_value_only_model_brackets_the_measured_n_q_sweep():
    """The n_q term is calibrated too: MEASURED value peak at sel/pe = off/off vs
    --norm_nq on the H100 NVL."""
    for n_q, measured in ((32, 1.34e9), (64, 2.16e9), (100, 3.09e9),
                          (200, 5.65e9), (400, 10.78e9), (800, 21.03e9)):
        p = _predicted(needs_grad=False, n_q=n_q)
        assert p >= measured, (n_q, p, measured)
        assert p <= 2.0 * measured, (n_q, p, measured)


def test_grad_model_brackets_the_two_measured_n_q_points():
    """The value+GRAD single-pass peak is ~97% dimension-scaled working set, not a
    58 GiB fixed term: MEASURED 26.89 GB at n_q=64 and 78.77 GB at n_q=200 on the
    same config.  The split must reproduce both to ~10%, or the model promises a
    single pass at raised --norm_nq against a peak twice the card."""
    for n_q, measured in ((64, 26.89e9), (200, 78.77e9)):
        p = _predicted(needs_grad=True, n_q=n_q,
                       static_state_bytes=STATIC_STATE_CAL_BYTES)
        assert 0.9 * measured <= p <= 1.15 * measured, (n_q, p, measured)
    # A pure-fixed model (the pre-split behaviour) would have predicted ~68 GB at
    # n_q=64 -- 2.5x the measured peak in the OPTIMISTIC direction once inverted
    # into a budget, which is what let --norm_nq defeat the guard.
    legacy_64 = TRUE_FIXED_BYTES + STATIC_STATE_CAL_BYTES + (
        N_SEL * 8_000 + N_EVENTS * N_SAMP * 9_000) * 64 / CAL_N_Q
    assert legacy_64 > 2.0 * 26.89e9


def test_grad_fixed_split_preserves_the_calibration_anchor():
    from darksirens.likelihood.block_sizing import (
        GRAD_RUNTIME_FIXED_BYTES, GRAD_WORKSET_FLOOR_BYTES,
    )
    assert GRAD_RUNTIME_FIXED_BYTES + GRAD_WORKSET_FLOOR_BYTES == TRUE_FIXED_BYTES
    # and the runtime part really is the small one
    assert GRAD_RUNTIME_FIXED_BYTES < 0.05 * GRAD_WORKSET_FLOOR_BYTES


def test_value_only_blocks_both_axes_together():
    """Measured: blocking ONE axis alone leaves the value peak unchanged (5.648 ->
    5.695 GB) because the unblocked axis's (N, n_q) intermediate still sets it.  So
    the value-only policy must never return a sel-blocked / pe-single-pass plan."""
    # A budget small enough that the selection axis must chunk.
    plan = _plan(8 * GB, needs_grad=False)
    assert plan.sel_batch_size is not None
    assert plan.pe_event_block is not None


def test_value_only_monotonic_in_free_bytes():
    def _sel(free):
        s = _plan(free, needs_grad=False).sel_batch_size
        return N_SEL if s is None else s
    sels = [_sel(f) for f in (2 * GB, 4 * GB, 8 * GB, 10 * GB, 40 * GB)]
    assert sels == sorted(sels)


def test_concurrent_evals_shrinks_blocks():
    # tinyns replacement_chains vmaps the loglike, so every intermediate is
    # batched: more concurrent evaluations must never yield a larger block.
    def _sel(n):
        s = _plan(40 * GB, needs_grad=False, concurrent_evals=n).sel_batch_size
        return N_SEL if s is None else s
    sels = [_sel(n) for n in (1, 2, 8, 16, 64)]
    assert sels == sorted(sels, reverse=True)
    assert sels[0] > sels[-1]


def test_value_only_explicit_still_wins():
    plan = _plan(40 * GB, needs_grad=False, sel_requested=65536, pe_requested=32)
    assert plan == BlockSizePlan(65536, 32, "explicit")


# ── CLI wiring: _block_sizing_inputs passes real values ──────────────────────────

def test_cli_block_sizing_inputs_plumbing():
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from darksirens.cli.inference import _block_sizing_inputs
    from darksirens.redshift.grid import zgrid

    zgals_pe = np.zeros((10, 50), np.float64)          # 10*50*8 = 4000 B
    data = {
        "m1detsels": np.zeros(1000, np.float64),        # n_sel = 1000
        "nEvents": 5, "nsamp": 100,
        "zgals_pe": zgals_pe,
        "catalog_memory": {
            "max_galaxies_per_unique_pixel": 50,
            "unique_pe_pixels": 10, "unique_sel_pixels": 10,
        },
    }
    opts = SimpleNamespace(
        survey_path="cat.h5", n_catalogs=1, gw_flows_path=None,
        drop_full_catalog=False, sampler="numpyro",
        sel_batch_size=BLOCK_AUTO, pe_event_block=BLOCK_AUTO,
    )
    kw = _block_sizing_inputs(opts, data)
    assert kw["n_sel"] == 1000
    assert kw["n_events"] == 5 and kw["n_samp"] == 100
    assert kw["has_catalog"] is True
    assert kw["n_catalogs"] == 1
    assert kw["max_gals_per_row"] == 50
    assert kw["n_q"] == CAL_N_Q
    assert kw["needs_grad"] is True and kw["concurrent_evals"] == 1
    ng = int(np.asarray(zgrid).shape[0])
    assert kw["n_grid"] == ng
    # Resolver reserves the PENDING (factory) static: KDE cache (10+10)*n_grid*8,
    # NOT the already-resident loaded zgals_pe (that is reflected in free_bytes).
    assert kw["static_state_bytes"] == (10 + 10) * ng * 8
    # The report-only FULL measured static also includes ALL loaded arrays
    # (m1detsels 1000*8 + zgals_pe 10*50*8) plus the same KDE estimate.
    assert kw["static_state_full_bytes"] == (
        data["m1detsels"].nbytes + zgals_pe.nbytes + (10 + 10) * ng * 8
    )
    # plan resolves end-to-end after stripping the report-only key (no sentinel leaks).
    kw.pop("static_state_full_bytes")
    plan = resolve_block_sizes(free_bytes=200 * GB, backend="gpu", **kw)
    assert plan.source in ("auto", "auto-single-pass", "auto-floor-reduced")


def test_cli_block_sizing_inputs_k2_bundles():
    """A K >= 2 multitracer run has NO top-level catalog_memory / zgals_* (data.py
    stubs catalog_inputs, so loaders never builds them) — the two dominant catalog
    dimensions have to come from data['catalogs'] or the model is blind."""
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from darksirens.cli.inference import _block_sizing_inputs
    from darksirens.redshift.grid import zgrid

    bundles = [
        {"zgals_pe": np.zeros((6, 2113), np.float64),
         "zgals_sel": np.zeros((6, 2113), np.float64)},
        {"zgals_pe": np.zeros((9, 800), np.float64),
         "zgals_sel": np.zeros((9, 800), np.float64)},
    ]
    data = {
        "m1detsels": np.zeros(1000, np.float64),
        "nEvents": 5, "nsamp": 100,
        "catalog_memory": None,           # stubbed away for K >= 2
        "zgals_pe": None, "zgals_sel": None,
        "catalogs": bundles,
    }
    opts = SimpleNamespace(
        survey_path="cat1.h5", n_catalogs=2, gw_flows_path=None,
        drop_full_catalog=False, sampler="dynesty",
        sel_batch_size=BLOCK_AUTO, pe_event_block=BLOCK_AUTO,
    )
    kw = _block_sizing_inputs(opts, data)
    ng = int(np.asarray(zgrid).shape[0])
    assert kw["has_catalog"] is True and kw["n_catalogs"] == 2
    # widest bundle row, NOT the silent 1 the stubbed catalog_memory produced
    assert kw["max_gals_per_row"] == 2113
    # per-bundle KDE caches, NOT 0
    assert kw["static_state_bytes"] == ((6 + 6) + (9 + 9)) * ng * 8
    assert kw["static_state_bytes"] > 0
    # dynesty is gradient-free
    assert kw["needs_grad"] is False


def test_cli_block_sizing_inputs_tinyns_concurrency():
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from darksirens.cli.inference import _block_sizing_inputs

    opts = SimpleNamespace(
        survey_path=None, n_catalogs=1, gw_flows_path=None,
        drop_full_catalog=False, sampler="tinyns",
        tinyns_resolved_config={"replacement_chains": 16},
        sel_batch_size=BLOCK_AUTO, pe_event_block=BLOCK_AUTO,
    )
    kw = _block_sizing_inputs(opts, {"m1detsels": np.zeros(10), "nEvents": 2,
                                     "nsamp": 4})
    assert kw["needs_grad"] is False and kw["concurrent_evals"] == 16


def test_cli_block_sizing_inputs_reads_norm_nq():
    pytest.importorskip("jax")
    from types import SimpleNamespace
    from darksirens.cli.inference import _block_sizing_inputs
    from darksirens.gw.populations.utils import (
        configure_normalization_grids, normalization_grid_settings,
    )

    original = normalization_grid_settings().n_q
    try:
        configure_normalization_grids(n_q=777)
        opts = SimpleNamespace(
            survey_path=None, n_catalogs=1, gw_flows_path=None,
            drop_full_catalog=False, sampler="numpyro",
            sel_batch_size=BLOCK_AUTO, pe_event_block=BLOCK_AUTO,
        )
        kw = _block_sizing_inputs(opts, {"m1detsels": np.zeros(10), "nEvents": 2,
                                         "nsamp": 4})
        assert kw["n_q"] == 777
    finally:
        configure_normalization_grids(n_q=original)


def test_lensing_cli_threads_sampler_profile(monkeypatch):
    """The lensing twin resolves its own block size; the halted lensing campaign
    runs --sampler tinyns, which must get the value-only model too."""
    pytest.importorskip("jax")
    from types import SimpleNamespace
    import darksirens.cli.inference_lensing as lens

    seen = {}

    def _fake_resolve(**kw):
        seen.update(kw)
        return BlockSizePlan(None, None, "auto-single-pass")

    monkeypatch.setattr(lens, "resolve_block_sizes", _fake_resolve)
    opts = SimpleNamespace(sampler="tinyns", sel_batch_size=BLOCK_AUTO,
                           tinyns_resolved_config={"replacement_chains": 4})
    inp = {"gw_sel": SimpleNamespace(m1det=np.zeros(500)),
           "nEvents": 10, "nsamp": 64}
    settings = {}
    lens._resolve_lensing_block_sizes(opts, inp, settings)
    assert seen["needs_grad"] is False
    assert seen["concurrent_evals"] == 4
    assert seen["n_q"] == CAL_N_Q
    assert settings["sel_batch_size"] is None


# ── free-memory probe: the platform-allocator fallback (issue #276) ──────────────
#
# Under XLA_PYTHON_CLIENT_ALLOCATOR=platform ``device.memory_stats()`` returns
# None. That was ``core.jax_config``'s default until 2026-08-23 and is still an
# honored explicit override. Before the nvidia-smi fallback every such CLI run
# fell through to the 4 GB default (measured: 4 GB reported on an H100 NVL with
# 92.4 GB actually free), so memory-aware sizing — and the calibrated
# single-pass constants — never engaged.

class _FakeDevice:
    def __init__(self, stats, platform="gpu", dev_id=0):
        self._stats, self.platform, self.id = stats, platform, dev_id

    def memory_stats(self):
        return self._stats


def _patch_devices(monkeypatch, device):
    import jax
    monkeypatch.setattr(jax, "devices", lambda *a, **k: [device])


def test_probe_env_override_wins(monkeypatch):
    from darksirens.likelihood.block_sizing import (
        DEVICE_MEM_ENV_VAR, probe_device_memory_bytes,
    )
    monkeypatch.setenv(DEVICE_MEM_ENV_VAR, "12.5")
    # Even with a working memory_stats, the explicit override takes priority.
    _patch_devices(monkeypatch, _FakeDevice({"bytes_limit": 99 * GB}))
    assert probe_device_memory_bytes() == (int(12.5e9), f"env:{DEVICE_MEM_ENV_VAR}")


@pytest.mark.parametrize("bad", ["", "not-a-number", "0", "-3"])
def test_probe_env_override_ignored_when_unusable(monkeypatch, bad):
    from darksirens.likelihood.block_sizing import (
        DEVICE_MEM_ENV_VAR, probe_device_memory_bytes,
    )
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.setenv(DEVICE_MEM_ENV_VAR, bad)
    _patch_devices(monkeypatch, _FakeDevice({"bytes_limit": 40 * GB}))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes", lambda i: None)
    nbytes, source = probe_device_memory_bytes()
    assert source == "device:gpu bytes_limit-in_use" and nbytes == 40 * GB


def test_probe_prefers_memory_stats_over_nvidia_smi(monkeypatch):
    """Allocator telemetry is the preferred signal when the device is not shared.

    nvidia-smi reports MORE free than the allocator's own headroom here (nothing
    else is on the card), so the telemetry value passes through untouched — it
    is the view the peak constants were calibrated against.
    """
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch, _FakeDevice({"bytes_limit": 40 * GB, "bytes_in_use": 10 * GB}))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes", lambda i: 35 * GB)
    assert bs.probe_device_memory_bytes() == (30 * GB, "device:gpu bytes_limit-in_use")


def test_probe_clips_allocator_headroom_to_physical_free(monkeypatch):
    """SHARED GPU: BFC's bytes_limit is the whole card, blind to another process.

    The physical free memory is the smaller, and therefore the only safe budget.
    """
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch, _FakeDevice({"bytes_limit": 80 * GB, "bytes_in_use": 0}))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes", lambda i: 12 * GB)
    nbytes, source = bs.probe_device_memory_bytes()
    assert nbytes == 12 * GB
    assert source == "min(device:gpu bytes_limit-in_use, nvidia-smi memory.free)"


def test_probe_clips_reservable_limit_to_physical_free(monkeypatch):
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch, _FakeDevice({"bytes_reservable_limit": 80 * GB}))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes", lambda i: 9 * GB)
    nbytes, source = bs.probe_device_memory_bytes()
    assert nbytes == 9 * GB
    assert source == "min(device:gpu bytes_reservable_limit, nvidia-smi memory.free)"


def test_probe_keeps_telemetry_when_nvidia_smi_unavailable(monkeypatch):
    """No nvidia-smi (or it fails): the allocator view is used as-is, not dropped."""
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch, _FakeDevice({"bytes_limit": 40 * GB, "bytes_in_use": 10 * GB}))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes", lambda i: None)
    assert bs.probe_device_memory_bytes() == (30 * GB, "device:gpu bytes_limit-in_use")


def test_probe_does_not_shell_out_for_telemetry_on_cpu(monkeypatch):
    """A CPU/TPU device never consults nvidia-smi, even with live telemetry."""
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch,
                   _FakeDevice({"bytes_limit": 40 * GB, "bytes_in_use": 10 * GB},
                               platform="tpu"))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes",
                        lambda i: pytest.fail("must not shell out on a non-GPU device"))
    assert bs.probe_device_memory_bytes() == (30 * GB, "device:tpu bytes_limit-in_use")


def test_probe_falls_back_to_nvidia_smi_when_memory_stats_is_none(monkeypatch):
    """The platform-allocator case: memory_stats() -> None on a real GPU."""
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch, _FakeDevice(None))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes", lambda i: 92 * GB)
    assert bs.probe_device_memory_bytes() == (92 * GB, "nvidia-smi memory.free")


def test_probe_defaults_when_nvidia_smi_also_unavailable(monkeypatch):
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch, _FakeDevice(None))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes", lambda i: None)
    assert bs.probe_device_memory_bytes(default_gb=4.0) == (4_000_000_000, "fallback-default")


def test_probe_does_not_shell_out_on_cpu(monkeypatch):
    """No nvidia-smi subprocess on a CPU-only host."""
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch, _FakeDevice(None, platform="cpu"))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes",
                        lambda i: pytest.fail("must not shell out on CPU"))
    assert bs.probe_device_memory_bytes()[1] == "fallback-default"


def _fake_run(stdout, returncode=0):
    from types import SimpleNamespace

    def run(cmd, **kwargs):
        if returncode:
            raise RuntimeError("nvidia-smi failed")
        return SimpleNamespace(stdout=stdout, returncode=0)
    return run


@pytest.mark.parametrize("stdout,expected", [
    ("92433\n", int(92433 * 1024 * 1024)),
    ("  92433  \n", int(92433 * 1024 * 1024)),
    ("92433\n81920\n", int(92433 * 1024 * 1024)),   # first line only
    ("", None),
    ("[N/A]\n", None),
    ("0\n", None),
])
def test_nvidia_smi_parsing(monkeypatch, stdout, expected):
    import subprocess
    from darksirens.likelihood.block_sizing import _nvidia_smi_free_bytes
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_run(stdout))
    assert _nvidia_smi_free_bytes(0) == expected


def test_nvidia_smi_missing_binary_returns_none(monkeypatch):
    import subprocess
    from darksirens.likelihood.block_sizing import _nvidia_smi_free_bytes
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(subprocess, "run", _fake_run("", returncode=1))
    assert _nvidia_smi_free_bytes(0) is None


def test_nvidia_smi_maps_through_cuda_visible_devices(monkeypatch):
    """JAX device 0 is the FIRST entry of CUDA_VISIBLE_DEVICES, not GPU 0."""
    import subprocess
    from darksirens.likelihood.block_sizing import _nvidia_smi_free_bytes
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    seen = {}

    def run(cmd, **kwargs):
        from types import SimpleNamespace
        seen["cmd"] = cmd
        return SimpleNamespace(stdout="1024\n", returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    assert _nvidia_smi_free_bytes(0) == 1024 * 1024 * 1024
    assert "--id=3" in seen["cmd"]
    _nvidia_smi_free_bytes(1)
    assert "--id=1" in seen["cmd"]
    # Out-of-range index must not silently query the wrong GPU.
    assert _nvidia_smi_free_bytes(5) is None
