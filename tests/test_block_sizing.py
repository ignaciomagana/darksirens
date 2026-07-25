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
    FIXED_OVERHEAD_BYTES,
    TRUE_FIXED_BYTES,
    STATIC_STATE_CAL_BYTES,
    SAFETY_FACTOR,
    block_size_arg,
    resolve_block_sizes,
    measure_static_state_bytes,
    estimate_pending_static_bytes,
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
    # members 3600 + KDE (2+2)*30*8=960 + base_miss 2*(5*30*8)=2400
    assert total == 3600 + 960 + 2400


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
        drop_full_catalog=False,
        sel_batch_size=BLOCK_AUTO, pe_event_block=BLOCK_AUTO,
    )
    kw = _block_sizing_inputs(opts, data)
    assert kw["n_sel"] == 1000
    assert kw["n_events"] == 5 and kw["n_samp"] == 100
    assert kw["has_catalog"] is True
    assert kw["n_catalogs"] == 1
    assert kw["max_gals_per_row"] == 50
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


# ── free-memory probe: the platform-allocator fallback (issue #276) ──────────────
#
# ``core.jax_config`` sets XLA_PYTHON_CLIENT_ALLOCATOR=platform, under which
# ``device.memory_stats()`` returns None. Before the nvidia-smi fallback every
# production CLI run therefore fell through to the 4 GB default (measured: 4 GB
# reported on an H100 NVL with 92.4 GB actually free), so memory-aware sizing —
# and the calibrated single-pass constants — never engaged.

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
    monkeypatch.setenv(DEVICE_MEM_ENV_VAR, bad)
    _patch_devices(monkeypatch, _FakeDevice({"bytes_limit": 40 * GB}))
    nbytes, source = probe_device_memory_bytes()
    assert source == "device:gpu bytes_limit-in_use" and nbytes == 40 * GB


def test_probe_prefers_memory_stats_over_nvidia_smi(monkeypatch):
    import darksirens.likelihood.block_sizing as bs
    monkeypatch.delenv(bs.DEVICE_MEM_ENV_VAR, raising=False)
    _patch_devices(monkeypatch, _FakeDevice({"bytes_limit": 40 * GB, "bytes_in_use": 10 * GB}))
    monkeypatch.setattr(bs, "_nvidia_smi_free_bytes",
                        lambda i: pytest.fail("nvidia-smi must not be reached"))
    assert bs.probe_device_memory_bytes() == (30 * GB, "device:gpu bytes_limit-in_use")


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
