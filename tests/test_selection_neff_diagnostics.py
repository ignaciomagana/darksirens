"""Tests for the selection-N_eff diagnostic, centred on its precision canary.

The canary is the reason this script's output can be trusted at all.  Its first
version reported N_eff = 136,200 where the truth is 118,960 -- a 14% error that
was deterministic, reproducible, and invisible to every dtype check, because
``jax_enable_x64`` read ``True`` while an internal cached constant was float32.

What makes it detectable is that the curated ``powerlaw+peak`` preset has NO hard
``m_max`` truncation (90% of its weight is an untapered Gaussian at 35 +- 5), so
"out of support" is really "underflowed", and the underflow edge is
precision-dependent: m1src > 223.3 in float64, > 101.1 in float32.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts" / "selection_neff_diagnostics.py"
_spec = importlib.util.spec_from_file_location("selection_neff_diagnostics", _S)
snd = importlib.util.module_from_spec(_spec)
sys.modules["selection_neff_diagnostics"] = snd
_spec.loader.exec_module(snd)


def test_canary_passes_on_the_real_population_model():
    """float64 gives about -265.5 at the probe mass; this pins the reference."""
    import jax.numpy as jnp

    from darksirens.gw.populations import pop_model_parser
    from darksirens.gw.populations.registry import get_fixed_population_params

    lpp = pop_model_parser(pop_model="powerlaw+peak")
    fid = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
    value = snd.assert_float64_population_path(lpp, fid)
    # A BAND, not a point.  The probe value carries the population's
    # normalisation constant, which is itself mildly process-state dependent
    # (measured -265.54 standalone vs -264.88 under pytest, a factor 1.9 in
    # normalisation).  That shift is common to every sample, so it cancels in
    # N_eff by scale invariance and shifts only log_mu -- it is not what the
    # canary is for.  The canary exists to separate a finite float64 value from
    # the float32 -inf cliff, so the band is wide enough to ignore
    # normalisation and far from -inf.
    assert -400.0 < value < -100.0, (
        f"probe value {value} is outside the float64 band; float32 gives -inf "
        "here, so a value near -inf means the precision defect is back"
    )


def test_canary_rejects_a_float32_style_underflow():
    """A model that returns -inf at the probe is exactly the failure mode.

    In float32 the untapered Gaussian underflows at m1src ~ 101, so the probe at
    150 comes back -inf.  The canary must refuse to report rather than proceed.
    """
    import jax.numpy as jnp

    def degraded(m1, q, z, chi, params):
        return jnp.full(jnp.shape(m1), -jnp.inf)

    with pytest.raises(SystemExit, match="precision canary FAILED"):
        snd.assert_float64_population_path(degraded, jnp.asarray([0.0]))


def test_canary_rejects_a_merely_implausible_value():
    """Guard the threshold too: anything past the float32 edge is refused."""
    import jax.numpy as jnp

    def suspicious(m1, q, z, chi, params):
        return jnp.full(jnp.shape(m1), -1.0e4)

    with pytest.raises(SystemExit, match="precision canary FAILED"):
        snd.assert_float64_population_path(suspicious, jnp.asarray([0.0]))


def test_verdict_sums_both_terms_and_matches_the_guard_formula():
    """The whole point is that BOTH terms enter; a one-term verdict is the
    error that produced two retractions during the 2026-08-12 session."""
    v = snd.verdict(118960.0, 259, 0.2728, 1.0)
    assert v["selection_variance"] == pytest.approx(259 ** 2 / 118960.0)
    assert v["sigma2_total"] == pytest.approx(0.2728 + 259 ** 2 / 118960.0)
    assert v["threshold"] == pytest.approx(259 ** 2 / (1.0 - 0.2728))
    assert v["margin"] == pytest.approx(118960.0 / v["threshold"])
    assert v["passes"] is True
    assert v["variance_criterion_limited"] is True


def test_verdict_fails_when_the_total_exceeds_the_cap():
    """The 282-event chieff configuration, which fails on the TOTAL even though
    each term alone is under the cap."""
    v = snd.verdict(118960.0, 282, 0.5632, 1.0)
    assert v["pe_variance_sum"] < 1.0 and v["selection_variance"] < 1.0
    assert v["sigma2_total"] > 1.0
    assert v["passes"] is False


def test_verdict_reports_the_sparse_floor_regime():
    v = snd.verdict(1e9, 3, 0.0, 1.0)
    assert v["threshold"] == pytest.approx(15.0)
    assert v["variance_criterion_limited"] is False


def test_raw_read_rejects_a_file_missing_pdraw(tmp_path):
    import h5py

    p = tmp_path / "bad.h5"
    with h5py.File(p, "w") as f:
        f.attrs["ndraw"] = 10.0
        for k in ("m1det", "m2det", "dL", "chieff"):
            f.create_dataset(k, data=np.ones(4))
    with pytest.raises(SystemExit, match="missing dataset"):
        snd._load_selection_arrays(p)


def test_raw_read_rejects_a_file_missing_ndraw(tmp_path):
    import h5py

    p = tmp_path / "bad2.h5"
    with h5py.File(p, "w") as f:
        for k in ("m1det", "m2det", "dL", "chieff", "pdraw"):
            f.create_dataset(k, data=np.ones(4))
    with pytest.raises(SystemExit, match="missing required attr 'ndraw'"):
        snd._load_selection_arrays(p)


def test_raw_read_surfaces_the_spin_basis(tmp_path):
    """The basis must be reported, because a non-chieff file is a raw read that
    the loader would have refused, and the caveat depends on knowing which."""
    import h5py

    p = tmp_path / "comp.h5"
    with h5py.File(p, "w") as f:
        f.attrs["ndraw"] = 100.0
        f.attrs["spin_basis"] = "component"
        f.attrs["format_version"] = "gwcat-selection-2.0"
        for k in ("m1det", "m2det", "dL", "chieff", "pdraw"):
            f.create_dataset(k, data=np.ones(4))
    _cols, meta = snd._load_selection_arrays(p)
    assert meta["spin_basis"] == "component"
    assert meta["format_version"] == "gwcat-selection-2.0"
    assert meta["n_detected"] == 4


def test_log_mu_is_flagged_as_non_portable(capsys, tmp_path):
    """`log_mu` must be labelled, because it is the one reported number that
    cannot be compared against anything from another process.

    It carries the population's normalisation constant, which is process-state
    dependent (measured: a factor 1.9 between two import orders). Within a
    likelihood evaluation that constant cancels exactly -- c^N_obs from the
    per-event numerators against c^N_obs from mu^N_obs -- so posteriors, logZ
    and N_eff are all unaffected. Only this isolated log_mu is not portable, and
    a reader comparing it to a stored reference would draw a false conclusion.
    """
    import h5py

    p = tmp_path / "sel.h5"
    rng = np.random.default_rng(3)
    n = 500
    with h5py.File(p, "w") as f:
        f.attrs["ndraw"] = 1.0e6
        f.create_dataset("m1det", data=rng.uniform(10.0, 60.0, n))
        f.create_dataset("m2det", data=rng.uniform(5.0, 30.0, n))
        f.create_dataset("dL", data=rng.uniform(200.0, 3000.0, n))
        f.create_dataset("chieff", data=rng.uniform(-0.3, 0.3, n))
        f.create_dataset("pdraw", data=rng.uniform(1e-9, 1e-7, n))
    out = tmp_path / "r.json"
    assert snd.main(["--selection_path", str(p), "--n_obs", "10",
                     "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "NOT comparable across processes" in printed
    assert "cancels exactly" in printed
    import json
    assert json.loads(out.read_text())["log_mu_comparable_across_processes"] is False
