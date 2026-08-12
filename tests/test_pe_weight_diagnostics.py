"""Tests for the PE-weight / variance-budget diagnostic.

The quantity under test is load-bearing: ``pe_variance_sum`` is subtracted from
``max_likelihood_variance`` to form the budget that sets the selection-N_eff
threshold (``darksirens/likelihood/selection.py:311-312``), so a wrong number
here would mis-state how much headroom a run has.  The estimator is therefore
pinned against cases with closed-form answers rather than against a golden.
"""
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pe_weight_diagnostics.py"
_spec = importlib.util.spec_from_file_location("pe_weight_diagnostics", _SCRIPT)
pwd = importlib.util.module_from_spec(_spec)
sys.modules["pe_weight_diagnostics"] = pwd
_spec.loader.exec_module(pwd)


def test_uniform_weights_give_full_ess_and_zero_variance():
    """Constant p_pe is the no-degeneracy limit: ESS = nsamp, sigma^2 = 0.

    sigma^2 = sum(w^2)/sum(w)^2 - 1/n = (n c^2)/(n c)^2 - 1/n = 0 exactly.
    """
    p = np.full((3, 64), 0.25)
    s = pwd.per_event_weight_stats(p)
    np.testing.assert_allclose(s["ess"], 64.0)
    np.testing.assert_allclose(s["ess_frac"], 1.0)
    np.testing.assert_allclose(s["variance"], 0.0, atol=1e-15)


def test_fully_degenerate_weights_saturate_the_variance():
    """One sample carrying all the weight is the other extreme: sigma^2 = 1 - 1/n.

    Cauchy-Schwarz bounds sigma^2 in [0, 1 - 1/n]; this is the upper edge, and
    it is the bound the guard's docstring quotes.
    """
    n = 32
    p = np.full((1, n), 1e12)
    p[0, 0] = 1.0            # w = 1/p, so sample 0 dominates by 1e12
    s = pwd.per_event_weight_stats(p)
    assert s["ess"][0] == pytest.approx(1.0, rel=1e-6)
    assert s["variance"][0] == pytest.approx(1.0 - 1.0 / n, rel=1e-6)


def test_non_positive_p_pe_is_zero_weighted_not_counted():
    """A zero p_pe must be dropped, matching the likelihood's own masking.

    ``likelihood/core.py:968`` masks ``prior_wt > 0``, so such a sample
    contributes no weight -- but it still COUNTS in n (it is a zero-weight
    member of the same PE set, per log_evidence_and_mc_variance's docstring).
    Here 8 of 16 samples are zeroed, so the ESS of the survivors is 8 while the
    1/n term still uses 16.
    """
    p = np.full((1, 16), 0.5)
    p[0, 8:] = 0.0
    s = pwd.per_event_weight_stats(p)
    assert s["n_masked"] == 8
    assert s["ess"][0] == pytest.approx(8.0)
    # sum(w^2)/sum(w)^2 = (8*4)/(8*2)^2 = 1/8, minus 1/16
    assert s["variance"][0] == pytest.approx(1.0 / 8.0 - 1.0 / 16.0)


def test_negative_p_pe_is_treated_like_zero():
    """Defensive: a negative density is nonsense, and must not become a
    negative weight that could cancel a real one."""
    p = np.full((1, 8), 0.5)
    p[0, 0] = -1.0
    s = pwd.per_event_weight_stats(p)
    assert s["n_masked"] == 1
    assert s["ess"][0] == pytest.approx(7.0)


def test_guard_threshold_inflation_matches_the_closed_form():
    """The reported inflation must be exactly budget^-1 relative to full budget."""
    g = pwd.guard_thresholds(0.2728, 259, 1.0)
    assert g["budget_remaining"] == pytest.approx(0.7272)
    assert g["threshold"] == pytest.approx(259 ** 2 / 0.7272)
    assert g["threshold_if_pe_variance_were_zero"] == pytest.approx(259 ** 2 / 1.0)
    assert g["inflation_factor"] == pytest.approx(1.0 / 0.7272, rel=1e-9)
    assert g["variance_criterion_limited"] is True
    assert g["sparse_floor"] == pytest.approx(5 * 259)


def test_sparse_floor_binds_for_small_event_counts():
    """With few events the 5*N_obs floor dominates, and must be reported as such.

    At N_obs = 3 and a full budget, N^2/budget = 9 < 5*N = 15, so the guard is
    floor-limited -- the opposite regime from the production configuration.
    """
    g = pwd.guard_thresholds(0.0, 3, 1.0)
    assert g["threshold"] == pytest.approx(15.0)
    assert g["variance_criterion_limited"] is False


def test_exhausted_budget_does_not_divide_by_zero():
    """pe_variance_sum >= max_likelihood_variance must clamp, not blow up."""
    g = pwd.guard_thresholds(1.5, 10, 1.0)
    assert np.isfinite(g["threshold"])
    assert g["budget_remaining"] == pytest.approx(pwd.MIN_VARIANCE_BUDGET)


def _write_pe(path, p_pe, nobs, nsamp):
    with h5py.File(path, "w") as f:
        f.attrs["nobs"] = nobs
        f.attrs["nsamp"] = nsamp
        f.create_dataset("p_pe", data=np.asarray(p_pe, dtype=float).ravel())


def test_report_end_to_end_on_a_synthetic_file(tmp_path, capsys):
    path = tmp_path / "pe.h5"
    _write_pe(path, np.full((4, 32), 0.5), 4, 32)
    out = pwd.report(path, 1.0)
    assert out["n_events"] == 4 and out["nsamp"] == 32
    assert out["pe_variance_sum"] == pytest.approx(0.0, abs=1e-15)
    assert out["ess_frac"]["median"] == pytest.approx(1.0)
    printed = capsys.readouterr().out
    assert "required selection N_eff" in printed


def test_report_rejects_a_ragged_layout(tmp_path):
    """nobs*nsamp must match p_pe: the layout is strictly rectangular and a
    mismatch means the file was written against a different contract."""
    path = tmp_path / "bad.h5"
    _write_pe(path, np.full(100, 0.5), 4, 32)   # 100 != 4*32
    with pytest.raises(SystemExit, match="strictly rectangular"):
        pwd.report(path, 1.0)


def test_report_rejects_a_file_without_p_pe(tmp_path):
    path = tmp_path / "nope.h5"
    with h5py.File(path, "w") as f:
        f.attrs["nobs"] = 1
        f.attrs["nsamp"] = 4
        f.create_dataset("dL", data=np.ones(4))
    with pytest.raises(SystemExit, match="no 'p_pe' dataset"):
        pwd.report(path, 1.0)


def test_cli_rejects_a_non_positive_budget(tmp_path):
    path = tmp_path / "pe.h5"
    _write_pe(path, np.full((2, 8), 0.5), 2, 8)
    with pytest.raises(SystemExit, match="must be positive"):
        pwd.main(["--gw_path", str(path), "--max_likelihood_variance", "0"])


def test_cli_writes_json(tmp_path):
    path = tmp_path / "pe.h5"
    _write_pe(path, np.full((2, 8), 0.5), 2, 8)
    out_json = tmp_path / "report.json"
    assert pwd.main(["--gw_path", str(path), "--json", str(out_json)]) == 0
    import json
    payload = json.loads(out_json.read_text())
    assert payload["n_events"] == 2
    assert "pe_variance_sum" in payload and "threshold" in payload
