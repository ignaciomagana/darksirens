"""Gate (b) of the P_det emulator validation must not fail a correct emulator.

The identity is E_{p_inj}[xi q/p_inj] = xi * int_supp q = xi (1 - leak), and the
flow's out-of-support mass (~1%) is a FIXED RELATIVE offset while the MC error
shrinks as 1/sqrt(n_pinj).  Comparing against xi alone therefore makes the
verdict a function of the sample count: sharpening the test by raising --n_pinj
would fail a correct emulator.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("matplotlib")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "pdet_emulator_validation.py"


@pytest.fixture(scope="module")
def val():
    spec = importlib.util.spec_from_file_location("_pdet_validation_script", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeFlow:
    """log q = log p_inj + log(1 - leak) on the support: a PERFECT emulator whose
    only defect is the documented out-of-support mass."""

    def __init__(self, log_pinj_of, leak, noise=None):
        self._log_pinj_of = log_pinj_of
        self._leak = leak
        self._noise = 0.0 if noise is None else noise

    def log_prob(self, theta):
        return (np.asarray(self._log_pinj_of(theta)) + np.log1p(-self._leak)
                + self._noise)


def _patched(val, monkeypatch, leak, n_pinj, noise_scale=0.0):
    xi = 0.01
    z_grid = np.linspace(0.0, 2.0, 64)
    z_pdf = np.ones_like(z_grid)

    theta = np.zeros((n_pinj, len(val.sel.PDET_COLUMNS)))
    log_pinj = np.log(np.linspace(0.5, 1.5, n_pinj))     # any proper density
    monkeypatch.setattr(val, "sample_pinj", lambda n, zg, zp, seed=123: theta)
    monkeypatch.setattr(val, "log_pinj_full", lambda t, zg, zp: log_pinj)
    monkeypatch.setattr(val, "flow_log_prob_batched",
                        lambda flow, t, batch=0: flow.log_prob(t))
    noise = None
    if noise_scale:
        # A mean-one multiplicative jitter, so E[p_det] is unchanged but the MC
        # error is nonzero (a noiseless emulator has mc_sigma = 0 exactly).
        rng = np.random.default_rng(0)
        u = rng.normal(0.0, noise_scale, n_pinj)
        noise = u - 0.5 * noise_scale ** 2
    flow = _FakeFlow(lambda t: log_pinj, leak, noise)
    return flow, xi, z_grid, z_pdf


def test_leakage_corrected_gate_passes_where_the_raw_one_fails(val, monkeypatch):
    leak, n_pinj = 0.01, 400_000
    flow, xi, z_grid, z_pdf = _patched(val, monkeypatch, leak, n_pinj)
    monkeypatch.setattr(val, "flow_support_leakage",
                        lambda path, nsamp, seed=17: leak)

    res = val.gate_b_xi(flow, xi, z_grid, z_pdf, n_pinj, flow_path="fake.npz",
                        leak_nsamp=1_000_000)
    # The emulator is exact up to the leakage: the corrected estimate sits on top
    # of xi(1-leak) ...
    assert res["ok"]
    assert res["n_sigma"] < 1e-6
    np.testing.assert_allclose(res["target"], xi * (1.0 - leak), rtol=1e-12)
    # ... while the uncorrected comparison against xi is a many-sigma "failure"
    # whose size is set purely by the sample count.
    assert res["n_sigma_raw"] > 4.0


def test_without_a_leakage_estimate_the_gate_degrades_to_the_raw_test(val, monkeypatch):
    leak, n_pinj = 0.0, 20_000
    flow, xi, z_grid, z_pdf = _patched(val, monkeypatch, leak, n_pinj,
                                       noise_scale=0.05)
    res = val.gate_b_xi(flow, xi, z_grid, z_pdf, n_pinj, flow_path=None)
    assert res["leak"] is None
    assert res["target"] == xi
    assert res["n_sigma"] == res["n_sigma_raw"]
    assert res["ok"]
