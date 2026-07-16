"""Matched (fiducial-shaped + detectability-tilted) unlensed injection
campaign: unbiasedness against the broad campaign, the Neff improvement it
exists for, mixture-density support, and event-stream invariance."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from darksirens.gw.populations.registry import get_fixed_population_params
from scripts.mock_lensing import generate_mock_lensing as gen

H0, OM0 = 67.74, 0.3075


@pytest.fixture(scope="module")
def setup_pop():
    gen.set_pop_model("powerlaw+peak@md")
    theta = np.asarray(get_fixed_population_params("powerlaw+peak@md"))
    model = gen.SNRModel(rho_thr=8.0, horizon_Mpc=3000.0, mc_bar=1.22)
    return theta, model


def _truth_stats(d, n_draw, theta):
    """mu_hat and Neff at the truth for a campaign dict in the canonical basis."""
    m1s = d["m1src"]
    q = d["m2src"] / d["m1src"]
    chi = d["chieff"]
    z = d["m1det"] / d["m1src"] - 1.0
    ddL = np.asarray(gen.ddL_of_z(jnp.asarray(z), jnp.asarray(d["dL"]), H0, OM0))
    zg, pdf_astro, _ = gen._build_z_cdf(theta, H0, OM0)
    pz = np.interp(z, zg, pdf_astro)
    p_true = gen._mixture_density(m1s, q, chi, theta) * pz / ((1.0 + z) * ddL)
    w = np.where(np.isfinite(p_true) & (p_true > 0), p_true, 0.0) / d["pdraw"]
    mu = w.sum() / n_draw
    neff = w.sum() ** 2 / (w**2).sum()
    sigma_mu = np.sqrt((w**2).sum()) / n_draw
    return mu, neff, sigma_mu


def _campaign(proposal, theta, model, n=60_000, seed=7):
    rng = np.random.default_rng(seed)
    d, ndet = gen.generate_unlensed_injections(
        n, model, rng, H0, OM0, proposal=proposal, theta=theta
    )
    return d, ndet


def test_analytic_proposal_density_matches_draw_g(setup_pop):
    theta, _ = setup_pop
    rng = np.random.default_rng(3)
    m1, q, chi, g = gen._analytic_proposal(2000, theta, rng)
    g2 = gen._analytic_proposal_density(m1, q, chi, theta)
    assert np.allclose(g, g2, rtol=1e-12)
    assert np.all(g2 > 0)


def test_matched_mu_unbiased_vs_broad(setup_pop):
    theta, model = setup_pop
    n = 60_000
    d_b, _ = _campaign("broad", theta, model, n=n, seed=11)
    d_m, _ = _campaign("matched", theta, model, n=n, seed=12)
    mu_b, _, s_b = _truth_stats(d_b, n, theta)
    mu_m, _, s_m = _truth_stats(d_m, n, theta)
    combined = np.hypot(s_b, s_m)
    assert abs(mu_m - mu_b) < 4.0 * combined, (mu_b, mu_m, combined)


def test_matched_neff_improvement(setup_pop):
    theta, model = setup_pop
    n = 60_000
    d_b, _ = _campaign("broad", theta, model, n=n, seed=21)
    d_m, _ = _campaign("matched", theta, model, n=n, seed=21)
    _, neff_b, _ = _truth_stats(d_b, n, theta)
    _, neff_m, _ = _truth_stats(d_m, n, theta)
    assert neff_m > 5.0 * neff_b, (neff_b, neff_m)


def test_matched_pdraw_positive_on_detected(setup_pop):
    theta, model = setup_pop
    d, ndet = _campaign("matched", theta, model, n=20_000, seed=5)
    assert ndet > 0
    assert np.all(np.isfinite(d["pdraw"])) and np.all(d["pdraw"] > 0)


def test_matched_requires_theta(setup_pop):
    _, model = setup_pop
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="requires the truth theta"):
        gen.generate_unlensed_injections(
            100, model, rng, H0, OM0, proposal="matched", theta=None
        )


def test_selection_file_records_proposal(setup_pop, tmp_path):
    theta, model = setup_pop
    import h5py

    out = tmp_path / "sel.h5"
    rng = np.random.default_rng(2)
    gen.generate_unlensed_injections(
        20_000, model, rng, H0, OM0, proposal="matched", theta=theta,
        out_path=str(out),
    )
    with h5py.File(out) as f:
        assert f.attrs["injection_proposal"] == "matched"
        assert 0.0 < f.attrs["matched_fraction"] < 1.0
        assert f.attrs["ndraw"] == 20_000


def test_event_stream_invariant_under_proposal(tmp_path):
    """The observed catalog (events, truth partition) must be identical for
    broad and matched campaigns at the same seed: events draw from the rng
    before the injection step, and the proposal flag must not touch them."""
    gen.set_pop_model("powerlaw+peak@md")
    from darksirens.lensing.slmarks import make_sis_lens_params
    from darksirens.lensing.wlmagnification import make_lognormal_wl_params

    sis = make_sis_lens_params(A_tau=5e-4, n_tau=3.0)
    wl = make_lognormal_wl_params(a=4e-3, b=1.5)
    outs = {}
    for prop in ("broad", "matched"):
        out_dir = tmp_path / prop
        gen.assemble(
            str(out_dir), n_universe=3000, seed=424, nsamp=8,
            n_sing_keep=3, n_pair_keep=1, conditioning="fixed_counts",
            max_sing_keep=3, max_pair_keep=1, rho_thr=8.0,
            horizon_Mpc=3000.0, n_unlensed_inj=4000, n_lensed_inj=2000,
            H0=H0, Om0=OM0, sis=sis, wl=wl,
            injection_proposal=prop,
        )
        cat = json.loads((out_dir / "observed_catalog.json").read_text())
        outs[prop] = (cat["events"], cat["truth_partition"], cat["event_order"])
    assert outs["broad"] == outs["matched"]
