"""The mock's PE samples must be the flat-prior posterior of a real measurement.

`p_pe = 1` declares the samples were drawn under a prior flat in the sampled
variables, so they have to be posterior draws conditioned on an actual noisy
observation.  The historical construction centred them on the TRUE parameters
instead, which is not the posterior of any measurement and biases the recovered
distance scale at O(sigma^2).

These tests pin the corrected construction against its analytic form, and pin the
legacy path so mocks generated before the fix remain reproducible.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mock_dark_sirens"))

import generate_mock_data as gmd  # noqa: E402


def _truth(nobs, rng, dl=1500.0, z=0.3):
    return {
        "z": np.full(nobs, z),
        "dl": np.full(nobs, dl),
        "ra": np.full(nobs, 1.0),
        "dec": np.full(nobs, 0.2),
        "m1": np.full(nobs, 35.0),
        "m2": np.full(nobs, 30.0),
        "chi": np.zeros(nobs),
        "snr": np.full(nobs, 15.0),
    }


def test_distance_samples_match_analytic_flat_prior_posterior():
    """ln dL | d_obs ~ N(ln d_obs + s^2, s^2), the flat-in-dL volume factor included."""
    s = 0.12
    nsamp = 400_000
    rng = np.random.default_rng(20260730)
    truth = _truth(1, rng)
    post, obs = gmd._posterior_samples(
        rng, truth, nsamp, dL_fractional_uncertainty=s,
        sky_uncertainty_deg=1.0, pe_centering="observed")

    lnd = np.log(post["dL"])
    d_obs = obs["obs_dL"][0]
    # Mean of ln dL sits a FULL s^2 above ln d_obs (not -s^2/2, and not at truth).
    assert np.isclose(lnd.mean() - np.log(d_obs), s**2, atol=6.0 * s / np.sqrt(nsamp))
    assert np.isclose(lnd.std(), s, rtol=0.02)

    # The samples' density must be proportional to the flat-prior posterior
    # exp(-(ln dL - ln d_obs)^2 / 2 s^2).  Compare a histogram of dL against that
    # shape, normalised over the same bins.
    lo, hi = np.exp(np.log(d_obs) - 3 * s), np.exp(np.log(d_obs) + 3 * s)
    edges = np.linspace(lo, hi, 41)
    counts, _ = np.histogram(post["dL"], bins=edges)
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)
    shape = np.exp(-0.5 * ((np.log(centres) - np.log(d_obs)) / s) ** 2)
    expected = counts.sum() * shape * width / np.sum(shape * width)
    resid = (counts - expected) / np.sqrt(np.maximum(expected, 1.0))
    assert np.abs(resid).max() < 6.0, f"max |residual| = {np.abs(resid).max():.2f} sigma"


def test_observation_is_noisy_not_the_truth():
    """Each event conditions on its own measurement, so observations scatter about truth."""
    s = 0.15
    nobs = 4000
    rng = np.random.default_rng(7)
    truth = _truth(nobs, rng)
    _, obs = gmd._posterior_samples(
        rng, truth, 2, dL_fractional_uncertainty=s, sky_uncertainty_deg=1.0,
        pe_centering="observed")
    ln_ratio = np.log(obs["obs_dL"] / truth["dl"])
    assert np.isclose(ln_ratio.std(), s, rtol=0.05)
    assert abs(ln_ratio.mean()) < 5.0 * s / np.sqrt(nobs)
    # Masses and spin also condition on measurements rather than truth.
    assert not np.allclose(obs["obs_chieff"], truth["chi"])


def test_truth_centering_reproduces_the_historical_draws_bit_for_bit():
    """The legacy flag must reproduce pre-fix mocks exactly, draw order included."""
    s = 0.1
    nsamp, nobs = 64, 3
    truth = _truth(nobs, np.random.default_rng(0))

    post, obs = gmd._posterior_samples(
        np.random.default_rng(1234), truth, nsamp,
        dL_fractional_uncertainty=s, sky_uncertainty_deg=2.0,
        pe_centering="truth")

    # Replay the historical sequence independently.
    rng = np.random.default_rng(1234)
    sigma_ang = np.deg2rad(2.0)
    exp_dl, exp_m1 = [], []
    for i in range(nobs):
        exp_dl.append(rng.lognormal(np.log(truth["dl"][i]) - 0.5 * s**2, s, nsamp))
        rng.normal(0.0, sigma_ang / max(np.cos(truth["dec"][i]), 0.1), nsamp)  # dra
        rng.normal(0.0, sigma_ang, nsamp)                                      # ddec
        m1det = truth["m1"][i] * (1.0 + truth["z"][i])
        m2det = truth["m2"][i] * (1.0 + truth["z"][i])
        exp_m1.append(np.clip(rng.normal(m1det, 0.08 * m1det, nsamp), 2.0, None))
        rng.normal(m2det, 0.10 * m2det, nsamp)                                 # m2
        rng.normal(truth["chi"][i], 0.08, nsamp)                               # chi
    assert np.array_equal(post["dL"], np.concatenate(exp_dl))
    assert np.array_equal(post["m1det"], np.concatenate(exp_m1))
    # Legacy mode conditions on nothing, so the recorded observations are the truths.
    assert np.array_equal(obs["obs_dL"], truth["dl"])


def test_legacy_centring_biases_the_distance_scale_and_the_fix_removes_it():
    """The estimator-facing consequence, at the level the bias actually appears.

    With samples used as flat-prior posterior draws, the quantity the likelihood
    effectively leans on is the posterior mean of 1/dL (a distance scale, hence an
    H0 scale).  Averaged over many events its fractional offset from 1/d_true must
    vanish for the corrected construction and must not for the legacy one, growing
    as sigma^2.
    """
    nobs, nsamp = 6000, 400
    results = {}
    for mode in ("truth", "observed"):
        for s in (0.05, 0.15):
            rng = np.random.default_rng(99)
            truth = _truth(nobs, rng)
            post, _ = gmd._posterior_samples(
                rng, truth, nsamp, dL_fractional_uncertainty=s,
                sky_uncertainty_deg=1.0, pe_centering=mode)
            inv = (1.0 / post["dL"].reshape(nobs, nsamp)).mean(axis=1)
            results[(mode, s)] = float(np.mean(inv) * truth["dl"][0] - 1.0)

    # Legacy: a positive O(s^2) offset in inverse distance (=> H0 pulled high in the
    # per-event term), growing with s.
    assert results[("truth", 0.15)] > results[("truth", 0.05)] > 0.0
    ratio = results[("truth", 0.15)] / results[("truth", 0.05)]
    assert 4.0 < ratio < 16.0, f"legacy offset ratio {ratio:.2f} not ~s^2 (9)"

    # Corrected: consistent with zero, and far smaller than the legacy offset at the
    # same sigma.
    for s in (0.05, 0.15):
        assert abs(results[("observed", s)]) < 0.35 * abs(results[("truth", s)]), (
            f"s={s}: corrected {results[('observed', s)]:+.5f} vs "
            f"legacy {results[('truth', s)]:+.5f}")


def test_rejects_unknown_centering():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="pe_centering"):
        gmd._posterior_samples(rng, _truth(1, rng), 4, pe_centering="nonsense")
