"""Detection must be a deterministic function of the data the PE conditions on.

A population likelihood evaluates
``prod_i [int p(d_i|theta) p(theta|Lambda) dtheta] / mu^N``, which is the correct
detected-set likelihood only when ``1[det(d_i)] = 1`` for every observed event,
i.e. only when detection is a function of the recorded data.  Thresholding a
latent amplitude instead leaves an extra ``P(det|theta)`` INSIDE each event's
integral, so such a mock is not drawn from the model the inference assumes.

The generator therefore records ``rho_obs = rho_opt(theta) + N(0, sigma_rho)``,
thresholds that one number, and hands the same measurement to the posterior.
These tests pin (a) the rule and its two Malmquist fingerprints, (b) the exact
closed-form selection function the rule implies, (c) that the injection campaign
applies the same rule while still storing TRUE parameters, and (d) the end-to-end
CLI contract.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats
from scipy.special import ndtr

_GMD = (Path(__file__).resolve().parents[1]
        / "scripts" / "mock_dark_sirens" / "generate_mock_data.py")
_spec = importlib.util.spec_from_file_location("generate_mock_data", _GMD)
gmd = importlib.util.module_from_spec(_spec)
sys.modules["generate_mock_data"] = gmd
_spec.loader.exec_module(gmd)

H0, OM0, ZMAX, THRESH = 67.74, 0.3075, 2.0, 8.0
MEAS = gmd.MeasurementConfig(snr_ref=6.278, snr_threshold=THRESH)


@pytest.fixture(scope="module")
def setup():
    cosmo = gmd._build_cosmology(H0, OM0, -1.0, 0.0)
    grids = gmd._cosmology_grids(cosmo, ZMAX)
    pop = gmd.PopulationConfig(gamma=0.0)
    rng = np.random.default_rng(11)
    cat = gmd._generate_complete_catalog(rng, 60_000, grids, gmd.SurveyConfig())
    return grids, pop, cat


def _events(setup, seed=3, nobs=40, **kw):
    grids, pop, cat = setup
    kw.setdefault("meas", MEAS)
    return gmd._draw_events_until_detected(
        np.random.default_rng(seed), nobs, cat, grids, pop, THRESH, **kw)


# --------------------------------------------------------------------------
# (a) the rule, and the two Malmquist observables it implies
# --------------------------------------------------------------------------
def test_the_measurement_is_recorded_in_the_measurement_basis(setup):
    truth, _ = _events(setup)
    for k in ("obs_rho", "obs_snr", "snr_true", "obs_lnmc", "obs_lnq", "obs_chieff",
              "obs_ra", "obs_dec", "obs_sigma_rho", "obs_sig_lnmc", "obs_sig_lnq",
              "obs_sig_chieff", "obs_sigma_ang", "obs_sig_ra",
              "obs_m1det", "obs_m2det", "obs_dL"):
        assert k in truth and truth[k].shape == truth["z"].shape, k
    # obs_snr is an alias of the recorded detection statistic, not a second one.
    np.testing.assert_array_equal(truth["obs_snr"], truth["obs_rho"])


def test_every_detected_event_passes_the_threshold_on_its_own_recorded_number(setup):
    """The defining property, and it reads ONE stored number."""
    truth, _ = _events(setup)
    assert np.all(truth["obs_rho"] >= THRESH)


def test_the_true_amplitude_is_recomputable_from_the_truth(setup):
    truth, _ = _events(setup)
    want = gmd._snr_from_detector_frame(truth["m1"] * (1.0 + truth["z"]),
                                        truth["m2"] * (1.0 + truth["z"]),
                                        truth["dl"], MEAS.snr_ref)
    np.testing.assert_allclose(truth["snr_true"], want, rtol=1e-12)


def test_observation_is_not_the_truth(setup):
    truth, _ = _events(setup)
    assert not np.allclose(truth["obs_rho"], truth["snr_true"])
    # detection has skewed the recorded amplitude high relative to the truth
    assert (truth["obs_rho"] - truth["snr_true"]).mean() > 0.0


def test_the_malmquist_fingerprints_are_both_present(setup):
    """Both are EXACTLY zero for a true-parameter cut and strictly positive for a
    data-space cut, so this is the cheapest proof of which rule a file used."""
    truth, rejected = _events(setup, nobs=60, rejected_keep=4000)
    assert rejected["obs_rho"].size > 100
    # every stored rejection fails the threshold on its OWN recorded number
    assert np.all(rejected["obs_rho"] < THRESH)
    # detections whose TRUE amplitude is sub-threshold: impossible under a
    # true-parameter cut, routine here
    assert float((truth["snr_true"] < THRESH).mean()) > 0.0

    # The mirror image, measured where the threshold actually bites.  Sampled
    # from the event loop it is a rare tail (almost every host of a deep catalog
    # sits far below threshold), so spread sources across rho_opt in [6, 10]
    # instead and read both fractions off the same draw.
    rng = np.random.default_rng(41)
    n = 200_000
    m1det, m2det = np.full(n, 35.0), np.full(n, 30.0)
    mc = gmd._mc_of_m1q(m1det, m2det / m1det)
    dl = gmd._dl_of_mc_rho(mc, rng.uniform(6.0, 10.0, n), MEAS.snr_ref)
    obs = gmd._measure(rng, m1det, m2det, np.zeros(n), dl, np.zeros(n), np.zeros(n),
                       MEAS, need_sky=False)
    det = obs["obs_rho"] >= THRESH
    assert float((obs["snr_true"][det] < THRESH).mean()) > 0.0
    assert float((obs["snr_true"][~det] > THRESH).mean()) > 0.0


def test_the_posterior_conditions_on_the_recorded_measurement(setup):
    truth, _ = _events(setup)
    post, obs = gmd._posterior_samples(np.random.default_rng(5), truth, 4000, MEAS)
    for key, val in obs.items():
        np.testing.assert_array_equal(val, truth[key])
    n = truth["z"].size
    # the ln Mc channel is an untruncated normal about the RECORDED value with
    # the RECORDED width -- no shift, no skew
    mc = gmd._mc_of_m1q(post["m1det"], post["m2det"] / post["m1det"]).reshape(n, -1)
    got = np.log(mc).mean(axis=1)
    err = truth["obs_sig_lnmc"] / np.sqrt(4000)
    assert np.all(np.abs(got - truth["obs_lnmc"]) < 6.0 * err)
    # and it is NOT centred on the truth
    lnmc_true = np.log(gmd._mc_of_m1q(truth["m1"] * (1.0 + truth["z"]),
                                      truth["m2"] / truth["m1"]))
    assert np.abs(got - lnmc_true).mean() > 2.0 * err.mean()


# --------------------------------------------------------------------------
# (b) the closed-form selection function the rule implies
# --------------------------------------------------------------------------
def test_pdet_matches_the_exact_gaussian_cdf_oracle():
    """``P_det(theta) = Phi((rho_opt(theta) - rho_th)/sigma_rho)`` exactly, because
    the threshold acts on a single additively-noisy number.  Compared against a
    brute-force Monte Carlo of the generator's OWN measure/detect code across
    ``P_det`` from 0.003 to 0.999."""
    n_mc = 200_000
    rho_opt_grid = np.linspace(THRESH - 2.8, THRESH + 3.1, 30)
    # A source is placed at the distance that realises each target rho_opt.
    m1det = np.full(n_mc, 35.0)
    m2det = np.full(n_mc, 30.0)
    mc = gmd._mc_of_m1q(m1det, m2det / m1det)
    rng = np.random.default_rng(31)
    exact, empirical = [], []
    for rho_opt in rho_opt_grid:
        dl = gmd._dl_of_mc_rho(mc, rho_opt, MEAS.snr_ref)
        obs = gmd._measure(rng, m1det, m2det, np.zeros(n_mc), dl,
                           np.zeros(n_mc), np.zeros(n_mc), MEAS, need_sky=False)
        empirical.append(float((obs["obs_rho"] >= THRESH).mean()))
        exact.append(float(ndtr((rho_opt - THRESH) / MEAS.sigma_rho)))
    exact = np.asarray(exact)
    empirical = np.asarray(empirical)
    assert exact.min() < 0.005 and exact.max() > 0.99
    mc_err = np.sqrt(np.maximum(exact * (1.0 - exact), 1e-12) / n_mc)
    pull = (empirical - exact) / mc_err
    assert np.abs(pull).max() < 5.0, np.abs(pull).max()
    assert abs(pull.mean()) < 3.0 / np.sqrt(pull.size)


def test_the_widths_carry_the_detection_statistic(setup):
    """Because every width is ``a/rho_obs``, conditioning on any stored width
    already conditions on the detection statistic -- there is no latent left
    inside the detection decision."""
    truth, _ = _events(setup)
    recovered = MEAS.a_mc * MEAS.snr_threshold / truth["obs_sig_lnmc"]
    np.testing.assert_allclose(recovered, truth["obs_rho"], rtol=1e-12)


# --------------------------------------------------------------------------
# (c) generative replay: the recorded file is a draw from the stated model
# --------------------------------------------------------------------------
def test_generative_replay_reproduces_the_detected_distributions(setup):
    """Fresh proposals through the same measure/detect code must have the same
    detected ``rho`` pull and host redshift as the stored file.  Catches any
    silent change of draw order or width law."""
    truth, _ = _events(setup, seed=3, nobs=600)
    replay, _ = _events(setup, seed=90210, nobs=600)
    pull_a = (truth["obs_rho"] - truth["snr_true"]) / MEAS.sigma_rho
    pull_b = (replay["obs_rho"] - replay["snr_true"]) / MEAS.sigma_rho
    assert stats.ks_2samp(pull_a, pull_b).pvalue > 1e-3
    assert stats.ks_2samp(truth["z"], replay["z"]).pvalue > 1e-3


# --------------------------------------------------------------------------
# (d) the injection campaign applies the same rule, on TRUE coordinates
# --------------------------------------------------------------------------
def test_injections_store_true_parameters(setup):
    grids, pop, _ = setup
    sel = gmd._draw_selection_batch(
        np.random.default_rng(4), 200_000, grids, pop, THRESH,
        proposal="population", meas=MEAS)
    assert sel["n_detected"] > 0
    # mu(theta) integrates over TRUE parameters: the stored coordinates must be
    # exactly consistent, i.e. m1det = (1+z) m1src with z from the stored dL.
    z = np.interp(sel["dL"], grids["dl"], grids["z"])
    np.testing.assert_allclose(sel["m1det"], sel["m1src"] * (1.0 + z), rtol=1e-6)
    # and pdraw is the untouched proposal density of those true parameters
    q = sel["m2src"] / sel["m1src"]
    np.testing.assert_allclose(
        sel["pdraw"],
        gmd._selection_pdraw("population", sel["m1src"], q, sel["chieff"], z,
                             grids, pop),
        rtol=1e-6)


def test_injection_numpy_and_jax_paths_agree_on_the_rule(setup):
    """The JAX kernel mirrors the numpy reference; they cannot be bit-identical
    (different PRNGs) so this pins the detected fraction to Poisson agreement."""
    grids, pop, _ = setup
    n = 400_000
    ref = gmd._draw_selection_batch(np.random.default_rng(9), n, grids, pop,
                                    THRESH, proposal="population", meas=MEAS)
    jx = gmd._selection_injections(np.random.default_rng(9), n, grids, pop,
                                   THRESH, n, verbose=False,
                                   proposal="population", meas=MEAS)
    a, b = ref["n_detected"], jx["n_detected"]
    assert a > 100 and b > 100
    assert abs(a - b) < 5.0 * np.sqrt(a + b), (a, b)


def test_injection_detected_fraction_matches_the_pdet_oracle(setup):
    """The injection campaign's own detected fraction must equal the average of
    the closed-form ``P_det`` over the proposal -- the statement that mu(theta)
    is the probability of passing the same rule the events passed."""
    grids, pop, _ = setup
    n = 300_000
    rng = np.random.default_rng(17)
    sel = gmd._draw_selection_batch(rng, n, grids, pop, THRESH,
                                    proposal="population", meas=MEAS)
    # Redraw the same proposal without the detection noise and average the oracle.
    rng2 = np.random.default_rng(17)
    z = gmd._sample_uniform_comoving_z(rng2, grids, n)
    _ra, _dec = gmd._sample_sky(rng2, n)
    dl = gmd._interp_dl(z, grids)
    m1src, use_peak = gmd._sample_powerlaw_peak_m1(rng2, n, pop, return_component=True)
    q = gmd._sample_q(rng2, m1src, pop, use_peak=use_peak)
    rho_opt = gmd._snr_from_detector_frame(m1src * (1.0 + z), q * m1src * (1.0 + z),
                                           dl, MEAS.snr_ref)
    expected = float(ndtr((rho_opt - THRESH) / MEAS.sigma_rho).mean())
    got = sel["n_detected"] / n
    assert abs(got - expected) < 5.0 * np.sqrt(expected * (1.0 - expected) / n), (got, expected)


def test_selection_kernel_snr_override_replaces_the_statistic(setup):
    """The bright-siren mock keeps a projection-latent rule and overrides the
    statistic; the override must REPLACE it, not compose with it."""
    import jax.numpy as jnp
    grids, pop, _ = setup
    kernel = gmd._make_selection_kernel(grids, pop, THRESH, "population",
                                        meas=MEAS,
                                        snr_fn=lambda key, state: jnp.asarray(0.0))
    out = gmd._run_selection_chunks(np.random.default_rng(1), 5000, grids, pop,
                                    "population", 5000, kernel)
    assert out["n_detected"] == 0


# --------------------------------------------------------------------------
# (e) the CLI contract
# --------------------------------------------------------------------------
def test_write_mock_data_end_to_end(tmp_path, monkeypatch):
    """The full CLI path: the truth group carries the recorded measurement in
    the measurement basis, the posterior echoes it (the obs_* datasets appear
    exactly once), p_pe is not the all-ones column, and the metadata records
    every constant of the measurement family."""
    import h5py
    import json

    monkeypatch.setattr(sys, "argv", [
        "generate_mock_data.py", "--outdir", str(tmp_path), "--seed", "5",
        "--n-galaxies", "4000", "--nobs", "3", "--nsamp", "16",
        "--nselection", "4000", "--nside", "8", "--snr-ref", "6.278",
        "--record-rejected", "50"])
    gmd.write_mock_data(gmd.parse_args())

    with h5py.File(tmp_path / "mock_gw_events.h5") as f:
        assert f.attrs["measurement_family"] == gmd.MEASUREMENT_FAMILY
        assert float(f.attrs["snr_ref"]) == 6.278
        assert float(f.attrs["sigma_rho"]) == gmd.SIGMA_RHO_DEFAULT
        tg = f["truth"]
        for k in ("obs_rho", "snr_true", "obs_lnmc", "obs_lnq", "obs_chieff",
                  "obs_ra", "obs_dec", "obs_sig_lnmc", "obs_sig_lnq",
                  "obs_sig_chieff", "obs_sigma_ang", "obs_sig_ra",
                  "obs_sigma_rho", "obs_snr"):
            assert k in tg, k
        rho = tg["obs_rho"][...]
        assert np.all(rho >= 8.0)
        # every width recomputable from the stored rho_obs, bitwise
        k = 8.0 / rho
        assert np.array_equal(tg["obs_sig_lnmc"][...], gmd.A_MC_DEFAULT * k)
        assert np.array_equal(tg["obs_sig_lnq"][...], gmd.A_Q_DEFAULT * k)
        assert not np.allclose(f["p_pe"][...], 1.0)

        meta = json.loads(f.attrs["metadata_json"])["measurement"]
        assert meta["family"] == gmd.MEASUREMENT_FAMILY
        for key in ("snr_ref", "snr_threshold", "sigma_rho", "a_mc", "a_q",
                    "a_chi", "sky_a_deg", "snr_ref_sigma", "cos_dec_floor"):
            assert key in meta, key
        assert "rejected" not in f or np.all(f["rejected/obs_rho"][...] < 8.0)

    with h5py.File(tmp_path / "mock_gw_selection.h5") as f:
        assert f.attrs["measurement_family"] == gmd.MEASUREMENT_FAMILY


@pytest.mark.parametrize("flag", ["--detection-data", "--pe-centering",
                                  "--dL-fractional-uncertainty",
                                  "--m1det-fractional-uncertainty",
                                  "--m2det-fractional-uncertainty"])
def test_removed_flags_fail_with_a_pointer(flag, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["generate_mock_data.py", flag, "0.1"])
    with pytest.raises(SystemExit, match="all-observable measurement family"):
        gmd.parse_args()


def test_threshold_and_measurement_config_must_agree(setup):
    grids, pop, cat = setup
    with pytest.raises(ValueError, match="snr_threshold"):
        gmd._draw_events_until_detected(np.random.default_rng(0), 2, cat, grids,
                                        pop, 9.0, meas=MEAS)
