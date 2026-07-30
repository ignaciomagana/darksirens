"""``--detection-data``: the SNR threshold must act on the data the PE conditions on.

Under the historical rule detection is decided by a ``Beta(2,5)**0.5`` projection
latent that never enters the data, so the detected-set likelihood carries an
extra ``P(det|theta)`` inside each event's integral -- a factor no population
code evaluates.  ``detection_data="observed"`` draws one measurement, thresholds
its SNR and hands the same measurement to the posterior, which restores the
premise that ``1[det(d)] = 1`` on the detected set.

These tests pin (a) bit-level preservation of the historical path, (b) that the
observed path really does share one draw between the threshold and the
posterior, and (c) that the injection campaign applies the same rule while still
storing TRUE parameters.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_GMD = (Path(__file__).resolve().parents[1]
        / "scripts" / "mock_dark_sirens" / "generate_mock_data.py")
_spec = importlib.util.spec_from_file_location("generate_mock_data", _GMD)
gmd = importlib.util.module_from_spec(_spec)
sys.modules["generate_mock_data"] = gmd
_spec.loader.exec_module(gmd)

H0, OM0, ZMAX, THRESH = 67.74, 0.3075, 2.0, 8.0


@pytest.fixture(scope="module")
def setup():
    cosmo = gmd._build_cosmology(H0, OM0, -1.0, 0.0)
    grids = gmd._cosmology_grids(cosmo, ZMAX)
    pop = gmd.PopulationConfig(gamma=0.0)
    rng = np.random.default_rng(11)
    cat = gmd._generate_complete_catalog(rng, 60_000, grids, gmd.SurveyConfig())
    return grids, pop, cat


def _events(setup, seed, **kw):
    grids, pop, cat = setup
    return gmd._draw_events_until_detected(
        np.random.default_rng(seed), 40, cat, grids, pop, THRESH, **kw)


# --------------------------------------------------------------------------
# (a) the historical path is untouched
# --------------------------------------------------------------------------
def test_default_is_the_historical_rule(setup):
    a = _events(setup, 7)
    b = _events(setup, 7, detection_data="true")
    assert set(a) == set(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_true_mode_records_no_observation(setup):
    truth = _events(setup, 7, detection_data="true")
    assert not [k for k in truth if k.startswith("obs_")]


def test_unknown_mode_raises(setup):
    with pytest.raises(ValueError, match="detection_data"):
        _events(setup, 7, detection_data="noisy")


# --------------------------------------------------------------------------
# (b) the observed path shares ONE draw between threshold and posterior
# --------------------------------------------------------------------------
def _obs_events(setup, seed=3, sigma=0.10, snr_ref=6.278):
    return _events(setup, seed, detection_data="observed", snr_ref=snr_ref,
                   pe_kwargs={"dL_fractional_uncertainty": sigma})


def test_observed_mode_records_the_measurement(setup):
    truth = _obs_events(setup)
    for k in ("obs_dL", "obs_m1det", "obs_m2det", "obs_chieff", "obs_ra",
              "obs_dec", "obs_sigma_dl", "obs_sigma_ang", "obs_snr"):
        assert k in truth and truth[k].shape == truth["z"].shape


def test_every_detected_event_passes_the_threshold_on_its_own_observation(setup):
    """The defining property: detection is a function of the recorded data."""
    truth = _obs_events(setup)
    rho = gmd._snr_from_detector_frame(truth["obs_m1det"], truth["obs_m2det"],
                                       truth["obs_dL"], 6.278)
    assert np.all(rho >= THRESH)
    np.testing.assert_allclose(rho, truth["obs_snr"], rtol=1e-12)


def test_observation_is_not_the_truth(setup):
    truth = _obs_events(setup)
    assert not np.allclose(truth["obs_dL"], truth["dl"])
    # multiplicative noise of the requested width, and detection has skewed it low
    lr = np.log(truth["obs_dL"] / truth["dl"])
    assert 0.02 < lr.std() < 0.30
    assert lr.mean() < 0.0


def test_posterior_conditions_on_the_recorded_observation(setup):
    truth = _obs_events(setup)
    post, obs = gmd._posterior_samples(
        np.random.default_rng(5), truth, 40_000,
        dL_fractional_uncertainty=0.10, pe_centering="observed",
        use_recorded_observation=True)
    np.testing.assert_allclose(obs["obs_dL"], truth["obs_dL"], rtol=0)
    n = truth["z"].size
    dl = post["dL"].reshape(n, -1)
    s = 0.10
    # flat-prior posterior of a lognormal measurement: ln dL ~ N(ln d_obs + s^2, s)
    got = np.log(dl).mean(axis=1)
    want = np.log(truth["obs_dL"]) + s * s
    np.testing.assert_allclose(got, want, atol=6.0 * s / np.sqrt(40_000))
    # and it is NOT centred on the truth
    assert np.abs(got - np.log(truth["dl"])).mean() > 2.0 * s / np.sqrt(40_000)


def test_recorded_observation_requires_the_keys(setup):
    truth = _events(setup, 7, detection_data="true")
    with pytest.raises(ValueError, match="missing"):
        gmd._posterior_samples(np.random.default_rng(0), truth, 8,
                               dL_fractional_uncertainty=0.10,
                               pe_centering="observed",
                               use_recorded_observation=True)


def test_recorded_observation_rejects_truth_centering(setup):
    truth = _obs_events(setup)
    with pytest.raises(ValueError, match="pe_centering"):
        gmd._posterior_samples(np.random.default_rng(0), truth, 8,
                               dL_fractional_uncertainty=0.10,
                               pe_centering="truth",
                               use_recorded_observation=True)


def test_dropping_the_projection_raises_the_detected_fraction(setup):
    """Pins the horizon change the mode implies, so --snr-ref recalibration is
    a documented consequence rather than a surprise."""
    grids, pop, cat = setup
    rng = np.random.default_rng(21)
    n = 200_000
    idx = rng.integers(0, cat["z"].size, n)
    z, dl = cat["z"][idx], gmd._interp_dl(cat["z"][idx], grids)
    m1, up = gmd._sample_powerlaw_peak_m1(rng, n, pop, return_component=True)
    m2 = gmd._sample_q(rng, m1, pop, use_peak=up) * m1
    f_true = float((gmd._network_snr(m1, m2, z, dl, rng) >= THRESH).mean())
    f_obs = float((gmd._snr_from_detector_frame(m1 * (1 + z), m2 * (1 + z), dl)
                   >= THRESH).mean())
    assert f_obs > 3.0 * f_true


def test_write_mock_data_end_to_end_observed(tmp_path, monkeypatch):
    """The full CLI path in observed mode: the events file must be writable
    (truth carries the recorded measurement AND the posterior echoes it -- the
    obs_* datasets appear exactly once), and every stored event must pass the
    threshold on its own recorded observation."""
    import h5py

    monkeypatch.setattr(sys, "argv", [
        "generate_mock_data.py", "--outdir", str(tmp_path), "--seed", "5",
        "--n-galaxies", "4000", "--nobs", "3", "--nsamp", "16",
        "--nselection", "4000", "--nside", "8",
        "--detection-data", "observed", "--snr-ref", "6.278"])
    gmd.write_mock_data(gmd.parse_args())

    with h5py.File(tmp_path / "mock_gw_events.h5") as f:
        assert f.attrs["detection_data"] == "observed"
        assert float(f.attrs["snr_ref"]) == 6.278
        tg = f["truth"]
        for k in ("obs_dL", "obs_m1det", "obs_m2det", "obs_chieff", "obs_ra",
                  "obs_dec", "obs_sigma_dl", "obs_snr"):
            assert k in tg, k
        rho = gmd._snr_from_detector_frame(
            tg["obs_m1det"][...], tg["obs_m2det"][...], tg["obs_dL"][...], 6.278)
        np.testing.assert_allclose(rho, tg["obs_snr"][...], rtol=1e-12)
        assert np.all(rho >= 8.0)

    with h5py.File(tmp_path / "mock_gw_selection.h5") as f:
        assert f.attrs["detection_data"] == "observed"


# --------------------------------------------------------------------------
# (c) the injection campaign applies the same rule, on TRUE coordinates
# --------------------------------------------------------------------------
def test_injections_store_true_parameters_under_observed_detection(setup):
    grids, pop, _ = setup
    sel = gmd._draw_selection_batch(
        np.random.default_rng(4), 200_000, grids, pop, THRESH,
        proposal="population", detection_data="observed", snr_ref=6.278,
        pe_kwargs={"dL_fractional_uncertainty": 0.10})
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


def test_injection_numpy_and_jax_paths_agree_on_the_observed_rule(setup):
    """The JAX kernel mirrors the numpy reference; they cannot be bit-identical
    (different PRNGs) so this pins the detected fraction to Poisson agreement."""
    grids, pop, _ = setup
    kw = dict(detection_data="observed", snr_ref=6.278,
              pe_kwargs={"dL_fractional_uncertainty": 0.10})
    n = 400_000
    ref = gmd._draw_selection_batch(np.random.default_rng(9), n, grids, pop,
                                    THRESH, proposal="population", **kw)
    jx = gmd._selection_injections(np.random.default_rng(9), n, grids, pop,
                                   THRESH, n, verbose=False,
                                   proposal="population", **kw)
    a, b = ref["n_detected"], jx["n_detected"]
    assert a > 100 and b > 100
    assert abs(a - b) < 5.0 * np.sqrt(a + b), (a, b)


def test_injection_true_mode_unchanged(setup):
    grids, pop, _ = setup
    a = gmd._draw_selection_batch(np.random.default_rng(2), 50_000, grids, pop,
                                  THRESH, proposal="population")
    b = gmd._draw_selection_batch(np.random.default_rng(2), 50_000, grids, pop,
                                  THRESH, proposal="population",
                                  detection_data="true")
    assert a["n_detected"] == b["n_detected"]
    np.testing.assert_array_equal(a["dL"], b["dL"])
    np.testing.assert_array_equal(a["pdraw"], b["pdraw"])
