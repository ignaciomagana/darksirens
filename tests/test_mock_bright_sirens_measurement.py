"""The bright-siren mock must be drawn from the model the inference assumes.

Two defects are pinned here, both measured on the sibling dark-siren campaign:

* the survey block declared a photo-z width while storing the TRUE redshift (and
  the counterpart file declared ``counterpart_dz`` while giving the true z), so
  the likelihood smoothed a comb that carried no error -- 7.6 sigma on a matched
  mock;
* every PE width was a function of LATENT quantities (``0.08 m1det_true``, a
  Beta(2,5)^0.5 projection amplitude) and detection thresholded that same latent
  amplitude -- the score identity E[C] = E[A] violated at 11.3 sigma, and
  -0.49 +- 0.08 km/s/Mpc in recovered H0 from the sky channel alone.

The generator now uses the dark generator's all-observable family, so every
stored width must be recomputable from the stored ``obs_rho``.
"""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("h5py")
pytest.importorskip("jax")
h5py = pytest.importorskip("h5py")

import jax as _jax  # noqa: E402

_jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
GEN_PATH = ROOT / "scripts" / "mock_bright_sirens" / "generate_mock_bright_sirens.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("mock_bright_gen_under_test",
                                                  GEN_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mock(gen, tmp_path_factory):
    out = tmp_path_factory.mktemp("bright")
    args = SimpleNamespace(
        outdir=str(out), seed=7, n0=2.0e-4, nobs=3, nsamp=64, ndraw=3000,
        nbatches=1, proposal="population", zmax=0.08, H0=67.74, Om0=0.3075,
        w0=-1.0, wa=0.0, snr_threshold=8.0,
        survey_z50=gen._dark.SurveyConfig.z50,
        survey_width=gen._dark.SurveyConfig.width,
        galaxy_density_delta=gen._dark.SurveyConfig.delta,
        selection_batch_size=None, selection_target_detections=None,
        selection_per_observation_factor=None,
        snr_ref=gen._dark.SNR_REF_DEFAULT,
        snr_uncertainty=gen._dark.SIGMA_RHO_DEFAULT,
        lnmc_uncertainty=gen._dark.A_MC_DEFAULT,
        lnq_uncertainty=gen._dark.A_Q_DEFAULT,
        chieff_uncertainty=gen._dark.A_CHI_DEFAULT,
        counterpart_dz=1.0e-4, verbose=False,
    )
    gen.write_mock_data(args)
    return out, args


def test_survey_block_stores_the_realised_photoz(gen, mock):
    out, args = mock
    with h5py.File(out / "mock_survey_raw.h5", "r") as f:
        z_survey = np.asarray(f["Z"])
        zerr_survey = np.asarray(f["ZERR"])
    with h5py.File(out / "mock_galaxy_catalog_complete.h5", "r") as f:
        z_true_all = np.asarray(f["z"])
        z_obs_all = np.asarray(f["z_obs"])

    survey = gen._dark.SurveyConfig(z50=args.survey_z50, width=args.survey_width,
                                    delta=args.galaxy_density_delta)
    # The declared width is evaluated at the RECORDED redshift, as in the dark
    # generator (dz = zerr(z_obs)), and the stored Z is a realised z_obs.
    np.testing.assert_allclose(
        zerr_survey, gen._dark._catalog_zerr(z_survey, survey), rtol=0, atol=0)
    assert np.isin(z_survey, z_obs_all).all()
    # And it is NOT the true redshift: the scatter is the declared width.
    assert not np.isin(z_survey, z_true_all).all()


def test_counterpart_redshift_realises_the_declared_error(gen, mock):
    out, args = mock
    items = json.loads((out / "bright_counterparts.json").read_text())["counterparts"]
    with h5py.File(out / "mock_bright_gw_events.h5", "r") as f:
        z_true = np.asarray(f["truth"]["z"])
    z_cp = np.array([c["z"] for c in items])
    dz = np.array([c["counterpart_dz"] for c in items])
    assert np.all(dz == args.counterpart_dz)
    assert not np.any(z_cp == z_true)                    # error actually drawn
    assert np.all(np.abs(z_cp - z_true) < 6.0 * dz)      # ...of the declared size


def test_every_pe_width_is_a_function_of_the_recorded_snr(gen, mock):
    """The all-observable property, checkable from the file alone."""
    out, args = mock
    meas = gen._dark.MeasurementConfig(
        snr_ref=args.snr_ref, snr_threshold=args.snr_threshold,
        sigma_rho=args.snr_uncertainty, a_mc=args.lnmc_uncertainty,
        a_q=args.lnq_uncertainty, a_chi=args.chieff_uncertainty,
    )
    with h5py.File(out / "mock_bright_gw_events.h5", "r") as f:
        truth = {k: np.asarray(v) for k, v in f["truth"].items()}
    want = gen._dark._measurement_widths(truth["obs_rho"], meas)
    for stored, key in (("obs_sig_lnmc", "sig_lnmc"), ("obs_sig_lnq", "sig_lnq"),
                        ("obs_sig_chieff", "sig_chieff"),
                        ("obs_sigma_ang", "sigma_ang")):
        np.testing.assert_array_equal(truth[stored], want[key])
    # Detection acted on the RECORDED number, not on a latent amplitude: every
    # stored event passes the threshold on its own obs_rho.
    assert np.all(truth["obs_rho"] >= args.snr_threshold)
    # ...and the true amplitude is allowed to sit below it (Malmquist scatter a
    # noise-free/latent-projection statistic cannot produce).
    assert np.all(np.isfinite(truth["snr_true"]))


def test_sky_samples_are_the_counterpart_direction(gen, mock):
    out, args = mock
    with h5py.File(out / "mock_bright_gw_events.h5", "r") as f:
        ra = np.asarray(f["ra"]).reshape(args.nobs, args.nsamp)
        dec = np.asarray(f["dec"]).reshape(args.nobs, args.nsamp)
        truth_ra = np.asarray(f["truth"]["ra"])
        truth_dec = np.asarray(f["truth"]["dec"])
    np.testing.assert_array_equal(ra, np.repeat(truth_ra[:, None], args.nsamp, 1))
    np.testing.assert_array_equal(dec, np.repeat(truth_dec[:, None], args.nsamp, 1))


def test_removed_truth_scaled_width_flags_are_refused(gen, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["gen", "--m1det-fractional-uncertainty", "0.08"])
    with pytest.raises(SystemExit, match="all-observable"):
        gen.parse_args()
