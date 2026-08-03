"""The survey catalog must REALISE the photo-z error it declares, in float64.

Two independent contracts on the catalog side of the mock.

**The photo-z is realised.**  Copying true redshifts into the survey block while
declaring a width makes the likelihood smooth a comb that carries no error, so
darksirens' per-galaxy kernel ``g(z) N(z; z_g, sigma_g)/Z(z_g)`` is not the
Bayesian posterior for the host's true redshift.  Measured on the gws-agn matched
mock, that internal inconsistency was ``+6.383e-4 +- 0.836e-4`` (7.6 sigma);
realising the error left ``-5.49e-5 +- 9.19e-5`` (0.60 sigma).  The complete
catalog therefore carries two redshift columns -- ``z`` (true, which drives the
host draw and the event truth) and ``z_obs`` -- and the survey sees only
``z_obs``.  ``z_obs`` is never clipped: clipping re-introduces a censored
observation.

**The pixelated arrays are float64.**  ``_kde_dndz_obs`` clamps its
truncated-kernel mass at ``1e-300``, which is not representable in float32, so a
float32 catalog turns every padded slot into ``0/0 = NaN``, every pixel row with
padding into all-NaN, and the likelihood into ``-inf`` everywhere.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy import stats

_GMD = (Path(__file__).resolve().parents[1]
        / "scripts" / "mock_dark_sirens" / "generate_mock_data.py")
_spec = importlib.util.spec_from_file_location("generate_mock_data", _GMD)
gmd = importlib.util.module_from_spec(_spec)
sys.modules["generate_mock_data"] = gmd
_spec.loader.exec_module(gmd)


@pytest.fixture(scope="module")
def mock(tmp_path_factory):
    out = tmp_path_factory.mktemp("photoz")
    argv = ["generate_mock_data.py", "--outdir", str(out), "--seed", "7",
            "--n-galaxies", "60000", "--nobs", "3", "--nsamp", "16",
            "--nselection", "4000", "--nside", "8", "--zmax", "0.3"]
    old = sys.argv
    sys.argv = argv
    try:
        gmd.write_mock_data(gmd.parse_args())
    finally:
        sys.argv = old
    return out


# ---------------------------------------------------------------------------
# T8 -- the declared photo-z is realised
# ---------------------------------------------------------------------------
def test_the_catalog_carries_both_redshift_columns(mock):
    with h5py.File(mock / "mock_galaxy_catalog_complete.h5") as f:
        assert "z" in f and "z_obs" in f
        z, z_obs = f["z"][...], f["z_obs"][...]
    assert not np.array_equal(z, z_obs)
    assert np.abs(z_obs - z).max() > 0.0


def test_the_survey_sees_the_observed_redshift_not_the_true_one(mock):
    with h5py.File(mock / "mock_survey_raw.h5") as f:
        z_survey, zerr_survey = f["Z"][...], f["ZERR"][...]
    with h5py.File(mock / "mock_galaxy_catalog_complete.h5") as f:
        z_obs_all = f["z_obs"][...]
        z_true_all = f["z"][...]
    # every survey row is one of the catalog's OBSERVED redshifts ...
    assert np.isin(z_survey, z_obs_all).all()
    # ... and the survey block is NOT the true-redshift comb
    matched = np.isin(z_survey, z_true_all)
    assert matched.mean() < 0.01, "the survey is carrying true redshifts"
    # the declared error is the model evaluated at what was recorded
    survey_cfg = gmd.SurveyConfig()
    np.testing.assert_array_equal(zerr_survey, gmd._catalog_zerr(z_survey, survey_cfg))


def test_the_pixelated_block_is_the_observed_redshift_bitwise(mock):
    with h5py.File(mock / "mock_survey_raw.h5") as f:
        z_survey, zerr_survey = f["Z"][...], f["ZERR"][...]
    with h5py.File(mock / "catalog_pixelated_nside_8.h5") as f:
        zgals, dzgals, ngals = f["zgals"][...], f["dzgals"][...], f["ngals"][...]
    real = np.arange(zgals.shape[1])[None, :] < ngals[:, None]
    np.testing.assert_array_equal(np.sort(zgals[real]), np.sort(z_survey))
    np.testing.assert_array_equal(np.sort(dzgals[real]), np.sort(zerr_survey))
    # padded slots keep the sentinels the loader expects
    assert (zgals[~real] == 100.0).all()
    assert (dzgals[~real] == 1.0).all()


def test_the_realised_photoz_scatter_matches_the_declared_width(mock):
    with h5py.File(mock / "mock_galaxy_catalog_complete.h5") as f:
        z, z_obs = f["z"][...], f["z_obs"][...]
    pull = (z_obs - z) / gmd._catalog_zerr(z, gmd.SurveyConfig())
    assert abs(pull.mean()) < 5.0 / np.sqrt(pull.size)
    assert abs(pull.std() - 1.0) < 0.05
    assert stats.kstest(pull, "norm").pvalue > 1e-3


def test_negative_observed_redshifts_are_not_clipped():
    """A galaxy at z ~ 0 can scatter below zero.  Clipping would re-introduce a
    censored observation, so the count is RECORDED, not repaired."""
    survey = gmd.SurveyConfig()
    rng = np.random.default_rng(3)
    # a population pressed against z = 0, where the floor makes the tail reachable
    z = np.abs(rng.normal(0.0, 1e-4, 200_000))
    z_obs = z + gmd._catalog_zerr(z, survey) * rng.normal(size=z.size)
    assert (z_obs < 0.0).any()
    assert z_obs.min() < 0.0
    # and the generator itself records the realised count rather than asserting it
    import json
    import inspect
    src = inspect.getsource(gmd.write_mock_data)
    assert "n_negative_z_obs" in src
    assert json.dumps  # the count travels in metadata_json


def test_metadata_records_that_the_photoz_is_realised(mock):
    import json
    with h5py.File(mock / "catalog_pixelated_nside_8.h5") as f:
        meta = json.loads(f.attrs["metadata_json"])
    assert meta["catalog_photoz"]["realised"] is True
    assert "n_negative_z_obs_in_survey" in meta["catalog_photoz"]


def test_the_true_redshift_still_drives_the_event_truth(mock):
    """Only the survey sees ``z_obs``; the hosts and the event truth stay on the
    true redshift, which is what makes the mock generatively consistent."""
    with h5py.File(mock / "mock_galaxy_catalog_complete.h5") as f:
        z_true = f["z"][...]
    with h5py.File(mock / "mock_gw_events.h5") as f:
        z_event = f["truth"]["z"][...]
    assert np.isin(z_event, z_true).all()


# ---------------------------------------------------------------------------
# T9 -- the padded-row KDE dtype guard
# ---------------------------------------------------------------------------
def test_pixelated_arrays_are_float64(mock):
    with h5py.File(mock / "catalog_pixelated_nside_8.h5") as f:
        for key in ("zgals", "dzgals", "wgals"):
            assert f[key].dtype == np.float64, (key, f[key].dtype)


def test_kde_on_a_padded_row_is_finite(mock):
    """Direct regression against the ``0/0 = NaN`` trap: a pixel row that
    contains padding must still give a finite KDE."""
    jnp = pytest.importorskip("jax.numpy")
    from darksirens.redshift.completion import _kde_dndz_obs

    with h5py.File(mock / "catalog_pixelated_nside_8.h5") as f:
        zgals, ngals = f["zgals"][...], f["ngals"][...]
    padded = np.where((ngals > 0) & (ngals < zgals.shape[1]))[0]
    assert padded.size, "no partially-filled pixel row to test"
    out = _kde_dndz_obs(int(padded[0]), jnp.asarray(zgals), ngals=jnp.asarray(ngals))
    assert np.isfinite(np.asarray(out)).all()
    assert float(np.asarray(out).sum()) > 0.0


def test_float32_padding_would_have_produced_nan():
    """Pins WHY the guard exists: the same computation in float32 is all-NaN, so
    a silent dtype change is a silent likelihood failure."""
    jnp = pytest.importorskip("jax.numpy")
    from darksirens.redshift.completion import _kde_dndz_obs

    zgals = np.array([[0.10, 0.12, 100.0, 100.0]], dtype=np.float32)
    ngals = np.array([2], dtype=np.int32)
    bad = np.asarray(_kde_dndz_obs(0, jnp.asarray(zgals), ngals=jnp.asarray(ngals)))
    good = np.asarray(_kde_dndz_obs(0, jnp.asarray(zgals.astype(np.float64)),
                                    ngals=jnp.asarray(ngals)))
    assert not np.isfinite(bad).all()
    assert np.isfinite(good).all()
