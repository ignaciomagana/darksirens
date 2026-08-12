"""Contract tests for ``darksirens.redshift.checks`` -- the startup guard.

The module advertises ``raise_on_failure=True`` as the thing standing between a
normalisation bug and "silently corrupted inference results", so the DEFAULT
check set has to pass on a healthy model.  It could not: ``models`` defaulted to
every ``PRIOR_REGISTRY`` entry, including ``bright_sirens``, which is ``-inf``
off the counterpart pixel BY CONSTRUCTION and therefore integrates to 0 (FAIL)
at every other test pixel -- and nothing in the package imported the module, so
the contract was never executed.  These tests execute it.
"""
import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift import zgrid
from darksirens.redshift.checks import (
    DEFAULT_CHECK_MODELS,
    check_catalog_prior_normalization,
    check_prior_normalization,
    run_all_checks,
)

NSIDE = 4
NG = len(zgrid)


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(z_depth=None):
    return SurveyParams(
        n0=1e-2,
        z50=1.0,
        w=0.5,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=0.5,
        z_depth=z_depth,
    )


def _catalog():
    """Two non-empty rows (row 0 is the counterpart pixel), tiny-fixture style
    of tests/test_bright_siren_prior.py."""
    return EMCatalog(
        apix=hp.nside2pixarea(NSIDE),
        zgals=jnp.array([[0.20, 0.28], [0.15, 0.33]]),
        dzgals=jnp.full((2, 2), 0.02),
        wgals=jnp.ones((2, 2)),
        ngals=jnp.array([2, 2], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((2, NG)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=jnp.array([7, 8], dtype=jnp.int32),
        counterpart_pixel=7,
    )


def test_run_all_checks_passes_on_a_healthy_model():
    """The advertised startup guard must not raise on a healthy model."""
    summary = run_all_checks(
        _cosmo(), _survey(), _catalog(), test_pixels=jnp.array([0, 1]),
        verbose=False, raise_on_failure=True,
    )
    assert summary["all_passed"] is True
    for model in DEFAULT_CHECK_MODELS:
        assert all(summary[f"prior_{model}"].values())


def test_bright_sirens_excluded_from_the_default_models():
    """It is normalised at the counterpart pixel ONLY -- so it cannot be part
    of a per-pixel default, but it must still check out where it is defined."""
    assert "bright_sirens" not in DEFAULT_CHECK_MODELS

    cosmo, survey, catalog = _cosmo(), _survey(), _catalog()
    at_counterpart = check_prior_normalization(
        "bright_sirens", cosmo, survey, catalog, jnp.array([0]), verbose=False
    )
    assert all(at_counterpart.values())
    off_counterpart = check_prior_normalization(
        "bright_sirens", cosmo, survey, catalog, jnp.array([1]), verbose=False
    )
    assert not any(off_counterpart.values())


def test_p_cat_check_honours_survey_z_depth():
    """The p_cat check must integrate the density the RUN uses: under a survey
    depth that is the truncated, below-depth-renormalised mixture (the one-shot
    ``log_catalog_prior`` ignores ``z_depth`` and would integrate another one)."""
    from darksirens.redshift.checks import _log_p_cat_grid

    cosmo, catalog = _cosmo(), _catalog()
    z_depth = 0.25
    log_p = _log_p_cat_grid(cosmo, _survey(z_depth=z_depth), catalog, 0)
    zg = np.asarray(zgrid)

    assert np.all(np.isneginf(log_p[zg > z_depth]))
    assert np.any(np.isfinite(log_p[zg <= z_depth]))
    assert all(check_catalog_prior_normalization(
        cosmo, _survey(z_depth=z_depth), catalog, jnp.array([0, 1]),
        verbose=False).values())


def test_nan_prior_cannot_pass_a_normalisation_check():
    """NaN is the failure mode these checks exist to catch: ``_integrate`` used
    to map every non-finite log_p to -inf, turning NaN into a density of exactly
    0 -- so a NaN-poisoned prior integrated to slightly LESS than 1 and PASSED.
    """
    from darksirens.redshift.checks import _check_result, _integrate

    z = np.linspace(0.0, 1.0, 11)
    log_p = np.zeros_like(z)                     # p = 1 on [0, 1]: integral 1
    assert _integrate(log_p, z) == pytest.approx(1.0)

    poisoned = log_p.copy()
    poisoned[5] = np.nan
    assert np.isnan(_integrate(poisoned, z))
    assert not _check_result("poisoned", _integrate(poisoned, z), 0.05, False)

    # -inf stays a legitimately zero density, not a failure.
    zeroed = log_p.copy()
    zeroed[5] = -np.inf
    assert np.isfinite(_integrate(zeroed, z))
