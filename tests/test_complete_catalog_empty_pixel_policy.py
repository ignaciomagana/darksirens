import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.redshift.catalog import log_catalog_prior_vmap
from darksirens.redshift.prior import _log_prior_complete_catalog
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(policy):
    return SurveyParams(
        n0=1.0,
        z50=1.0,
        w=0.5,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=0.5,
        complete_empty_pixel_policy=policy,
    )


def _catalog():
    return EMCatalog(
        apix=1.0,
        zgals=jnp.array([
            [0.0, 0.0],
            [0.2, 0.35],
        ]),
        dzgals=jnp.array([
            [1.0, 1.0],
            [0.02, 0.03],
        ]),
        wgals=jnp.array([
            [0.0, 0.0],
            [1.0, 0.5],
        ]),
        ngals=jnp.array([0, 2], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((2, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def test_complete_catalog_zero_policy_returns_negative_infinity_for_empty_pixel():
    z = jnp.array([0.25])
    pix = jnp.array([0], dtype=jnp.int32)

    actual = _log_prior_complete_catalog(z, pix, _cosmo(), _survey(0), _catalog())

    assert np.isneginf(np.asarray(actual)[0])


def test_complete_catalog_volume_policy_uses_finite_volume_fallback_for_empty_pixel():
    z = jnp.array([0.25])
    pix = jnp.array([0], dtype=jnp.int32)

    actual = _log_prior_complete_catalog(z, pix, _cosmo(), _survey(1), _catalog())

    assert np.isfinite(np.asarray(actual)[0])


def test_complete_catalog_non_empty_pixel_uses_catalog_prior_for_both_policies():
    z = jnp.array([0.25])
    pix = jnp.array([1], dtype=jnp.int32)
    catalog = _catalog()
    cosmo = _cosmo()

    expected = log_catalog_prior_vmap(z, pix, cosmo, _survey(0), catalog)
    strict = _log_prior_complete_catalog(z, pix, cosmo, _survey(0), catalog)
    fallback = _log_prior_complete_catalog(z, pix, cosmo, _survey(1), catalog)

    np.testing.assert_allclose(np.asarray(strict), np.asarray(expected), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(fallback), np.asarray(expected), rtol=1e-12)
