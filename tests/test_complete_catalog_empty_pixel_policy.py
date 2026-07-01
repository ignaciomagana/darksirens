import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.em.catalog import log_catalog_prior_vmap
from darksirens.em.prior import _log_prior_complete_catalog
from darksirens.utils.containers import CosmoParams, EMCatalog, SurveyParams


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


def test_complete_catalog_non_empty_pixel_is_normalized_density():
    # For a populated pixel the complete prior is a probability density in z:
    # it integrates to 1 over [0, zmax], and the empty-pixel policy is
    # irrelevant (it only governs empty pixels).
    from darksirens.em.utils import zgrid

    catalog = _catalog()
    cosmo = _cosmo()
    pix = jnp.array([1], dtype=jnp.int32)

    z = jnp.array([0.25])
    strict = _log_prior_complete_catalog(z, pix, cosmo, _survey(0), catalog)
    fallback = _log_prior_complete_catalog(z, pix, cosmo, _survey(1), catalog)
    np.testing.assert_allclose(np.asarray(strict), np.asarray(fallback), rtol=1e-12)

    zfine = jnp.asarray(np.linspace(1e-4, float(np.asarray(zgrid)[-1]), 4000))
    lp = _log_prior_complete_catalog(
        zfine, jnp.ones(zfine.size, jnp.int32), cosmo, _survey(0), catalog
    )
    integral = float(np.trapezoid(np.exp(np.asarray(lp)), np.asarray(zfine)))
    assert abs(integral - 1.0) < 1e-3
