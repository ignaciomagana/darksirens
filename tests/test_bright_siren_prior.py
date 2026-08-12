import jax
jax.config.update("jax_enable_x64", True)

import healpy as hp
import jax.numpy as jnp
import numpy as np

from darksirens.redshift.prior import PRIOR_REGISTRY, _log_prior_bright_sirens
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey():
    return SurveyParams(
        n0=1.0,
        z50=1.0,
        w=0.5,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=0.5,
        complete_empty_pixel_policy=0,
    )


def _catalog(counterpart_nside=2, sky_marginalized=False):
    counterpart_pixel = 7
    non_counterpart_pixel = 8
    unique_pixels = jnp.array([counterpart_pixel, non_counterpart_pixel], dtype=jnp.int32)
    return EMCatalog(
        apix=hp.nside2pixarea(counterpart_nside),
        zgals=jnp.array([[0.2], [0.0]]),
        dzgals=jnp.array([[0.01], [1.0]]),
        wgals=jnp.array([[1.0], [0.0]]),
        ngals=jnp.array([1, 0], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=unique_pixels,
        counterpart_pixel=counterpart_pixel,
        bright_siren_sky_marginalized=sky_marginalized,
    )


def test_bright_siren_prior_only_finite_in_counterpart_pixel_with_nside_gt_one():
    z = jnp.array([0.2, 0.2])
    # Compact row ids for GW samples in the matching and non-matching pixels.
    pix = jnp.array([0, 1], dtype=jnp.int32)

    actual = _log_prior_bright_sirens(z, pix, _cosmo(), _survey(), _catalog())

    actual_np = np.asarray(actual)
    assert np.isfinite(actual_np[0])
    assert np.isneginf(actual_np[1])


def test_bright_siren_sky_marginalized_mode_uses_counterpart_redshift_for_all_pixels():
    z = jnp.array([0.2, 0.2])
    pix = jnp.array([0, 1], dtype=jnp.int32)

    actual = _log_prior_bright_sirens(
        z, pix, _cosmo(), _survey(), _catalog(sky_marginalized=True)
    )

    actual_np = np.asarray(actual)
    assert np.all(np.isfinite(actual_np))
    np.testing.assert_allclose(actual_np[0], actual_np[1], rtol=1e-12)


def test_prior_registry_uses_dedicated_bright_siren_prior():
    assert PRIOR_REGISTRY["bright_sirens"] is _log_prior_bright_sirens


def test_bright_siren_prior_uses_active_counterpart_for_multi_event_catalog():
    unique_pixels = jnp.array([7, 8], dtype=jnp.int32)
    catalog = EMCatalog(
        apix=hp.nside2pixarea(2),
        zgals=jnp.array([[0.2], [0.35]]),
        dzgals=jnp.array([[0.01], [0.01]]),
        wgals=jnp.array([[1.0], [1.0]]),
        ngals=jnp.array([1, 1], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=unique_pixels,
        counterpart_pixels=jnp.array([7, 8], dtype=jnp.int32),
        counterpart_zs=jnp.array([0.2, 0.35]),
        counterpart_dzs=jnp.array([0.01, 0.01]),
        active_counterpart_index=1,
        bright_siren_sky_marginalized=False,
    )

    actual = _log_prior_bright_sirens(
        jnp.array([0.35, 0.2]),
        jnp.array([1, 0], dtype=jnp.int32),
        _cosmo(),
        _survey(),
        catalog,
    )

    actual_np = np.asarray(actual)
    assert np.isfinite(actual_np[0])
    assert np.isneginf(actual_np[1])


def test_counterpart_prior_carries_the_volumetric_population_factor():
    """The per-event counterpart branch must multiply the EM likelihood by the
    population's redshift density.

    ``selection_prior_model`` routes bright_sirens' selection integral through
    ``spectral_sirens`` (the normalised dV_c/dz volume prior), so a numerator
    carrying the EM Gaussian alone uses a different p(z | Lambda) than mu does.
    """
    from jax.scipy.stats import norm

    from darksirens.redshift.volume import log_volume_prior_vmap

    unique_pixels = jnp.array([7, 8], dtype=jnp.int32)
    catalog = EMCatalog(
        apix=hp.nside2pixarea(2),
        zgals=jnp.array([[0.2], [0.35]]),
        dzgals=jnp.array([[0.01], [0.01]]),
        wgals=jnp.array([[1.0], [1.0]]),
        ngals=jnp.array([1, 1], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        unique_pixels=unique_pixels,
        counterpart_pixels=jnp.array([7, 8], dtype=jnp.int32),
        counterpart_zs=jnp.array([0.2, 0.35]),
        counterpart_dzs=jnp.array([0.01, 0.01]),
        active_counterpart_index=0,
        bright_siren_sky_marginalized=True,
    )
    z = jnp.array([0.15, 0.2, 0.25])
    cosmo, survey = _cosmo(), _survey()
    actual = np.asarray(
        _log_prior_bright_sirens(z, jnp.zeros(3, dtype=jnp.int32), cosmo, survey, catalog)
    )
    expected = np.asarray(
        norm.logpdf(z, 0.2, 0.01) + log_volume_prior_vmap(z, cosmo, survey)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    # ... and the volumetric tilt is not a constant offset: it shifts the prior's
    # effective mean, which is the whole point.
    em_only = np.asarray(norm.logpdf(z, 0.2, 0.01))
    assert not np.allclose(actual - em_only, (actual - em_only)[0], rtol=1e-6)
