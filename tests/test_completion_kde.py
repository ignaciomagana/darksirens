import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.em.completion import _kde_dndz_obs, build_pixel_kde_cache
from darksirens.em import zgrid


def test_kde_masks_empty_pixel_with_wgals_indicator():
    """An empty padded pixel must not contribute a fake z=0 galaxy."""
    zgals = jnp.zeros((1, 4))
    wgals = jnp.zeros((1, 4))

    dndz = _kde_dndz_obs(0, zgals, wgals=wgals)

    np.testing.assert_allclose(np.asarray(dndz), 0.0, atol=1e-14)


def test_kde_masks_partially_padded_pixel_with_ngals_indicator():
    """Padded zeros after the real entries must not create a low-z spike."""
    zgals = jnp.array([[0.5, 0.0, 0.0, 0.0]])
    ngals = jnp.array([1], dtype=jnp.int32)

    dndz = _kde_dndz_obs(0, zgals, ngals=ngals)
    low_z_value = float(dndz[0])
    real_gal_idx = int(jnp.argmin(jnp.abs(zgrid - 0.5)))
    real_gal_value = float(dndz[real_gal_idx])

    assert low_z_value < 1e-5
    assert real_gal_value > 0.5


def test_build_pixel_kde_cache_masks_empty_and_partially_padded_pixels():
    """Cached KDEs use the same real-galaxy masks as the uncached path."""
    zgals = jnp.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
    ])
    ngals = jnp.array([0, 1], dtype=jnp.int32)

    dN_obs_kde, pixel_to_cache_idx = build_pixel_kde_cache(
        unique_pixels=np.array([0, 1], dtype=np.int32),
        zgals=zgals,
        n_pix_catalog=2,
        ngals=ngals,
    )

    np.testing.assert_allclose(
        np.asarray(dN_obs_kde[pixel_to_cache_idx[0]]), 0.0, atol=1e-14
    )
    assert float(dN_obs_kde[pixel_to_cache_idx[1], 0]) < 1e-5


def test_completion_clip_diagnostics_reports_grid_fractions():
    from darksirens.em.completion import completion_clip_diagnostics
    from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog

    zgals = jnp.array([[0.1, 0.2]])
    wgals = jnp.ones_like(zgals)
    ngals = jnp.array([2], dtype=jnp.int32)
    dN_obs_kde, pixel_to_cache_idx = build_pixel_kde_cache(
        unique_pixels=np.array([0], dtype=np.int32),
        zgals=zgals,
        n_pix_catalog=1,
        ngals=ngals,
    )
    catalog = EMCatalog(
        apix=1.0,
        zgals=zgals,
        dzgals=jnp.full_like(zgals, 0.01),
        wgals=wgals,
        ngals=ngals,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        dN_obs_kde=dN_obs_kde,
        pixel_to_cache_idx=pixel_to_cache_idx,
    )
    diagnostics = completion_clip_diagnostics(
        CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0),
        SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5),
        catalog,
        max_pixels=1,
    )

    assert diagnostics["n_zgrid"] == len(zgrid)
    assert diagnostics["n_pixels_checked"] == 1
    for key in (
        "mean_C_iso_clipped_fraction",
        "mean_C_eff_clipped_fraction",
        "mean_rho_miss_eff_clipped_fraction",
    ):
        assert 0.0 <= diagnostics[key] <= 1.0
