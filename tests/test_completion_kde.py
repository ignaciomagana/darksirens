import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.redshift.completion import _kde_dndz_obs, build_pixel_kde_cache
from darksirens.redshift import zgrid


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


def _padded_catalog(ngals_list, n_max_gals, seed=0):
    """A padded (n_pix, n_max_gals) catalog with distinct real redshifts per
    slot and zeros in the padded tail, so rows are separable and empty rows
    (ngals == 0) contribute nothing."""
    rng = np.random.default_rng(seed)
    ngals = np.asarray(ngals_list, dtype=np.int32)
    n_pix = ngals.size
    zgals = np.zeros((n_pix, n_max_gals), dtype=np.float64)
    for i, ng in enumerate(ngals):
        if ng > 0:
            zgals[i, :ng] = rng.uniform(0.05, 0.6, size=int(ng))
    return jnp.asarray(zgals), jnp.asarray(ngals)


def test_build_pixel_kde_cache_chunked_parity_with_remainder():
    """Chunked build (incl. a partial remainder chunk) matches the single-shot
    build and row-for-row direct _kde_dndz_obs calls."""
    n_max_gals = 4
    # 11 pixels, mixed ngals, two empty rows -> batch_size=4 gives 2 full
    # chunks (8 rows) + a 3-row remainder.
    ngals_list = [0, 1, 2, 3, 4, 2, 0, 3, 1, 4, 2]
    zgals, ngals = _padded_catalog(ngals_list, n_max_gals, seed=1)
    unique_pixels = np.arange(len(ngals_list), dtype=np.int32)

    chunked, idx_chunked = build_pixel_kde_cache(
        unique_pixels=unique_pixels,
        zgals=zgals,
        n_pix_catalog=len(ngals_list),
        ngals=ngals,
        batch_size=4,
    )
    single, idx_single = build_pixel_kde_cache(
        unique_pixels=unique_pixels,
        zgals=zgals,
        n_pix_catalog=len(ngals_list),
        ngals=ngals,
        batch_size=1000,
    )

    # Chunking is a pure host-side partition of the same jit(vmap) row op, so
    # the two builds agree to ~machine precision (GPU reduction tiling can
    # differ at ulp level; don't assert bitwise).
    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(single), rtol=1e-12
    )
    # The pixel->cache-idx map is independent of chunking.
    np.testing.assert_array_equal(
        np.asarray(idx_chunked), np.asarray(idx_single)
    )

    # Row-for-row against direct _kde_dndz_obs (empty rows must be exactly 0).
    for pix in unique_pixels:
        expected = _kde_dndz_obs(int(pix), zgals, ngals=ngals)
        np.testing.assert_allclose(
            np.asarray(chunked[int(pix)]), np.asarray(expected), rtol=1e-12
        )
    assert np.all(np.asarray(chunked[0]) == 0.0)
    assert np.all(np.asarray(chunked[6]) == 0.0)


def test_build_pixel_kde_cache_chunk_plan_default_batch():
    """A moderate synthetic build spanning multiple default (512) chunks plus a
    remainder matches the unchunked build in shape and value."""
    n_rows = 1030  # batch_size=512 -> 2 full chunks (1024) + 6-row remainder.
    n_max_gals = 3
    ngals_list = [(i % n_max_gals) + 1 for i in range(n_rows)]
    zgals, ngals = _padded_catalog(ngals_list, n_max_gals, seed=2)
    unique_pixels = np.arange(n_rows, dtype=np.int32)

    chunked, _ = build_pixel_kde_cache(
        unique_pixels=unique_pixels,
        zgals=zgals,
        n_pix_catalog=n_rows,
        ngals=ngals,
    )  # default batch_size=512
    single, _ = build_pixel_kde_cache(
        unique_pixels=unique_pixels,
        zgals=zgals,
        n_pix_catalog=n_rows,
        ngals=ngals,
        batch_size=2000,  # single shot
    )

    assert chunked.shape == (n_rows, zgrid.size)
    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(single), rtol=1e-12
    )


def test_completion_clip_diagnostics_reports_grid_fractions():
    from darksirens.redshift.completion import completion_clip_diagnostics
    from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog

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
