"""Row-direct KDE-cache indexing (E4).

``completion._row_C`` indexes ``dN_obs_kde`` directly by catalog row instead of
via the dense ``pixel_to_cache_idx[unique_pixels[row]]`` round-trip, which was a
pure identity map (cache rows are built 1:1 with ``unique_pixels`` in every
builder).  These tests pin the two facts that make that safe:

  * the KDE cache row ``k`` IS the density for ``unique_pixels[k]`` (row-1:1);
  * the completion (and the full likelihood) is INVARIANT to the magnitude of
    the global pixel ids -- so a catalog living at sparse, very HIGH pixel ids
    (which the old code would have keyed a ``max(pixel)+1``-sized dense array on)
    gives the identical result with no dense lookup built.
"""
import jax.numpy as jnp
import numpy as np

from darksirens.redshift import zgrid
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.redshift.completion import (
    build_pixel_kde_cache,
    completion_curves,
    _kde_dndz_obs,
)

NG = int(zgrid.size)
COSMO = CosmoParams(H0=67.74, Om0=0.3075)
SURVEY = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0, b_miss=0.0, alpha_miss=1.0)


def test_kde_cache_row_matches_unique_pixels():
    """build_pixel_kde_cache row k == the KDE for unique_pixels[k]."""
    # Full-sky catalog with a DISTINCT redshift per pixel so rows are separable.
    npix = 12
    zgals = jnp.asarray(
        np.linspace(0.05, 0.5, npix, dtype=float)[:, None] * np.ones((1, 2))
    )
    ngals = jnp.asarray(np.ones(npix, dtype=np.int32))
    unique_pixels = np.array([5, 2, 9, 0], dtype=np.int32)

    dN_obs_kde, _ = build_pixel_kde_cache(
        unique_pixels=unique_pixels,
        zgals=zgals,
        n_pix_catalog=npix,
        ngals=ngals,
    )
    assert dN_obs_kde.shape[0] == unique_pixels.size
    for k, pix in enumerate(unique_pixels):
        # Cache row k must be the KDE of unique_pixels[k] (tight tolerance: the
        # cache is a jit(vmap(_kde_dndz_obs)) so it re-associates vs the eager
        # reference at ~1e-15, but is unambiguously that pixel's density).
        expected = _kde_dndz_obs(int(pix), zgals, ngals=ngals)
        np.testing.assert_allclose(
            np.asarray(dN_obs_kde[k]), np.asarray(expected), rtol=1e-9, atol=1e-12
        )
        # ... and NOT another pixel's density (rows are genuinely distinct).
        other = _kde_dndz_obs(
            int(unique_pixels[(k + 1) % unique_pixels.size]), zgals, ngals=ngals
        )
        assert not np.allclose(np.asarray(dN_obs_kde[k]), np.asarray(other))


def _compact_catalog(unique_pixels):
    """A 2-row compact dark-siren catalog at the given global pixel ids, with a
    row-1:1 KDE cache and NO pixel_to_cache_idx lookup."""
    zg = jnp.array([[0.10, 0.20], [0.30, 0.00]])
    dz = jnp.array([[0.02, 0.02], [0.02, 0.02]])
    w = jnp.array([[1.0, 1.0], [1.0, 0.0]])
    ng = jnp.array([2, 1], dtype=jnp.int32)
    kde, _ = build_pixel_kde_cache(
        np.arange(2, dtype=np.int32), zg, 2, ngals=ng
    )
    return EMCatalog(
        apix=1.0, zgals=zg, dzgals=dz, wgals=w, ngals=ng,
        delta_g_pix_z=jnp.zeros((1, NG)),
        dN_obs_kde=kde, pixel_to_cache_idx=None,
        unique_pixels=jnp.asarray(unique_pixels, dtype=jnp.int32),
    )


def test_completion_invariant_to_global_pixel_id_magnitude():
    """Same compact content at LOW vs sparse-HIGH global pixel ids gives the
    bit-identical completion -- the cache is indexed by row, so the huge
    max(pixel)+1 lookup the old path allocated is never needed."""
    curves_low = completion_curves(COSMO, SURVEY, _compact_catalog([0, 1]))
    curves_high = completion_curves(
        COSMO, SURVEY, _compact_catalog([1_000_000, 2_000_003])
    )
    np.testing.assert_array_equal(
        np.asarray(curves_low.dN_miss), np.asarray(curves_high.dN_miss)
    )
    np.testing.assert_array_equal(
        np.asarray(curves_low.N_miss), np.asarray(curves_high.N_miss)
    )
    np.testing.assert_array_equal(
        np.asarray(curves_low.f), np.asarray(curves_high.f)
    )


def test_likelihood_parity_on_sparse_high_pixel_ids():
    """A K=1 bundle whose galaxies live at sparse, very HIGH global pixel ids
    yields the identical likelihood to the same content at low ids -- the
    completion indexes the KDE cache by row, so the pixel-id magnitude is inert
    (and no dense pixel_to_cache_idx is built)."""
    import jax.numpy as jnp
    from darksirens.likelihood.factory import make_likelihood
    from test_multitracer_likelihood import (
        APIX1, _base_opts, _mid_pop, _pop_bits, _shared_physics,
    )

    def _direct_bundle(unique_pixels):
        up = np.asarray(unique_pixels, dtype=np.int32)
        nrows = up.size
        z_u = np.full((nrows, 2), 0.10)
        dz_u = np.full((nrows, 2), 0.02)
        w_u = np.ones((nrows, 2))
        n_u = np.ones(nrows, dtype=np.int32)
        # nsamp=2 PE samples, n_sel=8 selection samples -> valid compact rows.
        s2u_pe = np.zeros(2, dtype=np.int32)
        s2u_sel = (np.arange(8, dtype=np.int32) % nrows).astype(np.int32)
        return dict(
            apix=APIX1, delta_g_pix_z=jnp.zeros((1, NG)),
            zgals_pe=z_u, dzgals_pe=dz_u, wgals_pe=w_u, ngals_pe=n_u,
            unique_pixels_pe=up, sample_to_unique_pe=s2u_pe,
            zgals_sel=z_u, dzgals_sel=dz_u, wgals_sel=w_u, ngals_sel=n_u,
            unique_pixels_sel=up, sample_to_unique_sel=s2u_sel,
        )

    _pl, _pu, _lbl, pop_fid, _s, fixed = _pop_bits()

    def _value(unique_pixels):
        data = dict(_shared_physics())
        data["apix"] = APIX1
        data["catalogs"] = [_direct_bundle(unique_pixels)]
        opts = _base_opts(n_catalogs=1)
        ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
        return float(ll(jnp.asarray([_mid_pop()])))

    val_low = _value([0, 1])
    val_high = _value([5_000_000, 9_000_017])
    assert np.isfinite(val_low)
    assert val_low == val_high  # bit-for-bit
