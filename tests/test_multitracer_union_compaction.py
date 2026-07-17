"""Per-bundle PE-union-selection compaction (E2).

``load_multitracer_catalog_bundles`` compacts each catalog's PE and selection
views ONCE over their pixel union, so the two EMCatalogs of a bundle carry the
SAME (``is``-identical) galaxy table, ONE KDE cache and singly-compacted
Q/member/mark rows -- the compact-view analogue of the flat single-catalog union
path in ``catalog_views.prepare_catalog_views``.  These tests pin, on a small
synthetic K=2 union bundle:

  * ONE KDE-cache build per bundle (not two) -- mirrors the cache_calls spy in
    tests/test_dark_sirens_startup_likelihood.py;
  * the PE/selection EMCatalogs alias their big array leaves (measured byte
    reduction);
  * EXACT likelihood parity vs a separate-object (non-aliased) build of the same
    pixel content -- the aliasing is a memory optimization, never a value change.

Fixtures reuse tests/test_multitracer_likelihood.py's GW physics / opts helpers.
"""
import jax
import jax.numpy as jnp
import numpy as np

from darksirens.redshift import zgrid
from darksirens.likelihood import factory as factory_mod
from darksirens.likelihood.factory import make_likelihood

from test_multitracer_likelihood import (
    APIX1,
    _base_opts,
    _mid_pop,
    _pop_bits,
    _shared_physics,
)

# PE has nsamp=2 samples, selection has n_sel=8 (see _shared_physics); the two
# views land in DIFFERENT pixel sets so the union is a genuine superset of each.
_PE_PIX = np.array([2, 3], dtype=np.int32)
_SEL_PIX = np.array([2, 5, 7, 5, 2, 7, 3, 5], dtype=np.int32)


def _full_catalog(npix=12, z=0.10, dz=0.02, nmax=2):
    return (
        np.full((npix, nmax), z, dtype=float),
        np.full((npix, nmax), dz, dtype=float),
        np.ones((npix, nmax), dtype=float),
        np.ones(npix, dtype=np.int32),
    )


def _union_rows(pe_pix, sel_pix, full):
    z, dz, w, n = full
    union = np.unique(
        np.concatenate([np.unique(pe_pix), np.unique(sel_pix)])
    ).astype(np.int32)
    s2u_pe = np.searchsorted(union, pe_pix).astype(np.int32)
    s2u_sel = np.searchsorted(union, sel_pix).astype(np.int32)
    return union, z[union], dz[union], w[union], n[union], s2u_pe, s2u_sel


def _union_bundle(apix, pe_pix, sel_pix, full):
    """Loader-shaped bundle: PE and selection reference the SAME union objects."""
    union, z_u, dz_u, w_u, n_u, s2u_pe, s2u_sel = _union_rows(pe_pix, sel_pix, full)
    return dict(
        apix=apix,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        zgals_pe=z_u, dzgals_pe=dz_u, wgals_pe=w_u, ngals_pe=n_u,
        unique_pixels_pe=union, sample_to_unique_pe=s2u_pe,
        # SAME objects on the selection side -> caller_union_views is detected.
        zgals_sel=z_u, dzgals_sel=dz_u, wgals_sel=w_u, ngals_sel=n_u,
        unique_pixels_sel=union, sample_to_unique_sel=s2u_sel,
    )


def _separate_bundle(apix, pe_pix, sel_pix, full):
    """Same pixel CONTENT, but distinct PE/selection objects (no aliasing): the
    pre-E2 separate-view path -- used as the parity control."""
    union, z_u, dz_u, w_u, n_u, s2u_pe, s2u_sel = _union_rows(pe_pix, sel_pix, full)
    return dict(
        apix=apix,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        zgals_pe=z_u.copy(), dzgals_pe=dz_u.copy(), wgals_pe=w_u.copy(),
        ngals_pe=n_u.copy(),
        unique_pixels_pe=union.copy(), sample_to_unique_pe=s2u_pe,
        zgals_sel=z_u.copy(), dzgals_sel=dz_u.copy(), wgals_sel=w_u.copy(),
        ngals_sel=n_u.copy(),
        unique_pixels_sel=union.copy(), sample_to_unique_sel=s2u_sel,
    )


def _k2_union_data(bundle_factory):
    data = dict(_shared_physics())
    data["apix"] = APIX1
    full = _full_catalog()
    data["catalogs"] = [
        bundle_factory(APIX1, _PE_PIX, _SEL_PIX, full),
        bundle_factory(APIX1, _PE_PIX, _SEL_PIX, full),
    ]
    return data


def _array_leaves(cat):
    for field in cat._fields:
        v = getattr(cat, field)
        if v is not None and hasattr(v, "nbytes") and hasattr(v, "shape") \
                and np.asarray(v).ndim >= 1:
            yield field, v


def test_union_bundle_builds_one_kde_cache_per_bundle(monkeypatch):
    """ONE KDE-cache build per bundle (K=2 => 2 calls), not two per bundle."""
    from darksirens.redshift.completion import build_pixel_kde_cache as real_builder

    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()

    cache_calls = []

    def spy_build_pixel_kde_cache(
        unique_pixels, zgals, n_pix_catalog, wgals=None, ngals=None
    ):
        cache_calls.append(np.asarray(unique_pixels).copy())
        return real_builder(
            unique_pixels=unique_pixels, zgals=zgals,
            n_pix_catalog=n_pix_catalog, wgals=wgals, ngals=ngals,
        )

    monkeypatch.setattr(
        factory_mod, "build_pixel_kde_cache", spy_build_pixel_kde_cache
    )

    data = _k2_union_data(_union_bundle)
    opts = _base_opts(n_catalogs=2)
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    val = float(ll(jnp.asarray([_mid_pop(), 0.3])))

    assert np.isfinite(val)
    # 2 bundles, ONE cache each (the pre-E2 separate path would build 4).
    assert len(cache_calls) == 2


def test_union_bundle_pe_sel_emcatalogs_alias_and_reduce_bytes(monkeypatch):
    """The bundle's PE and selection EMCatalogs share their big array leaves;
    the aliasing measurably reduces the referenced device bytes."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()

    captured = []
    orig_em_catalog = factory_mod.EMCatalog

    def _spy(*args, **kwargs):
        obj = orig_em_catalog(*args, **kwargs)
        captured.append(obj)
        return obj

    monkeypatch.setattr(factory_mod, "EMCatalog", _spy)

    data = _k2_union_data(_union_bundle)
    opts = _base_opts(n_catalogs=2)
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    val = float(ll(jnp.asarray([_mid_pop(), 0.3])))
    assert np.isfinite(val)

    # Loop order in _make_mixture_likelihood: per bundle, PE then selection.
    assert len(captured) == 4
    total_referenced = 0
    total_unique = 0
    for pe_cat, sel_cat in ((captured[0], captured[1]), (captured[2], captured[3])):
        # The large per-pixel leaves are the SAME object across the two views.
        assert pe_cat.zgals is sel_cat.zgals
        assert pe_cat.dzgals is sel_cat.dzgals
        assert pe_cat.wgals is sel_cat.wgals
        assert pe_cat.ngals is sel_cat.ngals
        assert pe_cat.unique_pixels is sel_cat.unique_pixels
        assert pe_cat.dN_obs_kde is sel_cat.dN_obs_kde
        assert pe_cat.pixel_to_cache_idx is sel_cat.pixel_to_cache_idx
        # ... but each view keeps its OWN per-sample map (that is the one leaf
        # that legitimately differs and which prepare never reads).
        assert pe_cat.sample_to_unique_idx is not sel_cat.sample_to_unique_idx

        seen = {}
        referenced = 0
        for _f, v in list(_array_leaves(pe_cat)) + list(_array_leaves(sel_cat)):
            nb = int(np.asarray(v).nbytes)
            referenced += nb
            seen[id(v)] = nb
        unique = sum(seen.values())
        total_referenced += referenced
        total_unique += unique
        assert unique < referenced  # aliasing saved bytes

    saved = total_referenced - total_unique
    print(
        f"\n[E2 memory] K=2 union bundle EMCatalog array bytes: "
        f"referenced={total_referenced} unique={total_unique} "
        f"saved={saved} ({100.0 * saved / total_referenced:.1f}%)"
    )
    assert saved > 0


def test_union_and_separate_bundles_are_bitwise_equal():
    """The aliasing/one-cache optimization is value-invariant: a union bundle
    and a separate-object build of the SAME pixel content give the identical
    likelihood (and gradient) bit-for-bit."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()
    coord = jnp.asarray([_mid_pop(), 0.37])

    opts = _base_opts(n_catalogs=2)
    ll_union = make_likelihood(
        opts, _k2_union_data(_union_bundle), pop_fid, fixed_parameter_values=fixed
    )
    ll_sep = make_likelihood(
        opts, _k2_union_data(_separate_bundle), pop_fid, fixed_parameter_values=fixed
    )

    val_union = float(ll_union(coord))
    val_sep = float(ll_sep(coord))
    assert np.isfinite(val_union)
    assert abs(val_union - val_sep) <= 1e-12

    g_union = np.asarray(jax.grad(lambda c: ll_union(c))(coord))
    g_sep = np.asarray(jax.grad(lambda c: ll_sep(c))(coord))
    assert np.allclose(g_union, g_sep, rtol=0.0, atol=1e-12)
