"""Row-chunked kernel-state builds must match the unchunked vmap row-for-row.

The chunked path (``_map_rows`` + ``configure_catalog_row_chunk``) exists to
bound peak memory for wide-sky catalog views (49k rows x 2k galaxies OOMs an
80 GB device under the plain vmap); it must be a pure execution-strategy
change with no numerical effect.
"""

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

from darksirens.redshift.catalog import (
    catalog_kernel_state,
    configure_catalog_row_chunk,
    marked_catalog_kernel_state,
)


@pytest.fixture(autouse=True)
def _restore_row_chunk_mode():
    yield
    configure_catalog_row_chunk("auto")


def _toy_catalog(rng, n_rows=37, n_max=11, with_ngals=True):
    zgals = jnp.asarray(rng.uniform(0.01, 1.4, size=(n_rows, n_max)))
    dzgals = jnp.asarray(rng.uniform(1e-4, 5e-3, size=(n_rows, n_max)))
    ngals_np = rng.integers(0, n_max + 1, size=n_rows)
    wgals_np = np.zeros((n_rows, n_max))
    for i, n in enumerate(ngals_np):
        wgals_np[i, :n] = rng.uniform(0.5, 2.0, size=n)
    em = SimpleNamespace(
        zgals=zgals,
        dzgals=dzgals,
        wgals=jnp.asarray(wgals_np),
        ngals=jnp.asarray(ngals_np) if with_ngals else None,
    )
    return em


def _log_g_grid():
    from darksirens.redshift.grid import zgrid
    return jnp.log(1.0 + jnp.asarray(zgrid) ** 2)


_SURVEY = SimpleNamespace(sigma_kde=1e-3)


@pytest.mark.parametrize("with_ngals", [True, False])
@pytest.mark.parametrize("volume_weighted", [True, False])
def test_kernel_state_chunked_matches_unchunked(with_ngals, volume_weighted):
    rng = np.random.default_rng(7)
    em = _toy_catalog(rng, with_ngals=with_ngals)
    log_g = _log_g_grid()

    configure_catalog_row_chunk(None)
    ref = catalog_kernel_state(None, _SURVEY, em, log_g_grid=log_g,
                               volume_weighted=volume_weighted)
    # 37 rows with chunk 8 exercises the zero-row padding path (5 chunks, 3 pad).
    configure_catalog_row_chunk(8)
    chunked = catalog_kernel_state(None, _SURVEY, em, log_g_grid=log_g,
                                   volume_weighted=volume_weighted)

    np.testing.assert_allclose(np.asarray(chunked.log_kw), np.asarray(ref.log_kw),
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(chunked.sig_eff), np.asarray(ref.sig_eff),
                               rtol=1e-12, atol=0.0)
    assert chunked.volume_weighted == ref.volume_weighted


def test_marked_kernel_state_chunked_matches_unchunked():
    rng = np.random.default_rng(11)
    em = _toy_catalog(rng, with_ngals=True)
    log_g = _log_g_grid()
    log_h = jnp.asarray(rng.normal(0.0, 0.3, size=em.zgals.shape))

    configure_catalog_row_chunk(None)
    ref_state, ref_nhost = marked_catalog_kernel_state(
        None, _SURVEY, em, log_h, log_g_grid=log_g
    )
    configure_catalog_row_chunk(8)
    ch_state, ch_nhost = marked_catalog_kernel_state(
        None, _SURVEY, em, log_h, log_g_grid=log_g
    )

    np.testing.assert_allclose(np.asarray(ch_state.log_kw), np.asarray(ref_state.log_kw),
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(ch_nhost), np.asarray(ref_nhost),
                               rtol=1e-12, atol=0.0)


def test_auto_mode_leaves_small_catalogs_unchunked():
    from darksirens.redshift.catalog import _resolve_row_chunk, _ROW_CHUNK_SIZE
    configure_catalog_row_chunk("auto")
    assert _resolve_row_chunk(3072, 500) is None            # mock-scale view
    assert _resolve_row_chunk(49152, 2113) == _ROW_CHUNK_SIZE  # DESI wide-sky view


def _volumetric_log_g_grid():
    """A realistically STEEP integrand: the actual volumetric measure
    log dV_c/dz at Planck-like cosmology (the toy log(1+z^2) used previously
    is ~500x flatter and let an inaccurate node count pass)."""
    from darksirens.redshift.grid import zgrid
    from darksirens.core.types import CosmoParams
    from darksirens.utils.cosmology import dV_of_z
    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    g = np.asarray(dV_of_z(jnp.asarray(zgrid), cosmo.H0, cosmo.Om0,
                           cosmo.w0, cosmo.wa))
    return jnp.log(jnp.maximum(jnp.asarray(g), 1e-300))


def test_kernel_quadrature_gl8_accuracy_across_sigma_kde():
    """8-node GL matches 24-node GL with the real volumetric measure, over the
    full sampled sigma_kde prior range [0, 0.05].

    This is the validation behind --kernel_gl_nodes 8 in the letter production
    jobs. GL-4 is deliberately NOT certified here: its per-galaxy error grows
    ~20x by sigma_kde = 0.05 (module review E2), so it is only safe when
    sigma_kde is held near zero.
    """
    from darksirens.redshift.catalog import configure_kernel_quadrature

    rng = np.random.default_rng(3)
    log_g = _volumetric_log_g_grid()
    for sigma_kde in (0.0, 0.02, 0.05):
        em = _toy_catalog(rng, n_rows=25, n_max=9)
        survey = SimpleNamespace(sigma_kde=sigma_kde)
        try:
            configure_kernel_quadrature(24)
            ref = catalog_kernel_state(None, survey, em, log_g_grid=log_g)
            configure_kernel_quadrature(8)
            fast = catalog_kernel_state(None, survey, em, log_g_grid=log_g)
        finally:
            configure_kernel_quadrature(24)
        mask = np.isfinite(np.asarray(ref.log_kw))
        # Per-galaxy bound: GL-8 worst case is ~1.1e-2 at sigma_kde=0.05 (GL-4
        # is ~4.5e-2 and must FAIL this). The binding end-to-end validation is
        # the measured 259-event scan: total-logL error <= 0.05 with an H0
        # tilt of ~1e-3 per km/s/Mpc (working/GATES.md, module review E2).
        np.testing.assert_allclose(
            np.asarray(fast.log_kw)[mask], np.asarray(ref.log_kw)[mask],
            rtol=0.0, atol=2e-2,
            err_msg=f"GL-8 vs GL-24 at sigma_kde={sigma_kde}",
        )


def test_configure_kernel_quadrature_rejects_bad_counts():
    from darksirens.redshift.catalog import configure_kernel_quadrature
    with pytest.raises(ValueError):
        configure_kernel_quadrature(1)
    with pytest.raises(ValueError):
        configure_catalog_row_chunk(0)
