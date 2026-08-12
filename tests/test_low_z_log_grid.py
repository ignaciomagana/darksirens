"""The first z-grid cell: log-space floors must not be linearly interpolated.

``zgrid[0]`` is exactly 0, where ``dV_c/dz`` vanishes, so node 0 of every log
grid built from the volume element is a floor SENTINEL (``log(tiny)`` = -708,
``log(1e-300)`` = -690.8) introduced to keep the gradient finite.  Linear
interpolation of the log array across the first cell (node 1 sits at
z ~ 1.8e-3) then ramps ~300 decades from that sentinel up to the physical value
instead of following ``g ∝ z^2``: the volume prior returned exp(-361) at
z = 9e-4, and a galaxy that nearby was silently deleted from the host prior.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift.completion import log_galaxy_measure_grid
from darksirens.redshift.grid import log_interp_zgrid, zgrid
from darksirens.redshift.volume import log_volume_prior
from darksirens.utils.cosmology import dV_of_z

H0, OM0, W0, WA = 70.0, 0.3, -1.0, 0.0
# Inside the first cell, plus the first node itself.
Z_FIRST_CELL = jnp.array([1.8e-4, 5.0e-4, 8.98e-4, 1.62e-3, float(zgrid[1])])


def _cosmo(h0=H0):
    return CosmoParams(H0=h0, Om0=OM0, w0=W0, wa=WA)


def _survey():
    return SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                        alpha_miss=1.0, sigma_kde=0.0)


def test_log_galaxy_measure_follows_the_z_squared_law_in_the_first_cell():
    log_g = log_galaxy_measure_grid(_cosmo(), _survey())
    got = np.asarray(log_interp_zgrid(Z_FIRST_CELL, log_g))
    truth = np.log(np.asarray(dV_of_z(Z_FIRST_CELL, H0, OM0, W0, WA)))
    # delta = 0, so g = dV_c/dz exactly; the residual is the O(z) curvature of
    # the power law, not the ~700 e-folds of the log-linear ramp.
    np.testing.assert_allclose(got, truth, rtol=0, atol=1e-2)


def test_volume_prior_is_a_density_below_the_first_node():
    cosmo = _cosmo()
    norm = float(jnp.trapezoid(dV_of_z(zgrid, H0, OM0, W0, WA), zgrid))
    got = np.array([float(log_volume_prior(z, cosmo, _survey()))
                    for z in Z_FIRST_CELL])
    truth = np.log(np.asarray(dV_of_z(Z_FIRST_CELL, H0, OM0, W0, WA))) - np.log(
        norm)
    np.testing.assert_allclose(got, truth, rtol=0, atol=1e-2)
    # z = 0 is genuinely zero density, and must stay finite with a finite
    # gradient (what the floor was introduced for).
    assert np.isfinite(float(log_volume_prior(0.0, cosmo, _survey())))
    for z in (0.0, 1e-4):
        g = float(jax.grad(
            lambda h: log_volume_prior(z, _cosmo(h), _survey()))(H0))
        assert np.isfinite(g)


def test_nearby_galaxy_is_not_deleted_from_the_host_prior():
    """Complete-catalog (volume-weighted) host weights scale with g(z_gal): the
    log-linear ramp multiplied a z ~ 9e-4 galaxy's weight by exp(-339)."""
    from darksirens.redshift.catalog import (
        catalog_kernel_state,
        eval_log_catalog_prior_state,
    )

    zs = np.array([[8.98e-4, 0.05]])
    cat = EMCatalog(
        apix=1.0,
        zgals=jnp.asarray(zs),
        dzgals=jnp.full((1, 2), 1e-3),
        wgals=jnp.ones((1, 2)),
        ngals=jnp.asarray([2], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, int(zgrid.size))),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    state = catalog_kernel_state(_cosmo(), _survey(), cat, volume_weighted=True)
    log_kw = np.asarray(state.log_kw)[0]
    # Weight ratio of the two galaxies is g(z1)/g(z2) = dV(z1)/dV(z2) (~ the
    # z^2 law at these redshifts); pre-fix it was that times exp(-339).
    dv = np.asarray(dV_of_z(jnp.asarray(zs[0]), H0, OM0, W0, WA))
    expect = float(np.log(dv[0]) - np.log(dv[1]))
    assert (log_kw[0] - log_kw[1]) == pytest.approx(expect, abs=2e-2)

    lp = float(eval_log_catalog_prior_state(
        jnp.asarray(zs[0, 0]), jnp.asarray(0, dtype=jnp.int32), state, cat))
    assert np.isfinite(lp) and lp > -20.0
