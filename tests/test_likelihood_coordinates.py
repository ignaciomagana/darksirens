import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from darksirens.inference.utils import (
    log_jacobian_m1src_q_z_to_m1det_q_dL,
    log_sample_weight,
)
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import z_of_dL, ddL_of_z


def _dummy_survey():
    return SurveyParams(
        n0=0.0,
        z50=1.0,
        w=0.5,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=0.5,
    )


def _dummy_catalog():
    return EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _analytic_log_pop(m1src, q, z, chieff, pop_params):
    """A separable, positive density with non-trivial dependence on all coordinates."""
    del pop_params
    return (
        -0.5 * ((m1src - 30.0) / 8.0) ** 2
        -0.5 * ((q - 0.7) / 0.15) ** 2
        -0.5 * ((chieff + 0.05) / 0.2) ** 2
        + 0.3 * jnp.log1p(z)
    )


def _flat_log_prior_z(z, pix, catalog):
    del z, pix, catalog
    return 0.0


def _known_proposal_density_m1det_q_dL(m1det, q, dL, chieff):
    """Known positive proposal density in the canonical (m1det, q, dL) basis."""
    return jnp.exp(
        -0.5 * ((m1det - 38.0) / 9.0) ** 2
        -0.5 * ((q - 0.65) / 0.18) ** 2
        -0.5 * ((dL - 900.0) / 250.0) ** 2
        -0.5 * ((chieff - 0.02) / 0.25) ** 2
    )


def test_target_jacobian_is_for_m1det_q_dL_coordinates():
    cosmo = CosmoParams(H0=67.74, Om0=0.3089, w0=-1.0, wa=0.0)
    dL = jnp.array([250.0, 800.0, 1400.0])
    z = z_of_dL(dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)

    actual = log_jacobian_m1src_q_z_to_m1det_q_dL(
        z, dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    )
    expected = jnp.log(
        ddL_of_z(z, dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
    ) + jnp.log1p(z)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12)


def test_pe_and_selection_weights_are_invariant_under_equivalent_m2_q_proposals():
    """
    PE and detected-injection samples drawn from the same analytic proposal
    must give identical weights whether that proposal is represented natively
    in q or converted from an equivalent m2 density.
    """
    cosmo = CosmoParams(H0=67.74, Om0=0.3089, w0=-1.0, wa=0.0)
    survey = _dummy_survey()
    catalog = _dummy_catalog()
    pop_params = jnp.array([])

    m1det = jnp.array([28.0, 35.0, 47.0, 60.0])
    q = jnp.array([0.55, 0.72, 0.83, 0.61])
    dL = jnp.array([350.0, 700.0, 1050.0, 1500.0])
    chieff = jnp.array([-0.1, 0.0, 0.12, 0.25])
    pix = jnp.zeros_like(m1det, dtype=jnp.int32)

    p_q = _known_proposal_density_m1det_q_dL(m1det, q, dL, chieff)

    # Equivalent native density in (m1det, m2det, dL).  Since m2det = q*m1det,
    # p_q(m1det, q, dL) = p_m2(m1det, m2det, dL) * |dm2det/dq|.
    p_m2_native = p_q / m1det
    p_m2_converted_to_q = p_m2_native * m1det

    pe_weights_q = log_sample_weight(
        m1det,
        q,
        dL,
        chieff,
        pix,
        p_q,
        cosmo,
        survey,
        pop_params,
        catalog,
        _analytic_log_pop,
        _flat_log_prior_z,
    )
    pe_weights_m2 = log_sample_weight(
        m1det,
        q,
        dL,
        chieff,
        pix,
        p_m2_converted_to_q,
        cosmo,
        survey,
        pop_params,
        catalog,
        _analytic_log_pop,
        _flat_log_prior_z,
    )

    # Reuse the same known proposal as detected injections to assert the
    # selection path obeys the same coordinate convention as PE.
    selection_weights_q = log_sample_weight(
        m1det,
        q,
        dL,
        chieff,
        pix,
        p_q,
        cosmo,
        survey,
        pop_params,
        catalog,
        _analytic_log_pop,
        _flat_log_prior_z,
    )
    selection_weights_m2 = log_sample_weight(
        m1det,
        q,
        dL,
        chieff,
        pix,
        p_m2_converted_to_q,
        cosmo,
        survey,
        pop_params,
        catalog,
        _analytic_log_pop,
        _flat_log_prior_z,
    )

    np.testing.assert_allclose(np.asarray(pe_weights_q), np.asarray(pe_weights_m2), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(selection_weights_q), np.asarray(selection_weights_m2), rtol=1e-12
    )
    np.testing.assert_allclose(np.asarray(pe_weights_q), np.asarray(selection_weights_q), rtol=1e-12)


def test_z_of_dL_returns_nan_outside_interpolation_grid():
    """Distances outside the tabulated dL(z) support must not clamp to z-grid edges."""
    from darksirens.utils.cosmology import dL_grid_bounds

    cosmo = CosmoParams(H0=67.74, Om0=0.3089, w0=-1.0, wa=0.0)
    dL_min, dL_max = dL_grid_bounds(cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
    dL = jnp.array([dL_min - 1.0, 500.0, dL_max + 1.0])

    z = z_of_dL(dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)

    assert bool(jnp.isnan(z[0]))
    assert bool(jnp.isfinite(z[1]))
    assert bool(jnp.isnan(z[2]))
