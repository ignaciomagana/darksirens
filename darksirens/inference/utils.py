"""
utils.py
--------
Shared utility functions for the hierarchical inference likelihood.

The key design principle here is that ``log_sample_weight`` is a *pure
function* of its arguments — no closures over data or parameters.  This
makes it:

  1. Independently unit-testable against known analytic cases.
  2. Profil-able in isolation (JAX's profiler can attribute cost here).
  3. Reusable from both the PE term and the selection term without
     code duplication.

The Jacobian
------------
The population models are densities in source-frame ``(m1, q, z)`` while
PE and selection samples are stored in detector-frame ``(m1_det, m2_det, dL)``.
Changing variables with

    m1_det = (1 + z) * m1
    m2_det = (1 + z) * m1 * q
    dL     = dL(z)

introduces a factor:

    |∂(dL, m1_det, m2_det) / ∂(z, m1, q)|
        = d(dL)/dz * (1+z)^2 * m1

In log space: log ddL_of_z + 2 log(1+z) + log(m1).

This is the *only* place in the codebase where this Jacobian is
computed.  Do not inline it elsewhere.
"""

from __future__ import annotations

import jax.numpy as jnp

from darksirens.utils.cosmology import z_of_dL, ddL_of_z
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog


def log_jacobian_dL_to_z(
    z: jnp.ndarray,
    dL: jnp.ndarray,
    H0: jnp.ndarray,
    Om0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Log-Jacobian for the partial variable change dL → z and
    detector-frame masses → source-frame masses at fixed mass ratio.

    This helper intentionally excludes the ``m1`` factor from the
    ``(m1, m2) → (m1, q)`` coordinate change.  Use
    ``log_jacobian_detector_to_source_q`` for likelihood weights.
    """
    return jnp.log(ddL_of_z(z, dL, H0, Om0)) + 2.0 * jnp.log1p(z)


def log_jacobian_detector_to_source_q(
    z: jnp.ndarray,
    dL: jnp.ndarray,
    m1src: jnp.ndarray,
    H0: jnp.ndarray,
    Om0: jnp.ndarray,
) -> jnp.ndarray:
    """
    Log-Jacobian for ``(z, m1_src, q) → (dL, m1_det, m2_det)``.

    Both PE and selection terms use samples represented as detector-frame
    ``(m1_det, m2_det, dL)`` but evaluate population densities in
    source-frame ``(m1_src, q, z)``.  The mass-coordinate determinant adds
    an ``m1_src`` factor on top of the usual luminosity-distance and
    redshifted-mass factors.

    Returns
    -------
    log |J| = log d(dL)/dz + 2 log(1+z) + log(m1_src)
    """
    return log_jacobian_dL_to_z(z, dL, H0, Om0) + jnp.log(m1src)


def log_sample_weight(
    m1det: jnp.ndarray,
    q: jnp.ndarray,
    dL: jnp.ndarray,
    chieff: jnp.ndarray,
    pix: jnp.ndarray,
    prior_wt: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    log_p_pop_fn,
    log_prior_z_fn,
) -> jnp.ndarray:
    """
    Per-sample log importance weight, shared by the PE and selection terms.

    The importance weight reweights samples drawn from ``prior_wt``
    (the PE proposal or the injection draw distribution) to the
    population-plus-cosmology model:

        log w = log p_pop(m1_src, q, z, chi_eff | λ)
              + log p_z(z | pix, Θ)
              - log |J(detector → source, q)|  ← change of variables
              - log p_draw(sample)             ← proposal density

    Parameters
    ----------
    m1det : detector-frame primary mass [M_sun]
    q : mass ratio m2/m1, pre-computed at event construction
    dL : luminosity distance [Mpc]
    chieff : effective inspiral spin
    pix : HEALPix pixel index
    prior_wt : PE prior weight / injection draw probability at this sample
    cosmo : CosmoParams
    survey : SurveyParams
    pop_params : flat parameter vector for the population model
    catalog : EMCatalog (PE catalog or selection catalog)
    log_p_pop_fn : callable(m1_src, q, z, chieff, pop_params) → log probability
    log_prior_z_fn : callable(z, pix, catalog) → log probability
        Should already incorporate the finite-value guard (replace -inf → -1e6).

    Returns
    -------
    log w : scalar or array matching the shape of the inputs
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    z     = z_of_dL(dL, H0, Om0)
    m1src = m1det / (1.0 + z)

    return (
        log_p_pop_fn(m1src, q, z, chieff, pop_params)
        + log_prior_z_fn(z, pix, catalog)
        - log_jacobian_detector_to_source_q(z, dL, m1src, H0, Om0)
        - jnp.log(prior_wt)
    )
