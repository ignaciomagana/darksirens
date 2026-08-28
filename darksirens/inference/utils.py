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

Integration variables
---------------------
The likelihood samples are stored as detector-frame component masses and
luminosity distance, but the hot path uses the derived mass ratio
``q = m2det / m1det``.  Therefore both posterior samples and detected
injections are integrated in the same coordinates

    (m1det, q, dL, chieff, sky pixel),

not in ``(m1det, m2det, dL)``.  Any proposal density divided out by the
likelihood (``p_pe`` for posterior samples or ``p_draw`` for injections)
must be expressed per unit ``m1det``, per unit ``q``, per Mpc of ``dL``
(and per unit ``chieff``/steradian when those factors are part of the
proposal).  A density native to ``(m1det, m2det, dL)`` is converted to the
canonical ``(m1det, q, dL)`` basis by multiplying by
``|dm2det/dq| = m1det``.

The Jacobian
------------
The population model is a density in source-frame variables
``(m1src, q, z)``.  For the canonical sample coordinates
``(m1det, q, dL)`` the change of variables is

    m1det = (1 + z) m1src,
    q     = q,
    dL    = dL(z),

so

    |∂(m1det, q, dL) / ∂(m1src, q, z)|
        = (1 + z) * d(dL)/dz.

In log space the target density in sample coordinates subtracts
``log ddL_of_z + log(1+z)``.  This is the only place in the codebase
where this canonical likelihood Jacobian is computed.  Do not inline it
elsewhere.
"""

from __future__ import annotations

import jax.numpy as jnp

from darksirens.utils.cosmology import z_of_dL, z_of_dL_precomputed, ddL_of_z, ddL_of_z_precomputed
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog


M1DET_Q_DL_COORDS = "m1det_q_dL"


def log_jacobian_m1src_q_z_to_m1det_q_dL(
    z: jnp.ndarray,
    dL: jnp.ndarray,
    H0: jnp.ndarray,
    Om0: jnp.ndarray,
    w0: jnp.ndarray = -1.0,
    wa: jnp.ndarray = 0.0,
    ddL_grid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Log-Jacobian for ``(m1src, q, z) → (m1det, q, dL)``.

    Parameters
    ----------
    z : redshift at the sample point
    dL : luminosity distance [Mpc] at the sample point
    H0, Om0, w0, wa : cosmological parameters
    ddL_grid : optional precomputed ``ddL_of_z(zgrid, dL_grid, H0, Om0, w0, wa)``
        array.  When provided, the per-sample ``E(z)`` evaluation inside
        ``ddL_of_z`` is replaced by a cheap 1-D interpolation.

    Returns
    -------
    log |J| = log d(dL)/dz + log(1+z)
    """
    if ddL_grid is not None:
        return jnp.log(ddL_of_z_precomputed(z, ddL_grid)) + jnp.log1p(z)
    return jnp.log(ddL_of_z(z, dL, H0, Om0, w0, wa)) + jnp.log1p(z)


# Backwards-compatible name for callers that imported the old helper.  The
# likelihood convention is now explicitly the canonical ``(m1det, q, dL)``
# basis documented above.
def log_jacobian_dL_to_z(
    z: jnp.ndarray,
    dL: jnp.ndarray,
    H0: jnp.ndarray,
    Om0: jnp.ndarray,
    w0: jnp.ndarray = -1.0,
    wa: jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """Alias for the canonical ``(m1src, q, z) → (m1det, q, dL)`` Jacobian."""
    return log_jacobian_m1src_q_z_to_m1det_q_dL(z, dL, H0, Om0, w0, wa)


def log_target_density_m1det_q_dL(
    m1det: jnp.ndarray,
    q: jnp.ndarray,
    dL: jnp.ndarray,
    chieff: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    log_p_pop_fn,
    log_prior_z_fn,
    spin: jnp.ndarray | None = None,
    dL_grid: jnp.ndarray | None = None,
    ddL_grid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Target density evaluated in the canonical sample basis.

    Returns ``log p(m1det, q, dL, chieff, pix | λ, Θ)`` when the
    source-frame population model supplies ``log p_pop(m1src, q, z,
    chieff | λ)`` and the EM term supplies ``log p_z(z | pix, Θ)``.

    ``spin`` is the optional (N, d) extra-spin block (GWEvent.spin).  It is
    forwarded to ``log_p_pop_fn`` as a keyword ONLY when present, so every
    existing chi_eff population model (signature ``(m1src, q, z, chieff,
    pop_params)``) is called exactly as before -- the d = 0 path is
    byte-identical by construction.

    ``dL_grid`` is the optional precomputed ``dL_of_z(zgrid, H0, Om0, w0,
    wa)`` array.  When provided, the ``z_of_dL`` inversion skips the
    redundant 4-D table lookup.

    ``ddL_grid`` is the optional precomputed ``ddL_of_z(zgrid, dL_grid, H0,
    Om0, w0, wa)`` array.  When provided, the per-sample ``E(z)`` evaluation
    inside the Jacobian is replaced by a 1-D interpolation.
    """
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    if dL_grid is not None:
        z = z_of_dL_precomputed(dL, dL_grid)
    else:
        z = z_of_dL(dL, H0, Om0, w0, wa)
    m1src = m1det / (1.0 + z)

    if spin is None:
        log_p_pop = log_p_pop_fn(m1src, q, z, chieff, pop_params)
    else:
        log_p_pop = log_p_pop_fn(m1src, q, z, chieff, pop_params, spin=spin)
    return (
        log_p_pop
        + log_prior_z_fn(z, pix, catalog)
        - log_jacobian_m1src_q_z_to_m1det_q_dL(z, dL, H0, Om0, w0, wa, ddL_grid=ddL_grid)
    )


def log_target_density_base_and_z(
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
    spin: jnp.ndarray | None = None,
    dL_grid: jnp.ndarray | None = None,
    ddL_grid: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Member-INDEPENDENT split of :func:`log_sample_weight`.

    Returns ``(base, z)`` where

        base = log p_pop(m1src, q, z, chieff | λ)
             - log[d(dL)/dz] - log(1+z)
             - log(prior_wt)

    is everything :func:`log_sample_weight` computes EXCEPT the redshift-prior
    contribution ``log_prior_z_fn(z, pix, catalog)``, and ``z = z_of_dL(dL)`` is
    the redshift it derived.  The population term, Jacobian, and proposal
    reweighting are independent of the LSS-completion member index; the ONLY
    member-dependent piece of the per-sample weight is ``log p_z(z | pix, Θ)``,
    which is additively separable:

        log_sample_weight(...) = base + log_prior_z_fn(z, pix, catalog).

    The factored LSS-marginalization path (``likelihood/core.py``) precomputes
    ``base`` once and adds the member-dependent mixture prior per member, so the
    O(N_max_gals) catalog-KDE work is not redone M times.  ``pix``, ``survey``
    and ``catalog`` are accepted for signature parity with
    :func:`log_target_density_m1det_q_dL`; only the population model, cosmology,
    and ``prior_wt`` enter ``base``.

    The arithmetic mirrors :func:`log_target_density_m1det_q_dL` /
    :func:`log_sample_weight` with the prior-z term removed; ``base + log p_z``
    reproduces the monolithic weight up to floating-point re-association only.

    ``dL_grid`` is the optional precomputed ``dL_of_z(zgrid, H0, Om0, w0,
    wa)`` array.  When provided, the ``z_of_dL`` inversion skips the
    redundant 4-D table lookup.

    ``ddL_grid`` is the optional precomputed ``ddL_of_z(zgrid, dL_grid, H0,
    Om0, w0, wa)`` array.  When provided, the per-sample ``E(z)`` evaluation
    inside the Jacobian is replaced by a 1-D interpolation.
    """
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    if dL_grid is not None:
        z = z_of_dL_precomputed(dL, dL_grid)
    else:
        z = z_of_dL(dL, H0, Om0, w0, wa)
    m1src = m1det / (1.0 + z)
    if spin is None:
        log_p_pop = log_p_pop_fn(m1src, q, z, chieff, pop_params)
    else:
        log_p_pop = log_p_pop_fn(m1src, q, z, chieff, pop_params, spin=spin)
    base = (
        log_p_pop
        - log_jacobian_m1src_q_z_to_m1det_q_dL(z, dL, H0, Om0, w0, wa, ddL_grid=ddL_grid)
        - jnp.log(prior_wt)
    )
    return base, z


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
    spin: jnp.ndarray | None = None,
    dL_grid: jnp.ndarray | None = None,
    ddL_grid: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Per-sample log importance weight, shared by the PE and selection terms.

    Both posterior samples and detected injections use the same canonical
    integration variables ``(m1det, q, dL)``.  The importance weight
    reweights samples drawn from ``prior_wt`` (the PE proposal or the
    injection draw distribution) to the population-plus-cosmology model:

        log w = log p_target(m1det, q, dL, chi_eff, pix | λ, Θ)
              - log p_proposal(m1det, q, dL, chi_eff, pix)

    with

        log p_target = log p_pop(m1src, q, z, chi_eff | λ)
                     + log p_z(z | pix, Θ)
                     - log[d(dL)/dz]
                     - log(1+z).

    Parameters
    ----------
    m1det : detector-frame primary mass [M_sun]
    q : mass ratio m2det/m1det, pre-computed at event construction
    dL : luminosity distance [Mpc]
    chieff : effective inspiral spin
    pix : HEALPix pixel index
    prior_wt : PE prior weight / injection draw probability in the canonical
        ``(m1det, q, dL)`` basis at this sample
    cosmo : CosmoParams
    survey : SurveyParams
    pop_params : flat parameter vector for the population model
    catalog : EMCatalog (PE catalog or selection catalog)
    log_p_pop_fn : callable(m1_src, q, z, chieff, pop_params) → log probability
    log_prior_z_fn : callable(z, pix, catalog) → log probability

    Returns
    -------
    log w : scalar or array matching the shape of the inputs
    """
    return (
        log_target_density_m1det_q_dL(
            m1det,
            q,
            dL,
            chieff,
            pix,
            cosmo,
            survey,
            pop_params,
            catalog,
            log_p_pop_fn,
            log_prior_z_fn,
            spin=spin,
            dL_grid=dL_grid,
            ddL_grid=ddL_grid,
        )
        - jnp.log(prior_wt)
    )
