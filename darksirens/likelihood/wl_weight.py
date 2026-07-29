"""
wl_weight.py
------------
WL-marginalized per-sample log-importance-weight.

This is the central new piece of commit 2: it implements the
marginalization over weak-lensing magnification inside the per-event PE
integral.  All other commit-2 changes (likelihood dispatch, prior
registration, CLI plumbing) are structural plumbing that routes data
to this function.

Mathematical specification
~~~~~~~~~~~~~~~~~~~~~~~~~~
The standard ``log_sample_weight`` in ``darksirens.inference.utils`` is:

    log w = log p_pop(m1_src, q, z, χ | λ)
          + log p_z(z | pix, Θ)        # p_vol for spectral sirens
          - log(1+z)
          - log [d(dL)/dz]
          - log p_PE

evaluated at the apparent z = z(d_L_app, H_0, Ω_m).

The WL extension treats μ as an additional latent variable and the
apparent dL as ``d_L_app = d_L(z_s) / sqrt(μ)``.  The Jacobian of
(m1_det, d_L_app) ← (m1_src, z_s) at fixed (q, χ, μ) is

    |J| = (1 + z_s) · dL'(z_s) / sqrt(μ),

so the *joint* log-target in apparent coordinates is

    log p_target^joint = log p_pop + log p_vol + log p_WL
                      - log(1+z_s) - log dL'(z_s) + (1/2) log μ.

Marginalizing over μ by Gauss-Legendre quadrature on ``ln μ`` with
nodes μ_ℓ = exp(x_ℓ) and weights W_ℓ:

    ∫ dμ exp[f(μ)] = ∫ exp[f(e^x)] e^x dx
                  ≈ Σ_ℓ W_ℓ exp[f(μ_ℓ)] μ_ℓ
                  = Σ_ℓ exp[log W_ℓ + f(μ_ℓ) + log μ_ℓ]

so in log-space the per-node integrand is

    log_integrand_ℓ = log_w_ℓ                        ← quadrature weight
                   + log p_pop(m1_src_ℓ, q, z_s_ℓ, χ | λ)
                   + log p_vol(z_s_ℓ)
                   + log p_WL(μ_ℓ | z_s_ℓ)
                   - log(1 + z_s_ℓ)
                   - log dL'(z_s_ℓ)
                   + (3/2) log μ_ℓ                    ← (1/2 physics + 1 quadrature)

where z_s_ℓ = z(dL_app · sqrt(μ_ℓ)) and m1_src_ℓ = m1_det / (1 + z_s_ℓ).

The full per-sample log-weight is then

    log w = -log p_PE + logsumexp_ℓ [log_integrand_ℓ].

Reduction to the standard hot path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When the lognormal variance parameter ``wl_a == 0``, the marginalization
is degenerate (a delta at μ=1) and the formula above is unstable
(s = 0 in ``norm.logpdf``).  This module short-circuits that case by
falling through to the unmarginalized ``log_sample_weight``.  This makes
the reduction test exact: ``wl_a == 0`` ⇒ identical numerical output to
``universe_model == "spectral_sirens"``.

Out-of-grid handling
~~~~~~~~~~~~~~~~~~~~
For any μ-node such that ``dL_app · sqrt(μ_ℓ)`` lies outside the
tabulated z(dL) grid, the corresponding ``z_s_ℓ`` returns NaN from
``z_of_dL``.  We mask those nodes with -inf so they contribute nothing
to the logsumexp; the surviving nodes carry the integral.  If *all*
nodes are out of grid, the sample is rejected (-inf), same as in the
standard hot path.

Selection
~~~~~~~~~
The default lensing CLI behavior still keeps singleton selection on the
standard ``log_sample_weight`` path for backward compatibility.  The lensing
cluster likelihood can optionally route singleton selection through the
lognormal/Hermite WL marginalization as well (``--wl_selection wl_lognormal``).
That option affects singleton injections only; the J=2 strong-lensing
lensed-injection selection estimator remains separate and unchanged.

References
~~~~~~~~~~
- Holz & Wald (1998), PRD 58, 063501.
- Mandel, Farr & Gair (2019), MNRAS 486, 1086, app. A.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import z_of_dL, dL_in_z_grid
from darksirens.inference.utils import (
    log_sample_weight,
    log_jacobian_m1src_q_z_to_m1det_q_dL,
)


def log_sample_weight_wl_marginalized(
    m1det: jnp.ndarray,
    q: jnp.ndarray,
    dL: jnp.ndarray,            # apparent dL (the data)
    chieff: jnp.ndarray,
    pix: jnp.ndarray,
    prior_wt: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    log_p_pop_fn: Callable,
    log_prior_z_fn: Callable,
    log_p_wl_fn: Callable,
    mu_nodes: jnp.ndarray,
    log_w_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """WL-marginalized per-sample log importance weight.

    Parameters
    ----------
    m1det, q, dL, chieff, pix, prior_wt
        Per-sample apparent quantities; same shape (broadcastable).
        ``dL`` is the *apparent* luminosity distance (the data), not the
        cosmological dL of the source.
    cosmo, survey, pop_params, catalog
        Standard hyperparameter containers.
    log_p_pop_fn
        ``log p_pop(m1_src, q, z, χ_eff, pop_params)`` callable.
    log_prior_z_fn
        ``log p_z(z, pix, catalog)`` callable. For ``spectral_sirens_wl``
        this is the comoving-volume prior.
    log_p_wl_fn
        JIT-friendly closure ``log p_WL(μ, z) -> log_p`` from
        ``darksirens.lensing.wlmagnification.make_lognormal_log_p_wl``
        (or its tabulated cousin).
    mu_nodes
        (Nmu,) Gauss-Legendre nodes in μ. Built once from
        ``darksirens.lensing.grids.make_log_mu_grid``.
    log_w_nodes
        (Nmu,) log of the Gauss-Legendre weights (including the (b-a)/2
        Jacobian for the ln-μ interval), as returned by the same builder.

    Returns
    -------
    log_w
        Scalar (or same shape as inputs minus trailing axis) per-sample
        log-weight. Returns -inf when all μ nodes are out of the
        cosmology grid support or when ``prior_wt <= 0``.

    Implementation notes
    --------------------
    All arrays are broadcast to shape (..., Nmu). Memory is therefore
    ``input_size × Nmu × 8 B``. For typical commit-2 use (nsamp=2000,
    Nmu=16, scalar input) this is 256 kB per event, comfortably fitting
    in cache.
    """
    # Full CPL: dropping w0/wa here ran the WL mu-marginalisation in LambdaCDM
    # while the surrounding likelihood (clamp bounds, volume prior, pop z
    # terms) used the sampled w0/wa (library review, likelihood finding 2 /
    # lensing finding 5): internally inconsistent for any --fix_de false
    # WL run.
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    # Broadcast: (..., Nmu)
    mu_b = mu_nodes                                            # (Nmu,)
    log_w_b = log_w_nodes                                      # (Nmu,)
    dL_b = dL[..., None]                                       # (..., 1)
    m1det_b = m1det[..., None]
    q_b = q[..., None]
    chieff_b = chieff[..., None]
    pix_b = jnp.broadcast_to(pix[..., None], dL_b.shape[:-1] + (mu_b.shape[0],)).astype(pix.dtype)

    # True (cosmological) dL for each μ node
    dL_true = dL_b * jnp.sqrt(mu_b)                            # (..., Nmu)
    in_grid = dL_in_z_grid(dL_true, H0, Om0, w0, wa)                   # (..., Nmu)

    # z_s on each μ node (NaN outside grid; we mask to -inf below)
    z_s = z_of_dL(dL_true, H0, Om0, w0, wa)                            # (..., Nmu)
    # Replace NaN with a finite dummy so downstream ops don't NaN-poison
    # the logsumexp; we'll mask with -inf at the end.
    z_s_safe = jnp.where(in_grid, z_s, 0.5)                    # arbitrary finite value
    m1src = m1det_b / (1.0 + z_s_safe)                         # (..., Nmu)

    # Physics pieces evaluated on the μ-grid
    log_pp  = log_p_pop_fn(m1src, q_b, z_s_safe, chieff_b, pop_params)  # (..., Nmu)
    log_pz  = log_prior_z_fn(z_s_safe.reshape(-1),
                             pix_b.reshape(-1),
                             catalog).reshape(z_s_safe.shape)            # (..., Nmu)
    log_pwl = log_p_wl_fn(mu_b, z_s_safe)                                # (..., Nmu)
    log_J   = log_jacobian_m1src_q_z_to_m1det_q_dL(
        z_s_safe, dL_true, H0, Om0, w0, wa
    )                                                                    # (..., Nmu)

    # Per-node log-integrand (BEFORE PE proposal subtraction)
    log_integrand = (
        log_w_b
        + log_pp
        + log_pz
        + log_pwl
        - log_J
        + 1.5 * jnp.log(mu_b)
    )                                                                    # (..., Nmu)

    # Mask out-of-grid nodes
    log_integrand = jnp.where(in_grid & jnp.isfinite(log_integrand),
                              log_integrand, -jnp.inf)

    # Marginalize over μ
    log_target = logsumexp(log_integrand, axis=-1)                       # (...)

    # Subtract PE proposal density
    valid = prior_wt > 0.0
    log_pPE = jnp.where(valid, jnp.log(prior_wt), 0.0)
    out = jnp.where(valid, log_target - log_pPE, -jnp.inf)
    return out


def log_sample_weight_wl_or_standard(
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
    log_p_pop_fn: Callable,
    log_prior_z_fn: Callable,
    log_p_wl_fn: Callable | None,
    mu_nodes: jnp.ndarray | None,
    log_w_nodes: jnp.ndarray | None,
    wl_enabled: bool,
) -> jnp.ndarray:
    """Dispatcher: WL-marginalize when ``wl_enabled`` is True, otherwise
    fall through to the standard ``log_sample_weight``.

    Use ``wl_enabled = False`` to obtain numerically identical output to
    the standard hot path — this is the basis of the reduction test.

    ``wl_enabled`` is intended to be a Python bool resolved at trace
    time, NOT a JAX array. Pass it as a static argument from the
    enclosing JIT'd function.
    """
    if not wl_enabled:
        return log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            log_p_pop_fn, log_prior_z_fn,
        )
    return log_sample_weight_wl_marginalized(
        m1det, q, dL, chieff, pix, prior_wt,
        cosmo, survey, pop_params, catalog,
        log_p_pop_fn, log_prior_z_fn,
        log_p_wl_fn, mu_nodes, log_w_nodes,
    )


# ============================================================================
# Lognormal-specialized Hermite-Gauss marginalization (robust at small s)
# ============================================================================

def log_sample_weight_wl_lognormal_hermite(
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
    log_p_pop_fn: Callable,
    log_prior_z_fn: Callable,
    wl_a: jnp.ndarray,
    wl_b: jnp.ndarray,
    u_nodes: jnp.ndarray,
    log_wH_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """Lognormal-specialized WL marginalization using Gauss-Hermite quadrature.

    Mathematical setup
    ~~~~~~~~~~~~~~~~~~
    For a lognormal WL PDF with mean μ=1 and variance s²(z) = a · z^b,
    introduce the standardized variable

        u = (ln μ - m(z)) / s(z),       m(z) = -s²(z)/2.

    Then ``p_WL(μ | z) dμ = N(u; 0, 1) du`` exactly (the substitution
    absorbs the entire WL PDF), and the integral

        ∫ dμ p_WL(μ|z) · F(μ, z(μ))
            = ∫ du N(u; 0, 1) · F(μ(u), z(μ(u)))

    is evaluated by Gauss-Hermite quadrature in ``u`` with FIXED nodes —
    no dependence on a or b. This eliminates the quadrature-resolution
    failure at small s that affects the generic
    ``log_sample_weight_wl_marginalized`` with ``make_log_mu_grid``.

    Per-node log-integrand
    ~~~~~~~~~~~~~~~~~~~~~~
    With z_s_ℓ = z(dL_app · √μ_ℓ), m1src_ℓ = m1det/(1+z_s_ℓ),

        log_integrand_ℓ = log w_ℓ^H               ← Hermite weight
                       + log p_pop(m1src_ℓ, ...)
                       + log p_vol(z_s_ℓ)
                       - log(1 + z_s_ℓ)
                       - log dL'(z_s_ℓ)
                       + (1/2) log μ_ℓ            ← physics √μ Jacobian only

    NOTE: there is NO ``+ log μ_ℓ`` from the substitution this time
    (compare the generic version's 3/2 coefficient). The Hermite
    substitution carries the entire WL PDF as the integration measure.

    A subtle case: when ``z`` varies with ``μ`` through z(dL_app · √μ),
    the lognormal's z-dependence ``s(z(μ))`` makes ``u(μ; z)`` itself
    μ-dependent. We use a **fixed** z = z(dL_app) (the apparent
    redshift) to set u, accepting an O(s²) error in the Jacobian — same
    order as the WL effect itself. For commit-2 accuracy this is fine.
    A future commit will iterate to self-consistency.
    """
    # Full CPL: dropping w0/wa here ran the WL mu-marginalisation in LambdaCDM
    # while the surrounding likelihood (clamp bounds, volume prior, pop z
    # terms) used the sampled w0/wa (library review, likelihood finding 2 /
    # lensing finding 5): internally inconsistent for any --fix_de false
    # WL run.
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    # Apparent z (μ=1) sets the lognormal scale
    z_app = z_of_dL(dL, H0, Om0, w0, wa)                                # (...,)
    z_app_safe = jnp.maximum(z_app, 1.0e-3)                     # avoid s=0 at z=0
    s2 = wl_a * jnp.power(z_app_safe, wl_b)                     # (...,)
    s  = jnp.sqrt(s2)
    m  = -0.5 * s2

    # u → ln μ → μ
    u_b = u_nodes                                                # (Nu,)
    log_w_b = log_wH_nodes                                       # (Nu,)
    log_mu = (m[..., None] + s[..., None] * u_b)                # (..., Nu)
    mu = jnp.exp(log_mu)

    dL_b = dL[..., None]
    m1det_b = m1det[..., None]
    q_b = q[..., None]
    chieff_b = chieff[..., None]
    pix_b = jnp.broadcast_to(pix[..., None], dL_b.shape[:-1] + (u_b.shape[0],)).astype(pix.dtype)

    dL_true = dL_b * jnp.sqrt(mu)                                # (..., Nu)
    in_grid = dL_in_z_grid(dL_true, H0, Om0, w0, wa)
    z_s = z_of_dL(dL_true, H0, Om0, w0, wa)
    z_s_safe = jnp.where(in_grid, z_s, 0.5)
    m1src = m1det_b / (1.0 + z_s_safe)

    log_pp = log_p_pop_fn(m1src, q_b, z_s_safe, chieff_b, pop_params)
    log_pz = log_prior_z_fn(z_s_safe.reshape(-1),
                            pix_b.reshape(-1), catalog).reshape(z_s_safe.shape)
    log_J  = log_jacobian_m1src_q_z_to_m1det_q_dL(z_s_safe, dL_true, H0, Om0, w0, wa)

    # Per-node log-integrand: NO extra +log μ from substitution
    # (Hermite quadrature substitution carries the WL PDF as measure).
    log_integrand = (
        log_w_b
        + log_pp
        + log_pz
        - log_J
        + 0.5 * log_mu                # physics √μ Jacobian only
    )
    log_integrand = jnp.where(in_grid & jnp.isfinite(log_integrand),
                              log_integrand, -jnp.inf)
    log_target = logsumexp(log_integrand, axis=-1)

    valid = prior_wt > 0.0
    log_pPE = jnp.where(valid, jnp.log(prior_wt), 0.0)
    return jnp.where(valid, log_target - log_pPE, -jnp.inf)
