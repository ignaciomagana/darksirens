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
is degenerate (a delta at μ=1).  The generic ln-μ quadrature above is
unstable there (s = 0 in ``norm.logpdf``), so the lognormal backend runs
the Hermite kernel below instead: in the standardized variable u the
s → 0 limit is exact by construction — every node collapses to μ = 1 and
the Hermite weights sum to 1 — so ``wl_a == 0`` reproduces the
unmarginalized ``log_sample_weight`` to round-off, which is what the
reduction tests pin.  The ``jnp.sqrt(s2)`` there carries a double-``where``
so the REVERSE pass is finite at s2 = 0 as well (see the kernel).
The dispatcher itself branches only on ``wl_enabled``; there is no
``wl_a == 0`` short-circuit.

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

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import (
    z_of_dL,
    z_of_dL_precomputed,
    dL_of_z,
    dL_in_z_grid,
    zgrid,
    Om0Planck,
    w0Fiducial,
    waFiducial,
)
from darksirens.inference.utils import (
    log_sample_weight,
    log_jacobian_m1src_q_z_to_m1det_q_dL,
)

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


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
    spin: jnp.ndarray | None = None,
    dL_grid: jnp.ndarray | None = None,
    ddL_grid: jnp.ndarray | None = None,
    log_prior_wt: jnp.ndarray | None = None,
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
    if dL_grid is not None:
        in_grid = (dL_true >= dL_grid[0]) & (dL_true <= dL_grid[-1])
        z_s = z_of_dL_precomputed(dL_true, dL_grid)
    else:
        in_grid = dL_in_z_grid(dL_true, H0, Om0, w0, wa)               # (..., Nmu)
        z_s = z_of_dL(dL_true, H0, Om0, w0, wa)                        # (..., Nmu)
    # Replace NaN with a finite dummy so downstream ops don't NaN-poison
    # the logsumexp; we'll mask with -inf at the end.
    z_s_safe = jnp.where(in_grid, z_s, 0.5)                    # arbitrary finite value
    m1src = m1det_b / (1.0 + z_s_safe)                         # (..., Nmu)

    # Physics pieces evaluated on the μ-grid
    if spin is None:
        log_pp = log_p_pop_fn(m1src, q_b, z_s_safe, chieff_b, pop_params)  # (..., Nmu)
    else:
        # (N, d) spin block broadcast against the mu-node axis: the extra
        # spin coordinates are z-independent, so each node sees the same row.
        log_pp = log_p_pop_fn(m1src, q_b, z_s_safe, chieff_b, pop_params,
                              spin=spin[:, None, :])
    log_pz  = log_prior_z_fn(z_s_safe.reshape(-1),
                             pix_b.reshape(-1),
                             catalog).reshape(z_s_safe.shape)            # (..., Nmu)
    log_pwl = log_p_wl_fn(mu_b, z_s_safe)                                # (..., Nmu)
    log_J   = log_jacobian_m1src_q_z_to_m1det_q_dL(
        z_s_safe, dL_true, H0, Om0, w0, wa, ddL_grid=ddL_grid,
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
    log_pw = log_prior_wt if log_prior_wt is not None else jnp.log(prior_wt)
    log_pPE = jnp.where(valid, log_pw, 0.0)
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
    spin: jnp.ndarray | None = None,
    dL_grid: jnp.ndarray | None = None,
    ddL_grid: jnp.ndarray | None = None,
    log_prior_wt: jnp.ndarray | None = None,
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
            spin=spin, dL_grid=dL_grid, ddL_grid=ddL_grid,
            log_prior_wt=log_prior_wt,
        )
    return log_sample_weight_wl_marginalized(
        m1det, q, dL, chieff, pix, prior_wt,
        cosmo, survey, pop_params, catalog,
        log_p_pop_fn, log_prior_z_fn,
        log_p_wl_fn, mu_nodes, log_w_nodes,
        spin=spin, dL_grid=dL_grid, ddL_grid=ddL_grid,
        log_prior_wt=log_prior_wt,
    )


# ============================================================================
# Lognormal-specialized Hermite-Gauss marginalization (robust at small s)
# ============================================================================

def _hermite_mu_geometry_and_log_ratio(
    dL: jnp.ndarray,
    wl_a: jnp.ndarray,
    wl_b: jnp.ndarray,
    u_nodes: jnp.ndarray,
    H0, Om0, w0, wa,
    dL_grid: jnp.ndarray | None = None,
) -> tuple:
    """Node geometry + proposal→target importance ratio of the Hermite kernel.

    Shared by ``log_sample_weight_wl_lognormal_hermite`` and the startup
    convergence self-check ``wl_hermite_quadrature_errors`` so the check
    provably exercises the exact algebra the likelihood integrates on.

    Returns ``(log_mu, mu, dL_true, in_grid, z_s_safe, log_ratio)``, each of
    shape ``(..., Nu)`` (broadcast of ``dL`` against the node axis).
    """
    # Apparent z (μ=1) sets the lognormal scale
    if dL_grid is not None:
        z_app = z_of_dL_precomputed(dL, dL_grid)
    else:
        z_app = z_of_dL(dL, H0, Om0, w0, wa)                    # (...,)
    z_app_safe = jnp.maximum(z_app, 1.0e-3)                     # avoid s=0 at z=0
    s2 = wl_a * jnp.power(z_app_safe, wl_b)                     # (...,)
    # Double-where so the reverse pass is finite at wl_a == 0 (the advertised
    # "reduces to standard" ablation).  The VALUE at s2 = 0 is already right
    # (every node collapses to mu = 1 and the Hermite weights sum to 1), but
    # d sqrt(s2) / d s2 = inf there, and ds2/dz_app = 0 at wl_a = 0, so the
    # unguarded chain returns inf * 0 = NaN for every cosmology gradient
    # through z_app.
    s2_pos = s2 > 0.0
    s  = jnp.where(s2_pos, jnp.sqrt(jnp.where(s2_pos, s2, 1.0)), 0.0)
    m  = -0.5 * s2

    # u → ln μ → μ
    u_b = u_nodes                                                # (Nu,)
    log_mu = (m[..., None] + s[..., None] * u_b)                # (..., Nu)
    mu = jnp.exp(log_mu)

    dL_true = dL[..., None] * jnp.sqrt(mu)                       # (..., Nu)
    if dL_grid is not None:
        in_grid = (dL_true >= dL_grid[0]) & (dL_true <= dL_grid[-1])
        z_s = z_of_dL_precomputed(dL_true, dL_grid)
    else:
        in_grid = dL_in_z_grid(dL_true, H0, Om0, w0, wa)
        z_s = z_of_dL(dL_true, H0, Om0, w0, wa)
    z_s_safe = jnp.where(in_grid, z_s, 0.5)

    # Proposal -> target density ratio (see the kernel docstring).  Without it
    # the Hermite backend integrates p_WL(mu | z_app), not the stated
    # p_WL(mu | z_s(mu)) the generic backend uses, and the two backends of the
    # same quantity disagree.  The lognormal at the NODE's source redshift,
    # with the same z clamp make_lognormal_log_p_wl applies:
    z_s_clamped = jnp.maximum(z_s_safe, 1.0e-3)
    s2_s = wl_a * jnp.power(z_s_clamped, wl_b)
    # Same double-where as s above: at wl_a == 0 both widths are exactly zero,
    # every node sits at mu = 1, and the ratio must be 0 with a finite reverse
    # pass rather than 0 * inf.
    s2_s_pos = s2_s > 0.0
    s2_s_safe = jnp.where(s2_s_pos, s2_s, 1.0)
    m_s = -0.5 * s2_s
    log_ratio = jnp.where(
        s2_s_pos & s2_pos[..., None],
        0.5 * jnp.log(jnp.where(s2_pos, s2, 1.0))[..., None]   # + log s_app
        - 0.5 * jnp.log(s2_s_safe)                             # - log s_s
        - jnp.square(log_mu - m_s) / (2.0 * s2_s_safe)
        + 0.5 * jnp.square(u_b),                               # proposal exponent
        0.0,
    )
    return log_mu, mu, dL_true, in_grid, z_s_safe, log_ratio


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
    spin: jnp.ndarray | None = None,
    dL_grid: jnp.ndarray | None = None,
    ddL_grid: jnp.ndarray | None = None,
    log_prior_wt: jnp.ndarray | None = None,
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

    Proposal vs target: the density ratio
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When ``z`` varies with ``μ`` through z(dL_app · √μ), the lognormal's
    z-dependence ``s(z(μ))`` makes ``u(μ; z)`` itself μ-dependent.  The
    nodes are therefore drawn from the PROPOSAL

        q(μ) = p_WL(μ | z_app),      z_app = z(dL_app),

    which is not the stated target's ``p_WL(μ | z_s(μ))``.  Evaluating the
    rest of the integrand at z_s while the measure stays at z_app
    integrates the wrong density: with a constant source weight the
    quadrature returns exactly 1, whereas direct quadrature of the stated
    target returns 0.999822, 0.999517, 0.998712 and 0.996507 at apparent
    z = 0.5, 1, 2, 4 for the default a = 0.004, b = 1.5.  That is the whole
    normalization error, and nonconstant population/volume factors move it
    off the small numbers above.

    So each node carries the importance weight

        log p_WL(μ | z_s(μ)) − log p_WL(μ | z_app)
            = log s_app − log s_s − (ln μ − m_s)² / (2 s_s²) + u²/2,

    with s_s = s(z_s(μ)), m_s = −s_s²/2, and the −ln μ terms cancelling
    between the two lognormals.  The identity ln μ = m_app + s_app·u is
    what turns the proposal's exponent into u²/2.  At ``wl_a == 0`` both
    widths vanish, every node collapses to μ = 1, and the ratio is
    identically zero — the advertised reduction to the unmarginalized
    weight is untouched.

    Convergence domain
    ~~~~~~~~~~~~~~~~~~
    The importance ratio's exponent grows like ``(u²/2)(1 − s_app²/s_s²)``,
    which is POSITIVE for u > 0 whenever ``wl_b > 0`` (s grows with z and
    z_s(μ) grows along u), so the effective Gauss-Hermite integrand is
    super-Gaussian and node-count convergence is NOT guaranteed for
    variance amplitudes well above the calibrated ``a ≈ 4e-3`` — the error
    can even grow with more nodes.  ``validate_wl_hermite_quadrature``
    below checks the actual (a, b) against a dense reference quadrature at
    startup; the lensing CLI runs it for the lognormal backend the way the
    tabulated backend runs ``validate_wl_mu_quadrature``.
    """
    # Full CPL: dropping w0/wa here ran the WL mu-marginalisation in LambdaCDM
    # while the surrounding likelihood (clamp bounds, volume prior, pop z
    # terms) used the sampled w0/wa (library review, likelihood finding 2 /
    # lensing finding 5): internally inconsistent for any --fix_de false
    # WL run.
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    u_b = u_nodes                                                # (Nu,)
    log_w_b = log_wH_nodes                                       # (Nu,)
    log_mu, mu, dL_true, in_grid, z_s_safe, log_ratio = (
        _hermite_mu_geometry_and_log_ratio(dL, wl_a, wl_b, u_b, H0, Om0, w0, wa,
                                           dL_grid=dL_grid)
    )

    dL_b = dL[..., None]
    m1det_b = m1det[..., None]
    q_b = q[..., None]
    chieff_b = chieff[..., None]
    pix_b = jnp.broadcast_to(pix[..., None], dL_b.shape[:-1] + (u_b.shape[0],)).astype(pix.dtype)
    m1src = m1det_b / (1.0 + z_s_safe)

    if spin is None:
        log_pp = log_p_pop_fn(m1src, q_b, z_s_safe, chieff_b, pop_params)
    else:
        log_pp = log_p_pop_fn(m1src, q_b, z_s_safe, chieff_b, pop_params,
                              spin=spin[:, None, :])
    log_pz = log_prior_z_fn(z_s_safe.reshape(-1),
                            pix_b.reshape(-1), catalog).reshape(z_s_safe.shape)
    log_J  = log_jacobian_m1src_q_z_to_m1det_q_dL(z_s_safe, dL_true, H0, Om0, w0, wa, ddL_grid=ddL_grid)

    # Per-node log-integrand: NO extra +log μ from substitution
    # (Hermite quadrature substitution carries the WL PDF as measure).
    log_integrand = (
        log_w_b
        + log_pp
        + log_pz
        + log_ratio                   # proposal q(mu|z_app) -> target p(mu|z_s)
        - log_J
        + 0.5 * log_mu                # physics √μ Jacobian only
    )
    log_integrand = jnp.where(in_grid & jnp.isfinite(log_integrand),
                              log_integrand, -jnp.inf)
    log_target = logsumexp(log_integrand, axis=-1)

    valid = prior_wt > 0.0
    log_pw = log_prior_wt if log_prior_wt is not None else jnp.log(prior_wt)
    log_pPE = jnp.where(valid, log_pw, 0.0)
    return jnp.where(valid, log_target - log_pPE, -jnp.inf)


# ============================================================================
# Startup convergence self-check for the Hermite importance-ratio quadrature
# ============================================================================

def wl_hermite_quadrature_errors(
    wl_a: float,
    wl_b: float,
    u_nodes: jnp.ndarray,
    log_wH_nodes: jnp.ndarray,
    z_app_test: np.ndarray | None = None,
    H0: float = 70.0,
    Om0: float = Om0Planck,
    w0: float = w0Fiducial,
    wa: float = waFiducial,
) -> tuple:
    """``|Δ log I|`` of the Hermite rule vs a dense reference, per test z.

    ``I(dL_app) = ∫ p_WL(μ | z_s(μ)) √μ dμ`` over the in-grid μ range is the
    kernel's pure-quadrature content (constant source weight: population,
    volume and Jacobian factors stripped).  The Hermite estimate is
    ``logsumexp(log_wH + log_ratio + ½ log μ)`` through the exact production
    algebra (``_hermite_mu_geometry_and_log_ratio``); the reference is a
    dense trapezoid in ``ln μ`` of the identical masked target.  Their
    difference is the per-sample log-weight error the importance-ratio
    quadrature commits at that apparent redshift.

    Test redshifts default to fractions of ``min(3.5, 0.7·z_grid_max)``:
    high enough to expose the super-Gaussian ratio growth and the z-grid
    edge truncation that break node-count convergence at amplified
    ``wl_a``, while the calibrated default (a = 4e-3) stays ≲ 1e-6 nats
    everywhere on this range.  Expressed in ``z_app`` the check is exactly
    H0-invariant (dL ∝ 1/H0 cancels between the node distances and the
    grid edge) and only weakly Om0/w0/wa-dependent.

    Returns ``(z_app_test, errors)`` as numpy arrays.
    """
    wl_a = float(wl_a)
    wl_b = float(wl_b)
    z_hi = float(zgrid[-1])
    if z_app_test is None:
        z_cap = min(3.5, 0.7 * z_hi)
        z_app_test = np.array([0.15, 0.3, 0.5, 0.75, 1.0]) * z_cap
    z_app_test = np.atleast_1d(np.asarray(z_app_test, dtype=np.float64))
    if wl_a <= 0.0:
        # Delta at mu = 1: every node collapses, the Hermite weights sum to 1
        # and the rule is exact by construction.
        return z_app_test, np.zeros_like(z_app_test)

    dL_test = np.asarray(
        dL_of_z(jnp.asarray(z_app_test), H0, Om0, w0, wa), dtype=np.float64
    )

    # Hermite estimate through the production kernel algebra.
    log_mu, _mu, _dL_true, in_grid, _z_s_safe, log_ratio = (
        _hermite_mu_geometry_and_log_ratio(
            jnp.asarray(dL_test), jnp.asarray(wl_a), jnp.asarray(wl_b),
            u_nodes, H0, Om0, w0, wa,
        )
    )
    log_integrand = log_wH_nodes + log_ratio + 0.5 * log_mu
    log_integrand = jnp.where(in_grid & jnp.isfinite(log_integrand),
                              log_integrand, -jnp.inf)
    log_I_hermite = np.asarray(logsumexp(log_integrand, axis=-1))

    # Dense reference: trapezoid in x = ln mu of the identical masked target
    #   N(x; m_s(x), s_s(x)) e^{x/2},  m_s = -s_s^2/2,  s_s^2 = a z_s(x)^b,
    # with the upper limit placed EXACTLY at the z-grid edge so the in-grid
    # mask is an endpoint, not an interior step the trapezoid would smear.
    dL_max = float(np.asarray(dL_of_z(jnp.asarray(z_hi), H0, Om0, w0, wa)))
    s_hi = np.sqrt(wl_a * max(z_hi, 1.0e-3) ** wl_b)
    errors = np.empty_like(z_app_test)
    for i in range(z_app_test.shape[0]):
        z_a = float(z_app_test[i])
        dL_a = float(dL_test[i])
        s2_app = wl_a * max(z_a, 1.0e-3) ** wl_b
        s_app = np.sqrt(s2_app)
        m_app = -0.5 * s2_app
        half = 15.0 * max(s_app, float(s_hi))
        x_lo = m_app - half
        x_hi = min(m_app + half, 2.0 * np.log(dL_max / dL_a))
        # ~300 points per proposal sigma: trapezoid error ~1e-6 nats, far
        # below the validation tolerance.  Capped for pathological a, b.
        n = int(min(2_000_001, max(20_001, round((x_hi - x_lo) / (s_app / 300.0)))))
        x = np.linspace(x_lo, x_hi, n)
        z_s = np.asarray(z_of_dL(jnp.asarray(dL_a * np.exp(0.5 * x)),
                                 H0, Om0, w0, wa))
        ok = np.isfinite(z_s)
        z_c = np.maximum(np.where(ok, z_s, 1.0), 1.0e-3)
        s2_s = wl_a * np.power(z_c, wl_b)
        log_f = (
            -0.5 * np.log(2.0 * np.pi * s2_s)
            - np.square(x + 0.5 * s2_s) / (2.0 * s2_s)
            + 0.5 * x
        )
        f = np.where(ok, np.exp(log_f), 0.0)
        errors[i] = abs(float(log_I_hermite[i]) - float(np.log(_trapezoid(f, x))))
    return z_app_test, errors


def validate_wl_hermite_quadrature(
    wl_a: float,
    wl_b: float,
    u_nodes: jnp.ndarray | None = None,
    log_wH_nodes: jnp.ndarray | None = None,
    z_app_test: np.ndarray | None = None,
    tol: float = 1.0e-4,
    context: str = "lognormal WL backend",
) -> np.ndarray:
    """Raise ``ValueError`` unless the Hermite rule converges for (a, b).

    The counterpart of ``darksirens.lensing.wlmagnification.
    validate_wl_mu_quadrature`` for the lognormal backend: checks
    ``|log I_hermite - log I_ref| <= tol`` (nats) at every test redshift,
    where I is the kernel's pure-quadrature integral (see
    ``wl_hermite_quadrature_errors``).  The importance ratio to
    p_WL(μ | z_s(μ)) is super-Gaussian for wl_b > 0, so amplified variance
    amplitudes (≳ 10× the calibrated a = 4e-3) silently break node-count
    convergence — the error does not shrink with more nodes.  Returns the
    per-redshift error array on success.
    """
    if u_nodes is None or log_wH_nodes is None:
        from darksirens.lensing.grids import make_hermite_u_grid
        u_nodes, log_wH_nodes = make_hermite_u_grid()
    z_test, err = wl_hermite_quadrature_errors(
        wl_a, wl_b, u_nodes, log_wH_nodes, z_app_test=z_app_test,
    )
    worst = int(np.argmax(np.where(np.isfinite(err), err, np.inf)))
    if not bool(np.all(np.isfinite(err))) or float(err[worst]) > float(tol):
        n_nodes = int(np.asarray(u_nodes).shape[0])
        raise ValueError(
            f"{context}: the {n_nodes}-node Gauss-Hermite importance-ratio "
            f"quadrature does not converge for wl_a = {wl_a:g}, "
            f"wl_b = {wl_b:g}. |log I_hermite - log I_ref| = "
            f"{float(err[worst]):.3g} nats at apparent z = "
            f"{float(z_test[worst]):.4g} (tolerance {float(tol):g}; "
            f"{int(np.sum(~np.isfinite(err) | (err > tol)))}/{int(err.shape[0])} "
            "test redshifts fail). The rule integrates p_WL(mu | z_s(mu)) "
            "through a proposal at z_app, and the importance ratio stays "
            "integrable only near the calibrated variance amplitude "
            "(a ~ 4e-3): in this regime per-event log-weights are silently "
            "wrong and do NOT improve with more nodes. Reduce "
            "--lensing_wl_a/--lensing_wl_b toward the calibrated values, or "
            "use --wl_backend tabulated with a table and mu-grid wide enough "
            "for the amplified variance."
        )
    return err
