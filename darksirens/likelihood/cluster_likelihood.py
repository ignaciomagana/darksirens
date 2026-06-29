"""
cluster_likelihood.py
---------------------
Cluster (J=2) pair likelihood for the joint BBH-population + strong-
lensing inference. Implements the SIS pair model with Janquart-style
KDE-on-PE estimator (Janquart, Haris, Hannuksela 2021;
Lo & Magaña Hernandez 2023).

Mathematical specification
~~~~~~~~~~~~~~~~~~~~~~~~~~
Given two GW events (d_i, d_j) declared a candidate J=2 cluster, with
SIS image magnifications μ_+(y) = (1+y)/y, μ_-(y) = (1-y)/y for impact
parameter y ∈ (0, 1) and p(y) = 2y, the pair likelihood is:

    L_2(d_i, d_j | λ, Θ) = (1/2) Σ_σ ∫ dy p(y) (1/N_i) Σ_s
        τ_2(z_s(s,y))
        · p_pop(θ_src(s,y) | λ)
        · p_z(z_s(s,y) | Θ)
        · p̂_j(θ_app_j_pred(s,y))
        · |J_app→src(z_s, dL_true, μ_σ(i)(y))|

where σ runs over the two image-assignment orderings (i→+,j→- and
i→-,j→+), s indexes event-i PE samples, and:

    dL_true(s,y)         = dL_app^(s) · √μ_σ(i)(y)
    z_s(s,y)             = z(dL_true)
    m_1_src(s,y)         = m_1_det^(s) / (1 + z_s)
    θ_src(s,y)           = (m_1_src, q^(s), z_s, χ_eff^(s))
    θ_app_j_pred(s,y)    = ((1 + z_s) m_1_src, q^(s),
                            dL(z_s) / √μ_σ(j)(y), χ_eff^(s))

The KDE p̂_j is built once per event in apparent-frame coordinates from
event-j's PE samples (see ``pair_kde.py``). The implied apparent-j
prediction is the standard "guess what event j would look like, given
event i's source-frame and the lens" calculation.

The Jacobian |J_app→src| is the inverse of commit 2's apparent-frame
Jacobian at fixed μ:
    log|J_app→src| = log(1 + z_s) + log dL'(z_s) - 0.5 log μ_σ(i).

Quadrature
~~~~~~~~~~
y is integrated by Gauss-Legendre on (0, 1) with 32 nodes (commit 1's
``make_y_grid``). p(y) = 2y is smooth, the integrand can have a soft
edge near y = 0 (μ → ∞) and y = 1 (μ_- → 0), both of which are mitigated
by the population prior strongly disfavouring extreme z_s.

Strong-lensing optical depth τ_2(z_s) uses ``slmarks.tau_2_SIS`` from
commit 1 (default A · z^n with A = 5×10⁻⁴, n = 3). T_0 / Δt are NOT
used in this commit — pure magnification-based pair likelihood. Time-
delay marks are deferred to commit 3.5.

Selection
~~~~~~~~~
**The cluster contribution to selection is NOT computed in this commit.**
This module returns log L_2 for a pair given hyperparameters, but the
proper rate-normalization (cluster-level expected count) requires the
augmented lensed-injection campaign of commit 4.

Two consequences:
  1. ``cluster_log_likelihood_pair`` returns the integrand to be COMBINED
     with the singleton log-likelihood and per-pair selection correction
     LATER, not the full log-evidence.
  2. The optical depth τ_2 appears INSIDE the integrand as a per-source
     rate suppression. The expected number of *observed* J=2 clusters
     also depends on detection efficiency at both magnifications, which
     this commit cannot estimate.

References
~~~~~~~~~~
- Janquart, Haris, Hannuksela, Van Den Broeck (2021), arXiv:2105.06384.
- Lo & Magaña Hernandez (2023), arXiv:2104.09339.
- Schneider, Kochanek & Wambsganss (2006), §8.6.
- Mandel, Farr & Gair (2019), MNRAS 486, 1086.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.scipy.special import logsumexp

from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import z_of_dL, dL_of_z, ddL_of_z, dL_in_z_grid
from darksirens.likelihood.pair_kde import PairKDE, log_eval_pair_kde
from darksirens.lensing.slmarks import (
    SISLensParams, tau_2_SIS, log_p_y_SIS,
    mu_plus_minus_from_y,
)
from darksirens.lensing.grids import make_y_grid


def _log_jac_app_to_src(z_s: jnp.ndarray,
                        dL_true: jnp.ndarray,
                        mu: jnp.ndarray,
                        H0: jnp.ndarray, Om0: jnp.ndarray) -> jnp.ndarray:
    """log|∂(m1src, q, z_s, χ) / ∂(m1det, q, dL_app, χ)|_at fixed μ.

    The map (m1det, dL_app) → (m1src, z_s) at fixed q, χ, μ has
    |J_app→src| = |J_src→app|^{-1}, so

        log|J_app→src| = -log(1 + z_s) - log dL'(z_s) + 0.5 log μ.

    Wait — that's the inverse Jacobian. Used here as a multiplier on
    the source-frame density to give the apparent-frame density at the
    PE-sample point. See cluster_likelihood derivation in module docstring.

    Returns
    -------
    log_J : same shape as inputs
        Signed log-Jacobian; ADD to log-integrand.
    """
    return -jnp.log1p(z_s) - jnp.log(ddL_of_z(z_s, dL_true, H0, Om0)) + 0.5 * jnp.log(mu)


def _pair_branch_log_integrand(
    # PE-i quantities (vectorized over samples)
    m1det_i: jnp.ndarray, q_i: jnp.ndarray, dL_app_i: jnp.ndarray, chieff_i: jnp.ndarray,
    prior_wt_i: jnp.ndarray, valid_i: jnp.ndarray, pix_i: jnp.ndarray,
    # Per-(s,y) magnification assignments
    mu_i: jnp.ndarray,  # μ assigned to event i along this branch, shape (N_y,)
    mu_j: jnp.ndarray,  # μ assigned to event j along this branch, shape (N_y,)
    log_py: jnp.ndarray,  # log p(y) at the y-nodes, shape (N_y,)
    log_wy: jnp.ndarray,  # log Gauss-Legendre weights at y-nodes, shape (N_y,)
    # Event-j KDE
    kde_j: PairKDE,
    # Hyperparameters
    cosmo: CosmoParams, survey: SurveyParams, pop_params: jnp.ndarray, catalog: EMCatalog,
    sis_params: SISLensParams,
    log_p_pop_fn: Callable, log_prior_z_fn: Callable,
) -> jnp.ndarray:
    """Compute the per-PE-sample log-integrand over y for one image-assignment branch.

    Returns
    -------
    log_per_sample : (N_pe_i, N_y) — log of the integrand contribution
        from each (PE-i sample, y-node) cell, BEFORE summing over y or s.
    """
    H0, Om0 = cosmo.H0, cosmo.Om0

    # Broadcast: (N_pe_i, N_y)
    mu_i_b = mu_i[None, :]                            # (1, N_y)
    mu_j_b = mu_j[None, :]                            # (1, N_y)
    dL_app_i_b = dL_app_i[:, None]                    # (N_pe_i, 1)
    m1det_i_b = m1det_i[:, None]
    q_i_b = q_i[:, None]
    chieff_i_b = chieff_i[:, None]

    # Map event-i PE sample s through assigned magnification mu_i to source frame
    dL_true_ij = dL_app_i_b * jnp.sqrt(mu_i_b)        # (N_pe_i, N_y)
    in_grid = dL_in_z_grid(dL_true_ij, H0, Om0)
    z_s = z_of_dL(dL_true_ij, H0, Om0)
    z_s_safe = jnp.where(in_grid, z_s, 0.5)           # finite dummy when out-of-grid
    m1src_ij = m1det_i_b / (1.0 + z_s_safe)           # (N_pe_i, N_y)

    # Predict event-j apparent parameters given same source-frame θ_s and μ_j
    dL_src_at_zs = dL_of_z(z_s_safe, H0, Om0)
    dL_app_j_pred = dL_src_at_zs / jnp.sqrt(mu_j_b)   # (N_pe_i, N_y)
    m1det_j_pred = (1.0 + z_s_safe) * m1src_ij        # (N_pe_i, N_y), algebraically = m1det_i_b
    # Broadcast (N_pe_i, 1) → (N_pe_i, N_y) for q and chieff
    target_shape = m1det_j_pred.shape
    q_j_pred = jnp.broadcast_to(q_i_b, target_shape)
    chieff_j_pred = jnp.broadcast_to(chieff_i_b, target_shape)

    # Stack into (N_pe_i, N_y, 4) for KDE evaluation
    theta_app_j_pred = jnp.stack([
        m1det_j_pred, q_j_pred, dL_app_j_pred, chieff_j_pred,
    ], axis=-1)                                       # (N_pe_i, N_y, 4)
    log_p_j = log_eval_pair_kde(kde_j, theta_app_j_pred)  # (N_pe_i, N_y)

    # Population & cosmological prior at source-frame θ_s
    log_pp = log_p_pop_fn(m1src_ij, q_i_b, z_s_safe, chieff_i_b, pop_params)
    log_pz = log_prior_z_fn(
        z_s_safe.reshape(-1),
        jnp.broadcast_to(pix_i[:, None], m1src_ij.shape).reshape(-1).astype(pix_i.dtype),
        catalog,
    ).reshape(z_s_safe.shape)

    # SIS optical depth at z_s
    log_tau = jnp.log(tau_2_SIS(z_s_safe, sis_params))

    # Jacobian and quadrature weights
    log_J = _log_jac_app_to_src(z_s_safe, dL_true_ij, mu_i_b, H0, Om0)
    log_quad = log_py[None, :] + log_wy[None, :]      # (1, N_y), broadcasts

    # PE proposal density: importance correction for event-i sample
    valid_i_b = (valid_i & (prior_wt_i > 0.0))[:, None]   # (N_pe_i, 1)
    log_pe_wt = jnp.where(valid_i_b, -jnp.log(prior_wt_i)[:, None], -jnp.inf)

    log_integrand = (
        log_pp + log_pz + log_tau + log_p_j + log_J + log_pe_wt + log_quad
    )
    # Mask out-of-grid cells and any cell where the population vanished
    log_integrand = jnp.where(
        in_grid & valid_i_b & jnp.isfinite(log_integrand),
        log_integrand,
        -jnp.inf,
    )
    return log_integrand    # (N_pe_i, N_y)


def cluster_log_likelihood_pair(
    event_i: dict,         # {'m1det', 'q', 'dL', 'chieff', 'prior_wt', 'valid', 'pixels'}
    event_j: dict,         # same fields
    kde_i: PairKDE,
    kde_j: PairKDE,
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    sis_params: SISLensParams,
    log_p_pop_fn: Callable,
    log_prior_z_fn: Callable,
    y_nodes: jnp.ndarray,
    log_wy: jnp.ndarray,
) -> jnp.ndarray:
    """Symmetric J=2 pair log-likelihood ``log L_2(d_i, d_j | λ, Θ)``.

    Symmetrizes over the two image-to-event assignments (i→+,j→- and
    i→-,j→+) by logsumexp. Each branch is itself a logsumexp over the
    PE samples of *the event mapped to μ_+* and the y-quadrature nodes;
    the *other* event's PE samples enter only through its KDE.

    By symmetrization, we drive the PE-sample sum from BOTH events:
    branch 1 uses event-i's samples and event-j's KDE; branch 2 uses
    event-j's samples and event-i's KDE. This is the variance-minimal
    formulation of the symmetric pair likelihood.

    Returns
    -------
    log_L2 : scalar
        log of the pair likelihood. -inf if both branches vanish.
    """
    # SIS magnifications at each y-node: μ_+ ≥ μ_-, both positive on (0, 1)
    mu_plus, mu_minus = mu_plus_minus_from_y(y_nodes)
    log_py = log_p_y_SIS(y_nodes)

    # Branch σ_a: i→μ_+, j→μ_-. Drive over event-i PE samples; KDE event-j.
    log_int_a = _pair_branch_log_integrand(
        m1det_i=event_i["m1det"], q_i=event_i["q"], dL_app_i=event_i["dL"],
        chieff_i=event_i["chieff"], prior_wt_i=event_i["prior_wt"],
        valid_i=event_i["valid"], pix_i=event_i["pixels"],
        mu_i=mu_plus, mu_j=mu_minus,
        log_py=log_py, log_wy=log_wy,
        kde_j=kde_j,
        cosmo=cosmo, survey=survey, pop_params=pop_params, catalog=catalog,
        sis_params=sis_params,
        log_p_pop_fn=log_p_pop_fn, log_prior_z_fn=log_prior_z_fn,
    )
    N_i = event_i["m1det"].shape[0]
    log_branch_a = logsumexp(log_int_a) - jnp.log(N_i)

    # Branch σ_b: j→μ_+, i→μ_-. Drive over event-j PE samples; KDE event-i.
    log_int_b = _pair_branch_log_integrand(
        m1det_i=event_j["m1det"], q_i=event_j["q"], dL_app_i=event_j["dL"],
        chieff_i=event_j["chieff"], prior_wt_i=event_j["prior_wt"],
        valid_i=event_j["valid"], pix_i=event_j["pixels"],
        mu_i=mu_plus, mu_j=mu_minus,
        log_py=log_py, log_wy=log_wy,
        kde_j=kde_i,
        cosmo=cosmo, survey=survey, pop_params=pop_params, catalog=catalog,
        sis_params=sis_params,
        log_p_pop_fn=log_p_pop_fn, log_prior_z_fn=log_prior_z_fn,
    )
    N_j = event_j["m1det"].shape[0]
    log_branch_b = logsumexp(log_int_b) - jnp.log(N_j)

    # Symmetric sum (1/2 because each assignment is one term of the average)
    return logsumexp(jnp.stack([log_branch_a, log_branch_b])) - jnp.log(2.0)
