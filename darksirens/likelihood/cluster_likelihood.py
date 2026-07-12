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

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import z_of_dL, dL_of_z, ddL_of_z, dL_in_z_grid
from darksirens.likelihood.pair_kde import PairKDE, log_eval_pair_kde
from darksirens.lensing.slmarks import (
    SISLensParams, tau_2_SIS, log_p_y_SIS,
    mu_plus_minus_from_y, delta_t_from_y,
)
from darksirens.lensing.grids import make_y_grid, make_hermite_u_grid


def _log_jac_app_to_src(z_s: jnp.ndarray,
                        dL_true: jnp.ndarray,
                        mu: jnp.ndarray,
                        H0: jnp.ndarray, Om0: jnp.ndarray,
                        w0: jnp.ndarray = -1.0, wa: jnp.ndarray = 0.0) -> jnp.ndarray:
    """log|∂(m1src, q, z_s, χ) / ∂(m1det, q, dL_app, χ)|_at fixed μ.

    The triangular map (m1src, z_s) → (m1det, dL_app) at fixed q, χ, μ has
    determinant (1 + z_s) · dL'(z_s) / √μ, so the factor that converts the
    source-frame density to the apparent-frame density at the PE-sample
    point is its reciprocal:

        log|J_app→src| = -log(1 + z_s) - log dL'(z_s) + 0.5 log μ.

    (Independently re-derived in the 2026-07-08 library review: the
    implementation below is correct; an earlier docstring stated the
    inverse and shipped with an unresolved "Wait —" note.)

    Returns
    -------
    log_J : same shape as inputs
        Signed log-Jacobian; ADD to log-integrand.
    """
    # Clamp-then-mask: exotic CPL corners can make dL(z) non-monotonic
    # (ddL <= 0); an unclamped log would NaN-poison the reverse pass even
    # for cells the caller later masks out.
    ddL = jnp.clip(ddL_of_z(z_s, dL_true, H0, Om0, w0, wa), 1e-30, None)
    return -jnp.log1p(z_s) - jnp.log(ddL) + 0.5 * jnp.log(mu)


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
    pair_marks: int = 0,
    delta_t_obs: jnp.ndarray | None = None,
    sigma_delta_t: jnp.ndarray | None = None,
    y_nodes: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute the per-PE-sample log-integrand over y for one image-assignment branch.

    Returns
    -------
    log_per_sample : (N_pe_i, N_y) — log of the integrand contribution
        from each (PE-i sample, y-node) cell, BEFORE summing over y or s.
    """
    # Full CPL (library review, lensing finding 5): dropping w0/wa ran the
    # pair kinematics in LambdaCDM against a CPL-sampled surrounding
    # likelihood.
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    # Broadcast: (N_pe_i, N_y)
    mu_i_b = mu_i[None, :]                            # (1, N_y)
    mu_j_b = mu_j[None, :]                            # (1, N_y)
    dL_app_i_b = dL_app_i[:, None]                    # (N_pe_i, 1)
    m1det_i_b = m1det_i[:, None]
    q_i_b = q_i[:, None]
    chieff_i_b = chieff_i[:, None]

    # Map event-i PE sample s through assigned magnification mu_i to source frame
    dL_true_ij = dL_app_i_b * jnp.sqrt(mu_i_b)        # (N_pe_i, N_y)
    in_grid = dL_in_z_grid(dL_true_ij, H0, Om0, w0, wa)
    z_s = z_of_dL(dL_true_ij, H0, Om0, w0, wa)
    z_s_safe = jnp.where(in_grid, z_s, 0.5)           # finite dummy when out-of-grid
    m1src_ij = m1det_i_b / (1.0 + z_s_safe)           # (N_pe_i, N_y)

    # Predict event-j apparent parameters given same source-frame θ_s and μ_j
    dL_src_at_zs = dL_of_z(z_s_safe, H0, Om0, w0, wa)
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
    log_J = _log_jac_app_to_src(z_s_safe, dL_true_ij, mu_i_b, H0, Om0, w0, wa)
    log_quad = log_py[None, :] + log_wy[None, :]      # (1, N_y), broadcasts

    # Optional pair mark: observed SIS time delay.  The mark is evaluated
    # inside the y quadrature so it constrains the same impact parameter as
    # the magnification marks. pair_marks=0 is exactly inert.
    if pair_marks == 1:
        if delta_t_obs is None or sigma_delta_t is None or y_nodes is None:
            raise ValueError("pair_marks=time requires delta_t_obs, sigma_delta_t, and y_nodes")
        dt_model = delta_t_from_y(y_nodes, sis_params)
        sig = jnp.asarray(sigma_delta_t, dtype=jnp.float64)
        log_time = (
            -0.5 * jnp.square((jnp.asarray(delta_t_obs, dtype=jnp.float64) - dt_model) / sig)
            - jnp.log(sig) - 0.5 * jnp.log(2.0 * jnp.pi)
        )
        log_quad = log_quad + log_time[None, :]

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


# Pair-mark modes (kept in sync with likelihood_with_clusters):
#   0 = none; 1 = Gaussian time mark evaluated on the y-quadrature;
#   2 = collapsed time mark — for sharp marks (sigma_dt << T0) the Gaussian
#       pins y to a window narrower than any practical global quadrature
#       spacing (the G-pilot failure mode: sigma/T0 ~ 0.008 vs node spacing
#       ~0.025 at 64 nodes). The y-integral is rewritten with the mark as
#       the measure and evaluated by Gauss-Hermite in the standardized
#       variable u = (T0 y - dt_obs)/sigma:
#           int dy p(y) N(dt | T0 y, s) G(y)
#             = (1/T0) sum_k w_k [p G](y* + (s/T0) u_k),   y* = dt_obs/T0,
#       exact for the Gaussian mark to Hermite order (the residual is the
#       non-polynomial part of [p G] across the ~sigma/T0 window; measured
#       below 1e-3 nats at production sharpness). Nodes falling outside the
#       SIS support (0, 1) are masked, so a pair whose delay exceeds T0 is
#       annihilated exactly.
PAIR_MARKS_DELTA_COLLAPSE = 2
_TIME_COLLAPSE_HERMITE_NODES = 8


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
    pair_marks: int = 0,
    delta_t_obs: jnp.ndarray | None = None,
    sigma_delta_t: jnp.ndarray | None = None,
    t_obs_window_sec: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Symmetric J=2 pair log-likelihood ``log L_2(d_i, d_j | λ, Θ)``.

    Time marks are COINCIDENCE ODDS. A time mark contributes the ratio of
    the observed |Δt| density under the lensed hypothesis to that under the
    unlensed one — p(Δt|lensed)/p(Δt|unlensed) — not the lensed density
    alone. The singleton branch of the partition posterior carries no
    arrival-time terms, so charging the pair branch only the lensed density
    p_L ~ 1/T0 silently taxes every pairing by ~ln T0 (≈13 nats at the SIS
    default), which is what erased true-pair recovery in the G pilots. For
    two independent events with arrivals uniform over an observing run of
    length T, the unordered separation density is
    p_U(Δt) = 2 (T − Δt) / T², so each time-marked pair branch adds
    −ln p_U(Δt) alongside the lensed density (both mark modes). Requires
    ``t_obs_window_sec`` (from the observed catalog's uniform observation
    times).

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
    log_inv_p_unlensed = jnp.asarray(0.0, dtype=jnp.float64)
    if pair_marks in (1, PAIR_MARKS_DELTA_COLLAPSE):
        if t_obs_window_sec is None:
            raise ValueError(
                "time marks require t_obs_window_sec (observing-run length) "
                "for the unlensed arrival-coincidence denominator"
            )
        T = jnp.asarray(t_obs_window_sec, dtype=jnp.float64)
        dt = jnp.abs(jnp.asarray(delta_t_obs, dtype=jnp.float64))
        # p_U(dt) = 2 (T - dt) / T^2: unordered separation density of two
        # independent arrivals uniform on a run of length T. A physical pair has
        # |dt| < T. If a recorded |dt| reaches/exceeds T (a mis-set window or a
        # cross-run time base -- caught by the preflight guard), p_U -> 0 and the
        # coincidence odds diverge. Clamp dt just below T so the reward stays
        # LARGE BUT FINITE. The previous +inf sentinel was flipped to -inf by the
        # caller's isfinite mask, silently ANNIHILATING an otherwise valid pair
        # (a bias toward no-lensing) -- the opposite of "infinitely favoured".
        dt = jnp.clip(dt, 0.0, T * (1.0 - 1e-9))
        p_u = 2.0 * (T - dt) / (T * T)
        log_inv_p_unlensed = -jnp.log(p_u)

    if pair_marks == PAIR_MARKS_DELTA_COLLAPSE:
        if delta_t_obs is None or sigma_delta_t is None:
            raise ValueError(
                "pair_marks=delta-collapse requires delta_t_obs and sigma_delta_t"
            )
        # Local Gauss-Hermite quadrature under the mark's Gaussian measure
        # (see PAIR_MARKS_DELTA_COLLAPSE above): the 1/T0 Jacobian replaces
        # the global quadrature weights; log_p_y_SIS supplies p(y) and,
        # together with the mask, the (0, 1) SIS-support guard.
        y_star = jnp.asarray(delta_t_obs, dtype=jnp.float64) / sis_params.T0
        sigma_y = jnp.asarray(sigma_delta_t, dtype=jnp.float64) / sis_params.T0
        u_nodes_t, log_wH_t = make_hermite_u_grid(_TIME_COLLAPSE_HERMITE_NODES)
        y_k = y_star + sigma_y * u_nodes_t
        in_support = (y_k > 0.0) & (y_k < 1.0)
        y_nodes = jnp.clip(y_k, 1e-9, 1.0 - 1e-9)
        log_wy = jnp.where(
            in_support,
            log_wH_t - jnp.log(sis_params.T0) + log_inv_p_unlensed,
            -jnp.inf,
        )
        pair_marks = 0  # mark absorbed as the quadrature measure
    elif pair_marks == 1:
        # Legacy quadrature mark: same coincidence denominator, applied to
        # the shared y-weights (scalar per pair).
        log_wy = log_wy + log_inv_p_unlensed

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
        pair_marks=pair_marks, delta_t_obs=delta_t_obs,
        sigma_delta_t=sigma_delta_t, y_nodes=y_nodes,
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
        pair_marks=pair_marks, delta_t_obs=delta_t_obs,
        sigma_delta_t=sigma_delta_t, y_nodes=y_nodes,
    )
    N_j = event_j["m1det"].shape[0]
    log_branch_b = logsumexp(log_int_b) - jnp.log(N_j)

    # Symmetric sum (1/2 because each assignment is one term of the average)
    return logsumexp(jnp.stack([log_branch_a, log_branch_b])) - jnp.log(2.0)


# ============================================================================
# Lensed-singleton (exactly-one-detected image) evidence channel
# ============================================================================

def _lensed_single_branch_log_integrand(
    m1det: jnp.ndarray, q: jnp.ndarray, dL_app: jnp.ndarray, chieff: jnp.ndarray,
    prior_wt: jnp.ndarray, valid: jnp.ndarray, pix: jnp.ndarray,
    mu_det: jnp.ndarray,        # magnification of the DETECTED image at y-nodes (N_y,)
    mu_partner: jnp.ndarray,    # magnification of the undetected partner (N_y,)
    log_py: jnp.ndarray, log_wy: jnp.ndarray,
    cosmo: CosmoParams, survey: SurveyParams, pop_params: jnp.ndarray,
    catalog: EMCatalog, sis_params: SISLensParams, fc_params,
    log_p_pop_fn: Callable, log_prior_z_fn: Callable,
) -> jnp.ndarray:
    """Per-(PE-sample, y-node) log-integrand for ONE image-identity branch of
    the lensed-singleton channel: the observed event is a strongly lensed
    image with magnification mu_det(y) whose partner image (mu_partner(y))
    was NOT detected.

    Identical structure to _pair_branch_log_integrand with the partner's
    PE-KDE term replaced by the censoring factor log[1 - P_det(partner)]
    (Finn-Chernoff analytic model — exact for the mock generator's rendering;
    swap fcpdet for an emulator for real data).
    """
    from darksirens.lensing.fcpdet import log_one_minus_pdet_fc

    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    mu_d = mu_det[None, :]                            # (1, N_y)
    mu_p = mu_partner[None, :]                        # (1, N_y)
    dL_app_b = dL_app[:, None]                        # (N_pe, 1)
    q_b = q[:, None]
    chieff_b = chieff[:, None]

    dL_true = dL_app_b * jnp.sqrt(mu_d)               # (N_pe, N_y)
    in_grid = dL_in_z_grid(dL_true, H0, Om0, w0, wa)
    z_s = z_of_dL(dL_true, H0, Om0, w0, wa)
    z_s_safe = jnp.where(in_grid, z_s, 0.5)
    m1src = m1det[:, None] / (1.0 + z_s_safe)

    log_pp = log_p_pop_fn(m1src, q_b, z_s_safe, chieff_b, pop_params)
    log_pz = log_prior_z_fn(
        z_s_safe.reshape(-1),
        jnp.broadcast_to(pix[:, None], m1src.shape).reshape(-1).astype(pix.dtype),
        catalog,
    ).reshape(z_s_safe.shape)
    log_tau = jnp.log(tau_2_SIS(z_s_safe, sis_params))

    # Partner censoring: apparent distance the partner WOULD have shown.
    dL_partner = dL_of_z(z_s_safe, H0, Om0, w0, wa) / jnp.sqrt(mu_p)
    log_miss = log_one_minus_pdet_fc(m1src, q_b, z_s_safe, dL_partner, fc_params)

    log_J = _log_jac_app_to_src(z_s_safe, dL_true, mu_d, H0, Om0, w0, wa)
    log_quad = log_py[None, :] + log_wy[None, :]

    valid_b = (valid & (prior_wt > 0.0))[:, None]
    log_pe_wt = jnp.where(valid_b, -jnp.log(prior_wt)[:, None], -jnp.inf)

    log_integrand = (
        log_pp + log_pz + log_tau + log_miss + log_J + log_pe_wt + log_quad
    )
    return jnp.where(
        in_grid & valid_b & jnp.isfinite(log_integrand),
        log_integrand,
        -jnp.inf,
    )


def lensed_single_log_likelihood_event(
    event: dict,           # {'m1det', 'q', 'dL', 'chieff', 'prior_wt', 'valid', 'pixels'}
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    sis_params: SISLensParams,
    fc_params,
    log_p_pop_fn: Callable,
    log_prior_z_fn: Callable,
    y_nodes: jnp.ndarray,
    log_wy: jnp.ndarray,
) -> jnp.ndarray:
    """log of the lensed-singleton evidence for one observed event:

        dN_1L(d) = ∫ dθ dy  τ₂(z) p(y) Σ_{j∈{+,-}} p(d | θ, μ_j(y)) ·
                   [1 - P_det(partner image)] · (population × volume terms)

    The two image identities are DISTINCT outcomes of the same source, so
    the branches are SUMMED (unlike the pair likelihood, which averages the
    two assignments of the same observation). Returns -inf if both vanish.
    """
    mu_plus, mu_minus = mu_plus_minus_from_y(y_nodes)
    log_py = log_p_y_SIS(y_nodes)
    N = event["m1det"].shape[0]

    common = dict(
        m1det=event["m1det"], q=event["q"], dL_app=event["dL"],
        chieff=event["chieff"], prior_wt=event["prior_wt"],
        valid=event["valid"], pix=event["pixels"],
        log_py=log_py, log_wy=log_wy,
        cosmo=cosmo, survey=survey, pop_params=pop_params, catalog=catalog,
        sis_params=sis_params, fc_params=fc_params,
        log_p_pop_fn=log_p_pop_fn, log_prior_z_fn=log_prior_z_fn,
    )
    # Branch: detected image is mu_+ (partner mu_-), and vice versa.
    log_int_plus = _lensed_single_branch_log_integrand(
        mu_det=mu_plus, mu_partner=mu_minus, **common,
    )
    log_int_minus = _lensed_single_branch_log_integrand(
        mu_det=mu_minus, mu_partner=mu_plus, **common,
    )
    log_branch_plus = logsumexp(log_int_plus) - jnp.log(N)
    log_branch_minus = logsumexp(log_int_minus) - jnp.log(N)
    return jnp.logaddexp(log_branch_plus, log_branch_minus)
