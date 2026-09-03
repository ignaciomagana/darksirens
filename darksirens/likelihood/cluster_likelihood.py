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

    L_2(d_i, d_j | λ, Θ) = Σ_σ ∫ dy p(y) (1/N_i) Σ_s
        τ_2(z_s(s,y))
        · p_pop(θ_src(s,y) | λ)
        · p_z(z_s(s,y) | Θ)
        · p̂_j(θ_app_j_pred(s,y))
        · |J_app→src(z_s, dL_true, μ_σ(i)(y))|

where σ runs over the two image-assignment orderings (i→+,j→- and
i→-,j→+) — SUMMED, not averaged: the observed datum is the UNORDERED pair
{d_i, d_j}, and the two assignments are distinct microstates of it, so the
intensity at the observed pair is the sum. That is also the convention that
integrates to the cluster selection integral it is normalized against: with
g_a(d_1,d_2) the (i→+, j→-) branch and g_b = g_a(d_2,d_1), each of
∫∫ g_a dd_1 dd_2 and ∫∫ g_b is μ_sel^(2) (which counts every both-detected
source once), and the unordered-space integral of their sum is
(1/2)(μ_sel^(2) + μ_sel^(2)) = μ_sel^(2). Averaging instead would make the
pair channel's intensity a factor 2 below the -N_tot log(μ_tot) penalty
built from it. s indexes event-i PE samples, and:

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
Jacobian at fixed μ (see ``_log_jac_app_to_src``):
    log|J_app→src| = -log(1 + z_s) - log dL'(z_s) + 0.5 log μ_σ(i).

Quadrature
~~~~~~~~~~
y is integrated by Gauss-Legendre on (0, 1) with 32 nodes (commit 1's
``make_y_grid``). p(y) = 2y is smooth, the integrand can have a soft
edge near y = 0 (μ → ∞) and y = 1 (μ_- → 0), both of which are mitigated
by the population prior strongly disfavouring extreme z_s.

Strong-lensing optical depth τ_2(z_s) enters through ``slmarks.tau_2_prob``
(``A · z^n`` clipped to [0, 1-1e-12]): every channel consumes τ as the
Bernoulli probability of multiple imaging, and the raw power law exceeds
one at prior corners — see ``tau_2_prob``'s docstring. T_0 / Δt are NOT
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

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import z_of_dL, dL_of_z, ddL_of_z, dL_in_z_grid
from darksirens.likelihood.pair_kde import PairKDE, log_eval_pair_kde
from darksirens.lensing.slmarks import (
    SISLensParams, tau_2_prob, log_p_y_SIS,
    mu_plus_minus_from_y, delta_t_from_y,
)
from darksirens.lensing.grids import make_hermite_u_grid


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
    log_tau = jnp.log(tau_2_prob(z_s_safe, sis_params))

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


def _n_valid_rows(event: dict) -> jnp.ndarray:
    """Number of PE rows that enter a branch's importance mean: valid and with a
    positive proposal weight (the same mask ``_pair_branch_log_integrand``
    applies), floored at 1 so an empty event stays finite (its branch is -inf
    through the integrand anyway)."""
    valid = jnp.asarray(event["valid"], dtype=bool) & (
        jnp.asarray(event["prior_wt"]) > 0.0
    )
    return jnp.maximum(jnp.sum(valid.astype(jnp.float64)), 1.0)


def _correlated_branch_variance(log_z_a, var_a, log_z_b, var_b):
    """Delta-method variance of ``log((Z_a + Z_b)/c)`` from two branch
    estimates, under the PERFECTLY-CORRELATED bound
    ``(alpha_a*sigma_a + alpha_b*sigma_b)^2`` with ``alpha_x = Z_x/(Z_a+Z_b)``.

    Correlated because the branches share randomness (the pair kernel's
    branches reuse the same two events through each other's KDEs; the
    singleton mixture's branches are driven by the SAME PE samples), and an
    independent combination would understate the variance the total guard
    exists to bound.  The constant ``c`` divides out of the delta method.

    Reverse-mode discipline (see ``_logsumexp_neginf_safe``): both branches
    of a ``where`` are differentiated, so ``-inf - -inf`` and ``sqrt(0)``
    (infinite derivative) must never appear in the dead branch.
    """
    log_den = jnp.logaddexp(log_z_a, log_z_b)
    den_ok = jnp.isfinite(log_den)
    safe_den = jnp.where(den_ok, log_den, 0.0)
    safe_a = jnp.where(jnp.isfinite(log_z_a), log_z_a, -1e3)
    safe_b = jnp.where(jnp.isfinite(log_z_b), log_z_b, -1e3)
    alpha_a = jnp.where(den_ok & jnp.isfinite(log_z_a),
                        jnp.exp(jnp.minimum(safe_a - safe_den, 0.0)), 0.0)
    alpha_b = jnp.where(den_ok & jnp.isfinite(log_z_b),
                        jnp.exp(jnp.minimum(safe_b - safe_den, 0.0)), 0.0)
    sig_a = jnp.where(var_a > 0.0, jnp.sqrt(jnp.maximum(var_a, 1e-300)), 0.0)
    sig_b = jnp.where(var_b > 0.0, jnp.sqrt(jnp.maximum(var_b, 1e-300)), 0.0)
    return jnp.square(alpha_a * sig_a + alpha_b * sig_b)


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
    pair_time_signed: bool = False,
    return_mc_variance: bool = False,
) -> jnp.ndarray:
    """Symmetric J=2 pair log-likelihood ``log L_2(d_i, d_j | λ, Θ)``.

    With ``return_mc_variance=True`` returns ``(log_L2, var)`` where ``var``
    is the delta-method variance of ``log_L2`` under the PE-sample
    randomness (see the reduction at the bottom); the default returns the
    scalar unchanged, bit-compatible with every existing caller.

    ``var`` bounds only the DRIVING-sample component. Each branch's partner
    event enters through its Gaussian KDE, whose own sampling error the delta
    method cannot see, and that term is not sub-dominant in the regime the
    total-log-likelihood guard exists for: when the predicted apparent point
    falls in the tail of the partner's KDE (the false-pair configurations the
    channel must reject) the kernel sum is carried by one or two nearest
    samples and log p̂ has O(1) scatter. The perfectly-correlated branch
    combination bounds the CROSS term between branches, not the missing
    KDE-side term. So a pair whose log-likelihood is Monte-Carlo noise can
    still pass a budget that advertises sigma(lnL) <= 1 nat; catching that
    needs a per-pair kernel-ESS diagnostic (or the ESS folded into the same
    delta-method algebra), which this estimator does not provide.

    Time marks are COINCIDENCE ODDS. A time mark contributes the ratio of
    the observed |Δt| density under the lensed hypothesis to that under the
    unlensed one — p(Δt|lensed)/p(Δt|unlensed) — not the lensed density
    alone. The singleton branch of the partition posterior carries no
    arrival-time terms, so charging the pair branch only the lensed density
    p_L ~ 1/T0 silently taxes every pairing by ~ln T0 (≈13 nats at the SIS
    default), which is what erased true-pair recovery in the G pilots. For
    two independent events with arrivals uniform over an observing run of
    length T, the SIGNED separation Δt = t_j − t_i has the triangular density
    p_U(Δt) = (T − |Δt|) / T², and the folded |Δt| has twice that,
    2 (T − |Δt|) / T². Each time-marked pair branch adds −ln p_U(Δt) alongside
    the lensed density (both mark modes), using the convention that matches
    the mark: the folded density for ``pair_time_signed=False`` (whose
    numerator sums the two arrival-order configurations at the same |Δt|) and
    the signed one for ``pair_time_signed=True`` (whose surviving branch is a
    density in Δt over R). Requires ``t_obs_window_sec`` (from the observed
    catalog's uniform observation times).

    Symmetrizes over the two image-to-event assignments (i→+,j→- and
    i→-,j→+) by logsumexp. Each branch is itself a logsumexp over the
    PE samples of *the event mapped to μ_+* and the y-quadrature nodes;
    the *other* event's PE samples enter only through its KDE.

    Arrival ORDER (``pair_time_signed``)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    For SIS the type-I minimum μ_+ = (1+y)/y **always** arrives before the
    type-II saddle μ_- = (1-y)/y: the module's own delay is
    Δt = t_- - t_+ = T_0 y > 0 (``slmarks``). So the two assignment branches
    predict OPPOSITE signs of the observed delay, and at most one of them is
    compatible with the data.

    With ``pair_time_signed=True`` the caller guarantees ``delta_t_obs`` is
    SIGNED with the convention ``Δt = t_j - t_i`` for the stored pair order
    ``(i, j)``. Branch a (i is μ_+) is then evaluated at ``+Δt`` and branch b
    (j is μ_+) at ``-Δt``, so the branch whose predicted arrival order
    contradicts the data is masked out of the SIS support and contributes
    nothing. What this buys is the ordering test itself: a false pairing in
    which the BRIGHTER event arrives second should be suppressed to the (tiny)
    value of the ordering-consistent branch, whereas with |Δt| it gets the
    ordering-INconsistent branch, which can be orders of magnitude larger —
    over-rewarding exactly the false pairings the arrival-order constraint
    exists to reject. Both conventions are correctly normalized (each pairs its
    numerator with the matching unlensed reference density, above); a
    determined pairing simply gains the log 2 that resolving the sign takes
    away from the unlensed reference.

    ``pair_time_signed=False`` (the default) keeps the legacy |Δt|-in-both-
    branches behaviour, for data whose recorded mark carries no trustworthy
    sign; the coincidence denominator then uses the FOLDED density, so that
    for two events whose branches are equal (the sign carries no information)
    the signed and unsigned odds agree exactly.

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
        # The reference density must live in the SAME space as the observable
        # the numerator is a density in. With a signed mark the per-branch
        # Gaussian N(dt | T0 y, sigma) is normalized over dt in R, so the
        # unlensed reference is the SIGNED (triangular) separation density
        # (T - |dt|)/T^2; folding it onto |dt| doubles it, and charging the
        # folded density against a signed numerator cost every signed pair an
        # extra log 2 of coincidence odds.
        p_u = (1.0 if pair_time_signed else 2.0) * (T - dt) / (T * T)
        log_inv_p_unlensed = -jnp.log(p_u)

    # Per-branch arrival-order orientation. Branch a assigns mu_+ to event i,
    # so SIS predicts t_j - t_i = +T0 y; branch b assigns mu_+ to event j and
    # predicts t_j - t_i = -T0 y. With a signed mark exactly one of these can
    # be satisfied; without one, both branches see |dt| (legacy).
    if pair_marks in (1, PAIR_MARKS_DELTA_COLLAPSE):
        _dt_raw = jnp.asarray(delta_t_obs, dtype=jnp.float64)
        dt_a = _dt_raw if pair_time_signed else jnp.abs(_dt_raw)
        dt_b = -_dt_raw if pair_time_signed else jnp.abs(_dt_raw)
    else:
        dt_a = dt_b = delta_t_obs

    if pair_marks == PAIR_MARKS_DELTA_COLLAPSE:
        if delta_t_obs is None or sigma_delta_t is None:
            raise ValueError(
                "pair_marks=delta-collapse requires delta_t_obs and sigma_delta_t"
            )
        # Local Gauss-Hermite quadrature under the mark's Gaussian measure
        # (see PAIR_MARKS_DELTA_COLLAPSE above): the 1/T0 Jacobian replaces
        # the global quadrature weights; log_p_y_SIS supplies p(y) and,
        # together with the mask, the (0, 1) SIS-support guard.  Built ONCE
        # PER BRANCH, because y* = dt_branch/T0 carries the branch's predicted
        # arrival order: the ordering-inconsistent branch lands at y* < 0 and
        # every node is masked.
        sigma_y = jnp.asarray(sigma_delta_t, dtype=jnp.float64) / sis_params.T0
        u_nodes_t, log_wH_t = make_hermite_u_grid(_TIME_COLLAPSE_HERMITE_NODES)

        def _collapse_grid(dt_branch):
            y_k = dt_branch / sis_params.T0 + sigma_y * u_nodes_t
            in_support = (y_k > 0.0) & (y_k < 1.0)
            return (
                jnp.clip(y_k, 1e-9, 1.0 - 1e-9),
                jnp.where(
                    in_support,
                    log_wH_t - jnp.log(sis_params.T0) + log_inv_p_unlensed,
                    -jnp.inf,
                ),
            )

        y_nodes_a, log_wy_a = _collapse_grid(dt_a)
        y_nodes_b, log_wy_b = _collapse_grid(dt_b)
        pair_marks = 0  # mark absorbed as the quadrature measure
    elif pair_marks == 1:
        # Legacy quadrature mark: same coincidence denominator, applied to
        # the shared y-weights (scalar per pair). The branch orientation
        # enters through delta_t_obs inside _pair_branch_log_integrand.
        y_nodes_a = y_nodes_b = y_nodes
        log_wy_a = log_wy_b = log_wy + log_inv_p_unlensed
    else:
        y_nodes_a = y_nodes_b = y_nodes
        log_wy_a = log_wy_b = log_wy

    # SIS magnifications at each y-node: μ_+ ≥ μ_-, both positive on (0, 1)
    mu_plus_a, mu_minus_a = mu_plus_minus_from_y(y_nodes_a)
    log_py_a = log_p_y_SIS(y_nodes_a)
    mu_plus_b, mu_minus_b = mu_plus_minus_from_y(y_nodes_b)
    log_py_b = log_p_y_SIS(y_nodes_b)

    # Branch σ_a: i→μ_+, j→μ_-. Drive over event-i PE samples; KDE event-j.
    log_int_a = _pair_branch_log_integrand(
        m1det_i=event_i["m1det"], q_i=event_i["q"], dL_app_i=event_i["dL"],
        chieff_i=event_i["chieff"], prior_wt_i=event_i["prior_wt"],
        valid_i=event_i["valid"], pix_i=event_i["pixels"],
        mu_i=mu_plus_a, mu_j=mu_minus_a,
        log_py=log_py_a, log_wy=log_wy_a,
        kde_j=kde_j,
        cosmo=cosmo, survey=survey, pop_params=pop_params, catalog=catalog,
        sis_params=sis_params,
        log_p_pop_fn=log_p_pop_fn, log_prior_z_fn=log_prior_z_fn,
        pair_marks=pair_marks, delta_t_obs=dt_a,
        sigma_delta_t=sigma_delta_t, y_nodes=y_nodes_a,
    )
    # Normalise by the VALID driving samples, as the partner KDE does
    # (pair_kde.log_eval_pair_kde, "Padding"): a padded row is already -inf in
    # the integrand, and dividing by the padded length would shift log L_2 by
    # -log(N_pad / N_valid), a per-event constant that does not cancel in the
    # pair Bayes factor.  Identical to the array length when nothing is padded.
    N_i = _n_valid_rows(event_i)
    log_branch_a = logsumexp(log_int_a) - jnp.log(N_i)

    # Branch σ_b: j→μ_+, i→μ_-. Drive over event-j PE samples; KDE event-i.
    log_int_b = _pair_branch_log_integrand(
        m1det_i=event_j["m1det"], q_i=event_j["q"], dL_app_i=event_j["dL"],
        chieff_i=event_j["chieff"], prior_wt_i=event_j["prior_wt"],
        valid_i=event_j["valid"], pix_i=event_j["pixels"],
        mu_i=mu_plus_b, mu_j=mu_minus_b,
        log_py=log_py_b, log_wy=log_wy_b,
        kde_j=kde_i,
        cosmo=cosmo, survey=survey, pop_params=pop_params, catalog=catalog,
        sis_params=sis_params,
        log_p_pop_fn=log_p_pop_fn, log_prior_z_fn=log_prior_z_fn,
        pair_marks=pair_marks, delta_t_obs=dt_b,
        sigma_delta_t=sigma_delta_t, y_nodes=y_nodes_b,
    )
    N_j = _n_valid_rows(event_j)
    log_branch_b = logsumexp(log_int_b) - jnp.log(N_j)

    # Symmetric SUM over the two image-to-event assignments: they are distinct
    # microstates of the same unordered observation, and only the sum integrates
    # to mu_sel^(2) over the unordered pair space (see the module docstring).
    log_L2 = logsumexp(jnp.stack([log_branch_a, log_branch_b]))
    if not return_mc_variance:
        return log_L2

    # Delta-method Monte-Carlo variance of log_L2.  The randomness is the
    # driving PE sample set of each branch (the y axis is deterministic
    # quadrature), so each branch is an importance mean over its per-sample
    # weights w_s = sum_y exp(log_int[s, y]) and gets the standard
    # sum(w^2)/(sum w)^2 - 1/N variance of its log.  The branches are
    # combined with the PERFECTLY-CORRELATED bound
    # (alpha_a*sigma_a + alpha_b*sigma_b)^2: they reuse the same two events
    # through each other's KDEs, whose sampling noise the delta method does
    # not see, so the independent combination would understate the variance
    # this term exists to guard against.
    from darksirens.likelihood.selection import log_evidence_and_mc_variance

    _, var_a = log_evidence_and_mc_variance(logsumexp(log_int_a, axis=1), N_i)
    _, var_b = log_evidence_and_mc_variance(logsumexp(log_int_b, axis=1), N_j)
    return log_L2, _correlated_branch_variance(
        log_branch_a, var_a, log_branch_b, var_b
    )


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
    pair_orientation_mode: str = "independent",
) -> jnp.ndarray:
    """Per-(PE-sample, y-node) log-integrand for ONE image-identity branch of
    the lensed-singleton channel: the observed event is a strongly lensed
    image with magnification mu_det(y) whose partner image (mu_partner(y))
    was NOT detected.

    Identical structure to _pair_branch_log_integrand with the partner's
    PE-KDE term replaced by the censoring factor log P(partner missed)
    (Finn-Chernoff analytic model — exact for the mock generator's rendering;
    swap fcpdet for an emulator for real data). ``pair_orientation_mode``
    selects the censoring convention: 'independent' (default; the pinned
    marginal log[1 - P_det(partner)], matching independently-rendered mocks)
    or 'shared_iota' (the joint-orientation conditional
    log P(partner missed | this image detected), for campaigns rendered with
    one (iota, psi, alpha, delta) per source; see darksirens.lensing.fcpdet).
    """
    from darksirens.lensing.fcpdet import log_pmiss_partner_fc

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
    log_tau = jnp.log(tau_2_prob(z_s_safe, sis_params))

    # Partner censoring: apparent distance the partner WOULD have shown.
    # The detected image's own apparent distance (the PE sample dL_app) only
    # matters in shared_iota mode, via the shared-inclination conditioning.
    dL_partner = dL_of_z(z_s_safe, H0, Om0, w0, wa) / jnp.sqrt(mu_p)
    log_miss = log_pmiss_partner_fc(
        m1src, q_b, z_s_safe, dL_app_b, dL_partner, fc_params,
        pair_orientation_mode=pair_orientation_mode,
    )

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
    pair_orientation_mode: str = "independent",
    return_mc_variance: bool = False,
) -> jnp.ndarray:
    """log of the lensed-singleton evidence for one observed event:

        dN_1L(d) = ∫ dθ dy  τ₂(z) p(y) Σ_{j∈{+,-}} p(d | θ, μ_j(y)) ·
                   P(partner image missed) · (population × volume terms)

    The two image identities are DISTINCT outcomes of the same source, so
    the branches are SUMMED (as are the pair likelihood's two image-to-event
    assignments, by the same argument). Returns -inf if both vanish.
    ``pair_orientation_mode`` picks the censoring convention (see
    ``_lensed_single_branch_log_integrand``); 'independent' is bit-identical
    to the historical behaviour.
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
        pair_orientation_mode=pair_orientation_mode,
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
    log_z = jnp.logaddexp(log_branch_plus, log_branch_minus)
    if not return_mc_variance:
        return log_z
    # Both image-identity branches are driven by the SAME N PE samples and
    # SUMMED, so the whole channel is one importance mean over per-sample
    # weights w_s = sum_y [exp(plus[s,y]) + exp(minus[s,y])] and the standard
    # delta-method variance is exact for it (no cross-branch correlation
    # bound needed, unlike the pair kernel).
    from darksirens.likelihood.selection import log_evidence_and_mc_variance

    rows = logsumexp(
        jnp.concatenate([log_int_plus, log_int_minus], axis=1), axis=1
    )
    _, var = log_evidence_and_mc_variance(rows, N)
    return log_z, var
