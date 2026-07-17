"""
selection.py
------------
Hierarchical selection integral for gravitational-wave population inference.

Physical picture
~~~~~~~~~~~~~~~~
The observed GW event rate depends on which signals pass the detection
threshold.  To avoid biasing the population inference, we must correct
for this selection effect via Thrane & Talbot (2019) / Farr (2019):

    log L_sel = -N_obs * log μ  +  N_obs(N_obs + 3) / (2 N_eff)

where μ is the expected number of detections per unit time under the
proposed population model, estimated as a Monte Carlo average over
injection samples:

    μ = (1/N_draw) Σ_{det inj}  p_pop(d_i|λ) / p_draw(d_i)

and N_eff is the effective sample size of the selection integral
(Farr 2019, arXiv:1904.10879, eq. 15):

    N_eff = μ² / Var(μ)   ≈   [Σ w_i]² / Σ w_i²

The coefficient N_obs(N_obs+3)/(2 N_eff) is the leading-order correction
from the uncertainty in μ on the log-likelihood (see the derivation in
the appendix of Talbot & Golomb 2023, arXiv:2304.06138, eq. A9).  This
is *not* the simpler N_obs²/(2 N_eff) term from the basic Farr (2019)
expansion; the extra factor of 3 arises from the next-order term in the
Taylor expansion of log μ around its mean.

Reliability criterion
~~~~~~~~~~~~~~~~~~~~~
Farr (2019) recommends discarding proposals where N_eff < 4 N_obs
(equivalently returning -inf); Vitale et al. (2022) use 5 N_obs.  Both
control the MEAN correction only.  The state-of-the-art criterion
(Essick & Farr 2022; Talbot & Golomb 2023; adopted by the GWTC-4.0 and
GWTC-5.0 population analyses) instead bounds the Monte-Carlo variance of
the TOTAL log-likelihood estimator,

    σ²_lnL = Σ_i σ²_i  +  N_obs² σ²_sel  <=  max_likelihood_variance

where σ²_i = Σ_j w_ij²/(Σ_j w_ij)² − 1/n_i is the per-event variance from
reweighting n_i PE samples (``log_evidence_and_mc_variance``) and
σ²_sel = Var[log μ] = 1/N_eff is the selection variance (this module's
N_eff = μ²/Var(μ) already contains the −1/N_draw term, so N_obs²/N_eff
is exactly the N_obs² σ²_sel component).  The likelihood carries
``-N_obs log mu``, so selection noise enters amplified by N_obs;
controlling it needs ``N_eff >> N_obs^2``, strictly stronger than
5 N_obs once N_obs > 5.  We therefore guard on

    N_eff > max(5 N_obs, N_obs² / (max_likelihood_variance − Σ_i σ²_i))

with ``max_likelihood_variance = 1`` by default (σ(lnL) <= 1 nat, the
GWTC-4.0/5.0 threshold), keeping the Vitale mean floor.  With
``Σ_i σ²_i = 0`` this reduces exactly to the selection-only variance
criterion ``N_eff > N_obs² / max_likelihood_variance``.

References
~~~~~~~~~~
- Farr, W.M. (2019). arXiv:1904.10879
- Thrane & Talbot (2019). PASA 36, e010
- Essick & Farr (2022). arXiv:2204.00461
- Talbot & Golomb (2023). MNRAS 526, 3495. arXiv:2304.06138
- Vitale et al. (2022). arXiv:2007.05579
- GWTC-4.0 populations: arXiv:2508.18083; GWTC-5.0: arXiv:2605.27226
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.utils.utils import logdiffexp
from darksirens.core.types import GWEvent, EMCatalog
from darksirens.likelihood.events import pad_gw_event_to_multiple

# Single source of truth for the guard's default budget (the GWTC-4.0/5.0
# threshold): argparse defaults and getattr fallbacks all reference this, so
# CLI and programmatic callers cannot silently diverge.
DEFAULT_MAX_LIKELIHOOD_VARIANCE = 1.0

# Numerical floor for the remaining variance budget once the per-event
# Monte-Carlo variances are subtracted: keeps the traced threshold finite
# (the guard then rejects through the Neff comparison, never through NaN).
# Its magnitude is load-bearing: with nEvents >= 1 the saturated threshold is
# >= 1/_MIN_VARIANCE_BUDGET = 1e12, which must exceed any attainable Neff
# (bounded by the injection-campaign size, <~ 1e9) so an over-budget proposal
# is ALWAYS rejected.  Raising this epsilon without re-deriving that bound
# can silently admit over-budget proposals; it also floors any requested
# budget below 1e-12 (irrelevant in practice).
_MIN_VARIANCE_BUDGET = 1e-12


# ============================================================
# Core estimators (testable in isolation)
# ============================================================

def _lse_to_log_mu_neff(
    lse: jnp.ndarray,
    lse2: jnp.ndarray,
    Ndraw: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convert logsumexp aggregates to (log_mu, N_eff, log_sigma2).
 
    Parameters
    ----------
    lse  : logsumexp(log_weights)
    lse2 : logsumexp(2 * log_weights)
    Ndraw : total number of generated injections
 
    Returns
    -------
    log_mu : scalar
    Neff   : scalar  (0.0 when log_mu = -inf; never NaN)
    log_sigma2 : scalar  (log Monte-Carlo variance of the selection integral μ;
        consumed by the strong-lensing cluster-selection combiner. May be
        -inf/NaN when log_mu = -inf, where Neff = 0 already forces the
        too-sparse -inf selection correction downstream.)
    """
    log_Ndraw  = jnp.log(Ndraw)
    log_mu     = lse  - log_Ndraw
    log_s2     = lse2 - 2.0 * log_Ndraw
 
    # Var(μ) estimator in log-space
    log_sigma2 = logdiffexp(log_s2, 2.0 * log_mu - log_Ndraw)
 
    # Guard: when log_mu = -inf (all weights −∞), both log_mu and
    # log_sigma2 are -inf.  The subtraction 2*(-inf) - (-inf) = NaN.
    # We instead set Neff = 0.0, which triggers too_sparse → return -inf.
    finite_mu = jnp.isfinite(log_mu)
    Neff = jnp.where(
        finite_mu,
        jnp.exp(2.0 * log_mu - log_sigma2),
        0.0,
    )
 
    return log_mu, Neff, log_sigma2


def log_evidence_and_mc_variance(
    ldw: jnp.ndarray,
    nsamp: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Per-event Monte-Carlo log-evidence and the variance of its log.

    The per-event term of the hierarchical likelihood is the importance
    average over the event's n PE samples,

        ln Ẑ = -log n + logsumexp(ldw)

    and the delta-method variance of ln Ẑ (Essick & Farr 2022,
    arXiv:2204.00461; Talbot & Golomb 2023, arXiv:2304.06138, eqs 8-9) is

        σ² = Σ_j w_j² / (Σ_j w_j)²  −  1/n

    Masked/invalid samples enter as w = 0 draws and COUNT in n (they are
    zero-weight members of the same PE sample set).  By Cauchy-Schwarz
    σ² ∈ [0, 1 − 1/n]: 0 for uniform weights, 1 − 1/n when a single
    sample carries all the weight.

    An event whose samples ALL mask to -inf returns ``(-inf, 0.0)``: the
    -inf log-evidence already kills the likelihood, and the variance must
    stay finite so it cannot NaN-poison the total-variance guard.  The
    aggregates are sanitized BEFORE the exp so the dead branch of the
    ``jnp.where`` carries no NaN cotangent (both branches of a ``where``
    are differentiated — the same reverse-mode NaN class documented in
    ``_logsumexp_neginf_safe``).

    Parameters
    ----------
    ldw : per-sample log importance weights (invalid samples at -inf)
    nsamp : number of PE samples, INCLUDING masked ones

    Returns
    -------
    log_evidence : scalar — ln Ẑ (-inf when all samples are masked)
    variance     : scalar — σ² of ln Ẑ, in [0, 1 − 1/n]; 0.0 when ln Ẑ = -inf
    """
    # Shared finite mask for both reductions (isfinite(2x) == isfinite(x)):
    # bit-identical to _logsumexp_neginf_safe applied to ldw and 2*ldw
    # separately (either sentinel underflows to exactly zero weight), one
    # nsamp-length isfinite/where/any pass cheaper inside the per-event scan.
    finite = jnp.isfinite(ldw)
    any_finite = jnp.any(finite)
    safe = jnp.where(finite, ldw, -1e30)
    lse = jnp.where(any_finite, logsumexp(safe), -jnp.inf)
    lse2 = jnp.where(any_finite, logsumexp(2.0 * safe), -jnp.inf)
    lse_safe = jnp.where(any_finite, lse, 0.0)
    lse2_safe = jnp.where(any_finite, lse2, 0.0)
    # exp(lse2 - 2*lse) - 1/n is algebraically 1/N_eff of _lse_to_log_mu_neff
    # at Ndraw = nsamp; kept in this direct form for the all-masked NaN safety.
    variance = jnp.where(
        any_finite,
        jnp.maximum(jnp.exp(lse2_safe - 2.0 * lse_safe) - 1.0 / nsamp, 0.0),
        0.0,
    )
    return -jnp.log(nsamp) + lse, variance


def selection_log_correction(
    log_mu: jnp.ndarray,
    Neff: jnp.ndarray,
    nEvents: int,
    soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    pe_variance_sum: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """
    Log selection correction term (Farr 2019 / Talbot & Golomb 2023).

    Returns ``-inf`` when the Monte-Carlo estimate of the log-likelihood
    is too noisy to trust, under BOTH criteria:

    * ``N_eff <= 5 N_obs`` — the Vitale et al. (2022) floor on the MEAN
      correction;
    * ``σ²_lnL = pe_variance_sum + N_obs²/N_eff > max_likelihood_variance``
      — the GWTC-4.0/5.0 bound on the VARIANCE of the TOTAL log-likelihood
      estimator (Essick & Farr 2022; Talbot & Golomb 2023).
      ``pe_variance_sum`` is Σ_i σ²_i over the per-event reweighting
      variances (``log_evidence_and_mc_variance``); ``N_obs²/N_eff`` is
      the selection component: the likelihood carries ``-N_obs log mu``,
      so a Monte-Carlo fluctuation ``sigma(log mu) ~ 1/sqrt(N_eff)``
      enters amplified by N_obs.  The 5 N_obs floor alone does NOT
      control this at larger catalogs: at N_obs = 50 it admits
      N_eff ~ 300, where an ordinary ~3.5 sigma dip in log mu across a
      parameter grid (0.19 nats at N_eff ~ 350) becomes an e^{9.5}
      likelihood spike — measured in end-to-end mock closures, where such
      spikes carried 30-86% of the posterior mass in a single grid cell
      and reproduced at the same parameter values across independent
      event realizations sharing an injection set.  The default
      ``max_likelihood_variance = 1.0`` caps sigma(lnL) at 1 nat, the
      threshold adopted by the GWTC-4.0/5.0 population analyses.  With
      ``pe_variance_sum = 0`` (the default) the criterion reduces exactly
      to the selection-only bound ``N_eff > N_obs²/max_likelihood_variance``.

    With ``soft_guard=True`` (gradient-based samplers) the hard wall is
    replaced by a steep smooth penalty — see the inline comment.  The
    wall stays keyed on ``Neff/threshold``; the per-event variances enter
    through the threshold, so it engages smoothly as the budget is spent.
    Once ``pe_variance_sum`` exceeds the budget the threshold saturates at
    ``N_obs²/1e-12`` (its gradient w.r.t. ``pe_variance_sum`` clamps to
    zero there) and the wall repels through the ``Neff``/``log_mu``
    channels at its validated full depth.

    The correction is:

        -N_obs * log μ  +  N_obs * (N_obs + 3) / (2 * N_eff)

    The first term is the standard Poisson selection factor.  The second
    is the leading uncertainty correction from Taylor-expanding log μ.

    Parameters
    ----------
    log_mu : log of the selection integral estimate
    Neff   : effective sample size of the selection integral
    nEvents : number of observed GW events
    max_likelihood_variance : cap on the variance of the total
        log-likelihood estimator, ``pe_variance_sum + N_obs²/N_eff``.
    pe_variance_sum : Σ_i σ²_i, the summed per-event Monte-Carlo
        variances of ``ln Ẑ_i`` (traced scalar or 0.0).  CAUTION: the
        total-variance criterion is only enforced when the caller threads
        this in; at the 0.0 default the guard silently reverts to the
        selection-only bound (currently the case for the cluster/lensing
        stack, whose per-pair variances arrive in a follow-up).

    Returns
    -------
    Scalar log-likelihood contribution from the selection term.
    """
    n = float(nEvents)
    # Remaining budget for the selection term after the per-event MC
    # variances; jnp.maximum (not Python max): the threshold is traced
    # whenever pe_variance_sum is.
    budget = jnp.maximum(max_likelihood_variance - pe_variance_sum, _MIN_VARIANCE_BUDGET)
    threshold = jnp.maximum(5.0 * n, (n * n) / budget)
    # ~(>) not (<=): a NaN Neff or threshold must guard, never admit.
    too_sparse = ~(Neff > threshold)
    if soft_guard:
        # Gradient-based samplers (NumPyro NUTS) cannot cross a hard -inf wall:
        # every trajectory that brushes it is flagged divergent (the H1-profile
        # smoke measured 100% divergent transitions with the sparse-Neff wall
        # 1.5 posterior-sigma from the mean). Replace the wall with a steep
        # smooth penalty whose MAGNITUDE TRACKS the -N log(mu) reward it is
        # guarding against: a fixed-height wall saturates (~ -1.5e3) while
        # -N log(mu) grows without bound as mu -> 0, opening a spurious
        # high-likelihood pocket in the deep-sparse region (library review,
        # likelihood finding 3). With the reward-tracking factor the wall
        # dominates the reward for Neff <~ 0.95x threshold; the residual
        # boundary layer (0.95 < Neff/threshold < 1) is thin, its reward is
        # already legitimately available just above threshold, and the
        # sampler cross-check (gate V9) verifies no posterior mass sits
        # there. Above ~1.05x threshold the gate underflows and the branch is
        # numerically identical to the hard guard's valid branch. The 1/Neff
        # Taylor term is frozen at the threshold so it cannot blow up
        # (positively) in the regime the wall is pushing away from.  NOTE:
        # with nonzero pe_variance_sum the threshold is the TOTAL-variance
        # boundary, inflated beyond the pure selection boundary N^2/max_var
        # (up to N^2/1e-12 once the budget is exhausted) — the gate turn-on
        # location and the frozen Taylor value move with it by design.
        x = Neff / threshold
        gate = jax.nn.softplus(200.0 * (1.0 - x) - 10.0)
        reward_mag = n * jax.nn.softplus(-log_mu)
        wall = -gate * (100.0 + 2.0 * reward_mag)
        Neff_taylor = jnp.maximum(Neff, threshold)
        correction = (
            -n * log_mu
            + n * (3.0 + n) / (2.0 * Neff_taylor)
        )
        total = correction + wall
        # mu == 0 exactly (all-invalid injections): correction and wall are
        # +/-inf and their sum is NaN — collapse to the hard verdict.
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)
    correction = (
        -n * log_mu
        + n * (3.0 + n) / (2.0 * Neff)
    )
    return jnp.where(too_sparse, -jnp.inf, correction)


# ============================================================
# Full selection term (batched or unbatched)
# ============================================================

def selection_reduce_from_ldw_provider(
    ldw_provider,
    N_sel: int,
    Ndraw: float,
    sel_batch_size: int | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reduce fully-composed selection log-weights to ``(log_mu, Neff, log_sigma2)``.

    ``ldw_provider(start, size) -> (size,)`` returns the FINAL, already-masked
    selection log importance weights for the injection slice
    ``[start, start + size)`` (i.e. the array ``_batch_lse`` computes just
    before its ``logsumexp``).  This mirrors :func:`compute_selection_term`'s
    reduction bit-for-bit -- the same per-batch ``(logsumexp(ldw),
    logsumexp(2·ldw))`` accumulation, the same all-batches ``logsumexp``
    combine (``logsumexp`` is additive across disjoint index sets), and the same
    :func:`_lse_to_log_mu_neff` -- so a factored caller that recomputes the
    per-member ldw through this helper matches the monolithic path's reduction.

    Factored out (rather than delegated from :func:`compute_selection_term`) so
    the existing public function's compute graph stays byte-identical; the
    LSS-completion member marginalization uses this with a member-specific
    provider that reuses the precomputed observed-catalog part.  When batching,
    ``N_sel`` must already be a multiple of ``sel_batch_size`` (the caller pads
    the injection arrays exactly as :func:`compute_selection_term` does).
    """
    if sel_batch_size is None:
        ldw = ldw_provider(0, N_sel)
        lse = logsumexp(ldw)
        lse2 = logsumexp(2.0 * ldw)
    else:
        N_batches = N_sel // sel_batch_size

        def _scan_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            ldw_b = ldw_provider(start, sel_batch_size)
            return None, (logsumexp(ldw_b), logsumexp(2.0 * ldw_b))

        _, (lse_all, lse2_all) = lax.scan(_scan_fn, None, jnp.arange(N_batches))
        lse = logsumexp(lse_all)
        lse2 = logsumexp(lse2_all)

    return _lse_to_log_mu_neff(lse, lse2, Ndraw)


def compute_selection_term(
    gw_sel: GWEvent,
    em_catalog_sel: EMCatalog,
    log_weight_fn,
    Ndraw: float,
    nEvents: int,
    sel_batch_size: int | None = None,
    sky_log_weight_fn=None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Estimate log μ, N_eff and log_sigma2 from the injection set.

    Returns the 3-tuple ``(log_mu, Neff, log_sigma2)`` (the third element is the
    log Monte-Carlo variance of μ, used by the strong-lensing cluster path; the
    standard single-event likelihood ignores it).

    Parameters
    ----------
    gw_sel : GWEvent
        Injection samples (detected).  If ``sel_batch_size`` is set and the
        length is not divisible by the batch size, the event is padded with
        explicit invalid sentinel rows before scanning.
    em_catalog_sel : EMCatalog
        EM catalog sliced to the injection sky positions.
    log_weight_fn : callable(m1det, q, dL, chieff, pix, prior_wt, catalog) → array
        Per-sample log importance weight.  Must broadcast over the batch
        dimension.  Typically a closure from ``likelihood.py`` that captures
        cosmo, survey, pop_params, and the finite-guard for log_prior_z.
    Ndraw : float
        Total number of generated injections (detected + missed).
    nEvents : int
        Number of observed GW events (for the N_eff reliability check).
    sel_batch_size : int or None
        If not None, process injections in chunks via ``lax.scan`` to
        limit peak GPU memory.  Non-divisible inputs are padded internally;
        padded rows have ``valid == False`` and contribute zero weight.
    sky_log_weight_fn : callable(nx, ny, nz, dL) → array or None
        Optional sky factor ``log g(n̂, z)`` added to each injection's log
        importance weight (the same factor applied to the PE term), so the
        selection integral reweights ``μ`` consistently.  ``dL`` is passed so the
        closure can derive the redshift ``z`` for 3-D ``g(n̂, z)`` models with the
        same cosmology as the PE term.  ``None`` (default) leaves the integral
        sky-agnostic — the isotropic / legacy behaviour.

    Returns
    -------
    log_mu : scalar — log of the selection integral estimate
    Neff   : scalar — effective sample size
    """
    def _batch_lse(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b, valid_b, nx_b, ny_b, nz_b):
        ldw = log_weight_fn(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b, em_catalog_sel)
        if sky_log_weight_fn is not None:
            ldw = ldw + sky_log_weight_fn(nx_b, ny_b, nz_b, dL_b)
        valid = valid_b & (pwt_b > 0.0)
        ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
        return logsumexp(ldw), logsumexp(2.0 * ldw)

    if sel_batch_size is None:
        # --- Unbatched: process all injections at once ---
        lse, lse2 = _batch_lse(
            gw_sel.dL,
            gw_sel.m1det,
            gw_sel.q,
            gw_sel.chieff,
            gw_sel.pixels,
            gw_sel.prior_wt,
            gw_sel.valid,
            gw_sel.nx,
            gw_sel.ny,
            gw_sel.nz,
        )
    else:
        # --- Batched via lax.scan ---
        # Peak GPU memory: O(sel_batch_size × N_grid) instead of O(N_sel × N_grid).
        gw_sel, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)
        N_sel = gw_sel.dL.shape[0]
        N_batches = N_sel // sel_batch_size

        def _scan_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, sel_batch_size)
            if sky_log_weight_fn is not None:
                nx_b, ny_b, nz_b = sl(gw_sel.nx), sl(gw_sel.ny), sl(gw_sel.nz)
            else:
                nx_b = ny_b = nz_b = None
            lse_b, lse2_b = _batch_lse(
                sl(gw_sel.dL),
                sl(gw_sel.m1det),
                sl(gw_sel.q),
                sl(gw_sel.chieff),
                sl(gw_sel.pixels),
                sl(gw_sel.prior_wt),
                sl(gw_sel.valid),
                nx_b,
                ny_b,
                nz_b,
            )
            return None, (lse_b, lse2_b)

        _, (lse_all, lse2_all) = lax.scan(
            _scan_fn, None, jnp.arange(N_batches)
        )
        # Combine per-batch logsumexp values: logsumexp is additive
        # across disjoint index sets.
        lse  = logsumexp(lse_all)
        lse2 = logsumexp(lse2_all)

    return _lse_to_log_mu_neff(lse, lse2, Ndraw)
