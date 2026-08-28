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
from the uncertainty in μ on the log-likelihood (Farr 2019, eq. 11).
The "+3" is a PRIOR CONVENTION on μ, not a higher-order Taylor term —
see the derivation below, and ``tests/test_selection_correction_coefficient.py``
which pins it numerically.

Derivation.  Marginalising the rate R out of the Poisson likelihood with
the scale-free prior p(R) ∝ 1/R leaves L ∝ μ^{-N}.  Monte-Carlo error
makes μ itself uncertain: μ ~ N(μ̂, σ²) with σ = μ̂/√N_eff.  Marginalising
that uncertainty under a scale-invariant prior p(μ) ∝ 1/μ,

    ⟨μ^{-N}⟩ / μ̂^{-N} = ⟨μ^{-N-1}⟩_G / ⟨μ^{-1}⟩_G
                       = [1 + (N+1)(N+2)/(2 N_eff)] / [1 + 2/(2 N_eff)]
                       = 1 + [(N+1)(N+2) − 2] / (2 N_eff)
                       = 1 + N(N+3) / (2 N_eff),

using ⟨μ^{-k}⟩_G/μ̂^{-k} ≈ 1 + k(k+1)σ²/(2μ̂²) for a Gaussian.  A FLAT
prior on μ would instead give N(N+1)/(2 N_eff); the two differ at O(1/N)
and the N(N+3) form is the one Farr (2019) states and that the reference
implementation uses (``gwpopulation.vt.ResamplingVT.vt_factor``:
``vt_factor = mu / exp((3 + n_events) / (2 n_effective))``, whose
``-N log vt_factor`` is exactly the expression above).  The N_obs²/(2 N_eff)
form sometimes quoted is the leading term only.

CAUTION: gwpopulation deprecates this marginalisation ("we recommend not
to use this as it is not completely understood if this uncertainty
marginalization is correct") in favour of bounding the variance of the
TOTAL log-likelihood — which is exactly the criterion this module already
enforces as its primary guard (next section).  The correction below is
kept because it is the community-standard closed form and is O(1/N_eff)
inside the region the guard admits, but the guard, not the correction, is
what protects the inference.

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

from darksirens.utils.utils import logsumexp_neginf_safe
from darksirens.core.types import GWEvent, EMCatalog
from darksirens.likelihood.events import pad_gw_event_to_multiple

# Single source of truth for the guard's default budget (the GWTC-4.0/5.0
# threshold): argparse defaults and getattr fallbacks all reference this, so
# CLI and programmatic callers cannot silently diverge.  Defined in
# core.constants (re-exported here under its historical home) so JAX-free
# tooling like io.results can read it without importing this JAX module.
from darksirens.core.constants import DEFAULT_MAX_LIKELIHOOD_VARIANCE

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
    Neff   : scalar  (0.0 when log_mu = -inf; +inf for an exactly-zero
        Monte-Carlo variance; never NaN for a finite input pair)
    log_sigma2 : scalar  (log Monte-Carlo variance of the selection integral μ;
        consumed by the strong-lensing cluster-selection combiner. -inf when
        log_mu = -inf, where Neff = 0 already forces the too-sparse -inf
        selection correction downstream, and -inf for an exactly-zero
        variance, where Neff = inf passes every reliability guard.)

    Reverse-mode discipline
    -----------------------
    The variance is carried as the INVERSE effective sample size

        1/N_eff = Var(μ̂)/μ̂² = exp(lse2 - 2·lse) - 1/Ndraw,

    the stable direct form :func:`log_evidence_and_mc_variance` already uses,
    NOT as ``logdiffexp(log_s2, 2·log_mu - log_Ndraw)``.  Uniform (or
    numerically uniform) importance weights make those two logdiffexp operands
    EQUAL: the forward value is correctly -inf, but the derivative runs through
    ``log1p(-exp(0))``, whose slope is 1/0, and reverse-mode AD has already
    evaluated that branch by the time a downstream ``jnp.where`` selects the
    other one — a ``where`` cannot erase a NaN cotangent.  MEASURED on master:
    four equal log weights gave the correct statistics (0, inf, -inf) with an
    all-NaN Jacobian, and ``selection_log_correction`` returned 0.0 with
    gradient [nan, nan, nan, nan].  A perfectly valid deterministic selection
    campaign therefore produced a finite log likelihood and unusable gradients
    for NUTS/HMC and every gradient optimizer.

    Both operands of every ``jnp.where`` below are finite, so the dead branch
    contributes a ZERO cotangent rather than a NaN one (the same discipline as
    ``redshift/catalog.py:_logsumexp_neginf_safe``).
    """
    log_Ndraw  = jnp.log(Ndraw)
    log_mu     = lse  - log_Ndraw

    finite_mu = jnp.isfinite(log_mu)
    lse_safe  = jnp.where(finite_mu, lse,  0.0)
    lse2_safe = jnp.where(finite_mu, lse2, 0.0)
    # >= 0 and finite by construction; the clamp also absorbs the round-off
    # that can push a numerically-uniform campaign a few ulps negative.
    inv_neff = jnp.maximum(jnp.exp(lse2_safe - 2.0 * lse_safe) - 1.0 / Ndraw, 0.0)

    zero_var = inv_neff <= 0.0
    inv_neff_safe = jnp.where(zero_var, 1.0, inv_neff)
    log_mu_safe   = jnp.where(finite_mu, log_mu, 0.0)

    # log_mu = -inf (all weights -inf): Neff = 0.0 triggers too_sparse -> -inf.
    # Zero variance: Neff = inf, which passes every reliability guard, and
    # log_sigma2 = -inf, so the channel drops out of the cluster combiner's sums.
    Neff = jnp.where(
        finite_mu,
        jnp.where(zero_var, jnp.inf, 1.0 / inv_neff_safe),
        0.0,
    )
    log_sigma2 = jnp.where(
        finite_mu & ~zero_var,
        2.0 * log_mu_safe + jnp.log(inv_neff_safe),
        -jnp.inf,
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
    # bit-identical to logsumexp_neginf_safe applied to ldw and 2*ldw
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

    The first term is the standard Poisson selection factor.  The second is
    the leading Monte-Carlo uncertainty correction of Farr (2019) eq. 11,
    matching ``gwpopulation.vt.ResamplingVT.vt_factor``.  The ``+3`` follows
    from marginalising μ under a scale-invariant p(μ) ∝ 1/μ — a flat prior
    would give N_obs(N_obs+1) instead — NOT from a higher-order Taylor term;
    the module docstring carries the derivation and
    ``tests/test_selection_correction_coefficient.py`` pins it numerically.

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
        selection-only bound.  Both the standard core and the
        cluster/lensing stack (per-event, lensed-singleton, and per-pair
        variances) thread it.

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
        # Neff = +inf (the zero-variance verdict _lse_to_log_mu_neff issues for
        # a deterministic/uniform campaign) must never reach the division: the
        # div VJP stores the inf numerator, and the underflowed gate's ZERO
        # cotangent times that inf is NaN — a NaN that flows into ``threshold``
        # -> ``pe_variance_sum``/``max_likelihood_variance`` while the returned
        # value stays finite, so the isfinite collapse below never fires.
        # Double-where (the ``_lse_to_log_mu_neff`` discipline): the dead
        # branch divides finite operands, and the inf is a gradient-free
        # constant that still drives the gate to exactly zero.
        finite_neff = jnp.isfinite(Neff)
        neff_safe = jnp.where(finite_neff, Neff, threshold)
        x = jnp.where(finite_neff, neff_safe / threshold, jnp.inf)
        gate = jax.nn.softplus(200.0 * (1.0 - x) - 10.0)
        reward_mag = n * jax.nn.softplus(-log_mu)
        wall = -gate * (100.0 + 2.0 * reward_mag)
        Neff_taylor = jnp.maximum(Neff, threshold)
        correction = (
            -n * log_mu
            + n * (3.0 + n) / (2.0 * Neff_taylor)
        )
        total = correction + wall
        # This collapse -- not the hard branch's ``too_sparse`` -- is the soft
        # path's NaN guarantee: mu == 0 exactly (all-invalid injections) makes
        # correction and wall +/-inf and their sum NaN; a NaN Neff propagates
        # through Neff_taylor -> correction (the double-where above shunts it
        # out of x), and a NaN threshold through x -> gate -> wall.  Either
        # way, collapse to the hard verdict.
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)
    # ~(>) not (<=): a NaN Neff or threshold must guard, never admit.
    too_sparse = ~(Neff > threshold)
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
    logsumexp(2·ldw))`` accumulation (through
    :func:`~darksirens.utils.utils.logsumexp_neginf_safe`, which is
    bit-identical to the plain reduction whenever any weight is finite), the
    same all-batches ``logsumexp`` combine (``logsumexp`` is additive across
    disjoint index sets), and the same
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
        lse = logsumexp_neginf_safe(ldw)
        lse2 = logsumexp_neginf_safe(2.0 * ldw)
    else:
        N_batches = N_sel // sel_batch_size

        def _scan_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            ldw_b = ldw_provider(start, sel_batch_size)
            return None, (
                logsumexp_neginf_safe(ldw_b),
                logsumexp_neginf_safe(2.0 * ldw_b),
            )

        _, (lse_all, lse2_all) = lax.scan(_scan_fn, None, jnp.arange(N_batches))
        lse = logsumexp_neginf_safe(lse_all)
        lse2 = logsumexp_neginf_safe(lse2_all)

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
        When ``gw_sel.spin`` is not None the callable is instead invoked with a
        trailing ``spin=(batch, d)`` keyword; chi_eff-basis weight functions
        (and test doubles) never see it.
        Per-sample log importance weight.  Must broadcast over the batch
        dimension.  Typically a closure from ``likelihood.py`` that captures
        cosmo, survey, pop_params, and the finite-guard for log_prior_z.
    Ndraw : float
        Total number of generated injections (detected + missed).
    nEvents : int
        Number of observed GW events.  ACCEPTED BUT UNUSED here (kept for the
        call-signature compatibility of every existing caller): the N_eff
        reliability check lives in :func:`selection_log_correction`, which the
        caller invokes with these outputs.  Estimating ``log_mu`` does not apply
        any guard -- passing ``nEvents`` here does NOT protect a caller that
        skips :func:`selection_log_correction`.
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
    log_sigma2 : scalar — log Monte-Carlo variance of μ (used by the
        strong-lensing cluster path; ignored by the standard likelihood)
    """
    def _batch_lse(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b, valid_b, nx_b, ny_b,
                   nz_b, spin_b=None):
        # ``spin_b`` is passed to ``log_weight_fn`` ONLY when the event carries
        # a spin block, so every existing weight function (and test double)
        # with the 7-argument signature keeps working and the d = 0 path is
        # call-for-call identical.
        if spin_b is None:
            ldw = log_weight_fn(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b, em_catalog_sel)
        else:
            ldw = log_weight_fn(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b,
                                em_catalog_sel, spin=spin_b)
        if sky_log_weight_fn is not None:
            ldw = ldw + sky_log_weight_fn(nx_b, ny_b, nz_b, dL_b)
        valid = valid_b & (pwt_b > 0.0)
        ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
        # ``ldw`` is finite-or--inf by the line above, so the neginf-safe
        # reduction is bit-identical to the plain one on any live batch and
        # only changes the ALL-invalid batch: -inf forward either way, but with
        # a zero instead of a NaN cotangent (softmax of all--inf is 0/0).
        return logsumexp_neginf_safe(ldw), logsumexp_neginf_safe(2.0 * ldw)

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
            gw_sel.spin,
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
                sl(gw_sel.spin) if gw_sel.spin is not None else None,
            )
            return None, (lse_b, lse2_b)

        _, (lse_all, lse2_all) = lax.scan(
            _scan_fn, None, jnp.arange(N_batches)
        )
        # Combine per-batch logsumexp values: logsumexp is additive
        # across disjoint index sets.  Neginf-safe for the same reason as the
        # per-batch reduction -- every batch can be all-invalid at once.
        lse  = logsumexp_neginf_safe(lse_all)
        lse2 = logsumexp_neginf_safe(lse2_all)

    return _lse_to_log_mu_neff(lse, lse2, Ndraw)
