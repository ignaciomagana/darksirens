"""
cluster_selection.py
--------------------
Cluster (J=2) contribution to the expected detected-event count.

Mathematical specification
~~~~~~~~~~~~~~~~~~~~~~~~~~
For SIS doubles, the expected number of detected J=2 clusters under the
both-detected approximation (commit-4 design choice b, see notes) is:

    μ_sel^(2)(λ, Θ)
      = ∫ dθ_src p_pop(θ_src|λ) · p_z(z_s|Θ) · τ_2(z_s|Θ)
      × ∫_0^1 dy p(y) · p_det^(1)(θ_app,+) · p_det^(1)(θ_app,-)

where p_det^(1) is the *single-event* detection probability, computed by
the same injection-rendering pipeline that produces darksirens'
singleton selection set.

Estimator
~~~~~~~~~
The injection campaign generated N_draw_sources source-frame samples
(θ_src^(s), y^(s)) from proposal p_prop_src · p_prop_y and rendered
BOTH images of each source through the detection pipeline, recording
detection flags d_+^(s), d_-^(s) ∈ {0, 1}. The estimator is:

    μ̂_sel^(2) = (1 / N_draw_sources) Σ_{s: d_+ = d_- = 1}
                  p_pop(θ_src^(s)|λ) · p_z(z_s^(s)|Θ) · τ_2(z_s^(s)|Θ) · p(y^(s))
                  / (p_prop_src(θ_src^(s)) · p_prop_y(y^(s)))

At load time, ``LensedInjectionSet.make_lensed_injection_set`` precomputes
the "both-detected" subset, so by the time we enter the JIT'd hot path
the sum is just over the kept rows. The normalization denominator is
still N_draw_sources (NOT N_kept) — this is the importance-sampling
correction.

The known bias from option (b)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Real LVK lensing analyses additionally require a *pair-identification*
statistic to flag the candidate (Bayes-factor or posterior-overlap
threshold). Our estimator implements ONLY the both-detected condition,
which is an upper bound on the true pair-detection probability.

Consequence: μ_sel^(2) is OVERestimated. The inferred lensing rate
parameters (τ_2 amplitude, n_τ) are correspondingly BIASED LOW —
because the master likelihood's -μ_sel^(2) penalty for hypothesized
high rates is too strong. Shape parameters in p_pop are largely
unaffected (the pair-id step is approximately independent of
(m1_src, q, χ_eff) at fixed signal-to-noise).

We accept this for commit 4. If you have a calibrated p_tag(θ_src, μ_+, μ_-)
from the LVK pair-id pipeline, multiply each kept-source weight by it
before summing — the API exposes this via the ``log_p_tag_per_source``
argument (default zero).

Frozen detection realization: FIX THE COSMOLOGY
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Both lensed channels (this one and ``compute_lensed_single_selection_term``)
reweight SOURCE-FRAME campaign columns (m1_src, q_src, z_src, chieff, y_source)
against a source-frame proposal, and the detection efficiency enters ONLY
through the subset membership — the per-image flags rendered at generation time
(``lensed_injections.py``). Detection is a function of the OBSERVABLES
(m1_det, dL_app = dL(z)/sqrt(mu)) and therefore of cosmology, so at fixed
(m1_src, q, z, y) a change in H0 (or w0/wa) changes P_det while the stored flags
cannot follow. These estimators are consequently valid only AT THE CAMPAIGN'S
FIDUCIAL COSMOLOGY: they target

    ∫ dθ_src dy τ_2 p_pop p_z(z|Θ) P_det(θ_app(θ_src, Θ_campaign))

with the efficiency pinned to Θ_campaign. The unlensed singleton campaign is
stored in the (m1det, q, dL, chieff) basis with p_draw in that same basis
precisely so that ITS flags are cosmology-independent and
``compute_selection_term`` is a valid estimator for any Θ; the two channels
summed into μ_tot therefore obey different selection conventions once cosmology
is sampled, and the ratio μ_sel^(2)/μ_sel^(1) that sets A_tau picks up a
spurious cosmology dependence. (Note that ``cosmo``/``survey`` reach
``_per_source_log_weight`` unused: only p_z sees cosmology, through the
closure — the tell-tale that no efficiency is recomputed.)

Run lensed channels with the cosmology FIXED (``--fix_cosmology true``, the
CLI default). Removing the restriction needs the campaign's full draw set plus
an analytic efficiency recomputed at the proposed cosmology (weight each source
by P_det(x_+) P_det(x_-), resp. the exactly-one-detected combination, from
``darksirens.lensing.fcpdet`` and the stored fc_rho_thr/fc_r0/fc_mc_bar attrs);
the file keeps only the already-selected subsets and records no fiducial
cosmology, so it cannot be done from the current format — and nothing here can
detect the violation, since H0 arrives traced.

Pair-orientation caveat (mode-dependent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ``p_det^(1)(θ_app,+) · p_det^(1)(θ_app,-)`` product above describes the
GENERATOR'S DEFAULT rendering, which treats the two images' orientation
factors as INDEPENDENT draws. The two images of one source actually share
their inclination and polarization, which makes their Finn-Chernoff Thetas
partially correlated (the antenna response decorrelates over the day-to-month
SIS delays, the inclination does not). This estimator itself is CONVENTION-
AGNOSTIC: it sums the injection file's RENDERED per-image detection flags, so
its convention is whatever ``scripts/mock_lensing/generate_mock_lensing.py``
rendered — ``--pair-orientation-mode independent`` (default; the product
form above) or ``--pair-orientation-mode shared_iota`` (one
(iota, psi, alpha, delta) per source, antenna at the two arrival times).
What must be kept matched by hand is the inference-side censoring factor:
run ``darksirens_inference_lensing --pair_orientation_mode`` with the SAME
mode the campaign was rendered with (the gap between conventions is up to
~2.6x on P(both) — and the J=2 / lensed-singleton ratio is exactly what sets
A_tau). See ``darksirens.lensing.fcpdet`` for the joint model and numbers.

N_eff condition
~~~~~~~~~~~~~~~
Mandel-Farr-Gair: require N_eff > 5 · N_clusters_observed where
N_clusters_observed is the number of candidate pairs in the data.
Below threshold, the selection-correction term diverges in variance
and the master log-likelihood is set to -inf.

References
~~~~~~~~~~
- Mandel, Farr & Gair (2019), MNRAS 486, 1086.
- Loredo (2004), AIP Conf. Proc. 735, 195 (the Poisson likelihood with
  selection-corrected normalization).
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.lensing.lensed_injections import LensedInjectionSet
from darksirens.lensing.slmarks import SISLensParams, tau_2_prob
from darksirens.utils.utils import logdiffexp
from darksirens.likelihood.selection import (
    DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    selection_log_correction,
)


def _neff_from_log_mu_sigma2(
    log_mu: jnp.ndarray, log_sigma2: jnp.ndarray
) -> jnp.ndarray:
    """``N_eff = μ̂² / Var(μ̂)`` with the same semantics as the standard
    singleton path (``likelihood/selection.py:_lse_to_log_mu_neff``).

    Only ``log_mu`` finiteness gates the estimate.  ``log_sigma2 = -inf`` is
    the EXACT-ZERO-VARIANCE case — a deterministic (or single-support)
    selection estimate — whose effective sample size is INFINITE, not zero:
    requiring ``isfinite(log_sigma2)`` (as the cluster stack did) mapped it to
    ``N_eff = 0``, which trips the sparse-N_eff guard and returns ``-inf`` for
    a selection integral that is in fact known perfectly.  ``log_mu = -inf``
    (empty channel) still gives ``N_eff = 0`` so the too-sparse verdict fires,
    and a NaN ``log_sigma2`` propagates to a NaN ``N_eff``, which the guard's
    ``~(N_eff > threshold)`` also rejects.

    Both branch arguments are sanitized before the ``exp`` so no NaN/inf rides
    the dead branch of a ``where`` into the backward pass (see
    ``redshift/catalog.py:_logsumexp_neginf_safe`` for that reverse-mode class).
    """
    finite_mu = jnp.isfinite(log_mu)
    zero_var = log_sigma2 == -jnp.inf
    log_mu_safe = jnp.where(finite_mu, log_mu, 0.0)
    log_sigma2_safe = jnp.where(zero_var, 0.0, log_sigma2)
    ratio = jnp.exp(2.0 * log_mu_safe - log_sigma2_safe)
    return jnp.where(finite_mu, jnp.where(zero_var, jnp.inf, ratio), 0.0)


def _per_source_log_weight(
    injections: LensedInjectionSet,
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    sis_params: SISLensParams,
    log_p_pop_fn: Callable,
    log_prior_z_fn: Callable,
    log_p_tag_per_source: jnp.ndarray,
) -> jnp.ndarray:
    """Compute log w_s for each kept-source row of the injection set.

    Returns
    -------
    log_w : (N_kept,)
        log of the per-source importance weight. -inf for any row that
        is padded (valid=False), has zero proposal density, or whose
        weight is non-finite for any other reason.
    """
    # Population term in source-frame coordinates
    log_pop = log_p_pop_fn(
        injections.m1_src,
        injections.q_src,
        injections.z_src,
        injections.chieff,
        pop_params,
    )

    # Redshift prior (comoving volume element for spectral sirens; same
    # function darksirens uses for unlensed injections)
    log_pz = log_prior_z_fn(
        injections.z_src,
        jnp.zeros_like(injections.source_id_kept),  # dummy pixel
        catalog,
    )

    # SIS optical depth at z_s
    log_tau = jnp.log(tau_2_prob(injections.z_src, sis_params))

    # Impact-parameter prior p(y) = 2y
    log_py = jnp.log(2.0 * injections.y_source)

    # Proposal density correction
    log_p_prop = jnp.log(injections.p_prop_src) + jnp.log(injections.p_prop_y)

    log_w_raw = (
        log_pop + log_pz + log_tau + log_py
        - log_p_prop
        + log_p_tag_per_source
    )

    valid_mask = injections.valid & (injections.p_prop_src > 0.0) & (injections.p_prop_y > 0.0)
    log_w = jnp.where(valid_mask & jnp.isfinite(log_w_raw), log_w_raw, -jnp.inf)
    return log_w


def compute_cluster_selection_term(
    injections: LensedInjectionSet,
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    sis_params: SISLensParams,
    log_p_pop_fn: Callable,
    log_prior_z_fn: Callable,
    log_p_tag_per_source: jnp.ndarray | None = None,
) -> tuple:
    """Importance-sampled cluster (J=2) selection integral.

    Valid only at the campaign's fiducial cosmology — the rendered detection
    flags freeze the efficiency (see the module docstring's "Frozen detection
    realization" section).

    Parameters
    ----------
    injections
        Precomputed LensedInjectionSet (both-detected sources already
        selected at load time; see lensed_injections.py).
    cosmo, survey, pop_params, catalog, sis_params
        Hyperparameter containers.
    log_p_pop_fn
        ``log p_pop(m1_src, q, z_s, chieff, pop_params)`` callable.
    log_prior_z_fn
        ``log p_z(z_s, pix, catalog)`` callable.
    log_p_tag_per_source
        Optional (N_kept,) array of log pair-tag probability per source.
        Pass zeros (default) for the both-detected approximation.

    Returns
    -------
    log_mu : scalar
        log of the importance-sampled estimate of μ_sel^(2).
    Neff : scalar
        Effective sample size of the cluster channel alone.
    log_sigma2 : scalar
        log of the variance Var(μ̂_sel^(2)). Required for the combined
        singleton + cluster marginalization (see
        ``combined_selection_log_correction``).
    """
    n_kept = injections.source_id_kept.shape[0]
    if log_p_tag_per_source is None:
        log_p_tag_per_source = jnp.zeros(n_kept, dtype=jnp.float64)

    log_w = _per_source_log_weight(
        injections, cosmo, survey, pop_params, catalog, sis_params,
        log_p_pop_fn, log_prior_z_fn, log_p_tag_per_source,
    )

    log_sum_w = logsumexp(log_w)
    log_n_draw = jnp.log(injections.n_draw_sources)
    log_mu = log_sum_w - log_n_draw

    log_sum_w2 = logsumexp(2.0 * log_w)
    log_var_term = logdiffexp(log_sum_w2, 2.0 * log_sum_w - log_n_draw)
    # Empty channel: variance is exactly zero; logdiffexp(-inf, -inf) is
    # NaN — pin to -inf (same guard as the lensed-single term).
    log_sigma2 = jnp.where(
        jnp.isfinite(log_sum_w), log_var_term - 2.0 * log_n_draw, -jnp.inf
    )

    Neff = _neff_from_log_mu_sigma2(log_mu, log_sigma2)
    return log_mu, Neff, log_sigma2


def compute_lensed_single_selection_term(
    singles,
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    catalog: EMCatalog,
    sis_params: SISLensParams,
    log_p_pop_fn: Callable,
    log_prior_z_fn: Callable,
) -> tuple:
    """Importance-sampled selection integral for the lensed-singleton channel:
    strongly lensed sources with EXACTLY ONE detected image, observed as
    ordinary singletons.

    Uses the same per-source weight convention as the cluster (J=2) term —
    w_s = p_pop * p_z * tau_2 * p(y) / (p_prop_src * p_prop_y) — over the
    exactly-one-detected subset (see LensedSingleImageSet). No pair-tag
    factor: single images are never pair-tagged. Detection realization is
    already encoded in the subset membership, exactly like the both-detected
    estimator; no P_det factors appear here — so this estimator, too, is valid
    only at the campaign's fiducial cosmology (module docstring, "Frozen
    detection realization").

    Returns (log_mu, Neff, log_sigma2) with the same conventions as
    compute_cluster_selection_term.
    """
    log_pop = log_p_pop_fn(
        singles.m1_src, singles.q_src, singles.z_src, singles.chieff, pop_params,
    )
    log_pz = log_prior_z_fn(
        singles.z_src,
        jnp.zeros(singles.m1_src.shape[0], dtype=jnp.int32),
        catalog,
    )
    log_tau = jnp.log(tau_2_prob(singles.z_src, sis_params))
    log_py = jnp.log(2.0 * singles.y_source)
    log_p_prop = jnp.log(singles.p_prop_src) + jnp.log(singles.p_prop_y)

    log_w_raw = log_pop + log_pz + log_tau + log_py - log_p_prop
    valid = singles.valid & (singles.p_prop_src > 0.0) & (singles.p_prop_y > 0.0)
    log_w = jnp.where(valid & jnp.isfinite(log_w_raw), log_w_raw, -jnp.inf)

    log_sum_w = logsumexp(log_w)
    log_n_draw = jnp.log(singles.n_draw_sources)
    log_mu = log_sum_w - log_n_draw

    log_sum_w2 = logsumexp(2.0 * log_w)
    log_var_term = logdiffexp(log_sum_w2, 2.0 * log_sum_w - log_n_draw)
    # Empty channel (all weights -inf, e.g. tau_A = 0): the variance is
    # exactly zero, but logdiffexp(-inf, -inf) is NaN — pin it to -inf so
    # the channel drops out of combined sums instead of NaN-poisoning them.
    log_sigma2 = jnp.where(
        jnp.isfinite(log_sum_w), log_var_term - 2.0 * log_n_draw, -jnp.inf
    )

    Neff = _neff_from_log_mu_sigma2(log_mu, log_sigma2)
    return log_mu, Neff, log_sigma2


def combined_selection_log_correction(
    log_mu_singleton: jnp.ndarray,
    log_sigma2_singleton: jnp.ndarray,
    log_mu_cluster: jnp.ndarray,
    log_sigma2_cluster: jnp.ndarray,
    n_singletons_observed: int,
    n_clusters_observed: int,
    soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    pe_variance_sum=0.0,
) -> jnp.ndarray:
    """Master-likelihood selection-correction for the marked-Poisson
    singleton + J=2 cluster model.

    Derivation
    ~~~~~~~~~~
    The marked-Poisson formulation has
        N_tot     = N_singletons + N_clusters
        μ_tot     = μ_sel^(1) + μ_sel^(2)
        σ²_tot    = σ²(μ̂^(1)) + σ²(μ̂^(2))     (independent channels)
        N_eff,tot = μ_tot² / σ²_tot

    After marginalizing over the population rate R_0 with a 1/R_0
    Jeffreys prior (Mandel-Farr-Gair 2019), the log-likelihood gains:

        -N_tot · log(μ_tot)  +  N_tot (N_tot + 3) / (2 · N_eff,tot)

    The variance term diverges when N_eff,tot drops below ~5 N_tot;
    we gate the correction with -inf in that regime (standard MFG).

    Parameters
    ----------
    log_mu_singleton, log_sigma2_singleton
        log estimates of μ_sel^(1) and Var(μ̂^(1)).
    log_mu_cluster, log_sigma2_cluster
        log estimates of μ_sel^(2) and Var(μ̂^(2)). If the analysis has
        no cluster channel (commit-2 style), pass log_mu_cluster = -inf
        and log_sigma2_cluster = -inf — the function reduces exactly to
        the singleton form.
    n_singletons_observed
        Number of declared singleton events (events NOT in any pair).
    n_clusters_observed
        Number of declared candidate pair clusters.
    max_likelihood_variance, pe_variance_sum
        Total-log-likelihood variance budget per GWTC-4/5, forwarded to
        ``selection_log_correction``. The cluster stack threads the summed
        per-event and per-pair delta-method variances here, so the guard
        bounds the TOTAL log-likelihood variance, not selection alone. Caveat:
        the per-pair term covers only each branch's DRIVING sample set — the
        partner event's KDE sampling noise is invisible to it (see
        ``cluster_likelihood.cluster_log_likelihood_pair``), so for the pair
        channel the budget is spent against a lower bound on the true variance.

    Returns
    -------
    ll : scalar
        Selection-correction contribution to log L. -inf when
        N_eff,tot ≤ 5 N_tot.

    Backward compatibility
    ----------------------
    When log_mu_cluster = -inf, this collapses to:
        log_mu_tot = log_mu_singleton
        log_sigma2_tot = log_sigma2_singleton
        N_eff = N_eff_singleton
        N_tot = N_singletons
    which is bit-identical to the original ``selection_log_correction``.
    A test enforces this.
    """
    log_mu_tot = jnp.logaddexp(log_mu_singleton, log_mu_cluster)
    log_sigma2_tot = jnp.logaddexp(log_sigma2_singleton, log_sigma2_cluster)
    # Zero total variance (every contributing channel deterministic) means an
    # INFINITE combined ESS -- see _neff_from_log_mu_sigma2.
    Neff_tot = _neff_from_log_mu_sigma2(log_mu_tot, log_sigma2_tot)
    n_tot = int(n_singletons_observed) + int(n_clusters_observed)
    if n_tot <= 0:
        return jnp.asarray(0.0)
    # Delegate to the singleton correction so the sparse-Neff guard semantics
    # (hard -inf for nested samplers, reward-tracking smooth wall for
    # gradient-based sampling) stay defined in exactly one place — the
    # cluster stack previously had only the hard wall and reproduced the
    # 100%-divergent-NUTS failure the main CLI was fixed for (library
    # review, lensing-stack finding 4).
    return selection_log_correction(
        log_mu_tot,
        Neff_tot,
        n_tot,
        soft_guard=soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        pe_variance_sum=pe_variance_sum,
    )
