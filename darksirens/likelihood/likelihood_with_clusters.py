"""
likelihood_with_clusters.py
---------------------------
Master likelihood for the marked-Poisson (singletons + J=2 clusters)
population inference (commit 4).

Architectural choice
~~~~~~~~~~~~~~~~~~~~
Rather than patch ``likelihood_core.darksiren_log_likelihood`` again
(which already grew in commit 2 to support spectral_sirens_wl), we
build a thin wrapper that:

  1. Computes the singleton selection μ_sel^(1), σ²^(1), and the
     singleton per-event log-likelihood contributions — for the events
     declared SINGLETONS, not all events.
  2. Computes the cluster selection μ_sel^(2), σ²^(2).
  3. Computes the per-pair log-likelihoods L_2 from commit 3.
  4. Combines via the marked-Poisson Mandel-Farr-Gair correction:
        ll_master = Σ_i log L_i + Σ_{(i,j)} log L_2_{ij}
                  + combined_selection_log_correction(...)

This keeps commit 2's hot path completely intact for analyses with no
clusters. When ``cluster_set.npairs == 0``, the wrapper produces
bit-identical output to commit 2's ``darksiren_log_likelihood`` (a
regression test enforces this).

What this commit does NOT do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- It does NOT marginalize over the partition. The user supplies the
  partition (singleton indices + pair indices) via a ClusterSet at
  data-load time. Partition-marginalization is a deeper extension.
- It does NOT support J=4 (quads). Wiring is present but inert.
- It uses the both-detected approximation for p_det^(2). See
  ``cluster_selection.py`` module docstring for the documented bias.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from darksirens.redshift import get_redshift_prior
from darksirens.gw.populations import pop_model_parser
from darksirens.likelihood.selection import (
    DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    compute_selection_term,
    log_evidence_and_mc_variance,
)
from darksirens.inference.utils import log_sample_weight
from darksirens.likelihood.wl_weight import (
    log_sample_weight_wl_or_standard,
    log_sample_weight_wl_lognormal_hermite,
)
from darksirens.likelihood.cluster_likelihood import (
    _correlated_branch_variance,
    cluster_log_likelihood_pair,
    lensed_single_log_likelihood_event,
)
from darksirens.likelihood.cluster_selection import (
    _neff_from_log_mu_sigma2,
    compute_cluster_selection_term,
    compute_lensed_single_selection_term,
    combined_selection_log_correction,
)
from darksirens.likelihood.pair_kde import _slice_event_kde_inside_jit
from darksirens.lensing.grids import (
    make_log_mu_grid, make_hermite_u_grid, make_y_grid,
    WL_MU_QUADRATURE_NODES, WL_MU_QUADRATURE_LOG_MU_RANGE,
)
from darksirens.lensing.wlmagnification import (
    make_tabulated_log_p_wl,
)
from darksirens.lensing.slmarks import SISLensParams, tau_2_prob
from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.utils.cosmology import (
    dL_of_z,
    threads_distance_table,
    z_of_dL_precomputed,
    zgrid as _cosmo_zgrid,
)


# Cluster-mode codes (static_argnames dispatch)
CLUSTER_MODE_OFF = 0          # singletons only — equivalent to commit 2
CLUSTER_MODE_J2 = 1           # singletons + J=2 pairs

# Quadrature node counts (same as commit 2 for WL; new for y). The tabulated
# backend's mu grid is owned by lensing.grids so the CLI's startup coverage
# check validates the SAME quadrature this module integrates on.
_WL_NMU_NODES = WL_MU_QUADRATURE_NODES
_WL_LOG_MU_RANGE = WL_MU_QUADRATURE_LOG_MU_RANGE
_WL_HERMITE_NODES = 16
_Y_NODES_FOR_CLUSTER_PAIR_LIKE = 32   # used inside L_2; finer than selection
                                       # since L_2 needs to localize y precisely

# WL backend codes (same as commit 2)
WL_BACKEND_DISABLED = -1
WL_BACKEND_LOGNORMAL = 0
WL_BACKEND_TABULATED = 1

# Singleton-selection weak-lensing policy.  STANDARD preserves the legacy
# commit-2/4 behavior: PE weights may be WL-marginalized, but singleton
# injections use the ordinary spectral-siren selection integral.  LOGNORMAL
# opts into the same lognormal/Hermite WL marginalization for singleton
# selection.  Cluster J=2 lensed-injection selection is intentionally not
# changed by this switch.
WL_SELECTION_STANDARD = 0
WL_SELECTION_LOGNORMAL = 1

PAIR_MARKS_NONE = 0
PAIR_MARKS_TIME = 1
# Delta-collapsed time mark: for sharp marks (sigma_dt << T0) the Gaussian
# pins y more tightly than any practical quadrature resolves; the y-integral
# is evaluated analytically at y* = dt/T0 (see cluster_likelihood
# PAIR_MARKS_DELTA_COLLAPSE). Selected automatically by the CLI when
# max(sigma_dt)/T0 is below the sharpness threshold.
PAIR_MARKS_TIME_DELTA = 2

# Lensed-singleton policy. OFF keeps the singleton channel purely unlensed
# (evidence: unlensed/WL only; selection: unlensed injections only) — the
# generator's default protocol that DROPS exactly-one-detected images. Note
# that OFF still gets the (1 - tau_2(z)) unlensed weight whenever a J=2 channel
# is present; see _unlensed_tau_suppression_enabled. MIXTURE
# models the realistic protocol where a strongly lensed source with exactly
# one detected image is observed as an ordinary singleton: the singleton
# evidence becomes (1 - tau_2(z)) x [WL branch] + tau_2(z)-weighted
# single-image branch with the analytic partner-censoring factor (fcpdet),
# and the singleton selection integral gains the exactly-one-detected
# lensed-injection subset. Enables the Mould-style per-event-only ablation
# via cluster_mode=OFF + MIXTURE.
SINGLETON_LENSING_OFF = 0
SINGLETON_LENSING_MIXTURE = 1


def _unlensed_tau_suppression_enabled(cluster_mode: int, singleton_lensing: int) -> bool:
    """Whether the singleton channel must carry the ``(1 - tau_2(z))`` factor.

    The singleton channel describes UNLENSED sources. Whenever a separate
    strongly-lensed channel is also summed into the Poisson normalization, the
    singleton intensity must be the unlensed one, ``(1 - tau_2) p_pop p_z``, or
    the strongly-lensed sources are counted twice: once in mu_sel^(1) (as if
    unlensed at mu = 1) and again in the lensed channel, while the observed data
    assign them only to the lensed channel.

    That is true in two cases:

    * ``singleton_lensing = MIXTURE`` -- the exactly-one-detected lensed
      channel is added to mu_sel^(1) (any cluster_mode).
    * ``cluster_mode = J2`` -- mu_sel^(2) is added to mu_tot. This case was
      MISSING: with the shipped default (cluster_mode=j2, singleton_lensing=off)
      mu_sel^(1) over-counted by ``int tau_2 P_det p_pop p_z``, an A_tau- and
      n_tau-dependent error in the ``-N_tot log(mu_tot)`` term.

    The mock generator confirms the convention: it draws
    ``is_double ~ Bernoulli(tau_2(z))`` per source and routes only the
    ``~is_double`` sources through the singleton detection pipeline, so the
    rendered singleton channel is (1 - tau_2)-weighted by construction.

    With ``cluster_mode = OFF`` and ``singleton_lensing = OFF`` there is no
    second channel and the unsuppressed integral is the correct
    "everything is a singleton" normalization -- the commit-2 reduction stays
    bit-identical.
    """
    return (
        singleton_lensing == SINGLETON_LENSING_MIXTURE
        or cluster_mode == CLUSTER_MODE_J2
    )


def _has_counterpart_arrays(catalog) -> bool:
    """True when the PE catalog carries per-event bright-siren counterpart
    arrays, i.e. when ``active_counterpart_index`` actually selects something.

    The block-vectorized singleton reduction evaluates a chunk of events in one
    flattened call and so cannot set a per-event index; presence of ANY
    counterpart array forces the exact per-event scan, keeping the bright path
    bit-identical to the historical body.  Same gate as
    ``likelihood/core.py``'s ``has_counterpart``.
    """
    return any(
        getattr(catalog, name, None) is not None
        for name in ("counterpart_pixels", "counterpart_zs", "counterpart_dzs")
    )


def _fold_shared_campaign_covariance(
    log_sigma2_2: jnp.ndarray, log_cov2: jnp.ndarray
) -> jnp.ndarray:
    """``log(sigma2_2 - cov2)`` in place: the P2-07 shared-campaign covariance
    fold for the channel-2 variance slot.

    logdiffexp, NaN-safe for dead channels (either side ``-inf`` makes the
    correction a no-op) AND for the SATURATED regime ``cov2 >= sigma2_2``,
    where the corrected slot clamps to zero variance.  Double-where
    discipline (same reverse-mode class as ``selection._lse_to_log_mu_neff``):
    the clamp's forward value ``log1p(-exp(0)) = -inf`` is correct, but
    ``log1p``'s slope there is infinite and the zero cotangent arriving
    through the downstream ``logaddexp`` multiplies it into NaN, poisoning the
    gradient of every hyperparameter feeding ``log_mu_1L``/``log_mu_2`` — so
    the saturated branch must be a CONSTANT ``-inf`` with no gradient path,
    and every dead branch's operand must keep a finite slope.
    """
    both_live = jnp.isfinite(log_sigma2_2) & jnp.isfinite(log_cov2)
    sat = both_live & (log_cov2 >= log_sigma2_2)
    safe_hi = jnp.where(both_live, log_sigma2_2, 0.0)
    safe_lo = jnp.where(both_live, jnp.minimum(log_cov2, safe_hi), -1.0)
    x = jnp.where(sat, -1.0, jnp.minimum(safe_lo - safe_hi, 0.0))
    corrected = jnp.where(sat, -jnp.inf, safe_hi + jnp.log1p(-jnp.exp(x)))
    return jnp.where(both_live, corrected, log_sigma2_2)


# ``threads_distance_table`` is ``jax.jit`` plus the contract that the 106.8 MB
# comoving-distance table arrives as an ARGUMENT rather than a closed-over global
# (a ``dense<>`` HLO constant worth ~214 MB of module text per embedding).  The
# lensing CLI calls this jit directly, so it is the outermost boundary on that
# path.  See ``utils.cosmology``.
@threads_distance_table(
    static_argnames=[
        "nEvents",
        "nsamp",
        "n_singletons",
        "n_pairs",
        "pop_model",
        "universe_model",
        "sel_batch_size",
        "wl_backend",
        "cluster_mode",
        "return_diagnostics",
        "wl_selection",
        "pair_marks",
        "pair_time_signed",
        "pair_batch_size",
        "y_nodes_pair",
        "singleton_lensing",
        "y_nodes_single",
        "selection_neff_soft_guard",
        "pair_orientation_mode",
        "pe_event_block",
    ],
)
def darksiren_log_likelihood_with_clusters(
    cosmo: CosmoParams,
    survey: SurveyParams,
    pop_params: jnp.ndarray,
    gw_pe: GWEvent,
    em_catalog_pe: EMCatalog,
    gw_sel: GWEvent,
    em_catalog_sel: EMCatalog,
    nEvents: int,
    nsamp: int,
    Ndraw: float,
    # Partition information
    singleton_indices: jnp.ndarray,    # (n_singletons,) int32 — event indices NOT in pairs
    pair_indices: jnp.ndarray,         # (n_pairs, 2) int32
    n_singletons: int,
    n_pairs: int,
    # Cluster-channel inputs (inert when cluster_mode = OFF)
    lensed_injections,                 # LensedInjectionSet or None
    pair_kdes,                         # tuple of PairKDE or None
    sis_params: SISLensParams,
    log_p_tag_per_source: jnp.ndarray,  # (N_kept,) or 0-d zero array
    # Static dispatch
    pop_model: str,
    universe_model: str,
    sel_batch_size: int | None = None,
    cluster_mode: int = CLUSTER_MODE_OFF,
    # WL hyperparameters (commit 2)
    wl_backend: int = WL_BACKEND_DISABLED,
    wl_a: float = 0.0,
    wl_b: float = 0.0,
    wl_z_grid: jnp.ndarray | None = None,
    wl_log_mu_grid: jnp.ndarray | None = None,
    wl_log_p_table: jnp.ndarray | None = None,
    return_diagnostics: bool = False,
    wl_selection: int = WL_SELECTION_STANDARD,
    pair_marks: int = PAIR_MARKS_NONE,
    pair_time_delta_t_obs: jnp.ndarray | None = None,
    pair_time_sigma: jnp.ndarray | None = None,
    pair_time_t_obs_window_sec: jnp.ndarray | None = None,
    pair_time_signed: bool = False,
    pair_batch_size: int = 0,
    y_nodes_pair: int = _Y_NODES_FOR_CLUSTER_PAIR_LIKE,
    singleton_lensing: int = SINGLETON_LENSING_OFF,
    lensed_singles=None,               # LensedSingleImageSet (MIXTURE only)
    fc_pdet_params=None,               # FCPdetParams (MIXTURE only)
    y_nodes_single: int = 32,
    selection_neff_soft_guard: bool = False,
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    pair_orientation_mode: str = "independent",
    # Events-axis block size for the singleton PE reduction (static Python).
    # None => all singletons in one vectorized block (fastest); an integer
    # chunks them, trading throughput for peak memory.  1 reproduces the
    # historical per-event scan row by row.
    pe_event_block: int | None = None,
    # Comoving-distance interpolation table, threaded as a jit ARGUMENT and bound
    # as the active table for this trace (see ``utils.cosmology``).  None resolves
    # to whatever is active in the CALLER's scope.  Must stay the TRAILING
    # parameter (threads_distance_table contract).
    distance_table: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Master log-likelihood with singleton + J=2 cluster channels.

    See module docstring. When ``cluster_mode == CLUSTER_MODE_OFF`` the
    function reduces bit-identically to commit 2's
    ``darksiren_log_likelihood``, summing over all ``nEvents`` events
    as singletons and ignoring the cluster inputs.

    When ``cluster_mode == CLUSTER_MODE_J2`` the function:
      - Sums singleton likelihoods over event indices listed in
        ``singleton_indices``.
      - Sums pair likelihoods over ``pair_indices``.
      - Combines singleton + cluster selection corrections via the
        marked-Poisson MFG formula.

    Parameters
    ----------
    pair_kdes
        If cluster_mode==J2, this is a tuple of nEvents PairKDE objects,
        indexed by the global event index. The pair-likelihood loop
        looks up the two PairKDEs by pair_indices[k, 0] and
        pair_indices[k, 1].

    Notes on JIT
    ------------
    ``pair_kdes`` is a Python tuple of NamedTuples — passed through as a
    pytree, JIT-traced over its leaves but iterated over its Python-level
    structure. nEvents is a static argname so the tuple unpack is
    resolved at trace time.
    """
    pop_params_shape = tuple(pop_params.shape)
    if pop_params.ndim == 0 or pop_params_shape[0] == 0:
        raise ValueError(
            "darksiren_log_likelihood_with_clusters received empty pop_params: "
            f"pop_model={pop_model!r}, pop_params.shape={pop_params_shape}. "
            "Verify parameter-space construction for this population model."
        )

    if pe_event_block is not None and pe_event_block < 1:
        raise ValueError(
            f"pe_event_block must be a positive integer or None; got "
            f"{pe_event_block}."
        )

    log_p_pop = pop_model_parser(pop_model=pop_model)

    wl_enabled = (wl_backend != WL_BACKEND_DISABLED)
    wl_selection_enabled = (
        wl_selection == WL_SELECTION_LOGNORMAL
        and wl_backend == WL_BACKEND_LOGNORMAL
    )
    if wl_selection == WL_SELECTION_LOGNORMAL and wl_backend == WL_BACKEND_TABULATED:
        # Mirror likelihood/core.py: silently downgrading to STANDARD here
        # would leave mu(Lambda) marginalized under a different observation
        # model than the per-event weights — a static configuration error,
        # not a fallthrough.
        raise ValueError(
            "wl_selection=WL_SELECTION_LOGNORMAL requires "
            "wl_backend=WL_BACKEND_LOGNORMAL; the tabulated backend has no "
            "matched selection integral, so the Hermite mu-marginalization the "
            "PE term applies cannot be applied to the injections. Use the "
            "lognormal backend, or pass wl_selection=WL_SELECTION_STANDARD for "
            "a deliberately mismatched ablation."
        )
    if wl_selection == WL_SELECTION_LOGNORMAL and wl_backend == WL_BACKEND_DISABLED:
        # Disabled WL must keep the exact legacy singleton-selection path:
        # with no WL anywhere, STANDARD selection is exact, so this downgrade
        # is a no-op rather than a mismatch (backward compatibility).
        wl_selection_enabled = False
    unlensed_tau_suppression = _unlensed_tau_suppression_enabled(
        cluster_mode, singleton_lensing
    )
    if singleton_lensing == SINGLETON_LENSING_MIXTURE and (
        lensed_singles is None or fc_pdet_params is None
    ):
        raise ValueError(
            "singleton_lensing=MIXTURE requires lensed_singles (the "
            "exactly-one-detected lensed-injection subset) and fc_pdet_params."
        )
    if pair_orientation_mode not in ("independent", "shared_iota"):
        raise ValueError(
            f"unknown pair_orientation_mode {pair_orientation_mode!r}; "
            "expected 'independent' or 'shared_iota'"
        )
    if wl_enabled and universe_model != "spectral_sirens_wl":
        raise ValueError(
            f"wl_backend={wl_backend} requires universe_model='spectral_sirens_wl', "
            f"got {universe_model!r}"
        )
    if universe_model == "spectral_sirens_wl" and not wl_enabled:
        raise ValueError(
            "universe_model='spectral_sirens_wl' requires wl_backend in {0, 1}."
        )

    pe_prior_name = "spectral_sirens" if wl_enabled else universe_model
    sel_prior_name = (
        "spectral_sirens" if universe_model in ("bright_sirens", "spectral_sirens_wl")
        else universe_model
    )
    raw_logPriorUniv = get_redshift_prior(pe_prior_name)
    raw_logPriorSelection = get_redshift_prior(sel_prior_name)
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    # ONE luminosity-distance grid per proposal, shared by every clamp and
    # inversion below (likelihood/core.py does the same).  ``dL_grid_bounds``,
    # ``z_of_dL`` and ``dL_in_z_grid`` each rebuild ``dL_of_z(zgrid, ...)`` from
    # the 4-D table, and the Hermite WL kernel rebuilt it three more times per
    # weight call when handed no grid.  Same arrays through the same
    # arithmetic, so every consumer is bit-identical to its rebuilding form.
    dL_grid = dL_of_z(_cosmo_zgrid, H0, Om0, w0, wa)
    dL_lo, dL_hi = dL_grid[0], dL_grid[-1]

    def log_prior_z(z, pix, catalog):
        return raw_logPriorUniv(z, pix, cosmo, survey, catalog)

    def log_prior_z_selection(z, pix, catalog):
        return raw_logPriorSelection(z, pix, cosmo, survey, catalog)

    # WL quadrature setup (identical to commit 2)
    if wl_enabled:
        if wl_backend == WL_BACKEND_LOGNORMAL:
            u_nodes, log_wH_nodes = make_hermite_u_grid(_WL_HERMITE_NODES)
            mu_nodes = jnp.zeros(1, dtype=jnp.float64)
            log_w_nodes = jnp.zeros(1, dtype=jnp.float64)
            log_p_wl_fn = None
        elif wl_backend == WL_BACKEND_TABULATED:
            mu_nodes, log_w_nodes = make_log_mu_grid(_WL_NMU_NODES, _WL_LOG_MU_RANGE)
            # validate=False: this factory runs inside the jitted likelihood
            # body, where the grids are tracers and the value checks would raise
            # TracerBoolConversionError.  The table is validated eagerly at load
            # time (cli/inference_lensing._load_wl_table_arrays).
            log_p_wl_fn = make_tabulated_log_p_wl(
                wl_z_grid, wl_log_mu_grid, wl_log_p_table, validate=False,
            )
            u_nodes = jnp.zeros(1, dtype=jnp.float64)
            log_wH_nodes = jnp.zeros(1, dtype=jnp.float64)
        else:
            raise ValueError(f"Unknown wl_backend={wl_backend}")
    else:
        mu_nodes = jnp.zeros(1, dtype=jnp.float64)
        log_w_nodes = jnp.zeros(1, dtype=jnp.float64)
        u_nodes = jnp.zeros(1, dtype=jnp.float64)
        log_wH_nodes = jnp.zeros(1, dtype=jnp.float64)
        log_p_wl_fn = None

    # ──────────────────────────────────────────────────────────────────
    # Singleton selection integral.  By default this uses the standard
    # log_sample_weight with the volume prior — unchanged from commit 2.
    # If explicitly requested via wl_selection=LOGNORMAL, and only for the
    # lognormal WL backend with nonzero wl_a, singleton injections use the
    # same Hermite WL marginalization as PE samples.  The wl_backend=disabled
    # case deliberately falls through to the standard path; wl_a=0 is handled
    # by the Hermite kernel itself and reduces numerically to the standard path.
    # ──────────────────────────────────────────────────────────────────
    def log_weight_sel(m1det, q, dL, chieff, pix, prior_wt, catalog):
        def _selection_prior(z, pix, catalog):
            return log_prior_z_selection(z, pix, catalog)
        # Clamp-then-mask, mirroring likelihood/core.py: an out-of-support dL
        # hits z_of_dL's NaN sentinel inside the weight arithmetic and
        # NaN-poisons the reverse pass despite the -inf mask (library review,
        # lensing-stack finding 1: the cluster wrappers missed the e39dc11
        # fix). Full CPL bounds: the old dL_in_z_grid(dL, H0, Om0) silently
        # dropped w0/wa (finding 5).
        supported = (dL >= dL_lo) & (dL <= dL_hi)
        dL_c = jnp.clip(dL, dL_lo, dL_hi)
        if wl_selection_enabled:
            ldw = log_sample_weight_wl_lognormal_hermite(
                m1det, q, dL_c, chieff, pix, prior_wt,
                cosmo, survey, pop_params, catalog,
                log_p_pop, _selection_prior,
                wl_a, wl_b, u_nodes, log_wH_nodes,
                dL_grid=dL_grid,
            )
        else:
            ldw = log_sample_weight(
                m1det, q, dL_c, chieff, pix, prior_wt,
                cosmo, survey, pop_params, catalog,
                log_p_pop, _selection_prior,
                dL_grid=dL_grid,
            )
        if unlensed_tau_suppression:
            # Poisson consistency: the singleton EVIDENCE suppresses the
            # unlensed branch by (1 - tau_2(z)) (see the PE reduction below),
            # so mu_sel^(1) must integrate the SAME unlensed rate density
            # (1 - tau_2) x p_pop x p_z x P_det. Without this factor
            # mu_sel^(1) over-counts the unlensed contribution by
            # ∫ tau_2 P_det p_pop p_z, an O(A_tau) term the same order as the
            # lensed channels added below — it biases the strong-lensing rate
            # parameters (A_tau, n_tau). tau_2 is evaluated
            # at the mu = 1 apparent redshift, mirroring the evidence side's
            # stated approximation (the mu-dependence of tau_2 is O(tau x s^2)).
            z_app_sel = z_of_dL_precomputed(dL_c, dL_grid)
            tau_app_sel = tau_2_prob(z_app_sel, sis_params)
            ldw = ldw + jnp.log1p(-tau_app_sel)
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    log_mu_1, Neff_1, log_sigma2_1 = compute_selection_term(
        gw_sel, em_catalog_sel, log_weight_sel, Ndraw, nEvents,
        sel_batch_size=sel_batch_size,
    )
    log_mu_1L = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    log_sigma2_1L = jnp.asarray(-jnp.inf, dtype=jnp.float64)
    if singleton_lensing == SINGLETON_LENSING_MIXTURE:
        # Observed singletons are a mixture of unlensed sources and lensed
        # sources with exactly one detected image; mu^(1) must count both
        # or the singleton normalization is inconsistent with the evidence
        # mixture below.  The unlensed injections are an independent campaign
        # (means and variances add), but the lensed-single VARIANCE is
        # combined with the J=2 term's below — the two share a campaign.
        log_mu_1L, _Neff_1L, log_sigma2_1L = compute_lensed_single_selection_term(
            lensed_singles, cosmo, survey, pop_params, em_catalog_sel,
            sis_params, log_p_pop, log_prior_z_selection,
        )
        log_mu_1 = jnp.logaddexp(log_mu_1, log_mu_1L)
        log_sigma2_1 = jnp.logaddexp(log_sigma2_1, log_sigma2_1L)
        Neff_1 = _neff_from_log_mu_sigma2(log_mu_1, log_sigma2_1)

    # ──────────────────────────────────────────────────────────────────
    # Cluster selection integral. Inert when cluster_mode == OFF —
    # NOTE: wl_selection affects only singleton injections above.  The J=2
    # lensed-injection selection estimator remains unchanged in this PR.
    # log_mu = -inf and log_sigma2 = -inf so that
    # combined_selection_log_correction collapses to the singleton form.
    # ──────────────────────────────────────────────────────────────────
    if cluster_mode == CLUSTER_MODE_J2:
        log_mu_2, Neff_2, log_sigma2_2 = compute_cluster_selection_term(
            lensed_injections, cosmo, survey, pop_params, em_catalog_sel,
            sis_params, log_p_pop, log_prior_z_selection,
            log_p_tag_per_source=log_p_tag_per_source,
        )
    else:
        log_mu_2 = jnp.asarray(-jnp.inf, dtype=jnp.float64)
        Neff_2 = jnp.asarray(0.0, dtype=jnp.float64)
        log_sigma2_2 = jnp.asarray(-jnp.inf, dtype=jnp.float64)

    # Correlated lensed-channel estimators (P2-07): the exactly-one-detected
    # (1L) and both-detected (2) estimators are built from MUTUALLY EXCLUSIVE
    # indicators over the SAME campaign draws, so their per-draw covariance is
    # exactly -mu_1L*mu_2/N and Var(mu_1L + mu_2) = Var_1L + Var_2
    # - 2*mu_1L*mu_2/N — algebraically the single-estimator variance of the
    # union sum (non-negative by Cauchy-Schwarz over the N campaign draws).
    # Treating them as independent overstated the total variance, biasing the
    # MFG finite-sample correction and spuriously tripping the Neff guard.
    # The adjustment is folded into the channel-2 slot; only the SUM of the
    # two slots enters combined_selection_log_correction.
    if (
        singleton_lensing == SINGLETON_LENSING_MIXTURE
        and cluster_mode == CLUSTER_MODE_J2
        and lensed_injections is not None
    ):
        log_cov2 = (
            jnp.log(2.0) + log_mu_1L + log_mu_2
            - jnp.log(lensed_injections.n_draw_sources)
        )
        log_sigma2_2 = _fold_shared_campaign_covariance(log_sigma2_2, log_cov2)

    # The combined selection correction is evaluated AFTER the singleton and
    # pair reductions below: the total-variance guard budgets the per-event
    # and per-pair importance-estimator variances alongside the selection
    # variance (mirroring the standard core), so their sums must exist first.
    log_mu_combined = jnp.logaddexp(log_mu_1, log_mu_2)
    log_sigma2_combined = jnp.logaddexp(log_sigma2_1, log_sigma2_2)
    # Same helper the gate itself uses (combined_selection_log_correction), so
    # the reported N_eff cannot disagree with the one the guard acted on: the
    # inline copy required isfinite(log_sigma2) and so reported N_eff = 0 for an
    # exactly-known (zero-variance) selection integral alongside a finite
    # logL_total, which reads as a contradiction during triage.
    Neff_combined = _neff_from_log_mu_sigma2(log_mu_combined, log_sigma2_combined)
    ll = jnp.asarray(0.0, dtype=jnp.float64)

    # ──────────────────────────────────────────────────────────────────
    # PE per-event term for singletons.
    # When cluster_mode == OFF: iterate over all nEvents (legacy behaviour).
    # When cluster_mode == J2: iterate over singleton_indices only.
    # ──────────────────────────────────────────────────────────────────
    def _log_sample_weight_if_supported(m1det, q, dL, chieff, pix, prior_wt, catalog):
        # Clamp-then-mask + full CPL bounds (see log_weight_sel above).
        supported = (dL >= dL_lo) & (dL <= dL_hi)
        dL_c = jnp.clip(dL, dL_lo, dL_hi)
        if wl_enabled and wl_backend == WL_BACKEND_LOGNORMAL:
            ldw = log_sample_weight_wl_lognormal_hermite(
                m1det, q, dL_c, chieff, pix, prior_wt,
                cosmo, survey, pop_params, catalog,
                log_p_pop, log_prior_z,
                wl_a, wl_b, u_nodes, log_wH_nodes,
                dL_grid=dL_grid,
            )
        else:
            ldw = log_sample_weight_wl_or_standard(
                m1det, q, dL_c, chieff, pix, prior_wt,
                cosmo, survey, pop_params, catalog,
                log_p_pop, log_prior_z,
                log_p_wl_fn, mu_nodes, log_w_nodes,
                wl_enabled=wl_enabled,
                dL_grid=dL_grid,
            )
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    if singleton_lensing == SINGLETON_LENSING_MIXTURE:
        y_nodes_s, log_wy_s = make_y_grid(y_nodes_single)

    def _pe_event_fn(_, event_idx):
        s = event_idx * nsamp
        sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
        valid = sl(gw_pe.valid) & (sl(gw_pe.prior_wt) > 0.0)
        catalog_ev = em_catalog_pe._replace(active_counterpart_index=event_idx)
        ldw = _log_sample_weight_if_supported(
            sl(gw_pe.m1det),
            sl(gw_pe.q),
            sl(gw_pe.dL),
            sl(gw_pe.chieff),
            sl(gw_pe.pixels),
            sl(gw_pe.prior_wt),
            catalog_ev,
        )
        ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
        if singleton_lensing != SINGLETON_LENSING_MIXTURE and not unlensed_tau_suppression:
            return None, log_evidence_and_mc_variance(ldw, nsamp)

        # (1 - tau_2(z)) x unlensed/WL branch [+ lensed-single branch, MIXTURE].
        # The (1 - tau_2) factor is evaluated per PE sample at the mu = 1
        # apparent redshift; inside the WL branch the mu-dependence of tau_2
        # is a cross term of order tau x s^2 (< 1e-5 at defaults) — stated
        # approximation, negligible against the tau ~ 1e-3 channel weights.
        # It must match the SAME factor applied to mu_sel^(1) above: the
        # per-event evidence and the selection integral have to describe one
        # intensity (see _unlensed_tau_suppression_enabled).
        dL_ev = sl(gw_pe.dL)
        z_app = z_of_dL_precomputed(jnp.clip(dL_ev, dL_lo, dL_hi), dL_grid)
        tau_app = tau_2_prob(z_app, sis_params)
        ldw_unlensed = jnp.where(
            jnp.isfinite(ldw), ldw + jnp.log1p(-tau_app), -jnp.inf
        )
        log_unlensed, var_unlensed = log_evidence_and_mc_variance(
            ldw_unlensed, nsamp
        )
        if singleton_lensing != SINGLETON_LENSING_MIXTURE:
            # J2 + singleton_lensing=off: the observed singletons are unlensed
            # by protocol (the generator DROPS exactly-one-detected images), so
            # the unlensed branch is the whole singleton evidence.
            return None, (log_unlensed, var_unlensed)
        event_dict = {
            "m1det": sl(gw_pe.m1det), "q": sl(gw_pe.q),
            "dL": dL_ev, "chieff": sl(gw_pe.chieff),
            "prior_wt": sl(gw_pe.prior_wt),
            "valid": sl(gw_pe.valid), "pixels": sl(gw_pe.pixels),
        }
        log_lensed, var_lensed = lensed_single_log_likelihood_event(
            event_dict, cosmo, survey, pop_params, catalog_ev,
            sis_params, fc_pdet_params, log_p_pop, log_prior_z,
            y_nodes_s, log_wy_s,
            pair_orientation_mode=pair_orientation_mode,
            return_mc_variance=True,
        )
        # Both branches are driven by the SAME PE samples, so their errors
        # are correlated: combine with the perfectly-correlated bound.
        var_mix = _correlated_branch_variance(
            log_unlensed, var_unlensed, log_lensed, var_lensed
        )
        return None, (jnp.logaddexp(log_unlensed, log_lensed), var_mix)

    # ----- Singleton PE reduction -------------------------------------
    # Same block-vectorized events axis likelihood/core.py uses: a chunk of
    # ``pe_block`` singletons is evaluated in ONE flattened (pe_block*nsamp,)
    # call and reduced per event by a row-wise log_evidence_and_mc_variance
    # (vmap), removing the sequential per-event kernel-launch overhead of the
    # scan (259 iterations on the singleton campaign).  Masked elements and
    # their per-row order are identical to the scan.  ``pe_event_block == 1``
    # reproduces the historical scan row by row (minus the counterpart
    # _replace); ``None`` (the default) is a single vectorized pass.
    #
    # Two branches keep the exact scan: bright sirens, whose per-event
    # ``active_counterpart_index`` the block path cannot express (same
    # ``has_counterpart`` gate as core.py), and singleton_lensing=MIXTURE,
    # whose lensed branch reduces over BOTH the sample and the y-quadrature
    # axis inside ``lensed_single_log_likelihood_event`` and so is not a
    # row-wise reduction of a flat per-sample vector.
    # Row count from the ARRAY the scan iterated, not the static
    # ``n_singletons`` count: the two are the same in every shipped caller, but
    # a dynamic_slice over a shorter array clamps (and would silently duplicate
    # the last row) where the scan would simply run fewer iterations.
    n_pe_rows = (
        int(jnp.shape(singleton_indices)[0]) if cluster_mode == CLUSTER_MODE_J2
        else nEvents
    )
    use_pe_block = (
        not _has_counterpart_arrays(em_catalog_pe)
        and singleton_lensing != SINGLETON_LENSING_MIXTURE
        and n_pe_rows >= 1
    )
    if use_pe_block:
        pe_block = n_pe_rows if pe_event_block is None else min(
            pe_event_block, n_pe_rows
        )

        def _pe_rows(start, m):
            """(m,) log-evidences and variances for the ``m`` singletons whose
            row index starts at ``start`` (traced or static, events units)."""
            if cluster_mode == CLUSTER_MODE_J2:
                # The singleton rows are an arbitrary subset of event indices,
                # so the chunk is a GATHER, not a contiguous slice.  Per-row
                # sample order is unchanged.
                idx = lax.dynamic_slice_in_dim(singleton_indices, start, m)
                flat = (
                    idx[:, None] * nsamp + jnp.arange(nsamp, dtype=idx.dtype)
                ).reshape(-1)
                take = lambda arr: jnp.take(arr, flat, axis=0)
            else:
                take = lambda arr: lax.dynamic_slice_in_dim(
                    arr, start * nsamp, m * nsamp
                )
            valid = take(gw_pe.valid) & (take(gw_pe.prior_wt) > 0.0)
            dL_ev = take(gw_pe.dL)
            ldw = _log_sample_weight_if_supported(
                take(gw_pe.m1det),
                take(gw_pe.q),
                dL_ev,
                take(gw_pe.chieff),
                take(gw_pe.pixels),
                take(gw_pe.prior_wt),
                em_catalog_pe,
            )
            ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
            if unlensed_tau_suppression:
                # Same per-sample (1 - tau_2) factor as the scan above.
                z_app = z_of_dL_precomputed(jnp.clip(dL_ev, dL_lo, dL_hi), dL_grid)
                tau_app = tau_2_prob(z_app, sis_params)
                ldw = jnp.where(
                    jnp.isfinite(ldw), ldw + jnp.log1p(-tau_app), -jnp.inf
                )
            rows = ldw.reshape(m, nsamp)
            return jax.vmap(
                lambda row: log_evidence_and_mc_variance(row, nsamp)
            )(rows)

        # Static chunk plan: ``n_full`` full chunks (lax.scan when >1, a direct
        # call when exactly 1) plus one Python-level remainder chunk, so every
        # shape stays static with no dynamic padding mask.
        n_full = n_pe_rows // pe_block
        rem = n_pe_rows - n_full * pe_block
        parts = []
        if n_full == 1:
            parts.append(_pe_rows(0, pe_block))
        elif n_full > 1:
            def _chunk_scan(_, chunk_idx):
                return None, _pe_rows(chunk_idx * pe_block, pe_block)

            _, stacked = lax.scan(_chunk_scan, None, jnp.arange(n_full))
            # (n_full, pe_block) -> (n_full*pe_block,) preserving event order.
            parts.append(
                jax.tree_util.tree_map(
                    lambda a: a.reshape((n_full * pe_block,) + a.shape[2:]),
                    stacked,
                )
            )
        if rem > 0:
            parts.append(_pe_rows(n_full * pe_block, rem))
        event_lls = jnp.concatenate([p[0] for p in parts])
        event_vars = jnp.concatenate([p[1] for p in parts])
    elif cluster_mode == CLUSTER_MODE_J2:
        # Sum singletons only
        _, (event_lls, event_vars) = lax.scan(
            _pe_event_fn, None, singleton_indices
        )
    else:
        # Legacy: sum all events
        _, (event_lls, event_vars) = lax.scan(
            _pe_event_fn, None, jnp.arange(nEvents)
        )
    singleton_logL_sum = jnp.sum(event_lls)
    singleton_variance_sum = jnp.sum(event_vars)
    ll += singleton_logL_sum

    # ──────────────────────────────────────────────────────────────────
    # Per-pair log-likelihood loop.  Skipped when cluster_mode == OFF.
    # ──────────────────────────────────────────────────────────────────
    per_pair_logL_values = []
    per_pair_var_values = []
    pair_logL_sum = jnp.asarray(0.0, dtype=jnp.float64)
    pair_variance_sum = jnp.asarray(0.0, dtype=jnp.float64)
    if cluster_mode == CLUSTER_MODE_J2 and n_pairs > 0:
        # y-grid for cluster pair likelihoods
        y_nodes, log_wy = make_y_grid(y_nodes_pair)

        # Pair KDEs are a Python tuple of PairKDE NamedTuples, indexed by
        # global event index. nEvents is static so the loop unrolls at
        # trace time.
        def _extract_event(idx):
            """Slice out one event's PE samples by global index."""
            s = idx * nsamp
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
            return {
                "m1det": sl(gw_pe.m1det), "q": sl(gw_pe.q),
                "dL": sl(gw_pe.dL), "chieff": sl(gw_pe.chieff),
                "prior_wt": sl(gw_pe.prior_wt),
                "valid": sl(gw_pe.valid), "pixels": sl(gw_pe.pixels),
            }

        def _pair_loglike_at_index(k):
            i = pair_indices[k, 0]
            j = pair_indices[k, 1]
            ev_i = _extract_event(i)
            ev_j = _extract_event(j)
            kde_i = _slice_event_kde_inside_jit(pair_kdes, i)
            kde_j = _slice_event_kde_inside_jit(pair_kdes, j)
            if pair_marks in (PAIR_MARKS_TIME, PAIR_MARKS_TIME_DELTA):
                dt_obs_k = pair_time_delta_t_obs[k]
                dt_sig_k = pair_time_sigma[k]
            else:
                dt_obs_k = None
                dt_sig_k = None
            ll_pair, var_pair = cluster_log_likelihood_pair(
                ev_i, ev_j, kde_i, kde_j,
                cosmo, survey, pop_params, em_catalog_pe, sis_params,
                log_p_pop, log_prior_z, y_nodes, log_wy,
                pair_marks=pair_marks,
                delta_t_obs=dt_obs_k, sigma_delta_t=dt_sig_k,
                t_obs_window_sec=pair_time_t_obs_window_sec,
                pair_time_signed=pair_time_signed,
                return_mc_variance=True,
            )
            return (
                jnp.where(jnp.isfinite(ll_pair), ll_pair, -jnp.inf),
                jnp.where(jnp.isfinite(var_pair), var_pair, 0.0),
            )

        if pair_batch_size and pair_batch_size > 0:
            # Chunk candidate pairs into fixed-size batches and scan over the
            # padded pair axis. This keeps the compiled graph size independent
            # of n_pairs while preserving the exact per-pair kernel.
            n_batches = (n_pairs + pair_batch_size - 1) // pair_batch_size
            n_padded = n_batches * pair_batch_size

            def _scan_pair(carry, k):
                active = k < n_pairs
                ll_pair_safe, var_pair_safe = lax.cond(
                    active,
                    _pair_loglike_at_index,
                    lambda _: (
                        jnp.asarray(0.0, dtype=jnp.float64),
                        jnp.asarray(0.0, dtype=jnp.float64),
                    ),
                    k,
                )
                carry_ll, carry_var = carry
                return (
                    (carry_ll + ll_pair_safe, carry_var + var_pair_safe),
                    (jnp.where(active, ll_pair_safe, -jnp.inf), var_pair_safe),
                )

            (pair_logL_sum, pair_variance_sum), (padded_pair_logL, padded_pair_var) = lax.scan(
                _scan_pair,
                (
                    jnp.asarray(0.0, dtype=jnp.float64),
                    jnp.asarray(0.0, dtype=jnp.float64),
                ),
                jnp.arange(n_padded),
            )
            ll = ll + pair_logL_sum
            if return_diagnostics:
                per_pair_logL_values = [padded_pair_logL[k] for k in range(n_pairs)]
                per_pair_var_values = [padded_pair_var[k] for k in range(n_pairs)]
        else:
            # Legacy small-n path: iterate pairs at Python level (n_pairs is static).
            for k in range(n_pairs):
                ll_pair_safe, var_pair_safe = _pair_loglike_at_index(k)
                if return_diagnostics:
                    per_pair_logL_values.append(ll_pair_safe)
                    per_pair_var_values.append(var_pair_safe)
                pair_logL_sum = pair_logL_sum + ll_pair_safe
                pair_variance_sum = pair_variance_sum + var_pair_safe
                ll = ll + ll_pair_safe

    # Combined selection correction, with the per-event/per-pair
    # importance-estimator variances spending part of the total-variance
    # budget (they used to be silently excluded, so the advertised
    # GWTC-4/5-style guard could pass a likelihood dominated by noisy event
    # or pair estimators — review P1-09). The pair term is a LOWER BOUND: it
    # sees each branch's driving PE samples but not the partner event's KDE
    # sampling noise (cluster_log_likelihood_pair, review F-096).
    pe_variance_sum = singleton_variance_sum + pair_variance_sum
    selection_correction_total = combined_selection_log_correction(
        log_mu_1, log_sigma2_1,
        log_mu_2, log_sigma2_2,
        n_singletons_observed=(n_singletons if cluster_mode == CLUSTER_MODE_J2 else nEvents),
        n_clusters_observed=(n_pairs if cluster_mode == CLUSTER_MODE_J2 else 0),
        soft_guard=selection_neff_soft_guard,
        max_likelihood_variance=max_likelihood_variance,
        pe_variance_sum=pe_variance_sum,
    )
    ll = ll + selection_correction_total

    logL_total = jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
    if return_diagnostics:
        per_pair_logL = (
            jnp.stack(per_pair_logL_values)
            if per_pair_logL_values
            else jnp.zeros((0,), dtype=jnp.float64)
        )
        per_pair_var = (
            jnp.stack(per_pair_var_values)
            if per_pair_var_values
            else jnp.zeros((0,), dtype=jnp.float64)
        )
        return {
            "logL_total": logL_total,
            # The per-row terms the sums above are made of, in the order of
            # ``singleton_indices`` / ``pair_indices``.  Every row is a function
            # of its own event(s) and the proposal only -- never of which
            # other rows are in the partition -- which is what lets the lensing
            # CLI evaluate them ONCE per proposal for every event and every
            # candidate edge and assemble each partition of the marginalisation
            # as gathers and sums (cli.inference_lensing._assemble_partition).
            "per_event_logL": event_lls,
            "per_event_var": event_vars,
            "per_pair_var": per_pair_var,
            "log_mu_singleton": log_mu_1,
            "Neff_singleton": Neff_1,
            "log_sigma2_singleton": log_sigma2_1,
            "log_mu_cluster": log_mu_2,
            "Neff_cluster": Neff_2,
            "log_sigma2_cluster": log_sigma2_2,
            "Neff_combined": Neff_combined,
            "selection_correction_total": selection_correction_total,
            "singleton_logL_sum": singleton_logL_sum,
            "singleton_variance_sum": singleton_variance_sum,
            "pair_variance_sum": pair_variance_sum,
            "pe_variance_sum": pe_variance_sum,
            "pair_logL_sum": pair_logL_sum,
            "per_pair_logL": per_pair_logL,
            "n_singletons": jnp.asarray(n_singletons if cluster_mode == CLUSTER_MODE_J2 else nEvents),
            "n_pairs": jnp.asarray(n_pairs if cluster_mode == CLUSTER_MODE_J2 else 0),
            "pair_batch_size": jnp.asarray(pair_batch_size),
            # Singleton rows per vectorized PE chunk; 1 when the exact
            # per-event scan is in force (bright sirens / MIXTURE).
            "pe_event_block": jnp.asarray(pe_block if use_pe_block else 1),
            "y_nodes_pair": jnp.asarray(y_nodes_pair),
            "pair_eval_shape_n": jnp.asarray(nsamp),
            "pair_eval_shape_y": jnp.asarray(y_nodes_pair),
            "wl_selection": jnp.asarray(wl_selection),
            "singleton_lensing": jnp.asarray(singleton_lensing),
            "unlensed_tau_suppression": jnp.asarray(unlensed_tau_suppression),
        }
    return logL_total


def darksiren_likelihood_diagnostics_with_clusters(*args, **kwargs):
    """Evaluate the cluster likelihood once and return component diagnostics.

    This uses a separate static JIT specialization from the scalar sampler path,
    so production likelihood calls keep returning only the scalar log-likelihood.
    """
    kwargs["return_diagnostics"] = True
    return darksiren_log_likelihood_with_clusters(*args, **kwargs)
