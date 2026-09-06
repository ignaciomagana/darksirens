"""Pure JIT body for the hierarchical dark-siren likelihood."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.redshift.prior import (
    DarkSirenPriorState,
    eval_dark_member_completion,
    eval_dark_member_completion_latent,
    eval_dark_obs_bracket,
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.redshift.completion import (
    _member_q_eff_from_logq,
    _resolve_member_logq_row,
    latent_b_gw,
    latent_enabled,
)
from darksirens.redshift.grid import zgrid
from darksirens.gw.populations import pop_model_parser
from darksirens.likelihood.selection import (
    DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    compute_selection_term,
    log_evidence_and_mc_variance,
    selection_log_correction,
    selection_reduce_from_ldw_provider,
)
from darksirens.inference.utils import log_sample_weight, log_target_density_base_and_z
from darksirens.likelihood.events import pad_gw_event_to_multiple
from darksirens.likelihood.wl_weight import (
    log_sample_weight_wl_or_standard,
    log_sample_weight_wl_lognormal_hermite,
)
from darksirens.lensing.grids import (
    make_log_mu_grid,
    make_hermite_u_grid,
    WL_MU_QUADRATURE_NODES,
    WL_MU_QUADRATURE_LOG_MU_RANGE,
)
from darksirens.lensing.wlmagnification import make_tabulated_log_p_wl
from darksirens.sky import sky_model_parser
from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.utils.cosmology import (
    dL_of_z,
    threads_distance_table,
    z_of_dL_precomputed,
    zgrid as _cosmo_zgrid,
)


# Weak-lensing quadrature node counts / ranges. The tabulated-backend mu grid
# lives in lensing.grids so the CLI's startup coverage check validates the SAME
# quadrature this module integrates on.
_WL_NMU_NODES = WL_MU_QUADRATURE_NODES
_WL_LOG_MU_RANGE = WL_MU_QUADRATURE_LOG_MU_RANGE
_WL_HERMITE_NODES = 16

# Backend codes for the static_argnames dispatch (weak-lensing magnification).
WL_BACKEND_DISABLED = -1
WL_BACKEND_LOGNORMAL = 0
WL_BACKEND_TABULATED = 1

# Selection-integral WL treatment (static dispatch), mirroring
# likelihood_with_clusters: STANDARD keeps the legacy un-marginalized
# selection weight; LOGNORMAL applies the same Hermite mu-marginalization to
# injection samples as the PE term.  LOGNORMAL needs the lognormal event-side
# backend: with WL DISABLED there is no magnification scatter to marginalize, so
# STANDARD *is* the marginalized weight and the request falls through; with the
# TABULATED backend there is no matched selection integral in this stack, so the
# fallthrough would silently normalize the hierarchy under a different
# observation model than the numerator and is refused below.
WL_SELECTION_STANDARD = 0
WL_SELECTION_LOGNORMAL = 1


def selection_prior_model(universe_model: str) -> str:
    """Redshift-prior model for the SELECTION integral of ``universe_model``.

    Bright sirens (host pinned by a counterpart) and the catalog-free WL model
    draw their sources from the population volume prior, so their selection
    integral uses ``spectral_sirens``. The dark models MUST use the same
    catalog-completed prior as their PE term — the self-calibrating estimator
    of the methods paper: the sampled survey block {log10n0, delta, b_miss,
    sigma_kde}, Q_LSS, and marked-host eta have to enter mu(Lambda, Theta),
    otherwise the likelihood can reshape p(z | pix, Theta) to fit the detected
    events with no detectability penalty. (Commit e779816 silently hard-wired
    every model to the volume prior — library review P0.2; this restores and
    pins the pre-e779816 behaviour.)
    """
    if universe_model in ("bright_sirens", "spectral_sirens_wl"):
        return "spectral_sirens"
    return universe_model


# EMCatalog leaves consumed by ``prepare_redshift_prior_state`` and its ENTIRE
# call tree, enumerated by reading that tree (do NOT guess-extend this list):
#   completion_curves -> _precompute_grids (apix),
#       _resolve_lss_completion_row_tables (lss_completion_logq/q/_members),
#       _row_C (unique_pixels, dN_obs_kde -- indexed directly by row, no longer
#           via pixel_to_cache_idx; zgals/wgals/ngals on the uncached fallback;
#           pixel_stratum_map under a STRATIFIED aggregate curve stack),
#       _assemble/_completion_curves_row (delta_g_pix_z);
#   catalog_kernel_state / marked_catalog_kernel_state (zgals, dzgals, wgals, ngals);
#   _row_counts (ngals, wgals);
#   the mark parser _gather_marks (mark_logmstar/logssfr/metallicity/color; zgals);
#   field_global_log_Z / _members / _marked -> _field_missing_curve
#       (field_dN_obs_s, field_n_empty, field_N_obs_total,
#       field_lss_q(+_empty_sum)(+_members), field_delta_g, field_mark_z/w/values,
#       field_depth_z/dz/c under a survey depth; pixel_stratum_map,
#       field_occupied_pixels, field_lss_q_empty_sum_strata(+_members) and
#       empty_stratum_counts under a STRATIFIED selection).
# Two later ladder steps added leaves to this same tree and are enumerated here
# for the same reason:
#   the PER-PIXEL selection fraction C_p = f_p C(z) (field-level PR-2):
#       _row_C (f_p_rows), _field_missing_curve (field_f_p_occ,
#       field_f_p_empty_sum).  These three were read by the tree but never
#       enumerated -- the omission dates from PR-2 and made the consumed-vs-
#       compared pin below red before the seam landed.
#   the LATENT Q seam (field-level PR-5, lss_field_mode='latent'):
#       completion_curves -> latent_posterior_mean_q /
#       latent_member_N_miss_integrals / latent_rho_member, and
#       field_global_log_Z(+_members) -> latent_member_logq_rows, which between
#       them read latent_row_fac, latent_phi_z, latent_A, latent_B,
#       latent_b_nodes, latent_P_F, latent_F_F and both row-map/mask pairs
#       (latent_row_map/latent_on_fp for catalog rows,
#       latent_field_row_map/latent_field_on_fp for the field normalizer's
#       occupied rows).  All are None in the shipped table mode, so listing them
#       cannot change any table-mode sharing verdict: two table-mode EMCatalogs
#       compare None-vs-None on each of them, which the ``is`` test passes.
# The state is a PURE function of (model, cosmo, survey, mark params,
# sky-weighting) plus these leaves, so two EMCatalogs sharing the SAME object for
# every one of them yield the identical state.
_PREPARE_STATE_CONSUMED_EMCATALOG_FIELDS = (
    "apix",
    "zgals", "dzgals", "wgals", "ngals",
    "delta_g_pix_z", "dN_obs_kde", "unique_pixels",
    "lss_completion_logq", "lss_completion_q",
    "lss_completion_logq_members", "lss_completion_q_members",
    "mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color",
    "field_dN_obs_s", "field_n_empty", "field_N_obs_total",
    "field_lss_q", "field_lss_q_empty_sum",
    "field_lss_q_members", "field_lss_q_empty_sum_members",
    "field_delta_g",
    "field_mark_z", "field_mark_w", "field_mark_values",
    "field_depth_z", "field_depth_dz", "field_depth_c",
    "pixel_stratum_map", "field_occupied_pixels",
    "empty_stratum_counts", "field_lss_q_empty_sum_strata",
    "field_lss_q_empty_sum_strata_members",
    # per-pixel selection fraction (field-level PR-2)
    "f_p_rows", "field_f_p_occ", "field_f_p_empty_sum", "f_p_total_sum",
    "field_lss_q_fp_empty_sum", "field_lss_q_fp_empty_sum_members",
    # the latent Q seam (field-level PR-5); all None in table mode
    "latent_row_fac", "latent_phi_z",
    "latent_row_map", "latent_on_fp",
    "latent_field_row_map", "latent_field_on_fp",
    "latent_A", "latent_B", "latent_b_nodes",
    "latent_P_F", "latent_F_F",
    # the amp(z) support (field-level PR-8); None on every pre-PR-8 anchor
    "latent_support",
    # the build-time H0 pin: catalog_kernel_state reads pinned_kernels and
    # field_observed_global_total reads field_depth_total_pinned, so two views
    # may share a prepared state only when they share the SAME pin objects
    # (the factory installs one pair on both views of a union bundle).
    "pinned_kernels", "field_depth_total_pinned",
)

# Leaves prepare NEVER reads -- sample_to_unique_idx, the counterpart_* plumbing,
# active_counterpart_index, bright_siren_sky_marginalized, pixel_to_cache_idx
# (dN_obs_kde is indexed directly by row now -- see completion._row_C), and
# lss_completion_indexing (consumed EAGERLY in the factory, before this call) --
# are deliberately EXCLUDED so a PE/selection pair that differs ONLY in those
# (the post-union multitracer bundle, and the flat K=1 union path) can still share.
_PREPARE_STATE_EXCLUDED_EMCATALOG_FIELDS = (
    "sample_to_unique_idx",
    "counterpart_pixel", "counterpart_pixels", "counterpart_zs", "counterpart_dzs",
    "active_counterpart_index", "bright_siren_sky_marginalized",
    "pixel_to_cache_idx", "lss_completion_indexing",
)

# What ``can_share_redshift_prior_state`` actually compares: EVERY EMCatalog leaf
# except the deliberately-excluded ones.  Derived (not hand-listed) so a leaf
# added to EMCatalog and read by a new completion path defaults to NOT sharing
# instead of silently sharing -- the earlier hand-maintained consumed list had
# fallen four leaves behind the stratified-completion tree.  Superset of
# :data:`_PREPARE_STATE_CONSUMED_EMCATALOG_FIELDS`, and equal to it whenever that
# enumeration is complete (pinned by
# tests/test_redshift_prior_state_sharing.py).
_PREPARE_STATE_COMPARED_EMCATALOG_FIELDS = tuple(
    name for name in EMCatalog._fields
    if name not in _PREPARE_STATE_EXCLUDED_EMCATALOG_FIELDS
)

# Prior models whose STATE is EMCatalog-derived and thus dedup-eligible across
# the PE / selection seams.  ``spectral_sirens`` (the WL and bright-siren
# SELECTION model) is excluded so WL configs never share -- its state is a cheap
# cosmo-only vector and its trace must stay untouched; ``bright_sirens`` returns
# ``None`` and its PE model already differs from the spectral selection model.
_SHAREABLE_PRIOR_MODELS = frozenset({"dark_sirens", "dark_sirens_complete"})


class FrozenRedshiftPrior(NamedTuple):
    """Per-sample ``log p(z | pix)`` of every PE sample and every injection,
    evaluated ONCE at build time for a run whose sampled set cannot move it.

    The dark-siren redshift prior is a pure function of (cosmology, survey
    block, catalog) and of the sample redshift ``z = z(dL; cosmology)``.  When
    no cosmology or survey label is sampled -- a population-only run, e.g. the
    ``run_tinyns_heavy_darksirens_likelihood.sh`` launcher's
    ``--fix_cosmology true --fix_survey true`` -- every one of those inputs is a
    run constant, so the per-sample prior is too, and the per-proposal work it
    represents (the kernel state, the completion curves, the windowed catalog
    KDE for ~10^6 samples: ~85% of a CPU call) is spent once instead of once
    per proposal.  Built by ``darksirens.likelihood.factory`` with the SAME
    functions the live graph would run (``prepare_redshift_prior_state`` +
    ``eval_redshift_prior_with_state`` on the concrete arrays), so the values
    are the ones the unfrozen trace computes up to floating-point association.

    SELF-VERIFYING, for the reason the H0 pin is: the decision to freeze is a
    Python test on the run's sampled LABELS, which is not part of any jit cache
    key.  ``probe_ref`` holds the fixed cosmology/survey scalars the prior
    reads (:func:`frozen_prior_probe_vector`); the live graph rebuilds the same
    vector from the proposal it is handed and poisons the log-likelihood to
    ``-inf`` unless the two agree exactly (fixed parameters decode to the
    identical floats every call, so equality is the right test).

    ``log_prior_pe`` / ``log_prior_sel`` are ``(N,)`` for a single catalog and
    ``(N, K)`` for the K-catalog mixture (the mixture weights may still be
    sampled: they are combined LIVE with the frozen per-catalog priors).
    ``log_prior_sel`` is aligned with the injections as the factory hands them
    to the likelihood (pixel-sorted, padded to the selection batch).
    """
    log_prior_pe: Any     # (N_pe,) | (N_pe, K)
    log_prior_sel: Any    # (N_sel,) | (N_sel, K)
    probe_ref: Any        # (P,) fixed cosmology + survey scalars at build time


#: SurveyParams scalars the redshift prior reads; the probe compares them.
#: Structural fields (``z_depth``, ``c_mode``, ``wl_params`` ...) are pytree
#: STRUCTURE and hence part of the jit cache key already.
_FROZEN_SURVEY_PROBE_FIELDS = (
    "n0", "z50", "w", "delta", "b_miss", "alpha_miss", "sigma_kde",
    "m_lim", "M0hat", "sigma_M",
    # Schechter selection family (``c_mode="selection"``): read by the
    # completion curves; ``Mstar_hat`` and ``alpha`` are sampled labels there.
    "Mstar_hat", "alpha", "M_faint_offset",
)


def frozen_prior_probe_vector(cosmo, surveys):
    """The cosmology + per-catalog survey scalars a :class:`FrozenRedshiftPrior`
    depends on, as one ``(P,)`` float64 vector (``None`` fields skipped: they are
    structure, identical between the build and every live call)."""
    vals = [cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa]
    for s in surveys:
        for name in _FROZEN_SURVEY_PROBE_FIELDS:
            v = getattr(s, name, None)
            if v is not None:
                vals.append(v)
    return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in vals])


def can_share_redshift_prior_state(pe_model, sel_model, cat_pe, cat_sel) -> bool:
    """Whether the PE and selection redshift-prior states are provably identical.

    True iff both seams resolve to the SAME dedup-eligible prior model AND every
    EMCatalog field that could feed ``prepare_redshift_prior_state``
    (:data:`_PREPARE_STATE_COMPARED_EMCATALOG_FIELDS`, i.e. all of them bar the
    leaves prepare provably never reads) is the identical object (``is``) in both
    catalogs.  In that case ``prepare_redshift_prior_state`` -- a pure function of
    the model, the (seam-shared) cosmo/survey/mark params and those leaves --
    returns the identical state, so it may be built ONCE and reused for both
    seams.  Object identity (never value equality) keeps this trace-safe and
    cheap and defaults to NOT sharing whenever any compared field differs.
    """
    if pe_model != sel_model or pe_model not in _SHAREABLE_PRIOR_MODELS:
        return False
    return all(
        getattr(cat_pe, name) is getattr(cat_sel, name)
        for name in _PREPARE_STATE_COMPARED_EMCATALOG_FIELDS
    )


def redshift_prior_state_sharing(universe_model, catalogs_pe, catalogs_sel) -> tuple:
    """Per-catalog PE/selection state-sharing verdict for ``universe_model``.

    Called EAGERLY by the likelihood factory on the CONCRETE (pre-jit)
    EMCatalogs -- the ``is`` identity ``can_share_redshift_prior_state`` relies on
    is erased once each leaf becomes a distinct jit tracer, so the verdict is
    resolved here and threaded into ``darksiren_log_likelihood`` as the static
    ``share_prior_state_by_catalog`` tuple.  Mirrors that function's PE/selection
    model resolution (``pe_model`` / ``selection_prior_model``).  Returns a tuple
    of bools aligned with the catalog order (element k True => the PE and
    selection redshift-prior states of catalog k are provably identical and may
    be built once and reused for both seams).
    """
    pe_model = (
        "spectral_sirens" if universe_model == "spectral_sirens_wl" else universe_model
    )
    sel_model = selection_prior_model(universe_model)
    return tuple(
        can_share_redshift_prior_state(pe_model, sel_model, cat_pe, cat_sel)
        for cat_pe, cat_sel in zip(catalogs_pe, catalogs_sel)
    )


def require_view_independent_mu_miss(
    mark_model, mark_names_by_catalog, catalog_sky_weighting,
    catalogs_pe, catalogs_sel,
) -> None:
    """Refuse a marked-host model whose ``mu_miss(z|eta)`` would differ between
    the PE and the selection seam.

    Every other quantity the marked dark-siren prior builds is per-ROW (kernels,
    dN_miss, Z[pix]), so restricting a view to a subset of pixels leaves it
    unchanged.  ``mu_miss(z|eta) = E_obs[h|z]`` is the one AGGREGATE: a single
    ``(N_grid,)`` curve estimated by z-binning ``h`` over the galaxies present in
    the EMCatalog it is handed (``_mu_miss_grid``), so its value depends on WHICH
    pixels the view holds.  When the PE and selection catalogs are compacted over
    different pixel sets the hierarchical likelihood then divides a numerator
    carrying ``mu_miss^PE`` by a beta carrying ``mu_miss^sel``: the population
    prior no longer cancels between the two seams, biasing eta and (through the
    missing-galaxy budget) H0.  This is the conditional-mode twin of the guard
    field mode already carries, which demands the view-independent full-sky flat
    marks for exactly this reason.

    Called EAGERLY on the concrete (pre-jit) EMCatalogs, like
    :func:`redshift_prior_state_sharing`.  Two escapes: the flat FULL-SKY marks
    (``field_mark_z`` / ``field_mark_values``), which
    ``prepare_redshift_prior_state`` uses for ``mu_miss`` in BOTH conventions
    when present, or PE/selection views over the same pixel set (the union views
    every in-repo loader builds).
    """
    import numpy as np

    if mark_model in (None, "none") or catalog_sky_weighting == "field":
        return
    for k, (cat_pe, cat_sel) in enumerate(zip(catalogs_pe, catalogs_sel)):
        names = (
            mark_names_by_catalog[k] if k < len(mark_names_by_catalog) else ()
        )
        if not names:
            continue  # h == 1 for this catalog: the plain galaxy-count model
        if cat_pe.field_mark_values is not None and cat_pe.field_mark_z is not None:
            continue  # mu_miss comes from the view-INDEPENDENT full-sky marks
        pe_pixels, sel_pixels = cat_pe.unique_pixels, cat_sel.unique_pixels
        same_view = (
            (pe_pixels is None and sel_pixels is None)
            or (
                pe_pixels is not None and sel_pixels is not None
                and np.array_equal(np.asarray(pe_pixels), np.asarray(sel_pixels))
            )
        )
        if not same_view:
            raise ValueError(
                f"catalog_sky_weighting='conditional' with mark_model="
                f"{mark_model!r} requires the PE and selection views of catalog "
                f"{k + 1} to cover the SAME pixels, or the flat FULL-SKY mark "
                "inputs (field_mark_z / field_mark_values) built via "
                "darksirens.redshift.completion.build_field_mark_inputs: "
                "mu_miss(z|eta) is a survey-level aggregate over the galaxies in "
                "the view, so two different views give the PE numerator and the "
                "selection beta different missing-galaxy modulations and the "
                "population prior no longer cancels between them."
            )


def _require_field_mode_scope(universe_model, wl_enabled, mark_model, catalogs):
    """Reject FIELD-convention sky weighting outside its supported scope.

    All checks are static (universe/model strings, static bools, and pytree
    STRUCTURE via ``is not None``), so they resolve once per trace -- mirroring
    the K>=2 mixture ``NotImplementedError`` guards.  LSS modulation of the
    missing-galaxy budget (a deterministic Q_LSS table, a Q ensemble, or a real
    per-pixel delta_g) IS supported -- including under ``lss_marginalize`` --
    but only when the survey-global ``field_*`` rows mirror it, so the per-pixel
    numerator and the global normalizer carry the SAME budget; that mirroring is
    what the per-catalog checks below require.  Both ``dark_sirens`` and
    ``dark_sirens_complete`` are supported under the field convention (the
    complete-model field normalizer Z = Sum_pix N_obs is theta-independent).
    """
    if universe_model not in ("dark_sirens", "dark_sirens_complete"):
        raise NotImplementedError(
            "catalog_sky_weighting='field' supports universe_model in "
            "{'dark_sirens', 'dark_sirens_complete'} only; got "
            f"{universe_model!r}."
        )
    if wl_enabled:
        raise NotImplementedError(
            "catalog_sky_weighting='field' is not supported with weak lensing."
        )
    for cat in catalogs:
        cat_has_marks = any(
            getattr(cat, name) is not None
            for name in (
                "mark_logmstar", "mark_logssfr", "mark_metallicity", "mark_color",
            )
        )
        if (mark_model not in (None, "none") and cat_has_marks
                and (cat.field_mark_values is None
                     or cat.field_mark_z is None
                     or cat.field_mark_w is None)):
            raise ValueError(
                "catalog_sky_weighting='field' with a marked-host model "
                "requires the flat FULL-SKY mark inputs on every MARKED catalog "
                "(field_mark_z / field_mark_w / field_mark_values); build them "
                "via build_field_mark_inputs."
            )
        if (
            any(
                getattr(cat, name) is not None
                for name in (
                    "lss_completion_logq_members",
                    "lss_completion_q_members",
                )
            )
            and cat.field_lss_q_members is None
        ):
            raise ValueError(
                "catalog_sky_weighting='field' with a Q_LSS ENSEMBLE requires "
                "the per-member survey-global Q rows (field_lss_q_members / "
                "field_lss_q_empty_sum_members) built from the GLOBAL ensemble."
            )
        if cat.field_dN_obs_s is None:
            raise ValueError(
                "catalog_sky_weighting='field' requires the survey-global field "
                "normalization inputs on every catalog (field_dN_obs_s / "
                "field_n_empty / field_N_obs_total); build them from the FULL-sky "
                "catalog via build_field_normalization_inputs."
            )
        # Budget-modulation consistency: a deterministic Q table or a real
        # per-pixel delta_g must be mirrored by the survey-global field_* rows,
        # or the numerator and the global normalizer would carry DIFFERENT
        # missing-galaxy budgets (measured 33% Z divergence in the adversarial
        # review of the unmodulated normalizer).  Static structure checks;
        # prepare_redshift_prior_state enforces the same rules for direct
        # callers.
        if (
            any(
                getattr(cat, name) is not None
                for name in ("lss_completion_logq", "lss_completion_q")
            )
            and cat.field_lss_q is None
        ):
            raise ValueError(
                "catalog_sky_weighting='field' with a deterministic Q_LSS table "
                "requires the survey-global Q rows (field_lss_q / "
                "field_lss_q_empty_sum) built from the GLOBAL table."
            )
        if (
            cat.delta_g_pix_z is not None
            and cat.delta_g_pix_z.shape[0] != 1
            and cat.field_delta_g is None
        ):
            raise ValueError(
                "catalog_sky_weighting='field' with a per-pixel delta_g "
                "overdensity requires the survey-global delta_g rows "
                "(field_delta_g)."
            )


def member_ess(ll_members: jnp.ndarray) -> jnp.ndarray:
    """Member effective sample size ``exp(-sum_m p_m log p_m)``, PLAN §6.4.

    ``p_m = softmax_m(ll_m)`` are exactly the weights the marginalization
    ``logsumexp_m ll_m - log M`` puts on each ensemble member, so this is the
    perplexity of that mixture: ``M`` when the members are indistinguishable,
    ``1`` when one member owns the estimate.  PLAN §6.5 is the reason it
    ships: ``log Zhat = logsumexp_m ll_m - log M`` is Jensen-biased by
    ``-(e^{sigma^2} - 1)/(2M)`` in the member spread ``sigma``, and ESS is the
    runtime read-out of that spread (``E[ESS]/M ~ exp(-sigma^2)`` for lognormal
    weights).  It is also the tripwire for **K5** (PLAN §9): member ESS below
    ``2`` at the production configuration, with P14 failing, is what would send
    PR-6a to §6.5's fixed-realization fallback.  PR-5b measured the shipped
    configuration and it does NOT: sigma runs 1.15 nats at ``H0 = 20`` through
    1.49e-2 at the anchor to 1.38e-3 at ``H0 = 140``, and P14 came out at
    7.07e-3 nat against the 0.1 nat gate, so PR-6a ships as a marginalization.
    ESS is the quantity that would show that changing.

    Costs nothing: ``ll_members`` is already materialized by both member
    marginalizations, and this is an O(M) reduction over a vector of length
    ``M_draw`` (8 in the shipped configuration).

    NaN-safe by construction, because ``ll_members`` is genuinely allowed to be
    ``-inf``: the per-member selection guard
    (:func:`~darksirens.likelihood.selection.selection_log_correction`, called
    INSIDE the member vmap) returns ``-inf`` for a member whose ``Neff_m``
    fails the Vitale floor or the total-variance criterion.  Members at
    ``-inf`` carry ``p_m = 0`` and contribute nothing (the ``0 log 0 = 0``
    convention); if EVERY member is dead the mixture is ``-inf`` and the ESS is
    reported as ``0.0`` rather than the ``nan`` that ``-inf - (-inf)`` would
    produce.
    """
    lse = logsumexp(ll_members)
    alive = jnp.isfinite(lse)
    # ``jnp.where`` on the SHIFT, not on the result: subtracting a -inf
    # normalizer would make every log-weight nan before the mask could act.
    log_p = ll_members - jnp.where(alive, lse, 0.0)
    p = jnp.where(jnp.isfinite(log_p), jnp.exp(log_p), 0.0)
    entropy = -jnp.sum(jnp.where(p > 0.0, p * log_p, 0.0))
    return jnp.where(alive, jnp.exp(entropy), 0.0)


# Refuse the overlapping tail chunk (see :func:`_pe_chunk_plan`) once it would
# recompute more than ``nEvents / _PE_TAIL_OVERLAP_DENOM`` events.
_PE_TAIL_OVERLAP_DENOM = 8


def _pe_chunk_plan(nEvents: int, pe_block: int) -> tuple[int, int, bool]:
    """Static chunk plan ``(n_full, rem, overlap_tail)`` for the block-vectorized
    per-event PE reduction.

    ``n_full`` full chunks of ``pe_block`` events cover the first
    ``n_full*pe_block`` events, leaving ``rem`` events over.  The historical plan
    evaluated those leftovers in a chunk of a DIFFERENT static event count, and
    because that count is baked into the slice lengths the whole per-sample
    kernel (``log_weight_ev`` -> the per-sample catalog KDE + redshift-prior
    gathers) is traced and lowered a SECOND time at the remainder shape.  It is
    the normal case, not a corner: ``block_sizing._even_split_block`` returns
    ``ceil(nEvents/k)``, so the shipped 259-event plans (blocks 87 and 8) both
    leave a remainder.  MEASURED on CPU at 259 events, block 87: the two-shape
    plan lowers 8611 HLO lines and costs 6.8-7.4 s to compile; the single-shape
    plan below lowers 6797 (-21%) and compiles in 2.8-3.7 s.  It is a
    once-per-process cost -- the shapes are static and steady state is unchanged
    to within the +0.8% of PE work the overlap recomputes.

    ``overlap_tail`` removes the second shape: take the tail chunk at the FULL
    block shape, ending at ``nEvents`` (flat start ``(nEvents - pe_block)*nsamp``,
    always in bounds because ``n_full >= 1``), and keep only its last ``rem``
    rows.  Every chunk then has ONE shape, so they all ride the same
    ``lax.scan`` and the kernel is lowered exactly once.  The retained rows are
    computed from the same masked samples in the same per-row order as before,
    so the per-event values -- and the concatenated ``(nEvents,)`` vector they
    are assembled into -- are unchanged.

    The overlap recomputes ``pe_block - rem`` events (2 of 259 at the shipped
    ``(32768, 87)`` plan, 5 of 259 at the floored one).  That is cheap only
    because the resolver picks a near-even split; a hand-set block close to
    ``nEvents`` (15 of 16 would recompute 14 events, nearly doubling the PE
    work) keeps the two-shape plan instead.
    """
    n_full = nEvents // pe_block
    rem = nEvents - n_full * pe_block
    overlap_tail = (
        rem > 0
        and n_full >= 1
        and _PE_TAIL_OVERLAP_DENOM * (pe_block - rem) <= nEvents
    )
    return n_full, rem, overlap_tail


# ``threads_distance_table`` is ``jax.jit`` plus the contract that the 106.8 MB
# comoving-distance table reaches this module as an ARGUMENT: without it, jax
# lowers the closed-over table to a ``dense<>`` HLO constant and this one jit
# carries ~214 MB of literal per embedding (measured 427.5 MB of module text for
# the production spectral likelihood).  See ``utils.cosmology``.
@threads_distance_table(
    static_argnames=[
        "nEvents",
        "nsamp",
        "pop_model",
        "shared_beta",
        "shared_spin",
        "shared_gamma",
        "universe_model",
        "sel_batch_size",
        "pe_event_block",
        "sky_model",
        "mark_model",
        "mark_names",
        "mark_names_all",
        "wl_backend",
        "wl_selection",
        "lss_marginalize",
        "lss_member_impl",
        "lss_field_mode",
        "lss_member_diagnostics",
        "materialize_redshift_prior_state",
        "selection_neff_soft_guard",
        "n_catalogs",
        "catalog_sky_weighting",
        "share_prior_state_by_catalog",
        "kde_window",
    ],
)
def darksiren_log_likelihood(
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
    pop_model: str,
    universe_model: str,
    shared_beta: bool = True,
    shared_spin: bool = True,
    shared_gamma: bool = True,
    sel_batch_size: int | None = None,
    # Events-axis block size for the per-event PE reduction (static).  None (the
    # default) evaluates ALL events in ONE flattened, vectorized pass -- fastest,
    # and the right choice for spectral sirens.  A finite value processes the
    # events axis in ceil(nEvents / block) chunks, bounding the live per-sample
    # intermediates to O(block x nsamp x N_max_gals) for dense dark-siren catalog
    # KDEs when XLA fails to fuse.  block == 1 reproduces the historical per-event
    # scan.  Inert (falls back to the exact per-event scan) whenever a catalog
    # carries per-event bright-siren counterpart arrays.
    pe_event_block: int | None = None,
    sky_model: str = "isotropic",
    sky_params: jnp.ndarray | None = None,
    mark_model: str = "none",
    mark_params: jnp.ndarray | None = None,
    mark_names: tuple = (),
    # Weak-lensing extension. All inert when wl_backend == WL_BACKEND_DISABLED.
    wl_backend: int = WL_BACKEND_DISABLED,
    wl_a: float = 0.0,
    wl_b: float = 0.0,
    wl_z_grid: jnp.ndarray | None = None,
    wl_log_mu_grid: jnp.ndarray | None = None,
    wl_log_p_table: jnp.ndarray | None = None,
    wl_selection: int = WL_SELECTION_STANDARD,
    lss_marginalize: bool = False,
    # Member-marginalization implementation (static; only consulted when
    # ``lss_marginalize`` is True).  "factored" (default) precomputes the
    # member-INDEPENDENT work (population model, Jacobian, proposal reweighting,
    # sky factor, and the O(N_max_gals) observed-catalog KDE) ONCE and vmaps
    # only the cheap missing-galaxy completion over the ensemble.  "reference"
    # keeps the historical whole-likelihood vmap (each member redoes everything)
    # as a numerically-pinned fallback; the two agree to float re-association.
    lss_member_impl: str = "factored",
    # Where the missing-galaxy modulation Q comes from (static; field-level
    # PR-5).  "table" (default) is the shipped behaviour in every respect: Q is
    # a resident (M, N_rows, N_grid) log-Q table gathered two nodes at a time.
    # "latent" GENERATES Q from the anchor artifact's compact row-factor leaves
    # through ``eval_dark_member_completion_latent`` -- the ONE substitution
    # (PLAN §3.6).  The flag is a routing declaration; the arrays that make it
    # work ride on the EMCatalog (``latent_*``), and the two are cross-checked
    # below so a mode string cannot disagree with the data it names.
    lss_field_mode: str = "table",
    # Runtime member diagnostics (static; PLAN §6.4, shipping with PR-6a).
    # False (default) returns the SCALAR log-likelihood and compiles exactly the
    # module it compiled before this flag existed -- the ESS reduction is behind
    # a Python-level branch, so it is not merely cheap, it is absent.  True
    # returns the diagnostics DICT described in :func:`darksiren_log_likelihood`
    # (``logL_total``, ``ll_members``, ``member_ess``, ``n_members``), following
    # the ``return_diagnostics`` convention of
    # ``likelihood_with_clusters.darksiren_log_likelihood_with_clusters``: a
    # SEPARATE static jit specialization, so the sampler's calls keep returning
    # a scalar and nothing about the production trace changes.  Only meaningful
    # under ``lss_marginalize`` -- there is no member axis otherwise, and asking
    # for one is an error rather than a dict of zeros.
    lss_member_diagnostics: bool = False,
    materialize_redshift_prior_state: bool = True,
    selection_neff_soft_guard: bool = False,
    # Cap on the Monte-Carlo variance of the total log-likelihood estimator
    # (per-event reweighting variances + N_obs^2/Neff_sel); traced on purpose
    # (arithmetic only) so sensitivity scans do not recompile.
    max_likelihood_variance: float = DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    # --- K-catalog mixture (dark_sirens only) -------------------------------
    # ``n_catalogs`` is static (jit specializes on the pytree structure); the
    # mixture operands are TRACED.  All default to the single-catalog values, so
    # K = 1 never allocates a mixture tuple and is bit-identical to the legacy
    # body.  ``mixture_log_weights`` is (K,) and MUST stay traced (it is sampled).
    n_catalogs: int = 1,
    mixture_surveys: tuple = (),
    mixture_em_catalogs_pe: tuple = (),
    mixture_em_catalogs_sel: tuple = (),
    mixture_log_weights: jnp.ndarray | None = None,
    # Per-catalog marked-host operands.  ``mark_names_all`` is a STATIC tuple
    # of per-catalog mark-name tuples; ``mark_params_all`` the matching traced
    # eta vectors.  Empty (default) = derive from the single-catalog
    # ``mark_names`` / ``mark_params`` (catalog 1 marked, catalogs 2..K
    # unmarked) -- the exact legacy semantics.
    mark_params_all: tuple = (),
    mark_names_all: tuple = (),
    # --- catalog sky-weighting convention (dark_sirens only) ----------------
    # "field": survey-global normalizer -> the JOINT catalog host-density
    # estimand (relative angular host density preserved; mixture weight is the
    # host FRACTION).  It is the CLI/user default at every K.  "conditional":
    # per-pixel normalizer Z[pix] -- radial-only legacy estimand, bit-identical
    # to pre-existing behaviour.  The signature default below stays "conditional"
    # as the safe fallback for callers that supply no field-normalization inputs.
    # Static; field is gated to the plain galaxy-count host model.
    catalog_sky_weighting: str = "conditional",
    # Per-catalog redshift-prior-state sharing verdict, computed EAGERLY by the
    # caller via ``redshift_prior_state_sharing`` (object identity is gone once
    # each EMCatalog leaf is a jit tracer, so it must be decided pre-jit).  Empty
    # (default) shares nothing -- every legacy caller keeps its exact trace.
    share_prior_state_by_catalog: tuple = (),
    # Static per-sample catalog-KDE window (Python int or None) for every
    # catalog's kernel state.  None (default) defers to the process-global
    # ``configure_catalog_kde_window`` size -- the legacy behaviour every direct
    # caller keeps; the factories pass a window sized from the bound catalogs
    # (``redshift.catalog.auto_kde_window``) so the evaluator never truncates a
    # sample's in-range galaxy block and one process may hold likelihoods over
    # differently dense catalogs.
    kde_window: int | None = None,
    # Build-time per-sample redshift prior (:class:`FrozenRedshiftPrior`) for a
    # run that samples no cosmology / survey / mark label.  A TRACED pytree
    # operand: its presence (None vs the tuple) is structure, hence part of the
    # jit cache key, and its values ride in as arguments.  None (default) is the
    # pre-existing per-proposal evaluation, op for op.
    frozen_prior: "FrozenRedshiftPrior | None" = None,
    # Build-time empty-catalog-row sample routing, one
    # ``(routing_pe, routing_sel)`` pair per catalog (see
    # ``redshift.prior.EmptyRowRouting``).  A TRACED pytree operand -- its
    # STRUCTURE (which entries are None, and each width tier's static column
    # cap) is part of the jit cache key and its index arrays ride in as
    # arguments.  ``()`` (default) is the pre-existing per-sample evaluation, op
    # for op.  A plan without width tiers only skips the catalog KDE on samples
    # whose pixel row holds no galaxies, and returns the prior vector
    # bit-identical to it, sample by sample and slot by slot.  A plan WITH width
    # tiers additionally reads only ``ngals``-sized column prefixes of the
    # catalog rows: the same galaxies over a shorter reduction, hence ulp-level
    # per sample.  (The TOTAL can move at the last bit either way -- several
    # vmaps in place of one change how XLA fuses the reductions above the
    # evaluator; see ``redshift.prior._eval_dark_routed``.)
    empty_row_routing: tuple = (),
    # Comoving-distance interpolation table (``utils.cosmology.rs``), threaded as
    # a jit ARGUMENT and bound as the active table for this trace.  None resolves
    # to whatever is active in the CALLER's scope, so no call site has to know
    # about it.
    distance_table: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return ``log p({d_i} | cosmo, survey, pop_params)``.

    The angular (sky) model contributes a factor ``g(n̂)`` to the source rate via
    ``log_g_sky``.  It is applied identically to the per-event PE term and the
    selection integral, so an isotropically-drawn injection set reweights ``μ``
    by the same ``g(n̂)`` and the detector's own anisotropy divides out.  When
    ``sky_model == "isotropic"`` the factor is skipped entirely (static branch),
    so the result is bit-for-bit identical to the sky-free likelihood.

    With ``lss_member_diagnostics=True`` (static; requires ``lss_marginalize``)
    the return value is instead the PLAN §6.4 diagnostics dict:

    ``logL_total``   the same scalar this function otherwise returns,
    ``ll_members``   the ``(M,)`` per-member log-likelihoods the marginalization
                     reduces -- already materialized, never rebuilt,
    ``member_ess``   ``exp(-sum_m p_m log p_m)`` with ``p_m = softmax_m(ll_m)``
                     (:func:`member_ess`),
    ``n_members``    ``M_draw``, so a caller can read ``ESS / M`` without
                     tracking the ensemble size separately.
    """
    pop_params_shape = tuple(pop_params.shape)
    if pop_params.ndim == 0 or pop_params_shape[0] == 0:
        raise ValueError(
            "darksiren_log_likelihood received empty pop_params: "
            f"pop_model={pop_model!r}, pop_params.shape={pop_params_shape}. "
            "Verify parameter-space construction for this population model."
        )
    if lss_member_diagnostics and not lss_marginalize:
        # The member ESS is a property of the member MIXTURE; without
        # ``lss_marginalize`` there is no ``ll_members`` vector to take the
        # softmax of (the deterministic path evaluates the posterior-mean Q
        # once).  Refuse rather than return an ESS of 1 that would read as
        # "the ensemble collapsed" instead of "there was no ensemble".
        raise ValueError(
            "lss_member_diagnostics=True requires lss_marginalize=True: the "
            "PLAN §6.4 member ESS is the perplexity of the member mixture "
            "softmax_m(ll_m), and the deterministic (posterior-mean Q) path "
            "never forms an ll_m vector."
        )

    log_p_pop = pop_model_parser(
        pop_model=pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )
    # Angular factor g(n̂).  ``apply_sky`` is a Python bool (static under jit), so
    # the isotropic path adds nothing to the compute graph.
    apply_sky = sky_model != "isotropic"
    log_g_sky = sky_model_parser(sky_model)
    # The sky factor g(n̂, z) may depend on redshift (3-D models); the selection
    # closure derives z from dL with the SAME cosmology as the PE term, so the
    # detector-anisotropy cancellation between the two is preserved.
    # Clamped distance, same rationale as the weight paths below: an
    # out-of-support dL makes z_of_dL return its NaN sentinel, and any
    # z-dependent sky model multiplying that z by sampled parameters
    # NaN-poisons the reverse pass despite the -inf value mask (library
    # review, likelihood finding 4). The invalid samples carry exactly zero
    # weight through the select, so the clamped garbage z is inert.
    def _sky_weight(nx, ny, nz, dL):
        z_c = z_of_dL_precomputed(jnp.clip(dL, _dL_lo, _dL_hi), _dL_grid)
        return log_g_sky(nx, ny, nz, z_c, sky_params)

    sky_log_weight_fn = _sky_weight if apply_sky else None
    # Weak-lensing dispatch (static): wl_enabled gates ALL WL machinery off for
    # every non-WL model, so they remain numerically identical to the non-WL code.
    wl_enabled = wl_backend != WL_BACKEND_DISABLED
    # Selection-side WL marginalization: opt-in, lognormal backend only.  A
    # DISABLED backend keeps the exact legacy selection path (no WL anywhere, so
    # STANDARD is exact); the TABULATED backend has no matched selection integral
    # here, so silently downgrading it to STANDARD would leave mu(Lambda)
    # marginalized under a different observation model than the per-event weights
    # — a static configuration error, not a fallthrough.
    wl_selection_enabled = (
        wl_selection == WL_SELECTION_LOGNORMAL
        and wl_backend == WL_BACKEND_LOGNORMAL
    )
    if wl_selection == WL_SELECTION_LOGNORMAL and wl_backend == WL_BACKEND_TABULATED:
        raise ValueError(
            "wl_selection=WL_SELECTION_LOGNORMAL requires "
            "wl_backend=WL_BACKEND_LOGNORMAL; the tabulated backend has no "
            "matched selection integral, so the Hermite mu-marginalization the "
            "PE term applies cannot be applied to the injections. Use the "
            "lognormal backend, or pass wl_selection=WL_SELECTION_STANDARD for a "
            "deliberately mismatched ablation."
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

    # The WL universe model reuses the spectral-sirens redshift prior for the PE
    # integral; WL and bright-siren selection both use the spectral-sirens
    # (population) redshift distribution for the selection integral (a
    # counterpart pins the bright-siren host, and the WL model is catalog-free,
    # so the population volume prior IS their source distribution). The dark
    # models use the SAME catalog-completed prior for selection as for the PE
    # term -- the self-calibrating estimator the methods paper describes: the
    # sampled survey block {log10n0, delta, b_miss, sigma_kde}, Q_LSS, and
    # marked-host eta must enter mu(Lambda, Theta), or the likelihood can
    # reshape p(z|pix, Theta) to fit the detected events with no detectability
    # penalty. (Restores the pre-e779816 behaviour; that commit silently
    # hard-wired ALL models to the volume prior -- library review P0.2.)
    pe_model = (
        "spectral_sirens" if universe_model == "spectral_sirens_wl" else universe_model
    )
    selection_model = selection_prior_model(universe_model)
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    _dL_grid = dL_of_z(_cosmo_zgrid, H0, Om0, w0, wa)
    _dL_lo = _dL_grid[0]
    _dL_hi = _dL_grid[-1]

    # FIELD-convention sky weighting scope gate (static; covers K = 1 and K >= 2).
    if catalog_sky_weighting == "field":
        _require_field_mode_scope(
            universe_model, wl_enabled, mark_model,
            (em_catalog_pe, em_catalog_sel)
            + tuple(mixture_em_catalogs_pe) + tuple(mixture_em_catalogs_sel),
        )
    elif catalog_sky_weighting != "conditional":
        raise ValueError(
            "catalog_sky_weighting must be 'conditional' or 'field', got "
            f"{catalog_sky_weighting!r}."
        )

    # ------------------------------------------------------------------
    # Unified K >= 1 catalog path.  The single-catalog operands are element 0
    # of length-K tuples; catalogs 2..K arrive via the mixture_* operands.
    # The redshift prior at the two seams becomes
    # log p_mix(z) = logsumexp_k(log w_k + log p_k(z, pix[:, k])), inserted at
    # the prior level so the population term is not recomputed K times.  K = 1
    # takes a STATIC shortcut inside the same closures (no + log w_1, no
    # single-element logsumexp), so it is bit-identical to the historical
    # single-catalog body by construction -- pinned per feature cell by
    # tests/test_unified_k1_golden.py.
    # ------------------------------------------------------------------
    if n_catalogs >= 2:
        if universe_model == "dark_sirens_complete":
            # The complete-model catalog mixture is a COHERENT estimand only under
            # the field (survey-global) normalizer: each catalog's complete prior
            # is then a density over the SAME survey field, so the per-sample
            # logsumexp mixes coherent quantities -- and, because that mixture is
            # all--inf-safe, a populated catalog RESCUES a node where a sparse
            # catalog's pixel is empty (-inf).  Under the conditional (per-pixel)
            # normalizer every catalog's complete prior is separately normalized
            # WITHIN its own pixel, so mixing them blends per-pixel-normalized
            # densities -- an incoherent estimand, hence forbidden.
            if catalog_sky_weighting != "field":
                raise NotImplementedError(
                    "The K-catalog mixture supports "
                    "universe_model='dark_sirens_complete' only under "
                    "catalog_sky_weighting='field'; the conditional (per-pixel) "
                    "normalizer makes the complete-model mixture an incoherent "
                    "estimand (mixing per-pixel-normalized complete priors)."
                )
        elif universe_model != "dark_sirens":
            raise NotImplementedError(
                "The K-catalog mixture likelihood supports "
                "universe_model='dark_sirens' (any sky weighting) or "
                "'dark_sirens_complete' (field sky weighting only); got "
                f"{universe_model!r}."
            )
        if wl_enabled:
            raise NotImplementedError(
                "Weak lensing is not supported with the K-catalog mixture."
            )
        if mark_model not in (None, "none") and not mark_names_all:
            raise NotImplementedError(
                "Marked-host models with the K-catalog mixture require the "
                "per-catalog mark operands (mark_names_all / mark_params_all); "
                "the single-catalog mark_names/mark_params spelling is K=1-only."
            )
        if mixture_log_weights is None:
            raise ValueError(
                "n_catalogs >= 2 requires mixture_log_weights (a (K,) array)."
            )
        if (
            len(mixture_surveys) != n_catalogs - 1
            or len(mixture_em_catalogs_pe) != n_catalogs - 1
            or len(mixture_em_catalogs_sel) != n_catalogs - 1
        ):
            raise ValueError(
                "Mixture operand counts must each be n_catalogs - 1 = "
                f"{n_catalogs - 1}; got surveys={len(mixture_surveys)}, "
                f"pe={len(mixture_em_catalogs_pe)}, "
                f"sel={len(mixture_em_catalogs_sel)}."
            )

    surveys_all = (survey,) + tuple(mixture_surveys)
    catalogs_pe_all = (em_catalog_pe,) + tuple(mixture_em_catalogs_pe)
    catalogs_sel_all = (em_catalog_sel,) + tuple(mixture_em_catalogs_sel)

    # Events-axis block size for the per-event PE reduction (all static Python).
    # None => all events in one vectorized block (fastest); else chunks of
    # ``block`` events, clipped to nEvents.  The per-sample weight kernels are
    # elementwise in the sample axis (the catalog KDE / redshift prior vmap
    # per-sample, the population term and Jacobians are pointwise), so a chunk of
    # ``block`` events is evaluated in ONE flattened (block*nsamp,) call and
    # reduced per event -- identical masked elements in identical order per row.
    if pe_event_block is not None and pe_event_block < 1:
        raise ValueError(
            f"pe_event_block must be a positive integer or None; got {pe_event_block}."
        )
    if nEvents < 1:
        # pe_block below becomes the block-plan divisor; nEvents = 0 reached
        # it as a divide-by-zero. Selection-only inference is not a supported
        # configuration, so fail with the configuration error instead.
        raise ValueError(
            f"darksiren_log_likelihood requires at least one event; got "
            f"nEvents={nEvents}."
        )
    pe_block = nEvents if pe_event_block is None else min(pe_event_block, nEvents)
    # The per-event counterpart selection (bright sirens) sets
    # ``active_counterpart_index`` per event, which the block path cannot express
    # (it left at the inert default).  Presence of ANY per-event counterpart
    # array on ANY PE catalog forces the exact per-event scan, keeping the bright
    # path bit-identical to the historical body.
    has_counterpart = any(
        getattr(c, "counterpart_pixels", None) is not None
        or getattr(c, "counterpart_zs", None) is not None
        or getattr(c, "counterpart_dzs", None) is not None
        for c in catalogs_pe_all
    )

    # Per-catalog mark operands: default to the legacy single-catalog spelling
    # (catalog 1 carries the marks, catalogs 2..K are unmarked).
    if not mark_names_all:
        mark_names_all = (tuple(mark_names),) + ((),) * (n_catalogs - 1)
    mark_names_all = tuple(tuple(names) for names in mark_names_all)
    if len(mark_names_all) != n_catalogs:
        raise ValueError(
            f"mark_names_all must have n_catalogs={n_catalogs} entries; got "
            f"{len(mark_names_all)}."
        )
    if not mark_params_all:
        mark_params_all = (mark_params,) + (None,) * (n_catalogs - 1)
    if len(mark_params_all) != n_catalogs:
        raise ValueError(
            f"mark_params_all must have n_catalogs={n_catalogs} entries; got "
            f"{len(mark_params_all)}."
        )

    def _mark_model_for(k):
        # A catalog with no marks runs the plain galaxy-count host model (h=1).
        return mark_model if mark_names_all[k] else "none"

    def _pix_col(pix, k):
        # GW pixel columns: (N,) for a single catalog (and every direct K=1
        # caller), (N, K) for the stacked mixture.  Static ndim branch, so a
        # 1-D direct caller never pays a reshape.
        return pix[:, k] if pix.ndim == 2 else pix

    def _mixture_logsumexp(lps):
        # Per-sample logsumexp over the K catalogs.  Uses the codebase's
        # all--inf-safe pattern (cf. _logsumexp_neginf_safe): a sample whose
        # z is impossible under EVERY catalog yields exactly -inf with a
        # zero (not NaN) backward pass, so jax.grad stays finite.
        stacked = jnp.stack(lps, axis=0)            # (K, nsamp)
        finite = jnp.isfinite(stacked)
        safe = jnp.where(finite, stacked, -1e30)
        return jnp.where(
            jnp.any(finite, axis=0), logsumexp(safe, axis=0), -jnp.inf
        )

    # Per-proposal, per-catalog prior states: O(N_rows × N_grid) precomputation
    # done ONCE here, then captured by the per-sample closures below.  Because
    # the closures only *read* these arrays, neither the per-event ``lax.scan``
    # nor the selection batching recomputes them (the state arrays are
    # loop-invariant operands of the scans).  For ``bright_sirens`` the state
    # is None and the evaluator uses the live per-event catalog.
    # The marked-host model (eta) reweights the dark-siren catalog prior; pass
    # it to both states so it is applied identically to the PE and selection
    # terms (prepare ignores it for non-dark_sirens models; at K >= 2 the guard
    # above pins mark_model to "none", the legacy mixture semantics).
    # PE / selection differ ONLY through (model, EMCatalog); cosmo, survey, mark
    # params and sky-weighting are identical per catalog.  ``can_share_redshift_
    # prior_state`` proves (EAGERLY, on the concrete EMCatalogs -- object
    # identity is erased once each leaf becomes a distinct jit tracer, so the
    # verdict CANNOT be recomputed here) that the two seams would build the SAME
    # state.  The caller passes that per-catalog verdict as the static tuple
    # ``share_prior_state_by_catalog`` (built via ``redshift_prior_state_sharing``
    # from the pre-jit catalogs); where it is True we build the state ONCE and
    # reuse the SAME object for both seams -- trace-level dedup that removes a
    # genuinely duplicated subgraph with the materialize barrier ON and saves
    # trace time with it OFF.  Default (empty tuple) shares nothing, so every
    # caller that does not opt in (bright_sirens / WL, and any external caller)
    # keeps its exact legacy trace.
    frozen = frozen_prior is not None
    if frozen:
        # The frozen prior replaces BOTH seams' states.  Its premise is the
        # factory's (frozen_prior_admissible); what is checked here is what
        # the graph can check: the paths it cannot serve are refused
        # statically, the array shapes must match the samples, and the value
        # premise is re-verified by the probe at the end of the body.
        if pe_model not in ("dark_sirens", "dark_sirens_complete"):
            raise ValueError(
                "frozen_prior is only defined for the catalog models "
                f"(dark_sirens / dark_sirens_complete); got {pe_model!r}.")
        if wl_enabled or lss_marginalize or has_counterpart:
            raise ValueError(
                "frozen_prior cannot serve weak lensing, an LSS-completion "
                "member marginalization, or per-event counterparts: each "
                "changes the redshift prior per member / per event / per "
                "magnification node.")
        n_pe = int(gw_pe.dL.shape[0])
        want_pe = (n_pe,) if n_catalogs == 1 else (n_pe, n_catalogs)
        if tuple(frozen_prior.log_prior_pe.shape) != want_pe:
            raise ValueError(
                f"frozen_prior.log_prior_pe has shape "
                f"{tuple(frozen_prior.log_prior_pe.shape)}, expected {want_pe} "
                "(one value per PE sample [per catalog]).")
        want_probe = frozen_prior_probe_vector(cosmo, surveys_all).shape
        if tuple(frozen_prior.probe_ref.shape) != tuple(want_probe):
            raise ValueError(
                "frozen_prior.probe_ref does not match this run's cosmology / "
                f"survey structure ({tuple(frozen_prior.probe_ref.shape)} vs "
                f"{tuple(want_probe)}); it was built for a different run.")

    univ_states = []
    sel_states = []
    for k in range(0 if frozen else n_catalogs):
        state_univ = prepare_redshift_prior_state(
            pe_model, cosmo, surveys_all[k], catalogs_pe_all[k],
            mark_model=_mark_model_for(k), mark_params=mark_params_all[k],
            mark_names=mark_names_all[k],
            materialize_state=materialize_redshift_prior_state,
            catalog_sky_weighting=catalog_sky_weighting,
            kde_window=kde_window,
        )
        share_k = (
            k < len(share_prior_state_by_catalog)
            and bool(share_prior_state_by_catalog[k])
        )
        if share_k:
            state_sel = state_univ
        else:
            state_sel = prepare_redshift_prior_state(
                selection_model, cosmo, surveys_all[k], catalogs_sel_all[k],
                mark_model=_mark_model_for(k), mark_params=mark_params_all[k],
                mark_names=mark_names_all[k],
                materialize_state=materialize_redshift_prior_state,
                catalog_sky_weighting=catalog_sky_weighting,
                kde_window=kde_window,
            )
        univ_states.append(state_univ)
        sel_states.append(state_sel)
    prior_states_univ = tuple(univ_states) if not frozen else (None,) * n_catalogs
    prior_states_sel = tuple(sel_states) if not frozen else (None,) * n_catalogs

    def _frozen_mix(lp):
        """Per-sample log p_mix(z) from the frozen per-catalog columns: the
        K = 1 column as is; for K >= 2 the LIVE mixture weights combine the
        frozen per-catalog priors through the same all--inf-safe logsumexp
        as ``_eval_prior_mix``."""
        if n_catalogs == 1:
            return lp
        return _mixture_logsumexp(
            [mixture_log_weights[k] + lp[:, k] for k in range(n_catalogs)]
        )

    # Per-catalog empty-row routing plans, split by side.  Absent (the default)
    # every entry is None and every call below is the historical one.
    _routing_pe = tuple(
        (empty_row_routing[k][0] if k < len(empty_row_routing) else None)
        for k in range(n_catalogs)
    )
    _routing_sel = tuple(
        (empty_row_routing[k][1] if k < len(empty_row_routing) else None)
        for k in range(n_catalogs)
    )

    def _eval_prior_mix(model, states, z, pix, catalogs, routings):
        """log p(z | pix) for ``model`` across the K-catalog mixture.

        K = 1 is a STATIC shortcut -- the bare per-catalog evaluation with no
        mixture weight and no logsumexp -- so the single-catalog compute graph
        is exactly the historical one.  For K >= 2 the per-catalog log priors
        are combined with the sampled log weights via the all--inf-safe
        logsumexp above.

        ``routings`` is this side's length-K tuple of
        :class:`~darksirens.redshift.prior.EmptyRowRouting` plans (or Nones).
        Each catalog carries its OWN plan because each has its own compact pixel
        column and its own ``ngals``; the evaluator returns the prior in the
        caller's sample order either way, so the mixture combine is untouched.
        """
        if n_catalogs == 1:
            return eval_redshift_prior_with_state(
                model, states[0], z, _pix_col(pix, 0), cosmo, surveys_all[0],
                catalogs[0], catalog_sky_weighting=catalog_sky_weighting,
                empty_routing=routings[0],
            )
        lps = [
            mixture_log_weights[k] + eval_redshift_prior_with_state(
                model, states[k], z, _pix_col(pix, k),
                cosmo, surveys_all[k], catalogs[k],
                catalog_sky_weighting=catalog_sky_weighting,
                empty_routing=routings[k],
            )
            for k in range(n_catalogs)
        ]
        return _mixture_logsumexp(lps)

    # WL quadrature plumbing — only allocated when wl_enabled (static).  Lognormal
    # uses Gauss-Hermite in the standardized variable u=(ln mu - m(z))/s(z); the
    # tabulated backend uses Gauss-Legendre in ln mu.  Sentinels otherwise.
    # Q-independent, so computed once and shared across LSS-completion members.
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

    def _ll_given_states(states_univ, states_sel):
        """Full log-likelihood (selection + PE) for ONE tuple-pair of
        per-catalog redshift-prior states.  Factored out so the LSS-completion
        ensemble can be marginalised over (one call per Q_LSS member); for the
        default deterministic path it is called exactly once, so the compute
        graph is unchanged."""
        # No finite guard on the redshift prior. -inf propagates correctly through
        # logsumexp and is caught by the final isfinite check.
        def log_prior_z(z, pix, catalogs):
            return _eval_prior_mix(
                pe_model, states_univ, z, pix, catalogs, _routing_pe
            )

        def log_prior_z_selection(z, pix, catalogs):
            return _eval_prior_mix(
                selection_model, states_sel, z, pix, catalogs, _routing_sel
            )

        def _log_sample_weight_if_supported(m1det, q, dL, chieff, pix, prior_wt, catalogs,
                                            spin=None, log_prior_vals=None):
            """PE per-sample weight; WL-marginalized when wl_enabled, else standard.

            ``catalogs`` is the length-K tuple of per-catalog EMCatalogs; the
            weight kernels pass it (and ``pix``) opaquely through to the
            ``log_prior_z`` closure, which resolves the per-catalog mixture.
            WL is statically K = 1 (guarded above).  ``log_prior_vals`` (the
            frozen path) supplies the per-sample ``log p(z | pix)`` directly, in
            place of the closure; the weight arithmetic is otherwise identical.

            Returns -inf for distances outside the tabulated z(dL) support.  The
            angular factor g(n̂, z) is applied separately in ``_pe_event_fn`` (so WL
            is only ever combined with an isotropic sky in this build — see the WL
            docs).

            The weight arithmetic runs on a CLAMPED distance: out-of-support
            samples hit z_of_dL's NaN sentinel, and although the -inf mask fixes
            the VALUE, every multiplication that stored a NaN operand replays it
            in the backward pass regardless of the zero cotangent (mul's VJP
            scales by the stored operand) — one out-of-grid sample then poisons
            d logL/dH0 for the whole event and NUTS cannot run.  Clamping is
            bit-identical for supported samples; the clamped garbage rows carry
            exactly zero cotangent through the select.
            """
            supported = (dL >= _dL_lo) & (dL <= _dL_hi)
            dL_c = jnp.clip(dL, _dL_lo, _dL_hi)
            prior_fn = (
                log_prior_z if log_prior_vals is None
                else (lambda z, pix_, catalogs_: log_prior_vals)
            )
            if not wl_enabled:
                ldw = log_sample_weight(
                    m1det, q, dL_c, chieff, pix, prior_wt, cosmo, survey, pop_params,
                    catalogs, log_p_pop, prior_fn,
                    spin=spin, dL_grid=_dL_grid,
                )
            elif wl_backend == WL_BACKEND_LOGNORMAL:
                ldw = log_sample_weight_wl_lognormal_hermite(
                    m1det, q, dL_c, chieff, pix, prior_wt,
                    cosmo, survey, pop_params, catalogs,
                    log_p_pop, prior_fn,
                    wl_a, wl_b, u_nodes, log_wH_nodes,
                    spin=spin, dL_grid=_dL_grid,
                )
            else:
                ldw = log_sample_weight_wl_or_standard(
                    m1det, q, dL_c, chieff, pix, prior_wt,
                    cosmo, survey, pop_params, catalogs,
                    log_p_pop, prior_fn,
                    log_p_wl_fn, mu_nodes, log_w_nodes,
                    wl_enabled=wl_enabled,
                    spin=spin, dL_grid=_dL_grid,
                )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

        def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalogs, spin=None,
                       log_prior_vals=None):
            """Selection weight in the canonical ``(m1det, q, dL, chieff)`` variables."""
            def _selection_prior(z, pix, catalogs):
                if log_prior_vals is not None:
                    return log_prior_vals
                return log_prior_z_selection(z, pix, catalogs)

            # Same clamped-distance treatment as the PE weights (see above):
            # z_of_dL's NaN sentinel must never enter the arithmetic.
            supported = (dL >= _dL_lo) & (dL <= _dL_hi)
            dL_c = jnp.clip(dL, _dL_lo, _dL_hi)
            if wl_selection_enabled:
                # Injection distances are apparent distances too: magnification
                # scatter changes detectability, so mu(Lambda) must be
                # marginalized with the SAME WL kernel as the PE term or the
                # selection normalization is inconsistent with the per-event
                # weights (previously only available in the cluster wrapper).
                ldw = log_sample_weight_wl_lognormal_hermite(
                    m1det, q, dL_c, chieff, pix, prior_wt,
                    cosmo, survey, pop_params, catalogs,
                    log_p_pop, _selection_prior,
                    wl_a, wl_b, u_nodes, log_wH_nodes,
                    spin=spin, dL_grid=_dL_grid,
                )
            else:
                ldw = log_sample_weight(
                    m1det, q, dL_c, chieff, pix, prior_wt, cosmo, survey, pop_params,
                    catalogs, log_p_pop, _selection_prior,
                    spin=spin, dL_grid=_dL_grid,
                )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

        def log_weight_ev(m1det, q, dL, chieff, pix, prior_wt, catalogs, spin=None,
                          log_prior_vals=None):
            """PE weight in the same ``(m1det, q, dL)`` variables as selection."""
            return _log_sample_weight_if_supported(
                m1det, q, dL, chieff, pix, prior_wt, catalogs, spin=spin,
                log_prior_vals=log_prior_vals,
            )

        if frozen:
            # Frozen prior: the injections' log p(z | pix) is a per-sample
            # operand, sliced alongside the injection arrays.  The reduction
            # goes through ``selection_reduce_from_ldw_provider`` -- the SAME
            # per-batch (logsumexp, logsumexp(2x)) accumulation, batch combine
            # and _lse_to_log_mu_neff as ``compute_selection_term`` -- with a
            # provider that reproduces ``_batch_lse`` mask for mask: the
            # weight's own supported/finite mask, then the sky factor, then
            # ``valid & prior_wt > 0``.  Padding (when the caller did not pad)
            # is the same ``pad_gw_event_to_multiple``; the padded rows carry
            # ``valid == False`` and never read their (zero) frozen value.
            gw_sel_f = gw_sel
            lp_sel = frozen_prior.log_prior_sel
            if sel_batch_size is not None:
                gw_sel_f, n_pad = pad_gw_event_to_multiple(gw_sel, sel_batch_size)
                if n_pad:
                    lp_sel = jnp.concatenate([
                        lp_sel,
                        jnp.zeros((n_pad,) + tuple(lp_sel.shape[1:]), lp_sel.dtype),
                    ])
            N_sel_f = int(gw_sel_f.dL.shape[0])
            if int(lp_sel.shape[0]) != N_sel_f:
                raise ValueError(
                    f"frozen_prior.log_prior_sel has {int(lp_sel.shape[0])} rows "
                    f"for {N_sel_f} (padded) injections; it must be aligned with "
                    "the injections exactly as they are handed to the likelihood.")

            def _frozen_sel_provider(start, size):
                sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, size)
                dL_b = sl(gw_sel_f.dL)
                pwt_b = sl(gw_sel_f.prior_wt)
                ldw = log_weight(
                    sl(gw_sel_f.m1det), sl(gw_sel_f.q), dL_b, sl(gw_sel_f.chieff),
                    sl(gw_sel_f.pixels), pwt_b, catalogs_sel_all,
                    spin=sl(gw_sel_f.spin) if gw_sel_f.spin is not None else None,
                    log_prior_vals=_frozen_mix(sl(lp_sel)),
                )
                if sky_log_weight_fn is not None:
                    ldw = ldw + sky_log_weight_fn(
                        sl(gw_sel_f.nx), sl(gw_sel_f.ny), sl(gw_sel_f.nz), dL_b)
                valid = sl(gw_sel_f.valid) & (pwt_b > 0.0)
                return jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)

            log_mu, Neff, _log_sigma2 = selection_reduce_from_ldw_provider(
                _frozen_sel_provider, N_sel_f, Ndraw, sel_batch_size
            )
        else:
            log_mu, Neff, _log_sigma2 = compute_selection_term(
                gw_sel,
                catalogs_sel_all,
                log_weight,
                Ndraw,
                nEvents,
                sel_batch_size=sel_batch_size,
                sky_log_weight_fn=sky_log_weight_fn,
            )

        # ----- Per-event PE reduction -------------------------------------
        # Each event contributes log Ẑ_i (importance average over its nsamp PE
        # samples) and the delta-method variance σ²_i.  For catalogs WITHOUT
        # per-event counterpart arrays the events axis is block-vectorized: a
        # chunk of ``pe_block`` events is evaluated in ONE flattened
        # (pe_block*nsamp,) call and reduced per event by a row-wise
        # log_evidence_and_mc_variance (vmap), removing the 259-iteration
        # sequential kernel-launch overhead of the per-event scan.  The masked
        # elements and their per-row order are identical to the scan, so the
        # per-event values are bit-compatible.  ``pe_block == 1`` reproduces the
        # historical scan (minus the counterpart _replace); ``pe_block ==
        # nEvents`` (the None default) is a single vectorized pass.  Bright
        # sirens set active_counterpart_index per event, which the block path
        # cannot express, so ``has_counterpart`` keeps the exact scan verbatim.

        def _pe_chunk_ldw(s, n):
            """Masked per-sample log-weights for the ``n`` contiguous PE samples
            starting at flat index ``s`` (``n = m*nsamp`` for a chunk of ``m``
            events).  No counterpart _replace -- the block path is only taken
            when no PE catalog carries per-event counterpart arrays."""
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, n)
            dL_ev = sl(gw_pe.dL)
            valid = sl(gw_pe.valid) & (sl(gw_pe.prior_wt) > 0.0)
            ldw = log_weight_ev(
                sl(gw_pe.m1det),
                sl(gw_pe.q),
                dL_ev,
                sl(gw_pe.chieff),
                sl(gw_pe.pixels),
                sl(gw_pe.prior_wt),
                catalogs_pe_all,
                spin=sl(gw_pe.spin) if gw_pe.spin is not None else None,
                log_prior_vals=(
                    _frozen_mix(sl(frozen_prior.log_prior_pe)) if frozen else None
                ),
            )
            # Angular/3-D factor log g(n̂, z) per sample (skipped when isotropic).
            # Clamped dL for the same reverse-NaN reason as the weight paths.
            if apply_sky:
                z_ev = z_of_dL_precomputed(
                    jnp.clip(dL_ev, _dL_lo, _dL_hi), _dL_grid)
                ldw = ldw + log_g_sky(
                    sl(gw_pe.nx), sl(gw_pe.ny), sl(gw_pe.nz), z_ev, sky_params
                )
            # NaN-safe reduction downstream: an event whose samples ALL mask to
            # -inf must contribute -inf, not a NaN backward softmax.
            return jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)

        if has_counterpart:
            # Bright sirens: active_counterpart_index is event-dependent, so keep
            # the exact per-event scan (bit-identical to the historical body).
            def _pe_event_fn(_, event_idx):
                s = event_idx * nsamp
                sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
                dL_ev = sl(gw_pe.dL)
                valid = sl(gw_pe.valid) & (sl(gw_pe.prior_wt) > 0.0)
                catalogs_ev = tuple(
                    c._replace(active_counterpart_index=event_idx)
                    for c in catalogs_pe_all
                )
                ldw = log_weight_ev(
                    sl(gw_pe.m1det),
                    sl(gw_pe.q),
                    dL_ev,
                    sl(gw_pe.chieff),
                    sl(gw_pe.pixels),
                    sl(gw_pe.prior_wt),
                    catalogs_ev,
                    spin=sl(gw_pe.spin) if gw_pe.spin is not None else None,
                )
                if apply_sky:
                    z_ev = z_of_dL_precomputed(
                        jnp.clip(dL_ev, _dL_lo, _dL_hi), _dL_grid)
                    ldw = ldw + log_g_sky(
                        sl(gw_pe.nx), sl(gw_pe.ny), sl(gw_pe.nz), z_ev, sky_params
                    )
                ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
                return None, log_evidence_and_mc_variance(ldw, nsamp)

            _, (event_lls, event_vars) = lax.scan(
                _pe_event_fn, None, jnp.arange(nEvents)
            )
        else:
            # Block-vectorized: ceil(nEvents/pe_block) chunks.  A static chunk
            # plan -- ``n_full`` full chunks (via lax.scan when >1, or a direct
            # call when exactly 1) plus, when the block does not divide nEvents,
            # a tail chunk -- keeps every shape static with no dynamic padding
            # mask.  ``_pe_chunk_plan`` decides whether that tail is taken at the
            # FULL block shape (overlapping the last full chunk, so ONE shape is
            # lowered for the whole plan) or at the remainder shape.
            block_samps = pe_block * nsamp
            n_full, rem, overlap_tail = _pe_chunk_plan(nEvents, pe_block)

            def _reduce_events(s, m):
                # s: flat sample start (traced or static); m: STATIC event count.
                ldw = _pe_chunk_ldw(s, m * nsamp).reshape(m, nsamp)
                return jax.vmap(
                    lambda row: log_evidence_and_mc_variance(row, nsamp)
                )(ldw)

            def _chunk_scan(_, s):
                return None, _reduce_events(s, pe_block)

            parts = []
            if overlap_tail:
                # n_full full-chunk starts plus the overlapping tail start, ALL
                # through one scan: the kernel is traced (and lowered) once.
                starts = jnp.asarray(
                    [i * block_samps for i in range(n_full)]
                    + [(nEvents - pe_block) * nsamp]
                )
                _, stacked = lax.scan(_chunk_scan, None, starts)
                # (n_full+1, pe_block) -> the n_full full chunks in event order,
                # then only the tail rows the full chunks did not already cover.
                parts.append(
                    jax.tree_util.tree_map(
                        lambda a: a[:n_full].reshape(
                            (n_full * pe_block,) + a.shape[2:]
                        ),
                        stacked,
                    )
                )
                parts.append(
                    jax.tree_util.tree_map(
                        lambda a: a[n_full, pe_block - rem:], stacked
                    )
                )
            else:
                if n_full == 1:
                    parts.append(_reduce_events(0, pe_block))
                elif n_full > 1:
                    _, stacked = lax.scan(
                        _chunk_scan, None, jnp.arange(n_full) * block_samps
                    )
                    # (n_full, pe_block) -> (n_full*pe_block,) in event order.
                    parts.append(
                        jax.tree_util.tree_map(
                            lambda a: a.reshape(
                                (n_full * pe_block,) + a.shape[2:]
                            ),
                            stacked,
                        )
                    )
                if rem > 0:
                    parts.append(_reduce_events(n_full * block_samps, rem))

            event_lls = jnp.concatenate([p[0] for p in parts])
            event_vars = jnp.concatenate([p[1] for p in parts])
        # Guard on the TOTAL log-likelihood variance: the per-event
        # reweighting variances spend part of the budget, so the correction
        # must be evaluated after the event scan.
        ll = selection_log_correction(
            log_mu,
            Neff,
            nEvents,
            soft_guard=selection_neff_soft_guard,
            max_likelihood_variance=max_likelihood_variance,
            pe_variance_sum=jnp.sum(event_vars),
        )
        return ll + jnp.sum(event_lls)

    def _factored_member_marginalization(n_members, is_field):
        """FACTORED LSS-completion marginalisation (default under lss_marginalize).

        Numerically equal to ``_reference_member_marginalization`` up to
        floating-point re-association, but the member-INDEPENDENT per-sample work
        -- the population model, the z(dL) inversion + Jacobians, the proposal
        reweighting, the sky factor, AND the O(N_max_gals) observed-catalog KDE
        (``eval_dark_obs_bracket``) -- is computed ONCE and only the cheap
        missing-galaxy completion (``eval_dark_member_completion``) + the
        per-event / selection reductions are vmapped over the M ensemble members.
        Everything below carries ONLY member-stacked leaves into the vmap.
        """
        # lss_marginalize already implies the dark-siren PE + selection prior and
        # no weak lensing.  Assert statically (unreachable otherwise) so the
        # factored seam's assumptions are pinned rather than silently violated.
        if not (pe_model == "dark_sirens" and selection_model == "dark_sirens"):
            raise ValueError(
                "lss_member_impl='factored' requires pe_model == selection_model "
                f"== 'dark_sirens' (got {pe_model!r} / {selection_model!r}); "
                "lss_marginalize is only supported for universe_model='dark_sirens'."
            )
        if wl_enabled:
            raise ValueError(
                "lss_member_impl='factored' is not compatible with weak lensing."
            )

        # --- Q source (static).  The mode string and the data must agree: a
        # run that says "latent" without the artifact leaves would silently
        # fall back to the table path (and find no table), and one that carries
        # the leaves but says "table" would ignore them.  Both are refused. ---
        if lss_field_mode not in ("table", "latent"):
            raise ValueError(
                f"lss_field_mode must be 'table' or 'latent', got {lss_field_mode!r}."
            )
        latent_mode = lss_field_mode == "latent"
        have_latent = [latent_enabled(c) for c in catalogs_pe_all + catalogs_sel_all]
        if latent_mode and not all(have_latent):
            raise ValueError(
                "lss_field_mode='latent' requires the latent-field leaves on "
                "EVERY PE and selection catalog (EMCatalog.latent_row_fac and "
                "friends, installed from the anchor artifact by the likelihood "
                "factory); some catalogs carry none. The seam cannot generate Q "
                "for a catalog it has no row map for."
            )
        if not latent_mode and any(have_latent):
            raise ValueError(
                "a catalog carries latent-field leaves but lss_field_mode is "
                "'table'; the leaves would be ignored and the run would consume "
                "a Q table the latent artifact does not describe. Pass "
                "--lss_field_mode latent or drop the artifact."
            )

        def _latent_row_gather(pixk, catalogs):
            """Per-sample footprint row index + mask, MEMBER-INDEPENDENT.

            Empty tuples in table mode, so the precompute pytree is
            structurally identical to the shipped one there.
            """
            if not latent_mode:
                return (), ()
            fit = tuple(jnp.asarray(catalogs[k].latent_row_map)[pixk[k]]
                        for k in range(n_catalogs))
            fp = tuple(jnp.asarray(catalogs[k].latent_on_fp)[pixk[k]]
                       for k in range(n_catalogs))
            return fit, fp

        dL_lo, dL_hi = _dL_lo, _dL_hi

        # Member-stacked leaves: per catalog (row-aligned RAW member log-Q table,
        # log_Z leaf).  These are the ONLY leaves carried into the member vmap;
        # the member density is reconstructed on the fly from the SHARED (member-
        # INDEPENDENT) ``base_miss`` curve so NO (M, N_rows, N_grid) cube is ever
        # a state leaf or a vmap operand -- only the resident data-constant log-Q
        # table (gathered two nodes at a time) and the compact log_Z leaf.  The
        # conditional normalizer is the (M, N_rows) per-pixel rows; the field
        # normalizer is the (M,) survey-global scalars.  ``base_miss`` /
        # ``member_is_log`` / ``z_depth`` are member-INDEPENDENT and captured as
        # per-side closures (below), never batched.
        # LATENT (field-level PR-5): the member-stacked leaf is the compact
        # (M, n_fit + 1, M_z) row factor and the (M, N_grid) budget normalizer
        # -- 11.7 MB + 0.07 MB at production rank -- in place of the
        # (M, N_rows, N_grid) log-Q table, which at DESI scale is 1.06 GB and,
        # being theta-dependent through rho, could not be a data constant at
        # any size.  ``rho`` is the ONLY per-proposal latent work: O(n_b +
        # N_grid) per member (PLAN §2.3).
        def _member_leaf_bundle(states, catalogs):
            out = []
            for s, c in zip(states, catalogs):
                logZ = s.log_Z_global_members if is_field else s.log_Z_members
                if latent_mode:
                    # ``rho`` comes from the STATE, not from a fresh grid
                    # build: it must be the normalizer of the same
                    # completeness curve ``base_miss`` carries, and it is
                    # already computed once per proposal in
                    # ``completion_curves`` (PLAN §4.2, one formation site).
                    out.append((jnp.asarray(c.latent_row_fac),
                                s.latent_rho, logZ))
                else:
                    logq_all, _ = _resolve_member_logq_row(c)  # (M, N_rows, N_grid) view
                    out.append((logq_all, logZ))
            return tuple(out)

        # ``base_miss`` is present only on ENSEMBLE states, which the guard in the
        # ``lss_marginalize`` block above requires on BOTH seams.
        base_miss_univ = tuple(s.base_miss for s in prior_states_univ)
        base_miss_sel = tuple(s.base_miss for s in prior_states_sel)
        member_is_log_univ = tuple(
            True if latent_mode else _resolve_member_logq_row(c)[1]
            for c in catalogs_pe_all
        )
        member_is_log_sel = tuple(
            True if latent_mode else _resolve_member_logq_row(c)[1]
            for c in catalogs_sel_all
        )
        z_depth_all = tuple(surveys_all[k].z_depth for k in range(n_catalogs))
        # PR-8 (the amp(z) support).  The Q-side depth relaxation is dropped on
        # a catalog whose anchor models the field ABOVE the fitted depth: there
        # the seam has an assumed value to say and ``Q := 1`` would delete it.
        # Everywhere else this tuple IS ``z_depth_all`` -- ``latent_support`` is
        # None on every table-mode and every pre-PR-8 latent catalog, and that
        # is a static pytree-STRUCTURE test, so nothing is decided at trace
        # time.  Only the LATENT branch reads it; the table evaluator keeps
        # ``z_depth_all``, because a resident Q table has no support of its own
        # to speak for and the depth relaxation is the only thing that stops it
        # being extrapolated.
        z_depth_q_all = tuple(
            None if (latent_mode
                     and catalogs_pe_all[k].latent_support is not None)
            else z_depth_all[k]
            for k in range(n_catalogs))
        # Member-INDEPENDENT latent constant, hoisted out of the member vmap.
        # Per SIDE (like ``base_miss`` / ``member_is_log``) rather than shared,
        # so the two seams cannot silently drift onto one catalog's grid.
        latent_phi_z_univ = (tuple(jnp.asarray(c.latent_phi_z)
                                   for c in catalogs_pe_all)
                             if latent_mode else ())
        latent_phi_z_sel = (tuple(jnp.asarray(c.latent_phi_z)
                                  for c in catalogs_sel_all)
                            if latent_mode else ())

        # Per-catalog member completion -> per-sample log p_mix(z), reusing the
        # SAME K=1 static shortcut / K>=2 _mixture_logsumexp seam as
        # _eval_prior_mix, but on the precomputed observed brackets.  ``base_tup``
        # / ``is_log_tup`` are the per-SIDE (PE vs selection) member-independent
        # closures; ``leaves[k] = (member_logq_m, logZ_m)`` is the per-catalog
        # member-axis leaf peeled by the outer vmap.
        def _log_p_mix(A_obs, idx, t, pixk, fitk, fpk, leaves, base_tup,
                       is_log_tup, phi_z_tup):
            def _one(k):
                if latent_mode:
                    # THE seam.  Identical to the table branch below in every
                    # respect except where logQ comes from (PLAN §3.6).
                    row_fac_m, rho_m, logZ_m = leaves[k]
                    return eval_dark_member_completion_latent(
                        A_obs[k], idx[k], t[k], pixk[k], fitk[k], fpk[k],
                        base_tup[k], row_fac_m, phi_z_tup[k], rho_m,
                        latent_b_gw(surveys_all[k]), logZ_m,
                        z_depth_q_all[k], is_field,
                    )
                member_logq_m, logZ_m = leaves[k]
                return eval_dark_member_completion(
                    A_obs[k], idx[k], t[k], pixk[k], base_tup[k],
                    member_logq_m, is_log_tup[k], logZ_m, z_depth_all[k], is_field,
                )
            if n_catalogs == 1:
                return _one(0)
            lps = [mixture_log_weights[k] + _one(k) for k in range(n_catalogs)]
            return _mixture_logsumexp(lps)

        # --- PE precompute (member-independent).  Block-vectorized over the
        # events axis with the SAME static chunk plan as #252's deterministic
        # per-event reduction in _ll_given_states above (mirrored here rather than
        # shared so the deterministic-path code stays textually untouched): a
        # chunk of ``m`` events is evaluated in ONE flattened (m*nsamp,) pass of
        # the member-INDEPENDENT kernels and reshaped to (m, nsamp, ...).  Those
        # kernels -- log_target_density_base_and_z, eval_dark_obs_bracket's
        # per-sample vmap, and log_g_sky -- are elementwise in the sample axis, so
        # a chunk is bit-identical to per-event evaluation (just reshaped), the
        # same justification as the deterministic path.  ``pe_event_block`` (via
        # pe_block) chooses the chunk size: None (default) => ALL events in one
        # pass (no scan, fastest); a finite block bounds the KDE peak memory to
        # O(block x nsamp x N_max_gals); block == 1 reproduces the historical
        # per-event scan.  No counterpart concern here (lss_marginalize implies
        # dark_sirens, statically asserted above), so no fallback branch. ---
        def _pe_precompute_flat(s, n):
            # Member-independent per-sample precompute over the ``n`` contiguous
            # PE samples from flat index ``s`` (``n = m*nsamp`` for m events).
            # Body identical to the historical per-event scan body; only the slice
            # LENGTH differs (m*nsamp vs nsamp).
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, n)
            dL_ev = sl(gw_pe.dL)
            supported = (dL_ev >= dL_lo) & (dL_ev <= dL_hi)
            dL_c = jnp.clip(dL_ev, dL_lo, dL_hi)
            valid = sl(gw_pe.valid) & (sl(gw_pe.prior_wt) > 0.0)
            base, z_c = log_target_density_base_and_z(
                sl(gw_pe.m1det), sl(gw_pe.q), dL_c, sl(gw_pe.chieff),
                sl(gw_pe.pixels), sl(gw_pe.prior_wt),
                cosmo, survey, pop_params, catalogs_pe_all[0], log_p_pop,
                spin=sl(gw_pe.spin) if gw_pe.spin is not None else None,
                dL_grid=_dL_grid,
            )
            pix_all = sl(gw_pe.pixels)
            obs = tuple(
                eval_dark_obs_bracket(
                    z_c, _pix_col(pix_all, k), prior_states_univ[k], catalogs_pe_all[k]
                )
                for k in range(n_catalogs)
            )
            A_obs = tuple(o[0] for o in obs)
            idx = tuple(o[1] for o in obs)
            t = tuple(o[2] for o in obs)
            pixk = tuple(_pix_col(pix_all, k) for k in range(n_catalogs))
            # LATENT: the pixel -> footprint-row map and the footprint mask are
            # MEMBER-INDEPENDENT, so they belong here (once) rather than inside
            # the member vmap (M times).  Empty tuples in table mode keep the
            # precompute pytree structurally identical to the shipped one.
            fitk, fpk = _latent_row_gather(pixk, catalogs_pe_all)
            # Sky factor uses the SAME clamped-dL redshift as base (z_c); it is
            # inserted AFTER the first mask in _pe_member_terms, exactly as before.
            sky = (
                log_g_sky(sl(gw_pe.nx), sl(gw_pe.ny), sl(gw_pe.nz), z_c, sky_params)
                if apply_sky else jnp.zeros_like(base)
            )
            return (base, supported, valid, sky, A_obs, idx, t, pixk, fitk, fpk)

        def _pe_precompute_chunk(s, m):
            # Evaluate m events (m*nsamp samples) in one flat pass, then split the
            # leading axis back to (m, nsamp, ...) so the stacked pytree keeps the
            # historical leaf shapes (nEvents, nsamp, ...) that _pe_member_terms
            # consumes UNCHANGED.
            flat = _pe_precompute_flat(s, m * nsamp)
            return jax.tree_util.tree_map(
                lambda a: a.reshape((m, nsamp) + a.shape[1:]), flat
            )

        # Same chunk plan as the shipped path above, including the overlapping
        # tail that keeps a single lowered shape (see :func:`_pe_chunk_plan`).
        block_samps = pe_block * nsamp
        n_full, rem, overlap_tail = _pe_chunk_plan(nEvents, pe_block)

        def _chunk_scan(_, s):
            return None, _pe_precompute_chunk(s, pe_block)

        parts = []
        if overlap_tail:
            starts = jnp.asarray(
                [i * block_samps for i in range(n_full)]
                + [(nEvents - pe_block) * nsamp]
            )
            _, stacked = lax.scan(_chunk_scan, None, starts)
            # (n_full+1, pe_block, nsamp, ...): the full chunks in event order,
            # then only the tail rows the full chunks did not already cover.
            parts.append(
                jax.tree_util.tree_map(
                    lambda a: a[:n_full].reshape((n_full * pe_block,) + a.shape[2:]),
                    stacked,
                )
            )
            parts.append(
                jax.tree_util.tree_map(lambda a: a[n_full, pe_block - rem:], stacked)
            )
        else:
            if n_full == 1:
                parts.append(_pe_precompute_chunk(0, pe_block))
            elif n_full > 1:
                _, stacked = lax.scan(
                    _chunk_scan, None, jnp.arange(n_full) * block_samps
                )
                # (n_full, pe_block, nsamp, ...) -> (n_full*pe_block, nsamp, ...).
                parts.append(
                    jax.tree_util.tree_map(
                        lambda a: a.reshape((n_full * pe_block,) + a.shape[2:]),
                        stacked,
                    )
                )
            if rem > 0:
                parts.append(_pe_precompute_chunk(n_full * block_samps, rem))
        # Concatenate full-chunk + remainder pytrees along the events axis,
        # preserving global event order (== the historical arange(nEvents) scan
        # order).  A single part (the None default and any exact-division block)
        # needs no concatenate.
        pe_pre = (
            parts[0]
            if len(parts) == 1
            else jax.tree_util.tree_map(
                lambda *ps: jnp.concatenate(ps, axis=0), *parts
            )
        )

        def _pe_member_terms(leaves):
            base, supported, valid, sky, A_obs, idx, t, pixk, fitk, fpk = pe_pre
            log_p_mix = _log_p_mix(
                A_obs, idx, t, pixk, fitk, fpk, leaves, base_miss_univ,
                member_is_log_univ, latent_phi_z_univ,
            )   # (nEvents, nsamp)
            # Reproduce _pe_event_fn's mask ORDER with base playing ldw's role:
            #   ldw -> mask(supported & isfinite) -> +sky -> mask(valid & isfinite).
            ldw = base + log_p_mix
            ldw = jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
            if apply_sky:
                ldw = ldw + sky
            ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
            return jax.vmap(lambda row: log_evidence_and_mc_variance(row, nsamp))(ldw)

        # --- Selection: the member reduction. Pad the injections (batched)
        # exactly as compute_selection_term, then precompute the member-
        # INDEPENDENT per-injection quantities ONCE (a batched scan bounds the KDE
        # peak memory to O(sel_batch_size x N_max_gals)); the per-member reduction
        # re-slices these cheaply. ---
        if sel_batch_size is not None:
            gw_sel_p, _ = pad_gw_event_to_multiple(gw_sel, sel_batch_size)
        else:
            gw_sel_p = gw_sel
        N_sel_p = gw_sel_p.dL.shape[0]

        def _sel_precompute(start, size):
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, size)
            dL_b = sl(gw_sel_p.dL)
            supported = (dL_b >= dL_lo) & (dL_b <= dL_hi)
            dL_c = jnp.clip(dL_b, dL_lo, dL_hi)
            valid = sl(gw_sel_p.valid) & (sl(gw_sel_p.prior_wt) > 0.0)
            base, z_c = log_target_density_base_and_z(
                sl(gw_sel_p.m1det), sl(gw_sel_p.q), dL_c, sl(gw_sel_p.chieff),
                sl(gw_sel_p.pixels), sl(gw_sel_p.prior_wt),
                cosmo, survey, pop_params, catalogs_sel_all[0], log_p_pop,
                spin=sl(gw_sel_p.spin) if gw_sel_p.spin is not None else None,
                dL_grid=_dL_grid,
            )
            pix_all = sl(gw_sel_p.pixels)
            obs = tuple(
                eval_dark_obs_bracket(
                    z_c, _pix_col(pix_all, k), prior_states_sel[k],
                    catalogs_sel_all[k],
                )
                for k in range(n_catalogs)
            )
            A_obs = tuple(o[0] for o in obs)
            idx = tuple(o[1] for o in obs)
            t = tuple(o[2] for o in obs)
            pixk = tuple(_pix_col(pix_all, k) for k in range(n_catalogs))
            fitk, fpk = _latent_row_gather(pixk, catalogs_sel_all)
            # _sky_weight clamps dL internally, matching _batch_lse (which
            # passes the UNCLAMPED dL_b to sky_log_weight_fn).
            sky = (
                sky_log_weight_fn(
                    sl(gw_sel_p.nx), sl(gw_sel_p.ny), sl(gw_sel_p.nz), dL_b
                )
                if apply_sky else jnp.zeros_like(base)
            )
            return base, supported, valid, sky, A_obs, idx, t, pixk, fitk, fpk

        if sel_batch_size is None:
            sel_pre = _sel_precompute(0, N_sel_p)
        else:
            N_batches = N_sel_p // sel_batch_size

            def _pre_scan(_, b):
                return None, _sel_precompute(b * sel_batch_size, sel_batch_size)

            _, sel_pre_stacked = lax.scan(_pre_scan, None, jnp.arange(N_batches))
            # (N_batches, batch, ...) -> (N_sel_p, ...) preserving order.
            sel_pre = jax.tree_util.tree_map(
                lambda a: a.reshape((N_sel_p,) + a.shape[2:]), sel_pre_stacked
            )

        def _sel_member_reduction(leaves):
            (base_a, sup_a, val_a, sky_a, A_obs_a, idx_a, t_a, pixk_a,
             fitk_a, fpk_a) = sel_pre

            def _provider(start, size):
                sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, size)
                log_p_mix = _log_p_mix(
                    tuple(sl(a) for a in A_obs_a),
                    tuple(sl(a) for a in idx_a),
                    tuple(sl(a) for a in t_a),
                    tuple(sl(a) for a in pixk_a),
                    tuple(sl(a) for a in fitk_a),
                    tuple(sl(a) for a in fpk_a),
                    leaves, base_miss_sel, member_is_log_sel,
                    latent_phi_z_sel,
                )
                # Reproduce log_weight's mask, then _batch_lse's +sky and mask.
                ldw = sl(base_a) + log_p_mix
                ldw = jnp.where(sl(sup_a) & jnp.isfinite(ldw), ldw, -jnp.inf)
                if apply_sky:
                    ldw = ldw + sl(sky_a)
                ldw = jnp.where(sl(val_a) & jnp.isfinite(ldw), ldw, -jnp.inf)
                return ldw

            return selection_reduce_from_ldw_provider(
                _provider, N_sel_p, Ndraw, sel_batch_size
            )

        # --- THE member vmap: carries ONLY member-stacked leaves. ---
        def _member_ll(univ_leaves, sel_leaves):
            event_lls, event_vars = _pe_member_terms(univ_leaves)
            log_mu_m, Neff_m, _ = _sel_member_reduction(sel_leaves)
            ll_m = selection_log_correction(
                log_mu_m, Neff_m, nEvents,
                soft_guard=selection_neff_soft_guard,
                max_likelihood_variance=max_likelihood_variance,
                pe_variance_sum=jnp.sum(event_vars),
            )
            return ll_m + jnp.sum(event_lls)

        univ_stack = _member_leaf_bundle(prior_states_univ, catalogs_pe_all)
        sel_stack = _member_leaf_bundle(prior_states_sel, catalogs_sel_all)
        ll_members = jax.vmap(_member_ll, in_axes=(0, 0))(univ_stack, sel_stack)

        ll_members = jnp.where(jnp.isfinite(ll_members), ll_members, -jnp.inf)
        # Both member impls return the (M,) vector alongside the reduction, so
        # the PLAN §6.4 ESS diagnostic reads the ALREADY-MATERIALIZED weights
        # rather than a second evaluation.  The scalar caller drops it and XLA
        # never sees it; nothing about the shipped trace changes.
        return logsumexp(ll_members) - jnp.log(n_members), ll_members

    def _reference_member_marginalization(n_members, is_field):
        """REFERENCE LSS-completion marginalisation: the historical
        whole-likelihood vmap (each member redoes the entire ``_ll_given_states``,
        including the observed-catalog KDE).  Retained verbatim as the numerical
        pin for the factored path; selected via ``lss_member_impl='reference'``."""
        # Per-member states reuse the (Q-independent) kernels + log_Nobs and swap
        # in the member missing-galaxy density / normalisation; ONE vmap over the
        # SHARED member axis of every catalog's state (the matched-realizations
        # assumption: member m of every catalog samples the same LSS draw).
        # Under the field convention each member also carries its OWN
        # survey-global normalizer; under conditional the broadcast scalar keeps
        # the vmap in_axes pytree (member_axes) leaf-aligned.  The factored path
        # never builds the (M, N_rows, N_grid) member cube; the reference path
        # MATERIALISES it here (base_miss * Q_eff_members) DELIBERATELY -- this is
        # the memory-heavy whole-likelihood pin, opt-in via lss_member_impl, so it
        # is allowed to reconstruct the dense cube the factored path avoids.  The
        # reconstruction is node-for-node identical to the factored bracket
        # evaluation, so value + grad match to floating-point re-association.
        def _member_cube(state, survey_k, cat):
            logq_all, is_log = _resolve_member_logq_row(cat)  # (M, N_rows, N_grid)
            depth_mask = (
                None if survey_k.z_depth is None else (zgrid <= survey_k.z_depth)
            )
            q_eff = _member_q_eff_from_logq(logq_all, depth_mask, is_log)
            return state.base_miss[None, :, :] * q_eff       # (M, N_rows, N_grid)

        def _member_states(state, survey_k, cat):
            return DarkSirenPriorState(
                kernels=state.kernels, log_Nobs=state.log_Nobs,
                dN_miss=_member_cube(state, survey_k, cat), log_Z=state.log_Z_members,
                log_Z_global=(
                    state.log_Z_global_members if is_field else jnp.asarray(0.0)
                ),
            )

        univ_members = tuple(
            _member_states(s, surveys_all[k], catalogs_pe_all[k])
            for k, s in enumerate(prior_states_univ)
        )
        sel_members = tuple(
            _member_states(s, surveys_all[k], catalogs_sel_all[k])
            for k, s in enumerate(prior_states_sel)
        )
        member_axes_one = DarkSirenPriorState(
            kernels=None, log_Nobs=None, dN_miss=0, log_Z=0,
            log_Z_global=(0 if is_field else None),
        )
        member_axes = (member_axes_one,) * n_catalogs
        ll_members = jax.vmap(
            _ll_given_states, in_axes=(member_axes, member_axes),
        )(
            univ_members, sel_members
        )
        ll_members = jnp.where(jnp.isfinite(ll_members), ll_members, -jnp.inf)
        return logsumexp(ll_members) - jnp.log(n_members), ll_members

    # LSS-completion marginalisation: logL = logsumexp_m logL(Λ; Q_m) − log M,
    # treating the M lognormal-completion members as Monte-Carlo draws of the
    # missing-galaxy field.  Opt-in (static); off by default so the deterministic
    # (posterior-mean Q) path is bit-for-bit unchanged.  Both member impls share
    # the structural checks below; ``lss_member_impl`` (static) selects between
    # the FACTORED default and the REFERENCE whole-likelihood vmap.
    if lss_marginalize:
        missing = [
            k for k, s in enumerate(prior_states_univ)
            if getattr(s, "base_miss", None) is None
        ]
        if missing:
            raise ValueError(
                "lss_marginalize=True requires an LSS-completion ENSEMBLE on "
                f"EVERY PE catalog; catalog(s) {[k + 1 for k in missing]} have "
                "none. Build Q_LSS with members "
                "(darksirens_build_lognormal_completion --n-members M > 0) and pass it "
                "via --lss_completion; only universe_model='dark_sirens' supports it."
            )
        member_counts = {
            int(s.log_Z_members.shape[0]) for s in prior_states_univ
        }
        if len(member_counts) != 1:
            raise ValueError(
                "lss_marginalize with a K-catalog mixture marginalizes over a "
                "SHARED member index (the catalogs sample the same LSS "
                "realization), so every catalog's Q ensemble must have the "
                f"same M; got per-catalog member counts {sorted(member_counts)}."
            )
        n_members = member_counts.pop()
        is_field = catalog_sky_weighting == "field"
        sel_members_flags = [
            getattr(s, "base_miss", None) is not None
            for s in prior_states_sel
        ]
        # The member average is a marginalization ONLY if numerator and
        # denominator are evaluated on the SAME Q member: each event's evidence
        # carries 1/Z_m from the member's redshift-prior normalizer, and it
        # cancels against mu(Q_m)^{-N_obs}.  Against a mean-Q mu it does not
        # cancel, leaving L_m ~ (Zbar/Z_m)^{N_obs} -- with N_obs ~ 10^2 a 1%
        # spread in Z_m tilts the logsumexp by e^{2.6} per member, so the
        # "average" collapses onto the smallest-Z_m member.  Reject the
        # asymmetric structure instead of silently returning that number.
        missing_sel = [k + 1 for k, f in enumerate(sel_members_flags) if not f]
        if missing_sel:
            raise ValueError(
                "lss_marginalize=True requires the LSS-completion ENSEMBLE on "
                f"EVERY SELECTION catalog too; catalog(s) {missing_sel} have "
                "none. The per-member redshift-prior normalizer Z_m cancels only "
                "against mu(Q_m); pairing member numerators with a "
                "posterior-mean-Q mu makes the member average a Z_m^{-N_obs}-"
                "weighted pick, not a marginalization. Build the selection "
                "catalog's Q_LSS with the same members as the PE catalog's."
            )

        if lss_member_impl == "factored":
            ll, ll_members = _factored_member_marginalization(n_members, is_field)
        elif lss_member_impl == "reference" and lss_field_mode == "latent":
            # The reference path exists to be the memory-heavy numerical pin:
            # it DELIBERATELY materialises the (M, N_rows, N_grid) member cube
            # from the resident table.  In latent mode there is no table, and
            # building the cube from the field would be 1.06 GB at DESI scale
            # while contradicting the seam's entire premise.  Refuse rather
            # than provide a "reference" that is not the reference.
            raise ValueError(
                "lss_member_impl='reference' is not available with "
                "lss_field_mode='latent': the reference path pins the factored "
                "one by materialising the member Q cube, which the latent seam "
                "exists to avoid. Use lss_member_impl='factored' (the default); "
                "the seam's own pin is tests/test_latent_seam.py."
            )
        elif lss_member_impl == "reference":
            ll, ll_members = _reference_member_marginalization(n_members, is_field)
        else:
            raise ValueError(
                "lss_member_impl must be 'factored' or 'reference', got "
                f"{lss_member_impl!r}."
            )
    else:
        ll = _ll_given_states(prior_states_univ, prior_states_sel)
        ll_members = None

    if frozen:
        # The frozen prior's value premise, re-verified IN THE GRAPH on the
        # proposal actually served: every cosmology / survey scalar the prior
        # reads must be the fixed value it was built at.  A violated premise
        # (a graph compiled for a frozen run replayed with a sampled H0, say)
        # turns the log-likelihood into -inf instead of a plausible number.
        live = frozen_prior_probe_vector(cosmo, surveys_all)
        ok = jnp.all(live == frozen_prior.probe_ref)
        ll = ll + jnp.where(ok, 0.0, jnp.nan)

    logL_total = jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
    if lss_member_diagnostics:
        # PLAN §6.4's runtime member diagnostics.  Python-level (static)
        # branch: with the flag off not one op below is traced, so the shipped
        # likelihood module is unchanged, and with it on this is a separate jit
        # specialization -- the same arrangement
        # ``darksiren_likelihood_diagnostics_with_clusters`` uses.
        #
        # The companion diagnostic PLAN §6.4 lists -- the per-member
        # Neff/variance guard -- is NOT built here, because it already exists:
        # ``_member_ll`` above calls ``selection_log_correction(log_mu_m,
        # Neff_m, nEvents, soft_guard=..., max_likelihood_variance=...,
        # pe_variance_sum=jnp.sum(event_vars))`` INSIDE the member vmap, so
        # every member is guarded on its OWN selection Neff and its OWN summed
        # per-event reweighting variance, and a member that fails either
        # criterion enters ``ll_members`` as -inf.  Rev 1 of the plan listed
        # this as new work (R1-SEV3-11); it is not, and adding a second guard
        # would double-count the wall.  ``tests/test_latent_p13.py`` asserts it
        # is live in latent mode rather than asserting it was written.
        return {
            "logL_total": logL_total,
            "ll_members": ll_members,
            "member_ess": member_ess(ll_members),
            "n_members": jnp.asarray(n_members),
        }
    return logL_total


def darksiren_member_diagnostics(*args, **kwargs):
    """Evaluate the likelihood once and return the PLAN §6.4 member diagnostics.

    The member-marginalization twin of
    ``darksiren_likelihood_diagnostics_with_clusters``: a separate static JIT
    specialization, so the sampler's own calls keep returning only the scalar
    log-likelihood and the production trace is untouched.  Requires
    ``lss_marginalize=True``; ``lss_field_mode`` is free (the ESS is a property
    of the member mixture, not of where ``Q`` came from), which is what lets the
    same read-out compare the table and latent arms.
    """
    kwargs["lss_member_diagnostics"] = True
    return darksiren_log_likelihood(*args, **kwargs)
