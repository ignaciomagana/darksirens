"""Pure JIT body for the hierarchical dark-siren likelihood."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.redshift.prior import (
    DarkSirenPriorState,
    eval_dark_member_completion,
    eval_dark_obs_bracket,
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.redshift.completion import (
    _member_q_eff_from_logq,
    _resolve_member_logq_row,
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
    dL_grid_bounds,
    threads_distance_table,
    z_of_dL,
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
# injection samples as the PE term (lognormal backend only — disabled or
# tabulated backends fall through to STANDARD for backward compatibility).
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
#           via pixel_to_cache_idx; zgals/wgals/ngals on the uncached fallback),
#       _assemble/_completion_curves_row (delta_g_pix_z);
#   catalog_kernel_state / marked_catalog_kernel_state (zgals, dzgals, wgals, ngals);
#   _row_counts (ngals, wgals);
#   the mark parser _gather_marks (mark_logmstar/logssfr/metallicity/color; zgals);
#   field_global_log_Z / _members / _marked (field_dN_obs_s, field_n_empty,
#       field_N_obs_total, field_lss_q(+_empty_sum)(+_members), field_delta_g,
#       field_mark_z/w/values, field_depth_z/dz/c under a survey depth).
# The state is a PURE function of (model, cosmo, survey, mark params,
# sky-weighting) plus these leaves, so two EMCatalogs sharing the SAME object for
# every one of them yield the identical state.  Leaves prepare NEVER reads --
# sample_to_unique_idx, the counterpart_* plumbing, active_counterpart_index,
# bright_siren_sky_marginalized, pixel_to_cache_idx (dN_obs_kde is indexed
# directly by row now -- see completion._row_C), and lss_completion_indexing
# (consumed EAGERLY in the factory, before this call) -- are deliberately
# EXCLUDED so a PE/selection pair that differs ONLY in those (the post-union
# multitracer bundle, and the flat K=1 union path) can still share.
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
)

# Prior models whose STATE is EMCatalog-derived and thus dedup-eligible across
# the PE / selection seams.  ``spectral_sirens`` (the WL and bright-siren
# SELECTION model) is excluded so WL configs never share -- its state is a cheap
# cosmo-only vector and its trace must stay untouched; ``bright_sirens`` returns
# ``None`` and its PE model already differs from the spectral selection model.
_SHAREABLE_PRIOR_MODELS = frozenset({"dark_sirens", "dark_sirens_complete"})


def can_share_redshift_prior_state(pe_model, sel_model, cat_pe, cat_sel) -> bool:
    """Whether the PE and selection redshift-prior states are provably identical.

    True iff both seams resolve to the SAME dedup-eligible prior model AND every
    EMCatalog field ``prepare_redshift_prior_state`` reads
    (:data:`_PREPARE_STATE_CONSUMED_EMCATALOG_FIELDS`) is the identical object
    (``is``) in both catalogs.  In that case ``prepare_redshift_prior_state`` --
    a pure function of the model, the (seam-shared) cosmo/survey/mark params and
    those leaves -- returns the identical state, so it may be built ONCE and
    reused for both seams.  Object identity (never value equality) keeps this
    trace-safe and cheap and defaults to NOT sharing whenever any consumed field
    differs.
    """
    if pe_model != sel_model or pe_model not in _SHAREABLE_PRIOR_MODELS:
        return False
    return all(
        getattr(cat_pe, name) is getattr(cat_sel, name)
        for name in _PREPARE_STATE_CONSUMED_EMCATALOG_FIELDS
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


def _require_field_mode_scope(
    universe_model, wl_enabled, lss_marginalize, mark_model, catalogs
):
    """Reject FIELD-convention sky weighting outside its supported scope.

    All checks are static (universe/model strings, static bools, and pytree
    STRUCTURE via ``is not None``), so they resolve once per trace -- mirroring
    the K>=2 mixture ``NotImplementedError`` guards.  The estimand only holds
    when the missing-galaxy budget carries no LSS modulation (dummy overdensity,
    no Q_LSS), so those are rejected here.  Both ``dark_sirens`` and
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
        "materialize_redshift_prior_state",
        "selection_neff_soft_guard",
        "n_catalogs",
        "catalog_sky_weighting",
        "share_prior_state_by_catalog",
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
    """
    pop_params_shape = tuple(pop_params.shape)
    if pop_params.ndim == 0 or pop_params_shape[0] == 0:
        raise ValueError(
            "darksiren_log_likelihood received empty pop_params: "
            f"pop_model={pop_model!r}, pop_params.shape={pop_params_shape}. "
            "Verify parameter-space construction for this population model."
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
        dL_lo, dL_hi = dL_grid_bounds(cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
        z_c = z_of_dL(jnp.clip(dL, dL_lo, dL_hi),
                      cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
        return log_g_sky(nx, ny, nz, z_c, sky_params)

    sky_log_weight_fn = _sky_weight if apply_sky else None
    # Weak-lensing dispatch (static): wl_enabled gates ALL WL machinery off for
    # every non-WL model, so they remain numerically identical to the non-WL code.
    wl_enabled = wl_backend != WL_BACKEND_DISABLED
    # Selection-side WL marginalization: opt-in, lognormal backend only
    # (same fallthrough semantics as likelihood_with_clusters — a disabled or
    # tabulated backend keeps the exact legacy selection path).
    wl_selection_enabled = (
        wl_selection == WL_SELECTION_LOGNORMAL
        and wl_backend == WL_BACKEND_LOGNORMAL
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

    # FIELD-convention sky weighting scope gate (static; covers K = 1 and K >= 2).
    if catalog_sky_weighting == "field":
        _require_field_mode_scope(
            universe_model, wl_enabled, lss_marginalize, mark_model,
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
    univ_states = []
    sel_states = []
    for k in range(n_catalogs):
        state_univ = prepare_redshift_prior_state(
            pe_model, cosmo, surveys_all[k], catalogs_pe_all[k],
            mark_model=_mark_model_for(k), mark_params=mark_params_all[k],
            mark_names=mark_names_all[k],
            materialize_state=materialize_redshift_prior_state,
            catalog_sky_weighting=catalog_sky_weighting,
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
            )
        univ_states.append(state_univ)
        sel_states.append(state_sel)
    prior_states_univ = tuple(univ_states)
    prior_states_sel = tuple(sel_states)

    def _eval_prior_mix(model, states, z, pix, catalogs):
        """log p(z | pix) for ``model`` across the K-catalog mixture.

        K = 1 is a STATIC shortcut -- the bare per-catalog evaluation with no
        mixture weight and no logsumexp -- so the single-catalog compute graph
        is exactly the historical one.  For K >= 2 the per-catalog log priors
        are combined with the sampled log weights via the all--inf-safe
        logsumexp above.
        """
        if n_catalogs == 1:
            return eval_redshift_prior_with_state(
                model, states[0], z, _pix_col(pix, 0), cosmo, surveys_all[0],
                catalogs[0], catalog_sky_weighting=catalog_sky_weighting,
            )
        lps = [
            mixture_log_weights[k] + eval_redshift_prior_with_state(
                model, states[k], z, _pix_col(pix, k),
                cosmo, surveys_all[k], catalogs[k],
                catalog_sky_weighting=catalog_sky_weighting,
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
            return _eval_prior_mix(pe_model, states_univ, z, pix, catalogs)

        def log_prior_z_selection(z, pix, catalogs):
            return _eval_prior_mix(selection_model, states_sel, z, pix, catalogs)

        def _log_sample_weight_if_supported(m1det, q, dL, chieff, pix, prior_wt, catalogs):
            """PE per-sample weight; WL-marginalized when wl_enabled, else standard.

            ``catalogs`` is the length-K tuple of per-catalog EMCatalogs; the
            weight kernels pass it (and ``pix``) opaquely through to the
            ``log_prior_z`` closure, which resolves the per-catalog mixture.
            WL is statically K = 1 (guarded above).

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
            dL_lo, dL_hi = dL_grid_bounds(H0, Om0, w0, wa)
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
            if not wl_enabled:
                ldw = log_sample_weight(
                    m1det, q, dL_c, chieff, pix, prior_wt, cosmo, survey, pop_params,
                    catalogs, log_p_pop, log_prior_z,
                )
            elif wl_backend == WL_BACKEND_LOGNORMAL:
                ldw = log_sample_weight_wl_lognormal_hermite(
                    m1det, q, dL_c, chieff, pix, prior_wt,
                    cosmo, survey, pop_params, catalogs,
                    log_p_pop, log_prior_z,
                    wl_a, wl_b, u_nodes, log_wH_nodes,
                )
            else:
                ldw = log_sample_weight_wl_or_standard(
                    m1det, q, dL_c, chieff, pix, prior_wt,
                    cosmo, survey, pop_params, catalogs,
                    log_p_pop, log_prior_z,
                    log_p_wl_fn, mu_nodes, log_w_nodes,
                    wl_enabled=wl_enabled,
                )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

        def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalogs):
            """Selection weight in the canonical ``(m1det, q, dL, chieff)`` variables."""
            def _selection_prior(z, pix, catalogs):
                return log_prior_z_selection(z, pix, catalogs)

            # Same clamped-distance treatment as the PE weights (see above):
            # z_of_dL's NaN sentinel must never enter the arithmetic.
            dL_lo, dL_hi = dL_grid_bounds(H0, Om0, w0, wa)
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
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
                )
            else:
                ldw = log_sample_weight(
                    m1det, q, dL_c, chieff, pix, prior_wt, cosmo, survey, pop_params,
                    catalogs, log_p_pop, _selection_prior,
                )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

        def log_weight_ev(m1det, q, dL, chieff, pix, prior_wt, catalogs):
            """PE weight in the same ``(m1det, q, dL)`` variables as selection."""
            return _log_sample_weight_if_supported(
                m1det, q, dL, chieff, pix, prior_wt, catalogs
            )

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
            )
            # Angular/3-D factor log g(n̂, z) per sample (skipped when isotropic).
            # Clamped dL for the same reverse-NaN reason as the weight paths.
            if apply_sky:
                dL_lo_s, dL_hi_s = dL_grid_bounds(H0, Om0, w0, wa)
                z_ev = z_of_dL(jnp.clip(dL_ev, dL_lo_s, dL_hi_s), H0, Om0, w0, wa)
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
                )
                if apply_sky:
                    dL_lo_s, dL_hi_s = dL_grid_bounds(H0, Om0, w0, wa)
                    z_ev = z_of_dL(jnp.clip(dL_ev, dL_lo_s, dL_hi_s), H0, Om0, w0, wa)
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
            # call when exactly 1) plus one Python-level remainder chunk -- keeps
            # every shape static with no dynamic padding mask.
            block_samps = pe_block * nsamp
            n_full = nEvents // pe_block
            rem = nEvents - n_full * pe_block

            def _reduce_events(s, m):
                # s: flat sample start (traced or static); m: STATIC event count.
                ldw = _pe_chunk_ldw(s, m * nsamp).reshape(m, nsamp)
                return jax.vmap(
                    lambda row: log_evidence_and_mc_variance(row, nsamp)
                )(ldw)

            parts = []
            if n_full == 1:
                parts.append(_reduce_events(0, pe_block))
            elif n_full > 1:
                def _chunk_scan(_, chunk_idx):
                    return None, _reduce_events(chunk_idx * block_samps, pe_block)

                _, stacked = lax.scan(_chunk_scan, None, jnp.arange(n_full))
                # (n_full, pe_block) -> (n_full*pe_block,) preserving event order.
                parts.append(
                    jax.tree_util.tree_map(
                        lambda a: a.reshape((n_full * pe_block,) + a.shape[2:]),
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

    def _factored_member_marginalization(n_members, is_field, sel_has_members):
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

        dL_lo, dL_hi = dL_grid_bounds(H0, Om0, w0, wa)

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
        def _member_leaf_bundle(states, catalogs):
            out = []
            for s, c in zip(states, catalogs):
                logq_all, _ = _resolve_member_logq_row(c)  # (M, N_rows, N_grid) view
                logZ = s.log_Z_global_members if is_field else s.log_Z_members
                out.append((logq_all, logZ))
            return tuple(out)

        # ``base_miss`` is present only on ENSEMBLE states; the selection states
        # may be plain (asymmetric PE-members / selection-no-members case), where
        # these tuples are built but never read (the member reduction is hoisted),
        # so ``getattr(..., None)`` keeps the trace from touching a missing leaf.
        base_miss_univ = tuple(getattr(s, "base_miss", None) for s in prior_states_univ)
        base_miss_sel = tuple(getattr(s, "base_miss", None) for s in prior_states_sel)
        member_is_log_univ = tuple(
            _resolve_member_logq_row(c)[1] for c in catalogs_pe_all
        )
        member_is_log_sel = tuple(
            _resolve_member_logq_row(c)[1] for c in catalogs_sel_all
        )
        z_depth_all = tuple(surveys_all[k].z_depth for k in range(n_catalogs))

        # Per-catalog member completion -> per-sample log p_mix(z), reusing the
        # SAME K=1 static shortcut / K>=2 _mixture_logsumexp seam as
        # _eval_prior_mix, but on the precomputed observed brackets.  ``base_tup``
        # / ``is_log_tup`` are the per-SIDE (PE vs selection) member-independent
        # closures; ``leaves[k] = (member_logq_m, logZ_m)`` is the per-catalog
        # member-axis leaf peeled by the outer vmap.
        def _log_p_mix(A_obs, idx, t, pixk, leaves, base_tup, is_log_tup):
            def _one(k):
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
            # Sky factor uses the SAME clamped-dL redshift as base (z_c); it is
            # inserted AFTER the first mask in _pe_member_terms, exactly as before.
            sky = (
                log_g_sky(sl(gw_pe.nx), sl(gw_pe.ny), sl(gw_pe.nz), z_c, sky_params)
                if apply_sky else jnp.zeros_like(base)
            )
            return (base, supported, valid, sky, A_obs, idx, t, pixk)

        def _pe_precompute_chunk(s, m):
            # Evaluate m events (m*nsamp samples) in one flat pass, then split the
            # leading axis back to (m, nsamp, ...) so the stacked pytree keeps the
            # historical leaf shapes (nEvents, nsamp, ...) that _pe_member_terms
            # consumes UNCHANGED.
            flat = _pe_precompute_flat(s, m * nsamp)
            return jax.tree_util.tree_map(
                lambda a: a.reshape((m, nsamp) + a.shape[1:]), flat
            )

        block_samps = pe_block * nsamp
        n_full = nEvents // pe_block
        rem = nEvents - n_full * pe_block
        parts = []
        if n_full == 1:
            parts.append(_pe_precompute_chunk(0, pe_block))
        elif n_full > 1:
            def _chunk_scan(_, chunk_idx):
                return None, _pe_precompute_chunk(chunk_idx * block_samps, pe_block)

            _, stacked = lax.scan(_chunk_scan, None, jnp.arange(n_full))
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
            base, supported, valid, sky, A_obs, idx, t, pixk = pe_pre
            log_p_mix = _log_p_mix(
                A_obs, idx, t, pixk, leaves, base_miss_univ, member_is_log_univ
            )   # (nEvents, nsamp)
            # Reproduce _pe_event_fn's mask ORDER with base playing ldw's role:
            #   ldw -> mask(supported & isfinite) -> +sky -> mask(valid & isfinite).
            ldw = base + log_p_mix
            ldw = jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)
            if apply_sky:
                ldw = ldw + sky
            ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
            return jax.vmap(lambda row: log_evidence_and_mc_variance(row, nsamp))(ldw)

        # --- Selection: when the selection side carries NO member ensemble,
        # log_mu/Neff are member-INDEPENDENT, so compute them ONCE (hoisted out of
        # the vmap) with the existing compute_selection_term + log_weight closure
        # -- identical to the reference's per-member-identical recomputation. ---
        if not sel_has_members:
            def _selection_prior(z, pix, catalogs):
                return _eval_prior_mix(
                    selection_model, prior_states_sel, z, pix, catalogs
                )

            def _log_weight_sel(m1det, q, dL, chieff, pix, prior_wt, catalogs):
                supported = (dL >= dL_lo) & (dL <= dL_hi)
                dL_c = jnp.clip(dL, dL_lo, dL_hi)
                ldw = log_sample_weight(
                    m1det, q, dL_c, chieff, pix, prior_wt, cosmo, survey, pop_params,
                    catalogs, log_p_pop, _selection_prior,
                )
                return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

            log_mu_fixed, Neff_fixed, _ = compute_selection_term(
                gw_sel, catalogs_sel_all, _log_weight_sel, Ndraw, nEvents,
                sel_batch_size=sel_batch_size, sky_log_weight_fn=sky_log_weight_fn,
            )
        else:
            # Pad the injections (batched) exactly as compute_selection_term, then
            # precompute the member-INDEPENDENT per-injection quantities ONCE (a
            # batched scan bounds the KDE peak memory to O(sel_batch_size x
            # N_max_gals)); the per-member reduction re-slices these cheaply.
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
                # _sky_weight clamps dL internally, matching _batch_lse (which
                # passes the UNCLAMPED dL_b to sky_log_weight_fn).
                sky = (
                    sky_log_weight_fn(
                        sl(gw_sel_p.nx), sl(gw_sel_p.ny), sl(gw_sel_p.nz), dL_b
                    )
                    if apply_sky else jnp.zeros_like(base)
                )
                return base, supported, valid, sky, A_obs, idx, t, pixk

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
                base_a, sup_a, val_a, sky_a, A_obs_a, idx_a, t_a, pixk_a = sel_pre

                def _provider(start, size):
                    sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, size)
                    log_p_mix = _log_p_mix(
                        tuple(sl(a) for a in A_obs_a),
                        tuple(sl(a) for a in idx_a),
                        tuple(sl(a) for a in t_a),
                        tuple(sl(a) for a in pixk_a),
                        leaves, base_miss_sel, member_is_log_sel,
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
            if sel_has_members:
                log_mu_m, Neff_m, _ = _sel_member_reduction(sel_leaves)
            else:
                log_mu_m, Neff_m = log_mu_fixed, Neff_fixed
            ll_m = selection_log_correction(
                log_mu_m, Neff_m, nEvents,
                soft_guard=selection_neff_soft_guard,
                max_likelihood_variance=max_likelihood_variance,
                pe_variance_sum=jnp.sum(event_vars),
            )
            return ll_m + jnp.sum(event_lls)

        univ_stack = _member_leaf_bundle(prior_states_univ, catalogs_pe_all)
        if sel_has_members:
            sel_stack = _member_leaf_bundle(prior_states_sel, catalogs_sel_all)
            ll_members = jax.vmap(_member_ll, in_axes=(0, 0))(univ_stack, sel_stack)
        else:
            ll_members = jax.vmap(
                lambda ul: _member_ll(ul, None), in_axes=(0,)
            )(univ_stack)

        ll_members = jnp.where(jnp.isfinite(ll_members), ll_members, -jnp.inf)
        return logsumexp(ll_members) - jnp.log(n_members)

    def _reference_member_marginalization(n_members, is_field, sel_has_members):
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
        sel_members = (
            tuple(
                _member_states(s, surveys_all[k], catalogs_sel_all[k])
                for k, s in enumerate(prior_states_sel)
            )
            if sel_has_members else prior_states_sel
        )
        member_axes_one = DarkSirenPriorState(
            kernels=None, log_Nobs=None, dN_miss=0, log_Z=0,
            log_Z_global=(0 if is_field else None),
        )
        member_axes = (member_axes_one,) * n_catalogs
        ll_members = jax.vmap(
            _ll_given_states,
            in_axes=(member_axes, member_axes if sel_has_members else None),
        )(
            univ_members, sel_members
        )
        ll_members = jnp.where(jnp.isfinite(ll_members), ll_members, -jnp.inf)
        return logsumexp(ll_members) - jnp.log(n_members)

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
        if any(sel_members_flags) and not all(sel_members_flags):
            raise ValueError(
                "lss_marginalize: the selection-side Q ensembles must be "
                "present on ALL catalogs or NONE (mixed member structure "
                "cannot share the member vmap)."
            )
        sel_has_members = all(sel_members_flags) and len(sel_members_flags) > 0

        if lss_member_impl == "factored":
            ll = _factored_member_marginalization(n_members, is_field, sel_has_members)
        elif lss_member_impl == "reference":
            ll = _reference_member_marginalization(n_members, is_field, sel_has_members)
        else:
            raise ValueError(
                "lss_member_impl must be 'factored' or 'reference', got "
                f"{lss_member_impl!r}."
            )
    else:
        ll = _ll_given_states(prior_states_univ, prior_states_sel)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
