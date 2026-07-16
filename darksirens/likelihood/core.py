"""Pure JIT body for the hierarchical dark-siren likelihood."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.redshift.prior import (
    DarkSirenEnsemblePriorState,
    DarkSirenPriorState,
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.gw.populations import pop_model_parser
from darksirens.likelihood.selection import (
    DEFAULT_MAX_LIKELIHOOD_VARIANCE,
    compute_selection_term,
    log_evidence_and_mc_variance,
    selection_log_correction,
)
from darksirens.inference.utils import log_sample_weight
from darksirens.likelihood.wl_weight import (
    log_sample_weight_wl_or_standard,
    log_sample_weight_wl_lognormal_hermite,
)
from darksirens.lensing.grids import make_log_mu_grid, make_hermite_u_grid
from darksirens.lensing.wlmagnification import make_tabulated_log_p_wl
from darksirens.sky import sky_model_parser
from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.utils.cosmology import dL_grid_bounds, dL_in_z_grid, z_of_dL


# Weak-lensing quadrature node counts / ranges.
_WL_NMU_NODES = 16
_WL_LOG_MU_RANGE = (-0.6, 0.6)
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
    if lss_marginalize:
        raise NotImplementedError(
            "catalog_sky_weighting='field' is not supported with lss_marginalize."
        )
    if mark_model not in (None, "none"):
        raise NotImplementedError(
            "catalog_sky_weighting='field' is not supported with a marked-host "
            "model (mark_model)."
        )
    for cat in catalogs:
        if any(
            getattr(cat, name) is not None
            for name in (
                "lss_completion_logq",
                "lss_completion_q",
                "lss_completion_logq_members",
                "lss_completion_q_members",
            )
        ):
            raise NotImplementedError(
                "catalog_sky_weighting='field' is not supported with an "
                "LSS-completion Q_LSS table/ensemble."
            )
        if cat.field_dN_obs_s is None:
            raise ValueError(
                "catalog_sky_weighting='field' requires the survey-global field "
                "normalization inputs on every catalog (field_dN_obs_s / "
                "field_n_empty / field_N_obs_total); build them from the FULL-sky "
                "catalog via build_field_normalization_inputs."
            )
        # field_global_log_Z hard-codes lss = 1, so the numerator's overdensity
        # modulation must be the legacy dummy (delta_g_pix_z shape (1, N_grid),
        # guaranteed zero-valued upstream by the use_LSS=False CLI gate). The
        # VALUE cannot be asserted here (delta_g is a tracer under jit); the
        # static SHAPE is the jit-safe proxy. A real per-pixel delta_g with
        # field mode would silently bias log10n0/delta by the un-modulated
        # normalizer (measured 33% Z divergence in the adversarial review).
        if cat.delta_g_pix_z is not None and cat.delta_g_pix_z.shape[0] != 1:
            raise NotImplementedError(
                "catalog_sky_weighting='field' requires the dummy (1, N_grid) "
                "overdensity grid; a per-pixel delta_g is not supported "
                "(field_global_log_Z assumes lss = 1)."
            )


@partial(
    jax.jit,
    static_argnames=[
        "nEvents",
        "nsamp",
        "pop_model",
        "shared_beta",
        "shared_spin",
        "shared_gamma",
        "universe_model",
        "sel_batch_size",
        "sky_model",
        "mark_model",
        "mark_names",
        "wl_backend",
        "wl_selection",
        "lss_marginalize",
        "materialize_redshift_prior_state",
        "selection_neff_soft_guard",
        "n_catalogs",
        "catalog_sky_weighting",
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
    # --- catalog sky-weighting convention (dark_sirens only) ----------------
    # "conditional" (default): per-pixel normalizer Z[pix] -- bit-identical
    # legacy behaviour.  "field": survey-global normalizer so the mixture weight
    # is the host FRACTION.  Static; gated to the plain galaxy-count host model.
    catalog_sky_weighting: str = "conditional",
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
        if lss_marginalize:
            raise NotImplementedError(
                "lss_marginalize is not supported with the K-catalog mixture."
            )
        if mark_model not in (None, "none"):
            raise NotImplementedError(
                "Marked-host models are not supported with the K-catalog mixture."
            )
        if sky_model != "isotropic":
            raise NotImplementedError(
                "Only the isotropic sky model is supported with the K-catalog "
                f"mixture; got sky_model={sky_model!r}."
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
    prior_states_univ = tuple(
        prepare_redshift_prior_state(
            pe_model, cosmo, surveys_all[k], catalogs_pe_all[k],
            mark_model=mark_model, mark_params=mark_params, mark_names=mark_names,
            materialize_state=materialize_redshift_prior_state,
            catalog_sky_weighting=catalog_sky_weighting,
        )
        for k in range(n_catalogs)
    )
    prior_states_sel = tuple(
        prepare_redshift_prior_state(
            selection_model, cosmo, surveys_all[k], catalogs_sel_all[k],
            mark_model=mark_model, mark_params=mark_params, mark_names=mark_names,
            materialize_state=materialize_redshift_prior_state,
            catalog_sky_weighting=catalog_sky_weighting,
        )
        for k in range(n_catalogs)
    )

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
            log_p_wl_fn = make_tabulated_log_p_wl(wl_z_grid, wl_log_mu_grid, wl_log_p_table)
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

        def _pe_event_fn(_, event_idx):
            s = event_idx * nsamp
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
            dL_ev = sl(gw_pe.dL)
            valid = sl(gw_pe.valid) & (sl(gw_pe.prior_wt) > 0.0)
            # Per-event counterpart selection (bright sirens); the traced index
            # is inert for catalogs without counterpart fields (dark models).
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
            # Angular/3-D factor log g(n̂, z) per sample (skipped when isotropic).
            # Clamped dL for the same reverse-NaN reason as the weight paths.
            if apply_sky:
                dL_lo_s, dL_hi_s = dL_grid_bounds(H0, Om0, w0, wa)
                z_ev = z_of_dL(jnp.clip(dL_ev, dL_lo_s, dL_hi_s), H0, Om0, w0, wa)
                ldw = ldw + log_g_sky(
                    sl(gw_pe.nx), sl(gw_pe.ny), sl(gw_pe.nz), z_ev, sky_params
                )
            ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
            # NaN-safe reduction: an event whose samples ALL mask to -inf (the
            # sampler exploring, e.g., mmax below every sample of one event)
            # must contribute -inf, not a NaN backward softmax.
            return None, log_evidence_and_mc_variance(ldw, nsamp)

        _, (event_lls, event_vars) = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
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

    # LSS-completion marginalisation: logL = logsumexp_m logL(Λ; Q_m) − log M,
    # treating the M lognormal-completion members as Monte-Carlo draws of the
    # missing-galaxy field.  Opt-in (static); off by default so the deterministic
    # (posterior-mean Q) path is bit-for-bit unchanged.
    if lss_marginalize:
        if getattr(prior_states_univ[0], "dN_miss_members", None) is None:
            raise ValueError(
                "lss_marginalize=True requires an LSS-completion ENSEMBLE on the "
                "PE catalog. Build Q_LSS with members "
                "(darksirens_build_lognormal_completion --n-members M > 0) and pass it "
                "via --lss_completion; only universe_model='dark_sirens' supports it."
            )
        n_members = prior_states_univ[0].log_Z_members.shape[0]
        # Per-member states reuse the (Q-independent) kernels + log_Nobs and swap
        # in the member missing-galaxy density / normalisation; ONE vmap over the
        # shared member axis of every catalog's state (K = 1 today: the K >= 2
        # guard above keeps the mixture off this path until the shared-member
        # design lands).  ``log_Z_global`` is inert here (field mode is gated off
        # for lss_marginalize); carry a broadcast scalar so the vmap in_axes
        # pytree (member_axes) has a matching leaf.
        def _member_states(state):
            return DarkSirenPriorState(
                kernels=state.kernels, log_Nobs=state.log_Nobs,
                dN_miss=state.dN_miss_members, log_Z=state.log_Z_members,
                log_Z_global=jnp.asarray(0.0),
            )

        univ_members = tuple(_member_states(s) for s in prior_states_univ)
        sel_has_members = (
            getattr(prior_states_sel[0], "dN_miss_members", None) is not None
        )
        sel_members = (
            tuple(_member_states(s) for s in prior_states_sel)
            if sel_has_members else prior_states_sel
        )
        member_axes_one = DarkSirenPriorState(
            kernels=None, log_Nobs=None, dN_miss=0, log_Z=0, log_Z_global=None
        )
        member_axes = (member_axes_one,) * n_catalogs
        ll_members = jax.vmap(
            _ll_given_states,
            in_axes=(member_axes, member_axes if sel_has_members else None),
        )(
            univ_members, sel_members
        )
        ll_members = jnp.where(jnp.isfinite(ll_members), ll_members, -jnp.inf)
        ll = logsumexp(ll_members) - jnp.log(n_members)
    else:
        ll = _ll_given_states(prior_states_univ, prior_states_sel)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
