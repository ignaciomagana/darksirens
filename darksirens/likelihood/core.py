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
from darksirens.redshift.catalog import _logsumexp_neginf_safe
from darksirens.gw.populations import pop_model_parser
from darksirens.likelihood.selection import compute_selection_term, selection_log_correction
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
        "lss_marginalize",
        "materialize_redshift_prior_state",
        "selection_neff_soft_guard",
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
    lss_marginalize: bool = False,
    materialize_redshift_prior_state: bool = True,
    selection_neff_soft_guard: bool = False,
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
    sky_log_weight_fn = (
        (lambda nx, ny, nz, dL: log_g_sky(
            nx, ny, nz, z_of_dL(dL, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa), sky_params
        )) if apply_sky else None
    )
    # Weak-lensing dispatch (static): wl_enabled gates ALL WL machinery off for
    # every non-WL model, so they remain numerically identical to the non-WL code.
    wl_enabled = wl_backend != WL_BACKEND_DISABLED
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
    # (population) redshift distribution for the selection integral.
    pe_model = (
        "spectral_sirens" if universe_model == "spectral_sirens_wl" else universe_model
    )
    selection_model = (
        "spectral_sirens"
        if universe_model in (
            "bright_sirens",
            "spectral_sirens_wl",
            "dark_sirens",
            "dark_sirens_complete",
        )
        else universe_model
    )
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    # Per-proposal prior states: O(N_rows × N_grid) precomputation done ONCE
    # here, then captured by the per-sample closures below.  Because the
    # closures only *read* these arrays, neither the per-event ``lax.scan``
    # nor the selection batching recomputes them (the state arrays are
    # loop-invariant operands of the scans).  For ``bright_sirens`` the state
    # is None and the evaluator uses the live per-event catalog.
    # The marked-host model (eta) reweights the dark-siren catalog prior; pass it
    # to both states so it is applied identically to the PE and selection terms
    # (prepare ignores it for non-dark_sirens models).
    prior_state_univ = prepare_redshift_prior_state(
        pe_model, cosmo, survey, em_catalog_pe,
        mark_model=mark_model, mark_params=mark_params, mark_names=mark_names,
        materialize_state=materialize_redshift_prior_state,
    )
    prior_state_sel = prepare_redshift_prior_state(
        selection_model, cosmo, survey, em_catalog_sel,
        mark_model=mark_model, mark_params=mark_params, mark_names=mark_names,
        materialize_state=materialize_redshift_prior_state,
    )

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

    def _ll_given_states(prior_state_univ, prior_state_sel):
        """Full log-likelihood (selection + PE) for ONE pair of redshift-prior
        states.  Factored out so the LSS-completion ensemble can be marginalised
        over (one call per Q_LSS member); for the default deterministic path it is
        called exactly once, so the compute graph is unchanged."""
        # No finite guard on the redshift prior. -inf propagates correctly through
        # logsumexp and is caught by the final isfinite check.
        def log_prior_z(z, pix, catalog):
            return eval_redshift_prior_with_state(
                pe_model, prior_state_univ, z, pix, cosmo, survey, catalog
            )

        def log_prior_z_selection(z, pix, catalog):
            return eval_redshift_prior_with_state(
                selection_model, prior_state_sel, z, pix, cosmo, survey, catalog
            )

        def _log_sample_weight_if_supported(m1det, q, dL, chieff, pix, prior_wt, catalog):
            """PE per-sample weight; WL-marginalized when wl_enabled, else standard.

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
                    catalog, log_p_pop, log_prior_z,
                )
            elif wl_backend == WL_BACKEND_LOGNORMAL:
                ldw = log_sample_weight_wl_lognormal_hermite(
                    m1det, q, dL_c, chieff, pix, prior_wt,
                    cosmo, survey, pop_params, catalog,
                    log_p_pop, log_prior_z,
                    wl_a, wl_b, u_nodes, log_wH_nodes,
                )
            else:
                ldw = log_sample_weight_wl_or_standard(
                    m1det, q, dL_c, chieff, pix, prior_wt,
                    cosmo, survey, pop_params, catalog,
                    log_p_pop, log_prior_z,
                    log_p_wl_fn, mu_nodes, log_w_nodes,
                    wl_enabled=wl_enabled,
                )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

        def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
            """Selection weight in the canonical ``(m1det, q, dL, chieff)`` variables."""
            def _selection_prior(z, pix, catalog):
                return log_prior_z_selection(z, pix, catalog)

            # Same clamped-distance treatment as the PE weights (see above):
            # z_of_dL's NaN sentinel must never enter the arithmetic.
            dL_lo, dL_hi = dL_grid_bounds(H0, Om0, w0, wa)
            supported = (dL >= dL_lo) & (dL <= dL_hi)
            dL_c = jnp.clip(dL, dL_lo, dL_hi)
            ldw = log_sample_weight(
                m1det, q, dL_c, chieff, pix, prior_wt, cosmo, survey, pop_params,
                catalog, log_p_pop, _selection_prior,
            )
            return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

        def log_weight_ev(m1det, q, dL, chieff, pix, prior_wt, catalog):
            """PE weight in the same ``(m1det, q, dL)`` variables as selection."""
            return _log_sample_weight_if_supported(
                m1det, q, dL, chieff, pix, prior_wt, catalog
            )

        log_mu, Neff, _log_sigma2 = compute_selection_term(
            gw_sel,
            em_catalog_sel,
            log_weight,
            Ndraw,
            nEvents,
            sel_batch_size=sel_batch_size,
            sky_log_weight_fn=sky_log_weight_fn,
        )
        ll = selection_log_correction(
            log_mu, Neff, nEvents, soft_guard=selection_neff_soft_guard
        )

        def _pe_event_fn(_, event_idx):
            s = event_idx * nsamp
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, s, nsamp)
            dL_ev = sl(gw_pe.dL)
            valid = sl(gw_pe.valid) & (sl(gw_pe.prior_wt) > 0.0)
            catalog_ev = em_catalog_pe._replace(active_counterpart_index=event_idx)
            ldw = log_weight_ev(
                sl(gw_pe.m1det),
                sl(gw_pe.q),
                dL_ev,
                sl(gw_pe.chieff),
                sl(gw_pe.pixels),
                sl(gw_pe.prior_wt),
                catalog_ev,
            )
            # Angular/3-D factor log g(n̂, z) per sample (skipped when isotropic).
            if apply_sky:
                z_ev = z_of_dL(dL_ev, H0, Om0, w0, wa)
                ldw = ldw + log_g_sky(
                    sl(gw_pe.nx), sl(gw_pe.ny), sl(gw_pe.nz), z_ev, sky_params
                )
            ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
            # NaN-safe reduction: an event whose samples ALL mask to -inf (the
            # sampler exploring, e.g., mmax below every sample of one event)
            # must contribute -inf, not a NaN backward softmax.
            return None, -jnp.log(nsamp) + _logsumexp_neginf_safe(ldw)

        _, event_lls = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
        return ll + jnp.sum(event_lls)

    # LSS-completion marginalisation: logL = logsumexp_m logL(Λ; Q_m) − log M,
    # treating the M lognormal-completion members as Monte-Carlo draws of the
    # missing-galaxy field.  Opt-in (static); off by default so the deterministic
    # (posterior-mean Q) path is bit-for-bit unchanged.
    if lss_marginalize:
        if getattr(prior_state_univ, "dN_miss_members", None) is None:
            raise ValueError(
                "lss_marginalize=True requires an LSS-completion ENSEMBLE on the "
                "PE catalog. Build Q_LSS with members "
                "(darksirens_build_lognormal_completion --n-members M > 0) and pass it "
                "via --lss_completion; only universe_model='dark_sirens' supports it."
            )
        n_members = prior_state_univ.log_Z_members.shape[0]
        # Per-member states reuse the (Q-independent) kernels + log_Nobs and swap in
        # the member missing-galaxy density / normalisation; vmap over the M axis.
        univ_members = DarkSirenPriorState(
            kernels=prior_state_univ.kernels, log_Nobs=prior_state_univ.log_Nobs,
            dN_miss=prior_state_univ.dN_miss_members, log_Z=prior_state_univ.log_Z_members,
        )
        sel_has_members = getattr(prior_state_sel, "dN_miss_members", None) is not None
        sel_members = (
            DarkSirenPriorState(
                kernels=prior_state_sel.kernels, log_Nobs=prior_state_sel.log_Nobs,
                dN_miss=prior_state_sel.dN_miss_members, log_Z=prior_state_sel.log_Z_members,
            )
            if sel_has_members else prior_state_sel
        )
        member_axes = DarkSirenPriorState(kernels=None, log_Nobs=None, dN_miss=0, log_Z=0)
        ll_members = jax.vmap(
            _ll_given_states,
            in_axes=(member_axes, member_axes if sel_has_members else None),
        )(
            univ_members, sel_members
        )
        ll_members = jnp.where(jnp.isfinite(ll_members), ll_members, -jnp.inf)
        ll = logsumexp(ll_members) - jnp.log(n_members)
    else:
        ll = _ll_given_states(prior_state_univ, prior_state_sel)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
