"""Pure JIT body for the hierarchical dark-siren likelihood."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.em.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.gw.populations import pop_model_parser
from darksirens.inference.selection import compute_selection_term, selection_log_correction
from darksirens.inference.utils import log_sample_weight
from darksirens.utils.containers import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.utils.cosmology import dL_in_z_grid


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
) -> jnp.ndarray:
    """Return ``log p({d_i} | cosmo, survey, pop_params)``."""
    log_p_pop = pop_model_parser(
        pop_model=pop_model,
        shared_beta=shared_beta,
        shared_spin=shared_spin,
        shared_gamma=shared_gamma,
    )
    selection_model = (
        "spectral_sirens" if universe_model == "bright_sirens" else universe_model
    )
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa

    # Per-proposal prior states: O(N_rows × N_grid) precomputation done ONCE
    # here, then captured by the per-sample closures below.  Because the
    # closures only *read* these arrays, neither the per-event ``lax.scan``
    # nor the selection batching recomputes them (the state arrays are
    # loop-invariant operands of the scans).  For ``bright_sirens`` the state
    # is None and the evaluator uses the live per-event catalog.
    prior_state_univ = prepare_redshift_prior_state(
        universe_model, cosmo, survey, em_catalog_pe
    )
    prior_state_sel = prepare_redshift_prior_state(
        selection_model, cosmo, survey, em_catalog_sel
    )

    # No finite guard on the redshift prior. -inf propagates correctly through
    # logsumexp and is caught by the final isfinite check.
    def log_prior_z(z, pix, catalog):
        return eval_redshift_prior_with_state(
            universe_model, prior_state_univ, z, pix, cosmo, survey, catalog
        )

    def log_prior_z_selection(z, pix, catalog):
        return eval_redshift_prior_with_state(
            selection_model, prior_state_sel, z, pix, cosmo, survey, catalog
        )

    def _log_sample_weight_if_supported(m1det, q, dL, chieff, pix, prior_wt, catalog):
        """Return -inf for distances outside the tabulated z(dL) support."""
        ldw = log_sample_weight(
            m1det,
            q,
            dL,
            chieff,
            pix,
            prior_wt,
            cosmo,
            survey,
            pop_params,
            catalog,
            log_p_pop,
            log_prior_z,
        )
        supported = dL_in_z_grid(dL, H0, Om0, w0, wa)
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
        """Selection weight in the canonical ``(m1det, q, dL)`` variables.

        Bright-siren selection injections already encode joint GW+EM
        detectability.  The selection integral should therefore use the
        population redshift distribution, not the observed counterparts'
        narrow redshift likelihoods.
        """
        def _selection_prior(z, pix, catalog):
            return log_prior_z_selection(z, pix, catalog)

        ldw = log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt, cosmo, survey, pop_params,
            catalog, log_p_pop, _selection_prior,
        )
        supported = dL_in_z_grid(dL, H0, Om0, w0, wa)
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    def log_weight_ev(m1det, q, dL, chieff, pix, prior_wt, catalog):
        """PE weight in the same ``(m1det, q, dL)`` variables as selection."""
        return _log_sample_weight_if_supported(
            m1det, q, dL, chieff, pix, prior_wt, catalog
        )

    log_mu, Neff = compute_selection_term(
        gw_sel,
        em_catalog_sel,
        log_weight,
        Ndraw,
        nEvents,
        sel_batch_size=sel_batch_size,
    )
    ll = selection_log_correction(log_mu, Neff, nEvents)

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
        ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
        return None, -jnp.log(nsamp) + logsumexp(ldw)

    _, event_lls = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
    ll += jnp.sum(event_lls)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
