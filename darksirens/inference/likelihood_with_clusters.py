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

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.em import get_redshift_prior
from darksirens.gw.populations import pop_model_parser
from darksirens.inference.selection import compute_selection_term, selection_log_correction
from darksirens.inference.utils import log_sample_weight
from darksirens.inference.wl_weight import (
    log_sample_weight_wl_or_standard,
    log_sample_weight_wl_lognormal_hermite,
)
from darksirens.inference.cluster_likelihood import cluster_log_likelihood_pair
from darksirens.inference.cluster_selection import (
    compute_cluster_selection_term,
    combined_selection_log_correction,
)
from darksirens.inference.pair_kde import PairKDE, _slice_event_kde_inside_jit
from darksirens.lensing.grids import (
    make_log_mu_grid, make_hermite_u_grid, make_y_grid,
)
from darksirens.lensing.wlmagnification import (
    WLParams, make_lognormal_log_p_wl, make_tabulated_log_p_wl,
)
from darksirens.lensing.slmarks import SISLensParams
from darksirens.utils.containers import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.utils.cosmology import dL_in_z_grid


# Cluster-mode codes (static_argnames dispatch)
CLUSTER_MODE_OFF = 0          # singletons only — equivalent to commit 2
CLUSTER_MODE_J2 = 1           # singletons + J=2 pairs

# Quadrature node counts (same as commit 2 for WL; new for y)
_WL_NMU_NODES = 16
_WL_LOG_MU_RANGE = (-0.6, 0.6)
_WL_HERMITE_NODES = 16
_Y_NODES_FOR_CLUSTER_PAIR_LIKE = 32   # used inside L_2; finer than selection
                                       # since L_2 needs to localize y precisely

# WL backend codes (same as commit 2)
WL_BACKEND_DISABLED = -1
WL_BACKEND_LOGNORMAL = 0
WL_BACKEND_TABULATED = 1


@partial(
    jax.jit,
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
    log_p_pop = pop_model_parser(pop_model=pop_model)

    wl_enabled = (wl_backend != WL_BACKEND_DISABLED)
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
    H0, Om0 = cosmo.H0, cosmo.Om0

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
            log_p_wl_fn = make_tabulated_log_p_wl(
                wl_z_grid, wl_log_mu_grid, wl_log_p_table,
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
    # Singleton selection integral.  Uses the standard log_sample_weight
    # with the volume prior — unchanged from commit 2.
    # ──────────────────────────────────────────────────────────────────
    def log_weight_sel(m1det, q, dL, chieff, pix, prior_wt, catalog):
        def _selection_prior(z, pix, catalog):
            return log_prior_z_selection(z, pix, catalog)
        ldw = log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt,
            cosmo, survey, pop_params, catalog,
            log_p_pop, _selection_prior,
        )
        supported = dL_in_z_grid(dL, H0, Om0)
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    log_mu_1, Neff_1, log_sigma2_1 = compute_selection_term(
        gw_sel, em_catalog_sel, log_weight_sel, Ndraw, nEvents,
        sel_batch_size=sel_batch_size,
    )

    # ──────────────────────────────────────────────────────────────────
    # Cluster selection integral. Inert when cluster_mode == OFF —
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
        log_sigma2_2 = jnp.asarray(-jnp.inf, dtype=jnp.float64)

    # Combined selection correction
    ll = combined_selection_log_correction(
        log_mu_1, log_sigma2_1,
        log_mu_2, log_sigma2_2,
        n_singletons_observed=(n_singletons if cluster_mode == CLUSTER_MODE_J2 else nEvents),
        n_clusters_observed=(n_pairs if cluster_mode == CLUSTER_MODE_J2 else 0),
    )

    # ──────────────────────────────────────────────────────────────────
    # PE per-event term for singletons.
    # When cluster_mode == OFF: iterate over all nEvents (legacy behaviour).
    # When cluster_mode == J2: iterate over singleton_indices only.
    # ──────────────────────────────────────────────────────────────────
    def _log_sample_weight_if_supported(m1det, q, dL, chieff, pix, prior_wt, catalog):
        if wl_enabled and wl_backend == WL_BACKEND_LOGNORMAL:
            ldw = log_sample_weight_wl_lognormal_hermite(
                m1det, q, dL, chieff, pix, prior_wt,
                cosmo, survey, pop_params, catalog,
                log_p_pop, log_prior_z,
                wl_a, wl_b, u_nodes, log_wH_nodes,
            )
        else:
            ldw = log_sample_weight_wl_or_standard(
                m1det, q, dL, chieff, pix, prior_wt,
                cosmo, survey, pop_params, catalog,
                log_p_pop, log_prior_z,
                log_p_wl_fn, mu_nodes, log_w_nodes,
                wl_enabled=wl_enabled,
            )
        supported = dL_in_z_grid(dL, H0, Om0)
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

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
        return None, -jnp.log(nsamp) + logsumexp(ldw)

    if cluster_mode == CLUSTER_MODE_J2:
        # Sum singletons only
        _, event_lls = lax.scan(_pe_event_fn, None, singleton_indices)
    else:
        # Legacy: sum all events
        _, event_lls = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
    ll += jnp.sum(event_lls)

    # ──────────────────────────────────────────────────────────────────
    # Per-pair log-likelihood loop.  Skipped when cluster_mode == OFF.
    # ──────────────────────────────────────────────────────────────────
    if cluster_mode == CLUSTER_MODE_J2 and n_pairs > 0:
        # y-grid for cluster pair likelihoods
        y_nodes, log_wy = make_y_grid(_Y_NODES_FOR_CLUSTER_PAIR_LIKE)

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

        # Iterate pairs at Python level (n_pairs is static_argnames)
        for k in range(n_pairs):
            i = pair_indices[k, 0]
            j = pair_indices[k, 1]
            ev_i = _extract_event(i)
            ev_j = _extract_event(j)
            kde_i = _slice_event_kde_inside_jit(pair_kdes, i)
            kde_j = _slice_event_kde_inside_jit(pair_kdes, j)
            ll_pair = cluster_log_likelihood_pair(
                ev_i, ev_j, kde_i, kde_j,
                cosmo, survey, pop_params, em_catalog_pe, sis_params,
                log_p_pop, log_prior_z, y_nodes, log_wy,
            )
            ll = ll + jnp.where(jnp.isfinite(ll_pair), ll_pair, -jnp.inf)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
