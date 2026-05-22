"""
Patch for darksirens/inference/likelihood_core.py
==================================================

This patch threads WL marginalization through the per-event PE scan when
``universe_model == "spectral_sirens_wl"``. All other code paths are
**unchanged**: an exact reduction test must show numerically identical
output to the pre-patch code for any non-wl universe model.

The patch is presented as the full new file content for clarity. The
delta against master is:

  • New imports for the WL pieces.
  • Two new static arguments: ``wl_backend`` (int code: 0=lognormal,
    1=tabulated, -1=disabled) and ``wl_params`` (a WLParams instance,
    passed as a traced argument — its array fields are traced; only the
    backend choice is static).
  • The redshift-prior dispatcher's universe-model translation now maps
    'spectral_sirens_wl' → 'spectral_sirens' for the selection prior
    (since commit 2 does NOT touch the selection integral; that lands in
    commit 4 with a proper lensed-injection treatment).
  • The per-event PE scan body uses ``log_sample_weight_wl_or_standard``
    instead of the raw ``log_sample_weight``. When WL is OFF, the
    dispatcher returns bit-identical output to the standard path
    (verified by tests/test_wl_weight.py::TestReductionExact).

Critical correctness note
~~~~~~~~~~~~~~~~~~~~~~~~~
The Jacobian (log(1+z) + log dL'(z)) must come INSIDE the μ-integral
because z is itself a function of μ. The standard ``log_sample_weight``
evaluates this Jacobian once at z = z(dL_app), which is wrong when
marginalizing over μ. The WL hot path uses ``log_sample_weight_wl_marginalized``
from ``darksirens.inference.wl_weight``, which does this correctly.

Selection
~~~~~~~~~
The selection integral is NOT marginalized over μ in commit 2. The
selection term uses the standard ``log_sample_weight`` regardless of
universe_model. The leading-order WL effect on the rate normalization
is small and common to all population proposals; a full cluster-level
selection treatment lands in commit 4.
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
from darksirens.lensing.grids import make_log_mu_grid, make_hermite_u_grid
from darksirens.lensing.wlmagnification import (
    WLParams,
    make_lognormal_log_p_wl,
    make_tabulated_log_p_wl,
)
from darksirens.utils.containers import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.utils.cosmology import dL_in_z_grid


# Quadrature node counts
_WL_NMU_NODES = 16
_WL_LOG_MU_RANGE = (-0.6, 0.6)
_WL_HERMITE_NODES = 16

# Backend codes for the static_argnames dispatch
WL_BACKEND_DISABLED = -1
WL_BACKEND_LOGNORMAL = 0
WL_BACKEND_TABULATED = 1


@partial(
    jax.jit,
    static_argnames=[
        "nEvents",
        "nsamp",
        "pop_model",
        "universe_model",
        "sel_batch_size",
        "wl_backend",
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
    sel_batch_size: int | None = None,
    # WL extension (commit 2). All inert when wl_backend == WL_BACKEND_DISABLED.
    wl_backend: int = WL_BACKEND_DISABLED,
    wl_a: float = 0.0,
    wl_b: float = 0.0,
    wl_z_grid: jnp.ndarray | None = None,
    wl_log_mu_grid: jnp.ndarray | None = None,
    wl_log_p_table: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Return ``log p({d_i} | cosmo, survey, pop_params)``.

    When ``universe_model == "spectral_sirens_wl"`` the per-event PE
    integral is marginalized over the weak-lensing magnification using
    the parameters in the ``wl_*`` arguments. The selection integral is
    NOT marginalized (commit 2 scope); see module docstring.

    The ``wl_backend`` Python int gates the dispatch at trace time:
        -1 → WL disabled (default; bit-identical to pre-commit-2 code)
         0 → lognormal: uses (wl_a, wl_b)
         1 → tabulated: uses (wl_z_grid, wl_log_mu_grid, wl_log_p_table)
    """
    log_p_pop = pop_model_parser(pop_model=pop_model)

    # `universe_model` is a Python string (static_argnames), so this
    # branch is resolved at trace time.
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

    raw_logPriorUniv      = get_redshift_prior(pe_prior_name)
    raw_logPriorSelection = get_redshift_prior(sel_prior_name)
    H0, Om0 = cosmo.H0, cosmo.Om0

    def log_prior_z(z, pix, catalog):
        return raw_logPriorUniv(z, pix, cosmo, survey, catalog)

    def log_prior_z_selection(z, pix, catalog):
        return raw_logPriorSelection(z, pix, cosmo, survey, catalog)

    # ──────────────────────────────────────────────────────────────────
    # WL plumbing — only allocated when wl_enabled is True.
    # For lognormal, we use Gauss-Hermite quadrature in the standardized
    # variable u = (ln μ - m(z)) / s(z) — quadrature is exact at any s.
    # For tabulated, we keep Gauss-Legendre in ln μ.
    # ──────────────────────────────────────────────────────────────────
    if wl_enabled:
        if wl_backend == WL_BACKEND_LOGNORMAL:
            u_nodes, log_wH_nodes = make_hermite_u_grid(_WL_HERMITE_NODES)
            # Sentinels for the generic-dispatcher arguments (unused on this path).
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
    # Selection integral — UNCHANGED.  Always uses log_sample_weight
    # with the standard volume prior.  WL correction to selection is
    # deferred to commit 4.
    # ──────────────────────────────────────────────────────────────────
    def log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
        def _selection_prior(z, pix, catalog):
            return log_prior_z_selection(z, pix, catalog)

        ldw = log_sample_weight(
            m1det, q, dL, chieff, pix, prior_wt, cosmo, survey, pop_params,
            catalog, log_p_pop, _selection_prior,
        )
        supported = dL_in_z_grid(dL, H0, Om0)
        return jnp.where(supported & jnp.isfinite(ldw), ldw, -jnp.inf)

    log_mu, Neff, _log_sigma2 = compute_selection_term(
        gw_sel, em_catalog_sel, log_weight, Ndraw, nEvents,
        sel_batch_size=sel_batch_size,
    )
    ll = selection_log_correction(log_mu, Neff, nEvents)

    # ──────────────────────────────────────────────────────────────────
    # PE per-event term — uses the WL dispatcher.
    # Dispatch by backend: lognormal uses Hermite-Gauss (s-independent
    # quadrature); tabulated and disabled use the generic Legendre path.
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

    _, event_lls = lax.scan(_pe_event_fn, None, jnp.arange(nEvents))
    ll += jnp.sum(event_lls)

    return jnp.where(jnp.isfinite(ll), ll, -jnp.inf)
