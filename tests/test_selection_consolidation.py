import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.special import logsumexp

from darksirens.inference.events import make_gw_event, pad_gw_event_to_multiple
from darksirens.inference.selection import (
    compute_selection_term,
    selection_log_correction,
)
from darksirens.utils.containers import EMCatalog
from darksirens.utils.utils import logdiffexp


def _catalog():
    return EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((2, 1)),
        dzgals=jnp.zeros((2, 1)),
        wgals=jnp.ones((2, 1)),
        ngals=jnp.ones(2, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((2, 1)),
        sigma_kernel=0.1,
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _fixture_event(n_sel=8):
    return make_gw_event(
        m1det=jnp.linspace(25.0, 39.0, n_sel),
        m2det=jnp.linspace(15.0, 23.0, n_sel),
        dL=jnp.linspace(250.0, 950.0, n_sel),
        chieff=jnp.linspace(-0.2, 0.3, n_sel),
        prior_wt=jnp.array([1.0, 0.4, 2.0, 0.0, 1.5, 0.8, 1.2, 0.7])[:n_sel],
        pixels=jnp.arange(n_sel, dtype=jnp.int32) % 2,
    )


def _fixture_log_weight(m1det, q, dL, chieff, pix, prior_wt, catalog):
    del catalog
    weights = (
        jnp.log(prior_wt)
        + 0.01 * m1det
        - 0.2 * q
        - 0.0003 * dL
        + 0.05 * chieff
        + 0.1 * pix
    )
    # Exercise the finite guard independently of the structural prior_wt mask.
    return jnp.where(dL > 900.0, jnp.inf, weights)


def _legacy_selection_term(gw_sel, em_catalog_sel, log_weight_fn, Ndraw, nEvents, sel_batch_size=None):
    """Copy of the pre-consolidation likelihood.py selection block."""

    def _sel_batch_lse(dL_b, m1det_b, q_b, chi_b, pix_b, pwt_b):
        ldw = log_weight_fn(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b, em_catalog_sel)
        valid = pwt_b > 0.0
        ldw = jnp.where(valid & jnp.isfinite(ldw), ldw, -jnp.inf)
        return logsumexp(ldw), logsumexp(2.0 * ldw)

    if sel_batch_size is None:
        lse, lse2 = _sel_batch_lse(
            gw_sel.dL,
            gw_sel.m1det,
            gw_sel.q,
            gw_sel.chieff,
            gw_sel.pixels,
            gw_sel.prior_wt,
        )
        log_mu = lse - jnp.log(Ndraw)
        log_s2 = lse2 - 2.0 * jnp.log(Ndraw)
    else:
        N_sel = gw_sel.dL.shape[0]
        if N_sel % sel_batch_size != 0:
            raise ValueError("gw_sel length must be divisible by sel_batch_size")
        N_batches = N_sel // sel_batch_size

        def _scan_fn(_, batch_idx):
            start = batch_idx * sel_batch_size
            sl = lambda arr: lax.dynamic_slice_in_dim(arr, start, sel_batch_size)
            lse_b, lse2_b = _sel_batch_lse(
                sl(gw_sel.dL),
                sl(gw_sel.m1det),
                sl(gw_sel.q),
                sl(gw_sel.chieff),
                sl(gw_sel.pixels),
                sl(gw_sel.prior_wt),
            )
            return None, (lse_b, lse2_b)

        _, (lse_all, lse2_all) = lax.scan(_scan_fn, None, jnp.arange(N_batches))
        log_mu = logsumexp(lse_all) - jnp.log(Ndraw)
        log_s2 = logsumexp(lse2_all) - 2.0 * jnp.log(Ndraw)

    log_sigma2 = logdiffexp(log_s2, 2.0 * log_mu - jnp.log(Ndraw))
    Neff = jnp.where(
        jnp.isfinite(log_mu),
        jnp.exp(2.0 * log_mu - log_sigma2),
        0.0,
    )

    ll = jnp.where(Neff <= 5 * nEvents, -jnp.inf, 0.0)
    ll += -nEvents * log_mu + nEvents * (3 + nEvents) / (2.0 * Neff)
    return log_mu, Neff, jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


def _new_selection_term(gw_sel, catalog, Ndraw, nEvents, sel_batch_size=None):
    log_mu, neff, _log_sigma2 = compute_selection_term(
        gw_sel,
        catalog,
        _fixture_log_weight,
        Ndraw=Ndraw,
        nEvents=nEvents,
        sel_batch_size=sel_batch_size,
    )
    ll = selection_log_correction(log_mu, neff, nEvents)
    return log_mu, neff, jnp.where(jnp.isfinite(ll), ll, -jnp.inf)


def test_consolidated_selection_matches_legacy_unbatched_fixture():
    gw_sel = _fixture_event()
    catalog = _catalog()
    Ndraw = 20.0
    nEvents = 1

    legacy = _legacy_selection_term(
        gw_sel, catalog, _fixture_log_weight, Ndraw, nEvents, sel_batch_size=None
    )
    consolidated = _new_selection_term(gw_sel, catalog, Ndraw, nEvents, sel_batch_size=None)

    for actual, expected in zip(consolidated, legacy):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12)


def test_consolidated_selection_matches_legacy_batched_fixture():
    gw_sel, _ = pad_gw_event_to_multiple(_fixture_event(), 3)
    catalog = _catalog()
    Ndraw = 20.0
    nEvents = 1

    legacy = _legacy_selection_term(
        gw_sel, catalog, _fixture_log_weight, Ndraw, nEvents, sel_batch_size=3
    )
    consolidated = _new_selection_term(gw_sel, catalog, Ndraw, nEvents, sel_batch_size=3)

    for actual, expected in zip(consolidated, legacy):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12)


def test_consolidated_selection_preserves_all_invalid_guard():
    gw_sel = _fixture_event()
    catalog = _catalog()

    def all_invalid(m1det, q, dL, chieff, pix, prior_wt, catalog):
        del m1det, q, chieff, pix, prior_wt, catalog
        return jnp.full_like(dL, jnp.inf)

    log_mu, neff, _log_sigma2 = compute_selection_term(
        gw_sel,
        catalog,
        all_invalid,
        Ndraw=20.0,
        nEvents=1,
        sel_batch_size=4,
    )
    ll = selection_log_correction(log_mu, neff, nEvents=1)

    assert np.isneginf(np.asarray(log_mu))
    np.testing.assert_allclose(np.asarray(neff), 0.0)
    assert np.isneginf(np.asarray(ll))

def test_selection_term_contract_matches_likelihood_core_path():
    """Regression: likelihood_core path expects (log_mu, Neff, log_sigma2)."""
    from darksirens.inference import likelihood_core

    gw_sel = _fixture_event()
    catalog = _catalog()

    result = likelihood_core.compute_selection_term(
        gw_sel,
        catalog,
        _fixture_log_weight,
        Ndraw=20.0,
        nEvents=1,
        sel_batch_size=4,
    )

    assert isinstance(result, tuple)
    assert len(result) == 3

    log_mu, neff, log_sigma2 = result
    np.testing.assert_allclose(
        np.asarray(neff),
        np.asarray(jnp.exp(2.0 * log_mu - log_sigma2)),
        rtol=1e-12,
    )
