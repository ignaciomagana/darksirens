"""Cluster-wrapper remediation regressions (adversarial review, batch ultra).

* P2-07 gradient safety — the shared-campaign covariance fold clamps the
  channel-2 variance slot to zero when ``2 mu_1L mu_2 / N >= sigma2_2``.  The
  clamp's forward value ``log1p(-exp(0)) = -inf`` is correct, but the slope is
  infinite there, and reverse-mode multiplied the zero cotangent from the
  downstream ``logaddexp`` into NaN, poisoning every hyperparameter gradient
  while logL stayed finite (the class commit d039ffc fixed in
  ``cluster_selection.py``).  The fold now lives in
  ``_fold_shared_campaign_covariance`` with double-where discipline.
* WL selection contract parity — ``darksiren_log_likelihood_with_clusters``
  silently downgraded ``wl_selection=LOGNORMAL`` under the TABULATED backend
  where ``likelihood/core.py`` raises (commit 5cab31a, F-140): numerator
  marginalized p_WL(mu|z) over an un-marginalized selection integral.  The
  wrapper now refuses too, and the lensing CLI resolves the
  ``--allow_mismatched_wl_selection`` ablation to the selection integral that
  actually runs ('standard') instead of relying on the silent downgrade.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.gw.populations import get_fixed_population_params
from darksirens.lensing.slmarks import make_sis_lens_params
from darksirens.likelihood.cluster_selection import combined_selection_log_correction
from darksirens.likelihood import likelihood_with_clusters as lwc
from darksirens.utils.cosmology import H0Planck, Om0Planck


# ---------------------------------------------------------------------------
# P2-07: covariance-fold gradients at saturation
# ---------------------------------------------------------------------------

def _p2_selection_loss(theta, soft_guard):
    """The fold plus its real consumer, as wired in the master likelihood."""
    log_mu_1L, log_mu_2, log_sigma2_2, log_n = theta
    log_cov2 = jnp.log(2.0) + log_mu_1L + log_mu_2 - log_n
    folded = lwc._fold_shared_campaign_covariance(log_sigma2_2, log_cov2)
    log_mu_1 = jnp.logaddexp(jnp.asarray(-1.0), log_mu_1L)
    log_sigma2_1 = jnp.asarray(-6.0)
    return combined_selection_log_correction(
        log_mu_1, log_sigma2_1, log_mu_2, folded,
        n_singletons_observed=3, n_clusters_observed=1,
        soft_guard=soft_guard,
    )


@pytest.mark.parametrize("soft_guard", [False, True])
def test_p2_covariance_fold_gradient_finite_at_saturation(soft_guard):
    """cov2 >= sigma2_2 clamps the slot to zero variance: the forward value is
    finite through the combined correction, and the gradient must be too —
    the pre-fix fold returned NaN for every input feeding the lensed means."""
    theta_sat = jnp.asarray([-2.0, -2.0, -30.0, jnp.log(10.0)])
    val = _p2_selection_loss(theta_sat, soft_guard)
    grad = jax.grad(_p2_selection_loss)(theta_sat, soft_guard)
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.asarray(grad))), grad


def test_p2_covariance_fold_forward_values():
    fold = lwc._fold_shared_campaign_covariance

    # Unsaturated live regime: exact logdiffexp, finite gradient.
    log_cov2 = float(np.log(2.0) - 4.0 - np.log(10.0))
    folded = fold(jnp.asarray(-3.0), jnp.asarray(log_cov2))
    np.testing.assert_allclose(
        float(folded), np.log(np.exp(-3.0) - np.exp(log_cov2)), rtol=1e-12
    )
    g = jax.grad(lambda s: fold(s, jnp.asarray(log_cov2)))(jnp.asarray(-3.0))
    assert np.isfinite(float(g))

    # Saturated: clamped to zero corrected variance.
    assert float(fold(jnp.asarray(-30.0), jnp.asarray(log_cov2))) == -np.inf

    # Dead channels: either side -inf leaves the slot untouched.
    assert float(fold(jnp.asarray(-jnp.inf), jnp.asarray(log_cov2))) == -np.inf
    assert float(fold(jnp.asarray(-3.0), jnp.asarray(-jnp.inf))) == -3.0


def test_p2_covariance_fold_is_wired_into_the_master():
    import inspect

    src = inspect.getsource(lwc.darksiren_log_likelihood_with_clusters)
    assert "_fold_shared_campaign_covariance(" in src, (
        "the master likelihood must fold the shared-campaign covariance "
        "through the gradient-safe helper, not an inline logdiffexp"
    )


# ---------------------------------------------------------------------------
# WL selection contract parity with likelihood/core.py
# ---------------------------------------------------------------------------

def _tiny_event(rng, n):
    return GWEvent(
        m1det=jnp.asarray(rng.uniform(20.0, 60.0, n)),
        m2det=jnp.asarray(rng.uniform(10.0, 30.0, n)),
        dL=jnp.asarray(rng.uniform(400.0, 3000.0, n)),
        chieff=jnp.asarray(rng.uniform(-0.3, 0.3, n)),
        prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, n)),
        pixels=jnp.zeros(n, dtype=jnp.int32),
        q=jnp.asarray(rng.uniform(0.3, 1.0, n)),
        valid=jnp.ones(n, dtype=jnp.bool_),
    )


def test_cluster_wrapper_refuses_wl_lognormal_selection_under_tabulated_backend():
    """The tabulated backend has NO matched selection integral: the wrapper
    must refuse like core.py (5cab31a), not silently run the standard one."""
    rng = np.random.default_rng(0)
    n_events, n_samp, n_sel = 2, 8, 16
    catalog = EMCatalog(
        apix=1.0, zgals=jnp.zeros((1, 1)), dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)), ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    with pytest.raises(ValueError, match="tabulated backend"):
        lwc.darksiren_log_likelihood_with_clusters(
            CosmoParams(H0=H0Planck, Om0=Om0Planck),
            SurveyParams(
                n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5,
            ),
            jnp.asarray(get_fixed_population_params("powerlaw+peak")),
            _tiny_event(rng, n_events * n_samp), catalog,
            _tiny_event(rng, n_sel), catalog,
            n_events, n_samp, 100.0,
            singleton_indices=jnp.arange(n_events, dtype=jnp.int32),
            pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            n_singletons=n_events, n_pairs=0,
            lensed_injections=None, pair_kdes=None,
            sis_params=make_sis_lens_params(A_tau=1e-3, n_tau=3.0),
            log_p_tag_per_source=jnp.zeros(0),
            pop_model="powerlaw+peak",
            universe_model="spectral_sirens_wl",
            cluster_mode=lwc.CLUSTER_MODE_OFF,
            wl_backend=lwc.WL_BACKEND_TABULATED,
            wl_selection=lwc.WL_SELECTION_LOGNORMAL,
        )


def _wl_opts(**kw):
    import argparse

    ns = argparse.Namespace(
        wl_selection="wl_lognormal",
        wl_backend="tabulated",
        lensing_wl_a=4e-3,
        allow_mismatched_wl_selection=False,
    )
    for key, val in kw.items():
        setattr(ns, key, val)
    return ns


def test_cli_refuses_wl_lognormal_selection_under_tabulated_backend():
    from darksirens.cli.inference_lensing import _resolve_wl_selection

    with pytest.raises(SystemExit, match="wl_lognormal needs"):
        _resolve_wl_selection(_wl_opts())


def test_cli_resolves_accepted_mismatch_to_the_selection_that_runs():
    """--allow_mismatched_wl_selection under wl_lognormal+tabulated must
    resolve to 'standard' (the integral that executes), since the likelihood
    entry points now refuse the LOGNORMAL+TABULATED pair outright."""
    from darksirens.cli.inference_lensing import _resolve_wl_selection

    opts = _wl_opts(allow_mismatched_wl_selection=True)
    with pytest.warns(RuntimeWarning, match="MISMATCHED"):
        _resolve_wl_selection(opts)
    assert opts.wl_selection == "standard"
    assert opts.wl_selection_requested == "wl_lognormal"
