"""Total-log-likelihood-variance guard (GWTC-4.0/5.0 criterion).

Two pieces are exercised here:

* ``log_evidence_and_mc_variance(ldw, nsamp)`` — the per-event Monte-Carlo
  log-evidence ``ln Z = -log n + logsumexp(ldw)`` and the delta-method
  variance of ``ln Z``, ``sigma^2 = Sum_j w_j^2 / (Sum_j w_j)^2 - 1/n``,
  computed NaN-safely.  Masked samples enter as ``-inf`` and COUNT in n;
  all-``-inf`` input returns ``(-inf, 0.0)``; ``sigma^2 in [0, 1 - 1/n]``.

* ``selection_log_correction(..., max_likelihood_variance, pe_variance_sum)``
  — the guard now bounds the variance of the TOTAL log-likelihood estimator,
  ``sigma^2_lnL = pe_variance_sum + N_obs^2 / N_eff <= max_likelihood_variance``,
  with the Vitale 5 N_obs mean floor retained.  ``pe_variance_sum = 0`` (the
  default) reduces exactly to the selection-only bound.

The end-to-end section drives ``darksiren_log_likelihood`` at the core level:
a PE fixture whose every event is effectively one-hot spends the whole
variance budget, so a selection integral that would pass on its own is
rejected at the default cap and admitted once the cap is relaxed.

All arithmetic runs in float64 with deterministic seeds, following the
conventions of ``test_selection_variance_guard.py`` /
``test_selection_soft_guard.py`` / ``test_likelihood_integration.py``.
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
for _p in (str(HERE), str(PKG_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from darksirens.likelihood.selection import (
    log_evidence_and_mc_variance,
    selection_log_correction,
)


# ============================================================
# 2a. Unit tests for log_evidence_and_mc_variance
# ============================================================

def test_uniform_weights_zero_variance():
    ldw = jnp.zeros(100)
    log_evidence, variance = log_evidence_and_mc_variance(ldw, 100)
    # sigma^2 = 0 exactly for uniform weights; ln Z = -log(100) + log(100) = 0.
    assert abs(float(variance) - 0.0) < 1e-12
    assert abs(float(log_evidence) - 0.0) < 1e-12


def test_one_hot_saturates_variance():
    n = 100
    ldw = jnp.full(n, -jnp.inf).at[0].set(0.0)
    log_evidence, variance = log_evidence_and_mc_variance(ldw, n)
    # A single sample carries all the weight: sigma^2 = 1 - 1/n.
    assert abs(float(variance) - (1.0 - 1.0 / n)) < 1e-12
    assert abs(float(log_evidence) - (-np.log(n))) < 1e-12


def test_all_masked_returns_neg_inf_and_zero_variance():
    ldw = jnp.full(7, -jnp.inf)
    log_evidence, variance = log_evidence_and_mc_variance(ldw, 7)
    # -inf log-evidence already kills the likelihood; the variance must stay
    # finite (0.0) so it cannot NaN-poison the total-variance guard.
    assert float(log_evidence) == -np.inf
    assert float(variance) == 0.0


def test_masked_samples_count_in_n():
    # n = 4 with two live (0.0) and two masked (-inf) samples:
    # Sum w^2 / (Sum w)^2 = 2 / 4 = 1/2, so sigma^2 = 1/2 - 1/4.
    ldw = jnp.asarray([0.0, 0.0, -jnp.inf, -jnp.inf])
    _, variance = log_evidence_and_mc_variance(ldw, 4)
    assert abs(float(variance) - (0.5 - 0.25)) < 1e-12


def test_reverse_mode_where_branch_is_nan_safe():
    # Both branches of the internal jnp.where are differentiated; with an
    # all-(-inf) base the dead branch must carry NO NaN cotangent.  Build the
    # perturbation so the -inf stays -inf: mask all False => where(mask, ., -inf)
    # is identically -inf regardless of t.
    n = 5
    x = jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = jnp.zeros(n, dtype=bool)

    def var_component(t):
        return log_evidence_and_mc_variance(jnp.where(mask, t * x, -jnp.inf), n)[1]

    def logZ_component(t):
        return log_evidence_and_mc_variance(jnp.where(mask, t * x, -jnp.inf), n)[0]

    g_var = float(jax.grad(var_component)(1.0))
    g_logZ = float(jax.grad(logZ_component)(1.0))
    assert np.isfinite(g_var), g_var
    assert np.isfinite(g_logZ), g_logZ


def test_variance_matches_numpy_identity_and_bootstrap():
    n = 1000
    ldw_np = 1.0 * np.random.default_rng(42).standard_normal(n)
    _, variance = log_evidence_and_mc_variance(jnp.asarray(ldw_np), n)
    variance = float(variance)

    # 1) Direct numpy identity Sum w^2 / (Sum w)^2 - 1/n.
    w = np.exp(ldw_np)
    direct = (w ** 2).sum() / w.sum() ** 2 - 1.0 / n
    assert np.isclose(variance, direct, rtol=1e-10, atol=0.0), (variance, direct)

    # 2) Bootstrap estimate of Var[log(mean(w_resampled))]: the delta-method
    #    variance is exactly this leading-order quantity, so they agree up to
    #    bootstrap noise (loose factor ~1.35).
    B = 2000
    rng = np.random.default_rng(7)
    logmeans = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        logmeans[b] = np.log(w[idx].mean())
    boot_var = float(logmeans.var())
    ratio = boot_var / direct
    assert 1.0 / 1.35 < ratio < 1.35, (boot_var, direct, ratio)


# ============================================================
# 2b. Guard behavior (selection_log_correction)
# ============================================================

def test_pe_variance_domination_rejects_healthy_selection():
    # N_obs = 50, selection comfortably healthy: Neff = 10 * max(5N, N^2).
    n = 50
    Neff = jnp.asarray(10.0 * max(5.0 * n, float(n * n)))  # 250000
    log_mu = jnp.asarray(0.0)

    # No per-event variance -> admitted.
    assert np.isfinite(float(
        selection_log_correction(log_mu, Neff, n, pe_variance_sum=0.0)))
    # pe_variance_sum = 1.5 exceeds the 1.0 budget -> rejected in hard mode
    # even though the selection integral alone is healthy.
    assert float(
        selection_log_correction(log_mu, Neff, n, pe_variance_sum=1.5)) == -np.inf

    # Soft mode: same point is driven far below the pe=0 value (wall engaged).
    soft0 = float(selection_log_correction(
        log_mu, Neff, n, soft_guard=True, pe_variance_sum=0.0))
    soft15 = float(selection_log_correction(
        log_mu, Neff, n, soft_guard=True, pe_variance_sum=1.5))
    assert np.isfinite(soft0) and np.isfinite(soft15)
    assert soft15 < soft0 - 100.0, (soft0, soft15)


def test_budget_split_shifts_the_threshold():
    # n = 50, pe_variance_sum = 0.6 -> budget 0.4 -> threshold N^2/0.4 = 6250.
    n = 50
    log_mu = jnp.asarray(0.0)
    assert float(selection_log_correction(
        log_mu, jnp.asarray(3000.0), n, pe_variance_sum=0.6)) == -np.inf
    assert np.isfinite(float(selection_log_correction(
        log_mu, jnp.asarray(7500.0), n, pe_variance_sum=0.6)))


def test_pe_variance_zero_is_bit_exact_reduction():
    # With pe_variance_sum = 0 the guard is bitwise identical to the legacy
    # selection-only criterion, hard and soft, on both sides of the threshold.
    n = 50
    log_mu = jnp.asarray(0.0)
    for Neff in (100.0, 249.0, 251.0, 2499.0, 2501.0, 1e6):
        for soft in (False, True):
            default = selection_log_correction(
                log_mu, jnp.asarray(Neff), n, soft_guard=soft)
            explicit = selection_log_correction(
                log_mu, jnp.asarray(Neff), n, soft_guard=soft, pe_variance_sum=0.0)
            a, b = float(default), float(explicit)
            # Bitwise: identical float (or -inf == -inf) on both sides.
            assert a == b or (a == -np.inf and b == -np.inf), (Neff, soft, a, b)
            assert np.asarray(default).tobytes() == np.asarray(explicit).tobytes(), (
                Neff, soft, a, b)


def test_soft_wall_continuous_and_grad_finite_in_pe_variance():
    # Fixed (log_mu, Neff, n); sweep pe_variance_sum through the wall turn-on.
    n = 50
    log_mu = jnp.asarray(0.0)
    Neff = jnp.asarray(1e6)

    def value(pe):
        return selection_log_correction(
            log_mu, Neff, n, soft_guard=True, pe_variance_sum=pe)

    # Coarse grid straddling the transition (near pe = 1 - N^2/Neff ~ 0.9975).
    coarse = [0.0, 0.5, 0.9, 0.99, 1.1, 2.0]
    # Fine grid resolving the (very narrow, ~2.5e-4-wide) gate turn-on.
    fine = list(np.linspace(0.995, 1.0, 201))

    for grid in (coarse, fine):
        vals = [float(value(jnp.asarray(pe))) for pe in grid]
        # (1) every value finite, no NaN/inf.
        assert all(np.isfinite(v) for v in vals), vals
        # (1) monotonically non-increasing as more budget is spent.
        assert all(vals[i] >= vals[i + 1] - 1e-9 for i in range(len(vals) - 1)), vals
        # (2) gradient w.r.t. pe finite at EVERY point (it may be exactly 0
        #     past budget exhaustion because of the 1e-12 clamp -- assert
        #     finiteness, NOT nonzeroness).
        for pe in grid:
            g = float(jax.grad(value)(jnp.asarray(float(pe))))
            assert np.isfinite(g), (pe, g)


def test_nan_neff_guards_in_hard_mode():
    # A NaN Neff must guard (return -inf), never admit: ~(Neff > threshold)
    # is True when the comparison is NaN.
    n = 50
    val = selection_log_correction(jnp.asarray(0.0), jnp.asarray(jnp.nan), n)
    assert float(val) == -np.inf


# ============================================================
# 2c. End-to-end through darksiren_log_likelihood (core level)
# ============================================================

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
from darksirens.utils.cosmology import H0Planck, Om0Planck
from darksirens.likelihood.core import darksiren_log_likelihood, WL_BACKEND_DISABLED
from darksirens.gw.populations import get_fixed_population_params


def _make_gw_event(n_events, n_samp, seed=0, onehot=False):
    """Tiny PE fixture (adapted from test_likelihood_integration).

    ``onehot=True`` masks every sample of each event except the first via
    ``prior_wt = 0`` (the core masks ``prior_wt <= 0`` samples to -inf), so
    each event keeps a single live sample and sigma^2_i = 1 - 1/n_samp.  The
    surviving sample is pinned to central, in-support values so its weight is
    finite (the -inf must come from the guard, not a dead event).
    """
    rng = np.random.default_rng(seed)
    total = n_events * n_samp
    m1det = np.asarray(rng.uniform(20.0, 60.0, total))
    m2det = np.asarray(rng.uniform(10.0, 30.0, total))
    dL = np.asarray(rng.uniform(400.0, 3000.0, total))
    chieff = np.asarray(rng.uniform(-0.3, 0.3, total))
    prior_wt = np.asarray(rng.uniform(0.5, 1.5, total))
    if onehot:
        prior_wt[:] = 0.0
        for e in range(n_events):
            i = e * n_samp
            prior_wt[i] = 1.0
            m1det[i], m2det[i], dL[i], chieff[i] = 35.0, 20.0, 1000.0, 0.0
    m1det = jnp.asarray(m1det)
    m2det = jnp.asarray(m2det)
    dL = jnp.asarray(dL)
    chieff = jnp.asarray(chieff)
    prior_wt = jnp.asarray(prior_wt)
    pixels = jnp.zeros(total, dtype=jnp.int32)
    valid = jnp.ones(total, dtype=jnp.bool_)
    q = m2det / m1det
    return GWEvent(m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
                   prior_wt=prior_wt, pixels=pixels, q=q, valid=valid)


# Shared toy builders: reuse the integration suite's fixtures (same
# cross-module import pattern as the LSS smoke below) instead of drifting
# copies; only _make_gw_event is local, for its onehot lever.
from test_likelihood_integration import _make_gw_sel, _toy_catalog  # noqa: E402


def _e2e_ll(gw_pe, max_likelihood_variance):
    cosmo = CosmoParams(H0=H0Planck, Om0=Om0Planck)
    survey = SurveyParams(n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5)
    cat = _toy_catalog()
    pop = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
    gw_sel = _make_gw_sel(500, seed=10)
    return darksiren_log_likelihood(
        cosmo, survey, pop, gw_pe, cat, gw_sel, cat,
        4, 200, 1000.0,
        pop_model="powerlaw+peak", universe_model="spectral_sirens",
        sel_batch_size=None, wl_backend=WL_BACKEND_DISABLED,
        max_likelihood_variance=max_likelihood_variance,
    )


def test_end_to_end_total_variance_guard():
    # Healthy fixture: finite at the default cap.
    gw_healthy = _make_gw_event(4, 200, seed=0, onehot=False)
    assert jnp.isfinite(_e2e_ll(gw_healthy, 1.0)), "healthy fixture not finite at default"

    # One-hot PE: each of the 4 events spends ~1 - 1/200 of variance, so
    # pe_variance_sum ~ 3.98 >> 1.  The SAME (healthy) selection set passes on
    # its own -- the -inf is purely the per-event budget exhaustion.
    gw_onehot = _make_gw_event(4, 200, seed=0, onehot=True)
    assert float(_e2e_ll(gw_onehot, 1.0)) == -np.inf, (
        "one-hot PE should exhaust the default variance budget")

    # Relaxing the cap to 10.0 restores a finite value.
    assert jnp.isfinite(_e2e_ll(gw_onehot, 10.0)), (
        "one-hot PE should be admitted with max_likelihood_variance=10.0")


# ============================================================
# 2d. LSS / vmap smoke: the traced threshold must vmap over members
# ============================================================

def test_lss_marginalize_path_finite_at_default():
    # Reuse the existing LSS-marginalisation fixture; the lss_marginalize=True
    # path vmaps the per-member likelihood, so the traced total-variance
    # threshold must survive vmap and still return a finite ll at the default.
    import test_lss_marginalization as L

    cat = L._dark_catalog(logq_members=L._members_table())
    ll = float(L._ll(cat, marginalize=True))
    assert np.isfinite(ll), ll
