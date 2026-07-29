"""Per-event/pair Monte-Carlo variance in the cluster stack's total guard.

Review finding P1-09: the standard core accumulates each event's delta-method
variance of ln Ẑ_i and spends it against the GWTC-4/5-style total-variance
budget, but the cluster/lensing stack computed its singleton and pair
evidences without any variance accumulation and called the combined selection
correction without ``pe_variance_sum`` — so its advertised total-variance
guard could pass a likelihood dominated by arbitrarily noisy event or pair
estimators as long as the SELECTION integral was well-sampled.

These tests pin the new plumbing end to end: the kernel-level variance
returns, the correlated branch combination, and the master-likelihood guard
firing on a low-ESS event fixture whose selection ESS is high.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from darksirens.likelihood.cluster_likelihood import (
    _correlated_branch_variance,
    cluster_log_likelihood_pair,
)

# Reuse the synthetic lensed-pair fixtures and toy physics from the main
# cluster-likelihood test module.
from test_cluster_likelihood import (
    _cosmo,
    _survey,
    _synth_lensed_pair,
    _toy_catalog,
    _toy_log_p_pop,
    _toy_volume_prior,
)

from darksirens.lensing.grids import make_y_grid
from darksirens.lensing.slmarks import make_sis_lens_params
from darksirens.likelihood.pair_kde import make_pair_kde


@pytest.fixture(scope="module")
def pair_setup():
    ev_i, ev_j = _synth_lensed_pair(
        z_true=0.7, m1src_true=30.0, q_true=0.7,
        chieff_true=0.0, y_true=0.4, n_pe=200, seed=0,
    )
    kde_i = make_pair_kde(
        ev_i["m1det"], ev_i["q"], ev_i["dL"], ev_i["chieff"], ev_i["prior_wt"],
    )
    kde_j = make_pair_kde(
        ev_j["m1det"], ev_j["q"], ev_j["dL"], ev_j["chieff"], ev_j["prior_wt"],
    )
    y_nodes, log_wy = make_y_grid(16)
    return {
        "ev_i": ev_i, "ev_j": ev_j, "kde_i": kde_i, "kde_j": kde_j,
        "y_nodes": y_nodes, "log_wy": log_wy,
        "sis_params": make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0),
        "cosmo": _cosmo(), "survey": _survey(), "catalog": _toy_catalog(),
        "pop_params": jnp.array([]),
    }


def _pair_ll(setup, ev_i, ev_j, **kw):
    return cluster_log_likelihood_pair(
        ev_i, ev_j, setup["kde_i"], setup["kde_j"],
        setup["cosmo"], setup["survey"], setup["pop_params"], setup["catalog"],
        setup["sis_params"], _toy_log_p_pop, _toy_volume_prior,
        setup["y_nodes"], setup["log_wy"], **kw,
    )


def test_pair_variance_return_is_backward_compatible(pair_setup):
    """Default return is the unchanged scalar; the variance path returns the
    SAME evidence plus a finite non-negative variance."""
    ll = _pair_ll(pair_setup, pair_setup["ev_i"], pair_setup["ev_j"])
    ll_v, var = _pair_ll(
        pair_setup, pair_setup["ev_i"], pair_setup["ev_j"],
        return_mc_variance=True,
    )
    assert float(ll) == float(ll_v)
    assert np.isfinite(float(var))
    assert float(var) >= 0.0


def test_pair_variance_grows_when_samples_collapse(pair_setup):
    """Masking all but a handful of PE samples must inflate the pair's
    delta-method variance toward its single-sample bound."""
    def _masked(ev, keep):
        valid = np.zeros(len(ev["valid"]), dtype=bool)
        valid[:keep] = True
        out = dict(ev)
        out["valid"] = jnp.asarray(valid)
        return out

    _, var_full = _pair_ll(
        pair_setup, pair_setup["ev_i"], pair_setup["ev_j"],
        return_mc_variance=True,
    )
    _, var_starved = _pair_ll(
        pair_setup,
        _masked(pair_setup["ev_i"], 2),
        _masked(pair_setup["ev_j"], 2),
        return_mc_variance=True,
    )
    assert float(var_starved) > float(var_full)
    # A 2-sample importance mean cannot beat sigma^2 <= 1 - 1/n.
    assert float(var_starved) <= 1.0


def test_correlated_branch_variance_edge_cases():
    inf = jnp.asarray(-jnp.inf)
    z = jnp.asarray(-3.0)

    # Dead branch: the live branch's variance passes through exactly.
    v = _correlated_branch_variance(z, jnp.asarray(0.04), inf, jnp.asarray(0.0))
    np.testing.assert_allclose(float(v), 0.04, rtol=1e-12)

    # Both dead: zero, not NaN.
    assert float(_correlated_branch_variance(inf, jnp.asarray(0.0),
                                             inf, jnp.asarray(0.0))) == 0.0

    # Equal branches with equal variance: perfectly-correlated combination
    # keeps the SAME variance ((0.5*s + 0.5*s)^2 = s^2).
    v = _correlated_branch_variance(z, jnp.asarray(0.09), z, jnp.asarray(0.09))
    np.testing.assert_allclose(float(v), 0.09, rtol=1e-12)

    # And it is differentiable at the edges (the guard is part of the soft
    # wall NUTS differentiates): no NaN cotangents.
    import jax

    g = jax.grad(
        lambda a: _correlated_branch_variance(
            a, jnp.asarray(0.04), inf, jnp.asarray(0.0)
        )
    )(z)
    assert np.isfinite(float(g))


class TestMasterGuardIncludesEventVariance:
    """A low-ESS EVENT fixture (selection ESS high) must now fail the total
    guard — the review's required negative test."""

    @pytest.fixture(scope="class")
    def fixture(self):
        from darksirens.core.types import GWEvent
        from darksirens.gw.populations.registry import get_fixed_population_params

        rng = np.random.default_rng(0)
        n_events, n_samp, n_sel = 4, 200, 5000
        total = n_events * n_samp

        prior_wt = rng.uniform(0.5, 1.5, total)
        # One sample per event carries ~all the importance weight
        # (weight ~ 1/prior_wt): per-event sigma^2 -> 1 - 1/n, so four
        # events overspend the default 1.0 total budget on their own.
        prior_wt[::n_samp] = 1e-9

        gw_pe = GWEvent(
            m1det=jnp.asarray(rng.uniform(20.0, 60.0, total)),
            m2det=jnp.asarray(rng.uniform(10.0, 30.0, total)),
            dL=jnp.asarray(rng.uniform(400.0, 3000.0, total)),
            chieff=jnp.asarray(rng.uniform(-0.3, 0.3, total)),
            prior_wt=jnp.asarray(prior_wt),
            pixels=jnp.zeros(total, dtype=jnp.int32),
            q=jnp.asarray(rng.uniform(0.3, 1.0, total)),
            valid=jnp.ones(total, dtype=jnp.bool_),
        )
        # Well-sampled selection set: high selection ESS by construction.
        gw_sel = GWEvent(
            m1det=jnp.asarray(rng.uniform(15.0, 70.0, n_sel)),
            m2det=jnp.asarray(rng.uniform(8.0, 35.0, n_sel)),
            dL=jnp.asarray(rng.uniform(200.0, 3000.0, n_sel)),
            chieff=jnp.asarray(rng.uniform(-0.3, 0.3, n_sel)),
            prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, n_sel)),
            pixels=jnp.zeros(n_sel, dtype=jnp.int32),
            q=jnp.asarray(rng.uniform(0.3, 1.0, n_sel)),
            valid=jnp.ones(n_sel, dtype=jnp.bool_),
        )
        return {
            "cosmo": _cosmo(), "survey": _survey(),
            "gw_pe": gw_pe, "gw_sel": gw_sel, "catalog": _toy_catalog(),
            "n_events": n_events, "n_samp": n_samp, "Ndraw": 10000.0,
            "pop_params": get_fixed_population_params("powerlaw+peak"),
        }

    def _diag(self, fixture, **kw):
        from darksirens.likelihood.likelihood_with_clusters import (
            CLUSTER_MODE_OFF,
            darksiren_likelihood_diagnostics_with_clusters,
        )

        return darksiren_likelihood_diagnostics_with_clusters(
            fixture["cosmo"], fixture["survey"], fixture["pop_params"],
            fixture["gw_pe"], fixture["catalog"],
            fixture["gw_sel"], fixture["catalog"],
            fixture["n_events"], fixture["n_samp"], fixture["Ndraw"],
            singleton_indices=jnp.arange(fixture["n_events"], dtype=jnp.int32),
            pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            n_singletons=fixture["n_events"], n_pairs=0,
            lensed_injections=None,
            pair_kdes=None,
            sis_params=make_sis_lens_params(A_tau=5e-4, n_tau=3.0, T0_seconds=1.0),
            log_p_tag_per_source=jnp.zeros(0),
            pop_model="powerlaw+peak",
            universe_model="spectral_sirens",
            sel_batch_size=None,
            cluster_mode=CLUSTER_MODE_OFF,
            **kw,
        )

    def test_low_event_ess_overspends_the_budget_and_fails(self, fixture):
        diag = self._diag(fixture)
        pe_var = float(diag["pe_variance_sum"])
        assert pe_var > 1.0, (
            "fixture must overspend the default 1.0 total budget through "
            f"EVENT variance alone (got {pe_var})"
        )
        # Selection alone is healthy: its ESS-driven variance is tiny.
        assert float(diag["Neff_singleton"]) > 100.0
        assert not np.isfinite(float(diag["logL_total"])), (
            "the total-variance guard must fail a likelihood dominated by "
            "noisy per-event estimators even when selection ESS is high"
        )

    def test_relaxed_budget_recovers_a_finite_likelihood(self, fixture):
        diag = self._diag(fixture, max_likelihood_variance=100.0)
        assert np.isfinite(float(diag["logL_total"]))
        assert float(diag["singleton_variance_sum"]) > 1.0
        assert float(diag["pair_variance_sum"]) == 0.0
