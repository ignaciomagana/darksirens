"""The sparse-selection Neff guard: hard wall for nested samplers, smooth wall
for gradient-based sampling (NumPyro NUTS), resolved per sampler.

The hard -inf at Neff <= 5 N_obs sat 1.5 posterior-sigma from the H1-profile
posterior mean and divergence-flagged 100% of NUTS transitions; the soft
guard replaces it with a steep smooth penalty that is negligible where the
hard guard passes and strongly repulsive where it fails, keeping the
likelihood differentiable everywhere.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from darksirens.likelihood.selection import selection_log_correction
from darksirens.likelihood.factory import _resolve_redshift_prior_materialization


N_EVENTS = 120
# effective guard threshold: max(Vitale 5 N_obs floor, variance criterion
# N_obs^2 / max_likelihood_variance) — at N=120 the variance criterion rules.
THRESHOLD = max(5.0 * N_EVENTS, N_EVENTS**2 / 1.0)


def test_hard_guard_unchanged():
    ll_ok = selection_log_correction(jnp.asarray(0.0), jnp.asarray(10 * THRESHOLD), N_EVENTS)
    ll_bad = selection_log_correction(jnp.asarray(0.0), jnp.asarray(0.5 * THRESHOLD), N_EVENTS)
    assert np.isfinite(float(ll_ok))
    assert float(ll_bad) == -np.inf


def test_soft_guard_matches_hard_when_neff_comfortable():
    for neff_mult in (2.0, 10.0, 100.0):
        neff = jnp.asarray(neff_mult * THRESHOLD)
        hard = float(selection_log_correction(jnp.asarray(0.0), neff, N_EVENTS))
        soft = float(selection_log_correction(jnp.asarray(0.0), neff, N_EVENTS, soft_guard=True))
        # penalty is exp(-15)-scale by 2x threshold — far below any logL scale
        assert abs(hard - soft) < 1e-4, (neff_mult, hard, soft)


def test_soft_guard_finite_and_repulsive_below_threshold():
    log_mu = jnp.asarray(0.0)
    vals = [
        float(selection_log_correction(log_mu, jnp.asarray(x * THRESHOLD), N_EVENTS, soft_guard=True))
        for x in (1.0, 0.75, 0.5, 0.25, 0.05)
    ]
    assert all(np.isfinite(v) for v in vals)
    # monotone decreasing as Neff shrinks below the threshold: the wall
    # repels, never rewards. (ABOVE the threshold the 1/Neff Taylor
    # correction legitimately grows as Neff falls — same as the hard guard —
    # so monotonicity is only a property of the sub-threshold wall.)
    assert all(a > b for a, b in zip(vals, vals[1:])), vals
    # strongly penalised well below the threshold
    assert vals[-1] < vals[0] - 100.0


def test_soft_guard_differentiable_across_threshold():
    def f(neff):
        return selection_log_correction(jnp.asarray(0.0), neff, N_EVENTS, soft_guard=True)

    for x in (2.0, 1.0, 0.9, 0.5, 0.1):
        g = float(jax.grad(f)(jnp.asarray(x * THRESHOLD)))
        assert np.isfinite(g), (x, g)
    # inside the wall the repulsion dominates: gradient pushes toward more Neff
    for x in (0.9, 0.5, 0.1):
        g = float(jax.grad(f)(jnp.asarray(x * THRESHOLD)))
        assert g > 0.0, (x, g)


def test_soft_guard_dominates_unbounded_reward():
    """The retained -N log(mu) term grows without bound as mu -> 0; a
    fixed-height wall saturates and opens a spurious high-likelihood pocket
    in the deep-sparse region (library review, likelihood finding 3). The
    reward-tracking wall must dominate at the review's demonstrated exploit
    points, and crossing from valid Neff into the deep-sparse region at the
    same mu must be strongly penalised."""
    # (log_mu, Neff, nEvents) exploit points from the review:
    v1 = float(selection_log_correction(jnp.asarray(-300.0), jnp.asarray(1.0), 100, soft_guard=True))
    assert v1 < -1e5, v1
    v2 = float(selection_log_correction(jnp.asarray(-50.0), jnp.asarray(10.0), 40, soft_guard=True))
    assert v2 < -1e4, v2
    # no incentive to cross below threshold at fixed mu:
    for log_mu in (-300.0, -50.0, -5.0):
        above = float(selection_log_correction(
            jnp.asarray(log_mu), jnp.asarray(1.2 * THRESHOLD), N_EVENTS, soft_guard=True))
        deep = float(selection_log_correction(
            jnp.asarray(log_mu), jnp.asarray(0.5 * THRESHOLD), N_EVENTS, soft_guard=True))
        assert deep < above - 1e3, (log_mu, above, deep)


def test_barrier_resolver_disables_for_numpyro():
    # lax.optimization_barrier has no differentiation rule; NUTS on the
    # catalog (dark/bright) paths died with NotImplementedError until the
    # auto mode dropped the barrier for numpyro.
    opts = SimpleNamespace(redshift_prior_barrier="auto", sampler="numpyro")
    assert _resolve_redshift_prior_materialization(opts) is False
    opts = SimpleNamespace(redshift_prior_barrier="auto", sampler="dynesty")
    assert _resolve_redshift_prior_materialization(opts) is True
    opts = SimpleNamespace(redshift_prior_barrier="on", sampler="numpyro")
    assert _resolve_redshift_prior_materialization(opts) is True
