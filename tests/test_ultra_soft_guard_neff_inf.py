"""Soft-guard wall at Neff = +inf: a finite likelihood must keep a finite gradient.

``_lse_to_log_mu_neff`` deliberately returns ``Neff = +inf`` for an
exactly-zero-variance (deterministic / uniform-weight) selection campaign —
the verdict that passes every reliability guard.  The soft-guard branch of
``selection_log_correction`` used to form ``x = Neff / threshold`` directly:
the division's VJP stores the +inf numerator, the underflowed softplus gate
hands it an exactly-zero cotangent, and ``0 * inf = NaN`` flowed into
``threshold`` -> ``pe_variance_sum`` / ``max_likelihood_variance`` -> every
sampled parameter, while the returned log likelihood stayed finite (so the
``isfinite(total)`` collapse never fired).  The hard-guard branch handled the
same verdict correctly.  MEASURED pre-fix: value 10.0 with
d/d pe_variance_sum = nan and d/d max_likelihood_variance = nan.

Regression for the double-where at the ratio: the dead branch divides finite
operands and the +inf is a gradient-free constant.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.likelihood.selection import (
    selection_log_correction,
    selection_reduce_from_ldw_provider,
)


NDRAW = 4.0

# The two variance operands the NaN used to poison; values are the base point.
_VARIANCE_OPERANDS = {"max_likelihood_variance": 1.0, "pe_variance_sum": 0.3}


def _reduce(log_weights):
    """Run the production reduction on a concrete weight vector."""
    return selection_reduce_from_ldw_provider(
        lambda start, size: jax.lax.dynamic_slice_in_dim(log_weights, start, size),
        log_weights.shape[0],
        NDRAW,
        None,
    )


def _correction(v, soft_guard, wrt):
    kwargs = dict(_VARIANCE_OPERANDS)
    kwargs[wrt] = v
    return selection_log_correction(
        jnp.asarray(-1.0),
        jnp.asarray(jnp.inf),
        nEvents=10,
        soft_guard=soft_guard,
        **kwargs,
    )


@pytest.mark.parametrize("wrt", sorted(_VARIANCE_OPERANDS))
def test_soft_guard_gradient_is_finite_at_infinite_neff(wrt):
    """d/d(variance operand) at Neff = inf is finite and matches the hard guard."""
    v = jnp.asarray(_VARIANCE_OPERANDS[wrt])
    soft_value = _correction(v, True, wrt)
    hard_value = _correction(v, False, wrt)
    # Neff = inf passes every guard: both branches are the pure Poisson term.
    assert bool(jnp.isfinite(soft_value))
    assert float(soft_value) == pytest.approx(float(hard_value))

    g_soft = jax.grad(lambda t: _correction(t, True, wrt))(v)
    g_hard = jax.grad(lambda t: _correction(t, False, wrt))(v)
    assert bool(jnp.isfinite(g_soft)), f"soft-guard d/d {wrt} = {g_soft}"
    # The wall is fully disengaged at Neff = inf, so the sensitivity to the
    # variance budget must be the hard guard's: exactly zero.
    np.testing.assert_allclose(np.asarray(g_soft), np.asarray(g_hard), rtol=0, atol=0)


@pytest.mark.parametrize("case", ["uniform", "nearly_uniform"])
def test_soft_guard_full_chain_gradient_is_finite_on_a_uniform_campaign(case):
    """Weights -> reduction -> soft guard with a traced pe_variance_sum.

    Both cases produce the zero-variance Neff = inf verdict (``nearly_uniform``
    through the round-off clamp in ``_lse_to_log_mu_neff``); pre-fix, every
    weight and the variance operand came back NaN at a finite value.
    """
    w = {
        "uniform": jnp.zeros(4),
        "nearly_uniform": jnp.array([0.0, 0.0, 0.0, 1e-12]),
    }[case]

    def correction(x, pe_var):
        log_mu, neff, _ = _reduce(x)
        return selection_log_correction(
            log_mu, neff, nEvents=1, soft_guard=True, pe_variance_sum=pe_var
        )

    pe = jnp.asarray(0.3)
    value = correction(w, pe)
    grads = jax.grad(correction, argnums=(0, 1))(w, pe)
    flat = jnp.concatenate([jnp.ravel(jnp.asarray(g)) for g in grads])
    assert bool(jnp.isfinite(value)), f"{case}: value {value}"
    assert bool(jnp.all(jnp.isfinite(flat))), f"{case}: non-finite grad {grads}"
    # The uniform campaign's analytic gradient: -N_obs * d(log mu)/d(log w_i).
    if case == "uniform":
        np.testing.assert_allclose(
            np.asarray(grads[0]), -0.25 * np.ones(4), rtol=0, atol=1e-12
        )


@pytest.mark.parametrize(
    "neff", [4.0, 40.0], ids=["below_threshold", "above_threshold"]
)
def test_soft_guard_finite_neff_forward_and_gradient_are_unchanged(neff):
    """The double-where must be forward-identical off the inf branch."""
    n = 1
    pe = 0.1
    budget = max(_VARIANCE_OPERANDS["max_likelihood_variance"] - pe, 1e-12)
    threshold = max(5.0 * n, (n * n) / budget)

    def soft(neff_val, pe_var):
        return selection_log_correction(
            jnp.asarray(-1.0),
            neff_val,
            nEvents=n,
            soft_guard=True,
            pe_variance_sum=pe_var,
        )

    # Pre-fix formula, transcribed: x = Neff / threshold with no where.
    x = neff / threshold
    gate = float(jax.nn.softplus(200.0 * (1.0 - x) - 10.0))
    reward_mag = n * float(jax.nn.softplus(1.0))
    wall = -gate * (100.0 + 2.0 * reward_mag)
    reference = n * 1.0 + n * (3.0 + n) / (2.0 * max(neff, threshold)) + wall

    value = soft(jnp.asarray(neff), jnp.asarray(pe))
    assert float(value) == pytest.approx(reference, rel=1e-14)
    grads = jax.grad(soft, argnums=(0, 1))(jnp.asarray(neff), jnp.asarray(pe))
    assert all(bool(jnp.isfinite(g)) for g in grads), f"non-finite grad {grads}"


@pytest.mark.parametrize("soft_guard", [False, True])
def test_nan_neff_still_collapses_to_the_hard_verdict(soft_guard):
    """The double-where shunts a NaN Neff out of x; it must still guard."""
    result = selection_log_correction(
        jnp.asarray(-1.0),
        jnp.asarray(jnp.nan),
        nEvents=10,
        soft_guard=soft_guard,
        pe_variance_sum=0.3,
    )
    assert bool(jnp.isneginf(result))


def test_nan_threshold_still_collapses_to_the_hard_verdict():
    """A NaN budget (NaN pe_variance_sum) must guard, never admit."""
    result = selection_log_correction(
        jnp.asarray(-1.0),
        jnp.asarray(50.0),
        nEvents=10,
        soft_guard=True,
        pe_variance_sum=jnp.asarray(jnp.nan),
    )
    assert bool(jnp.isneginf(result))
