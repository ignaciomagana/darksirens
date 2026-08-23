"""Selection-integral estimators must be differentiable everywhere they are finite.

The Monte-Carlo variance of the selection integral used to be formed in
log-space as ``logdiffexp(log_s2, 2*log_mu - log_Ndraw)``.  Uniform importance
weights -- a deterministic or near-deterministic selection campaign -- make
those two operands EQUAL.  The forward value is then correctly ``-inf`` (zero
variance), but the derivative runs through ``log1p(-exp(0)) = log1p(-1)``,
whose slope is ``1/0``; reverse-mode AD has already evaluated that branch by
the time the downstream ``jnp.where`` picks the other one, and a ``where``
cannot erase a NaN cotangent.  MEASURED on master: four equal log weights gave
the correct statistics ``(0, inf, -inf)`` with an all-NaN Jacobian, and
``selection_log_correction`` returned ``0.0`` with gradient
``[nan, nan, nan, nan]`` -- a finite log likelihood with unusable gradients,
which silently destroys NUTS/HMC and every gradient optimizer.

The same class of failure lives at the other end of the estimator: an
all-invalid campaign reduces through ``logsumexp`` of an all--inf array, whose
softmax is 0/0 = NaN, and that NaN survives multiplication by a ZERO upstream
cotangent.

Both the singleton estimator (``likelihood/selection.py``) and the two cluster
estimators (``likelihood/cluster_selection.py``) are covered here.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.likelihood.selection import (
    _lse_to_log_mu_neff,
    selection_log_correction,
    selection_reduce_from_ldw_provider,
)
from darksirens.utils.utils import logsumexp_neginf_safe


NDRAW = 4.0

# Every weight vector a selection campaign can realistically produce.
WEIGHT_CASES = {
    # Exactly uniform: Var(mu_hat) is EXACTLY zero -> the logdiffexp singularity.
    "uniform": jnp.zeros(4),
    # Numerically uniform: the subtraction rounds to zero at double precision.
    "nearly_uniform": jnp.array([0.0, 0.0, 0.0, 1e-12]),
    # Every injection masked out: log_mu = -inf, the all--inf logsumexp.
    "all_invalid": jnp.full(4, -jnp.inf),
    # Partially masked: the ordinary production shape.
    "half_invalid": jnp.array([0.0, -jnp.inf, -jnp.inf, 0.0]),
    # Ordinary spread-out weights: the reference case, must be untouched.
    "ordinary": jnp.array([0.0, 1.0, -0.5, 2.0]),
}


def _reduce(log_weights, sel_batch_size=None):
    """Run the production reduction on a concrete weight vector."""
    return selection_reduce_from_ldw_provider(
        lambda start, size: jax.lax.dynamic_slice_in_dim(log_weights, start, size),
        log_weights.shape[0],
        NDRAW,
        sel_batch_size,
    )


@pytest.mark.parametrize("case", sorted(WEIGHT_CASES))
@pytest.mark.parametrize("sel_batch_size", [None, 2])
def test_selection_reduction_jacobian_is_finite(case, sel_batch_size):
    """jacrev of (log_mu, Neff, log_sigma2) never contains a NaN."""
    w = WEIGHT_CASES[case]
    jac = jax.jacrev(lambda x: jnp.stack(_reduce(x, sel_batch_size)))(w)
    assert not bool(jnp.any(jnp.isnan(jac))), f"{case}: NaN in Jacobian\n{jac}"


@pytest.mark.parametrize("case", sorted(WEIGHT_CASES))
def test_estimator_jacobian_is_finite(case):
    """The same, one level down: the raw (lse, lse2) -> statistics converter."""
    w = WEIGHT_CASES[case]

    def statistics(x):
        lse = logsumexp_neginf_safe(x)
        lse2 = logsumexp_neginf_safe(2.0 * x)
        return jnp.stack(_lse_to_log_mu_neff(lse, lse2, NDRAW))

    jac = jax.jacrev(statistics)(w)
    assert not bool(jnp.any(jnp.isnan(jac))), f"{case}: NaN in Jacobian\n{jac}"


@pytest.mark.parametrize("case", sorted(WEIGHT_CASES))
def test_selection_log_correction_gradient_is_finite(case):
    """jax.grad through the full public correction, the NUTS-facing path."""
    w = WEIGHT_CASES[case]

    def correction(x):
        log_mu, neff, _ = _reduce(x)
        return selection_log_correction(log_mu, neff, nEvents=1)

    value = correction(w)
    grad = jax.grad(correction)(w)
    assert not bool(jnp.any(jnp.isnan(grad))), f"{case}: NaN gradient {grad}"
    if bool(jnp.isfinite(value)):
        # A finite log likelihood must come with a usable gradient, not just a
        # non-NaN one: this is the property master violated.
        assert bool(jnp.all(jnp.isfinite(grad))), f"{case}: non-finite grad {grad}"


def test_uniform_weights_are_a_perfect_estimate_not_a_sparse_one():
    """Zero variance means INFINITE effective sample size, so the guard passes."""
    log_mu, neff, log_sigma2 = _reduce(WEIGHT_CASES["uniform"])
    assert float(log_mu) == pytest.approx(0.0)          # mu = 4/4 = 1
    assert bool(jnp.isposinf(neff))
    assert bool(jnp.isneginf(log_sigma2))
    # ... and the correction is the pure Poisson term, with no Farr inflation.
    assert float(selection_log_correction(log_mu, neff, nEvents=1)) == pytest.approx(0.0)


def test_uniform_weights_gradient_is_the_analytic_one():
    """-N_obs * d(log mu)/d(log w_i) = -1/4 for four equal weights."""

    def correction(x):
        log_mu, neff, _ = _reduce(x)
        return selection_log_correction(log_mu, neff, nEvents=1)

    grad = jax.grad(correction)(WEIGHT_CASES["uniform"])
    np.testing.assert_allclose(np.asarray(grad), -0.25 * np.ones(4), rtol=0, atol=1e-12)


def test_ordinary_weights_statistics_are_unchanged():
    """The rewritten estimator must reproduce the log-space formula it replaced."""
    w = WEIGHT_CASES["ordinary"]
    log_mu, neff, log_sigma2 = _reduce(w)
    s1 = float(jnp.sum(jnp.exp(w)))
    s2 = float(jnp.sum(jnp.exp(2.0 * w)))
    mu_ref = s1 / NDRAW
    sigma2_ref = s2 / NDRAW**2 - s1**2 / NDRAW**3
    assert float(log_mu) == pytest.approx(np.log(mu_ref), rel=1e-14)
    assert float(log_sigma2) == pytest.approx(np.log(sigma2_ref), rel=1e-13)
    assert float(neff) == pytest.approx(mu_ref**2 / sigma2_ref, rel=1e-13)


def test_all_invalid_still_rejects():
    """The NaN-free path must not have turned an empty campaign into a live one."""
    log_mu, neff, log_sigma2 = _reduce(WEIGHT_CASES["all_invalid"])
    assert bool(jnp.isneginf(log_mu))
    assert float(neff) == 0.0
    assert bool(jnp.isneginf(log_sigma2))
    assert bool(jnp.isneginf(selection_log_correction(log_mu, neff, nEvents=1)))


def test_nan_weight_is_not_silently_dropped():
    """A poisoned weight must still reach the guard, not be sanitized away."""
    w = jnp.array([0.0, jnp.nan, 0.0, 0.0])
    log_mu, neff, _ = _reduce(w)
    assert bool(jnp.isnan(log_mu))
    assert float(neff) == 0.0
    assert bool(jnp.isneginf(selection_log_correction(log_mu, neff, nEvents=1)))


# ---------------------------------------------------------------------------
# Cluster / lensed-singleton estimators (likelihood/cluster_selection.py)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(WEIGHT_CASES))
def test_cluster_estimator_reduction_jacobian_is_finite(case):
    """Both cluster estimators reduce through the same converter; pin its tail.

    ``compute_cluster_selection_term`` and
    ``compute_lensed_single_selection_term`` differ only in how they build
    ``log_w``; everything downstream of ``log_w`` is this reduction, and it is
    where the zero-variance/empty-channel NaNs lived.
    """
    from darksirens.likelihood.cluster_selection import _neff_from_log_mu_sigma2

    w = WEIGHT_CASES[case]
    n_draw = jnp.asarray(NDRAW)

    def reduce_cluster(x):
        log_sum_w = logsumexp_neginf_safe(x)
        log_sum_w2 = logsumexp_neginf_safe(2.0 * x)
        return jnp.stack(_lse_to_log_mu_neff(log_sum_w, log_sum_w2, n_draw))

    jac = jax.jacrev(reduce_cluster)(w)
    assert not bool(jnp.any(jnp.isnan(jac))), f"{case}: NaN in Jacobian\n{jac}"

    # And the verdicts the combiner reads are the ones it was written against:
    # the reduction's Neff still equals the cluster stack's own
    # (log_mu, log_sigma2) -> Neff map, which the combiner reuses for the
    # channel-summed total.
    log_mu, neff, log_sigma2 = reduce_cluster(w)
    reference = _neff_from_log_mu_sigma2(log_mu, log_sigma2)
    if bool(jnp.isfinite(neff)):
        assert float(neff) == pytest.approx(float(reference), rel=1e-12)
    else:
        assert float(neff) == float(reference)


@pytest.mark.parametrize("scale", [0.0, 1e-12])
def test_cluster_selection_term_gradient_is_finite_on_a_uniform_campaign(
    monkeypatch, scale
):
    """End-to-end through ``compute_cluster_selection_term`` itself.

    ``scale = 0`` is the exactly-uniform campaign, ``scale = 1e-12`` the
    numerically-uniform one; both hit the zero-variance singularity.
    """
    import darksirens.likelihood.cluster_selection as cs

    n = 4

    class _Injections:
        source_id_kept = jnp.zeros(n, dtype=jnp.int32)
        n_draw_sources = float(n)

    # Bypass the lensing physics: hand the estimator per-source log weights
    # that are (numerically) uniform but still depend on a differentiable
    # parameter, so the gradient below is the estimator's own, not a constant.
    def _uniform_log_weight(injections, cosmo, survey, pop_params, *args, **kwargs):
        return scale * pop_params[0] * jnp.arange(n, dtype=jnp.float64)

    monkeypatch.setattr(cs, "_per_source_log_weight", _uniform_log_weight)

    def term(pop_params):
        log_mu, neff, _ = cs.compute_cluster_selection_term(
            _Injections(), None, None, pop_params, None, None, None, None,
        )
        return selection_log_correction(log_mu, neff, nEvents=1)

    value = term(jnp.array([1.0]))
    grad = jax.grad(term)(jnp.array([1.0]))
    assert bool(jnp.isfinite(value))
    assert not bool(jnp.any(jnp.isnan(grad))), f"NaN gradient {grad}"
