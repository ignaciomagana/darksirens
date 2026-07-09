"""Stick-breaking math for the K-catalog multitracer mixture.

The mixture weights w_1..w_K are built from Beta(1, K-j) "stick" draws
v_1..v_{K-1} (the sampled labels fcat_2..fcat_K) via the uniform-Dirichlet(1,
..., 1) construction implemented by ``_sticks_to_log_weights``
(darksirens/inference/parameters.py) and the closed-form Beta(1, b) PPF
branch of ``make_prior_transform`` (darksirens/inference/prior.py).  See
MT_IMPLEMENTATION_SPEC.md, "Math".
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import stats

from darksirens.inference.parameters import _sticks_to_log_weights
from darksirens.inference.prior import make_prior_transform


# ---------------------------------------------------------------------------
# Closed forms for K = 2, 3, 4
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("K", [2, 3, 4])
def test_sticks_to_log_weights_sum_to_one(K):
    rng = np.random.default_rng(1000 + K)
    v = jnp.asarray(rng.uniform(0.05, 0.95, size=K - 1))
    w = np.asarray(jnp.exp(_sticks_to_log_weights(v)))
    assert w.shape == (K,)
    assert np.all(w >= 0.0)
    assert abs(w.sum() - 1.0) < 1e-10


def test_sticks_to_log_weights_k2_exact():
    """K=2: w = (1 - fcat_2, fcat_2) exactly (a single stick v_1 = fcat_2)."""
    for fcat_2 in (0.0, 0.01, 0.37, 0.5, 0.99, 1.0):
        v = jnp.asarray([fcat_2])
        w = np.asarray(jnp.exp(_sticks_to_log_weights(v)))
        np.testing.assert_allclose(w, [1.0 - fcat_2, fcat_2], atol=1e-12)


@pytest.mark.parametrize("K", [2, 3, 4])
def test_sticks_to_log_weights_boundary_sticks_finite_no_nan(K):
    """A boundary stick (0 or 1) drives some catalogs to exactly zero weight
    (log w = -inf) but must never produce NaN, and the -inf entries must land
    on the catalogs the stick-breaking construction predicts."""
    base = np.linspace(0.2, 0.8, K - 1)
    for j in range(K - 1):
        for edge in (0.0, 1.0):
            v_np = base.copy()
            v_np[j] = edge
            v = jnp.asarray(v_np)
            log_w = np.asarray(_sticks_to_log_weights(v))
            assert not np.any(np.isnan(log_w)), (K, j, edge, log_w)

            if edge == 0.0:
                # v_j = 0: catalog (j + 2) gets exactly zero weight (log w = -inf);
                # nothing downstream of j is forced to zero by THIS stick alone.
                assert log_w[j + 1] == -np.inf
            else:
                # v_j = 1: stick j takes everything that remained, so catalog 1
                # (the remainder) and every catalog AFTER j + 2 collapse to -inf;
                # catalog (j + 2) itself gets the entire remaining mass.
                assert log_w[0] == -np.inf
                assert np.isfinite(log_w[j + 1])
                for m in range(j + 2, K):
                    assert log_w[m] == -np.inf


# ---------------------------------------------------------------------------
# Beta(1, b) closed-form PPF (make_prior_transform's "beta" branch)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("b", [0.5, 1.0, 1.7, 2.3, 5.0])
def test_beta_ppf_matches_scipy(b):
    u = np.linspace(0.02, 0.98, 33)
    transform = make_prior_transform([0.0], [1.0], [("beta", 1.0, b)])
    x = np.asarray(transform(jnp.asarray(u)[:, None]))[:, 0]
    x_scipy = stats.beta.ppf(u, 1.0, b)
    np.testing.assert_allclose(x, x_scipy, atol=1e-9, rtol=1e-9)


@pytest.mark.parametrize("b", [0.5, 1.0, 1.7, 2.3, 5.0])
def test_beta_ppf_round_trips_through_analytic_cdf(b):
    """F(x) = 1 - (1 - x)^b is the Beta(1, b) CDF; round-tripping the PPF
    through it must recover u (self-consistency of the closed-form inverse)."""
    u = np.linspace(0.001, 0.999, 41)
    transform = make_prior_transform([0.0], [1.0], [("beta", 1.0, b)])
    x = np.asarray(transform(jnp.asarray(u)[:, None]))[:, 0]
    F = 1.0 - (1.0 - x) ** b
    np.testing.assert_allclose(F, u, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Uniform-simplex Monte Carlo check (K = 3): E[w_k] = 1/K, Var[w_k] = the
# Dirichlet(1,...,1) value alpha_0 = K => Var = (K-1)/(K^2 (K+1)).
# ---------------------------------------------------------------------------

def test_uniform_simplex_monte_carlo_k3():
    K = 3
    N = 40_000
    rng = np.random.default_rng(42)
    u = rng.uniform(0.0, 1.0, size=(N, K - 1))

    # fcat_m ~ Beta(1, K - m + 1): fcat_2 -> b=2, fcat_3 -> b=1.
    prior_kinds = [("beta", 1.0, 2.0), ("beta", 1.0, 1.0)]
    transform = make_prior_transform([0.0, 0.0], [1.0, 1.0], prior_kinds)
    v = np.asarray(transform(jnp.asarray(u)))
    assert v.shape == (N, K - 1)

    log_w = jax.vmap(_sticks_to_log_weights)(jnp.asarray(v))
    w = np.asarray(jnp.exp(log_w))
    assert w.shape == (N, K)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)

    mean_w = w.mean(axis=0)
    var_w = w.var(axis=0)

    expected_mean = 1.0 / K
    expected_var = (K - 1) / (K ** 2 * (K + 1))  # = 1/18 for K=3
    assert expected_var == pytest.approx(1.0 / 18.0)

    se_mean = np.sqrt(expected_var / N)
    for k in range(K):
        assert abs(mean_w[k] - expected_mean) < 3.0 * se_mean, (k, mean_w[k])
        assert abs(var_w[k] - expected_var) < 0.2 * expected_var, (k, var_w[k])
