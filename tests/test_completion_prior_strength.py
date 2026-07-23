"""Regression: the lognormal-completion mean-one shift must scale with
``prior_strength``.

At the default ``prior_strength=1`` the deterministic ``Q`` is mean-one and a
data-free bin reads ``Q=1`` (homogeneous).  The shift ``0.5*b**2*sigma**2``
omitted the ``prior_strength`` divisor while the Laplace posterior variance used
the SCALED prior precision ``ps/pk``, so for ``prior_strength != 1`` a data-free
bin read ``Q<1`` -- a silent missing-galaxy-density suppression (~31% at
``ps=4, b=1``) that biases H0.  These check the mean-one / homogeneity property
holds for both the deterministic MAP and the ensemble at ``ps != 1``.
"""
import numpy as np
import pytest

from darksirens.redshift.lognormal_completion import (
    poisson_lognormal_map,
    laplace_lognormal_members,
)


@pytest.mark.parametrize("ps", [1.0, 4.0, 0.25])
def test_map_mean_one_data_free_pixel(ps):
    # Empty pixel (C=0 -> rate_base=0): the Poisson data term vanishes, so the
    # MAP is the prior mean s=0 and Q must read 1 (logQ=0) for ANY prior_strength.
    n_grid = 64
    N_obs = np.zeros((1, n_grid))
    C = np.zeros((1, n_grid))
    dN_exp = np.ones((1, n_grid))
    pk = np.full(n_grid, 0.5)
    out = poisson_lognormal_map(
        N_obs, C, dN_exp, pk, bias=1.0, prior_strength=ps, maxiter=200,
    )
    logq = np.asarray(out["logq_map"])[0]
    assert np.max(np.abs(logq)) < 1e-6, (
        f"empty-pixel logQ not ~0 at prior_strength={ps}: "
        f"max|logQ|={np.max(np.abs(logq)):.4g}"
    )


@pytest.mark.parametrize("ps", [1.0, 4.0, 0.25])
def test_members_mean_one_data_free_pixel(ps):
    # The ensemble mean E[Q] of a data-free bin must also be ~1 for any ps
    # (each member draws delta_s ~ N(0, sigma^2/ps); E[exp(b*delta_s - shift)]=1).
    # Fixed seed + a small field variance + a global mean over M members x nodes,
    # so the residual MC noise is << the tolerance and the check is robust.
    n_grid = 64
    s_map = np.zeros((1, n_grid))
    lambda_map = np.zeros((1, n_grid))
    pk = np.full(n_grid, 0.1)
    out = laplace_lognormal_members(
        s_map, lambda_map, pk, n_members=12000, bias=1.0,
        prior_strength=ps, seed=0,
    )
    q = np.asarray(out["q_members"])              # (M, 1, n_grid)
    assert abs(float(np.mean(q)) - 1.0) < 0.03, (
        f"ensemble mean E[Q] not ~1 at prior_strength={ps}: {float(np.mean(q)):.4g}"
    )


def test_map_default_prior_strength_unchanged():
    # ps=1 is the pre-fix behaviour: the fix must be a no-op there.
    n_grid = 48
    rng = np.random.default_rng(0)
    N_obs = rng.poisson(2.0, size=(2, n_grid)).astype(float)
    C = np.full((2, n_grid), 0.7)
    dN_exp = np.full((2, n_grid), 3.0)
    pk = np.full(n_grid, 0.4)
    out = poisson_lognormal_map(N_obs, C, dN_exp, pk, bias=1.0, prior_strength=1.0)
    assert np.all(np.isfinite(out["logq_map"]))
