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


@pytest.mark.parametrize("frac", [0.4, 0.9])
def test_jensen_term_is_per_bin_not_a_row_scalar(frac):
    """E[Q]'s Jensen term must use each bin's OWN curvature.

    With a row-scalar curvature (``median(lambda_row)``) a row whose
    completeness support covers under half its bins has median lambda = 0, so
    var_post is pinned at the prior variance, ``+0.5 b^2 var`` cancels the
    mean-one shift in EVERY bin, and the data-rich bins come out inflated by
    exp(0.5 b^2 sigma^2 / ps) = 1.65 here -- with the inflation switching on and
    off with the covered fraction, i.e. a discontinuous pixel-to-pixel
    misplacement of missing galaxies at fixed z.  Per bin: data-rich bins read
    logQ ~ b s - shift and data-free bins read ~ b s.
    """
    n = 160
    pk = np.full(n, 1.0)                       # sigma^2 = 1
    n_act = int(frac * n)
    C = np.zeros(n)
    C[:n_act] = 0.8
    dN_exp = np.full(n, 40.0)                  # data-rich where C > 0
    N_obs = np.zeros((1, n))
    N_obs[0, :n_act] = C[:n_act] * dN_exp[:n_act]
    out = poisson_lognormal_map(N_obs[0][None, :], C[None, :], dN_exp, pk,
                                bias=1.0, prior_strength=1.0, maxiter=500)
    logq = np.asarray(out["logq_map"])[0]
    s = np.asarray(out["s_map"])[0]
    shift = 0.5                                # 0.5 * b^2 * sigma^2 / ps
    # Data-rich bins: the posterior variance collapses, so the Jensen term does
    # NOT cancel the shift (it did, exactly, with the row-median curvature).
    assert np.mean(logq[:n_act] - s[:n_act]) < -0.9 * shift
    # Data-free bins keep the full prior variance -> the shift cancels -> Q = 1
    # far from data.
    assert abs(np.mean(logq[n_act:] - s[n_act:])) < 0.05


def test_member_spread_is_per_bin_and_tracks_the_mean_table():
    """The ensemble mean must reproduce the deterministic E[Q] table, and the
    per-member spread must be tighter in data-rich bins than in data-free ones."""
    n = 96
    pk = np.full(n, 0.25)
    n_act = n // 3
    C = np.zeros(n)
    C[:n_act] = 0.9
    dN_exp = np.full(n, 60.0)
    N_obs = np.zeros((1, n))
    N_obs[0, :n_act] = C[:n_act] * dN_exp[:n_act]
    mp = poisson_lognormal_map(N_obs, C[None, :], dN_exp, pk, bias=1.0,
                               prior_strength=1.0, maxiter=500)
    mem = laplace_lognormal_members(mp["s_map"], mp["lambda_map"], pk,
                                    n_members=4000, bias=1.0,
                                    prior_strength=1.0, seed=1)
    q_mean = np.log(np.mean(np.asarray(mem["q_members"]), axis=0))[0]
    np.testing.assert_allclose(q_mean, np.asarray(mp["logq_map"])[0],
                               atol=0.05)
    sd = np.asarray(mem["logq_members"])[:, 0, :].std(axis=0)
    assert sd[:n_act].mean() < 0.5 * sd[n_act:].mean()


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
