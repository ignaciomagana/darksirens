"""Wasted-work regressions in the Poisson-lognormal completion builder.

Two shapes of duplicated arithmetic, both fixed WITHOUT moving a single bit of
the output:

* ``_map_solve_row`` handed L-BFGS-B a separate ``fun`` and ``jac``.  scipy's
  ``ScalarFunction`` updates the two independently, so each of the ~10k
  evaluations per row ran the ``exp`` twice and a length-``n_grid``
  ``np.fft.fft(s)`` twice at the SAME ``x``.  The objective is now fused
  (``jac=True``).
* ``_laplace_diag_variance`` evaluated a function of the SCALAR ``lambda[r, v]``
  by materialising a ``(chunk, n_grid, n_grid)`` cube, including for every bin
  whose ``lambda`` is exactly zero -- the whole grid above the survey depth plus
  the wrap pad, typically a fifth of the table, all of which share one value.

Both are bit-identity claims, so the tests below compare against the previous
implementations transcribed verbatim rather than against a tolerance.
"""
import time

import numpy as np
import pytest

from darksirens.redshift.lognormal_completion import (
    _laplace_diag_variance,
    _map_solve_row,
    gaussian_correlation_spectrum,
    poisson_lognormal_map,
)


# ---------------------------------------------------------------------------
# Previous implementations, transcribed verbatim as A/B references
# ---------------------------------------------------------------------------

def _cube_diag_variance(lam, pk, bias, prior_strength):
    """The superseded ``(chunk, n_grid, n_grid)`` form of the same quantity."""
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    pk = np.asarray(pk, dtype=float)
    n_rows, n_grid = lam.shape
    A = float(prior_strength) / pk
    b2 = float(bias) * float(bias)
    out = np.empty((n_rows, n_grid), dtype=float)
    chunk = max(1, int(8_000_000 // max(n_grid * n_grid, 1)))
    for start in range(0, n_rows, chunk):
        blk = lam[start:start + chunk]
        H = A[None, None, :] + b2 * blk[:, :, None]
        out[start:start + chunk] = np.mean(1.0 / np.maximum(H, 1e-30), axis=-1)
    return out


def _split_fun_jac_solve_row(nobs, c_row, dN_exp_row, pk, bias, prior_strength,
                             shift, maxiter):
    """The superseded separate-``fun``/``jac`` MAP row solve."""
    from scipy import optimize

    n_grid = int(pk.size)
    b, ps = float(bias), float(prior_strength)
    rate_base = c_row * dN_exp_row
    mask = rate_base > 0.0
    log_rate = np.where(mask, np.log(np.where(mask, rate_base, 1.0)), 0.0)

    def _neg_log_post(s):
        log_lam = log_rate + (b * s - shift)
        lam = np.where(mask, np.exp(log_lam), 0.0)
        data = np.sum(lam - nobs * np.where(mask, log_lam, 0.0))
        sk = np.fft.fft(s)
        prior = 0.5 * ps * np.sum((np.abs(sk) ** 2) / pk) / n_grid
        return float(data + prior)

    def _grad(s):
        log_lam = log_rate + (b * s - shift)
        lam = np.where(mask, np.exp(log_lam), 0.0)
        g_data = np.where(mask, b * (lam - nobs), 0.0)
        g_prior = ps * np.real(np.fft.ifft(np.fft.fft(s) / pk))
        return g_data + g_prior

    res = optimize.minimize(
        _neg_log_post, np.zeros(n_grid), jac=_grad, method="L-BFGS-B",
        options={"maxiter": int(maxiter), "maxfun": 21 * int(maxiter)},
    )
    lam = np.where(mask, np.exp(log_rate + (b * res.x - shift)), 0.0)
    return res.x, lam, bool(res.success)


def _sparse_row(n_grid, seed):
    """A pixel shaped like a depth-cut radial build: sparse counts, and a
    completeness that switches off well before the end of the grid."""
    rng = np.random.default_rng(seed)
    zc = np.linspace(0.0, 1.0, n_grid)
    dN = 0.01 * (1e3 * zc ** 2 + 1.0)
    c = np.clip(1.2 - 1.4 * zc, 0.0, 1.0)
    c[int(0.82 * n_grid):] = 0.0
    delta = 1.0 + 0.3 * np.sin(6.0 * np.pi * zc + rng.uniform(0.0, 6.28))
    return rng.poisson(np.maximum(c * dN * delta, 0.0)).astype(float), c, dN


# ---------------------------------------------------------------------------
# Fused value+gradient (rank 9)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_grid", [64, 179])
def test_fused_objective_is_bit_identical_to_separate_fun_and_jac(n_grid):
    """Fusing changes only HOW MANY TIMES each operation runs, so the whole
    L-BFGS-B iterate path -- not merely the answer to a tolerance -- must be
    reproduced exactly."""
    pytest.importorskip("scipy")
    pk = gaussian_correlation_spectrum(n_grid, 8.0, 0.8)
    for seed in range(3):
        nobs, c_row, dN = _sparse_row(n_grid, seed)
        x_ref, lam_ref, ok_ref = _split_fun_jac_solve_row(
            nobs, c_row, dN, pk, 1.0, 1.0, 0.5, 20000)
        x_new, lam_new, ok_new, _ = _map_solve_row(
            nobs, c_row, dN, pk, 1.0, 1.0, 0.5, 20000)
        assert ok_new == ok_ref
        assert np.array_equal(x_new, x_ref), (
            f"seed {seed}: max|delta s| = {np.max(np.abs(x_new - x_ref)):.3e}; "
            "the fused objective must not move the iterate path"
        )
        assert np.array_equal(lam_new, lam_ref)


def test_map_solve_passes_a_fused_objective_to_lbfgsb(monkeypatch):
    """Regression guard: with a separate ``jac`` callable scipy re-evaluates the
    exp and the FFT at every x, which is what this fix removed."""
    so = pytest.importorskip("scipy.optimize")

    seen = []
    real_minimize = so.minimize

    def spy(fun, x0, **kwargs):
        seen.append(kwargs.get("jac"))
        return real_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(so, "minimize", spy)
    n_grid = 32
    nobs, c_row, dN = _sparse_row(n_grid, 0)
    poisson_lognormal_map(nobs, c_row, dN,
                          gaussian_correlation_spectrum(n_grid, 4.0, 0.8),
                          maxiter=50)
    assert seen and all(j is True for j in seen), (
        f"expected jac=True (fused value+gradient), saw {seen}"
    )


# ---------------------------------------------------------------------------
# Per-bin Laplace variance (rank 13)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_grid", [37, 179, 256])
@pytest.mark.parametrize("bias,prior_strength", [(1.0, 1.0), (1.7, 4.0)])
def test_diag_variance_matches_the_dense_cube_bit_for_bit(n_grid, bias,
                                                          prior_strength):
    rng = np.random.default_rng(n_grid)
    lam = rng.gamma(2.0, 50.0, size=(64, n_grid))
    lam[:, int(0.82 * n_grid):] = 0.0          # above the survey depth / wrap pad
    pk = gaussian_correlation_spectrum(n_grid, 8.0, 0.9)
    ref = _cube_diag_variance(lam, pk, bias, prior_strength)
    got = _laplace_diag_variance(lam, pk, bias, prior_strength)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref), (
        f"max|diff| = {np.max(np.abs(got - ref)):.3e}"
    )


@pytest.mark.parametrize("lam", [
    np.zeros((5, 48)),                                   # every bin data-free
    np.full((5, 48), 3.0),                               # every bin occupied
    np.linspace(0.0, 10.0, 48),                          # 1-D row, one zero bin
])
def test_diag_variance_edge_shapes_match_the_dense_cube(lam):
    pk = gaussian_correlation_spectrum(48, 6.0, 1.1)
    ref = _cube_diag_variance(lam, pk, 1.3, 2.0)
    got = _laplace_diag_variance(lam, pk, 1.3, 2.0)
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def test_data_free_bins_read_the_effective_prior_variance():
    """The lambda = 0 answer is a single scalar shared by every empty bin --
    that is what lets the fast path skip them, and it is also the value that
    makes E[Q] -> 1 above the survey depth."""
    n_grid = 96
    pk = gaussian_correlation_spectrum(n_grid, 6.0, 1.0)
    ps, b = 4.0, 1.7
    lam = np.zeros((3, n_grid))
    lam[:, :10] = 5.0
    got = _laplace_diag_variance(lam, pk, b, ps)
    expected = float(np.mean(1.0 / np.maximum(ps / pk, 1e-30)))
    assert np.all(got[:, 10:] == expected)
    # sanity: the effective prior variance is sigma_s^2 / prior_strength, and
    # occupied bins are tighter than it.
    assert expected == pytest.approx(float(np.mean(pk)) / ps, rel=0.05)
    assert np.all(got[:, :10] < expected)


def test_data_free_bins_are_not_evaluated_one_by_one():
    """Perf guard.  A table that is entirely above the survey depth costs one
    scalar; the superseded cube form spent O(n_rows * n_grid^2) on it (measured
    ~23 s for this shape, against the ~1 s budget below)."""
    n_rows, n_grid = 4000, 1000
    pk = gaussian_correlation_spectrum(n_grid, 20.0, 1.0)
    lam = np.zeros((n_rows, n_grid))
    t0 = time.perf_counter()
    out = _laplace_diag_variance(lam, pk, 1.0, 1.0)
    elapsed = time.perf_counter() - t0
    assert out.shape == (n_rows, n_grid)
    assert elapsed < 1.0, f"empty bins are being evaluated per-bin ({elapsed:.2f} s)"
