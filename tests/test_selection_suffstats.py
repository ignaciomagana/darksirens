"""Joint-term cross-check: the binned magnitude likelihood vs the exact one,
and the Gaussian(Laplace)-prior path vs the full joint posterior.

Certifies the plan's cheap default: at survey sample sizes the offline fit's
Laplace posterior IS the magnitude likelihood, so sampling theta under that
Gaussian prior loses nothing against carrying the explicit joint term.
"""

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from scipy.stats import norm  # noqa: E402

from darksirens.redshift.selection import (  # noqa: E402
    fit_selection_from_mags,
    magnitude_loglike_from_stats,
    magnitude_suffstats,
    reference_absolute_mags,
)
from darksirens.utils.cosmology import distance_modulus  # noqa: E402

M_LIM = 22.5


def _sample(rng, n=3000, h0_true=70.0, m0=-21.0, sig=1.0):
    z = rng.uniform(0.05, 0.45, size=6 * n)
    M = rng.normal(m0, sig, size=z.size)
    dm = np.asarray(distance_modulus(jnp.asarray(z), h0_true))
    m = M + dm
    keep = m <= M_LIM
    return m[keep][:n], z[keep][:n]


def _exact_nll(m, z, mu, sig):
    Mhat = reference_absolute_mags(m, z)
    T = M_LIM - (m - Mhat)
    return -np.sum(norm.logpdf((Mhat - mu) / sig) - np.log(sig)
                   - norm.logcdf((T - mu) / sig))


def test_binned_loglike_matches_exact():
    rng = np.random.default_rng(31)
    m, z = _sample(rng)
    stats = magnitude_suffstats(m, z, M_LIM, n_bins=256)
    for mu, sig in ((-20.2, 1.0), (-20.5, 0.8), (-19.9, 1.3)):
        exact = -_exact_nll(m, z, mu, sig)
        binned = float(magnitude_loglike_from_stats(mu, sig, stats))
        # Constant offset (the -0.5 log 2pi per galaxy) is intentional;
        # compare DIFFERENCES, which is all a likelihood term contributes.
        exact0 = -_exact_nll(m, z, -20.2, 1.0)
        binned0 = float(magnitude_loglike_from_stats(-20.2, 1.0, stats))
        assert abs((binned - binned0) - (exact - exact0)) < 0.05, (mu, sig)


def test_gaussian_prior_path_matches_joint_posterior():
    """1-D M0hat posterior (sigma_M at truth): full joint vs Laplace Gaussian."""
    rng = np.random.default_rng(32)
    m, z = _sample(rng, n=4000)
    fit = fit_selection_from_mags(m, z, M_LIM)
    sd = float(np.sqrt(fit.cov[0, 0]))

    grid = np.linspace(fit.M0hat - 6 * sd, fit.M0hat + 6 * sd, 241)
    stats = magnitude_suffstats(m, z, M_LIM, n_bins=256)
    ll = np.array([float(magnitude_loglike_from_stats(mu, fit.sigma_M, stats))
                   for mu in grid])
    post = np.exp(ll - ll.max())
    post /= np.trapz(post, grid)
    mean = np.trapz(grid * post, grid)
    var = np.trapz((grid - mean) ** 2 * post, grid)

    # Joint-posterior mean within a small fraction of the Laplace sd of the
    # MLE, and the widths agree to a few percent: the Gaussian-prior path is
    # certified against the explicit joint term.
    assert abs(mean - fit.M0hat) < 0.2 * sd
    assert abs(np.sqrt(var) / sd - 1.0) < 0.05
