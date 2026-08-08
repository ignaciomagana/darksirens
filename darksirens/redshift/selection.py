"""Parametric magnitude-limited selection functions and their offline fit.

The differential completeness of a magnitude-limited survey is a property of
the SELECTION, not of the galaxy field: conditioned on detection at redshift
``z``, the apparent-magnitude distribution is the truncated luminosity
function regardless of how clustered the galaxies are (the thinning theorem).
This module provides

* analytic selection curves ``C_sel(z; theta)`` for a Gaussian and a Schechter
  luminosity function (JAX, evaluated in-likelihood by
  ``darksirens.redshift.completion`` under ``c_mode="selection"``), and
* an OFFLINE maximum-likelihood fit of ``theta`` from the survey's per-galaxy
  apparent magnitudes (numpy/scipy; consumed by ``darksirens_fit_selection``),
  whose Laplace covariance becomes the Gaussian prior on the sampled ``theta``.

h-scaling convention (the H0 firewall).  Absolute magnitudes are carried as
``M0hat = M0 - 5 log10 h`` (h = H0/100).  Because the tabulated luminosity
distance is EXACTLY proportional to 1/H0 (:func:`darksirens.utils.cosmology.
distance_modulus`), the combination ``M0 + DM(z)`` = ``M0hat + DM(z; H0=100)``
is H0-independent, so a selection curve built from ``m_lim - M0 - DM(z)``
carries no H0 information: magnitudes constrain the LF shape and the
completeness budget, never the Hubble constant.  The offline fit likewise
works in reference absolute magnitudes ``Mhat_i = m_i - DM(z_i; H0=100)``,
which equal ``M0hat + scatter`` independent of the (unknown) true H0.

Strata.  Real compilations have direction-dependent depth; the fit accepts a
stratum label per galaxy and returns one ``theta`` per stratum sharing the LF
shape convention.  The mock program uses a single stratum.

Not covered here (documented limits): photometric-error convolution of the
magnitude likelihood, K-corrections, extinction -- real-catalog ingestion
concerns layered on top of these primitives.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaincc, ndtr

from darksirens.utils.cosmology import (
    H0Planck,
    Om0Planck,
    distance_modulus,
    w0Fiducial,
    waFiducial,
)

#: Reference Hubble constant of the h-scaled magnitude convention.
H0_REF = 100.0


def m0_absolute(M0hat, H0):
    """Absolute magnitude from its h-scaled form: ``M0 = M0hat + 5 log10 h``."""
    return M0hat + 5.0 * jnp.log10(H0 / H0_REF)


def c_sel_gaussian(z, m_lim, M0hat, sigma_M, H0, Om0=Om0Planck,
                   w0=w0Fiducial, wa=waFiducial):
    """Gaussian-LF selection ``P(m <= m_lim | z) = Phi((m_lim - M0 - DM)/sigma)``.

    ``M0hat`` is h-scaled; the ``+5 log10 h`` restored here cancels the
    ``-5 log10 h`` inside ``DM`` exactly, so the returned curve is
    H0-invariant to float precision (pinned by
    tests/test_selection_function.py).  At ``z -> 0`` the modulus diverges
    negatively and ``C -> 1``: a magnitude limit misses nothing nearby.
    """
    dm = distance_modulus(z, H0, Om0, w0, wa)
    M0 = m0_absolute(M0hat, H0)
    return ndtr((m_lim - M0 - dm) / sigma_M)


def c_sel_schechter(z, m_lim, Mstar_hat, alpha, M_faint_offset, H0,
                    Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """Schechter-LF selection via regularized upper incomplete gamma ratios.

    ``C_sel(z) = Gamma(alpha+1, x_lim(z)) / Gamma(alpha+1, x_faint)`` with
    ``x = L/L* = 10^{-0.4 (M - M*)}``, ``M_lim(z) = m_lim - DM(z)`` and a
    faint-end integration cutoff ``M_faint = M* + M_faint_offset`` (the
    Schechter integrand diverges faint-ward for ``alpha <= -1``; the cutoff is
    part of the model and must match the population the completion assumes).
    ``Mstar_hat`` is h-scaled like ``M0hat``.  Requires ``alpha > -1`` for the
    unregularized-at-zero ratio to stay finite at bright limits; the
    regularized form used here is finite for ``alpha + 1 > 0`` arguments and
    clipped into [0, 1].  API pinned for real catalogs; the mock program
    exercises the Gaussian family.
    """
    dm = distance_modulus(z, H0, Om0, w0, wa)
    Mstar = m0_absolute(Mstar_hat, H0)
    x_lim = 10.0 ** (-0.4 * (m_lim - dm - Mstar))
    x_faint = 10.0 ** (-0.4 * M_faint_offset)
    num = gammaincc(alpha + 1.0, x_lim)
    den = gammaincc(alpha + 1.0, x_faint)
    return jnp.clip(num / jnp.maximum(den, 1e-300), 0.0, 1.0)


def reference_absolute_mags(m, z, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial):
    """``Mhat_i = m_i - DM(z_i; H0=100)`` -- h-scaled absolute magnitudes.

    Independent of the true H0: with ``m = M0 + scatter + DM(z; H0_true)``,
    ``Mhat = M0 - 5 log10 h_true + scatter = M0hat + scatter`` exactly.
    """
    dm = np.asarray(distance_modulus(jnp.asarray(z, dtype=float), H0_REF,
                                     Om0, w0, wa))
    return np.asarray(m, dtype=float) - dm


def magnitude_suffstats(m, z, m_lim, n_bins=64, Om0=Om0Planck,
                        w0=w0Fiducial, wa=waFiducial):
    """Compress the magnitude sample into per-truncation-bin statistics.

    The truncated-Gaussian likelihood has a PER-GALAXY truncation
    ``T_i = m_lim - DM(z_i; H0=100)``, so the naive (N, sum x, sum x^2)
    reduction is not exact.  Binning galaxies by T restores a finite
    representation: within a bin the truncation is approximated by the
    bin-mean T, and the joint likelihood becomes a sum of per-bin terms in
    (N_b, sum Mhat_b, sum Mhat_b^2, T_b) -- exact in the fine-bin limit
    (the approximation error is O(bin width^2) in log Phi's curvature).

    This is the cheap exact-enough carrier of the FULL magnitude likelihood
    for the joint-term cross-check (the Gaussian-prior path is the default;
    see the module docstring).
    """
    Mhat = reference_absolute_mags(m, z, Om0, w0, wa)
    T = np.asarray(m_lim, dtype=float) - (np.asarray(m, dtype=float) - Mhat)
    edges = np.linspace(T.min() - 1e-9, T.max() + 1e-9, n_bins + 1)
    idx = np.clip(np.digitize(T, edges) - 1, 0, n_bins - 1)
    stats = {"n": np.zeros(n_bins), "sum": np.zeros(n_bins),
             "sumsq": np.zeros(n_bins), "T": np.zeros(n_bins)}
    np.add.at(stats["n"], idx, 1.0)
    np.add.at(stats["sum"], idx, Mhat)
    np.add.at(stats["sumsq"], idx, Mhat ** 2)
    np.add.at(stats["T"], idx, T)
    keep = stats["n"] > 0
    return {k: v[keep] for k, v in stats.items()} | {
        "T": stats["T"][keep] / stats["n"][keep]}


def magnitude_loglike_from_stats(M0hat, sigma_M, stats):
    """Truncated-Gaussian magnitude log-likelihood from the binned statistics.

    ``sum_b [ -N_b log sigma - (S2_b - 2 mu S1_b + N_b mu^2)/(2 sigma^2)
              - N_b log Phi((T_b - mu)/sigma) ]``  (+ const).
    JAX-differentiable in (M0hat, sigma_M); usable as an explicit joint term.
    """
    n, s1, s2, T = (jnp.asarray(stats[k]) for k in ("n", "sum", "sumsq", "T"))
    mu, sig = M0hat, sigma_M
    quad = (s2 - 2.0 * mu * s1 + n * mu ** 2) / (2.0 * sig ** 2)
    log_trunc = jnp.log(jnp.maximum(ndtr((T - mu) / sig), 1e-300))
    return jnp.sum(-n * jnp.log(sig) - quad - n * log_trunc)


def load_selection_fit_json(path):
    """Load and validate a ``darksirens_fit_selection`` JSON; return the
    single-stratum theta dict ``{family, m_lim, M0hat, sigma_M, cov, ...}``.

    Multi-stratum payloads are rejected here for now: the single-survey
    builder and the K=1 likelihood carry one theta; per-stratum consumption
    arrives with real-catalog ingestion.
    """
    import json

    with open(path) as f:
        payload = json.load(f)
    fmt = payload.get("format_version")
    if fmt != "darksirens-selection-fit-1.0":
        raise ValueError(
            f"{path}: unknown selection-fit format {fmt!r} (expected "
            "darksirens-selection-fit-1.0 from darksirens_fit_selection).")
    strata = payload.get("strata") or []
    if len(strata) != 1:
        raise NotImplementedError(
            f"{path}: {len(strata)} strata; the single-survey consumers "
            "carry exactly one selection stratum for now.")
    s = dict(strata[0])
    for key in ("family", "m_lim", "M0hat", "sigma_M", "cov"):
        if key not in s:
            raise ValueError(f"{path}: stratum missing required key {key!r}.")
    if s["family"] != "gaussian":
        raise NotImplementedError(
            f"{path}: family {s['family']!r}; consumers support the gaussian "
            "family for now.")
    return s


@dataclass
class SelectionFit:
    """One stratum's fitted Gaussian-LF selection parameters."""

    family: str
    m_lim: float
    M0hat: float
    sigma_M: float
    cov: np.ndarray          # (2, 2) Laplace covariance of (M0hat, sigma_M)
    n_gal: int
    stratum: str = "all"
    meta: dict = field(default_factory=dict)

    def to_jsonable(self) -> dict:
        return {
            "family": self.family, "m_lim": self.m_lim,
            "M0hat": self.M0hat, "sigma_M": self.sigma_M,
            "cov": np.asarray(self.cov).tolist(), "n_gal": self.n_gal,
            "stratum": self.stratum, "meta": dict(self.meta),
        }


def fit_selection_from_mags(m, z, m_lim, *, family="gaussian",
                            Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial,
                            stratum="all"):
    """Truncated-LF maximum likelihood for one stratum (offline, numpy/scipy).

    The data are per-galaxy apparent magnitudes ``m_i`` at redshifts ``z_i``
    with a KNOWN hard limit ``m_i <= m_lim`` (the truncation datum of the
    selection protocol -- not a fitted parameter: inside the selection curve
    only the combination ``m_lim - M0hat`` is identified, so fitting both
    is an exact flat direction).  In reference absolute magnitudes the model
    is an upper-truncated Gaussian with per-galaxy truncation
    ``T_i = m_lim - DM(z_i; H0=100)``:

        L(theta) = prod_i  phi((Mhat_i - M0hat)/sigma) / sigma
                           / Phi((T_i - M0hat)/sigma) .

    Returns the MLE and the Laplace covariance from the numerical Hessian at
    the optimum (finite differences of the exact gradient-free objective).
    The z-shape of the sample never enters: this likelihood is exactly
    independent of the galaxy density field (thinning), which is what makes
    the fitted selection clustering-safe.
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    if family != "gaussian":
        raise NotImplementedError(
            f"offline fit implemented for the gaussian family only (got "
            f"{family!r}); the schechter curve is API-pinned for real-catalog "
            "ingestion, whose fit ships with that stage.")

    m = np.asarray(m, dtype=float)
    z = np.asarray(z, dtype=float)
    if m.shape != z.shape or m.ndim != 1 or m.size < 10:
        raise ValueError("need matching 1-D m, z with at least 10 galaxies")
    over = m > m_lim + 1e-9
    if over.any():
        raise ValueError(
            f"{int(over.sum())} galaxies are FAINTER than the declared "
            f"m_lim={m_lim}: the truncation datum does not describe this "
            "sample (wrong m_lim, or the survey is not magnitude-limited).")

    Mhat = reference_absolute_mags(m, z, Om0, w0, wa)
    T = m_lim - (m - Mhat)          # = m_lim - DM(z; H0_REF), per galaxy

    def nll(theta):
        mu, log_sig = theta
        sig = np.exp(log_sig)
        resid = (Mhat - mu) / sig
        log_trunc = norm.logcdf((T - mu) / sig)
        return -np.sum(norm.logpdf(resid) - np.log(sig) - log_trunc)

    theta0 = np.array([np.median(Mhat), np.log(max(np.std(Mhat), 1e-3))])
    res = minimize(nll, theta0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
    if not res.success:
        raise RuntimeError(f"selection fit did not converge: {res.message}")
    mu_hat, log_sig_hat = res.x
    sig_hat = float(np.exp(log_sig_hat))

    # Laplace covariance in (M0hat, sigma_M): numerical Hessian of the NLL
    # reparametrized to (mu, sigma) so the prior is Gaussian in the sampled
    # coordinates.
    def nll_sig(theta):
        return nll(np.array([theta[0], np.log(theta[1])]))

    x0 = np.array([mu_hat, sig_hat])
    h = np.array([1e-4, 1e-4])
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i] * h[i]
            ej = np.eye(2)[j] * h[j]
            H[i, j] = (nll_sig(x0 + ei + ej) - nll_sig(x0 + ei - ej)
                       - nll_sig(x0 - ei + ej) + nll_sig(x0 - ei - ej)
                       ) / (4.0 * h[i] * h[j])
    cov = np.linalg.inv(H)
    if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        raise RuntimeError("selection-fit Hessian is not positive definite")

    return SelectionFit(
        family="gaussian", m_lim=float(m_lim), M0hat=float(mu_hat),
        sigma_M=sig_hat, cov=cov, n_gal=int(m.size), stratum=str(stratum),
        meta={"Om0": float(Om0), "w0": float(w0), "wa": float(wa),
              "H0_ref": H0_REF, "nll": float(res.fun)},
    )
