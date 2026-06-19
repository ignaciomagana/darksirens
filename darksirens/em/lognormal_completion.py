"""
lognormal_completion.py
-----------------------
**Offline** builder for the LSS-conditioned lognormal completion field
``Q_LSS(p, z)`` consumed by :mod:`darksirens.em.completion`.

This module is **never imported by the GW likelihood** — the likelihood stays
deterministic and consumes fixed ``Q`` arrays (see
``EMCatalog.lss_completion_*``).  Here we *build* those arrays from a
per-pixel, 1-D Poisson-lognormal model along the redshift grid.

Model (per HEALPix pixel / catalog row, independent)
----------------------------------------------------
A latent Gaussian field ``s`` on the redshift grid is correlated along
comoving distance with a stationary covariance ``C`` (a built-in Gaussian
correlation, summarised by its power spectrum ``P``).  The observed binned
counts are Poisson with a clustered, completeness-modulated rate::

    s ~ Normal(0, C[P])
    N_obs_v ~ Poisson[ lambda_v ],   lambda_v = C_v dN_exp_v exp(bias s_v - bias^2 sigma_s^2 / 2)

so the completion factor is the mean-one lognormal ``Q_v = exp(bias s_v -
bias^2 sigma_s^2 / 2)``.  The MAP minimises::

    0.5 * prior_strength * s^T C^{-1} s  +  sum_v [ lambda_v - N_obs_v log lambda_v ].

All linear algebra is done in Fourier space on the (treated-as-uniform) grid
index lattice — ``C`` is represented by its circulant eigenvalues ``P`` (the
``power_spectrum``), so there is no dense covariance and no ill-conditioned
Cholesky.  ``sigma_s^2 = mean(P)`` is the marginal field variance.

The Laplace ensemble draws approximate posterior samples around the MAP with an
**FFT-diagonal** Hessian ``H(k) ~= prior_strength / P(k) + bias^2 median(lambda_map)``
— a robust, deterministic-given-seed approximation (not a full BORG sampler).

This module uses NumPy/SciPy only.
"""
from __future__ import annotations

import json
import numpy as np


def _require_scipy():
    try:
        from scipy import optimize  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Building LSS-conditioned lognormal completion requires SciPy "
            "(scipy.optimize). Install scipy to use "
            "darksirens.em.lognormal_completion."
        ) from exc
    return optimize


# ------------------------------------------------------------
# Built-in parametric prior: Gaussian correlation along the grid
# ------------------------------------------------------------

def gaussian_correlation_spectrum(n_grid: int, ell_grid: float, sigma: float,
                                  floor_frac: float = 1e-8) -> np.ndarray:
    """Circulant power spectrum ``P`` of a 1-D Gaussian correlation.

    The real-space periodic covariance is ``c[d] = sigma^2 exp(-d^2 / 2 ell^2)``
    (``d`` the periodic grid-index distance), and ``P = Re fft(c)`` are the
    eigenvalues of the circulant covariance.  ``mean(P) = c[0] = sigma^2`` is the
    marginal variance.  Eigenvalues are floored to a small positive fraction of
    the maximum for numerical stability.

    Parameters
    ----------
    n_grid : int
        Number of redshift-grid points (field length).
    ell_grid : float
        Correlation length in **grid-index units** (convert a physical Mpc
        length with the median comoving spacing of the grid).
    sigma : float
        Marginal standard deviation of the latent Gaussian field.
    """
    n = int(n_grid)
    d = np.arange(n)
    d = np.minimum(d, n - d).astype(float)          # periodic distance
    ell = max(float(ell_grid), 1e-6)
    c = (float(sigma) ** 2) * np.exp(-0.5 * (d / ell) ** 2)
    pk = np.fft.fft(c).real
    pk = np.maximum(pk, floor_frac * float(pk.max()))
    return pk


def _prior_sigma2(power_spectrum: np.ndarray) -> float:
    """Marginal field variance sigma_s^2 = c[0] = mean(P)."""
    return float(np.mean(np.asarray(power_spectrum, dtype=float)))


# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------

def poisson_lognormal_map(
    N_obs: np.ndarray,
    C: np.ndarray,
    dN_exp: np.ndarray,
    power_spectrum: np.ndarray,
    *,
    bias: float = 1.0,
    prior_strength: float = 1.0,
    maxiter: int = 300,
    logq_clip: float = 7.0,
) -> dict:
    """Per-pixel MAP of the 1-D Poisson-lognormal completion field.

    Parameters
    ----------
    N_obs : (N_rows, N_grid)
        Observed galaxy counts binned on the redshift grid.
    C : (N_rows, N_grid) or (N_grid,)
        Differential completeness on the grid (broadcast over rows if 1-D).
    dN_exp : (N_rows, N_grid) or (N_grid,)
        Homogeneous expected counts on the grid (broadcast over rows if 1-D).
    power_spectrum : (N_grid,)
        Circulant eigenvalues of the latent-field covariance (see
        :func:`gaussian_correlation_spectrum`).
    bias, prior_strength, maxiter, logq_clip
        Linear bias of the field, prior precision scaling, optimiser iteration
        cap, and the symmetric clip applied to ``log Q``.

    Returns
    -------
    dict with ``s_map``, ``logq_map``, ``q_map``, ``lambda_map`` (each
    ``(N_rows, N_grid)``) and ``diagnostics``.
    """
    optimize = _require_scipy()

    N_obs = np.atleast_2d(np.asarray(N_obs, dtype=float))
    n_rows, n_grid = N_obs.shape
    pk = np.asarray(power_spectrum, dtype=float)
    if pk.shape != (n_grid,):
        raise ValueError(
            f"power_spectrum has shape {pk.shape}, expected ({n_grid},) to match N_obs."
        )
    C = np.broadcast_to(np.asarray(C, dtype=float), (n_rows, n_grid))
    dN_exp = np.broadcast_to(np.asarray(dN_exp, dtype=float), (n_rows, n_grid))

    sigma2 = _prior_sigma2(pk)
    shift = 0.5 * bias * bias * sigma2
    b = float(bias)
    ps = float(prior_strength)

    s_map = np.zeros((n_rows, n_grid), dtype=float)
    lam_map = np.zeros((n_rows, n_grid), dtype=float)
    n_converged = 0

    for r in range(n_rows):
        nobs = N_obs[r]
        rate_base = C[r] * dN_exp[r]            # >= 0
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
            options={"maxiter": int(maxiter)},
        )
        s_map[r] = res.x
        lam_map[r] = np.where(mask, np.exp(log_rate + (b * res.x - shift)), 0.0)
        n_converged += int(bool(res.success))

    logq_map = np.clip(b * s_map - shift, -logq_clip, logq_clip)
    q_map = np.exp(logq_map)
    return {
        "s_map": s_map,
        "logq_map": logq_map,
        "q_map": q_map,
        "lambda_map": lam_map,
        "diagnostics": {
            "n_rows": int(n_rows),
            "n_grid": int(n_grid),
            "sigma_s2": float(sigma2),
            "bias": b,
            "prior_strength": ps,
            "n_converged": int(n_converged),
            "logq_clip": float(logq_clip),
        },
    }


# ------------------------------------------------------------
# Laplace / FFT-diagonal ensemble
# ------------------------------------------------------------

def laplace_lognormal_members(
    s_map: np.ndarray,
    lambda_map: np.ndarray,
    power_spectrum: np.ndarray,
    *,
    n_members: int = 32,
    bias: float = 1.0,
    prior_strength: float = 1.0,
    method: str = "fft_diagonal",
    seed: int | None = None,
    logq_clip: float = 7.0,
) -> dict:
    """Approximate posterior ensemble of ``Q`` around the MAP.

    **FFT-diagonal Laplace approximation** (not a full BORG sampler): the
    per-pixel Hessian is approximated as ``H(k) ~= prior_strength / P(k) +
    bias^2 median(lambda_map_row)``, residuals are drawn with circulant
    covariance ``H^{-1}`` via the exact spectral synthesis
    ``delta_s = Re ifft( sqrt(1/H) * fft(white) )``, added to ``s_map``, and
    ``log Q`` is clipped.  Deterministic given ``seed``.

    Returns ``{"logq_members", "q_members", "logq_mean", "diagnostics"}`` with
    member arrays shaped ``(M, N_rows, N_grid)``.
    """
    if method != "fft_diagonal":
        raise ValueError(
            f"Unknown member method '{method}'. Only 'fft_diagonal' is implemented "
            "(a Laplace/FFT-diagonal approximation, not full BORG)."
        )
    s_map = np.atleast_2d(np.asarray(s_map, dtype=float))
    lambda_map = np.atleast_2d(np.asarray(lambda_map, dtype=float))
    n_rows, n_grid = s_map.shape
    pk = np.asarray(power_spectrum, dtype=float)
    sigma2 = _prior_sigma2(pk)
    shift = 0.5 * bias * bias * sigma2
    b = float(bias)
    ps = float(prior_strength)
    M = int(n_members)
    rng = np.random.default_rng(seed)

    logq_members = np.empty((M, n_rows, n_grid), dtype=float)
    for r in range(n_rows):
        lam_scale = float(np.median(lambda_map[r]))
        H = ps / pk + (b * b) * lam_scale          # (N_grid,) FFT-diagonal Hessian
        inv_eig = 1.0 / np.maximum(H, 1e-30)        # circulant eigenvalues of H^{-1}
        sqrt_eig = np.sqrt(inv_eig)
        for m in range(M):
            g = rng.standard_normal(n_grid)
            delta_s = np.real(np.fft.ifft(sqrt_eig * np.fft.fft(g)))
            logq_members[m, r] = np.clip(b * (s_map[r] + delta_s) - shift, -logq_clip, logq_clip)

    q_members = np.exp(logq_members)
    logq_mean = np.log(np.mean(q_members, axis=0))   # log of posterior-mean Q
    return {
        "logq_members": logq_members,
        "q_members": q_members,
        "logq_mean": logq_mean,
        "diagnostics": {
            "n_members": M,
            "n_rows": int(n_rows),
            "n_grid": int(n_grid),
            "method": method,
            "bias": b,
            "prior_strength": ps,
            "seed": (None if seed is None else int(seed)),
        },
    }


# ------------------------------------------------------------
# HDF5 I/O
# ------------------------------------------------------------

def save_lss_completion_hdf5(
    path: str,
    *,
    logq_map: np.ndarray | None = None,
    logq_members: np.ndarray | None = None,
    zgrid: np.ndarray | None = None,
    indexing: str = "compact",
    completion_kind: str | None = None,
    metadata: dict | None = None,
) -> str:
    """Write a completion file with layout ``/lss_completion/{logq_map,logq_members,zgrid}``."""
    import h5py

    if logq_map is None and logq_members is None:
        raise ValueError("save_lss_completion_hdf5 needs logq_map and/or logq_members.")
    if indexing not in ("compact", "global"):
        raise ValueError(f"indexing must be 'compact' or 'global', got {indexing!r}.")
    if completion_kind is None:
        completion_kind = "laplace_members" if logq_members is not None else "map"

    with h5py.File(path, "w") as f:
        grp = f.create_group("lss_completion")
        if logq_map is not None:
            grp.create_dataset("logq_map", data=np.asarray(logq_map, dtype=float))
        if logq_members is not None:
            grp.create_dataset("logq_members", data=np.asarray(logq_members, dtype=float))
        if zgrid is not None:
            grp.create_dataset("zgrid", data=np.asarray(zgrid, dtype=float))
        grp.attrs["indexing"] = indexing
        grp.attrs["model"] = "poisson_lognormal"
        grp.attrs["completion_kind"] = completion_kind
        grp.attrs["created_by"] = "darksirens.em.lognormal_completion"
        if metadata:
            grp.attrs["diagnostics"] = json.dumps(metadata, default=str)
    return path


def load_lss_completion_hdf5(path: str) -> dict:
    """Read a completion file written by :func:`save_lss_completion_hdf5`.

    Returns a dict with ``logq_map`` / ``logq_members`` / ``zgrid`` (any may be
    ``None``) plus ``indexing``, ``model``, ``completion_kind`` and parsed
    ``diagnostics``.
    """
    import h5py

    out = {"logq_map": None, "logq_members": None, "zgrid": None,
           "indexing": "compact", "model": None, "completion_kind": None,
           "diagnostics": None}
    with h5py.File(path, "r") as f:
        grp = f["lss_completion"] if "lss_completion" in f else f
        if "logq_map" in grp:
            out["logq_map"] = np.asarray(grp["logq_map"])
        if "logq_members" in grp:
            out["logq_members"] = np.asarray(grp["logq_members"])
        if "zgrid" in grp:
            out["zgrid"] = np.asarray(grp["zgrid"])
        for key in ("indexing", "model", "completion_kind"):
            if key in grp.attrs:
                val = grp.attrs[key]
                out[key] = val.decode() if isinstance(val, bytes) else str(val)
        if "diagnostics" in grp.attrs:
            raw = grp.attrs["diagnostics"]
            raw = raw.decode() if isinstance(raw, bytes) else raw
            try:
                out["diagnostics"] = json.loads(raw)
            except Exception:
                out["diagnostics"] = raw
    return out
