"""
lognormal_completion.py
-----------------------
**Offline** builder for the LSS-conditioned lognormal completion field
``Q_LSS(p, z)`` consumed by :mod:`darksirens.redshift.completion`.

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

Caveats (the default **radial** completion, ``mode="radial"``):
- The field is independent **per pixel** — no angular coupling between
  neighbouring lines of sight.  The 3-D angular-coupling builder
  (``mode="gp3d"``; :func:`build_lowrank_operator` /
  :func:`poisson_lognormal_gp3d_map` / :func:`eval_logq_gp3d` below) lifts this:
  it solves ONE low-rank field over occupied (pixel x z) voxels with the
  (sphere x z) GP so empty pixels borrow angularly from their neighbours.
- The completeness ``C`` and the fitted ``Q`` come from the **same** observed
  counts, so ``Q`` is the sub-smoothing radial residual, not a separately
  identifiable completeness; and the model assumes **missing galaxies trace the
  observed clustering** along the line of sight (not validatable from the data
  alone).
- Built at fixed fiducial cosmology/survey parameters; the GW likelihood consumes
  the deterministic/posterior-mean ``Q`` (not the fully-marginalised ensemble).

The radial builder uses NumPy/SciPy only; the gp3d builder additionally uses
JAX and the (sphere x z) GP of :mod:`darksirens.sky.models`, both imported
lazily so importing this module for the radial API never requires JAX.
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
            "darksirens.redshift.lognormal_completion."
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
# 3-D angular-coupling (mode="gp3d"): ONE low-rank Poisson-lognormal field over
# occupied (pixel x z) voxels, reusing the (sphere x z) GP
# ------------------------------------------------------------
#
# The radial builder above solves an INDEPENDENT 1-D field per pixel.  The gp3d
# builder instead solves a SINGLE field over all occupied (pixel x z) voxels,
# coupled by the whitened finite-rank (sphere x z) GP of
# :mod:`darksirens.sky.models` (chordal-RBF on n-hat x RBF on zeta=log1p(z),
# Fibonacci-sphere x z inducing nodes).  Because the field is
#     f(x) = k(x, Z) @ L^{-T} xi = Phi @ xi   (Phi = k(x,Z) L^{-T}, xi ~ N(0,I)),
# it is LINEAR in the M~192 whitened latents xi, so the Poisson-lognormal MAP is
# a SINGLE convex GLM (Newton over an M x M Hessian) instead of a per-pixel loop.
# The output table is the Laplace POSTERIOR-MEAN E[Q] (:func:`eval_logq_gp3d`):
# under-observed pixels BORROW angularly from neighbours, and pixels far from any
# data read as exactly Q=1 (there the posterior variance equals the prior
# variance, so the lognormal-mean and the mean-one shift cancel).


def _require_jax_kernel():
    """Lazy import of JAX + the (sphere x z) GP kernel for the gp3d builder.

    Reuses the EXACT kernel and inducing-node geometry of the online sky model
    (:func:`darksirens.sky.models._sphere_z_kernel`,
    :func:`darksirens.sky.models._fibonacci_sphere`) so the offline completion
    field and the online sky field share one construction.  Enables float64
    (``jax_enable_x64``) defensively — the chordal/zeta kernel + Cholesky need
    double precision and a direct caller (e.g. a unit test) may not have imported
    a module that already set it.  Kept out of the module import so the radial
    NumPy/SciPy API never requires JAX.
    """
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        import jax.scipy.linalg as jsl
        from darksirens.sky.models import _sphere_z_kernel, _fibonacci_sphere
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The 3-D angular-coupling completion (mode='gp3d') requires JAX and "
            "darksirens.sky.models. Install jax to use the gp3d builder, or use "
            "the radial builder (mode='radial')."
        ) from exc
    return jnp, jsl, _sphere_z_kernel, _fibonacci_sphere


def lowrank_inducing_nodes(n_inducing_sphere: int = 32, n_inducing_z: int = 6,
                           z_node_hi: float = 3.0):
    """Inducing nodes ``(Zn, Zz)`` of the (sphere x z) GP.

    IDENTICAL to :class:`darksirens.sky.models._SphereZGPBase`: a Fibonacci
    sphere (``M_sph`` points) crossed with ``linspace(0, log1p(z_node_hi), M_z)``
    in ``zeta = log1p(z)``, flattened with ordering ``i = i_sph * M_z + i_z`` so
    the offline builder and the online sky GP agree node-for-node (asserted in
    the tests).

    Returns
    -------
    Zn : (M, 3) unit-vector sphere coordinates of the nodes.
    Zz : (M,)   ``zeta = log1p(z)`` coordinates of the nodes.
    """
    jnp, _, _, _fibonacci_sphere = _require_jax_kernel()
    M_sph = int(n_inducing_sphere)
    M_z = int(n_inducing_z)
    Z_sph = _fibonacci_sphere(M_sph)                                   # (M_sph, 3)
    zeta_nodes = jnp.linspace(0.0, float(jnp.log1p(z_node_hi)), M_z)   # (M_z,)
    Zn = jnp.repeat(Z_sph, M_z, axis=0)                               # (M, 3)
    Zz = jnp.tile(zeta_nodes, M_sph)                                  # (M,)
    return Zn, Zz


def build_lowrank_operator(Zn, Zz, X_n, X_z, *, amp, ls_sph, ls_z,
                           jitter_rel: float = 1e-4, jitter_abs: float = 1e-9):
    """Whitened finite-rank GP design matrix ``Phi`` and Cholesky factor ``L``.

    With ``K = k(Z,Z) + jitter I`` and ``L = chol(K)``, returns
    ``Phi = k(X, Z) @ L^{-T}`` (shape ``(V, M)``) so the field at the voxels is
    ``f = Phi @ xi`` for whitened latents ``xi ~ N(0, I)`` — the EXACT
    construction of :meth:`_SphereZGPBase._field` (there ``alpha = L^{-T} xi``,
    ``f = k(X,Z) @ alpha``).  ``sum(Phi**2, axis=1) = k(x,Z) K^{-1} k(Z,x)`` is
    the per-voxel prior (Nystrom) variance used for the mean-one shift.

    ``jitter = jitter_rel * amp**2 + jitter_abs`` matches the GP's own floor.
    """
    jnp, jsl, _sphere_z_kernel, _ = _require_jax_kernel()
    M = int(Zn.shape[0])
    jitter = jitter_rel * float(amp) ** 2 + jitter_abs
    K = _sphere_z_kernel(Zn, Zz, Zn, Zz, amp, ls_sph, ls_z) + jitter * jnp.eye(M)
    L = jnp.linalg.cholesky(K)
    Kxz = _sphere_z_kernel(X_n, X_z, Zn, Zz, amp, ls_sph, ls_z)        # (V, M)
    # Phi = Kxz @ L^{-T}.  solve_triangular(L, Kxz.T, lower=True) = L^{-1} Kxz.T,
    # whose transpose is Kxz @ L^{-T}.
    Phi = jsl.solve_triangular(L, Kxz.T, lower=True).T                 # (V, M)
    return Phi, L


def poisson_lognormal_gp3d_map(
    N_obs,
    base,
    Phi,
    *,
    bias: float = 1.0,
    sigma2_vox=None,
    max_newton: int = 50,
    tol: float = 1e-8,
    logq_clip: float = 7.0,
    field_clip: float = 10.0,
) -> dict:
    """Single convex Poisson-lognormal MAP over the ``M`` whitened GP latents.

    Minimises (convex in ``xi``)::

        J(xi) = 0.5 ||xi||^2 + sum_v [ lam_v - N_obs_v log lam_v ],
        lam_v   = base_v * exp(bias * (Phi @ xi)_v - shift_v),
        shift_v = 0.5 * bias^2 * sigma2_vox_v        (per-voxel mean-one shift),

    over voxels ``v`` with ``base_v = C_v * dN_exp_v >= 0`` (active mask
    ``base_v > 0``).  ``sigma2_vox`` defaults to ``sum(Phi**2, axis=1)`` (the
    Nystrom prior variance) for the per-voxel lognormal mean-one shift.  Solved by
    Newton with Armijo backtracking on the ``M x M`` SPD Hessian
    ``H = I + bias^2 Phi^T diag(lam) Phi`` (SPD even when ``V < M`` thanks to the
    prior ``I``); falls back to L-BFGS-B if Newton stalls.

    Returns ``{xi_map, H_chol, sigma2_vox, f_solve, logq_solve, lambda_solve,
    diagnostics}``.  ``H_chol`` (lower Cholesky of ``H`` at the MAP) feeds BOTH the
    Laplace ensemble (:func:`laplace_lognormal_gp3d_members`) and the deterministic
    posterior-mean output table (:func:`eval_logq_gp3d`).  ``logq_solve`` is the
    per-draw MAP value (a diagnostic), not the output table.
    """
    optimize = _require_scipy()
    Phi = np.asarray(Phi, dtype=float)              # (V, M)
    N_obs = np.asarray(N_obs, dtype=float).ravel()  # (V,)
    base = np.asarray(base, dtype=float).ravel()    # (V,)
    V, M = Phi.shape
    b = float(bias)
    mask = base > 0.0
    logbase = np.where(mask, np.log(np.where(mask, base, 1.0)), 0.0)
    if sigma2_vox is None:
        sigma2_vox = np.sum(Phi ** 2, axis=1)       # (V,) Nystrom variance
    else:
        sigma2_vox = np.asarray(sigma2_vox, dtype=float).ravel()
    shift = 0.5 * b * b * sigma2_vox                # (V,)
    eye = np.eye(M)

    def _lam(xi):
        fld = np.clip(Phi @ xi, -field_clip, field_clip)
        return fld, np.where(mask, np.exp(logbase + b * fld - shift), 0.0)

    def _objective(xi):
        fld, lam = _lam(xi)
        loglam = logbase + b * fld - shift
        data = float(np.sum(lam - N_obs * np.where(mask, loglam, 0.0)))
        return 0.5 * float(np.dot(xi, xi)) + data

    def _grad(xi):
        _, lam = _lam(xi)
        return xi + b * (Phi.T @ np.where(mask, lam - N_obs, 0.0))

    # Scale-aware gradient tolerance: the Poisson data gradient scales with the
    # observed counts, so convergence is judged on the gradient inf-norm relative
    # to the problem scale (``tol`` is the relative tolerance).
    gscale = max(1.0, b * float(np.max(np.abs(N_obs)))) if N_obs.size else 1.0
    gtol = float(tol) * gscale

    xi = np.zeros(M, dtype=float)
    n_iter = 0
    for n_iter in range(1, int(max_newton) + 1):
        _, lam = _lam(xi)
        g = xi + b * (Phi.T @ np.where(mask, lam - N_obs, 0.0))
        gnorm = float(np.max(np.abs(g))) if M else 0.0
        if gnorm < gtol:
            break
        W = Phi * (np.where(mask, lam, 0.0) * (b * b))[:, None]    # (V, M)
        H = eye + Phi.T @ W                                        # (M, M) SPD
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:                             # pragma: no cover
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        # Armijo backtracking on the convex objective.
        obj0 = _objective(xi)
        slope = float(np.dot(g, step))
        t = 1.0
        accepted = False
        for _ls in range(40):
            xi_try = xi - t * step
            obj_try = _objective(xi_try)
            if np.isfinite(obj_try) and obj_try <= obj0 - 1e-4 * t * slope:
                xi = xi_try
                accepted = True
                break
            t *= 0.5
        if not accepted:
            break  # line search stalled -> hand off to the L-BFGS fallback

    g = _grad(xi)
    gnorm = float(np.max(np.abs(g))) if M else 0.0
    if (gnorm >= gtol) or (not np.all(np.isfinite(xi))):
        res = optimize.minimize(
            _objective, np.zeros(M), jac=_grad, method="L-BFGS-B",
            options={"maxiter": 1000},
        )
        if np.all(np.isfinite(res.x)) and _objective(res.x) <= _objective(xi):
            xi = np.asarray(res.x, dtype=float)
            g = _grad(xi)
            gnorm = float(np.max(np.abs(g))) if M else 0.0
    converged = bool(gnorm < gtol)

    # Hessian Cholesky at the MAP (feeds the Laplace ensemble).
    _, lam = _lam(xi)
    W = Phi * (np.where(mask, lam, 0.0) * (b * b))[:, None]
    H = eye + Phi.T @ W
    try:
        H_chol = np.linalg.cholesky(H)              # lower; H = H_chol H_chol^T
    except np.linalg.LinAlgError:                   # pragma: no cover
        H_chol = np.linalg.cholesky(H + 1e-8 * eye)

    fld = np.clip(Phi @ xi, -field_clip, field_clip)
    logq_solve = np.clip(b * fld - shift, -logq_clip, logq_clip)
    return {
        "xi_map": xi,
        "H_chol": H_chol,
        "sigma2_vox": sigma2_vox,
        "f_solve": fld,
        "logq_solve": logq_solve,
        "lambda_solve": lam,
        "diagnostics": {
            "n_voxels": int(V),
            "M": int(M),
            "bias": b,
            "n_iter": int(n_iter),
            "converged": bool(converged),
            "grad_inf": float(gnorm),
            "logq_clip": float(logq_clip),
        },
    }


def laplace_lognormal_gp3d_members(xi_map, H_chol, *, n_members: int = 32,
                                   seed: int | None = None) -> np.ndarray:
    """Laplace posterior ensemble of the whitened latents around the MAP.

    Draws ``xi_m = xi_map + L_H^{-T} g_m``, ``g_m ~ N(0, I_M)``, where
    ``H = L_H L_H^T`` is the MAP Hessian Cholesky, so ``Cov(xi_m) = H^{-1}``.
    One ``M x M`` factor for the whole ensemble — cheaper and more principled than
    the radial FFT-diagonal members (it captures the full cross-pixel/z posterior
    correlation).  Deterministic given ``seed``.  Returns ``(n_members, M)``.
    """
    from scipy.linalg import solve_triangular
    xi_map = np.asarray(xi_map, dtype=float).ravel()
    L_H = np.asarray(H_chol, dtype=float)
    M = xi_map.shape[0]
    rng = np.random.default_rng(seed)
    out = np.empty((int(n_members), M), dtype=float)
    for m in range(int(n_members)):
        g = rng.standard_normal(M)
        out[m] = xi_map + solve_triangular(L_H, g, lower=True, trans="T")
    return out


def eval_logq_gp3d(
    xi,
    Zn,
    Zz,
    *,
    amp,
    ls_sph,
    ls_z,
    n_hat_out,
    z_out,
    bias: float = 1.0,
    logq_clip: float = 7.0,
    pix_chunk: int = 512,
    L=None,
    H_chol=None,
) -> np.ndarray:
    """Evaluate the completion ``logQ`` on (pixel x z) output voxels, CHUNKED over
    pixels.  The continuous field is evaluated directly on ``z_out`` (the package
    zgrid), so no interpolation-back is needed.  Two modes:

    * **Deterministic posterior-mean** (``xi`` is the MAP ``(M,)`` *and*
      ``H_chol`` is given) -> ``(n_pix, n_grid)``:
      ``logQ = bias*f_MAP - 0.5*bias^2*(prior_var - post_var)`` = log of the
      Laplace posterior mean ``E[Q]`` (with ``prior_var = k(x,Z)K^{-1}k(Z,x)`` and
      ``post_var = phi(x) H^{-1} phi(x)^T``).  Data-free voxels have
      ``post_var = prior_var`` so ``logQ = 0`` (Q = 1, homogeneous); near data the
      variance shrinks and the field borrows; data-rich voxels recover the MAP.
      **This is the table the GW likelihood consumes.**
    * **Per-draw** (``H_chol`` is ``None``): the lognormal mean-one draw
      ``logQ = bias*f - 0.5*bias^2*prior_var`` for each latent vector; ``xi`` may
      be ``(M,)`` -> ``(n_pix, n_grid)`` or ``(n_members, M)`` ->
      ``(n_members, n_pix, n_grid)`` (the HDF5 members layout).  The empirical mean
      of the members reproduces the deterministic posterior mean.

    Pass ``L`` (the Cholesky from :func:`build_lowrank_operator`) to avoid
    recomputing it; if omitted it is rebuilt with the same default jitter.
    """
    jnp, jsl, _sphere_z_kernel, _ = _require_jax_kernel()
    xi = jnp.asarray(xi)
    single = xi.ndim == 1
    Xi = xi[:, None] if single else xi.T              # (M, K)
    K = int(Xi.shape[1])
    Zn = jnp.asarray(Zn)
    Zz = jnp.asarray(Zz)
    M = int(Zn.shape[0])
    n_hat_out = np.asarray(n_hat_out, dtype=float)     # (n_pix, 3)
    n_pix = n_hat_out.shape[0]
    z_out = np.asarray(z_out, dtype=float)
    n_grid = z_out.shape[0]
    zeta_out = jnp.log1p(jnp.clip(jnp.asarray(z_out), 0.0, None))  # (n_grid,)
    b = float(bias)

    posterior_mean = (H_chol is not None) and single
    Hc = jnp.asarray(H_chol) if posterior_mean else None

    if L is None:
        jitter = 1e-4 * float(amp) ** 2 + 1e-9
        Kzz = _sphere_z_kernel(Zn, Zz, Zn, Zz, amp, ls_sph, ls_z) + jitter * jnp.eye(M)
        L = jnp.linalg.cholesky(Kzz)
    else:
        L = jnp.asarray(L)

    out = (np.empty((n_pix, n_grid), dtype=float) if single
           else np.empty((K, n_pix, n_grid), dtype=float))

    step = max(int(pix_chunk), 1)
    for start in range(0, n_pix, step):
        nh = jnp.asarray(n_hat_out[start:start + step])    # (Pc, 3)
        Pc = int(nh.shape[0])
        Xn = jnp.repeat(nh, n_grid, axis=0)                # (Pc*n_grid, 3)
        Xz = jnp.tile(zeta_out, Pc)                        # (Pc*n_grid,)
        Kxz = _sphere_z_kernel(Xn, Xz, Zn, Zz, amp, ls_sph, ls_z)  # (Pc*n_grid, M)
        Phi = jsl.solve_triangular(L, Kxz.T, lower=True).T         # (Pc*n_grid, M)
        prior_var = jnp.sum(Phi ** 2, axis=1)              # (Pc*n_grid,)
        if posterior_mean:
            U = jsl.solve_triangular(Hc, Phi.T, lower=True)        # (M, Pc*n_grid)
            post_var = jnp.sum(U ** 2, axis=0)                     # (Pc*n_grid,)
            f = Phi @ Xi[:, 0]                                     # (Pc*n_grid,)
            logq = jnp.clip(b * f - 0.5 * b * b * (prior_var - post_var),
                            -logq_clip, logq_clip)
            out[start:start + Pc] = np.asarray(logq).reshape(Pc, n_grid)
        else:
            f = Phi @ Xi                                          # (Pc*n_grid, K)
            shift = 0.5 * b * b * prior_var
            logq = jnp.clip(b * f - shift[:, None], -logq_clip, logq_clip)
            logq_np = np.asarray(logq).reshape(Pc, n_grid, K)
            if single:
                out[start:start + Pc] = logq_np[..., 0]
            else:
                out[:, start:start + Pc, :] = np.moveaxis(logq_np, 2, 0)
    return out


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
        grp.attrs["created_by"] = "darksirens.redshift.lognormal_completion"
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
