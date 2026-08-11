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
- Budget convention: shipped tables are per-z mean-one renormalized under the
  missing-budget weights ``(1 - C) dN_exp`` over the fitted footprint
  (:func:`renormalize_q_mean_one`) — ``Q`` only PLACES the missing budget,
  never rescales it (the total is C's and n0's job).  The removed monopole and
  a boolean stamp are recorded in the HDF5 attrs; unstamped (legacy) tables
  draw a loud warning at load.
- Completeness target: the fit is residual to a completeness base, either the
  legacy per-pixel ratio (``c_mode="per_pixel"``) or the opt-in sky-aggregate
  ``Cbar`` with empty pixels included as N_obs = 0 rows (``c_mode="aggregate"``;
  Q then carries the FULL observed overdensity, voids included).  The mode is
  stamped in the HDF5 attrs (absent = legacy per_pixel) and must match the
  consuming survey's ``c_mode`` — the two targets differ by the entire
  clustering signal, so the inference loader hard-errors on a mismatch.

The radial builder uses NumPy/SciPy only; the gp3d builder additionally uses
JAX and the (sphere x z) GP of :mod:`darksirens.sky.models`, both imported
lazily so importing this module for the radial API never requires JAX.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
import warnings
from contextlib import contextmanager

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


def _laplace_diag_variance(lam, pk, bias, prior_strength):
    """Per-BIN Laplace posterior variance of ``s`` under the FFT-diagonal Hessian.

    The Hessian is ``H = prior_strength C^{-1} + bias^2 diag(lambda)``.  Freezing
    the data curvature at each bin's OWN ``lambda_v`` keeps the operator
    circulant locally, so the diagonal of ``H^{-1}`` reads

        var_v = mean_k 1 / (prior_strength / P_k + bias^2 lambda_v),

    which is exact in both limits that matter: a data-free bin
    (``lambda_v = 0``) gets the effective PRIOR variance ``sigma_s^2 /
    prior_strength`` -- so ``E[Q] = exp(b s + b^2 var/2 - shift)`` reads
    ``Q -> 1`` there -- and a data-rich bin gets ``~1/(b^2 lambda_v) -> 0``, so
    ``E[Q] -> Q(s_map)``.

    Using ONE row scalar instead (``median(lambda_row)``) was wrong wherever a
    row's completeness support covers less than half its bins -- the norm for a
    z < 1 survey on the z <= 5 package grid: the median is then exactly 0, the
    row's var is pinned at the prior variance, the ``+0.5 b^2 var`` term cancels
    ``shift`` in EVERY bin, and the data-rich bins are inflated by
    ``exp(0.5 b^2 sigma_s^2 / prior_strength)`` (measured 1.65x at
    ``b = sigma = ps = 1``).  Worse, the inflation switched on and off with the
    fraction of covered bins, so pixels at fixed z got discontinuously
    different Jensen boosts driven by grid coverage rather than by data.

    Cost is ``O(n_rows * n_grid^2)`` (chunked over rows to bound the
    intermediate), i.e. milliseconds per row against the seconds of that row's
    MAP solve.
    """
    lam = np.atleast_2d(np.asarray(lam, dtype=float))
    pk = np.asarray(pk, dtype=float)
    n_rows, n_grid = lam.shape
    A = float(prior_strength) / pk                       # (n_grid,)
    b2 = float(bias) * float(bias)
    out = np.empty((n_rows, n_grid), dtype=float)
    # ~64 MB of (chunk, n_grid, n_grid) float64 intermediate at a time.
    chunk = max(1, int(8_000_000 // max(n_grid * n_grid, 1)))
    for start in range(0, n_rows, chunk):
        blk = lam[start:start + chunk]                   # (c, n_grid)
        H = A[None, None, :] + b2 * blk[:, :, None]      # (c, n_grid, n_grid)
        out[start:start + chunk] = np.mean(
            1.0 / np.maximum(H, 1e-30), axis=-1)
    return out


# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------

def _map_solve_row(nobs, c_row, dN_exp_row, pk, bias, prior_strength, shift, maxiter):
    """One pixel's 1-D Poisson-lognormal MAP solve.

    Module-level (not a closure) so a process pool can pickle it; the row solves
    are independent, which is what makes the loop below parallelisable.
    """
    optimize = _require_scipy()
    n_grid = int(pk.size)
    b, ps = float(bias), float(prior_strength)
    rate_base = c_row * dN_exp_row                  # >= 0
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
        # maxfun: scipy's DEFAULT (15000) silently capped every large
        # --maxiter run.  L-BFGS-B stops with status=1 ("TOTAL NO. of f
        # AND g EVALUATIONS EXCEEDS LIMIT") once function EVALUATIONS --
        # not iterations -- reach 15000, i.e. around iteration ~12k at
        # DESI scale, no matter how high maxiter was raised.  That is why
        # raising maxiter beyond ~15k changed neither the result nor the
        # wall time while res.success stayed False.  maxls (default 20)
        # bounds evaluations per iteration, so 21*maxiter can never bind
        # before maxiter itself does.
        options={"maxiter": int(maxiter), "maxfun": 21 * int(maxiter)},
    )
    lam = np.where(mask, np.exp(log_rate + (b * res.x - shift)), 0.0)
    ok = bool(res.success)
    return res.x, lam, ok, (None if ok else str(getattr(res, "message", "")))


def _map_solve_chunk(payload):
    """Solve a contiguous block of rows — one process-pool task."""
    N_obs, C, dN_exp, pk, bias, prior_strength, shift, maxiter = payload
    n = C.shape[0]
    s_out = np.empty((n, C.shape[1]), dtype=float)
    lam_out = np.empty((n, C.shape[1]), dtype=float)
    n_converged = 0
    last_failure = None
    for i in range(n):
        s, lam, ok, message = _map_solve_row(
            N_obs[i], C[i], dN_exp[i], pk, bias, prior_strength, shift, maxiter)
        s_out[i] = s
        lam_out[i] = lam
        n_converged += int(ok)
        if not ok:
            last_failure = message
    return s_out, lam_out, n_converged, last_failure


_THREAD_PIN_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@contextmanager
def _pinned_parent_env():
    """Export the one-thread pins in the PARENT while workers are spawned.

    Spawn children re-exec python and inherit ``os.environ`` at exec time,
    BEFORE any of their imports — so parent-side variables reach the child's
    BLAS/OpenMP load, which is where eager builds size their (spin-waiting)
    thread pools.  The parent's own values are restored afterwards; its
    already-loaded libraries were sized long ago and are unaffected either
    way.
    """
    import os

    saved = {var: os.environ.get(var) for var in _THREAD_PIN_VARS}
    for var in _THREAD_PIN_VARS:
        os.environ[var] = "1"
    try:
        yield
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value


def _init_worker():
    """Pin each worker to one BLAS/FFT thread so W processes don't oversubscribe.

    Belt only — the braces are :func:`_pinned_parent_env` below.  A spawned
    worker re-imports ``__main__`` (and numpy/BLAS with it) BEFORE the pool
    initializer runs, and OpenBLAS sizes its spin-waiting thread pool at
    library load from the environment, so setting the variables here is too
    late for builds that read them eagerly (measured: 16 workers x ~20
    spin-waiting OpenBLAS threads pushed a 20-core box to load ~160).
    """
    import os

    for var in _THREAD_PIN_VARS:
        os.environ[var] = "1"


def poisson_lognormal_map(
    N_obs: np.ndarray,
    C: np.ndarray,
    dN_exp: np.ndarray,
    power_spectrum: np.ndarray,
    *,
    bias: float = 1.0,
    prior_strength: float = 1.0,
    # 300 stopped every DESI-scale n_grid=1000 solve at the iteration cap
    # (status=1, |grad|_inf ~ 21), under-relaxing logQ ~5x toward Q = 1, and
    # 10000 was still capped on near-empty pixels (measured: the 10000-iterate
    # of a 4-galaxy pixel reads logQ 0.556 where the fixed point is 0.924).
    # L-BFGS-B SELF-TERMINATES once converged, so the high cap only costs on
    # solves that genuinely need it.
    maxiter: int = 200000,
    logq_clip: float = 7.0,
    workers: int = 1,
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
    workers
        Row solves to run in parallel processes (1 = serial, the default).  The
        per-pixel solves are independent, so this is exact — only the wall clock
        changes.  At DESI nside=128 there are ~69k occupied pixels at ~2.2 s
        each, so the serial loop is ~42 h and a 16-way pool is ~3 h.

        With ``workers > 1`` the call MUST sit behind
        ``if __name__ == "__main__":`` in the calling program: workers are
        spawned, so each one re-imports the caller's ``__main__``, and an
        unguarded top-level call recursively spawns processes until CPython
        aborts it.  ``workers=1`` has no such constraint.

    Returns
    -------
    dict with ``s_map``, ``logq_map``, ``q_map``, ``lambda_map`` (each
    ``(N_rows, N_grid)``) and ``diagnostics``.
    """
    # Probe here, not only inside the row solve: with workers > 1 a missing
    # scipy would otherwise surface as an opaque BrokenProcessPool from a child.
    _require_scipy()

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
    b = float(bias)
    ps = float(prior_strength)
    # Mean-one shift uses the EFFECTIVE prior variance sigma2/ps.  The prior
    # imposed below is 0.5*ps*s^T C^-1 s, whose marginal variance is sigma2/ps,
    # and the Laplace posterior variance (var_post, via H = ps/pk + ...) scales
    # the same way; omitting the /ps divisor left the E[Q] mean-one/homogeneity
    # cancellation exact ONLY at ps=1, so a data-free bin read Q<1 (a ~31%
    # missing-density suppression at prior_strength=4, bias=1) for any other
    # prior_strength.
    shift = 0.5 * b * b * sigma2 / ps

    s_map = np.zeros((n_rows, n_grid), dtype=float)
    lam_map = np.zeros((n_rows, n_grid), dtype=float)
    n_converged = 0
    last_failure = None

    n_workers = max(1, int(workers))
    if n_workers > 1 and n_rows > 1:
        # Independent per-row solves: farm contiguous blocks out to a process
        # pool.  Several blocks per worker so the (data-dependent) per-row cost
        # load-balances instead of stranding one slow block at the end.
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # NOT the default fork: this module is reachable from a live JAX process,
        # and forking a multithreaded JAX runtime both warns and (measured) runs
        # ~10x SLOWER than serial from thread contention in the children.
        #
        # Spawned workers start as clean interpreters that never touch the GPU.
        #
        # CALLER CONTRACT: with workers > 1 the call must be reachable only under
        # `if __name__ == "__main__":`.  Every non-fork start method re-imports
        # the caller's __main__ in each worker, so an unguarded top-level call
        # re-enters this function in the child and recursively spawns until
        # CPython's _check_not_importing_main aborts.  forkserver does NOT avoid
        # this -- the check lives in multiprocessing.spawn.get_preparation_data,
        # which forkserver uses too.  cli/build_lognormal_completion.py guards
        # its main(), which is why the CLI path is safe.
        mp_ctx = multiprocessing.get_context("spawn")
        n_workers = min(n_workers, n_rows)
        edges = np.linspace(0, n_rows, min(4 * n_workers, n_rows) + 1).astype(int)
        spans = [(int(a), int(z)) for a, z in zip(edges[:-1], edges[1:]) if z > a]
        payloads = [
            (N_obs[a:z], C[a:z], dN_exp[a:z], pk, b, ps, shift, maxiter)
            for a, z in spans
        ]
        with _pinned_parent_env():
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx,
                                     initializer=_init_worker) as pool:
                for (a, z), (s_blk, lam_blk, nconv, blk_failure) in zip(
                        spans, pool.map(_map_solve_chunk, payloads)):
                    s_map[a:z] = s_blk
                    lam_map[a:z] = lam_blk
                    n_converged += int(nconv)
                    if blk_failure is not None:
                        last_failure = blk_failure
    else:
        for r in range(n_rows):
            s, lam, ok, message = _map_solve_row(
                N_obs[r], C[r], dN_exp[r], pk, b, ps, shift, maxiter)
            s_map[r] = s
            lam_map[r] = lam
            n_converged += int(ok)
            if not ok:
                last_failure = message

    # Deterministic table = Laplace posterior-mean E[Q], using the SAME
    # FFT-diagonal Hessian approximation as `laplace_lognormal_members`, so the
    # deterministic table agrees with the mean of its own ensemble and with the
    # gp3d output convention: a data-free bin reads logQ -> 0 (Q = 1,
    # homogeneous; var_post -> sigma_s^2/ps cancels the mean-one shift) instead
    # of the raw per-draw MAP's -0.5 b^2 sigma_s^2 (library review, radial
    # finding 2). Data-rich bins are unchanged to O(var_post): the posterior
    # variance collapses and E[Q] -> Q(s_map). The curvature is PER BIN (each
    # bin's own lambda); a row scalar cancelled the shift everywhere on any row
    # whose completeness support covered under half its bins -- see
    # `_laplace_diag_variance`.
    var_post = _laplace_diag_variance(lam_map, pk, b, ps)   # (n_rows, n_grid)
    logq_map = np.clip(
        b * s_map - shift + 0.5 * (b * b) * var_post,
        -logq_clip, logq_clip,
    )
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
            "converged": bool(n_converged == n_rows),
            # The message of the LAST unconverged solve: distinguishes an
            # iteration-cap stop (raise --maxiter) from a line-search failure.
            "last_failure_message": last_failure,
            "maxiter": int(maxiter),
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
    b = float(bias)
    ps = float(prior_strength)
    # Mean-one shift uses the EFFECTIVE prior variance sigma2/ps.  The prior
    # imposed below is 0.5*ps*s^T C^-1 s, whose marginal variance is sigma2/ps,
    # and the Laplace posterior variance (var_post, via H = ps/pk + ...) scales
    # the same way; omitting the /ps divisor left the E[Q] mean-one/homogeneity
    # cancellation exact ONLY at ps=1, so a data-free bin read Q<1 (a ~31%
    # missing-density suppression at prior_strength=4, bias=1) for any other
    # prior_strength.
    shift = 0.5 * b * b * sigma2 / ps
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
# Per-z mean-one budget renormalization ("C says how much, Q says where")
# ------------------------------------------------------------

def renormalize_q_mean_one(logq: np.ndarray, weights: np.ndarray):
    """Per-z mean-one renormalization of a ``logQ`` table under budget weights.

    The GW likelihood forms the missing density as ``dN_miss = (1 - C) dN_exp Q``
    (see :mod:`darksirens.redshift.completion`), so any per-z monopole in ``Q``
    RESCALES the total missing budget instead of only redistributing it — the
    Laplace posterior-mean ``E[Q] = exp(b s + b^2 var_post / 2 - shift)`` carries
    exactly such a monopole because ``var_post`` is largest where data are sparse
    (spatially varying Jensen bias; measured +55% budget inflation for radial
    tables).  The budget is C's and n0's job; ``Q`` only places it.  This helper
    enforces that division of labour by construction::

        logQ_p(z) -= log[ sum_p w_p(z) Q_p(z) / sum_p w_p(z) ]

    with the caller-supplied weights ``w_p(z) = (1 - C_p(z)) * dN_exp(z)``
    evaluated at the build-time fiducial, summed over the FITTED FOOTPRINT
    (the rows supplied here — the pixels the builder actually fit, never the
    full sky: out-of-footprint pixels' homogeneous budget must not absorb the
    footprint's monopole).  Afterwards ``sum_p w_p Q_p == sum_p w_p`` holds
    per z-bin to float precision.

    Parameters
    ----------
    logq : (n_rows, n_grid) or (n_members, n_rows, n_grid)
        MAP table, or a member cube — each member is renormalized
        INDEPENDENTLY (its own monopole), so members carry placement
        uncertainty but zero budget uncertainty.
    weights : (n_rows, n_grid)
        Non-negative budget weights over the fitted footprint.

    Returns
    -------
    (logq_renorm, log_monopole)
        ``log_monopole`` is the removed per-z curve, shaped ``(n_grid,)`` for a
        MAP table and ``(n_members, n_grid)`` for a member cube.  Bins with
        ``sum_p w_p(z) == 0`` (completeness saturates everywhere, no missing
        budget to place) are left unchanged (``log_monopole = 0`` there).
    """
    lq = np.asarray(logq, dtype=float)
    w = np.asarray(weights, dtype=float)
    if lq.ndim not in (2, 3):
        raise ValueError(
            f"logq must be (n_rows, n_grid) or (n_members, n_rows, n_grid); "
            f"got shape {lq.shape}.")
    if w.shape != lq.shape[-2:]:
        raise ValueError(
            f"weights shape {w.shape} does not match the logq table's "
            f"(n_rows, n_grid) = {lq.shape[-2:]}.")
    if np.any(w < 0.0):
        raise ValueError(
            "budget weights must be non-negative: w_p = (1 - C_p) * dN_exp "
            "with C in [0, 1] and dN_exp >= 0.")
    wsum = np.sum(w, axis=0)                                    # (n_grid,)
    ok = wsum > 0.0
    safe_wsum = np.where(ok, wsum, 1.0)
    # exp(logq) is bounded by the builder's logq_clip (|logQ| <= ~7), so the
    # weighted sum neither overflows nor underflows to an all-zero bin.
    mono = np.sum(w * np.exp(lq), axis=-2) / safe_wsum          # (..., n_grid)
    log_mono = np.where(ok, np.log(np.where(mono > 0.0, mono, 1.0)), 0.0)
    return lq - log_mono[..., None, :], log_mono


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
    max_newton: int = 100,
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
    prior ``I``); falls back to L-BFGS-B if Newton stalls.  The gradient/Hessian
    are KKT-projected at the ``field_clip`` boundary: a voxel pinned at the clip
    and pressing outward contributes nothing (its clipped field is locally
    constant in ``xi``), while one pulling back inward keeps its pull so iterates
    can re-enter.  This keeps gradient, Hessian, and objective describing the
    same clipped function (no line-search stall against phantom pull from
    saturated voxels) and lets a legitimate kink minimum — data demanding a field
    beyond the clip, e.g. ``N_obs >> base`` — report ``converged``.

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
        raw = Phi @ xi
        fld = np.clip(raw, -field_clip, field_clip)
        return raw, fld, np.where(mask, np.exp(logbase + b * fld - shift), 0.0)

    def _objective(xi):
        _, fld, lam = _lam(xi)
        loglam = logbase + b * fld - shift
        data = float(np.sum(lam - N_obs * np.where(mask, loglam, 0.0)))
        return 0.5 * float(np.dot(xi, xi)) + data

    # Voxels pinned at the field clip and pressing OUTWARD contribute nothing to
    # d(objective)/d(xi) (the clipped field is locally constant in xi there);
    # voxels at the clip pulling back INWARD keep their pull so iterates can
    # re-enter. Without this KKT projection the gradient describes a different
    # function than the objective, the Armijo test can never be satisfied, and
    # the line search stalls — and a legitimate kink minimum (a voxel the data
    # push beyond the clip, e.g. N_obs >> base) could never report converged.
    # Boundary tolerance for the KKT test: a kink minimizer is approached from
    # strictly inside the clip in floating point, where the outward pull is
    # still fully counted; treating a 1e-6 field-unit sliver as "at the clip"
    # is physically nothing (the output table is clipped to logq_clip anyway).
    clip_eps = 1e-6

    def _kkt_residual(raw, lam):
        w = np.where(mask, lam - N_obs, 0.0)
        pinned = (((raw >= field_clip - clip_eps) & (w < 0.0))
                  | ((raw <= -(field_clip - clip_eps)) & (w > 0.0)))
        return np.where(pinned, 0.0, w), pinned

    def _grad(xi):
        raw, _, lam = _lam(xi)
        w, _ = _kkt_residual(raw, lam)
        return xi + b * (Phi.T @ w)

    # Scale-aware gradient tolerance: the Poisson data gradient scales with the
    # observed counts, so convergence is judged on the gradient inf-norm relative
    # to the problem scale (``tol`` is the relative tolerance).
    gscale = max(1.0, b * float(np.max(np.abs(N_obs)))) if N_obs.size else 1.0
    gtol = float(tol) * gscale

    xi = np.zeros(M, dtype=float)
    n_iter = 0
    for n_iter in range(1, int(max_newton) + 1):
        raw, _, lam = _lam(xi)
        w, pinned = _kkt_residual(raw, lam)
        g = xi + b * (Phi.T @ w)
        gnorm = float(np.max(np.abs(g))) if M else 0.0
        if gnorm < gtol:
            break
        W = Phi * (np.where(mask & ~pinned, lam, 0.0) * (b * b))[:, None]    # (V, M)
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
    gnorm_raw = gnorm
    n_pin = 0
    if gnorm >= gtol:
        # KKT stationarity at a kink minimum: with voxels pinned at the field
        # clip, the projected gradient need not vanish — the kept residual must
        # be BALANCED by bounded multipliers on the pinned voxels' feature rows,
        # g = sum_p c_p phi_p with c_p between 0 and -bias*w_p (the dropped
        # outward pull). Complementary slackness: only voxels AT the boundary
        # (within clip_eps) admit a multiplier — a voxel strictly beyond the
        # clip has a locally flat objective and its subdifferential is {0}, so
        # granting it a multiplier would certify non-stationary points as
        # converged (module review of the fix, finding 1). Fit by bounded least
        # squares and judge convergence on what the multipliers cannot absorb.
        # Skipped for absurdly large boundary sets (grossly inconsistent inputs
        # SHOULD read unconverged).
        raw, _, lam = _lam(xi)
        w_all = np.where(mask, lam - N_obs, 0.0)
        _, pinned = _kkt_residual(raw, lam)
        at_bnd = pinned & (np.abs(raw) <= field_clip + clip_eps)
        n_pin = int(np.sum(pinned))
        n_bnd = int(np.sum(at_bnd))
        if 0 < n_bnd <= 4 * M:
            A = Phi[at_bnd].T                               # (M, P)
            cb = -b * w_all[at_bnd]
            lb, ub = np.minimum(0.0, cb), np.maximum(0.0, cb)
            try:
                fit = optimize.lsq_linear(A, g, bounds=(lb, ub))
                gnorm = float(np.max(np.abs(g - A @ fit.x)))
            except Exception:                               # pragma: no cover
                pass                                        # keep raw gnorm
    converged = bool(gnorm < gtol)

    # Hessian Cholesky at the MAP (feeds the Laplace ensemble); clip-pinned
    # voxels carry no curvature, same masking as the solve.
    raw, _, lam = _lam(xi)
    _, pinned = _kkt_residual(raw, lam)
    W = Phi * (np.where(mask & ~pinned, lam, 0.0) * (b * b))[:, None]
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
            # grad_inf is what `converged` was judged on: the projected-gradient
            # inf-norm after absorbing the bounded boundary-KKT multipliers (it
            # equals grad_inf_raw whenever no voxel sat at the clip boundary).
            "grad_inf": float(gnorm),
            "grad_inf_raw": float(gnorm_raw),
            "n_pinned": int(n_pin),
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
    realization_set_id: str | None = None,
    budget_renormalized: bool | None = None,
    budget_monopole_logq: np.ndarray | None = None,
    c_mode: str | None = None,
) -> str:
    """Write a completion file with layout ``/lss_completion/{logq_map,logq_members,zgrid}``.

    ``budget_renormalized`` records whether the tables were per-z mean-one
    renormalized under the missing-budget weights ``(1 - C) dN_exp`` (see
    :func:`renormalize_q_mean_one`), and ``budget_monopole_logq`` is the
    removed per-z monopole curve of the MAP table.  ``None`` (the default)
    writes NO stamp — the loader then treats the file as a legacy,
    non-renormalized table and warns loudly.  Stamping ``True`` REQUIRES the
    removed curve (fail-closed provenance: a table claimed mean-one without
    the record of what was removed cannot be audited).

    ``c_mode`` records which completeness estimator the fit was residual to:
    ``"per_pixel"`` (the per-pixel matched-kernel base ``C_p dN_exp``) or
    ``"aggregate"`` (the sky-aggregate base ``Cbar dN_exp``, empty pixels
    included as N_obs = 0 rows).  ``None`` writes no attr, which the loader
    reads as legacy ``per_pixel``.  The two targets differ by the entire
    clustering signal, so the loader hard-errors when a table's ``c_mode``
    does not match the consuming survey's.

    A ``realization_set_id`` (a fresh ``uuid4().hex`` unless one is supplied)
    is stamped on the ``/lss_completion`` group so a K>=2 mixture can verify
    that per-catalog ensembles share ONE matched set of LSS realizations
    before it marginalizes over a shared member index (see
    ``load_multitracer_catalog_bundles``).  Pass an explicit
    ``realization_set_id`` from a future JOINT multi-survey builder to stamp
    the SAME id across several files.  When members are present the group also
    carries a ``member_content_sha256`` (over the exact member bytes) and an
    integer ``n_members`` for provenance.

    Fail-closed contract: a table with non-finite entries is never written (a
    NaN cell would silently poison the missing-density term of every event in
    that pixel), and the file appears at ``path`` ATOMICALLY -- content is
    written to a sibling temp file and renamed only after a successful flush,
    so a died build can never leave a truncated artifact that a later
    inference run trusts.
    """
    import h5py

    if logq_map is None and logq_members is None:
        raise ValueError("save_lss_completion_hdf5 needs logq_map and/or logq_members.")
    if indexing not in ("compact", "global"):
        raise ValueError(f"indexing must be 'compact' or 'global', got {indexing!r}.")
    for name, arr in (("logq_map", logq_map), ("logq_members", logq_members)):
        if arr is not None:
            arr = np.asarray(arr, dtype=float)
            n_bad = int(np.sum(~np.isfinite(arr)))
            if n_bad:
                raise ValueError(
                    f"refusing to save {name} with {n_bad} non-finite entries: "
                    "a NaN/inf logQ cell poisons the missing-density term for "
                    "every event in its pixel. Fix the build (raise --maxiter, "
                    "check the catalog) instead of persisting the artifact."
                )
    if completion_kind is None:
        completion_kind = "laplace_members" if logq_members is not None else "map"
    if realization_set_id is None:
        realization_set_id = uuid.uuid4().hex
    if budget_renormalized and budget_monopole_logq is None:
        raise ValueError(
            "budget_renormalized=True requires budget_monopole_logq: the "
            "removed per-z monopole curve is the provenance record of what "
            "the renormalization took out of Q; a table stamped mean-one "
            "without it cannot be audited. Pass the curve returned by "
            "renormalize_q_mean_one."
        )
    if budget_monopole_logq is not None:
        budget_monopole_logq = np.asarray(budget_monopole_logq, dtype=float)
        n_bad = int(np.sum(~np.isfinite(budget_monopole_logq)))
        if n_bad:
            raise ValueError(
                f"refusing to save budget_monopole_logq with {n_bad} "
                "non-finite entries: a NaN/inf monopole means the "
                "renormalization itself was fed a poisoned table. Fix the "
                "build instead of persisting the artifact."
            )
    if c_mode is not None and c_mode not in ("per_pixel", "aggregate", "selection"):
        raise ValueError(
            f"c_mode must be 'per_pixel', 'aggregate' or 'selection', got {c_mode!r}."
        )

    tmp_path = f"{path}.tmp-{os.getpid()}"
    try:
        with h5py.File(tmp_path, "w") as f:
            grp = f.create_group("lss_completion")
            if logq_map is not None:
                grp.create_dataset("logq_map", data=np.asarray(logq_map, dtype=float))
            if logq_members is not None:
                members_arr = np.asarray(logq_members, dtype=float)
                grp.create_dataset("logq_members", data=members_arr)
            if zgrid is not None:
                grp.create_dataset("zgrid", data=np.asarray(zgrid, dtype=float))
            grp.attrs["indexing"] = indexing
            grp.attrs["model"] = "poisson_lognormal"
            grp.attrs["completion_kind"] = completion_kind
            grp.attrs["created_by"] = "darksirens.redshift.lognormal_completion"
            grp.attrs["realization_set_id"] = realization_set_id
            if budget_renormalized is not None:
                grp.attrs["budget_renormalized"] = bool(budget_renormalized)
            if budget_monopole_logq is not None:
                grp.attrs["budget_monopole_logq"] = budget_monopole_logq
            if c_mode is not None:
                grp.attrs["c_mode"] = c_mode
            if logq_members is not None:
                grp.attrs["member_content_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(members_arr).tobytes()
                ).hexdigest()
                grp.attrs["n_members"] = int(members_arr.shape[0])
            if metadata:
                grp.attrs["diagnostics"] = json.dumps(metadata, default=str)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return path


def load_lss_completion_hdf5(path: str) -> dict:
    """Read a completion file written by :func:`save_lss_completion_hdf5`.

    Returns a dict with ``logq_map`` / ``logq_members`` / ``zgrid`` (any may be
    ``None``) plus ``indexing``, ``model``, ``completion_kind`` and parsed
    ``diagnostics``.  The ensemble-provenance attrs ``realization_set_id`` /
    ``member_content_sha256`` / ``n_members`` are returned as ``None`` when
    absent (legacy files written before provenance stamping).  The budget
    attrs ``budget_renormalized`` / ``budget_monopole_logq`` follow the same
    convention — ``None`` when absent — but an absent ``budget_renormalized``
    additionally draws a loud warning: the table's Q then rescales the missing
    budget (Jensen inflation) instead of only redistributing it.  ``c_mode``
    is likewise ``None`` when absent, which consumers must read as legacy
    ``"per_pixel"`` (every table written before the aggregate mode existed was
    fit residual to the per-pixel base).

    Validation: non-finite table content is rejected outright (one NaN cell
    poisons every event in its pixel).  Convergence diagnostics are surfaced
    as loud warnings rather than errors: a NEW-STYLE file (one carrying the
    builder's ``allow_unconverged`` stamp) can only be unconverged through
    that explicit, recorded override, and a LEGACY file's ``n_converged``
    counter is unreliable -- the builder never set L-BFGS-B's ``maxfun``, so
    scipy's default 15000-evaluation cap stopped large-``maxiter`` solves
    with ``success=False`` near their fixed point, reading 0 converged for
    tables that were in fact essentially converged.
    """
    import h5py

    out = {"logq_map": None, "logq_members": None, "zgrid": None,
           "indexing": "compact", "model": None, "completion_kind": None,
           "diagnostics": None, "realization_set_id": None,
           "member_content_sha256": None, "n_members": None,
           "budget_renormalized": None, "budget_monopole_logq": None,
           "c_mode": None}
    with h5py.File(path, "r") as f:
        grp = f["lss_completion"] if "lss_completion" in f else f
        if "logq_map" in grp:
            out["logq_map"] = np.asarray(grp["logq_map"])
        if "logq_members" in grp:
            out["logq_members"] = np.asarray(grp["logq_members"])
        if "zgrid" in grp:
            out["zgrid"] = np.asarray(grp["zgrid"])
        for key in ("indexing", "model", "completion_kind",
                    "realization_set_id", "member_content_sha256", "c_mode"):
            if key in grp.attrs:
                val = grp.attrs[key]
                out[key] = val.decode() if isinstance(val, bytes) else str(val)
        if "n_members" in grp.attrs:
            out["n_members"] = int(grp.attrs["n_members"])
        if "budget_renormalized" in grp.attrs:
            out["budget_renormalized"] = bool(grp.attrs["budget_renormalized"])
        if "budget_monopole_logq" in grp.attrs:
            out["budget_monopole_logq"] = np.asarray(
                grp.attrs["budget_monopole_logq"], dtype=float)
        if "diagnostics" in grp.attrs:
            raw = grp.attrs["diagnostics"]
            raw = raw.decode() if isinstance(raw, bytes) else raw
            try:
                out["diagnostics"] = json.loads(raw)
            except Exception:
                out["diagnostics"] = raw

    for name in ("logq_map", "logq_members"):
        arr = out[name]
        if arr is not None and not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(np.asarray(arr))))
            raise ValueError(
                f"LSS completion '{path}' has {n_bad} non-finite {name} "
                "entries; a NaN/inf logQ cell poisons the missing-density "
                "term for every event in its pixel. Rebuild the completion."
            )

    if out["budget_renormalized"] is None:
        # Same tolerate-but-warn policy as the legacy convergence counters: a
        # verified-good table keeps loading while the operator sees the flag.
        warnings.warn(
            f"LSS completion '{path}' carries no 'budget_renormalized' stamp "
            "(legacy table, or a builder that does not renormalize): its Q "
            "was NOT renormalized to the per-z mean-one budget convention, so "
            "the total missing budget can be RESCALED by Q's per-z monopole "
            "(measured +55% Jensen inflation for radial tables) instead of "
            "only redistributed. Rebuild with the current "
            "darksirens_build_lognormal_completion, which renormalizes by "
            "default.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Post-renorm clip-railing provenance (deferred review MINOR): a shipped
    # cell at/beyond the solve's |logQ| clip means the CLIP, not the data,
    # chose that Q value.  Sparse railing is tolerable (the clip exists as a
    # numerical wall); a non-negligible fraction means the solve saturated
    # and the table should be rebuilt with a wider clip or more iterations.
    _diag = out.get("diagnostics")
    if isinstance(_diag, dict):
        _rail = _diag.get("post_renorm_logq_rail_frac")
        if _rail is not None and float(_rail) > 1e-3:
            warnings.warn(
                f"LSS completion '{path}': {float(_rail):.2%} of the fitted "
                f"post-renorm logQ cells rail at/beyond the solve's "
                f"|logQ| <= {_diag.get('logq_clip', 7.0)} clip (abs max "
                f"{_diag.get('post_renorm_logq_abs_max')}): those Q values "
                "were chosen by the clip, not the data. Rebuild with more "
                "--maxiter (or inspect the catalog) if this footprint "
                "matters.",
                RuntimeWarning,
                stacklevel=2,
            )

    _warn_if_unconverged(path, out.get("diagnostics"))
    return out


def _warn_if_unconverged(path: str, diag) -> None:
    """Surface builder convergence diagnostics at load time, loudly.

    New-style files (builder stamped ``allow_unconverged``) can only be
    unconverged through that explicit override -- warn that a research
    ablation artifact is entering inference.  Substituted non-finite cells
    (``n_nonfinite_substituted``, only stampable under the same override) are
    treated identically: the eval degraded even when the solve itself
    converged, and the builder also skips the budget renormalization for such
    tables, so a Jensen-inflated Q must not circulate silently.  Legacy files'
    counters are unreliable (see load_lss_completion_hdf5's docstring): warn
    without rejecting, and say why the counter cannot be trusted, so a
    verified-good production table keeps loading while the operator still
    sees the flag.
    """
    if not isinstance(diag, dict):
        return
    converged = diag.get("converged")
    n_conv, n_occ = diag.get("n_converged"), diag.get("n_occupied")
    n_sub = int(diag.get("n_nonfinite_substituted") or 0)
    looks_unconverged = (converged is False) or (
        converged is None
        and n_conv is not None and n_occ and int(n_conv) < int(n_occ)
    ) or n_sub > 0
    if not looks_unconverged:
        return
    if "allow_unconverged" in diag:
        if diag.get("allow_unconverged"):
            parts = []
            if n_sub == 0 or converged is False:
                parts.append("unconverged solve")
            if n_sub:
                parts.append(
                    f"{n_sub} non-finite logQ cells substituted with Q = 1, "
                    "which also skips the budget renormalization"
                )
            warnings.warn(
                f"LSS completion '{path}' was built under the explicit "
                f"--allow-unconverged override ({'; '.join(parts)}). "
                "Research ablation artifact; do not use it for a production "
                "result.",
                RuntimeWarning,
                stacklevel=3,
            )
        return
    warnings.warn(
        f"LSS completion '{path}' reports unconverged diagnostics "
        f"(converged={converged!r}, n_converged={n_conv!r}/{n_occ!r}). This "
        "file predates honest convergence accounting: its builder never set "
        "L-BFGS-B's maxfun, so scipy's 15000-evaluation default stopped "
        "large-maxiter solves with success=False near their fixed point and "
        "the counter reads unconverged for tables that may be fine. Validate "
        "the table (darksirens_diagnose_lognormal_completion) or rebuild it "
        "with the current builder.",
        RuntimeWarning,
        stacklevel=3,
    )
