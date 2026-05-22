"""
wlmagnification.py
------------------
Weak-lensing magnification PDF ``p_WL(μ | z)``.

Two backends
~~~~~~~~~~~~
``backend = 0`` — lognormal
    p_WL(μ | z) = (1/μ) · N(ln μ | m(z), s²(z))

    with the mean ``m(z) = -s²(z)/2`` chosen so that ``⟨μ⟩ = 1`` (flux
    conservation). The variance is parameterized as

        s²(z) = a · z^b

    Default constants ``a = 4×10⁻³, b = 1.5`` match Takahashi+11
    (arXiv:1106.3823) and Hilbert+11 (arXiv:1107.5663) ray-tracing
    PDFs for BBH-relevant source redshifts to better than 10% on
    μ ∈ [0.5, 2], z ∈ [0.1, 2]. For higher-fidelity work at z > 2,
    use the tabulated backend with a precomputed ray-traced grid.

``backend = 1`` — tabulated
    Bilinear interpolation in ``(z, ln μ)`` of a precomputed table
    of ``log p_WL``. The user supplies the table on a regular grid.

JIT contract
~~~~~~~~~~~~
``log_p_wl(mu, z, params)`` is JIT-compiled, vmappable, and returns
arrays of the same shape as ``mu`` and ``z`` (which must broadcast).
The dispatcher branches on the static integer ``params.backend`` —
the branch is selected at trace time, no runtime cost.

References
~~~~~~~~~~
- Holz & Wald (1998), PRD 58, 063501.
- Takahashi, Oguri, Sato, Hamana (2011), ApJ 742, 15.
- Hilbert, Hartlap, Schneider (2011), A&A 536, A85.
"""

from __future__ import annotations

from typing import NamedTuple, Any

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats import norm


# Sentinel grid for the lognormal backend (never read; required for static
# pytree shape).
_DUMMY_GRID_1D = jnp.zeros(2, dtype=jnp.float64)
_DUMMY_GRID_2D = jnp.zeros((2, 2), dtype=jnp.float64)


class WLParams(NamedTuple):
    """JAX-compatible container for weak-lensing PDF parameters.

    Fields
    ------
    backend
        0 = lognormal, 1 = tabulated. **Static** under JIT (compile-time
        constant — flipping it triggers retrace).
    a, b
        Lognormal variance parameters s²(z) = a · z^b. Both broadcast-ok.
    z_grid, log_mu_grid, log_p_table
        Tabulated arrays. Ignored for backend = 0 (pass dummies).
    """

    backend: Any           # int (0 or 1) — static under JIT
    a: Any                 # float
    b: Any                 # float
    z_grid: Any            # (Nz,)
    log_mu_grid: Any       # (Nmu,)
    log_p_table: Any       # (Nz, Nmu)  — log p_WL on (z, ln μ) grid


def make_lognormal_wl_params(a: float = 4.0e-3, b: float = 1.5) -> WLParams:
    """Lognormal WL parameters with default Takahashi+11-calibrated constants.

    Parameters
    ----------
    a, b
        s²(z) = a · z^b. Defaults are reasonable on z ∈ [0.1, 2].
    """
    return WLParams(
        backend=0,
        a=jnp.asarray(a, dtype=jnp.float64),
        b=jnp.asarray(b, dtype=jnp.float64),
        z_grid=_DUMMY_GRID_1D,
        log_mu_grid=_DUMMY_GRID_1D,
        log_p_table=_DUMMY_GRID_2D,
    )


def make_tabulated_wl_params(
    z_grid: jnp.ndarray,
    log_mu_grid: jnp.ndarray,
    log_p_table: jnp.ndarray,
) -> WLParams:
    """Tabulated WL parameters from a user-supplied log-p table.

    Parameters
    ----------
    z_grid : (Nz,)
        Strictly increasing source redshifts.
    log_mu_grid : (Nmu,)
        Strictly increasing values of ln μ.
    log_p_table : (Nz, Nmu)
        log p_WL(μ | z) on the (z, ln μ) grid. **Note**: this is the PDF
        in μ, evaluated at the gridded ln μ.  The interpolation is done
        in (z, ln μ) space because p_WL is smoother as a function of ln μ.
    """
    z_grid = jnp.asarray(z_grid, dtype=jnp.float64)
    log_mu_grid = jnp.asarray(log_mu_grid, dtype=jnp.float64)
    log_p_table = jnp.asarray(log_p_table, dtype=jnp.float64)

    if z_grid.ndim != 1:
        raise ValueError(f"z_grid must be 1D, got shape {z_grid.shape}.")
    if log_mu_grid.ndim != 1:
        raise ValueError(f"log_mu_grid must be 1D, got shape {log_mu_grid.shape}.")
    if log_p_table.shape != (z_grid.shape[0], log_mu_grid.shape[0]):
        raise ValueError(
            f"log_p_table shape {log_p_table.shape} must equal "
            f"(len(z_grid), len(log_mu_grid)) = "
            f"({z_grid.shape[0]}, {log_mu_grid.shape[0]})."
        )

    return WLParams(
        backend=1,
        a=jnp.asarray(0.0, dtype=jnp.float64),
        b=jnp.asarray(0.0, dtype=jnp.float64),
        z_grid=z_grid,
        log_mu_grid=log_mu_grid,
        log_p_table=log_p_table,
    )


# ============================================================================
# Lognormal backend
# ============================================================================

@jit
def _log_p_wl_lognormal(mu: jnp.ndarray, z: jnp.ndarray, p: WLParams) -> jnp.ndarray:
    """Lognormal WL log-PDF with mean μ = 1.

    Math:
        s²(z) = a · z^b
        m(z)  = -s²(z) / 2    (so that ⟨μ⟩ = 1)
        p_WL(μ | z) = (1/μ) · N(ln μ | m(z), s²(z))

    The clip ``z ≥ 1e-3`` prevents the variance from diverging at z = 0;
    for z below the clip the PDF effectively becomes a delta at μ = 1.
    """
    z_safe = jnp.maximum(z, 1.0e-3)
    s2 = p.a * jnp.power(z_safe, p.b)
    s = jnp.sqrt(s2)
    m = -0.5 * s2

    log_mu = jnp.log(mu)
    log_pdf_lnmu = norm.logpdf(log_mu, loc=m, scale=s)
    return log_pdf_lnmu - log_mu


# ============================================================================
# Tabulated backend
# ============================================================================

def _bilinear_interp(
    z: jnp.ndarray,
    log_mu: jnp.ndarray,
    z_grid: jnp.ndarray,
    log_mu_grid: jnp.ndarray,
    table: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinear interpolation of ``table`` at points ``(z, log_mu)``.

    Out-of-range queries are clamped to the grid edges (constant
    extrapolation). For weak-lensing tables the grid should be wide
    enough that this is harmless; the caller is responsible.
    """
    nz = z_grid.shape[0]
    nm = log_mu_grid.shape[0]

    iz = jnp.clip(jnp.searchsorted(z_grid, z) - 1, 0, nz - 2)
    im = jnp.clip(jnp.searchsorted(log_mu_grid, log_mu) - 1, 0, nm - 2)

    z0 = z_grid[iz]
    z1 = z_grid[iz + 1]
    m0 = log_mu_grid[im]
    m1 = log_mu_grid[im + 1]

    # Avoid 0/0 at duplicate grid points (shouldn't happen for strictly
    # increasing grids, but be defensive).
    tz = jnp.where(z1 > z0, (z - z0) / (z1 - z0), 0.0)
    tm = jnp.where(m1 > m0, (log_mu - m0) / (m1 - m0), 0.0)

    f00 = table[iz, im]
    f10 = table[iz + 1, im]
    f01 = table[iz, im + 1]
    f11 = table[iz + 1, im + 1]

    f0 = f00 * (1.0 - tm) + f01 * tm
    f1 = f10 * (1.0 - tm) + f11 * tm
    return f0 * (1.0 - tz) + f1 * tz


@jit
def _log_p_wl_tabulated(mu: jnp.ndarray, z: jnp.ndarray, p: WLParams) -> jnp.ndarray:
    """Bilinear interp in (z, ln μ) of the supplied log-p table."""
    log_mu = jnp.log(mu)
    # vmap the scalar bilinear interp over the flattened input
    flat_z = z.reshape(-1)
    flat_lm = log_mu.reshape(-1)
    interp = vmap(
        lambda zi, lmi: _bilinear_interp(zi, lmi, p.z_grid, p.log_mu_grid, p.log_p_table)
    )
    return interp(flat_z, flat_lm).reshape(mu.shape)


# ============================================================================
# Dispatcher
# ============================================================================

def log_p_wl(mu: jnp.ndarray, z: jnp.ndarray, p: WLParams) -> jnp.ndarray:
    """Log of the weak-lensing magnification PDF, dispatched on backend.

    Parameters
    ----------
    mu
        Magnification values, ``μ > 0``. Broadcastable with ``z``.
    z
        Source redshift. Broadcastable with ``mu``.
    p
        ``WLParams`` from ``make_lognormal_wl_params`` or
        ``make_tabulated_wl_params``.

    Returns
    -------
    log_p : array
        ``log p_WL(μ | z)``, same shape as the broadcast of ``mu`` and ``z``.

    Notes
    -----
    The dispatch is a Python conditional on ``p.backend`` (a static integer),
    so each backend compiles to its own JIT trace. No runtime branching.
    """
    if int(p.backend) == 0:
        return _log_p_wl_lognormal(mu, z, p)
    elif int(p.backend) == 1:
        return _log_p_wl_tabulated(mu, z, p)
    else:
        raise ValueError(
            f"Unknown WL backend {p.backend}. Use 0 (lognormal) or 1 (tabulated)."
        )
