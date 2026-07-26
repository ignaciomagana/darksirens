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

import warnings
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


def _interp_table_broadcast(
    mu: jnp.ndarray,
    z: jnp.ndarray,
    z_grid: jnp.ndarray,
    log_mu_grid: jnp.ndarray,
    log_p_table: jnp.ndarray,
) -> jnp.ndarray:
    """Bilinear interp of ``log_p_table`` at the broadcast of ``(mu, z)``.

    ``mu`` and ``z`` are broadcast against each other FIRST (per the module's
    documented contract) and only then flattened for the vmap.  Flattening the
    two independently — as this did before — raised
    ``vmap got inconsistent sizes`` for every shape pair that was merely
    broadcast-compatible rather than identical, which is exactly the pattern
    the production caller uses: ``wl_weight.log_sample_weight_wl_marginalized``
    passes ``mu`` of shape (Nmu,) against ``z`` of shape (nsamp, Nmu).
    """
    mu_b, z_b = jnp.broadcast_arrays(
        jnp.asarray(mu, dtype=jnp.float64), jnp.asarray(z, dtype=jnp.float64)
    )
    flat_lm = jnp.log(mu_b).reshape(-1)
    flat_z = z_b.reshape(-1)
    interp = vmap(
        lambda zi, lmi: _bilinear_interp(zi, lmi, z_grid, log_mu_grid, log_p_table)
    )
    return interp(flat_z, flat_lm).reshape(mu_b.shape)


@jit
def _log_p_wl_tabulated(mu: jnp.ndarray, z: jnp.ndarray, p: WLParams) -> jnp.ndarray:
    """Bilinear interp in (z, ln μ) of the supplied log-p table."""
    return _interp_table_broadcast(
        mu, z, p.z_grid, p.log_mu_grid, p.log_p_table
    )


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


# ============================================================================
# JIT-friendly closure factories (commit 2)
# ============================================================================
#
# The ``log_p_wl(mu, z, p)`` dispatcher above branches on ``int(p.backend)``,
# which raises ``ConcretizationTypeError`` if ``p`` is a tracer inside ``jit``.
# These factory functions build closures that bake the backend choice in at
# Python level, so the returned callable is fully JIT-friendly. They match
# the architectural pattern used by ``pop_model_parser`` and
# ``get_redshift_prior`` in the rest of darksirens.
#
# Use these inside any JIT-compiled inference path. The free functions above
# remain available for tests, diagnostics, and any code that runs outside JIT.


def make_lognormal_log_p_wl(a: jnp.ndarray, b: jnp.ndarray):
    """Build a JIT-friendly lognormal log-PDF closure.

    Returns a callable ``log_p_wl_fn(mu, z) -> log p_WL(μ | z)`` with the
    lognormal parameters baked in as traced arrays. The closure can be
    called from inside JIT'd code without dispatch issues.

    Parameters
    ----------
    a, b
        Variance schedule s²(z) = a · z^b. Both passed as JAX scalars
        (will be auto-converted from Python floats).

    Notes
    -----
    The closure clips z ≥ 1e-3 inside to prevent the variance vanishing
    at z = 0 (which would produce a degenerate PDF). This is identical
    behaviour to ``_log_p_wl_lognormal``.
    """
    a_jx = jnp.asarray(a, dtype=jnp.float64)
    b_jx = jnp.asarray(b, dtype=jnp.float64)

    def log_p_wl_fn(mu: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        z_safe = jnp.maximum(z, 1.0e-3)
        s2 = a_jx * jnp.power(z_safe, b_jx)
        s = jnp.sqrt(s2)
        m = -0.5 * s2
        log_mu = jnp.log(mu)
        return norm.logpdf(log_mu, loc=m, scale=s) - log_mu

    return log_p_wl_fn


def make_tabulated_log_p_wl(
    z_grid: jnp.ndarray,
    log_mu_grid: jnp.ndarray,
    log_p_table: jnp.ndarray,
):
    """Build a JIT-friendly tabulated log-PDF closure.

    Returns a callable ``log_p_wl_fn(mu, z) -> log p_WL(μ | z)`` with the
    interpolation table baked in.
    """
    z_grid = jnp.asarray(z_grid, dtype=jnp.float64)
    log_mu_grid = jnp.asarray(log_mu_grid, dtype=jnp.float64)
    log_p_table = jnp.asarray(log_p_table, dtype=jnp.float64)

    # Known limitation (library review, lensing finding): a coarse log_mu_grid
    # cannot resolve the narrow low-z p(mu|z) of the lognormal WL model, so the
    # per-event mu-marginal integrates to ~0 for z <~ 0.3 and those events are
    # silently dropped. The CLI DOES expose this backend
    # (--wl_backend tabulated), so the crude node-count warning below is
    # backed by the hard startup check in ``validate_wl_mu_quadrature``, which
    # the lensing CLI runs on the loaded table against the very mu-quadrature
    # the likelihood integrates on.
    if int(log_mu_grid.shape[-1]) < 64:
        warnings.warn(
            f"tabulated WL backend: log_mu_grid has {int(log_mu_grid.shape[-1])} "
            "nodes, which under-resolves the narrow low-z p(mu|z) and can "
            "silently drop z<~0.3 events (int p(mu|z) dmu ~ 0). Refine the "
            "mu-grid or use the lognormal (Gauss-Hermite) WL backend, and "
            "verify int p(mu|z) dmu ~ 1 across your z-grid.",
            RuntimeWarning,
            stacklevel=2,
        )

    def log_p_wl_fn(mu: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        return _interp_table_broadcast(mu, z, z_grid, log_mu_grid, log_p_table)

    return log_p_wl_fn


def make_log_p_wl_from_params(p: WLParams):
    """Build a JIT-friendly closure from a ``WLParams`` container.

    Convenience adapter: inspects ``p.backend`` at Python level (so the
    branch is resolved before tracing) and delegates to the appropriate
    factory.
    """
    backend = int(p.backend)
    if backend == 0:
        return make_lognormal_log_p_wl(p.a, p.b)
    elif backend == 1:
        return make_tabulated_log_p_wl(p.z_grid, p.log_mu_grid, p.log_p_table)
    else:
        raise ValueError(
            f"Unknown WL backend {backend}. Use 0 (lognormal) or 1 (tabulated)."
        )


# ============================================================================
# Startup validation of the μ-quadrature against a tabulated table
# ============================================================================
#
# The tabulated backend's mu-marginal is evaluated on a FIXED Gauss-Legendre
# grid in ln mu (``grids.make_wl_mu_quadrature``), not on the table's own
# log_mu_grid.  If the table's p_WL(mu | z) is narrower than that grid's node
# spacing, the marginal
#
#     I(z) = int p_WL(mu | z) dmu ~= sum_l w_l mu_l p_WL(mu_l | z)
#
# silently collapses towards 0 and the affected events are DELETED from the
# likelihood with no error.  The default 16-node grid on ln mu in (-0.6, 0.6)
# gives I = 2e-5 at z = 0.1 and 5e-2 at z = 0.2 under the default lognormal
# (a = 4e-3, b = 1.5).  Validate up front and fail loudly rather than
# rescaling: a table whose marginal is off by a z-dependent factor is not
# a PDF the sampler can be trusted with.


def wl_mu_quadrature_coverage(
    mu_nodes: jnp.ndarray,
    log_w_nodes: jnp.ndarray,
    z_grid: jnp.ndarray,
    log_mu_grid: jnp.ndarray,
    log_p_table: jnp.ndarray,
    z_test: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """``∫ p_WL(μ|z) dμ`` on the supplied μ-quadrature, one value per test z.

    Parameters
    ----------
    mu_nodes, log_w_nodes
        The quadrature the likelihood integrates on, from
        ``darksirens.lensing.grids.make_wl_mu_quadrature`` (nodes in μ,
        log-weights for the ``ln μ`` interval — the extra ``ln μ`` from
        ``dμ = μ d(ln μ)`` is added here).
    z_grid, log_mu_grid, log_p_table
        The tabulated backend's grids, as passed to
        ``make_tabulated_log_p_wl``.
    z_test
        Redshifts at which to check the marginal. Defaults to ``z_grid``.

    Returns
    -------
    coverage : (Nz_test,) array
        ``∫ p_WL(μ|z) dμ``; should be ≈ 1 at every test redshift.
    """
    mu_nodes = jnp.asarray(mu_nodes, dtype=jnp.float64)
    log_w_nodes = jnp.asarray(log_w_nodes, dtype=jnp.float64)
    z_test = jnp.asarray(z_grid if z_test is None else z_test, dtype=jnp.float64)
    log_p = _interp_table_broadcast(
        mu_nodes[None, :], z_test[:, None], z_grid, log_mu_grid, log_p_table,
    )                                                       # (Nz_test, Nmu)
    log_integrand = log_w_nodes[None, :] + log_p + jnp.log(mu_nodes)[None, :]
    return jnp.sum(jnp.exp(log_integrand), axis=-1)


def validate_wl_mu_quadrature(
    mu_nodes: jnp.ndarray,
    log_w_nodes: jnp.ndarray,
    z_grid: jnp.ndarray,
    log_mu_grid: jnp.ndarray,
    log_p_table: jnp.ndarray,
    z_test: jnp.ndarray | None = None,
    tol: float = 0.02,
    context: str = "tabulated WL backend",
) -> jnp.ndarray:
    """Raise ``ValueError`` unless the μ-quadrature resolves the table.

    Checks ``|∫ p_WL(μ|z) dμ - 1| <= tol`` at every test redshift. The failure
    message names the worst redshift and its marginal so the caller can either
    refine the table's μ-support or switch to the lognormal (Gauss-Hermite)
    backend. Returns the coverage array on success.
    """
    coverage = wl_mu_quadrature_coverage(
        mu_nodes, log_w_nodes, z_grid, log_mu_grid, log_p_table, z_test=z_test,
    )
    z_checked = jnp.asarray(z_grid if z_test is None else z_test, dtype=jnp.float64)
    err = jnp.abs(coverage - 1.0)
    worst = int(jnp.argmax(err))
    if not bool(jnp.all(jnp.isfinite(coverage))) or float(err[worst]) > float(tol):
        raise ValueError(
            f"{context}: the mu-quadrature does not resolve p_WL(mu|z). "
            f"int p_WL(mu|z) dmu = {float(coverage[worst]):.6g} at "
            f"z = {float(z_checked[worst]):.4g} (tolerance |I - 1| <= {float(tol):g}; "
            f"{int(jnp.sum(err > tol))}/{int(err.shape[0])} test redshifts fail). "
            "Events at those redshifts would be silently deleted from the "
            "likelihood. Widen/refine the table's log_mu support so the "
            "quadrature interval brackets it, or use --wl_backend lognormal "
            "(Gauss-Hermite in the standardized variable, resolution-independent)."
        )
    return coverage
