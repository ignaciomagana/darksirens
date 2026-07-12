"""
sampling.py
-----------
Grid-based inverse-CDF samplers for the population model, used by the
flow-surrogate likelihood path.

The flow path flips the per-event integral: instead of importance-sampling
stored PE samples, it draws source-frame points from the *current* population
target at every likelihood call and scores them with each event's flow.  The
draws are deterministic functions of a fixed base-uniform array (common
random numbers), so the likelihood stays a deterministic, continuous function
of the hyperparameters — safe for nested samplers.

All samplers here return the EXACT log density ``log_s`` of their own
proposal (a normalised piecewise-constant histogram on the grid, or the
analytic truncated normal).  Downstream weights use ``log t(θ) − log s(θ)``,
which makes the estimator unbiased at any finite grid resolution: the grid
only controls proposal quality, never correctness.  A relative density floor
(``FLOOR_REL``) keeps every grid cell sampleable so support mismatches show
up as small weights rather than silently missing probability mass.

Everything is pure ``jax.numpy`` with static shapes and runs inside the
jitted likelihood body.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfinv, logsumexp


# Relative floor applied to histogram cell densities: guarantees full grid
# support for the proposal. Floored cells carry ~1e-12 of the proposal mass,
# and the exact-reweight weights suppress them wherever the target is zero.
FLOOR_REL = 1e-12


class HistogramSample(NamedTuple):
    x: jnp.ndarray          # sampled coordinate
    cell: jnp.ndarray       # cell index (int)
    log_s: jnp.ndarray      # exact log proposal density at x
    log_norm: jnp.ndarray   # log Σ cell masses before normalisation


def _floored(log_cells: jnp.ndarray) -> jnp.ndarray:
    """Apply the relative density floor in log space."""
    log_floor = jnp.max(log_cells) + jnp.log(FLOOR_REL)
    return jnp.logaddexp(log_cells, log_floor)


def sample_histogram_1d(
    u: jnp.ndarray,
    edges: jnp.ndarray,
    log_cell_dens: jnp.ndarray,
) -> HistogramSample:
    """Draw from a piecewise-constant density on ``edges`` via inverse CDF.

    Parameters
    ----------
    u : (J,) uniforms in [0, 1)
    edges : (K+1,) monotonically increasing cell edges
    log_cell_dens : (K,) log of the (unnormalised, floored by the caller or
        not) cell densities per unit x

    Returns
    -------
    HistogramSample with the exact normalised log density at each draw.
    """
    widths = jnp.diff(edges)
    log_mass = log_cell_dens + jnp.log(widths)
    log_norm = logsumexp(log_mass)
    cdf = jnp.cumsum(jnp.exp(log_mass - log_norm))

    k = jnp.clip(jnp.searchsorted(cdf, u, side="right"), 0, widths.shape[0] - 1)
    c_lo = jnp.where(k > 0, cdf[jnp.maximum(k - 1, 0)], 0.0)
    denom = jnp.maximum(cdf[k] - c_lo, jnp.finfo(cdf.dtype).tiny)
    frac = jnp.clip((u - c_lo) / denom, 0.0, 1.0)
    x = edges[k] + frac * widths[k]

    log_s = log_cell_dens[k] - log_norm
    return HistogramSample(x=x, cell=k, log_s=log_s, log_norm=log_norm)


class MassQSample(NamedTuple):
    m1: jnp.ndarray
    q: jnp.ndarray
    log_s: jnp.ndarray      # exact joint log proposal density per unit (m1, q)
    log_norm: jnp.ndarray   # log ∬ floored target over the grid box


def sample_mass_q(
    u1: jnp.ndarray,
    u2: jnp.ndarray,
    m1_edges: jnp.ndarray,
    q_edges: jnp.ndarray,
    log_t_cells: jnp.ndarray,
) -> MassQSample:
    """Draw (m1, q) from a 2-D piecewise-constant proposal.

    ``log_t_cells`` is the target log density evaluated at the (R, C) cell
    centers (see :func:`cell_centers`).  Sampling is m1-row marginal via
    ``u1`` then q-column conditional via ``u2``, uniform within the selected
    cell, so the exact proposal density at the draw is
    ``t_cells[r, c] / norm``.
    """
    log_t_cells = _floored(log_t_cells)
    dm = jnp.diff(m1_edges)
    dq = jnp.diff(q_edges)

    log_mass = log_t_cells + jnp.log(dm)[:, None] + jnp.log(dq)[None, :]
    log_norm = logsumexp(log_mass)

    # Row (m1) marginal.
    log_row_mass = logsumexp(log_mass, axis=1)
    row = sample_histogram_1d(u1, m1_edges, log_row_mass - jnp.log(dm))
    r = row.cell
    m1 = row.x

    # Column (q) conditional within the selected row.
    cond_cdf = jnp.cumsum(
        jnp.exp(log_mass - logsumexp(log_mass, axis=1, keepdims=True)), axis=1
    )
    cdf_r = cond_cdf[r]  # (J, C)
    c = jnp.clip(
        jax.vmap(lambda cdf, ui: jnp.searchsorted(cdf, ui, side="right"))(cdf_r, u2),
        0,
        dq.shape[0] - 1,
    )
    take = jax.vmap(lambda cdf, ci: cdf[ci])
    c_hi = take(cdf_r, c)
    c_lo = jnp.where(c > 0, take(cdf_r, jnp.maximum(c - 1, 0)), 0.0)
    denom = jnp.maximum(c_hi - c_lo, jnp.finfo(c_hi.dtype).tiny)
    frac = jnp.clip((u2 - c_lo) / denom, 0.0, 1.0)
    q = q_edges[c] + frac * dq[c]

    log_s = log_t_cells[r, c] - log_norm
    return MassQSample(m1=m1, q=q, log_s=log_s, log_norm=log_norm)


class TruncNormSample(NamedTuple):
    x: jnp.ndarray
    log_s: jnp.ndarray      # exact normalised log density at x


def truncnorm_sample(
    u: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    lo: float = -1.0,
    hi: float = 1.0,
) -> TruncNormSample:
    """Draw from a truncated Normal on [lo, hi] via the analytic inverse CDF."""
    sigma = jnp.maximum(sigma, 1e-6)
    sqrt2 = jnp.sqrt(2.0)

    def _Phi(x):
        return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt2)))

    Phi_lo, Phi_hi = _Phi(lo), _Phi(hi)
    span = jnp.maximum(Phi_hi - Phi_lo, jnp.finfo(u.dtype).tiny)
    up = Phi_lo + u * span
    # Keep erfinv strictly inside (-1, 1).
    arg = jnp.clip(2.0 * up - 1.0, -1.0 + 1e-15, 1.0 - 1e-15)
    x = jnp.clip(mu + sigma * sqrt2 * erfinv(arg), lo, hi)

    log_s = (
        -0.5 * ((x - mu) / sigma) ** 2
        - jnp.log(sigma)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - jnp.log(span)
    )
    return TruncNormSample(x=x, log_s=log_s)


def sample_z_from_grid(
    u: jnp.ndarray,
    zgrid: jnp.ndarray,
    log_t_nodes: jnp.ndarray,
) -> HistogramSample:
    """Draw z from node values on ``zgrid`` (piecewise-constant proposal).

    Cells span consecutive grid nodes; each cell's density is the
    (linear-space) mean of its two node values, floored for full support.
    """
    log_cell_dens = _floored(
        jnp.logaddexp(log_t_nodes[:-1], log_t_nodes[1:]) - jnp.log(2.0)
    )
    return sample_histogram_1d(u, zgrid, log_cell_dens)


# ── grid construction helpers (host- or trace-side) ─────────────────────────


def cell_centers(edges: jnp.ndarray) -> jnp.ndarray:
    """Midpoints of consecutive edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def make_mass_q_edges(
    m1_lo: float, m1_hi: float, n_m1: int = 512, n_q: int = 256, q_lo: float = 0.01
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Log-spaced m1 edges and linear q edges for the (m1, q) proposal grid."""
    m1_edges = jnp.geomspace(m1_lo, m1_hi, n_m1 + 1)
    q_edges = jnp.linspace(q_lo, 1.0, n_q + 1)
    return m1_edges, q_edges


def resolve_mass_grid_bounds(model) -> tuple[float, float]:
    """Grid box for m1: wide enough for any prior draw of the mass model.

    Uses the prior upper bounds of tagged parameters: power-law support can
    extend to ``m_max + dm_max`` (high-mass smoothing), Gaussian peaks to
    ``mu + 5 sigma``.  Overshooting is harmless (floored log-spaced cells in
    zero-density regions carry negligible proposal mass), so the box takes
    the max of those and a 200 M_sun default.  Lower edge: min(2, m_min).
    """
    by_name: dict[str, list[tuple[float, float]]] = {}
    lows, highs, _ = model.prior_bounds()
    for spec, lo, hi in zip(model.param_specs, lows, highs):
        name = getattr(spec, "name", "") or ""
        short = name.split(".", 1)[-1]
        by_name.setdefault(short, []).append((float(lo), float(hi)))

    def _max_hi(short, default=0.0):
        return max((hi for _, hi in by_name.get(short, [])), default=default)

    def _min_lo(short, default=np.inf):
        return min((lo for lo, _ in by_name.get(short, [])), default=default)

    m1_lo = min(2.0, _min_lo("m_min", default=2.0))
    m1_hi = max(
        200.0,
        _max_hi("m_max") + _max_hi("dm_max"),
        _max_hi("mu") + 5.0 * _max_hi("sigma"),
    )
    return float(m1_lo), float(m1_hi)
