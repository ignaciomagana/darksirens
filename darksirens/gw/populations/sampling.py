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

Every truncated sampler is paired with a ``*_logpdf`` evaluator that returns
the same density at an ARBITRARY point, not just at points the sampler
produced.  Mixture proposals need exactly that: a draw from component B must
be scored under component A's density (see
``darksirens/likelihood/flow_events.py``), and scoring it with the sampler's
own returned ``log_s`` — the mistake that silently biases naive multiple-
importance-sampling code — is impossible when the density is a standalone
function of the point.

Everything is pure ``jax.numpy`` with static shapes and runs inside the
jitted likelihood body.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfinv, log_ndtr, logsumexp


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


def truncnorm_log_span(mu, sigma, lo, hi):
    """``log[Phi((hi-mu)/sigma) - Phi((lo-mu)/sigma)]``, stable in both tails.

    Shared by :func:`truncnorm_sample` and :func:`truncnorm_logpdf` so the
    sampler and its density can never drift apart.  See the sampler for why
    the naive difference of CDFs is unusable here.
    """
    sigma = jnp.maximum(sigma, 1e-6)
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    # Work in whichever tail keeps both endpoints on the same (lower) side, so
    # the subtraction below is always between two log-CDFs of the same sign.
    flip = a > 0.0
    aa = jnp.where(flip, -b, a)
    bb = jnp.where(flip, -a, b)
    log_lo, log_hi = log_ndtr(aa), log_ndtr(bb)      # log_lo <= log_hi
    # log(e^log_hi - e^log_lo) = log_hi + log(-expm1(log_lo - log_hi))
    tiny = jnp.finfo(jnp.result_type(float(0.0), log_hi)).tiny
    delta = jnp.minimum(log_lo - log_hi, -tiny)
    return log_hi + jnp.log(-jnp.expm1(delta))


def truncnorm_logpdf(
    x: jnp.ndarray,
    mu: jnp.ndarray,
    sigma: jnp.ndarray,
    lo=-1.0,
    hi=1.0,
) -> jnp.ndarray:
    """Exact log density of :func:`truncnorm_sample` at an arbitrary ``x``.

    ``-inf`` outside ``[lo, hi]`` — the sampler cannot produce those points,
    so a mixture partner scoring them under this component must see zero.
    """
    sigma = jnp.maximum(sigma, 1e-6)
    log_span = truncnorm_log_span(mu, sigma, lo, hi)
    logp = (
        -0.5 * ((x - mu) / sigma) ** 2
        - jnp.log(sigma)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_span
    )
    return jnp.where((x >= lo) & (x <= hi), logp, -jnp.inf)


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

    # log(span) computed IN LOG SPACE, not as log of the difference above.
    # Once the window sits more than ~8.3 sigma from mu, Phi_lo and Phi_hi round
    # to the same float and their difference underflows to exactly 0; the tiny
    # floor then replaces the true span (e.g. 7.62e-24 at 10 sigma) with
    # 2.2e-308, making -log(span) 708.4 instead of 53.2 -- log_s too LARGE by
    # ~655 nats, which kills every draw for that event and punches a spurious
    # hole in the likelihood surface.  Reachable because the truncation bounds
    # here are the per-event support box, not a fixed [-1, 1]: mu_chi at a prior
    # edge with sigma_chi at its floor puts mu tens of sigma outside a narrow box.
    log_span = truncnorm_log_span(mu, sigma, lo, hi)
    # NO floor here.  log_ndtr is finite for any finite argument and delta <= 0
    # keeps -expm1(delta) in (0, 1], so log_span is always finite -- including
    # the degenerate lo == hi case, where it collapses to log_hi + log(tiny).
    # Flooring at log(tiny) would clamp legitimately small spans: a window 70
    # sigma out has a true log_span of -2456, and the floor would report -708.

    log_s = (
        -0.5 * ((x - mu) / sigma) ** 2
        - jnp.log(sigma)
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_span
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


# ── window-truncated samplers (per-event support boxes) ─────────────────────
#
# A single population-wide proposal starves narrow event posteriors (a
# low-mass, low-z event may receive ~none of J draws).  These variants draw
# from the SAME target restricted to a window [lo, hi], with the truncation
# normaliser folded into the exact proposal density — plain importance
# sampling with a better proposal, unbiased for any window that covers the
# integrand's support.  Windows may be traced scalars (hyperparameter-
# dependent), and the samplers are designed to be vmapped over events.


def _hist_cdf_nodes(edges: jnp.ndarray, log_cell_dens: jnp.ndarray):
    """Piecewise-linear CDF of a histogram density at the cell edges."""
    log_mass = log_cell_dens + jnp.log(jnp.diff(edges))
    log_norm = logsumexp(log_mass)
    cdf = jnp.concatenate(
        [jnp.zeros(1, dtype=log_mass.dtype), jnp.cumsum(jnp.exp(log_mass - log_norm))]
    )
    return cdf, log_norm


def _hist_cdf_at(edges, cdf_nodes, log_cell_dens, log_norm, x):
    """Exact histogram CDF evaluated at arbitrary x (linear within cells)."""
    k = jnp.clip(jnp.searchsorted(edges, x, side="right") - 1, 0, edges.shape[0] - 2)
    return jnp.clip(
        cdf_nodes[k] + jnp.exp(log_cell_dens[k] - log_norm) * (x - edges[k]), 0.0, 1.0
    )


def _hist_trunc_span(edges, cdf_nodes, log_cell_dens, log_norm, lo, hi):
    """Clipped window and its CDF mass, shared by the sampler and the pdf."""
    lo = jnp.clip(lo, edges[0], edges[-1])
    hi = jnp.clip(hi, edges[0], edges[-1])
    hi = jnp.maximum(hi, lo)
    F_lo = _hist_cdf_at(edges, cdf_nodes, log_cell_dens, log_norm, lo)
    F_hi = _hist_cdf_at(edges, cdf_nodes, log_cell_dens, log_norm, hi)
    span = jnp.maximum(F_hi - F_lo, jnp.finfo(cdf_nodes.dtype).tiny)
    return lo, hi, F_lo, span


def _sample_histogram_trunc_tab(u, edges, log_cell_dens, cdf_nodes, log_norm, lo, hi):
    """:func:`sample_histogram_trunc` with the CDF nodes supplied by the caller."""
    lo, hi, F_lo, span = _hist_trunc_span(
        edges, cdf_nodes, log_cell_dens, log_norm, lo, hi
    )
    up = F_lo + u * span
    K = edges.shape[0] - 1
    k = jnp.clip(jnp.searchsorted(cdf_nodes, up, side="right") - 1, 0, K - 1)
    denom = jnp.maximum(cdf_nodes[k + 1] - cdf_nodes[k], jnp.finfo(up.dtype).tiny)
    frac = jnp.clip((up - cdf_nodes[k]) / denom, 0.0, 1.0)
    x = jnp.clip(edges[k] + frac * (edges[k + 1] - edges[k]), lo, hi)

    log_s = log_cell_dens[k] - log_norm - jnp.log(span)
    return HistogramSample(x=x, cell=k, log_s=log_s, log_norm=log_norm)


def _histogram_trunc_logpdf_tab(x, edges, log_cell_dens, cdf_nodes, log_norm, lo, hi):
    """:func:`histogram_trunc_logpdf` with the CDF nodes supplied by the caller."""
    lo, hi, _F_lo, span = _hist_trunc_span(
        edges, cdf_nodes, log_cell_dens, log_norm, lo, hi
    )
    K = edges.shape[0] - 1
    k = jnp.clip(jnp.searchsorted(edges, x, side="right") - 1, 0, K - 1)
    logp = log_cell_dens[k] - log_norm - jnp.log(span)
    return jnp.where((x >= lo) & (x <= hi), logp, -jnp.inf)


def sample_histogram_trunc(
    u: jnp.ndarray,
    edges: jnp.ndarray,
    log_cell_dens: jnp.ndarray,
    lo,
    hi,
) -> HistogramSample:
    """Inverse-CDF draw from a histogram density truncated to [lo, hi].

    ``log_s`` is the exact density of the truncated, renormalised proposal.
    A window carrying (numerically) no target mass yields a huge ``log_s``
    (tiny normaliser), so downstream ``log t - log s`` weights collapse to
    -inf — the correct "population excludes this event" behaviour.
    """
    cdf_nodes, log_norm = _hist_cdf_nodes(edges, log_cell_dens)
    return _sample_histogram_trunc_tab(
        u, edges, log_cell_dens, cdf_nodes, log_norm, lo, hi
    )


def histogram_trunc_logpdf(
    x: jnp.ndarray,
    edges: jnp.ndarray,
    log_cell_dens: jnp.ndarray,
    lo,
    hi,
) -> jnp.ndarray:
    """Exact log density of :func:`sample_histogram_trunc` at arbitrary ``x``.

    Piecewise constant on the grid cells, renormalised by the window's CDF
    mass, and ``-inf`` strictly outside the (clipped) window.  Evaluating the
    density here rather than reusing the sampler's returned ``log_s`` is what
    lets a mixture proposal score EVERY draw under EVERY component.
    """
    cdf_nodes, log_norm = _hist_cdf_nodes(edges, log_cell_dens)
    return _histogram_trunc_logpdf_tab(
        x, edges, log_cell_dens, cdf_nodes, log_norm, lo, hi
    )


def q_marginal_log_dens(
    m1_edges: jnp.ndarray, q_edges: jnp.ndarray, log_t_cells: jnp.ndarray
) -> jnp.ndarray:
    """(C,) log density per unit q of the m1-marginalised target."""
    dm = jnp.diff(m1_edges)
    dq = jnp.diff(q_edges)
    log_mass = log_t_cells + jnp.log(dm)[:, None] + jnp.log(dq)[None, :]
    return logsumexp(log_mass, axis=0) - jnp.log(dq)


def sample_q_marginal_trunc(
    u: jnp.ndarray,
    m1_edges: jnp.ndarray,
    q_edges: jnp.ndarray,
    log_t_cells: jnp.ndarray,
    q_win,
) -> HistogramSample:
    """q draw from the m1-marginalised target truncated to ``q_win``.

    First stage of the chirp-band mass sampler: q comes from the full-m1
    marginal (a proposal choice, corrected exactly downstream); the m1 stage
    (:func:`sample_m1_given_q_trunc`) then applies per-draw m1 windows that
    may depend on the drawn q (e.g. a chirp-mass band).  The caller owns any
    density floor on ``log_t_cells``.
    """
    log_col_dens = q_marginal_log_dens(m1_edges, q_edges, log_t_cells)
    return sample_histogram_trunc(u, q_edges, log_col_dens, q_win[0], q_win[1])


class ColumnCdfTables(NamedTuple):
    """Per-q-column CDF nodes and normalisers of the m1 conditional.

    ``cdf_nodes`` is (C, K+1) and ``log_norm`` is (C,).  Built ONCE per
    likelihood call and reused by every draw of every event: the previous
    code rebuilt a column's CDF inside a per-draw ``vmap`` (O(J*K) work per
    event), which the mixture proposal would have paid three times over.
    """

    cdf_nodes: jnp.ndarray
    log_norm: jnp.ndarray


def build_column_cdf_tables(
    m1_edges: jnp.ndarray, log_t_cells: jnp.ndarray
) -> ColumnCdfTables:
    """Tabulate the m1-conditional CDF of every q column of ``log_t_cells``."""
    log_mass = log_t_cells + jnp.log(jnp.diff(m1_edges))[:, None]   # (K, C)
    log_norm = logsumexp(log_mass, axis=0)                          # (C,)
    cdf = jnp.cumsum(jnp.exp(log_mass - log_norm), axis=0)          # (K, C)
    nodes = jnp.concatenate(
        [jnp.zeros((1, cdf.shape[1]), dtype=cdf.dtype), cdf], axis=0
    )                                                               # (K+1, C)
    return ColumnCdfTables(cdf_nodes=nodes.T, log_norm=log_norm)


def sample_m1_given_q_trunc(
    u: jnp.ndarray,
    m1_edges: jnp.ndarray,
    log_t_cells: jnp.ndarray,
    q_cells: jnp.ndarray,
    m1_lo: jnp.ndarray,
    m1_hi: jnp.ndarray,
    tables: ColumnCdfTables | None = None,
):
    """m1 draw from the conditional t(m1 | q-cell), per-draw window [lo, hi].

    Second stage of the chirp-band sampler: each draw j inverts the CDF of
    column ``q_cells[j]`` of the target grid restricted to its own window
    (e.g. the event's chirp-mass band at the drawn q and z).  Returns
    ``(m1, log_s)`` with the exact truncated conditional density.  Pass
    ``tables`` from :func:`build_column_cdf_tables` to reuse a precomputed
    per-column CDF instead of rebuilding it per draw.
    """
    tab = build_column_cdf_tables(m1_edges, log_t_cells) if tables is None else tables

    def _one(u_j, c_j, lo_j, hi_j):
        out = _sample_histogram_trunc_tab(
            u_j[None], m1_edges, log_t_cells[:, c_j],
            tab.cdf_nodes[c_j], tab.log_norm[c_j], lo_j, hi_j,
        )
        return out.x[0], out.log_s[0]

    return jax.vmap(_one)(u, q_cells, m1_lo, m1_hi)


def m1_given_q_trunc_logpdf(
    m1: jnp.ndarray,
    m1_edges: jnp.ndarray,
    log_t_cells: jnp.ndarray,
    q_cells: jnp.ndarray,
    m1_lo: jnp.ndarray,
    m1_hi: jnp.ndarray,
    tables: ColumnCdfTables | None = None,
) -> jnp.ndarray:
    """Exact log density of :func:`sample_m1_given_q_trunc` at arbitrary m1."""
    tab = build_column_cdf_tables(m1_edges, log_t_cells) if tables is None else tables

    def _one(x_j, c_j, lo_j, hi_j):
        return _histogram_trunc_logpdf_tab(
            x_j[None], m1_edges, log_t_cells[:, c_j],
            tab.cdf_nodes[c_j], tab.log_norm[c_j], lo_j, hi_j,
        )[0]

    return jax.vmap(_one)(m1, q_cells, m1_lo, m1_hi)


def sample_mass_q_trunc(
    u1: jnp.ndarray,
    u2: jnp.ndarray,
    m1_edges: jnp.ndarray,
    q_edges: jnp.ndarray,
    log_t_cells: jnp.ndarray,
    m1_win,
    q_win,
) -> MassQSample:
    """(m1, q) draw from the 2-D target truncated to a per-event box.

    m1 comes from the full-q marginal truncated to ``m1_win``; q from the
    selected row's conditional truncated to ``q_win``.  Both truncation
    normalisers enter ``log_s`` exactly, so the estimator stays unbiased for
    any box.
    """
    log_t_cells = _floored(log_t_cells)
    dm = jnp.diff(m1_edges)
    dq = jnp.diff(q_edges)
    log_mass = log_t_cells + jnp.log(dm)[:, None] + jnp.log(dq)[None, :]
    log_norm = logsumexp(log_mass)

    # m1 marginal density per unit m1 (integrated over the FULL q range —
    # a proposal choice; the q window enters through the conditional below).
    log_row_dens = logsumexp(log_mass, axis=1) - jnp.log(dm)
    row = sample_histogram_trunc(u1, m1_edges, log_row_dens, m1_win[0], m1_win[1])
    r, m1 = row.cell, row.x

    # q | row, truncated to q_win: vmap the 1-D truncated sampler over draws.
    def _q_draw(u_j, dens_row):
        out = sample_histogram_trunc(u_j[None], q_edges, dens_row, q_win[0], q_win[1])
        return out.x[0], out.log_s[0]

    q, log_s_q = jax.vmap(_q_draw)(u2, log_t_cells[r])

    log_s = row.log_s + log_s_q
    return MassQSample(m1=m1, q=q, log_s=log_s, log_norm=log_norm)


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
