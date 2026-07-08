"""
catalog.py
----------
EM catalog redshift prior: p_cat(z | pix), the *shape* of the host
probability over the observed galaxies in a pixel.

Each real galaxy i contributes a kernel that is the normalised
volumetric photo-z posterior

    p(z | gal i) = N(z; z_i, sigma_eff,i) * g(z) / Z_i,
    Z_i          = ∫_0^zmax N(z'; z_i, sigma_eff,i) g(z') dz',

with the galaxy measure g(z) = dV_c/dz * (1+z)^delta and

    sigma_eff,i = max( sqrt(sigma_cat,i^2 + sigma_kde^2), 1e-4 ).

The Gaussian is the photo-z/instrumental likelihood broadened by the
LSS kernel sigma_kde (variances add for Gaussian overlap); g(z) is the
volumetric interim prior on the galaxy's true redshift (Gray et al.
2020, arXiv:1908.06050).  Because Z_i normalises each kernel to unit
mass on [0, zmax], every galaxy carries exactly its base weight:

    p_cat(z | pix) = Σ_i  w~_i * p(z | gal i),     w~_i = w_i / Σ_j w_j,

which integrates to 1 per pixel.  Crucially the volumetric prior tilts
each kernel but does NOT rescale a galaxy's total host probability —
multiplying the mixture weights by dV(z_i) (a previous implementation)
makes a galaxy more probable as a host merely for being far away.

Z_i is evaluated by Gauss–Legendre quadrature in the Gaussian quantile
variable (exact treatment of the [0, zmax] truncation; robust for any
sigma_eff from the 1e-4 floor up to broad photo-z), with g interpolated
from a per-proposal grid.  The sigma_eff floor (~30 km/s) only protects
numerics when spectroscopic errors underflow; it is far below any
physical redshift uncertainty.

Hot paths precompute the per-galaxy quantities once per parameter
proposal via ``catalog_kernel_state`` and evaluate per sample with
``eval_log_catalog_prior_state``; the scalar ``log_catalog_prior`` keeps
the historical signature for the complete-catalog and bright-siren
models and for tests.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import logsumexp, ndtr, ndtri
from jax.scipy.stats import norm
from typing import NamedTuple

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog

from darksirens.redshift.grid import zgrid
from .completion import log_galaxy_measure_grid

_ZMAX: float = float(np.asarray(zgrid)[-1])

#: Numerical floor on sigma_eff [redshift units]; ~30 km/s, well below any
#: physical photo-z or peculiar-velocity scale.  Protects against
#: spectroscopic dzgals -> 0 with sigma_kde = 0.
SIGMA_EFF_FLOOR: float = 1e-4

# Gauss–Legendre nodes/weights on [0, 1] for the kernel normalisation.
# The default 24 nodes are conservative for broad photo-z kernels; for
# spectroscopic catalogs (sigma_eff ~ 1e-3, g(z) locally smooth) far fewer
# nodes are exact to likelihood precision and the quadrature dominates the
# per-proposal cost of wide-sky runs, so the count is configurable.
_GL_NODES = 24
_GL_X = None
_GL_W = None


def configure_kernel_quadrature(n_nodes: int = 24):
    """Set the Gauss–Legendre node count for the kernel normalisation Z_i.

    Trace-time configuration: call BEFORE the first likelihood evaluation
    (reconfiguring after a jit trace does not affect the compiled graph).
    Node-count guidance: 24 (default) is safe for any kernel width; 8 is
    validated for spectroscopic catalogs across the sampled sigma_kde prior
    (see tests/test_catalog_row_chunk.py); 4 is only safe when sigma_kde is
    FIXED near zero — its error grows ~20x by sigma_kde = 0.05.
    """
    global _GL_NODES, _GL_X, _GL_W
    n_nodes = int(n_nodes)
    if n_nodes < 2:
        raise ValueError(f"kernel quadrature needs >= 2 nodes, got {n_nodes}")
    _GL_NODES = n_nodes
    _glx, _glw = np.polynomial.legendre.leggauss(_GL_NODES)
    _GL_X = jnp.asarray(0.5 * (_glx + 1.0))
    _GL_W = jnp.asarray(0.5 * _glw)


configure_kernel_quadrature(_GL_NODES)


def _row_real_mask(zs, ws, ngal):
    """Real-galaxy mask for one padded row."""
    if ngal is not None:
        return jnp.arange(zs.shape[0]) < ngal
    return ws > 0


# Row-chunked mapping over catalog rows.  The per-row kernel-norm quadrature
# materialises (N_max_gals, K) intermediates; a plain vmap over all rows holds
# (N_rows, N_max_gals, K) at once, which OOMs for wide-sky runs (e.g. a
# 49k-row x 2113-galaxy DESI view requests ~80 GB).  Above the element
# threshold the vmap is wrapped in ``lax.map`` over fixed-size row chunks:
# identical per-row arithmetic (no cross-row ops), bounded peak memory.
_ROW_CHUNK_AUTO_THRESHOLD: int = 2**25   # N_rows * N_max_gals elements
_ROW_CHUNK_SIZE: int = 512
_ROW_CHUNK_MODE = "auto"                 # "auto" | None | int (for tests)


def configure_catalog_row_chunk(mode="auto"):
    """Set row-chunking for kernel-state builds: "auto", None (off), or an int.

    Trace-time configuration: call BEFORE the first likelihood evaluation.
    Reconfiguring after a function has been jit-traced has no effect on the
    already-compiled graph (the mode is read when the trace is built).
    """
    global _ROW_CHUNK_MODE
    if mode is not None and mode != "auto":
        mode = int(mode)
        if mode < 1:
            raise ValueError(f"row chunk must be >= 1, got {mode}")
    _ROW_CHUNK_MODE = mode


def _resolve_row_chunk(n_rows: int, n_max: int) -> int | None:
    if _ROW_CHUNK_MODE is None:
        return None
    if _ROW_CHUNK_MODE != "auto":
        return int(_ROW_CHUNK_MODE)
    if n_rows * n_max > _ROW_CHUNK_AUTO_THRESHOLD:
        return _ROW_CHUNK_SIZE
    return None


def _map_rows(row_fn, args: tuple):
    """vmap ``row_fn`` over the leading (row) axis of every array in ``args``,
    chunking with ``lax.map`` when the catalog view is large (see above).

    Zero-padded rows evaluate through the same masked per-row path (empty
    ``real`` mask) and are sliced off the result, so outputs are identical to
    the unchunked vmap row-for-row.
    """
    n_rows = args[0].shape[0]
    n_max = args[0].shape[1] if args[0].ndim > 1 else 1
    chunk = _resolve_row_chunk(n_rows, n_max)
    if chunk is None or chunk >= n_rows:
        return vmap(row_fn)(*args)

    n_pad = (-n_rows) % chunk
    def _prep(a):
        if n_pad:
            pad = jnp.zeros((n_pad,) + a.shape[1:], dtype=a.dtype)
            a = jnp.concatenate([a, pad], axis=0)
        return a.reshape((n_rows + n_pad) // chunk, chunk, *a.shape[1:])
    chunked = tuple(_prep(a) for a in args)
    out = lax.map(lambda ch: vmap(row_fn)(*ch), chunked)
    def _post(o):
        return o.reshape(-1, *o.shape[2:])[:n_rows]
    if isinstance(out, tuple):
        return tuple(_post(o) for o in out)
    return _post(out)


def _row_log_kernel_norms(zs, sig_eff, real, log_g_grid):
    """
    log Z_i for one row: Z_i = ∫_0^zmax N(z; z_i, sig_i) g(z) dz.

    Substituting u = Phi((z - z_i)/sig_i) maps the truncated integral to
    ∫_a^b g(z_i + sig_i * Phi^{-1}(u)) du with a = Phi(-z_i/sig),
    b = Phi((zmax - z_i)/sig): truncation handled exactly, integrand
    smooth, Gauss–Legendre converges fast for any sigma.
    """
    a = ndtr(-zs / sig_eff)
    b = ndtr((_ZMAX - zs) / sig_eff)
    span = b - a                                            # (N_max,)
    u = a[..., None] + span[..., None] * _GL_X              # (N_max, K)
    u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
    z_node = jnp.clip(zs[..., None] + sig_eff[..., None] * ndtri(u), 0.0, _ZMAX)
    g = jnp.exp(
        jnp.interp(z_node.reshape(-1), zgrid, log_g_grid)
    ).reshape(z_node.shape)
    Z = span * (g * _GL_W).sum(axis=-1)                     # (N_max,)
    return jnp.where(real & (Z > 0.0), jnp.log(jnp.maximum(Z, 1e-300)), 0.0)


def _row_kernel_state(zs, dzs, ws, ngal, sigma_kde, log_g_grid, volume_weighted=False):
    """
    Per-galaxy kernel quantities for one row, under one of two host-weight
    conventions for the galaxy measure g(z) = dV_c/dz * (1+z)^delta:

    - ``volume_weighted=False`` (incomplete-catalog default): each galaxy's
      total host probability is its base weight w~_i; g only tilts the kernel
      shape, divided back out by Z_i = ∫ N(z;z_i,sig) g(z) dz so each kernel has
      unit mass.  The g(z) front factor is reapplied per sample in the evaluator.

    - ``volume_weighted=True`` (complete-catalog): each galaxy's host
      probability scales with the comoving volume at its redshift, weight
      w_i * g(z_i), with a plain N(z;z_i,sig) kernel (no Z_i, no front g(z)).
      The catalog is the full universe, so the host rate must track the number
      of candidate hosts per redshift shell.

    Returns ``(log_kw, sig_eff)``; the evaluator's g(z) handling is selected by
    the same ``volume_weighted`` flag carried on :class:`CatalogKernelState`.
    """
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR)
    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1e-300)), -jnp.inf)

    if volume_weighted:
        log_w = log_w + jnp.where(real, jnp.interp(zs, zgrid, log_g_grid), 0.0)

    lse = logsumexp(log_w)
    has_galaxies = jnp.isfinite(lse)
    log_w_norm = jnp.where(real, log_w - jnp.where(has_galaxies, lse, 0.0), -jnp.inf)

    if volume_weighted:
        log_kw = log_w_norm
    else:
        log_Z = _row_log_kernel_norms(zs, sig_eff, real, log_g_grid)
        log_kw = jnp.where(real, log_w_norm - log_Z, -jnp.inf)
    return log_kw, sig_eff


class CatalogKernelState(NamedTuple):
    """Per-galaxy kernel quantities for all catalog rows (one proposal)."""
    log_g_grid: jnp.ndarray  # (N_grid,)
    log_kw: jnp.ndarray      # (N_rows, N_max)
    sig_eff: jnp.ndarray     # (N_rows, N_max)
    volume_weighted: bool = False


def catalog_kernel_state(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    log_g_grid: jnp.ndarray | None = None,
    volume_weighted: bool = False,
) -> CatalogKernelState:
    """Precompute per-galaxy kernel quantities once per parameter proposal."""
    if log_g_grid is None:
        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals
    ngals = em_catalog.ngals

    if ngals is not None:
        log_kw, sig_eff = _map_rows(
            lambda zs, dzs, ws, ng: _row_kernel_state(
                zs, dzs, ws, ng, survey.sigma_kde, log_g_grid, volume_weighted
            ),
            (zgals, dzgals, wgals, ngals),
        )
    else:
        log_kw, sig_eff = _map_rows(
            lambda zs, dzs, ws: _row_kernel_state(
                zs, dzs, ws, None, survey.sigma_kde, log_g_grid, volume_weighted
            ),
            (zgals, dzgals, wgals),
        )

    return CatalogKernelState(
        log_g_grid=log_g_grid, log_kw=log_kw, sig_eff=sig_eff,
        volume_weighted=volume_weighted,
    )


# ------------------------------------------------------------
# Marked-host kernel state (galaxy marks -> BBH-host efficiency)
# ------------------------------------------------------------

def _row_marked_kernel_state(zs, dzs, ws, log_h_row, ngal, sigma_kde, log_g_grid):
    """Per-galaxy kernel quantities for one row using host-efficiency weights.

    Identical to :func:`_row_kernel_state` but the per-pixel-normalised weight is
    ``w_i·h_i`` (host efficiency ``h_i = exp(log_h_row_i)``) instead of ``w_i``,
    and the row's marked total ``log_N_host = log Σ_i w_i h_i`` is returned (it
    replaces the integer count in the assembled prior).  With ``log_h_row ≡ 0``
    and unit weights this reduces to :func:`_row_kernel_state`.
    """
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR)

    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1e-300)), -jnp.inf)
    log_wh = jnp.where(real, log_w + log_h_row, -jnp.inf)   # log(w_i h_i)
    lse = logsumexp(log_wh)                                 # log Σ_i w_i h_i
    has_galaxies = jnp.isfinite(lse)
    log_wh_norm = jnp.where(real, log_wh - jnp.where(has_galaxies, lse, 0.0), -jnp.inf)

    log_Z = _row_log_kernel_norms(zs, sig_eff, real, log_g_grid)
    log_kw = jnp.where(real, log_wh_norm - log_Z, -jnp.inf)
    log_N_host = jnp.where(has_galaxies, lse, -jnp.inf)
    return log_kw, sig_eff, log_N_host


def marked_catalog_kernel_state(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    log_h: jnp.ndarray,
    log_g_grid: jnp.ndarray | None = None,
):
    """Marked per-galaxy kernel state + per-row marked total ``log_N_host``.

    ``log_h`` is ``(N_rows, N_max_gals)`` per-galaxy log host efficiency (from
    :mod:`darksirens.marks`).  Returns ``(CatalogKernelState, log_N_host)`` where
    the state's ``log_kw`` carries the marked (per-pixel-normalised) weights, so
    the existing per-sample evaluator is reused unchanged.
    """
    if log_g_grid is None:
        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals
    ngals = em_catalog.ngals

    if ngals is not None:
        log_kw, sig_eff, log_N_host = _map_rows(
            lambda zs, dzs, ws, lh, ng: _row_marked_kernel_state(
                zs, dzs, ws, lh, ng, survey.sigma_kde, log_g_grid
            ),
            (zgals, dzgals, wgals, log_h, ngals),
        )
    else:
        log_kw, sig_eff, log_N_host = _map_rows(
            lambda zs, dzs, ws, lh: _row_marked_kernel_state(
                zs, dzs, ws, lh, None, survey.sigma_kde, log_g_grid
            ),
            (zgals, dzgals, wgals, log_h),
        )

    return CatalogKernelState(log_g_grid=log_g_grid, log_kw=log_kw, sig_eff=sig_eff), log_N_host


def _logsumexp_neginf_safe(terms):
    """logsumexp returning exactly -inf for an all--inf row WITHOUT the NaN
    backward pass of the plain reduction: softmax of all--inf is 0/0 = NaN,
    and that NaN survives multiplication by a ZERO upstream cotangent (mul's
    VJP scales by the stored NaN), poisoning every parameter's gradient —
    this is what broke NumPyro NUTS for dark sirens (empty catalog pixels).
    Rows with any finite entry are bit-identical to plain logsumexp: the
    sanitized padding's weight exp(-1e30 - max) underflows to exactly zero.
    """
    finite = jnp.isfinite(terms)
    safe = jnp.where(finite, terms, -1e30)
    return jnp.where(jnp.any(finite), logsumexp(safe), -jnp.inf)


def eval_log_catalog_prior_state(
    z: float,
    pix: int,
    state: CatalogKernelState,
    em_catalog: EMCatalog,
) -> float:
    """
    log p_cat(z | pix) using a precomputed ``CatalogKernelState``.

    O(N_max) per sample: one row gather, one Gaussian logpdf per galaxy,
    one logsumexp, one 1-D interpolation for log g(z).
    """
    zs = em_catalog.zgals[pix]
    log_kw = state.log_kw[pix]
    sig = state.sig_eff[pix]
    log_mix = _logsumexp_neginf_safe(log_kw + norm.logpdf(z, zs, sig))
    # Volume-weighted (complete-catalog) kernels already carry g(z_i) in their
    # weights, so no front g(z); otherwise reapply the per-sample galaxy measure
    # g(z) that Z_i divided out per kernel.  ``volume_weighted`` is a static bool.
    log_g_front = jnp.where(state.volume_weighted, 0.0, jnp.interp(z, zgrid, state.log_g_grid))
    return log_g_front + log_mix


@jit
def log_catalog_prior(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> float:
    r"""
    Log of the EM-catalog redshift prior at redshift z for catalog row pix.

    Historical scalar signature; builds the per-row kernel state on the
    fly.  Under ``vmap`` over samples, the (cosmo, survey)-only pieces
    (the log g grid) are hoisted automatically; the per-row pieces are
    recomputed per sample, so hot paths should use
    ``catalog_kernel_state`` + ``eval_log_catalog_prior_state`` instead.

    Empty rows return -inf (no host candidates), never NaN.
    """
    zs = em_catalog.zgals[pix]
    dzs = em_catalog.dzgals[pix]
    ws = em_catalog.wgals[pix]
    ngal = None if em_catalog.ngals is None else em_catalog.ngals[pix]

    log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    log_kw, sig_eff = _row_kernel_state(
        zs, dzs, ws, ngal, survey.sigma_kde, log_g_grid, True
    )
    return _logsumexp_neginf_safe(log_kw + norm.logpdf(z, zs, sig_eff))


# Vectorised over (z, pix) pairs — both vmapped simultaneously so the
# call signature matches all prior assembly functions.
log_catalog_prior_vmap = jit(
    vmap(log_catalog_prior, in_axes=(0, 0, None, None, None), out_axes=0)
)
