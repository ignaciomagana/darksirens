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
from jax import jit, vmap
from jax.scipy.special import logsumexp, ndtr, ndtri
from jax.scipy.stats import norm
from typing import NamedTuple

from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog

from .utils import zgrid
from .completion import log_galaxy_measure_grid

_ZMAX: float = float(np.asarray(zgrid)[-1])

#: Numerical floor on sigma_eff [redshift units]; ~30 km/s, well below any
#: physical photo-z or peculiar-velocity scale.  Protects against
#: spectroscopic dzgals -> 0 with sigma_kde = 0.
SIGMA_EFF_FLOOR: float = 1e-4

# Gauss–Legendre nodes/weights on [0, 1] for the kernel normalisation.
_GL_NODES = 24
_glx, _glw = np.polynomial.legendre.leggauss(_GL_NODES)
_GL_X = jnp.asarray(0.5 * (_glx + 1.0))
_GL_W = jnp.asarray(0.5 * _glw)


def _row_real_mask(zs, ws, ngal):
    """Real-galaxy mask for one padded row."""
    if ngal is not None:
        return jnp.arange(zs.shape[0]) < ngal
    return ws > 0


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


def _row_kernel_state(zs, dzs, ws, ngal, sigma_kde, log_g_grid):
    """
    Per-galaxy kernel quantities for one row.

    Returns
    -------
    log_kw : (N_max,) log[ w~_i / Z_i ]  (-inf for padded slots and for
        empty rows, where the weight normalisation is undefined).
    sig_eff : (N_max,) effective kernel widths (floored).
    """
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR)

    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1e-300)), -jnp.inf)
    lse = logsumexp(log_w)
    has_galaxies = jnp.isfinite(lse)
    log_w_norm = jnp.where(real, log_w - jnp.where(has_galaxies, lse, 0.0), -jnp.inf)

    log_Z = _row_log_kernel_norms(zs, sig_eff, real, log_g_grid)
    log_kw = jnp.where(real, log_w_norm - log_Z, -jnp.inf)
    return log_kw, sig_eff


class CatalogKernelState(NamedTuple):
    """Per-galaxy kernel quantities for all catalog rows (one proposal)."""
    log_g_grid: jnp.ndarray  # (N_grid,)
    log_kw: jnp.ndarray      # (N_rows, N_max) log[w~_i / Z_i]
    sig_eff: jnp.ndarray     # (N_rows, N_max)


def catalog_kernel_state(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    log_g_grid: jnp.ndarray | None = None,
) -> CatalogKernelState:
    """Precompute per-galaxy kernel quantities once per parameter proposal."""
    if log_g_grid is None:
        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals
    ngals = em_catalog.ngals

    if ngals is not None:
        per_row = vmap(_row_kernel_state, in_axes=(0, 0, 0, 0, None, None))
        log_kw, sig_eff = per_row(
            zgals, dzgals, wgals, ngals, survey.sigma_kde, log_g_grid
        )
    else:
        per_row = vmap(
            lambda zs, dzs, ws: _row_kernel_state(
                zs, dzs, ws, None, survey.sigma_kde, log_g_grid
            ),
            in_axes=(0, 0, 0),
        )
        log_kw, sig_eff = per_row(zgals, dzgals, wgals)

    return CatalogKernelState(log_g_grid=log_g_grid, log_kw=log_kw, sig_eff=sig_eff)


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
    log_g_z = jnp.interp(z, zgrid, state.log_g_grid)
    return log_g_z + logsumexp(log_kw + norm.logpdf(z, zs, sig))


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
        zs, dzs, ws, ngal, survey.sigma_kde, log_g_grid
    )
    log_g_z = jnp.interp(z, zgrid, log_g_grid)
    return log_g_z + logsumexp(log_kw + norm.logpdf(z, zs, sig_eff))


# Vectorised over (z, pix) pairs — both vmapped simultaneously so the
# call signature matches all prior assembly functions.
log_catalog_prior_vmap = jit(
    vmap(log_catalog_prior, in_axes=(0, 0, None, None, None), out_axes=0)
)
