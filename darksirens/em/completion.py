"""
completion.py
-------------
Catalog completion model: characterises the missing-galaxy distribution
for pixels / redshifts where the EM survey is incomplete.

Galaxy number density follows n(z) = n0 (1+z)^delta, so the expected
count per redshift shell scales as:

    dN_exp/dz ∝ n0 * apix * dV_c/dz * (1+z)^delta

The exponent is delta, not (delta-1).  Merger rate evolution is handled
elsewhere in the pipeline and must not appear here.

Completeness model
------------------
Completeness is computed as a *differential* ratio per redshift shell:

    C_iso(z) = clip( (dN_obs/dz)(z) / (dN_exp/dz)(z) , 0, 1 ) * P_rolloff(z)

Both numerator and denominator are smoothed with a Gaussian kernel of
width ``_SIGMA_SMOOTH`` before the ratio is formed.

Per-pixel KDE cache
-------------------
``_kde_dndz_obs`` builds an (N_grid,) density estimate for a given pixel.
Inside a vmap over (z, pix) PE samples, the same pixel may appear hundreds
of times — naively triggering hundreds of identical KDE evaluations.

``build_pixel_kde_cache`` precomputes this KDE for every unique pixel
appearing in the PE and selection sample sets, once at startup.  The result
is stored as two arrays in ``EMCatalog``:

  dN_obs_kde         : (N_unique, N_grid) — KDE grids
  pixel_to_cache_idx : (N_pix_catalog,)  — pixel → row lookup

Inside ``_catalog_completion_inner``, cache presence is detected by
checking ``em_catalog.dN_obs_kde is not None`` at *trace time* (before
JIT).  When the cache is present the inner function uses an O(1) array
lookup; when absent it falls back to the full KDE (correct, slower).

Public API
----------
build_pixel_kde_cache(unique_pixels, zgals, n_pix_catalog)
    Precompute KDE grids for unique pixels.  Call once in make_likelihood.

catalog_completion(z, pix, cosmo, survey, em_catalog)
    Returns (f, p_miss, C) for a single (z, pix) pair.

catalog_completion_vmap(z, pix, cosmo, survey, em_catalog)
    Same signature, vectorised over arrays of (z, pix) pairs.
    Computes dN_exp_smooth once internally, then vmaps the inner function.

compute_lss_overdensity(zgals, nside)
    Pre-computes delta_g(pix, z) on the global zgrid for all HEALPix pixels.
    Call once at startup and store the result in EMCatalog.delta_g_pix_z.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.nn import sigmoid
from typing import NamedTuple

from darksirens.utils.cosmology import dV_of_z
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog

from .utils import zgrid


# Gaussian kernel width for smoothing both dN_obs/dz and dN_exp/dz.
_SIGMA_SMOOTH: float = 0.05

# Module-level kernel matrix.  8 MB for float64.  Built once.
_K_SMOOTH: jnp.ndarray | None = None


def _build_kernel() -> jnp.ndarray:
    K = jnp.exp(
        -0.5 * ((zgrid[:, None] - zgrid[None, :]) / _SIGMA_SMOOTH) ** 2
    )
    return K / K.sum(axis=1, keepdims=True)


# ------------------------------------------------------------
# Per-pixel observed dN/dz via Gaussian KDE
# ------------------------------------------------------------

def _kde_dndz_obs(pix: int, zgals: jnp.ndarray) -> jnp.ndarray:
    """
    Kernel density estimate of dN_obs/dz for pixel ``pix`` on ``zgrid``.

    Uses module-level ``_K_SMOOTH``.  Safe to vmap over ``pix`` with
    ``in_axes=(0, None)`` — used by ``build_pixel_kde_cache``.

    Parameters
    ----------
    pix : int or scalar jnp array
    zgals : (N_pix_catalog, N_max_gals)

    Returns
    -------
    (N_grid,) smoothed observed galaxy density.
    """
    global _K_SMOOTH
    if _K_SMOOTH is None:
        _K_SMOOTH = _build_kernel()

    zs = zgals[pix]  # (N_max_gals,) — padded, zeros for empty slots
    # Evaluate Gaussian centred at each galaxy position on the full zgrid.
    # Shape: (N_grid, N_max_gals) → sum over galaxies → (N_grid,)
    raw = jnp.exp(
        -0.5 * ((zgrid[:, None] - zs[None, :]) / _SIGMA_SMOOTH) ** 2
    ).sum(axis=1)
    # Smooth with the row-normalised kernel.
    return _K_SMOOTH @ raw  # (N_grid,)


# ------------------------------------------------------------
# Pixel KDE cache — precomputed at startup
# ------------------------------------------------------------

def build_pixel_kde_cache(
    unique_pixels: np.ndarray,
    zgals: jnp.ndarray,
    n_pix_catalog: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute ``_kde_dndz_obs`` for all unique pixels in the PE+selection sets.

    Call this *once* in ``make_likelihood`` before the JIT closure is built.
    Typical cost: O(N_unique × N_grid × N_max_gals) ≈ milliseconds for
    N_unique ~ 1000, N_grid = 1000, N_max_gals ~ 50.

    Parameters
    ----------
    unique_pixels : (N_unique,) int array
        Unique HEALPix pixel indices appearing in gw_pe.pixels + gw_sel.pixels.
        Compute with ``np.unique(np.concatenate([pixels_pe, pixels_sel]))``.
    zgals : (N_pix_catalog, N_max_gals)
        Galaxy redshift array from the EM catalog.
    n_pix_catalog : int
        Total number of HEALPix pixels in the catalog
        (``hp.nside2npix(nside)``).

    Returns
    -------
    dN_obs_kde : jnp.ndarray, shape (N_unique, N_grid)
        KDE grids, one row per unique pixel.
    pixel_to_cache_idx : jnp.ndarray, shape (n_pix_catalog,) int32
        Dense lookup: ``pixel_to_cache_idx[p]`` gives the row index of
        pixel ``p`` in ``dN_obs_kde``.  Pixels absent from ``unique_pixels``
        map to 0 (they are never visited during inference).
    """
    global _K_SMOOTH
    if _K_SMOOTH is None:
        _K_SMOOTH = _build_kernel()

    # Batch KDE over unique pixels — one jit+vmap call, not a Python loop.
    _batch_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None)))
    dN_obs_kde = _batch_kde(
        jnp.asarray(unique_pixels, dtype=jnp.int32),
        jnp.asarray(zgals),
    )  # (N_unique, N_grid)

    # Dense lookup array: pixel → index in dN_obs_kde
    pixel_to_cache_idx = np.zeros(n_pix_catalog, dtype=np.int32)
    for i, p in enumerate(unique_pixels):
        pixel_to_cache_idx[int(p)] = i

    return dN_obs_kde, jnp.asarray(pixel_to_cache_idx, dtype=jnp.int32)


# ------------------------------------------------------------
# Precomputed grid bundle (pixel-independent, per cosmo/survey)
# ------------------------------------------------------------

class _CompletionGrids(NamedTuple):
    """Pixel-independent grids computed once per likelihood evaluation."""
    dN_exp_smooth: jnp.ndarray  # (N_grid,) smoothed expected dN/dz
    pvol: jnp.ndarray           # (N_grid,) volume element on zgrid


def _survey_rolloff(z: jnp.ndarray, z50: float, w: float) -> jnp.ndarray:
    """Sigmoid survey roll-off: 1 near z=0, → 0 past z50."""
    return 1.0 - sigmoid((z - z50) / w)


def _precompute_grids(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> _CompletionGrids:
    """
    Compute pixel-independent grids once per likelihood evaluation.

    Both outputs depend only on (cosmo, survey, apix), not on individual
    pixels or redshift samples.  Inside a JIT body, JAX will hoist this
    computation out of any subsequent vmap/scan.

    Returns
    -------
    _CompletionGrids with (dN_exp_smooth, pvol).
    """
    H0, Om0 = cosmo.H0, cosmo.Om0
    n0, delta = survey.n0, survey.delta
    apix = em_catalog.apix

    dV = dV_of_z(zgrid, H0, Om0)                    # (N_grid,)
    pvol = dV / jnp.trapezoid(dV, zgrid)             # normalised volume element

    # Expected dN/dz: n0 * apix * dV * (1+z)^delta
    dN_exp_raw = n0 * apix * dV * (1.0 + zgrid) ** delta

    global _K_SMOOTH
    if _K_SMOOTH is None:
        _K_SMOOTH = _build_kernel()

    dN_exp_smooth = _K_SMOOTH @ dN_exp_raw           # (N_grid,)

    return _CompletionGrids(dN_exp_smooth=dN_exp_smooth, pvol=pvol)


# ------------------------------------------------------------
# Inner completion function (vmappable over z, pix)
# ------------------------------------------------------------

def _catalog_completion_inner(
    z: float,
    pix: int,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Evaluate catalog completion at a single (z, pix) point.

    If ``em_catalog.dN_obs_kde`` is not None (set by ``build_pixel_kde_cache``),
    the per-pixel KDE is fetched via an O(1) array lookup.  Otherwise it is
    recomputed on the fly — correct but O(N_grid × N_max_gals) per call.

    Note: the branch on ``em_catalog.dN_obs_kde is not None`` is evaluated
    at *trace time* (Python-level), not inside JAX's functional graph.
    This means no dynamic branching overhead at runtime.

    Returns
    -------
    f : float
        Scalar pixel-level completeness fraction (diagnostics).
    p_miss : float
        Normalised missing-galaxy PDF at z.
    C : float
        Redshift-dependent completeness C_eff(z|pix).
    """
    z50, w, delta, b_miss, alpha_miss = (
        survey.z50, survey.w, survey.delta, survey.b_miss, survey.alpha_miss,
    )
    zgals         = em_catalog.zgals
    delta_g_pix_z = em_catalog.delta_g_pix_z

    dN_exp_smooth = grids.dN_exp_smooth
    pvol          = grids.pvol

    # --- Step 1: Per-pixel observed dN/dz ---
    # Cache path: O(1) lookup — evaluated at trace time, no runtime branch.
    if em_catalog.dN_obs_kde is not None:
        cache_idx = em_catalog.pixel_to_cache_idx[pix]
        dN_obs    = em_catalog.dN_obs_kde[cache_idx]   # (N_grid,)
    else:
        # Fallback: recompute on the fly (correct, slower).
        dN_obs = _kde_dndz_obs(pix, zgals)             # (N_grid,)

    # --- Step 2: Differential completeness curve ---
    dN_exp_safe = jnp.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    C_iso = jnp.clip(dN_obs / dN_exp_safe, 0.0, 1.0) * _survey_rolloff(zgrid, z50, w)

    # --- Step 3: Isotropic missing physical density ---
    rho_miss_iso = (1.0 - C_iso) * pvol

    # --- Step 4: LSS-modulated missing density ---
    delta_g_z = delta_g_pix_z[pix]
    delta_g_z = delta_g_z - jnp.mean(delta_g_z)
    rho_miss_lss = rho_miss_iso * (1.0 + b_miss * delta_g_z)

    # --- Step 5: Effective missing density (alpha_miss blend) ---
    rho_miss_eff = (1.0 - alpha_miss) * rho_miss_iso + alpha_miss * rho_miss_lss
    rho_miss_eff = jnp.clip(rho_miss_eff, 0.0, jnp.inf)

    # --- Step 6: Effective completeness curve C_eff(z) ---
    pvol_safe = jnp.where(pvol > 0.0, pvol, 1.0)
    C_eff = jnp.clip(1.0 - rho_miss_eff / pvol_safe, 0.0, 1.0)

    # --- Step 7: Scalar completeness fraction (diagnostics) ---
    f = 1.0 - jnp.trapezoid(rho_miss_eff, zgrid) / jnp.trapezoid(pvol, zgrid)

    # --- Step 8: Normalised missing PDF ---
    norm_factor = jnp.trapezoid(rho_miss_eff, zgrid)
    norm_factor = jnp.where(norm_factor > 0.0, norm_factor, 1.0)
    p_miss_grid = rho_miss_eff / norm_factor

    return f, jnp.interp(z, zgrid, p_miss_grid), jnp.interp(z, zgrid, C_eff)


_catalog_completion_inner_vmap = vmap(
    _catalog_completion_inner,
    in_axes=(0, 0, None, None, None),
    out_axes=(0, 0, 0),
)


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

@jit
def catalog_completion(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Characterise catalog incompleteness at a single (z, pix) point.

    Returns
    -------
    f : float — scalar completeness fraction [0, 1] (diagnostics).
    p_miss : float — normalised missing-galaxy PDF at z.
    C : float — C_eff(z|pix), the redshift-dependent mixing weight.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    return _catalog_completion_inner(z, pix, grids, survey, em_catalog)


@jit
def catalog_completion_vmap(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Vectorised catalog completion over arrays of (z, pix) pairs.

    ``_precompute_grids`` runs once; the vmap shares the result.

    Parameters
    ----------
    z : (N,) array
    pix : (N,) int array
    cosmo, survey, em_catalog : shared across the batch.

    Returns
    -------
    f, p_miss, C : (N,) arrays.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    return _catalog_completion_inner_vmap(z, pix, grids, survey, em_catalog)


def compute_lss_overdensity(zgals: jnp.ndarray, nside: int) -> jnp.ndarray:
    """
    Pre-compute the LSS overdensity field delta_g(pix, z) on ``zgrid``.

    Call once at startup.  Store the result in ``EMCatalog.delta_g_pix_z``.

    Parameters
    ----------
    zgals : (N_pix, N_max_gals) galaxy redshifts (padded).
    nside : HEALPix nside.

    Returns
    -------
    delta_g : (N_pix, N_grid) float array.
    """
    import healpy as hp
    n_pix = hp.nside2npix(nside)

    global _K_SMOOTH
    if _K_SMOOTH is None:
        _K_SMOOTH = _build_kernel()

    # KDE for every pixel, then smooth
    _all_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None)))(
        jnp.arange(n_pix, dtype=jnp.int32), jnp.asarray(zgals)
    )  # (N_pix, N_grid)

    mean_density = _all_kde.mean(axis=0, keepdims=True)  # (1, N_grid)
    mean_safe    = jnp.where(mean_density > 0.0, mean_density, 1.0)
    delta_g      = (_all_kde - mean_density) / mean_safe  # (N_pix, N_grid)
    return delta_g


# Initialise module-level kernel.
_K_SMOOTH = _build_kernel()