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
build_pixel_kde_cache(unique_pixels, zgals, n_pix_catalog, wgals=None, ngals=None)
    Precompute KDE grids for unique pixels.  Call once in make_likelihood.

catalog_completion(z, pix, cosmo, survey, em_catalog)
    Returns (f, p_miss, C) for a single (z, pix) pair.

catalog_completion_vmap(z, pix, cosmo, survey, em_catalog)
    Same signature, vectorised over arrays of (z, pix) pairs.
    Computes dN_exp_smooth once internally, then vmaps the inner function.

compute_lss_overdensity(zgals, nside, wgals=None, ngals=None)
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

def _kde_dndz_obs(
    pix: int,
    zgals: jnp.ndarray,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Kernel density estimate of dN_obs/dz for pixel ``pix`` on ``zgrid``.

    The observed-count numerator for completeness uses *raw galaxy counts*,
    not luminosity/completeness weights.  This keeps dN_obs/dz as the direct
    number-count counterpart to the expected ``n0 * dV/dz * (1+z)^delta``
    model.  ``catalog.py`` uses ``wgals`` as base weights for the normalized
    catalog redshift prior; here ``wgals`` is used only as a real-galaxy mask
    when ``ngals`` is unavailable, so padded zero-redshift slots never add
    artificial low-z density.

    Uses module-level ``_K_SMOOTH``.  Safe to vmap over ``pix`` with
    ``in_axes=(0, None, None, None)`` — used by ``build_pixel_kde_cache``.

    Parameters
    ----------
    pix : int or scalar jnp array
    zgals : (N_pix_catalog, N_max_gals)
        Padded galaxy redshift array.
    wgals : (N_pix_catalog, N_max_gals), optional
        Padded galaxy weight array.  A 1D array supplied here is interpreted
        as ``ngals`` for backward-compatible positional calls.  When ``ngals``
        is not supplied, entries with ``wgals[pix] > 0`` are treated as real
        galaxies.  The positive weights are not used as amplitudes in
        dN_obs/dz.
    ngals : (N_pix_catalog,), optional
        Number of real galaxies per pixel.  Preferred over ``wgals`` when
        both are supplied.

    Returns
    -------
    (N_grid,) smoothed observed galaxy density.
    """
    global _K_SMOOTH
    if _K_SMOOTH is None:
        _K_SMOOTH = _build_kernel()

    # Accept a 1D ngals array passed positionally as the third argument.
    if ngals is None and wgals is not None and wgals.ndim == 1:
        ngals = wgals
        wgals = None

    if wgals is None and ngals is None:
        raise ValueError(
            "_kde_dndz_obs requires either wgals or ngals to mask padded galaxies"
        )

    zs = zgals[pix]  # (N_max_gals,) — padded, zeros for empty slots
    if ngals is not None:
        real_gal = jnp.arange(zs.shape[0]) < ngals[pix]
    else:
        real_gal = wgals[pix] > 0

    # Evaluate Gaussian centred at each real galaxy position on the full zgrid.
    # Shape: (N_grid, N_max_gals) → masked sum over galaxies → (N_grid,)
    gaussian = jnp.exp(
        -0.5 * ((zgrid[:, None] - zs[None, :]) / _SIGMA_SMOOTH) ** 2
    )
    raw = (gaussian * real_gal[None, :].astype(gaussian.dtype)).sum(axis=1)
    # Smooth with the row-normalised kernel.
    return _K_SMOOTH @ raw  # (N_grid,)


# ------------------------------------------------------------
# Pixel KDE cache — precomputed at startup
# ------------------------------------------------------------

def build_pixel_kde_cache(
    unique_pixels: np.ndarray,
    zgals: jnp.ndarray,
    n_pix_catalog: int,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
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
    wgals : (N_pix_catalog, N_max_gals), optional
        Galaxy weight array used only to identify real (positive-weight)
        entries when ``ngals`` is unavailable.
    ngals : (N_pix_catalog,), optional
        Number of real galaxies per pixel.  Preferred over ``wgals`` when
        both are supplied.

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
    # Accept a 1D ngals array passed positionally as the fourth argument.
    if ngals is None and wgals is not None and jnp.asarray(wgals).ndim == 1:
        ngals = wgals
        wgals = None

    if wgals is None and ngals is None:
        raise ValueError(
            "build_pixel_kde_cache requires either wgals or ngals to mask padded galaxies"
        )

    wgals_jax = None if wgals is None else jnp.asarray(wgals)
    ngals_jax = None if ngals is None else jnp.asarray(ngals)

    # Batch KDE over unique pixels — one jit+vmap call, not a Python loop.
    _batch_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
    dN_obs_kde = _batch_kde(
        jnp.asarray(unique_pixels, dtype=jnp.int32),
        jnp.asarray(zgals),
        wgals_jax,
        ngals_jax,
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
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    n0, delta = survey.n0, survey.delta
    apix = em_catalog.apix

    dV = dV_of_z(zgrid, H0, Om0, w0, wa)                    # (N_grid,)
    pvol = dV / jnp.trapezoid(dV, zgrid)             # normalised volume element

    global _K_SMOOTH
    if _K_SMOOTH is None:
        _K_SMOOTH = _build_kernel()

    # Expected dN/dz: n0 * apix * dV * (1+z)^delta
    dN_exp_raw = n0 * apix * dV * (1.0 + zgrid) ** delta
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
    The uncached fallback is retained only for direct unit tests and backward
    compatibility; production dark-siren inference should construct the cache
    in ``make_likelihood`` before entering this function.

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
    wgals         = em_catalog.wgals
    ngals         = em_catalog.ngals
    delta_g_pix_z = em_catalog.delta_g_pix_z

    dN_exp_smooth = grids.dN_exp_smooth
    pvol          = grids.pvol

    global_pix = pix if em_catalog.unique_pixels is None else em_catalog.unique_pixels[pix]

    # --- Step 1: Per-pixel observed dN/dz ---
    # Cache path: O(1) lookup — evaluated at trace time, no runtime branch.
    if em_catalog.dN_obs_kde is not None:
        cache_idx = em_catalog.pixel_to_cache_idx[global_pix]
        dN_obs    = em_catalog.dN_obs_kde[cache_idx]   # (N_grid,)
    else:
        # Fallback: recompute on the fly (correct, slower).  This path is
        # retained only for tests/backward compatibility; production
        # dark-siren inference errors before constructing an uncached catalog.
        dN_obs = _kde_dndz_obs(
            pix, zgals, wgals=wgals, ngals=ngals
        )  # (N_grid,)

    # --- Step 2: Differential completeness curve ---
    dN_exp_safe = jnp.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    C_iso = jnp.clip(dN_obs / dN_exp_safe, 0.0, 1.0) * _survey_rolloff(zgrid, z50, w)

    # --- Step 3: Isotropic missing physical density ---
    rho_miss_iso = (1.0 - C_iso) * pvol

    # --- Step 4: LSS-modulated missing density ---
    # Compact catalogs pass ``pix`` as the compact row index.  Translate back
    # to the global HEALPix pixel only for the LSS field, which remains a
    # true full-pixel lookup when LSS is enabled.  Non-LSS runs carry a single
    # dummy row.
    delta_idx = jnp.where(delta_g_pix_z.shape[0] == 1, 0, global_pix)
    delta_g_z = delta_g_pix_z[delta_idx]
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


def _completion_clip_fractions_for_pixel(
    pix: int,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> dict[str, float]:
    """Return clipping fractions on ``zgrid`` for one catalog pixel.

    This diagnostic mirrors ``_catalog_completion_inner`` but keeps the
    pre-clipped arrays so broad or poorly scaled survey parameters can be
    identified before a long sampler run.  Fractions are computed over the
    module redshift grid and therefore have denominator ``len(zgrid)``.
    """
    z50, w, b_miss, alpha_miss = (
        survey.z50, survey.w, survey.b_miss, survey.alpha_miss,
    )
    global_pix = pix if em_catalog.unique_pixels is None else em_catalog.unique_pixels[pix]

    if em_catalog.dN_obs_kde is not None:
        cache_idx = em_catalog.pixel_to_cache_idx[global_pix]
        dN_obs = em_catalog.dN_obs_kde[cache_idx]
    else:
        dN_obs = _kde_dndz_obs(
            pix, em_catalog.zgals, wgals=em_catalog.wgals, ngals=em_catalog.ngals
        )

    dN_exp_smooth = grids.dN_exp_smooth
    pvol = grids.pvol
    dN_exp_safe = jnp.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)

    C_iso_raw = dN_obs / dN_exp_safe
    C_iso_clipped_mask = (C_iso_raw < 0.0) | (C_iso_raw > 1.0)
    C_iso = jnp.clip(C_iso_raw, 0.0, 1.0) * _survey_rolloff(zgrid, z50, w)

    rho_miss_iso = (1.0 - C_iso) * pvol
    delta_idx = jnp.where(em_catalog.delta_g_pix_z.shape[0] == 1, 0, global_pix)
    delta_g_z = em_catalog.delta_g_pix_z[delta_idx]
    delta_g_z = delta_g_z - jnp.mean(delta_g_z)
    rho_miss_lss = rho_miss_iso * (1.0 + b_miss * delta_g_z)

    rho_miss_eff_raw = (
        (1.0 - alpha_miss) * rho_miss_iso + alpha_miss * rho_miss_lss
    )
    rho_miss_eff_clipped_mask = rho_miss_eff_raw < 0.0
    rho_miss_eff = jnp.clip(rho_miss_eff_raw, 0.0, jnp.inf)

    pvol_safe = jnp.where(pvol > 0.0, pvol, 1.0)
    C_eff_raw = 1.0 - rho_miss_eff / pvol_safe
    C_eff_clipped_mask = (C_eff_raw < 0.0) | (C_eff_raw > 1.0)

    return {
        "C_iso_clipped_fraction": float(jnp.mean(C_iso_clipped_mask)),
        "C_eff_clipped_fraction": float(jnp.mean(C_eff_clipped_mask)),
        "rho_miss_eff_clipped_fraction": float(jnp.mean(rho_miss_eff_clipped_mask)),
    }


def completion_clip_diagnostics(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    pixels: np.ndarray | None = None,
    max_pixels: int = 64,
) -> dict[str, object]:
    """Summarise completion clipping over a representative pixel set.

    Parameters
    ----------
    cosmo, survey, em_catalog
        Same containers used by ``catalog_completion``.
    pixels : array-like, optional
        Catalog-row pixel ids to inspect.  For compact catalogs these are row
        indices, not global HEALPix ids.  Defaults to all available compact
        rows, truncated by ``max_pixels``.
    max_pixels : int
        Maximum number of pixels to inspect for an inexpensive dry-run check.

    Returns
    -------
    dict
        JSON-serialisable summary with per-field mean/max clipping fractions
        and per-pixel fractions for the inspected rows.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)

    if pixels is None:
        if em_catalog.unique_pixels is not None:
            pixels_np = np.arange(np.asarray(em_catalog.unique_pixels).size, dtype=np.int32)
        else:
            pixels_np = np.arange(np.asarray(em_catalog.zgals).shape[0], dtype=np.int32)
    else:
        pixels_np = np.asarray(pixels, dtype=np.int32).reshape(-1)

    if max_pixels is not None and max_pixels > 0:
        pixels_np = pixels_np[:max_pixels]

    per_pixel = []
    for pix in pixels_np:
        fractions = _completion_clip_fractions_for_pixel(
            int(pix), grids, survey, em_catalog
        )
        fractions["pixel"] = int(pix)
        if em_catalog.unique_pixels is not None:
            fractions["global_pixel"] = int(np.asarray(em_catalog.unique_pixels)[int(pix)])
        per_pixel.append(fractions)

    fields = [
        "C_iso_clipped_fraction",
        "C_eff_clipped_fraction",
        "rho_miss_eff_clipped_fraction",
    ]
    summary: dict[str, object] = {
        "n_zgrid": int(zgrid.size),
        "z_min": float(zgrid[0]),
        "z_max": float(zgrid[-1]),
        "n_pixels_checked": int(len(per_pixel)),
        "per_pixel": per_pixel,
    }
    for field in fields:
        vals = np.array([item[field] for item in per_pixel], dtype=float)
        summary[f"mean_{field}"] = float(vals.mean()) if vals.size else 0.0
        summary[f"max_{field}"] = float(vals.max()) if vals.size else 0.0

    return summary


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


def compute_lss_overdensity(
    zgals: jnp.ndarray,
    nside: int,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Pre-compute the LSS overdensity field delta_g(pix, z) on ``zgrid``.

    Call once at startup.  Store the result in ``EMCatalog.delta_g_pix_z``.

    Parameters
    ----------
    zgals : (N_pix, N_max_gals) galaxy redshifts (padded).
    nside : HEALPix nside.
    wgals : (N_pix, N_max_gals), optional
        Galaxy weights used only as a real-galaxy mask when ``ngals`` is not
        supplied.
    ngals : (N_pix,), optional
        Number of real galaxies per pixel.  Preferred over ``wgals``.

    Returns
    -------
    delta_g : (N_pix, N_grid) float array.
    """
    import healpy as hp
    n_pix = hp.nside2npix(nside)

    global _K_SMOOTH
    if _K_SMOOTH is None:
        _K_SMOOTH = _build_kernel()
    # Accept a 1D ngals array passed positionally as the third mask argument.
    if ngals is None and wgals is not None and jnp.asarray(wgals).ndim == 1:
        ngals = wgals
        wgals = None

    if wgals is None and ngals is None:
        raise ValueError(
            "compute_lss_overdensity requires either wgals or ngals to mask padded galaxies"
        )

    # KDE for every pixel, then smooth
    _all_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))(
        jnp.arange(n_pix, dtype=jnp.int32),
        jnp.asarray(zgals),
        None if wgals is None else jnp.asarray(wgals),
        None if ngals is None else jnp.asarray(ngals),
    )  # (N_pix, N_grid)

    mean_density = _all_kde.mean(axis=0, keepdims=True)  # (1, N_grid)
    mean_safe    = jnp.where(mean_density > 0.0, mean_density, 1.0)
    delta_g      = (_all_kde - mean_density) / mean_safe  # (N_pix, N_grid)
    return delta_g


# Initialise module-level kernel.
_K_SMOOTH = _build_kernel()