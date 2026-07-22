"""
completion.py
-------------
Catalog completion model: characterises the missing-galaxy distribution
for pixels / redshifts where the EM survey is incomplete.

Model
-----
Galaxy number density follows n(z) = n0 (1+z)^delta, so the expected
count per redshift shell in a pixel of solid angle ``apix`` is

    dN_exp/dz = n0 * apix * dV_c/dz * (1+z)^delta  =  n0 * apix * g(z),

with ``g(z) = dV_c/dz * (1+z)^delta`` the *galaxy measure*.  The exponent
is delta, not (delta-1); merger-rate evolution is handled by the
population model and must not appear here.

Completeness is the matched-kernel differential ratio

    C(z|pix) = clip( dN_obs_s(z|pix) / dN_exp_s(z) , 0, 1 ),

where *both* sides are smoothed by the same linear operator: a Gaussian
kernel of width ``_SIGMA_SMOOTH``, truncated to [0, zmax] and
renormalised per source point.  The numerator is the per-galaxy KDE
(each observed galaxy contributes a unit-mass truncated kernel); the
denominator applies the identical operator to dN_exp/dz via quadrature
on ``zgrid``.  Because the operator is linear and shared, a constant
true completeness passes through the ratio exactly, including at the
z = 0 boundary.  There is no parametric roll-off: the ratio itself is
the completeness estimator (survey depth shows up as the data-driven
decline of dN_obs_s).

The missing-galaxy *density* (count units, per unit z) is

    dN_miss(z|pix) = (1 - C(z|pix)) * dN_exp(z) * max(1 + b_eff * delta_g(pix,z), 0),

with ``b_eff = alpha_miss * b_miss``.  (``alpha_miss`` and ``b_miss``
enter the model only through this product — they are exactly degenerate —
so only ``b_miss`` is sampled and ``alpha_miss`` defaults to 1.)
``dN_miss`` carries the same (1+z)^delta evolution as dN_exp, and is the
quantity the assembled redshift prior adds to the catalog counts; see
``darksirens/redshift/prior.py``.

Per-pixel KDE cache
-------------------
``_kde_dndz_obs`` builds an (N_grid,) smoothed observed density for one
pixel.  ``build_pixel_kde_cache`` precomputes it for every unique pixel
appearing in the PE and selection sample sets, once at startup, stored
as ``EMCatalog.dN_obs_kde`` with the lookup ``EMCatalog.pixel_to_cache_idx``
indexed by the *same key space the cache was built with* (global HEALPix
pixel in the production union path; see
``darksirens/likelihood/catalog_views.py``).

Public API
----------
build_pixel_kde_cache(unique_pixels, zgals, n_pix_catalog, wgals=None, ngals=None)
catalog_completion(z, pix, cosmo, survey, em_catalog)        -> (f, p_miss, C_eff)
catalog_completion_vmap(z, pix, cosmo, survey, em_catalog)   -> vectorised
completion_curves(cosmo, survey, em_catalog)                 -> per-row curve bundle
compute_lss_overdensity(zgals, nside, wgals=None, ngals=None)
log_galaxy_measure_grid(cosmo, survey)                       -> log g(zgrid)

The scalar/vmap entry points recompute the per-row curves per call and
are intended for checks and diagnostics.  Hot paths (the likelihood)
should use ``completion_curves`` once per parameter proposal via
``darksirens.redshift.prior.prepare_redshift_prior_state``.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import ndtr
from typing import NamedTuple

from darksirens.utils.cosmology import dV_of_z
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog

from darksirens.redshift.grid import zgrid


# Gaussian kernel width for the completeness ratio (both sides).
_SIGMA_SMOOTH: float = 0.05

_ZMAX: float = float(np.asarray(zgrid)[-1])
_SQRT2PI: float = float(np.sqrt(2.0 * np.pi))


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Trapezoidal quadrature weights for a (possibly non-uniform) grid."""
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += 0.5 * dx
    w[1:] += 0.5 * dx
    return w


_TRAPW_NP = _trapezoid_weights(np.asarray(zgrid, dtype=np.float64))
_TRAPW = jnp.asarray(_TRAPW_NP)


def _truncated_kernel_mass_np(z0: np.ndarray, sigma: float) -> np.ndarray:
    """Mass of N(.; z0, sigma) inside [0, zmax] (NumPy, import time)."""
    from scipy.special import ndtr as _ndtr
    return _ndtr((_ZMAX - z0) / sigma) - _ndtr(-z0 / sigma)


def _build_smoothing_operator() -> jnp.ndarray:
    """
    Linear operator S such that (S @ f) approximates

        (S f)(z_g) = ∫_0^zmax  N(z_g; z', σ) / M(z')  f(z') dz'

    via trapezoid quadrature on ``zgrid``, where M(z') is the truncated
    kernel mass.  This is the *same* per-source-point kernel used by the
    observed-galaxy KDE, so ratios of S-smoothed quantities are exact for
    constant completeness, boundary included.
    """
    z = np.asarray(zgrid, dtype=np.float64)
    M = np.maximum(_truncated_kernel_mass_np(z, _SIGMA_SMOOTH), 1e-300)
    pdf = np.exp(-0.5 * ((z[:, None] - z[None, :]) / _SIGMA_SMOOTH) ** 2)
    pdf /= _SQRT2PI * _SIGMA_SMOOTH
    S = (pdf / M[None, :]) * _TRAPW_NP[None, :]
    return jnp.asarray(S)


# (N_grid, N_grid) ~8 MB float64; built once at import.
_S_EXP: jnp.ndarray = _build_smoothing_operator()


# ------------------------------------------------------------
# Shared galaxy measure g(z) = dV/dz * (1+z)^delta
# ------------------------------------------------------------

def log_galaxy_measure_grid(cosmo: CosmoParams, survey: SurveyParams) -> jnp.ndarray:
    """
    log g(zgrid) with g(z) = dV_c/dz(z) * (1+z)^delta.

    g(0) = 0 (dV vanishes); the log is floored so that interpolation near
    z = 0 stays finite (the prior density correctly -> 0 there).
    Depends only on (cosmo, survey.delta) — inside a JIT trace JAX hoists
    this out of any vmap over samples.
    """
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    g = dV_of_z(zgrid, H0, Om0, w0, wa) * (1.0 + zgrid) ** survey.delta
    return jnp.log(jnp.maximum(g, 1e-300))


# ------------------------------------------------------------
# Per-pixel observed dN/dz via boundary-corrected Gaussian KDE
# ------------------------------------------------------------

def _kde_dndz_obs(
    pix: int,
    zgals: jnp.ndarray,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Smoothed observed number density dN_obs_s/dz for pixel ``pix`` on ``zgrid``.

    Each *real* galaxy contributes a unit-mass Gaussian of width
    ``_SIGMA_SMOOTH`` truncated and renormalised on [0, zmax]:

        k(z; z_i) = N(z; z_i, σ) / M(z_i).

    The estimate is Σ_i k(z; z_i) — properly normalised (counts per unit
    z) and boundary-corrected, matching the operator applied to the
    expected counts.  Raw *counts* are used (no luminosity weights): this
    keeps the numerator the direct counterpart of n0·apix·g(z).
    ``wgals``/``ngals`` serve only to mask padded slots.

    Safe to vmap over ``pix`` with ``in_axes=(0, None, None, None)``.
    """
    # Accept a 1D ngals array passed positionally as the third argument.
    if ngals is None and wgals is not None and wgals.ndim == 1:
        ngals = wgals
        wgals = None
    if wgals is None and ngals is None:
        raise ValueError(
            "_kde_dndz_obs requires either wgals or ngals to mask padded galaxies"
        )

    zs = zgals[pix]  # (N_max_gals,)
    if ngals is not None:
        real_gal = jnp.arange(zs.shape[0]) < ngals[pix]
    else:
        real_gal = wgals[pix] > 0

    mass = ndtr((_ZMAX - zs) / _SIGMA_SMOOTH) - ndtr(-zs / _SIGMA_SMOOTH)
    mass = jnp.maximum(mass, 1e-300)
    pdf = jnp.exp(-0.5 * ((zgrid[:, None] - zs[None, :]) / _SIGMA_SMOOTH) ** 2)
    pdf = pdf / (_SQRT2PI * _SIGMA_SMOOTH)
    kern = (pdf / mass[None, :]) * real_gal[None, :].astype(pdf.dtype)
    return kern.sum(axis=1)  # (N_grid,)


# ------------------------------------------------------------
# Pixel KDE cache — precomputed at startup
# ------------------------------------------------------------

def build_pixel_kde_cache(
    unique_pixels: np.ndarray,
    zgals: jnp.ndarray,
    n_pix_catalog: int,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
    batch_size: int = 512,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute ``_kde_dndz_obs`` for all unique pixels.  Call once in
    ``make_likelihood`` before the JIT closure is built.

    Computation is chunked (``batch_size`` pixels per JIT call) to bound
    peak memory at high nside.

    Returns
    -------
    dN_obs_kde : (N_unique, N_grid) — smoothed observed densities.
    pixel_to_cache_idx : (n_pix_catalog,) int32 — lookup from the index
        space of ``unique_pixels`` to rows of ``dN_obs_kde``.  Whatever
        key space ``unique_pixels`` lives in (global HEALPix pixels or
        compact rows) is the key space of this lookup; callers must index
        it consistently (see catalog_views.py for the compact-row case).
    """
    if ngals is None and wgals is not None and jnp.asarray(wgals).ndim == 1:
        ngals = wgals
        wgals = None
    if wgals is None and ngals is None:
        raise ValueError(
            "build_pixel_kde_cache requires either wgals or ngals to mask padded galaxies"
        )

    zgals_jax = jnp.asarray(zgals)
    wgals_jax = None if wgals is None else jnp.asarray(wgals)
    ngals_jax = None if ngals is None else jnp.asarray(ngals)

    pix_idx = np.asarray(unique_pixels)
    n_unique = int(pix_idx.size)

    _batch_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
    dN_obs_kde = np.empty((n_unique, zgrid.size), dtype=np.float64)
    for start in range(0, n_unique, batch_size):
        stop = min(start + batch_size, n_unique)
        dN_obs_kde[start:stop] = np.asarray(
            _batch_kde(
                jnp.asarray(pix_idx[start:stop], dtype=jnp.int32),
                zgals_jax,
                wgals_jax,
                ngals_jax,
            )
        )
    dN_obs_kde = jnp.asarray(dN_obs_kde)

    pixel_to_cache_idx = np.zeros(n_pix_catalog, dtype=np.int32)
    for i, p in enumerate(unique_pixels):
        pixel_to_cache_idx[int(p)] = i

    return dN_obs_kde, jnp.asarray(pixel_to_cache_idx, dtype=jnp.int32)


# ------------------------------------------------------------
# FIELD-convention sky-weighting: survey-global normalization inputs
# ------------------------------------------------------------

class FieldNormalizationInputs(NamedTuple):
    """Survey-global ingredients for the FIELD-convention normalizer.

    ``dN_obs_s / n_empty / N_obs_total`` are the legacy trio;
    ``occupied_pixels`` keys the per-pixel budget modulations
    (:func:`build_field_lss_q_inputs`, :func:`build_field_delta_g_inputs`) to
    the same row order as ``dN_obs_s``.
    """

    dN_obs_s: jnp.ndarray          # (n_occupied, N_grid) float32
    n_empty: int
    N_obs_total: float
    occupied_pixels: np.ndarray    # (n_occupied,) int32, host-side


def build_field_normalization_inputs(
    full_z: jnp.ndarray,
    full_w: jnp.ndarray | None,
    full_n: jnp.ndarray | None,
    batch_size: int = 4096,
) -> FieldNormalizationInputs:
    """Host-side precompute for the FIELD-convention global normalizer.

    Builds the survey-global ingredients consumed by :func:`field_global_log_Z`
    to normalize the catalog redshift prior by the GLOBAL count
    ``Z(theta) = Sum_all-pixels [N_obs,pix + N_miss,pix(theta)]`` instead of the
    per-pixel ``Z[pix]``.  Called ONCE at startup (before compaction /
    ``--drop_full_catalog``), over the FULL-sky catalog rows so empty pixels are
    counted.

    The observed density is the SAME smoothed per-galaxy KDE the in-likelihood
    completion uses (``_kde_dndz_obs``, raw counts, matched truncated kernel),
    so the per-pixel completeness ``C`` computed in ``field_global_log_Z``
    reproduces exactly what ``_assemble_curves`` computes per occupied pixel.

    Parameters
    ----------
    full_z : (N_pix, N_max_gals)
        Full-sky padded galaxy redshifts (one row per HEALPix pixel).
    full_w : (N_pix, N_max_gals) or None
        Padded galaxy base weights (used only to mask padded slots when
        ``full_n`` is absent).
    full_n : (N_pix,) or None
        Real-galaxy count per pixel.  Preferred padding mask and the source of
        ``N_obs_total``; when ``None`` falls back to ``full_w > 0``.

    Returns
    -------
    FieldNormalizationInputs
        ``dN_obs_s`` (n_occupied, N_grid) float32 smoothed observed density for
        occupied pixels; ``n_empty`` galaxy-free pixel count; ``N_obs_total``
        total observed real-galaxy count; ``occupied_pixels`` (n_occupied,)
        int32 global pixel ids keying the per-pixel budget modulations.
    """
    full_z = np.asarray(full_z)
    n_pix_total = int(full_z.shape[0])

    if full_n is not None:
        ngals_np = np.asarray(full_n).reshape(-1).astype(np.int64)
        occupied = ngals_np > 0
        N_obs_total = float(ngals_np.sum())
    elif full_w is not None:
        full_w_np = np.asarray(full_w)
        per_pix = (full_w_np > 0.0).sum(axis=-1)
        occupied = per_pix > 0
        N_obs_total = float(per_pix.sum())
    else:
        raise ValueError(
            "build_field_normalization_inputs requires either full_w or full_n "
            "to mask padded galaxies and count observed galaxies."
        )

    occupied_pixels = np.nonzero(occupied)[0].astype(np.int32)
    n_occ = int(occupied_pixels.size)
    n_empty = int(n_pix_total - n_occ)

    zgals_jax = jnp.asarray(full_z)
    wgals_jax = None if full_w is None else jnp.asarray(full_w)
    ngals_jax = None if full_n is None else jnp.asarray(full_n)

    _batch_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
    dN_obs_s = np.empty((n_occ, zgrid.size), dtype=np.float64)
    for start in range(0, n_occ, batch_size):
        stop = min(start + batch_size, n_occ)
        dN_obs_s[start:stop] = np.asarray(
            _batch_kde(
                jnp.asarray(occupied_pixels[start:stop], dtype=jnp.int32),
                zgals_jax,
                wgals_jax,
                ngals_jax,
            )
        )

    # Store as float32: for a full-sky (n_occupied, N_grid) table this halves the
    # device footprint; it only feeds the GLOBAL normalizer (a survey-scale
    # constant) so the f32 rounding is immaterial to the estimand.
    field_dN_obs_s = jnp.asarray(dN_obs_s, dtype=jnp.float32)
    return FieldNormalizationInputs(
        dN_obs_s=field_dN_obs_s,
        n_empty=n_empty,
        N_obs_total=N_obs_total,
        occupied_pixels=occupied_pixels,
    )


def build_field_lss_q_inputs(
    logq_map: jnp.ndarray,
    occupied_pixels: np.ndarray,
    n_pix_total: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Field-normalizer Q_LSS modulation rows from a GLOBAL log-Q table.

    ``logq_map`` must be globally pixel-indexed with exactly ``n_pix_total``
    rows (the field normalizer sums over the FULL sky, so a compact per-view
    table cannot supply the empty-pixel budget).  Returns

    - ``field_lss_q``: (n_occupied, N_grid) float32 LINEAR Q rows aligned with
      ``FieldNormalizationInputs.occupied_pixels`` / ``field_dN_obs_s``;
    - ``field_lss_q_empty_sum``: (N_grid,) float64 ``Sum_{p empty} Q_p(z)`` —
      a data constant (Q is loaded data, never theta-dependent), the
      empty-pixel budget curve (empty pixels have C == 0).
    """
    logq_np = np.asarray(logq_map, dtype=np.float64)
    if logq_np.ndim != 2 or logq_np.shape[0] != int(n_pix_total):
        raise ValueError(
            "build_field_lss_q_inputs requires a GLOBAL (n_pix_total, N_grid) "
            f"log-Q table; got shape {logq_np.shape} for n_pix_total="
            f"{int(n_pix_total)}. The field normalizer sums the full sky, so a "
            "compact per-view Q table cannot be used -- rebuild Q over the "
            "full nside."
        )
    if logq_np.shape[1] != int(zgrid.size):
        raise ValueError(
            f"log-Q table has {logq_np.shape[1]} grid nodes but zgrid has "
            f"{int(zgrid.size)}."
        )
    occ = np.asarray(occupied_pixels, dtype=np.int64).reshape(-1)
    q = np.exp(logq_np)
    occ_mask = np.zeros(int(n_pix_total), dtype=bool)
    occ_mask[occ] = True
    q_occ = jnp.asarray(q[occ], dtype=jnp.float32)
    q_empty_sum = jnp.asarray(q[~occ_mask].sum(axis=0), dtype=jnp.float64)
    return q_occ, q_empty_sum


def build_field_lss_q_member_inputs(
    logq_members: jnp.ndarray,
    occupied_pixels: np.ndarray,
    n_pix_total: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Per-member field-normalizer Q rows from a GLOBAL (M, n_pix, N_grid)
    log-Q ensemble: the member analogue of :func:`build_field_lss_q_inputs`.

    Returns ``(field_lss_q_members, field_lss_q_empty_sum_members)`` —
    (M, n_occupied, N_grid) float32 and (M, N_grid) float64.
    """
    logq_np = np.asarray(logq_members, dtype=np.float64)
    if logq_np.ndim != 3 or logq_np.shape[1] != int(n_pix_total):
        raise ValueError(
            "build_field_lss_q_member_inputs requires a GLOBAL "
            "(M, n_pix_total, N_grid) log-Q ensemble; got shape "
            f"{logq_np.shape} for n_pix_total={int(n_pix_total)}."
        )
    if logq_np.shape[2] != int(zgrid.size):
        raise ValueError(
            f"log-Q ensemble has {logq_np.shape[2]} grid nodes but zgrid has "
            f"{int(zgrid.size)}."
        )
    occ = np.asarray(occupied_pixels, dtype=np.int64).reshape(-1)
    q = np.exp(logq_np)
    occ_mask = np.zeros(int(n_pix_total), dtype=bool)
    occ_mask[occ] = True
    q_occ = jnp.asarray(q[:, occ, :], dtype=jnp.float32)
    q_empty_sum = jnp.asarray(q[:, ~occ_mask, :].sum(axis=1), dtype=jnp.float64)
    return q_occ, q_empty_sum


def build_field_delta_g_inputs(
    delta_g_pix_z: jnp.ndarray,
    occupied_pixels: np.ndarray,
) -> jnp.ndarray:
    """Field-normalizer overdensity rows: ``delta_g`` gathered to occupied pixels.

    ``delta_g_pix_z`` must be the REAL per-pixel table (first axis == n_pix;
    the (1, N_grid) dummy means "no modulation" and must not reach here).
    Empty pixels carry ``delta_g == 0`` by construction
    (:func:`compute_lss_overdensity`), so their budget factor is exactly 1 and
    only the occupied rows are stored.
    """
    dg = np.asarray(delta_g_pix_z)
    if dg.ndim != 2 or dg.shape[0] <= 1:
        raise ValueError(
            "build_field_delta_g_inputs requires the real per-pixel "
            f"delta_g table; got shape {dg.shape} (the (1, N_grid) dummy "
            "means no modulation)."
        )
    occ = np.asarray(occupied_pixels, dtype=np.int64).reshape(-1)
    if occ.size and int(occ.max()) >= dg.shape[0]:
        raise ValueError(
            f"delta_g table has {dg.shape[0]} pixel rows but an occupied "
            f"pixel index reaches {int(occ.max())}."
        )
    return jnp.asarray(dg[occ], dtype=jnp.float32)


# ------------------------------------------------------------
# Precomputed grid bundle (pixel-independent, per cosmo/survey)
# ------------------------------------------------------------

class _CompletionGrids(NamedTuple):
    """Pixel-independent grids computed once per likelihood evaluation."""
    log_g: jnp.ndarray          # (N_grid,) log galaxy measure
    dN_exp: jnp.ndarray         # (N_grid,) n0 * apix * g(z)
    dN_exp_smooth: jnp.ndarray  # (N_grid,) S-smoothed expected counts


def _precompute_grids(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> _CompletionGrids:
    log_g = log_galaxy_measure_grid(cosmo, survey)
    dN_exp = survey.n0 * em_catalog.apix * jnp.exp(log_g)
    dN_exp_smooth = _S_EXP @ dN_exp
    return _CompletionGrids(log_g=log_g, dN_exp=dN_exp, dN_exp_smooth=dN_exp_smooth)


# ------------------------------------------------------------
# Per-row completion curves (the unit the state API vmaps over)
# ------------------------------------------------------------

class CompletionCurves(NamedTuple):
    """Per-catalog-row completion outputs on ``zgrid``.

    ``base_miss`` / ``N_miss_members`` are populated only when the catalog
    carries an LSS-completion *ensemble* (``lss_completion_*_members``); they
    support the Bayesian redshift-prior diagnostic and the ``lss_marginalize``
    member vmap, and default to ``None`` (the scalar API and the deterministic GW
    likelihood ignore them).

    ``base_miss`` is the member-INDEPENDENT missing-density base ``(1 - C)
    dN_exp`` (with the ``z_depth`` relaxation to ``dN_exp`` beyond the depth
    already applied), so member ``m``'s missing density is exactly
    ``base_miss * Q_eff_m`` with ``Q_eff_m == 1`` beyond the depth -- the full
    ``(M, N_rows, N_grid)`` member cube is NEVER materialised; it is reconstructed
    two grid nodes at a time at query redshifts (see
    :func:`darksirens.redshift.prior.eval_dark_member_completion`).
    ``N_miss_members`` is the grid-integrated per-member missing mass ``(M,
    N_rows)`` -- a compact scalar table computed by the streamed
    :func:`member_N_miss_integrals` (no cube).
    """
    f: jnp.ndarray        # (N_rows,) scalar number-weighted completeness (diagnostic)
    dN_miss: jnp.ndarray  # (N_rows, N_grid) missing-galaxy density [counts / unit z]
    C_eff: jnp.ndarray    # (N_rows, N_grid) effective completeness (diagnostic)
    N_miss: jnp.ndarray   # (N_rows,) integral of dN_miss
    base_miss: jnp.ndarray = None        # (N_rows, N_grid) member-independent base | None
    N_miss_members: jnp.ndarray = None   # (M, N_rows) | None


# ------------------------------------------------------------
# LSS-conditioned lognormal completion factor Q_LSS (optional)
# ------------------------------------------------------------

#: log Q is clipped to this symmetric range before exponentiating, so that a
#: heavy lognormal tail cannot blow up the missing-galaxy density.
_LOGQ_CLIP: float = 7.0


def _check_lss_grid(arr):
    """Static ``N_grid == zgrid.size`` check for a Q_LSS table (no interpolation)."""
    if arr.shape[-1] != int(zgrid.size):
        raise ValueError(
            f"LSS completion table has N_grid={arr.shape[-1]} but the package "
            f"zgrid has size {int(zgrid.size)}; the offline builder must use the "
            "same redshift grid (no silent interpolation)."
        )


def _row_align_lss(table, row_axis, em_catalog: EMCatalog, n_rows: int):
    """Align a Q_LSS table's ``row_axis`` to catalog row order.

    Compact-vs-global is decided by STATIC shapes only (``K`` and ``n_rows`` are
    concrete even under a trace, so nothing here concretises a tracer).  In the
    jitted likelihood the table is ALWAYS already compact -- the factory slices
    global->compact before the jit -- so ``K == n_rows`` and it is returned
    untouched (the hot path; the returned object is a VIEW of the resident data
    constant, so no ``(M, N_rows, N_grid)`` copy is made).  ``K != n_rows`` is
    reachable ONLY on the eager path (a full global table handed straight to
    ``completion_curves`` by a test or diagnostic), never under jit, so
    ``unique_pixels`` is concrete here and the coverage check is safe.
    """
    K = table.shape[row_axis]
    if K == n_rows:
        return table
    if em_catalog.unique_pixels is None:
        idx = jnp.arange(n_rows, dtype=jnp.int32)
    else:
        idx = jnp.asarray(em_catalog.unique_pixels, dtype=jnp.int32)
    if int(idx.max()) >= K:
        raise ValueError(
            f"global LSS completion table has {K} rows but a catalog pixel index "
            f"reaches {int(idx.max())}; the table does not cover all catalog pixels "
            "(rebuild Q over the full nside, or pass a compact table)."
        )
    return jnp.take(table, idx, axis=row_axis)


def _to_q(logq_arr, q_arr):
    """Clip-and-exp a Q_LSS table: ``exp(clip(logQ, ±_LOGQ_CLIP))``.

    Accepts either a log table (``logq_arr``) or a linear table (``q_arr``, then
    log it first).  Used for the (small, ``(N_rows, N_grid)``) DETERMINISTIC Q
    only -- the member ensemble defers this to the 2-node gathers / streamed chunk
    so no ``(M, N_rows, N_grid)`` intermediate is built.
    """
    if logq_arr is not None:
        lq = jnp.clip(jnp.asarray(logq_arr, dtype=float), -_LOGQ_CLIP, _LOGQ_CLIP)
    elif q_arr is not None:
        lq = jnp.clip(
            jnp.log(jnp.maximum(jnp.asarray(q_arr, dtype=float), 1e-300)),
            -_LOGQ_CLIP, _LOGQ_CLIP,
        )
    else:
        return None
    return jnp.exp(lq)


def _member_q_eff_from_logq(logq_arr, depth_mask, member_is_log: bool):
    """One member's completion factor ``Q_eff = exp(clip(logQ, ±_LOGQ_CLIP))``,
    relaxed to ``1`` beyond ``survey.z_depth``.

    ``logq_arr`` is a RAW (un-clipped, un-exponentiated) log table when
    ``member_is_log`` else a linear-Q table (then log it first).  Bit-identical at
    grid nodes to the pre-existing whole-table ``_to_q`` followed by the
    ``_completion_member_row`` depth relaxation, but applied to whatever slice /
    2-node gather the caller passes (never the full cube).  ``depth_mask`` is a
    static ``(N_grid,)`` (or gathered ``(...,)``) boolean, or ``None`` when
    ``z_depth is None`` (Q_eff is then just ``exp(clip(logQ))`` everywhere).
    """
    if not member_is_log:
        logq_arr = jnp.log(jnp.maximum(logq_arr, 1e-300))
    q = jnp.exp(jnp.clip(logq_arr, -_LOGQ_CLIP, _LOGQ_CLIP))
    if depth_mask is None:
        return q
    return jnp.where(depth_mask, q, 1.0)


def _resolve_member_logq_row(em_catalog: EMCatalog):
    """Row-aligned RAW member log-Q table ``(M, N_rows, N_grid)`` + ``is_log`` flag.

    Returns the ensemble table WITHOUT clipping or exponentiating -- both are
    deferred to the 2-node gathers in the evaluator and to the per-member chunk in
    :func:`member_N_miss_integrals` -- so no ``(M, N_rows, N_grid)`` intermediate
    is ever built on the hot jit path (there the table is already compact and
    row-aligned, ``K == N_rows``, and is returned untouched, a view of the
    resident data constant).  ``member_is_log`` is ``True`` for a log-Q table and
    ``False`` for a linear-Q table (then the consumer takes ``log`` at the gathered
    nodes).  Returns ``(None, True)`` when the catalog carries no ensemble.
    """
    logq_m = em_catalog.lss_completion_logq_members
    q_m = em_catalog.lss_completion_q_members
    if logq_m is None and q_m is None:
        return None, True
    n_rows = em_catalog.zgals.shape[0]  # static int (shape) -- safe under trace
    if logq_m is not None:
        tab, is_log = jnp.asarray(logq_m, dtype=float), True
    else:
        tab, is_log = jnp.asarray(q_m, dtype=float), False
    _check_lss_grid(tab)
    return _row_align_lss(tab, 1, em_catalog, n_rows), is_log


def _member_posterior_mean_q(logq_members_row, member_is_log: bool):
    """Posterior-MEAN Q ``(N_rows, N_grid)`` = ``mean_m exp(clip(logQ_m))``.

    Streamed with ``lax.scan`` over the M members so the ``(M, N_rows, N_grid)``
    exponentiated cube is never materialised; bit-identical at grid nodes to the
    old ``jnp.mean(exp(clip(logQ_members)), axis=member)``.  Q is theta-INDEPENDENT
    (it is loaded data), so this carries no reverse-mode cost.  No depth relaxation
    here -- the deterministic path applies ``z_depth`` downstream in
    ``_assemble_curves`` exactly as before.
    """
    M = logq_members_row.shape[0]

    def _body(acc, logq_m):
        return acc + _member_q_eff_from_logq(logq_m, None, member_is_log), None

    total, _ = lax.scan(
        _body, jnp.zeros(logq_members_row.shape[1:], dtype=float), logq_members_row
    )
    return total / M


def member_N_miss_integrals(base_miss, em_catalog: EMCatalog, survey: SurveyParams):
    """Streamed per-member missing-mass integrals ``N_miss_members`` ``(M, N_rows)``.

    ``N_miss[m, row] = integral base_miss[row] * Q_eff_m[row] dz`` on ``zgrid``,
    computed WITHOUT ever materialising the ``(M, N_rows, N_grid)`` member density
    cube: a ``lax.scan`` over the M ensemble members forms ONE member's ``(N_rows,
    N_grid)`` density from the SHARED ``base_miss`` and that member's row-aligned
    log-Q, integrates it to ``(N_rows,)``, and discards it.  The scan body is
    ``jax.checkpoint``-wrapped so the reverse pass REMATERIALISES each member's
    density instead of stacking ``(M, N_rows, N_grid)`` residuals -- keeping BOTH
    the forward and backward peak at ``O(N_rows x N_grid) + O(M x N_rows)``, the
    crux that lets ``--lss_marginalize`` scale.  ``base_miss`` already carries the
    ``survey.z_depth`` relaxation (``== dN_exp`` beyond the depth) and ``Q_eff`` is
    forced to ``1`` there, so the integrand is bit-identical at grid nodes to the
    pre-existing ``_completion_member_row`` cube.
    """
    logq_members_row, member_is_log = _resolve_member_logq_row(em_catalog)
    depth_mask = None if survey.z_depth is None else (zgrid <= survey.z_depth)

    def _body(carry, logq_m):                 # logq_m: (N_rows, N_grid)
        q_eff = _member_q_eff_from_logq(logq_m, depth_mask, member_is_log)
        dN_m = base_miss * q_eff              # (N_rows, N_grid) -- transient, discarded
        return carry, jnp.trapezoid(dN_m, zgrid, axis=-1)  # (N_rows,)

    _, N = lax.scan(jax.checkpoint(_body), None, logq_members_row)  # (M, N_rows)
    return N


def _resolve_lss_completion_row_tables(em_catalog: EMCatalog):
    """Resolve the (optional) Q_LSS tables to **row order** for the vmap.

    Trace-safe consumer: this runs INSIDE the jitted likelihood (via
    :func:`completion_curves` <- ``prepare_redshift_prior_state`` <-
    ``darksiren_log_likelihood``), so every ``em_catalog`` field is a tracer and no
    concrete ``int()``/``.max()`` may touch one.  The indexing-enum decode, the
    global->compact row gather and the pixel-coverage validation are already done
    EAGERLY (host-side, before the jit) in the factory, which delivers a compact,
    row-aligned table.  Here we only consume it with static shapes.

    Returns ``(q_row, logq_members_row)``:

    * ``q_row``            — ``(N_rows, N_grid)`` DETERMINISTIC Q (already clipped
      + exponentiated), or ``None``.
    * ``logq_members_row`` — ``(M, N_rows, N_grid)`` RAW ensemble LOG-Q (row-major,
      NOT clipped/exponentiated -- deferred to the gathers / streamed chunk so no
      cube is built), or ``None``.

    Resolution: prefer ``logq`` over ``q`` for the deterministic table; clip to
    ``[-_LOGQ_CLIP, _LOGQ_CLIP]`` then exponentiate; validate ``N_grid ==
    zgrid.size``.  If ONLY an ensemble is supplied, ``q_row`` is the posterior-mean
    Q (streamed, no cube) so the scalar prior path still runs.  Q is **not**
    renormalised -- it is a physical density ratio.  Callers that need the member
    ``is_log`` flag call :func:`_resolve_member_logq_row` directly.
    """
    logq = em_catalog.lss_completion_logq
    q = em_catalog.lss_completion_q
    logq_members_row, member_is_log = _resolve_member_logq_row(em_catalog)
    if logq is None and q is None and logq_members_row is None:
        return None, None  # no Q table -> legacy local-overdensity path

    n_rows = em_catalog.zgals.shape[0]  # static int (shape) — safe under trace

    q_det = _to_q(logq, q)  # (K, N_grid) | None
    q_row = None
    if q_det is not None:
        _check_lss_grid(q_det)
        q_row = _row_align_lss(q_det, 0, em_catalog, n_rows)   # (N_rows, N_grid)
    elif logq_members_row is not None:
        # Only an ensemble supplied: the deterministic branch uses the
        # posterior-mean Q, streamed so the exponentiated member cube is not built.
        q_row = _member_posterior_mean_q(logq_members_row, member_is_log)

    return q_row, logq_members_row


def _row_C(row, grids: _CompletionGrids, em_catalog: EMCatalog):
    """Differential completeness ``C(z)`` for one catalog row (shared core).

    ``row`` is the compact catalog row index (== global HEALPix pixel for legacy
    full catalogs).  Returns ``(C, global_pix)``; depends on ``(row, Θ)`` only,
    never on a sample redshift.
    """
    global_pix = row if em_catalog.unique_pixels is None else em_catalog.unique_pixels[row]

    # --- observed density: O(1) cache lookup, or on-the-fly fallback ---
    if em_catalog.dN_obs_kde is not None:
        # The KDE cache is built row-for-row with THIS catalog's ``unique_pixels``
        # (every builder -- build_pixel_kde_cache, catalog_views, the CLI dry-run
        # -- aligns ``dN_obs_kde[k]`` with row ``k``), so the historical
        # ``pixel_to_cache_idx[global_pix]`` round-trip is exactly the identity
        # ``k -> k``.  Index the cache directly by row and skip the dense
        # ``pixel_to_cache_idx`` lookup entirely (12*nside^2 on the flat union
        # path).  Pinned by tests/test_cache_row_indexing.py.
        dN_obs = em_catalog.dN_obs_kde[row]
    else:
        dN_obs = _kde_dndz_obs(row, em_catalog.zgals, wgals=em_catalog.wgals, ngals=em_catalog.ngals)

    # --- differential completeness: matched-kernel ratio, no roll-off ---
    dN_exp_safe = jnp.where(grids.dN_exp_smooth > 0.0, grids.dN_exp_smooth, 1.0)
    C = jnp.clip(dN_obs / dN_exp_safe, 0.0, 1.0)
    return C, global_pix


def _assemble_curves(C, lss, grids: _CompletionGrids, survey: SurveyParams):
    """Assemble the per-row completion outputs from ``C`` and the rate factor ``lss``.

    ``survey.z_depth`` encodes prior knowledge that the EM survey does not
    catalog galaxies past its own depth: beyond ``z_depth`` completeness is
    exactly zero, so *every* modeled host there is uncatalogued (missing), and
    the LSS overdensity ``delta_g`` (measured from the survey and undefined
    outside its coverage) relaxes to the mean, ``lss -> 1``.  The missing
    density therefore relaxes to the full expected count,

        dN_miss(z) = (1 - C) dN_exp lss   for z <= z_depth,
                   = dN_exp               for z >  z_depth,

    i.e. the source prior above the depth in an empty pixel is the plain
    volumetric x population shape ``dN_exp`` -- NOT zero.  ``z_depth`` is prior
    knowledge about completeness, not an analysis cutoff: ``N_miss`` is
    integrated over the FULL grid, and the diagnostics follow (``C_eff`` is 0
    beyond the depth because ``dN_miss == dN_exp`` there, and ``f`` uses the
    full-grid ``N_exp``).

    ``z_depth`` is a concrete Python float (or ``None``) at trace time --
    never a sampled/traced value -- so the branch below is a Python-level
    ``if``, resolved once per trace, not a ``jnp.where``/``lax.cond``.

    ``z_depth is None`` takes the ORIGINAL expression untouched (no mask is
    even constructed), which is the legacy full-grid budget and is
    guaranteed bit-identical to the pre-existing behaviour.
    """
    dN_miss = (1.0 - C) * grids.dN_exp * lss

    if survey.z_depth is None:
        # Legacy full-grid missing budget -- EXACTLY the pre-existing expression.
        N_miss = jnp.trapezoid(dN_miss, zgrid)

        dN_exp_pos = jnp.where(grids.dN_exp > 0.0, grids.dN_exp, 1.0)
        C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
        N_exp = jnp.trapezoid(grids.dN_exp, zgrid)
        f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)
        return f, dN_miss, C_eff, N_miss

    # Beyond the depth C := 0 and lss := 1, so the missing density relaxes to
    # the full expected count dN_exp (hosts there are missing, not nonexistent).
    depth_mask = zgrid <= survey.z_depth
    dN_miss = jnp.where(depth_mask, dN_miss, grids.dN_exp)
    N_miss = jnp.trapezoid(dN_miss, zgrid)

    # Identical downstream to the None branch -- the only difference is the
    # relaxed dN_miss above.  C_eff is 0 wherever dN_miss == dN_exp (beyond the
    # depth), and f uses the full-grid N_exp.
    dN_exp_pos = jnp.where(grids.dN_exp > 0.0, grids.dN_exp, 1.0)
    C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
    N_exp = jnp.trapezoid(grids.dN_exp, zgrid)
    f = 1.0 - N_miss / jnp.where(N_exp > 0.0, N_exp, 1.0)
    return f, dN_miss, C_eff, N_miss


def _completion_curves_row(
    row: int,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Legacy completion curves for one row (local-overdensity LSS factor).

    Used whenever no ``Q_LSS`` table is supplied — physically identical to the
    pre-existing behaviour: ``lss = max(1 + alpha_miss*b_miss*delta_g(pix,z), 0)``.
    """
    C, global_pix = _row_C(row, grids, em_catalog)
    delta_g_pix_z = em_catalog.delta_g_pix_z
    b_eff = survey.alpha_miss * survey.b_miss
    if delta_g_pix_z.shape[0] == 1:        # static shape -> trace-time branch
        delta_g_z = delta_g_pix_z[0]
    else:
        delta_g_z = delta_g_pix_z[global_pix]
    lss = jnp.maximum(1.0 + b_eff * delta_g_z, 0.0)
    return _assemble_curves(C, lss, grids, survey)


def _completion_curves_row_q(
    row: int,
    q_row: jnp.ndarray,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Completion curves for one row using a precomputed Q_LSS factor.

    ``q_row`` (``(N_grid,)``, already row-aligned) **replaces** the legacy
    local-overdensity factor: ``dN_miss = (1 - C) dN_exp Q_LSS``.
    """
    C, _ = _row_C(row, grids, em_catalog)
    return _assemble_curves(C, q_row, grids, survey)


def _base_miss_row(
    row: int,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Member-INDEPENDENT missing-density base for one row: ``(1 - C) dN_exp``.

    Mirrors ``_assemble_curves``'s ``dN_miss`` with the LSS rate factor set to 1
    (Q_LSS is applied per member as ``base_miss * Q_eff_m``).  The
    ``survey.z_depth`` relaxation is baked in EXACTLY as in ``_assemble_curves``
    (Python-level branch on the concrete ``z_depth``; ``None`` builds no mask):
    beyond the depth C := 0 and lss := 1, so the base relaxes to the full expected
    count ``dN_exp`` -- and because ``Q_eff`` is forced to 1 there too, every
    member's density is ``dN_exp`` beyond the depth, matching the pre-existing
    per-member cube node-for-node.
    """
    C, _ = _row_C(row, grids, em_catalog)
    base = (1.0 - C) * grids.dN_exp
    if survey.z_depth is not None:
        depth_mask = zgrid <= survey.z_depth
        base = jnp.where(depth_mask, base, grids.dN_exp)
    return base


_completion_curves_rows_vmap = vmap(
    _completion_curves_row, in_axes=(0, None, None, None), out_axes=(0, 0, 0, 0)
)
_completion_curves_rows_q_vmap = vmap(
    _completion_curves_row_q, in_axes=(0, 0, None, None, None), out_axes=(0, 0, 0, 0)
)
_base_miss_rows_vmap = vmap(
    _base_miss_row, in_axes=(0, None, None, None), out_axes=0
)


def completion_curves(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> CompletionCurves:
    """All per-row completion curves for one parameter proposal.

    Eager (host-side); called once per proposal by
    :func:`darksirens.redshift.prior.prepare_redshift_prior_state`.  When the catalog
    carries an LSS-conditioned lognormal completion table the per-row factor is
    ``Q_LSS`` (replacing the legacy ``max(1 + b_eff delta_g, 0)``); with an
    ensemble it additionally returns the member-INDEPENDENT ``base_miss`` curve and
    the compact per-member ``N_miss_members`` scalar table (both cube-free) so the
    member marginalisation / diagnostic can reconstruct each member's density on
    the fly (``base_miss * Q_eff_m``) without ever storing ``(M, N_rows, N_grid)``.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    n_rows = em_catalog.zgals.shape[0]
    rows = jnp.arange(n_rows, dtype=jnp.int32)

    q_row, logq_members_row = _resolve_lss_completion_row_tables(em_catalog)

    if q_row is None:
        f, dN_miss, C_eff, N_miss = _completion_curves_rows_vmap(rows, grids, survey, em_catalog)
    else:
        f, dN_miss, C_eff, N_miss = _completion_curves_rows_q_vmap(
            rows, q_row, grids, survey, em_catalog
        )

    base_miss = None
    N_miss_members = None
    if logq_members_row is not None:
        base_miss = _base_miss_rows_vmap(rows, grids, survey, em_catalog)   # (N_rows, N_grid)
        N_miss_members = member_N_miss_integrals(base_miss, em_catalog, survey)  # (M, N_rows)

    return CompletionCurves(
        f=f, dN_miss=dN_miss, C_eff=C_eff, N_miss=N_miss,
        base_miss=base_miss, N_miss_members=N_miss_members,
    )


# ------------------------------------------------------------
# FIELD-convention global normalizer Z(theta)
# ------------------------------------------------------------

def field_global_log_Z(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    chunk_size: int = 4096,
) -> jnp.ndarray:
    """log of the survey-GLOBAL FIELD normalizer for one parameter proposal.

    ``Z(theta) = Sum_all-pixels [ N_obs,pix + N_miss,pix(theta) ]`` with, per
    pixel, ``N_miss = integral (1 - C) dN_exp lss_p`` -- the SAME missing-galaxy
    budget modulation as the per-pixel numerator:

        lss_p = 1                                (legacy; no modulation inputs)
        lss_p = Q_p(z)                           (``field_lss_q`` rows;
                                                  empty-pixel budget is the data
                                                  constant ``field_lss_q_empty_sum``)
        lss_p = max(1 + b_eff*delta_g_p(z), 0)   (``field_delta_g`` rows;
                                                  empty pixels carry delta_g == 0
                                                  by construction, so lss = 1)

    The reduction accumulates the (N_grid,) field missing CURVE
    ``V(z; theta) = Sum_occ (1 - C_p)(z) lss_p(z) + V_empty(z)`` and integrates
    ``dN_exp * V`` once -- the curve form is what lets a z-dependent factor
    (Q, delta_g, and the marked-host ``mu_miss``) modulate the budget before
    the quadrature.  ``C`` reuses ``_row_C``'s exact recipe --
    ``clip(dN_obs_s / where(dN_exp_smooth > 0, dN_exp_smooth, 1), 0, 1)`` -- and
    ``dN_exp = survey.n0 * apix * g(z)`` is the identical grid used per pixel, so
    the reduction is consistent with ``_assemble_curves`` term by term.  The
    ``survey.z_depth`` completeness prior is applied EXACTLY as in
    ``_assemble_curves`` (Python-level branch on the concrete ``z_depth``;
    ``None`` takes the untouched full-grid expression): beyond the depth every
    pixel has C == 0 and lss == 1, so ``V`` relaxes to the total pixel count and
    the survey-global missing curve there is the full ``dN_exp`` per pixel.

    Fully differentiable in ``theta`` (n0, cosmology, delta, and b_miss through
    the delta_g mode): the frozen inputs are ``field_dN_obs_s`` and the
    modulation rows (data constants).  The occupied-pixel reduction is chunked
    with ``lax.scan`` to bound peak memory at high nside.
    """
    V_total, dN_exp = _field_missing_curve(
        cosmo, survey, em_catalog, chunk_size=chunk_size
    )
    N_obs_total = jnp.asarray(em_catalog.field_N_obs_total, dtype=dN_exp.dtype)

    N_miss_total = jnp.trapezoid(dN_exp * V_total, zgrid)

    Z = N_obs_total + N_miss_total
    return jnp.log(jnp.maximum(Z, 1e-300))


def _field_missing_curve(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    chunk_size: int = 4096,
):
    """The (N_grid,) field missing curve ``V(z; theta)`` plus its companions.

    ``V = Sum_occ (1 - C_p)(z) lss_p(z) + V_empty(z)`` -- everything of the
    survey-global missing budget EXCEPT the ``dN_exp`` quadrature, so callers
    can insert a z-dependent factor (the marked-host ``mu_miss``) before
    integrating.  Beyond ``survey.z_depth`` the curve is relaxed to the total
    pixel count (every pixel has C == 0 and lss == 1 there), so the beyond-depth
    completeness prior is already baked in and callers integrate ``dN_exp * V``
    over the FULL grid.  Returns ``(V_total, dN_exp)``.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    dN_exp = grids.dN_exp                                    # (N_grid,) theta-dependent
    dN_exp_safe = jnp.where(grids.dN_exp_smooth > 0.0, grids.dN_exp_smooth, 1.0)

    field_obs = jnp.asarray(em_catalog.field_dN_obs_s)       # (n_occ, N_grid) f32 constant
    n_occ = int(field_obs.shape[0])                          # static
    n_empty = jnp.asarray(em_catalog.field_n_empty, dtype=dN_exp.dtype)

    # Static (pytree-structure) modulation dispatch, mirroring the numerator's
    # rule: a Q table REPLACES the local-overdensity factor.
    has_q = em_catalog.field_lss_q is not None
    has_dg = em_catalog.field_delta_g is not None
    if has_q and has_dg:
        raise NotImplementedError(
            "field_global_log_Z: field_lss_q and field_delta_g are mutually "
            "exclusive (Q_LSS replaces the local-overdensity factor, matching "
            "the per-pixel numerator)."
        )
    if has_q and em_catalog.field_lss_q_empty_sum is None:
        raise ValueError(
            "field_global_log_Z: field_lss_q requires field_lss_q_empty_sum "
            "(the empty-pixel Q budget); build both via "
            "build_field_lss_q_inputs."
        )
    if has_q:
        mod_rows = jnp.asarray(em_catalog.field_lss_q)       # (n_occ, N_grid) f32
    elif has_dg:
        mod_rows = jnp.asarray(em_catalog.field_delta_g)     # (n_occ, N_grid) f32
    else:
        mod_rows = jnp.zeros((n_occ, 1), dtype=jnp.float32)  # inert placeholder
    b_eff = survey.alpha_miss * survey.b_miss                # traced (delta_g mode)

    def _row_V(obs_row, mod_row):
        obs_row = obs_row.astype(dN_exp.dtype)
        C = jnp.clip(obs_row / dN_exp_safe, 0.0, 1.0)
        if has_q:
            lss = mod_row.astype(dN_exp.dtype)
        elif has_dg:
            lss = jnp.maximum(1.0 + b_eff * mod_row.astype(dN_exp.dtype), 0.0)
        else:
            lss = 1.0
        return (1.0 - C) * lss                               # (N_grid,)

    # Chunked scan over occupied rows: pad to a whole number of ``chunk_size``
    # blocks and mask the padding so padded (all-zero) rows -- which would
    # otherwise read as empty pixels with C == 0 -- contribute nothing.
    pad = (-n_occ) % chunk_size
    n_pad = n_occ + pad
    obs_pad = jnp.pad(field_obs, ((0, pad), (0, 0)))
    mod_pad = jnp.pad(mod_rows, ((0, pad), (0, 0)))
    valid = jnp.arange(n_pad) < n_occ
    n_chunks = n_pad // chunk_size if chunk_size > 0 else 0
    obs_chunks = obs_pad.reshape(n_chunks, chunk_size, zgrid.size)
    mod_chunks = mod_pad.reshape(n_chunks, chunk_size, mod_rows.shape[1])
    valid_chunks = valid.reshape(n_chunks, chunk_size)

    def _body(acc, xs):
        obs_c, mod_c, val_c = xs
        Vc = vmap(_row_V)(obs_c, mod_c)                      # (chunk, N_grid)
        Vc = jnp.where(val_c[:, None], Vc, 0.0)
        return acc + jnp.sum(Vc, axis=0), None

    V_occ, _ = lax.scan(
        _body,
        jnp.zeros(zgrid.size, dtype=dN_exp.dtype),
        (obs_chunks, mod_chunks, valid_chunks),
    )

    # Empty pixels: C == 0, so their budget curve is lss_p itself -- n_empty
    # for the legacy/delta_g modes (delta_g == 0 on empty pixels), the data
    # constant Sum_empty Q_p(z) for the Q mode.
    if has_q:
        V_empty = jnp.asarray(em_catalog.field_lss_q_empty_sum, dtype=dN_exp.dtype)
    else:
        V_empty = n_empty

    V_total = V_occ + V_empty
    # ``z_depth`` is concrete at trace time -> Python-level branch (mirrors
    # ``_assemble_curves``); ``None`` never constructs a mask.  Beyond the depth
    # every pixel (occupied and empty) has C == 0 and lss == 1, so the global
    # missing curve relaxes to the total pixel count: the full expected
    # population is uncatalogued there, not nonexistent.
    if survey.z_depth is not None:
        depth_mask = zgrid <= survey.z_depth
        n_pix_total = n_occ + n_empty
        V_total = jnp.where(depth_mask, V_total, n_pix_total)
    return V_total, dN_exp


def field_global_log_Z_members(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
) -> jnp.ndarray:
    """Per-member survey-GLOBAL normalizers ``log Z_m(theta)``, shape (M,).

    One :func:`field_global_log_Z` evaluation per Q-ensemble member, sharing
    everything except the member's Q rows / empty-pixel budget
    (``field_lss_q_members`` / ``field_lss_q_empty_sum_members``).  Consumed by
    the lss_marginalize member vmap in the likelihood core (each member state
    carries its own global normalizer under the field convention).
    """
    q_members = em_catalog.field_lss_q_members
    q_empty_members = em_catalog.field_lss_q_empty_sum_members
    if q_members is None or q_empty_members is None:
        raise ValueError(
            "field_global_log_Z_members requires field_lss_q_members and "
            "field_lss_q_empty_sum_members; build them via "
            "build_field_lss_q_member_inputs."
        )

    def _one(q_m, q_empty_m):
        cat_m = em_catalog._replace(
            field_lss_q=q_m, field_lss_q_empty_sum=q_empty_m
        )
        return field_global_log_Z(cosmo, survey, cat_m)

    return vmap(_one)(jnp.asarray(q_members), jnp.asarray(q_empty_members))


def field_global_log_Z_marked(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    mu_miss: jnp.ndarray,
    log_h_flat: jnp.ndarray,
) -> jnp.ndarray:
    """Marked-host survey-GLOBAL normalizer.

    ``Z(theta, eta) = Sum_{i in obs, full sky} w_i h(m_i | eta)
                      + integral mu_miss(z; eta) dN_exp(z) V(z; theta) dz``

    -- the marked analogue of :func:`field_global_log_Z`: the observed term is
    the full-sky marked mass (the marked kernel state rescales each pixel's
    catalog mass from N_obs to Sum w_i h_i, so the global total must follow),
    and the missing budget carries the SAME ``mu_miss(z | eta)`` factor as the
    numerator's ``dN_miss``.  ``mu_miss`` and ``log_h_flat`` must be built from
    the FULL-SKY flat marks (``field_mark_*``), so the PE and selection states
    produce the SAME Z for the same (theta, eta) and the constants cancel
    structurally between the two likelihood seams.
    """
    V_total, dN_exp = _field_missing_curve(cosmo, survey, em_catalog)

    w_flat = jnp.asarray(em_catalog.field_mark_w, dtype=dN_exp.dtype)
    h_flat = jnp.exp(jnp.asarray(log_h_flat, dtype=dN_exp.dtype))
    S_obs = jnp.sum(w_flat * h_flat)

    # ``V_total`` already carries the beyond-depth relaxation (C == 0, lss == 1),
    # where ``mu_miss`` defaults to the homogeneous 1 (no observed galaxies to
    # inform the host efficiency), so the integrand is taken over the FULL grid.
    integrand = jnp.asarray(mu_miss, dtype=dN_exp.dtype) * dN_exp * V_total
    N_miss_total = jnp.trapezoid(integrand, zgrid)

    Z = S_obs + N_miss_total
    return jnp.log(jnp.maximum(Z, 1e-300))


def field_global_log_Z_marked_members(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    mu_miss: jnp.ndarray,
    log_h_flat: jnp.ndarray,
) -> jnp.ndarray:
    """Per-member MARKED survey-GLOBAL normalizers ``log Z_m(theta, eta)``, (M,).

    The marked analogue of :func:`field_global_log_Z_members`: one
    :func:`field_global_log_Z_marked` evaluation per Q-ensemble member, swapping
    only the member's Q rows / empty-pixel budget (``field_lss_q_members`` /
    ``field_lss_q_empty_sum_members``) into the missing curve.  The observed
    marked mass ``Sum w_i h_i`` and the ``mu_miss(z | eta)`` factor are
    member-INDEPENDENT (full-sky flat marks), so they are shared verbatim across
    members -- reusing ``field_global_log_Z_marked`` guarantees the observed term
    and mu_miss integrand are op-for-op identical to the scalar marked path.
    Consumed by the lss_marginalize member vmap in the likelihood core (each
    member state carries its own marked global normalizer under field weighting).
    """
    q_members = em_catalog.field_lss_q_members
    q_empty_members = em_catalog.field_lss_q_empty_sum_members
    if q_members is None or q_empty_members is None:
        raise ValueError(
            "field_global_log_Z_marked_members requires field_lss_q_members and "
            "field_lss_q_empty_sum_members; build them via "
            "build_field_lss_q_member_inputs."
        )

    def _one(q_m, q_empty_m):
        cat_m = em_catalog._replace(
            field_lss_q=q_m, field_lss_q_empty_sum=q_empty_m
        )
        return field_global_log_Z_marked(cosmo, survey, cat_m, mu_miss, log_h_flat)

    return vmap(_one)(jnp.asarray(q_members), jnp.asarray(q_empty_members))


def build_field_mark_inputs(
    full_z: jnp.ndarray,
    full_w: jnp.ndarray | None,
    full_n: jnp.ndarray | None,
    mark_tables: dict,
    mark_names: tuple,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flat FULL-SKY per-galaxy inputs for the MARKED field normalizer.

    ``mark_tables`` maps mark name -> (N_pix, N_max) z-centred mark array (the
    same ``attach_mark_inputs`` products the numerator uses); real (non-padded)
    slots are selected by ``full_n`` (fallback ``full_w > 0``).  Returns
    ``(field_mark_z, field_mark_w, field_mark_values)`` — (N_gal,), (N_gal,),
    and (N_gal, n_marks) float32 with columns ordered by ``mark_names``.
    """
    z_np = np.asarray(full_z)
    if full_n is not None:
        n_np = np.asarray(full_n).reshape(-1).astype(np.int64)
        real = np.arange(z_np.shape[1])[None, :] < n_np[:, None]
    elif full_w is not None:
        real = np.asarray(full_w) > 0.0
    else:
        raise ValueError(
            "build_field_mark_inputs requires full_n or full_w to mask padded "
            "galaxy slots."
        )
    z_flat = jnp.asarray(z_np[real], dtype=jnp.float32)
    w_flat = jnp.asarray(
        (np.asarray(full_w)[real] if full_w is not None
         else np.ones(int(real.sum()))),
        dtype=jnp.float32,
    )
    missing = [name for name in mark_names if mark_tables.get(name) is None]
    if missing:
        raise ValueError(
            f"build_field_mark_inputs: marks {missing} not present in "
            "mark_tables; the field normalizer needs every selected mark over "
            "the full sky."
        )
    values = np.stack(
        [np.asarray(mark_tables[name])[real] for name in mark_names], axis=1
    )
    return z_flat, w_flat, jnp.asarray(values, dtype=jnp.float32)


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

def _completion_clip_fractions_for_pixel(
    pix: int,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    q_row=None,
) -> dict[str, float]:
    """Clipping fractions on ``zgrid`` for one catalog row.

    ``C_iso_clipped_fraction`` is the raw-ratio clip and
    ``C_eff_clipped_fraction`` the effective-completeness clip (both always
    valid).  ``rho_miss_eff_clipped_fraction`` depends on the missing-galaxy
    rate factor: for the legacy local-overdensity model it is the fraction where
    ``1 + b_eff*delta_g < 0``; for an LSS-conditioned ``Q_LSS`` table (``q_row``
    supplied, the row-aligned ``(N_grid,)`` factor) ``Q >= 0`` always, so it is
    instead the fraction where ``logQ`` hit the ``±_LOGQ_CLIP`` bound.
    ``lss_source`` records which factor was used.
    """
    global_pix = pix if em_catalog.unique_pixels is None else em_catalog.unique_pixels[pix]

    if em_catalog.dN_obs_kde is not None:
        # Cache row == catalog row by construction (see _row_C); index directly.
        dN_obs = em_catalog.dN_obs_kde[pix]
    else:
        dN_obs = _kde_dndz_obs(
            pix, em_catalog.zgals, wgals=em_catalog.wgals, ngals=em_catalog.ngals
        )

    dN_exp_safe = jnp.where(grids.dN_exp_smooth > 0.0, grids.dN_exp_smooth, 1.0)
    C_raw = dN_obs / dN_exp_safe
    C_clipped_mask = (C_raw < 0.0) | (C_raw > 1.0)
    C = jnp.clip(C_raw, 0.0, 1.0)

    if q_row is not None:
        lss = q_row[pix]                       # row-aligned Q_LSS factor (>= 0)
        lss_source = "Q_LSS"
        q_hi = float(jnp.exp(_LOGQ_CLIP)) * (1.0 - 1e-6)
        q_lo = float(jnp.exp(-_LOGQ_CLIP)) * (1.0 + 1e-6)
        lss_clipped_mask = (lss >= q_hi) | (lss <= q_lo)
    else:
        b_eff = survey.alpha_miss * survey.b_miss
        if em_catalog.delta_g_pix_z.shape[0] == 1:
            delta_g_z = em_catalog.delta_g_pix_z[0]
        else:
            delta_g_z = em_catalog.delta_g_pix_z[global_pix]
        lss_raw = 1.0 + b_eff * delta_g_z
        lss = jnp.maximum(lss_raw, 0.0)
        lss_source = "legacy_delta_g"
        lss_clipped_mask = lss_raw < 0.0

    dN_miss = (1.0 - C) * grids.dN_exp * lss
    dN_exp_pos = jnp.where(grids.dN_exp > 0.0, grids.dN_exp, 1.0)
    C_eff_raw = 1.0 - dN_miss / dN_exp_pos
    C_eff_clipped_mask = (C_eff_raw < 0.0) | (C_eff_raw > 1.0)

    return {
        "C_iso_clipped_fraction": float(jnp.mean(C_clipped_mask)),
        "C_eff_clipped_fraction": float(jnp.mean(C_eff_clipped_mask)),
        "rho_miss_eff_clipped_fraction": float(jnp.mean(lss_clipped_mask)),
        "lss_source": lss_source,
    }


def completion_clip_diagnostics(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    pixels: np.ndarray | None = None,
    max_pixels: int = 64,
) -> dict[str, object]:
    """Summarise completion clipping over a representative pixel set."""
    grids = _precompute_grids(cosmo, survey, em_catalog)
    # Q_LSS-aware: resolve the (optional) completion table once and report the
    # Q-clip instead of the (inapplicable) delta_g-negativity when it is present.
    q_row, _ = _resolve_lss_completion_row_tables(em_catalog)

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
            int(pix), grids, survey, em_catalog, q_row=q_row
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
        "lss_source": "Q_LSS" if q_row is not None else "legacy_delta_g",
        "per_pixel": per_pixel,
    }
    for field in fields:
        vals = np.array([item[field] for item in per_pixel], dtype=float)
        summary[f"mean_{field}"] = float(vals.mean()) if vals.size else 0.0
        summary[f"max_{field}"] = float(vals.max()) if vals.size else 0.0

    return summary


# ------------------------------------------------------------
# Public scalar / vmapped API (checks & diagnostics; slow path)
# ------------------------------------------------------------

def _curves_scalar(z, pix, curves: CompletionCurves):
    """Interpolate ``(f, p_miss(z), C_eff(z))`` from precomputed per-row curves."""
    nm = jnp.where(curves.N_miss[pix] > 0.0, curves.N_miss[pix], 1.0)
    p_miss = jnp.interp(z, zgrid, curves.dN_miss[pix] / nm)
    C_eff = jnp.interp(z, zgrid, curves.C_eff[pix])
    return curves.f[pix], p_miss, C_eff


def catalog_completion(
    z: float,
    pix: int,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """
    Characterise catalog incompleteness at a single (z, pix) point.

    **Q_LSS-aware**: delegates to :func:`completion_curves` (the single source of
    truth that applies any LSS-conditioned completion table), then interpolates
    the per-row curve at ``z``.  Eager / diagnostic slow path (NOT jitted — the
    hot likelihood uses ``completion_curves`` via ``darksirens.redshift.prior``).

    Returns
    -------
    f : float — scalar number-weighted completeness fraction (diagnostic).
    p_miss : float — normalised missing-galaxy PDF at z.
    C : float — effective completeness C_eff(z|pix).
    """
    curves = completion_curves(cosmo, survey, em_catalog)
    return _curves_scalar(z, jnp.asarray(pix), curves)


def catalog_completion_vmap(
    z: jnp.ndarray,
    pix: jnp.ndarray,
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Vectorised ``catalog_completion`` over arrays of (z, pix) pairs.

    Q_LSS-aware (delegates to ``completion_curves``); eager / diagnostic slow
    path.  Hot paths use ``completion_curves`` via the state API in
    ``darksirens.redshift.prior``.
    """
    curves = completion_curves(cosmo, survey, em_catalog)
    z = jnp.asarray(z)
    pix = jnp.asarray(pix)
    return vmap(lambda z_i, p_i: _curves_scalar(z_i, p_i, curves))(z, pix)


# ------------------------------------------------------------
# LSS overdensity field
# ------------------------------------------------------------

def compute_lss_overdensity(
    zgals: jnp.ndarray,
    nside: int,
    wgals: jnp.ndarray | None = None,
    ngals: jnp.ndarray | None = None,
    batch_size: int = 4096,
) -> jnp.ndarray:
    """
    Pre-compute the LSS overdensity field delta_g(pix, z) on ``zgrid``.

    Call once at startup; store the result in ``EMCatalog.delta_g_pix_z``.

    Construction
    ------------
    The mean density at each z is taken over *occupied* pixels only
    (pixels with at least one real galaxy), so survey footprints do not
    dilute the mean.  Pixels with no galaxies carry delta_g = 0: the
    observed catalog cannot distinguish a void from an unobserved region,
    and zero keeps the missing-galaxy density isotropic there (the
    completeness ratio already handles the incompleteness itself).  For
    the same reason delta_g -> 0 wherever the mean density vanishes
    (beyond the catalog depth).  No per-pixel mean subtraction is applied
    downstream; this construction is already centred.

    Caveat: delta_g is estimated from the *observed* galaxies, so in
    partially incomplete pixels it mixes true structure with the local
    selection.  ``b_miss`` absorbs the amplitude but not the z-dependence
    of that mixing.

    Computation is chunked (``batch_size`` pixels per JIT call) to bound
    peak memory at high nside.
    """
    import healpy as hp
    n_pix = hp.nside2npix(nside)

    if ngals is None and wgals is not None and jnp.asarray(wgals).ndim == 1:
        ngals = wgals
        wgals = None
    if wgals is None and ngals is None:
        raise ValueError(
            "compute_lss_overdensity requires either wgals or ngals to mask padded galaxies"
        )

    zgals_jax = jnp.asarray(zgals)
    wgals_jax = None if wgals is None else jnp.asarray(wgals)
    ngals_jax = None if ngals is None else jnp.asarray(ngals)

    _batch = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
    kde = np.empty((n_pix, zgrid.size), dtype=np.float64)
    for start in range(0, n_pix, batch_size):
        stop = min(start + batch_size, n_pix)
        kde[start:stop] = np.asarray(
            _batch(
                jnp.arange(start, stop, dtype=jnp.int32),
                zgals_jax,
                wgals_jax,
                ngals_jax,
            )
        )

    if ngals_jax is not None:
        occupied = np.asarray(ngals_jax) > 0
    else:
        occupied = np.asarray(jnp.any(wgals_jax > 0, axis=-1))
    occupied = occupied.astype(bool)[:n_pix]

    n_occ = max(int(occupied.sum()), 1)
    mean_density = kde[occupied].sum(axis=0) / n_occ if occupied.any() else np.zeros(zgrid.size)
    mean_safe = np.where(mean_density > 0.0, mean_density, 1.0)

    delta_g = (kde - mean_density[None, :]) / mean_safe[None, :]
    delta_g[~occupied] = 0.0
    delta_g[:, mean_density <= 0.0] = 0.0
    return jnp.asarray(delta_g)
