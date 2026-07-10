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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Precompute ``_kde_dndz_obs`` for all unique pixels.  Call once in
    ``make_likelihood`` before the JIT closure is built.

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

    wgals_jax = None if wgals is None else jnp.asarray(wgals)
    ngals_jax = None if ngals is None else jnp.asarray(ngals)

    _batch_kde = jit(vmap(_kde_dndz_obs, in_axes=(0, None, None, None)))
    dN_obs_kde = _batch_kde(
        jnp.asarray(unique_pixels, dtype=jnp.int32),
        jnp.asarray(zgals),
        wgals_jax,
        ngals_jax,
    )

    pixel_to_cache_idx = np.zeros(n_pix_catalog, dtype=np.int32)
    for i, p in enumerate(unique_pixels):
        pixel_to_cache_idx[int(p)] = i

    return dN_obs_kde, jnp.asarray(pixel_to_cache_idx, dtype=jnp.int32)


# ------------------------------------------------------------
# FIELD-convention sky-weighting: survey-global normalization inputs
# ------------------------------------------------------------

def build_field_normalization_inputs(
    full_z: jnp.ndarray,
    full_w: jnp.ndarray | None,
    full_n: jnp.ndarray | None,
    batch_size: int = 4096,
) -> tuple[jnp.ndarray, int, float]:
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
    field_dN_obs_s : (n_occupied, N_grid) float32
        Smoothed observed density for occupied pixels only (device array).
    n_empty : int
        Number of galaxy-free survey pixels (``N_pix - n_occupied``).
    N_obs_total : float
        Total observed real-galaxy count over ALL pixels.
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
    return field_dN_obs_s, n_empty, N_obs_total


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

    ``dN_miss_members`` / ``N_miss_members`` are populated only when the
    catalog carries an LSS-completion *ensemble* (``lss_completion_*_members``);
    they support the Bayesian redshift-prior diagnostic and default to ``None``
    (the scalar API and the GW likelihood ignore them).
    """
    f: jnp.ndarray        # (N_rows,) scalar number-weighted completeness (diagnostic)
    dN_miss: jnp.ndarray  # (N_rows, N_grid) missing-galaxy density [counts / unit z]
    C_eff: jnp.ndarray    # (N_rows, N_grid) effective completeness (diagnostic)
    N_miss: jnp.ndarray   # (N_rows,) integral of dN_miss
    dN_miss_members: jnp.ndarray = None  # (M, N_rows, N_grid) | None
    N_miss_members: jnp.ndarray = None   # (M, N_rows) | None


# ------------------------------------------------------------
# LSS-conditioned lognormal completion factor Q_LSS (optional)
# ------------------------------------------------------------

#: log Q is clipped to this symmetric range before exponentiating, so that a
#: heavy lognormal tail cannot blow up the missing-galaxy density.
_LOGQ_CLIP: float = 7.0


def _resolve_lss_completion_row_tables(em_catalog: EMCatalog):
    """Resolve the (optional) Q_LSS tables to **row order** for the vmap.

    Trace-safe consumer: this runs INSIDE the jitted likelihood (via
    :func:`completion_curves` <- ``prepare_redshift_prior_state`` <-
    ``darksiren_log_likelihood``), so every ``em_catalog`` field is a tracer and no
    concrete ``int()``/``.max()`` may touch one.  The indexing-enum decode, the
    global->compact row gather and the pixel-coverage validation are already done
    EAGERLY (host-side, before the jit) in
    ``inference/likelihood.py::make_likelihood._compact_lss_q``, which delivers a
    compact, row-aligned table.  Here we only consume it with static shapes.

    Returns ``(q_row, q_members_row)``:

    * ``q_row``           — ``(N_rows, N_grid)`` deterministic Q, or ``None``.
    * ``q_members_row``   — ``(N_rows, M, N_grid)`` ensemble Q (row-major
      leading axis, ready to vmap), or ``None``.

    Resolution rules: prefer ``logq`` over ``q``; clip ``logq`` to
    ``[-_LOGQ_CLIP, _LOGQ_CLIP]`` then exponentiate; validate ``N_grid ==
    zgrid.size`` (no silent interpolation; a static-shape check).  A table whose
    row axis already equals ``N_rows`` is used as-is (the normal jit path); a
    non-compacted global table (only reachable on the eager *diagnostic* path,
    never under jit) is gathered to row order via ``unique_pixels`` after an eager
    pixel-coverage check.  If only an ensemble is supplied, the deterministic
    ``q_row`` is the posterior-mean
    Q so the scalar prior path still runs.  Q is **not** renormalised here — it is
    a physical density ratio.
    """
    logq = em_catalog.lss_completion_logq
    q = em_catalog.lss_completion_q
    logq_m = em_catalog.lss_completion_logq_members
    q_m = em_catalog.lss_completion_q_members
    if logq is None and q is None and logq_m is None and q_m is None:
        return None, None  # no Q table -> legacy local-overdensity path

    n_rows = em_catalog.zgals.shape[0]  # static int (shape) — safe under trace

    def _to_q(logq_arr, q_arr):
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

    def _check_grid(arr):
        if arr.shape[-1] != int(zgrid.size):
            raise ValueError(
                f"LSS completion table has N_grid={arr.shape[-1]} but the package "
                f"zgrid has size {int(zgrid.size)}; the offline builder must use the "
                "same redshift grid (no silent interpolation)."
            )

    def _row_align(table, row_axis):
        # Compact-vs-global is decided by STATIC shapes only (``K`` and ``n_rows``
        # are concrete even under a trace, so nothing here concretises a tracer).
        # In the jitted likelihood the table is ALWAYS already compact --
        # make_likelihood._compact_lss_q slices global->compact before the jit -- so
        # ``K == n_rows`` and we return it untouched (the hot path).
        K = table.shape[row_axis]
        if K == n_rows:
            return table
        # ``K != n_rows`` is reachable ONLY on the eager path (a full global table
        # handed straight to completion_curves by a test or diagnostic), never under
        # jit, so ``unique_pixels`` is concrete here and the coverage check is safe.
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

    q_det = _to_q(logq, q)        # (K, N_grid) | None
    q_mem = _to_q(logq_m, q_m)    # (M, K, N_grid) | None

    q_row = None
    if q_det is not None:
        _check_grid(q_det)
        q_row = _row_align(q_det, 0)                       # (N_rows, N_grid)

    q_members_row = None
    if q_mem is not None:
        _check_grid(q_mem)
        q_mem_rows = _row_align(q_mem, 1)                  # (M, N_rows, N_grid)
        q_members_row = jnp.transpose(q_mem_rows, (1, 0, 2))  # (N_rows, M, N_grid)

    if q_row is None and q_members_row is not None:
        # Only an ensemble supplied: deterministic branch uses posterior-mean Q.
        q_row = jnp.mean(q_members_row, axis=1)            # (N_rows, N_grid)

    return q_row, q_members_row


def _row_C(row, grids: _CompletionGrids, em_catalog: EMCatalog):
    """Differential completeness ``C(z)`` for one catalog row (shared core).

    ``row`` is the compact catalog row index (== global HEALPix pixel for legacy
    full catalogs).  Returns ``(C, global_pix)``; depends on ``(row, Θ)`` only,
    never on a sample redshift.
    """
    global_pix = row if em_catalog.unique_pixels is None else em_catalog.unique_pixels[row]

    # --- observed density: O(1) cache lookup, or on-the-fly fallback ---
    if em_catalog.dN_obs_kde is not None:
        cache_idx = em_catalog.pixel_to_cache_idx[global_pix]
        dN_obs = em_catalog.dN_obs_kde[cache_idx]
    else:
        dN_obs = _kde_dndz_obs(row, em_catalog.zgals, wgals=em_catalog.wgals, ngals=em_catalog.ngals)

    # --- differential completeness: matched-kernel ratio, no roll-off ---
    dN_exp_safe = jnp.where(grids.dN_exp_smooth > 0.0, grids.dN_exp_smooth, 1.0)
    C = jnp.clip(dN_obs / dN_exp_safe, 0.0, 1.0)
    return C, global_pix


def _assemble_curves(C, lss, grids: _CompletionGrids, survey: SurveyParams):
    """Assemble the per-row completion outputs from ``C`` and the rate factor ``lss``.

    ``survey.z_depth`` optionally bounds the missing-galaxy budget to
    ``zgrid <= z_depth``: a survey is never designed to detect galaxies past
    its own depth, so counting them as "missing" over the FULL grid up to
    ``DARKSIRENS_ZMAX`` inflates ``N_miss`` and dilutes the catalog term.
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

    # Bounded budget: zero the missing-galaxy density beyond the survey depth
    # BEFORE the N_miss quadrature, and bound the N_exp used by the diagnostic
    # f the same way so it stays consistent with the truncated N_miss (C_eff
    # is unaffected -- it is already 1.0 wherever dN_miss is zeroed).
    depth_mask = zgrid <= survey.z_depth
    dN_miss = jnp.where(depth_mask, dN_miss, 0.0)
    N_miss = jnp.trapezoid(dN_miss, zgrid)

    dN_exp_pos = jnp.where(grids.dN_exp > 0.0, grids.dN_exp, 1.0)
    C_eff = jnp.clip(1.0 - dN_miss / dN_exp_pos, 0.0, 1.0)
    dN_exp_bounded = jnp.where(depth_mask, grids.dN_exp, 0.0)
    N_exp = jnp.trapezoid(dN_exp_bounded, zgrid)
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


def _completion_member_row(
    row: int,
    q_members_row: jnp.ndarray,
    grids: _CompletionGrids,
    survey: SurveyParams,
    em_catalog: EMCatalog,
):
    """Per-row ensemble missing densities for ``q_members_row`` ``(M, N_grid)``.

    Mirrors the ``survey.z_depth`` missing-galaxy-budget bound applied in
    ``_assemble_curves`` (Python-level branch; ``z_depth`` is concrete at
    trace time, never a tracer) so the ensemble diagnostics stay consistent
    with the deterministic ``dN_miss``/``N_miss`` curves.  ``z_depth is None``
    takes the untouched original expression -- no mask is built.
    """
    C, _ = _row_C(row, grids, em_catalog)
    dN_miss_m = (1.0 - C)[None, :] * grids.dN_exp[None, :] * q_members_row  # (M, N_grid)
    if survey.z_depth is not None:
        depth_mask = zgrid <= survey.z_depth
        dN_miss_m = jnp.where(depth_mask[None, :], dN_miss_m, 0.0)
    N_miss_m = jnp.trapezoid(dN_miss_m, zgrid, axis=-1)                     # (M,)
    return dN_miss_m, N_miss_m


_completion_curves_rows_vmap = vmap(
    _completion_curves_row, in_axes=(0, None, None, None), out_axes=(0, 0, 0, 0)
)
_completion_curves_rows_q_vmap = vmap(
    _completion_curves_row_q, in_axes=(0, 0, None, None, None), out_axes=(0, 0, 0, 0)
)
_completion_member_rows_vmap = vmap(
    _completion_member_row, in_axes=(0, 0, None, None, None), out_axes=(0, 0)
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
    ensemble it additionally returns member missing densities for diagnostics.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    n_rows = em_catalog.zgals.shape[0]
    rows = jnp.arange(n_rows, dtype=jnp.int32)

    q_row, q_members_row = _resolve_lss_completion_row_tables(em_catalog)

    if q_row is None:
        f, dN_miss, C_eff, N_miss = _completion_curves_rows_vmap(rows, grids, survey, em_catalog)
    else:
        f, dN_miss, C_eff, N_miss = _completion_curves_rows_q_vmap(
            rows, q_row, grids, survey, em_catalog
        )

    dN_miss_members = None
    N_miss_members = None
    if q_members_row is not None:
        dM, NM = _completion_member_rows_vmap(rows, q_members_row, grids, survey, em_catalog)
        dN_miss_members = jnp.transpose(dM, (1, 0, 2))  # (M, N_rows, N_grid)
        N_miss_members = jnp.transpose(NM, (1, 0))      # (M, N_rows)

    return CompletionCurves(
        f=f, dN_miss=dN_miss, C_eff=C_eff, N_miss=N_miss,
        dN_miss_members=dN_miss_members, N_miss_members=N_miss_members,
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
    pixel, ``N_miss = integral (1 - C) dN_exp`` (missing-galaxy budget, LSS
    factor == 1: the field convention is gated to the legacy dummy-overdensity /
    no-Q_LSS regime).  Decomposed as

        Z = N_obs_total
            + Sum_occupied integral (1 - C_pix) dN_exp * depthmask
            + n_empty * integral dN_exp * depthmask,

    where empty pixels have ``C == 0`` (no observed galaxies) so their missing
    budget is the full ``dN_exp``.  ``C`` reuses ``_row_C``'s exact recipe --
    ``clip(dN_obs_s / where(dN_exp_smooth > 0, dN_exp_smooth, 1), 0, 1)`` -- and
    ``dN_exp = survey.n0 * apix * g(z)`` is the identical grid used per pixel, so
    the reduction is consistent with ``_assemble_curves`` term by term.  The
    ``survey.z_depth`` bound is applied EXACTLY as in ``_assemble_curves``
    (Python-level branch on the concrete ``z_depth``; ``None`` takes the untouched
    full-grid expression).

    Fully differentiable in ``theta`` (n0, cosmology, delta): the only frozen
    input is ``field_dN_obs_s`` (a data constant).  No ``optimization_barrier``
    is applied here.  The occupied-pixel reduction is chunked with ``lax.scan``
    to bound peak memory at high nside.
    """
    grids = _precompute_grids(cosmo, survey, em_catalog)
    dN_exp = grids.dN_exp                                    # (N_grid,) theta-dependent
    dN_exp_safe = jnp.where(grids.dN_exp_smooth > 0.0, grids.dN_exp_smooth, 1.0)

    field_obs = jnp.asarray(em_catalog.field_dN_obs_s)       # (n_occ, N_grid) f32 constant
    n_occ = int(field_obs.shape[0])                          # static
    N_obs_total = jnp.asarray(em_catalog.field_N_obs_total, dtype=dN_exp.dtype)
    n_empty = jnp.asarray(em_catalog.field_n_empty, dtype=dN_exp.dtype)

    # ``z_depth`` is concrete at trace time -> Python-level branch (mirrors
    # ``_assemble_curves``); ``None`` never constructs a mask.
    depth_mask = None if survey.z_depth is None else (zgrid <= survey.z_depth)

    def _row_N_miss(obs_row):
        obs_row = obs_row.astype(dN_exp.dtype)
        C = jnp.clip(obs_row / dN_exp_safe, 0.0, 1.0)
        dN_miss = (1.0 - C) * dN_exp
        if depth_mask is not None:
            dN_miss = jnp.where(depth_mask, dN_miss, 0.0)
        return jnp.trapezoid(dN_miss, zgrid)

    # Chunked scan over occupied rows: pad to a whole number of ``chunk_size``
    # blocks and mask the padding so padded (all-zero) rows -- which would
    # otherwise read as empty pixels with C == 0 -- contribute nothing.
    pad = (-n_occ) % chunk_size
    n_pad = n_occ + pad
    obs_pad = jnp.pad(field_obs, ((0, pad), (0, 0)))
    valid = jnp.arange(n_pad) < n_occ
    n_chunks = n_pad // chunk_size if chunk_size > 0 else 0
    obs_chunks = obs_pad.reshape(n_chunks, chunk_size, zgrid.size)
    valid_chunks = valid.reshape(n_chunks, chunk_size)

    def _body(acc, xs):
        obs_c, val_c = xs
        Nm = vmap(_row_N_miss)(obs_c)                        # (chunk,)
        Nm = jnp.where(val_c, Nm, 0.0)
        return acc + jnp.sum(Nm), None

    occ_miss_total, _ = lax.scan(
        _body, jnp.asarray(0.0, dtype=dN_exp.dtype), (obs_chunks, valid_chunks)
    )

    # Empty pixels: C == 0 -> dN_miss == dN_exp (same depth bound).
    empty_dN_miss = dN_exp if depth_mask is None else jnp.where(depth_mask, dN_exp, 0.0)
    empty_N_miss = jnp.trapezoid(empty_dN_miss, zgrid)

    Z = N_obs_total + occ_miss_total + n_empty * empty_N_miss
    return jnp.log(jnp.maximum(Z, 1e-300))


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
        cache_idx = em_catalog.pixel_to_cache_idx[global_pix]
        dN_obs = em_catalog.dN_obs_kde[cache_idx]
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
