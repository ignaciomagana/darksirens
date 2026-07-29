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

import weakref

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import logsumexp, ndtr, ndtri
from jax.scipy.stats import norm
from typing import NamedTuple, Any

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


# ------------------------------------------------------------
# Windowed per-sample catalog KDE (issue #280)
# ------------------------------------------------------------
# ``eval_log_catalog_prior_state`` historically gathered the whole padded row
# and evaluated a Gaussian for every galaxy in it, per sample: under vmap over
# samples that materialises (N_samp, N_max) transients per event even though at
# sigma_eff ~ 0.02 the kernel support is a few percent of the row's redshift
# range.  When the catalog rows are z-sorted (the load-time invariant
# established by ``darksirens.catalogs.io.sort_survey_rows_by_z``), a binary
# search locates the galaxies within +/- n_sigma * max(sigma_eff) of the sample
# and only a STATIC-size window of W galaxies is gathered and evaluated.
#
# W is static (jit-compatible); the half-width is a TRACED quantity derived
# from the row's max sigma_eff (sigma_kde is a sampled parameter, so kernel
# widths are not compile-time constants).  The window start is centred on the
# in-range galaxy block, so when the block fits in W it is covered entirely,
# when it overflows the truncation is symmetric, and when it is empty the
# window straddles the insertion point (nearest galaxies on both sides keep
# the far-tail evaluation finite).  Windows containing zero real galaxies go
# through the same ``_logsumexp_neginf_safe`` path as empty full rows
# (bit-identical -inf, finite gradients).
_KDE_WINDOW_SIZE = 1024                # static window size W; None = full row
_KDE_WINDOW_NSIGMA: float = 8.0        # half-width = n_sigma * max(sig_eff)


def configure_catalog_kde_window(size=1024, n_sigma=8.0):
    """Configure the windowed per-sample catalog-KDE evaluator.

    Trace-time configuration: call BEFORE the first likelihood evaluation
    (reconfiguring after a jit trace does not affect the compiled graph).

    ``size`` is the static window length W (galaxies gathered per sample);
    ``None`` disables windowing entirely — the full-row escape hatch for A/B
    validation.  ``n_sigma`` scales the traced half-width
    ``n_sigma * max_row(sigma_eff)`` used to locate contributing galaxies.
    The default (W=1024, n_sigma=8) holds max |delta log p_cat| < 1e-6 against
    the full-row evaluator across the ENTIRE sampled sigma_kde prior
    [0, 0.05] on a realistic 2113-galaxy row (see
    tests/test_catalog_kde_window.py); W=384 matches the external review's
    fiducial-only table but is NOT safe at the wide end of the prior.

    Windowing additionally requires rows verified z-sorted (see
    ``_rows_sorted_for_windowing``); catalogs loaded with
    ``load_survey(..., sort_rows_by_z=False)`` or built ad hoc fall back to
    the full-row path automatically.
    """
    global _KDE_WINDOW_SIZE, _KDE_WINDOW_NSIGMA
    if size is not None:
        size = int(size)
        if size < 2:
            raise ValueError(f"KDE window size must be >= 2, got {size}")
    n_sigma = float(n_sigma)
    if not n_sigma > 0.0:
        raise ValueError(f"KDE window n_sigma must be > 0, got {n_sigma}")
    _KDE_WINDOW_SIZE = size
    _KDE_WINDOW_NSIGMA = n_sigma


def recommended_kde_window(zgals, ngals, dzgals, sigma_kde_max, n_sigma=6.0):
    """Data-driven window size for :func:`configure_catalog_kde_window`.

    Returns the maximum number of galaxies any interval of length
    ``2 * n_sigma * max_row(sigma_eff)`` contains, over every row and every
    interval position, evaluated at the WIDEST kernel the sampled prior
    admits (``sigma_kde_max``; the prior upper bound in
    ``darksirens/inference/prior.py``, 0.05 by default).  A window at least
    this large keeps the nearest-neighbour truncation beyond ``n_sigma`` on
    both sides for every sample; at the default ``n_sigma=6`` the truncated
    kernel mass is < 2e-9 per galaxy, comfortably inside the 1e-6
    |delta log p_cat| validation bar.  Host-side numpy diagnostic — run once
    per catalog when sizing W, not in the hot path.
    """
    z = np.asarray(zgals)
    ng = np.asarray(ngals)
    dz = np.asarray(dzgals)
    if z.ndim != 2:
        raise ValueError("zgals must be (N_rows, N_max)")
    worst = 0
    for r in range(z.shape[0]):
        n = int(ng[r])
        if n < 2:
            worst = max(worst, n)
            continue
        zr = np.sort(z[r, :n])
        sig_max = float(
            np.max(np.sqrt(dz[r, :n] ** 2 + float(sigma_kde_max) ** 2))
        )
        width = 2.0 * float(n_sigma) * max(sig_max, SIGMA_EFF_FLOOR)
        hi = np.searchsorted(zr, zr + width, side="right")
        worst = max(worst, int(np.max(hi - np.arange(n))))
    return worst


# Verified-sorted verdicts keyed by id(zgals); each entry pins nothing (a
# weakref) and is revalidated if the id was recycled by a different array.
_SORTED_ROWS_CACHE: dict = {}

# Build-time attestation for jit-ARGUMENT catalogs.  The production likelihood
# (darksiren_log_likelihood, a module-level jit) receives every EMCatalog as a
# traced argument, so inside its trace _rows_sorted_for_windowing cannot verify
# the concrete data and the evaluator would fall back to the full row — i.e.
# the windowed hot path would NEVER engage where it matters most.  The
# likelihood factory holds the concrete arrays at build time: it verifies every
# catalog view it binds and arms this flag, which lets the evaluator window
# TRACED catalogs too.  Process-global, last-attestation-wins; a factory build
# with ANY unsorted view disarms it.  The concrete-array check still runs first
# and never consults this flag, so eager/closure callers and the bitwise
# unsorted-fallback contract are unaffected.
_ROWS_SORTED_ATTESTED: bool = False


def attest_rows_sorted_for_windowing(*em_catalogs) -> bool:
    """Verify the z-sort invariant on CONCRETE catalogs and arm windowing for
    the traced (jit-argument) evaluator path.

    Call with every catalog view the likelihood will bind (PE, selection, and
    all mixture views).  Returns the armed verdict: True only if every catalog
    with per-galaxy rows verifiably satisfies the invariant.  Catalog views
    without rows (``zgals is None``) are ignored rather than disarming.
    """
    global _ROWS_SORTED_ATTESTED
    verdict = True
    for cat in em_catalogs:
        zgals = getattr(cat, "zgals", None)
        ngals = getattr(cat, "ngals", None)
        if zgals is None or ngals is None:
            continue
        if not _rows_sorted_for_windowing(zgals, ngals):
            verdict = False
            break
    _ROWS_SORTED_ATTESTED = verdict
    return verdict


def _rows_sorted_for_windowing(zgals, ngals) -> bool:
    """True iff ``zgals``/``ngals`` are CONCRETE arrays whose rows verifiably
    satisfy the z-sort invariant (real prefix non-decreasing).

    This is the trace-time gate for the windowed evaluator: it never trusts a
    flag — the invariant is checked (once per array; verdicts cached by id +
    weakref identity) on the actual data.  Traced arrays return False, so any
    call path that feeds the catalog through a jit boundary as an argument
    keeps the bit-identical full-row path.
    """
    if zgals is None or ngals is None:
        return False
    key = id(zgals)
    entry = _SORTED_ROWS_CACHE.get(key)
    if entry is not None:
        ref, verdict = entry
        if ref() is zgals:
            return verdict
        del _SORTED_ROWS_CACHE[key]
    try:
        z = np.asarray(zgals)
        ng = np.asarray(ngals)
    except Exception:
        return False  # traced (or otherwise non-concrete) arrays
    if z.ndim != 2 or ng.ndim != 1 or ng.shape[0] != z.shape[0]:
        return False
    cols = np.arange(1, z.shape[1])[None, :]
    verdict = bool(np.all((np.diff(z, axis=1) >= 0) | (cols >= ng[:, None])))
    try:
        ref = weakref.ref(zgals)
    except TypeError:
        return verdict  # unlikely: not weakref-able -> verify every trace
    _SORTED_ROWS_CACHE[key] = (ref, verdict)
    return verdict


def _sorted_row_window_start(zgals, pix, z, half, n_real, window):
    """Start index of the W-galaxy window in row ``pix`` of z-sorted ``zgals``.

    Binary-searches ONLY the real prefix ``[0, n_real)`` of the row (scalar
    gathers per iteration; the padded tail is never consulted, so trailing
    padding values need no sentinel).  With ``i_lo``/``i_hi``/``i_z`` the
    insertion points of ``z - half`` / ``z + half`` / ``z``:

    - block fits (``i_hi - i_lo <= W``): the window covers [i_lo, i_hi)
      entirely, positioned as close to z-centred as that constraint allows
      (``clip(i_z - W//2, i_hi - W, i_lo)``); an EMPTY block degenerates to a
      pure straddle of the insertion point, keeping the nearest galaxies on
      both sides so far-tail evaluations stay finite.
    - block overflows W: nearest-neighbour truncation, ``i_z - W//2`` — the
      W/2 index-nearest galaxies on each side of z (index order IS z order),
      so the excluded terms on each side are z-farther than every included
      one on that side.  Centring on the BLOCK midpoint instead would skew
      the window into a dense cluster on one side and could exclude the
      sample's own neighbourhood entirely.
    """
    n_max = zgals.shape[1]
    # Fixed iteration count: each step halves [lo, hi); log2(n_max)+1 is
    # enough to reach lo == hi from any initial span <= n_max.
    n_iter = int(np.ceil(np.log2(max(int(n_max), 2)))) + 1
    targets = jnp.stack([z - half, z, z + half])
    lo0 = jnp.zeros((3,), dtype=jnp.int32)
    hi0 = jnp.full((3,), jnp.asarray(n_real, dtype=jnp.int32))

    def _body(_, lh):
        lo, hi = lh
        active = lo < hi
        mid = (lo + hi) // 2
        v = zgals[pix, mid]                     # (3,) scalar gathers
        go_right = active & (v < targets)
        lo = jnp.where(go_right, mid + 1, lo)
        hi = jnp.where(active & ~go_right, mid, hi)
        return lo, hi

    lo, _ = lax.fori_loop(0, n_iter, _body, (lo0, hi0))
    i_lo, i_z, i_hi = lo[0], lo[1], lo[2]
    centred = i_z - window // 2
    fits = (i_hi - i_lo) <= window
    start = jnp.where(
        fits, jnp.clip(centred, i_hi - window, i_lo), centred
    )
    return jnp.clip(start, 0, n_max - window)


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


def _row_log_kernel_norms(zs, sig_eff, real, log_g_grid, z_hi=_ZMAX):
    """
    log Z_i for one row: Z_i = ∫_0^{z_hi} N(z; z_i, sig_i) g(z) dz.

    Substituting u = Phi((z - z_i)/sig_i) maps the truncated integral to
    ∫_a^b g(z_i + sig_i * Phi^{-1}(u)) du with a = Phi(-z_i/sig),
    b = Phi((z_hi - z_i)/sig): truncation handled exactly, integrand
    smooth, Gauss–Legendre converges fast for any sigma.

    ``z_hi`` is the upper truncation limit (default the grid ``zMax``).
    Passing ``survey.z_depth`` yields the below-depth kernel mass Z_i^depth used
    to renormalise the catalog prior when a survey depth is set (see
    :func:`_renorm_log_kw_below_depth`).
    """
    a = ndtr(-zs / sig_eff)
    b = ndtr((z_hi - zs) / sig_eff)
    span = b - a                                            # (N_max,)
    u = a[..., None] + span[..., None] * _GL_X              # (N_max, K)
    u = jnp.clip(u, 1e-12, 1.0 - 1e-12)
    z_node = jnp.clip(zs[..., None] + sig_eff[..., None] * ndtri(u), 0.0, z_hi)
    g = jnp.exp(
        jnp.interp(z_node.reshape(-1), zgrid, log_g_grid)
    ).reshape(z_node.shape)
    Z = span * (g * _GL_W).sum(axis=-1)                     # (N_max,)
    return jnp.where(real & (Z > 0.0), jnp.log(jnp.maximum(Z, 1e-300)), 0.0)


def _renorm_log_kw_below_depth(
    log_kw, zs, sig_eff, real, log_g_grid, z_depth, has_galaxies
):
    """Renormalise per-galaxy kernel weights so the depth-truncated catalog
    prior ``p_cat(z | pix)`` stays a proper density.

    A magnitude-limited survey catalogs nothing past ``z_depth``, so the
    per-sample evaluator (:func:`eval_log_catalog_prior_state`) zeroes
    ``p_cat`` there.  Without a correction the mixture would then integrate to
    its below-depth mass ``m < 1``, breaking the additive-density prior's
    per-pixel unit normalisation (and, under the field convention, the match
    between the numerator's catalog mass and the ``N_obs`` in the global
    normalizer).  The mixture is therefore divided by

        m = ∫_0^{z_depth} p_cat(z|pix) dz = Σ_i exp(log_kw_i) · Z_i^depth,

    with ``Z_i^depth`` the same GL kernel norm truncated at ``z_depth``.  Empty
    rows (``log_kw ≡ -inf``) carry an inert ``m`` and are returned unchanged.

    ``m`` is ALSO returned, and the caller MUST scale the row's observed count
    by it.  Renormalising the shape without rescaling the amplitude would leave
    the observed branch integrating to the full ``N_obs`` below the depth, i.e.
    it would relocate the host probability of every above-depth catalogued
    galaxy onto the below-depth redshifts -- while ``_assemble_curves`` is
    simultaneously counting those same galaxies in the missing branch, which
    relaxes to the FULL ``dN_exp`` above the depth.  Measured on a 10-galaxy row
    with 4 galaxies below ``z_depth=0.3``, that placed 10.00 hosts below the
    depth instead of 3.81 -- a 2.6x over-weighting of exactly the redshifts that
    drive the dark-siren H0 constraint.  With ``N_obs -> N_obs * m`` the
    observed branch integrates to the number of galaxies actually catalogued
    below the depth, and the above-depth ones are represented once, by
    ``dN_miss = dN_exp`` (the stated intent: hosts beyond the depth are
    *missing*, not nonexistent).
    """
    log_Z_depth = _row_log_kernel_norms(zs, sig_eff, real, log_g_grid, z_hi=z_depth)
    log_m = jnp.where(
        has_galaxies, _logsumexp_neginf_safe(log_kw + log_Z_depth), 0.0
    )
    return jnp.where(real, log_kw - log_m, -jnp.inf), log_m


def _row_kernel_state(
    zs, dzs, ws, ngal, sigma_kde, log_g_grid, volume_weighted=False, z_depth=None
):
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

    Returns ``(log_kw, sig_eff, log_depth_mass)``; the evaluator's g(z) handling
    is selected by the same ``volume_weighted`` flag carried on
    :class:`CatalogKernelState`.  ``log_depth_mass`` is 0.0 (mass 1) whenever no
    depth truncation applies.
    """
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR)
    # The 1e-300 floor is a numerical BACKSTOP only (it keeps padding slots from
    # producing log(0) = -inf * 0 = NaN inside the traced reduction).  A real
    # galaxy with w <= 0 is a data error, not something to floor: it would carry
    # a ~-690 log-weight while still counting in ``ngal``, so the observed count
    # and the mixture would measure different galaxy sets.  Such weights are
    # rejected at the data-entry boundary by ``darksirens_pixelate``.
    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1e-300)), -jnp.inf)
    log_depth_mass = jnp.zeros((), dtype=sig_eff.dtype)

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
        # ``z_depth`` (concrete Python float or None; never traced) renormalises
        # the mixture to unit mass on [0, z_depth] so the depth-truncated prior
        # stays proper.  ``None`` skips it entirely (bit-identical legacy path).
        # ``log_depth_mass`` is the mass the mixture had below the depth BEFORE
        # renormalising; the caller must scale the row's observed count by it,
        # or the galaxies above the depth are counted twice (see
        # :func:`_renorm_log_kw_below_depth`).
        if z_depth is not None:
            log_kw, log_depth_mass = _renorm_log_kw_below_depth(
                log_kw, zs, sig_eff, real, log_g_grid, z_depth, has_galaxies
            )
    return log_kw, sig_eff, log_depth_mass


class CatalogKernelState(NamedTuple):
    """Per-galaxy kernel quantities for all catalog rows (one proposal)."""
    log_g_grid: jnp.ndarray  # (N_grid,)
    log_kw: jnp.ndarray      # (N_rows, N_max)
    sig_eff: jnp.ndarray     # (N_rows, N_max)
    volume_weighted: bool = False
    #: (N_rows,) log of the mixture mass that lay below ``z_depth`` BEFORE the
    #: shape was renormalised.  Callers scale the row's observed galaxy count by
    #: it so the depth-truncated catalog branch integrates to the number of
    #: galaxies actually catalogued below the depth.  All-zero when
    #: ``z_depth is None``.
    log_depth_mass: Any = 0.0
    # ``survey.z_depth`` (concrete Python float or None; never traced): when set,
    # ``log_kw`` is already renormalised to unit mass on [0, z_depth] and the
    # per-sample evaluator zeroes p_cat beyond it.  ``None`` means no depth
    # truncation (the legacy, bit-identical path).
    z_depth: Any = None
    #: (N_rows,) per-row max sigma_eff, the traced input to the windowed
    #: evaluator's half-width ``n_sigma * sig_eff_row_max[pix]``.  ``None``
    #: (e.g. a state built by hand) disables windowing for that state.
    sig_eff_row_max: Any = None


def catalog_kernel_state(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    log_g_grid: jnp.ndarray | None = None,
    volume_weighted: bool = False,
    z_depth=None,
) -> CatalogKernelState:
    """Precompute per-galaxy kernel quantities once per parameter proposal.

    ``z_depth`` (a concrete Python float or ``None``) is the survey depth beyond
    which the catalog asserts nothing: when set, each row's ``log_kw`` is
    renormalised to unit mass on ``[0, z_depth]`` and stored on the returned
    state so the evaluator zeroes ``p_cat`` there.  It is threaded ONLY from the
    incomplete ``dark_sirens`` prior; the complete-catalog and bright-siren paths
    pass ``None`` (they model the full universe, so there is no depth to bound).
    """
    if log_g_grid is None:
        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals
    ngals = em_catalog.ngals

    if ngals is not None:
        log_kw, sig_eff, log_depth_mass = _map_rows(
            lambda zs, dzs, ws, ng: _row_kernel_state(
                zs, dzs, ws, ng, survey.sigma_kde, log_g_grid, volume_weighted,
                z_depth,
            ),
            (zgals, dzgals, wgals, ngals),
        )
    else:
        log_kw, sig_eff, log_depth_mass = _map_rows(
            lambda zs, dzs, ws: _row_kernel_state(
                zs, dzs, ws, None, survey.sigma_kde, log_g_grid, volume_weighted,
                z_depth,
            ),
            (zgals, dzgals, wgals),
        )

    return CatalogKernelState(
        log_g_grid=log_g_grid, log_kw=log_kw, sig_eff=sig_eff,
        volume_weighted=volume_weighted, z_depth=z_depth,
        log_depth_mass=log_depth_mass,
        sig_eff_row_max=jnp.max(sig_eff, axis=1),
    )


# ------------------------------------------------------------
# Marked-host kernel state (galaxy marks -> BBH-host efficiency)
# ------------------------------------------------------------

def _row_marked_kernel_state(
    zs, dzs, ws, log_h_row, ngal, sigma_kde, log_g_grid, z_depth=None
):
    """Per-galaxy kernel quantities for one row using host-efficiency weights.

    Identical to :func:`_row_kernel_state` but the per-pixel-normalised weight is
    ``w_i·h_i`` (host efficiency ``h_i = exp(log_h_row_i)``) instead of ``w_i``,
    and the row's marked total ``log_N_host = log Σ_i w_i h_i`` is returned (it
    replaces the integer count in the assembled prior).  With ``log_h_row ≡ 0``
    and unit weights this reduces to :func:`_row_kernel_state`.
    """
    real = _row_real_mask(zs, ws, ngal)
    sig_eff = jnp.maximum(jnp.sqrt(dzs**2 + sigma_kde**2), SIGMA_EFF_FLOOR)

    # 1e-300: numerical backstop for padding slots only; real galaxies with
    # w <= 0 are rejected by ``darksirens_pixelate`` (see _row_kernel_state).
    log_w = jnp.where(real, jnp.log(jnp.maximum(ws, 1e-300)), -jnp.inf)
    log_wh = jnp.where(real, log_w + log_h_row, -jnp.inf)   # log(w_i h_i)
    lse = logsumexp(log_wh)                                 # log Σ_i w_i h_i
    has_galaxies = jnp.isfinite(lse)
    log_wh_norm = jnp.where(real, log_wh - jnp.where(has_galaxies, lse, 0.0), -jnp.inf)

    log_Z = _row_log_kernel_norms(zs, sig_eff, real, log_g_grid)
    log_kw = jnp.where(real, log_wh_norm - log_Z, -jnp.inf)
    # Depth renormalisation mirrors the unmarked twin: the marked mixture SHAPE
    # is renormalised to unit mass on [0, z_depth] and the below-depth mass is
    # returned so the caller scales the row's marked amplitude by it — the
    # marked count Σ_i w_i h_i alone would keep every above-depth galaxy's host
    # probability while _assemble_curves counts those galaxies again in the
    # missing branch (the double count 349b717 removed from the unmarked path).
    log_depth_mass = jnp.zeros((), dtype=sig_eff.dtype)
    if z_depth is not None:
        log_kw, log_depth_mass = _renorm_log_kw_below_depth(
            log_kw, zs, sig_eff, real, log_g_grid, z_depth, has_galaxies
        )
    log_N_host = jnp.where(has_galaxies, lse, -jnp.inf)
    return log_kw, sig_eff, log_N_host, log_depth_mass


def marked_catalog_kernel_state(
    cosmo: CosmoParams,
    survey: SurveyParams,
    em_catalog: EMCatalog,
    log_h: jnp.ndarray,
    log_g_grid: jnp.ndarray | None = None,
    z_depth=None,
):
    """Marked per-galaxy kernel state + per-row marked total ``log_N_host``.

    ``log_h`` is ``(N_rows, N_max_gals)`` per-galaxy log host efficiency (from
    :mod:`darksirens.marks`).  Returns ``(CatalogKernelState, log_N_host)`` where
    the state's ``log_kw`` carries the marked (per-pixel-normalised) weights, so
    the existing per-sample evaluator is reused unchanged.  ``z_depth`` behaves
    exactly as in :func:`catalog_kernel_state`: the marked mixture is
    renormalised to unit mass on ``[0, z_depth]`` and the state's
    ``log_depth_mass`` carries the below-depth mass, by which the caller must
    scale the marked amplitude ``exp(log_N_host)`` (see
    :func:`_renorm_log_kw_below_depth`).
    """
    if log_g_grid is None:
        log_g_grid = log_galaxy_measure_grid(cosmo, survey)
    zgals, dzgals, wgals = em_catalog.zgals, em_catalog.dzgals, em_catalog.wgals
    ngals = em_catalog.ngals

    if ngals is not None:
        log_kw, sig_eff, log_N_host, log_depth_mass = _map_rows(
            lambda zs, dzs, ws, lh, ng: _row_marked_kernel_state(
                zs, dzs, ws, lh, ng, survey.sigma_kde, log_g_grid, z_depth
            ),
            (zgals, dzgals, wgals, log_h, ngals),
        )
    else:
        log_kw, sig_eff, log_N_host, log_depth_mass = _map_rows(
            lambda zs, dzs, ws, lh: _row_marked_kernel_state(
                zs, dzs, ws, lh, None, survey.sigma_kde, log_g_grid, z_depth
            ),
            (zgals, dzgals, wgals, log_h),
        )

    return CatalogKernelState(
        log_g_grid=log_g_grid, log_kw=log_kw, sig_eff=sig_eff,
        log_depth_mass=log_depth_mass, z_depth=z_depth,
        sig_eff_row_max=jnp.max(sig_eff, axis=1),
    ), log_N_host


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

    O(W) per sample on the windowed hot path (rows z-sorted, see
    :func:`configure_catalog_kde_window`): a binary search over the row's real
    prefix, one fused 2-D windowed gather per array, one Gaussian logpdf per
    windowed galaxy, one logsumexp.  Falls back to the historical O(N_max)
    full-row evaluation when windowing is disabled, the rows are not
    verifiably sorted, or the row is not longer than the window.
    """
    window = _KDE_WINDOW_SIZE
    use_window = (
        window is not None
        and state.sig_eff_row_max is not None
        and em_catalog.ngals is not None
        and getattr(em_catalog.zgals, "ndim", 0) == 2
        and em_catalog.zgals.shape[1] > window
        and (
            _rows_sorted_for_windowing(em_catalog.zgals, em_catalog.ngals)
            # Traced catalogs (jit arguments — the production likelihood path)
            # cannot be verified here; the factory attests them at build time
            # with the concrete arrays (attest_rows_sorted_for_windowing).
            or (
                _ROWS_SORTED_ATTESTED
                and isinstance(em_catalog.zgals, jax.core.Tracer)
            )
        )
    )
    if use_window:
        pix_i = jnp.asarray(pix, dtype=jnp.int32)
        n_real = em_catalog.ngals[pix_i]
        half = _KDE_WINDOW_NSIGMA * state.sig_eff_row_max[pix_i]
        start = _sorted_row_window_start(
            em_catalog.zgals, pix_i, z, half, n_real, window
        )

        # Fused 2-D windowed gathers (contiguous dynamic slices; no index
        # vector, never the full row).  Padded slots inside the window carry
        # log_kw = -inf exactly as in the full row, so an all-padding window
        # reduces through the SAME ``_logsumexp_neginf_safe`` -inf path
        # (bit-identical, finite grads).
        def _win(a):
            return lax.dynamic_slice(a, (pix_i, start), (1, window))[0]

        zs = _win(em_catalog.zgals)
        log_kw = _win(state.log_kw)
        sig = _win(state.sig_eff)
    else:
        zs = em_catalog.zgals[pix]
        log_kw = state.log_kw[pix]
        sig = state.sig_eff[pix]
    log_mix = _logsumexp_neginf_safe(log_kw + norm.logpdf(z, zs, sig))
    # Volume-weighted (complete-catalog) kernels already carry g(z_i) in their
    # weights, so no front g(z); otherwise reapply the per-sample galaxy measure
    # g(z) that Z_i divided out per kernel.  ``volume_weighted`` is a static bool.
    log_g_front = jnp.where(state.volume_weighted, 0.0, jnp.interp(z, zgrid, state.log_g_grid))
    log_p_cat = log_g_front + log_mix
    # Depth truncation: a magnitude-limited survey catalogs nothing past
    # ``z_depth``, so p_cat asserts nothing there -- zero it (the missing branch
    # supplies the full expected population beyond the depth).  ``z_depth`` is a
    # concrete Python float or ``None`` (never traced), so this is a Python-level
    # branch resolved once per trace; ``None`` leaves p_cat untouched (legacy).
    # ``log_kw`` is already renormalised to unit mass on [0, z_depth], so p_cat
    # stays a proper density.
    if state.z_depth is not None:
        log_p_cat = jnp.where(z <= state.z_depth, log_p_cat, -jnp.inf)
    return log_p_cat


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
    # Unit-mass kernel convention, matching the dark_sirens_complete prior
    # state (prepare_redshift_prior_state): catalog counts already track
    # hosts per redshift shell, so volume-weighted (True) kernels would
    # double-count dV_c/dz for any catalog whose dN/dz follows the volume.
    log_kw, sig_eff, _log_depth_mass = _row_kernel_state(
        zs, dzs, ws, ngal, survey.sigma_kde, log_g_grid, False
    )
    log_g_z = jnp.interp(z, zgrid, log_g_grid)
    return log_g_z + _logsumexp_neginf_safe(log_kw + norm.logpdf(z, zs, sig_eff))


# Vectorised over (z, pix) pairs — both vmapped simultaneously so the
# call signature matches all prior assembly functions.
log_catalog_prior_vmap = jit(
    vmap(log_catalog_prior, in_axes=(0, 0, None, None, None), out_axes=0)
)
