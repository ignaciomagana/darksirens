"""FIELD-convention catalog sky weighting (``--catalog_sky_weighting field``).

The field convention replaces the per-pixel catalog-prior normalizer ``Z[pix] =
N_obs + N_miss`` with a survey-GLOBAL ``Z(theta) = Sum_all-pixels[N_obs +
N_miss]``, so a K>=2 mixture weight ``fcat_k`` measures the HOST FRACTION
(number-density / sky-clustering contrast) rather than the per-pixel z-shape
preference.  Empty pixels then carry weight proportional to their n0-implied
missing count, and ``log10n0`` becomes genuinely informative -- the estimand fix
motivated by the gws-agn campaign.

Design under test:
  * ``darksirens.redshift.completion.build_field_normalization_inputs`` /
    ``field_global_log_Z`` -- the survey-global normalizer, term-consistent with
    ``_assemble_curves``'s per-pixel completeness ``C``.
  * ``darksirens.redshift.prior`` -- ``DarkSirenPriorState.log_Z_global`` and the
    static ``catalog_sky_weighting`` branch in ``_eval_dark_scalar``.
  * ``darksirens.likelihood.{core,factory}`` -- static ``catalog_sky_weighting``
    threading, K=1/K>=2 wiring, and the scope gates.

Conventions mirror ``tests/test_completion_depth.py`` (x64 via conftest) and
``tests/test_multitracer_likelihood.py`` (tiny in-memory bundle fixtures; ``==``
for bit-identity, tight tolerances for the mixture identities).
"""
import jax
jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.redshift.completion import (
    _kde_dndz_obs,
    _precompute_grids,
    build_field_depth_inputs,
    build_field_normalization_inputs,
    field_global_log_Z,
)
from darksirens.utils.cosmology import dV_of_z
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood.factory import make_likelihood

# ``np.trapz`` was renamed ``np.trapezoid`` in NumPy 2; bind whichever the
# installed version provides (identical algorithm) so the brute-force
# assemblies below run on either.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(n0=1e-2, z_depth=None):
    return SurveyParams(
        n0=n0, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=1.0,
        z_depth=z_depth,
    )


def _synthetic_full_sky(npix=12, maxg=3, seed=0):
    """A tiny full-sky catalog: a few occupied pixels, the rest empty."""
    zgals = np.zeros((npix, maxg))
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    occ = {1: [0.10], 3: [0.20, 0.25], 4: [0.15], 7: [0.30, 0.32, 0.28]}
    for p, zs in occ.items():
        for j, z in enumerate(zs):
            zgals[p, j] = z
            wgals[p, j] = 1.0
        ngals[p] = len(zs)
    dzgals = np.full((npix, maxg), 0.02)
    return zgals, dzgals, wgals, ngals


def _catalog_with_field(zgals, wgals, ngals, apix=1.0, dzgals=None):
    """Build an EMCatalog carrying the field-normalization inputs (on-the-fly
    KDE fallback, ``dN_obs_kde=None`` -- the same recipe field_global_log_Z
    reproduces), including the flat full-sky DEPTH inputs the global observed
    term needs whenever a ``z_depth`` is in force."""
    if dzgals is None:
        dzgals = np.full_like(zgals, 0.02)
    fobs, n_empty, N_obs_total, _occ = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    depth = build_field_depth_inputs(
        jnp.asarray(zgals), jnp.asarray(dzgals), jnp.asarray(wgals),
        jnp.asarray(ngals),
    )
    return EMCatalog(
        apix=apix,
        zgals=jnp.asarray(zgals),
        dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals),
        ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        field_dN_obs_s=fobs,
        field_n_empty=jnp.asarray(float(n_empty)),
        field_N_obs_total=jnp.asarray(float(N_obs_total)),
        field_depth_z=depth.z,
        field_depth_dz=depth.dz,
        field_depth_c=depth.c,
    )


def _log_g_numpy(cosmo, survey, zq):
    """g(z) = dV_c/dz (1+z)^delta on an arbitrary NumPy grid."""
    g = np.asarray(
        dV_of_z(jnp.asarray(zq), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
    ) * (1.0 + zq) ** survey.delta
    return g


def _kernel_mass_numpy(cosmo, survey, z_i, sig, z_hi, n=40001):
    """``integral_0^{z_hi} N(z; z_i, sig) g(z) dz`` by dense NumPy quadrature.

    Independent of the library's Gauss-Legendre substitution: a +-12 sigma window
    (the discarded tails are ~1e-32 of the mass) sampled densely enough that the
    trapezoid error is ~1e-12 relative.  The normalising 1/(sig sqrt(2pi)) is
    dropped -- only ratios of these masses are used.
    """
    lo = max(0.0, z_i - 12.0 * sig)
    hi = min(z_hi, z_i + 12.0 * sig)
    if hi <= lo:
        return 0.0
    zq = np.linspace(lo, hi, n)
    k = np.exp(-0.5 * ((zq - z_i) / sig) ** 2)
    return float(_trapz(k * _log_g_numpy(cosmo, survey, zq), zq))


def _below_depth_mass_numpy(cosmo, survey, zgals, dzgals, wgals, ngals, p, z_depth):
    """``m_pix``: the catalog mixture's mass below ``z_depth`` for pixel ``p``.

    ``m = Sum_i (w_i / W_pix) Z_i^depth / Z_i`` -- the quantity the per-pixel
    numerator scales ``N_obs`` by (``kernels.log_depth_mass``), rebuilt here from
    first principles in NumPy.
    """
    if z_depth is None:
        return 1.0
    zmax = float(np.asarray(zgrid)[-1])
    n_real = int(ngals[p])
    w = np.asarray(wgals, dtype=float)[p, :n_real]
    zs = np.asarray(zgals, dtype=float)[p, :n_real]
    dzs = np.asarray(dzgals, dtype=float)[p, :n_real]
    sig = np.maximum(np.sqrt(dzs ** 2 + float(survey.sigma_kde) ** 2), 1e-4)
    W = w.sum()
    m = 0.0
    for i in range(n_real):
        full = _kernel_mass_numpy(cosmo, survey, zs[i], sig[i], zmax)
        below = _kernel_mass_numpy(cosmo, survey, zs[i], sig[i], z_depth)
        m += (w[i] / W) * (below / full)
    return m


def _brute_force_Z(cosmo, survey, zgals, ngals, apix=1.0, z_depth=None,
                   dzgals=None, wgals=None):
    """Pure-NumPy global normalizer Z from first principles: per pixel
    ``N_obs * m_pix + trapz(dN_miss)`` with the missing density
    ``(1 - C) dN_exp`` below ``z_depth`` and the FULL ``dN_exp`` (C := 0) beyond
    it (hosts past the depth are missing, not nonexistent); ``C`` is computed
    independently via the matched KDE / smoothed expected-count ratio.

    The observed term carries the NUMERATOR's depth convention: ``m_pix`` is the
    catalog mixture's below-depth mass, so a galaxy catalogued beyond the depth is
    represented ONCE, by the relaxed missing branch.  Multiplying by ``m_pix`` is
    exactly what the pre-fix ``field_global_log_Z`` omitted -- it added the raw
    ``N_obs``, double counting every above-depth galaxy.
    """
    npix = zgals.shape[0]
    if dzgals is None:
        dzgals = np.full_like(zgals, 0.02)
    if wgals is None:
        wgals = (np.arange(zgals.shape[1])[None, :] < np.asarray(ngals)[:, None]
                 ).astype(float)
    cat = EMCatalog(
        apix=apix, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(zgals),
        wgals=None, ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, len(zgrid))), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(cosmo, survey, cat)
    dN_exp = np.asarray(grids.dN_exp)
    dN_exp_smooth = np.asarray(grids.dN_exp_smooth)
    dN_exp_safe = np.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    zg = np.asarray(zgrid)
    below = np.ones_like(zg) if z_depth is None else (zg <= z_depth).astype(float)

    Z = 0.0
    for p in range(npix):
        nobs = int(ngals[p])
        if nobs > 0:
            Z += nobs * _below_depth_mass_numpy(
                cosmo, survey, zgals, dzgals, wgals, ngals, p, z_depth
            )
            obs = np.asarray(
                _kde_dndz_obs(p, jnp.asarray(zgals), ngals=jnp.asarray(ngals))
            )
        else:
            obs = np.zeros_like(zg)
        C = np.clip(obs / dN_exp_safe, 0.0, 1.0)
        dN_miss = below * (1.0 - C) * dN_exp + (1.0 - below) * dN_exp
        Z += _trapz(dN_miss, zg)
    return Z


# ---------------------------------------------------------------------------
# Completion-level: global Z vs brute force
# ---------------------------------------------------------------------------

def test_field_global_Z_matches_bruteforce():
    """The depths cover all three regimes: no depth; a depth ABOVE every galaxy
    (m == 1, the no-op z_depth asserts); and a depth BITING the catalog (galaxies
    at 0.25/0.28/0.30/0.32 sit beyond 0.22), where the pre-fix normalizer added
    the raw N_obs and the brute force adds N_obs * m_pix."""
    cosmo = _cosmo()
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    apix = hp.nside2pixarea(1)

    for z_depth in (None, float(np.asarray(zgrid)[len(zgrid) // 3]), 0.22):
        survey = _survey(n0=1e-2, z_depth=z_depth)
        cat = _catalog_with_field(zgals, wgals, ngals, apix=apix, dzgals=dzgals)
        Z = float(np.exp(np.asarray(field_global_log_Z(cosmo, survey, cat))))
        Z_bf = _brute_force_Z(cosmo, survey, zgals, ngals, apix=apix,
                              z_depth=z_depth, dzgals=dzgals, wgals=wgals)
        rel = abs(Z - Z_bf) / Z_bf
        # f32 field table -> the achieved tolerance is far tighter than the 1e-6
        # budget (the empty-pixel + total-count terms are exact f64), but assert
        # against the documented ceiling.
        assert rel <= 1e-6, (z_depth, Z, Z_bf, rel)


def _deep_full_sky(npix=12, n_occ=9, n_gal=6, z_depth=0.3, seed=3):
    """The F-1/F-2 fixture: 9 of 12 pixels occupied, 6 galaxies each, HALF of
    them beyond ``z_depth`` -- the regime where the pre-fix global normalizer
    counted the above-depth galaxies twice."""
    rng = np.random.default_rng(seed)
    zgals = np.zeros((npix, n_gal))
    wgals = np.zeros((npix, n_gal))
    ngals = np.zeros(npix, dtype=np.int32)
    for p in range(n_occ):
        below = rng.uniform(0.05, z_depth - 0.02, size=n_gal // 2)
        above = rng.uniform(z_depth + 0.05, 0.9, size=n_gal - n_gal // 2)
        zgals[p, :] = np.concatenate([below, above])
        wgals[p, :] = 1.0
        ngals[p] = n_gal
    return zgals, np.full((npix, n_gal), 0.02), wgals, ngals


def _sky_integrated_prior_mass(cosmo, survey, cat, npix):
    """``Sum_pix integral p_field(z, pix) dz`` -- must be 1 for a FULL-SKY catalog.

    The rows of ``cat`` ARE the sky (no compaction), so summing each row's prior
    mass covers every pixel: occupied rows contribute their catalog + missing
    branches, empty rows their pure missing branch.
    """
    state = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat, catalog_sky_weighting="field"
    )
    zg = np.asarray(zgrid)
    total = 0.0
    for p in range(npix):
        lp = np.asarray(eval_redshift_prior_with_state(
            "dark_sirens", state, jnp.asarray(zg),
            jnp.full(zg.size, p, dtype=jnp.int32), cosmo, survey, cat,
            catalog_sky_weighting="field",
        ))
        total += float(_trapz(np.exp(lp), zg))
    return total, state


@pytest.mark.parametrize("z_depth", [None, 0.3])
def test_field_prior_integrates_to_unity_over_the_sky(z_depth):
    """THE field-convention invariant: the prior is a density over (z, pixel), so
    over a FULL-SKY catalog it must integrate to 1 -- for every n0 and with or
    without a survey depth.

    Regression for F-1/F-2: the global normalizer used the RAW
    ``field_N_obs_total`` while the per-pixel numerator's observed branch
    integrates to ``N_obs * m_pix``, so every above-depth catalogued galaxy was
    counted once as observed and again in the (relaxed) missing budget.  On this
    fixture the sky mass fell to 0.640 / 0.770 / 0.898 / 0.988 for
    n0 = 1e-11 / 3e-11 / 1e-10 / 1e-9 -- worst exactly where the observed count is
    comparable to the missing budget (the well-observed regime).  No pre-existing
    test checked this identity.
    """
    cosmo = _cosmo()
    zgals, dzgals, wgals, ngals = _deep_full_sky()
    npix = zgals.shape[0]
    apix = hp.nside2pixarea(1)
    cat = _catalog_with_field(zgals, wgals, ngals, apix=apix, dzgals=dzgals)

    for n0 in (1e-11, 3e-11, 1e-10, 1e-9, 1e-2):
        survey = _survey(n0=n0, z_depth=z_depth)
        mass, _state = _sky_integrated_prior_mass(cosmo, survey, cat, npix)
        # 1e-3 covers the trapezoid error of integrating the sigma = 0.02 catalog
        # kernels on the shared (log-spaced, ~0.0023-wide) zgrid; the defect this
        # guards against is 36% at the first n0.
        assert abs(mass - 1.0) < 1e-3, (z_depth, n0, mass)


def test_field_depth_normalizer_is_jittable_and_differentiable():
    """The depth-scaled observed term is a lax.scan over the flat full-sky
    galaxies, so it must jit and carry finite reverse-mode gradients in every
    channel it depends on -- including ``sigma_kde``, which the pre-fix constant
    ``field_N_obs_total`` did not depend on at all."""
    zgals, dzgals, wgals, ngals = _deep_full_sky()
    cat = _catalog_with_field(zgals, wgals, ngals, apix=hp.nside2pixarea(1),
                              dzgals=dzgals)

    def f(H0, log10n0, delta, sigma_kde):
        cosmo = CosmoParams(H0=H0, Om0=0.3075, w0=-1.0, wa=0.0)
        survey = SurveyParams(
            n0=10.0 ** log10n0, z50=1.0, w=0.5, delta=delta, b_miss=1.0,
            alpha_miss=1.0, sigma_kde=sigma_kde, z_depth=0.3,
        )
        return field_global_log_Z(cosmo, survey, cat)

    args = (67.74, -10.0, 0.2, 0.03)
    assert np.isfinite(float(jax.jit(f)(*args)))
    grads = jax.grad(f, argnums=(0, 1, 2, 3))(*args)
    for i, g in enumerate(grads):
        assert np.isfinite(float(g)) and float(g) != 0.0, (i, float(g))
        a, b = list(args), list(args)
        h = 1e-4 if i == 0 else 1e-5
        a[i] += h
        b[i] -= h
        fd = (float(f(*a)) - float(f(*b))) / (2.0 * h)
        np.testing.assert_allclose(float(g), fd, rtol=1e-5)


def test_field_observed_total_is_the_full_sky_depth_scaled_count():
    """Exact (1e-12) identity between the two halves of the fix: the survey-global
    observed term equals ``Sum_pix N_obs,pix * exp(log_depth_mass_pix)``, the
    per-pixel amplitude the numerator uses.  Non-uniform weights, so the flat
    reduction's ``c_i = N_obs,pix * w_i / W_pix`` is exercised."""
    from darksirens.redshift.catalog import catalog_kernel_state
    from darksirens.redshift.completion import field_observed_global_total

    cosmo = _cosmo()
    zgals, dzgals, wgals, ngals = _deep_full_sky()
    rng = np.random.default_rng(5)
    wgals = np.where(wgals > 0.0, rng.uniform(0.3, 3.0, size=wgals.shape), 0.0)
    cat = _catalog_with_field(zgals, wgals, ngals, dzgals=dzgals)

    for z_depth in (None, 0.3):
        survey = _survey(z_depth=z_depth)
        got = float(field_observed_global_total(cosmo, survey, cat))
        kernels = catalog_kernel_state(cosmo, survey, cat, z_depth=z_depth)
        expect = float(np.sum(
            np.asarray(ngals, dtype=float)
            * np.exp(np.asarray(kernels.log_depth_mass))
        ))
        np.testing.assert_allclose(got, expect, rtol=1e-12)
    # The depth genuinely bites: half the galaxies sit beyond it.
    assert float(field_observed_global_total(cosmo, _survey(z_depth=0.3), cat)) \
        < 0.6 * float(np.asarray(cat.field_N_obs_total))


@pytest.mark.parametrize("z_depth", [None, 0.3])
def test_field_global_Z_is_the_full_sky_sum_of_per_pixel_Z(z_depth):
    """Exact algebraic form of the same invariant: for a full-sky catalog the
    survey-global normalizer IS the sum of the per-pixel (conditional) ones.

    Non-uniform galaxy weights, so the per-galaxy coefficient
    ``c_i = N_obs,pix * w_i / W_pix`` in the flat depth reduction is genuinely
    exercised (it is 1 only for unit weights).
    """
    cosmo = _cosmo()
    zgals, dzgals, wgals, ngals = _deep_full_sky()
    rng = np.random.default_rng(11)
    wgals = np.where(wgals > 0.0, rng.uniform(0.3, 3.0, size=wgals.shape), 0.0)
    apix = hp.nside2pixarea(1)
    cat = _catalog_with_field(zgals, wgals, ngals, apix=apix, dzgals=dzgals)

    for n0 in (1e-11, 1e-10, 1e-2):
        survey = _survey(n0=n0, z_depth=z_depth)
        Z_global = float(np.exp(np.asarray(field_global_log_Z(cosmo, survey, cat))))
        st_cond = prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat, catalog_sky_weighting="conditional"
        )
        Z_sum = float(np.sum(np.exp(np.asarray(st_cond.log_Z))))
        rel = abs(Z_global - Z_sum) / Z_sum
        # f32 field_dN_obs_s in the global completeness is the only difference.
        assert rel <= 1e-6, (z_depth, n0, Z_global, Z_sum, rel)


# ---------------------------------------------------------------------------
# Prior-state level: field changes the normalizer; PE/selection share Z
# ---------------------------------------------------------------------------

def test_field_prior_uses_global_not_per_pixel_normalizer():
    cosmo, survey = _cosmo(), _survey()
    zgals, _dz, wgals, ngals = _synthetic_full_sky()
    cat = _catalog_with_field(zgals, wgals, ngals)

    st_c = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat, catalog_sky_weighting="conditional"
    )
    st_f = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat, catalog_sky_weighting="field"
    )
    z = jnp.array([0.20, 0.30])
    pix = jnp.array([3, 7], dtype=jnp.int32)  # occupied rows
    lp_c = np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", st_c, z, pix, cosmo, survey, cat,
        catalog_sky_weighting="conditional",
    ))
    lp_f = np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", st_f, z, pix, cosmo, survey, cat,
        catalog_sky_weighting="field",
    ))
    assert np.all(np.isfinite(lp_c)) and np.all(np.isfinite(lp_f))
    # Field vs conditional differ: the numerator is identical, only the
    # denominator swaps Z[pix] -> Z_global, so the shift is exactly the
    # per-pixel log-normalizer difference.
    assert np.all(np.abs(lp_c - lp_f) > 1e-6)
    diff = lp_c - lp_f
    expected = np.asarray(st_f.log_Z_global) - np.asarray(st_c.log_Z)[np.asarray(pix)]
    np.testing.assert_allclose(diff, expected, rtol=1e-12, atol=1e-12)


def test_field_pe_selection_share_same_global_Z():
    """PE and selection states built from the SAME catalog + theta must carry
    the SAME log_Z_global (the constant cancels structurally between the PE and
    selection terms)."""
    cosmo, survey = _cosmo(), _survey()
    zgals, _dz, wgals, ngals = _synthetic_full_sky()
    cat_pe = _catalog_with_field(zgals, wgals, ngals)
    cat_sel = _catalog_with_field(zgals, wgals, ngals)  # same arrays -> same Z

    st_pe = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat_pe, catalog_sky_weighting="field"
    )
    st_sel = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat_sel, catalog_sky_weighting="field"
    )
    assert float(st_pe.log_Z_global) == float(st_sel.log_Z_global)


def test_field_empty_pixel_alive_and_n0_scaling():
    """In field mode an empty pixel's prior is finite and MONOTONE in n0 (it
    decreases as log10n0 -> very low), whereas in conditional mode it is exactly
    n0-independent (n0 cancels between the numerator and the per-pixel Z).  This
    pins the estimand difference the field convention was built for."""
    cosmo = _cosmo()
    # Row 0 empty, rows 1/2 occupied.
    zgals = np.array([[0.0, 0.0], [0.15, 0.0], [0.25, 0.30]])
    wgals = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    ngals = np.array([0, 1, 2], dtype=np.int32)
    z = jnp.array([0.20])
    empty_pix = jnp.array([0], dtype=jnp.int32)

    def _prior(mode, n0):
        survey = _survey(n0=n0)
        cat = _catalog_with_field(zgals, wgals, ngals)
        st = prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat, catalog_sky_weighting=mode
        )
        return float(eval_redshift_prior_with_state(
            "dark_sirens", st, z, empty_pix, cosmo, survey, cat,
            catalog_sky_weighting=mode,
        )[0])

    n0s = [1e-1, 1e-3, 1e-6]

    # Conditional: n0-invariant empty-pixel prior (the estimand blind spot).
    cond_vals = [_prior("conditional", n0) for n0 in n0s]
    assert np.all(np.isfinite(cond_vals))
    np.testing.assert_allclose(cond_vals, cond_vals[0], rtol=1e-9, atol=1e-9)

    # Field: finite at high n0, strictly decreasing as n0 -> low (log10n0 informative).
    field_vals = [_prior("field", n0) for n0 in n0s]
    assert np.isfinite(field_vals[0])
    assert field_vals[0] > field_vals[1] > field_vals[2]


# ---------------------------------------------------------------------------
# Likelihood level: full-sky single-catalog data path
# ---------------------------------------------------------------------------

def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled}
    return pop_lower, pop_upper, pop_labels, pop_fid, sampled, fixed


def _mid_pop():
    pop_lower, pop_upper, *_ = _pop_bits()
    return 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))


def _full_sky_data(nsamp=2, n_sel=8, deep=False):
    """test_selection_prior_model-style full-sky single-catalog data dict (the
    union/full-catalog path, so make_likelihood builds field inputs itself).

    ``deep=True`` adds a second galaxy per pixel BEYOND a 0.3 depth, so the
    depth-scaled and raw observed totals differ by ~2x.
    """
    nside = 1
    n_pix = hp.nside2npix(nside)
    n_gal = 2 if deep else 1
    zgals = np.full((n_pix, n_gal), 0.10, dtype=float)
    zgals[2, 0] = 0.28
    if deep:
        zgals[:, 1] = 0.70
    dzgals = np.full((n_pix, n_gal), 0.02, dtype=float)
    wgals = np.ones((n_pix, n_gal), dtype=float)
    ngals = np.full(n_pix, n_gal, dtype=np.int32)
    return {
        "nEvents": 1, "nsamp": nsamp, "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(nside), "nside": nside, "n_pix_catalog": n_pix,
        "zgals": zgals, "dzgals": dzgals, "wgals": wgals, "ngals_catalog": ngals,
        "zgals_catalog": zgals, "dzgals_catalog": dzgals, "wgals_catalog": wgals,
        # Compact (1, N_grid) dummy overdensity, matching what the production
        # loaders always emit for non-LSS runs (the field-mode gate rejects
        # expanded per-pixel grids by static shape).
        "delta_g_pix_z": jnp.zeros((1, len(zgrid))),
        "m1det": jnp.array([36.0, 38.0]), "m2det": jnp.array([28.8, 30.4]),
        "dL": jnp.array([460.0, 500.0]), "chieff": jnp.array([0.0, 0.02]),
        "p_pe": jnp.ones(nsamp), "pixels_pe": jnp.array([7, 7], dtype=jnp.int32),
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel), "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.array([2, 7, 2, 7, 2, 7, 2, 7], dtype=jnp.int32),
    }


def _base_opts(**overrides):
    pop_lower, pop_upper, _labels, _fid, sampled, fixed = _pop_bits()
    kwargs = dict(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True,
        prior_overrides={sampled: [float(pop_lower[0]), float(pop_upper[0])]},
        fixed_parameter_values=fixed, complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _single_ll_value(opts, data):
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    return ll, float(ll(jnp.asarray([_mid_pop()])))


def test_conditional_default_bit_identical_k1():
    """K=1: no flag == catalog_sky_weighting='conditional' (float ==)."""
    data = _full_sky_data()
    _ll_a, v_default = _single_ll_value(_base_opts(), dict(data))
    _ll_b, v_explicit = _single_ll_value(
        _base_opts(catalog_sky_weighting="conditional"), dict(data)
    )
    assert np.isfinite(v_default)
    assert v_default == v_explicit


def test_field_mode_changes_normalizer_and_grad_finite():
    """Field != conditional for a multi-pixel catalog, and jax.grad of the
    field-mode likelihood w.r.t. the sampler coord is finite."""
    data = _full_sky_data()
    _ll_c, v_cond = _single_ll_value(_base_opts(), dict(data))
    ll_field, v_field = _single_ll_value(
        _base_opts(catalog_sky_weighting="field"), dict(data)
    )
    assert np.isfinite(v_cond) and np.isfinite(v_field)
    assert v_field != v_cond

    grad = jax.grad(lambda c: ll_field(c))(jnp.asarray([_mid_pop()]))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_k1_field_likelihood_is_invariant_to_the_global_Z_convention():
    """K=1 INVARIANCE: ``log_Z_global`` enters the per-event PE terms and the
    ``-N log mu`` selection term the same number of times, so it cancels exactly.
    The F-1/F-2 fix therefore must NOT move any single-catalog likelihood value --
    only K>=2, where each catalog's ``Z_k`` sits inside its own mixture branch.

    Verified by running the SAME configuration twice, once with the pre-fix global
    observed term (the raw ``field_N_obs_total``, monkeypatched in) on a catalog
    whose galaxies extend well past the depth -- a ~2x error in Z -- and asserting
    the log-likelihood is unchanged.  ``darksiren_log_likelihood`` is a
    MODULE-LEVEL ``jax.jit``, so its cache is keyed on static args + input avals,
    NOT on the factory closure: without ``jax.clear_caches()`` the second build
    silently reuses the first trace and this test would pass vacuously.  The spy
    counter is the guard that it did not.
    """
    import darksirens.redshift.completion as completion

    data = _full_sky_data(deep=True)
    opts = _base_opts(
        catalog_sky_weighting="field",
        survey_z_depth=0.3, resolved_survey_z_depths=[0.3],
    )
    _ll, v_fixed = _single_ll_value(opts, dict(data))

    fixed_fn = completion.field_observed_global_total
    cosmo, survey = _cosmo(), _survey(z_depth=0.3)
    cat = _catalog_with_field(
        np.asarray(data["zgals"]), np.asarray(data["wgals"]),
        np.asarray(data["ngals_catalog"]), apix=data["apix"],
        dzgals=np.asarray(data["dzgals"]),
    )
    n_depth = float(fixed_fn(cosmo, survey, cat))
    n_raw = float(np.asarray(cat.field_N_obs_total))
    assert n_depth < 0.6 * n_raw, (n_depth, n_raw)   # the conventions differ a lot

    calls = []

    def _prefix_convention(_cosmo, _survey, em):
        calls.append(1)
        return jnp.asarray(em.field_N_obs_total, dtype=jnp.float64)

    completion.field_observed_global_total = _prefix_convention
    jax.clear_caches()
    try:
        _ll_b, v_prefix = _single_ll_value(opts, dict(data))
    finally:
        completion.field_observed_global_total = fixed_fn
        jax.clear_caches()

    assert calls, "the pre-fix convention was never traced (stale jit cache)"
    assert np.isfinite(v_fixed)
    # Analytic cancellation; only floating-point re-association separates them.
    assert abs(v_fixed - v_prefix) <= 1e-9 * max(1.0, abs(v_fixed)), (
        v_fixed, v_prefix
    )


def test_field_with_depth_requires_the_flat_depth_inputs():
    """Loud failure instead of a silent double count: a field-mode catalog with a
    survey depth but no ``field_depth_*`` arrays must be rejected."""
    cosmo, survey = _cosmo(), _survey(z_depth=0.3)
    zgals, dzgals, wgals, ngals = _deep_full_sky()
    cat = _catalog_with_field(zgals, wgals, ngals, dzgals=dzgals)
    cat = cat._replace(field_depth_z=None, field_depth_dz=None, field_depth_c=None)
    with pytest.raises(ValueError, match="field_depth_z"):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat, catalog_sky_weighting="field"
        )
    # ... and the normalizer itself refuses, not just the state gate.
    with pytest.raises(ValueError, match="field_depth"):
        field_global_log_Z(cosmo, survey, cat)


# ---------------------------------------------------------------------------
# Mixture (K=2) field identities -- mirror test_multitracer_likelihood.py
# ---------------------------------------------------------------------------

APIX1 = hp.nside2pixarea(1)
Z_A = 0.10
Z_B = 0.30


def _shared_physics(nsamp=2, n_sel=8):
    return dict(
        nEvents=1, nsamp=nsamp, Ndraw=float(n_sel),
        m1det=jnp.array([36.0, 38.0]), m2det=jnp.array([28.8, 30.4]),
        dL=jnp.array([460.0, 500.0]), chieff=jnp.array([0.0, 0.02]),
        p_pe=jnp.ones(nsamp),
        m1detsels=jnp.linspace(34.0, 40.0, n_sel),
        m2detsels=0.8 * jnp.linspace(34.0, 40.0, n_sel),
        dLsels=jnp.linspace(430.0, 530.0, n_sel),
        chieffsels=jnp.zeros(n_sel), p_draw=jnp.ones(n_sel),
    )


def _field_bundle(apix, z, dz=0.02, nsamp=2, n_sel=8, n_empty=3):
    """A single-occupied-row compact bundle PLUS survey-global field inputs (one
    observed galaxy, ``n_empty`` empty survey pixels).  Same compact-view
    contract as load_multitracer_catalog_bundles, extended with the precomputed
    field-normalization inputs so the mixture builder can read them directly."""
    bundle = dict(
        apix=apix,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        zgals_pe=np.array([[z]]), dzgals_pe=np.array([[dz]]),
        wgals_pe=np.array([[1.0]]), ngals_pe=np.array([1], dtype=np.int32),
        unique_pixels_pe=np.array([0], dtype=np.int32),
        sample_to_unique_pe=np.zeros(nsamp, dtype=np.int32),
        zgals_sel=np.array([[z]]), dzgals_sel=np.array([[dz]]),
        wgals_sel=np.array([[1.0]]), ngals_sel=np.array([1], dtype=np.int32),
        unique_pixels_sel=np.array([0], dtype=np.int32),
        sample_to_unique_sel=np.zeros(n_sel, dtype=np.int32),
    )
    fobs, _ne, nobs, _occ = build_field_normalization_inputs(
        jnp.asarray([[z]]), None, jnp.asarray([1], dtype=jnp.int32)
    )
    bundle["field_dN_obs_s"] = fobs
    bundle["field_n_empty"] = float(n_empty)
    bundle["field_N_obs_total"] = float(nobs)
    return bundle


def _k1_field_value(z):
    """K=1 field likelihood for catalog ``z`` (single-catalog path reads the
    field inputs supplied in ``data``)."""
    data = dict(_shared_physics())
    data.update(_field_bundle(APIX1, z))
    opts = _base_opts(catalog_sky_weighting="field")
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    return float(ll(jnp.asarray([_mid_pop()])))


def _k2_field_likelihood(z_a, z_b):
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_field_bundle(APIX1, z_a), _field_bundle(APIX1, z_b)]
    opts = _base_opts(n_catalogs=2, catalog_sky_weighting="field")
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    return make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


def test_conditional_default_bit_identical_k2():
    """K=2: no flag == catalog_sky_weighting='conditional' (float ==)."""
    def _conditional_bundle(z):
        b = _field_bundle(APIX1, z)
        for key in ("field_dN_obs_s", "field_n_empty", "field_N_obs_total"):
            b.pop(key)
        return b

    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_conditional_bundle(Z_A), _conditional_bundle(Z_B)]

    ll_default = make_likelihood(_base_opts(n_catalogs=2), dict(data),
                                 pop_fid, fixed_parameter_values=fixed)
    ll_explicit = make_likelihood(
        _base_opts(n_catalogs=2, catalog_sky_weighting="conditional"),
        dict(data), pop_fid, fixed_parameter_values=fixed,
    )
    coord = jnp.asarray([_mid_pop(), 0.37])
    v_default = float(ll_default(coord))
    v_explicit = float(ll_explicit(coord))
    assert np.isfinite(v_default)
    assert v_default == v_explicit


def test_k2_field_duplicated_equals_k1():
    """K=2 with catalog B == catalog A equals K=1(A) at ANY fcat_2 (weights sum
    to 1), in field mode -- tolerance <=1e-12."""
    val_k1 = _k1_field_value(Z_A)
    ll_k2 = _k2_field_likelihood(Z_A, Z_A)
    val_k2 = float(ll_k2(jnp.asarray([_mid_pop(), 0.37])))
    assert np.isfinite(val_k1)
    assert abs(val_k2 - val_k1) <= 1e-12


def test_k2_field_fcat_endpoints():
    """K=2 field: fcat_2 -> 0 collapses onto catalog A, fcat_2 -> 1 onto catalog
    B -- tolerance <=1e-9."""
    val_a = _k1_field_value(Z_A)
    val_b = _k1_field_value(Z_B)
    assert val_a != val_b
    ll_k2 = _k2_field_likelihood(Z_A, Z_B)
    val_at_0 = float(ll_k2(jnp.asarray([_mid_pop(), 0.0])))
    val_at_1 = float(ll_k2(jnp.asarray([_mid_pop(), 1.0])))
    assert abs(val_at_0 - val_a) <= 1e-9
    assert abs(val_at_1 - val_b) <= 1e-9


# ---------------------------------------------------------------------------
# Scope gates
# ---------------------------------------------------------------------------

def test_field_gate_rejects_marks_ensemble_and_inconsistent_budgets():
    cosmo, survey = _cosmo(), _survey()
    zgals, _dz, wgals, ngals = _synthetic_full_sky()

    # Marked-host model WITHOUT the flat full-sky mark inputs: rejected (the
    # marked global normalizer needs view-independent mu_miss / S_obs).
    cat_marks = _catalog_with_field(zgals, wgals, ngals)
    with pytest.raises(ValueError, match="field_mark"):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_marks,
            mark_model="loglinear", mark_names=("logmstar",),
            catalog_sky_weighting="field",
        )

    # Q ENSEMBLE without the per-member survey-global rows: rejected with the
    # build hint (numerator and per-member normalizers must share budgets).
    cat_members = cat_marks._replace(
        lss_completion_logq_members=jnp.zeros((2, zgals.shape[0], len(zgrid)))
    )
    with pytest.raises(ValueError, match="field_lss_q_members"):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_members, catalog_sky_weighting="field"
        )

    # Deterministic Q table WITHOUT the survey-global Q rows: the numerator
    # and normalizer would carry different budgets -> rejected with a build hint.
    cat_q_inconsistent = cat_marks._replace(
        lss_completion_logq=jnp.zeros((zgals.shape[0], len(zgrid)))
    )
    with pytest.raises(ValueError, match="field_lss_q"):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_q_inconsistent,
            catalog_sky_weighting="field",
        )

    # Per-pixel delta_g WITHOUT the survey-global delta_g rows: rejected.
    cat_dg_inconsistent = cat_marks._replace(
        delta_g_pix_z=jnp.zeros((zgals.shape[0], len(zgrid)))
    )
    with pytest.raises(ValueError, match="field_delta_g"):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_dg_inconsistent,
            catalog_sky_weighting="field",
        )

    # Missing field-normalization inputs: rejected (needs the full-sky catalog).
    cat_nofield = EMCatalog(
        apix=1.0, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(zgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, len(zgrid))), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    with pytest.raises(ValueError):
        prepare_redshift_prior_state(
            "dark_sirens", cosmo, survey, cat_nofield, catalog_sky_weighting="field"
        )
