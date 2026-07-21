"""Two-pixel physics regression for FIELD vs CONDITIONAL catalog sky weighting.

The default dark-siren estimand is ``field`` (the JOINT catalog host-density
estimand): the per-pixel numerator ``N_obs*p_cat + dN_miss`` is normalized by the
survey-GLOBAL ``Z(theta)`` rather than by the per-pixel ``Z[pix] = N_obs +
N_miss``, so RELATIVE angular host density is PRESERVED.  The legacy
``conditional`` estimand divides each pixel by its own ``Z[pix]``, so every pixel
integrates to unit mass and that relative angular weighting is DISCARDED.

This pins the physics at the redshift-prior level with two pixels that carry
IDENTICAL radial catalog shapes but 1 vs 100 observed hosts (the normalized
per-pixel catalog prior ``p_cat`` is host-count-INDEPENDENT -- ``log_kw`` divides
by ``logsumexp`` over the row -- so the ONLY thing distinguishing the pixels is
the count ``N_obs`` in the numerator):

  * ``field``       -> exp(log-prior) ratio B/A equals the host-count ratio
                       100:1 (the angular weighting field preserves).
  * ``conditional`` -> ratio 1:1 (each pixel re-normalized to unit mass).
  * the two modes differ ONLY by a per-pixel constant in z (same radial shape),
    exactly ``log_Z_global - log_Z[pix]``.

The missing branch is made negligible (tiny ``n0`` so ``N_miss << N_obs``) so the
count ratio is uncontaminated by ``dN_miss``; the residual is bounded by an
explicit tolerance.  Built through the same public builders
(``build_field_normalization_inputs`` + ``prepare_redshift_prior_state`` +
``eval_redshift_prior_with_state``) as tests/test_catalog_sky_weighting.py.
"""
import jax
jax.config.update("jax_enable_x64", True)

import healpy as hp
import jax.numpy as jnp
import numpy as np

from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_field_normalization_inputs
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog


NPIX = hp.nside2npix(1)     # 12 full-sky pixels
APIX = hp.nside2pixarea(1)
PIX_A = 1                   # 1 host
PIX_B = 3                   # 100 hosts
Z0 = 0.20                   # shared host redshift -> identical p_cat shape
N_A, N_B = 1, 100
# Tiny n0 keeps the survey-global expected count minuscule, so the
# missing-galaxy budget N_miss << N_obs and the field/conditional contrast is the
# pure host-count ratio, uncontaminated by dN_miss.
N0 = 1e-14


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey():
    return SurveyParams(
        n0=N0, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=1.0,
        z_depth=None,
    )


def _two_pixel_catalog():
    """Full-sky nside=1 catalog: PIX_A has 1 host at Z0, PIX_B has 100 hosts all
    at Z0 (identical per-host redshift -> identical NORMALIZED p_cat), the rest
    empty.  Carries the field-normalization inputs (on-the-fly KDE fallback,
    ``dN_obs_kde=None``), mirroring tests/test_catalog_sky_weighting.py."""
    maxg = N_B
    zgals = np.zeros((NPIX, maxg))
    wgals = np.zeros((NPIX, maxg))
    ngals = np.zeros(NPIX, dtype=np.int32)

    zgals[PIX_A, 0] = Z0
    wgals[PIX_A, 0] = 1.0
    ngals[PIX_A] = N_A

    zgals[PIX_B, :N_B] = Z0
    wgals[PIX_B, :N_B] = 1.0
    ngals[PIX_B] = N_B

    dzgals = np.full((NPIX, maxg), 0.02)
    fobs, n_empty, N_obs_total, _occ = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    return EMCatalog(
        apix=APIX,
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
    )


def _field_state():
    """A single FIELD prior state.  It carries BOTH log_Z[pix] and
    log_Z_global, so the same numerator can be evaluated under either estimand
    (mode string picks the denominator in ``_eval_dark_scalar``) -- guaranteeing
    the two modes share a bit-identical numerator."""
    cosmo, survey = _cosmo(), _survey()
    cat = _two_pixel_catalog()
    st = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat, catalog_sky_weighting="field"
    )
    return cosmo, survey, cat, st


def _lp(mode, st, cosmo, survey, cat, z, pix):
    return np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", st, jnp.asarray(z, dtype=float),
        jnp.asarray(pix, dtype=jnp.int32), cosmo, survey, cat,
        catalog_sky_weighting=mode,
    ))


def test_field_ratio_is_host_count_ratio_conditional_is_unity():
    cosmo, survey, cat, st = _field_state()
    z = np.array([Z0, Z0])
    pix = np.array([PIX_A, PIX_B])

    lp_f = _lp("field", st, cosmo, survey, cat, z, pix)
    lp_c = _lp("conditional", st, cosmo, survey, cat, z, pix)
    assert np.all(np.isfinite(lp_f)) and np.all(np.isfinite(lp_c))

    # FIELD: relative angular host density preserved -> B/A == host-count ratio.
    field_ratio = float(np.exp(lp_f[1] - lp_f[0]))
    np.testing.assert_allclose(field_ratio, N_B / N_A, rtol=1e-2)

    # CONDITIONAL: each pixel re-normalized to unit mass -> B/A == 1.
    cond_ratio = float(np.exp(lp_c[1] - lp_c[0]))
    np.testing.assert_allclose(cond_ratio, 1.0, atol=1e-2)

    # And the two estimands genuinely disagree here (not a trivially-equal setup).
    assert abs(field_ratio - cond_ratio) > 1.0


def test_field_and_conditional_differ_by_per_pixel_constant_in_z():
    cosmo, survey, cat, st = _field_state()
    # A band of redshifts spanning the shared host z (same grid for both pixels).
    zs = np.linspace(0.15, 0.28, 7)
    for pix in (PIX_A, PIX_B):
        p = np.full(zs.shape, pix)
        lp_f = _lp("field", st, cosmo, survey, cat, zs, p)
        lp_c = _lp("conditional", st, cosmo, survey, cat, zs, p)
        diff = lp_c - lp_f  # conditional - field

        # Same RADIAL shape: the modes differ ONLY by a z-independent per-pixel
        # constant, so the difference has zero spread across z.
        assert np.ptp(diff) < 1e-9, (pix, float(np.ptp(diff)))

        # That constant is exactly log_Z_global - log_Z[pix].
        expected = float(np.asarray(st.log_Z_global)) - float(
            np.asarray(st.log_Z)[pix]
        )
        np.testing.assert_allclose(diff, expected, rtol=1e-9, atol=1e-9)

    # The per-pixel constants themselves differ between the two pixels: that
    # difference is precisely the host-count contrast conditional removes.
    const_a = float(np.asarray(st.log_Z_global)) - float(np.asarray(st.log_Z)[PIX_A])
    const_b = float(np.asarray(st.log_Z_global)) - float(np.asarray(st.log_Z)[PIX_B])
    np.testing.assert_allclose(const_b - const_a, np.log(N_A / N_B), atol=1e-2)
