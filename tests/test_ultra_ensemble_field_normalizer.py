"""Ensemble-ONLY Q_LSS catalogs under the FIELD convention.

Two seams pinned here:

1. **The scalar global normalizer carries the posterior-mean-Q budget.**  When
   a catalog holds ONLY an LSS-completion ensemble (no deterministic
   ``lss_completion_logq`` table), the per-pixel numerator carries the
   posterior-MEAN Q (``_resolve_lss_completion_row_tables`` ->
   ``_member_posterior_mean_q``), so ``field_global_log_Z`` (and its marked
   twin) must integrate the SAME budget -- the member mean of
   ``field_lss_q_members`` / ``field_lss_q_empty_sum_members`` -- not the
   ``Q == 1`` budget that ``_field_missing_curve``'s ``field_lss_q is None``
   dispatch would silently fall back to.  This is the loaded-table twin of the
   latent scalar injection at the top of ``field_global_log_Z``.

2. **The per-member prior diagnostic refuses latent states loudly.**
   ``eval_redshift_prior_members_with_state`` has no latent member path (latent
   runs GENERATE each member's log-Q in-likelihood); it must raise a
   self-describing ``NotImplementedError`` -- mirroring
   ``completion_clip_diagnostics`` -- instead of subscripting the ``None`` that
   ``_resolve_member_logq_row`` returns for a latent catalog.

Fixtures mirror tests/test_field_normalizer_modulations.py (tiny synthetic
full sky, x64 via conftest).
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift import zgrid
from darksirens.redshift.completion import (
    build_field_depth_inputs,
    build_field_lss_q_member_inputs,
    build_field_normalization_inputs,
    field_global_log_Z,
    field_global_log_Z_marked,
)
from darksirens.redshift.prior import (
    DarkSirenEnsemblePriorState,
    eval_redshift_prior_members_with_state,
    prepare_redshift_prior_state,
)

NG = len(zgrid)
NPIX = 12
M = 3


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(n0=1e-2):
    return SurveyParams(
        n0=n0, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=1.0,
    )


def _synthetic_full_sky(maxg=3):
    zgals = np.zeros((NPIX, maxg))
    wgals = np.zeros((NPIX, maxg))
    ngals = np.zeros(NPIX, dtype=np.int32)
    occ = {1: [0.10], 3: [0.20, 0.25], 4: [0.15], 7: [0.30, 0.32, 0.28]}
    for p, zs in occ.items():
        for j, z in enumerate(zs):
            zgals[p, j] = z
            wgals[p, j] = 1.0
        ngals[p] = len(zs)
    dzgals = np.full((NPIX, maxg), 0.02)
    return zgals, dzgals, wgals, ngals


def _logq_members():
    """(M, NPIX, NG) global log-Q ensemble with a nonzero mean-Q excess.

    Per-member offsets around a per-pixel base: Jensen makes
    ``mean_m exp(logQ_m) > 1`` even at zero-mean log offsets, so the
    posterior-mean budget measurably differs from the ``Q == 1`` one -- the
    difference the shipped normalizer dropped.
    """
    base = np.linspace(-0.4, 0.4, NG)
    det = np.array([base * np.cos(0.7 * p) for p in range(NPIX)])
    return np.stack([det + 0.25 * (m - 1) for m in range(M)])


def _catalog(zgals, dzgals, wgals, ngals, *, q_mode):
    """EMCatalog with field-normalization inputs.

    ``q_mode``: "ensemble" attaches ONLY the member tables (the configuration
    under test); "det_mean" attaches the posterior-mean rows as a
    deterministic table (the reference budget); "none" attaches no Q.
    """
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    depth = build_field_depth_inputs(
        jnp.asarray(zgals), jnp.asarray(dzgals), jnp.asarray(wgals),
        jnp.asarray(ngals),
    )
    occupied = np.asarray(field.occupied_pixels)
    kwargs = dict(
        apix=1.0,
        zgals=jnp.asarray(zgals),
        dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals),
        ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, NG)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
        field_dN_obs_s=field.dN_obs_s,
        field_n_empty=jnp.asarray(float(field.n_empty)),
        field_N_obs_total=jnp.asarray(float(field.N_obs_total)),
        field_occupied_pixels=jnp.asarray(occupied, dtype=jnp.int32),
        field_depth_z=depth.z,
        field_depth_dz=depth.dz,
        field_depth_c=depth.c,
    )
    logq_m = _logq_members()
    qm_occ, qm_empty = build_field_lss_q_member_inputs(
        jnp.asarray(logq_m), occupied, NPIX
    )
    if q_mode == "ensemble":
        kwargs["lss_completion_logq_members"] = jnp.asarray(logq_m)
        kwargs["lss_completion_indexing"] = 2
        kwargs["field_lss_q_members"] = qm_occ
        kwargs["field_lss_q_empty_sum_members"] = qm_empty
    elif q_mode == "det_mean":
        kwargs["field_lss_q"] = jnp.mean(jnp.asarray(qm_occ), axis=0)
        kwargs["field_lss_q_empty_sum"] = jnp.mean(jnp.asarray(qm_empty), axis=0)
    elif q_mode != "none":
        raise ValueError(q_mode)
    return EMCatalog(**kwargs)


def _cats():
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    return (
        _catalog(zgals, dzgals, wgals, ngals, q_mode="ensemble"),
        _catalog(zgals, dzgals, wgals, ngals, q_mode="det_mean"),
        _catalog(zgals, dzgals, wgals, ngals, q_mode="none"),
    )


# ---------------------------------------------------------------------------
# 1. Scalar field normalizer on ensemble-only catalogs
# ---------------------------------------------------------------------------

def test_ensemble_only_field_normalizer_carries_the_posterior_mean_q():
    cosmo, survey = _cosmo(), _survey()
    cat_ens, cat_det, cat_plain = _cats()

    Z_ens = float(field_global_log_Z(cosmo, survey, cat_ens))
    Z_det = float(field_global_log_Z(cosmo, survey, cat_det))
    Z_plain = float(field_global_log_Z(cosmo, survey, cat_plain))

    # The reference budget genuinely differs from Q == 1, so this test bites:
    # the shipped normalizer returned Z_plain for the ensemble-only catalog.
    assert abs(Z_det - Z_plain) > 1e-4
    np.testing.assert_allclose(Z_ens, Z_det, rtol=1e-12)


def test_ensemble_only_marked_field_normalizer_carries_the_posterior_mean_q():
    cosmo, survey = _cosmo(), _survey()
    cat_ens, cat_det, cat_plain = _cats()
    # mu_miss == 1 and a precomputed S_obs isolate the missing-budget V(z):
    # the marked twin shares _field_missing_curve with the unmarked one.
    mu = jnp.ones(NG)
    Zm_ens = float(field_global_log_Z_marked(
        cosmo, survey, cat_ens, mu, None, S_obs=7.0))
    Zm_det = float(field_global_log_Z_marked(
        cosmo, survey, cat_det, mu, None, S_obs=7.0))
    Zm_plain = float(field_global_log_Z_marked(
        cosmo, survey, cat_plain, mu, None, S_obs=7.0))

    assert abs(Zm_det - Zm_plain) > 1e-4
    np.testing.assert_allclose(Zm_ens, Zm_det, rtol=1e-12)


def test_prepared_ensemble_state_global_normalizer_matches_the_numerator():
    """End-to-end through prepare_redshift_prior_state: the scalar
    ``log_Z_global`` stored on the ensemble state must be the posterior-mean-Q
    normalizer -- the budget its own ``dN_miss`` numerator carries."""
    cosmo, survey = _cosmo(), _survey()
    cat_ens, cat_det, _ = _cats()

    state = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat_ens,
        catalog_sky_weighting="field",
    )
    assert isinstance(state, DarkSirenEnsemblePriorState)
    Z_det = float(field_global_log_Z(cosmo, survey, cat_det))
    np.testing.assert_allclose(float(state.log_Z_global), Z_det, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Per-member diagnostic on latent states: loud structural refusal
# ---------------------------------------------------------------------------

def test_member_diagnostic_refuses_latent_states():
    # Reuse the structurally complete latent fixture (aggregate c_mode, field
    # normalization + latent leaves) rather than duplicating ~100 lines of it.
    from tests.test_latent_p13 import CAT, THETA, _cosmo as _lat_cosmo, \
        _survey as _lat_survey

    t = THETA[0]
    cosmo, survey = _lat_cosmo(t), _lat_survey(t)
    state = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, CAT, catalog_sky_weighting="field",
    )
    assert isinstance(state, DarkSirenEnsemblePriorState)
    z = jnp.array([0.05, 0.20])
    pix = jnp.array([1, 3])
    with pytest.raises(NotImplementedError, match="latent"):
        eval_redshift_prior_members_with_state(
            "dark_sirens", state, z, pix, cosmo, survey, CAT,
            catalog_sky_weighting="field",
        )
