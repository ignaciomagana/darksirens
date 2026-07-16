"""K>=2 catalog mixture for ``universe_model='dark_sirens_complete'`` under the
FIELD-convention sky weighting (``--catalog_sky_weighting field``).

PR-4 extends the K-catalog mixture to the *complete*-catalog host model, but
FIELD MODE ONLY.  For the complete model the field normalizer is
theta-INDEPENDENT -- ``Z = Sum_all-pixels N_obs`` (no missing-galaxy budget) --
so the per-pixel complete prior ``p_cat(z|pix)`` is reweighted by the pixel's
share ``N_obs[pix] / N_obs_total`` of the total observed count, and an EMPTY
pixel carries zero count weight -> ``-inf`` REGARDLESS of
``complete_empty_pixel_policy``.

Because the mixture per-sample logsumexp is all--inf-safe (core.py
``_mixture_logsumexp``), a POPULATED catalog rescues a node where a sparse
catalog's pixel is empty -- retiring the fixed-node ``-inf`` artifact of
complete x sparse single-catalog scans.

Conditional-complete K>=2 stays FORBIDDEN: mixing per-pixel-normalized complete
priors is the incoherent estimand.

Conventions mirror ``tests/test_catalog_sky_weighting.py`` and
``tests/test_multitracer_likelihood.py``: tiny in-memory bundle fixtures, ``==``
for bit-identity, 1e-12 for the mixture identities.
"""
import jax
jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_field_normalization_inputs
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood.factory import make_likelihood


APIX1 = hp.nside2pixarea(1)
# Both catalog redshifts sit within a few galaxy-KDE widths of the toy GW event
# (dL ~ 460-500 Mpc -> z ~ 0.10 for the fixed cosmology).  Unlike dark_sirens,
# the COMPLETE model has no missing-galaxy budget to smoothly rescue a distant
# galaxy, so a catalog whose only galaxy is far from the event redshift gives a
# hard -inf -- Z_B is kept near the event (0.14, distinct from Z_A=0.10) so both
# single-catalog complete+field likelihoods are finite AND distinguishable.
Z_A = 0.10
Z_B = 0.14


# ---------------------------------------------------------------------------
# Population / cosmology / survey fixtures
# ---------------------------------------------------------------------------

def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled}
    return pop_lower, pop_upper, pop_labels, pop_fid, sampled, fixed


def _mid_pop():
    pop_lower, pop_upper, *_ = _pop_bits()
    return 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))


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


def _base_opts(**overrides):
    pop_lower, pop_upper, _labels, _fid, sampled, fixed = _pop_bits()
    kwargs = dict(
        pop_model="powerlaw+peak", universe_model="dark_sirens_complete",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True,
        prior_overrides={sampled: [float(pop_lower[0]), float(pop_upper[0])]},
        fixed_parameter_values=fixed, complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


# ---------------------------------------------------------------------------
# Catalog-bundle fixtures (same compact-view contract the loaders produce)
# ---------------------------------------------------------------------------

def _field_bundle(apix, z, dz=0.02, nsamp=2, n_sel=8, n_empty=3):
    """Single-occupied-row compact bundle + survey-global field inputs (one
    observed galaxy, ``n_empty`` empty survey pixels).  Mirrors
    ``tests/test_catalog_sky_weighting.py::_field_bundle``."""
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


def _empty_pixel_bundle(apix, nsamp=2, n_sel=8, n_empty=3, N_obs_total=5.0):
    """A catalog whose PE/selection samples all land in an EMPTY pixel (compact
    row with ``ngals=0``): the complete-model field prior is ``-inf`` there.  The
    survey-global ``field_N_obs_total`` is still positive (the catalog's galaxies
    live in OTHER pixels), so this is a genuinely sparse -- not degenerate --
    catalog whose K=1 complete likelihood is ``-inf`` at this GW event."""
    bundle = dict(
        apix=apix,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        zgals_pe=np.array([[0.0]]), dzgals_pe=np.array([[1.0]]),
        wgals_pe=np.array([[0.0]]), ngals_pe=np.array([0], dtype=np.int32),
        unique_pixels_pe=np.array([0], dtype=np.int32),
        sample_to_unique_pe=np.zeros(nsamp, dtype=np.int32),
        zgals_sel=np.array([[0.0]]), dzgals_sel=np.array([[1.0]]),
        wgals_sel=np.array([[0.0]]), ngals_sel=np.array([0], dtype=np.int32),
        unique_pixels_sel=np.array([0], dtype=np.int32),
        sample_to_unique_sel=np.zeros(n_sel, dtype=np.int32),
    )
    # field inputs from an occupied synthetic full-sky (galaxies elsewhere);
    # field_dN_obs_s is unread by the complete path but must be present for the
    # field-mode scope gate, and field_N_obs_total > 0 is the survey total.
    fobs, _ne, _nobs, _occ = build_field_normalization_inputs(
        jnp.asarray([[0.20]]), None, jnp.asarray([1], dtype=jnp.int32)
    )
    bundle["field_dN_obs_s"] = fobs
    bundle["field_n_empty"] = float(n_empty)
    bundle["field_N_obs_total"] = float(N_obs_total)
    return bundle


def _conditional_bundle(z):
    """A ``_field_bundle`` with the field-normalization keys stripped -- the
    conditional single-/multi-catalog path never reads them."""
    b = _field_bundle(APIX1, z)
    for key in ("field_dN_obs_s", "field_n_empty", "field_N_obs_total"):
        b.pop(key)
    return b


# ---------------------------------------------------------------------------
# Likelihood builders
# ---------------------------------------------------------------------------

def _k1_field_value(z):
    data = dict(_shared_physics())
    data.update(_field_bundle(APIX1, z))
    opts = _base_opts(catalog_sky_weighting="field")
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    return float(ll(jnp.asarray([_mid_pop()])))


def _k2_field_likelihood(bundle_a, bundle_b):
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [bundle_a, bundle_b]
    opts = _base_opts(n_catalogs=2, catalog_sky_weighting="field")
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    return make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_complete_k2_field_finite_and_grad():
    """K=2 complete + field: the mixture logL is finite and its jax.grad w.r.t.
    the sampler coord (pop param + fcat_2) is finite -- the all--inf-safe
    mixture logsumexp keeps the backward pass clean."""
    ll_k2 = _k2_field_likelihood(_field_bundle(APIX1, Z_A), _field_bundle(APIX1, Z_B))
    coord = jnp.asarray([_mid_pop(), 0.3])
    val = float(ll_k2(coord))
    assert np.isfinite(val)

    grad = jax.grad(lambda c: ll_k2(c))(coord)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_complete_field_empty_pixel_zero():
    """In field mode the complete-model prior at an EMPTY pixel is exactly
    ``-inf`` for BOTH ``complete_empty_pixel_policy`` values (0 = zero, 1 =
    volume): the theta-independent field normalizer gives an empty pixel zero
    count weight, so the volume fallback is irrelevant."""
    cosmo = _cosmo()
    # Row 0 empty; rows 1, 2 occupied.
    zgals = np.array([[0.0, 0.0], [0.15, 0.0], [0.25, 0.30]])
    wgals = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    ngals = np.array([0, 1, 2], dtype=np.int32)
    fobs, n_empty, N_obs_total, _occ = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )

    def _cat():
        return EMCatalog(
            apix=1.0, zgals=jnp.asarray(zgals),
            dzgals=jnp.asarray(np.full_like(zgals, 0.02)),
            wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
            delta_g_pix_z=jnp.zeros((1, len(zgrid))), dN_obs_kde=None,
            pixel_to_cache_idx=None, field_dN_obs_s=fobs,
            field_n_empty=jnp.asarray(float(n_empty)),
            field_N_obs_total=jnp.asarray(float(N_obs_total)),
        )

    z = jnp.array([0.20, 0.25])
    empty_pix = jnp.array([0, 0], dtype=jnp.int32)
    occ_pix = jnp.array([2, 2], dtype=jnp.int32)

    for policy in (0, 1):  # 0 = zero/-inf, 1 = volume fallback
        survey = SurveyParams(
            n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=1.0,
            complete_empty_pixel_policy=policy,
        )
        cat = _cat()
        st = prepare_redshift_prior_state(
            "dark_sirens_complete", cosmo, survey, cat, catalog_sky_weighting="field"
        )
        lp_empty = np.asarray(eval_redshift_prior_with_state(
            "dark_sirens_complete", st, z, empty_pix, cosmo, survey, cat,
            catalog_sky_weighting="field",
        ))
        lp_occ = np.asarray(eval_redshift_prior_with_state(
            "dark_sirens_complete", st, z, occ_pix, cosmo, survey, cat,
            catalog_sky_weighting="field",
        ))
        assert np.all(np.isneginf(lp_empty)), (policy, lp_empty)
        assert np.all(np.isfinite(lp_occ)), (policy, lp_occ)


def test_complete_k2_duplicated_equals_k1_field():
    """K=2 with catalog B == catalog A equals K=1(A) at ANY fcat_2 (the weights
    sum to 1, so w_1 p_A + w_2 p_A = p_A identically), in complete+field mode --
    tolerance <=1e-12."""
    val_k1 = _k1_field_value(Z_A)
    ll_k2 = _k2_field_likelihood(_field_bundle(APIX1, Z_A), _field_bundle(APIX1, Z_A))
    val_k2 = float(ll_k2(jnp.asarray([_mid_pop(), 0.37])))
    assert np.isfinite(val_k1)
    assert abs(val_k2 - val_k1) <= 1e-12


def test_complete_k2_field_fcat_endpoints():
    """K=2 complete+field: fcat_2 -> 0 collapses onto catalog A, fcat_2 -> 1 onto
    catalog B -- tolerance <=1e-9."""
    val_a = _k1_field_value(Z_A)
    val_b = _k1_field_value(Z_B)
    assert val_a != val_b  # genuinely distinguishable catalogs

    ll_k2 = _k2_field_likelihood(_field_bundle(APIX1, Z_A), _field_bundle(APIX1, Z_B))
    val_at_0 = float(ll_k2(jnp.asarray([_mid_pop(), 0.0])))
    val_at_1 = float(ll_k2(jnp.asarray([_mid_pop(), 1.0])))
    assert abs(val_at_0 - val_a) <= 1e-9
    assert abs(val_at_1 - val_b) <= 1e-9


def test_complete_conditional_k2_still_guarded():
    """A K>=2 mixture with ``universe_model='dark_sirens_complete'`` under the
    CONDITIONAL (default) normalizer must raise ``NotImplementedError`` at the
    core guard (mixing per-pixel-normalized complete priors is the incoherent
    estimand).  Exercised through make_likelihood so the core guard is hit."""
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_field_bundle(APIX1, Z_A), _field_bundle(APIX1, Z_B)]
    opts = _base_opts(n_catalogs=2, catalog_sky_weighting="conditional")
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    with pytest.raises(NotImplementedError):
        ll(jnp.asarray([_mid_pop(), 0.3]))


def test_complete_k1_conditional_bit_identical():
    """K=1 complete under the default flag == an explicit
    ``catalog_sky_weighting='conditional'``: the complete conditional path is
    untouched by PR-4 (the new state leaves are never read).  Assert float
    ``==`` as the contract."""
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()
    data = dict(_shared_physics())
    data.update(_conditional_bundle(Z_A))

    ll_default = make_likelihood(_base_opts(), dict(data),
                                 pop_fid, fixed_parameter_values=fixed)
    ll_explicit = make_likelihood(
        _base_opts(catalog_sky_weighting="conditional"),
        dict(data), pop_fid, fixed_parameter_values=fixed,
    )
    coord = jnp.asarray([_mid_pop()])
    v_default = float(ll_default(coord))
    v_explicit = float(ll_explicit(coord))
    assert np.isfinite(v_default)
    assert v_default == v_explicit


def test_mixture_rescues_sparse_empty_nodes():
    """The notch-rescue mechanism: catalog 2 is so sparse that the GW event's
    pixel is EMPTY there, so its K=1 complete+field logL is ``-inf``; but the
    K=2 field mixture (catalog 1 populated, fcat_2 < 1) is FINITE because the
    all--inf-safe per-sample logsumexp lets the populated catalog carry the
    node.  This retires the fixed-node ``-inf`` artifact of complete x sparse
    single-catalog scans."""
    _l, _u, _lab, pop_fid, _s, fixed = _pop_bits()

    # K=1 with the sparse catalog alone -> -inf (its pixel is empty).
    data_k1 = dict(_shared_physics())
    data_k1.update(_empty_pixel_bundle(APIX1))
    ll_k1_sparse = make_likelihood(
        _base_opts(catalog_sky_weighting="field"), data_k1,
        pop_fid, fixed_parameter_values=fixed,
    )
    v_sparse = float(ll_k1_sparse(jnp.asarray([_mid_pop()])))
    assert np.isneginf(v_sparse)

    # K=2 (populated catalog 1 + sparse/empty catalog 2), fcat_2 = 0.5 -> finite.
    ll_k2 = _k2_field_likelihood(_field_bundle(APIX1, Z_A), _empty_pixel_bundle(APIX1))
    v_mix = float(ll_k2(jnp.asarray([_mid_pop(), 0.5])))
    assert np.isfinite(v_mix)
