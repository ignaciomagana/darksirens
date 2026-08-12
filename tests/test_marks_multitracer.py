"""Per-catalog marked-host models in the K-catalog mixture (PR-6 of the
multitracer unification): each catalog carries its own mark set and eta block
(``eta_<mark>`` for catalog 1, ``eta_<mark>_c{k}`` for catalogs 2..K), decoded
per catalog and applied inside that catalog's redshift-prior state; a catalog
with no marks runs the plain galaxy-count host model (h == 1) inside the
mixture."""
import jax
jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.redshift.completion import (
    build_field_mark_inputs,
    build_field_normalization_inputs,
)
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.inference.prior import build_parameter_space
from darksirens.inference.parameters import build_parameter_decoder
from darksirens.likelihood.factory import make_likelihood

NG = len(zgrid)


# ---------------------------------------------------------------------------
# Parameter space / decoder layout
# ---------------------------------------------------------------------------

def _space(**kw):
    res = build_parameter_space(
        pop_model="powerlaw+peak",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=True,
        universe_model="dark_sirens",
        mark_model="loglinear",
        **kw,
    )
    return res[0]  # labels


def test_label_order_k2_with_differing_per_catalog_marks():
    labels = _space(
        n_catalogs=2,
        mark_names=("logmstar", "logssfr"),
        mark_names_by_catalog=(("logmstar", "logssfr"), ("logmstar",)),
    )
    assert labels == ["fcat_2", "eta_logmstar", "eta_logssfr", "eta_logmstar_c2"]


def test_k1_label_layout_unchanged_by_new_kwarg():
    legacy = _space(mark_names=("logmstar",))
    explicit = _space(
        mark_names=("logmstar",), mark_names_by_catalog=(("logmstar",),)
    )
    assert legacy == explicit == ["eta_logmstar"]


def test_decode_mixture_slices_per_catalog_etas():
    opts = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        fix_population=True, fix_cosmology=True, fix_survey=True,
        prior_overrides=None, fixed_parameter_values={},
        mark_model="loglinear",
        mark_names=("logmstar", "logssfr"),
        mark_names_by_catalog=(("logmstar", "logssfr"), ("logmstar",)),
        n_catalogs=2,
        complete_empty_pixel_policy="volume",
    )
    pop_fid = get_fixed_population_params("powerlaw+peak")
    dec = build_parameter_decoder(opts, pop_fid)
    # Sampled coordinate order per the label test above.
    coord = jnp.asarray([0.3, 1.1, -0.7, 2.2])
    _c, _s, _p, _sky, mark_params_all, log_w = dec.decode_mixture(coord)
    assert len(mark_params_all) == 2
    np.testing.assert_allclose(np.asarray(mark_params_all[0]), [1.1, -0.7])
    np.testing.assert_allclose(np.asarray(mark_params_all[1]), [2.2])
    np.testing.assert_allclose(float(jnp.exp(log_w[1])), 0.3, rtol=1e-12)


# ---------------------------------------------------------------------------
# Likelihood-level identities (bundle source, FIELD weighting)
# ---------------------------------------------------------------------------

def _synthetic_full_sky(npix=12, maxg=3):
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


def _mark_table(npix=12, maxg=3, scale=1.0):
    return scale * (
        0.3 * np.sin(0.9 * np.arange(npix))[:, None]
        + 0.1 * np.arange(maxg)[None, :]
    )


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


def _marked_bundle(mark_scale=None, nsamp=2, n_sel=8):
    """Bundle over the synthetic sky with field inputs; marks (logmstar)
    attached when mark_scale is not None."""
    zgals, dzgals, wgals, ngals = _synthetic_full_sky()
    pe_pix = np.array([7, 7], dtype=np.int32)[:nsamp]
    sel_pix = np.array([1, 3, 4, 7, 1, 3, 4, 7], dtype=np.int32)[:n_sel]
    up_pe, s2u_pe = np.unique(pe_pix, return_inverse=True)
    up_se, s2u_se = np.unique(sel_pix, return_inverse=True)
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals)
    )
    bundle = dict(
        nside=1, apix=hp.nside2pixarea(1), n_pix_catalog=12,
        delta_g_pix_z=jnp.zeros((1, NG)),
        zgals_pe=zgals[up_pe], dzgals_pe=dzgals[up_pe], wgals_pe=wgals[up_pe],
        ngals_pe=ngals[up_pe],
        unique_pixels_pe=up_pe.astype(np.int32),
        sample_to_unique_pe=s2u_pe.astype(np.int32),
        zgals_sel=zgals[up_se], dzgals_sel=dzgals[up_se], wgals_sel=wgals[up_se],
        ngals_sel=ngals[up_se],
        unique_pixels_sel=up_se.astype(np.int32),
        sample_to_unique_sel=s2u_se.astype(np.int32),
        field_dN_obs_s=field.dN_obs_s,
        field_n_empty=field.n_empty,
        field_N_obs_total=field.N_obs_total,
        field_occupied_pixels=field.occupied_pixels,
    )
    if mark_scale is not None:
        marks = _mark_table(scale=mark_scale)
        bundle["mark_logmstar"] = jnp.asarray(marks)
        fz, fw, fvals = build_field_mark_inputs(
            zgals, wgals, ngals, {"logmstar": marks}, ("logmstar",)
        )
        bundle["field_mark_z"] = fz
        bundle["field_mark_w"] = fw
        bundle["field_mark_values"] = fvals
    return bundle


def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {
        lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled
    }
    overrides = {sampled: [float(pop_lower[0]), float(pop_upper[0])]}
    mid = 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))
    return pop_fid, overrides, fixed, mid


def _marked_likelihood(bundles, mark_names_by_catalog, fixed, pop_fid, overrides):
    data = dict(_shared_physics())
    data["apix"] = hp.nside2pixarea(1)
    data["catalogs"] = list(bundles)
    mark_names = tuple(mark_names_by_catalog[0]) if mark_names_by_catalog else ()
    opts = SimpleNamespace(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True, fix_population=False, fix_survey=True,
        prior_overrides=overrides,
        fixed_parameter_values=fixed,
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
        catalog_sky_weighting="field",
        n_catalogs=len(bundles),
        mark_model="loglinear",
        mark_names=mark_names,
        mark_names_by_catalog=tuple(tuple(n) for n in mark_names_by_catalog),
    )
    return make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


def test_k2_duplicated_marked_equals_k1_marked_with_tied_etas():
    """K=2 (A+marks, A+marks) with eta_logmstar_c2 FIXED equal to the sampled
    eta_logmstar's value == K=1 (A+marks) at the same eta, for any fcat_2."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    eta_val = 0.8

    fixed_k1 = dict(fixed)
    ll_k1 = _marked_likelihood(
        [_marked_bundle(mark_scale=1.0)], (("logmstar",),),
        fixed_k1, pop_fid, overrides,
    )
    val_k1 = float(ll_k1(jnp.asarray([mid, eta_val])))
    assert np.isfinite(val_k1)

    fixed_k2 = dict(fixed)
    fixed_k2["eta_logmstar_c2"] = eta_val
    ll_k2 = _marked_likelihood(
        [_marked_bundle(mark_scale=1.0), _marked_bundle(mark_scale=1.0)],
        (("logmstar",), ("logmstar",)),
        fixed_k2, pop_fid, overrides,
    )
    # Sampled coords: [pop, fcat_2, eta_logmstar] (eta_c2 fixed).
    val_k2 = float(ll_k2(jnp.asarray([mid, 0.37, eta_val])))
    assert abs(val_k2 - val_k1) <= 1e-12


def test_eta_c2_moves_likelihood_only_through_catalog_2():
    """eta_logmstar_c2 is inert at fcat_2 = 0 (catalog 2 carries zero weight)
    and live at an interior fcat_2."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    ll = _marked_likelihood(
        [_marked_bundle(mark_scale=1.0), _marked_bundle(mark_scale=1.4)],
        (("logmstar",), ("logmstar",)),
        fixed, pop_fid, overrides,
    )
    # Sampled coords: [pop, fcat_2, eta_logmstar, eta_logmstar_c2].
    v_a0 = float(ll(jnp.asarray([mid, 0.0, 0.5, 0.2])))
    v_b0 = float(ll(jnp.asarray([mid, 0.0, 0.5, 1.5])))
    assert v_a0 == v_b0  # weight-0 catalog: eta_c2 exactly inert

    v_a = float(ll(jnp.asarray([mid, 0.5, 0.5, 0.2])))
    v_b = float(ll(jnp.asarray([mid, 0.5, 0.5, 1.5])))
    assert np.isfinite(v_a) and np.isfinite(v_b)
    assert v_a != v_b


def test_unmarked_catalog_runs_h1_inside_marked_mixture():
    """K=2 with marks on catalog 1 ONLY: catalog 2 runs the plain host model;
    the run is finite and eta (catalog 1's) is live."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    ll = _marked_likelihood(
        [_marked_bundle(mark_scale=1.0), _marked_bundle(mark_scale=None)],
        (("logmstar",), ()),
        fixed, pop_fid, overrides,
    )
    # Sampled coords: [pop, fcat_2, eta_logmstar] (no eta for catalog 2).
    v0 = float(ll(jnp.asarray([mid, 0.5, 0.0])))
    v1 = float(ll(jnp.asarray([mid, 0.5, 1.2])))
    assert np.isfinite(v0) and np.isfinite(v1)
    assert v0 != v1


def test_factory_rejects_saturating_marks():
    """An uncentred mark table must be refused where the likelihood is built.

    Marks this large put every real galaxy past the +-7 log-h rail everywhere
    in the eta prior, so the clipped log h is constant in eta and the eta
    posterior collapses to the h == 1 null.  Nothing downstream errors on that
    -- the run just reports a flat posterior -- so the factory has to catch it.
    """
    pop_fid, overrides, fixed, _ = _pop_bits()
    with pytest.raises(ValueError, match="clip"):
        _marked_likelihood(
            [_marked_bundle(mark_scale=20.0)], (("logmstar",),),
            fixed, pop_fid, overrides,
        )


def test_factory_accepts_centred_marks():
    """The same fixture at its normal scale builds cleanly (guard is not a
    blanket ban on marked runs)."""
    pop_fid, overrides, fixed, _ = _pop_bits()
    _marked_likelihood(
        [_marked_bundle(mark_scale=1.0)], (("logmstar",),),
        fixed, pop_fid, overrides,
    )
