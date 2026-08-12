"""K-catalog multitracer mixture likelihood (darksirens/likelihood/{core,factory}.py).

Mirrors the tiny in-memory fixture style of test_selection_prior_model.py: a
one-event, nsamp=2, n_sel=8 dark-siren toy problem, fixed cosmology/survey, one
free population parameter.  Catalog bundles use the SAME compact-view keys
(``zgals_pe``/``unique_pixels_pe``/``sample_to_unique_pe`` etc.) that
``load_multitracer_catalog_bundles`` produces, so ``prepare_catalog_views``
consumes them identically whether they arrive via the real loader or (as
here) hand-built directly.

Invariant under test throughout: K=1 is the bit-identical pre-existing path
(darksirens/likelihood/factory.py::make_likelihood dispatches to
``_make_mixture_likelihood`` ONLY for n_catalogs >= 2); the mixture math lives
in darksirens/likelihood/core.py's ``n_catalogs >= 2`` branch of
``darksiren_log_likelihood``.
"""
from types import SimpleNamespace

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood import factory as factory_mod
from darksirens.likelihood.factory import make_likelihood
from darksirens.inference.prior import build_parameter_space


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _bundle(apix, z, dz=0.02, nsamp=2, n_sel=8):
    """A single-galaxy, single-(compact)-row catalog bundle: one PE row, one
    selection row, ALL PE/selection samples map to that one row.  This is the
    same compact-view contract ``load_multitracer_catalog_bundles`` produces
    (see darksirens/inference/loaders.py), just built directly so the test
    fixture has no dependency on real HEALPix ang2pix / HDF5 I/O."""
    zgals_pe = np.array([[z]])
    dzgals_pe = np.array([[dz]])
    wgals_pe = np.array([[1.0]])
    ngals_pe = np.array([1], dtype=np.int32)
    unique_pixels_pe = np.array([0], dtype=np.int32)
    sample_to_unique_pe = np.zeros(nsamp, dtype=np.int32)

    zgals_sel = np.array([[z]])
    dzgals_sel = np.array([[dz]])
    wgals_sel = np.array([[1.0]])
    ngals_sel = np.array([1], dtype=np.int32)
    unique_pixels_sel = np.array([0], dtype=np.int32)
    sample_to_unique_sel = np.zeros(n_sel, dtype=np.int32)

    return dict(
        apix=apix,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        zgals_pe=zgals_pe, dzgals_pe=dzgals_pe, wgals_pe=wgals_pe, ngals_pe=ngals_pe,
        unique_pixels_pe=unique_pixels_pe, sample_to_unique_pe=sample_to_unique_pe,
        zgals_sel=zgals_sel, dzgals_sel=dzgals_sel, wgals_sel=wgals_sel, ngals_sel=ngals_sel,
        unique_pixels_sel=unique_pixels_sel, sample_to_unique_sel=sample_to_unique_sel,
    )


def _shared_physics(nsamp=2, n_sel=8):
    """GW PE/selection physics arrays shared across every fixture in this
    file (identical numbers to test_selection_prior_model.py / the
    test_numpyro_sampler.py ``_small_likelihood_inputs`` template)."""
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


def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {
        lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled
    }
    return pop_lower, pop_upper, pop_labels, pop_fid, sampled, fixed


APIX1 = hp.nside2pixarea(1)
Z_A = 0.10
Z_B = 0.30


def _base_opts(**overrides):
    pop_lower, pop_upper, _pop_labels, _pop_fid, sampled, fixed = _pop_bits()
    kwargs = dict(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=False,
        fix_survey=True,
        prior_overrides={sampled: [float(pop_lower[0]), float(pop_upper[0])]},
        fixed_parameter_values=fixed,
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _mid_pop():
    pop_lower, pop_upper, *_ = _pop_bits()
    return 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))


# ---------------------------------------------------------------------------
# K=1 bit-identity
# ---------------------------------------------------------------------------

def test_k1_bit_identical_to_pre_existing_path():
    """make_likelihood built WITHOUT any multitracer opts attributes must
    equal one built with n_catalogs=1 set explicitly: the factory dispatches
    to the mixture builder ONLY for n_catalogs >= 2, so this is by
    construction the SAME code path -- but assert float equality (==) as the
    contract, not isclose."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()

    data = dict(_shared_physics())
    data.update(_bundle(APIX1, Z_A))

    opts_default = _base_opts()  # no n_catalogs attribute at all
    opts_explicit = _base_opts(n_catalogs=1)

    ll_default = make_likelihood(opts_default, data, pop_fid, fixed_parameter_values=fixed)
    ll_explicit = make_likelihood(opts_explicit, data, pop_fid, fixed_parameter_values=fixed)

    coord = jnp.asarray([_mid_pop()])
    val_default = float(ll_default(coord))
    val_explicit = float(ll_explicit(coord))
    assert np.isfinite(val_default)
    assert val_default == val_explicit


# ---------------------------------------------------------------------------
# Duplicated-catalog identity and fcat_2 limits
# ---------------------------------------------------------------------------

def _k1_value(z, fixed, pop_fid):
    data = dict(_shared_physics())
    data.update(_bundle(APIX1, z))
    opts = _base_opts()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    return float(ll(jnp.asarray([_mid_pop()])))


def test_k1_bundle_source_matches_flat_source():
    """A single catalog supplied as ONE bundle (data["catalogs"] = [b]) must
    reproduce the flat-data K=1 likelihood: same parameters (plain decoder, no
    sticks), same value.  This is the unified builder's K=1 coherence contract
    -- a K=1 config scales to K>=2 by appending a bundle, nothing else."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()

    val_flat = _k1_value(Z_A, fixed, pop_fid)

    data = dict(_shared_physics())
    data["apix"] = APIX1  # make_likelihood reads data["apix"] unconditionally
    data["catalogs"] = [_bundle(APIX1, Z_A)]
    opts = _base_opts(n_catalogs=1)
    ll_bundle = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    val_bundle = float(ll_bundle(jnp.asarray([_mid_pop()])))

    assert np.isfinite(val_flat)
    assert abs(val_bundle - val_flat) <= 1e-12


def _k2_likelihood(z_a, z_b, fixed, pop_fid):
    data = dict(_shared_physics())
    data["apix"] = APIX1  # make_likelihood reads data["apix"] unconditionally
    data["catalogs"] = [_bundle(APIX1, z_a), _bundle(APIX1, z_b)]
    opts = _base_opts(n_catalogs=2)
    return make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


def _k2_labels():
    _pop_lower, _pop_upper, _pop_labels, _pop_fid, sampled, fixed = _pop_bits()
    labels, *_ = build_parameter_space(
        "powerlaw+peak", False, True, True,
        prior_overrides={sampled: [-1e9, 1e9]},  # irrelevant to labels, keep wide
        fixed_parameter_values=fixed, universe_model="dark_sirens", n_catalogs=2,
    )
    return labels


def test_k2_duplicated_catalog_equals_k1_at_arbitrary_fcat2():
    """K=2 with catalog B == catalog A must equal K=1(A) at ANY fcat_2 (the
    weights always sum to 1, so w_1 p_A + w_2 p_A = p_A identically) --
    exercised at fcat_2=0.37 per the implementation contract."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()
    labels = _k2_labels()
    assert labels == [_pop_bits()[4], "fcat_2"]

    val_k1 = _k1_value(Z_A, fixed, pop_fid)
    ll_k2 = _k2_likelihood(Z_A, Z_A, fixed, pop_fid)
    coord = jnp.asarray([_mid_pop(), 0.37])
    val_k2 = float(ll_k2(coord))

    assert np.isfinite(val_k1)
    assert abs(val_k2 - val_k1) <= 1e-12


def test_k2_limits_recover_single_catalog_endpoints():
    """K=2 (A, B): fcat_2 -> 0 collapses onto catalog A alone; fcat_2 -> 1
    collapses onto catalog B alone. Tolerance <=1e-9 per the implementation
    contract (measured exactly 0.0 in practice)."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()

    val_a = _k1_value(Z_A, fixed, pop_fid)
    val_b = _k1_value(Z_B, fixed, pop_fid)
    assert val_a != val_b  # catalogs must be genuinely distinguishable

    ll_k2 = _k2_likelihood(Z_A, Z_B, fixed, pop_fid)
    val_at_0 = float(ll_k2(jnp.asarray([_mid_pop(), 0.0])))
    val_at_1 = float(ll_k2(jnp.asarray([_mid_pop(), 1.0])))

    assert abs(val_at_0 - val_a) <= 1e-9
    assert abs(val_at_1 - val_b) <= 1e-9


def test_k2_interior_fcat2_lies_between_single_catalog_endpoints():
    """Selection-additivity check (weaker acceptable form per the
    implementation contract: the mixture's internal per-catalog selection
    weight closures are not exported for direct re-composition via
    logaddexp, so this exercises the same invariant through the endpoints
    already pinned above plus an interior point strictly between them)."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()

    val_a = _k1_value(Z_A, fixed, pop_fid)
    val_b = _k1_value(Z_B, fixed, pop_fid)
    ll_k2 = _k2_likelihood(Z_A, Z_B, fixed, pop_fid)
    val_mid = float(ll_k2(jnp.asarray([_mid_pop(), 0.5])))

    lo, hi = sorted((val_a, val_b))
    assert lo <= val_mid <= hi


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

def test_k2_gradient_finite_and_fcat2_component_nonzero():
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()
    ll_k2 = _k2_likelihood(Z_A, Z_B, fixed, pop_fid)
    coord = jnp.asarray([_mid_pop(), 0.3])

    grad = jax.grad(lambda c: ll_k2(c))(coord)
    grad_np = np.asarray(grad)
    assert np.all(np.isfinite(grad_np))
    # d logL / d fcat_2 must be nonzero for genuinely distinct catalogs.
    assert grad_np[1] != 0.0


# ---------------------------------------------------------------------------
# Different-nside catalogs
# ---------------------------------------------------------------------------

def test_different_nside_catalogs_each_get_own_apix(monkeypatch):
    """Catalog A at nside=8, catalog B at nside=16: each EMCatalog built
    inside _make_mixture_likelihood must carry its OWN apix (its own nside),
    not a shared/borrowed one.  Verified directly by spying on the EMCatalog
    constructor calls inside darksirens.likelihood.factory."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()

    apix8 = hp.nside2pixarea(8)
    apix16 = hp.nside2pixarea(16)
    assert apix8 != apix16

    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_bundle(apix8, Z_A), _bundle(apix16, Z_B)]
    opts = _base_opts(n_catalogs=2)

    captured = []
    orig_em_catalog = factory_mod.EMCatalog

    def _spy(*args, **kwargs):
        obj = orig_em_catalog(*args, **kwargs)
        captured.append(obj)
        return obj

    monkeypatch.setattr(factory_mod, "EMCatalog", _spy)
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    val = float(ll(jnp.asarray([_mid_pop(), 0.3])))

    assert np.isfinite(val)
    # PE and selection EMCatalog for catalog A, then for catalog B (loop order
    # in _make_mixture_likelihood: per bundle, PE then selection).
    assert len(captured) == 4
    assert captured[0].apix == apix8
    assert captured[1].apix == apix8
    assert captured[2].apix == apix16
    assert captured[3].apix == apix16


def test_different_nside_catalogs_differ_from_same_nside_control():
    """Fallback / corroborating check: swapping catalog B's nside (holding
    its z content fixed) changes the mixture likelihood, confirming the
    per-catalog apix actually enters the computation (not just recorded)."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()
    apix8 = hp.nside2pixarea(8)
    apix16 = hp.nside2pixarea(16)

    def _val(apix_a, apix_b):
        data = dict(_shared_physics())
        data["apix"] = APIX1
        data["catalogs"] = [_bundle(apix_a, Z_A), _bundle(apix_b, Z_B)]
        opts = _base_opts(n_catalogs=2)
        ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
        return float(ll(jnp.asarray([_mid_pop(), 0.3])))

    val_diff_nside = _val(apix8, apix16)
    val_same_nside = _val(apix8, apix8)
    assert val_diff_nside != val_same_nside


# ---------------------------------------------------------------------------
# Static guards
# ---------------------------------------------------------------------------

def test_guard_universe_model_must_be_dark_sirens():
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_bundle(APIX1, Z_A), _bundle(APIX1, Z_B)]
    opts = _base_opts(universe_model="spectral_sirens", n_catalogs=2)
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    with pytest.raises(NotImplementedError):
        ll(jnp.asarray([_mid_pop(), 0.3]))


def test_marked_mixture_requires_per_catalog_mark_data():
    """Marks at K>=2 are SUPPORTED via the per-catalog eta blocks
    (tests/test_marks_multitracer.py), but a catalog whose bundle carries no
    mark arrays for its selected marks fails with the clear missing-mark
    error rather than a silent h=1 fallback."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, sampled, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_bundle(APIX1, Z_A), _bundle(APIX1, Z_B)]
    opts = _base_opts(
        n_catalogs=2, mark_model="loglinear", mark_names=("logmstar",),
        mark_names_by_catalog=(("logmstar",), ("logmstar",)),
    )
    labels, lower, upper, *_ = build_parameter_space(
        "powerlaw+peak", False, True, True,
        prior_overrides={sampled: [float(_pop_lower[0]), float(_pop_upper[0])]},
        fixed_parameter_values=fixed, universe_model="dark_sirens", n_catalogs=2,
        mark_model="loglinear", mark_names=("logmstar",),
        mark_names_by_catalog=(("logmstar",), ("logmstar",)),
    )
    coord = jnp.asarray(0.5 * (np.asarray(lower) + np.asarray(upper)))
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    with pytest.raises(ValueError, match="mark"):
        ll(coord)


def test_lss_marginalize_without_members_raises_at_k2():
    """lss_marginalize is SUPPORTED at K>=2 (shared member index), but every
    catalog must carry a Q ensemble: bundles without members raise the clear
    per-catalog error (functional replacement for the old blanket K>=2
    NotImplementedError guard)."""
    _pop_lower, _pop_upper, _pop_labels, pop_fid, _sampled, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_bundle(APIX1, Z_A), _bundle(APIX1, Z_B)]
    opts = _base_opts(n_catalogs=2, lss_marginalize=True,
                      catalog_sky_weighting="conditional")
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    with pytest.raises(ValueError, match="ENSEMBLE on EVERY"):
        ll(jnp.asarray([_mid_pop(), 0.3]))


@pytest.mark.parametrize(
    "key", ["counterpart_pixel", "counterpart_pixels", "wl_params",
            "pixel_stratum_map"])
def test_guard_bundle_path_rejects_operands_it_cannot_carry(key):
    """The bundle EMCatalogs carry no counterpart / stratum inputs and the body
    forwards no WL operands, and at K=1 nothing downstream rejects them (a
    dropped counterpart is SILENT -- prior.py substitutes an arbitrary
    catalogued pixel), so the factory must refuse them at build time."""
    _pl, _pu, _plabels, pop_fid, _sampled, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_bundle(APIX1, Z_A)]
    data[key] = 7 if key == "counterpart_pixel" else jnp.asarray([7])
    opts = _base_opts(n_catalogs=1)
    with pytest.raises(NotImplementedError, match=key):
        make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)


def test_guard_bundle_path_rejects_bright_sirens_by_name():
    """A bright-siren bundle run carrying no counterpart arrays is refused by
    MODEL name -- without it the run would silently host every event on an
    arbitrary catalogued pixel.  (``spectral_sirens_wl`` is covered by the
    wl_params case above: that model cannot be built without wl_params.)"""
    _pl, _pu, _plabels, pop_fid, _sampled, fixed = _pop_bits()
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [_bundle(APIX1, Z_A)]
    opts = _base_opts(universe_model="bright_sirens", n_catalogs=1)
    with pytest.raises(NotImplementedError, match="bright_sirens"):
        make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
