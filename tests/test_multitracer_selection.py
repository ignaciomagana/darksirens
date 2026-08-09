"""K>=2 mixture x c_mode="selection": the end-to-end likelihood fixture.

The per-catalog selection machinery (#344) was pinned at the parameter-space
and CLI-resolver level (tests/test_percatalog_selection.py) but never through
an assembled likelihood -- the "no K=2 likelihood-eval fixture" gap.  These
tests close it on the tiny in-memory two-catalog fixture (the
test_multitracer_likelihood.py / test_marks_c_mode.py template):

  * a K=2 c_mode="selection" likelihood assembles and evaluates finite, with
    the mixture weight genuinely live;
  * TWO IDENTICAL catalogs at identical theta collapse to the K=1 value
    exactly, for ANY mixture weight (a mixture of identical components is the
    component -- the strongest cheap end-to-end identity);
  * catalog k's theta reaches catalog k's completeness and no one else's:
    ``M0hat_c2`` moves the likelihood, and swapping (catalog, theta, weight)
    simultaneously leaves it invariant -- a crossed routing would break the
    swap while keeping every single-catalog limit intact;
  * the per-catalog K(z) template (``opts.selection_kcorr_by_catalog``)
    reaches its own catalog's curve through the full decoder->likelihood
    chain.

Fixture constraints inherited from test_marks_c_mode.py's selection cell:
``n_sel = 64`` (at 8 injections the parametric budget trips the selection-term
variance guard to exactly -inf) and ``m_lim = 20.0`` pinned through
``fixed_parameter_values`` (at the fiducial 24.0 the survey is near-complete
over the injection redshifts).
"""
from types import SimpleNamespace

import healpy as hp
import jax.numpy as jnp
import numpy as np

from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.inference.parameters import build_parameter_decoder
from darksirens.likelihood.factory import make_likelihood
from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_field_normalization_inputs

N_PIX = 12
APIX = hp.nside2pixarea(1)
NG = len(zgrid)

# Distinct thetas for the routing tests (both inside the registry box).
THETA_1 = -20.9
THETA_2 = -19.9


def _full_sky(occ, maxg=3):
    """(npix, maxg) full-sky arrays from an {pixel: [z, ...]} occupancy map."""
    zgals = np.zeros((N_PIX, maxg))
    wgals = np.zeros((N_PIX, maxg))
    ngals = np.zeros(N_PIX, dtype=np.int32)
    for p, zs in occ.items():
        for j, z in enumerate(zs):
            zgals[p, j] = z
            wgals[p, j] = 1.0
        ngals[p] = len(zs)
    return zgals, np.full((N_PIX, maxg), 0.02), wgals, ngals


#: Two different skies so the K=2 catalogs are genuinely distinct tracers.
OCC_A = {1: [0.10], 3: [0.20, 0.25], 4: [0.15], 7: [0.30, 0.32, 0.28]}
OCC_B = {2: [0.12], 3: [0.22], 5: [0.18, 0.21], 7: [0.29, 0.31]}


def _selection_bundle(occ, sel_pixels, nsamp=2, n_sel=64):
    """Loader-shaped compact-view bundle with the FIELD normalization inputs
    c_mode="selection" requires (catalog_sky_weighting="field" is REQUIRED for
    the parametric budget's global normalizer)."""
    zgals, dzgals, wgals, ngals = _full_sky(occ)
    pe_pix = np.array([7, 7], dtype=np.int32)[:nsamp]
    sel_pix = np.resize(np.asarray(sel_pixels, dtype=np.int32), n_sel)
    up_pe, s2u_pe = np.unique(pe_pix, return_inverse=True)
    up_se, s2u_se = np.unique(sel_pix, return_inverse=True)
    field = build_field_normalization_inputs(
        jnp.asarray(zgals), jnp.asarray(wgals), jnp.asarray(ngals))
    return dict(
        nside=1, apix=APIX, n_pix_catalog=N_PIX,
        delta_g_pix_z=jnp.zeros((1, NG)),
        zgals_pe=zgals[up_pe], dzgals_pe=dzgals[up_pe], wgals_pe=wgals[up_pe],
        ngals_pe=ngals[up_pe], unique_pixels_pe=up_pe.astype(np.int32),
        sample_to_unique_pe=s2u_pe.astype(np.int32),
        zgals_sel=zgals[up_se], dzgals_sel=dzgals[up_se],
        wgals_sel=wgals[up_se], ngals_sel=ngals[up_se],
        unique_pixels_sel=up_se.astype(np.int32),
        sample_to_unique_sel=s2u_se.astype(np.int32),
        field_dN_obs_s=field.dN_obs_s, field_n_empty=field.n_empty,
        field_N_obs_total=field.N_obs_total,
        field_occupied_pixels=field.occupied_pixels,
    )


def _bundle_a(**kw):
    return _selection_bundle(OCC_A, [1, 3, 4, 7], **kw)


def _bundle_b(**kw):
    return _selection_bundle(OCC_B, [2, 3, 5, 7], **kw)


def _shared_physics(nsamp=2, n_sel=64):
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
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser(
        "powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    fixed = {lbl: float(pop_fid[i])
             for i, lbl in enumerate(pop_labels) if lbl != sampled}
    overrides = {sampled: [float(pop_lower[0]), float(pop_upper[0])]}
    mid = 0.5 * (float(pop_lower[0]) + float(pop_upper[0]))
    return pop_fid, overrides, fixed, mid


def _sel_opts(n_catalogs, overrides, fixed, **kw):
    kwargs = dict(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, fix_cosmology=True, fix_population=False,
        fix_survey=True, prior_overrides=overrides,
        fixed_parameter_values=fixed,
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
        catalog_sky_weighting="field", n_catalogs=n_catalogs,
        c_mode="selection", use_LSS=False,
    )
    kwargs.update(kw)
    return SimpleNamespace(**kwargs)


def _k2_value(bundles, extra_fixed, coord_tail, **opts_kw):
    """Assemble a K=len(bundles) selection likelihood and evaluate it at
    [mid_pop, *coord_tail] with ``extra_fixed`` folded into the pins."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    fixed = dict(fixed, **extra_fixed)
    data = dict(_shared_physics())
    data["apix"] = APIX
    data["catalogs"] = list(bundles)
    opts = _sel_opts(len(bundles), overrides, fixed, **opts_kw)
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    return float(ll(jnp.asarray([mid, *coord_tail])))


#: Both catalogs' m_lim pinned to the guard-friendly datum (see module doc).
M_LIM_PINS = {"m_lim": 20.0, "m_lim_c2": 20.0}


def test_k2_selection_likelihood_finite_and_weight_live():
    """Smoke: K=2 x c_mode="selection" assembles, evaluates finite at several
    mixture weights, and fcat_2 is genuinely live (distinct catalogs =>
    distinct component priors => the weight must matter)."""
    vals = [
        _k2_value([_bundle_a(), _bundle_b()], M_LIM_PINS, [w])
        for w in (0.2, 0.5, 0.8)
    ]
    assert np.all(np.isfinite(vals))
    assert len(set(vals)) == 3


def test_k2_identical_catalogs_collapse_to_k1_for_any_weight():
    """TWO IDENTICAL catalogs at IDENTICAL theta are the K=1 likelihood
    exactly, whatever the mixture weight: the component redshift priors are
    equal, so the logsumexp over weights is the single-component value.  This
    is the end-to-end identity that fails loudly if ANY per-catalog selection
    ingredient (theta routing, field normalizer, empty-pixel budget) diverges
    between the mixture path and the single-catalog path."""
    pop_fid, overrides, fixed, mid = _pop_bits()
    fixed_k1 = dict(fixed, m_lim=20.0)
    data = dict(_shared_physics())
    data["apix"] = APIX
    data["catalogs"] = [_bundle_a()]
    opts = _sel_opts(1, overrides, fixed_k1)
    ll_k1 = make_likelihood(opts, data, pop_fid,
                            fixed_parameter_values=fixed_k1)
    val_k1 = float(ll_k1(jnp.asarray([mid])))
    assert np.isfinite(val_k1)

    for w in (0.25, 0.5, 0.9):
        val_k2 = _k2_value([_bundle_a(), _bundle_a()], M_LIM_PINS, [w])
        assert abs(val_k2 - val_k1) <= 5e-9, (w, val_k2, val_k1)


def test_percatalog_theta_reaches_its_own_catalog_only():
    """``M0hat_c2`` moves the K=2 likelihood (catalog 2's curve consumes its
    OWN theta), and the simultaneous swap (catalogs, thetas, weight ->
    complement) is an exact invariance -- crossed routing (catalog 2's budget
    formed from catalog 1's theta) breaks the swap identity while every
    diagonal case still looks plausible."""
    base = _k2_value(
        [_bundle_a(), _bundle_b()],
        dict(M_LIM_PINS, M0hat=THETA_1, M0hat_c2=THETA_1), [0.3])
    moved = _k2_value(
        [_bundle_a(), _bundle_b()],
        dict(M_LIM_PINS, M0hat=THETA_1, M0hat_c2=THETA_2), [0.3])
    assert np.isfinite(base) and np.isfinite(moved)
    assert base != moved            # theta_c2 is live

    # Swap catalogs AND their thetas AND the weight: identical mixture.
    swapped = _k2_value(
        [_bundle_b(), _bundle_a()],
        dict(M_LIM_PINS, M0hat=THETA_2, M0hat_c2=THETA_1), [0.7])
    assert abs(moved - swapped) <= 5e-9, (moved, swapped)


def test_percatalog_kcorr_reaches_its_own_catalog_only():
    """The structural K(z) template is PER CATALOG: a template on catalog 2
    changes the likelihood, and putting the SAME template on catalog 1 instead
    gives a DIFFERENT value (the two catalogs are distinct tracers), pinning
    that each template lands on its own catalog's curve."""
    thetas = dict(M_LIM_PINS, M0hat=THETA_1, M0hat_c2=THETA_2)
    plain = _k2_value([_bundle_a(), _bundle_b()], thetas, [0.3],
                      selection_kcorr_by_catalog=[None, None])
    k_on_2 = _k2_value([_bundle_a(), _bundle_b()], thetas, [0.3],
                       selection_kcorr_by_catalog=[None, (0.8,)])
    k_on_1 = _k2_value([_bundle_a(), _bundle_b()], thetas, [0.3],
                       selection_kcorr_by_catalog=[(0.8,), None])
    assert np.all(np.isfinite([plain, k_on_2, k_on_1]))
    assert plain != k_on_2          # catalog 2's template is live
    assert plain != k_on_1          # catalog 1's template is live
    assert k_on_1 != k_on_2         # ... and they are not the same knob


def test_sampled_percatalog_theta_block_end_to_end():
    """The PRODUCTION configuration: fix_survey=False with a per-catalog
    magnitude-fit prior anchoring every sampled selection label
    (M0hat/sigma_M and M0hat_c2/sigma_M_c2 -- the all-or-nothing rule), the
    whole survey block SAMPLED, evaluated through the assembled K=2
    likelihood.  Pins that the suffixed theta coordinates exist in the
    decoder's space, route to their own catalog (each is live), and the
    evaluation is finite -- the config a K=2 real-catalog run samples."""
    pop_fid, _overrides, _fixed, _mid = _pop_bits()
    fixed = dict(M_LIM_PINS)        # population fixed whole via fix_population
    selection_prior = {
        "M0hat": (THETA_1, 0.02), "sigma_M": (0.9, 0.015),
        "M0hat_c2": (THETA_2, 0.02), "sigma_M_c2": (0.9, 0.015),
    }
    opts = _sel_opts(2, None, fixed, fix_population=True, fix_survey=False,
                     selection_prior=selection_prior)

    dec = build_parameter_decoder(opts, pop_fid, fixed_parameter_values=fixed)
    labels = list(dec.sampled_labels)
    for lbl in ("M0hat", "sigma_M", "M0hat_c2", "sigma_M_c2", "fcat_2"):
        assert lbl in labels, (lbl, labels)

    #: mid-box values by BASE name (suffixed labels share their base default),
    #: thetas at the fit centers.
    defaults = {"log10n0": -2.5, "delta": 0.3, "sigma_kde": 0.01,
                "M0hat": THETA_1, "sigma_M": 0.9, "fcat_2": 0.3}
    defaults_c2 = {"M0hat": THETA_2}

    def _coord(**over):
        vals = []
        for lbl in labels:
            base = lbl[:-3] if lbl.endswith("_c2") else lbl
            v = over.get(lbl)
            if v is None:
                v = (defaults_c2.get(base, defaults[base])
                     if lbl.endswith("_c2") else defaults[base])
            vals.append(float(v))
        return jnp.asarray(vals)

    data = dict(_shared_physics())
    data["apix"] = APIX
    data["catalogs"] = [_bundle_a(), _bundle_b()]
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)

    v0 = float(ll(_coord()))
    assert np.isfinite(v0)
    # Every sampled selection coordinate is live through the likelihood.
    for lbl, dv in (("M0hat", 0.5), ("sigma_M", 0.2),
                    ("M0hat_c2", 0.5), ("sigma_M_c2", 0.2)):
        base = lbl[:-3] if lbl.endswith("_c2") else lbl
        v = float(ll(_coord(**{
            lbl: (defaults_c2.get(base, defaults[base]) + dv)})))
        assert np.isfinite(v) and v != v0, (lbl, v, v0)


def test_two_real_fit_jsons_through_resolver_to_likelihood(tmp_path):
    """The FULL production chain from disk: two darksirens-selection-fit-1.0
    JSONs (one per catalog, different thetas, catalog 2 with a K(z) template)
    -> ``_resolve_selection_fits`` (per-catalog records, anchored suffixed
    priors, m_lim pins, structural K templates) -> the resolver's own
    projections fed unmodified into the assembled K=2 likelihood, evaluated
    finite with every anchored theta coordinate live.  This is exactly the
    path a two-catalog real-catalog run takes from --selection_fit A,B."""
    import json

    from darksirens.cli import inference as cli
    from darksirens.redshift.selection import SelectionFit

    def _write(path, m_lim, M0hat, sigma_M, kcorr=()):
        fit = SelectionFit(family="gaussian", m_lim=m_lim, M0hat=M0hat,
                           sigma_M=sigma_M, cov=np.eye(2) * 4e-4, n_gal=1000,
                           k_corr_coeffs=tuple(kcorr))
        path.write_text(json.dumps({
            "format_version": "darksirens-selection-fit-1.0",
            "strata": [fit.to_jsonable()]}))
        return str(path)

    p1 = _write(tmp_path / "fitA.json", 20.0, THETA_1, 0.90)
    p2 = _write(tmp_path / "fitB.json", 20.0, THETA_2, 0.80, kcorr=(0.8,))

    pop_fid, _overrides, _fixed, _mid = _pop_bits()
    fixed = {}
    ropts = SimpleNamespace(n_catalogs=2, c_mode="selection",
                            selection_fit_paths=[p1, p2], stratum_map=None)
    cli._resolve_selection_fits(ropts, {}, fixed)

    # Resolver projections: anchored suffixed priors, pins, K(z) routing.
    assert ropts.selection_family == "gaussian"
    assert set(ropts.selection_prior) == {
        "M0hat", "sigma_M", "M0hat_c2", "sigma_M_c2"}
    assert ropts.selection_prior["M0hat"][0] == THETA_1
    assert ropts.selection_prior["M0hat_c2"][0] == THETA_2
    assert fixed == {"m_lim": 20.0, "m_lim_c2": 20.0}
    assert ropts.selection_kcorr_by_catalog == [None, (0.8,)]
    assert [f["theta"]["M0hat"] for f in ropts.selection_fits] == [
        THETA_1, THETA_2]

    # Feed the projections, UNMODIFIED, into the assembled likelihood.
    opts = _sel_opts(2, None, fixed, fix_population=True, fix_survey=False,
                     selection_prior=ropts.selection_prior,
                     selection_kcorr_by_catalog=ropts.selection_kcorr_by_catalog,
                     selection_strata_by_catalog=ropts.selection_strata_by_catalog)
    dec = build_parameter_decoder(opts, pop_fid, fixed_parameter_values=fixed)
    labels = list(dec.sampled_labels)

    defaults = {"log10n0": -2.5, "delta": 0.3, "sigma_kde": 0.01,
                "M0hat": THETA_1, "sigma_M": 0.9, "fcat_2": 0.3}
    defaults_c2 = {"M0hat": THETA_2, "sigma_M": 0.8}

    def _coord(**over):
        vals = []
        for lbl in labels:
            base = lbl[:-3] if lbl.endswith("_c2") else lbl
            v = over.get(lbl)
            if v is None:
                v = (defaults_c2.get(base, defaults[base])
                     if lbl.endswith("_c2") else defaults[base])
            vals.append(float(v))
        return jnp.asarray(vals)

    data = dict(_shared_physics())
    data["apix"] = APIX
    data["catalogs"] = [_bundle_a(), _bundle_b()]
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)

    v0 = float(ll(_coord()))
    assert np.isfinite(v0)
    for lbl in ("M0hat", "sigma_M", "M0hat_c2", "sigma_M_c2"):
        base = lbl[:-3] if lbl.endswith("_c2") else lbl
        center = (defaults_c2.get(base, defaults[base])
                  if lbl.endswith("_c2") else defaults[base])
        v = float(ll(_coord(**{lbl: center + 0.3})))
        assert np.isfinite(v) and v != v0, (lbl, v, v0)
