"""Empty-catalog-row sample routing (perf campaign C02 stage 1).

A dark-siren sample whose pixel row holds NO galaxies pays the full per-sample
catalog KDE for a value the evaluator throws away: ``row_empty[pix]`` selects an
exact ``-inf`` for ``log p_cat`` and ``log_Nobs[pix]`` is ``-inf`` too, so the
prior collapses to the missing-galaxy branch alone.  The pixel of a PE sample or
an injection is DATA, so the factory partitions each sample set at BUILD time
(:class:`darksirens.redshift.prior.EmptyRowRouting`) and the evaluator runs the
KDE only on the occupied group.

The contract these tests pin:

* the routed likelihood is BIT-IDENTICAL (``==`` on the float, not a tolerance)
  to the unrouted one, at several H0 spanning the campaign's [20, 140] prior,
  under BOTH sky-weighting conventions and at K = 1 and K = 2 -- an H0 SCAN and
  not a single fiducial coordinate, because a decorrelated per-sample array
  shows up as an H0-correlated TILT that a single coordinate can sit on top of;
* the admissibility predicate refuses the configurations where the routing
  cannot pay or cannot be proven exact;
* the in-graph verdict drives the whole log-likelihood to ``-inf`` if a row the
  plan called empty turns out to carry a finite kernel mixture (fail-closed: a
  per-sample NaN would be swallowed by the likelihood's own isfinite mask).

Conventions follow ``tests/test_catalog_sky_weighting.py`` (compact in-memory
bundles, x64 via conftest).
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
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood import factory as factory_mod
from darksirens.likelihood.factory import (
    _empty_row_routing_plan,
    empty_row_routing_admissible,
    make_likelihood,
)
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift.prior import EmptyRowRouting

APIX1 = hp.nside2pixarea(1)
POP_MODEL = "powerlaw+peak"
H0_SCAN = (20.0, 45.0, 70.0, 100.0, 140.0)

# 4 compact rows: two occupied, two EMPTY (ngals == 0).  Padding follows the
# loader contract -- real galaxies occupy the row prefix [0, ngals), the tail is
# z = 100, dz = 1, w = 0.
N_ROWS = 4
N_MAX = 2
_ZG = np.full((N_ROWS, N_MAX), 100.0)
_DZ = np.full((N_ROWS, N_MAX), 1.0)
_WG = np.zeros((N_ROWS, N_MAX))
_NG = np.zeros(N_ROWS, dtype=np.int32)
for _r, _zs in ((0, (0.08, 0.14)), (1, (0.11, 0.19))):
    _ZG[_r, : len(_zs)] = _zs
    _DZ[_r, : len(_zs)] = 0.02
    _WG[_r, : len(_zs)] = 1.0
    _NG[_r] = len(_zs)

NSAMP = 8
N_EVENTS = 2
N_PE = N_EVENTS * NSAMP
N_SEL = 24
# Half of each sample set lands on an empty row (production sits at 0.39-0.42).
_PE_ROWS = np.tile(np.array([0, 2, 1, 3, 0, 3, 1, 2], dtype=np.int32), N_EVENTS)
_SEL_ROWS = np.tile(np.array([0, 1, 2, 3], dtype=np.int32), N_SEL // 4)


def _cosmo():
    return CosmoParams(H0=70.0, Om0=0.3, w0=-1.0, wa=0.0)


def _survey():
    return SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                        alpha_miss=1.0)


def _catalog():
    """The same 4-row catalog as an EMCatalog, for the evaluator-level tests."""
    return EMCatalog(
        apix=APIX1,
        zgals=jnp.asarray(_ZG), dzgals=jnp.asarray(_DZ), wgals=jnp.asarray(_WG),
        ngals=jnp.asarray(_NG),
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        dN_obs_kde=None, pixel_to_cache_idx=None,
    )


def _pop_bits():
    pop_fid = get_fixed_population_params(POP_MODEL)
    _lo, _hi, pop_labels, _a, _b = pop_model_prior_parser(POP_MODEL)
    fixed = {lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels)}
    return pop_fid, fixed


def _bundle():
    """Compact catalog bundle with empty rows, plus the field-normalization
    inputs the field convention reads."""
    b = dict(
        apix=APIX1,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        zgals_pe=_ZG.copy(), dzgals_pe=_DZ.copy(), wgals_pe=_WG.copy(),
        ngals_pe=_NG.copy(),
        unique_pixels_pe=np.arange(N_ROWS, dtype=np.int32),
        sample_to_unique_pe=_PE_ROWS.copy(),
        zgals_sel=_ZG.copy(), dzgals_sel=_DZ.copy(), wgals_sel=_WG.copy(),
        ngals_sel=_NG.copy(),
        unique_pixels_sel=np.arange(N_ROWS, dtype=np.int32),
        sample_to_unique_sel=_SEL_ROWS.copy(),
    )
    fobs, _ne, nobs, _occ = build_field_normalization_inputs(
        jnp.asarray(_ZG), None, jnp.asarray(_NG)
    )
    b["field_dN_obs_s"] = fobs
    b["field_n_empty"] = float(hp.nside2npix(1) - 2)
    b["field_N_obs_total"] = float(nobs)
    return b


def _data(n_catalogs=1):
    d = dict(
        nEvents=N_EVENTS, nsamp=NSAMP, Ndraw=float(N_SEL), apix=APIX1,
        m1det=jnp.asarray(np.linspace(30.0, 42.0, N_PE)),
        m2det=jnp.asarray(0.8 * np.linspace(30.0, 42.0, N_PE)),
        dL=jnp.asarray(np.linspace(300.0, 900.0, N_PE)),
        chieff=jnp.zeros(N_PE), p_pe=jnp.ones(N_PE),
        m1detsels=jnp.asarray(np.linspace(30.0, 42.0, N_SEL)),
        m2detsels=jnp.asarray(0.8 * np.linspace(30.0, 42.0, N_SEL)),
        dLsels=jnp.asarray(np.linspace(300.0, 900.0, N_SEL)),
        chieffsels=jnp.zeros(N_SEL), p_draw=jnp.ones(N_SEL),
    )
    if n_catalogs == 1:
        d.update(_bundle())
    else:
        d["catalogs"] = [_bundle() for _ in range(n_catalogs)]
    return d


def _opts(catalog_sky_weighting, n_catalogs=1):
    return SimpleNamespace(
        pop_model=POP_MODEL, universe_model="dark_sirens", sel_batch_size=None,
        fix_cosmology=False, fix_population=True, fix_survey=True, fix_de=True,
        fixed_parameter_values={"Om0": 0.3},
        prior_overrides={"H0": [20.0, 140.0]},
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
        n_catalogs=n_catalogs, catalog_sky_weighting=catalog_sky_weighting,
    )


def _build(catalog_sky_weighting, n_catalogs, routed):
    """Likelihood with the routing ON (``routed``) or forced OFF."""
    pop_fid, fixed = _pop_bits()
    fixed = dict(fixed, Om0=0.3)
    opts = _opts(catalog_sky_weighting, n_catalogs)
    real = factory_mod._empty_row_routing
    if not routed:
        factory_mod._empty_row_routing = lambda *a, **k: ()
    try:
        return make_likelihood(
            opts, _data(n_catalogs), pop_fid, fixed_parameter_values=fixed
        )
    finally:
        factory_mod._empty_row_routing = real


def _coord(h0, n_catalogs):
    return jnp.asarray([h0] if n_catalogs == 1 else [h0, 0.37])


# ---------------------------------------------------------------------------
# The gate: bit-identity across the H0 prior, both conventions, K = 1 and K = 2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("catalog_sky_weighting", ["field", "conditional"])
@pytest.mark.parametrize("n_catalogs", [1, 2])
def test_routing_is_bit_identical_across_the_H0_prior(
    catalog_sky_weighting, n_catalogs
):
    """max |dlogL| == 0.0 EXACTLY at 5 H0 spanning [20, 140].

    Not a tolerance: the routed group's expression is ``_eval_dark_scalar`` with
    the single substitution ``log p_cat = -inf``, which is the value the KDE
    returns on an empty row, and the inverse gather restores the caller's sample
    order before any reduction -- so every downstream sum is over the same
    values in the same slots.
    """
    ll_on = _build(catalog_sky_weighting, n_catalogs, routed=True)
    ll_off = _build(catalog_sky_weighting, n_catalogs, routed=False)
    assert ll_on.empty_row_routing != ()  # the routing really is armed
    for h0 in H0_SCAN:
        coord = _coord(h0, n_catalogs)
        v_on = float(ll_on(coord))
        v_off = float(ll_off(coord))
        assert np.isfinite(v_off), (h0, v_off)
        assert v_on == v_off, (catalog_sky_weighting, n_catalogs, h0, v_on, v_off)


def test_routing_partitions_every_sample_exactly_once():
    """The plan is a permutation: the two groups are disjoint, cover the set,
    and ``inv_order`` undoes ``concatenate([idx_occ, idx_empty])``."""
    ll = _build("field", 1, routed=True)
    plans = ll.empty_row_routing
    assert len(plans) == 1
    # The injections are pixel-sorted by ``_injection_pixel_order`` before the
    # plan is built, so the SEL side's row labels are the SORTED ones.
    for side, n_total, rows in ((0, N_PE, _PE_ROWS),
                                (1, N_SEL, np.sort(_SEL_ROWS, kind="stable"))):
        r = plans[0][side]
        assert isinstance(r, EmptyRowRouting)
        io = np.asarray(r.idx_occ)
        ie = np.asarray(r.idx_empty)
        inv = np.asarray(r.inv_order)
        assert io.size + ie.size == n_total
        np.testing.assert_array_equal(np.sort(np.concatenate([io, ie])),
                                      np.arange(n_total))
        order = np.concatenate([io, ie])
        np.testing.assert_array_equal(order[inv], np.arange(n_total))
        # The groups say what they mean, and each keeps its original order.
        assert np.all(_NG[rows[io]] > 0)
        assert np.all(_NG[rows[ie]] == 0)
        np.testing.assert_array_equal(io, np.sort(io))
        np.testing.assert_array_equal(ie, np.sort(ie))
        np.testing.assert_array_equal(np.asarray(r.empty_rows), _NG == 0)


def test_ngals_zero_implies_row_empty_on_every_kernel_builder():
    """The routing predicate's ONE structural premise.

    ``ngals == 0`` makes ``_row_real_mask`` (``arange < ngals``) all-False, so
    every slot's ``log_kw`` is non-finite and
    ``row_empty = ~any(isfinite(log_kw))`` is True -- which is what supplies the
    exact ``-inf`` the shortcut substitutes.  Pinned here on the builders the
    dark-siren prior actually uses, plain and marked, because the shortcut is
    exact for no other reason.
    """
    from darksirens.redshift.catalog import (
        catalog_kernel_state, marked_catalog_kernel_state,
    )

    cosmo, survey = _cosmo(), _survey()
    cat = _catalog()
    plain = catalog_kernel_state(cosmo, survey, cat)
    assert np.all(np.asarray(plain.row_empty)[_NG == 0])
    marked, _ = marked_catalog_kernel_state(
        cosmo, survey, cat, jnp.zeros_like(jnp.asarray(_ZG))
    )
    assert np.all(np.asarray(marked.row_empty)[_NG == 0])


def test_routed_evaluator_has_finite_gradients():
    """The empty group must not poison d log p / dz: its ``-inf`` catalog term
    is exactly the all--inf reduction that used to return NaN cotangents."""
    from darksirens.redshift.prior import (
        eval_redshift_prior_with_state, prepare_redshift_prior_state,
    )

    cosmo, survey, cat = _cosmo(), _survey(), _catalog()
    state = prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat)
    pix = jnp.asarray(_PE_ROWS[:8], dtype=jnp.int32)
    routing = _empty_row_routing_plan(np.asarray(_PE_ROWS[:8]), _NG)
    assert routing is not None

    def _total(z):
        return jnp.sum(
            jnp.where(
                jnp.isfinite(
                    eval_redshift_prior_with_state(
                        "dark_sirens", state, z, pix, cosmo, survey, cat,
                        empty_routing=routing,
                    )
                ),
                eval_redshift_prior_with_state(
                    "dark_sirens", state, z, pix, cosmo, survey, cat,
                    empty_routing=routing,
                ),
                0.0,
            )
        )

    z = jnp.full(8, 0.12)
    assert np.all(np.isfinite(np.asarray(jax.grad(_total)(z))))


# ---------------------------------------------------------------------------
# Admissibility / plan construction
# ---------------------------------------------------------------------------

def test_admissible_only_for_the_live_incomplete_catalog_prior():
    assert empty_row_routing_admissible("dark_sirens", None)
    # A frozen prior has already paid for every sample at build time.
    assert not empty_row_routing_admissible("dark_sirens", object())
    for model in ("spectral_sirens", "bright_sirens", "dark_sirens_complete",
                  "spectral_sirens_wl"):
        assert not empty_row_routing_admissible(model, None)


@pytest.mark.parametrize(
    "pix, ngals, why",
    [
        (np.zeros(100, dtype=np.int32), None, "no ngals: row prefix unproven"),
        (np.zeros(100, dtype=np.int32), np.array([2, 0]), "no empty-row sample"),
        (np.ones(100, dtype=np.int32), np.array([2, 0]), "no occupied sample"),
        (np.array([1] + [0] * 99, dtype=np.int32), np.array([2, 0]),
         "empty group below the 10% floor"),
        (np.zeros(100, dtype=np.int32), np.array([2, 3]), "no empty row at all"),
        (np.zeros(0, dtype=np.int32), np.array([2, 0]), "no samples"),
    ],
)
def test_plan_refuses_the_configurations_it_cannot_pay_for(pix, ngals, why):
    assert _empty_row_routing_plan(pix, ngals) is None, why


def test_plan_is_built_when_the_empty_group_is_worth_routing():
    pix = np.array([0, 1] * 50, dtype=np.int32)
    r = _empty_row_routing_plan(pix, np.array([2, 0]))
    assert isinstance(r, EmptyRowRouting)
    assert np.asarray(r.idx_occ).size == 50
    assert np.asarray(r.idx_empty).size == 50


# ---------------------------------------------------------------------------
# The in-graph verdict
# ---------------------------------------------------------------------------

def test_a_row_wrongly_called_empty_poisons_the_likelihood():
    """The shortcut is exact only because ``ngals == 0`` implies
    ``row_empty``.  Hand the graph a plan that claims an OCCUPIED row is empty
    and the answer must become NaN -- never a plausible wrong number."""
    pop_fid, fixed = _pop_bits()
    fixed = dict(fixed, Om0=0.3)
    real = factory_mod._empty_row_routing

    def _lying(universe_model, frozen_prior, cats_pe, cats_sel):
        plans = real(universe_model, frozen_prior, cats_pe, cats_sel)
        out = []
        for pe, sel in plans:
            out.append(tuple(
                None if r is None
                else r._replace(empty_rows=jnp.ones_like(r.empty_rows))
                for r in (pe, sel)
            ))
        return tuple(out)

    factory_mod._empty_row_routing = _lying
    try:
        ll = make_likelihood(
            _opts("field", 1), _data(1), pop_fid, fixed_parameter_values=fixed
        )
    finally:
        factory_mod._empty_row_routing = real
    # Fail-closed as ``-inf`` and not NaN: every per-sample NaN in this
    # likelihood dies at ``where(valid & isfinite(ldw), ldw, -inf)`` long before
    # a reduction, so the only verdict that can reach the caller is one that
    # makes the whole log-likelihood impossible.
    assert float(ll(_coord(70.0, 1))) == -np.inf
