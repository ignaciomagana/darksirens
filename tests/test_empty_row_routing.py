"""Empty-catalog-row sample routing (perf campaign C02 stage 1).

A dark-siren sample whose pixel row holds NO galaxies pays the full per-sample
catalog KDE for a value the evaluator throws away: ``row_empty[pix]`` selects an
exact ``-inf`` for ``log p_cat`` and ``log_Nobs[pix]`` is ``-inf`` too, so the
prior collapses to the missing-galaxy branch alone.  The pixel of a PE sample or
an injection is DATA, so the factory partitions each sample set at BUILD time
(:class:`darksirens.redshift.prior.EmptyRowRouting`) and the evaluator runs the
KDE only on the occupied group.

The contract these tests pin:

* the routed PRIOR VECTOR is bit-identical (``==`` on the float) to the
  unrouted one, sample by sample -- that part IS exact and is asserted as an
  equality;
* the routed LIKELIHOOD is ulp-level: splitting one vmap into two changes how
  XLA fuses and orders the reductions ABOVE the prior evaluator, which on the
  CPU backend moves the total at the last bit on a small minority of
  coordinates (measured <= 4e-14 relative, and 0.0 at every coordinate measured
  on the production H100 build).  The gate is therefore a DENSE H0 scan over
  the campaign's [20, 140] prior with a relative bound and a signed-mean check
  -- dense and signed because a decorrelated per-sample array would show up as
  an H0-correlated TILT that a sparse grid can miss and a max-abs bound cannot
  distinguish from rounding;
* the admissibility predicate refuses the configurations where the routing
  cannot pay, cannot be proven exact, or cannot be consumed (a blocking pin);
* the in-graph verdict drives the whole log-likelihood to ``-inf`` -- never a
  plausible finite number -- for BOTH ways the routing premise can break: a row
  the plan called empty that carries a finite kernel mixture, and a sample the
  plan routed that does not sit on an empty row.  Fail-closed as ``-inf`` and
  not NaN, because a per-sample NaN would be swallowed by the likelihood's own
  isfinite mask.

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
    _empty_row_routing_side_is_consumable,
    empty_row_routing_admissible,
    make_likelihood,
)
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift.prior import EmptyRowRouting

APIX1 = hp.nside2pixarea(1)
POP_MODEL = "powerlaw+peak"
# A DENSE grid across the campaign's [20, 140] prior: the last-bit moves are
# sparse in H0 (1-2 coordinates in 49 on the CPU backend), so a five-point grid
# can miss them entirely -- and missing them is how a float ``==`` gate turns
# into a CI flake on the next backend or BLAS.  49 points at 2.5 spacing.
H0_SCAN = tuple(float(h) for h in np.arange(20.0, 140.001, 2.5))
# ulp-level: the campaign's bar for a change with no coherent sign.
H0_SCAN_REL_TOL = 1e-12
# A coherent tilt with H0 is the failure mode that matters; rounding averages
# out, a decorrelated per-sample array does not.
H0_SCAN_SIGNED_MEAN_TOL = 1e-13

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
def test_routing_is_ulp_level_across_the_H0_prior(
    catalog_sky_weighting, n_catalogs
):
    """The routed and unrouted likelihoods agree to ulp across [20, 140].

    The PRIOR is exact sample by sample (see
    ``test_routed_prior_vector_is_bit_identical_per_sample``), but the total is
    not asserted with ``==``: two vmaps in place of one change how XLA fuses and
    schedules the reductions ABOVE this function, which on the CPU backend moves
    the last bit at a small minority of coordinates.  What must hold, and what
    is checked here, is that the move is rounding and not signal -- bounded at
    ``1e-12`` relative, and with a signed mean two orders below that, so it
    cannot be an H0-correlated tilt of the kind a decorrelated per-sample array
    produces.
    """
    ll_on = _build(catalog_sky_weighting, n_catalogs, routed=True)
    ll_off = _build(catalog_sky_weighting, n_catalogs, routed=False)
    assert ll_on.empty_row_routing != ()  # the routing really is armed
    rel = []
    for h0 in H0_SCAN:
        coord = _coord(h0, n_catalogs)
        v_on = float(ll_on(coord))
        v_off = float(ll_off(coord))
        assert np.isfinite(v_off), (h0, v_off)
        assert np.isfinite(v_on), (h0, v_on)
        r = (v_on - v_off) / abs(v_off)
        assert abs(r) <= H0_SCAN_REL_TOL, (
            catalog_sky_weighting, n_catalogs, h0, v_on, v_off, r)
        rel.append(r)
    signed_mean = float(np.mean(rel))
    assert abs(signed_mean) <= H0_SCAN_SIGNED_MEAN_TOL, (
        catalog_sky_weighting, n_catalogs, signed_mean, rel)


@pytest.mark.parametrize("catalog_sky_weighting", ["field", "conditional"])
def test_routed_prior_vector_is_bit_identical_per_sample(catalog_sky_weighting):
    """Where exactness IS provable, assert it as an equality.

    The routed group's expression is ``_eval_dark_scalar`` with the single
    substitution ``log p_cat = -inf`` -- the value the KDE returns on an empty
    row -- and the inverse gather restores the caller's sample order before the
    vector leaves the evaluator.  So the vector itself, sample by sample and
    slot by slot, is the unrouted vector's bits.  Everything ulp-level in this
    change happens strictly ABOVE this point, in XLA's scheduling of the
    reductions that consume it.
    """
    from darksirens.redshift.prior import (
        eval_redshift_prior_with_state, prepare_redshift_prior_state,
    )

    cosmo, survey, cat = _cosmo(), _survey(), _catalog()
    state = prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat)
    pix = jnp.asarray(_PE_ROWS, dtype=jnp.int32)
    routing = _empty_row_routing_plan(np.asarray(_PE_ROWS), _NG)
    assert routing is not None
    for h0 in (20.0, 45.0, 70.0, 110.0, 140.0):
        cosmo_h = CosmoParams(H0=h0, Om0=0.3, w0=-1.0, wa=0.0)
        st = prepare_redshift_prior_state("dark_sirens", cosmo_h, survey, cat)
        z = jnp.linspace(0.05, 0.25, _PE_ROWS.size)
        kw = dict(catalog_sky_weighting=catalog_sky_weighting)
        v_on = np.asarray(eval_redshift_prior_with_state(
            "dark_sirens", st, z, pix, cosmo_h, survey, cat,
            empty_routing=routing, **kw))
        v_off = np.asarray(eval_redshift_prior_with_state(
            "dark_sirens", st, z, pix, cosmo_h, survey, cat, **kw))
        assert np.array_equal(v_on, v_off), (h0, v_on, v_off)
    del state


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

def _build_with_corrupted_plans(corrupt):
    """Build the K = 1 likelihood with every plan passed through ``corrupt``."""
    pop_fid, fixed = _pop_bits()
    fixed = dict(fixed, Om0=0.3)
    real = factory_mod._empty_row_routing

    def _lying(*a, **k):
        plans = real(*a, **k)
        assert plans != (), "the poison needs a plan to corrupt"
        return tuple(
            tuple(None if r is None else corrupt(r) for r in (pe, sel))
            for pe, sel in plans
        )

    factory_mod._empty_row_routing = _lying
    try:
        return make_likelihood(
            _opts("field", 1), _data(1), pop_fid, fixed_parameter_values=fixed
        )
    finally:
        factory_mod._empty_row_routing = real


def test_a_row_wrongly_called_empty_poisons_the_likelihood():
    """The shortcut is exact only because ``ngals == 0`` implies ``row_empty``.

    Hand the graph a plan that claims an OCCUPIED row is empty and the answer
    must become ``-inf`` -- never a plausible wrong number.  ``-inf`` and not
    NaN, deliberately: see the comment on the assertion.
    """
    ll = _build_with_corrupted_plans(
        lambda r: r._replace(empty_rows=jnp.ones_like(r.empty_rows))
    )
    # Fail-closed as ``-inf`` and not NaN: every per-sample NaN in this
    # likelihood dies at ``where(valid & isfinite(ldw), ldw, -inf)`` long before
    # a reduction, so the only verdict that can reach the caller is one that
    # makes the whole log-likelihood impossible.
    assert float(ll(_coord(70.0, 1))) == -np.inf


def test_a_sample_routed_off_an_empty_row_poisons_the_likelihood():
    """The OTHER half of the premise, and the one a row check cannot see.

    ``empty_rows`` stays honest here -- every row it names really is empty --
    but one sample whose row holds galaxies is moved into the routed group, so
    its ``log p_cat`` is replaced by ``-inf`` where its KDE would have returned
    a finite number.  This is the shape a plan/``pix`` desynchronisation takes
    (the factory derives the plan from the catalog view's
    ``sample_to_unique_idx``, the evaluator consumes ``gw.pixels[:, k]``), and
    the plan is selected by sample-vector LENGTH alone, so nothing upstream
    catches it.  Without the sample-side term in the in-graph verdict this
    returns a finite, plausible, WRONG likelihood.
    """
    def _move_one_occupied_sample_into_the_routed_group(r):
        io = np.asarray(r.idx_occ)
        ie = np.asarray(r.idx_empty)
        assert io.size >= 1
        io_new = io[1:]
        ie_new = np.concatenate([ie, io[:1]]).astype(np.int32)
        order = np.concatenate([io_new, ie_new])
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size, dtype=order.dtype)
        return r._replace(
            idx_occ=jnp.asarray(io_new, dtype=jnp.int32),
            idx_empty=jnp.asarray(ie_new, dtype=jnp.int32),
            inv_order=jnp.asarray(inv, dtype=jnp.int32),
        )

    ll = _build_with_corrupted_plans(
        _move_one_occupied_sample_into_the_routed_group)
    assert float(ll(_coord(70.0, 1))) == -np.inf


def test_the_substituted_log_p_cat_is_exactly_neginf():
    """Pin the ONE line the whole shortcut rests on.

    Two independent statements.  (1) The KDE really does return exactly ``-inf``
    on a row with no galaxies -- the value ``_eval_dark_scalar_empty_row``
    substitutes.  (2) The substituted constant is that ``-inf`` and not merely
    something very negative: on a real state ``log_Nobs`` is ``-inf`` on an
    empty row, which masks ANY finite substitute, so the comparison is made
    against a state whose ``log_Nobs`` is large and FINITE on the empty rows.
    There the two evaluators agree only if the substitution is exactly ``-inf``
    (``-1e30`` would leave ``log_Nobs + log_p_cat`` at ``0.0`` and move the
    numerator by nats).
    """
    from darksirens.redshift.catalog import eval_log_catalog_prior_state
    from darksirens.redshift.prior import (
        _eval_dark_scalar, _eval_dark_scalar_empty_row,
        prepare_redshift_prior_state,
    )

    cosmo, survey, cat = _cosmo(), _survey(), _catalog()
    state = prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat)

    empty_rows = np.flatnonzero(_NG == 0)
    assert empty_rows.size >= 1
    for row in empty_rows:
        for z in (0.05, 0.12, 0.30):
            lp = float(eval_log_catalog_prior_state(
                jnp.asarray(z), jnp.asarray(int(row)), state.kernels, cat))
            assert lp == -np.inf, (row, z, lp)
        # ... and the real state's own ``log_Nobs`` is -inf there too, which is
        # exactly why the check below has to fabricate a finite one.
        assert float(np.asarray(state.log_Nobs)[row]) == -np.inf

    log_Nobs_finite = np.asarray(state.log_Nobs).copy()
    log_Nobs_finite[empty_rows] = 1e30
    state_finite = state._replace(log_Nobs=jnp.asarray(log_Nobs_finite))
    for row in empty_rows:
        for z in (0.05, 0.12, 0.30):
            for weighting in ("field", "conditional"):
                ref = float(_eval_dark_scalar(
                    jnp.asarray(z), jnp.asarray(int(row)), state_finite, cat,
                    weighting))
                got = float(_eval_dark_scalar_empty_row(
                    jnp.asarray(z), jnp.asarray(int(row)), state_finite,
                    weighting))
                assert (ref == got) or (np.isnan(ref) and np.isnan(got)), (
                    row, z, weighting, ref, got)


# ---------------------------------------------------------------------------
# Blocking pins: a side the evaluator would chop up gets no plan at all
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "side, sel_batch_size, pe_event_block, nEvents, expected",
    [
        ("sel", None, None, 259, True),
        ("sel", 131072, None, 259, False),
        ("sel", 8, 32, 259, False),
        ("pe", None, None, 259, True),
        ("pe", None, 32, 259, False),
        ("pe", None, 259, 259, True),      # min(block, nEvents) is the whole set
        ("pe", None, 1024, 259, True),
        ("pe", 131072, None, 259, True),   # a sel pin does not touch the PE side
    ],
)
def test_a_blocked_side_is_not_consumable(
    side, sel_batch_size, pe_event_block, nEvents, expected
):
    assert _empty_row_routing_side_is_consumable(
        side, sel_batch_size, pe_event_block, nEvents) is expected


def test_a_sel_batch_pin_builds_no_selection_plan():
    """Not just "the evaluator falls back": the arrays are never uploaded.

    ``likelihood.empty_row_routing`` is what a reader (and the H0 gate above)
    consults to know whether the routing is live, so a side that cannot be
    consumed must report ``None`` there rather than a plan the evaluator will
    silently discard -- which on the production selection set is ~17 MB of
    device memory nothing would ever read.
    """
    pop_fid, fixed = _pop_bits()
    fixed = dict(fixed, Om0=0.3)
    opts = _opts("field", 1)
    opts.sel_batch_size = 4
    ll = make_likelihood(opts, _data(1), pop_fid, fixed_parameter_values=fixed)
    assert ll.empty_row_routing != ()
    pe, sel = ll.empty_row_routing[0]
    assert sel is None
    assert isinstance(pe, EmptyRowRouting)


def test_a_pe_event_block_pin_builds_no_pe_plan():
    pop_fid, fixed = _pop_bits()
    fixed = dict(fixed, Om0=0.3)
    opts = _opts("field", 1)
    opts.pe_event_block = 1
    ll = make_likelihood(opts, _data(1), pop_fid, fixed_parameter_values=fixed)
    assert ll.empty_row_routing != ()
    pe, sel = ll.empty_row_routing[0]
    assert pe is None
    assert isinstance(sel, EmptyRowRouting)


def test_a_fully_pinned_run_is_unchanged_by_the_routing():
    """Both sides refused: the build is the historical one, bit for bit."""
    pop_fid, fixed = _pop_bits()
    fixed = dict(fixed, Om0=0.3)

    def _opts_pinned():
        o = _opts("field", 1)
        o.sel_batch_size = 4
        o.pe_event_block = 1
        return o

    ll_on = make_likelihood(
        _opts_pinned(), _data(1), pop_fid, fixed_parameter_values=fixed)
    assert ll_on.empty_row_routing == ()
    real = factory_mod._empty_row_routing
    factory_mod._empty_row_routing = lambda *a, **k: ()
    try:
        ll_off = make_likelihood(
            _opts_pinned(), _data(1), pop_fid, fixed_parameter_values=fixed)
    finally:
        factory_mod._empty_row_routing = real
    for h0 in (20.0, 70.0, 140.0):
        assert float(ll_on(_coord(h0, 1))) == float(ll_off(_coord(h0, 1)))
