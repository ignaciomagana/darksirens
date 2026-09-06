"""Static row-WIDTH tiers for the catalog KDE (perf campaign C02, stage 2).

What the change is
------------------
The compact catalog is a rectangle padded to the GLOBAL densest row -- 1719
slots on the production DESI nside-64 view -- while the sample-weighted mean
real row holds 436 (PE) / 457 (selection) galaxies.  Every sample therefore
gathers ~2.7x more slots than its own pixel contains, and the extra slots are
padding whose ``log_kw_eff = -1e30`` contributes an exact ``0.0`` to the sum.

A sample's pixel is DATA, so its row length is known before the run starts.
The build partitions the OCCUPIED samples (the empty ones are already routed
past the KDE by :class:`EmptyRowRouting`) into a static ladder of width tiers by
``ngals[pix]`` and evaluates each tier over the column PREFIX ``a[pix, :cap]``
of the unchanged resident arrays.  No row is relabelled, no table is compacted,
no other per-sample array moves.

What these tests pin
--------------------
* the accuracy CLASS: ulp-level, with no H0 trend -- the campaign's real bar.
  Same galaxies, same arithmetic, a shorter reduction, so only the association
  changes; the gate is a 49-point H0 scan across the sampled prior [20, 140]
  with a relative bound, a signed-mean bound, and a bound on the SLOPE of the
  residual against H0.
* the partition: every sample in exactly one group, and every sample's cap at
  least its row's galaxy count.
* the two halves of the exactness promise, each where it is enforced --
  live slots are the row prefix (build time, on the array the graph reads), and
  ``ngals[pix] <= cap`` per tier (in graph, against THIS ``pix`` vector).
* the refusals, which must be LOUD: no ``ngals``, a live slot past ``ngals``,
  a 1-D view, an out-of-range cap, and above all an ARMED KDE WINDOW -- the
  windowed branch is a different (truncating) estimator, worth up to 0.28 nats
  per sample on the production catalog in a z-dependent (hence H0-correlated)
  way, and a narrowed view would silently switch it off.
"""

from types import SimpleNamespace

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_field_normalization_inputs
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood import factory as factory_mod
from darksirens.likelihood.factory import make_likelihood
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
from darksirens.redshift import catalog as catalog_mod
from darksirens.redshift.catalog import (
    catalog_kernel_state,
    eval_log_catalog_prior_state,
)
from darksirens.redshift.prior import EmptyRowRouting, WidthTier

APIX1 = hp.nside2pixarea(1)
POP_MODEL = "powerlaw+peak"
# The same dense grid the empty-row routing gate uses: last-bit moves are sparse
# in H0, and a five-point grid can miss them entirely.
H0_SCAN = tuple(float(h) for h in np.arange(20.0, 140.001, 2.5))
# ulp-level: relative drift with no coherent sign.
H0_SCAN_REL_TOL = 1e-12
H0_SCAN_SIGNED_MEAN_TOL = 1e-13
# The verifier's absolute gate, restated: < 1e-9 nats anywhere on the prior and
# a residual slope consistent with zero at 1e-11 nats per (km/s/Mpc).
H0_SCAN_ABS_NATS = 1e-9
H0_SCAN_SLOPE_NATS_PER_H0 = 1e-11

# Six compact rows, padded to 8 slots: real galaxies in the prefix [0, ngals),
# tail at z = 100, dz = 1, w = 0 (the loader contract).  Row lengths 2/4/0/7/0/3
# straddle the test ladder's cuts at 2 and 4 in both directions, including the
# boundary case ngals == cap.
N_ROWS = 6
N_MAX = 8
_NGALS = np.array([2, 4, 0, 7, 0, 3], dtype=np.int32)
TEST_CUTS = (2, 4)
TEST_CAPS = (2, 4, 8)

_ZG = np.full((N_ROWS, N_MAX), 100.0)
_DZ = np.full((N_ROWS, N_MAX), 1.0)
_WG = np.zeros((N_ROWS, N_MAX))
_rng = np.random.default_rng(20260906)
for _r, _n in enumerate(_NGALS):
    if _n:
        _ZG[_r, :_n] = np.sort(_rng.uniform(0.05, 0.25, size=int(_n)))
        _DZ[_r, :_n] = _rng.uniform(0.01, 0.03, size=int(_n))
        _WG[_r, :_n] = _rng.uniform(0.5, 1.5, size=int(_n))

NSAMP = 8
N_EVENTS = 2
N_PE = N_EVENTS * NSAMP
N_SEL = 24
# Every row is hit, and a third of each sample set lands on an empty one
# (production sits at 0.39-0.42, four times the routing's 0.10 floor).
_PE_ROWS = np.tile(np.array([0, 2, 1, 3, 5, 4, 1, 0], dtype=np.int32), N_EVENTS)
_SEL_ROWS = np.tile(np.array([0, 1, 2, 3, 5, 4], dtype=np.int32), N_SEL // 6)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cosmo():
    return CosmoParams(H0=70.0, Om0=0.3, w0=-1.0, wa=0.0)


def _survey():
    return SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                        alpha_miss=1.0)


def _catalog(ngals=None, zgals=None):
    return EMCatalog(
        apix=APIX1,
        zgals=jnp.asarray(_ZG if zgals is None else zgals),
        dzgals=jnp.asarray(_DZ), wgals=jnp.asarray(_WG),
        ngals=None if ngals is False else jnp.asarray(
            _NGALS if ngals is None else ngals),
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        dN_obs_kde=None, pixel_to_cache_idx=None,
    )


def _pop_bits():
    pop_fid = get_fixed_population_params(POP_MODEL)
    _lo, _hi, pop_labels, _a, _b = pop_model_prior_parser(POP_MODEL)
    fixed = {lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels)}
    return pop_fid, fixed


def _bundle():
    b = dict(
        apix=APIX1,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        zgals_pe=_ZG.copy(), dzgals_pe=_DZ.copy(), wgals_pe=_WG.copy(),
        ngals_pe=_NGALS.copy(),
        unique_pixels_pe=np.arange(N_ROWS, dtype=np.int32),
        sample_to_unique_pe=_PE_ROWS.copy(),
        zgals_sel=_ZG.copy(), dzgals_sel=_DZ.copy(), wgals_sel=_WG.copy(),
        ngals_sel=_NGALS.copy(),
        unique_pixels_sel=np.arange(N_ROWS, dtype=np.int32),
        sample_to_unique_sel=_SEL_ROWS.copy(),
    )
    fobs, _ne, nobs, _occ = build_field_normalization_inputs(
        jnp.asarray(_ZG), None, jnp.asarray(_NGALS)
    )
    b["field_dN_obs_s"] = fobs
    b["field_n_empty"] = float(hp.nside2npix(1) - 2)
    b["field_N_obs_total"] = float(nobs)
    return b


def _data():
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
    d.update(_bundle())
    return d


def _opts(catalog_sky_weighting="field", **kw):
    base = dict(
        pop_model=POP_MODEL, universe_model="dark_sirens", sel_batch_size=None,
        fix_cosmology=False, fix_population=True, fix_survey=True, fix_de=True,
        fixed_parameter_values={"Om0": 0.3},
        prior_overrides={"H0": [20.0, 140.0]},
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
        n_catalogs=1, catalog_sky_weighting=catalog_sky_weighting,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _build(monkeypatch, tiered, catalog_sky_weighting="field", **optkw):
    """Likelihood with the width ladder ON (``tiered``) or absent.

    Both arms keep the empty-row routing, so the A/B isolates stage 2: the only
    difference is whether the OCCUPIED samples are cut by row width.  The
    reference arm is spelled as an EMPTY cut list rather than a patched
    predicate, so it is a configuration the shipped code really produces (a
    catalog no wider than the first cut) and not a test-only branch.
    """
    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS",
                        TEST_CUTS if tiered else ())
    pop_fid, fixed = _pop_bits()
    fixed = dict(fixed, Om0=0.3)
    return make_likelihood(
        _opts(catalog_sky_weighting, **optkw), _data(), pop_fid,
        fixed_parameter_values=fixed,
    )


def _plans(lik):
    """``(pe, sel)`` plan of the single catalog."""
    routing = lik.empty_row_routing
    assert routing, "expected a routing plan"
    return routing[0]


# ---------------------------------------------------------------------------
# The gate: ulp-level, with no H0 trend
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("catalog_sky_weighting", ["field", "conditional"])
def test_tiering_is_ulp_level_with_no_H0_trend(monkeypatch, catalog_sky_weighting):
    """The campaign's bar for stage 2, on the sampled H0 prior.

    A width tier evaluates the same galaxies over a shorter row, so the residual
    is pure logsumexp re-association.  Re-association is not H0-INDEPENDENT --
    the summands are functions of ``z = z_of_dL(dL; H0)``, so the residual is a
    smooth function of H0 and "the partition is data, therefore no H0
    correlation" is a non-sequitur.  What makes it harmless is its SIZE, so that
    is what this asserts: a relative bound, a signed mean (a coherent tilt is
    the failure mode; rounding averages out) and the slope against H0.
    """
    lo = _build(monkeypatch, tiered=False,
                catalog_sky_weighting=catalog_sky_weighting)
    hi = _build(monkeypatch, tiered=True,
                catalog_sky_weighting=catalog_sky_weighting)
    assert _plans(hi)[0].tiers, "the tiered arm did not tier"
    assert not _plans(lo)[0].tiers, "the reference arm tiered"

    h0 = np.asarray(H0_SCAN)
    ref = np.array([float(lo(jnp.asarray([h]))) for h in h0])
    got = np.array([float(hi(jnp.asarray([h]))) for h in h0])
    assert np.array_equal(np.isfinite(ref), np.isfinite(got))
    fin = np.isfinite(ref)
    assert fin.sum() >= 5
    d = got[fin] - ref[fin]
    rel = np.abs(d) / np.maximum(np.abs(ref[fin]), 1.0)
    assert rel.max() <= H0_SCAN_REL_TOL, rel.max()
    assert abs(np.mean(rel * np.sign(d))) <= H0_SCAN_SIGNED_MEAN_TOL
    assert np.abs(d).max() <= H0_SCAN_ABS_NATS, np.abs(d).max()
    slope = np.polyfit(h0[fin], d, 1)[0]
    assert abs(slope) <= H0_SCAN_SLOPE_NATS_PER_H0, slope


def test_the_tiered_prior_vector_is_ulp_level_sample_by_sample(monkeypatch):
    """Below the reductions: the per-sample ``log p(z | pix)`` itself.

    The whole-likelihood gate above can only see what survives two logsumexps.
    This compares the prior VECTOR the evaluator returns, in the caller's sample
    order, so a per-sample decorrelation (the failure mode that tilts with H0)
    cannot hide inside a reduction.
    """
    from darksirens.redshift.prior import (
        eval_redshift_prior_with_state, prepare_redshift_prior_state,
    )

    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS", TEST_CUTS)
    cat = _catalog()
    state = prepare_redshift_prior_state(
        "dark_sirens", _cosmo(), _survey(), cat, kde_window=None,
    )
    plan = factory_mod._empty_row_routing_plan(
        _PE_ROWS, _NGALS, tier_caps=factory_mod._row_width_tier_caps(N_MAX))
    flat = factory_mod._empty_row_routing_plan(_PE_ROWS, _NGALS, tier_caps=None)
    assert plan.tiers and not flat.tiers

    z = jnp.asarray(np.linspace(0.06, 0.24, N_PE))
    pix = jnp.asarray(_PE_ROWS)
    kw = dict(model="dark_sirens", state=state, z=z, pix=pix, cosmo=_cosmo(),
              survey=_survey(), em_catalog=cat, catalog_sky_weighting="field")
    ref = np.asarray(eval_redshift_prior_with_state(**kw, empty_routing=flat))
    got = np.asarray(eval_redshift_prior_with_state(**kw, empty_routing=plan))
    assert np.array_equal(np.isfinite(ref), np.isfinite(got))
    f = np.isfinite(ref)
    assert f.sum() >= N_PE // 2
    assert np.max(np.abs(got[f] - ref[f]) / np.abs(ref[f])) <= H0_SCAN_REL_TOL


def test_a_capped_gather_drops_only_padding():
    """The mechanism, at the evaluator: ``a[pix, :cap]`` for ``cap >= ngals``.

    Every slot the cap removes carries the build-time ``-1e30`` sentinel, whose
    ``exp(-1e30 - m)`` is already exactly ``0.0`` in the full-row sum, so the
    real number the reduction represents does not change at all.  On rows this
    small XLA has no room to re-associate either, and the result is bit-equal;
    the ulp allowance in the class exists for the 1719-slot production rows.
    """
    cat = _catalog()
    kern = catalog_kernel_state(_cosmo(), _survey(), cat)
    zs = np.linspace(0.06, 0.24, 7)
    for row, ng in enumerate(_NGALS):
        if ng == 0:
            continue
        cap = int(min(c for c in TEST_CAPS if c >= ng))
        for z in zs:
            full = float(eval_log_catalog_prior_state(
                jnp.asarray(z), jnp.asarray(row), kern, cat))
            cut = float(eval_log_catalog_prior_state(
                jnp.asarray(z), jnp.asarray(row), kern, cat, col_cap=cap))
            assert full == cut, (row, ng, cap, z, full, cut)


def test_the_empty_row_select_survives_a_capped_gather():
    """An empty row still returns EXACTLY ``-inf`` under a cap.

    ``row_empty`` is a per-ROW scalar select and the cap only narrows the
    per-COLUMN gather, so the contract the routing's substitution rests on is
    untouched -- and the tier ladder must not become the thing that quietly
    turns an empty pixel into a finite ``-1e30 + log cap``.
    """
    cat = _catalog()
    kern = catalog_kernel_state(_cosmo(), _survey(), cat)
    for row in np.flatnonzero(_NGALS == 0):
        v = float(eval_log_catalog_prior_state(
            jnp.asarray(0.1), jnp.asarray(int(row)), kern, cat, col_cap=2))
        assert v == -np.inf


def test_the_tiered_evaluator_has_finite_gradients(monkeypatch):
    """The ladder must not reintroduce the empty-row NaN cotangent.

    ``logsumexp`` of an all--inf row has a 0/0 softmax whose NaN survives
    multiplication by a zero upstream cotangent; that is what broke NUTS on
    empty catalog pixels, and the sanitized padding plus the double ``where``
    are what fixed it.  A narrower gather changes how many sanitized slots are
    present, so re-pin it here, at the seam that owns the reduction.
    """
    from darksirens.redshift.prior import (
        eval_redshift_prior_with_state, prepare_redshift_prior_state,
    )

    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS", TEST_CUTS)
    cosmo, survey, cat = _cosmo(), _survey(), _catalog()
    state = prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat,
                                         kde_window=None)
    pix = jnp.asarray(_PE_ROWS, dtype=jnp.int32)
    plan = factory_mod._empty_row_routing_plan(
        _PE_ROWS, _NGALS, tier_caps=factory_mod._row_width_tier_caps(N_MAX))
    assert plan.tiers

    def _total(z):
        v = eval_redshift_prior_with_state(
            "dark_sirens", state, z, pix, cosmo, survey, cat,
            empty_routing=plan,
        )
        return jnp.sum(jnp.where(jnp.isfinite(v), v, 0.0))

    z = jnp.full(N_PE, 0.12)
    assert np.all(np.isfinite(np.asarray(jax.grad(_total)(z))))


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------

def test_the_ladder_partitions_every_sample_exactly_once(monkeypatch):
    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS", TEST_CUTS)
    caps = factory_mod._row_width_tier_caps(N_MAX)
    for pixels in (_PE_ROWS, _SEL_ROWS):
        plan = factory_mod._empty_row_routing_plan(pixels, _NGALS,
                                                   tier_caps=caps)
        assert plan.idx_occ is None, "the tiers ARE the occupied group"
        order = np.concatenate(
            [np.asarray(t.idx) for t in plan.tiers]
            + [np.asarray(plan.idx_empty)]
        )
        assert np.array_equal(np.sort(order), np.arange(pixels.size))
        inv = np.asarray(plan.inv_order)
        assert np.array_equal(order[inv], np.arange(pixels.size))


def test_every_sample_reads_at_least_its_own_row(monkeypatch):
    """The invariant the whole change rests on: ``cap_t >= ngals[pix]``.

    Violate it and real galaxies are silently dropped -- not a ulp error but a
    truncating estimator wearing an exact one's name.
    """
    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS", TEST_CUTS)
    caps = factory_mod._row_width_tier_caps(N_MAX)
    plan = factory_mod._empty_row_routing_plan(_PE_ROWS, _NGALS, tier_caps=caps)
    seen = set()
    for tier in plan.tiers:
        n_at = _NGALS[_PE_ROWS[np.asarray(tier.idx)]]
        assert n_at.min() > 0, "empty rows belong to the routed group"
        assert n_at.max() <= tier.cap, (tier.cap, n_at.max())
        seen.add(tier.cap)
        # The ladder is TIGHT as well as safe: no sample sits two rungs up.
        below = [c for c in caps if c < tier.cap]
        if below:
            assert n_at.max() > max(below)
    assert seen <= set(caps)


def test_the_ladder_caps_are_the_cuts_inside_the_row_plus_n_max(monkeypatch):
    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS", TEST_CUTS)
    assert factory_mod._row_width_tier_caps(8) == (2, 4, 8)
    # Cuts at or beyond n_max are dropped; n_max is always the last cap, so no
    # row can be narrowed below its own galaxy count.
    assert factory_mod._row_width_tier_caps(4) == (2, 4)
    assert factory_mod._row_width_tier_caps(3) == (2, 3)
    # A catalog no wider than the first cut yields ONE cap: no ladder at all.
    assert factory_mod._row_width_tier_caps(2) == (2,)


def test_the_production_ladder_is_three_tiers():
    """The shipped ladder, and the measurement that chose it.

    MEASURED on an H100 NVL over the real production sample sets, on top of the
    one-pass reduction: caps (0, 1024, 1719) run the catalog KDE in 8.41 ms
    against 9.08 ms for (0, 256, 512, 768, 1024, 1280, 1719) and 15.54 ms flat.
    The finer ladder gathers fewer slots and is SLOWER -- its small tiers are
    launch-bound -- so the default must stay a single interior cut.
    """
    assert factory_mod._KDE_ROW_WIDTH_TIER_CUTS == (1024,)
    assert factory_mod._row_width_tier_caps(1719) == (1024, 1719)


def test_a_tier_that_ends_up_empty_is_dropped(monkeypatch):
    """No zero-size kernel gets lowered for a rung nothing lands on."""
    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS", TEST_CUTS)
    # Samples only on rows 0 (ngals 2) and 2 (empty): the 4 and 8 rungs are
    # unpopulated, leaving ONE occupied group -- which is the untiered plan.
    pixels = np.array([0, 2, 0, 2, 0, 2, 0, 2], dtype=np.int32)
    plan = factory_mod._empty_row_routing_plan(
        pixels, _NGALS, tier_caps=factory_mod._row_width_tier_caps(N_MAX))
    assert plan.tiers == ()
    assert plan.idx_occ is not None


# ---------------------------------------------------------------------------
# The two halves of the exactness promise
# ---------------------------------------------------------------------------

def test_a_cap_below_its_samples_row_poisons_the_likelihood(monkeypatch):
    """The IN-GRAPH half: ``all(ngals[pix[tier.idx]] <= tier.cap)``.

    Build a good likelihood, then hand the same compiled body a plan whose
    widest tier has been narrowed -- exactly what a plan built against a
    different (same-length) sample vector or a stale ``ngals`` would look like.
    Without the check the result is finite, plausible and WRONG (real galaxies
    truncated away); with it the whole log-likelihood is ``-inf`` and the run
    cannot start.
    """
    lik = _build(monkeypatch, tiered=True)
    pe, sel = _plans(lik)
    assert np.isfinite(float(lik(jnp.asarray([70.0]))))

    widest = max(t.cap for t in pe.tiers)
    bad = pe._replace(tiers=tuple(
        WidthTier(2 if t.cap == widest else t.cap, t.idx) for t in pe.tiers))
    operands = list(lik.operands)
    operands[-2] = ((bad, sel),)
    v = float(lik.jitted_body(jnp.asarray([70.0]), tuple(operands),
                              lik.distance_table, lik.smoothing_operator))
    assert v == -np.inf


def test_a_live_slot_past_ngals_refuses_the_ladder():
    """The BUILD-TIME half, asserted on the array the graph will read.

    ``catalog._row_real_mask`` guarantees live slots occupy ``[0, ngals)`` when
    ``ngals`` is present -- but a build-time kernel PIN is a stored array, and a
    pin built from a stale or differently ordered catalog would break the
    invariant without breaking the rule that made it.  So the check is on the
    live slot COLUMN INDEX itself, and it refuses rather than falling back.
    """
    cat = _catalog()
    kern = catalog_kernel_state(_cosmo(), _survey(), cat)
    good = SimpleNamespace(log_kw_eff=kern.log_kw_eff)
    factory_mod._assert_live_slots_are_row_prefix(
        cat._replace(pinned_kernels=good))

    lk = np.asarray(kern.log_kw_eff).copy()
    lk[0, N_MAX - 1] = -1.0          # a live slot past row 0's ngals == 2
    bad = SimpleNamespace(log_kw_eff=jnp.asarray(lk))
    with pytest.raises(ValueError, match="live kernel slots"):
        factory_mod._assert_live_slots_are_row_prefix(
            cat._replace(pinned_kernels=bad))


def test_the_ladder_refuses_a_catalog_without_ngals():
    """Without ``ngals`` the live slots are ``ws > 0`` -- not a prefix."""
    with pytest.raises(ValueError, match="ngals"):
        factory_mod._assert_live_slots_are_row_prefix(_catalog(ngals=False))


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def test_an_armed_kde_window_refuses_the_ladder(monkeypatch):
    """The refusal the merged candidate record left out entirely.

    The windowed branch keeps the ``window`` galaxies nearest ``z`` and drops
    the rest: a DIFFERENT estimator, MEASURED at up to 0.28 nats per sample
    against the full row on the production catalog, and z-dependent -- hence
    H0-correlated, the class this campaign refuses outright.  It arms on
    ``zgals.shape[1] > window``, which a ``cap``-wide view is not, so tiering
    under an armed window would silently run two estimators in one likelihood.
    """
    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS", TEST_CUTS)
    cat = _catalog()
    # n_max 8 against a window of 4: the untiered evaluator WOULD window.
    assert not factory_mod._row_width_tiering_admissible(cat, kde_window=4)
    # A window at or above n_max never arms, so the ladder is admissible.
    assert factory_mod._row_width_tiering_admissible(cat, kde_window=N_MAX)
    assert factory_mod._row_width_tiering_admissible(cat, kde_window=64)


def test_a_windowed_build_ships_no_tiers(monkeypatch):
    """End to end: ``--kde_window`` below ``n_max`` keeps the untiered graph."""
    monkeypatch.setattr(factory_mod, "_KDE_ROW_WIDTH_TIER_CUTS", TEST_CUTS)
    monkeypatch.setattr(catalog_mod, "_KDE_WINDOW_SIZE", 4)
    lik = _build(monkeypatch, tiered=True, kde_window=4)
    assert _plans(lik)[0].tiers == ()
    assert _plans(lik)[1].tiers == ()


def test_the_evaluator_refuses_a_cap_under_an_armed_window(monkeypatch):
    """Belt and braces: the process window can change AFTER the build.

    ``configure_catalog_kde_window`` writes a module global that is not part of
    any jit cache key, so a graph built while the full-row branch was the live
    one can be traced later under an armed window.  The evaluator raises at
    trace time rather than silently taking the other estimator.
    """
    cat = _catalog()
    kern = catalog_kernel_state(_cosmo(), _survey(), cat, kde_window=4)
    with pytest.raises(ValueError, match="windowed"):
        eval_log_catalog_prior_state(
            jnp.asarray(0.1), jnp.asarray(0), kern, cat, col_cap=2)


@pytest.mark.parametrize("cap", [0, -1, N_MAX + 1])
def test_the_evaluator_refuses_a_cap_outside_the_row(cap):
    cat = _catalog()
    kern = catalog_kernel_state(_cosmo(), _survey(), cat)
    with pytest.raises(ValueError, match="col_cap must lie"):
        eval_log_catalog_prior_state(
            jnp.asarray(0.1), jnp.asarray(0), kern, cat, col_cap=cap)


def test_the_evaluator_refuses_a_cap_without_ngals():
    cat = _catalog(ngals=False)
    kern = catalog_kernel_state(_cosmo(), _survey(), cat)
    with pytest.raises(ValueError, match="ngals"):
        eval_log_catalog_prior_state(
            jnp.asarray(0.1), jnp.asarray(0), kern, cat, col_cap=2)


def test_the_evaluator_refuses_a_cap_on_a_1d_view():
    cat = EMCatalog(
        apix=APIX1, zgals=jnp.asarray(_ZG[0]), dzgals=jnp.asarray(_DZ[0]),
        wgals=jnp.asarray(_WG[0]), ngals=None,
        delta_g_pix_z=jnp.zeros((1, len(zgrid))),
        dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    kern = catalog_kernel_state(_cosmo(), _survey(), _catalog())
    with pytest.raises(ValueError, match="2-D"):
        eval_log_catalog_prior_state(
            jnp.asarray(0.1), jnp.asarray(0), kern, cat, col_cap=2)


# ---------------------------------------------------------------------------
# The plumbing
# ---------------------------------------------------------------------------

def test_the_cap_is_pytree_aux_data_not_a_leaf():
    """A static column width has to survive into the traced graph as an int.

    It is also part of the jit cache key, because two ladders are two different
    graphs -- which is what stops a body compiled for caps (2, 4, 8) from being
    replayed on a plan cut at (2, 8).
    """
    tier = WidthTier(4, jnp.arange(3, dtype=jnp.int32))
    leaves, treedef = jax.tree_util.tree_flatten(tier)
    assert len(leaves) == 1 and leaves[0].shape == (3,)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert rebuilt.cap == 4 and isinstance(rebuilt.cap, int)
    other = jax.tree_util.tree_structure(
        WidthTier(8, jnp.arange(3, dtype=jnp.int32)))
    assert treedef != other


def test_the_untiered_plan_still_carries_its_flat_index():
    """``tiers=()`` must leave stage 1 exactly as it shipped."""
    plan = factory_mod._empty_row_routing_plan(_PE_ROWS, _NGALS, tier_caps=None)
    assert plan.tiers == ()
    assert plan.idx_occ is not None
    assert isinstance(plan, EmptyRowRouting)
