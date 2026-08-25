"""Parity tests for the block-vectorized singleton PE reduction in
``darksirens/likelihood/likelihood_with_clusters.py``.

The cluster master likelihood used to drive its singleton channel with a
sequential per-event ``lax.scan`` (one tiny iteration per event -- 259 on the
singleton campaign), long after ``likelihood/core.py`` had replaced the same
loop with a block-vectorized reduction: a chunk of ``pe_event_block`` events is
evaluated in ONE flattened ``(block*nsamp,)`` call and reduced per event by a
row-wise (vmapped) ``log_evidence_and_mc_variance``.  The per-sample weight
kernels are elementwise in the sample axis, so the block path reduces the
identical masked elements in the identical per-row order.

Two configurations deliberately KEEP the exact scan and must be shown to:

  * a PE catalog carrying per-event bright-siren counterpart arrays, whose
    ``active_counterpart_index`` the flattened call cannot express, and
  * ``singleton_lensing=MIXTURE``, whose lensed branch reduces over BOTH the
    sample and the y-quadrature axis inside
    ``lensed_single_log_likelihood_event`` and so is not a row-wise reduction
    of a flat per-sample vector.

The old path is exercised in-process by forcing the ``has_counterpart`` gate
on, which selects the historical ``_pe_event_fn`` scan verbatim (its
``_replace(active_counterpart_index=...)`` is inert on a catalog with no
counterpart arrays).

Run with ``JAX_PLATFORMS=cpu``.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import darksirens.likelihood.likelihood_with_clusters as lwc
from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.lensing.fcpdet import make_fc_pdet_params
from darksirens.lensing.lensed_injections import (
    LensedSingleImageSet,
    make_lensed_injection_set,
)
from darksirens.lensing.slmarks import make_sis_lens_params
from darksirens.likelihood.likelihood_with_clusters import (
    CLUSTER_MODE_J2,
    CLUSTER_MODE_OFF,
    SINGLETON_LENSING_MIXTURE,
    SINGLETON_LENSING_OFF,
    WL_BACKEND_DISABLED,
    darksiren_likelihood_diagnostics_with_clusters,
    darksiren_log_likelihood_with_clusters,
)

N_EVENTS, N_SAMP, N_SEL = 9, 120, 400
# Non-contiguous singleton subset: the J2 chunk plan must GATHER these rows,
# not slice a contiguous event range.
J2_SINGLETONS = (1, 2, 4, 5, 7, 8)
J2_PAIRS = ((0, 3), (6, 0))


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075)


def _survey():
    return SurveyParams(
        n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5,
    )


def _toy_catalog(**kw):
    return EMCatalog(
        apix=1.0, zgals=jnp.zeros((1, 1)), dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)), ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None, pixel_to_cache_idx=None, **kw,
    )


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(0)
    total = N_EVENTS * N_SAMP
    gw_pe = GWEvent(
        m1det=jnp.asarray(rng.uniform(20.0, 60.0, total)),
        m2det=jnp.asarray(rng.uniform(10.0, 30.0, total)),
        dL=jnp.asarray(rng.uniform(400.0, 3000.0, total)),
        chieff=jnp.asarray(rng.uniform(-0.3, 0.3, total)),
        prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, total)),
        pixels=jnp.zeros(total, dtype=jnp.int32),
        q=jnp.asarray(rng.uniform(0.3, 1.0, total)),
        valid=jnp.ones(total, dtype=jnp.bool_),
    )
    gw_sel = GWEvent(
        m1det=jnp.asarray(rng.uniform(15.0, 70.0, N_SEL)),
        m2det=jnp.asarray(rng.uniform(8.0, 35.0, N_SEL)),
        dL=jnp.asarray(rng.uniform(200.0, 3000.0, N_SEL)),
        chieff=jnp.asarray(rng.uniform(-0.3, 0.3, N_SEL)),
        prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, N_SEL)),
        pixels=jnp.zeros(N_SEL, dtype=jnp.int32),
        q=jnp.asarray(rng.uniform(0.3, 1.0, N_SEL)),
        valid=jnp.ones(N_SEL, dtype=jnp.bool_),
    )
    return {
        "cosmo": _cosmo(), "survey": _survey(), "catalog": _toy_catalog(),
        "gw_pe": gw_pe, "gw_sel": gw_sel, "Ndraw": 1000.0,
        "pop_params": get_fixed_population_params("powerlaw+peak"),
        "lensed": _lensed_injections(),
    }


def _lensed_injections(n=400, seed=1234):
    """Minimal both-detected lensed campaign: SIS doubles from uniform
    proposals, everything flagged detected.  cluster_mode=J2 needs a campaign
    for mu_sel^(2); its value is irrelevant here (the singleton reduction is
    what is under test) but it must be the SAME on both sides of every A/B."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.05, 0.95, n)
    return make_lensed_injection_set(
        source_id=np.repeat(np.arange(n, dtype=np.int32), 2),
        image_id=np.tile(np.array([0, 1], dtype=np.int32), n),
        m1_src=np.repeat(rng.uniform(10.0, 70.0, n), 2),
        q_src=np.repeat(rng.uniform(0.3, 1.0, n), 2),
        z_src=np.repeat(rng.uniform(0.05, 1.5, n), 2),
        chieff=np.repeat(rng.uniform(-0.4, 0.4, n), 2),
        y_source=np.repeat(y, 2),
        mu=np.stack([(1.0 + y) / y, (1.0 - y) / y], axis=1).reshape(-1),
        detected=np.ones(2 * n, dtype=bool),
        p_prop_src=np.full(2 * n, 1.0 / (60 * 0.7 * 1.45 * 0.8)),
        p_prop_y=np.full(2 * n, 1.0 / 0.9),
        n_draw_sources=4000,
    )


def _diagnostics(fx, *, pe_event_block, j2=False, catalog=None, **kw):
    """Singleton-channel diagnostics for one block-size setting.

    The total-variance guard drives ``logL_total`` to -inf on this deliberately
    small toy selection set, so the comparands are the singleton sums, which is
    exactly what the reduction produces.
    """
    j2_singletons = jnp.asarray(J2_SINGLETONS, dtype=jnp.int32)
    return darksiren_likelihood_diagnostics_with_clusters(
        fx["cosmo"], fx["survey"], fx["pop_params"],
        fx["gw_pe"], fx["catalog"] if catalog is None else catalog,
        fx["gw_sel"], fx["catalog"],
        N_EVENTS, N_SAMP, fx["Ndraw"],
        singleton_indices=(
            j2_singletons if j2 else jnp.arange(N_EVENTS, dtype=jnp.int32)
        ),
        pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
        n_singletons=len(J2_SINGLETONS) if j2 else N_EVENTS,
        n_pairs=0,
        lensed_injections=fx["lensed"] if j2 else None, pair_kdes=None,
        # cluster_mode=J2 with no pairs still turns on the (1 - tau_2)
        # singleton suppression, which is the second elementwise branch the
        # block path has to reproduce.  A_tau is inflated so it bites.
        sis_params=make_sis_lens_params(A_tau=0.3, n_tau=0.0, T0_seconds=1.0),
        log_p_tag_per_source=(
            jnp.zeros(fx["lensed"].n_kept) if j2 else jnp.zeros(0)
        ),
        pop_model="powerlaw+peak", universe_model="spectral_sirens",
        sel_batch_size=None,
        cluster_mode=CLUSTER_MODE_J2 if j2 else CLUSTER_MODE_OFF,
        wl_backend=WL_BACKEND_DISABLED,
        pe_event_block=pe_event_block,
        **kw,
    )


def _sums(diag):
    return (
        float(np.asarray(diag["singleton_logL_sum"])),
        float(np.asarray(diag["singleton_variance_sum"])),
    )


def _scan_reference(fx, **kw):
    """The historical per-event ``lax.scan``, forced by pinning the
    ``has_counterpart`` gate on.  The scan's per-event
    ``_replace(active_counterpart_index=...)`` is inert on a catalog with no
    counterpart arrays, so this is the pre-vectorization body verbatim."""
    original = lwc._has_counterpart_arrays
    darksiren_log_likelihood_with_clusters.jitted.clear_cache()
    try:
        lwc._has_counterpart_arrays = lambda catalog: True
        diag = _diagnostics(fx, **kw)
        # The A/B is worthless if the patch did not take (a cached executable,
        # or the gate moved): the scan arm must REPORT the scan.
        assert int(np.asarray(diag["pe_event_block"])) == 1
        return _sums(diag)
    finally:
        lwc._has_counterpart_arrays = original
        darksiren_log_likelihood_with_clusters.jitted.clear_cache()


# ---------------------------------------------------------------------------
# (a) The block path reproduces the per-event scan.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("j2", [False, True], ids=["cluster_off", "cluster_j2"])
def test_block_reduction_matches_per_event_scan(fixture, j2):
    """One vectorized block over the singletons == the sequential scan.

    MEASURED bit-identical on CPU and on an H100 for both chunk plans (the
    contiguous cluster_mode=OFF slice and the cluster_mode=J2 gather); the
    tolerance allows the ULP-level reassociation of the final sum that XLA is
    free to choose, five orders below the ~1e-9 this repo already treats as
    benign compile-context drift.
    """
    ll_scan, var_scan = _scan_reference(fixture, pe_event_block=None, j2=j2)
    ll_block, var_block = _sums(_diagnostics(fixture, pe_event_block=None, j2=j2))
    assert np.isfinite(ll_scan) and np.isfinite(var_scan)
    np.testing.assert_allclose(ll_block, ll_scan, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(var_block, var_scan, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("j2", [False, True], ids=["cluster_off", "cluster_j2"])
@pytest.mark.parametrize("block", [1, 3, 4])
def test_chunked_block_plan_matches_single_pass(fixture, j2, block):
    """``pe_event_block=1`` (row by row), an exact divisor, and a plan with a
    remainder chunk (9 = 2*4 + 1, 6 = 1*4 + 2) all agree with the single pass."""
    ll_ref, var_ref = _sums(_diagnostics(fixture, pe_event_block=None, j2=j2))
    ll_blk, var_blk = _sums(_diagnostics(fixture, pe_event_block=block, j2=j2))
    np.testing.assert_allclose(ll_blk, ll_ref, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(var_blk, var_ref, rtol=1e-13, atol=0.0)


def test_block_larger_than_the_row_count_is_clipped(fixture):
    """A block wider than the singleton count must not pad or wrap."""
    ll_ref, var_ref = _sums(_diagnostics(fixture, pe_event_block=None, j2=True))
    ll_big, var_big = _sums(_diagnostics(fixture, pe_event_block=1000, j2=True))
    np.testing.assert_allclose(ll_big, ll_ref, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(var_big, var_ref, rtol=1e-13, atol=0.0)


# ---------------------------------------------------------------------------
# (b) The two configurations that keep the exact scan.
# ---------------------------------------------------------------------------

def test_counterpart_catalog_keeps_the_exact_scan(fixture):
    """Bright sirens set ``active_counterpart_index`` per event, which the
    flattened block call cannot express, so any per-event counterpart array on
    the PE catalog must pin the scan (reported as pe_event_block=1)."""
    assert not lwc._has_counterpart_arrays(fixture["catalog"])
    for field in ("counterpart_pixels", "counterpart_zs", "counterpart_dzs"):
        bright = _toy_catalog(**{field: jnp.zeros((N_EVENTS,))})
        assert lwc._has_counterpart_arrays(bright), field
    # The legacy scalar counterpart_pixel is NOT a per-event array and must not
    # trip the gate (it would forfeit the vectorization for nothing).
    assert not lwc._has_counterpart_arrays(_toy_catalog(counterpart_pixel=3))
    diag = _diagnostics(
        fixture, pe_event_block=None,
        catalog=_toy_catalog(counterpart_zs=jnp.full((N_EVENTS,), 0.1),
                             counterpart_dzs=jnp.full((N_EVENTS,), 0.01)),
    )
    assert int(np.asarray(diag["pe_event_block"])) == 1


def _lensed_singles(n=48, seed=3):
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.05, 0.95, n)
    return LensedSingleImageSet(
        m1_src=jnp.asarray(rng.uniform(10.0, 70.0, n)),
        q_src=jnp.asarray(rng.uniform(0.3, 1.0, n)),
        z_src=jnp.asarray(rng.uniform(0.05, 1.5, n)),
        chieff=jnp.asarray(rng.uniform(-0.4, 0.4, n)),
        y_source=jnp.asarray(y),
        mu_det=jnp.asarray((1.0 + y) / y),
        mu_partner=jnp.asarray((1.0 - y) / y),
        image_is_plus=jnp.ones(n, dtype=jnp.bool_),
        p_prop_src=jnp.full(n, 1.0 / (60 * 0.7 * 1.45 * 0.8)),
        p_prop_y=jnp.full(n, 1.0 / 0.9),
        valid=jnp.ones(n, dtype=jnp.bool_),
        n_draw_sources=jnp.asarray(4000.0),
    )


def test_mixture_keeps_the_exact_scan(fixture):
    """``singleton_lensing=MIXTURE`` reduces over BOTH the sample and the
    y-quadrature axis inside ``lensed_single_log_likelihood_event``, so it is
    not a row-wise reduction of a flat per-sample vector and must stay on the
    scan -- reported as pe_event_block=1, and inert under the knob."""
    assert int(np.asarray(
        _diagnostics(fixture, pe_event_block=None)["pe_event_block"]
    )) == N_EVENTS
    mix = dict(
        singleton_lensing=SINGLETON_LENSING_MIXTURE,
        lensed_singles=_lensed_singles(),
        fc_pdet_params=make_fc_pdet_params(rho_thr=8.0, horizon_mpc=3000.0),
        y_nodes_single=16,
    )
    diag = _diagnostics(fixture, pe_event_block=None, **mix)
    assert int(np.asarray(diag["pe_event_block"])) == 1
    # ... and the knob cannot change the answer on a path that ignores it.
    diag_blocked = _diagnostics(fixture, pe_event_block=2, **mix)
    assert (
        float(np.asarray(diag_blocked["singleton_logL_sum"]))
        == float(np.asarray(diag["singleton_logL_sum"]))
    )


# ---------------------------------------------------------------------------
# (c) Contract guards.
# ---------------------------------------------------------------------------

def test_pe_event_block_is_static(fixture):
    """A TRACED block size would silently defeat the static chunk plan (and
    make the ``min``/``//`` plan arithmetic a trace error), so two settings
    must compile to two DIFFERENT executables."""
    jitted = darksiren_log_likelihood_with_clusters.jitted
    jitted.clear_cache()
    _diagnostics(fixture, pe_event_block=None)
    n_one = jitted._cache_size()
    assert n_one == 1
    _diagnostics(fixture, pe_event_block=3)
    assert jitted._cache_size() == 2


@pytest.mark.parametrize("bad", [0, -1, -7])
def test_non_positive_pe_event_block_raises(fixture, bad):
    with pytest.raises(ValueError, match="pe_event_block must be a positive"):
        _diagnostics(fixture, pe_event_block=bad)
