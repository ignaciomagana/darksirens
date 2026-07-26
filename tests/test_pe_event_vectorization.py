"""Parity tests for the block-vectorized per-event PE reduction.

``darksirens/likelihood/core.py`` replaced the per-event ``lax.scan`` (one
sequential tiny iteration per event) with a block-vectorized reduction: a chunk
of ``pe_event_block`` events is evaluated in ONE flattened ``(block*nsamp,)``
call and reduced per event by a row-wise (vmapped) ``log_evidence_and_mc_variance``.
The per-sample weight kernels are elementwise in the sample axis (the redshift
prior / catalog KDE vmap per-sample; the population term and Jacobians are
pointwise), so:

  * ``pe_event_block=None`` (all events in one vectorized block) MUST equal
    ``pe_event_block=1`` (the historical per-event scan) bit-for-bit, and
  * a partial block (remainder chunk) MUST equal both.

Bright-siren configs carry per-event counterpart arrays (``active_counterpart_
index`` is event-dependent), so they KEEP the exact per-event scan; a bright
config's value must be independent of ``pe_event_block``.

Run with ``JAX_PLATFORMS=cpu``.
"""
import jax

jax.config.update("jax_enable_x64", True)

from types import SimpleNamespace

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.inference.prior import build_parameter_space
from darksirens.likelihood.factory import make_likelihood

NG = len(zgrid)
NSIDE = 1
NPIX = hp.nside2npix(NSIDE)
NSAMP = 2
NSEL = 8


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    prior_overrides = {sampled: [float(pop_lower[0]), float(pop_upper[0])]}
    fixed = {
        lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled
    }
    return pop_fid, prior_overrides, fixed


def _unit_dirs(n, phase):
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    nx = np.cos(ang) * np.sqrt(0.75)
    ny = np.sin(ang) * np.sqrt(0.75)
    nz = np.full(n, 0.5)
    return jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz)


def _full_sky_multi(nEvents, nsamp=NSAMP, n_sel=NSEL):
    """Full-sky flat-data dict with ``nEvents`` events (each event's nsamp
    samples vary in mass / distance / sky pixel so the per-event evidences
    genuinely differ)."""
    npe = nEvents * nsamp
    zgals = np.full((NPIX, 1), 0.10, dtype=float)
    dzgals = np.full((NPIX, 1), 0.02, dtype=float)
    wgals = np.ones((NPIX, 1), dtype=float)
    ngals = np.ones(NPIX, dtype=np.int32)
    nx_pe, ny_pe, nz_pe = _unit_dirs(npe, 0.1)
    nx_sel, ny_sel, nz_sel = _unit_dirs(n_sel, 0.7)
    # Per-event varying PE samples.
    m1 = np.linspace(34.0, 42.0, npe)
    dL = np.linspace(430.0, 560.0, npe)
    chieff = np.linspace(-0.05, 0.08, npe)
    pixels_pe = np.array(
        [(7 + e) % NPIX for e in range(nEvents) for _ in range(nsamp)],
        dtype=np.int32,
    )
    return {
        "nEvents": nEvents,
        "nsamp": nsamp,
        "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(NSIDE),
        "nside": NSIDE,
        "n_pix_catalog": NPIX,
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals_catalog": ngals,
        "zgals_catalog": zgals,
        "dzgals_catalog": dzgals,
        "wgals_catalog": wgals,
        "delta_g_pix_z": jnp.zeros((NPIX, NG)),
        "m1det": jnp.asarray(m1),
        "m2det": jnp.asarray(0.8 * m1),
        "dL": jnp.asarray(dL),
        "chieff": jnp.asarray(chieff),
        "p_pe": jnp.ones(npe),
        "pixels_pe": jnp.asarray(pixels_pe),
        "nx_pe": nx_pe, "ny_pe": ny_pe, "nz_pe": nz_pe,
        "m1detsels": jnp.linspace(34.0, 40.0, n_sel),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, n_sel),
        "dLsels": jnp.linspace(430.0, 530.0, n_sel),
        "chieffsels": jnp.zeros(n_sel),
        "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.asarray([2, 7, 2, 7, 2, 7, 2, 7][:n_sel], dtype=jnp.int32),
        "nx_sel": nx_sel, "ny_sel": ny_sel, "nz_sel": nz_sel,
    }


def _bundle_multi(z, nEvents, nsamp=NSAMP, n_sel=NSEL, dz=0.02):
    """Single compact-row catalog bundle (multitracer compact-view contract):
    ALL PE/selection samples map to the one row."""
    npe = nEvents * nsamp
    return dict(
        apix=hp.nside2pixarea(NSIDE),
        delta_g_pix_z=jnp.zeros((1, NG)),
        zgals_pe=np.array([[z]]), dzgals_pe=np.array([[dz]]),
        wgals_pe=np.array([[1.0]]), ngals_pe=np.array([1], dtype=np.int32),
        unique_pixels_pe=np.array([0], dtype=np.int32),
        sample_to_unique_pe=np.zeros(npe, dtype=np.int32),
        zgals_sel=np.array([[z]]), dzgals_sel=np.array([[dz]]),
        wgals_sel=np.array([[1.0]]), ngals_sel=np.array([1], dtype=np.int32),
        unique_pixels_sel=np.array([0], dtype=np.int32),
        sample_to_unique_sel=np.zeros(n_sel, dtype=np.int32),
    )


def _shared_physics_multi(nEvents, nsamp=NSAMP, n_sel=NSEL):
    """GW PE/selection physics arrays (compact-view path) for ``nEvents``."""
    npe = nEvents * nsamp
    m1 = np.linspace(34.0, 42.0, npe)
    dL = np.linspace(430.0, 560.0, npe)
    chieff = np.linspace(-0.05, 0.08, npe)
    return dict(
        nEvents=nEvents, nsamp=nsamp, Ndraw=float(n_sel),
        apix=hp.nside2pixarea(NSIDE),
        m1det=jnp.asarray(m1), m2det=jnp.asarray(0.8 * m1),
        dL=jnp.asarray(dL), chieff=jnp.asarray(chieff),
        p_pe=jnp.ones(npe),
        m1detsels=jnp.linspace(34.0, 40.0, n_sel),
        m2detsels=0.8 * jnp.linspace(34.0, 40.0, n_sel),
        dLsels=jnp.linspace(430.0, 530.0, n_sel),
        chieffsels=jnp.zeros(n_sel), p_draw=jnp.ones(n_sel),
    )


def _mark_table():
    vals = 10.0 + 0.2 * np.arange(NPIX, dtype=float).reshape(NPIX, 1)
    return jnp.asarray(vals)


def _base_opts(**overrides):
    _pop_fid, prior_overrides, fixed = _pop_bits()
    kwargs = dict(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        pe_event_block=None,
        fix_cosmology=True,
        fix_population=False,
        fix_survey=True,
        prior_overrides=prior_overrides,
        fixed_parameter_values=fixed,
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _space_for(opts):
    res = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        opts.fix_cosmology,
        opts.fix_survey,
        prior_overrides=opts.prior_overrides,
        fixed_parameter_values=opts.fixed_parameter_values,
        universe_model=opts.universe_model,
        sky_model=getattr(opts, "sky_model", "isotropic"),
        mark_model=getattr(opts, "mark_model", "none"),
        mark_names=tuple(getattr(opts, "mark_names", ()) or ()),
        n_catalogs=getattr(opts, "n_catalogs", 1),
    )
    labels, lower, upper = res[0], np.asarray(res[1]), np.asarray(res[2])
    return labels, lower, upper


def _mid_coord(opts):
    _labels, lower, upper = _space_for(opts)
    return lower + 0.5 * (upper - lower)


# ---------------------------------------------------------------------------
# Cell builders: each returns (opts_factory, data).  opts_factory(**kw) makes a
# fresh opts so we can vary pe_event_block / redshift_prior_barrier per call.
# ---------------------------------------------------------------------------

def _cell_dark(nEvents):
    data = _full_sky_multi(nEvents)
    return (lambda **kw: _base_opts(**kw)), data


def _cell_spectral(nEvents):
    data = _full_sky_multi(nEvents)
    return (lambda **kw: _base_opts(universe_model="spectral_sirens", **kw)), data


def _cell_marks(nEvents):
    data = _full_sky_multi(nEvents)
    data["mark_logmstar"] = _mark_table()
    return (
        lambda **kw: _base_opts(mark_model="loglinear", mark_names=("logmstar",), **kw),
        data,
    )


def _cell_field(nEvents):
    data = _full_sky_multi(nEvents)
    # The field scope gate accepts the legacy dummy (1, NG) overdensity grid.
    data["delta_g_pix_z"] = jnp.zeros((1, NG))
    return (lambda **kw: _base_opts(catalog_sky_weighting="field", **kw)), data


def _cell_sel_batch(nEvents):
    data = _full_sky_multi(nEvents)
    return (lambda **kw: _base_opts(sel_batch_size=3, **kw)), data


def _cell_k2(nEvents):
    data = dict(_shared_physics_multi(nEvents))
    data["catalogs"] = [_bundle_multi(0.10, nEvents), _bundle_multi(0.30, nEvents)]
    return (lambda **kw: _base_opts(n_catalogs=2, **kw)), data


K1_CELLS = {
    "dark": _cell_dark,
    "spectral": _cell_spectral,
    "marks": _cell_marks,
    "field": _cell_field,
    "sel_batch": _cell_sel_batch,
}
ALL_CELLS = dict(K1_CELLS)
ALL_CELLS["k2_mixture"] = _cell_k2


def _build_ll(opts_factory, data, pe_block, barrier="auto"):
    pop_fid, _ov, fixed = _pop_bits()
    opts = opts_factory(pe_event_block=pe_block, redshift_prior_barrier=barrier)
    return make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed), opts


# ---------------------------------------------------------------------------
# (a) Exact parity: None == 1 for every feature cell.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ALL_CELLS))
def test_none_equals_one_value(name):
    opts_factory, data = ALL_CELLS[name](4)
    ll_none, opts = _build_ll(opts_factory, data, None)
    ll_one, _ = _build_ll(opts_factory, data, 1)
    coord = jnp.asarray(_mid_coord(opts))
    v_none = float(ll_none(coord))
    v_one = float(ll_one(coord))
    assert np.isfinite(v_none), (name, v_none)
    # The flattened block evaluation reduces the identical masked elements in the
    # identical per-row order as the per-event scan, so this is exact up to XLA
    # reassociation of the final reduction.  It WAS bit-for-bit while the sampler-
    # facing closure ran eagerly around the jitted core; now that the closure is
    # jitted too (factory._jit_likelihood_body), the whole call is one XLA module
    # and the unblocked (pe_event_block=None) reduction reassociates by ~2 ULP on
    # the catalog cells -- MEASURED -0.27199374534573906 vs -0.27199374534573195,
    # i.e. 2.6e-14 relative.  The pe_event_block=1 value is unchanged to the bit.
    # Same tolerance and rationale as the block=3 comparison below: 1e-12 is five
    # orders below the ~1e-9 the repo already treats as benign compile-context
    # reassociation, and eight below the >= 1e-4 a masking/ordering bug moves.
    np.testing.assert_allclose(v_none, v_one, rtol=1e-12, atol=0.0,
                               err_msg=f"{name}: None vs 1")


# ---------------------------------------------------------------------------
# (b) Remainder chunk: nEvents=7, block=3 == None == 1.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(ALL_CELLS))
def test_remainder_chunk_matches(name):
    opts_factory, data = ALL_CELLS[name](7)  # 7 = 2 full chunks of 3 + remainder 1
    coord_opts = opts_factory(pe_event_block=None)
    coord = jnp.asarray(_mid_coord(coord_opts))
    ll_none, _ = _build_ll(opts_factory, data, None)
    ll_one, _ = _build_ll(opts_factory, data, 1)
    ll_three, _ = _build_ll(opts_factory, data, 3)
    v_none = float(ll_none(coord))
    v_one = float(ll_one(coord))
    v_three = float(ll_three(coord))
    assert np.isfinite(v_none), (name, v_none)
    # None (one block of 7) reproduces the per-event scan (block=1) up to the same
    # ~ULP reassociation of the final reduction as test_none_equals_one_value.
    np.testing.assert_allclose(v_none, v_one, rtol=1e-12, atol=0.0,
                               err_msg=f"{name}: None vs 1")
    # block=3 splits 7 events into chunks of (3, 3, 1).  Each event's log Ẑ_i is
    # still computed from the identical masked samples in the identical order, so
    # the PER-EVENT values match bitwise; only the FINAL cross-event jnp.sum
    # reassociates when the (nEvents,) vector is assembled from unequal chunks (a
    # concatenate of 6+1 vs a contiguous 7), a ~1 ULP effect.  Tolerance far
    # tighter than the golden pin (1e-12).
    np.testing.assert_allclose(v_three, v_none, rtol=1e-12, atol=0.0,
                               err_msg=f"{name}: block=3 vs None")


# ---------------------------------------------------------------------------
# (a, grad) Reverse-mode parity, barrier OFF, on one spectral and one dark case.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["dark", "spectral"])
def test_grad_none_equals_one(name):
    opts_factory, data = K1_CELLS[name](4)
    ll_none, opts = _build_ll(opts_factory, data, None, barrier="off")
    ll_one, _ = _build_ll(opts_factory, data, 1, barrier="off")
    coord = jnp.asarray(_mid_coord(opts))
    g_none = np.asarray(jax.grad(lambda c: ll_none(c))(coord))
    g_one = np.asarray(jax.grad(lambda c: ll_one(c))(coord))
    assert np.all(np.isfinite(g_none)), (name, g_none)
    # The forward value is bit-identical (test_none_equals_one); the reverse-mode
    # gradient agrees to ~1 ULP because the backward pass accumulates the
    # per-event cotangents in a slightly different order (vmap-reduce vs scan).
    np.testing.assert_allclose(g_none, g_one, rtol=1e-11, atol=1e-13,
                               err_msg=name)


# ---------------------------------------------------------------------------
# (c) Bright/counterpart config keeps the scan: value independent of the block.
# ---------------------------------------------------------------------------

def _bright_data(nEvents):
    data = _full_sky_multi(nEvents)
    # One counterpart per event, all in pixel 7 at z=0.10.
    data.update(
        counterpart_pixel=7,
        counterpart_pixels=jnp.full(nEvents, 7, dtype=jnp.int32),
        counterpart_zs=jnp.full(nEvents, 0.10),
        counterpart_dzs=jnp.full(nEvents, 0.02),
        bright_siren_sky_marginalized=False,
    )
    # Bright siren PE hosts must sit in the counterpart pixel.
    data["pixels_pe"] = jnp.full(nEvents * NSAMP, 7, dtype=jnp.int32)
    return data


def test_bright_uses_scan_block_independent():
    data = _bright_data(4)
    opts_factory = lambda **kw: _base_opts(universe_model="bright_sirens", **kw)
    ll_none, opts = _build_ll(opts_factory, data, None)
    ll_one, _ = _build_ll(opts_factory, data, 1)
    ll_two, _ = _build_ll(opts_factory, data, 2)
    coord = jnp.asarray(_mid_coord(opts))
    v_none = float(ll_none(coord))
    v_one = float(ll_one(coord))
    v_two = float(ll_two(coord))
    assert np.isfinite(v_none), v_none
    # has_counterpart -> the exact per-event scan is used regardless of the
    # (inert) pe_event_block, so all three are bit-identical.
    assert v_none == v_one == v_two, (v_none, v_one, v_two)
