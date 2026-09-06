"""Exact per-sample savings in the windowed catalog KDE (perf PR-2).

Three changes, each with a numerical contract:

* ``CatalogKernelState.log_kw_eff`` fuses the sample-independent part of every
  galaxy's log kernel term at build time, so the evaluator gathers three arrays
  instead of four.  Same arithmetic up to re-association (<= 1e-12 here).
* ``auto_kde_window`` sizes the static window from the data so every sample's
  in-range block fits: the evaluator NEVER truncates, and its answer agrees
  with the full-row evaluator to the ``exp(-n_sigma^2/2)`` truncation contract
  of ``configure_catalog_kde_window``.
* The likelihood factory sorts the injections by compact pixel (cache locality
  of the row gathers) and threads the data-sized window as a static argument;
  neither moves the log-likelihood beyond floating-point re-association.
"""
import os

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap

from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams
import darksirens.redshift.catalog as C

COSMO = CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(sigma_kde=0.0):
    return SurveyParams(
        n0=1e-2, z50=1.0, w=0.3, delta=0.0, b_miss=0.0, alpha_miss=0.0,
        sigma_kde=sigma_kde,
    )


def _rows(seed=0, n_gal=600, n_max=640, z_hi=1.4):
    """Four z-sorted rows: a dense clumpy one, a sparse one, an empty one, a
    short one.  Widths span spectroscopic to mildly photometric."""
    rng = np.random.default_rng(seed)
    z = np.sort(np.concatenate([
        rng.uniform(0.02, z_hi, n_gal // 2),
        np.clip(rng.normal(0.4, 0.02, n_gal // 4), 0.02, z_hi),
        np.clip(rng.normal(0.9, 0.05, n_gal - n_gal // 2 - n_gal // 4), 0.02, z_hi),
    ]))
    zg = np.full((4, n_max), 100.0)
    dz = np.full((4, n_max), 1.0)
    wg = np.zeros((4, n_max))
    ng = np.array([n_gal, 30, 0, 3], dtype=np.int32)
    zg[0, :n_gal] = z
    dz[0, :n_gal] = rng.uniform(0.002, 0.02, n_gal)
    wg[0, :n_gal] = rng.lognormal(0.0, 0.3, n_gal)
    zg[1, :30] = np.sort(rng.uniform(0.05, 1.2, 30))
    dz[1, :30] = rng.uniform(0.002, 0.02, 30)
    wg[1, :30] = 1.0
    zg[3, :3] = np.sort(rng.uniform(0.2, 0.6, 3))
    dz[3, :3] = 0.005
    wg[3, :3] = 1.0
    return EMCatalog(
        apix=1e-3, zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(wg), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, 10)), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _samples(rng, n=1500, z_hi=1.4):
    z = jnp.asarray(np.concatenate([
        rng.uniform(0.0, z_hi + 0.1, n // 2),
        np.clip(rng.normal(0.4, 0.03, n // 4), 0.0, z_hi + 0.1),
        rng.uniform(z_hi, 2.5, n - n // 2 - n // 4),
    ]))
    pix = jnp.asarray(rng.integers(0, 4, z.shape[0]), dtype=jnp.int32)
    return z, pix


def _eval(state, cat, z, pix):
    return np.asarray(vmap(
        lambda zi, pi: C.eval_log_catalog_prior_state(zi, pi, state, cat)
    )(z, pix))


def _max_abs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if (np.isneginf(a) != np.isneginf(b)).any():
        return np.inf
    both = np.isneginf(a) & np.isneginf(b)
    with np.errstate(invalid="ignore"):
        return float(np.max(np.where(both, 0.0, np.abs(a - b))))


@pytest.fixture(autouse=True)
def _restore_window():
    yield
    C.configure_catalog_kde_window()


# ---------------------------------------------------------------------------
# log_kw_eff
# ---------------------------------------------------------------------------

def test_log_kw_eff_is_the_fused_sample_independent_term():
    cat = _rows()
    st = C.catalog_kernel_state(COSMO, _survey(0.01), cat)
    live = np.asarray(st.log_kw) > -1e29
    want = np.asarray(st.log_kw) - np.log(np.asarray(st.sig_eff)) - C._HALF_LOG_2PI
    got = np.asarray(st.log_kw_eff)
    np.testing.assert_allclose(got[live], want[live], rtol=0.0, atol=1e-13)
    # Padding keeps the EXACT -1e30 sentinel (the pin's sentinel cut relies on it).
    assert np.all(got[~live] == -1e30)


@pytest.mark.parametrize("window", [None, 64, 256])
def test_three_gather_evaluator_matches_the_historical_arithmetic(window):
    """The one-pass evaluator reproduces the historical four-gather logsumexp
    wherever the build-time offset has the f64 exponent range to hold the
    answer, and returns exactly -inf well below it.

    The reduction subtracts a BUILD-TIME per-row offset
    (``log_kw_eff_rowmax``) instead of the sample's own maximum, which is what
    removes the ``logsumexp`` amax pass.  The price is a finite offset budget
    set by where ``exp`` underflows on the backend (708.4 nats on XLA CPU,
    744.4 on CUDA), with a graded band just above the edge --
    :func:`test_the_offset_budget_is_the_backend_exp_underflow_edge` pins that.
    Agreement to 1e-12 is therefore a property of samples whose deficit stays
    below the band, which is every sample on this fixture (its per-row
    ``log_kw_eff`` spread is small, so the row max and the sample max coincide
    to well under a nat); what is pinned here is that the floor fires only
    hundreds of nats down, and never the other way round.
    """
    rng = np.random.default_rng(1)
    cat = _rows()
    z, pix = _samples(rng)
    C.configure_catalog_kde_window(window) if window else C.configure_catalog_kde_window(None)
    st = C.catalog_kernel_state(COSMO, _survey(0.01), cat)
    new = _eval(st, cat, z, pix)
    old = _eval(st._replace(log_kw_eff=None), cat, z, pix)   # four-gather arithmetic
    floored = np.isfinite(old) & np.isneginf(new)
    # The floor is one-way: nothing the two-pass form called -inf comes back.
    assert not (np.isneginf(old) & np.isfinite(new)).any()
    peak = np.max(old[np.isfinite(old)])
    assert np.all(old[floored] < peak - 700.0)
    assert _max_abs_delta(new[~floored], old[~floored]) < 1e-12


def test_one_pass_offset_is_the_row_max_of_the_served_log_kw_eff():
    """``log_kw_eff_rowmax`` IS the maximum of the array the evaluator gathers.

    The offset is not a correctness-critical value -- any offset within the
    f64 exponent range returns the same number -- but a STALE one silently
    moves the underflow edge, so it is pinned against the served array on every
    builder, including the pinned quadrature at a shifted H0 (where the offset
    must carry the same scalar ``3 ln(H0/H0_ref)`` as ``log_kw_eff`` itself).
    """
    cat = _rows()
    survey = _survey(0.01)
    st = C.catalog_kernel_state(COSMO, survey, cat)
    lkwe = np.asarray(st.log_kw_eff)
    want = np.max(lkwe, axis=1)
    occupied = np.asarray(cat.ngals) > 0
    np.testing.assert_array_equal(
        np.asarray(st.log_kw_eff_rowmax)[occupied], want[occupied])
    # An all-padding row is clamped to 0.0, not left at the -1e30 sentinel:
    # with -1e30 as the offset the padding exponents would be 0 and the row
    # would return a finite ~-1e30 + log(N_max) instead of -inf.
    assert np.all(np.asarray(st.log_kw_eff_rowmax)[~occupied] == 0.0)
    assert np.isneginf(
        float(C.eval_log_catalog_prior_state(0.5, 2, st, cat)))

    # The pin shifts the offset by exactly the scalar it shifts log_kw_eff by.
    pin = C.build_pinned_kernel_quadrature(COSMO, survey, cat)
    assert pin.log_kw_eff_rowmax is not None and pin.inv_sig_eff is not None
    live = C.catalog_kernel_state(
        COSMO._replace(H0=90.0), survey, cat, pinned=pin)
    served = np.asarray(live.log_kw_eff)
    got = np.asarray(live.log_kw_eff_rowmax)
    np.testing.assert_allclose(
        got[occupied], np.max(served, axis=1)[occupied],
        rtol=0.0, atol=C.KERNEL_PIN_TOL)
    # Every exponent the evaluator forms is <= 0 by construction at any H0.
    assert np.all(served <= got[:, None])


def test_inv_sig_eff_is_the_reciprocal_on_real_slots_and_zero_on_padding():
    cat = _rows()
    st = C.catalog_kernel_state(COSMO, _survey(0.01), cat)
    live = np.asarray(st.log_kw_eff) > -1e29
    inv = np.asarray(st.inv_sig_eff)
    np.testing.assert_array_equal(
        inv[live], (1.0 / np.asarray(st.sig_eff))[live])
    # Padding carries 0.0 so its deviate is exactly 0 and the exponent is the
    # bare -1e30 sentinel, whatever the padding widths are.
    assert np.all(inv[~live] == 0.0)


def test_every_builder_populates_the_one_pass_leaves():
    """A builder that forgets a leaf silently falls back to the two-pass form
    and loses the 27 ms/call, with every other test still green.  Pin all four
    construction sites, including the marked-host one the evaluator shares.
    """
    cat = _rows()
    survey = _survey(0.01)

    st = C.catalog_kernel_state(COSMO, survey, cat)
    assert st.log_kw_eff_rowmax is not None and st.inv_sig_eff is not None

    pin = C.build_pinned_kernel_quadrature(COSMO, survey, cat)
    assert pin.log_kw_eff_rowmax is not None and pin.inv_sig_eff is not None

    live = C.catalog_kernel_state(
        COSMO._replace(H0=90.0), survey, cat, pinned=pin)
    assert live.log_kw_eff_rowmax is not None and live.inv_sig_eff is not None

    log_h = jnp.asarray(
        np.where(np.asarray(cat.wgals) > 0.0, -0.5, -np.inf))
    marked, _ = C.marked_catalog_kernel_state(COSMO, survey, cat, log_h)
    assert marked.log_kw_eff_rowmax is not None
    assert marked.inv_sig_eff is not None
    served = np.asarray(marked.log_kw_eff)
    occupied = np.asarray(cat.ngals) > 0
    got = np.asarray(marked.log_kw_eff_rowmax)
    np.testing.assert_array_equal(
        got[occupied], np.max(served, axis=1)[occupied])
    assert np.all(served[occupied] <= got[occupied][:, None])


def _has_reduce_max(state, cat):
    """Does the traced evaluator still contain a row-wide max reduction?"""
    jx = jax.make_jaxpr(
        lambda z, p: C.eval_log_catalog_prior_state(z, p, state, cat)
    )(jnp.asarray(0.5), jnp.int32(0))
    return "reduce_max" in str(jx)


@pytest.mark.parametrize("window", [None, 256])
def test_the_fast_one_pass_branch_is_actually_taken(window):
    """The perf contract as a TEST, not a comment.

    The whole win is the disappearance of the ``logsumexp`` amax pass (and with
    it the (N_samples, N_max) f64 intermediate XLA materialised between the two
    reductions).  With the leaves present the traced evaluator must contain NO
    ``reduce_max``; drop the offset and it must come back.
    """
    cat = _rows()
    C.configure_catalog_kde_window(window)
    st = C.catalog_kernel_state(COSMO, _survey(0.01), cat)
    assert not _has_reduce_max(st, cat)
    assert _has_reduce_max(st._replace(log_kw_eff_rowmax=None), cat)


def _backend_exp_underflow_edge():
    """Nats below zero at which ``jnp.exp`` reaches 0 on the LIVE backend.

    ``ln(smallest normal f64) = 708.396`` where subnormal results are flushed
    (the XLA CPU backend), ``ln(smallest subnormal) = 744.440`` where they are
    not (CUDA).  Measured, not assumed, so the pin follows the backend.
    """
    flushes = float(jnp.exp(jnp.asarray(-709.0))) == 0.0
    return 708.396 if flushes else 744.440


def test_the_offset_budget_is_the_backend_exp_underflow_edge():
    """The build-time offset buys a FINITE budget, and the edge is graded.

    One galaxy in the row, so a sample at redshift ``z`` sits exactly
    ``0.5 * ((z - z_gal) / sig_eff)**2`` nats below the row's maximum and the
    deficit is a dial.  Well inside the budget the one-pass form is exact;
    well past it the row sums to zero and returns -inf.  Between them the
    finite value degrades with the subnormal mantissa (MEASURED on an H100 NVL:
    relative drift 4e-15 at a deficit of 720 nats, 2e-12 at 725, 0.25 nats at
    744), which is why the docs state a graded band rather than a clean floor.
    Nothing downstream can see it: the sample's weight in the caller's
    ``logaddexp`` is ``exp(-708) = 0`` whatever value comes back.
    """
    n_max = 4
    zg = np.full((1, n_max), 100.0)
    dz = np.full((1, n_max), 1.0)
    wg = np.zeros((1, n_max))
    zg[0, 0], dz[0, 0], wg[0, 0] = 0.5, 0.01, 1.0
    cat = EMCatalog(
        apix=1e-3, zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(wg), ngals=jnp.asarray(np.array([1], np.int32)),
        delta_g_pix_z=jnp.zeros((1, 10)), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    C.configure_catalog_kde_window(None)
    st = C.catalog_kernel_state(COSMO, _survey(0.0), cat)
    sig = float(np.asarray(st.sig_eff)[0, 0])

    edge = _backend_exp_underflow_edge()
    # 20 nats of guard on each side of the measured band (the drift crosses
    # 1e-12 about 20 nats below the CUDA edge; the CPU term-drop band is
    # narrower still).
    inside = np.array([1.0, 100.0, 400.0, edge - 20.0])
    outside = np.array([edge + 1.0, edge + 20.0, edge + 100.0])
    deficits = np.concatenate([inside, outside])
    z = jnp.asarray(0.5 + sig * np.sqrt(2.0 * deficits))
    pix = jnp.zeros(z.shape, dtype=jnp.int32)

    new = _eval(st, cat, z, pix)
    old = _eval(st._replace(log_kw_eff_rowmax=None), cat, z, pix)
    n_in = inside.size
    # Inside the budget: exact to re-association, and the two-pass reference is
    # finite there (the reference has no offset budget at all).
    assert np.all(np.isfinite(old[:n_in])) and np.all(np.isfinite(new[:n_in]))
    assert _max_abs_delta(new[:n_in], old[:n_in]) < 1e-12
    # Past the edge: exactly -inf, and the two-pass form still finite -- i.e.
    # the floor is the one-pass form's, not the arithmetic's.
    assert np.all(np.isneginf(new[n_in:]))
    assert np.all(np.isfinite(old[n_in:]))
    # And the value that is lost is unobservable: it is more than 700 nats
    # under the row's peak, so exp() of it is 0 in the caller's logaddexp.
    peak = float(np.max(old))
    assert np.all(old[n_in:] < peak - 700.0)


def test_one_pass_state_leaves_gate_the_fast_form():
    """Dropping EITHER new leaf falls back to the historical two-pass form.

    The admissibility condition is the presence of the leaves, not the H0 pin:
    the same branch serves pinned, unpinned, marked, volume-weighted and
    windowed states.
    """
    rng = np.random.default_rng(7)
    cat = _rows()
    z, pix = _samples(rng, n=200)
    C.configure_catalog_kde_window(None)
    st = C.catalog_kernel_state(COSMO, _survey(0.01), cat)
    ref = _eval(st._replace(log_kw_eff=None), cat, z, pix)
    for drop in ({"log_kw_eff_rowmax": None}, {"inv_sig_eff": None},
                 {"log_kw_eff_rowmax": None, "inv_sig_eff": None}):
        got = _eval(st._replace(**drop), cat, z, pix)
        # The fused three-gather logsumexp, which agrees with the four-gather
        # arithmetic to re-association and has NO underflow floor.
        assert not np.isneginf(got[np.isfinite(ref)]).any()
        assert _max_abs_delta(got, ref) < 1e-12


def test_pinned_state_carries_the_fused_array_with_the_shift():
    cat = _rows()
    survey = _survey(0.01)
    pin = C.build_pinned_kernel_quadrature(COSMO, survey, cat)
    assert pin.log_kw_eff is not None
    live = C.catalog_kernel_state(
        COSMO._replace(H0=90.0), survey, cat, pinned=pin)
    ref = C.catalog_kernel_state(COSMO._replace(H0=90.0), survey, cat)
    mask = np.asarray(ref.log_kw) > -1e29
    np.testing.assert_allclose(
        np.asarray(live.log_kw_eff)[mask], np.asarray(ref.log_kw_eff)[mask],
        rtol=0.0, atol=1e-9)
    assert np.all(np.asarray(live.log_kw_eff)[~mask] == -1e30)


# ---------------------------------------------------------------------------
# Data-sized window: never truncates, agrees with the full row
# ---------------------------------------------------------------------------

def test_auto_kde_window_covers_the_widest_block_and_is_granular():
    cat = _rows()
    for sig_kde in (0.0, 0.02, 0.05):
        need = C.recommended_kde_window(
            np.asarray(cat.zgals), np.asarray(cat.ngals), np.asarray(cat.dzgals),
            sig_kde, n_sigma=C._KDE_WINDOW_NSIGMA)
        w = C.auto_kde_window([cat], sig_kde)
        assert w % 64 == 0 and w >= need + 1 and w < need + 65


@pytest.mark.parametrize("sig_kde", [0.0, 0.02, 0.05])
def test_auto_window_matches_the_full_row_to_the_truncation_contract(sig_kde):
    rng = np.random.default_rng(2)
    cat = _rows()
    z, pix = _samples(rng)
    w = C.auto_kde_window([cat], sig_kde)
    # Only meaningful when the window is shorter than the row (else full-row).
    assert w < cat.zgals.shape[1], (w, cat.zgals.shape)
    st_w = C.catalog_kernel_state(COSMO, _survey(sig_kde), cat, kde_window=w)
    assert int(st_w.kde_window) == w
    C.configure_catalog_kde_window(None)
    st_full = C.catalog_kernel_state(COSMO, _survey(sig_kde), cat)
    full = _eval(st_full, cat, z, pix)
    # The state's own window overrides the (disabled) process-global one.
    win = _eval(st_w, cat, z, pix)
    assert (np.isneginf(win) == np.isneginf(full)).all()
    # Fewer terms can only lower a logsumexp (up to rounding).
    finite = np.isfinite(full)
    assert np.all(win[finite] <= full[finite] + 1e-10)
    # The truncation contract holds wherever the catalog prior is not itself
    # negligible: a sample many row-max widths beyond every galaxy has
    # p_cat ~ exp(-hundreds), set by whichever far galaxy is widest, and the
    # missing branch owns those samples in the assembled prior.
    live = finite & (full > np.max(full[finite]) - 60.0)
    assert live.sum() > 500
    assert np.max(np.abs(win[live] - full[live])) < 1e-10


def test_state_window_overrides_the_process_global_one():
    """A state built for window W keeps W after the global is reconfigured."""
    rng = np.random.default_rng(3)
    cat = _rows()
    z, pix = _samples(rng, n=300)
    calls = []
    orig = C._sorted_row_window_start

    def spy(zgals, pix_i, z_i, n_real, window):
        calls.append(int(window))
        return orig(zgals, pix_i, z_i, n_real, window)

    st = C.catalog_kernel_state(COSMO, _survey(0.0), cat, kde_window=128)
    C.configure_catalog_kde_window(512)
    C._sorted_row_window_start = spy
    try:
        _eval(st, cat, z, pix)
    finally:
        C._sorted_row_window_start = orig
    assert calls and set(calls) == {128}


def test_auto_window_at_or_above_the_row_takes_the_full_row_path():
    rng = np.random.default_rng(4)
    cat = _rows()
    z, pix = _samples(rng, n=300)
    n_max = int(cat.zgals.shape[1])
    st = C.catalog_kernel_state(COSMO, _survey(0.0), cat, kde_window=n_max)
    calls = []
    orig = C._sorted_row_window_start
    C._sorted_row_window_start = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        got = _eval(st, cat, z, pix)
    finally:
        C._sorted_row_window_start = orig
    assert not calls
    C.configure_catalog_kde_window(None)
    ref = _eval(C.catalog_kernel_state(COSMO, _survey(0.0), cat), cat, z, pix)
    assert _max_abs_delta(got, ref) == 0.0


def test_auto_window_returns_none_without_rows():
    assert C.auto_kde_window([EMCatalog(apix=1.0, zgals=None, dzgals=None, wgals=None,
                                        ngals=None, delta_g_pix_z=None,
                                        dN_obs_kde=None, pixel_to_cache_idx=None)],
                             0.05) is None


# ---------------------------------------------------------------------------
# Factory: injections sorted by pixel, data-sized window threaded statically
# ---------------------------------------------------------------------------

def _factory_fixture(seed=5, n_sel=64, n_gal=12):
    """A tiny flat single-catalog run over nside=1 with several galaxies per
    pixel and injections scattered over the sky (draw order)."""
    import healpy as hp
    from types import SimpleNamespace

    from darksirens.gw.populations import pop_model_prior_parser
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.redshift import zgrid

    rng = np.random.default_rng(seed)
    npix = hp.nside2npix(1)
    zg = np.sort(rng.uniform(0.02, 0.6, (npix, n_gal)), axis=1)
    dz = rng.uniform(0.003, 0.02, (npix, n_gal))
    wg = np.ones((npix, n_gal))
    ng = np.full(npix, n_gal, dtype=np.int32)
    nsamp = 4
    pixels_pe = np.array([7, 7, 3, 3], dtype=np.int32)
    pixels_sel = rng.integers(0, npix, n_sel).astype(np.int32)
    assert not np.all(np.diff(pixels_sel) >= 0)

    def _dirs(n, phase):
        ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
        return (jnp.asarray(np.cos(ang) * np.sqrt(0.75)),
                jnp.asarray(np.sin(ang) * np.sqrt(0.75)), jnp.full(n, 0.5))

    nx_pe, ny_pe, nz_pe = _dirs(nsamp, 0.1)
    nx_sel, ny_sel, nz_sel = _dirs(n_sel, 0.7)
    data = {
        "nEvents": 1, "nsamp": nsamp, "Ndraw": float(n_sel),
        "apix": hp.nside2pixarea(1), "nside": 1, "n_pix_catalog": npix,
        "zgals": zg, "dzgals": dz, "wgals": wg, "ngals_catalog": ng,
        "zgals_catalog": zg, "dzgals_catalog": dz, "wgals_catalog": wg,
        "delta_g_pix_z": jnp.zeros((npix, len(zgrid))),
        "m1det": jnp.array([36.0, 38.0, 35.0, 37.0]),
        "m2det": jnp.array([28.8, 30.4, 28.0, 29.6]),
        "dL": jnp.array([460.0, 500.0, 900.0, 1200.0]),
        "chieff": jnp.array([0.0, 0.02, 0.01, -0.01]),
        "p_pe": jnp.ones(nsamp), "pixels_pe": jnp.asarray(pixels_pe),
        "nx_pe": nx_pe, "ny_pe": ny_pe, "nz_pe": nz_pe,
        "m1detsels": jnp.asarray(rng.uniform(30.0, 45.0, n_sel)),
        "m2detsels": jnp.asarray(rng.uniform(20.0, 30.0, n_sel)),
        "dLsels": jnp.asarray(rng.uniform(300.0, 2500.0, n_sel)),
        "chieffsels": jnp.zeros(n_sel), "p_draw": jnp.ones(n_sel),
        "pixels_sel": jnp.asarray(pixels_sel),
        "nx_sel": nx_sel, "ny_sel": ny_sel, "nz_sel": nz_sel,
    }
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    opts = SimpleNamespace(
        universe_model="dark_sirens", pop_model="powerlaw+peak", sampler="dynesty",
        fix_cosmology=False, fix_de=True, fix_population=True, fix_survey=True,
        fixed_cosmology=False, fixed_de=True,
        prior_overrides=None, fixed_parameter_values={"Om0": 0.3075},
        use_LSS=False, counterpart=None, lss_completions=None,
        sel_batch_size=None, pe_event_block=None, sky_model="isotropic",
        mark_model="none", mark_names=(), catalog_sky_weighting="conditional",
        n_catalogs=1, kde_window=None, drop_full_catalog=False,
    )
    return opts, data, jnp.asarray(pop_fid), {"Om0": 0.3075}


def _build(monkeypatch, sort=True, kde_window=None):
    import darksirens.likelihood.factory as F

    opts, data, pop_fid, fixed = _factory_fixture()
    opts.kde_window = kde_window
    if not sort:
        monkeypatch.setattr(F, "_injection_pixel_order", lambda pixels: None)
    like = F.make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    return like


def test_factory_sorts_injections_by_pixel_and_the_likelihood_is_unchanged(monkeypatch):
    sorted_like = _build(monkeypatch, sort=True)
    gw_sel = sorted_like.operands[2]
    p = np.asarray(gw_sel.pixels)
    assert np.all(p[1:] >= p[:-1]), "injections are not pixel-sorted"
    # The selection view's sample->row map moved with them.
    assert np.array_equal(np.asarray(sorted_like.operands[3].sample_to_unique_idx), p)
    # Every per-injection column moved by the SAME permutation: the (m1, dL)
    # pairs are a permutation of the draw-order pairs.
    plain_like = _build(monkeypatch, sort=False)
    gw0 = plain_like.operands[2]
    pairs_sorted = np.lexsort((np.asarray(gw_sel.dL), np.asarray(gw_sel.m1det)))
    pairs_plain = np.lexsort((np.asarray(gw0.dL), np.asarray(gw0.m1det)))
    np.testing.assert_array_equal(np.asarray(gw_sel.m1det)[pairs_sorted],
                                  np.asarray(gw0.m1det)[pairs_plain])
    np.testing.assert_array_equal(np.asarray(gw_sel.pixels)[pairs_sorted],
                                  np.asarray(gw0.pixels)[pairs_plain])
    for coord in (jnp.array([67.0]), jnp.array([80.0])):
        a = float(sorted_like(coord))
        b = float(plain_like(coord))
        assert np.isfinite(a) and abs(a - b) <= 1e-12 * max(1.0, abs(b))


def test_factory_sizes_the_window_from_the_data_and_explicit_wins(monkeypatch):
    like = _build(monkeypatch)
    # 12 galaxies per row: the data-sized window is one granule, and it exceeds
    # the row length, so the evaluator takes the full-row path.
    assert like.kde_window == 64
    like2 = _build(monkeypatch, kde_window=8)
    assert like2.kde_window == 8
    like3 = _build(monkeypatch, kde_window=0)
    assert like3.kde_window is None


def test_auto_window_warns_when_it_exceeds_the_old_default(recwarn):
    """A dense row whose in-range block exceeds 1024 galaxies must be reported:
    the former fixed window truncated it silently."""
    import darksirens.likelihood.factory as F
    from types import SimpleNamespace

    rng = np.random.default_rng(6)
    n = 1500
    zg = np.sort(rng.uniform(0.2, 0.3, (1, n)), axis=1)
    cat = EMCatalog(apix=1.0, zgals=jnp.asarray(zg), dzgals=jnp.full((1, n), 0.05),
                    wgals=jnp.ones((1, n)), ngals=jnp.asarray([n], dtype=jnp.int32),
                    delta_g_pix_z=None, dN_obs_kde=None, pixel_to_cache_idx=None)
    dec = SimpleNamespace(sampled_labels=("H0",), pop_labels=())
    monkeypatch_target = F._reference_params
    F._reference_params = lambda d, k: (None, (_survey(0.0),))
    try:
        opts = SimpleNamespace(kde_window=None, prior_overrides=None)
        w = F._resolve_kde_window(opts, dec, (cat,), 1)
    finally:
        F._reference_params = monkeypatch_target
    assert w >= n  # never truncates: the whole row is in range
    assert any("TRUNCATED" in str(x.message) for x in recwarn.list)


def test_sigma_kde_upper_bound_uses_the_prior_when_sampled():
    import darksirens.likelihood.factory as F
    from types import SimpleNamespace

    surveys = (_survey(0.003),)
    fixed = SimpleNamespace(sampled_labels=("H0",))
    assert F._sigma_kde_upper_bound(SimpleNamespace(prior_overrides=None), fixed, surveys) == 0.003
    sampled = SimpleNamespace(sampled_labels=("H0", "sigma_kde"))
    assert F._sigma_kde_upper_bound(SimpleNamespace(prior_overrides=None), sampled, surveys) == 0.05
    assert F._sigma_kde_upper_bound(
        SimpleNamespace(prior_overrides={"sigma_kde": [0.0, 0.02]}), sampled, surveys) == 0.02


def test_sigma_kde_upper_bound_ignores_the_coordinate_placeholder_of_a_sampled_label():
    """``_reference_params`` decodes the sampled block at a coordinate
    PLACEHOLDER (0.5), which for a sampled ``sigma_kde`` is 10x the prior's
    upper bound.  The bound must come from the prior, never from that value:
    with it, the data-sized window overshot ``N_max`` and windowing switched
    itself OFF for every run that samples the survey block."""
    import darksirens.likelihood.factory as F
    from darksirens.inference.parameters import ParameterDecoder
    from types import SimpleNamespace

    def _decoder(sampled):
        return ParameterDecoder(
            sampled_labels=tuple(sampled),
            fixed_parameter_values={"Om0": 0.3075, "delta": 0.94},
            pop_labels=(), pop_params_fid=(),
            complete_empty_pixel_policy=0, z_depths=(0.3,),
        )

    dec = _decoder(("H0", "log10n0", "delta", "sigma_kde"))
    _, surveys = F._reference_params(dec, 1)
    assert float(surveys[0].sigma_kde) == 0.5      # the placeholder, as decoded
    opts = SimpleNamespace(prior_overrides=None)
    assert F._sigma_kde_upper_bound(opts, dec, surveys) == 0.05
    # A fixed catalog beside a sampled one still contributes its fixed value.
    dec2 = _decoder(("H0", "sigma_kde_c2"))
    surveys2 = (_survey(0.08), _survey(0.5))
    assert F._sigma_kde_upper_bound(opts, dec2, surveys2) == 0.08
    surveys3 = (_survey(0.5), _survey(0.01))
    dec3 = _decoder(("H0", "sigma_kde"))
    assert F._sigma_kde_upper_bound(opts, dec3, surveys3) == 0.05


def test_static_int_equality_and_window_validation():
    assert C.StaticInt(64) == 64 and C.StaticInt(64) == C.StaticInt(64)
    assert not (C.StaticInt(64) == "x")             # no ValueError
    assert C.StaticInt(64) != None                   # noqa: E711
    assert C._static_window(None) is None
    assert int(C._static_window(2)) == 2
    with pytest.raises(ValueError):
        C._static_window(1)
    with pytest.raises(ValueError):
        C.catalog_kernel_state(
            CosmoParams(H0=70.0, Om0=0.3, w0=-1.0, wa=0.0), _survey(),
            _rows(), kde_window=0)


def test_auto_kde_window_refuses_rows_without_widths_and_accepts_missing_counts():
    from types import SimpleNamespace
    cat = _rows()
    zg, dz, ng = cat.zgals, cat.dzgals, cat.ngals
    with pytest.raises(ValueError):
        C.auto_kde_window([SimpleNamespace(zgals=zg, dzgals=None, ngals=ng)], 0.0)
    # No counts: every slot is a galaxy; the padding at z=100 is far from
    # everything, so the answer matches the counted view on the real rows.
    full = C.auto_kde_window([SimpleNamespace(zgals=zg, dzgals=dz, ngals=None)], 0.0)
    counted = C.auto_kde_window([SimpleNamespace(zgals=zg, dzgals=dz, ngals=ng)], 0.0)
    assert full is not None and counted is not None and full >= counted


def test_ragged_short_row_is_not_truncated_by_a_row_length_cap():
    """A window as long as a SHORT row is not that row: centred on a sample
    past the row's support it starts at ``n - W//2`` and drops the front half.
    2000-galaxy spectroscopic row beside a 300-galaxy photo-z clump (every
    galaxy of the clump within one width of every other): the capped rule gave
    W = 300 and -0.53 nats at z = 0.36; uncapped it is 2 x 300 and exact."""
    n_max = 2000
    zg = np.full((2, n_max), 100.0); dz = np.full((2, n_max), 1.0); wg = np.zeros((2, n_max))
    ng = np.array([2000, 300], dtype=np.int32)
    rng = np.random.default_rng(0)
    zg[0, :2000] = np.sort(rng.uniform(0.0, 2.0, 2000)); dz[0, :2000] = 0.001; wg[0, :2000] = 1.0
    zg[1, :300] = np.sort(rng.uniform(0.30, 0.35, 300)); dz[1, :300] = 0.05; wg[1, :300] = 1.0
    cat = EMCatalog(apix=1e-3, zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz),
                    wgals=jnp.asarray(wg), ngals=jnp.asarray(ng),
                    delta_g_pix_z=jnp.zeros((1, 10)), dN_obs_kde=None, pixel_to_cache_idx=None)
    assert C.recommended_kde_window(zg, ng, dz, 0.0, n_sigma=C._KDE_WINDOW_NSIGMA) == 600
    W = C.auto_kde_window([cat], 0.0)
    assert 600 < W < n_max                      # windowing stays armed
    cosmo = CosmoParams(H0=70.0, Om0=0.3, w0=-1.0, wa=0.0)
    zs = jnp.asarray([0.36, 0.355, 0.34, 0.30, 0.28, 1.0])
    px = jnp.asarray([1, 1, 1, 1, 1, 0], dtype=jnp.int32)
    st_w = C.catalog_kernel_state(cosmo, _survey(), cat, kde_window=W)
    C.configure_catalog_kde_window(None)
    try:
        st_f = C.catalog_kernel_state(cosmo, _survey(), cat)
    finally:
        C.configure_catalog_kde_window()
    win = vmap(lambda z, p: C.eval_log_catalog_prior_state(z, p, st_w, cat))(zs, px)
    full = vmap(lambda z, p: C.eval_log_catalog_prior_state(z, p, st_f, cat))(zs, px)
    np.testing.assert_allclose(np.asarray(win), np.asarray(full), rtol=0, atol=1e-9)


def test_resolve_kde_window_warns_when_a_pinned_window_truncates():
    import darksirens.likelihood.factory as F
    from darksirens.inference.parameters import ParameterDecoder
    from types import SimpleNamespace
    dec = ParameterDecoder(
        sampled_labels=("H0",), fixed_parameter_values={"Om0": 0.3075, "sigma_kde": 0.05},
        pop_labels=(), pop_params_fid=(), complete_empty_pixel_policy=0, z_depths=(0.3,),
    )
    cat = _rows()
    auto = F._resolve_kde_window(SimpleNamespace(kde_window=None, prior_overrides=None), dec, [cat], 1)
    assert auto is not None and auto >= 2
    with pytest.warns(RuntimeWarning, match="below the data-sized window"):
        assert F._resolve_kde_window(
            SimpleNamespace(kde_window=2, prior_overrides=None), dec, [cat], 1) == 2
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert F._resolve_kde_window(
            SimpleNamespace(kde_window=auto, prior_overrides=None), dec, [cat], 1) == auto
        assert F._resolve_kde_window(
            SimpleNamespace(kde_window=0, prior_overrides=None), dec, [cat], 1) is None


def _brute_recommended_kde_window(z, ng, dz, sigma_kde_max, n_sigma):
    """The pre-prune reference: every row, in row order."""
    worst = 0
    for r in range(z.shape[0]):
        n = int(ng[r])
        if n < 1:
            continue
        zr = np.sort(z[r, :n])
        sig_max = float(np.max(np.sqrt(dz[r, :n] ** 2 + float(sigma_kde_max) ** 2)))
        width = float(n_sigma) * max(sig_max, C.SIGMA_EFF_FLOOR)
        idx = np.arange(n)
        right = np.searchsorted(zr, zr + width, side="right") - idx
        left = idx - np.searchsorted(zr, zr - width, side="left") + 1
        worst = max(worst, 2 * int(max(np.max(right), np.max(left))))
    return worst


@pytest.mark.parametrize("sigma_kde_max", [0.0, 0.003, 0.05])
@pytest.mark.parametrize("n_sigma", [2.0, 6.0, 8.0])
def test_descending_scan_early_exit_is_bit_identical_to_the_full_scan(sigma_kde_max, n_sigma):
    """The prune only reorders visitation and stops where 2n <= worst bounds
    every remaining row (one_sided <= n); the returned max is unchanged."""
    rng = np.random.default_rng(7)
    n_rows, n_max = 40, 120
    zg = np.full((n_rows, n_max), 100.0)
    dz = np.full((n_rows, n_max), 1.0)
    ng = rng.integers(0, n_max + 1, size=n_rows).astype(np.int32)
    ng[3] = 0                                   # an empty row
    ng[5] = n_max                               # a full row
    for r in range(n_rows):
        n = int(ng[r])
        if n:
            zg[r, :n] = np.sort(rng.uniform(0.0, 0.4 + 2.0 * (r % 3), n))
            dz[r, :n] = rng.uniform(0.001, 0.06, n)
    want = _brute_recommended_kde_window(zg, ng, dz, sigma_kde_max, n_sigma)
    got = C.recommended_kde_window(zg, ng, dz, sigma_kde_max, n_sigma=n_sigma)
    assert got == want


def _two_row_loose_bound_fixture():
    """A catalog whose maximum comes from a row that a LOOSE break would skip.

    Row 0 holds 10 galaxies one unit apart at needle-thin widths, so its
    one-sided count is 1 and it contributes 2.  Row 1 holds 2 COINCIDENT
    galaxies, one-sided 2, contributing 4.  Scanned in descending count order
    row 1 is reached with ``worst == 2``, and only the sound threshold
    ``2 * n <= worst`` (4 <= 2, false) still visits it; the loose
    ``n <= worst`` (2 <= 2, true) stops there and returns 2 -- a window half
    the size the catalog needs.
    """
    zg = np.zeros((2, 10))
    dz = np.full((2, 10), 1e-6)
    zg[0, :10] = np.arange(10, dtype=float)
    zg[1, :2] = 0.5
    ng = np.array([10, 2], dtype=np.int32)
    return zg, ng, dz


def test_break_threshold_is_twice_the_row_count_not_the_row_count():
    """Pins the admissibility predicate the bit-identity argument rests on."""
    zg, ng, dz = _two_row_loose_bound_fixture()
    want = _brute_recommended_kde_window(zg, ng, dz, 0.0, 6.0)
    assert want == 4                       # the coincident pair sets the max
    assert C.recommended_kde_window(zg, ng, dz, 0.0, n_sigma=6.0) == want
    # The densest row alone would give 2: the prune must NOT stop at it.
    assert _brute_recommended_kde_window(
        zg[:1], ng[:1], dz[:1], 0.0, 6.0) == 2


def test_recommended_kde_window_is_empty_and_all_zero_safe():
    zg = np.zeros((0, 5)); dz = np.zeros((0, 5)); ng = np.zeros((0,), dtype=np.int32)
    assert C.recommended_kde_window(zg, ng, dz, 0.01) == 0
    zg = np.zeros((3, 5)); dz = np.ones((3, 5)); ng = np.zeros((3,), dtype=np.int32)
    assert C.recommended_kde_window(zg, ng, dz, 0.01) == 0
    # Trailing empty rows end the descending scan; the answer is unchanged.
    zg2, ng2, dz2 = _two_row_loose_bound_fixture()
    zg3 = np.concatenate([zg2, np.zeros((4, 10))])
    dz3 = np.concatenate([dz2, np.full((4, 10), 1e-6)])
    ng3 = np.concatenate([ng2, np.zeros(4, dtype=np.int32)])
    assert C.recommended_kde_window(zg3, ng3, dz3, 0.0, n_sigma=6.0) == 4


def test_recommended_kde_window_refuses_a_mismatched_ngals():
    """Counts are read by row index, so a short ngals must fail, not truncate."""
    zg, ng, dz = _two_row_loose_bound_fixture()
    with pytest.raises(ValueError, match="ngals must be"):
        C.recommended_kde_window(zg, ng[:1], dz, 0.0, n_sigma=6.0)
    with pytest.raises(ValueError, match="ngals must be"):
        C.recommended_kde_window(zg, np.array([], dtype=np.int32), dz, 0.0)
    with pytest.raises(ValueError, match="ngals must be"):
        C.recommended_kde_window(zg, np.array(10), dz, 0.0)
    with pytest.raises(ValueError, match="ngals must be"):
        C.recommended_kde_window(zg, ng.reshape(2, 1), dz, 0.0)


def test_auto_kde_window_dedups_aliased_views_without_changing_the_answer(monkeypatch):
    """The flat-union path binds the SAME arrays to the PE and selection
    views; the result is a max over views, so the repeat is scanned once."""
    from types import SimpleNamespace
    cat = _rows()
    view = SimpleNamespace(zgals=np.asarray(cat.zgals), dzgals=np.asarray(cat.dzgals),
                           ngals=np.asarray(cat.ngals))
    calls = []
    real = C.recommended_kde_window
    monkeypatch.setattr(C, "recommended_kde_window",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    one = C.auto_kde_window([view], 0.003)
    assert len(calls) == 1
    calls.clear()
    two = C.auto_kde_window([view, view], 0.003)
    assert two == one
    assert len(calls) == 1                       # the alias was not rescanned
    # A DISTINCT view with the same contents is still scanned.
    calls.clear()
    other = SimpleNamespace(zgals=view.zgals.copy(), dzgals=view.dzgals.copy(),
                            ngals=view.ngals.copy())
    assert C.auto_kde_window([view, other], 0.003) == one
    assert len(calls) == 2


def test_a_view_missing_widths_is_refused_wherever_it_sits():
    """A widths-less view is refused whatever its position in the list.

    Not an ordering pin on the dedup: the key carries ``id(dzgals)``, so a
    ``dzgals=None`` view can never collide with one that carries widths and the
    refusal would fire on its first occurrence even if the dedup ran first.
    """
    from types import SimpleNamespace
    cat = _rows()
    zg, dz, ng = np.asarray(cat.zgals), np.asarray(cat.dzgals), np.asarray(cat.ngals)
    good = SimpleNamespace(zgals=zg, dzgals=dz, ngals=ng)
    bad = SimpleNamespace(zgals=zg, dzgals=None, ngals=ng)
    with pytest.raises(ValueError):
        C.auto_kde_window([good, bad], 0.003)
    with pytest.raises(ValueError):
        C.auto_kde_window([bad, good], 0.003)
