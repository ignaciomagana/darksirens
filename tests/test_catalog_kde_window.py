"""Windowed per-sample catalog KDE (issue #280).

The windowed evaluator must be indistinguishable from the full-row one at
likelihood precision ACROSS THE SAMPLED sigma_kde PRIOR ([0, 0.05] in
darksirens/inference/prior.py), not just at the fiducial kernel width; empty
and all-padding windows must reduce through the same
``_logsumexp_neginf_safe`` -inf path bit-for-bit, with finite gradients (the
NaN-poisoning contract that broke NUTS historically); and the load-time
per-row z-sort invariant must permute every co-indexed per-galaxy array
through the same permutation.
"""

import os
import tempfile

import h5py
import numpy as np
import jax
import jax.experimental
import jax.numpy as jnp
from jax import vmap
import pytest

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.redshift.catalog import (
    catalog_kernel_state,
    marked_catalog_kernel_state,
    eval_log_catalog_prior_state,
    configure_catalog_kde_window,
    recommended_kde_window,
)
import darksirens.catalogs.io as catalogs_io
from darksirens.catalogs.io import (
    ROW_Z_SORT_INVARIANT_ERROR,
    _row_z_sort_order,
    _row_z_sort_order_device,
    _sort_survey_rows_by_z_device,
    device_row_sort_admissible,
    load_survey,
    load_survey_marks,
    sort_survey_rows_by_z,
)

SIGMA_KDE_PRIOR = [0.0, 0.01, 0.02, 0.035, 0.05]  # spans the sampled prior

COSMO = CosmoParams(H0=70.0, Om0=0.3, w0=-1.0, wa=0.0)


def _survey(sigma_kde):
    return SurveyParams(
        n0=1e-2, z50=1.0, w=0.3, delta=0.0, b_miss=0.0, alpha_miss=0.0,
        sigma_kde=sigma_kde,
    )


@pytest.fixture(autouse=True)
def _restore_window_config():
    yield
    configure_catalog_kde_window()


def _raw_mock(rng, n_gal=900, n_max=960, z_hi=1.52):
    """Review-shaped row (uniform + cluster lumps + a gap) plus a sparse row,
    an empty row, and a short row.  Returned UNSORTED."""
    z = rng.uniform(0.02, z_hi, 3 * n_gal)
    lump = rng.uniform(size=3 * n_gal) < 0.15
    centers = rng.choice([0.35, 0.62, 1.1], size=3 * n_gal)
    z = np.where(lump, np.clip(rng.normal(centers, 0.03), 0.02, z_hi), z)
    z = z[(z < 0.9) | (z > 0.97)][:n_gal]
    n_gal = len(z)

    zg = np.zeros((4, n_max))
    dz = np.zeros((4, n_max))
    wg = np.zeros((4, n_max))
    ng = np.array([n_gal, 40, 0, 5])
    zg[0, :n_gal] = z
    dz[0, :n_gal] = rng.uniform(0.010, 0.025, n_gal)
    wg[0, :n_gal] = rng.lognormal(0.0, 0.4, n_gal)
    zg[1, :40] = rng.uniform(0.05, 1.2, 40)
    dz[1, :40] = rng.uniform(0.010, 0.025, 40)
    wg[1, :40] = rng.lognormal(0.0, 0.4, 40)
    zg[3, :5] = rng.uniform(0.2, 0.6, 5)
    dz[3, :5] = 0.01
    wg[3, :5] = 1.0
    return zg, dz, wg, ng


def _em(zg, dz, wg, ng):
    return EMCatalog(
        apix=1e-3, zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(wg), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, 10)), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _samples(rng, zg, ng, n_samp=1200, z_hi=1.52):
    """Uniform + galaxy-adjacent + in-gap + beyond-support redshifts, spread
    over all rows (including the empty one)."""
    n4 = n_samp // 4
    z_u = rng.uniform(0.0, z_hi + 0.1, n4 * 2)
    picks = rng.integers(0, int(ng[0]), n4 // 2)
    z_n = np.clip(np.asarray(zg)[0, picks] + rng.normal(0, 0.02, n4 // 2), 0.0, z_hi + 0.1)
    z_g = rng.uniform(0.90, 0.97, n4 // 2)
    z_f = rng.uniform(z_hi, 3.0, n_samp - 2 * n4 - n4 // 2 - n4 // 2)
    z_all = jnp.asarray(np.concatenate([z_u, z_n, z_g, z_f]))
    pix = jnp.asarray(rng.integers(0, 4, z_all.shape[0]), dtype=jnp.int32)
    return z_all, pix


def _eval_all(emc, sigma_kde, zsamp, pixs, window, n_sigma=8.0,
              volume_weighted=False):
    if window is None:
        configure_catalog_kde_window(None)
    else:
        configure_catalog_kde_window(window, n_sigma)
    try:
        st = catalog_kernel_state(
            COSMO, _survey(sigma_kde), emc, volume_weighted=volume_weighted
        )
        return np.asarray(
            vmap(
                lambda z_i, p_i: eval_log_catalog_prior_state(z_i, p_i, st, emc)
            )(zsamp, pixs)
        )
    finally:
        configure_catalog_kde_window()


def _max_abs_delta(a, b):
    """max |a - b| with agreeing -inf pairs counting as zero and any -inf
    disagreement as +inf."""
    a = np.asarray(a)
    b = np.asarray(b)
    if (np.isneginf(a) != np.isneginf(b)).any():
        return np.inf
    both = np.isneginf(a) & np.isneginf(b)
    with np.errstate(invalid="ignore"):
        return float(np.max(np.where(both, 0.0, np.abs(a - b))))


# ------------------------------------------------------------
# Sort invariant (catalogs/io.py)
# ------------------------------------------------------------

def test_sort_permutes_every_coindexed_array_coherently():
    rng = np.random.default_rng(3)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=200, n_max=230)
    mark = rng.normal(size=zg.shape)
    zs, dzs, wss, ngs, (mark_s,) = sort_survey_rows_by_z(
        zg, dz, wg, ng, extras=(mark,)
    )
    assert np.array_equal(ngs, ng)
    for r in range(zg.shape[0]):
        n = int(ng[r])
        # real prefix ascending
        assert np.all(np.diff(zs[r, :n]) >= 0)
        # padding untouched, in place
        assert np.array_equal(zs[r, n:], zg[r, n:])
        assert np.array_equal(dzs[r, n:], dz[r, n:])
        # per-galaxy tuples (z, dz, w, mark) preserved as a multiset
        before = sorted(zip(zg[r, :n], dz[r, :n], wg[r, :n], mark[r, :n]))
        after = sorted(zip(zs[r, :n], dzs[r, :n], wss[r, :n], mark_s[r, :n]))
        assert before == after


def test_device_sort_is_byte_identical_to_the_host_sort():
    """to_device=True must be the SAME permutation and the SAME bytes.

    The device path is a permutation plus gathers -- no arithmetic on the
    values -- and both implementations run a stable sort on the same
    ``+inf``-padded key, so ties (every padding slot included) break by column
    index in both.  Anything weaker than byte equality here would move the
    likelihood.
    """
    rng = np.random.default_rng(3141)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=200, n_max=230)
    mark = rng.normal(size=zg.shape)

    order_h = _row_z_sort_order(zg, ng)
    order_d = np.asarray(_row_z_sort_order_device(zg, ng))
    assert order_d.dtype == np.int32  # halves the transient device index table
    assert np.array_equal(order_d.astype(np.int64), order_h)

    host = sort_survey_rows_by_z(zg, dz, wg, ng, extras=(mark, None))
    # The implementation directly, NOT via to_device: on the CPU-only gate
    # runs device_row_sort_admissible() is False and the dispatch would hand
    # back the numpy result, making this test vacuous.
    dev = _sort_survey_rows_by_z_device(zg, dz, wg, ng, extras=(mark, None))
    for a, b in zip(host[:4], dev[:4]):
        assert np.array_equal(np.asarray(b), np.asarray(a))
    assert dev[4][1] is None
    assert np.array_equal(np.asarray(dev[4][0]), host[4][0])
    for arr in dev[:3]:
        assert isinstance(arr, jax.Array)


def test_device_sort_asserts_on_nan_redshift():
    """The invariant check is a device reduce, not a dropped check."""
    rng = np.random.default_rng(4)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=50, n_max=60)
    zg[0, 3] = np.nan
    with pytest.raises(AssertionError, match="row z-sort invariant violated"):
        _sort_survey_rows_by_z_device(zg, dz, wg, ng)
    with pytest.raises(AssertionError) as exc:
        sort_survey_rows_by_z(zg, dz, wg, ng)
    assert str(exc.value) == ROW_Z_SORT_INVARIANT_ERROR


def test_load_survey_device_and_host_paths_agree_bitwise(tmp_path, monkeypatch):
    """The two load_survey paths return the same bytes, marks co-indexed.

    The gate is forced open so the device implementation is genuinely what
    the ``to_device=True`` arm runs even on a CPU-only test box (where
    :func:`device_row_sort_admissible` is False by design).
    """
    monkeypatch.setattr(catalogs_io, "device_row_sort_admissible", lambda: True)
    rng = np.random.default_rng(2718)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=120, n_max=140)
    path = os.path.join(tmp_path, "survey.hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 16
        f.create_dataset("zgals", data=zg)
        f.create_dataset("dzgals", data=dz)
        f.create_dataset("wgals", data=wg)
        f.create_dataset("ngals", data=ng)

    host = load_survey(path, to_device=False)
    dev = load_survey(path, to_device=True)
    assert host[0] == dev[0] and host[5] == dev[5]
    for a, b in zip(host[1:5], dev[1:5]):
        assert np.array_equal(np.asarray(b), np.asarray(a))

    # marks re-derive the permutation on the host and stay co-indexed with
    # the device-sorted galaxy arrays.
    order = _row_z_sort_order(zg, ng)
    assert np.array_equal(np.asarray(dev[2]), np.take_along_axis(zg, order, axis=1))


def test_device_row_sort_gate_requires_an_accelerator_and_x64(monkeypatch):
    """The admissibility predicate, and that the dispatch obeys it.

    ``to_device`` says "the caller will upload these arrays", not "there is
    an accelerator".  On a CPU-only install XLA-CPU's argsort is ~3.2x slower
    than ``np.argsort(kind="stable")`` on production-shaped rows, so the
    device implementation must NOT be taken there; with x64 off it would also
    build a float32 key whose permutation diverges from the float64 one
    :func:`load_survey_marks` re-derives.
    """
    on_cpu = jax.default_backend() == "cpu"
    x64 = bool(jax.config.jax_enable_x64)
    assert device_row_sort_admissible() is (not on_cpu and x64)
    if on_cpu:  # the regression this gate exists for
        assert device_row_sort_admissible() is False
    with jax.experimental.disable_x64():  # whatever the backend
        assert device_row_sort_admissible() is False

    def _boom(*a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("device sort taken while inadmissible")

    monkeypatch.setattr(catalogs_io, "_sort_survey_rows_by_z_device", _boom)
    monkeypatch.setattr(catalogs_io, "device_row_sort_admissible", lambda: False)
    rng = np.random.default_rng(11)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=40, n_max=50)
    out = catalogs_io.sort_survey_rows_by_z(zg, dz, wg, ng, to_device=True)
    assert isinstance(out[0], np.ndarray)  # the numpy implementation ran
    assert np.array_equal(out[0], sort_survey_rows_by_z(zg, dz, wg, ng)[0])


def test_device_sort_handles_a_zero_row_catalog():
    """The two implementations agree on a zero-row catalog (gather edge)."""
    z = np.zeros((0, 5))
    ng = np.zeros(0, dtype=int)
    host = sort_survey_rows_by_z(z, z, z, ng, extras=(z, None))
    dev = _sort_survey_rows_by_z_device(z, z, z, ng, extras=(z, None))
    for a, b in zip(host[:4], dev[:4]):
        assert np.asarray(b).shape == np.asarray(a).shape
    assert dev[4][1] is None
    assert np.asarray(dev[4][0]).shape == (0, 5)


def test_sort_asserts_on_nan_redshift():
    rng = np.random.default_rng(4)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=50, n_max=60)
    zg[0, 3] = np.nan
    with pytest.raises(AssertionError):
        sort_survey_rows_by_z(zg, dz, wg, ng)


def test_load_survey_and_marks_stay_coindexed(tmp_path):
    rng = np.random.default_rng(5)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=120, n_max=140)
    # a mark encoding each galaxy's identity: recoverable pairing check
    mark = np.where(
        np.arange(zg.shape[1])[None, :] < ng[:, None], 1000.0 * zg + 7.0, 0.0
    )
    path = os.path.join(tmp_path, "survey.hdf5")
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 16
        f.create_dataset("zgals", data=zg)
        f.create_dataset("dzgals", data=dz)
        f.create_dataset("wgals", data=wg)
        f.create_dataset("ngals", data=ng)
        f.create_dataset("mark_logmstar", data=mark)

    _, ng_l, zg_l, dz_l, wg_l, _ = load_survey(path, to_device=False)
    marks = load_survey_marks(path)
    for r in range(zg.shape[0]):
        n = int(ng[r])
        assert np.all(np.diff(zg_l[r, :n]) >= 0)
        assert np.allclose(
            marks["mark_logmstar"][r, :n], 1000.0 * zg_l[r, :n] + 7.0
        )

    # opt-out keeps the raw file order in BOTH loaders
    _, _, zg_raw, _, _, _ = load_survey(path, to_device=False, sort_rows_by_z=False)
    marks_raw = load_survey_marks(path, sort_rows_by_z=False)
    assert np.array_equal(zg_raw, zg)
    assert np.array_equal(marks_raw["mark_logmstar"], mark)


# ------------------------------------------------------------
# Windowed evaluator: parity across the sigma_kde prior
# ------------------------------------------------------------

def test_windowed_matches_full_row_across_sigma_kde_prior():
    rng = np.random.default_rng(280)
    zg, dz, wg, ng = _raw_mock(rng)
    zg, dz, wg, ng, _ = sort_survey_rows_by_z(zg, dz, wg, ng)
    emc = _em(zg, dz, wg, ng)
    zsamp, pixs = _samples(rng, zg, ng)
    # W scaled to the mock the same way the 1024 default is scaled to a
    # 2113-galaxy row: sized by the data-driven rule at the prior's top.
    W = recommended_kde_window(zg, ng, dz, sigma_kde_max=0.05)
    assert W < int(ng[0])  # windowing is actually exercised
    for sk in SIGMA_KDE_PRIOR:
        full = _eval_all(emc, sk, zsamp, pixs, window=None)
        win = _eval_all(emc, sk, zsamp, pixs, window=W)
        assert _max_abs_delta(win, full) < 1e-6, f"sigma_kde={sk}"


def test_windowed_matches_full_row_marked_state():
    rng = np.random.default_rng(281)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=400, n_max=430)
    zg, dz, wg, ng, _ = sort_survey_rows_by_z(zg, dz, wg, ng)
    emc = _em(zg, dz, wg, ng)
    log_h = jnp.asarray(rng.normal(0.0, 0.5, zg.shape))
    zsamp, pixs = _samples(rng, zg, ng, n_samp=400)
    W = recommended_kde_window(zg, ng, dz, sigma_kde_max=0.05)

    def _eval_marked(sk, window):
        if window is None:
            configure_catalog_kde_window(None)
        else:
            configure_catalog_kde_window(window)
        try:
            st, _ = marked_catalog_kernel_state(
                COSMO, _survey(sk), emc, log_h
            )
            return np.asarray(
                vmap(
                    lambda z_i, p_i: eval_log_catalog_prior_state(z_i, p_i, st, emc)
                )(zsamp, pixs)
            )
        finally:
            configure_catalog_kde_window()

    for sk in (0.0, 0.05):
        assert _max_abs_delta(_eval_marked(sk, W), _eval_marked(sk, None)) < 1e-6


def test_volume_weighted_state_windowed_parity():
    rng = np.random.default_rng(282)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=400, n_max=430)
    zg, dz, wg, ng, _ = sort_survey_rows_by_z(zg, dz, wg, ng)
    emc = _em(zg, dz, wg, ng)
    zsamp, pixs = _samples(rng, zg, ng, n_samp=400)
    W = recommended_kde_window(zg, ng, dz, sigma_kde_max=0.05)
    for sk in (0.0, 0.05):
        full = _eval_all(emc, sk, zsamp, pixs, None, volume_weighted=True)
        win = _eval_all(emc, sk, zsamp, pixs, W, volume_weighted=True)
        assert _max_abs_delta(win, full) < 1e-6


# ------------------------------------------------------------
# -inf / NaN gradient contract
# ------------------------------------------------------------

def test_all_padding_window_bitwise_neginf_and_finite_grad():
    """A window containing zero real galaxies (empty pixel) must return
    exactly -inf through the same safe-logsumexp path as the full row, and
    jax.grad through the term must stay finite for every input."""
    rng = np.random.default_rng(283)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=300, n_max=330)
    zg, dz, wg, ng, _ = sort_survey_rows_by_z(zg, dz, wg, ng)
    emc = _em(zg, dz, wg, ng)
    EMPTY = 2  # ng[2] == 0

    def term(sigma_kde, z, window):
        if window is None:
            configure_catalog_kde_window(None)
        else:
            configure_catalog_kde_window(window)
        try:
            st = catalog_kernel_state(COSMO, _survey(sigma_kde), emc)
            return eval_log_catalog_prior_state(z, EMPTY, st, emc)
        finally:
            configure_catalog_kde_window()

    v_win = term(0.02, 0.5, 64)
    v_full = term(0.02, 0.5, None)
    # bitwise-identical handling of the empty/padded window
    assert np.isneginf(v_win) and np.isneginf(v_full)
    assert np.asarray(v_win).tobytes() == np.asarray(v_full).tobytes()

    g_sk, g_z = jax.grad(lambda sk, z: term(sk, z, 64), argnums=(0, 1))(0.02, 0.5)
    assert np.isfinite(g_sk) and np.isfinite(g_z)

    # the empty pixel must not poison a batched sum's gradient either
    zsamp, pixs = _samples(rng, zg, ng, n_samp=64)

    def total(sk):
        configure_catalog_kde_window(64)
        try:
            st = catalog_kernel_state(COSMO, _survey(sk), emc)
            vals = vmap(
                lambda z_i, p_i: eval_log_catalog_prior_state(z_i, p_i, st, emc)
            )(zsamp, pixs)
            return jnp.sum(jnp.where(jnp.isfinite(vals), vals, 0.0))
        finally:
            configure_catalog_kde_window()

    assert np.isfinite(jax.grad(total)(0.02))


def test_windowed_gradient_matches_full_row():
    rng = np.random.default_rng(284)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=400, n_max=430)
    zg, dz, wg, ng, _ = sort_survey_rows_by_z(zg, dz, wg, ng)
    emc = _em(zg, dz, wg, ng)
    zsamp, pixs = _samples(rng, zg, ng, n_samp=256)
    W = recommended_kde_window(zg, ng, dz, sigma_kde_max=0.05)

    def total(sk, window):
        if window is None:
            configure_catalog_kde_window(None)
        else:
            configure_catalog_kde_window(window)
        try:
            st = catalog_kernel_state(COSMO, _survey(sk), emc)
            vals = vmap(
                lambda z_i, p_i: eval_log_catalog_prior_state(z_i, p_i, st, emc)
            )(zsamp, pixs)
            return jnp.sum(jnp.where(jnp.isfinite(vals), vals, 0.0))
        finally:
            configure_catalog_kde_window()

    for sk in (0.001, 0.05):
        gf = float(jax.grad(total)(sk, None))
        gw = float(jax.grad(total)(sk, W))
        assert np.isfinite(gf) and np.isfinite(gw)
        assert abs(gw - gf) <= 1e-4 * max(abs(gf), 1.0), f"sigma_kde={sk}"


# ------------------------------------------------------------
# Fallbacks and the escape hatch
# ------------------------------------------------------------

def test_unsorted_rows_fall_back_to_full_row_bitwise():
    rng = np.random.default_rng(285)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=300, n_max=330)  # NOT sorted
    emc = _em(zg, dz, wg, ng)
    zsamp, pixs = _samples(rng, zg, ng, n_samp=200)
    win = _eval_all(emc, 0.02, zsamp, pixs, window=64)
    full = _eval_all(emc, 0.02, zsamp, pixs, window=None)
    assert np.array_equal(win, full)  # windowing silently disabled


def test_row_shorter_than_window_uses_full_row_bitwise():
    rng = np.random.default_rng(286)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=100, n_max=120)
    zg, dz, wg, ng, _ = sort_survey_rows_by_z(zg, dz, wg, ng)
    emc = _em(zg, dz, wg, ng)
    zsamp, pixs = _samples(rng, zg, ng, n_samp=200)
    win = _eval_all(emc, 0.02, zsamp, pixs, window=512)   # W >= N_max
    full = _eval_all(emc, 0.02, zsamp, pixs, window=None)
    assert np.array_equal(win, full)


def test_configure_validation():
    with pytest.raises(ValueError):
        configure_catalog_kde_window(1)
    with pytest.raises(ValueError):
        configure_catalog_kde_window(64, n_sigma=0.0)
    configure_catalog_kde_window(None)   # escape hatch accepted
    configure_catalog_kde_window()       # defaults restored by fixture too


# ------------------------------------------------------------
# Window sizing diagnostic
# ------------------------------------------------------------

def test_recommended_kde_window_bounds_block_counts():
    rng = np.random.default_rng(287)
    zg, dz, wg, ng = _raw_mock(rng, n_gal=300, n_max=330)
    zg, dz, wg, ng, _ = sort_survey_rows_by_z(zg, dz, wg, ng)
    w_narrow = recommended_kde_window(zg, ng, dz, sigma_kde_max=0.005)
    w_wide = recommended_kde_window(zg, ng, dz, sigma_kde_max=0.05)
    assert 0 < w_narrow <= w_wide <= 2 * int(np.max(ng))   # 2 x one-sided, uncapped
    # brute force on the dense row: every +/- 6 sigma_max block must fit
    n = int(ng[0])
    zr = np.asarray(zg)[0, :n]
    sig_max = float(np.max(np.sqrt(np.asarray(dz)[0, :n] ** 2 + 0.05 ** 2)))
    half = 6.0 * sig_max
    counts = [
        int(((zr >= c - half) & (zr <= c + half)).sum())
        for c in np.linspace(zr[0], zr[-1], 500)
    ]
    assert max(counts) <= w_wide


# ---------------------------------------------------------------------------
# Traced (jit-argument) path: build-time attestation must arm windowing
# ---------------------------------------------------------------------------
# The production likelihood (darksiren_log_likelihood, module-level jit) takes
# every EMCatalog as a traced ARGUMENT, so the evaluator's concrete sortedness
# check cannot run and — before the attestation hook — the windowed path never
# engaged where it matters most.  These tests pin the arming contract.


def _tiny_sorted_catalog(n=20, seed=0):
    zs = np.sort(np.random.default_rng(seed).uniform(0.05, 1.0, n))
    return EMCatalog(
        apix=1.0,
        zgals=jnp.asarray(zs[None, :]),
        dzgals=jnp.full((1, n), 0.01),
        wgals=jnp.ones((1, n)),
        ngals=jnp.asarray([n], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 8)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


@pytest.fixture
def _window8():
    """Small static window + guaranteed-clean attestation state."""
    import darksirens.redshift.catalog as C

    old_attested = C._ROWS_SORTED_ATTESTED
    C.configure_catalog_kde_window(size=8, n_sigma=8.0)
    try:
        yield C
    finally:
        C.configure_catalog_kde_window()
        C._ROWS_SORTED_ATTESTED = old_attested


def _traced_window_calls(C, cat, state):
    """Trace the evaluator with catalog+state as jit ARGUMENTS; count window
    searches and return (calls, value)."""
    calls = [0]
    orig = C._sorted_row_window_start

    def spy(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    C._sorted_row_window_start = spy
    try:
        f = jax.jit(
            lambda z, pix, st, ct: C.eval_log_catalog_prior_state(z, pix, st, ct)
        )
        val = float(f(jnp.asarray(0.3), jnp.asarray(0, dtype=jnp.int32), state, cat))
    finally:
        C._sorted_row_window_start = orig
    return calls[0], val


def test_jit_argument_path_windows_only_after_attestation(_window8):
    C = _window8
    cat = _tiny_sorted_catalog()
    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)
    state = C.catalog_kernel_state(cosmo, survey, cat)

    # Full-row reference (windowing off entirely).
    C.configure_catalog_kde_window(size=None)
    ref = float(C.eval_log_catalog_prior_state(
        jnp.asarray(0.3), jnp.asarray(0, dtype=jnp.int32), state, cat
    ))
    C.configure_catalog_kde_window(size=8, n_sigma=8.0)

    # Un-attested: traced catalogs cannot be verified -> full-row fallback.
    C._ROWS_SORTED_ATTESTED = False
    calls, val = _traced_window_calls(C, cat, state)
    assert calls == 0
    assert val == pytest.approx(ref, rel=1e-12)

    # Attested with the concrete arrays -> the traced path windows.
    assert C.attest_rows_sorted_for_windowing(cat) is True
    calls, val = _traced_window_calls(C, cat, state)
    assert calls == 1
    assert val == pytest.approx(ref, rel=1e-10)


def test_attestation_does_not_arm_an_unattested_view(_window8):
    """Arming is keyed to the attested ROW SHAPES, not to the process: a view
    nobody verified must not be windowed just because some other build attested
    its own views.  The evaluator's soundness rests on the z-sort invariant, and
    on an unsorted row the window can miss the sample's nearest galaxies with no
    symptom other than a wrong log p_cat."""
    C = _window8
    good = _tiny_sorted_catalog(n=20)
    assert C.attest_rows_sorted_for_windowing(good) is True

    # A DIFFERENT (unattested, and here unsorted) view reaching the evaluator
    # through a jit boundary — e.g. a diagnostic call through PRIOR_REGISTRY.
    zs = np.random.default_rng(1).uniform(0.05, 1.0, 24)     # not sorted
    other = _tiny_sorted_catalog(n=24)._replace(zgals=jnp.asarray(zs[None, :]))
    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)
    state = C.catalog_kernel_state(cosmo, survey, other)
    calls, _ = _traced_window_calls(C, other, state)
    assert calls == 0

    # The attested view itself still windows.
    state_good = C.catalog_kernel_state(cosmo, survey, good)
    calls, _ = _traced_window_calls(C, good, state_good)
    assert calls == 1


def test_attestation_disarms_on_unsorted_catalog(_window8):
    C = _window8
    good = _tiny_sorted_catalog()
    zs = np.asarray(good.zgals)[:, ::-1].copy()      # descending: violates invariant
    bad = good._replace(zgals=jnp.asarray(zs))

    assert C.attest_rows_sorted_for_windowing(good, bad) is False
    assert C._ROWS_SORTED_ATTESTED is False

    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)
    state = C.catalog_kernel_state(cosmo, survey, bad)
    calls, _ = _traced_window_calls(C, bad, state)
    assert calls == 0   # disarmed: traced path stays on the full row


def test_factory_attests_bound_catalog_views():
    """Both likelihood factories must attest the concrete catalog views they
    bind — without the call, the traced (jit-argument) production path never
    windows.  Wiring pin: the call must appear at both operand-binding sites."""
    import inspect

    import darksirens.likelihood.factory as F

    src = inspect.getsource(F)
    assert src.count("attest_rows_sorted_for_windowing(") >= 2, (
        "the factories no longer attest their bound catalog views; the "
        "windowed catalog KDE will silently fall back to the full row in "
        "production (catalogs cross the jit boundary as arguments)"
    )


# ---------------------------------------------------------------------------
# The compiled windowed branch must be self-verifying (external review JAX-05)
# ---------------------------------------------------------------------------
# The trace-time decision to window reads mutable process globals
# (_KDE_WINDOW_SIZE, _ROWS_SORTED_ATTESTED, _ATTESTED_ROW_SHAPES), none of which
# is part of a jit cache key.  A callable compiled while a sorted view was
# attested is therefore replayed verbatim on an unsorted view of the SAME shape:
# the tests above hide that by building a fresh jitted function per case.
# MEASURED on master with ONE jitted callable, window 8, a (1, 20) row and its
# own permutation: the cache returned 0.4571784999571946 where a fresh trace
# gives 0.4597108139040813, a silent 2.5e-3 nat corruption of a later
# programmatic likelihood in the same process.


def _one_jitted_evaluator(C, cosmo, survey):
    """A SINGLE jitted callable, as a factory builds once and reuses."""

    @jax.jit
    def evaluate(z, pix, ct):
        st = C.catalog_kernel_state(cosmo, survey, ct)
        return C.eval_log_catalog_prior_state(z, pix, st, ct)

    return evaluate


def test_cached_windowed_branch_refuses_unsorted_data_of_the_same_shape(_window8):
    C = _window8
    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)

    zs = np.random.default_rng(0).uniform(0.05, 1.0, 20)
    good = _tiny_sorted_catalog(n=20)._replace(zgals=jnp.asarray(np.sort(zs)[None, :]))
    bad = good._replace(zgals=jnp.asarray(zs[None, :]))    # same shape, unsorted

    evaluate = _one_jitted_evaluator(C, cosmo, survey)
    z, pix = jnp.asarray(0.3), jnp.asarray(0, dtype=jnp.int32)

    assert C.attest_rows_sorted_for_windowing(good) is True
    windowed = float(evaluate(z, pix, good))
    assert np.isfinite(windowed)

    # Disarm and replay the SAME compiled callable on unsorted data.  It must
    # not hand back a plausible number computed by a window that is invalid
    # there; the in-graph verdict collapses it to NaN.
    C.attest_rows_sorted_for_windowing(bad)
    assert np.isnan(float(evaluate(z, pix, bad)))


def test_the_windowed_verdict_rides_in_the_graph_not_in_a_global(_window8):
    """Same callable, same shape, no re-attestation at all: correctness must
    still track the DATA, because nothing else reaches the compiled branch."""
    C = _window8
    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)

    zs = np.random.default_rng(3).uniform(0.05, 1.0, 20)
    good = _tiny_sorted_catalog(n=20)._replace(zgals=jnp.asarray(np.sort(zs)[None, :]))
    bad = good._replace(zgals=jnp.asarray(zs[None, :]))

    assert C.attest_rows_sorted_for_windowing(good) is True
    evaluate = _one_jitted_evaluator(C, cosmo, survey)
    z, pix = jnp.asarray(0.3), jnp.asarray(0, dtype=jnp.int32)

    assert np.isfinite(float(evaluate(z, pix, good)))
    assert np.isnan(float(evaluate(z, pix, bad)))


def test_the_guard_leaves_the_attested_sorted_path_bit_identical(_window8):
    """The guard must cost nothing in correctness terms: on attested sorted
    rows the windowed value is still bitwise the full-row value."""
    C = _window8
    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)
    cat = _tiny_sorted_catalog(n=20)
    state = C.catalog_kernel_state(cosmo, survey, cat)

    C.configure_catalog_kde_window(size=None)
    ref = float(C.eval_log_catalog_prior_state(
        jnp.asarray(0.3), jnp.asarray(0, dtype=jnp.int32), state, cat
    ))
    C.configure_catalog_kde_window(size=8, n_sigma=8.0)

    assert C.attest_rows_sorted_for_windowing(cat) is True
    evaluate = _one_jitted_evaluator(C, cosmo, survey)
    got = float(evaluate(jnp.asarray(0.3), jnp.asarray(0, dtype=jnp.int32), cat))
    assert got == pytest.approx(ref, rel=1e-10)


def test_concrete_catalogs_pay_no_runtime_guard(_window8):
    """An eagerly-evaluated catalog is verified from its own arrays at trace
    time, so no ``rows_sorted`` node is built for it at all."""
    C = _window8
    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)
    state = C.catalog_kernel_state(cosmo, survey, _tiny_sorted_catalog(n=20))
    assert state.rows_sorted is None
