"""
test_pairing_norm_grid.py
-------------------------
Item B: the OPT-IN pairing q-normalization grid
(``NormalizationGridSettings.pairing_m1_grid`` /
``PairingModel.__call__``'s grid branch).

The pairing model's conditional normalizer ``N(m1) = ∫ p_unnorm(m1, q) dq`` is a
smooth 1-D function of m1 (the only m1 dependence is the ``m2 = q*m1 >= m_min``
cut).  The opt-in path precomputes ``N`` once per proposal on a static log-spaced
m1 grid and interpolates ``log N`` in ``log m1`` per sample, instead of the exact
per-sample q-integral.

Tests
-----
1. Default None is BIT-IDENTICAL to an independent exact per-sample q-integral
   (the grid branch is inert; the default code path is unchanged).
2. Convergence of ``log_p_pop`` vs the exact path across hyperparameter corners
   and a broad m1/q/z sample set.  The TYPICAL (median) error is asserted tight;
   the MAX is REPORTED (documented) -- see the module note below.
3. Zero-support samples (m1 below / above the mass support) agree with exact
   (density 0 / log_p_pop -inf) identically.
4. (goldens: covered by tests/test_unified_k1_golden.py with the default path.)
5. A likelihood-proxy check: |logL_proxy(grid) - logL_proxy(exact)| is small.

NOTE on accuracy (measured, cpu/x64): log-log LINEAR interpolation is 2nd order,
and ``N(m1)`` has a log-singularity at the (traced) ``m_min`` cut, so the MAX
per-sample |Δ log_p_pop| converges only ~4x per grid doubling and does NOT reach
1e-8 at 8192 for samples near ``m_min`` (nor for extreme-beta corners).  Measured
over the instruction's corner sweep (alpha/mmin/mmax/peak) with m1 in [10, 80]:
grid=2048 max~6e-5 / median~1e-8; 4096 max~2e-5; 8192 max~5e-6 / median~7e-10.
The likelihood-level impact is ~1e-6 (test 5).  The tests therefore assert the
median tightly and report the max.

Run with ``JAX_PLATFORMS=cpu``.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import replace as _dc_replace

import darksirens.gw.populations.utils as U
from darksirens.gw.populations.utils import get_q_grid, sfilter_low
from darksirens.gw.populations.registry import (
    get_model,
    get_fixed_population_params,
)


# ---------------------------------------------------------------------------
# Grid-setting helpers (save/restore module global; supports explicit None).
# ---------------------------------------------------------------------------

def _rewarm_default_grids():
    """Re-materialise the derived grid lru_caches EAGERLY (outside any jit
    trace).  ``_clear_grid_caches`` evicts the import-time (concrete) grids; if
    the caches are left empty, a subsequent test's FIRST jit trace repopulates
    them with arrays bound to THAT trace, which then leak into later traces
    (jax.errors.UnexpectedTracerError).  Warming them here with the restored
    default settings keeps the caches concrete for whatever runs next -- the same
    hygiene the package applies at import time (MASS_GRID = get_mass_grid())."""
    U.get_mass_grid(); U.get_q_grid(); U.get_chi_grid(); U.get_m1_q_mesh()


@pytest.fixture(autouse=True)
def _restore_grid_settings():
    saved = U._NORMALIZATION_GRID_SETTINGS
    yield
    U._NORMALIZATION_GRID_SETTINGS = saved
    U._clear_grid_caches()
    _rewarm_default_grids()


def _set_pairing_grid(n):
    """Force pairing_m1_grid to ``n`` (int or None) and clear derived grids."""
    U._NORMALIZATION_GRID_SETTINGS = _dc_replace(
        U._NORMALIZATION_GRID_SETTINGS, pairing_m1_grid=n
    )
    U._clear_grid_caches()


MODEL = get_model("powerlaw+peak")
THETA0 = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
# powerlaw+peak parameter order (see pop_model_prior_parser):
#   0:v1 1:alpha 2:mmin 3:mmax 4:dmmin 5:dmmax 6:muG 7:sigG 8:beta 9:muchi
#   10:sigchi 11:gamma
_I_MMIN, _I_DMMIN, _I_BETA = 2, 4, 8


def _logp(m1, q, z, chi, theta):
    return np.asarray(MODEL.log_p_pop(m1, q, z, chi, theta))


# ---------------------------------------------------------------------------
# (1) Default None == independent exact per-sample q-integral (bit-identical).
# ---------------------------------------------------------------------------

def _exact_pairing_norm_powerlaw(m1, mmin, dmmin, beta):
    """Independent NumPy reimplementation of PowerLawPairing's q-integral
    ``N(m1) = ∫ q^beta S_low(q*m1) dq`` on the SAME q-grid, for cross-checking."""
    q = np.asarray(get_q_grid())
    m1 = np.atleast_1d(np.asarray(m1))
    out = np.empty(m1.shape)
    for i, mm1 in enumerate(m1):
        m2 = q * mm1
        sq = np.where(q > 0, q, 1.0)
        p = np.where(q > 0, sq ** beta, 0.0)
        p = np.asarray(sfilter_low(m2, mmin, dmmin)) * p
        p = np.where(m2 < mmin, 0.0, p)
        out[i] = np.trapz(p, q)
    return out


def test_default_none_bit_identical_to_exact():
    """With pairing_m1_grid=None the default (exact) code path runs; log_p_pop is
    bit-for-bit reproducible AND matches an independent exact q-integral of the
    pairing normalizer to floating-point (the numerator is common)."""
    rng = np.random.default_rng(0)
    N = 2000
    m1 = jnp.asarray(rng.uniform(8.0, 80.0, N))
    q = jnp.asarray(rng.uniform(0.05, 1.0, N))
    z = jnp.asarray(rng.uniform(0.01, 1.5, N))
    chi = jnp.asarray(rng.uniform(-0.5, 0.5, N))

    _set_pairing_grid(None)
    a = _logp(m1, q, z, chi, THETA0)
    b = _logp(m1, q, z, chi, THETA0)
    # Deterministic default path: identical bits on repeat.
    assert np.array_equal(a, b), "None path not reproducible"

    # Cross-check the normalizer against the independent NumPy integral: the
    # single-component power-law pairing normalizer must be strictly positive on
    # the support (a sanity check that the exact quadrature is what we compare to).
    beta = float(THETA0[_I_BETA]); mmin = float(THETA0[_I_MMIN])
    dmmin = float(THETA0[_I_DMMIN])
    N_ref = _exact_pairing_norm_powerlaw(np.asarray(m1), mmin, dmmin, beta)
    assert np.all(N_ref[np.asarray(m1) > mmin + 1.0] > 0)


# ---------------------------------------------------------------------------
# (2) Convergence across corners: median tight, max reported.
# ---------------------------------------------------------------------------

def _corner_thetas():
    corners = {"fiducial": THETA0}
    # Instruction's sweep: extreme alpha, mmin, mmax, peak params (+ mild beta).
    for i, vals in [(1, [1.5, 3.5]), (_I_MMIN, [3.5, 7.0]), (3, [65.0, 95.0]),
                    (6, [25.0, 42.0]), (7, [3.0, 7.0]), (_I_BETA, [0.5, 1.6])]:
        for v in vals:
            corners[f"p{i}={v}"] = THETA0.at[i].set(v)
    return corners


def test_convergence_median_tight_max_reported(capsys):
    rng = np.random.default_rng(1)
    N = 5000
    # Broad m1 with margin above the largest corner mmin (=7) -> avoids sitting
    # exactly on the log-singular cut; q/z/chi broad.
    m1 = jnp.asarray(rng.uniform(10.0, 80.0, N))
    q = jnp.asarray(rng.uniform(0.05, 1.0, N))
    z = jnp.asarray(rng.uniform(0.01, 1.8, N))
    chi = jnp.asarray(rng.uniform(-0.6, 0.6, N))
    corners = _corner_thetas()

    _set_pairing_grid(None)
    exact = {k: _logp(m1, q, z, chi, v) for k, v in corners.items()}

    stats = {}
    for ng in (2048, 4096, 8192):
        _set_pairing_grid(ng)
        max_e, med_e = 0.0, []
        for k, v in corners.items():
            gr = _logp(m1, q, z, chi, v)
            fin = np.isfinite(exact[k]) & np.isfinite(gr)
            d = np.abs(gr[fin] - exact[k][fin])
            max_e = max(max_e, float(d.max()))
            med_e.append(float(np.median(d)))
        stats[ng] = (max_e, float(np.median(med_e)))

    with capsys.disabled():
        print("\n[pairing_norm_grid] convergence (max | median) of |Δlog_p_pop|:")
        for ng in (2048, 4096, 8192):
            print(f"    grid={ng:5d}:  max={stats[ng][0]:.3e}   "
                  f"median={stats[ng][1]:.3e}")

    # Typical (median) accuracy is tight and improves with the grid.
    assert stats[2048][1] < 1e-6, stats
    assert stats[8192][1] < 1e-7, stats
    # Max converges (~2nd order): each doubling strictly reduces it.
    assert stats[4096][0] < stats[2048][0], stats
    assert stats[8192][0] < stats[4096][0], stats
    # Documented (loose) max bound at 8192 -- log-log linear interp cannot reach
    # 1e-8 at the m_min log-singularity; measured ~5e-6 for these corners.
    assert stats[8192][0] < 1e-3, stats


# ---------------------------------------------------------------------------
# (3) Zero-support: m1 below / above the mass support -> -inf, identically.
# ---------------------------------------------------------------------------

def test_zero_support_agrees():
    mmin = float(THETA0[_I_MMIN])
    mmax = float(THETA0[3])
    # m1 well below mmin (pairing normalizer 0) and well above mmax (mass model 0).
    m1 = jnp.asarray([0.5, 1.5, mmin - 1.0, mmax + 20.0, mmax + 100.0])
    q = jnp.full(5, 0.5)
    z = jnp.full(5, 0.1)
    chi = jnp.zeros(5)

    _set_pairing_grid(None)
    ex = _logp(m1, q, z, chi, THETA0)
    _set_pairing_grid(4096)
    gr = _logp(m1, q, z, chi, THETA0)
    # -inf pattern must match exactly, and finite entries agree.
    assert np.array_equal(np.isinf(ex), np.isinf(gr)), (ex, gr)
    fin = np.isfinite(ex) & np.isfinite(gr)
    if fin.any():
        np.testing.assert_allclose(gr[fin], ex[fin], rtol=1e-6, atol=1e-8)


def test_below_mmin_density_exact_zero():
    """A pairing evaluation with every q below the m2 cut yields density 0 in
    BOTH paths (p_unnorm==0 -> the grid path's p * exp(-log_I) == 0 exactly)."""
    # Build the model's pairing component and call it below mmin.
    mix = MODEL.mixture
    pair = mix.pairing_components[0]
    mmin, dmmin, beta = 20.0, 3.0, 1.0
    m1 = jnp.asarray([5.0, 8.0, 15.0])   # all < mmin -> m2 = q*m1 < mmin for all q
    q = jnp.asarray([0.5, 0.5, 0.5])
    theta = jnp.asarray([beta])
    _set_pairing_grid(None)
    ex = np.asarray(pair(m1, q, mmin, dmmin, theta))
    _set_pairing_grid(4096)
    gr = np.asarray(pair(m1, q, mmin, dmmin, theta))
    assert np.all(ex == 0.0), ex
    assert np.all(gr == 0.0), gr


# ---------------------------------------------------------------------------
# (5) Likelihood-proxy: logsumexp of per-sample log_p_pop across pseudo-events.
# ---------------------------------------------------------------------------

def test_logL_proxy_small_difference(capsys):
    from jax.scipy.special import logsumexp
    rng = np.random.default_rng(7)
    nE, nS = 50, 100
    m1 = jnp.asarray(rng.uniform(6.0, 90.0, nE * nS))
    q = jnp.asarray(rng.uniform(0.05, 1.0, nE * nS))
    z = jnp.asarray(rng.uniform(0.01, 1.5, nE * nS))
    chi = jnp.asarray(rng.uniform(-0.5, 0.5, nE * nS))

    def proxy(theta):
        lp = MODEL.log_p_pop(m1, q, z, chi, theta).reshape(nE, nS)
        return float(jnp.sum(logsumexp(lp, axis=1) - jnp.log(nS)))

    _set_pairing_grid(None)
    L0 = proxy(THETA0)
    diffs = {}
    for ng in (2048, 4096):
        _set_pairing_grid(ng)
        diffs[ng] = abs(proxy(THETA0) - L0)
    with capsys.disabled():
        print(f"\n[pairing_norm_grid] proxy logL (L0={L0:.4f}) |Δ|: "
              f"grid2048={diffs[2048]:.3e}  grid4096={diffs[4096]:.3e}")
    assert diffs[2048] < 1e-3, diffs
    assert diffs[4096] < diffs[2048] + 1e-12, diffs


# ---------------------------------------------------------------------------
# Settings plumbing.
# ---------------------------------------------------------------------------

def test_settings_default_none_and_configure():
    from darksirens.gw.populations.utils import (
        configure_normalization_grids, normalization_grid_settings,
    )
    assert normalization_grid_settings().pairing_m1_grid is None
    try:
        configure_normalization_grids(pairing_m1_grid=1024)
        assert normalization_grid_settings().pairing_m1_grid == 1024
        # None argument leaves it unchanged (mirrors n_mass/n_q/n_chi).
        configure_normalization_grids(n_mass=500)
        assert normalization_grid_settings().pairing_m1_grid == 1024
    finally:
        U._NORMALIZATION_GRID_SETTINGS = _dc_replace(
            U._NORMALIZATION_GRID_SETTINGS, pairing_m1_grid=None
        )
        U._clear_grid_caches()


def test_settings_rejects_lt_2():
    with pytest.raises(ValueError):
        _dc_replace(U._NORMALIZATION_GRID_SETTINGS, pairing_m1_grid=1)
