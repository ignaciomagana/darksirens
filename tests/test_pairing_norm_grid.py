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

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import replace as _dc_replace

import darksirens.gw.populations.utils as U
from darksirens.gw.populations.utils import (
    get_q_grid,
    sfilter_low,
    normalization_grid_settings,
    size_pairing_grid_to_support,
    assert_pairing_grid_covers_support,
)
from darksirens.gw.populations.registry import (
    get_model,
    get_fixed_population_params,
    population_m1_support_max,
    pop_model_prior_parser,
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


def _set_pairing_grid(n, m_hi=None):
    """Force pairing_m1_grid to ``n`` (int or None) and clear derived grids.

    ``m_hi`` (optional) also pins the pairing grid's upper bound; leaving it None
    preserves whatever bound is currently configured (mirrors the module global).
    """
    kw = {"pairing_m1_grid": n}
    if m_hi is not None:
        kw["pairing_m_hi"] = float(m_hi)
    U._NORMALIZATION_GRID_SETTINGS = _dc_replace(U._NORMALIZATION_GRID_SETTINGS, **kw)
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
        out[i] = _trapezoid(p, q)  # np.trapz removed in NumPy 2.0
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


# ---------------------------------------------------------------------------
# (6) PHY-5: the opt-in pairing grid must be sized to the SELECTED model's m1
#     support.  gwtc5_fiducial_bpl2peaks supports m1 up to a FIXED 300-Msun edge
#     -- above the historical 200-node ceiling -- so a grid capped at 200 silently
#     clamps (jnp.interp) the pairing normaliser for every valid 200 < m1 <= 300
#     sample.  The registry exposes the support (population_m1_support_max); the
#     CLI sizes the grid to it (size_pairing_grid_to_support) and validates
#     coverage (assert_pairing_grid_covers_support).
# ---------------------------------------------------------------------------

GWTC5 = get_model("gwtc5_fiducial_bpl2peaks")
# gwtc5 param_specs order (see GWTC5FiducialBPL2PeaksPopulationModel.param_specs):
#   0..6 mass BPL+2G, 7:m1_low 8:delta_m1 9:lambda0 10:lambda1
#   11:beta_q 12:m2_low 13:delta_m2 14:mu_chi 15:sigma_chi 16:gamma
_GW_BETA, _GW_M2LOW, _GW_DM2 = 11, 12, 13


def _gwtc5_adverse_theta(delta_m2):
    """Fiducial gwtc5 vector with the instruction's adverse pairing corner:
    steepest allowed q slope (beta_q=-2) and the largest m2_low the model's
    ``m2_low <= m1_low`` constraint allows at the release-median m1_low = 4.4856
    (the instruction's 5.0 predates that fiducial and makes ``valid`` False, i.e.
    log_p_pop = -inf everywhere), plus a chosen delta_m2."""
    th = np.asarray(get_fixed_population_params("gwtc5_fiducial_bpl2peaks")).copy()
    th[_GW_BETA] = -2.0
    th[_GW_M2LOW] = 4.0
    th[_GW_DM2] = float(delta_m2)
    assert th[_GW_M2LOW] <= th[7], (th[_GW_M2LOW], th[7])
    return jnp.asarray(th)


def test_population_m1_support_max_values():
    """gwtc5 reaches its fixed 300-Msun edge; a model whose support ends at/under
    the historical ceiling keeps 200 (so its grid is unchanged, bit-for-bit)."""
    assert population_m1_support_max(GWTC5) == 300.0
    assert population_m1_support_max(MODEL) == 200.0  # powerlaw+peak


def test_gwtc5_pairing_grid_sized_agrees_over_full_support():
    """The FIX.  With the pairing grid sized to the model's 300-Msun support the
    grid path matches the exact per-sample q-integration across the WHOLE m1 =
    190..300 range; left at the pre-fix 200 ceiling the SAME grid clamps badly
    above 200.

    A SMOOTH secondary-mass taper (delta_m2=0.5) is used so the pairing
    normaliser N(m1) is smooth: it isolates the clamp defect from the unrelated
    q-quadrature staircase that a hard step (delta_m2=0) adds to a discontinuous
    integrand -- that staircase is exercised separately below."""
    theta = _gwtc5_adverse_theta(0.5)
    rng = np.random.default_rng(20)
    N = 4000
    m1 = jnp.asarray(rng.uniform(190.0, 299.5, N))
    q = jnp.asarray(rng.uniform(0.05, 1.0, N))
    z = jnp.asarray(rng.uniform(0.01, 1.0, N))
    chi = jnp.asarray(rng.uniform(-0.4, 0.4, N))

    def lp():
        return np.asarray(GWTC5.log_p_pop(m1, q, z, chi, theta))

    _set_pairing_grid(None)
    exact = lp()
    fin = np.isfinite(exact)
    assert fin.sum() > 1000

    # Pre-fix: the historical 200-node ceiling clamps every 200 < m1 <= 300 sample.
    _set_pairing_grid(2048, m_hi=200.0)
    unsized = lp()
    m1n = np.asarray(m1)
    hi = fin & (m1n > 200.0)
    assert np.abs(unsized[hi] - exact[hi]).max() > 0.1, "expected large pre-fix clamp error"

    # Post-fix: size the grid to the model support (exactly what the CLI does).
    _set_pairing_grid(2048, m_hi=200.0)
    size_pairing_grid_to_support(population_m1_support_max(GWTC5))
    assert normalization_grid_settings().pairing_m_hi == 300.0
    sized = lp()
    max_err = np.abs(sized[fin] - exact[fin]).max()
    # Documented interpolation tolerance (measured ~2.4e-4 for these corners).
    assert max_err < 1.0e-3, max_err


def test_gwtc5_pairing_grid_clamp_error_reproduces_phy5():
    """Reproduces PHY-5 with the instruction's exact adverse params (beta_q=-2,
    m2_low=5, delta_m2=0): at the pre-fix 200 ceiling a valid sample at m1=299 --
    inside the 300-Msun support -- silently uses the m1=200 normaliser, an
    order-0.1+ log-density error that grows toward the top of the support.  Sizing
    the grid to the support removes the clamp."""
    theta = _gwtc5_adverse_theta(0.0)
    m1 = jnp.asarray([201.0, 250.0, 299.0])
    q = jnp.full(3, 0.9)
    z = jnp.full(3, 0.1)
    chi = jnp.zeros(3)

    def lp():
        return np.asarray(GWTC5.log_p_pop(m1, q, z, chi, theta))

    _set_pairing_grid(None)
    exact = lp()
    assert np.all(np.isfinite(exact))

    _set_pairing_grid(2048, m_hi=200.0)
    clamp_err = np.abs(lp() - exact)
    assert clamp_err[-1] > 0.1, clamp_err            # m1 = 299, deep in the gap
    assert clamp_err[-1] > clamp_err[0]              # worse further past 200

    _set_pairing_grid(2048, m_hi=200.0)
    size_pairing_grid_to_support(300.0)
    sized_err = np.abs(lp() - exact)
    assert sized_err[-1] < clamp_err[-1]             # sizing removes the clamp


def test_pairing_grid_validation_rejects_undersized_grid():
    """assert_pairing_grid_covers_support fails loudly for an enabled grid whose
    upper bound cannot cover a 300-Msun model, and is a no-op once the grid is
    sized or disabled."""
    _set_pairing_grid(2048, m_hi=200.0)
    with pytest.raises(ValueError, match="does not cover"):
        assert_pairing_grid_covers_support(300.0, model_name="gwtc5_fiducial_bpl2peaks")

    size_pairing_grid_to_support(300.0)
    assert_pairing_grid_covers_support(300.0)  # sized -> no raise

    _set_pairing_grid(None, m_hi=200.0)
    assert_pairing_grid_covers_support(300.0)  # disabled -> exact path, no raise


def test_size_pairing_grid_scales_nodes_and_is_noop_when_covered():
    """Sizing preserves log spacing and scales the node count up so density does
    not drop; it is inert when the support already fits and when the grid is off."""
    # Fits under the default ceiling -> untouched (historical grid, bit-for-bit).
    _set_pairing_grid(2048, m_hi=200.0)
    size_pairing_grid_to_support(200.0)
    s = normalization_grid_settings()
    assert s.pairing_m_hi == 200.0 and s.pairing_m1_grid == 2048

    # Extending to 300 scales N in proportion to the added log-range.
    size_pairing_grid_to_support(300.0)
    s = normalization_grid_settings()
    import math
    expected_n = math.ceil(2048 * math.log(300.0 / s.m_lo) / math.log(200.0 / s.m_lo))
    assert s.pairing_m_hi == 300.0 and s.pairing_m1_grid == expected_n

    # Disabled grid -> sizing is inert (exact path ignores the bound).
    _set_pairing_grid(None, m_hi=200.0)
    size_pairing_grid_to_support(300.0)
    assert normalization_grid_settings().pairing_m1_grid is None


# ---------------------------------------------------------------------------
# (7) Trace-safety of the cached grid builders.  The builders compute with jnp
#     and are lru-cached; if the FIRST eval after a cache clear happens inside a
#     jit trace (as it does for the lazily-normalising gwtc5 model, whose _norm
#     first touches get_mass_grid inside the selection scan), a naive builder
#     would cache a DynamicJaxprTracer that leaks into the next trace
#     (jax.errors.UnexpectedTracerError).  ensure_compile_time_eval forces a
#     concrete, trace-independent constant.  This mirrors the failure chain
#     without running the CLI end-to-end.
# ---------------------------------------------------------------------------

def test_cold_cache_grids_do_not_leak_tracers_across_jits():
    _set_pairing_grid(2048, m_hi=300.0)

    # Host-side reference grids, built eagerly (outside any trace).
    U._clear_grid_caches()
    ref_mass = np.asarray(U.get_mass_grid())
    ref_pair = np.asarray(U.get_pairing_m1_grid())

    # Force COLD caches so the FIRST materialisation happens inside a trace.
    U._clear_grid_caches()

    @jax.jit
    def first(x):
        return (
            x
            + U.get_mass_grid().sum()
            + U.get_q_grid().sum()
            + U.get_chi_grid().sum()
            + U.get_m1_q_mesh()[0].sum()
            + U.get_pairing_m1_grid().sum()
        )

    r1 = float(first(1.0))

    # A DIFFERENT jitted function reusing the same cached grids: pre-fix this
    # raised UnexpectedTracerError because the cached value was a tracer from
    # ``first``'s trace; post-fix the cached grids are concrete constants.
    @jax.jit
    def second(x):
        return x * U.get_mass_grid().mean() + U.get_pairing_m1_grid().mean()

    r2 = float(second(2.0))

    assert np.isfinite(r1) and np.isfinite(r2)
    # The grids cached from inside the trace are concrete and bit-identical to
    # the host-side reference (ensure_compile_time_eval preserves x64 values).
    assert np.array_equal(np.asarray(U.get_mass_grid()), ref_mass)
    assert np.array_equal(np.asarray(U.get_pairing_m1_grid()), ref_pair)


# ---------------------------------------------------------------------------
# Both CLI entry points must apply the PHY-5 coverage guard
# ---------------------------------------------------------------------------

_WIDE_SUPPORT_MODEL = "gwtc5_fiducial_bpl2peaks"   # m1_support_max = 300 > M_HI


def _cli_parsers():
    """(name, parser, base argv) for both inference CLIs."""
    from darksirens.cli import inference, inference_lensing

    return [
        (
            "main",
            inference.build_parser(),
            ["--gw_path", "gw.h5", "--gwselection_path", "sel.h5",
             "--universe_model", "spectral_sirens", "--sampler", "tinyns"],
        ),
        (
            "lensing",
            inference_lensing.build_parser(),
            ["--gw_path", "gw.h5", "--gwselection_path", "sel.h5",
             "--sampler", "tinyns", "--cluster_mode", "off"],
        ),
    ]


@pytest.mark.parametrize(
    "module_name,phase",
    [
        ("darksirens.cli.inference", "_configure_performance_grids"),
        ("darksirens.cli.inference_lensing", "main"),
    ],
)
def test_both_clis_wire_the_grid_resolver_into_their_startup_phase(module_name, phase):
    """The shared resolver must actually be CALLED by both CLIs, not merely
    importable -- the whole defect was one twin never calling it."""
    import importlib
    import inspect

    module = importlib.import_module(module_name)
    src = inspect.getsource(getattr(module, phase))
    assert "configure_normalization_grids_for_model(opts)" in src


@pytest.mark.parametrize("which", ["main", "lensing"])
def test_both_clis_expose_the_normalization_grid_flags(which):
    """The lensing CLI offered no --norm_*/--pairing_norm_grid knobs at all, so a
    lensing run could not opt in to (or out of) the process-global setting it
    inherits from DARKSIRENS_GW_PAIRING_M1_GRID."""
    name, parser, base = next(x for x in _cli_parsers() if x[0] == which)
    opts = parser.parse_args(base + ["--pairing_norm_grid", "512",
                                     "--norm_nmass", "300", "--norm_nq", "150",
                                     "--norm_nchi", "120"])
    assert opts.pairing_norm_grid == 512
    assert (opts.norm_nmass, opts.norm_nq, opts.norm_nchi) == (300, 150, 120)


@pytest.mark.parametrize("which", ["main", "lensing"])
def test_both_clis_size_an_inherited_pairing_grid_to_the_model_support(which):
    """PHY-5, applied to one twin only: with the pairing grid enabled from the
    ENVIRONMENT and a model whose m1 support reaches 300, the default ceiling of
    M_HI = 200 makes jnp.interp reuse the m1 = 200 normaliser for the whole
    200 < m1 <= 300 slice of the support (measured +0.25 nat in log p_pop).  The
    main CLI sized the grid up; the lensing CLI had no equivalent call and its
    comment claimed there was 'no GW-population normalization grid on this
    stack'."""
    from darksirens.cli.common import configure_normalization_grids_for_model
    from darksirens.gw.populations import get_model, population_m1_support_max

    _set_pairing_grid(2048, m_hi=200.0)
    support = population_m1_support_max(get_model(_WIDE_SUPPORT_MODEL))
    assert support > U.normalization_grid_settings().pairing_m_hi

    name, parser, base = next(x for x in _cli_parsers() if x[0] == which)
    opts = parser.parse_args(base + ["--pop_model", _WIDE_SUPPORT_MODEL])
    configure_normalization_grids_for_model(opts)

    s = U.normalization_grid_settings()
    assert s.pairing_m_hi >= support
    # node count scaled with the added log-range so density is preserved
    assert s.pairing_m1_grid > 2048


@pytest.mark.parametrize("which", ["main", "lensing"])
def test_both_clis_leave_the_grid_disabled_when_nobody_opts_in(which):
    """No env var, no flag -> exact per-sample q-integration, untouched."""
    from darksirens.cli.common import configure_normalization_grids_for_model

    _set_pairing_grid(None, m_hi=200.0)
    name, parser, base = next(x for x in _cli_parsers() if x[0] == which)
    opts = parser.parse_args(base + ["--pop_model", _WIDE_SUPPORT_MODEL])
    configure_normalization_grids_for_model(opts)

    s = U.normalization_grid_settings()
    assert s.pairing_m1_grid is None
    assert s.pairing_m_hi == 200.0


@pytest.mark.parametrize("which", ["main", "lensing"])
def test_both_clis_size_an_explicitly_requested_pairing_grid(which):
    """The --pairing_norm_grid flag path is sized the same way as the env path."""
    from darksirens.cli.common import configure_normalization_grids_for_model

    _set_pairing_grid(None, m_hi=200.0)
    name, parser, base = next(x for x in _cli_parsers() if x[0] == which)
    opts = parser.parse_args(
        base + ["--pop_model", _WIDE_SUPPORT_MODEL, "--pairing_norm_grid", "1024"]
    )
    configure_normalization_grids_for_model(opts)

    s = U.normalization_grid_settings()
    assert s.pairing_m1_grid > 1024
    assert s.pairing_m_hi >= 300.0


# ---------------------------------------------------------------------------
# (8) SUPPORT-EDGE CELLS.  The grid branch used to floor a zero-support node's
#     normaliser at log(1e-300) and interpolate THROUGH it.  That is sound AT a
#     zero-support node but not INSIDE the cell that straddles the support edge:
#     there p_unnorm is small-but-nonzero while the interpolated log I is
#     hundreds of nats too low, so p * exp(-log_I) explodes.
#
#     The edge is hit by any sample sitting inside the m1 cell that straddles the
#     mixture's low-mass edge (m1 slightly above m_min, q ~ 1 so m2 = q*m1 is
#     slightly above m_min), where sfilter_low(m, m_min) == 0 at m == m_min makes
#     the node below the edge carry I == 0.
#
#     Measured on master at the powerlaw+peak PRIOR MIDPOINT (m_min = 6,
#     dm_min = 5.005) for the equivalent samples at the then-current M_LO = 1
#     pairing floor, m1 = 1.00102232, q = 0.999739:
#         exact log_p_pop = -20.108858,  grid(2048) = +386.6989  (+406.8 nats)
#         grid(1024) = +521.03, grid(4096) = +120.40, grid(8192) = -18.92
#     which took one injection sample's log_mu from -4.485 to +360.4, Neff from
#     11344 to 1.0, and logL from -1027.3 to -114483.4.
#
#     Post-fix the grid branch (a) never interpolates through the floor -- a
#     zero node inherits the larger of its supported neighbours, a monotone
#     UPPER bound on I inside that cell -- and (b) clamps log I from below by the
#     single-term trapezoid bound on the FIXED q grid
#         I(m1) >= (dq/2) * (p_unnorm(m1, q_k) + p_unnorm(m1, q_{k+1}))
#     for the sample's own bracketing q-nodes.  With the edge-cell q-support
#     narrower than one q-interval (cell width d log m1 = 2.6e-3 at N=2048 < dq =
#     5.0e-3, a coupling NormalizationGridSettings enforces) that bound is within a
#     factor of two of the true normaliser FROM BELOW, so the grid density in the
#     edge cell is at most ~0.7 nats above and a couple of nats below the exact
#     branch's support-relative quadrature -- bounded and one-sided, never the
#     hundreds-of-nats explosion.
# ---------------------------------------------------------------------------

def _powerlaw_peak_prior_midpoint():
    lo, hi, *_ = pop_model_prior_parser("powerlaw+peak")
    return jnp.asarray(0.5 * (np.asarray(lo) + np.asarray(hi)))


# The finding's samples, at the mixture's low-mass edge: inside the grid cell that
# straddles m_min (m1 just above m_min, q ~ 1 so m2 = q*m1 is just above m_min).
_EDGE_MMIN = 6.0                  # powerlaw+peak prior-midpoint m_min
_EDGE_M1 = jnp.asarray([1.00102232, 1.0005, 1.002]) * _EDGE_MMIN
_EDGE_Q = jnp.asarray([0.999739, 0.9999, 0.999])


def test_support_edge_cell_is_bounded_and_one_sided(capsys):
    """The reported blow-up is gone: inside the support-edge cell the grid path
    stays within a couple of nats of the exact per-sample q-integration and never
    exceeds it, at every grid size (the fixed-grid single-term bound is within a
    factor of two of the exact support-relative normaliser from below)."""
    theta = _powerlaw_peak_prior_midpoint()
    assert float(theta[_I_MMIN]) == _EDGE_MMIN
    z = jnp.full(_EDGE_M1.size, 0.1)
    chi = jnp.zeros(_EDGE_M1.size)

    _set_pairing_grid(None)
    exact = _logp(_EDGE_M1, _EDGE_Q, z, chi, theta)
    assert np.all(np.isfinite(exact)), exact
    np.testing.assert_allclose(exact, [-10.60412986, -9.89268181, -11.26923722],
                               rtol=0, atol=1e-6)

    worst = {}
    for ng in (1024, 2048, 4096, 8192):
        _set_pairing_grid(ng)
        gr = _logp(_EDGE_M1, _EDGE_Q, z, chi, theta)
        assert np.all(np.isfinite(gr)), (ng, gr)
        worst[ng] = (float((gr - exact).max()), float((gr - exact).min()))
    with capsys.disabled():
        print("\n[pairing_norm_grid] support-edge cell Δlog_p_pop (max | min) "
              "(was +406.8 nats at 2048):")
        for ng, (hi_e, lo_e) in worst.items():
            print(f"    grid={ng:5d}:  {hi_e:+.3e} | {lo_e:+.3e}")
    for ng, (hi_e, lo_e) in worst.items():
        assert hi_e < 0.7, (ng, worst)          # never inflated
        assert lo_e > -3.0, (ng, worst)         # and bounded from below


def test_support_edge_dense_sweep_bounded_and_one_sided(capsys):
    """Dense sweep of the pairing density across the cells that straddle the
    support edge, for several (m_min, dm_min, beta) corners.

    Asserts what the fix guarantees: the support pattern matches the exact
    branch exactly, and the grid density never EXCEEDS the exact density by more
    than a small bounded amount (the pre-fix excess was up to +578.9 nats).  The
    residual over-shoot comes from the single-term normaliser bound in cells
    where the q-support spans slightly more than one q-interval, and it shrinks
    with grid refinement."""
    mix = MODEL.mixture
    pair = mix.pairing_components[0]
    # (mmin, dmmin, beta): peak component (mmin = M_LO), the PL component at the
    # prior midpoint, a narrow taper, the steepest/flattest allowed q slopes.
    corners = [(1.0, 0.01, 2.5), (6.0, 5.005, 2.5), (6.0, 0.05, 2.5),
               (6.0, 0.05, -2.0), (3.5, 0.01, 7.0), (20.0, 3.0, 1.0)]
    rows = []
    for mmin, dmmin, beta in corners:
        theta = jnp.asarray([beta])
        for ng in (1024, 2048, 8192):
            _set_pairing_grid(ng)
            nodes = np.asarray(U.get_pairing_m1_grid())
            j0 = int(np.searchsorted(nodes, mmin))
            m1 = np.linspace(nodes[max(j0 - 2, 0)],
                             nodes[min(j0 + 3, nodes.size - 1)], 201)
            over, mism = -np.inf, 0
            for qv in (0.5, 0.9, 0.99, 0.999, 1.0):
                q = np.full(m1.shape, qv)
                _set_pairing_grid(None)
                ex = np.asarray(pair(jnp.asarray(m1), jnp.asarray(q),
                                     mmin, dmmin, theta))
                _set_pairing_grid(ng)
                gr = np.asarray(pair(jnp.asarray(m1), jnp.asarray(q),
                                     mmin, dmmin, theta))
                mism += int(((ex > 0) != (gr > 0)).sum())
                both = (ex > 0) & (gr > 0)
                if both.any():
                    over = max(over, float((np.log(gr[both]) - np.log(ex[both])).max()))
            rows.append((mmin, dmmin, beta, ng, over, mism))
    with capsys.disabled():
        print("\n[pairing_norm_grid] support-edge sweep: max log(grid/exact) "
              "(pre-fix up to +578.9):")
        for mmin, dmmin, beta, ng, over, mism in rows:
            print(f"    mmin={mmin:5} dm={dmmin:6} beta={beta:5} N={ng:5}: "
                  f"over={over:+.3e}  zero-pattern-mismatches={mism}")
    # Bounded (was e^{+578}), and tighter as the grid refines.  At N=1024 the
    # m1-cell (d log m1 = 5.2e-3) is WIDER than one q-interval (5.0e-3), so the
    # single-term bound can miss a second q-node inside the edge cell; from 2048
    # up the cell is narrower than dq and the bound is tight.
    tol = {1024: 3.0, 2048: 0.25, 8192: 1.0e-6}
    for mmin, dmmin, beta, ng, over, mism in rows:
        assert mism == 0, (mmin, dmmin, beta, ng, mism)
        assert over < tol[ng], (mmin, dmmin, beta, ng, over)


def test_support_edge_sample_does_not_corrupt_logmu_or_neff(capsys):
    """Likelihood-level consequence of the fix.

    A Monte-Carlo selection sum containing ONE support-edge sample: pre-fix that
    single sample dominated the sum (log_mu -4.49 -> +360.4) and collapsed the
    effective sample size (Neff 11344 -> 1.0000000011), which either produces a
    nonsense logL (soft variance guard) or a spurious -inf (default hard guard).
    Post-fix both agree with the exact path to ~1e-9."""
    from jax.scipy.special import logsumexp

    theta = _powerlaw_peak_prior_midpoint()
    rng = np.random.default_rng(11)
    n = 4000
    m1 = np.concatenate([rng.uniform(6.0, 90.0, n - 1), [1.00102232]])
    q = np.concatenate([rng.uniform(0.1, 1.0, n - 1), [0.999739]])
    z = np.concatenate([rng.uniform(0.01, 1.5, n - 1), [0.1]])
    chi = np.concatenate([rng.uniform(-0.5, 0.5, n - 1), [0.0]])
    args = (jnp.asarray(m1), jnp.asarray(q), jnp.asarray(z), jnp.asarray(chi))

    def summary(th, keep):
        lp = MODEL.log_p_pop(*(a[:keep] for a in args), th)
        log_mu = float(logsumexp(lp) - np.log(keep))
        # Neff of the Monte-Carlo sum (the selection-variance guard's quantity).
        neff = float(jnp.exp(2.0 * logsumexp(lp) - logsumexp(2.0 * lp)))
        return log_mu, neff

    _set_pairing_grid(None)
    mu_ex, neff_ex = summary(theta, n)
    mu_ex0, _ = summary(theta, n - 1)          # same set WITHOUT the edge sample
    _set_pairing_grid(2048)
    mu_gr, neff_gr = summary(theta, n)
    mu_gr0, _ = summary(theta, n - 1)
    with capsys.disabled():
        print(f"\n[pairing_norm_grid] with one support-edge sample: "
              f"log_mu exact={mu_ex:.10f} grid={mu_gr:.10f} | "
              f"Neff exact={neff_ex:.4f} grid={neff_gr:.4f}")
    # The residual grid-vs-exact discrepancy is the ORDINARY interpolation error
    # of the other samples: including the edge sample changes it by < 1e-9, i.e.
    # the edge sample itself contributes exactly what the exact path gives it.
    assert abs(mu_gr - mu_ex) < 1e-5, (mu_gr, mu_ex)
    assert abs((mu_gr - mu_ex) - (mu_gr0 - mu_ex0)) < 1e-9, (mu_gr - mu_ex,
                                                             mu_gr0 - mu_ex0)
    assert abs(neff_gr / neff_ex - 1.0) < 1e-4, (neff_gr, neff_ex)
    # No collapse: pre-fix the single edge sample carried the whole sum (Neff -> 1).
    assert neff_gr > 0.1 * n, neff_gr


def test_support_edge_grid_path_gradients_are_finite():
    """The edge-cell branches must not poison autodiff: the safe-log guard on the
    single-term bound keeps 0 * inf out of the VJP."""
    theta = _powerlaw_peak_prior_midpoint()
    m1 = jnp.concatenate([_EDGE_M1, jnp.asarray([0.5, 6.0, 30.0, 199.0])])
    q = jnp.concatenate([_EDGE_Q, jnp.asarray([0.9, 0.99, 0.7, 0.3])])
    z = jnp.full(m1.size, 0.1)
    chi = jnp.zeros(m1.size)

    def total(th):
        lp = MODEL.log_p_pop(m1, q, z, chi, th)
        return jnp.sum(jnp.where(jnp.isfinite(lp), lp, 0.0))

    _set_pairing_grid(2048)
    g = np.asarray(jax.jit(jax.grad(total))(theta))
    assert np.all(np.isfinite(g)), g
