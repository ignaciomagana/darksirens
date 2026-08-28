"""
test_pairing_edge_fix.py
------------------------
The pairing m1-grid's SUPPORT-EDGE treatment.

Until 2026-08-28 samples the m1 interpolant could not resolve were handed a
normaliser built from ONE interval of the FIXED q grid,
``I_lb = dq/2 * (p(m1, q_k) + p(m1, q_{k+1}))``.  Inside the edge cell the whole
q-support is narrower than one q-interval, so ``q_k <= m_min/m1`` (p there is 0)
and ``q_{k+1} = 1``: the normaliser collapsed to ``dq * p(m1, 1) / 2``, which
carries NO m1, m_min, or beta dependence at all.  The density at the top of the
support was therefore pinned to the constant ``2/dq = 2 (n_q - 1) = 398``,
measured 251x below the exact value at the powerlaw+peak prior midpoint and
unbounded as m1 -> m_min, with ``d log p / d m_min`` coming out -7.7e3 against a
true +1.67e7 (wrong sign, wrong order).

The replacement integrates I on the SAMPLE'S OWN support with a Gauss-Legendre
rule (``pairing_edge_nq`` nodes) wherever the node-level second difference of
log I says linear interpolation is outside ``pairing_edge_tol``.

Run with ``JAX_PLATFORMS=cpu``.
"""
import numpy as np

_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from dataclasses import replace as _dc_replace

import darksirens.gw.populations.utils as U
from darksirens.gw.populations.registry import get_model

MODEL = get_model("powerlaw+peak")
PAIR = MODEL.mixture.pairing_components[0]
N_REF = 200001

# (m_min, dm_min, beta): the prior midpoint, the fiducial, a narrow taper at
# both beta extremes, the lowest m_min with the widest taper, and the highest.
CORNERS = [(6.0, 5.005, 2.5), (5.0, 3.0, 1.0), (6.0, 0.05, 2.5),
           (6.0, 0.05, -2.0), (3.5, 0.01, 7.0), (2.0, 10.0, 0.0),
           (10.0, 10.0, 7.0)]


def _rewarm():
    U.get_mass_grid(); U.get_q_grid(); U.get_chi_grid(); U.get_m1_q_mesh()


@pytest.fixture(autouse=True)
def _restore_grid_settings():
    saved = U._NORMALIZATION_GRID_SETTINGS
    yield
    U._NORMALIZATION_GRID_SETTINGS = saved
    U._clear_grid_caches()
    _rewarm()


def _set_pairing_grid(n, **kw):
    U._NORMALIZATION_GRID_SETTINGS = _dc_replace(
        U._NORMALIZATION_GRID_SETTINGS, pairing_m1_grid=n, **kw)
    U._clear_grid_caches()
    _rewarm()


def _reference_norm(m1, m_min, dm_min, theta, chunk=20):
    """Converged support-relative trapezoid of the SAME integrand.

    ``N_REF = 200001`` nodes: 1000x the exact branch's n_q, which is what it
    takes to resolve the taper boundary layer inside the edge cell (the exact
    branch itself is 3.1e-2 off there -- see the module note)."""
    t = jnp.linspace(0.0, 1.0, N_REF)
    out = np.empty(np.shape(m1))
    for s in range(0, np.size(m1), chunk):
        mm = jnp.asarray(np.asarray(m1)[s:s + chunk])
        q_cut = jnp.clip(m_min / mm, 0.0, 1.0)
        width = 1.0 - q_cut
        q_n = q_cut[:, None] + t * width[:, None]
        p = PAIR._eval_unnorm(mm[:, None], q_n, m_min, dm_min, theta)
        sc = jnp.max(p, axis=-1, keepdims=True)
        sc = jnp.where(sc > 0, sc, 1.0)
        out[s:s + chunk] = np.asarray(
            jnp.trapezoid(p / sc, dx=1.0 / (N_REF - 1), axis=-1) * width
            * sc[:, 0])
    return out


def _edge_points(m_min, n_cells=24, n=120):
    """m1 inside the first ``n_cells`` grid cells above the support edge."""
    s = U.normalization_grid_settings()
    dlog = np.log(s.pairing_m_hi / s.m_lo) / (s.pairing_m1_grid - 1)
    return m_min * np.exp(np.linspace(1e-8, n_cells * dlog, n))


# ---------------------------------------------------------------------------
# 1. The headline pin: near the support edge the grid path now tracks a
#    CONVERGED reference, and does so better than the exact branch it
#    approximates.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_grid", [1056, 2048])
def test_support_edge_tracks_a_converged_reference(n_grid, capsys):
    rows = []
    for m_min, dm_min, beta in CORNERS:
        theta = jnp.asarray([beta])
        _set_pairing_grid(n_grid)
        m1 = _edge_points(m_min)
        q = m_min / m1 + 0.7 * (1.0 - m_min / m1)
        p = np.asarray(PAIR._eval_unnorm(jnp.asarray(m1), jnp.asarray(q),
                                         m_min, dm_min, theta))
        ref = p / _reference_norm(m1, m_min, dm_min, theta)
        grid = np.asarray(PAIR(jnp.asarray(m1), jnp.asarray(q), m_min, dm_min,
                               theta))
        _set_pairing_grid(None)
        exact = np.asarray(PAIR(jnp.asarray(m1), jnp.asarray(q), m_min, dm_min,
                                theta))
        good = (ref > 0) & (grid > 0) & (exact > 0)
        assert good.all(), (m_min, dm_min, beta, "support pattern mismatch")
        e_grid = float(np.abs(np.log(grid[good]) - np.log(ref[good])).max())
        e_exact = float(np.abs(np.log(exact[good]) - np.log(ref[good])).max())
        rows.append((m_min, dm_min, beta, e_grid, e_exact))
    with capsys.disabled():
        print(f"\n[pairing_edge_fix] N_grid={n_grid}: max |Delta log p(q|m1)| "
              f"over the 24 cells above m_min, vs a {N_REF}-node reference "
              f"(pre-fix: up to 22.6 nats)")
        for m_min, dm_min, beta, eg, ee in rows:
            print(f"    m_min={m_min:5} dm={dm_min:6} beta={beta:5}: "
                  f"grid={eg:.3e}  exact-branch={ee:.3e}")
    for m_min, dm_min, beta, eg, ee in rows:
        # Absolute accuracy, and never worse than the branch it approximates.
        assert eg < 5.0e-3, (m_min, dm_min, beta, eg)
        assert eg <= ee + 1.0e-9, (m_min, dm_min, beta, eg, ee)


# ---------------------------------------------------------------------------
# 2. The specific collapse: the clamp pinned the density to a constant.
# ---------------------------------------------------------------------------

def test_edge_density_follows_the_one_over_width_law():
    """Deep inside the taper the conditional density at q = 1 must scale as
    1/(1 - m_min/m1): the support shrinks, the normalised density diverges.

    The removed clamp pinned it to 2/dq = 2 (n_q - 1) = 398 for EVERY m1."""
    m_min, dm_min, beta = 6.0, 5.005, 2.5
    theta = jnp.asarray([beta])
    deltas = np.array([1e-5, 1e-4, 1e-3])
    m1 = m_min * (1.0 + deltas)
    q = np.ones_like(m1)
    width = 1.0 - m_min / m1
    _set_pairing_grid(2048)
    grid = np.asarray(PAIR(jnp.asarray(m1), jnp.asarray(q), m_min, dm_min,
                           theta))
    _set_pairing_grid(None)
    exact = np.asarray(PAIR(jnp.asarray(m1), jnp.asarray(q), m_min, dm_min,
                            theta))
    # 1/width law to 1%, and no trace of the 398 pin.
    np.testing.assert_allclose(grid * width, np.ones(3), rtol=1.0e-2)
    np.testing.assert_allclose(grid, exact, rtol=5.0e-2)
    assert grid.min() > 500.0, grid          # the pin was 398 at every m1


# ---------------------------------------------------------------------------
# 3. The clamp destroyed the gradient; the fix restores it.
# ---------------------------------------------------------------------------

def test_edge_cell_gradients_track_the_exact_branch():
    m_min, dm_min, beta = 6.0, 5.005, 2.5
    m1 = jnp.asarray(np.concatenate([
        m_min * np.exp(np.linspace(1e-8, 0.02, 40)),
        np.linspace(12.0, 90.0, 40)]))
    q = jnp.clip(m_min / m1 + 0.7 * (1.0 - m_min / m1), 0.0, 1.0)

    def total(a):
        d = PAIR(m1, q, a[0], a[1], jnp.asarray([a[2]]))
        return jnp.sum(jnp.where(d > 0, jnp.log(jnp.where(d > 0, d, 1.0)), 0.0))

    x = jnp.asarray([m_min, dm_min, beta])
    _set_pairing_grid(None)
    g_exact = np.asarray(jax.jit(jax.grad(total))(x))
    _set_pairing_grid(2048)
    g_grid = np.asarray(jax.jit(jax.grad(total))(x))
    assert np.all(np.isfinite(g_grid)), g_grid
    # d/d m_min is the one the clamp inverted (-7.7e3 against +1.7e7).
    assert np.sign(g_grid[0]) == np.sign(g_exact[0]), (g_grid, g_exact)
    np.testing.assert_allclose(g_grid[0], g_exact[0], rtol=1.0e-4)
    np.testing.assert_allclose(g_grid[2], g_exact[2], rtol=1.0e-3)


# ---------------------------------------------------------------------------
# 4. Settings plumbing.
# ---------------------------------------------------------------------------

def test_edge_quadrature_settings_validate_and_reconfigure():
    s = U.normalization_grid_settings()
    assert s.pairing_edge_nq >= 2 and s.pairing_edge_tol > 0.0
    t, w = U._gauss_legendre_01(s.pairing_edge_nq)
    assert t.shape == w.shape == (s.pairing_edge_nq,)
    # open interval, unit total weight, symmetric
    assert float(t.min()) > 0.0 and float(t.max()) < 1.0
    np.testing.assert_allclose(float(w.sum()), 1.0, rtol=0, atol=1e-14)
    with pytest.raises(ValueError):
        _dc_replace(U._NORMALIZATION_GRID_SETTINGS, pairing_edge_nq=1)
    with pytest.raises(ValueError):
        _dc_replace(U._NORMALIZATION_GRID_SETTINGS, pairing_edge_tol=0.0)
    got = U.configure_normalization_grids(pairing_edge_nq=12,
                                          pairing_edge_tol=1e-3)
    assert got.pairing_edge_nq == 12 and got.pairing_edge_tol == 1e-3
    assert U.get_pairing_edge_quadrature()[0].shape == (12,)


def test_edge_quadrature_cold_cache_does_not_leak_tracers():
    """Same hygiene as the other grid builders: a first evaluation INSIDE a jit
    trace must cache a concrete constant, not a tracer bound to that trace."""
    _set_pairing_grid(1056)
    U._clear_grid_caches()

    @jax.jit
    def first(x):
        t, w = U.get_pairing_edge_quadrature()
        return x * t.sum() * w.sum()

    @jax.jit
    def second(x):
        t, _ = U.get_pairing_edge_quadrature()
        return x + t.mean()

    first(1.0)
    second(1.0)      # would raise UnexpectedTracerError if the cache leaked
    _rewarm()
