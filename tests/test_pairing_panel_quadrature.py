"""
test_pairing_panel_quadrature.py
--------------------------------
The pairing normaliser's TWO-PANEL Gauss-Legendre rule
(``PairingModel._taper_shoulder`` / ``_panel_nodes`` / ``_panel_norm``,
``utils.get_pairing_panel_quadrature``), which replaced the 200-node uniform
support-relative trapezoid on 2026-09-05.

``N(m1) = int_{q_cut}^{1} p(q|m1) dq`` is integrated as two panels split at the
taper shoulder ``q_a = m_shoulder/m1``: a Planck-taper boundary layer on
``[q_cut, q_a]`` and the bare (analytic) pairing kernel on ``[q_a, 1]``.  The
joint integrand has a corner at ``q_a`` that neither piece has; splitting there
hands Gauss-Legendre two smooth pieces.

What these tests pin
--------------------
1. The rule is what it says it is: an INDEPENDENT NumPy reimplementation of the
   two-panel split reproduces the shipped normaliser to floating point.  Since
   2026-09-06 the upper panel is a closed form rather than a quadrature
   (``PairingModel._plateau_integral``, pinned in
   ``tests/test_pairing_plateau_closed_form.py``), so the reimplementation
   integrates it in closed form too; everything below is unchanged by that, and
   the ``reference_norm`` these bounds are measured against is still a pure
   composite Gauss-Legendre rule.
2. The BOUND: worst |Delta log N| against a composite-Gauss-Legendre reference
   over prior draws, both NEAR THE SUPPORT EDGE (7.6e-3 nats) and over the FULL
   prior box (7.4e-8, at m_min = 8.42, dm_min = 8.67, beta = -0.84, m1 = 15.8).
   Both residuals now belong to the TAPER panel: since 2026-09-06 the plateau is
   integrated in closed form, which is why the box bound is five orders of
   magnitude under the near-edge one instead of 6x over it (the pure-GL plateau
   was 4.6e-2 at m_min = 2, dm_min = 0, beta = -2, m1 = 250, growing with
   m1/q_a).  Also the strictly smaller H0-COHERENT component that is
   the reason the change is admissible at all (a smaller worst case does NOT
   imply a smaller tilt -- measured inversions exist -- so the tilt is asserted
   directly, at the amplitude-worst corners too).
2b. The PANEL CONTRACT: GL-16 per panel is only right for a kernel that is
   smooth on the scale of a panel.  A subclass whose kernel carries a narrower
   feature declares its edges through ``PairingModel._panel_edges``;
   ``GaussianPairing`` does, and is 2.7e+01 nats off without them.
3. The GATE: the split must come from the pairing's OWN taper parameters.
   ``GWTC5FiducialBPL2PeaksPairing`` deletes the caller's ``(m_min, dm_min)`` and
   tapers on ``theta = (beta, m2_low, delta_m2)``, so it overrides
   ``_taper_shoulder``.  Today's single caller happens to pass ``m2_low`` as
   ``m_min``; ``MixtureModel.component_densities`` would pass the MASS
   component's floor instead.  The test drives that mismatch directly and also
   shows the base-class hook FAILS it, so the override is load-bearing.
4. Degenerate panels: ``dm_min = 0`` (panel A collapses), ``m_shoulder >= m1``
   (panel B collapses), ``m_min >= m1`` (both collapse -> exactly 0.0).
5. Node staticness (one compile across proposals) and NaN-free gradients under
   jit in the taper toe, the two properties the old rule's comments record as
   having been paid for once already.

Run with ``JAX_PLATFORMS=cpu``.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import darksirens.gw.populations.utils as U
from darksirens.gw.populations.base import ParamSpec, PairingModel
from darksirens.gw.populations.parametric import (
    GWTC5FiducialBPL2PeaksPairing,
    PowerLawPairing,
)

PL = PowerLawPairing(ParamSpec(r"$\beta$", -2.0, 7.0))
G5 = GWTC5FiducialBPL2PeaksPairing(ParamSpec(r"$\beta$", -2.0, 7.0),
                                   ParamSpec(r"$m_2$", 3.0, 10.0),
                                   ParamSpec(r"$\delta m_2$", 0.0, 10.0))

_GLX, _GLW = np.polynomial.legendre.leggauss(16)
_GLT, _GLWT = 0.5 * (_GLX + 1.0), 0.5 * _GLW


def _panels(m1, q_cut, q_a, n_sub):
    """Sub-panel edges and GL-16 nodes on each of ``n_sub`` sub-intervals."""
    edges = q_cut[:, None] + (q_a - q_cut)[:, None] * np.linspace(0.0, 1.0, n_sub + 1)
    h = np.diff(edges, axis=1)
    return edges[:, :-1, None] + h[:, :, None] * _GLT, h


def reference_norm(pair, m1, m_min, dm_min, theta, n_a=256, n_b=32):
    """Composite Gauss-Legendre reference for ``N(m1)``.

    Same two panels, but each subdivided into many sub-intervals with GL-16 on
    each -- a rule of a different (much higher) total order on the same split.
    Self-converged: doubling both sub-panel counts moves it by <1e-15 relative.
    """
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    m_edge, m_shoulder = pair._taper_shoulder(m_min, dm_min, theta)
    q_cut = np.clip(float(m_edge) / m1, 0.0, 1.0)
    q_a = np.clip(float(m_shoulder) / m1, q_cut, 1.0)
    out = np.zeros(m1.shape)
    for lo, hi, n_sub in ((q_cut, q_a, n_a), (q_a, np.ones_like(q_a), n_b)):
        qn, h = _panels(m1, lo, hi, n_sub)
        p = np.asarray(pair._eval_unnorm(jnp.asarray(m1)[:, None, None],
                                         jnp.asarray(qn), m_min, dm_min, theta))
        out += np.sum(np.sum(p * _GLWT, axis=-1) * h, axis=-1)
    return out


def numpy_two_panel_norm(pair, m1, m_min, dm_min, theta):
    """INDEPENDENT NumPy reimplementation of the shipped rule (no scale factoring).

    Written from the specification -- ``q_cut = m_edge/m1``,
    ``q_a = clip(m_shoulder/m1, q_cut, 1)``, GL-16 on the taper panel and the
    closed form ``(1 - q_a**(beta+1))/(beta+1)`` on the plateau, where the
    integrand is a bare ``q**beta`` -- not from the jax code, so agreement pins
    the implementation, not a shared helper.  ``beta`` is ``theta[0]`` for both
    production pairings.
    """
    m1 = np.atleast_1d(np.asarray(m1, dtype=float))
    m_edge, m_shoulder = pair._taper_shoulder(m_min, dm_min, theta)
    q_cut = np.clip(float(m_edge) / m1, 0.0, 1.0)
    q_a = np.clip(float(m_shoulder) / m1, q_cut, 1.0)
    w = q_a - q_cut
    qn = q_cut[:, None] + w[:, None] * _GLT
    p = np.asarray(pair._eval_unnorm(jnp.asarray(m1)[:, None], jnp.asarray(qn),
                                     m_min, dm_min, theta))
    out = np.sum(p * _GLWT, axis=-1) * w
    beta = float(theta[0])
    b1 = beta + 1.0
    plateau = (np.log(1.0 / q_a) if b1 == 0.0
               else (1.0 - q_a ** b1) / b1)
    return out + np.where(q_a < 1.0, plateau, 0.0)


def shipped_norm(pair, m1, m_min, dm_min, theta):
    """``N(m1)`` as the shipped exact branch forms it (scale * n_sc)."""
    t, w = U.get_pairing_panel_quadrature()
    n_sc, scale = pair._panel_norm(jnp.asarray(np.atleast_1d(m1)),
                                   m_min, dm_min, theta, t, w)
    return np.asarray(n_sc * scale)


# (m_min, dm_min, beta): prior midpoint, fiducial, narrow tapers at both beta
# extremes, widest taper at the lowest and highest m_min.
CORNERS = [(6.0, 5.005, 2.5), (5.0, 3.0, 1.0), (6.0, 0.05, 2.5),
           (6.0, 0.05, -2.0), (3.5, 0.01, 7.0), (2.0, 10.0, 0.0),
           (10.0, 10.0, 7.0), (5.0, 3.0, -1.0)]


# ---------------------------------------------------------------------------
# 1. The rule is the rule: independent reimplementation, and the reference is
#    converged.
# ---------------------------------------------------------------------------

def test_matches_an_independent_numpy_two_panel_rule():
    m1 = np.exp(np.linspace(np.log(2.0), np.log(200.0), 250))
    for m_min, dm_min, beta in CORNERS:
        theta = jnp.asarray([beta])
        got = shipped_norm(PL, m1, m_min, dm_min, theta)
        want = numpy_two_panel_norm(PL, m1, m_min, dm_min, theta)
        ok = want > 0
        rel = np.abs(got[ok] / want[ok] - 1.0)
        # Only the row-max scale factoring differs (association order).
        assert np.max(rel) < 1e-12, (m_min, dm_min, beta, np.max(rel))
        # Off the support both must be exactly zero.
        assert np.array_equal(got[~ok], np.zeros(int((~ok).sum())))


def test_reference_is_converged():
    """The composite reference the bounds below are measured against."""
    m1 = np.exp(np.linspace(np.log(2.0), np.log(200.0), 120))
    for m_min, dm_min, beta in CORNERS:
        theta = jnp.asarray([beta])
        a = reference_norm(PL, m1, m_min, dm_min, theta, 256, 32)
        b = reference_norm(PL, m1, m_min, dm_min, theta, 512, 64)
        # Everywhere the normaliser is not itself an underflow sliver.  (Deep in
        # the taper toe, N ~ 1e-58, the composite rule's own summation loses
        # ~1e-8 relative -- irrelevant: nothing there survives any logsumexp.)
        ok = a > 1e-30
        assert ok.any()
        assert np.max(np.abs(b[ok] / a[ok] - 1.0)) < 1e-13, (m_min, dm_min, beta)
        pos = a > 0
        assert np.max(np.abs(b[pos] / a[pos] - 1.0)) < 1e-6, (m_min, dm_min, beta)


# ---------------------------------------------------------------------------
# 2. The bound, in both currencies -- amplitude AND coherence.
# ---------------------------------------------------------------------------


def _trap200(m1, m_min, dm_min, theta):
    """The 200-node uniform support-relative trapezoid the panel split replaced.

    Carried here so every comparison against it is measured, not quoted."""
    t = jnp.linspace(0.0, 1.0, 200)
    q_cut = jnp.clip(m_min / jnp.asarray(m1), 0.0, 1.0)
    width = 1.0 - q_cut
    qn = q_cut[:, None] + t * width[:, None]
    p = PL._eval_unnorm(jnp.asarray(m1)[:, None], qn, m_min, dm_min, theta)
    return np.asarray(jnp.trapezoid(p, dx=1.0 / 199.0, axis=-1) * width)


def test_worst_case_bound_and_improvement_over_the_trapezoid(capsys):
    """Worst |Delta log N| NEAR THE SUPPORT EDGE, 50 prior draws.

    This is the near-edge bound only: the m1 sweep spans the 24 grid cells above
    m_min, i.e. m1/m_min in [1, 1.064], which is where the trapezoid's endpoint
    deficit is worst and where the opt-in grid branch's interpolant gives up.
    The rule's error GROWS with m1/q_a, so the bound over the whole prior box is
    a different (larger) number -- see
    :func:`test_worst_case_bound_over_the_full_prior_box`.

    The old 200-node uniform support-relative trapezoid is carried here
    explicitly so the comparison is measured, not quoted."""
    trap200 = _trap200

    rng = np.random.default_rng(7)
    draws = [(float(rng.uniform(2.0, 10.0)), float(rng.uniform(0.0, 10.0)),
              float(rng.uniform(-2.0, 7.0))) for _ in range(50)]
    # The 24 grid cells above the support edge, the region the m1 interpolant
    # cannot resolve and where the trapezoid is worst.
    dlog = np.log(U.normalization_grid_settings().pairing_m_hi
                  / U.normalization_grid_settings().m_lo) / (2048 - 1)
    worst_new = worst_old = 0.0
    for m_min, dm_min, beta in draws:
        theta = jnp.asarray([beta])
        m1 = m_min * np.exp(np.linspace(1e-8, 24 * dlog, 120))
        ref = reference_norm(PL, m1, m_min, dm_min, theta)
        for fn, key in ((shipped_norm, "new"), (trap200, "old")):
            v = fn(PL, m1, m_min, dm_min, theta) if key == "new" else fn(
                m1, m_min, dm_min, theta)
            ok = (ref > 0) & (v > 0)
            if not ok.any():
                continue
            e = float(np.max(np.abs(np.log(v[ok]) - np.log(ref[ok]))))
            if key == "new":
                worst_new = max(worst_new, e)
            else:
                worst_old = max(worst_old, e)
    with capsys.disabled():
        print(f"\n[pairing_panel] worst |Delta log N| over the 24 cells above "
              f"m_min, 50 prior draws, vs a composite-GL reference:\n"
              f"    GL-16 + closed plateau (shipped): {worst_new:.3e}\n"
              f"    200-node trapezoid (pre-2026-09-05): {worst_old:.3e}")
    assert worst_new < 1.5e-2, worst_new           # documented NEAR-EDGE bound
    assert worst_new < 0.5 * worst_old, (worst_new, worst_old)


def test_worst_case_bound_over_the_full_prior_box(capsys):
    """The bound over the WHOLE support, not just the cells above m_min.

    Historically this was the LARGER of the two bounds: with the plateau panel
    integrated by Gauss-Legendre its residual grew with m1/q_a -- the panel
    ``[q_a, 1]`` gets longer and, at a steeply negative beta, its integrand
    becomes a boundary layer at the panel's own left edge -- so the near-edge
    bound did not bound the prior box (4.6e-2 nats, at (2.0, 0.0, -2.0),
    m1 = 250).  Since 2026-09-06 that panel is a closed form
    (``PairingModel._plateau_integral``), so what is left here is the TAPER
    panel's residual, which is largest at a WIDE taper and mild beta.  Swept
    over the reachable powerlaw+peak corners (m_min in [2, 10], dm_min in
    [0, 10], beta in [-2, 7]) plus 40 draws, m1 log-spaced over [m_min, 250]:

        GL-16 + closed plateau (shipped)     7.4e-8 nats, at
                                             (8.42, 8.67, -0.84), m1 = 15.8
        200-node trapezoid (pre-2026-09-05)  2.9e-1 nats

    i.e. better on EVERY draw (the loop fails if any draw inverts).  The
    statistic that decides admissibility is the coherent part, which
    :func:`test_h0_coherent_component_is_far_below_the_tilt_budget` pins at the
    near-edge-worst corners; at the box-worst corner above the coherent part is
    1.1e-3 nats and is the SAME for the pure-GL rule, because the plateau is not
    what carries it.
    """
    rng = np.random.default_rng(11)
    draws = [(2.0, 0.01, -2.0), (2.0, 0.0, -2.0), (3.0, 0.0, -2.0),
             (10.0, 0.05, -2.0), (2.0, 10.0, 0.0), (5.0, 3.0, 1.0),
             (10.0, 10.0, 7.0), (3.5, 0.01, 7.0)]
    draws += [(float(rng.uniform(2.0, 10.0)), float(rng.uniform(0.0, 10.0)),
               float(rng.uniform(-2.0, 7.0))) for _ in range(40)]
    worst_new = worst_old = 0.0
    for m_min, dm_min, beta in draws:
        theta = jnp.asarray([beta])
        m1 = np.exp(np.linspace(np.log(m_min), np.log(250.0), 60))
        m1 = np.clip(m1, m_min * (1.0 + 1e-9), None)
        ref = reference_norm(PL, m1, m_min, dm_min, theta)
        new_v = shipped_norm(PL, m1, m_min, dm_min, theta)
        old_v = _trap200(m1, m_min, dm_min, theta)
        ok = (ref > 0) & (new_v > 0) & (old_v > 0)
        if not ok.any():
            continue
        e_new = float(np.max(np.abs(np.log(new_v[ok]) - np.log(ref[ok]))))
        e_old = float(np.max(np.abs(np.log(old_v[ok]) - np.log(ref[ok]))))
        assert e_new <= e_old, (m_min, dm_min, beta, e_new, e_old)
        worst_new = max(worst_new, e_new)
        worst_old = max(worst_old, e_old)
    with capsys.disabled():
        print(f"\n[pairing_panel] worst |Delta log N| over the FULL prior box "
              f"(m1 in [m_min, 250], 48 corners/draws):\n"
              f"    GL-16 + closed plateau (shipped): {worst_new:.3e}\n"
              f"    200-node trapezoid (pre-2026-09-05): {worst_old:.3e}")
    assert worst_new < 1.0e-6, worst_new           # documented FULL-BOX bound
    assert worst_new < 0.25 * worst_old, (worst_new, worst_old)


def test_h0_coherent_component_is_far_below_the_tilt_budget(capsys):
    """The statistic that decides this rule: the POPULATION-AVERAGED error.

    ``m1src = m1det/(1+z(H0))``, so to first order an H0 move rescales the whole
    source-frame population by a common factor s.  The H0-correlated part of the
    259-event PE term is therefore ``259 * ptp_s <dlogN(m1 s)>``.  A smaller
    worst case does NOT imply a smaller tilt (measured inversions exist between
    quadrature rules), so it is asserted directly, against the 0.05-nat budget.
    """
    rng = np.random.default_rng(0)
    n = 6000
    u = rng.random(n)
    a, lo, hi = 3.5, 5.0, 100.0
    pl = (lo ** (1 - a) + u * (hi ** (1 - a) - lo ** (1 - a))) ** (1 / (1 - a))
    pop = np.where(rng.random(n) < 0.85, pl, np.clip(rng.normal(34.0, 4.0, n), lo, hi))
    svals = np.exp(np.linspace(np.log(0.75), np.log(1.35), 11))

    rows = []
    # The last three are the corners where the AMPLITUDE bound is worst
    # (steeply negative beta, lowest floor, narrowest taper): the amplitude and
    # the tilt do not order rules the same way, so they are pinned here too.
    for m_min, dm_min, beta in [(5.0, 3.0, 1.0), (5.0, 0.5, 1.0), (4.0, 5.0, -1.5),
                                (2.0, 0.05, -2.0), (2.0, 0.01, -1.5),
                                (10.0, 0.05, -2.0)]:
        theta = jnp.asarray([beta])
        means = []
        for s in svals:
            m1 = pop * s
            ref = reference_norm(PL, m1, m_min, dm_min, theta)
            got = shipped_norm(PL, m1, m_min, dm_min, theta)
            ok = (ref > 0) & (got > 0)
            means.append(float(np.mean(np.log(got[ok]) - np.log(ref[ok]))))
        rows.append((m_min, dm_min, beta, 259.0 * float(np.ptp(np.asarray(means)))))
    with capsys.disabled():
        print("\n[pairing_panel] H0-coherent component, 259 x ptp_s <dlogN>:")
        for m_min, dm_min, beta, tilt in rows:
            print(f"    m_min={m_min:5} dm={dm_min:6} beta={beta:5}: {tilt:.3e} nats")
    for m_min, dm_min, beta, tilt in rows:
        assert tilt < 5.0e-3, (m_min, dm_min, beta, tilt)


# ---------------------------------------------------------------------------
# 2b. THE PANEL CONTRACT: a kernel with a feature narrower than a panel must
#     declare that feature's edges (PairingModel._panel_edges).
# ---------------------------------------------------------------------------

def test_panel_edges_default_is_the_two_panel_split():
    """The default hook declares nothing, and the split is exactly two panels.

    Both production pairings take this path -- their kernel above the shoulder
    is a bare ``q**beta``, which GL-16 integrates to round-off -- so the default
    must stay the cheap two-panel rule.
    """
    theta = jnp.asarray([1.0])
    assert PL._panel_edges(jnp.asarray([30.0]), 5.0, 3.0, theta) == ()
    assert G5._panel_edges(jnp.asarray([30.0]), 5.0, 3.0,
                           jnp.asarray([1.0, 5.0, 3.0])) == ()
    t, _ = U.get_pairing_panel_quadrature()
    panels = PL._panel_nodes(jnp.asarray([30.0, 60.0]), 5.0, 3.0, theta, t)
    assert len(panels) == 2
    # ... and they tile the support (q_cut, 1] exactly, in order.
    q_cut = 5.0 / 30.0
    lo_a = float(panels[0][0][0, 0] - t[0] * panels[0][1][0])
    np.testing.assert_allclose(lo_a, q_cut, rtol=0, atol=1e-15)
    np.testing.assert_allclose(float(panels[0][1][0] + panels[1][1][0]),
                               1.0 - q_cut, rtol=0, atol=1e-15)


def _gaussian_pairing():
    from darksirens.gw.populations.parametric import GaussianPairing
    return GaussianPairing(ParamSpec(r"$\mu_q$", 0.0, 1.0),
                           ParamSpec(r"$\sigma_q$", 0.0, 1.0))


# (mu_q, sigma_q, m_min, dm_min, m1); sigma_q spans four decades of feature
# width, from far narrower than a panel to comparable with the whole support.
_GAUSS_CASES = [(0.9, 0.2, 5.0, 3.0, 60.0), (0.5, 0.02, 5.0, 3.0, 60.0),
                (0.95, 0.01, 5.0, 3.0, 12.0), (0.5, 0.02, 5.0, 0.05, 60.0),
                (0.3, 0.03, 5.0, 3.0, 20.0), (0.2, 0.01, 5.0, 8.0, 40.0),
                (0.6, 0.001, 5.0, 3.0, 60.0), (0.5, 0.5, 5.0, 3.0, 60.0),
                (0.15, 0.05, 5.0, 3.0, 60.0)]


def _quad_norm(pair, m1, m_min, dm_min, theta, breaks):
    """Gauss-Kronrod reference for N(m1) -- a different algorithm from the
    composite-GL rule used elsewhere in this file, broken at every panel edge."""
    quad = pytest.importorskip("scipy.integrate").quad

    def f(x):
        return float(np.asarray(pair._eval_unnorm(jnp.asarray(m1), jnp.asarray(x),
                                                  m_min, dm_min, theta)))
    pts = sorted({float(np.clip(b, breaks[0], 1.0)) for b in breaks} | {1.0})
    total = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        if hi > lo:
            total += quad(f, lo, hi, epsabs=1e-18, epsrel=1e-13, limit=400)[0]
    return total


def test_gaussian_pairing_resolves_a_feature_narrower_than_a_panel(capsys):
    """The panel contract, measured on the one shipped kernel that needs it.

    ``GaussianPairing`` puts a Gaussian of width ``sigma_q`` inside the panel
    above the shoulder.  GL-16 cannot see a feature it straddles no nodes of, so
    the class declares ``mu_q -/+ 5 sigma_q`` through ``_panel_edges``.  Against
    a Gauss-Kronrod reference over nine corners with sigma_q from 0.001 to 0.5,
    worst |Delta log N|: 2.7e+01 nats without the declared edges, 6.7e-7 with
    them.  The class is not grammar-registered today, but it is public API and
    it is what a registered Gaussian pairing would inherit.
    """
    GP = _gaussian_pairing()
    worst = 0.0
    for mu, sig, m_min, dm_min, m1 in _GAUSS_CASES:
        theta = jnp.asarray([mu, sig])
        got = shipped_norm(GP, m1, m_min, dm_min, theta)[0]
        q_cut = float(np.clip(m_min / m1, 0.0, 1.0))
        want = _quad_norm(GP, m1, m_min, dm_min, theta,
                          [q_cut, float(np.clip((m_min + dm_min) / m1, q_cut, 1.0)),
                           float(np.clip(mu - 8 * sig, q_cut, 1.0)),
                           float(np.clip(mu + 8 * sig, q_cut, 1.0))])
        worst = max(worst, abs(float(np.log(got / want))))
    with capsys.disabled():
        print(f"\n[pairing_panel] GaussianPairing, worst |Delta log N| over "
              f"9 corners (sigma_q 0.001 .. 0.5): {worst:.3e}")
    assert worst < 5.0e-6, worst


def test_the_default_hook_would_fail_that_contract(monkeypatch):
    """The declared edges are load-bearing: without them the same corners are
    off by tens of nats, which is why the contract is a contract."""
    from darksirens.gw.populations.parametric import GaussianPairing
    GP = _gaussian_pairing()
    monkeypatch.setattr(GaussianPairing, "_panel_edges", PairingModel._panel_edges)
    worst = 0.0
    for mu, sig, m_min, dm_min, m1 in _GAUSS_CASES:
        theta = jnp.asarray([mu, sig])
        got = shipped_norm(GP, m1, m_min, dm_min, theta)[0]
        q_cut = float(np.clip(m_min / m1, 0.0, 1.0))
        want = _quad_norm(GP, m1, m_min, dm_min, theta,
                          [q_cut, float(np.clip((m_min + dm_min) / m1, q_cut, 1.0)),
                           float(np.clip(mu - 8 * sig, q_cut, 1.0)),
                           float(np.clip(mu + 8 * sig, q_cut, 1.0))])
        worst = max(worst, abs(float(np.log(got / want))))
    assert worst > 1.0, worst


def test_declared_edges_are_clipped_and_sorted():
    """A declared edge outside the support costs a zero-width panel, never a
    wrong answer: mu_q below the shoulder or above 1 must still normalise."""
    GP = _gaussian_pairing()
    for mu, sig in ((0.02, 0.01), (1.4, 0.05), (0.5, 0.9)):
        theta = jnp.asarray([mu, sig])
        for m_min, dm_min, m1 in ((5.0, 3.0, 60.0), (5.0, 3.0, 6.0)):
            got = shipped_norm(GP, m1, m_min, dm_min, theta)[0]
            ref = reference_norm(GP, m1, m_min, dm_min, theta, 2048, 2048)[0]
            assert np.isfinite(got) and got >= 0.0, (mu, sig, m1)
            # Only where the normaliser is not itself an underflow sliver: at
            # mu_q = 0.02 with q_cut = 0.083 the whole support sits >6 sigma out
            # and N ~ 3e-16, where the composite reference's own summation is
            # the larger error.  Nothing there survives a logsumexp.
            if ref > 1e-12:
                np.testing.assert_allclose(got, ref, rtol=2e-5)


def test_zero_primary_mass_is_zero_with_finite_gradients():
    """m1 = 0 makes m_edge/m1 an inf whose clip VJP is 0 * inf = NaN.  The
    density there is exactly 0.0 either way (m2 = 0 < m_min kills every node),
    but the double-where in ``_panel_nodes`` keeps the gradient finite for a
    padded store.  Before it, d/d(m_min) AND d/d(dm_min) were both NaN."""
    m1 = jnp.asarray([0.0, 30.0])
    q = jnp.asarray([0.5, 0.5])
    val = np.asarray(PL(m1, q, 5.0, 3.0, jnp.asarray([1.0])))
    assert val[0] == 0.0 and val[1] > 0.0

    def f(p):
        return jnp.sum(PL(m1, q, p[0], p[1], jnp.asarray([p[2]])))

    g = np.asarray(jax.grad(f)(jnp.asarray([5.0, 3.0, 1.0])))
    assert np.all(np.isfinite(g)), g


# ---------------------------------------------------------------------------
# 3. THE GATE: the split comes from the pairing's OWN taper parameters.
# ---------------------------------------------------------------------------

def test_shoulder_hook_defaults_to_the_callers_floor():
    for m_min, dm_min in ((5.0, 3.0), (2.0, 0.0)):
        assert PL._taper_shoulder(m_min, dm_min, jnp.asarray([1.0])) == (
            m_min, m_min + dm_min)


def test_gwtc5_pairing_ignores_a_mismatched_caller_floor():
    """The production pairing tapers on theta, so the caller's floor must not
    move the panel split.

    ``GWTC5FiducialBPL2PeaksPairing._eval_unnorm`` deletes ``m_min``/``dm_min``
    and uses ``(m2_low, delta_m2)``.  The single caller today passes ``m2_low``
    as ``m_min``, but ``MixtureModel.component_densities`` hands a shared pairing
    the MASS component's ``(mmin, dmmin)``.  With the override the result is
    exactly independent of what the caller passes."""
    m1 = np.exp(np.linspace(np.log(3.0), np.log(200.0), 200))
    q = 0.35 + 0.6 * np.linspace(0.0, 1.0, 200)
    for beta, m2_low, delta_m2 in ((1.0, 5.0, 3.0), (-1.5, 8.0, 0.5),
                                   (4.0, 3.5, 9.0)):
        theta = jnp.asarray([beta, m2_low, delta_m2])
        matched = np.asarray(G5(jnp.asarray(m1), jnp.asarray(q),
                                m2_low, delta_m2, theta))
        for bad_m_min, bad_dm in ((2.0, 9.0), (10.0, 0.0), (0.0, 0.0)):
            other = np.asarray(G5(jnp.asarray(m1), jnp.asarray(q),
                                  bad_m_min, bad_dm, theta))
            assert np.array_equal(matched, other), (beta, bad_m_min, bad_dm)
        # ... and it is the RIGHT value, not merely a stable one.
        ref = reference_norm(G5, m1, 2.0, 9.0, theta)
        un = np.asarray(G5._eval_unnorm(jnp.asarray(m1), jnp.asarray(q),
                                        2.0, 9.0, theta))
        ok = (ref > 0) & (un > 0) & (matched > 0)
        assert ok.any()
        err = np.max(np.abs(np.log(matched[ok]) - np.log(un[ok] / ref[ok])))
        assert err < 1e-2, (beta, m2_low, delta_m2, err)


def test_the_base_class_hook_would_fail_that_gate(monkeypatch):
    """The override is load-bearing: with the DEFAULT hook the panel boundary
    lands off the shoulder and the normaliser is wrong by orders of magnitude."""
    m1 = np.exp(np.linspace(np.log(3.0), np.log(200.0), 200))
    beta, m2_low, delta_m2 = 1.0, 5.0, 3.0
    theta = jnp.asarray([beta, m2_low, delta_m2])
    good = shipped_norm(G5, m1, 2.0, 9.0, theta)
    monkeypatch.setattr(GWTC5FiducialBPL2PeaksPairing, "_taper_shoulder",
                        PairingModel._taper_shoulder)
    bad = shipped_norm(G5, m1, 2.0, 9.0, theta)
    ref = reference_norm(G5, m1, 2.0, 9.0, theta)
    ok = (ref > 0) & (good > 0) & (bad > 0)
    e_good = float(np.max(np.abs(np.log(good[ok]) - np.log(ref[ok]))))
    e_bad = float(np.max(np.abs(np.log(bad[ok]) - np.log(ref[ok]))))
    assert e_good < 1e-2, e_good
    assert e_bad > 50.0 * max(e_good, 1e-12), (e_good, e_bad)


# ---------------------------------------------------------------------------
# 4. Degenerate panels.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m_min,dm_min,beta,tag", [
    (5.0, 0.0, 1.0, "dm=0: panel A has zero width"),
    (5.0, 0.0, -2.0, "dm=0 at the beta floor"),
    (5.0, 40.0, 1.0, "shoulder above m1: panel B has zero width"),
    (5.0, 3.0, -1.0, "beta = -1 (no special case: panel B is quadrature)"),
])
def test_degenerate_panels_are_finite_and_correct(m_min, dm_min, beta, tag):
    m1 = np.asarray([5.5, 8.0, 30.0, 120.0])
    theta = jnp.asarray([beta])
    got = shipped_norm(PL, m1, m_min, dm_min, theta)
    ref = reference_norm(PL, m1, m_min, dm_min, theta)
    assert np.all(np.isfinite(got)), (tag, got)
    ok = ref > 0
    assert np.max(np.abs(got[ok] / ref[ok] - 1.0)) < 5e-3, (tag, got, ref)


def test_empty_support_returns_exactly_zero():
    """``m_min >= m1`` collapses BOTH panels; the ``n_sc > 0`` guard must return
    exactly 0.0 (not a small positive number, and not NaN)."""
    m1 = jnp.asarray([1.0, 3.0, 4.999, 5.0])
    q = jnp.asarray([0.5, 0.9, 1.0, 1.0])
    for beta in (-2.0, -1.0, 0.0, 7.0):
        v = np.asarray(PL(m1, q, 5.0, 3.0, jnp.asarray([beta])))
        assert np.array_equal(v, np.zeros(4)), (beta, v)


def test_out_of_support_q_is_zero():
    """q <= 0 and q > 1 are outside the interval the normaliser integrates."""
    m1 = jnp.asarray([30.0, 30.0, 30.0])
    q = jnp.asarray([-0.1, 0.0, 1.2])
    v = np.asarray(PL(m1, q, 5.0, 3.0, jnp.asarray([1.0])))
    assert np.array_equal(v, np.zeros(3)), v


# ---------------------------------------------------------------------------
# 5. Static nodes, and NaN-free gradients in the taper toe.
# ---------------------------------------------------------------------------

def test_panel_nodes_are_a_compile_time_constant():
    """The node set must not depend on any setting, or every proposal retraces."""
    t, w = U.get_pairing_panel_quadrature()
    t2, w2 = U.get_pairing_panel_quadrature()
    assert t is t2 and w is w2                      # cached, one array
    assert t.shape == (U.PAIRING_PANEL_NQ,)
    np.testing.assert_allclose(float(np.sum(np.asarray(w))), 1.0, rtol=0, atol=1e-15)

    def f(theta, m1, q):
        return PL(m1, q, 5.0, 3.0, theta)

    fj = jax.jit(f)
    m1, q = jnp.asarray([30.0, 12.0]), jnp.asarray([0.5, 0.9])
    for beta in (-2.0, 0.5, 7.0):
        fj(jnp.asarray([beta]), m1, q)
    assert fj._cache_size() == 1, fj._cache_size()


def test_gradients_are_finite_in_the_taper_toe_under_jit():
    """The scale factoring exists for this: at m1 = m_min + 0.01 both p and its
    q-integral are ~exp(-130)-tiny and an unfactored p/N poisons the cotangent
    (measured regression; it only reproduced under jit)."""
    def logp(params):
        m_min, dm_min, beta = params
        m1 = jnp.asarray([5.01, 5.1, 8.0, 30.0])
        q = jnp.asarray([1.0, 0.999, 0.7, 0.4])
        v = PL(m1, q, m_min, dm_min, jnp.asarray([beta]))
        return jnp.sum(jnp.log(jnp.where(v > 0, v, 1.0)))

    for beta in (-2.0, -1.0, 1.0, 7.0):
        g = np.asarray(jax.jit(jax.grad(logp))(jnp.asarray([5.0, 3.0, beta])))
        assert np.all(np.isfinite(g)), (beta, g)


# ---------------------------------------------------------------------------
# 6. The admissibility predicate the grid branch's invariant rests on.
# ---------------------------------------------------------------------------

def test_edge_rule_is_at_least_as_fine_as_the_exact_branch():
    """``tests/test_pairing_edge_fix.py`` asserts the grid branch is never worse
    than the exact branch it approximates.  Since both now run the SAME two-panel
    split, that holds BY CONSTRUCTION exactly while the edge rule carries at
    least as many nodes per panel.  Pin the default."""
    s = U.normalization_grid_settings()
    assert s.pairing_edge_nq >= U.PAIRING_PANEL_NQ, (s.pairing_edge_nq,
                                                     U.PAIRING_PANEL_NQ)
    t_e, w_e = U.get_pairing_edge_quadrature()
    assert t_e.shape == (s.pairing_edge_nq,)
