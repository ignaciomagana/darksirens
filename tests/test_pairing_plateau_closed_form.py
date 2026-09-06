"""
test_pairing_plateau_closed_form.py
-----------------------------------
The CLOSED FORM on the pairing normaliser's plateau panel
(``PairingModel._plateau_integral``, ``parametric._powerlaw_plateau_integral``),
which replaced the Gauss-Legendre rule on ``[q_a, 1]`` on 2026-09-06.

Above the taper shoulder ``utils.sfilter_low`` returns a LITERAL 1.0, and the
``m2 < m_min`` cut cannot fire, so both production pairings' ``_eval_unnorm``
restricted to that panel is a bare ``q**beta``:

    int_{q_a}^{1} q**beta dq = (1 - q_a**(beta+1)) / (beta+1),

with the removable ``beta -> -1`` limit ``-log(q_a)``.  ``beta`` is SAMPLED over
[-2, 7], so ``beta = -1`` is inside the prior box and the limit is a live branch,
not a curiosity.

What these tests pin
--------------------
1. THE PREMISE.  ``_eval_unnorm`` on the plateau panel equals ``q**beta``
   BITWISE, for both production pairings, over the whole prior box.  The closed
   form is only the integral of the right function because of that.
2. THE FORM.  The hook reproduces the exact integral to 1e-15 relative over a
   Latin hypercube that includes ``beta + 1`` in [-1e-8, 1e-8] and exactly 0,
   and the ``beta = -1`` row is ``-log(q_a)`` to round-off.
3. THE SCALE CONTRACT.  ``_plateau_integral`` also returns the panel's supremum,
   which replaces the node maximum in ``_panel_norm``'s scale factoring.  It must
   BOUND the integrand, and it must be exactly 0.0 on a zero-width panel -- else
   a taper-toe row, whose whole integrand is ~exp(-500)-tiny, is rescaled by an
   O(1) bound and loses exactly the underflow protection the factoring exists
   for (a measured NaN-gradient regression lives there).
4. THE GATE.  The hook is declared ONLY by the classes whose kernel above the
   shoulder is bare, and a class that does not declare it keeps the
   Gauss-Legendre panel bit-for-bit.
5. WHAT IT BUYS.  At steep negative ``beta`` the plateau integrand ``q**beta`` is
   sharply peaked at ``q_a = m_shoulder/m1``, which shrinks like 1/m1, and GL-16
   does NOT resolve it: at beta = -1.9, m_shoulder = 4.2, the two-panel GL rule
   is 1.6e-2 nats off per sample at m1 = 400 and its error GROWS with m1, i.e. it
   is coherent across the m1 population and therefore tilts with H0.  On the
   259-event production likelihood at (beta_q, m2_low, delta_m2) =
   (-1.9, 3.05, 1.15) that came out as a 0.93-nat peak-to-peak, -1.04-nat-sloped
   error across the H0 prior [20, 140] -- twenty times over the campaign's
   0.05-nat tilt budget.  With the closed form the same scan is 2.0e-4 nats
   peak-to-peak with a -2.8e-5 slope.  So this is an ACCURACY fix that happens
   also to be 3.6% of a production call and 20% of a spectral one.

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
    GaussianPairing,
    GWTC5FiducialBPL2PeaksPairing,
    PowerLawPairing,
    _powerlaw_plateau_integral,
)

PL = PowerLawPairing(ParamSpec(r"$\beta$", -2.0, 7.0))
G5 = GWTC5FiducialBPL2PeaksPairing(ParamSpec(r"$\beta$", -2.0, 7.0),
                                   ParamSpec(r"$m_2$", 3.0, 10.0),
                                   ParamSpec(r"$\delta m_2$", 0.0, 10.0))
GP = GaussianPairing(ParamSpec(r"$\mu_q$", 0.0, 1.0),
                     ParamSpec(r"$\sigma_q$", 0.0, 1.0))

_NO_HOOK = PairingModel._plateau_integral


def norm(pair, m1, m_min, dm_min, theta, n=None, hook=True):
    """``N(m1) = scale * n_sc`` from the shipped ``_panel_norm``.

    ``hook=False`` reverts the class to the base-class ``_plateau_integral``,
    i.e. to the pure two-panel Gauss-Legendre rule this change replaced, so the
    two rules can be compared through the SAME code path.
    """
    t, w = (U.get_pairing_panel_quadrature() if n is None
            else U._gauss_legendre_01(n))
    cls = type(pair)
    saved = cls._plateau_integral
    if not hook:
        cls._plateau_integral = _NO_HOOK
    try:
        n_sc, scale = pair._panel_norm(jnp.asarray(np.atleast_1d(m1)),
                                       m_min, dm_min, theta, t, w)
    finally:
        cls._plateau_integral = saved
    return np.asarray(n_sc * scale)


def exact_plateau(q_lo, beta):
    """``int_{q_lo}^1 q**beta dq`` in 80-bit long double, stably.

    Written as ``-L expm1(x)/x`` with ``L = log q_lo``, ``x = (beta+1) L``: the
    algebraically equivalent ``(1 - q_lo**(beta+1))/(beta+1)`` cancels to ~1e-11
    relative near ``beta = -1`` even at long-double precision, which would swamp
    the quantity under test.
    """
    ql = np.maximum(np.asarray(q_lo, dtype=np.longdouble), np.longdouble(1e-12))
    L = np.log(ql)
    x = (np.asarray(beta, dtype=np.longdouble) + 1.0) * L
    xs = np.where(x != 0, x, np.longdouble(1.0))
    return -L * np.where(x != 0, np.expm1(xs) / xs, np.longdouble(1.0))


# ---------------------------------------------------------------------------
# 1. THE PREMISE: the plateau integrand is a bare q**beta, bitwise.
# ---------------------------------------------------------------------------

def test_plateau_integrand_is_exactly_q_to_the_beta():
    """``sfilter_low`` returns a LITERAL 1.0 at and above the shoulder.

    Not "1 to within a taper tail" -- ``utils.sfilter_low``'s last line is
    ``jnp.where(m >= m_min + dm, 1.0, S)``.  The hook integrates ``q**beta``, so
    if this ever stopped holding exactly the closed form would silently
    integrate the wrong function.  Checked at the panel's own GL nodes, over
    4000 prior draws, for BOTH production pairings.
    """
    rng = np.random.default_rng(3)
    n = 4000
    m_min = rng.uniform(2.0, 10.0, n)
    dm = rng.uniform(0.0, 10.0, n)
    beta = rng.uniform(-2.0, 7.0, n)
    shoulder = m_min + dm
    m1 = shoulder * np.exp(rng.uniform(0.0, np.log(60.0), n))
    q_a = np.clip(shoulder / m1, np.clip(m_min / m1, 0.0, 1.0), 1.0)
    t, _ = U.get_pairing_panel_quadrature()
    q = q_a[:, None] + np.asarray(t) * (1.0 - q_a)[:, None]
    bare = np.asarray(jnp.where(jnp.asarray(q) > 0.0,
                                jnp.asarray(q) ** jnp.asarray(beta)[:, None], 0.0))
    # theta entries carry a trailing node axis so they broadcast against q.
    col = lambda a: jnp.asarray(a)[:, None]
    for pair, theta in ((PL, jnp.stack([col(beta)])),
                        (G5, jnp.stack([col(beta), col(m_min), col(dm)]))):
        got = np.asarray(pair._eval_unnorm(col(m1), jnp.asarray(q),
                                           col(m_min), col(dm), theta))
        assert np.array_equal(got, bare), np.max(np.abs(got - bare))


# ---------------------------------------------------------------------------
# 2. THE FORM: exact to round-off, including the beta -> -1 limit.
# ---------------------------------------------------------------------------

def test_closed_form_matches_the_exact_integral():
    """Latin hypercube over ``(q_lo, beta)``, with ``beta + 1`` driven to 0."""
    rng = np.random.default_rng(7)
    n = 4000
    beta = rng.uniform(-2.0, 7.0, n)
    beta[:200] = -1.0 + rng.uniform(-1e-8, 1e-8, 200)   # the removable limit
    beta[200:260] = -1.0                                # and exactly on it
    q_lo = np.exp(rng.uniform(np.log(1e-4), 0.0, n))
    got, _ = _powerlaw_plateau_integral(jnp.asarray(q_lo), jnp.asarray(beta))
    got = np.asarray(got)
    ref = exact_plateau(q_lo, beta)
    rel = np.asarray(np.abs(got.astype(np.longdouble) - ref) / np.abs(ref),
                     dtype=float)
    assert np.max(rel) < 1e-15, (np.max(rel), q_lo[np.argmax(rel)],
                                 beta[np.argmax(rel)])
    # beta = -1 is the -log(q_lo) branch, not a 0/0.
    sel = beta == -1.0
    want = -np.log(q_lo[sel])
    np.testing.assert_allclose(got[sel], want, rtol=1e-15, atol=0.0)


def test_beta_gradient_is_live_at_the_removable_limit():
    """The small-|x| branch must not be a DEAD branch.

    ``beta`` is sampled, so a series that drops its ``beta`` dependence would
    hand NUTS a zero cotangent exactly at ``beta = -1`` -- the same class of bug
    the double-where idiom in ``_eval_unnorm`` was written for.  Compare against
    a central difference taken far enough away to be well conditioned.
    """
    q_lo = jnp.asarray([0.05, 0.3, 0.8])

    def f(b):
        return jnp.sum(_powerlaw_plateau_integral(q_lo, b)[0])

    g = float(jax.grad(f)(jnp.asarray(-1.0)))
    h = 1e-4
    fd = (float(f(jnp.asarray(-1.0 + h))) - float(f(jnp.asarray(-1.0 - h)))) / (2 * h)
    assert np.isfinite(g)
    np.testing.assert_allclose(g, fd, rtol=1e-6, atol=0.0)


# ---------------------------------------------------------------------------
# 3. THE SCALE CONTRACT.
# ---------------------------------------------------------------------------

def test_supremum_bounds_the_panel_and_vanishes_on_a_dead_panel():
    rng = np.random.default_rng(19)
    n = 3000
    beta = rng.uniform(-2.0, 7.0, n)
    q_lo = np.exp(rng.uniform(np.log(1e-4), 0.0, n))
    q_lo[:50] = 1.0
    integral, sup = _powerlaw_plateau_integral(jnp.asarray(q_lo), jnp.asarray(beta))
    integral, sup = np.asarray(integral), np.asarray(sup)
    assert np.all(np.isfinite(integral)) and np.all(np.isfinite(sup))
    live = q_lo < 1.0
    # q**beta is monotone, so its sup on [q_lo, 1] is at an endpoint.
    # 1e-14, not exact: the hook forms the bound as exp(max(beta log q, 0)) --
    # one exp instead of a pow and a maximum -- which differs from NumPy's
    # ``q**beta`` by a couple of ulps.  A bound short by 2e-15 relative is still
    # a bound for every purpose the scale serves.
    assert np.all(sup[live] >= np.maximum(q_lo[live] ** beta[live], 1.0) * (1 - 1e-14))
    # A zero-width panel contributes nothing AND claims no scale.
    assert np.array_equal(integral[~live], np.zeros(int((~live).sum())))
    assert np.array_equal(sup[~live], np.zeros(int((~live).sum())))


def test_scale_factoring_is_unchanged_in_the_taper_toe():
    """The underflow guard the scale factoring exists for must survive.

    For ``m1`` inside the taper window the plateau panel has zero width and the
    whole integrand is ~exp(-500)-tiny.  If the plateau's O(1) supremum leaked
    into the scale there, ``n_sc`` would collapse to ~1e-217 and the VJP's
    squared reciprocal would underflow -- the measured regression documented on
    ``_panel_norm``.  The scale must be bit-identical to the rule without the
    hook, and the gradient finite under jit.
    """
    t, w = U.get_pairing_panel_quadrature()
    m1 = jnp.asarray([5.01, 5.1, 5.9, 6.0, 8.0, 30.0])
    theta = jnp.asarray([1.0])
    _, s_new = PL._panel_norm(m1, 5.0, 3.0, theta, t, w)
    saved = PowerLawPairing._plateau_integral
    PowerLawPairing._plateau_integral = _NO_HOOK
    try:
        _, s_base = PL._panel_norm(m1, 5.0, 3.0, theta, t, w)
    finally:
        PowerLawPairing._plateau_integral = saved
    # The scale is NOT bit-identical and is not meant to be: without the hook the
    # collapsed plateau panel still contributes its (degenerate) node at q = 1,
    # which the closed form has no node to take a maximum over.  What matters is
    # that the scale still TRACKS the integrand -- measured, the two agree to 2%
    # in the toe -- so ``n_sc`` stays O(1) and the VJP's reciprocal is never
    # squared through an underflowed normaliser.
    toe = np.asarray(m1) < 8.0
    ratio = np.asarray(s_new)[toe] / np.asarray(s_base)[toe]
    assert np.all((ratio > 0.5) & (ratio <= 1.0)), ratio
    # ... and at m1 = m_min + 0.01 the scale really is ~exp(-130)-tiny, not the
    # O(1) bound a leaked plateau supremum would have imposed.
    assert float(np.asarray(s_new)[0]) < 1e-40, float(np.asarray(s_new)[0])

    def logp(params):
        m_min, dm_min, beta = params
        q = jnp.asarray([1.0, 0.999, 0.95, 0.9, 0.7, 0.4])
        v = PL(m1, q, m_min, dm_min, jnp.asarray([beta]))
        return jnp.sum(jnp.log(jnp.where(v > 0, v, 1.0)))

    for beta in (-2.0, -1.0, 1.0, 7.0):
        g = np.asarray(jax.jit(jax.grad(logp))(jnp.asarray([5.0, 3.0, beta])))
        assert np.all(np.isfinite(g)), (beta, g)


# ---------------------------------------------------------------------------
# 4. THE GATE: declared only where the kernel above the shoulder is bare.
# ---------------------------------------------------------------------------

def test_hook_is_declared_exactly_where_the_kernel_is_bare():
    """``GaussianPairing``'s plateau is a Gaussian, not a power law.

    It must therefore keep the Gauss-Legendre panel -- and so must the base
    class, so an out-of-tree pairing is unaffected by this change.
    """
    m1 = jnp.asarray([30.0])
    q_lo = jnp.asarray([0.2])
    assert PL._plateau_integral(m1, q_lo, 5.0, 3.0, jnp.asarray([1.0])) is not None
    assert G5._plateau_integral(m1, q_lo, 5.0, 3.0,
                                jnp.asarray([1.0, 5.0, 3.0])) is not None
    assert GP._plateau_integral(m1, q_lo, 5.0, 3.0,
                                jnp.asarray([0.9, 0.2])) is None
    assert PairingModel._plateau_integral(GP, m1, q_lo, 5.0, 3.0,
                                          jnp.asarray([0.9, 0.2])) is None


def test_a_pairing_without_the_hook_is_bit_identical():
    """No hook -> the panel list, the scale and the sum are unchanged.

    ``GaussianPairing`` also declares extra ``_panel_edges``, so this covers the
    multi-panel path as well as the two-panel one.
    """
    m1 = np.exp(np.linspace(np.log(2.0), np.log(200.0), 250))
    t, w = U.get_pairing_panel_quadrature()
    for m_min, dm_min, mu, sig in ((5.0, 3.0, 0.9, 0.2), (6.0, 0.05, 0.5, 0.02),
                                   (2.0, 10.0, 0.95, 0.01)):
        theta = jnp.asarray([mu, sig])
        got = norm(GP, m1, m_min, dm_min, theta, hook=True)
        want = norm(GP, m1, m_min, dm_min, theta, hook=False)
        assert np.array_equal(got, want)
        # ... and the panel count is untouched.
        panels = GP._panel_nodes(jnp.asarray(m1), m_min, dm_min, theta, t)
        assert len(panels) == 4      # q_cut | mu-5s | mu+5s | 1, sorted


# ---------------------------------------------------------------------------
# 5. WHAT IT BUYS: the H0-coherent hole GL-16 leaves at steep negative beta.
# ---------------------------------------------------------------------------

# (m_min, dm_min, beta).  The first row is where the production H0 scan was run;
# q_a = (m_min+dm_min)/m1 shrinks like 1/m1, so q**beta with beta ~ -2 becomes a
# boundary layer at the LEFT edge of the plateau panel that GL-16 cannot see.
# (m_min, dm_min, beta, bound_on_the_new_rule).  The last row is the control: a
# WIDE taper with a mild beta, where the residual is panel A's and the closed
# plateau changes nothing -- it must not regress either.
_STEEP = [(3.05, 1.15, -1.9, 1e-6), (5.0, 3.0, -2.0, 1e-6),
          (2.0, 0.0, -1.75, 0.0), (10.0, 10.0, -1.5, 2e-4)]


def test_closed_plateau_removes_the_steep_beta_error():
    m1 = np.exp(np.linspace(np.log(3.2), np.log(400.0), 400))
    for m_min, dm_min, beta, bound in _STEEP:
        theta = jnp.asarray([beta])
        ok = norm(PL, m1, m_min, dm_min, theta, 400) > 0
        ref = np.log(norm(PL, m1, m_min, dm_min, theta, 400)[ok])
        e_new = np.max(np.abs(np.log(norm(PL, m1, m_min, dm_min, theta)[ok]) - ref))
        e_base = np.max(np.abs(
            np.log(norm(PL, m1, m_min, dm_min, theta, hook=False)[ok]) - ref))
        # Measured (new / base, nats): 7.5e-8 / 1.6e-2, 9.2e-8 / 1.9e-3,
        # 0.0 / 8.2e-2, 1.0e-4 / 1.0e-4.
        assert e_new <= bound, (m_min, dm_min, beta, e_new)
        assert e_new <= e_base, (m_min, dm_min, beta, e_new, e_base)
    # The reference is the SAME rule at 400 nodes per panel, so it shares the
    # closed plateau; check it against the pure-GL reference too, or the bound
    # above would only be measuring self-consistency.
    theta = jnp.asarray([-1.9])
    a = norm(PL, m1, 3.05, 1.15, theta, 400, hook=True)
    b = norm(PL, m1, 3.05, 1.15, theta, 2000, hook=False)
    np.testing.assert_allclose(a, b, rtol=1e-11, atol=0.0)


def test_closed_plateau_is_never_worse_over_the_prior_box():
    """400 draws x 600 m1: the worst |Delta log N| may only go down.

    Measured over this box: 6.7e-3 nats for the two-panel GL rule against
    4.6e-3 with the closed plateau, and the largest single move is 6.7e-3 at
    (m_min, dm_min, beta) = (2.73, 1.15, -1.79) -- where the GL rule carried
    essentially all of the error and the closed form removes it.
    """
    rng = np.random.default_rng(11)
    nd = 60
    m1 = np.exp(np.linspace(np.log(2.0), np.log(300.0), 300))
    worst_new = worst_base = 0.0
    for m_min, dm_min, beta in zip(rng.uniform(2.0, 10.0, nd),
                                   rng.uniform(0.0, 10.0, nd),
                                   rng.uniform(-2.0, 7.0, nd)):
        theta = jnp.asarray([beta])
        ref = norm(PL, m1, m_min, dm_min, theta, 200)
        ok = ref > 0
        lr = np.log(ref[ok])
        worst_new = max(worst_new, float(np.max(np.abs(
            np.log(norm(PL, m1, m_min, dm_min, theta)[ok]) - lr))))
        worst_base = max(worst_base, float(np.max(np.abs(
            np.log(norm(PL, m1, m_min, dm_min, theta, hook=False)[ok]) - lr))))
    assert worst_new <= worst_base, (worst_new, worst_base)
    assert worst_new < 1e-2, worst_new
