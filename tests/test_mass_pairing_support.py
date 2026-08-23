"""Declared-support contracts for the mass and pairing components.

Two leaks the common evaluation path did not close:

* PHY-08 -- a grammar :class:`MassComponent` is normalised over the mass
  grid but nothing masked its density to that grid.  The grammar Gaussian is
  positive for every finite mass, so at the fixed/custom corner mu = 100,
  sigma = 20 it returned p_G(201) = 5.78e-8 per solar mass, with 2.87e-7 of
  nominal normalisation above M_HI = 200 and 3.71e-7 below M_LO = 1 -- density
  scored against a normaliser that integrated neither.  Three bespoke models
  are documented exceptions and MUST keep their out-of-grid density: Golomb
  (its interpolation already carries left=right=0 edges), GWTC-3 (Eq. B4's G
  is normalized over the real line by construction), and GWTC-5 (which
  declares its own 300 Msun support above the shared 200 Msun ceiling).

* PHY-09 -- :class:`PairingModel` normalises p(q|m1) over (m_min/m1, 1], but
  the concrete models only require q > 0 and m2 >= m_min, so a row with
  m2 > m1 was scored with a finite density from outside the normalisation
  domain.  Measured on PowerLawPairing at beta = 1, m_min = 5, dm_min = 3,
  m1 = 30: the density integrated to 1.0000 over the support and still
  returned 2.52 at q = 1.2 and 4.20 at q = 2.  q = 1 (equal masses) is a
  physical boundary and stays IN support.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

from darksirens.gw.populations import parametric as par
from darksirens.gw.populations import utils as pop_utils
from darksirens.gw.populations.base import ParamSpec

M_LO = pop_utils.M_LO
M_HI = pop_utils.M_HI
EPS = 1.0e-6


def _spec(n):
    return [ParamSpec("x", 0.0, 1.0) for _ in range(n)]


# ---------------------------------------------------------------------------
# PHY-08: grammar mass components respect the normalisation grid
# ---------------------------------------------------------------------------
GRAMMAR_CASES = [
    ("powerlaw", par.PowerLaw(*_spec(5)),
     jnp.asarray([2.0, 5.0, 80.0, 3.0, 10.0])),
    ("brokenpowerlaw", par.BrokenPowerLaw(*_spec(7)),
     jnp.asarray([2.0, 4.0, 30.0, 5.0, 80.0, 3.0, 10.0])),
    ("peak", par.Gaussian(*_spec(2)), jnp.asarray([35.0, 5.0])),
    # The review's corner: a peak parked at the ceiling, reachable by a fixed
    # or custom-prior parameter even though the shipped prior stops at mu = 50.
    ("peak-at-the-ceiling", par.Gaussian(*_spec(2)),
     jnp.asarray([100.0, 20.0])),
]


@pytest.mark.parametrize("name,component,theta", GRAMMAR_CASES)
def test_grammar_component_is_zero_outside_the_grid(name, component, theta):
    outside = jnp.asarray([M_LO - EPS, M_HI + EPS, M_HI + 1.0, 10.0 * M_HI])
    dens = np.asarray(component(outside, theta))
    assert np.all(dens == 0.0), f"{name}: leaked density {dens} outside the grid"


@pytest.mark.parametrize("name,component,theta", GRAMMAR_CASES)
def test_grammar_component_keeps_its_density_inside_the_grid(name, component, theta):
    inside = jnp.asarray([M_LO + EPS, 0.5 * (M_LO + M_HI), M_HI - EPS])
    dens = np.asarray(component(inside, theta))
    assert np.all(np.isfinite(dens))
    assert np.all(dens >= 0.0)
    # The peak cases must actually carry density in the interior, or the test
    # above would pass on an all-zero component.
    if name.startswith("peak"):
        assert dens[1] > 0.0


def test_the_reviews_leaked_integral_is_gone():
    """p_G(201) = 5.78e-8 with 2.87e-7 of mass above the ceiling."""
    peak = par.Gaussian(*_spec(2))
    theta = jnp.asarray([100.0, 20.0])
    assert float(peak(jnp.asarray(201.0), theta)) == 0.0
    above = jnp.linspace(M_HI + EPS, 10.0 * M_HI, 200001)
    assert float(jnp.trapezoid(peak(above, theta), above)) == 0.0
    below = jnp.linspace(-M_HI, M_LO - EPS, 200001)
    assert float(jnp.trapezoid(peak(below, theta), below)) == 0.0


# ---------------------------------------------------------------------------
# PHY-08: the documented bespoke exceptions keep their out-of-grid density
# ---------------------------------------------------------------------------
def test_gwtc5_keeps_its_own_300_msun_support():
    mass = par.GWTC5FiducialBPL2PeaksPopulationModel().mass_component
    theta = jnp.asarray([2.0, 4.0, 30.0, 10.0, 3.0, 35.0, 5.0, 5.0, 3.0, 0.3, 0.2])
    assert mass.m1_support_max == 300.0
    above_ceiling = np.asarray(mass(jnp.asarray([201.0, 250.0, 299.0]), theta))
    assert np.all(above_ceiling > 0.0), above_ceiling
    # ...and its own edge is still an edge.
    assert float(mass(jnp.asarray(301.0), theta)) == 0.0


def test_golomb_keeps_its_interpolation_edges():
    """left=right=0 on the phi_1G table is the model's own support."""
    golomb = par.GolombRemnantMass1G(*_spec(5))
    theta = jnp.asarray([1.0, 4.0, 40.0, 10.0, 0.1])
    assert golomb.support == (None, None)
    out = np.asarray(golomb(jnp.asarray([201.0, 250.0]), theta))
    assert np.all(out > 0.0), out


def test_gwtc3_keeps_its_real_line_gaussian_tail():
    """Eq. B4's G is normalized over the real line, not truncated."""
    gwtc3 = par.GWTC3PowerLawPeakMass(*_spec(7))
    assert gwtc3.support == (None, None)
    # A peak wide enough for the tail to be numerically visible past M_HI.
    theta = jnp.asarray([3.5, 5.0, 87.0, 0.5, 50.0, 40.0, 4.8])
    out = np.asarray(gwtc3(jnp.asarray([201.0, 250.0]), theta))
    assert np.all(out > 0.0), out


# ---------------------------------------------------------------------------
# PHY-09: pairing models are zero outside 0 < q <= 1
# ---------------------------------------------------------------------------
M_MIN, DM_MIN = 5.0, 3.0

PAIRING_CASES = [
    ("powerlaw", par.PowerLawPairing(*_spec(1)), jnp.asarray([1.0])),
    ("gaussian", par.GaussianPairing(*_spec(2)), jnp.asarray([0.9, 0.2])),
    ("gwtc5", par.GWTC5FiducialBPL2PeaksPairing(*_spec(3)),
     jnp.asarray([1.0, 5.0, 3.0])),
]


@pytest.mark.parametrize("name,pairing,theta", PAIRING_CASES)
def test_pairing_is_zero_above_unit_mass_ratio(name, pairing, theta):
    m1 = jnp.full(4, 30.0)
    q = jnp.asarray([1.0 + 1e-9, 1.2, 2.0, 10.0])
    dens = np.asarray(pairing(m1, q, M_MIN, DM_MIN, theta))
    assert np.all(dens == 0.0), f"{name}: density {dens} outside q <= 1"


@pytest.mark.parametrize("name,pairing,theta", PAIRING_CASES)
def test_pairing_keeps_the_q_equals_one_boundary(name, pairing, theta):
    """Equal masses are physical and inside the normalisation domain."""
    dens = float(pairing(jnp.asarray(30.0), jnp.asarray(1.0), M_MIN, DM_MIN, theta))
    assert dens > 0.0, name


@pytest.mark.parametrize("name,pairing,theta", PAIRING_CASES)
def test_pairing_normalises_to_one_over_its_support(name, pairing, theta):
    m1 = 30.0
    q = jnp.linspace(M_MIN / m1, 1.0, 40001)
    dens = pairing(jnp.full_like(q, m1), q, M_MIN, DM_MIN, theta)
    assert float(jnp.trapezoid(dens, q)) == pytest.approx(1.0, rel=1e-4)


def test_pairing_grid_path_also_masks_q_above_one():
    """The opt-in pairing_m1_grid branch is a separate return path."""
    pairing = par.PowerLawPairing(*_spec(1))
    theta = jnp.asarray([1.0])
    m1 = jnp.full(3, 30.0)
    q = jnp.asarray([0.9, 1.0, 1.2])
    exact = np.asarray(pairing(m1, q, M_MIN, DM_MIN, theta))
    try:
        pop_utils.configure_normalization_grids(pairing_m1_grid=2048)
        grid = np.asarray(pairing(m1, q, M_MIN, DM_MIN, theta))
    finally:
        pop_utils.configure_normalization_grids(pairing_m1_grid=None)
    assert exact[2] == 0.0 and grid[2] == 0.0
    np.testing.assert_allclose(grid[:2], exact[:2], rtol=1e-6)


def test_pairing_gradient_is_finite_at_and_across_the_boundary():
    """The mask must not turn the beta channel into a NaN at q >= 1."""
    pairing = par.PowerLawPairing(*_spec(1))

    def logp(beta, q):
        d = pairing(jnp.asarray(30.0), q, M_MIN, DM_MIN, jnp.asarray([beta]))
        return jnp.log(jnp.where(d > 0, d, 1.0))

    for q in (0.9, 1.0, 1.2):
        g = jax.grad(logp)(1.0, jnp.asarray(q))
        assert np.isfinite(float(g)), q
