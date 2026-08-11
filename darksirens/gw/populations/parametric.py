"""Parametric building blocks for compact-binary population models.

This module defines the simple analytic distributions used by the population
registry.  Each class implements one of the abstract component interfaces from
:mod:`darksirens.gw.populations.base` and deliberately returns an *unnormalised*
density from ``_eval_unnorm``.  The base classes perform numerical
normalisation on fixed JAX grids so that the same component can be evaluated for
posterior samples and selection injections.

The ``ParamSpec`` fields stored on every dataclass describe the ordering,
labels, and prior bounds of the component parameters.  Registry constructors
combine these specs into a single inference vector; the methods here therefore
expect ``t`` to be ordered exactly as returned by ``param_specs``.
"""

import math
from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.special import erf, erfc

from .base import MassComponent, PairingModel, ParamSpec, SpinModel, pack_specs
from .components import (
    ALPHA,
    ALPHA_BPL,
    BETA,
    CHI_MU,
    CHI_SIGMA,
    DM_MAX,
    DM_MIN,
    GAUSS_MU,
    GAUSS_SIGMA,
    M_BREAK,
    M_MAX,
    M_MIN,
    ParamBlueprint,
    component,
    mass_token,
)
from .utils import get_mass_grid, get_q_grid, sfilter_high, sfilter_low


@mass_token("powerlaw", plural="powerlaws", short="PL", params=(
    ParamBlueprint("alpha", r"\alpha", ALPHA, 2.3),
    ParamBlueprint("m_min", r"m_{\min}", M_MIN, 5.0),
    ParamBlueprint("m_max", r"m_{\max}", M_MAX, 80.0),
    ParamBlueprint("dm_min", r"\delta m_{\min}", DM_MIN, 3.0),
    ParamBlueprint("dm_max", r"\delta m_{\max}", DM_MAX, 10.0),
))
@dataclass
class PowerLaw(MassComponent):
    """Smoothed primary-mass power law.

    Parameters are ``alpha``, ``m_min``, ``m_max``, ``dm_min``, and ``dm_max``.
    The unnormalised density is proportional to ``m**(-alpha)`` inside the mass
    range, with logistic low- and high-mass tapering supplied by
    :func:`sfilter_low` and :func:`sfilter_high`.  The component is useful as a
    baseline black-hole primary-mass spectrum or as one element of a mixture.
    """

    alpha_spec: ParamSpec
    m_min_spec: ParamSpec
    m_max_spec: ParamSpec
    dm_min_spec: ParamSpec
    dm_max_spec: ParamSpec

    @property
    def param_specs(self):
        """Return parameter metadata in the order consumed by ``_eval_unnorm``."""
        return [
            self.alpha_spec,
            self.m_min_spec,
            self.m_max_spec,
            self.dm_min_spec,
            self.dm_max_spec,
        ]

    @property
    def m1_support_max(self) -> float:
        """Support ends at the sampled high-mass edge (``sfilter_high`` is 0 for
        ``m >= m_max``), so the largest possible support is the ``m_max`` prior
        upper bound."""
        return float(self.m_max_spec.high)

    def _norm(self, theta):
        """Exact-in-the-edges normalisation (see :func:`_tapered_edge_norm`)."""
        a, mmin, mmax, dmmin, dmmax = (theta[0], theta[1], theta[2],
                                       theta[3], theta[4])
        return _tapered_edge_norm(
            lambda m: self._eval_unnorm(m, theta),
            lambda lo, hi: _powerlaw_segment_norm(lo, hi, 1.0, a),
            mmin, dmmin, mmax, dmmax,
        )

    def _eval_unnorm(self, m, t):
        """Evaluate the tapered unnormalised density at source-frame mass ``m``."""
        a, mmin, mmax, dmmin, dmmax = t[0], t[1], t[2], t[3], t[4]
        S = sfilter_low(m, mmin, dmmin) * sfilter_high(m, mmax, dmmax)
        return S * m ** (-a)


@mass_token("brokenpowerlaw", plural="brokenpowerlaws", short="BPL", params=(
    ParamBlueprint("alpha1", r"\alpha_1", ALPHA_BPL, 2.0),
    ParamBlueprint("alpha2", r"\alpha_2", ALPHA_BPL, 4.0),
    ParamBlueprint("m_break", r"m_{\rm break}", M_BREAK, 30.0),
    ParamBlueprint("m_min", r"m_{\min}", M_MIN, 5.0),
    ParamBlueprint("m_max", r"m_{\max}", M_MAX, 80.0),
    ParamBlueprint("dm_min", r"\delta m_{\min}", DM_MIN, 3.0),
    ParamBlueprint("dm_max", r"\delta m_{\max}", DM_MAX, 10.0),
))
@dataclass
class BrokenPowerLaw(MassComponent):
    """Two-slope primary-mass power law with a continuous break.

    Below ``m_break`` the spectrum follows ``m**(-alpha1)``.  Above the break it
    follows ``m**(-alpha2)`` multiplied by the continuity factor
    ``m_break**(alpha2 - alpha1)``.  Low- and high-mass smoothing parameters
    apply in the same way as :class:`PowerLaw`.
    """

    alpha1_spec: ParamSpec
    alpha2_spec: ParamSpec
    m_break_spec: ParamSpec
    m_min_spec: ParamSpec
    m_max_spec: ParamSpec
    dm_min_spec: ParamSpec
    dm_max_spec: ParamSpec

    @property
    def param_specs(self):
        """Return parameter metadata in broken-power-law order."""
        return [
            self.alpha1_spec,
            self.alpha2_spec,
            self.m_break_spec,
            self.m_min_spec,
            self.m_max_spec,
            self.dm_min_spec,
            self.dm_max_spec,
        ]

    @property
    def m1_support_max(self) -> float:
        """Support ends at the sampled high-mass edge; the largest possible value
        is the ``m_max`` prior upper bound."""
        return float(self.m_max_spec.high)

    def _norm(self, theta):
        """Exact-in-the-edges normalisation (see :func:`_tapered_edge_norm`)."""
        a1, a2, mb, mmin, mmax, dmmin, dmmax = (theta[0], theta[1], theta[2],
                                                theta[3], theta[4], theta[5],
                                                theta[6])
        # This component's plateau is ``m**-a1`` / ``mb**(a2-a1) m**-a2``, i.e.
        # ``mb**-a1`` times broken_powerlaw_norm's ``(m/mb)**-a`` integrand.
        return _tapered_edge_norm(
            lambda m: self._eval_unnorm(m, theta),
            lambda lo, hi: (mb ** (-a1)
                            * broken_powerlaw_norm(a1, a2, mb, lo, hi)),
            mmin, dmmin, mmax, dmmax,
        )

    def _eval_unnorm(self, m, t):
        """Evaluate the smoothed, continuous broken power law at ``m``."""
        a1, a2, mb, mmin, mmax, dmmin, dmmax = (
            t[0],
            t[1],
            t[2],
            t[3],
            t[4],
            t[5],
            t[6],
        )
        S = sfilter_low(m, mmin, dmmin) * sfilter_high(m, mmax, dmmax)
        join = mb ** (a2 - a1)
        p = jnp.where(m < mb, m ** (-a1), join * m ** (-a2))
        return S * p


@mass_token("peak", plural="peaks", short="G", params=(
    ParamBlueprint("mu", r"\mu", GAUSS_MU, 35.0),
    ParamBlueprint("sigma", r"\sigma", GAUSS_SIGMA, 5.0),
))
@dataclass
class Gaussian(MassComponent):
    """Unnormalised Gaussian primary-mass peak.

    This component is typically mixed with a power-law continuum to model an
    excess around a characteristic mass.  It has parameters ``mu`` and
    ``sigma`` and does not include explicit truncation; normalisation over the
    global mass grid is handled by :class:`MassComponent`.
    """

    mu_spec: ParamSpec
    sigma_spec: ParamSpec

    @property
    def param_specs(self):
        """Return ``mu`` and ``sigma`` parameter specs."""
        return [self.mu_spec, self.sigma_spec]

    def _norm(self, theta):
        """Exact ``int_{m_lo}^{m_hi} exp(-(m-mu)^2/2sigma^2) dm``.

        Same normalisation domain as the base class's grid trapezoid (the
        component has no truncation of its own), but exact: the trapezoid carries
        an O((h/sigma)^2) error -- 1.3% at the ``sigma`` prior floor of 1 Msun
        against h = 0.4 -- which enters the mixture as a spurious rescaling of
        this component's weight relative to the (now exactly normalised) power
        laws.
        """
        grid = get_mass_grid()
        return truncated_gaussian_norm(theta[0], theta[1], grid[0], grid[-1])

    def _eval_unnorm(self, m, t):
        """Evaluate the Gaussian peak at primary mass ``m``."""
        mu, sig = t[0], t[1]
        return jnp.exp(-0.5 * ((m - mu) / sig) ** 2)


# ── Analytic truncated-component normalisers ─────────────────────────────────
# Components whose truncation edge is a SAMPLED parameter cannot be normalised by
# a trapezoid over a fixed mass grid: the hard ``m >= m_low`` mask makes the
# quadrature a step function of ``m_low`` (it jumps whenever the edge crosses a
# node) and kills the ``d/dm_low`` channel entirely (a boolean mask has zero
# derivative), while a grid with h = 0.6 Msun cannot resolve a peak with
# sigma <~ h/2 at all.  The closed forms below are exact for every parameter
# value and differentiable in the truncation edge.

_SQRT2 = math.sqrt(2.0)
_SQRT_PI_2 = math.sqrt(0.5 * math.pi)
_NORM_SIGMA_FLOOR = 1.0e-12


def _pos(n):
    """Divide-safe normaliser: a zero/negative norm leaves the density untouched
    (its numerator is zero there too), matching ``MassComponent.__call__``."""
    return jnp.where(n > 0.0, n, 1.0)


def truncated_gaussian_norm(mu, sigma, lo, hi):
    r"""Exact :math:`\int_{lo}^{hi} \exp[-(m-\mu)^2/(2\sigma^2)]\,dm`.

    Equals :math:`\sigma\sqrt{2\pi}\,[\Phi((hi-\mu)/\sigma) - \Phi((lo-\mu)/\sigma)]`.
    The difference of CDFs is formed through ``erfc`` on whichever side keeps it
    well conditioned (both arguments positive, both negative, or straddling), so
    it stays accurate for a peak far outside ``[lo, hi]`` instead of cancelling
    two values of 1.  Every branch is finite for any finite input -- ``erfc``
    saturates at 0/2 -- so no ``where``-guard is needed and the gradients (which
    carry a ``exp(-x^2)`` factor) underflow to 0 rather than blowing up.

    A zero ``sigma`` is inside the GWTC-5 priors; it is floored to 1e-12 exactly
    as the density is, so the pair (density, normaliser) stays consistent and the
    component degenerates to a vanishing spike rather than 0/0.
    """
    s = jnp.maximum(sigma, _NORM_SIGMA_FLOOR)
    a = (lo - mu) / (s * _SQRT2)
    b = (hi - mu) / (s * _SQRT2)
    diff = jnp.where(
        a >= 0.0,
        erfc(a) - erfc(b),
        jnp.where(b <= 0.0, erfc(-b) - erfc(-a), erf(b) - erf(a)),
    )
    return s * _SQRT_PI_2 * jnp.maximum(diff, 0.0)


def _powerlaw_segment_norm(lo, hi, m_break, alpha):
    r"""Exact :math:`\int_{lo}^{hi} (m/m_{\rm break})^{-\alpha}\,dm` (0 if hi <= lo).

    Written as :math:`m_b e^{uL_{lo}} \,\mathrm{expm1}(u d)/u` with
    :math:`u = 1-\alpha`, :math:`L = \log(m/m_b)`, :math:`d = L_{hi} - L_{lo}`,
    so the :math:`\alpha \to 1` limit is the removable one
    (:math:`\to m_b \log(hi/lo)`); a short series replaces the ratio for
    :math:`|ud| < 10^{-6}` to keep it finite AND differentiable in ``alpha``
    there.
    """
    lo_s = jnp.maximum(lo, _NORM_SIGMA_FLOOR)
    hi_s = jnp.maximum(hi, lo_s)
    u = 1.0 - alpha
    d = jnp.log(hi_s / lo_s)
    ud = u * d
    small = jnp.abs(ud) < 1.0e-6
    ud_s = jnp.where(small, 1.0, ud)
    ratio = jnp.where(small,
                      d * (1.0 + 0.5 * ud + ud * ud / 6.0),
                      d * jnp.expm1(ud_s) / ud_s)
    seg = m_break * jnp.exp(u * jnp.log(lo_s / m_break)) * ratio
    return jnp.where(hi > lo, seg, 0.0)


def _tapered_edge_norm(eval_unnorm, plateau_norm, m_min, dm_min, m_max, dm_max):
    r"""Normaliser of a component tapered between SAMPLED edges, edge-exact.

    Splits the support ``[m_min, m_max]`` into the two taper windows and the
    plateau between them::

        N = int_{m_min}^{w_1} S p dm + int_{w_1}^{w_2} p dm + int_{w_2}^{m_max} S p dm

    with ``w_1 = m_min + dm_min``, ``w_2 = m_max - dm_max`` and ``S = 1``
    identically on the plateau.  ``plateau_norm(lo, hi)`` supplies the plateau in
    closed form; the two windows are integrated on WINDOW-RELATIVE nodes
    (``m = a + t (b - a)``, ``t`` uniform on [0, 1] -- the same trick
    :meth:`GWTC5FiducialBPL2PeaksMass._taper_window_grid` uses), so the nodes
    follow the sampled window instead of being crossed by it.  ``w_1`` is clipped
    into the support and ``w_2`` floored at ``w_1``, so overlapping windows
    (allowed by the curated ``m_max`` overrides) neither double-count nor leave a
    gap: the whole support is then covered by the two window integrals of the
    FULL tapered integrand.

    Why not the base class's fixed grid: with ``dm_min`` below the mass-grid
    spacing (h = 0.4 Msun at n_mass = 500, against a DM_MIN prior floor of 0.01)
    the quadrature sees a hard edge at a SAMPLED location, so the normaliser is a
    staircase in ``m_min`` -- measured on ``powerlaw+peak``: exactly flat over
    each 0.4-Msun cell with 10.2% drops at every node crossing, 0.112-nat
    discontinuities in ``log_p_pop``, and ``d log p/d m_min`` = 0 inside a cell
    (1.37 at a node) against a dense-quadrature reference of 0.27.
    """
    t = jnp.linspace(0.0, 1.0, get_mass_grid().size)
    hi = jnp.maximum(m_max, m_min)
    w1 = jnp.clip(m_min + dm_min, m_min, hi)
    w2 = jnp.clip(m_max - dm_max, w1, hi)

    def _window(a, b):
        m = a + t * (b - a)
        return jnp.trapezoid(eval_unnorm(m), m)

    return _window(m_min, w1) + plateau_norm(w1, w2) + _window(w2, hi)


def broken_powerlaw_norm(alpha1, alpha2, m_break, lo, hi):
    r"""Exact normaliser of the two-slope power law on ``[lo, hi]``.

    Matches ``(m/m_break)**-alpha1`` on ``[lo, min(m_break, hi))`` and
    ``(m/m_break)**-alpha2`` on ``[max(m_break, lo), hi)``, i.e. exactly the
    support the model's masks impose, for any ordering of ``lo``, ``m_break``
    and ``hi``.
    """
    brk = jnp.clip(m_break, lo, hi)
    return (_powerlaw_segment_norm(lo, brk, m_break, alpha1)
            + _powerlaw_segment_norm(brk, hi, m_break, alpha2))


@dataclass
class GWTC5FiducialBPL2PeaksMass(MassComponent):
    r"""GWTC-5 fiducial Broken Power Law + 2 Peaks primary-mass model.

    This is the DefaultBBH mass model from GWTC-5 Table 5 / Eqs. B10--B14:
    a normalized broken power law plus two truncated Gaussian peaks, with the
    full mixture multiplied by the low-mass Planck taper
    ``S(m_1 | m_{1,low}, delta_m_1)``.  The high-mass edge is fixed to
    ``m_high = 300 Msun`` and is therefore not sampled.

    Parameter order is ``alpha_1``, ``alpha_2``, ``m_break``, ``mu_1``,
    ``sigma_1``, ``mu_2``, ``sigma_2``, ``m_1_low``, ``delta_m_1``,
    ``lambda_0``, and ``lambda_1``.
    """

    alpha1_spec: ParamSpec
    alpha2_spec: ParamSpec
    m_break_spec: ParamSpec
    mu1_spec: ParamSpec
    sigma1_spec: ParamSpec
    mu2_spec: ParamSpec
    sigma2_spec: ParamSpec
    m1_low_spec: ParamSpec
    delta_m1_spec: ParamSpec
    lambda0_spec: ParamSpec
    lambda1_spec: ParamSpec
    m_high: float = 300.0

    @property
    def param_specs(self):
        return [
            self.alpha1_spec,
            self.alpha2_spec,
            self.m_break_spec,
            self.mu1_spec,
            self.sigma1_spec,
            self.mu2_spec,
            self.sigma2_spec,
            self.m1_low_spec,
            self.delta_m1_spec,
            self.lambda0_spec,
            self.lambda1_spec,
        ]

    @property
    def m1_support_max(self) -> float:
        """Support runs up to the FIXED high-mass edge ``m_high`` (300 Msun by
        default); the broken power law and both Gaussian peaks are cut to
        ``m < m_high``.  This exceeds the default ``M_HI`` ceiling, so the
        opt-in pairing grid must be sized to it."""
        return float(self.m_high)

    def _bpl_raw(self, m, alpha1, alpha2, m_break, m1_low):
        below = (m >= m1_low) & (m < m_break)
        above = (m >= m_break) & (m < self.m_high)
        p_below = (m / m_break) ** (-alpha1)
        p_above = (m / m_break) ** (-alpha2)
        return jnp.where(below, p_below, jnp.where(above, p_above, 0.0))

    def _trunc_gauss_raw(self, m, mu, sigma, m1_low):
        sigma_safe = jnp.maximum(sigma, 1.0e-12)
        support = (m >= m1_low) & (m < self.m_high)
        return jnp.where(support, jnp.exp(-0.5 * ((m - mu) / sigma_safe) ** 2), 0.0)

    def _mass_grid(self):
        base_grid = get_mass_grid()
        return jnp.linspace(base_grid[0], self.m_high, base_grid.size)

    def _taper_window_grid(self, m1_low, delta_m1):
        """Quadrature nodes across the low-mass taper window.

        The nodes are placed RELATIVE to the window, ``m = m1_low + t*delta_m1``
        with ``t`` uniform on [0, 1], so they follow the (sampled) window instead
        of being crossed by it: the taper deficit in :meth:`_norm` is then a
        smooth function of ``m1_low`` and ``delta_m1`` with no node-crossing
        steps, and a zero-width window integrates to exactly 0.
        """
        t = jnp.linspace(0.0, 1.0, get_mass_grid().size)
        width = jnp.maximum(jnp.minimum(m1_low + delta_m1, self.m_high) - m1_low, 0.0)
        return m1_low + t * width

    def _component_norms(self, t):
        """Exact normalisers of the three components over ``[m1_low, m_high]``."""
        alpha1, alpha2, m_break, mu1, sigma1, mu2, sigma2, m1_low = t[:8]
        return (
            broken_powerlaw_norm(alpha1, alpha2, m_break, m1_low, self.m_high),
            truncated_gaussian_norm(mu1, sigma1, m1_low, self.m_high),
            truncated_gaussian_norm(mu2, sigma2, m1_low, self.m_high),
        )

    def _mixture_pretaper(self, m, t):
        """``sum_i lambda_i p_i(m)`` with each ``p_i`` EXACTLY normalised over
        ``[m1_low, m_high]``; the caller applies the low-mass taper."""
        alpha1, alpha2, m_break, mu1, sigma1, mu2, sigma2, m1_low = t[:8]
        lambda0, lambda1 = t[9], t[10]
        n_bpl, n_g1, n_g2 = self._component_norms(t)
        bpl = self._bpl_raw(m, alpha1, alpha2, m_break, m1_low) / _pos(n_bpl)
        g1 = self._trunc_gauss_raw(m, mu1, sigma1, m1_low) / _pos(n_g1)
        g2 = self._trunc_gauss_raw(m, mu2, sigma2, m1_low) / _pos(n_g2)
        return lambda0 * bpl + lambda1 * g1 + (1.0 - lambda0 - lambda1) * g2

    def _norm(self, theta) -> jnp.ndarray:
        r"""Normalisation of the tapered mixture -- exact up to the taper window.

        Each component is now normalised analytically over ``[m1_low, m_high]``,
        so :math:`\int p_i\,dm = 1` exactly and the ONLY quadrature left is the
        probability the low-mass taper removes,

            D_i = \int_{m1_low}^{m1_low+\delta m_1} p_i(m)\,[1 - S(m)]\,dm,

        over a window-relative grid (see :meth:`_taper_window_grid`).  Each
        ``1 - D_i`` is clipped to [0, 1] so an unresolvably narrow peak sitting
        inside the taper window can never drive the normalisation negative.
        """
        alpha1, alpha2, m_break, mu1, sigma1, mu2, sigma2, m1_low, delta_m1, lambda0, lambda1 = theta
        lambda2 = 1.0 - lambda0 - lambda1
        valid_weights = (lambda0 >= 0.0) & (lambda1 >= 0.0) & (lambda2 >= 0.0)

        n_bpl, n_g1, n_g2 = self._component_norms(theta)
        m_w = self._taper_window_grid(m1_low, delta_m1)
        cut = 1.0 - sfilter_low(m_w, m1_low, delta_m1)

        def kept(raw, n):
            deficit = jnp.trapezoid(raw * cut, m_w) / _pos(n)
            return jnp.where(n > 0.0, jnp.clip(1.0 - deficit, 0.0, 1.0), 0.0)

        total = (
            lambda0 * kept(self._bpl_raw(m_w, alpha1, alpha2, m_break, m1_low), n_bpl)
            + lambda1 * kept(self._trunc_gauss_raw(m_w, mu1, sigma1, m1_low), n_g1)
            + lambda2 * kept(self._trunc_gauss_raw(m_w, mu2, sigma2, m1_low), n_g2)
        )
        return jnp.where(valid_weights, total, 0.0)

    def _eval_unnorm(self, m, t):
        m1_low, delta_m1, lambda0, lambda1 = t[7], t[8], t[9], t[10]
        lambda2 = 1.0 - lambda0 - lambda1
        valid_weights = (lambda0 >= 0.0) & (lambda1 >= 0.0) & (lambda2 >= 0.0)
        tapered = self._mixture_pretaper(m, t) * sfilter_low(m, m1_low, delta_m1)
        return jnp.where(valid_weights, tapered, 0.0)


@dataclass
class GWTC5FiducialBPL2PeaksPairing(PairingModel):
    r"""GWTC-5 fiducial low-mass-tapered mass-ratio power law.

    The conditional mass-ratio distribution follows ``q**beta_q`` and applies
    the separate secondary-mass taper ``S(m_2 | m_{2,low}, delta_m_2)`` from
    GWTC-5 Table 5.  This intentionally keeps ``delta_m_2`` independent from
    the primary-mass taper ``delta_m_1``.
    """

    beta_spec: ParamSpec
    m2_low_spec: ParamSpec
    delta_m2_spec: ParamSpec

    @property
    def param_specs(self):
        return [self.beta_spec, self.m2_low_spec, self.delta_m2_spec]

    def _eval_unnorm(self, m1, q, m_min, dm_min, theta):
        del m_min, dm_min
        beta_q, m2_low, delta_m2 = theta
        m2 = q * m1
        safe_q = jnp.where(q > 0.0, q, 1.0)
        p = jnp.where(q > 0.0, safe_q ** beta_q, 0.0)
        p = p * sfilter_low(m2, m2_low, delta_m2)
        return jnp.where(m2 < m2_low, 0.0, p)


class GWTC5FiducialBPL2PeaksPopulationModel:
    """GWTC-5 fiducial DefaultBBH Broken Power Law + 2 Peaks model."""

    def __init__(self):
        self.mass_component = GWTC5FiducialBPL2PeaksMass(
            ParamSpec(r"$\alpha_1$", -4.0, 12.0),
            ParamSpec(r"$\alpha_2$", -4.0, 12.0),
            ParamSpec(r"$m_{\rm break}$", 20.0, 50.0),
            ParamSpec(r"$\mu_1$", 5.0, 20.0),
            ParamSpec(r"$\sigma_1$", 0.0, 10.0),
            ParamSpec(r"$\mu_2$", 25.0, 60.0),
            ParamSpec(r"$\sigma_2$", 0.0, 10.0),
            ParamSpec(r"$m_{1,{\rm low}}$", 3.0, 10.0),
            ParamSpec(r"$\delta m_1$", 0.0, 10.0),
            ParamSpec(r"$\lambda_0$", 0.0, 1.0),
            ParamSpec(r"$\lambda_1$", 0.0, 1.0),
        )
        self.pairing_component = GWTC5FiducialBPL2PeaksPairing(
            ParamSpec(r"$\beta_q$", -2.0, 7.0),
            ParamSpec(r"$m_{2,{\rm low}}$", 3.0, 10.0),
            ParamSpec(r"$\delta m_2$", 0.0, 10.0),
        )
        self.spin_component = TruncatedGaussianSpin(
            ParamSpec(r"$\mu_\chi$", 0.0, 1.0),
            ParamSpec(r"$\sigma_\chi$", 0.005, 1.0),
        )
        self.gamma_spec = ParamSpec(r"$\gamma$", -10.0, 10.0)

    @property
    def param_specs(self):
        return [
            *self.mass_component.param_specs,
            *self.pairing_component.param_specs,
            *self.spin_component.param_specs,
            self.gamma_spec,
        ]

    @property
    def constraint_groups(self):
        """Joint prior constraints for the constraint-preserving cube maps.

        ``log_p_pop`` rejects lambda0 + lambda1 > 1 and m2_low > m1_low, but
        rejection alone leaves 75% of the nominal unit cube with zero
        likelihood: nested proposals waste 3 of 4 draws and the evidence is
        shifted by log(1/4) relative to the normalized constrained prior.
        Declaring the constraints here lets ``make_prior_transform`` map the
        cube ONTO the constrained region (measure-preserving fold/sort), so
        the sampled prior is the normalized uniform density on it; the
        ``valid`` mask in ``log_p_pop`` remains as a backstop for samplers
        that build their own prior (numpyro) and for overridden bounds.
        """
        return (
            ("simplex", (r"$\lambda_0$", r"$\lambda_1$")),
            ("ordered_le", (r"$m_{2,{\rm low}}$", r"$m_{1,{\rm low}}$")),
        )

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_p_pop(self, m1, q, z, chieff, theta):
        idx = 0
        tm = theta[idx : idx + self.mass_component.n_params]
        idx += self.mass_component.n_params
        tp = theta[idx : idx + self.pairing_component.n_params]
        idx += self.pairing_component.n_params
        ts = theta[idx : idx + self.spin_component.n_params]
        gamma = theta[idx + self.spin_component.n_params]

        m1_low = tm[7]
        lambda0, lambda1 = tm[9], tm[10]
        m2_low = tp[1]
        valid = (lambda0 + lambda1 <= 1.0) & (m2_low <= m1_low)

        mass_norm = self.mass_component._norm(tm)
        spin_norm = self.spin_component._norm(ts)
        p = (
            self.mass_component(m1, tm, norm=mass_norm)
            * self.pairing_component(m1, q, m2_low, tp[2], tp)
            * self.spin_component(chieff, ts, norm=spin_norm)
        )
        p = jnp.where(valid, p, 0.0)
        log_p = jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)
        return log_p + (gamma - 1.0) * jnp.log1p(z)


_LOG_SQRT_2PI = 0.9189385332046727
_EPS = 1.0e-30


@dataclass
class GolombRemnantMass1G(MassComponent):
    r"""Golomb--Isi--Farr 1G remnant mass distribution.

    This implements the physical model

        progenitor IMF  p(M_I)
        remnant map     M_I -> M_BH
        lognormal scatter in ln M_BH

    The unnormalised 1G BH mass density is

        phi_1G(m) = int dM_I p(M_I) p(m | M_I),

    where p(M_I) is a broken power law with fixed break 20 Msun,
    and p(m | M_I) is lognormal around the piecewise linear/quadratic
    remnant map.

    Parameter order
    ---------------
    a
        Low-progenitor-mass IMF slope below 20 Msun.
    b
        High-progenitor-mass IMF slope above 20 Msun.
    m_tr
        Linear-to-quadratic transition mass.
    delta_m
        Difference M_BH,max - M_tr. This enforces M_BH,max > M_tr.
    sigma
        Lognormal scatter in ln M_BH at fixed progenitor mass.
    """

    a_spec: ParamSpec
    b_spec: ParamSpec
    m_tr_spec: ParamSpec
    delta_m_spec: ParamSpec
    sigma_spec: ParamSpec

    @property
    def param_specs(self):
        return [
            self.a_spec,
            self.b_spec,
            self.m_tr_spec,
            self.delta_m_spec,
            self.sigma_spec,
        ]

    def _mean_remnant_mass(self, m_i, m_tr, delta_m):
        """Piecewise Golomb remnant map.

        We parameterise M_BH,max = M_tr + delta_m so that the physical
        constraint M_BH,max > M_tr is automatic when delta_m > 0.
        """
        m_bh_max = m_tr + delta_m
        m_disrupt = 2.0 * m_bh_max - m_tr

        # Original denominator is 4 * (M_tr - M_BH,max) = -4 delta_m.
        denom = -4.0 * jnp.maximum(delta_m, _EPS)

        quad = m_bh_max + (m_i - 2.0 * m_bh_max + m_tr) ** 2 / denom

        mbar = jnp.where(
            m_i < m_tr,
            m_i,
            jnp.where(m_i < m_disrupt, quad, 0.0),
        )

        valid = (delta_m > 0.0) & (m_tr > 0.0)
        return jnp.where(valid & (mbar > 0.0), mbar, 0.0)

    def _progenitor_imf(self, m_i, a, b):
        """Broken power-law progenitor distribution with fixed break 20 Msun."""
        x = jnp.maximum(m_i / 20.0, _EPS)
        return jnp.where(m_i < 20.0, x ** (-a), x ** (-b))

    def _eval_1g_unnorm(self, m, t):
        """Unnormalised 1G BH mass density phi_1G(m)."""
        a, b, m_tr, delta_m, sigma = t

        m_shape = jnp.shape(m)
        m_eval = jnp.reshape(jnp.asarray(m), (-1, 1))

        m_i_grid = get_mass_grid()
        m_i = jnp.reshape(m_i_grid, (1, -1))

        mbar = self._mean_remnant_mass(m_i, m_tr, delta_m)
        imf = self._progenitor_imf(m_i, a, b)

        sigma_safe = jnp.maximum(sigma, _EPS)
        log_m = jnp.log(jnp.maximum(m_eval, _EPS))
        log_mbar = jnp.log(jnp.maximum(mbar, _EPS))

        # Lognormal density p(m | M_I). Constants matter only up to final
        # normalisation, but keeping 1/m and 1/sigma gives correct shape.
        log_kernel = (
            -0.5 * ((log_m - log_mbar) / sigma_safe) ** 2
            - jnp.log(jnp.maximum(m_eval, _EPS))
            - jnp.log(sigma_safe)
            - _LOG_SQRT_2PI
        )

        kernel = jnp.exp(log_kernel)
        kernel = jnp.where((m_eval > 0.0) & (mbar > 0.0) & (sigma > 0.0), kernel, 0.0)

        integrand = imf * kernel
        phi = jnp.trapezoid(integrand, m_i_grid, axis=-1)

        return jnp.reshape(phi, m_shape)

    def _eval_unnorm(self, m, t):
        return self._eval_1g_unnorm(m, t)


@dataclass
class GolombRemnantMass1GPlusTail(GolombRemnantMass1G):
    r"""Golomb 1G remnant mass distribution plus high-mass power-law tail.

    This implements the non-evolving version of the full mass model:

        phi(m) = phi_1G(m)
               + taper(m | M_BH,max) f_PL phi_1G(M_BH,max)
                 (m / M_BH,max)^(-c)

    The tail approximates 2G/high-mass contamination. I parameterise the
    amplitude by log10_f_pl for sampling stability.
    """

    c_spec: ParamSpec
    log10_fpl_spec: ParamSpec
    dm_tail_spec: ParamSpec

    @property
    def param_specs(self):
        return [
            *super().param_specs,
            self.c_spec,
            self.log10_fpl_spec,
            self.dm_tail_spec,
        ]

    def _tail_taper(self, m, m_bh_max, dm_tail):
        """Smooth turn-on ending at M_BH,max.

        This is a logistic approximation to the paper's exponential taper.
        It is 0 well below M_BH,max - dm_tail and 1 above M_BH,max.
        """
        width = jnp.maximum(dm_tail, _EPS)
        x = (m - (m_bh_max - width)) / width
        return 1.0 / (1.0 + jnp.exp(-12.0 * (x - 0.5)))

    def _eval_unnorm(self, m, t):
        t_1g = t[:5]
        c, log10_fpl, dm_tail = t[5], t[6], t[7]

        a, b, m_tr, delta_m, sigma = t_1g
        m_bh_max = m_tr + delta_m

        phi_1g = self._eval_1g_unnorm(m, t_1g)

        # Relative height of the tail at M_BH,max.
        phi_peak = self._eval_1g_unnorm(m_bh_max, t_1g)
        fpl = 10.0 ** log10_fpl

        m_safe = jnp.maximum(m, _EPS)
        mbh_safe = jnp.maximum(m_bh_max, _EPS)

        taper = self._tail_taper(m_safe, m_bh_max, dm_tail)
        tail = taper * fpl * phi_peak * (m_safe / mbh_safe) ** (-c)

        valid = (delta_m > 0.0) & (sigma > 0.0) & (m_bh_max > 0.0)
        return jnp.where(valid, phi_1g + tail, 0.0)


@dataclass
class GolombSymmetricMassPopulationModel:
    r"""Exact Golomb-style symmetric component-mass population.

    This class follows the public PopulationModel interface:

        param_specs
        prior_bounds()
        log_p_pop(m1, q, z, chieff, theta)

    but does not use MixtureModel, because Golomb's mass model is coupled:

        dN / dm1 dm2 ∝ (m1 + m2)^beta phi(m1) phi(m2).

    Since your likelihood evaluates in (m1, q), this class includes the
    Jacobian m2 = q m1, dm2 = m1 dq:

        p(m1, q) ∝ m1 (m1 + q m1)^beta phi(m1) phi(q m1).

    The 2D normalisation over (m1, q) is computed on the same cached grids.
    """

    mass_component: MassComponent
    spin_component: SpinModel
    beta_spec: ParamSpec
    gamma_spec: ParamSpec

    @property
    def param_specs(self):
        return [
            *self.mass_component.param_specs,
            self.beta_spec,
            *self.spin_component.param_specs,
            self.gamma_spec,
        ]

    @property
    def n_params(self):
        return len(self.param_specs)

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def _mass_grid_pdf(self, theta_m):
        """Evaluate the normalised one-body mass pdf on the mass grid once."""
        m_grid = get_mass_grid()
        phi_unnorm = self.mass_component._eval_unnorm(m_grid, theta_m)
        norm = jnp.trapezoid(phi_unnorm, m_grid)
        phi = phi_unnorm / jnp.where(norm > 0.0, norm, 1.0)
        phi = jnp.where(norm > 0.0, phi, 0.0)
        return m_grid, phi

    def _interp_mass_pdf(self, m, m_grid, phi_grid):
        return jnp.interp(m, m_grid, phi_grid, left=0.0, right=0.0)

    def _mass_pair_norm(self, theta_m, beta):
        """Normalise p(m1, q) over the configured mass and q grids."""
        m_grid, phi_grid = self._mass_grid_pdf(theta_m)
        q_grid = get_q_grid()

        m1 = m_grid[:, None]
        q = q_grid[None, :]
        m2 = m1 * q

        phi1 = phi_grid[:, None]
        phi2 = self._interp_mass_pdf(jnp.ravel(m2), m_grid, phi_grid)
        phi2 = jnp.reshape(phi2, jnp.shape(m2))

        mtot = m1 + m2

        # Density in dm1 dq, hence the Jacobian factor m1.
        raw = m1 * phi1 * phi2 * mtot**beta
        raw = jnp.where((q > 0.0) & (q <= 1.0) & (m2 > 0.0), raw, 0.0)

        norm_q = jnp.trapezoid(raw, q_grid, axis=1)
        norm = jnp.trapezoid(norm_q, m_grid)
        return norm

    def _mass_pair_pdf(self, m1, q, theta_m, beta):
        m2 = q * m1
        m_grid, phi_grid = self._mass_grid_pdf(theta_m)

        phi1 = self._interp_mass_pdf(m1, m_grid, phi_grid)
        phi2 = self._interp_mass_pdf(m2, m_grid, phi_grid)

        mtot = m1 + m2
        raw = m1 * phi1 * phi2 * mtot**beta
        raw = jnp.where((q > 0.0) & (q <= 1.0) & (m2 > 0.0), raw, 0.0)

        norm = self._mass_pair_norm(theta_m, beta)
        return raw / jnp.where(norm > 0.0, norm, 1.0)

    def log_p_pop(self, m1, q, z, chieff, theta):
        n_m = self.mass_component.n_params
        n_s = self.spin_component.n_params

        theta_m = theta[:n_m]
        beta = theta[n_m]
        theta_s = theta[n_m + 1 : n_m + 1 + n_s]
        gamma = theta[-1]

        p_mq = self._mass_pair_pdf(m1, q, theta_m, beta)

        spin_norm = self.spin_component._norm(theta_s)
        p_spin = self.spin_component(chieff, theta_s, norm=spin_norm)

        p = p_mq * p_spin
        log_p = jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)

        # Keep your current redshift convention:
        # p(z) ∝ (1 + z)^(gamma - 1).
        return log_p + (gamma - 1.0) * jnp.log1p(z)


@component("pl_pairing", kind="pairing", params=(
    ParamBlueprint("beta", r"\beta", BETA, 1.0),
))
@dataclass
class PowerLawPairing(PairingModel):
    """Mass-ratio pairing model proportional to ``q**beta``.

    The model is evaluated conditional on a primary mass ``m1``.  It suppresses
    samples whose secondary mass ``m2 = q * m1`` lies below ``m_min`` using the
    same low-mass smoothing convention as the primary-mass models, then returns
    zero for samples that remain outside the allowed domain.
    """

    beta_spec: ParamSpec

    @property
    def param_specs(self):
        """Return the single ``beta`` parameter spec."""
        return [self.beta_spec]

    def _eval_unnorm(self, m1, q, m_min, dm_min, t):
        """Evaluate the unnormalised conditional density ``p(q | m1)``."""
        beta = t[0]
        m2 = q * m1
        # Safe pow: the normalisation q-grid contains q = 0 exactly, where
        # 0**beta = inf for beta < 0; the value is masked below but pow's VJP
        # then yields 0*inf = NaN in d/dbeta for EVERY beta < 0 (prior floor
        # is -2), silently walling NUTS out of beta <= 0. Same double-where
        # idiom as GWTC5FiducialBPL2PeaksPairing above.
        safe_q = jnp.where(q > 0.0, q, 1.0)
        p = jnp.where(q > 0.0, safe_q**beta, 0.0)
        p = sfilter_low(m2, m_min, dm_min) * p
        return jnp.where(m2 < m_min, 0.0, p)


@dataclass
class GaussianPairing(PairingModel):
    """Gaussian mass-ratio pairing model.

    Parameters ``mu_q`` and ``sigma_q`` define a Gaussian in mass ratio ``q``.
    The low-secondary-mass smoothing and hard domain behavior match
    :class:`PowerLawPairing`.
    """

    mu_q_spec: ParamSpec
    sigma_q_spec: ParamSpec

    @property
    def param_specs(self):
        """Return ``mu_q`` and ``sigma_q`` parameter specs."""
        return [self.mu_q_spec, self.sigma_q_spec]

    def _eval_unnorm(self, m1, q, m_min, dm_min, t):
        """Evaluate the unnormalised conditional density ``p(q | m1)``."""
        mu, sig = t[0], t[1]
        m2 = q * m1
        p = jnp.exp(-0.5 * ((q - mu) / sig) ** 2)
        p = sfilter_low(m2, m_min, dm_min) * p
        return jnp.where(m2 < m_min, 0.0, p)


@component("spin", kind="spin", params=(
    ParamBlueprint("mu_chi", r"\mu_\chi", CHI_MU, 0.0),
    ParamBlueprint("sigma_chi", r"\sigma_\chi", CHI_SIGMA, 0.1),
))
@dataclass
class TruncatedGaussianSpin(SpinModel):
    """Gaussian effective-spin distribution truncated to ``[-1, 1]``.

    The component uses parameters ``mu_chi`` and ``sigma_chi``.  It returns zero
    outside the physically allowed effective-spin range and is normalised over
    the shared spin grid by :class:`SpinModel`.
    """

    mu_chi_spec: ParamSpec
    sigma_chi_spec: ParamSpec

    @property
    def param_specs(self):
        """Return ``mu_chi`` and ``sigma_chi`` parameter specs."""
        return [self.mu_chi_spec, self.sigma_chi_spec]

    def _eval_unnorm(self, chieff, t):
        """Evaluate the unnormalised truncated Gaussian in effective spin."""
        mu, sig = t[0], t[1]
        p = jnp.exp(-0.5 * ((chieff - mu) / sig) ** 2)
        return jnp.where((chieff >= -1.0) & (chieff <= 1.0), p, 0.0)
