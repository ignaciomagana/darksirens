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

from dataclasses import dataclass

import jax.numpy as jnp

from .base import MassComponent, PairingModel, ParamSpec, SpinModel
from .utils import sfilter_high, sfilter_low


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

    def _eval_unnorm(self, m, t):
        """Evaluate the tapered unnormalised density at detector-frame mass ``m``."""
        a, mmin, mmax, dmmin, dmmax = t[0], t[1], t[2], t[3], t[4]
        S = sfilter_low(m, mmin, dmmin) * sfilter_high(m, mmax, dmmax)
        return S * m ** (-a)


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

    def _eval_unnorm(self, m, t):
        """Evaluate the Gaussian peak at primary mass ``m``."""
        mu, sig = t[0], t[1]
        return jnp.exp(-0.5 * ((m - mu) / sig) ** 2)


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
        p = q**beta
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
