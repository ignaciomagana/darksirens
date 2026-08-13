"""4-D component-spin population model (DS-08).

The GWTC-3 "Default" spin model: independent, identically distributed
component-spin magnitudes ``a_i ~ Beta(alpha_chi, beta_chi)`` on [0, 1] and a
tilt mixture ``cos t_i ~ zeta_spin * N_trunc(cos t; 1, sigma_t) +
(1 - zeta_spin) * 1/2`` on [-1, 1] (isotropic plus preferentially aligned).

Everything is normalised ANALYTICALLY -- the Beta by ``betaln``, the
truncated Gaussian by ``erf`` -- so no new quadrature enters the likelihood:
the existing 1-D ``SpinModel`` machinery integrates its chi_eff density over
``CHI_GRID`` per proposal, and a 4-D grid twin of that would be the exact
cost this design avoids.

The model consumes the event's ``spin`` block (``GWEvent.spin``, columns
``(a1, a2, cost1, cost2)``) and IGNORES ``chieff``: chi_eff is a deterministic
function of the component spins and q, so a density over the components
already fixes its distribution, and multiplying by an extra chi_eff factor
would double-count.  It therefore only makes sense against a
component-basis store, whose ``p_pe``/``pdraw`` carry the flat component
draw density -- the basis-negotiation gate (DS-09) is what enforces that
pairing.  N.B. the selection-side variance measurement (gwcat GW-24) found
the component basis consumed by a 1-D chi_eff population WORSE than the
chieff basis end-to-end: this model existing is the precondition for the
component basis being usable at all, not an optimisation.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.special import betaln, erf

from .base import ParamSpec, SpinModel

#: Column order of the spin block this model consumes (= GWEvent.spin).
COMPONENT_SPIN_COLUMNS = ("a1", "a2", "cost1", "cost2")

_SQRT_HALF_PI = 1.2533141373155003  # sqrt(pi/2)
_SQRT2 = 1.4142135623730951


@dataclass
class ComponentSpinModel(SpinModel):
    """iid Beta magnitudes x (isotropic + aligned-Gaussian) tilt mixture.

    Parameters (in ``theta`` order): ``alpha_chi``, ``beta_chi`` (Beta shape
    parameters of each magnitude; bounded below by 1 so the density is
    non-singular at the endpoints, matching the LVK convention),
    ``zeta_spin`` (aligned fraction), ``sigma_t`` (width of the aligned
    tilt Gaussian, truncated to [-1, 1] and peaked at cos t = 1).

    This is a :class:`SpinModel` so it slots into every population model's
    ``spin_component`` field and its ``ParamSpec``s reach the flat
    labels/lower/upper vector through the ordinary ``param_specs`` chain --
    but it consumes the (N, 4) spin BLOCK, not chi_eff, so the base class's
    grid normalisation is bypassed (``_norm`` is analytically 1).
    """

    alpha_chi_spec: ParamSpec
    beta_chi_spec: ParamSpec
    zeta_spin_spec: ParamSpec
    sigma_t_spec: ParamSpec

    #: Population models dispatch on this: True means __call__ requires the
    #: event spin block and the paired store must be component-basis.
    consumes_spin_block = True
    spin_columns = COMPONENT_SPIN_COLUMNS

    @property
    def param_specs(self):
        return [
            self.alpha_chi_spec,
            self.beta_chi_spec,
            self.zeta_spin_spec,
            self.sigma_t_spec,
        ]

    # -- analytic pieces ---------------------------------------------------
    @staticmethod
    def _log_p_magnitude(a, alpha, beta):
        """log Beta(a; alpha, beta) on [0, 1]; -inf outside the open support."""
        in_support = (a > 0.0) & (a < 1.0)
        a_safe = jnp.clip(a, 1e-300, 1.0 - 1e-16)
        lp = (
            (alpha - 1.0) * jnp.log(a_safe)
            + (beta - 1.0) * jnp.log1p(-a_safe)
            - betaln(alpha, beta)
        )
        return jnp.where(in_support, lp, -jnp.inf)

    @staticmethod
    def _log_p_tilt(c, zeta, sigma_t):
        """log of the tilt mixture on [-1, 1]; -inf outside.

        The aligned component is a Gaussian peaked at cos t = 1 truncated to
        [-1, 1], normalised in closed form:
        Z = sigma * sqrt(pi/2) * erf(sqrt(2)/sigma).
        """
        in_support = (c >= -1.0) & (c <= 1.0)
        z_aligned = sigma_t * _SQRT_HALF_PI * erf(_SQRT2 / sigma_t)
        p_aligned = jnp.exp(-0.5 * ((c - 1.0) / sigma_t) ** 2) / z_aligned
        p = zeta * p_aligned + (1.0 - zeta) * 0.5
        return jnp.where(
            in_support, jnp.log(jnp.maximum(p, 1e-300)), -jnp.inf
        )

    def log_prob(self, spin, theta):
        """Normalised log density of the (..., 4) spin block ``(a1, a2, cost1, cost2)``."""
        alpha, beta, zeta, sigma_t = theta[0], theta[1], theta[2], theta[3]
        return (
            self._log_p_magnitude(spin[..., 0], alpha, beta)
            + self._log_p_magnitude(spin[..., 1], alpha, beta)
            + self._log_p_tilt(spin[..., 2], zeta, sigma_t)
            + self._log_p_tilt(spin[..., 3], zeta, sigma_t)
        )

    # -- SpinModel interface -----------------------------------------------
    def _eval_unnorm(self, chieff, theta):
        raise TypeError(
            "ComponentSpinModel has no 1-D chi_eff density: it is a 4-D "
            "density over (a1, a2, cost1, cost2), and its chi_eff marginal "
            "is not analytic. Evaluate log_prob(spin, theta), or use a "
            "chi_eff SpinModel with a chieff-basis store."
        )

    def _norm(self, theta):
        # Analytically normalised; the grid integral the base class would run
        # is both unnecessary and undefined (no 1-D density to integrate).
        return jnp.asarray(1.0)

    def __call__(self, chieff, theta, norm=None, spin=None):
        if spin is None:
            raise TypeError(
                "ComponentSpinModel requires the event spin block "
                "(a1, a2, cost1, cost2): the paired gwcat store must be "
                "exported in the component spin basis. chieff-basis stores "
                "carry no component columns; use a chi_eff spin model with "
                "them instead."
            )
        return jnp.exp(self.log_prob(spin, theta))


def default_component_spin() -> ComponentSpinModel:
    """The default component-spin block (GWTC-3 Default-model parameterisation).

    Bounds: Beta shapes in [1, 10] (non-singular at the endpoints), aligned
    fraction in [0, 1], aligned-tilt width in [0.1, 4] -- the LVK Default
    spin prior ranges.
    """
    return ComponentSpinModel(
        ParamSpec(r"$\alpha_\chi$", 1.0, 10.0, name="alpha_chi"),
        ParamSpec(r"$\beta_\chi$", 1.0, 10.0, name="beta_chi"),
        ParamSpec(r"$\zeta_{\rm spin}$", 0.0, 1.0, name="zeta_spin"),
        ParamSpec(r"$\sigma_t$", 0.1, 4.0, name="sigma_t"),
    )
