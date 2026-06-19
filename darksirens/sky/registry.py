"""
registry.py
-----------
Sky-model registry — the public lookup API, mirroring the population registry
(:mod:`darksirens.gw.populations.registry`).

A sky model name resolves to an object exposing ``param_specs``,
``prior_bounds()`` and ``log_g_sky(nx, ny, nz, theta)``.  ``isotropic`` is the
default and the null hypothesis; ``dipole`` and ``sphere_gp`` are the
anisotropic alternatives, distinguished from isotropy by Bayesian evidence
(the samplers already report ``logZ``).
"""

from __future__ import annotations

import jax.numpy as jnp

from .models import (
    IsotropicSky,
    DipoleSky,
    SphereGPSky,
    SphereZGPSky,
    OverdensityGP3D,
    MultipoleSky,
)

#: Canonical sky-model names (``isotropic`` is the default / null model).
SKY_MODEL_NAMES = (
    "isotropic", "dipole", "sphere_gp", "sphere_gp_z", "overdensity_gp",
    "multipole", "multipole_l3",
)

_SKY_FACTORIES = {
    "isotropic": IsotropicSky,
    "dipole": DipoleSky,
    "sphere_gp": SphereGPSky,
    "sphere_gp_z": SphereZGPSky,
    "overdensity_gp": OverdensityGP3D,
    "multipole": lambda: MultipoleSky(lmax=2),
    "multipole_l3": lambda: MultipoleSky(lmax=3),
}

SKY_MODEL_LATEX = {
    "isotropic": r"\text{Isotropic}",
    "dipole": r"\text{Dipole}",
    "sphere_gp": r"\text{Sphere GP}",
    "sphere_gp_z": r"\text{Sphere GP }(\hat n, z)",
    "overdensity_gp": r"\text{3D Overdensity GP}",
    "multipole": r"\text{Multipole }(\ell\le2)",
    "multipole_l3": r"\text{Multipole }(\ell\le3)",
}

_SKY_REGISTRY: dict = {}


def get_sky_model(sky_model: str):
    """Return (and cache) the sky-model object for ``sky_model``."""
    if sky_model not in _SKY_FACTORIES:
        available = ", ".join(f'"{k}"' for k in SKY_MODEL_NAMES)
        raise ValueError(
            f"Unknown sky model '{sky_model}'. Available: {available}."
        )
    if sky_model not in _SKY_REGISTRY:
        _SKY_REGISTRY[sky_model] = _SKY_FACTORIES[sky_model]()
    return _SKY_REGISTRY[sky_model]


def sky_model_parser(sky_model: str):
    """Return the ``log_g_sky(nx, ny, nz, theta)`` callable for ``sky_model``."""
    return get_sky_model(sky_model).log_g_sky


def sky_model_prior_parser(sky_model: str):
    """Return ``(lows, highs, labels, kinds, latex)`` for the sky parameter block.

    ``kinds`` are ``(prior_kind, loc, scale)`` triples aligned to ``labels`` —
    the same channel populations use to declare whitened-GP ``xi`` latents as
    standard normal to the sampler.
    """
    model = get_sky_model(sky_model)
    lows, highs, labels = model.prior_bounds()
    kinds = [(s.prior_kind, s.prior_loc, s.prior_scale) for s in model.param_specs]
    return lows, highs, labels, kinds, SKY_MODEL_LATEX[sky_model]


def sky_fiducial(sky_model: str) -> tuple[float, ...]:
    """Fiducial sky parameters corresponding to **isotropy** (``g ≡ 1``).

    For every model the fiducial point reduces to the isotropic limit: the
    dipole vector is zero, and the GP latents ``xi`` are zero (so ``f ≡ 0``)
    with hyperparameters at the centre of their prior range.
    """
    model = get_sky_model(sky_model)
    values = []
    for spec in model.param_specs:
        if (
            spec.name.startswith("sky_xi")
            or spec.name.startswith("sky_a")          # multipole a_lm
            or spec.name in ("sky_dx", "sky_dy", "sky_dz")
        ):
            values.append(0.0)
        else:  # log_amp / log_ls hyperparameters: centre of the prior range
            values.append(0.5 * (spec.low + spec.high))
    return tuple(values)


def get_fixed_sky_params(sky_model: str) -> jnp.ndarray:
    """Return the fiducial (isotropic) sky parameter vector as a JAX array."""
    return jnp.array(sky_fiducial(sky_model), dtype=float)
