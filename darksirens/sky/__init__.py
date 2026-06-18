# sky/__init__.py
"""Angular (sky-distribution) models for GW source isotropy/anisotropy inference.

The sky model supplies a *mean-one angular density* ``g(n̂)`` that multiplies the
source rate: ``R(θ, z, n̂) = R_pop(θ, z) · g(n̂)``.  Isotropy corresponds to
``g ≡ 1`` (``log_g_sky ≡ 0``), so the overall rate normalisation is never
degenerate with the angular shape.  See :mod:`darksirens.sky.models` for the
member models and :mod:`darksirens.sky.registry` for the public lookup API.
"""

from .registry import (
    SKY_MODEL_NAMES,
    get_sky_model,
    sky_model_parser,
    sky_model_prior_parser,
    get_fixed_sky_params,
)

__all__ = [
    "SKY_MODEL_NAMES",
    "get_sky_model",
    "sky_model_parser",
    "sky_model_prior_parser",
    "get_fixed_sky_params",
]
