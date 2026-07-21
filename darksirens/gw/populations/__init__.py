# populations/__init__.py
from .registry import (
    get_model,
    pop_model_parser,
    pop_model_prior_parser,
    get_fixed_population_params,
    population_m1_support_max,
)

__all__ = [
    "get_model",
    "pop_model_parser",
    "pop_model_prior_parser",
    "get_fixed_population_params",
    "population_m1_support_max",
]