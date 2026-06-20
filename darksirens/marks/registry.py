"""
registry.py
-----------
Mark-model registry — the public lookup API, mirroring the sky-model registry
(:mod:`darksirens.sky.registry`).  A mark-model name + the list of marks present
resolves to an object exposing ``param_specs``, ``prior_bounds()`` and
``log_h(em_catalog, eta)``.  ``none`` is the default null model (``h ≡ 1``,
no sampled parameters, the legacy galaxy-count host model).
"""
from __future__ import annotations

from .models import NoMarks, LogLinearMarks

#: Canonical mark-model names (``none`` is the default / legacy model).
MARK_MODEL_NAMES = ("none", "loglinear")

MARK_MODEL_LATEX = {
    "none": r"\text{No marks}",
    "loglinear": r"\text{Log-linear marks}",
}


def get_mark_model(mark_model: str, mark_names=()):
    """Return the mark-model object for ``mark_model`` over ``mark_names``."""
    if mark_model == "none":
        return NoMarks()
    if mark_model == "loglinear":
        return LogLinearMarks(mark_names)
    available = ", ".join(f'"{k}"' for k in MARK_MODEL_NAMES)
    raise ValueError(f"Unknown mark model '{mark_model}'. Available: {available}.")


def mark_model_parser(mark_model: str, mark_names=()):
    """Return the ``log_h(em_catalog, eta)`` callable for ``mark_model``."""
    return get_mark_model(mark_model, mark_names).log_h


def mark_model_prior_parser(mark_model: str, mark_names=()):
    """Return ``(lows, highs, labels, kinds, latex)`` for the ``eta`` block.

    ``kinds`` are ``(prior_kind, loc, scale)`` triples aligned to ``labels`` —
    the same channel the sampler uses; the mark coefficients are plain uniform.
    """
    model = get_mark_model(mark_model, mark_names)
    lows, highs, labels = model.prior_bounds()
    kinds = [(s.prior_kind, s.prior_loc, s.prior_scale) for s in model.param_specs]
    return lows, highs, labels, kinds, MARK_MODEL_LATEX[mark_model]


def mark_fiducial(mark_model: str, mark_names=()) -> tuple:
    """Fiducial mark parameters = ``eta = 0`` (the ``h ≡ 1`` no-preference limit)."""
    model = get_mark_model(mark_model, mark_names)
    return tuple(0.0 for _ in model.param_specs)
