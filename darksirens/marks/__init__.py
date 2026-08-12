"""
darksirens.marks
----------------
Marked-host model: per-galaxy "marks" (logM*, sSFR, metallicity, colour, ...)
and a sampled BBH-host efficiency ``h(m | eta)`` that reweights the catalog's
contribution to the dark-siren redshift prior.  Mirrors :mod:`darksirens.sky`.
"""
from .models import (
    MARK_FIELDS,
    MARK_LATEX,
    MARK_SATURATION_MAX_FRACTION,
    NoMarks,
    LogLinearMarks,
    available_marks,
    check_marks_centred,
    check_flat_marks_centred,
)
from .registry import (
    MARK_MODEL_NAMES,
    MARK_MODEL_LATEX,
    get_mark_model,
    mark_model_parser,
    mark_model_flat_parser,
    mark_model_prior_parser,
    mark_fiducial,
)

__all__ = [
    "MARK_FIELDS",
    "MARK_LATEX",
    "MARK_SATURATION_MAX_FRACTION",
    "NoMarks",
    "LogLinearMarks",
    "available_marks",
    "check_marks_centred",
    "check_flat_marks_centred",
    "MARK_MODEL_NAMES",
    "MARK_MODEL_LATEX",
    "get_mark_model",
    "mark_model_parser",
    "mark_model_flat_parser",
    "mark_model_prior_parser",
    "mark_fiducial",
]
