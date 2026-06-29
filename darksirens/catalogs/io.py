"""Catalog I/O compatibility exports for the new package layout.

Preferred internal imports for survey-loading helpers should use this module.
The implementations still live in ``darksirens.em.utils`` until the final
compatibility-wrapper cleanup.
"""

from darksirens.em.utils import (  # noqa: F401
    load_survey,
    load_survey_marks,
    zMax,
    zgrid,
)

__all__ = ["load_survey", "load_survey_marks", "zMax", "zgrid"]
