"""
Redshift prior and completion models for dark-siren inference.

This package contains redshift-prior, catalog-completion, lognormal LSS
completion, and volume-prior math for dark-siren inference.
"""

from darksirens.catalogs.io import zgrid, zMax, load_survey
from darksirens.redshift.catalog import log_catalog_prior, log_catalog_prior_vmap
from darksirens.redshift.prior import get_redshift_prior, PRIOR_REGISTRY
from darksirens.redshift.volume import log_volume_prior
from darksirens.redshift.completion import (
    catalog_completion,
    catalog_completion_vmap,
    completion_clip_diagnostics,
    compute_lss_overdensity,
)

__all__ = [
    "get_redshift_prior",
    "PRIOR_REGISTRY",
    "log_volume_prior",
    "log_catalog_prior",
    "log_catalog_prior_vmap",
    "catalog_completion",
    "catalog_completion_vmap",
    "completion_clip_diagnostics",
    "compute_lss_overdensity",
    "zgrid",
    "zMax",
]
