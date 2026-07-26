"""
darksirens.lensing
==================
Lensing forward model for joint BBH population + lensing hierarchical inference.

Submodules
----------
wlmagnification
    Weak-lensing magnification PDF ``p_WL(μ | z)``. Two backends:
    a flux-conserving lognormal with redshift-dependent variance, and a
    user-supplied tabulated grid for higher-fidelity ray-tracing fits.

slmarks
    Strong-lensing optical depth ``τ_J(z)`` and the joint mark distribution
    of image magnifications and parities for the SIS lens model. MVP only
    covers ``J=2`` (doubles); ``J=4`` (quads) hooks are present but inert.

clusters
    Container and HDF5 loader for pre-identified candidate multi-image
    clusters supplied by an external pairwise-Bayes-factor analysis.

grids
    Gauss-Legendre quadrature helpers in ``ln μ`` and ``y`` for the
    weak- and strong-lensing marginalizations.

Status
------
Commit 1: standalone subpackage with unit tests. No integration with
the inference hot path yet — that arrives in commit 2.

Design contract
---------------
Every function here is either a pure JAX function (``@jit``-able, vmappable)
or a NumPy/Python loader that runs once at startup. No closures over
mutable state, no module-level caches that depend on ``cosmo`` or ``survey``.
"""

from darksirens.lensing.wlmagnification import (
    WLParams,
    make_lognormal_wl_params,
    make_tabulated_wl_params,
    log_p_wl,
    # Closure factories (commit 2, for use inside JIT)
    make_lognormal_log_p_wl,
    make_tabulated_log_p_wl,
    make_log_p_wl_from_params,
    wl_mu_quadrature_coverage,
    validate_wl_mu_quadrature,
)
from darksirens.lensing.slmarks import (
    SISLensParams,
    make_sis_lens_params,
    tau_2_SIS,
    log_p_y_SIS,
    mu_plus_minus_from_y,
    y_from_mu_plus,
    delta_t_from_y,
)
from darksirens.lensing.clusters import ClusterSet, load_clusters, save_clusters
from darksirens.lensing.grids import (
    make_log_mu_grid,
    make_wl_mu_quadrature,
    make_y_grid,
)

__all__ = [
    # WL
    "WLParams",
    "make_lognormal_wl_params",
    "make_tabulated_wl_params",
    "log_p_wl",
    "make_lognormal_log_p_wl",
    "make_tabulated_log_p_wl",
    "make_log_p_wl_from_params",
    "wl_mu_quadrature_coverage",
    "validate_wl_mu_quadrature",
    # SL marks
    "SISLensParams",
    "make_sis_lens_params",
    "tau_2_SIS",
    "log_p_y_SIS",
    "mu_plus_minus_from_y",
    "y_from_mu_plus",
    "delta_t_from_y",
    # Clusters
    "ClusterSet",
    "load_clusters",
    "save_clusters",
    # Grids
    "make_log_mu_grid",
    "make_wl_mu_quadrature",
    "make_y_grid",
]
