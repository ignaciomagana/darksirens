"""Shared plumbing for the completeness-visualization closure experiment.

Everything here reuses darksirens / mock-generator machinery rather than
re-deriving it: the fiducial cosmology comes from
``darksirens.utils.cosmology`` (the same Planck values the Q-table builder
fixes), the survey-selection and pixelation physics come from
``scripts/mock_dark_sirens/generate_mock_data.py`` loaded as a module, and the
EMCatalog construction mirrors ``darksirens_diagnose_lognormal_completion``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.redshift import zgrid  # noqa: E402
from darksirens.catalogs.io import load_survey  # noqa: E402
from darksirens.core.types import CosmoParams, EMCatalog, SurveyParams  # noqa: E402
from darksirens.utils.cosmology import (  # noqa: E402
    H0Planck,
    Om0Planck,
    w0Fiducial,
    waFiducial,
)

#: EMCatalog.lss_completion_indexing enum values (see darksirens/core/types.py).
INDEXING = {"compact": 1, "global": 2}


def load_dark_mock_module():
    """Load ``scripts/mock_dark_sirens/generate_mock_data.py`` as a module.

    Same importlib pattern as ``scripts/mock_bright_sirens``: the generator is a
    standalone script, not a package module.
    """
    path = ROOT / "scripts" / "mock_dark_sirens" / "generate_mock_data.py"
    spec = importlib.util.spec_from_file_location("dark_mock_data_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import dark-siren mock helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fiducial_cosmo() -> CosmoParams:
    """The exact fiducial the Q-table builder conditions on."""
    return CosmoParams(H0=H0Planck, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial)


def fiducial_survey(n0: float, *, b_miss: float = 1.0, z_depth=None) -> SurveyParams:
    """SurveyParams matching the builder's ``_fiducial_cosmo_survey`` plus depth.

    ``z50``/``w`` are inert for the dark-siren completeness (the kernel-ratio C
    is data-driven); ``n0`` must be the INJECTED comoving density — a mismatch
    is absorbed into Q as spurious redshift structure.
    """
    return SurveyParams(
        n0=float(n0), z50=1.0, w=0.5, delta=0.0,
        b_miss=float(b_miss), alpha_miss=1.0, sigma_kde=0.0,
        z_depth=None if z_depth is None else float(z_depth),
    )


def make_em_catalog(catalog_path, **lss):
    """Full-sky EMCatalog from a pixelated survey file.

    Mirrors ``diagnose_lognormal_completion._full_catalog``; also returns the
    file's nside and ``z_depth`` attr so callers can thread the depth into
    SurveyParams.
    """
    import healpy as hp

    nside, ngals, zgals, dzgals, wgals, z_depth = load_survey(catalog_path)
    cat = EMCatalog(
        apix=float(hp.nside2pixarea(int(nside))),
        zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, int(zgrid.size))),
        dN_obs_kde=None, pixel_to_cache_idx=None, unique_pixels=None, **lss,
    )
    return int(nside), cat, z_depth


def eval_truth_field_logq(xi, Zn, Zz, n_hat, z_values, *, amp, ls_sph, ls_z,
                          bias=1.0, pix_chunk=64):
    """Truth log Q on (pixels × z_values) for whitened latents ``xi``.

    Uses the SAME whitened low-rank operator as the gp3d builder
    (``build_lowrank_operator``) and its mean-one convention
    ``logQ = b·Φξ − ½ b² Σ_j Φ²`` (Nystrom prior variance), so the truth field
    lives in exactly the family the fitter assumes.
    """
    from darksirens.redshift.lognormal_completion import build_lowrank_operator

    n_hat = np.asarray(n_hat, dtype=float)
    zeta = np.log1p(np.asarray(z_values, dtype=float))
    n_pix, n_z = n_hat.shape[0], zeta.size
    out = np.empty((n_pix, n_z), dtype=float)
    for start in range(0, n_pix, pix_chunk):
        stop = min(start + pix_chunk, n_pix)
        X_n = np.repeat(n_hat[start:stop], n_z, axis=0)
        X_z = np.tile(zeta, stop - start)
        Phi, _ = build_lowrank_operator(Zn, Zz, X_n, X_z,
                                        amp=amp, ls_sph=ls_sph, ls_z=ls_z)
        Phi = np.asarray(Phi, dtype=float)
        f = Phi @ np.asarray(xi, dtype=float)
        var = np.sum(Phi**2, axis=1)
        out[start:stop] = (bias * f - 0.5 * bias**2 * var).reshape(stop - start, n_z)
    return out


def comoving_xyz(ra, dec, z, z_grid, chi_grid):
    """Comoving Cartesian coordinates [Mpc] from (ra, dec [rad], z)."""
    chi = np.interp(np.asarray(z, dtype=float), z_grid, chi_grid)
    cd = np.cos(dec)
    return (chi * cd * np.cos(ra), chi * cd * np.sin(ra), chi * np.sin(dec))


def xyz_to_pix_z(x, y, z_cart, nside, z_grid, chi_grid):
    """(HEALPix RING pixel, redshift) for comoving Cartesian points [Mpc].

    Returns ``(pix, z, chi)``; callers mask points with ``chi`` outside the
    tabulated range themselves.
    """
    import healpy as hp

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z_cart = np.asarray(z_cart, dtype=float)
    chi = np.sqrt(x**2 + y**2 + z_cart**2)
    safe = np.maximum(chi, 1e-12)
    pix = hp.vec2pix(int(nside), x / safe, y / safe, z_cart / safe)
    z = np.interp(chi, chi_grid, z_grid)
    return pix, z, chi
