"""
darksirens_build_lognormal_completion
-------------------------------------
**Offline** preprocessor that builds an LSS-conditioned lognormal completion
file ``Q_LSS(p, z)`` from a pixelated survey catalog, for later consumption by
the dark-siren redshift prior (``darksirens_inference --lss_completion``).

It must NOT be run inside the GW likelihood — it is a one-off build step.

Pipeline
--------
1. Load the survey catalog (``darksirens.em.utils.load_survey``).
2. Bin observed galaxies onto the package ``zgrid`` per pixel  -> ``N_obs``.
3. Compute the matched-kernel completeness ``C`` and homogeneous expected
   counts ``dN_exp`` with the existing completion machinery
   (:func:`darksirens.em.completion._precompute_grids`,
   :func:`darksirens.em.completion._kde_dndz_obs`) under a *fiducial*
   cosmology/survey (the same fiducials the inference dry-run uses).
4. Build a per-pixel 1-D Gaussian-correlation power spectrum from the **fixed**
   SurveyParams/CosmoParams hyperparameters (correlation length in Mpc mapped to
   grid units via the comoving-distance grid; field amplitude ``lss_sigma``;
   bias ``b_miss``) — never CLI knobs, never marginalised.
5. Run :func:`poisson_lognormal_map` (and optionally
   :func:`laplace_lognormal_members`) and save an HDF5 completion file.
"""
from __future__ import annotations

import argparse

import numpy as np
import jax.numpy as jnp

from darksirens.em import zgrid
from darksirens.em.utils import load_survey
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import r_of_z, H0Planck, Om0Planck, w0Fiducial, waFiducial
from darksirens.em.completion import _precompute_grids, _kde_dndz_obs, _TRAPW_NP
from darksirens.em.lognormal_completion import (
    gaussian_correlation_spectrum,
    poisson_lognormal_map,
    laplace_lognormal_members,
    save_lss_completion_hdf5,
)


def _fiducial_cosmo_survey():
    """Fiducial cosmology + survey (matching the inference dry-run defaults)."""
    cosmo = CosmoParams(H0=H0Planck, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial)
    survey = SurveyParams(
        n0=10.0 ** -2.0, z50=1.0, w=0.5, delta=0.0,
        b_miss=1.0, alpha_miss=1.0, sigma_kde=0.0,
    )  # lss_corr_length_mpc / lss_sigma take their container defaults
    return cosmo, survey


def _zgrid_bin_edges() -> np.ndarray:
    """Bin edges around the (non-uniform) ``zgrid`` points (midpoints)."""
    z = np.asarray(zgrid, dtype=float)
    mids = 0.5 * (z[:-1] + z[1:])
    return np.concatenate([[z[0]], mids, [z[-1]]])


def build_completion(
    catalog_path: str,
    *,
    n_members: int = 32,
    seed: int = 1234,
    prior_strength: float = 1.0,
    maxiter: int = 300,
):
    """Build the MAP (and optional ensemble) log Q tables from a survey catalog."""
    import healpy as hp

    nside, ngals, zgals, dzgals, wgals = load_survey(catalog_path)
    n_pix = int(np.asarray(zgals).shape[0])
    n_grid = int(zgrid.size)
    apix = float(hp.nside2pixarea(int(nside)))

    cosmo, survey = _fiducial_cosmo_survey()
    em = EMCatalog(
        apix=apix, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, n_grid)), dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(cosmo, survey, em)
    dN_exp_density = np.asarray(grids.dN_exp, dtype=float)
    dN_exp_smooth = np.asarray(grids.dN_exp_smooth, dtype=float)
    dN_exp_count = dN_exp_density * _TRAPW_NP  # expected detected counts per z-bin

    # Observed completeness (matched-kernel ratio) and binned counts per pixel.
    edges = _zgrid_bin_edges()
    ngals_np = np.asarray(ngals).astype(int)
    zgals_np = np.asarray(zgals, dtype=float)
    C = np.empty((n_pix, n_grid), dtype=float)
    N_obs = np.zeros((n_pix, n_grid), dtype=float)
    safe_smooth = np.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    for r in range(n_pix):
        dN_obs_s = np.asarray(_kde_dndz_obs(r, em.zgals, ngals=em.ngals), dtype=float)
        C[r] = np.clip(dN_obs_s / safe_smooth, 0.0, 1.0)
        zs = zgals_np[r, : ngals_np[r]]
        if zs.size:
            N_obs[r], _ = np.histogram(zs, bins=edges)

    # Built-in Gaussian-correlation P(k): correlation length (Mpc) -> grid units.
    chi = np.asarray(r_of_z(jnp.asarray(zgrid), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa), dtype=float)
    dchi = np.diff(chi)
    med_dchi = float(np.median(dchi[dchi > 0.0])) if np.any(dchi > 0.0) else 1.0
    ell_grid = float(survey.lss_corr_length_mpc) / max(med_dchi, 1e-6)
    pk = gaussian_correlation_spectrum(n_grid, ell_grid, float(survey.lss_sigma))

    bias = float(survey.b_miss)
    mp = poisson_lognormal_map(
        N_obs, C, dN_exp_count, pk,
        bias=bias, prior_strength=prior_strength, maxiter=maxiter,
    )
    diagnostics = dict(mp["diagnostics"])
    diagnostics.update({
        "nside": int(nside), "n_pix": n_pix, "ell_grid": ell_grid,
        "lss_corr_length_mpc": float(survey.lss_corr_length_mpc),
        "lss_sigma": float(survey.lss_sigma), "median_dchi_mpc": med_dchi,
    })

    logq_members = None
    if n_members and n_members > 0:
        members = laplace_lognormal_members(
            mp["s_map"], mp["lambda_map"], pk,
            n_members=int(n_members), bias=bias, prior_strength=prior_strength, seed=int(seed),
        )
        logq_members = members["logq_members"]
        diagnostics.update(members["diagnostics"])

    return mp["logq_map"], logq_members, diagnostics


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Build an LSS-conditioned lognormal completion file (offline)."
    )
    p.add_argument("--catalog", required=True, help="Pixelated survey catalog HDF5 (load_survey schema).")
    p.add_argument("--out", required=True, help="Output completion HDF5 path.")
    p.add_argument("--n-members", type=int, default=32,
                   help="Number of Laplace/FFT-diagonal ensemble members (0 = MAP only).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--prior-strength", type=float, default=1.0)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--indexing", choices=["compact", "global"], default="global",
                   help="How the inference should index the Q rows. A full survey "
                        "catalog is global-HEALPix-pixel indexed (default).")
    opts = p.parse_args(argv)

    print(f"[*] Building LSS lognormal completion from {opts.catalog}")
    logq_map, logq_members, diagnostics = build_completion(
        opts.catalog, n_members=opts.n_members, seed=opts.seed,
        prior_strength=opts.prior_strength, maxiter=opts.maxiter,
    )
    print(f"    MAP logq_map shape {logq_map.shape}; "
          f"members {'none' if logq_members is None else logq_members.shape}")
    save_lss_completion_hdf5(
        opts.out, logq_map=logq_map, logq_members=logq_members,
        zgrid=np.asarray(zgrid), indexing=opts.indexing, metadata=diagnostics,
    )
    print(f"[*] Wrote {opts.out}")


if __name__ == "__main__":
    main()
