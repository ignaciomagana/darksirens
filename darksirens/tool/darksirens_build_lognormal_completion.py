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
5. Run the solver and save an HDF5 completion file (same table contract either
   way): ``--mode radial`` (default) -> independent per-pixel 1-D
   :func:`poisson_lognormal_map`; ``--mode gp3d`` -> ONE low-rank
   Poisson-lognormal field over occupied (pixel x z) voxels reusing the
   (sphere x z) GP, so empty pixels borrow angularly from their neighbours.
"""
from __future__ import annotations

import argparse

import numpy as np
import jax.numpy as jnp

from darksirens.em import zgrid
from darksirens.em.utils import load_survey
from darksirens.utils.containers import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import r_of_z, H0Planck, Om0Planck, w0Fiducial, waFiducial
from darksirens.em.completion import _precompute_grids, _kde_dndz_obs
from darksirens.em.lognormal_completion import (
    gaussian_correlation_spectrum,
    poisson_lognormal_map,
    laplace_lognormal_members,
    save_lss_completion_hdf5,
    lowrank_inducing_nodes,
    build_lowrank_operator,
    poisson_lognormal_gp3d_map,
    laplace_lognormal_gp3d_members,
    eval_logq_gp3d,
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


def _rebin_counts_to_uniform(counts_z, chi, chi_u):
    """Reassign per-zgrid-bin counts to the uniform-chi bin each point falls in
    (conserves the total, unlike interpolation)."""
    n = chi_u.size
    dchi_u = (chi_u[1] - chi_u[0]) if n > 1 else 1.0
    idx = np.clip(np.round((chi - chi_u[0]) / dchi_u).astype(int), 0, n - 1)
    out = np.zeros(n, dtype=float)
    np.add.at(out, idx, counts_z)
    return out


def _build_completion_radial(
    catalog_path: str,
    *,
    n_members: int = 32,
    seed: int = 1234,
    prior_strength: float = 1.0,
    maxiter: int = 300,
):
    """Radial (per-pixel, independent 1-D) MAP + optional ensemble log Q tables.

    The original LSS completion builder: an independent 1-D Poisson-lognormal
    field per occupied pixel on a uniform comoving-distance grid (no angular
    coupling).  See :func:`_build_completion_gp3d` for the 3-D upgrade.
    """
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
    safe_smooth = np.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    edges = _zgrid_bin_edges()
    ngals_np = np.asarray(ngals).astype(int)
    zgals_np = np.asarray(zgals, dtype=float)

    # Grid-aware P(k): solve on a UNIFORM comoving-distance grid so the Gaussian
    # correlation length is constant in Mpc (zgrid is log-spaced ⇒ Δχ varies).
    chi = np.asarray(r_of_z(jnp.asarray(zgrid), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa), dtype=float)
    chi_u = np.linspace(float(chi[0]), float(chi[-1]), n_grid)
    dchi_u = float(chi_u[1] - chi_u[0]) if n_grid > 1 else 1.0
    ell_grid = float(survey.lss_corr_length_mpc) / max(dchi_u, 1e-6)  # now constant in Mpc
    pk = gaussian_correlation_spectrum(n_grid, ell_grid, float(survey.lss_sigma))
    dN_exp_count_u = np.interp(chi_u, chi, dN_exp_density) * dchi_u  # expected counts / χ-bin

    # Build only OCCUPIED pixels (DESI footprints are mostly empty ⇒ huge speedup);
    # empty pixels get logQ = 0 (Q = 1, homogeneous) by the zero-init below.
    occ = np.nonzero(ngals_np > 0)[0]
    n_occ = int(occ.size)
    C_u = np.empty((n_occ, n_grid), dtype=float)
    N_obs_u = np.zeros((n_occ, n_grid), dtype=float)
    for i, r in enumerate(occ):
        dN_obs_s = np.asarray(_kde_dndz_obs(int(r), em.zgals, ngals=em.ngals), dtype=float)
        C_u[i] = np.clip(np.interp(chi_u, chi, dN_obs_s / safe_smooth), 0.0, 1.0)
        zs = zgals_np[r, : ngals_np[r]]
        counts_z, _ = np.histogram(zs, bins=edges)
        N_obs_u[i] = _rebin_counts_to_uniform(counts_z, chi, chi_u)

    bias = float(survey.b_miss)
    mp = poisson_lognormal_map(
        N_obs_u, C_u, dN_exp_count_u, pk,
        bias=bias, prior_strength=prior_strength, maxiter=maxiter,
    )
    # Map logQ back from uniform-χ to zgrid and scatter occupied rows into the
    # full (n_pix, n_grid) table (empties stay logQ = 0).
    logq_map = np.zeros((n_pix, n_grid), dtype=float)
    for i, r in enumerate(occ):
        logq_map[r] = np.interp(chi, chi_u, mp["logq_map"][i])

    diagnostics = dict(mp["diagnostics"])
    diagnostics.update({
        "nside": int(nside), "n_pix": n_pix, "n_occupied": n_occ,
        "ell_grid_uniform_chi": ell_grid, "dchi_uniform_mpc": dchi_u,
        "lss_corr_length_mpc": float(survey.lss_corr_length_mpc),
        "lss_sigma": float(survey.lss_sigma),
        # Fixed fiducials Q was built at (inference varies these — see the warning
        # printed at load): cosmology, density/evolution, and the field bias.
        "fiducial_H0": float(cosmo.H0), "fiducial_Om0": float(cosmo.Om0),
        "fiducial_w0": float(cosmo.w0), "fiducial_wa": float(cosmo.wa),
        "fiducial_n0": float(survey.n0), "fiducial_delta": float(survey.delta),
        "bias_b_miss": float(survey.b_miss),
    })

    logq_members = None
    if n_members and n_members > 0:
        members = laplace_lognormal_members(
            mp["s_map"], mp["lambda_map"], pk,
            n_members=int(n_members), bias=bias, prior_strength=prior_strength, seed=int(seed),
        )
        lm_u = members["logq_members"]                       # (M, n_occ, n_grid) on χ_u
        M = int(n_members)
        logq_members = np.zeros((M, n_pix, n_grid), dtype=float)
        for i, r in enumerate(occ):
            for m in range(M):
                logq_members[m, r] = np.interp(chi, chi_u, lm_u[m, i])
        diagnostics.update(members["diagnostics"])

    return logq_map, logq_members, diagnostics


def _build_completion_gp3d(
    catalog_path: str,
    *,
    n_members: int = 32,
    seed: int = 1234,
    gp3d_nz_solve: int = 32,
    gp3d_pix_chunk: int = 512,
    lss_corr_length_ang=None,
):
    """3-D angular-coupling MAP + optional ensemble log Q tables.

    Solves ONE low-rank Poisson-lognormal field over the occupied (pixel x z)
    voxels using the (sphere x z) GP, so empty pixels borrow angularly from their
    neighbours and pixels far from any data read as exactly Q = 1.  Output is the
    SAME global ``(n_pix, n_grid)`` log Q table contract as the radial builder, so
    the inference side is unchanged.
    """
    import healpy as hp

    M_SPH, M_Z = 32, 6

    nside, ngals, zgals, dzgals, wgals = load_survey(catalog_path)
    n_pix = int(np.asarray(zgals).shape[0])
    n_grid = int(zgrid.size)
    apix = float(hp.nside2pixarea(int(nside)))

    cosmo, survey = _fiducial_cosmo_survey()
    if lss_corr_length_ang is not None:
        survey = survey._replace(lss_corr_length_ang=float(lss_corr_length_ang))

    em = EMCatalog(
        apix=apix, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, n_grid)), dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(cosmo, survey, em)
    dN_exp_density = np.asarray(grids.dN_exp, dtype=float)
    dN_exp_smooth = np.asarray(grids.dN_exp_smooth, dtype=float)
    safe_smooth = np.where(dN_exp_smooth > 0.0, dN_exp_smooth, 1.0)
    zgrid_np = np.asarray(zgrid, dtype=float)
    ngals_np = np.asarray(ngals).astype(int)
    zgals_np = np.asarray(zgals, dtype=float)

    bias = float(survey.b_miss)
    amp = float(survey.lss_sigma)
    ls_sph = float(survey.lss_corr_length_ang)

    # All-pixel output directions (RING ordering — matches the catalog pixelation
    # and the inference event->pixel mapping).
    n_hat_all = np.asarray(
        hp.pix2vec(int(nside), np.arange(n_pix), nest=False), dtype=float).T  # (n_pix, 3)

    occ = np.nonzero(ngals_np > 0)[0]
    n_occ = int(occ.size)

    def _base_diag(extra):
        d = {
            "mode": "gp3d", "nside": int(nside), "n_pix": n_pix, "n_occupied": n_occ,
            "lss_corr_length_mpc": float(survey.lss_corr_length_mpc),
            "lss_sigma": amp, "lss_corr_length_ang": ls_sph,
            # Fixed fiducials Q was built at (inference varies these; see load warning).
            "fiducial_H0": float(cosmo.H0), "fiducial_Om0": float(cosmo.Om0),
            "fiducial_w0": float(cosmo.w0), "fiducial_wa": float(cosmo.wa),
            "fiducial_n0": float(survey.n0), "fiducial_delta": float(survey.delta),
            "bias_b_miss": bias,
        }
        d.update(extra)
        return d

    # Empty catalog: nothing to solve, Q == 1 everywhere.
    if n_occ == 0:
        print("    [!] no occupied pixels — writing Q = 1 (logQ = 0) everywhere.")
        logq_map = np.zeros((n_pix, n_grid), dtype=float)
        logq_members = (np.zeros((int(n_members), n_pix, n_grid), dtype=float)
                        if n_members and n_members > 0 else None)
        return logq_map, logq_members, _base_diag({})

    # Coarse SOLVE z-grid: the field is low-rank in z (only M_z nodes), so a fine
    # solve grid adds no resolvable radial signal.  Output is on the full zgrid.
    z_s = np.linspace(0.0, float(zgrid_np[-1]), int(gp3d_nz_solve))
    dz_s = np.gradient(z_s)
    edges_s = np.concatenate([[z_s[0]], 0.5 * (z_s[:-1] + z_s[1:]), [z_s[-1]]])
    zeta_s = np.log1p(z_s)
    G_s = int(z_s.size)
    # Expected counts per coarse z-bin (delta-weighted density — same footing as
    # the integer observed counts).
    dN_exp_count_s = np.interp(z_s, zgrid_np, dN_exp_density) * dz_s        # (G_s,)

    n_hat_occ = np.asarray(hp.pix2vec(int(nside), occ, nest=False), dtype=float).T  # (n_occ,3)
    C_vox = np.empty((n_occ, G_s), dtype=float)
    N_obs_vox = np.zeros((n_occ, G_s), dtype=float)
    for i, r in enumerate(occ):
        dN_obs_s = np.asarray(_kde_dndz_obs(int(r), em.zgals, ngals=em.ngals), dtype=float)
        C_vox[i] = np.clip(np.interp(z_s, zgrid_np, dN_obs_s / safe_smooth), 0.0, 1.0)
        zs = zgals_np[r, : ngals_np[r]]
        counts_s, _ = np.histogram(zs, bins=edges_s)
        N_obs_vox[i] = counts_s
    base_vox = C_vox * dN_exp_count_s[None, :]                             # (n_occ, G_s)

    # Map the fixed radial Mpc length into the kernel's zeta = log1p(z) units at a
    # count-weighted reference redshift: ls_z = L_mpc * dzeta/dchi |_{z_ref}.
    counts_z = N_obs_vox.sum(axis=0)
    z_ref = (float(np.sum(z_s * counts_z) / np.sum(counts_z))
             if np.sum(counts_z) > 0 else 1.0)
    eps = 1e-3
    zlo, zhi = max(z_ref - eps, 0.0), z_ref + eps
    dchi_dz = float(
        (r_of_z(jnp.asarray(zhi), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
         - r_of_z(jnp.asarray(zlo), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa))
        / (zhi - zlo)
    )
    ls_z = float(survey.lss_corr_length_mpc) / max((1.0 + z_ref) * dchi_dz, 1e-6)

    # Low-rank operator over the occupied voxels.
    Zn, Zz = lowrank_inducing_nodes(n_inducing_sphere=M_SPH, n_inducing_z=M_Z)
    M = int(np.asarray(Zn).shape[0])
    X_n = np.repeat(n_hat_occ, G_s, axis=0)                                # (n_occ*G_s, 3)
    X_z = np.tile(zeta_s, n_occ)                                           # (n_occ*G_s,)
    Phi, L = build_lowrank_operator(Zn, Zz, X_n, X_z, amp=amp, ls_sph=ls_sph, ls_z=ls_z)

    mp = poisson_lognormal_gp3d_map(N_obs_vox.ravel(), base_vox.ravel(), Phi, bias=bias)

    # Global all-pixel output table on the full package zgrid (chunked over
    # pixels): the Laplace posterior-mean E[Q] (empty pixels -> Q=1, near-data
    # pixels borrow), so H_chol is passed.
    logq_map = np.asarray(eval_logq_gp3d(
        mp["xi_map"], Zn, Zz, amp=amp, ls_sph=ls_sph, ls_z=ls_z,
        n_hat_out=n_hat_all, z_out=zgrid_np, bias=bias,
        pix_chunk=int(gp3d_pix_chunk), L=L, H_chol=mp["H_chol"],
    ), dtype=float)
    if not np.all(np.isfinite(logq_map)):
        n_bad = int(np.sum(~np.isfinite(logq_map)))
        print(f"    [!] {n_bad} non-finite logQ entries -> set to 0 (Q = 1).")
        logq_map = np.where(np.isfinite(logq_map), logq_map, 0.0)

    sig2 = np.asarray(mp["sigma2_vox"], dtype=float)
    diagnostics = _base_diag({
        "M_sph": M_SPH, "M_z": M_Z, "M": M, "gp3d_nz_solve": G_s,
        "ls_z_zeta": ls_z, "z_ref": z_ref, "ls_sph_chordal": ls_sph,
        "sigma2_vox_min": float(sig2.min()), "sigma2_vox_max": float(sig2.max()),
        "n_iter": int(mp["diagnostics"]["n_iter"]),
        "converged": bool(mp["diagnostics"]["converged"]),
        "grad_inf": float(mp["diagnostics"]["grad_inf"]),
    })

    logq_members = None
    if n_members and n_members > 0:
        xi_mem = laplace_lognormal_gp3d_members(
            mp["xi_map"], mp["H_chol"], n_members=int(n_members), seed=int(seed))
        logq_members = np.asarray(eval_logq_gp3d(
            xi_mem, Zn, Zz, amp=amp, ls_sph=ls_sph, ls_z=ls_z,
            n_hat_out=n_hat_all, z_out=zgrid_np, bias=bias,
            pix_chunk=int(gp3d_pix_chunk), L=L,
        ), dtype=float)
        logq_members = np.where(np.isfinite(logq_members), logq_members, 0.0)
        diagnostics["n_members"] = int(n_members)

    return logq_map, logq_members, diagnostics


def build_completion(
    catalog_path: str,
    *,
    mode: str = "radial",
    n_members: int = 32,
    seed: int = 1234,
    prior_strength: float = 1.0,
    maxiter: int = 300,
    gp3d_nz_solve: int = 32,
    gp3d_pix_chunk: int = 512,
    lss_corr_length_ang=None,
):
    """Build the log Q completion tables from a survey catalog.

    ``mode="radial"`` (default) -> independent per-pixel 1-D Poisson-lognormal
    (:func:`_build_completion_radial`).  ``mode="gp3d"`` -> the 3-D
    angular-coupling low-rank field (:func:`_build_completion_gp3d`).  Both return
    ``(logq_map, logq_members, diagnostics)`` with the SAME global table contract.
    """
    if mode == "radial":
        return _build_completion_radial(
            catalog_path, n_members=n_members, seed=seed,
            prior_strength=prior_strength, maxiter=maxiter,
        )
    if mode == "gp3d":
        return _build_completion_gp3d(
            catalog_path, n_members=n_members, seed=seed,
            gp3d_nz_solve=gp3d_nz_solve, gp3d_pix_chunk=gp3d_pix_chunk,
            lss_corr_length_ang=lss_corr_length_ang,
        )
    raise ValueError(f"Unknown build mode {mode!r}; expected 'radial' or 'gp3d'.")


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
    p.add_argument("--mode", choices=["radial", "gp3d"], default="radial",
                   help="Completion model: 'radial' (default; independent per-pixel "
                        "1-D Poisson-lognormal) or 'gp3d' (3-D angular-coupling "
                        "low-rank field — empty pixels borrow from neighbours).")
    p.add_argument("--gp3d-nz-solve", type=int, default=32,
                   help="(gp3d) Number of coarse redshift points for the solve grid.")
    p.add_argument("--gp3d-pix-chunk", type=int, default=512,
                   help="(gp3d) Pixel chunk size for the all-pixel field evaluation.")
    p.add_argument("--lss-corr-length-ang", type=float, default=None,
                   help="(gp3d) Override the fixed angular (chordal) correlation "
                        "length; default is the SurveyParams fiducial.")
    p.add_argument("--indexing", choices=["compact", "global"], default="global",
                   help="How the inference should index the Q rows. A full survey "
                        "catalog is global-HEALPix-pixel indexed (default).")
    opts = p.parse_args(argv)

    indexing = opts.indexing
    if opts.mode == "gp3d" and indexing != "global":
        print("    [!] gp3d builds a global all-pixel table; forcing indexing='global'.")
        indexing = "global"

    print(f"[*] Building LSS lognormal completion ({opts.mode}) from {opts.catalog}")
    logq_map, logq_members, diagnostics = build_completion(
        opts.catalog, mode=opts.mode, n_members=opts.n_members, seed=opts.seed,
        prior_strength=opts.prior_strength, maxiter=opts.maxiter,
        gp3d_nz_solve=opts.gp3d_nz_solve, gp3d_pix_chunk=opts.gp3d_pix_chunk,
        lss_corr_length_ang=opts.lss_corr_length_ang,
    )
    print(f"    MAP logq_map shape {logq_map.shape}; "
          f"members {'none' if logq_members is None else logq_members.shape}")
    if opts.mode == "gp3d":
        print(f"    gp3d: converged={diagnostics.get('converged')} "
              f"n_iter={diagnostics.get('n_iter')} grad_inf={diagnostics.get('grad_inf'):.2e} "
              f"ls_z={diagnostics.get('ls_z_zeta'):.4f} z_ref={diagnostics.get('z_ref'):.3f}")
    save_lss_completion_hdf5(
        opts.out, logq_map=logq_map, logq_members=logq_members,
        zgrid=np.asarray(zgrid), indexing=indexing, metadata=diagnostics,
    )
    print(f"[*] Wrote {opts.out}")


if __name__ == "__main__":
    main()
