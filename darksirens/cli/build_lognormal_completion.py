"""
darksirens_build_lognormal_completion
-------------------------------------
**Offline** preprocessor that builds an LSS-conditioned lognormal completion
file ``Q_LSS(p, z)`` from a pixelated survey catalog, for later consumption by
the dark-siren redshift prior (``darksirens_inference --lss_completion``).

It must NOT be run inside the GW likelihood — it is a one-off build step.

Pipeline
--------
1. Load the survey catalog (``darksirens.catalogs.io.load_survey``).
2. Bin observed galaxies onto the package ``zgrid`` per pixel  -> ``N_obs``.
3. Compute the matched-kernel completeness ``C`` and homogeneous expected
   counts ``dN_exp`` with the existing completion machinery
   (:func:`darksirens.redshift.completion._precompute_grids`,
   :func:`darksirens.redshift.completion._kde_dndz_obs`) under a *fiducial*
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
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp

from darksirens.cli.common import _banner, _section, _row, _end, _ok, _warn
from darksirens.redshift import zgrid
from darksirens.catalogs.io import load_survey
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.utils.cosmology import r_of_z, H0Planck, Om0Planck, w0Fiducial, waFiducial
from darksirens.redshift.completion import _precompute_grids, _kde_dndz_obs
from darksirens.redshift.lognormal_completion import (
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


def _fiducial_cosmo_survey(log10n0=None, delta=None):
    """Fiducial cosmology + survey (matching the inference dry-run defaults).

    ``log10n0``/``delta`` override the expected-density normalisation
    ``dN_exp = n0 * apix * dV_c/dz * (1+z)^delta`` that the Poisson-lognormal
    fit is conditioned on.  Calibrate ``log10n0`` to the survey's observed
    comoving density: a mis-set n0 cannot be separated from completeness and
    is absorbed into Q with spurious redshift structure (which then biases
    any downstream H0 inference — see working/GATES.md V2/V3).
    """
    cosmo = CosmoParams(H0=H0Planck, Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial)
    survey = SurveyParams(
        n0=10.0 ** (float(log10n0) if log10n0 is not None else -2.0),
        z50=1.0, w=0.5,
        delta=float(delta) if delta is not None else 0.0,
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
    log10n0=None,
    delta=None,
):
    """Radial (per-pixel, independent 1-D) MAP + optional ensemble log Q tables.

    The original LSS completion builder: an independent 1-D Poisson-lognormal
    field per occupied pixel on a uniform comoving-distance grid (no angular
    coupling).  See :func:`_build_completion_gp3d` for the 3-D upgrade.
    """
    import healpy as hp

    nside, ngals, zgals, dzgals, wgals, _z_depth = load_survey(catalog_path)
    n_pix = int(np.asarray(zgals).shape[0])
    n_grid = int(zgrid.size)
    apix = float(hp.nside2pixarea(int(nside)))

    cosmo, survey = _fiducial_cosmo_survey(log10n0=log10n0, delta=delta)
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

    # Expected counts per uniform-chi bin: INTEGRATE the per-unit-z density in z
    # between the bin-edge redshifts (cumulative trapezoid on the fine grid,
    # differenced at the chi-bin edges; chi(z) is monotone so the edges map
    # 1:1). The previous point-evaluation `interp(chi_u, chi, density) * dchi_u`
    # multiplied a per-unit-z density by a bin width in Mpc — missing the
    # dz/dchi Jacobian (~ c/H(z) ~ 4e3) and inflating the homogeneous
    # expectation ~1e3x, which crushed the field and railed every radial table
    # at the -7 clip (library review P0.1). Bin edges are the chi-node
    # midpoints, matching the round-to-nearest convention of
    # `_rebin_counts_to_uniform` used for N_obs.
    zg = np.asarray(zgrid, dtype=float)
    dz_fine = np.diff(zg)
    edges_chi = np.concatenate([[chi_u[0]], 0.5 * (chi_u[:-1] + chi_u[1:]), [chi_u[-1]]])

    def _bin_integral(density_fine):
        cum = np.concatenate([
            [0.0], np.cumsum(0.5 * (density_fine[1:] + density_fine[:-1]) * dz_fine)
        ])
        return np.diff(np.interp(edges_chi, chi, cum))

    dN_exp_count_u = _bin_integral(dN_exp_density)               # (n_grid,)
    exp_safe = np.where(dN_exp_count_u > 0.0, dN_exp_count_u, 1.0)

    # Build only OCCUPIED pixels (DESI footprints are mostly empty ⇒ huge speedup);
    # empty pixels get logQ = 0 (Q = 1, homogeneous) by the zero-init below.
    occ = np.nonzero(ngals_np > 0)[0]
    n_occ = int(occ.size)
    C_u = np.empty((n_occ, n_grid), dtype=float)
    N_obs_u = np.zeros((n_occ, n_grid), dtype=float)
    for i, r in enumerate(occ):
        dN_obs_s = np.asarray(_kde_dndz_obs(int(r), em.zgals, ngals=em.ngals), dtype=float)
        # Expectation-weighted bin completeness: C_bin = int(C * dN_exp) / int(dN_exp),
        # so the solver's rate_base = C_u * dN_exp_count_u is exactly the
        # bin-integrated expected OBSERVED counts — the same footing as N_obs.
        prod = np.clip(dN_obs_s / safe_smooth, 0.0, 1.0) * dN_exp_density
        C_u[i] = np.clip(_bin_integral(prod) / exp_safe, 0.0, 1.0)
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


class _GP3DSurveyAssembly(NamedTuple):
    """Per-survey gp3d voxel inputs on the SHARED coarse solve grid ``z_s``.

    Everything :func:`_build_completion_gp3d` (K=1) and
    :func:`darksirens.cli.build_joint_lognormal_completion.build_joint_completion`
    (K>=1) need to stack a survey into the low-rank Poisson-lognormal field: the
    occupied-voxel geometry (``n_hat_occ``), the all-pixel output directions
    (``n_hat_all``), the coarse-bin observed counts (``N_obs_vox``) and
    bin-integrated expected observed base (``base_vox``), and the per-survey
    coarse-z count profile (``counts_z``) that the joint ``z_ref`` weighting sums
    over all surveys.
    """
    nside: int
    n_pix: int
    apix: float
    occ: np.ndarray          # (n_occ,) occupied RING pixel indices
    n_occ: int
    n_hat_all: np.ndarray    # (n_pix, 3) all-pixel output directions
    n_hat_occ: np.ndarray    # (n_occ, 3) occupied-voxel directions
    N_obs_vox: np.ndarray    # (n_occ, G_s) coarse-bin histogram counts
    base_vox: np.ndarray     # (n_occ, G_s) bin-integrated expected observed counts
    counts_z: np.ndarray     # (G_s,) coarse-z count profile (for z_ref weighting)


def _gp3d_base_diagnostics(cosmo, survey, *, nside, n_pix, n_occ, amp, ls_sph,
                           bias, mode="gp3d"):
    """The gp3d diagnostics block every build (single or joint) records.

    ``mode`` is ``"gp3d"`` for the single-survey builder and ``"gp3d_joint"``
    for the joint multi-survey builder; the fiducial keys are what the inference
    loader prints at Q load time (build-time cosmology/density/bias).
    """
    return {
        "mode": mode, "nside": int(nside), "n_pix": int(n_pix),
        "n_occupied": int(n_occ),
        "lss_corr_length_mpc": float(survey.lss_corr_length_mpc),
        "lss_sigma": amp, "lss_corr_length_ang": ls_sph,
        # Fixed fiducials Q was built at (inference varies these; see load warning).
        "fiducial_H0": float(cosmo.H0), "fiducial_Om0": float(cosmo.Om0),
        "fiducial_w0": float(cosmo.w0), "fiducial_wa": float(cosmo.wa),
        "fiducial_n0": float(survey.n0), "fiducial_delta": float(survey.delta),
        "bias_b_miss": bias,
    }


def _assemble_gp3d_survey(catalog_path, *, cosmo, survey, z_s, edges_s):
    """Load a survey and assemble its gp3d voxel inputs on the shared ``z_s`` grid.

    Extracted verbatim from the single-survey ``_build_completion_gp3d`` body so
    the K=1 driver and the JOINT multi-survey builder share ONE assembly path
    (bit-identical at K=1 — same operation order).  ``z_s``/``edges_s`` are the
    SHARED coarse solve grid; they depend only on the package ``zgrid`` and
    ``gp3d_nz_solve`` (never on the survey), so every survey in a joint fit lands
    on the same radial bins.  ``survey`` carries this survey's own fiducial n0 /
    delta (its expected-density normalisation); the field bias ``b_miss`` does
    NOT enter the assembly (``_precompute_grids`` depends only on n0/delta/cosmo),
    so a per-survey bias is applied later by scaling the design matrix.
    """
    import healpy as hp

    nside, ngals, zgals, dzgals, wgals, _z_depth = load_survey(catalog_path)
    n_pix = int(np.asarray(zgals).shape[0])
    n_grid = int(zgrid.size)
    apix = float(hp.nside2pixarea(int(nside)))

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

    # All-pixel output directions (RING ordering — matches the catalog pixelation
    # and the inference event->pixel mapping).
    n_hat_all = np.asarray(
        hp.pix2vec(int(nside), np.arange(n_pix), nest=False), dtype=float).T  # (n_pix, 3)

    occ = np.nonzero(ngals_np > 0)[0]
    n_occ = int(occ.size)
    n_hat_occ = np.asarray(hp.pix2vec(int(nside), occ, nest=False), dtype=float).T  # (n_occ,3)

    # Expected OBSERVED counts per coarse z-bin: integrate C(z) * dN_exp(z) over
    # each bin on the fine grid, on the same footing as the histogram that fills
    # N_obs_vox over the same bin. Point-evaluating C at the coarse node instead
    # is catastrophic wherever the KDE support and the bin contents disagree
    # (e.g. the z=0 node of a low-z-complete catalog: C(0) ~ KDE tail ~ 0 while
    # the bin holds real galaxies, so the fit demands field values beyond
    # field_clip and the solve saturates).
    G_s = int(np.asarray(z_s).size)
    dz_fine = np.diff(zgrid_np)
    N_obs_vox = np.zeros((n_occ, G_s), dtype=float)
    base_vox = np.empty((n_occ, G_s), dtype=float)
    for i, r in enumerate(occ):
        dN_obs_s = np.asarray(_kde_dndz_obs(int(r), em.zgals, ngals=em.ngals), dtype=float)
        prod = np.clip(dN_obs_s / safe_smooth, 0.0, 1.0) * dN_exp_density
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (prod[1:] + prod[:-1]) * dz_fine)])
        base_vox[i] = np.diff(np.interp(edges_s, zgrid_np, cum))
        zs = zgals_np[r, : ngals_np[r]]
        counts_s, _ = np.histogram(zs, bins=edges_s)
        N_obs_vox[i] = counts_s

    counts_z = N_obs_vox.sum(axis=0)
    return _GP3DSurveyAssembly(
        nside=int(nside), n_pix=n_pix, apix=apix, occ=occ, n_occ=n_occ,
        n_hat_all=n_hat_all, n_hat_occ=n_hat_occ,
        N_obs_vox=N_obs_vox, base_vox=base_vox, counts_z=counts_z,
    )


def _count_weighted_zref_lsz(counts_per_survey, z_s, cosmo, corr_length_mpc):
    """Count-weighted reference redshift over one-or-many surveys, and the
    ``zeta = log1p(z)``-units GP length ``ls_z`` at that redshift.

    Map the fixed radial Mpc length into the kernel's zeta units at a
    count-weighted reference redshift ``z_ref = sum(z_s * N) / sum(N)`` summed
    over ALL surveys' coarse-z count profiles: ``ls_z = L_mpc * dzeta/dchi
    |_{z_ref}``.  K=1 reduces to the single-survey arithmetic EXACTLY (one
    summand added to a zero accumulator is a no-op in float; same formula, same
    operation order).
    """
    z_s = np.asarray(z_s, dtype=float)
    counts_z = np.zeros(int(z_s.size), dtype=float)
    for c in counts_per_survey:
        counts_z = counts_z + np.asarray(c, dtype=float)
    z_ref = (float(np.sum(z_s * counts_z) / np.sum(counts_z))
             if np.sum(counts_z) > 0 else 1.0)
    eps = 1e-3
    zlo, zhi = max(z_ref - eps, 0.0), z_ref + eps
    dchi_dz = float(
        (r_of_z(jnp.asarray(zhi), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
         - r_of_z(jnp.asarray(zlo), cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa))
        / (zhi - zlo)
    )
    ls_z = float(corr_length_mpc) / max((1.0 + z_ref) * dchi_dz, 1e-6)
    return z_ref, ls_z


def _build_completion_gp3d(
    catalog_path: str,
    *,
    n_members: int = 32,
    seed: int = 1234,
    gp3d_nz_solve: int = 32,
    gp3d_pix_chunk: int = 512,
    lss_corr_length_ang=None,
    log10n0=None,
    delta=None,
):
    """3-D angular-coupling MAP + optional ensemble log Q tables.

    Solves ONE low-rank Poisson-lognormal field over the occupied (pixel x z)
    voxels using the (sphere x z) GP, so empty pixels borrow angularly from their
    neighbours and pixels far from any data read as exactly Q = 1.  Output is the
    SAME global ``(n_pix, n_grid)`` log Q table contract as the radial builder, so
    the inference side is unchanged.

    The per-survey voxel assembly (:func:`_assemble_gp3d_survey`) and the
    ``z_ref``/``ls_z`` map (:func:`_count_weighted_zref_lsz`) are the SAME helpers
    the joint multi-survey builder stacks over, so this K=1 path is one instance
    of the joint fit.
    """
    M_SPH, M_Z = 32, 6

    cosmo, survey = _fiducial_cosmo_survey(log10n0=log10n0, delta=delta)
    if lss_corr_length_ang is not None:
        survey = survey._replace(lss_corr_length_ang=float(lss_corr_length_ang))

    bias = float(survey.b_miss)
    amp = float(survey.lss_sigma)
    ls_sph = float(survey.lss_corr_length_ang)
    zgrid_np = np.asarray(zgrid, dtype=float)
    n_grid = int(zgrid.size)

    # Coarse SOLVE z-grid: the field is low-rank in z (only M_z nodes), so a fine
    # solve grid adds no resolvable radial signal.  Output is on the full zgrid.
    # Survey-independent (depends only on zgrid and gp3d_nz_solve), so the joint
    # builder shares it across surveys.
    z_s = np.linspace(0.0, float(zgrid_np[-1]), int(gp3d_nz_solve))
    edges_s = np.concatenate([[z_s[0]], 0.5 * (z_s[:-1] + z_s[1:]), [z_s[-1]]])
    zeta_s = np.log1p(z_s)
    G_s = int(z_s.size)

    survey_data = _assemble_gp3d_survey(
        catalog_path, cosmo=cosmo, survey=survey, z_s=z_s, edges_s=edges_s)
    nside, n_pix, n_occ = survey_data.nside, survey_data.n_pix, survey_data.n_occ

    def _base_diag(extra):
        d = _gp3d_base_diagnostics(
            cosmo, survey, nside=nside, n_pix=n_pix, n_occ=n_occ,
            amp=amp, ls_sph=ls_sph, bias=bias)
        d.update(extra)
        return d

    # Empty catalog: nothing to solve, Q == 1 everywhere.
    if n_occ == 0:
        _warn("no occupied pixels — writing Q = 1 (logQ = 0) everywhere.")
        logq_map = np.zeros((n_pix, n_grid), dtype=float)
        logq_members = (np.zeros((int(n_members), n_pix, n_grid), dtype=float)
                        if n_members and n_members > 0 else None)
        return logq_map, logq_members, _base_diag({})

    z_ref, ls_z = _count_weighted_zref_lsz(
        [survey_data.counts_z], z_s, cosmo, survey.lss_corr_length_mpc)

    # Low-rank operator over the occupied voxels.
    Zn, Zz = lowrank_inducing_nodes(n_inducing_sphere=M_SPH, n_inducing_z=M_Z)
    M = int(np.asarray(Zn).shape[0])
    X_n = np.repeat(survey_data.n_hat_occ, G_s, axis=0)                    # (n_occ*G_s, 3)
    X_z = np.tile(zeta_s, n_occ)                                           # (n_occ*G_s,)
    Phi, L = build_lowrank_operator(Zn, Zz, X_n, X_z, amp=amp, ls_sph=ls_sph, ls_z=ls_z)

    mp = poisson_lognormal_gp3d_map(
        survey_data.N_obs_vox.ravel(), survey_data.base_vox.ravel(), Phi, bias=bias)

    # Global all-pixel output table on the full package zgrid (chunked over
    # pixels): the Laplace posterior-mean E[Q] (empty pixels -> Q=1, near-data
    # pixels borrow), so H_chol is passed.
    logq_map = np.asarray(eval_logq_gp3d(
        mp["xi_map"], Zn, Zz, amp=amp, ls_sph=ls_sph, ls_z=ls_z,
        n_hat_out=survey_data.n_hat_all, z_out=zgrid_np, bias=bias,
        pix_chunk=int(gp3d_pix_chunk), L=L, H_chol=mp["H_chol"],
    ), dtype=float)
    if not np.all(np.isfinite(logq_map)):
        n_bad = int(np.sum(~np.isfinite(logq_map)))
        _warn(f"{n_bad} non-finite logQ entries -> set to 0 (Q = 1).")
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
            n_hat_out=survey_data.n_hat_all, z_out=zgrid_np, bias=bias,
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
    log10n0=None,
    delta=None,
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
            log10n0=log10n0, delta=delta,
        )
    if mode == "gp3d":
        return _build_completion_gp3d(
            catalog_path, n_members=n_members, seed=seed,
            gp3d_nz_solve=gp3d_nz_solve, gp3d_pix_chunk=gp3d_pix_chunk,
            lss_corr_length_ang=lss_corr_length_ang,
            log10n0=log10n0, delta=delta,
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
    p.add_argument("--log10n0", type=float, default=None,
                   help="Override log10 of the expected comoving galaxy density "
                        "[Mpc^-3] the fit is conditioned on (default -2.0). "
                        "CALIBRATE this to the catalog (N / (f_sky * V_c)); a "
                        "mis-set n0 is absorbed into Q as spurious z-structure.")
    p.add_argument("--delta", type=float, default=None,
                   help="Override the expected-density evolution exponent "
                        "(1+z)^delta (default 0.0).")
    p.add_argument("--indexing", choices=["compact", "global"], default="global",
                   help="How the inference should index the Q rows. A full survey "
                        "catalog is global-HEALPix-pixel indexed (default).")
    opts = p.parse_args(argv)

    print()
    _banner("LSS LOGNORMAL COMPLETION")
    print()

    indexing = opts.indexing
    if opts.mode == "gp3d" and indexing != "global":
        _warn("gp3d builds a global all-pixel table; forcing indexing='global'.")
        indexing = "global"

    _section(f"Building  [{opts.mode}]")
    _row("Catalog", opts.catalog)
    _row("Members", opts.n_members)
    _row("Seed", opts.seed)
    logq_map, logq_members, diagnostics = build_completion(
        opts.catalog, mode=opts.mode, n_members=opts.n_members, seed=opts.seed,
        prior_strength=opts.prior_strength, maxiter=opts.maxiter,
        gp3d_nz_solve=opts.gp3d_nz_solve, gp3d_pix_chunk=opts.gp3d_pix_chunk,
        lss_corr_length_ang=opts.lss_corr_length_ang,
        log10n0=opts.log10n0, delta=opts.delta,
    )
    _ok(f"MAP logq_map shape {logq_map.shape}; "
        f"members {'none' if logq_members is None else logq_members.shape}")
    if opts.mode == "gp3d":
        _ok(f"gp3d: converged={diagnostics.get('converged')} "
            f"n_iter={diagnostics.get('n_iter')} grad_inf={diagnostics.get('grad_inf'):.2e} "
            f"ls_z={diagnostics.get('ls_z_zeta'):.4f} z_ref={diagnostics.get('z_ref'):.3f}")
    elif opts.mode == "radial":
        n_conv = diagnostics.get("n_converged")
        n_occ = diagnostics.get("n_occupied")
        if n_conv is not None and n_occ:
            frac = n_conv / n_occ if n_occ else float("nan")
            line = f"radial: {n_conv}/{n_occ} occupied-pixel solves converged ({frac:.1%})"
            if n_conv == n_occ:
                _ok(line)
            else:
                _warn(line + " — some pixel solves did not converge")
    save_lss_completion_hdf5(
        opts.out, logq_map=logq_map, logq_members=logq_members,
        zgrid=np.asarray(zgrid), indexing=indexing, metadata=diagnostics,
    )
    _ok(f"completion  →  {opts.out}")
    _end()


if __name__ == "__main__":
    main()
