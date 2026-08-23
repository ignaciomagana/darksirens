#!/usr/bin/env python3
"""Generate a mock galaxy catalog with KNOWN lognormal clustering.

Closure-test generator: the clustering truth is a mean-one lognormal field
drawn from the SAME (sphere × z) low-rank GP family the darksirens gp3d
Q-builder fits (``--truth-kernel gp3d``, default), or independent per-pixel
radial Gaussian-correlated fields matching the radial builder's assumptions
(``--truth-kernel radial``).  Galaxy counts are Poisson draws from
``n0 · ΔV_c(voxel) · Q_truth(voxel)``; per-galaxy magnitudes, photo-z errors,
and the survey selection reuse the dark-siren mock generator verbatim, so the
"observed" pixelated catalog is exactly what the completeness fitters consume.

Outputs (under --outdir):
  catalog_pixelated_nside_<n>.h5  — load_survey schema, with z_depth attr
  truth.h5                        — xi_true, logq truth on voxels AND on the
                                    package zgrid, complete catalog columns,
                                    analytic selection ingredients
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import common  # noqa: F401  (side effect: puts the repo ROOT on sys.path)
from common import (
    ROOT,  # noqa: F401  (re-exported for provenance printing)
    eval_truth_field_logq,
    fiducial_cosmo,
    load_dark_mock_module,
)
from darksirens.redshift import zgrid

_dark = load_dark_mock_module()


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nside", type=int, default=16)
    p.add_argument("--zmax", type=float, default=0.5,
                   help="Survey depth: hard z cut AND the z_depth attr.")
    p.add_argument("--log10n0", type=float, default=None,
                   help="Comoving density [Mpc^-3]; default derives from --n-target.")
    p.add_argument("--n-target", type=float, default=3.0e5,
                   help="Target complete-catalog size when --log10n0 is unset.")
    p.add_argument("--truth-kernel", choices=["gp3d", "radial"], default="gp3d")
    p.add_argument("--lss-sigma", type=float, default=1.0,
                   help="Field amplitude (matches SurveyParams.lss_sigma).")
    p.add_argument("--corr-ang", type=float, default=0.2,
                   help="Chordal angular correlation length (gp3d).")
    p.add_argument("--corr-mpc", type=float, default=800.0,
                   help="Radial correlation length in Mpc.  Default 800 (NOT "
                        "the SurveyParams fiducial 50): 50 Mpc maps to "
                        "ls_z ~ 0.01 in zeta units, far below the gp3d "
                        "inducing-node spacing (0.277) and solve-grid "
                        "resolution, so the shipped gp3d family cannot "
                        "represent it and the radial fit is shot-noise "
                        "dominated (<1 galaxy per correlation cell at mock "
                        "densities).  800 Mpc sits inside the resolvable band "
                        "so the closure test measures the estimator, not the "
                        "representation gap.")
    p.add_argument("--n-zbins", type=int, default=192,
                   help="Uniform-comoving-distance truth voxel bins.")
    p.add_argument("--footprint", action="store_true",
                   help="Apply the fiducial dec footprint [-40, 80] deg "
                        "(default: full sky, so every pixel is occupied).")
    p.add_argument("--z50", type=float, default=0.30,
                   help="Logistic completeness midpoint (fiducial mock: 0.75; "
                        "lowered so the roll-off bites well inside --zmax).")
    p.add_argument("--width", type=float, default=0.08)
    p.add_argument("--mag-limit", type=float, default=24.0)
    p.add_argument("--aniso-dmag", type=float, default=0.0,
                   help="Anisotropic survey depth: hemispheres at "
                        "mag_limit +/- dmag (north dec>=0 deeper). The "
                        "per-galaxy limit and a stratum id are written to "
                        "truth.h5; the counts-based aggregate Cbar provably "
                        "cannot represent this (the misspecification-alarm "
                        "closure variant), while per-stratum magnitude fits "
                        "recover each hemisphere's selection.")
    return p.parse_args(argv)


def _sample_sky_in_pixels(rng, counts, nside):
    """Uniform sky positions grouped by HEALPix RING pixel.

    Returns (ra, dec) arrays of length counts.sum(), ordered pixel-by-pixel
    (pixel 0's galaxies first).  Rejection from the full sphere, batched.
    """
    import healpy as hp

    counts = np.asarray(counts, dtype=int)
    npix = counts.size
    total = int(counts.sum())
    starts = np.concatenate([[0], np.cumsum(counts)])
    ra_out = np.empty(total)
    dec_out = np.empty(total)
    filled = np.zeros(npix, dtype=int)
    n_done = 0
    while n_done < total:
        batch = max(4 * (total - n_done), 100_000)
        ra = rng.uniform(0.0, 2.0 * np.pi, batch)
        dec = np.arcsin(rng.uniform(-1.0, 1.0, batch))
        pix = hp.ang2pix(nside, np.pi / 2.0 - dec, ra)
        order = np.argsort(pix, kind="stable")
        pix_s, ra_s, dec_s = pix[order], ra[order], dec[order]
        uniq, first, navail = np.unique(pix_s, return_index=True, return_counts=True)
        for u, f0, k in zip(uniq, first, navail):
            need = counts[u] - filled[u]
            if need <= 0:
                continue
            t = min(int(need), int(k))
            dst = starts[u] + filled[u]
            ra_out[dst:dst + t] = ra_s[f0:f0 + t]
            dec_out[dst:dst + t] = dec_s[f0:f0 + t]
            filled[u] += t
            n_done += t
    return ra_out, dec_out


def _radial_truth_logq(rng, npix, chi_centers, *, sigma, corr_mpc):
    """Independent per-pixel radial fields: squared-exponential covariance on
    the uniform-χ bin centers, mean-one lognormal shift ``−σ²/2``."""
    d = chi_centers[:, None] - chi_centers[None, :]
    cov = sigma**2 * np.exp(-0.5 * (d / corr_mpc) ** 2)
    cov[np.diag_indices_from(cov)] += 1e-8 * sigma**2
    L = np.linalg.cholesky(cov)
    s = rng.standard_normal((npix, chi_centers.size)) @ L.T
    return s - 0.5 * sigma**2


def main(argv=None):
    import healpy as hp

    args = _parse_args(argv)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cosmo = fiducial_cosmo()
    grids = _dark._cosmology_grids(
        _dark._build_cosmology(cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa), args.zmax)
    zg, dc, dvc_dz = grids["z"], grids["dc"], grids["dvc_dz"]
    Vcum = np.concatenate([[0.0], np.cumsum(0.5 * (dvc_dz[1:] + dvc_dz[:-1]) * np.diff(zg))])
    V_tot = float(Vcum[-1])

    n0 = 10.0 ** args.log10n0 if args.log10n0 is not None else args.n_target / V_tot
    log10n0 = float(np.log10(n0))

    npix = hp.nside2npix(args.nside)
    nbin = args.n_zbins
    chi_edges = np.linspace(0.0, float(dc[-1]), nbin + 1)
    z_edges = np.interp(chi_edges, dc, zg)
    chi_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    z_centers = np.interp(chi_centers, dc, zg)
    dV_bin = np.diff(np.interp(z_edges, zg, Vcum))          # (nbin,) full-sky shell volumes

    # ---- truth field -------------------------------------------------------
    # ls_z: same Mpc→zeta map as the builder's _count_weighted_zref_lsz, at a
    # volume-weighted reference redshift (the builder uses count-weighted; the
    # small difference is deliberate — the fitter re-derives its own ls_z).
    z_ref = float(np.trapz(zg * dvc_dz, zg) / np.trapz(dvc_dz, zg))
    dchi_dz = float(np.interp(z_ref, zg, np.gradient(dc, zg)))
    ls_z = args.corr_mpc / max((1.0 + z_ref) * dchi_dz, 1e-6)

    n_hat = np.asarray(hp.pix2vec(args.nside, np.arange(npix), nest=False), dtype=float).T
    zgrid_np = np.asarray(zgrid, dtype=float)

    if args.truth_kernel == "gp3d":
        from darksirens.redshift.lognormal_completion import lowrank_inducing_nodes
        Zn, Zz = lowrank_inducing_nodes(32, 6, 3.0)
        xi_true = rng.standard_normal(int(np.asarray(Zn).shape[0]))
        kw = dict(amp=args.lss_sigma, ls_sph=args.corr_ang, ls_z=ls_z, bias=1.0)
        logq_vox = eval_truth_field_logq(xi_true, Zn, Zz, n_hat, z_centers, **kw)
        logq_on_zgrid = eval_truth_field_logq(xi_true, Zn, Zz, n_hat, zgrid_np, **kw)
    else:
        xi_true = np.zeros(0)
        logq_vox = _radial_truth_logq(rng, npix, chi_centers,
                                      sigma=args.lss_sigma, corr_mpc=args.corr_mpc)
        # zgrid export: interpolate the per-pixel field in chi (flat beyond zmax).
        chi_of_zgrid = np.interp(zgrid_np, zg, dc, left=0.0, right=float(dc[-1]))
        logq_on_zgrid = np.empty((npix, zgrid_np.size))
        for p in range(npix):
            logq_on_zgrid[p] = np.interp(chi_of_zgrid, chi_centers, logq_vox[p])

    Q_vox = np.exp(logq_vox)
    print(f"[truth] kernel={args.truth_kernel}  mean(Q)={Q_vox.mean():.4f}  "
          f"std(logQ)={logq_vox.std():.4f}  z_ref={z_ref:.4f}  ls_z(zeta)={ls_z:.4f}")

    # ---- Poisson galaxy draw ----------------------------------------------
    mu_vox = n0 * (dV_bin[None, :] / npix) * Q_vox           # (npix, nbin)
    N_vox = rng.poisson(mu_vox)
    total = int(N_vox.sum())
    print(f"[draw] n0={n0:.4e} Mpc^-3 (log10n0={log10n0:.4f})  "
          f"E[N]={mu_vox.sum():.0f}  N={total}")

    bin_flat = np.concatenate([np.repeat(np.arange(nbin), N_vox[p]) for p in range(npix)])
    # z within each bin: uniform in comoving volume (inverse CDF restricted to bin).
    cdf = Vcum / V_tot
    cdf_edges = np.interp(z_edges, zg, cdf)
    u = rng.uniform(cdf_edges[bin_flat], cdf_edges[bin_flat + 1])
    z_true = np.interp(u, cdf, zg)
    # pix_flat is already pixel-ordered, matching the sky sampler's grouping.
    ra, dec = _sample_sky_in_pixels(rng, N_vox.sum(axis=1), args.nside)

    # ---- per-galaxy physics: magnitudes, photo-z (mock-generator verbatim) --
    survey_cfg = _dark.SurveyConfig(
        footprint_dec_min_deg=-40.0 if args.footprint else -90.0,
        footprint_dec_max_deg=80.0 if args.footprint else 90.0,
        z_hard_max=args.zmax, magnitude_limit=args.mag_limit,
        z50=args.z50, width=args.width,
    )
    abs_mag = rng.normal(survey_cfg.absolute_mag_mean, survey_cfg.absolute_mag_sigma, total)
    dl_pc = _dark._interp_dl(z_true, grids) * 1.0e6
    app_mag = abs_mag + 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)
    zerr = _dark._catalog_zerr(z_true, survey_cfg)
    z_obs = z_true + zerr * rng.standard_normal(total)

    catalog = {"ra": ra, "dec": dec, "z": z_true, "z_obs": z_obs,
               "abs_mag": abs_mag, "app_mag": app_mag}
    if args.aniso_dmag != 0.0:
        # Hemisphere-dependent depth: replicate _apply_survey_selection with a
        # PER-GALAXY magnitude limit (same rng draw count as the isotropic
        # path: one uniform per galaxy for the logistic completeness).
        m_lim_gal = np.where(dec >= 0.0,
                             args.mag_limit + args.aniso_dmag,
                             args.mag_limit - args.aniso_dmag)
        depth = (z_true <= survey_cfg.z_hard_max) & (app_mag <= m_lim_gal)
        comp = 1.0 / (1.0 + np.exp(-(survey_cfg.z50 - z_true) / survey_cfg.width))
        observed = depth & (rng.uniform(size=total) < comp)
        catalog["m_lim_gal"] = m_lim_gal
        catalog["stratum"] = (dec >= 0.0).astype(np.int8)
    else:
        observed = _dark._apply_survey_selection(rng, catalog, survey_cfg)
    n_obs = int(observed.sum())
    # Analytic expectation of the observed fraction (footprint aside).
    from scipy.special import expit
    from scipy.stats import norm
    dm = 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)
    c_true_gal = (norm.cdf((survey_cfg.magnitude_limit - survey_cfg.absolute_mag_mean - dm)
                           / survey_cfg.absolute_mag_sigma)
                  * expit((survey_cfg.z50 - z_true) / survey_cfg.width))
    print(f"[select] observed {n_obs}/{total} = {n_obs / total:.4f}  "
          f"(analytic selection mean {c_true_gal.mean():.4f})")

    # ---- pixelated observed catalog ---------------------------------------
    # Survey convention (mock generator): recorded z is z_obs, recorded error
    # is the model width AT the recorded value.
    z_obs_s = z_obs[observed]
    zerr_s = _dark._catalog_zerr(z_obs_s, survey_cfg)
    pixelated = _dark._pixelate_catalog(
        ra[observed], dec[observed], z_obs_s, zerr_s,
        np.ones(n_obs), args.nside,
        marks={"gal_app_mag": app_mag[observed]})

    meta = dict(
        aniso_dmag=args.aniso_dmag,
        seed=args.seed, nside=args.nside, zmax=args.zmax, n0=n0, log10n0=log10n0,
        truth_kernel=args.truth_kernel, lss_sigma=args.lss_sigma,
        corr_ang=args.corr_ang, corr_mpc=args.corr_mpc, ls_z=ls_z, z_ref=z_ref,
        n_zbins=nbin, z50=args.z50, width=args.width, mag_limit=args.mag_limit,
        abs_mag_mean=survey_cfg.absolute_mag_mean,
        abs_mag_sigma=survey_cfg.absolute_mag_sigma,
        footprint=bool(args.footprint), n_total=total, n_observed=n_obs,
    )

    pixel_path = out / f"catalog_pixelated_nside_{args.nside}.h5"
    with h5py.File(pixel_path, "w") as f:
        f.attrs["nside"] = int(args.nside)
        f.attrs["z_depth"] = float(args.zmax)
        f.attrs["mock_data"] = True
        f.attrs["metadata_json"] = json.dumps(meta)
        for key, val in pixelated.items():
            if key != "ngals":
                assert val.dtype == np.float64, (
                    f"pixelated {key!r} is {val.dtype}, not float64 "
                    "(see _pixelate_catalog: float32 padding NaN-poisons the KDE)")
            f.create_dataset(key, data=val, compression="gzip", shuffle=True)

    truth_path = out / "truth.h5"
    with h5py.File(truth_path, "w") as f:
        for k, v in meta.items():
            f.attrs[k] = v
        f.create_dataset("xi_true", data=xi_true)
        f.create_dataset("logq_truth_vox", data=logq_vox, compression="gzip")
        f.create_dataset("logq_truth_on_zgrid", data=logq_on_zgrid, compression="gzip")
        f.create_dataset("z_bin_edges", data=z_edges)
        f.create_dataset("z_bin_centers", data=z_centers)
        f.create_dataset("chi_bin_centers", data=chi_centers)
        f.create_dataset("dV_bin", data=dV_bin)
        f.create_dataset("cosmo_z", data=zg)
        f.create_dataset("cosmo_dc", data=dc)
        f.create_dataset("cosmo_dl", data=grids["dl"])
        f.create_dataset("cosmo_dvc_dz", data=dvc_dz)
        g = f.create_group("catalog")
        for k in ("ra", "dec", "z", "z_obs", "abs_mag", "app_mag"):
            g.create_dataset(k, data=catalog[k], compression="gzip", shuffle=True)
        for k in ("m_lim_gal", "stratum"):
            if k in catalog:
                g.create_dataset(k, data=catalog[k], compression="gzip",
                                 shuffle=True)
        g.create_dataset("observed", data=observed)

    print(f"[write] {pixel_path}")
    print(f"[write] {truth_path}")
    print(f"[fit]   use --log10n0 {log10n0:.6f} (injected; anything else leaks "
          "into Q as spurious redshift structure)")


if __name__ == "__main__":
    main()
