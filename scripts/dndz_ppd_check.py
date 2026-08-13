#!/usr/bin/env python
"""dN/dz posterior-predictive check of the selection-mode budget (PR-3 CLI).

Compares the OBSERVED per-shell galaxy totals ``T_g`` of a pixelated catalog
against the selection-mode prediction

    T_g^pred = sum_p f_p * int_{shell g} C_sel(z; theta_hat) n0 (1+z)^delta
               (apix dV_c/dz) dz

and reports per-shell Poisson z-scores plus the total budget ratio — the
"dN/dz PPD gate" queued in selection-channel-followups, made standalone.
This is a DIAGNOSTIC of the (C_sel, n0, delta) budget shape against the
counts (OWNER DECISION 13(a): it never enters the headline posterior; its
shape residual is exactly what the latent count channel is allowed to
absorb as placement, so large per-shell z-scores here with a good total are
the expected signature of clustering, while a bad TOTAL means the budget
calibration and the catalog disagree).

Usage (from a directory whose common.py pins DARKSIRENS_ZMAX, or with the
env var set):
    python scripts/dndz_ppd_check.py --survey catalog_pixelated_nside_64.h5 \
        --selection-fit selection_fit_union.json \
        --n0-calibration n0_calibration.json \
        [--z-depth 0.3] [--n-shells 12] [--h0 67.74] [--om0 0.3089] \
        [--per-pixel-completeness mth_map.h5]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

CLIGHT = 299792.458


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--survey", required=True)
    ap.add_argument("--selection-fit", required=True)
    ap.add_argument("--n0-calibration", required=True)
    ap.add_argument("--z-depth", type=float, default=0.3)
    ap.add_argument("--n-shells", type=int, default=12)
    ap.add_argument("--h0", type=float, default=67.74)
    ap.add_argument("--om0", type=float, default=0.3089)
    ap.add_argument("--per-pixel-completeness", default=None)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args(argv)

    import h5py
    import healpy as hp
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from darksirens.redshift.selection import (
        c_sel_gaussian, load_selection_fit_json)

    sel = load_selection_fit_json(args.selection_fit)
    cal = json.load(open(args.n0_calibration))
    n0 = 10.0 ** float(cal["log10n0"])
    delta = float(cal["delta"])

    with h5py.File(args.survey) as f:
        zg, ng = f["zgals"][...], f["ngals"][...]
        nside = int(f.attrs.get("nside", hp.npix2nside(zg.shape[0])))
    npix = 12 * nside ** 2
    apix = hp.nside2pixarea(nside)
    occ = np.where(ng > 0)[0]

    if args.per_pixel_completeness:
        from darksirens.catalogs.depth_map import load_selection_fraction
        f_p = load_selection_fraction(
            args.per_pixel_completeness, nside).f_p[occ]
    else:
        f_p = np.ones(occ.size)

    edges = np.linspace(0.0, args.z_depth, args.n_shells + 1)
    T_obs = np.zeros(args.n_shells)
    for p in occ:
        n = int(ng[p])
        if n:
            h, _ = np.histogram(zg[p, :n], bins=edges)
            T_obs += h

    zf = np.linspace(1e-4, args.z_depth, 600)
    E = np.sqrt(args.om0 * (1 + zf) ** 3 + (1 - args.om0))
    dc = np.concatenate([[0.0], np.cumsum(
        0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(zf))]) * CLIGHT / args.h0
    dVdz = 4 * np.pi * (CLIGHT / args.h0) * dc ** 2 / E
    Cz = np.asarray(c_sel_gaussian(
        jnp.asarray(zf), float(sel["m_lim"]), float(sel["M0hat"]),
        float(sel["sigma_M"]), args.h0, args.om0,
        k_corr_coeffs=tuple(sel["k_corr_coeffs"])))
    dens = Cz * n0 * (1 + zf) ** delta * dVdz * (apix / (4 * np.pi))

    fsum = float(f_p.sum())
    T_pred = np.array([
        fsum * np.trapz(np.where((zf >= lo) & (zf < hi), dens, 0.0), zf)
        for lo, hi in zip(edges[:-1], edges[1:])])

    zsc = (T_obs - T_pred) / np.sqrt(np.maximum(T_pred, 1.0))
    ratio = T_obs.sum() / T_pred.sum()
    print(f"{'shell':>5} {'z_lo':>6} {'z_hi':>6} {'observed':>12} "
          f"{'predicted':>12} {'z-score':>9}")
    for g in range(args.n_shells):
        print(f"{g:5d} {edges[g]:6.3f} {edges[g + 1]:6.3f} "
              f"{T_obs[g]:12.0f} {T_pred[g]:12.1f} {zsc[g]:9.2f}")
    print(f"\ntotal observed / predicted = {ratio:.4f}  "
          f"(the BUDGET check; the per-shell z-scores are shape+clustering)")
    print(f"max |z| = {np.abs(zsc).max():.1f}   chi2/dof = "
          f"{(zsc ** 2).mean():.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(dict(edges=edges.tolist(), T_obs=T_obs.tolist(),
                           T_pred=T_pred.tolist(), z_scores=zsc.tolist(),
                           total_ratio=float(ratio)), f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
