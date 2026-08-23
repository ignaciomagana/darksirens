"""Fit the polynomial K(z) template for the r-band selection model.

The union catalog itself encodes the K-correction each galaxy was actually
corrected with upstream: M_APP - MAG_R - DM(z; 67.74/0.3075) = K_eff per
galaxy (Chilingarian 2010 at the galaxy's own g-r, median-color fallback).
The template is a no-constant-term cubic fit to the count-weighted MEDIAN
K_eff per z-bin -- with it, the selection fit's Mhat = m - DM - K(z)
reproduces the catalog's own rest-frame M_r on average, and the per-galaxy
color scatter around the median is absorbed into sigma_M (documented).

The fixed-median-color Chilingarian curve (K(0.30) = 0.047) is kept in the
JSON as a cross-check; it under-corrects at high z because the OBSERVED
median color reddens with z (the depth-scan's "data-implied K ~ 0.39z").
"""

from __future__ import annotations

import importlib.util
import json

import h5py
import numpy as np

import common as C

Z_LO, Z_HI, N_BINS = 0.02, 0.30, 56


def main() -> None:
    with h5py.File(C.DATA_DIR / "desi_union_raw.h5", "r") as f:
        z = f["Z"][...]
        keff = f["APP_MAG"][...] - f["MAG_R"][...]
    # DM at the upstream build cosmology (the one MAG_R was derived with).
    from astropy.cosmology import FlatLambdaCDM

    cosmo = FlatLambdaCDM(H0=67.74, Om0=0.3075)
    keff -= cosmo.distmod(np.clip(z, 1e-4, None)).value

    edges = np.linspace(Z_LO, Z_HI, N_BINS + 1)
    ib = np.digitize(z, edges) - 1
    ok = (ib >= 0) & (ib < N_BINS)
    zc, med, n = [], [], []
    for b in range(N_BINS):
        sel = ok & (ib == b)
        if sel.sum() < 50:
            continue
        zc.append(0.5 * (edges[b] + edges[b + 1]))
        med.append(np.median(keff[sel]))
        n.append(sel.sum())
    zc, med, n = np.array(zc), np.array(med), np.array(n, dtype=float)

    # Count-weighted least squares in (z, z^2, z^3): no constant term (c0 is
    # exactly degenerate with M0hat in the selection model).
    A = np.stack([zc, zc**2, zc**3], axis=1) * np.sqrt(n)[:, None]
    coeffs, *_ = np.linalg.lstsq(A, med * np.sqrt(n), rcond=None)
    fitcurve = np.stack([zc, zc**2, zc**3], axis=1) @ coeffs
    resid = med - fitcurve

    # Cross-check: Chilingarian at the fixed median color.
    spec = importlib.util.spec_from_file_location("calc_kcor", C.KCORR_MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    k_chil_030 = float(mod.calc_kcor("r", 0.30, "g - r", C.MEDIAN_GR))

    out = C.DATA_DIR / "kcorr_poly.json"
    payload = {
        "band": "r",
        "source": "data-implied median K_eff = M_APP - MAG_R - DM(z; 67.74/0.3075)",
        "z_fit_range": [Z_LO, Z_HI],
        "n_bins_used": int(zc.size),
        "basis": "z, z^2, z^3 (no constant term: c0 degenerate with M0hat)",
        "k_corr_coeffs": [float(c) for c in coeffs],
        "max_abs_resid_mag": float(np.abs(resid).max()),
        "rms_resid_mag": float(np.sqrt(np.mean(resid**2))),
        "K_at_z030": float(np.polyval(np.r_[coeffs[::-1], 0.0][::-1], 0.30)
                           if False else coeffs @ np.array([0.30, 0.09, 0.027])),
        "chilingarian_fixed_color_K_at_z030": k_chil_030,
        "median_binned_curve": {"z": zc.tolist(), "K_median": med.tolist(),
                                "n": n.tolist()},
        "note": "per-galaxy color scatter around the median K is absorbed "
                "into the fitted sigma_M (documented limitation)",
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    C.write_provenance(out, payload)
    print(f"K(z) ~ {coeffs[0]:+.5f} z {coeffs[1]:+.5f} z^2 {coeffs[2]:+.5f} z^3"
          f"  (rms resid {payload['rms_resid_mag']:.4f} mag, "
          f"K(0.30) = {payload['K_at_z030']:.4f}, "
          f"Chilingarian fixed-color {k_chil_030:.4f})")


if __name__ == "__main__":
    main()
