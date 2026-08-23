"""Stage A diagnostics of the real-DESI selection channel (numbers first).

Writes data/diagnostics_stage_a.json with:
  1. counts-Cbar alarm: the counts-based aggregate completeness estimate
     (dN_obs/dz) / (n0_true * f_sky * dV/dz) overlaid on the fitted C_sel(z)
     -- under clustering-free truth these agree; systematic departures are
     the misspecification alarm (mock precedent: counts-Cbar injects a
     spurious logQ dipole).
  2. dN/dz closure: observed vs n0 * C_sel * dV per z-bin.
  3. Stratified fits: north/south (LS DR9 vs DR10 imaging) and hemisphere
     splits refit with the same K(z) template; pooled-vs-stratified
     Delta-M0hat significances (mock precedent: pooled misspecified -43 sigma).
  4. Reference completeness comparison vs the Uchuu-derived count numbers
     (weighted_completeness.json: spec ~26%, phot ~61% by count).
  5. Off-footprint audit: per-event PE probability mass outside the occupied
     n64 footprint for gwsamples_44 (the c_mode=selection Q->0 risk).
"""

from __future__ import annotations

import json

import h5py
import healpy as hp
import numpy as np

import common as C  # noqa: F401  (pins DARKSIRENS_ZMAX)

from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.redshift.selection import (  # noqa: E402
    c_sel_gaussian,
    fit_selection_from_mags,
    load_selection_fit_json,
)
from darksirens.utils.cosmology import H0Planck, Om0Planck, dV_of_z  # noqa: E402


def _binned_completeness(z, w, n0, f_sky, z_depth, n_bins=40, z_lo=0.02):
    edges = np.linspace(z_lo, z_depth, n_bins + 1)
    hist, _ = np.histogram(z, bins=edges, weights=w)
    zc = 0.5 * (edges[:-1] + edges[1:])
    dz = np.diff(edges)
    zf = np.asarray(zgrid)
    dv = 4.0 * np.pi * np.asarray(dV_of_z(zf, H0Planck, Om0Planck))
    dN_exp = n0 * f_sky * np.interp(zc, zf, dv)          # per unit z
    return zc, hist / dz, dN_exp


def _fit_subset(m, z, m_lim, coeffs, label):
    fit = fit_selection_from_mags(m, z, m_lim, k_corr_coeffs=coeffs,
                                  stratum=label)
    sd = np.sqrt(np.diag(fit.cov))
    return {"stratum": label, "n_gal": fit.n_gal,
            "M0hat": fit.M0hat, "M0hat_sd": float(sd[0]),
            "sigma_M": fit.sigma_M, "sigma_M_sd": float(sd[1])}


def main() -> None:
    sel = load_selection_fit_json(C.DATA_DIR / "selection_fit_union.json")
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    coeffs = tuple(sel["k_corr_coeffs"])
    out = {"selection_fit": {k: sel[k] for k in
                             ("m_lim", "M0hat", "sigma_M", "k_corr_coeffs")},
           "n0_calibration": {k: cal[k] for k in
                              ("log10n0", "n0_true_Mpc3", "delta",
                               "f_sky_occupied", "C_eff_volume_weighted")}}

    # --- raw per-galaxy arrays (strata live here) -------------------------
    with h5py.File(C.DATA_DIR / "desi_union_raw.h5", "r") as f:
        z = f["Z"][...]
        m = f["APP_MAG"][...]
        w = f["WEIGHT"][...]
        stratum = f["STRATUM"][...]
        code = f["SURVEY_CODE"][...]
        dec = f["TARGET_DEC"][...]

    n0 = cal["n0_true_Mpc3"]
    f_sky = cal["f_sky_occupied"]
    z_depth = float(C.Z_DEPTH)

    # 1+2: counts-Cbar alarm + dN/dz closure ------------------------------
    zc, dN_obs, dN_exp = _binned_completeness(z, w, n0, f_sky, z_depth)
    zf = np.asarray(zgrid)
    csel_c = np.asarray(c_sel_gaussian(
        zf, sel["m_lim"], sel["M0hat"], sel["sigma_M"], H0Planck,
        k_corr_coeffs=coeffs))
    csel_at = np.interp(zc, zf, csel_c)
    counts_cbar = dN_obs / dN_exp
    closure_resid = dN_obs / (dN_exp * np.clip(csel_at, 1e-6, None)) - 1.0
    out["counts_cbar_vs_csel"] = {
        "z": zc.tolist(),
        "counts_cbar": counts_cbar.tolist(),
        "c_sel": csel_at.tolist(),
        "closure_resid_frac": closure_resid.tolist(),
        "max_abs_closure_resid_z_le_025":
            float(np.abs(closure_resid[zc <= 0.25]).max()),
    }

    # 3: stratified fits ---------------------------------------------------
    m_lim = float(sel["m_lim"])
    fits = {"pooled": {"M0hat": sel["M0hat"],
                       "M0hat_sd": float(np.sqrt(np.asarray(sel["cov"])[0, 0])),
                       "sigma_M": sel["sigma_M"], "n_gal": int(sel["n_gal"])}}
    for label, mask in (("south", stratum == 0), ("north", stratum == 1),
                        ("dec_neg", dec < 0.0), ("dec_pos", dec >= 0.0),
                        ("spec_only", code != 1), ("phot_only", code == 1)):
        fits[label] = _fit_subset(m[mask], z[mask], m_lim, coeffs, label)
    dm = fits["north"]["M0hat"] - fits["south"]["M0hat"]
    sdm = float(np.hypot(fits["north"]["M0hat_sd"], fits["south"]["M0hat_sd"]))
    fits["delta_M0hat_north_minus_south"] = {"value": dm, "sd": sdm,
                                             "significance": dm / sdm}
    out["stratified_fits"] = fits

    # 4: reference completeness comparison --------------------------------
    ref_p = (C.DESI_REPO / "final/experiments/experiment_uchuu_desi_mock/"
             "results/weighted_completeness.json")
    if ref_p.exists():
        out["uchuu_reference_counts"] = json.load(open(ref_p))
    csel_mean = cal["C_eff_volume_weighted"]
    out["model_volume_weighted_completeness"] = csel_mean

    # 5: off-footprint audit ----------------------------------------------
    with h5py.File(C.DATA_DIR / "pixelated_n64" / "catalog_pixelated_nside_64.h5",
                   "r") as f:
        occupied = f["ngals"][...] > 0
        nside = int(f.attrs["nside"])
    with h5py.File(C.GW_SAMPLES_44, "r") as f:
        ra = f["ra"][...].reshape(44, -1)
        dc = f["dec"][...].reshape(44, -1)
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f["event_names"][...]]
    pix = hp.ang2pix(nside, np.pi / 2.0 - dc, ra)
    off_frac = (~occupied[pix]).mean(axis=1)
    order = np.argsort(off_frac)[::-1]
    out["off_footprint_pe_mass"] = {
        "per_event": {names[i]: float(off_frac[i]) for i in order},
        "n_events_over_50pct": int((off_frac > 0.5).sum()),
        "n_events_over_20pct": int((off_frac > 0.2).sum()),
    }

    outp = C.DATA_DIR / "diagnostics_stage_a.json"
    with open(outp, "w") as f:
        json.dump(out, f, indent=1)
    C.write_provenance(outp, {"inputs": ["desi_union_raw.h5",
                                         "selection_fit_union.json",
                                         "n0_calibration.json",
                                         str(C.GW_SAMPLES_44)]})
    print(json.dumps({k: out[k] for k in
                      ("stratified_fits", "off_footprint_pe_mass")},
                     indent=1)[:2000])
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
