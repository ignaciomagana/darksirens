"""Calibrate the completeness-budget fiducials n0 (and delta) for the builder.

n0 = sum(w) / (f_sky_occ * C_eff * V_c(z <= z_depth)) with the selection
correction C_eff = int C_sel dV / int dV: the builder's budget compares
N_obs against n0 * C_sel * dN_exp, so n0 must be the TRUE (selection-
corrected) mean density, not the raw observed one -- using the raw density
would double-count the incompleteness.

delta is fit from the observed dN/dz shape against
C_sel(z) * (1+z)^delta * dVc/dz over [Z_FIT_LO, z_depth].

A mis-set n0 is absorbed into Q as spurious redshift structure; this script
is the single source of the builder's --log10n0/--delta.
"""

from __future__ import annotations

import json

import h5py
import numpy as np

import common as C  # noqa: F401  (pins the FULL-RANGE ZMAX=6.0 grid)

from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.redshift.selection import c_sel_gaussian, load_selection_fit_json  # noqa: E402
from darksirens.utils.cosmology import H0Planck, Om0Planck, dV_of_z  # noqa: E402

Z_FIT_LO = 0.05


def main() -> None:
    surv = C.SURVEY_N64
    fitp = C.FIT_JSON
    sel = load_selection_fit_json(fitp)

    with h5py.File(surv, "r") as f:
        nside = int(f.attrs["nside"])
        z_depth = float(f.attrs["z_depth"])
        ngals = f["ngals"][...]
        zg = f["zgals"][...]
        wg = f["wgals"][...]
    real = np.arange(zg.shape[1])[None, :] < ngals[:, None]
    z = zg[real]
    w = wg[real]
    n_occ = int((ngals > 0).sum())
    npix_tot = 12 * nside * nside
    f_sky = n_occ / npix_tot

    zf = np.asarray(zgrid)
    zmask = zf <= z_depth
    # dV_of_z is per steradian; the full-sky 4*pi cancels against f_sky's
    # occupied-fraction normalization ONLY if included -- keep it explicit.
    dv = 4.0 * np.pi * np.asarray(dV_of_z(zf, H0Planck, Om0Planck))
    csel = np.asarray(c_sel_gaussian(
        zf, sel["m_lim"], sel["M0hat"], sel["sigma_M"], H0Planck,
        k_corr_coeffs=sel["k_corr_coeffs"]))
    v_c = np.trapz(dv[zmask], zf[zmask])
    c_eff = np.trapz((csel * dv)[zmask], zf[zmask]) / v_c

    n0_raw = float(w.sum() / (f_sky * v_c))

    # delta FIRST: count-weighted fit of the log dN/dz shape.  The fit is
    # normalization-free (const absorbs it), so it does not depend on n0.
    edges = np.linspace(Z_FIT_LO, z_depth, 40)
    hist, _ = np.histogram(z, bins=edges, weights=w)
    zc = 0.5 * (edges[:-1] + edges[1:])
    model0 = (np.interp(zc, zf, csel) * np.interp(zc, zf, dv))
    keep = (hist > 100) & (model0 > 0)
    # log(dN/dz / (C_sel dVc/dz)) = const + delta log(1+z)
    y = np.log(hist[keep] / model0[keep])
    x = np.log1p(zc[keep])
    A = np.stack([np.ones_like(x), x], axis=1)
    wls = np.sqrt(hist[keep])
    (const, delta), *_ = np.linalg.lstsq(A * wls[:, None], y * wls, rcond=None)

    # n0 with the SAME budget the likelihood integrates: the likelihood's
    # dN_exp carries n0 * C_sel * (1+z)^delta * dV, so n0 must normalize the
    # counts against that FULL integrand.  The previous convention dropped
    # the (1+z)^delta factor here while the likelihood kept it, overpricing
    # the budget by mean[(1+z)^delta] ~ 17% (found by the dN/dz
    # posterior-predictive gate, 2026-08-09); n0_true_Mpc3_no_evo records
    # the superseded value -- do not quote it.
    evo = (1.0 + zf) ** float(delta)
    budget_int = np.trapz((csel * evo * dv)[zmask], zf[zmask])
    n0 = float(w.sum() / (f_sky * budget_int))
    n0_no_evo = float(w.sum() / (f_sky * c_eff * v_c))

    out = C.DATA_DIR / "n0_calibration.json"
    payload = {
        "survey": str(surv),
        "selection_fit": str(fitp),
        "nside": nside, "n_occupied": n_occ, "f_sky_occupied": f_sky,
        "z_depth": z_depth,
        "sum_weights": float(w.sum()),
        "V_c_Mpc3": float(v_c),
        "C_eff_volume_weighted": float(c_eff),
        "n0_raw_observed_Mpc3": n0_raw,
        "n0_true_Mpc3": n0,
        "log10n0": float(np.log10(n0)),
        "n0_true_Mpc3_no_evo": n0_no_evo,
        "log10n0_no_evo": float(np.log10(n0_no_evo)),
        "delta": float(delta),
        "delta_fit_z_range": [Z_FIT_LO, z_depth],
        "note": "n0 normalizes the counts against the FULL likelihood "
                "budget integrand C_sel * (1+z)^delta * dVc/dz (convention "
                "fixed 2026-08-09; *_no_evo are the superseded pre-fix "
                "values -- do not quote); delta from the count-weighted "
                "dN/dz shape residual to C_sel * dVc/dz",
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=1)
    C.write_provenance(out, payload)
    print(f"f_sky_occ = {f_sky:.4f}  C_eff = {c_eff:.4f}  "
          f"n0_raw = {n0_raw:.4e}  n0_true = {n0:.4e}  "
          f"log10n0 = {payload['log10n0']:.4f}  delta = {delta:+.3f}")


if __name__ == "__main__":
    main()
