"""Fit the parametric magnitude selection from a pixelated survey's magnitudes.

Reads the padded ``gal_app_mag`` dataset (written by ``darksirens_pixelate``
or the mock generator), fits the truncated-luminosity-function selection per
stratum (``darksirens.redshift.selection.fit_selection_from_mags``), and
writes a selection JSON consumed by

* ``darksirens_build_lognormal_completion --c-mode selection --selection-fit``
  (the Q-table base is ``C_sel(z; theta_hat) dN_exp``), and
* ``darksirens_inference --selection_fit`` (theta_hat and the marginal
  Laplace sds become the Gaussian prior on the SAMPLED ``M0hat``/``sigma_M``).

The fit works in reference absolute magnitudes ``m - DM(z; H0=100)``, so it
is exactly independent of the true H0 (h-scaled convention) and of the galaxy
density field (thinning): clustering cannot leak into the fitted selection.
The recorded (photometric) redshifts are used as-is; their scatter inflates
``sigma_M`` slightly and is a documented limitation, not corrected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from darksirens.catalogs.io import load_survey, load_survey_galprops
from darksirens.cli.common import _banner, _end, _fatal, _ok, _row, _section
from darksirens.redshift.selection import fit_selection_from_mags


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--survey_path", required=True,
                   help="Pixelated survey HDF5 carrying gal_app_mag.")
    p.add_argument("--m_lim", type=float, required=True,
                   help="The survey's hard apparent-magnitude limit (a "
                        "truncation DATUM of the selection protocol, not a "
                        "fitted parameter: only m_lim - M0hat is identified).")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: selection_fit.json next "
                        "to the survey file).")
    p.add_argument("--family", default="gaussian", choices=["gaussian"],
                   help="Luminosity-function family (schechter ships with "
                        "real-catalog ingestion).")
    p.add_argument("--k_corr_coeffs", default=None,
                   help="Comma-separated polynomial coefficients c1[,c2,...] "
                        "of a fixed K-correction template K(z) = sum_j c_j "
                        "z**j applied to the OBSERVED magnitudes (no constant "
                        "term: c0 is exactly degenerate with M0hat). Default: "
                        "no K-correction.")
    opts = p.parse_args(argv)

    k_corr_coeffs = None
    if opts.k_corr_coeffs:
        try:
            k_corr_coeffs = tuple(
                float(c) for c in opts.k_corr_coeffs.split(",") if c.strip())
        except ValueError:
            _fatal(f"--k_corr_coeffs is not a comma-separated float list: "
                   f"{opts.k_corr_coeffs!r}")

    survey = Path(opts.survey_path)
    out = Path(opts.out) if opts.out else survey.parent / "selection_fit.json"

    print()
    _banner("DARK SIRENS SELECTION FIT")
    print()
    _section("Loading survey magnitudes")
    _row("Input", survey)
    if not survey.is_file():
        _fatal(f"survey file not found: {survey}")
    nside, ngals, zgals, dzgals, wgals, z_depth = load_survey(survey)
    props = load_survey_galprops(survey)
    if "gal_app_mag" not in props:
        _fatal("survey carries no gal_app_mag dataset; re-pixelate from a "
               "raw catalog with an APP_MAG column (darksirens_pixelate), or "
               "regenerate the mock (the generator writes it since the "
               "selection-channel plumbing).")
    zg = np.asarray(zgals)
    ng = np.asarray(ngals)
    mag = np.asarray(props["gal_app_mag"])
    real = np.arange(zg.shape[1])[None, :] < ng[:, None]
    m = mag[real]
    z = zg[real]
    _ok(f"Galaxies: {m.size:,} in {int((ng > 0).sum()):,} occupied pixels")
    _end()

    _section("Fitting truncated luminosity function")
    # Single stratum for now; the strata seam is the per-galaxy mask above.
    fit = fit_selection_from_mags(m, z, opts.m_lim, family=opts.family,
                                  k_corr_coeffs=k_corr_coeffs)
    sd = np.sqrt(np.diag(fit.cov))
    _row("family", fit.family)
    _row("m_lim (fixed datum)", f"{fit.m_lim:.4f}")
    if k_corr_coeffs:
        _row("K(z) coeffs", ", ".join(f"{c:.5g}" for c in k_corr_coeffs))
    _row("M0hat", f"{fit.M0hat:.5f} +/- {sd[0]:.5f}  (h-scaled: M0 - 5 log10 h)")
    _row("sigma_M", f"{fit.sigma_M:.5f} +/- {sd[1]:.5f}")
    _end()

    payload = {
        "format_version": "darksirens-selection-fit-1.0",
        "strata": [fit.to_jsonable()],
        "survey_path": str(survey),
        "survey_sha256": hashlib.sha256(survey.read_bytes()).hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    _section("Output")
    _ok(f"selection fit  →  {out}")
    _end()
    print()
    _banner("DONE")
    print()
    return 0


if __name__ == "__main__":
    main()
