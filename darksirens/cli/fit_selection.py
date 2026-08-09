"""Fit the parametric magnitude selection from a pixelated survey's magnitudes.

Reads the padded ``gal_app_mag`` dataset (written by ``darksirens_pixelate``
or the mock generator), fits the truncated-luminosity-function selection per
stratum (``darksirens.redshift.selection.fit_selection_from_mags``), and
writes a selection JSON consumed by

* ``darksirens_build_lognormal_completion --c-mode selection --selection-fit``
  (the Q-table base is ``C_sel(z; theta_hat) dN_exp``), and
* ``darksirens_inference --selection_fit`` (theta_hat and the marginal
  Laplace sds become the Gaussian prior on the SAMPLED theta -- ``M0hat``/
  ``sigma_M`` for the gaussian family, ``Mstar_hat``/``alpha`` for the
  schechter one, which additionally pins the ``M_faint_offset`` protocol
  constant).

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
from darksirens.redshift.selection import (
    M_FAINT_OFFSET_DEFAULT,
    SELECTION_FAMILIES,
    SELECTION_SAMPLED_FIELDS,
    fit_selection_from_mags,
)


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
    p.add_argument("--family", default="gaussian",
                   choices=list(SELECTION_FAMILIES),
                   help="Luminosity-function family of the selection model: "
                        "'schechter' is the real-catalog LF (samples "
                        "Mstar_hat, alpha; no K(z) template, single stratum); "
                        "'gaussian' is the mock program's truncated-normal "
                        "magnitude model (samples M0hat, sigma_M).")
    p.add_argument("--m_faint_offset", type=float,
                   default=M_FAINT_OFFSET_DEFAULT,
                   help="Schechter faint-end cutoff as an offset from M*: "
                        "M_faint = Mstar_hat + offset [mag] (default "
                        f"{M_FAINT_OFFSET_DEFAULT} = 0.01 L*). A PROTOCOL "
                        "constant, not a fitted parameter: it fixes the "
                        "completeness denominator, so the Q-table build and "
                        "the inference run must carry the SAME value. The fit "
                        "REFUSES samples reaching fainter than Mstar_hat + "
                        "offset (the MLE would be pinned at the support "
                        "boundary).")
    p.add_argument("--k_corr_coeffs", default=None,
                   help="Comma-separated polynomial coefficients c1[,c2,...] "
                        "of a fixed K-correction template K(z) = sum_j c_j "
                        "z**j applied to the OBSERVED magnitudes (no constant "
                        "term: c0 is exactly degenerate with M0hat). Default: "
                        "no K-correction.")
    p.add_argument("--strata", action="store_true",
                   help="Fit one theta PER STRATUM using the survey's "
                        "gal_stratum labels (pixelated from a STRATUM "
                        "column). Writes a multi-entry strata JSON "
                        "(format 1.1 when more than one stratum is present; "
                        "a single stratum stays 1.0-compatible).")
    p.add_argument("--m_lim_per_stratum", default=None,
                   help="Optional per-stratum magnitude limits as "
                        "'label:m_lim,label:m_lim' (e.g. '0:19.5,1:21.0'); "
                        "strata not listed use --m_lim. Only with --strata.")
    opts = p.parse_args(argv)

    if opts.family == "schechter":
        # Both are structural limits of the pinned c_sel_schechter, refused
        # here so a long fit never produces an unconsumable payload.
        if opts.k_corr_coeffs:
            _fatal("--family schechter with --k_corr_coeffs: the pinned "
                   "c_sel_schechter carries no K(z) template, so the fitted "
                   "theta could not be consumed in-likelihood. Fit the "
                   "gaussian family with --k_corr_coeffs, or extend "
                   "c_sel_schechter (and its H0-invariance pin) first.")
        if opts.strata or opts.m_lim_per_stratum:
            _fatal("--family schechter with --strata/--m_lim_per_stratum: the "
                   "stratified in-likelihood curve stack is built from the "
                   "gaussian common-mode + offset decomposition and has no "
                   "schechter counterpart. Fit each stratum as its own "
                   "single-stratum schechter file, or use --family gaussian.")

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

    # Per-galaxy stratum labels (only under --strata).
    labels = None
    if opts.strata:
        if "gal_stratum" not in props:
            _fatal("--strata needs a gal_stratum dataset; re-pixelate from a "
                   "raw catalog carrying a STRATUM column "
                   "(darksirens_pixelate).")
        labels = np.asarray(props["gal_stratum"])[real].astype(np.int64)
        uniq = np.unique(labels)
        _ok(f"Strata: {uniq.size} "
            f"({', '.join(str(int(u)) for u in uniq)})")
    elif opts.m_lim_per_stratum:
        _fatal("--m_lim_per_stratum requires --strata.")
    _end()

    m_lim_by = {}
    if opts.m_lim_per_stratum:
        try:
            for item in opts.m_lim_per_stratum.split(","):
                lbl, val = item.split(":")
                m_lim_by[int(lbl)] = float(val)
        except ValueError:
            _fatal("--m_lim_per_stratum must look like '0:19.5,1:21.0' "
                   f"(got {opts.m_lim_per_stratum!r})")

    _section("Fitting truncated luminosity function")
    fits = []
    groups = ([("all", np.ones(m.size, dtype=bool))] if labels is None else
              [(str(int(u)), labels == u) for u in np.unique(labels)])
    for name, mask in groups:
        m_lim_s = m_lim_by.get(int(name) if name != "all" else -1, opts.m_lim)
        fit = fit_selection_from_mags(m[mask], z[mask], m_lim_s,
                                      family=opts.family, stratum=name,
                                      M_faint_offset=opts.m_faint_offset,
                                      k_corr_coeffs=k_corr_coeffs)
        fits.append(fit)
        sd = np.sqrt(np.diag(fit.cov))
        _row("stratum", f"{name}  (n={fit.n_gal:,})")
        _row("  m_lim (fixed datum)", f"{fit.m_lim:.4f}")
        if k_corr_coeffs:
            _row("  K(z) coeffs", ", ".join(f"{c:.5g}" for c in k_corr_coeffs))
        # Driven by SELECTION_SAMPLED_FIELDS so the sd row order can never
        # drift from the covariance's row/column order.
        for i, name_ in enumerate(SELECTION_SAMPLED_FIELDS[fit.family]):
            scaled = "  (h-scaled)" if name_ in ("M0hat", "Mstar_hat") else ""
            _row(f"  {name_}",
                 f"{float(getattr(fit, name_)):.5f} +/- {sd[i]:.5f}{scaled}")
        if fit.family == "schechter":
            _row("  M_faint_offset",
                 f"{fit.M_faint_offset:.4f}  (fixed protocol constant; "
                 "must match the Q-table build and the inference run)")
    _end()

    # A single stratum stays byte-compatible with the 1.0 consumers; only a
    # genuinely multi-entry payload advances the format version.
    fmt = ("darksirens-selection-fit-1.0" if len(fits) == 1
           else "darksirens-selection-fit-1.1")
    payload = {
        "format_version": fmt,
        "strata": [f_.to_jsonable() for f_ in fits],
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
