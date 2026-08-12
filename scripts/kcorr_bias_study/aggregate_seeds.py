#!/usr/bin/env python
"""Aggregate the per-realization propagate_to_h0.py outputs.

The bias is the PAIRED difference (model - reanchored) within each mock, so
the realization-to-realization scatter of the recovered H0 -- which is large
and common to all four models -- cancels.  Reported with the standard error
of the paired mean, so "is the shift real" is answerable.
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np

REF = "reanchored"
MODES = ["stamped", "firstorder_meandm", "firstorder_mle"]
PARS = ["H0", "Om0", "w0", "wa"]


def main(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no files matching {pattern}")
    runs = [json.load(open(f)) for f in files]
    n = len(runs)
    print(f"# {n} realizations from {pattern}")
    c = runs[0]["config"]
    print(f"# mock: ngal={c['ngal']} nevents={c['nevents']} "
          f"dl_frac_err={c['dl_frac_err']} m_lim={c['m_lim']} "
          f"z in [{c['zlo']},{c['zhi']}] kcorr={c['kcorr']} "
          f"H0_true={c['H0_true']}")
    print(f"# grid: Om0x{c['nom0']} w0x{c['nw0']} wax{c['nwa']} "
          f"M0hatx{c['nm0']} H0x{c['nh0']} (H0 in [{c['h0_lo']},{c['h0_hi']}])")

    sd = np.array([r["fit"]["laplace_sd_M0hat"] for r in runs])
    dmin = np.array([r["delta_range"][0] for r in runs])
    dmax = np.array([r["delta_range"][1] for r in runs])
    nobs = np.array([r["mock"]["n_obs"] for r in runs])
    print(f"\n## offline fit: n_obs = {nobs.mean():.0f}, "
          f"Laplace sd(M0hat) = {sd.mean():.5f} mag")
    print(f"## delta(Theta) over the sampled prior box: "
          f"{dmin.mean():+.4f} .. {dmax.mean():+.4f} mag "
          f"= {dmin.mean()/sd.mean():+.1f} .. {dmax.mean()/sd.mean():+.1f} "
          f"Laplace sd")
    if "dsigma_range" in runs[0]:
        sds = np.array([r["laplace_sd_sigma_M"] for r in runs])
        lo = np.array([r["dsigma_range"][0] for r in runs])
        hi = np.array([r["dsigma_range"][1] for r in runs])
        print(f"## dsigma_M(Theta) over the prior box: {lo.mean():+.4f} .. "
              f"{hi.mean():+.4f} mag = {lo.mean()/sds.mean():+.1f} .. "
              f"{hi.mean()/sds.mean():+.1f} Laplace sd "
              f"(sd(sigma_M) = {sds.mean():.5f})")
    if "approx_residual_mag" in runs[0]:
        print("\n## worst-case anchor residual left by each treatment [mag]")
        print(f"{'treatment':<20} {'max|resid|':>11} {'/ Laplace sd':>13}")
        for k in ("stamped", "firstorder_meandm", "firstorder_mle"):
            v = np.array([r["approx_residual_mag"][k]["max_abs"] for r in runs])
            s = np.array([r["approx_residual_mag"][k]["max_abs_over_laplace_sd"]
                          for r in runs])
            print(f"{k:<20} {v.mean():>11.4f} {s.mean():>13.1f}")
        sl = {k: np.array([r["mle_slopes"][k] for r in runs])
              for k in runs[0]["mle_slopes"]}
        print("   MLE slopes dM0hat/dTheta: "
              + ", ".join(f"{k} {v.mean():+.4f}" for k, v in sl.items()))

    print(f"\n## recovered posteriors (mean over {n} realizations)")
    print(f"{'model':<20} " + " ".join(f"{p+' mean':>10} {p+' sd':>8}"
                                       for p in ("H0",)))
    for mode in [REF] + MODES:
        m = np.array([r["models"][mode]["H0"]["mean"] for r in runs])
        s = np.array([r["models"][mode]["H0"]["sd"] for r in runs])
        print(f"{mode:<20} {m.mean():>10.3f} {s.mean():>8.3f}   "
              f"(realization scatter of the mean: {m.std(ddof=1):.3f})")

    print(f"\n## PAIRED bias vs '{REF}' (the exact per-proposal re-anchor)")
    for par in PARS:
        ref_sd = np.array([r["models"][REF][par]["sd"] for r in runs])
        print(f"\n### {par}   (sigma_stat = {ref_sd.mean():.4f})")
        print(f"{'model':<20} {'bias':>10} {'sem':>9} {'bias/sigma':>11} "
              f"{'sem(b/s)':>9}")
        for mode in MODES:
            d = np.array([r["models"][mode][par]["mean"]
                          - r["models"][REF][par]["mean"] for r in runs])
            bos = d / ref_sd
            print(f"{mode:<20} {d.mean():>+10.4f} {d.std(ddof=1)/np.sqrt(n):>9.4f} "
                  f"{bos.mean():>+11.4f} {bos.std(ddof=1)/np.sqrt(n):>9.4f}")

    print("\n## posterior WIDTH ratio vs the exact re-anchor "
          "(>1 = broader, <1 = over-confident)")
    print(f"{'model':<20} " + " ".join(f"{p:>10}" for p in PARS))
    for mode in MODES:
        row = []
        for par in PARS:
            a = np.array([r["models"][mode][par]["sd"] for r in runs])
            b = np.array([r["models"][REF][par]["sd"] for r in runs])
            row.append(f"{(a / b).mean():>10.4f}")
        print(f"{mode:<20} " + " ".join(row))


if __name__ == "__main__":
    main(sys.argv[1])
