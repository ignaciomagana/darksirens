"""The mock's PE calibration across eight INDEPENDENT realizations.

`pe_calibration.py` measured one realization and eliminated PE over-sharpness as
the source of Tier C's dispersion (residual sd 1.059 against the 2.6 that has to
be explained).  Two things that one number cannot settle: whether the sd is
representative or a lucky draw, and what to make of the residual MEAN, which came
out at +0.486 with a KS p of 0.0055.

Eight fresh mocks (seeds 8101-8108, same injection set, generated on js2h100)
were run through the same PP test.  This aggregates them.

**Read the sd and the mean separately, and do not read stability as a verdict.**
The sd is the quantity that speaks to dispersion.  The mean is expected to be
POSITIVE for detected events -- detection selects upward SNR fluctuations, which
under-estimate `dL`, so the truth sits above the PE mean -- and the hierarchical
likelihood's selection term exists to absorb exactly that.  A mean that
reproduces across realizations therefore does NOT distinguish "selection effect,
working as designed" from "DAG inconsistency": both are stable, and only noise
is not.  What would distinguish them is the offset's dependence on SNR (a
selection offset is largest at threshold and vanishes at high SNR) and whether
the estimator's own bias stays small once selection is included.

The uniformity re-test maps each stored quantile back through the Gaussian,
`z = Phi^-1(u)`, removes the per-seed mean, and re-tests -- so it inherits an
assumption of near-Gaussian `dL` posteriors that the raw KS does not.  It is
there to answer one question only: is the non-uniformity nothing but the mean?
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def aggregate(paths):
    per = []
    us = []
    for p in sorted(paths):
        d = json.load(open(p))
        u = np.asarray(d["quantiles"], float)
        us.append(u)
        per.append(dict(
            seed=int(os.path.basename(p).split("_")[1].split(".")[0]),
            n_events=d["n_events"], ks_p=d["ks_p"], frac_in_90=d["frac_in_90"],
            frac_in_68=d["frac_in_68"], n_outside_99=d["n_outside_99"],
            resid_mean=d["resid_mean"], resid_sd=d["resid_sd"],
            median_frac_dl_error=d["median_frac_dl_error"],
            snr_median=d["snr_median"]))

    u_all = np.concatenate(us)
    means = np.array([r["resid_mean"] for r in per])
    sds = np.array([r["resid_sd"] for r in per])
    n = len(u_all)
    ks = stats.kstest(u_all, "uniform")

    # Is the non-uniformity nothing but the mean?  Map u -> z, de-mean per seed,
    # map back.  Quantiles are discrete at 1/nsamp, so clip off the boundary.
    z_res = []
    for u in us:
        z = stats.norm.ppf(np.clip(u, 0.5 / 512, 1 - 0.5 / 512))
        z_res.append(z - z.mean())
    u_dm = stats.norm.cdf(np.concatenate(z_res))
    ks_dm = stats.kstest(u_dm, "uniform")

    return dict(
        n_realizations=len(per), n_events_total=int(n), per_seed=per,
        resid_mean_across_seeds=float(means.mean()),
        resid_mean_sd_across_seeds=float(means.std(ddof=1)),
        resid_mean_se=float(sds.mean() / np.sqrt(n)),
        resid_mean_min=float(means.min()), resid_mean_max=float(means.max()),
        resid_mean_all_positive=bool((means > 0).all()),
        resid_sd_mean=float(sds.mean()), resid_sd_spread=float(sds.std(ddof=1)),
        resid_sd_min=float(sds.min()), resid_sd_max=float(sds.max()),
        required_for_tier_c=2.6,
        frac_in_90_pooled=float(((u_all >= 0.05) & (u_all <= 0.95)).mean()),
        frac_in_68_pooled=float(((u_all >= 0.16) & (u_all <= 0.84)).mean()),
        n_outside_99_pooled=int(((u_all < 0.005) | (u_all > 0.995)).sum()),
        ks_stat_pooled=float(ks.statistic), ks_p_pooled=float(ks.pvalue),
        ks_stat_demeaned=float(ks_dm.statistic),
        ks_p_demeaned=float(ks_dm.pvalue))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--glob", default="pecal_multiseed/pecal_*.json")
    p.add_argument("--out", default="pecal_multiseed.json")
    a = p.parse_args(argv)
    r = aggregate(glob.glob(a.glob))
    print(json.dumps({k: v for k, v in r.items() if k != "per_seed"}, indent=2))
    print(f"{'seed':>6} {'resid_mean':>11} {'resid_sd':>9} {'ks_p':>8} "
          f"{'f90':>6} {'f68':>6} {'out99':>6}")
    for s in r["per_seed"]:
        print(f"{s['seed']:>6} {s['resid_mean']:>+11.3f} {s['resid_sd']:>9.3f} "
              f"{s['ks_p']:>8.4f} {s['frac_in_90']:>6.3f} "
              f"{s['frac_in_68']:>6.3f} {s['n_outside_99']:>6d}")
    with open(a.out, "w") as f:
        json.dump(r, f, indent=1)
    print(f"[write] {a.out}")


if __name__ == "__main__":
    main()
