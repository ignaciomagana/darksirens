"""Is the mock PE's ``+0.4 sigma`` offset the selection effect, or a defect?

``pecal_multiseed.py`` measured a residual mean of ``+0.399`` that reproduces
across eight independent mocks and is the ENTIRE non-uniformity of the PP test
(pooled KS ``p = 9.8e-17`` raw, ``0.32`` after removing the per-seed mean).
Stability alone does not say what causes it: a working selection effect and a
generator defect are both stable.

The discriminator is the offset's SNR dependence.  Detection keeps upward SNR
fluctuations, which UNDER-estimate ``dL``, so the truth sits above the PE mean
-- but only for events near threshold.  A loud event is detected whatever its
noise draw, so its residual must be centred.  A construction defect need not
behave that way.

The prediction, from the mock's own measurement model (``sigma_rho = 1``,
``rho_threshold = 8``, ``dL = k/rho`` at fixed masses).  With ``n = rho_obs -
rho_true`` and ``a = rho_thr - rho_true``,

    z = (dL_true - mean_PE) / sd_PE  ~  n + (n^2 - 1)/rho_true
    E[z | detected] ~ lambda(a) + a lambda(a) / rho_true,   lambda(a) = phi(a)/(1-Phi(a))

which is ``+0.80`` at threshold and falls through ``+0.14`` at ``rho = 9.5`` to
``~0`` by ``rho = 12``.  It is an APPROXIMATION -- it takes the ``dL`` posterior
straight from the ``rho`` posterior and ignores the mass/sky marginalisation
that widens it -- so the test is the SHAPE (a decline to zero with SNR) plus an
order-of-magnitude agreement, not a fit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import h5py
import numpy as np
from scipy import stats


def per_event(path):
    with h5py.File(path) as f:
        n = int(f.attrs["nobs"])
        ns = int(f.attrs["nsamp"])
        thr = float(f.attrs.get("snr_threshold", 8.0))
        dL = f["dL"][...].reshape(n, ns)
        dl_true = f["truth"]["dl"][...]
        rho_true = np.asarray(f["truth"]["snr_true"][...]
                              if "snr_true" in f["truth"] else
                              f["truth"]["snr"][...])
        rho_obs = np.asarray(f["truth"]["obs_rho"][...])
    z = (dl_true - dL.mean(axis=1)) / dL.std(axis=1)
    u = np.array([(dL[i] < dl_true[i]).mean() for i in range(n)])
    return dict(z=z, u=u, rho_true=rho_true, rho_obs=rho_obs, thr=thr)


def predicted(rho_true, thr):
    a = thr - np.asarray(rho_true, dtype=float)
    lam = stats.norm.pdf(a) / np.clip(stats.norm.sf(a), 1e-300, None)
    return lam + a * lam / np.asarray(rho_true, dtype=float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--glob", default="../../../runs/q/mock_81*/gw_events.h5")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="additional gw_events.h5 paths (e.g. the Tier-B mock)")
    ap.add_argument("--out", default="pe_offset_vs_snr.json")
    a = ap.parse_args(argv)

    paths = sorted(glob.glob(a.glob)) + list(a.extra)
    if not paths:
        raise SystemExit(f"no mocks matched {a.glob!r}")
    per = [per_event(p) for p in paths]
    z = np.concatenate([d["z"] for d in per])
    rt = np.concatenate([d["rho_true"] for d in per])
    ro = np.concatenate([d["rho_obs"] for d in per])
    thr = float(per[0]["thr"])
    pred = predicted(rt, thr)

    edges = np.array([0.0, 8.0, 9.0, 10.0, 11.0, 12.0, 15.0, 1e9])
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rt >= lo) & (rt < hi)
        if m.sum() < 3:
            continue
        bins.append(dict(
            rho_lo=float(lo), rho_hi=float(hi), n=int(m.sum()),
            rho_true_mean=float(rt[m].mean()),
            z_mean=float(z[m].mean()),
            z_sem=float(z[m].std(ddof=1) / np.sqrt(m.sum())),
            z_sd=float(z[m].std(ddof=1)),
            predicted_mean=float(pred[m].mean())))

    # the shape test: does the offset decline with true SNR?
    lo_m, hi_m = rt < 10.0, rt >= 12.0
    slope, icept, r, p_slope, se = stats.linregress(rt, z)
    out = dict(
        n_mocks=len(paths), n_events=int(z.size), snr_threshold=thr,
        z_mean_all=float(z.mean()),
        z_sem_all=float(z.std(ddof=1) / np.sqrt(z.size)),
        predicted_mean_all=float(pred.mean()),
        bins=bins,
        slope_z_vs_rho_true=float(slope), slope_se=float(se),
        slope_p=float(p_slope),
        z_mean_below_10=float(z[lo_m].mean()) if lo_m.any() else None,
        n_below_10=int(lo_m.sum()),
        z_mean_above_12=float(z[hi_m].mean()) if hi_m.any() else None,
        z_sem_above_12=(float(z[hi_m].std(ddof=1) / np.sqrt(hi_m.sum()))
                        if hi_m.sum() > 1 else None),
        n_above_12=int(hi_m.sum()),
        corr_z_rho_obs=float(np.corrcoef(z, ro)[0, 1]),
        paths=[os.path.abspath(p) for p in paths])
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("bins", "paths")}, indent=2))
    print(f"{'rho_true':>16} {'n':>5} {'<z> measured':>14} {'+-':>7} "
          f"{'<z> predicted':>14}")
    for b in bins:
        print(f"{b['rho_lo']:7.1f}-{b['rho_hi']:<8.1f} {b['n']:>5} "
              f"{b['z_mean']:>+14.3f} {b['z_sem']:>7.3f} "
              f"{b['predicted_mean']:>+14.3f}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("PE_OFFSET_SNR_DONE", flush=True)


if __name__ == "__main__":
    main()
