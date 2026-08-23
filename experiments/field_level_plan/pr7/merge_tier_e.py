#!/usr/bin/env python
"""Merge the chunked Tier E runs into one `tier_e.json` and re-summarize.

The summary statistics are recomputed here from the merged rows rather than
averaged across chunks, so the chunk boundaries leave no trace at all in the
reported numbers (the seeds are `seed0 + step * (offset + k)`, so a chunk is a
contiguous slice of one campaign, not a separate experiment).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", type=float, default=2.0)
    args = ap.parse_args(argv)

    rows, sweep, verify, cfg = [], [], None, None
    for p in sorted(glob.glob(os.path.join(args.dir, "tier_e_chunk_*.json"))):
        d = json.load(open(p))
        rows += d["rows"]
        cfg = cfg or d["config"]
    for p in sorted(glob.glob(os.path.join(args.dir, "tier_e_sweep_*.json"))):
        sweep += json.load(open(p))["sweep"]
    p = os.path.join(args.dir, "tier_e_verify.json")
    if os.path.exists(p):
        verify = json.load(open(p)).get("verify")

    rows.sort(key=lambda r: r["seed"])
    pulls = np.array([r["pull_shared"] for r in rows])
    sig_sh = np.array([r["shared"]["sigma"] for r in rows])
    sig_de = np.array([r["decoupled"]["sigma"] for r in rows])
    summary = dict(
        n_real=len(rows),
        gate_i_within_2sigma=int(np.sum(np.abs(pulls) < 2.0)),
        pull_mean=float(pulls.mean()), pull_sd=float(pulls.std(ddof=1)),
        pull_sem=float(pulls.std(ddof=1) / np.sqrt(len(pulls))),
        r_mean_shared=float(np.mean([r["shared"]["ratio"] for r in rows])),
        r_sd_shared=float(np.std([r["shared"]["ratio"] for r in rows], ddof=1)),
        r_median_shared=float(np.median([r["shared"]["ratio"] for r in rows])),
        r_median_decoupled=float(np.median([r["decoupled"]["ratio"]
                                            for r in rows])),
        sigma_median_shared=float(np.median(sig_sh)),
        sigma_median_decoupled=float(np.median(sig_de)),
        gate_ii_shared_tighter=int(np.sum(sig_sh < sig_de)),
        width_ratio_median=float(np.median(sig_de / sig_sh)),
        width_ratio_min=float(np.min(sig_de / sig_sh)),
        width_ratio_max=float(np.max(sig_de / sig_sh)),
        corr_median_shared=float(np.median([r["shared"]["corr"] for r in rows])),
        corr_min_shared=float(np.min([r["shared"]["corr"] for r in rows])),
        corr_max_abs_decoupled=float(np.max(np.abs(
            [r["decoupled"]["corr"] for r in rows]))),
        pull_decoupled_mean=float(np.mean([r["pull_decoupled"] for r in rows])),
        pull_decoupled_sd=float(np.std([r["pull_decoupled"] for r in rows],
                                       ddof=1)),
        grad_inf_max_shared=float(np.max([r["shared"]["grad_inf"]
                                          for r in rows])),
        profile_grad_max_shared=float(np.max(
            [np.max(np.abs(r["shared"]["profile_grad"])) for r in rows])),
        grad_ratio_max_shared=float(np.max(np.abs(
            [r["shared"]["grad_ratio"] for r in rows]))),
        n_converged_shared=int(sum(r["shared"]["converged"] for r in rows)),
        profile_grad_max_decoupled=float(np.max(
            [np.max(r["decoupled"]["profile_grad"]) for r in rows])),
        wall_s_total=float(sum(r["shared"]["wall_s"] + r["decoupled"]["wall_s"]
                               for r in rows)),
    )
    sweep_summary = []
    for s in sorted({x["prior_s"] for x in sweep}):
        sub = [x for x in sweep if x["prior_s"] == s]
        sweep_summary.append(dict(
            prior_s=float(s), n=len(sub),
            sigma_shared=float(np.median([x["shared"]["sigma"] for x in sub])),
            sigma_decoupled=float(np.median([x["decoupled"]["sigma"]
                                             for x in sub])),
            sigma_log_shared=float(np.median([x["shared"]["sigma_log"]
                                              for x in sub])),
            sigma_log_decoupled=float(np.median([x["decoupled"]["sigma_log"]
                                                 for x in sub])),
            r_shared=float(np.median([x["shared"]["ratio"] for x in sub])),
            r_decoupled=float(np.median([x["decoupled"]["ratio"]
                                         for x in sub]))))
    print(json.dumps(dict(summary=summary, sweep=sweep_summary,
                          verify=verify), indent=2))
    with open(args.out, "w") as f:
        json.dump(dict(config=cfg, summary=summary, sweep_summary=sweep_summary,
                       rows=rows, sweep=sweep, verify=verify), f, indent=1,
                  default=str)


if __name__ == "__main__":
    main()
