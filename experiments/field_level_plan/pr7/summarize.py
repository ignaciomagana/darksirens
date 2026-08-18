#!/usr/bin/env python
"""Emit REPORT.md's tables straight from the JSONs.

Every number in `REPORT.md` is printed by this script and pasted, so the report
cannot drift from the campaign by a transcription.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

D = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    te = _load("tier_e.json")
    ov = _load("tier_e_overlap.json")

    if te:
        s = te["summary"]
        rows = te["rows"]
        print("## Gate (i) — bias-ratio recovery\n")
        print(f"n_real                      {s['n_real']}")
        print(f"|pull| < 2                  {s['gate_i_within_2sigma']}/{s['n_real']}")
        print(f"pull mean                   {s['pull_mean']:+.4f} +- "
              f"{s['pull_sem']:.4f} (sem)")
        print(f"pull sd                     {s['pull_sd']:.4f}")
        print(f"r mean                      {s['r_mean_shared']:.6f}")
        print(f"r scatter over realizations {s['r_sd_shared']:.6f}")
        print(f"quoted sigma_r (median)     {s['sigma_median_shared']:.6f}")
        print(f"scatter / quoted            "
              f"{s['r_sd_shared'] / s['sigma_median_shared']:.4f}")
        print(f"max |pull|                  "
              f"{max(abs(r['pull_shared']) for r in rows):.4f}")
        print()
        print("## Gate (ii) / (iii') — shared vs decoupled\n")
        print(f"shared tighter in           "
              f"{s['gate_ii_shared_tighter']}/{s['n_real']}")
        print(f"width ratio (median)        x{s['width_ratio_median']:.2f}")
        print(f"width ratio (min, max)      x{s['width_ratio_min']:.2f}, "
              f"x{s['width_ratio_max']:.2f}")
        print(f"sigma_r shared (median)     {s['sigma_median_shared']:.6f}")
        print(f"sigma_r decoupled (median)  {s['sigma_median_decoupled']:.6f}")
        print(f"r decoupled (median)        {s['r_median_decoupled']:.6f}")
        print(f"pull decoupled              {s['pull_decoupled_mean']:+.4f} +- "
              f"{s['pull_decoupled_sd']:.4f}")
        de_in = sum(abs(r["pull_decoupled"]) < 2 for r in rows)
        print(f"|pull_decoupled| < 2        {de_in}/{s['n_real']}")
        print(f"corr(b1,b2) shared median   {s['corr_median_shared']:.8f}")
        print(f"corr(b1,b2) shared min      {s['corr_min_shared']:.8f}")
        print(f"corr(b1,b2) decoupled max   "
              f"{s['corr_max_abs_decoupled']:.1e}  (exactly 0 by construction)")
        print()
        print("## Convergence\n")
        print(f"inner solve grad_inf max    {s['grad_inf_max_shared']:.3e}")
        print(f"outer |dP/du| max (any dir) {s['profile_grad_max_shared']:.3e}")
        print(f"outer |dP/du| ratio dir max "
              f"{s['grad_ratio_max_shared']:.3e}")
        # The scale-free statement: what the residual gradient implies for r.
        off = [abs(r["shared"]["grad_ratio"]) * r["shared"]["sigma_log"] ** 2
               / max(r["shared"]["sigma_log"], 1e-300) for r in rows]
        print(f"implied |dlog r| / sigma_log max  {max(off):.3e}")
        print(f"wall (shared+decoupled)     {s['wall_s_total']:.0f} s")
        print()
        if te.get("sweep_summary"):
            print("## P-E4 — the log-bias prior sweep\n")
            print("| prior sd s | sigma_log r shared | sigma_log r decoupled | "
                  "r shared | r decoupled |")
            print("|---|---|---|---|---|")
            for x in te["sweep_summary"]:
                print(f"| {x['prior_s']:.2f} | {x['sigma_log_shared']:.6f} | "
                      f"{x['sigma_log_decoupled']:.6f} | {x['r_shared']:.5f} | "
                      f"{x['r_decoupled']:.5f} |")
            sh = [x["sigma_log_shared"] for x in te["sweep_summary"]]
            dec = [x["sigma_log_decoupled"] for x in te["sweep_summary"]]
            ps = [x["prior_s"] for x in te["sweep_summary"]]
            print(f"\nprior width spans x{max(ps) / min(ps):.0f}; shared "
                  f"sigma_log r spans x{max(sh) / min(sh):.3f}, decoupled "
                  f"spans x{max(dec) / min(dec):.3f}")
            print(f"decoupled sigma_log r / (sqrt(2) s): "
                  + ", ".join(f"{d / (np.sqrt(2) * p):.3f}"
                              for d, p in zip(dec, ps)))
            print()
        if te.get("verify"):
            v = te["verify"]
            print("## The curvature IS the width\n")
            print(f"seed {v['seed']}, r = {v['ratio']:.6f}, "
                  f"sigma_log r = {v['sigma_log']:.6f}")
            print("| offset | Delta J (nat) | Gaussian expects |")
            print("|---|---|---|")
            for p in v["points"]:
                print(f"| {p['n_sigma']:+.0f} sigma | {p['delta_J']:.4f} | "
                      f"{p['expected']:.4f} |")
            print()

    if ov:
        print("## R14 — the overlap arm\n")
        print("| overlap phi | r mean | r scatter | quoted sigma_r | pull | "
              "|pull| < 2 |")
        print("|---|---|---|---|---|---|")
        for x in ov["summary"]:
            print(f"| {x['overlap']:.2f} | {x['ratio_mean']:.5f} | "
                  f"{x['ratio_sd']:.5f} | {x['sigma_mean']:.5f} | "
                  f"{x['pull_mean']:+.2f} +- {x['pull_sd']:.2f} | "
                  f"{x['within_2sigma']}/{x['n']} |")
        print()


if __name__ == "__main__":
    main()
