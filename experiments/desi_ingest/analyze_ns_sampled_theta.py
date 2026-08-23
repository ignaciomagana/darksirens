"""Sampled-theta NS run: convergence, theta prior-vs-posterior pulls, H0.

The two deliverables of the sampled-theta item:

  1. H0 marginalized over selection-model uncertainty, against the
     theta-fixed grid baselines (old-n0 sel 75.7 [57.2, 87.8]; recal
     fixed-theta sel 72.96 [57.0, 84.3]).
  2. The theta posterior-vs-prior comparison as a MISSPECIFICATION GATE:
     the prior is the 22.8M-galaxy magnitude fit's truncated normal, so a
     posterior pulled off it means the GW data disagree with the catalog
     about the selection model.  Expectation from the +/-5 sigma ablation
     (dlogL <= 5e-4): posterior sits on the prior (pulls ~0, width ratio
     ~1), H0 unchanged from the fixed-theta scan.

Usage:  python analyze_ns_sampled_theta.py --run data/ns_sampled_theta
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

import common as C  # noqa: F401  (pins DARKSIRENS_ZMAX=0.75)

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="data/ns_sampled_theta")
    p.add_argument("--out", default="data/ns_sampled_theta_summary.json")
    args = p.parse_args(argv)

    run_dirs = sorted(glob.glob(str(Path(args.run) / "*" )))
    run_dir = next((d for d in run_dirs
                    if (Path(d) / "samples.npy").exists()), None)
    if run_dir is None:
        raise SystemExit(f"no run dir with samples.npy under {args.run}")
    samples = np.load(Path(run_dir) / "samples.npy")
    settings = json.load(open(Path(run_dir) / "settings.json"))
    labels = settings["expected_sampled_labels"]
    assert list(labels) == ["H0", "M0hat", "sigma_M"], labels
    cols = {n: samples[:, i] for i, n in enumerate(labels)}

    # The fit prior (the 22.8M-galaxy anchor) this run sampled under.
    sel = load_selection_fit_json(C.DATA_DIR / "selection_fit_union.json")
    sd = np.sqrt(np.diag(np.asarray(sel["cov"], dtype=float)))
    prior = {"M0hat": (float(sel["M0hat"]), float(sd[0])),
             "sigma_M": (float(sel["sigma_M"]), float(sd[1]))}

    def _q(x, a):
        return float(np.percentile(x, a))

    out = {"run_dir": run_dir, "n_samples": int(samples.shape[0])}
    out["H0"] = {"median": _q(cols["H0"], 50),
                 "ci68": [_q(cols["H0"], 16), _q(cols["H0"], 84)],
                 "ci90": [_q(cols["H0"], 5), _q(cols["H0"], 95)]}
    out["theta_gate"] = {}
    for name in ("M0hat", "sigma_M"):
        mu0, s0 = prior[name]
        med, std = float(np.median(cols[name])), float(np.std(cols[name]))
        out["theta_gate"][name] = {
            "prior": [mu0, s0], "posterior_median": med,
            "posterior_sd": std,
            "pull": (med - mu0) / s0,          # posterior center vs prior, in prior sd
            "width_ratio": std / s0,           # ~1 = prior-dominated (expected)
        }
    # Baselines for the comparison table.
    for tag, path in (("grid_old_n0", "data/h0_real/h0_real_scans.json"),
                      ("grid_recal", "data/h0_recal/h0_real_scans.json")):
        try:
            r = json.load(open(path))["results"]["sel"]
            out[tag] = {"median": r["median"], "ci68": r["ci68"]}
        except Exception:
            out[tag] = None

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    C.write_provenance(Path(args.out), {"run_dir": run_dir})

    print(f"[ns] {out['n_samples']} equal-weight samples from {run_dir}")
    print(f"[ns] H0 = {out['H0']['median']:.2f} "
          f"[{out['H0']['ci68'][0]:.1f}, {out['H0']['ci68'][1]:.1f}]")
    for name, g in out["theta_gate"].items():
        print(f"[ns] {name}: pull = {g['pull']:+.3f} prior-sd, "
              f"width ratio = {g['width_ratio']:.3f}")
    for tag in ("grid_recal", "grid_old_n0"):
        if out[tag]:
            print(f"[ns] baseline {tag}: {out[tag]['median']:.2f} "
                  f"{out[tag]['ci68']}")


if __name__ == "__main__":
    main()
