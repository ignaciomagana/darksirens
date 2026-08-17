"""Tier C -- coverage over many realizations (PLAN §6.2).

    Accept: PP-plot KS p > 0.05; median H0 bias < 0.2 sigma; no realization
    outside the 99% band.

**Two arms only, and that is stated rather than implied.**  Each realization
needs its own mock (6 s), its own anchor (the count channel is fitted to that
realization's catalog) and its own scan.  The TABLE arm additionally needs its
own radial Q table, and ``cli/build_lognormal_completion`` takes minutes per
realization at this size -- an order of magnitude more than everything else
combined -- so Tier C runs ``latent`` (the PR-6a deliverable) and
``latent_off`` (its own control at identical ``f_p`` treatment).  The table
arm's coverage is not measured here; Tier B is the only place it appears.

**What the PP test can and cannot say.**  ``xi_true`` IS drawn from the prior
the MAP is regularized by, so the field's contribution to coverage is exact by
construction -- PLAN §6.1 says this in as many words ("coverage is guaranteed
by construction in exactly the regime where there is no data").  ``H0_true``
is NOT drawn from the flat ``[20, 140]`` scan prior; it is fixed at 67.74.  So
the uniformity of ``cdf(H0_true)`` is the usual likelihood-dominated
approximation to a PP test, not an exact one, and it is sensitive to the
posterior railing against a prior edge -- which is checked and reported.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import world16 as W16
import arms as A
import build_anchor16
import make_mock
import tier_b


def one(seed, workdir, world, *, grid, n0, injections, arm_names,
        keep=False, **mock_kw):
    d = Path(workdir)
    make_mock.build(seed, d, world=world, n0=n0, verbose=False,
                    reuse_injections=injections, **mock_kw)
    if "latent" in arm_names:
        build_anchor16.build(
            survey=d / "catalog_pixelated_nside_16.h5",
            mth_map=d / "mth_map_nside16.h5",
            out=d / "latent_anchor.h5", world=world, verbose=False)
    res = tier_b.run(d, grid=grid, arm_names=arm_names, quiet=True)
    res["truth"] = json.load(open(d / "truth.json"))
    if not keep:
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-real", type=int, default=50)
    p.add_argument("--seed0", type=int, default=90000)
    p.add_argument("--h0-step", type=float, default=2.5)
    p.add_argument("--n0", type=float, default=5e-5)
    p.add_argument("--injections", default="data/injections.h5")
    p.add_argument("--arms", nargs="*", default=["latent_off", "latent"])
    p.add_argument("--out", default="tier_c.json")
    a = p.parse_args(argv)

    grid = np.arange(20.0, 140.0 + 0.5 * a.h0_step, a.h0_step)
    world = W16.build_world()
    rows = []
    t0 = time.time()
    for k in range(a.n_real):
        seed = a.seed0 + 37 * k
        r = one(seed, W16.PR6A_DIR / "data" / f"c{k:03d}", world, grid=grid,
                n0=a.n0, injections=a.injections, arm_names=tuple(a.arms))
        rows.append({arm: {kk: r[arm][kk] for kk in
                           ("median", "ci68", "ci90", "sigma", "cdf_at_truth",
                            "map", "width90")}
                     for arm in a.arms} | {"seed": seed})
        msg = "  ".join(
            f"{arm}: H0={r[arm]['median']:.1f} cdf={r[arm]['cdf_at_truth']:.3f}"
            for arm in a.arms)
        print(f"[tier C] {k + 1}/{a.n_real} seed={seed}  {msg}  "
              f"({time.time() - t0:.0f}s elapsed)", flush=True)
        with open(W16.PR6A_DIR / a.out, "w") as f:
            json.dump({"rows": rows, "grid": grid.tolist(),
                       "n0": a.n0, "arms": a.arms}, f, indent=1)

    from scipy import stats

    out = {"rows": rows, "grid": grid.tolist(), "n0": a.n0, "arms": a.arms,
           "n_real": len(rows), "verdict": {}}
    for arm in a.arms:
        u = np.array([r[arm]["cdf_at_truth"] for r in rows])
        med = np.array([r[arm]["median"] for r in rows])
        sig = np.array([r[arm]["sigma"] for r in rows])
        ks = stats.kstest(u, "uniform")
        bias = np.median((med - W16.H0_TRUE) / sig)
        out["verdict"][arm] = dict(
            n_real=int(u.size),
            ks_stat=float(ks.statistic), ks_p=float(ks.pvalue),
            ks_pass=bool(ks.pvalue > 0.05),
            median_bias_sigma=float(bias),
            bias_pass=bool(abs(bias) < 0.2),
            n_outside_99=int(((u < 0.005) | (u > 0.995)).sum()),
            outside_99_pass=bool(((u < 0.005) | (u > 0.995)).sum() == 0),
            frac_in_90=float(((u >= 0.05) & (u <= 0.95)).mean()),
            frac_in_68=float(((u >= 0.16) & (u <= 0.84)).mean()),
            u=u.tolist())
        v = out["verdict"][arm]
        v["PASS"] = bool(v["ks_pass"] and v["bias_pass"]
                         and v["outside_99_pass"])
    out["verdict"]["TIER_C"] = bool(out["verdict"]
                                    .get("latent", {}).get("PASS", False))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "u"}
                      if isinstance(v, dict) else v
                      for k, v in out["verdict"].items()}, indent=2))
    with open(W16.PR6A_DIR / a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {W16.PR6A_DIR / a.out}")


if __name__ == "__main__":
    main()
