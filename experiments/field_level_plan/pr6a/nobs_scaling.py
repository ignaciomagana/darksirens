"""Two families remain after the variance split.  This tells them apart.

``variance_split.py`` localizes ~80% of Tier C's excess variance to the EVENT
channel: at a FIXED catalog, redrawing the 60 events scatters ``H0`` by ~15
km/s against a quoted ``sigma`` of ~7.  Two very different defects produce that
signature, and they have opposite ``N_obs`` scalings:

  **(A) per-event over-sharpness.**  Each event's likelihood claims a factor
  ``k^2`` too much information -- a mis-specified per-event term (the catalog
  z-kernel, the PE measure ``p_pe``, the sky treatment, a pinned population that
  should be marginalized).  Then BOTH the true spread and the quoted ``sigma``
  fall like ``1/sqrt(N)`` and the ratio is FLAT in ``N``, pinned at ``k``.

  **(B) a per-realization common mode in the event channel.**  Something that
  is one number per realization rather than per event -- the selection integral
  ``beta``, a shared calibration, an ``N``-independent normalization error.
  Then the quoted ``sigma`` falls like ``1/sqrt(N)`` while the offset does not,
  and the ratio GROWS like ``sqrt(N)``.

So: hold the catalog fixed, hold everything else fixed, and sweep ``N_obs``.
A flat ratio says (A) and sends the next workstream into the per-event
likelihood.  A ``sqrt(N)`` ratio says (B) and sends it into the selection
normalization.  The two are not close, and 3 x n_event cells resolve them.

Only ``latent_off`` is needed -- it carries no field and no ``b_gal``, so
whatever this measures cannot be attributed to PR-6a -- but ``latent`` is run
alongside for free (the anchor is built once per cell either way).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import world16 as W16
import build_anchor16
import make_mock
import tier_b


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=90000,
                   help="the ONE catalog; held fixed across every cell")
    p.add_argument("--nobs", nargs="*", type=int, default=[30, 60, 120])
    p.add_argument("--n-event", type=int, default=6)
    p.add_argument("--event-seed0", type=int, default=410000)
    p.add_argument("--h0-step", type=float, default=2.5)
    p.add_argument("--n0", type=float, default=5e-5)
    p.add_argument("--injections", default="data/injections.h5")
    p.add_argument("--arms", nargs="*", default=["latent_off", "latent"])
    p.add_argument("--out", default="nobs_scaling.json")
    a = p.parse_args(argv)

    grid = np.arange(20.0, 140.0 + 0.5 * a.h0_step, a.h0_step)
    world = W16.build_world()
    cells = []
    t0 = time.time()
    for nobs in a.nobs:
        for j in range(a.n_event):
            esd = a.event_seed0 + 131 * j
            d = W16.PR6A_DIR / "data" / f"n{nobs:04d}_{j:02d}"
            make_mock.build(a.seed, d, world=world, n0=a.n0, verbose=False,
                            reuse_injections=a.injections, event_seed=esd,
                            nobs=int(nobs))
            if "latent" in a.arms:
                build_anchor16.build(
                    survey=d / "catalog_pixelated_nside_16.h5",
                    mth_map=d / "mth_map_nside16.h5",
                    out=d / "latent_anchor.h5", world=world, verbose=False)
            r = tier_b.run(d, grid=grid, arm_names=tuple(a.arms), quiet=True)
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()
            cells.append({arm: {k: r[arm][k] for k in
                                ("median", "sigma", "cdf_at_truth", "width90")}
                          for arm in a.arms}
                         | {"nobs": int(nobs), "event_seed": esd})
            print(f"[nobs] N={nobs:4d} e={esd}  "
                  + "  ".join(f"{arm}: H0={r[arm]['median']:6.2f} "
                              f"sig={r[arm]['sigma']:5.2f}"
                              for arm in a.arms)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            with open(W16.PR6A_DIR / a.out, "w") as f:
                json.dump({"cells": cells, "arms": a.arms, "seed": a.seed},
                          f, indent=1)

    out = {"cells": cells, "arms": a.arms, "seed": a.seed, "verdict": {}}
    for arm in a.arms:
        rows = []
        for nobs in a.nobs:
            m = np.array([c[arm]["median"] for c in cells
                          if c["nobs"] == nobs])
            s = np.array([c[arm]["sigma"] for c in cells
                          if c["nobs"] == nobs])
            rows.append(dict(nobs=int(nobs), n=int(m.size),
                             spread=float(np.std(m, ddof=1)),
                             mean_sigma=float(s.mean()),
                             overconfidence=float(np.std(m, ddof=1)
                                                  / s.mean())))
        out["verdict"][arm] = rows
        for r in rows:
            print(f"  [{arm}] N={r['nobs']:4d}  spread {r['spread']:7.3f}  "
                  f"mean sigma {r['mean_sigma']:6.3f}  "
                  f"overconfidence {r['overconfidence']:6.3f}")
    with open(W16.PR6A_DIR / a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {W16.PR6A_DIR / a.out}")


if __name__ == "__main__":
    main()
