"""Where Tier C's dispersion comes from -- a two-way variance decomposition.

Tier C measures a realization-to-realization spread of the ``H0`` median of
~19 km/s against a quoted ``sigma`` of ~7 km/s: the estimator is CENTRED and
its interval is ~2.6x too narrow, in the latent arm AND in ``latent_off``,
which has no field at all.  The first closure pass localized that to "not the
latent seam" and stopped there.  This script goes one step further and asks
which HALF of the realization the variance lives in.

``make_mock.build`` draws one realization from one ``rng``, in this order:

    seed -> xi_true -> complete catalog -> mask/survey -> 60 events -> PE

so ``make_mock.build(seed, event_seed=e)`` re-seeds the stream immediately
before the event draw.  Everything upstream (the field, the galaxies, the
footprint, the survey catalog the likelihood conditions on) is a function of
``seed``; everything downstream (which hosts are detected, and their PE noise)
is a function of ``e``.  Two blocks then separate cleanly:

  **ROW block** -- one catalog, many event sets.  ``seed`` fixed,
  ``event_seed`` varied.  This is the spread the posterior IS supposed to
  quantify: 60 events is a small sample and their scatter is exactly the
  statistical uncertainty ``sigma`` claims to be.  If the ROW spread alone is
  ~19 km/s, the likelihood is simply over-sharp PER EVENT and the fix is in the
  event likelihood or the PE model -- nothing to do with catalogs or fields.

  **CATALOG term** -- obtained by DIFFERENCE, not by a second block, and the
  distinction is worth being precise about.  Holding ``event_seed`` fixed and
  varying ``seed`` does NOT hold the events fixed: the hosts are drawn by
  rejection from ``complete``, so the same random stream applied to a different
  universe yields different events.  A column of this grid is therefore a
  sample of the TOTAL variance, not of a catalog-only block.  What the grid
  does give, exactly, is the law of total variance

      Var(H0) = E_cat[ Var_evt(H0 | cat) ]  +  Var_cat( E_evt[H0 | cat] )
                \------- within rows ------/    \------ between rows ------/

  with the second term debiased by ``within/n_event`` because each row mean is
  itself estimated from ``n_event`` noisy cells.  So: within-row variance is
  measured directly, total variance is measured directly, and the catalog
  common mode is what is left over.

Reading it: the posterior's quoted ``sigma`` is, to the accuracy of
Bernstein-von Mises, an estimate of the EVENT-CONDITIONAL spread -- it
conditions on the catalog as given.  If ``within-row / mean sigma`` is already
~2.6 the likelihood is over-sharp per event and the catalog is innocent.  If it
is ~1 and the catalog term carries the rest, the missing variance is a
realization-level common mode the posterior structurally cannot see.

Arms: ``latent_off`` is the one that matters here (no field, no ``b_gal``
inflation, so a positive result cannot be blamed on or credited to PR-6a), but
``latent`` is run alongside so the decomposition is stated for the deliverable
too.
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


def one(seed, event_seed, workdir, world, *, grid, n0, injections, arm_names):
    d = Path(workdir)
    make_mock.build(seed, d, world=world, n0=n0, verbose=False,
                    reuse_injections=injections, event_seed=event_seed)
    if "latent" in arm_names or "latent_bgal" in arm_names:
        build_anchor16.build(
            survey=d / "catalog_pixelated_nside_16.h5",
            mth_map=d / "mth_map_nside16.h5",
            out=d / "latent_anchor.h5", world=world, verbose=False)
    if "latent_bgal" in arm_names:
        build_anchor16.build(
            survey=d / "catalog_pixelated_nside_16.h5",
            mth_map=d / "mth_map_nside16.h5",
            out=d / "latent_anchor_bgal.h5", world=world, verbose=False,
            b_gal_dispersion=True)
    res = tier_b.run(d, grid=grid, arm_names=arm_names, quiet=True)
    for f in d.glob("*"):
        f.unlink()
    d.rmdir()
    return res


def stats_for(rows, arm):
    med = np.array([r[arm]["median"] for r in rows], dtype=float)
    sig = np.array([r[arm]["sigma"] for r in rows], dtype=float)
    return dict(n=len(rows), spread=float(np.std(med, ddof=1)),
                mean_sigma=float(np.mean(sig)),
                overconfidence=float(np.std(med, ddof=1) / np.mean(sig)),
                medians=med.tolist(), sigmas=sig.tolist())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-seed", type=int, default=6,
                   help="catalog realizations (the COLUMN axis)")
    p.add_argument("--n-event", type=int, default=6,
                   help="event/PE realizations per catalog (the ROW axis)")
    p.add_argument("--seed0", type=int, default=90000)
    p.add_argument("--event-seed0", type=int, default=310000)
    p.add_argument("--h0-step", type=float, default=2.5)
    p.add_argument("--n0", type=float, default=5e-5)
    p.add_argument("--injections", default="data/injections.h5")
    p.add_argument("--arms", nargs="*", default=["latent_off", "latent"])
    p.add_argument("--out", default="variance_split.json")
    a = p.parse_args(argv)

    grid = np.arange(20.0, 140.0 + 0.5 * a.h0_step, a.h0_step)
    world = W16.build_world()
    cells = []
    t0 = time.time()
    n = 0
    for i in range(a.n_seed):
        for j in range(a.n_event):
            seed = a.seed0 + 37 * i
            esd = a.event_seed0 + 131 * j
            r = one(seed, esd, W16.PR6A_DIR / "data" / f"v{i:02d}_{j:02d}",
                    world, grid=grid, n0=a.n0, injections=a.injections,
                    arm_names=tuple(a.arms))
            cells.append({arm: {k: r[arm][k] for k in
                                ("median", "sigma", "cdf_at_truth", "width90")}
                          for arm in a.arms}
                         | {"i": i, "j": j, "seed": seed, "event_seed": esd})
            n += 1
            msg = "  ".join(f"{arm}: H0={r[arm]['median']:.1f}"
                            for arm in a.arms)
            print(f"[vsplit] {n}/{a.n_seed * a.n_event} "
                  f"seed={seed} event_seed={esd}  {msg}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            with open(W16.PR6A_DIR / a.out, "w") as f:
                json.dump({"cells": cells, "arms": a.arms}, f, indent=1)

    out = {"cells": cells, "arms": a.arms, "n_seed": a.n_seed,
           "n_event": a.n_event, "verdict": {}}
    for arm in a.arms:
        M = np.full((a.n_seed, a.n_event), np.nan)
        S = np.full((a.n_seed, a.n_event), np.nan)
        for c in cells:
            M[c["i"], c["j"]] = c[arm]["median"]
            S[c["i"], c["j"]] = c[arm]["sigma"]
        # Balanced two-way random-effects decomposition on the H0 median.
        # Row means vary => catalog common mode; within-row scatter => events.
        grand = float(np.nanmean(M))
        row_mean = np.nanmean(M, axis=1)          # E_evt[H0 | catalog i]
        v_within = float(np.nanmean(np.nanvar(M, axis=1, ddof=1)))
        v_total = float(np.nanvar(M, ddof=1))
        # Debias: each row mean is itself an average of n_event noisy cells, so
        # Var(row_mean) carries v_within/n_event of pure event noise.  Subtract
        # it.  The estimate can go negative when the catalog term is small
        # compared with its own sampling error -- report it as measured rather
        # than clipping it to zero, and quote the n it was measured at.
        v_between_raw = float(np.nanvar(row_mean, ddof=1))
        v_catalog = v_between_raw - v_within / float(a.n_event)
        sg = float(np.nanmean(S))
        out["verdict"][arm] = dict(
            n_cells=int(np.isfinite(M).sum()),
            n_seed=int(a.n_seed), n_event=int(a.n_event),
            mean_quoted_sigma=sg,
            total_spread=float(v_total ** 0.5),
            overconfidence_total=float(v_total ** 0.5 / sg),
            # ROW block, measured directly: catalog held fixed, events varied.
            event_spread_within_catalog=float(v_within ** 0.5),
            overconfidence_events_only=float(v_within ** 0.5 / sg),
            # Catalog common mode, by difference (see the module docstring).
            catalog_spread_raw=float(v_between_raw ** 0.5),
            catalog_spread_debiased=(float(v_catalog ** 0.5)
                                     if v_catalog > 0 else
                                     -float((-v_catalog) ** 0.5)),
            var_within=v_within, var_total=v_total,
            var_between_raw=v_between_raw, var_catalog_debiased=v_catalog,
            frac_variance_from_events=float(v_within / v_total),
            grand_mean=grand, H0_true=float(W16.H0_TRUE),
            medians=M.tolist(), sigmas=S.tolist())
    print(json.dumps({k: {kk: vv for kk, vv in v.items()
                          if kk not in ("medians", "sigmas")}
                      for k, v in out["verdict"].items()}, indent=2))
    with open(W16.PR6A_DIR / a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[write] {W16.PR6A_DIR / a.out}")


if __name__ == "__main__":
    main()
