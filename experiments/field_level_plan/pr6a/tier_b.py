"""Tier B -- single-realization H0 closure, three arms (PLAN §6.2).

    Accept: H0_true inside the 90% CI in every arm; latent-frozen and table
    agree within 0.3 sigma at theta_fid; and latent-on CI width >= table CI
    width.

**The third criterion is reported with its validity, not just its value.**
PLAN §6.2 says it "is now a valid check because §3.4 propagates ``b_gal``"
via ``Cov(xi) = H^-1 + s_b^2 (d xi_hat/d b)(d xi_hat/d b)^T``.  Grepped on
this branch: ``latent_counts.laplace_draws(xi_hat, H_chol, n_draw, key)``
takes ``H_chol`` and nothing else, its only production caller
(``cli/build_latent_field.py:193``) passes the bare ``L``, and ``s_b`` --
which §3.4 defines as the profile curvature ``[-d^2 log p_count/db^2]^-1``
-- does not exist anywhere in ``darksirens/``.  The ``b_gal`` column of
``sensitivity_S`` IS built (``dgrad_db`` -> ``sensitivity``) and IS stored,
and ``latent_q.load_latent_plan`` reads it into ``LatentQPlan.S``, but no
consumer inflates the draw covariance with it.  So the member ensemble
carries ``H^-1`` alone and the CI-width comparison is measuring the
conditional field uncertainty at a FIXED ``b_gal``, not the propagated one.
The number is reported; the criterion is marked VACUOUS rather than passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import world16 as W16
import arms as A


def paths_for(d):
    d = Path(d)
    return dict(catalog=d / "catalog_pixelated_nside_16.h5",
                gw=d / "gw_events.h5", selection=d / "gw_selection.h5",
                mth=d / "mth_map_nside16.h5",
                selection_fit=d / "selection_fit.json",
                n0_calibration=d / "n0_calibration.json",
                anchor=d / "latent_anchor.h5", q_table=d / "q_radial.h5")


def run(dirname, *, grid=None, soft_guard=False, max_var=1e6,
        arm_names=("latent_off", "latent_off_nofp", "table", "latent"), quiet=False):
    import time

    p = paths_for(dirname)
    h0 = A.DEFAULT_GRID if grid is None else grid
    cache = {}
    out = {}
    for arm in arm_names:
        t0 = time.time()
        logl, opts, data = A.build(p, arm, data_cache=cache,
                                   soft_guard=soft_guard, max_var=max_var)
        t_build = time.time() - t0
        t0 = time.time()
        vals = A.scan_h0(logl, h0)
        s = A.summarize(h0, vals)
        s["build_s"] = t_build
        s["scan_s"] = time.time() - t0
        s["ms_per_eval"] = 1e3 * s["scan_s"] / h0.size
        out[arm] = s
        if not quiet:
            print(f"  [{arm}] H0 = {s['median']:.2f} "
                  f"68% [{s['ci68'][0]:.2f}, {s['ci68'][1]:.2f}] "
                  f"90% [{s['ci90'][0]:.2f}, {s['ci90'][1]:.2f}] "
                  f"sigma={s['sigma']:.2f}  cdf(H0_true)={s['cdf_at_truth']:.4f}"
                  f"  ({s['ms_per_eval']:.0f} ms/eval, build {t_build:.0f}s)",
                  flush=True)
    out["_h0"] = h0.tolist()
    return out


def verdict(res):
    h0t = W16.H0_TRUE
    v = {}
    for arm in ("latent_off", "latent_off_nofp", "table", "latent"):
        if arm not in res:
            continue
        s = res[arm]
        v[f"{arm}_in_ci90"] = bool(s["ci90"][0] <= h0t <= s["ci90"][1])
    if "table" in res and "latent" in res:
        st, sl = res["table"], res["latent"]
        d = abs(sl["median"] - st["median"])
        sig = 0.5 * (st["sigma"] + sl["sigma"])
        v["latent_vs_table_shift_sigma"] = float(d / sig)
        v["latent_vs_table_within_0p3sigma"] = bool(d / sig <= 0.3)
        v["ci90_width_latent"] = float(sl["width90"])
        v["ci90_width_table"] = float(st["width90"])
        v["latent_width_ge_table"] = bool(sl["width90"] >= st["width90"])
        v["width_criterion_status"] = (
            "VACUOUS -- b_gal is not propagated into the member draws; "
            "laplace_draws takes H_chol only and s_b is not implemented "
            "anywhere in darksirens/ (see the module docstring)")
    # The decomposition: latent-vs-table is NOT a pure field comparison,
    # because the table arm cannot carry ``--per_pixel_completeness``
    # (loaders.py:1021 refuses the pair).  Split it.
    def _shift(a, b):
        if a not in res or b not in res:
            return None
        sa, sb = res[a], res[b]
        return float(abs(sa["median"] - sb["median"])
                     / (0.5 * (sa["sigma"] + sb["sigma"])))

    v["shift_sigma_field_only__latent_vs_latent_off"] = _shift("latent",
                                                               "latent_off")
    v["shift_sigma_table_field__table_vs_latent_off_nofp"] = _shift(
        "table", "latent_off_nofp")
    v["shift_sigma_fp_channel__latent_off_vs_nofp"] = _shift(
        "latent_off", "latent_off_nofp")
    v["TIER_B"] = bool(all(v.get(f"{a}_in_ci90", False)
                           for a in ("latent_off", "table", "latent"))
                       and v.get("latent_vs_table_within_0p3sigma", False))
    return v


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="data/real0")
    p.add_argument("--out", default="tier_b.json")
    p.add_argument("--h0-step", type=float, default=1.0)
    p.add_argument("--soft-guard", action="store_true")
    a = p.parse_args(argv)
    grid = np.arange(20.0, 140.0 + 0.5 * a.h0_step, a.h0_step)
    truth = json.load(open(Path(a.dir) / "truth.json"))
    print(f"[tier B] {a.dir}: {truth['nobs']} events, "
          f"{truth['n_survey']} catalog galaxies, H0_true={truth['H0_true']}",
          flush=True)
    res = run(a.dir, grid=grid, soft_guard=a.soft_guard)
    res["truth"] = truth
    res["verdict"] = verdict(res)
    res["guard"] = dict(soft_guard=bool(a.soft_guard),
                        max_likelihood_variance=1e6,
                        convention="PR-0 clean arm" if not a.soft_guard
                        else "shipped-scan soft guard")
    print(json.dumps(res["verdict"], indent=2))
    with open(W16.PR6A_DIR / a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"[write] {W16.PR6A_DIR / a.out}")


if __name__ == "__main__":
    main()
