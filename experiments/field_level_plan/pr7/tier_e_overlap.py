#!/usr/bin/env python
"""Tier E, overlap arm — R14 measured rather than stipulated (field-level PR-7).

PLAN's risk table gives R14 ("tracer overlap at K >= 2, AGN subset of
galaxies") the mitigation "OWNER DECISION 9: disjoint partition" and the
detection "Tier E on an OVERLAPPING mock".  Disjointness is structural
everywhere PR-7 builds counts -- `counts_from_catalog_by_tracer` splits on a
single-valued per-galaxy label, and `tier_e.py` draws K independent
multinomials -- so an overlapping world has to be constructed on purpose.

Here tracer 2 is `(1 - phi)` fresh galaxies drawn at `b_2` plus `phi` galaxies
drawn WITHOUT REPLACEMENT out of tracer 1's own counts (a multivariate
hypergeometric per shell, which is exactly "the same objects, re-observed").
Those shared galaxies are distributed as `b_1`, and the stacked objective adds
both copies of them.  `phi = 0` reproduces `tier_e.py`'s disjoint draw bit for
bit, so the arm is a controlled deformation of the gate, not a second
experiment.

Predictions P-E6 and P-E7 are in `PREDICTION.md`, written before this ran.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tier_e import build_world, make_stack                      # noqa: E402


def draw_overlapping(world, seed, biases, phi):
    """Tracer 1 disjointly; tracer 2 with a fraction `phi` shared with it."""
    import jax.numpy as jnp

    basis, W = world["basis"], world["W"]
    rng = np.random.default_rng(seed)
    xi_true = rng.normal(size=basis.rank)
    phi_shell = np.asarray(jnp.asarray(W) @ basis.phi_z_fine)
    proj = np.asarray(basis.proj_sph)
    f_true = (proj @ xi_true.reshape(basis.m_sph, basis.m_z)) @ phi_shell.T

    def _pi(k, b):
        a = np.log(world["f_p"][k])[:, None] + b * f_true
        p = np.exp(a - a.max(axis=0, keepdims=True))
        return p / p.sum(axis=0, keepdims=True)

    G = f_true.shape[1]
    p1, p2 = _pi(0, biases[0]), _pi(1, biases[1])
    c1 = np.stack([rng.multinomial(int(world["T_g"][g]), p1[:, g])
                   for g in range(G)], axis=1).astype(float)
    c2 = np.zeros_like(c1)
    for g in range(G):
        T = int(world["T_g"][g])
        n_share = int(round(phi * T))
        if n_share:
            # Without replacement out of tracer 1's OWN galaxies in this shell:
            # the shared objects are the same objects, so they carry b_1.
            c2[:, g] += rng.multivariate_hypergeometric(
                c1[:, g].astype(np.int64), n_share).astype(float)
        if T - n_share:
            c2[:, g] += rng.multinomial(T - n_share, p2[:, g]).astype(float)
    return xi_true, [c1, c2]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    rb = os.path.join(here, "..", "pr6a", "data", "rb")
    ap.add_argument("--survey", default=os.path.join(rb, "catalog_pixelated_nside_16.h5"))
    ap.add_argument("--mth-map", default=os.path.join(rb, "mth_map_nside16.h5"))
    ap.add_argument("--out", default=os.path.join(here, "tier_e_overlap.json"))
    ap.add_argument("--overlap", type=float, nargs="*",
                    default=[0.0, 0.05, 0.10, 0.25, 0.50])
    ap.add_argument("--n-real", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=880000)
    ap.add_argument("--seed-step", type=int, default=137)
    ap.add_argument("--z-depth", type=float, default=0.30)
    ap.add_argument("--n-shells", type=int, default=12)
    ap.add_argument("--n-gal", type=int, default=180000)
    ap.add_argument("--m-sph", type=int, default=40)
    ap.add_argument("--m-z", type=int, default=8)
    ap.add_argument("--ls-sph", type=float, default=0.6)
    ap.add_argument("--ls-z", type=float, default=0.05)
    ap.add_argument("--sigma-z", type=float, default=0.023)
    ap.add_argument("--b1", type=float, default=1.0)
    ap.add_argument("--ratio", type=float, default=2.0)
    ap.add_argument("--b-start", type=float, default=1.0)
    ap.add_argument("--fp2-power", type=float, default=1.6)
    ap.add_argument("--fp2-scale", type=float, default=0.85)
    ap.add_argument("--n-outer", type=int, default=30,
                    help="outer profile-Newton trips. 30, not 10: at 10 the "
                         "ratio direction is NOT converged and the error is "
                         "large and silent -- measured on overlap seed 880000, "
                         "n_outer=10 returned r = 1.940019 with a profile "
                         "gradient of -1816 nat per unit log-ratio, and "
                         "n_outer=20 returned r = 1.996658 at gradient 0.000. "
                         "The residual gradient is now reported per "
                         "realization and gated by --profile-grad-tol.")
    ap.add_argument("--max-log-step", type=float, default=1.0,
                    help="trust region on the outer step, nat per trip. The "
                         "bias AMPLITUDE is soft and travels far (the counts "
                         "put it near b ~ 50 on this world, against a prior "
                         "centred at b = 1), so 0.5 nat/trip spends most of "
                         "the budget travelling rather than converging.")
    ap.add_argument("--profile-grad-tol", type=float, default=1e-3,
                    help="max |dP/du| accepted at the returned point. A "
                         "realization above it is reported as NOT CONVERGED "
                         "rather than quoted.")
    ap.add_argument("--prior-s", type=float, default=1.0)
    args = ap.parse_args(argv)

    import jax
    jax.config.update("jax_enable_x64", True)
    from darksirens.redshift.latent_counts import (
        bias_profile, bias_ratio_from_profile)

    world = build_world(args)
    rows = []
    seeds = [args.seed0 + args.seed_step * k for k in range(args.n_real)]
    for phi in args.overlap:
        for s in seeds:
            _, ck = draw_overlapping(world, s, [args.b1, args.b1 * args.ratio],
                                     phi)
            mop = make_stack(world, ck, [args.b_start, args.b_start])
            t0 = time.time()
            sh = bias_profile(mop, log_b_prior=(0.0, args.prior_s),
                              n_outer=args.n_outer,
                              max_log_step=args.max_log_step)
            rr = bias_ratio_from_profile(sh["b_hat"], sh["cov_log_b"])
            gu = np.asarray(sh["profile_grad_log"])
            rows.append(dict(overlap=float(phi), seed=int(s), **rr,
                             pull=float((rr["ratio"] - args.ratio)
                                        / rr["sigma"]),
                             profile_grad_log=gu.tolist(),
                             # Split into the two directions that mean
                             # something: the RATIO gradient is the one the
                             # gate depends on, and the AMPLITUDE gradient is
                             # the soft one that takes the trips.
                             grad_ratio=float((gu[1] - gu[0]) / 2.0),
                             grad_amp=float((gu[1] + gu[0]) / 2.0),
                             converged=bool(np.max(np.abs(gu))
                                            < args.profile_grad_tol),
                             wall_s=time.time() - t0))
        sub = [r for r in rows if r["overlap"] == phi]
        pulls = np.array([r["pull"] for r in sub])
        print(f"[overlap] phi = {phi:5.2f}  r = "
              f"{np.mean([r['ratio'] for r in sub]):.5f} +- "
              f"{np.std([r['ratio'] for r in sub], ddof=1):.5f} (scatter), "
              f"quoted sigma {np.mean([r['sigma'] for r in sub]):.5f}, "
              f"pull {pulls.mean():+.3f} +- {pulls.std(ddof=1):.3f}, "
              f"|pull| < 2 in {int(np.sum(np.abs(pulls) < 2))}/{len(sub)}, "
              f"converged {int(sum(r['converged'] for r in sub))}/{len(sub)} "
              f"(max |grad_ratio| "
              f"{np.max(np.abs([r['grad_ratio'] for r in sub])):.2e})",
              flush=True)

    summary = []
    for phi in args.overlap:
        sub = [r for r in rows if r["overlap"] == phi]
        pulls = np.array([r["pull"] for r in sub])
        summary.append(dict(
            overlap=float(phi), n=len(sub),
            ratio_mean=float(np.mean([r["ratio"] for r in sub])),
            ratio_sd=float(np.std([r["ratio"] for r in sub], ddof=1)),
            sigma_mean=float(np.mean([r["sigma"] for r in sub])),
            pull_mean=float(pulls.mean()), pull_sd=float(pulls.std(ddof=1)),
            within_2sigma=int(np.sum(np.abs(pulls) < 2)),
            n_converged=int(sum(r["converged"] for r in sub)),
            grad_ratio_max=float(np.max(np.abs(
                [r["grad_ratio"] for r in sub])))))
    print("[summary] " + json.dumps(summary, indent=1), flush=True)
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), summary=summary, rows=rows), f,
                  indent=1, default=str)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
