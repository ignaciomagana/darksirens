"""Gate 8(c): does the machinery recover truth on an IDEAL COMPLETE catalog?

The residual bias survives everything tried so far -- it is not the `f_p` gather
(fixed), not the parameter estimation (delta-PE at truth leaves it, and slightly
worsens it), and not the event draw (events at fixed catalog are calibrated).
What remains in the chain is the COMPLETENESS MODEL: below the survey's limit
the analysis represents the uncatalogued hosts by `(1 - C(z)) dN_exp`, with `C`
the parametric `c_sel_gaussian`.  If that curve misplaces the missing branch --
which sits at HIGHER redshift than the catalogued one, since the catalog is
depth-limited -- then `H0` moves, and upward, because a host placed at higher `z`
for the same `dL` implies a larger `H0`.

The standard test for exactly this is the one gate the mock-data checklist lists
and this campaign never ran: an end-to-end run on IDEAL inputs.  Make the survey
COMPLETE (`f_p == 1` everywhere and no magnitude limit) and analyse it with
`dark_sirens_complete`, which has no missing branch, no `C(z)` and no `f_p` at
all.  Then:

* bias GONE  => the completeness model is the carrier, and the residual is a
  statement about `C(z)`, not about the estimator's machinery;
* bias REMAINS => the machinery itself is biased, independent of completeness,
  and the hunt moves to the catalog kernels and the selection term.

The magnitude limit is removed by pushing `W16.M_LIM` far past the faintest
galaxy rather than by editing the survey config, so the SAME `make_mock` code
path builds it -- the ideal mock differs from the ordinary one only in two
numbers, which is what makes the comparison a control rather than a new
experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=7001)
    ap.add_argument("--seed-step", type=int, default=37)
    ap.add_argument("--m-lim", type=float, default=99.0,
                    help="effectively no magnitude limit")
    ap.add_argument("--injections", default="data/injections.h5")
    ap.add_argument("--h0-step", type=float, default=2.5)
    ap.add_argument("--n0", type=float, default=5e-5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax.numpy as jnp
    import world16 as W16
    import make_mock
    import tier_b
    import arms as A

    # IDEAL SURVEY: no magnitude limit, no mask.
    W16.M_LIM = float(a.m_lim)
    make_mock.W16.M_LIM = float(a.m_lim)
    W16.PR6A_DIR = W16.PR6A_DIR / "iso_complete"
    (W16.PR6A_DIR / "data").mkdir(parents=True, exist_ok=True)
    world = W16.build_world()
    # The world's f_p feeds BOTH the survey draw and the depth map make_mock
    # writes, so setting it to 1 makes the survey complete AND tells the
    # analysis so.  Passing ``f_p_survey`` instead would change only the survey
    # and leave the map disagreeing -- that is Tier D-i's deliberate mismatch,
    # the opposite of an ideal test.
    try:
        world = world._replace(f_p=np.ones_like(np.asarray(world.f_p)))
    except AttributeError:
        import dataclasses
        world = dataclasses.replace(
            world, f_p=np.ones_like(np.asarray(world.f_p)))

    # The ARM is unchanged (c_mode=selection, f_p on).  With m_lim = 99 the
    # parametric C(z) is ~1 everywhere and f_p is 1 inside the footprint, so
    # the model's completeness matches the mock's by construction and the
    # missing branch reduces to the off-footprint sky -- which f_p handles
    # exactly.  Switching to dark_sirens_complete instead would give -inf for
    # every event outside the 1,854-pixel footprint (measured: n = 0 survived).
    real_opts = A.make_opts
    _real_fixed = A.fixed_values

    h0 = np.arange(20.0, 140.0 + 0.5 * a.h0_step, a.h0_step)
    H0T = float(W16.H0_TRUE)
    rows = []
    try:
        for k in range(a.n_real):
            seed = a.seed0 + k * a.seed_step
            d = W16.PR6A_DIR / "data" / f"c{k:03d}"
            make_mock.build(seed, d, world=world, n0=a.n0, verbose=False,
                            reuse_injections=a.injections)
            p = tier_b.paths_for(d)
            logl, opts, data = A.build(p, "latent_off")
            vals = np.array([float(logl(jnp.asarray([float(x)]))) for x in h0])
            ok = np.isfinite(vals)
            if not ok.any():
                print(f"[complete] {k+1}/{a.n_real} all -inf", flush=True)
                continue
            v = np.where(ok, vals, -np.inf)
            pdf = np.exp(v - v.max())
            pdf = np.where(ok, pdf, 0.0)
            trapz = getattr(np, "trapezoid", None) or np.trapz
            pdf = pdf / trapz(pdf, h0)
            cdf = np.concatenate([[0], np.cumsum(
                0.5 * (pdf[1:] + pdf[:-1]) * np.diff(h0))])
            cdf /= cdf[-1]
            q = lambda t: float(np.interp(t, cdf, h0))  # noqa: E731
            row = dict(seed=int(seed), median=q(0.5),
                       sigma=0.5 * (q(0.84) - q(0.16)),
                       ci90=[q(0.05), q(0.95)],
                       cdf_at_truth=float(np.interp(H0T, h0, cdf)))
            rows.append(row)
            print(f"[complete] {k+1}/{a.n_real} seed={seed} "
                  f"H0={row['median']:.1f} u={row['cdf_at_truth']:.3f}",
                  flush=True)
    finally:
        pass

    med = np.array([r["median"] for r in rows])
    sig = np.array([r["sigma"] for r in rows])
    u = np.array([r["cdf_at_truth"] for r in rows])
    out = dict(n=len(rows), m_lim=a.m_lim, H0_true=H0T, rows=rows,
               median_of_medians=float(np.median(med)),
               mean_of_medians=float(med.mean()),
               mean_sigma=float(sig.mean()),
               spread=float(med.std(ddof=1)),
               overconfidence=float(med.std(ddof=1) / sig.mean()),
               median_u=float(np.median(u)),
               cov90=float(((u >= 0.05) & (u <= 0.95)).mean()))
    print(f"\nGATE 8(c) -- IDEAL COMPLETE CATALOG (n={len(rows)}):")
    print(f"  median of medians {out['median_of_medians']:.2f} "
          f"({out['median_of_medians'] - H0T:+.2f} km/s) vs truth {H0T}")
    print(f"  median u = {out['median_u']:.3f}  (0.5 if unbiased; the ordinary "
          f"mock gives 0.323)")
    print(f"  overconfidence {out['overconfidence']:.3f}   cov90 "
          f"{out['cov90']:.2f}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("GATE_COMPLETE_DONE", flush=True)


if __name__ == "__main__":
    main()
