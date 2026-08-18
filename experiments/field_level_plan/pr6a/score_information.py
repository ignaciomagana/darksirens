"""The information identity, decomposed -- which term breaks it.

For a correctly specified model the OBSERVED information
``H = -d2 logL/dH0^2`` and the EXPECTED information ``J = Var(dlogL/dH0)``
agree, and the MLE variance is ``H^-1``.  When they disagree the correct
variance is the sandwich ``H^-1 J H^-1``, and the naive interval is too narrow
by ``sqrt(J/H)``.  Tier C's ~2.3-2.9x overconfidence is exactly that statement,
so ``J/H`` should be ~5-8.

What makes this a DISCRIMINATOR rather than a restatement: the injection set is
SHARED across realizations, so the selection term's score
``-N d log mu/dH0`` is IDENTICAL in every realization and contributes **zero**
variance.  Therefore

    J = Var over realizations of the EVENT score alone
    H = |curv(events)| - |curv(selection)|      (the 65% cancellation)

so J is a pure event-term quantity while H is the cancelled net.  Computing
both directly says whether the event term carries more information than its own
curvature implies (a misspecified redshift prior) or whether the cancellation
is what breaks the identity.

Stores the full ``logL(H0)`` curve for every realization so the score and the
curvature are both computed from the same object, rather than one being
inferred from the spread of the other.
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
    ap.add_argument("--injections", default="data/injections.h5")
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--h0-lo", type=float, default=40.0)
    ap.add_argument("--h0-hi", type=float, default=110.0)
    ap.add_argument("--h0-step", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import make_mock
    import world16 as W16
    import arms as A
    import tier_b

    # private tree: tier scripts share W16.PR6A_DIR/"data"/... (see
    # tier_c_truncated.py for the collision this avoids)
    W16.PR6A_DIR = W16.PR6A_DIR / "iso_score"
    (W16.PR6A_DIR / "data").mkdir(parents=True, exist_ok=True)

    world = W16.build_world()
    h0 = np.arange(a.h0_lo, a.h0_hi + 0.5 * a.h0_step, a.h0_step)
    rows = []
    for k in range(a.n_real):
        seed = a.seed0 + k * a.seed_step
        d = W16.PR6A_DIR / "data" / f"s{k:03d}"
        make_mock.build(seed, d, world=world, verbose=False,
                        reuse_injections=a.injections)
        p = tier_b.paths_for(d)
        logl, opts, data = A.build(p, a.arm)
        import jax.numpy as jnp
        vals = np.array([float(logl(jnp.asarray([float(x)]))) for x in h0])
        rows.append(dict(seed=int(seed), logl=vals.tolist()))
        print(f"[score] {k+1}/{a.n_real} seed={seed} "
              f"peak H0={h0[int(np.nanargmax(vals))]:.0f}", flush=True)

    L = np.array([r["logl"] for r in rows])
    ok = np.all(np.isfinite(L), axis=0)
    hh, LL = h0[ok], L[:, ok]
    truth = float(W16.H0_TRUE) if hasattr(W16, "H0_TRUE") else 67.74
    i = int(np.argmin(np.abs(hh - truth)))
    i = min(max(i, 1), hh.size - 2)
    s = hh[1] - hh[0]

    score = (LL[:, i + 1] - LL[:, i - 1]) / (2 * s)          # dlogL/dH0 at truth
    obs = -(LL[:, i + 1] - 2 * LL[:, i] + LL[:, i - 1]) / s ** 2   # H per realization
    J = float(score.var(ddof=1))
    H = float(obs.mean())
    out = dict(arm=a.arm, n=len(rows), h0_at=float(hh[i]),
               J_expected_information=J, H_observed_information=H,
               ratio_J_over_H=J / H if H else None,
               implied_width_inflation=float(np.sqrt(J / H)) if H > 0 else None,
               score_mean=float(score.mean()), score_sd=float(score.std(ddof=1)),
               H_sd=float(obs.std(ddof=1)), h0=hh.tolist(),
               rows=rows)
    print(f"\nINFORMATION IDENTITY at H0={hh[i]:.1f} (arm={a.arm}, n={len(rows)}):")
    print(f"  J (expected, = Var of the score) = {J:.6f}")
    print(f"  H (observed, = mean curvature)   = {H:.6f}")
    print(f"  J/H = {J/H:.3f}   -> intervals too narrow by x{np.sqrt(J/H):.3f}")
    print(f"  (Tier C measured x2.1-2.9 directly from the scatter)")
    print(f"  score mean={score.mean():+.4f} sd={score.std(ddof=1):.4f}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("SCORE_INFO_DONE", flush=True)


if __name__ == "__main__":
    main()
