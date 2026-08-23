"""Does an event's score depend on its COMPANY?  Leave-one-out, no capture.

The regrouping result implies it must: if each event's score were a function of
that event alone, rebuilding datasets from a shared pool would give `R = 1` by
construction, and it gives 5.57.  But the per-event capture cannot be used to
show it directly -- its order is data-dependent, which invalidated the first
attempt at exactly this statement.

Leave-one-out needs no capture.  For a dataset `D` and an event `e in D`,

    u_e(D) = d/dH0 [ logL(D) - logL(D \\ e) ]

is a difference of TOTAL log-likelihoods, so it is immune to whatever order the
event reduction runs in.  Evaluate the same event `e` inside two datasets that
share only `e` and a few others, and compare:

* `u_e(X) == u_e(Y)` to numerical precision  =>  the score is per-event, and the
  common mode must come from somewhere else entirely;
* `u_e(X) != u_e(Y)`                          =>  the estimator couples events,
  and the spread ACROSS datasets is the size of the coupling.

One term is known to differ by construction and is not the effect: the selection
correction is not linear in `N`, so the `60 -> 59` step carries a piece that is
common to every event of a dataset.  It cancels in the event-to-event SPREAD
within a dataset, and the run reports both the raw `u_e` and the
mean-subtracted `u_e - <u>_D`, which is the quantity `J_OPG` actually consumes.

Run it with `f_p` on and off (`--arm latent_off` / `latent_off_nofp`) and the
coupling's channel is read off directly.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from event_reshuffle import build_regrouped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--pattern", default="c00_e{:03d}")
    ap.add_argument("--n-sources", type=int, default=16)
    ap.add_argument("--nobs", type=int, default=60)
    ap.add_argument("--n-shared", type=int, default=12,
                    help="events common to both datasets -- the ones compared")
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--h0", type=float, default=68.0)
    ap.add_argument("--dh", type=float, default=2.0)
    ap.add_argument("--n-datasets", type=int, default=2,
                    help="datasets sharing the SAME n_shared events. With M > 2 "
                         "the across-dataset variance of the shared events' "
                         "mean LOO score estimates Var(c) directly, which is "
                         "the quantity R = 1 + N Var(c)/Var(a) needs.")
    ap.add_argument("--max-var", type=float, default=None,
                    help="override max_likelihood_variance. The RELIABILITY "
                         "GUARD is the only term in this likelihood that can "
                         "couple events: its threshold is a function of the "
                         "SET's total variance, so raising the budget until "
                         "the guard is inert tests whether the coupling is the "
                         "guard's rather than f_p's modelling.")
    ap.add_argument("--seed", type=int, default=515)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax.numpy as jnp
    import arms as A
    import tier_b

    if a.max_var is not None:
        _real = A.make_opts

        def _opts(paths, arm, **kw):
            kw.setdefault("max_var", a.max_var)
            return _real(paths, arm, **kw)

        A.make_opts = _opts

    sources = [Path(a.tree) / a.pattern.format(i) for i in range(a.n_sources)]
    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    hs = [a.h0 - a.dh, a.h0, a.h0 + a.dh]

    # The shared events come from source 0; each dataset's remaining 48 are
    # drawn from the OTHER sources, so X and Y have exactly ``n_shared`` events
    # in common and nothing else.
    shared = [(0, e) for e in range(a.n_shared)]
    pool = [(s, e) for s in range(1, a.n_sources) for e in range(a.nobs)]
    m_rest = a.nobs - a.n_shared
    pick = rng.choice(len(pool), size=a.n_datasets * m_rest, replace=False)
    labels = [chr(ord("X") + k) if k < 2 else f"D{k}"
              for k in range(a.n_datasets)]
    groups = {labels[k]: shared + [pool[i] for i in
                                   pick[k * m_rest:(k + 1) * m_rest]]
              for k in range(a.n_datasets)}

    def score_of(picks, tag):
        d = build_regrouped(sources, picks, work / tag)
        p = tier_b.paths_for(d)
        logl, opts, data = A.build(p, a.arm)
        vals = [float(logl(jnp.asarray([float(x)]))) for x in hs]
        return (vals[2] - vals[0]) / (2 * a.dh), vals[1]

    out = dict(arm=a.arm, h0=a.h0, dh=a.dh, max_var=a.max_var,
               n_shared=a.n_shared,
               nobs=a.nobs, datasets={})
    t0 = time.time()
    for name, picks in groups.items():
        full_score, full_ll = score_of(picks, f"{name}_full")
        u, lls = [], []
        for k in range(a.n_shared):
            drop = picks[:k] + picks[k + 1:]
            s_k, ll_k = score_of(drop, f"{name}_loo{k:02d}")
            u.append(full_score - s_k)
            lls.append(ll_k)
            print(f"[loo] {name} {k + 1}/{a.n_shared}  u={u[-1]:+.6f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        u = np.array(u)
        out["datasets"][name] = dict(
            full_score=float(full_score), full_logL=float(full_ll),
            u=[float(x) for x in u], u_mean=float(u.mean()),
            u_centred=[float(x) for x in (u - u.mean())])

    U = np.array([out["datasets"][l]["u"] for l in labels])   # (M, n_shared)
    if a.n_datasets > 2:
        # Var(c) DIRECTLY, by a TWO-WAY decomposition.  The same events appear in
        # every dataset, so
        #
        #     u[k, i] = mu + a_i + c_k + e[k, i]
        #
        # with a_i the event effect, c_k the dataset common mode we want, and e
        # the noise.  A dataset's mean is mu + abar + c_k + ebar_k, so
        #
        #     Var(row means) = Var(c) + Var(e) / n_shared
        #
        # and the debias needs Var(e) ALONE.
        #
        # THE PREVIOUS VERSION SUBTRACTED within.mean()/n_shared, where
        # ``within`` is the per-row variance ACROSS EVENTS and therefore
        # estimates Var(a) + Var(e).  Var(a) is the event-to-event spread, which
        # dominates here (0.1025 vs a row-mean sd of 0.0237), so the subtraction
        # removed far more than the sampling error and drove Var(c) NEGATIVE
        # (-3.2e-4), which then clamped to 0 and reported "no common mode,
        # R_predicted = 1.00".  That null was an artifact of the estimator and it
        # was used to retract a correct result.
        #
        # Var(e) is identified by the residual after removing BOTH effects --
        # the standard two-way ANOVA residual, on (M-1)(n-1) degrees of freedom.
        means = U.mean(axis=1)                       # (M,)  dataset means
        col = U.mean(axis=0)                         # (n,)  event means
        grand = float(U.mean())
        resid = U - means[:, None] - col[None, :] + grand
        M, n = U.shape
        dof = max((M - 1) * (n - 1), 1)
        var_e = float((resid ** 2).sum() / dof)
        var_c = float(means.var(ddof=1) - var_e / n)
        # Var(a): the event effect, debiased for its own sampling error.
        var_a_event = float(max(col.var(ddof=1) - var_e / M, 0.0))
        # The denominator R is built from is the WITHIN-DATASET event spread the
        # OPG estimator sees, which is Var(a) + Var(e) -- unchanged.
        var_a = float(U.var(axis=1, ddof=1).mean())
        # AND ITS POWER, because the point estimate alone was read twice as a
        # result and is not one: bootstrap over DATASETS (the replicated unit).
        _rng = np.random.default_rng(0)

        def _est(X):
            m = X.mean(axis=1)
            c = X.mean(axis=0)
            r = X - m[:, None] - c[None, :] + float(X.mean())
            ve = float((r ** 2).sum() / max((X.shape[0] - 1) * (n - 1), 1))
            vc = float(m.var(ddof=1) - ve / n)
            va = float(X.var(axis=1, ddof=1).mean())
            return vc, 1.0 + a.nobs * max(vc, 0.0) / va

        _R = np.array([_est(U[_rng.integers(0, M, M)])[1] for _ in range(4000)])
        out.update(var_e_residual=var_e, var_a_event_only=var_a_event,
                   R_bootstrap_ci90=[float(np.percentile(_R, 5)),
                                     float(np.percentile(_R, 95))],
                   R_bootstrap_median=float(np.median(_R)))
        out.update(
            n_datasets=a.n_datasets,
            dataset_mean_u=[float(x) for x in means],
            var_c_debiased=var_c, var_a=var_a,
            sd_ratio_c_over_a=float(np.sqrt(max(var_c, 0.0) / var_a)),
            # what R this common mode PREDICTS at the tier's own N
            R_predicted=float(1.0 + a.nobs * max(var_c, 0.0) / var_a))
        print(f"\nVar(c) FROM {a.n_datasets} DATASETS, {a.n_shared} shared "
              f"events (arm={a.arm}):")
        print(f"  dataset means of u: sd {means.std(ddof=1):.6f}   "
              f"within-dataset sd {np.sqrt(var_a):.6f}")
        print(f"  Var(c) debiased = {var_c:.3e}   sd(c)/sd(a) = "
              f"{out['sd_ratio_c_over_a']:.3f}")
        print(f"  => R predicted at N={a.nobs}: {out['R_predicted']:.2f} "
              f"(bootstrap 90% CI [{out['R_bootstrap_ci90'][0]:.2f}, "
              f"{out['R_bootstrap_ci90'][1]:.2f}] over {M} datasets)")
        print("     If that CI spans both 1 and the capture route's value, this "
              "probe is UNINFORMATIVE at this M -- not a disagreement.")
        print("     (R measured by event_reshuffle.py: 5.57 with f_p at its "
              "peak, 0.98 without at its own -- compare to the ARM being run)")
    uX, uY = U[0], U[1]
    cX, cY = uX - uX.mean(), uY - uY.mean()
    out.update(
        delta_raw=[float(x) for x in (uY - uX)],
        delta_raw_mean=float((uY - uX).mean()),
        delta_raw_sd=float((uY - uX).std(ddof=1)),
        delta_centred=[float(x) for x in (cY - cX)],
        delta_centred_sd=float((cY - cX).std(ddof=1)),
        within_dataset_sd=float(0.5 * (cX.std(ddof=1) + cY.std(ddof=1))),
        corr_XY=float(np.corrcoef(uX, uY)[0, 1]),
        # the number that matters: how big is the company effect against the
        # event-to-event spread J_OPG is built from?
        coupling_over_spread=float(
            (cY - cX).std(ddof=1) / (0.5 * (cX.std(ddof=1) + cY.std(ddof=1)))))
    print(f"\nSAME {a.n_shared} EVENTS, TWO DATASETS (arm={a.arm}):")
    print(f"  raw u_e:      delta mean {out['delta_raw_mean']:+.6f}  "
          f"sd {out['delta_raw_sd']:.6f}   (the N-dependent common piece is in "
          f"the mean)")
    print(f"  centred u_e:  delta sd {out['delta_centred_sd']:.6f} against a "
          f"within-dataset spread of {out['within_dataset_sd']:.6f}")
    print(f"  ratio = {out['coupling_over_spread']:.3f}   corr(X, Y) = "
          f"{out['corr_XY']:+.3f}")
    print("  ratio ~ 0 => the score is per-event.  ratio ~ 1 => an event's "
          "score is as much its company as itself.")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("LOO_COUPLING_DONE", flush=True)


if __name__ == "__main__":
    main()
