"""Is the single-dataset OPG estimator of ``J`` unbiased?  Calibrated on the mock.

Production has no ensemble, so its ``J`` is estimated from ONE dataset by the
outer product of per-event gradients (``desi_full259/production_opg.py``).  That
estimator assumes the per-event scores are independent: it computes
``sum_i (u_i - ubar)^2`` and drops every ``Cov(u_i, u_j)``.  Catalog structure
correlates events that share it, and positive correlation only RAISES ``J``, so
the OPG number has always been quoted as a LOWER BOUND -- with no measurement of
how loose the bound is.

The mock can measure it, because there the ensemble is available.  Hold the
catalog FIXED and vary only the event draw (``make_mock.build(seed,
event_seed=e)`` re-seeds the stream immediately before the events, the same
split ``variance_split.py`` uses).  Then

    J_ensemble(catalog) = Var_e( dlogL/dH0 )          -- the truth, at fixed catalog
    J_OPG(e)            = sum_i (u_i - ubar)^2 N/(N-1) -- what production can compute

and their ratio ``R = J_ensemble / mean_e J_OPG`` is exactly the factor the
production estimator misses.  ``R ~ 1`` says the bound is tight and production's
``J/H ~ 1`` stands as a measurement; ``R >> 1`` says production's ratio must be
scaled up before it is compared with the mock's 6.03.

The comparison is at FIXED CATALOG on both sides deliberately: the analysis
conditions on the catalog it observed, and it is the event-conditional variance
that the quoted ``sigma`` claims to be.

The per-event capture and its two checks (completeness, ordering) are ported
from ``production_opg.py`` unchanged in substance -- including the correction
attribution, which is what made the ordering check pass there: the ACTUAL
correction ``logL - sum_i ll_i`` is spread ``1/N`` per event, not the idealised
``-N log mu``, because the shipped correction also carries the Vitale ``5 N_obs``
floor and the variance criterion.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalogs", type=int, default=3)
    ap.add_argument("--events", type=int, default=16)
    ap.add_argument("--seed0", type=int, default=9000)
    ap.add_argument("--seed-step", type=int, default=137)
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--h0", type=float, default=68.0)
    ap.add_argument("--dh", type=float, default=2.0)
    ap.add_argument("--injections", default="data/injections.h5")
    ap.add_argument("--tag", default="iso_opg",
                    help="private realization tree (two concurrent runs must "
                         "not share one)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import make_mock
    import world16 as W16
    import arms as A
    import tier_b
    import darksirens.likelihood.core as _core
    from darksirens.likelihood import selection as _sel

    W16.PR6A_DIR = W16.PR6A_DIR / a.tag
    (W16.PR6A_DIR / "data").mkdir(parents=True, exist_ok=True)
    world = W16.build_world()
    hs = [a.h0 - a.dh, a.h0, a.h0 + a.dh]

    ev_store, mu_store = [], []
    real_ev = _core.log_evidence_and_mc_variance
    real_corr = _sel.selection_log_correction

    def _spy_ev(ldw, nsamp):
        ll, var = real_ev(ldw, nsamp)
        jax.debug.callback(lambda x: ev_store.append(
            np.asarray(x).ravel().copy()), ll)
        return ll, var

    def _spy_corr(log_mu, Neff, nEvents, **kw):
        jax.debug.callback(lambda m, n: mu_store.append(float(m)),
                           log_mu, Neff)
        return real_corr(log_mu, Neff, nEvents, **kw)

    rows, t0 = [], time.time()
    _core.log_evidence_and_mc_variance = _spy_ev
    _core.selection_log_correction = _spy_corr
    _sel.selection_log_correction = _spy_corr
    try:
        for c in range(a.catalogs):
            seed = a.seed0 + c * a.seed_step
            for e in range(a.events):
                d = W16.PR6A_DIR / "data" / f"c{c:02d}_e{e:03d}"
                make_mock.build(seed, d, world=world, verbose=False,
                                reuse_injections=a.injections, event_seed=e)
                p = tier_b.paths_for(d)
                logl, opts, data = A.build(p, a.arm)
                LL, EV = [], []
                for x in hs:
                    ev_store.clear(); mu_store.clear()
                    v = float(logl(jnp.asarray([float(x)])))
                    LL.append(v)
                    EV.append(np.concatenate(ev_store) if ev_store
                              else np.array([]))
                N = int(data["nEvents"])
                bad = [f"node {k}: captured {EV[k].size} != {N}"
                       for k in range(3) if EV[k].size != N]
                row = dict(catalog=c, seed=int(seed), event_seed=int(e),
                           n_events=N, logL=[float(x) for x in LL],
                           checks_failed=bad)
                if not bad:
                    corr = np.array([LL[k] - EV[k].sum() for k in range(3)])
                    dcorr = (corr[2] - corr[0]) / (2 * a.dh)
                    u = (EV[2] - EV[0]) / (2 * a.dh) + dcorr / N
                    score = (LL[2] - LL[0]) / (2 * a.dh)
                    rel = abs(float(u.sum()) - score) / max(abs(score), 1e-12)
                    dll = (EV[2] - EV[0]) / (2 * a.dh)
                    row.update(
                        score=float(score),
                        H=float(-(LL[2] - 2 * LL[1] + LL[0]) / a.dh ** 2),
                        J_opg=float(np.sum((u - u.mean()) ** 2) * N / (N - 1)),
                        ordering_rel=float(rel),
                        # the decomposition that says WHY J_OPG can be biased:
                        # the event-term score, the correction's derivative
                        # (a common mode across events, removed by the OPG
                        # centring), and the within-dataset event dispersion.
                        event_score=float(dll.sum()),
                        dcorr=float(dcorr),
                        var_within_events=float(dll.var(ddof=1)),
                        n_events_used=int(dll.size))
                    if rel > 0.02:
                        row["checks_failed"] = [
                            f"ordering check failed at {100 * rel:.1f}%"]
                rows.append(row)
                print(f"[opg] cat {c} evt {e:3d}  score="
                      f"{row.get('score', float('nan')):+.5f}  J_opg="
                      f"{row.get('J_opg', float('nan')):.6f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        _core.log_evidence_and_mc_variance = real_ev
        _core.selection_log_correction = real_corr
        _sel.selection_log_correction = real_corr

    good = [r for r in rows if not r["checks_failed"]]
    per_cat, pooled_R = [], []
    for c in range(a.catalogs):
        g = [r for r in good if r["catalog"] == c]
        if len(g) < 3:
            continue
        sc = np.array([r["score"] for r in g])
        jo = np.array([r["J_opg"] for r in g])
        H = float(np.mean([r["H"] for r in g]))
        J_ens = float(sc.var(ddof=1))
        R = J_ens / float(jo.mean())
        ev = np.array([r["event_score"] for r in g])
        dc = np.array([r["dcorr"] for r in g])
        vw = np.array([r["var_within_events"] for r in g])
        Nev = int(g[0]["n_events_used"])
        iid_pred = Nev * float(vw.mean())      # what i.i.d. events would give
        per_cat.append(dict(catalog=c, n=len(g), J_ensemble=J_ens,
                            J_opg_mean=float(jo.mean()),
                            J_opg_sd=float(jo.std(ddof=1)),
                            ratio_R=R, H_mean=H,
                            J_ens_over_H=J_ens / H if H else None,
                            J_opg_over_H=float(jo.mean()) / H if H else None,
                            var_event_score=float(ev.var(ddof=1)),
                            var_dcorr=float(dc.var(ddof=1)),
                            cov_event_dcorr=float(
                                np.cov(ev, dc, ddof=1)[0, 1]),
                            iid_prediction=iid_pred,
                            event_score_over_iid=(
                                float(ev.var(ddof=1)) / iid_pred
                                if iid_pred else None),
                            n_events=Nev))
        pooled_R.append(R)
        print(f"\n  CATALOG {c} (n={len(g)}): J_ensemble={J_ens:.6f}  "
              f"mean J_OPG={jo.mean():.6f}  R={R:.3f}   "
              f"J_ens/H={J_ens / H:.3f} vs J_OPG/H={jo.mean() / H:.3f}",
              flush=True)
        print(f"      decomposition: Var(event score)={ev.var(ddof=1):.6f} "
              f"vs i.i.d. prediction N*E[var_within]={iid_pred:.6f} "
              f"(ratio {ev.var(ddof=1) / iid_pred:.2f});  "
              f"Var(dcorr)={dc.var(ddof=1):.6f};  "
              f"Cov={np.cov(ev, dc, ddof=1)[0, 1]:+.6f}", flush=True)
    out = dict(arm=a.arm, h0=a.h0, dh=a.dh, n_catalogs=a.catalogs,
               n_events_per_catalog=a.events, rows=rows, per_catalog=per_cat,
               R_median=float(np.median(pooled_R)) if pooled_R else None,
               R_min=float(np.min(pooled_R)) if pooled_R else None,
               R_max=float(np.max(pooled_R)) if pooled_R else None,
               n_failed=len(rows) - len(good))
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    if pooled_R:
        print(f"\nOPG CALIBRATION: R = J_ensemble / J_OPG = "
              f"{np.median(pooled_R):.3f} (median over {len(pooled_R)} "
              f"catalogs, range {np.min(pooled_R):.3f}-{np.max(pooled_R):.3f})")
        print("  R ~ 1: the production OPG bound is tight.  R >> 1: production's "
              "J/H is an underestimate by that factor.")
    print(f"[wrote] {a.out}")
    print("OPG_CALIB_DONE", flush=True)


if __name__ == "__main__":
    main()
