"""The decisive test: regroup the SAME events and see if the anomaly survives.

At fixed catalog the per-event scores carry a per-dataset common mode: the
dataset means scatter ~6x more than i.i.d. events allow (`opg_calibration.py`,
bootstrap `p < 5e-4`, survives delta-PE, capture verified).  Two explanations
remain and they license opposite conclusions:

* the mock's event DRAW does not produce independent events -- a generator
  property, and Tier C would be measuring the mock;
* the per-event score has a DATASET-level dependence -- a property of the
  likelihood, which is what Tier C's width deficit is about.

Regrouping separates them without changing a single event.  Pool the events of
`M` datasets that share one catalog, then build new datasets by drawing 60
events from that pool -- deliberately spanning source datasets, so whatever the
generator's grouping carried is destroyed while the event population is
IDENTICAL by construction.

Two measurements, and the first is the sharper one:

**(1) Per-event invariance.**  Re-run one regrouped dataset and compare each
event's captured log-evidence with the value it had in its ORIGINAL dataset.
If an event's score depends on the company it keeps, the estimator couples
events and the anomaly is the likelihood's.  If the values are identical, the
estimator is per-event and (2) must return `R = 1`.

**(2) `R` over the regrouped ensemble.**  `R = Var(score) / mean(J_OPG)` exactly
as `opg_calibration.py` computes it.  `R ~ 1` puts the common mode in the event
draw; `R ~ 6` puts it in the estimator, and then (1) says where.

The two are a consistency pair: (1) predicts (2).  Reporting both means a
disagreement between them is visible rather than absorbed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

#: Files a realization directory needs besides ``gw_events.h5``; all are
#: functions of the CATALOG seed alone, so they are shared by every dataset in
#: the pool and can be linked rather than copied.
_SHARED = ("catalog_pixelated_nside_16.h5", "gw_selection.h5",
           "mth_map_nside16.h5", "selection_fit.json", "n0_calibration.json",
           "truth.json")


def _read_events(path):
    """Every per-event array of a mock GW file, unflattened to (n, nsamp)."""
    with h5py.File(path) as f:
        n, ns = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
        samples = {k: f[k][...].reshape(n, ns) for k in f
                   if k != "truth" and isinstance(f[k], h5py.Dataset)}
        truth = {k: f["truth"][k][...] for k in f["truth"]}
        attrs = dict(f.attrs)
    return samples, truth, n, ns, attrs


def _write_events(path, samples, truth, attrs, n, ns):
    with h5py.File(path, "w") as f:
        for k, v in attrs.items():
            f.attrs[k] = v
        f.attrs["nobs"] = int(n)
        f.attrs["nsamp"] = int(ns)
        for k, v in samples.items():
            f.create_dataset(k, data=np.asarray(v).reshape(-1))
        g = f.create_group("truth")
        for k, v in truth.items():
            g.create_dataset(k, data=np.asarray(v))


def build_regrouped(sources, picks, outdir):
    """One regrouped dataset from ``picks`` = [(source index, event index), …]."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    src0 = Path(sources[0])
    for name in _SHARED:
        dst = outdir / name
        if not dst.exists():
            shutil.copyfile(src0 / name, dst)
    cache = {}
    rows_s, rows_t = [], []
    for si, ei in picks:
        if si not in cache:
            cache[si] = _read_events(Path(sources[si]) / "gw_events.h5")
        samples, truth, n, ns, attrs = cache[si]
        rows_s.append({k: v[ei] for k, v in samples.items()})
        rows_t.append({k: v[ei] for k, v in truth.items()})
    samples0, truth0, n0, ns, attrs = cache[picks[0][0]]
    out_s = {k: np.stack([r[k] for r in rows_s]) for k in samples0}
    out_t = {k: np.array([r[k] for r in rows_t]) for k in truth0}
    _write_events(outdir / "gw_events.h5", out_s, out_t, attrs,
                  len(picks), ns)
    return outdir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True,
                    help="directory holding the source realizations")
    ap.add_argument("--pattern", default="c00_e{:03d}")
    ap.add_argument("--n-sources", type=int, default=16)
    ap.add_argument("--n-regrouped", type=int, default=16)
    ap.add_argument("--nobs", type=int, default=60)
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--sky-weighting", default=None,
                    choices=["field", "conditional"],
                    help="bisect: 'conditional' drops the survey-global field "
                         "normalizer, so a coupling that lives in the field "
                         "convention disappears while a per-pixel one does not")
    ap.add_argument("--h0", type=float, default=68.0)
    ap.add_argument("--dh", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import arms as A
    import tier_b
    import darksirens.likelihood.core as _core
    from darksirens.likelihood import selection as _sel

    tree = Path(a.tree)
    sources = [tree / a.pattern.format(i) for i in range(a.n_sources)]
    for s in sources:
        if not (s / "gw_events.h5").exists():
            raise SystemExit(f"missing {s/'gw_events.h5'}")
    work = Path(a.workdir or (tree.parent / "regrouped"))
    work.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(a.seed)
    # THE NULL CONTROL, and it must run before anything is concluded from the
    # invariance check: rebuild source 0 from its own events, in its own order.
    # Every per-event log-evidence must then come back IDENTICAL. If it does
    # not, the round trip (file write, event indexing, capture order) is at
    # fault and no statement about regrouping is licensed.
    control = [(0, e) for e in range(a.nobs)]
    # THE SECOND CONTROL, and the first one does not substitute for it: the
    # identity rebuild passes even if the loader REORDERS events, because the
    # same 60 events reorder the same way.  So also rebuild source 0 with its
    # events REVERSED.  If the capture follows file order the values come back
    # reversed; if they come back in the original order the loader sorts, and
    # the (source, event) -> capture-slot mapping the invariance check assumes
    # is wrong.
    permuted = [(0, e) for e in reversed(range(a.nobs))]
    # Draw 60 (source, event) pairs per regrouped dataset from the whole pool,
    # without replacement WITHIN a dataset.  Spanning sources is the point: a
    # regrouped dataset holds events from ~every source realization, so the
    # generator's grouping cannot survive.
    pool = [(s, e) for s in range(a.n_sources) for e in range(a.nobs)]
    groups = []
    for k in range(a.n_regrouped):
        idx = rng.choice(len(pool), size=a.nobs, replace=False)
        groups.append([pool[i] for i in idx])
    span = [len({s for s, _ in g}) for g in groups]
    print(f"[regroup] {a.n_regrouped} datasets of {a.nobs} events; each spans "
          f"{min(span)}-{max(span)} of the {a.n_sources} source realizations",
          flush=True)

    ev_store = []
    real_ev = _core.log_evidence_and_mc_variance
    real_corr = _sel.selection_log_correction

    def _spy_ev(ldw, nsamp):
        ll, var = real_ev(ldw, nsamp)
        jax.debug.callback(
            lambda x: ev_store.append(np.asarray(x).ravel().copy()), ll)
        return ll, var

    hs = [a.h0 - a.dh, a.h0, a.h0 + a.dh]

    real_make_opts = A.make_opts

    def _opts_weighted(paths, arm, **kw):
        o, sel = real_make_opts(paths, arm, **kw)
        if a.sky_weighting:
            o.catalog_sky_weighting = a.sky_weighting
        return o, sel

    A.make_opts = _opts_weighted

    def run_one(d):
        p = tier_b.paths_for(d)
        logl, opts, data = A.build(p, a.arm)
        LL, EV = [], []
        for x in hs:
            ev_store.clear()
            v = float(logl(jnp.asarray([float(x)])))
            LL.append(v)
            EV.append(np.concatenate(ev_store) if ev_store else np.array([]))
        return LL, EV, int(data["nEvents"])

    rows, invariance = [], None
    t0 = time.time()
    _core.log_evidence_and_mc_variance = _spy_ev
    _core.selection_log_correction = real_corr
    try:
        # --- (1) per-event invariance, on the SOURCES then on one regrouping
        src_ll = {}
        for si in range(a.n_sources):
            LL, EV, N = run_one(sources[si])
            if EV[1].size == N:
                src_ll[si] = EV[1].copy()
            print(f"[source] {si+1}/{a.n_sources} logL={LL[1]:.6f}", flush=True)

        dc = build_regrouped(sources, control, work / "control")
        LLc, EVc, Nc = run_one(dc)
        want_c = src_ll[0]
        got_c = EVc[1]
        dev_c = np.abs(got_c - want_c) if got_c.size == want_c.size else None
        control_check = dict(
            n=int(want_c.size), logL_source=None, logL_control=float(LLc[1]),
            max_abs=(float(dev_c.max()) if dev_c is not None else None),
            n_exact=(int((dev_c == 0).sum()) if dev_c is not None else None),
            matched_as_multiset=(bool(np.allclose(
                np.sort(got_c), np.sort(want_c), rtol=0, atol=1e-9))
                if dev_c is not None else None))
        print(f"\n[CONTROL] rebuild of source 0 from its own events: "
              f"max |delta| = {control_check['max_abs']:.3e} "
              f"({control_check['n_exact']} exact of {control_check['n']}), "
              f"multiset {control_check['matched_as_multiset']}", flush=True)

        dp = build_regrouped(sources, permuted, work / "permuted")
        LLp, EVp, Np = run_one(dp)
        got_p = EVp[1]
        want_p = want_c[::-1]
        permute_check = dict(
            n=int(got_p.size),
            follows_file_order=bool(np.array_equal(got_p, want_p)),
            follows_source_order=bool(np.array_equal(got_p, want_c)),
            max_abs_vs_file_order=float(np.abs(got_p - want_p).max()),
            max_abs_vs_source_order=float(np.abs(got_p - want_c).max()),
            logL_same=bool(abs(LLp[1] - LLc[1]) < 1e-9),
            logL_delta=float(LLp[1] - LLc[1]))
        print(f"[CONTROL 2] source 0 with events REVERSED: capture follows "
              f"file order = {permute_check['follows_file_order']}, "
              f"source order = {permute_check['follows_source_order']}; "
              f"logL identical = {permute_check['logL_same']}", flush=True)

        d0 = build_regrouped(sources, groups[0], work / "r000")
        LL0, EV0, N0 = run_one(d0)
        # The capture's ORDER within a dataset is the event order in the file,
        # which is the order of ``groups[0]`` -- verified against the sources by
        # value, so a permutation would show up as a mismatch rather than pass.
        want = np.array([src_ll[s][e] for s, e in groups[0]])
        got = EV0[1]
        if got.size == want.size:
            dev = np.abs(got - want)
            invariance = dict(
                n=int(want.size), original=[float(x) for x in want],
                regrouped=[float(x) for x in got],
                delta=[float(x) for x in (got - want)],
                # A CONSTANT delta means a shared normalizer moved; a
                # structured one means the coupling is per-event.
                delta_mean=float((got - want).mean()),
                delta_sd=float((got - want).std(ddof=1)),
                delta_min=float((got - want).min()),
                delta_max=float((got - want).max()),
                max_abs=float(dev.max()),
                max_rel=float((dev / np.maximum(np.abs(want), 1e-300)).max()),
                n_exact=int((dev == 0).sum()),
                matched_as_multiset=bool(np.allclose(
                    np.sort(got), np.sort(want), rtol=0, atol=1e-9)))
            print(f"\n[invariance] {invariance['n']} events regrouped: "
                  f"max |delta log-evidence| = {invariance['max_abs']:.3e} "
                  f"({invariance['n_exact']} exact)", flush=True)

        # --- (2) R over the regrouped ensemble
        for k in range(a.n_regrouped):
            d = (d0 if k == 0 else
                 build_regrouped(sources, groups[k], work / f"r{k:03d}"))
            LL, EV, N = (LL0, EV0, N0) if k == 0 else run_one(d)
            bad = [f"node {j}: captured {EV[j].size} != {N}"
                   for j in range(3) if EV[j].size != N]
            row = dict(k=k, n_events=N, logL=[float(x) for x in LL],
                       n_sources_spanned=len({s for s, _ in groups[k]}),
                       checks_failed=bad)
            if not bad:
                corr = np.array([LL[j] - EV[j].sum() for j in range(3)])
                dcorr = (corr[2] - corr[0]) / (2 * a.dh)
                u = (EV[2] - EV[0]) / (2 * a.dh) + dcorr / N
                score = (LL[2] - LL[0]) / (2 * a.dh)
                rel = abs(float(u.sum()) - score) / max(abs(score), 1e-12)
                row.update(score=float(score), dcorr=float(dcorr),
                           H=float(-(LL[2] - 2 * LL[1] + LL[0]) / a.dh ** 2),
                           J_opg=float(np.sum((u - u.mean()) ** 2) * N / (N - 1)),
                           var_within_events=float(
                               ((EV[2] - EV[0]) / (2 * a.dh)).var(ddof=1)),
                           ordering_rel=float(rel), u=[float(x) for x in u])
            rows.append(row)
            print(f"[regrouped] {k+1}/{a.n_regrouped} score="
                  f"{row.get('score', float('nan')):+.5f} J_opg="
                  f"{row.get('J_opg', float('nan')):.6f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        _core.log_evidence_and_mc_variance = real_ev
        _core.selection_log_correction = real_corr

    good = [r for r in rows if not r["checks_failed"]]
    out = dict(tree=str(tree), arm=a.arm, h0=a.h0, dh=a.dh,
               n_sources=a.n_sources, n_regrouped=a.n_regrouped,
               control=control_check, permute_control=permute_check,
               invariance=invariance, rows=rows)
    if len(good) >= 3:
        sc = np.array([r["score"] for r in good])
        jo = np.array([r["J_opg"] for r in good])
        vw = np.array([r["var_within_events"] for r in good])
        H = float(np.mean([r["H"] for r in good]))
        J_ens = float(sc.var(ddof=1))
        out.update(J_ensemble=J_ens, J_opg_mean=float(jo.mean()),
                   ratio_R=J_ens / float(jo.mean()), H_mean=H,
                   iid_prediction=float(good[0]["n_events"] * vw.mean()),
                   J_ens_over_H=J_ens / H if H else None,
                   J_opg_over_H=float(jo.mean()) / H if H else None)
        print(f"\nREGROUPED ENSEMBLE (n={len(good)}): J_ensemble="
              f"{J_ens:.6f}  mean J_OPG={jo.mean():.6f}  R={out['ratio_R']:.3f}")
        print("  R ~ 1 => the common mode was in the event DRAW (a generator "
              "property).")
        print("  R ~ 6 => it is in the ESTIMATOR, and the invariance check "
              "above says how.")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("RESHUFFLE_DONE", flush=True)


if __name__ == "__main__":
    main()
