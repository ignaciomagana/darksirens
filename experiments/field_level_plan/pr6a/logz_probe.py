"""Which term of the f_p path is event-set dependent?  Capture log Z directly.

The additivity test localized the coupling to the ``f_p`` CODE PATH: the residual
``E(A u B) - E(A) - E(B)`` is exactly 0 without ``f_p``, -18.3 nat with the real
map, and -55.3 nat with ``f_p == 1`` -- where multiplying the completeness by 1.0
is the IEEE identity, so the arithmetic is the no-``f_p`` arm's and any residual
is structural.

Under the field convention every event's term subtracts the SAME survey-global
``log Z``, so if ``log Z`` were event-set independent the sum would be additive by
construction.  The arithmetic runs the other way too: with ``|A| = |B| = 12``,

    residual = -[24 logZ_AB - 12 logZ_A - 12 logZ_B]

so a residual of -18.3 nat implies ``log Z`` moving by ~0.76 nat between event
sets.  This captures ``log Z`` for each set and checks that identity, which either
confirms the normalizer as the carrier or rules it out and sends the hunt to the
per-row terms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from event_reshuffle import build_regrouped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--pattern", default="c00_e{:03d}")
    ap.add_argument("--n-sources", type=int, default=16)
    ap.add_argument("--n-a", type=int, default=12)
    ap.add_argument("--n-b", type=int, default=12)
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--h0", type=float, default=68.0)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import arms as A
    import tier_b
    import darksirens.redshift.completion as _comp

    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    sources = [Path(a.tree) / a.pattern.format(i) for i in range(a.n_sources)]

    store = {}
    real_Z = _comp.field_global_log_Z
    real_obs = _comp.field_observed_global_total
    real_miss = _comp._field_missing_curve

    def _spyZ(*args, **kw):
        out = real_Z(*args, **kw)
        jax.debug.callback(lambda v: store.setdefault("logZ", []).append(
            float(np.asarray(v))), out)
        return out

    def _spyObs(*args, **kw):
        out = real_obs(*args, **kw)
        jax.debug.callback(lambda v: store.setdefault("N_obs_total", []).append(
            float(np.asarray(v))), out)
        return out

    def _spyMiss(*args, **kw):
        V, dN = real_miss(*args, **kw)
        jax.debug.callback(
            lambda v: store.setdefault("V_total_sum", []).append(
                float(np.asarray(v).sum())), V)
        return V, dN

    rows = {}
    sets = {"A": [(0, e) for e in range(a.n_a)],
            "B": [(1, e) for e in range(a.n_b)]}
    sets["AB"] = sets["A"] + sets["B"]
    _comp.field_global_log_Z = _spyZ
    _comp.field_observed_global_total = _spyObs
    _comp._field_missing_curve = _spyMiss
    try:
        for name, picks in sets.items():
            d = build_regrouped(sources, picks, work / name)
            p = tier_b.paths_for(d)
            logl, opts, data = A.build(p, a.arm)
            store.clear()
            v = float(logl(jnp.asarray([float(a.h0)])))
            rows[name] = dict(
                n_events=len(picks), logL=v,
                logZ=sorted(set(store.get("logZ", []))),
                N_obs_total=sorted(set(store.get("N_obs_total", []))),
                V_total_sum=sorted(set(store.get("V_total_sum", []))),
                n_rows=int(np.asarray(data["unique_pixels_pe"]).size)
                if data.get("unique_pixels_pe") is not None else None)
            print(f"[logZ] {name}: rows={rows[name]['n_rows']}  "
                  f"logZ={rows[name]['logZ']}  "
                  f"N_obs={rows[name]['N_obs_total']}  "
                  f"V_sum={rows[name]['V_total_sum']}", flush=True)
    finally:
        _comp.field_global_log_Z = real_Z
        _comp.field_observed_global_total = real_obs
        _comp._field_missing_curve = real_miss

    def one(name, key):
        v = rows[name][key]
        return float(v[0]) if v else float("nan")

    zA, zB, zAB = (one(n, "logZ") for n in ("A", "B", "AB"))
    pred = -(24.0 * zAB - 12.0 * zA - 12.0 * zB) if np.isfinite(zAB) else None
    print(f"\nlog Z: A={zA:.9f}  B={zB:.9f}  AB={zAB:.9f}")
    print(f"  AB - A = {zAB - zA:+.6e}   AB - B = {zAB - zB:+.6e}")
    print(f"  predicted additivity residual from logZ alone = {pred:+.6e} nat")
    print(f"  (measured by additivity.py: -1.832e+01 with the real f_p map)")
    print(f"N_obs_total: A={one('A','N_obs_total'):.6f} "
          f"B={one('B','N_obs_total'):.6f} AB={one('AB','N_obs_total'):.6f}")
    print(f"V_total sum: A={one('A','V_total_sum'):.6f} "
          f"B={one('B','V_total_sum'):.6f} AB={one('AB','V_total_sum'):.6f}")
    json.dump(dict(arm=a.arm, h0=a.h0, rows=rows,
                   logZ_A=zA, logZ_B=zB, logZ_AB=zAB,
                   predicted_residual=pred), open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("LOGZ_PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
