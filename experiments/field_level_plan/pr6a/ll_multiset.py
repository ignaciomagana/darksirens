"""Do individual per-event log-evidences change when the event set changes?

The additivity residual says the event SUM is not additive under ``f_p``.  The
survey-global normalizer is ruled out -- ``N_obs_total`` and the total missing
curve are bit-identical across event sets.  So either the per-event values
themselves move, or they do not and the sum is being assembled differently.

A MULTISET comparison settles it without needing the capture's order, which is
data-dependent and has already invalidated one attempt at this question: A's 12
events are a subset of AB's 24, so if each per-event value depended on its own
event alone, ``sorted(ll(A))`` would appear inside ``sorted(ll(AB))`` to within
float tolerance.  Sorting is exactly the operation that makes this immune to the
reduction order.
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
    import darksirens.likelihood.core as _core

    work = Path(a.workdir)
    work.mkdir(parents=True, exist_ok=True)
    sources = [Path(a.tree) / a.pattern.format(i) for i in range(a.n_sources)]
    ev = []
    real = _core.log_evidence_and_mc_variance

    def _spy(ldw, nsamp):
        ll, var = real(ldw, nsamp)
        jax.debug.callback(lambda x: ev.append(np.asarray(x).ravel().copy()), ll)
        return ll, var

    sets = {"A": [(0, e) for e in range(a.n_a)],
            "B": [(1, e) for e in range(a.n_b)]}
    sets["AB"] = sets["A"] + sets["B"]
    got = {}
    _core.log_evidence_and_mc_variance = _spy
    try:
        for name, picks in sets.items():
            d = build_regrouped(sources, picks, work / name)
            p = tier_b.paths_for(d)
            logl, opts, data = A.build(p, a.arm)
            ev.clear()
            float(logl(jnp.asarray([float(a.h0)])))
            got[name] = np.sort(np.concatenate(ev))
            print(f"[ll] {name}: {got[name].size} values, "
                  f"sum {got[name].sum():.9f}", flush=True)
    finally:
        _core.log_evidence_and_mc_variance = real

    # Is sorted(A) + sorted(B) the same multiset as sorted(AB)?
    merged = np.sort(np.concatenate([got["A"], got["B"]]))
    same_size = merged.size == got["AB"].size
    worst = float(np.abs(merged - got["AB"]).max()) if same_size else None
    n_exact = int((merged == got["AB"]).sum()) if same_size else None
    print(f"\nMULTISET (arm={a.arm}): |A|+|B| = {merged.size}, "
          f"|AB| = {got['AB'].size}")
    if same_size:
        print(f"  max |sorted(A u B) - sorted(AB)| = {worst:.6e}   "
              f"({n_exact} of {merged.size} exact)")
        d = merged - got["AB"]
        print(f"  per-value deltas: min {d.min():+.4e} max {d.max():+.4e} "
              f"mean {d.mean():+.4e}")
        print("  0 => every per-event value is unchanged, so the "
              "non-additivity is NOT in the per-event terms.")
    json.dump(dict(arm=a.arm, h0=a.h0,
                   ll_A=got["A"].tolist(), ll_B=got["B"].tolist(),
                   ll_AB=got["AB"].tolist(),
                   max_abs_multiset_diff=worst, n_exact=n_exact,
                   sum_A=float(got["A"].sum()), sum_B=float(got["B"].sum()),
                   sum_AB=float(got["AB"].sum())),
              open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("LL_MULTISET_DONE", flush=True)


if __name__ == "__main__":
    main()
