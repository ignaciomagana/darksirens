"""Is the per-event capture faithful?  The check the `R` result depends on.

`opg_calibration.py` measures `J_ensemble / J_OPG = 5-7` at fixed catalog, and
under DELTA-PE that is mathematically impossible if the per-event scores are
i.i.d.: the events are drawn independently from a fixed catalog, everything else
is byte-identical, so `Var(sum_i u_i)` must equal `N Var(u_i)`.  It does not, by
a factor 6.  Either the events are not i.i.d., or the capture is lying.

The capture wraps `log_evidence_and_mc_variance` -- the function the event
reduction vmaps -- with a host callback.  Its existing checks are (a) the values
sum to `logL - correction` and (b) the per-event scores sum to the full score.
BOTH ARE SUMS.  A permuted capture passes both, and so does a capture that
returns the right multiset attached to the wrong events.

This checks what those cannot:

* **determinism** -- the same dataset captured twice returns the identical array,
  so the callback order is at least stable within a configuration;
* **block invariance** -- re-running with a different ``pe_event_block`` must
  return the same MULTISET (sorted values identical).  If it does not, the
  capture is not per-event at all;
* **order invariance** -- and if the sorted values agree while the raw arrays do
  not, the capture is permuted, which is exactly the failure the two sum checks
  are blind to.

A permuted capture would INFLATE `J_OPG` (it adds pairing noise to `u_i`), so it
cannot explain an `R` that is too LARGE -- but that argument is only worth
making once the capture has been checked rather than reasoned about.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    help="a realization directory (…/c00_e000)")
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--h0", type=float, default=68.0)
    ap.add_argument("--blocks", type=int, nargs="+", default=[8, 4, 16])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import arms as A
    import tier_b
    import darksirens.likelihood.core as _core

    real_ev = _core.log_evidence_and_mc_variance
    store = []

    def _spy(ldw, nsamp):
        ll, var = real_ev(ldw, nsamp)
        jax.debug.callback(lambda x: store.append(np.asarray(x).ravel().copy()),
                           ll)
        return ll, var

    p = tier_b.paths_for(a.dataset)
    runs = {}
    _core.log_evidence_and_mc_variance = _spy
    try:
        for blk in a.blocks:
            real_opts = A.make_opts

            def _opts_blk(paths, arm, _blk=blk, **kw):
                o, sel = real_opts(paths, arm, **kw)
                o.pe_event_block = int(_blk)
                return o, sel

            for rep in (0, 1):
                store.clear()
                A.make_opts = _opts_blk
                try:
                    logl, opts, data = A.build(p, a.arm)
                finally:
                    A.make_opts = real_opts
                assert int(opts.pe_event_block) == int(blk)
                v = float(logl(jnp.asarray([float(a.h0)])))
                ev = np.concatenate(store) if store else np.array([])
                runs[f"blk{blk}_rep{rep}"] = dict(
                    logL=v, ev=ev.tolist(), n=int(ev.size))
                print(f"  block={blk} rep={rep}: logL={v:.6f} captured {ev.size}",
                      flush=True)
    finally:
        _core.log_evidence_and_mc_variance = real_ev

    keys = list(runs)
    ref = np.asarray(runs[keys[0]]["ev"])
    out = dict(dataset=a.dataset, arm=a.arm, h0=a.h0, checks={})
    for k in keys[1:]:
        x = np.asarray(runs[k]["ev"])
        same_order = (x.shape == ref.shape) and bool(np.array_equal(x, ref))
        same_multiset = (x.shape == ref.shape) and bool(
            np.allclose(np.sort(x), np.sort(ref), rtol=0, atol=1e-9))
        out["checks"][k] = dict(
            same_size=int(x.size) == int(ref.size),
            identical_order=same_order, identical_multiset=same_multiset,
            max_abs_diff_sorted=(float(np.abs(np.sort(x) - np.sort(ref)).max())
                                 if x.shape == ref.shape else None),
            logL_delta=runs[k]["logL"] - runs[keys[0]]["logL"])
        print(f"  {k}: order={same_order} multiset={same_multiset} "
              f"dlogL={out['checks'][k]['logL_delta']:+.3e}")
    out["runs"] = runs
    json.dump(out, open(a.out, "w"), indent=1)
    ok = all(c["identical_multiset"] for c in out["checks"].values())
    print(f"\nCAPTURE {'IS' if ok else 'IS NOT'} block-invariant as a multiset")
    print("CAPTURE_CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
