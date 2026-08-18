"""PR-10's compaction question, measured rather than banked.

PLAN section 7's PR-10 entry proposes a ``z < z_depth`` sample-compaction
experiment for the seam: "most GWTC-5 PE samples are above ``z_depth``, so the
gather may be largely skippable -- measure, do not bank".  This measures it.

The seam's per-sample work is a two-node gather into ``base_miss`` and
``phi_z`` for EVERY PE sample and EVERY injection.  A sample above ``z_depth``
contributes ``logQ = 0`` by construction (the field has no support there), so
its gather is arithmetic whose answer is known in advance.  If the below-depth
fraction is small, compacting to it is a large constant-factor saving on the
one part of the likelihood the latent seam added.

The fraction is a function of ``H0``: ``z(dL)`` moves with the distance ladder,
so a sample sits below the depth at one trial value and above it at another.
That is why this is a curve and not a number, and why compaction has to be
sized at the WORST ``H0`` in the prior rather than at the fiducial.

Reports the same fraction for the injection set, which is the larger of the two
gathers (1.07M injections against 259 x 4096 PE samples) and therefore the one
that decides whether the experiment is worth doing.
"""
from __future__ import annotations

import argparse
import json

import common as C  # noqa: F401  (pins ZMAX; must be first)

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--z-depth", type=float, default=0.30)
    ap.add_argument("--h0", nargs="*", type=float,
                    default=[20.0, 40.0, 67.74, 90.0, 120.0, 140.0])
    ap.add_argument("--out", default="pe_support_fraction.json")
    args = ap.parse_args(argv)

    import h5py
    from darksirens.utils.cosmology import z_of_dL
    import jax.numpy as jnp

    def _fraction(path, key_dl):
        with h5py.File(path, "r") as f:
            dl = np.asarray(f[key_dl][...]).ravel()
        dl = dl[np.isfinite(dl) & (dl > 0)]
        rows = []
        for h in args.h0:
            z = np.asarray(z_of_dL(jnp.asarray(dl), h, C.OM0))
            frac = float(np.mean(z <= args.z_depth))
            rows.append(dict(h0=float(h), frac_below_depth=frac,
                             n=int(dl.size)))
            print(f"    H0={h:6.2f}  below z_depth: {100*frac:7.4f}%",
                  flush=True)
        return rows

    out = {}
    print(f"[PE samples] {C.GW_259}", flush=True)
    try:
        out["pe"] = _fraction(str(C.GW_259), "dL")
    except KeyError as e:
        print(f"    (dL key not found: {e}; listing)", flush=True)
        with h5py.File(str(C.GW_259)) as f:
            print("    keys:", list(f.keys())[:20])
        raise
    print(f"[injections] {C.INJ_PLAIN}", flush=True)
    out["injections"] = _fraction(str(C.INJ_PLAIN), "dL")

    worst = max(r["frac_below_depth"] for r in out["injections"])
    print(f"\nWORST-CASE below-depth injection fraction over the H0 prior: "
          f"{100*worst:.4f}%")
    print(f"=> compaction would skip {100*(1-worst):.4f}% of the injection "
          f"gather at the worst H0 in the prior")
    out["summary"] = dict(z_depth=args.z_depth,
                          worst_inj_frac_below=worst,
                          skippable_at_worst=1.0 - worst)
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
