"""PR-5b supplement: what the PRODUCTION variance guard does to the member spread.

The main campaign quotes PR-0's ``*_nogv`` clean arm (soft guard off,
``max_likelihood_variance = 1e6``) on the argument that the soft guard's
~-1e6 nat wall is a function of ``Neff`` and ``log mu``, both member-dependent,
so measuring ``sigma`` under it would measure the spread of the guard.

The main campaign's own "soft-guard" arm did **not** test that: it flipped
``selection_neff_soft_guard`` while leaving the variance cap lifted at 1e6, so
the Neff criterion passed and the guard never engaged -- it returned
bit-identical likelihoods (offset exactly 0.0 at all five nodes, and 0.0
spread of that offset across members). That is a real result (the soft guard is
inert once the cap is lifted) but it is not the one the convention rests on.

This script runs the two arms that actually decide it, at the PRODUCTION cap
``max_likelihood_variance = 1.0``:

    guard_hard   soft guard OFF, cap 1.0   -- expected -inf everywhere (PR-0)
    guard_soft   soft guard ON,  cap 1.0   -- the shipped-scan convention

and reports, per node, the member spread under each against the clean arm's.
Five nodes, ``M_draw = 8``, ~2 minutes.
"""
from __future__ import annotations

import json
import time

import latent_harness as H
from latent_harness import jnp, np

NODES = np.array([20.0, 50.0, 80.0, 110.0, 140.0])


def main():
    anchor = H.resolve_anchor_m8()
    data = H.load_data(H.clean_arm_opts(anchor))
    out = {"h0_nodes": NODES.tolist(), "anchor": str(anchor), "arms": {}}
    t0 = time.time()

    for tag, soft, cap in (("clean_nogv", False, 1e6),
                           ("guard_hard", False, 1.0),
                           ("guard_soft", True, 1.0)):
        with H.member_ll_patch(8) as seen:
            logl = H.build_likelihood(
                H.clean_arm_opts(anchor, soft_guard=soft, max_var=cap), data)
            ll = np.stack([np.asarray(logl(jnp.asarray([h]))) + np.log(8.0)
                           for h in NODES])
        if not seen["n"]:
            raise SystemExit("member-ll patch never fired")
        finite = np.isfinite(ll).all(axis=1)
        sd = np.where(finite, ll.std(axis=1, ddof=0), np.nan)
        out["arms"][tag] = {
            "soft_guard": soft, "max_likelihood_variance": cap,
            "ll": ll.tolist(), "sigma": sd.tolist(),
            "n_finite_nodes": int(finite.sum()),
            "mean_ll": np.where(finite, ll.mean(axis=1), np.nan).tolist(),
        }
        print(f"[{tag}] soft={soft} cap={cap:g}  finite {int(finite.sum())}/"
              f"{NODES.size}", flush=True)
        for i, h in enumerate(NODES):
            print(f"   H0={h:6.1f}  mean_ll={ll[i].mean():14.6f}  "
                  f"sigma={sd[i]:.6e}", flush=True)

    # The decisive comparison: does the guard change the MEMBER SPREAD?
    c = np.asarray(out["arms"]["clean_nogv"]["sigma"])
    for tag in ("guard_hard", "guard_soft"):
        s = np.asarray(out["arms"][tag]["sigma"])
        with np.errstate(invalid="ignore", divide="ignore"):
            out["arms"][tag]["sigma_over_clean"] = (s / c).tolist()
        print(f"[{tag}] sigma / sigma_clean = "
              f"{np.array2string(s / c, precision=6)}", flush=True)

    out["total_seconds"] = time.time() - t0
    out["git_sha"] = H.C.git_sha()
    p = H.PR5B_DIR / "guard_arm.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"[guard] wrote {p}  ({out['total_seconds'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
