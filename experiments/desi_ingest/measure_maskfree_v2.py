"""Measure mask-freedom of a --depth-map Q artifact, independently of its stamp.

Reopens the finished table and recomputes what ``_verify_mask_free`` checked,
plus the v1-comparable statistics the checkpoint recorded only as prose:

  v1 (ad hoc, 2026-08-20):  off-footprint logQ mean -0.073 sd 0.5694
                            corr(Q, f_p) +0.392 @ z-slice 150, +0.333 @ 250,
                            -0.995 @ 400

Writes a JSON record beside the artifact so v2 has a machine-readable baseline
(v1 left none).  Exit code 0 regardless of verdict -- the verdict is the point.
"""
import argparse
import json

import h5py
import numpy as np

V1_PROSE = {
    "off_footprint_logq_mean": -0.073,
    "off_footprint_logq_sd": 0.5694,
    "corr_slices": {"150": 0.392, "250": 0.333, "400": -0.995},
    "source": "experiments/CHECKPOINT.md (2026-08-20), no artifact survived",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("depth_map")
    ap.add_argument("catalog")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from darksirens.catalogs.depth_map import load_selection_fraction
    from darksirens.catalogs.io import load_survey

    nside = int(load_survey(args.catalog, to_device=False)[0])
    fp = np.asarray(load_selection_fraction(args.depth_map, nside).f_p,
                    dtype=float)

    with h5py.File(args.artifact, "r") as f:
        g = f["lss_completion"]
        lq = np.asarray(g["logq_map"][...], dtype=float)
        stamped = g.attrs.get("f_p_aware", None)
        stamped = None if stamped is None else bool(stamped)

    assert lq.shape[0] == fp.size, (lq.shape, fp.size)
    off = fp <= 0.0
    on = ~off
    n_grid = lq.shape[1]

    off_lq = lq[off]
    off_stats = dict(
        n_off=int(off.sum()), n_on=int(on.sum()),
        mean=float(off_lq.mean()), sd=float(off_lq.std()),
        max_abs=float(np.abs(off_lq).max()),
    )

    # SIGNED correlations at the v1 slices (guarded against a shorter grid),
    # plus the builder's own 9-slice worst-|corr| sweep.
    def corr_at(zi):
        q = np.exp(lq[:, zi])
        if float(np.std(q[on])) == 0.0 or float(np.std(fp[on])) == 0.0:
            return 0.0
        return float(np.corrcoef(q[on], fp[on])[0, 1])

    v1_slices = {str(zi): corr_at(zi) for zi in (150, 250, 400) if zi < n_grid}
    nine = np.linspace(0, n_grid - 1, 9).astype(int)
    sweep = {str(int(zi)): corr_at(int(zi)) for zi in nine}
    worst = max(abs(v) for v in sweep.values())

    # The builder's own tolerances, read from the module so this report cannot
    # drift from the stamp's definition.
    from darksirens.cli.build_lognormal_completion import (
        _MASK_FREE_CORR_TOL, _MASK_FREE_OFF_LOGQ_TOL)
    earned = (off_stats["max_abs"] <= _MASK_FREE_OFF_LOGQ_TOL
              and worst <= _MASK_FREE_CORR_TOL)

    rec = dict(
        artifact=args.artifact, depth_map=args.depth_map, nside=nside,
        n_grid=n_grid, stamped_f_p_aware=stamped,
        recomputed_f_p_aware=bool(earned),
        tolerances=dict(off_logq=float(_MASK_FREE_OFF_LOGQ_TOL),
                        corr=float(_MASK_FREE_CORR_TOL)),
        off_footprint=off_stats,
        corr_signed_v1_slices=v1_slices,
        corr_signed_nine_slice_sweep=sweep,
        worst_abs_corr=float(worst),
        v1_prose_baseline=V1_PROSE,
    )

    print(f"[maskfree-v2] stamped f_p_aware = {stamped}, "
          f"recomputed = {earned}")
    print(f"[maskfree-v2] off-footprint ({off_stats['n_off']} px): "
          f"logQ mean {off_stats['mean']:+.4e} sd {off_stats['sd']:.4e} "
          f"max|.| {off_stats['max_abs']:.3e}  "
          f"(v1: mean -0.073 sd 0.5694; need max <= "
          f"{_MASK_FREE_OFF_LOGQ_TOL:g})")
    for zi, c in v1_slices.items():
        v1 = V1_PROSE["corr_slices"].get(zi)
        print(f"[maskfree-v2] corr(Q, f_p) @ z-slice {zi}: {c:+.3f}"
              f"   (v1: {v1:+.3f})")
    print(f"[maskfree-v2] worst |corr| over 9 slices: {worst:.3f} "
          f"(need <= {_MASK_FREE_CORR_TOL:g})")
    if stamped is not None and bool(stamped) != bool(earned):
        print("[maskfree-v2] WARNING: stamp and recomputation DISAGREE -- "
              "trust the recomputation, and treat the artifact as unstamped.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"[maskfree-v2] record -> {args.json}")


if __name__ == "__main__":
    main()
