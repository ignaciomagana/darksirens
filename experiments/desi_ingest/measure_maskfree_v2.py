"""Measure mask-freedom of a --depth-map Q artifact, independently of its stamp.

Reopens the finished table and recomputes the verdict by calling THE SAME
function the builder stamps with (``measure_mask_free``), so the report can
never drift from the stamp's definition -- only the inputs differ (a file on
disk instead of an in-memory array).  Adds the v1-comparable statistics the
checkpoint recorded only as prose:

  v1 (ad hoc, 2026-08-20):  off-footprint logQ mean -0.073 sd 0.5694
                            corr(Q, f_p) +0.392 @ z-slice 150, +0.333 @ 250,
                            -0.995 @ 400

Since v4 the correlation criterion is anchored to the CATALOG's own
density-depth coupling rather than to zero: the covered DESI sky really does
have corr(N/f_p, f_p) ~ +0.11 to +0.24 across z, and a faithful Q must
reproduce it.  The record therefore carries both profiles and the per-slice
delta; ``worst_abs_corr`` is kept for direct comparison with the v2/v3 records,
but the VERDICT rests on ``worst_abs_delta``.

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

    from darksirens.cli.build_lognormal_completion import (
        _MASK_FREE_CORR_TOL,
        _MASK_FREE_OFF_LOGQ_TOL,
        format_mask_free_report,
        measure_mask_free,
    )

    with h5py.File(args.artifact, "r") as f:
        g = f["lss_completion"]
        lq = np.asarray(g["logq_map"][...], dtype=float)
        # The artifact's OWN zgrid, so the band edges do not depend on whatever
        # DARKSIRENS_ZMAX this process happens to be running under.
        zg = np.asarray(g["zgrid"][...], dtype=float) if "zgrid" in g else None
        stamped = g.attrs.get("f_p_aware", None)
        stamped = None if stamped is None else bool(stamped)

    res = measure_mask_free(lq, args.catalog, args.depth_map, zgrid_nodes=zg)
    assert not res.get("skipped"), res.get("reason")
    n_grid = res["n_grid"]

    # SIGNED correlations at the v1 slices, for continuity with the prose
    # baseline.  Recomputed here rather than in measure_mask_free: 150/250/400
    # are not on the nine-slice sweep, and they exist only for the comparison.
    from darksirens.catalogs.depth_map import load_selection_fraction
    from darksirens.catalogs.io import load_survey

    nside = int(load_survey(args.catalog, to_device=False)[0])
    fp = np.asarray(load_selection_fraction(args.depth_map, nside).f_p,
                    dtype=float)
    on = fp > 0.0

    def corr_at(zi):
        q = np.exp(lq[:, zi])
        if float(np.std(q[on])) == 0.0 or float(np.std(fp[on])) == 0.0:
            return 0.0
        return float(np.corrcoef(q[on], fp[on])[0, 1])

    v1_slices = {str(zi): corr_at(zi) for zi in (150, 250, 400) if zi < n_grid}
    # The nine-slice sweep keeps its v2/v3 shape EXACTLY -- all nine entries,
    # 0.0 where the slice is constant-Q -- so the four records stay directly
    # comparable.  The verdict's own (skipped-slices-omitted) profile is the
    # corr_q/corr_data/corr_delta triple beside it.
    nine = np.linspace(0, n_grid - 1, 9).astype(int)
    sweep = {str(int(zi)): corr_at(int(zi)) for zi in nine}
    data_profile = dict(res["corr_data"])
    delta_profile = dict(res["corr_delta"])
    worst = max(abs(v) for v in sweep.values())
    earned = bool(res["ok"])
    off_stats = res["off_footprint"]

    rec = dict(
        artifact=args.artifact, depth_map=args.depth_map, nside=res["nside"],
        n_grid=n_grid, stamped_f_p_aware=stamped,
        recomputed_f_p_aware=earned,
        tolerances=dict(off_logq=float(_MASK_FREE_OFF_LOGQ_TOL),
                        corr=float(_MASK_FREE_CORR_TOL)),
        off_footprint=off_stats,
        corr_signed_v1_slices=v1_slices,
        corr_signed_nine_slice_sweep=sweep,
        # v4: the anchor and the criterion it feeds.  These three carry only the
        # slices actually compared (constant-Q slices are skipped).
        corr_q_compared_slices=dict(res["corr_q"]),
        corr_data_nine_slice_sweep=data_profile,
        corr_delta_nine_slice_sweep=delta_profile,
        worst_abs_delta=float(res["worst_abs_delta"]),
        worst_signed_delta=float(res["worst_signed_delta"]),
        worst_delta_slice=res["worst_delta_slice"],
        band_half_nodes=int(res["band_half_nodes"]),
        n_fit_cols=int(res["n_fit_cols"]),
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
    for line in format_mask_free_report(res, prefix="[maskfree-v2]"):
        print(line)
    print(f"[maskfree-v2] worst |corr(Q, f_p)| over 9 slices: {worst:.3f} "
          f"(v2/v3-comparable; the VERDICT is the delta above)")
    if stamped is not None and bool(stamped) != earned:
        print("[maskfree-v2] WARNING: stamp and recomputation DISAGREE -- "
              "trust the recomputation, and treat the artifact as unstamped.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"[maskfree-v2] record -> {args.json}")


if __name__ == "__main__":
    main()
