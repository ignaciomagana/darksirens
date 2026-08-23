"""Is ``f_p_rows`` aligned with the catalog rows for EVERY event set?

The coupling has survived every SHARED-term explanation: not the selection
correction (its derivative is bit-identical across datasets), not the
survey-global normalizer (conditional weighting keeps ``R = 6.70``), and not the
reliability guard (raising ``max_likelihood_variance`` 1e6 -> 1e12 leaves
``logL`` bit-identical, so the guard is already inert here).  Nothing shared
across events is doing it.

That leaves a mechanism that is not shared yet is still event-set dependent, and
there is exactly one such object: the COMPACT catalog.  Rows are compacted over
``unique_inference_pixels(pixels_pe, pixels_sel)``, so changing the event set
changes how many rows exist and in what order, and ``f_p_rows`` is a per-row
gather of the full-sky map.  A gather misaligned with the rows the completeness
indexes would make an event's own ``C_p`` depend on which OTHER events share its
catalog -- exactly the observed signature, and a bug rather than a subtlety.

Checked directly, with no likelihood evaluation: build the views for several
event sets and ask whether ``f_p_rows[row]`` equals the loader's own degraded
``f_p`` at the global pixel that row represents.  Comparing against the loader's
degraded map rather than the raw nside-128 file is deliberate -- this tests the
ROW GATHER, not the degrade.

A pass eliminates the compact catalog too.  A failure is a production defect.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import arms as A
    import tier_b
    from darksirens.inference.data import load_all_data
    from darksirens.likelihood.catalog_views import prepare_catalog_views

    rows = []
    for d in a.datasets:
        p = tier_b.paths_for(d)
        opts, _ = A.make_opts(p, "latent_off")
        data = load_all_data(opts)
        fp_map = np.asarray(data["f_p_map"])
        views = prepare_catalog_views(opts, data, opts.universe_model, None)
        fp_rows = np.asarray(views.f_p_rows_pe)
        up = data.get("unique_pixels_pe")
        up = None if up is None else np.asarray(up).reshape(-1)
        if up is None:
            worst = float(np.abs(fp_rows - fp_map.astype(fp_rows.dtype)).max())
            n = fp_rows.size
        else:
            worst = float(np.abs(fp_rows - fp_map[up].astype(fp_rows.dtype)).max())
            n = up.size
        occ = views.field_f_p_occ
        occ_pix = views.field_occupied_pixels
        occ_worst = None
        if occ is not None and occ_pix is not None:
            occ = np.asarray(occ)
            occ_worst = float(np.abs(
                occ - fp_map[np.asarray(occ_pix)].astype(occ.dtype)).max())
        rows.append(dict(dataset=str(d), n_rows=int(n), compact=up is not None,
                         row_gather_max_abs=worst, exact=worst == 0.0,
                         field_occ_max_abs=occ_worst,
                         fp_rows_sum=float(fp_rows.sum())))
        print(f"[align] {d}: rows={n} compact={up is not None} "
              f"row-gather max|diff|={worst:.3e} "
              f"field_f_p_occ max|diff|={occ_worst}", flush=True)

    same = len({r["n_rows"] for r in rows}) == 1
    all_ok = all(r["exact"] for r in rows)
    print(f"\n[align] identical row counts across event sets: {same}")
    print(f"[align] every gather exact: {all_ok}")
    print("  exact => f_p is correctly aligned for every event set, and the "
          "compact catalog is NOT the coupling either.")
    json.dump(dict(rows=rows, same_row_counts=same, all_exact=all_ok),
              open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("ALIGN_DONE", flush=True)


if __name__ == "__main__":
    main()
