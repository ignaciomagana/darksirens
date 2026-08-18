"""The PR-8 deliverable: H0 versus the ASSUMED amp(z > z_depth), at nside 16.

This is a SENSITIVITY SCAN and not a posterior over amp(z).  There are no
counts above the fitted depth -- R1 measures the in-support fraction of the
missing budget at 6e-5 -- so every number in the table below is a function of a
number that was chosen, and marginalizing over the rows would be quoting a
prior as a measurement (PLAN OWNER DECISION 7).  What the table bounds is the
systematic PLAN §4.2 states and declines to fix: "``Q == 1`` off-footprint and
above ``z_depth``" assigns ZERO variance where there is no data rather than the
prior variance, and the scan says how much H0 moves if that variance is put
back at an assumed amplitude.

Arms, all on the ONE realization ``pr6a/data/rb`` (seed 7001, 60 events) and the
one solve of ``build_anchor_amp.solve_once``:

  ``latent_off``   no field at all -- the ``sel`` configuration of
                   ``experiments/desi_full259``, for scale.
  ``noprofile``    the extended geometry (M_z = 11, nodes to z = 1.5) with the
                   PRE-PR-8 artifact shape: consumption rows truncated at the
                   depth, no amp machinery anywhere.
  ``amp<X>``       the same solve consumed under amp(z > 0.3) = X.

``noprofile`` and ``amp0`` are the same physics reached two ways, and the scan
CHECKS that they agree bit-for-bit rather than assuming it: that is the gate
that the amp machinery is inert at the legacy value, measured on the full
likelihood instead of on a basis array.

The guard convention is PR-6a's clean arm (``selection_neff_soft_guard=False``,
``max_likelihood_variance=1e6``), for the reason PR-5b's report gives: under the
soft guard a member ensemble's spread is the wall's spread, not the
likelihood's.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys

PR8 = Path(__file__).resolve().parent
sys.path.insert(0, str(PR8.parent / "pr6a"))

import world16 as W16                                          # noqa: E402
import arms as A                                               # noqa: E402
import tier_b as TB                                            # noqa: E402


def run(*, data, anchors, grid=None, quiet=False, with_off=True):
    import jax.numpy as jnp                                     # noqa: F401

    p = TB.paths_for(data)
    h0 = A.DEFAULT_GRID if grid is None else grid
    cache = {}
    out = {}

    def _one(name, anchor_path):
        pp = dict(p)
        if anchor_path is not None:
            pp["anchor"] = str(anchor_path)
        arm = "latent_off" if anchor_path is None else "latent"
        t0 = time.time()
        logl, opts, data_obj = A.build(pp, arm, data_cache=cache)
        t_build = time.time() - t0
        t0 = time.time()
        vals = A.scan_h0(logl, h0)
        s = A.summarize(h0, vals)
        s["build_s"] = t_build
        s["scan_s"] = time.time() - t0
        s["ms_per_eval"] = 1e3 * s["scan_s"] / h0.size
        s["anchor"] = None if anchor_path is None else str(anchor_path)
        out[name] = s
        if not quiet:
            print(f"  [{name}] H0 = {s['median']:.3f}  "
                  f"90% [{s['ci90'][0]:.3f}, {s['ci90'][1]:.3f}] "
                  f"w90 = {s['width90']:.3f}  sigma = {s['sigma']:.3f}  "
                  f"cdf(H0_true) = {s['cdf_at_truth']:.4f}  "
                  f"({s['ms_per_eval']:.0f} ms/eval, build {t_build:.0f}s)",
                  flush=True)
        return s

    if with_off:
        _one("latent_off", None)
    for name, path in anchors:
        _one(name, path)
    out["_h0"] = h0.tolist()
    return out


def table(res):
    """The deliverable, plus the two gates that make it readable."""
    rows = []
    ref = res.get("amp0")
    for k, v in res.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        rows.append(dict(
            arm=k, median=v["median"], ci90=v["ci90"], width90=v["width90"],
            sigma=v["sigma"], cdf_at_truth=v["cdf_at_truth"],
            d_median_vs_amp0=(None if ref is None
                              else v["median"] - ref["median"]),
            d_width_vs_amp0=(None if ref is None
                             else v["width90"] - ref["width90"])))
    gates = {}
    if "noprofile" in res and "amp0" in res:
        a, b = res["noprofile"], res["amp0"]
        d = np.abs(np.asarray(a["logl"]) - np.asarray(b["logl"]))
        gates["amp0_vs_noprofile_max_abs_logl_diff"] = float(d.max())
        gates["amp0_vs_noprofile_bit_identical"] = bool(d.max() == 0.0)
        gates["amp0_vs_noprofile_median_diff"] = float(
            b["median"] - a["median"])
    return dict(rows=rows, gates=gates, H0_true=float(W16.H0_TRUE))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(PR8.parent / "pr6a" / "data" / "rb"))
    ap.add_argument("--anchors", default=str(PR8 / "anchors"))
    ap.add_argument("--out", default=str(PR8 / "scan_amp.json"))
    ap.add_argument("--h0-lo", type=float, default=20.0)
    ap.add_argument("--h0-hi", type=float, default=140.0)
    ap.add_argument("--h0-step", type=float, default=1.0)
    ap.add_argument("--no-off", action="store_true")
    ap.add_argument("--only", default=None,
                    help="comma-separated arm names to run")
    a = ap.parse_args(argv)

    ad = Path(a.anchors)
    order = [("noprofile", ad / "anchor_noprofile.h5"),
             ("amp0", ad / "anchor_amp0.h5"),
             ("amp0.05", ad / "anchor_amp0.05.h5"),
             ("amp0.1", ad / "anchor_amp0.1.h5"),
             ("amp0.2", ad / "anchor_amp0.2.h5"),
             ("amp0.4", ad / "anchor_amp0.4.h5"),
             ("growth0.2", ad / "anchor_growth0.2.h5")]
    order = [(n, q) for n, q in order if q.exists()]
    if a.only:
        keep = set(a.only.split(","))
        order = [(n, q) for n, q in order if n in keep]

    grid = np.arange(a.h0_lo, a.h0_hi + 0.5 * a.h0_step, a.h0_step)
    print(f"[scan] {len(order)} anchors, {grid.size} H0 nodes "
          f"[{grid[0]}, {grid[-1]}]", flush=True)
    res = run(data=a.data, anchors=order, grid=grid,
              with_off=not a.no_off)
    res["_table"] = table(res)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["_table"], indent=1, default=str))


if __name__ == "__main__":
    main()
