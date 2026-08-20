"""Parity gate: darksirens-dev vs the pinned loa_faint_joint42 posterior.

Re-evaluates the desi repo's joint42 configuration (42 events, per_pixel
complete-catalog likelihood, nside-128 pixelated union catalog, matched
betaS, GWTC-5 fixed population, H0 grid 20..140 step 1) with THIS repo's
likelihood, and compares the grid posterior to the pinned
runs/loa_faint_joint42/h0_scan.csv (darksirens-a14fix code).

Run with DARKSIRENS_ZMAX UNSET (the pinned runs used the package default);
do NOT import common.py here.  Agreement gates the dev code before the
selection-channel run; a shape difference must be understood first.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if os.environ.get("DARKSIRENS_ZMAX"):
    raise SystemExit("unset DARKSIRENS_ZMAX for the parity scan (the pinned "
                     "reference ran at the package default)")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

DESI = Path("/hildafs/projects/phy230014p/magana/desi_darksirens_selection")
LOA = DESI / "final/experiments/experiment_loa_rebuild"
GW42 = (DESI / "final/experiments/experiment_combined_desi_ls_des/inputs/"
        "cells_armC42/joint/gw.h5")
BETA42 = LOA / "inputs/selection_betaS_v2_loaFaint_marg_s05_noom.h5"
SURVEY128 = LOA / "inputs/rebuild_loa_faint_pixelated/catalog_pixelated_nside_128.h5"
REF_CSV = LOA / "runs/loa_faint_joint42/h0_scan.csv"


def _opts():
    return SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks",
        universe_model="dark_sirens_complete",
        survey_path=str(SURVEY128), gw_path=str(GW42),
        gwselection_path=str(BETA42), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=None, lss_marginalize=False,
        c_mode="per_pixel", catalog_sky_weighting="field",
        complete_empty_pixel_policy="zero",
        mark_model="none", mark_names=(), sky_model="isotropic",
        # The reference run recorded drop_full_catalog=true, but in dev the
        # field-convention normalizer is built from the full-sky rows inside
        # catalog_views, AFTER load_all_data would have dropped them; keeping
        # the full catalog only changes memory, never the likelihood value.
        drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True,
        fix_survey=True, shared_beta=True, shared_spin=True,
        shared_gamma=True, sel_batch_size=16384, pe_event_block=8,
        selection_neff_soft_guard=True,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h0-min", type=float, default=20.0)
    p.add_argument("--h0-max", type=float, default=140.0)
    p.add_argument("--h0-step", type=float, default=1.0)
    p.add_argument("--outdir", default="data/parity")
    args = p.parse_args(argv)

    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    opts = _opts()
    fixed = {"Om0": 0.3075, "sigma_kde": 0.003}
    data = load_all_data(opts)
    opts.resolved_survey_z_depths = (data.get("z_depth"),)
    pop_fid = get_fixed_population_params(opts.pop_model)
    logl = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)

    h0 = np.arange(args.h0_min, args.h0_max + 0.5 * args.h0_step, args.h0_step)
    vals = np.empty(h0.size)
    for i, h in enumerate(h0):
        vals[i] = float(logl(jnp.asarray([h])))
        if i % 20 == 0:
            print(f"  H0={h:.0f}: logL={vals[i]:.3f}", flush=True)

    # Normalized grid posterior + comparison to the pinned reference.
    logp = vals - vals.max()
    pdf = np.exp(logp)
    pdf /= np.trapz(pdf, h0)

    ref = {}
    with open(REF_CSV) as f:
        for row in csv.DictReader(f):
            ref.setdefault("h0", []).append(float(row.get("H0") or row.get("h0")))
            key = next(k for k in row if k.lower() in
                       ("logl", "log_likelihood", "loglike", "logpost", "pdf"))
            ref.setdefault("val", []).append(float(row[key]))
    ref_h0 = np.asarray(ref["h0"])
    ref_val = np.asarray(ref["val"])
    ref_pdf = (np.exp(ref_val - ref_val.max()) if ref_val.max() < 100
               else ref_val)
    ref_pdf = ref_pdf / np.trapz(ref_pdf, ref_h0)

    common = np.intersect1d(np.round(h0, 6), np.round(ref_h0, 6))
    ours = np.interp(common, h0, pdf)
    theirs = np.interp(common, ref_h0, ref_pdf)
    tv = 0.5 * np.trapz(np.abs(ours - theirs), common)

    def _summ(hh, pp):
        cdf = np.concatenate([[0], np.cumsum(0.5 * (pp[1:] + pp[:-1])
                                             * np.diff(hh))])
        cdf /= cdf[-1]
        q = lambda a: float(np.interp(a, cdf, hh))  # noqa: E731
        return {"median": q(0.5), "ci68": [q(0.16), q(0.84)],
                "map": float(hh[np.argmax(pp)])}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "inputs": {"gw": str(GW42), "beta": str(BETA42),
                   "survey": str(SURVEY128), "fixed": fixed,
                   "reference": str(REF_CSV)},
        "h0": h0.tolist(), "logl": vals.tolist(), "pdf": pdf.tolist(),
        "ours": _summ(h0, pdf), "reference": _summ(ref_h0, ref_pdf),
        "total_variation_distance": float(tv),
    }
    with open(outdir / "parity_scan.json", "w") as f:
        json.dump(payload, f, indent=1)
    print(json.dumps({k: payload[k] for k in
                      ("ours", "reference", "total_variation_distance")},
                     indent=1))


if __name__ == "__main__":
    main()
