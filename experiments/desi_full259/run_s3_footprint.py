"""S-3, measured: what the unmasked footprint costs the SHIPPED Q-table line.

Under ``c_mode=selection`` the completeness is a survey curve ``Cbar(z)``
applied to every pixel, empty ones included.  For a pixel that is empty because
DESI never observed it, that says the survey catalogued a fraction ``Cbar`` of
its galaxies -- so those hosts leave the missing budget while the numerator
still places hosts there.  On this line 38% of the sky is off-footprint.

Until now the fix (``--per_pixel_completeness``, ``C_p = f_p Cbar`` with
``f_p = 0`` off-footprint) was REFUSED alongside a Q table, because the field
normalizer had no ``Sum_{p empty} f_p Q_p(z)`` budget.  It has one now, so the
shipped ``selq_radial`` configuration can be run both ways and the exposure
becomes a number rather than an argument.

Arms (identical except for the mask; the CLEAN guard convention throughout,
PR-5b, so the soft guard's wall does not shape the comparison):

``q_nofp``  Q table, no mask -- the exposed configuration, now behind
            ``--allow_unmasked_footprint`` and run here deliberately.
``q_fp``    Q table + mask -- the S-3 fix.

The Q-free pair (``fp`` / ``nofp``) is already measured by ``run_h0_latent.py``
and is the control this reads against: there the mask moved the median by 4.3
sigma.  If the Q table changed that, the difference is Q's, not the mask's.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import common as C  # noqa: F401  (pins DARKSIRENS_ZMAX=6.0; must be first)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402

# numpy 1.26 (the pinned stack on both machines) has no ``trapezoid``; 2.x
# renamed ``trapz`` to it.  Bind once rather than per call.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _opts(*, q_path, per_pixel_completeness=None, allow_unmasked=False):
    o = SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks",
        universe_model="dark_sirens",
        survey_path=str(C.SURVEY_N64), gw_path=str(C.GW_259),
        gwselection_path=str(C.INJ_PLAIN), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=str(q_path),
        lss_marginalize=False,
        c_mode="selection", catalog_sky_weighting="field",
        complete_empty_pixel_policy="zero",
        mark_model="none", mark_names=(), sky_model="isotropic",
        drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True,
        fix_survey=True, shared_beta=True, shared_spin=True,
        shared_gamma=True, sel_batch_size=16384, pe_event_block=8,
        # CLEAN arm (PR-0/PR-5b), not the shipped scan's soft guard.
        selection_neff_soft_guard=False,
        max_likelihood_variance=1e6,
        per_pixel_completeness=per_pixel_completeness,
        allow_unmasked_footprint=allow_unmasked,
        lss_field_mode="table", lss_field_artifact=None,
    )
    sel = load_selection_fit_json(str(C.FIT_JSON))
    o.selection_fit = str(C.FIT_JSON)
    o.selection_kcorr_by_catalog = [tuple(sel["k_corr_coeffs"]) or None]
    return o


def _summarize(h0, vals):
    finite = np.isfinite(vals)
    if not finite.any():
        return {"h0": h0.tolist(), "logl": vals.tolist(),
                "all_nonfinite": True}
    v = np.where(finite, vals, -np.inf)
    pdf = np.exp(v - v.max())
    pdf = np.where(finite, pdf, 0.0)
    pdf /= _trapz(pdf, h0)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                         * np.diff(h0))])
    cdf /= cdf[-1]
    q = lambda a: float(np.interp(a, cdf, h0))  # noqa: E731
    return {"h0": h0.tolist(), "logl": vals.tolist(), "pdf": pdf.tolist(),
            "median": q(0.5), "ci68": [q(0.16), q(0.84)],
            "ci90": [q(0.05), q(0.95)], "map": float(h0[np.argmax(pdf)]),
            "n_finite": int(finite.sum())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h0-min", type=float, default=20.0)
    p.add_argument("--h0-max", type=float, default=140.0)
    p.add_argument("--h0-step", type=float, default=1.0)
    p.add_argument("--arms", nargs="*", default=["q_nofp", "q_fp"])
    p.add_argument("--q", default=None,
                   help="Q table (default: data/fits/q_radial.h5)")
    p.add_argument("--mth-map", default=None,
                   help="depth map for the masked arm (default: the ingest "
                        "nside-128 map)")
    p.add_argument("--outdir", default="data/s3_footprint")
    args = p.parse_args(argv)

    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    q_path = args.q or (C.DATA_DIR / "fits" / "q_radial.h5")
    mth = args.mth_map or (C.INGEST_DATA / "mth_map_nside128.h5")
    sel = load_selection_fit_json(str(C.FIT_JSON))
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    fixed = {"Om0": C.OM0, "sigma_kde": 0.003,
             "log10n0": float(cal["log10n0"]), "delta": float(cal["delta"]),
             "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
             "sigma_M": float(sel["sigma_M"])}

    ARMS = {
        "q_nofp": dict(allow_unmasked=True),
        "q_fp": dict(per_pixel_completeness=str(mth)),
    }
    h0 = np.arange(args.h0_min, args.h0_max + 0.5 * args.h0_step, args.h0_step)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    resfile = outdir / "s3_footprint_scans.json"
    results = json.load(open(resfile))["results"] if resfile.exists() else {}

    for name in args.arms:
        opts = _opts(q_path=q_path, **ARMS[name])
        print(f"=== {name}: Q={q_path} f_p="
              f"{opts.per_pixel_completeness or 'OFF (exposed)'}", flush=True)
        data = load_all_data(opts)
        opts.resolved_survey_z_depths = (data.get("z_depth"),)
        pop_fid = get_fixed_population_params(opts.pop_model)
        logl = make_likelihood(opts, data, pop_fid,
                               fixed_parameter_values=dict(fixed))
        vals = np.empty(h0.size)
        t0 = time.time()
        for i, h in enumerate(h0):
            vals[i] = float(logl(jnp.asarray([h])))
            if i % 20 == 0:
                print(f"  H0={h:6.1f}: logL={vals[i]:14.4f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
        wall = time.time() - t0
        r = _summarize(h0, vals)
        r.update(arm=name, wall_s=wall, ms_per_eval=1e3 * wall / h0.size,
                 q_table=str(q_path),
                 f_p=str(opts.per_pixel_completeness or ""),
                 guard="clean (soft_guard=False, max_var=1e6)",
                 fixed={k: float(v) for k, v in fixed.items()})
        results[name] = r
        if "median" in r:
            print(f"    {name}: H0 = {r['median']:.2f} "
                  f"[{r['ci90'][0]:.1f}, {r['ci90'][1]:.1f}] (90%)  "
                  f"{r['ms_per_eval']:.0f} ms/eval  "
                  f"{r['n_finite']}/{h0.size} finite", flush=True)
        with open(resfile, "w") as f:
            json.dump({"results": results}, f)

    if {"q_nofp", "q_fp"} <= set(results) and all(
            "median" in results[k] for k in ("q_nofp", "q_fp")):
        a, b = results["q_nofp"], results["q_fp"]
        sig = 0.5 * (b["ci68"][1] - b["ci68"][0])
        print(f"\nS-3 ON THE Q-TABLE LINE: mask off {a['median']:.2f} -> "
              f"mask on {b['median']:.2f} km/s/Mpc "
              f"({(b['median'] - a['median']) / sig:+.2f} sigma of the masked "
              f"arm's own width)", flush=True)
    print(f"[s3] wrote {resfile}")
    print("S3_FOOTPRINT_DONE", flush=True)


if __name__ == "__main__":
    main()
