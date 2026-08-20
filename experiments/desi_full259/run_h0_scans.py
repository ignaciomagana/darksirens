"""FULL-RANGE H0 grid scans: GWTC-5 259 events, plain injections, NO cuts.

Owner decision (2026-08-10): no windows, no z cuts anywhere -- "the point
of completeness corrections is not to have cuts".  The pair (both fully
processed, both at pe cosmology Om0 = 0.3089, both UNWINDOWED):

  events      GWTC5_S1_spin_fullnuts/gwsamples_bbh_whitelist_all_events_final.h5
              (gwcat-1.0, 259 BBH, 4096 samp/ev, dL to 27.1 Gpc)
  injections  GWTC5_S1_spin_fullnuts/selection_o3o4ab_allsky.h5
              (gwcat-selection-1.0, allsky, 1.07M detected of Ndraw 944M,
               pdraw carried, NO p_pass, T_obs 2.59 yr)

KNOWN CAVEAT for the writeup: the injection campaigns are O3+O4ab while
the 259 events span O1-O5; the campaign-mix representativeness is an
owner-level modeling choice, inherited here.

Grid: this line's common.py pins DARKSIRENS_ZMAX = 6.0
(z(27119 Mpc; H0=140, Om0=0.3089) = 5.735 + margin; 1086 log-spaced
nodes).  The n0 calibration and Q tables are rebuilt on THIS grid in this
directory; the desi_ingest 0.75-grid products are incompatible (the Q
loader enforces the match) and untouched.

Every config runs at the RECALIBRATED budget ((1+z)^delta normalization
fix) and theta FIXED at theta_hat (grid-scan convention; the
sampled-theta NS run lives on the 44-event line).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import common as C  # noqa: F401  (pins the FULL-RANGE ZMAX=6.0; must be first)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402

GW_259 = str(C.GW_259)
INJ_PLAIN = str(C.INJ_PLAIN)


def _opts(survey, c_mode, q=None, universe="dark_sirens",
          empty_policy="zero"):
    return SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks",
        universe_model=universe,
        survey_path=str(survey), gw_path=GW_259,
        gwselection_path=INJ_PLAIN, pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=(str(q) if q else None),
        lss_marginalize=False,
        c_mode=c_mode, catalog_sky_weighting="field",
        # S-3: no mask on a footprint-limited catalog -- the exposed
        # configuration, run here deliberately (loaders' guard).
        allow_unmasked_footprint=True,
        complete_empty_pixel_policy=empty_policy,
        mark_model="none", mark_names=(), sky_model="isotropic",
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
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--outdir", default="data/h0_scans")
    args = p.parse_args(argv)

    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    survey = C.SURVEY_N64
    fit_json = C.FIT_JSON
    sel = load_selection_fit_json(fit_json)
    theta = {"m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
             "sigma_M": float(sel["sigma_M"])}
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    if "n0_true_Mpc3_no_evo" not in cal:
        raise SystemExit(
            "n0_calibration.json predates the (1+z)^delta normalization fix; "
            "rerun calibrate_n0.py first -- the whole point of this scan is "
            "the recalibrated, beta-consistent construction.")

    fixed_common = {"Om0": C.OM0, "sigma_kde": 0.003,
                    "log10n0": float(cal["log10n0"]),
                    "delta": float(cal["delta"])}

    CONFIGS = {
        "per_pixel":  dict(c_mode="per_pixel"),
        "sel":        dict(c_mode="selection", fixed=theta, fit=True),
        "selq_radial": dict(c_mode="selection", fixed=theta, fit=True,
                            q=C.DATA_DIR / "fits" / "q_radial.h5"),
        "sel_strat": dict(c_mode="selection", strat=True),
    }
    names = args.configs or list(CONFIGS)
    h0 = np.arange(args.h0_min, args.h0_max + 0.5 * args.h0_step, args.h0_step)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    resfile = outdir / "h0_full259_scans.json"
    results = json.load(open(resfile))["results"] if resfile.exists() else {}

    for name in names:
        cfg = CONFIGS[name]
        opts = _opts(survey, cfg["c_mode"], cfg.get("q"),
                     cfg.get("universe", "dark_sirens"),
                     empty_policy=cfg.get("empty_policy", "zero"))
        if cfg.get("fit"):
            opts.selection_fit = str(fit_json)
            _selj = load_selection_fit_json(fit_json)
            opts.selection_kcorr_by_catalog = [
                tuple(_selj["k_corr_coeffs"]) or None]
        if cfg.get("strat"):
            from darksirens.redshift.selection import load_selection_fit_strata
            strata = load_selection_fit_strata(
                C.FIT_NS_JSON)
            ref = strata[0]
            opts.selection_kcorr_by_catalog = [
                tuple(ref["k_corr_coeffs"]) or None]
            opts.selection_strata_by_catalog = [[
                (float(s["m_lim"]),
                 float(s["M0hat"]) - float(ref["M0hat"]),
                 float(s["sigma_M"]) / float(ref["sigma_M"]))
                for s in strata]]
            cfg = dict(cfg, fixed={"m_lim": float(ref["m_lim"]),
                                   "M0hat": float(ref["M0hat"]),
                                   "sigma_M": float(ref["sigma_M"])})
        fixed = dict(fixed_common, **cfg.get("fixed", {}))
        print(f"=== {name}: c_mode={opts.c_mode} Q={opts.lss_completion}",
              flush=True)
        data = load_all_data(opts)
        if cfg.get("strat"):
            import h5py
            with h5py.File(C.STRATUM_MAP, "r") as f:
                data["pixel_stratum_map"] = f["stratum_map"][...]
        opts.resolved_survey_z_depths = (data.get("z_depth"),)
        pop_fid = get_fixed_population_params(opts.pop_model)
        logl = make_likelihood(opts, data, pop_fid,
                               fixed_parameter_values=fixed)
        vals = np.empty(h0.size)
        for i, h in enumerate(h0):
            vals[i] = float(logl(jnp.asarray([h])))
            if i % 20 == 0:
                print(f"  H0={h:.0f}: logL={vals[i]:.3f}", flush=True)
        pdf = np.exp(vals - vals.max())
        pdf /= np.trapz(pdf, h0)
        cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                             * np.diff(h0))])
        cdf /= cdf[-1]
        q = lambda a: float(np.interp(a, cdf, h0))  # noqa: E731
        results[name] = {
            "h0": h0.tolist(), "logl": vals.tolist(), "pdf": pdf.tolist(),
            "median": q(0.5), "ci68": [q(0.16), q(0.84)],
            "ci90": [q(0.05), q(0.95)], "map": float(h0[np.argmax(pdf)]),
            "fixed": {k: float(v) for k, v in fixed.items()},
            "q_table": str(opts.lss_completion),
            "gw_path": GW_259, "gwselection_path": INJ_PLAIN,
            "n0_calibration": "recal_evo_fix",
        }
        print(f"    {name}: H0 = {results[name]['median']:.2f} "
              f"[{results[name]['ci68'][0]:.1f}, {results[name]['ci68'][1]:.1f}]",
              flush=True)
        with open(resfile, "w") as f:
            json.dump({"results": results}, f)
    print(f"[h0_full259] wrote {resfile}")


if __name__ == "__main__":
    main()
