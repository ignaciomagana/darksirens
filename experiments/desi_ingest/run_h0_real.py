"""Real-GWTC H0 grid scans on the DESI union catalog, selection channel.

Configurations (all 44 S>=0.495 events + the MATCHED betaS -- never pair
these events with a plain injection set):

  per_pixel     legacy per-pixel completeness on the n64 catalog (family
                comparison / continuity with the parity gate)
  sel           c_mode=selection, homogeneous missing branch (no Q)
  selq_radial   + radial Q table        (data/fits/q_radial.h5)
  selq_gp3d     + gp3d Q table          (data/fits/q_gp3d.h5)
  sel_M0hat_hi/lo  theta shifted +/- 5 fit-sd: the H0 curve must not move
                (C_sel is H0-invariant; motion = magnitude-channel leakage)

theta is FIXED at theta_hat for the grid scans (the full sampled-theta run
is the nested-sampling follow-up); m_lim pinned, K(z) template from the fit.
H0 grid matches the reference convention (20..140 step 1, flat prior).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import common as C  # noqa: F401  (pins DARKSIRENS_ZMAX=0.75; must be first)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402


def _opts(survey, c_mode, q=None, universe="dark_sirens"):
    return SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks",
        # "dark_sirens" carries the missing-galaxy branch (c_mode / C_sel / Q
        # all live there); "dark_sirens_complete" treats the catalog as
        # complete and ignores them entirely -- reference channel only.
        universe_model=universe,
        survey_path=str(survey), gw_path=str(C.GW_SAMPLES_44),
        gwselection_path=str(C.BETA_S0495), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=(str(q) if q else None),
        lss_marginalize=False,
        c_mode=c_mode, catalog_sky_weighting="field",
        # S-3: no mask on a footprint-limited catalog -- the exposed
        # configuration, run here deliberately (loaders' guard).
        allow_unmasked_footprint=True,
        complete_empty_pixel_policy="zero",
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
    p.add_argument("--outdir", default="data/h0_real")
    args = p.parse_args(argv)

    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    survey = C.DATA_DIR / "pixelated_n64" / "catalog_pixelated_nside_64.h5"
    fit_json = C.DATA_DIR / "selection_fit_union.json"
    sel = load_selection_fit_json(fit_json)
    sd_M0 = float(np.sqrt(np.asarray(sel["cov"])[0, 0]))
    theta = {"m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
             "sigma_M": float(sel["sigma_M"])}
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))

    fixed_common = {"Om0": 0.3075, "sigma_kde": 0.003,
                    "log10n0": float(cal["log10n0"]),
                    "delta": float(cal["delta"])}

    CONFIGS = {
        "complete":   dict(c_mode="per_pixel", universe="dark_sirens_complete"),
        "per_pixel":  dict(c_mode="per_pixel"),
        "sel":        dict(c_mode="selection", fixed=theta, fit=True),
        "selq_radial": dict(c_mode="selection", fixed=theta, fit=True,
                            q=C.DATA_DIR / "fits" / "q_radial.h5"),
        "selq_gp3d":  dict(c_mode="selection", fixed=theta, fit=True,
                           q=C.DATA_DIR / "fits" / "q_gp3d.h5"),
        "sel_M0hat_hi": dict(c_mode="selection", fit=True,
                             fixed=dict(theta, M0hat=theta["M0hat"] + 5 * sd_M0)),
        "sel_M0hat_lo": dict(c_mode="selection", fit=True,
                             fixed=dict(theta, M0hat=theta["M0hat"] - 5 * sd_M0)),
        # Stratified north/south selection (PR-3): common-mode theta = the
        # SOUTH stratum's fit, fixed inter-stratum offsets, per-stratum
        # V_empty in the global normalizer. No Q (guard: stratified builder
        # base not implemented).
        "sel_strat": dict(c_mode="selection", strat=True),
        # Stratified base + stratified Q table (PR-A validation): per-pixel
        # C_sel rows in the table's fixed budget AND per-stratum empty-pixel
        # Q sums in the global normalizer.
        "selq_strat": dict(c_mode="selection", strat=True,
                           q=C.DATA_DIR / "fits" / "q_radial_strat.h5"),
    }
    names = args.configs or list(CONFIGS)
    h0 = np.arange(args.h0_min, args.h0_max + 0.5 * args.h0_step, args.h0_step)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    resfile = outdir / "h0_real_scans.json"
    results = json.load(open(resfile))["results"] if resfile.exists() else {}

    for name in names:
        cfg = CONFIGS[name]
        opts = _opts(survey, cfg["c_mode"], cfg.get("q"),
                     cfg.get("universe", "dark_sirens"))
        if cfg.get("fit"):
            opts.selection_fit = str(fit_json)
            # Grid scan fixes theta; the Gaussian prior machinery is for the
            # sampled-theta run. Record the fit for the K(z)/provenance path
            # (per-catalog spelling: the scalar selection_fit_kcorr was
            # RETIRED by the K>=2 plumbing, #344).
            _selj = load_selection_fit_json(fit_json)
            opts.selection_kcorr_by_catalog = [
                tuple(_selj["k_corr_coeffs"]) or None]
        if cfg.get("strat"):
            from darksirens.redshift.selection import load_selection_fit_strata
            strata = load_selection_fit_strata(
                C.DATA_DIR / "selection_fit_ns.json")
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
            with h5py.File(C.DATA_DIR / "stratum_map_ns_nside64.h5", "r") as f:
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
        }
        print(f"    {name}: H0 = {results[name]['median']:.2f} "
              f"[{results[name]['ci68'][0]:.1f}, {results[name]['ci68'][1]:.1f}]",
              flush=True)
        with open(resfile, "w") as f:
            json.dump({"results": results}, f)
    print(f"[h0_real] wrote {resfile}")


if __name__ == "__main__":
    main()
