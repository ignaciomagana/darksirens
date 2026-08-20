#!/usr/bin/env python3
"""H0 likelihood scans of the clustered-mock GW events, per completeness method.

Evaluates the full dark-siren log-likelihood (catalog-completed redshift prior
+ self-calibrating selection integral) on an H0 grid with every other
parameter FIXED at the generative truth, for each completeness configuration:

  gate_complete           complete catalog at true z (machinery closure: must
                          peak at truth before any other number is read)
  homog_pp   / homog_agg  homogeneous missing branch, per-pixel / aggregate C
  deltag_pp  / deltag_agg legacy 1+b*delta_g               (b_miss = 1 truth)
  qradial_pp / qradial_agg  radial Q tables from output_pp_s0 / output_agg
  qgp3d_pp   / qgp3d_agg    gp3d Q tables   from output_pp_s0 / output_agg

Optionally a 2-D (H0, log10n0) grid for selected configs (--twod) to profile
over the density normalization the budget monopole talks to.

Selection-integral diagnostics (log_mu, Neff) are recorded at every grid
point; a collapsing Neff across the grid means injection coverage is thinning
(DAG rule: the injection set must cover every TRIAL value, not just truth).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: F401  (sys.path + x64)
import jax
import jax.numpy as jnp

from darksirens.utils.cosmology import H0Planck, Om0Planck, w0Fiducial, waFiducial

# The complete-catalog gate file has maxgals=1257 galaxies in its densest
# pixel; the field-KDE builder's default 4096-pixel batch then materializes a
# ~31 GB (pixels, zgrid, maxgals) intermediate. Cap the batch at the two
# import sites (bound early, so patching the source module is not enough).
import functools
from darksirens.redshift import completion as _completion
from darksirens.inference import loaders as _loaders
from darksirens.likelihood import catalog_views as _catalog_views
_bfn_small = functools.partial(_completion.build_field_normalization_inputs,
                               batch_size=256)
_loaders.build_field_normalization_inputs = _bfn_small
_catalog_views.build_field_normalization_inputs = _bfn_small


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--obs-catalog", default="../output/catalog_pixelated_nside_16.h5")
    p.add_argument("--fits-pp", default="../output_pp_s0/fits")
    p.add_argument("--fits-agg", default="../output_agg/fits")
    p.add_argument("--truth-file", default="../output/truth.h5")
    p.add_argument("--outdir", default="results")
    p.add_argument("--h0-min", type=float, default=50.0)
    p.add_argument("--h0-max", type=float, default=90.0)
    p.add_argument("--h0-n", type=int, default=41)
    p.add_argument("--configs", nargs="*", default=None,
                   help="Subset of config names (default: all).")
    p.add_argument("--sel-catalog", default=None,
                   help="Magnitude-pure pixelated catalog for the "
                        "c_mode=selection configs (enables them).")
    p.add_argument("--sel-fit-json", default=None,
                   help="selection_fit.json for the selection configs.")
    p.add_argument("--fits-sel", default=None,
                   help="Fits dir with selection-base Q tables (z052 grid).")
    p.add_argument("--twod", nargs="*",
                   default=["homog_pp", "homog_agg", "qgp3d_agg"],
                   help="Configs that also get the (H0, log10n0) grid.")
    p.add_argument("--n0-halfwidth", type=float, default=0.5,
                   help="log10n0 grid half-width around truth (dex).")
    p.add_argument("--n0-n", type=int, default=21)
    return p.parse_args(argv)


def _base_opts(survey_path, gw_path, sel_path):
    return SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        survey_path=str(survey_path), gw_path=str(gw_path),
        gwselection_path=str(sel_path), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=None, lss_marginalize=False,
        c_mode="per_pixel", catalog_sky_weighting="field",
        # S-3: no mask on a footprint-limited catalog -- the exposed
        # configuration, run here deliberately (loaders' guard).
        allow_unmasked_footprint=True,
        complete_empty_pixel_policy="volume",
        mark_model="none", mark_names=(), sky_model="isotropic",
        drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True,
        fix_survey=True, shared_beta=True, shared_spin=True,
        # pe_event_block bounds the PE catalog-window gather: 16 events x
        # 4096 samples x 1257-galaxy window (complete-catalog gate) ~ 0.7 GB.
        shared_gamma=True, sel_batch_size=65536, pe_event_block=16,
        # Scan context: the 1-nat GWTC-style variance budget is for quoting
        # catalog posteriors; here pe_variance_sum alone is a few nats (300
        # events x finite nsamp) and the soft guard's reward-tracking wall
        # otherwise dominates logL with a spurious H0 slope. A few nats of
        # MC noise is acceptable against O(10^2) method separations.
        selection_neff_soft_guard=True, max_likelihood_variance=25.0,
    )


def _make_logl(opts, fixed, free_labels):
    """Build the jitted logL and a per-call (log_mu, Neff) recorder."""
    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood
    import darksirens.likelihood.core as core

    data = load_all_data(opts)
    opts.resolved_survey_z_depths = (data.get("z_depth"),)
    pop_fid = get_fixed_population_params(opts.pop_model)

    diag = []
    orig = core.compute_selection_term

    def recorder(*a, **k):
        out = orig(*a, **k)
        diag.append(tuple(float(np.asarray(x)) for x in out[:2]))
        return out

    core.compute_selection_term = recorder
    try:
        logl = make_likelihood(opts, data, pop_fid,
                               fixed_parameter_values=fixed)
    finally:
        core.compute_selection_term = orig
    # The recorder was captured inside the jitted closure at trace time; the
    # traced call records nothing, so re-wrap: evaluate selection separately
    # is overkill — instead run logl un-jitted? No: compute_selection_term is
    # traced into the jit, so per-point capture requires calling with jit
    # disabled. We record diagnostics on a sparse sub-grid that way.
    return logl, data


def _eval_grid(logl, thetas):
    vals = np.empty(len(thetas))
    for i, th in enumerate(thetas):
        vals[i] = float(logl(jnp.asarray(th)))
    return vals


def _selection_diag(opts, fixed, thetas):
    """(log_mu, Neff) at each theta, evaluated with jit disabled so the
    monkeypatched recorder actually fires."""
    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood
    import darksirens.likelihood.core as core

    data = load_all_data(opts)
    opts.resolved_survey_z_depths = (data.get("z_depth"),)
    pop_fid = get_fixed_population_params(opts.pop_model)
    rows = []
    orig = core.compute_selection_term

    def recorder(*a, **k):
        out = orig(*a, **k)
        rows.append(tuple(float(np.asarray(x)) for x in out[:2]))
        return out

    core.compute_selection_term = recorder
    try:
        with jax.disable_jit():
            logl = make_likelihood(opts, data, pop_fid,
                                   fixed_parameter_values=fixed)
            for th in thetas:
                logl(jnp.asarray(th))
    finally:
        core.compute_selection_term = orig
    return rows


def main(argv=None):
    args = _parse_args(argv)
    ddir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gw = ddir / "gw_events.h5"
    sel = ddir / "gw_selection.h5"

    import h5py
    with h5py.File(args.truth_file, "r") as f:
        log10n0_true = float(f.attrs["log10n0"])

    fixed_common = {"Om0": Om0Planck, "w0": w0Fiducial, "wa": waFiducial,
                    "log10n0": log10n0_true, "delta": 0.0, "sigma_kde": 0.0}
    # b_miss exists as a parameter only when delta_g is active (use_LSS).

    CONFIGS = {
        "gate_complete": dict(survey=ddir / "catalog_complete.h5"),
        "homog_pp":    dict(survey=args.obs_catalog, c_mode="per_pixel"),
        "homog_agg":   dict(survey=args.obs_catalog, c_mode="aggregate"),
        "deltag_pp":   dict(survey=args.obs_catalog, c_mode="per_pixel", use_LSS=True),
        "deltag_agg":  dict(survey=args.obs_catalog, c_mode="aggregate", use_LSS=True),
        "qradial_pp":  dict(survey=args.obs_catalog, c_mode="per_pixel",
                            q=f"{args.fits_pp}/q_radial.h5"),
        "qradial_agg": dict(survey=args.obs_catalog, c_mode="aggregate",
                            q=f"{args.fits_agg}/q_radial.h5"),
        "qgp3d_pp":    dict(survey=args.obs_catalog, c_mode="per_pixel",
                            q=f"{args.fits_pp}/q_gp3d.h5"),
        "qgp3d_agg":   dict(survey=args.obs_catalog, c_mode="aggregate",
                            q=f"{args.fits_agg}/q_gp3d.h5"),
    }
    if args.sel_catalog:
        import sys as _sys
        from darksirens.redshift.selection import load_selection_fit_json
        sel_fit = load_selection_fit_json(args.sel_fit_json)
        sel_fixed = {"m_lim": float(sel_fit["m_lim"]),
                     "M0hat": float(sel_fit["M0hat"]),
                     "sigma_M": float(sel_fit["sigma_M"])}
        sd_M0 = float(np.sqrt(np.asarray(sel_fit["cov"])[0, 0]))
        # Selection-mode configs on the magnitude-pure catalog, plus the
        # same-catalog aggregate baseline for apples-to-apples, plus the
        # H0-leakage ablation: the H0 curve at M0hat = theta_hat +/- 5 sd_fit
        # must not move (C_sel is H0-invariant; a shift means the magnitude
        # channel leaks into H0).
        CONFIGS.update({
            "homogmp_agg": dict(survey=args.sel_catalog, c_mode="aggregate"),
            "sel": dict(survey=args.sel_catalog, c_mode="selection",
                        fixed=sel_fixed),
            "selq_gp3d": dict(survey=args.sel_catalog, c_mode="selection",
                              fixed=sel_fixed,
                              q=f"{args.fits_sel}/q_gp3d.h5"),
            "sel_M0hat_hi": dict(survey=args.sel_catalog, c_mode="selection",
                                 fixed=dict(sel_fixed,
                                            M0hat=sel_fixed["M0hat"] + 5 * sd_M0)),
            "sel_M0hat_lo": dict(survey=args.sel_catalog, c_mode="selection",
                                 fixed=dict(sel_fixed,
                                            M0hat=sel_fixed["M0hat"] - 5 * sd_M0)),
        })
    names = args.configs or list(CONFIGS)

    h0 = np.linspace(args.h0_min, args.h0_max, args.h0_n)
    n0_grid = np.linspace(log10n0_true - args.n0_halfwidth,
                          log10n0_true + args.n0_halfwidth, args.n0_n)
    # Resume-friendly: merge into any existing results file so a crashed run
    # can be continued with --configs <remaining>.
    results = {}
    prev = outdir / "scan_results.json"
    if prev.exists():
        results = json.load(open(prev)).get("results", {})
    for name in names:
        cfg = CONFIGS[name]
        opts = _base_opts(cfg["survey"], gw, sel)
        opts.c_mode = cfg.get("c_mode", "per_pixel")
        opts.use_LSS = bool(cfg.get("use_LSS", False))
        opts.lss_completion = cfg.get("q")
        fixed = dict(fixed_common, **({"b_miss": 1.0} if opts.use_LSS else {}),
                     **cfg.get("fixed", {}))
        print(f"=== {name}: c_mode={opts.c_mode} use_LSS={opts.use_LSS} "
              f"Q={opts.lss_completion}")
        logl, _ = _make_logl(opts, fixed, ["H0"])
        vals = _eval_grid(logl, [[h] for h in h0])
        # selection diagnostics on a sparse sub-grid (un-jitted, slow)
        sparse = h0[:: max(1, len(h0) // 8)]
        diag = _selection_diag(opts, fixed, [[h] for h in sparse])
        res = {"h0": h0.tolist(), "logl": vals.tolist(),
               "h0_sparse": sparse.tolist(),
               "sel_log_mu": [d[0] for d in diag],
               "sel_neff": [d[1] for d in diag]}
        if name in (args.twod or []):
            fixed2 = {k: v for k, v in fixed_common.items() if k != "log10n0"}
            opts2 = _base_opts(cfg["survey"], gw, sel)
            opts2.c_mode = cfg.get("c_mode", "per_pixel")
            opts2.use_LSS = bool(cfg.get("use_LSS", False))
            opts2.lss_completion = cfg.get("q")
            opts2.fix_survey = False
            logl2, _ = _make_logl(opts2, fixed2, ["H0", "log10n0"])
            grid = np.array([[float(logl2(jnp.asarray([hh, nn])))
                              for nn in n0_grid] for hh in h0])
            res["n0_grid"] = n0_grid.tolist()
            res["logl_2d"] = grid.tolist()
        results[name] = res
        peak = h0[np.argmax(vals)]
        print(f"    peak H0 = {peak:.2f} (truth {H0Planck:.2f}), "
              f"Neff range [{min(res['sel_neff']):.0f}, {max(res['sel_neff']):.0f}]")
        with open(outdir / "scan_results.json", "w") as f:
            json.dump({"H0_true": H0Planck, "log10n0_true": log10n0_true,
                       "results": results}, f)
    print(f"[scan] wrote {outdir/'scan_results.json'}")


if __name__ == "__main__":
    main()
