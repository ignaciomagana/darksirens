#!/usr/bin/env python3
"""Fit the darksirens completeness correction in every available configuration.

Configurations (see README):
  homog    — kernel-ratio C only, homogeneous missing branch (b_miss = 0)
  delta_g  — legacy local-overdensity factor 1 + b·delta_g(pix, z)
  q_radial — Q_LSS table, independent per-pixel radial lognormal (MAP + members)
  q_gp3d   — Q_LSS table, 3-D sphere×z low-rank lognormal (MAP + members)
  q_radial_n0off — radial rebuilt with log10n0 mis-set by +0.3 dex (ablation:
                   density miscalibration is absorbed into Q as spurious
                   redshift structure)

For each configuration this script saves the completion curves
(C_eff, dN_miss, ...) and the assembled redshift prior p(z|pix) on a set of
diagnostic pixels, under BOTH sky-weighting conventions (conditional / field).
All outputs are plain .npz/.h5 so plotting never re-runs JAX-heavy code.

``--c-mode aggregate`` switches the WHOLE chain to the sky-aggregate
completeness base: the Q builders fit residual to Cbar·dN_exp (empty pixels
included as N_obs=0 rows) and the consuming SurveyParams carries
c_mode=C_MODE_AGGREGATE, so every dumped curve/prior uses Cbar.  The default
``per_pixel`` reproduces the legacy reference run exactly.  Run the two modes
in SEPARATE --outdir's: cached Q tables are c_mode-stamped and refuse reuse
across bases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import common  # noqa: F401  (side effect: puts the repo ROOT on sys.path)
from common import INDEXING, fiducial_cosmo, fiducial_survey, make_em_catalog

import jax.numpy as jnp
from darksirens.redshift import zgrid
from darksirens.redshift.completion import (
    C_MODE_AGGREGATE,
    C_MODE_SELECTION,
    build_field_delta_g_inputs,
    build_field_depth_inputs,
    build_field_lss_q_inputs,
    build_field_lss_q_member_inputs,
    build_field_normalization_inputs,
    completion_curves,
    compute_lss_overdensity,
)
from darksirens.redshift.prior import (
    eval_redshift_prior_members_with_state,
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.cli.build_lognormal_completion import build_completion
from darksirens.redshift.lognormal_completion import save_lss_completion_hdf5


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="output")
    p.add_argument("--catalog", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--n-members", type=int, default=16)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel per-pixel radial MAP solves.")
    p.add_argument("--n-prior-pixels", type=int, default=12)
    p.add_argument("--builder-corr-mpc", default="truth",
                   help="Radial correlation length [Mpc] the Q builders are "
                        "conditioned on: 'truth' (default) matches the injected "
                        "field so the closure test measures the estimator; "
                        "'default' keeps the shipped SurveyParams fiducial "
                        "(50 Mpc — NOT CLI-exposed in production, and far below "
                        "the gp3d node resolution); or a number.")
    p.add_argument("--c-mode", default="per_pixel",
                   choices=["per_pixel", "aggregate", "selection"],
                   help="Completeness estimator: 'per_pixel' (legacy default, "
                        "reproduces the reference run exactly), 'aggregate' "
                        "(one sky-aggregate Cbar; the Q builders fit against "
                        "the Cbar base with empty pixels as N_obs=0 rows, and "
                        "the survey consumes the curves with c_mode=1 so the "
                        "dumped priors/curves use the SAME base), or "
                        "'selection' (the parametric C_sel(z; theta_hat) from "
                        "--selection-fit; same all-pixel fit as aggregate but "
                        "no counts in the budget).")
    p.add_argument("--selection-fit", default=None,
                   help="selection_fit.json from darksirens_fit_selection "
                        "(required with --c-mode selection).")
    p.add_argument("--builder-sigma", default=None,
                   help="Override the builders' lss_sigma field amplitude "
                        "(default: keep the shipped SurveyParams fiducial).")
    p.add_argument("--gp3d-nz-nodes", type=int, default=None,
                   help="gp3d radial inducing-node count (default: builder's "
                        "historical 6; the builder hard-errors if the spacing "
                        "cannot resolve ls_z).")
    p.add_argument("--gp3d-nsph-nodes", type=int, default=None,
                   help="gp3d sphere inducing-node count (default 32).")
    p.add_argument("--gp3d-z-node-hi", default=None,
                   help="gp3d radial node range upper edge in z (default: "
                        "package zgrid max).")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["q_radial", "q_gp3d", "q_radial_n0off", "priors"])
    return p.parse_args(argv)


def _attach_field_inputs(cat, *, logq_map=None, logq_members=None, delta_g=None):
    """Attach the survey-global field_* inputs the field normalizer requires."""
    fni = build_field_normalization_inputs(cat.zgals, cat.wgals, cat.ngals)
    upd = dict(
        field_dN_obs_s=fni.dN_obs_s, field_n_empty=fni.n_empty,
        field_N_obs_total=fni.N_obs_total, field_occupied_pixels=fni.occupied_pixels,
    )
    depth = build_field_depth_inputs(cat.zgals, cat.dzgals, cat.wgals, cat.ngals)
    upd.update(field_depth_z=depth.z, field_depth_dz=depth.dz, field_depth_c=depth.c)
    n_pix = int(np.asarray(cat.zgals).shape[0])
    if logq_map is not None:
        q_occ, q_empty = build_field_lss_q_inputs(logq_map, fni.occupied_pixels, n_pix)
        upd.update(field_lss_q=q_occ, field_lss_q_empty_sum=q_empty)
    if logq_members is not None:
        qm, qm_empty = build_field_lss_q_member_inputs(
            logq_members, fni.occupied_pixels, n_pix)
        upd.update(field_lss_q_members=qm, field_lss_q_empty_sum_members=qm_empty)
    if delta_g is not None:
        upd.update(field_delta_g=build_field_delta_g_inputs(delta_g, fni.occupied_pixels))
    return cat._replace(**upd)


def _save_curves(path, curves):
    out = {}
    for name in ("f", "dN_miss", "C_eff", "N_miss", "base_miss", "N_miss_members"):
        val = getattr(curves, name, None)
        if val is not None:
            out[name] = np.asarray(val)
    np.savez_compressed(path, **out)


def _eval_priors(cosmo, survey, cat, pixels, *, weighting, members):
    """Assembled prior p(z|pix) on zgrid for each diagnostic pixel."""
    state = prepare_redshift_prior_state(
        "dark_sirens", cosmo, survey, cat,
        catalog_sky_weighting=weighting)
    z = jnp.asarray(zgrid)
    logp = np.empty((len(pixels), int(zgrid.size)))
    for i, p in enumerate(pixels):
        pix = jnp.full(int(zgrid.size), int(p), jnp.int32)
        logp[i] = np.asarray(eval_redshift_prior_with_state(
            "dark_sirens", state, z, pix, cosmo, survey, cat,
            catalog_sky_weighting=weighting))
    out = {"logp": logp}
    if members:
        lpm = np.empty((0,))
        rows = []
        for p in pixels:
            pix = jnp.full(int(zgrid.size), int(p), jnp.int32)
            rows.append(np.asarray(eval_redshift_prior_members_with_state(
                "dark_sirens", state, z, pix, cosmo, survey, cat)))
        lpm = np.stack(rows, axis=1)                      # (M, n_pix_sel, N_grid)
        out["logp_members"] = lpm
    return out


def main(argv=None):
    args = _parse_args(argv)
    out = Path(args.outdir)
    fits = out / "fits"
    fits.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.truth, "r") as f:
        n0 = float(f.attrs["n0"])
        log10n0 = float(f.attrs["log10n0"])
        corr_ang = float(f.attrs["corr_ang"])
        corr_mpc_truth = float(f.attrs["corr_mpc"])
        logq_truth = np.asarray(f["logq_truth_on_zgrid"])
    cosmo = fiducial_cosmo()

    # Condition the offline builders on the requested radial correlation
    # length.  lss_corr_length_mpc is a FIXED SurveyParams hyperparameter with
    # no builder CLI knob, so we override the builder's fiducial factory here
    # (experiment-local; production runs keep the shipped 50 Mpc).
    if args.builder_corr_mpc != "default":
        corr_mpc = (corr_mpc_truth if args.builder_corr_mpc == "truth"
                    else float(args.builder_corr_mpc))
        import darksirens.cli.build_lognormal_completion as _blc
        _orig_fiducial = _blc._fiducial_cosmo_survey

        def _patched(log10n0=None, delta=None):
            c, s = _orig_fiducial(log10n0=log10n0, delta=delta)
            return c, s._replace(lss_corr_length_mpc=corr_mpc)

        _blc._fiducial_cosmo_survey = _patched
        print(f"[fit] builders conditioned on lss_corr_length_mpc={corr_mpc}")

    nside, cat0, z_depth = make_em_catalog(args.catalog)
    assert z_depth is not None, "catalog is missing the z_depth attr"
    survey = fiducial_survey(n0, b_miss=1.0, z_depth=z_depth)
    # Aggregate mode: the consuming survey must request the SAME completeness
    # base the Q builders fit against (one sky-aggregate Cbar broadcast to all
    # pixels).  c_mode is structural (never traced): per_pixel rides the
    # default None, aggregate the concrete int C_MODE_AGGREGATE.
    if args.c_mode == "aggregate":
        survey = survey._replace(c_mode=C_MODE_AGGREGATE)
    selection_fit = None
    if args.c_mode == "selection":
        from darksirens.redshift.selection import load_selection_fit_json
        if args.selection_fit is None:
            raise SystemExit("--c-mode selection requires --selection-fit")
        selection_fit = load_selection_fit_json(args.selection_fit)
        survey = survey._replace(
            c_mode=C_MODE_SELECTION,
            m_lim=float(selection_fit["m_lim"]),
            M0hat=float(selection_fit["M0hat"]),
            sigma_M=float(selection_fit["sigma_M"]))
    survey_homog = survey._replace(b_miss=0.0)
    n_pix = int(np.asarray(cat0.zgals).shape[0])
    print(f"[fit] nside={nside} n_pix={n_pix} z_depth={z_depth} "
          f"log10n0={log10n0:.4f} c_mode={args.c_mode}")

    # Diagnostic pixels: occupied, spanning the truth-Q quantiles of the
    # depth-limited mean |logQ| (extremes + evenly spaced interior).
    ngals = np.asarray(cat0.ngals)
    below = np.asarray(zgrid) <= z_depth
    score = logq_truth[:, below].mean(axis=1)
    occ = np.nonzero(ngals > 0)[0]
    order = occ[np.argsort(score[occ])]
    sel = order[np.unique(np.linspace(0, order.size - 1, args.n_prior_pixels).astype(int))]
    np.save(fits / "prior_pixels.npy", sel)
    print(f"[fit] diagnostic pixels: {sel.tolist()}")

    # ---- Q-table builds ----------------------------------------------------
    # Builder knobs shared by every build: the completeness base (c_mode) and
    # the S0c hyperparameter overrides.  None means "keep the builder default"
    # so the legacy per_pixel run stays bit-identical to the reference.
    common_kw = dict(c_mode=args.c_mode, selection_fit=selection_fit)
    if args.builder_sigma is not None:
        common_kw["lss_sigma"] = float(args.builder_sigma)
    gp3d_kw = {}
    if args.gp3d_nz_nodes is not None:
        gp3d_kw["gp3d_nz_nodes"] = int(args.gp3d_nz_nodes)
    if args.gp3d_nsph_nodes is not None:
        gp3d_kw["gp3d_nsph_nodes"] = int(args.gp3d_nsph_nodes)
    if args.gp3d_z_node_hi is not None:
        gp3d_kw["gp3d_z_node_hi"] = float(args.gp3d_z_node_hi)

    builds = {}
    if "q_radial" not in args.skip:
        builds["q_radial"] = dict(mode="radial", log10n0=log10n0, **common_kw)
    if "q_gp3d" not in args.skip:
        builds["q_gp3d"] = dict(mode="gp3d", log10n0=log10n0,
                                lss_corr_length_ang=corr_ang,
                                **common_kw, **gp3d_kw)
    if "q_radial_n0off" not in args.skip:
        builds["q_radial_n0off"] = dict(mode="radial", log10n0=log10n0 + 0.3,
                                        **common_kw)

    tables = {}
    for name, kw in builds.items():
        path = fits / f"{name}.h5"
        if path.exists():
            from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5
            loaded = load_lss_completion_hdf5(str(path))
            # Fail-closed, mirroring catalogs/lss.py: Q is the fit residual to
            # its completeness base, so a cached table from the other c_mode
            # (absent attr = legacy per-pixel-base) must not be reused.
            table_c_mode = loaded.get("c_mode") or "per_pixel"
            if table_c_mode != args.c_mode:
                raise ValueError(
                    f"cached table {path} was built against the "
                    f"{table_c_mode!r} completeness base but this run uses "
                    f"--c-mode {args.c_mode}; delete it or use a fresh "
                    "--outdir.")
            tables[name] = (loaded["logq_map"], loaded["logq_members"])
            print(f"[build] {name}: reusing {path}")
            continue
        print(f"[build] {name}: mode={kw['mode']} ...")
        mem = 0 if name == "q_radial_n0off" else args.n_members
        logq_map, logq_members, diag = build_completion(
            args.catalog, n_members=mem, seed=args.seed,
            workers=args.workers, delta=0.0, **kw)
        # Pop the S0b budget stamps into their dedicated attrs (same as the
        # production builder CLI) so the loader does not flag the cached
        # tables as legacy non-renormalized files on reuse.
        budget_mono = diag.pop("budget_monopole_logq", None)
        budget_stamp = diag.pop("budget_renormalized", None)
        save_lss_completion_hdf5(
            str(path), logq_map=logq_map, logq_members=logq_members,
            zgrid=np.asarray(zgrid), indexing="global", metadata=diag,
            c_mode=args.c_mode,
            budget_renormalized=budget_stamp,
            budget_monopole_logq=budget_mono)
        with open(fits / f"{name}_diagnostics.json", "w") as f:
            json.dump({k: v for k, v in diag.items()
                       if isinstance(v, (int, float, str, bool))}, f, indent=2)
        tables[name] = (np.asarray(logq_map),
                        None if logq_members is None else np.asarray(logq_members))
        print(f"[build] {name}: wrote {path}")

    # ---- legacy delta_g ----------------------------------------------------
    delta_g = np.asarray(compute_lss_overdensity(cat0.zgals, nside, ngals=cat0.ngals))
    np.savez_compressed(fits / "delta_g.npz", delta_g=delta_g)
    print(f"[fit] delta_g table {delta_g.shape}, range "
          f"[{delta_g.min():.3f}, {delta_g.max():.3f}]")

    # ---- per-config catalogs, curves, priors -------------------------------
    configs = {"homog": (survey_homog, cat0)}
    configs["delta_g"] = (survey, cat0._replace(delta_g_pix_z=jnp.asarray(delta_g)))
    for name, (logq_map, logq_members) in tables.items():
        lss = dict(lss_completion_logq=jnp.asarray(logq_map),
                   lss_completion_indexing=INDEXING["global"])
        if logq_members is not None:
            lss["lss_completion_logq_members"] = jnp.asarray(logq_members)
        _, cat_q, _ = make_em_catalog(args.catalog, **lss)
        configs[name] = (survey, cat_q)

    for name, (svy, cat) in configs.items():
        curves = completion_curves(cosmo, svy, cat)
        _save_curves(fits / f"curves_{name}.npz", curves)
        print(f"[fit] curves_{name}: f mean={float(np.asarray(curves.f).mean()):.4f}")

        if "priors" in args.skip:
            continue
        has_members = getattr(cat, "lss_completion_logq_members", None) is not None
        pri_cond = _eval_priors(cosmo, svy, cat, sel,
                                weighting="conditional", members=has_members)
        np.savez_compressed(fits / f"priors_{name}_conditional.npz", **pri_cond)
        # Field weighting requires the survey-global field_* inputs.
        dgt = np.asarray(cat.delta_g_pix_z)
        cat_f = _attach_field_inputs(
            cat,
            logq_map=(np.asarray(cat.lss_completion_logq)
                      if getattr(cat, "lss_completion_logq", None) is not None else None),
            logq_members=(np.asarray(cat.lss_completion_logq_members)
                          if has_members else None),
            delta_g=dgt if dgt.shape[0] != 1 else None,
        )
        pri_field = _eval_priors(cosmo, svy, cat_f, sel,
                                 weighting="field", members=has_members)
        np.savez_compressed(fits / f"priors_{name}_field.npz", **pri_field)
        print(f"[fit] priors_{name}: conditional+field on {sel.size} pixels")

    # Observed-branch ingredients for the component-separated plot: the
    # per-pixel smoothed observed density (the same KDE the completion uses)
    # and the observed counts.
    fni = build_field_normalization_inputs(cat0.zgals, cat0.wgals, cat0.ngals)
    np.savez_compressed(
        fits / "observed_branch.npz",
        dN_obs_s=np.asarray(fni.dN_obs_s, dtype=np.float64),
        occupied_pixels=np.asarray(fni.occupied_pixels),
        N_obs=np.asarray(ngals, dtype=np.float64),
    )
    print("[fit] done")


if __name__ == "__main__":
    main()
