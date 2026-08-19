"""FULL-RANGE H0 scan with the LATENT field arm (field-level PR-6a).

The table-mode companion is ``run_h0_scans.py``; this script adds the arm the
ladder exists to produce and the controls it must be read against.  Same
259 events, same plain injection set, same recalibrated budget, same fixed
theta -- the ONLY thing that varies across arms is where ``Q`` comes from.

Arms
----
``nofp``    no per-pixel completeness, no field.  The oldest baseline.
``fp``      ``--per_pixel_completeness`` on, no field.  This is the control
            that matters: PR-6a measured that 97.2% of latent mode's runtime
            overhead, and 91% of its Tier-B shift, is the ``f_p`` channel
            rather than the field.  Comparing ``latent`` against ``nofp``
            would credit the field with ``f_p``'s effect.
``latent``  the deliverable: ``f_p`` on, field on, 8-member ensemble.

Guard convention
----------------
The clean arm (``selection_neff_soft_guard=False``,
``max_likelihood_variance=1e6``) -- PR-0's convention, NOT the shipped scan's.
PR-5b measured why this is not a detail: under the shipped soft guard the
member spread reads 24-40 nats against the clean arm's 1.5e-2, 610x to
34,206x larger and non-monotone in H0, because the guard's wall responds to
member-dependent ``Neff``.  A latent-vs-table comparison run in that
convention measures the guard, not the field.  The shipped convention's own
139 rail was shaped by the same wall.

This script does NOT produce a sampled-theta posterior.  It is a grid scan at
fixed theta, which is what the 259-event line has always quoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import common as C  # noqa: F401  (pins DARKSIRENS_ZMAX=6.0; must be first)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402


def _opts(*, per_pixel_completeness=None, latent_artifact=None):
    """Options shared by every arm; the two kwargs are the only differences."""
    o = SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks",
        universe_model="dark_sirens",
        survey_path=str(C.SURVEY_N64), gw_path=str(C.GW_259),
        gwselection_path=str(C.INJ_PLAIN), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=None,
        c_mode="selection", catalog_sky_weighting="field",
        complete_empty_pixel_policy="zero",
        mark_model="none", mark_names=(), sky_model="isotropic",
        drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True,
        fix_survey=True, shared_beta=True, shared_spin=True,
        shared_gamma=True, sel_batch_size=16384, pe_event_block=8,
        # THE CLEAN ARM -- see the module docstring.  Not the shipped default.
        selection_neff_soft_guard=False,
        max_likelihood_variance=1e6,
        # Latent plumbing; None/"table" is the shipped path exactly.
        per_pixel_completeness=per_pixel_completeness,
        # S-3: the ``nofp`` arm runs the exposed configuration on purpose --
        # 38% of this line's sky is off-footprint and without f_p the missing
        # budget models it as Cbar-complete.  That IS the control.
        allow_unmasked_footprint=per_pixel_completeness is None,
        lss_field_mode="latent" if latent_artifact else "table",
        lss_field_artifact=latent_artifact,
        lss_marginalize=bool(latent_artifact),
    )
    sel = load_selection_fit_json(str(C.FIT_JSON))
    o.selection_fit = str(C.FIT_JSON)
    o.selection_kcorr_by_catalog = [tuple(sel["k_corr_coeffs"]) or None]
    return o


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h0-min", type=float, default=20.0)
    p.add_argument("--h0-max", type=float, default=140.0)
    p.add_argument("--h0-step", type=float, default=1.0)
    p.add_argument("--arms", nargs="*",
                   default=["nofp", "fp", "latent"])
    p.add_argument("--anchor", default=None,
                   help="latent anchor artifact (required by the latent arm)")
    p.add_argument("--mth-map", default=None,
                   help="per-pixel completeness map (fp and latent arms)")
    p.add_argument("--b-gw", type=float, default=1.0,
                   help="b_GW; in latent mode b_miss IS b_GW (PLAN 4.3)")
    p.add_argument("--outdir", default="data/h0_latent")
    args = p.parse_args(argv)

    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    sel = load_selection_fit_json(str(C.FIT_JSON))
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    if "n0_true_Mpc3_no_evo" not in cal:
        raise SystemExit(
            "n0_calibration.json predates the (1+z)^delta normalization fix.")

    # ``b_miss`` is arm-DEPENDENT and the guard in inference/prior.py is right
    # to insist on it: with ``--use_lss`` off and no Q table, ``delta_g`` is the
    # all-zero dummy, so ``1 + alpha_miss*b_miss*delta_g == 1`` for any value
    # and pinning it would assert a parameter the configuration does not have.
    # In LATENT mode the same symbol IS ``b_GW`` -- the bias with which GW hosts
    # trace the field, which enters ``logQ`` directly and is genuinely (if
    # weakly) identified.  PLAN 4.3 inverts the guard for exactly this reason,
    # so the fixed value is supplied to the latent arm ONLY.
    fixed_base = {"Om0": C.OM0, "sigma_kde": 0.003,
                  "log10n0": float(cal["log10n0"]),
                  "delta": float(cal["delta"]),
                  "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
                  "sigma_M": float(sel["sigma_M"])}

    ARMS = {
        "nofp":   dict(),
        "fp":     dict(per_pixel_completeness=args.mth_map),
        "latent": dict(per_pixel_completeness=args.mth_map,
                       latent_artifact=args.anchor),
    }
    h0 = np.arange(args.h0_min, args.h0_max + 0.5 * args.h0_step, args.h0_step)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    resfile = outdir / "h0_latent_scans.json"
    results = json.load(open(resfile))["results"] if resfile.exists() else {}

    for name in args.arms:
        kw = ARMS[name]
        if name in ("fp", "latent") and not args.mth_map:
            raise SystemExit(f"arm {name!r} needs --mth-map")
        if name == "latent" and not args.anchor:
            raise SystemExit("arm 'latent' needs --anchor")
        opts = _opts(**kw)
        print(f"=== {name}: field_mode={opts.lss_field_mode} "
              f"f_p={'on' if kw.get('per_pixel_completeness') else 'off'} "
              f"marginalize={opts.lss_marginalize}", flush=True)
        data = load_all_data(opts)
        opts.resolved_survey_z_depths = (data.get("z_depth"),)
        pop_fid = get_fixed_population_params(opts.pop_model)
        fixed = dict(fixed_base)
        if opts.lss_field_mode == "latent":
            fixed["b_miss"] = args.b_gw
        logl = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)

        import time
        vals = np.empty(h0.size)
        t0 = time.time()
        for i, h in enumerate(h0):
            vals[i] = float(logl(jnp.asarray([h])))
            if i % 20 == 0:
                print(f"  H0={h:6.1f}: logL={vals[i]:14.4f}"
                      f"  ({time.time()-t0:.0f}s)", flush=True)
        wall = time.time() - t0

        finite = np.isfinite(vals)
        if not finite.any():
            print(f"    {name}: ALL NODES -inf -- recording and continuing",
                  flush=True)
            results[name] = {"h0": h0.tolist(), "logl": vals.tolist(),
                             "all_nonfinite": True, "wall_s": wall}
        else:
            v = np.where(finite, vals, -np.inf)
            pdf = np.exp(v - v.max())
            pdf = np.where(finite, pdf, 0.0)
            pdf /= np.trapz(pdf, h0)
            cdf = np.concatenate([[0], np.cumsum(
                0.5 * (pdf[1:] + pdf[:-1]) * np.diff(h0))])
            cdf /= cdf[-1]
            q = lambda a: float(np.interp(a, cdf, h0))  # noqa: E731
            results[name] = {
                "h0": h0.tolist(), "logl": vals.tolist(), "pdf": pdf.tolist(),
                "median": q(0.5), "ci68": [q(0.16), q(0.84)],
                "ci90": [q(0.05), q(0.95)], "map": float(h0[np.argmax(pdf)]),
                "n_finite": int(finite.sum()), "wall_s": wall,
                "ms_per_eval": 1e3 * wall / h0.size,
                "field_mode": opts.lss_field_mode,
                "f_p": bool(kw.get("per_pixel_completeness")),
                "marginalize": bool(opts.lss_marginalize),
                "b_gw": args.b_gw,
                "guard": "clean (soft_guard=False, max_var=1e6)",
                "anchor": args.anchor, "fixed": {k: float(v) for k, v in fixed.items()},
            }
            r = results[name]
            print(f"    {name}: H0 = {r['median']:.2f} "
                  f"[{r['ci90'][0]:.1f}, {r['ci90'][1]:.1f}] (90%)  "
                  f"{r['ms_per_eval']:.0f} ms/eval  "
                  f"{r['n_finite']}/{h0.size} finite", flush=True)
        with open(resfile, "w") as f:
            json.dump({"results": results}, f)
    print(f"[h0_latent] wrote {resfile}")


if __name__ == "__main__":
    main()
