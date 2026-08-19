"""PR-0 item 3: osc_H0[ logL(Q on) - logL(Q ≡ 1) ] at the anchor, guard-decomposed.

The existing artifact (desi_full259/data/h0_scans/h0_full259_scans.json,
2026-08-10) compares sel (Q ≡ 1) vs selq_radial (shipped radial table) with
selection_neff_soft_guard=True and reports a raw oscillation of ~2.1e5 nats.
That number is dominated by the soft-guard wall term
-gate * (100 + 2 N softplus(-log mu)) (likelihood/selection.py:313-339)
responding to Neff/log_mu differences, NOT by missing-host placement, and
both arms rail at the H0=139 grid edge — the vacuousness K2/K8 warn about.

This script measures the same pair on a coarse grid with FOUR arms:
    sel / selq_radial  x  soft_guard {on, off}
Guard-off logL is the clean likelihood wherever the hard Vitale/variance
floor passes (else -inf).  Reported:
  * osc over commonly-finite guard-off nodes  (the honest item-3 number)
  * osc over guard-on nodes                   (reproduces/validates the artifact)
  * per-node reproduction of the 2026-08-10 guard-on values under current
    master (code-drift check after the 182-file pull)

Run from experiments/desi_full259 (its common.py pins DARKSIRENS_ZMAX=6.0):
    python ../field_level_plan/pr0/run_osc_item3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PR0_DIR = Path(__file__).resolve().parent
FULL259 = PR0_DIR.parent.parent / "desi_full259"
sys.path.insert(0, str(FULL259))

import common as C  # noqa: E402  (ZMAX pin; must precede darksirens imports)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402


def _opts(q=None, soft_guard=True, max_var=None):
    return SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks",
        universe_model="dark_sirens",
        survey_path=str(C.SURVEY_N64), gw_path=str(C.GW_259),
        gwselection_path=str(C.INJ_PLAIN), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=(str(q) if q else None),
        lss_marginalize=False,
        c_mode="selection", catalog_sky_weighting="field",
        # S-3: no mask on a footprint-limited catalog -- the exposed
        # configuration, run here deliberately (loaders' guard).
        allow_unmasked_footprint=True,
        complete_empty_pixel_policy="zero",
        mark_model="none", mark_names=(), sky_model="isotropic",
        drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True,
        fix_survey=True, shared_beta=True, shared_spin=True,
        shared_gamma=True, sel_batch_size=16384, pe_event_block=8,
        selection_neff_soft_guard=soft_guard,
        **({"max_likelihood_variance": max_var} if max_var else {}),
    )


def main():
    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    sel = load_selection_fit_json(C.FIT_JSON)
    theta = {"m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
             "sigma_M": float(sel["sigma_M"])}
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    fixed = {"Om0": C.OM0, "sigma_kde": 0.003,
             "log10n0": float(cal["log10n0"]),
             "delta": float(cal["delta"]), **theta}

    h0 = np.arange(25.0, 140.0 + 0.5, 5.0)
    q_radial = C.DATA_DIR / "fits" / "q_radial.h5"

    ARMS = {
        "sel_soft": dict(q=None, soft=True),
        "selq_soft": dict(q=q_radial, soft=True),
        "sel_hard": dict(q=None, soft=False),
        "selq_hard": dict(q=q_radial, soft=False),
        # Clean-likelihood arms: the GWTC-4/5 variance cap lifted (the hard
        # guard fails EVERYWHERE on this line: pe_variance_sum = 0.2733
        # inflates the required selection Neff to ~92k), keeping only the
        # Vitale 5*N_obs floor.  The Delta of these two arms is the honest
        # placement oscillation, carrying the caveat that sigma(lnL) > 1 nat
        # of MC noise by the same criterion that was lifted.
        "sel_nogv": dict(q=None, soft=False, max_var=1e6),
        "selq_nogv": dict(q=q_radial, soft=False, max_var=1e6),
    }

    outfile = PR0_DIR / "osc_item3_results.json"
    if outfile.exists():
        out = json.load(open(outfile))
    else:
        out = {"h0": h0.tolist(), "fixed": fixed,
               "git_sha": C.git_sha(), "arms": {}}
    import darksirens.likelihood.core as core
    from darksirens.likelihood.selection import selection_log_correction
    captured = {}

    def wrapped(log_mu, Neff, nEvents, **kw):
        def rec(lm, ne):
            captured["log_mu"] = float(lm)
            captured["Neff"] = float(ne)
        jax.debug.callback(rec, log_mu, Neff)
        return selection_log_correction(log_mu, Neff, nEvents, **kw)

    core.selection_log_correction = wrapped

    for name, cfg in ARMS.items():
        if name in out["arms"]:
            continue
        opts = _opts(cfg["q"], cfg["soft"], cfg.get("max_var"))
        opts.selection_fit = str(C.FIT_JSON)
        opts.selection_kcorr_by_catalog = [
            tuple(sel["k_corr_coeffs"]) or None]
        print(f"=== {name}: Q={opts.lss_completion} "
              f"soft_guard={cfg['soft']}", flush=True)
        data = load_all_data(opts)
        opts.resolved_survey_z_depths = (data.get("z_depth"),)
        pop_fid = get_fixed_population_params(opts.pop_model)
        logl = make_likelihood(opts, data, pop_fid,
                               fixed_parameter_values=fixed)
        vals = np.empty(h0.size)
        neffs = np.empty(h0.size)
        logmus = np.empty(h0.size)
        for i, h in enumerate(h0):
            vals[i] = float(logl(jnp.asarray([h])))
            neffs[i] = captured.get("Neff", np.nan)
            logmus[i] = captured.get("log_mu", np.nan)
            print(f"  H0={h:.0f}: logL={vals[i]:.3f} "
                  f"Neff={neffs[i]:.1f}", flush=True)
        out["arms"][name] = vals.tolist()
        out.setdefault("neff", {})[name] = neffs.tolist()
        out.setdefault("log_mu", {})[name] = logmus.tolist()
        with open(outfile, "w") as f:
            json.dump(out, f, indent=1)

    a = {k: np.array(v) for k, v in out["arms"].items()}
    for tag in ("soft", "hard"):
        d = a[f"selq_{tag}"] - a[f"sel_{tag}"]
        m = np.isfinite(d)
        if m.any():
            print(f"[{tag}] finite nodes {m.sum()}/{len(d)}  "
                  f"osc = {d[m].max() - d[m].min():.4f} nat  "
                  f"(min {d[m].min():.4f} @H0={h0[m][d[m].argmin()]:.0f}, "
                  f"max {d[m].max():.4f} @H0={h0[m][d[m].argmax()]:.0f})",
                  flush=True)
        else:
            print(f"[{tag}] no commonly finite nodes", flush=True)
    print(f"[pr0-item3] wrote {outfile}")


if __name__ == "__main__":
    main()
