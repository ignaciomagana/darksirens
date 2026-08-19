"""PR-0 item 1: production-likelihood cost baseline, three columns.

Measured on the production 259-event configuration (GPU, value-only path):
  (i)   no-LSS production baseline            (Q table absent)
  (ii)  table mode, deterministic Q           (q_radial.h5, lss_marginalize off)
  (iii) table mode, member marginalization    (q_radial.h5 8-member ensemble,
                                               lss_marginalize on -> M_draw = 8)
M_draw in {32, 64} have no production-scale artifact; they are projected from
the measured M-scaling of scripts/profile_member_marginalization.py and
reported alongside.  The prospective latent column is (iii) plus the
member-independent row-factor arithmetic of PLAN 2.3 and is reported there as
a projection, not a measurement.

Run from experiments/desi_full259 on a GPU node:
    python ../field_level_plan/pr0/run_cost_baseline.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PR0_DIR = Path(__file__).resolve().parent
FULL259 = PR0_DIR.parent.parent / "desi_full259"
sys.path.insert(0, str(FULL259))

import common as C  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402

ANCHOR_H0 = 67.74
N_WARM = 3
N_TIME = 20


def _opts(q=None, marginalize=False):
    return SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks", universe_model="dark_sirens",
        survey_path=str(C.SURVEY_N64), gw_path=str(C.GW_259),
        gwselection_path=str(C.INJ_PLAIN), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1, use_LSS=False,
        lss_completion=(str(q) if q else None),
        lss_marginalize=marginalize,
        c_mode="selection", catalog_sky_weighting="field",
        # S-3: no mask on a footprint-limited catalog -- the exposed
        # configuration, run here deliberately (loaders' guard).
        allow_unmasked_footprint=True,
        complete_empty_pixel_policy="zero", mark_model="none", mark_names=(),
        sky_model="isotropic", drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True,
        fix_survey=True, shared_beta=True, shared_spin=True,
        shared_gamma=True, sel_batch_size=16384, pe_event_block=8,
        selection_neff_soft_guard=True,
    )


def main():
    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    sel = load_selection_fit_json(C.FIT_JSON)
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    fixed = {"Om0": C.OM0, "sigma_kde": 0.003,
             "log10n0": float(cal["log10n0"]), "delta": float(cal["delta"]),
             "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
             "sigma_M": float(sel["sigma_M"])}

    q_radial = C.DATA_DIR / "fits" / "q_radial.h5"
    ARMS = {
        "no_lss": dict(q=None, marg=False),
        "table_M1": dict(q=q_radial, marg=False),
        "table_marg_M8": dict(q=q_radial, marg=True),
    }
    h0s = np.linspace(30.0, 130.0, N_TIME)

    out = {"device": str(jax.devices()[0]), "git_sha": C.git_sha(),
           "n_time": N_TIME, "arms": {}}
    for name, cfg in ARMS.items():
        opts = _opts(cfg["q"], cfg["marg"])
        opts.selection_fit = str(C.FIT_JSON)
        opts.selection_kcorr_by_catalog = [tuple(sel["k_corr_coeffs"]) or None]
        print(f"=== {name}", flush=True)
        data = load_all_data(opts)
        opts.resolved_survey_z_depths = (data.get("z_depth"),)
        pop_fid = get_fixed_population_params(opts.pop_model)
        logl = make_likelihood(opts, data, pop_fid,
                               fixed_parameter_values=fixed)
        t0 = time.perf_counter()
        v = float(logl(jnp.asarray([ANCHOR_H0])))
        compile_s = time.perf_counter() - t0
        for _ in range(N_WARM):
            float(logl(jnp.asarray([ANCHOR_H0])))
        ts = []
        for h in h0s:
            t0 = time.perf_counter()
            float(logl(jnp.asarray([h])))
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts)
        out["arms"][name] = dict(
            compile_s=compile_s, logL_anchor=v,
            median_ms=float(np.median(ts) * 1e3),
            p10_ms=float(np.percentile(ts, 10) * 1e3),
            p90_ms=float(np.percentile(ts, 90) * 1e3))
        print(f"  compile {compile_s:.1f}s  warm median "
              f"{np.median(ts)*1e3:.2f} ms  "
              f"[{np.percentile(ts,10)*1e3:.2f}, "
              f"{np.percentile(ts,90)*1e3:.2f}]", flush=True)
        with open(PR0_DIR / "cost_baseline.json", "w") as f:
            json.dump(out, f, indent=1)

    b = out["arms"]["no_lss"]["median_ms"]
    for k, v in out["arms"].items():
        print(f"{k:16s} {v['median_ms']:8.2f} ms   "
              f"{100 * (v['median_ms'] / b - 1):+6.1f}% vs no-LSS")
    print(f"[pr0-cost] wrote {PR0_DIR / 'cost_baseline.json'}")


if __name__ == "__main__":
    main()
