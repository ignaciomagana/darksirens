"""PR-0 item 2a: sum_i phi_i — event-weighted in-support missing-branch mass.

phi_i is event i's missing-branch fraction of its total prior mass, restricted
to the in-support region (occupied footprint pixels, z <= z_depth = 0.30).
Measured through the production likelihood itself (no re-implementation of the
prior conventions): evaluate at the anchor (H0 = 67.74, theta_hat, calibrated
budget, hard guard) with

  arm A  "sel"     : no Q table              (homogeneous completion)
  arm B  "killed"  : synthetic Q table with logQ = -60 on occupied-footprint
                     pixels at zgrid <= 0.30, logQ = 0 elsewhere -> the
                     in-support missing branch is deleted from BOTH the event
                     terms and the selection integral
  arm C  "qradial" : the shipped radial table (reference)

The selection term is separated by capturing (log_mu, Neff) with a
jax.debug.callback wrapper around selection_log_correction, so

  sum_i -log(1 - phi_i) = [logL_A - logL_B]
                          + N_obs (log mu_A - log mu_B)
                          + [farr_B - farr_A]        (Farr 1/Neff correction)

which upper-bounds (and for small phi equals) sum_i phi_i.

Run from experiments/desi_full259:  python ../field_level_plan/pr0/compute_phi.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PR0_DIR = Path(__file__).resolve().parent
FULL259 = PR0_DIR.parent.parent / "desi_full259"
sys.path.insert(0, str(FULL259))

import common as C  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import h5py  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402

ANCHOR_H0 = 67.74
LOGQ_KILL = -60.0
KILLER = PR0_DIR / "q_killer_insupport.h5"


def build_killer_table():
    src = C.DATA_DIR / "fits" / "q_radial.h5"
    with h5py.File(src) as f:
        g = f["lss_completion"]
        zgrid = g["zgrid"][...]
        attrs = dict(g.attrs)
    with h5py.File(C.SURVEY_N64) as f:
        occupied = f["ngals"][...] > 0
    logq = np.zeros((occupied.size, zgrid.size))
    logq[np.ix_(occupied, zgrid <= C.Z_DEPTH)] = LOGQ_KILL
    with h5py.File(KILLER, "w") as f:
        g = f.create_group("lss_completion")
        g.create_dataset("logq_map", data=logq)
        g.create_dataset("zgrid", data=zgrid)
        for k, v in attrs.items():
            if k in ("member_content_sha256", "n_members"):
                continue
            g.attrs[k] = v
        g.attrs["created_by_note"] = (
            "PR-0 phi probe: in-support missing branch deleted; NOT a "
            "completion product")
    print(f"[killer] occupied={occupied.sum()} pixels, "
          f"z nodes <= {C.Z_DEPTH}: {(zgrid <= C.Z_DEPTH).sum()}, "
          f"wrote {KILLER}")


def _opts(q=None):
    return SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks", universe_model="dark_sirens",
        survey_path=str(C.SURVEY_N64), gw_path=str(C.GW_259),
        gwselection_path=str(C.INJ_PLAIN), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1, use_LSS=False,
        lss_completion=(str(q) if q else None), lss_marginalize=False,
        c_mode="selection", catalog_sky_weighting="field",
        # S-3: no mask on a footprint-limited catalog -- the exposed
        # configuration, run here deliberately (loaders' guard).
        allow_unmasked_footprint=True,
        complete_empty_pixel_policy="zero", mark_model="none", mark_names=(),
        sky_model="isotropic", drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True,
        fix_survey=True, shared_beta=True, shared_spin=True,
        shared_gamma=True, sel_batch_size=16384, pe_event_block=8,
        selection_neff_soft_guard=False,
        # The GWTC-4/5 variance cap fails everywhere on this line
        # (pe_variance_sum = 0.2733 -> required Neff ~ 92k vs ~31k at the
        # anchor); lift it to the Vitale floor so the event-term Delta is
        # finite.  The phi measurement is a ratio of event terms, which the
        # guard never touches.
        max_likelihood_variance=1e6,
    )


def main():
    if not KILLER.exists():
        build_killer_table()

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

    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    sel = load_selection_fit_json(C.FIT_JSON)
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    fixed = {"Om0": C.OM0, "sigma_kde": 0.003,
             "log10n0": float(cal["log10n0"]), "delta": float(cal["delta"]),
             "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
             "sigma_M": float(sel["sigma_M"])}

    ARMS = {"sel": None, "killed": KILLER,
            "qradial": C.DATA_DIR / "fits" / "q_radial.h5"}
    out = {"anchor_H0": ANCHOR_H0, "git_sha": C.git_sha(), "arms": {}}
    for name, q in ARMS.items():
        opts = _opts(q)
        opts.selection_fit = str(C.FIT_JSON)
        opts.selection_kcorr_by_catalog = [tuple(sel["k_corr_coeffs"]) or None]
        print(f"=== {name}: Q={q}", flush=True)
        data = load_all_data(opts)
        opts.resolved_survey_z_depths = (data.get("z_depth"),)
        pop_fid = get_fixed_population_params(opts.pop_model)
        logl = make_likelihood(opts, data, pop_fid,
                               fixed_parameter_values=fixed)
        val = float(logl(jnp.asarray([ANCHOR_H0])))
        out["arms"][name] = {"logL": val, **captured}
        print(f"  logL={val:.4f} log_mu={captured.get('log_mu'):.6f} "
              f"Neff={captured.get('Neff'):.1f}", flush=True)

    A, B = out["arms"]["sel"], out["arms"]["killed"]
    n = 259.0
    farr = lambda ne: n * (n + 3.0) / (2.0 * ne)
    ev = ((A["logL"] - B["logL"])
          + n * (A["log_mu"] - B["log_mu"])
          + (farr(B["Neff"]) - farr(A["Neff"])))
    out["sum_neglog1mphi"] = ev
    out["dlogmu_sel_minus_killed"] = A["log_mu"] - B["log_mu"]
    out["note"] = ("sum_i -log(1-phi_i) ~= sum_i phi_i for small phi; "
                   "K0 threshold 1e-3")
    with open(PR0_DIR / "phi_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"[pr0-phi] sum_i -log(1-phi_i) = {ev:.6e}   "
          f"(event-term channel; K0 threshold 1e-3)")
    print(f"[pr0-phi] selection-channel in-support missing fraction "
          f"dlogmu = {A['log_mu'] - B['log_mu']:.6e}")
    print(f"[pr0-phi] wrote {PR0_DIR / 'phi_results.json'}")


if __name__ == "__main__":
    main()
