"""Targeted leave-one-out jackknife of the selection-channel H0 posterior.

Drops each flagged event (the >20% off-footprint PE-mass list from
diagnose_stage_a) in turn from gwsamples_44.h5, rescans H0 under the `sel`
configuration (c_mode=selection, no Q, theta fixed at theta_hat), and
reports the posterior shift.  The betaS normalizer is event-independent
(p_pass is baked into pdraw), so dropping an event only removes its
numerator term -- the 43-event file is scanned against the same beta.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import common as C

import h5py
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402

FLAGGED = ["GW240915_105151", "GW151226_033853", "GW231224_024321",
           "GW231231_154016", "GW231113_200417"]


def _drop_event(src, dst, name):
    with h5py.File(src, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f["event_names"][...]]
        i = names.index(name)
        nobs = int(f.attrs["nobs"])
        nsamp = int(f.attrs["nsamp"])
        keep = np.ones(nobs * nsamp, dtype=bool)
        keep[i * nsamp:(i + 1) * nsamp] = False
        with h5py.File(dst, "w") as g:
            for k, v in f.attrs.items():
                g.attrs[k] = v
            g.attrs["nobs"] = nobs - 1
            for k in f.keys():
                arr = f[k][...]
                if k == "event_names":
                    g.create_dataset(k, data=np.array(
                        [n for j, n in enumerate(names) if j != i],
                        dtype=h5py.string_dtype()))
                elif arr.shape[0] == nobs * nsamp:
                    g.create_dataset(k, data=arr[keep])
                else:
                    g.create_dataset(k, data=arr)


def _scan(gw_path, survey, fit_json, cal, h0):
    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    sel = load_selection_fit_json(fit_json)
    opts = SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks", universe_model="dark_sirens",
        survey_path=str(survey), gw_path=str(gw_path),
        gwselection_path=str(C.BETA_S0495), pdet_flow_path=None,
        gw_flows_path=None, n_catalogs=1,
        use_LSS=False, lss_completion=None, lss_marginalize=False,
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
        selection_neff_soft_guard=True,
        # Per-catalog spelling (the scalar selection_fit_kcorr was RETIRED
        # by the K>=2 plumbing, #344).
        selection_kcorr_by_catalog=[tuple(sel["k_corr_coeffs"]) or None],
    )
    fixed = {"Om0": 0.3075, "sigma_kde": 0.003,
             "log10n0": float(cal["log10n0"]), "delta": float(cal["delta"]),
             "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
             "sigma_M": float(sel["sigma_M"])}
    data = load_all_data(opts)
    opts.resolved_survey_z_depths = (data.get("z_depth"),)
    pop_fid = get_fixed_population_params(opts.pop_model)
    logl = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    vals = np.array([float(logl(jnp.asarray([h]))) for h in h0])
    pdf = np.exp(vals - vals.max())
    pdf /= np.trapz(pdf, h0)
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1])
                                         * np.diff(h0))])
    cdf /= cdf[-1]
    return {"median": float(np.interp(0.5, cdf, h0)),
            "map": float(h0[np.argmax(pdf)]), "pdf": pdf.tolist()}


def main() -> None:
    survey = C.DATA_DIR / "pixelated_n64" / "catalog_pixelated_nside_64.h5"
    fit_json = C.DATA_DIR / "selection_fit_union.json"
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    h0 = np.arange(20.0, 140.5, 1.0)
    full = json.load(open(C.DATA_DIR / "h0_real" / "h0_real_scans.json"))
    ref = full["results"]["sel"]["median"]

    out = {"full_median": ref, "loo": {}}
    tmpdir = C.DATA_DIR / "loo"
    tmpdir.mkdir(exist_ok=True)
    for name in FLAGGED:
        dst = tmpdir / f"gw43_minus_{name}.h5"
        if not dst.exists():
            _drop_event(C.GW_SAMPLES_44, dst, name)
        res = _scan(dst, survey, fit_json, cal, h0)
        out["loo"][name] = {"median": res["median"], "map": res["map"],
                            "delta_median": res["median"] - ref}
        print(f"  drop {name}: median {res['median']:.2f} "
              f"(shift {res['median'] - ref:+.2f})", flush=True)
    with open(C.DATA_DIR / "loo" / "loo_results.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", C.DATA_DIR / "loo" / "loo_results.json")


if __name__ == "__main__":
    main()
