"""The volume-limited H0 scan: selected events, complete catalog, matched beta.

``universe_model = dark_sirens_complete`` is the whole point of the volume
limit -- inside the selected volume LOA+LS is treated as complete, so there is
no missing-galaxy branch, no ``C(z)``, no luminosity function and no ``Q``.
The redshift prior is the catalog itself.  Every completeness knob this repo
normally carries is therefore ABSENT rather than set to a value, which is why
this line has its own runner instead of a flag on the 259-event one.

The selection term uses ``data/injections_contained.h5``: the same detected
injection set with ``pdraw`` divided by the containment probability, so the
integral carries detection AND containment -- the two-stage criterion that
selected the events.

Also runs the events-only arm (uniform-in-comoving-volume prior, no catalog) as
the control that says how much of any structure is the catalog's.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import common as C  # noqa: F401

import h5py  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def subset_gw(src, dst, keep_idx):
    """Write a gwcat-1.0 file holding only ``keep_idx`` events."""
    with h5py.File(src) as f, h5py.File(dst, "w") as g:
        n, ns = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
        for k, v in f.attrs.items():
            g.attrs[k] = v
        g.attrs["nobs"] = len(keep_idx)
        for k in f:
            if not isinstance(f[k], h5py.Dataset):
                continue
            g.create_dataset(k, data=f[k][...].reshape(n, ns)[keep_idx].reshape(-1))
        for k in ("event_names", "cosmology_H0_per_event",
                  "cosmology_Om0_per_event", "sample_set_name_per_event",
                  "sample_set_approximant_per_event",
                  "sample_set_selection_reason"):
            if k in f.attrs:
                g.attrs[k] = np.asarray(f.attrs[k])[keep_idx]
    return dst


def _opts(gw, inj, universe):
    # The field convention is a dark-siren construct (it weights each occupied
    # pixel by its share of the observed count), so the spectral control -- which
    # has no catalog at all -- must use the conditional convention.  Forcing
    # 'field' there is refused by the likelihood, correctly.
    weighting = ("conditional" if universe == "spectral_sirens" else "field")
    return SimpleNamespace(
        pop_model="gwtc5_fiducial_bpl2peaks", universe_model=universe,
        survey_path=str(C.SURVEY_N64), gw_path=str(gw),
        gwselection_path=str(inj), pdet_flow_path=None, gw_flows_path=None,
        n_catalogs=1, use_LSS=False, lss_completion=None, lss_marginalize=False,
        c_mode=None, catalog_sky_weighting=weighting,
        complete_empty_pixel_policy="zero",
        mark_model="none", mark_names=(), sky_model="isotropic",
        drop_full_catalog=False, survey_z_depth=None,
        fix_cosmology=False, fix_de=True, fix_population=True, fix_survey=True,
        shared_beta=True, shared_spin=True, shared_gamma=True,
        sel_batch_size=16384, pe_event_block=8,
        selection_neff_soft_guard=True, max_likelihood_variance=1e6,
        per_pixel_completeness=None, allow_unmasked_footprint=True,
        lss_field_mode="table", lss_field_artifact=None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selected", default="data/selected_events.json")
    p.add_argument("--injections", default="data/injections_contained.h5")
    p.add_argument("--h0-min", type=float, default=20.0)
    p.add_argument("--h0-max", type=float, default=140.0)
    p.add_argument("--h0-step", type=float, default=2.0)
    p.add_argument("--arms", nargs="*",
                   default=["complete", "spectral"])
    p.add_argument("--outdir", default="data/h0")
    a = p.parse_args(argv)

    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood

    sel = json.load(open(a.selected))
    idx = sorted(sel["events"][k]["index"] for k in sel["selected"])
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    gw = subset_gw(C.GW_259, out / "gw_selected.h5", idx)
    print(f"[h0] {len(idx)} selected event(s): {sel['selected']}")

    h0 = np.arange(a.h0_min, a.h0_max + 0.5 * a.h0_step, a.h0_step)
    ARMS = {"complete": "dark_sirens_complete", "spectral": "spectral_sirens"}
    results = {}
    for name in a.arms:
        opts = _opts(gw, a.injections, ARMS[name])
        print(f"\n=== {name}: universe_model={opts.universe_model}", flush=True)
        data = load_all_data(opts)
        opts.resolved_survey_z_depths = (data.get("z_depth"),)
        logl = make_likelihood(
            opts, data, get_fixed_population_params(opts.pop_model),
            fixed_parameter_values={"Om0": C.OM0, "sigma_kde": 0.003})
        vals = np.empty(h0.size)
        t0 = time.time()
        for i, x in enumerate(h0):
            vals[i] = float(logl(jnp.asarray([float(x)])))
        wall = time.time() - t0
        finite = np.isfinite(vals)
        r = dict(h0=h0.tolist(), logl=vals.tolist(),
                 n_finite=int(finite.sum()), wall_s=wall,
                 universe_model=opts.universe_model, n_events=len(idx))
        if finite.any():
            v = np.where(finite, vals, -np.inf)
            pdf = np.exp(v - v.max())
            pdf = np.where(finite, pdf, 0.0)
            trapz = getattr(np, "trapezoid", None) or np.trapz
            pdf /= trapz(pdf, h0)
            cdf = np.concatenate([[0], np.cumsum(
                0.5 * (pdf[1:] + pdf[:-1]) * np.diff(h0))])
            cdf /= cdf[-1]
            q = lambda t: float(np.interp(t, cdf, h0))  # noqa: E731
            r.update(pdf=pdf.tolist(), median=q(0.5),
                     ci68=[q(0.16), q(0.84)], ci90=[q(0.05), q(0.95)],
                     map=float(h0[np.argmax(pdf)]))
            # How much of the prior does the likelihood actually shape?
            flat = np.ones_like(pdf) / (h0[-1] - h0[0])
            kl = float(trapz(np.where(pdf > 0, pdf * np.log(pdf / flat), 0.0),
                             h0))
            r["kl_from_flat_nats"] = kl
            print(f"    {name}: H0 = {r['median']:.2f} "
                  f"[{r['ci90'][0]:.1f}, {r['ci90'][1]:.1f}] (90%)  "
                  f"{r['n_finite']}/{h0.size} finite  "
                  f"KL from flat = {kl:.4f} nat", flush=True)
        else:
            print(f"    {name}: ALL NODES -inf", flush=True)
        results[name] = r
        json.dump({"results": results, "selected": sel["selected"],
                   "n_selected": len(idx)},
                  open(out / "h0_vollim.json", "w"))
    print(f"[h0] wrote {out / 'h0_vollim.json'}")
    print("VOLLIM_H0_DONE", flush=True)


if __name__ == "__main__":
    main()
