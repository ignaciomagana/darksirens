#!/usr/bin/env python3
"""Dissect the H0 slope: per-event log-marginals and selection pieces."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp

from scan_h0 import _base_opts  # reuse opts assembly (also applies KDE patch)
from darksirens.utils.cosmology import Om0Planck, w0Fiducial, waFiducial
from darksirens.inference.data import load_all_data
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.likelihood.factory import make_likelihood
import darksirens.likelihood.core as core

data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "canonical")
survey = sys.argv[2] if len(sys.argv) > 2 else str(data_dir / "catalog_pixelated_nside_16.h5")
h0_pair = (60.0, 85.0)

opts = _base_opts(survey, data_dir / "gw_events.h5", data_dir / "gw_selection.h5")
fixed = {"Om0": Om0Planck, "w0": w0Fiducial, "wa": waFiducial,
         "log10n0": -5.0128, "delta": 0.0, "sigma_kde": 0.0}
data = load_all_data(opts)
opts.resolved_survey_z_depths = (data.get("z_depth"),)
pop_fid = get_fixed_population_params(opts.pop_model)

records = {"ev": [], "sel": [], "corr": []}
orig_ev = core.log_evidence_and_mc_variance
orig_sel = core.compute_selection_term
orig_corr = core.selection_log_correction


def ev_rec(ldw, nsamp):
    out = orig_ev(ldw, nsamp)
    jax.debug.callback(
        lambda ll, best: records["ev"].append(
            (np.atleast_1d(np.asarray(ll)), np.atleast_1d(np.asarray(best)))),
        out[0], jnp.max(ldw, axis=-1))
    return out


def sel_rec(*a, **k):
    out = orig_sel(*a, **k)
    records["sel"].append(tuple(float(np.asarray(x)) for x in out[:2]))
    return out


def corr_rec(log_mu, Neff, nEvents, **k):
    out = orig_corr(log_mu, Neff, nEvents, **k)
    records["corr"].append((float(np.asarray(out)),
                            float(np.asarray(k.get("pe_variance_sum", 0.0)))))
    return out


core.log_evidence_and_mc_variance = ev_rec
core.compute_selection_term = sel_rec
core.selection_log_correction = corr_rec

with jax.disable_jit():
    logl = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    per_h0 = {}
    for h in h0_pair:
        records["ev"], records["sel"], records["corr"] = [], [], []
        total = float(logl(jnp.asarray([h])))
        per_h0[h] = {"total": total,
                     "ev": np.concatenate([r[0] for r in records["ev"]]),
                     "best": np.concatenate([r[1] for r in records["ev"]]),
                     "sel": records["sel"][:], "corr": records["corr"][:]}

a, b = (per_h0[h] for h in h0_pair)
print(f"total: {h0_pair[0]}: {a['total']:.1f}   {h0_pair[1]}: {b['total']:.1f}   "
      f"delta={b['total']-a['total']:.1f}")
print(f"sel (log_mu, Neff): {a['sel']} -> {b['sel']}")
print(f"sel corr: {a['corr']} -> {b['corr']}")
d = b["ev"] - a["ev"]
print(f"event lls: n={d.size}  sum(delta)={d.sum():.1f}  "
      f"median={np.median(d):.2f}  min/max={d.min():.2f}/{d.max():.2f}")
print(f"event ll magnitudes at {h0_pair[0]}: median={np.median(a['ev']):.1f} "
      f"min={a['ev'].min():.1f} max={a['ev'].max():.1f}")
print(f"best-sample ldw at {h0_pair[0]}: median={np.median(a['best']):.1f}")
for i in np.argsort(a["ev"])[:5]:
    print(f"  worst ev {i}: ll@{h0_pair[0]}={a['ev'][i]:.1f} "
          f"ll@{h0_pair[1]}={b['ev'][i]:.1f} best_ldw={a['best'][i]:.1f}")
