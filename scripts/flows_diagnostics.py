#!/usr/bin/env python
"""Flow-surrogate diagnostics: fidelity, likelihood scans, per-event ESS.

Compares the flow-surrogate likelihood (population draws scored by per-event
flows) against the stored-PE-sample likelihood on the SAME events, selection
set, and population model, driving the estimator kernels directly (no CLI).

Outputs (under --outdir):
    fidelity/<EVENT>.png          flow samples vs stored PE samples (marginals)
    fidelity_summary.json         per-event mean flow log-density of PE samples
    likelihood_scans/scan_<p>.png 1-D total-lnL scans, flow vs PE path
    likelihood_scans/scans.json   scan values
    event_terms.json              per-event lnZ_i (both paths) + flow ESS at fiducial
    event_terms.png               scatter of per-event lnZ_i + ESS histogram

Usage:
    python scripts/flows_diagnostics.py \
        --flows flows/m1m2dLchi_eff \
        --pe_store flows/pe_store_flowsubset.h5 \
        --gwselection_path selection_o3o4ab_allsky.h5 \
        --outdir flows [--flows_nsamp 65536] [--fidelity_events 6]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.core.constants import H0_FID, OM0_FID
from darksirens.gw.flows import (
    compute_support_boxes,
    ensemble_sample,
    load_flow_ensemble,
    make_ensemble_log_prob,
    make_ensemble_log_prob_per_event,
)
from darksirens.gw.utils import load_gw_samples, load_selection_samples
from darksirens.gw.populations.registry import get_fixed_population_params, get_model
from darksirens.gw.populations.sampling import (
    make_mass_q_edges,
    resolve_mass_grid_bounds,
)
from darksirens.likelihood.core import darksiren_log_likelihood
from darksirens.likelihood.flow_events import (
    PePriorTables,
    build_chi_eff_prior_table,
    build_flow_loglike,
    build_pe_dl_prior_table,
)
from darksirens.likelihood.selection import log_evidence_and_mc_variance
from darksirens.inference.utils import log_sample_weight
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)
from darksirens.utils.cosmology import dL_grid_bounds


LABELS = [r"$m_{1,\rm det}$", r"$m_{2,\rm det}$", r"$d_L$ [Mpc]", r"$\chi_{\rm eff}$"]

# powerlaw+peak parameter vector indices (v1, alpha, m_min, m_max, dm_min,
# dm_max, mu_G, sigma_G, beta, mu_chi, sigma_chi, gamma).
SCANS = {
    "alpha": (1, np.linspace(1.0, 4.0, 21)),
    "mu_G": (6, np.linspace(25.0, 45.0, 21)),
    "gamma": (11, np.linspace(-4.0, 6.0, 21)),
}


def _toy_catalog():
    return EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 1)),
        dzgals=jnp.ones((1, 1)),
        wgals=jnp.ones((1, 1)),
        ngals=jnp.ones((1,), dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


def _gw_event_from(m1det, m2det, dL, chieff, prior_wt):
    n = m1det.shape[0]
    return GWEvent(
        m1det=jnp.asarray(m1det),
        m2det=jnp.asarray(m2det),
        dL=jnp.asarray(dL),
        chieff=jnp.asarray(chieff),
        prior_wt=jnp.asarray(prior_wt),
        pixels=jnp.zeros(n, dtype=jnp.int32),
        q=jnp.asarray(m2det / m1det),
        valid=jnp.ones(n, dtype=bool),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--pe_store", required=True)
    ap.add_argument("--gwselection_path", required=True)
    ap.add_argument("--outdir", default="flows")
    ap.add_argument("--flows_nsamp", type=int, default=65536)
    ap.add_argument("--flows_seed", type=int, default=42)
    ap.add_argument("--fidelity_events", type=int, default=6)
    ap.add_argument("--pe_cosmology", default="67.74,0.3089")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "fidelity").mkdir(parents=True, exist_ok=True)
    (outdir / "likelihood_scans").mkdir(parents=True, exist_ok=True)

    cosmo = CosmoParams(H0=H0_FID, Om0=OM0_FID)
    survey = SurveyParams(n0=1e-3, z50=1.0, w=0.5, delta=0.0, b_miss=1.0, alpha_miss=0.5)
    catalog = _toy_catalog()
    theta_fid = np.asarray(get_fixed_population_params("powerlaw+peak"))
    model = get_model("powerlaw+peak")

    # ── inputs ──────────────────────────────────────────────────────────
    ens = load_flow_ensemble(args.flows)
    print(ens.summary().splitlines()[0])
    (m1det, m2det, dL, chieff, _ra, _dec, p_pe, nEvents, nsamp) = load_gw_samples(
        args.pe_store
    )
    assert nEvents == ens.n_flows, (nEvents, ens.n_flows)
    gw_pe = _gw_event_from(m1det, m2det, dL, chieff, p_pe)

    (m1s, m2s, dLs, chis, _ras, _decs, p_draw, Ndraw) = load_selection_samples(
        args.gwselection_path
    )
    gw_sel = _gw_event_from(m1s, m2s, dLs, chis, p_draw)
    print(f"selection: {m1s.shape[0]:,} detected injections, Ndraw={Ndraw:.3g}")

    pe_H0, pe_Om0 = (float(x) for x in args.pe_cosmology.split(","))
    log_dl, log_p_dl = build_pe_dl_prior_table(pe_H0, pe_Om0)
    qg, cg, tab = build_chi_eff_prior_table(0.99)
    pe_tables = PePriorTables(
        log_dl_grid=jnp.asarray(log_dl),
        log_p_dl=jnp.asarray(log_p_dl),
        chi_q_grid=jnp.asarray(qg),
        chi_grid=jnp.asarray(cg),
        chi_table=jnp.asarray(tab),
    )
    m1_lo, m1_hi = resolve_mass_grid_bounds(model)
    m1_edges, q_edges = make_mass_q_edges(m1_lo, m1_hi)
    u_base = jax.random.uniform(
        jax.random.PRNGKey(args.flows_seed),
        (nEvents, args.flows_nsamp, 4),
        dtype=jnp.float64,
    )
    boxes = compute_support_boxes(
        ens, key=jax.random.key(args.flows_seed + 1)
    )

    ll_flow = build_flow_loglike(
        model=model,
        eval_logflows=make_ensemble_log_prob_per_event(ens),
        group_params=ens.group_params(),
        u_base=u_base,
        m1_edges=m1_edges,
        q_edges=q_edges,
        pe_tables=pe_tables,
        support_boxes=boxes,
        gw_sel=gw_sel,
        em_catalog_sel=catalog,
        Ndraw=Ndraw,
        nEvents=nEvents,
        max_likelihood_variance=1e9,  # diagnostics: never clamp
    )

    def ll_pe(theta):
        return darksiren_log_likelihood(
            cosmo, survey, jnp.asarray(theta), gw_pe, catalog, gw_sel, catalog,
            nEvents, nsamp, Ndraw, "powerlaw+peak", "spectral_sirens",
            max_likelihood_variance=1e9,
        )

    # ── fidelity: flow samples vs stored PE samples ─────────────────────
    print("fidelity ...")
    n_fid = 4096
    fsamples = np.asarray(ensemble_sample(ens, jax.random.key(1), n_fid))
    pe_blocks = {
        "m1det": np.asarray(m1det).reshape(nEvents, nsamp),
        "m2det": np.asarray(m2det).reshape(nEvents, nsamp),
        "dL": np.asarray(dL).reshape(nEvents, nsamp),
        "chieff": np.asarray(chieff).reshape(nEvents, nsamp),
    }
    X_pe = jnp.asarray(
        np.stack([pe_blocks[k] for k in ("m1det", "m2det", "dL", "chieff")], axis=-1)
    )  # (nEvents, nsamp, 4)
    eval_lf = jax.jit(make_ensemble_log_prob(ens))
    # Mean log flow density of each event's OWN PE samples (higher = better fit).
    mean_lnp = []
    for i in range(nEvents):
        lp = eval_lf(ens.group_params(), X_pe[i])[i]
        mean_lnp.append(float(jnp.mean(lp)))
    fidelity = {
        name: {"mean_flow_logpdf_of_pe_samples": v}
        for name, v in zip(ens.names, mean_lnp)
    }
    with open(outdir / "fidelity_summary.json", "w") as f:
        json.dump(fidelity, f, indent=1)

    # Corner-style overlays for a representative subset (worst + spread).
    order = np.argsort(mean_lnp)
    picks = sorted(set(
        list(order[: max(1, args.fidelity_events // 2)])
        + list(order[:: max(1, nEvents // max(1, args.fidelity_events // 2))])
    ))[: args.fidelity_events]
    for i in picks:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
        for d in range(4):
            pe_d = pe_blocks[list(pe_blocks)[d]][i]
            lo, hi = np.percentile(pe_d, [0.2, 99.8])
            span = hi - lo
            bins = np.linspace(lo - 0.15 * span, hi + 0.15 * span, 60)
            axes[d].hist(pe_d, bins=bins, density=True, alpha=0.45,
                         label="PE samples", color="C0")
            axes[d].hist(fsamples[i, :, d], bins=bins, density=True,
                         histtype="step", lw=1.6, label="flow", color="C3")
            axes[d].set_xlabel(LABELS[d])
            axes[d].set_yticks([])
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(f"{ens.names[i]}  (mean flow logpdf of PE samples: "
                     f"{mean_lnp[i]:.2f})")
        fig.tight_layout()
        fig.savefig(outdir / "fidelity" / f"{ens.names[i]}.png", dpi=140)
        plt.close(fig)

    # ── per-event terms + flow ESS at the fiducial hyperparameters ──────
    print("per-event terms ...")
    H0, Om0, w0, wa = cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa
    state = prepare_redshift_prior_state("spectral_sirens", cosmo, survey, catalog)

    def _log_prior_z(z, pix, cat):
        return eval_redshift_prior_with_state(
            "spectral_sirens", state, z, pix, cosmo, survey, cat
        )

    theta = jnp.asarray(theta_fid)
    # PE-path per-event lnZ_i.
    dL_lo, dL_hi = dL_grid_bounds(H0, Om0, w0, wa)
    supported = (gw_pe.dL >= dL_lo) & (gw_pe.dL <= dL_hi)
    ldw_pe = log_sample_weight(
        gw_pe.m1det, gw_pe.q, jnp.clip(gw_pe.dL, dL_lo, dL_hi), gw_pe.chieff,
        gw_pe.pixels, gw_pe.prior_wt, cosmo, survey, theta, catalog,
        model.log_p_pop, _log_prior_z,
    )
    ldw_pe = jnp.where(supported & jnp.isfinite(ldw_pe), ldw_pe, -jnp.inf)
    lnZ_pe, var_pe = jax.vmap(
        lambda row: log_evidence_and_mc_variance(row, nsamp)
    )(ldw_pe.reshape(nEvents, nsamp))

    # Flow-path per-event lnZ_i + ESS: the SAME jitted code path the
    # likelihood uses (event-windowed population proposal).
    lnZ_fl, var_fl, ess_j = ll_flow.event_diagnostics(cosmo, survey, theta)
    ess = np.asarray(ess_j)

    event_terms = {
        name: {
            "lnZ_pe": float(a),
            "lnZ_flow": float(b),
            "var_pe": float(vp),
            "var_flow": float(vf),
            "flow_ess": float(e),
        }
        for name, a, b, vp, vf, e in zip(
            ens.names, lnZ_pe, lnZ_fl, var_pe, var_fl, ess
        )
    }
    with open(outdir / "event_terms.json", "w") as f:
        json.dump(
            {
                "fiducial_theta": list(map(float, theta_fid)),
                "flows_nsamp": args.flows_nsamp,
                "total_var_flow": float(jnp.sum(var_fl)),
                "total_var_pe": float(jnp.sum(var_pe)),
                "events": event_terms,
            },
            f,
            indent=1,
        )

    d = np.asarray(lnZ_fl) - np.asarray(lnZ_pe)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(np.asarray(lnZ_pe), np.asarray(lnZ_fl), s=14)
    lo = min(float(lnZ_pe.min()), float(lnZ_fl.min()))
    hi = max(float(lnZ_pe.max()), float(lnZ_fl.max()))
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set_xlabel(r"$\ln \hat Z_i$ (stored PE)")
    axes[0].set_ylabel(r"$\ln \hat Z_i$ (flow)")
    axes[1].hist(d - np.median(d), bins=30)
    axes[1].set_xlabel(r"$\ln\hat Z_i^{\rm flow} - \ln\hat Z_i^{\rm PE}$ - median")
    axes[2].hist(np.log10(ess), bins=30)
    axes[2].set_xlabel(r"$\log_{10}$ flow ESS per event")
    fig.suptitle(
        f"per-event terms at fiducial (J={args.flows_nsamp:,}); "
        f"median offset {np.median(d):+.2f}, scatter (MAD) "
        f"{1.4826*np.median(np.abs(d - np.median(d))):.3f}"
    )
    fig.tight_layout()
    fig.savefig(outdir / "event_terms.png", dpi=140)
    plt.close(fig)
    print(
        f"  per-event lnZ offset: median {np.median(d):+.3f}, "
        f"MAD-scatter {1.4826*np.median(np.abs(d - np.median(d))):.3f}; "
        f"flow ESS: min {ess.min():.0f}, median {np.median(ess):.0f}"
    )

    # ── 1-D total-lnL scans ──────────────────────────────────────────────
    print("likelihood scans ...")
    scans = {}
    t0 = time.perf_counter()
    for pname, (idx, grid) in SCANS.items():
        vals_f, vals_p = [], []
        for v in grid:
            th = theta_fid.copy()
            th[idx] = v
            vals_f.append(float(ll_flow(cosmo, survey, jnp.asarray(th))))
            vals_p.append(float(ll_pe(th)))
        scans[pname] = {
            "grid": grid.tolist(),
            "lnL_flow": vals_f,
            "lnL_pe": vals_p,
        }
        vf = np.asarray(vals_f)
        vp = np.asarray(vals_p)
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax1.plot(grid, vf - vf.max(), "-o", ms=3, label="flow path")
        ax1.plot(grid, vp - vp.max(), "-s", ms=3, label="stored-PE path")
        ax1.set_ylabel(r"$\ln L - \max$")
        ax1.legend(frameon=False)
        resid = (vf - vf.max()) - (vp - vp.max())
        ax2.plot(grid, resid, "-", color="C3")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_ylabel("difference")
        ax2.set_xlabel(pname)
        fig.tight_layout()
        fig.savefig(outdir / "likelihood_scans" / f"scan_{pname}.png", dpi=140)
        plt.close(fig)
    dt = time.perf_counter() - t0
    with open(outdir / "likelihood_scans" / "scans.json", "w") as f:
        json.dump(scans, f, indent=1)
    n_eval = sum(len(g) for _, (i, g) in SCANS.items())
    print(f"  {n_eval} x 2 likelihood evaluations in {dt:.1f}s "
          f"({dt/(2*n_eval)*1e3:.0f} ms/eval average)")


if __name__ == "__main__":
    main()
