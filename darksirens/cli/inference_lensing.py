#!/usr/bin/env python3
"""
darksirens_inference_lensing.py
===============================
Full hierarchical inference CLI for the darksirens **strong-lensing branch**,
covering the joint singleton + J=2 cluster (pair) likelihood.

The stock ``darksirens_inference`` runs the WL singleton channel
(``--universe_model spectral_sirens_wl``) but has no flags for the cluster
channel. This tool adds:

  * loading of the lensed-injection set + per-pair PE,
  * the cluster master likelihood (``darksiren_log_likelihood_with_clusters``),
  * ``--cluster_mode {off,j2}`` so it subsumes the pure-singleton fit,
  * memory-safe PE down-sampling for the O(N_pe^2 N_y) pair likelihood.

It REUSES the branch's machinery wherever possible: prior construction,
``run_sampler`` (tinyns/dynesty/numpyro), result saving, and the singleton data
loaders. Only the cluster-specific assembly is added here.

Design notes
------------
* Singleton-only run: ``--cluster_mode off`` (no ``--lensed_*`` paths needed) is
  numerically the commit-2 / branch singleton likelihood.
* Joint run: ``--cluster_mode j2`` requires ``--lensed_injections_path`` and
  ``--pair_pe_path`` and a ``--partition_path`` (the candidate-pair list).
* WL: ``--wl_backend {lognormal,disabled}`` with ``--lensing_wl_a/b``.

Examples
--------
# Joint singleton + J=2, WL on, NUTS on GPU (recommended for 12-D):
darksirens_inference_lensing \
    --gw_path              mock_gw_pe.h5 \
    --gwselection_path     mock_gw_selection.h5 \
    --lensed_injections_path mock_lensed_injections.h5 \
    --pair_pe_path         mock_pair_pe.h5 \
    --partition_path       partition.json \
    --pop_model            powerlaw+peak \
    --cluster_mode         j2 \
    --wl_backend           lognormal --lensing_wl_a 4e-3 --lensing_wl_b 1.5 \
    --fix_cosmology true --fix_survey true \
    --sampler numpyro --nuts_warmup 500 --nuts_samples 2000 --nuts_chains 4 \
    --pe_max_per_pair 400 \
    --save_path ./run_joint/

# Pure singleton (no pairs), tinyns (rwalk + jax kernel):
darksirens_inference_lensing \
    --gw_path mock_gw_pe.h5 --gwselection_path mock_gw_selection.h5 \
    --pop_model powerlaw+peak --cluster_mode off \
    --wl_backend lognormal --lensing_wl_a 4e-3 --lensing_wl_b 1.5 \
    --fix_cosmology true --fix_survey true \
    --sampler tinyns --nlive 2000 --tinyns_sample rwalk --tinyns_kernel jax \
    --save_path ./run_singleton/
"""
from __future__ import annotations
# JAX memory configuration (before any JAX import)
from darksirens.core.jax_config import configure_jax_runtime

configure_jax_runtime(mem_fraction="0.90")

import os
import sys
import json
import time
import argparse
import datetime

import numpy as np
import h5py
import healpy as hp
import jax
import jax.numpy as jnp

# ── branch machinery we reuse ────────────────────────────────────────────────
from darksirens.redshift import zgrid
from darksirens.core.types import (
    CosmoParams, SurveyParams, EMCatalog, GWEvent,
)
from darksirens.utils.cosmology import H0Planck, Om0Planck
from darksirens.gw.populations.registry import get_fixed_population_params, get_model

from darksirens.inference.prior import build_parameter_space, make_prior_transform
from darksirens.inference.sampling import run_sampler
from darksirens.inference.parameters import (
    build_parameter_decoder, complete_empty_pixel_policy_code,
)
from darksirens.gw.samples import load_gw_samples, load_selection_samples

# ── lensing / cluster pieces ─────────────────────────────────────────────────
from darksirens.lensing.slmarks import make_sis_lens_params
from darksirens.lensing.wlmagnification import make_lognormal_wl_params
from darksirens.lensing.lensed_injections import load_lensed_injections
from darksirens.likelihood.pair_kde import make_pair_kde, stack_pair_kdes
from darksirens.likelihood.likelihood_with_clusters import (
    darksiren_log_likelihood_with_clusters,
    CLUSTER_MODE_J2, CLUSTER_MODE_OFF,
    WL_BACKEND_LOGNORMAL, WL_BACKEND_DISABLED,
)


# =============================================================================
# Container builders (mirror darksirens.likelihood.factory)
# =============================================================================
def _empty_em_catalog(nside=1):
    npix = hp.nside2npix(nside)
    return EMCatalog(
        apix=hp.nside2pixarea(nside),
        zgals=jnp.full((npix, 1), 0.1), dzgals=jnp.full((npix, 1), 0.02),
        wgals=jnp.ones((npix, 1)), ngals=jnp.ones(npix, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((npix, len(zgrid))),
        dN_obs_kde=None, pixel_to_cache_idx=None, unique_pixels=None,
        sample_to_unique_idx=None, counterpart_pixel=None, counterpart_pixels=None,
        counterpart_zs=None, counterpart_dzs=None, active_counterpart_index=0,
        bright_siren_sky_marginalized=False,
    )


def _gw_event(m1det, m2det, dL, chieff, prior_wt):
    m1det = jnp.asarray(m1det); m2det = jnp.asarray(m2det)
    dL = jnp.asarray(dL); chieff = jnp.asarray(chieff)
    return GWEvent(m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
                   prior_wt=jnp.asarray(prior_wt),
                   pixels=jnp.zeros_like(dL, dtype=jnp.int32),
                   q=m2det / m1det, valid=jnp.ones_like(dL, dtype=bool))


# =============================================================================
# Data loading
# =============================================================================
def _downsample(arrs_dict, n_keep, rng):
    """Down-sample a dict of per-sample arrays to n_keep (memory control for the
    O(N_pe^2) pair KDE). Returns the same dict with shorter arrays."""
    n = len(next(iter(arrs_dict.values())))
    if n_keep >= n:
        return arrs_dict
    idx = rng.choice(n, size=n_keep, replace=False)
    return {k: np.asarray(v)[idx] for k, v in arrs_dict.items()}


def load_inputs(opts):
    """Load singleton PE + selection, and (for j2) the lensed injections + pair
    PE + partition. Returns the assembled inputs for the cluster likelihood."""
    rng = np.random.default_rng(opts.seed)

    # --- singleton PE (event-major flatten) ---
    out = load_gw_samples(opts.gw_path)
    m1det, m2det, dL, chieff, ra, dec, p_pe, n_sing, nsamp = out
    m1det = np.asarray(m1det); m2det = np.asarray(m2det); dL = np.asarray(dL)
    chieff = np.asarray(chieff); p_pe = np.asarray(p_pe)

    # --- selection ---
    sel = load_selection_samples(opts.gwselection_path)
    m1s, m2s, dLs, chis, ras, decs, pdraw, Ndraw = \
        [np.asarray(x) for x in sel[:7]] + [sel[7]]
    gw_sel = _gw_event(m1s, m2s, dLs, chis, pdraw)

    cluster_mode = CLUSTER_MODE_J2 if opts.cluster_mode == "j2" else CLUSTER_MODE_OFF

    if cluster_mode == CLUSTER_MODE_OFF:
        gw_pe = _gw_event(m1det, m2det, dL, chieff, p_pe)
        # trivial KDE per singleton (never indexed when no pairs)
        kdes = []
        for i in range(n_sing):
            sl = slice(i * nsamp, (i + 1) * nsamp)
            kdes.append(make_pair_kde(m1det[sl], m2det[sl] / m1det[sl],
                                      dL[sl], chieff[sl], p_pe[sl]))
        pair_kdes = stack_pair_kdes(kdes) if kdes else None
        return dict(
            gw_pe=gw_pe, gw_sel=gw_sel, nEvents=n_sing, nsamp=nsamp,
            Ndraw=float(Ndraw),
            singleton_indices=jnp.arange(n_sing, dtype=jnp.int32),
            pair_indices=jnp.zeros((0, 2), dtype=jnp.int32),
            n_singletons=n_sing, n_pairs=0,
            pair_kdes=pair_kdes, lensed=None,
        )

    # --- j2: lensed injections + pair PE + partition ---
    if not (opts.lensed_injections_path and opts.pair_pe_path and opts.partition_path):
        raise SystemExit("--cluster_mode j2 requires --lensed_injections_path, "
                         "--pair_pe_path, and --partition_path")
    lensed = load_lensed_injections(opts.lensed_injections_path)
    partition = json.load(open(opts.partition_path))

    pairs = []
    with h5py.File(opts.pair_pe_path) as f:
        npairs = int(f.attrs["npairs"])
        for k in range(npairs):
            g = f[f"pair_{k}"]; imgs = []
            for name in ("image0", "image1"):
                gi = g[name]
                d = dict(m1det=np.array(gi["m1det"]), q=np.array(gi["q"]),
                         dL_app=np.array(gi["dL_app"]), chieff=np.array(gi["chieff"]),
                         prior_wt=np.array(gi["prior_wt"]))
                if opts.pe_max_per_pair > 0:
                    d = _downsample(d, opts.pe_max_per_pair, rng)
                imgs.append(d)
            pairs.append(imgs)
    P = len(pairs)

    # Build the full event catalog: [singletons 0..S-1, then pair images].
    # NOTE the singleton PE arrays are already nsamp-per-event; pair images use
    # (possibly) a different per-image sample count after down-sampling. The
    # branch's GWEvent stores ragged events via the flat (nEvents*nsamp) layout
    # ONLY when all events share nsamp. To keep shapes uniform we down-sample the
    # singleton PE to the SAME pe count as the pairs when they differ.
    pe_per_pair = len(pairs[0][0]["m1det"]) if P else nsamp
    if pe_per_pair != nsamp:
        # downsample singleton PE to pe_per_pair for a uniform array
        new_m1, new_m2, new_dL, new_chi, new_pw = [], [], [], [], []
        for i in range(n_sing):
            sl = slice(i * nsamp, (i + 1) * nsamp)
            d = _downsample(dict(m1=m1det[sl], m2=m2det[sl], dL=dL[sl],
                                 chi=chieff[sl], pw=p_pe[sl]), pe_per_pair, rng)
            new_m1.append(d["m1"]); new_m2.append(d["m2"]); new_dL.append(d["dL"])
            new_chi.append(d["chi"]); new_pw.append(d["pw"])
        m1det = np.concatenate(new_m1); m2det = np.concatenate(new_m2)
        dL = np.concatenate(new_dL); chieff = np.concatenate(new_chi); p_pe = np.concatenate(new_pw)
        nsamp = pe_per_pair

    m1_all = [m1det]; m2_all = [m2det]; dL_all = [dL]; chi_all = [chieff]; pw_all = [p_pe]
    for imgs in pairs:
        for img in imgs:
            m1_all.append(img["m1det"]); m2_all.append(img["q"] * img["m1det"])
            dL_all.append(img["dL_app"]); chi_all.append(img["chieff"]); pw_all.append(img["prior_wt"])
    gw_pe = _gw_event(np.concatenate(m1_all), np.concatenate(m2_all),
                      np.concatenate(dL_all), np.concatenate(chi_all),
                      np.concatenate(pw_all))

    # per-event KDEs (singletons trivial; pair images real)
    kdes = []
    for i in range(n_sing):
        sl = slice(i * nsamp, (i + 1) * nsamp)
        kdes.append(make_pair_kde(m1det[sl], m2det[sl] / m1det[sl], dL[sl], chieff[sl], p_pe[sl]))
    for imgs in pairs:
        for img in imgs:
            kdes.append(make_pair_kde(img["m1det"], img["q"], img["dL_app"],
                                      img["chieff"], img["prior_wt"]))
    pair_kdes = stack_pair_kdes(kdes)

    return dict(
        gw_pe=gw_pe, gw_sel=gw_sel, nEvents=n_sing + 2 * P, nsamp=nsamp,
        Ndraw=float(Ndraw),
        singleton_indices=jnp.asarray(partition["singleton_indices"], dtype=jnp.int32),
        pair_indices=jnp.asarray(partition["pair_indices"], dtype=jnp.int32),
        n_singletons=int(partition["n_singletons"]), n_pairs=P,
        pair_kdes=pair_kdes, lensed=lensed,
    )


# =============================================================================
# Likelihood closure
# =============================================================================
def build_cluster_likelihood(opts, inp, decoder):
    """Return a closure logL(sampler_coord) using the branch ParameterDecoder.

    ``decoder.decode(coord) -> (cosmo, survey, pop_params)`` and the decoder was
    built with our ``wl_params``, so ``survey`` already carries the lognormal WL
    parameters and the integer empty-pixel policy. We pass them straight through.
    """
    em = _empty_em_catalog()
    sis = make_sis_lens_params()
    cluster_mode = CLUSTER_MODE_J2 if opts.cluster_mode == "j2" else CLUSTER_MODE_OFF
    wl_backend = WL_BACKEND_LOGNORMAL if opts.wl_backend == "lognormal" else WL_BACKEND_DISABLED
    universe_model = "spectral_sirens_wl" if wl_backend != WL_BACKEND_DISABLED else "spectral_sirens"

    nkept = 0 if inp["lensed"] is None else int(np.asarray(inp["lensed"].m1_src).shape[0])
    log_p_tag = jnp.zeros(nkept)

    def loglike(coord):
        # decode() returns a 5-tuple (cosmo, survey, pop, sky, mark) on current
        # master; the cluster likelihood is WL-only (sky/mark unused here).
        cosmo, survey, pop_params, _sky_params, _mark_params = decoder.decode(
            jnp.asarray(coord)
        )
        return darksiren_log_likelihood_with_clusters(
            cosmo, survey, pop_params,
            inp["gw_pe"], em, inp["gw_sel"], em,
            inp["nEvents"], inp["nsamp"], inp["Ndraw"],
            inp["singleton_indices"], inp["pair_indices"],
            inp["n_singletons"], inp["n_pairs"],
            inp["lensed"], inp["pair_kdes"], sis, log_p_tag,
            opts.pop_model, universe_model,
            sel_batch_size=opts.sel_batch_size, cluster_mode=cluster_mode,
            wl_backend=wl_backend, wl_a=opts.lensing_wl_a, wl_b=opts.lensing_wl_b,
        )
    return loglike


# =============================================================================
# CLI
# =============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="darksirens lensing inference (singleton + J=2 cluster)")
    # data
    p.add_argument("--gw_path", required=True)
    p.add_argument("--gwselection_path", required=True)
    p.add_argument("--lensed_injections_path", default=None)
    p.add_argument("--pair_pe_path", default=None)
    p.add_argument("--partition_path", default=None)
    # model
    p.add_argument("--pop_model", default="powerlaw+peak")
    p.add_argument("--cluster_mode", choices=["off", "j2"], default="j2")
    p.add_argument("--wl_backend", choices=["lognormal", "disabled"], default="lognormal")
    p.add_argument("--lensing_wl_a", type=float, default=4e-3)
    p.add_argument("--lensing_wl_b", type=float, default=1.5)
    p.add_argument("--sl_tau_A", type=float, default=5e-4)
    p.add_argument("--sl_tau_n", type=float, default=3.0)
    # fixing
    p.add_argument("--fix_cosmology", default="true")
    p.add_argument("--fix_survey", default="true")
    p.add_argument("--fix_population", default="false")
    p.add_argument("--fixed_parameter_values", default=None,
                   help="JSON dict of {label: value}")
    p.add_argument("--prior_overrides", default=None,
                   help="JSON dict of {label: [lo, hi]}")
    # sampler
    p.add_argument("--sampler", required=True,
                   choices=["tinyns", "dynesty", "numpyro"])
    p.add_argument("--nlive", type=int, default=2000)
    p.add_argument("--dlogz", type=float, default=0.1)
    p.add_argument("--max_samples", type=int, default=2_000_000)
    p.add_argument("--tinyns_sample", default="rwalk",
                   choices=["rwalk", "slice", "rslice", "prior"],
                   help="tinyns proposal: random walk (default), slice, reflective "
                        "slice, or prior.")
    p.add_argument("--tinyns_kernel", default="jax", choices=["jax", "python"],
                   help="tinyns proposal kernel: jitted JAX (default) or pure Python.")
    p.add_argument("--tinyns_walks", type=int, default=25,
                   help="tinyns: number of random-walk steps per update (sample=rwalk).")
    p.add_argument("--tinyns_replacement_chains", type=int, default=1,
                   help="tinyns: independent random-walk chains run in parallel per "
                        "replacement (rwalk+jax only; default 1).")
    p.add_argument("--tinyns_replacement_chain_schedule", type=str, default=None,
                   help="tinyns: adaptive rwalk+jax escalation schedule, e.g. "
                        "'1,4,16,64,256' (ascending). Starts small and escalates only "
                        "when a stage fails. Mutually exclusive with "
                        "--tinyns_replacement_chains.")
    p.add_argument("--tinyns_max_attempts", type=int, default=None,
                   help="tinyns: max constrained-proposal attempts per replacement "
                        "(tinyns default 10000). Must be >= walks*replacement_chains; "
                        "if unset it auto-raises to that product when needed.")
    p.add_argument("--tinyns_slices", type=int, default=5,
                   help="tinyns: number of slice directions per update.")
    p.add_argument("--tinyns_slice_steps", type=int, default=10,
                   help="tinyns: max stepping-out steps per slice.")
    p.add_argument("--tinyns_step_scale", type=float, default=0.1,
                   help="tinyns: initial step scale as a fraction of the prior width.")
    p.add_argument("--tinyns_progress_interval", type=int, default=100,
                   help="tinyns: iterations between progress-bar updates.")
    p.add_argument("--nuts_warmup", type=int, default=500)
    p.add_argument("--nuts_samples", type=int, default=2000)
    p.add_argument("--nuts_chains", type=int, default=4)
    # perf / memory
    p.add_argument("--pe_max_per_pair", type=int, default=400,
                   help="down-sample PE per pair image (0=keep all). Controls "
                        "the O(N_pe^2 N_y) pair-KDE memory.")
    p.add_argument("--sel_batch_size", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--show_progress", action="store_true")
    p.add_argument("--save_path", default="./")
    return p


def _str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y", "t")


def main():
    opts = build_parser().parse_args()
    # build_parameter_space expects real bools (it does `if not fix_population:`)
    opts.fix_cosmology = _str2bool(opts.fix_cosmology)
    opts.fix_survey = _str2bool(opts.fix_survey)
    opts.fix_population = _str2bool(opts.fix_population)
    os.makedirs(opts.save_path, exist_ok=True)

    print(f"=== darksirens_inference_lensing  [{opts.cluster_mode} | wl={opts.wl_backend}] ===")
    print("loading data ...", flush=True)
    inp = load_inputs(opts)
    print(f"  events: {inp['nEvents']}  ({inp['n_singletons']} singletons "
          f"+ {inp['n_pairs']} pairs)  nsamp/event={inp['nsamp']}", flush=True)

    # --- build parameter space + prior + decoder using branch machinery ---
    fixed = json.loads(opts.fixed_parameter_values) if opts.fixed_parameter_values else {}
    overrides = json.loads(opts.prior_overrides) if opts.prior_overrides else {}
    pop_params_fid = get_fixed_population_params(opts.pop_model)

    # the branch parses "true"/"false" strings into bools internally via opts;
    # build_parameter_space takes the raw opts.fix_* values it was given.
    space = build_parameter_space(
        opts.pop_model, opts.fix_population, opts.fix_cosmology, opts.fix_survey,
        prior_overrides=overrides, fixed_parameter_values=fixed,
    )
    labels = space[0]
    lower = np.asarray(space[1]); upper = np.asarray(space[2])
    prior_transform = make_prior_transform(lower, upper)
    print(f"  free parameters ({len(labels)}): {labels}", flush=True)

    # decoder carries the WL params so decode() returns a survey with wl_params set
    opts.prior_overrides = overrides           # decoder reads getattr(opts,'prior_overrides')
    wl_params = make_lognormal_wl_params(a=opts.lensing_wl_a, b=opts.lensing_wl_b)
    decoder = build_parameter_decoder(
        opts, pop_params_fid, fixed_parameter_values=fixed, wl_params=wl_params,
    )

    loglike = build_cluster_likelihood(opts, inp, decoder)

    # smoke eval at the prior midpoint so JIT compile errors surface early
    mid = 0.5 * (lower + upper)
    t = time.time()
    v = float(loglike(jnp.asarray(mid)))
    print(f"  logL(prior midpoint) = {v:.3f}  [compile {time.time()-t:.1f}s]", flush=True)

    # --- sample ---
    print(f"sampling with {opts.sampler} ...", flush=True)
    results = run_sampler(method=opts.sampler, likelihood=loglike,
                          prior_transform=prior_transform, labels=labels,
                          lower_bound=lower, upper_bound=upper, opts=opts)

    # --- save ---
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = os.path.join(opts.save_path,
                           f"{opts.pop_model}__{opts.cluster_mode}__{opts.sampler}__{ts}")
    os.makedirs(run_dir, exist_ok=True)
    samples = np.asarray(results["samples"])
    np.save(os.path.join(run_dir, "samples.npy"), samples)
    with h5py.File(os.path.join(run_dir, "results.hdf5"), "w") as f:
        f.create_dataset("samples", data=samples)
        f.attrs["labels"] = json.dumps(labels)
        f.attrs["cluster_mode"] = opts.cluster_mode
        f.attrs["wl_backend"] = opts.wl_backend
        f.attrs["n_events"] = inp["nEvents"]
        f.attrs["n_pairs"] = inp["n_pairs"]
        if results.get("logZ") is not None:
            f.attrs["logZ"] = float(results["logZ"])
    with open(os.path.join(run_dir, "settings.json"), "w") as f:
        json.dump({k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in vars(opts).items()}, f, indent=2)
    print(f"saved {samples.shape[0]} samples -> {run_dir}", flush=True)

    # corner (best-effort)
    try:
        from darksirens.utils.plotting import make_production_corner
        fig = make_production_corner(samples, labels)
        fig.savefig(os.path.join(run_dir, "corner.pdf"), bbox_inches="tight", dpi=200)
        print("  corner.pdf written", flush=True)
    except Exception as e:
        print(f"  corner skipped: {e}", flush=True)


if __name__ == "__main__":
    main()