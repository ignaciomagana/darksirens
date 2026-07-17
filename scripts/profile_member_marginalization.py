#!/usr/bin/env python
"""Profile the LSS-completion member marginalization: FACTORED vs REFERENCE.

Builds a synthetic dark-siren fixture (a K-catalog Q_LSS ensemble) and times
``darksiren_log_likelihood(..., lss_marginalize=True)`` under both
``lss_member_impl`` implementations, for a grid of member counts M and catalog
counts K, reporting compile time, warm median call time, and warm median grad
time as a markdown table.

The factored path precomputes the member-INDEPENDENT per-sample work ONCE (the
population model, z(dL) + Jacobians, proposal reweighting, sky factor, and the
O(N_max_gals) observed-catalog KDE) and vmaps ONLY the cheap missing-galaxy
completion over the M members; the reference re-runs the whole per-member
likelihood.  The speedup grows with M and with the KDE cost (N_rows, galaxies).

Usage (CPU, modest sizes; keep total runtime < ~10 min):

    JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
        python scripts/profile_member_marginalization.py

Sizes are configurable via flags (see ``--help``).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Run-as-script guard: ``python scripts/profile_member_marginalization.py`` puts
# ``scripts/`` on sys.path[0], so a bare ``import darksirens`` would resolve to an
# installed copy (e.g. the main checkout) rather than THIS worktree -- and would
# silently miss the ``lss_member_impl`` static arg.  Prepend the worktree root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.redshift import zgrid
from darksirens.redshift.completion import build_pixel_kde_cache
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog, GWEvent
from darksirens.utils.cosmology import H0Planck, Om0Planck
from darksirens.gw.populations import get_fixed_population_params
from darksirens.likelihood.core import darksiren_log_likelihood

NG = int(zgrid.size)
_Z = np.asarray(zgrid)

COSMO = CosmoParams(H0=H0Planck, Om0=Om0Planck)
# Incomplete catalog over the event range so the missing branch (hence Q) moves
# the likelihood; b_miss = 0 so the members drive the missing branch purely.
SURVEY = SurveyParams(n0=1.0, z50=0.15, w=0.08, delta=0.0, b_miss=0.0, alpha_miss=1.0)
POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))


def _member_logq(slope):
    return slope * (_Z - 0.2)


def _dark_catalog(n_rows, M, gals_per_row=4, seed=0):
    """``n_rows``-row dark-siren catalog carrying an ``M``-member tilted Q_LSS
    ensemble (KDE cache built, compact rows)."""
    rng = np.random.default_rng(seed)
    nmax = gals_per_row
    zg = np.full((n_rows, nmax), 100.0)
    dz = np.full((n_rows, nmax), 1.0)
    w = np.zeros((n_rows, nmax))
    ng = np.zeros(n_rows, dtype=np.int32)
    for i in range(n_rows):
        zs = rng.uniform(0.08, 0.34, nmax)
        zg[i] = zs
        dz[i] = 0.003
        w[i] = 1.0
        ng[i] = nmax
    zg, dz, w, ng = (jnp.asarray(a) for a in (zg, dz, w, ng))
    kde, idx = build_pixel_kde_cache(np.arange(n_rows, dtype=np.int32), zg, n_rows, ngals=ng)
    slopes = np.linspace(-5.0, 5.0, M)
    members = np.stack([np.broadcast_to(_member_logq(s), (n_rows, NG)) for s in slopes])
    return EMCatalog(
        apix=1.0, zgals=zg, dzgals=dz, wgals=w, ngals=ng,
        delta_g_pix_z=jnp.zeros((n_rows, NG)), dN_obs_kde=kde, pixel_to_cache_idx=idx,
        unique_pixels=None,
        lss_completion_logq_members=jnp.asarray(members),
    )


def _gw(n_events, n_samp, seed, n_rows):
    rng = np.random.default_rng(seed)
    total = n_events * n_samp
    m1det = jnp.asarray(rng.uniform(20.0, 60.0, total))
    m2det = jnp.asarray(rng.uniform(8.0, 30.0, total))
    dL = jnp.asarray(rng.uniform(420.0, 1500.0, total))
    chieff = jnp.asarray(rng.uniform(-0.2, 0.2, total))
    prior_wt = jnp.asarray(rng.uniform(0.5, 1.5, total))
    pixels = jnp.asarray(rng.integers(0, n_rows, total), dtype=jnp.int32)
    valid = jnp.ones(total, dtype=jnp.bool_)
    return GWEvent(m1det=m1det, m2det=m2det, dL=dL, chieff=chieff,
                   prior_wt=prior_wt, pixels=pixels, q=m2det / m1det, valid=valid)


def _make_call(nEvents, nsamp, N_sel, N_rows, M, K, impl, materialize, sel_batch_size):
    cat = _dark_catalog(N_rows, M, seed=1)
    gw_pe = _gw(nEvents, nsamp, seed=0, n_rows=N_rows)
    gw_sel = _gw(N_sel, 1, seed=10, n_rows=N_rows)
    common = dict(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=sel_batch_size, lss_marginalize=True, lss_member_impl=impl,
        materialize_redshift_prior_state=materialize,
    )
    if K == 1:
        def call(pop):
            return darksiren_log_likelihood(
                COSMO, SURVEY, pop, gw_pe, cat, gw_sel, cat,
                nEvents, nsamp, float(N_sel), **common,
            )
    else:
        pix2_pe = jnp.stack([gw_pe.pixels] * K, axis=1)
        pix2_sel = jnp.stack([gw_sel.pixels] * K, axis=1)
        gw_pe_k = gw_pe._replace(pixels=pix2_pe)
        gw_sel_k = gw_sel._replace(pixels=pix2_sel)
        mw = jnp.asarray(np.log(np.full(K, 1.0 / K)))

        def call(pop):
            return darksiren_log_likelihood(
                COSMO, SURVEY, pop, gw_pe_k, cat, gw_sel_k, cat,
                nEvents, nsamp, float(N_sel),
                n_catalogs=K,
                mixture_surveys=(SURVEY,) * (K - 1),
                mixture_em_catalogs_pe=(cat,) * (K - 1),
                mixture_em_catalogs_sel=(cat,) * (K - 1),
                mixture_log_weights=mw, **common,
            )
    return call


def _time_median(fn, arg, repeats):
    fn(arg).block_until_ready()  # compile / warm
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(arg).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def _time_compile(fn, arg):
    t0 = time.perf_counter()
    fn(arg).block_until_ready()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nEvents", type=int, default=8)
    ap.add_argument("--nsamp", type=int, default=256)
    ap.add_argument("--N_sel", type=int, default=800)
    ap.add_argument("--N_rows", type=int, default=16)
    ap.add_argument("--gals", type=int, default=8, help="galaxies per catalog row")
    ap.add_argument("--Ms", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--Ks", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sel_batch_size", type=int, default=None)
    args = ap.parse_args()

    print(f"# Member-marginalization profile (backend={jax.default_backend()})")
    print(
        f"# nEvents={args.nEvents} nsamp={args.nsamp} N_sel={args.N_sel} "
        f"N_rows={args.N_rows} gals/row={args.gals} "
        f"sel_batch_size={args.sel_batch_size} repeats={args.repeats}"
    )
    print()
    header = (
        "| K | M | value ref (ms) | value fac (ms) | value speedup | "
        "grad ref (ms) | grad fac (ms) | grad speedup |"
    )
    print(header)
    print("|---|---|---|---|---|---|---|---|")

    rows = []
    for K in args.Ks:
        for M in args.Ms:
            res = {}
            # Value path: barrier ON (production dynesty value path).
            for impl in ("reference", "factored"):
                call = _make_call(args.nEvents, args.nsamp, args.N_sel, args.N_rows,
                                  M, K, impl, materialize=True,
                                  sel_batch_size=args.sel_batch_size)
                res[("value", impl)] = 1e3 * _time_median(
                    call, POP.at[0].set(float(POP[0])), args.repeats
                )
            # Grad path: barrier OFF (production numpyro NUTS path).
            for impl in ("reference", "factored"):
                call = _make_call(args.nEvents, args.nsamp, args.N_sel, args.N_rows,
                                  M, K, impl, materialize=False,
                                  sel_batch_size=args.sel_batch_size)
                gfn = jax.grad(lambda p0, c=call: c(POP.at[0].set(p0)))
                res[("grad", impl)] = 1e3 * _time_median(gfn, float(POP[0]), args.repeats)

            vr, vf = res[("value", "reference")], res[("value", "factored")]
            gr, gf = res[("grad", "reference")], res[("grad", "factored")]
            print(
                f"| {K} | {M} | {vr:.2f} | {vf:.2f} | {vr / vf:.2f}x | "
                f"{gr:.2f} | {gf:.2f} | {gr / gf:.2f}x |"
            )
            rows.append((K, M, vr, vf, gr, gf))

    print()
    print("Value path uses the redshift-prior optimization barrier (materialize="
          "True, dynesty value path); grad path uses materialize=False (numpyro "
          "NUTS path).  speedup = reference / factored.")


if __name__ == "__main__":
    main()
