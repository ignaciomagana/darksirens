"""Which per-row term changes with the compact row set?  Spy on the real state.

`log Z` is `log(N_obs_total + trapezoid(dN_exp * V))` and both ingredients were
measured bit-identical across event sets, so the survey-global normalizer cannot
be carrying the non-additivity.  That leaves the per-row numerator:

    ll_i = logaddexp( log_Nobs[pix] + log p_cat(z|pix),  log dN_miss[pix](z) )
           - log Z_global

All of those are per-row, so for a pixel present in two views they must agree --
and the additivity residual says they do not.  This intercepts
`prepare_redshift_prior_state` in the REAL likelihood run (rather than
reconstructing an EMCatalog by hand, which would be a second, unverified code
path) and compares `log_Nobs`, `N_miss` and `log_Z_global` element by element for
the pixels two views SHARE.

`A` and `B` draw events from different source realizations, but `build_regrouped`
copies the shared files -- catalog, depth map, injections, fits -- from source 0,
so all views index ONE catalog and a shared pixel means the same galaxies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--views", nargs="+", required=True)
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--h0", type=float, default=68.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax
    import jax.numpy as jnp
    import arms as A
    import tier_b
    import darksirens.likelihood.core as _core

    real_prep = _core.prepare_redshift_prior_state
    grab = {}

    def _spy(model, cosmo, survey, em_catalog, **kw):
        st = real_prep(model, cosmo, survey, em_catalog, **kw)

        def _rec(lz, ln, nm):
            grab.setdefault("log_Z_global", []).append(float(np.asarray(lz)))
            grab.setdefault("log_Nobs", []).append(np.asarray(ln).copy())
            grab.setdefault("N_miss", []).append(np.asarray(nm).copy())

        lz = getattr(st, "log_Z_global", None)
        if lz is not None:
            jax.debug.callback(_rec, lz, st.log_Nobs,
                               jnp.trapezoid(st.dN_miss, axis=-1))
        return st

    got = {}
    _core.prepare_redshift_prior_state = _spy
    try:
        for v in a.views:
            name = Path(v).name
            p = tier_b.paths_for(v)
            logl, opts, data = A.build(p, a.arm)
            grab.clear()
            float(logl(jnp.asarray([float(a.h0)])))
            up_raw = data.get("unique_pixels_pe")
            up = (None if up_raw is None
                  else np.asarray(up_raw).reshape(-1))
            # Take every captured state.  The row count tells us whether the
            # state is over the COMPACT rows or the FULL sky, and that is itself
            # the answer to where an event-set dependence could enter.
            sizes = [int(x.size) for x in grab.get("log_Nobs", [])]
            k = 0
            n_rows = sizes[k] if sizes else 0
            # rows are global pixels when the state is full-sky
            pix = (np.arange(n_rows) if up is None or n_rows != up.size
                   else up)
            got[name] = dict(pix=pix, log_Nobs=grab["log_Nobs"][k],
                             N_miss=grab["N_miss"][k],
                             log_Z_global=grab["log_Z_global"][k],
                             all_sizes=sizes,
                             all_logZ=[float(z) for z in grab["log_Z_global"]],
                             n_pe_rows=(None if up is None else int(up.size)))
            print(f"[rows] {name}: state rows {sizes}, PE view rows "
                  f"{got[name]['n_pe_rows']},  log_Z_global = "
                  f"{got[name]['all_logZ']}", flush=True)
    finally:
        _core.prepare_redshift_prior_state = real_prep

    names = list(got)
    out = dict(arm=a.arm, h0=a.h0,
               log_Z_global={n: got[n]["log_Z_global"] for n in names})
    print("\nlog_Z_global:  " + "   ".join(
        f"{n}={got[n]['log_Z_global']:.9f}" for n in names))
    ref = names[-1]
    for n in names[:-1]:
        common, ia, ib = np.intersect1d(got[n]["pix"], got[ref]["pix"],
                                        return_indices=True)
        if common.size == 0:
            print(f"  {n} vs {ref}: no shared pixels")
            continue
        d_nobs = np.abs(got[n]["log_Nobs"][ia] - got[ref]["log_Nobs"][ib])
        d_miss = np.abs(got[n]["N_miss"][ia] - got[ref]["N_miss"][ib])
        rel = d_miss / np.maximum(np.abs(got[ref]["N_miss"][ib]), 1e-300)
        print(f"  {n} vs {ref}: {common.size} shared pixels | "
              f"max|d log_Nobs| = {d_nobs.max():.3e} | "
              f"max|d N_miss| = {d_miss.max():.3e} (rel {rel.max():.3e}) | "
              f"d log_Z = {got[n]['log_Z_global'] - got[ref]['log_Z_global']:+.3e}")
        out[f"{n}_vs_{ref}"] = dict(
            n_shared=int(common.size),
            max_d_log_Nobs=float(d_nobs.max()),
            max_d_N_miss=float(d_miss.max()),
            max_rel_d_N_miss=float(rel.max()),
            d_log_Z=float(got[n]["log_Z_global"] - got[ref]["log_Z_global"]))
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"[wrote] {a.out}")
    print("ROW_TERMS_DONE", flush=True)


if __name__ == "__main__":
    main()
