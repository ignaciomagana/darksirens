"""The sandwich correction, measured on the production configuration.

On the closure mock the naive interval is too narrow by `sqrt(J/H) = 2.456`,
where `H = -d2 logL/dH0^2` is the observed information and `J = Var(dlogL/dH0)`
the expected one; the two agree for a correctly specified model and their ratio
is the coverage correction when it is not.  That factor must NOT be assumed to
carry to production: it is a property of a configuration two orders of
magnitude smaller in galaxies.  This measures it here.

Production has ONE dataset, so `J` cannot be taken from an ensemble of
realisations the way it was on the mock.  It does not need to be.  The score is
additive over events,

    dlogL/dH0 = sum_i d(ll_i)/dH0  -  N d(log mu)/dH0
              = sum_i u_i,      u_i = d(ll_i)/dH0 - d(log mu)/dH0

so splitting the 259 events into `K` DISJOINT subsets and evaluating the
likelihood on each gives `K` independent draws of a subset score.  With
`n_k` events in subset `k`, `Var(score_k) = n_k Var(u)`, hence

    J = N Var(u) = N * mean_k[ Var(score_k) / n_k ]

estimated from the spread of the `K` subset scores.  Each subset carries its own
selection term with its own `n_k`, which is what makes the subsets genuine
independent replicates rather than a partition of one number.

`H` is the curvature of the FULL-dataset likelihood, so the two ingredients come
from the same events and the same estimator.
"""
from __future__ import annotations

import argparse
import json

import common as C  # noqa: F401  (pins ZMAX; must be first)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="fp", choices=["nofp", "fp", "latent"])
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--mth-map", default=None)
    ap.add_argument("--k", type=int, default=10, help="disjoint event subsets")
    ap.add_argument("--h0", type=float, default=72.0,
                    help="evaluate the identity here (the full-data peak)")
    ap.add_argument("--dh", type=float, default=2.0)
    ap.add_argument("--out", default="production_sandwich.json")
    a = ap.parse_args(argv)

    import run_h0_latent as R
    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.redshift.selection import load_selection_fit_json

    kw = {}
    if a.arm in ("fp", "latent"):
        kw["per_pixel_completeness"] = a.mth_map
    if a.arm == "latent":
        kw["latent_artifact"] = a.anchor
    opts = R._opts(**kw)
    sel = load_selection_fit_json(str(C.FIT_JSON))
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    fixed = {"Om0": C.OM0, "sigma_kde": 0.003,
             "log10n0": float(cal["log10n0"]), "delta": float(cal["delta"]),
             "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
             "sigma_M": float(sel["sigma_M"])}
    if a.arm == "latent":
        fixed["b_miss"] = 1.0
    pop = get_fixed_population_params(opts.pop_model)
    hs = np.array([a.h0 - a.dh, a.h0, a.h0 + a.dh])

    def curve(o, d):
        o.resolved_survey_z_depths = (d.get("z_depth"),)
        f = make_likelihood(o, d, pop, fixed_parameter_values=fixed)
        return np.array([float(f(jnp.asarray([float(x)]))) for x in hs])

    # ---- full dataset: H ----
    data = load_all_data(opts)
    full = curve(opts, data)
    H = -(full[2] - 2 * full[1] + full[0]) / a.dh ** 2
    print(f"[full] logL {full}  ->  H = {H:.6f}", flush=True)

    # ---- disjoint event subsets: J ----
    nEv = int(data["nEvents"])
    rng = np.random.default_rng(0)
    order = rng.permutation(nEv)
    parts = np.array_split(order, a.k)
    scores, sizes = [], []
    for j, idx in enumerate(parts):
        d = dict(data)
        # slice the per-event axis of everything that carries one
        nsamp = int(data["nsamp"])
        keep_s = np.concatenate([np.arange(i * nsamp, (i + 1) * nsamp)
                                 for i in np.sort(idx)])
        for k_, v in list(d.items()):
            v_ = np.asarray(v) if not isinstance(v, (int, float, str, type(None))) else v
            if isinstance(v_, np.ndarray) and v_.ndim >= 1 and v_.shape[0] == nEv * nsamp:
                d[k_] = v_[keep_s]
        d["nEvents"] = int(idx.size)
        o = R._opts(**kw)
        o.selection_fit = opts.selection_fit
        o.selection_kcorr_by_catalog = opts.selection_kcorr_by_catalog
        try:
            c = curve(o, d)
        except Exception as e:      # noqa: BLE001
            print(f"  subset {j}: FAILED ({type(e).__name__}: {e})", flush=True)
            continue
        s = (c[2] - c[0]) / (2 * a.dh)
        scores.append(float(s)); sizes.append(int(idx.size))
        print(f"  subset {j}: n={idx.size} score={s:+.5f}", flush=True)

    if len(scores) < 3:
        raise SystemExit("too few usable subsets to estimate J")
    sc = np.array(scores); nk = np.array(sizes, float)
    var_u = float(np.var(sc, ddof=1) / nk.mean())
    J = nEv * var_u
    out = dict(arm=a.arm, h0=a.h0, k=len(scores), n_events=nEv,
               H=float(H), J=float(J), ratio=float(J / H),
               width_inflation=float(np.sqrt(J / H)) if H > 0 else None,
               subset_scores=scores, subset_sizes=sizes)
    print(f"\nPRODUCTION information identity at H0={a.h0} (arm={a.arm}):")
    print(f"  H (observed) = {H:.6f}")
    print(f"  J (expected) = {J:.6f}   from {len(scores)} disjoint subsets")
    print(f"  J/H = {J/H:.3f}   -> width inflation x{np.sqrt(J/H):.3f}")
    print(f"  (mock: J/H = 6.034, x2.456)")
    json.dump(out, open(a.out, "w"), indent=1)
    print("PROD_SANDWICH_DONE", flush=True)


if __name__ == "__main__":
    main()
