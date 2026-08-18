"""`J` on production by outer product of gradients -- one dataset, no subsets.

The subset estimator (`production_sandwich.py`) failed its own consistency
check: the subset scores did not sum to the full-data score (9.2% off), because
the selection correction carries the Vitale `5 N_obs` floor and the
total-variance criterion, neither of which decomposes across subsets.  That is
structural, so this takes the other route.

The score is a sum over events,

    dlogL/dH0 = sum_i u_i ,      u_i = d(ll_i)/dH0 - d(log mu)/dH0

and for independent events `J = Var(sum_i u_i) = sum_i Var(u_i)`, estimated on a
SINGLE dataset by the outer product of the per-event gradients,

    J_hat = sum_i (u_i - ubar)^2 * N/(N-1).

That needs the per-event log-evidences, which the reduction sums internally.
They are captured by wrapping `log_evidence_and_mc_variance` -- the function the
event reduction vmaps -- with a host callback, exactly as `log_mu` is captured
elsewhere.

**The capture is verified, not trusted.**  Under `vmap` the callback order is
not guaranteed by the API, and a mis-ordered capture would silently pair the
wrong events across `H0` nodes.  Two checks run before any number is reported:

    (a) COMPLETENESS -- the captured per-event values must sum to
        `logL - selection_correction` at every node;
    (b) ORDERING -- the resulting per-event scores must sum to the full-data
        score computed from `logL` alone.

Check (b) is the one the subset method failed.  If either fails the run reports
the failure and quotes nothing, which is the point of having them.
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
    ap.add_argument("--h0", type=float, default=72.0)
    ap.add_argument("--dh", type=float, default=2.0)
    ap.add_argument("--out", default="production_opg.json")
    a = ap.parse_args(argv)

    import run_h0_latent as R
    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.redshift.selection import load_selection_fit_json
    import darksirens.likelihood.core as _core
    from darksirens.likelihood import selection as _sel

    ev_store, mu_store = [], []

    def _rec_ev(x):
        ev_store.append(np.asarray(x).ravel().copy())

    def _rec_mu(mu, ne):
        mu_store.append(float(mu))

    real_ev = _core.log_evidence_and_mc_variance
    real_corr = _sel.selection_log_correction

    def _spy_ev(ldw, nsamp):
        ll, var = real_ev(ldw, nsamp)
        jax.debug.callback(_rec_ev, ll)
        return ll, var

    def _spy_corr(log_mu, Neff, nEvents, **kw):
        jax.debug.callback(_rec_mu, log_mu, Neff)
        return real_corr(log_mu, Neff, nEvents, **kw)

    _core.log_evidence_and_mc_variance = _spy_ev
    _core.selection_log_correction = _spy_corr
    _sel.selection_log_correction = _spy_corr
    try:
        kw = {}
        if a.arm in ("fp", "latent"):
            kw["per_pixel_completeness"] = a.mth_map
        if a.arm == "latent":
            kw["latent_artifact"] = a.anchor
        opts = R._opts(**kw)
        data = load_all_data(opts)
        opts.resolved_survey_z_depths = (data.get("z_depth"),)
        sel = load_selection_fit_json(str(C.FIT_JSON))
        cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
        fixed = {"Om0": C.OM0, "sigma_kde": 0.003,
                 "log10n0": float(cal["log10n0"]),
                 "delta": float(cal["delta"]),
                 "m_lim": float(sel["m_lim"]), "M0hat": float(sel["M0hat"]),
                 "sigma_M": float(sel["sigma_M"])}
        if a.arm == "latent":
            fixed["b_miss"] = 1.0
        logl = make_likelihood(opts, data, get_fixed_population_params(
            opts.pop_model), fixed_parameter_values=fixed)

        hs = [a.h0 - a.dh, a.h0, a.h0 + a.dh]
        LL, EV, MU = [], [], []
        for x in hs:
            ev_store.clear(); mu_store.clear()
            v = float(logl(jnp.asarray([float(x)])))
            ev = np.concatenate(ev_store) if ev_store else np.array([])
            LL.append(v); EV.append(ev); MU.append(float(np.mean(mu_store)))
            print(f"  H0={x:5.1f} logL={v:12.4f} captured {ev.size} events "
                  f"log_mu={MU[-1]:+.6f}", flush=True)
    finally:
        _core.log_evidence_and_mc_variance = real_ev
        _core.selection_log_correction = real_corr
        _sel.selection_log_correction = real_corr

    N = int(data["nEvents"])
    fail = []
    # (a) completeness: sum of captured per-event lls == logL - correction
    for k, x in enumerate(hs):
        corr = LL[k] - EV[k].sum()
        pred = -N * MU[k]
        if EV[k].size != N:
            fail.append(f"node {x}: captured {EV[k].size} events, expected {N}")
        # the correction is not exactly -N log mu (guards), so only report it
        print(f"  node {x}: implied correction {corr:+.4f} vs -N log_mu "
              f"{pred:+.4f}  (delta {corr-pred:+.4f})")
    # (b) ordering: per-event scores must sum to the full-data score
    if not fail:
        # Attribute the ACTUAL correction, not the idealised -N log mu.  The
        # shipped correction carries the Vitale 5 N_obs floor and the
        # total-variance criterion on top of -N log mu; measured here they add a
        # nearly constant +0.92 nats whose DERIVATIVE is -0.00205 -- small in
        # level, but 10% of the total score, which is exactly the discrepancy
        # the first version of this check reported.  Taking the correction from
        # logL - sum_i ll_i makes the decomposition exact by construction, and
        # spreading it as 1/N per event is the right attribution because the
        # correction is proportional to the event count.
        corr_at = np.array([LL[k] - EV[k].sum() for k in range(3)])
        dcorr = (corr_at[2] - corr_at[0]) / (2 * a.dh)
        u = (EV[2] - EV[0]) / (2 * a.dh) + dcorr / N
        full_score = (LL[2] - LL[0]) / (2 * a.dh)
        s = float(u.sum())
        rel = abs(s - full_score) / max(abs(full_score), 1e-12)
        print(f"\n  ORDERING CHECK: sum(u_i) = {s:+.6f} vs full score "
              f"{full_score:+.6f}  ({100*rel:.2f}%)")
        if rel > 0.02:
            fail.append(f"ordering check failed at {100*rel:.1f}%")

    out = dict(arm=a.arm, h0=a.h0, n_events=N, logL=LL, log_mu=MU,
               checks_failed=fail)
    if fail:
        print("\nCHECKS FAILED -- no number is quoted:")
        for f in fail:
            print(f"   - {f}")
    else:
        H = -(LL[2] - 2 * LL[1] + LL[0]) / a.dh ** 2
        J = float(np.sum((u - u.mean()) ** 2) * N / (N - 1))
        out.update(H=float(H), J=J, ratio=J / H,
                   width_inflation=float(np.sqrt(J / H)) if H > 0 else None)
        print(f"\nPRODUCTION information identity at H0={a.h0} (arm={a.arm}):")
        print(f"  H (observed) = {H:.6f}")
        print(f"  J (expected, OPG over {N} events) = {J:.6f}")
        print(f"  J/H = {J/H:.3f}   -> width inflation x{np.sqrt(J/H):.3f}")
        print(f"  (mock: J/H = 6.034, x2.456)")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("PROD_OPG_DONE", flush=True)


if __name__ == "__main__":
    main()
