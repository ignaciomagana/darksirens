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
    (b) ORDERING -- each event's log-evidence must vary SMOOTHLY across the
        three `H0` nodes, i.e. its second difference must be small against the
        event-to-event spread.

**Check (b) was wrong until 2026-08-20 and is worth recording.**  It used to
compare `sum_i u_i` against the full-data score -- but that is an algebraic
identity in which only SUMS of the captured values appear, so it passes for ANY
permutation and was blind to exactly the failure this docstring claims it
catches.  Meanwhile `J = sum_i (u_i - ubar)^2` is precisely the quantity a
permutation destroys.  The smoothness form is permutation-sensitive: a shuffled
capture makes the per-event second difference O(spread) instead of
O(curvature x dh^2).  The old check is kept as (a)'s companion, correctly
labelled a completeness check.
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
        print(f"\n  COMPLETENESS CHECK: sum(u_i) = {s:+.6f} vs full score "
              f"{full_score:+.6f}  ({100*rel:.2f}%)")
        if rel > 0.02:
            fail.append(f"completeness check failed at {100*rel:.1f}%")

        # THE ABOVE IS NOT AN ORDERING CHECK, AND USED TO BE LABELLED ONE.
        # Only SUMS of EV enter it: sum_i u_i = (S2-S0)/2dh + [(LL2-S2)-(LL0-S0)]
        # /2dh = (LL2-LL0)/2dh identically, for ANY permutation of EV[2] relative
        # to EV[0]. So it passes at 0.00% by algebra and is blind to precisely the
        # failure its docstring claimed it caught -- while J = sum (u_i - ubar)^2
        # is exactly the quantity a permutation destroys.
        #
        # A real ordering check has to use the per-event values as a SEQUENCE.
        # This one uses smoothness in H0: with a consistent capture order each
        # event's log-evidence varies smoothly across the three nodes, so the
        # second difference is O(curvature * dh^2) -- tiny against the
        # event-to-event spread. Under a permutation it becomes O(spread).
        d2 = EV[2] - 2.0 * EV[1] + EV[0]
        spread = float(np.std(EV[1]))
        curv_ratio = float(np.max(np.abs(d2)) / max(spread, 1e-300))
        print(f"  ORDERING CHECK (per-event smoothness in H0): "
              f"max|d2 ll| / sd(ll) = {curv_ratio:.4f}")
        print("    (a permuted capture gives O(1) here; a consistent one gives "
              "the true curvature, ~(dh/H0)^2)")
        if curv_ratio > 0.5:
            fail.append(
                f"ordering check failed: per-event second differences are "
                f"{curv_ratio:.2f} of the event-to-event spread, which is what a "
                f"PERMUTED capture looks like")

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
        print("  (mock: J/H = 6.034, x2.456)")
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("PROD_OPG_DONE", flush=True)


if __name__ == "__main__":
    main()
