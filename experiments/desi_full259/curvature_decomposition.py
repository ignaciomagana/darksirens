"""Is production's H0 width set by the same near-cancellation as the mock's?

On the closure mock the posterior width comes from a 65% cancellation:

    total     d2 logL/dH0^2 = -0.022314   -> sigma = 6.69 km/s
    events                  = -0.064303   (288% of the total)
    selection -N log mu     = +0.041988   (-188% of the total)

and Tier C says the net is ~5.4x too curved for the scatter the estimator
actually produces, i.e. a ~30-40% error in ONE of two large, mostly-cancelling
terms.  Gate 8(a) localised that to the estimator rather than to the mock's PE.

The question this answers is the one that matters for the shipped number: does
the PRODUCTION likelihood have the same structure?  A near-cancellation is not
itself a defect -- it is the normal shape of a hierarchical likelihood with
selection -- but the DEGREE of cancellation says how much a fractional error in
either term is amplified in the width.  If production cancels as hard as the
mock, then whatever is wrong on the mock would be similarly amplified there; if
production's terms do not cancel, the mock's sensitivity is a property of the
small configuration and does not transfer.

`log_mu` is taken from the LIVE estimator by wrapping
`selection_log_correction` (the one function the likelihood hands it to), not
re-derived, so the decomposition is of the numbers the run actually used.
"""
from __future__ import annotations

import argparse
import json

import common as C  # noqa: F401  (pins ZMAX; must be first)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from darksirens.redshift.selection import load_selection_fit_json  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="fp", choices=["nofp", "fp", "latent"])
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--mth-map", default=None)
    ap.add_argument("--h0-lo", type=float, default=60.0)
    ap.add_argument("--h0-hi", type=float, default=86.0)
    ap.add_argument("--h0-step", type=float, default=2.0)
    ap.add_argument("--out", default="curvature_decomposition.json")
    a = ap.parse_args(argv)

    import run_h0_latent as R
    from darksirens.inference.data import load_all_data
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.likelihood.factory import make_likelihood
    from darksirens.likelihood import selection as _sel
    import darksirens.likelihood.core as _core

    captured = []

    def _record(mu, ne):
        captured.append((float(mu), float(ne)))

    real = _sel.selection_log_correction

    def _spy(log_mu, Neff, nEvents, **kw):
        jax.debug.callback(_record, log_mu, Neff)
        return real(log_mu, Neff, nEvents, **kw)

    _sel.selection_log_correction = _spy
    _core.selection_log_correction = _spy
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

        h = np.arange(a.h0_lo, a.h0_hi + 0.5 * a.h0_step, a.h0_step)
        rows = []
        for x in h:
            captured.clear()
            v = float(logl(jnp.asarray([float(x)])))
            mus = [c[0] for c in captured]
            rows.append(dict(h0=float(x), logl=v,
                             log_mu=float(np.mean(mus)) if mus else None))
            print(f"  H0={x:6.1f} logL={v:14.4f} log_mu={rows[-1]['log_mu']}",
                  flush=True)
    finally:
        _sel.selection_log_correction = real
        _core.selection_log_correction = real

    hh = np.array([r["h0"] for r in rows])
    ll = np.array([r["logl"] for r in rows])
    lm = np.array([r["log_mu"] for r in rows])
    N = 259
    selterm = -N * lm
    ev = ll - selterm
    i = int(np.argmax(ll))
    i = min(max(i, 1), len(hh) - 2)
    s = hh[1] - hh[0]
    curv = lambda y: (y[i + 1] - 2 * y[i] + y[i - 1]) / s ** 2  # noqa: E731
    ct, ce, cs = curv(ll), curv(ev), curv(selterm)
    out = dict(arm=a.arm, rows=rows, n_obs=N, peak_h0=float(hh[i]),
               curv_total=float(ct), curv_events=float(ce),
               curv_selection=float(cs),
               sigma_from_curv=float(np.sqrt(-1 / ct)) if ct < 0 else None,
               cancellation=float(abs(cs) / abs(ce)) if ce else None)
    print("\nPRODUCTION curvature decomposition at the peak "
          f"(H0={out['peak_h0']:.0f}, arm={a.arm}):")
    print(f"  total     = {ct:+.6f}   -> sigma = {out['sigma_from_curv']} km/s")
    print(f"  events    = {ce:+.6f}   ({100*ce/ct:6.1f}% of total)")
    print(f"  selection = {cs:+.6f}   ({100*cs/ct:6.1f}% of total)")
    print(f"  cancellation |sel|/|events| = {out['cancellation']:.3f}"
          "   (mock: 0.653)")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")


if __name__ == "__main__":
    main()
