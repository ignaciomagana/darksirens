"""Rule 6 of the mock-data-DAG checklist, tested on the closure mock.

Tier C is overconfident by ~2.5x in the latent arm AND in the no-field
control, and 82-92% of the H0 variance is the EVENT draw at fixed catalog.
``pe_calibration.py`` already ruled out the most economical explanation: the
synthetic PE is NOT over-sharp (residual sd 1.059, not the 2.6 required).  What
it found instead is a residual MEAN of +0.486 with a failing PP
(KS p = 0.0055) -- a DISPLACEMENT, not a width.

Essick & Fishbach's checklist has exactly one entry whose signature is
"displaced AND spuriously narrow", and it is not about the PE:

    a noisy or mis-sloped selection integral produces posteriors that are
    BOTH displaced AND spuriously narrow

and the standing way to make one is rule 6: an injection set that covers the
detectable population at the TRUE hyperparameters but not at the TRIAL ones.
That truncates the selection integral at the edges of the scan and puts a slope
in ``log mu`` which is an artifact of coverage, not physics.  It sits UPSTREAM
of the field -- which is why the no-field control suffers it identically -- and
it lives in the event/selection channel, which is where the variance split put
the variance.  Every symptom is accounted for by one cause.

This measures the two quantities rule 6 gates on, taken from the LIVE
likelihood rather than re-derived: ``selection_log_correction`` is the single
function the estimator hands ``(log_mu, Neff)`` to, so wrapping it records
exactly what the estimator saw.

    Neff(H0)    the selection integral's effective sample count.  A collapse
                away from the truth IS the coverage failure.
    log_mu(H0)  the selection integral; its SLOPE is what biases H0.

Gate: Neff must not collapse relative to its value at the truth, and log_mu
must be smooth across the scan.  A Neff that falls off at the edges while the
posterior is quoted over the whole range means the interval is being set by
Monte-Carlo noise in mu rather than by the events.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="data/rb")
    ap.add_argument("--injections", default=None)
    ap.add_argument("--arm", default="latent_off")
    ap.add_argument("--h0-min", type=float, default=30.0)
    ap.add_argument("--h0-max", type=float, default=130.0)
    ap.add_argument("--h0-step", type=float, default=5.0)
    ap.add_argument("--out", default="selection_coverage.json")
    args = ap.parse_args(argv)

    import jax.numpy as jnp
    import arms
    from darksirens.likelihood import selection as _sel

    # Capture (log_mu, Neff) as the estimator computes them.  Wrapping rather
    # than re-deriving: a re-derivation that disagreed would leave it ambiguous
    # which of the two was wrong.
    captured = []
    real_corr = _sel.selection_log_correction

    # ``selection_log_correction`` is called INSIDE the jit, so its arguments
    # are tracers and cannot be cast host-side.  ``jax.debug.callback`` runs a
    # host function at execution time with the concrete values, which is the
    # supported way to observe an intermediate without changing what the
    # function returns -- the estimator's arithmetic is untouched.
    import jax as _jax

    def _record(mu, ne):
        captured.append((float(mu), float(ne)))

    def _spy(log_mu, Neff, nEvents, **kw):
        _jax.debug.callback(_record, log_mu, Neff)
        return real_corr(log_mu, Neff, nEvents, **kw)

    _sel.selection_log_correction = _spy
    # The likelihood core imported the symbol directly, so patch it there too.
    import darksirens.likelihood.core as _core
    real_core = _core.selection_log_correction
    _core.selection_log_correction = _spy

    try:
        # Reuse tier_b's own path map rather than duplicating the key names:
        # a private copy that drifted would silently point this diagnostic at
        # a different world than the tier it is explaining.
        from tier_b import paths_for
        paths = paths_for(args.outdir)
        if args.injections:
            paths["selection"] = args.injections
        logl, _opts, _data = arms.build(paths, args.arm)
        h0 = np.arange(args.h0_min, args.h0_max + 0.5 * args.h0_step,
                       args.h0_step)
        rows = []
        for h in h0:
            captured.clear()
            v = float(logl(jnp.asarray([float(h)])))
            # the member vmap calls the correction once per member; the
            # selection integral is member-dependent only through Q, so record
            # the spread as well as the value
            mus = [c[0] for c in captured]
            neffs = [c[1] for c in captured]
            rows.append(dict(h0=float(h), logl=v,
                             log_mu=float(np.mean(mus)) if mus else None,
                             log_mu_spread=float(np.ptp(mus)) if len(mus) > 1 else 0.0,
                             neff=float(np.mean(neffs)) if neffs else None,
                             neff_min=float(np.min(neffs)) if neffs else None,
                             n_calls=len(captured)))
            print(f"  H0={h:6.1f} logL={v:12.3f} log_mu={rows[-1]['log_mu']} "
                  f"Neff={rows[-1]['neff']}", flush=True)
    finally:
        _sel.selection_log_correction = real_corr
        _core.selection_log_correction = real_core

    ne = np.array([r["neff"] for r in rows if r["neff"] is not None])
    lm = np.array([r["log_mu"] for r in rows if r["log_mu"] is not None])
    summary = dict(
        arm=args.arm, n_nodes=len(rows),
        neff_min=float(ne.min()) if ne.size else None,
        neff_max=float(ne.max()) if ne.size else None,
        neff_ratio=float(ne.max() / ne.min()) if ne.size and ne.min() > 0 else None,
        log_mu_range=float(lm.max() - lm.min()) if lm.size else None,
    )
    json.dump(dict(summary=summary, rows=rows), open(args.out, "w"), indent=1)
    print("\nSUMMARY:", json.dumps(summary, indent=1))
    if summary["neff_ratio"] and summary["neff_ratio"] > 3.0:
        print(f"\n*** Neff varies by {summary['neff_ratio']:.1f}x across the "
              f"scan -- rule 6 coverage failure is LIVE ***")


if __name__ == "__main__":
    main()
