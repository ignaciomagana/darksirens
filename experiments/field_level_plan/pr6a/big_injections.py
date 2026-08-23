"""Rebuild the closure world with a MUCH larger injection set, and nothing else.

Tier C is overconfident by ~2.5x, centred, in the latent arm and equally in the
no-field control.  ``selection_coverage.py`` measured the selection integral's
effective sample count across the scan and found the Monte-Carlo standard
deviation of the ``-N_obs log mu`` term running **0.37 to 0.63 nats**, against a
likelihood curvature of **0.279 nats** over +-5 km/s at the peak.

Why that produces exactly Tier C's signature.  The injection set is SHARED
across every realization (``tier_c.py`` passes ``reuse_injections``), so its
Monte-Carlo error is one fixed, smooth-ish function of ``H0`` added to
**every** realization's log-likelihood.  A common additive distortion with
structure on the curvature scale changes the WIDTH of every posterior while
leaving the ensemble CENTRED -- the peak locations still scatter with the
events, but each interval is drawn from a landscape whose curvature is partly
Monte-Carlo noise.  Centred, too narrow, identical in both arms, with the
scatter living in the event draw: that is the whole of what Tier C reports.

The mock's set is small compared with the line it stands in for: **65,791
detected from 3e6 draws**, against production's **1,067,946 from 9.44e8**.

This script rebuilds ONE world with ``target_det`` raised, changing nothing
else -- same seed, same catalog, same events.  If the overconfidence falls when
only the injection count changes, the cause is the selection integral's
Monte-Carlo noise and Tier C's design is what needs fixing, not the estimator.
If it does not move, this suspect is eliminated with a number and the search
continues.
"""
from __future__ import annotations

import argparse

import make_mock


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7001)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--target-det", type=int, required=True)
    ap.add_argument("--ndraw", type=int, default=200_000_000)
    args = ap.parse_args(argv)

    print(f"[big-inj] target_det={args.target_det:,} ndraw<={args.ndraw:,}",
          flush=True)
    make_mock.build(args.seed, args.outdir, ndraw=args.ndraw,
                    target_det=args.target_det, verbose=True)
    print("BIG_INJ_DONE", flush=True)


if __name__ == "__main__":
    main()
