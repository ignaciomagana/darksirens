"""Tier C under delta-function PE at truth -- checklist gate 8(a).

Runs Tier C's OWN loop (imported, not reimplemented, so seeds/arms/anchors/
statistics are identical) with one intervention: each realization's PE samples
are collapsed onto that realization's true parameters immediately after the
mock is written.  See ``delta_pe.py`` for why the frames and the PE prior have
to be handled explicitly.

Read the outcome as: correct coverage => the likelihood is calibrated and the
dispersion is the mock's; wrong coverage => the width deficit is the
estimator's and is independent of the mock's PE.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import make_mock
import tier_c
import world16 as W16
from delta_pe import make_delta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=7001)
    ap.add_argument("--injections", default="data/injections.h5")
    ap.add_argument("--arms", nargs="*", default=["latent_off"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    real_build = make_mock.build
    spreads = []

    def _build_then_delta(seed, outdir, **kw):
        r = real_build(seed, outdir, **kw)
        gw = Path(outdir) / "gw_events.h5"
        spreads.append(make_delta(gw, gw))
        return r

    # tier_c writes into W16.PR6A_DIR/"data"/c{k:03d}, a HARDCODED path, so two
    # variants running at once clobber each other's realizations (found the
    # hard way: a FileNotFoundError on truth.json when the other run had
    # already moved on). Redirect this driver to a private tree.
    _real_dir = W16.PR6A_DIR
    W16.PR6A_DIR = _real_dir / "iso_deltape"
    (W16.PR6A_DIR / "data").mkdir(parents=True, exist_ok=True)
    make_mock.build = _build_then_delta
    try:
        tier_c.main(["--n-real", str(a.n_real), "--seed0", str(a.seed0),
                     "--injections", a.injections, "--arms", *a.arms,
                     "--out", a.out])
    finally:
        make_mock.build = real_build
        W16.PR6A_DIR = _real_dir

    if spreads:
        print(f"\n[delta-PE] {len(spreads)} realizations, max within-event "
              f"dL spread over all = {max(spreads):.3e}")
    print("TIERC_DELTA_DONE", flush=True)


if __name__ == "__main__":
    main()
