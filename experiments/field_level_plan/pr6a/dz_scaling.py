"""How does the H0 bias scale with the kernel's FRACTIONAL width?

The photo-`z` bias is now known to be a production-scale effect, but its size on
production is only bounded, not measured: the mock's hosts sit at a median
`z` of ~0.11 against production's 0.237, so the same `dz` is a ~21% fractional
kernel here and ~10% there.

Rather than move the mock's hosts (which would change the detection horizon and
half a dozen other things), scale the KERNEL and read off the curve.  Scaling
production's empirical `dz` distribution by ``s`` gives a fractional width
``s x 21%`` at the mock's own hosts, so

    s = 1.00  ->  ~21%   (the faithful mock; measured u = 0.251)
    s = 0.50  ->  ~10%   <-- PRODUCTION'S FRACTIONAL WIDTH
    s = 0.25  ->  ~5%
    s = 0.00  ->  0%     (the flat-1e-4 run; measured u = 0.401)

The ``s = 0.5`` point is therefore not an extrapolation: it is the mock evaluated
at production's own fractional kernel, which is the number the production line
needs.  The other points give the shape, so the reading does not rest on one run.

Each scale runs Tier C's own loop on the same seeds, so everything except the
kernel is held fixed.
"""
from __future__ import annotations

import argparse
import json

import make_mock
import tier_c
import world16 as W16


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dz-scale", type=float, required=True)
    ap.add_argument("--n-real", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=7001)
    ap.add_argument("--injections", default="data/injections.h5")
    ap.add_argument("--arms", nargs="*", default=["latent_off", "latent"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    real_build = make_mock.build

    def _scaled(seed, outdir, **kw):
        kw.setdefault("dz_scale", float(a.dz_scale))
        return real_build(seed, outdir, **kw)

    # Private tree per scale: tier_c writes into W16.PR6A_DIR/"data"/c{k:03d},
    # a HARDCODED path, so concurrent scales would clobber each other's
    # realizations (this cost two invalidated runs earlier in the campaign).
    _real = W16.PR6A_DIR
    W16.PR6A_DIR = _real / f"iso_dz{a.dz_scale:g}"
    (W16.PR6A_DIR / "data").mkdir(parents=True, exist_ok=True)
    make_mock.build = _scaled
    try:
        tier_c.main(["--n-real", str(a.n_real), "--seed0", str(a.seed0),
                     "--injections", a.injections, "--arms", *a.arms,
                     "--out", a.out])
    finally:
        make_mock.build = real_build
        W16.PR6A_DIR = _real
    print(f"[dz-scaling] dz_scale={a.dz_scale:g} done")
    print("DZ_SCALING_DONE", flush=True)


if __name__ == "__main__":
    main()
