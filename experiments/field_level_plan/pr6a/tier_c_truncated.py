"""Tier C with the catalog truncated at ``z_depth``, and nothing else changed.

Production truncates its catalog exactly at the depth (22,787,566 DESI
galaxies, max z = 0.3000, zero above).  The closure mock does not: 4.868% of
its galaxies sit above ``z_depth = 0.30``, which it carries only as an HDF5
attribute.  That is checklist rule 7's metadata-only pattern, and it makes the
mock unfaithful to the line it validates.

This runs Tier C's OWN loop -- imported, not reimplemented, so the seeds, the
arms, the anchors and the summary statistics are identical -- with one
intervention: ``make_mock.build`` is wrapped so each realization's catalog is
truncated to ``z <= z_depth`` immediately after it is written.

The intervention is clean because ``c_mode="selection"`` makes ``C(z)`` a
parametric function of the selection fit rather than of the catalog counts, so
removing above-depth rows does not move the completeness model.  The only thing
that changes is which galaxies the observed branch can place a host on.
"""
from __future__ import annotations

import argparse

import make_mock
import tier_c
from truncate_catalog import truncate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=7001)
    ap.add_argument("--z-depth", type=float, default=0.30)
    ap.add_argument("--injections", default="data/injections.h5")
    ap.add_argument("--arms", nargs="*", default=["latent", "latent_off"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    real_build = make_mock.build
    stats = []

    def _build_then_truncate(seed, outdir, **kw):
        r = real_build(seed, outdir, **kw)
        from pathlib import Path
        cat = Path(outdir) / "catalog_pixelated_nside_16.h5"
        dropped, tot, mx = truncate(cat, cat, a.z_depth)
        stats.append(dict(seed=int(seed), dropped=int(dropped),
                          kept=int(tot), max_z=float(mx)))
        return r

    make_mock.build = _build_then_truncate
    try:
        tier_c.main(["--n-real", str(a.n_real), "--seed0", str(a.seed0),
                     "--injections", a.injections, "--arms", *a.arms,
                     "--out", a.out])
    finally:
        make_mock.build = real_build

    if stats:
        import numpy as np
        d = np.array([s["dropped"] for s in stats])
        print(f"\n[truncation] {len(stats)} realizations, dropped "
              f"{d.mean():.0f} +- {d.std():.0f} galaxies each; "
              f"max z now {max(s['max_z'] for s in stats):.4f}")
    print("TIERC_TRUNC_DONE", flush=True)


if __name__ == "__main__":
    main()
