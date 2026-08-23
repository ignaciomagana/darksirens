"""Tier C with the CATALOG side perturbed -- gate 8(a)'s next test.

Gate 8(a) collapsed every event's PE onto its true parameters and the
realization-to-realization scatter did not move, so the ~2.5x width deficit is
in the ESTIMATOR's own terms rather than in the data or the event construction.
Of those terms the selection integral is clean (a power law of index +3.814,
curvature agreeing with it to 16%), which leaves the event/catalog term, and
the information identity localizes it there quantitatively: the event term's
score varies 1.73x more than its own curvature implies.  In words, the redshift
prior is more informative than the catalog warrants.

That statement has three testable mechanisms, and this runs Tier C's OWN loop
(imported, so seeds/arms/statistics are identical) once per mechanism:

``kde_wide``    broaden the catalog redshift kernels (``sigma_kde``).  If the
                prior is over-informative because each galaxy is placed too
                sharply in redshift, this closes the gap.
``fp_scaled``   scale the per-pixel selection fraction by ``s < 1``, i.e. assert
                the survey is LESS complete than the model believes.  Moves
                weight from the catalog branch to the volumetric missing branch
                without touching the kernels.
``per_pixel``   swap the completeness ESTIMATOR: the matched-kernel per-pixel
                ratio measured from the counts, instead of the fitted survey
                curve ``C(z; theta_sel)``.  ``f_p`` is dropped with it, and not
                as a confound -- a count-derived ``C`` already contains the mask
                loss, so multiplying would double-count it.

None of these is a fix; they are a decomposition.  What each returns is the
overconfidence ratio against the shipped arm's 2.48-2.65, and the mechanism
that moves it is the one carrying the defect.
"""
from __future__ import annotations

import argparse

import h5py
import numpy as np

import arms as A
import make_mock
import tier_c
import world16 as W16


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--intervention", required=True,
                    choices=["kde_wide", "fp_scaled", "per_pixel"])
    ap.add_argument("--sigma-kde", type=float, default=0.01,
                    help="kde_wide: replaces the shipped 0.003")
    ap.add_argument("--fp-scale", type=float, default=0.8,
                    help="fp_scaled: f_p -> s f_p (s < 1 = less complete)")
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=7001)
    ap.add_argument("--injections", default="data/injections.h5")
    ap.add_argument("--arms", nargs="*", default=["latent_off"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    real_build = make_mock.build
    real_opts = A.make_opts
    real_fixed = A.fixed_values

    def _build_then_scale_fp(seed, outdir, **kw):
        r = real_build(seed, outdir, **kw)
        from pathlib import Path
        p = Path(outdir) / "mth_map_nside16.h5"
        with h5py.File(p, "r+") as f:
            mf = np.asarray(f["masked_frac"][...], dtype=np.float64)
            # f_p = 1 - masked_frac, so f_p -> s f_p is
            # masked_frac -> 1 - s (1 - masked_frac).
            f["masked_frac"][...] = 1.0 - a.fp_scale * (1.0 - mf)
        return r

    def _opts_per_pixel(paths, arm, **kw):
        o, sel = real_opts(paths, arm, **kw)
        o.c_mode = "per_pixel"
        o.per_pixel_completeness = None
        return o, sel

    def _fixed_no_theta_sel(paths, sel):
        d = dict(real_fixed(paths, sel))
        for k in ("m_lim", "M0hat", "sigma_M"):
            d.pop(k, None)
        return d

    def _fixed_wide_kde(paths, sel):
        d = dict(real_fixed(paths, sel))
        d["sigma_kde"] = float(a.sigma_kde)
        return d

    tag = {"kde_wide": f"iso_kde{a.sigma_kde:g}",
           "fp_scaled": f"iso_fp{a.fp_scale:g}",
           "per_pixel": "iso_perpix"}[a.intervention]
    _real_dir = W16.PR6A_DIR
    W16.PR6A_DIR = _real_dir / tag
    (W16.PR6A_DIR / "data").mkdir(parents=True, exist_ok=True)
    if a.intervention == "fp_scaled":
        make_mock.build = _build_then_scale_fp
    elif a.intervention == "per_pixel":
        A.make_opts = _opts_per_pixel
        A.fixed_values = _fixed_no_theta_sel
    else:
        A.fixed_values = _fixed_wide_kde
    try:
        tier_c.main(["--n-real", str(a.n_real), "--seed0", str(a.seed0),
                     "--injections", a.injections, "--arms", *a.arms,
                     "--out", a.out])
    finally:
        make_mock.build = real_build
        A.make_opts = real_opts
        A.fixed_values = real_fixed
        W16.PR6A_DIR = _real_dir
    print(f"[catalog-side] intervention={a.intervention} "
          f"sigma_kde={a.sigma_kde} fp_scale={a.fp_scale}")
    print("TIERC_CATSIDE_DONE", flush=True)


if __name__ == "__main__":
    main()
