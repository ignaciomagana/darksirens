"""Does the residual bias survive a SPECTROSCOPIC catalog?  One knob, one answer.

Everything else is eliminated: the `f_p` gather (fixed), the parameter estimation
(delta-PE at truth leaves the bias), the event draw (calibrated at fixed catalog
by three routes), the completeness model (gate 8(c), an ideal complete catalog
keeps it), and the selection integral (its detected-`z` support matches the
events to 2%, and the residual has the wrong sign).

What is left in the chain that can move an inferred REDSHIFT is the catalog's
photo-`z` treatment.  The mock scatters every galaxy by `SIGMA_Z = 0.023` and the
analysis represents it by a Gaussian kernel of exactly that width -- verified,
`dz = 0.02300` on every real row.  But a 0.023 scatter at a median host `z` of
0.11 is a 21% fractional width, and two things follow that a matched width does
not fix:

* the kernel is not truncated at `z >= 0`, and 129 galaxies per realization have
  `z_obs < 0`.  A Gaussian centred below zero can only put mass at higher `z`.
* at 21% fractional width the kernel interacts with the steeply rising
  volumetric prior, so whether the analysis convolves in the right direction
  matters at exactly the few-percent level the bias sits at.

Setting `SIGMA_Z = 0` makes the catalog spectroscopic: `z_obs == z_true`, the
kernel collapses to its `1e-4` numerical floor, and every one of those concerns
vanishes at once.  Then:

* bias GONE   => the photo-`z` kernel treatment is the carrier;
* bias STAYS  => it is not, and what remains is the depth relaxation and the
  clustering the analysis does not model.

Run against `gate_complete.py`'s numbers (ideal complete, photo-`z` ON): median
`u` = 0.277, median-of-medians +3.00 km/s, overconfidence 1.714.  This run keeps
the ideal-complete survey too, so the ONLY difference is `SIGMA_Z`.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=7001)
    ap.add_argument("--seed-step", type=int, default=37)
    ap.add_argument("--sigma-z", type=float, default=0.0)
    ap.add_argument("--m-lim", type=float, default=99.0)
    ap.add_argument("--injections", default="data/injections.h5")
    ap.add_argument("--h0-step", type=float, default=2.5)
    ap.add_argument("--n0", type=float, default=5e-5)
    ap.add_argument("--tag", default="iso_specz")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    import jax.numpy as jnp
    import world16 as W16
    import make_mock
    import tier_b
    import arms as A

    # THE SINGLE KNOB.  It is ``make_mock.SIGMA_Z_CAT``, NOT ``W16.SIGMA_Z``:
    # make_mock builds its SurveyConfig with
    # ``redshift_error_floor=SIGMA_Z_CAT`` (its own module constant), and that
    # floor is what both scatters the galaxies and becomes the catalog's stored
    # ``dz``.  Patching W16.SIGMA_Z is a NO-OP -- verified the hard way: a first
    # run came back identical to photo-z ON in all three statistics to three
    # digits, and the catalog's dz was still 0.02300.
    make_mock.SIGMA_Z_CAT = float(a.sigma_z)
    W16.M_LIM = float(a.m_lim)
    make_mock.W16.M_LIM = float(a.m_lim)
    W16.PR6A_DIR = W16.PR6A_DIR / a.tag
    (W16.PR6A_DIR / "data").mkdir(parents=True, exist_ok=True)

    world = W16.build_world()
    try:
        world = world._replace(f_p=np.ones_like(np.asarray(world.f_p)))
    except AttributeError:
        import dataclasses
        world = dataclasses.replace(
            world, f_p=np.ones_like(np.asarray(world.f_p)))

    h0 = np.arange(20.0, 140.0 + 0.5 * a.h0_step, a.h0_step)
    H0T = float(W16.H0_TRUE)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    rows = []
    for k in range(a.n_real):
        seed = a.seed0 + k * a.seed_step
        d = W16.PR6A_DIR / "data" / f"c{k:03d}"
        make_mock.build(seed, d, world=world, n0=a.n0, verbose=False,
                        reuse_injections=a.injections)
        p = tier_b.paths_for(d)
        logl, opts, data = A.build(p, "latent_off")
        vals = np.array([float(logl(jnp.asarray([float(x)]))) for x in h0])
        ok = np.isfinite(vals)
        if not ok.any():
            print(f"[specz] {k+1}/{a.n_real} all -inf", flush=True)
            continue
        v = np.where(ok, vals, -np.inf)
        pdf = np.exp(v - v.max())
        pdf = np.where(ok, pdf, 0.0)
        pdf = pdf / trapz(pdf, h0)
        cdf = np.concatenate([[0], np.cumsum(
            0.5 * (pdf[1:] + pdf[:-1]) * np.diff(h0))])
        cdf /= cdf[-1]
        q = lambda t: float(np.interp(t, cdf, h0))  # noqa: E731
        rows.append(dict(seed=int(seed), median=q(0.5),
                         sigma=0.5 * (q(0.84) - q(0.16)),
                         cdf_at_truth=float(np.interp(H0T, h0, cdf))))
        print(f"[specz] {k+1}/{a.n_real} seed={seed} "
              f"H0={rows[-1]['median']:.1f} u={rows[-1]['cdf_at_truth']:.3f}",
              flush=True)

    med = np.array([r["median"] for r in rows])
    sig = np.array([r["sigma"] for r in rows])
    u = np.array([r["cdf_at_truth"] for r in rows])
    out = dict(n=len(rows), sigma_z=a.sigma_z, m_lim=a.m_lim, H0_true=H0T,
               rows=rows, median_of_medians=float(np.median(med)),
               mean_sigma=float(sig.mean()),
               spread=float(med.std(ddof=1)),
               overconfidence=float(med.std(ddof=1) / sig.mean()),
               median_u=float(np.median(u)),
               cov90=float(((u >= 0.05) & (u <= 0.95)).mean()))
    print(f"\nSPEC-Z GATE (sigma_z = {a.sigma_z}, n = {len(rows)}):")
    print(f"  median u = {out['median_u']:.3f}   "
          f"(photo-z ON gave 0.277; 0.5 is unbiased)")
    print(f"  median of medians {out['median_of_medians']:.2f} "
          f"({out['median_of_medians'] - H0T:+.2f} km/s; photo-z ON gave +3.00)")
    print(f"  overconfidence {out['overconfidence']:.3f} "
          f"(photo-z ON gave 1.714)   cov90 {out['cov90']:.2f}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("GATE_SPECZ_DONE", flush=True)


if __name__ == "__main__":
    main()
