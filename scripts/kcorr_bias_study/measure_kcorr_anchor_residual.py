#!/usr/bin/env python
"""Measure the selection-fit zero-point (K-correction anchor) residual.

Review task #18 measurement.  READ-ONLY: nothing in ``darksirens`` is
modified; this script only calls the public offline-fit and selection-curve
API.

What is actually stale
----------------------
``c_sel_gaussian(z, m_lim, M0hat, sigma_M, H0, Om0, w0, wa, k_corr_coeffs)``
evaluates ``DM(z)`` AT THE PROPOSAL cosmology, so the z-SHAPE of the distance
modulus is already proposal-consistent.  The only quantity frozen at the
fiducial background is the pair ``theta = (M0hat, sigma_M)`` measured by the
offline fit, whose Laplace covariance becomes the Gaussian prior on the
sampled theta.  Therefore the residual between the stamped model and a
fully self-consistent per-proposal re-anchor is EXACTLY

    delta(Theta) = M0hat(Theta) - M0hat(Theta_fid)      [mag, constant in z]
    dsig(Theta)  = sigma_M(Theta) - sigma_M(Theta_fid)

because both curves use the same ``DM(z; Theta)``; only the zero point (and
width) differ.  The "z-dependent 0.1 mag" of the module docstring is the
DRIVER of delta -- the z-shape of ``Delta DM(z; Theta)`` over the fit
sample -- not the residual seen by the likelihood.  Both are reported.

Stages
------
  1. ``dm``     : Delta DM(z; Theta) = DM(z;100,Theta) - DM(z;100,fid) over
                  the sampled prior box; confirms/corrects the "~0.1 mag".
  2. ``refit``  : synthesize a magnitude-limited catalog, fit at the fiducial
                  background, then RE-FIT at each Theta on the prior box to
                  measure delta(Theta) and dsig(Theta) directly, against the
                  fiducial fit's own Laplace sd.
  3. ``csel``   : propagate delta into C_sel(z) and into the out-of-catalog
                  ("missing") budget that the completion table prices.

Usage
-----
    JAX_PLATFORMS=cpu python scripts/kcorr_bias_study/measure_kcorr_anchor_residual.py \
        --stage all --ngal 200000 --out /path/to/outdir
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Run-from-anywhere: the repo is used in-place (no pip install), and running
# ``python scripts/.../this.py`` puts the SCRIPT's directory on sys.path, not
# the repo root.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from darksirens.redshift.selection import (  # noqa: E402
    H0_REF,
    c_sel_gaussian,
    fit_selection_from_mags,
    k_of_z,
    reference_absolute_mags,
)
from darksirens.utils.cosmology import (  # noqa: E402
    Om0Planck,
    Om0PriorLower,
    Om0PriorUpper,
    distance_modulus,
    w0Fiducial,
    w0PriorLower,
    w0PriorUpper,
    waFiducial,
    waPriorLower,
    waPriorUpper,
)

FID = dict(Om0=Om0Planck, w0=w0Fiducial, wa=waFiducial)

#: Prior-box corners the sampler can actually reach (inference/prior.py
#: cosmo_lower/cosmo_upper), plus the fiducial and a few interior points.
CORNERS = [
    ("fiducial", FID["Om0"], FID["w0"], FID["wa"]),
    ("Om0_lo", Om0PriorLower, FID["w0"], FID["wa"]),
    ("Om0_hi", Om0PriorUpper, FID["w0"], FID["wa"]),
    ("w0_lo", FID["Om0"], w0PriorLower, FID["wa"]),
    ("w0_hi", FID["Om0"], w0PriorUpper, FID["wa"]),
    ("wa_lo", FID["Om0"], FID["w0"], waPriorLower),
    ("wa_hi", FID["Om0"], FID["w0"], waPriorUpper),
    ("Om0lo_w0lo_walo", Om0PriorLower, w0PriorLower, waPriorLower),
    ("Om0lo_w0lo_wahi", Om0PriorLower, w0PriorLower, waPriorUpper),
    ("Om0hi_w0hi_walo", Om0PriorUpper, w0PriorUpper, waPriorLower),
    ("Om0hi_w0hi_wahi", Om0PriorUpper, w0PriorUpper, waPriorUpper),
    ("Om0lo_w0hi_wahi", Om0PriorLower, w0PriorUpper, waPriorUpper),
    ("Om0hi_w0lo_walo", Om0PriorUpper, w0PriorLower, waPriorLower),
    # The docstring's quoted comparison points.
    ("docstring_Om0_0.25", 0.25, FID["w0"], FID["wa"]),
    ("docstring_Om0_0.40", 0.40, FID["w0"], FID["wa"]),
    ("docstring_w0_-0.8", FID["Om0"], -0.8, FID["wa"]),
    ("docstring_w0_-1.2", FID["Om0"], -1.2, FID["wa"]),
    # A "realistic" displacement: 1-sigma-ish Planck/DESI moves.
    ("realistic_Om0_+0.02", FID["Om0"] + 0.02, FID["w0"], FID["wa"]),
    ("realistic_w0_-0.9", FID["Om0"], -0.9, FID["wa"]),
    ("realistic_w0_-0.9_wa_-0.5", FID["Om0"], -0.9, -0.5),
]


def dm100(z, Om0, w0, wa):
    """DM(z) at H0 = 100 (the h-scaled convention of the offline fit)."""
    return np.asarray(distance_modulus(np.asarray(z, dtype=float), H0_REF,
                                       Om0, w0, wa), dtype=float)


# ----------------------------------------------------------------- stage dm
def stage_dm(zlo, zhi):
    zg = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    base = dm100(zg, **FID)
    rows = []
    for name, Om0, w0, wa in CORNERS:
        d = dm100(zg, Om0, w0, wa) - base
        rows.append(dict(name=name, Om0=Om0, w0=w0, wa=wa,
                         dDM=dict(zip([f"{z:g}" for z in zg],
                                      [float(x) for x in d]))))
    # docstring cross-check: Om0 0.25 -> 0.40, w0 -0.8 -> -1.2
    chk = {}
    for lab, a, b in (("Om0_0.25_to_0.40",
                       dict(Om0=0.25, w0=-1.0, wa=0.0),
                       dict(Om0=0.40, w0=-1.0, wa=0.0)),
                      ("w0_-0.8_to_-1.2",
                       dict(Om0=Om0Planck, w0=-0.8, wa=0.0),
                       dict(Om0=Om0Planck, w0=-1.2, wa=0.0))):
        for z in (0.05, 0.5):
            chk[f"{lab}@z={z}"] = float(
                dm100(np.array([z]), **b)[0] - dm100(np.array([z]), **a)[0])
    # z-dependent (mean-subtracted over [zlo, zhi]) spread of Delta DM
    zfine = np.linspace(max(zlo, 1e-3), zhi, 400)
    bfine = dm100(zfine, **FID)
    spread = {}
    for name, Om0, w0, wa in CORNERS:
        d = dm100(zfine, Om0, w0, wa) - bfine
        spread[name] = dict(min=float(d.min()), max=float(d.max()),
                            ptp=float(d.ptp()), mean_flat=float(d.mean()))
    return dict(zgrid=[float(z) for z in zg], rows=rows,
                docstring_check=chk, spread_over_zrange=spread,
                zrange=[zlo, zhi])


# -------------------------------------------------------------- stage refit
def synth_catalog(rng, *, ngal, m_lim, M0hat_true, sigma_true, zlo, zhi,
                  kcorr, Om0, w0, wa):
    """Magnitude-limited Gaussian-LF sample drawn at (Om0, w0, wa).

    Redshifts follow the comoving-volume element of the drawing cosmology
    (dV/dz ~ chi^2/E(z), approximated by the tabulated dL through
    ``distance_modulus``-consistent machinery is unnecessary here: the fit is
    exactly independent of the z-shape by the thinning theorem, so any
    reasonable n(z) is legitimate -- what matters is the RANGE and the
    detected z-distribution, which the m_lim truncation then imposes).
    """
    # Oversample in z with a dV/dz-like weight, then apply the magnitude cut.
    z_pool = rng.uniform(zlo, zhi, size=int(20 * ngal))
    w = z_pool ** 2                      # crude but monotone dV/dz proxy
    w = w / w.sum()
    z_pool = rng.choice(z_pool, size=int(20 * ngal), replace=True, p=w)
    M = M0hat_true + sigma_true * rng.standard_normal(z_pool.size)
    dm = dm100(z_pool, Om0, w0, wa)
    kz = k_of_z(z_pool, kcorr, xp=np)
    m = M + dm + (kz if kz is not None else 0.0)
    keep = m <= m_lim
    z_pool, m = z_pool[keep], m[keep]
    if z_pool.size < ngal:
        raise RuntimeError(
            f"only {z_pool.size} galaxies survived the m_lim={m_lim} cut; "
            "raise --m-lim or lower --ngal.")
    idx = rng.choice(z_pool.size, size=ngal, replace=False)
    return m[idx], z_pool[idx]


def stage_refit(args, rng):
    kcorr = tuple(args.kcorr)
    m, z = synth_catalog(rng, ngal=args.ngal, m_lim=args.m_lim,
                         M0hat_true=args.M0hat_true,
                         sigma_true=args.sigma_true,
                         zlo=args.zlo, zhi=args.zhi, kcorr=kcorr, **FID)
    base = fit_selection_from_mags(m, z, args.m_lim, family="gaussian",
                                   k_corr_coeffs=kcorr or None, **FID)
    cov = np.asarray(base.cov, dtype=float)
    sd_M0hat = float(np.sqrt(cov[0, 0]))
    sd_sigma = float(np.sqrt(cov[1, 1]))

    # Fit-sample <Delta DM>: the quantity option (b) would need persisted.
    zfit = z[z >= 0.01]
    base_dm = dm100(zfit, **FID)

    rows = []
    for name, Om0, w0, wa in CORNERS:
        f = fit_selection_from_mags(m, z, args.m_lim, family="gaussian",
                                    k_corr_coeffs=kcorr or None,
                                    Om0=Om0, w0=w0, wa=wa)
        d_dm_mean = float((dm100(zfit, Om0, w0, wa) - base_dm).mean())
        delta = float(f.M0hat - base.M0hat)
        rows.append(dict(
            name=name, Om0=float(Om0), w0=float(w0), wa=float(wa),
            M0hat=float(f.M0hat), sigma_M=float(f.sigma_M),
            delta_M0hat=delta, delta_sigma_M=float(f.sigma_M - base.sigma_M),
            mean_dDM_over_fit_sample=d_dm_mean,
            # if the fit were a plain mean, delta would be exactly -<dDM>
            residual_vs_meanshift=float(delta + d_dm_mean),
            delta_over_laplace_sd=delta / sd_M0hat,
        ))
    return dict(
        n_gal=int(base.n_gal), m_lim=args.m_lim, kcorr=list(kcorr),
        zrange=[args.zlo, args.zhi],
        fiducial=dict(M0hat=float(base.M0hat), sigma_M=float(base.sigma_M),
                      laplace_sd_M0hat=sd_M0hat,
                      laplace_sd_sigma_M=sd_sigma,
                      cov=cov.tolist()),
        rows=rows,
        zfit_summary=dict(n=int(zfit.size), mean=float(zfit.mean()),
                          median=float(np.median(zfit)),
                          p10=float(np.percentile(zfit, 10)),
                          p90=float(np.percentile(zfit, 90))),
    ), (m, z, base)


# --------------------------------------------------------------- stage csel
def stage_csel(args, refit):
    """Turn delta(Theta) into a completeness error and a missing-budget error.

    C_sel(z) is the fraction of galaxies the catalog HAS; 1 - C_sel is the
    out-of-catalog budget the completion table redistributes.  A zero-point
    error delta shifts the curve's argument by -delta/sigma_M.
    """
    base = refit["fiducial"]
    M0hat, sig = base["M0hat"], base["sigma_M"]
    kcorr = tuple(args.kcorr) or None
    zg = np.linspace(max(args.zlo, 1e-3), args.zhi, 200)
    out = []
    for r in refit["rows"]:
        Om0, w0, wa = r["Om0"], r["w0"], r["wa"]
        c_mod = np.asarray(c_sel_gaussian(zg, args.m_lim, M0hat, sig,
                                          70.0, Om0, w0, wa,
                                          k_corr_coeffs=kcorr))
        c_cor = np.asarray(c_sel_gaussian(zg, args.m_lim, r["M0hat"],
                                          r["sigma_M"], 70.0, Om0, w0, wa,
                                          k_corr_coeffs=kcorr))
        dC = c_mod - c_cor
        miss_mod = float(np.trapz(1.0 - c_mod, zg))
        miss_cor = float(np.trapz(1.0 - c_cor, zg))
        j = int(np.argmax(np.abs(dC)))
        out.append(dict(
            name=r["name"], delta_M0hat=r["delta_M0hat"],
            max_abs_dC=float(abs(dC[j])), z_at_max_dC=float(zg[j]),
            C_model_at_max=float(c_mod[j]), C_correct_at_max=float(c_cor[j]),
            missing_budget_model=miss_mod,
            missing_budget_correct=miss_cor,
            missing_budget_frac_error=(
                (miss_mod - miss_cor) / miss_cor if miss_cor > 0 else None),
            mean_C_model=float(c_mod.mean()),
            mean_C_correct=float(c_cor.mean()),
        ))
    return dict(zgrid=[float(z) for z in zg], rows=out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all",
                   choices=["dm", "refit", "csel", "all"])
    p.add_argument("--ngal", type=int, default=200000)
    p.add_argument("--m-lim", dest="m_lim", type=float, default=19.0)
    p.add_argument("--M0hat-true", dest="M0hat_true", type=float, default=-20.5)
    p.add_argument("--sigma-true", dest="sigma_true", type=float, default=0.9)
    p.add_argument("--zlo", type=float, default=0.05)
    p.add_argument("--zhi", type=float, default=0.5)
    p.add_argument("--kcorr", type=float, nargs="*", default=[2.0, -1.0],
                   help="K(z) polynomial coefficients (c1, c2, ...).")
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    res = {"config": vars(a)}
    if a.stage in ("dm", "all"):
        res["dm"] = stage_dm(a.zlo, a.zhi)
    if a.stage in ("refit", "csel", "all"):
        res["refit"], _ = stage_refit(a, rng)
    if a.stage in ("csel", "all"):
        res["csel"] = stage_csel(a, res["refit"])

    txt = json.dumps(res, indent=2, sort_keys=False)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
