#!/usr/bin/env python
"""Build a tabulated selection integral ("beta table") for darksirens_inference.

What this computes
------------------
mu(Lambda) as a Monte-Carlo average over the detected injections::

    mu = (1/Ndraw) sum_i  p_target(d_i | Lambda) / p_draw(d_i)

with the target density carrying ONLY the cosmology-dependent factors::

    log p_target = log p_z(z)  -  log[ d(dL)/dz ]  -  log(1 + z)

i.e. the comoving-volume redshift prior and the (m1src, q, z) -> (m1det, q, dL)
Jacobian.  Two terms present in the full per-event weight are deliberately
ABSENT:

* ``p_pop(m1src, q, chi_eff)`` -- the population is FIXED for a tabulated run,
  so this contributes a constant factor to mu, hence a constant offset in
  log mu, hence a constant shift in ``-N log mu`` that cancels in the
  posterior.  It cannot affect the inferred cosmology.  (It is also very nearly
  cancelled by ``p_draw``, which was drawn from the same population: measured
  N_eff for ``p_pop/p_draw`` is 51,829 of 50,070 detections.)

* ``p_z(z | pix)`` -- the per-pixel galaxy-catalog prior.  mu is a sky- and
  population-averaged detectability normalisation, so where an injection falls
  relative to individual galaxies should average out of it.  Keeping it does
  not add signal, it adds variance: with the per-pixel comb the per-node N_eff
  is 20-95 out of 50,070, and an 11-node H0 table came out with 0.22 nats of
  per-node Monte-Carlo scatter and 0.52 nats of interpolation error against a
  total signal of only 1.04 nats -- a spurious 75% spike at H0 = 72 (job
  1174307).  Dropping it leaves the smooth comoving-volume prior.

Because both dropped terms are cosmology-independent (the first exactly, the
second by the sky-averaging argument), the SHAPE of log mu vs the cosmology is
preserved and only an additive constant is lost -- which the posterior does not
see.

Usage
-----
    python scripts/build_beta_table.py \
        --out beta_table.h5 \
        --gwselection_path injections.h5 \
        --axis 'H0=60,80,50' \
        [--axis 'Om0=0.25,0.35,5'] \
        [--z_horizon 0.5] [--plot beta_vs_H0.png] [--loo]

``--axis LABEL=lo,hi,n`` may be repeated (H0, Om0, w0, wa only).  No inference
CLI is invoked and no galaxy catalog is read: this needs only the injection
file, so it runs in seconds on a CPU.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _parse_axis(spec):
    """``LABEL=lo,hi,n`` -> (label, np.linspace(lo, hi, n))."""
    if "=" not in spec:
        raise SystemExit(f"--axis {spec!r} must look like 'H0=60,80,50'.")
    label, rhs = spec.split("=", 1)
    parts = rhs.split(",")
    if len(parts) != 3:
        raise SystemExit(f"--axis {spec!r} must give lo,hi,n.")
    lo, hi, n = float(parts[0]), float(parts[1]), int(parts[2])
    if not hi > lo:
        raise SystemExit(f"--axis {spec!r}: hi must exceed lo.")
    if n < 2:
        raise SystemExit(f"--axis {spec!r}: n must be >= 2.")
    return label.strip(), np.linspace(lo, hi, n)


def _load_injections(path):
    """(dL, p_draw, Ndraw) from a gwcat selection file."""
    import h5py

    with h5py.File(path, "r") as f:
        dL = np.asarray(f["dL"][:], dtype=float)
        pdraw = np.asarray(f["pdraw"][:], dtype=float)
        ndraw = float(f.attrs["ndraw"])
    good = np.isfinite(dL) & np.isfinite(pdraw) & (pdraw > 0.0)
    return dL[good], pdraw[good], ndraw, int(good.size), int((~good).sum())


def _log_mu_at(dL, pdraw, ndraw, H0, Om0, w0, wa, z_horizon, nz=4000):
    """log mu and N_eff at one cosmology.

    The redshift grid is rebuilt per cosmology because dL(z) is exactly what
    the cosmology changes; everything else is a lookup against it.
    """
    from astropy.cosmology import Flatw0waCDM
    import astropy.units as u

    cos = Flatw0waCDM(H0=H0, Om0=Om0, w0=w0, wa=wa)
    zg = np.linspace(1e-6, max(2.0 * z_horizon, 1.0), nz)
    dLg = cos.luminosity_distance(zg).to(u.Mpc).value
    # dVc/dz [Mpc^3/sr] and d(dL)/dz, both on the grid then interpolated.
    dVdzg = cos.differential_comoving_volume(zg).to(u.Mpc ** 3 / u.sr).value
    ddLg = np.gradient(dLg, zg)

    z = np.interp(dL, dLg, zg, left=np.nan, right=np.nan)
    ok = np.isfinite(z) & (z > 0.0) & (z <= z_horizon)
    if not ok.any():
        return -np.inf, 0.0, 0

    dVdz = np.interp(z[ok], zg, dVdzg)
    ddL = np.interp(z[ok], zg, ddLg)
    # log p_target = log(dVc/dz) - log(ddL/dz) - log(1+z);  w = p_target/p_draw
    log_w = (np.log(dVdz) - np.log(np.maximum(ddL, 1e-300))
             - np.log1p(z[ok]) - np.log(pdraw[ok]))

    m = log_w.max()
    s = np.exp(log_w - m)
    lse = m + np.log(s.sum())
    lse2 = 2.0 * m + np.log((s ** 2).sum())
    log_mu = lse - np.log(ndraw)
    inv_neff = np.exp(lse2 - 2.0 * lse) - 1.0 / ndraw
    neff = (1.0 / inv_neff) if inv_neff > 0 else np.inf
    return float(log_mu), float(neff), int(ok.sum())


def _plot_1d(path, label, nodes, log_mu, log_mu_err):
    """log mu and mu vs the single table axis, with per-node MC error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
    err = np.where(np.isfinite(log_mu_err), log_mu_err, 0.0)
    ax[0].errorbar(nodes, log_mu, yerr=err, marker="o", ms=3, lw=1.2, capsize=2)
    ax[0].set_ylabel(r"$\log\mu$")
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Tabulated selection integral vs {label}")
    mid = log_mu[len(nodes) // 2]
    rel = np.exp(log_mu - mid)
    ax[1].errorbar(nodes, rel, yerr=rel * err, marker="o", ms=3, lw=1.2, capsize=2)
    ax[1].set_ylabel(r"$\mu / \mu(\mathrm{mid})$")
    ax[1].set_xlabel(label)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[build] wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="Output .h5 table path.")
    p.add_argument("--gwselection_path", required=True,
                   help="gwcat selection/injection HDF5 (needs dL, pdraw, attrs ndraw).")
    p.add_argument("--axis", action="append", required=True, metavar="LABEL=lo,hi,n",
                   help="Grid axis; repeatable. H0, Om0, w0, wa only.")
    p.add_argument("--Om0", type=float, default=0.3075,
                   help="Fixed Om0 when it is not an axis (default: %(default)s).")
    p.add_argument("--w0", type=float, default=-1.0,
                   help="Fixed w0 when it is not an axis (default: %(default)s).")
    p.add_argument("--wa", type=float, default=0.0,
                   help="Fixed wa when it is not an axis (default: %(default)s).")
    p.add_argument("--z_horizon", type=float, default=0.5,
                   help="Detection horizon: P_det = 0 beyond it (default: %(default)s).")
    p.add_argument("--plot", default=None, metavar="PNG",
                   help="Write a log mu / mu vs axis plot (1-D grids only).")
    p.add_argument("--loo", action="store_true",
                   help="Estimate the interpolation error by leave-one-out on "
                        "interior nodes.")
    args = p.parse_args()

    from darksirens.likelihood.beta_interp import (
        MU_DEPENDENT_LABELS, BetaTable, save_beta_table,
    )

    axes = [_parse_axis(s) for s in args.axis]
    for lab, _ in axes:
        if lab not in MU_DEPENDENT_LABELS:
            raise SystemExit(
                f"--axis {lab!r} is not a cosmological parameter; choose from "
                f"{MU_DEPENDENT_LABELS}."
            )
    labels = tuple(lab for lab, _ in axes)
    nodes = tuple(nd for _, nd in axes)
    shape = tuple(nd.size for nd in nodes)
    n_nodes = int(np.prod(shape))

    dL, pdraw, ndraw, n_tot, n_bad = _load_injections(args.gwselection_path)
    print(f"[build] injections {dL.size:,} of {n_tot:,} usable"
          + (f" ({n_bad:,} dropped)" if n_bad else "")
          + f"; Ndraw {ndraw:,.0f}", flush=True)
    print(f"[build] grid {dict(zip(labels, shape))} = {n_nodes} nodes; "
          f"z_horizon {args.z_horizon:g}", flush=True)
    print(f"[build] fixed: Om0={args.Om0:g} w0={args.w0:g} wa={args.wa:g} "
          "(overridden per-axis where applicable)", flush=True)

    log_mu = np.full(shape, np.nan)
    log_mu_err = np.full(shape, np.nan)
    for flat in range(n_nodes):
        midx = np.unravel_index(flat, shape)
        vals = {"H0": 70.0, "Om0": args.Om0, "w0": args.w0, "wa": args.wa}
        for k, lab in enumerate(labels):
            vals[lab] = float(nodes[k][midx[k]])
        lm, ne, n_in = _log_mu_at(dL, pdraw, ndraw, vals["H0"], vals["Om0"],
                                  vals["w0"], vals["wa"], args.z_horizon)
        log_mu[midx] = lm
        log_mu_err[midx] = (1.0 / np.sqrt(ne)) if np.isfinite(ne) and ne > 0 else np.inf
        coord = ", ".join(f"{lab}={vals[lab]:g}" for lab in labels)
        print(f"  [{flat + 1}/{n_nodes}] {coord}  log_mu={lm:.6f}  "
              f"Neff={ne:,.0f}  n_in_horizon={n_in:,}", flush=True)

    interp_err = 0.0
    if args.loo and all(s >= 3 for s in shape):
        errs = []
        for flat in range(n_nodes):
            midx = np.unravel_index(flat, shape)
            if any(i == 0 or i == s - 1 for i, s in zip(midx, shape)):
                continue
            for k in range(len(shape)):
                lo_i = list(midx); lo_i[k] -= 1
                hi_i = list(midx); hi_i[k] += 1
                x0, x1 = nodes[k][midx[k] - 1], nodes[k][midx[k] + 1]
                t = (nodes[k][midx[k]] - x0) / (x1 - x0)
                pred = (1 - t) * log_mu[tuple(lo_i)] + t * log_mu[tuple(hi_i)]
                errs.append(abs(pred - log_mu[midx]))
        if errs:
            interp_err = float(np.max(errs))
            print(f"[build] leave-one-out interpolation error: max {interp_err:.6f} "
                  f"nats (median {np.median(errs):.6f}) over {len(errs)} checks",
                  flush=True)
            print("[build] NOTE measured on a grid of HALF the density, so this "
                  "OVERSTATES the delivered table's error -- an upper bound.",
                  flush=True)

    if args.plot and len(shape) == 1:
        _plot_1d(args.plot, labels[0], nodes[0], log_mu, log_mu_err)

    meta = {
        "gwselection_path": args.gwselection_path,
        "z_horizon": args.z_horizon,
        "fixed": {"Om0": args.Om0, "w0": args.w0, "wa": args.wa},
        "target_terms": "dVc/dz, 1/(ddL/dz), 1/(1+z)",
        "dropped_terms": "p_pop (constant, population fixed); "
                         "p_z(z|pix) (sky-averaged out of mu)",
    }
    table = BetaTable(labels, nodes, log_mu, log_mu_err, interp_err, ndraw, meta)
    save_beta_table(args.out, table)
    print(f"\n[build] wrote {args.out}", flush=True)
    print(f"[build] error budget: max per-node MC {np.nanmax(log_mu_err):.6f} nats, "
          f"interpolation {interp_err:.6f} nats", flush=True)
    print(f"[build] log_mu range [{np.nanmin(log_mu):.6f}, {np.nanmax(log_mu):.6f}] "
          f"(spread {np.nanmax(log_mu) - np.nanmin(log_mu):.6f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())