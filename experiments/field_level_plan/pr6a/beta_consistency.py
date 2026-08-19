"""Do the events' redshift distribution and the injections' agree?  DAG rule 3.

Reading the generator settles the parts of rule 3 that are about the STATISTIC:
events threshold `obs_rho` from `_measure`, injections threshold `rho_opt +
N(0, sigma_rho)` in `_make_selection_kernel`, and both use the identical
`snr_ref (mc_det/30)^(5/6) (1000/dl)` with the same `sigma_rho` and the same
threshold.  Both also carry the `(1+z)^(gamma-1)` rate factor -- events by
rejection, injections inside `pdraw`.  So the detection rule is shared.

What reading does NOT settle is the SOURCE distribution.  Events are drawn from
CATALOG HOSTS (a discrete, clustered, depth-limited set); injections are drawn
from a smooth `dV_c/dz`.  The likelihood's `mu(theta)` is the injections'
integral, so if the two disagree in the redshift range the events occupy, every
posterior is shifted the same way -- with no realization-to-realization scatter,
which is exactly the observed signature (a bias everywhere, dispersion fine at
fixed catalog).

The test is a direct comparison of the DETECTED redshift distributions, weighted
as the likelihood weights them:

* events: the detected hosts' true `z`, pooled over realizations;
* injections: detected injections, weighted by `1/pdraw` -- the same weight
  `mu` uses, so this IS the selection integral's own view of `p(z | detected)`.

A KS/mean comparison of those two is a direct check on `mu`'s support.  If they
disagree, the sign of the disagreement predicts the sign of the `H0` bias: `mu`
over-weighting HIGH `z` means the analysis expects more distant detections than
the mock produced, and the fit compensates by raising `H0`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy import stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", required=True,
                   help="a tier_c data tree (c000, c001, ... realizations)")
    ap.add_argument("--pattern", default="c{:03d}")
    ap.add_argument("--n-real", type=int, default=24)
    ap.add_argument("--injections", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    # --- events: pooled detected host redshifts -----------------------------
    z_ev, snr_ev = [], []
    for k in range(a.n_real):
        p = Path(a.tree) / a.pattern.format(k) / "gw_events.h5"
        if not p.exists():
            continue
        with h5py.File(p) as f:
            z_ev.append(np.asarray(f["truth"]["z"][...]))
            snr_ev.append(np.asarray(f["truth"]["snr"][...]))
    z_ev = np.concatenate(z_ev)
    snr_ev = np.concatenate(snr_ev)
    print(f"[events] {z_ev.size} detected events over {a.n_real} realizations")

    # --- injections: detected, weighted by 1/pdraw --------------------------
    with h5py.File(a.injections) as f:
        # The mock injection set stores dL, not redshift.  Invert at the TRUTH
        # cosmology, which is what the likelihood does at H0_true and therefore
        # the right frame for asking "does mu's support match the events'".
        dl_in = np.asarray(f["dL"][...])
        pdraw = np.asarray(f["pdraw"][...])
    import world16 as W16
    from darksirens.utils.cosmology import dL_of_z
    zg = np.linspace(0.0, 3.0, 4001)
    dlg = np.asarray([float(dL_of_z(z, W16.H0_TRUE, W16.OM0)) for z in zg])
    z_in = np.interp(dl_in, dlg, zg)
    w = 1.0 / pdraw
    w = w / w.sum()
    print(f"[inj] {z_in.size} detected injections, Neff = "
          f"{1.0 / np.square(w).sum():.1f}")

    def wq(x, ww, t):
        o = np.argsort(x)
        c = np.cumsum(ww[o])
        return float(np.interp(t, c, x[o]))

    ev_q = [float(np.quantile(z_ev, t)) for t in (0.1, 0.25, 0.5, 0.75, 0.9)]
    in_q = [wq(z_in, w, t) for t in (0.1, 0.25, 0.5, 0.75, 0.9)]
    ev_mean = float(z_ev.mean())
    in_mean = float((w * z_in).sum())

    # Weighted KS: compare the events' ECDF against the injections' weighted one
    grid = np.linspace(0.0, max(z_ev.max(), 0.5), 512)
    o = np.argsort(z_in)
    cdf_in = np.interp(grid, z_in[o], np.cumsum(w[o]))
    cdf_ev = np.searchsorted(np.sort(z_ev), grid, side="right") / z_ev.size
    ks = float(np.abs(cdf_in - cdf_ev).max())
    # a KS p-value at the events' effective n (injections are far better sampled)
    p_ks = float(stats.kstwo.sf(ks, z_ev.size)) if ks > 0 else 1.0

    print(f"\nDETECTED REDSHIFT DISTRIBUTIONS (the selection integral's support):")
    print(f"  {'quantile':>10} {'events':>10} {'injections':>12}")
    for t, e, i in zip((0.1, 0.25, 0.5, 0.75, 0.9), ev_q, in_q):
        print(f"  {t:>10.2f} {e:>10.4f} {i:>12.4f}")
    print(f"  {'mean':>10} {ev_mean:>10.4f} {in_mean:>12.4f}   "
          f"(injections {'HIGHER' if in_mean > ev_mean else 'LOWER'} by "
          f"{100 * (in_mean / ev_mean - 1):+.1f}%)")
    print(f"  KS distance = {ks:.4f}   p = {p_ks:.3e}")
    print("  injections weighted HIGH => mu expects more distant detections "
          "than the mock made => the fit raises H0.")

    out = dict(n_events=int(z_ev.size), n_inj=int(z_in.size),
               ev_quantiles=ev_q, inj_quantiles=in_q,
               ev_mean=ev_mean, inj_mean=in_mean,
               rel_mean_shift=float(in_mean / ev_mean - 1.0),
               ks=ks, ks_p=p_ks,
               ev_snr_median=float(np.median(snr_ev)))
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[wrote] {a.out}")
    print("BETA_CONSISTENCY_DONE", flush=True)


if __name__ == "__main__":
    main()
