"""Select the events whose FULL V_90 lies inside LOA+LS, and freeze the list.

The criterion, stated so it can be reproduced on injections later:

    an event is SELECTED when every cell of its 90% credible localization
    volume is (a) inside the survey footprint and (b) closer than
    dL(z = 0.30; H0 = 140).

``V_90`` is built as a sample-based highest-density region on an
(nside-32 sky cell) x (dL bin) grid: rank cells by posterior mass, accumulate
until 90%, and that set IS the volume.  Containment then asks whether the set
is a subset of the survey.

Two choices in that construction are conservative on purpose:

* **the radial edge** uses ``H0 = 140`` (see ``common``), so the cut does not
  move as ``H0`` is scanned;
* **the sky edge** degrades the nside-128 depth map to the nside-32 cells by
  requiring EVERY child pixel to clear ``f_p >= FP_MIN``.  A cell straddling
  the footprint boundary is therefore outside, not partially inside.

The report includes, per event, the containment margin -- how much closer the
event is than the cut, and what fraction of its V_90 cells are covered --
because a list that only just passes is a different object from one that passes
comfortably, and the ``f_p`` scan at the end shows which this is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import common as C  # noqa: F401  (pins DARKSIRENS_ZMAX; must be first)

import h5py  # noqa: E402
import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402


def covered_cells(fp_min, nside_v90=C.NSIDE_V90, nside_map=C.NSIDE_MAP):
    """nside-32 cells every one of whose nside-128 children clears ``fp_min``.

    ``hp.ud_grade`` would AVERAGE the children; the guard we want is the
    MINIMUM, so a cell counts as covered only if its worst child does.
    """
    with h5py.File(C.MTH_MAP) as f:
        masked = np.asarray(f["masked_frac"][...], dtype=np.float64)
        counts = np.asarray(f["counts"][...])
    f_p = np.where(np.isfinite(masked), 1.0 - masked, 0.0)
    ok_fine = (f_p >= fp_min) & (counts > 0)
    ok_nest = hp.reorder(ok_fine.astype(np.int8), r2n=True).astype(bool)
    k = (nside_map // nside_v90) ** 2
    ok_coarse_nest = ok_nest.reshape(-1, k).all(axis=1)
    return hp.reorder(ok_coarse_nest.astype(np.int8), n2r=True).astype(bool)


def v90_cells(ra, dec, dL, nside, credible, n_dl_bins=32):
    """Sample-based highest-density credible region as (pixel, dL-bin) cells.

    The dL binning is per event (its own sample range), so the cell aspect
    ratio adapts to the localization instead of being fixed globally.
    """
    pix = hp.ang2pix(nside, np.pi / 2 - dec, ra)
    lo, hi = float(dL.min()), float(dL.max())
    edges = np.linspace(lo, hi * (1 + 1e-9), n_dl_bins + 1)
    ib = np.clip(np.digitize(dL, edges) - 1, 0, n_dl_bins - 1)
    key = pix.astype(np.int64) * n_dl_bins + ib
    uniq, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts)
    uniq, counts = uniq[order], counts[order]
    cum = np.cumsum(counts) / counts.sum()
    n_keep = int(np.searchsorted(cum, credible) + 1)
    sel = uniq[:n_keep]
    return ((sel // n_dl_bins).astype(np.int64),
            (sel % n_dl_bins).astype(int), edges, float(cum[n_keep - 1]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fp-min", type=float, default=C.FP_MIN)
    p.add_argument("--credible", type=float, default=C.CREDIBLE)
    p.add_argument("--nside", type=int, default=C.NSIDE_V90)
    p.add_argument("--scan", type=float, nargs="*",
                   default=[0.0, 0.25, 0.5, 0.75, 0.9])
    p.add_argument("--out", default="data/selected_events.json")
    a = p.parse_args(argv)

    from darksirens.utils.cosmology import dL_of_z
    dl_cut = float(dL_of_z(C.Z_CUT, max(C.H0_PRIOR), C.OM0))
    assert abs(dl_cut - C.DL_CUT_MPC) < 1.0, (dl_cut, C.DL_CUT_MPC)
    dl_fid = float(dL_of_z(C.Z_CUT, 67.74, C.OM0))
    print(f"[cut] dL(z={C.Z_CUT}; H0={max(C.H0_PRIOR)}, Om0={C.OM0}) = "
          f"{dl_cut:.1f} Mpc")
    print(f"[cut] the same redshift at H0=67.74 is {dl_fid:.1f} Mpc -- the "
          f"conservative choice costs a factor {(dl_fid / dl_cut) ** 3:.1f} "
          f"in volume")

    with h5py.File(C.GW_259) as f:
        n = int(f.attrs["nobs"])
        ns = int(f.attrs["nsamp"])
        names = [x.decode() if isinstance(x, bytes) else str(x)
                 for x in f.attrs["event_names"]]
        ra = f["ra"][...].reshape(n, ns)
        dec = f["dec"][...].reshape(n, ns)
        dL = f["dL"][...].reshape(n, ns)
    print(f"[events] {n} events x {ns} samples")

    apix = hp.nside2pixarea(a.nside, degrees=True)
    # V_90 is f_p-independent, so build it once and test containment per
    # threshold -- 259 events x 5 thresholds otherwise repeats the same
    # decomposition five times.
    v90 = [v90_cells(ra[i], dec[i], dL[i], a.nside, a.credible)
           for i in range(n)]

    rows, selected, scan_counts = None, None, {}
    for thr in sorted(set(list(a.scan) + [a.fp_min])):
        ok = covered_cells(thr, a.nside)
        sel_names, detail = [], {}
        for i in range(n):
            pixels, ibins, edges, mass = v90[i]
            dl_hi = float(edges[ibins.max() + 1])
            dl_lo = float(edges[ibins.min()])
            sky_ok = bool(ok[pixels].all())
            rad_ok = bool(dl_hi <= dl_cut)
            contained = sky_ok and rad_ok
            if contained:
                sel_names.append(names[i])
            detail[names[i]] = dict(
                index=i, contained=contained, sky_ok=sky_ok, rad_ok=rad_ok,
                v90_dl_lo=dl_lo, v90_dl_hi=dl_hi,
                dl_median=float(np.median(dL[i])),
                radial_margin_mpc=float(dl_cut - dl_hi),
                v90_area_deg2=float(np.unique(pixels).size * apix),
                v90_cells=int(pixels.size), v90_mass=float(mass),
                frac_cells_covered=float(ok[pixels].mean()),
                frac_samples_in_footprint=float(
                    ok[hp.ang2pix(a.nside, np.pi / 2 - dec[i], ra[i])].mean()),
                frac_samples_below_cut=float((dL[i] <= dl_cut).mean()))
        scan_counts[f"{thr:g}"] = len(sel_names)
        if abs(thr - a.fp_min) < 1e-12:
            rows, selected = detail, sel_names
        print(f"  f_p >= {thr:<4g} -> {len(sel_names):3d} events selected")

    out = dict(
        dl_cut_mpc=dl_cut, dl_at_fiducial_mpc=dl_fid, z_cut=C.Z_CUT,
        h0_prior=list(C.H0_PRIOR), Om0=C.OM0, credible=a.credible,
        nside=a.nside, fp_min=a.fp_min, n_events_total=n,
        n_selected=len(selected), selected=selected, fp_scan=scan_counts,
        events=rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n[selected] {len(selected)} of {n} events at f_p >= {a.fp_min}")
    print(f"[wrote] {a.out}")
    print("SELECT_DONE", flush=True)


if __name__ == "__main__":
    main()
