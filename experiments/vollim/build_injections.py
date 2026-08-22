"""Reweight the injection set by the containment probability.

The hierarchical selection term is a Monte Carlo sum over detected injections,

    mu(theta)  =  (1/N_draw) Sum_i  w(theta_i; theta) / pdraw_i ,

so folding in a second selection stage means multiplying each term by that
injection's containment probability.  Dividing ``pdraw`` by ``P`` does exactly
that and nothing else::

    w_i / (pdraw_i / P_i)  ==  P_i * w_i / pdraw_i

``N_draw`` is left untouched -- it counts DRAWS, and the containment stage does
not change how many were drawn -- and injections with ``P = 0`` are removed,
which is the same statement as a zero weight.

This is exact, not an approximation, given ``P``.  What is approximate is ``P``
itself, and ``proxy.py`` reports its validation: the expected contained count
against the observed one.  That validation carries the weight of a SINGLE
observed event, so it constrains the selection function to a factor of a few and
no better -- which is recorded here because it propagates directly into every
number this line produces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import common as C  # noqa: F401

import h5py  # noqa: E402
import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proxy", default="data/proxy.json")
    ap.add_argument("--selected", default="data/selected_events.json")
    ap.add_argument("--out", default="data/injections_contained.h5")
    ap.add_argument("--chunk", type=int, default=100_000)
    a = ap.parse_args(argv)

    pr = json.load(open(a.proxy))
    sel = json.load(open(a.selected))
    nside = int(pr["nside"])
    dl_cut = float(pr["dl_cut_mpc"])
    b_r, a_r, q_r = pr["fit_ratio"]
    b_a, a_a, q_a = pr["fit_area"]
    radii = np.array(pr["radii_ladder"])
    ero = np.load(Path(a.proxy).with_suffix(".eroded.npz"))
    ladder = {float(r): ero[f"r{r:g}"] for r in radii}

    # the empirical localization-quality residuals, from the 259 real events
    ev = sel["events"]
    med = np.array([v["dl_median"] for v in ev.values()])
    hi = np.array([v["v90_dl_hi"] for v in ev.values()])
    area = np.array([v["v90_area_deg2"] for v in ev.values()])
    lm = np.log(med)
    ratio_res = np.exp(np.log(hi / med) - (b_r * lm + a_r))
    area_res = np.exp(np.log(area) - (b_a * lm + a_a))

    def p_contained(dl, pix):
        base_r = np.exp(b_r * np.log(dl) + a_r)
        base_a = np.exp(b_a * np.log(dl) + a_a)
        acc = np.zeros(dl.size)
        for rr, aa in zip(ratio_res, area_res):
            rad_ok = dl * base_r * rr <= dl_cut
            if not rad_ok.any():
                continue
            radius = np.degrees(np.sqrt(base_a * aa / np.pi
                                        * (np.pi / 180.0) ** 2))
            j = np.searchsorted(radii, radius)
            sky = np.zeros(dl.size, dtype=bool)
            for k, rung in enumerate(radii):
                m = (j == k) & rad_ok
                if m.any():
                    sky[m] = ladder[float(rung)][pix[m]]
            acc += (rad_ok & sky).astype(float)
        return acc / ratio_res.size

    with h5py.File(C.INJ_PLAIN) as f:
        n = int(f["dL"].shape[0])
        dL = f["dL"][...]
        ra = f["ra"][...]
        dec = f["dec"][...]
        attrs = dict(f.attrs)
        keys = [k for k in f if isinstance(f[k], h5py.Dataset)]
    print(f"[inj] {n:,} detected injections, Ndraw={attrs.get('ndraw')}")

    pix = hp.ang2pix(nside, np.pi / 2 - dec, ra)
    P = np.zeros(n)
    for lo in range(0, n, a.chunk):
        s = slice(lo, min(lo + a.chunk, n))
        P[s] = p_contained(dL[s], pix[s])
        print(f"  [{lo + a.chunk:>9,}/{n:,}] running mean P = "
              f"{P[:min(lo + a.chunk, n)].mean():.5f}", flush=True)

    keep = P > 0.0
    print(f"\n[inj] P > 0 for {int(keep.sum()):,} of {n:,} injections "
          f"({100 * keep.mean():.3f}%)")
    print(f"[inj] mean P over all = {P.mean():.6f}; over the kept = "
          f"{P[keep].mean():.4f}")
    print(f"[inj] the containment stage keeps an effective "
          f"{P.sum():.1f} injections of {n:,} -- a factor "
          f"{n / max(P.sum(), 1e-9):,.0f} reduction in the selection integral")

    with h5py.File(C.INJ_PLAIN) as fin, h5py.File(a.out, "w") as fout:
        for k, v in attrs.items():
            fout.attrs[k] = v
        fout.attrs["vollim_containment_applied"] = True
        fout.attrs["vollim_dl_cut_mpc"] = dl_cut
        fout.attrs["vollim_p_sum"] = float(P.sum())
        fout.attrs["vollim_n_kept"] = int(keep.sum())
        for k in keys:
            arr = fin[k][...][keep]
            if k == "pdraw":
                arr = arr / P[keep]
            fout.create_dataset(k, data=arr)
    inv = 1.0 / (np.asarray(h5py.File(a.out)["pdraw"][...]))
    neff = float(inv.sum() ** 2 / np.square(inv).sum())
    print(f"[inj] wrote {a.out}")
    print(f"[inj] selection Neff (1/pdraw moments) = {neff:.1f} against "
          f"Vitale's 5 N_obs floor of {5 * sel['n_selected']}")
    json.dump(dict(n_in=n, n_kept=int(keep.sum()), p_sum=float(P.sum()),
                   p_mean=float(P.mean()), neff=neff,
                   reduction=float(n / max(P.sum(), 1e-9))),
              open(Path(a.out).with_suffix(".stats.json"), "w"), indent=1)
    print("BUILD_INJ_DONE", flush=True)


if __name__ == "__main__":
    main()
