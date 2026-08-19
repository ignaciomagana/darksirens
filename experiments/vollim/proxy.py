"""The injection-level containment proxy, and its validation on the real events.

The hierarchical likelihood's selection term must integrate the SAME criterion
that selected the events: detection AND ``V_90`` containment.  The injection set
carries true parameters and ``pdraw`` but no posteriors, so the containment
indicator cannot be evaluated on it directly.  This builds a proxy from the
quantities an injection does have -- its true ``dL`` and sky position -- by
calibrating, on the 259 real events, how far a posterior extends beyond its own
median:

    radial   dL_med * r_hi  <=  dL_cut          r_hi = quantile of
                                                 v90_dl_hi / dL_med
    sky      the disc of the typical V_90 radius around the position is
             entirely covered  (a footprint EROSION, precomputed per radius)

Both use an upper quantile rather than a mean, because the proxy has to
reproduce an ALL-cells-inside criterion; a mean-sized posterior would count
events whose tails leave the volume.

**The proxy is validated, not assumed.**  Applied to the 259 real events -- for
which the true ``V_90`` answer is known -- it yields a confusion matrix, and the
number that matters is the selection-integral bias: the proxy's predicted pass
RATE against the truth's, since ``beta`` depends on the rate and not on which
individual events pass.  A proxy that misclassifies symmetrically is fine; one
that is systematically loose or tight is not, and the report says which.

The proxy is deliberately H0-INDEPENDENT: both the boundary (``H0 = 140``, the
conservative end) and the posterior geometry are fixed data, so the indicator
does not move as ``H0`` is scanned.  That is what lets it be implemented by
FILTERING the injection set once -- dropping an injection is exactly
multiplying its weight by a zero indicator, provided ``pdraw`` and ``Ndraw``
are left untouched, which they are.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import common as C  # noqa: F401

import h5py  # noqa: E402
import healpy as hp  # noqa: E402
import numpy as np  # noqa: E402

from select_events import covered_cells  # noqa: E402


def eroded_map(ok, radius_deg, nside):
    """Cells whose whole disc of ``radius_deg`` is covered (a footprint erosion).

    Computed once per radius over the covered cells only, which is ~10^4
    ``query_disc`` calls rather than one per injection.
    """
    out = np.zeros(ok.size, dtype=bool)
    r = np.radians(radius_deg)
    for p in np.flatnonzero(ok):
        vec = hp.pix2vec(nside, p)
        disc = hp.query_disc(nside, vec, r, inclusive=True)
        out[p] = bool(ok[disc].all())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selected", default="data/selected_events.json")
    ap.add_argument("--quantile", type=float, default=0.90,
                    help="upper quantile of v90_dl_hi/dL_med and of the V_90 "
                         "radius used by the proxy")
    ap.add_argument("--out", default="data/proxy.json")
    a = ap.parse_args(argv)

    sel = json.load(open(a.selected))
    ev = sel["events"]
    dl_cut = float(sel["dl_cut_mpc"])
    nside = int(sel["nside"])

    med = np.array([v["dl_median"] for v in ev.values()])
    hi = np.array([v["v90_dl_hi"] for v in ev.values()])
    area = np.array([v["v90_area_deg2"] for v in ev.values()])
    truth = np.array([v["contained"] for v in ev.values()])
    truth_sky = np.array([v["sky_ok"] for v in ev.values()])
    truth_rad = np.array([v["rad_ok"] for v in ev.values()])

    # Both extents are DISTANCE-DEPENDENT and a single global quantile is
    # useless here: the ratio runs 1.6 at 300-600 Mpc to 2.6 beyond 2 Gpc, and
    # the area from ~300 to ~2400 deg^2 over the same span.  A global q90 would
    # apply a 5,000 deg^2 event's radius to a 60 deg^2 one and erode the whole
    # footprint (measured: 0 of 6,798 cells survive a 16.6 deg erosion).  So
    # fit both in log-log against dL and evaluate per injection at its own.
    lm = np.log(med)
    b_r, a_r = np.polyfit(lm, np.log(hi / med), 1)
    b_a, a_a = np.polyfit(lm, np.log(area), 1)
    # An upper quantile of the RESIDUAL, so the proxy still reproduces an
    # all-cells-inside criterion rather than a median-sized posterior.
    res_r = np.log(hi / med) - (b_r * lm + a_r)
    res_a = np.log(area) - (b_a * lm + a_a)
    q_r = float(np.quantile(res_r, a.quantile))
    q_a = float(np.quantile(res_a, a.quantile))

    def ratio_of(dl):
        return np.exp(b_r * np.log(dl) + a_r + q_r)

    def radius_of(dl):
        return np.degrees(np.sqrt(
            np.exp(b_a * np.log(dl) + a_a + q_a) * (np.pi / 180.0) ** 2
            / np.pi))

    print(f"[proxy] log(v90_dl_hi/dL_med) = {b_r:+.3f} log dL {a_r:+.3f}, "
          f"q{a.quantile:g} residual {q_r:+.3f}")
    print(f"[proxy] log(V90 area/deg^2)   = {b_a:+.3f} log dL {a_a:+.3f}, "
          f"q{a.quantile:g} residual {q_a:+.3f}")
    for dl in (200.0, 400.0, 600.0):
        print(f"[proxy]   at dL={dl:.0f}: extent x{float(ratio_of(dl)):.2f} "
              f"-> {float(dl * ratio_of(dl)):.0f} Mpc, radius "
              f"{float(radius_of(dl)):.1f} deg")

    ok = covered_cells(sel["fp_min"], nside)
    # Erosion depends on the injection's own radius, so precompute a ladder and
    # look each injection up in the nearest bin (rounded UP, so the proxy never
    # claims more coverage than the ladder supports).
    radii = np.array([2.0, 4.0, 6.0, 9.0, 13.0, 18.0, 25.0])
    ero_ladder = {}
    for rr in radii:
        e = eroded_map(ok, float(rr), nside)
        ero_ladder[float(rr)] = e
        print(f"[proxy] erosion {rr:5.1f} deg -> {int(e.sum()):5d} of "
              f"{int(ok.sum())} cells survive "
              f"({100 * e.sum() / max(ok.sum(), 1):5.1f}%)")
    ero = ero_ladder[float(radii[0])]  # placeholder for the artifact below

    # --- validation on the real events -------------------------------------
    with h5py.File(C.GW_259) as f:
        n, ns = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
        ra = f["ra"][...].reshape(n, ns)
        dec = f["dec"][...].reshape(n, ns)
    ra_med = np.array([np.median(x) for x in ra])
    dec_med = np.array([np.median(x) for x in dec])
    pix_med = hp.ang2pix(nside, np.pi / 2 - dec_med, ra_med)
    # --- the PROBABILISTIC proxy ------------------------------------------
    # A deterministic threshold cannot reproduce this criterion and the
    # measurement above says why: V_90 area correlates with dL at only 0.45, so
    # at fixed (dL, position) the containment answer is essentially a draw from
    # the localization-quality distribution.  A quantile proxy therefore either
    # passes almost everything or nothing (q0.9: 0 of 259, against a truth of 1).
    #
    # But ``beta`` does not need the answer for an individual injection -- it
    # needs E[1_contained | theta], a PROBABILITY.  So carry the empirical
    # distribution explicitly: for an injection at ``dL``, rescale each of the
    # 259 real events' (extent ratio, V_90 radius) pairs to that distance by the
    # fitted slopes and count the fraction that would be contained.  That is a
    # Monte Carlo over localization quality rather than a point estimate of it.
    ratio_res = np.exp(res_r)          # the 259 residual factors
    area_res = np.exp(res_a)

    def p_contained(dl, pix):
        dl = np.atleast_1d(np.asarray(dl, dtype=float))
        pix = np.atleast_1d(np.asarray(pix))
        base_r = np.exp(b_r * np.log(dl) + a_r)
        base_a = np.exp(b_a * np.log(dl) + a_a)
        acc = np.zeros(dl.size)
        for rr_res, aa_res in zip(ratio_res, area_res):
            rad_ok = dl * base_r * rr_res <= dl_cut
            if not rad_ok.any():
                continue
            radius = np.degrees(np.sqrt(base_a * aa_res / np.pi
                                        * (np.pi / 180.0) ** 2))
            j = np.searchsorted(radii, radius)
            sky = np.zeros(dl.size, dtype=bool)
            for k, rung in enumerate(radii):
                m = (j == k) & rad_ok
                if m.any():
                    sky[m] = ero_ladder[float(rung)][pix[m]]
            acc += (rad_ok & sky).astype(float)
        return acc / ratio_res.size

    def sky_pass(dl, pix):
        """Per-object erosion lookup, radius rounded UP to the next rung."""
        rr = radius_of(np.asarray(dl, dtype=float))
        out = np.zeros(np.shape(pix), dtype=bool)
        idx = np.searchsorted(radii, rr)
        for j, rung in enumerate(radii):
            m = idx == j
            if m.any():
                out[m] = ero_ladder[float(rung)][np.asarray(pix)[m]]
        # radii above the ladder's top never pass
        out[idx >= radii.size] = False
        return out

    pred_rad = med * ratio_of(med) <= dl_cut
    pred_sky = sky_pass(med, pix_med)
    pred = pred_rad & pred_sky

    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    tn = int((~pred & ~truth).sum())
    rate_pred = float(pred.mean())
    rate_true = float(truth.mean())
    print(f"\n[validate] on {n} real events, proxy vs the true V_90 criterion:")
    print(f"    TP {tp}   FP {fp}   FN {fn}   TN {tn}")
    print(f"    pass rate: proxy {rate_pred:.4f} ({pred.sum()} events) vs "
          f"truth {rate_true:.4f} ({truth.sum()} events)")
    print(f"    rate ratio (the selection-integral bias) = "
          f"{rate_pred / rate_true if rate_true else float('nan'):.3f}")
    print(f"    radial only: proxy {int(pred_rad.sum())} vs truth "
          f"{int(truth_rad.sum())};  sky only: proxy {int(pred_sky.sum())} vs "
          f"truth {int(truth_sky.sum())}")

    p_ev = p_contained(med, pix_med)
    exp_n = float(p_ev.sum())
    print(f"\n[validate] PROBABILISTIC proxy on the same 259 events:")
    print(f"    expected contained = {exp_n:.2f} events against an observed "
          f"{int(truth.sum())}")
    print(f"    rate ratio = "
          f"{exp_n / max(truth.sum(), 1):.3f}   (1.0 = an unbiased selection "
          f"integral)")
    print(f"    P: max {p_ev.max():.4f}, mean {p_ev.mean():.5f}, "
          f"{int((p_ev > 0).sum())} events with P > 0")
    # Does the probability rank the events the truth actually selected?
    if truth.any():
        print(f"    P at the selected event(s): "
              f"{[round(float(x), 4) for x in p_ev[truth]]}, "
              f"percentile {100 * (p_ev < p_ev[truth].min()).mean():.1f}")
    out_prob = dict(expected_contained=exp_n, observed_contained=int(truth.sum()),
                    rate_ratio=float(exp_n / max(truth.sum(), 1)),
                    p_max=float(p_ev.max()), p_mean=float(p_ev.mean()),
                    n_p_positive=int((p_ev > 0).sum()))

    dl_grid = np.linspace(50.0, 1200.0, 200)
    passes = dl_grid * ratio_of(dl_grid) <= dl_cut
    dl_inj_cut = float(dl_grid[passes].max()) if passes.any() else 0.0
    out = dict(fit_ratio=[float(b_r), float(a_r), q_r],
               fit_area=[float(b_a), float(a_a), q_a],
               radii_ladder=[float(x) for x in radii],
               quantile=a.quantile,
               dl_cut_mpc=dl_cut, dl_inj_cut_mpc=dl_inj_cut, nside=nside,
               fp_min=sel["fp_min"],
               n_cells_footprint=int(ok.sum()),
               n_cells_eroded={f"{r:g}": int(e.sum())
                               for r, e in ero_ladder.items()},
               validation=dict(TP=tp, FP=fp, FN=fn, TN=tn,
                               rate_proxy=rate_pred, rate_truth=rate_true,
                               rate_ratio=(rate_pred / rate_true
                                           if rate_true else None),
                               n_events=n,
                               radial_proxy=int(pred_rad.sum()),
                               radial_truth=int(truth_rad.sum()),
                               sky_proxy=int(pred_sky.sum()),
                               sky_truth=int(truth_sky.sum())),
               probabilistic=out_prob)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    np.savez(Path(a.out).with_suffix(".eroded.npz"),
             radii=radii, **{f"r{r:g}": e for r, e in ero_ladder.items()})
    print(f"[wrote] {a.out}")
    print("PROXY_DONE", flush=True)


if __name__ == "__main__":
    main()
