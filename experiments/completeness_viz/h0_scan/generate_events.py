#!/usr/bin/env python3
"""DAG-consistent GW dark-siren events hosted by the clustered closure mock.

Hosts are drawn uniformly over the COMPLETE catalog rows of
``../output/truth.h5`` (hosts ∝ galaxies — exactly the count-odds weighting the
assembled redshift prior assumes), and everything downstream reuses
``scripts/mock_dark_sirens/generate_mock_data.py`` machinery unchanged:

* detection cuts on the OBSERVED SNR (noise draw included),
* the PE posterior conditions on the SAME recorded noise realization,
* the selection injections use the identical statistic + threshold and the
  ``population+uniform`` proposal (10% uniform floor) so the injection set
  stays reweightable across the trial-H0 grid without coverage collapse.

The GW horizon is placed well inside the catalog depth (``--snr-ref``): the
model's volumetric missing branch beyond z_depth then contributes negligibly,
so the model universe and the generative universe agree where detections live.

Outputs (all under --outdir):
  gw_events.h5           gwcat-1.0 PE file (9 datasets, mock_data=True)
  gw_selection.h5        gwcat-selection-1.0 injections
  catalog_complete.h5    load_survey pixelated file of the COMPLETE catalog at
                         TRUE z (machinery-closure gate: prior == truth)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: F401
from common import load_dark_mock_module
from darksirens.utils.cosmology import H0Planck, Om0Planck, w0Fiducial, waFiducial


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="data")
    p.add_argument("--truth", default="../output/truth.h5")
    p.add_argument("--nobs", type=int, default=300)
    p.add_argument("--nsamp", type=int, default=512)
    p.add_argument("--ndraw", type=int, default=400_000)
    p.add_argument("--batch-size", type=int, default=100_000)
    p.add_argument("--snr-ref", type=float, default=10.0,
                   help="SNR at Mc=30, dL=1000 Mpc; sets the GW horizon. 10 "
                        "puts the heaviest binaries' horizon at z~0.44 < 0.5.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--zmax-grids", type=float, default=1.0,
                   help="Cosmology-grid reach for the injection proposal; "
                        "generous so trial-H0 reweighting stays covered.")
    p.add_argument("--complete-dz", type=float, default=0.003,
                   help="Kernel width for the complete-catalog gate file.")
    return p.parse_args(argv)


def _pixelate(z, ra, dec, nside, dz_val):
    """load_survey-schema padded arrays (float64, pad z=100 dz=1 w=0)."""
    import healpy as hp
    pix = hp.ang2pix(nside, np.pi / 2.0 - dec, ra)
    npix = hp.nside2npix(nside)
    order = np.argsort(pix, kind="stable")
    pix_s, z_s = pix[order], z[order]
    counts = np.bincount(pix_s, minlength=npix)
    maxg = max(1, int(counts.max()))
    zg = np.full((npix, maxg), 100.0)
    dzg = np.ones((npix, maxg))
    wg = np.zeros((npix, maxg))
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for p in np.nonzero(counts)[0]:
        s, n = starts[p], counts[p]
        zg[p, :n] = z_s[s:s + n]
        dzg[p, :n] = dz_val
        wg[p, :n] = 1.0
    return zg, dzg, wg, counts.astype(np.int32)


def main(argv=None):
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mock = load_dark_mock_module()

    with h5py.File(args.truth, "r") as f:
        cat = {k: np.asarray(f["catalog"][k]) for k in ("z", "ra", "dec")}
        attrs = dict(f.attrs)
    zmax_cat = float(attrs["zmax"])
    nside = int(attrs["nside"])

    cosmo = mock._build_cosmology(H0Planck, Om0Planck, w0Fiducial, waFiducial)
    grids = mock._cosmology_grids(cosmo, args.zmax_grids)
    pop = mock.PopulationConfig()
    meas = mock.MeasurementConfig(snr_ref=args.snr_ref)

    # ---- events: hosts from the complete clustered catalog ----------------
    truth_ev, _ = mock._draw_events_until_detected(
        rng, args.nobs, cat, grids, pop, meas.snr_threshold, meas=meas)
    zq = np.percentile(truth_ev["z"], [5, 50, 95, 100])
    print(f"[events] n={len(truth_ev['z'])}  z quantiles 5/50/95/max = "
          f"{zq[0]:.3f}/{zq[1]:.3f}/{zq[2]:.3f}/{zq[3]:.3f}  "
          f"(catalog depth {zmax_cat})")
    if zq[3] > zmax_cat:
        raise RuntimeError("detected event beyond catalog depth — lower --snr-ref")

    samples, observations = mock._posterior_samples(rng, truth_ev, args.nsamp, meas)
    z_of_dl = np.interp(samples["dL"], grids["dl"], grids["z"])
    m1src = samples["m1det"] / (1.0 + z_of_dl)
    m2src = samples["m2det"] / (1.0 + z_of_dl)

    meta = dict(vars(args), H0=H0Planck, Om0=Om0Planck, w0=w0Fiducial,
                wa=waFiducial, host_catalog=str(args.truth),
                gamma=pop.gamma, generator="h0_scan/generate_events.py")
    with h5py.File(outdir / "gw_events.h5", "w") as f:
        for k in ("ra", "dec", "dL", "m1det", "m2det", "chieff", "p_pe"):
            f.create_dataset(k, data=np.asarray(samples[k], dtype=np.float64))
        f.create_dataset("m1src", data=m1src)
        f.create_dataset("m2src", data=m2src)
        f.attrs.update({
            "format_version": "gwcat-1.0", "nobs": len(truth_ev["z"]),
            "nsamp": args.nsamp, "pe_cosmology_H0": H0Planck,
            "pe_cosmology_Om0": Om0Planck, "chi_eff_in_p_pe": True,
            "chi_eff_amax": 0.99, "mock_data": True,
            "snr_ref": meas.snr_ref, "snr_threshold": meas.snr_threshold,
            "metadata_json": json.dumps(meta),
        })
        g = f.create_group("truth")
        for k, v in truth_ev.items():
            g.create_dataset(k, data=np.asarray(v))

    # ---- selection injections (same statistic, threshold, meas) -----------
    sel = mock._selection_injections(
        rng, args.ndraw, grids, pop, meas.snr_threshold, args.batch_size,
        proposal="population+uniform", meas=meas, verbose=True)
    ndet = len(np.asarray(sel["dL"]))
    print(f"[selection] ndraw={sel.get('ndraw', args.ndraw)} detected={ndet}")
    with h5py.File(outdir / "gw_selection.h5", "w") as f:
        for k in mock.SELECTION_KEYS:
            f.create_dataset(k, data=np.asarray(sel[k], dtype=np.float64))
        f.attrs.update({
            "format_version": "gwcat-selection-1.0",
            "ndraw": int(sel.get("ndraw", args.ndraw)),
            "chi_eff_swap_applied": True, "chi_eff_amax": 0.99,
            "cosmology_H0": H0Planck, "cosmology_Om0": Om0Planck,
            "selection_proposal": "population+uniform",
            "metadata_json": json.dumps(meta),
        })

    # ---- complete-catalog gate file (TRUE z, all galaxies) -----------------
    zg, dzg, wg, ng = _pixelate(cat["z"], cat["ra"], cat["dec"], nside,
                                args.complete_dz)
    with h5py.File(outdir / "catalog_complete.h5", "w") as f:
        f.create_dataset("zgals", data=zg)
        f.create_dataset("dzgals", data=dzg)
        f.create_dataset("wgals", data=wg)
        f.create_dataset("ngals", data=ng)
        f.attrs["nside"] = nside
        f.attrs["z_depth"] = zmax_cat
    print(f"[gate] complete catalog pixelated: {ng.sum()} galaxies, "
          f"maxgals={zg.shape[1]}")


if __name__ == "__main__":
    main()
