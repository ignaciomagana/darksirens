"""dN/dz posterior-predictive gate: observed counts vs the model at POSTERIOR theta.

The Stage A closure compared the catalog's observed dN/dz against
``n0 * C_sel * dV/dz`` at the CALIBRATION point (theta_hat, Planck H0) --
within +/-7% with a coherent LSS dip at z ~ 0.11-0.19.  This script runs the
same comparison in POSTERIOR-PREDICTIVE form: the model curve is evaluated at
draws from the sampled-theta nested-sampling posterior (H0, M0hat, sigma_M --
run_ns_sampled_theta.sh), so the gate asks whether the completeness budget the
INFERENCE actually converged to still prices the catalog's counts.

Model per posterior draw s (op-for-op the likelihood's budget ingredients):

    dN_model(z | s) = n0 * f_sky_occ * 4pi * dV_c/dz(z; H0_s, Om0)
                      * (1+z)^delta * C_sel(z; m_lim, M0hat_s, sigma_M_s, K)

with (n0, delta, Om0, m_lim, K) the run's FIXED values and (H0_s, M0hat_s,
sigma_M_s) the draw.  C_sel is H0-invariant by construction (the h-scaled
firewall), so the band width in the C_sel factor comes from theta alone; the
dV factor carries the H0 posterior.

Outputs (``data/ppd_dndz.json`` + ``figures/ppd_dndz.png``):
  - per-bin observed dN/dz vs the 16/50/84 and 2.5/97.5 posterior-predictive
    band of dN_model;
  - per-bin band residual (obs - median) / (68% half-width);
  - the gate numbers: max |band residual| at z <= 0.25 and the fraction of
    bins outside the 95% band (Poisson counting error is negligible at these
    counts; the band is MODEL uncertainty only, so residuals carry the same
    LSS wiggles the Stage A closure showed -- the gate is about the ENVELOPE,
    not bin-level chi^2).

Usage:
    python ppd_dndz.py --run data/ns_sampled_theta            # the gate
    python ppd_dndz.py --theta-hat                            # machinery check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import common as C  # noqa: F401  (pins DARKSIRENS_ZMAX; must be first)

import jax.numpy as jnp  # noqa: E402

from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.redshift.selection import (  # noqa: E402
    c_sel_gaussian,
    load_selection_fit_json,
)
from darksirens.utils.cosmology import H0Planck, Om0Planck, dV_of_z  # noqa: E402

N_DRAWS = 256           # thinned posterior draws for the band
N_BINS = 40             # same binning as diagnose_stage_a
Z_LO = 0.02


def _observed_dndz(z, w, z_depth):
    edges = np.linspace(Z_LO, z_depth, N_BINS + 1)
    hist, _ = np.histogram(z, bins=edges, weights=w)
    zc = 0.5 * (edges[:-1] + edges[1:])
    return zc, hist / np.diff(edges)


def _model_dndz(zc, H0, M0hat, sigma_M, *, m_lim, coeffs, n0, f_sky, delta):
    zf = np.asarray(zgrid)
    dv = 4.0 * np.pi * np.asarray(dV_of_z(jnp.asarray(zf), H0, Om0Planck))
    csel = np.asarray(c_sel_gaussian(
        jnp.asarray(zf), m_lim, M0hat, sigma_M, H0, Om0Planck,
        k_corr_coeffs=coeffs))
    curve = n0 * f_sky * dv * (1.0 + zf) ** delta * np.clip(csel, 0.0, 1.0)
    return np.interp(zc, zf, curve)


def _posterior_draws(run_dir, n_draws, rng):
    """(H0, M0hat, sigma_M) columns from the run dir's equal-weight samples."""
    run_dir = Path(run_dir)
    samples = np.load(run_dir / "samples.npy")
    labels = json.load(open(run_dir / "settings.json")).get(
        "expected_sampled_labels")
    if labels is None or len(labels) != samples.shape[1]:
        raise SystemExit(
            f"{run_dir}/settings.json carries no usable "
            f"expected_sampled_labels for the {samples.shape} samples array; "
            "the column mapping cannot be guessed.")
    need = ("H0", "M0hat", "sigma_M")
    missing = [n for n in need if n not in labels]
    if missing:
        raise SystemExit(
            f"run {run_dir} did not sample {missing} (labels: {labels}); "
            "this gate is for the sampled-theta configuration.")
    idx = rng.choice(samples.shape[0], size=min(n_draws, samples.shape[0]),
                     replace=False)
    cols = {n: samples[idx, labels.index(n)] for n in need}
    return cols["H0"], cols["M0hat"], cols["sigma_M"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default=None,
                   help="NS run directory (samples.npy + settings.json).")
    p.add_argument("--theta-hat", action="store_true",
                   help="Machinery check: a zero-width 'posterior' at "
                        "theta_hat + Planck H0 (reproduces the Stage A "
                        "closure).")
    p.add_argument("--out-json", default="data/ppd_dndz.json")
    p.add_argument("--out-fig", default="figures/ppd_dndz.png")
    args = p.parse_args(argv)
    if (args.run is None) == (not args.theta_hat):
        raise SystemExit("pass exactly one of --run RUNDIR / --theta-hat")

    sel = load_selection_fit_json(C.DATA_DIR / "selection_fit_union.json")
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    coeffs = tuple(sel["k_corr_coeffs"])
    fixed = dict(m_lim=float(sel["m_lim"]), coeffs=coeffs,
                 n0=float(cal["n0_true_Mpc3"]),
                 f_sky=float(cal["f_sky_occupied"]),
                 delta=float(cal["delta"]))

    with h5py.File(C.DATA_DIR / "desi_union_raw.h5", "r") as f:
        z_gal = f["Z"][...]
        w_gal = f["WEIGHT"][...]
    zc, dN_obs = _observed_dndz(z_gal, w_gal, float(C.Z_DEPTH))

    rng = np.random.default_rng(20260809)
    if args.theta_hat:
        H0s = np.array([float(H0Planck)])
        M0s = np.array([float(sel["M0hat"])])
        sMs = np.array([float(sel["sigma_M"])])
        source = "theta_hat (machinery check)"
    else:
        H0s, M0s, sMs = _posterior_draws(args.run, N_DRAWS, rng)
        source = str(args.run)

    band = np.stack([
        _model_dndz(zc, float(h), float(m0), float(sm), **fixed)
        for h, m0, sm in zip(H0s, M0s, sMs)
    ])
    q = {lev: np.percentile(band, lev, axis=0)
         for lev in (2.5, 16.0, 50.0, 84.0, 97.5)}
    half68 = 0.5 * (q[84.0] - q[16.0])
    # The GATE range is [0.05, 0.25]: the delta-fit range's lower edge (the
    # calibration models the shape there) x the Stage A z<=0.25 convention.
    # z < 0.05 is reported SEPARATELY, not folded into the gate: it holds
    # < 2% of the budget volume and carries the local large-scale structure
    # (a coherent +20-40% count excess) that the smooth n0*(1+z)^delta
    # budget never modelled -- Q's job, not the gate's.
    in_gate = (zc >= 0.05) & (zc <= 0.25)
    low_z = zc < 0.05
    if band.shape[0] == 1:
        # Degenerate single-draw band: report plain fractional residuals.
        resid = dN_obs / q[50.0] - 1.0
        gate = {"mode": "fractional",
                "max_abs_resid_gate_range":
                    float(np.abs(resid[in_gate]).max()),
                "max_abs_resid_z_lt_005":
                    float(np.abs(resid[low_z]).max())}
    else:
        resid = (dN_obs - q[50.0]) / np.where(half68 > 0, half68, np.inf)
        frac = dN_obs / q[50.0] - 1.0
        outside95 = (dN_obs < q[2.5]) | (dN_obs > q[97.5])
        gate = {"mode": "band",
                "max_abs_band_resid_gate_range":
                    float(np.abs(resid[in_gate]).max()),
                "max_abs_frac_resid_gate_range":
                    float(np.abs(frac[in_gate]).max()),
                "max_abs_frac_resid_z_lt_005":
                    float(np.abs(frac[low_z]).max()),
                "frac_gate_bins_outside_95":
                    float(outside95[in_gate].mean()),
                "n_draws": int(band.shape[0])}

    out = {"source": source, "z": zc.tolist(), "dN_obs": dN_obs.tolist(),
           "model_q": {str(k): v.tolist() for k, v in q.items()},
           "band_resid": resid.tolist(), "gate": gate,
           "fixed": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in fixed.items()},
           "posterior_summary": {
               "H0": [float(np.median(H0s)), float(np.std(H0s))],
               "M0hat": [float(np.median(M0s)), float(np.std(M0s))],
               "sigma_M": [float(np.median(sMs)), float(np.std(sMs))]}}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1)
    C.write_provenance(Path(args.out_json), {"source": source, "gate": gate})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.06})
    if band.shape[0] > 1:
        ax.fill_between(zc, q[2.5], q[97.5], alpha=0.18, color="C0",
                        label="model 95% (posterior theta)")
        ax.fill_between(zc, q[16.0], q[84.0], alpha=0.35, color="C0",
                        label="model 68%")
    ax.plot(zc, q[50.0], color="C0", lw=1.2, label="model median")
    ax.plot(zc, dN_obs, "k.", ms=4, label="observed dN/dz")
    ax.set_ylabel("dN/dz  [counts / unit z]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"dN/dz posterior predictive — {source}", fontsize=9)
    axr.axhline(0.0, color="0.6", lw=0.8)
    if band.shape[0] > 1:
        axr.axhspan(-1, 1, color="C0", alpha=0.15)
        axr.set_ylabel("(obs − med) / 68% half-width")
    else:
        axr.set_ylabel("obs/model − 1")
    axr.plot(zc, resid, "k.", ms=4)
    axr.set_xlabel("z")
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=160, bbox_inches="tight")
    print(f"[ppd_dndz] source: {source}")
    print(f"[ppd_dndz] gate: {gate}")
    print(f"[ppd_dndz] wrote {args.out_json} + {args.out_fig}")


if __name__ == "__main__":
    main()
