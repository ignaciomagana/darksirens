#!/usr/bin/env python3
"""Visualize the fitted completeness corrections against the injected truth.

Reads ``truth.h5`` and ``fits/*`` produced by ``generate_clustered_mock.py``
and ``fit_completeness.py``; writes PNG diagnostics plus a machine-readable
closure summary (``plots/closure_summary.json``) with soft PASS/FAIL checks.

Color conventions (consistent across every figure):
  truth = black, gp3d = C0 blue, radial = C1 orange, delta_g = C2 green,
  n0-ablation = C3 red, homogeneous = gray dotted.  Fields and residuals use a
  diverging map centered at logQ = 0 (Q = 1 is the neutral, mean-density state).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import common  # noqa: F401  (sets up sys.path / x64)
from common import xyz_to_pix_z
from darksirens.redshift import zgrid

COLORS = {"truth": "k", "q_gp3d": "C0", "q_radial": "C1",
          "delta_g": "C2", "q_radial_n0off": "C3", "homog": "0.45"}
LABELS = {"truth": "truth", "q_gp3d": "Q gp3d", "q_radial": "Q radial",
          "delta_g": r"legacy $1+b\,\delta_g$", "q_radial_n0off": r"Q radial ($n_0$ +0.3 dex)",
          "homog": "homogeneous"}


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="output")
    p.add_argument("--slab-half-thickness", type=float, default=100.0,
                   help="Half-thickness [Mpc] of the galaxy-scatter slab overlay.")
    return p.parse_args(argv)


def _load(outdir):
    d = {}
    with h5py.File(outdir / "truth.h5", "r") as f:
        d["attrs"] = dict(f.attrs)
        d["logq_truth"] = np.asarray(f["logq_truth_on_zgrid"])
        d["cosmo_z"] = np.asarray(f["cosmo_z"])
        d["cosmo_dc"] = np.asarray(f["cosmo_dc"])
        d["cosmo_dl"] = np.asarray(f["cosmo_dl"])
        d["cosmo_dvc_dz"] = np.asarray(f["cosmo_dvc_dz"])
        d["cat"] = {k: np.asarray(f["catalog"][k]) for k in f["catalog"]}
    fits = outdir / "fits"
    d["curves"] = {}
    for p in sorted(fits.glob("curves_*.npz")):
        d["curves"][p.stem[len("curves_"):]] = dict(np.load(p))
    d["tables"] = {}
    from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5
    for name in ("q_radial", "q_gp3d", "q_radial_n0off"):
        path = fits / f"{name}.h5"
        if path.exists():
            loaded = load_lss_completion_hdf5(str(path))
            d["tables"][name] = (np.asarray(loaded["logq_map"]),
                                 None if loaded["logq_members"] is None
                                 else np.asarray(loaded["logq_members"]))
    dg = fits / "delta_g.npz"
    d["delta_g"] = np.load(dg)["delta_g"] if dg.exists() else None
    d["priors"] = {}
    for p in sorted(fits.glob("priors_*.npz")):
        d["priors"][p.stem[len("priors_"):]] = dict(np.load(p))
    d["prior_pixels"] = np.load(fits / "prior_pixels.npy")
    d["obs"] = dict(np.load(fits / "observed_branch.npz"))
    # Homogeneous expected density per pixel on zgrid (NaN beyond the
    # tabulated cosmology range; every consumer masks to z <= z_depth).
    # Homogeneous expected density per pixel on zgrid, from the SAME galaxy
    # measure the fitter uses (n0 · apix · g(z)).
    import healpy as hp
    from common import fiducial_cosmo, fiducial_survey
    from darksirens.redshift.completion import log_galaxy_measure_grid
    n_pix = d["logq_truth"].shape[0]
    apix = hp.nside2pixarea(hp.npix2nside(n_pix))
    n0 = float(d["attrs"]["n0"])
    log_g = np.asarray(log_galaxy_measure_grid(
        fiducial_cosmo(), fiducial_survey(n0)))
    d["dN_exp"] = n0 * apix * np.exp(log_g)
    return d


def _c_true(z, attrs, cosmo_z, cosmo_dl):
    """Analytic selection completeness C_true(z): Gaussian-CDF magnitude cut
    times the logistic z roll-off (the exact generative selection)."""
    from scipy.special import expit
    from scipy.stats import norm
    dl_pc = np.interp(z, cosmo_z, cosmo_dl) * 1.0e6
    dm = 5.0 * np.log10(np.maximum(dl_pc, 10.0) / 10.0)
    mag = norm.cdf((attrs["mag_limit"] - attrs["abs_mag_mean"] - dm) / attrs["abs_mag_sigma"])
    return mag * expit((attrs["z50"] - z) / attrs["width"])


def plot_c_calibration(d, plots, summary):
    """Aggregate kernel-ratio completeness vs the generative selection.

    The matched comparison is aggregate-vs-aggregate: the estimator's
    convention is a KDE-smoothed ratio, so the truth must be pushed through
    the SAME smoothing operator (S @ (C_true·⟨Q⟩·dN_exp)) / (S @ dN_exp).
    Per-pixel C is additionally noisy and clip-asymmetric — shown as a band.
    """
    import matplotlib.pyplot as plt
    z = np.asarray(zgrid)
    zd = float(d["attrs"]["zmax"])
    sl = z <= zd
    C = d["curves"]["homog"]["C_eff"]                        # (n_pix, N_grid)
    occ = d["obs"]["N_obs"] > 0
    lo, hi = np.percentile(C[occ], 16, axis=0), np.percentile(C[occ], 84, axis=0)
    ct = _c_true(z, d["attrs"], d["cosmo_z"], d["cosmo_dl"])
    meanQ = np.exp(d["logq_truth"]).mean(axis=0)             # realization sky-mean

    # Aggregate estimator and matched-smoothed truth (same smoothing operator).
    from darksirens.redshift.completion import smoothing_operator
    S = np.asarray(smoothing_operator())
    dN_exp = d["dN_exp"]                                     # (N_grid,) per pixel
    dN_exp_s = S @ dN_exp
    n_pix = C.shape[0]
    C_agg = d["obs"]["dN_obs_s"].sum(axis=0) / np.maximum(n_pix * dN_exp_s, 1e-300)
    truth_sm = (S @ np.nan_to_num(ct * meanQ * dN_exp)) / np.maximum(dN_exp_s, 1e-300)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True,
                                  height_ratios=[3, 1])
    ax.plot(z[sl], ct[sl], color="0.6", ls=":",
            label=r"analytic selection $C_{\rm true}(z)$")
    ax.plot(z[sl], (ct * meanQ)[sl], color="k",
            label=r"$C_{\rm true}\,\langle Q_{\rm truth}\rangle_{\rm sky}$")
    ax.plot(z[sl], truth_sm[sl], color="k", ls="--",
            label="same, matched-smoothed")
    ax.plot(z[sl], C_agg[sl], color="C0",
            label="fitted kernel-ratio $C$ (aggregate)")
    ax.fill_between(z[sl], lo[sl], hi[sl], color="C0", alpha=0.15,
                    label="per-pixel 16–84%")
    ymax = 1.02 * max(1.0, float(np.nanmax((ct * meanQ)[sl])))
    ax.set(ylabel="completeness", xlim=(0, zd), ylim=(0, ymax))
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0)
    ax.set_title("Kernel-ratio completeness vs generative selection")
    ratio = np.where(truth_sm > 0.05, C_agg / np.maximum(truth_sm, 1e-6), np.nan)
    axr.axhline(1.0, color="k", lw=1)
    axr.plot(z[sl], ratio[sl], color="C0")
    rr = ratio[sl][np.isfinite(ratio[sl])]
    ylim_r = (min(0.95, float(rr.min()) - 0.02), max(1.05, float(rr.max()) + 0.02))
    axr.set(xlabel="z", ylabel="aggregate / truth", ylim=ylim_r)
    fig.savefig(plots / "c_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    m = sl & (truth_sm > 0.1)
    dev = float(np.nanmedian(np.abs(ratio[m] - 1.0)))
    summary["c_calibration_median_abs_dev"] = dev
    summary["c_calibration_pass"] = bool(dev < 0.10)


def plot_dndz_closure(d, plots, summary):
    import matplotlib.pyplot as plt
    z = np.asarray(zgrid)
    zd = float(d["attrs"]["zmax"])
    sl = z <= zd
    # Truth and observed dN/dz from the complete catalog.
    bins = np.linspace(0.0, zd, 64)
    zc = 0.5 * (bins[:-1] + bins[1:])
    h_true, _ = np.histogram(d["cat"]["z"], bins=bins)
    h_obs, _ = np.histogram(d["cat"]["z"][d["cat"]["observed"].astype(bool)], bins=bins)
    dz = np.diff(bins)
    # Observed branch on zgrid: sum of the per-pixel smoothed KDE densities.
    dN_obs_tot = d["obs"]["dN_obs_s"].sum(axis=0)

    scale = 1e-6  # plot in units of 10^6 galaxies per unit z (avoids the
    # offset-text "1e6" colliding with the title)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(zc, h_true / dz * scale, color="k", lw=2,
            label="truth (complete catalog)")
    ax.plot(zc, h_obs / dz * scale, color="0.55", lw=2, ls="--", label="observed")
    n_true_below = int(h_true.sum())
    for name in ("homog", "delta_g", "q_radial", "q_gp3d"):
        if name not in d["curves"]:
            continue
        dN_miss_tot = d["curves"][name]["dN_miss"].sum(axis=0)
        rec = dN_obs_tot + dN_miss_tot
        ax.plot(z[sl], rec[sl] * scale, color=COLORS[name], lw=1.4,
                label=f"observed + missing ({LABELS[name]})")
        n_rec = float(np.trapz(rec[sl], z[sl]))
        summary[f"dndz_recovered_over_truth_{name}"] = n_rec / n_true_below
    ax.set(xlabel="z", ylabel=r"$dN/dz$  [$10^6$]", xlim=(0, zd),
           title="Total-galaxy budget closure")
    ax.legend(fontsize=9, loc="upper left")
    fig.savefig(plots / "dndz_closure.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    ratios = [v for k, v in summary.items()
              if k.startswith("dndz_recovered_over_truth_")]
    summary["dndz_closure_pass"] = bool(
        ratios and min(abs(r - 1.0) for r in ratios) < 0.10)


def plot_q_z_curves(d, plots):
    import matplotlib.pyplot as plt
    z = np.asarray(zgrid)
    zd = float(d["attrs"]["zmax"])
    sl = z <= zd
    pixels = d["prior_pixels"]
    show = pixels[np.unique(np.linspace(0, pixels.size - 1, 6).astype(int))]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True, sharey=True)
    shown = []  # everything plotted, to pick a shared log-scale window
    for ax, p in zip(axes.ravel(), show):
        ax.axhline(1.0, color="0.7", ls=":", lw=1)
        qt = np.exp(d["logq_truth"][p, sl])
        ax.plot(z[sl], qt, color="k", lw=2, label="truth")
        shown.append(qt)
        for name in ("q_radial", "q_gp3d"):
            if name not in d["tables"]:
                continue
            logq, mem = d["tables"][name]
            qf = np.exp(logq[p, sl])
            ax.plot(z[sl], qf, color=COLORS[name], lw=1.4,
                    label=f"{LABELS[name]} MAP")
            shown.append(qf)
            if mem is not None:
                qm = np.exp(mem[:, p, :][:, sl])
                lo, hi = np.percentile(qm, 16, 0), np.percentile(qm, 84, 0)
                ax.fill_between(z[sl], lo, hi, color=COLORS[name],
                                alpha=0.18, label=f"{LABELS[name]} 16–84%")
                shown.extend([lo, hi])
        ax.set_title(f"pixel {p}", fontsize=11)
    # Q is multiplicative: a log axis keeps the near-unity pixels readable
    # next to the extreme one instead of flattening them onto Q ≈ 0.
    ymin = max(3e-2, 0.7 * min(float(v.min()) for v in shown))
    ymax = 1.4 * max(float(v.max()) for v in shown)
    for ax in axes.ravel():
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
    for ax in axes[-1]:
        ax.set_xlabel("z")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$Q_{\rm LSS}(p,z)$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, 0.95), frameon=False)
    fig.suptitle("Completion factor: fitted vs truth (pixels at truth-Q extremes)",
                 y=0.99)
    fig.subplots_adjust(top=0.85)
    fig.savefig(plots / "q_z_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_q_sky_slices(d, plots):
    import healpy as hp
    import matplotlib.pyplot as plt
    z = np.asarray(zgrid)
    zd = float(d["attrs"]["zmax"])
    z_slices = [0.2 * zd, 0.5 * zd, 0.8 * zd]
    cols = [("truth", d["logq_truth"])]
    for name in ("q_gp3d", "q_radial"):
        if name in d["tables"]:
            cols.append((name, d["tables"][name][0]))
    have_res = "q_gp3d" in d["tables"]
    ncol = len(cols) + int(have_res)
    # Round the symmetric color range so the colorbar end labels stay short.
    vmax = max(0.15, float(np.percentile(np.abs(d["logq_truth"]), 99)))
    vmax = float(np.ceil(vmax * 10.0) / 10.0)
    fig = plt.figure(figsize=(4.1 * ncol, 3.0 * len(z_slices)))
    for i, zs in enumerate(z_slices):
        j = int(np.argmin(np.abs(z - zs)))
        for c, (name, table) in enumerate(cols):
            hp.mollview(table[:, j], fig=fig.number,
                        sub=(len(z_slices), ncol, i * ncol + c + 1),
                        title=f"{LABELS.get(name, name)}  z={z[j]:.2f}",
                        cmap="RdBu_r", min=-vmax, max=vmax, cbar=(c == 0),
                        format="%.1f", unit=r"$\log Q$" if c == 0 else None)
        if have_res:
            res = d["tables"]["q_gp3d"][0][:, j] - d["logq_truth"][:, j]
            hp.mollview(res, fig=fig.number,
                        sub=(len(z_slices), ncol, i * ncol + ncol),
                        title=f"gp3d − truth  z={z[j]:.2f}",
                        cmap="RdBu_r", min=-vmax, max=vmax, cbar=False)
    fig.suptitle(r"$\log Q$ sky maps at redshift slices", y=1.02)
    fig.savefig(plots / "q_sky_slices.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_slabs(d, plots, half_thick):
    import matplotlib.pyplot as plt
    attrs = d["attrs"]
    nside = int(attrs["nside"])
    zd = float(attrs["zmax"])
    chi_max = float(np.interp(zd, d["cosmo_z"], d["cosmo_dc"]))
    grid = np.linspace(-chi_max, chi_max, 400)
    G1, G2 = np.meshgrid(grid, grid)
    zero = np.zeros_like(G1)
    panels = [("truth", d["logq_truth"])]
    for name in ("q_gp3d", "q_radial"):
        if name in d["tables"]:
            panels.append((name, d["tables"][name][0]))
    if "q_gp3d" in d["tables"]:
        panels.append(("residual", d["tables"]["q_gp3d"][0] - d["logq_truth"]))
    vmax = max(0.15, float(np.percentile(np.abs(d["logq_truth"]), 99)))

    ra_g = d["cat"]["ra"]
    dec_g = d["cat"]["dec"]
    z_g = d["cat"]["z"]
    obs_g = d["cat"]["observed"].astype(bool)
    chi_g = np.interp(z_g, d["cosmo_z"], d["cosmo_dc"])
    xg = chi_g * np.cos(dec_g) * np.cos(ra_g)
    yg = chi_g * np.cos(dec_g) * np.sin(ra_g)
    zg3 = chi_g * np.sin(dec_g)

    for axis, (name1, name2) in zip("XYZ", [("Y", "Z"), ("X", "Z"), ("X", "Y")]):
        if axis == "X":
            x, y, zc = zero, G1, G2
            g_perp, g_a, g_b = xg, yg, zg3
        elif axis == "Y":
            x, y, zc = G1, zero, G2
            g_perp, g_a, g_b = yg, xg, zg3
        else:
            x, y, zc = G1, G2, zero
            g_perp, g_a, g_b = zg3, xg, yg
        pix, zval, chi = xyz_to_pix_z(x, y, zc, nside, d["cosmo_z"], d["cosmo_dc"])
        mask = chi <= chi_max
        # Nearest zgrid node for every slab point (zgrid is sorted).
        zg = np.asarray(zgrid)
        zidx = np.searchsorted(zg, zval).clip(1, zg.size - 1)
        zidx = np.where(zval - zg[zidx - 1] < zg[zidx] - zval, zidx - 1, zidx)
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6),
                                 sharex=True, sharey=True)
        in_slab = obs_g & (np.abs(g_perp) < half_thick)
        for ax, (name, table) in zip(np.atleast_1d(axes), panels):
            field = np.where(mask, table[pix, zidx], np.nan)
            im = ax.pcolormesh(grid, grid, field, cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax, shading="auto")
            ax.scatter(g_a[in_slab], g_b[in_slab], s=0.5, c="k", alpha=0.25,
                       linewidths=0, rasterized=True)
            ax.set(title=LABELS.get(name, name), xlabel=f"{name1} [Mpc]",
                   aspect="equal")
        np.atleast_1d(axes)[0].set_ylabel(f"{name2} [Mpc]")
        fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), shrink=0.85,
                     label=r"$\log Q$")
        fig.suptitle(rf"Comoving slab ${axis}=0$ (galaxies: observed, "
                     rf"$|{axis}|<{half_thick:.0f}$ Mpc)")
        fig.savefig(plots / f"slabs_{axis.lower()}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


def plot_dnmiss_and_priors(d, plots):
    import healpy as hp
    import matplotlib.pyplot as plt
    z = np.asarray(zgrid)
    zd = float(d["attrs"]["zmax"])
    sl = z <= zd
    pixels = d["prior_pixels"]
    show = pixels[np.unique(np.linspace(0, pixels.size - 1, 4).astype(int))]
    order = [n for n in ("homog", "delta_g", "q_radial", "q_gp3d")
             if n in d["curves"]]

    # Every curve in BOTH panels is renormalized to unit area over the
    # DISPLAYED range: raw amplitudes are not comparable (field priors carry
    # the survey-global Z, conditional priors carry mass beyond z_depth, and
    # the branch curves carry each configuration's own missing budget — shown
    # in dndz_closure.png), so these panels compare SHAPES.
    def _unit_area(pr):
        return pr / max(np.trapz(pr[sl], z[sl]), 1e-300)

    # Truth per pixel: the complete catalog's galaxies (same RING pixelization
    # as the generator) and the smooth expectation dN_exp · Q_truth.
    nside = int(d["attrs"]["nside"])
    gal_pix = hp.ang2pix(nside, np.pi / 2.0 - d["cat"]["dec"], d["cat"]["ra"])
    z_all = d["cat"]["z"]
    dn_true = np.nan_to_num(d["dN_exp"])[None, :] * np.exp(d["logq_truth"])
    bins = np.linspace(0.0, zd, 26)
    for p in show:
        i_sel = int(np.nonzero(pixels == p)[0][0])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6),
                                       layout="constrained")
        # Branch shapes: observed branch (smoothed KDE) + missing branch.
        h_pix, _ = np.histogram(z_all[gal_pix == p], bins=bins, density=True)
        n_pix_gals = int((gal_pix == p).sum())
        ax1.stairs(h_pix, bins, color="k", lw=1.2, alpha=0.5,
                   label=f"truth (complete catalog, {n_pix_gals} gals)")
        ys = [h_pix, _unit_area(dn_true[p])[sl]]
        ax1.plot(z[sl], ys[-1], color="k", lw=2,
                 label=r"truth (expected, $dN_{\rm exp}\,Q_{\rm truth}$)")
        occ_idx = np.nonzero(d["obs"]["occupied_pixels"] == p)[0]
        if occ_idx.size:
            ys.append(_unit_area(d["obs"]["dN_obs_s"][occ_idx[0]])[sl])
            ax1.plot(z[sl], ys[-1], color="0.4", ls="--", lw=1.6,
                     label="observed branch (KDE)")
        for name in order:
            ys.append(_unit_area(d["curves"][name]["dN_miss"][p])[sl])
            ax1.plot(z[sl], ys[-1], color=COLORS[name], lw=1.4,
                     ls=":" if name == "homog" else "-",
                     label=f"missing ({LABELS[name]})")
        # The missing branches can spike at the depth edge (C -> 0); scale the
        # view to the curves away from the edge so the spike cannot flatten
        # everything else (it stays visible, truncated).
        inner = z[sl] <= 0.9 * zd
        ymax = 1.3 * max(float(np.nanmax(y[inner] if y.size == inner.size
                                         else y[:-1])) for y in ys)
        ax1.set_ylim(0.0, ymax)
        ax1.set(xlabel="z", ylabel="shape (unit area, $z\\leq z_{\\rm depth}$)",
                title=f"Branch shapes, pixel {p}")
        ax1.legend(fontsize=8, loc="upper left")
        ax2.plot(z[sl], _unit_area(dn_true[p])[sl], color="k", lw=2,
                 label="truth (expected shape)")
        for name in order:
            for weighting, ls in (("conditional", "-"), ("field", "--")):
                key = f"{name}_{weighting}"
                if key not in d["priors"]:
                    continue
                pr = _unit_area(np.exp(d["priors"][key]["logp"][i_sel]))
                ax2.plot(z[sl], pr[sl], color=COLORS[name], ls=ls, lw=1.3,
                         label=f"{LABELS[name]} ({weighting})")
            key = f"{name}_conditional"
            if key in d["priors"] and "logp_members" in d["priors"][key]:
                pm = np.exp(d["priors"][key]["logp_members"][:, i_sel, :])
                pm = np.stack([_unit_area(m) for m in pm])
                ax2.fill_between(z[sl], np.percentile(pm, 16, 0)[sl],
                                 np.percentile(pm, 84, 0)[sl],
                                 color=COLORS[name], alpha=0.15)
        ax2.set(xlabel="z",
                ylabel=r"$p(z\,|\,{\rm pix})$ (unit area, $z\leq z_{\rm depth}$)",
                title=f"Assembled redshift prior, pixel {p}")
        ax2.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   borderaxespad=0.0)
        fig.savefig(plots / f"dnmiss_and_prior_pix{p}.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)


def plot_closure_scatter(d, plots, summary):
    """PRIMARY closure: the missing-galaxy density dN_miss = (1−C)·dN_exp·Q.

    The kernel-ratio C is estimated per pixel from the observed counts, so it
    already absorbs every LSS mode resolvable above the KDE smoothing scale —
    the fitted Q is only the sub-smoothing residual (module docstring caveat),
    NOT an estimator of the injected Q.  What the redshift prior actually
    consumes is dN_miss, so that is the voxel-by-voxel closure target; the
    Q-vs-Q comparison is kept as a secondary panel (q_scatter.png).
    """
    import matplotlib.pyplot as plt
    z = np.asarray(zgrid)
    zd = float(d["attrs"]["zmax"])
    ct = _c_true(z, d["attrs"], d["cosmo_z"], d["cosmo_dl"])
    # Mask: below depth, meaningful incompleteness, occupied pixels.
    sl = (z <= zd) & (1.0 - ct > 0.05) & np.isfinite(d["dN_exp"])
    occ = d["obs"]["N_obs"] > 0
    dn_true = ((1.0 - ct) * d["dN_exp"])[None, :] * np.exp(d["logq_truth"])
    with np.errstate(divide="ignore"):
        log_true_grid = np.log(dn_true[:, sl])
    # ANOMALY convention: remove each field's own per-z sky mean, so the
    # scatter measures the SPATIAL structure of the correction, not the
    # (1−C_true)·dN_exp z-trend every configuration shares.
    log_true_anom = log_true_grid - np.nanmean(log_true_grid[occ], axis=0)
    log_true = log_true_anom[occ].ravel()

    order = [n for n in ("homog", "delta_g", "q_radial", "q_gp3d",
                         "q_radial_n0off") if n in d["curves"]]
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.4), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    sub = rng.choice(log_true.size, size=min(20000, log_true.size), replace=False)
    lo_l, hi_l = np.percentile(log_true, [1.0, 99.0])
    pad = 0.08 * (hi_l - lo_l)
    lims = (lo_l - pad, hi_l + pad)
    for ax, name in zip(np.atleast_1d(axes), order):
        with np.errstate(divide="ignore"):
            log_fit_grid = np.log(d["curves"][name]["dN_miss"][:, sl])
        log_fit_grid[~np.isfinite(log_fit_grid)] = np.nan
        log_fit = (log_fit_grid - np.nanmean(log_fit_grid[occ], axis=0))[occ].ravel()
        ok = np.isfinite(log_fit) & np.isfinite(log_true)
        r = float(np.corrcoef(log_true[ok], log_fit[ok])[0, 1])
        slope = float(np.polyfit(log_true[ok], log_fit[ok], 1)[0])
        rms = float(np.sqrt(np.mean((log_fit[ok] - log_true[ok]) ** 2)))
        summary[f"dnmiss_r_{name}"] = r
        summary[f"dnmiss_slope_{name}"] = slope
        summary[f"dnmiss_rms_{name}"] = rms
        ax.plot(lims, lims, color="0.6", lw=1)
        ax.scatter(log_true[sub], log_fit[sub], s=2.0, c=COLORS[name],
                   alpha=0.25, linewidths=0, rasterized=True)
        ax.set_title(f"{LABELS[name]}\nr={r:.3f}, slope={slope:.2f}, rms={rms:.2f}",
                     fontsize=11)
        ax.set(xlabel=r"truth $\Delta\log dN_{\rm miss}$", xlim=lims, ylim=lims,
               aspect="equal")
    np.atleast_1d(axes)[0].set_ylabel(r"fitted $\Delta\log dN_{\rm miss}$")
    fig.suptitle(r"Missing-density spatial closure (per-$z$ sky-mean removed; "
                 r"$z \leq z_{\rm depth}$, $1-C_{\rm true}>0.05$)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(plots / "closure_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # TOTAL host-density closure: observed + missing — the spatial structure
    # the assembled prior actually carries.  With the HOMOGENEOUS missing
    # branch, dN_obs + (1−C)·dN_exp ≈ dN_exp: the correction exactly CANCELS
    # the catalog's clustering (r ≈ 0 by construction); Q/δ_g must restore it.
    dN_obs_full = np.zeros((d["logq_truth"].shape[0], int(sl.sum())))
    dN_obs_full[d["obs"]["occupied_pixels"]] = d["obs"]["dN_obs_s"][:, sl]
    with np.errstate(divide="ignore"):
        log_tot_true = np.log(d["dN_exp"][None, sl] * np.exp(d["logq_truth"][:, sl]))
    tot_true_anom = (log_tot_true - np.nanmean(log_tot_true[occ], axis=0))[occ].ravel()
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 4.4), sharex=True, sharey=True)
    for ax, name in zip(np.atleast_1d(axes), order):
        tot = dN_obs_full + d["curves"][name]["dN_miss"][:, sl]
        with np.errstate(divide="ignore"):
            log_tot = np.log(tot)
        log_tot[~np.isfinite(log_tot)] = np.nan
        anom = (log_tot - np.nanmean(log_tot[occ], axis=0))[occ].ravel()
        ok = np.isfinite(anom) & np.isfinite(tot_true_anom)
        r = float(np.corrcoef(tot_true_anom[ok], anom[ok])[0, 1])
        slope = float(np.polyfit(tot_true_anom[ok], anom[ok], 1)[0])
        summary[f"total_r_{name}"] = r
        summary[f"total_slope_{name}"] = slope
        ax.plot(lims, lims, color="0.6", lw=1)
        ax.scatter(tot_true_anom[sub], anom[sub], s=2.0, c=COLORS[name],
                   alpha=0.25, linewidths=0, rasterized=True)
        ax.set_title(f"{LABELS[name]}\nr={r:.3f}, slope={slope:.2f}", fontsize=11)
        ax.set(xlabel=r"truth $\Delta\log(dN_{\rm tot}/dz)$", xlim=lims, ylim=lims,
               aspect="equal")
    np.atleast_1d(axes)[0].set_ylabel(r"prior $\Delta\log(dN_{\rm obs}+dN_{\rm miss})$")
    fig.suptitle("Assembled host-density spatial closure — does the correction "
                 "preserve the catalog's clustering?")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(plots / "total_density_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    # PASS if the LSS-aware configurations restore more of the true structure
    # than the homogeneous branch retains.
    r0 = summary.get("total_r_homog", 0.0)
    s0 = summary.get("total_slope_homog", 0.0)
    summary["structure_restored_pass"] = bool(
        summary.get("total_r_delta_g", 0.0) > r0 + 0.05
        and summary.get("total_slope_delta_g", 0.0) > s0
        and summary.get("total_r_q_gp3d", 0.0) > r0 + 0.05
        and summary.get("total_slope_q_gp3d", 0.0) > s0)

    # SECONDARY: fitted Q vs injected Q (interpretation caveat above).
    truth = d["logq_truth"][np.ix_(occ, sl)].ravel()
    cases = []
    for name in ("q_gp3d", "q_radial", "q_radial_n0off"):
        if name in d["tables"]:
            cases.append((name, d["tables"][name][0][np.ix_(occ, sl)].ravel()))
    if d["delta_g"] is not None:
        with np.errstate(divide="ignore"):
            logq_dg = np.log(np.maximum(1.0 + d["delta_g"], 1e-3))
        cases.append(("delta_g", logq_dg[np.ix_(occ, sl)].ravel()))
    fig, axes = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 4.4),
                             sharex=True, sharey=True)
    lim = max(0.5, 1.05 * float(np.percentile(np.abs(truth), 99.5)))
    for ax, (name, fit) in zip(np.atleast_1d(axes), cases):
        ok = np.isfinite(fit) & np.isfinite(truth)
        r = float(np.corrcoef(truth[ok], fit[ok])[0, 1])
        slope = float(np.polyfit(truth[ok], fit[ok], 1)[0])
        summary[f"q_residual_r_{name}"] = r
        summary[f"q_residual_slope_{name}"] = slope
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1)
        ax.scatter(truth[sub], fit[sub], s=2.0, c=COLORS[name], alpha=0.25,
                   linewidths=0, rasterized=True)
        ax.set_title(f"{LABELS[name]}\nr={r:.3f}, slope={slope:.2f}", fontsize=11)
        ax.set(xlabel=r"truth $\log Q$", xlim=(-lim, lim), ylim=(-lim, lim),
               aspect="equal")
    np.atleast_1d(axes)[0].set_ylabel(r"fitted $\log Q$")
    fig.suptitle("Q table vs injected field (Q is the sub-smoothing residual "
                 "on top of per-pixel C — slope < 1 expected)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(plots / "q_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Member-band empirical coverage of the truth Q (same caveat).
    for name in ("q_gp3d", "q_radial"):
        if name not in d["tables"] or d["tables"][name][1] is None:
            continue
        mem = d["tables"][name][1][:, occ, :][:, :, sl]
        lo = np.percentile(mem, 16, axis=0).ravel()
        hi = np.percentile(mem, 84, axis=0).ravel()
        t = d["logq_truth"][np.ix_(occ, sl)].ravel()
        summary[f"member_band_coverage_{name}"] = float(
            np.mean((t >= lo) & (t <= hi)))


def main(argv=None):
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    plots = outdir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    from darksirens.utils.plotting import set_publication_style
    set_publication_style()

    d = _load(outdir)
    summary = {"mean_Q_truth": float(np.exp(d["logq_truth"]).mean())}
    summary["mean_Q_truth_pass"] = bool(abs(summary["mean_Q_truth"] - 1.0) < 0.05)

    plot_closure_scatter(d, plots, summary)
    plot_c_calibration(d, plots, summary)
    plot_dndz_closure(d, plots, summary)
    plot_q_z_curves(d, plots)
    plot_q_sky_slices(d, plots)
    plot_slabs(d, plots, args.slab_half_thickness)
    plot_dnmiss_and_priors(d, plots)

    with open(plots / "closure_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    fails = [k for k, v in summary.items() if k.endswith("_pass") and not v]
    print("[closure] " + ("ALL CHECKS PASS" if not fails else f"FAILED: {fails}"))


if __name__ == "__main__":
    main()
