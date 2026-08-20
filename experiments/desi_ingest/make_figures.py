"""Preliminary figures for the DESI real-catalog ingestion (Stage A + gates).

Reads only the result JSONs / HDF5 products already in data/; writes PNGs to
figures/.  Series colors are the Okabe-Ito colorblind-safe set with fixed
identity per channel across every figure.
"""

from __future__ import annotations

import json

import h5py
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402

FIGDIR = C.EXP_DIR / "figures"
FIGDIR.mkdir(exist_ok=True)

# Fixed channel identity (Okabe-Ito).
COL = {
    "complete": "#0072B2",     # blue
    "per_pixel": "#E69F00",    # orange
    "sel": "#009E73",          # green
    "selq_radial": "#CC79A7",  # magenta
    "selq_gp3d": "#56B4E9",    # sky blue
    "sel_strat": "#000000",    # black (overlay)
}
LBL = {
    "complete": "complete (joint42-style)",
    "per_pixel": "per-pixel counts",
    "sel": "selection (no Q)",
    "selq_radial": "selection + radial Q",
    "selq_gp3d": "selection + gp3d Q",
    "sel_strat": "selection, N/S stratified",
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})


def fig_h0_posteriors():
    r = json.load(open(C.DATA_DIR / "h0_real" / "h0_real_scans.json"))["results"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for k in ("complete", "per_pixel", "sel", "selq_radial", "selq_gp3d"):
        v = r[k]
        lw = 2 if k in ("complete", "per_pixel", "sel") else 1.4
        ax.plot(v["h0"], v["pdf"], color=COL[k], lw=lw, label=LBL[k])
    v = r["sel_strat"]
    ax.plot(v["h0"], v["pdf"], color=COL["sel_strat"], lw=1.2, ls="--",
            label=LBL["sel_strat"])
    for k in ("complete", "per_pixel", "sel"):
        v = r[k]
        i = int(np.argmax(v["pdf"]))
        ax.annotate(f"{v['median']:.1f}", (v["h0"][i], v["pdf"][i]),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8, color="#444444")
    ax.set_xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylabel("posterior density")
    ax.set_xlim(20, 140)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("44 GWTC events x DESI union catalog - completeness channels",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "h0_posteriors_channels.png")
    plt.close(fig)


def fig_counts_cbar():
    d = json.load(open(C.DATA_DIR / "diagnostics_stage_a.json"))
    cb = d["counts_cbar_vs_csel"]
    z = np.array(cb["z"])
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 4.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]})
    ax.plot(z, cb["c_sel"], color="#009E73", lw=2,
            label=r"fitted $C_{\rm sel}(z)$")
    ax.plot(z, cb["counts_cbar"], color="#E69F00", lw=2,
            label=r"counts-based $\bar{C}(z)$ (alarm)")
    ax.set_ylabel("completeness")
    ax.set_ylim(0.75, 1.12)
    ax.axhline(1.0, color="#999999", lw=0.7)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Counts-based aggregate vs fitted magnitude selection "
                 "(the misspecification alarm)", fontsize=9)
    resid = np.array(cb["closure_resid_frac"])
    ax2.axhline(0.0, color="#999999", lw=0.7)
    ax2.fill_between(z, resid, 0, color="#E69F00", alpha=0.45, lw=0)
    ax2.set_ylabel("closure resid.")
    ax2.set_xlabel("z")
    ax2.set_ylim(-0.15, 0.15)
    ax2.annotate("coherent LSS dip - absorbed by Q", (0.15, -0.08),
                 fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(FIGDIR / "counts_cbar_alarm.png")
    plt.close(fig)


def fig_kcorr():
    d = json.load(open(C.DATA_DIR / "kcorr_poly.json"))
    b = d["median_binned_curve"]
    z = np.linspace(0, 0.32, 200)
    c1, c2, c3 = d["k_corr_coeffs"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(b["z"], b["K_median"], s=10, color="#666666",
               label="catalog median $K_{\\rm eff}$ per z-bin", zorder=3)
    ax.plot(z, c1 * z + c2 * z**2 + c3 * z**3, color="#009E73", lw=2,
            label="cubic template (rms 3.6 mmag)")
    ax.plot(z, 0.39 * z, color="#0072B2", lw=1.2, ls=":",
            label=r"0.39$\,z$ (depth-scan note)")
    ax.axhline(d["chilingarian_fixed_color_K_at_z030"], color="#CC79A7",
               lw=1.2, ls="--")
    ax.annotate("Chilingarian @ fixed g-r, z=0.30", (0.005, 0.052),
                fontsize=8, color="#CC79A7")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$K_r(z)$ [mag]")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Data-implied r-band K-correction template", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "kcorr_template.png")
    plt.close(fig)


def fig_stratified_fits():
    d = json.load(open(C.DATA_DIR / "diagnostics_stage_a.json"))
    s = d["stratified_fits"]
    order = ["pooled", "south", "north", "dec_neg", "dec_pos"]
    labels = ["pooled", "LS south\n(DR10)", "LS north\n(DR9)",
              "dec < 0", "dec >= 0"]
    vals = [s[k]["M0hat"] for k in order]
    errs = [s[k]["M0hat_sd"] for k in order]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    xs = np.arange(len(order))
    ax.errorbar(xs, vals, yerr=np.array(errs) * 10, fmt="o", ms=6,
                color="#0072B2", ecolor="#0072B2", capsize=3, lw=1.5)
    ax.axhline(s["pooled"]["M0hat"], color="#999999", lw=0.8, ls="--")
    ax.set_xticks(xs, labels)
    ax.set_ylabel(r"$\hat{M}_0$ (h-scaled)")
    dm = s["delta_M0hat_north_minus_south"]
    ax.set_title(
        f"Per-subset selection fits - N-S offset "
        f"{dm['value']:+.4f} mag ({dm['significance']:.0f}$\\sigma$); "
        "error bars x10", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "stratified_fits.png")
    plt.close(fig)


def fig_loo():
    d = json.load(open(C.DATA_DIR / "loo" / "loo_results.json"))
    names = list(d["loo"])
    shifts = [d["loo"][n]["delta_median"] for n in names]
    off = json.load(open(C.DATA_DIR / "diagnostics_stage_a.json"))[
        "off_footprint_pe_mass"]["per_event"]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    ys = np.arange(len(names))[::-1]
    ax.axvline(0, color="#999999", lw=0.8)
    ax.scatter(shifts, ys, s=40, color="#0072B2", zorder=3)
    for y, n, sh in zip(ys, names, shifts):
        ax.annotate(f"{off.get(n, 0) * 100:.0f}% off-footprint",
                    (sh, y), textcoords="offset points", xytext=(8, -3),
                    fontsize=7.5, color="#666666")
    ax.set_yticks(ys, [n.split("_")[0] for n in names])
    ax.set_xlabel(r"$\Delta$ median $H_0$ when dropped [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_xlim(-4.5, 4.5)
    ax.set_title("Leave-one-out shifts, flagged off-footprint events "
                 "(posterior sigma ~ 15)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGDIR / "loo_jackknife.png")
    plt.close(fig)


def fig_skymaps():
    with h5py.File(C.DATA_DIR / "pixelated_n64" /
                   "catalog_pixelated_nside_64.h5", "r") as f:
        ngals = f["ngals"][...].astype(float)
    with h5py.File(C.DATA_DIR / "stratum_map_ns_nside64.h5", "r") as f:
        smap = f["stratum_map"][...].astype(float)
    fig = plt.figure(figsize=(9.2, 3.2))
    m = np.where(ngals > 0, np.log10(np.maximum(ngals, 1)), np.nan)
    hp.mollview(m, sub=(1, 2, 1), title="log10 galaxies / pixel (nside 64)",
                cmap="viridis", badcolor="#eeeeee", cbar=True, fig=fig.number)
    sm = np.where(ngals > 0, smap, np.nan)
    hp.mollview(sm, sub=(1, 2, 2),
                title="stratum map: 0 = LS south (DR10), 1 = LS north (DR9)",
                cmap=matplotlib.colors.ListedColormap(["#E69F00", "#0072B2"]),
                badcolor="#eeeeee", cbar=False, fig=fig.number)
    fig.savefig(FIGDIR / "skymaps.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_h0_posteriors()
    fig_counts_cbar()
    fig_kcorr()
    fig_stratified_fits()
    fig_loo()
    fig_skymaps()
    for p in sorted(FIGDIR.glob("*.png")):
        print(p, f"{p.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
