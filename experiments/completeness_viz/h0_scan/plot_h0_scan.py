#!/usr/bin/env python3
"""Comparison figures for the H0 scans (reads results/scan_results.json)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"gate_complete": "k", "homog": "0.45", "deltag": "C2",
          "qradial": "C1", "qgp3d": "C0"}
LABELS = {"gate_complete": "complete catalog (gate)", "homog": "homogeneous",
          "deltag": r"legacy $1+b\,\delta_g$", "qradial": "Q radial",
          "qgp3d": "Q gp3d"}


def _interval(h0, ll, dchi2=1.0):
    """Peak and the ±sqrt(dchi2)-sigma-equivalent interval from dlogL."""
    d = ll - ll.max()
    i = int(np.argmax(d))
    inside = h0[d > -dchi2 / 2.0]
    return h0[i], (inside[0], inside[-1]) if inside.size else (np.nan, np.nan)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results_final/scan_results.json")
    p.add_argument("--outdir", default="results_final")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import common  # noqa: F401
    from darksirens.utils.plotting import set_publication_style
    set_publication_style()

    blob = json.load(open(args.results))
    h0_true = blob["H0_true"]
    res = blob["results"]
    outdir = Path(args.outdir)

    # ---- 1-D scan comparison: per-pixel vs aggregate panels ---------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                             layout="constrained")
    summary = {}
    for ax, mode in zip(axes, ("pp", "agg")):
        ax.axvline(h0_true, color="0.75", lw=1.5, ls=":")
        for base in ("gate_complete", "homog", "deltag", "qradial", "qgp3d"):
            name = base if base == "gate_complete" else f"{base}_{mode}"
            if name not in res:
                continue
            h0 = np.array(res[name]["h0"])
            ll = np.array(res[name]["logl"])
            d = ll - ll.max()
            pk, (lo, hi) = _interval(h0, ll)
            summary[name] = {"peak": pk, "lo68": lo, "hi68": hi,
                             "neff_min": min(res[name]["sel_neff"])}
            ax.plot(h0, d, color=COLORS[base], lw=2 if base == "gate_complete"
                    else 1.5, ls="--" if base == "gate_complete" else "-",
                    label=f"{LABELS[base]}  ({pk:.1f})")
        ax.set(xlabel=r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]",
               title="per-pixel C (legacy)" if mode == "pp"
               else r"aggregate $\bar{C}$ (this work)",
               ylim=(-30, 1.5))
        ax.legend(fontsize=8, loc="lower right",
                  title=r"config (peak $H_0$)", title_fontsize=8)
    axes[0].set_ylabel(r"$\Delta \ln \mathcal{L}$")
    fig.suptitle(r"$H_0$ scans: 300 dark sirens on the clustered mock, "
                 "survey block fixed at truth")
    fig.savefig(outdir / "h0_scan_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- 2-D profiles ------------------------------------------------------
    twod = [n for n in res if "logl_2d" in res[n]]
    if twod:
        fig, axes = plt.subplots(1, len(twod), figsize=(5.4 * len(twod), 4.6),
                                 sharey=True, layout="constrained")
        for ax, name in zip(np.atleast_1d(axes), twod):
            h0 = np.array(res[name]["h0"])
            n0g = np.array(res[name]["n0_grid"])
            Z = np.array(res[name]["logl_2d"])
            prof = Z.max(axis=1)
            base = name.rsplit("_", 1)[0]
            ax.plot(h0, prof - prof.max(), color=COLORS.get(base, "C3"),
                    lw=1.8, label=r"profiled over $\log_{10} n_0$")
            fixed_ll = np.array(res[name]["logl"])
            ax.plot(h0, fixed_ll - fixed_ll.max(), color=COLORS.get(base, "C3"),
                    lw=1.2, ls="--", label=r"$\log_{10} n_0$ fixed at truth")
            ax.axvline(h0_true, color="0.75", lw=1.5, ls=":")
            pk = h0[np.argmax(prof)]
            nhat = n0g[np.argmax(Z[np.argmax(prof)])]
            summary[name]["peak_profiled"] = pk
            summary[name]["log10n0_hat"] = float(nhat)
            ax.set(xlabel=r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]", ylim=(-30, 1.5),
                   title=f"{LABELS.get(base, base)} "
                         f"[{name.rsplit('_', 1)[1]}]\n"
                         rf"profiled peak {pk:.1f}, "
                         rf"$\widehat{{\log n_0}}$={nhat:.2f}")
            ax.legend(fontsize=8, loc="lower right")
        np.atleast_1d(axes)[0].set_ylabel(r"$\Delta \ln \mathcal{L}$")
        fig.suptitle(r"Profile scans: $H_0$ with $\log_{10} n_0$ free")
        fig.savefig(outdir / "h0_profile_comparison.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    with open(outdir / "scan_summary.json", "w") as f:
        json.dump({"H0_true": h0_true, "summary": summary}, f, indent=2,
                  sort_keys=True)
    for k, v in summary.items():
        print(f"{k:16s} peak={v['peak']:5.1f}  68%~[{v['lo68']:.1f},"
              f"{v['hi68']:.1f}]  Neff_min={v['neff_min']:.0f}"
              + (f"  profiled={v['peak_profiled']:.1f}"
                 if "peak_profiled" in v else ""))


if __name__ == "__main__":
    main()
