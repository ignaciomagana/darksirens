#!/usr/bin/env python
"""Overlay the stored-PE baseline and flow-surrogate posteriors.

Reads two darksirens results.hdf5 run directories (run A: --gw_path,
run B: --gw_flows_path), overlays the sampled-parameter posteriors, and
writes a comparison table (medians, 90% CIs, logZ, runtimes).

Usage:
    python scripts/flows_compare_runs.py --runs_dir flows/runs \
        --outdir flows/inference_comparison
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_run(run_dir: Path):
    with open(run_dir / "settings.json") as f:
        settings = json.load(f)
    with h5py.File(run_dir / "results.hdf5", "r") as f:
        samples = f["samples"][...]
        labels = [l.decode() if isinstance(l, bytes) else str(l) for l in f["labels"][...]]
        attrs = dict(f.attrs)
    return dict(settings=settings, samples=samples, labels=labels, attrs=attrs,
                dir=str(run_dir))


def _find_runs(runs_dir: Path):
    run_a = run_b = None
    for d in sorted(runs_dir.glob("*__spectral_sirens__tinyns__*")):
        if not (d / "results.hdf5").exists():
            continue
        settings = json.load(open(d / "settings.json"))
        opts = settings.get("opts", settings)
        gw_flows = opts.get("gw_flows_path") or settings.get("gw_flows_path")
        if gw_flows:
            run_b = d  # latest wins
        else:
            run_a = d
    return run_a, run_b


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs_dir", default="flows/runs")
    ap.add_argument("--run_a", default=None, help="explicit stored-PE run dir")
    ap.add_argument("--run_b", default=None, help="explicit flow run dir")
    ap.add_argument("--outdir", default="flows/inference_comparison")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    run_a_dir, run_b_dir = _find_runs(runs_dir)
    if args.run_a:
        run_a_dir = Path(args.run_a)
    if args.run_b:
        run_b_dir = Path(args.run_b)
    if run_a_dir is None or run_b_dir is None:
        raise SystemExit(f"Could not locate both runs under {runs_dir} "
                         f"(A={run_a_dir}, B={run_b_dir}).")

    A = _load_run(run_a_dir)
    B = _load_run(run_b_dir)
    assert A["labels"] == B["labels"], (A["labels"], B["labels"])
    labels = A["labels"]
    ndim = len(labels)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── overlay corner ───────────────────────────────────────────────────
    fig, axes = plt.subplots(ndim, ndim, figsize=(3.2 * ndim, 3.2 * ndim))
    if ndim == 1:
        axes = np.array([[axes]])
    ranges = []
    for i in range(ndim):
        lo = min(np.percentile(A["samples"][:, i], 0.3),
                 np.percentile(B["samples"][:, i], 0.3))
        hi = max(np.percentile(A["samples"][:, i], 99.7),
                 np.percentile(B["samples"][:, i], 99.7))
        pad = 0.05 * (hi - lo)
        ranges.append((lo - pad, hi + pad))

    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            if i == j:
                bins = np.linspace(*ranges[i], 50)
                ax.hist(A["samples"][:, i], bins=bins, density=True, alpha=0.45,
                        color="C0", label="stored PE")
                ax.hist(B["samples"][:, i], bins=bins, density=True,
                        histtype="step", lw=1.8, color="C3", label="flows")
                ax.set_yticks([])
                if i == 0:
                    ax.legend(frameon=False, fontsize=9)
            else:
                for S, color in ((A, "C0"), (B, "C3")):
                    h, xe, ye = np.histogram2d(
                        S["samples"][:, j], S["samples"][:, i],
                        bins=40, range=[ranges[j], ranges[i]],
                    )
                    h = h.T
                    hs = np.sort(h.ravel())[::-1]
                    cs = np.cumsum(hs) / hs.sum()
                    lv = sorted({hs[np.searchsorted(cs, q)] for q in (0.39, 0.86)})
                    if len(lv) < 2 or lv[0] == lv[-1]:
                        continue
                    ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]),
                               h, levels=lv, colors=color, linewidths=1.3)
                ax.set_xlim(ranges[j]); ax.set_ylim(ranges[i])
            if i == ndim - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
            elif j > 0:
                ax.set_yticklabels([])
    fig.suptitle("stored-PE baseline (blue) vs flow surrogates (red)")
    fig.tight_layout()
    fig.savefig(outdir / "corner_overlay.png", dpi=150)
    plt.close(fig)

    # ── table ────────────────────────────────────────────────────────────
    def _row(S, i):
        s = S["samples"][:, i]
        lo, med, hi = np.percentile(s, [5, 50, 95])
        return med, med - lo, hi - med

    lines = [
        "| parameter | stored PE (median -5%/+95%) | flows (median -5%/+95%) | Δmedian / σ_PE |",
        "|---|---|---|---|",
    ]
    summary = {}
    for i, lab in enumerate(labels):
        mA, loA, hiA = _row(A, i)
        mB, loB, hiB = _row(B, i)
        sigA = 0.5 * (loA + hiA) / 1.645
        lines.append(
            f"| {lab} | {mA:.3f} (-{loA:.3f}/+{hiA:.3f}) "
            f"| {mB:.3f} (-{loB:.3f}/+{hiB:.3f}) | {(mB-mA)/max(sigA,1e-12):+.2f} |"
        )
        summary[lab] = dict(pe=[mA, loA, hiA], flows=[mB, loB, hiB],
                            shift_sigma=(mB - mA) / max(sigA, 1e-12))

    for S, name in ((A, "stored PE"), (B, "flows")):
        lines.append("")
        lines.append(
            f"- {name}: logZ = {S['attrs'].get('logZ', float('nan')):.2f} "
            f"± {S['attrs'].get('logZerr', float('nan')):.2f}; "
            f"sampling runtime {S['attrs'].get('sampling_runtime', '?')}; "
            f"run dir `{S['dir']}`"
        )

    (outdir / "comparison.md").write_text("\n".join(lines) + "\n")
    with open(outdir / "comparison.json", "w") as f:
        json.dump(dict(run_a=A["dir"], run_b=B["dir"], parameters=summary,
                       logZ_a=float(A["attrs"].get("logZ", np.nan)),
                       logZ_b=float(B["attrs"].get("logZ", np.nan))), f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
