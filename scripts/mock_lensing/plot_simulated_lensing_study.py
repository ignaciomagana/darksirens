#!/usr/bin/env python
"""Plot diagnostics from ``run_simulated_lensing_study.py`` outputs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np

REQUIRED_ARTIFACTS = [
    "validation_summary.json",
    "posterior_pair_probabilities.csv",
    "truth_recovery_summary.csv",
    "bias_summary.csv",
    "partition_component_summary.csv",
    "run_manifest.json",
]


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _float(value: Any, default: float = np.nan) -> float:
    if value in (None, "", "None"):
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _case_order(manifest: dict[str, Any], *row_sets: list[dict[str, Any]]) -> list[str]:
    cases = list((manifest.get("cases") or {}).keys())
    seen = set(cases)
    for rows in row_sets:
        for row in rows:
            case = str(row.get("case", ""))
            if case and case not in seen:
                cases.append(case); seen.add(case)
    return cases


def _candidate_edge_scores(study_dir: Path, cases: list[str]) -> dict[tuple[str, int, int], dict[str, Any]]:
    scores: dict[tuple[str, int, int], dict[str, Any]] = {}
    for case in cases:
        data = _load_json(study_dir / "cases" / case / "candidate_pairs.json", {}) or {}
        for edge in data.get("pairs", data.get("candidate_pairs", [])):
            try:
                i, j = sorted((int(edge["i"]), int(edge["j"])))
            except (KeyError, TypeError, ValueError):
                continue
            scores[(case, i, j)] = edge
    return scores


def _save(fig: Any, outdir: Path, name: str, fmt: str, show: bool) -> Path:
    path = outdir / f"{name}.{fmt}"
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)
    return path


def plot_pair_probabilities(ctx: dict[str, Any]) -> Path | None:
    rows = ctx["pair_rows"]
    if not rows:
        return None
    true = [_float(r.get("posterior_probability")) for r in rows if _bool(r.get("is_true_edge"))]
    false = [_float(r.get("posterior_probability")) for r in rows if not _bool(r.get("is_true_edge"))]
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [np.asarray(false, dtype=float), np.asarray(true, dtype=float)]
    ax.boxplot(data, showmeans=True)
    ax.set_xticks([1, 2], labels=["false edge", "true edge"])
    for x, vals in enumerate(data, start=1):
        if vals.size:
            jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size > 1 else np.array([0.0])
            ax.scatter(np.full(vals.size, x) + jitter, vals, alpha=0.7)
    ax.set_ylabel("posterior p_pair")
    ax.set_title("Pair probabilities by truth label")
    ax.set_ylim(bottom=0)
    return _save(fig, ctx["outdir"], "fig_pair_probabilities", ctx["format"], ctx["show"])


def plot_pair_prob_vs_edge_score(ctx: dict[str, Any]) -> Path | None:
    scores = _candidate_edge_scores(ctx["study_dir"], ctx["cases"])
    xs=[]; ys=[]; truth=[]
    for r in ctx["pair_rows"]:
        key = (r.get("case"), int(_float(r.get("i"), -1)), int(_float(r.get("j"), -1)))
        edge = scores.get(key)
        if not edge:
            continue
        x = _float(edge.get("log_prior_odds"))
        y = _float(r.get("posterior_probability"))
        if math.isfinite(x) and math.isfinite(y) and y >= ctx["min_edge_probability"]:
            xs.append(x); ys.append(y); truth.append(_bool(r.get("is_true_edge")))
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    truth_arr = np.asarray(truth)
    ax.scatter(np.asarray(xs)[~truth_arr], np.asarray(ys)[~truth_arr], label="false edge", alpha=0.7)
    ax.scatter(np.asarray(xs)[truth_arr], np.asarray(ys)[truth_arr], label="true edge", marker="x")
    ax.set_xlabel("candidate log_prior_odds")
    ax.set_ylabel("posterior p_pair")
    ax.set_title("Pair probability vs edge score")
    ax.legend()
    return _save(fig, ctx["outdir"], "fig_pair_prob_vs_edge_score", ctx["format"], ctx["show"])


def plot_evidence_matrix(ctx: dict[str, Any]) -> Path | None:
    summary = ctx["summary"]
    vals=[]; labels=[]
    for case, rec in (summary.get("cases") or {}).items():
        val = _float(rec.get("delta_logZ_j2_minus_off", rec.get("logZ")))
        if not math.isfinite(val):
            val = _float((rec.get("results_attrs") or {}).get("logZ"))
        if math.isfinite(val):
            labels.append(case); vals.append(val)
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(vals)), 4))
    ax.imshow(np.asarray(vals, dtype=float)[None, :], aspect="auto")
    ax.set_yticks([0], labels=["logZ / ΔlogZ"])
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_title("Available evidence diagnostics")
    for k, val in enumerate(vals):
        ax.text(k, 0, f"{val:.2g}", ha="center", va="center")
    return _save(fig, ctx["outdir"], "fig_evidence_matrix", ctx["format"], ctx["show"])


def plot_lens_rate_recovery(ctx: dict[str, Any]) -> Path | None:
    labels=[]; med=[]; lo=[]; hi=[]
    for case, rec in (ctx["summary"].get("cases") or {}).items():
        s = rec.get("lens_rate_posterior_summary") or (rec.get("posterior_summary") or {}).get("log10_tau_A") or {}
        m = _float(s.get("median", s.get("mean")))
        if math.isfinite(m):
            labels.append(case); med.append(m); lo.append(_float(s.get("q05"), m)); hi.append(_float(s.get("q95"), m))
    if not labels:
        return None
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(labels))))
    ax.errorbar(med, y, xerr=[np.asarray(med)-np.asarray(lo), np.asarray(hi)-np.asarray(med)], fmt="o")
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("log10_tau_A posterior summary")
    ax.set_title("Lens-rate recovery")
    return _save(fig, ctx["outdir"], "fig_lens_rate_recovery", ctx["format"], ctx["show"])


def plot_candidate_graph_summary(ctx: dict[str, Any]) -> Path | None:
    rows = ctx["component_rows"]
    if not rows:
        return None
    labels=[r["case"] for r in rows]
    edges=[_float(r.get("n_candidate_edges"), 0) for r in rows]
    parts=[_float(r.get("map_n_pairs"), 0) for r in rows]
    x=np.arange(len(labels)); width=0.38
    fig, ax = plt.subplots(figsize=(max(7, 0.55*len(labels)), 4))
    ax.bar(x-width/2, edges, width, label="candidate edges")
    ax.bar(x+width/2, parts, width, label="MAP pairs")
    ax.set_xticks(x, labels=labels, rotation=45, ha="right")
    ax.set_ylabel("count")
    ax.set_title("Candidate graph summary")
    ax.legend()
    return _save(fig, ctx["outdir"], "fig_candidate_graph_summary", ctx["format"], ctx["show"])


def plot_ablation_summary(ctx: dict[str, Any]) -> Path | None:
    rows = ctx["truth_rows"]
    pick = {"G_": "full marks", "E_": "no sky", "F_": "no time", "D_": "bad p_tag"}
    vals=[]; labels=[]
    for prefix, label in pick.items():
        for r in rows:
            if str(r.get("case", "")).startswith(prefix):
                vals.append(_float(r.get("true_edge_posterior_probability_mean"))); labels.append(label); break
    if not vals or not any(math.isfinite(v) for v in vals):
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals)
    ax.set_ylabel("mean true-edge p_pair")
    ax.set_title("Ablation summary")
    return _save(fig, ctx["outdir"], "fig_ablation_summary", ctx["format"], ctx["show"])


def plot_false_positive_summary(ctx: dict[str, Any]) -> Path | None:
    rows = ctx["truth_rows"]
    if not rows:
        return None
    labels=[r["case"] for r in rows]
    maxv=[_float(r.get("false_edge_posterior_probability_max"), 0) for r in rows]
    sumv=[_float(r.get("false_edge_posterior_probability_sum"), 0) for r in rows]
    x=np.arange(len(labels)); width=0.38
    fig, ax = plt.subplots(figsize=(max(7, 0.55*len(labels)), 4))
    ax.bar(x-width/2, maxv, width, label="max false p")
    ax.bar(x+width/2, sumv, width, label="sum false p")
    ax.set_xticks(x, labels=labels, rotation=45, ha="right")
    ax.set_ylabel("posterior probability")
    ax.set_title("False-positive edge summary")
    ax.legend()
    return _save(fig, ctx["outdir"], "fig_false_positive_summary", ctx["format"], ctx["show"])


PLOTS: dict[str, Callable[[dict[str, Any]], Path | None]] = {
    "fig_pair_probabilities": plot_pair_probabilities,
    "fig_pair_prob_vs_edge_score": plot_pair_prob_vs_edge_score,
    "fig_evidence_matrix": plot_evidence_matrix,
    "fig_lens_rate_recovery": plot_lens_rate_recovery,
    "fig_candidate_graph_summary": plot_candidate_graph_summary,
    "fig_ablation_summary": plot_ablation_summary,
    "fig_false_positive_summary": plot_false_positive_summary,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study_dir", required=True)
    ap.add_argument("--outdir")
    ap.add_argument("--format", choices=["png", "pdf"], default="png")
    ap.add_argument("--show", type=_str2bool, default=False)
    ap.add_argument("--min_edge_probability", type=float, default=0.0)
    args = ap.parse_args(argv)

    study_dir = Path(args.study_dir).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else study_dir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    warnings = []
    for name in REQUIRED_ARTIFACTS:
        if not (study_dir / name).exists():
            warnings.append(f"missing artifact: {name}")
    summary = _load_json(study_dir / "validation_summary.json", {}) or {}
    manifest = _load_json(study_dir / "run_manifest.json", {}) or {}
    pair_rows = _read_csv(study_dir / "posterior_pair_probabilities.csv")
    truth_rows = _read_csv(study_dir / "truth_recovery_summary.csv")
    bias_rows = _read_csv(study_dir / "bias_summary.csv")
    component_rows = _read_csv(study_dir / "partition_component_summary.csv")
    cases = _case_order(manifest, pair_rows, truth_rows, bias_rows, component_rows)
    if summary.get("diagnostics_only"):
        warnings.append("diagnostic-only mode: sampler evidence plots may be skipped or non-science diagnostics")

    ctx = dict(study_dir=study_dir, outdir=outdir, format=args.format, show=args.show, min_edge_probability=args.min_edge_probability, summary=summary, manifest=manifest, pair_rows=pair_rows, truth_rows=truth_rows, bias_rows=bias_rows, component_rows=component_rows, cases=cases)
    produced=[]; skipped=[]
    for name, func in PLOTS.items():
        try:
            path = func(ctx)
        except Exception as exc:  # keep optional plotting failures isolated
            warnings.append(f"{name}: {exc}")
            path = None
        if path is None:
            skipped.append({"name": name, "reason": "required inputs unavailable"})
        else:
            produced.append({"name": name, "path": str(path)})
    manifest_out = {"study_dir": str(study_dir), "outdir": str(outdir), "format": args.format, "produced": produced, "skipped": skipped, "warnings": warnings}
    (outdir / "plot_manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
