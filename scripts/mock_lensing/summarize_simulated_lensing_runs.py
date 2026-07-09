#!/usr/bin/env python3
"""Aggregate multiple run_simulated_lensing_study.py workdirs (seeds) into
combined paper-ready CSVs.

Inputs: one or more study workdirs (globs allowed), each containing
validation_summary.json + the per-study CSVs. Outputs (under --outdir):

  combined_run_table.csv            profile/seed/case/run-kind status + sampler
                                    settings + graph scale + wall times
  combined_truth_recovery.csv       per seed x case edge/partition recovery
  combined_hyperparameter_recovery.csv  per seed x case x run x label with
                                    truth_in_90ci (coverage inputs)
  hyperparameter_coverage.csv       coverage fraction per run kind x label
  combined_evidence.csv             logZ_j2/off/singles + deltas per seed x case
  summary_metadata.json             command, timestamp, inputs, git commit

Every value is read from study outputs — nothing is recomputed.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _utc() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
    except Exception:
        return None


def _load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def collect_study(workdir: Path) -> dict[str, Any]:
    summary = _load_json(workdir / "validation_summary.json")
    if summary is None:
        return {}
    cfg = summary.get("resolved_config", {})
    study = cfg.get("study", {})
    inference = cfg.get("inference", {})
    mock = cfg.get("mock", {})
    meta = {
        "workdir": str(workdir),
        "profile": summary.get("profile"),
        "seed": study.get("seed"),
        "sampler": inference.get("sampler"),
        "nlive": inference.get("nlive"),
        "dlogz": inference.get("dlogz"),
        "pop_model": mock.get("pop_model", "powerlaw+peak"),
        "fix_population": inference.get("fix_population", True),
        "singleton_lensing": inference.get("singleton_lensing", "off"),
        "diagnostics_only": summary.get("diagnostics_only", False),
    }
    return {"summary": summary, "meta": meta}


def run_table_rows(meta: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case, rec in summary.get("cases", {}).items():
        graph = {
            **(rec.get("candidate_graph_summary") or {}),
            **{k: v for k, v in (rec.get("candidate_graph_audit") or {}).items() if v is not None},
        }
        kinds = [("j2", rec.get("j2") or {}), ("off", rec.get("off") or {})]
        if rec.get("singles_only"):
            kinds.append(("singles_only", rec["singles_only"]))
        for kind, sub in kinds:
            rows.append({
                **{k: meta[k] for k in ("profile", "seed", "sampler", "nlive", "dlogz",
                                        "pop_model", "fix_population", "singleton_lensing")},
                "case": case,
                "run": kind,
                "status": sub.get("status", rec.get("status")),
                "logZ": sub.get("logZ"),
                "logZerr": sub.get("logZerr"),
                "n_events": graph.get("n_events"),
                "n_candidate_edges": graph.get("n_candidate_edges"),
                "n_components": graph.get("n_components"),
                "approximate_total_partitions": graph.get("approximate_total_partitions"),
                "run_dir": sub.get("run_dir"),
            })
    return rows


def evidence_rows(meta: dict[str, Any], summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case, rec in summary.get("cases", {}).items():
        singles = rec.get("singles_only") or {}
        rows.append({
            "profile": meta["profile"], "seed": meta["seed"], "case": case,
            "logZ_j2": (rec.get("j2") or {}).get("logZ"),
            "logZ_off": (rec.get("off") or {}).get("logZ"),
            "logZ_singles": singles.get("logZ"),
            "delta_logZ_j2_minus_off": rec.get("delta_logZ_j2_minus_off"),
            "delta_logZerr": rec.get("delta_logZerr"),
            "diagnostics_only": meta["diagnostics_only"],
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="study workdirs (globs allowed)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)

    workdirs: list[Path] = []
    for pattern in args.runs:
        matches = sorted(glob.glob(pattern))
        workdirs.extend(Path(m) for m in matches if Path(m).is_dir())
    if not workdirs:
        print("no study workdirs matched", file=sys.stderr)
        return 1

    outdir = Path(args.outdir)
    run_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    hyper_rows: list[dict[str, Any]] = []
    evid_rows: list[dict[str, Any]] = []
    used: list[str] = []

    for wd in workdirs:
        study = collect_study(wd)
        if not study:
            print(f"skipping {wd}: no validation_summary.json", file=sys.stderr)
            continue
        used.append(str(wd))
        meta, summary = study["meta"], study["summary"]
        run_rows.extend(run_table_rows(meta, summary))
        evid_rows.extend(evidence_rows(meta, summary))
        for row in _read_csv(wd / "truth_recovery_summary.csv"):
            truth_rows.append({"profile": meta["profile"], "seed": meta["seed"], **row})
        for row in _read_csv(wd / "hyperparameter_recovery.csv"):
            hyper_rows.append({"profile": meta["profile"], "seed": meta["seed"], **row})

    _write_csv(outdir / "combined_run_table.csv", run_rows,
               ["profile", "seed", "case", "run", "status", "sampler", "nlive", "dlogz",
                "pop_model", "fix_population", "singleton_lensing", "logZ", "logZerr",
                "n_events", "n_candidate_edges", "n_components",
                "approximate_total_partitions", "run_dir"])
    _write_csv(outdir / "combined_truth_recovery.csv", truth_rows,
               ["profile", "seed", "case", "injected_n_pairs", "expected_n_pairs",
                "map_n_pairs", "true_edge_posterior_probability_mean",
                "false_edge_posterior_probability_max",
                "false_edge_posterior_probability_sum",
                "map_partition_exact_truth_match"])
    _write_csv(outdir / "combined_hyperparameter_recovery.csv", hyper_rows,
               ["profile", "seed", "case", "run", "label", "truth", "mean", "median",
                "q05", "q95", "truth_in_90ci"])
    _write_csv(outdir / "combined_evidence.csv", evid_rows,
               ["profile", "seed", "case", "logZ_j2", "logZ_off", "logZ_singles",
                "delta_logZ_j2_minus_off", "delta_logZerr", "diagnostics_only"])

    # Coverage: fraction of (seed, case) combos whose truth lies in the 90% CI.
    coverage: dict[tuple[str, str], list[int]] = {}
    for row in hyper_rows:
        flag = str(row.get("truth_in_90ci", "")).strip().lower()
        if flag in ("true", "false"):
            coverage.setdefault((row["run"], row["label"]), []).append(1 if flag == "true" else 0)
    cov_rows = [
        {"run": run, "label": label, "n": len(hits),
         "coverage_90": sum(hits) / len(hits)}
        for (run, label), hits in sorted(coverage.items())
    ]
    _write_csv(outdir / "hyperparameter_coverage.csv", cov_rows,
               ["run", "label", "n", "coverage_90"])

    (outdir / "summary_metadata.json").write_text(json.dumps({
        "created_at": _utc(),
        "command": " ".join(sys.argv),
        "inputs": used,
        "git_commit": _git_commit(),
        "n_run_rows": len(run_rows),
        "n_truth_rows": len(truth_rows),
        "n_hyper_rows": len(hyper_rows),
    }, indent=2) + "\n")
    print(f"combined {len(used)} studies -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
