#!/usr/bin/env python
"""Opt-in evidence/recovery validation for spectral-siren J=2 lensing mocks.

This runner is intentionally heavier than ``run_lensing_validation.py``: it
launches real samplers and compares evidence/log-likelihood diagnostics across a
small validation matrix.  It remains mock-only and is not a production LVK
science workflow.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "mock_lensing" / "generate_mock_lensing.py"

PROFILES: dict[str, dict[str, int]] = {
    "tiny_evidence": dict(n_universe=4000, n_sing=2, n_pair=2, nsamp=56, n_unlensed_inj=1200, n_lensed_inj=1200, pe_max=56, default_nlive=40),
    "small_evidence": dict(n_universe=10000, n_sing=4, n_pair=3, nsamp=96, n_unlensed_inj=3000, n_lensed_inj=3500, pe_max=96, default_nlive=80),
}


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def _run(cmd: list[str], *, cwd: Path = ROOT) -> float:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    start = time.time()
    subprocess.run(cmd, cwd=cwd, check=True)
    return time.time() - start


def _mock_ready(mock_dir: Path, *, n_pair_keep: int) -> bool:
    needed = ["mock_gw_pe.h5", "mock_gw_selection.h5", "mock_lensed_injections.h5", "mock_pair_pe.h5", "partition.json", "candidate_pairs.json"]
    if not all((mock_dir / p).exists() for p in needed):
        return False
    try:
        part = json.loads((mock_dir / "partition.json").read_text())
        return len(part.get("pair_indices", [])) == n_pair_keep
    except Exception:
        return False


def _generate_mock(mock_dir: Path, cfg: dict[str, int], *, seed: int, n_pair_keep: int, reuse: bool, conditioning: str = "fixed_counts") -> list[str]:
    cmd = [
        sys.executable, str(GEN), "--outdir", str(mock_dir), "--conditioning", conditioning,
        "--n-universe", str(cfg["n_universe"]), "--n-sing-keep", str(cfg["n_sing"]),
        "--n-pair-keep", str(n_pair_keep), "--max-sing-keep", str(cfg["n_sing"]),
        "--max-pair-keep", str(n_pair_keep), "--nsamp", str(cfg["nsamp"]),
        "--n-unlensed-inj", str(cfg["n_unlensed_inj"]), "--n-lensed-inj", str(cfg["n_lensed_inj"]),
        "--seed", str(seed),
    ]
    if reuse and _mock_ready(mock_dir, n_pair_keep=n_pair_keep):
        print(f"[evidence-validation] reusing mock {mock_dir}", flush=True)
        return cmd
    if mock_dir.exists():
        shutil.rmtree(mock_dir)
    mock_dir.mkdir(parents=True, exist_ok=True)
    _run(cmd)
    return cmd


def _write_wrong_partition(mock_dir: Path, out_path: Path) -> None:
    part = json.loads((mock_dir / "partition.json").read_text())
    pairs = [list(p) for p in part["pair_indices"]]
    if len(pairs) < 2:
        raise RuntimeError("wrong-partner validation requires at least two true pairs")
    seconds = [p[1] for p in pairs]
    part["pair_indices"] = [[p[0], q] for p, q in zip(pairs, seconds[1:] + seconds[:1])]
    out_path.write_text(json.dumps(part, indent=2) + "\n")


def _latest_run_dir(save_root: Path) -> Path:
    diags = sorted(save_root.glob("**/diagnostics.json"), key=lambda p: p.stat().st_mtime)
    if not diags:
        raise RuntimeError(f"no diagnostics.json found under {save_root}")
    return diags[-1].parent


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _collect_run(save_root: Path, *, runtime_s: float, command: list[str]) -> dict[str, Any]:
    run_dir = _latest_run_dir(save_root)
    diag_path = run_dir / "diagnostics.json"
    diagnostics = json.loads(diag_path.read_text())
    attrs: dict[str, Any] = {}
    posterior_summary: dict[str, Any] = {}
    results_path = run_dir / "results.hdf5"
    if results_path.exists():
        with h5py.File(results_path, "r") as f:
            attrs = {k: _jsonable(v) for k, v in f.attrs.items()}
            labels = json.loads(attrs.get("labels", "[]")) if isinstance(attrs.get("labels"), str) else []
            samples = np.asarray(f["samples"]) if "samples" in f else np.empty((0, 0))
            for label in ["log10_tau_A", "tau_n"]:
                if label in labels and samples.size:
                    col = samples[:, labels.index(label)]
                    posterior_summary[label] = {"mean": float(np.mean(col)), "median": float(np.median(col)), "q05": float(np.quantile(col, 0.05)), "q95": float(np.quantile(col, 0.95))}
    return {
        "run_dir": str(run_dir), "diagnostics_path": str(diag_path), "results_path": str(results_path) if results_path.exists() else None,
        "diagnostics": diagnostics, "results_attrs": attrs, "logZ": attrs.get("logZ"), "logZerr": attrs.get("logZerr"),
        "labels": json.loads(attrs.get("labels", "[]")) if isinstance(attrs.get("labels"), str) else attrs.get("labels"),
        "posterior_summary": posterior_summary, "runtime_s": runtime_s, "command": command,
    }


def _cli_cmd(mock_dir: Path, save_root: Path, *, cluster_mode: str, partition: Path | None, sampler: str, nlive: int, dlogz: float, seed: int, cfg: dict[str, int], pair_batch_size: int = 0, partition_mode: str = "fixed", use_unified_observed_catalog: bool = False, fix_lens_rate: bool = True) -> list[str]:
    cmd = [sys.executable, "-m", "darksirens.cli.inference_lensing", "--gw_path", str(mock_dir / ("mock_observed_gw_pe.h5" if use_unified_observed_catalog else "mock_gw_pe.h5")), "--gwselection_path", str(mock_dir / "mock_gw_selection.h5"), "--wl_backend", "lognormal", "--pop_model", "powerlaw+peak", "--fix_cosmology", "true", "--fix_survey", "true", "--fix_population", "true", "--fix_lens_rate", str(fix_lens_rate).lower(), "--sampler", sampler, "--nlive", str(nlive), "--dlogz", str(dlogz), "--max_samples", "5000", "--pe_max_per_pair", str(cfg["pe_max"]), "--pair_batch_size", str(pair_batch_size), "--seed", str(seed), "--cluster_mode", cluster_mode, "--save_path", str(save_root)]
    if not fix_lens_rate:
        cmd += ["--fixed_parameter_values", '{"tau_n": 3.0}', "--lens_prior_overrides", '{"log10_tau_A": [-5.0, -2.5]}']
    if cluster_mode == "j2":
        cmd += ["--lensed_injections_path", str(mock_dir / "mock_lensed_injections.h5"), "--pair_pe_path", str(mock_dir / "mock_pair_pe.h5"), "--partition_mode", partition_mode]
        if partition_mode == "marginalize_exact":
            cmd += ["--candidate_pairs_path", str(mock_dir / "candidate_pairs.json")]
        else:
            cmd += ["--partition_path", str(partition or mock_dir / "partition.json")]
    return cmd


def _run_case(name: str, cmd: list[str], save_root: Path) -> dict[str, Any]:
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    runtime = _run(cmd)
    return _collect_run(save_root, runtime_s=runtime, command=cmd)


def _finite_diag(run: dict[str, Any]) -> bool:
    diag = run["diagnostics"]
    return all(math.isfinite(float(diag[k])) for k in ["logL_total", "singleton_logL_sum", "pair_logL_sum", "selection_correction_total"] if k in diag)


def _diff(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str) -> float | None:
    if not a or not b:
        return None
    av = a.get("logZ") if key == "logZ" else a["diagnostics"].get(key)
    bv = b.get("logZ") if key == "logZ" else b["diagnostics"].get(key)
    if av is None or bv is None:
        return None
    return float(av) - float(bv)


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [f"# Lensing evidence validation ({summary['profile']})", "", "Mock-only validation; not an LVK science run.", "", "## Checks", ""]
    for k, v in summary["checks"].items():
        lines.append(f"- {'PASS' if v else 'FAIL'} `{k}`")
    lines += ["", "## Runs", ""]
    for name, run in summary["runs"].items():
        diag = run.get("diagnostics", {})
        lines.append(f"- `{name}`: logZ={run.get('logZ')} logL_total={diag.get('logL_total', diag.get('logL_marginalized'))} pair_logL_sum={diag.get('pair_logL_sum', diag.get('map_pair_logL_sum'))} n_pairs={diag.get('n_pairs', diag.get('expected_n_pairs'))} runtime_s={run.get('runtime_s'):.1f}")
    path.write_text("\n".join(lines) + "\n")


def build_plan(args: argparse.Namespace, cfg: dict[str, int], work: Path) -> dict[str, Any]:
    mock = work / f"mock_{args.profile}"
    null_mock = work / f"mock_{args.profile}_null"
    runs = work / "runs"
    nlive = args.nlive if args.nlive is not None else cfg["default_nlive"]
    cases = {
        "off_true_catalog": _cli_cmd(mock, runs / "off_true_catalog", cluster_mode="off", partition=None, sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_fixed_true": _cli_cmd(mock, runs / "j2_fixed_true", cluster_mode="j2", partition=mock / "partition.json", sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_fixed_wrong": _cli_cmd(mock, runs / "j2_fixed_wrong", cluster_mode="j2", partition=work / "wrong_partition.json", sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_null": _cli_cmd(null_mock, runs / "j2_null", cluster_mode="j2", partition=null_mock / "partition.json", sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg),
        "off_null": _cli_cmd(null_mock, runs / "off_null", cluster_mode="off", partition=None, sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg),
        "j2_marginalized": _cli_cmd(mock, runs / "j2_marginalized", cluster_mode="j2", partition=None, sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, partition_mode="marginalize_exact", use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_batched": _cli_cmd(mock, runs / "j2_batched", cluster_mode="j2", partition=mock / "partition.json", sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, pair_batch_size=max(1, args.pair_batch_size), use_unified_observed_catalog=args.use_unified_observed_catalog),
    }
    if args.run_lens_rate_recovery:
        rate_mock = work / f"mock_{args.profile}_poisson_rate"
        cases["lens_rate_recovery"] = _cli_cmd(rate_mock, runs / "lens_rate_recovery", cluster_mode="j2", partition=rate_mock / "partition.json", sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, fix_lens_rate=False)
    return {"profile": args.profile, "workdir": str(work), "mock_commands": {"true": [sys.executable, str(GEN), "--outdir", str(mock), "--conditioning", "fixed_counts", "--n-universe", str(cfg["n_universe"]), "--n-sing-keep", str(cfg["n_sing"]), "--n-pair-keep", str(cfg["n_pair"]), "--nsamp", str(cfg["nsamp"]), "--n-unlensed-inj", str(cfg["n_unlensed_inj"]), "--n-lensed-inj", str(cfg["n_lensed_inj"]), "--seed", str(args.seed)], "null": [sys.executable, str(GEN), "--outdir", str(null_mock), "--conditioning", "fixed_counts", "--n-universe", str(cfg["n_universe"]), "--n-sing-keep", str(cfg["n_sing"]), "--n-pair-keep", "0", "--nsamp", str(cfg["nsamp"]), "--n-unlensed-inj", str(cfg["n_unlensed_inj"]), "--n-lensed-inj", str(cfg["n_lensed_inj"]), "--seed", str(args.seed + 17)]}, "cases": cases}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=PROFILES, default="tiny_evidence")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sampler", choices=["dynesty", "tinyns"], default="dynesty")
    ap.add_argument("--nlive", type=int, default=None)
    ap.add_argument("--dlogz", type=float, default=10.0)
    ap.add_argument("--pair_batch_size", type=int, default=8)
    ap.add_argument("--use_unified_observed_catalog", type=_str2bool, default=False)
    ap.add_argument("--run_lens_rate_recovery", type=_str2bool, default=False)
    ap.add_argument("--dry_run", type=_str2bool, default=False)
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args(argv)

    cfg = PROFILES[args.profile]
    work = Path(args.workdir).resolve(); work.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        plan = build_plan(args, cfg, work)
        (work / "validation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2))
        return 0

    mock = work / f"mock_{args.profile}"; null_mock = work / f"mock_{args.profile}_null"; runs_root = work / "runs"
    _generate_mock(mock, cfg, seed=args.seed, n_pair_keep=cfg["n_pair"], reuse=args.reuse)
    _write_wrong_partition(mock, work / "wrong_partition.json")
    _generate_mock(null_mock, cfg, seed=args.seed + 17, n_pair_keep=0, reuse=args.reuse)
    nlive = args.nlive if args.nlive is not None else cfg["default_nlive"]

    runs: dict[str, Any] = {}
    def cmd(case, m, root, **kw): return _cli_cmd(m, runs_root / root, sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, **kw)
    cases = {
        "off_true_catalog": cmd("off_true_catalog", mock, "off_true_catalog", cluster_mode="off", partition=None, use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_fixed_true": cmd("j2_fixed_true", mock, "j2_fixed_true", cluster_mode="j2", partition=mock / "partition.json", use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_fixed_wrong": cmd("j2_fixed_wrong", mock, "j2_fixed_wrong", cluster_mode="j2", partition=work / "wrong_partition.json", use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_null": cmd("j2_null", null_mock, "j2_null", cluster_mode="j2", partition=null_mock / "partition.json"),
        "off_null": cmd("off_null", null_mock, "off_null", cluster_mode="off", partition=None),
        "j2_batched": cmd("j2_batched", mock, "j2_batched", cluster_mode="j2", partition=mock / "partition.json", pair_batch_size=max(1, args.pair_batch_size), use_unified_observed_catalog=args.use_unified_observed_catalog),
    }
    if (mock / "candidate_pairs.json").exists():
        cases["j2_marginalized"] = cmd("j2_marginalized", mock, "j2_marginalized", cluster_mode="j2", partition=None, partition_mode="marginalize_exact", use_unified_observed_catalog=args.use_unified_observed_catalog)
    if args.run_lens_rate_recovery:
        rate_mock = work / f"mock_{args.profile}_poisson_rate"
        _generate_mock(rate_mock, cfg, seed=args.seed + 31, n_pair_keep=cfg["n_pair"], reuse=args.reuse, conditioning="poisson_counts")
        cases["lens_rate_recovery"] = _cli_cmd(rate_mock, runs_root / "lens_rate_recovery", cluster_mode="j2", partition=rate_mock / "partition.json", sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, fix_lens_rate=False)
    for name, c in cases.items():
        runs[name] = _run_case(name, c, runs_root / name)

    checks = {"all_requested_runs_completed": set(cases) <= set(runs), "all_diagnostics_finite_where_expected": all(_finite_diag(r) for r in runs.values()), "true_pair_logL_sum_exceeds_wrong": float(runs["j2_fixed_true"]["diagnostics"].get("pair_logL_sum", -math.inf)) > float(runs["j2_fixed_wrong"]["diagnostics"].get("pair_logL_sum", math.inf)), "null_mock_reports_n_pairs_zero": int(runs["j2_null"]["diagnostics"].get("n_pairs", -1)) == 0, "off_mode_reports_n_pairs_zero": int(runs["off_true_catalog"]["diagnostics"].get("n_pairs", -1)) == 0, "batched_unbatched_logL_total_agree": abs(float(runs["j2_batched"]["diagnostics"].get("logL_total")) - float(runs["j2_fixed_true"]["diagnostics"].get("logL_total"))) < 1e-6}
    if runs["j2_fixed_true"].get("logZ") is not None and runs["j2_fixed_wrong"].get("logZ") is not None:
        checks["logZ_true_exceeds_wrong"] = float(runs["j2_fixed_true"]["logZ"]) > float(runs["j2_fixed_wrong"]["logZ"])
    if runs["j2_null"].get("logZ") is not None and runs["off_null"].get("logZ") is not None:
        checks["null_j2_does_not_strongly_beat_off"] = float(runs["j2_null"]["logZ"]) - float(runs["off_null"]["logZ"]) < 5.0
    if "j2_marginalized" in runs:
        md = runs["j2_marginalized"]["diagnostics"]
        ma = runs["j2_marginalized"].get("results_attrs", {})
        checks.update({
            "marginalized_logL_finite": math.isfinite(float(md.get("logL_marginalized", md.get("logL_total")))),
            "marginalized_posterior_pair_probabilities_present": bool(md.get("posterior_pair_probabilities")),
            "marginalized_expected_n_pairs_finite": math.isfinite(float(md.get("expected_n_pairs"))),
            "marginalized_results_partition_mode_attr": ma.get("partition_mode") == "marginalize_exact",
            "marginalized_results_expected_n_pairs_attr": "expected_n_pairs" in ma,
            "marginalized_results_reference_partition_n_pairs_attr": "reference_partition_n_pairs" in ma,
            "marginalized_results_map_partition_n_pairs_attr": "map_partition_n_pairs" in ma,
        })
    if "lens_rate_recovery" in runs:
        posterior = runs["lens_rate_recovery"].get("posterior_summary", {}).get("log10_tau_A", {})
        injected = math.log10(5.0e-4)
        checks["lens_rate_recovery_log10_tau_A_summary_present"] = bool(posterior)
        if posterior:
            checks["lens_rate_recovery_injected_value_in_wide_interval"] = float(posterior.get("q05", -math.inf)) <= injected <= float(posterior.get("q95", math.inf))
    summary = {"profile": args.profile, "checks": checks, "runs": runs, "evidence_differences": {"j2_fixed_true_minus_off": _diff(runs.get("j2_fixed_true"), runs.get("off_true_catalog"), "logZ"), "j2_fixed_wrong_minus_true": _diff(runs.get("j2_fixed_wrong"), runs.get("j2_fixed_true"), "logZ"), "j2_null_minus_off_null": _diff(runs.get("j2_null"), runs.get("off_null"), "logZ"), "j2_marginalized_minus_fixed_true": _diff(runs.get("j2_marginalized"), runs.get("j2_fixed_true"), "logZ")}}
    (work / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    _write_markdown(work / "validation_summary.md", summary)
    for k, v in checks.items(): print(f"{'PASS' if v else 'FAIL'} {k}")
    return 0 if all(checks.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
