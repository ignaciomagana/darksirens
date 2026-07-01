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
import datetime
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


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


def _run(cmd: list[str], *, cwd: Path = ROOT) -> tuple[float, int]:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    start = time.time()
    completed = subprocess.run(cmd, cwd=cwd)
    return time.time() - start, int(completed.returncode)


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
    runtime, return_code = _run(cmd)
    if return_code != 0:
        raise RuntimeError(f"mock generation failed with return code {return_code}: {cmd}")
    return cmd


def _write_wrong_partition(mock_dir: Path, out_path: Path) -> list[list[int]]:
    part = json.loads((mock_dir / "partition.json").read_text())
    pairs = [list(p) for p in part["pair_indices"]]
    if len(pairs) < 2:
        raise RuntimeError("wrong-partner validation requires at least two true pairs")
    seconds = [p[1] for p in pairs]
    wrong_pairs = [[p[0], q] for p, q in zip(pairs, seconds[1:] + seconds[:1])]
    part["pair_indices"] = wrong_pairs
    out_path.write_text(json.dumps(part, indent=2) + "\n")
    return wrong_pairs


def _write_wrong_pair_pe(mock_dir: Path, out_path: Path, wrong_pairs: list[list[int]]) -> None:
    shutil.copy2(mock_dir / "mock_pair_pe.h5", out_path)
    with h5py.File(out_path, "r+") as f:
        for k, pair in enumerate(wrong_pairs):
            g = f[f"pair_{k}"]
            g.attrs["event_index_image0"] = int(pair[0])
            g.attrs["event_index_image1"] = int(pair[1])


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


def _cli_cmd(mock_dir: Path, save_root: Path, *, cluster_mode: str, partition: Path | None, sampler: str, nlive: int, dlogz: float, seed: int, cfg: dict[str, int], pair_batch_size: int = 0, partition_mode: str = "fixed", use_unified_observed_catalog: bool = False, fix_lens_rate: bool = True, pair_pe_path: Path | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "darksirens.cli.inference_lensing", "--gw_path", str(mock_dir / ("mock_observed_gw_pe.h5" if use_unified_observed_catalog else "mock_gw_pe.h5")), "--gwselection_path", str(mock_dir / "mock_gw_selection.h5"), "--wl_backend", "lognormal", "--pop_model", "powerlaw+peak", "--fix_cosmology", "true", "--fix_survey", "true", "--fix_population", "true", "--fix_lens_rate", str(fix_lens_rate).lower(), "--sampler", sampler, "--nlive", str(nlive), "--dlogz", str(dlogz), "--max_samples", "5000", "--pe_max_per_pair", str(cfg["pe_max"]), "--pair_batch_size", str(pair_batch_size), "--seed", str(seed), "--cluster_mode", cluster_mode, "--save_path", str(save_root)]
    if not fix_lens_rate:
        cmd += ["--fixed_parameter_values", '{"tau_n": 3.0}', "--lens_prior_overrides", '{"log10_tau_A": [-5.0, -2.5]}']
    if cluster_mode == "j2":
        cmd += ["--lensed_injections_path", str(mock_dir / "mock_lensed_injections.h5"), "--partition_mode", partition_mode]
        if (not use_unified_observed_catalog) or pair_pe_path is not None:
            cmd += ["--pair_pe_path", str(pair_pe_path or mock_dir / "mock_pair_pe.h5")]
        if partition_mode == "marginalize_exact":
            # Time-delay marks are currently fixed-partition only; exact
            # marginalization must remain on pair_marks=none until candidate-edge
            # time metadata is implemented.
            cmd += ["--candidate_pairs_path", str(mock_dir / "candidate_pairs.json"), "--pair_marks", "none"]
        else:
            cmd += ["--partition_path", str(partition or mock_dir / "partition.json")]
    return cmd


def _preflight_command(cmd: list[str], preflight_json: Path) -> list[str]:
    return [*cmd, "--preflight_only", "true", "--preflight_json", str(preflight_json)]


def _diagnostics_only_command(cmd: list[str]) -> list[str]:
    out = list(cmd)
    def set_arg(flag: str, value: str) -> None:
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out.extend([flag, value])
    set_arg("--max_samples", "0")
    set_arg("--nlive", "8")
    set_arg("--dlogz", "50")
    return out


def _empty_case_record(cmd: list[str], preflight_cmd: list[str] | None, preflight_json: Path | None, started_at: str) -> dict[str, Any]:
    return {
        "command": cmd,
        "preflight_command": preflight_cmd,
        "preflight_json": str(preflight_json) if preflight_json else None,
        "preflight_report": None,
        "started_at": started_at,
        "finished_at": None,
        "runtime_s": None,
        "return_code": None,
        "status": "missing_outputs",
        "run_dir": None,
        "diagnostics_path": None,
        "results_path": None,
        "diagnostics": {},
        "results_attrs": {},
        "logZ": None,
        "logZerr": None,
        "labels": None,
        "posterior_summary": {},
    }


def _run_case(name: str, cmd: list[str], save_root: Path, *, skip_preflight: bool = False) -> dict[str, Any]:
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    preflight_json = save_root / "preflight.json"
    pre_cmd = None if skip_preflight else _preflight_command(cmd, preflight_json)
    record = _empty_case_record(cmd, pre_cmd, None if skip_preflight else preflight_json, started_at)
    if not skip_preflight:
        pre_runtime, pre_rc = _run(pre_cmd)
        if preflight_json.exists():
            try:
                record["preflight_report"] = json.loads(preflight_json.read_text())
            except Exception as exc:
                record["preflight_report"] = {"ok": False, "errors": [f"failed to read preflight JSON: {exc}"]}
        if pre_rc != 0 or not (record.get("preflight_report") or {}).get("ok", False):
            record.update({"finished_at": _utc_now(), "runtime_s": pre_runtime, "return_code": pre_rc, "status": "failed_preflight"})
            return record
    runtime, return_code = _run(cmd)
    record.update({"finished_at": _utc_now(), "runtime_s": runtime, "return_code": return_code})
    if return_code != 0:
        record["status"] = "failed_sampler"
        return record
    try:
        collected = _collect_run(save_root, runtime_s=runtime, command=cmd)
        record.update(collected)
        record["finished_at"] = record["finished_at"] or _utc_now()
        record["return_code"] = return_code
        record["status"] = "passed" if record.get("diagnostics_path") and record.get("results_path") else "missing_outputs"
    except Exception as exc:
        record["status"] = "missing_outputs"
        record["collection_error"] = str(exc)
    return record


def _finite_diag(run: dict[str, Any]) -> bool:
    diag = run["diagnostics"]
    return all(math.isfinite(float(diag[k])) for k in ["logL_total", "singleton_logL_sum", "pair_logL_sum", "selection_correction_total"] if k in diag)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _diff(a: dict[str, Any] | None, b: dict[str, Any] | None, key: str) -> float | None:
    if not a or not b:
        return None
    av = _safe_float(a.get("logZ") if key == "logZ" else a.get("diagnostics", {}).get(key))
    bv = _safe_float(b.get("logZ") if key == "logZ" else b.get("diagnostics", {}).get(key))
    if av is None or bv is None or not (math.isfinite(av) and math.isfinite(bv)):
        return None
    return av - bv


def _posterior_pair_probability_check(diagnostics: dict[str, Any], *, tol: float = 1e-6) -> tuple[bool, str | None]:
    probs = diagnostics.get("posterior_pair_probabilities")
    if not probs:
        return False, "posterior_pair_probabilities missing"
    vals = np.asarray(list(probs.values()) if isinstance(probs, dict) else probs, dtype=float)
    if vals.size == 0 or not np.all(np.isfinite(vals)):
        return False, "posterior_pair_probabilities are empty or non-finite"
    if not np.all((vals >= 0.0) & (vals <= 1.0)):
        return False, "posterior_pair_probabilities outside [0, 1]"
    expected = diagnostics.get("expected_n_pairs")
    if expected is None or not math.isfinite(float(expected)):
        return False, "expected_n_pairs missing or non-finite"
    if abs(float(np.sum(vals)) - float(expected)) > tol:
        return False, "posterior_pair_probabilities sum does not match expected_n_pairs"
    return True, None


def _format_status_object(name: str, run: dict[str, Any]) -> dict[str, Any]:
    diag = run.get("diagnostics", {}) or {}
    return {
        "case": name,
        "status": run.get("status", "missing_outputs"),
        "logZ": run.get("logZ"),
        "logZerr": run.get("logZerr"),
        "logL_total": diag.get("logL_total", diag.get("logL_marginalized")),
        "pair_logL_sum": diag.get("pair_logL_sum", diag.get("map_pair_logL_sum")),
        "n_pairs": diag.get("n_pairs", diag.get("expected_n_pairs")),
        "runtime_s": run.get("runtime_s"),
        "run_dir": run.get("run_dir"),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [f"# Lensing evidence validation ({summary['profile']})", "", "Mock-only validation; not an LVK science run.", "", "## Checks", ""]
    for k, v in summary["checks"].items():
        lines.append(f"- {'PASS' if v else 'FAIL'} `{k}`")
    lines += ["", "## Runs", "", "| case | status | logZ | logZerr | logL_total | pair_logL_sum or map_pair_logL_sum | n_pairs or expected_n_pairs | runtime_s | run_dir |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, run in summary["runs"].items():
        row = _format_status_object(name, run)
        runtime = "" if row["runtime_s"] is None else f"{float(row['runtime_s']):.1f}"
        lines.append(f"| {row['case']} | {row['status']} | {row['logZ']} | {row['logZerr']} | {row['logL_total']} | {row['pair_logL_sum']} | {row['n_pairs']} | {runtime} | {row['run_dir']} |")
    path.write_text("\n".join(lines) + "\n")


def build_plan(args: argparse.Namespace, cfg: dict[str, int], work: Path) -> dict[str, Any]:
    mock = work / f"mock_{args.profile}"
    null_mock = work / f"mock_{args.profile}_null"
    runs = work / "runs"
    nlive = args.nlive if args.nlive is not None else cfg["default_nlive"]
    cases = {
        "off_true_catalog": _cli_cmd(mock, runs / "off_true_catalog", cluster_mode="off", partition=None, sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_fixed_true": _cli_cmd(mock, runs / "j2_fixed_true", cluster_mode="j2", partition=mock / "partition.json", sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_fixed_wrong": _cli_cmd(mock, runs / "j2_fixed_wrong", cluster_mode="j2", partition=work / "wrong_partition.json", sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, use_unified_observed_catalog=args.use_unified_observed_catalog, pair_pe_path=(None if args.use_unified_observed_catalog else work / "wrong_pair_pe.h5")),
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
    ap.add_argument("--diagnostics_only", type=_str2bool, default=False)
    ap.add_argument("--skip_preflight", type=_str2bool, default=False)
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
    wrong_pairs = _write_wrong_partition(mock, work / "wrong_partition.json")
    if not args.use_unified_observed_catalog:
        _write_wrong_pair_pe(mock, work / "wrong_pair_pe.h5", wrong_pairs)
    _generate_mock(null_mock, cfg, seed=args.seed + 17, n_pair_keep=0, reuse=args.reuse)
    nlive = args.nlive if args.nlive is not None else cfg["default_nlive"]

    runs: dict[str, Any] = {}
    def cmd(case, m, root, **kw): return _cli_cmd(m, runs_root / root, sampler=args.sampler, nlive=nlive, dlogz=args.dlogz, seed=args.seed, cfg=cfg, **kw)
    cases = {
        "off_true_catalog": cmd("off_true_catalog", mock, "off_true_catalog", cluster_mode="off", partition=None, use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_fixed_true": cmd("j2_fixed_true", mock, "j2_fixed_true", cluster_mode="j2", partition=mock / "partition.json", use_unified_observed_catalog=args.use_unified_observed_catalog),
        "j2_fixed_wrong": cmd("j2_fixed_wrong", mock, "j2_fixed_wrong", cluster_mode="j2", partition=work / "wrong_partition.json", use_unified_observed_catalog=args.use_unified_observed_catalog, pair_pe_path=(None if args.use_unified_observed_catalog else work / "wrong_pair_pe.h5")),
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
        case_cmd = _diagnostics_only_command(c) if args.diagnostics_only else c
        runs[name] = _run_case(name, case_cmd, runs_root / name, skip_preflight=args.skip_preflight)

    logl_unbatched = _safe_float(runs.get("j2_fixed_true", {}).get("diagnostics", {}).get("logL_total"))
    logl_batched = _safe_float(runs.get("j2_batched", {}).get("diagnostics", {}).get("logL_total"))
    delta_batched = None if logl_unbatched is None or logl_batched is None or not (math.isfinite(logl_unbatched) and math.isfinite(logl_batched)) else logl_batched - logl_unbatched
    true_pair_sum = _safe_float(runs.get("j2_fixed_true", {}).get("diagnostics", {}).get("pair_logL_sum"))
    wrong_pair_sum = _safe_float(runs.get("j2_fixed_wrong", {}).get("diagnostics", {}).get("pair_logL_sum"))
    checks = {"all_requested_runs_completed": set(cases) <= set(runs), "all_cases_passed": all(r.get("status") == "passed" for r in runs.values()), "all_diagnostics_finite_where_expected": all((r.get("status") != "passed") or _finite_diag(r) for r in runs.values()), "true_pair_logL_sum_exceeds_wrong": true_pair_sum is not None and wrong_pair_sum is not None and true_pair_sum > wrong_pair_sum, "null_mock_reports_n_pairs_zero": int(runs.get("j2_null", {}).get("diagnostics", {}).get("n_pairs", -1)) == 0, "off_mode_reports_n_pairs_zero": int(runs.get("off_true_catalog", {}).get("diagnostics", {}).get("n_pairs", -1)) == 0, "batched_unbatched_logL_total_agree": delta_batched is not None and abs(delta_batched) < 1e-6}
    true_logz = _safe_float(runs.get("j2_fixed_true", {}).get("logZ"))
    wrong_logz = _safe_float(runs.get("j2_fixed_wrong", {}).get("logZ"))
    if true_logz is not None and wrong_logz is not None and math.isfinite(true_logz) and math.isfinite(wrong_logz):
        checks["logZ_true_exceeds_wrong"] = true_logz > wrong_logz
    warnings: list[str] = []
    null_logz_diff = None
    null_logl_diff = None
    j2_null_logz = _safe_float(runs.get("j2_null", {}).get("logZ"))
    off_null_logz = _safe_float(runs.get("off_null", {}).get("logZ"))
    if j2_null_logz is not None and off_null_logz is not None and math.isfinite(j2_null_logz) and math.isfinite(off_null_logz):
        null_logz_diff = j2_null_logz - off_null_logz
        checks["null_j2_does_not_strongly_beat_off"] = null_logz_diff < 5.0
    else:
        null_logl_diff = _diff(runs.get("j2_null"), runs.get("off_null"), "logL_total")
        warnings.append("logZ unavailable for null comparison; using prior-midpoint logL diagnostic difference, not evidence")
        checks["null_j2_does_not_strongly_beat_off"] = null_logl_diff is not None and null_logl_diff < 5.0
    if "j2_marginalized" in runs:
        md = runs["j2_marginalized"]["diagnostics"]
        ma = runs["j2_marginalized"].get("results_attrs", {})
        prob_ok, prob_msg = _posterior_pair_probability_check(md)
        if prob_msg:
            warnings.append(f"marginalized posterior probability check: {prob_msg}")
        marg_logl = _safe_float(md.get("logL_marginalized", md.get("logL_total")))
        checks.update({
            "marginalized_logL_finite": marg_logl is not None and math.isfinite(marg_logl),
            "marginalized_partition_mode_diagnostic": md.get("partition_mode") == "marginalize_exact",
            "marginalized_n_partitions_positive": int(md.get("n_partitions", 0) or 0) >= 1,
            "marginalized_posterior_pair_probabilities_valid": prob_ok,
        })
        for attr in ["expected_n_pairs", "map_partition_n_pairs"]:
            if attr not in ma:
                warnings.append(f"marginalized results attr {attr!r} missing; tolerated for branches before PR E")
    if "lens_rate_recovery" in runs:
        posterior = runs["lens_rate_recovery"].get("posterior_summary", {}).get("log10_tau_A", {})
        injected = math.log10(5.0e-4)
        checks["lens_rate_recovery_log10_tau_A_summary_present"] = bool(posterior)
        if posterior:
            checks["lens_rate_recovery_injected_value_in_wide_interval"] = float(posterior.get("q05", -math.inf)) <= injected <= float(posterior.get("q95", math.inf))
    summary = {"profile": args.profile, "diagnostics_only": args.diagnostics_only, "skip_preflight": args.skip_preflight, "checks": checks, "warnings": warnings, "runs": runs, "batched_unbatched": {"logL_total_unbatched": logl_unbatched, "logL_total_batched": logl_batched, "delta_logL_batched_minus_unbatched": delta_batched}, "null_comparison": {"logZ_j2_null_minus_off_null": null_logz_diff, "logL_j2_null_minus_off_null": null_logl_diff, "logL_warning_not_evidence": null_logz_diff is None}, "evidence_differences": {"j2_fixed_true_minus_off": _diff(runs.get("j2_fixed_true"), runs.get("off_true_catalog"), "logZ"), "j2_fixed_wrong_minus_true": _diff(runs.get("j2_fixed_wrong"), runs.get("j2_fixed_true"), "logZ"), "j2_null_minus_off_null": null_logz_diff, "j2_marginalized_minus_fixed_true": _diff(runs.get("j2_marginalized"), runs.get("j2_fixed_true"), "logZ")}}
    (work / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    _write_markdown(work / "validation_summary.md", summary)
    for k, v in checks.items(): print(f"{'PASS' if v else 'FAIL'} {k}")
    return 0 if all(checks.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
