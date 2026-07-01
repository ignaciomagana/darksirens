#!/usr/bin/env python
"""Lightweight local validation for the spectral-siren lensing CLI.

The suite intentionally uses tiny in-repo mocks and prior-midpoint diagnostics
(zero free parameters), not production nested-sampling runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "scripts" / "mock_lensing" / "generate_mock_lensing.py"

PROFILES = {
    "tiny": dict(n_universe=2500, n_sing=1, n_pair=2, nsamp=48, n_unlensed_inj=800, n_lensed_inj=1000, pe_max=48),
    "small": dict(n_universe=8000, n_sing=3, n_pair=3, nsamp=96, n_unlensed_inj=2500, n_lensed_inj=3000, pe_max=96),
}


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _mock_ready(mock_dir: Path, *, n_pair_keep: int) -> bool:
    needed = ["mock_gw_pe.h5", "mock_gw_selection.h5", "mock_lensed_injections.h5", "mock_pair_pe.h5", "partition.json", "candidate_pairs.json"]
    if not all((mock_dir / p).exists() for p in needed):
        return False
    try:
        part = json.loads((mock_dir / "partition.json").read_text())
        return len(part.get("pair_indices", [])) == n_pair_keep
    except Exception:
        return False


def _generate_mock(mock_dir: Path, cfg: dict[str, int], *, seed: int, n_pair_keep: int, reuse: bool) -> None:
    if reuse and _mock_ready(mock_dir, n_pair_keep=n_pair_keep):
        print(f"[validation] reusing mock {mock_dir}", flush=True)
        return
    if mock_dir.exists():
        shutil.rmtree(mock_dir)
    mock_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(GEN), "--outdir", str(mock_dir), "--conditioning", "fixed_counts",
        "--n-universe", str(cfg["n_universe"]), "--n-sing-keep", str(cfg["n_sing"]),
        "--n-pair-keep", str(n_pair_keep), "--nsamp", str(cfg["nsamp"]),
        "--n-unlensed-inj", str(cfg["n_unlensed_inj"]), "--n-lensed-inj", str(cfg["n_lensed_inj"]),
        "--seed", str(seed),
    ]
    _run(cmd)


def _write_wrong_partition(mock_dir: Path, out_path: Path) -> None:
    part = json.loads((mock_dir / "partition.json").read_text())
    pairs = [list(p) for p in part["pair_indices"]]
    if len(pairs) < 2:
        raise RuntimeError("wrong-partner validation requires at least two true pairs")
    # Keep the first image of each pair but rotate second-image partners.
    seconds = [p[1] for p in pairs]
    rotated = seconds[1:] + seconds[:1]
    part["pair_indices"] = [[p[0], q] for p, q in zip(pairs, rotated)]
    out_path.write_text(json.dumps(part, indent=2) + "\n")


def _latest_diagnostics(save_root: Path) -> dict:
    files = sorted(save_root.glob("*/*/diagnostics.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        files = sorted(save_root.glob("*/diagnostics.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise RuntimeError(f"no diagnostics.json found under {save_root}")
    return json.loads(files[-1].read_text())


def _run_cli(mock_dir: Path, save_root: Path, *, cluster_mode: str, partition: Path | None, cfg: dict[str, int], seed: int, pair_batch_size: int = 0, partition_mode: str = "fixed") -> dict:
    if save_root.exists():
        shutil.rmtree(save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "darksirens.cli.inference_lensing",
        "--gw_path", str(mock_dir / "mock_gw_pe.h5"),
        "--gwselection_path", str(mock_dir / "mock_gw_selection.h5"),
        "--wl_backend", "lognormal", "--pop_model", "powerlaw+peak",
        "--fix_cosmology", "true", "--fix_survey", "true", "--fix_population", "true",
        "--sampler", "dynesty", "--nlive", "20", "--dlogz", "10", "--max_samples", "0",
        "--pe_max_per_pair", str(cfg["pe_max"]), "--pair_batch_size", str(pair_batch_size), "--seed", str(seed),
        "--cluster_mode", cluster_mode, "--save_path", str(save_root),
    ]
    if cluster_mode == "j2":
        cmd += [
            "--lensed_injections_path", str(mock_dir / "mock_lensed_injections.h5"),
            "--pair_pe_path", str(mock_dir / "mock_pair_pe.h5"),
            "--partition_mode", partition_mode,
        ]
        if partition_mode == "marginalize_exact":
            cmd += ["--candidate_pairs_path", str(mock_dir / "candidate_pairs.json")]
        else:
            cmd += ["--partition_path", str(partition or mock_dir / "partition.json")]
    _run(cmd)
    return _latest_diagnostics(save_root)


def _finite_diag(diag: dict) -> bool:
    keys = ["logL_total", "singleton_logL_sum", "pair_logL_sum", "selection_correction_total", "Neff_combined"]
    return all(math.isfinite(float(diag[k])) for k in keys if k in diag)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", choices=PROFILES, default="tiny")
    ap.add_argument("--workdir", default="/tmp/ds_lensing_validation")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--reuse", action="store_true", help="reuse compatible mock directories if present")
    ap.add_argument("--pair_batch_size", type=int, default=0,
                    help="forwarded to the lensing CLI for J=2 pair likelihood batching")
    ap.add_argument("--run_marginalized", action="store_true",
                    help="also run --partition_mode marginalize_exact when candidate_pairs.json exists")
    args = ap.parse_args(argv)

    cfg = PROFILES[args.profile]
    work = Path(args.workdir).resolve()
    mock = work / f"mock_{args.profile}"
    null_mock = work / f"mock_{args.profile}_null"
    runs = work / "runs"
    work.mkdir(parents=True, exist_ok=True)

    _generate_mock(mock, cfg, seed=args.seed, n_pair_keep=cfg["n_pair"], reuse=args.reuse)
    wrong_part = work / "wrong_partition.json"
    _write_wrong_partition(mock, wrong_part)
    _generate_mock(null_mock, cfg, seed=args.seed + 17, n_pair_keep=0, reuse=args.reuse)

    off = _run_cli(mock, runs / "off", cluster_mode="off", partition=None, cfg=cfg, seed=args.seed)
    true = _run_cli(mock, runs / "j2_true", cluster_mode="j2", partition=mock / "partition.json", cfg=cfg, seed=args.seed, pair_batch_size=args.pair_batch_size)
    wrong = _run_cli(mock, runs / "j2_wrong", cluster_mode="j2", partition=wrong_part, cfg=cfg, seed=args.seed, pair_batch_size=args.pair_batch_size)
    null = _run_cli(null_mock, runs / "j2_null", cluster_mode="j2", partition=null_mock / "partition.json", cfg=cfg, seed=args.seed, pair_batch_size=args.pair_batch_size)
    marginalized = None
    if args.run_marginalized and (mock / "candidate_pairs.json").exists():
        marginalized = _run_cli(
            mock, runs / "j2_marginalized", cluster_mode="j2", partition=None, cfg=cfg,
            seed=args.seed, pair_batch_size=args.pair_batch_size, partition_mode="marginalize_exact",
        )

    checks = {
        "true_j2_finite_logL_and_diagnostics": math.isfinite(float(true["logL_total"])) and _finite_diag(true),
        "wrong_pair_j2_lower_pair_logL_sum": float(wrong["pair_logL_sum"]) < float(true["pair_logL_sum"]),
        "null_case_reports_no_observed_pairs": int(null["n_pairs"]) == 0 and float(null.get("pair_logL_sum", 0.0)) == 0.0,
        "off_mode_singleton_only": off["cluster_mode"] == "off" and int(off["n_pairs"]) == 0 and float(off.get("pair_logL_sum", 0.0)) == 0.0,
    }
    if marginalized is not None:
        probs = [float(x["p_pair"]) for x in marginalized.get("posterior_pair_probabilities", [])]
        checks.update({
            "marginalized_finite_logL": math.isfinite(float(marginalized["logL_marginalized"])),
            "marginalized_has_partitions": int(marginalized["n_partitions"]) >= 1,
            "marginalized_expected_n_pairs_finite": math.isfinite(float(marginalized["expected_n_pairs"])),
            "marginalized_pair_probabilities_in_unit_interval": all(0.0 <= p <= 1.0 for p in probs),
        })
    print("\n[validation] diagnostics summary")
    summary_items = [("off", off), ("j2_true", true), ("j2_wrong", wrong), ("j2_null", null)]
    if marginalized is not None:
        summary_items.append(("j2_marginalized", marginalized))
    for name, diag in summary_items:
        print(f"  {name}: logL={diag.get('logL_total', diag.get('logL_marginalized'))} singleton={diag.get('singleton_logL_sum', diag.get('map_singleton_logL_sum'))} pair={diag.get('pair_logL_sum', diag.get('map_pair_logL_sum'))} n_pairs={diag.get('n_pairs', diag.get('expected_n_pairs'))}")
    print("[validation] checks")
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'} {k}")
    diagnostics_out = {"off": off, "j2_true": true, "j2_wrong": wrong, "j2_null": null}
    if marginalized is not None:
        diagnostics_out["j2_marginalized"] = marginalized
    (work / "validation_summary.json").write_text(json.dumps({"checks": checks, "diagnostics": diagnostics_out}, indent=2, allow_nan=True) + "\n")
    if not all(checks.values()):
        return 1
    print(f"[validation] complete: {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
