#!/usr/bin/env python3
"""Short-budget TinyNS benchmark sweep for ``darksirens_inference``."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import h5py
except Exception:  # pragma: no cover - optional fallback
    h5py = None

DEFAULT_SWEEP: list[dict[str, Any]] = [
    {"name": "recommended", "args": ["--tinyns_preset", "recommended"]},
    {"name": "cheap", "tinyns_replacement_chains": 8, "tinyns_walks": 8, "tinyns_step_scale": 0.05, "tinyns_min_accepts": 1, "tinyns_max_attempts": 50000, "tinyns_jax_block_size": 128},
    {"name": "fast16", "tinyns_replacement_chains": 16, "tinyns_walks": 20, "tinyns_step_scale": 0.03, "tinyns_min_accepts": 3, "tinyns_max_attempts": 100000, "tinyns_jax_block_size": 64},
    {"name": "fast32", "tinyns_replacement_chains": 32, "tinyns_walks": 15, "tinyns_step_scale": 0.035, "tinyns_min_accepts": 3, "tinyns_max_attempts": 150000, "tinyns_jax_block_size": 64},
    {"name": "fast16_B128", "tinyns_replacement_chains": 16, "tinyns_walks": 20, "tinyns_step_scale": 0.03, "tinyns_min_accepts": 3, "tinyns_max_attempts": 100000, "tinyns_jax_block_size": 128},
]

SUMMARY_FIELDS = [
    "config_name", "status", "returncode", "elapsed_seconds", "nlive", "dlogz", "max_samples",
    "tinyns_preset", "tinyns_replacement_chains", "tinyns_walks", "tinyns_step_scale", "tinyns_min_accepts", "tinyns_max_attempts", "tinyns_jax_block_size",
    "logZ", "logZerr", "niter", "ncall", "seconds", "niter_per_sec", "ncall_per_sec", "calls_per_iter", "final_delta_logz", "live_weight_fraction", "posterior_ess",
    "replacement_mean_batches", "replacement_max_batches", "replacement_failures", "replacement_rescue_used", "replacement_rescue_attempts", "insertion_rank_mean_z", "insertion_rank_std_ratio", "message",
]
NO_NORMALIZATION_GRID_FLAGS = {"--normalization_nz", "--normalization_zmax", "--normalization_chunk_size", "--selection_chunk_size"}


def sweep_entry_to_args(entry: dict[str, Any]) -> list[str]:
    if "args" in entry:
        return [str(x) for x in entry["args"]]
    args: list[str] = []
    for key, value in entry.items():
        if key == "name" or value is None:
            continue
        flag = "--" + key
        args.extend([flag, str(value).lower() if isinstance(value, bool) else str(value)])
    return args


def load_sweep(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(x) for x in DEFAULT_SWEEP]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--sweep-json must contain a JSON list")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or not item.get("name"):
            raise ValueError(f"sweep item {i} must be an object with a name")
    return data


def build_command(opts: argparse.Namespace, entry: dict[str, Any], run_save_path: Path) -> list[str]:
    cmd = [
        opts.executable, "--gw_path", opts.gw_path, "--gwselection_path", opts.gwselection_path,
        "--universe_model", opts.universe_model, "--pop_model", opts.pop_model,
        "--sampler", "tinyns", "--nlive", str(opts.nlive), "--dlogz", str(opts.dlogz),
        "--max_samples", str(opts.max_samples), "--seed", str(opts.seed),
        "--tinyns_progress_interval", "10", "--save_path", str(run_save_path),
    ]
    if opts.survey_path:
        cmd.extend(["--survey_path", opts.survey_path])
    cmd.extend(opts.extra_arg or [])
    if opts.extra_args_json:
        extra = json.loads(opts.extra_args_json)
        if not isinstance(extra, list):
            raise ValueError("--extra-args-json must decode to a list")
        cmd.extend(str(x) for x in extra)
    cmd.extend(sweep_entry_to_args(entry))
    return cmd


def _find_latest(path: Path, name: str) -> Path | None:
    matches = [p for p in path.rglob(name) if p.is_file()]
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return value.item()
    except AttributeError:
        return value


def parse_diagnostics(run_root: Path, stdout_path: Path | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    sidecar = _find_latest(run_root, "tinyns_diagnostics.json")
    if sidecar:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        runtime = payload.get("tinyns_runtime_diagnostics") or {}
        out.update(runtime)
        if payload.get("tinyns_resolved_config"):
            out.update({k: v for k, v in payload["tinyns_resolved_config"].items() if k.startswith("tinyns_")})
        out["message"] = f"diagnostics: {sidecar}"
        return out
    h5path = _find_latest(run_root, "results.hdf5")
    if h5path and h5py is not None:
        with h5py.File(h5path, "r") as f:
            for key in SUMMARY_FIELDS:
                attr = "tinyns_" + key if not key.startswith("tinyns_") else key
                if attr in f.attrs:
                    out[key] = _decode_attr(f.attrs[attr])
            if "logZ" in f.attrs: out["logZ"] = float(f.attrs["logZ"])
            if "logZerr" in f.attrs: out["logZerr"] = float(f.attrs["logZerr"])
        out["message"] = f"hdf5 attrs: {h5path}"
        return out
    if stdout_path and stdout_path.exists():
        text = stdout_path.read_text(errors="ignore")
        for key in ("niter_per_sec", "ncall_per_sec", "calls_per_iter", "final_delta_logz"):
            m = re.search(rf"{key}\s*[=:]\s*([-+0-9.eE]+)", text)
            if m: out[key] = float(m.group(1))
    return out


def write_summary(rows: list[dict[str, Any]], bench_dir: Path) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    with (bench_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    (bench_dir / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# Correctness gates.  The knobs this sweep varies (walks, step_scale,
# min_accepts) control whether the live-point replacement is a valid Markov move;
# under-mixing does not fail loudly, it biases logZ and the posterior while making
# the run FASTER.  Ranking on speed behind two replacement-health flags therefore
# recommends the fastest BIASED configuration, which is worse than no harness.
INSERTION_Z_MAX = 3.0                    # |z| of the insertion-rank statistic
INSERTION_STD_RATIO_RANGE = (0.8, 1.25)  # its spread relative to uniform
LOGZ_SIGMA_MAX = 3.0                     # cross-config logZ agreement
LOGZ_REFERENCE = "recommended"


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _insertion_rank_ok(row: dict[str, Any]) -> tuple[bool, str]:
    """Is the sampler's insertion-rank distribution consistent with uniform?

    The standard nested-sampling correctness statistic: the insertion rank of each
    new live point is uniform iff the replacement really draws from the
    constrained prior.  ``|mean_z| >~ 3`` (or a spread well away from 1) means the
    chain is under-mixed and the evidence is not trustworthy, however fast it ran.
    Missing diagnostics are not treated as a failure — they are reported as such.
    """
    z = _num(row.get("insertion_rank_mean_z"))
    ratio = _num(row.get("insertion_rank_std_ratio"))
    if z is not None and abs(z) >= INSERTION_Z_MAX:
        return False, f"insertion_rank_mean_z={z:+.2f} (|z| >= {INSERTION_Z_MAX})"
    lo, hi = INSERTION_STD_RATIO_RANGE
    if ratio is not None and not (lo <= ratio <= hi):
        return False, f"insertion_rank_std_ratio={ratio:.2f} (outside [{lo}, {hi}])"
    return True, ""


def logz_outliers(rows: list[dict[str, Any]],
                  reference: str = LOGZ_REFERENCE) -> dict[str, str]:
    """Configs whose logZ disagrees with the reference preset by > 3 sigma.

    Every config samples the SAME posterior, so their evidences must agree within
    their own quoted errors; a disagreement is a tuning artefact, not a result.
    """
    ref = next((r for r in rows if r.get("config_name") == reference
                and _num(r.get("logZ")) is not None), None)
    if ref is None:
        return {}
    z_ref, e_ref = _num(ref["logZ"]), _num(ref.get("logZerr")) or 0.0
    out: dict[str, str] = {}
    for r in rows:
        if r.get("config_name") == reference:
            continue
        z, e = _num(r.get("logZ")), _num(r.get("logZerr")) or 0.0
        if z is None:
            continue
        sigma = (e * e + e_ref * e_ref) ** 0.5
        nsig = abs(z - z_ref) / sigma if sigma > 0 else float("inf")
        if nsig > LOGZ_SIGMA_MAX:
            out[str(r.get("config_name"))] = (
                f"logZ={z:.3f} disagrees with {reference} ({z_ref:.3f}) by "
                f"{nsig:.1f} sigma")
    return out


def _score(row: dict[str, Any], outliers: dict[str, str] | None = None
           ) -> tuple[int, int, float, float]:
    outliers = outliers or {}
    has_diag = 1 if row.get("niter_per_sec") not in (None, "") else 0
    correct = 1 if (_insertion_rank_ok(row)[0]
                    and str(row.get("config_name")) not in outliers) else 0
    failures = float(row.get("replacement_failures") or 0)
    rescue = 1.0 if str(row.get("replacement_rescue_used")).lower() == "true" else 0.0
    # Correctness outranks speed: an under-mixed config is FASTER, so any ranking
    # that puts niter/sec ahead of the mixing diagnostics selects for the bias.
    return (has_diag, correct, -failures - rescue, float(row.get("niter_per_sec") or 0))


def print_ranking(rows: list[dict[str, Any]]) -> None:
    print("\nTinyNS short-budget benchmark summary\n")
    print(f"{'name':<16} {'status':<10} {'niter/sec':>10} {'ncall/sec':>10} {'calls/iter':>11} {'repl_mean_batches':>18} {'failures':>9} {'rescue':>7} {'final_dlogz':>11}")
    outliers = logz_outliers(rows)
    for r in sorted(rows, key=lambda r: _score(r, outliers), reverse=True):
        print(f"{r.get('config_name',''):<16} {r.get('status',''):<10} {str(r.get('niter_per_sec','')):>10} {str(r.get('ncall_per_sec','')):>10} {str(r.get('calls_per_iter','')):>11} {str(r.get('replacement_mean_batches','')):>18} {str(r.get('replacement_failures','')):>9} {str(r.get('replacement_rescue_used','')):>7} {str(r.get('final_delta_logz','')):>11}")
    for r in rows:
        ok, reason = _insertion_rank_ok(r)
        if not ok:
            print(f"WARNING: {r.get('config_name')} FAILS the sampler-correctness "
                  f"gate: {reason}; its evidence and posterior are not usable.")
    for name, reason in outliers.items():
        print(f"WARNING: {name} {reason}; the configs sample the same posterior, "
              "so this is a tuning artefact, not a result.")
    healthy = [r for r in rows
               if r.get("niter_per_sec") not in (None, "")
               and not r.get("replacement_failures")
               and str(r.get("replacement_rescue_used")).lower() != "true"
               and _insertion_rank_ok(r)[0]
               and str(r.get("config_name")) not in outliers]
    if healthy:
        print(f"\nBest healthy candidate by niter/sec: {max(healthy, key=lambda r: float(r.get('niter_per_sec') or 0))['config_name']}")
    else:
        print("\nNo config passed both the replacement-health and "
              "sampler-correctness gates; do NOT pick one on speed alone.")
    if not any(r.get("niter_per_sec") not in (None, "") for r in rows): print("WARNING: no config produced diagnostics")
    if all(r.get("status") == "failed" for r in rows): print("WARNING: all configs failed")
    if rows and all(str(r.get("replacement_rescue_used")).lower() == "true" for r in rows if r.get("replacement_rescue_used") not in (None, "")): print("WARNING: all configs with rescue diagnostics used rescue")
    if any(float(r.get("replacement_mean_batches") or 0) > 10 for r in rows): print("WARNING: replacement_mean_batches is large for at least one config")


def run_one(cmd: list[str], config_dir: Path, timeout_minutes: float) -> tuple[str, int | None, float]:
    stdout = config_dir / "stdout.log"; stderr = config_dir / "stderr.log"
    start = time.monotonic()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_minutes * 60)
            status = "completed" if rc == 0 else "failed"
        except subprocess.TimeoutExpired:
            status = "timeout"; rc = None
            os.killpg(proc.pid, signal.SIGTERM)
            try: proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    return status, rc, time.monotonic() - start


def _extract_extra_args(argv: list[str] | None) -> tuple[list[str] | None, list[str]]:
    if argv is None:
        argv = sys.argv[1:]
    cleaned: list[str] = []
    extra: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--extra-arg":
            if i + 1 >= len(argv):
                raise SystemExit("--extra-arg requires one following argument")
            extra.append(argv[i + 1])
            i += 2
            continue
        if token.startswith("--extra-arg="):
            extra.append(token.split("=", 1)[1])
            i += 1
            continue
        cleaned.append(token)
        i += 1
    return cleaned, extra


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv, extracted_extra = _extract_extra_args(argv)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable", default="darksirens_inference", help="CLI executable to launch")
    p.add_argument("--gw-path", required=True); p.add_argument("--gwselection-path", required=True); p.add_argument("--survey-path")
    p.add_argument("--universe-model", default="spectral_sirens"); p.add_argument("--pop-model", default="powerlaw+peak")
    p.add_argument("--base-save-path", required=True); p.add_argument("--nlive", type=int, default=400); p.add_argument("--dlogz", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=2000); p.add_argument("--seed", type=int, default=21); p.add_argument("--timeout-minutes", type=float, default=20)
    p.add_argument("--sweep-json"); p.add_argument("--extra-args-json")
    opts = p.parse_args(argv)
    opts.extra_arg = extracted_extra
    return opts


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv); sweep = load_sweep(opts.sweep_json)
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    bench_dir = Path(opts.base_save_path) / f"tinyns_short_budget__{stamp}"; bench_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in sweep:
        name = entry["name"]; config_dir = bench_dir / name; run_root = config_dir / "run"; run_root.mkdir(parents=True, exist_ok=True)
        cmd = build_command(opts, entry, run_root)
        (config_dir / "command.txt").write_text(shlex.join(cmd) + "\n", encoding="utf-8")
        status, rc, elapsed = run_one(cmd, config_dir, opts.timeout_minutes)
        diag = parse_diagnostics(run_root, config_dir / "stdout.log")
        row = {"config_name": name, "status": status, "returncode": rc, "elapsed_seconds": round(elapsed, 3), "nlive": opts.nlive, "dlogz": opts.dlogz, "max_samples": opts.max_samples}
        row.update({k: v for k, v in entry.items() if k != "args"})
        row.update(diag); rows.append(row); write_summary(rows, bench_dir)
    print_ranking(rows); print(f"\nWrote benchmark summaries under {bench_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
