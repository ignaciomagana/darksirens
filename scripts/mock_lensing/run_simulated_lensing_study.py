#!/usr/bin/env python
"""Run a simulated end-to-end spectral-siren lensing study.

This is mock-only.  It generates observed-event catalogs, builds candidate
pair graphs from observed events (never from truth), runs preflight and
inference, and then reads truth labels only for recovery summaries.
"""
from __future__ import annotations

import argparse, csv, datetime, json, math, shutil, shlex, subprocess, sys, time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from darksirens.lensing.simulation_config import resolve_config, write_config
from darksirens.lensing import file_contract
from darksirens.lensing.partitions import connected_components_from_candidate_pairs, exact_component_partitions, validate_candidate_pairs
from darksirens.lensing.marginal_diagnostics import component_partition_diagnostic_rows, partition_diagnostic_rows

GEN = ROOT / "scripts" / "mock_lensing" / "generate_mock_lensing.py"
BUILD = ROOT / "scripts" / "mock_lensing" / "build_candidate_pairs_from_observed.py"

PROFILES: dict[str, dict[str, int]] = {
    "tiny": dict(n_universe=4000, n_sing=2, n_pair=2, nsamp=48, n_unlensed_inj=1000, n_lensed_inj=1000, pe_max=48, default_nlive=32, max_total_edges=8),
    "small": dict(n_universe=12000, n_sing=8, n_pair=4, nsamp=96, n_unlensed_inj=4000, n_lensed_inj=5000, pe_max=96, default_nlive=80, max_total_edges=24),
    "paper": dict(n_universe=120000, n_sing=200, n_pair=40, nsamp=1000, n_unlensed_inj=200000, n_lensed_inj=300000, pe_max=512, default_nlive=1000, max_total_edges=400),
}

CASE_ORDER = [
    "A_no_true_pairs_sparse_wrong_graph", "B_true_pairs_clean_graph", "C_true_pairs_many_wrong_edges",
    "D_true_pairs_bad_pair_tag", "E_true_pairs_no_sky_marks", "F_true_pairs_no_time_marks",
    "G_true_pairs_full_marks", "H_ambiguous_components", "H_no_time_ambiguous_components",
]

def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool): return v
    if v.lower() in {"1","true","yes","y","on"}: return True
    if v.lower() in {"0","false","no","n","off"}: return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")

def _utc() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

def _run(cmd: list[str]) -> tuple[float, int]:
    result = _run_logged(cmd)
    return float(result["runtime_sec"]), int(result["return_code"])

def _run_logged(cmd: list[str], log_prefix: Path | None = None) -> dict[str, Any]:
    """Run a command, streaming output while optionally saving full logs."""
    print("+", " ".join(map(str, cmd)), flush=True)
    stdout_path = stderr_path = None
    out_fh = err_fh = None
    if log_prefix is not None:
        log_prefix.parent.mkdir(parents=True, exist_ok=True)
        stdout_path = log_prefix.with_suffix(".stdout")
        stderr_path = log_prefix.with_suffix(".stderr")
        out_fh = stdout_path.open("wb")
        err_fh = stderr_path.open("wb")
    t = time.time()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    import selectors
    sel = selectors.DefaultSelector()
    assert p.stdout is not None and p.stderr is not None
    sel.register(p.stdout, selectors.EVENT_READ, (sys.stdout.buffer, out_fh))
    sel.register(p.stderr, selectors.EVENT_READ, (sys.stderr.buffer, err_fh))
    while sel.get_map():
        for key, _ in sel.select():
            chunk = key.fileobj.read1(8192)
            stream, fh = key.data
            if chunk:
                stream.write(chunk); stream.flush()
                if fh is not None:
                    fh.write(chunk); fh.flush()
            else:
                sel.unregister(key.fileobj)
    rc = int(p.wait())
    if out_fh is not None: out_fh.close()
    if err_fh is not None: err_fh.close()
    runtime = time.time() - t
    if log_prefix is not None:
        print(f"  command exited {rc} in {runtime:.1f}s; logs: {stdout_path}, {stderr_path}", flush=True)
    return {"runtime_sec": runtime, "return_code": rc, "stdout_path": str(stdout_path) if stdout_path else None, "stderr_path": str(stderr_path) if stderr_path else None}

def _load_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.exists() else default

def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value

def _case_spec(name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    edge_keys = cfg.get("edge_mark_prior_keys", ["log_sky_overlap"])
    spec = dict(n_pair=cfg["n_pair"], max_edges_per_event=cfg.get("max_edges_per_event", 2), max_total_edges=cfg["max_total_edges"],
                include_sky_marks=cfg.get("include_sky_marks", True), include_time_marks=cfg.get("include_time_marks", True), time_window_sec=math.inf,
                pair_marks="time" if cfg.get("include_time_marks", True) else "none", edge_mark_prior_keys_csv=",".join(edge_keys), pair_tag_model=cfg.get("pair_tag_model", "snr_time_sky"),
                pair_tag_perturb_logit=cfg.get("pair_tag_perturb_logit", 0.0), pair_tag_constant=cfg.get("pair_tag_constant", 1.0),
                edge_mark_prior_keys=edge_keys)
    if name.startswith("A_"):
        spec.update(n_pair=0, max_edges_per_event=1, max_total_edges=max(1, cfg["n_sing"]), include_sky_marks=False, include_time_marks=False, pair_marks="none", edge_mark_prior_keys_csv="")
    elif name.startswith("B_"):
        spec.update(
            max_edges_per_event=1,
            max_total_edges=max(1, cfg["n_pair"]),
            include_time_marks=False,
            pair_marks="none",
            pair_tag_model="snr_sky",
        )
    elif name.startswith("C_"):
        spec.update(max_edges_per_event=4, max_total_edges=max(8, 4 * cfg["n_pair"]))
    elif name.startswith("D_"):
        spec.update(pair_tag_model="snr_time_sky", pair_tag_perturb_logit=1.5)
    elif name.startswith("E_"):
        spec.update(include_sky_marks=False, edge_mark_prior_keys_csv="", pair_tag_model="snr_time")
    elif name.startswith("F_"):
        spec.update(include_time_marks=False, pair_marks="none", pair_tag_model="snr_sky")
    elif name.startswith("G_"):
        # Full-marks case: pair_marks=time needs a physical |Delta t| on EVERY
        # edge, which only exists under uniform observation times. Cap the
        # graph like H so components stay exactly enumerable at paper scale.
        spec.update(
            observation_times="uniform",
            max_edges_per_event=2,
            max_total_edges=max(4, 2 * cfg["n_pair"]),
        )
    elif name == "H_no_time_ambiguous_components":
        spec.update(
            max_edges_per_event=3,
            max_total_edges=max(6, 3 * cfg["n_pair"]),
            include_time_marks=False,
            pair_marks="none",
            pair_tag_model="snr_sky",
            edge_mark_prior_keys_csv="",
        )
    elif name.startswith("H_"):
        spec.update(max_edges_per_event=3, max_total_edges=max(6, 3 * cfg["n_pair"]), edge_mark_prior_keys_csv="")
    if cfg.get("singleton_lensing", "off") == "sl_mixture":
        # The lensed-singleton mixture requires a CERTAIN pair tag (CLI guard:
        # untagged both-detected pairs would leak into the singleton stream).
        # Joint-campaign runs therefore use perfect pair identification — a
        # stated simplification; the case-specific tag models remain for the
        # pairs-only analyses.
        spec.update(pair_tag_model="constant", pair_tag_constant=1.0,
                    pair_tag_perturb_logit=0.0)
    return spec

@lru_cache(maxsize=1)
def _known_inference_flags() -> set[str]:
    from darksirens.cli.inference_lensing import build_parser
    return {
        opt
        for action in build_parser()._actions
        for opt in action.option_strings
        if opt.startswith("--")
    }

def validate_known_inference_flags(cmd: list[str]) -> None:
    """Reject stale or unknown long options in generated inference commands."""
    if "--edge_prior_marks" in cmd:
        raise ValueError("generated inference command uses stale flag --edge_prior_marks; use --edge_mark_prior_keys")
    try:
        known = _known_inference_flags()
    except Exception:
        known = {"--edge_mark_prior_keys", "--preflight_only", "--preflight_json"}
    unknown = []
    for tok in cmd:
        if isinstance(tok, str) and tok.startswith("--") and tok not in known:
            unknown.append(tok)
    if unknown:
        raise ValueError(f"generated inference command contains unknown inference flags: {sorted(set(unknown))}")

def _file_contract_report(case_dir: Path, config_path: Path) -> dict[str, Any]:
    report = {
        "observed_gw_pe": file_contract.validate_observed_gw_pe(case_dir / "mock_observed_gw_pe.h5"),
        "observed_catalog": file_contract.validate_observed_catalog(case_dir / "observed_catalog.json"),
        "candidate_pairs": file_contract.validate_candidate_pairs(case_dir / "candidate_pairs.json"),
        "gwselection": file_contract.validate_selection_inputs(case_dir / "mock_gw_selection.h5"),
        "lensed_injections": file_contract.validate_selection_inputs(case_dir / "mock_lensed_injections.h5"),
        "run_config": file_contract.validate_run_config(config_path),
    }
    report["ok"] = all(v.get("ok", False) for v in report.values() if isinstance(v, dict))
    return report

def _candidate_graph_summary(preflight: dict[str, Any], file_report: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    ps = preflight.get("summary", {}) if isinstance(preflight, dict) else {}
    cs = file_report.get("candidate_pairs", {}).get("summary", {}) if isinstance(file_report, dict) else {}
    return {
        "n_events": ps.get("n_events", cs.get("n_events")),
        "n_candidate_edges": ps.get("n_candidate_pairs", cs.get("n_candidate_pairs")),
        "n_components": ps.get("n_components"),
        "component_n_partitions": ps.get("component_n_partitions"),
        "available_edge_mark_keys": ps.get("available_edge_mark_keys", cs.get("mark_keys")),
        "requested_edge_mark_prior_keys": ps.get("edge_mark_prior_keys", spec.get("edge_mark_prior_keys", [])),
        "pair_marks": ps.get("pair_marks", spec.get("pair_marks")),
        "pair_tag_model": ps.get("pair_tag_model", spec.get("pair_tag_model")),
    }

def write_preflight_summary(work: Path, summary: dict[str, Any]) -> None:
    (work / "preflight_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    lines = [
        f"# Simulated lensing preflight ({summary.get('profile')})", "",
        "| case | j2 preflight | off preflight | n_events | n_edges | n_components | n_partitions | warnings | errors |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, rec in summary.get("cases", {}).items():
        g = rec.get("candidate_graph_summary", {})
        parts = g.get("component_n_partitions")
        if isinstance(parts, list) and all(isinstance(x, (int, float)) for x in parts):
            n_partitions = math.prod(int(x) for x in parts)
        else:
            n_partitions = ""
        j2 = rec.get("j2", {})
        off = rec.get("off", {})
        j2_status = j2.get("status", rec.get("status"))
        off_status = off.get("status", "skipped")
        lines.append(f"| {name} | {j2_status} | {off_status} | {g.get('n_events')} | {g.get('n_candidate_edges')} | {g.get('n_components')} | {n_partitions} | {len(rec.get('warnings', []))} | {len(rec.get('errors', []))} |")
    (work / "preflight_summary.md").write_text("\n".join(lines) + "\n")

def generate_cmd(case_dir: Path, spec: dict[str, Any], cfg: dict[str, Any], seed: int) -> list[str]:
    cmd = [sys.executable, str(GEN), "--outdir", str(case_dir), "--conditioning", str(cfg["conditioning"]),
           "--n-universe", str(cfg["n_universe"]), "--n-sing-keep", str(cfg["n_sing"]),
           "--n-pair-keep", str(spec["n_pair"]), "--max-sing-keep", str(cfg["n_sing"]),
           "--max-pair-keep", str(spec["n_pair"]), "--nsamp", str(cfg["nsamp"]),
           "--n-unlensed-inj", str(cfg["n_unlensed_inj"]), "--n-lensed-inj", str(cfg["n_lensed_inj"]),
           "--pop_model", str(cfg.get("pop_model", "powerlaw+peak")),
           "--include-lensed-singletons", str(bool(cfg.get("include_lensed_singletons", False))).lower(),
           "--observation-times", str(spec.get("observation_times", cfg.get("observation_times", "placeholder"))),
           "--t-obs-days", str(spec.get("t_obs_days", cfg.get("t_obs_days", 365.25))),
           "--seed", str(seed), "--write-unified-observed-catalog", "true", "--write-legacy-pair-pe", "false"]
    if cfg.get("tau_A") is not None:
        cmd.extend(["--tau-A", str(cfg["tau_A"])])
    if cfg.get("tau_n") is not None:
        cmd.extend(["--tau-n", str(cfg["tau_n"])])
    return cmd

def build_graph_cmd(case_dir: Path, spec: dict[str, Any], seed: int) -> list[str]:
    return [sys.executable, str(BUILD), "--gw_path", str(case_dir / "mock_observed_gw_pe.h5"),
            "--observed_catalog_path", str(case_dir / "observed_catalog.json"), "--out", str(case_dir / "candidate_pairs.json"),
            "--max_edges_per_event", str(spec["max_edges_per_event"]), "--max_total_edges", str(spec["max_total_edges"]),
            "--include_time_marks", str(spec["include_time_marks"]).lower(), "--include_sky_marks", str(spec["include_sky_marks"]).lower(),
            "--include_truth_labels", "false", "--sky_overlap_weight", "0.0", "--seed", str(seed)]

def read_results_attrs(run_dir: Path | None) -> dict[str, Any]:
    """Read lightweight result-file attributes from a timestamped run directory."""
    if run_dir is None:
        return {}
    results_path = run_dir / "results.hdf5"
    if not results_path.exists():
        return {}
    try:
        import h5py
        with h5py.File(results_path, "r") as f:
            attrs = {k: _jsonable(v) for k, v in f.attrs.items()}
            labels = json.loads(attrs.get("labels", "[]")) if isinstance(attrs.get("labels"), str) else []
            if "samples" in f and "log10_tau_A" in labels:
                col = np.asarray(f["samples"])[:, labels.index("log10_tau_A")]
                attrs["log10_tau_A_summary"] = {"mean": float(np.mean(col)), "median": float(np.median(col)), "q05": float(np.quantile(col, 0.05)), "q95": float(np.quantile(col, 0.95))}
            if "samples" in f and labels:
                samples = np.asarray(f["samples"])
                attrs["hyperparameter_summaries"] = {
                    label: {
                        "mean": float(np.mean(samples[:, k])),
                        "median": float(np.median(samples[:, k])),
                        "q05": float(np.quantile(samples[:, k], 0.05)),
                        "q95": float(np.quantile(samples[:, k], 0.95)),
                    }
                    for k, label in enumerate(labels)
                    if k < samples.shape[1]
                }
            return attrs
    except Exception as exc:
        return {"read_error": str(exc)}

def _extract_finite_float(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in record and record[key] is not None:
            try:
                value = float(record[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None

def _has_present_value(record: dict[str, Any], keys: tuple[str, ...]) -> bool:
    return any(key in record and record[key] is not None for key in keys)

def extract_logz(results_attrs: dict[str, Any]) -> float | None:
    """Return finite sampler log-evidence from result attributes when available."""
    return _extract_finite_float(results_attrs, ("logZ", "logz", "log_evidence", "log_evidence_final"))

def extract_midpoint_loglike(diagnostics: dict[str, Any] | None) -> float | None:
    """Return finite prior-midpoint log-likelihood from diagnostics/midpoint JSON."""
    if not isinstance(diagnostics, dict):
        return None
    return _extract_finite_float(diagnostics, ("logL_total", "loglike", "logL", "log_likelihood"))

def classify_run_outputs(results_attrs: dict[str, Any] | None, diagnostics: dict[str, Any] | None, diagnostics_only: bool) -> dict[str, str]:
    """Classify process-independent midpoint and evidence usability."""
    results_attrs = results_attrs or {}
    diagnostics = diagnostics or {}
    midpoint_keys = ("logL_total", "loglike", "logL", "log_likelihood")
    logz_keys = ("logZ", "logz", "log_evidence", "log_evidence_final")
    if extract_midpoint_loglike(diagnostics) is not None:
        midpoint_status = "finite_midpoint"
    elif _has_present_value(diagnostics, midpoint_keys):
        midpoint_status = "nonfinite_midpoint"
    else:
        midpoint_status = "missing_midpoint"
    if diagnostics_only:
        evidence_status = "diagnostics_only_not_meaningful"
    elif extract_logz(results_attrs) is not None:
        evidence_status = "finite_logZ"
    elif _has_present_value(results_attrs, logz_keys):
        evidence_status = "nonfinite_logZ"
    else:
        evidence_status = "missing_logZ"
    return {"process_status": "passed", "midpoint_status": midpoint_status, "evidence_status": evidence_status}

def extract_logzerr(results_attrs: dict[str, Any]) -> float | None:
    for key in ("logZerr", "logzerr", "logZ_error", "log_evidence_err"):
        if key in results_attrs and results_attrs[key] is not None:
            try:
                value = float(results_attrs[key])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return None

def evidence_delta(logz_j2: float | None, logz_off: float | None, diagnostics_only: bool = False) -> float | None:
    if diagnostics_only or logz_j2 is None or logz_off is None:
        return None
    return logz_j2 - logz_off

def hyperparameter_truth_values(truth: dict[str, Any], pop_model: str) -> dict[str, float]:
    """Map sampled hyperparameter labels to mock truth values.

    Population labels are the registry's LaTeX labels, aligned positionally
    with the generator's truth theta vector (both derive from the same
    param_specs ordering). Lens-rate labels are plain names.
    """
    out: dict[str, float] = {}
    theta = truth.get("theta") or []
    try:
        from darksirens.gw.populations.registry import pop_model_prior_parser
        _lows, _highs, labels, _kinds, _latex = pop_model_prior_parser(pop_model)
        if len(labels) == len(theta):
            out.update({str(l): float(v) for l, v in zip(labels, theta)})
    except Exception:
        pass
    if truth.get("tau_A"):
        out["log10_tau_A"] = float(math.log10(float(truth["tau_A"])))
    if truth.get("tau_n") is not None:
        out["tau_n"] = float(truth["tau_n"])
    return out


def hyperparameter_recovery_rows(case: str, run_kind: str, results_attrs: dict[str, Any], truth_values: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for label, s in (results_attrs.get("hyperparameter_summaries") or {}).items():
        tv = truth_values.get(label)
        rows.append({
            "case": case, "run": run_kind, "label": label,
            "truth": tv, "mean": s.get("mean"), "median": s.get("median"),
            "q05": s.get("q05"), "q95": s.get("q95"),
            "truth_in_90ci": (None if tv is None else bool(s.get("q05", -math.inf) <= tv <= s.get("q95", math.inf))),
        })
    return rows


def run_diagnostics_path(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    path = run_dir / "midpoint_diagnostics.json"
    if path.exists():
        return str(path)
    path = run_dir / "diagnostics.json"
    return str(path) if path.exists() else None

def run_failure_path(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    path = run_dir / "failure.json"
    return str(path) if path.exists() else None

def off_control_nonfinite_warning(off_class: dict[str, str], off_run: Path | None, cfg: dict[str, Any], seed: int) -> str | None:
    if off_class.get("midpoint_status") == "finite_midpoint" and off_class.get("evidence_status") == "finite_logZ":
        return None
    if off_class.get("midpoint_status") not in {"nonfinite_midpoint", "missing_midpoint"} and off_class.get("evidence_status") not in {"nonfinite_logZ", "missing_logZ"}:
        return None
    diag_path = run_diagnostics_path(off_run)
    failure_path = run_failure_path(off_run)
    retry = ""
    if cfg.get("off_retry_n_unlensed_inj"):
        retry = f"; consider rerunning seed {seed} with mock.n_unlensed_inj={cfg['off_retry_n_unlensed_inj']}"
    else:
        retry = f"; consider rerunning seed {seed} with a larger mock.n_unlensed_inj"
    return (
        "off-control produced nonfinite midpoint/logZ; delta_logZ unavailable; "
        f"inspect off diagnostics: {diag_path or 'missing'}"
        + (f"; failure: {failure_path}" if failure_path else "")
        + retry
    )

def inference_cmd(case_dir: Path, run_dir: Path, spec: dict[str, Any], cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    diagnostics_only = bool(cfg.get("diagnostics_only", args.diagnostics_only))
    # 0 = run to the dlogz criterion. The old hardcoded 5000-call cap
    # silently truncated every pilot/production posterior ("sampling was
    # stopped short"): ~5000 rwalk calls past init is ~200 iterations at
    # nlive 512 — ESS ~ 1 at paper scale.
    max_samples = "0" if diagnostics_only else str(cfg.get("max_samples") or 0)
    nlive = args.nlive or cfg["nlive"]
    fix_lens_rate = str(cfg.get("fix_lens_rate", False)).lower()
    fixed_parameter_values = json.dumps(cfg.get("fixed_parameter_values", {"tau_n": 3.0}))
    lens_prior_overrides = json.dumps(cfg.get("lens_prior_overrides", {"log10_tau_A": [-5.0, -2.5]}))
    cmd = [sys.executable, "-m", "darksirens.cli.inference_lensing", "--gw_path", str(case_dir / "mock_observed_gw_pe.h5"),
           "--observed_catalog_path", str(case_dir / "observed_catalog.json"), "--gwselection_path", str(case_dir / "mock_gw_selection.h5"),
           "--lensed_injections_path", str(case_dir / "mock_lensed_injections.h5"), "--pair_metadata_path", str(case_dir / "mock_pair_metadata.h5"),
           "--candidate_pairs_path", str(case_dir / "candidate_pairs.json"), "--partition_mode", str(cfg["partition_mode"]), "--partition_component_mode", str(cfg.get("partition_component_mode", "componentwise")), "--max_exact_partitions", str(cfg.get("max_exact_partitions", 10000)), "--cluster_mode", "j2",
           "--wl_backend", "lognormal", "--pop_model", str(cfg.get("pop_model", "powerlaw+peak")), "--fix_cosmology", "true", "--fix_survey", "true",
           "--fix_population", str(bool(cfg.get("fix_population", True))).lower(),
           "--singleton_lensing", str(cfg.get("singleton_lensing", "off")),
           "--fix_lens_rate", fix_lens_rate, "--fixed_parameter_values", fixed_parameter_values, "--lens_prior_overrides", lens_prior_overrides,
           "--sampler", str(cfg["sampler"]), "--nlive", str(nlive), "--dlogz", str(cfg["dlogz"]), "--max_samples", max_samples, "--pe_max_per_pair", str(cfg["pe_max"]),
           *(["--sel_batch_size", str(cfg["sel_batch_size"])] if cfg.get("sel_batch_size") else []),
           "--seed", str(cfg["seed"]), "--pair_marks", spec["pair_marks"], "--pair_tag_model", spec["pair_tag_model"],
           "--pair_tag_constant", str(spec["pair_tag_constant"]), "--pair_tag_perturb_logit", str(spec["pair_tag_perturb_logit"]),
           "--edge_mark_prior_keys", spec["edge_mark_prior_keys_csv"], "--save_path", str(run_dir)]
    if cfg.get("prior_overrides"):
        cmd.extend(["--prior_overrides", json.dumps(cfg["prior_overrides"])])
    for flag, key in (
        ("--max_component_events", "max_component_events"),
        ("--max_component_edges", "max_component_edges"),
        ("--max_component_partitions", "max_component_partitions"),
        ("--max_total_partitions", "max_total_partitions"),
    ):
        value = cfg.get(key)
        if value is not None:
            cmd.extend([flag, str(value)])
    if diagnostics_only:
        cmd[cmd.index("--nlive") + 1] = "8"; cmd[cmd.index("--dlogz") + 1] = "50"
    return cmd

def off_control_cmd(case_dir: Path, run_dir: Path, cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    diagnostics_only = bool(cfg.get("diagnostics_only", args.diagnostics_only))
    # 0 = run to the dlogz criterion. The old hardcoded 5000-call cap
    # silently truncated every pilot/production posterior ("sampling was
    # stopped short"): ~5000 rwalk calls past init is ~200 iterations at
    # nlive 512 — ESS ~ 1 at paper scale.
    max_samples = "0" if diagnostics_only else str(cfg.get("max_samples") or 0)
    nlive = args.nlive or cfg["nlive"]
    cmd = [sys.executable, "-m", "darksirens.cli.inference_lensing", "--gw_path", str(case_dir / "mock_observed_gw_pe.h5"),
           "--observed_catalog_path", str(case_dir / "observed_catalog.json"), "--gwselection_path", str(case_dir / "mock_gw_selection.h5"),
           "--cluster_mode", "off", "--wl_backend", "lognormal", "--pop_model", str(cfg.get("pop_model", "powerlaw+peak")),
           "--fix_cosmology", "true", "--fix_survey", "true",
           "--fix_population", str(bool(cfg.get("fix_population", True))).lower(),
           "--fix_lens_rate", "true",
           "--sampler", str(cfg["sampler"]), "--nlive", str(nlive), "--dlogz", str(cfg["dlogz"]), "--max_samples", max_samples,
           "--seed", str(cfg["seed"]), "--save_path", str(run_dir)]
    if cfg.get("prior_overrides"):
        cmd.extend(["--prior_overrides", json.dumps(cfg["prior_overrides"])])
    if diagnostics_only:
        cmd[cmd.index("--nlive") + 1] = "8"; cmd[cmd.index("--dlogz") + 1] = "50"
    return cmd


def singles_only_cmd(case_dir: Path, run_dir: Path, cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Mould-style per-event-only ablation: no pair channel, but observed
    singletons carry the lensed-single-image mixture (cluster off +
    sl_mixture), with the same population/lens-rate treatment as the main
    run. Requires a mock generated with include_lensed_singletons."""
    diagnostics_only = bool(cfg.get("diagnostics_only", args.diagnostics_only))
    # 0 = run to the dlogz criterion. The old hardcoded 5000-call cap
    # silently truncated every pilot/production posterior ("sampling was
    # stopped short"): ~5000 rwalk calls past init is ~200 iterations at
    # nlive 512 — ESS ~ 1 at paper scale.
    max_samples = "0" if diagnostics_only else str(cfg.get("max_samples") or 0)
    nlive = args.nlive or cfg["nlive"]
    fix_lens_rate = str(cfg.get("fix_lens_rate", False)).lower()
    fixed_parameter_values = json.dumps(cfg.get("fixed_parameter_values", {"tau_n": 3.0}))
    lens_prior_overrides = json.dumps(cfg.get("lens_prior_overrides", {"log10_tau_A": [-5.0, -2.5]}))
    cmd = [sys.executable, "-m", "darksirens.cli.inference_lensing", "--gw_path", str(case_dir / "mock_observed_gw_pe.h5"),
           "--observed_catalog_path", str(case_dir / "observed_catalog.json"), "--gwselection_path", str(case_dir / "mock_gw_selection.h5"),
           "--lensed_injections_path", str(case_dir / "mock_lensed_injections.h5"),
           "--cluster_mode", "off", "--wl_backend", "lognormal", "--pop_model", str(cfg.get("pop_model", "powerlaw+peak")),
           "--singleton_lensing", "sl_mixture",
           "--fix_cosmology", "true", "--fix_survey", "true",
           "--fix_population", str(bool(cfg.get("fix_population", True))).lower(),
           "--fix_lens_rate", fix_lens_rate, "--fixed_parameter_values", fixed_parameter_values, "--lens_prior_overrides", lens_prior_overrides,
           "--sampler", str(cfg["sampler"]), "--nlive", str(nlive), "--dlogz", str(cfg["dlogz"]), "--max_samples", max_samples,
           "--seed", str(cfg["seed"]), "--save_path", str(run_dir)]
    if cfg.get("prior_overrides"):
        cmd.extend(["--prior_overrides", json.dumps(cfg["prior_overrides"])])
    if diagnostics_only:
        cmd[cmd.index("--nlive") + 1] = "8"; cmd[cmd.index("--dlogz") + 1] = "50"
    return cmd

def off_control_retry_cmd(case_dir: Path, run_dir: Path, cfg: dict[str, Any], args: argparse.Namespace) -> list[str] | None:
    if not cfg.get("off_retry_on_nonfinite", False):
        return None
    cmd = off_control_cmd(case_dir, run_dir, cfg, args)
    retry_nlive = cfg.get("off_retry_nlive")
    if retry_nlive is not None:
        cmd[cmd.index("--nlive") + 1] = str(retry_nlive)
    return cmd

def preflight_cmd(cmd: list[str], path: Path) -> list[str]:
    return [*cmd, "--preflight_only", "true", "--preflight_json", str(path)]

def preflight_status(j2_return_code: int, j2_report: dict[str, Any], off_return_code: int | None = None, off_report: dict[str, Any] | None = None, run_off_controls: bool = False) -> str:
    if j2_return_code or not j2_report.get("ok", False):
        return "failed_preflight"
    if run_off_controls and (off_return_code or not (off_report or {}).get("ok", False)):
        return "failed_off_preflight"
    return "passed_preflight"

def combined_inference_status(j2_status: str, off_status: str, run_off_controls: bool = False) -> str:
    j2_failed = j2_status != "passed"
    off_failed = run_off_controls and off_status != "passed"
    if j2_failed and off_failed:
        return "failed_both"
    if j2_failed:
        return "failed_j2_inference"
    if off_failed:
        return "failed_off_inference"
    return "passed"

def _numeric_summary(values: list[float]) -> dict[str, float | None]:
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return {"min": None, "median": None, "max": None}
    return {"min": float(np.min(vals)), "median": float(np.median(vals)), "max": float(np.max(vals))}

def audit_candidate_graph(candidate_pairs_path: str | Path, observed_catalog_path: str | Path) -> dict[str, Any]:
    """Audit whether a simulated observed-event candidate graph kept truth edges.

    Truth fields from the observed catalog are used only here, after graph construction,
    for validation/evaluation and are not added to inference inputs.
    """
    candidate_pairs_path = Path(candidate_pairs_path)
    observed_catalog_path = Path(observed_catalog_path)
    cand = _load_json(candidate_pairs_path, {}) or {}
    n_events, pairs = validate_candidate_pairs(cand)
    pair_edges = {(min(p.i, p.j), max(p.i, p.j)) for p in pairs}
    true_edges = true_edges_from_catalog(observed_catalog_path)
    kept_true = pair_edges & true_edges
    false_edges = pair_edges - true_edges
    degrees = [0] * n_events
    for p in pairs:
        degrees[p.i] += 1; degrees[p.j] += 1
    components = connected_components_from_candidate_pairs(n_events, pairs)
    component_sizes = [len(c["event_indices"]) for c in components]
    component_edge_counts = [len(c["candidate_edge_indices"]) for c in components]
    component_partition_counts: list[int | None] = []
    for comp in components:
        try:
            component_partition_counts.append(len(exact_component_partitions(comp, pairs, max_partitions=100_000)))
        except Exception:
            component_partition_counts.append(None)
    mark_keys = sorted({k for p in pairs for k in p.marks.to_dict()})
    mark_summary = {key: _numeric_summary([p.marks.to_dict().get(key) for p in pairs]) for key in ("log_sky_overlap", "log_mass_distance_score", "delta_t_obs") if key in mark_keys}
    true_false_mark_summary = {}
    for key in ("log_sky_overlap", "log_mass_distance_score", "delta_t_obs", "log_prior_odds"):
        if key == "log_prior_odds":
            true_vals = [p.log_prior_odds for p in pairs if (p.i, p.j) in true_edges]
            false_vals = [p.log_prior_odds for p in pairs if (p.i, p.j) not in true_edges]
        else:
            true_vals = [p.marks.to_dict().get(key) for p in pairs if (p.i, p.j) in true_edges]
            false_vals = [p.marks.to_dict().get(key) for p in pairs if (p.i, p.j) not in true_edges]
        if true_vals or false_vals:
            true_false_mark_summary[key] = {"true": _numeric_summary(true_vals), "false": _numeric_summary(false_vals)}
    return {
        "candidate_pairs_path": str(candidate_pairs_path),
        "observed_catalog_path": str(observed_catalog_path),
        "n_events": n_events,
        "n_candidate_edges": len(pairs),
        "n_true_edges_in_catalog": len(true_edges),
        "n_true_edges_in_candidate_graph": len(kept_true),
        "true_edge_survival_fraction": (len(kept_true) / len(true_edges)) if true_edges else None,
        "missing_true_edges": [list(e) for e in sorted(true_edges - pair_edges)],
        "n_false_edges": len(false_edges),
        "max_edges_per_event_observed": max(degrees) if degrees else 0,
        "n_components": len(components),
        "component_sizes": component_sizes,
        "component_edge_counts": component_edge_counts,
        "component_partition_counts": component_partition_counts,
        "approximate_total_partitions": math.prod(c for c in component_partition_counts if c is not None) if component_partition_counts and all(c is not None for c in component_partition_counts) else None,
        "mark_key_availability": {key: sum(1 for p in pairs if key in p.marks.to_dict()) for key in mark_keys},
        "available_mark_keys": mark_keys,
        "mark_summary_by_key": mark_summary,
        "true_vs_false_mark_summaries": true_false_mark_summary,
        "truth_label_source": "observed_catalog truth_* fields",
    }

def _candidate_audit_csv_row(case: str, audit: dict[str, Any]) -> dict[str, Any]:
    return {"case": case, "n_events": audit.get("n_events"), "n_candidate_edges": audit.get("n_candidate_edges"), "n_true_edges": audit.get("n_true_edges_in_catalog"), "n_true_edges_kept": audit.get("n_true_edges_in_candidate_graph"), "true_edge_survival_fraction": audit.get("true_edge_survival_fraction"), "n_false_edges": audit.get("n_false_edges"), "n_components": audit.get("n_components"), "max_component_events": max(audit.get("component_sizes") or [0]), "max_component_edges": max(audit.get("component_edge_counts") or [0]), "max_component_partitions": max([x for x in (audit.get("component_partition_counts") or []) if x is not None] or [None]), "available_mark_keys": ";".join(audit.get("available_mark_keys") or [])}

def latest_run(run_root: Path) -> Path | None:
    ds = sorted(run_root.glob("**/diagnostics.json"), key=lambda p: p.stat().st_mtime)
    return ds[-1].parent if ds else None

def latest_attempt(run_root: Path) -> Path | None:
    ds = sorted(list(run_root.glob("**/diagnostics.json")) + list(run_root.glob("**/failure.json")), key=lambda p: p.stat().st_mtime)
    return ds[-1].parent if ds else None

def _log_path(log: dict[str, Any], key: str) -> str | None:
    return log.get(key) if isinstance(log, dict) else None

def true_edges_from_catalog(catalog_path: Path) -> set[tuple[int,int]]:
    cat = _load_json(catalog_path, {}) or {}; events = cat.get("events", [])
    out = set()
    for i in range(len(events)):
        for j in range(i+1, len(events)):
            if events[i].get("truth_source_id") == events[j].get("truth_source_id") and events[i].get("truth_is_lensed_image") and events[j].get("truth_is_lensed_image") and events[i].get("truth_image_index") != events[j].get("truth_image_index"):
                out.add((i,j))
    return out

def posterior_probability_items(diagnostics: dict[str, Any], candidate_path: Path) -> list[tuple[tuple[int,int], float]]:
    probs = diagnostics.get("posterior_pair_probabilities") or {}
    candidate_data = _load_json(candidate_path, {}) or {}
    cand = candidate_data.get("pairs") or candidate_data.get("candidate_pairs") or []
    items = []
    if isinstance(probs, dict):
        for k, v in probs.items():
            parts = str(k).replace(",", "-").split("-")
            if len(parts) >= 2 and all(p.strip().lstrip("-").isdigit() for p in parts[:2]):
                items.append(((min(int(parts[0]), int(parts[1])), max(int(parts[0]), int(parts[1]))), float(v)))
    elif all(isinstance(entry, dict) for entry in probs):
        for idx, entry in enumerate(probs):
            missing = {key for key in ("i", "j", "p_pair") if key not in entry}
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(f"posterior_pair_probabilities[{idx}] is missing required key(s): {missing_text}")
            try:
                i = int(entry["i"])
                j = int(entry["j"])
                probability = float(entry["p_pair"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"posterior_pair_probabilities[{idx}] must have integer i/j and float p_pair") from exc
            items.append(((min(i, j), max(i, j)), probability))
    else:
        for e, v in zip(cand, probs):
            items.append(((min(int(e["i"]), int(e["j"])), max(int(e["i"]), int(e["j"]))), float(v)))
    return items

def _normalized_pair_set(pairs: Any) -> set[tuple[int, int]]:
    return {tuple(sorted(map(int, x))) for x in pairs if isinstance(x, (list, tuple)) and len(x) == 2}


def recovery_metrics(true_edges: set[tuple[int,int]], posterior_items: list[tuple[tuple[int,int], float]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    p = dict(posterior_items); false = [v for e, v in posterior_items if e not in true_edges]; truep = [p.get(e, 0.0) for e in true_edges]
    map_partition = diagnostics.get("map_partition") if isinstance(diagnostics.get("map_partition"), dict) else {}
    map_pairs_source = None
    if "pair_indices" in map_partition:
        map_pairs_source = map_partition.get("pair_indices")
    elif "pairs" in map_partition:
        map_pairs_source = map_partition.get("pairs")
    elif "map_pairs" in diagnostics:
        map_pairs_source = diagnostics.get("map_pairs")
    map_set = _normalized_pair_set(map_pairs_source or [])
    if "map_n_pairs" in diagnostics:
        map_n_pairs = diagnostics.get("map_n_pairs")
    elif "n_pairs" in map_partition:
        map_n_pairs = map_partition.get("n_pairs")
    elif map_pairs_source is not None:
        map_n_pairs = len(map_set)
    else:
        map_n_pairs = None
    return {"injected_n_pairs": len(true_edges), "expected_n_pairs": diagnostics.get("expected_n_pairs"), "map_n_pairs": map_n_pairs,
            "true_edge_posterior_probability_mean": float(np.mean(truep)) if truep else None,
            "false_edge_posterior_probability_max": float(np.max(false)) if false else 0.0,
            "false_edge_posterior_probability_sum": float(np.sum(false)) if false else 0.0,
            "map_partition_exact_truth_match": (map_set == true_edges) if map_set or true_edges else None}

def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def build_plan(args: argparse.Namespace, cfg: dict[str, Any], resolved_config: dict[str, Any], work: Path) -> dict[str, Any]:
    cases = {}
    selected_cases = resolved_config["study"].get("cases") or CASE_ORDER
    for n, name in enumerate(selected_cases):
        spec = _case_spec(name, cfg); cdir = work / "cases" / name; rdir = work / "runs" / name
        icmd = inference_cmd(cdir, rdir, spec, cfg, args)
        validate_known_inference_flags(icmd)
        case_plan = {"spec": spec, "generate": generate_cmd(cdir, spec, cfg, args.seed + 101*n), "build_graph": build_graph_cmd(cdir, spec, args.seed + 1001*n), "preflight": preflight_cmd(icmd, rdir / "preflight.json"), "inference": icmd}
        if args.run_off_controls:
            off_dir = work / "runs" / f"{name}__off"
            ocmd = off_control_cmd(cdir, off_dir, cfg, args)
            validate_known_inference_flags(ocmd)
            case_plan["off"] = ocmd
            case_plan["off_preflight"] = preflight_cmd(ocmd, off_dir / "preflight.json")
        if cfg.get("run_singles_only_ablation", False):
            singles_dir = work / "runs" / f"{name}__singles"
            scmd = singles_only_cmd(cdir, singles_dir, cfg, args)
            validate_known_inference_flags(scmd)
            case_plan["singles_only"] = scmd
        cases[name] = case_plan
    return {"created_at": _utc(), "profile": resolved_config["study"]["profile"], "diagnostics_only": cfg["diagnostics_only"], "run_off_controls": args.run_off_controls, "resolved_config": resolved_config, "cases": cases}

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--override", action="append", default=[], help="Override config value with dotted.key=value; may be repeated")
    ap.add_argument("--allow_unknown_config_keys", type=_str2bool, default=False)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--profile", choices=PROFILES, default="tiny")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sampler", choices=["dynesty", "tinyns"], default="dynesty")
    ap.add_argument("--nlive", type=int)
    ap.add_argument("--dlogz", type=float, default=10.0)
    ap.add_argument("--diagnostics_only", type=_str2bool, default=False)
    ap.add_argument("--reuse", type=_str2bool, default=False)
    ap.add_argument("--dry_run", type=_str2bool, default=False)
    ap.add_argument("--preflight_only", type=_str2bool, default=False)
    ap.add_argument("--run_off_controls", type=_str2bool, default=True)
    ap.add_argument("--require_true_edge_survival", type=_str2bool, default=False)
    args = ap.parse_args(argv)
    resolved_config = resolve_config(args.config, args.override, profile=args.profile, allow_unknown=args.allow_unknown_config_keys)
    if args.seed != 2026:
        resolved_config["study"]["seed"] = args.seed
    if args.sampler != "dynesty":
        resolved_config["inference"]["sampler"] = args.sampler
    if args.dlogz != 10.0:
        resolved_config["inference"]["dlogz"] = args.dlogz
    if args.diagnostics_only:
        resolved_config["inference"]["diagnostics_only"] = True
    args.profile = resolved_config["study"]["profile"]
    args.seed = int(resolved_config["study"]["seed"])
    args.sampler = resolved_config["inference"]["sampler"]
    args.dlogz = float(resolved_config["inference"]["dlogz"])
    args.diagnostics_only = bool(resolved_config["inference"]["diagnostics_only"])
    mock = resolved_config["mock"]; graph = resolved_config["candidate_graph"]; inf = resolved_config["inference"]
    cfg = {"n_universe": mock["n_universe"], "n_sing": mock["n_singletons"], "n_pair": mock["n_lensed_pairs"], "nsamp": mock["nsamp"], "n_unlensed_inj": mock["n_unlensed_inj"], "n_lensed_inj": mock["n_lensed_inj"], "conditioning": mock["conditioning"], "pop_model": mock.get("pop_model", "powerlaw+peak"), "include_lensed_singletons": bool(mock.get("include_lensed_singletons", False)), "tau_A": mock.get("tau_A"), "tau_n": mock.get("tau_n"), "observation_times": mock.get("observation_times", "placeholder"), "t_obs_days": mock.get("t_obs_days", 365.25), "pe_max": (inf.get("pe_max_per_pair") or min(mock["nsamp"], 512)), "seed": resolved_config["study"]["seed"], **graph, **resolved_config["selection"], **inf}
    work = Path(args.workdir).resolve(); work.mkdir(parents=True, exist_ok=True)
    write_config(work / "resolved_config.yaml", resolved_config)
    plan = build_plan(args, cfg, resolved_config, work)
    (work / "run_manifest.json").write_text(json.dumps(plan, indent=2, allow_nan=True) + "\n")
    if args.dry_run:
        (work / "validation_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=True) + "\n"); return 0
    summary = {"created_at": _utc(), "profile": args.profile, "diagnostics_only": args.diagnostics_only, "preflight_only": args.preflight_only, "resolved_config": resolved_config, "diagnostics_only_note": "diagnostics_only: evidence deltas are not meaningful" if args.diagnostics_only else None, "run_off_controls": args.run_off_controls, "cases": {}}
    pair_rows=[]; comp_rows=[]; truth_rows=[]; bias_rows=[]; audit_rows=[]; partition_rows=[]; component_partition_rows=[]; hyper_rows=[]
    for name, entry in plan["cases"].items():
        cdir = work / "cases" / name; rdir = work / "runs" / name
        if not args.reuse or not (cdir / "mock_observed_gw_pe.h5").exists():
            if cdir.exists(): shutil.rmtree(cdir)
            rt, rc = _run(entry["generate"])
            if rc:
                summary["cases"][name] = {"status":"failed_generate", "return_code":rc, "command":shlex.join(entry["inference"]), "preflight_command":shlex.join(entry["preflight"]), "generated_files": [], "candidate_graph_summary": _candidate_graph_summary({}, {}, entry["spec"]), "warnings": [], "errors": ["mock generation command failed"]}
                continue
        rt, rc = _run(entry["build_graph"])
        if rc:
            summary["cases"][name] = {"status":"failed_build_graph", "return_code":rc, "command":shlex.join(entry["inference"]), "preflight_command":shlex.join(entry["preflight"]), "generated_files": [str(cdir / x) for x in ["mock_observed_gw_pe.h5", "observed_catalog.json", "mock_gw_selection.h5", "mock_lensed_injections.h5"]], "candidate_graph_summary": _candidate_graph_summary({}, {}, entry["spec"]), "warnings": [], "errors": ["candidate graph build command failed"]}
            continue
        rdir.mkdir(parents=True, exist_ok=True)
        audit = audit_candidate_graph(cdir / "candidate_pairs.json", cdir / "observed_catalog.json")
        (cdir / "candidate_graph_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=True) + "\n")
        audit_rows.append(_candidate_audit_csv_row(name, audit))
        if args.require_true_edge_survival and audit["n_true_edges_in_candidate_graph"] < audit["n_true_edges_in_catalog"]:
            summary["cases"][name] = {"status":"failed_candidate_graph_audit", "candidate_graph_audit": audit, "candidate_graph_summary": audit, "command":shlex.join(entry["inference"]), "preflight_command":shlex.join(entry["preflight"]), "warnings": [], "errors": ["candidate graph audit failed: missing injected true edge"]}
            continue
        fc = _file_contract_report(cdir, work / "resolved_config.yaml")
        (rdir / "file_contract_report.json").write_text(json.dumps(fc, indent=2, allow_nan=True) + "\n")
        if not fc.get("ok", False):
            summary["cases"][name] = {"status":"failed_preflight", "file_contract_report":fc, "candidate_graph_summary":_candidate_graph_summary({}, fc, entry["spec"]), "command":shlex.join(entry["inference"]), "preflight_command":shlex.join(entry["preflight"]), "generated_files": [str(cdir / x) for x in ["mock_observed_gw_pe.h5", "observed_catalog.json", "candidate_pairs.json", "mock_gw_selection.h5", "mock_lensed_injections.h5"]], "warnings":[], "errors":[f"file contract failed: {k}" for k,v in fc.items() if isinstance(v, dict) and not v.get("ok", True)]}; continue
        pre_log = _run_logged(entry["preflight"], rdir / "preflight"); prt, prc = pre_log["runtime_sec"], pre_log["return_code"]; pre = _load_json(rdir / "preflight.json", {})
        off_pre_log = {}
        off_pre = {}; off_prc = None; off_dir = work / "runs" / f"{name}__off"
        if args.run_off_controls:
            off_dir.mkdir(parents=True, exist_ok=True)
            off_pre_log = _run_logged(entry["off_preflight"], off_dir / "preflight")
            _oprt, off_prc = off_pre_log["runtime_sec"], off_pre_log["return_code"]
            off_pre = _load_json(off_dir / "preflight.json", {})
        status = preflight_status(prc, pre, off_prc, off_pre, args.run_off_controls)
        warnings = list(pre.get("warnings", [])) + list(off_pre.get("warnings", []))
        errors = list(pre.get("errors", [])) + list(off_pre.get("errors", []))
        base_rec = {"status": status, "return_code":prc, "off_return_code": off_prc, "preflight_report":pre, "off_preflight_report": off_pre if args.run_off_controls else None, "file_contract_report":fc, "candidate_graph_summary":_candidate_graph_summary(pre, fc, entry["spec"]), "candidate_graph_audit": audit, "command":shlex.join(entry["inference"]), "preflight_command":shlex.join(entry["preflight"]), "off_command": shlex.join(entry["off"]) if args.run_off_controls else None, "off_preflight_command": shlex.join(entry["off_preflight"]) if args.run_off_controls else None, "generated_files": [str(cdir / x) for x in ["mock_observed_gw_pe.h5", "observed_catalog.json", "candidate_pairs.json", "mock_gw_selection.h5", "mock_lensed_injections.h5"]], "warnings":warnings, "errors":errors, "preflight_stdout_path": _log_path(pre_log, "stdout_path"), "preflight_stderr_path": _log_path(pre_log, "stderr_path"), "off_preflight_stdout_path": _log_path(locals().get("off_pre_log", {}), "stdout_path"), "off_preflight_stderr_path": _log_path(locals().get("off_pre_log", {}), "stderr_path"), "j2": {"stdout_path": _log_path(pre_log, "stdout_path"), "stderr_path": _log_path(pre_log, "stderr_path"), "status": "passed_preflight" if status != "failed_preflight" else "failed_preflight", "return_code": prc, "preflight_report": pre, "run_dir": str(rdir)}, "off": {"stdout_path": _log_path(locals().get("off_pre_log", {}), "stdout_path"), "stderr_path": _log_path(locals().get("off_pre_log", {}), "stderr_path"), "status": ("skipped" if not args.run_off_controls else ("passed_preflight" if off_prc == 0 and off_pre.get("ok", False) else "failed_preflight")), "return_code": off_prc, "preflight_report": off_pre if args.run_off_controls else None, "run_dir": str(off_dir) if args.run_off_controls else None}}
        if status != "passed_preflight": summary["cases"][name] = base_rec; continue
        if args.preflight_only:
            summary["cases"][name] = base_rec; continue
        inf_log = _run_logged(entry["inference"], rdir / "inference")
        rt, rc = inf_log["runtime_sec"], inf_log["return_code"]
        run = latest_attempt(rdir); diag = _load_json(run / "diagnostics.json", {}) if run else {}; mid_diag = _load_json(run / "midpoint_diagnostics.json", _load_json(run / "midpoint.json", {})) if run else {}; failure = _load_json(run / "failure.json", None) if run else None
        results_attrs = read_results_attrs(run)
        off_log = {}; off_failure = None
        off_run = None; off_diag = {}; off_mid_diag = {}; off_attrs = {}; off_rc = None; off_status = "skipped"
        if args.run_off_controls:
            off_dir.mkdir(parents=True, exist_ok=True)
            off_log = _run_logged(entry["off"], off_dir / "inference")
            _ot, off_rc = off_log["runtime_sec"], off_log["return_code"]
            off_run = latest_attempt(off_dir); off_diag = _load_json(off_run / "diagnostics.json", {}) if off_run else {}; off_mid_diag = _load_json(off_run / "midpoint_diagnostics.json", _load_json(off_run / "midpoint.json", {})) if off_run else {}; off_failure = _load_json(off_run / "failure.json", None) if off_run else None; off_attrs = read_results_attrs(off_run)
            off_status = "passed" if off_rc == 0 and off_run and (off_run / "diagnostics.json").exists() else "failed_inference"
            initial_off_class = classify_run_outputs(off_attrs, off_mid_diag, args.diagnostics_only)
            if off_status == "passed" and cfg.get("off_retry_on_nonfinite", False) and (
                initial_off_class["midpoint_status"] != "finite_midpoint" or initial_off_class["evidence_status"] != "finite_logZ"
            ):
                retry_cmd = off_control_retry_cmd(cdir, off_dir, cfg, args)
                if retry_cmd is not None and retry_cmd != entry["off"]:
                    retry_prefix = off_dir / "inference_retry"
                    off_log = _run_logged(retry_cmd, retry_prefix)
                    _ot, off_rc = off_log["runtime_sec"], off_log["return_code"]
                    off_run = latest_attempt(off_dir); off_diag = _load_json(off_run / "diagnostics.json", {}) if off_run else {}; off_mid_diag = _load_json(off_run / "midpoint_diagnostics.json", _load_json(off_run / "midpoint.json", {})) if off_run else {}; off_failure = _load_json(off_run / "failure.json", None) if off_run else None; off_attrs = read_results_attrs(off_run)
                    off_status = "passed" if off_rc == 0 and off_run and (off_run / "diagnostics.json").exists() else "failed_inference"
                else:
                    warnings.append("off_retry_on_nonfinite requested, but no safer off-control retry settings were provided; J2 was not rerun")
        singles_log = {}; singles_run = None; singles_attrs = {}; singles_rc = None; singles_status = "skipped"
        if "singles_only" in entry:
            singles_dir = work / "runs" / f"{name}__singles"
            singles_dir.mkdir(parents=True, exist_ok=True)
            singles_log = _run_logged(entry["singles_only"], singles_dir / "inference")
            _st, singles_rc = singles_log["runtime_sec"], singles_log["return_code"]
            singles_run = latest_attempt(singles_dir)
            singles_attrs = read_results_attrs(singles_run)
            singles_status = "passed" if singles_rc == 0 and singles_run and (singles_run / "diagnostics.json").exists() else "failed_inference"
        logz_j2 = extract_logz(results_attrs); logzerr_j2 = extract_logzerr(results_attrs)
        logz_off = extract_logz(off_attrs); logzerr_off = extract_logzerr(off_attrs)
        delta_logz = evidence_delta(logz_j2, logz_off, args.diagnostics_only)
        delta_logzerr = math.sqrt(logzerr_j2**2 + logzerr_off**2) if (delta_logz is not None and logzerr_j2 is not None and logzerr_off is not None) else None
        true_edges = true_edges_from_catalog(cdir / "observed_catalog.json")
        items = posterior_probability_items(diag, cdir / "candidate_pairs.json")
        rec = recovery_metrics(true_edges, items, diag)
        by_edge = {(int(x.get("i")), int(x.get("j"))): x for x in diag.get("posterior_pair_probabilities", []) if isinstance(x, dict) and "i" in x and "j" in x}
        for (i,j), prob in items:
            extra = by_edge.get((i, j), by_edge.get((j, i), {}))
            pair_rows.append({
                "case": name, "i": i, "j": j, "posterior_probability": prob, "is_true_edge": (i,j) in true_edges,
                "log_prior_odds_raw": extra.get("log_prior_odds_raw"),
                "log_prior_odds_effective": extra.get("log_prior_odds_effective", extra.get("log_prior_odds")),
                "edge_mark_prior_contribution": extra.get("edge_mark_prior_contribution"),
                "pair_time_delta_t_obs": extra.get("pair_time_delta_t_obs"),
                "pair_time_sigma": extra.get("pair_time_sigma"),
                "pair_time_placeholder_warning": extra.get("pair_time_placeholder_warning"),
                "marks_json": json.dumps(extra.get("marks", {}), sort_keys=True),
            })
        partition_rows.extend(partition_diagnostic_rows(diag, case=name, truth_edges=true_edges))
        component_partition_rows.extend(component_partition_diagnostic_rows(diag, case=name, truth_edges=true_edges))
        truth_rows.append({"case": name, **rec})
        case_truth = _load_json(cdir / "truth.json", {}) or {}
        truth_vals = hyperparameter_truth_values(case_truth, str(cfg.get("pop_model", "powerlaw+peak")))
        hyper_rows.extend(hyperparameter_recovery_rows(name, "j2", results_attrs, truth_vals))
        if args.run_off_controls:
            hyper_rows.extend(hyperparameter_recovery_rows(name, "off", off_attrs, truth_vals))
        if singles_status != "skipped":
            hyper_rows.extend(hyperparameter_recovery_rows(name, "singles_only", singles_attrs, truth_vals))
        bias_rows.append({"case": name, "pair_tag_perturb_logit": entry["spec"]["pair_tag_perturb_logit"], "p_tag_model": entry["spec"]["pair_tag_model"], "delta_expected_n_pairs_minus_injected": (float(rec["expected_n_pairs"]) - rec["injected_n_pairs"]) if rec.get("expected_n_pairs") is not None else None, "logZ_j2": logz_j2, "logZ_off": logz_off, "delta_logZ_j2_minus_off": delta_logz})
        cand = _load_json(cdir / "candidate_pairs.json", {}) or {}; comp_rows.append({"case":name,"n_events":cand.get("n_events"),"n_candidate_edges":len(cand.get("pairs", cand.get("candidate_pairs", []))),"expected_n_pairs":rec.get("expected_n_pairs"),"map_n_pairs":rec.get("map_n_pairs")})
        j2_class = classify_run_outputs(results_attrs, mid_diag, args.diagnostics_only)
        off_class = classify_run_outputs(off_attrs, off_mid_diag, args.diagnostics_only) if args.run_off_controls else {"process_status": "passed", "midpoint_status": "missing_midpoint", "evidence_status": "missing_logZ"}
        j2_status = "passed" if rc == 0 and run and (run / "diagnostics.json").exists() else "failed_inference"
        if j2_status != "passed":
            j2_class["process_status"] = "failed_inference"
        if args.run_off_controls and off_status != "passed":
            off_class["process_status"] = "failed_inference"
        case_status = combined_inference_status(j2_status, off_status, args.run_off_controls)
        evidence_unusable = (
            not args.diagnostics_only
            and (
                (j2_status == "passed" and j2_class["evidence_status"] != "finite_logZ")
                or (args.run_off_controls and off_status == "passed" and off_class["evidence_status"] != "finite_logZ")
            )
        )
        if evidence_unusable:
            case_status = "failed_unusable_evidence"
        warnings = (["diagnostics_only: evidence deltas are not meaningful"] if args.diagnostics_only else []) + list(warnings)
        if len(items) > 0 and all(float(prob) == 0.0 for _edge, prob in items):
            warnings.append("all candidate edge posterior probabilities are zero; inspect partition_diagnostics.csv")
        if diag.get("candidate_time_marks_suspicious"):
            warnings.append("candidate time marks look synthetic/placeholder-like; time-mark likelihood may dominate")
        off_warning = off_control_nonfinite_warning(off_class, off_run, cfg, args.seed) if args.run_off_controls and off_status == "passed" and not args.diagnostics_only else None
        if off_warning:
            warnings.append(off_warning)
        if singles_status == "failed_inference":
            warnings.append("singles-only (Mould-style) ablation run failed; see its run dir")
        summary["cases"][name] = {"status": case_status, "return_code": rc, "off_return_code": off_rc, "run_dir": str(run) if run else None, "preflight_stdout_path": _log_path(pre_log, "stdout_path"), "preflight_stderr_path": _log_path(pre_log, "stderr_path"), "off_preflight_stdout_path": _log_path(off_pre_log, "stdout_path"), "off_preflight_stderr_path": _log_path(off_pre_log, "stderr_path"), "diagnostics": diag, "results_attrs": results_attrs, "recovery": rec, "candidate_graph_audit": audit, "j2": {"status": j2_status, "process_status": j2_class["process_status"], "run_dir": str(run) if run else None, "logZ": logz_j2, "logZerr": logzerr_j2, "midpoint_status": j2_class["midpoint_status"], "evidence_status": j2_class["evidence_status"], "failure": failure, "failure_path": run_failure_path(run), "stdout_path": _log_path(inf_log, "stdout_path"), "stderr_path": _log_path(inf_log, "stderr_path"), "diagnostics": diag, "diagnostics_path": run_diagnostics_path(run), "midpoint_diagnostics": mid_diag, "results_attrs": results_attrs}, "off": {"status": off_status, "process_status": off_class["process_status"], "run_dir": str(off_run) if off_run else None, "logZ": logz_off, "logZerr": logzerr_off, "midpoint_status": off_class["midpoint_status"], "evidence_status": off_class["evidence_status"], "failure": locals().get("off_failure"), "failure_path": run_failure_path(off_run), "stdout_path": _log_path(locals().get("off_log", {}), "stdout_path"), "stderr_path": _log_path(locals().get("off_log", {}), "stderr_path"), "diagnostics": off_diag, "diagnostics_path": run_diagnostics_path(off_run), "midpoint_diagnostics": off_mid_diag, "results_attrs": off_attrs}, "singles_only": ({"status": singles_status, "return_code": singles_rc, "run_dir": str(singles_run) if singles_run else None, "logZ": extract_logz(singles_attrs), "logZerr": extract_logzerr(singles_attrs), "results_attrs": singles_attrs, "stdout_path": _log_path(singles_log, "stdout_path"), "stderr_path": _log_path(singles_log, "stderr_path"), "command": shlex.join(entry["singles_only"])} if "singles_only" in entry else None), "hyperparameter_summaries": results_attrs.get("hyperparameter_summaries", {}), "delta_logZ_j2_minus_off": delta_logz, "delta_logZerr": delta_logzerr, "warnings": warnings, "lens_rate_posterior_summary": results_attrs.get("log10_tau_A_summary", {}), "p_tag_model_bias_summary": bias_rows[-1]}
    if args.preflight_only:
        _write_csv(work / "candidate_graph_audit.csv", audit_rows, ["case","n_events","n_candidate_edges","n_true_edges","n_true_edges_kept","true_edge_survival_fraction","n_false_edges","n_components","max_component_events","max_component_edges","max_component_partitions","available_mark_keys"])
        write_preflight_summary(work, summary)
        return 0 if all(c.get("status") == "passed_preflight" for c in summary["cases"].values()) else 1
    _write_csv(work / "candidate_graph_audit.csv", audit_rows, ["case","n_events","n_candidate_edges","n_true_edges","n_true_edges_kept","true_edge_survival_fraction","n_false_edges","n_components","max_component_events","max_component_edges","max_component_partitions","available_mark_keys"])
    _write_csv(work / "posterior_pair_probabilities.csv", pair_rows, ["case","i","j","posterior_probability","is_true_edge","log_prior_odds_raw","log_prior_odds_effective","edge_mark_prior_contribution","pair_time_delta_t_obs","pair_time_sigma","pair_time_placeholder_warning","marks_json"])
    _write_csv(work / "partition_diagnostics.csv", partition_rows, ["case","partition_index","n_pairs","pair_edges","log_likelihood","log_prior_weight","log_posterior_weight","posterior_probability","is_map_partition","is_truth_partition","n_true_edges","n_false_edges"])
    _write_csv(work / "partition_component_summary.csv", comp_rows, ["case","n_events","n_candidate_edges","expected_n_pairs","map_n_pairs"])
    _write_csv(work / "truth_recovery_summary.csv", truth_rows, ["case","injected_n_pairs","expected_n_pairs","map_n_pairs","true_edge_posterior_probability_mean","false_edge_posterior_probability_max","false_edge_posterior_probability_sum","map_partition_exact_truth_match"])
    _write_csv(work / "bias_summary.csv", bias_rows, ["case","p_tag_model","pair_tag_perturb_logit","delta_expected_n_pairs_minus_injected","logZ_j2","logZ_off","delta_logZ_j2_minus_off"])
    _write_csv(work / "hyperparameter_recovery.csv", hyper_rows, ["case","run","label","truth","mean","median","q05","q95","truth_in_90ci"])
    (work / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    lines = [
        f"# Simulated lensing study ({args.profile})",
        "",
        "Diagnostics-only: " + str(args.diagnostics_only),
        "",
        "Process status records whether the command completed; midpoint/evidence status records whether finite, usable likelihood/evidence was produced.",
        "",
        "| case | j2 status | j2 midpoint | j2 evidence | off status | off midpoint | off evidence | logZ_j2 | logZ_off | delta_logZ | warnings | expected_n_pairs | run dirs |",
        "| --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for name, rec in summary["cases"].items():
        r = rec.get("recovery", {}); j2 = rec.get("j2", {}); off = rec.get("off", {})
        warning_text = "<br>".join(str(w) for w in rec.get("warnings", []))
        lines.append(f"| {name} | {j2.get('status', rec.get('status'))} | {j2.get('midpoint_status')} | {j2.get('evidence_status')} | {off.get('status')} | {off.get('midpoint_status')} | {off.get('evidence_status')} | {j2.get('logZ')} | {off.get('logZ')} | {rec.get('delta_logZ_j2_minus_off')} | {warning_text} | {r.get('expected_n_pairs')} | {j2.get('run_dir')}<br>{off.get('run_dir')} |")
    (work / "validation_summary.md").write_text("\n".join(lines) + "\n")
    return 0 if all(c.get("status") == "passed" for c in summary["cases"].values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
