#!/usr/bin/env python
"""Run a simulated end-to-end spectral-siren lensing study.

This is mock-only.  It generates observed-event catalogs, builds candidate
pair graphs from observed events (never from truth), runs preflight and
inference, and then reads truth labels only for recovery summaries.
"""
from __future__ import annotations

import argparse, csv, datetime, json, math, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from darksirens.lensing.simulation_config import resolve_config, write_config

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
    "G_true_pairs_full_marks", "H_ambiguous_components",
]

def _str2bool(v: str | bool) -> bool:
    if isinstance(v, bool): return v
    if v.lower() in {"1","true","yes","y","on"}: return True
    if v.lower() in {"0","false","no","n","off"}: return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")

def _utc() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")

def _run(cmd: list[str]) -> tuple[float, int]:
    print("+", " ".join(map(str, cmd)), flush=True)
    t = time.time(); p = subprocess.run(cmd, cwd=ROOT); return time.time() - t, int(p.returncode)

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
        spec.update(max_edges_per_event=1, max_total_edges=max(1, cfg["n_pair"]))
    elif name.startswith("C_"):
        spec.update(max_edges_per_event=4, max_total_edges=max(8, 4 * cfg["n_pair"]))
    elif name.startswith("D_"):
        spec.update(pair_tag_model="snr_time_sky", pair_tag_perturb_logit=1.5)
    elif name.startswith("E_"):
        spec.update(include_sky_marks=False, edge_mark_prior_keys_csv="", pair_tag_model="snr_time")
    elif name.startswith("F_"):
        spec.update(include_time_marks=False, pair_marks="none", pair_tag_model="snr_time_sky")
    elif name.startswith("G_"):
        pass
    elif name.startswith("H_"):
        spec.update(max_edges_per_event=3, max_total_edges=max(6, 3 * cfg["n_pair"]), edge_mark_prior_keys_csv="")
    return spec

def generate_cmd(case_dir: Path, spec: dict[str, Any], cfg: dict[str, Any], seed: int) -> list[str]:
    return [sys.executable, str(GEN), "--outdir", str(case_dir), "--conditioning", str(cfg["conditioning"]),
            "--n-universe", str(cfg["n_universe"]), "--n-sing-keep", str(cfg["n_sing"]),
            "--n-pair-keep", str(spec["n_pair"]), "--max-sing-keep", str(cfg["n_sing"]),
            "--max-pair-keep", str(spec["n_pair"]), "--nsamp", str(cfg["nsamp"]),
            "--n-unlensed-inj", str(cfg["n_unlensed_inj"]), "--n-lensed-inj", str(cfg["n_lensed_inj"]),
            "--seed", str(seed), "--write-unified-observed-catalog", "true", "--write-legacy-pair-pe", "false"]

def build_graph_cmd(case_dir: Path, spec: dict[str, Any], seed: int) -> list[str]:
    return [sys.executable, str(BUILD), "--gw_path", str(case_dir / "mock_observed_gw_pe.h5"),
            "--observed_catalog_path", str(case_dir / "observed_catalog.json"), "--out", str(case_dir / "candidate_pairs.json"),
            "--max_edges_per_event", str(spec["max_edges_per_event"]), "--max_total_edges", str(spec["max_total_edges"]),
            "--include_time_marks", str(spec["include_time_marks"]).lower(), "--include_sky_marks", str(spec["include_sky_marks"]).lower(),
            "--include_truth_labels", "false", "--sky_overlap_weight", "0.0", "--seed", str(seed)]

def inference_cmd(case_dir: Path, run_dir: Path, spec: dict[str, Any], cfg: dict[str, Any], args: argparse.Namespace) -> list[str]:
    diagnostics_only = bool(cfg.get("diagnostics_only", args.diagnostics_only))
    max_samples = "0" if diagnostics_only else "5000"
    nlive = args.nlive or cfg["nlive"]
    cmd = [sys.executable, "-m", "darksirens.cli.inference_lensing", "--gw_path", str(case_dir / "mock_observed_gw_pe.h5"),
           "--observed_catalog_path", str(case_dir / "observed_catalog.json"), "--gwselection_path", str(case_dir / "mock_gw_selection.h5"),
           "--lensed_injections_path", str(case_dir / "mock_lensed_injections.h5"), "--pair_metadata_path", str(case_dir / "mock_pair_metadata.h5"),
           "--candidate_pairs_path", str(case_dir / "candidate_pairs.json"), "--partition_mode", str(cfg["partition_mode"]), "--cluster_mode", "j2",
           "--wl_backend", "lognormal", "--pop_model", "powerlaw+peak", "--fix_cosmology", "true", "--fix_survey", "true", "--fix_population", "true",
           "--fix_lens_rate", "false", "--fixed_parameter_values", '{"tau_n": 3.0}', "--lens_prior_overrides", '{"log10_tau_A": [-5.0, -2.5]}',
           "--sampler", str(cfg["sampler"]), "--nlive", str(nlive), "--dlogz", str(cfg["dlogz"]), "--max_samples", max_samples, "--pe_max_per_pair", str(cfg["pe_max"]),
           "--seed", str(cfg["seed"]), "--pair_marks", spec["pair_marks"], "--pair_tag_model", spec["pair_tag_model"],
           "--pair_tag_constant", str(spec["pair_tag_constant"]), "--pair_tag_perturb_logit", str(spec["pair_tag_perturb_logit"]),
           "--edge_mark_prior_keys", spec["edge_mark_prior_keys_csv"], "--save_path", str(run_dir)]
    if diagnostics_only:
        cmd[cmd.index("--nlive") + 1] = "8"; cmd[cmd.index("--dlogz") + 1] = "50"
    return cmd

def preflight_cmd(cmd: list[str], path: Path) -> list[str]:
    return [*cmd, "--preflight_only", "true", "--preflight_json", str(path)]

def latest_run(run_root: Path) -> Path | None:
    ds = sorted(run_root.glob("**/diagnostics.json"), key=lambda p: p.stat().st_mtime)
    return ds[-1].parent if ds else None

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
    cand = (_load_json(candidate_path, {}) or {}).get("pairs") or (_load_json(candidate_path, {}) or {}).get("candidate_pairs") or []
    items = []
    if isinstance(probs, dict):
        for k, v in probs.items():
            parts = str(k).replace(",", "-").split("-")
            if len(parts) >= 2 and all(p.strip().lstrip("-").isdigit() for p in parts[:2]):
                items.append(((min(int(parts[0]), int(parts[1])), max(int(parts[0]), int(parts[1]))), float(v)))
    else:
        for e, v in zip(cand, probs):
            items.append(((min(int(e["i"]), int(e["j"])), max(int(e["i"]), int(e["j"]))), float(v)))
    return items

def recovery_metrics(true_edges: set[tuple[int,int]], posterior_items: list[tuple[tuple[int,int], float]], diagnostics: dict[str, Any]) -> dict[str, Any]:
    p = dict(posterior_items); false = [v for e, v in posterior_items if e not in true_edges]; truep = [p.get(e, 0.0) for e in true_edges]
    map_pairs = diagnostics.get("map_partition_pairs") or diagnostics.get("map_pairs") or []
    map_set = {tuple(sorted(map(int, x))) for x in map_pairs if isinstance(x, (list, tuple)) and len(x) == 2}
    return {"injected_n_pairs": len(true_edges), "expected_n_pairs": diagnostics.get("expected_n_pairs"), "map_n_pairs": diagnostics.get("map_partition_n_pairs", len(map_set) if map_set else None),
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
        cases[name] = {"spec": spec, "generate": generate_cmd(cdir, spec, cfg, args.seed + 101*n), "build_graph": build_graph_cmd(cdir, spec, args.seed + 1001*n), "preflight": preflight_cmd(icmd, rdir / "preflight.json"), "inference": icmd}
    return {"created_at": _utc(), "profile": resolved_config["study"]["profile"], "diagnostics_only": cfg["diagnostics_only"], "resolved_config": resolved_config, "cases": cases}

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
    cfg = {"n_universe": mock["n_universe"], "n_sing": mock["n_singletons"], "n_pair": mock["n_lensed_pairs"], "nsamp": mock["nsamp"], "n_unlensed_inj": mock["n_unlensed_inj"], "n_lensed_inj": mock["n_lensed_inj"], "conditioning": mock["conditioning"], "pe_max": min(mock["nsamp"], 512), "seed": resolved_config["study"]["seed"], **graph, **resolved_config["selection"], **inf}
    work = Path(args.workdir).resolve(); work.mkdir(parents=True, exist_ok=True)
    write_config(work / "resolved_config.yaml", resolved_config)
    plan = build_plan(args, cfg, resolved_config, work)
    (work / "run_manifest.json").write_text(json.dumps(plan, indent=2, allow_nan=True) + "\n")
    if args.dry_run:
        (work / "validation_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=True) + "\n"); return 0
    summary = {"created_at": _utc(), "profile": args.profile, "diagnostics_only": args.diagnostics_only, "resolved_config": resolved_config, "diagnostics_only_note": "Evidence values are not meaningful when max_samples=0." if args.diagnostics_only else None, "cases": {}}
    pair_rows=[]; comp_rows=[]; truth_rows=[]; bias_rows=[]
    for name, entry in plan["cases"].items():
        cdir = work / "cases" / name; rdir = work / "runs" / name
        if not args.reuse or not (cdir / "mock_observed_gw_pe.h5").exists():
            if cdir.exists(): shutil.rmtree(cdir)
            rt, rc = _run(entry["generate"])
            if rc: summary["cases"][name] = {"status":"failed_generate", "return_code":rc}; continue
        rt, rc = _run(entry["build_graph"])
        if rc: summary["cases"][name] = {"status":"failed_build_graph", "return_code":rc}; continue
        rdir.mkdir(parents=True, exist_ok=True)
        prt, prc = _run(entry["preflight"]); pre = _load_json(rdir / "preflight.json", {})
        if prc or not pre.get("ok", False): summary["cases"][name] = {"status":"failed_preflight", "return_code":prc, "preflight_report":pre}; continue
        rt, rc = _run(entry["inference"])
        run = latest_run(rdir); diag = _load_json(run / "diagnostics.json", {}) if run else {}
        results_attrs = {}
        if run and (run / "results.hdf5").exists():
            try:
                import h5py
                with h5py.File(run / "results.hdf5", "r") as f:
                    results_attrs = {k: _jsonable(v) for k,v in f.attrs.items()}
                    labels = json.loads(results_attrs.get("labels", "[]")) if isinstance(results_attrs.get("labels"), str) else []
                    if "samples" in f and "log10_tau_A" in labels:
                        col = np.asarray(f["samples"])[:, labels.index("log10_tau_A")]
                        results_attrs["log10_tau_A_summary"] = {"mean": float(np.mean(col)), "median": float(np.median(col)), "q05": float(np.quantile(col, 0.05)), "q95": float(np.quantile(col, 0.95))}
            except Exception as exc: results_attrs = {"read_error": str(exc)}
        true_edges = true_edges_from_catalog(cdir / "observed_catalog.json")
        items = posterior_probability_items(diag, cdir / "candidate_pairs.json")
        rec = recovery_metrics(true_edges, items, diag)
        for (i,j), prob in items: pair_rows.append({"case":name,"i":i,"j":j,"posterior_probability":prob,"is_true_edge":(i,j) in true_edges})
        truth_rows.append({"case": name, **rec})
        bias_rows.append({"case": name, "pair_tag_perturb_logit": entry["spec"]["pair_tag_perturb_logit"], "p_tag_model": entry["spec"]["pair_tag_model"], "delta_expected_n_pairs_minus_injected": (float(rec["expected_n_pairs"]) - rec["injected_n_pairs"]) if rec.get("expected_n_pairs") is not None else None})
        cand = _load_json(cdir / "candidate_pairs.json", {}) or {}; comp_rows.append({"case":name,"n_events":cand.get("n_events"),"n_candidate_edges":len(cand.get("pairs", cand.get("candidate_pairs", []))),"expected_n_pairs":rec.get("expected_n_pairs"),"map_n_pairs":rec.get("map_n_pairs")})
        summary["cases"][name] = {"status":"passed" if rc == 0 and run else "failed_inference", "return_code":rc, "run_dir":str(run) if run else None, "diagnostics":diag, "results_attrs":results_attrs, "recovery":rec, "delta_logZ_j2_minus_off": None, "lens_rate_posterior_summary": results_attrs.get("log10_tau_A_summary", {}), "p_tag_model_bias_summary": bias_rows[-1]}
    _write_csv(work / "posterior_pair_probabilities.csv", pair_rows, ["case","i","j","posterior_probability","is_true_edge"])
    _write_csv(work / "partition_component_summary.csv", comp_rows, ["case","n_events","n_candidate_edges","expected_n_pairs","map_n_pairs"])
    _write_csv(work / "truth_recovery_summary.csv", truth_rows, ["case","injected_n_pairs","expected_n_pairs","map_n_pairs","true_edge_posterior_probability_mean","false_edge_posterior_probability_max","false_edge_posterior_probability_sum","map_partition_exact_truth_match"])
    _write_csv(work / "bias_summary.csv", bias_rows, ["case","p_tag_model","pair_tag_perturb_logit","delta_expected_n_pairs_minus_injected"])
    (work / "validation_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")
    lines = [f"# Simulated lensing study ({args.profile})", "", "Diagnostics-only: " + str(args.diagnostics_only), "", "| case | status | injected pairs | expected pairs | max false edge p | run dir |", "| --- | --- | ---: | ---: | ---: | --- |"]
    for name, rec in summary["cases"].items():
        r = rec.get("recovery", {}); lines.append(f"| {name} | {rec.get('status')} | {r.get('injected_n_pairs')} | {r.get('expected_n_pairs')} | {r.get('false_edge_posterior_probability_max')} | {rec.get('run_dir')} |")
    (work / "validation_summary.md").write_text("\n".join(lines) + "\n")
    return 0 if all(c.get("status") == "passed" for c in summary["cases"].values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
