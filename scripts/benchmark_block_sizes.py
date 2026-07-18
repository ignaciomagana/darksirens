#!/usr/bin/env python
"""Benchmark peak device memory vs the two likelihood block-size knobs.

Calibrates the linear peak-memory model in
``darksirens/likelihood/block_sizing.py``: for the real spectral-sirens
likelihood (gwsamples + O3/O4 injections, powerlaw+peak, fixed cosmology &
survey) it sweeps ``--sel_batch_size`` (at ``--pe_event_block off``) and
``--pe_event_block`` (at ``--sel_batch_size off``) and records the peak
``bytes_in_use`` of the value+grad evaluation, plus compile and warm times.

Each config runs in its OWN subprocess (``--worker``) so ``peak_bytes_in_use``
is that config's clean peak and an OOM in one config cannot poison the rest.
The orchestrator classifies each config as ok / OOM (RESOURCE_EXHAUSTED) /
timeout / error and writes a JSON blob + a markdown table.  Fit the sel- and
pe-axis slopes (peak vs block size) to obtain SEL_BYTES_PER_INJECTION and
PE_BYTES_PER_SAMPLE; the single-pass intercept gives FIXED_OVERHEAD_BYTES.

Usage
-----
    # smoke (validate the harness — a few configs):
    python scripts/benchmark_block_sizes.py --smoke \
        --gw-path GW.h5 --gwselection-path SEL.h5

    # full calibration:
    python scripts/benchmark_block_sizes.py --repeats 10 \
        --gw-path GW.h5 --gwselection-path SEL.h5 \
        --out scripts/benchmarks/block_sizes_<device>.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# Run-as-script guard: put the repo root (not scripts/) first so ``import
# darksirens`` resolves to THIS checkout rather than an installed copy.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_GW = "gwsamples_bbh_whitelist_all_events.h5"
_DEFAULT_SEL = "selection_o3o4ab_allsky.h5"

# Sweep grids (§PR3.5).  "off" == a single pass (no chunking on that axis).
_SEL_GRID = ["off", "32768", "65536", "131072", "262144", "524288", "1048576"]
_PE_GRID = ["off", "8", "16", "32", "64", "128", "259"]
_SMOKE_SEL = ["off", "131072"]
_SMOKE_PE = ["16"]


# ── worker: one likelihood build + timed value/grad in a clean process ───────────

def _run_worker(args) -> int:
    # Measure peak with the BFC allocator: memory_stats() exposes
    # ``peak_bytes_in_use`` only under BFC, NOT under the 'platform' (cudaMalloc)
    # allocator that ``configure_jax_runtime`` would otherwise set.  Preallocation
    # stays off so bytes reflect real on-demand growth.  These must be set before
    # any JAX import; ``configure_jax_runtime``'s setdefault() then leaves them.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "default")
    from darksirens.core.jax_config import configure_jax_runtime
    configure_jax_runtime()

    import contextlib
    import numpy as np
    import jax
    import jax.numpy as jnp

    # Build the run exactly as the CLI does, but stop before sampling.  The
    # verbose section/table prints go to stderr so stdout carries only the
    # BENCHMARK_RESULT line.
    from darksirens.cli.inference import (
        build_parser,
        _normalize_multitracer_paths,
        _resolve_catalog_sky_weighting,
        _validate_multitracer_config,
        _canonicalize_fixed_flags,
        _configure_performance_grids,
        _parse_structured_options,
        _resolve_sampler_config,
        _apply_bright_siren_overrides,
        _validate_run_config,
        _load_and_report_data,
        _resolve_single_catalog_marks,
        _build_and_report_parameter_space,
        _build_likelihood,
    )

    argv = [
        "--gw_path", args.gw_path,
        "--gwselection_path", args.gwselection_path,
        "--universe_model", "spectral_sirens",
        "--pop_model", "powerlaw+peak",
        "--fix_cosmology", "true",
        "--fix_survey", "true",
        "--sampler", "numpyro",              # grad path = the memory-limiting case
        "--redshift_prior_barrier", "off",   # vmappable (materialize=False)
        "--sel_batch_size", args.sel,
        "--pe_event_block", args.pe,
    ]

    with contextlib.redirect_stdout(sys.stderr):
        opts = build_parser().parse_args(argv)
        _normalize_multitracer_paths(opts)
        _resolve_catalog_sky_weighting(opts)
        _validate_multitracer_config(opts)
        _canonicalize_fixed_flags(opts)
        _configure_performance_grids(opts)
        prior_overrides, fixed_parameter_values = _parse_structured_options(opts)
        _resolve_sampler_config(opts)
        _apply_bright_siren_overrides(opts)
        _validate_run_config(opts)
        data = _load_and_report_data(opts)   # explicit sel/pe pass through the resolver
        _resolve_single_catalog_marks(opts, data)
        pspace = _build_and_report_parameter_space(
            opts, data, prior_overrides, fixed_parameter_values)
        likelihood = _build_likelihood(opts, data, pspace, fixed_parameter_values)

    ndim = len(pspace.labels)
    coord = jnp.asarray(pspace.prior_transform(np.full(ndim, 0.5)))

    def _median(fn, arg, repeats):
        fn(arg).block_until_ready()
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(arg).block_until_ready()
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    def _compile(fn, arg):
        t0 = time.perf_counter()
        fn(arg).block_until_ready()
        return time.perf_counter() - t0

    dev = jax.devices()[0]
    grad_fn = jax.grad(lambda c: likelihood(c).sum() if getattr(likelihood(c), "ndim", 0)
                       else likelihood(c))

    v_compile = _compile(likelihood, coord)
    peak_value = int((dev.memory_stats() or {}).get("peak_bytes_in_use", 0))
    v_warm = _median(likelihood, coord, args.repeats)

    g_compile = _compile(grad_fn, coord)
    g_warm = _median(grad_fn, coord, args.repeats)
    stats = dev.memory_stats() or {}
    peak = int(stats.get("peak_bytes_in_use", 0))          # = grad peak (conservative)
    in_use = int(stats.get("bytes_in_use", 0))

    result = {
        "sel_batch_size": None if args.sel == "off" else int(args.sel),
        "pe_event_block": None if args.pe == "off" else int(args.pe),
        "peak_bytes": peak,
        "peak_value_bytes": peak_value,
        "bytes_in_use_end": in_use,
        "value_compile_s": round(v_compile, 4),
        "value_warm_ms": round(1e3 * v_warm, 4),
        "grad_compile_s": round(g_compile, 4),
        "grad_warm_ms": round(1e3 * g_warm, 4),
        "ndim": ndim,
        "n_sel": int(np.asarray(data["m1detsels"]).shape[0]),
        "n_events": int(data["nEvents"]),
        "n_samp": int(data["nsamp"]),
        "backend": jax.default_backend(),
        "device": str(dev),
    }
    print("BENCHMARK_RESULT: " + json.dumps(result), flush=True)
    return 0


# ── orchestrator: one subprocess per config, classify, tabulate ──────────────────

def _run_config(sel, pe, args):
    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--sel", sel, "--pe", pe, "--repeats", str(args.repeats),
        "--gw-path", args.gw_path, "--gwselection-path", args.gwselection_path,
    ]
    label = f"sel={sel:>8} pe={pe:>4}"
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.per_config_timeout)
    except subprocess.TimeoutExpired:
        print(f"  {label}  -> TIMEOUT ({args.per_config_timeout}s)", flush=True)
        return {"sel_batch_size": None if sel == "off" else int(sel),
                "pe_event_block": None if pe == "off" else int(pe),
                "status": "timeout"}
    wall = time.perf_counter() - t0
    for line in proc.stdout.splitlines():
        if line.startswith("BENCHMARK_RESULT: "):
            res = json.loads(line[len("BENCHMARK_RESULT: "):])
            res["status"] = "ok"
            res["config_wall_s"] = round(wall, 1)
            print(f"  {label}  -> {res['peak_bytes']/1e9:6.3f} GB  "
                  f"(grad {res['grad_warm_ms']:.1f} ms, {wall:.0f}s)", flush=True)
            return res
    status = "oom" if "RESOURCE_EXHAUSTED" in proc.stderr else "error"
    tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
    print(f"  {label}  -> {status.upper()}  (rc={proc.returncode})\n    {tail}", flush=True)
    return {"sel_batch_size": None if sel == "off" else int(sel),
            "pe_event_block": None if pe == "off" else int(pe),
            "status": status, "stderr_tail": tail}


def _markdown(results):
    lines = ["| sel_batch_size | pe_event_block | status | peak GB | grad ms | value ms |",
             "|---|---|---|---|---|---|"]
    for r in results:
        peak = f"{r['peak_bytes']/1e9:.3f}" if r.get("peak_bytes") else "—"
        gm = f"{r.get('grad_warm_ms'):.1f}" if r.get("grad_warm_ms") else "—"
        vm = f"{r.get('value_warm_ms'):.1f}" if r.get("value_warm_ms") else "—"
        lines.append(f"| {r.get('sel_batch_size')} | {r.get('pe_event_block')} | "
                     f"{r['status']} | {peak} | {gm} | {vm} |")
    return "\n".join(lines)


def _orchestrate(args):
    sel_grid = _SMOKE_SEL if args.smoke else _SEL_GRID
    pe_grid = _SMOKE_PE if args.smoke else _PE_GRID
    configs = [(s, "off") for s in sel_grid] + [("off", p) for p in pe_grid if p != "off"]
    # dedup (off,off) if present twice
    seen, uniq = set(), []
    for c in configs:
        if c not in seen:
            seen.add(c); uniq.append(c)

    print(f"# block-size benchmark: {len(uniq)} configs, repeats={args.repeats}, "
          f"{'SMOKE' if args.smoke else 'FULL'}", flush=True)
    results = [_run_config(s, p, args) for s, p in uniq]

    meta = {"repeats": args.repeats, "smoke": args.smoke,
            "gw_path": args.gw_path, "gwselection_path": args.gwselection_path,
            "results": results}
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        meta["device"] = ok[0].get("device")
        meta["backend"] = ok[0].get("backend")
        meta["n_sel"] = ok[0].get("n_sel")
        meta["n_events"] = ok[0].get("n_events")
        meta["n_samp"] = ok[0].get("n_samp")

    print("\n" + _markdown(results), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nwrote {args.out}", flush=True)
    n_ok = len(ok)
    n_oom = sum(1 for r in results if r["status"] == "oom")
    print(f"\nsummary: {n_ok} ok, {n_oom} oom, {len(results) - n_ok - n_oom} other",
          flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--sel", default="off", help="worker: sel_batch_size value")
    ap.add_argument("--pe", default="off", help="worker: pe_event_block value")
    ap.add_argument("--gw-path", default=_DEFAULT_GW)
    ap.add_argument("--gwselection-path", default=_DEFAULT_SEL)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid to validate the harness")
    ap.add_argument("--per-config-timeout", type=int, default=900)
    ap.add_argument("--out", default=None, help="write JSON results here")
    args = ap.parse_args()
    return _run_worker(args) if args.worker else _orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
