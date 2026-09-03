#!/usr/bin/env python
"""Time one likelihood call of ``darksirens_inference``, built EXACTLY as the CLI
builds it, and attribute the cost to its components.

Why a separate script
---------------------
``scripts/benchmark_block_sizes.py`` measures PEAK MEMORY per block-size plan;
this one measures WALL TIME per call and records the log-likelihood values at a
fixed set of prior draws, so two builds (a base commit and a candidate) can be
compared for speed AND for value drift with one command each:

    # base
    python scripts/benchmarks/bench_likelihood_call.py --tag base --out base.json -- \\
        --gw_path GW.h5 --gwselection_path SEL.h5 --survey_path CAT.h5 \\
        --universe_model dark_sirens --pop_model powerlaw+peak --sampler dynesty \\
        --fix_population true --fix_survey true --fix_de true \\
        --fixed_parameter_values '{"Om0": 0.3075}' --save_path /tmp/bench_run

    # candidate (same data, same CLI args): prints max |dlogL| and the speedup
    python scripts/benchmarks/bench_likelihood_call.py --tag cand --compare base.json -- ...

Everything after ``--`` is passed verbatim to the ``darksirens_inference``
parser, and the option resolution, data load, parameter space and likelihood
factory are the CLI's own functions (``darksirens.cli.inference``), so the
timed callable is the one the sampler would call.  ``--components`` additionally
times the per-proposal state build, the per-sample population/Jacobian kernel,
the redshift-prior evaluation and the catalog KDE on the PE and selection sets
in isolation (each as its own jit, with the prepared state passed as an
ARGUMENT so the consumer is timed without its producer).

Notes
-----
* Timings are wall-clock medians over ``--n-calls`` warm calls, after one
  compile call and three warm-up calls.  Report the device: CPU and GPU
  profiles differ qualitatively (the windowed catalog KDE is gather-bound on a
  GPU and exp-bound on a CPU).
* On a CPU box pass explicit ``--sel_batch_size`` / ``--pe_event_block`` (the
  auto policy is a single pass on non-GPU backends, which can exceed host RAM
  on a dense catalog) and ``--row_chunk 256`` for the build-time quadrature.
* The value comparison is only meaningful for the SAME data and the SAME
  ``--seed``-derived coordinates; the script checks the coordinates match.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

# Run-as-script guard: ``python scripts/benchmarks/...`` puts this directory on
# sys.path[0]; make ``import darksirens`` resolve to THIS worktree.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


class _Built:
    """The CLI's own build products for one option set."""


def build_from_cli(cli_args, quiet=True):
    """Resolve options, load data, build the parameter space and the likelihood
    with the ``darksirens.cli.inference`` phase functions, in the CLI's order."""
    from darksirens.cli import inference as cli

    optp = cli.build_parser()
    opts = optp.parse_args(cli_args)
    sink = io.StringIO() if quiet else None
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    b = _Built()
    with ctx:
        cli._normalize_multitracer_paths(opts)
        cli._check_latent_field_mode(opts)
        cli._stamp_latent_artifact_fingerprint(opts)
        cli._resolve_catalog_sky_weighting(opts)
        cli._validate_multitracer_config(opts)
        cli._canonicalize_fixed_flags(opts)
        cli._configure_performance_grids(opts)
        prior_overrides, fixed_parameter_values = cli._parse_structured_options(opts)
        cli._resolve_sampler_config(opts)
        cli._apply_bright_siren_overrides(opts)
        cli._validate_run_config(opts)
        t0 = time.perf_counter()
        data = cli._load_and_report_data(opts)
        b.t_load = time.perf_counter() - t0
        cli._resolve_single_catalog_marks(opts, data)
        pspace = cli._build_and_report_parameter_space(
            opts, data, prior_overrides, fixed_parameter_values)
        t0 = time.perf_counter()
        likelihood = cli._build_likelihood(opts, data, pspace, fixed_parameter_values)
        b.t_build = time.perf_counter() - t0
    b.opts, b.data, b.pspace, b.likelihood = opts, data, pspace, likelihood
    b.prior_overrides, b.fixed_parameter_values = prior_overrides, fixed_parameter_values
    b.cli_log = sink.getvalue() if quiet else ""
    return b


def draw_coords(pspace, n, seed, mode="prior", opts=None, jitter=0.02):
    """``n`` coordinates in the run's sampled space (deterministic).

    ``mode="prior"``: draws through the run's own prior transform -- the
    sampler's own distribution of proposals, but with a wide population prior
    most draws land in the selection guard's -inf, which makes value
    comparisons vacuous.  ``mode="fiducial"``: the registered fiducial value of
    every sampled label (cosmology, survey, population, sky, marks), each
    multiplied by ``1 + jitter * N(0, 1)`` and clipped to the prior box, so
    the log-likelihood is finite and the timing is that of a live region.
    """
    rng = np.random.default_rng(seed)
    labels = list(pspace.labels)
    ndim = len(labels)
    if mode == "prior":
        return np.asarray([
            np.asarray(pspace.prior_transform(jnp.asarray(rng.random(ndim))))
            for _ in range(n)
        ])
    from darksirens.core.constants import (
        H0_FID, OM0_FID, SURVEY_PARAMS_FID_BY_NAME, W0_FID, WA_FID,
    )
    from darksirens.gw.populations import pop_model_prior_parser
    from darksirens.gw.populations.registry import get_fixed_population_params

    pop_labels = list(pop_model_prior_parser(opts.pop_model)[2])
    pop_fid = np.asarray(get_fixed_population_params(opts.pop_model), dtype=float)
    fid = {"H0": H0_FID, "Om0": OM0_FID, "w0": W0_FID, "wa": WA_FID}
    fid.update({k: float(v) for k, v in SURVEY_PARAMS_FID_BY_NAME.items()})
    fid.update(dict(zip(pop_labels, pop_fid)))
    lo = np.asarray(pspace.lower_bound, dtype=float)
    hi = np.asarray(pspace.upper_bound, dtype=float)
    base = np.empty(ndim)
    for i, lbl in enumerate(labels):
        key = lbl.split("_c")[0] if "_c" in lbl and lbl not in fid else lbl
        base[i] = fid.get(key, 0.5 * (lo[i] + hi[i]))
    out = []
    for _ in range(n):
        c = base * (1.0 + jitter * rng.standard_normal(ndim))
        out.append(np.clip(c, lo, hi))
    return np.asarray(out)


def timeit(fn, n_warm=2, n_iter=10):
    for _ in range(n_warm):
        jax.block_until_ready(fn())
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        ts.append(time.perf_counter() - t0)
    return np.asarray(ts)


def _fmt(ts):
    return f"{np.median(ts) * 1e3:9.1f} ms (min {np.min(ts) * 1e3:8.1f})"


def run_components(b, coord, n_iter):
    """Time the likelihood's components in isolation on the SAME operands."""
    from darksirens.inference.parameters import build_parameter_decoder
    from darksirens.redshift.prior import (
        prepare_redshift_prior_state, eval_redshift_prior_with_state,
    )
    from darksirens.redshift.catalog import (
        catalog_kernel_state, eval_log_catalog_prior_state,
    )
    from darksirens.redshift.completion import (
        completion_curves, log_galaxy_measure_grid, field_global_log_Z,
        bound_smoothing_operator,
    )
    from darksirens.gw.populations import pop_model_parser
    from darksirens.inference.utils import log_target_density_base_and_z
    from darksirens.utils import cosmology
    from darksirens.utils.cosmology import dL_of_z, z_of_dL_precomputed, zgrid as czg
    from jax import vmap

    opts, data, pspace, likelihood = b.opts, b.data, b.pspace, b.likelihood
    dec = build_parameter_decoder(
        opts, pspace.pop_params_fid, fixed_parameter_values=b.fixed_parameter_values,
        wl_params=data.get("wl_params"))
    gw_pe, em_pe, gw_sel, em_sel = likelihood.operands[:4]
    csw = getattr(opts, "catalog_sky_weighting", "conditional")
    universe = opts.universe_model
    dt, smo = likelihood.distance_table, likelihood.smoothing_operator
    log_p_pop = pop_model_parser(pop_model=opts.pop_model)

    def wrap(fn):
        """jit ``fn`` with the distance / smoothing tables bound as the factory does."""
        def body(*args):
            with cosmology.bound_distance_table(args[-2]), bound_smoothing_operator(args[-1]):
                return fn(*args[:-2])
        j = jax.jit(body)
        return lambda *args: j(*args, dt, smo)

    def _decode(c):
        return dec.decode(c)

    print(f"  PE samples={int(gw_pe.dL.shape[0]):,}  sel injections={int(gw_sel.dL.shape[0]):,}"
          f"  rows={None if em_pe.zgals is None else tuple(em_pe.zgals.shape)}"
          f"  kernel pin={'on' if getattr(em_pe, 'pinned_kernels', None) is not None else 'off'}")

    prep = wrap(lambda c, em: prepare_redshift_prior_state(
        universe, *_decode(c)[:2], em, catalog_sky_weighting=csw))
    print(f"  {'prepare_redshift_prior_state (as built)':44s} "
          f"{_fmt(timeit(lambda: prep(coord, em_pe), 1, n_iter))}")
    state = prep(coord, em_pe)
    if getattr(em_pe, "pinned_kernels", None) is not None:
        em_unp = em_pe._replace(pinned_kernels=None, field_depth_total_pinned=None)
        print(f"  {'prepare_redshift_prior_state (pin OFF)':44s} "
              f"{_fmt(timeit(lambda: prep(coord, em_unp), 1, max(2, n_iter // 2)))}")
    if universe == "dark_sirens":
        lgg = wrap(lambda c: log_galaxy_measure_grid(*_decode(c)[:2]))
        print(f"  {'  log_galaxy_measure_grid':44s} {_fmt(timeit(lambda: lgg(coord), 1, n_iter))}")
        cc = wrap(lambda c, em: completion_curves(*_decode(c)[:2], em))
        print(f"  {'  completion_curves':44s} {_fmt(timeit(lambda: cc(coord, em_pe), 1, n_iter))}")
        ks = wrap(lambda c, em: catalog_kernel_state(
            *_decode(c)[:2], em, z_depth=_decode(c)[1].z_depth,
            pinned=getattr(em, 'pinned_kernels', None)))
        print(f"  {'  catalog_kernel_state (as built)':44s} {_fmt(timeit(lambda: ks(coord, em_pe), 1, n_iter))}")
        if csw == "field":
            fz = wrap(lambda c, em: field_global_log_Z(*_decode(c)[:2], em))
            print(f"  {'  field_global_log_Z':44s} {_fmt(timeit(lambda: fz(coord, em_pe), 1, n_iter))}")

    def _base(c, gw, em):
        cosmo, survey, pop_params, _, _ = _decode(c)
        dLg = dL_of_z(czg, cosmo.H0, cosmo.Om0, cosmo.w0, cosmo.wa)
        dL_c = jnp.clip(gw.dL, dLg[0], dLg[-1])
        return log_target_density_base_and_z(
            gw.m1det, gw.q, dL_c, gw.chieff, gw.pixels, gw.prior_wt,
            cosmo, survey, pop_params, em, log_p_pop, spin=gw.spin, dL_grid=dLg)
    base_j = wrap(_base)
    prior_j = wrap(lambda c, st, z, gw, em: eval_redshift_prior_with_state(
        universe, st, z, gw.pixels, *_decode(c)[:2], em, catalog_sky_weighting=csw))
    kde_j = wrap(lambda st, z, gw, em: vmap(
        lambda zi, pi: eval_log_catalog_prior_state(zi, pi, st.kernels, em))(z, gw.pixels))

    for tag, gw, em, st in (("PE", gw_pe, em_pe, state),
                            ("SEL", gw_sel, em_sel, None)):
        if st is None:
            st = prep(coord, em)
        print(f"  {tag + ': pop model + Jacobian + z_of_dL':44s} "
              f"{_fmt(timeit(lambda: base_j(coord, gw, em), 1, n_iter))}")
        _, z = base_j(coord, gw, em)
        print(f"  {tag + ': redshift prior eval (KDE + missing)':44s} "
              f"{_fmt(timeit(lambda: prior_j(coord, st, z, gw, em), 1, n_iter))}")
        if universe in ("dark_sirens", "dark_sirens_complete") and em.zgals is not None:
            print(f"  {tag + ':   catalog KDE only':44s} "
                  f"{_fmt(timeit(lambda: kde_j(st, z, gw, em), 1, n_iter))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="label for this build in the report / json")
    ap.add_argument("--n-calls", type=int, default=20)
    ap.add_argument("--n-coords", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--coords", choices=["prior", "fiducial"], default="prior",
                    help="where to evaluate: prior draws, or the fiducial point jittered by 2%%")
    ap.add_argument("--out", default=None, help="write the summary json here")
    ap.add_argument("--compare", default=None, help="a previous --out json to compare values/speed against")
    ap.add_argument("--components", action="store_true", help="also time the components in isolation")
    ap.add_argument("--verbose", action="store_true", help="show the CLI's own build output")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- <darksirens_inference args>")
    a = ap.parse_args()
    cli_args = [x for x in a.rest if x != "--"]

    b = build_from_cli(cli_args, quiet=not a.verbose)
    opts, data, pspace, likelihood = b.opts, b.data, b.pspace, b.likelihood
    coords = draw_coords(pspace, a.n_coords, a.seed, mode=a.coords, opts=opts)

    t0 = time.perf_counter()
    jax.block_until_ready(likelihood(jnp.asarray(coords[0])))
    t_first = time.perf_counter() - t0
    for c in coords[:3]:
        jax.block_until_ready(likelihood(jnp.asarray(c)))

    times, values = [], {}
    for i in range(a.n_calls):
        k = i % len(coords)
        t0 = time.perf_counter()
        v = jax.block_until_ready(likelihood(jnp.asarray(coords[k])))
        times.append(time.perf_counter() - t0)
        values.setdefault(k, float(v))
    times = np.asarray(times)

    zg = data.get("zgals")
    summary = {
        "tag": a.tag, "cli_args": cli_args, "labels": list(pspace.labels),
        "backend": jax.default_backend(),
        "nEvents": int(data["nEvents"]), "nsamp": int(data["nsamp"]),
        "n_sel": int(np.asarray(data["dLsels"]).shape[0]),
        "n_rows": None if zg is None else int(np.asarray(zg).shape[0]),
        "n_max": None if zg is None else int(np.asarray(zg).shape[1]),
        "sel_batch_size": opts.sel_batch_size, "pe_event_block": opts.pe_event_block,
        "kde_window": getattr(likelihood, "kde_window", None),
        "frozen_redshift_prior": bool(getattr(likelihood, "frozen_redshift_prior", False)),
        "t_load_s": b.t_load, "t_build_s": b.t_build, "t_first_call_s": t_first,
        "t_call_median_s": float(np.median(times)), "t_call_min_s": float(np.min(times)),
        "t_call_mean_s": float(np.mean(times)), "t_call_std_s": float(np.std(times)),
        "values": [values[k] for k in sorted(values)],
        "coords": coords.tolist(),
    }
    print("=" * 78)
    print(f"[{a.tag}] backend={summary['backend']} labels={summary['labels']}")
    print(f"  nEvents={summary['nEvents']} nsamp={summary['nsamp']} n_sel={summary['n_sel']:,} "
          f"rows={summary['n_rows']} n_max={summary['n_max']} "
          f"sel_batch={summary['sel_batch_size']} pe_block={summary['pe_event_block']} "
          f"kde_window={summary['kde_window']} frozen_prior={summary['frozen_redshift_prior']}")
    print(f"  load {b.t_load:.1f}s  build {b.t_build:.1f}s  first call (compile+run) {t_first:.1f}s")
    print(f"  call: median {np.median(times) * 1e3:.1f} ms  min {np.min(times) * 1e3:.1f} ms  "
          f"mean {np.mean(times) * 1e3:.1f} +- {np.std(times) * 1e3:.1f} ms  (n={a.n_calls})")
    print(f"  values: {summary['values']}")
    if a.compare:
        base = json.load(open(a.compare))
        if np.array_equal(np.asarray(base["coords"]), coords):
            bv, nv = np.asarray(base["values"]), np.asarray(summary["values"])
            d = nv - bv
            rel = np.abs(d) / np.maximum(np.abs(bv), 1.0)
            print(f"  vs [{base['tag']}]: max|dlogL| = {np.max(np.abs(d)):.3e}  max rel = {np.max(rel):.3e}  "
                  f"bit-identical = {bool(np.array_equal(bv, nv))}  "
                  f"speedup = {base['t_call_median_s'] / summary['t_call_median_s']:.2f}x")
            summary["compare"] = {"base_tag": base["tag"], "max_abs_dlogL": float(np.max(np.abs(d))),
                                  "max_rel_dlogL": float(np.max(rel)),
                                  "speedup": float(base["t_call_median_s"] / summary["t_call_median_s"])}
        else:
            print("  vs baseline: coordinates differ (different labels/seed) -- no value comparison")
    if a.components:
        print("  components (isolated jits on the built operands):")
        run_components(b, jnp.asarray(coords[0]), max(3, a.n_calls // 4))
    print("=" * 78)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
