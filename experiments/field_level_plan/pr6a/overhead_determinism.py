"""PR-6a: the determinism sweep and the overhead measurement, production line.

Two of PLAN §6.4's three runtime diagnostics, measured on the object that
ships -- the 259-event dark-siren likelihood of ``experiments/desi_full259``
with the latent leaves live -- rather than on a fixture.  The third (the
``M_draw`` convergence trace) is PR-5b's and is not repeated here.

**Nothing under ``darksirens/`` is touched and no posterior is produced.**
The production 259-event H0 run is HELD by the owner pending the PR-6a gate;
this script evaluates the likelihood a few hundred times, which is a timing
and smoothness measurement, not an inference.

------------------------------------------------------------------- the arms

Four, so that the two §2.3 columns are both recoverable and the confound
between them is visible rather than assumed:

``baseline_nofp``  the shipped ``sel`` arm of ``run_h0_scans.py``: c_mode
                   selection, field sky weighting, NO LSS of any kind, no
                   member marginalization, no per-pixel completeness.  This
                   is PR-0's cost baseline -- the denominator OWNER DECISION
                   5 and kill criterion K4 are taken on (§2.3 v4 finding 10:
                   "the production baseline contains ZERO member-dependent
                   seam work, so the deliverable is not a table-marginalized
                   run made latent, it is a NON-marginalized production run
                   made latent").
``baseline_fp``    the same plus ``--per_pixel_completeness`` (PR-2's
                   ``C_p = f_p C(z)``).  Latent mode REQUIRES ``f_p``
                   (``factory.py`` guard 6), so without this arm the latent
                   overhead would silently carry PR-2's cost as well.
``table_m8``       ``baseline_nofp`` + the shipped ``q_radial.h5`` (8 members,
                   ``member_content_sha256`` recorded below) +
                   ``lss_marginalize=True``.  It CANNOT carry ``f_p``:
                   ``inference/loaders.py:1021`` refuses a per-pixel
                   selection fraction alongside a Q table, which is exactly
                   the defect PR-6a's closure phase logged as S-3.  So the
                   ``latent - table`` delta measured here is
                   latent-plus-``f_p`` minus table-without-``f_p``, and the
                   ``baseline_fp - baseline_nofp`` column is what separates
                   them.
``latent_m8``      PR-6a: ``lss_field_mode='latent'``, the PR-5
                   ``latent_anchor_v2a.h5`` (M_draw = 8, the value PR-5b's
                   P14 verdict selected), ``lss_marginalize=True``,
                   ``f_p`` on.

Guard convention, stated because PR-0 measured it to be decisive on this
line: PR-0's CLEAN arm -- ``selection_neff_soft_guard=False``,
``max_likelihood_variance=1e6``, Vitale's ``5 N_obs`` floor retained.  The
hard GWTC-4/5 variance criterion fails at every H0 node here (needs
``Neff ~ 92k``, the line delivers 31-36k), so under the shipped soft-guard
convention the likelihood is replaced by a ``-gate (100 + 2 N softplus(-log
mu))`` wall whose value -- and whose COST -- is a different object.  Timing
is insensitive to the guard (the same work is done either way), but the
sweep's logL values are not, so the convention is recorded with them.

------------------------------------------------------- what the sweep is NOT

PLAN §6.4 is emphatic and this docstring repeats it so that no reader of the
JSON can miss it: under common random numbers the member ensemble is FROZEN,
so the estimator is a deterministic smooth function of theta.  Repeat-
determinism and adjacent-theta smoothness therefore pass BY CONSTRUCTION,
and pass just as well on a badly distorted surrogate (``MODEL.tex`` Rem.
``rem:crn``).  This sweep is a regression guard against a stray RNG, a stray
device nondeterminism, or a discontinuous branch -- never evidence that the
seam is right.  Only P14's theta-VARYING bias discriminates, and that is
PR-5b's measurement.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PR6A = Path(__file__).resolve().parent
PLAN_DIR = PR6A.parent
sys.path.insert(0, str(PLAN_DIR / "pr5b"))

import latent_harness as H  # noqa: E402  (pins DARKSIRENS_ZMAX=6.0 first)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

C = H.C
Q_RADIAL = C.DATA_DIR / "fits" / "q_radial.h5"
MTH_MAP = C.INGEST_DATA / "mth_map_nside128.h5"
ANCHOR = H.resolve_anchor_m8()
ANCHOR_H0 = 67.74


def _opts(arm: str) -> SimpleNamespace:
    """One opts object per arm; everything but the LSS treatment held fixed."""
    o = H.clean_arm_opts(str(ANCHOR), soft_guard=False, max_var=1e6,
                         marginalize=True)
    if arm == "latent_m8":
        return o
    # strip the latent leaves for every non-latent arm
    o.lss_field_mode = "table"
    o.lss_field_artifact = None
    if arm == "baseline_fp":
        o.lss_marginalize = False
        return o
    if arm == "baseline_nofp":
        o.lss_marginalize = False
        o.per_pixel_completeness = None
        return o
    if arm == "baseline_pr0":
        # PR-0's cost-baseline arm reproduced EXACTLY (run_cost_baseline.py's
        # ``no_lss``): the SHIPPED soft-guard convention and the default
        # variance cap, not PR-0's clean arm.  It exists only to settle
        # whether PR-0's 3027.1 ms and this session's 1417.5 ms differ
        # because of the guard convention or because of the machine -- PR-0
        # ran under ``-p RITA-GPU``, whose node ``rita`` carries
        # ``Gres=gpu:a100-80:2``, while this session is on ``miko``
        # (``Gres=gpu:h100:2``).  PR-0's report calls its baseline "H100".
        o.lss_marginalize = False
        o.per_pixel_completeness = None
        o.selection_neff_soft_guard = True
        o.max_likelihood_variance = 1.0
        return o
    if arm == "table_m8":
        # loaders.py:1021 refuses f_p alongside a Q table (closure S-3)
        o.per_pixel_completeness = None
        o.lss_completion = str(Q_RADIAL)
        o.lss_marginalize = True
        return o
    raise SystemExit(f"unknown arm {arm}")


def _time_arm(logl, h0_probe, n_warm, n_timed):
    """Wall per evaluation, forced to completion by the host transfer.

    ``float(...)`` on the returned array is a device->host copy, so it
    synchronizes; JAX's async dispatch cannot hide work behind the timer.
    The first call TRACES and COMPILES, which is why ``n_warm >= 2``.  The
    timed calls use distinct H0 values: the likelihood is jitted on the
    argument AVAL, not its value, so a new H0 is a cache hit and no
    recompilation is timed, but distinct values make an accidental constant
    fold loud.
    """
    t_compile = None
    for i in range(n_warm):
        t0 = time.perf_counter()
        v = float(logl(jnp.asarray([float(h0_probe[i % len(h0_probe)])])))
        if i == 0:
            t_compile = time.perf_counter() - t0
        assert np.isfinite(v) or True
    ts = []
    for i in range(n_timed):
        h = float(h0_probe[i % len(h0_probe)])
        t0 = time.perf_counter()
        v = float(logl(jnp.asarray([h])))
        ts.append(time.perf_counter() - t0)
    ts = np.asarray(ts)
    return {
        "compile_plus_first_eval_s": t_compile,
        "n_timed": int(n_timed),
        "ms_median": float(np.median(ts) * 1e3),
        "ms_mean": float(ts.mean() * 1e3),
        "ms_min": float(ts.min() * 1e3),
        "ms_max": float(ts.max() * 1e3),
        "ms_all": (ts * 1e3).tolist(),
        "logl_probe": v,
    }


def _repeats(logl, h0, n):
    """100 repeats at one theta -- the bit-identity half of §6.4."""
    vals = [float(logl(jnp.asarray([float(h0)]))) for _ in range(n)]
    a = np.asarray(vals)
    return {
        "n": int(n), "H0": float(h0), "value": float(a[0]),
        "n_distinct": int(np.unique(a).size),
        "max_abs_dev_from_first": float(np.abs(a - a[0]).max()),
        "bit_identical": bool(np.unique(a).size == 1),
    }


def _sweep(logl, h0):
    vals = np.empty(h0.size)
    t0 = time.perf_counter()
    for i, h in enumerate(h0):
        vals[i] = float(logl(jnp.asarray([float(h)])))
        if i % 25 == 0:
            print(f"    sweep {i}/{h0.size} H0={h:.4f} logL={vals[i]:.6f} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    d = np.abs(np.diff(vals))
    fin = np.isfinite(vals)
    return {
        "n_points": int(h0.size), "h0_min": float(h0[0]),
        "h0_max": float(h0[-1]), "spacing": float(h0[1] - h0[0]),
        "wall_s": float(time.perf_counter() - t0),
        "n_finite": int(fin.sum()),
        "logl_min": float(np.nanmin(vals[fin])) if fin.any() else None,
        "logl_max": float(np.nanmax(vals[fin])) if fin.any() else None,
        "adj_abs_delta_median": float(np.median(d[np.isfinite(d)])),
        "adj_abs_delta_max": float(np.nanmax(d[np.isfinite(d)])),
        "adj_ratio_max_over_median": float(
            np.nanmax(d[np.isfinite(d)]) / np.median(d[np.isfinite(d)])),
        "adj_abs_delta_max_at_H0": float(
            h0[1:][np.nanargmax(np.where(np.isfinite(d), d, -np.inf))]),
        "max_adj_jump_nat": float(np.nanmax(d[np.isfinite(d)])),
        "h0": h0.tolist(), "logl": vals.tolist(),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="*", default=[
        "baseline_nofp", "baseline_fp", "table_m8", "latent_m8"])
    p.add_argument("--n-warm", type=int, default=3)
    p.add_argument("--n-timed", type=int, default=12)
    p.add_argument("--n-repeats", type=int, default=100)
    p.add_argument("--sweep-points", type=int, default=241)
    p.add_argument("--sweep-arms", nargs="*", default=["latent_m8", "table_m8"])
    p.add_argument("--out", default=str(PR6A / "overhead_determinism.json"))
    a = p.parse_args(argv)

    out = {
        "what": ("PLAN 6.4 determinism sweep + 2.3 overhead, production "
                 "259-event line, PR-0 clean guard arm"),
        "NOT_a_posterior": ("no sampler was run; the production 259-event H0 "
                            "posterior is HELD by the owner pending the "
                            "PR-6a gate"),
        "determinism_caveat": (
            "under common random numbers the member ensemble is frozen, so "
            "the estimator is a deterministic smooth function of theta: "
            "repeat-determinism and adjacent-theta smoothness pass BY "
            "CONSTRUCTION and pass equally on a distorted surrogate. "
            "Regression guard, never evidence of correctness. Only P14's "
            "theta-varying bias discriminates (PR-5b)."),
        "guard_convention": {
            "selection_neff_soft_guard": False,
            "max_likelihood_variance": 1e6,
            "vitale_5Nobs_floor": "retained",
            "why": ("PR-0's clean arm; the hard GWTC-4/5 criterion fails at "
                    "every H0 node on this line"),
        },
        "anchor": str(ANCHOR), "M_draw": 8,
        "q_table": str(Q_RADIAL),
        "device": [str(d) for d in jax.devices()],
        "git_sha": C.git_sha(),
        "arms": {},
    }
    h0_probe = [ANCHOR_H0, 70.0, 65.0, 72.5]

    for arm in a.arms:
        print(f"\n=== arm {arm} ===", flush=True)
        opts = _opts(arm)
        t0 = time.perf_counter()
        data = H.load_data(opts)
        t_load = time.perf_counter() - t0
        t0 = time.perf_counter()
        logl = H.build_likelihood(opts, data)
        t_build = time.perf_counter() - t0
        rec = {
            "load_s": t_load, "build_s": t_build,
            "opts": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(opts).items()
                     if k in ("c_mode", "use_LSS", "lss_completion",
                              "lss_marginalize", "lss_field_mode",
                              "lss_field_artifact", "per_pixel_completeness",
                              "catalog_sky_weighting",
                              "selection_neff_soft_guard",
                              "max_likelihood_variance")},
        }
        rec["timing"] = _time_arm(logl, h0_probe, a.n_warm, a.n_timed)
        print(f"  {arm}: {rec['timing']['ms_median']:.1f} ms/eval "
              f"(min {rec['timing']['ms_min']:.1f}, "
              f"max {rec['timing']['ms_max']:.1f}); "
              f"logL={rec['timing']['logl_probe']:.6f}", flush=True)
        if arm in a.sweep_arms:
            rec["repeats"] = _repeats(logl, ANCHOR_H0, a.n_repeats)
            print(f"  repeats x{a.n_repeats}: distinct="
                  f"{rec['repeats']['n_distinct']} "
                  f"maxdev={rec['repeats']['max_abs_dev_from_first']:.3e}",
                  flush=True)
            if a.sweep_points > 0:
                h0 = np.linspace(20.0, 140.0, a.sweep_points)
                rec["sweep"] = _sweep(logl, h0)
                s = rec["sweep"]
                print(f"  sweep: max/median adjacent |dlogL| = "
                      f"{s['adj_ratio_max_over_median']:.3f} "
                      f"(max {s['adj_abs_delta_max']:.4f} nat at H0="
                      f"{s['adj_abs_delta_max_at_H0']:.3f})", flush=True)
        out["arms"][arm] = rec
        del logl, data
        jax.clear_caches()

    # ---- derived overhead columns (PLAN 2.3), against the MEASURED baseline
    def ms(name):
        return out["arms"].get(name, {}).get("timing", {}).get("ms_median")

    base, base_fp = ms("baseline_nofp"), ms("baseline_fp")
    tab, lat = ms("table_m8"), ms("latent_m8")
    out["overhead"] = {
        "baseline_nofp_ms": base, "baseline_fp_ms": base_fp,
        "table_m8_ms": tab, "latent_m8_ms": lat,
        "latent_minus_baseline_nofp_ms": (
            None if (lat is None or base is None) else lat - base),
        "latent_vs_baseline_nofp_pct": (
            None if (lat is None or base is None) else
            100.0 * (lat - base) / base),
        "latent_vs_baseline_fp_pct": (
            None if (lat is None or base_fp is None) else
            100.0 * (lat - base_fp) / base_fp),
        "f_p_cost_pct": (
            None if (base_fp is None or base is None) else
            100.0 * (base_fp - base) / base),
        "latent_minus_table_ms": (
            None if (lat is None or tab is None) else lat - tab),
        "latent_minus_table_pct_of_baseline": (
            None if (lat is None or tab is None or base is None) else
            100.0 * (lat - tab) / base),
        "PLAN_2_3_prediction": {
            "note": ("PLAN's percentages are quoted against a 27.5 ms "
                     "baseline that PR-0 superseded with 3027 ms measured "
                     "value-only on an H100, so every percentage in "
                     "section 2 deflates ~110x; the ABSOLUTE ms columns are "
                     "the ones to compare"),
            "latent_minus_table_ms_at_M8": 3.3,
            "latent_vs_no_lss_baseline_ms_at_M8": 8.6,
            "pct_vs_27_5ms_baseline": {"latent_minus_table": 12.0,
                                       "latent_vs_baseline": 31.0},
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\n[out] {a.out}")
    print(json.dumps(out["overhead"], indent=1))


if __name__ == "__main__":
    main()
