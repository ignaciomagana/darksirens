"""PLAN §6.4's determinism sweep at its LITERAL size: 1e4 points, both modes.

The production line costs ~3.2 s per likelihood evaluation (measured, this
rung, ``overhead_determinism.json``), so a 1e4-point sweep there is ~9 GPU-
hours and was not run; ``overhead_determinism.py`` sweeps that object at 241
points instead.  This script runs the gate at its stated size on the smallest
object that is still the WHOLE latent stack: the end-to-end fixture of
``tests/test_latent_seam_e2e.py`` -- real ``darksiren_log_likelihood``, real
member vmap, real ``eval_dark_member_completion_latent``, real
``latent_q`` seam with the footprint/off-footprint split (rows 0..3 are the
fitted footprint, rows 4..5 are outside it, so pin P13b's branch is live), 3
events x 48 PE samples, 240 injections, ``M_draw = 3``.

Two arms, because the gate has to be able to fail:

``latent``   ``lss_field_mode='latent'`` with the fixture's latent leaves.
``table``    the same likelihood fed the table the seam generates, which the
             e2e module pins equal to the latent stack to 1e-8.  It is the
             control: whatever the sweep reports for latent it must also
             report here, or the latent branch has introduced a
             discontinuity that the table branch does not have.

WHAT THIS DOES NOT SHOW.  PLAN §6.4, verbatim in intent: under common random
numbers the member draws are FROZEN, so the estimator is a deterministic
smooth function of theta.  Repeat-determinism and adjacent-theta smoothness
therefore pass BY CONSTRUCTION -- and pass just as well on a badly distorted
surrogate (``MODEL.tex`` Rem. ``rem:crn``).  A reader who takes this file as
validation of the seam has been misled.  It is a regression guard: it fires
on a stray RNG re-draw per evaluation, on a non-deterministic reduction, and
on a discontinuous branch (a clip, a ``where`` on a theta-dependent
predicate, a table lookup that changes cell).  Only P14's theta-VARYING bias
(PR-5b) discriminates a correct estimator from a distorted one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DARKSIRENS_ZMAX", "1.0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

PR6A = Path(__file__).resolve().parent
REPO = PR6A.parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import numpy as np  # noqa: E402

import test_latent_seam_e2e as E  # noqa: E402
from darksirens.core.types import CosmoParams  # noqa: E402
from darksirens.likelihood.core import darksiren_log_likelihood  # noqa: E402


def _ll_at(h0, cat, survey, field_mode):
    """``tests/test_latent_seam_e2e._ll`` with H0 promoted to an argument."""
    return float(darksiren_log_likelihood(
        CosmoParams(H0=float(h0), Om0=E.COSMO.Om0), survey, E.POP,
        E._GW_PE, cat, E._GW_SEL, cat, E._N_EV, E._N_SAMP, float(E._N_SEL),
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None, lss_marginalize=True,
        lss_field_mode=field_mode,
    ))


def _arm(name):
    if name == "latent":
        return E._catalog(latent=True), E.SURVEY_LATENT, "latent"
    if name == "table":
        return (E._catalog(logq_members=E._generated_logq_members()),
                E.SURVEY_TABLE, "table")
    raise SystemExit(name)


def run(name, n_points, n_repeats, h0_lo, h0_hi):
    cat, survey, mode = _arm(name)
    h0 = np.linspace(h0_lo, h0_hi, n_points)

    # --- 100 repeats at one theta: bit-identity
    ref = float(np.median(h0))
    vals = np.asarray([_ll_at(ref, cat, survey, mode) for _ in range(n_repeats)])
    rep = {"n": int(n_repeats), "H0": ref, "value": float(vals[0]),
           "n_distinct": int(np.unique(vals).size),
           "max_abs_dev": float(np.abs(vals - vals[0]).max()),
           "bit_identical": bool(np.unique(vals).size == 1)}

    # --- the 1e4-point sweep
    t0 = time.perf_counter()
    out = np.empty(n_points)
    for i, h in enumerate(h0):
        out[i] = _ll_at(h, cat, survey, mode)
        if i % 2000 == 0:
            print(f"  [{name}] {i}/{n_points} H0={h:.4f} logL={out[i]:.9f} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    wall = time.perf_counter() - t0
    d = np.abs(np.diff(out))
    med = float(np.median(d[np.isfinite(d)]))
    res = {
        "arm": name, "n_points": int(n_points),
        "h0_range": [h0_lo, h0_hi], "spacing": float(h0[1] - h0[0]),
        "wall_s": wall, "ms_per_eval": 1e3 * wall / n_points,
        "n_finite": int(np.isfinite(out).sum()),
        "logl_first": float(out[0]), "logl_last": float(out[-1]),
        "logl_min": float(out.min()), "logl_max": float(out.max()),
        "adj_abs_delta_median": med,
        "adj_abs_delta_max": float(np.nanmax(d[np.isfinite(d)])),
        "adj_ratio_max_over_median": float(
            np.nanmax(d[np.isfinite(d)]) / med),
        "adj_abs_delta_max_at_H0": float(
            h0[1:][int(np.nanargmax(np.where(np.isfinite(d), d, -np.inf)))]),
        "n_zero_adjacent_deltas": int((d == 0).sum()),
        "monotone_deltas_sign_changes": int(
            np.sum(np.diff(np.sign(np.diff(out))) != 0)),
        "repeats": rep,
    }
    return res, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-points", type=int, default=10000)
    p.add_argument("--n-repeats", type=int, default=100)
    p.add_argument("--h0-lo", type=float, default=20.0)
    p.add_argument("--h0-hi", type=float, default=140.0)
    p.add_argument("--out", default=str(PR6A / "determinism_1e4.json"))
    a = p.parse_args(argv)

    payload = {
        "what": ("PLAN 6.4 determinism sweep at 1e4 points on the "
                 "tests/test_latent_seam_e2e end-to-end fixture (the whole "
                 "latent stack, fixture-scale data)"),
        "limits": ("under CRN the estimator is a deterministic smooth "
                   "function of theta; both halves of this gate pass BY "
                   "CONSTRUCTION and pass equally on a distorted surrogate. "
                   "Regression guard, NEVER evidence of correctness. Only "
                   "P14 (PR-5b) discriminates."),
        "gate": ("100 repeats bit-identical; max|dlogL| between adjacent "
                 "points < 10x the median"),
        "DARKSIRENS_ZMAX": os.environ["DARKSIRENS_ZMAX"],
        "platform": os.environ.get("JAX_PLATFORMS"),
        "fixture": {"n_events": E._N_EV, "n_samples": E._N_SAMP,
                    "n_selection": E._N_SEL, "M_draw": E.M_DRAW,
                    "n_rows": E.N_ROWS, "n_footprint": E.N_FIT,
                    "b_GW": E.B_GW, "z_depth": E.Z_DEPTH,
                    "n_zgrid": E.NG},
        "arms": {},
    }
    for name in ("latent", "table"):
        res, curve = run(name, a.n_points, a.n_repeats, a.h0_lo, a.h0_hi)
        payload["arms"][name] = res
        np.save(PR6A / f"determinism_1e4_{name}.npy", curve)
        print(f"[{name}] repeats bit-identical={res['repeats']['bit_identical']} "
              f"| adj max/median = {res['adj_ratio_max_over_median']:.4f} "
              f"(gate < 10)", flush=True)
    Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"[out] {a.out}")


if __name__ == "__main__":
    main()
