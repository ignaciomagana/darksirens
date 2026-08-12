#!/usr/bin/env python
"""How much of the total-variance budget does a PE file spend before inference?

Why this exists
---------------
The hierarchical likelihood's reliability guard bounds the variance of the TOTAL
log-likelihood estimator, not the selection term alone
(``darksirens/likelihood/selection.py``)::

    budget    = max(max_likelihood_variance - pe_variance_sum, _MIN_VARIANCE_BUDGET)
    threshold = max(5 * N_obs, N_obs**2 / budget)      # required selection N_eff

``pe_variance_sum`` is the sum over events of the delta-method variance of each
event's log-evidence, so a PE product whose importance weights are degenerate
spends the budget before the population model, redshift prior, Jacobian or
selection term contribute anything -- and every nat it spends RAISES the
selection N_eff the run must clear.

Nothing surfaced that.  ``pe_variance_sum`` is threaded into the guard from
``likelihood/core.py`` and reported in the LENSING CLI's diagnostics, but a
standard dark-siren run has no way to see it, so a run can sit at three quarters
of its budget, or fail the guard, with no indication of why.

Measured on ``gwsamples_bbh_whitelist_all_events.h5`` (259 events x 4096
samples) when this script was written: ``pe_variance_sum = 0.2728`` of a budget
of 1.0, i.e. **27% already spent**, inflating the required selection N_eff from
67,081 to 92,270 (+37%).  108/259 events had ESS/nsamp < 0.5 and 30 below 0.1;
the worst 5 events carried a third of the total.

What this measures, and what it does not
----------------------------------------
This reports the **PE-prior reweighting** contribution: the ESS of ``1/p_pe``
within each event.  The real ``pe_variance_sum`` is computed from the FULL
per-sample weight (population + redshift prior + Jacobian), which can only be
had by running the likelihood at a parameter point.  So the number here is a
FLOOR on the PE term, and the true remaining budget is smaller than reported.
It is the cosmology-independent part -- it depends on the file alone, which is
what makes it worth checking before a run rather than after.

Usage
-----
    python scripts/pe_weight_diagnostics.py --gw_path gwsamples_bbh.h5
    python scripts/pe_weight_diagnostics.py --gw_path ... --max_likelihood_variance 2.0
    python scripts/pe_weight_diagnostics.py --gw_path ... --json out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

# Mirrors likelihood/selection.py so this script needs no JAX runtime.
DEFAULT_MAX_LIKELIHOOD_VARIANCE = 1.0
SPARSE_FLOOR_FACTOR = 5.0
MIN_VARIANCE_BUDGET = 1e-12


def per_event_weight_stats(p_pe: np.ndarray) -> dict:
    """ESS and delta-method variance of the ``1/p_pe`` reweighting, per event.

    ``p_pe`` is ``(n_events, nsamp)``.  Non-positive entries are given zero
    weight, matching the likelihood's own ``prior_wt > 0`` masking
    (``likelihood/core.py:968`` and siblings) -- they are dropped, not counted.

    The variance is the same estimator ``log_evidence_and_mc_variance`` uses,
    ``sigma^2 = sum(w^2)/sum(w)^2 - 1/n``, with ``n`` the FULL sample count
    including masked entries, because masked samples are zero-weight members of
    the same PE set (see that function's docstring).
    """
    n_events, nsamp = p_pe.shape
    good = p_pe > 0.0
    w = np.where(good, 1.0 / np.where(good, p_pe, 1.0), 0.0)
    sw = w.sum(axis=1)
    sw2 = (w ** 2).sum(axis=1)
    ess = np.where(sw2 > 0.0, sw ** 2 / np.where(sw2 > 0.0, sw2, 1.0), 0.0)
    variance = np.maximum(
        np.where(sw > 0.0, sw2 / np.where(sw > 0.0, sw ** 2, 1.0), 0.0) - 1.0 / nsamp,
        0.0,
    )
    return {
        "n_events": int(n_events),
        "nsamp": int(nsamp),
        "ess": ess,
        "ess_frac": ess / nsamp,
        "variance": variance,
        "n_masked": int((~good).sum()),
    }


def guard_thresholds(pe_variance_sum: float, n_events: int,
                     max_likelihood_variance: float) -> dict:
    """Required selection N_eff, with and without the PE variance spend."""
    budget = max(max_likelihood_variance - pe_variance_sum, MIN_VARIANCE_BUDGET)
    n = float(n_events)
    with_pe = max(SPARSE_FLOOR_FACTOR * n, n * n / budget)
    without_pe = max(SPARSE_FLOOR_FACTOR * n, n * n / max_likelihood_variance)
    return {
        "budget_remaining": budget,
        "threshold": with_pe,
        "threshold_if_pe_variance_were_zero": without_pe,
        "inflation_factor": with_pe / without_pe if without_pe else float("nan"),
        "sparse_floor": SPARSE_FLOOR_FACTOR * n,
        "variance_criterion_limited": (n * n / budget) > (SPARSE_FLOOR_FACTOR * n),
    }


def report(gw_path: Path, max_likelihood_variance: float) -> dict:
    with h5py.File(gw_path, "r") as f:
        if "p_pe" not in f:
            raise SystemExit(f"{gw_path}: no 'p_pe' dataset (not a gwcat PE export?)")
        for attr in ("nobs", "nsamp"):
            if attr not in f.attrs:
                raise SystemExit(f"{gw_path}: missing required attr {attr!r}")
        n_events = int(f.attrs["nobs"])
        nsamp = int(f.attrs["nsamp"])
        raw = np.asarray(f["p_pe"], dtype=float)
        if raw.size != n_events * nsamp:
            raise SystemExit(
                f"{gw_path}: p_pe has {raw.size} entries but nobs*nsamp = "
                f"{n_events * nsamp}; the layout is strictly rectangular"
            )
        p_pe = raw.reshape(n_events, nsamp)

    stats = per_event_weight_stats(p_pe)
    frac = stats["ess_frac"]
    var = stats["variance"]
    total = float(var.sum())
    guard = guard_thresholds(total, n_events, max_likelihood_variance)

    worst = np.argsort(var)[::-1]
    worst5 = worst[:5]

    out = {
        "gw_path": str(gw_path),
        "n_events": n_events,
        "nsamp": nsamp,
        "n_masked_samples": stats["n_masked"],
        "max_likelihood_variance": max_likelihood_variance,
        "ess_frac": {
            "min": float(frac.min()), "p05": float(np.quantile(frac, 0.05)),
            "median": float(np.median(frac)), "mean": float(frac.mean()),
            "max": float(frac.max()),
        },
        "n_events_ess_frac_below_0p5": int((frac < 0.5).sum()),
        "n_events_ess_frac_below_0p1": int((frac < 0.1).sum()),
        "pe_variance_sum": total,
        "pe_variance_fraction_of_budget": total / max_likelihood_variance,
        "worst_events": [
            {"index": int(i), "ess_frac": float(frac[i]), "variance": float(var[i])}
            for i in worst5
        ],
        "worst5_variance_share": (
            float(var[worst5].sum() / total) if total > 0 else 0.0
        ),
        **guard,
    }

    print(f"PE weight diagnostics: {gw_path}")
    print(f"  events x samples          {n_events} x {nsamp}")
    if stats["n_masked"]:
        print(f"  non-positive p_pe samples {stats['n_masked']} "
              "(zero-weighted, as the likelihood does)")
    print("  ESS/nsamp                 "
          f"min {frac.min():.4f}  p05 {np.quantile(frac, 0.05):.4f}  "
          f"median {np.median(frac):.4f}  max {frac.max():.4f}")
    print(f"  events below 0.5 / 0.1    {int((frac < 0.5).sum())} / "
          f"{int((frac < 0.1).sum())}  of {n_events}")
    print()
    print(f"  pe_variance_sum           {total:.4f}  "
          f"({100.0 * total / max_likelihood_variance:.1f}% of "
          f"max_likelihood_variance = {max_likelihood_variance:g})")
    print(f"  worst 5 events            {out['worst5_variance_share'] * 100:.1f}% "
          "of the total variance")
    print(f"  budget remaining          {guard['budget_remaining']:.4f}")
    print()
    print(f"  required selection N_eff  {guard['threshold']:,.0f}")
    print(f"    if PE variance were 0   {guard['threshold_if_pe_variance_were_zero']:,.0f}"
          f"   (inflation x{guard['inflation_factor']:.2f})")
    print(f"    sparse floor 5*N_obs    {guard['sparse_floor']:,.0f}"
          f"   {'(not binding)' if guard['variance_criterion_limited'] else '(BINDING)'}")
    print()
    print("  NOTE: PE-prior contribution only. The real pe_variance_sum uses the full")
    print("  per-sample weight (population + redshift prior + Jacobian), so this is a")
    print("  FLOOR and the true remaining budget is smaller.")
    if total / max_likelihood_variance > 0.2:
        print()
        print("  The cheaper lever is usually the PE file, not --max_likelihood_variance:")
        print("  re-analysing or dropping the worst few events buys back most of the spend.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gw_path", required=True, type=Path,
                    help="gwcat PE export (needs p_pe, nobs, nsamp)")
    ap.add_argument("--max_likelihood_variance", type=float,
                    default=DEFAULT_MAX_LIKELIHOOD_VARIANCE,
                    help="total-variance budget the run will use (default 1.0)")
    ap.add_argument("--json", type=Path, default=None,
                    help="also write the report as JSON")
    args = ap.parse_args(argv)

    if args.max_likelihood_variance <= 0.0:
        raise SystemExit("--max_likelihood_variance must be positive")
    out = report(args.gw_path, args.max_likelihood_variance)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=float) + "\n")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
