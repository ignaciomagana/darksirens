#!/usr/bin/env python
"""Diagnose the selection variance guard on YOUR darksirens_inference run.

When a nested-sampling run (dynesty/tinyns) hangs at "find initial live
points", or the new preflight reports ``0/32 prior draws have finite logL``,
the cause is almost always the selection variance guard returning ``-inf``:
the GWTC-4.0/5.0 total-log-likelihood-variance criterion

    sigma^2_lnL = sum_i sigma_i^2 + N_obs^2 / Neff_sel  >  max_likelihood_variance

(plus the Vitale ``Neff > 5*N_obs`` floor).  This script measures those
quantities on your OWN data and prints a verdict: the measured sigma^2_lnL at
the fiducial population point, whether the guard admits it, the smallest
``--max_likelihood_variance`` that would, and the recommended flags.

It does this WITHOUT running a sampler.  It replays your exact CLI invocation
through ``darksirens.cli.inference.main()`` with two monkeypatches:

  * ``run_sampler`` is intercepted to CAPTURE the built likelihood, prior
    transform and bounds and then stop (no sampling happens);
  * ``selection_log_correction`` is wrapped -- in its source module and in every
    likelihood module that has already bound it, so the standard, flow-surrogate
    and cluster stacks are all covered -- to report ``Neff``,
    ``pe_variance_sum`` and ``log_mu`` (via ``jax.debug.print`` and a
    ``jax.debug.callback`` that stashes the concrete values) each time the
    guard is evaluated.

Usage
-----
    python scripts/diagnose_selection_guard.py [--diagnose-draws N] \
        -- <the exact args you pass to darksirens_inference>

Everything after ``--`` is forwarded verbatim to the inference CLI.  A
``--sampler`` is injected (``dynesty``) if you did not pass one, since the CLI
requires it (the sampler is never actually run).

Dependency-light: numpy, jax, and darksirens only.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import jax
import jax.numpy as jnp


class _CapturedRunSampler(Exception):
    """Sentinel raised inside the patched run_sampler to stop before sampling."""


def _split_argv(argv):
    """Return (script_args, passthrough_args) split on the first bare ``--``."""
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1:]
    # No separator: treat everything as passthrough so a forgotten ``--`` still
    # does something sensible.
    return [], argv


def _patch_selection_guard(wrapper, original):
    """Install ``wrapper`` everywhere ``selection_log_correction`` is reachable.

    Every likelihood stack binds the symbol in its OWN module namespace via
    ``from ..selection import selection_log_correction`` -- ``core``,
    ``flow_events`` (the ``--gw_flows_path`` path), ``cluster_selection`` (the
    lensing stack) -- so patching a single module measures one code path and
    silently misses the others, which is how a flow-surrogate run used to collect
    zero records and be told the guard was never evaluated.  Patching the SOURCE
    module also fixes the stacks the CLI imports LAZILY, since their ``from ...
    import`` then picks up the wrapper.

    Returns the list of patched modules (pass to :func:`_restore_selection_guard`).
    """
    import darksirens.likelihood.selection as selection_mod

    patched = [selection_mod] + [
        mod for name, mod in list(sys.modules.items())
        if name.startswith("darksirens.") and mod is not selection_mod
        and getattr(mod, "selection_log_correction", None) is original
    ]
    for mod in patched:
        mod.selection_log_correction = wrapper
    return patched


def _restore_selection_guard(patched, original):
    for mod in patched:
        mod.selection_log_correction = original


def _fiducial_point(labels, lower, upper, opts):
    """Sampled-space point at the registry fiducial, midpoint fallback, clipped.

    The selection injection campaigns are sized at the fiducial population, so
    the guard is clear there by construction while random prior draws collapse
    Neff.  Labels without a population fiducial (cosmology/survey/mixture
    blocks) fall back to the prior midpoint.
    """
    from darksirens.gw.populations import (
        get_fixed_population_params,
        pop_model_prior_parser,
    )

    shared = dict(
        shared_beta=getattr(opts, "shared_beta", True),
        shared_spin=getattr(opts, "shared_spin", True),
        shared_gamma=getattr(opts, "shared_gamma", True),
    )
    fid_vec = np.asarray(
        get_fixed_population_params(opts.pop_model, **shared), dtype=float
    )
    _, _, pop_labels, _, _ = pop_model_prior_parser(opts.pop_model, **shared)
    fid = {lab: float(v) for lab, v in zip(pop_labels, fid_vec)}

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    mid = 0.5 * (lower + upper)
    point = np.array([fid.get(lab, m) for lab, m in zip(labels, mid)], dtype=float)
    eps = 1e-6 * (upper - lower)
    return np.clip(point, lower + eps, upper - eps)


def _finite_fraction(likelihood, prior_transform, ndims, n_draws, seed):
    """Fraction of ``n_draws`` seeded prior draws with a finite logL."""
    rng = np.random.default_rng(int(seed) ^ 0xD1A6)
    n_finite = 0
    for _ in range(n_draws):
        u = rng.random(ndims)
        theta = prior_transform(jnp.asarray(u))
        val = float(np.asarray(likelihood(jnp.asarray(theta))))
        if np.isfinite(val):
            n_finite += 1
    return n_finite, n_draws


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    script_args, passthrough = _split_argv(argv)

    ap = argparse.ArgumentParser(
        prog="diagnose_selection_guard.py",
        description="Measure the selection variance guard on your data "
                    "(everything after -- is forwarded to darksirens_inference).",
    )
    ap.add_argument("--diagnose-draws", type=int, default=64,
                    help="Prior draws used to estimate the finite fraction "
                         "(default 64).")
    ap.add_argument("--diagnose-seed", type=int, default=0,
                    help="Seed for the finite-fraction prior draws (default 0); "
                         "distinct from the CLI's own --seed, which stays in the "
                         "passthrough args after `--`.")
    ns = ap.parse_args(script_args)

    if not passthrough:
        ap.error("no darksirens_inference args given; pass them after `--`.")

    # The CLI requires --sampler; the sampler is never run (we stop at
    # run_sampler), so inject a harmless default when absent.
    if "--sampler" not in passthrough:
        passthrough = passthrough + ["--sampler", "dynesty"]
        print("[diagnose] no --sampler given; injecting `--sampler dynesty` "
              "(not actually run).", flush=True)

    import darksirens.cli.inference as inference_cli
    import darksirens.likelihood.selection as selection_mod
    from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE

    # ---- monkeypatch 1: capture the likelihood/prior and stop ---------------
    captured = {}

    def _capture_run_sampler(method, likelihood, prior_transform, labels,
                             lower_bound, upper_bound, opts, prior_kinds=None):
        captured.update(
            likelihood=likelihood,
            prior_transform=prior_transform,
            labels=list(labels),
            lower=np.asarray(lower_bound, dtype=float),
            upper=np.asarray(upper_bound, dtype=float),
            opts=opts,
        )
        raise _CapturedRunSampler()

    inference_cli.run_sampler = _capture_run_sampler

    # ---- monkeypatch 2: report the guard inputs -----------------------------
    guard_records = []
    _orig_selection = selection_mod.selection_log_correction

    def _record(n, p, m, ne, mv):
        # Runs under jax.debug.callback with CONCRETE values, so it is safe to
        # cast here even when the CLI threads max_likelihood_variance / nEvents
        # as traced arrays into the jitted likelihood (a plain float() at trace
        # time raises ConcretizationTypeError).
        guard_records.append(dict(
            Neff=float(n), pe_variance_sum=float(p), log_mu=float(m),
            nEvents=float(ne), max_likelihood_variance=float(mv),
        ))

    def _wrapped_selection(*args, **kwargs):
        log_mu, Neff, nEvents = args[0], args[1], args[2]
        pe_var = kwargs.get("pe_variance_sum", 0.0)
        max_var = kwargs.get("max_likelihood_variance",
                             DEFAULT_MAX_LIKELIHOOD_VARIANCE)
        jax.debug.print(
            "[guard] Neff={n}  pe_variance_sum={p}  log_mu={m}",
            n=Neff, p=pe_var, m=log_mu,
        )
        jax.debug.callback(_record, Neff, pe_var, log_mu, nEvents, max_var)
        return _orig_selection(*args, **kwargs)

    _patched = _patch_selection_guard(_wrapped_selection, _orig_selection)

    # ---- replay the user's CLI up to (but not into) the sampler -------------
    print("[diagnose] replaying darksirens_inference "
          f"(args: {' '.join(passthrough)}) ...", flush=True)
    saved_argv = sys.argv
    sys.argv = ["darksirens_inference"] + passthrough
    try:
        try:
            inference_cli.main()
        except _CapturedRunSampler:
            pass
        except SystemExit as exc:
            if exc.code not in (0, None):
                print(f"[diagnose] the inference CLI exited early (code "
                      f"{exc.code}) before the sampler was reached; fix the "
                      "reported error and re-run.", flush=True)
                return int(exc.code)
        finally:
            sys.argv = saved_argv
        # The guard wrapper stays installed through the fiducial and prior-draw
        # evaluations below (the captured likelihood re-enters core at call
        # time); _diagnose does the analysis and returns an exit code.
        return _diagnose(captured, guard_records, ns)
    finally:
        _restore_selection_guard(_patched, _orig_selection)


def _diagnose(captured, guard_records, ns):
    if "likelihood" not in captured:
        print("[diagnose] ERROR: never reached run_sampler; the likelihood was "
              "not built. See the CLI output above for the failure.", flush=True)
        return 2

    likelihood = captured["likelihood"]
    prior_transform = captured["prior_transform"]
    labels = captured["labels"]
    lower = captured["lower"]
    upper = captured["upper"]
    opts = captured["opts"]
    ndims = len(labels)

    print()
    print("=" * 72)
    print("SELECTION-GUARD DIAGNOSIS")
    print("=" * 72)
    print(f"  free dimensions : {ndims}  {labels}")

    if ndims == 0:
        print("  0 free parameters — the sampler is short-circuited and no "
              "guard diagnosis is needed.")
        return 0

    # ---- evaluate the fiducial population point -----------------------------
    fid_point = _fiducial_point(labels, lower, upper, opts)
    guard_records.clear()
    fid_logl = float(np.asarray(likelihood(jnp.asarray(fid_point))))
    print(f"  fiducial logL   : {fid_logl:.6g}")

    cap = float(getattr(opts, "max_likelihood_variance",
                        guard_records[-1]["max_likelihood_variance"]
                        if guard_records else 1.0))

    if not guard_records:
        print("  [!] the selection guard was not evaluated at the fiducial "
              "(the likelihood may short-circuit to -inf earlier, this universe "
              "model may not use it, or a likelihood module resolved the symbol "
              "before the wrapper was installed). Inspect the CLI output above.")
    else:
        rec = guard_records[-1]
        Neff = rec["Neff"]
        pe_var = rec["pe_variance_sum"]
        n_obs = rec["nEvents"]
        sel_component = (n_obs * n_obs) / Neff if Neff > 0 else np.inf
        sigma2 = pe_var + sel_component
        vitale_floor = 5.0 * n_obs
        vitale_ok = Neff > vitale_floor
        variance_ok = sigma2 < cap
        admitted = vitale_ok and variance_ok

        print()
        print("  measured at the fiducial population point:")
        print(f"    N_obs                 = {n_obs:.0f}")
        print(f"    Neff_sel              = {Neff:.6g}")
        print(f"    pe_variance_sum       = {pe_var:.6g}")
        print(f"    N_obs^2 / Neff_sel    = {sel_component:.6g}")
        print(f"    sigma^2_lnL           = {sigma2:.6g}")
        print(f"    max_likelihood_variance (cap) = {cap:.6g}")
        print()
        print("  criteria:")
        print(f"    variance: sigma^2_lnL < cap   -> {sigma2:.4g} < {cap:.4g} "
              f"= {variance_ok}")
        print(f"    Vitale:   Neff > 5*N_obs      -> {Neff:.4g} > "
              f"{vitale_floor:.4g} = {vitale_ok}")
        print()
        if admitted:
            print("  VERDICT: the fiducial is ADMITTED by the guard.")
            print("    If the sampler still stalls, the problem is elsewhere in "
                  "the prior box (run with the built-in preflight ON to see the "
                  "finite fraction).")
        else:
            print("  VERDICT: the fiducial is GUARDED (-inf).")
            if not vitale_ok:
                print(f"    The Vitale floor is VIOLATED (Neff={Neff:.4g} <= "
                      f"5*N_obs={vitale_floor:.4g}); NO value of "
                      "--max_likelihood_variance can admit it — the selection "
                      "injection campaign is too small at this point.")
            else:
                smallest_cap = sigma2
                print(f"    Smallest --max_likelihood_variance that admits the "
                      f"fiducial (variance-wise): {smallest_cap:.6g} "
                      f"(use a value strictly above it, e.g. "
                      f"{smallest_cap * 1.1:.4g}).")
            print()
            print("  Recommended flags:")
            print("    --selection_neff_guard soft      (finite penalized wall; "
                  "the sampler initializes and is pushed toward the valid "
                  "region — numpyro-style smoothing, valid for nested "
                  "diagnosis)")
            if vitale_ok:
                print(f"    --max_likelihood_variance {sigma2 * 1.1:.4g}   "
                      "(accept a larger MC variance; verify your science can "
                      "tolerate it)")

    # ---- finite fraction over the prior -------------------------------------
    n_finite, n_total = _finite_fraction(
        likelihood, prior_transform, ndims, ns.diagnose_draws, ns.diagnose_seed
    )
    frac = n_finite / n_total if n_total else 0.0
    print()
    print(f"  prior finite fraction: {n_finite}/{n_total} draws finite "
          f"({100.0 * frac:.1f}%).")
    if n_finite == 0:
        print("    dynesty/tinyns would reject-sample forever here; use the "
              "soft guard or a larger cap (see the verdict above).")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
