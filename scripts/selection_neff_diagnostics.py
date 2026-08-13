#!/usr/bin/env python
"""Will this (selection file, population) pair clear the reliability guard?

The guard bounds the variance of the TOTAL log-likelihood estimator
(``darksirens/likelihood/selection.py``)::

    sigma^2_total = pe_variance_sum + N_obs^2 / N_eff  <=  max_likelihood_variance

``scripts/pe_weight_diagnostics.py`` and ``load_gw_samples`` report the FIRST
term.  This reports the second -- which on the production configuration is the
LARGER one (0.564 against 0.273) and is the term that decides the guard.

Read the precision canary section before trusting any number this prints.

Precision canary -- why this script asserts before it measures
-------------------------------------------------------------
The first version of this script silently reported N_eff = 136,200 where the
truth is 118,960: a 14% error, deterministic, and invisible to every obvious
check.  ``jax_enable_x64`` read ``True``, every input array was float64, the
z-grid and distance table were float64, and the two answers were each perfectly
reproducible.  The cause was x64 being enabled AFTER some module-level constant
or cached normalisation had already been built, leaving an internal float32 path
that the flag no longer describes.

It is detectable because the curated ``powerlaw+peak`` preset has NO hard
``m_max`` truncation -- 90% of its weight is an UNTAPERED Gaussian at 35 +- 5 --
so above ``m_max`` the density decays smoothly and "leaves support" only where it
underflows.  That threshold is precision-dependent::

    0.5 ((m - 35)/5)^2 = -ln(realmin)   ->   float64: m > 223.3
                                             float32: m > 101.1

which on the production injection set is 2,372 samples versus 45,038.  So
evaluating ``log_p_pop`` at one probe mass distinguishes the two outright:
at m1src = 150 float64 gives about -265.5 and float32 gives ``-inf``.

:func:`assert_float64_population_path` is that probe.  It costs one tiny
evaluation and it is the only reason a number from this script can be trusted.

What N_eff is NOT
-----------------
Not a property of the file.  It is a property of (product x population model x
redshift prior x parameter point) and varies by orders of magnitude across
population choices, so the population is an explicit argument and the evaluated
point is reported.  The redshift prior here is the comoving-volume prior, i.e.
the SPECTRAL-siren case; a dark-siren run substitutes the catalog prior at each
injection's pixel and will differ.

Usage
-----
    python scripts/selection_neff_diagnostics.py --selection_path sel.h5 --n_obs 259
    python scripts/selection_neff_diagnostics.py --selection_path sel.h5 --gw_path pe.h5
"""
from __future__ import annotations

# MUST precede every other darksirens import: configure_jax_runtime enables x64,
# and its own docstring says callers invoke it "at module startup".  Deferring it
# is the defect described above.
from darksirens.core.jax_config import configure_jax_runtime  # noqa: E402

configure_jax_runtime()

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

#: Probe mass and its float64 log-density under the curated powerlaw+peak
#: fiducial at q = 0.8, z = 0.3.  float32 returns -inf here.
CANARY_M1SRC = 150.0
CANARY_LOG_P_MIN = -400.0


def assert_float64_population_path(log_p_pop, pop_params) -> float:
    """Fail loudly if the population model is running on a float32 path.

    See the module docstring: dtype inspection cannot detect this, because the
    inputs and the advertised flag are both float64 while an internal cached
    constant is not.  Evaluating past the float32 underflow edge can.
    """
    import jax
    import jax.numpy as jnp

    if not jax.config.jax_enable_x64:
        raise SystemExit(
            "x64 is not enabled; selection weights would be float32 and N_eff "
            "wrong by ~14%. configure_jax_runtime() must run at module import."
        )
    m1 = jnp.asarray([CANARY_M1SRC])
    value = float(
        log_p_pop(m1, jnp.asarray([0.8]), jnp.asarray([0.3]),
                  jnp.asarray([0.0]), pop_params)[0]
    )
    if not np.isfinite(value) or value < CANARY_LOG_P_MIN:
        raise SystemExit(
            f"precision canary FAILED: log_p_pop at m1src={CANARY_M1SRC} is "
            f"{value!r}, but a float64 path gives about -265.5. An internal "
            "float32 constant is in play (x64 enabled too late), and N_eff "
            "would be silently wrong by ~14%. Refusing to report a number."
        )
    return value


def _load_selection_arrays(path: Path):
    """Raw read of the columns the selection weight needs.

    Deliberately not ``load_selection_samples``: that loader hard-rejects any
    ``spin_basis`` other than ``chieff``, and comparing a component-basis product
    against a chieff one is what this script is for.  No chi_eff prior swap is
    applied here -- whatever convention the file's pdraw carries is used.
    """
    with h5py.File(path, "r") as f:
        need = ("m1det", "m2det", "dL", "chieff", "pdraw")
        missing = [k for k in need if k not in f]
        if missing:
            raise SystemExit(f"{path}: missing dataset(s) {missing}")
        if "ndraw" not in f.attrs:
            raise SystemExit(f"{path}: missing required attr 'ndraw'")
        cols = {k: np.asarray(f[k], dtype=float) for k in need}
        for k in ("a1", "a2", "cost1", "cost2"):
            if k in f:
                cols[k] = np.asarray(f[k], dtype=float)
        basis = f.attrs.get("spin_basis", b"chieff")
        if isinstance(basis, bytes):
            basis = basis.decode()
        meta = {
            "ndraw": float(f.attrs["ndraw"]),
            "spin_basis": str(basis),
            "format_version": str(f.attrs.get("format_version", "?")),
            "n_detected": int(cols["m1det"].size),
        }
    return cols, meta


def selection_neff(selection_path: Path, pop_model: str, H0: float, Om0: float,
                   w0: float, wa: float) -> dict:
    """log_mu and N_eff for this (product, population, volume-prior) point."""
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp

    from darksirens.core.types import CosmoParams, SurveyParams
    from darksirens.gw.populations import pop_model_parser
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.utils import log_target_density_base_and_z
    from darksirens.likelihood.selection import _lse_to_log_mu_neff
    from darksirens.redshift.volume import log_volume_prior_vmap

    log_p_pop = pop_model_parser(pop_model=pop_model)
    pop_params = jnp.asarray(get_fixed_population_params(pop_model))
    # The canary detects PROCESS state (a float32 constant cached before x64
    # was enabled), not a property of the measured model, and its -265 band
    # is specific to the curated preset's untapered Gaussian tail -- a
    # hard-truncated mass model is legitimately -inf at the probe mass.  So
    # it always runs on the curated powerlaw+peak regardless of --pop_model.
    canary = assert_float64_population_path(
        pop_model_parser(pop_model="powerlaw+peak"),
        jnp.asarray(get_fixed_population_params("powerlaw+peak")))

    # A component-spin population (DS-08) consumes the (N, 4) spin block
    # (a1, a2, cost1, cost2) and ignores chieff -- the decisive measurement
    # the chieff-marginal read below cannot make.
    from darksirens.gw.populations.registry import get_model

    _model = get_model(pop_model)
    _spin_comps = ([_model.spin_component]
                   if hasattr(_model, "spin_component") else [])
    _mixture = getattr(_model, "mixture", None)
    if _mixture is not None:
        _spin_comps.extend(getattr(_mixture, "spin_components", ()))
    consumes_spin_block = any(
        getattr(c, "consumes_spin_block", False) for c in _spin_comps)

    cols, meta = _load_selection_arrays(selection_path)
    spin = None
    if consumes_spin_block:
        missing = [k for k in ("a1", "a2", "cost1", "cost2") if k not in cols]
        if missing:
            raise SystemExit(
                f"--pop_model {pop_model} consumes the component-spin block, "
                f"but {selection_path} carries no {missing} datasets; use a "
                "component-basis export.")
        spin = jnp.asarray(np.column_stack(
            [cols[k] for k in ("a1", "a2", "cost1", "cost2")]))
    m1 = jnp.asarray(cols["m1det"])
    q = jnp.asarray(cols["m2det"] / np.maximum(cols["m1det"], 1e-300))
    dL = jnp.asarray(cols["dL"])
    chieff = jnp.asarray(cols["chieff"])
    pdraw = jnp.asarray(cols["pdraw"])
    n = int(m1.shape[0])

    cosmo = CosmoParams(H0=H0, Om0=Om0, w0=w0, wa=wa)
    # Survey nuisances do not enter the selection weight; fiducials for signature.
    survey = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0,
                          b_miss=0.0, alpha_miss=1.0)
    base, z = log_target_density_base_and_z(
        m1, q, dL, chieff, jnp.zeros(n, dtype=jnp.int32), pdraw,
        cosmo, survey, pop_params, None, log_p_pop,
        spin=spin,
    )
    # Reported because it is the float64/float32 discriminator on this preset.
    if spin is None:
        _lp = log_p_pop(m1 / (1.0 + z), q, z, chieff, pop_params)
    else:
        _lp = log_p_pop(m1 / (1.0 + z), q, z, chieff, pop_params, spin=spin)
    n_out_of_support = int((~np.isfinite(np.asarray(_lp))).sum())
    ldw = base + log_volume_prior_vmap(z, cosmo, survey)
    ldw = jnp.where(jnp.isfinite(ldw) & (pdraw > 0.0), ldw, -jnp.inf)
    finite = jnp.isfinite(ldw)
    safe = jnp.where(finite, ldw, -1e30)
    log_mu, Neff, _ = _lse_to_log_mu_neff(
        logsumexp(safe), logsumexp(2.0 * safe), meta["ndraw"])
    # log_mu is reported for diagnosis but is NOT portable: it carries the
    # population normalisation, which is process-state dependent.  The constant
    # cancels within a likelihood evaluation (c^N_obs in the numerator against
    # c^N_obs from mu^N_obs), so posteriors, logZ and N_eff are all unaffected --
    # but two log_mu values from different processes are not comparable.
    meta.update(n_finite_weights=int(finite.sum()), log_mu=float(log_mu),
                log_mu_comparable_across_processes=False,
                Neff=float(Neff), pop_model=pop_model, H0=H0, Om0=Om0,
                w0=w0, wa=wa, canary_log_p=canary,
                n_pop_out_of_support=n_out_of_support,
                consumes_spin_block=bool(consumes_spin_block))
    return meta


def verdict(Neff: float, n_obs: int, pe_variance_sum: float,
            max_likelihood_variance: float) -> dict:
    """Both terms of the budget, and whether the guard clears."""
    sel_var = (n_obs * n_obs) / Neff if Neff > 0 else float("inf")
    total = pe_variance_sum + sel_var
    budget = max(max_likelihood_variance - pe_variance_sum, 1e-12)
    threshold = max(5.0 * n_obs, (n_obs * n_obs) / budget)
    return {
        "n_obs": n_obs, "pe_variance_sum": pe_variance_sum,
        "selection_variance": sel_var, "sigma2_total": total,
        "max_likelihood_variance": max_likelihood_variance,
        "threshold": threshold,
        "margin": Neff / threshold if threshold else float("nan"),
        "passes": bool(total < max_likelihood_variance),
        "sparse_floor": 5.0 * n_obs,
        "variance_criterion_limited": ((n_obs * n_obs) / budget) > (5.0 * n_obs),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection_path", required=True, type=Path)
    ap.add_argument("--gw_path", type=Path, default=None,
                    help="PE file; pe_variance_sum and N_obs are measured from it")
    ap.add_argument("--n_obs", type=int, default=None)
    ap.add_argument("--pe_variance_sum", type=float, default=None)
    ap.add_argument("--pop_model", default="powerlaw+peak")
    ap.add_argument("--max_likelihood_variance", type=float, default=1.0)
    ap.add_argument("--H0", type=float, default=67.74)
    ap.add_argument("--Om0", type=float, default=0.3089)
    ap.add_argument("--w0", type=float, default=-1.0)
    ap.add_argument("--wa", type=float, default=0.0)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    pe_var = args.pe_variance_sum if args.pe_variance_sum is not None else 0.0
    n_obs = args.n_obs
    if args.gw_path is not None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_pwd", Path(__file__).with_name("pe_weight_diagnostics.py"))
        pwd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pwd)
        with h5py.File(args.gw_path, "r") as f:
            nobs_f, nsamp = int(f.attrs["nobs"]), int(f.attrs["nsamp"])
            p_pe = np.asarray(f["p_pe"], dtype=float).reshape(nobs_f, nsamp)
        if args.pe_variance_sum is None:
            pe_var = float(pwd.per_event_weight_stats(p_pe)["variance"].sum())
        if n_obs is None:
            n_obs = nobs_f
    if n_obs is None:
        raise SystemExit("--n_obs is required unless --gw_path is given")
    if args.max_likelihood_variance <= 0.0:
        raise SystemExit("--max_likelihood_variance must be positive")

    m = selection_neff(args.selection_path, args.pop_model,
                       args.H0, args.Om0, args.w0, args.wa)
    v = verdict(m["Neff"], n_obs, pe_var, args.max_likelihood_variance)

    print(f"Selection N_eff diagnostics: {args.selection_path}")
    print(f"  format / spin basis      {m['format_version']} / {m['spin_basis']}")
    print(f"  detected / ndraw         {m['n_detected']:,} / {m['ndraw']:,.0f}"
          f"   ({m['n_finite_weights']:,} finite weights)")
    print(f"  population               {m['pop_model']} at registry fiducial")
    print(f"  precision canary         log_p_pop({CANARY_M1SRC:g}) = "
          f"{m['canary_log_p']:.2f}  [float64 OK]")
    suffix = ("   (underflow edge, NOT m_max: this preset's peak is untapered)"
              if m["pop_model"] == "powerlaw+peak" else "")
    print(f"  pop out-of-support       {m['n_pop_out_of_support']:,}{suffix}")
    print(f"  log_mu                   {m['log_mu']:+.4f}"
          "   [NOT comparable across processes -- see below]")
    print(f"  selection N_eff          {m['Neff']:,.0f}")
    print()
    print(f"  sigma^2 = pe + selection = total   (cap {v['max_likelihood_variance']:g})")
    print(f"    pe_variance_sum        {v['pe_variance_sum']:.4f}")
    print(f"    selection variance     {v['selection_variance']:.4f}"
          f"   (= N_obs^2 / N_eff at N_obs={n_obs})")
    print(f"    TOTAL                  {v['sigma2_total']:.4f}"
          f"   -> {'PASS' if v['passes'] else '*** FAIL ***'}")
    print(f"  required N_eff           {v['threshold']:,.0f}"
          f"   (margin {v['margin']:.2f}x)")
    print(f"  sparse floor 5*N_obs     {v['sparse_floor']:,.0f}"
          f"   {'(not binding)' if v['variance_criterion_limited'] else '(BINDING)'}")
    print()
    print("  N_eff depends on the population model, the redshift prior and the")
    print("  parameter point -- it is not a property of the file. This is the")
    print("  comoving-volume (spectral-siren) prior at one fiducial point.")
    print()
    print("  log_mu carries the population's normalisation constant, which is")
    print("  process-state dependent (measured: a factor 1.9 between two import")
    print("  orders). Do NOT compare this log_mu against one from inside a run,")
    print("  from another process, or against a stored reference. Within a single")
    print("  likelihood evaluation the constant cancels exactly -- it multiplies")
    print("  each per-event numerator by c and mu by c, giving c^N_obs on both")
    print("  sides -- so posteriors and logZ are unaffected. N_eff is likewise")
    print("  scale-invariant. It is only this reported log_mu, taken in")
    print("  isolation, that is not a portable number.")
    if m.get("consumes_spin_block"):
        print()
        print("  spin: the population model consumed the file's component-spin block")
        print("  (a1, a2, cost1, cost2) directly -- this IS the 4-D spin measurement,")
        print("  with no flat-orthogonal assumption.")
    elif m["spin_basis"] != "chieff":
        print()
        print(f"  [!] spin_basis={m['spin_basis']!r} under a 1-D chi_eff population model:")
        print("      applying it to a non-chieff draw density assumes the population is")
        print("      flat in the orthogonal spin directions. N_eff is scale-invariant so")
        print("      that does not affect the number, but a 4-D spin model would differ")
        print("      (measure it with --pop_model gwtc3_plpeak_component_spin).")

    out = {**m, **v}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=float) + "\n")
        print(f"\n  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
