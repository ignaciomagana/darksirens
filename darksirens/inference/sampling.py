import os

import numpy as np
import jax
import jax.numpy as jnp
from darksirens.inference.checkpointing import (
    install_dynesty_checkpointing,
    plan_from_opts,
    restore_dynesty_sampler,
)
from darksirens.inference.tinyns_config import build_tinyns_config, tinyns_run_kwargs, tinyns_sampler_kwargs

_TINYNS_DIAGNOSTIC_FIELDS = (
    "success", "message", "logz", "logzerr", "information", "niter",
    "ndead", "nlive", "ndim", "ncall", "seconds", "posterior_ess",
    "max_weight_fraction", "live_weight_fraction", "dead_weight_fraction",
    "final_delta_logz", "final_logz_dead", "final_logl_live_max",
    "replacement_chains", "replacement_batch_ncall", "replacement_mean_ncall",
    "replacement_mean_batches", "replacement_max_batches", "replacement_failures",
    "replacement_rescue_used", "replacement_rescue_attempts",
    "replacement_rescue_successes", "replacement_rescue_failures",
    "replacement_rescue_stage_counts", "replacement_rescue_ncall",
    "replacement_rescue_max_stage", "terminated_after_partial_block_failure",
    "partial_block_failure_delta_logz", "insertion_rank_mean_z",
    "insertion_rank_std_ratio",
)


def _dead_point_block(logl, logwt, n_live=None):
    """Package a nested sampler's DEAD-POINT arrays for :mod:`darksirens.io.results`.

    ``logl``/``logwt`` are indexed by retired live point, in retirement order,
    NOT by posterior sample: the ``samples`` a run returns are the equal-weight
    resample of these, drawn with replacement.  They are kept because logZ, the
    logX shrinkage ladder, the information H, evidence bootstraps and dynesty
    runplots are functions of the dead points alone and are unrecoverable once
    the resample has been taken.

    Returns ``None`` (and warns) when the sampler did not expose a usable pair,
    so an unfamiliar result object costs a diagnostic, never the run: this is
    called at the very end of a possibly multi-day job.
    """
    if logl is None or logwt is None:
        # An older/unfamiliar sampler build that does not expose the record.
        return None
    try:
        logl_arr = np.asarray(logl, dtype=float).ravel()
        logwt_arr = np.asarray(logwt, dtype=float).ravel()
    except (TypeError, ValueError) as exc:      # pragma: no cover - defensive
        print(f"  [!] dead-point arrays unreadable ({exc}); not persisting them.",
              flush=True)
        return None
    if logl_arr.size == 0 or logl_arr.shape != logwt_arr.shape:
        print(
            "  [!] dead-point logl/logwt have shapes "
            f"{logl_arr.shape}/{logwt_arr.shape}; not persisting them.",
            flush=True,
        )
        return None
    block = {"logl": logl_arr, "logwt": logwt_arr, "n_dead": int(logl_arr.size)}
    if n_live is not None:
        block["n_live"] = int(n_live)
    return block


def _json_safe_tinyns_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe_tinyns_value(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe_tinyns_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_tinyns_value(v) for v in value]
    try:
        import jax
        if isinstance(value, jax.Array):
            arr = np.asarray(value)
            return arr.item() if arr.shape == () else _json_safe_tinyns_value(arr)
    except Exception:
        pass
    return str(value)


def _mapping_from_call(result, name):
    if not hasattr(result, name):
        return {}
    try:
        value = getattr(result, name)()
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def normalize_tinyns_diagnostics(result) -> dict:
    """Collect TinyNS runtime diagnostics defensively into a JSON-safe dict."""
    data = {}
    sources = [_mapping_from_call(result, "diagnostics"), _mapping_from_call(result, "summary")]
    for src in sources:
        for key in _TINYNS_DIAGNOSTIC_FIELDS:
            if key in src and src[key] is not None:
                data[key] = _json_safe_tinyns_value(src[key])
    for key in _TINYNS_DIAGNOSTIC_FIELDS:
        if key not in data and hasattr(result, key):
            try:
                value = getattr(result, key)
            except Exception:
                continue
            if value is not None and not callable(value):
                data[key] = _json_safe_tinyns_value(value)

    def _num(key):
        try:
            value = data.get(key)
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    seconds = _num("seconds")
    ncall = _num("ncall")
    niter = _num("niter")
    if seconds and seconds > 0:
        if ncall is not None:
            data["ncall_per_sec"] = ncall / seconds
        if niter is not None:
            data["niter_per_sec"] = niter / seconds
    if ncall is not None and niter is not None:
        data["calls_per_iter"] = ncall / max(niter, 1.0)
    failures = _num("replacement_failures")
    if failures is not None and niter is not None:
        data["replacement_failure_rate"] = failures / max(niter, 1.0)
    return data


def print_tinyns_diagnostics(diag: dict) -> None:
    if not diag:
        return
    def g(*keys):
        for key in keys:
            if diag.get(key) is not None:
                return diag[key]
        return None
    def fmt(value):
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)
    lines = ["TinyNS Diagnostics", "------------------"]
    if g("success") is not None:
        lines.append(f"success: {str(g('success')).lower()}")
    if g("message") is not None:
        lines.append(f"message: {g('message')}")
    if g("logz") is not None:
        err = f" ± {fmt(g('logzerr'))}" if g("logzerr") is not None else ""
        lines.append(f"logZ: {fmt(g('logz'))}{err}")
    diagnostic_lines = (
        ("niter", "niter"),
        ("ncall", "ncall"),
        ("wall time", "seconds"),
        ("niter/sec", "niter_per_sec"),
        ("ncall/sec", "ncall_per_sec"),
        ("calls/iter", "calls_per_iter"),
        ("final dlogz", "final_delta_logz"),
        ("live weight fraction", "live_weight_fraction"),
        ("posterior ESS", "posterior_ess"),
        ("replacement mean batches", "replacement_mean_batches"),
        ("replacement max batches", "replacement_max_batches"),
        ("replacement failures", "replacement_failures"),
        ("replacement rescue used", "replacement_rescue_used"),
        ("insertion rank mean z", "insertion_rank_mean_z"),
        ("insertion rank std ratio", "insertion_rank_std_ratio"),
    )
    for label, key in diagnostic_lines:
        if g(key) is not None:
            suffix = " s" if key == "seconds" else ""
            lines.append(f"{label}: {fmt(g(key))}{suffix}")
    print("\n".join(lines), flush=True)

def dynesty_diagnostics_dir(opts):
    """Where ``--dynesty_diagnostics`` plots go: the PER-RUN directory.

    They used to go to ``<save_path>/dynesty_diagnostics/`` with a plot counter
    that restarts at 0 in every process, so two concurrent dynesty jobs sharing
    one ``--save_path`` (the normal SLURM-array pattern) silently overwrote each
    other's ``runplot_0001.pdf``/``traceplot_0001.pdf``.  ``opts.run_dir`` is
    set by both CLIs before sampling starts; library callers that do not set it
    keep the historical behaviour.
    """
    root = getattr(opts, "run_dir", None) or getattr(opts, "save_path", ".")
    return os.path.join(root, "dynesty_diagnostics")


def _nested_sampler_preflight(likelihood, prior_transform, ndims, opts, n_probe=32):
    """Probe the likelihood on a handful of prior draws before a nested sampler.

    dynesty and tinyns both seed their run by rejection-sampling the unit cube
    until they collect ``nlive`` points with a FINITE ``logL``.  When the
    likelihood is ``-inf`` across the whole prior — most often the selection
    variance guard firing at every draw (the GWTC-4.0/5.0 total-variance
    criterion) — that loop never terminates and the run hangs silently at the
    "find initial live points" step with nothing in the log to say why.

    This draws ``n_probe`` points from the prior (through ``prior_transform``,
    from an INDEPENDENT RNG seeded off ``opts.seed`` so the sampler's own RNG
    stream is untouched and ``--seed`` reproducibility is preserved), evaluates
    the likelihood on each, and:

    * always prints ``preflight: k/N prior draws have finite logL`` with the
      finite range and the elapsed wall time (its cost is ``n_probe``
      likelihood calls);
    * raises ``RuntimeError`` fail-fast when ``k == 0`` (the sampler would spin
      forever), naming the guard criterion and the remedies;
    * prints a loud, non-fatal warning when ``0 < k <= 3`` (initialization will
      be slow), with the same remedies.

    ``--sampler_preflight off`` bypasses the probe entirely.
    """
    import time

    if getattr(opts, "sampler_preflight", "on") == "off":
        return

    remedies = (
        "Remedies: --selection_neff_guard soft (finite penalized wall — the "
        "sampler initializes and is pushed toward the region satisfying the "
        "criterion); --max_likelihood_variance <cap> (accept a larger MC "
        "variance — the measured sigma^2 at your best-fit point must be below "
        "<cap>); python scripts/diagnose_selection_guard.py -- <your "
        "darksirens_inference args> (measure sigma^2_lnL on your data and "
        "report the smallest admitting cap). --sampler_preflight off skips "
        "this probe."
    )
    criterion = (
        "the selection variance guard (GWTC-4.0/5.0 total-log-likelihood "
        "variance: -inf when sigma^2_lnL = sum_i sigma_i^2 + N_obs^2/Neff_sel "
        "> max_likelihood_variance, plus the Vitale Neff > 5*N_obs floor)"
    )

    # Independent RNG: dynesty seeds its own np.random.default_rng(opts.seed)
    # and this must not perturb that stream (test_dynesty_seed pins it).
    seed = int(getattr(opts, "seed", 0))
    rng = np.random.default_rng(seed ^ 0xC0FFEE)

    t0 = time.perf_counter()
    finite_vals = []
    for _ in range(n_probe):
        u = rng.random(ndims)
        theta = prior_transform(jnp.asarray(u))
        val = float(np.asarray(likelihood(jnp.asarray(theta))))
        if np.isfinite(val):
            finite_vals.append(val)
    elapsed = time.perf_counter() - t0
    k = len(finite_vals)

    if k > 0:
        detail = f" (finite logL in [{min(finite_vals):.4g}, {max(finite_vals):.4g}])"
    else:
        detail = ""
    print(
        f"[*] preflight: {k}/{n_probe} prior draws have finite logL{detail} "
        f"[{elapsed:.2f} s]",
        flush=True,
    )

    if k == 0:
        raise RuntimeError(
            f"Nested-sampler preflight: the likelihood is -inf on ALL {n_probe} "
            "probed prior draws, so dynesty/tinyns would reject-sample forever "
            "without ever finding an initial live point. The most common cause "
            f"is {criterion}. {remedies}"
        )

    if k <= 3:
        nlive = int(getattr(opts, "nlive", 0) or 0)
        frac = k / n_probe
        est = f"~{int(np.ceil(nlive / frac)):,}" if nlive > 0 else "many"
        print(
            "  [!] preflight WARNING: only "
            f"{k}/{n_probe} prior draws are finite (finite fraction "
            f"{100.0 * frac:.1f}%). The sampler must reject-sample about {est} "
            f"draws to seed {nlive} live points, so initialization will be "
            f"SLOW. If it stalls, consider: {remedies}",
            flush=True,
        )


def run_sampler(method, likelihood, prior_transform, labels,
                lower_bound, upper_bound, opts, prior_kinds=None,
                joint_constraints=None):
    """
    method: "tinyns", "dynesty", or "numpyro"
    likelihood: function(coord) -> logL (expects 1D array)
    prior_transform: maps unit cube -> parameter space (expects 1D array)
    labels: list of parameter names
    lower_bound, upper_bound: arrays
    opts: argparse namespace
    joint_constraints: index-resolved joint prior constraints ALREADY applied by
        ``prior_transform``'s cube maps (nested samplers); the numpyro model
        builds independent per-parameter sites, so they are only reported there

    Returns a dict:
        {
            "samples": array of shape (Nsamp, ndim),
            "logZ": float or None,
            "logZerr": float or None
        }
    """

    ndims = len(labels)

    # --------------------------------------------------------
    # Zero free parameters
    # --------------------------------------------------------
    # Every block is fixed (e.g. --sky_model isotropic with
    # --fix_population --fix_cosmology, the null model of the sky
    # ladder).  The prior is then a point mass at the fixed point, so the
    # evidence is exact: Z = L(theta_fixed)  =>  logZ = logL.  Evaluate the
    # likelihood once and short-circuit BEFORE any sampler dispatch:
    # dynesty/tinyns cannot build a 0-dimensional proposal (dynesty
    # raises LAPACK "dsyevr: il=1" on the 0x0 bounding ellipsoid), and NUTS
    # would only burn warmup on an empty model.  This makes the ladder's
    # null baseline produce an exact logZ for the Bayes-factor comparison.
    if ndims == 0:
        log_l_fixed = float(np.asarray(likelihood(jnp.zeros(0))))
        print(
            "[*] 0 free parameters (all blocks fixed) - skipping nested "
            "sampling; evidence is exact at the fixed point.",
            flush=True,
        )
        print(f"    log Z = log L(fixed point) = {log_l_fixed:.6f}", flush=True)
        return {
            "samples": np.zeros((1, 0), dtype=float),   # one point, zero free dims
            "logZ": log_l_fixed,
            "logZerr": 0.0,                             # delta prior => no MC error
            "log_likelihood": np.array([log_l_fixed], dtype=float),
        }

    # --------------------------------------------------------
    # Nested-sampler preflight (dynesty & tinyns share the hard -inf wall)
    # --------------------------------------------------------
    # numpyro runs its own gradient preflight below; skip it here.  The zero-
    # free-params path above already returned, so ndims >= 1 by construction.
    #
    # Never on a RESUME: the probe exists because a FRESH nested run must
    # reject-sample the PRIOR until it collects nlive finite-logL points, and a
    # checkpoint already carries nlive live points with finite logL.  A
    # well-advanced posterior occupies a vanishing fraction of the prior, so
    # k == 0 out of n_probe prior draws is the EXPECTED outcome there and the
    # fail-fast would kill a requeued job the checkpoint would have finished --
    # while every remedy the message prints (--selection_neff_guard,
    # --max_likelihood_variance) is semantic and would then make
    # check_resume_fingerprint refuse the checkpoint.  It also saves n_probe
    # full likelihood evaluations on every requeue.
    if method in ("dynesty", "tinyns"):
        _resuming = bool(plan_from_opts(opts, method).resuming) or (
            method == "tinyns" and bool(getattr(opts, "tinyns_resume_from", None))
        )
        if _resuming:
            if getattr(opts, "sampler_preflight", "on") != "off":
                print(
                    "[*] preflight skipped: resuming from a checkpoint, whose "
                    "live points are already finite (the probe only guards a "
                    "fresh run's initial-live-point search).",
                    flush=True,
                )
        else:
            _nested_sampler_preflight(likelihood, prior_transform, ndims, opts)

    # --------------------------------------------------------
    # tinyns  (lightweight JAX nested sampler, dynesty-compatible)
    # --------------------------------------------------------
    if method == "tinyns":
        from tinyns import NestedSampler

        # tinyns is JAX-native, so the branch's JAX likelihood and prior
        # transform pass straight through (no host round-trip like dynesty
        # needs).  loglike takes a 1-D theta and returns a scalar logL;
        # prior_transform maps the unit cube to parameter space.
        def tinyns_loglike(theta):
            return likelihood(jnp.asarray(theta))

        def tinyns_ptform(u):
            return jnp.asarray(prior_transform(jnp.asarray(u)))

        config = build_tinyns_config(opts)
        sampler = NestedSampler(
            tinyns_loglike,
            tinyns_ptform,
            ndim=ndims,
            nlive=config.nlive,
            **tinyns_sampler_kwargs(config),
        )

        # SPLIT the seed key into independent run/resampling streams up
        # front.  The old flow handed `key` itself to sampler.run and later
        # split the SAME key for posterior resampling: split() is a pure
        # function, so whatever subkeys the sampler derived internally from
        # `key` overlapped the resampling stream (reused randomness between
        # the nested run and its equal-weight resample).  Deriving both from
        # the seed BEFORE the run keeps the resample key identical for a
        # fresh and a resumed run of the same configuration -- the
        # interrupted-vs-uninterrupted bit-reproducibility contract -- and
        # the resume fingerprint gate pins the seed, so an explicit resume
        # cannot silently swap the lineage either.
        run_key, resample_key = jax.random.split(jax.random.PRNGKey(config.seed))
        run_kwargs = tinyns_run_kwargs(config)

        # --max_samples reaches tinyns as ``maxiter`` (ITERATIONS) and dynesty as
        # ``maxcall`` (likelihood CALLS): tinyns exposes no call cap, and one
        # rwalk iteration costs walks x max_active_chains evaluations (5 for the
        # 'recommended' preset, 1280 for heavy_darksirens).  The two budgets are
        # therefore NOT comparable, so print the resolved cap and its call
        # equivalent rather than leaving the reader to infer it from the flag.
        _maxiter = run_kwargs.get("maxiter")
        if _maxiter is None:
            print("[*] tinyns iteration cap: none (runs to the dlogz "
                  "criterion)", flush=True)
        else:
            _max_active = (max(config.replacement_chain_schedule)
                           if config.replacement_chain_schedule
                           else int(config.replacement_chains))
            print(
                f"[*] tinyns iteration cap: maxiter={int(_maxiter):,} "
                f"(--max_samples), i.e. up to ~{int(_maxiter) * int(config.walks) * _max_active:,} "
                f"likelihood calls at walks={config.walks} x "
                f"{_max_active} chain(s)",
                flush=True,
            )

        # Checkpoint/resume policy: the shared --checkpoint_interval / --resume
        # flags place a checkpoint in the run directory, while the legacy
        # explicit --tinyns_checkpoint_path / --tinyns_resume_from still win so
        # existing scripts behave exactly as before.  tinyns' own cadence is in
        # ITERATIONS (config.checkpoint_interval, default 100) -- the seconds in
        # --checkpoint_interval only decide whether checkpointing is on.
        plan = plan_from_opts(opts, "tinyns")
        checkpoint_path = config.checkpoint_path or (plan.path if plan.enabled else None)
        resume_from = config.resume_from or plan.resume_from
        checkpoint_path_out = config.checkpoint_path_out
        if (
            resume_from
            and checkpoint_path_out is None
            and checkpoint_path
            and os.path.abspath(checkpoint_path) != os.path.abspath(resume_from)
        ):
            # Resuming a checkpoint that is not the file we are told to write:
            # be explicit about the output so tinyns does not overwrite the
            # source checkpoint (its default is checkpoint_path_out=source).
            checkpoint_path_out = checkpoint_path

        if resume_from:
            print(
                f"[*] tinyns resuming from checkpoint {resume_from} "
                f"(writing {checkpoint_path_out or resume_from} every "
                f"{config.checkpoint_interval} iterations)",
                flush=True,
            )
            result = sampler.resume(
                resume_from,
                **run_kwargs,
                checkpoint_path_out=checkpoint_path_out,
            )
        else:
            if checkpoint_path:
                print(
                    f"[*] tinyns checkpointing to {checkpoint_path} every "
                    f"{config.checkpoint_interval} iterations "
                    "(resume with --resume auto)",
                    flush=True,
                )
            result = sampler.run(
                run_key,
                **run_kwargs,
                checkpoint_path=checkpoint_path,
            )

        # Equal-weight posterior samples (tinyns mirrors dynesty's resampler),
        # from the dedicated resample stream split above.
        samples = np.asarray(result.resample_equal(resample_key))

        out = {
            "samples": samples,
            "logZ": float(result.logz),
            "logZerr": float(result.logzerr),
        }
        # tinyns' result.logl/.logwt are the weighted dead-point record (dead
        # points followed by the final live points); `samples` above is the
        # equal-weight resample, whose length is the posterior ESS instead.
        out["dead_points"] = _dead_point_block(
            getattr(result, "logl", None),
            getattr(result, "logwt", None),
            n_live=getattr(result, "nlive", None),
        )
        if hasattr(result, "diagnostics"):
            try:
                out["tinyns_diagnostics"] = result.diagnostics()
            except Exception:
                pass
        if hasattr(result, "summary"):
            try:
                out["tinyns_summary"] = result.summary()
            except Exception:
                pass
        out["tinyns_runtime_diagnostics"] = normalize_tinyns_diagnostics(result)
        print_tinyns_diagnostics(out["tinyns_runtime_diagnostics"])
        return out


    # --------------------------------------------------------
    # NumPyro / NUTS
    # --------------------------------------------------------
    elif method == "numpyro":
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS
        from numpyro.infer.initialization import init_to_value

        lower = jnp.asarray(lower_bound, dtype=jnp.result_type(float))
        upper = jnp.asarray(upper_bound, dtype=jnp.result_type(float))
        midpoint = 0.5 * (lower + upper)

        if not np.all(np.isfinite(np.asarray(lower))) or not np.all(
            np.isfinite(np.asarray(upper))
        ):
            raise ValueError(
                "NumPyro sampler requires finite lower and upper prior bounds."
            )
        # Truncated Beta (mixture-stick) bounds are supported by the nested
        # samplers' closed-form PPF but not by numpyro's Beta site; validate
        # EAGERLY here (concrete numpy values) rather than inside the traced
        # model, so the error is clear and tracing never sees a float() cast.
        if prior_kinds is not None:
            _lo_np = np.asarray(lower_bound, dtype=float)
            _hi_np = np.asarray(upper_bound, dtype=float)
            for _i, (_kind, _kloc, _kscale) in enumerate(prior_kinds):
                if _kind == "beta" and (_lo_np[_i] != 0.0 or _hi_np[_i] != 1.0):
                    raise ValueError(
                        f"Truncated Beta prior bounds [{_lo_np[_i]}, {_hi_np[_i]}] "
                        f"for '{labels[_i]}' are not supported by the numpyro "
                        "sampler; use dynesty/tinyns or keep the default [0, 1] "
                        "bounds."
                    )
        if not np.all(np.asarray(upper) > np.asarray(lower)):
            raise ValueError(
                "NumPyro sampler requires every prior upper bound to exceed its "
                "lower bound."
            )

        # JOINT prior constraints (dipole unit ball, mixture simplex,
        # ordered_le) are reparameterized away by make_prior_transform's cube
        # maps, so the nested samplers propose only inside the constrained
        # region.  The sites below are INDEPENDENT per parameter, so here the
        # same region is carved out by likelihood-side rejection instead: the
        # posterior is the same measure (uniform on the constrained set either
        # way), but NUTS must integrate against a hard -inf wall whose gradient
        # is NaN, so divergences become structural rather than diagnostic.  Say
        # so instead of leaving the operator to read it out of the divergence
        # fraction.
        if joint_constraints:
            print(
                "  [!] joint prior constraints "
                + ", ".join(f"{kind}{tuple(idx)}"
                            for kind, idx in joint_constraints)
                + " are enforced for NUTS by likelihood-side REJECTION (the "
                "nested samplers reparameterize them into the unit cube "
                "instead). The posterior is the same truncated measure, but "
                "the -inf boundary has no gradient: expect a structurally "
                "elevated divergence fraction and reduced ESS. Prefer "
                "--sampler dynesty/tinyns for "
                + ", ".join(sorted({kind for kind, _ in joint_constraints}))
                + " models.",
                flush=True,
            )

        def _site(i, name):
            # Per-parameter prior, matching make_prior_transform's PER-DIMENSION
            # measure so nested and NUTS infer the same posterior.  Any JOINT
            # constraint is NOT reproduced here (see the note above): it is left
            # to likelihood-side rejection.  "normal" gives whitened GP latents
            # the unit-scale geometry NUTS needs (Option A); low/high act as
            # truncation bounds for every kind.
            kind, kloc, kscale = ("uniform", None, None)
            if prior_kinds is not None:
                kind, kloc, kscale = prior_kinds[i]
            if kind == "normal":
                loc = 0.0 if kloc is None else float(kloc)
                sc = 1.0 if kscale is None else float(kscale)
                return numpyro.sample(name, dist.TruncatedNormal(
                    loc=loc, scale=sc, low=lower[i], high=upper[i]))
            if kind == "lognormal":
                loc = 0.0 if kloc is None else float(kloc)
                sc = 1.0 if kscale is None else float(kscale)
                base = dist.TruncatedNormal(
                    loc=loc, scale=sc,
                    low=jnp.log(lower[i]), high=jnp.log(upper[i]))
                return numpyro.sample(name, dist.TransformedDistribution(
                    base, dist.transforms.ExpTransform()))
            if kind == "beta":
                # Mixture-weight stick fcat_m ~ Beta(1, b); b in the scale slot.
                # Matches make_prior_transform's closed-form Beta(1,b) PPF so NUTS
                # and the nested samplers infer the same posterior.  The nested
                # transform supports truncated bounds; numpyro's Beta does not,
                # so reject non-default bounds rather than silently sampling a
                # different prior than dynesty/tinyns would.
                return numpyro.sample(name, dist.Beta(1.0, float(kscale or 1.0)))
            return numpyro.sample(name, dist.Uniform(low=lower[i], high=upper[i]))

        def model():
            # Independent priors per parameter (uniform unless a parameter
            # declares otherwise via prior_kinds).  NumPyro maps to an
            # unconstrained space internally so NUTS samples there.
            theta_parts = [_site(i, name) for i, name in enumerate(labels)]
            theta = jnp.stack(theta_parts) if theta_parts else jnp.array([])
            log_l = likelihood(theta)
            numpyro.deterministic("log_likelihood", log_l)
            numpyro.factor("likelihood", log_l)

        init_values = {name: midpoint[i] for i, name in enumerate(labels)}
        nuts_init_tries = int(getattr(opts, "nuts_init_tries", 32))
        nuts_init_seed_offset = int(getattr(opts, "nuts_init_seed_offset", 100_000))
        target_accept = float(getattr(opts, "nuts_target_accept", 0.8))
        max_tree_depth = int(getattr(opts, "nuts_max_tree_depth", 10))
        num_warmup = int(getattr(opts, "nuts_warmup", 500))
        num_samples = int(getattr(opts, "nuts_samples", 1000))
        num_chains = int(getattr(opts, "nuts_chains", 1))
        chain_method = getattr(opts, "nuts_chain_method", "sequential")

        if num_warmup < 0 or num_samples <= 0 or num_chains <= 0 or nuts_init_tries <= 0:
            raise ValueError(
                "NumPyro requires nuts_warmup >= 0, nuts_samples > 0, "
                "nuts_chains > 0, and nuts_init_tries > 0."
            )
        midpoint_log_l = float(np.asarray(likelihood(midpoint)))
        if not np.isfinite(midpoint_log_l):
            rng = np.random.default_rng(int(opts.seed) + nuts_init_seed_offset)
            lower_np = np.asarray(lower, dtype=float)
            upper_np = np.asarray(upper, dtype=float)
            best_theta = None
            best_log_l = -np.inf
            for _ in range(nuts_init_tries):
                candidate = rng.uniform(lower_np, upper_np)
                candidate_log_l = float(
                    np.asarray(likelihood(jnp.asarray(candidate, dtype=lower.dtype)))
                )
                if np.isfinite(candidate_log_l) and candidate_log_l > best_log_l:
                    best_log_l = candidate_log_l
                    best_theta = candidate
            if best_theta is None:
                raise RuntimeError(
                    "Failed to find a finite NumPyro NUTS initial point after "
                    f"{nuts_init_tries} attempts. "
                    f"parameter_names={list(labels)}, "
                    f"bounds_min={lower_np.tolist()}, bounds_max={upper_np.tolist()}. "
                    "Hint: run a likelihood dry-run diagnostic across prior bounds "
                    "to identify non-finite regions."
                )
            init_values = {name: best_theta[i] for i, name in enumerate(labels)}

        theta0 = jnp.asarray([init_values[name] for name in labels], dtype=lower.dtype)
        log_l0 = likelihood(theta0)

        def _likelihood_scalar(theta):
            return jnp.asarray(likelihood(theta), dtype=lower.dtype)

        grad_l0 = jax.grad(_likelihood_scalar)(theta0)
        grad_np = np.asarray(grad_l0, dtype=float)
        grad_isfinite = np.isfinite(grad_np)
        grad_isnan = np.isnan(grad_np)
        grad_isinf = np.isinf(grad_np)
        log_l0_finite = bool(np.asarray(jnp.isfinite(log_l0)))
        grad_l0_finite = bool(np.asarray(jnp.all(jnp.isfinite(grad_l0))))
        if not (log_l0_finite and grad_l0_finite):
            lower_np = np.asarray(lower, dtype=float)
            upper_np = np.asarray(upper, dtype=float)
            theta0_np = np.asarray(theta0, dtype=float)
            widths = upper_np - lower_np
            near_boundary = np.minimum(theta0_np - lower_np, upper_np - theta0_np) <= (
                1e-6 * np.maximum(1.0, widths)
            )
            bad_grad_params = [
                name for i, name in enumerate(labels) if not bool(grad_isfinite[i])
            ]
            print(
                "NumPyro NUTS preflight failure at initial point "
                f"(finite_logL={log_l0_finite}, finite_grad={grad_l0_finite}):",
                flush=True,
            )
            print(
                f"  bad_grad_params={bad_grad_params} "
                f"({len(bad_grad_params)}/{len(labels)})",
                flush=True,
            )
            for i, name in enumerate(labels):
                print(
                    "  "
                    f"{name}: value={theta0_np[i]:.12g}, "
                    f"bounds=({lower_np[i]:.12g}, {upper_np[i]:.12g}), "
                    f"near_boundary={bool(near_boundary[i])}, "
                    f"grad={grad_np[i]:.6g}, "
                    f"grad_finite={bool(grad_isfinite[i])}, "
                    f"grad_nan={bool(grad_isnan[i])}, "
                    f"grad_inf={bool(grad_isinf[i])}",
                    flush=True,
                )
            raise RuntimeError(
                "NumPyro preflight check failed at initial point: "
                f"finite_log_likelihood={log_l0_finite}, "
                f"finite_log_likelihood_gradient={grad_l0_finite}. "
                "Review the per-parameter log output (with gradients reported for "
                "each parameter) and adjust parameter bounds or initialization."
            )

        kernel = NUTS(
            model,
            target_accept_prob=target_accept,
            max_tree_depth=max_tree_depth,
            init_strategy=init_to_value(values=init_values),
        )
        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=opts.show_progress,
        )
        key = jax.random.PRNGKey(opts.seed)
        print(
            "Starting NumPyro NUTS run: "
            f"warmup={num_warmup}, samples={num_samples}, chains={num_chains}",
            flush=True,
        )
        mcmc.run(key, extra_fields=("diverging", "num_steps", "accept_prob"))
        posterior = mcmc.get_samples(group_by_chain=False)
        samples = (
            jnp.column_stack([posterior[name] for name in labels])
            if labels
            else jnp.zeros((num_samples * num_chains, 0))
        )
        log_likelihood = posterior.get("log_likelihood")

        # Sampler-health diagnostics. Divergences mean the leapfrog integrator
        # hit numerically unstable curvature (or a -inf likelihood cliff inside
        # the prior box); their post-warmup fraction is THE headline NUTS
        # quality number and must be visible in the log and the saved results.
        extra = mcmc.get_extra_fields(group_by_chain=False)
        diverging = np.asarray(extra.get("diverging", np.zeros(0, dtype=bool)))
        num_steps = np.asarray(extra.get("num_steps", np.zeros(0, dtype=int)))
        accept = np.asarray(extra.get("accept_prob", np.zeros(0, dtype=float)))
        max_steps = int(2 ** max_tree_depth - 1)
        numpyro_diagnostics = {
            "n_divergent": int(diverging.sum()),
            "divergence_fraction": float(diverging.mean()) if diverging.size else 0.0,
            "mean_accept_prob": float(accept.mean()) if accept.size else float("nan"),
            "mean_num_steps": float(num_steps.mean()) if num_steps.size else float("nan"),
            "frac_at_max_tree_depth": (
                float((num_steps >= max_steps).mean()) if num_steps.size else 0.0
            ),
            "max_tree_depth": max_tree_depth,
            "target_accept": target_accept,
        }
        print(
            "NumPyro NUTS diagnostics: "
            f"divergences {numpyro_diagnostics['n_divergent']}"
            f"/{diverging.size} "
            f"({100.0 * numpyro_diagnostics['divergence_fraction']:.1f}%), "
            f"mean accept {numpyro_diagnostics['mean_accept_prob']:.3f}, "
            f"mean leapfrog steps {numpyro_diagnostics['mean_num_steps']:.1f}, "
            f"at max tree depth {100.0 * numpyro_diagnostics['frac_at_max_tree_depth']:.1f}%",
            flush=True,
        )

        return {
            "samples": np.asarray(samples),
            "logZ": None,
            "logZerr": None,
            "log_likelihood": (
                None if log_likelihood is None else np.asarray(log_likelihood)
            ),
            "numpyro_diagnostics": numpyro_diagnostics,
        }

    # --------------------------------------------------------
    # dynesty
    # --------------------------------------------------------

    elif method == "dynesty":
        from dynesty import NestedSampler
        from dynesty.utils import resample_equal
        import dynesty.plotting as dyplot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import threading

        # Tracker variables to peek inside Dynesty
        eval_count = 0
        valid_count = 0

        # 3. WRAPPER: Strip JAX DeviceArrays and track evaluations
        def dynesty_loglike(theta):
            nonlocal eval_count, valid_count
            eval_count += 1

            val = float(np.asarray(likelihood(jnp.asarray(theta))))

            if np.isfinite(val):
                valid_count += 1

            # Print an update every 500 evaluations, stopping after 5000
            if eval_count % 500 == 0 and eval_count <= 5000:
                print(f"  ... [Dynesty Setup] Likelihood calls: {eval_count} | Valid points found: {valid_count}", flush=True)

            return val

        def dynesty_ptform(u):
            return np.asarray(prior_transform(jnp.asarray(u)))

        maxcall = getattr(opts, "max_samples", None)
        if maxcall is not None and maxcall <= 0:
            maxcall = None

        diag_dir = dynesty_diagnostics_dir(opts)
        enable_diag = bool(getattr(opts, "dynesty_diagnostics", False))
        diag_interval = 600  # 10 minutes in seconds
        _diag_index = [0]
        _stop_diag = threading.Event()

        def _write_dynesty_diagnostics(sampler_ref):
            # ``sampler_ref.results`` assembles several parallel arrays from the
            # LIVE run while the main thread is appending to saved_run, so a
            # concurrent read can see mismatched lengths and raise (dynesty 2.1
            # checks that itself).  Unguarded, one such read killed this daemon
            # thread permanently and silently -- no diagnostics for the rest of a
            # multi-day run, which is exactly what --dynesty_diagnostics exists
            # to avoid.  Skip the sample and try again next interval.
            try:
                res = sampler_ref.results
                n_samples = len(res.samples)
            except Exception as e:
                print(f"[dynesty diag] skipped an inconsistent concurrent read "
                      f"of sampler.results: {e}", flush=True)
                return
            if n_samples < 2:
                return
            _diag_index[0] += 1
            idx = _diag_index[0]
            os.makedirs(diag_dir, exist_ok=True)
            out_dir = diag_dir
            try:
                fig, _ = dyplot.runplot(res, label_kwargs={"fontsize": 10})
                fig.savefig(os.path.join(out_dir, f"runplot_{idx:04d}.pdf"), bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[dynesty diag] runplot failed: {e}", flush=True)
            try:
                fig, _ = dyplot.traceplot(res, labels=labels,
                                          label_kwargs={"fontsize": 8},
                                          title_kwargs={"fontsize": 8})
                fig.savefig(os.path.join(out_dir, f"traceplot_{idx:04d}.pdf"), bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"[dynesty diag] traceplot failed: {e}", flush=True)
            print(f"[dynesty diag] wrote diagnostics #{idx} to {out_dir}", flush=True)

        # ---- checkpoint / resume -------------------------------------------
        # Without this a SLURM timeout, preemption or OOM after sampling
        # started discarded the entire run (the failure mode that halted the
        # July 2026 lensing campaign).  See darksirens/inference/checkpointing.py
        # for why the checkpoint cannot simply pickle the JAX closures.
        plan = plan_from_opts(opts, "dynesty")
        if plan.resuming:
            print(
                f"[*] Restoring dynesty sampler from checkpoint {plan.resume_from}",
                flush=True,
            )
            sampler = restore_dynesty_sampler(
                plan.resume_from, dynesty_loglike, dynesty_ptform
            )
            if int(getattr(sampler, "ndim", ndims)) != ndims:
                raise RuntimeError(
                    f"Checkpoint {plan.resume_from} has ndim="
                    f"{sampler.ndim} but this run has {ndims} free parameters; "
                    "the configurations do not match."
                )
            if int(getattr(sampler, "nlive", opts.nlive)) != int(opts.nlive):
                print(
                    f"  [!] checkpoint nlive={sampler.nlive} differs from "
                    f"--nlive {opts.nlive}; the checkpoint's value wins.",
                    flush=True,
                )
            # Seed contract: the checkpoint carries the Generator that --seed
            # created, so the resumed run continues that exact stream and never
            # draws fresh entropy.  Do NOT re-seed here.
            dynesty_rstate = sampler.rstate
            print(
                f"[*] Resumed at iteration {getattr(sampler, 'it', 0)} "
                f"({getattr(sampler, 'ncall', 0)} likelihood calls already spent).",
                flush=True,
            )
        else:
            print(f"[*] Asking Dynesty to find {opts.nlive} initial live points. This may take a minute...", flush=True)
            # Seeded RNG: without an explicit rstate dynesty draws fresh global
            # NumPy entropy, so --seed had NO effect on dynesty runs while being
            # recorded as the run seed in results.hdf5 (library review, CLI
            # finding 2). tinyns and numpyro already honour opts.seed.
            dynesty_rstate = np.random.default_rng(int(opts.seed))
            sampler = NestedSampler(
                dynesty_loglike,
                dynesty_ptform,
                ndims,
                bound="multi",
                sample="rwalk",
                nlive=opts.nlive,
                rstate=dynesty_rstate,
            )

        checkpoint_kwargs = {}
        if plan.enabled:
            install_dynesty_checkpointing(sampler)
            checkpoint_kwargs = dict(
                checkpoint_file=plan.path,
                checkpoint_every=plan.interval_seconds,
            )
            print(
                f"[*] Checkpointing to {plan.path} every "
                f"{plan.interval_seconds:g} s (resume with --resume auto).",
                flush=True,
            )
        elif plan.resuming:
            print(
                "  [!] --checkpoint_interval is off, so this resumed run is "
                "itself unprotected: another kill loses everything since the "
                "restored checkpoint.",
                flush=True,
            )

        if enable_diag:
            def _diag_thread_fn():
                # Wait one full interval before the first plot so dynesty has real samples.
                _stop_diag.wait(timeout=diag_interval)
                while not _stop_diag.is_set():
                    # Belt and braces around the guarded writer: nothing raised
                    # from a diagnostics plot may terminate this thread (or the
                    # remaining days of a run lose their diagnostics silently).
                    try:
                        _write_dynesty_diagnostics(sampler)
                    except Exception as e:
                        print(f"[dynesty diag] diagnostics pass failed: {e}",
                              flush=True)
                    _stop_diag.wait(timeout=diag_interval)

            diag_thread = threading.Thread(target=_diag_thread_fn, daemon=True)
            diag_thread.start()
            print(f"[*] Diagnostic plots enabled — writing to {diag_dir}/ every 10 min.", flush=True)

        if not plan.resuming:
            print("[*] Initial live points found! Starting main nested sampling loop...", flush=True)
        if maxcall is not None:
            print(f"[*] Dynesty call cap: maxcall={maxcall}", flush=True)
        try:
            sampler.run_nested(
                dlogz=opts.dlogz,
                maxcall=maxcall,
                print_progress=opts.show_progress,
                resume=plan.resuming,
                **checkpoint_kwargs,
            )
        finally:
            if enable_diag:
                _stop_diag.set()
                diag_thread.join(timeout=120)
        res = sampler.results

        # Weighted posterior samples
        logw = np.asarray(res["logwt"], dtype=float)
        finite_logw = logw[np.isfinite(logw)]
        if finite_logw.size == 0:
            raise RuntimeError("dynesty returned no finite posterior weights.")
        
        weights = np.exp(logw - np.max(finite_logw))
        weight_sum = float(np.sum(weights))
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            raise RuntimeError("dynesty posterior weights could not be normalized.")
        weights /= weight_sum
        samples = resample_equal(res.samples, weights, rstate=dynesty_rstate)

        logZ = float(res.logz[-1])
        logZerr = float(res.logzerr[-1])

        # dynesty's res.logl/res.logwt are per DEAD POINT (niter entries, the
        # final live points already appended).  resample_equal above returns as
        # many equal-weight rows as it was given, so n_samples happens to equal
        # n_dead here -- the rows are still unrelated point sets.  "nlive" is
        # absent from a dynamic run's Results (it carries samples_n instead).
        dead_points = _dead_point_block(
            res["logl"],
            logw,
            n_live=(int(res["nlive"]) if "nlive" in res else None),
        )

        return {
            "samples": np.asarray(samples),
            "logZ": logZ,
            "logZerr": logZerr,
            "dead_points": dead_points,
            # The live-point count the sampler ACTUALLY ran with: a resumed
            # run keeps the checkpoint's nlive over --nlive (warned above), so
            # results.hdf5 must not record the request as if it were the state.
            "nlive_actual": int(getattr(sampler, "nlive", opts.nlive)),
        }

    else:
        raise ValueError(f"Unknown sampler: {method}")
