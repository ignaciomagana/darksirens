import numpy as np
import jax
import jax.numpy as jnp

def run_sampler(method, likelihood, prior_transform, labels,
                lower_bound, upper_bound, opts, prior_kinds=None):
    """
    method: "tinyns", "dynesty", or "numpyro"
    likelihood: function(coord) -> logL (expects 1D array)
    prior_transform: maps unit cube -> parameter space (expects 1D array)
    labels: list of parameter names
    lower_bound, upper_bound: arrays
    opts: argparse namespace

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
    # --fix_population --fixed_cosmology, the null model of the sky
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

        # dynesty's call cap (--max_samples) doubles as tinyns' iteration
        # cap; one nested-sampling iteration retires ~one live point, so the
        # budget is comparable.  0/None means "run to the dlogz criterion".
        maxiter = getattr(opts, "max_samples", None)
        if maxiter is not None and maxiter <= 0:
            maxiter = None

        sampler = NestedSampler(
            tinyns_loglike,
            tinyns_ptform,
            ndim=ndims,
            nlive=int(getattr(opts, "nlive", 500)),
            sample=getattr(opts, "tinyns_sample", "slice"),
            slices=int(getattr(opts, "tinyns_slices", 5)),
            slice_steps=int(getattr(opts, "tinyns_slice_steps", 10)),
            step_scale=float(getattr(opts, "tinyns_step_scale", 0.1)),
        )

        key = jax.random.PRNGKey(int(getattr(opts, "seed", 0)))
        result = sampler.run(
            key,
            dlogz=float(getattr(opts, "dlogz", 0.1)),
            maxiter=maxiter,
            progress=bool(getattr(opts, "show_progress", True)),
            progress_interval=int(getattr(opts, "tinyns_progress_interval", 100)),
        )

        # Equal-weight posterior samples (tinyns mirrors dynesty's resampler).
        key, resample_key = jax.random.split(key)
        samples = np.asarray(result.resample_equal(resample_key))

        return {
            "samples": samples,
            "logZ": float(result.logz),
            "logZerr": float(result.logzerr),
        }


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
        if not np.all(np.asarray(upper) > np.asarray(lower)):
            raise ValueError(
                "NumPyro sampler requires every prior upper bound to exceed its "
                "lower bound."
            )

        def _site(i, name):
            # Per-parameter prior, matching make_prior_transform's measure so
            # nested and NUTS infer the same posterior.  "normal" gives whitened
            # GP latents the unit-scale geometry NUTS needs (Option A); low/high
            # act as truncation bounds for every kind.
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
        mcmc.run(key)
        posterior = mcmc.get_samples(group_by_chain=False)
        samples = (
            jnp.column_stack([posterior[name] for name in labels])
            if labels
            else jnp.zeros((num_samples * num_chains, 0))
        )
        log_likelihood = posterior.get("log_likelihood")

        return {
            "samples": np.asarray(samples),
            "logZ": None,
            "logZerr": None,
            "log_likelihood": (
                None if log_likelihood is None else np.asarray(log_likelihood)
            ),
        }

    # --------------------------------------------------------
    # dynesty
    # --------------------------------------------------------

    elif method == "dynesty":
        import time
        import os
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

        save_path = getattr(opts, "save_path", ".")
        enable_diag = bool(getattr(opts, "dynesty_diagnostics", False))
        diag_interval = 600  # 10 minutes in seconds
        _diag_index = [0]
        _stop_diag = threading.Event()

        def _write_dynesty_diagnostics(sampler_ref):
            res = sampler_ref.results
            if len(res.samples) < 2:
                return
            _diag_index[0] += 1
            idx = _diag_index[0]
            out_dir = os.path.join(save_path, "dynesty_diagnostics")
            os.makedirs(out_dir, exist_ok=True)
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

        print(f"[*] Asking Dynesty to find {opts.nlive} initial live points. This may take a minute...", flush=True)
        sampler = NestedSampler(
            dynesty_loglike,
            dynesty_ptform,
            ndims,
            bound="multi",
            sample="rwalk",
            nlive=opts.nlive
        )

        if enable_diag:
            def _diag_thread_fn():
                # Wait one full interval before the first plot so dynesty has real samples.
                _stop_diag.wait(timeout=diag_interval)
                while not _stop_diag.is_set():
                    _write_dynesty_diagnostics(sampler)
                    _stop_diag.wait(timeout=diag_interval)

            diag_thread = threading.Thread(target=_diag_thread_fn, daemon=True)
            diag_thread.start()
            print(f"[*] Diagnostic plots enabled — writing to {save_path}/dynesty_diagnostics/ every 10 min.", flush=True)

        print(f"[*] Initial live points found! Starting main nested sampling loop...", flush=True)
        if maxcall is not None:
            print(f"[*] Dynesty call cap: maxcall={maxcall}", flush=True)
        try:
            sampler.run_nested(
                dlogz=opts.dlogz,
                maxcall=maxcall,
                print_progress=opts.show_progress,
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
        samples = resample_equal(res.samples, weights)

        logZ = float(res.logz[-1])
        logZerr = float(res.logzerr[-1])

        return {
            "samples": np.asarray(samples),
            "logZ": logZ,
            "logZerr": logZerr
        }

    else:
        raise ValueError(f"Unknown sampler: {method}")
