import numpy as np
import jax
import jax.numpy as jnp

def run_sampler(method, likelihood, prior_transform, labels,
                lower_bound, upper_bound, opts):
    """
    method: "jaxns", "dynesty", "emcee", or "numpyro"
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
    # JAXNS
    # --------------------------------------------------------
    if method == "jaxns":
        import tensorflow_probability.substrates.jax as tfp
        tfpd = tfp.distributions
        from jaxns import NestedSampler
        from jaxns.framework.model import Model
        from jaxns.framework.prior import Prior

        # Prior model: returns a vector theta of shape (ndim,)
        def prior_model():
            params = []
            for i, name in enumerate(labels):
                low = float(lower_bound[i])
                high = float(upper_bound[i])
                x = yield Prior(tfpd.Uniform(low=low, high=high), name=name)
                params.append(x)
            return jnp.stack(params)

        def log_likelihood(theta):
            return likelihood(jnp.asarray(theta))

        model = Model(
            prior_model=prior_model,
            log_likelihood=log_likelihood,
        )

        ns = NestedSampler(
            model=model,
            num_live_points=opts.nlive,
            max_samples=opts.max_samples,
            verbose=opts.show_progress,
        )

        key = jax.random.PRNGKey(opts.seed)
        term, state = ns(key)
        results = ns.to_results(term, state)

        posterior = results.samples  # dict of arrays
        samples = jnp.column_stack([posterior[name] for name in labels])

        return {
            "samples": np.asarray(samples),
            "logZ": None,        # JAXNS evidence not extracted here
            "logZerr": None
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

        def model():
            # Use the same independent uniform priors as the nested samplers and
            # emcee boundary checks.  NumPyro handles the constrained-to-real
            # transform internally so NUTS samples in an unconstrained space.
            theta_parts = []
            for i, name in enumerate(labels):
                theta_parts.append(
                    numpyro.sample(
                        name,
                        dist.Uniform(low=lower[i], high=upper[i]),
                    )
                )
            theta = jnp.stack(theta_parts) if theta_parts else jnp.array([])
            log_l = likelihood(theta)
            numpyro.deterministic("log_likelihood", log_l)
            numpyro.factor("likelihood", log_l)

        init_values = {name: midpoint[i] for i, name in enumerate(labels)}
        target_accept = float(getattr(opts, "nuts_target_accept", 0.8))
        max_tree_depth = int(getattr(opts, "nuts_max_tree_depth", 10))
        num_warmup = int(getattr(opts, "nuts_warmup", 500))
        num_samples = int(
            getattr(opts, "nuts_samples", getattr(opts, "nsteps", 1000))
        )
        num_chains = int(getattr(opts, "nuts_chains", 1))
        chain_method = getattr(opts, "nuts_chain_method", "sequential")

        if num_warmup < 0 or num_samples <= 0 or num_chains <= 0:
            raise ValueError(
                "NumPyro requires nuts_warmup >= 0, nuts_samples > 0, "
                "and nuts_chains > 0."
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
        from dynesty import NestedSampler
        from dynesty.utils import resample_equal
        
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

        print(f"[*] Asking Dynesty to find {opts.nlive} initial live points. This may take a minute...", flush=True)
        sampler = NestedSampler(
            dynesty_loglike, 
            dynesty_ptform,  
            ndims,
            bound="multi", 
            sample="rwalk",
            nlive=opts.nlive
        )
        
        print(f"[*] Initial live points found! Starting main nested sampling loop...", flush=True)
        if maxcall is not None:
            print(f"[*] Dynesty call cap: maxcall={maxcall}", flush=True)
        sampler.run_nested(
            dlogz=opts.dlogz,
            maxcall=maxcall,
            print_progress=opts.show_progress,
        )
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
    # --------------------------------------------------------
    # emcee
    # --------------------------------------------------------
    elif method == "emcee":
        import emcee
        import h5py
        import os
        import time
        from pathlib import Path

        # JIT the likelihood for fast single-point evaluation and batch it when possible.
        batched_likelihood = jax.vmap(likelihood)

        # --- NEW: Define a safe batch size to prevent GPU OOM ---
        # 8 is usually a safe sweet spot. If it still crashes, drop to 4 or 2.
        # If your GPU has lots of memory (e.g., 40GB A100), you can push it to 16.
        BATCH_SIZE = 1

        def batched_log_prob(coords):
            coords = np.asarray(coords)

            # emcee calls the log-probability on a single walker at a time unless vectorization is enabled.
            if coords.ndim == 1:
                if np.any((coords < lower_bound) | (coords > upper_bound)):
                    return -np.inf
                return float(np.asarray(likelihood(coords)))
            if coords.ndim != 2:
                raise ValueError(f"Expected emcee coordinates with ndim 1 or 2, got shape {coords.shape}.")

            # 1. Find which walkers are out of bounds (boolean mask)
            out_of_bounds = np.any((coords < lower_bound) | (coords > upper_bound), axis=1)
            
            # 2. Evaluate likelihood in chunks to save GPU memory
            logl_list = []
            for i in range(0, len(coords), BATCH_SIZE):
                batch_coords = coords[i : i + BATCH_SIZE]
                batch_logl = batched_likelihood(batch_coords)
                logl_list.append(np.asarray(batch_logl))
            
            logl = np.concatenate(logl_list)
            
            # 3. Apply -inf to the out-of-bounds walkers
            logl[out_of_bounds] = -np.inf
            return logl

        p0 = np.random.uniform(lower_bound, upper_bound,
                               size=(opts.nwalkers, ndims))

        # Set up checkpointing via emcee backend
        checkpoint_dir = Path(opts.save_path) / "emcee_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Auto-isolate checkpoint files per run so concurrent jobs do not contend for one HDF5 lock.
        job_tag = os.environ.get("SLURM_JOB_ID")
        if not job_tag:
            job_tag = f"{int(time.time())}_{os.getpid()}"
        backend_filename = checkpoint_dir / f"chain_{job_tag}.h5"
        
        backend = emcee.backends.HDFBackend(str(backend_filename))
        backend.reset(opts.nwalkers, ndims)

        def finalize_backend() -> None:
            sync = getattr(backend, "sync", None)
            if callable(sync):
                sync()
                return

            flush = getattr(backend, "flush", None)
            if callable(flush):
                flush()

        sampler = emcee.EnsembleSampler(
            opts.nwalkers, ndims, batched_log_prob,
            backend=backend,
            moves=[(emcee.moves.DEMove(), 0.8),
                   (emcee.moves.DESnookerMove(), 0.2)]
        )
        
        # Run with periodic checkpointing (save every hour or every 10,000 steps, whichever is first)
        checkpoint_interval = 10_000  # steps
        last_checkpoint_time = time.time()
        checkpoint_time_interval = 3600  # seconds (1 hour)
        
        print(f"Starting emcee run: nwalkers={opts.nwalkers}, nsteps={opts.nsteps}", flush=True)
        print(f"Checkpoints will be saved to: {backend_filename}", flush=True)
        
        for i in range(0, opts.nsteps, checkpoint_interval):
            n_steps = min(checkpoint_interval, opts.nsteps - i)
            sampler.run_mcmc(p0, n_steps, progress=opts.show_progress)
            p0 = sampler.get_last_sample()
            
            current_time = time.time()
            elapsed_since_checkpoint = current_time - last_checkpoint_time
            
            # Log progress
            print(f"Completed step {i + n_steps}/{opts.nsteps} ({100*(i+n_steps)/opts.nsteps:.1f}%) - "
                  f"Elapsed: {elapsed_since_checkpoint:.1f}s", flush=True)
            
            if elapsed_since_checkpoint >= checkpoint_time_interval:
                finalize_backend()
                print(f"Checkpoint saved at step {i + n_steps}", flush=True)
                last_checkpoint_time = current_time
        
        # Final sync to ensure all data is saved
        finalize_backend()
        print(f"Sampling complete. Final checkpoint saved to: {backend_filename}", flush=True)
        
        chain = sampler.flatchain
        samples = chain[len(chain)//2:]

        return {
            "samples": np.asarray(samples),
            "logZ": None,
            "logZerr": None
        }

    else:
        raise ValueError(f"Unknown sampler: {method}")