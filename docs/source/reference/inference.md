# Inference engine (`darksirens.inference`)

The `inference` subpackage is the computational core: it loads and compacts the
data, builds the parameter space and prior transform, evaluates the
selection-corrected hierarchical likelihood, and drives the samplers. The
estimator is derived on the [Theory & methods](../theory.md) page.

## `darksirens.inference.data`

`load_all_data` is the single entry that reads everything a run needs: GW PE
samples and selection injections ([`gw.utils`](populations.md)), the optional
pixelated survey, the optional LSS completion table $Q_{\rm LSS}$, optional
galaxy marks (z-centred at load), and — for `spectral_sirens_wl` — the
weak-lensing magnification parameters. It validates array shapes and the
completion grid before the JIT boundary.

```{automodule} darksirens.inference.data
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.catalog_views`

Compacts the per-sample catalog so only the pixels actually touched by the PE
and injection sets are transferred to the device: repeated sample pixels are
replaced by `unique_pixels_*` plus a `sample_to_unique_*` gather map. `barrier`
wraps a host→device transfer so a large static operand is materialised once and
captured by the JIT, not re-sent per call.

```{automodule} darksirens.inference.catalog_views
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.events`

The `GWEvent` padding helpers. `pad_gw_event_to_multiple` appends explicit
invalid sentinel rows so the selection set length is divisible by the scan batch
size, keeping `lax.scan` shapes static without changing the estimate.

```{automodule} darksirens.inference.events
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.parameters`

Decodes a flat sampler coordinate into typed parameter blocks.
`ParameterDecoder.decode(coord)` returns the 5-tuple `(cosmo, survey,
pop_params, sky_params, mark_params)`, filling fixed parameters from fiducials;
the weak-lensing magnification model rides as a fixed field on `SurveyParams`
(`wl_params`), so the decode arity is unchanged. `build_parameter_decoder`
constructs the decoder from the CLI options and the `build_parameter_space`
ordering.

```{automodule} darksirens.inference.parameters
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.prior`

Builds the sampled parameter space and the prior transform. `build_parameter_space`
assembles the cosmology, population, survey, sky, and mark blocks, returning the
labels, bounds, per-parameter prior families, and the sky/mark label lists. Only
the survey parameters in the per-model `_ACTIVE_SURVEY_PARAMS` allow-list are
sampled (so fixed builder hyperparameters such as the LSS correlation lengths
never become nuisance dimensions). `make_prior_transform` is the unit-cube →
parameter inverse-CDF map, prior-family aware (uniform / normal / lognormal).

```{automodule} darksirens.inference.prior
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.utils`

The shared per-sample importance weight. `log_sample_weight` evaluates
$\ln w$ in the canonical $(m_1^{\rm det}, q, d_L)$ basis — population density ×
redshift prior × the source→detector Jacobian
`log_jacobian_m1src_q_z_to_m1det_q_dL`, divided by the PE prior — and is called
identically by the PE and selection terms.

```{automodule} darksirens.inference.utils
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.selection`

The Monte-Carlo selection correction (Farr 2019). `_lse_to_log_mu_neff`
converts the log-sum-exp aggregates of the injection weights into
$(\ln\mu, N_{\rm eff}, \ln\widehat\sigma^2_\mu)$, with
$N_{\rm eff} = \mu^2/\widehat\sigma^2_\mu$ and $\ln\widehat\sigma^2_\mu$ the log
Monte-Carlo variance of $\mu$. `selection_log_correction` applies the
$N_{\rm eff} > 4\,N_{\rm obs}$ (Vitale et al. 2022) too-sparse veto, returning
$-\infty$ otherwise. `compute_selection_term` evaluates the injection weights
(optionally batched via `sel_batch_size`), accepts a `sky_log_weight_fn` hook for
the angular factor, and returns the 3-tuple — the third element feeds the
strong-lensing cluster combiner ([`lensing`](lensing.md)).

```{automodule} darksirens.inference.selection
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.likelihood_core`

The pure, JIT-compiled likelihood body `darksiren_log_likelihood`: a `lax.scan`
over events of the log-sum-exp PE term, minus the selection correction. It adds
the mean-one sky factor $\ln g(\hat n, z)$ to the shared weight when a
non-isotropic sky model is active, and threads the marked-host model and the
LSS completion through the prior states. The static `wl_backend` code gates the
weak-lensing magnification marginalisation (disabled / lognormal / tabulated),
which is bit-for-bit inert for every non-WL model; `pe_model` / `selection_model`
fall back to `spectral_sirens` for the WL and bright-siren cases.

```{automodule} darksirens.inference.likelihood_core
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.likelihood`

`make_likelihood` is the host-side factory: it prepares the compact catalog
views, builds the parameter decoder, resolves the weak-lensing backend from
`data["wl_params"]`, and returns the `likelihood(coord)` closure handed to the
sampler. It marks the eager/traced boundary — the heavy per-proposal
precomputation happens here, while the returned closure calls the JIT core.

```{automodule} darksirens.inference.likelihood
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.sampling`

`run_sampler` adapts a likelihood + prior transform to `jaxns`, `dynesty`,
`emcee`, or `numpyro` behind one interface. When the parameter space has zero
free dimensions (everything fixed) it short-circuits before any sampler — the
prior is a point mass so the evidence is exact, $\ln Z = \ln \mathcal L$ at the
fixed point.

```{automodule} darksirens.inference.sampling
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.inference.pop_extractor`

Builds the population-density callable used by the posterior-predictive analysis
from a saved run's settings, reconstructing the model and binding fixed
parameters so `darksirens_analyze` can recompute mass/spin/redshift spectra.

```{automodule} darksirens.inference.pop_extractor
:members:
:undoc-members:
:show-inheritance:
```
