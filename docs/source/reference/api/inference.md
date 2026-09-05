# inference (`darksirens.inference`)

The run assembly layer between the CLIs and the likelihood: staging input
files, building the parameter space and prior transform, resolving sampler
configuration, and gating resume on a semantic fingerprint. See
[Inference](../../guide/inference.md) for the run-level view.

## `darksirens.inference.checkpointing`

The checkpoint and resume policy shared by both inference CLIs: cadence
planning, dynesty checkpoint installation, and restoration of a sampler from a
run directory.

```{eval-rst}
.. automodule:: darksirens.inference.checkpointing
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.data`

`load_all_data(opts)`, the single entry point that stages every input file for
a run and records the input provenance attrs carried into the artifacts.

```{eval-rst}
.. automodule:: darksirens.inference.data
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.loaders`

The staged loaders behind `load_all_data`: GW and selection stores, catalog
inputs and multitracer bundles, sky pixels and vectors, and the `attach_*`
steps for selection fractions, LSS tables, marks and WL inputs.

```{eval-rst}
.. automodule:: darksirens.inference.loaders
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.parameters`

`build_parameter_decoder`: turns a sampled coordinate vector into
`CosmoParams`, per-catalog `SurveyParams` and the population, sky and mark
blocks, including the mixture stick transforms.

```{eval-rst}
.. automodule:: darksirens.inference.parameters
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.pop_extractor`

`make_pop_extractor(settings)`, the single source of truth for slicing the
population sub-vector out of theta, plus the catalog-stick to host-fraction
conversions used in post-processing.

```{eval-rst}
.. automodule:: darksirens.inference.pop_extractor
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.prior`

Builds the run's parameter space: which labels are sampled given the model
flags, their bounds after `--prior_overrides`, and the prior transform the
samplers call.

```{eval-rst}
.. automodule:: darksirens.inference.prior
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.q_provenance`

Enforces that a prebuilt Q_LSS table's build-time conditioning (cosmology,
`n0`, `delta`, bias, completeness base) matches the configuration the run is
about to use.

```{eval-rst}
.. automodule:: darksirens.inference.q_provenance
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.run_fingerprint`

The semantic run fingerprint written as `run_fingerprint.json`: the
resume-compatibility gate over inputs by content, priors, fixed values, model
flags and sampler settings.

```{eval-rst}
.. automodule:: darksirens.inference.run_fingerprint
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.sampling`

`run_sampler`: the dynesty, TinyNS and NumPyro NUTS adapters, the
nested-sampling preflight probe, the dynesty prior-transform dispatch and the
diagnostics each backend reports.

```{eval-rst}
.. automodule:: darksirens.inference.sampling
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.tinyns_config`

Resolves and validates the TinyNS configuration from `--tinyns_preset` plus
the explicit `--tinyns_*` overrides.

```{eval-rst}
.. automodule:: darksirens.inference.tinyns_config
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.utils`

Shared pure functions of the hierarchical likelihood, notably
`log_sample_weight`, reused by both the PE and the selection term.

```{eval-rst}
.. automodule:: darksirens.inference.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.inference.validation`

The canonical post-load validation of a multitracer run: every per-catalog
config sequence must agree with `n_catalogs` before the parameter space is
built.

```{eval-rst}
.. automodule:: darksirens.inference.validation
   :members:
   :undoc-members:
   :show-inheritance:
```
