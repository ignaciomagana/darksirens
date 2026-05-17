# Module reference

This page exposes the Sphinx autodoc output for the package.  The short notes
before each section summarize how the modules fit together; the generated member
lists provide class, function, and parameter-level details from the code
comments and docstrings.

## EM package

Electromagnetic modules load HEALPix galaxy surveys, validate catalog metadata,
construct smooth observed-galaxy completion terms, and provide redshift priors
for the likelihood.  These routines are used by both complete-catalog and
incomplete-catalog dark-siren analyses.

```{automodule} darksirens.em
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.em.catalog
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.em.checks
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.em.completion
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.em.prior
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.em.utils
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.em.volume
:members:
:undoc-members:
:show-inheritance:
```

## GW package

Gravitational-wave modules load event posterior samples and injection samples,
compute sky pixels, and supply the lower-level utilities needed by population
and selection calculations.

```{automodule} darksirens.gw.selection
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.gw.utils
:members:
:undoc-members:
:show-inheritance:
```

## GW population models

Population modules define the source population density.  Analytic and
Gaussian-process components implement reusable mass, mass-ratio, and spin
factors.  Registry functions assemble named models, parameter bounds, and labels
that are consumed by inference priors and samplers.

```{automodule} darksirens.gw.populations
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.gw.populations.base
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.gw.populations.gp
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.gw.populations.parametric
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.gw.populations.registry
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.gw.populations.utils
:members:
:undoc-members:
:show-inheritance:
```

## Inference package

Inference modules turn loaded data into JAX containers, compact repeated catalog
pixels, define prior transforms, evaluate the hierarchical likelihood, and adapt
that likelihood to the supported sampler backends.  The JIT-compiled likelihood
core is intentionally separated from Python-side setup to keep sampler iterations
fast and deterministic.

```{automodule} darksirens.inference.catalog_views
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.data
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.events
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.likelihood
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.likelihood_core
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.parameters
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.pop_extractor
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.prior
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.sampling
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.selection
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.inference.utils
:members:
:undoc-members:
:show-inheritance:
```

## Tool package

Tool modules back the command-line programs.  They mostly parse arguments,
validate user-facing configuration, call the documented loaders and samplers,
and write plots or posterior products.

```{automodule} darksirens.tool.darksirens_analyze
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.tool.darksirens_inference
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.tool.darksirens_pixelate
:members:
:undoc-members:
:show-inheritance:
```

## Utilities package

Utility modules contain shared cosmology, interpolation, plotting, and container
helpers.  They are intentionally small and are safe to import from scripts that
need package-compatible numerical conventions.

```{automodule} darksirens.utils.containers
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.utils.cosmology
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.utils.interp2d
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.utils.plotting
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} darksirens.utils.utils
:members:
:undoc-members:
:show-inheritance:
```
