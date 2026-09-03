# core (`darksirens.core`)

Typed contracts, constants, universe-model groupings and JAX runtime
configuration. Nothing here imports the samplers or the likelihood, so these
modules are safe to import from tooling and tests.

## `darksirens.core.constants`

Central constants shared across darksirens modules: fiducial dark-energy and
survey parameter values, the `c_mode`, selection-family and fiducial-set
enumerations, and the default likelihood-variance cap.

```{eval-rst}
.. automodule:: darksirens.core.constants
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.core.jax_config`

`configure_jax_runtime()`, the single place that sets the JAX runtime knobs
(including x64) before JAX is used.

```{eval-rst}
.. automodule:: darksirens.core.jax_config
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.core.model_kinds`

Canonical universe-model groupings used across inference helpers: which
`--universe_model` values are galaxy-aware and which are bright-siren models.

```{eval-rst}
.. automodule:: darksirens.core.model_kinds
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.core.types`

The containers that define the shape contract between data loading and the
JIT-compiled likelihood: `CosmoParams`, `SurveyParams`, `EMCatalog`, `GWEvent`,
and the leaf-less structural flags that carry trace-time modes across jit.

```{eval-rst}
.. automodule:: darksirens.core.types
   :members:
   :undoc-members:
   :show-inheritance:
```
