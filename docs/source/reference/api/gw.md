# gw (`darksirens.gw`)

Gravitational-wave inputs and source-population models: gwcat PE and selection
store loading with format and spin-basis negotiation, the population registry
and its model-name grammar, and the optional normalizing-flow surrogates. The
`--pop_model` grammar is described in [Populations](../../guide/populations.md).

`darksirens.gw.populations` re-exports the stable registry surface
(`get_model`, `pop_model_parser`, `pop_model_prior_parser`, `FIDUCIAL_SETS`,
`get_fixed_population_params`, `population_m1_support_max`) from
`darksirens.gw.populations.registry`.

## `darksirens.gw.flows`

Per-event normalizing-flow posterior surrogates: checkpoint loading and
batched ensemble evaluation. Requires the `flows` extras
(`pip install 'darksirens[flows]'`).

```{eval-rst}
.. automodule:: darksirens.gw.flows
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.base`

Abstract component base classes and the `MixtureModel` / `PopulationModel`
assemblers, including the stick-breaking parameterisation of mixture weights
and the grid normalisation every component inherits.

```{eval-rst}
.. automodule:: darksirens.gw.populations.base
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.component_spin`

The 4-D component-spin population model: i.i.d. Beta spin magnitudes and a
tilt mixture, both normalised analytically so no new quadrature enters the
likelihood.

```{eval-rst}
.. automodule:: darksirens.gw.populations.component_spin
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.components`

The declarative blueprint table of the population registry: each component
declares its parameter names, LaTeX labels, default prior bounds, fiducials and
grammar token exactly once.

```{eval-rst}
.. automodule:: darksirens.gw.populations.components
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.gp`

Standalone Gaussian-process population models whose source-parameter density
is a GP prior over one or more of `(m1, q, chi_eff, z)`; they bypass the
stick-breaking mixture grammar entirely.

```{eval-rst}
.. automodule:: darksirens.gw.populations.gp
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.grammar`

The model-name grammar and generic mixture assembly: parses `--pop_model`
strings such as `brokenpowerlaw+2peaks` into a component composition.

```{eval-rst}
.. automodule:: darksirens.gw.populations.grammar
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.parametric`

The analytic building blocks: power-law, broken-power-law and Gaussian mass
components, pairing models, the spin model, and the curated GWTC-style
fiducial population models.

```{eval-rst}
.. automodule:: darksirens.gw.populations.parametric
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.registry`

The curated model registry and the stable named helpers `get_model`,
`pop_model_parser`, `pop_model_prior_parser` and
`get_fixed_population_params`.

```{eval-rst}
.. automodule:: darksirens.gw.populations.registry
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.sampling`

Grid-based inverse-CDF samplers for the population model, used by the
flow-surrogate likelihood path with common random numbers.

```{eval-rst}
.. automodule:: darksirens.gw.populations.sampling
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.populations.utils`

Normalisation-grid configuration for the population models: the mass,
mass-ratio and spin grid sizes and the optional pairing grid, resolved from the
CLI flags or the `DARKSIRENS_GW_*` environment variables.

```{eval-rst}
.. automodule:: darksirens.gw.populations.utils
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.samples`

The public loading surface for GW PE and selection samples: re-exports
`GWStore`, `SelectionStore`, `load_gw_store`, `load_gw_samples`,
`load_selection_store` and `load_selection_samples`.

```{eval-rst}
.. automodule:: darksirens.gw.samples
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.selection`

Turns a trained normalizing-flow detection-probability emulator into the
pseudo-injection set the hierarchical selection machinery consumes
(`--pdet_flow_path`).

```{eval-rst}
.. automodule:: darksirens.gw.selection
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.store_contract`

One declarative requirement table per accepted gwcat `format_version`: the
datasets, attrs and quality sets that both the loaders and the lensing
preflight validators check.

```{eval-rst}
.. automodule:: darksirens.gw.store_contract
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.gw.utils`

Reads gwcat PE and selection stores: format-version and spin-basis
negotiation, chi_eff swap validity, PE weight health reporting, and the
flattened sample arrays the likelihood consumes.

```{eval-rst}
.. automodule:: darksirens.gw.utils
   :members:
   :undoc-members:
   :show-inheritance:
```
