# lensing (`darksirens.lensing`)

The lensing forward model for joint population and lensing inference: the
weak-lensing magnification PDF, the SIS strong-lensing optical depth and marks,
the candidate-pair bookkeeping, and the file contracts the lensing CLI
validates. See [Lensing](../../guide/lensing.md).

## `darksirens.lensing.clusters`

Container and HDF5 I/O for pre-identified candidate multi-image clusters
supplied by an external pairwise-Bayes-factor analysis; the inference treats the
set as fixed.

```{eval-rst}
.. automodule:: darksirens.lensing.clusters
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.fcpdet`

The JAX Finn and Chernoff (1993) detection-probability model behind the
lensed-singleton (exactly-one-detected image) evidence and its partner
censoring factor.

```{eval-rst}
.. automodule:: darksirens.lensing.fcpdet
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.file_contract`

Formal file-contract validators for unified lensing analyses. They return
structured reports rather than raising, so preflight and standalone scripts
share them.

```{eval-rst}
.. automodule:: darksirens.lensing.file_contract
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.grids`

Quadrature grids for the magnification marginalisations: Gauss-Legendre nodes
in `ln mu` for weak lensing and in `y` on (0, 1) for the SIS integrals.

```{eval-rst}
.. automodule:: darksirens.lensing.grids
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.lensed_injections`

Container and loader for pre-rendered lensed-injection sets, the per-image flat
HDF5 layout the cluster and lensed-singleton selection integrals consume.

```{eval-rst}
.. automodule:: darksirens.lensing.lensed_injections
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.marginal_diagnostics`

Posterior diagnostics for exact partition-marginalized lensing runs.

```{eval-rst}
.. automodule:: darksirens.lensing.marginal_diagnostics
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.observed_catalog`

Schema helpers for the unified observed spectral-siren lensing catalog that
`--observed_catalog_path` points at.

```{eval-rst}
.. automodule:: darksirens.lensing.observed_catalog
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.pair_tag_selection`

Deterministic mock pair-tag selection models for simulated studies:
simulation-only scaffolding, not calibrated to any real search pipeline.

```{eval-rst}
.. automodule:: darksirens.lensing.pair_tag_selection
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.partitions`

Validates candidate-pair JSON files and enumerates the compatible pair
matchings for exact marginalization, with each edge carrying its
`log_prior_odds` relative to leaving both endpoints unpaired.

```{eval-rst}
.. automodule:: darksirens.lensing.partitions
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.preflight`

Preflight validation of the strong-lensing inference inputs, the checks
`--preflight_only` runs and writes to JSON.

```{eval-rst}
.. automodule:: darksirens.lensing.preflight
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.simulation_config`

Configuration helpers for simulated end-to-end lensing studies.

```{eval-rst}
.. automodule:: darksirens.lensing.simulation_config
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.slmarks`

The SIS optical depth `tau_J(z)` and the joint mark distribution of image
magnifications and parities. `J=2` (doubles) is implemented; the `J=4` hooks
are inert.

```{eval-rst}
.. automodule:: darksirens.lensing.slmarks
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.lensing.wlmagnification`

The weak-lensing magnification PDF `p_WL(mu | z)`, with a flux-conserving
lognormal backend and a user-supplied tabulated backend.

```{eval-rst}
.. automodule:: darksirens.lensing.wlmagnification
   :members:
   :undoc-members:
   :show-inheritance:
```
