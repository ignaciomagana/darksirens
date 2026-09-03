# likelihood (`darksirens.likelihood`)

The hierarchical likelihood itself: the JIT body, its factory, the selection
integral, and the lensing extensions that wrap them. Memory and speed knobs are
covered in [Performance](../../guide/performance.md).

## `darksirens.likelihood.block_sizing`

Memory-aware auto-sizing of the two block-size knobs, `--sel_batch_size` and
`--pe_event_block`, from probed device memory.

```{eval-rst}
.. automodule:: darksirens.likelihood.block_sizing
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.catalog_views`

Catalog compaction and KDE cache setup for dark-siren inference, performed once
before the likelihood closure is built.

```{eval-rst}
.. automodule:: darksirens.likelihood.catalog_views
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.cluster_likelihood`

The J=2 (double-image) pair likelihood: the SIS pair model evaluated with the
Janquart-style KDE-on-PE estimator.

```{eval-rst}
.. automodule:: darksirens.likelihood.cluster_likelihood
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.cluster_selection`

The J=2 cluster contribution to the expected detected-event count, including
the both-detected approximation and its exactly-one-detected refinements.

```{eval-rst}
.. automodule:: darksirens.likelihood.cluster_selection
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.core`

`darksiren_log_likelihood`, the pure JIT body of the hierarchical dark-siren
likelihood.

```{eval-rst}
.. automodule:: darksirens.likelihood.core
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.events`

Factory for `GWEvent` containers: applies the `lax.optimization_barrier`
wrapping that keeps large data arrays out of the HLO graph, and pre-computes
the mass ratio.

```{eval-rst}
.. automodule:: darksirens.likelihood.events
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.factory`

`make_likelihood`: closes the likelihood body over the loaded data, resolves
the table-versus-latent Q mode and its provenance guards, and returns the
callable the samplers evaluate.

```{eval-rst}
.. automodule:: darksirens.likelihood.factory
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.flow_events`

The flow-surrogate hierarchical likelihood for spectral sirens: population
draws scored by each event's flow instead of stored PE samples. Requires the
`flows` extras.

```{eval-rst}
.. automodule:: darksirens.likelihood.flow_events
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.latent_q`

The latent-field seam, the one place `Q` is generated in
`--lss_field_mode latent`, with the closed-form budget normalizer that
conserves the missing count at every redshift.

```{eval-rst}
.. automodule:: darksirens.likelihood.latent_q
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.likelihood_with_clusters`

`darksiren_log_likelihood_with_clusters`, the master marked-Poisson likelihood
over singletons plus J=2 clusters, and its diagnostics twin.

```{eval-rst}
.. automodule:: darksirens.likelihood.likelihood_with_clusters
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.pair_kde`

The precomputed per-event Gaussian KDE over apparent-frame PE samples used by
the cluster-pair likelihood.

```{eval-rst}
.. automodule:: darksirens.likelihood.pair_kde
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.selection`

The hierarchical selection integral (Thrane & Talbot 2019; Farr 2019), its
effective sample size, and the Monte-Carlo variance the guards act on.

```{eval-rst}
.. automodule:: darksirens.likelihood.selection
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.likelihood.wl_weight`

The weak-lensing-marginalized per-sample log importance weight: the
magnification integral inside the per-event PE term.

```{eval-rst}
.. automodule:: darksirens.likelihood.wl_weight
   :members:
   :undoc-members:
   :show-inheritance:
```
