# Command-line tools (`darksirens.cli`)

Each `darksirens.cli` module backs an installed console script. `darksirens.tool` remains available as a compatibility-only module path (see
[`setup.py` entry points](../cli.md)). This page documents what each program
does and the modules behind them; full flag listings are in the
[CLI reference](../cli.md).

## `darksirens_pixelate`

Bins a raw galaxy catalog (RA, Dec, $z$, optional weights and marks) onto a
HEALPix grid at a chosen `nside`, producing the padded per-pixel survey datasets
(`zgals`, `ngals`, `dzgals`, `wgals`, and any mark columns) that
[`catalogs.io.load_survey`](api.md) reads.

```{automodule} darksirens.cli.pixelate
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens_inference`

The main hierarchical-inference driver: loads the data, builds the parameter
space and likelihood, runs the chosen sampler, and writes `results.hdf5` plus
`settings.json`. It selects the universe model, population model, sky model,
marked-host model, and (optionally) the LSS completion table and weak-lensing
magnification via CLI flags.

```{automodule} darksirens.cli.inference
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens_inference_lensing`

The strong-lensing variant: splits events into singletons and lensed image
pairs and drives `darksiren_log_likelihood_with_clusters`
([`lensing`](lensing.md)) over a `ClusterSet` and a `LensedInjectionSet`.

```{automodule} darksirens.cli.inference_lensing
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens_analyze`

Post-processes one or more completed runs: recomputes posterior-predictive
population spectra ($p(m_1), p(m_2), p(q), p(z), p(\chi)$ and the 2-D
$p(m_1, m_2)$) on a memory-safe chunked grid, plots cosmology posteriors and the
detection-rate $\mathrm{d}N/\mathrm{d}z$, and compares models via relative
evidences and the pairwise Bayes-factor matrix. `load_run` reads the current
grouped `results.hdf5` (with a legacy `samples.npy` fallback).

```{automodule} darksirens.cli.analyze
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens_build_lognormal_completion`

The **offline** preprocessor that builds the LSS-conditioned lognormal
completion file $Q_{\rm LSS}(p,z)$ from a pixelated catalog
([`em.lognormal_completion`](em.md)). The `--mode radial` path solves an
independent 1-D field per pixel; `--mode gp3d` solves one low-rank
$(\text{sphere}\times z)$ field so empty pixels borrow angularly from their
neighbours. Both write the same HDF5 table consumed by
`darksirens_inference --lss_completion`.

```{automodule} darksirens.cli.build_lognormal_completion
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens_diagnose_lognormal_completion`

Diagnostic plots for a completion file at a chosen pixel: $Q_{\rm LSS}(p,z)$
(MAP, posterior mean, and member band), the missing-galaxy density, and the
assembled $p(z\mid\text{pix})$ including the Bayesian completion mean.

```{automodule} darksirens.cli.diagnose_lognormal_completion
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens_skymaps_to_samples`

Converts GW localisation sky maps into the per-event posterior samples consumed
by the loaders, drawing distance/sky samples consistent with the map.

```{automodule} darksirens.cli.skymaps_to_samples
:members:
:undoc-members:
:show-inheritance:
```
