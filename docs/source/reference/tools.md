# Command-line tools (`darksirens.cli`)

Each `darksirens.cli` module backs an installed console script (see
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

The spectral-siren lensing driver: splits events into singletons and optional
J=2 lensed image pairs and drives `darksiren_log_likelihood_with_clusters`
([`lensing`](lensing.md)) over a `ClusterSet` and a `LensedInjectionSet`.  This
CLI is for `spectral_sirens` / `spectral_sirens_wl` lensing only; it does not
run galaxy-catalog dark-siren, LSS-completion, or catalog-host lensing
inference.

Important data and model flags:

* `--gw_path` and `--gwselection_path` are required for all runs.
* `--cluster_mode off` runs the singleton spectral-siren likelihood only.
* `--cluster_mode j2` adds J=2 strong-lensing pairs and requires
  `--lensed_injections_path` plus the pair inputs. Those now come from the
  unified observed catalog (`--observed_catalog_path`, with optional
  `--pair_metadata_path`); `--pair_pe_path` is the DEPRECATED split-pair layout,
  which the mock generator stopped writing by default and which preflight's
  event-index range check has never accepted.
* `--partition_mode fixed` uses one explicit `--partition_path`;
  `--partition_mode marginalize_exact` uses `--candidate_pairs_path` and
  `--max_exact_partitions` to exactly marginalise over a small candidate graph.
* `--wl_backend {lognormal,tabulated,disabled}` selects `spectral_sirens_wl`
  (lognormal or tabulated WL magnification) or ordinary `spectral_sirens`;
  `tabulated` reads `--lensing_wl_table_path` (HDF5 with datasets `z_grid`,
  `log_mu_grid`, `log_p_table` giving `log p_WL(mu|z)`).  This CLI is the sole
  owner of the `spectral_sirens_wl` universe model.  `--wl_selection
  {standard,wl_lognormal}` controls whether singleton selection also receives
  the lognormal/Hermite WL treatment.
* `--pair_marks {none,time}` optionally adds SIS time-delay marks from pair-PE
  metadata, with `--pair_time_sigma_sec` as a fallback uncertainty.
* `--fix_lens_rate true` fixes `--sl_tau_A` and `--sl_tau_n`;
  `--fix_lens_rate false` samples `log10_tau_A` and `tau_n`, optionally with
  `--lens_prior_overrides` and `--fixed_parameter_values`.
* `--pe_max_per_pair`, `--pair_batch_size`, `--y_nodes_pair`, and
  `--sel_batch_size` are memory/performance controls for pair and selection
  integrals.

Every run writes `results.hdf5`, `settings.json`, `diagnostics.json`, and
`diagnostics.hdf5` under the run directory.  Use
`darksirens_inference_lensing` or `python -m darksirens.cli.inference_lensing`;
there is no supported `darksirens.tool` path for this workflow.

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
([`redshift.lognormal_completion`](em.md)). The `--mode radial` path solves an
independent 1-D field per pixel; `--mode gp3d` solves one low-rank
$(\text{sphere}\times z)$ field so empty pixels borrow angularly from their
neighbours. Both write the same HDF5 table consumed by
`darksirens_inference --lss_completion`.

```{automodule} darksirens.cli.build_lognormal_completion
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens_build_joint_lognormal_completion`

The **joint** multi-survey variant, for `K >= 2` catalogs analysed as a
multitracer mixture. It infers ONE latent LSS field from all K catalogs at once
(`gp3d` only, per-survey bias $b_k$ absorbed into the design matrix) and writes
K per-survey $Q_{\rm LSS}$ files stamped with a single shared
`realization_set_id`, so member $m$ of every file is the same LSS realization.
That is exactly what `darksirens_inference --lss_marginalize` requires at
`K >= 2`; `--mode radial` is rejected because per-pixel independent fits carry
no shared field to match members across surveys.

```{automodule} darksirens.cli.build_joint_lognormal_completion
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
