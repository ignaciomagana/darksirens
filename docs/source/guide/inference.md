# Running inference

End-to-end reference for `darksirens_inference`: which universe model to pick,
what each parameter block contributes, how to fix or re-bound individual
parameters, which sampler to run, how to checkpoint a multi-day job, and what
lands in the run directory. Input formats are in
[Inputs](../getting-started/inputs.md), every option in the
[CLI reference](../reference/cli.md).

## Universe models

`--universe_model` selects the redshift prior and what the run needs on disk.

| `--universe_model` | Also requires | Redshift prior |
| --- | --- | --- |
| `spectral_sirens` | nothing beyond GW data | Normalised comoving-volume element $p(z) \propto dV_c/dz$ (`log_volume_prior_vmap`). |
| `dark_sirens` | `--survey_path` | Completed catalog prior $p(z\mid\mathrm{pix}) = [N_{\rm obs}\,p_{\rm cat}(z\mid\mathrm{pix}) + dN_{\rm miss}/dz]/(N_{\rm obs}+N_{\rm miss})$, normalised per pixel. |
| `dark_sirens_complete` | `--survey_path` | Catalog prior alone, $p(z\mid\mathrm{pix}) = p_{\rm cat}(z\mid\mathrm{pix})$, with the empty-pixel rule from `--complete_empty_pixel_policy`. |
| `bright_sirens` | `--counterpart RA DEC Z` | Counterpart redshift likelihood $\mathcal{N}(z; z_{\rm cp}, \sigma_z)$ times the volume prior, gated to the counterpart sky pixel unless `--bright_siren_sky_marginalized true`. |

Every run also needs exactly one of `--gw_path` (stored PE samples) or
`--gw_flows_path` (flow surrogates, see [Populations](populations.md)), and
exactly one of `--gwselection_path` (injection campaign) or `--pdet_flow_path`
(emulator). `bright_sirens` forces `--sky_model isotropic`. The dark-siren
catalog options (`--use_lss`, `--c_mode`, `--lss_completion`, `--mark_model`,
K-catalog mixtures, `--sky_model`) are covered in [Catalogs](catalogs.md); the
lensing channels have their own command, described in [Lensing](lensing.md).

## Parameter blocks

The sampled coordinate vector is assembled block by block, always in this order:
cosmology, population, survey (catalog 1, then `_c{k}` catalogs), sky, marks,
catalog mixture weights. The population block is set by `--pop_model` and the
three sharing flags; its grammar, curated names, label spellings and flow
surrogates are documented in [Populations](populations.md).

### Cosmology

A flat CPL model, bounded by a fixed half width around the Planck-2015 centre.

| Label | Meaning | Default prior |
| --- | --- | --- |
| `H0` | Hubble constant in km/s/Mpc. | `[20, 120]` |
| `Om0` | Matter density fraction. | `[0.2075, 0.4075]` |
| `w0` | Present-day CPL equation of state. | `[-2, 0]` |
| `wa` | CPL evolution parameter. | `[-2, 2]` |

`w0 = -1`, `wa = 0` is flat $\Lambda$CDM and is what the block-fixing flags pin.

### Survey

From the `_SURVEY_BLOCK` registry in `darksirens/inference/prior.py`, which owns
the labels, their order, their bounds and their per-configuration activity.

| Label | Meaning | Default prior | Sampled when |
| --- | --- | --- | --- |
| `log10n0` | $\log_{10}$ of the comoving galaxy number density $n_0$ in Mpc$^{-3}$. | `[-4, -1]` | `dark_sirens` |
| `delta` | Density evolution exponent, $n(z) = n_0 (1+z)^\delta$. | `[-3, 3]` | `dark_sirens` |
| `b_miss` | Bias amplitude of the LSS-modulated missing-galaxy density (dimensionless). | `[0, 3]` | `dark_sirens` with `--use_lss true` and no `Q_LSS` table for that catalog |
| `sigma_kde` | Extra redshift width added in quadrature to the catalog kernels. | `[0, 0.05]` | `dark_sirens`, `dark_sirens_complete` |
| `M0hat` | $h$-scaled magnitude zero point of the Gaussian selection family. | `[-23, -18]` | `--c_mode selection`, gaussian family |
| `sigma_M` | Width of the Gaussian selection family. | `[0.05, 3]` | `--c_mode selection`, gaussian family |
| `Mstar_hat` | $h$-scaled Schechter $M^\ast$. | `[-23, -18]` | `--c_mode selection`, schechter family |
| `alpha` | Schechter faint-end slope. | `[-1.9, 0]` | `--c_mode selection`, schechter family |

The gaussian and schechter pairs are mutually exclusive, so the block length
never changes and no later label's index moves. `z50`, `w`, `alpha_miss`,
`m_lim` and `M_faint_offset` are `SurveyParams` fields no universe model samples:
they carry no prior, and naming one in `--prior_overrides` is an error quoting the
registry's reason (`m_lim` and `M_faint_offset` can still be pinned with
`--fixed_parameter_values`).

## Fixing parameters and overriding priors

Block flags remove a whole block and pin it at its fiducial.

| Flag | Effect |
| --- | --- |
| `--fix_cosmology true` | Removes `H0`, `Om0`, `w0`, `wa`. |
| `--fix_de true` | Removes only `w0` and `wa`; ignored when `--fix_cosmology` is true. |
| `--fix_survey true` | Removes the survey block. |
| `--fix_population true` | Removes the population block; `--population_fiducials {legacy,in_prior_v2}` chooses which curated fiducial vector it pins to. |

Two JSON options act per label: `--prior_overrides` takes `[lower, upper]` pairs
and re-bounds a sampled label, `--fixed_parameter_values` takes scalars and
removes the label from the sampled vector.

```bash
--prior_overrides        '{"H0": [60.0, 80.0], "w0": [-1.2, -0.8]}'
--fixed_parameter_values '{"Om0": 0.3075, "w0": -1.0, "wa": 0.0}'
```

An unknown key raises `KeyError` listing every valid label for the resolved
configuration, and a survey label this configuration does not sample is refused
with the registry's own reason.

```{warning}
Keys must be the exact printed label, LaTeX spelling and component tag included,
for example `$\alpha_{\rm PL}$` or `$\mu_{\rm G2}$`. Copy them from the startup
table printed under "Parameter Space", whose three sections are the sampled
parameters with their (possibly overridden) bounds, the individually fixed ones,
and the block-fixed ones with their pinned fiducials.
```

```text
  │    Parameter                      Lower         Upper  Status
  │    ──────────────────────── ────────────  ────────────  ────────────────────
  │    H0                                 60            80  ← overridden
  │    $\alpha_{\rm PL}$                  -4             6
  │    $\mu_{\rm G}$                      20            50
  │    $\gamma$                          -10            10
  │    ──────────────────────── ────────────  ────────────  ────────────────────
  │    Om0                                 —             —  fixed = 0.3075
  │    ──────────────────────── ────────────  ────────────  ────────────────────
```

## Samplers

`--sampler` is required.

| Sampler | Output | Key knobs | Use it when |
| --- | --- | --- | --- |
| `tinyns` | Samples plus evidence | `--nlive` (1000), `--dlogz` (0.1), `--max_samples` (iteration cap), `--tinyns_preset`, `--tinyns_walks`, `--tinyns_step_scale`, `--tinyns_replacement_chains`, `--tinyns_jax_block_size` | The likelihood is on a GPU and you want the vectorised JAX random-walk kernel. |
| `dynesty` | Samples plus evidence | `--nlive`, `--dlogz`, `--max_samples` (call cap), `--dynesty_diagnostics`, `--prior_transform_dispatch` | You want the reference nested sampler, wall-clock checkpointing honoured exactly, and runplot/traceplot diagnostics. |
| `numpyro` | Samples only, no evidence | `--nuts_warmup` (500), `--nuts_samples` (1000), `--nuts_chains` (1), `--nuts_target_accept` (0.8), `--nuts_max_tree_depth` (10), `--nuts_chain_method`, `--nuts_init_tries` | The posterior is high dimensional and smooth and you do not need $\log Z$. |

`--max_samples` (default `1000000`, `0` = unlimited) is a dynesty call cap but a
tinyns iteration cap, and one tinyns `rwalk` iteration costs
`walks * max_active_chains` likelihood evaluations, so the same number buys far
more compute there; the resolved cap and its call equivalent are printed at
sampler start.

`--tinyns_preset` defaults to `recommended`: `sample=rwalk`, `kernel=jax`,
isotropic proposals, `walks=5`, `step_scale=0.1`, `replacement_chains=1`,
`bound=none`, `jax_block_size=32`. `heavy_darksirens` and
`heavy_darksirens_strong` raise `walks` to 80 and 160 with 16 replacement chains
for expensive dark-siren likelihoods; `python_debug` avoids the JAX kernel.
Explicit `--tinyns_*` flags override the preset, and the resolved values are
stored as `tinyns_resolved_config` in `settings.json` and `results.hdf5`.

Two guards act at startup: `--sampler_preflight on` (default) probes 32 prior
draws and fails fast if all are `-inf`, and `--selection_neff_guard auto` takes
the smooth `soft` wall for `numpyro` and the hard `-inf` wall for the nested
samplers, against `--max_likelihood_variance` (default `1.0`). See
[Troubleshooting](troubleshooting.md).

## Checkpoint and resume

Checkpointing is on by default and both nested samplers honour it.

- `--checkpoint_interval SECONDS|off` (default `1800`) sets the cadence and writes
  `checkpoint.dynesty.pkl` or `checkpoint.tinyns.npz` into the run directory.
  dynesty follows the wall clock exactly; tinyns saves every
  `--tinyns_checkpoint_interval` iterations (default `100`), which this flag only
  enables or disables.
- `--resume auto|PATH|off` (default `off`) restores a checkpoint and continues
  inside its original run directory. `auto` takes the most recently modified
  checkpoint under `--save_path` whose directory name carries this run's
  `<pop_model>__<universe_model>__<sampler>__seed<seed>__` prefix, skips
  directories already holding a complete `results.hdf5`, and starts fresh
  silently when there is nothing to resume. `PATH` may be a run directory or a
  checkpoint file and must exist.
- A resume continues the checkpoint's `--seed`-derived RNG stream, so the resumed
  chain is the chain the uninterrupted run would have produced.

A resume is gated on the `run_fingerprint.json` written into every run directory
at creation time. It covers the sampled labels, their bounds, the per-parameter
prior families, the resolved joint prior constraints, the overridden and fixed
values, every model flag, the sampler and its stopping settings, the seed, the
normalisation and redshift grids, and the SHA-256 content digest of every input
file (a sampled digest above 1 GiB). Operational knobs (`--save_path`,
checkpoint cadence, progress and diagnostic flags, performance chunking) are
excluded, so requeuing the identical command always matches. Code identity (git
sha, package versions) is advisory and only warns. `--resume_force` bypasses the
gate, and a forced mismatch mixes two statistical targets.

## The run directory

Each run creates `<save_path>/<pop_model>__<universe_model>__<sampler>__seed<seed>__<timestamp>`,
adding a `-01`, `-02` suffix rather than reusing an existing one.

| File | Written | Contents |
| --- | --- | --- |
| `settings.json` | before sampling | Every CLI option, the label list, bounds, fixed values, prior overrides, run metadata, `normalization_grid`, and the environment block. |
| `run_fingerprint.json` | before sampling | The semantic fingerprint above, plus its `digest` and the advisory code block. |
| `checkpoint.dynesty.pkl` / `checkpoint.tinyns.npz` | during sampling | Sampler state for `--resume`. |
| `samples.npy` | after sampling | The bare `(N_samples, N_dim)` chain, written before `results.hdf5` as a crash-recovery copy. |
| `results.hdf5` | after sampling | Samples, metadata and evidence (below). Renamed into place only after a clean write and stamped `result_complete`. |
| `tinyns_diagnostics.json` | after sampling | `tinyns_runtime_diagnostics`, `tinyns_resolved_config`, `tinyns_summary`. |
| `corner.pdf`, `latents.pdf` | after sampling | Corner of cosmology, hyperparameters and survey block (skipped when nothing is free), plus a latent summary for models that have latents. |
| `dynesty_diagnostics/` | every 10 min | Runplot and traceplot PDFs, with `--dynesty_diagnostics true`. |
| `failure.json` | on a crash | The stage that failed and its exception. |
| `settings.resume-<timestamp>.json`, `run_fingerprint.forced-<timestamp>.json` | per resume attempt | One per attempt, beside the originals, which are never overwritten. |

`results.hdf5` holds `samples` `(N_samples, N_dim)`, `labels`, `lower_bound`,
`upper_bound` and, when available, `fixed_labels`, `fixed_values`, `log_weights`,
`log_likelihood`, `counterparts`, plus the dead-point record
`logl_dead`/`logwt_dead` (length `n_dead`, not row-aligned with `samples`). Its
attributes carry the evidence, every model and sampler setting, the input paths
and the runtimes ([Analysis](analysis.md) shows how to read them).

## Common recipes

Spectral-siren production run, resumable after a SLURM requeue:

```bash
darksirens_inference \
  --gw_path gw_events.h5 \
  --gwselection_path gw_selection.h5 \
  --universe_model spectral_sirens \
  --pop_model gwtc3_fiducial_plpeak \
  --sampler dynesty --nlive 2000 --dlogz 0.1 \
  --checkpoint_interval 1800 --resume auto \
  --save_path runs/spectral_plpeak
```

Dark-siren run against a pixelated catalog, fixed population, tinyns:

```bash
darksirens_inference \
  --gw_path gw_events.h5 \
  --gwselection_path gw_selection.h5 \
  --survey_path catalog_nside_32.h5 \
  --universe_model dark_sirens \
  --pop_model brokenpowerlaw+2peaks \
  --fix_population true \
  --fix_de true \
  --sampler tinyns --tinyns_preset heavy_darksirens \
  --nlive 2000 --dlogz 0.11 \
  --save_path runs/dark_bpl2peaks
```

Fixed-parameter smoke test, checking loading, selection and output writing:

```bash
darksirens_inference \
  --gw_path gw_events.h5 \
  --gwselection_path gw_selection.h5 \
  --sampler tinyns --nlive 100 --dlogz 10 \
  --fix_de true --fix_survey true \
  --fixed_parameter_values '{"H0": 67.74, "Om0": 0.3075}' \
  --save_path runs/smoke
```

Model comparison: same data, sampler settings and seed per model, one directory
each, then compare evidences with [`darksirens_analyze`](analysis.md):

```bash
for MODEL in powerlaw+peak brokenpowerlaw+2peaks gwtc5_fiducial_bpl2peaks; do
  darksirens_inference \
    --gw_path gw_events.h5 \
    --gwselection_path gw_selection.h5 \
    --universe_model spectral_sirens \
    --pop_model "$MODEL" \
    --sampler dynesty --nlive 2000 --seed 22 \
    --save_path runs/compare
done

darksirens_analyze --run_dirs runs/compare/*/ --outdir figures/compare
```
