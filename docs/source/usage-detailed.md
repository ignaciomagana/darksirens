# Detailed user guide

`darksirens` is a Python package for hierarchical cosmological inference with gravitational-wave (GW) events and galaxy surveys. It supports:

- spectral-siren inference using GW observations and GW selection effects;
- dark-siren inference using pixelated electromagnetic galaxy catalogs;
- bright-siren inference using known electromagnetic counterparts;
- posterior-predictive analysis of fitted population models; and
- survey pixelation into HEALPix-based HDF5 catalogs.

The package exposes three command-line tools:

| Command | Purpose |
| --- | --- |
| `darksirens_pixelate` | Convert a raw galaxy survey HDF5 file into the pixelated HEALPix catalog format used by inference. |
| `darksirens_inference` | Run hierarchical inference with spectral-siren, dark-siren, complete-catalog dark-siren, or bright-siren models. |
| `darksirens_analyze` | Analyze completed inference runs and generate posterior-predictive and model-comparison plots. |

## Installation

Install the package in editable mode from the repository root:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

Additional sampler-specific packages may be required depending on the selected backend:

- `tinyns` for `--sampler tinyns` (installed from GitHub via `requirements.txt`);
- `dynesty` for `--sampler dynesty`; and
- `numpyro` for `--sampler numpyro`.

Build the documentation locally with:

```bash
python -m pip install -r docs/requirements.txt
make docs-html
```

For a stricter documentation build, run:

```bash
make docs-strict
```

## Input data overview

Most workflows require two gwcat-generated GW inputs:

1. posterior samples for one or more GW events; and
2. GW selection samples generated from the relevant injection products.

Dark-siren workflows additionally require a pixelated galaxy catalog. Bright-siren workflows use counterpart coordinates and redshifts instead of a survey catalog.

### GW posterior sample file

The GW posterior sample file is a gwcat HDF5 export passed with:

```bash
--gw_path PATH_TO_GW_POSTERIORS.h5
```

Create it with `gwcat.GWCatalog._to_darksirens_format(...)`. The loader requires `format_version="gwcat-1.0"` plus the gwcat-exported posterior datasets and metadata documented in [Data formats](data-formats.md). It rejects raw PE files and files that rely on darksirens to fill in missing `chieff`, `p_pe`, or source-mass arrays.

The likelihood convention uses the canonical sample basis:

```text
(m1det, q, dL)
```

where:

```text
q = m2det / m1det
```

`gwcat` is responsible for storing `p_pe` in that basis before darksirens inference starts.

### GW selection file

The GW selection file is a gwcat HDF5 export passed with:

```bash
--gwselection_path PATH_TO_SELECTION.h5
```

Create it with `gwcat.SelectionSet.to_darksirens(...)` or `gwcat.CombinedSelectionSet.to_darksirens(...)`. The loader requires `format_version="gwcat-selection-1.0"`; raw LVK injection files should be preprocessed with gwcat first. It is used to compute the selection correction for each sampled population and cosmology point.

For large gwcat selection files, reduce memory pressure with:

```bash
--sel_batch_size 200000
```

The best value depends on available RAM/GPU memory and the dimensionality of the selected model.

### Raw survey file

`darksirens_pixelate` expects a table-like HDF5 survey file with:

| Dataset | Meaning | Units |
| --- | --- | --- |
| `TARGET_RA` | Right ascension | degrees |
| `TARGET_DEC` | Declination | degrees |
| `Z` | Galaxy redshift | dimensionless |
| `ZERR` | Redshift uncertainty | dimensionless |
| `WEIGHT` | Galaxy weight | arbitrary, usually non-negative |

### Pixelated survey file

The pixelation tool writes:

```text
catalog_pixelated_nside_<nside>.h5
```

with:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `zgals` | `(npix, max_galaxies_per_pixel)` | Galaxy redshifts per HEALPix pixel. Empty slots are padded with `100.0`. |
| `dzgals` | `(npix, max_galaxies_per_pixel)` | Redshift uncertainties. Empty slots are padded with `1.0`. |
| `wgals` | `(npix, max_galaxies_per_pixel)` | Galaxy weights. Empty slots are padded with `0.0`. |
| `ngals` | `(npix,)` | Number of real galaxies in each HEALPix pixel. |

The output file also stores the HEALPix NSIDE as an HDF5 attribute named `nside`.

## Pixelating a galaxy survey

Use `darksirens_pixelate` to convert a raw survey file into the dense catalog layout used by dark-siren inference.

```bash
darksirens_pixelate \
  --survey_path data/raw_survey.h5 \
  --save_path data/pixelated \
  --nside 64 \
  --add_plots
```

### Options

| Option | Required | Default | Description |
| --- | --- | --- | --- |
| `--survey_path` | yes | none | Path to the raw survey HDF5 file. |
| `--save_path` | no | `./` | Directory for the pixelated catalog and optional plots. |
| `--nside` | no | `64` | HEALPix NSIDE used to assign galaxies to sky pixels. |
| `--add_plots` | no | false | Save diagnostic plots for sky density, redshift distribution, and pixel occupancy. |

### Outputs

The command writes:

```text
catalog_pixelated_nside_64.h5
```

If `--add_plots` is enabled, it also writes:

```text
skymap_density_nside_64.png
redshift_distribution_nside_64.png
pixel_occupancy_distribution_nside_64.png
```

## Running inference

The main inference command is:

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --sampler dynesty \
  --universe_model spectral_sirens \
  --pop_model powerlaw+peak \
  --save_path runs/example
```

### Universe models

`--universe_model` controls the redshift-prior and host-galaxy model.

| Value | Description |
| --- | --- |
| `spectral_sirens` | GW-only spectral-siren inference. No survey catalog is required. |
| `spectral_sirens_wl` | Spectral sirens with weak-lensing magnification marginalization (no survey catalog). Run via the lensing CLI: `darksirens_inference_lensing --cluster_mode off --wl_backend lognormal\|tabulated` (no longer a `darksirens_inference` choice). |
| `dark_sirens` | Incomplete-catalog dark-siren inference with survey completion modeling. Requires `--survey_path` (one catalog, or K >= 2 for the multitracer mixture). |
| `dark_sirens_complete` | Complete-catalog dark-siren inference. Requires `--survey_path`; a K >= 2 mixture uses the field sky weighting only (special case). |
| `bright_sirens` | Bright-siren inference using known counterpart coordinates/redshifts via `--counterpart`. |

### Population-model names in CLI runs

`--pop_model` accepts either a grammar-built mixture or an explicitly registered bespoke model. Grammar mixtures combine `powerlaw`, `brokenpowerlaw`, and `peak` mass tokens with optional count prefixes (`2peaks`, `3powerlaws`). For example:

```bash
--pop_model powerlaw+peak
--pop_model brokenpowerlaw+2peaks+powerlaw
--pop_model 2powerlaws+3peaks
```

Use `--shared_beta false`, `--shared_spin false`, or `--shared_gamma false` to switch from the default shared beta/pairing, spin, or redshift-evolution parameters to per-component parameters. The first `k - 1` mixture parameters are `$v_i$` stick-breaking coordinates for a `k`-component mixture. The startup parameter table is the authoritative source for CLI JSON labels and bounds; copy labels from that table when using `--prior_overrides` or `--fixed_parameter_values`.

### Spectral-siren workflow

A spectral-siren run uses GW posterior samples and GW selection samples, but no galaxy survey.

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --universe_model spectral_sirens \
  --pop_model powerlaw+peak \
  --sampler dynesty \
  --nlive 2000 \
  --dlogz 0.1 \
  --save_path runs/spectral_powerlaw_peak
```

Use this mode when you want the population model and cosmology to be constrained by GW data alone.

### Dark-siren workflow with an incomplete catalog

First pixelate the survey:

```bash
darksirens_pixelate \
  --survey_path data/raw_survey.h5 \
  --save_path data/pixelated \
  --nside 64
```

Then run inference:

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --survey_path data/pixelated/catalog_pixelated_nside_64.h5 \
  --universe_model dark_sirens \
  --pop_model powerlaw+peak \
  --sampler dynesty \
  --save_path runs/dark_powerlaw_peak
```

The incomplete-catalog model combines catalog galaxies with an additive missing-galaxy density. Completion is estimated from a matched-kernel ratio: observed per-pixel galaxy counts are smoothed on the redshift grid with the same truncated Gaussian operator used for the expected `n0 * dV_c/dz * (1 + z)^delta` counts. The likelihood then adds `(1 - C) * dN_exp/dz`, optionally modulated by `max(1 + alpha_miss * b_miss * delta_g, 0)` when LSS information is enabled. `z50`, `w`, and `alpha_miss` are legacy/degenerate compatibility parameters that are not sampled for `dark_sirens` by default; `z50` and `w` do not impose a logistic rolloff in the current completion model.

### Multitracer dark-siren workflow (K >= 2 catalogs)

Pass multiple pixelated surveys to sample the per-catalog GW host fractions alongside cosmology and population:

```bash
darksirens_inference \
  --gw_path gw_samples.h5 \
  --gwselection_path selection.h5 \
  --universe_model dark_sirens \
  --survey_path galaxies.h5 agn.h5 \
  --pop_model powerlaw+peak \
  --sampler dynesty \
  --save_path runs/dark_gal_agn
```

This K=2 run samples the usual cosmology/population block, TWO survey nuisance blocks (`log10n0`/`delta`/`b_miss`/`sigma_kde` and their `_c2` twins), and the mixture stick `fcat_2`. The sky-weighting convention auto-resolves to `field` (survey-global normalizer) so `fcat_2` **is** catalog 2's GW host fraction (`w_2 = f_agn` for a GAL+AGN pair); `darksirens_analyze` prints the 5/50/95% quantiles of the derived `w_1..w_K` and saves `catalog_weights_<tag>.{pdf,npy}`. Per-catalog `--lss_completion` tables (0 or K paths, `""` placeholders), `--use_lss`, `--lss_marginalize` (equal-M ensembles over matched LSS realizations), `--mark_model` (per-catalog `eta` blocks), and anisotropic `--sky_model` choices all compose; see the CLI reference ("Multitracer catalog mixtures") for the rules.

### Complete-catalog dark-siren workflow

Use:

```bash
--universe_model dark_sirens_complete
```

Example:

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --survey_path data/pixelated/catalog_pixelated_nside_64.h5 \
  --universe_model dark_sirens_complete \
  --pop_model powerlaw+peak \
  --sampler dynesty \
  --complete_empty_pixel_policy zero \
  --save_path runs/dark_complete_powerlaw_peak
```

#### Empty-pixel behavior

`--complete_empty_pixel_policy` accepts:

| Value | Meaning |
| --- | --- |
| `zero` | Formal complete-catalog behavior. Pixels with no real galaxies contribute zero host probability. |
| `volume` | Robustness approximation. Empty pixels fall back to a comoving-volume redshift prior. |

Use `zero` for strict complete-catalog analyses and `volume` for sensitivity checks with sparse or high-resolution catalogs.

### Bright-siren workflow

Bright-siren inference uses known electromagnetic counterparts.

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --universe_model bright_sirens \
  --counterpart RA1 DEC1 Z1 \
  --counterpart_dz 1.0e-4 \
  --counterpart_nside 1 \
  --sampler dynesty \
  --save_path runs/bright_siren
```

For multiple events, pass one `(RA, DEC, Z)` triplet per event in event order:

```bash
--counterpart RA1 DEC1 Z1 RA2 DEC2 Z2 RA3 DEC3 Z3
```

Conventions:

- RA and Dec are in radians.
- Redshift is dimensionless.
- Triplets must follow the same event order as the posterior samples in `--gw_path`.

Use:

```bash
--bright_siren_sky_marginalized true
```

to ignore the counterpart sky-pixel gate and apply only the counterpart redshift prior.

## Sampler options

`--sampler` is required and accepts:

| Sampler | Description |
| --- | --- |
| `tinyns` | Lightweight JAX nested sampler (dynesty-compatible) with a JAX-kernel random-walk proposal and evidence output. TinyNS supports `rwalk`/`prior`; use `dynesty` for slice-style comparisons. Default. |
| `dynesty` | Dynamic/static nested-sampling backend with evidence output. |
| `numpyro` | JAX NUTS (HMC) backend; posterior samples, no evidence. |

Common sampler options:

| Option | Default | Used by | Description |
| --- | --- | --- | --- |
| `--nlive` | `1000` | `tinyns`, `dynesty` | Number of live points. |
| `--dlogz` | `0.1` | `tinyns`, `dynesty` | Evidence stopping threshold. |
| `--max_samples` | `1000000` | `dynesty` call cap, `tinyns` iteration cap | Maximum call/iteration budget (`0` = unlimited). |
| `--tinyns_sample` | `rwalk` | `tinyns` | Proposal method: `rwalk` or `prior` (TinyNS no longer supports `slice`/`rslice`). |
| `--tinyns_kernel` | `jax` | `tinyns` | Proposal kernel: `jax` (jitted) or `python`. |
| `--tinyns_walks` | `25` | `tinyns` | Number of random-walk steps per update (`sample=rwalk`). |
| `--tinyns_replacement_chains` | `1` | `tinyns` | Independent random-walk chains run in parallel per replacement (`rwalk`+`jax` only). |
| `--tinyns_replacement_chain_schedule` | `none` | `tinyns` | Adaptive `rwalk`+`jax` escalation schedule (ascending, e.g. `1,4,16,64,256`); escalates only when a stage fails. Mutually exclusive with `--tinyns_replacement_chains`. |
| `--tinyns_max_attempts` | auto (`10000`) | `tinyns` | Max constrained-proposal attempts per replacement. Must be `>= walks * replacement_chains`; auto-raised to that product when unset. |
| `--tinyns_step_scale` | `0.1` | `tinyns` | Initial proposal step scale (fraction of prior width). |
| `--tinyns_progress_interval` | `100` | `tinyns` | Iterations between progress-bar updates. |
| `--nuts_warmup` | `500` | `numpyro` | Number of NUTS adaptation/warmup steps. |
| `--nuts_samples` | `1000` | `numpyro` | Number of NUTS posterior samples per chain. |
| `--nuts_chains` | `1` | `numpyro` | Number of NUTS chains. |
| `--nuts_target_accept` | `0.8` | `numpyro` | Target acceptance probability for NUTS. |
| `--nuts_max_tree_depth` | `10` | `numpyro` | Maximum NUTS tree depth. |
| `--nuts_init_tries` | `32` | `numpyro` | Number of fallback prior-box candidates tested if midpoint init is non-finite. |
| `--nuts_init_seed_offset` | `100000` | `numpyro` | Added to `--seed` for deterministic fallback-init candidate generation. |
| `--seed` | `22` | all | Random seed. |
| `--show_progress` | `true` | all | Enable progress output where supported. |
| `--dynesty_diagnostics` | `false` | `dynesty` | Write periodic dynesty runplot/traceplot PDF diagnostics under `<save_path>/dynesty_diagnostics/`. |

## Parameter configuration

### Fixing parameter blocks

The following boolean options control whether major parameter blocks are sampled or fixed:

| Option | Default | Description |
| --- | --- | --- |
| `--fix_population` | `false` | Fix population parameters to model defaults. |
| `--fix_cosmology` | `false` | Fix all cosmology parameters (`H0`, `Om0`, `w0`, `wa`). |
| `--fix_de` | `false` | Fix only the CPL dark-energy parameters (`w0`, `wa`), leaving `H0` and `Om0` sampled unless otherwise fixed. |
| `--fix_survey` | `false` | Fix survey/completion parameters. |

Boolean options accept values such as:

```text
true, false, 1, 0, yes, no, t, f, y, n
```

### Prior overrides

Use `--prior_overrides` to change prior bounds for named parameters:

```bash
--prior_overrides '{"H0": [60.0, 80.0], "Om0": [0.2, 0.4], "w0": [-1.2, -0.8], "wa": [-0.5, 0.5]}'
```

### Fixed individual parameters

Use `--fixed_parameter_values` to fix individual parameters:

```bash
--fixed_parameter_values '{"H0": 67.74, "Om0": 0.3075, "w0": -1.0, "wa": 0.0}'
```

Both options must be JSON objects. Quote them carefully in your shell.

Population parameters are addressed by their printed LaTeX labels, e.g.
`'{"$\\alpha_{\\rm PL}$": 2.3}'` for the power-law slope in `powerlaw+peak`.
Run a small dry run first and copy the labels from the printed parameter
table; the label conventions are described in
[Concepts → Population models](concepts.md#population-models). Mixture weights use
stick-breaking labels (`$v_1$`, `$v_2$`, ...) rather than direct final fractions;
convert desired final fractions before fixing them in JSON.

### Survey/completion parameters

Incomplete-catalog dark-siren runs use survey/completion parameters.

| Parameter | Meaning |
| --- | --- |
| `log10n0` | Base-10 logarithm of the comoving galaxy density normalization `n0` in the expected count model. |
| `z50` | Legacy compatibility parameter; retained in settings/priors but not used by the matched-kernel completion likelihood. |
| `w` | Legacy compatibility parameter; retained in settings/priors but not used as a logistic rolloff width. |
| `delta` | Power-law evolution of expected galaxy density, `n(z) = n0 * (1 + z)^delta`. |
| `b_miss` | Bias amplitude for LSS-modulated missing galaxies. |
| `alpha_miss` | Multiplies `b_miss` exactly; the model depends on `alpha_miss * b_miss`, so these two labels are degenerate and the default fixed value is `alpha_miss = 1`. |
| `sigma_kde` | Extra catalog-redshift KDE broadening, added in quadrature to catalog redshift errors. |

Before a long dark-siren run, validate the completion model:

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --survey_path data/pixelated/catalog_pixelated_nside_64.h5 \
  --universe_model dark_sirens \
  --sampler dynesty \
  --validate_completion true \
  --completion_validation_pixels 64 \
  --save_path runs/completion_validation
```

This loads the data, builds the per-pixel observed-galaxy KDE cache, computes completion clipping diagnostics, writes a file like:

```text
completion_validation__YYYY-MM-DDTHH-MM-SS.json
```

and exits before likelihood construction and sampling. The JSON includes the representative survey/cosmology values used for the dry run plus per-pixel and summary clipping fractions for `C_iso`, `C_eff`, and the LSS missing-density factor.

## Performance tuning

### Selection batching

For large gwcat selection files:

```bash
--sel_batch_size 200000
```

Decrease this value if the run exceeds memory limits.

### Population normalization grids

Population normalization grids can be controlled with:

```bash
--norm_nmass 2000
--norm_nq 1000
--norm_nchi 1000
```

or with environment variables:

```text
DARKSIRENS_GW_N_MASS
DARKSIRENS_GW_N_Q
DARKSIRENS_GW_N_CHI
```

Use higher-resolution grids for production runs with narrow mass, mass-ratio, or spin features. For final evidence checks, compare against a higher-resolution rerun and confirm that posterior/evidence changes are negligible.

## Inference outputs

Each inference run creates a timestamped directory under `--save_path`:

```text
<save_path>/<pop_model>__<universe_model>__<sampler>__seed<seed>__YYYY-MM-DDTHH-MM-SS/
```

If that name is already taken (e.g. two identically-configured jobs started
in the same second), a `-01`, `-02`, ... suffix is appended instead of
reusing the existing directory.

Expected contents include:

```text
results.hdf5
settings.json
corner.pdf
```

`results.hdf5` contains:

| Dataset / attribute | Meaning |
| --- | --- |
| `samples` | Posterior samples with shape `(N_samples, N_dim)`. |
| `labels` | Parameter labels corresponding to sample columns. |
| `lower_bound` | Prior lower bounds. |
| `upper_bound` | Prior upper bounds. |
| `fixed_labels` | Labels of individually fixed parameters, when present. |
| `fixed_values` | Values of individually fixed parameters, when present. |
| `log_weights` | Optional per-sample log weights, when available. |
| `log_likelihood` | Optional per-sample log likelihoods, when available. |
| HDF5 attributes | Model name, sampler, paths, runtime, evidence, environment metadata, and run settings. |

`settings.json` is a human-readable record of CLI options, parameter labels, bounds, metadata, and environment information.

## Analyzing inference outputs

Run:

```bash
darksirens_analyze --run_dirs runs/spectral_powerlaw_peak \
  --mmin 1 \
  --mmax 100 \
  --nm 300 \
  --nq 100 \
  --nz 50 \
  --nchi 50
```

Important options:

| Option | Default | Description |
| --- | --- | --- |
| `--run_dirs` | required | One or more directories containing saved inference products. |
| `--mmin` | `1.0` | Minimum mass grid value. |
| `--mmax` | `100.0` | Maximum mass grid value. |
| `--nm` | `300` | Number of mass grid points. |
| `--nq` | `100` | Number of mass-ratio grid points. |
| `--nz` | `50` | Number of redshift grid points. |
| `--nchi` | `50` | Number of effective-spin grid points. |
| `--chimin` | `-1.0` | Minimum effective-spin grid value. |
| `--chimax` | `1.0` | Maximum effective-spin grid value. |
| `--batch_size` | auto | Posterior-predictive batch size. |
| `--cred_lo` | `5.0` | Lower credible interval percentile. |
| `--cred_hi` | `95.0` | Upper credible interval percentile. |

The analyzer reads current `results.hdf5` runs (including root-level or grouped posterior sample datasets), still supports legacy `samples.npy` runs, and produces posterior-predictive plots such as:

```text
pm1_all_models.pdf
pm2_all_models.pdf
pq_all_models.pdf
pz_all_models.pdf
pchi_all_models.pdf
posterior_parameter_corner.pdf
ppd_median_means.pdf
model_evidences.pdf
bayes_factors_pairwise.pdf
```

## Recommended production checklist

Before launching a large run:

- Confirm the GW posterior file contains the required attributes and datasets.
- Confirm the gwcat selection file matches the selection assumptions for the analysis.
- Pixelate the survey at a HEALPix resolution compatible with the GW sky localization data.
- Run `--validate_completion true` for incomplete-catalog dark-siren analyses.
- Run a small fixed-parameter smoke test.
- Inspect the printed parameter table.
- Choose sampler settings appropriate for the dimensionality of the model.
- Save each model comparison run to a separate directory.
- Archive `settings.json`, `results.hdf5`, logs, and exact input data versions.

## Troubleshooting

### Missing survey path

If using:

```bash
--universe_model dark_sirens
```

or:

```bash
--universe_model dark_sirens_complete
```

make sure to pass:

```bash
--survey_path data/pixelated/catalog_pixelated_nside_64.h5
```

### Invalid JSON

If `--prior_overrides` or `--fixed_parameter_values` fails, validate that the argument is a JSON object:

```bash
--prior_overrides '{"H0": [60.0, 80.0]}'
```

Use single quotes around the whole JSON string in most POSIX shells.

### Out-of-memory errors

Try:

```bash
--sel_batch_size 50000
```

or lower the population normalization grids for a smoke test.

### Sparse complete catalogs

For strict complete-catalog analyses, use:

```bash
--complete_empty_pixel_policy zero
```

For robustness tests with sparse pixelation, compare against:

```bash
--complete_empty_pixel_policy volume
```

### Slow posterior-predictive analysis

Reduce grid sizes during testing:

```bash
darksirens_analyze --run_dirs RUN_DIR \
  --nm 100 \
  --nq 50 \
  --nz 25 \
  --nchi 25
```

Increase them again for final figures.
