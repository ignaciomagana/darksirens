# Command-line interface

Installing the package exposes three console scripts.

## `darksirens_pixelate`

Convert a raw galaxy survey HDF5 file into the dense HEALPix layout used by dark-siren inference.

```bash
darksirens_pixelate --survey_path SURVEY.h5 [--save_path OUTDIR] [--nside 64] [--add_plots]
```

Options:

- `--survey_path`: required path to the raw HDF5 survey file.
- `--save_path`: output directory; defaults to the current directory.
- `--nside`: HEALPix NSIDE; defaults to `64`.
- `--add_plots`: create diagnostic skymap, redshift, and occupancy plots.

## `darksirens_inference`

Run hierarchical inference.

```bash
darksirens_inference \
  --gw_path GW.h5 \
  --gwselection_path INJECTIONS.h5 \
  --sampler dynesty \
  [options]
```

### Data options

- `--gw_path`: required GW posterior-sample HDF5 file.
- `--gwselection_path`: required gwcat selection HDF5 file.
- `--survey_path`: pixelated survey HDF5 file; required for dark-siren models.
- `--counterpart RA1 DEC1 Z1 [RA2 DEC2 Z2 ...]`: bright-siren counterpart coordinates and redshifts, one triplet per GW event in event order; RA and Dec use radians, matching the GW sample convention. Required for `--universe_model bright_sirens`.
- `--counterpart_dz`: Gaussian redshift uncertainty assigned to the synthetic counterpart catalog entry; defaults to `1e-4`.
- `--counterpart_nside`: HEALPix NSIDE for the synthetic counterpart catalog; defaults to `1`.
- `--save_path`: directory for settings, samples, plots, and summaries.

### Physical-model options

- `--universe_model`: one of `spectral_sirens`, `bright_sirens`, `dark_sirens`, or `dark_sirens_complete`.
- `--pop_model`: population model name, for example `powerlaw+peak`.
- `--fix_population`: fix all population parameters to fiducial values.
- `--fix_cosmology`: fix all cosmological parameters (`H0`, `Om0`, `w0`, `wa`) to fiducial values.
- `--fix_de`: fix only the CPL dark-energy parameters (`w0=-1`, `wa=0`) while leaving `H0` and `Om0` available unless fixed separately.
- `--fix_survey`: fix survey-completion parameters to fiducial values.
- `--prior_overrides`: JSON object mapping parameter names to `[lower, upper]` prior bounds, e.g. `{"H0": [60, 80], "w0": [-1.2, -0.8], "wa": [-0.5, 0.5]}`.
- `--fixed_parameter_values`: JSON object mapping parameter names to fixed scalar values, e.g. `{"Om0": 0.3075, "w0": -1.0, "wa": 0.0}`.
- `--bright_siren_sky_marginalized BOOL`: for `bright_sirens`, ignore the counterpart sky-pixel gate and apply only the counterpart redshift prior. Defaults to `False`. Accepted true values are `true`, `t`, `1`, `yes`, and `y`; accepted false values are `false`, `f`, `0`, `no`, and `n` (case-insensitive).
- `--complete_empty_pixel_policy {zero,volume}`: controls genuinely empty pixels for `dark_sirens_complete` and `bright_sirens`. `zero` is the formal default and returns zero probability (`-inf` log-prior) when `ngals == 0`; `volume` uses the comoving-volume prior as a robustness approximation for sparse pixelations.

### Catalog options

- `--use_LSS`: include large-scale-structure overdensity where supported.
- `--validate_completion`: run a dry-run completion clipping diagnostic, save `completion_validation__*.json` under `--save_path`, and exit before likelihood construction or sampling.
- `--completion_validation_pixels`: maximum number of unique catalog pixels to inspect during `--validate_completion`; defaults to `64`.

### Sampler options

- `--sampler`: required; one of `jaxns`, `dynesty`, `emcee`, or `numpyro`.
- `--nlive`: live points for nested samplers.
- `--dlogz`: evidence stopping threshold where supported.
- `--max_samples`: maximum samples for samplers that expose this limit.
- `--nwalkers`: number of walkers for `emcee`.
- `--nsteps`: number of steps for `emcee`.
- `--nuts_warmup`: NUTS warmup/adaptation steps for `numpyro`.
- `--nuts_samples`: NUTS posterior samples per chain for `numpyro`.
- `--nuts_chains`: number of NUTS chains for `numpyro`.
- `--nuts_target_accept`: target acceptance probability for `numpyro` NUTS.
- `--nuts_max_tree_depth`: maximum NUTS tree depth for `numpyro`.
- `--nuts_init_tries`: fallback initial-point draws from priors when midpoint has non-finite likelihood.
- `--nuts_init_seed_offset`: seed offset used for the NUTS fallback initial-point search.
- `--seed`: random seed.
- `--show_progress`: enable or disable progress bars.

### Performance options

- `--sel_batch_size`: optional injection-selection batch size.
- `--norm_nmass`, `--norm_nq`, `--norm_nchi`: mass, mass-ratio, and spin grid sizes used for GW-population normalization quadrature. They default to `500`, `200`, and `200`, respectively, and can also be set with `DARKSIRENS_GW_N_MASS`, `DARKSIRENS_GW_N_Q`, and `DARKSIRENS_GW_N_CHI`. The inference command prints the active values and saves them in `settings.json` under `normalization_grid`.

## `darksirens_analyze`

Analyze saved inference products and compute posterior-predictive summaries. The analyzer reads the current `results.hdf5` output format and still supports legacy `samples.npy` runs.

```bash
darksirens_analyze [--run_dirs RUN_DIR [RUN_DIR ...]] [--mmin 1] [--mmax 100] [--nm 300]
```

Important options:

- `--run_dirs`: optional list of one or more directories produced by `darksirens_inference`; when omitted, the current directory is analyzed.
- `--mmin`, `--mmax`, `--nm`: primary-mass grid bounds and size.
- `--nq`: mass-ratio grid size.
- `--nz`: redshift grid size.
- `--nchi`, `--chimin`, `--chimax`: spin grid configuration.
- `--batch_size`: posterior-predictive evaluation batch size.
- `--cred_lo`, `--cred_hi`: lower and upper credible intervals.
