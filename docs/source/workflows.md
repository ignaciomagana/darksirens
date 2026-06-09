# Workflows

## Spectral-siren production run

1. Validate the GW posterior and selection HDF5 files.
2. Choose a population model.
3. Run a nested sampler with enough live points for the dimensionality of your parameter space.
4. Inspect the saved corner plot and evidence summary.
5. Use `darksirens_analyze` to compare posterior-predictive distributions across models.

Template:

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --universe_model spectral_sirens \
  --pop_model powerlaw+peak \
  --sampler dynesty \
  --nlive 2000 \
  --save_path runs/spectral_powerlaw_peak
```

## Dark-siren catalog run

1. Pixelate the raw survey at the NSIDE resolution used by the GW sky localization products.
2. Run inference with `--survey_path` and `--universe_model dark_sirens`.
3. Compare against `dark_sirens_complete` to understand sensitivity to incompleteness modeling.
4. Tune survey priors only after validating the catalog format.

Template:

```bash
darksirens_pixelate \
  --survey_path data/raw_survey.h5 \
  --save_path data/pixelated \
  --nside 64

darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --survey_path data/pixelated/catalog_pixelated_nside_64.h5 \
  --universe_model dark_sirens \
  --pop_model brokenpowerlaw+2peaks \
  --sampler dynesty \
  --save_path runs/dark_bpl_2peaks
```

## Fixed-parameter validation run

Before launching a large production job, fix expensive or well-understood parameter blocks and verify that data loading, selection corrections, and output writing complete successfully.

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/selection.h5 \
  --sampler emcee \
  --nwalkers 32 \
  --nsteps 200 \
  --fix_de true \
  --fix_survey true \
  --fixed_parameter_values '{"H0": 67.74, "Om0": 0.3075}' \
  --save_path runs/smoke_test
```

## Model-comparison run set

Use separate output directories for each model and keep the same input data, sampler settings, and random seed where practical.

```text
runs/
  spectral_powerlaw_peak/
  spectral_brokenpowerlaw_2peaks/
  dark_powerlaw_peak/
  dark_complete_powerlaw_peak/
```

Then run the analyzer on each directory and compare evidence estimates and Bayes factors.
