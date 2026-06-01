# Quickstart

This page shows the smallest end-to-end workflow. The commands are templates: replace the HDF5 paths with products from your own analyses.

## 1. Prepare input data

A typical run needs:

1. GW posterior samples for one or more events.
2. GW selection/injection samples.
3. A pixelated galaxy catalog for dark-siren analyses.

For spectral sirens, only the GW posterior and selection files are required.

## 2. Pixelate a galaxy survey

If your survey is stored as a table-like HDF5 file with right ascension, declination, redshift, redshift error, and weight columns, convert it into the dense HEALPix format expected by the inference code:

```bash
darksirens_pixelate \
  --survey_path data/raw_survey.h5 \
  --save_path data/pixelated \
  --nside 64 \
  --add_plots
```

The command writes `catalog_pixelated_nside_64.h5` plus optional diagnostic plots.

## 3. Run spectral-siren inference

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/injections.h5 \
  --sampler dynesty \
  --pop_model powerlaw+peak \
  --universe_model spectral_sirens \
  --nlive 2000 \
  --save_path runs/spectral_powerlaw_peak
```

## 4. Run dark-siren inference

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/injections.h5 \
  --survey_path data/pixelated/catalog_pixelated_nside_64.h5 \
  --sampler dynesty \
  --pop_model powerlaw+peak \
  --universe_model dark_sirens \
  --save_path runs/dark_powerlaw_peak
```

## 5. Fix or override parameters

Use JSON objects for prior overrides and fixed parameter values. Quote JSON carefully in your shell:

```bash
darksirens_inference \
  --gw_path data/gw_events.h5 \
  --gwselection_path data/injections.h5 \
  --sampler dynesty \
  --universe_model spectral_sirens \
  --prior_overrides '{"H0": [60.0, 80.0]}' \
  --fixed_parameter_values '{"Om0": 0.3075}'
```

## 6. Analyze a run

```bash
darksirens_analyze --run_dirs runs/spectral_powerlaw_peak \
  --mmin 1 \
  --mmax 100 \
  --nm 300 \
  --nq 100 \
  --nz 50
```

The analyzer loads saved run products, summarizes evidences, and can generate posterior-predictive distributions on configurable mass, mass-ratio, redshift, and spin grids.
