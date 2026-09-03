# Quickstart

This page runs the whole pipeline once, end to end, on data the repository
generates for itself: a small dark-siren mock, then a spectral-siren fit, a
dark-siren fit, and post-processing. The sizes below are the ones the
repository's Tier-1 smoke tests use (see [Testing](../guide/testing.md)), so
every step runs on a laptop CPU. Generating the mock is quick; the two
nested-sampling fits dominate the wall time, and `--show_progress true`
(the default) prints the sampler's progress while they run.

Pick a scratch directory first; HDF5 products are git-ignored.

```bash
DATA=data/quickstart
```

## 1. Generate the mock

```bash
python scripts/mock_dark_sirens/generate_mock_data.py \
    --outdir $DATA \
    --seed 1 \
    --n-galaxies 3000 \
    --nobs 8 \
    --nsamp 256 \
    --ndraw 20000 \
    --nside 8 \
    --zmax 0.1
```

This draws 3000 galaxies uniform in comoving volume out to `z = 0.1`, applies
an EM survey selection to them, draws 8 detected GW events from the complete
catalog with 256 posterior samples each, and proposes 20000 selection
injections (the detected subset is what gets stored). It writes five files:

| File | Consumed by |
| --- | --- |
| `mock_galaxy_catalog_complete.h5` | Nothing in the pipeline; the generator's own record of the pre-selection catalog, including true redshifts |
| `mock_survey_raw.h5` | `darksirens_pixelate --survey_path` |
| `catalog_pixelated_nside_8.h5` | `darksirens_inference --survey_path` |
| `mock_gw_events.h5` | `darksirens_inference --gw_path` |
| `mock_gw_selection.h5` | `darksirens_inference --gwselection_path` |

Use `--n0` instead of `--n-galaxies` to set a physical comoving galaxy density;
it scales with volume and makes the dark-siren precompute much slower than
this walkthrough needs.

## 2. Pixelate the survey

The generator already wrote `catalog_pixelated_nside_8.h5`, so you can skip
straight to step 3. The command below is the step you would run on a real
survey table, and it rebuilds the same catalog from `mock_survey_raw.h5`:

```bash
darksirens_pixelate \
    --survey_path $DATA/mock_survey_raw.h5 \
    --save_path $DATA/pixelated \
    --nside 8 \
    --add_plots
```

It bins the galaxies into HEALPix pixels and writes the dense padded catalog
`catalog_pixelated_nside_8.h5` under `--save_path`, plus three diagnostic PNGs
(sky density, redshift distribution, pixel occupancy) because of
`--add_plots`. The dataset layout is in [Input files](inputs.md).

## 3. Run a spectral-siren inference

Spectral sirens use the GW data alone: no catalog, the redshifts come from the
mass spectrum.

```bash
darksirens_inference \
    --gw_path $DATA/mock_gw_events.h5 \
    --gwselection_path $DATA/mock_gw_selection.h5 \
    --universe_model spectral_sirens \
    --pop_model powerlaw+peak \
    --fix_cosmology true \
    --fix_survey true \
    --sampler dynesty \
    --nlive 60 \
    --save_path out/spectral
```

`--fix_cosmology true` fixes the whole cosmology block (`H0`, `Om0`, `w0`,
`wa`) and `--fix_survey true` fixes the survey hyperparameters, which leaves
only the population parameters sampled: that is what makes 60 live points
enough for a first look. Drop both flags for a real run and raise `--nlive`.

## 4. Run a dark-siren inference

Adding `--survey_path` and `--universe_model dark_sirens` switches on the
galaxy-catalog redshift prior, with the incompleteness model on top.

```bash
darksirens_inference \
    --gw_path $DATA/mock_gw_events.h5 \
    --gwselection_path $DATA/mock_gw_selection.h5 \
    --survey_path $DATA/catalog_pixelated_nside_8.h5 \
    --universe_model dark_sirens \
    --pop_model powerlaw+peak \
    --fix_cosmology true \
    --fix_survey true \
    --sampler dynesty \
    --nlive 60 \
    --save_path out/dark
```

## 5. What a run writes

`--save_path` is a parent directory, not the run directory. Each run creates
one directory under it named
`<pop_model>__<universe_model>__<sampler>__seed<seed>__<timestamp>`, so
repeated runs accumulate side by side instead of overwriting each other:

```text
out/spectral/powerlaw+peak__spectral_sirens__dynesty__seed22__<timestamp>/
├── settings.json          every resolved option, written BEFORE sampling
├── run_fingerprint.json   the run's semantic identity, checked by --resume
├── checkpoint.dynesty.pkl sampler state, rewritten every --checkpoint_interval
├── samples.npy            the numeric chain, written before results.hdf5
├── results.hdf5           samples, labels, prior bounds, evidence, metadata
├── corner.pdf             corner plot of the sampled parameters
└── failure.json           only if the run died, with the stage and traceback
```

`results.hdf5` is the canonical output: `samples` `(N, ndim)`, `labels`,
`lower_bound`, `upper_bound`, the individually fixed labels and values, and
log weights / log likelihoods when the sampler provides them.

## 6. Post-process

```bash
darksirens_analyze \
    --run_dirs out/spectral/powerlaw+peak__spectral_sirens__dynesty__seed22__* \
               out/dark/powerlaw+peak__dark_sirens__dynesty__seed22__* \
    --outdir out/figures
```

`darksirens_analyze` reads each run's `results.hdf5` (falling back to
`samples.npy`), recomputes the posterior-predictive population
distributions on a mass / mass-ratio / redshift / spin grid, and writes the
figures plus a relative-evidence and pairwise-Bayes-factor comparison of the
runs into `--outdir`. Grid sizes are set with `--mmin`, `--mmax`, `--nm`,
`--nq`, `--nz`, `--nchi`.

## Next steps

- [Inference](../guide/inference.md): the models, the fix/override/prior JSON,
  the samplers, checkpoint and resume.
- [Galaxy catalogs](../guide/catalogs.md): completeness, LSS completion,
  marks, multitracer mixtures.
- [Lensing](../guide/lensing.md): magnification and strongly lensed pairs.
