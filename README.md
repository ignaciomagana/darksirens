# darksirens

`darksirens` is a Python package for joint gravitational-wave inference with large-scale galaxy surveys. It provides command-line tools for spectral-siren and dark-siren hierarchical inference, survey pixelation, and posterior-predictive analysis.

## Documentation

Hosted documentation can be built with Sphinx and published on Read the Docs using the included `.readthedocs.yaml` configuration.

Build the docs locally:

```bash
python -m pip install -r docs/requirements.txt
make docs-html
```

For a stricter pre-publish check that treats warnings as errors while continuing through all warnings, run:

```bash
make docs-strict
```

Start with the documentation source at [`docs/source/index.md`](docs/source/index.md), or see the quickstart guide at [`docs/source/quickstart.md`](docs/source/quickstart.md).


## Cosmology model

The inference cosmology block samples a flat CPL dark-energy model with labels `H0`, `Om0`, `w0`, and `wa`. The default dark-energy point `w0=-1` and `wa=0` recovers flat ΛCDM. Use `--fix_cosmology true` to hold all four cosmology labels fixed, `--fix_de true` to hold only `w0` and `wa` fixed, and `--prior_overrides`/`--fixed_parameter_values` to narrow or pin individual labels.

## Command-line tools

Installing the package exposes:

- `darksirens_pixelate` — convert a raw galaxy survey HDF5 file into a pixelated HEALPix catalog.
- `darksirens_inference` — run spectral-siren or dark-siren hierarchical inference.
- `darksirens_analyze` — analyze saved inference products and posterior-predictive distributions.
- `darksirens_skymaps_to_samples` — convert a directory of 3D skymap FITS files into GW posterior-like samples (`gwdata.h5`) with broad uninformative mass/spin surrogates for low-latency runs.

## Population and completion models

Population mixtures are selected with `--pop_model` using a compositional name grammar. Tokens such as `powerlaw`, `brokenpowerlaw`, and `peak` can be combined directly (`powerlaw+peak`, `brokenpowerlaw+2peaks`, `2powerlaws+3peaks`), with curated names receiving physics-tuned priors and arbitrary grammar combinations using blueprint defaults. Use `--shared_beta`, `--shared_spin`, and `--shared_gamma` to choose shared (default) versus per-component pairing, spin, and redshift-evolution parameters. Mixture weights are sampled as stick-breaking parameters labeled `$v_i$`; copy the printed startup parameter table when passing population labels to `--prior_overrides` or `--fixed_parameter_values`.

Incomplete-catalog dark-siren runs use a data-driven completion model rather than a parametric logistic rolloff: the observed per-pixel galaxy redshift KDE is divided by the identically smoothed expected `n0 * dV_c/dz * (1 + z)^delta` density, clipped to `[0, 1]`, and converted into an additive missing-galaxy density. `z50` and `w` remain in the survey parameter block for compatibility, but the current completion likelihood is controlled by `log10n0`, `delta`, `b_miss` (with fixed `alpha_miss = 1` unless overridden), and `sigma_kde`. Run `--validate_completion true` for dry-run clipping diagnostics before long dark-siren analyses.

## Minimal installation

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

`requirements.txt` installs `gwcat` from a pinned `ignaciomagana/gwcat` Git commit. Keep `gwcat` external rather than vendoring it into this repository: `gwcat` owns preprocessing of raw GW PE and selection/injection products, while `darksirens` consumes the resulting HDF5 catalogs for inference. Additional sampler-specific packages such as `dynesty` or `emcee` may be required for the workflows you choose.
