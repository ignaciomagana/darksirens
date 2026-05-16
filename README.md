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

## Command-line tools

Installing the package exposes:

- `darksirens_pixelate` — convert a raw galaxy survey HDF5 file into a pixelated HEALPix catalog.
- `darksirens_inference` — run spectral-siren or dark-siren hierarchical inference.
- `darksirens_analyze` — analyze saved inference products and posterior-predictive distributions.

## Minimal installation

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

Additional sampler-specific packages such as `dynesty` or `emcee` may be required for the workflows you choose.
