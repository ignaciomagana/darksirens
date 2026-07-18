# darksirens documentation

`darksirens` is a Python package for hierarchical cosmological inference with gravitational-wave events and, when available, large-scale galaxy surveys. The command-line tools support spectral-siren and dark-siren workflows, including survey pixelation, posterior sampling, and posterior-predictive analysis.

```{toctree}
:maxdepth: 2
:caption: User guide

installation
quickstart
concepts
theory
data-formats
cli
usage-detailed
workflows
mock-data
testing
configuration
performance
troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/api
```

```{toctree}
:maxdepth: 1
:caption: Project

contributing
changelog
refactor-migration
```

## What is included?

- **GW data loading** for event posterior samples and gwcat selection samples.
- **EM survey handling** for pixelated HEALPix galaxy catalogs.
- **Redshift priors** for spectral sirens, complete-catalog dark sirens, and incomplete-catalog dark sirens.
- **Population models** composed directly from the `--pop_model` name (e.g. `brokenpowerlaw+2peaks`): parametric components combine through a naming grammar, with Gaussian-process variants and curated physics-tuned priors for standard models.
- **Sampling front end** for `tinyns`, `dynesty`, and `numpyro`.
- **Analysis utilities** for evidences, Bayes factors, and posterior-predictive mass/redshift distributions.

## Documentation status

The hosted documentation is designed for Read the Docs or any Sphinx-compatible static documentation service. It intentionally keeps examples data-light: replace file paths with your own GW posterior, injection, and survey products.
