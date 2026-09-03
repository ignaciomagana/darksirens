# darksirens

Hierarchical Bayesian inference of cosmology and the compact-binary population
from gravitational-wave events, with or without galaxy catalogs. `darksirens`
implements spectral sirens, dark sirens with complete or incomplete galaxy
catalogs, bright sirens with electromagnetic counterparts, and gravitationally
lensed sirens, behind nine command-line programs built on JAX.

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} Get started
:link: getting-started/quickstart
:link-type: doc

Install the package and run a complete mock analysis in a few minutes:
generate data, pixelate a survey, run inference, analyze the posterior.
:::

:::{grid-item-card} User guide
:link: guide/inference
:link-type: doc

How each analysis is configured and what it computes: universe models,
population models, galaxy catalogs, lensing, performance, testing.
:::

:::{grid-item-card} Reference
:link: reference/cli
:link-type: doc

Every command-line option, the Python API generated from the source, and the
theory behind the likelihood.
:::

::::

## What it computes

The likelihood is the standard hierarchical population likelihood with a
Monte-Carlo selection correction: each event's posterior samples are reweighted
from their parameter-estimation prior to the population and redshift prior
implied by the sampled hyperparameters, and the expected number of detections
is estimated from a set of found injections. Sampled blocks are the flat-CPL
cosmology (`H0`, `Om0`, `w0`, `wa`), the galaxy-survey completeness block, and
the population hyperparameters of a `--pop_model` such as `powerlaw+peak`.

| Universe model | Redshift information | Inputs |
| --- | --- | --- |
| `spectral_sirens` | none beyond the GW data (comoving-volume prior) | GW posteriors, selection injections |
| `dark_sirens` | an incomplete galaxy catalog plus a missing-galaxy model | + pixelated catalog |
| `dark_sirens_complete` | a catalog treated as complete | + pixelated catalog |
| `bright_sirens` | electromagnetic counterparts | + counterpart positions and redshifts |
| `spectral_sirens_wl` (lensing CLI) | weak-lensing magnification, optional strongly lensed image pairs | + lensed injections, candidate pairs |

```{toctree}
:maxdepth: 1
:caption: Getting started
:hidden:

getting-started/installation
getting-started/quickstart
getting-started/inputs
```

```{toctree}
:maxdepth: 1
:caption: User guide
:hidden:

guide/inference
guide/populations
guide/catalogs
guide/lensing
guide/analysis
guide/performance
guide/testing
guide/troubleshooting
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

reference/cli
reference/api/index
reference/theory
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

about/contributing
about/changelog
```
