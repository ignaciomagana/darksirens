# darksirens

Hierarchical Bayesian inference of cosmology and the compact-binary population
from gravitational-wave events, with or without galaxy catalogs.

`darksirens` runs the standard population likelihood with a Monte-Carlo
selection correction on JAX, and covers four ways of getting redshift
information into it:

| Analysis | `--universe_model` | Needs |
| --- | --- | --- |
| Spectral sirens | `spectral_sirens` | GW posteriors, found injections |
| Dark sirens, incomplete catalog | `dark_sirens` | + a pixelated galaxy catalog |
| Dark sirens, complete catalog | `dark_sirens_complete` | + a pixelated galaxy catalog |
| Bright sirens | `bright_sirens` | + counterpart positions and redshifts |
| Lensed sirens | `darksirens_inference_lensing` | + lensed injections, candidate image pairs |

Sampled blocks: the flat-CPL cosmology (`H0`, `Om0`, `w0`, `wa`), the survey
completeness block, and the hyperparameters of a population model named by a
grammar such as `powerlaw+peak` or `brokenpowerlaw+2peaks`. Samplers:
`tinyns`, `dynesty`, `numpyro` (`--sampler` is required).

**Documentation:** <https://darksirens.readthedocs.io> (source in `docs/source`).

## Install

Python 3.11 or newer.

```bash
git clone https://github.com/ignaciomagana/darksirens.git
cd darksirens
python -m pip install -e .
python -m pip install -r requirements.txt   # pins gwcat and tinyns from GitHub
darksirens_inference --help
```

Optional extras: `pip install -e ".[gp]"` for Gaussian-process population
models, `".[flows]"` for normalizing-flow single-event surrogates. For a GPU,
install the JAX wheel for your CUDA stack first (see the JAX install guide);
the package requires `jax>=0.4.34`.

## Run something

Generate a small mock, then run and analyze a dark-siren inference on it:

```bash
python scripts/mock_dark_sirens/generate_mock_data.py --outdir data/smoke --nobs 20 --nsamp 500 --nside 8

darksirens_inference \
  --gw_path data/smoke/mock_gw_events.h5 \
  --gwselection_path data/smoke/mock_gw_selection.h5 \
  --survey_path data/smoke/catalog_pixelated_nside_8.h5 \
  --universe_model dark_sirens --pop_model powerlaw+peak \
  --fix_cosmology true --fix_survey true \
  --sampler dynesty --nlive 60 \
  --save_path runs/smoke

darksirens_analyze --run_dirs runs/smoke
```

The full walkthrough, with what each output file contains, is the
[quickstart](docs/source/getting-started/quickstart.md).

## Command-line programs

| Program | Purpose |
| --- | --- |
| `darksirens_inference` | spectral-, dark- and bright-siren inference |
| `darksirens_inference_lensing` | weak-lensing magnification and strongly lensed image pairs |
| `darksirens_analyze` | evidences, Bayes factors, posterior-predictive distributions |
| `darksirens_pixelate` | raw galaxy survey HDF5 to a HEALPix-pixelated catalog |
| `darksirens_skymaps_to_samples` | 3-D skymap FITS files to posterior-like samples |
| `darksirens_build_lognormal_completion` | LSS-conditioned completion field for one catalog |
| `darksirens_build_joint_lognormal_completion` | one latent LSS field for K catalogs |
| `darksirens_diagnose_lognormal_completion` | per-pixel completion diagnostics |
| `darksirens_fit_selection` | magnitude-limited selection fits for a catalog |

Each is also runnable as `python -m darksirens.cli.<name>`. Every option is
listed in the [CLI reference](docs/source/reference/cli.md).

## Inputs

GW posterior samples and found injections are HDF5 exports written by
[gwcat](https://github.com/ignaciomagana/gwcat) (`gwcat-1.0`,
`gwcat-selection-1.0`). Galaxy catalogs are pixelated once with
`darksirens_pixelate`. The [inputs page](docs/source/getting-started/inputs.md)
lists every dataset the loaders read.

## Development

```bash
JAX_PLATFORMS=cpu python -m pytest $(grep -v '^#' tests/fast_subset.txt) -q   # Tier-0, what CI runs
JAX_PLATFORMS=cpu python -m pytest tests/test_unified_k1_golden.py -q         # likelihood golden gate
make docs-strict                                                              # docs, warnings as errors
```

See [Testing](docs/source/guide/testing.md) and
[Contributing](docs/source/about/contributing.md).

## License

MIT. See `LICENSE`.
