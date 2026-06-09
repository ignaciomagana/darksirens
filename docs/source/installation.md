# Installation

## Requirements

`darksirens` targets Python 3.11 or newer. The core scientific stack includes JAX, NumPy, SciPy, Matplotlib, Healpy, and HDF5 support. GW posterior and selection inputs are preprocessed by `gwcat`, which is installed from the pinned Git commit recorded in `requirements.txt` because the intended `ignaciomagana/gwcat` package is not consumed from PyPI here. Some samplers and plotting paths use optional packages such as `dynesty`, `emcee`, `corner`, and `arviz` depending on the workflow you run.

## Install from a local checkout

```bash
git clone https://github.com/ignaciomagana/darksirens.git
cd darksirens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

If you use optional sampler/post-processing packages outside `requirements.txt`, install them in the same environment:

```bash
python -m pip install dynesty emcee numpyro corner arviz h5py tqdm
```

## JAX notes

The inference entry point configures JAX for high precision and conservative memory allocation before importing JAX. If you run on GPU or TPU hardware, install the JAX wheel that matches your accelerator stack using the official JAX installation instructions for your platform.

## Build the documentation locally

```bash
python -m pip install -r docs/requirements.txt
make docs-html
python -m http.server --directory docs/_build/html 8000
```

Then open <http://localhost:8000> in a browser. Before publishing, maintainers can run the stricter documentation check:

```bash
make docs-strict
```

## Hosted documentation

The repository includes a `.readthedocs.yaml` configuration. On Read the Docs, create a project pointing at the repository and keep the default build command; Read the Docs will install the package plus the documentation requirements and build `docs/source/conf.py`.
