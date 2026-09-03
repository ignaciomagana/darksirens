# Installation

This page tells you how to install `darksirens` from a checkout and verify it.

## Requirements

`darksirens` requires Python 3.11 or newer. Its runtime dependencies live in
`requirements.txt`, which `setup.py` reads into `install_requires`, so
`pip install .` enforces them: `jax`/`jaxlib` (>= 0.4.34), `numpy>=1.26,<3`,
`scipy>=1.11`, `matplotlib`, `healpy`, `h5py`, `astropy`, `tqdm`, `dynesty`,
`numpyro`, `seaborn`, plus the two pinned Git dependencies below.

## Install from a checkout

```bash
git clone https://github.com/ignaciomagana/darksirens.git
cd darksirens
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
```

## The two pinned Git dependencies

Two dependencies are installed from GitHub at an exact commit rather than from
PyPI: they are separate projects that are not published there, and their file
schemas are part of this package's input contract.

| Package | Role | Pin |
| --- | --- | --- |
| `gwcat` | Writes the GW posterior-sample and selection-injection inputs (`--gw_path`, `--gwselection_path`). | `git+https://github.com/ignaciomagana/gwcat.git@3184bdb` |
| `tinyns` | The nested sampler behind `--sampler tinyns`. | `git+https://github.com/ignaciomagana/tinyns.git@3f9e1b2` |

`requirements.txt` carries the full commit hashes. Bump a pin deliberately:
the `gwcat` commit fixes which `gwcat-pe-*` / `gwcat-selection-*` format
versions you can produce (see [Input files](inputs.md)), and the `tinyns`
commit fixes which sampler internals produced a recorded evidence.

## Optional extras

```bash
python -m pip install -e ".[gp]"     # tinygp: GP / gppop population models
python -m pip install -e ".[flows]"  # flowjax, paramax, equinox: --gw_flows_path
python -m pip install -e ".[test]"   # pytest
```

`flows` installs `flowjax>=17.1,<18`, `paramax` and `equinox>=0.11,<0.13`.
Both `tinygp` and `flowjax` are imported lazily, so everything else runs
without them.

## GPU

`requirements.txt` declares only the floor `jax>=0.4.34` / `jaxlib>=0.4.34`,
which pip satisfies with a CPU wheel. For GPU or TPU, install the accelerator
wheel that matches your CUDA/ROCm stack by following the official JAX
installation instructions at
<https://docs.jax.dev/en/latest/installation.html>.

`darksirens.core.jax_config.configure_jax_runtime()`, which the CLI calls at
startup, enables x64, selects the BFC device allocator and leaves
preallocation off (the block-size planner sizes its blocks against the
device's *free* memory). Both allocator knobs use `setdefault`, so an explicit
`XLA_PYTHON_CLIENT_ALLOCATOR` / `XLA_PYTHON_CLIENT_PREALLOCATE` still wins.

## Verify

```bash
python -c "import darksirens"
darksirens_inference --help
```

The first exercises the package import; the second prints the full option list
of the main entry point. `setup.py` installs nine console scripts, listed with
their options in [CLI reference](../reference/cli.md).

## Build the documentation

```bash
python -m pip install -r docs/requirements.txt
make docs-html
python -m http.server --directory docs/_build/html 8000
```

`make docs-strict` runs the same build with `-W --keep-going`, turning Sphinx
warnings into errors.

Next: [Quickstart](quickstart.md) runs the whole pipeline on a generated mock.
