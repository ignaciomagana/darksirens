# Contributing

## Development setup

```bash
git clone https://github.com/ignaciomagana/darksirens.git
cd darksirens
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install -r docs/requirements.txt
```

## Before opening a pull request

1. Run the Tier-0 suite the way CI does (see [Testing](../guide/testing.md)):

   ```bash
   JAX_PLATFORMS=cpu python -m pytest $(grep -v '^#' tests/fast_subset.txt) -q
   ```

2. Run the golden gate when you touch the likelihood:

   ```bash
   JAX_PLATFORMS=cpu python -m pytest tests/test_unified_k1_golden.py -q
   ```

   A change that moves a golden cell by more than the 1e-12 relative tolerance
   is a change of the numerical answer and needs to say so in the pull request.

3. Build the documentation with warnings as errors:

   ```bash
   make docs-strict
   ```

## Conventions

- Numerical changes come with a measurement: what was compared, on which
  inputs, and the largest difference observed. Bit-identical is stated as such
  only when it was measured.
- Every command-line option is documented in [the CLI reference](../reference/cli.md)
  in the pull request that adds it.
- Loaders document the HDF5 dataset names they read; see
  [Input files](../getting-started/inputs.md).
- API documentation is generated from docstrings; workflow explanations belong
  in the user guide, not in docstrings.
- Tests that are cheap enough to run before every commit go into
  `tests/fast_subset.txt`; update its header count when the manifest changes.
