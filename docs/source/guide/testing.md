# Testing

This page answers one question: what do you run to check that `darksirens`
still works, and what is already known to fail. Tier 0 is the pre-commit gate;
the higher tiers cost more and are run on demand.

## Tier 0: the fast pytest subset

`tests/fast_subset.txt` is the single source of truth for the gate. Run exactly
what CI runs:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
  python -m pytest -q $(grep -v '^#' tests/fast_subset.txt)
```

`.github/workflows/ci.yml`'s `fast-tests` job reads the same manifest with the
same `grep`, on `ubuntu-latest` with `timeout-minutes: 30`, against a pinned
CPU stack (jax 0.4.34, numpy 1.26.4, scipy 1.12.0, pytest 8.3.4) and the same
two environment variables. It installs neither `gwcat` nor `tinyns`: both are
imported lazily and the subset stubs `gwcat`. One file runs the same way, e.g.
`... python -m pytest -q tests/test_unified_k1_golden.py`.

Recorded size, measured 2026-08-23 (CPython 3.11.10, jax 0.4.34 CPU, numpy
1.26.4): **68 files, 805 tests collected**. The last end-to-end timing is
17m47s in GitHub CI and 21m10s on a shared review node, both at the
then-current 49 files / 519 tests.

That record is enforced: `tests/test_fast_subset_record.py` asserts that every
listed file exists, that none is listed twice, that the manifest header's
`N files, M tests collected` line matches the number of listed files, and that
this page quotes the same two numbers. The manifest header carries the two
commands to re-measure with; run them and update both places in the commit that
changes the manifest. Admission rules for the manifest: green standalone and
green in the batch on CPU, no dependence on the numpy major version (nothing
calling `np.trapezoid` or `np.trapz` directly), and seconds rather than minutes
(no sampler runs, no GPU, no large mocks).

```{warning}
Do not run the full `tests/` tree CPU-only. Under `JAX_PLATFORMS=cpu` it has
been observed to grow to ~258 GB RSS and get OOM-killed. The full tree is a
GPU-only gate: run `python -m pytest tests/ -q` on a GPU box with nothing else
running. Collection alone is cheap and useful:
`python -m pytest tests/ --collect-only -q` should report 0 errors.
```

## The golden gate

`tests/test_unified_k1_golden.py` pins `ll(coord)` at three deterministic
coordinates for each cell of the K=1 feature matrix (both `dark_sirens` array
paths, deterministic `Q_LSS`, the Q-ensemble marginalization, marked hosts, a
live `delta_g`, an anisotropic sky model, bright-siren counterparts, both
empty-pixel policies of `dark_sirens_complete`, spectral sirens, lognormal weak
lensing, field sky weighting, selection batching) against
`tests/golden/unified_k1_golden.json`. Values must match to `rtol <= 1e-12`
with `atol = 0`; `DARKSIRENS_GOLDEN_EXACT=1` enforces `==` instead. Goldens are
stored per JAX backend (`cpu` and `gpu` reductions differ at the ULP level)
keyed by `jax.default_backend()`, and a backend with no recorded goldens skips.
A second bank of tests perturbs each cell's feature and asserts the value
moves, so an inert fixture cannot pin a meaningless golden.

Known pre-existing drift, reproduced on this checkout on the `cpu` backend
(2026-09-03, 3 failed / 24 passed in 178 s): the three `Q_LSS` cells `qdet`,
`use_lss` and `ensemble_marg` exceed `rtol 1e-12` by about a factor two (max
relative difference 2.3e-12, max absolute 3.0e-13 on values near 0.127). The
module docstring additionally records two stale `gpu:*` `wl_lognormal` entries
that predate the lognormal Hermite proposal-to-target density ratio (which
moves that cell by ~6%); the `cpu` entry carries the ratio and the A100/H100
entries must be regenerated on that hardware. No other cell is affected.

## Tier 1: generating mock data

HDF5 outputs are git-ignored, so use a scratch directory.
`bash scripts/smoke_tests/run_smoke_tests.sh` drives all of this; the three
generators by hand, at smoke sizes:
```bash
DATA=data/smoke

# GW events + selection injections + pixelated survey, one call
python scripts/mock_dark_sirens/generate_mock_data.py \
    --outdir $DATA --seed 1 --n-galaxies 3000 --nobs 8 --nsamp 256 \
    --ndraw 20000 --nside 8 --zmax 0.1

# synthetic galaxy marks, so --mark_model loglinear has columns to read
python scripts/smoke_tests/make_marks.py \
    --catalog $DATA/catalog_pixelated_nside_8.h5 --out $DATA/catalog_marked.h5

# bright-siren mock: mock_bright_gw_events.h5, mock_bright_gw_selection.h5,
# bright_counterparts.json
python scripts/mock_bright_sirens/generate_mock_bright_sirens.py \
    --outdir data/smoke_bright --seed 2 --nobs 3 --nsamp 256 \
    --ndraw 20000 --zmax 0.1
```

`darksirens_build_lognormal_completion --mode radial|gp3d --n-members 0` turns
the pixelated catalog into the `Q_LSS` tables the LSS cases read. `--n-galaxies`
caps the catalog; `--n0` requests a physical density instead and scales with
volume to ~1e5 galaxies, which makes the dark-siren precompute slow for a smoke
run. Strong-lensing inputs for `--cluster_mode j2` come from a separate local
simulation pipeline that is not shipped here.

Every measurement width in the dark-siren generator is
`sigma_x = a_x * (rho_th / rho_obs)`, a function of the recorded
signal-to-noise ratio and of nothing else, with detection on the recorded
`rho_obs >= rho_th`: the family of Fishbach, Holz and Farr (2018,
[arXiv:1805.10270](https://arxiv.org/abs/1805.10270), eqs. 29-31) as released
in `GWMockCat` (Farah et al. 2023,
[arXiv:2301.00834](https://arxiv.org/abs/2301.00834), App. A) at its released
defaults (`sigma_rho = 1`, `a_Mc = 0.08`, `a_chi = 0.2`, `rho_th = 8`;
`a_q = 0.60` from the GW150914 anchor). So `dL` is derived from
`(Mc_det, rho)` rather than measured, there is no distance-width argument
(`sigma_ln dL = sqrt((5/6 sigma_lnMc)^2 + (sigma_rho/rho)^2)`), and nothing
recorded or sampled is clipped: physical ranges are imposed on the PE prior by
exact inverse-CDF truncation instead.

## Tier 2: the use-case matrix

Each case is one run on the Tier-1 mock with cosmology and survey fixed:

```bash
DATA=data/smoke; GWE=$DATA/mock_gw_events.h5; SEL=$DATA/mock_gw_selection.h5
CAT=$DATA/catalog_pixelated_nside_8.h5
COMMON="--gw_path $GWE --gwselection_path $SEL --pop_model powerlaw+peak \
  --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60"
darksirens_inference $COMMON --universe_model spectral_sirens --save_path out/U-spec
```

| Case | Added flags |
|---|---|
| dark sirens | `--universe_model dark_sirens --survey_path $CAT` |
| complete catalog | `--universe_model dark_sirens_complete --survey_path $CAT` |
| bright sirens | `--universe_model bright_sirens --counterpart RA DEC Z --counterpart_dz 1e-4` |
| population variants | `--pop_model brokenpowerlaw+2peaks`, `gp1d_m1`, `gppop` (GP models need `tinygp`) |
| sky anisotropy | `--fix_population true --sky_model {isotropic,dipole,multipole,multipole_l3}` |
| GP sky | `--sky_model {sphere_gp,sphere_gp_z,overdensity_gp} --sampler numpyro --nuts_warmup 10 --nuts_samples 10 --nuts_max_tree_depth 3` |
| marked hosts | `--universe_model dark_sirens --survey_path $DATA/catalog_marked.h5 --mark_model loglinear --marks logmstar,logssfr` |
| LSS completion | `--universe_model dark_sirens --survey_path $CAT` plus either `--use_lss true` or `--lss_completion $DATA/q_radial.h5` |
| weak lensing | `darksirens_inference_lensing --cluster_mode off --wl_backend lognormal --lensing_wl_a 4e-3 --lensing_wl_b 1.5` |
| J=2 clusters | `darksirens_inference_lensing --cluster_mode j2` plus the strong-lensing inputs |

For a full run drop the `--fix_*` flags you want to sample, raise the sampler
effort (`--nlive 1000`, or `--nuts_warmup 500 --nuts_samples 1000`), and use a
larger mock (`--nobs 100 --nside 32`). [Inference](inference.md) documents the
blocks these flags belong to, [Analysis](analysis.md) the run directory.

## Known pre-existing failures

- The three golden cells above and the two stale `gpu:*` `wl_lognormal` goldens.
- **Combined-run `sys.modules` pollution.** Several files install a stub
  `tinygp` (and a stub `gwcat`) at import time, e.g.
  `tests/test_parameter_table.py`, so GP-dependent tests can behave differently
  in a batch than standalone; `tests/test_population_registry_golden.py`
  detects a stub (no `__file__`) and skips rather than erroring.
- **Backend-dependent block/chunk equivalence.** Some equivalence tests compare
  reductions XLA reassociates differently per backend, so they can pass on one
  device and not another.
- The GP tests (`tests/test_gppop_population.py` and others) need real
  `tinygp`; numpyro- and jax-gated tests skip cleanly when those are absent.
  Anything else is usually one of the cases in
  [Troubleshooting](troubleshooting.md).
