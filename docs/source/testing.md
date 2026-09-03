# Testing & smoke runs

This page is a practical map of how to check that `darksirens` works across its
use cases: a fast **pytest** layer, a one-command **smoke driver** that runs
every use case on tiny mock data, and the **per-case commands** (smoke + full)
so you can drive any single case by hand. The science behind each case is on the
[Theory & methods](theory.md) page.

## Quick start

```bash
# from the repo root, with the darksirens-dev conda env
bash scripts/smoke_tests/run_smoke_tests.sh
```

That one command generates tiny mocks once, runs every use case (universe models,
population models, sky models, marks, LSS completion, weak + strong lensing,
samplers), and prints a `PASS / FAIL / SKIP` table. Useful flags:

```bash
bash scripts/smoke_tests/run_smoke_tests.sh --list             # list cases, run nothing
bash scripts/smoke_tests/run_smoke_tests.sh --cases U-spec,S-dip   # run a subset
bash scripts/smoke_tests/run_smoke_tests.sh --full             # realistic settings (slow)
bash scripts/smoke_tests/run_smoke_tests.sh --pytest           # also run the Tier-0 pytest layer
bash scripts/smoke_tests/run_smoke_tests.sh --keep             # REUSE the previous run's _out/
```

Per-case logs and run outputs land under `scripts/smoke_tests/_out/`. A case is
**PASS** if its program exits 0 (the inference ran and wrote `results.hdf5`),
**FAIL** if it errored (see the log), **SKIP** if a prerequisite is missing
(e.g. `tinygp`, or the local strong-lensing inputs). `_out/` is deleted and
rebuilt at the start of every run unless you pass `--keep`.

The summary line reports three independent failure counts, and the script exits
nonzero if **any** of them is nonzero:

* `FAIL` — smoke cases that ran and errored;
* `PREP_FAIL` — mock-data preparation steps that failed. These make their
  dependent cases `SKIP`, which is not the same as passing, so they are counted
  as failures in their own right;
* `PYTEST` — the exit status of the `--pytest` layer (`not run` without it),
  which runs the Tier-0 manifest below.

## Environment

The smoke driver runs each command through `conda run -n $ENV` when a conda
executable is available (`$CONDA`, else one on `PATH`) and otherwise through the
`python` already on `PATH` — so an activated environment needs no configuration.
Override either with the `CONDA` / `ENV` environment variables.

To run commands by hand in an activated environment, just call them; to pin an
environment explicitly:

```bash
RUN="conda run --no-capture-output -n darksirens-dev"
$RUN python -m darksirens.cli.inference --help
```

GP / binned-GP population models and the GP sky models need **`tinygp`**:

```bash
$RUN python -m pip install tinygp
```

The tools are invoked as modules (`python -m darksirens.cli.<name>`) so they
work regardless of whether the console scripts are on `PATH`. The two
long-running drivers behave identically either way: `darksirens_inference` and
`darksirens_inference_lensing` enter through `console_main`, which applies the
same `run_cli` hard-exit that the `__main__` guard does (without it, CPython
teardown can block in the CUDA exit handlers and leave a *finished* run idling
until the SLURM cgroup kills it).

## Tier 0 — pytest (fast subset)

**Run this before every commit.** It is the same list `.github/workflows/ci.yml`
runs, read from `tests/fast_subset.txt` so the local gate and the CI gate cannot
drift apart:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu \
  $RUN python -m pytest -q $(grep -v '^#' tests/fast_subset.txt)
```

Measured 2026-08-23 (CPython 3.11.10, jax 0.4.34 CPU, numpy 1.26.4): **68 files,
803 tests collected**, with the last end-to-end timing **17m47s in GitHub CI**
(and 21m10s on a shared review node) at the then-current 49 files / 519 tests.
Every file in it is verified green both standalone and in the batch, and none of
them depends on the numpy major version.

> This block said "32 files, 214 passed, ~3 min" until 2026-08-23, having been
> measured on 2026-07-29. The gate had meanwhile grown ~2.5x, so a contributor
> was told to expect three minutes before every commit and got closer to twenty.
> It is a *record*, not an aspiration: re-measure it when you add files, with
> the commands in the `tests/fast_subset.txt` header.

**Capacity.** `.github/workflows/ci.yml` gives this job `timeout-minutes: 30`,
so ~18 min leaves roughly 12 min of headroom — not urgent, but the trend is one
way. When a run comes in over ~25 min, split the manifest into fresh-process
shards (which also releases the JAX caches between groups) rather than raising
the timeout. Do **not** reach for a monolithic CPU full-suite job to relieve it,
for the RSS reason below.

Collection over the whole tree is still a useful cheap check:

```bash
$RUN python -m pytest tests/ --collect-only -q     # should report 0 errors
```

**Do not run the full `tests/` tree CPU-only.** With `JAX_PLATFORMS=cpu` it has
been observed to grow to ~258 GB RSS and get OOM-killed, and it competes with any
SLURM job on the same node. The full tree is a **Tier 0b, GPU-only** gate:

```bash
# GPU box, nothing else running:
$RUN python -m pytest tests/ -q
```

### Known pre-existing failures (not caused by your change)

The previous version of this page claimed one known failure. There are several
classes, so a contributor cannot tell a regression from the baseline without
this list. Verified on master, 2026-07-26:

- **numpy major-version split — FIXED 2026-07-26.** Library code
  (`gw/selection.py`, `redshift/checks.py`), both mock generators, and every
  test file that named `np.trapezoid`/`np.trapz` directly now go through a
  local compat alias (`_trapezoid = np.trapezoid if hasattr(...) else
  np.trapz`), so the tree runs on numpy 1.26 **and** numpy 2. If you write a
  new test that integrates, use the same alias — a bare `np.trapezoid` will
  regress the validated 1.26 env, a bare `np.trapz` the numpy-2 one.
- **`tests/test_pdet_selection.py`** — the two `DID NOT WARN` failures this
  page used to list were collateral of the numpy `AttributeError` aborts in
  the same session; with the compat alias in place the file passes whole
  (17 passed, 2 skipped).
- **Combined-run `sys.modules` pollution.** Several files install a **stub**
  `tinygp` (and a stub `gwcat`) into `sys.modules` at import time, e.g.
  `tests/test_parameter_table.py`. In a combined run that stub leaks into later
  files, so GP-dependent tests can behave differently in a batch than
  standalone; `tests/test_population_registry_golden.py` explicitly detects a
  stub (no `__file__`) and skips rather than erroring.
- **Backend-dependent block-size equivalence.** Some block/chunk equivalence
  tests compare reductions that XLA reassociates differently per backend, so
  they can pass on one device and flake on another.
- `tests/test_gppop_population.py` and other GP tests **need real `tinygp`**;
  they skip (or fail) without it. numpyro / jax-gated tests skip cleanly when
  those packages are absent — both are present in `darksirens-dev`.
- `tests/test_fixed_parameter_coordinates.py` used to be documented here as
  *the* known failure. It passes today (14 passed), so the `--ignore` this page
  used to prescribe is obsolete and has been dropped.

## Tier 1 — generating mock data

The smoke driver does all of this for you; the commands are here for manual runs.
HDF5 outputs are git-ignored. Use a scratch dir, e.g. `DATA=data/smoke`.

```bash
DATA=data/smoke

# Dark-siren mock: GW events + selection injections + pixelated survey, one call.
# --n-galaxies caps the catalog (use --n0 for a physical density; it scales with
# volume to ~1e5 galaxies, which makes the dark-siren precompute slow for a smoke).
$RUN python scripts/mock_dark_sirens/generate_mock_data.py \
    --outdir $DATA --seed 1 --n-galaxies 3000 --nobs 8 --nsamp 256 --ndraw 20000 --nside 8 --zmax 0.1
# -> $DATA/mock_gw_events.h5  mock_gw_selection.h5  catalog_pixelated_nside_8.h5

# Add synthetic galaxy marks (so --mark_model loglinear has columns to read)
$RUN python scripts/smoke_tests/make_marks.py \
    --catalog $DATA/catalog_pixelated_nside_8.h5 --out $DATA/catalog_marked.h5

# LSS-conditioned completion tables (radial + 3-D angular-coupling)
$RUN python -m darksirens.cli.build_lognormal_completion \
    --catalog $DATA/catalog_pixelated_nside_8.h5 --out $DATA/q_radial.h5 --mode radial --n-members 0
$RUN python -m darksirens.cli.build_lognormal_completion \
    --catalog $DATA/catalog_pixelated_nside_8.h5 --out $DATA/q_gp3d.h5 --mode gp3d --n-members 0

# Bright-siren mock: GW + selection + bright_counterparts.json
$RUN python scripts/mock_bright_sirens/generate_mock_bright_sirens.py \
    --outdir data/smoke_bright --seed 2 --n-galaxies 3000 --nobs 3 --nsamp 256 --ndraw 20000 --zmax 0.1
```

**Strong-lensing inputs** (for `--cluster_mode j2`) are produced by a separate,
local simulation pipeline (`SIM_LENSING_donotpush/sim_step6.py`, not shipped in
the repo). When present, the driver runs it; by hand:

```bash
PYTHONPATH=SIM_LENSING_donotpush $RUN python -c \
  "import sim_step6; sim_step6.assemble('data/lensing', n_universe=8000, seed=7, nsamp=200, \
   n_sing_keep=15, n_pair_keep=5, n_unlensed_inj=20000, n_lensed_inj=20000)"
# -> data/lensing/{mock_gw_pe.h5, mock_pair_pe.h5, mock_gw_selection.h5,
#                  mock_lensed_injections.h5, partition.json}
```

## Tier 2 — the use-case matrix

Each case below has a **smoke** command (tiny + fast; cosmology/often population
fixed to keep dimensions small). For a **full** run, drop the `--fix*` flags you
want to sample and raise the sampler effort (`--nlive 1000`, or numpyro
`--nuts_warmup 500 --nuts_samples 1000`), and use a larger mock
(`--nobs 100 --nside 32`). Common path setup:

```bash
DATA=data/smoke; GWE=$DATA/mock_gw_events.h5; SEL=$DATA/mock_gw_selection.h5
CAT=$DATA/catalog_pixelated_nside_8.h5
```

### Universe models

```bash
# U-spec  spectral sirens (GW only)
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL \
    --universe_model spectral_sirens --pop_model powerlaw+peak \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/U-spec

# U-wl  spectral sirens + weak-lensing magnification (lognormal s^2(z)=a z^b, via the lensing CLI)
$RUN python -m darksirens.cli.inference_lensing --gw_path $GWE --gwselection_path $SEL \
    --cluster_mode off --wl_backend lognormal --lensing_wl_a 4e-3 --lensing_wl_b 1.5 \
    --pop_model powerlaw+peak --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/U-wl

# U-dark  incomplete galaxy catalog
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL --survey_path $CAT \
    --universe_model dark_sirens --pop_model powerlaw+peak \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/U-dark

# U-complete  complete-catalog formalism
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL --survey_path $CAT \
    --universe_model dark_sirens_complete --pop_model powerlaw+peak \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/U-complete

# U-bright  EM counterparts (extract RA/Dec/z from bright_counterparts.json)
CP=$($RUN python -c "import json;d=json.load(open('data/smoke_bright/bright_counterparts.json'));\
it=d['counterparts'] if isinstance(d,dict) else d;print(' '.join(f\"{c['ra_rad']} {c['dec_rad']} {c['z']}\" for c in it))")
$RUN python -m darksirens.cli.inference \
    --gw_path data/smoke_bright/mock_bright_gw_events.h5 \
    --gwselection_path data/smoke_bright/mock_bright_gw_selection.h5 \
    --universe_model bright_sirens --counterpart $CP --counterpart_dz 1e-4 \
    --pop_model powerlaw+peak --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/U-bright
```

### Population models

`--pop_model` is a naming grammar; GP variants need `tinygp` and are best sampled
with numpyro (high dimensional). `powerlaw+peak` is covered by `U-spec`.

```bash
# P-bpl2pk  broken power law + 2 peaks
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL \
    --universe_model spectral_sirens --pop_model brokenpowerlaw+2peaks \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/P-bpl2pk

# P-gp1d  1-D Gaussian-process mass function (tinygp)
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL \
    --universe_model spectral_sirens --pop_model gp1d_m1 \
    --fix_cosmology true --fix_survey true --sampler numpyro --nuts_warmup 10 --nuts_samples 10 --nuts_max_tree_depth 3 --save_path out/P-gp1d

# P-gppop  binned-GP population (tinygp). Uses dynesty (gradient-free): gppop
# currently trips the numpyro/NUTS gradient preflight (NaN gradient at the
# initial point) — a known issue to investigate; dynesty exercises the model.
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL \
    --universe_model spectral_sirens --pop_model gppop \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 80 --dlogz 10 --save_path out/P-gppop
```

### Sky-anisotropy models

All on spectral sirens with cosmology + population fixed (only sky parameters
free). The high-dimensional GP sky models use numpyro.

```bash
for SKY in isotropic dipole multipole multipole_l3; do
  $RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL \
      --universe_model spectral_sirens --sky_model $SKY --pop_model powerlaw+peak \
      --fix_cosmology true --fix_population true --fix_survey true \
      --sampler dynesty --nlive 60 --save_path out/S-$SKY
done
for SKY in sphere_gp sphere_gp_z overdensity_gp; do
  $RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL \
      --universe_model spectral_sirens --sky_model $SKY --pop_model powerlaw+peak \
      --fix_cosmology true --fix_population true --fix_survey true \
      --sampler numpyro --nuts_warmup 10 --nuts_samples 10 --nuts_max_tree_depth 3 --save_path out/S-$SKY
done
```

To actually *recover* injected structure (a full run), generate a structured
mock (`--sky-dipole-amp 0.5 --sky-dipole-z-pivot 1.0 --sky-blob-amp 0.8 …`) and
compare evidences with `scripts/run_sky_ladder.sh`.

### Marked-host model

```bash
# M-loglin  log-linear host efficiency over galaxy marks (needs the marked catalog)
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL \
    --survey_path $DATA/catalog_marked.h5 --universe_model dark_sirens \
    --mark_model loglinear --marks logmstar,logssfr --pop_model powerlaw+peak \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/M-loglin
```

### LSS-conditioned completion

```bash
# L-legacy  legacy local-overdensity factor
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL --survey_path $CAT \
    --universe_model dark_sirens --use_lss true --pop_model powerlaw+peak \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/L-legacy

# L-radial / L-gp3d  precomputed lognormal completion table
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL --survey_path $CAT \
    --universe_model dark_sirens --lss_completion $DATA/q_radial.h5 --pop_model powerlaw+peak \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/L-radial
$RUN python -m darksirens.cli.inference --gw_path $GWE --gwselection_path $SEL --survey_path $CAT \
    --universe_model dark_sirens --lss_completion $DATA/q_gp3d.h5 --pop_model powerlaw+peak \
    --fix_cosmology true --fix_survey true --sampler dynesty --nlive 60 --save_path out/L-gp3d
```

### Lensing

Weak lensing is `U-wl` above. Strong-lensing clusters use the dedicated driver:

```bash
# X-cloff  singleton-only (no cluster inputs needed)
$RUN python -m darksirens.cli.inference_lensing --gw_path $GWE --gwselection_path $SEL \
    --cluster_mode off --wl_backend lognormal --pop_model powerlaw+peak \
    --fix_cosmology true --fix_survey true --fix_population true --sampler dynesty --nlive 60 --save_path out/X-cloff

# X-cl2  J=2 image pairs (needs the SIM_LENSING dataset from Tier 1)
LD=data/lensing
$RUN python -m darksirens.cli.inference_lensing \
    --gw_path $LD/mock_gw_pe.h5 --gwselection_path $LD/mock_gw_selection.h5 \
    --lensed_injections_path $LD/mock_lensed_injections.h5 --pair_pe_path $LD/mock_pair_pe.h5 \
    --partition_path $LD/partition.json --cluster_mode j2 --wl_backend lognormal \
    --pop_model powerlaw+peak --fix_cosmology true --fix_survey true --fix_population false \
    --sampler dynesty --nlive 60 --save_path out/X-cl2
```

### Samplers

Only `dynesty` and `numpyro` are exercised. Swap `--sampler` on any case above;
e.g. on `U-spec`:

```bash
# dynesty
... --sampler dynesty --nlive 60
# numpyro (NUTS)
... --sampler numpyro --nuts_warmup 10 --nuts_samples 10 --nuts_max_tree_depth 3 --nuts_chains 1
```

## Tier 3 — analyze & model comparison

`darksirens_analyze` recomputes posterior-predictive spectra and, given several
run directories, the relative evidences + pairwise Bayes-factor matrix:

```bash
$RUN python -m darksirens.cli.analyze \
    --run_dirs out/S-iso/* out/S-dip/* out/S-mult/* --outdir out/figs
```

(Each `out/<case>/*` is the timestamped run directory containing `results.hdf5`.)

## Interpreting results

- **PASS** — the case ran and wrote `results.hdf5`. For smoke runs this confirms
  the code path (data loading, likelihood, sampler, save) works end-to-end; it is
  *not* a statistical-accuracy check (use the `--full` profile + recovery mocks
  for that).
- **FAIL** — a genuine breakage; read `scripts/smoke_tests/_out/logs/<case>.log`.
- **SKIP** — a prerequisite is missing: `tinygp` (GP cases), the local
  `SIM_LENSING_donotpush/` pipeline (the `X-cl2` strong-lensing case), or the
  `[slow]` GP-sky cases (`S-sgpz`, `S-od`) unless `--slow`/`--full` is passed.

## Known findings (current master)

A default run reports **PASS for 21/22** non-slow cases (23/24 with `--slow`).
The radial-Q (`L-radial`), GP3D-Q (`L-gp3d`) and marked-host (`M-loglin`)
JIT-path failures documented here previously have been **fixed** — the Q-table
resolver and the marked galaxy-measure grid are now trace-safe, and
`tests/test_traced_prior_state.py` pins the end-to-end jitted consumption of
both. The remaining known case:

- **`X-cl2` (FAIL).** The strong-lensing cluster (J=2) end-to-end run depends on
  the local `SIM_LENSING_donotpush/` pipeline. With its `POP_NAME` updated to the
  current grammar it now *generates* the dataset, but the files use the
  pre-`gwcat` HDF5 schema (`mock_data=True`), which current `load_gw_samples`
  rejects — it requires `format_version="gwcat-1.0"` plus `m1src`/`m2src` and the
  `pe_cosmology_*` / `chi_eff_*` attrs. Update the sim's HDF5 writers
  (`sim_step4/5`) to that schema to run this case. The cluster *code* itself is
  covered by the pytest layer (`test_cluster_likelihood`, `test_cluster_selection`,
  `test_lensing`); `X-cloff` (singleton path) passes end-to-end.
- **`S-sgpz`, `S-od` (slow, gated).** The 192-latent (sphere×z) GP sky models run
  correctly but take ~10–20 min each under NUTS (gradient *compilation*
  dominates), so they are gated behind `--slow`/`--full`. The lower-rank
  `sphere_gp` (`S-sgp`) runs in ~80 s by default.

The now-fixed LSS-completion and marked-host failures shared one root cause —
both features were unit-tested through their *eager* entry points but never
driven end-to-end through `@jit darksiren_log_likelihood`. The fix made the
Q-table resolver and the marked galaxy-measure grid trace-safe (no Python
`int()`/concrete branching on traced catalog fields); the moral survives them:
new likelihood features need an end-to-end jitted test, not just eager units.

