# Command-line interface

Installing the package exposes three console scripts.

## `darksirens_pixelate`

Convert a raw galaxy survey HDF5 file into the dense HEALPix layout used by dark-siren inference.

```bash
darksirens_pixelate --survey_path SURVEY.h5 [--save_path OUTDIR] [--nside 64] [--add_plots]
```

Options:

- `--survey_path`: required path to the raw HDF5 survey file.
- `--save_path`: output directory; defaults to the current directory.
- `--nside`: HEALPix NSIDE; defaults to `64`.
- `--add_plots`: create diagnostic skymap, redshift, and occupancy plots.

## `darksirens_inference`

Run hierarchical inference.

```bash
darksirens_inference \
  --gw_path GW.h5 \
  --gwselection_path INJECTIONS.h5 \
  --sampler tinyns \
  [options]
```

### Data options

- `--gw_path`: required GW posterior-sample HDF5 file.
- `--gwselection_path`: required gwcat selection HDF5 file.
- `--survey_path`: pixelated survey HDF5 file; required for dark-siren models.
- `--counterpart RA1 DEC1 Z1 [RA2 DEC2 Z2 ...]`: bright-siren counterpart coordinates and redshifts, one triplet per GW event in event order; RA and Dec use radians, matching the GW sample convention. Required for `--universe_model bright_sirens`.
- `--counterpart_dz`: Gaussian redshift uncertainty assigned to the synthetic counterpart catalog entry; defaults to `1e-4`.
- `--counterpart_nside`: HEALPix NSIDE for the synthetic counterpart catalog; defaults to `1`.
- `--save_path`: directory for settings, samples, plots, and summaries.

### Physical-model options

- `--universe_model`: one of `spectral_sirens`, `bright_sirens`, `dark_sirens`, or `dark_sirens_complete`.
- `--pop_model`: population model name. Parametric mixture names are parsed as a composition grammar: `+`-separated mass tokens (`powerlaw`, `brokenpowerlaw`, `peak`) with optional digit count prefixes, e.g. `powerlaw+peak`, `brokenpowerlaw+2peaks`, `2powerlaws+3peaks`. Any grammar composition works with blueprint-default priors; curated names additionally carry physics-tuned priors and fiducials. Bespoke names such as `gp_mass`, `gp_mass_pairing`, `gp_mass_pairing_joint`, `golomb_1g`, `golomb_1g+tail`, and `gwtc5_fiducial_bpl2peaks` are registered explicitly. See [Concepts → Population models](concepts.md#population-models).
- `--shared_beta`: whether to use one shared beta/pairing distribution (`true`, default) or per-component beta parameters (`false`).
- `--shared_spin`: whether to use one shared spin distribution (`true`, default) or per-component spin parameters (`false`).
- `--shared_gamma`: whether to use one shared redshift-evolution gamma (`true`, default) or per-component gamma parameters (`false`).
- `--fix_population`: fix all population parameters to fiducial values.
- `--fix_cosmology`: fix all cosmological parameters (`H0`, `Om0`, `w0`, `wa`) to fiducial values.
- `--fix_de`: fix only the CPL dark-energy parameters (`w0=-1`, `wa=0`) while leaving `H0` and `Om0` available unless fixed separately.
- `--fix_survey`: fix survey-completion parameters to fiducial values.
- `--prior_overrides`: JSON object mapping parameter labels to `[lower, upper]` prior bounds, e.g. `{"H0": [60, 80], "w0": [-1.2, -0.8], "wa": [-0.5, 0.5]}`. Population parameters use the printed LaTeX labels from the startup parameter table, e.g. `{"$\\alpha_{\\rm PL}$": [0.0, 4.0]}`. Multi-component mass labels are tagged by slot (`PL`, `BPL`, `G1`, `G2`, ...).
- `--fixed_parameter_values`: JSON object mapping parameter labels to fixed scalar values, e.g. `{"Om0": 0.3075, "w0": -1.0, "wa": 0.0}` or `{"$\\mu_{\\rm G}$": 35.0}`. Mixture-weight labels are stick-breaking inputs (`$v_1$`, `$v_2$`, ...), not final component fractions.
- `--bright_siren_sky_marginalized BOOL`: for `bright_sirens`, ignore the counterpart sky-pixel gate and apply only the counterpart redshift prior. Defaults to `False`. Accepted true values are `true`, `t`, `1`, `yes`, and `y`; accepted false values are `false`, `f`, `0`, `no`, and `n` (case-insensitive).
- `--complete_empty_pixel_policy {zero,volume}`: controls genuinely empty pixels for `dark_sirens_complete` and `bright_sirens`. `zero` is the formal default and returns zero probability (`-inf` log-prior) when `ngals == 0`; `volume` uses the comoving-volume prior as a robustness approximation for sparse pixelations.

### Catalog options

- `--use_LSS`: include large-scale-structure overdensity where supported.
- `--validate_completion`: run a dry-run completion clipping diagnostic, save `completion_validation__*.json` under `--save_path`, and exit before likelihood construction or sampling. The diagnostic uses the same matched-kernel completeness ratio as the likelihood and reports clipping fractions for the raw ratio, LSS modulation, and effective completeness.
- `--completion_validation_pixels`: maximum number of unique catalog pixels to inspect during `--validate_completion`; defaults to `64`.

### Sampler options

- `--sampler`: required; one of `tinyns`, `dynesty`, or `numpyro`.
- `--nlive`: live points for nested samplers (`tinyns`, `dynesty`).
- `--dlogz`: evidence stopping threshold for nested samplers (`tinyns`, `dynesty`).
- `--max_samples`: maximum call/iteration budget for nested samplers (`dynesty` call cap, `tinyns` iteration cap); `0` = unlimited.
- `--tinyns_sample`: `tinyns` proposal method: `rwalk` or `prior` (TinyNS no longer supports `slice`/`rslice`; use `dynesty` for slice-style comparisons).
- `--tinyns_kernel`: `tinyns` proposal kernel: `jax` (default, jitted) or `python`.
- `--tinyns_walks`: `tinyns` number of random-walk steps per update (`sample=rwalk`).
- `--tinyns_replacement_chains`: `tinyns` independent random-walk chains run in parallel per replacement (`rwalk`+`jax` only; default `1`).
- `--tinyns_replacement_chain_schedule`: `tinyns` adaptive `rwalk`+`jax` escalation schedule, e.g. `1,4,16,64,256` (ascending). Starts at the smallest batch and escalates only when a stage fails, returning as soon as any stage succeeds. Mutually exclusive with `--tinyns_replacement_chains`.
- `--tinyns_max_attempts`: `tinyns` maximum constrained-proposal attempts per replacement (tinyns default `10000`). Must be `>= walks * replacement_chains`; if left unset it auto-raises to that product when needed.
- `--tinyns_step_scale`: `tinyns` initial proposal step scale as a fraction of the prior width.
- `--tinyns_progress_interval`: `tinyns` iterations between progress-bar updates.
- `--nuts_warmup`: NUTS warmup/adaptation steps for `numpyro`.
- `--nuts_samples`: NUTS posterior samples per chain for `numpyro`.
- `--nuts_chains`: number of NUTS chains for `numpyro`.
- `--nuts_target_accept`: target acceptance probability for `numpyro` NUTS.
- `--nuts_max_tree_depth`: maximum NUTS tree depth for `numpyro`.
- `--nuts_init_tries`: fallback initial-point draws from priors when midpoint has non-finite likelihood.
- `--nuts_init_seed_offset`: seed offset used for the NUTS fallback initial-point search.
- `--seed`: random seed.
- `--show_progress`: enable or disable progress bars.
- `--dynesty_diagnostics`: when using `--sampler dynesty`, write periodic runplot/traceplot PDF diagnostics under `<save_path>/dynesty_diagnostics/`.

### Performance options

- `--sel_batch_size`: optional injection-selection batch size.
- `--norm_nmass`, `--norm_nq`, `--norm_nchi`: mass, mass-ratio, and spin grid sizes used for GW-population normalization quadrature. They default to `500`, `200`, and `200`, respectively, and can also be set with `DARKSIRENS_GW_N_MASS`, `DARKSIRENS_GW_N_Q`, and `DARKSIRENS_GW_N_CHI`. The inference command prints the active values and saves them in `settings.json` under `normalization_grid`.

## `darksirens_analyze`

Analyze saved inference products and compute posterior-predictive summaries. The analyzer reads the current `results.hdf5` output format (including root-level or grouped posterior samples) and still supports legacy `samples.npy` runs.

```bash
darksirens_analyze [--run_dirs RUN_DIR [RUN_DIR ...]] [--mmin 1] [--mmax 100] [--nm 300]
```

Important options:

- `--run_dirs`: optional list of one or more directories produced by `darksirens_inference`; when omitted, the current directory is analyzed.
- `--mmin`, `--mmax`, `--nm`: primary-mass grid bounds and size.
- `--nq`: mass-ratio grid size.
- `--nz`: redshift grid size.
- `--nchi`, `--chimin`, `--chimax`: spin grid configuration.
- `--batch_size`: posterior-predictive evaluation batch size.
- `--cred_lo`, `--cred_hi`: lower and upper credible intervals.


## TinyNS presets

TinyNS support targets the current upstream sampler API. TinyNS modes are `rwalk`
and `prior`; TinyNS no longer supports `slice`/`rslice` modes. Use `dynesty` for
slice-style nested-sampling comparisons.

The default `--tinyns_preset recommended` uses the current TinyNS recommended
B32 fast path: `sample=rwalk`, `kernel=jax`, unbounded isotropic random-walk
proposals, `walks=5`, `replacement_chains=1`, and `jax_block_size=32`. B64/B128
are not darksirens defaults. Live-cov proposals and bounded/fused-bound modes
remain experimental and must be selected explicitly.

Heavy realistic dark-siren likelihoods can opt in to diagnostic starting points:

- `--tinyns_preset heavy_darksirens`
- `--tinyns_preset heavy_darksirens_strong`

These heavy presets are expensive and target-specific. They are not guaranteed
production settings; validate each target by inspecting `tinyns_resolved_config`,
`tinyns_diagnostics`, `tinyns_summary`, replacement metadata such as
`replacement_failures` and `replacement_rescue_used`, insertion-rank diagnostics
when present, and logZ scatter across independent seeds. Explicit `--tinyns_*`
options override the selected preset, and the resolved configuration is recorded
in `tinyns_resolved_config` in `settings.json` and supported HDF5 outputs.


### TinyNS JAX rwalk and redshift-prior barriers

JAX TinyNS rwalk evaluates replacement proposals in a vmapped JAX kernel. Some
JAX primitives, including `lax.optimization_barrier`, cannot be batched. The
`--redshift_prior_barrier auto` default disables the likelihood-internal
redshift-prior materialization barrier for TinyNS JAX rwalk while preserving it
for ordinary non-vmapped paths.

If users see:

```text
NotImplementedError: Batching rule for 'optimization_barrier' not implemented
```

then use:

```bash
--redshift_prior_barrier off
```

or keep the default:

```bash
--redshift_prior_barrier auto --sampler tinyns --tinyns_preset recommended
```

For diagnostic fallback only, `--tinyns_preset python_debug` avoids the JAX rwalk
path, but it is slow and should not be treated as the production default.

### Heavy darksirens TinyNS templates

Copy-pasteable runner script:

```bash
GW_PATH=/path/to/gw.h5 \
GWSELECTION_PATH=/path/to/injections.h5 \
SURVEY_PATH=/path/to/catalog.h5 \
SAVE_PATH=./runs/heavy_tinyns \
scripts/run_tinyns_heavy_darksirens_likelihood.sh
```

Direct CLI equivalent:

```bash
darksirens_inference \
  --gw_path /path/to/gw.h5 \
  --gwselection_path /path/to/injections.h5 \
  --survey_path /path/to/catalog.h5 \
  --sampler tinyns \
  --tinyns_preset heavy_darksirens \
  --universe_model dark_sirens \
  --pop_model brokenpowerlaw+2peaks \
  --fixed_cosmology true \
  --fix_survey true \
  --nlive 2000 \
  --dlogz 0.11 \
  --max_samples 0 \
  --sel_batch_size 4096 \
  --drop_full_catalog true \
  --show_progress true \
  --save_path ./runs/heavy_tinyns
```

For a stronger opt-in starting point after target-specific validation, change the
preset in either command to:

```bash
--tinyns_preset heavy_darksirens_strong
```
