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
- `--survey_path`: one or more pixelated survey HDF5 files; required for dark-siren models. A single path is the classic single-catalog analysis; `K >= 2` paths run the K-catalog **multitracer mixture** (see [Multitracer catalog mixtures](#multitracer-catalog-mixtures-k--2)), sampling the per-catalog host fractions.
- `--counterpart RA1 DEC1 Z1 [RA2 DEC2 Z2 ...]`: bright-siren counterpart coordinates and redshifts, one triplet per GW event in event order; RA and Dec use radians, matching the GW sample convention. Required for `--universe_model bright_sirens`.
- `--counterpart_dz`: Gaussian redshift uncertainty assigned to the synthetic counterpart catalog entry; defaults to `1e-4`.
- `--counterpart_nside`: HEALPix NSIDE for the synthetic counterpart catalog; defaults to `1`.
- `--save_path`: directory for settings, samples, plots, and summaries.

### Physical-model options

- `--universe_model`: one of `spectral_sirens`, `spectral_sirens_wl`, `bright_sirens`, `dark_sirens`, or `dark_sirens_complete`.
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

- `--use_LSS`: include the large-scale-structure overdensity `delta_g(pix, z)` in the missing-galaxy budget (`max(1 + b_eff*delta_g, 0)`). With a K-catalog mixture, each catalog computes its own overdensity field from its own rows, coupled to its own sampled `b_miss` (`b_miss_c{k}` for catalogs 2..K).
- `--catalog_sky_weighting {conditional,field}`: the dark-siren redshift-prior normalization convention. `conditional` normalizes each pixel by its own `Z[pix]` (the per-pixel z-shape estimand); `field` normalizes by the survey-global `Z(theta)` so a K>=2 mixture weight `fcat_k` measures the catalog's GW **host fraction** and `log10n0` becomes informative. **Unset auto-resolves by K** (conditional at K=1, field at K>=2); for `dark_sirens` the degenerate explicit combinations — `field` at K=1 (the global normalizer cancels between the PE and selection terms) and `conditional` at K>=2 (the railing z-shape-only `fcat` estimand) — are fatal. The field normalizer carries the SAME modulated missing budget as the numerator (Q_LSS, `delta_g`, marked-host `mu_miss`, per ensemble member under `--lss_marginalize`), so it composes with all catalog options below; it needs the full-sky rows (incompatible with `--drop_full_catalog`).
- `--lss_completion PATH [PATH ...]`: precomputed LSS-conditioned lognormal completion table(s) `Q_LSS` replacing the legacy overdensity factor. With K catalogs pass 0 or exactly K paths, positionally aligned with `--survey_path` (`""` = no external completion for that catalog); one table is never broadcast across catalogs.
- `--lss_marginalize`: fully-Bayesian marginalization over the Q_LSS ensemble, `logL = logsumexp_m logL(Q_m) - log M`. With a K-catalog mixture the marginalization uses **one shared member index** — member `m` of every catalog must sample the same LSS realization. Build the per-catalog ensembles jointly with `darksirens_build_joint_lognormal_completion` (see the joint-builder note below) so they share one `realization_set_id` and equal `--n-members`, or the run aborts (unless `--allow_unverified_shared_lss_members` accepts the independent-fields approximation).
- `--mark_model {none,loglinear}` and `--marks LIST`: the marked-host model (host efficiency `h(m|eta)` over z-centred galaxy marks). With a K-catalog mixture the marks are resolved **per catalog** from each survey file's datasets (`available ∩ --marks`): catalog 1 samples the unsuffixed `eta_<mark>` coefficients, catalogs 2..K sample `eta_<mark>_c{k}` blocks, and a catalog with no selected marks runs the plain galaxy-count host model (`h = 1`) inside the mixture.
- `--validate_completion`: run a dry-run completion clipping diagnostic, save `completion_validation__*.json` under `--save_path`, and exit before likelihood construction or sampling. The diagnostic uses the same matched-kernel completeness ratio as the likelihood and reports clipping fractions for the raw ratio, LSS modulation, and effective completeness. With K catalogs one diagnostic runs per catalog (`completion_validation_c{k}__*.json`).
- `--completion_validation_pixels`: maximum number of unique catalog pixels to inspect during `--validate_completion`; defaults to `64`.

### Multitracer catalog mixtures (K >= 2)

Passing `K >= 2` files to `--survey_path` runs the K-catalog mixture: the catalog-completed redshift prior becomes `log p_mix(z) = logsumexp_k [log w_k + log p_k(z, pix_k)]`, with per-catalog nside/pixelization, per-catalog survey nuisance blocks (`log10n0_c{k}`, `delta_c{k}`, `b_miss_c{k}`, `sigma_kde_c{k}`), and sampled stick-breaking weights `fcat_2..fcat_K` (Beta(1, K-m+1) priors; `w_2 = fcat_2` exactly at K=2).

- **Estimand.** Under the (auto-resolved) `field` sky weighting, `w_k` is the fraction of GW hosts drawn from catalog `k`'s tracer population — e.g. the AGN host fraction `f_agn` for a GAL+AGN K=2 run. `darksirens_analyze` derives and plots `w_1..w_K` from the sampled sticks automatically (`catalog_weights_<tag>.{pdf,npy}`).
- **Universe models.** `dark_sirens` (the general case) or `dark_sirens_complete` (special case: field weighting only). `spectral_sirens`, `spectral_sirens_wl`, and `bright_sirens` are inherently single-catalog.
- **Composability.** Per-catalog Q_LSS tables, `--use_LSS` overdensities, `--lss_marginalize` ensembles (shared member index), marked-host models (per-catalog eta blocks), and anisotropic `--sky_model` choices (one population-level `g(n, z)` factor shared across catalogs) all compose with the mixture; the survey-global field normalizer carries the same modulated missing budget as each catalog's numerator.
- **Weak lensing** (`spectral_sirens_wl`) and `--counterpart`/`bright_sirens` remain single-catalog.

#### Joint Q_LSS ensembles (`darksirens_build_joint_lognormal_completion`)

`--lss_marginalize` at `K >= 2` needs the per-catalog Q_LSS ensembles to share **one** LSS realization set (member `m` of every catalog is the same missing-galaxy field). The offline `darksirens_build_joint_lognormal_completion` produces exactly that: it infers **one** latent large-scale-structure field from all K catalogs jointly (the `gp3d` low-rank Poisson-lognormal model, per-survey bias `b_k` absorbed into the design matrix) and writes K per-survey `Q_LSS` files stamped with a single shared `realization_set_id`, so the loader's shared-member provenance check passes without `--allow_unverified_shared_lss_members`.

- `--catalogs PATH [PATH ...]` and `--outs PATH [PATH ...]`: the K input survey catalogs and their K output completion files (same count, positionally aligned).
- `--bias`, `--log10n0`, `--delta`: per-survey field bias / expected-density normalization / evolution exponent, each **1 value (shared) or K values** (one per catalog).
- `--n-members`, `--seed`: size and seed of the ONE shared Laplace ensemble (matched member `m` across all K files).
- `--realization-set-id`: stamp an explicit shared id (default: a fresh `uuid4` shared across the K outputs).
- `--mode` is `gp3d` only; `--mode radial` is rejected (the radial completion fits each pixel independently, so it carries no shared field to match members across surveys).

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
- `--selection_neff_guard`: sparse-selection validity guard: `auto` (default) resolves to a smooth `soft` wall for `numpyro` and the historical `hard` `-inf` wall for `dynesty`/`tinyns`; `hard`/`soft` force one. The resolved mode, the cap, and the criterion are printed on one always-on `selection guard: …` line at startup.
- `--max_likelihood_variance`: cap (default `1.0`, the GWTC-4.0/5.0 criterion) on the Monte-Carlo variance of the total log-likelihood estimator, `sigma^2_lnL = sum_i sigma_i^2 + N_obs^2/Neff_sel`. Proposals above it are guarded (hard `-inf` or the soft wall); the Vitale `Neff > 5*N_obs` floor always applies.
- `--sampler_preflight`: `on` (default) probes 32 prior draws before nested sampling (`dynesty`/`tinyns`) and prints `preflight: k/32 prior draws have finite logL`. If all 32 are `-inf` it raises immediately instead of letting the sampler reject-sample forever; `off` skips the probe.

#### If dynesty cannot find initial live points

`dynesty`/`tinyns` seed a run by rejection-sampling the prior until they collect `--nlive` points with a finite `logL`. When the selection variance guard (`sigma^2_lnL > --max_likelihood_variance`, or the `Neff > 5*N_obs` floor) returns `-inf` across the whole prior, that step hangs with nothing in the log. The `--sampler_preflight` probe now catches this up front — a `preflight: 0/32 prior draws have finite logL` message and a hard error naming the criterion. To recover, either switch to `--selection_neff_guard soft` (a finite penalized wall the sampler can initialize on and that pushes it toward the valid region) or raise `--max_likelihood_variance` to accept a larger Monte-Carlo variance. To measure `sigma^2_lnL` on your own data and get the smallest admitting cap, run `python scripts/diagnose_selection_guard.py -- <your darksirens_inference args>`.

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

When using `--sampler tinyns`, Darksirens prints a compact TinyNS diagnostics
summary and saves full diagnostics in `results.hdf5` and
`tinyns_diagnostics.json`. These fields help diagnose slow runs: `niter/sec`,
`ncall/sec`, `calls/iter`, replacement batches, replacement failures, rescue
usage, final `dlogz`, live-weight fraction, and insertion-rank diagnostics when
available.


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

## TinyNS short-budget benchmark sweep

The repository includes a lightweight benchmark driver at
`scripts/benchmark_tinyns_darksirens_short_budget.py` for comparing a small set
of TinyNS-only settings on realistic Darksirens inputs.  It launches
`darksirens_inference` repeatedly, keeps each run under a per-configuration wall
clock limit, and writes simple logs plus `summary.csv` and `summary.json`.

The benchmark relies on the main `darksirens_inference` output path writing
`tinyns_diagnostics.json` next to `results.hdf5`; this is the preferred source
for speed and health fields.  If that sidecar is missing, the script falls back
to TinyNS attributes in `results.hdf5`, then to limited stdout parsing.

Example spectral-sirens short-budget sweep, with no survey catalog required:

```bash
python scripts/benchmark_tinyns_darksirens_short_budget.py \
  --gw-path "$GWE" \
  --gwselection-path "$SEL" \
  --universe-model spectral_sirens \
  --pop-model powerlaw+peak \
  --base-save-path "$R/tinyns_bench" \
  --nlive 400 \
  --dlogz 0.5 \
  --max-samples 2000 \
  --seed 21 \
  --timeout-minutes 20 \
  --extra-arg --fixed_cosmology --extra-arg true \
  --extra-arg --fix_survey --extra-arg true
```

Example dark-sirens sweep with a galaxy catalog:

```bash
python scripts/benchmark_tinyns_darksirens_short_budget.py \
  --gw-path "$GWE" \
  --gwselection-path "$SEL" \
  --survey-path "$CAT" \
  --universe-model dark_sirens \
  --pop-model powerlaw+peak \
  --base-save-path "$R/tinyns_bench" \
  --nlive 400 \
  --dlogz 0.5 \
  --max-samples 2000 \
  --seed 21 \
  --timeout-minutes 20
```

By default the sweep changes only TinyNS settings: `recommended`, `cheap`,
`fast16`, `fast32`, and `fast16_B128`.  It always passes `--sampler tinyns`, the
requested `--nlive`, `--dlogz`, `--max_samples`, `--seed`,
`--tinyns_progress_interval 10`, and a unique `--save_path` for each
configuration.  It does not alter normalization-grid settings by default.

Custom sweeps can be supplied with `--sweep-json`, for example:

```json
[
  {
    "name": "my_config",
    "tinyns_replacement_chains": 16,
    "tinyns_walks": 20,
    "tinyns_step_scale": 0.03,
    "tinyns_min_accepts": 3,
    "tinyns_jax_block_size": 64,
    "tinyns_max_attempts": 100000
  }
]
```

Each benchmark directory is named like
`tinyns_short_budget__YYYY-MM-DDTHH-MM-SS/` and contains one subdirectory per
configuration, `stdout.log`, `stderr.log`, `command.txt`, `summary.csv`, and
`summary.json`.  Summary fields include status, return code, elapsed time,
TinyNS settings, evidence fields, iteration/call rates, replacement statistics,
rescue usage, insertion-rank diagnostics, and a short parser message.

Interpretation guidelines:

- `niter_per_sec` is the primary short-budget speed indicator printed in the
  ranking table.
- `replacement_mean_batches` and `replacement_max_batches` indicate how hard it
  was to replace live points; large values suggest the configuration may be
  inefficient for the target likelihood.
- `replacement_failures` records replacement failures and should be treated as a
  health warning.
- `replacement_rescue_used` indicates that rescue logic was needed; the ranking
  table flags this and avoids selecting rescued configurations as healthy
  candidates.
- `final_delta_logz` is useful context for how close the short run came to its
  stopping target.

This benchmark is a short-budget diagnostic for TinyNS speed and sampler health,
not final evidence validation.  Promising configurations should still be checked
with production-quality budgets and normal scientific validation.
