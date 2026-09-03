# Troubleshooting

This page answers one question: a run failed or warned, what caused it and what
do you change. Flags are documented in [Inference](inference.md),
[Lensing](lensing.md) and the [CLI reference](../reference/cli.md).

## Startup and data

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError` for `dynesty` / `numpyro` / `tinyns` | sampler backends are imported lazily and only when selected | install the one you asked for, or switch `--sampler` |
| `gwcat is required to load gwcat PE/selection files` | `--gw_path` / `--gwselection_path` point at gwcat files and `gwcat` is not installed | install `gwcat` (see `requirements.txt`), or supply precomputed prior weights |
| `the installed gwcat predates the -inf out-of-support convention` | too old a `gwcat` for the `chi_eff` prior contract | bump `gwcat` to the pin in `requirements.txt` |
| `KeyError` on a raw survey file | `darksirens_pixelate` requires `TARGET_RA`, `TARGET_DEC`, `Z`, `ZERR`, `WEIGHT` | rename the datasets in the input file |
| `KeyError` on a pixelated survey file | `--survey_path` files must carry `zgals`, `dzgals`, `wgals`, `ngals` and the `nside` attribute | re-run `darksirens_pixelate` on the raw table |
| `ZERR must be finite and non-negative` / `WEIGHT must be finite and strictly positive` | failure sentinels (`Z = -1`, `ZERR = -1`) survived into the raw table | drop the failed rows before pixelating |

## Memory

| Symptom | Cause | Fix |
|---|---|---|
| out-of-memory on a GPU | the `auto` plan is sized from probed free memory, which is blind to another job started after the probe | pin `--sel_batch_size` and `--pe_event_block`, or add `--drop_full_catalog true` (not with `--use_lss` or bright sirens) |
| out-of-memory on a CPU with default flags | `auto` blocks only on a GPU-class backend; elsewhere `resolve_block_sizes` returns a single pass tagged `cpu` and host RAM is never probed | pin all three of `--sel_batch_size`, `--pe_event_block`, `--row_chunk` (start at `--row_chunk 512`) |
| host RSS spike while building the catalog kernel state | the full-row vmap holds `(N_rows, N_max_gals, K)` at once | `--row_chunk 512` |
| analyzer runs out of memory | posterior-predictive grids are per-dimension | lower `--nm`, `--nq`, `--nz`, `--nchi`; keep `--norm_nmass`, `--norm_nq`, `--norm_nchi` dimension-specific rather than raising all of them |
| the cluster enforces its own JAX memory policy | the entry point sets `XLA_PYTHON_CLIENT_PREALLOCATE` and `XLA_PYTHON_CLIENT_ALLOCATOR` with `setdefault` | export the values you want in the job script before launching |

See [Performance](performance.md) for what each block-size knob costs.

## Likelihood returns `-inf`

| Symptom | Cause | Fix |
|---|---|---|
| the run hangs at "find initial live points", or preflight reports `0/32 prior draws have finite logL` | the selection guard: `N_eff` fails `max(5 N_obs, N_obs^2 / (max_likelihood_variance - sum_i sigma_i^2))`, so `selection_log_correction` returns `-inf` | run `python scripts/diagnose_selection_guard.py` with your own CLI line; it reports the measured `sigma^2_lnL` and the smallest `--max_likelihood_variance` that admits it. The real fix is more injections |
| every NUTS trajectory is flagged divergent | the hard `-inf` wall cannot be crossed by a gradient sampler | `--selection_neff_guard soft` (the `auto` default for `--sampler numpyro`), then check post hoc that the posterior clears the `N_eff` boundary |
| `logL` is `-inf` at every point of a frozen-prior run | `--freeze_redshift_prior true` re-verifies its premise in the graph and poisons the likelihood when the fixed cosmology or survey scalars differ from the build values | re-run with `--freeze_redshift_prior false` to confirm, then report it: with the label gate satisfied the two must agree |

## Warnings worth acting on

| Symptom | Cause | Fix |
|---|---|---|
| `--kde_window <W> is below the data-sized window <N>` | a pinned window is centred by index and never repositioned, so it truncates the in-range galaxy block on the densest rows and the catalog prior is wrong, not just cheaper | unset `--kde_window` to size from the data, or pin at or above the reported `N` |
| `catalog KDE window sized from the data: <N> galaxies` with `N > 1024` | a dense photo-z catalog needs a wide window for the exact answer | nothing is wrong; budget the per-sample cost, or trade accuracy deliberately with `--kde_window` |
| `pair y-quadrature:` a delta above tolerance between 32 and 128 nodes | the SIS pair integrand is peaked and its Gauss-Legendre convergence is pair-dependent | raise `--y_nodes_pair` when the delta is comparable to the evidence difference being measured |
| `DeprecationWarning` naming a population model | old spellings such as `twopowerlaws+peak` or `gwtc5_fiducial_brokenpowerlaw+2peaks` still resolve | switch to the canonical name (`2powerlaws+peak`, `gwtc5_fiducial_bpl2peaks`); saved `settings.json` and HDF5 `pop_model` attributes with old names stay readable |

## Refusals from the lensing CLI

| Symptom | Cause | Fix |
|---|---|---|
| `Refusing to INFER lensing-rate parameters under the both-detected pair approximation` | every kept pair has `p_tag = 1`, an upper bound on the true pair-detection probability, which overestimates `mu_sel^(2)` and biases `log10_tau_A` / `n_tau` low, exactly the quantities `--fix_lens_rate false` samples | supply a calibrated efficiency (`--pair_tag_model file --pair_tag_selection_path ...`), or `--fix_lens_rate true`, or acknowledge with `--allow_both_detected_approx true` (stamped into `settings.json` and `results.hdf5`) |
| `--pair_marks time: a candidate/pair time mark ... reaches the observing-run length t_obs` | two arrivals from one run cannot be that far apart, so either the mark or `t_obs_days` is wrong; the unlensed coincidence density vanishes there and the pair would be rewarded by ~+20 nats | fix the time base, or regenerate the observed catalog with the right `t_obs_days` |
| `--pair_marks time requires an observed catalog with observation_times='uniform'` | the time-mark coincidence odds need the run length | regenerate the mock with `--observation-times uniform` |
| `suspicious candidate time marks` | placeholder marks (integer-second `delta_t` with `sigma_delta_t = 1 s`) are astronomically sharp on the SIS time-delay scale and dominate the y-integral | fix the candidate-pair file, or `--allow_suspicious_time_marks true` |
| `non-finite candidate time marks` / `time-mark magnitude disagreement` | a NaN or inf delay, or a mark whose magnitude does not reproduce the catalog's arrival separation | fix the candidate-pair file, or `--allow_time_mark_mismatch true` after checking the two agree on which event came first |
| `--partition_mode marginalize_exact requires --cluster_mode j2` / `--candidate_pairs_path` | exact partition marginalisation has no pairs to marginalise over otherwise | supply the pair inputs, or use `--partition_mode fixed --partition_path ...` |

## Parameters and model names

| Symptom | Cause | Fix |
|---|---|---|
| `--pop_model` name rejected | the name is parsed as a composition grammar; the error lists known component tokens and suggests close matches | write `powerlaw+peak` or `powerlaw+2peaks`, never a bare plural (`powerlaw+peaks`); do not append `_shared_*` suffixes, use `--shared_beta`, `--shared_spin`, `--shared_gamma`. A name that parses but is not curated (e.g. `powerlaw+4peaks`) builds with blueprint-default priors and only logs a message |
| a JSON override or fixed value is ignored, or the label is unknown | population labels carry their component tag in a mixture (`\alpha_{\rm PL}`, `\mu_{\rm G2}`) | run with a tiny sampler configuration, read the printed parameter table, and copy those labels exactly. See [Populations](populations.md) |
