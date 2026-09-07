# CLI reference

Every console script the package installs, with its synopsis and its complete
option set as reported by `--help`. Task-oriented walkthroughs live in
[Inference](../guide/inference.md), [Catalogs](../guide/catalogs.md),
[Lensing](../guide/lensing.md) and [Analysis](../guide/analysis.md). `BOOL`
options take `true`/`false`, `flag` options take no value, `none` marks an unset
default, and every script also runs as `python -m darksirens.cli.<module>`.

## `darksirens_inference`

Runs the spectral-siren, dark-siren and bright-siren hierarchical inference and
writes a run directory. Module form: `python -m darksirens.cli.inference`.

```bash
darksirens_inference \
  --gw_path gw_events.h5 \
  --gwselection_path gw_selection.h5 \
  --sampler dynesty
```

Required: `--sampler`, one of `--gw_path` / `--gw_flows_path`, and one of
`--gwselection_path` / `--pdet_flow_path`.

### Data

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--gw_path` | path | none | gwcat PE posterior-samples file (`gwcat-1.0`, `gwcat-pe-2.0`, `gwcat-pe-2.1`) |
| `--gw_flows_path` | path | none | Directory of per-event flow checkpoints replacing stored PE samples; spectral sirens only |
| `--gwselection_path` | path | none | gwcat selection/injection file (`gwcat-selection-1.0`, `-2.0`, `-2.1`) |
| `--allow_invalid_spin_swap` | flag | off | Ablation: load a chi_eff selection file whose campaign evidence invalidates the analytic spin swap |
| `--survey_path` | path(s) | none | Galaxy survey catalog(s); several paths define a K-catalog dark-siren mixture |
| `--save_path` | path | `./` | Directory in which the run directory is created |

### Flow surrogates (with `--gw_flows_path`)

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--flows_nsamp` | int | 16384 | Population draws J per event per likelihood call (common random numbers) |
| `--flows_seed` | int | 42 | Seed of the fixed base-uniform array |
| `--flows_pattern` | str | `*/*_flow.npz` | Checkpoint glob relative to `--gw_flows_path` |
| `--flows_on_mismatch` | `error`, `skip`, `load` | `error` | Structural-check policy for checkpoints that do not match the installed flowjax |
| `--flows_chieff_amax` | float | 0.99 | amax of the 1-D isotropic chi_eff PE prior |
| `--flows_pe_cosmology` | `H0,Om0` | `67.74,0.3089` | PE prior cosmology of the UniformSourceFrame distance prior |
| `--flows_grid_nm` | int | 512 | m1 cells of the (m1, q) population sampling grid |
| `--flows_grid_nq` | int | 256 | q cells of the (m1, q) population sampling grid |
| `--flows_support_margin` | float | 0.25 | Fractional per-side expansion of each flow's sampled parameter range |
| `--flows_support_nsamples` | int | 4096 | Flow draws used to measure each event's support box |
| `--flows_wfull` | float | 0.05 | Fraction of draws taken from the full population support instead of the event window |

### Selection-function emulator (alternative to `--gwselection_path`)

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--pdet_flow_path` | path | none | NF P_det emulator checkpoint driving the selection integral through pseudo-injections |
| `--pdet_nsamp` | int | 1000000 | Pseudo-injections drawn from the emulator flow at load |
| `--pdet_seed` | int | 42 | Seed of the emulator pseudo-injection draw |
| `--pdet_cosmology` | `H0,Om0` | `67.9,0.3065` | Injection-campaign cosmology used for z to dL and the Jacobian |
| `--pdet_chieff_amax` | float | 0.99 | Reference amax of the chi_eff spin-prior swap |

### Physical model

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--universe_model` | `spectral_sirens`, `dark_sirens`, `dark_sirens_complete`, `bright_sirens` | `spectral_sirens` | Redshift-prior regime of the run |
| `--sky_model` | `isotropic`, `dipole`, `sphere_gp`, `sphere_gp_z`, `overdensity_gp`, `multipole`, `multipole_l3` | `isotropic` | Sky distribution of the source rate; compared to isotropy by evidence |
| `--mark_model` | `none`, `loglinear` | `none` | Marked-host model reweighting catalog galaxies by a BBH-host efficiency |
| `--marks` | comma list | all present | Marks used by `--mark_model loglinear`: `logmstar`, `logssfr`, `metallicity`, `color` |
| `--pop_model` | str | `powerlaw+peak` | Population model composition; mass tokens joined by `+` with count prefixes |
| `--shared_beta` | BOOL | true | One shared beta/pairing distribution; false gives one per mass component |
| `--shared_spin` | BOOL | true | One shared spin distribution; false gives one per mass component |
| `--shared_gamma` | BOOL | true | One shared redshift-evolution gamma; false gives one per mass component |
| `--fix_population` | BOOL | false | Fix the population block at the curated fiducial vector |
| `--population_fiducials` | `legacy`, `in_prior_v2` | `legacy` | Which curated fiducial vector `--fix_population` uses; `in_prior_v2` is the in-prior set |
| `--allow_skymap_population` | BOOL | false | Ablation: free population on a `darksirens_skymaps_to_samples` PE file |
| `--fix_cosmology` / `--fixed_cosmology` | BOOL | false | Fix the full cosmology block (H0, Om0, w0, wa); `--fixed_cosmology` is deprecated |
| `--fix_de` / `--fixed_de` | BOOL | false | Fix only w0 and wa; ignored when `--fix_cosmology` is true |
| `--fix_survey` | BOOL | false | Fix the survey/completeness block |
| `--redshift_prior_barrier` | `auto`, `on`, `off` | `auto` | Internal redshift-prior optimization barrier; `auto` disables it for TinyNS JAX rwalk |
| `--selection_neff_guard` | `auto`, `hard`, `soft` | `auto` | Sparse-selection validity guard: hard -inf wall or smooth penalty |
| `--max_likelihood_variance` | float | 1.0 | Cap on the Monte-Carlo variance of the total log-likelihood estimator |
| `--prior_overrides` | JSON | none | JSON dict of `{label: [lo, hi]}` prior bounds |
| `--fixed_parameter_values` | JSON | none | JSON dict of `{label: value}` fixed parameters |
| `--counterpart` | RA DEC Z triplet(s) | none | Bright-siren counterpart positions ordered by event; angles in radians |
| `--counterpart_dz` | float | 1e-4 | Gaussian redshift uncertainty for `--counterpart` |
| `--counterpart_nside` | int | 1 | HEALPix NSIDE of the synthetic bright-siren counterpart catalog |
| `--bright_siren_sky_marginalized` | BOOL | false | Ignore the counterpart sky-pixel gate and apply only its redshift prior |
| `--complete_empty_pixel_policy` | `zero`, `volume` | `zero` | Empty-pixel policy for complete-catalog models; `volume` is the historical approximation |

### Catalog

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--use_lss` / `--use_LSS` | BOOL | false | Enable the legacy local-overdensity missing-galaxy factor |
| `--lss_floor` | `conserve`, `legacy` | `conserve` | Number-conservation policy of that floored factor; `legacy` leaves it unrenormalized |
| `--catalog_sky_weighting` | `conditional`, `field` | `field` | Catalog redshift-prior normalization; `conditional` discards relative angular host density |
| `--survey_z_depth` | float | file attr | Survey redshift depth as a completeness prior; overrides every per-catalog `z_depth` |
| `--c_mode` | `per_pixel`, `aggregate`, `selection` | `per_pixel` | Completeness estimator: per-pixel kernel ratio, sky-aggregate curve, or parametric C_sel |
| `--selection_fit` | JSON path list | none | `selection_fit.json` per catalog in `--survey_path` order; declares the luminosity-function family |
| `--stratum_map` | path | none | Full-sky `stratum_map` HDF5 required by a multi-stratum `--selection_fit` |
| `--per_pixel_completeness` | path | none | Depth-map HDF5 supplying the per-pixel selection fraction `f_p = 1 - masked_frac` |
| `--allow_double_counted_mask` | flag | off | Ablation: pair `--per_pixel_completeness` with a Q table that lacks `f_p_aware` |
| `--allow_selection_fit_free_background` | flag | off | Ablation: sample Om0/w0/wa while a magnitude `--selection_fit` prior is active |
| `--allow_unmasked_footprint` | flag | off | Ablation: aggregate/selection completeness on a footprint-limited catalog with no mask |
| `--validate_completion` | BOOL | false | Run the completion clipping diagnostic, save JSON, and exit before the likelihood |
| `--completion_validation_pixels` | int | 64 | Maximum unique catalog pixels inspected by `--validate_completion` |
| `--lss_completion` | path(s) | in-catalog group | Precomputed LSS completion file(s), positionally aligned with `--survey_path` |
| `--lss_field_mode` | `table`, `latent` | `table` | Source of the missing-galaxy modulation Q: resident log-Q table or in-likelihood latent field |
| `--lss_field_artifact` | path | none | Latent-field anchor artifact; required by and only legal with `--lss_field_mode latent` |
| `--lss_field_sha256` | hex | none | Pin the identity of `--lss_field_artifact`; a mismatch is fatal |
| `--allow_unanchored_budget` | flag | off | Ablation: latent mode with a flat, uncalibrated `log10n0` or `delta` prior |
| `--lss_marginalize` | BOOL | false | Marginalize the GW likelihood over the Q_LSS ensemble instead of the posterior-mean Q |
| `--allow_unverified_shared_lss_members` | BOOL | false | Ablation: marginalize K>=2 ensembles that do not share a realization set |

### Sampler

TinyNS (`--tinyns_*`) and NUTS (`--nuts_*`) options are listed under
[Shared TinyNS and NUTS options](#shared-tinyns-and-nuts-options).

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--sampler` | `tinyns`, `dynesty`, `numpyro` | required | Sampler backend |
| `--nlive` | int | 1000 | Live points for the nested samplers |
| `--dlogz` | float | 0.1 | Evidence tolerance stopping criterion |
| `--max_samples` | int | 1000000 | Call/iteration budget for nested samplers; 0 = unlimited |
| `--seed` | int | 22 | Random seed |
| `--show_progress` | BOOL | true | Print sampler progress |
| `--dynesty_diagnostics` | BOOL | false | Write dynesty runplot/traceplot PDFs every 10 minutes |
| `--prior_transform_dispatch` | `auto`, `eager` | `auto` | How dynesty calls the prior transform; `eager` forces the op-by-op device path |
| `--sampler_preflight` | `on`, `off` | `on` | Probe up to 32 prior draws (stopping at the 4th finite one) and fail fast if the likelihood is everywhere -inf |
| `--checkpoint_interval` | seconds or `off` | 1800 | Seconds between sampler checkpoints written into the run directory |
| `--resume` | `auto`, PATH, `off` | `off` | Restore a checkpointed run; `auto` picks the newest matching checkpoint |
| `--resume_force` | flag | off | UNSAFE: resume although `run_fingerprint.json` is missing or mismatched |

### Performance

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--sel_batch_size` | N, `auto`, `off` | `auto` | Injections per selection-integral chunk; `off` forces a single pass |
| `--pe_event_block` | N, `auto`, `off` | `auto` | Events per PE-reduction chunk; `off` keeps one vectorized block |
| `--drop_full_catalog` | BOOL | false | Discard the dense full-sky galaxy arrays after compacting to inference pixels |
| `--norm_nmass` | int | env / module | Mass-grid size for GW-population normalisation (`DARKSIRENS_GW_N_MASS`) |
| `--norm_nq` | int | env / module | Mass-ratio-grid size for the GP baselines and the stratified-q normalisation (`DARKSIRENS_GW_N_Q`); it no longer sizes the pairing normaliser (fixed two-panel Gauss-Legendre rule) |
| `--norm_nchi` | int | env / module | Spin-grid size for GW-population normalisation (`DARKSIRENS_GW_N_CHI`) |
| `--pairing_norm_grid` | int | exact | Interpolate the pairing q-normalization from an N-node log-spaced m1 grid |
| `--kernel_gl_nodes` | int | 24 | Gauss-Legendre nodes for the per-galaxy kernel normalisation Z_i |
| `--kernel_gl_domain` | `cdf`, `zspace` | `cdf` | Quadrature domain of the kernel normalisation; `zspace` avoids ndtri |
| `--kernel_gl_nsigma` | float | 5.0 | Half-width of the z-space kernel window in units of sigma_eff (`zspace` only) |
| `--kde_window` | int | data-sized | Galaxies nearest each sample's redshift evaluated by the catalog KDE; 0 disables windowing |
| `--freeze_redshift_prior` | BOOL | true | Evaluate the catalog redshift prior once at build time for population-only runs |
| `--kde_window_nsigma` | float | 8 | Build-time sizing multiplier of the KDE window in sigma_eff units |
| `--row_chunk` | `auto`, `off`, N | `auto` | Row-chunking for catalog kernel-state builds (`lax.map` over N-row chunks) |

## `darksirens_inference_lensing`

Runs the strong-lensing branch: the singleton plus J=2 cluster likelihood, and
the weak-lensing (`spectral_sirens_wl`) universe model this CLI alone owns.
Module form: `python -m darksirens.cli.inference_lensing`.

```bash
darksirens_inference_lensing \
  --gw_path gw_events.h5 \
  --gwselection_path gw_selection.h5 \
  --sampler dynesty
```

### Data

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--gw_path` | path | required | gwcat PE posterior-samples file |
| `--gwselection_path` | path | required | gwcat selection/injection file |
| `--lensed_injections_path` | path | none | Pre-rendered lensed-injection set for the cluster and singleton-mixture selection |
| `--pair_pe_path` | path | none | Deprecated split-pair layout; use the unified observed catalog instead |
| `--pair_metadata_path` | path | none | Optional pair/candidate-edge metadata file, preferred in unified observed mode |
| `--partition_path` | path | none | Fixed pair partition to evaluate |
| `--candidate_pairs_path` | path | none | Candidate-pair JSON with edge log prior odds and marks |
| `--observed_catalog_path` | path | none | Explicit `observed_catalog.json` for unified observed lensing mode |
| `--partition_mode` | `fixed`, `marginalize_exact` | `fixed` | Use one partition or marginalize exactly over compatible matchings |
| `--max_exact_partitions` | int | 10000 | Cap on enumerated partitions in exact marginalization |
| `--partition_component_mode` | `global`, `componentwise` | `componentwise` | Marginalize over the whole graph or per connected component |
| `--max_component_events` | int | none | Cap on events per connected component |
| `--max_component_edges` | int | none | Cap on candidate edges per connected component |
| `--max_component_partitions` | int | none | Cap on partitions enumerated per component |
| `--max_total_partitions` | int | none | Cap on the product of per-component partition counts |

### Model

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--pop_model` | str | `powerlaw+peak` | Population model composition (same grammar as `darksirens_inference`) |
| `--cluster_mode` | `off`, `j2` | `j2` | Enable the J=2 (double-image) cluster channel |
| `--wl_backend` | `lognormal`, `tabulated`, `disabled` | `lognormal` | Weak-lensing magnification PDF backend for the singleton channel |
| `--lensing_wl_table_path` | path | none | HDF5 table of log p_WL(mu given z) for `--wl_backend tabulated` |
| `--wl_selection` | `auto`, `standard`, `wl_lognormal` | `auto` | Singleton selection treatment; `auto` matches the event-side `--wl_backend` |
| `--allow_mismatched_wl_selection` | flag | off | UNSAFE ablation: normalize the hierarchy under a different observation model |
| `--lensing_wl_a` | float | 0.004 | Amplitude `a` of the lognormal WL variance `s^2 = a z^b` |
| `--lensing_wl_b` | float | 1.5 | Redshift exponent `b` of the lognormal WL variance `s^2 = a z^b` |
| `--sl_tau_A` | float | 5e-4 | Amplitude `A` of the SIS optical depth `tau_2 = A z^n` |
| `--sl_tau_n` | float | 3.0 | Redshift exponent `n` of the SIS optical depth `tau_2 = A z^n` |
| `--sl_T0_sec` | float | 5.36e6 | SIS time-delay scale T0 in seconds (Delta t = T0 y) |
| `--fix_lens_rate` | BOOL | true | Fix the SIS optical-depth parameters instead of sampling lensing hyperparameters |
| `--lens_prior_overrides` | JSON | none | JSON dict of SIS lens prior overrides |
| `--pair_marks` | `none`, `time` | `none` | Optional J=2 pair marks; `time` uses the candidate-pair or metadata time marks |
| `--pair_time_sigma_sec` | float | none | Fallback sigma_delta_t when pair time metadata omits sigma |
| `--pair_tag_model` | `constant`, `snr_only`, `snr_sky`, `snr_time`, `snr_time_sky`, `file` | `constant` | Pair-tagging probability model |
| `--pair_tag_constant` | float | 1.0 | Constant tagging probability for `--pair_tag_model constant` |
| `--pair_tag_perturb_logit` | float | 0.0 | Logit perturbation applied to the tagging probability |
| `--pair_tag_selection_path` | path | none | Tag-selection file for `--pair_tag_model file` |
| `--allow_both_detected_approx` | BOOL | false | Ablation: keep p_tag = 1 while inferring lensing rates (biases optical depth low) |
| `--edge_mark_prior_keys` | comma list | empty | `log_*` candidate edge marks added to edge log prior odds |
| `--edge_mark_likelihood_keys` | comma list | empty | Edge mark likelihood keys; only `time`/`delta_t_obs` is implemented |
| `--allow_suspicious_time_marks` | BOOL | false | Ablation: downgrade the placeholder/synthetic time-mark error to a warning |
| `--allow_time_mark_mismatch` | BOOL | false | Ablation: downgrade non-finite or catalog-disagreeing time-mark errors to warnings |
| `--pair_time_mark_impl` | `auto`, `quadrature`, `delta` | `auto` | Time-mark y-integral implementation; `auto` delta-collapses sharp marks |
| `--singleton_lensing` | `off`, `sl_mixture` | `off` | Model observed singletons as an unlensed plus one-image-detected lensed mixture |
| `--y_nodes_single` | int | 32 | Gauss-Legendre y nodes for the lensed-singleton evidence |
| `--fc_rho_thr` | float | file attr | Override the injection file's `fc_rho_thr` attribute |
| `--fc_r0` | float | file attr | Override the injection file's `fc_r0` attribute |
| `--fc_mc_bar` | float | file attr | Override the injection file's `fc_mc_bar` attribute |
| `--pair_orientation_mode` | `independent`, `shared_iota` | `independent` | Two-image orientation convention of the lensed-singleton censoring factor |
| `--allow_pair_orientation_mismatch` | BOOL | false | Ablation: downgrade the campaign-versus-runtime orientation mismatch to a warning |

### Fixing

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--fix_cosmology` | BOOL | true | false samples H0/Om0/w0/wa; refused with `--cluster_mode j2` or `sl_mixture` |
| `--fix_survey` | BOOL | true | Fix the survey/completeness block |
| `--fix_population` | BOOL | false | Fix the population block |
| `--fixed_parameter_values` | JSON | none | JSON dict of `{label: value}` |
| `--prior_overrides` | JSON | none | JSON dict of `{label: [lo, hi]}` |
| `--redshift_prior_barrier` | `auto`, `on`, `off` | `auto` | Internal redshift-prior optimization barrier; `auto` disables it for TinyNS JAX rwalk |
| `--selection_neff_guard` | `auto`, `hard`, `soft` | `auto` | Sparse-selection guard for the combined singleton plus cluster correction |
| `--max_likelihood_variance` | float | 1.0 | Cap on the total Monte-Carlo variance of the log-likelihood estimator |

### Sampler

TinyNS and NUTS options are listed under
[Shared TinyNS and NUTS options](#shared-tinyns-and-nuts-options).

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--sampler` | `tinyns`, `dynesty`, `numpyro` | required | Sampler backend |
| `--nlive` | int | 2000 | Live points for the nested samplers |
| `--dlogz` | float | 0.1 | Evidence tolerance stopping criterion |
| `--max_samples` | int | 2000000 | Call/iteration budget for nested samplers |
| `--prior_transform_dispatch` | `auto`, `eager` | `auto` | How dynesty calls the prior transform; `eager` forces the device path |
| `--checkpoint_interval` | seconds or `off` | 1800 | Seconds between sampler checkpoints written into the run directory |
| `--resume` | `auto`, PATH, `off` | `off` | Restore a checkpointed run; `auto` picks the newest matching checkpoint |
| `--resume_force` | flag | off | UNSAFE: resume although `run_fingerprint.json` is missing or mismatched |

### Performance

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--pe_max_per_pair` | int | 400 | Down-sample PE per pair image; 0 keeps all and drives pair-KDE memory |
| `--pair_kde_bandwidth_scale` | float | 1.0 | Multiplier on the pair KDE's Silverman bandwidth |
| `--pair_batch_size` | int | 0 | Candidate-pair batch size for J=2 scans; 0 keeps the unbatched path |
| `--y_nodes_pair` | int | 32 | Gauss-Legendre y nodes for each J=2 pair likelihood |
| `--sel_batch_size` | N, `auto`, `off` | `auto` | Injections per selection-integral chunk; `off` forces a single pass |
| `--pe_event_block` | N, `auto`, `off` | `auto` | Singletons per vectorized PE-reduction chunk; 1 reproduces the per-event scan |
| `--norm_nmass` | int | env / module | Mass-grid size for GW-population normalisation (`DARKSIRENS_GW_N_MASS`) |
| `--norm_nq` | int | env / module | Mass-ratio-grid size for the GP baselines and the stratified-q normalisation (`DARKSIRENS_GW_N_Q`); it no longer sizes the pairing normaliser (fixed two-panel Gauss-Legendre rule) |
| `--norm_nchi` | int | env / module | Spin-grid size for GW-population normalisation (`DARKSIRENS_GW_N_CHI`) |
| `--pairing_norm_grid` | int | exact | Interpolate the pairing q-normalization from an N-node log-spaced m1 grid |

### Output

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--seed` | int | 42 | Random seed |
| `--show_progress` | BOOL | false | Print sampler progress |
| `--save_path` | path | `./` | Directory in which the run directory is created |
| `--preflight_only` | BOOL | false | Run the lensing input preflight checks, write JSON, and exit before sampling |
| `--preflight_json` | path | `save_path/preflight.json` | Output path for the preflight JSON |

## `darksirens_analyze`

Post-processes and plots one or more inference runs: posterior-predictive
population spectra, cosmology posteriors, dN/dz, a GP-latent summary and
relative evidences. Module: `python -m darksirens.cli.analyze`.

```bash
darksirens_analyze --run_dirs run_a run_b --outdir figures/
```

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--run_dirs` | path(s) | `.` | One or more run directories |
| `--mmin`, `--mmax` | float | 1.0, 100.0 | Primary-mass range of the posterior-predictive grid |
| `--nm`, `--nq`, `--nz`, `--nchi` | int | 128, 48, 32, 24 | Grid sizes in m1, q, z and chi |
| `--zmax` | float | 2.0 | Upper redshift of the posterior-predictive grid |
| `--chimin`, `--chimax` | float | -1.0, 1.0 | Effective-spin range of the grid |
| `--batch_size` | int | auto | Per-sample batch size; auto-chosen from device memory if unset |
| `--grid_chunk` | int | auto | Rows of the flattened (m1, q) plane per density-evaluation slab |
| `--max_mem_gb` | float | probed | Memory budget in GB for posterior-predictive sizing |
| `--mem_safe_frac` | float | 0.4 | Fraction of the memory budget the posterior-predictive peak may use |
| `--cred_lo`, `--cred_hi` | float | 5.0, 95.0 | Credible-band percentiles |
| `--overlay_events` | flag | off | Overlay observed detector-frame m1 medians on p(m1) |
| `--sky_nside` | int | 16 | HEALPix nside for the `sphere_gp` posterior sky map |
| `--outdir` | path | `.` | Directory for output figures |
| `--allow_legacy_pickle` | flag | off | UNSAFE: read a very old pickled-dict `samples.npy` (arbitrary code execution) |

## `darksirens_pixelate`

Processes galaxy survey data into HEALPix pixels, writing the pixelated catalog
that dark-siren runs read. Module form: `python -m darksirens.cli.pixelate`.

```bash
darksirens_pixelate --survey_path survey_raw.h5 --save_path . --nside 32
```

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--survey_path` | path | required | Path to the input HDF5 survey data |
| `--save_path` | path | `./` | Directory to save the outputs |
| `--nside` | int | 64 | HEALPix Nside parameter |
| `--add_plots` | flag | off | Generate diagnostic plots |
| `--z_depth` | float | none | Survey redshift depth written as `f.attrs['z_depth']` and read as a completeness prior |

## `darksirens_skymaps_to_samples`

Converts 3-D LVK skymaps into a darksirens GW importance-sample HDF5.
Module form: `python -m darksirens.cli.skymaps_to_samples`.

```bash
darksirens_skymaps_to_samples --skymap_dir skymaps/ --output gw_events.h5
```

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--skymap_dir` | path | required | Directory of per-event 3D skymap FITS files |
| `--output` | path | required | Output `gwdata.h5` path |
| `--nsamp` | int | 5000 | Samples per event; larger reduces surrogate-mass MC noise |
| `--seed` | int | fresh OS seed | RNG seed; a drawn seed is printed and stamped as `attrs['seed']` |
| `--pattern` | str | `*.fits*` | Glob for skymap files |
| `--zmax` | float | 1.5 | Declared max redshift of the analysis; a floor for the default `m1det_max` |
| `--h0_max` | float | 120.0 | Upper edge of the H0 prior the file will be analysed under |
| `--pop_m_min` | float | 2 | Lower edge of the population `m_min` prior |
| `--pop_m_max` | float | 100 | Upper edge of the population `m_max` prior |
| `--m1det_min` | float | `pop_m_min` | Surrogate `m1det` lower bound |
| `--m1det_max` | float | derived | Surrogate `m1det` upper bound: `pop_m_max*(1+max(zmax, z at --h0_max))` |
| `--q_min` | float | 0.05 | Lower bound of the surrogate mass-ratio draw |
| `--chi_abs_max` | float | 0.99 | Bound on the absolute surrogate effective spin |
| `--n_dist_grid` | int | 1024 | Grid resolution for the dL ansatz inverse-CDF |
| `--pe_H0` | float | Planck15 | Fiducial H0 used only to invert dL samples to redshift |
| `--pe_Om0` | float | Planck15 | Fiducial Om0 for the same dL to z inversion |

## `darksirens_build_lognormal_completion`

Builds an LSS-conditioned lognormal completion file `Q_LSS(p, z)` offline from a
pixelated catalog. Module form:
`python -m darksirens.cli.build_lognormal_completion`.

```bash
darksirens_build_lognormal_completion \
  --catalog catalog_nside_32.h5 \
  --out lss_completion.h5
```

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--catalog` | path | required | Pixelated survey catalog HDF5 (`load_survey` schema) |
| `--out` | path | required | Output completion HDF5 path |
| `--n-members` | int | 32 | Laplace/FFT-diagonal ensemble members; 0 = MAP only |
| `--seed` | int | 1234 | Seed of the ensemble member draws |
| `--prior-strength` | float | 1.0 | Scaling of the latent-field prior in the MAP solve |
| `--maxiter` | int | 200000 | Max L-BFGS-B iterations per pixel MAP solve; a binding cap fails the build |
| `--workers` | int | 1 | Parallel processes for the per-pixel MAP solves (`--mode radial` only) |
| `--mode` | `radial`, `gp3d` | `radial` | Completion model: independent per-pixel 1-D, or 3-D angular-coupling low-rank field |
| `--gp3d-nz-solve` | int | 32 | (gp3d) Number of coarse redshift points for the solve grid |
| `--gp3d-pix-chunk` | int | 512 | (gp3d) Pixel chunk size for the all-pixel field evaluation |
| `--lss-corr-length-ang` | float | fiducial | (gp3d) Override the fixed angular (chordal) correlation length |
| `--lss-corr-length-mpc` | float | 50 | Override the fixed radial GP correlation length in Mpc; build-time only |
| `--lss-sigma` | float | 1.0 | Override the fixed GP field amplitude; build-time only |
| `--gp3d-nz-nodes` | int | 6 | (gp3d) Radial inducing nodes M_z; spacing in log1p(z) must not exceed ls_z |
| `--gp3d-nsph-nodes` | int | 32 | (gp3d) Fibonacci sphere inducing nodes M_sph |
| `--gp3d-z-node-hi` | float | `zgrid[-1]` | (gp3d) Top redshift of the radial inducing nodes |
| `--c-mode` | `per_pixel`, `aggregate`, `selection` | `per_pixel` | Completeness base the fit is residual to; stamped and checked at inference |
| `--depth-map` | path | none | Depth map supplying `f_p`, folded into the model completeness so Q carries clustering only |
| `--q-support-depth` | float | none | Truncate the radial fit to the catalog's redshift support; logQ is 0 above it |
| `--selection-fit` | path | none | `selection_fit.json`; required by and only legal with `--c-mode selection` |
| `--log10n0` | float | -2.0 | Override log10 of the expected comoving galaxy density the fit is conditioned on |
| `--delta` | float | 0.0 | Override the expected-density evolution exponent of `(1+z)^delta` |
| `--z-depth` | float | catalog attr | Override the catalog's `z_depth`, which truncates the completeness denominator |
| `--stratum-map` | path | none | Full-sky `stratum_map` HDF5 required by a multi-stratum `--selection-fit` |
| `--indexing` | `compact`, `global` | `global` | How the inference indexes the Q rows; `compact` is forced back to `global` |
| `--allow-unconverged` | flag | off | Ablation: save despite unconverged or non-finite solves (substituted with Q=1) |
| `--no-budget-renorm` | flag | off | Ablation: skip the per-z mean-one budget renormalization of Q |

## `darksirens_build_joint_lognormal_completion`

Builds K matched per-survey `Q_LSS` ensembles from one shared latent field, so a
K>=2 `--lss_marginalize` run uses one realization set. Module: `python -m
darksirens.cli.build_joint_lognormal_completion`.

```bash
darksirens_build_joint_lognormal_completion \
  --catalogs catalog_a.h5 catalog_b.h5 \
  --outs lss_a.h5 lss_b.h5
```

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--catalogs` | path(s) | required | K pixelated survey catalogs (`load_survey` schema) |
| `--outs` | path(s) | required | K output completion HDF5 paths, positionally aligned with `--catalogs` |
| `--n-members` | int | 32 | Matched Laplace ensemble members; 0 = MAP only |
| `--seed` | int | 1234 | Seed for the one shared Laplace member draw set |
| `--bias` | float(s) | 1.0 | Per-survey field bias `b_k` (1 shared value or K values) |
| `--log10n0` | float(s) | -2.0 | Per-survey log10 expected comoving density the fit is conditioned on |
| `--delta` | float(s) | 0.0 | Per-survey expected-density evolution exponent `(1+z)^delta` |
| `--gp3d-nz-solve` | int | 32 | Coarse redshift points for the shared solve grid |
| `--gp3d-pix-chunk` | int | 512 | Pixel chunk size for the per-survey all-pixel evaluation |
| `--lss-corr-length-ang` | float | fiducial | Override the shared angular (chordal) correlation length |
| `--lss-corr-length-mpc` | float | 50 | Override the shared radial GP correlation length in Mpc; build-time only |
| `--lss-sigma` | float | 1.0 | Override the shared GP field amplitude; build-time only |
| `--allow-unconverged` | flag | off | Ablation: save despite an unconverged or non-finite shared solve |
| `--no-budget-renorm` | flag | off | Ablation: skip the per-survey per-z mean-one budget renormalization |
| `--realization-set-id` | str | fresh uuid4 | Shared `realization_set_id` stamped on all K files |
| `--mode` | `gp3d`, `radial` | `gp3d` | Joint completion model; `radial` is rejected (no shared field to match) |

## `darksirens_diagnose_lognormal_completion`

Diagnoses an LSS lognormal completion file for one pixel, plotting `Q_LSS(p,z)`,
`dN_miss` and `p(z|p)`. Module: `python -m darksirens.cli.diagnose_lognormal_completion`.

```bash
darksirens_diagnose_lognormal_completion --catalog catalog_nside_32.h5 \
  --lss-completion lss_completion.h5 --pixel 1024
```

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--catalog` | path | required | Pixelated survey catalog HDF5 |
| `--lss-completion` | path | required | Completion file to diagnose |
| `--pixel` | int | required | Catalog row / global HEALPix pixel index |
| `--outdir` | path | `.` | Directory for the output plots |
| `--stratum-map` | path | none | Stratum map the table was built with; verified by sha256 |

## `darksirens_fit_selection`

Fits the parametric magnitude selection from a pixelated survey's magnitudes and
writes the selection JSON that the Q-table build and the inference read. Module
form: `python -m darksirens.cli.fit_selection`.

```bash
JAX_PLATFORMS=cpu darksirens_fit_selection \
  --survey_path catalog_nside_32.h5 \
  --m_lim 19.5
```

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--survey_path` | path | required | Pixelated survey HDF5 carrying `gal_app_mag` |
| `--m_lim` | float | required | The survey's hard apparent-magnitude limit (a truncation datum, not fitted) |
| `--out` | path | next to survey | Output JSON path (default `selection_fit.json`) |
| `--family` | `gaussian`, `schechter` | `gaussian` | Luminosity-function family: real-catalog Schechter LF or the mock truncated-normal model |
| `--m_faint_offset` | float | 5.0 | Schechter faint-end cutoff as an offset from M*; a protocol constant, never fitted |
| `--m_faint_cut` | float | none | Absolute-magnitude cut applied to the fit sample (schechter only) |
| `--k_corr_coeffs` | comma list | none | Polynomial coefficients of a fixed K-correction template applied to observed magnitudes |
| `--strata` | flag | off | Fit one theta per stratum using the survey's `gal_stratum` labels |
| `--m_lim_per_stratum` | `label:m_lim,...` | none | Per-stratum magnitude limits; only with `--strata` |

## Shared TinyNS and NUTS options

Both `darksirens_inference` and `darksirens_inference_lensing` accept the
options below. Unless noted, every `--tinyns_*` knob is unset by default and
resolved from `--tinyns_preset`. Options grouped in one row share a description.

| Option | Type / choices | Default | Meaning |
|---|---|---|---|
| `--tinyns_preset` | `recommended`, `heavy_darksirens`, `heavy_darksirens_strong`, `conservative`, `python_debug`, `prior`, `batched_gpu`, `adaptive_gpu`, `bounded_single`, `bounded_multi`, `fused_bounded_multi`, `custom` | `recommended` | TinyNS preset; explicit TinyNS options override preset defaults |
| `--tinyns_sample` | `rwalk`, `prior` | preset | TinyNS sampler |
| `--tinyns_kernel` | `jax`, `python` | preset | TinyNS proposal kernel |
| `--tinyns_vectorized`, `--tinyns_jax_vectorized` | BOOL | preset | Preset-resolved; no help text |
| `--tinyns_walks`, `--tinyns_step_scale`, `--tinyns_max_attempts`, `--tinyns_min_accepts` | int / float | preset | Preset-resolved; no help text |
| `--tinyns_batch_size`, `--tinyns_jax_block_size` | int | preset | Preset-resolved; no help text |
| `--tinyns_replacement_chains` | int | preset | Preset-resolved; no help text |
| `--tinyns_replacement_chain_schedule` | comma list | preset | Positive increasing schedule, e.g. `1,4,16,64` |
| `--tinyns_rwalk_proposal` | `isotropic` | preset | Preset-resolved; no help text |
| `--tinyns_bound` | `none`, `single`, `multi` | preset | Preset-resolved; no help text |
| `--tinyns_bound_enlargement`, `--tinyns_bound_update_interval`, `--tinyns_bound_jitter`, `--tinyns_bound_max_draws` | int / float | preset | Preset-resolved; no help text |
| `--tinyns_multi_bound_max_ellipsoids`, `--tinyns_multi_bound_min_points`, `--tinyns_multi_bound_split_threshold`, `--tinyns_multi_bound_enlargement`, `--tinyns_multi_bound_overlap_correction` | int / float / BOOL | preset | Preset-resolved; no help text |
| `--tinyns_rwalk_seed`, `--tinyns_rwalk_seed_fallback`, `--tinyns_bound_seed_kernel` | `live`/`bound`, BOOL, `python`/`jax` | preset | Preset-resolved; no help text |
| `--tinyns_allow_unused_bound`, `--tinyns_fused_bound_rwalk` | BOOL | preset | Preset-resolved; no help text |
| `--tinyns_bound_rebuild_on_failure`, `--tinyns_bound_failure_rebuild_threshold` | BOOL / int | preset | Preset-resolved; no help text |
| `--tinyns_checkpoint_path` | path | run dir | Explicit checkpoint file, overriding `<run_dir>/checkpoint.tinyns.npz` |
| `--tinyns_checkpoint_interval` | int | 100 | Checkpoint cadence in iterations; `--checkpoint_interval` only enables or disables it |
| `--tinyns_resume_from` | path | none | Explicit TinyNS checkpoint to resume; overrides `--resume` |
| `--tinyns_checkpoint_path_out`, `--tinyns_progress_interval` | path / int | preset | Preset-resolved; no help text |
| `--nuts_warmup` | int | 500 | NumPyro `MCMC(num_warmup=...)` |
| `--nuts_samples` | int | 1000 (2000 for lensing) | NumPyro `MCMC(num_samples=...)` |
| `--nuts_chains` | int | 1 (4 for lensing) | NumPyro `MCMC(num_chains=...)` |
| `--nuts_target_accept` | float | 0.8 | NUTS `target_accept_prob` |
| `--nuts_max_tree_depth` | int | 10 | NUTS `max_tree_depth` |
| `--nuts_chain_method` | `sequential`, `parallel`, `vectorized` | `sequential` | NumPyro `MCMC(chain_method=...)` |
| `--nuts_init_tries` | int | 32 | Random draws tried for a finite initial point before NUTS starts |
| `--nuts_init_seed_offset` | int | 100000 | Offset added to `--seed` for those initial-point draws (`darksirens_inference` only) |
