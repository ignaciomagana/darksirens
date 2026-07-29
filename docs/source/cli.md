# Command-line interface

Installing the package exposes eight console scripts.

| Command | Stage | What it does |
|---------|-------|--------------|
| [`darksirens_pixelate`](#darksirens_pixelate) | prepare | raw galaxy-survey HDF5 → dense HEALPix catalog |
| [`darksirens_skymaps_to_samples`](#darksirens_skymaps_to_samples) | prepare | 3-D LVK skymap FITS → GW importance-sample `gwdata.h5` |
| [`darksirens_build_lognormal_completion`](#darksirens_build_lognormal_completion) | prepare | offline LSS-conditioned completion field `Q_LSS(p, z)` |
| [`darksirens_build_joint_lognormal_completion`](#multitracer-catalog-mixtures-k--2) | prepare | K matched `Q_LSS` ensembles from **one** shared latent field |
| [`darksirens_diagnose_lognormal_completion`](#darksirens_diagnose_lognormal_completion) | prepare | per-pixel diagnostic plot of a completion file |
| [`darksirens_inference`](#darksirens_inference) | run | spectral / bright / dark-siren hierarchical inference |
| [`darksirens_inference_lensing`](#darksirens_inference_lensing) | run | weak-lensing and J=2 strong-lensing cluster inference |
| [`darksirens_analyze`](#darksirens_analyze) | analyze | posterior-predictive summaries and plots from run directories |

Each is also runnable as a module — `python -m darksirens.cli.<name>` — which is
the form the repository's own driver scripts use.

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
- `--z_depth`: optional survey redshift depth — the redshift beyond which this
  survey catalogs no galaxies. It is written as `f.attrs['z_depth']` and read by
  `darksirens_inference` as a **completeness prior**: completeness is zero beyond
  the depth, so every modeled host there is uncatalogued (missing), and the
  source prior above the depth is the plain volumetric × population shape, NOT
  truncated to zero. Omit it for the legacy behaviour (no attribute written;
  completeness estimated over the full `[0, DARKSIRENS_ZMAX]` grid).

## `darksirens_skymaps_to_samples`

Convert a directory of 3-D LVK skymap FITS files into the GW importance-sample
HDF5 (`gwdata.h5`) that `--gw_path` expects. This is the low-latency route: a
public 3-D skymap carries the sky × luminosity-distance posterior but no mass or
spin samples, so each event's `--nsamp` rows pair distance draws from the skymap's
distance ansatz with **deliberately uninformative surrogate** mass/spin draws. The
resulting file is flagged `mock_data=True`, and the surrogate proposal is divided
out and reweighted to the sampled population downstream — so it constrains
cosmology through distance and sky position, not through masses.

```bash
darksirens_skymaps_to_samples \
  --skymap_dir skymaps/ \
  --output data/gwdata_skymaps.h5 \
  --nsamp 5000 --zmax 1.5 --seed 21
```

**The population must be held fixed.** Because the mass/spin coordinates are
surrogates rather than posterior samples, fitting a population to them measures
the surrogate proposal, and the mass/spin factors stop cancelling against the
injection-based selection integral. The output therefore carries
`requires_fixed_population = True`, and `darksirens_inference` refuses to start
on it with `--fix_population false`:

```bash
darksirens_inference --gw_path data/gwdata_skymaps.h5 ... --fix_population true
```

Make sure the injection set's mass/spin draws are reweighted to that same fixed
model. `--allow_skymap_population` downgrades the refusal to a loud warning for
deliberate methodological studies; the population posterior of such a run is not
a measurement.

Options:

- `--skymap_dir`, `--output`: required input directory and output `gwdata.h5`.
- `--pattern`: glob for the skymap files inside `--skymap_dir` (default `*.fits*`).
- `--nsamp`: samples per event (default `5000`). Larger values reduce the
  surrogate-mass Monte-Carlo noise.
- `--zmax`: maximum redshift of the analysis (default `1.5`). It sets the default
  `--m1det_max`, which is what keeps the mass marginalization unbiased across the
  whole `H0` prior — raise it with the analysis, not after the fact.
- `--pop_m_min`, `--pop_m_max`: edges of the population mass prior the surrogate
  must cover (defaults `2` and `100`). `--m1det_min` / `--m1det_max` override the
  derived detector-frame bounds (`pop_m_min` and `pop_m_max * (1 + zmax)`).
- `--q_min`, `--chi_abs_max`: surrogate mass-ratio and spin support (defaults
  `0.05`, `0.99`).
- `--n_dist_grid`: grid resolution of the distance-ansatz inverse CDF (default
  `1024`).
- `--pe_H0`, `--pe_Om0`: fiducial cosmology used **only** to invert detector-frame
  `dL` samples into the `m1src`/`m2src` datasets the loader requires (Planck15 by
  default). It does not enter `p_pe` and does not bias `H0`.
- `--seed`: random seed for the surrogate draws.

## `darksirens_build_lognormal_completion`

Offline preprocessor for the LSS-conditioned lognormal completion field
`Q_LSS(p, z)`: a Poisson-lognormal fit of the missing-galaxy density against the
observed counts of a pixelated catalog, written once and then passed to inference
with `--lss_completion`. It replaces the legacy `1 + b_eff*delta_g` overdensity
factor with a field that has a posterior, so `--n-members > 0` produces a Laplace
ensemble that `--lss_marginalize` can integrate over.

```bash
# deterministic (MAP) table, plus a 32-member ensemble for --lss_marginalize
darksirens_build_lognormal_completion \
  --catalog data/catalog_pixelated_nside_16.h5 \
  --out data/q_radial.h5 --mode radial --n-members 0 --log10n0 -2.4

darksirens_build_lognormal_completion \
  --catalog data/catalog_pixelated_nside_16.h5 \
  --out data/q_radial_ens.h5 --mode radial --n-members 32
```

Options:

- `--catalog`, `--out`: required pixelated survey catalog (the `load_survey`
  schema `darksirens_pixelate` writes) and output completion HDF5.
- `--mode {radial,gp3d}`: `radial` (default) fits each pixel independently as a
  1-D Poisson-lognormal; `gp3d` fits a 3-D angular-coupling low-rank field, so
  **empty pixels borrow from their neighbours**. Only `gp3d` carries a shared
  latent field, which is why the joint multi-survey builder below is `gp3d`-only.
- `--n-members`: size of the Laplace/FFT-diagonal ensemble (default `32`; `0`
  writes the MAP field alone). Anything meant for `--lss_marginalize` needs
  `> 0`.
- `--log10n0`, `--delta`: the expected comoving galaxy density
  `n0 * (1+z)^delta` [Mpc^-3] the fit is conditioned on (defaults `-2.0`, `0.0`).
  **Calibrate `--log10n0` to the catalog** (`N / (f_sky * V_c)`): a mis-set `n0`
  is absorbed into `Q` as spurious redshift structure that biases `H0`. The
  conditioning values are stamped into the output and enforced at run time — see
  `--lss_completion` under [inference](#catalog-options).
- `--indexing {compact,global}`: how inference indexes the `Q` rows. Both build
  modes emit a full global-HEALPix-pixel table, so `global` is the only stamp
  the builder writes; a requested `compact` is forced back to `global` with a
  warning (a mis-stamp would misindex `Q` rows at inference).
- `--seed`, `--prior-strength`, `--maxiter`: ensemble seed (default `1234`),
  prior strength (`1.0`), and solver iteration cap (default `200000`; converged
  solves self-terminate long before this, and an unconverged build **fails**
  instead of writing a silently under-relaxed table).
- `--allow-unconverged`: save the completion even when solves are unconverged
  or produced non-finite cells (substituted with `Q = 1`). The override is
  stamped in the file's diagnostics and warned about at load time — research
  ablations only, never production.
- `--gp3d-nz-solve`, `--gp3d-pix-chunk`, `--lss-corr-length-ang`: `gp3d`-only
  solve-grid size (`32`), pixel chunk for the all-pixel evaluation (`512`), and
  an override of the angular correlation length (default: the `SurveyParams`
  fiducial).

## `darksirens_diagnose_lognormal_completion`

Inspect one pixel of a completion file before trusting it in a run. Writes
`<outdir>/lss_completion_pixel<PIXEL>.pdf`, a three-panel figure comparing the
homogeneous reference (`Q = 1`), the MAP field, and — when the file carries an
ensemble — the member 16–84% band, for: the completion factor `Q_LSS(p, z)`, the
missing-galaxy density `dN_miss/dz`, and the resulting dark-siren redshift prior
`p(z | p)`. A pixel whose `Q` band brackets 1 across the grid is telling you the
LSS conditioning bought nothing there; a `Q` running away at high `z` is the
classic mis-set `--log10n0` signature.

```bash
darksirens_diagnose_lognormal_completion \
  --catalog data/catalog_pixelated_nside_16.h5 \
  --lss-completion data/q_radial_ens.h5 \
  --pixel 1234 --outdir figs
```

Options:

- `--catalog`, `--lss-completion`: required catalog and completion file.
- `--pixel`: required catalog row / global HEALPix pixel index to diagnose.
- `--outdir`: output directory (default: the current directory).

## `darksirens_inference`

Run hierarchical inference.

```bash
darksirens_inference \
  --gw_path GW.h5 \
  --gwselection_path INJECTIONS.h5 \
  --sampler tinyns \
  [options]
```

### Option conventions

- **Boolean flags** take an explicit value — `true`/`t`/`1`/`yes`/`y` or `false`/`f`/`0`/`no`/`n`, case-insensitive — and are shown as `BOOL` in `--help`. An unrecognized value is a hard parse error (exit code 2), not a silent `false`. A few flags (for example `--allow_unverified_shared_lss_members`) also accept a bare form, where `--flag` means `--flag true`.
- **Deprecated spellings.** `--fix_cosmology`, `--fix_de`, and `--use_lss` are the canonical spellings. The older `--fixed_cosmology`, `--fixed_de`, and `--use_LSS` still work as aliases but print a one-line deprecation notice. The persisted settings keys are unchanged, so existing `settings.json` files and post-processing continue to load.

### Data options

- `--gw_path`: required GW posterior-sample HDF5 file.
- `--gwselection_path`: required gwcat selection HDF5 file.
- `--survey_path`: one or more pixelated survey HDF5 files; required for dark-siren models. A single path is the classic single-catalog analysis; `K >= 2` paths run the K-catalog **multitracer mixture** (see [Multitracer catalog mixtures](#multitracer-catalog-mixtures-k--2)), sampling the per-catalog host fractions.
- `--counterpart RA1 DEC1 Z1 [RA2 DEC2 Z2 ...]`: bright-siren counterpart coordinates and redshifts, one triplet per GW event in event order; RA and Dec use radians, matching the GW sample convention. Required for `--universe_model bright_sirens`.
- `--counterpart_dz`: Gaussian redshift uncertainty assigned to the synthetic counterpart catalog entry; defaults to `1e-4`.
- `--counterpart_nside`: HEALPix NSIDE for the synthetic counterpart catalog; defaults to `1`.
- `--save_path`: directory for settings, samples, plots, and summaries.

### Physical-model options

- `--universe_model`: one of `spectral_sirens`, `bright_sirens`, `dark_sirens`, or `dark_sirens_complete`. The weak-lensing model `spectral_sirens_wl` moved to the [lensing CLI](reference/lensing.md) — run `darksirens_inference_lensing --cluster_mode off --wl_backend lognormal|tabulated ...`.
- `--pop_model`: population model name. Parametric mixture names are parsed as a composition grammar: `+`-separated mass tokens (`powerlaw`, `brokenpowerlaw`, `peak`) with optional digit count prefixes, e.g. `powerlaw+peak`, `brokenpowerlaw+2peaks`, `2powerlaws+3peaks`. Any grammar composition works with blueprint-default priors; curated names additionally carry physics-tuned priors and fiducials. Bespoke names such as `gp_mass`, `gp_mass_pairing`, `gp_mass_pairing_joint`, `golomb_1g`, `golomb_1g+tail`, and `gwtc5_fiducial_bpl2peaks` are registered explicitly. See [Concepts → Population models](concepts.md#population-models).
- `--shared_beta`: whether to use one shared beta/pairing distribution (`true`, default) or per-component beta parameters (`false`).
- `--shared_spin`: whether to use one shared spin distribution (`true`, default) or per-component spin parameters (`false`).
- `--shared_gamma`: whether to use one shared redshift-evolution gamma (`true`, default) or per-component gamma parameters (`false`).
- `--fix_population`: fix all population parameters to fiducial values. Required
  when `--gw_path` is a `darksirens_skymaps_to_samples` product (it declares
  `requires_fixed_population`); see
  [`darksirens_skymaps_to_samples`](#darksirens_skymaps_to_samples).
- `--allow_skymap_population`: downgrade that requirement from a startup refusal
  to a loud warning. Methodological studies only — the population posterior of
  such a run measures the surrogate proposal, not the astrophysical population.
- `--fix_cosmology`: fix all cosmological parameters (`H0`, `Om0`, `w0`, `wa`) to fiducial values.
- `--fix_de`: fix only the CPL dark-energy parameters (`w0=-1`, `wa=0`) while leaving `H0` and `Om0` available unless fixed separately.
- `--fix_survey`: fix survey-completion parameters to fiducial values.
- `--prior_overrides`: JSON object mapping parameter labels to `[lower, upper]` prior bounds, e.g. `{"H0": [60, 80], "w0": [-1.2, -0.8], "wa": [-0.5, 0.5]}`. Population parameters use the printed LaTeX labels from the startup parameter table, e.g. `{"$\\alpha_{\\rm PL}$": [0.0, 4.0]}`. Multi-component mass labels are tagged by slot (`PL`, `BPL`, `G1`, `G2`, ...).
- `--fixed_parameter_values`: JSON object mapping parameter labels to fixed scalar values, e.g. `{"Om0": 0.3075, "w0": -1.0, "wa": 0.0}` or `{"$\\mu_{\\rm G}$": 35.0}`. Mixture-weight labels are stick-breaking inputs (`$v_1$`, `$v_2$`, ...), not final component fractions.
- `--bright_siren_sky_marginalized BOOL`: for `bright_sirens`, ignore the counterpart sky-pixel gate and apply only the counterpart redshift prior. Defaults to `False`. Accepted true values are `true`, `t`, `1`, `yes`, and `y`; accepted false values are `false`, `f`, `0`, `no`, and `n` (case-insensitive).
- `--complete_empty_pixel_policy {zero,volume}`: controls genuinely empty pixels for `dark_sirens_complete` and `bright_sirens`. `zero` is the formal default and returns zero probability (`-inf` log-prior) when `ngals == 0`; `volume` uses the comoving-volume prior as a robustness approximation for sparse pixelations.

### Catalog options

- `--use_lss`: include the large-scale-structure overdensity `delta_g(pix, z)` in the missing-galaxy budget (`max(1 + b_eff*delta_g, 0)`). With a K-catalog mixture, each catalog computes its own overdensity field from its own rows, coupled to its own sampled `b_miss` (`b_miss_c{k}` for catalogs 2..K).
- `--catalog_sky_weighting {conditional,field}`: the dark-siren redshift-prior normalization convention. `field` (**default, all K**) is the JOINT catalog host-density estimand: it normalizes by the survey-global `Z(theta) = Sum_all-pixels[N_obs+N_miss]`, so RELATIVE angular host density is preserved (a pixel with 100 candidate hosts carries ~100x the angular weight of a pixel with one), and for a K>=2 mixture the weight `fcat_k` measures the catalog's GW **host fraction** (number-density / sky-clustering contrast). `conditional` is the radial-only **legacy** estimand: it normalizes each pixel by its own `Z[pix]` so every pixel integrates to unit mass and relative angular host density is discarded (bit-identical to the pre-existing single-catalog behaviour; kept for reproducing older runs). **Unset auto-resolves to `field` at every K.** At K=1 field's `log10n0` (number-density) channel cancels between the PE and selection terms, so `log10n0` is only weakly identified there and marginalizes against its prior — what field restores at K=1 is the relative angular host weighting `conditional` discards. For `dark_sirens` the degenerate explicit `conditional` at K>=2 (the railing z-shape-only `fcat` estimand) stays fatal. The field normalizer carries the SAME modulated missing budget as the numerator (Q_LSS, `delta_g`, marked-host `mu_miss`, per ensemble member under `--lss_marginalize`) AND the same depth convention (with a `z_depth` its observed term is the depth-scaled count `Sum_pix N_obs*m_pix`, so a galaxy catalogued beyond the depth is counted once, as missing), so it composes with all catalog options below; it needs the full-sky rows (incompatible with `--drop_full_catalog`).
- `--lss_completion PATH [PATH ...]`: precomputed LSS-conditioned lognormal completion table(s) `Q_LSS` replacing the legacy overdensity factor. With K catalogs pass 0 or exactly K paths, positionally aligned with `--survey_path` (`""` = no external completion for that catalog); one table is never broadcast across catalogs. **Provenance is enforced.** `Q_LSS` is a *fit* conditioned on the build-time `n0`, `delta`, `b_miss` and `Om0/w0/wa`, and a mismatch is absorbed into the completion field as spurious redshift structure that biases `H0`. The run therefore **aborts** if any of those is sampled, or is fixed to a value differing from the table's stamped fiducial — pin them with `--fixed_parameter_values`, or rebuild the table at the values you intend to use. Note that `dark_sirens` samples `log10n0` and `delta` by default, so an existing config will need one of those two changes. `H0` is exempt (it is the measurand, and enters `Q`'s dimensionless density ratio only at second order); tables built before the fiducials were stamped cannot be verified and are rejected.
- `--lss_marginalize`: fully-Bayesian marginalization over the Q_LSS ensemble, `logL = logsumexp_m logL(Q_m) - log M`. With a K-catalog mixture the marginalization uses **one shared member index** — member `m` of every catalog must sample the same LSS realization. Build the per-catalog ensembles jointly with `darksirens_build_joint_lognormal_completion` (see the joint-builder note below) so they share one `realization_set_id` and equal `--n-members`, or the run aborts (unless `--allow_unverified_shared_lss_members` accepts the independent-fields approximation).
- `--mark_model {none,loglinear}` and `--marks LIST`: the marked-host model (host efficiency `h(m|eta)` over z-centred galaxy marks). With a K-catalog mixture the marks are resolved **per catalog** from each survey file's datasets (`available ∩ --marks`): catalog 1 samples the unsuffixed `eta_<mark>` coefficients, catalogs 2..K sample `eta_<mark>_c{k}` blocks, and a catalog with no selected marks runs the plain galaxy-count host model (`h = 1`) inside the mixture.
- `--validate_completion`: run a dry-run completion clipping diagnostic, save `completion_validation__*.json` under `--save_path`, and exit before likelihood construction or sampling. The diagnostic uses the same matched-kernel completeness ratio as the likelihood and reports clipping fractions for the raw ratio, LSS modulation, and effective completeness. With K catalogs one diagnostic runs per catalog (`completion_validation_c{k}__*.json`).
- `--completion_validation_pixels`: maximum number of unique catalog pixels to inspect during `--validate_completion`; defaults to `64`.

### Multitracer catalog mixtures (K >= 2)

Passing `K >= 2` files to `--survey_path` runs the K-catalog mixture: the catalog-completed redshift prior becomes `log p_mix(z) = logsumexp_k [log w_k + log p_k(z, pix_k)]`, with per-catalog nside/pixelization, per-catalog survey nuisance blocks (`log10n0_c{k}`, `delta_c{k}`, `b_miss_c{k}`, `sigma_kde_c{k}`), and sampled stick-breaking weights `fcat_2..fcat_K` (Beta(1, K-m+1) priors; `w_2 = fcat_2` exactly at K=2).

- **Estimand.** Under the (auto-resolved) `field` sky weighting, `w_k` is the fraction of GW hosts drawn from catalog `k`'s tracer population — e.g. the AGN host fraction `f_agn` for a GAL+AGN K=2 run. `darksirens_analyze` derives and plots `w_1..w_K` from the sampled sticks automatically (`catalog_weights_<tag>.{pdf,npy}`).
- **Universe models.** `dark_sirens` (the general case) or `dark_sirens_complete` (special case: field weighting only). `spectral_sirens` and `bright_sirens` are inherently single-catalog.
- **Composability.** Per-catalog Q_LSS tables, `--use_lss` overdensities, `--lss_marginalize` ensembles (shared member index), marked-host models (per-catalog eta blocks), and anisotropic `--sky_model` choices (one population-level `g(n, z)` factor shared across catalogs) all compose with the mixture; the survey-global field normalizer carries the same modulated missing budget as each catalog's numerator.
- **Weak lensing** (the `spectral_sirens_wl` model, now driven by `darksirens_inference_lensing`) and `--counterpart`/`bright_sirens` remain single-catalog.

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

### Checkpoint and resume

A production nested-sampling run is a multi-day job, and a SLURM `TIMEOUT`,
preemption or OOM used to discard all of it. Checkpointing is therefore **on by
default** and both nested samplers honour it; the flags are identical on
`darksirens_inference` and `darksirens_inference_lensing`.

- `--checkpoint_interval SECONDS|off`: seconds between sampler checkpoints
  written **into the run directory** (default `1800`; `off`/`0` disables).
  `dynesty` honours the wall-clock cadence exactly. `tinyns` checkpoints every
  `--tinyns_checkpoint_interval` **iterations** (default `100`) and this flag
  only enables or disables it.
- `--resume auto|PATH|off`: restore a checkpointed run and continue it. `auto`
  picks the most recently modified checkpoint under `--save_path` for this
  sampler **and this configuration** (the model/sampler/seed run-directory
  prefix), skips run directories that already hold a `results.hdf5`, and
  continues inside that run directory. With nothing to resume it starts fresh
  silently, so it is safe to hard-code in a submit script. `PATH` may be a run
  directory or a checkpoint file and must exist. `off` (the default) always
  starts fresh.
- The resumed run continues the `--seed`-derived RNG stream carried in the
  checkpoint rather than drawing fresh entropy, so a resumed chain is the chain
  the uninterrupted run would have produced.
- **Resume is fingerprint-gated.** Every run writes a `run_fingerprint.json`
  recording its full semantic configuration: sampled labels, bounds and prior
  families, fixed/overridden values, every model flag, sampler settings, the
  seed, the normalization grids, and the **content** (SHA-256; sampled digest
  above 1 GiB) of every input file. A resume refuses to restore sampler state
  unless the fingerprint matches exactly — restoring under a different target
  would silently mix samples and evidence from two statistical models.
  Operational knobs (`--save_path`, checkpoint cadence, progress/diagnostic
  flags, performance chunking) are excluded, so requeueing the identical
  command always matches. A code change (new `git` commit) with an unchanged
  configuration warns but proceeds, so a requeue can straddle a deployment.
- `--resume_force`: **unsafe** override of the fingerprint gate, for
  deliberate expert use and for continuing checkpoints created before
  fingerprinting existed (which have no `run_fingerprint.json`). A forced
  mismatch is warned about loudly; the output mixes two targets and is not a
  science result.
- A resume attempt never overwrites the original `settings.json`; each attempt
  writes its own timestamped `settings.resume-<timestamp>.json` beside it.
- Explicit `--tinyns_checkpoint_path` / `--tinyns_resume_from` still win over
  these flags, so existing tinyns scripts behave exactly as before.

Requeue the identical command to continue a killed run:

```bash
darksirens_inference ... --sampler dynesty --nlive 2000 \
  --checkpoint_interval 1800 --resume auto --save_path runs/full/A2-dark-joint
```

### Performance options

- `--sel_batch_size`: optional injection-selection batch size.
- `--norm_nmass`, `--norm_nq`, `--norm_nchi`: mass, mass-ratio, and spin grid sizes used for GW-population normalization quadrature. They default to `500`, `200`, and `200`, respectively, and can also be set with `DARKSIRENS_GW_N_MASS`, `DARKSIRENS_GW_N_Q`, and `DARKSIRENS_GW_N_CHI`. The inference command prints the active values and saves them in `settings.json` under `normalization_grid`.
- `--kde_window W`: static window size for the per-sample catalog KDE — only the `W` galaxies nearest each sample's redshift are evaluated (catalog rows are z-sorted at load). The default `W = 1024` holds `|delta log p_cat| < 1e-6` against the full row across the ENTIRE sampled `sigma_kde` prior `[0, 0.05]` on a review-scale 2113-galaxy row. **Denser rows need a larger `W`** — size it with `darksirens.redshift.catalog.recommended_kde_window`, which returns the largest galaxy count any window-width interval contains at the widest kernel the prior admits. `0` disables windowing entirely and evaluates the full row, the escape hatch for A/B validation.
- `--kde_window_nsigma X`: half-width multiplier for that window — contributing galaxies lie within `± X * max_row(sigma_eff)` of the sample (default `8`). The half-width is traced, because `sigma_kde` is sampled; only the window SIZE `W` is static.

## `darksirens_inference_lensing`

Run the lensing likelihood: spectral sirens with weak-lensing magnification
marginalization, optionally plus the **J=2 strong-lensing cluster** channel in
which candidate pairs of multiply-imaged sirens are evaluated with the SIS pair
likelihood against a lensed-injection cluster selection term. This CLI is the
sole owner of the `spectral_sirens_wl` universe model, which moved here out of
`darksirens_inference`. It deliberately does **not** implement galaxy-catalog
dark sirens, LSS completion or catalog host probabilities; use
`darksirens_inference` for those.

The measurement it exists to produce is an evidence difference — the same data
run with the pair channel on and off — so the two arms are run as a pair and
`logZ` / `logZerr` are archived in `results.hdf5` and `settings.json`:

```bash
# J=2 arm: candidate pairs marginalized over partitions
darksirens_inference_lensing \
  --gw_path mock_observed_gw_pe.h5 \
  --gwselection_path mock_gw_selection.h5 \
  --observed_catalog_path observed_catalog.json \
  --lensed_injections_path mock_lensed_injections.h5 \
  --pair_metadata_path mock_pair_metadata.h5 \
  --candidate_pairs_path candidate_pairs.json \
  --cluster_mode j2 --partition_mode marginalize_exact \
  --partition_component_mode componentwise --max_exact_partitions 10000 \
  --wl_backend lognormal --pop_model powerlaw+peak \
  --fix_cosmology true --fix_survey true --fix_population true \
  --fix_lens_rate false --fixed_parameter_values '{"tau_n": 3.0}' \
  --lens_prior_overrides '{"log10_tau_A": [-5.0, -2.5]}' \
  --sampler tinyns --nlive 2000 --dlogz 0.1 --max_samples 0 \
  --pe_max_per_pair 400 --checkpoint_interval 1800 --resume auto \
  --seed 4001 --save_path runs/lensing/seed4001_j2

# control arm: identical data and population treatment, pair channel off
darksirens_inference_lensing \
  --gw_path mock_observed_gw_pe.h5 \
  --gwselection_path mock_gw_selection.h5 \
  --observed_catalog_path observed_catalog.json \
  --cluster_mode off --wl_backend lognormal --pop_model powerlaw+peak \
  --fix_cosmology true --fix_survey true --fix_population true \
  --fix_lens_rate true \
  --sampler tinyns --nlive 2000 --dlogz 0.1 --max_samples 0 \
  --checkpoint_interval 1800 --resume auto \
  --seed 4001 --save_path runs/lensing/seed4001_off
```

### Data options

- `--gw_path`, `--gwselection_path`: required GW posterior samples and gwcat
  selection file.
- `--observed_catalog_path`: the unified `observed_catalog.json` — the current
  observed-mode entry point. `--pair_metadata_path` adds the pair/candidate-edge
  metadata. (`--pair_pe_path` is the DEPRECATED split-pair layout; preflight's
  event-index range check has never accepted it.)
- `--lensed_injections_path`: lensed injections, required by `--cluster_mode j2`
  and by `--singleton_lensing sl_mixture`.
- `--candidate_pairs_path`, `--partition_path`: the candidate-edge graph used by
  `--partition_mode marginalize_exact`, and the fixed image-assignment partition
  used by `--partition_mode fixed`.
- `--partition_mode {fixed,marginalize_exact}`: `fixed` (default) trusts one
  supplied image assignment; `marginalize_exact` sums the likelihood over
  partitions of the candidate graph, i.e. over which events really are images of
  each other. `--partition_component_mode {global,componentwise}` (default
  `componentwise`) factorizes that sum over disconnected components, which is
  what makes it tractable; `--max_exact_partitions` (default `10000`),
  `--max_component_events`, `--max_component_edges`,
  `--max_component_partitions` and `--max_total_partitions` bound the
  enumeration.

### Model options

- `--cluster_mode {off,j2}`: the pair channel (default `j2`). `off` is the
  singleton-only control arm of the evidence comparison.
- `--wl_backend {lognormal,tabulated,disabled}`: weak-lensing magnification
  marginalization (default `lognormal`, with `--lensing_wl_a` / `--lensing_wl_b`
  setting the `sigma_mu(z)` scale and slope, defaults `4e-3` and `1.5`).
  `tabulated` reads `log p_WL(mu|z)` from `--lensing_wl_table_path`.
- `--wl_selection {auto,standard,wl_lognormal}`: singleton **selection**
  treatment. `auto` (the default) matches the event-side `--wl_backend`
  (`lognormal` → `wl_lognormal`, `disabled` → `standard`) so the hierarchical
  numerator and denominator share one observation model; an explicit
  mismatched pair is fatal unless `--allow_mismatched_wl_selection` records a
  deliberate ablation. `wl_lognormal` applies the same lognormal/Hermite WL
  marginalization to singleton injections; `--lensing_wl_a 0` reduces it to
  `standard`. `tabulated` has no matched selection integral and is refused by
  `auto`. The pre-resolution request and the resolved value are both recorded
  in `settings.json`.
- `--pop_model`: population model (default `powerlaw+peak`; the same grammar as
  the main CLI).
- `--sl_tau_A`, `--sl_tau_n`, `--sl_T0_sec`: SIS optical-depth amplitude and
  exponent (defaults `5e-4`, `3.0`) and the time-delay scale `T0` in seconds
  (default `5.36e6`, ~62 d — the SIS scale at `z_L=0.5`, `z_s=1`,
  `sigma_v=200 km/s` under this repo's cosmology). Candidate pairs with
  `|dt| >= T0` fall outside the SIS support `y in (0,1)` and get an exactly
  `-inf` time-marked pair likelihood.
- `--fix_lens_rate BOOL`: `true` (default) pins the SIS optical depth to
  `--sl_tau_A`/`--sl_tau_n`; `false` samples the lensing hyperparameters, with
  `--lens_prior_overrides` supplying their bounds, e.g.
  `'{"log10_tau_A": [-5.0, -2.5]}'`.
- `--pair_marks {none,time}` and `--pair_time_sigma_sec`: the optional
  time-delay mark on candidate pairs.
  `--pair_time_mark_impl {auto,quadrature,delta}` chooses the `y`-integral
  implementation — `auto` delta-collapses when `max(sigma_dt)/T0 < 0.02`, since
  marks that sharp are unresolvable by the quadrature.
  `--allow_suspicious_time_marks true` downgrades the placeholder/synthetic
  time-mark hard error to a warning.
- `--pair_tag_model {constant,snr_only,snr_sky,snr_time,snr_time_sky,file}`
  with `--pair_tag_constant` / `--pair_tag_perturb_logit` /
  `--pair_tag_selection_path`: the model for the probability that a true image
  pair is tagged as a candidate.
- `--edge_mark_prior_keys`, `--edge_mark_likelihood_keys`: comma-separated
  candidate-edge marks folded into the edge log prior odds, and into the edge
  likelihood, under exact marginalization.
- `--singleton_lensing {off,sl_mixture}`: `off` (default) keeps the legacy
  single-image protocol. `sl_mixture` models observed singletons as a mixture of
  unlensed sources and strongly lensed sources with exactly one detected image
  (evidence mixture + exactly-one-detected selection subset + analytic
  Finn–Chernoff partner censoring); it needs `--lensed_injections_path` and a
  mock built with lensed singletons. `--y_nodes_single` (default `32`) sets its
  Gauss–Legendre `y` nodes, and `--fc_rho_thr` / `--fc_r0` / `--fc_mc_bar`
  override the injection file's Finn–Chernoff attrs.
- `--fix_cosmology` / `--fix_survey` (both default `true`) and
  `--fix_population` (default `false`), plus `--fixed_parameter_values` and
  `--prior_overrides` as JSON — same conventions as the main CLI.

### Sampler, checkpoint, and performance options

`--sampler` (required; `tinyns`, `dynesty` or `numpyro`), `--nlive` (default
`2000`), `--dlogz` (default `0.1`), `--max_samples`, the whole `--tinyns_*` and
`--nuts_*` families, `--selection_neff_guard`, `--redshift_prior_barrier`,
`--sel_batch_size`, `--norm_n*`, `--pairing_norm_grid`, `--seed`,
`--show_progress` and `--save_path` behave as they do on
[`darksirens_inference`](#sampler-options), including
[`--checkpoint_interval` and `--resume`](#checkpoint-and-resume). What is
different or lensing-only:

- `--pe_max_per_pair`: PE samples kept per pair image (default `400`; `0` keeps
  all). This is the control on the `O(N_pe^2 N_y)` pair-KDE memory, and the
  first thing to reduce when a J=2 run runs out of host memory.
- `--pair_batch_size`: candidate-pair batch size for the J=2 likelihood scan
  (`0`, the default, keeps the unbatched path).
- `--y_nodes_pair`: Gauss–Legendre `y` nodes per J=2 pair likelihood (default
  `32`).
- `--max_likelihood_variance`: same flag and same default (`1.0`, the
  GWTC-4.0/5.0 criterion), but on this cluster/lensing stack the cap currently
  bounds the SELECTION component only (`N_obs^2/Neff_sel`) — the
  per-event/per-pair variance term of the full `sigma^2_lnL` criterion is not
  yet threaded here, whereas `darksirens_inference` enforces the full total.
  The Vitale `Neff > 5*N_obs` mean floor applies on both.
- `--preflight_only true`: run the lensing input preflight checks, write the
  JSON (to `--preflight_json`, else `<save_path>/preflight.json`) and exit before
  compilation or sampling. Cheap, and worth running before every long submission.

See [Lensing: weak magnification & strong-lensing clusters](reference/lensing.md)
for the physics, the mock generator, the file contract, and the validation
drivers.

## `darksirens_analyze`

Analyze saved inference products and compute posterior-predictive summaries. The analyzer reads the current `results.hdf5` output format (including root-level or grouped posterior samples) and falls back to the numeric `samples.npy` crash-recovery chain (metadata from `settings.json`, no evidence).

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
- `--allow_legacy_pickle`: permit reading a very old pickled-dict `samples.npy`. This deserializes arbitrary Python from the file; only use it on files you trust.


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
  --fix_cosmology true \
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
  --extra-arg --fix_cosmology --extra-arg true \
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
