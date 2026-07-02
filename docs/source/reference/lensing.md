# Lensing: weak magnification & strong-lensing clusters

This page documents the **spectral-siren lensing** workflow.  The current
lensing branch does two things:

* `spectral_sirens_wl`: spectral-siren singleton inference with optional
  weak-lensing magnification marginalisation (`--wl_backend lognormal`).
* `spectral_sirens` / `spectral_sirens_wl` plus **J=2 strong-lensing clusters**:
  candidate pairs of multiply-imaged sirens are evaluated with the SIS pair
  likelihood and the lensed-injection cluster selection term
  (`--cluster_mode j2`).

It deliberately does **not** implement galaxy-catalog dark sirens, LSS
completion, catalog host probabilities, or dark-siren/LSS lensing inference.
Those features remain out of scope for the lensing CLI; use the ordinary
`darksirens_inference` and LSS tools for non-lensing catalog workflows.

The physics is summarised on the [Theory & methods](../theory.md) page.

## End-to-end spectral-siren lensing workflow

### 1. Generate a mock

`scripts/mock_lensing/generate_mock_lensing.py` writes a standalone spectral-siren
lensing mock with current `gwcat-1.0`, `gwcat-selection-1.0`, lensed-injection,
pair-PE, fixed-partition, candidate-pair, truth, and manifest files.

```bash
python scripts/mock_lensing/generate_mock_lensing.py \
  --outdir data/mock_lensing_fixed \
  --conditioning fixed_counts \
  --n-universe 2000 --n-sing-keep 5 --n-pair-keep 2 \
  --n-unlensed-inj 1000 --n-lensed-inj 1000

python scripts/mock_lensing/generate_mock_lensing.py \
  --outdir data/mock_lensing_poisson \
  --conditioning poisson_counts \
  --n-universe 2000 --max-sing-keep 10 --max-pair-keep 5 \
  --n-unlensed-inj 1000 --n-lensed-inj 1000
```

`--conditioning fixed_counts` preserves the requested kept singleton/pair counts
when enough detections exist.  `--conditioning poisson_counts` keeps the observed
counts from the simulated universe, with `--max-sing-keep` and `--max-pair-keep`
serving only as optional file-size caps.  The generated `manifest.json` and
`truth.json` record the conditioning mode, kept counts, lensing hyperparameters,
weak-lensing hyperparameters, selection model, PE prior convention, and pair
truth mapping.

### 2. Run singleton-only and J=2 inference

All commands should use the installed console script `darksirens_inference_lensing`
or the equivalent module form `python -m darksirens.cli.inference_lensing`.
There is no `darksirens.tool` lensing path.

Singleton-only mode ignores lensed-injection, pair-PE, and partition paths:

```bash
darksirens_inference_lensing \
  --gw_path data/mock_lensing_fixed/mock_gw_pe.h5 \
  --gwselection_path data/mock_lensing_fixed/mock_gw_selection.h5 \
  --cluster_mode off \
  --wl_backend lognormal \
  --fix_cosmology true --fix_survey true --fix_population true \
  --sampler dynesty --nlive 40 --dlogz 10 --max_samples 0 \
  --save_path runs/lensing/off
```

J=2 mode adds lensed selection injections, pair PE, and either a fixed partition
or exact marginalisation over a candidate-pair graph:

```bash
darksirens_inference_lensing \
  --gw_path data/mock_lensing_fixed/mock_gw_pe.h5 \
  --gwselection_path data/mock_lensing_fixed/mock_gw_selection.h5 \
  --lensed_injections_path data/mock_lensing_fixed/mock_lensed_injections.h5 \
  --pair_pe_path data/mock_lensing_fixed/mock_pair_pe.h5 \
  --partition_path data/mock_lensing_fixed/partition.json \
  --partition_mode fixed \
  --cluster_mode j2 \
  --wl_backend lognormal \
  --fix_cosmology true --fix_survey true --fix_population true \
  --sampler dynesty --nlive 40 --dlogz 10 --max_samples 0 \
  --pe_max_per_pair 64 \
  --save_path runs/lensing/j2_fixed
```

For exact candidate-pair marginalisation, replace the fixed partition with the
candidate graph written by the mock generator:

```bash
darksirens_inference_lensing \
  --gw_path data/mock_lensing_fixed/mock_gw_pe.h5 \
  --gwselection_path data/mock_lensing_fixed/mock_gw_selection.h5 \
  --lensed_injections_path data/mock_lensing_fixed/mock_lensed_injections.h5 \
  --pair_pe_path data/mock_lensing_fixed/mock_pair_pe.h5 \
  --candidate_pairs_path data/mock_lensing_fixed/candidate_pairs.json \
  --partition_mode marginalize_exact --max_exact_partitions 10000 \
  --cluster_mode j2 \
  --wl_backend lognormal \
  --fix_cosmology true --fix_survey true --fix_population true \
  --sampler dynesty --nlive 40 --dlogz 10 --max_samples 0 \
  --pe_max_per_pair 64 \
  --save_path runs/lensing/j2_marginalized
```

### Fixed partitions vs candidate-pair marginalisation

`--partition_mode fixed` evaluates one explicit event split from
`--partition_path`.  The mock generator's `partition.json` is the truth
partition and is useful for validation and controlled recovery studies.

`--partition_mode marginalize_exact` enumerates compatible partitions from
`--candidate_pairs_path`, combines each partition likelihood with its prior
weight, and log-sum-exp marginalises over the finite candidate graph.  Use this
only for small candidate sets; `--max_exact_partitions` is a guardrail against
accidental combinatorial explosions.

### Pair-tag selection

Mock generation supports `--pair-tag-model {none,constant,min_snr_proxy}`.
`none` stores `p_tag=1`; `constant` stores `--pair-tag-prob`; and
`min_snr_proxy` writes a deterministic mock-only tag probability based on the
weaker image detection proxy.  The tag factor is stored in
`mock_lensed_injections.h5` and affects the J=2 cluster selection integral, not
the per-pair PE likelihood.

### Optional time-delay marks

The mock generator writes optional SIS time-delay metadata into
`mock_pair_pe.h5`: `delta_t_obs`, `sigma_delta_t`, `true_y`, `true_delta_t`, and
image arrival-time attributes.  These marks are inert by default.  Add
`--pair_marks time` to include the Gaussian time-delay mark in the J=2 likelihood.
For fixed partitions, the likelihood continues to read ordinal `pair_pe`
metadata (`pair_0`, `pair_1`, ...); `--pair_time_sigma_sec` is only a fallback
when pair metadata has `delta_t_obs` but omits `sigma_delta_t`.  For exact
candidate-pair marginalization (`--partition_mode marginalize_exact`), time
marks must be stored on every selectable candidate edge in `candidate_pairs.json`
so each evaluated partition can gather marks by candidate-edge identity.

Example candidate edge with time marks::

  {
    "i": 4,
    "j": 19,
    "log_prior_odds": -2.0,
    "label": "candidate",
    "marks": {
      "delta_t_obs": 12345.0,
      "sigma_delta_t": 3600.0
    }
  }

Both `marks.delta_t_obs` and positive finite `marks.sigma_delta_t` must be
present together.  Marginalized `--pair_marks time` runs fail preflight if any
candidate edge lacks these marks.

### Simulated sky-overlap edge mark

The mock lensing pipeline can attach a lightweight event-pair sky-consistency
mark to candidate edges.  The generator assigns simulated source truth positions
(`ra_true`, `dec_true`) and observed circular-Gaussian sky localizations
(`ra_mean`, `dec_mean`, `sky_sigma_rad`) to each mock event in
`observed_catalog.json`.  For lensed images of the same source, the two observed
means are independent noisy draws around the same true sky position, so they tend
to have larger overlap than unrelated events.

`scripts/mock_lensing/build_candidate_pairs_from_observed.py` can write
`marks.log_sky_overlap`, an approximate tangent-plane log overlap for two
circular Gaussian sky posteriors.  This is an event-pair sky-consistency score
for simulated candidate-pair ranking/prior studies; it is **not** a
galaxy-catalog dark-siren likelihood, does not use LSS information, and does not
model host-galaxy probabilities.  It is intended only to mimic a real
candidate-pair prior/score inside the spectral-siren lensing workflow.

The builder options are:

* `--include_sky_marks true|false` to write or omit `marks.log_sky_overlap`.
* `--sky_sigma_floor_rad FLOAT` to regularize very small circular Gaussian sky
  widths.
* `--sky_overlap_weight FLOAT` to optionally add a weighted sky-overlap term to
  the builder's candidate-edge `log_prior_odds`.

For inference-time use with exact candidate-pair marginalization, keep
`marks.log_sky_overlap` in `candidate_pairs.json` and request it as an edge-prior
mark, for example `--edge_mark_prior_keys log_sky_overlap`.  Preflight fails if
that requested mark is missing or non-finite on any selectable candidate edge.

### Lens-rate sampling

By default, `--fix_lens_rate true` fixes SIS optical-depth hyperparameters to
`--sl_tau_A` and `--sl_tau_n`.  Set `--fix_lens_rate false` to sample
`log10_tau_A` and `tau_n`; use `--lens_prior_overrides` to change their bounds,
and `--fixed_parameter_values` to hold one of them fixed for a controlled
validation run.

### Diagnostics files

Every inference run writes `results.hdf5`, `settings.json`, `diagnostics.json`,
and `diagnostics.hdf5` under the timestamped run directory.  The diagnostics are
evaluated at the prior midpoint before sampling and include total, singleton,
pair, and selection log-likelihood components; observed singleton/pair counts;
cluster and weak-lensing modes; pair batching and quadrature settings; and lens
rate settings.  They are the fastest way to verify that `cluster_mode=off` is
singleton-only, `cluster_mode=j2` sees the expected pairs, and exact
marginalisation found the intended candidate partitions.

## Validation

### Minimal local validation

The quickest local smoke test generates a tiny mock, runs `cluster_mode off`,
runs `cluster_mode j2`, and, when `RUN_MARGINALIZE_EXACT=1`, also runs exact
candidate-pair marginalisation if `candidate_pairs.json` exists:

```bash
bash scripts/mock_lensing/run_tiny_lensing_validation.sh
```

Useful environment overrides include `OUTDIR`, `RUNROOT`, `N_UNIVERSE`,
`MAX_SING_KEEP`, `MAX_PAIR_KEEP`, `NSAMP`, `N_UNLENSED_INJ`, `N_LENSED_INJ`,
`SAMPLER_ARGS`, and `RUN_MARGINALIZE_EXACT=1`.

### Full validation

For the lightweight F1/F2-style validation matrix, run the Python validator.  It
uses tiny in-repo mocks and prior-midpoint diagnostics to check singleton-only
mode, true J=2 pairs, deliberately wrong pair partners, and a null J=2 mock with
zero observed pairs:

```bash
python scripts/mock_lensing/run_lensing_validation.py \
  --profile tiny \
  --workdir /tmp/ds_lensing_validation

python -m pytest tests/test_lensing_validation_script.py
```

The validation passes only if true-pair J=2 diagnostics are finite, wrong pair
partners have a lower `pair_logL_sum` than the truth partition, the null mock
reports zero observed pairs, and `cluster_mode=off` remains singleton-only.


### Evidence/recovery validation

The evidence/recovery validator is an opt-in, heavier mock-only matrix for
spectral-siren lensing.  Keep using the tiny diagnostic validation above for
fast default checks; it is unchanged and remains the recommended smoke test:

```bash
bash scripts/mock_lensing/run_tiny_lensing_validation.sh
```

Preview the heavier evidence validation without generating mocks or launching a
sampler with dry-run mode.  `--dry_run true` writes `validation_plan.json` with
the mock-generation commands and per-case inference commands, then exits:

```bash
python scripts/mock_lensing/run_lensing_evidence_validation.py \
  --profile tiny_evidence \
  --workdir /tmp/ds_lens_evidence \
  --dry_run true
```

For a fast end-to-end wiring check, use diagnostics-only mode.  This still
generates the tiny mocks, runs the lensing CLI preflight before every case, and
executes the prior-midpoint compile/diagnostics path, but it passes
`--max_samples 0` so it does **not** produce meaningful sampler evidence:

```bash
python scripts/mock_lensing/run_lensing_evidence_validation.py \
  --profile tiny_evidence \
  --workdir /tmp/ds_lens_evidence_diag \
  --use_unified_observed_catalog true \
  --diagnostics_only true
```

A local opt-in evidence run uses loose nested-sampling settings so that it is
manageable on a workstation rather than production scale.  This is the mode that
can populate evidence-based `logZ` comparisons; it is intentionally not part of
default CI and should be run locally or on a cluster before producing paper
figures:

```bash
python scripts/mock_lensing/run_lensing_evidence_validation.py \
  --profile tiny_evidence \
  --workdir /tmp/ds_lens_evidence \
  --sampler dynesty \
  --nlive 40 \
  --dlogz 10 \
  --pair_batch_size 8 \
  --reuse
```

By default the runner also launches a preflight-only command before each actual
inference command and writes `<case>/preflight.json`; `--skip_preflight true` is
available only for debugging.  If a preflight fails, that case is marked
`failed_preflight`, the sampler is not launched for that case, the remaining
independent cases continue, and the runner exits nonzero.

The runner writes `validation_summary.json` and `validation_summary.md`.  The
matrix compares a singleton-only true catalog (`off_true_catalog`), fixed true
J=2 partition (`j2_fixed_true`), deliberately shuffled/wrong fixed partition
(`j2_fixed_wrong`), zero-pair null mock (`j2_null` plus `off_null` for evidence
control), exact marginalized candidate-pair mode when `candidate_pairs.json` is
available (`j2_marginalized`), and a batched fixed-true run (`j2_batched`).  It
collects each case status, command, preflight command and JSON report, start and
finish times, runtime, return code, latest run directory, `diagnostics.json`,
`results.hdf5` attributes, sampler evidence and `logZerr` when available,
labels, lens-parameter posterior summaries when sampled, and runtime metadata.
The Markdown summary includes a table with case, status, `logZ`, `logZerr`,
prior-midpoint `logL_total`, pair likelihood sum, pair count, runtime, and run
directory.

Pass/fail checks cover run completion, finite diagnostics, true-pair
`pair_logL_sum` exceeding the shuffled partition, null/off modes reporting zero
pairs, batched and unbatched prior-midpoint `logL_total` agreement at `1e-6`,
evidence ordering when `logZ` is available, and marginalized-run posterior-pair
metadata.  Evidence-based checks are those that use `results.hdf5` `logZ`
attributes, such as true-vs-wrong evidence ordering and the null J=2 vs off
comparison when both evidences exist.  Prior-midpoint diagnostic checks use
`diagnostics.json`; they prove the likelihood plumbing compiled and evaluated at
the prior midpoint, but they are not evidence.  If null-run `logZ` values are
unavailable, the runner records `logL_j2_null_minus_off_null` and warns that the
fallback is diagnostic-only, not evidence.

Exact marginalized checks require `partition_mode == "marginalize_exact"`, at
least one enumerated partition, finite `expected_n_pairs`, posterior pair
probabilities in `[0, 1]`, and posterior probabilities whose sum matches
`expected_n_pairs`.  Newer result files may also include `expected_n_pairs` and
`map_partition_n_pairs` attributes; absence of those attributes is warned, not
failed, for compatibility with older branches.  The optional
`--run_lens_rate_recovery true` path is experimental; it uses a
Poisson-conditioned mock and samples `log10_tau_A` while fixing `tau_n` through
`--fixed_parameter_values`.


### Simulated end-to-end study runner with J=2/off evidence pairs

`run_simulated_lensing_study.py` is a mock-only end-to-end study runner for the
candidate-graph recovery cases.  By default it now plans a matched
`cluster_mode=off` control for every J=2 case (`--run_off_controls true`).  Each
control uses the same observed GW PE file, observed catalog, GW selection file,
weak-lensing backend, population model, and fixed cosmology/survey/population
choices as the J=2 run, but intentionally omits J=2-only inputs such as lensed
injections, pair metadata, candidate-pair graphs, partition mode, pair marks, and
pair-tag settings.  The off control fixes the lens-rate hyperparameters
(`--fix_lens_rate true`) because those parameters are irrelevant when
`--cluster_mode off`.

Preview the paired commands without generating mocks or running samplers:

```bash
python scripts/mock_lensing/run_simulated_lensing_study.py \
  --profile tiny \
  --workdir /tmp/ds_simulated_lensing_study \
  --dry_run true \
  --run_off_controls true
```

A future evidence-producing run should disable diagnostics-only mode and use real
sampler settings.  The runner writes each J=2 case under `runs/<case>` and the
matched off control under `runs/<case>__off`:

```bash
python scripts/mock_lensing/run_simulated_lensing_study.py \
  --profile small \
  --workdir /tmp/ds_simulated_lensing_study_evidence \
  --sampler dynesty \
  --nlive 80 \
  --dlogz 10 \
  --diagnostics_only false \
  --run_off_controls true
```

`validation_summary.json` stores a nested record for each case with separate
`j2` and `off` sections (`status`, `run_dir`, `logZ`, `logZerr`, diagnostics,
and result attributes).  When both required sampler evidences exist, the runner
computes `delta_logZ_j2_minus_off = logZ_j2 - logZ_off`; if both evidence errors
are available it also writes `delta_logZerr`.  If either evidence is missing, the
delta is left unset and the case status reflects the failed J=2 and/or off-control
inference.  The Markdown summary includes a paired evidence table with J=2
status, off status, both evidences, the evidence delta, expected pair count, and
run directories.  Candidate recovery outputs such as
`posterior_pair_probabilities.csv` and `truth_recovery_summary.csv` remain
J=2-based diagnostics.

`--diagnostics_only true` still runs the paired J=2/off commands with
`--max_samples 0` when `--run_off_controls true`, which is useful for cheap
wiring checks.  Those outputs are not sampler evidence; the runner records the
warning `diagnostics_only: evidence deltas are not meaningful` and leaves
`delta_logZ_j2_minus_off` unset unless a future backend produces meaningful
log-evidence values despite diagnostics-only mode.

### Simulated study plots

After running the simulated end-to-end lensing study, generate one diagnostic
figure per plot with the plotting helper.  The helper reads the study directory
written by `run_simulated_lensing_study.py`, skips plots whose optional inputs are
missing, and writes `plot_manifest.json` with produced plots, skipped plots, and
warnings:

```bash
python scripts/mock_lensing/plot_simulated_lensing_study.py \
  --study_dir /tmp/ds_simulated_lensing_study \
  --outdir /tmp/ds_simulated_lensing_study/plots \
  --format png \
  --show false \
  --min_edge_probability 0.0
```

The deterministic output names are:

* `fig_pair_probabilities.{png,pdf}`: posterior pair probabilities grouped by
  truth label from `posterior_pair_probabilities.csv`.
* `fig_pair_prob_vs_edge_score.{png,pdf}`: posterior pair probability versus
  candidate-edge `log_prior_odds` from per-case `candidate_pairs.json`; this is
  skipped if candidate-edge scores are unavailable.
* `fig_evidence_matrix.{png,pdf}`: available sampler evidence or evidence-delta
  values from `validation_summary.json`.  This requires evidence-bearing runs;
  diagnostics-only studies usually skip it or should treat it as non-science
  diagnostics.
* `fig_lens_rate_recovery.{png,pdf}`: posterior `log10_tau_A` summaries when
  lens-rate sampling produced them.  It is skipped for fixed-rate or
  diagnostics-only studies without posterior samples.
* `fig_candidate_graph_summary.{png,pdf}`: candidate-edge and MAP-pair counts
  from `partition_component_summary.csv`.
* `fig_ablation_summary.{png,pdf}`: full marks versus no-sky, no-time, and bad
  `p_tag` ablations from `truth_recovery_summary.csv`.
* `fig_false_positive_summary.{png,pdf}`: maximum and summed false-edge posterior
  probabilities from `truth_recovery_summary.csv`.

The pair-probability, edge-score, candidate-graph, ablation, and false-positive
plots are diagnostic summaries and can be produced from diagnostics-only study
outputs when the corresponding CSV/JSON artifacts exist.  Evidence plots require
real sampler outputs (`results.hdf5` attributes propagated into
`validation_summary.json`) and should not be interpreted from `--diagnostics_only
true` runs, where `max_samples=0` makes evidence unavailable or non-meaningful.
The plotting utility is still simulation-only and does not ingest GWTC-5, galaxy
catalogs, LSS maps, or dark-siren host information.

This validation is still not a real LVK science run.  It validates methodology
on controlled mocks only and deliberately avoids dark-siren, LSS, or
catalog-host inference.

## `darksirens.lensing`

The package `__init__` re-exports the lensing parameter containers and factory
helpers.

```{automodule} darksirens.lensing
:members:
:undoc-members:
:show-inheritance:
```
## `darksirens.lensing.wlmagnification`

The weak-lensing magnification PDF $p(\mu\mid z)$. `WLParams` carries the backend
and its parameters. The lognormal backend models $\ln\mu\sim\mathcal N(-\tfrac12
s^2, s^2)$ with variance $s^2(z) = a\,z^b$ (so $\mathbb E[\mu]=1$); the tabulated
backend wraps an external $p(\ln\mu\mid z)$ grid (`make_tabulated_log_p_wl`).
`make_lognormal_wl_params` / `make_tabulated_wl_params` construct the containers
consumed by the likelihood.

```{automodule} darksirens.lensing.wlmagnification
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.grids`

Quadrature nodes for the lensing integrals: `make_hermite_u_grid`
(Gauss-Hermite in the standardised variable $u = (\ln\mu - m(z))/s(z)$, exact for
the lognormal), `make_log_mu_grid` (Gauss-Legendre in $\ln\mu$ for the tabulated
backend), and `make_y_grid` (source-position nodes for strong lensing).

```{automodule} darksirens.lensing.grids
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.slmarks`

Singular-isothermal-sphere (SIS) strong-lensing relations. `SISLensParams` holds
the lens model; `tau_2_SIS` / `tau_4_SIS` are the (doubly/quadruply imaged)
optical depths, `log_p_y_SIS` the source-position PDF $p(y)\propto y$, and
`mu_plus_minus_from_y` the image magnifications $\mu_\pm = 1 \pm 1/y$ (with the
inverse `y_from_mu_plus` and the time delay `delta_t_from_y`).

```{automodule} darksirens.lensing.slmarks
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.clusters`

The `ClusterSet` container and its HDF5 I/O (`make_cluster_set`,
`load_clusters`, `save_clusters`, `empty_cluster_set`) describing the
multiply-imaged systems associated with each event.

```{automodule} darksirens.lensing.clusters
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.lensed_injections`

The `LensedInjectionSet` container plus builders/loaders
(`make_lensed_injection_set`, `load_lensed_injections`, `save_lensed_injections`)
for the lensed selection injections used by the cluster selection term.

```{automodule} darksirens.lensing.lensed_injections
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.wl_weight`

The weak-lensing per-sample weights. `log_sample_weight_wl_lognormal_hermite`
marginalises the PE weight over magnification by Gauss-Hermite quadrature (the
redshift-prior Jacobian is evaluated **inside** the integral because $z$ depends
on $\mu$); `log_sample_weight_wl_or_standard` dispatches to the tabulated
quadrature or the standard (non-WL) weight. Both reduce exactly to
`log_sample_weight` as the lensing variance $\to 0$.

```{automodule} darksirens.likelihood.wl_weight
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.pair_kde`

A fixed-bandwidth Gaussian KDE over the image-pair parameters
$(m_1^{\rm det}, q, d_L^{\rm app}, \chi_{\rm eff})$ used by the cluster
likelihood. `make_pair_kde` builds a per-event KDE (Silverman diagonal
bandwidth), `stack_pair_kdes` batches them, and `log_eval_pair_kde` evaluates the
log-density inside the JIT.

```{automodule} darksirens.likelihood.pair_kde
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.cluster_likelihood`

The strong-lensing cluster likelihood for a lensed image pair: it combines the
SIS lens marks, the source-position PDF, and the pair KDE into the per-pair
contribution to the hierarchical likelihood.  When `pair_marks=time`, an additional Gaussian `log p(delta_t_obs | y, T0)` term is added inside the same `y` quadrature; `pair_marks=none` omits this term for backward compatibility.

```{automodule} darksirens.likelihood.cluster_likelihood
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.cluster_selection`

The cluster selection correction. `compute_cluster_selection_term` returns its
own $(\ln\mu, N_{\rm eff}, \ln\widehat\sigma^2_\mu)$ from the lensed injection
set, and `combined_selection_log_correction` merges the singleton and cluster
selection integrals (using the variance term from each) into one
Vitale-criterion-guarded correction.

```{automodule} darksirens.likelihood.cluster_selection
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.likelihood_with_clusters`

The top-level driver `darksiren_log_likelihood_with_clusters` that splits events
into singletons and lensed pairs, evaluates each with the appropriate likelihood,
and applies the combined selection correction. The lensing CLI accepts `--pair_marks {none,time}` and `--pair_time_sigma_sec` as a fallback when time-delay metadata provides `delta_t_obs` but omits `sigma_delta_t`. It is invoked by the
[`darksirens_inference_lensing`](tools.md) CLI.

```{automodule} darksirens.likelihood.likelihood_with_clusters
:members:
:undoc-members:
:show-inheritance:
```

## Weak-lensing singleton selection consistency

`darksirens.cli.inference_lensing` exposes `--wl_selection {standard,wl_lognormal}`
for spectral-siren lensing runs.  The default, `standard`, preserves the legacy
behavior: `--wl_backend lognormal` marginalizes weak-lensing magnification in the
singleton PE weights, while singleton injection selection uses the ordinary
spectral-siren selection integral.

Set `--wl_selection wl_lognormal` to apply the same lognormal/Hermite
weak-lensing marginalization to singleton selection weights.  This option only
applies when `--wl_backend lognormal`.  If weak lensing is disabled, the
standard selection path is used; if `--lensing_wl_a 0`, the WL-aware Hermite
path reduces numerically to the standard selection path.  The option is recorded in both
`settings.json` and the `results.hdf5` `wl_selection` attribute.

This switch affects singleton selection only.  The J=2 strong-lensing
lensed-injection selection estimator may include the optional `p_tag` pair-tag
factor described above, but still does not add dark-siren, LSS, or
catalog-selection support.


### Scaling exact partition marginalization

Exact candidate-pair marginalization enumerates every matching compatible with the candidate graph.  This is exact, but the number of matchings can grow exponentially with graph size, so dense or weakly pruned simulated candidate graphs can otherwise hang before inference starts.

For `--partition_mode marginalize_exact`, the default `--partition_component_mode componentwise` decomposes the candidate graph into connected components, enumerates matchings inside each component, and reports component complexity in preflight and marginalized diagnostics.  Small graphs still produce the same global partitions, posterior pair probabilities, and partition-prior normalizer as global exact enumeration; the decomposition is a scaling guardrail, not an approximation.  If you need legacy behavior, pass `--partition_component_mode global`.

Use the component caps to protect simulation campaigns:

```bash
--max_component_events 12 \
--max_component_edges 30 \
--max_component_partitions 10000 \
--max_total_partitions 50000
```

If any component or the Cartesian product of component partitions exceeds these caps, inference fails early with an error suggesting candidate-graph pruning.  Pruning should happen before inference, for example by tightening time-delay, sky-overlap, mass-distance, spin, or prior-odds cuts in the candidate-pair builder.  This PR does not introduce an approximate sampler over partitions and does not change dark-siren/LSS/catalog-host workflows.

### Marginalized partition diagnostics

For spectral-siren lensing runs with `--cluster_mode j2`, fixed-partition diagnostics describe exactly one materialized partition: the `singleton_indices` and `pair_indices` supplied by `--partition_path`.  Fixed partitions use pair-ordinal time-delay metadata from `pair_pe`.  In contrast, `--partition_mode marginalize_exact` enumerates every compatible matching from `--candidate_pairs_path`, evaluates the scalar likelihood for each partition, and writes diagnostics marginalized over the posterior partition weights.  Marginalized `--pair_marks time` uses edge-indexed candidate marks rather than pair-ordinal `pair_pe` metadata.

A marginalized `diagnostics.json` includes the prior normalizer (`log_z_partition_prior`), marginalized scalar likelihood (`logL_marginalized`, also aliased as `logL_total`), per-partition prior weights and likelihoods, posterior partition probabilities, posterior pair probabilities in the validated candidate-pair order, connected-component summaries (`n_components`, `component_event_indices`, `component_candidate_edge_indices`, `component_n_partitions`, `component_log_z_partition_prior`, `component_expected_n_pairs`, and `component_max_p_pair`), the posterior expected numbers of pairs and singletons, and a compact MAP partition object.  Candidate pairs are treated as unordered edges; if an input pair is written as `(j, i)`, the validated diagnostics report the normalized order `(min(i, j), max(i, j))` while preserving optional `label` and `log_prior_odds` fields.

For marginalized runs, the legacy `results.hdf5` `n_pairs` attribute remains a reference-partition field for backward compatibility and is annotated by `n_pairs_meaning = "reference_partition_n_pairs"`.  Use `expected_n_pairs`, `map_partition_n_pairs`, and `posterior_pair_probabilities` for marginalized inference summaries instead of interpreting `n_pairs` as a posterior pair count.

Example:

```bash
python -m darksirens.cli.inference_lensing \
  --cluster_mode j2 \
  --partition_mode marginalize_exact \
  --candidate_pairs_path candidate_pairs.json \
  --gw_path mock_gw_pe.h5 \
  --gwselection_path mock_gw_selection.h5 \
  --lensed_injections_path mock_lensed_injections.h5 \
  --pair_pe_path mock_pair_pe.h5 \
  --pop_model powerlaw+peak \
  --wl_backend lognormal \
  --fix_cosmology true --fix_survey true --fix_population true \
  --sampler dynesty --nlive 20 --max_samples 0 \
  --save_path runs/marginalized
```

Use fixed mode when you want diagnostics for one assumed partition; use `marginalize_exact` when you want interpretable posterior partition weights, posterior pair probabilities, and the MAP partition for a small candidate graph.

Unified observed-catalog mocks
------------------------------

Unified observed mode
^^^^^^^^^^^^^^^^^^^^^

Spectral-siren lensing inference supports a unified observed-catalog mode in
which PE samples live only in ``mock_observed_gw_pe.h5`` (or another unified GW
PE file).  Fixed ``partition.json`` and ``candidate_pairs.json`` edges use
global observed event indices into that file.  In this mode ``mock_pair_pe.h5``
is optional: when supplied it is read only for pair/edge metadata such as
event-index attributes or time-delay marks, and any duplicated ``image0``/
``image1`` posterior samples are ignored rather than appended to the observed
event catalog.  The old ``mock_pair_pe.h5`` image duplication is retained only
as legacy split-pair mock mode.  Inference consumes observed events, a fixed
partition or candidate graph, selection inputs, and optional marks; truth-shaped
pair PE belongs in validation products, not inference inputs.


PE samples vs pair metadata
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Spectral-siren lensing inputs now distinguish posterior samples from pair or
candidate-edge metadata:

* ``observed_gw_pe.h5`` (or the mock ``mock_observed_gw_pe.h5``) contains the
  observed-event posterior samples.  Unified observed-catalog inference reads PE
  samples from this file only.
* ``candidate_pairs.json`` contains the candidate graph.  Its edge ``marks`` are
  the authoritative source for marginalized ``--pair_marks time`` runs and can
  also provide fixed-partition time marks when the fixed edge appears in the
  candidate graph.
* ``--pair_metadata_path`` names an optional metadata-only HDF5 file for extra
  diagnostics or fixed-partition marks.  The metadata-only schema uses
  ``format_version = "lensing-pair-metadata-1.0"`` and ``pair_k`` groups with
  event indices, labels/truth flags, time-delay marks, lens truth, and source
  identifiers, but no ``image0``/``image1`` PE sample groups.
* ``--pair_pe_path`` with ``image0``/``image1`` groups is the legacy split-pair
  format.  It remains supported for legacy split-pair workflows.  In unified
  observed-catalog mode it is accepted only as a backward-compatible metadata
  source; duplicated image PE groups are ignored and preflight emits a warning.

If both ``--pair_metadata_path`` and ``--pair_pe_path`` are provided, they must
refer to the same path.

Mock lensing generation also writes a unified observed-event catalog by default:
``mock_observed_gw_pe.h5`` plus ``observed_catalog.json``.  This file uses the
same ``gwcat-1.0`` event-major schema as the existing GW PE files, but contains
every observed event in one index system: all unlensed singleton detections
first, followed by image 0 and image 1 for each detected lensed pair.  The
``candidate_pairs.json`` graph refers to these unified observed-event indices,
which makes it the preferred mock format for unknown-partition and
candidate-pair marginalization tests.  The truth partition is retained only as
metadata in ``partition.json`` and ``observed_catalog.json``.

Observed catalog schema
^^^^^^^^^^^^^^^^^^^^^^^

Unified observed-catalog runs should pass ``--observed_catalog_path
observed_catalog.json``.  If omitted, inference accepts a GW PE file whose HDF5
attributes declare ``format_version = "observed-lensing-pe-1.0"`` and
``event_indexing = "global"``.  The older event-count heuristic is retained only
as a deprecated fallback and emits a preflight warning.

The observed-catalog JSON schema is intentionally small:

```json
{
  "format_version": "observed-lensing-catalog-1.0",
  "event_indexing": "global",
  "n_events": 12,
  "events": [
    {
      "event_index": 0,
      "event_id": "mock_event_000",
      "kind": "singleton_or_image",
      "gps_time": 1234567890.0,
      "truth_source_id": 0,
      "truth_image_index": null,
      "truth_is_lensed_image": false
    }
  ]
}
```

Inference requires only the global indexing contract: ``format_version``,
``event_indexing = "global"``, ``n_events``, and one unique ``event_id`` for
each contiguous ``event_index`` from ``0`` through ``n_events - 1``.  ``gps_time``
is optional for now, but if present it must be finite.  The ``truth_*`` fields
are validation-only mock metadata; inference ignores them and must not depend on
them.  Mock ``mock_observed_gw_pe.h5`` files also carry
``format_version = "observed-lensing-pe-1.0"``, ``event_indexing = "global"``,
``n_events``, ``source = "mock_lensing"``, and the observed-catalog path and/or
SHA-256 digest.

The fixed-partition workflow remains useful for controlled validation, where the
true singleton and pair assignment is intentionally supplied to the likelihood.
For unknown-partition validation, pass the unified observed catalog as
``--gw_path`` and use ``--partition_mode marginalize_exact`` with
``candidate_pairs.json``.

Generate a unified mock::

  python scripts/mock_lensing/generate_mock_lensing.py \
    --outdir /tmp/ds_lens_unified \
    --conditioning fixed_counts \
    --n-universe 3000 --n-sing-keep 2 --n-pair-keep 2 \
    --n-unlensed-inj 1000 --n-lensed-inj 1000 \
    --write-unified-observed-catalog true

Run with the fixed truth partition on the unified observed catalog::

  python -m darksirens.cli.inference_lensing \
    --gw_path /tmp/ds_lens_unified/mock_observed_gw_pe.h5 \
    --observed_catalog_path /tmp/ds_lens_unified/observed_catalog.json \
    --gwselection_path /tmp/ds_lens_unified/mock_gw_selection.h5 \
    --lensed_injections_path /tmp/ds_lens_unified/mock_lensed_injections.h5 \
    --pair_metadata_path /tmp/ds_lens_unified/mock_pair_metadata.h5 \
    --partition_path /tmp/ds_lens_unified/partition.json \
    --partition_mode fixed --cluster_mode j2

Run exact candidate-pair marginalization::

  python -m darksirens.cli.inference_lensing \
    --gw_path /tmp/ds_lens_unified/mock_observed_gw_pe.h5 \
    --observed_catalog_path /tmp/ds_lens_unified/observed_catalog.json \
    --gwselection_path /tmp/ds_lens_unified/mock_gw_selection.h5 \
    --lensed_injections_path /tmp/ds_lens_unified/mock_lensed_injections.h5 \
    --pair_metadata_path /tmp/ds_lens_unified/mock_pair_metadata.h5 \
    --candidate_pairs_path /tmp/ds_lens_unified/candidate_pairs.json \
    --partition_mode marginalize_exact --cluster_mode j2

## Preflight checks

Before a spectral-siren lensing run, the lensing CLI can validate the input
schema without loading the full likelihood or triggering expensive JAX
compilation/sampling:

```bash
python -m darksirens.cli.inference_lensing \
  --gw_path mock_gw_pe.h5 \
  --gwselection_path mock_gw_selection.h5 \
  --lensed_injections_path mock_lensed_injections.h5 \
  --pair_pe_path mock_pair_pe.h5 \
  --partition_path partition.json \
  --cluster_mode j2 \
  --sampler dynesty \
  --preflight_only true
```

`--preflight_json PATH` writes the structured report to a chosen location; when
omitted in preflight-only mode the CLI writes `preflight.json` under
`--save_path`. Normal lensing inference also runs the lightweight preflight
before data loading and stops early on fatal errors.

Common failures include missing J=2 files, a fixed partition that reuses an event
more than once, `candidate_pairs.json` whose `n_events` differs from the GW PE
file, malformed pair-image `prior_wt` values, pair-time marks without
`delta_t_obs`/positive `sigma_delta_t`, and weak-lensing options such as
`wl_selection=wl_lognormal` paired with a non-lognormal WL backend.

### Simulated candidate graph builder from observed events

`candidate_pairs.json` can be built directly from the unified simulated observed
catalog rather than from the truth partition.  The simulation-only builder is:

```bash
python scripts/mock_lensing/build_candidate_pairs_from_observed.py \
  --gw_path data/mock_lensing/mock_observed_gw_pe.h5 \
  --observed_catalog_path data/mock_lensing/observed_catalog.json \
  --truth_path data/mock_lensing/truth.json \
  --out data/mock_lensing/candidate_pairs.json \
  --max_edges_per_event 4 \
  --max_total_edges 1000 \
  --time_window_sec inf \
  --mass_distance_top_k 0 \
  --include_time_marks true \
  --include_truth_labels true \
  --seed 2026
```

The same step can be requested during mock generation with
`--build_candidate_pairs_from_observed true`.  In that mode the generator first
writes the simulated observed PE file and `observed_catalog.json`, then overwrites
`candidate_pairs.json` with edges scored from observed event metadata and PE
posteriors.

The builder uses only inference-available observed quantities to decide which
edges are included and how they are scored: observed GPS-time separation when
available, posterior-sample summaries for detector-frame mass and mass ratio,
an apparent-distance/magnification compatibility proxy, and optional effective
spin consistency.  These observable scores determine `log_prior_odds`; truth is
not consulted for inclusion or ranking.

When `--include_truth_labels true` is used and truth fields exist in the
simulated `observed_catalog.json`, each edge receives a validation-only `label`:
`"true"` for two lensed images with the same `truth_source_id`, and `"wrong"`
otherwise.  Inference ignores this label; it is intended only for diagnostics and
end-to-end validation.  Runs without truth labels are supported with
`--include_truth_labels false`.

The output format is `candidate-pairs-1.0`:

```json
{
  "format_version": "candidate-pairs-1.0",
  "n_events": 6,
  "pairs": [
    {
      "i": 0,
      "j": 5,
      "log_prior_odds": -3.4,
      "label": "wrong",
      "marks": {
        "delta_t_obs": 12345.0,
        "sigma_delta_t": 100.0,
        "log_mass_distance_score": -0.3
      }
    }
  ],
  "builder": {
    "name": "build_candidate_pairs_from_observed"
  }
}
```

For backward compatibility with existing marginalization loaders, the generated
file also includes a legacy `candidate_pairs` alias containing the same edge
list.  Preflight validates the `format_version`, checks `n_events`, and requires
all numeric mark fields to be finite.  This is the simulation-realistic path for
current end-to-end studies and is intended to be GWTC-ready later, but it does
not ingest real GWTC-5 data and does not add galaxy-catalog, LSS, or dark-siren
candidate support.

## Candidate edge marks

Simulation candidate-pair files can attach structured, edge-level information in
an optional `marks` object.  Time-delay fields remain supported, and additional
interpretable log-scores can be carried without changing the pair schema:

```json
{
  "format_version": "candidate-pairs-1.0",
  "n_events": 6,
  "candidate_pairs": [
    {
      "i": 0,
      "j": 5,
      "log_prior_odds": -2.1,
      "label": "true",
      "marks": {
        "delta_t_obs": 12345.0,
        "sigma_delta_t": 3600.0,
        "log_sky_overlap": -0.4,
        "log_mass_distance_score": -1.2,
        "log_spin_score": -0.1,
        "log_custom_sim_score": -0.7
      }
    }
  ]
}
```

Validation rules are intentionally strict.  `delta_t_obs` and
`sigma_delta_t` must either both be present or both be absent, and
`sigma_delta_t` must be positive and finite.  The built-in log-score fields
`log_sky_overlap`, `log_mass_distance_score`, and `log_spin_score` must be
finite when present.  Custom simulation marks are accepted only when their key
starts with `log_` and their value is finite; unknown non-`log_` keys are
rejected so misspellings do not silently change inference.

Edge marks have two roles:

* **Prior marks** are requested with `--edge_mark_prior_keys`, a comma-separated
  list of `log_*` keys.  Each requested mark is added to the effective edge
  `log_prior_odds` before exact partition enumeration.  For example,
  `--edge_mark_prior_keys log_sky_overlap,log_mass_distance_score` lets a
  simulation use sky-overlap and mass-distance compatibility as interpretable
  edge-prior contributions while preserving the base `log_prior_odds` field.
* **Likelihood marks** are requested with `--edge_mark_likelihood_keys`.  In
  this PR, only the existing time-delay likelihood is implemented, via
  `pair_marks=time` or the `time`/`delta_t_obs` likelihood key.  Other
  likelihood marks are parsed and rejected with a clear not-implemented error
  until corresponding likelihood terms are added.

By default, no extra edge marks are used: existing `log_prior_odds` behavior is
unchanged, and time marks affect the likelihood only when the run requests the
time mark.  Marginalized diagnostics include a `marks` dictionary on each
`posterior_pair_probabilities` entry so downstream simulation studies can audit
which edge-level information contributed to each candidate pair.

## Simulated pair-tag selection

The strong-lensing mock pipeline can attach a simulated pair-tag probability,
`p_tag`, to lensed injection pairs.  This quantity is the probability that an
already both-detected image pair is identified or tagged as a candidate pair by a
pair-level statistic.  It is not the single-event detection probability, which is
still represented by each image's `detected` flag in the lensed-injection file.

This support is simulation-calibration scaffolding only.  It is intended for
robustness studies of spectral-siren lensing inference and is not a GWTC-5 or
real-search calibration.  The available deterministic mock models are:

* `constant`: uses `--pair_tag_constant` for every kept pair.
* `snr_time`: computes `p_tag` from `snr_image0`, `snr_image1`, and
  `delta_t_obs` (or `true_delta_t`) stored in the mock lensed injections.
* `snr_time_sky`: also requires `log_sky_overlap`.
* `file`: reads a small JSON model specification from `--pair_tag_selection_path`.

Inference defaults preserve the existing behavior: if no nontrivial model is
selected and the lensed-injection file already contains `p_tag_per_source` or
`log_p_tag_per_source`, that stored selection correction is used.  Passing
`--pair_tag_perturb_logit` intentionally shifts the model in logit space so
validation runs can compare correct and perturbed pair-tag selection.

Recommended robustness studies should run at least three simulated cases: the
correct injected `p_tag` model, a perturbed `p_tag` model, and a constant `p_tag`
model.  Differences in the recovered lens-rate parameters quantify sensitivity
to pair-identification/tagging selection uncertainty.

## Simulated end-to-end study

`scripts/mock_lensing/run_simulated_lensing_study.py` is the mock-only runner for
an end-to-end spectral-siren lensing validation study.  A single command
constructs observed-event mocks, rebuilds the candidate graph from observed PE
and observed metadata, runs CLI preflight checks, launches inference, and then
uses truth labels only in the final evaluation step:

```bash
python scripts/mock_lensing/run_simulated_lensing_study.py \
  --workdir runs/simulated_lensing_study \
  --profile tiny \
  --sampler dynesty \
  --diagnostics_only true
```

The runner intentionally does not ingest real GWTC releases and does not use
galaxy catalogs, LSS fields, host probabilities, or dark-siren catalog support.
Its inference inputs are the unified observed-event PE file
`mock_observed_gw_pe.h5`, `observed_catalog.json`, the selection files, and the
observed-data candidate graph `candidate_pairs.json`; truth-shaped pair PE is
not part of the inference path.

### Study cases

The built-in matrix covers the following controlled simulation cases:

* `A_no_true_pairs_sparse_wrong_graph`: a null mock with no injected lensed
  pairs and a sparse graph of candidate wrong edges.
* `B_true_pairs_clean_graph`: injected pairs with a low-degree graph intended to
  isolate the strongest observed-data candidates.
* `C_true_pairs_many_wrong_edges`: injected pairs with a larger wrong-edge
  budget to test false-positive control.
* `D_true_pairs_bad_pair_tag`: injected pairs with a deliberately perturbed
  pair-tag model at inference time to quantify `p_tag` model bias.
* `E_true_pairs_no_sky_marks`: injected pairs with sky marks omitted from the
  candidate graph and edge-prior model.
* `F_true_pairs_no_time_marks`: injected pairs with time-delay marks omitted
  from the pair likelihood.
* `G_true_pairs_full_marks`: injected pairs with time and sky marks enabled.
* `H_ambiguous_components`: injected pairs with higher-degree components to
  exercise exact candidate-partition marginalisation in ambiguous graphs.

### Outputs

Each run writes a reproducible manifest and summary products under `--workdir`:

* `run_manifest.json`: full command plan, case settings, seeds, preflight
  commands, and inference commands.  With `--dry_run true`, this manifest and
  `validation_plan.json` are written without generating mocks, building graphs,
  running CLI preflight, compiling likelihoods, or sampling.
* `preflight_summary.json` / `preflight_summary.md`: written by
  `--preflight_only true` with per-case pass/fail status, generated files,
  candidate graph complexity, requested/available edge marks, pair-tag settings,
  warnings, and errors.
* `runs/<case>/preflight.json` and `runs/<case>/file_contract_report.json`:
  per-case CLI preflight and unified file-contract reports.
* `validation_summary.json`: per-case preflight/inference status, diagnostics,
  result attributes, recovery metrics, evidence placeholders, lens-rate summary
  fields, and pair-tag bias fields.
* `validation_summary.md`: compact Markdown table for quick inspection.
* `posterior_pair_probabilities.csv`: posterior probability for each candidate
  edge, annotated with truth labels only after inference.
* `partition_component_summary.csv`: candidate graph size and partition-level
  summary fields per case.
* `truth_recovery_summary.csv`: injected pair count, expected pair count, MAP
  pair count, mean true-edge posterior probability, maximum/summed false-edge
  posterior probability, and MAP exact-truth-match flag when available.
* `bias_summary.csv`: case-level pair-tag model and perturbation summary for
  diagnosing bad-`p_tag` runs.

### Dry-run, preflight-only, diagnostics-only, and evidence runs

`--dry_run true` is a command-plan preview only: it writes `run_manifest.json`
and `validation_plan.json`, validates generated inference flags for obvious
stale options, and exits before generating mocks or building candidate graphs.

`--preflight_only true` is the cheapest executable end-to-end gate.  It
generates or reuses the simulated mocks, builds `candidate_pairs.json` from the
observed events, validates the split simulation files against the file contract,
runs `darksirens.cli.inference_lensing --preflight_only true` for every selected
case, and, when `--run_off_controls true`, also runs the off-control preflight
into `runs/<case>__off/preflight.json`.  The resulting `preflight_summary.json`
and `preflight_summary.md` report both J=2 and off-control preflight status and
the runner exits nonzero if either required preflight fails.  It then stops before
likelihood JIT compilation, midpoint diagnostics, or sampler execution:

```bash
python scripts/mock_lensing/run_simulated_lensing_study.py \
  --profile tiny \
  --workdir /tmp/ds_lens_study_preflight \
  --preflight_only true
```

`--diagnostics_only true` runs after preflight and forces the inference command
onto the compile and likelihood-diagnostic path (`max_samples=0`, small `nlive`,
loose `dlogz`).  This is appropriate for validating finite diagnostic likelihood
terms without meaningful sampler evidence.  It is intentionally more expensive
than preflight-only mode because it compiles/evaluates the likelihood.  Evidence
values from diagnostics-only runs are not meaningful and should not be used for
paper claims.

Use `--diagnostics_only false` with an adequate profile, sampler, `--nlive`, and
`--dlogz` for evidence and posterior summaries.  These full runs are the basis
for paper figures that compare recovery, evidence differences such as
`delta_logZ_j2_minus_off`, false-positive behavior, and pair-tag model bias
across the simulated study matrix.

## Simulation config

Simulated end-to-end lensing studies can be driven by a reproducible YAML or
JSON config instead of relying only on command-line flags.  The config system is
limited to mock simulations: it does not ingest GWTC-5 data, galaxy catalogs,
large-scale-structure information, or dark-siren host catalogs.

The study runner accepts `--config` plus repeatable dotted-key overrides:

```bash
python scripts/mock_lensing/run_simulated_lensing_study.py \
  --config configs/mock_lensing/tiny_simulated_study.yaml \
  --workdir runs/mock_lensing/tiny \
  --override study.seed=1234 \
  --override inference.diagnostics_only=true
```

JSON configs are always supported.  YAML configs are supported when PyYAML is
installed; otherwise the runner exits with a message asking you to install
PyYAML or use JSON.  Unknown keys fail validation by default.  Use
`--allow_unknown_config_keys true` only when intentionally carrying auxiliary
metadata through a config file.

Example config:

```yaml
study:
  profile: tiny
  seed: 2026
  cases: [B_true_pairs_clean_graph]
mock:
  n_universe: 4000
  n_singletons: 2
  n_lensed_pairs: 2
  nsamp: 48
  n_unlensed_inj: 1000
  n_lensed_inj: 1000
  conditioning: fixed_counts
candidate_graph:
  max_edges_per_event: 2
  max_total_edges: 8
  include_time_marks: true
  include_sky_marks: true
  include_mass_distance_score: true
  edge_mark_prior_keys: [log_sky_overlap]
selection:
  pair_tag_model: snr_time_sky
  pair_tag_constant: 1.0
  pair_tag_perturb_logit: 0.0
inference:
  partition_mode: marginalize_exact
  sampler: dynesty
  nlive: 32
  dlogz: 10.0
  pair_batch_size: 256
  y_nodes_pair: 64
  diagnostics_only: false
```

Every run writes the resolved configuration to `resolved_config.yaml` when
PyYAML is available, or `resolved_config.json` otherwise.  The same resolved
configuration is embedded in `run_manifest.json` with the planned commands, so a
dry run records the exact simulation, candidate-graph, selection, and inference
settings that would be used.

## Unified lensing file contract

The simulated end-to-end lensing study now targets the same file/interface
contract that a future GWTC adapter must produce.  This interface is the adapter
boundary only: this release does **not** download, parse, or ingest GWTC-5 data,
and it does not add galaxy-catalog, LSS, or dark-siren inputs to the lensing
workflow.

The inference CLI consumes these contract files:

* `observed_gw_pe.h5`: posterior samples for all observed events in one global
  event-index system.  Required root attrs are `format_version =
  "observed-lensing-pe-1.0"`, `event_indexing = "global"`, `n_events` (also
  accepted as legacy `nobs`), and `nsamp`.  Required event-major flattened
  datasets include `m1det`, `m2det`, `dL`, `chieff`, and `p_pe`, each of length
  `n_events * nsamp`.  Event index `k` owns slice `[k*nsamp:(k+1)*nsamp]`.
* `observed_catalog.json`: event metadata with `format_version =
  "observed-lensing-catalog-1.0"`, `event_indexing = "global"`, `n_events`, and
  an `events` list.  Every event requires contiguous `event_index`, unique
  `event_id`, and optional finite `gps_time`.  Truth fields such as
  `truth_source_id`, `truth_image_index`, labels, and true sky positions are
  validation-only conveniences for simulations and are not required by
  inference.
* `candidate_pairs.json`: a graph over global event indices with
  `format_version = "candidate-pairs-1.0"`, `n_events`, and a `pairs` list.  Each
  edge requires distinct in-range `i` and `j` plus finite `log_prior_odds`.
  Optional `marks` may include `delta_t_obs`/`sigma_delta_t` together and any
  finite `log_*` mark used by priors.  Optional `label` values are ignored by
  inference.  The legacy `candidate_pairs` edge-list alias is still accepted but
  documented as legacy.
* `selection_inputs.h5`: consolidated selection data with root
  `format_version = "lensing-selection-inputs-1.0"` and `unlensed`/`lensed`
  groups.  During transition, legacy `gwcat-selection-1.0` singleton-selection
  files and lensed-injection files with `p_tag_per_source` metadata are accepted
  by validators with warnings.  If pair-tag selection is used, `p_tag` fields
  must be present and finite in the lensed selection component.
* `run_config.yaml` or `run_config.json`: the resolved simulation/inference
  configuration.  New files should use `format_version =
  "lensing-run-config-1.0"`; existing resolved simulation configs without this
  field validate with a compatibility warning.

Validate a complete set with:

```bash
python scripts/mock_lensing/validate_lensing_file_contract.py \
  --gw_path observed_gw_pe.h5 \
  --observed_catalog_path observed_catalog.json \
  --candidate_pairs_path candidate_pairs.json \
  --selection_path selection_inputs.h5 \
  --config run_config.yaml \
  --out validation_report.json
```

Preflight uses the same validators and reports file-format versions, event
counts, candidate-pair counts, selection summaries, and compatibility warnings.
Simulation output includes both current CLI filenames and the contract aliases so
future real-data pipelines can satisfy the same interface without changing the
inference command semantics.

## Simulated candidate-graph audit

Simulated studies write a cheap candidate-graph audit before likelihood
compilation or sampling.  The audit validates the graph that was produced from
observed-event quantities and then, only after graph construction, reads the
simulation truth fields in `observed_catalog.json` to evaluate whether the graph
still contains the injected lensed-image edges.  These truth fields are not added
to `candidate_pairs.json` and are not inference inputs.

For each generated case the runner writes
`cases/<case>/candidate_graph_audit.json`.  The top-level study directory also
contains `candidate_graph_audit.csv` with one row per case.  The report includes
basic graph size, the number of injected true edges, how many true edges survived
candidate pruning, the number of false candidate edges, maximum observed degree,
connected-component sizes and edge counts, approximate exact-matching partition
counts per component, available mark keys, mark summaries, and true-versus-false
summaries for available marks and edge prior odds.

True-edge survival is a prerequisite for recovery: if the candidate builder
prunes an injected true edge, downstream partition enumeration and sampling have
no state that can recover that true pair.  The audit therefore runs in
`--preflight_only` and normal/diagnostics workflows immediately after the
candidate graph is built and before the preflight inference command.  `--dry_run`
still writes only planned commands because observed catalogs and candidate files
do not exist yet.

By default the audit is informational, which keeps stress tests that intentionally
prune hard graphs from failing early.  For debugging candidate-builder settings,
run the simulated study with `--require_true_edge_survival true`; any case with
at least one missing injected true edge will stop before preflight/inference.
