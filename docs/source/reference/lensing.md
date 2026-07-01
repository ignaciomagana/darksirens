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

### Marginalized partition diagnostics

For spectral-siren lensing runs with `--cluster_mode j2`, fixed-partition diagnostics describe exactly one materialized partition: the `singleton_indices` and `pair_indices` supplied by `--partition_path`.  Fixed partitions use pair-ordinal time-delay metadata from `pair_pe`.  In contrast, `--partition_mode marginalize_exact` enumerates every compatible matching from `--candidate_pairs_path`, evaluates the scalar likelihood for each partition, and writes diagnostics marginalized over the posterior partition weights.  Marginalized `--pair_marks time` uses edge-indexed candidate marks rather than pair-ordinal `pair_pe` metadata.

A marginalized `diagnostics.json` includes the prior normalizer (`log_z_partition_prior`), marginalized scalar likelihood (`logL_marginalized`, also aliased as `logL_total`), per-partition prior weights and likelihoods, posterior partition probabilities, posterior pair probabilities in the validated candidate-pair order, the posterior expected numbers of pairs and singletons, and a compact MAP partition object.  Candidate pairs are treated as unordered edges; if an input pair is written as `(j, i)`, the validated diagnostics report the normalized order `(min(i, j), max(i, j))` while preserving optional `label` and `log_prior_odds` fields.

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
