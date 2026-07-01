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
`--pair_marks time` to include the Gaussian time-delay mark in the J=2 likelihood;
`--pair_time_sigma_sec` is only a fallback when pair metadata has `delta_t_obs`
but omits `sigma_delta_t`.

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

For spectral-siren lensing runs with `--cluster_mode j2`, fixed-partition diagnostics describe exactly one materialized partition: the `singleton_indices` and `pair_indices` supplied by `--partition_path`.  In contrast, `--partition_mode marginalize_exact` enumerates every compatible matching from `--candidate_pairs_path`, evaluates the scalar likelihood for each partition, and writes diagnostics marginalized over the posterior partition weights.

A marginalized `diagnostics.json` includes the prior normalizer (`log_z_partition_prior`), marginalized scalar likelihood (`logL_marginalized`, also aliased as `logL_total`), per-partition prior weights and likelihoods, posterior partition probabilities, posterior pair probabilities in the validated candidate-pair order, the posterior expected numbers of pairs and singletons, and a compact MAP partition object.  Candidate pairs are treated as unordered edges; if an input pair is written as `(j, i)`, the validated diagnostics report the normalized order `(min(i, j), max(i, j))` while preserving optional `label` and `log_prior_odds` fields.

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
