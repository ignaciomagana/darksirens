# Lensed sirens

How to run `darksirens_inference_lensing`: spectral sirens with weak lensing
marginalised, the optional J=2 strong-lensing pair channel, and the two-arm
evidence comparison the CLI exists to produce.

## What it computes

The likelihood is a spectral-siren hierarchical fit whose per-event PE integral
is marginalised over the weak-lensing magnification $\mu$, with
$d_L^{\rm obs} = d_L(z_s)/\sqrt{\mu}$ and the Jacobian carried exactly
(`darksirens.likelihood.wl_weight`). `--cluster_mode j2` adds the strong-lensing
pair channel: candidate pairs of multiply-imaged sirens scored with the SIS pair
likelihood against a lensed-injection cluster selection term, combined with the
singleton terms by the marked-Poisson correction
(`darksirens.likelihood.likelihood_with_clusters`). This CLI owns the
`spectral_sirens_wl` universe model and implements no galaxy-catalog dark
sirens, LSS completion or catalog host probabilities; use `darksirens_inference`
for those ([Galaxy catalogs](catalogs.md)).

The measurement is an **evidence difference**: two arms on the same data, one
`--cluster_mode j2` and one `--cluster_mode off`, identical otherwise. Both
archive `logZ` and `logZerr` in `results.hdf5` and `settings.json`, and the pair
channel's support for lensing is their difference.

## Inputs

| Flag | File |
| --- | --- |
| `--gw_path` | gwcat PE posterior samples for the observed events |
| `--gwselection_path` | gwcat unlensed selection/injection campaign |
| `--observed_catalog_path` | unified `observed_catalog.json` (the observed-mode entry point) |
| `--pair_metadata_path` | pair / candidate-edge metadata for the observed events |
| `--lensed_injections_path` | lensed J=2 injections (required by `j2` and by `sl_mixture`) |
| `--candidate_pairs_path` | candidate-edge graph for `--partition_mode marginalize_exact` |
| `--partition_path` | one fixed image assignment for `--partition_mode fixed` |

Schemas and dataset shapes are in [Input files](../getting-started/inputs.md).
`--pair_pe_path` is the deprecated split-pair layout, unusable by preflight's
event-index range check; use the unified observed catalog plus
`--pair_metadata_path`. `--preflight_only true` runs the input preflight checks,
writes the JSON (`--preflight_json`, else `<save_path>/preflight.json`) and exits
before compilation; run it before every long submission.

## Weak-lensing backends

`--wl_backend lognormal` (the default) uses $\ln\mu \sim \mathcal N(-s^2/2,
s^2)$ with $s^2(z) = a\,z^b$ from `--lensing_wl_a` (default `4e-3`) and
`--lensing_wl_b` (default `1.5`), integrated by Gauss-Hermite quadrature with
the mean fixed so $\langle\mu\rangle = 1$; the defaults match the Takahashi et
al. (2011) and Hilbert et al. (2011) ray-tracing PDFs for BBH-relevant source
redshifts (`darksirens.lensing.wlmagnification`). `tabulated` interpolates an
external $\log p_{\rm WL}(\mu\mid z)$ from `--lensing_wl_table_path` (HDF5
datasets `z_grid`, `log_mu_grid`, `log_p_table`); `disabled` turns it off.

`--wl_selection` sets the treatment on the **selection** side and must match the
event side, or the hierarchy is normalised under a different observation model
than its numerator. `auto` (the default) resolves `lognormal -> wl_lognormal`
and `disabled -> standard`, and refuses `tabulated`, which has no matched
selection integral here. An explicit mismatched pair is fatal unless
`--allow_mismatched_wl_selection` records a deliberate ablation, which biases the
posterior and is stamped in `settings.json` beside the pre-resolution request.
`--lensing_wl_a 0` collapses the kernel to a delta, so `standard` is then
identical, not mismatched.

Both backends are validated eagerly at startup. `validate_wl_hermite_quadrature`
compares the production 16-node importance-ratio rule against a dense reference
across the $z$ grid and hard-fails on disagreement, since the rule converges in
node count only near the calibrated amplitude while `--lensing_wl_a` is an
unbounded float; a tabulated table is checked for quadrature coverage.

## The pair channel

`--cluster_mode j2` (the default) turns the pair channel on; `off` is the
singleton-only control arm. `--partition_mode fixed` (the default) evaluates the
one explicit image assignment in `--partition_path` (the mock generator's
`partition.json` is the truth partition). `--partition_mode marginalize_exact`
enumerates the compatible partitions of the candidate graph in
`--candidate_pairs_path`, weights each by its edge `log_prior_odds` and
log-sum-exp marginalises over them, that is over which events really are images
of one another.

`--partition_component_mode componentwise` (the default) factorises that sum over
the graph's disconnected components, which is what keeps it tractable; `global`
enumerates the whole graph at once. The enumeration is bounded by
`--max_exact_partitions` (default `10000`), `--max_component_events`,
`--max_component_edges`, `--max_component_partitions` and
`--max_total_partitions`; a graph exceeding a cap is refused with a request to
prune it. Cost per proposal is **one master-likelihood evaluation** scoring every
event as a singleton row and every candidate edge as a pair row; each partition
is then gathers and sums of those rows plus the marked-Poisson selection
correction at its own counts and variance budget, so the partition count does not
multiply the likelihood cost (`_assemble_partition`).

## Time marks

`--pair_marks time` adds the SIS arrival-time mark to the pair likelihood: in
`marginalize_exact` mode read per candidate edge from `candidate_pairs.json`
(`marks.delta_t_obs` and a positive finite `marks.sigma_delta_t` on every
selectable edge), in `fixed` mode from the pair metadata, with
`--pair_time_sigma_sec` as the fallback width. `--pair_time_mark_impl auto` (the
default) delta-collapses the $y$-integral when `max(sigma_dt)/T0 < 0.02`, marks
that sharp being unresolvable by quadrature; `quadrature`/`delta` force a path.

**Sign orientation.** For SIS the type-I minimum arrives before the type-II
saddle, so the two image assignments predict opposite signs of the delay and at
most one fits the data. Recorded marks are magnitudes, so the CLI restores the
sign from the observed catalog's arrival times, leaving magnitude and width
untouched. A non-finite mark, or one whose magnitude does not reproduce the
catalog's arrival separation, is fatal; `--allow_time_mark_mismatch true`
downgrades both to warnings and `--allow_suspicious_time_marks true` downgrades
the placeholder/synthetic time-mark error.

**Observing window.** The mark needs the run length $T$: the observed catalog
must declare `observation_times = "uniform"` with `t_obs_days`, and every mark
must satisfy $|\Delta t| < T$, both enforced at startup. The unlensed
coincidence density vanishes at $|\Delta t| = T$, so a mark at or beyond the
window (a mis-set `t_obs_days`, or a cross-run time base) would reward every
such pair. Separately, pairs with $|\Delta t| \ge$ `--sl_T0_sec` fall outside
the SIS support $y \in (0,1)$ and get an exactly $-\infty$ pair likelihood.

## Pair-tag models

`--pair_tag_model` sets the probability that a truly imaged, both-detected pair
is identified as a candidate: `constant` (the default, with
`--pair_tag_constant`, default `1.0`), the deterministic mock scores `snr_only`,
`snr_sky`, `snr_time`, `snr_time_sky`, or `file` with
`--pair_tag_selection_path`, offset in logit space by
`--pair_tag_perturb_logit`. It enters the J=2 cluster selection integral, not
the per-pair PE likelihood.

```{warning}
A resolved $p_{\rm tag} = 1$ for every kept source is the **both-detected
approximation**: an upper bound on the true pair-detection probability, so it
overestimates $\mu_{\rm sel}^{(2)}$ and biases the inferred optical-depth
parameters low. A run inferring lensing rates (`--fix_lens_rate false`) under it
is refused unless `--allow_both_detected_approx true` acknowledges the bias,
which is then stamped into `settings.json` and `results.hdf5`; the alternatives
are a calibrated efficiency (`--pair_tag_model file`) or a fixed rate.
```

## Lens-rate sampling

`--fix_lens_rate true` (the default) pins the SIS optical depth to `--sl_tau_A`
(default `5e-4`) and `--sl_tau_n` (default `3.0`). `--fix_lens_rate false`
samples `log10_tau_A` and `tau_n`; `--lens_prior_overrides '{"log10_tau_A":
[-5.0, -2.5]}'` sets their bounds, and naming one in `--fixed_parameter_values`
holds it fixed while the other samples. `--sl_T0_sec` is the time-delay scale in
$\Delta t = T_0 y$ (default `5.36e+06` s, about 62 d: the SIS scale at $z_L =
0.5$, $z_s = 1$, $\sigma_v = 200$ km/s under this repo's cosmology).
`--fix_cosmology false` is refused whenever a lensed-injection channel is on
(`--cluster_mode j2` or `--singleton_lensing sl_mixture`): those selection terms
are valid only at the campaign's fiducial cosmology.

`--singleton_lensing sl_mixture` replaces the legacy drop-single-image protocol
with a mixture in which observed singletons are unlensed sources plus strongly
lensed sources with exactly one detected image (evidence mixture,
exactly-one-detected selection subset, analytic Finn-Chernoff partner
censoring). It requires `--lensed_injections_path` and a mock generated with
`--include-lensed-singletons true`; `--y_nodes_single` (default `32`) sets its
$y$ nodes, `--fc_rho_thr`/`--fc_r0`/`--fc_mc_bar` override the injection file's
Finn-Chernoff attributes, and `--pair_orientation_mode` must match the
campaign's rendering convention.

## Startup diagnostics

Every run writes `settings.json`, `midpoint.json`, `midpoint_diagnostics.json`,
`diagnostics.json`, `diagnostics.hdf5` and `results.hdf5` into its timestamped
run directory, plus `failure.json` if it dies. The diagnostics are evaluated
before sampling at a guard-clear point (point and label travel with the numbers)
and carry `logL_total`, the
selection correction, the singleton and pair log-likelihood sums, the singleton,
cluster and combined `Neff`, the observed counts, the cluster and WL modes and
the batching and quadrature settings: the fastest way to confirm that `off` is
singleton-only, that `j2` sees the expected pairs and that exact marginalisation
found the intended partitions. The sampler log-likelihood is cross-checked
against them at that point, and a disagreement aborts the run.

`midpoint_diagnostics.json` also carries `pair_y_quadrature_check`, which
re-evaluates the total with four times `--y_nodes_pair` (default `32`) and
records `abs_delta_logL`, a tolerance of `1e-3` nats and a `converged` flag.
The SIS pair integrand is peaked and its Gauss-Legendre convergence is
pair-dependent, so a delta comparable to the evidence difference being measured
means the quadrature, not the physics, is setting the answer: raise
`--y_nodes_pair`. Delta-collapsed time-marked pairs do not read it.

## A two-arm example

Generate a mock with a unified observed catalog, a candidate graph and physical
arrival times:

```bash
python scripts/mock_lensing/generate_mock_lensing.py \
  --outdir data/mock_lensing --conditioning poisson_counts \
  --n-universe 2000 --max-sing-keep 10 --max-pair-keep 5 --nsamp 256 \
  --n-unlensed-inj 1000 --n-lensed-inj 1000 --candidate-time-marks true \
  --write-unified-observed-catalog true --observation-times uniform \
  --t-obs-days 365 --seed 2026
```

Run both arms on it, sharing the singleton configuration:

```bash
COMMON=(
  --gw_path data/mock_lensing/mock_observed_gw_pe.h5
  --observed_catalog_path data/mock_lensing/observed_catalog.json
  --gwselection_path data/mock_lensing/mock_gw_selection.h5
  --wl_backend lognormal --wl_selection auto
  --pop_model powerlaw+peak
  --fix_cosmology true --fix_survey true --fix_population true
  --fix_lens_rate true
  --sampler dynesty --nlive 500 --dlogz 0.1 --max_samples 0
  --checkpoint_interval 1800 --resume auto
  --seed 2026
)

# control arm: no pair channel
darksirens_inference_lensing "${COMMON[@]}" \
  --cluster_mode off --save_path runs/lensing/off

# J=2 arm: candidate pairs marginalised over partitions
darksirens_inference_lensing "${COMMON[@]}" \
  --cluster_mode j2 \
  --lensed_injections_path data/mock_lensing/mock_lensed_injections.h5 \
  --pair_metadata_path data/mock_lensing/mock_pair_metadata.h5 \
  --candidate_pairs_path data/mock_lensing/candidate_pairs.json \
  --partition_mode marginalize_exact --max_exact_partitions 10000 \
  --partition_component_mode componentwise \
  --pair_marks time --pair_time_mark_impl auto \
  --pe_max_per_pair 400 --y_nodes_pair 32 \
  --save_path runs/lensing/j2
```

`bash scripts/mock_lensing/run_tiny_lensing_validation.sh` runs the tiny version
of exactly this (mock, `off`, `j2` with a fixed partition, and with
`RUN_MARGINALIZE_EXACT=1` an exact-marginalisation run); see
[Testing](testing.md). Sampler, checkpoint and memory flags behave as on
`darksirens_inference` ([Running inference](inference.md),
[Performance](performance.md)); every option is in the
[CLI reference](../reference/cli.md).
