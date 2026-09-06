# Performance

This page answers one question: how do you make a `darksirens` run fit in the
memory you have and finish in the time you have. Every flag below is in
`darksirens_inference --help` or `darksirens_inference_lensing --help`
(see [CLI reference](../reference/cli.md)); every number states the
configuration it was measured on.

## What one likelihood call costs

`scripts/benchmarks/bench_likelihood_call.py` builds the likelihood with the
CLI's own option resolution, data load, parameter space and factory, then times
the compiled callable on a fixed set of prior draws and records the
log-likelihood values. Two builds can therefore be compared for speed and for
value drift with one command each:

```bash
python scripts/benchmarks/bench_likelihood_call.py --tag base --out base.json -- \
    --gw_path gw_events.h5 \
    --gwselection_path gw_selection.h5 \
    --survey_path catalog_nside_32.h5 \
    --universe_model dark_sirens \
    --pop_model powerlaw+peak \
    --sampler dynesty \
    --fix_population true --fix_survey true --fix_de true \
    --fixed_parameter_values '{"Om0": 0.3075}' \
    --save_path /tmp/bench_run

python scripts/benchmarks/bench_likelihood_call.py --tag cand --compare base.json -- ...
```

Everything after `--` goes verbatim to the `darksirens_inference` parser.
Timings are wall-clock medians over `--n-calls` warm calls (default 20) after
one compile call and three warm-ups; `--compare` prints the speedup and the
max `|dlogL|` against the recorded run, and checks the coordinates match.

`--components` additionally times the per-proposal state build, the
population/Jacobian kernel, the redshift-prior evaluation and the catalog KDE
on the PE and selection sets as isolated jits, with the prepared state passed
as an argument so a consumer is timed without its producer.

```{note}
Report the backend. The script prints `backend=` from `jax.default_backend()`,
and the profiles differ qualitatively: the windowed catalog KDE is gather-bound
on a GPU and `exp`-bound on a CPU. On a CPU box also pass explicit
`--sel_batch_size` / `--pe_event_block` and `--row_chunk 256`, because the auto
block plan is a single pass on non-GPU backends.
```

## Memory: the block-size knobs

| Flag | What it chunks | Default |
|---|---|---|
| `--sel_batch_size N\|auto\|off` | injections per selection-integral chunk | `auto` |
| `--pe_event_block N\|auto\|off` | events per PE-reduction chunk | `auto` |
| `--row_chunk auto\|off\|N` | catalog rows per kernel-state build chunk (`lax.map` instead of one full vmap) | `auto` |
| `--drop_full_catalog BOOL` | discards the dense full-sky `(npix, n_max_gals)` galaxy arrays after compacting to inference pixels | `false` |

For the first two, `off` / `none` / `0` forces a single pass (fastest, widest
working set) and a positive integer pins the block. `auto` sizes the block from
probed free device memory after the data load; the resolved values and a
provenance tag (`block_size_resolution`: `explicit`, `cpu`, `auto-single-pass`,
`auto`, `auto-floor-reduced`) are printed and persisted to `settings.json`.
The planner lives in `darksirens.likelihood.block_sizing`; its floors are
`SEL_MIN_BATCH = 32768` and `PE_MIN_BLOCK = 8` and it spends
`SAFETY_FACTOR = 0.7` of the probed free memory.

```{warning}
`auto` blocks only on a GPU-class backend. On any other backend
`resolve_block_sizes` returns a single pass with the tag `cpu`, and host RAM is
never probed, so a dense dark-siren catalog can exhaust it. Pin
`--sel_batch_size`, `--pe_event_block` and `--row_chunk` explicitly for CPU
runs.
```

`--drop_full_catalog true` cuts startup device memory at large `nside`. It is
incompatible with `--use_lss` and with the bright-siren model, which read the
full-sky rows.

Measured on a DESI-scale slice on a CPU, a forced `--row_chunk 512` was 5.1x
faster and 1.7x lighter in peak RSS than the `auto` default; for CPU-heavy
catalog work that is the first knob to try.

## The catalog KDE window

The dark-siren redshift prior evaluates, for every PE sample and every
injection, a Gaussian mixture over the galaxies of the sample's pixel row
(rows are z-sorted at load). Only the `W` galaxies nearest the sample's
redshift are evaluated.

| Flag | Meaning |
|---|---|
| `--kde_window W` | pins `W`; `0` disables windowing (full row) |
| `--kde_window_nsigma X` | build-time sizing multiplier (default 8; kernel mass beyond it is below `exp(-32)` per galaxy) |

Unset, `W` is sized from the data by
`darksirens.redshift.catalog.auto_kde_window`: the largest in-range galaxy
block any bound row holds at the widest `sigma_kde` the run can reach (its
fixed value, or the prior's upper bound when it is sampled), so the evaluator
never truncates. The window is centred on the sample by index and not
repositioned, so a pinned `W` below the data-sized one truncates the catalog
prior and `_resolve_kde_window` warns at build. A second warning fires when the
data-sized window exceeds 1024, because a dense photo-z catalog then pays for
the exact answer.

Measured on a DESI-like mixed spectroscopic-plus-photometric catalog (73% of
widths in `[0.02, 0.10]`, 2158 galaxies per row), a pinned `W = 1024` moved
`log p_cat` by 0.17 nats on average and 0.36 at worst against the full-row
evaluator, coherently in one direction. Measured on the 4-core CPU mock
(nside-16, 3072 rows by 2158 galaxies, 64 events by 4096 samples, 839k
injections, `H0` the only sampled label) the windowed evaluator plus the
pixel-sorted injection order runs 5.01 s/call against 7.07 s/call (1.41x) with
log-likelihoods bit-identical at five prior draws.

### The one-pass mixture reduction

The mixture itself is reduced in ONE pass over the row. A `logsumexp` is two
reductions -- a maximum, then the exponential sum -- and the gathered
per-galaxy exponent cannot be produced once and consumed twice inside one XLA
fusion, so the compiler materialises it: on the production run that was an
`f64[1, 1060864, 1719]` intermediate, 14.6 GB per sample set, written by one
kernel and re-read by the next, which also re-gathered the fused weights.

`CatalogKernelState` therefore carries the row's own maximum
(`log_kw_eff_rowmax`, a build-time constant of the catalog) and `1 / sigma_eff`
(`inv_sig_eff`, theta-invariant once `sigma_kde` is fixed). Subtracting the
build-time maximum makes every exponent non-positive by construction, so the
first reduction is not needed and the whole evaluation collapses into one
register-resident fusion; the reciprocal removes the per-galaxy division that
was forcing XLA to break that fusion in the first place. Both are needed: on an
H100 NVL either one alone is worth about 6 ms and leaves the intermediate in
place. A state without the two leaves (a hand-built one) falls back to the
`logsumexp` form. The windowed evaluator shares the full-row offset (a maximum
over a subset is no larger, so the offset is still valid); measured on the real
DESI catalog a window gives away at most 12 nats of the offset budget below,
and the production run never windows at all (`W` 3456 exceeds the 1719-galaxy
rows).

Measured on the full production likelihood (259 events, DESI nside-64, field
weighting, `zspace`-24 `n_sigma` 6, `sigma_kde` pinned, auto blocking) on an
H100 NVL, median of 20 warm calls: 32.2 ms/call against 59.2 ms (1.84x), with
the log-likelihood bit-identical at eight prior draws spanning `H0` in
`[20, 140]` (the five finite ones agree to the last bit and the same three are
`-inf`). Peak device memory, read as `peak_bytes_in_use` in the benchmark
process after those eight draws, falls from 22.6-24.3 GB to 12.2-13.1 GB
(-45.9% on interleaved same-slot arms; the absolute number moves with how many
calls preceded the read, the ratio does not). The spectral configuration has no
catalog KDE and is unchanged (13.9 against 14.0 ms, bit-identical).

The two leaves are BUILD-time, not per-call. Under the `H0` kernel pin (the
configuration benchmarked above) they are lifted straight off the pinned
quadrature and the only per-proposal work is one `(N_rows,)` add. When the pin
is inadmissible -- any run that samples `Om0`, `w0`, `wa`, `delta` or
`sigma_kde`, see `kernel_pin_admissible` -- `catalog_kernel_state` rebuilds
both leaves per proposal: a `(49152, 1719)` f64 reciprocal (676 MB) and a
row reduction over the same array, on top of the fused weights it already
built. That extra build work is bought back many times over by the same call:
measured on an H100 NVL with `sigma_kde` moved out of the fixed set (21 sampled
labels, everything else as above, median of 10 warm calls, interleaved
same-slot arms), 344.7 ms/call before and 314.4 ms/call after -- 1.10x, with
the log-likelihood bit-identical at all eight prior draws (seven finite, the
same one `-inf`). The per-call cost there is dominated by rebuilding the whole
kernel state, which is why the fraction saved is smaller than under the pin.

The build-time offset costs one thing: a finite OFFSET BUDGET below the row's
maximum, set by where `exp` underflows on the backend. The XLA CPU backend
flushes subnormal `exp` results to zero, so the budget there is
`ln(smallest normal f64) = 708.40` nats; CUDA keeps subnormals and floors at
`ln(smallest subnormal) = 744.44`. The transition is graded, not binary. Just
above the edge the value is still finite but no longer exact: measured on an
H100 NVL with a one-galaxy row, the relative drift against the two-pass form is
4e-15 at 720 nats of deficit, 2e-12 at 725, and 3.4e-4 (0.25 nats) at 744
before `-inf` from 745. On the CPU backend the same band appears by a different
mechanism -- the offset is the ROW's maximum, not the sample's, so whole kernel
terms flush to zero -- and costs 5.9 nats on a finite value at 707.5-708.2 nats
of deficit on a synthetic 1001-galaxy row whose weights span 1 nat.

Nothing downstream can see any of it: a sample 708 nats under its row's peak
carries weight `exp(-708) = 0` in the `logaddexp` the caller weighs it in,
whatever value it returns. In the production configuration the set is empty by
a wide margin -- the deepest headroom actually used below `z_depth = 0.3` is
178 nats of the 708 available, measured over 51,959 PE samples and 52,594
injections at five values of `H0`. A run on a catalog with no `z_depth`
attribute and no `--survey_z_depth` enters the band for at most 2e-3 of
samples, monotonically more of them as `H0` rises.

### Routing the empty catalog rows

A sample whose pixel row holds no galaxies pays the whole per-sample KDE for a
value the evaluator throws away. `row_empty[pix]` selects an exact `-inf` for
`log p_cat` and `log_Nobs[pix]` is `-inf` too, so the prior collapses to the
missing-galaxy branch alone -- `logaddexp(-inf, log_miss)`, which returns
`log_miss` bit for bit. On the production DESI nside-64 catalog 18,682 of the
49,152 rows are empty, and 41.7% of the PE samples and 38.9% of the injections
land in one.

A sample's pixel is data: it never depends on a proposal. So the factory
partitions each sample set once, at build time, into "my row holds galaxies"
and "my row is empty" (`EmptyRowRouting` in `darksirens/redshift/prior.py`), and
the evaluator runs the unchanged KDE on the first group and the KDE-free
expression on the second. It then gathers the concatenation back through the
stored inverse permutation, so the vector it RETURNS is in the caller's sample
order: the per-event `(nEvents, nsamp)` reshape and the selection `logsumexp`
see the same values in the same slots, and no other per-sample array has to
move. The plan is built only for `dark_sirens` without a frozen prior, only on
a catalog that carries `ngals` (the row-prefix invariant that makes
`ngals == 0` imply `row_empty`), and only when the empty group is at least a
tenth of the set. A `--sel_batch_size` pin batches (and pads) the injections
and a `--pe_event_block` chunk splits the PE samples, so neither evaluator can
be handed a whole-sample-set plan: that side is refused at build time and never
uploads its index arrays, and the run keeps today's speed and today's bits.

Both halves of the premise are re-checked IN THE GRAPH on every call, and a
violation of either drives the whole log-likelihood to `-inf`. The row half --
every row the plan calls empty really does produce `-inf` -- is one `(N_rows,)`
reduce. The sample half -- every sample the plan routed really does sit on one
of those rows -- is one gather over the routed group, and it is the one that
binds the plan to the pixel vector the evaluator is actually holding (the plan
is selected by sample-vector length, and the factory derives it from the
catalog view's `sample_to_unique_idx` while the evaluator consumes
`gw.pixels[:, k]`). The verdict is `-inf` rather than NaN by necessity: every
per-sample NaN in this likelihood dies at the `isfinite` mask before it can
reach a reduction, so `-inf` is the only verdict that can reach the caller.

Measured on the same production likelihood and the same eight prior draws as
the one-pass reduction above, on an H100 NVL, median of 20 warm calls, three
launches per arm interleaved in one session: 28.4 ms/call against 32.6 ms with
the routing off (1.15x), and 2.09x against the 59.2 ms `f770956` baseline. Peak
device memory is 13.06 GB against 13.14 GB -- the plan costs `2N` `int32` per
sample set (`idx_occ` and `idx_empty` partition the set, plus `inv_order`), 17.0
MB across both, and the narrower KDE intermediate more than pays for it. The
spectral configuration has no catalog KDE and cannot reach this path.

On accuracy the routing is exact where it can be and ulp-level where it cannot.
The prior vector it returns is bit-identical to the plain path sample by sample
and slot by slot, and on the production build the whole log-likelihood is too:
`max |dlogL| = 0.000e+00` at the benchmark's five finite prior draws (the same
three are `-inf`) and at nine H0 values spanning [20, 140] with every other
label at its fiducial. What is NOT guaranteed is the last bit of the total,
because splitting one `vmap` into two changes how XLA fuses and orders the
reductions ABOVE the evaluator: on the CPU backend a small minority of
coordinates move by <= 4e-14 relative (1-2 in a 49-point H0 scan, no coherent
sign, twelve orders below the 0.3-nat bar this campaign works to). The gate in
`tests/test_empty_row_routing.py` is that dense scan with a `1e-12` relative
bound and a signed-mean check, plus an exact equality on the prior vector
itself.

## The frozen redshift prior (population-only runs)

`--freeze_redshift_prior BOOL` (default true) evaluates the per-sample catalog
redshift prior once at build time, so the per-proposal kernel state, completion
curves and windowed KDE over every PE sample and injection are spent once
instead of once per proposal.

`frozen_prior_admissible` in `darksirens/likelihood/factory.py` is the gate, and
it reads the sampled LABELS, never their values: the prior is frozen only for
`dark_sirens` / `dark_sirens_complete` with `--mark_model none`, no LSS-member
marginalization, and every sampled label a population label, a sky-model label
or a mixture stick `fcat_k`. Any cosmology or survey label, `H0` included,
blocks it. The premise is re-verified in the graph on every call: the fixed
cosmology and survey scalars the prior reads are compared exactly against the
values it was built at, and a mismatch returns `-inf`.
`--freeze_redshift_prior false` is the A/B switch.

Measured on the 4-core CPU mock (3072 rows by 2158 galaxies, 64 by 4096 PE
samples, 839k injections, `powerlaw+peak` fully sampled, cosmology and survey
fixed): 3.19 s/call against 7.38 s/call (2.3x), log-likelihood bit-identical at
the fiducial population.

## The per-galaxy kernel quadrature

The per-galaxy kernel normalisation `Z_i` is a Gauss-Legendre quadrature
configured by `darksirens.redshift.catalog.configure_kernel_quadrature`.

| Flag | Default | Notes |
|---|---|---|
| `--kernel_gl_nodes N` | 24 | safe for any kernel width; do not reduce for broad photo-z kernels |
| `--kernel_gl_domain {cdf,zspace}` | `cdf` | `zspace` integrates directly in redshift and avoids `ndtri` |
| `--kernel_gl_nsigma X` | 5.0 (module default) | half-width of the z-space window in units of `sigma_eff`; `zspace` only |

Measured on the full production likelihood (259 events, DESI nside-64, field
weighting) on an H100: the production setting is `zspace` with 24 nodes and
`--kernel_gl_nsigma 6` (357 ms/call, 1.8x faster than `cdf`, the most accurate
measured); the speed option is 16 nodes at `n_sigma` 5 (214 ms, 3.0x, residual
at most 0.3 nats against the 24/6 reference). Do not use 8 nodes at `n_sigma` 5
(about 9e-3 per galaxy, 130x worse than `cdf` at 24 nodes), and do not use
`n_sigma` 4, which pins a -8e-5 truncation bias no node count removes.

`kernel_pin_admissible` (same file) decides whether the quadrature is built
once at `KERNEL_PIN_H0_REF` instead of once per proposal. It requires
`--universe_model dark_sirens`, `--mark_model none`, and that none of `Om0`,
`w0`, `wa`, `delta`, `sigma_kde` (nor their per-catalog `_c<k>` variants) is
sampled. When the pin holds, the traced graph replaces the quadrature with the
scalar shift `3 ln(H0/H0_ref)` behind its own in-graph probe. When it does not
hold, the norms are rebuilt every proposal: measured on the 4-core CPU mock
with `H0` and `Om0` sampled, 19.2 s/call of which 13.9 s of the remaining
18.6 s (75%, by ablation) is that quadrature.

## Lensing

Under `--partition_mode marginalize_exact` the lensing CLI makes ONE call to
the cluster master likelihood per proposal (every event as a singleton row,
every candidate edge as a pair row, the pair rows through the master's
`lax.scan`) and assembles each partition from the returned per-row terms as
gathers and sums plus the marked-Poisson selection correction at that
partition's own counts and total-variance budget (`_assemble_partition` in
`darksirens/cli/inference_lensing.py`).

Measured on the 20-event by 250-sample mock with a 6-edge candidate graph
(30 partitions, 4-core CPU): 1 master evaluation per proposal instead of 8,
15.7 s against 38.7 s per `loglike`, and the value bit-identical
(`-174.60377127321635`) on the componentwise and the global path.

| Flag | Default | Effect |
|---|---|---|
| `--y_nodes_pair N` | 32 | Gauss-Legendre `y` nodes per J=2 pair likelihood |
| `--pe_max_per_pair N` | 400 | PE down-sampling per pair image (`0` keeps all); controls the `O(N_pe^2 N_y)` pair-KDE memory |
| `--pair_batch_size N` | 0 | candidate-pair batch size for J=2 scans (`0` keeps the unbatched path) |

`_check_pair_y_quadrature` re-evaluates the diagnostics total at four times
`--y_nodes_pair` and warns when `|delta logL|` exceeds 1e-3 nats. The SIS pair
integrand is peaked and its convergence is pair-dependent, so raise
`--y_nodes_pair` when that delta is comparable to the evidence difference being
measured.

## GPU notes

`darksirens.core.jax_config.configure_jax_runtime`, called at CLI import, sets
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_ALLOCATOR=default`
(XLA's BFC allocator) and then enables `jax_enable_x64` and
`jax_default_matmul_precision="highest"`. Both environment variables are set
with `setdefault`, so an explicit export in a job script wins.

Preallocation stays off because the block-size planner sizes its blocks against
the device's free memory, which a preallocating allocator reports as fully
consumed. Under an explicit `XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
`memory_stats()` is inert and the probe falls back to `nvidia-smi` free memory;
measured on the real spectral likelihood (1,067,946 injections, 259 events,
dynesty value-only, `off:off` blocking, 20 warm repetitions, clean process on
an H100 NVL), that allocator cost 23.0 ms/call against BFC's 13.7 ms (1.68x).

For an OOM that survives all of the above, see
[Troubleshooting](troubleshooting.md).
