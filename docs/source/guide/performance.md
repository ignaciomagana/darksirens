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

The sizing scan itself visits rows in descending galaxy-count order and stops
once the running maximum reaches twice the largest remaining count (a row of
`n` galaxies can contribute at most `2n`), and aliased catalog views — the
flat-union path binds the same arrays to the PE and selection views — are
scanned once. On the 259-event DESI nside-64 production run (49152 × 1719
rows, 30470 occupied, 22.8M galaxies; H100 NVL, jax 0.4.34) the whole sizing
step goes from 1.83 s to ~0.17 s, what is left being the device-to-host
transfer of the catalog arrays rather than the scan (which visits one row and
takes ~2 ms). The build phase follows it down by about 1.5 s: over three
interleaved launches per arm, 15.2 / 15.9 / 15.5 s before against
14.0 / 13.8 / 14.0 s after (medians 15.5 s → 14.0 s; the arms do not overlap,
and launch-to-launch build spread is a few tenths of a second). `W` is 3456 on
both arms, so the compiled graph and the likelihood are unchanged
(max |dlogL| 0.0 across the benchmark coordinates) and the per-call median is
flat at 58.6–59.2 ms.

Shape reads follow the same rule as the sizing scan: a build-time site that
needs only a galaxy table's row or column count reads `.shape` from the
`jax.Array` rather than `np.asarray(...)`, which runs the whole table through
numpy — 676 MB per table on that catalog, ~0.17 s of device-to-host copy.

What that buys is narrower than it looks, and the distinction is worth
keeping. `np.asarray` caches its host copy on the `jax.Array`, so converting a
shape-only reader that sits ahead of a site which genuinely consumes the same
array's values only MOVES the one download. Instrumenting every first host
materialisation of the production build shows exactly that: `full_z`'s 676 MB
moves from `build_field_normalization_inputs` to `build_field_depth_inputs`,
and the union `zgals`' 676 MB from `_spread_probe_rows` to
`_rows_sorted_for_windowing`. Both of those consumers need the values, so that
traffic is not recoverable without rewriting the reductions to run on device.
What is actually removed is `field_depth_z`'s 182 MB (nothing else reads it on
the host) and the 676 MB host-to-device re-upload that
`build_field_normalization_inputs` used to pay by rebinding `full_z` to its own
numpy copy — with it, one fewer live 676 MB device buffer at build peak.
Build-phase device-to-host traffic goes from 4259 MB in 161 transfers to
4076 MB in 160.

The wall-clock effect is correspondingly small. Over six interleaved launches
per arm on the same run and box, the build-phase median goes from 13.62 s to
13.50 s (base 13.43–13.68 s against 13.21–13.63 s, so the ranges overlap) and
load-plus-build-plus-first-call from 32.96 s to 32.86 s, which is inside the
launch-to-launch spread — read the total startup as unchanged. The per-call
median is flat at 58.9–59.0 ms and max |dlogL| is 0.0 on every benchmark
coordinate (no value is read, only a shape).

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

## Startup: the XLA compilation cache

A production dark-siren startup compiles ~350 XLA modules before the sampler's
first live point: ~100 building module-level constants at import, ~200 in the
eager build-time pin/KDE steps inside `make_likelihood`, and ~33 on the first
likelihood call. JAX can serve all of them from disk instead. Set
`DARKSIRENS_XLA_CACHE` to a local directory to opt in:

```bash
export DARKSIRENS_XLA_CACHE=$SCRATCH/darksirens-xla
```

Unset or empty (the default) leaves the cache off and nothing about the run
changes. Entries land in `$DARKSIRENS_XLA_CACHE/<host>-jaxlib<version>/`, one
subdirectory per host and JAX build, and the size is bounded by
`jax_compilation_cache_max_size` (2 GiB, set in
`darksirens.core.jax_config`; any value other than `-1` is also what puts a
`filelock` around the cache writes, which is what makes it safe for two jobs on
one box).

Measured on an H100 NVL with `scripts/benchmarks/bench_likelihood_call.py
--n-calls 20`, 259-event DESI nside-64 production configuration (H0 +
population + survey sampled, `zspace`-24/`ns6` quadrature, auto blocking),
five launches per arm across two sessions:
(the `load` column below predates the device row sort described in the next
section, which takes ~2.1-2.4 s off it in every arm on a GPU box; `build`,
`first call` and the call medians are unaffected)

| Arm | load | build | first call | XLA compiles | call median |
|---|---|---|---|---|---|
| cache off | 11.2-11.5 s | 15.3-15.9 s | 9.9-10.5 s | 347 | 59.4-59.8 ms |
| cold (writes the cache) | 11.4-11.5 s | 16.1 s | 10.0-10.1 s | 338 | 59.7-59.9 ms |
| warm | 10.9-11.2 s | 7.4-7.9 s | 1.8-2.0 s | 0 | 59.5-59.8 ms |

That is 37.2 s of setup against 20.7 s warm, 16.5 s saved per launch, for
676 entries / 4.3 MB on disk. The 259-event spectral configuration goes from
7.7 s of setup (162 compiles) to 4.1 s (0 compiles), 3.6 s saved, for
312 entries / 1.8 MB. The per-call medians of the two arms straddle each other
and the master baseline (59.2 ms): the cache cannot move per-call time, and the
range above is the process-to-process spread.

The COLD run pays for the writes, by a small but real margin: build 16.1 s
against cache-off builds of 15.3-15.9 s in the same sessions, and +1.7 s of
build (about +2 s of setup) in an independent reproduction on the same box.
Once `jax_compilation_cache_max_size` is set, `LRUCache._evict_if_needed` runs
on every write and rescans the directory, so the penalty grows with the entry
count. It is paid once per configuration, and it is the reason the cache root
should be a LOCAL directory: 338 writes each rescanning a growing directory on
a shared network filesystem would be considerably worse (nothing refuses such a
path, so this is a judgement call, not a guardrail).

Per-call time is untouched — the cache returns the same serialized executable
`backend_compile` produced, keyed on the HLO module plus the jaxlib version,
backend, compile options and device kind, so a mismatched entry cannot be
served. The log-likelihood at all eight benchmark draws is byte-equal warm
against cache-off, on both configurations. Any change to the graph simply
misses the cache and compiles.

So this is worth ~0.03% of a 1e6-call production dynesty wall and roughly half
the wall of a smoke run or a repeated benchmark launch.

`bench_likelihood_call.py` records `compiles_total` (real compilations),
`compile_requests_total` and `compile_requests_in_timed_loop` (executables JAX
asked for, compiled OR served from the cache) and `xla_cache_dir` in its
summary JSON, and prints them, so a benchmark JSON states which cache state it
was measured in. Startup phase timings are only comparable between arms with
the same `xla_cache_dir`, and no cache directory should ever be shared between
the two arms of a timed A/B.

`--recompile-guard` exits nonzero if anything is lowered inside the timed loop,
which is how a per-call shape or static-argument leak is caught as itself
rather than as a fat, noisy median. It counts compile REQUESTS, not
compilations, precisely because of the cache above: a leak whose modules are
already on disk never calls `backend_compile` and would otherwise be reported
as clean while still paying a lowering, a cache read and a deserialize on every
call (measured on a CPU probe: a four-shape loop is 6 compilations cold and 0
compilations / 6 compile requests warm). The summary JSON is written before the
guard exits, so a tripped run still leaves its evidence.

## Startup: the catalog row sort runs where the catalog lands

`load_survey` establishes the per-row z-sort invariant the windowed catalog
KDE depends on: one stable per-row permutation of the padded `(npix, maxgals)`
catalog, applied to `zgals` / `dzgals` / `wgals`. On the 259-event DESI
nside-64 catalog that is a 49,152 x 1,719 float64 key, an argsort over 84.5M
elements and three 676 MB gathers.

When the arrays are bound for the accelerator anyway
(`load_survey(..., to_device=True)`, the default and the production path) AND
there is an accelerator to run them on, the sort now runs there instead of on
one host core: the raw arrays go up first,
`jnp.argsort(..., stable=True)` builds the permutation as `int32`, and three
`jnp.take_along_axis` gathers apply it, one array at a time so the raw upload
of each is released as soon as its gather is done. The invariant is a device
reduction with one scalar transfer rather than a full-width `np.diff` on the
host. Callers that compact on the host before transferring
(`--drop_full_catalog`, the multitracer bundles, the selection-function fit)
pass `to_device=False` and keep the numpy implementation unchanged; the
callers that DO take the device path are the inference loaders and the offline
`darksirens_build_lognormal_completion` / `darksirens_diagnose_lognormal_completion`
table builders, which load the full catalog with the default `to_device=True`.

`to_device` answers "will the caller upload these arrays", not "is there an
accelerator to upload them to", so the dispatch also asks
`darksirens.catalogs.io.device_row_sort_admissible()`: a non-CPU backend and
x64 on. On a CPU-only install XLA-CPU's `argsort` is the SLOWER of the two --
measured on this box under `JAX_PLATFORMS=cpu`, x64 on, backend pre-warmed,
the full 49,152 x 1,719 production catalog: 2.74 s for numpy against 10.0 s
for XLA-CPU, i.e. +7.2 s per `load_survey` and +0.28 GiB of peak RSS -- so a
CPU-only run keeps the numpy implementation and costs 2.76 s, the same as
before. With x64 off `jnp` would build the key in float32 while
`load_survey_marks` still derives its permutation from a float64 numpy key,
which would silently decouple marks from galaxies; the gate falls back to the
host implementation there rather than raising on a call that used to work.

Both implementations run the same stable sort on the same `+inf`-padded key, so
they produce the same permutation, tie for tie, and therefore the same bytes:
verified with `np.array_equal` on the permutation and on all three sorted
arrays over the full 49,152 x 1,719 production catalog, and the log-likelihood
at all eight benchmark draws is byte-equal between the two arms.

Measured on an H100 NVL with `scripts/benchmarks/bench_likelihood_call.py
--n-calls 20`, 259-event DESI nside-64 production configuration (H0 +
population + survey sampled, `zspace`-24/`ns6` quadrature, auto blocking, no
XLA cache), three interleaved launches per arm, one arm per process:

| Arm | row sort | `load_survey` | load phase | call median |
|---|---|---|---|---|
| host (numpy) | 2.66-2.79 s | 7.91-8.46 s | 10.8-11.3 s | 58.8-59.1 ms |
| device | 1.09-1.15 s | 5.71-5.86 s | 8.5-8.7 s | 59.0-59.2 ms |

That is -2.3 s of the load phase on these medians, 21% of it. Three
independent reproductions on this box, different launch orders and different
prior draw sets, put the figure between -2.1 s and -2.5 s (19-23%); quote
-2.1 to -2.4 s. The device sort's own 1.1 s is not
the cost of the sort -- warm, the four device ops take 15 ms -- it is the CUDA
backend initialization and first-op compilation that the host arm instead pays
a few lines later on the first `jnp.asarray`. The right figure to quote is the
`load_survey` or load-phase delta, not the sort-level one.

Per-call time is untouched (the hot path never sees this code) and the
log-likelihood is byte-equal at all eight benchmark draws in every launch of
both arms, and so is the block-size plan: both arms report `auto-single-pass`
with `free 67.6 GiB` and the same 22.6 GiB peak device memory, because the transients of the sort are
dead long before the planner probes. The spectral configuration carries no
`--survey_path` and never reaches `load_survey`, so nothing about it changes.

## Startup: the catalog read decompresses its chunks in parallel

The 259-event DESI nside-64 catalog is 480 MB on disk and 2.03 GB in memory:
`zgals` / `dzgals` / `wgals` are 49,152 x 1,719 float64 tables stored in
(768, 27) chunks behind a shuffle + deflate filter pipeline, 4,096 chunks each.
Reading them is decompression-bound, not I/O-bound -- the raw chunk bytes come
off this Lustre filesystem in 0.46 s cold (1053 MB/s) and 0.09 s warm, while
`np.asarray(f['zgals'])` and its two siblings take 4.6 s with the page cache
warm and 5.5 s cold.

h5py cannot be parallelised: HDF5 serializes, and four threads on four
separate file handles measure 4.55-4.62 s against 4.59 s sequential. What does
parallelise is doing the inflate ourselves. `load_survey` reads its four
datasets through `darksirens.catalogs.io.read_dataset_chunked`, which pulls
each stored chunk with `dataset.id.read_direct_chunk` on the calling thread and
hands it to a small `ThreadPoolExecutor`, where `zlib.decompress` (which
releases the GIL) inflates it and the byte un-shuffle transposes it straight
into its disjoint slice of one preallocated array.

The pool is four workers, not more. Only the inflate is GIL-free: on
`zgals` (4,096 chunks) the per-chunk split is 1.51 s of `zlib` against 1.04 s of
un-shuffle plus 0.36 s of assemble, i.e. 48% of the work holds the GIL and the
Amdahl ceiling is 2.08x. Measured on this box, warm cache, median of three
reads of the four datasets: sequential 4.68 s, 2 workers 3.60 s, 4 workers
1.93 s, 6 workers 2.63 s, 8 workers 4.27 s, 16 workers 3.48 s. Past ~4 workers
GIL convoying eats the win, so raising the worker count is a regression, not a
bigger gain.

**The floor is not "degrades to sequential" -- below two usable CPUs the
threaded read is slower than the h5py read it replaces**, because the
un-shuffle and the assemble copy are then pure added Python with no inflate to
overlap against: one worker measures 6.41 s against 4.53 s sequential, a 1.4x
regression. So the pool is sized by `len(os.sched_getaffinity(0))`, not by
`os.cpu_count()` -- the two differ on any cpuset-restricted allocation (a
`--cpus-per-task=1` batch job still reports 64 host cores) -- and the whole
raw-chunk path is refused below two usable CPUs. Two is already a win
(`taskset -c 20,21`: 3.34 s against 4.42 s). A cgroup CPU *quota* is a
fractional limit rather than a mask and no affinity call can see it, so
`DARKSIRENS_CATALOG_CHUNKED_READ=0` forces the plain h5py read everywhere; the
arrays are the same bytes either way.

Host memory is bounded. The calling thread reads raw chunks about ten times
faster than the pool inflates them, so an unbounded submit loop would hold the
whole *compressed* dataset in the executor queue. The loop caps the in-flight
chunks at four per worker, which costs nothing measurable (2.01 s bounded
against 2.10 s unbounded, medians of three) and holds peak host RSS at the
sequential read's: 0.76-0.77 GB bounded, 0.78 GB sequential, 0.90-0.91 GB
unbounded.

The arrays are the same bytes, so nothing numeric changes. This is not a
tolerance claim: the result of `read_dataset_chunked` is byte-for-byte the
result of `np.asarray(dset)`, verified with `tobytes()` equality on all four
production datasets and on synthetic files exercising every edge (partial edge
chunks on both axes, 4- and 8-byte dtypes, 1-D datasets, 2/4/8 workers).

`read_dataset_chunked` reimplements exactly one filter pipeline -- shuffle,
then deflate -- so it refuses everything else and reads it with plain h5py
instead. The refusal tests the dataset creation property list for pipeline
*equality*, not for the presence of individual filters: `dataset.compression`,
`.shuffle`, `.fletcher32` and `.scaleoffset` each answer "is this filter in the
pipeline" and cannot see a third filter or the order, so a legal
`(shuffle, nbit, deflate)` dataset passes every one of them, raises no error,
and decodes to different bytes on every element, while `(deflate, shuffle)` is
indistinguishable from the pipeline handled here. Pipeline equality refuses
both deliberately, and with them fletcher32, scale-offset, lzf and any
third-party filter id. Also refused: contiguous datasets (a virtual dataset is
one of these), non-numeric dtypes, datasets under 64 MB, and partially
allocated datasets -- a chunk that was never written has no stored bytes, and
h5py's exception for one is not even of a stable type, so the chunk count is
checked up front instead. Finally, at read time, any chunk whose `filter_mask`
is non-zero (HDF5 stored it with part of the pipeline skipped) sends the whole
dataset to h5py, with a `RuntimeWarning` naming the dataset -- a fallback taken
after the gate admitted the file is worth reporting rather than silently losing
the 2.2x.

Measured on an H100 NVL with `scripts/benchmarks/bench_likelihood_call.py
--n-calls 20`, 259-event DESI nside-64 production configuration (H0 +
population + survey sampled, `zspace`-24/`ns6` quadrature, auto blocking, no
XLA cache), on top of the device row sort above, three interleaved launches per
arm, one arm per process:

| Arm | `load_survey` | load phase | call median |
|---|---|---|---|
| sequential h5py | 5.61-5.91 s | 8.69-8.99 s | 59.0-59.8 ms |
| threaded chunks | 3.10-3.33 s | 6.16-6.32 s | 59.2-59.6 ms |

That is -2.55 s of `load_survey` (1.79x) and -2.63 s of the load phase on the
medians. Per-call time is untouched (the hot path never sees this code), peak
device memory is unchanged at 22.6 GiB, and the log-likelihood is byte-equal at
all eight benchmark draws. The spectral configuration carries no
`--survey_path` and never reaches `load_survey`, so nothing about it changes
(14.0 ms/call in both arms).
