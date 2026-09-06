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
bigger gain; the pool is `min(4, os.cpu_count())`.

The arrays are the same bytes, so nothing numeric changes. This is not a
tolerance claim: the result of `read_dataset_chunked` is byte-for-byte the
result of `np.asarray(dset)`, verified with `tobytes()` equality on all four
production datasets and on synthetic files exercising every edge (partial edge
chunks on both axes, 4- and 8-byte dtypes, 1-D datasets, 1/2/4/8 workers).
`read_dataset_chunked` reimplements exactly one filter pipeline, so it refuses
everything else and reads it with plain h5py instead: contiguous datasets, any
compressor other than gzip, gzip without shuffle, a fletcher32 checksum or a
scale-offset filter, virtual datasets, non-numeric dtypes, datasets under
64 MB, and -- per chunk, at read time -- any chunk whose `filter_mask` is
non-zero (HDF5 stored it with part of the pipeline skipped) or that is not
allocated at all, in which case the whole dataset falls back.

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
