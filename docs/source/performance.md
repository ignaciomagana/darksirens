# Performance: memory-aware block sizing

The hierarchical likelihood has two static block-size knobs that trade throughput
for peak device memory. Both default to **`auto`**, which sizes them from a probed
free-memory budget so a run does not abort with an out-of-memory (OOM) error on a
dense dark-siren catalog or a smaller GPU — and falls back to the historical
single pass whenever memory is ample.

| Flag | What it chunks | Single pass (`off`) |
|------|----------------|---------------------|
| `--sel_batch_size` | injections per selection-integral chunk (`N_sel` ≈ 1.07 × 10⁶ on real data) | all injections in one vectorized pass — fastest, widest working set |
| `--pe_event_block` | events per PE-reduction chunk (`N_events` ≈ 259, each with `n_samp` ≈ 4096 samples) | all events in one block |

Accepted values (both CLIs): **`auto`** (default), **`off`** / `none` / `0` (an
explicit single pass), or a **positive integer** (pin the block). Resolution
happens after the data load; the chosen values and a provenance tag
(`block_size_resolution`: `auto` / `auto-single-pass` / `explicit` / `cpu`) are
printed and persisted to `settings.json`.

## The memory model

There are **two** peak models, selected by whether the run's sampler
differentiates the likelihood. NumPyro NUTS does; `dynesty` and `tinyns` (rwalk /
prior replacement only) never do, and the gradient-free peak is **14× smaller** on
the real spectral problem — 5.59 GB against 78.77 GB for value+grad, both measured
in the same calibration run.

**Gradient path** (`needs_grad=True`, NumPyro) — the value and the reverse-mode
residuals of both axes are live at once, so the two transient working sets add:

```
peak ≈ GRAD_RUNTIME_FIXED_BYTES + scale · GRAD_WORKSET_FLOOR_BYTES
       + static_state_bytes
       + sel_batch · SEL_BYTES_PER_INJECTION[_CAT] · scale
       + pe_block  · n_samp · PE_BYTES_PER_SAMPLE[_CAT] · scale
```

Note the *fixed* term splits. The historical 58 GiB anchor was described as
"JAX/XLA runtime + workspace" — dimension-independent — but a second `n_q` point
shows only ~1.7 GB of it is: the single-pass peak is **26.89 GB at `n_q = 64`** and
**78.77 GB at `n_q = 200`**, i.e. ~97 % of it is working set that scales with the
dimensions like the slopes do. `GRAD_RUNTIME_FIXED_BYTES +
GRAD_WORKSET_FLOOR_BYTES == TRUE_FIXED_BYTES` exactly, so at the calibration
dimensions every plan is preserved bit-for-bit; the split only fixes the
extrapolation. Before it, `--norm_nq 400` on a 141 GB card was promised a full
single pass against a real ~155 GB peak.

**Gradient-free path** (`needs_grad=False`, dynesty / tinyns) — measured on the
H100 NVL, blocking *one* axis leaves the value peak untouched (5.648 → 5.695 GB)
because the unblocked axis's `(N, n_q)` intermediate still sets it; blocking both
drops it to 0.703 GB. So it is a **max**, not a sum:

```
peak ≈ TRUE_FIXED_VALUE_BYTES + static_state_bytes
       + max( sel_batch · SEL_BYTES_PER_INJECTION_VALUE[_CAT] · scale,
              pe_block  · n_samp · PE_BYTES_PER_SAMPLE_VALUE[_CAT] · scale )
```

`sel_batch` defaults to the full `N_sel` and `pe_block` to the full `N_events` for
a single pass. `scale` normalises the slopes to the calibration config:

| Factor | Why |
|--------|-----|
| `n_grid / CAL_N_GRID` | redshift-grid nodes; more nodes → more bytes per unit |
| `n_q / CAL_N_Q` | the population **q-quadrature**, evaluated *per sample* by the default pairing normaliser — measured ~26 MB per q-node at `N_sel = 1.07 × 10⁶`, i.e. ~93 % of the value peak at the default `n_q = 200`. `n_chi` deliberately does **not** enter: the spin normalisation is a θ-only `(n_chi,)` quadrature, never a per-sample tensor |
| `(max_gals_per_row / CAL_MAX_GALS_PER_ROW) · n_catalogs` | catalog runs only (`_CAT` slopes): galaxies per union pixel and the K-catalog mixture both multiply the redshift-prior work. `max_gals_per_row = None` means the caller could not determine it and falls back to the calibration reference **with a loud warning**, never below the catalog-free slopes |
| `concurrent_evals` | samplers that `vmap` several proposals at once (tinyns `replacement_chains`) hold that many copies of every intermediate |

`static_state_bytes` is the **pending** static state — what the likelihood factory
allocates *after* the memory probe, so the probe is blind to it: the KDE caches,
the completion `base_miss` curves, the compact union galaxy tables
`prepare_catalog_views` gathers from the full-sky rows, the device transfer *and*
compact slice of the host-resident Q ensemble, and the field-convention
`field_lss_q_members` rows. The already-loaded arrays are excluded — they are
device-resident by then and hence already reflected in the probed free memory.
Constants and estimators live in `darksirens/likelihood/block_sizing.py`.

## Decision procedure

`resolve_block_sizes()` (pure; unit-tested on CPU with injected budgets):

1. **Explicit wins.** An integer or `off` on a knob passes through unchanged.
2. **Non-GPU backend** → single pass (host RAM is not the bottleneck).
3. **Budget** = `SAFETY_FACTOR · free_bytes − fixed − static_state_bytes`
   (`SAFETY_FACTOR = 0.7`, headroom for fragmentation / transient copies / a
   shared box), where `fixed` is the (scaled) gradient fixed term above or
   `TRUE_FIXED_VALUE_BYTES` on the gradient-free path.
4. **Gradient path — selection axis first** (it dominates at ~10⁶ injections): if
   the full pass fits alongside the full PE pass, keep `None` (single pass); else
   split into `k = ceil(N_sel / B_fit)` **even** chunks, rounding the block up to a
   multiple of 256 (`N_sel` pads badly against power-of-two blocks) with a floor of
   `SEL_MIN_BATCH = 32768`. The **PE axis** is blocked only if a floored selection
   block still overflows, floored at `PE_MIN_BLOCK = 8`.
5. **Gradient-free path** — each axis is sized against the *whole* budget
   independently (the peak is the max of the two), so a plan never pays for
   blocking one axis without the memory saving that only comes from blocking both.
6. **Floor guard.** If even a floor-sized block will not fit next to the pending
   static state, the floor is dropped (down to 1) and a loud diagnostic is printed
   (`block_size_resolution = auto-floor-reduced`). An unreliable free-memory
   reading never triggers this.

Example resolution for the real spectral run (`N_sel = 1,067,946`,
`N_events = 259`, `n_samp = 4096`, `n_q = 200`) at the current constants:

| Free memory | Sampler | `sel_batch_size` | `pe_event_block` |
|-------------|---------|------------------|------------------|
| 141 GB | numpyro | — (single pass) | — (single pass) |
| 88 GB | numpyro | 32,768 (floor) | 87 |
| 40 GB | numpyro | 32,768 (floor) | 8 (floor) |
| 88 GB | dynesty / tinyns | — (single pass) | — (single pass) |
| 16 GB | dynesty / tinyns | — (single pass) | — (single pass) |
| 8 GB | dynesty / tinyns | 534,016 (2 chunks) | 130 (2 chunks) |

The 88 GB dynesty row is the defect this split fixes: the gradient model chose
(32,768, 87) there, which cost **49.3 ms/call instead of 27.5 ms** (1.79×) to hold
2.27 GB instead of 5.65 GB on a card with 95 GB.

## Constants status

`CONSTANTS_VERSION = "measured-h100-80gb-decomposed+valueonly+nq-2026-07-26"`.

* **Gradient slopes** — MEASURED on an H100-80GB (`scripts/benchmarks/block_sizes_h100_80gb.json`):
  SEL 7,865 → 8,000 B/injection, PE 8,102 → 9,000 B/sample, with the 58 GiB fixed
  anchor decomposed into `TRUE_FIXED_BYTES` plus the calibration config's own
  static state so the calibration point is preserved bit-for-bit; `TRUE_FIXED_BYTES`
  is in turn split into 1.7 GB of runtime and a dimension-scaled working-set floor
  from the `n_q = 64` / `n_q = 200` pair (26.89 / 78.77 GB).
* **Value-only slopes** — MEASURED on an H100 NVL
  (`scripts/benchmarks/block_sizes_h100_80gb_value_only.json`, 15 configs): the
  two-point system `F + B·c = 0.703 GB` / `F + N_sel·c = 5.648 GB` gives
  `c = 4,777 B/unit` and `F = 0.51 GiB`, rounded up to 5,000 and 1 GiB. The model
  predicts ≥ the measured peak at every measured config and within 2× of it.
* **`_CAT` (dark-siren) slopes** — still 2× the spectral ones, **unmeasured**;
  erring high over-blocks (slower, safe) rather than under-blocks (OOM).
* **Catalog density scales the INCREMENT, not the whole peak** — the catalog and
  spectral paths share the population log density, the exact `q` normalisation
  and the per-sample selection/PE weights, so a catalog estimate is
  `spectral_common + catalog_increment` and can never fall below the
  catalog-free baseline. Multiplying the whole per-unit cost (and the gradient
  working-set floor) by `gals_ratio × n_catalogs` made `max_gals_per_row = 1`
  predict a **1.746 GB** gradient peak against the equivalent spectral path's
  **80.283 GB** — ~46× below a working set the two share, enough for `auto` to
  promise a single pass on a 40–60 GB card. At and above the calibration density
  (`cat_scale ≥ 1`) predictions are bit-identical to what shipped.

### Calibrating precisely

`scripts/benchmark_block_sizes.py` fits the slopes on a GPU. It runs one
subprocess per config (clean per-config `peak_bytes_in_use`, OOM isolation) under
the **BFC allocator** (the `platform` allocator does not track a peak), building
the real spectral likelihood via the CLI phase functions and recording the peak
plus compile/warm times.

Since 2026-08-23 BFC (`XLA_PYTHON_CLIENT_ALLOCATOR=default`, preallocation still
off) is also the **production** default in `darksirens.core.jax_config`, so the
calibration and the runs it sizes share an allocator. The old `platform` default
measured **23.0 ms/call against BFC's 13.7 ms** (1.68x) on the shipped real
spectral likelihood — 1,067,946 injections, 259 events, Dynesty value-only,
`off:off` blocking, 20 warm repetitions, clean process on an H100 NVL — and
reported no memory statistics at all. An explicit
`export XLA_PYTHON_CLIENT_ALLOCATOR=platform` is still honored; on that path
`memory_stats()` is inert and the probe falls back to `nvidia-smi` free memory.
On a shared GPU the probe now takes the *smaller* of the allocator headroom and
the physical free memory.

```bash
# smoke (validate the harness):
python scripts/benchmark_block_sizes.py --smoke \
    --gw-path gwsamples_bbh_whitelist_all_events.h5 \
    --gwselection-path selection_o3o4ab_allsky.h5

# full value+grad calibration (numpyro):
python scripts/benchmark_block_sizes.py --repeats 10 \
    --gw-path GW.h5 --gwselection-path SEL.h5 \
    --out scripts/benchmarks/block_sizes_<device>.json

# value-only calibration (the dynesty / tinyns peak), including the
# both-axes-blocked points that pin the max-model intercept:
python scripts/benchmark_block_sizes.py --repeats 8 \
    --sampler dynesty --mode value \
    --configs "off:off,32768:off,off:8,32768:8,65536:16,131072:32" \
    --gw-path GW.h5 --gwselection-path SEL.h5 \
    --out scripts/benchmarks/block_sizes_<device>_value_only.json

# n_q sensitivity (the axis that multiplies the working set):
python scripts/benchmark_block_sizes.py --sampler dynesty --mode value \
    --norm-nq 800 --configs "off:off" --gw-path GW.h5 --gwselection-path SEL.h5
```

Fit `peak_bytes` vs `sel_batch_size` (at `pe=off`) and vs `pe_event_block`
(at `sel=off`) to get `SEL_BYTES_PER_INJECTION` and `PE_BYTES_PER_SAMPLE`; the
single-pass intercept gives `FIXED_OVERHEAD_BYTES`. For the value-only set use
`peak_value_bytes` and the both-axes-blocked configs (one-axis blocking is flat).
Transcribe the fitted values and re-stamp `CONSTANTS_VERSION` with the device.
Peak memory is a **per-process** statistic, so the calibration stays valid even
when the box is shared with another job (only the secondary *timing* numbers get
noisy under contention).

## Timing one likelihood call

`scripts/benchmarks/bench_likelihood_call.py` builds the likelihood with the
CLI's own phase functions (option resolution, data load, parameter space,
factory) and times the compiled callable on a fixed set of prior draws,
recording the log-likelihood values so a candidate build can be compared with a
base build for speed and value drift in one command each:

```bash
python scripts/benchmarks/bench_likelihood_call.py --tag base --out base.json -- \
    --gw_path GW.h5 --gwselection_path SEL.h5 --survey_path CAT.h5 \
    --universe_model dark_sirens --pop_model powerlaw+peak --sampler dynesty \
    --fix_population true --fix_survey true --fix_de true \
    --fixed_parameter_values '{"Om0": 0.3075}' --save_path /tmp/bench_run
python scripts/benchmarks/bench_likelihood_call.py --tag cand --compare base.json -- ...
```

`--components` additionally times the per-proposal state build, the
population/Jacobian kernel, the redshift-prior evaluation and the catalog KDE
on the PE and selection sets as isolated jits (the prepared state is passed as
an argument, so a consumer is timed without its producer). Report the backend:
CPU and GPU profiles differ qualitatively — the windowed catalog KDE is
gather-bound on a GPU and `exp`-bound on a CPU — and on a CPU box pass explicit
`--sel_batch_size` / `--pe_event_block` / `--row_chunk 256`, because the auto
plan is a single pass on non-GPU backends and a dense catalog then exceeds host
RAM.

## The per-sample catalog KDE: window sized from the data, injections sorted by pixel

The dark-siren redshift prior evaluates, for every PE sample and every
injection, a Gaussian mixture over the galaxies of the sample's pixel row. On a
CPU it is ~85% of a dark-siren likelihood call once the kernel quadrature is
pinned; on a GPU it is gather-bound (each sample reads its window of the row
arrays). Three exact changes:

* **Window sized from the data, never truncating.** The static window `W` used
  to be a fixed 1024. The factory now sizes it from the bound catalogs
  (`redshift.catalog.auto_kde_window`): the largest number of galaxies any row
  holds within `n_sigma` row-max kernel widths at the widest `sigma_kde` the run
  can reach (its fixed value, or the prior's upper bound when it is sampled),
  plus one, rounded up to a multiple of 64. Every sample's in-range block then
  fits, so the evaluator evaluates exactly the galaxies within `n_sigma` widths
  and never falls to its nearest-`W`-by-index truncation. MEASURED on a
  DESI-like mixed spectro+photo catalog (73% of widths in [0.02, 0.10], 2158
  galaxies per row) the old fixed window moved `log p_cat` by 0.17 nats on
  average and 0.36 at worst against the full-row evaluator, coherently in one
  direction; the data-sized window reproduces the full row. When the data-sized
  window exceeds 1024 the factory warns, because a dense photo-z catalog then
  pays for the exact answer; `--kde_window` pins it, `--kde_window 0` is the
  full-row escape hatch. The row-max width itself was also wrong: it was taken
  over the padded row (`dzgals` pads at 1.0), so every row shorter than `N_max`
  reported a half-width of 8 in redshift and the block never fit a window
  shorter than the row. It is now the maximum over the real galaxies.
* **Three gathers instead of four.** `CatalogKernelState.log_kw_eff` fuses
  `log_kw - log(sigma_eff) - log sqrt(2 pi)` at state-build time, so the
  evaluator gathers `z_i`, `sigma_i` and that array, and the per-galaxy work is
  one subtraction, one division and one FMA before the `logsumexp`. Same
  arithmetic up to re-association.
* **Injections sorted by pixel.** The factory permutes every per-injection
  array by the compact pixel index (a stable sort) so consecutive samples
  re-read the same row and the gathers hit cache. The selection integral is a
  sum over injections, so the order changes only the floating-point
  association of the batched `logsumexp`.
* **One binary search per sample.** The window used to be located with three
  dependent searches (the in-range block's two edges and the sample's
  insertion index) followed by a fit test that shifted the window onto the
  block. It is now the `W` index-nearest galaxies centred on the insertion
  index — one search — and exactness moved into the SIZING:
  `recommended_kde_window` returns twice the largest ONE-sided count of
  galaxies within `n_sigma` row-max widths (capped at the row length), which is
  the smallest `W` for which the centred window provably holds every galaxy
  within `n_sigma` widths of any sample. For an evenly populated row that is
  the same count as the two-sided block; only a sample beside a dense clump
  pays for the clump on both sides. The three searches were ~19% of the
  windowed evaluation on a CPU, and dependent scalar gathers are latency-bound
  on a GPU.

MEASURED on a 4-core CPU with the nside-16 mock of the previous section (3072
rows × 2158, 64 events × 4096 samples, 839k injections, H0 the only sampled
label): 7.07 s/call → 5.01 s/call (1.41×), log-likelihoods bit-identical at
five prior draws. The pixel sort alone was 1010 → 770 ms per 131k injections
of KDE evaluation.

## Population-only runs: the redshift prior is frozen at build time

A run that samples no cosmology, survey or mark label — the
`scripts/run_tinyns_heavy_darksirens_likelihood.sh` launcher's
`--fix_cosmology true --fix_survey true` population inference — has a
redshift prior that cannot move: `log p(z | pix)` is a pure function of the
fixed cosmology and survey block, of the catalog, and of `z = z(dL)` at the
fixed cosmology. The factory (`frozen_prior_admissible`, a test on the sampled
LABELS like the kernel pin's) then evaluates it once for every PE sample and
every injection with the same `prepare_redshift_prior_state` /
`eval_redshift_prior_with_state` the live graph would run, and hands the two
vectors to `darksiren_log_likelihood` as the `frozen_prior` operand. The body
skips both seams' state builds and adds the per-sample values in place of the
prior closure; the selection reduction goes through
`selection_reduce_from_ldw_provider` with a provider that reproduces
`compute_selection_term`'s masks. Sky-model labels (applied outside the prior)
and mixture sticks (combined live with the frozen per-catalog columns) may still
be sampled. The premise is re-verified IN THE GRAPH on every call: the fixed
cosmology/survey scalars the prior reads are compared exactly against the
values it was built at, and a mismatch poisons the log-likelihood to `-inf`.
`--freeze_redshift_prior false` is the A/B switch.

MEASURED on the 4-core CPU mock (3072 rows × 2158, 64 × 4096 PE samples, 839k
injections, powerlaw+peak fully sampled, cosmology and survey fixed): 7.38 s/call
→ 3.19 s/call (2.3×), log-likelihood bit-identical at the fiducial population.
What remains is the population term itself — dominated by the exact per-sample
`q`-quadrature of the pairing normaliser — and the reductions.

## The likelihood closure is jitted

The callable returned by `make_likelihood` / `_make_mixture_likelihood` /
`make_flow_likelihood` is `jax.jit`-wrapped. Only the inner
`darksiren_log_likelihood` used to be, so parameter decoding (~30 individual eager
device ops) and — on the flat K=1 path — the GW-container rebuild and selection
re-padding ran eagerly on **every** call. `dynesty` calls that closure once per
live point, so it was 7.7–13.8 ms of a 30–51 ms spectral call and 44 % of a
dark-siren call. `tinyns` and NumPyro escaped it because they trace the closure
themselves.

Two implementation constraints are load-bearing:

* the coord-independent operands (GW containers, `EMCatalog`s, WL tables) are
  built **once** at factory time, so nothing per-call re-materialises them;
* they are passed to the jitted body as **arguments**, never captured by the
  closure. `jax.jit` lowers a closed-over concrete array to a `dense<>` HLO
  *constant* (verified on jax 0.4.34: module text grows ~8 bytes per element), so
  capturing them would embed the entire static state in the module and duplicate
  already-resident buffers.

The largest such array was not an operand at all but the comoving-distance
interpolation table `utils.cosmology.rs` — 21×41×31×500 float64, 106.8 MB — which
every likelihood reaches through `z_of_dL` / `dL_grid_bounds` far below the factory
frame, and which was read as a module global. Measured on an H100 NVL:

| lowered module | before | after |
| --- | --- | --- |
| spectral (real data, single pass) | 427.5 MB | 0.44 MB |
| dark-siren mock | 443.6 MB | 16.6 MB |
| XLA `generated_code_size` (spectral) | 107.5 MB | 0.70 MB |
| first likelihood call (trace + lower + compile + run) | 6.90 s | 2.76 s |

`argument_size` grows by exactly 106,764,000 bytes — the table moved from the
generated code into the parameter list, and the buffer it now points at is the
one that was already device-resident (`bytes_in_use` and peak are unchanged).
Log-likelihoods are bitwise identical at 12 spectral and 8 dark-siren prior draws.

The table is threaded by `utils.cosmology.threads_distance_table`, which replaces
`@jax.jit` on every function that reaches it: the public wrapper resolves the
table outside the jit and passes it as an argument, and inside the jit the
decorator rebinds it as the active table so the ~45 deep call sites keep their
signatures. **Any new module-level jit that reaches the table must use that
decorator too** — a plain `@jit` that merely reads the active table closes over
the caller's tracer, which JAX does not key its tracing cache on and will replay
from a dead trace (`UnexpectedTracerError`, reproduced on jax 0.4.34).

The lensing CLI's `loglike` is deliberately **not** jitted as a whole: under
`--partition_mode marginalize_exact` it builds one likelihood call per partition in
a Python loop, and fusing those into a single XLA program is exactly the
host-memory blow-up that halted the lensing campaign. Its per-partition eager lens
decode is hoisted to once per call instead.
