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
peak ≈ TRUE_FIXED_BYTES + static_state_bytes
       + sel_batch · SEL_BYTES_PER_INJECTION[_CAT] · scale
       + pe_block  · n_samp · PE_BYTES_PER_SAMPLE[_CAT] · scale
```

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
3. **Budget** = `SAFETY_FACTOR · free_bytes − true_fixed − static_state_bytes`
   (`SAFETY_FACTOR = 0.7`, headroom for fragmentation / transient copies / a
   shared box), where `true_fixed` is `TRUE_FIXED_BYTES` on the gradient path and
   `TRUE_FIXED_VALUE_BYTES` on the gradient-free one.
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
  static state so the calibration point is preserved bit-for-bit.
* **Value-only slopes** — MEASURED on an H100 NVL
  (`scripts/benchmarks/block_sizes_h100_80gb_value_only.json`, 15 configs): the
  two-point system `F + B·c = 0.703 GB` / `F + N_sel·c = 5.648 GB` gives
  `c = 4,777 B/unit` and `F = 0.51 GiB`, rounded up to 5,000 and 1 GiB. The model
  predicts ≥ the measured peak at every measured config and within 2× of it.
* **`_CAT` (dark-siren) slopes** — still 2× the spectral ones, **unmeasured**;
  erring high over-blocks (slower, safe) rather than under-blocks (OOM).

### Calibrating precisely

`scripts/benchmark_block_sizes.py` fits the slopes on a GPU. It runs one
subprocess per config (clean per-config `peak_bytes_in_use`, OOM isolation) under
the **BFC allocator** (the `platform` allocator does not track a peak), building
the real spectral likelihood via the CLI phase functions and recording the peak
plus compile/warm times.

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

The lensing CLI's `loglike` is deliberately **not** jitted as a whole: under
`--partition_mode marginalize_exact` it builds one likelihood call per partition in
a Python loop, and fusing those into a single XLA program is exactly the
host-memory blow-up that halted the lensing campaign. Its per-partition eager lens
decode is hoisted to once per call instead.
