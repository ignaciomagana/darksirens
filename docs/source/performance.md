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

Peak device bytes during one likelihood value+grad are modelled as a linear
function of the two block sizes:

```
peak ≈ FIXED_OVERHEAD_BYTES
       + sel_batch · SEL_BYTES_PER_INJECTION[_CAT]
       + pe_block  · n_samp · PE_BYTES_PER_SAMPLE[_CAT]
```

where `sel_batch` defaults to the full `N_sel` and `pe_block` to the full
`N_events` for a single pass. The `_CAT` variants apply when a galaxy catalog is
loaded (dark sirens), whose per-injection/per-sample redshift-prior state is
heavier than the catalog-free spectral path. The constants live in
`darksirens/likelihood/block_sizing.py`.

## Decision procedure

`resolve_block_sizes()` (pure; unit-tested on CPU with injected budgets):

1. **Explicit wins.** An integer or `off` on a knob passes through unchanged.
2. **Non-GPU backend** → single pass (host RAM is not the bottleneck).
3. **Budget** = `SAFETY_FACTOR · free_bytes − FIXED_OVERHEAD_BYTES`
   (`SAFETY_FACTOR = 0.7`, headroom for fragmentation / transient copies / a
   shared box).
4. **Selection axis first** (it dominates at ~10⁶ injections): if the full pass
   fits alongside the full PE pass, keep `None` (single pass); else split into
   `k = ceil(N_sel / B_fit)` **even** chunks, rounding the block up to a multiple
   of 256 (`N_sel` pads badly against power-of-two blocks) with a floor of
   `SEL_MIN_BATCH = 32768`.
5. **PE axis** only if a floored selection block still overflows, floored at
   `PE_MIN_BLOCK = 8`.

Example resolution for the real spectral run (`N_sel = 1,067,946`,
`N_events = 259`, `n_samp = 4096`) at the current constants:

| Free memory | `sel_batch_size` | `pe_event_block` | source |
|-------------|------------------|------------------|--------|
| 80 GB | — (single pass) | — (single pass) | `auto-single-pass` |
| 40 GB | 534,016 (2 chunks) | — | `auto` |
| 24 GB | 267,008 (4 chunks) | — | `auto` |
| 12 GB | 32,768 (floor) | 130 | `auto` |

## Constants status — conservative, not yet slope-calibrated

`CONSTANTS_VERSION = "conservative-estimate-a100-2026-07-18"`.

The current constants are **conservative estimates, not measured slopes**. A smoke
run on an A100-80GB confirmed only the anchor that a full single-pass value+grad
of the spectral likelihood fits within the ~0.7·free budget (~50 GB); the
constants are chosen so the model predicts a ~40 GB full working set, so `auto`
keeps a single pass on an ample GPU and blocks only when memory is genuinely
tight. Erring high is deliberate — over-estimating bytes/unit over-blocks (slower,
safe) rather than under-blocks (OOM). The `_CAT` (dark-siren) values are set to
2× the spectral ones pending a catalog run.

### Calibrating precisely

`scripts/benchmark_block_sizes.py` fits the slopes on a GPU. It runs one
subprocess per config (clean per-config `peak_bytes_in_use`, OOM isolation) under
the **BFC allocator** (the `platform` allocator does not track a peak), building
the real spectral likelihood via the CLI phase functions and recording the
value+grad peak plus compile/warm times.

```bash
# smoke (validate the harness):
python scripts/benchmark_block_sizes.py --smoke \
    --gw-path gwsamples_bbh_whitelist_all_events.h5 \
    --gwselection-path selection_o3o4ab_allsky.h5

# full calibration:
python scripts/benchmark_block_sizes.py --repeats 10 \
    --gw-path GW.h5 --gwselection-path SEL.h5 \
    --out scripts/benchmarks/block_sizes_<device>.json
```

Fit `peak_bytes` vs `sel_batch_size` (at `pe=off`) and vs `pe_event_block`
(at `sel=off`) to get `SEL_BYTES_PER_INJECTION` and `PE_BYTES_PER_SAMPLE`; the
single-pass intercept gives `FIXED_OVERHEAD_BYTES`. Transcribe the fitted values
and re-stamp `CONSTANTS_VERSION` with the device. Peak memory is a **per-process**
statistic, so the calibration stays valid even when the box is shared with
another job (only the secondary *timing* numbers get noisy under contention).
