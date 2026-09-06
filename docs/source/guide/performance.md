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

## The pairing mass-ratio normaliser

The conditional pairing density `p(q | m1)` is normalised per sample,
`N(m1) = int_{q_cut}^{1} p(q | m1) dq` with `q_cut = m_min/m1`, twice per
likelihood call (once on the PE set, once on the injections). Until 2026-09-05
that integral was a 200-node uniform trapezoid on the support-relative interval,
i.e. 200 density evaluations per sample; it is now **two 16-node Gauss-Legendre
panels split at the taper shoulder** `q_a = (m_min + dm_min)/m1`, i.e. 32.

The split is what makes 32 nodes beat 200. Below `q_a` the integrand is the
Planck-taper boundary layer, above it the bare pairing kernel (`q**beta`) with
the taper identically one, so the combined integrand has a corner at `q_a` that
neither piece has. Gauss-Legendre resolves each smooth piece; a single rule
spanning the corner does not. Each pairing model supplies its own shoulder
(`PairingModel._taper_shoulder`), because `gwtc5_fiducial_bpl2peaks` tapers on
its sampled `(m2_low, delta_m2)` rather than on the mixture's `(m_min, dm_min)`.

Being bare above the shoulder, that upper panel is **not quadratured at all**
for either production pairing: `PairingModel._plateau_integral` returns
`int_{q_a}^{1} q**beta dq = (1 - q_a**(beta+1))/(beta+1)` (the `beta -> -1` limit
`-log q_a` is a live branch — `beta` is sampled over `[-2, 7]`), so those models
spend 16 nodes per sample, not 32, and the remaining nodes all sit in the
boundary layer. A pairing without an analytic plateau — `GaussianPairing`, or
any out-of-tree model — does not implement the hook and keeps the
Gauss-Legendre panel unchanged.

Sixteen nodes per panel are calibrated for a `q**beta` kernel — what both
production pairings are above the shoulder. A pairing model whose kernel carries
a feature *narrower than a panel* is not resolved by any fixed 16-node rule and
must declare that feature's edges through `PairingModel._panel_edges`, which
buys it one extra panel per edge; `GaussianPairing` (not grammar-registered, but
public API) declares `mu_q ± 5 sigma_q` and is 27 nats off without them.

There is **no flag**: the node count is a module constant
(`darksirens.gw.populations.utils.PAIRING_PANEL_NQ`), calibrated as a pair
against a converged reference and exercised on every call. In particular
`--norm_nq` / `DARKSIRENS_GW_N_Q` no longer changes this quadrature — it still
sizes `get_q_grid()` (the GP baselines and the stratified-q tables) and is still
what the block-size resolver reads, so blocking plans are unchanged.

Measured on this repository at `f770956` on an NVIDIA H100 NVL, float64,
20 warm calls per launch, three interleaved launches per arm:

| Configuration | Before | After | Speedup | Peak device memory |
|---|---|---|---|---|
| 259-event spectral (`spectral_sirens`, `powerlaw+peak`, all sampled) | 13.90 ms/call | 2.88 ms/call | 4.83x | 3.672 -> 0.640 GiB |
| 259-event dark sirens (DESI nside-64, `gwtc5_fiducial_bpl2peaks`, field weighting, auto blocking) | 59.2 ms/call | 46.6 ms/call | 1.27x | 22.620 -> 22.620 GiB |
| the same run on the shipped sbatch pins (`--sel_batch_size 131072 --pe_event_block 32`) | 68.2 ms/call | 55.1 ms/call | 1.24x | 10.598 -> 10.599 GiB |

`Before` is master `f770956`, `After` the head of this change. The pinned row
was re-measured for that head with both arms interleaved in one lock hold
(master 68.65 / 67.91 / 68.20 ms against 55.27 / 55.09 / 54.59); the
peak-memory column comes from a separate hold that replays eight fiducial calls
per arm and reads the device allocator.

The closed plateau is the last 0.69 ms of the spectral row (3.57 -> 2.88 ms,
1.24x, and the 0.776 -> 0.640 GiB of the memory column), 0.90 ms of the
auto-blocked dark-siren row (47.49 -> 46.59 ms, 1.02x) and 1.36 ms of the pinned
one (55.70 -> 54.34 ms, 1.025x), each measured as three interleaved launches per
arm; the dark-siren peak memory is unmoved by it (22.620 GiB in both arms of
that A/B), because the catalog KDE, not the q axis, sets it.

The absolute saving is ~11-13 ms in all three; it is a larger fraction of the
spectral call because that call has no catalog term. **A production run started
from `experiments/desi_full259/sbatch_ns_joint_sel.sh` gets the third row**, not
the second, unless it is switched to auto blocking first: the speedups are
close, but the pinned call is ~9 ms/call more expensive in absolute terms (it
buys that back in peak memory, 10.6 GiB against 22.6). On the spectral
configuration the pairing q-axis was also the dominant allocation, which is why
the peak device memory falls 5.7x there and does not move at all on the
auto-blocked dark-siren configuration, whose blocking is unchanged.

Accuracy is a **correction**, not a regression. Against a converged
composite-Gauss-Legendre reference, holding the population and survey blocks at
their fiducials and scanning `H0` across its full prior:

| Configuration | Rule | Mean error | Trend across the H0 prior |
|---|---|---|---|
| 259-event dark sirens, H0 in [20, 140] | 200-node trapezoid | -0.349 nats | **-0.101 nats** |
| | 2-panel GL-16 | +1.8e-4 nats | +1.2e-3 nats |
| | GL-16 + closed plateau | +1.8e-4 nats | +1.2e-3 nats |
| 259-event spectral, H0 in [20, 120] | 200-node trapezoid | -0.303 nats | +0.044 nats |
| | 2-panel GL-16 | -1.1e-5 nats | +9.1e-7 nats |
| | GL-16 + closed plateau | -1.1e-5 nats | +9.1e-7 nats |

At the fiducial population the closed plateau changes nothing measurable (the
two rules differ by 2.1e-8 nats across the whole dark-siren H0 prior, and not at
all on the spectral one). It matters at **steep negative `beta`**, where the
plateau integrand `q**beta` is itself a boundary layer at `q_a = m_shoulder/m1`
— which shrinks like `1/m1`, so the Gauss-Legendre residual grows with `m1` and
does not self-average over the mass population. Repeating the same scan with
`(beta_q, m2_low, delta_m2) = (-1.9, 3.05, 1.15)`, inside the shipped prior:

| Rule | Error across the H0 prior [20, 140] | Trend |
|---|---|---|
| 2-panel GL-16 | 0.93 nats peak-to-peak | **-1.04 nats** |
| GL-16 + closed plateau | 2.0e-4 nats peak-to-peak | -2.8e-5 nats |

So the closed form removes an H0-correlated systematic twenty times over the
0.05-nat budget in a corner of the prior a sampler visits. Over a 400-draw prior
box swept from the support edge up (600 log-spaced `m1` each, so the taper
panel's own near-edge residual dominates it) the worst `|Delta log N|` also
falls, 6.7e-3 nats to 4.6e-3, and no draw gets worse.

The trapezoid's residual is a one-sided endpoint deficit that does not
self-average over the mass population, so it tilts with `H0`
(`m1src = m1det/(1+z(H0))` rescales the whole population). Removing an
H0-correlated systematic of -0.10 nats is the reason this change is admissible;
the per-call worst case improves too, ~4x near the support edge (7.6e-3 nats
against 3.1e-2 — that residual belongs to the taper panel, which the closed
plateau does not touch) and by six orders of magnitude over the whole prior box
(7.4e-8 against 2.9e-1; the two-panel rule was 4.6e-2 there before the plateau
was closed). Where the box worst case sits moved with it: it is no longer the
steep-`beta` corner `m_min = 2`, `dm_min = 0`, `beta = -2`, `m1 = 250` — the
closed form's H0-coherent error there is 3e-15 nats against the GL plateau's
2.6e-3 — but a WIDE taper, `m_min = 8.42`, `dm_min = 8.67`, `beta = -0.84`,
`m1 = 15.8`, where the residual is the taper panel's and its H0-coherent part is
1.1e-3 nats, the same for both rules. Golden log-likelihoods move
accordingly and were re-blessed (population registry: max 7.7e-5 nats, 1.5e-5
relative, one-sided).

## The luminosity-distance inversion

Every call inverts `dL -> z` twice in the dark-siren and spectral
configurations — once for the PE samples, once for the selection injections —
by interpolating the per-proposal `dL_grid` (`z_of_dL_precomputed`,
`darksirens/utils/cosmology.py`); the cluster and weak-lensing paths call the
same inversion at more sites, so what follows is the two-site case. `jnp.interp`
finds the bracketing node with `searchsorted(side='right')` at JAX's
module-default `method='scan'`, a serial `ceil(log2 N)`-level `lax.scan` (10
levels for the 543-node `zgrid` at `DARKSIRENS_ZMAX=6`). On a GPU that lowers to
a `while` op XLA cannot fuse through, so it also splits the surrounding
per-sample kernel in two.

`z_of_dL_precomputed` therefore spells the interpolation out — jax 0.4.34's own
`_interp` body, expression for expression, including the `dx0` zero-width-cell
guard, the two out-of-range fills and the `shape(xp) != shape(fp)` fail-fast —
and asks for `method='scan_unrolled'`, which lowers the same binary search to
straight-line code that fuses into the neighbouring memory-bound work. The
method selects only **how** the insertion index is found, never which index, so
the result is **bit-identical**; the `in_grid` mask that returns `NaN` outside
the grid is unchanged. `DARKSIRENS_INTERP_SCAN=1` (read once at import) restores
the stock `jnp.interp` call for an A/B.

Measured on an NVIDIA H100 NVL, float64, `DARKSIRENS_ZMAX=6`, 20 warm calls per
launch, three launches per arm interleaved inside one lock hold. `Before` is the
head of the pairing-normaliser change above; both columns are the MEAN of the
three launch medians, and `Saving` is the launch-paired difference, which is the
quantity this design resolves:

| Configuration | Before | After | Saving | Speedup | Peak device memory |
|---|---|---|---|---|---|
| 259-event spectral (`spectral_sirens`, `powerlaw+peak`, all sampled) | 2.778 ms/call | 2.403 ms/call | -0.375 ± 0.016 ms | 1.16x | 0.640 -> 0.644 GiB |
| 259-event dark sirens (DESI nside-64, `gwtc5_fiducial_bpl2peaks`, field weighting, auto blocking) | 46.453 ms/call | 46.163 ms/call | -0.290 ± 0.052 ms | 1.006x | 22.620 -> 22.637 GiB |

Per-launch medians were 2.572 / 2.872 / 2.891 ms against 2.167 / 2.519 / 2.523
ms on the spectral configuration and 46.439 / 46.502 / 46.419 ms against 46.208
/ 46.106 / 46.174 ms on the dark-siren one. The saving is the same fixed
0.2-0.4 ms in both, because it is the same two inversions: 13.5% of the spectral
call, 0.6% of the dark-siren one. A second, independent interleaved A/B (three
launches per arm spectral, two production, 25-30 warm calls per launch, launch
order rotated) reproduced -0.334 ms / 1.13x on the spectral configuration and
-0.18 ms / 1.004x on the dark-siren one, so read the dark-siren row as -0.18 to
-0.29 ms (0.4-0.6%): that is at this box's 0.3 ms process-to-process drift, and
only the launch pairing resolves it — the sign is consistent across all five
pairs. That run also measured the third arm: the same candidate build with
`DARKSIRENS_INTERP_SCAN=1` lands back on the base (2.849 against 2.825 ms
spectral), which pins the whole saving on the `searchsorted` lowering and none
of it on the surrounding restructure.

The log-likelihoods are equal bit for bit at every benchmark coordinate in both
configurations (`max |dlogL| = 0`, identical `-inf` pattern), and at eight
fiducial coordinates per configuration.

The unroll is paid for in compile time and in live intermediates, both small:
`t_first_call` 3.9-5.6 s -> 4.4-6.7 s on the spectral build and 10.0-10.3 s ->
10.5-10.9 s on the dark-siren one (one-time per run, against 1e5-1e6 calls, and
measured for the two-site case), and the peak-memory column above.
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
