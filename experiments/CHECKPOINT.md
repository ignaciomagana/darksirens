# Checkpoint — 2026-08-20 15:07 EDT

Written after a session crash, to make the next start cold-readable. Nothing here
is authorized work; §5 is a queue, not a plan in motion.

## 1. Compute in flight — healthy, self-resuming, do not restart

The `sigma_kde = 0.006` population band, in a **different tree**:

    /hildafs/projects/phy230014p/magana/desi_darksirens_selection
      final/experiments/experiment_loa_rebuild/{scripts,runs,logs,results}

| | |
|---|---|
| driver | `scripts/run_band_ext_overnight.sh` → `scripts/run_band_sk006.sh` (PID 17496 → 24252), up since 06:46 |
| progress | 68 / 72 draws (draws 00–67 done; draw 68 started 14:58) |
| cadence | 12.4 min/draw, steady across the last 26 |
| ETA | draws 69–71 → sweep ends ≈ 15:47, then compose + figs run automatically |
| failures | none — no `FAILED rc=` in `logs/band_loa44_sk006/driver_ext.log` |

**It resumes itself.** `run_band_sk006.sh` skips any draw with an existing
`draw_$i/joint/h0_scan.summary.json`, so a crash costs at most the in-flight draw.
If the driver itself dies, re-running `run_band_ext_overnight.sh` is safe and
idempotent: it waits for any live sweep, then continues from the first missing draw.

**Why the extension exists** (from the driver's own header): the 32-draw band had
ESS ≈ 7.7 with top weight 0.31 — dropping the heaviest draw moved the formal
marginalisation by +0.74. 72 draws is the fix. Draws 32+ come from the same GWTC-5
hyperposterior (sha256 `aa159c4259ccdbb9`, seed 20260706, permutation-based, so
00–31 reproduce bit-identically and 32+ are genuinely new, zero overlap).

**First thing to read when it finishes:** the recomposed ESS and top weight in
`results/band_loa44_sk006_diagnostics.json`. That number is the whole point of the
run; the per-draw medians (56.9–71.1, 90% CIs ≈ [44, 85]) are not.

## 2. Repo state — `src/darksirens-dev`, 8 open PRs

Working tree clean. Two branches moved on 2026-08-20:

- **`feat/s3-followups`** — checked out, PR #407. HEAD `2e594fb`: the `--depth-map`
  fix **did not work**, measured and recorded (see §3).
- **`feat/field-level-h100-production`** — PR #405. HEAD `6c8ef98`: ITEM 1 closed —
  the photo-z bias knee sits essentially *at* production's own fractional kernel
  width (bias +0.89 / +1.92 / +2.64 / +2.82 at 0 / 5 / 10 / 21%), so production
  gets neither the relief of a narrower kernel nor the protection of saturation.
  Dispersion is *insensitive* to the kernel (1.449, 1.224, 1.322, 1.283 at n=50);
  the quoted sigma grows with it (3.74 → 6.24), as it must.

Which branch owns which experiment directory — this cost time to work out, so:

| directory | owned by |
|---|---|
| `completeness_viz`, `desi_ingest` | `feat/s3-followups` (committed `ccc9eb0`) |
| `desi_full259` | **both** — different file sets on each; `ccc9eb0` added the h0-scan/n0-calibration side, `feat/field-level-h100-production` has the ensemble/OPG/sandwich side |
| `vollim` | `feat/field-level-h100-production` only (`9a6988e`); on this branch only its `.gitignore` exists and the on-disk data is ignored |

## 3. The one thing that is NOT running and should be

Commit `2e594fb` ends "a v2 build with both changes is queued." **It was not
queued** — no process, no artifact, no script. It died with the session.

What v1 measured (this is the failure, not the target):

    off-footprint logQ   mean -0.073  sd 0.5694     (mask-freedom needs EXACTLY 0)
    corr(Q, f_p)         +0.392 @ z-slice 150
                         +0.333 @ 250
                         -0.995 @ 400

Mechanism: with `f_p` folded in, an off-footprint pixel's model rate is
`f_p C dN_exp Q = 0` for *any* Q while `N_obs = 0` too, so the likelihood is flat
in Q there and the MAP is set by the lognormal prior plus per-z budget
renormalization. The footprint stays imprinted — differently, and sign-inverted at
high z, which is worse than leaving it alone.

The two changes now in `build_lognormal_completion.py`, awaiting measurement:
1. off-footprint pixels **excluded** from the fit, so they keep `logQ = 0` by
   zero-init (what the per-pixel branch already does for empty pixels);
2. `f_p_aware` is **earned by measurement** — `_verify_mask_free` checks the
   finished table (off-footprint `|logQ| <= 1e-6`, `|corr(Q, f_p)| <= 0.10` on
   covered sky, nine z slices) and reports False loudly rather than raising.

Change 2 is the one that matters beyond this bug: the inference loader admits the
`f_p × Q` pairing on the strength of that attr, so an unearned True converts a
refusal into a silent double-count. Pairing a footprint-absorbing Q with f_p put
H0 at **41.24 [36.1, 46.3]** against a truth of 67.74.

**The open question v2 answers:** is exclusion sufficient, or does Q still
correlate with f_p through the budget renormalization alone?

**Gap to close first:** the v1 build was invoked ad hoc and never scripted, which
is why nothing survived the crash. Write the build as a script under
`experiments/desi_ingest/` (alongside `run_qbuilds.sh`) before launching, so v2 is
reproducible and v3 is cheap. Shape of the invocation, from the CLI's own help —
`--depth-map` requires `--c-mode aggregate|selection`:

    darksirens_build_lognormal_completion \
      --catalog <pixelated catalog h5> --out <q_v2.h5> \
      --c-mode selection --depth-map <mth_map_nside128.h5> [--mode radial|gp3d]

Hold the launch until the band sweep frees the GPU (≈15:47); `run_qbuilds.sh`
notes the radial builder forks workers, so `JAX_PLATFORMS=cpu` for that path.

## 4. What is tracked in `experiments/`, and what is not

`ccc9eb0` committed 992 KB of the 26 GB on disk: scripts, notes, and the small
JSON results. Ignored by extension (`*.h5 *.npz *.npy *.png`), never by `data/` —
a bare `data/` pattern is a data-loss pattern in this repo, and had already eaten
`desi_ingest`'s only copies of its measured numbers. `.provenance.json` sidecars
stay tracked beside their ignored artifacts. Two disqualifier notes travel with
the numbers they disqualify: `desi_ingest/data/h0_full259_INVALID_raw_pe/DO_NOT_QUOTE.md`
and `desi_full259/data/h0_scans/DIAGNOSTIC_ONLY.md`.

## 5. Queue for the next session — gated, awaiting explicit go

1. Read `results/band_loa44_sk006_diagnostics.json`: did 72 draws lift ESS out of
   the ~7.7 regime, and did the marginalised H0 move?
2. Script the v2 `--depth-map` Q build, then run it and measure `_verify_mask_free`
   on the artifact. Do not trust the stamp without reading the report.
3. Decide what #405 and #407 need before merge; 8 PRs are open and stacked.

Operational pins that bite: `DARKSIRENS_ZMAX=0.75` (import `common.py` first),
`JAX_PLATFORMS=cpu` for the forking radial builder and for golden tests
(`test_field_weighting_is_live` fails on GPU on master — pre-existing),
`XLA_PYTHON_CLIENT_PREALLOCATE=false` when sharing the H100.
