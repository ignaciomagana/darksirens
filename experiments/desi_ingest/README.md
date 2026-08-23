# DESI real-catalog ingestion (selection-completeness channel)

Real-data completion of the audit's Stage 2: ingest the DESI Loa + Legacy
Survey union catalog into the merged magnitude-selection channel
(`c_mode="selection"`, PRs #338/#339) and run it end-to-end through a
real-GWTC H0 posterior. Plan: `~/.claude/plans/noble-gliding-pinwheel.md`.

## Sources (READ-ONLY)

- Catalog basis: `experiment_loa_rebuild/inputs/rebuild_loa_faint_pixelate_input.h5`
  (22,787,835 rows; LS DR9-north/DR10-south photo backbone + DESI Loa BGS
  spec upgrades; `M_APP` = dereddened apparent r; `SURVEY_CODE`
  0=spec-bright, 1=LS-photo, 2=spec-faint). South rows `[0, 17907592)`,
  north `[17907592, 22787567)`, then 268 unmatched DESI-only rows.
- GW leg: `gwsamples_44.h5` (gwcat-1.0, 44 events, S>=0.495 rule) +
  matched beta `selection_betaS_v2_loaFaint_marg_s0495_noom.h5`
  (p_pass baked; **never** swap in a plain injection set without dropping
  the S-cut event list — event selection and beta are matched by construction).
- Reference posterior for the parity gate: `runs/loa_faint_joint42/`
  (42 events, per_pixel, nside 128, H0 = 68.94 +10.1/-9.5, grid 20-140).

## Pins

- `DARKSIRENS_ZMAX=0.75` (import `common.py` first; z(max PE dL; H0=140)=0.731).
- `z_depth = 0.30` (catalog truncation, survey attr — separate from ZMAX).
- `m_lim = 21.0` (union detection set by the deepest layer, LS r<=21).
- Ingestion cut: `isfinite(M_APP) & M_APP <= 21.0` — drops exactly 269 rows
  (268 unmatched-tail NaNs + 1 spec row at 21.166).

## Known modeling risk (documented, not patched in Stage A)

Off-footprint pixels are fit as N_obs=0 under `c_mode=selection`, driving
Q -> 0 there ("no hosts off-footprint"). Mitigated by the on-footprint
S-cut event set + matched beta; `diagnose_stage_a.py` audits each event's
PE probability mass off-footprint.

## Env

- Radial Q builder forks workers: `JAX_PLATFORMS=cpu`.
- gp3d/inference on the shared H100: `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

## Results (2026-08-09)

**Selection fit (union, m_lim=21, data-implied cubic K(z), 22.8M gals):**
M0hat = -20.3098 +/- 0.0002, sigma_M = 0.7144 (=> M0 = -21.16 at h = 0.6774,
at the r-band M*). K template from the catalog's own effective correction
(`M_APP - MAG_R - DM`): K(z) = 1.13465 z - 4.88963 z^2 + 8.58828 z^3
(rms 3.6 mmag; K(0.30) = 0.132 -- the fixed-color Chilingarian curve gives
only 0.047 because the observed median color evolves).

**Calibration:** n0 = 4.80e-3 Mpc^-3 (log10 = -2.3184, selection-corrected),
delta = +0.94, f_sky_occ = 0.620. Model C_eff(z<=0.30) = 0.98 is the
Gaussian-core estimand -- NOT comparable to Uchuu's 61% all-galaxy count.

**Stage A diagnostics** (`data/diagnostics_stage_a.json`): counts-Cbar vs
C_sel within +/-7% (coherent LSS dip z ~ 0.11-0.19, absorbed by Q);
north-south Delta M0hat = +0.0545 +/- 0.0004 (139 sigma, DR9 vs DR10);
off-footprint PE mass: 0 events > 50%, 5 events > 20% (worst GW240915, 44%).

**m_th map** (`data/mth_map_nside128.h5`): the footprint is uniformly
limited by the r=21 retention cut -- 0.07% of pixels shallower; depth
strata collapse, so the physical stratification is north/south.

**Parity gate (PASSED):** dev code on the pinned joint42 inputs: median
69.42 vs 68.94 (0.05 sigma), MAP within 1 grid step, TV distance 0.09.

**Real-event H0 grid scans** (44 events + matched betaS, 20-140):
| config | H0 median | 68% | note |
|---|---|---|---|
| complete (joint42-style) | 67.9 | [59.4, 75.8] | reference channel |
| per_pixel (dark_sirens) | 61.0 | [51.8, 71.7] | counts channel pulls low |
| sel (no Q) | 75.7 | [57.2, 87.8] | selection channel pulls high |
| selq_radial | 76.2 | [58.8, 88.0] | Q shifts +0.5 |
| selq_gp3d | 75.6 | [58.0, 87.6] | Q family consistent to <0.6 |
| sel_strat (N/S stratified) | 75.6 | [57.2, 87.8] | dlogL <= 0.005 vs pooled |
| sel_M0hat +/-5sigma | 75.7 | -- | leakage null: dlogL <= 5e-4 |

The **M*-H0 firewall holds on real data** (the +/-5sigma ablation).
**Interpretation caveat**: the betaS p_pass was built under the
complete-catalog weighting; pairing it with incomplete-universe channels
mixes conventions -- part of the sel-channel high pull may be that
mismatch. Recompute p_pass under the dark_sirens weighting before quoting
the selection-channel H0.

**LOO jackknife** (5 flagged off-footprint events, sel config): median
shifts -2.9 to +2.8 (<= 0.2 sigma), two-sided -- no dominating event.

**Core PRs:** #340 (K(z) template), #341 (multi-stratum fits + stratified
likelihood, stacked). Stratified channel validated end-to-end on real
data; stratified + prebuilt Q refused until the builder grows a
stratified base.

## Contents

| file | purpose |
|---|---|
| `common.py` | ZMAX pin, paths, provenance helper (import first) |
| `ingest_loa_rebuild.py` | loa flat -> darksirens raw schema (`data/desi_union_raw.h5`) |
| `fit_kcorr_poly.py` | cubic K(z) template from Chilingarian kcorr at median g-r |
| `calibrate_n0.py` | n0 = sum(w)/(f_sky_occ * Vc(z<=0.30)) for the Q builder |
| `diagnose_stage_a.py` | counts-Cbar alarm, dN/dz closure, N/S split, off-footprint audit |
| `run_*.sh` | run scripts (fit / builds / parity scan / real-event H0) |

## 2026-08-10: n0 recalibration ((1+z)^delta budget fix)

Q tables rebuilt at log10n0 = -2.3996 (the evo-fix calibration; see
n0_calibration.json, old values preserved in n0_calibration.json.pre_evo_fix
as SUPERSEDED -- do not quote). The pre-fix tables were deleted for quota
(3 x 3.5 GB); they are deterministic rebuilds (seed 22) from the pre-fix
calibration file if ever needed. Grid-scan results against the OLD tables
remain in data/h0_real/; recalibrated scans live in data/h0_recal/ (44-event)
and data/h0_full259/ (259-event, plain injections).
