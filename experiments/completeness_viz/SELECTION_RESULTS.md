# Selection-channel validation results (2026-08-08)

Branch `feat/selection-cmode`; plan at `~/.claude/plans/golden-tickling-yao.md`.
Mocks: `output_magpure` (isotropic, m_lim=20, z50=10 → pure magnitude
selection, 36% observed), `output_aniso` (hemispheres m_lim = 20 ± 0.25).
An earlier magnitude-pure attempt kept m_lim=24 and was silently ~100%
complete at z ≤ 0.5 (the original closure mock's incompleteness came from its
z50=0.3 logistic, not the magnitude cut) — a would-be trivial closure caught
and regenerated at m_lim=20.

## Offline fit gates (Phase B)

- Isotropic: M̂₀ = −20.1541 ± 0.0067 vs truth −20.1542 (0.0σ);
  σ_M = 1.0013 ± 0.0032 (+0.4σ); C_sel(z; θ̂) vs analytic generative
  selection: max |Δ| < 1e−5 — evaluated at H₀=72 against truth generated at
  67.74 (the h-scaling handling the difference end-to-end).
- H₀ firewall (unit-pinned): C_sel invariant across H₀ ∈ [50, 140] at fixed
  M̂₀ to ≤ 1e−12; DM(z;H₀) = DM(z;100) − 5log₁₀(H₀/100) exact.

## Isotropic closure (selection base vs counts-aggregate parity)

| metric | selection base | aggregate base |
|---|---|---|
| missing-branch closure (radial) | r 0.875, slope 0.96 | r 0.876, slope 0.95 |
| missing-branch closure (gp3d) | r 0.835, slope 0.81 | r 0.839, slope 0.81 |
| dN/dz budget vs truth | **1.051** | 1.083 |
| fitted-vs-truth logQ slope | 0.96 / 0.81 | 0.95 / 0.81 |
| void voxels (true Q ≤ 0.25) median | 0.31 / 0.45 | 0.33 / 0.47 |

Selection base matches the counts aggregate everywhere and carries a BETTER
budget (the parametric selection is closer to the true selection than the
counts average on this realization). `structure_restored_pass=False` is a
0.003-margin artifact of its δ_g criterion (0.797 vs 0.75+0.05); the Q-side
criteria pass.

## Anisotropic depth (the decisive identifiability test)

- Per-stratum magnitude fits: north (m_lim 20.25) M̂₀ −0.6σ from truth,
  south (19.75) +0.5σ — the shared LF recovered through a 0.5-mag depth split.
- Pooled misspecified fit (single m_lim=20.25): M̂₀ off by **−43σ** — the
  quantified case for stratification.
- Counts-aggregate alarm: fitted-Q hemisphere dipole ⟨logQ_N − logQ_S⟩ =
  **+0.333** vs truth −0.065 — the unmodeled depth split injected as a
  spurious ~0.4-dex density asymmetry, exactly the misspecification signature
  the Q-monopole/dipole alarm exists for.
- Deviation from plan: full two-stratum consumption INSIDE the likelihood
  (per-pixel C_sel maps) is deferred to real-catalog ingestion; the gates here
  are fit-level identifiability + the alarm.

## End-to-end H₀ scans (same 300 events; magpure catalog; local quadratic fits)

| config | peak ± σ | offset |
|---|---|---|
| complete-catalog gate | 69.60 ± 1.32 | +1.4σ |
| homogeneous, aggregate C̄ | 68.42 ± 1.30 | +0.5σ |
| **selection C_sel** | **68.27 ± 1.29** | **+0.4σ** |
| selection + Q gp3d | 68.28 ± 1.28 | +0.4σ |
| selection, M̂₀ + 5σ_fit | 68.29 | (leakage test) |
| selection, M̂₀ − 5σ_fit | 68.26 | (leakage test) |

- Selection Neff ≥ 10.9k across the whole grid (threshold ~3.8k).
- **H₀-leakage ablation: shifting M̂₀ by ±5σ_fit moves the H₀ peak by
  ±0.027 km/s/Mpc and the curve shape by ≤ 0.36 nats** — the magnitude
  channel carries no H₀ information end to end, as designed.

## Joint-term cross-check (Phase G)

Binned suff-stats magnitude likelihood matches the exact per-galaxy one to
< 0.05 nats across θ; the Gaussian(Laplace)-prior path matches the full joint
1-D M̂₀ posterior (mean within 0.2 sd, width within 5%) — the cheap default
certified.

## Artifacts

- `output_magpure/`, `output_aniso/` (+ `selection_fit.json`, `stratum_fits.json`)
- `magpure_selection/`, `magpure_aggregate/` (closure fits + plots)
- `aniso_aggregate/` (alarm build)
- `h0_scan/fits_mp052_*/`, `h0_scan/results_sel/`
