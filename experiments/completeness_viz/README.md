# Completeness-correction closure experiment

Visual, end-to-end test of the darksirens completeness machinery against a mock
galaxy catalog with **known injected clustering**. The truth field is a mean-one
lognormal drawn from the *same* (sphere × z) low-rank GP family the gp3d
Q-builder fits (so closure is exact in principle, up to Poisson noise and prior
shrinkage), the survey selection is the fiducial dark-siren mock's (Gaussian
absolute magnitudes + apparent-magnitude limit + logistic z roll-off + realized
photo-z errors), and the completeness is then re-fitted *blind* from the
observed pixelated catalog with every available configuration.

## Pipeline

```
bash run_all.sh                 # or override: OUT=... NSIDE=... bash run_all.sh
```

1. `generate_clustered_mock.py` — truth field Q_truth(pix, z) (gp3d kernel by
   default; `--truth-kernel radial` for independent per-pixel fields), Poisson
   galaxy draw from `n0 · ΔV_c · Q_truth`, per-galaxy magnitudes/photo-z and
   selection reused from `scripts/mock_dark_sirens/generate_mock_data.py`.
   Writes `catalog_pixelated_nside_<n>.h5` (load_survey schema, `z_depth` attr)
   and `truth.h5` (truth field on voxels and on the exact package zgrid).
2. `fit_completeness.py` — fits/loads every configuration:
   - `homog` — kernel-ratio C(z|pix) only, homogeneous missing branch,
   - `delta_g` — legacy local-overdensity factor `1 + b·δ_g`,
   - `q_radial` — radial Poisson-lognormal Q table, MAP + Laplace members,
   - `q_gp3d` — 3-D low-rank Q table, MAP + members,
   - `q_radial_n0off` — ablation with log10n0 mis-set by +0.3 dex (shows the
     density-miscalibration artifact the builder docs warn about).
   For each: completion curves (`C_eff`, `dN_miss`, …) and the assembled
   redshift prior p(z|pix) on 12 diagnostic pixels under both `conditional`
   and `field` sky weighting.
3. `plot_completeness.py` — figures under `output/plots/` plus
   `closure_summary.json` with soft PASS/FAIL metrics.

## Figures

| File | Shows |
|---|---|
| `closure_scatter.png` | fitted vs truth logQ voxel-by-voxel (r, slope, rms per config; the pass/fail gate) |
| `c_calibration.png` | fitted kernel-ratio C(z) vs the analytic generative selection C_true(z) |
| `dndz_closure.png` | observed + missing budget vs the true total dN/dz |
| `q_z_curves.png` | Q(z) at truth-extreme pixels: truth vs radial/gp3d MAP + member bands |
| `q_sky_slices.png` | mollweide logQ maps at three redshift slices + gp3d residual |
| `slabs_{x,y,z}.png` | comoving Cartesian midplane slabs of truth/fitted logQ with observed-galaxy overlay |
| `dnmiss_and_prior_pix*.png` | component-separated observed/missing branches and assembled p(z|pix), conditional vs field |

## Results (reference run: nside 16, zmax 0.5, 277k galaxies, corr 800 Mpc, seed 42)

- **Kernel-ratio C is essentially exact at the aggregate level**: the summed
  observed KDE matches the matched-smoothed expectation to ~0.3%
  (`c_calibration.png`). Per-pixel C is noisy and clip-asymmetric (band).
- **The homogeneous missing branch anti-correlates with the true missing
  density** (anomaly r ≈ −0.43, `closure_scatter.png`): per-pixel C absorbs
  the clustering, so an overdense pixel gets a *smaller* (1−C) and fewer
  predicted missing galaxies — backwards. In the assembled prior the observed
  and missing branches then largely cancel the catalog's clustering
  (`total_density_scatter.png`, homog slope 0.25).
- **δ_g restores the structure best** (total-density slope 0.77, r 0.68) and
  closes the total dN/dz budget to 0.6%. **Q gp3d** is intermediate (slope
  0.29): the Q table is by construction only the *sub-smoothing residual* on
  top of C (module docstring caveat), so it cannot restore structure C already
  absorbed — `q_scatter.png` slope < 1 is expected, not a failure.
- **Q radial is shot-noise dominated** at these densities (fitted logQ noise
  ~1) and its lognormal noise inflates the missing budget by ~55% (Jensen);
  its member band (89% coverage) at least brackets the truth, while the gp3d
  Laplace band is far too narrow (5%).
- The homogeneous/gp3d totals sit ~13% high near depth — the realization's
  sky-mean Q ≈ 0.89 there (sample variance) that a homogeneous n0 cannot know.
- `q_radial_n0off` (+0.3 dex): the density miscalibration is visibly absorbed
  into spurious logQ structure, as the builder docs warn.

## Findings on the shipped hyperparameters

- The gp3d field's z-representation is 6 inducing nodes over z ∈ [0, 3]
  (spacing 0.277 in ζ = log1p z). The **fixed** `lss_corr_length_mpc = 50`
  maps to ls_z ≈ 0.01 — ~30× below the node spacing — so at the shipped value
  the gp3d family can only represent radial structure as thin shells at the
  node redshifts, and a 50 Mpc (or even 800 Mpc) truth field is nearly
  invisible to it (measured slope of fitted vs truth logQ ≈ 0.04).
- The builders have **no CLI knob** for `lss_corr_length_mpc`;
  `fit_completeness.py --builder-corr-mpc truth` (default) overrides the
  builder fiducial factory so the closure test conditions the fit on the
  injected correlation length. `--builder-corr-mpc default` reproduces the
  shipped behavior.
- At mock-like densities (~1e-5 Mpc⁻³) a 50 Mpc correlation cell holds ≪ 1
  galaxy, so the per-pixel radial fit is shot-noise dominated regardless.

## Pitfalls encoded here

- The package `zgrid` is read once at import: never set `DARKSIRENS_ZMAX`
  differently between steps; Q tables and the truth export share the exact grid.
- `log10n0` passed to the builder and to `SurveyParams` must be the injected
  density (printed by the generator).
- The pixelated file must be float64 with the standard padding (z=100, dz=1,
  w=0) — reused verbatim from the mock generator — and must carry the
  `z_depth` attr (the stock writer omits it; we write it).
- Q tables are global-indexed (`lss_completion_indexing=2`); field-mode priors
  need the `build_field_*_inputs` products attached to the EMCatalog.
- Q tables and the legacy `delta_g` are mutually exclusive in the likelihood;
  here they live on separate EMCatalog instances for diagnostics only.
