# Implementation plan: aggregate completeness budget (Stage 0 + Stage 1)

Branch: `feat/completeness-aggregate-budget` (from master ae9cd41). Validate on
the closure experiment BEFORE any PR to master. No end-to-end H0 inference
required; gates are unit tests + the closure metrics + visual prior checks.
Rationale and literature grounding: `AUDIT.md`.

## Target architecture

"C says HOW MUCH is missing (radial-only budget); Q says WHERE the missing
galaxies go (mean-one spatial field)." Legacy per-pixel behavior stays the
default and bit-identical; everything new is opt-in and provenance-stamped.

## Work items (implemented sequentially, one commit each)

### S0a — depth-edge truncation (completion.py)
When `z_depth` is a concrete float, truncate the expected side identically to
the observed side: `dN_exp_smooth = S @ (dN_exp · 1[z ≤ z_depth])`, in both
`_precompute_grids` and the field-mode recompute. A constant-completeness
survey then passes through the ratio exactly at the edge (mirrors the existing
z=0 boundary treatment). `z_depth=None` path must stay bit-identical
(pinned by tests/test_completion_depth.py::test_z_depth_none_bit_identical).
New unit test: no C dip / no dN_miss spike within 2σ_smooth of the edge for a
constant-completeness synthetic survey.

### S0b — per-z mean-one Q renormalization (lognormal_completion.py)
After building any Q table (radial or gp3d), renormalize per z-bin over the
fitted footprint: `logQ_p(z) -= log[Σ_p w_p(z)·Q_p(z) / Σ_p w_p(z)]` with
`w_p(z) = (1−C_p(z))·dN_exp(z)` (build-time fiducial). Apply the same
per-member renormalization to Laplace members. Q becomes a pure redistribution
field: the missing budget with Q equals the homogeneous budget identically
(kills the measured +55% Jensen inflation by construction). Default ON for new
tables; removed monopole curve + a boolean recorded in HDF5 attrs; loader
tolerant of legacy tables (attr absent → not renormalized, warn).
The invariant is defined over the FITTED FOOTPRINT (the rows the weights
cover): occupied pixels for the per-pixel base, the whole sky for the
aggregate base. The gp3d "borrowing halo" — nonzero logQ evaluated onto
EMPTY pixels outside the per-pixel fitted footprint — deliberately sits
outside the identity (far pixels read Q = 1 and must not absorb the
footprint's monopole), so the survey-TOTAL budget including the halo is not
exactly invariant; only the fitted-footprint sum is.
New unit test: budget invariance Σ_p dN_miss(Q) == Σ_p dN_miss(homog) per z
over the fitted footprint.

### S0c — hyperparameter knobs + resolution guard (CLI + types)
CLI flags on the builder: `--lss-corr-length-mpc`, `--lss-sigma`,
`--gp3d-nz-nodes`, `--gp3d-nsph-nodes`, `--gp3d-z-node-hi` (default: package
zgrid max, not the hardwired 3.0). Hard error (not warning) at build time when
inducing-node spacing in ζ exceeds ls_z (the shipped 50 Mpc default is ~30×
under-resolved → fitted slope 0.04; Burt et al. 2019 collapse). Unit test for
the error path and for flag plumbing.

### S1 — aggregate C mode (types + completion.py + builder)
Opt-in `c_mode="aggregate"` (legacy `"per_pixel"` default):
- `C̄(z) = clip( Σ_p dN_obs_s(z|p) / (N_pix_total · dN_exp_smooth(z)), 0, 1 )`
  with `N_pix_total = round(4π/apix)` (occupied + empty). One curve per
  proposal, broadcast in place of per-pixel C; field-mode normalizer uses the
  same curve. Closure experiment already validated this aggregate to 0.3%.
- Builder in aggregate mode: base becomes `C̄·dN_exp` and the fit INCLUDES
  empty pixels (N_obs = 0 rows) so voids are informative (fixes Q≈1 voids).
- Provenance: `c_mode` attr written to Q HDF5; fail-closed at attach time
  (a per-pixel-base table consumed in aggregate mode, or vice versa, is a
  hard error — the two targets differ by the whole clustering signal).
- Known caveat to note in code docs: C̄ ∝ 1/n0 through a global clip → watch
  sampler smoothness in n0 (not gated here; flagged for the inference stage).

### S1h — closure-harness support (experiments/completeness_viz)
`fit_completeness.py`: add `--c-mode {per_pixel,aggregate}` and pass-through
for the new builder knobs; keep current defaults reproducing the legacy run
exactly. `run_all.sh`: env overrides for the new flags.

## Validation gates (before PR)

1. Fast subset green: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu
   python -m pytest -q $(grep -v '^#' tests/fast_subset.txt)` plus the
   completion/lss test files.
2. Legacy regression, two parts: (a) bit-identity vs master at DEFAULT
   settings (z_depth=None, unset c_mode) is pinned by unit tests
   (test_z_depth_none_bit_identical, test_per_pixel_default_bit_identical);
   (b) a per-pixel-mode closure rerun quantifies the S0-only deltas vs the
   2026-08-07 reference — expected: budget closure improves (radial +55% → ~0
   over the fitted footprint), edge spike gone, anti-correlation UNCHANGED
   (that defect is per-pixel C itself; S1 fixes it). Bit-identity is NOT
   expected here because S0a/S0b intentionally change per-pixel outputs when
   z_depth is set.
3. Aggregate closure (rebuilt Q tables, gp3d solve on the H100;
   radial builder stays JAX_PLATFORMS=cpu because it forks workers):
   - aggregate C̄ vs generative selection ≤ 0.5% (median |ratio−1|);
   - missing-branch spatial closure: r > 0.6 with POSITIVE slope (vs −0.62);
   - assembled total-density slope ≥ 0.77 (beat legacy δ_g);
   - dN/dz budget closure |Δ| < 2% for radial AND gp3d (radial was +55%);
     measured over the FITTED footprint (see S0b — the gp3d empty-pixel
     borrowing halo is outside the renormalized identity by design);
   - fitted-vs-truth logQ slope in [0.8, 1.2] for gp3d (was 0.04 shipped /
     0.33 truth-corr override);
   - void pixels (true Q ≈ 0.2) recovered at Q < 0.5;
   - no dN_miss excess > 10% within 2σ_smooth of z_depth;
   - visual: q_z_curves / dnmiss_and_prior / closure_scatter regenerated and
     eyeballed (member-band coverage reported, not gated).
4. PR to master only after all gates pass; PR body cites AUDIT.md findings
   and the before/after closure metrics.

## Out of scope (later stages, see AUDIT.md)

Magnitude-based C_sel (APP_MAG plumbing, θ sampled in the likelihood),
marginal-likelihood hyperparameter grid / mixture-of-Laplace members,
anisotropic-depth closure variants, spherical-harmonic features.
