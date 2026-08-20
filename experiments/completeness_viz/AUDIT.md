# Completeness-machinery audit (2026-08-07)

Multi-agent audit (12 agents: 3 code readers, 4 literature researchers, 3
designers, 1 novelty assessor, 1 adversarial critic). Full agent outputs in
`audit_2026-08-07_raw.json`. Triggered by the closure-experiment failures
recorded in `README.md` / `output/plots/closure_summary.json`.

## Root cause (confirmed in code)

`completion.py` computes `C(z|pix) = clip(dN_obs_s(z|pix) / dN_exp_smooth(z), 0, 1)`
with a **per-pixel numerator** (the observed-galaxy KDE) and an **isotropic
denominator** (`n0·apix·g(z)`, same curve for every pixel). So the estimator is
really `C_sel(z) · (1 + δ_obs(pix,z))` clipped to [0,1]:

- C absorbs the angular clustering of the observed galaxies; overdense complete
  pixels read C > 1 and are clipped, silently discarding their excess.
- `(1 − C)` therefore anti-tracks the true missing density → the measured
  anti-correlation (r = −0.43, slope = −0.62) is structural, not noise.
- Q is fit to the *same counts residual to C* (`base_v = C·dN_exp`), so it can
  only ever recover sub-smoothing structure (slope 0.29 ceiling) — and the
  builder fits **occupied pixels only**, so deep voids (true Q ≈ 0.2) never
  even enter the fit (fitted at Q ≈ 1).
- This is the thinning/Cox-process degeneracy: a free selection surface and a
  free density field are not jointly identifiable from one catalog's counts
  (preferential-sampling literature: Diggle et al. 2010).

Secondary defects, each with an identified mechanism:

- **Depth-edge spike**: galaxy kernels renormalize on [0, zmax] but
  `dN_exp_smooth = S @ dN_exp` integrates the full grid, so C dips just below
  the edge and `(1−C)·dN_exp` spikes (completion.py `_precompute_grids` /
  `_kde_dndz_obs`).
- **Radial-Q +55% budget inflation**: the table is the Laplace posterior-mean
  E[Q] = exp(b·s − shift + ½b²var_post); var_post is largest where data are
  sparse → spatially varying exp(σ²/2) Jensen bias (standard LGCP result,
  Møller et al. 1998).
- **gp3d slope 0.04 at shipped hyperparameters**: fixed
  `lss_corr_length_mpc = 50` → ls_z ~30× below the 6-node inducing spacing in
  ζ = log1p z; low-rank GP posteriors collapse to the prior when node spacing ≫
  lengthscale (Burt et al. 2019, arXiv:1903.03571). No CLI knob exists.
- **5% member-band coverage**: mostly the prior-collapsed field at wrong fixed
  hyperparameters, compounded by single-Laplace overconfidence in latent
  Poisson models (Taylor & Diggle, arXiv:1202.1738; fix = INLA-style
  marginal-likelihood hyperparameter learning, Rue et al. 2009).

## What the literature says

- **Imaging surveys never estimate completeness from the observed counts**:
  injections (Balrog arXiv:2012.12825, 2501.05683; Obiwan 2007.08992,
  2405.16299), survey-property template regression debiased on mocks
  (Weaverdyck & Huterer 2007.14499; Rezaie 1907.11355), or magnitude limits —
  precisely so the estimate cannot absorb clustering.
- The counts-based per-pixel construction exists only in the dark-siren
  literature: DarkSirensStat's cone/mask completeness (Finke et al.
  2101.12660), which *acknowledges* the clustering-absorption pathology and
  mitigates by coarse-graining. gwcosmo (Gray et al. 2111.04629, 2308.02281)
  is the clustering-safe alternative: per-pixel m_th + Schechter LF — counts
  enter only through the magnitude distribution.
- **Clustering-informed completion competitors**: Dalang & Baker 2310.08991 and
  Dalang, Fiorini & Baker 2410.03275 (variance completion, gwcosmo-ready);
  Leyde, Baker & Enzi 2409.20531 (Bayesian field-level lognormal
  reconstruction + magnitude-function inference — the closest to our Q).
- Cross-correlation line (Oguri 1603.02356; Mukherjee et al. 2203.03643, …)
  bypasses completion entirely.

## Novelty verdict (blunt)

Genuinely novel and defensible:
1. **The jointly-sampled, selection-consistent hierarchical likelihood** with a
   clustering-informed completion field: survey block {log10 n0, δ, b_miss,
   σ_kde} + Q members sampled/marginalized with the identical completed prior
   in both the per-event term and the selection integral; cube-free member
   marginalization is a real implementation contribution. No published
   pipeline does this (gwcosmo/ICAROGW precompute fixed LOS priors; Leyde
   stops short of the GW likelihood).
2. **The conditional-vs-field estimand analysis** (per-pixel normalization =
   radial-only estimand; K=1 field mode reduces to relative angular weighting;
   K≥2 mixture weights = host fractions).
3. The matched-kernel smoothing operator (constant completeness is an exact
   fixed point of the ratio) — a refinement, not a headline.
4. The closure experiment itself: a citable quantitative negative result
   against counts-based per-pixel C (the DarkSirensStat lineage).

Not novel (do not claim): in/out mixture (Gray/Gair), counts-based per-pixel
completeness (Finke), 1+b·δ_g (= multiplicative completion; Dalang & Baker
already did the principled interpolation), Poisson-lognormal + Laplace
(textbook LGCP / BORG / Leyde), low-rank sphere×radial GPs.

Referee risk if published as-is: the legacy δ_g factor outperforms the
flagship GP machinery on our own closure test.

## Recommended path (post-adversarial-critique ranking: Merged > C > A > B)

Three proposals were designed and attacked (full text in the raw JSON):
A = aggregate counts-C̄ + full-overdensity Q; B = minimal surgical fixes;
C = joint/two-step model with magnitude-based C_sel. A and C share one
architecture — radial-only budget × mean-one full-field Q ("C says how much,
Q says where") — differing in the budget estimator (counts vs magnitudes).
C's magnitude channel is what breaks the thinning degeneracy and survives
anisotropic-depth catalogs (GLADE+); A's counts aggregate remains the free
cross-check (they already agree to 0.3% on the mock).

- **Stage 0 (days, architecture-independent bug fixes)**
  - Truncate the expected side at z_depth: `dN_exp_smooth = S @ (dN_exp·1[z≤z_depth])`
    (kills the depth-edge spike).
  - Per-z mean-one renormalization of Q tables and members under weights
    w_p = (1−C_p)·dN_exp (kills the 55% Jensen budget inflation by
    construction; record the removed monopole in provenance).
  - CLI knobs for `lss_corr_length_mpc`, `lss_sigma`, z-node count/range,
    M_sph; hard error when node spacing > lengthscale; z_node_hi := zgrid[-1].
- **Stage 1 (~week)**: opt-in `c_mode="aggregate"`: one sky-aggregate C̄(z)
  (validated to 0.3% already), Q retargeted to the full overdensity, empty
  pixels included in the fit (voids!), fail-closed provenance so per-pixel-base
  tables cannot be consumed in aggregate mode. Gates: missing-branch r > 0.6
  with slope ~ +1; voids recovered < 0.5.
  Known caveats from the critique: C̄ ∝ 1/n0 with a single global clip makes
  the likelihood kinked in n0 (watch sampler behavior); isotropy fails on real
  heterogeneous catalogs → stratify.
- **Stage 2**: magnitude-based budget where magnitudes exist: plumb APP_MAG
  through generator/pixelate/io (mock already draws magnitudes and even the
  analytic per-galaxy selection; the survey HDF5 currently drops them), fit
  C_sel(z;θ) = Φ((m_lim−M0−DM(z))/σ_M) from the per-galaxy magnitude
  likelihood (independent of Q by thinning), per-stratum for anisotropic
  depth; counts-C̄ stays as misspecification alarm. θ must be sampled (or
  h-scaled) inside the GW likelihood — fixing it at fiducial cosmology
  re-imports the M*–H0 degeneracy — and must enter the selection integral.
- **Stage 3 (hardening)**: closure variants with anisotropic depth and
  50–200 Mpc truth lengthscales (the 800 Mpc isotropic Gaussian-LF mock is
  friendly to every proposal); marginal-likelihood hyperparameter grid with
  mixture-of-Laplace members (fixes coverage honestly); scale M or move to
  spherical-harmonic features (Dutordoir 2020) if short lengthscales needed.

## Cross-cutting invariant

Any newly sampled selection parameter must enter μ(Λ) via the pinned
selection-prior model in `likelihood/core.py`, or the prior can reshape with
no detectability penalty (the self-calibration property is the package's most
valuable asset — never bypass it).
