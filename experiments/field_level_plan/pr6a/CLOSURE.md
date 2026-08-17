# PR-6a mock closure — Tiers A–D (PLAN §6.2)

nside 16, `DARKSIRENS_ZMAX=1.5`, `H0_true = 67.74`, `Om0 = 0.3075`, `b_gal = b_GW = 1`.
Scripts and data live in `experiments/field_level_plan/pr6a/`. **No production 259-event
run was launched**; everything here is fixture-scale (CPU for Tier A, one H100 for
Tiers B–D).

Guard convention throughout: **PR-0's clean arm** — `selection_neff_soft_guard=False`,
`max_likelihood_variance=1e6`, Vitale's `5 N_obs` floor retained. On this mock the hard
criterion does not fire (`pe_variance_sum = 0.0101–0.0110`, injection
`Neff_fiducial = 37111` against 60 events), so the choice is not load-bearing here — but
it is stated, because on the production line PR-5b measured the soft-guard convention to
report σ 610×–34206× larger.

---

## Verdicts

| tier | gate | measured | verdict |
|---|---|---|---|
| **A** field recovery | slope `1.00 ± 0.05` | **0.99908 ± 0.00433** | **PASS** |
| **A** | Pearson `r > 0.90` | **0.99945 ± 0.00012** | **PASS** |
| **A** | `‖ξ̂−ξ_true‖/√M` within 20% of `√(tr H⁻¹/M)` | ratio **1.0236 ± 0.0591** | **PASS** |
| **B** H0 closure | `H0_true` in the 90% CI, every arm | 0 of 4 arms | **FAIL** |
| **B** | latent vs table within `0.3 σ` | **1.571 σ** | **FAIL** |
| **B** | latent CI width ≥ table CI width | 22.30 vs 32.64 | **VACUOUS** (see §B.3) |
| **C** coverage, latent arm | KS `p > 0.05` | **p = 0.0111** (`D = 0.3198`, n = 24) | **FAIL** |
| **C** | median `H0` bias `< 0.2 σ` | **−0.1437 σ** | pass (but see §C: unresolvable at n = 24) |
| **C** | no realization outside the 99% band | **9 of 24** | **FAIL** |
| **D** (i) `f_p` perturbed by 0.104 | bias `< 0.5 σ` | paired **+0.059 ± 0.040 σ** | **PASS** |
| **D** (ii) 5% fibre-assignment proxy | `< 0.5 σ` | paired **+0.025 ± 0.006 σ** | **PASS** |
| **D** (iii) `ls_ang` wrong by 2× | `< 0.5 σ` | paired **+0.002 ± 0.026 σ** | **PASS** |
| **D** (iv) non-Gaussian tail | `< 0.5 σ` | **+0.390 ± 1.395 σ** (unpaired) | pass, but consistent with anything |

Tier A is the tier that tests the object PR-6a adds — the count-channel field solve and the
`logQ` it generates — and it passes cleanly. Tier D's four misspecifications, at PLAN's own
measured amplitudes, move `H0` by at most 0.06 σ where the comparison is paired. **Tiers B
and C fail on DISPERSION, not on bias**: the median `H0` is centred (−0.14 σ over 24
realizations) but the realization-to-realization spread of the median is **2.6× the
uncertainty the posterior quotes**, and a bisection (§B.4) puts that outside the latent
seam. A second, separate and fully deterministic failure — 16 of 16 runs — is the
no-`f_p` arm, which is the only configuration a Q table can occupy (§S-3).

---

## Tier A — field recovery, no GW (CPU, nside 16) — **PASS**

`tier_a.py` → `tier_a.json`. Command:

```
JAX_PLATFORMS=cpu python tier_a.py --n-real 8 --out tier_a.json      # 2 min
```

**Setup.** nside 16; the footprint is the real DESI depth map
(`experiments/desi_ingest/data/mth_map_nside128.h5`) degraded to nside 16, keeping the
1854 pixels (60.4% of the sky) whose nside-128 children are ≥ 90% covered. Their `f_p` has
mean **0.8722**, sd **0.0450**, min **0.6136**. (At the map's native nside 64 the same file
gives 17009/49152 zero pixels — 34.6% of the sky — and, over the 32143 non-zero pixels,
`f_p` mean 0.8543 sd 0.1087, i.e. `masked_frac` mean 0.1457 sd 0.1087: PLAN §1.2's
0.1368/0.1039 recovered from the artifact. Degrading averages 64 children, which is why
the nside-16 sd is 0.045 and not 0.109 — the native scatter is what Tier D-i re-injects.)

Basis: `factored-v1`, `M_sph = 64`, `M_z = 5` → **M = 320**, `ls_sph = 0.5`
(`√(4π/64) = 0.443 ≤ 0.5`, the builder's resolution guard), `ls_z = 0.10`
(`log1p(0.3)/4 = 0.0656 ≤ 0.10`), `z_depth = 0.30`, 12 shells, `σ_z = 0.023`, `b_gal = 1`.
Counts: Poisson on a 400-node fine `z` grid at intensity
`f_p · base(z) · exp(b_gal f(p,z))`, then thinned into shells by the same Gaussian
photo-z mass `W` models — so PLAN §0.5 D1's within-shell linearization is live, not
assumed away. **1.0998e6 galaxies**, 1854 × 12 = 22248 voxels, 16215 ± 801 with `N ≥ 10`.

**The gauge.** The count channel is shell-total-conditioned, so `π_pg` is invariant under
`η_pg → η_pg + c_g`: the per-shell monopole carries zero information, the MAP shrinks it to
the prior mean while `ξ_true` keeps its drawn value. The primary slope is therefore computed
on the gauge-fixed field (each shell's footprint mean subtracted from both sides — exactly
the quotient the likelihood sees); the raw slope is reported beside it.

| statistic | gate | measured (8 realizations) |
|---|---|---|
| slope, gauge-fixed, `N ≥ 10` | `1.00 ± 0.05` | **0.99908 ± 0.00433** |
| Pearson `r`, gauge-fixed | `> 0.90` | **0.99945 ± 0.00012** |
| `‖ξ̂−ξ_true‖/√M` | — | 0.56948 ± 0.03164 |
| `√(tr H⁻¹/M)` | — | 0.55645 ± 0.00780 |
| ratio | within 20% | **1.02365 ± 0.05908** |
| slope, RAW (monopole not removed) | reported | 0.97370 ± 0.04394 |
| slope, exactly-representable draw | reported | 0.99921 ± 0.00088 |

The exactly-representable arm (counts drawn from the fit operator's own `phi_shell`, so the
truth is inside the model class) gives 0.99921; the fine-grid arm gives 0.99908. **The
within-shell linearization costs 1.3e-4 in slope** at this geometry — measured, not assumed.

**Prior-collapse contrast.** PLAN §6.2 and §11 quote "the 0.04 prior-collapse signature"
(§11: "any argument that reaches for a smaller `M` is reaching for the prior-collapse regime
the hard radial guard exists to refuse (measured fitted-vs-truth `logQ` slope 0.04)"). It is a
rank/scale signature, so the control is the rank-starved fit, and the count ladder is
reported next to it so both axes are on the record:

| control | slope (all voxels) |
|---|---|
| same 1.1e6 galaxies fitted with `M = 12 × 2 = 24` (`ls_sph = 1.5`, `ls_z = 0.30`) | **0.7808** (`r = 0.801`) |
| `N = 1.1e6` (the tier) | 0.99998 |
| `N = 1.1e5` | 0.99730 |
| `N = 1.1e4` | 0.97064 |
| `N = 1.1e3` | 0.85424 |
| `N = 1.1e2` | 0.58924 |
| `N = 11` | 0.20102 |

**I could not reproduce 0.04 on either axis.** The rank-starved fit reaches 0.78 and the
count ladder bottoms at 0.20 with eleven galaxies in the whole survey. The inherited 0.04 is
from a different configuration (the shipped radial builder on the `completeness_viz` mock);
it is quoted in PLAN as a measured number and it is not reproduced here. The contrast the
tier needs is still decisive — 0.999 against 0.78/0.20 — but the specific figure 0.04 should
not be carried forward as if this tier confirmed it.

---

## Tier B — single-realization H0 closure (GPU, nside 16, 60 events) — **FAIL**

`make_mock.py` + `build_anchor16.py` + `arms.py` + `tier_b.py` → `tier_b_rb.json`.

**The mock** (`data/rb`, seed 7001, `truth.json` carries every number):
`n0 = 5e-5 Mpc⁻³`, `δ = 0`, universe to `z = 0.60`, catalog truncated at the declared
depth `z_depth = 0.30` (`completion.py:1588`: "beyond `z_depth` completeness is 0"),
2.650e6 complete galaxies, **192757 catalogued**, 60 detected events
(median `z = 0.1177`, max 0.2083, **all** below the depth), PE `dL` max 1551 Mpc,
65791 detected injections of `Ndraw = 3e6` (`Neff_fiducial = 37111`), `σ_z = 0.023`
(matching `cli/build_latent_field.py:156`'s frozen `W`), `m_lim = 20`,
`M0hat = −20.1542`, `σ_M = 1`. GW hosts are drawn uniformly from the COMPLETE catalog
over the whole sky and the whole universe, so masked, faint and beyond-depth hosts are
all reachable and all carry the field — PLAN §6.1's extension (ii). The injection set
comes from the same `grids`/`pop`/`MeasurementConfig` objects as the detection rule and
the posteriors (`mock-data-dag`).

Anchor: `build_anchor16.py`, 1854 pixels, 183267 galaxies in 12 **equal-comoving-volume**
shells (see §B.5), `grad_inf = 5.79e-10` (P6 passes), `M_draw = 8`,
`sha256 = cc834af0627dc2f5…`.

### B.1 The four arms

| arm | LSS treatment | `f_p` | H0 median | 68% CI | 90% CI | σ | CDF at truth | ms/eval |
|---|---|---|---|---|---|---|---|---|
| `latent_off` | none | yes | 80.95 | [74.37, 88.18] | [70.35, 93.36] | 6.90 | 0.0184 | 187 |
| `latent_off_nofp` | none | **no** | 93.59 | [85.56, 101.95] | [80.72, 107.94] | 8.20 | 0.0002 | 127 |
| `table` | radial Q, 8 members | **no** | 92.55 | [82.88, 101.42] | [77.77, 110.41] | 9.27 | 0.0010 | 126 |
| `latent` (PR-6a) | latent, `M_draw = 8`, marginalized | yes | 80.02 | [73.63, 86.99] | [69.72, 92.02] | 6.68 | 0.0235 | 224 |

`H0_true = 67.74` is outside the 90% CI in **all four** arms. Gate 1 fails.

Latent vs table: `|80.02 − 92.55| / ½(6.68+9.27) = **1.571 σ**` against a `0.3 σ` gate.
Gate 2 fails — but see §B.2 before reading that as a statement about the field.

### B.2 The latent-vs-table comparison is not a field comparison — it cannot be

`inference/loaders.py:1021` **refuses** `--per_pixel_completeness` together with a Q table
("needs an f_p-weighted empty-pixel Q budget that is not implemented; drop one of them"),
while `factory.py`'s guard 6 **requires** it in latent mode. **No configuration exists in
which the table arm and the latent arm differ only by the field.** The table arm must
also drop PR-2's `C_p = f_p C(z)`.

`latent_off_nofp` exists to split that. Decomposed (each in units of the mean σ):

| comparison | what it isolates | shift |
|---|---|---|
| `latent` vs `latent_off` | the FIELD, at identical `f_p` treatment | **0.136 σ** |
| `table` vs `latent_off_nofp` | the table's field, at identical (no-`f_p`) treatment | **0.118 σ** |
| `latent_off` vs `latent_off_nofp` | the `f_p` channel alone | **1.674 σ** |

So the 1.571 σ "latent vs table" disagreement is **91% the `f_p` channel and 9% the
field**. PLAN §6.2's second acceptance criterion, as written, is not measurable on a
footprint-limited survey with the shipped guards.

### B.3 The CI-width criterion is VACUOUS, and the reason is a missing implementation

PLAN §6.2: "**latent-on CI width ≥ table CI width** — now a valid check because §3.4
propagates `b_gal`". §3.4 specifies

```
Cov(xi) = H^-1 + s_b^2 (d xi_hat/d b)(d xi_hat/d b)^T,   s_b^2 = [-d^2 log p_count/db^2]^-1
```

Grepped on this branch: `latent_counts.laplace_draws(xi_hat, H_chol, n_draw, key)` takes
`H_chol` and nothing else; its only production caller,
`cli/build_latent_field.py:193`, passes the bare `L`; and **`s_b` — the profile curvature
§3.4 defines — does not exist anywhere under `darksirens/`.** The `b_gal` column of
`sensitivity_S` IS built (`dgrad_db` → `sensitivity`, `build_latent_field.py:183`), IS
stored, and IS read into `LatentQPlan.S` by `latent_q.load_latent_plan`, but no consumer
inflates the draw covariance with it. **The member ensemble carries `H^-1` alone.**

Measured anyway: 90% width 22.30 (latent) vs 32.64 (table) — latent is NARROWER. Reporting
that as a failed gate would be wrong twice over: the widths differ mostly through the `f_p`
channel (§B.2), and the propagation the gate is predicated on is not implemented. **Marked
VACUOUS, not failed.**

### B.4 Where the failure actually is — a bisection

Five configurations, homogeneous truth (`field_scale = 0`, so the field is exactly 1 and
cannot contribute), three seeds each, latent off. `bisection.txt` has every row.

| rung | sky | universe | mask | model has `f_p` | CDF at truth (3 seeds) |
|---|---|---|---|---|---|
| V0 | full | `z_univ = z_depth = 0.30` | none | n/a | 0.355, 0.870, 0.768 |
| V1 | full | `z_univ = 0.60 > z_depth` | none | n/a | 0.172, 0.143, 0.403 |
| V1b | full | 0.60 | none | yes (`f_p ≡ 1`) | 0.182, 0.146, 0.478 |
| V2b | DESI footprint | 0.60 | none (`f_p ∈ {0,1}`) | yes | 0.240, 0.490, 0.170 |
| V3b | DESI footprint | 0.60 | real (`f_p ∈ (0,1)`) | yes | 0.210, 0.285, 0.069 |
| **V2** | DESI footprint | 0.60 | none | **no** | **0.0000, 0.0000, 0.0000** |
| **V3** | DESI footprint | 0.60 | real | **no** | **0.0000, 0.0000, 0.0000** |

**Finding B-1 (structural, deterministic).** With a footprint and `c_mode = selection`,
dropping `--per_pixel_completeness` rails `H0` to 125.4–137.6 in **16 of 16** runs
(V2, V3, and the ten `nofp` rows of the `fs = 0`/`fs = 1` scan in `bisection.txt`), at
CDF `0.0000` every time. The mechanism is stated in the code's own docstring
(`completion.py:2041`): "In AGGREGATE mode … `C_bar` … replaces the per-pixel ratio for
every pixel, occupied AND empty (**empty pixels have `C == C_bar` too, not 0**)". So
without `f_p` to supply the zero, the 1218 off-footprint pixels (40% of the sky) are
modelled as ~90% complete when they are 0% complete, and 40% of the missing budget
vanishes. `--complete_empty_pixel_policy zero` does NOT fix it: that flag acts on the
CONDITIONAL-mode per-pixel catalog prior (`redshift/prior.py:880-889`), not on the
field-mode global normalizer. **Consequence: the `table` arm — which cannot carry `f_p`
by `loaders.py:1021` — is structurally incapable of a correct run on any
footprint-limited survey**, so Tier B's table arm is not a valid comparison object and
the shipped `selq_radial` production configuration inherits the same exposure.

**Finding B-2 (the actual Tier-B failure: overconfidence, not a localized bug).** Every
`f_p`-aware rung above places `H0_true` inside the 90% CI. But on the SAME configuration
as V3b at a different seed block, five homogeneous realizations give
`H0 = 28.2, 30.4, 28.5, 78.3, 33.6` at quoted `σ = 5.8, 6.6, 6.2, 10.1, 7.4`. Pooling the
eight homogeneous V3b-configuration realizations: **spread 24.0 against a mean quoted
σ of 7.6 — a factor 3.2 of overconfidence**, with no consistent sign. **Tier C confirms it
at 24 clustered realizations: spread of the medians 18.83 km/s against a mean quoted σ of
7.26, a ratio of 2.59, while the median bias is −0.14 σ — the estimator is CENTRED and
its interval is ~2.6× too narrow.** Tier B's realization (seed 7001) is one draw from that
distribution, which is why it sits at +1.9 σ. The `H0` posterior is smooth and unimodal —
`logl_curve_raw.txt` has a full 25-node curve, peak at 25, monotone down to −61.4 nats at
140 — so this is not a multimodality artefact of the summary.

### B.5 Two mock-scale departures from the shipped builder, both recorded

* **Equal-comoving-volume shells.** `cli/build_latent_field.py:117` uses
  `linspace(0, z_depth, n_shells+1)` and occupancy guard 7 demands ≥ 1e4 galaxies and
  ≥ 500 occupied pixels per shell. At `z_depth = 0.30` the lowest LINEAR shell holds
  `(1/12)³ = 5.8e-4` of the volume, so guard 7 would need ~1.7e7 catalogued galaxies —
  more than the DESI union — before it could pass. `build_anchor16.py` uses equal-volume
  edges instead; guard 7 is still evaluated and stored in the artifact
  (`gal_ok`/`pix_ok`, per-shell counts) but is not enforced. At `n0 = 5e-5` the smallest
  shell holds 5615 galaxies (`gal_ok = False`, `pix_ok = True`); at `n0 = 2e-4` it holds
  22462 and guard 7 passes outright.
* **A damped solve.** See Finding S-1 in §Findings.

### B.6 The anchor the latent arm consumed, and why its field moves `H0` so little

`data/rb/latent_anchor.h5`, read back through the shipped
`factory.latent_artifact_fingerprint` and `latent_q.load_latent_plan`:

| quantity | value |
|---|---|
| stamped `sha256` | `cc834af0627dc2f5c6ed76ef72965a609ea599f624caade655b00cd2b61f906a` |
| guard-1 content digest | `86b90a9f8667fd4b56ced7a3bb5722170f9628d93063be8f85ae19a94ca0b284` |
| `format_version` | `darksirens-latent-field-1.0` |
| `‖ξ̂‖` / per-mode amplitude | 14.208 / **0.7943** (production anchor: 2.46) |
| `√(tr H⁻¹/M)` | **0.6864** — the prior sd is 1, so the count channel shrinks by only 31% at this catalog size |
| generated field `f_m(p,z)` | sd **1.0922**, range [−3.271, +3.271] |
| member-mean field (the frozen anchor field) | sd **1.0474** |
| member-to-member sd at fixed `(p,z)` | **0.2827 nat** |

So the latent seam is not inert here: it applies a `logQ` with sd 1.05 and carries a
0.28-nat member ensemble on top of it. It nevertheless moves the `H0` median by only
**0.136 σ** (§B.2). The reason is structural rather than numerical: `Q` modulates the
MISSING budget, and at the events' redshifts (`z` median 0.118) the survey is nearly
complete — `C(0.118) = 0.966`, times `f_p = 0.872`, so the missing channel carries ~16% of
the weight. A field of sd 1 acting on 16% of the budget is a sub-tenth-σ effect on `H0` at
60 events. **PR-6a's field is doing what it is built to do; the tier that fails is failing
for the reasons in §B.4, not because the seam is switched off.**

---

## Tier C — coverage — **FAIL** (24 realizations, not 50)

`tier_c.py` → `tier_c.json`. **24 realizations, not the 50 PLAN §6.2 asks for**, and the
reduction is stated here rather than buried: each realization needs its own mock, its own
anchor (the count channel is fitted to that realization's catalog) and its own scan, and
the TABLE arm would additionally need its own radial `Q` table at ~10 min of CPU each,
which is an order of magnitude more than everything else combined. So Tier C runs two arms
— `latent` (the PR-6a deliverable) and `latent_off` (its control at identical `f_p`
treatment) — at 24 realizations. The table arm's coverage is not measured anywhere in this
report.

**Reduced power, quoted.** At `n = 24` the two-sided KS 5% critical value is
`1.36/√24 = 0.278`; the measured `D` is 0.3198, so the KS rejection is not marginal but the
test cannot resolve `D < 0.28`. The "no realization outside the 99% band" gate has an
expected count of `0.24` at `n = 24`, so any non-zero observation is already decisive. The
median-bias gate has a standard error of roughly `1.25/√24 ≈ 0.26 σ` — i.e. **at `n = 24`
the `0.2 σ` bias gate cannot be resolved**, and its "pass" below should be read as "not
detected", not as "confirmed".

Grid: `H0 ∈ [20, 140]` step 2.5 (49 nodes), flat prior. Seeds `90000 + 37k`, `k = 0…23`,
`n0 = 5e-5`, one shared injection set.

| statistic | gate | `latent` (PR-6a) | `latent_off` |
|---|---|---|---|
| KS `D` on `CDF(H0_true)` | — | 0.3198 | 0.3070 |
| KS `p` | `> 0.05` | **0.0111** | **0.0166** |
| median `H0` bias | `< 0.2 σ` | −0.1437 σ | +0.0350 σ |
| realizations outside the 99% band | 0 | **9 / 24** | **8 / 24** |
| fraction inside the 90% band | 0.90 | **0.583** | 0.458 |
| fraction inside the 68% band | 0.68 | **0.292** | 0.208 |
| spread of the medians | — | 18.83 km/s | 17.49 km/s |
| mean quoted `σ` | — | 7.26 km/s | 7.74 km/s |
| **overconfidence ratio** | 1 | **2.59** | **2.26** |

**The failure is dispersion, not bias.** The median bias passes both arms (−0.14 σ, +0.04 σ),
so the estimator is centred. What fails is the width: the realization-to-realization spread
of the `H0` median is **2.6×** the uncertainty the posterior quotes, 9 of 24 realizations
land outside a 99% interval that should contain 23.8 of them, and the 68% interval contains
29% instead of 68%. This is the quantitative form of Finding B-2, and it is what makes the
single Tier-B realization sit at +1.9 σ.

**What the latent arm does to the answer** (paired, same 24 realizations):

| | value |
|---|---|
| `latent` − `latent_off` median shift | **−3.450 ± 4.231 km/s** = **−0.513 σ**, negative in **22 of 24** |
| `latent` − `latent_off` 90% width | **−1.545 km/s**; latent is WIDER in only **5 of 24** |

So the field is not inert — it moves the median by half a sigma with a consistent sign — but
it **narrows** the interval in 19 of 24 realizations. Under a correctly propagated ensemble
that would be the wrong direction; under the ensemble that actually ships it is the expected
one, because conditioning on a measured field removes variance and, per S-2, the `b_gal`
term that would put variance back is not implemented. And it does not repair the coverage:
the overconfidence ratio is 2.59 with the field on against 2.26 with it off.

---

## Tier D — misspecification at measured amplitudes — **paired shifts all pass; absolute gate not interpretable**

`tier_d.py` → `tier_d.json`. 8 seeds per stress (`50000 + 37k`), latent arm, `H0` grid step
2.5, `n0 = 5e-5`, shared injection set. Each stress is applied on ONE side only.

**Read the paired column, not the absolute one.** Tier C measured a per-realization bias
scatter of ~3 σ on this configuration, so at 8 seeds the median absolute bias carries a
bootstrap SE of ~1.6 σ and cannot resolve a 0.5 σ systematic — the `matched` control itself
sits at −0.857 ± 1.584 σ. The paired shift (stress minus matched at the SAME seed) removes
that scatter almost completely for the three stresses that leave the mock's random stream
untouched or nearly so, and it is the number PLAN's 0.5 σ gate is really about.

| stress | amplitude | median bias (absolute) | **paired shift vs `matched`** | verdict |
|---|---|---|---|---|
| `matched` (control) | — | −0.857 ± 1.584 σ | — | the reference |
| (i) `f_p` perturbed | `masked_frac + N(0, 0.104)` — PLAN §1.2's measured DESI sd; 2.3× the nside-16 map's own sd of 0.045 | −0.709 ± 1.625 σ | **+0.059 ± 0.040 σ** | **PASS** |
| (ii) fibre-assignment proxy | `0.05 (z/z_depth) tanh(logQ)`, survey-side only | −0.842 ± 1.596 σ | **+0.025 ± 0.006 σ** | **PASS** |
| (iii) `ls_ang` wrong by 2× | anchor built at `ls_sph = 1.0`, truth 0.5 | −0.790 ± 1.623 σ | **+0.002 ± 0.026 σ** | **PASS** |
| (iv) lognormal / non-Gaussian tail | variance-preserving skewed mixture, `a = 0.5` | −0.236 ± 2.092 σ | **+0.390 ± 1.395 σ** | **PASS**, but unpaired — see below |

Every stress moves `H0` by **less than 0.5 σ**, three of them by less than 0.06 σ. Reported
as systematics for PR-6a:

* **completeness-map error at the measured DESI amplitude: +0.36 km/s (0.059 σ).**
* **unmodelled 5% `z`- and density-dependent incompleteness: +0.17 km/s (0.025 σ).**
* **angular length scale wrong by 2×: +0.03 km/s (0.002 σ).** This one is exactly paired —
  only the anchor's basis changes, the mock is bit-identical at each seed — so 0.002 σ is a
  clean statement: at 60 events and a 16%-weight missing channel, the kernel scale is not a
  systematic at all.
* **non-Gaussian tail: +3.92 km/s (0.390 σ), SE 1.395 σ.** This stress changes `ξ_true`
  itself, so the realization is completely different from its matched partner and the
  "paired" shift is unpaired by construction — the 1.395 σ SE says so. **The number is
  consistent with zero and cannot be distinguished from a 0.5 σ systematic at 8 seeds.**
  Reported as measured, not as a pass on evidence.

The absolute-bias column is not a Tier-D result: it is Tier C's overconfidence, sampled
eight times. That is why Tier D is recorded above as "paired shifts pass; the absolute gate
is not interpretable while Tier B/C fail".

---

## Findings in `darksirens/` (reported, not fixed — PR-6a edits nothing under `darksirens/`)

### S-1 — `count_map_solve`'s undamped Fisher iteration diverges, and its docstring says it cannot

`redshift/latent_counts.py:175` runs a FIXED trip count of UNDAMPED Fisher steps and states
"`H >= I` makes the un-damped Fisher step globally well-posed for this objective in
practice". Measured at nside 16, 1.1e6 galaxies over 1854 × 12 voxels, on `ξ_true` drawn
from the model's own prior — **5 of 8 realizations fail P6** (`shipped_p6_pass = 0.375`),
`grad_inf` up to 5.0e5. Trace for the first failing realization
(`seed = 6100 + 977`):

```
it 0  |g|inf 7.85e4   J 8.578e6   |dx| 41.5      (||xi_true|| = 16.9)
it 1  |g|inf 1.55e5   J 1.001e7   |dx| 238       <- J RISES
it 2  |g|inf 6.87e4   J 7.425e7   |dx| 2914
...
it 8+ |g|inf 4.4e5    J 5.06e11 / 7.17e11        <- period-2 limit cycle
```

For a multinomial logit the canonical link makes the Fisher information EQUAL the observed
Hessian, so `count_map_solve` is exact Newton on a convex objective — locally convergent,
globally convergent only with a line search. `H >= I` bounds the step, it does not make it
an acceptable descent step. **The failure is loud, not silent**:
`cli/build_latent_field.py:170` gates on `grad_inf > 1e-8` and the production anchor
converged (`grad_inf = 1.09e-10`, `latent_anchor_v2a.h5`), so this is a robustness defect
in the builder, not a correctness defect in any shipped artifact. Fix: Armijo backtracking
on the same `objective`/`gradient`/`hessian_separable` (≈ 15 lines); `world16.solve_damped`
is a working reference. **One numerical detail is load-bearing** — the Armijo test needs an
absolute slack (`1e-12 |J|`): with `J ~ 8e6` the true decrease near the optimum is below
`eps |J| = 1.8e-9`, and a strict test backtracks to `t = 2⁻³⁰` while the gradient sits at
1.4e-6, i.e. it fails P6 for a floating-point reason rather than a convergence one
(observed on 1 of 8 realizations before the slack was added).

### S-2 — PLAN §3.4's `b_gal` rank-1 covariance inflation is not implemented

`s_b` (the profile curvature `[-d²log p_count/db²]⁻¹`) appears nowhere under `darksirens/`.
`laplace_draws(xi_hat, H_chol, n_draw, key)` has no covariance argument, and
`build_latent_field.py:193` passes the bare `L`. The `b_gal` sensitivity column is computed
(`dgrad_db` at `:183`) and stored, and `load_latent_plan` reads it into `LatentQPlan.S` —
but nothing consumes it for the draws. **The member ensemble is `N(ξ̂, H⁻¹)` at fixed
`b_gal`.** This makes Tier B's third acceptance criterion vacuous (§B.3) and it also means
the shipped `M_draw = 8` ensemble under-represents the field uncertainty by whatever
`s_b² (dξ̂/db)(dξ̂/db)ᵀ` contributes — a number nobody has measured, because computing it
requires `s_b`, which does not exist.

### S-3 — off-footprint pixels are modelled as `C_bar`-complete unless `f_p` supplies the zero

See Finding B-1 (§B.4): 16 of 16 footprint runs without `--per_pixel_completeness` rail
`H0` to 125–138 at CDF `0.0000`. `completion.py:2041` documents the behaviour
("empty pixels have `C == C_bar` too, not 0") and `--complete_empty_pixel_policy zero`
does not reach it (`prior.py:880-889` is the conditional-mode catalog prior, a different
object). Because `loaders.py:1021` refuses `--per_pixel_completeness` alongside a Q table,
**every Q-table configuration on a footprint-limited survey is exposed**, including the
production `selq_radial`. This is upstream of PR-6a — the latent mode is the one path that
is required to carry `f_p` — but PR-6a's Tier-B comparison cannot be stated without it.

### S-4 — `loaders.py:1044` still reads `data["ngals"]`, which K=1 dark sirens never sets

PR-5b reported this; it is unchanged on this branch. `inference/data.py:196` stores the
counts as `ngals_catalog`, so `attach_selection_fraction_inputs` raises
`KeyError: 'ngals'` and `--per_pixel_completeness` — MANDATORY in latent mode
(`factory.py:398`) — cannot reach a likelihood evaluation on the real loader path. Every
run in this report installs PR-5b's alias shim around `load_all_data`
(`arms.ngals_key_shim`). One line fixes it.

### S-5 — `fit_selection_from_mags` is biased by the photo-z it is not told about

Run on this mock's 775197-galaxy catalog (`darksirens_fit_selection --m_lim 20`) the
shipped fitter returns `M0hat = −19.97682 ± 0.00230`, `σ_M = 1.12690 ± 0.00135` against the
injected `(−20.15420, 1.0)` — an ≈ 8 σ pull on both. The cause is the mock's own photo-z:
`σ_z = 0.023` at `z ≈ 0.15` is a 15% distance error, i.e. 0.33 mag of scatter in the
distance modulus, which the truncated-LF likelihood has no term for and absorbs into
`σ_M`. Consequence: `C(z_depth)` moves from **0.4909** (truth) to **0.4208** (fit), a 14%
error in the missing budget at the survey edge. The closure tiers therefore run with
`theta_sel` at the injected truth (recorded in every `selection_fit.json` and in
`truth.json`); carrying the fit instead would have put a selection systematic into all four
Tier-B arms. Not a PR-6a defect — but it is a live systematic for any photometric survey
run through this fitter, and the Laplace `cov` it reports (5.3e-6, 1.8e-6) would state that
bias as a `10⁻³`-mag prior.

---

## Files and reproduction

All paths relative to `experiments/field_level_plan/pr6a/`. `DARKSIRENS_ZMAX=1.5` is
pinned inside `world16.py` (like `experiments/desi_full259/common.py` pins 6.0) and must
agree between the mock build, the anchor build and every inference call —
`latent_q.load_latent_plan` refuses an artifact whose `z_sub` is not the run grid's
below-depth prefix.

| file | what it is |
|---|---|
| `world16.py` | the shared nside-16 world: DESI depth map degraded to nside 16, factored basis, shell response, Poisson count draw, and `solve_damped` (see S-1) |
| `make_mock.py` | one realization: clustered complete catalog, masked survey, hosts from the true field, matched injections, depth map, selection fit, `n0` calibration, `truth.json` |
| `build_anchor16.py` | the nside-16 anchor artifact, same `/latent_field` schema and `format_version` as `cli/build_latent_field.py` |
| `arms.py` | the four arm configurations, the `ngals` shim (S-4), the H0 scan and posterior summary |
| `tier_a.py` / `tier_b.py` / `tier_c.py` / `tier_d.py` | the tiers |
| `tier_a.json`, `tier_b_rb.json`, `tier_c.json`, `tier_d.json` | measured output |
| `bisection.txt` | every row of §B.4's bisection and the `fs = 0` / `fs = 1` scan |
| `logl_curve_raw.txt` | a 25-node `logL(H0)` curve, showing the posterior is smooth and unimodal |
| `tier_c_run.log`, `tier_d_run.log` | the per-realization campaign logs |
| `data/rb/` | the Tier-B realization (seed 7001) with its anchor and Q table |
| `data/injections.h5` | the shared injection set (65791 detected of `Ndraw = 3e6`) |

```
# Tier A (CPU, ~2 min)
JAX_PLATFORMS=cpu python tier_a.py --n-real 8 --out tier_a.json

# Tier B (one GPU, ~4 min after the products exist)
python make_mock.py --seed 7001 --outdir data/rb            # or make_mock.build(...)
JAX_PLATFORMS=cpu python build_anchor16.py --survey data/rb/catalog_pixelated_nside_16.h5 \
    --mth-map data/rb/mth_map_nside16.h5 --out data/rb/latent_anchor.h5
OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -m darksirens.cli.build_lognormal_completion \
    --catalog data/rb/catalog_pixelated_nside_16.h5 --out data/rb/q_radial.h5 \
    --mode radial --c-mode selection --n-members 8 \
    --selection-fit data/rb/selection_fit.json --log10n0 -4.301029995663981 --delta 0.0 \
    --z-depth 0.30 --indexing global --lss-corr-length-mpc 50.0 --seed 1234 --workers 8
python tier_b.py --dir data/rb --h0-step 1.0 --out tier_b_rb.json

# Tiers C and D
python tier_c.py --n-real 24 --h0-step 2.5 --out tier_c.json
python tier_d.py --n-real 8  --h0-step 2.5 --out tier_d.json
```

### Stated idealizations

1. **`theta_sel` and `n0` are set at the injected truth** (§S-5 gives the measured cost of
   not doing so). They are survey-level calibrations, not per-event constants, so this does
   not violate `mock-data-dag`'s rule; it is recorded in every `truth.json`.
2. **One injection set is shared across realizations.** The injection set is a property of
   the detector and the DAG, not of the field realization; redrawing 3e6 proposals per
   realization would have dominated the campaign wall. Every realization's events and
   catalog are redrawn.
3. **`xi_true` is drawn from the prior the MAP is regularized by**, so Tiers A–C are
   "guaranteed by construction in exactly the regime where there is no data"
   (PLAN §6.1's own words). Tier D is the only tier that tests misspecification, and no
   tier tests the `z > z_depth` extrapolation.
4. **The mock's detection horizon is tuned inward** (`snr_ref = 5.0` instead of the
   generator's default) so the ~700 Mpc `rho = 8` horizon sits inside the `z <= 0.3`
   catalog. With the shipped default the horizon is ~2.9 Gpc, essentially no event would
   be inside the catalog, and every arm's `H0` posterior would be prior-wide — "`H0_true`
   inside the 90% CI" would pass in all arms while measuring nothing.
5. **`log10n0 = −4.301` sits below the package's default prior bound `[−4, −1]`.** It is
   pinned, not sampled, so it is inert; the run prints `prior.py:1096`'s warning about it.

---

## Provenance of the code these runs executed against

`git HEAD = 377092d` ("PR-5b: the member spread measured"), branch
`feat/field-level-pr6a-ensemble`. The working tree additionally carried the two other
PR-6a workstreams' UNCOMMITTED changes at run time —
`likelihood/core.py` (+134/−0: `member_ess`, the `lss_member_diagnostics` static kwarg),
`likelihood/factory.py` (+202: `latent_artifact_fingerprint`, `_latent_guard_fingerprint`),
`cli/inference.py` (+374/−17: guard 5, the selection-family guard, the §4.4 provenance
rewiring), `inference/run_fingerprint.py` (+12). Every run in this report goes through
`factory.make_likelihood`, so the new guard code was live; none of it is on the CLI path
these runs use, and none of it was modified here. **This workstream edited nothing under
`darksirens/` or `tests/`.**

---

## What this means for PR-6a

1. **The object PR-6a adds works.** Tier A recovers the field at slope 0.99908 ± 0.00433
   with `r = 0.99945` and a Laplace covariance calibrated to 2.4%, on a mock whose
   within-shell Jensen gap, photo-z convolution, real DESI mask and real footprint are all
   live. The nside-16 anchor it produces carries a `logQ` of sd 1.05 with a 0.28-nat member
   ensemble (§B.6). Nothing in Tiers B–D is evidence against the seam.
2. **Tier B's second acceptance criterion cannot be evaluated as written**, on this mock or
   on the production line, because no configuration lets the table arm and the latent arm
   differ only by the field (§B.2, §S-3).
3. **Tier B's third criterion is vacuous** until `s_b` and the rank-1 draw-covariance
   inflation of PLAN §3.4 are implemented (§B.3, §S-2). Measured: the latent arm NARROWS
   the 90% interval in 19 of 24 realizations.
4. **The blocking result is the dispersion.** A 60-event dark-siren posterior on this mock
   quotes σ ≈ 7 km/s and scatters by ≈ 19 km/s. The bisection (§B.4) rules out the field,
   the beyond-depth universe, the footprint and the areal mask individually — every
   `f_p`-aware rung places `H0_true` inside its 90% CI — so the residual is either a
   genuine cosmic-variance/selection-normalization term that neither PR-6a nor the shipped
   likelihood models, or a defect in the catalog-versus-missing normalization that this
   report has localized but not identified. **It should be identified before PR-6a's
   posterior is read as calibrated**, and it is upstream of the latent seam either way.
5. **Nothing here licenses or blocks the production 259-event run**, which remains held by
   the owner. The production line differs from this mock in every direction that matters
   for point 4 (259 events instead of 60, a 62%-covered nside-64 catalog, events out to
   `z ~ 5.7` where the catalog carries almost no weight), so its dispersion has to be
   measured on its own terms, not inherited from here.
