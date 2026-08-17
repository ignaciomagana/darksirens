# PR-6a mock closure, second pass — Tiers B and C after S-1 and S-2

Same world as `CLOSURE.md`: nside 16, `DARKSIRENS_ZMAX=1.5`, `H0_true = 67.74`, `Om0 =
0.3075`, `b_gal = b_GW = 1`, PR-0's clean guard arm. **No production 259-event run was
launched.**

The first closure pass failed Tiers B and C and named two defects in `darksirens/`:

* **S-1** — `latent_counts.count_map_solve`'s undamped fixed-trip Fisher iteration
  diverges on 5 of 8 nside-16 realizations.
* **S-2** — PLAN §3.4's rank-1 draw-covariance inflation
  `Cov(xi) = H^-1 + s_b^2 (dxi_hat/db)(dxi_hat/db)^T` **was not implemented anywhere**,
  so every member ensemble ever shipped was drawn from `H^-1` alone and PLAN §6.2's
  Tier-B width gate was vacuous.

Both are now fixed, on this branch, in commit `16e8195`. This pass re-runs Tiers B and C
against them and answers the question the fix was made to answer.

## The answer, first

**The `b_gal` dispersion did not close the coverage gap. It did not partially close it.
It moved the overconfidence ratio by +0.04%, in the wrong direction, and the diagnosis
that pointed at it was wrong.** S-2 was a real hole in the design and closing it was
right — the width gate is a
real gate now, and the artifact records its own draw covariance — but it is not where the
missing variance is. The variance decomposition in §V below localizes it instead, and it
is **not** in the latent seam, **not** in `b_gal`, and **not** in PR-6a.

| the question | the measurement | the answer |
|---|---|---|
| Did the `b_gal` dispersion close the coverage gap? | Tier C overconfidence `latent` **2.5515** → `latent_bgal` **2.5526** at n = 50; **2.5930 → 2.5938** on the first pass's own 24 seeds | **No.** It moved by **+0.04%**, and upward. |
| Partially? | paired over 50 realizations: 90% width ratio **1.00065**, wider in **24 of 50** | **No.** The sign is right; the size is 0.06% of a width that needs to be ×2.55 larger. |
| Was the diagnosis right? | §S: forcing `s_b` from 0 to 5 inflates the member spread ×4.54 and **narrows** the 90% `H0` interval by 0.6% | **No.** No `s_b` closes it — the hypothesis is refuted independently of the value S-2 measured. |
| Is `latent_off` still overconfident? | **2.3509** at n = 50 (was 2.2590 at n = 24) | **Yes.** The control has no field and no `b_gal`. |
| So is PR-6a blocked, or the mock? | §V: **92%** (`latent_off`) / **82%** (`latent`) of the excess variance survives with the **catalog held byte-identical** | **The mock/likelihood is.** PR-6a is not the cause and, measurably, cannot be the fix. |
| Was S-2 worth doing anyway? | §B.3: PLAN §6.2's third Tier-B gate is no longer vacuous; §0.1: S-1 was load-bearing — the Tier-B anchor's own operator **diverges** under the pre-S-1 solve | **Yes**, on both counts. Neither makes the tiers pass. |

---

## §0 The fixes are in the tree, and they were verified before anything expensive ran

```
git rev-parse --abbrev-ref HEAD          feat/field-level-pr6a-ensemble
git log --oneline -2                     16e8195 S-2: b_gal's rank-1 draw-covariance ...
                                         f45be7c PR-6a: the ensemble turned on ...
git diff --stat HEAD                     (empty -- every source change is committed)
```

`_damped_newton_step` (S-1) and `b_gal_profile_sigma` / `B_GAL_SYSTEMATIC_FLOOR_FRAC` /
`laplace_draws(..., s_b=, v_b=)` (S-2) are all present in
`darksirens/redshift/latent_counts.py` at HEAD.

```
DARKSIRENS_ZMAX=1.0 JAX_PLATFORMS=cpu pytest \
    tests/test_latent_b_gal_dispersion.py tests/test_latent_solve_damping.py -q
    -> 29 passed in 31.97s
```

**One bookkeeping defect, reported not fixed:** `tests/test_latent_solve_damping.py` is
**untracked** (`git status` shows it as `??`). S-1's 22 pins exist and pass, but they are
not in `16e8195` and would be lost by a clean checkout. Whoever owns the branch should
`git add` it before the PR.

### §0.1 S-1, re-verified on the closure's own operator — and it was load-bearing

The closure anchors are built with `world16.solve_damped`, the closure workstream's own
60-trip line search, not with the shipped `count_map_solve`. That was a workaround for
S-1. Now that S-1 is fixed, the shipped solve was run on the **actual Tier-B operator**
(`data/rb`, 1854 x 12 voxels, rank 320, 183267 galaxies):

| solve | `grad_inf` | note |
|---|---|---|
| shipped `count_map_solve`, pre-S-1 (`max_backtrack=0`, which reproduces it bit-for-bit) | **5.6844e+04** | **P6 FAILS**; `‖xi‖ = 1.314e5` against a true 14.208, `J = 1.572e10` |
| shipped `count_map_solve`, post-S-1 | **1.8423e-12** | P6 passes; `alpha = [0.5, 1, 1, …]` — **one halving, on trip 0, and nowhere else** |
| `world16.solve_damped` (what the tiers were built with) | 5.7906e-10 | the two agree: `max|Δxi| = 1.69e-11`, `max|ΔH_chol| = 7.34e-12` |

So **the Tier-B realization is itself one of the divergent ones**: the shipped builder as
it stood through PR-6a would have raised its own P6 gate on this mock. S-1 was not a
robustness nicety for the closure; without it the closure could not have been run through
the shipped solve at all. And the fixed shipped solve lands on the same anchor the tiers
were built against, to 1.7e-11 in `xi_hat` and 7.3e-12 in `H_chol` — `H_chol` being the
half that the Laplace draws and the sensitivity solves are built on.

### §0.2 S-2, measured on the Tier-B anchor

`build_anchor16.py` gained `--b-gal-dispersion` (default OFF **here**, so the "before" arm
stays reproducible from the same script; the shipped `cli/build_latent_field.py` defaults
it ON). It reads `v = S[:, 'b_gal']` out of the `sensitivity_S` block it has already built
and hands it, with `dgrad[:, -1]`, to `latent_counts.b_gal_profile_sigma`.

On `data/rb` (seed 7001, 1854 pixels, 183267 galaxies, 12 equal-comoving-volume shells,
`M = 320`, `M_draw = 8`):

| quantity | value |
|---|---|
| conditional curvature `J_bb` (field held fixed) | 1.728860e+05 → conditional sd 2.405e-03 |
| **profile curvature** `J_bb + (dgrad/db)·v` | **5.090137e+02** → **`s_b`(profile) = 4.432363e-02** |
| profiling `xi` out broadens `s_b` by | **×18.4** (curvature ratio 339.6) — the conditional slice would have been badly wrong |
| systematics floor, 5% of `b_gal = 1` | **5.000000e-02** |
| **`s_b` used** (floor wins, `s_b_floor_active = True`) | **5.000000e-02** |
| `tr H^-1` | 1.507769e+02 |
| `‖v‖²` | 1.489764e+02 |
| `s_b² ‖v‖²` | 3.724411e-01 |
| **overall member-spread inflation** `sqrt(1 + s_b²‖v‖² / tr H^-1)` | **1.001234** (+0.12%) |
| variance along `v` | 5.5199e-02 → 4.2764e-01, **×7.747 in variance, ×2.783 in sd** |
| member-to-member `row_fac` sd at fixed (p, mode) | 0.510045 → 0.511842 (**+0.35%**) |

Dataset-by-dataset against the dispersion-off anchor built from the same seed:
`xi_hat`, `H_chol`, `sensitivity_S`, `g_members`, `counts`, `completeness`,
`shell_response`, `fit_pixels`, `z_sub`, `z_count_edges`, `b_nodes` — **all bit-identical**.
Only `Xi_members`, `row_fac`, `eps_members` and the moments they generate move. The
inflation is exactly additive, as S-2's tests pin it to be.

**That table already contains the answer.** The inflation is ×2.78 in sd *along one
direction out of 320*, which is +0.35% on the member spread the likelihood actually
integrates over. Nothing that moves a member ensemble by 0.35% can move a ×2.59
overconfidence.

---

## §B Tier B — single-realization H0 closure, seed 7001, **five** arms

Same mock (`data/rb`), same seed, same guards, same `H0` grid (`[20, 140]` step 1.0) as
the first pass, so the comparison is like-for-like. The fifth arm is new:

* `latent` — anchor with `Cov = H^-1`. **This is now the historical control**: it is what
  the first closure pass ran and what shipped through PR-6a.
* `latent_bgal` — anchor with `Cov = H^-1 + s_b^2 v v^T`, `s_b = 0.05`. **This is the
  deliverable**, because S-2 made the inflation the shipped builder's default. The two
  arms share one loaded data object, one solve, one `g` stream and one set of guards; the
  only difference in the entire run is `+ s_b eps_m v` in the members.

| arm | LSS | `f_p` | H0 median | 68% CI | 90% CI | σ | 90% width | CDF at truth | ms/eval |
|---|---|---|---|---|---|---|---|---|---|
| `latent_off` | none | yes | 80.9502 | [74.372, 88.175] | [70.354, 93.364] | 6.9016 | 23.0096 | 0.01837 | 183 |
| `latent_off_nofp` | none | **no** | 93.5884 | [85.555, 101.947] | [80.721, 107.940] | 8.1959 | 27.2188 | 0.00017 | 123 |
| `table` | radial Q, 8 members | **no** | 92.5548 | [82.879, 101.424] | [77.770, 110.407] | 9.2729 | 32.6372 | 0.00100 | 125 |
| `latent` (Cov = H^-1) | latent, M=8 | yes | 80.0241 | [73.632, 86.991] | [69.718, 92.021] | 6.6794 | 22.3029 | 0.02350 | 218 |
| **`latent_bgal`** (Cov = H^-1 + s_b²vvᵀ) | latent, M=8 | yes | **80.0240** | [73.632, 86.991] | [69.719, 92.021] | **6.6792** | **22.3023** | 0.02349 | 217 |

**Every one of the four original arms reproduces the first pass to the last digit
printed** (80.95 / 93.59 / 92.55 / 80.02, CDF 0.0184 / 0.0002 / 0.0010 / 0.0235). The
re-run is a re-run, not a different experiment.

### B.1 The gates

| gate | first pass | this pass | verdict |
|---|---|---|---|
| `H0_true` in the 90% CI, every arm | 0 of 4 | **0 of 5** | **FAIL**, unchanged |
| latent vs table within 0.3 σ at `theta_fid` | 1.571 σ | **1.5710 σ** | **FAIL**, unchanged |
| latent CI width ≥ table CI width | 22.30 vs 32.64, **VACUOUS** | 22.3023 vs 32.6372, **REAL and FAILING** | see B.3 |

### B.2 The channel decomposition — identical to the first pass

| comparison | what it isolates | first pass | this pass |
|---|---|---|---|
| `latent` vs `latent_off` | the FIELD, at identical `f_p` treatment | 0.136 σ | **0.1364 σ** |
| `table` vs `latent_off_nofp` | the table's field, at identical (no-`f_p`) treatment | 0.118 σ | **0.1183 σ** |
| `latent_off` vs `latent_off_nofp` | the `f_p` channel alone | 1.674 σ | **1.6742 σ** |
| **`latent_bgal` vs `latent`** | **the `b_gal` propagation alone** | did not exist | **1.56e-05 σ** |

The 1.571 σ "latent vs table" disagreement is still **91% the `f_p` channel and 9% the
field**, for the reason `CLOSURE.md` §B.2 gives: `loaders.py:1021` refuses
`--per_pixel_completeness` alongside a Q table while `factory.py`'s guard 6 requires it in
latent mode, so no configuration exists in which the two arms differ only by the field.
S-2 does not touch that and was never going to.

The new row is the one that matters: **the entire `b_gal` propagation moves this
realization's `H0` by 1.6e-5 σ** — a median shift of `-1.04e-4` km/s and a 90%-width ratio
of `0.99997`.

### B.3 The width criterion is now REAL — and it fails, for a reason that is not `b_gal`

PLAN §6.2 predicated the gate on §3.4's propagation existing: "latent-on CI width ≥ table
CI width — now a valid check because §3.4 propagates `b_gal` (rev 1's version compared
against a point estimate and would have passed while under-dispersed)". The first pass
marked it VACUOUS because `s_b` did not exist. **It exists now, it is measured (not
dialled), and the reference arm's members carry it, so the gate is no longer vacuous.**

Measured: `latent_bgal` 90% width **22.3023** against `table` **32.6372**. The latent arm
is still narrower, by 32%. **The gate FAILS.**

But it fails by 32% while the propagation it was predicated on contributes **0.003%**
(22.3029 → 22.3023 — the inflated arm is in fact 6e-4 km/s *narrower*, which is
grid-interpolation noise, not a signal). So the honest statement is:

> The criterion is real for the first time, and it fails — but it fails through the `f_p`
> channel (`latent_off` 23.0096 vs `latent_off_nofp` 27.2188 at identical LSS treatment),
> not through the field and not through `b_gal`. As an acceptance criterion on the field
> it remains unusable on a footprint-limited survey until `loaders.py:1021` is lifted;
> S-2 removed the *vacuousness*, not the *confound*.

---

## §C Tier C — coverage at **n = 50**, three arms

`tier_c.py --n-real 50 --h0-step 2.5 --arms latent_off latent latent_bgal --out
tier_c_v2.json`. Seeds `90000 + 37k`, `k = 0…49`, `n0 = 5e-5`, one shared injection set,
`H0 ∈ [20, 140]` step 2.5, flat prior. **50 realizations, the number PLAN §6.2 asks for**
— the first pass ran 24 and said so; this pass runs the planned 50, and `k = 0…23` are
the first pass's own seeds, so the comparison is exact and the extension is free of
selection.

Each realization builds its own mock, its own **two** anchors (dispersion off and on,
from one catalog and one solve, so the two latent arms are paired to the bit), and its
own scan. Wall: 4136 s for 50 x 3 arms on one H100 (shared with the diagnostics below).

**Like-for-like, verified rather than asserted:** over the 24 shared seeds, the re-run's
`latent_off` and `latent` medians differ from `tier_c.json` by **max |Δ| = 0.000e+00** —
bit-identical. Everything below that says "first pass" is the same numbers, not a
re-measurement of them.

| statistic | gate | `latent_off` | `latent` (Cov = H⁻¹) | **`latent_bgal`** (Cov = H⁻¹ + s_b²vvᵀ) |
|---|---|---|---|---|
| KS `D` on CDF(H0_true) | — | 0.27369 | 0.27841 | 0.27983 |
| KS `p` | > 0.05 | **8.43e-04** | **6.42e-04** | **5.91e-04** |
| median `H0` bias | < 0.2 σ | **+0.3033 σ** | −0.0361 σ | −0.0336 σ |
| realizations outside the 99% band | 0 | **18 / 50** | **19 / 50** | **19 / 50** |
| fraction inside the 90% band | 0.90 | 0.46 | 0.58 | 0.58 |
| fraction inside the 68% band | 0.68 | 0.30 | 0.34 | 0.34 |
| spread of the medians | — | 19.1137 km/s | 19.7270 km/s | 19.7498 km/s |
| mean quoted σ | — | 8.1303 km/s | 7.7314 km/s | 7.7370 km/s |
| **overconfidence ratio** | 1 | **2.3509** | **2.5515** | **2.5526** |
| verdict | | **FAIL** | **FAIL** | **FAIL** |

**Power at n = 50, stated.** The two-sided KS 5% critical value is `1.36/√50 = 0.1923`
against a measured `D ≈ 0.28`, so the rejection is not marginal and the test now resolves
`D > 0.19` (it could not resolve `D < 0.28` at n = 24). The "no realization outside the
99% band" gate has an expected count of `0.5`; 18–19 were observed. The median-bias gate
has a standard error of about `1.25/√50 ≈ 0.18 σ`, so `latent_off`'s `+0.303 σ` now
**fails the stated gate** while being only 1.7 σ from zero — it should be read as "fails
the gate, not resolved as a bias", and note it PASSED at n = 24 (`+0.035 σ`). The latent
arms are centred either way (−0.036 σ).

### C.1 Before and after — the whole point of the re-run

| | first pass, n = 24 | this pass, **same 24 seeds** | this pass, n = 50 |
|---|---|---|---|
| `latent_off` overconfidence | 2.2590 | 2.2590 | 2.3509 |
| `latent` overconfidence | 2.5930 | 2.5930 | 2.5515 |
| **`latent_bgal` overconfidence** | (did not exist) | **2.5938** | **2.5526** |
| `latent` → `latent_bgal` change | — | **+0.0008 (+0.03%)** | **+0.0011 (+0.04%)** |
| `latent` KS p | 0.01107 | 0.01107 | 0.00064 |
| `latent_bgal` KS p | — | 0.01106 | 0.00059 |
| `latent` outside 99% | 9 / 24 | 9 / 24 | 19 / 50 |
| `latent_bgal` outside 99% | — | 9 / 24 | 19 / 50 |

**The overconfidence went UP, by 0.04%.** On the first pass's own 24 realizations the
`b_gal` propagation moved it from 2.5930 to 2.5938.

Paired over all 50 realizations (same mock, same catalog, same solve, same `g` stream —
the two arms differ only by `+ s_b eps_m v`):

| paired quantity | value |
|---|---|
| `latent_bgal` − `latent` median shift | **−0.0209 ± 0.0891 km/s** |
| `latent_bgal` − `latent` 90% width shift | **+0.0186 km/s**, ratio **1.00065** |
| `latent_bgal` wider than `latent` in | **24 of 50** — a coin flip |

The width does move in the right direction on average (+0.019 km/s, ratio 1.0006), which
is the sign PLAN §3.4 predicts, and the sign is the only thing about it that is
detectable. **0.06% of an interval that needs to be ×2.55 wider.**

For contrast, the field itself is not inert on this mock, at 50 realizations:

| paired quantity | first pass (n = 24) | this pass (n = 50) |
|---|---|---|
| `latent` − `latent_off` median shift | −3.450 ± 4.231 km/s | **−2.891 ± 3.655 km/s** |
| `latent` − `latent_off` 90% width ratio | narrower in 19 of 24 | **0.9559, wider in 15 of 50** |

So the seam still moves the answer by roughly half a σ and still NARROWS the interval in
70% of realizations — and S-2's inflation, which was supposed to be what put that
variance back, contributes 0.06% of a width against the field's −4.4%.

---

---

## §S Could ANY `s_b` have closed it? — `s_b_sensitivity.py` → `s_b_sensitivity.json`

`s_b` is measured, not dialled, so this is not a proposal to turn a knob. It is a
falsification test that does not depend on the value S-2 chose, on the 5% floor, or on
the profile-versus-conditional argument. **If the closure's diagnosis were right — the
ensemble is under-dispersed and `b_gal` is the missing dispersion — then SOME `s_b` would
reproduce the ~2.6× interval coverage needs.** The Tier-B anchor was rebuilt across three
decades of `s_b` (forced by setting the systematics floor, which only ever raises `s_b`)
and the same `latent` arm run on the same mock and the same loaded data:

| `s_b` | member-to-member sd | H0 median | σ | **90% width** | CDF at truth |
|---|---|---|---|---|---|
| 0 (feature off) | 0.51004 | 80.024 | 6.679 | **22.303** | 0.0235 |
| **0.05 (shipped)** | 0.51184 | 80.024 | 6.679 | **22.302** | 0.0235 |
| 0.20 | 0.52448 | 80.026 | 6.678 | **22.300** | 0.0235 |
| 1.00 | 0.72353 | 80.107 | 6.663 | **22.252** | 0.0224 |
| 5.00 | 2.31447 | 80.438 | 6.629 | **22.162** | 0.0189 |
| 25.0 | 10.94925 | 57.629 | 9.782 | **32.976** | 0.8522 |

**Between `s_b = 0` and `s_b = 5` the member spread rises ×4.54 and the 90% `H0` interval
gets 0.6% NARROWER.** The response is not merely small — over the entire physically
admissible range it has the wrong sign. `dW90/d(member sd)` is `-0.078` km/s per nat over
that span. To move the interval at all one has to reach `s_b = 25` — 500× `b_gal`, a
member ensemble with 10.9 nats of `logQ` scatter — and there the arm does not become
calibrated, it becomes wrong in the other direction (`H0` 57.6, CDF 0.85).

**No `s_b` closes Tier C.** The under-dispersion hypothesis is refuted on its own terms,
and the refutation is independent of every choice made in S-2.

*Why the width does not respond:* `Q` modulates only the MISSING budget, and at this
mock's event redshifts (`z` median 0.118) the survey is nearly complete — `C(0.118) =
0.966` times `f_p = 0.872`, so the missing channel carries ~16% of the weight
(`CLOSURE.md` §B.6). Marginalizing a wider ensemble over 16% of the budget cannot widen
`H0` by 160%. That was measurable before S-2 was written and is the reason the fix was
never going to be sufficient — which is a criticism of the diagnosis, not of the fix.

---

## §V Where the missing variance actually is — `variance_split.py`

`make_mock.build` draws one realization from one stream in the order

    seed -> xi_true -> complete catalog -> mask/survey -> 60 events -> PE

so an `event_seed` inserted immediately before the event draw splits the realization in
two: **everything the likelihood conditions on** (the field, the galaxies, the footprint,
the mask, the survey catalog, `f_p`, the anchor) is a function of `seed`; **the GW data**
is a function of `event_seed`. A 5 × 5 grid then gives the law of total variance

    Var(H0) = E_cat[ Var_evt(H0 | cat) ]  +  Var_cat( E_evt[H0 | cat] )

with the second term debiased by `within/n_event`. The posterior's quoted `sigma`
conditions on the catalog as given, so **the first term is what it is supposed to
estimate.**

25 cells (5 catalogs × 5 event sets, `H0` step 2.5, seeds `90000 + 37i`, event seeds
`310000 + 131j`):

| | `latent_off` (no field, no `b_gal`) | `latent` |
|---|---|---|
| mean quoted σ | 7.1698 km/s | 6.8172 km/s |
| **TOTAL spread of medians** | **16.8316** → overconfidence **2.348** | **17.7440** → **2.603** |
| **EVENT spread at FIXED catalog** | **16.1588** → overconfidence **2.254** | **16.1029** → **2.362** |
| catalog common mode (debiased) | 5.1608 km/s | 8.1644 km/s |
| **fraction of variance from events** | **0.9217** | **0.8236** |

Per-catalog event sd, `latent_off`: 8.78, 13.63, 17.37, 17.32, 21.00 — every catalog on
its own already scatters by more than twice its quoted σ.

**This reproduces Tier C's overconfidence from an entirely independent design** (2.348 vs
Tier C's 2.259 for `latent_off`, 2.603 vs 2.593 for `latent`) **and it reproduces it with
the catalog held fixed.** So:

* it is **not** the latent field — `latent_off` has none and is overconfident 2.25× at
  fixed catalog;
* it is **not** `b_gal` — §S measured that at 1.6e-5 σ and refuted it at every `s_b`;
* it is **not** `f_p`, the depth map, the completeness model, or the anchor — all of them
  are byte-identical across the five cells of a row;
* it is **not** cosmic variance in the galaxy field — that is the catalog term, and it is
  8% of the variance in `latent_off` and 18% in `latent`.

**Between 82% and 92% of the excess variance is in the GW-event channel**, present when
the galaxy side of the problem is held completely fixed.

### V.1 One suspect inside that channel, ruled out with a number

The cheapest explanation for a centred estimator with a too-narrow interval is that the
mock's synthetic PE posteriors are narrower than the scatter of the observation around
the truth — `mock-data-dag`'s standing pitfall, which would make every downstream
interval too tight by the same factor. `pe_calibration.py` measures it on the Tier-B
realization: for each event, the standardized residual of the TRUE `dL` inside its own PE
samples.

| statistic | measured | expected |
|---|---|---|
| residual sd = **PE width mis-scaling** | **1.0593** | 1 |
| residual mean | +0.4858 | 0 for a blind draw; **positive is correct here** |
| KS p on the PE quantiles | 5.54e-03 | — |
| median fractional `dL` error | 0.1159 | — |

**The PE is width-calibrated to 6%.** It cannot supply a factor 2.3. The mean offset of
+0.486 σ is the expected Malmquist signature of conditioning on detection — detection
selects upward SNR fluctuations, which under-estimate `dL` — and is exactly what the
hierarchical selection term exists to absorb; it drives the KS rejection and is not
evidence of a width defect.

### V.2 Inside the event channel, two families remain — and this pass does **not** separate them

`nobs_scaling.py`. One catalog (seed 90000) held fixed for every cell; only `N_obs` and
the event/PE stream change. Two defects produce the same signature at fixed `N` and have
opposite `N` scalings:

* **(A) per-event over-sharpness** — every event's likelihood claims a factor `k²` too
  much information (a mis-specified per-event term: the catalog `z` kernel, the `p_pe`
  measure, the sky treatment, a pinned population that should be marginalized). Then the
  true spread and the quoted σ both fall like `N^-0.5` and the ratio is FLAT.
* **(B) a per-realization common mode in the event channel** — one number per
  realization, not per event (the selection integral, a shared normalization). Then σ
  falls like `N^-0.5` and the offset does not, so the ratio grows like `N^0.5`.

| `N_obs` | cells | spread of medians | mean quoted σ | overconfidence |
|---|---|---|---|---|
| 30 | 22 | 18.861 ± 2.910 | 13.553 | **1.392 ± 0.215** |
| 60 | 6 | 22.509 ± 7.118 | 8.131 | **2.768 ± 0.875** |
| 120 | 22 | 12.202 ± 1.883 | 5.877 | **2.076 ± 0.320** |

Fitting `spread ∝ N^-a` over the ×4 lever arm from 30 to 120:

> **`a = 0.3141 ± 0.1574`.** Pure per-event statistics (`a = 0.5`) is **1.18 σ** away.
> A pure common mode (`a = 0`) is **2.00 σ** away.

**The test does not resolve them.** It leans toward (A) — per-event over-sharpness — and
it disfavours a pure common mode at 2 σ, but neither hypothesis is excluded and it would
be dishonest to report this as a localization. Two things additionally contaminate it,
both stated:

1. **Prior truncation at the low-`N` end.** 3 of the 22 `N = 30` realizations have their
   nominal 90% interval reaching past the flat prior's `[20, 140]` edge, against 0 of 22
   at `N = 120`. Truncation shrinks the quoted σ, so it biases the σ exponent high and the
   overconfidence at `N = 30` low. Measured on the tier itself: the quoted σ falls as
   `N^-0.603` where `N^-0.5` is the statistical rate, and the `N = 30` truncation is the
   leading candidate for that excess. *(In Tier C proper, 5 of 50 realizations touch below
   `H0 = 25` and none above 135, so ~10% of that tier carries the same mild contamination
   — enough to matter at the few-percent level, nowhere near enough to manufacture 2.55.)*
2. **One catalog.** Everything here is conditional on seed 90000, so the `N` scaling is
   the event-conditional one and says nothing about the 8–18% catalog term.

**What would resolve it**, precisely: repeat this design at 3 catalogs × 3 `N` values
(30 / 120 / 480) × 25 event sets, with the `H0` grid widened to `[10, 200]` so no
posterior touches a prior edge at `N = 30`. The ×16 lever arm plus `n = 25` gives
`sd(a) ≈ 0.05`, which separates `a = 0` from `a = 0.5` at 10 σ. That is ~225 cells, about
4 GPU-hours at this scale — an order of magnitude less than the campaign it would settle,
and it is the single measurement the next workstream should take before anything else.


---

## §F Findings

### F-1 — S-1 is fixed, and it mattered more than "robustness"

The Tier-B realization's own count operator **diverges** under the pre-S-1 shipped solve
(`grad_inf = 5.68e4`, `‖xi‖ = 1.31e5` against a true 14.208). `cli/build_latent_field.py`
would have raised its own P6 gate on this mock. Post-S-1 it converges at `1.84e-12` with
one halving on trip 0, and lands within `1.7e-11` (`xi_hat`) and `7.3e-12` (`H_chol`) of
the closure's private `world16.solve_damped`. **`build_anchor16.py` can now drop
`world16.solve_damped` and call `count_map_solve` directly**; it was left on the private
solve for this pass so the before/after comparison had exactly one moving part.

### F-2 — S-1's tests are not committed

`tests/test_latent_solve_damping.py` (22 pins, all passing) is **untracked**. The source
change is in `16e8195`; the tests are not in any commit and a clean checkout loses them.
`git add` before the PR.

### F-3 — S-2 closed a real hole, and the gate it unblocked still fails on `f_p`

PLAN §6.2's third Tier-B criterion was vacuous and is not any more (§B.3). It fails —
22.30 (latent) vs 32.64 (table) — but the confound `CLOSURE.md` §B.2 identified is
untouched: 91% of the latent-vs-table difference is the `f_p` channel, which the table arm
cannot carry (`loaders.py:1021`) and latent mode must (`factory.py` guard 6). **S-2
removed the vacuousness; it did not remove the confound, and no acceptance criterion
stated as "latent vs table" is measurable on a footprint-limited survey until
`loaders.py:1021` is lifted.**

### F-4 — the blocking defect is in the GW-event channel of this mock, not in the galaxy channel

Measured, not argued: 92% (`latent_off`) / 82% (`latent`) of the excess variance is
present with the catalog, the field, `f_p`, the mask, the depth map and the anchor all
held byte-identical (§V). The PE is width-calibrated to 6% (§V.1). The mechanism inside
that channel is **not** resolved by this pass — §V.2's `N`-scaling gives
`a = 0.3141 ± 0.1574` where per-event over-sharpness predicts 0.5 and a per-realization
common mode predicts 0 — and §V.2 states the ~4-GPU-hour design that would resolve it.
`CLOSURE.md` §B.4's homogeneous control (`field_scale = 0`, ×3.2 overconfidence) is
consistent with all of this and is further evidence that clustering is not the source.

### F-5 — the earlier findings are unchanged

S-3 (off-footprint pixels modelled as `C_bar`-complete without `f_p`), S-4
(`loaders.py:1044` reads `data["ngals"]`; every run here still installs PR-5b's shim),
and S-5 (`fit_selection_from_mags` biased by photo-z) are untouched by this pass and stand
as `CLOSURE.md` reports them.

---

## §R What this means for PR-6a

1. **The object PR-6a adds still works.** Tier A is unchanged (slope 0.99908 ± 0.00433,
   `r = 0.99945`) and nothing measured here is evidence against the seam. The seam moves
   `H0` by 0.136 σ on Tier B and −2.891 ± 3.655 km/s paired over 50 Tier-C realizations.
2. **S-1 and S-2 are both real fixes and both should ship.** S-1 turns a 0.375 P6 pass
   rate into 1.000 and is bit-identical where the undamped solve converged; S-2 implements
   a covariance the plan specifies and that nothing implemented, and it makes a stated
   acceptance criterion evaluable for the first time. Their tests pass (29), the full
   latent suite passes (228 passed, 1 skipped), and the P12 goldens are 23/23 bit-exact.
3. **Neither of them makes the mock closure pass, and the second one was aimed at the
   wrong target.** The `b_gal` propagation contributes 1.6e-5 σ on Tier B and +0.04% of
   the Tier-C overconfidence. The diagnosis that it would close the gap is refuted at
   every `s_b` from 0 to 5.
4. **Tier B and Tier C remain FAILED, at n = 50, and the failure is upstream of the latent
   seam.** It is in the GW-event channel of this mock: it survives holding the catalog
   fixed, it is present with no field at all, and it is present in a homogeneous universe.
   **PR-6a is not blocked by its own design; the mock's calibration is what is blocked**,
   and the closure ladder cannot certify a posterior until §V.2's measurement is taken.
5. **Nothing here licenses or blocks the production 259-event run**, which remains held by
   the owner. Everything in `CLOSURE.md`'s point 5 still applies: the production line
   differs from this mock in every direction that matters for point 4, and its dispersion
   has to be measured on its own terms.

---

## §T Tests

All CPU. `test_unified_k1_golden.py` at **default** `zMax` (`DARKSIRENS_ZMAX=1.0` shifts
the grid and fails 16 cells for that reason alone).

```
DARKSIRENS_ZMAX=1.0 JAX_PLATFORMS=cpu pytest \
    tests/test_latent_b_gal_dispersion.py tests/test_latent_solve_damping.py -q
    -> 29 passed in 31.97s                                  [S-1 + S-2, run first]

DARKSIRENS_ZMAX=1.0 JAX_PLATFORMS=cpu pytest tests/test_latent_*.py -q
    -> 228 passed, 1 skipped in 161.16s
       (14 files; was 199 passed / 1 skipped before S-1 and S-2 added 29 pins)

JAX_PLATFORMS=cpu DARKSIRENS_GOLDEN_EXACT=1 pytest tests/test_unified_k1_golden.py -q
    -> 23 passed in 59.83s                                  [P12 bit-exact]
```

---

## §P Files and reproduction

All paths relative to `experiments/field_level_plan/pr6a/`. `DARKSIRENS_ZMAX=1.5` is
pinned inside `world16.py` and must agree between the mock build, the anchor build and
every inference call.

| file | what it is | new? |
|---|---|---|
| `CLOSURE_v2.md` | this | new |
| `tier_b_rb_v2.json` | Tier B, 5 arms, seed 7001 | new |
| `tier_c_v2.json`, `tier_c_v2_run.log` | Tier C, n = 50, 3 arms | new |
| `variance_split.py`, `variance_split_analyze.py`, `variance_split.json`, `variance_split_decomposition.json`, `variance_split_run.log` | the 5 × 5 catalog × event-set decomposition | new |
| `s_b_sensitivity.py`, `s_b_sensitivity.json`, `s_b_sensitivity_run.log` | the `s_b` sweep that refutes the diagnosis | new |
| `pe_calibration.py`, `pe_calibration.json` | the mock's own PE PP test | new |
| `nobs_scaling.py`, `nobs_scaling{,_b,_c}.json`, `nobs_scaling{,_b,_c}_run.log` | the `N_obs` lever arm (22/6/22 event sets) | new |
| `build_anchor16.py` | gained `--b-gal-dispersion` / `--s-b-floor-frac`, stamps `s_b`, both curvatures, `eps_members` | edited |
| `make_mock.py` | gained `event_seed` (re-seeds immediately before the event draw; `None` reproduces every existing seed bit-for-bit) | edited |
| `arms.py` | gained the `latent_bgal` arm (identical opts, different artifact) | edited |
| `tier_b.py` / `tier_c.py` | 5 arms / 3 arms, the paired blocks, the overconfidence statistic in the JSON | edited |
| `CLOSURE.md`, `tier_a.json`, `tier_b_rb.json`, `tier_c.json`, `tier_d.json`, `bisection.txt` | the first pass, unmodified | unchanged |

**Note on the `.log` files.** `.gitignore:91` ignores `*.log` repo-wide, so every campaign
log listed above — including the first pass's `tier_c_run.log` and `tier_d_run.log` — is
present on disk and **not** in the PR. The `.json` next to each one carries the numbers
this report quotes; the logs are the per-realization trace only.

```
# Tier B (one GPU, ~2 min; data/rb and its Q table already exist)
JAX_PLATFORMS=cpu python build_anchor16.py --survey data/rb/catalog_pixelated_nside_16.h5 \
    --mth-map data/rb/mth_map_nside16.h5 --out data/rb/latent_anchor.h5
JAX_PLATFORMS=cpu python build_anchor16.py --survey data/rb/catalog_pixelated_nside_16.h5 \
    --mth-map data/rb/mth_map_nside16.h5 --out data/rb/latent_anchor_bgal.h5 \
    --b-gal-dispersion
python tier_b.py --dir data/rb --h0-step 1.0 --out tier_b_rb_v2.json

# Tier C (one GPU, 4136 s)
python tier_c.py --n-real 50 --h0-step 2.5 \
    --arms latent_off latent latent_bgal --out tier_c_v2.json

# The diagnostics
python variance_split.py --n-seed 5 --n-event 5 --out variance_split.json   # 1357 s
python variance_split_analyze.py --in variance_split.json
python s_b_sensitivity.py --dir data/rb --h0-step 1.0                       #  238 s
python pe_calibration.py --gw data/rb/gw_events.h5                          #   <1 s
python nobs_scaling.py --seed 90000 --nobs 30 60 120 --n-event 6            # 1084 s
python nobs_scaling.py --seed 90000 --nobs 30 120 --n-event 8 \
    --event-seed0 420000 --arms latent_off --out nobs_scaling_b.json
python nobs_scaling.py --seed 90000 --nobs 30 120 --n-event 8 \
    --event-seed0 430000 --arms latent_off --out nobs_scaling_c.json
```

### Stated idealizations

`CLOSURE.md`'s five stated idealizations are unchanged and all still apply (`theta_sel`
and `n0` at the injected truth; one shared injection set; `xi_true` drawn from the prior
the MAP is regularized by; the detection horizon tuned inward; `log10n0` below the default
prior bound but pinned). Two are added by this pass:

6. **`build_anchor16.py` still uses `world16.solve_damped`, not the fixed
   `count_map_solve`.** Deliberate: the whole point of the pass is a before/after with one
   moving part. §0.1 measures that the two agree to 1.7e-11 and that the shipped solve now
   converges on this operator, so the choice costs nothing and F-1 records that it can now
   be dropped.
7. **`build_anchor16.py` defaults `b_gal_dispersion=False` while the shipped
   `cli/build_latent_field.py` defaults it ON.** Also deliberate, and for the same reason:
   the "before" arm must be reproducible from the same script. Every artifact stamps
   `b_gal_dispersion`, `s_b` and `draw_covariance`, so no anchor is ambiguous about which
   covariance it carries.

## Provenance

`git HEAD = 16e8195` ("S-2: b_gal's rank-1 draw-covariance inflation, measured not
dialled"), branch `feat/field-level-pr6a-ensemble`, working tree clean against HEAD for
everything under `darksirens/`. **`CLOSURE.md`'s statement "this workstream edited nothing
under `darksirens/` or `tests/`" is superseded**: commits `f45be7c` and `16e8195` on this
branch now carry S-1 and S-2 in `darksirens/redshift/latent_counts.py` and
`darksirens/cli/build_latent_field.py`. This second pass edited nothing under
`darksirens/` or `tests/` — only `experiments/field_level_plan/pr6a/`.
