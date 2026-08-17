# PR-6a — the frozen-anchor member ensemble (2026-08-17)

**THE PRODUCTION 259-EVENT `H0` RUN IS HELD.** Nothing in this report is a
posterior on the production line. The owner gated the headline run on PR-6a;
PR-6a is now measured, and the gate is theirs to take. Everything below is a
likelihood-level measurement (timings, sweeps, identities, guard exits) plus
the nside-16 mock closure of `CLOSURE.md`. A reader who takes any number here
as "the `H0` result" has read the wrong document — that number does not exist.

Modules that ship: `darksirens/likelihood/core.py` (member ESS + the
`darksiren_member_diagnostics` wrapper), `darksirens/cli/inference.py`
(guards 1 and 5, the selection-family refusal, the §4.4 provenance rewiring
routed both ways, `--lss_field_sha256` / `--allow_unanchored_budget`),
`darksirens/likelihood/factory.py` (`latent_artifact_fingerprint`,
`_latent_guard_fingerprint`), `darksirens/inference/run_fingerprint.py` (both
new flags declared non-semantic so every table-mode digest stays
byte-identical). Pins: `tests/test_latent_p13.py` (30),
`tests/test_latent_guards.py` (41), on top of PR-1..PR-5's suites. **The tree
is not committed**; the working diff is +716/−17 across five tracked files
plus two new test modules, and it is entirely this workstream's.

## The campaign

Four phases, three of which ran elsewhere and are cited rather than repeated:
the build phase (member ESS, P13), the guard matrix and the §4.4 provenance
rewiring, the nside-16 mock closure (`CLOSURE.md`, Tiers A–D), and this rung —
the determinism sweep (PLAN §6.4), the overhead measurement (§2.3), and an
independent re-verification of the guards.

Production line for the timing and sweep work: `experiments/desi_full259`,
`GW_259` (259 events × 4096 PE samples), `INJ_PLAIN` (1,067,946 detected
injections), `SURVEY_N64` (30,470 footprint pixels, 22,787,566 galaxies),
`Om0 = 0.3089`, `z_depth = 0.30`, `DARKSIRENS_ZMAX = 6.0`; the latent anchor is
PR-5's `latent_anchor_v2a.h5` at `M_draw = 8`, the value PR-5b's P14 verdict
selected; `b_GW = b_miss = 1` read back from the decoded survey. Guard
convention throughout: **PR-0's clean arm** — `selection_neff_soft_guard =
False`, `max_likelihood_variance = 1e6`, Vitale's `5 N_obs` floor retained.
The hard GWTC-4/5 criterion fails at every `H0` node on this line (needs
`Neff ~ 92k`, the line delivers 31–36k), so the shipped soft-guard convention
replaces the likelihood with a `-gate (100 + 2 N softplus(-log mu))` wall; PR-5b
measured that wall at 610×–34206× the clean arm and non-monotone in `H0`. Every
`logL` below is the clean arm, and every timing was re-checked under PR-0's own
soft-guard convention (below) so that the choice costs nothing.

Wall: `overhead_determinism.py` 4 arms + 2 sweeps + 200 repeats = 34 min on one
H100 NVL; `determinism_1e4.py` 20,200 fixture evaluations = 57 s on CPU;
`verify_guards.py` 11 cells in 40 s on CPU.

A cross-session reproducibility check fell out for free: the latent likelihood
at the anchor node reads **−766.7914443719592** here (H100 NVL, `miko`) against
PR-5b's **−766.7914443725257** (A100-40, six hours earlier, a different
process). Agreement to **5.67e-10 nat** — not bit-identical across machines,
which is expected, and far below anything that moves a posterior.

## Gates

| gate | value | threshold | verdict |
|---|---|---|---|
| P12 golden, latent off | 23/23, `DARKSIRENS_GOLDEN_EXACT=1`, default zMax | bit-identical | **pass** |
| P13 budget identity (4), consumed | worst **3.8e-15**; `log Z_m` spread **0.0 nat, bit-exact** | `1e-12`, all z, all members, 5 theta | **pass** |
| P13b off-footprint routing | bit-zero `logQ` | bit-identical | **pass** (PR-5) |
| P14 `M_draw` bias, shipped `M = 8` | **7.073e-3 nat** | < 0.1 nat | **pass** (PR-5b; caveats below) |
| P16 complete-catalog limit at `b_GW = b_gal` | bit-identical | bit-identical | **pass** (PR-5) |
| P17 Gaussian-marginalization limit | reconciled | within MC error | **pass** (PR-5b) |
| P18 rung-1 inertness | — | — | **N/A**: PR-6b demoted by K9 |
| member ESS, production anchor | `ESS/M = 0.999789` → **7.998 of 8** | K5 fires below 2 of `M_draw` | **pass**, K5 unfired |
| member ESS, e2e fixture | **2.9275 of 3**, reproduced independently from `softmax(ll_m)` | — | live |
| Tier A slope (gauge-fixed) | **0.99908 ± 0.00433** | `1.00 ± 0.05` | **PASS** |
| Tier A Pearson `r` | **0.99945 ± 0.00012** | `> 0.90` | **PASS** |
| Tier A `‖ξ̂−ξ_true‖/√M` vs `√(tr H⁻¹/M)` | ratio **1.0236 ± 0.0591** | within 20% | **PASS** |
| Tier B `H0_true` in the 90% CI | **0 of 4 arms** | every arm | **FAIL** |
| Tier B latent vs table | **1.571 σ** | ≤ 0.3 σ | **FAIL**, and unmeasurable as written |
| Tier B latent CI ≥ table CI | 22.30 vs 32.64 | — | **VACUOUS** (S-2) |
| Tier C KS | **p = 0.0111**, `D = 0.3198`, n = 24 | `p > 0.05` | **FAIL** |
| Tier C median bias | **−0.1437 σ** | < 0.2 σ | pass, unresolvable at n = 24 |
| Tier C outside the 99% band | **9 of 24** | none | **FAIL** |
| Tier D (i) `f_p` perturbed 0.104 | paired **+0.059 ± 0.040 σ** | < 0.5 σ | **PASS** |
| Tier D (ii) 5% fibre proxy | paired **+0.025 ± 0.006 σ** | < 0.5 σ | **PASS** |
| Tier D (iii) `ls_ang` × 2 | paired **+0.002 ± 0.026 σ** | < 0.5 σ | **PASS** |
| Tier D (iv) non-Gaussian tail | **+0.390 ± 1.395 σ** unpaired | < 0.5 σ | pass, consistent with anything |
| **determinism, repeats — latent, production** | **100/100 bit-identical**, max dev `0.0` | bit-identical | **PASS** |
| **determinism, repeats — table, production** | **100/100 bit-identical**, max dev `0.0` | bit-identical | **PASS** |
| **determinism, sweep — latent, production, 241 pt** | max/median adjacent abs-delta = **2.539** | < 10 | **PASS** |
| determinism, sweep — table, production, 241 pt (control) | **21.65** | < 10 | **FAIL** — `−inf` rail, not the seam |
| determinism, sweep — 1e4 pt, fixture, **both modes** | **77.53** | < 10 | **FAIL**, identically in both |
| **overhead, seam + 8-member ensemble** | **+37.95 ms = +2.68%** of the no-LSS baseline | OD5 budget: +31% at `M_draw = 8` | **PASS**, 11.6× under |
| overhead, PR-6a configuration as a whole | **+1352.7 ms = +95.4%** | — | reported; 97.2% of it is PR-2's `f_p`, not the field |
| §6.5 fallback trigger (`sigma > 1.5` **and** P14 unmeetable at `M ≤ 128`) | `sigma_max = 1.1525`, P14 = 7.07e-3 | — | **NOT triggered** |
| guard matrix, independently re-verified | **11 of 11** cells | — | **PASS** |

## The determinism sweep, and what it is not

PLAN §6.4 is emphatic and this report will not let the emphasis be lost. Under
common random numbers the member ensemble is **frozen**: `laplace_draws` keys
its normals on `(n_draw // 2, M)` and nothing in the `H0` loop touches them, so
the estimator is a **deterministic smooth function of theta**. Repeat-
determinism and adjacent-theta smoothness therefore pass **by construction**,
and they pass just as well on a badly distorted surrogate (`MODEL.tex` Rem.
`rem:crn`). **This sweep is a regression guard and is never evidence that the
seam is correct.** It fires on a stray RNG re-draw, on a nondeterministic
reduction, on a discontinuous branch. It cannot fire on a seam that is
deterministic and wrong. Only P14's *theta-varying* bias discriminates, and
that is PR-5b's measurement, not this one.

**Repeats.** 100 evaluations at `H0 = 67.74`, production line, both arms:
`n_distinct = 1`, `max|dev| = 0.0` exactly. Bit-identical.

**Sweep, production scale.** 241 points over `[20, 140]`, spacing 0.5, one
evaluation each — 669 s for the latent arm at 2.78 s/point, 347 s for the table
arm at 1.44 s/point.

| arm | finite | median adj abs-delta | max adj abs-delta | at `H0` | **max/median** | `dlogL/dH0` range |
|---|---|---|---|---|---|---|
| **latent, `M_draw = 8`** | **241/241** | 0.6407 | 1.6269 | 20.5 | **2.539** | 0.033 … 3.254 |
| table, `M_draw = 8` | 231/241 | 0.5805 | 12.5711 | 25.5 | **21.65** | 0.007 … 25.14 |

The shipped latent configuration clears the gate by 3.9×. The table arm does
not, and the reason is visible in the second column: it returns `−inf` at every
node in `[20, 24.5]` **even in the clean arm**, and the finite curve adjacent to
that rail is 8× steeper than the latent one (`logL` range 316 nats against the
latent arm's 71). This is not a latent-mode defect and not a table-mode defect
— the two arms differ by `f_p` as well as by the completion model, because
`inference/loaders.py:1021` refuses a per-pixel selection fraction alongside a
`Q` table (closure finding S-3). It is recorded so that nobody reads the latent
arm's 2.539 as a comparative claim.

**Sweep, literal size.** A 1e4-point sweep on the production line is
2.78 s × 1e4 = **7.7 GPU-hours**, and was not run; PLAN §6.4's stated size was
executed instead on `tests/test_latent_seam_e2e`'s end-to-end fixture — the
whole latent stack (real `darksiren_log_likelihood`, real member vmap, real
`eval_dark_member_completion_latent`, real footprint/off-footprint split with 4
of 6 rows fitted so P13b's branch is live), at 3 events × 48 samples and
`M_draw = 3`. Two arms: latent, and the table the seam generates.

| arm | repeats | adj abs-delta median | max | at `H0` | **max/median** |
|---|---|---|---|---|---|
| latent | **100/100 bit-identical** | 8.8265e-5 | 6.8435e-3 | 98.824 | **77.53** |
| table (control) | **100/100 bit-identical** | 8.8265e-5 | 6.8435e-3 | 98.824 | **77.53** |

**Both arms fail the stated threshold, at the same point, by the same factor.**
The two curves agree to **5.35e-11** over all 10,000 nodes, so PR-6a introduces
nothing. Localized: a **derivative kink at `H0 = 98.830`**, where the slope goes
from `−5.71e-1` to `+4.15e-3` per unit `H0` over one 0.012-wide interval; the
second difference there is `1.13e-3` against a median of `4.86e-9`. The same
sweep with the missing-galaxy channel off (`lss_marginalize = False`, no field,
no table) has `max|2nd diff| = 1.18e-8` against a median `1.19e-9` — perfectly
smooth — so the kink lives in the completion channel that both `Q` modes
multiply, not in the latent seam and not in the base likelihood. **I did not
identify its mechanism.** It is not the `clip(C_bar_raw, 0, 1)` saturation (that
grid is identically zero on this fixture across the crossing) and it is not the
PE or injection `dL` range crossing `z_depth` (those crossings are at
`H0 = 92.83` and `92.67`). It is a fixture-scale feature of a synthetic
likelihood, present in the shipped table path, and it is left open rather than
explained away.

**The gate statistic itself is weak, and this rung is where that becomes
visible.** `max/median` of adjacent abs-deltas measures the *dynamic range of
`|dlogL/dtheta|` across the prior*, which converges as the spacing shrinks to a
property of the likelihood curve — where its peak is, how sharply it falls, and
whether the guard kills a rail — and not to a property of the estimator. That is
why the same code passes at 2.539 on 241 production nodes and fails at 77.53 on
10,000 fixture nodes, and why the table control fails at 21.65 while the latent
arm passes. The discriminating statistics are the two on either side of it: the
bit-identity of the repeats (passed everywhere, exactly) and P14.

## Overhead

`overhead_determinism.json`. H100 NVL (`miko`), git `377092d`, x64, value-only
path, 12 timed evaluations per arm after 3 warm-ups, `float()` on the return
forcing the device→host sync so async dispatch cannot hide work behind the
timer. `sel_batch_size = 16384`, `pe_event_block = 8`, exactly the shipped
scans.

| arm | LSS treatment | `f_p` | median ms/eval | delta vs no-LSS baseline |
|---|---|---|---|---|
| `baseline_nofp` | none | no | **1417.46** | — |
| `baseline_pr0` (PR-0's exact config: soft guard, cap 1.0) | none | no | 1429.93 | +12.5 ms = **+0.88%** |
| `baseline_fp` | none | **yes** | 2732.22 | +1314.76 ms = **+92.75%** |
| `table_m8` | `q_radial.h5`, 8 members, marginalized | no | 1428.67 | +11.20 ms = **+0.79%** |
| **`latent_m8` (PR-6a)** | latent, `M_draw = 8`, marginalized | **yes** | **2770.17** | +1352.71 ms = **+95.43%** |

Two numbers come out of that table and they say opposite-sounding things, so
both are stated.

**The seam is nearly free.** At equal `f_p` treatment the latent field plus its
8-member ensemble costs **2770.17 − 2732.22 = +37.95 ms**, which is **+1.39%**
of the `f_p` baseline and **+2.68%** of the no-LSS production baseline. The
table arm's own member marginalization costs `+11.20 ms` on its baseline, so the
`latent − table` delta at matched treatment — additive, and it has to be
additive because no configuration can carry both `f_p` and a `Q` table — is
**+26.75 ms**, `+1.89%` of baseline. PLAN §2.3 predicted `+3.3 ms` for
`latent − table` and `+8.6 ms` for latent-vs-baseline at `M_draw = 8`. Measured:
**8.1× and 4.4× the predicted absolute cost**. As *fractions*, the plan's own
currency, they are 6.4× and 11.6× **under** OWNER DECISION 5's budget (+12% and
+31%), because that budget was quoted against a 27.5 ms baseline that is 52×
smaller than the real one. **K4 cannot fire on cost.**

**The configuration is not nearly free, and the cost is not the field.** Turning
PR-6a on against the shipped production baseline is `+95.4%` per evaluation —
the likelihood roughly doubles — and **1314.76 ms of that 1352.71 ms, 97.2%, is
`--per_pixel_completeness`**, PR-2's `C_p = f_p C(z)`, which latent mode
*requires* (`factory.py` guard 6). PLAN §2.3's cost table has no `f_p` row at
all: it costs 34.7× what the entire latent seam costs, and it would be paid by
any run that adopted PR-2 with no field at all. This is the number an operator
budgeting the production run needs, and the plan does not contain it.

**A correction to PR-0's baseline.** PR-0's report quotes `3027.1 ms` and calls
it "H100, value-only path". Its job ran under `-p RITA-GPU`; `scontrol show node
rita` reports `Gres=gpu:a100-80:2`. Reproducing PR-0's arm bit-for-bit on this
session's machine — same opts, same soft-guard convention, same variance cap,
`logL` on the wall as expected — gives **1429.9 ms** over 20 timed evaluations
(an earlier 20-evaluation pass of the same arm gave 1419.2 ms, so run-to-run
spread on this node is ~0.8%). The guard convention therefore costs under 1%,
and **the 3027 ms figure is an A100-80 number, not an H100 one**;
the H100 value is 1417.5 ms, 2.14× faster. Every "3 s per evaluation" statement
inherited from PR-0 and PR-5b is partition-specific. It does not change any
conclusion — the deflation of PLAN §2's percentages is 52× instead of 110×, and
every cost gate still clears by an order of magnitude — but it should not be
re-quoted as an H100 number.

## Guards, re-verified rather than taken on trust

`verify_guards.py` does not run `tests/test_latent_guards.py` and does not read
it. It parses **real `argv`** — `experiments/desi_full259/sbatch_ns_joint_sel.sh`
verbatim, the command the production run will use — with the shipped
`build_parser()`, then calls the shipped guards in the order `cli/inference.main`
calls them, including a real `build_parameter_space`. Refusals are asserted on
the **message**.

| # | configuration | mechanism | result |
|---|---|---|---|
| R1 | latent + `--c_mode per_pixel` | guard 6, pre-load | **REFUSED** — "the per-pixel matched-kernel ratio absorbs the observed angular clustering INTO C … the two completion models would double-count" |
| R2 | latent without `--lss_field_artifact` | guard 6, pre-load | **REFUSED** — "there is no default and no in-catalog fallback" |
| R3 | `--lss_field_sha256` = 64 hex, wrong, against the **real** anchor | guard 1, build | **REFUSED** — recomputed digest quoted back |
| R4 | production config with the budget pins removed → samples `['log10n0', 'delta']` flat | guard 5, post-pspace | **REFUSED** — names "ZERO information", "shell totals", "K10", "calibrate_n0.py" |
| R5 | `selection_family = schechter` | post-fits | **REFUSED** — names `M_faint_offset` vs `m_faint_cut` and PLAN PR-2 |
| R6 | table mode + orphan `--lss_field_sha256` | pre-load | **REFUSED** — "the pin would be checked against nothing" |
| R7 | **table** mode + `--c_mode aggregate`, no `Q` (control for P3) | pre-load | **REFUSED** |
| P1 | latent + selection + `f_p` + **matching** `sha256` | guards 6 + 1 | **PERMITTED**, and stamps `lss_field_stored_sha256` / `lss_field_content_sha256` / `theta_ref` / `b_gal = 1.0` onto `opts` |
| P2 | production budget anchoring via `--fixed_parameter_values` | guard 5 | **PERMITTED, silently**; zero budget labels sampled |
| P3 | latent + `--c_mode aggregate`, **no** `Q` table | pre-load | **PERMITTED** — the mode routing is real, R7 is the same function refusing |
| P4 | `--allow_unanchored_budget` | guard 5 | **PERMITTED with a warning** that says "not a measurement" |

11 of 11. The two digests of the shipped anchor, recomputed here:
`latent_anchor_v2a.h5` stamps
`0cdfa78a4e355dc037b1c0f04f14a99a20e341931bedaa2e92ca89a82ae1cf93`, guard-1
content `4e45daa6830a341eb0a4532f75dd6e6427740ce0d6dfd8f06aeebdf5e72012e4`,
`format_version = darksirens-latent-field-1.0`, `b_gal = 1.0`. **Pin the stamped
digest** for the production run — it is what distinguishes v2a from PR-4's
`latent_anchor_a.h5`, which shares the content digest because `A_moments` /
`B_moments` are deliberately not guard-1 ingredients and are the only thing the
two artifacts differ in (2.66e-7, the eq. (4) fix).

A fourth permitted cell was verified end to end rather than at the guard: the
production `latent_m8` arm above **loaded the real artifact through
`factory._resolve_latent_leaves` and evaluated 356 times (3 warm-up + 12 timed + 100 repeats + 241 sweep)**, so guard 1's
fingerprint path and guard 6's post-load re-assertion are exercised on the
shipped file, not on a fixture.

Two gaps the build phase flagged, both confirmed by reading the code, neither
blocking:

1. **"Anchored only under the calibration prior" has one reachable branch.**
   Guard 5 accepts `("normal", loc, scale)` on `log10n0`/`delta`, but
   `inference/prior.py` writes `("uniform", …)` for every survey label and only
   `--selection_prior` flips a label to normal (selection labels only). The pin
   is real and the acceptance is real; there is no CLI route to produce the
   configuration it accepts. The production line uses the
   `--fixed_parameter_values` pin, which P2 verifies, so PR-6a is unblocked.
2. **`content_sha256` is a forward slot.** The guard verifies it when present;
   `cli/build_latent_field.py` does not write it. Until it does, the
   "artifact edited after build" defence rests on `--lss_field_sha256` plus the
   run fingerprint's input-file hash.

## Verdict

**PR-6a ships as a marginalization. PLAN §6.5's fixed-realization fallback does
not apply, and the reason is measured rather than argued.** The fallback is
conditioned on a conjunction — `sigma > 1.5` nats **and** P14 unmeetable at
`M_draw ≤ 128`. PR-5b measured `sigma` monotone in `H0` from `1.3814e-3`
(`H0 = 140`) to `1.1525` (`H0 = 20`), maximum **1.1525 < 1.5**; and P14 at the
shipped `M_draw = 8` is **7.073e-3 nat against 0.1**, met at every `M` from 4 to
128 under the shipped antithetic construction. Neither leg holds, so the
conjunction cannot fire. Member ESS at the production anchor is `7.998 of 8`, so
K5 is unfired as well. The estimator converges, the budget identity it rests on
closes to `3.8e-15` at the integral actually evaluated, the guards refuse every
configuration that would make the number unreadable, the evaluation is
deterministic to the bit under repeats, and the seam costs `+2.7%`.

**Three things that verdict does not cover, and the report will not let them
pass as covered.**

*First, P14's margin is a property of this artifact, not of `M = 8`.* PR-5b
resampled it: over 400 random antithetically-balanced 8-member subsets of a
256-member ensemble, **21.7% fail P14 over the full prior** (median `5.2e-2`,
p90 `1.31e-1`). `M = 64` is the first `M` at which none of 400 fail. The
shipped ensemble passes at `7.07e-3` and under CRN it is the only ensemble that
exists, so the gate is legitimately met — but the margin is luck, not design,
and it is a one-artifact sample. The `M = 256` reference additionally carries
its own theta-varying MC floor near `1e-2` nat, so "clears by 14×" over-reads
the data. If the owner wants the claim to be about the method rather than about
this file, `M_draw = 64` costs roughly `+0.2 s` on a 1.4 s baseline and buys it.

*Second, the mock closure fails, it fails in the arm with no field, and the fix
aimed at it has now been implemented and measured not to help.* Tier B puts
`H0_true` outside the 90% CI in **all five** arms — including `latent_off`,
which has no LSS treatment at all. The first pass attributed the failure to
under-dispersed member draws and named the missing implementation (closure S-2:
PLAN §3.4's `Cov(xi) = H^-1 + s_b^2 (dxi_hat/db)(dxi_hat/db)^T` existed nowhere
in `darksirens/`). **It was then implemented (commit `16e8195`, `s_b` measured
as the profile curvature and floored at 5% of `b_gal`) and the tiers re-run.
`CLOSURE_v2.md` reports the result: the coverage gap did not close.** Tier C at
**n = 50** — the number PLAN §6.2 asks for, with the first pass's 24 seeds
reproduced bit-for-bit inside it — gives an overconfidence ratio of **2.5515**
with the inflation off and **2.5526** with it on, a change of **+0.04%, upward**;
on the first pass's own 24 realizations it moved 2.5930 → 2.5938. Paired over
50 realizations the 90% width ratio is **1.00065** and the inflated arm is wider
in **24 of 50**, a coin flip. The whole `b_gal` propagation moves Tier B's `H0`
by **1.6e-05 σ**. Sweeping `s_b` over three decades **refutes the diagnosis
outright**: raising it from 0 to 5 inflates the member spread ×4.54 and makes the
90% `H0` interval 0.6% *narrower*, so no value of `s_b` — defensible or not —
would have closed Tier C.

*Second (continued): where the variance actually is.* Splitting each mock's
random stream immediately before the event draw gives a law-of-total-variance
decomposition over 25 catalog × event-set cells, and it is unambiguous:
**92% (`latent_off`) / 82% (`latent`) of the excess variance is present with the
catalog, the field, `f_p`, the mask, the depth map and the anchor all held
byte-identical.** The design reproduces Tier C's own ratio independently (2.348
vs 2.351, 2.603 vs 2.552). The mock's synthetic PE is width-calibrated to **6%**
(residual sd 1.0593), so it cannot supply a factor 2.4 either. **The blocking
defect is in the GW-event channel of this mock, upstream of the latent seam.**
Which defect is not resolved: the `N_obs` lever arm gives
`spread ∝ N^-0.3141 ± 0.1574` where per-event over-sharpness predicts 0.5 and a
per-realization common mode predicts 0, and `CLOSURE_v2.md` §V.2 states the
~4-GPU-hour design that would separate them. **PR-6a is not the cause and,
measurably, cannot be the fix; the ladder's Tier-B/C gate is still not met, and
the production run must not be read as validated by a closure that failed.**

*Third, the comparison PLAN wants at PR-6a still cannot be made — but one of its
three criteria is now real.* PLAN §6.2's third Tier-B gate ("latent-on CI width
≥ table CI width") was marked VACUOUS by the first pass because §3.4's
propagation did not exist. **S-2 makes it a real gate, and it fails**: 22.30
(latent) vs 32.64 (table). It fails through the `f_p` channel, not the field —
`loaders.py:1021` refuses `f_p` alongside a `Q` table while `factory.py`'s guard
6 requires it in latent mode, so **no configuration exists in which the table
and latent arms differ only by the field**, and the `1.571 σ` Tier-B gap remains
91% the `f_p` channel and 9% the field. Closure finding S-3 goes further: in
`c_mode=selection` without `--per_pixel_completeness`, off-footprint pixels are
modelled as `C_bar`-complete, and **16 of 16 footprint-limited runs railed `H0`
to 125–138 at CDF 0.0000**. Every `Q`-table configuration on a footprint-limited
survey is exposed, **including the shipped production `selq_radial` arm**. That
is a defect in the table path, surfaced by PR-6a and owed to whoever owns
`loaders.py`.

*Fourth, this branch no longer only reports its source findings — it fixes two of
them, and that changes what the PR contains.* `f45be7c` and `16e8195` carry S-1
(Armijo-damped `count_map_solve`: P6 pass rate 0.375 → 1.000 over 8 nside-16
realizations, bit-identical on the three that already converged) and S-2 into
`darksirens/redshift/latent_counts.py` and `darksirens/cli/build_latent_field.py`.
S-1 is load-bearing rather than cosmetic: the Tier-B realization's own count
operator **diverges** under the pre-fix solve (`grad_inf = 5.68e4`, `‖xi‖ = 1.3e5`
against a true 14.2), so the shipped builder would have raised its own P6 gate on
this mock. Both fixes are additive and table-mode-inert — 228 latent tests pass
and the P12 goldens are 23/23 bit-exact. **One bookkeeping defect blocks the PR:
`tests/test_latent_solve_damping.py` (22 pins, all passing) is untracked and is
in no commit.**

**What is held.** The production 259-event `H0` run awaits the owner's gate. It
was not launched, no posterior was produced, and no `H0` number for the
production line exists in this workstream. When it is taken, the run should pin
`--lss_field_sha256 0cdfa78a…`, keep `log10n0`/`delta` on the
`calibrate_n0.py` pins (guard 5 refuses the alternative), and state which
variance-guard convention it used — the hard GWTC-4/5 criterion fails at every
`H0` node on this line, so it must be lifted or cleared, and that choice changes
how every number reads.

## K9 context

The benign branch fired at PR-3: `osc_theta[a · (xi_hat_theta − xi_hat_ref)] =
5.4e-4 nat` against a 0.1-nat gate, 300× under. The GW-side field shift under
theta is negligible, so **PR-6b is a no-op on the GW channel and is demoted from
deliverable to the P18 inertness flag** — the `Δtheta = 0` branch ships because
it is the same code path, but no claim rests on it, and PR-6a is the ladder's
deliverable. The galaxy-side evidence (P7e, 59.9 nats of `dN/dz`-shape
information about `delta` and `theta_sel` against 22.8M galaxies) stays
**diagnostic-only per OWNER DECISION 13(a)** and never enters the headline
posterior; guard 5 is the mechanism that keeps it there, and R4 above shows it
firing on the exact configuration that would let it in.

## Tests

All CPU. `tests/test_unified_k1_golden.py` is run at **default** `zMax`
(`DARKSIRENS_ZMAX=1.0` shifts the grid and fails 16 cells for that reason
alone) and on CPU (`test_field_weighting_is_live` fails on GPU **on master** —
pre-existing, not this workstream's).

```
DARKSIRENS_ZMAX=1.0 JAX_PLATFORMS=cpu pytest tests/test_latent_*.py -q
    -> 228 passed, 1 skipped in 161.16s      [re-run after S-1 and S-2]
       (14 files: the 12 above -- anchor, block_sizing, cli, counts, factory,
        field, guards, p11, p13, p17, seam, seam_e2e -- plus
        b_gal_dispersion (7 pins, S-2) and solve_damping (22 pins, S-1);
        was 199 passed / 1 skipped in 126.42s before those two landed.  The
        skip is still test_latent_block_sizing.py:459 "device peak not
        measurable here: no GPU")

DARKSIRENS_GOLDEN_EXACT=1 JAX_PLATFORMS=cpu pytest tests/test_unified_k1_golden.py -q
    -> 23 passed in 59.83s                                    [P12, after S-1+S-2]
       (59.20s before them; still bit-exact, both fixes are table-mode-inert)

JAX_PLATFORMS=cpu pytest tests/test_lss_marginalization.py \
                        tests/test_marks_lss_marginalize.py -q
    -> 16 passed in 98.63s
```

Independently exercised outside pytest, on the e2e fixture: the member-ESS
diagnostic returns `2.9275458064774766` of `M = 3`, reproduced to the last bit
from `exp(−sum_m p log p)` on the returned `ll_members`; the all-dead guard
returns `0.0` rather than `nan`; `lss_member_diagnostics=True` without
`lss_marginalize` raises; and the diagnostics specialization differs from the
scalar path by `2.0e-15` (~9 ulp, XLA re-association — the diagnostics module
also consumes `logsumexp(ll_members)`).

## Artifacts

All under `experiments/field_level_plan/pr6a/`:

| file | what |
|---|---|
| `REPORT.md` | this |
| `CLOSURE.md` | Tiers A–D, the bisection, source findings S-1…S-5 (**first pass**) |
| `CLOSURE_v2.md` | Tiers B and C re-run after S-1 and S-2 landed; the `s_b` sweep, the catalog/event variance split, the PE PP test, the `N_obs` lever arm |
| `tier_b_rb_v2.json`, `tier_c_v2.json` + `.log` | Tier B at 5 arms; Tier C at **n = 50**, 3 arms |
| `variance_split*.py` / `.json` / `.log` | the 5 × 5 catalog × event-set decomposition — where the dispersion actually is |
| `s_b_sensitivity.py` / `.json` / `.log` | the `s_b` sweep that refutes the under-dispersion diagnosis |
| `pe_calibration.py` / `.json` | the mock's own PE PP test (width mis-scaling 1.0593) |
| `nobs_scaling.py`, `nobs_scaling{,_b,_c}.json` / `.log` | the `N_obs` lever arm, 22 / 6 / 22 event sets |
| `overhead_determinism.py` / `.json` / `.log` | the 4 production arms, 200 repeats, 2 × 241-point sweeps |
| `overhead_pr0_control.json` | PR-0's exact baseline arm re-measured on this machine |
| `determinism_1e4.py` / `.json` / `.log` | the literal 1e4-point sweep, both modes |
| `determinism_1e4_latent.npy`, `determinism_1e4_table.npy` | the two 10,000-point curves |
| `verify_guards.py` / `.json` | the 11 independently re-verified guard cells |
| `tests.log`, `tests_exact.log` | the suites above, verbatim |
| `world16.py`, `make_mock.py`, `build_anchor16.py`, `arms.py`, `tier_a–d.py`, `tier_*.json`, `bisection.txt` | the closure campaign |
