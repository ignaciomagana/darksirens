# PR-5b — the member-spread measurement and the P14 gate (2026-08-17)

No code ships from this rung. Nothing under `darksirens/` was modified. The deliverable is `member_spread.json` (the raw `ll_m` matrices), the P14 gate value, and the `M_draw` recommendation for OWNER DECISION 5.

PLAN §6.5 called the member spread "the plan's largest open number". Rev 1 estimated up to **2.6 nats**, at which `M_draw = 8` is an `ESS ~ 0.01` evaluation with 54 nats of theta-dependent bias and the marginalization is unshippable. PR-0 predicted **6.0e-4 nats** from a factored Fisher. PR-5b's first phase re-derived the closed form with the corrected `H^{-1}` norm of §6.5 item 5 (v4) and predicted **0.1028 nats** at the anchor. This rung measures it.

## The campaign (jobs 1136116 / 1136121 / 1136122 / 1136125, 62 min on one A100-40)

Production 259-event line: `GW_259` (4096 PE samples/event), `INJ_PLAIN` (1.07M detected injections), `SURVEY_N64`, `Om0 = 0.3089`, `z_depth = 0.30`, `b_GW = 1` (read back from the decoded `survey.b_miss`, not assumed). 30,470 footprint pixels, 22,787,566 galaxies, `N_z_sub = 147`, rank 2520, `P_F = 30470`, `F_F = 26251.725175964035`. 34 `H0` nodes = a 33-point `linspace[20,140]` plus the anchor 67.74. Three arms at every node — `m256` (256 members), `m8` (the shipped anchor), `map` (`ll(xi_hat)`) — 374 likelihood evaluations under common random numbers.

`ll_m` is not returned by the shipped code (`core.py:1489` returns only `logsumexp(ll_m) - log M`), so the member reduction's `logsumexp` is replaced by the identity at trace time, keyed on the call signature. `jax.clear_caches()` before every build is load-bearing: the first smoke run took a JIT cache hit, never re-traced, and returned the unpatched scalar where the member vector should have been.

**Every number below was recomputed from the raw `ll_m` matrices in this session, not read from the campaign's derived fields.**

## Gates

| gate | value | threshold | verdict |
|---|---|---|---|
| G1 same posterior as the `M=8` anchor (`xi_hat`, `H_chol`, `counts`, `f_p`, `W`) | **0.0** exactly, all five | bit-identical | pass |
| G2 P6 convergence | `grad_inf = 1.09e-10` | < 1e-8 | pass |
| G3 antithetic draws (§6.5 item 3), **re-verified from the HDF5** | `max\|xi_k + xi_{k+M/2} − 2 xi_hat\| = 1.78e-15`; `max\|g_k+g_{k+M/2}\| = 0` | exact | pass |
| G4 the 256 draws are not a superset of the 8 | `4.379`, asserted nonzero | > 0 | pass |
| patch gate, `m8` | reduced `−766.7914443725257` vs unpatched `−766.7914443725264` | δ = 6.8e-13 | pass |
| patch gate, `m256` chunk | δ = **0.0** exactly | | pass |
| chunking exactness | max dev **2.46e-11** nats (3e-14 rel) | | pass |
| CRN determinism | anchor node re-evaluated after every chunk, **bit-identical** 11/11 | | pass |
| finiteness | 34/34 nodes finite in all three arms | | pass |
| **P14, shipped balanced prefixes** | see table | **< 0.1 nat** | **pass at every M in {4,…,128}** |
| §6.5 fallback trigger (`sigma > 1.5` **and** P14 unmeetable at `M<=128`) | max sigma **1.1525** nats | | **not triggered** |

## sigma(H0) and ESS — the deliverable (`m256`, recomputed)

Strictly monotone decreasing in `H0`. Range **1.3814e-03 (H0=140) … 1.1525e+00 (H0=20)**; anchor node **1.4882e-02**. `ESS/M` range **0.834945 … 0.999998**, anchor **0.999789**.

| H0 | sigma | ESS/M | `exp(−sigma²)` | skew | ex. kurt |
|---|---|---|---|---|---|
| 20.00 | 1.1525e+00 | 0.834945 | 0.264918 | −7.43 | 74.1 |
| 27.50 | 3.3189e-01 | 0.959941 | 0.895701 | −7.24 | 81.6 |
| 35.00 | 1.1890e-01 | 0.989831 | 0.985963 | −3.64 | 27.0 |
| 50.00 | 3.7105e-02 | 0.998749 | 0.998624 | −2.74 | 15.1 |
| **67.74** | **1.4882e-02** | **0.999789** | 0.999779 | −3.51 | 32.3 |
| 91.25 | 6.1209e-03 | 0.999963 | 0.999963 | −0.06 | −0.14 |
| 140.00 | 1.3814e-03 | 0.999998 | 0.999998 | +0.04 | +0.11 |

**The lognormal law `E[ESS]/M ~ exp(−sigma²)` fails badly at the low-`H0` rail, in the safe direction**: at `H0 = 20` it predicts 0.265 and the measured value is 0.835. The reason is in the last two columns — the member log-weights are strongly **left**-skewed with enormous excess kurtosis. `sigma` there is set by a handful of catastrophic members (`max|dll_m| = 13.70` nats over the whole campaign) while the bulk of the ensemble is tight; the importance weights are governed by the bulk. So `sigma = 1.15` at `H0 = 20` is a tail statistic, not an ESS statistic, and §6.5's sharp bias table — which assumes lognormal weights — is conservative here rather than optimistic.

The campaign attributed the high ESS to antithetic pairing making the weight distribution "symmetric and light-tailed". **That is not what the data show.** Decomposing each node into even and odd parts about the antithetic pairing:

| H0 | sd(member) | sd(pair mean) | vs. independent | odd (linear) fraction |
|---|---|---|---|---|
| 20.00 | 1.1525e+00 | 8.1582e-01 | **1.0011** | 0.499 |
| 35.00 | 1.1890e-01 | 6.7285e-02 | 0.8003 | 0.680 |
| 67.74 | 1.4882e-02 | 6.8782e-03 | 0.6536 | 0.786 |
| 91.25 | 6.1209e-03 | 8.5456e-04 | 0.1974 | 0.981 |
| 140.00 | 1.3814e-03 | 7.8789e-05 | **0.0807** | 0.997 |

At `H0 = 20` antithetic pairing buys **exactly nothing** — the pair mean has the variance of two independent draws, and the response is 50% even, i.e. fully nonlinear. Pairing is worth a factor 12 in sd at `H0 = 140`, precisely where `sigma` is already 1.4e-3 and nothing depends on it.

## P14 — the gate (PLAN §6.5 item 2)

`max_H0 [ (log Zhat_M − log Zhat_256) − mean_H0(...) ] < 0.1 nat`. Absolute deviation quoted (stricter than the plan's signed max); prefixes antithetically balanced as `[0..M/2−1] + [128..128+M/2−1]`, since `laplace_draws` builds `g = [g_half, −g_half]` and the partner of member `k` is `k+M/2`.

| M | mean offset | **P14 full [20,140]** | P14 signed | P14 `H0>=30` | P14 bulk [75,105] | naive prefix, full |
|---|---|---|---|---|---|---|
| 4 | +2.970e-3 | 4.086e-2 | 4.086e-2 | 4.037e-3 | 3.267e-4 | 3.258e-1 **fail** |
| **8 (shipped)** | +2.896e-4 | **7.073e-3** | 4.725e-3 | 4.180e-3 | 2.651e-4 | 1.826e-1 **fail** |
| 16 | +2.475e-3 | 2.222e-2 | 2.222e-2 | 4.519e-3 | 2.695e-4 | 4.401e-2 |
| 32 | −6.483e-3 | 4.643e-2 | 6.530e-3 | 1.959e-2 | 6.859e-5 | 2.077e-2 |
| 64 | −9.461e-4 | 8.222e-3 | 1.266e-3 | 3.044e-3 | 1.606e-4 | 3.505e-3 |
| 128 | +7.541e-5 | 1.507e-2 | 3.172e-3 | 2.583e-3 | 1.942e-4 | 1.077e-2 |

Largest absolute bias anywhere: **5.29e-2 nats** (`M=32`, `H0=23.75`). **P14 is met at 0.1 nat by every `M_draw` from 4 to 128 with the shipped antithetic construction, and the shipped `M_draw = 8` clears it by 14x.** Antithetic balancing is decisive at small `M`: a naive `[0..M−1]` prefix is an unpaired half-ensemble and *fails* P14 at `M = 4` and `M = 8`. The shipped artifact is balanced by construction.

Two things this table does not say, and the report will not let stand:

**(i) The `M=256` reference has its own theta-varying MC noise floor.** Over 200 random splits of the 256 into two disjoint antithetically-balanced 128s, the theta-variation of the difference has median **2.01e-2** nats (p90 4.87e-2, max 8.84e-2), implying ~**1.0e-2** nats of theta-var error on `log Zhat_256` itself. Every P14 entry below ~2e-2 is at or under that floor and is not resolved; the non-monotonicity of the series in `M` is this noise, not structure. The gate at 0.1 nat is still cleared with a factor ~2 of margin at the worst `M`, but "clears by 14x" over-reads the data.

**(ii) The shipped `M=8` prefix is a favourable draw.** Repeating P14 over random antithetically-balanced subsets of the 256 (400 draws each):

| M | P14 median | P14 p90 | P14 max | fraction > 0.1 nat |
|---|---|---|---|---|
| 4 | 7.14e-2 | 1.93e-1 | 6.60e-1 | **0.335** |
| 8 | 5.22e-2 | 1.31e-1 | 3.19e-1 | **0.217** |
| 16 | 4.17e-2 | 9.93e-2 | 2.95e-1 | 0.095 |
| 32 | 2.79e-2 | 7.11e-2 | 2.08e-1 | 0.030 |
| 64 | 1.99e-2 | 4.55e-2 | 8.64e-2 | **0.000** |
| 128 | 1.11e-2 | 2.75e-2 | 5.22e-2 | **0.000** |

A *typical* 8-member CRN ensemble fails P14 over the full prior 22% of the time. The one PR-6a would actually ship passes at 7.07e-3 — and under CRN that is the only ensemble that exists, so the gate is legitimately met. But the margin is luck, not design, and it is a one-artifact sample. `M = 64` is the first `M` at which no subset out of 400 fails. Restricted to `H0 >= 30` the question evaporates: every `M` is at 2–20e-3 nats.

## Reconciliation: prediction against measurement

| | anchor (H0=67.74) | measured/predicted, median over 34 nodes |
|---|---|---|
| PR-0 factored-Fisher | 6.0185e-04 | **19.7x too small** (range 4.4x – 336x) |
| PR-5b closed form, `H^{-1}` norm (§6.5 eq. 6, v4) | 1.0278e-01 | **5.5x too large** (range 1.16x – 7.1x) |
| PLAN v3 Euclidean `‖a‖₂` | 1.5577e-01 | 10.5x too large at the anchor |
| **measured, `m256`** | **1.4882e-02** | — |

**Neither prediction is right, and the measurement sits between them** — 24.7x above PR-0's at the anchor, 6.9x below the corrected closed form. Both disagreements are real and neither is papered over.

*PR-0's 6.0e-4 is too small, and its stated reason has its sign backwards.* `pr0/compute_sigma.py:25-27` argues the unconditioned Poisson Fisher is conservative because the multinomial Fisher "is smaller, so sigma is over-predicted". A smaller `H` gives a larger `H^{-1}` and therefore a *larger* sigma. The first phase of this rung reproduced PR-0's `a` vector to 13 digits (`sum_i phi_i = 1.4671212117641965`, `‖a‖₂ = 0.1557736074896596`, `sigma_pr0 = 6.018525089540753e-4`) on today's code, so the entire gap is `H`. PR-0's downstream statements — "`M_draw=8` bias ~2e-8 nats", "P14/OD5 trivially met" — do not follow from its own arithmetic. Its *conclusion* is nevertheless vindicated by measurement, for a different reason.

*The corrected closed form is too large, in the direction P17 predicted in advance.* The P17 phase showed that §1.6/§6.3's stated arm-(a) target drops the budget normalizer `rho`, which the shipped seam applies at every evaluation and which contributes at the same order `b²`; carrying it through replaces the raw kernel by its **monopole-projected** form, `psi_i = Phi(x_i) − <Phi(·,z_i)>_F`, and measured `c_inf = +3.042196` against `c_naive = +26.931816`, a factor 8.9. Equation (6)'s `a` uses the same unprojected `Phi_i`, so it must over-predict `sigma` by whatever the monopole carries — and it does, by a median 5.5x. That is a *mechanistic* reconciliation made before the measurement, not a fit after it.

**What PLAN §6.5 says a discrepancy means, and what this one actually means.** §6.5 reads a prediction/measurement gap as evidence that `b_GW f` is *not* small — the regime where the marginalization does real work rather than reproducing a Gaussian. **That reading does not apply here, because the sign is wrong for it**: the measurement is *below* the linear-response prediction almost everywhere, and a nonlinearity that mattered would push the spread up, not down. The gap is a **missing projection in the closed form**, not a failure of linear response. The one place the two nearly agree is `H0 = 20` (measured/predicted = 0.861) — and that is exactly the node where the response *is* genuinely nonlinear (50% even part, skew −7.4). The agreement there is a coincidence of two errors, not a success.

*P17 arm (b) — the prediction's premise fails, with a mechanism.* `LSE_m ll_m − log M − ll(xi_hat)` on `m256` is **negative at every node below `H0 = 83.75`**, anchor value **−5.605e-4** nats, turning positive above and peaking at +1.79e-4 near 102.5. The closed form `→ 0.5 aᵀH⁻¹a = 0.5 sigma²` assumes `ll` linear in `xi`, so only the Jensen term survives and the result must be positive. At the anchor: `mean_m(dll) = −6.694e-4` (curvature) against `0.5 sd² = +1.107e-4` (Jensen) — the negative second-order curvature dominates by ~6x. On `m8` the same quantity is **+1.882e-4**; the sign is member-set dependent, which is the statement that 8 members do not resolve it.

*R4, the pre-registered zero-MC-error test.* Per-member `dll_m` at the anchor, `m8` arm, against the vector `PREDICTION.md` published in advance:

| m | measured | predicted | ratio |
|---|---|---|---|
| 0 | +1.262923e-02 | +1.323711e-02 | **0.954** |
| 1 | −3.322497e-03 | +1.426287e-01 | −0.023 |
| 2 | −1.187115e-02 | +1.457755e-01 | −0.081 |
| 3 | +1.315404e-02 | −3.020071e-02 | −0.436 |

(members 4–7 are the exact antithetic negatives, both sides). Uncentered cosine — the right measure, since a linear-response prediction has no intercept — is **−0.5461**; sd ratio 0.1065. The prediction gets the antithetic pair 0/4 right to **4.6%** and members 1, 2 wrong by factors of 43 and 12 *and in sign*. It is not a global sign flip or a global rescaling; the closed form has one direction right and the rest wrong. (The centered 4-vector correlation is −0.9475, but centering a 4-point sample is not a meaningful statistic here.) Measured antithetic residual on the `m8` `dll` vector is 5.99e-4, 5.45% of rms — the ensemble is antithetic in `xi` exactly (1.78e-15) but only to 5% in `ll`.

Scored against the six pre-registered criteria in `PREDICTION.md` §6: **R1 refuted** (1.488e-2 vs band [0.0514, 0.2056]); **R2 refuted** on dynamic range (`sigma(20)/sigma(anchor) = 77.44` vs band [6.5, 26]) though its monotone leg holds; **R3 pass** (the `H^{-1}` direction is right — measured/Euclidean = 0.0955 — it simply shrinks sigma further than predicted); **R4 refuted**; **R5 refuted** (sign negative below `H0 = 83.75`, which R5 named as its own refutation); **R6 refuted** (P14 at `M=8` is 7.07e-3, not the predicted 2.8x miss). The closed form got the sign, the mechanism and the monotone shape right, and the level, the dynamic range, the per-member structure and the `M_draw` decision wrong.

## Verdict

**The 2.6-nat catastrophe of §6.5 is excluded by measurement.** The worst member spread anywhere in the `H0` prior is **1.1525 nats**, at `H0 = 20`, where `ESS/M` is still **0.835**; over the posterior bulk it is 5e-3 to 1e-2 nats and `ESS/M` is 0.99996. P14 is met at 0.1 nat by every `M_draw` from 4 to 128 with the shipped antithetic construction, and by every `M_draw >= 16` even with an unpaired prefix. **K5 is unbinding and PR-6a ships as a marginalization.**

§6.5's fixed-realization fallback — quote the field at `xi_hat` and `±1 sigma` and report the `H0` spread — **is not triggered**: it requires `sigma > 1.5` nats *and* P14 unmeetable at `M <= 128`, and **neither half holds** (max sigma 1.1525 < 1.5; P14 met at every `M >= 4`). It is not needed and should not be adopted.

This vindicates PR-0's operational conclusion while refuting its arithmetic, and refutes the corrected closed form's level while confirming its mechanism. The honest summary for the paper is that the completion field's *posterior uncertainty* is negligible in the GW likelihood on this catalog/event pairing — which, with PR-3's `osc_theta = 5.4e-4` nat for its *theta*-dependence, means what the field contributes is its MAP placement (PR-0's 10.85-nat Q-on/Q-off oscillation) and nothing else. The ensemble machinery is correctness insurance, and by the cost table below it is nearly free.

## OWNER DECISION 5 — recommendation

**Budget: P14 at 0.1 nat of theta-varying bias across the `H0` prior, as recommended in §6.5. Ship `M_draw = 32`.**

Cost, measured on this rung (A100-40, 374 evaluations; linear fit through `M = 1, 8, 32` gives **4.92 ms/member**, intercept 7003.6 ms):

| M | per-eval | vs. `M=1` latent | vs. PR-0's 3027 ms no-LSS baseline |
|---|---|---|---|
| 1 | 7003.2 ms (measured) | — | +131.4% |
| 8 | 7049.9 ms (measured) | +0.67% | +132.9% |
| 16 | 7082 ms | +1.13% | +134.0% |
| **32** | **7160 ms (measured)** | **+2.25%** | **+136.6%** |
| 64 | 7318 ms | +4.50% | +141.8% |
| 256 | 8263 ms | +18.0% | +173.0% |

**PLAN §2.3's projected `M_draw` costs (+12% at 8, +26–47% at 32, +53–95% at 64) are wrong by an order of magnitude — far too pessimistic.** The latent seam costs +131% over the no-LSS baseline *at `M = 1`*, and that is paid whether or not the marginalization runs; the incremental cost of members on top is 4.92 ms each, i.e. **+2.25% at `M = 32` and +4.5% at `M = 64`** against the latent likelihood one actually runs. The `M_draw` axis is not where the money goes.

The recommendation is `M_draw = 32` rather than the shipped 8 for one reason only: `M = 8` passes P14 as-shipped (7.07e-3) but 22% of alternative balanced 8-member ensembles do not, and the margin at `M = 8` is a property of one artifact rather than of the estimator. `M = 32` cuts that to 3% and `M = 64` to 0/400, at +2.25% and +4.5% wall. **`M_draw = 8` is defensible on the measurement and would not be wrong to keep** — it meets the gate on the ensemble that will actually run — but 32 buys robustness for two percent. If the analysis is restricted to `H0 >= 30`, `M_draw = 8` is unambiguously sufficient (P14 = 4.18e-3).

## Consequences downstream

- **PR-6a ships as a marginalization** at `M_draw = 32` (or 8, per the above), frozen-anchor ensemble, antithetic pairs mandatory — an unpaired prefix fails P14 at `M <= 8` by a factor of 26.
- **K5 does not fire.** The `M_draw` axis is closed as a risk.
- The `M = 256` reference's ~1e-2 nat theta-var noise floor means any future tightening of P14 below ~0.05 nat needs a larger reference, not a larger `M`.
- §6.5's sharp bias table (`bias = −(e^{sigma²}−1)/2M`) should be annotated as an upper bound on this line: it assumes lognormal weights and over-predicts the ESS collapse by 3.2x at the worst node.
- Equation (6) should carry the monopole projection (`psi_i = Phi(x_i) − <Phi(·,z_i)>_F`) that P17 derived, or be labelled an upper bound. As written it over-predicts `sigma` by a median 5.5x, and the same `a` is P7c's gate statistic — PR-3's `osc_theta = 5.4e-4` nat is therefore also an over-estimate, in the benign direction.

## Anchor artifact — the fallback caveat is discharged

The campaign ran against `pr4/latent_anchor_a.h5` because `pr5/latent_anchor_v2a.h5` did not exist when it was submitted. **Job 1136102 has since completed (12:09), and I compared the two artifacts directly:**

| dataset | pr4 vs v2a |
|---|---|
| `xi_hat`, `H_chol`, `counts`, `completeness`, `row_fac`, `Xi_members`, `b_nodes`, `z_sub` | **bit-identical (0.0)** |
| `A_moments`, `B_moments` | rel. dev. 2.66e-07 / 2.63e-07 |

The eq. (4) f32 fix touches only the `(A,B)` contraction. The closed-form prediction consumes `xi_hat`, `H_chol` and `row_fac` only, so it is *exactly* unaffected. The `m256` and `map` artifacts were built with today's `sky_moments` and share `xi_hat`/`H_chol` bit-identically with v2a, so **the deliverable arm is exactly what it would have been on v2a**. Only the `m8` arm's `(A,B)` predate the fix, at 2.7e-7 relative — five orders below anything quoted, and the reason P17 arm (b) is quoted from `m256`.

## Guard convention

Quoted throughout: PR-0's clean arm (`selection_neff_soft_guard=False`, `max_likelihood_variance=1e6`, Vitale `5 N_obs` floor kept). `guard_arm.json` measures the alternatives at the production cap 1.0 over 5 nodes at `M_draw=8`: `guard_hard` returns **`−inf` at all 5 nodes** (reproducing PR-0 item 3), and `guard_soft` returns `logL ~ −2e6` with member spreads of **295.7 / 11.47 / 23.96 / 40.23 / 35.73 nats** — 610x to 34,206x the clean arm, non-monotone in `H0`, entirely the guard wall responding to member-dependent `Neff` and `log mu`. Run in the shipped-scan convention, PR-5b would have reported `sigma ~ 24–40` nats, "confirmed" the 2.6-nat catastrophe and killed PR-6a on an artifact of the guard. The campaign's own inline `m8_soft_guard` arm inherited the lifted cap and is vacuous (offset exactly 0.0); use `guard_arm.json`, not that arm.

## Test suite — 2 FAILURES, and the pin is zmax-sensitive

```
DARKSIRENS_ZMAX=1.0 JAX_PLATFORMS=cpu pytest tests/test_latent_seam.py \
    tests/test_latent_seam_e2e.py tests/test_latent_p17.py -q
  -> 2 failed, 29 passed in 74.86s
```

| file | `DARKSIRENS_ZMAX=1.0` | `DARKSIRENS_ZMAX=6.0` |
|---|---|---|
| `tests/test_latent_seam.py` + `tests/test_latent_seam_e2e.py` | **24 passed** (24.3 s) | — |
| `tests/test_latent_p17.py` | **2 failed, 5 passed** | **7 passed** (48.1 s) |

Failing at zmax 1.0:
- `test_p17_member_weight_is_the_closed_form_logq` — "member log-weight departs from `sum_i logQ(x_i)` by 3.40e-04 relative".
- `test_p17_convergence_rate_is_b_to_the_fourth` — per-step residual ratios `[3.632, 3.780, 3.845, 3.843]`, below the asserted 3.7 floor (b⁴ → 4.00).

`tests/test_latent_p17.py` contains no `DARKSIRENS_ZMAX` handling of any kind and inherits the environment; the seam tests do not either but are insensitive to it. The P17 phase's reported "7 passed, 42 s" was obtained at the production zmax of 6.0. **This is a real defect in the pin, not a measurement problem: a test whose tolerances hold only at one zmax will fail in any CI that does not pin it.** The fix is a one-line `monkeypatch.setenv("DARKSIRENS_ZMAX", "6.0")` fixture (or a widened b⁴ band); PR-5b ships no code and did not apply it. `tests/test_latent_p17.py` is untracked in the working tree.

## Production defects found on the way in (reported, NOT fixed)

- **D1 — `--per_pixel_completeness` cannot reach the real loader, so `--lss_field_mode latent` cannot run at all.** `darksirens/inference/loaders.py:1044` reads `data["ngals"]`, but `darksirens/inference/data.py:196` stores full-sky counts as `ngals_catalog`. `load_all_data` raises `KeyError: 'ngals'` before `f_p_map` is attached. Fatal for the latent seam specifically, because `darksirens/likelihood/factory.py:398` makes `--per_pixel_completeness` mandatory in latent mode. The only coverage, `tests/test_per_pixel_completeness.py:258`, hand-builds `data = dict(nside=2, ngals=...)` and cannot see it. One-line fix: `data.get("ngals", data.get("ngals_catalog"))`. Worked around here by aliasing the key around `load_all_data` (`latent_harness._ngals_key_shim`). **Verified in this session.**
- **D2 — the `b_miss → b_GW` inversion never reaches the factory.** PLAN §4.3's rule inversion lives only in `darksirens/cli/inference.py:3225` (`_b_miss_identified = use_LSS or _latent`); `darksirens/inference/parameters.py:530` re-derives the space with `use_lss=bool(opts.use_LSS)`, so every non-CLI caller is refused a fixed `b_miss` and `b_GW` is pinned at the `SurveyParams` fiducial 1.0. That is what PR-5b wanted, but a `b_GW != 1` study is currently CLI-only. **Verified in this session.**
- **D3 — `darksirens/cli/build_latent_field.py:247` writes `create_dataset("g_members", data=draws)`**, so in `pr4/latent_anchor_a.h5` the dataset named `g_members` holds the members `xi_m`, byte-identical to `Xi_members` reshaped — not the standard normals its own header (line 12) advertises. **Verified: per-row sd 2.5166, matching `xi_hat`'s scale, not 1.0.** The `m256`/`map` artifacts built on this rung write true normals (per-row sd 0.9996) and stamp `pr5b_g_members_are_normals = True`. Anything trusting the name and skipping `− xi_hat` silently picks up `a·xi_hat`.

## What could not be measured

- **`sigma` under the eq. (4) f32 fix was not re-measured**, only shown to be unaffected by construction. No arm was re-run on `v2a` end to end.
- **The `M = 256` reference is not converged to the P14 floor.** Its own theta-var MC error is ~1e-2 nats, so P14 differences below ~2e-2 between values of `M` are noise. Resolving them needs `M = 1024` or more, ~3.5 h.
- **Only one member artifact (seed 22) exists.** The subset-resampling in the P14 robustness table draws from within those 256 members and is therefore *not* an independent check of the artifact itself. A second seed would settle whether `M = 8`'s comfortable margin is generic; it was not run.
- **`dA_moments`/`dB_moments` were skipped** in the `M=256` build (zero-width theta axis, `pr5b_dA_dB_skipped = True`) at a measured 107 ms per (member, b) pass = 15.1 min at `M = 256`. Nothing on this path reads them; a PR-6b consumer will fail on shape rather than silently read a zero correction.
- Timings are A100-40 (TWIG), not the A100-80 (RITA) PR-0 timed on. The 7003 ms `M=1` figure is **not** comparable to PR-0's 3027 ms; only the 4.92 ms/member slope is measured cleanly on one device.

---

## Files (all absolute)

- `/hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5b/member_spread.json` — raw `ll_m` matrices (`m256` 34×256, `m8` 34×8, `map` 34×1)
- `.../pr5b/sigma_prediction.json`, `PREDICTION.md`, `a_vector_anchor.npz`, `predict_sigma.py`
- `.../pr5b/latent_anchor_m256.h5` (319 MB, sha `480a7f23…`), `latent_anchor_map_m1.h5` (53 MB, sha `7d2bd21b…`)
- `.../pr5b/guard_arm.json`, `anchor_m256_build.json`, `smoke_latent.json`
- `.../pr5b/latent_harness.py`, `build_anchor_m256.py`, `run_member_spread.py`, `run_guard_arm.py`, `summarize.py`
- logs: `anchor256_1136116.out`, `smoke_1136121.out`, `campaign_1136122.out`, `guard_1136125.out`
- `/hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5/latent_anchor_v2a.h5` — now exists (12:09), sha `0cdfa78a4e355dc037b1c0f04f14a99a20e341931bedaa2e92ca89a82ae1cf93`
- **`REPORT.md` was NOT written** (harness blocks subagent report files) — content above.

---

## Appendix — what the report changed against the campaign's own claims

The harness blocks subagents from writing report files, so `REPORT.md` was **not** written to disk — its full content is below and must be written by the caller to `/hildafs/projects/phy230014p/magana/src/darksirens-dev/experiments/field_level_plan/pr5b/REPORT.md`.

---

# WHAT I CHANGED vs. THE INPUT CLAIMS (read this first)

I reloaded `member_spread.json` and recomputed everything from the raw `ll_m` matrices. Most numbers reproduce exactly (`sigma` to 0.0, `ESS/M` to 1.9e-13, all shipped-prefix P14 values). **Five claims did not survive**:

1. **Test suite FAILS.** The campaign/P17 phases reported "7 passed". At the requested `DARKSIRENS_ZMAX=1.0` it is **2 failed, 29 passed**. `tests/test_latent_p17.py` passes only at `ZMAX=6.0` and contains no zmax handling — the pin is silently zmax-sensitive.
2. **The shipped `M=8` P14 margin is luck.** 21.7% of random antithetically-balanced 8-member subsets *fail* P14 over the full prior (median 5.2e-2, p90 1.31e-1). "Clears by 14x" is a one-artifact statement.
3. **The `M=256` reference has a ~1e-2 nat theta-var MC noise floor** (median split-half 2.01e-2). Every P14 entry below ~2e-2 is unresolved; the non-monotone series in `M` is that noise.
4. **"Antithetic pairing makes the weight distribution symmetric and light-tailed" is false.** At `H0=20` pairing buys *exactly nothing* (pair-mean sd / independent = 1.0011). ESS survives because the weights are strongly **left-skewed and heavy-tailed** (skew −7.43, ex. kurt 74.1), not light-tailed.
5. **The anchor fallback caveat is discharged.** Job 1136102 finished at 12:09. I compared `pr4/latent_anchor_a.h5` against `pr5/latent_anchor_v2a.h5` directly: `xi_hat`, `H_chol`, `counts`, `completeness`, `row_fac`, `Xi_members`, `b_nodes`, `z_sub` are **bit-identical (0.0)**; only `A_moments`/`B_moments` differ, at 2.66e-7 relative. The `m256` deliverable arm is exactly what it would have been on v2a.

Minor: the campaign's R4 correlation `−0.5461` is the uncentered 8-vector cosine and is correct as the right measure; the 4-vector *centered* correlation is `−0.9475`, which is misleading (centering a 4-point sample). I quote the cosine. Also, D1/D2/D3 all verified independently.

---
