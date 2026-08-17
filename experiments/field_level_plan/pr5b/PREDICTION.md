# PR-5b task 1 — the closed-form member spread, predicted (2026-08-17)

PLAN §6.5 item 5, with the v4 §0.5-D4 correction to the inner product:

```
a      =  b_GW * ( sum_i phi_i Phi_i  -  N_obs * <Phi>_sel )                (6)
sigma  =  || L_H^{-1} a ||_2        [the H^{-1} norm, NOT the Euclidean norm]
```

Everything below is produced by `predict_sigma.py` (this directory), run from
`experiments/desi_full259` on `bd53d33`, against the real 259-event GWTC-5 BBH
set (4096 PE samples each), the real 1 067 946-injection plain all-sky set,
the real DESI union catalog (30 470 occupied `nside = 64` pixels, 22.8 M
galaxies), and the PR-4 anchor artifact. Raw numbers:
`sigma_prediction.json`; run log: `predict_sigma.log`; the `a` matrix and the
`H` spectrum: `a_vector_anchor.npz`. Total wall: **553 s on 8 CPU threads**
for 34 `H0` nodes including the Hessian re-assembly gate — this is an hour-scale
calculation exactly as §6.5 item 5 promised, and it did not need a GPU.

---

## 1. The number

At the anchor (`H0 = 67.74`, `theta_ref`, `b_GW = b_gal = 1`):

| quantity | value | what it is |
|---|---|---|
| `sigma = ||L_H^{-1} a||_2` | **1.0278e-1 nats** | **the prediction** |
| `||a||_2` | 1.5577e-1 nats | the Euclidean norm PLAN v3 would have used |
| ratio `||a||_2 / sigma` | **1.5156** | the size of the error the v4 correction fixes |
| `sum_i phi_i` | 1.46712 | event-weighted in-support missing-branch mass |
| in-support PE samples | 24.604 % | of all 259 x 4096 |
| in-support missing share of `mu` | 2.5661e-3 | the selection-side lever |
| `a` from the event term alone, in the `H^{-1}` norm | 1.8824e-1 | |
| `a` from `N_obs <Phi>_sel` alone, in the `H^{-1}` norm | 8.7569e-2 | |

The two branches of eq. (6) are **the same sign and partially cancel**: 0.1882
against 0.0876 gives 0.1028, so the `- N_obs <Phi>_sel` subtraction removes
about 45 % of the event-side spread. It is not a small correction and a
derivation that dropped it (v3's P17 statement) would have over-predicted
`sigma` by 1.8x on its own.

**The v4 correction is worth a factor 1.52 in `sigma` at the anchor, not the
factor 259 PR-0's number implied** (see §2). Because the bias goes as
`e^{sigma^2} - 1`, a factor 1.52 in `sigma` is a factor 2.35 in `sigma^2` and,
at these small `sigma`, a factor ~2.3 in the bias — real, worth having, and
nowhere near decisive. The correction only becomes decisive at large `sigma`,
and the place where `sigma` IS large is the low-`H0` rail (§4), where the same
ratio is 1.22 — *smaller*. So the honest statement is: **the `H^{-1}` norm
matters least exactly where it would have mattered most.** The Euclidean form
is a usable upper bound on this problem, within 25 % of the truth over the
whole region where `sigma > 0.5` nat.

---

## 2. Did we reproduce PR-0's `6.0e-4`? Yes, exactly — and it is wrong by 171x

`predict_sigma.py` carries PR-0's approximate Fisher as a control arm and
reproduces its published number to 13 significant figures:

| quantity | PR-0 (`pr0/sigma_results.json`) | this script | agreement |
|---|---|---|---|
| `sum_i phi_i` | 1.4671212117641965 | 1.4671212117641965 | exact |
| `phi_max` | 0.03927977311867687 | 0.03927977311867687 | exact |
| in-support PE fraction | 0.2460437907215251 | 0.2460437907215251 | exact |
| `f_missing_sel` | 2.5661155395590783e-3 | 2.5661155395590783e-3 | exact |
| `||a||_2` | 0.15577360748965957 | 0.1557736074896596 | 16 digits |
| `sigma` with PR-0's Fisher | 6.018525089540707e-4 | 6.018525089540753e-4 | 13 digits |

So `a` itself — the event side, the selection side, the basis, the KDE
missing-branch estimator — is byte-reproducible on today's code, five commits
and one selection-channel campaign later. **The entire difference is `H`.**

PR-0 had no anchor artifact (PR-4 did not exist) and therefore *synthesized* a
Fisher: the **unconditioned Poisson** one, `H = I + b^2 S_sph (x) S_z`, with a
sky-uniform missing base so it factorizes and can be inverted through the two
factor eigenbases without assembling `M x M`. This script instead loads the
real `H_chol` from the artifact: the **shell-total-conditioned multinomial**
Fisher of `latent_counts` eq. (3), at the real MAP, on the real footprint with
per-pixel `f_p`, rank-1 term included.

| arm | `H` | `sigma` (nats) |
|---|---|---|
| Euclidean | `H = I` | 1.5577e-1 |
| **anchor** | **`latent_counts` eq. (3), from `H_chol`** | **1.0278e-1** |
| PR-0 control | unconditioned Poisson, Kronecker-factored | 6.0185e-4 |

**PR-0's caveat has its sign backwards.** `pr0/compute_sigma.py:25-27` and
`pr0/REPORT.md:42-44` both say the unconditioned Poisson Fisher is
*conservative* because "the multinomial Fisher is smaller, so `sigma` is if
anything over-predicted". A smaller `H` is a **larger** `H^{-1}`, hence a
**larger** `sigma`. The direction of the approximation is the unsafe one, and
the measured size of the error is **171x in `sigma`, 2.9e4 in `sigma^2`** —
not the "factors, not the ~5 orders needed to change the conclusion" PR-0
asserted. Two of PR-0's stated consequences do not survive:

* "predicted `M_draw`-bias at `M_draw = 8` is ~`2e-8` nats" — the correct
  anchor figure is **6.6e-4 nats**, and at the low-`H0` rail **0.31 nats**.
* "P14/OD5: the marginalization-accuracy budget is trivially met" — predicted
  **false over the full prior**; P14 needs `M_draw >= 32` (§5).

What survives, and survives comfortably: **the 2.6-nat catastrophe of PLAN
§6.5 is still excluded.** The worst node in the whole prior is 1.34 nats, and
`M_draw = 8` is not an `ESS ~ 0.01` evaluation anywhere.

### Why the two Fishers disagree by that much

Both spectra top out in the same place (`H` eigenvalues reach 4.10e5 on the
anchor, 3.98e5 in PR-0's factored form — the 22.8 M galaxies are the same
galaxies). They differ at the **bottom**, and `H^{-1}` is a bottom-of-spectrum
object:

| `H` eigenvalue below | anchor (conditioned multinomial) | PR-0 (unconditioned Poisson) |
|---|---|---|
| 1.01 | **199** modes | 47 modes |
| 1.1 | 387 | 140 |
| 2 | 679 | 346 |
| 10 | 1023 | 665 |
| 100 | 1466 | 1160 |

Conditioning on the shell totals `T_g` is the whole point of eq. (1') — it
deletes the monopole, and with it `n0`, `H0` and the budget, from the field
posterior — but what it deletes is *information*, and in every direction it
deletes, `H = I` and the `H^{-1}` norm buys nothing. The rank-1 subtraction
`- T_g u_g u_g^T` of eq. (3) is exactly that deletion, and PR-0's Poisson form
does not have it. The band decomposition of `sigma^2` at the anchor makes the
mechanism unambiguous:

| `H` eigenvalue band | modes | share of `||a||^2` | share of `sigma^2` |
|---|---|---|---|
| `[1, 1.01)` | 199 | 42.46 % | **97.50 %** |
| `[1.01, 2)` | 480 | 0.87 % | 1.62 % |
| `[2, 10)` | 344 | 1.26 % | 0.65 % |
| `[10, 100)` | 443 | 2.46 % | 0.21 % |
| `[100, 1e4)` | 794 | 6.38 % | 0.018 % |
| `[1e4, inf)` | 260 | 46.58 % | 0.0014 % |

`a` is bimodal in the `H` eigenbasis: 47 % of its squared norm lives in the
most tightly constrained 260 modes, where the count channel annihilates it
completely, and 42 % lives in the 199 prior-dominated modes, where the count
channel does nothing at all. **`sigma` is, to 97.5 %, the projection of `a`
onto the subspace the shell conditioning threw away.** That is a physically
sensible answer — the missing-branch weight `sum_i phi_i Phi_i` is a smooth,
nearly monopolar pattern over the footprint, and a nearly monopolar pattern is
precisely what conditioning on shell totals declines to constrain — and it is
also why the answer is robust: it is set by a *geometric* projection, not by a
delicate cancellation.

---

## 3. Consistency gates the prediction had to clear first

The basis is not re-derived here; it is rebuilt with
`latent_field.build_latent_basis` at the artifact's own `basis_meta`
(`factored-v1`, `ls_sph = 0.2` chordal, `ls_z = 0.039` in `zeta = log1p z`,
`M_sph = 315`, `M_z = 8`, jitter `1e-6` per factor, `z_node_hi = 0.30`) and
then checked *against the artifact*, because a prediction quoted against the
wrong basis is not a prediction of anything.

| gate | result |
|---|---|
| `latent_counts.gradient` at the stored `xi_hat`, stored `W`, `counts`, `f_p`, `b_gal`, rebuilt basis | `grad_inf = 3.77e-9`, inside P6's 1e-8 (the build stamped 1.09e-10; the gap is accumulation ordering in the rebuilt basis — a whitening-convention or node-ordering mismatch would land at O(1), not 4e-9) |
| `latent_counts.hessian_separable` re-assembled vs `H_chol H_chol^T` | max relative deviation **2.26e-14** |
| survey occupancy vs the artifact's `fit_pixels` | identical, element for element |
| flattening order `i = i_sph * M_z + i_z` | as `latent_field.row_factor` defines it |
| `H >= I` | measured `eig_min = 1.0000000019`, `eig_max = 4.09606e5` |

**Independent confirmation from the shipped members.** The eight members in
the artifact are `xi_m = xi_hat + L_H^{-T} g_m` with antithetic `g_m`, so the
first-order offsets `ll_m - ll(xi_hat) = a . (xi_m - xi_hat)` are computable
directly from the stored draws, with no reference to the closed form:

```
predicted ll_m - ll(xi_hat), M_draw = 8:
  +0.0132, +0.1426, +0.1458, -0.0302, -0.0132, -0.1426, -0.1458, +0.0302
```

The antithetic structure is visible (rows 4-7 are exactly minus rows 0-3), and
the sd realized by those eight draws is **1.0330e-1** against the population
`sigma = 1.0278e-1` — **0.5 % agreement**, from a four-independent-sample
estimate. Two entirely different routes to the same number.

> **Artifact defect found in passing (reported, not fixed — PR-5b ships no
> code).** `build_latent_field.py:247` writes
> `create_dataset("g_members", data=draws)`, so the dataset *named*
> `g_members` holds the members `xi_m`, byte-identical to `Xi_members`
> reshaped — **not** the standard normals `g_m` its own header block (line 12)
> advertises. Verified: per-row sd 2.52, matching `xi_hat`'s 2.46, not 1.0.
> Harmless for every current consumer (the seam reads `row_fac` /
> `Xi_members`), but any future code that trusts the name and skips the
> `- xi_hat` will silently pick up `a . xi_hat` instead of the member offset —
> which is what this script did on its first pass, giving eight spuriously
> equal-sign offsets around `+0.09` and an sd 7x too small.

---

## 4. `sigma(H0)` — the prediction is a curve, not a scalar

`a` is theta-dependent; `H_chol` is not re-solved per node, because at rung 0
the shipped seam consumes ONE anchor built at `theta_ref` (PLAN §1.7; K9's
benign branch retired rung 1), and the count channel is H0-free by
construction anyway. So `sigma(H0)` moves entirely through `a`, and it moves a
lot: at low `H0` the same measured `d_L` maps to lower `z`, so far more PE mass
lands inside the catalog's `z <= 0.30` support (52.5 % of all samples at
`H0 = 20` against 7.8 % at `H0 = 140`), `sum_i phi_i` rises from 0.162 to
12.6, and `sigma` rises with it.

| `H0` | `sum_i phi_i` | in-supp % | `||a||_2` | **`sigma`** | ratio | PR-0's `H` |
|---|---|---|---|---|---|---|
| 20.00 | 12.6118 | 52.50 | 1.6399 | **1.3388** | 1.225 | 3.426e-3 |
| 23.75 | 11.1916 | 49.10 | 1.4204 | 1.1667 | 1.217 | 3.073e-3 |
| 27.50 | 9.1789 | 45.68 | 1.1195 | 0.91717 | 1.221 | 2.559e-3 |
| 31.25 | 7.4165 | 42.60 | 0.87963 | 0.71447 | 1.231 | 2.184e-3 |
| 35.00 | 5.9917 | 39.87 | 0.68790 | 0.55303 | 1.244 | 1.864e-3 |
| 38.75 | 4.8942 | 37.47 | 0.54683 | 0.43448 | 1.259 | 1.689e-3 |
| 42.50 | 4.0610 | 35.33 | 0.44767 | 0.34885 | 1.283 | 1.504e-3 |
| 46.25 | 3.3688 | 33.37 | 0.36662 | 0.27873 | 1.315 | 1.289e-3 |
| 50.00 | 2.7922 | 31.54 | 0.30247 | 0.22304 | 1.356 | 1.083e-3 |
| 53.75 | 2.3800 | 29.89 | 0.25187 | 0.18439 | 1.366 | 9.117e-4 |
| 57.50 | 2.0415 | 28.36 | 0.21126 | 0.15014 | 1.407 | 7.963e-4 |
| 61.25 | 1.7898 | 26.93 | 0.18280 | 0.12459 | 1.467 | 7.229e-4 |
| 65.00 | 1.5855 | 25.57 | 0.16829 | 0.11405 | 1.475 | 6.416e-4 |
| **67.74** | **1.4671** | **24.60** | **0.15577** | **0.10278** | **1.516** | **6.019e-4** |
| 68.75 | 1.4262 | 24.26 | 0.15253 | 0.10093 | 1.511 | 5.905e-4 |
| 72.50 | 1.2770 | 22.97 | 0.13980 | 0.084255 | 1.659 | 5.698e-4 |
| 76.25 | 1.1368 | 21.68 | 0.12801 | 0.077662 | 1.648 | 5.153e-4 |
| 80.00 | 1.0047 | 20.42 | 0.10938 | 0.060920 | 1.795 | 5.147e-4 |
| 83.75 | 0.8951 | 19.21 | 0.097183 | 0.054260 | 1.791 | 4.693e-4 |
| 87.50 | 0.7820 | 18.02 | 0.087065 | 0.045724 | 1.904 | 4.796e-4 |
| 91.25 | 0.6876 | 16.90 | 0.077433 | 0.038876 | 1.992 | 4.478e-4 |
| 95.00 | 0.6140 | 15.85 | 0.072153 | 0.031829 | 2.267 | 4.789e-4 |
| 98.75 | 0.5416 | 14.86 | 0.066211 | 0.023030 | 2.875 | 5.111e-4 |
| 102.50 | 0.4806 | 13.93 | 0.061681 | 0.017650 | 3.495 | 5.100e-4 |
| 106.25 | 0.4209 | 13.04 | 0.055050 | 0.012531 | 4.393 | 5.045e-4 |
| 110.00 | 0.3685 | 12.23 | 0.048678 | 0.0096369 | 5.051 | 4.451e-4 |
| 113.75 | 0.3274 | 11.48 | 0.044921 | 0.0080602 | 5.573 | 4.090e-4 |
| 117.50 | 0.2928 | 10.79 | 0.043678 | 0.0060629 | 7.204 | 3.669e-4 |
| 121.25 | 0.2621 | 10.17 | 0.041519 | 0.0053129 | 7.815 | 3.370e-4 |
| 125.00 | 0.2355 | 9.60 | 0.041157 | 0.0058850 | 6.994 | 3.265e-4 |
| 128.75 | 0.2115 | 9.08 | 0.042051 | 0.0072293 | 5.817 | 3.181e-4 |
| 132.50 | 0.1932 | 8.60 | 0.042807 | 0.0085219 | 5.023 | 3.151e-4 |
| 136.25 | 0.1776 | 8.17 | 0.043431 | 0.0079389 | 5.471 | 3.083e-4 |
| 140.00 | 0.1617 | 7.76 | 0.045570 | 0.0097242 | 4.686 | 3.116e-4 |

Landmarks: `sigma` is monotone in `H0` down to a floor of 5.3e-3 near
`H0 = 121` and then flattens (the residual there is the selection-side term,
not the events); it crosses 0.1 nat at `H0 ~ 69` and 1.0 nat at `H0 ~ 26`; the
maximum over the entire prior is **1.339 nats at `H0 = 20`**, the prior edge.
Note the ratio column: the `H^{-1}` correction is worth 1.22x where `sigma` is
largest and 7.8x where `sigma` is 5e-3 and nothing depends on it.

Two thresholds from PLAN §6.5, evaluated against this curve:

* **`sigma > 1.5` nats anywhere -> the fixed-realization fallback.** Predicted
  maximum is 1.339. **Not predicted to fire — but the margin is 11 %, at the
  prior edge.** This is the one place the prediction sits near a decision
  boundary at all: the anchor Fisher puts the worst node within 1.12x of the
  trigger, PR-0's Fisher put it 437x below it.
* **`sigma ~ 2.6` nats (rev 1's estimate).** Excluded everywhere: the largest
  value in the prior is 0.51 of it.

---

## 5. What the measurement phase will be compared against

### 5.1 ESS and absolute bias at the anchor

`E[ESS]/M ~ exp(-sigma^2)` and `bias = -(e^{sigma^2} - 1)/(2M)` (PLAN §6.5's
sharp form; computed with `expm1`, which at `sigma = 1e-1` is the difference
between an answer and floating-point noise).

At the anchor, `sigma = 0.102782`, so **`E[ESS]/M = 0.98949`** — the member
ensemble is 99 % efficient — and `M > 0.0531` suffices for a 0.1-nat absolute
bias, i.e. every `M_draw` on the ladder:

| `M_draw` | predicted bias (nats), `H0 = 67.74` | predicted bias (nats), `H0 = 20` |
|---|---|---|
| 4 | -1.3275e-3 | -6.2551e-1 |
| 8 | -6.6376e-4 | -3.1276e-1 |
| 16 | -3.3188e-4 | -1.5638e-1 |
| 32 | -1.6594e-4 | -7.8189e-2 |
| 64 | -8.2970e-5 | -3.9095e-2 |
| 128 | -4.1485e-5 | -1.9547e-2 |
| 256 | -2.0743e-5 | -9.7737e-3 |

`E[ESS]/M` across the prior: 0.1665 (`H0 = 20`), 0.9515 (50), 0.98949 (67.74),
0.99849 (91.25), 0.99991 (140). The low-`H0` rail is the only place the member
ensemble is meaningfully inefficient, and even there one member in six is
effective — a factor 17 from the `ESS ~ 0.01` scenario §6.5 names.

### 5.2 P14 — the gate that actually decides shippability

P14 gates the **theta-variation** of the bias, not its level:
`max_H0 [ (log Zhat_M - log Zhat_256) - mean_H0(...) ] < 0.1 nat`. Predicted,
from the `sigma(H0)` curve above:

| `M_draw` | predicted P14 over the full prior `[20, 140]` | over the clean posterior bulk `[75, 105]` |
|---|---|---|
| 4 | 5.765e-1 | 4.648e-4 |
| 8 | **2.837e-1** | 2.287e-4 |
| 16 | **1.373e-1** | 1.107e-4 |
| 32 | 6.405e-2 | 5.165e-5 |
| 64 | 2.745e-2 | 2.213e-5 |
| 128 | 9.150e-3 | 7.378e-6 |

**Predicted smallest `M_draw` meeting the 0.1-nat budget: 32 over the full
prior, 4 over the posterior bulk.** This is the operationally important
prediction and it directly contradicts PR-0's "trivially met": `M_draw = 8`,
the value the artifact currently ships, is predicted to **miss** P14 by 2.8x
if the gate is evaluated over the whole `[20, 140]` prior, and to clear it by
440x if evaluated over `H0 in [75, 105]`, where PR-0 item 3 measured the clean
fixed-population posterior to actually live (`H0 = 90 +/- 5`). **The two
readings of "across the `H0` prior" give opposite answers**, so PLAN §6.5 /
OWNER DECISION 5 has to say which one it means before P14 can be scored. The
cost of not caring is small: PR-0 item 1 measured the production baseline at
3027 ms/eval and the member seam at +3.3 ms for `M = 8` / +69 ms for `M = 64`,
so `M_draw = 32` is roughly +1 % of the real wall clock. **Take `M_draw = 32`
and the question does not arise.**

### 5.3 P17 arm (b) — the same object, measured a second way

PLAN P17 arm (b): `LSE_m ll_m - log M - ll(xi_hat) -> 0.5 a^T H^{-1} a
= 0.5 sigma^2`. Predicted at the anchor:

| quantity | value |
|---|---|
| asymptotic target `0.5 sigma^2` | **5.2821e-3 nats** |
| the same at the shipped `M_draw = 8`, evaluated on the artifact's own eight members | **5.3299e-3 nats** |

The two agree to 0.9 %, which is the statement that at `sigma ~ 0.1` an
8-member antithetic estimator is already in its asymptotic regime. Both are
first-order predictions; the measurement runs the full nonlinear seam, so the
gap between measured and predicted is the second-order content of `logQ`.

---

## 6. What would refute this prediction

A prediction that cannot be refuted is not a prediction. PR-5b's measurement
phase emits the `ll_m` vector at `M_draw = 256` at 33 `H0` nodes across
`[20, 140]` with common random numbers, which is exactly the object every
number above is a prediction of. The prediction is **refuted** if any of the
following holds.

**R1 — the level, at the anchor.** Measured `sd_m(ll_m)` at `H0 = 67.74`
differs from 0.10278 nats by more than a factor of 2 in either direction
(measured outside `[0.0514, 0.2056]`). A first-order prediction is entitled to
tens of percent; a factor of 2 means `b_GW f` is not small and the
marginalization is doing genuinely nonlinear work — which PLAN §6.5 item 5
already flags as *diagnostic rather than fatal*, but it would kill the closed
form as a design tool. Note the 256-member estimator's own sampling error on a
standard deviation is `1/sqrt(2 * 255) = 4.4 %`, so a factor 2 is more than 10
sigma of estimator noise: this criterion is not noise-limited.

**R2 — the shape.** Measured `sigma(H0)` fails to be monotone-decreasing from
`H0 = 20` to `H0 ~ 120`, or the measured ratio
`sigma_meas(20) / sigma_meas(67.74)` falls outside `[6.5, 26]` (predicted
13.0, i.e. a factor of 2 either way). The shape is driven by a *counting* fact
— the in-support PE fraction falling 52.5 % -> 7.8 % — so a wrong shape means
the missing-branch weighting in the seam is not the one eq. (6) assumes.

**R3 — the mechanism.** §2 claims 97.5 % of `sigma^2` comes from the 199
`H`-eigenvalue-below-1.01 modes that shell conditioning leaves unconstrained.
Refuted if the measured `ll_m` spread is reproduced *better* by the Euclidean
norm than by the `H^{-1}` norm — concretely, if measured `sigma` lands within
10 % of `||a||_2 = 0.15577` while sitting more than 25 % away from 0.10278.
That would mean `H_chol` is not the operator the members are actually drawn
through.

**R4 — the member-level test, which is the sharpest one available.** The
predicted per-member offsets of §3 are specific signed numbers, not a
distribution: `ll_m - ll(xi_hat)` for the artifact's eight members should be
`(+0.0132, +0.1426, +0.1458, -0.0302)` and their exact negatives. Refuted if
the measured eight-vector fails to be antithetic to 1 %, or if its correlation
with the predicted eight-vector is below 0.9, or if its sd departs from
0.10330 by more than a factor of 2. This one has no Monte-Carlo error at all —
it is eight deterministic evaluations of the shipped likelihood — so it is
the test to run first and it costs eight likelihood calls.

**R5 — P17 arm (b).** Measured `LSE_m ll_m - log M - ll(xi_hat)` at
`M_draw = 8` outside `[2.7e-3, 1.1e-2]` nats (a factor 2 around 5.33e-3), or
of the wrong sign. The sign is a real prediction: Jensen guarantees the
quantity is positive, so a measured negative value at any node means the
estimator is not evaluating what §6.5 eq. (0) says it is.

**R6 — the decision.** Measured P14 at `M_draw = 32` exceeds 0.1 nat over the
full prior, or measured P14 at `M_draw = 8` comes in below 0.1 nat over the
full prior. Either outcome falsifies §5.2's `M_draw` recommendation, which is
the only part of this document that changes what ships.

**What would NOT refute it**, and should not be reported as if it did: a
constant offset between measured and predicted `log Zhat` (CRN makes the
estimator deterministic and a constant bias is absorbed into the evidence —
PLAN §6.5 item 2); disagreement at the 10-30 % level anywhere (first-order
prediction); or the measured `sigma` at `H0 > 120` disagreeing by any factor,
since the prediction there is 5e-3 nats and both numbers are noise.

---

## 7. Caveats carried with the numbers

1. **Anchor artifact: the PR-4 build, not PR-5's `latent_anchor_v2a.h5`.**
   `experiments/field_level_plan/pr5/latent_anchor_v2a.h5` **does not exist
   yet** — its rebuild (job 1136102, `pr5/sbatch_anchor_v2.sh`) was still
   `PD (Resources)` on RITA-GPU for the whole of this work, so everything above
   uses `pr4/latent_anchor_a.h5`, sha256 `adb2841813c15c74...`, `grad_inf =
   1.09e-10`. This is believed inert for `sigma`: the eq. (4) f32 fix changed
   only how `sky_moments` contracts `(A, B)` from the stored `row_fac`
   (`build_latent_field.py:197-203`); `xi_hat` and `H_chol` come from
   `count_map_solve` upstream of it and are not touched. When v2a lands,
   re-running `predict_sigma.py --h0-scan` (9 minutes) against it confirms
   that in one shot, and the §3 gates will catch it if the belief is wrong.
2. **Uniform PE-sample weights on the event side.** This is PR-0's
   approximation, retained deliberately so the reproduction is exact. PR-0
   measured its size: the sample-wise `sum_i phi_i = 1.467` against 0.993 from
   the production likelihood itself, so the event term is over-weighted by
   1.48x. Rescaling `T_ev` by 0.6772 (the event term enters `a` linearly)
   gives `sigma = 4.3393e-2` nats at the anchor instead of 1.0278e-1, and a
   `M_draw = 8` bias of -1.18e-4 instead of -6.64e-4. **The prediction is
   therefore conservative by a factor ~2.4** in the direction that matters,
   and the refutation bands of §6 are stated around the unrescaled number
   because that is the one PR-0 published.
3. **First order in `b_GW f` throughout.** Eq. (6) is the leading term; the
   measurement runs the full seam. The gap is the physics §6.5 item 5 calls
   diagnostic.
4. **`b_GW = 1`.** `a` is exactly linear in `b_GW`, so `sigma` scales
   linearly and every entry above rescales trivially; the bias, going as
   `e^{sigma^2} - 1`, does not.
5. **`H_chol` is the anchor's, held fixed across `H0`** — correct for the
   shipped rung-0 seam (§4), and the reason the 34-node scan costs 9 minutes
   rather than 34 anchor builds.
6. The event/injection pair is the blessed Aug-10 gwcat-1.0 product with the
   known inert ~0.28 % endo3 `pdraw` caveat, the same pair PR-0 used.
