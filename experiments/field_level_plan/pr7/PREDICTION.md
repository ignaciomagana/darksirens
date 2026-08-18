# Tier E — what is predicted, written before the campaign ran

PR-7's Tier E is the gate on **differentiator 2**: one `xi` shared across K tracers makes
"member m of every catalog is the same realization" a *theorem* rather than a stamped
assertion (PLAN §4.4). This file states the predictions first so the measurement can
falsify them; `REPORT.md` carries what was measured.

## The gate, in the v4 form

PLAN §0.5 finding 12 **demoted v3's gate (iii)** — "K=2 × `c_mode=selection` ×
`--lss_marginalize` runs without `--allow_unverified_shared_lss_members`" — **to a statement
of fact**, because the flag and its check live on the table-loader path
(`inference/loaders.py:352-395`) that latent mode deletes. It would pass by deletion of the
check, which is the same routing tautology review caught in rev 1. The gate is therefore

* **(i)** bias-ratio recovery at `b_2/b_1 = 2`, within `2 sigma`, over 20 realizations;
* **(ii)** shared-`xi` coupling **demonstrably tighter** than two independent fits;
* **(iii')** the shared-`xi` likelihood **and** an artificially decoupled two-field variant on
  the **same mock**, with the bias-ratio credible region differing **in the predicted
  direction** — i.e. demonstrate the coupling the flag throws away.

## Why this is measured in the count channel and not on an H0 posterior

`experiments/field_level_plan/pr6a/CLOSURE_v2.md` §V measured that **82–92% of the excess
`H0` variance on this mock survives with the catalog held byte-identical**, and §V.1 that
the mock's PE is width-calibrated to 6%, so the defect is in the GW-event channel and is
present with no field at all (`latent_off` is overconfident ×2.35). An `H0`-based Tier E on
this mock would therefore measure the mock's PE calibration, not the tracer coupling —
CLOSURE_v2 exists precisely so that this is not rediscovered. The bias ratio `b_2/b_1` is a
**count-channel** object: it lives entirely in `log p_count(xi; b_1, b_2)`, it never touches
an event, and it is exactly the quantity PLAN §3.4 says "comes from its own 2x2 profile
curvature" at K >= 2. So Tier E is run where its gate is defined and where the mock is not
blocked.

## The mechanism, and therefore the predictions

Eq. (1')'s exponent is `eta_pg = b_k (Phi_s Xi phi~_z)_pg`. Two consequences follow before
any code runs:

1. **The overall bias amplitude is not identified by the counts.** The transformation
   `xi -> xi/s`, `b_k -> s b_k` (one common `s`) leaves every `eta` unchanged; only the ridge
   `0.5||xi||^2` sees `s`, and it *decreases* monotonically as `s` grows. The profile
   likelihood in the amplitude has no interior maximum. A log-normal prior on `log b_k` is
   therefore applied — **identically in both arms** — so that a covariance can be quoted at
   all, and the prior width is **swept** rather than chosen, because the sweep is the result.

2. **The RATIO is exactly the invariant of that direction.** `b_2/b_1` is unchanged by the
   rescaling, so in the SHARED model it is determined by the data: tracer 1's counts pin the
   one `xi`, and tracer 2's counts then read `b_2` against a field it did not have to fit.

3. **Decoupling destroys precisely that.** `decoupled_objective` gives each tracer its own
   `xi_k` and its own ridge — which is what `--allow_unverified_shared_lss_members`
   marginalizes over, in the code's own words an *independent-fields product prior*. The
   objective then separates into K independent problems, each with its own unidentified
   amplitude, and the ratio of two separately-unidentified amplitudes is determined by
   nothing but the two priors.

### Stated predictions

| # | prediction | falsified by |
|---|---|---|
| P-E1 | Shared arm recovers `r = 2`: 20/20 realizations within `2 sigma`, pull mean within `2/sqrt(20) = 0.45` of zero and pull sd of order 1 | a biased or mis-scaled pull distribution |
| P-E2 | `sigma_shared(r) << sigma_decoupled(r)` in **every** realization | any realization where the decoupled arm is tighter |
| P-E3 | The shared profile covariance has a **large positive** `corr(b_1, b_2)`; the decoupled one has **exactly zero** by construction (its Hessian is block diagonal) | a shared correlation near zero — that would say the tracers are not coupled |
| P-E4 | Sweeping the log-bias prior width `s` over a factor 16: `sigma_shared(r)` is nearly **flat** in `s`, while `sigma_decoupled(r) ~ sqrt(2) s r` (pure prior) | a shared width that tracks the prior — that would say the ratio is prior-driven, not data-driven |
| P-E5 | The decoupled arm's `r` is pulled toward the prior centre `r = 1` and its credible region **covers `r = 2` only because it is wide**, not because it locates it | a decoupled arm that locates `r = 2` sharply |

P-E4 is the one that makes (ii) and (iii') non-circular: the prior is a disclosed, identical
ingredient of both arms, and the finding is the *insensitivity* of one arm to it.

## The overlap arm — R14, written before it ran

PLAN's risk table gives R14 ("tracer overlap at K>=2, AGN subset of galaxies") the
mitigation "OWNER DECISION 9: disjoint partition" and the detection "**Tier E on an
overlapping mock**". Disjointness is structural in the shipped generator
(`counts_from_catalog_by_tracer` splits by a single-valued per-galaxy label) and in
`tier_e.py`'s own draw (K independent multinomials), so the overlap arm has to be
constructed deliberately: tracer 2 is built as a `(1 - phi)` fraction of fresh galaxies
drawn at `b_2` plus a `phi` fraction drawn *without replacement out of tracer 1's own
counts* — galaxies that are therefore distributed as `b_1`, and are counted twice by a
stacked objective that assumes they are not.

| # | prediction | mechanism |
|---|---|---|
| P-E6 | The recovered ratio is biased **toward 1** and the bias grows with `phi` | the shared galaxies carry `b_1`'s clustering, so tracer 2's effective bias is a mixture of `b_1` and `b_2` |
| P-E7 | The quoted `sigma_log r` does **not** grow to cover it, so `|pull|` grows roughly like `phi` divided by a nearly constant width | the double-counted galaxies also double-count their information: the stacked Fisher adds both copies |

If P-E6/P-E7 hold, OWNER DECISION 9 is a measured requirement rather than a stipulation,
and the number to quote is the `phi` at which the pull leaves the 2-sigma band.

## What this does NOT claim

* Nothing here is an `H0` posterior, and nothing licenses the held 259-event production run.
* The mock is drawn from the model it is fitted with (`xi_true ~ N(0, I)` through the same
  basis, counts multinomial through the same `W`). It is a recovery test of the estimator,
  not a misspecification test — Tier D is where misspecification lives.
* The shell totals are conditioned on, by construction of eq. (1'), so this measures angular
  placement only. That is the whole design (PLAN §1.1), not a limitation introduced here.
