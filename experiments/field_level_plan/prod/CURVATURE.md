# Production cancels as hard as the mock (2026-08-18)

Tier C's width deficit is ~2.5x and gate 8(a) localised it to the estimator's
own terms.  The mock's `H0` width turns out to be set by a **near-cancellation**
between the event sum and the selection term, which is what makes it so
sensitive: a modest fractional error in either term is amplified into a large
error in the net.

The question for the shipped number is whether production has the same
structure.  It does.

| | events `E` | selection `S` | net | `abs(S/E)` | sigma from curvature |
|---|---|---|---|---|---|
| **mock** (`latent_off`, 60 events) | -0.064303 | +0.041988 | -0.022314 | **0.653** | 6.69 |
| **production** (`fp`, 259 events) | -0.244402 | +0.155870 | -0.088532 | **0.638** | 3.36 |

Both taken from the LIVE estimator by wrapping `selection_log_correction`, the
one function the likelihood hands `log_mu` to -- not re-derived.

**The cancellation is 64-65% in both.**  So it is a generic property of this
hierarchical likelihood, not an artifact of the small mock, and it removes one
escape route: "the mock's sensitivity is a property of its 60 events and 1,854
pixels" is not available, because production cancels just as hard.

## The conditional that follows, stated as a conditional

On the mock, matching the measured 15.57 km/s scatter requires the net
curvature to be -0.004125 instead of -0.022314.  Holding `E` fixed, that is a
**+43.3%** change in the selection term.  Applying the SAME fractional error to
production:

    net    -0.088532  ->  -0.021006
    sigma       3.36  ->       6.90  km/s
    90% CI widens by x2.05:  [65.18, 79.38]  ->  roughly [57.7, 86.9]

**This is NOT a claim that production's interval is wrong.**  Three things have
to be true for that, and only the first is established:

1. production cancels as hard as the mock -- **measured, yes**;
2. the mock's deficit is caused by an error in one of the two terms (rather
   than by something else that happens to show up as excess curvature) --
   **not established**;
3. that error is present in the production configuration too -- **not
   established, and not testable on the mock**.

What it does establish is that the closure tiers cannot be dismissed as a
small-configuration curiosity, and that the size of the effect they would imply
for the shipped interval is a factor ~2 in width rather than something
negligible.  That is the reason to finish the diagnosis before the production
90% CI is quoted as a credible interval.

The MEDIANS and the arm-to-arm SHIFTS are unaffected by any of this -- the
field's 0.06 sigma and `f_p`'s 4.31 sigma are differences between arms that
share the same estimator and the same selection term.


## The selection term has no structural pathology

If the excess curvature were the selection term's, the most likely mechanism
would be a distorted `log mu(H0)`.  Measured on production over
`H0` in [60, 86]:

    d log_mu / d log H0 = +3.814        (a smooth power law)
    measured d2 log_mu/dH0^2 = -5.862e-04
    the same power law predicts         -6.964e-04   (agrees to 16%)

`log mu(H0)` is a clean power law with a **positive** index, which is the right
sign: raising `H0` shrinks `dL` at fixed `z`, so sources are closer, louder and
more detectable, and `mu` rises.  Its curvature is consistent with that power
law to 16% over the interval that sets the width.  There is no kink, no
flattening at the edges, and no sign of the coverage failure rule 6 warns about
(independently confirmed: zero injections fall off the grid at any trial `H0`).

So the selection term is smooth and behaves as it should.  That does not
exclude an error in its INDEX or NORMALISATION -- matching the mock's scatter
would need its curvature ~43% larger, i.e. an effective index near 5.5 rather
than 3.81 -- but it does exclude the structural pathologies that would be
visible in the curve, and it shifts weight toward the other side of the
cancellation.

## Where that leaves the two candidates

* **selection term ~43% under-curved** -- no structural evidence for it; would
  require the index to be wrong by 1.7, in a quantity fixed by the injection
  set, whose coverage and noise have both been tested and cleared.
* **event/catalog term ~28% over-curved** -- untested.  With delta-PE the event
  term is the catalog redshift prior evaluated at exactly-known redshifts, so
  "the prior is more informative than the catalog warrants" is the remaining
  shape of the hypothesis, and it is consistent with gate 8(a) having localised
  the deficit to the estimator.

The second is now the more likely of the two and is where the next test should
go.


## An attempt to measure `J/H` on production, and why its answer is not quoted

The mock's `sqrt(J/H) = 2.456` must not be assumed to carry to production, so it
was measured here.  Production has one dataset, so `J` cannot come from an
ensemble; instead the score's additivity over events was used, splitting the
259 events into 10 disjoint subsets and reading `J` off the spread of the
subset scores.

The method carries its own consistency check -- since `sum_k n_k = N`, the
subset scores must SUM to the full-data score.  **They do not:**

    sum of the 10 subset scores = -0.018050
    full-data score             = -0.019870
    discrepancy                  9.2%

so the decomposition the estimator rests on is not exact here, and the number it
produced (`J/H = 0.385`, i.e. intervals too WIDE by 1.6x) is **not reported as
a result**.

Two reasons it fails, both identified rather than guessed:

1. **The selection correction is not linear in the event count.**  It is not
   simply `-N log mu`: it carries the Vitale `5 N_obs` floor and the
   total-variance criterion, and those do not decompose across subsets.  This
   is structural -- more subsets will not fix it.
2. **Ten subsets give a 47% error on a variance**, so even a valid `J` would
   read `0.385 +- 0.18` here.  That alone would not support a conclusion.

There is also a physical objection to the design: 26-event subsets are not
exchangeable with the 259-event configuration whose interval is in question --
the guards sit at different points and the likelihood is far flatter -- so even
a self-consistent subset estimator would be measuring a different regime.

**What a correct production measurement needs.**  Either per-event score
contributions from a single evaluation (an outer-product-of-gradients estimator,
`J = sum_i u_i u_i^T`, which requires exposing the per-event likelihood terms
the reduction currently sums internally), or an ensemble of synthetic 259-event
datasets built on the production catalog -- the mock route at production scale.
The first is cheap and is the better target.

**So the production `sqrt(J/H)` remains unmeasured**, and the interval caveat
stands exactly as written above: the medians and arm-to-arm differences may be
quoted, the 90% intervals may not yet be quoted as credible intervals, and the
factor-2 figure is a conditional carried over from the mock rather than a
production measurement.


## `J/H` on production by outer product of gradients

The subset route failed structurally, so `J` was taken instead from the
per-event score gradients on the single production dataset,
`J = sum_i (u_i - ubar)^2 N/(N-1)`, with the per-event log-evidences captured
from the reduction the likelihood vmaps over events.

**The first attempt failed its ordering check by 10.3%, and the failure was
diagnostic.**  The shipped selection correction is not exactly `-N log mu`: it
carries the Vitale `5 N_obs` floor and the total-variance criterion, which add a
nearly constant `+0.92` nats whose DERIVATIVE is `-0.00205` -- 10% of the total
score.  Attributing the actual correction (`logL - sum_i ll_i`) rather than the
idealised `-N log mu`, spread as `1/N` per event because the correction is
proportional to the event count, makes the decomposition exact:

    ORDERING CHECK: sum(u_i) = -0.019870   vs full score -0.019870   (0.00%)

Both estimators had failed in the same direction by the same ~10%, which is why
this was a real omission and not either method's artifact.

With the decomposition exact:

    H (observed)                       = 0.088532
    J (expected, OPG over 259 events)  = 0.044733
    J/H = 0.505      ->  width factor x0.711

## Why this is a LOWER BOUND, not "production is conservative"

Read naively this says production's intervals are ~1.4x too WIDE -- the opposite
of the mock.  That reading is not available, because the two `J`s do not measure
the same thing:

* on the mock, `J` came from an ENSEMBLE of realisations, which redraws the
  events and their host assignments and therefore captures every correlation
  between events, including those induced by the shared catalog;
* on production, the OPG estimator sums per-event variances and thereby assumes
  the events are INDEPENDENT.

Events sharing a catalog are not independent -- neighbouring events read the same
galaxies -- and positive correlation makes `Var(sum_i u_i) > sum_i Var(u_i)`.
So the OPG value can only under-state `J`:

    J_true >= J_OPG   =>   (J/H)_true >= 0.505

**0.505 is a floor.**  It does not establish that production is conservative,
and it does not exclude the mock's 6.03.  What it does do is bound the problem
from below on the real configuration and identify precisely what an unbiased
production measurement still needs: an ensemble of synthetic 259-event datasets
built on the production catalog, which is the mock route at production scale and
the only construction that carries the catalog-induced correlations.

The interval caveat is therefore unchanged: medians and arm-to-arm differences
may be quoted; the 90% intervals may not yet be quoted as credible intervals.


## The identity holds on production WITHOUT `f_p`, and not with it

The first arm comparison evaluated every arm at `H0 = 72`, which is `fp`'s peak
but 2.6 sigma from `nofp`'s (90.25).  The information identity is a statement at
the peak, so that was not like-for-like.  Re-run with each arm at its own:

| arm | evaluated at | `H` | `J` (OPG) | `J/H` | width factor |
|---|---|---|---|---|---|
| `nofp` (no `f_p`, no field) | 90.0 | 0.071762 | 0.070276 | **0.979** | x0.990 |
| `fp` (`f_p`, no field) | 72.0 | 0.088532 | 0.044733 | **0.505** | x0.711 |
| `latent` | -- | -- | -- | refused | -- |

Both ordering checks pass at 0.00%.  The `latent` arm was REFUSED by the same
check: under marginalisation the event reduction runs once per member, so the
capture returned `2072 = 259 x 8` values instead of 259.  Handling the member
axis is a straightforward extension and is not done here; the check doing its
job is why no latent number is quoted.

**Without the per-pixel completeness the identity holds** -- `J/H = 0.979`
against the 1.000 a correctly specified model requires.  **With it, `J/H` halves.**
The same estimator is applied to both, so the contrast is meaningful even though
each value is a lower bound; and `nofp` landing on 1.0 is evidence that the OPG
bound is TIGHT in this configuration rather than loose, which is what makes
`fp`'s 0.505 worth taking seriously.

## What this does and does not say

It says the `f_p` channel -- already responsible for the entire `-4.31 sigma`
shift and for 97.2% of the runtime -- is also where production's information
identity departs from unity.  That is a third independent way in which the
per-pixel completeness, not the latent field, is the consequential part of this
programme.

It does NOT say production reproduces the mock's defect.  The directions are
opposite: the mock's `J/H = 6.03` makes intervals too NARROW, production's
`f_p` arm at `0.505` would make them too WIDE.  Whatever breaks the identity in
the `f_p` arm at production scale is therefore not the mechanism that breaks
Tier C, and the two must not be conflated.  The mock's factor still has no
production counterpart, and the ensemble measurement that would supply one --
synthetic 259-event datasets on the production catalog -- remains unbuilt.

The interval caveat stands unchanged.


---

# CORRECTION: the production `J/H` values above are WRONG, and the claim built on them is withdrawn

A stability check -- re-evaluating each arm at neighbouring centres -- shows the
production numbers are not robust:

| arm | `H0` | `H` | `J` | `J/H` |
|---|---|---|---|---|
| `fp` | 70 | 0.044608 | 0.045648 | 1.023 |
| `fp` | 72 | 0.088532 | 0.044733 | 0.505 |
| `fp` | 74 | 0.089742 | 0.043603 | 0.486 |
| `nofp` | 88 | **-0.071459** | 0.073224 | **-1.025** |
| `nofp` | 90 | 0.071762 | 0.070276 | 0.979 |
| `nofp` | 92 | 0.086862 | 0.068374 | 0.787 |

`J` is stable to 4%.  **`H` is not** -- it swings by a factor 2 over 2 km/s and
goes NEGATIVE at one node.  A three-point second difference at +-2 km/s is
measuring small-scale structure in the likelihood curve, not the width.

**The check that should have been run from the start** is whether `1/sqrt(H)`
reproduces the posterior's actual width:

    MOCK        H = 0.018418  ->  sigma 7.37   against a quoted 7.5    AGREES
    PRODUCTION  H = 0.088532  ->  sigma 3.36   against a posterior 4.20  DOES NOT

The mock's `H` passes this and its `J/H = 6.03` stands (it was also verified
stable across nodes, median 6.07 over the width-setting window).  The
production `H` fails it.  Fitting a quadratic over +-8 km/s instead gives
`H_fit` that does reproduce the width -- `fp` 0.05232 -> sigma 4.37 against 4.20,
`nofp` 0.05668 -> 4.20 against 4.14 -- and the corrected ratios are:

| arm | previously quoted | **corrected** |
|---|---|---|
| `fp` | 0.505 | **0.855** |
| `nofp` | 0.979 | **1.240** |

**So the claim "the identity holds without `f_p` and halves with it" does not
survive and is withdrawn.**  Both arms are consistent with `J/H ~ 1` at this
precision; there is no measured departure from the identity in the production
configuration in either arm, and no evidence from this that the `f_p` channel
breaks it.

What survives unchanged: the mock's `J/H = 6.03`, its agreement with the
directly measured overconfidence, the cancellation structure (`|S/E|` 0.638 vs
0.653, computed from `log mu` and `logL` and not from second differences), and
the interval caveat -- which never rested on these production ratios.

The corrected reading is the more conservative one: production shows **no**
evidence of the mock's defect, and also no evidence against it, because
`J_OPG` remains a lower bound that ignores catalog-induced correlation between
events.
