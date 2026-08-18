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
