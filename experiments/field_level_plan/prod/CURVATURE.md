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
