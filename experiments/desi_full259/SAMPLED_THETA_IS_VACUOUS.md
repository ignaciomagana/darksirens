# The sampled-theta deliverable is vacuous on this fit, and the number says why

The 259-event line has carried an open item since the ingest work: run the
selection-channel posterior with `theta_sel = (M0hat, sigma_M)` SAMPLED under the
magnitude fit's prior rather than fixed at its point estimate.  Two things were
supposed to come out of it -- `H0` marginalized over selection-model uncertainty
("the honest budget"), and the theta posterior-vs-prior pull as a
misspecification gate.

**The prior has no width.**  `selection_fit_union.json` was fit to 22,786,660
galaxies, and its Laplace covariance gives

    sd(M0hat)   = 1.60e-4 mag
    sd(sigma_M) = 1.20e-4 mag

which the CLI's own banner reports, correctly, as `M0hat=-20.310 ± 0.000,
sigma_M=0.714 ± 0.000`.  A nested run confirms it from the other end: after 1,085
iterations the two parameters' 90% intervals are `[-20.310, -20.310]` and
`[0.714, 0.715]` -- they have not moved off the point estimate and cannot.

So there is no marginalization to buy.  **The honest budget equals the
fixed-theta budget**, to four decimal places in `theta_sel`, and the
`+-5 sigma` ablation already on record (`dlogL <= 5e-4`) is consistent with
that: five of these sigmas is 8e-4 mag.

## What follows

* **The sampled-theta run is not worth its cost on this fit.**  Two of its three
  dimensions are frozen by their priors, so it is an `H0`-only run paying for a
  3-D cube.  The owner's instinct that the space should be cut was right on the
  numbers, and my defence of the three parameters -- that they buy the
  theta-marginalized `H0` -- was wrong: it assumed a prior width the fit does
  not have.
* **The misspecification gate is unaffected and remains available.**  A posterior
  that CANNOT move off the prior is exactly why a pull would be informative if
  one appeared; with `sd = 1.6e-4` the gate is simply very tight, not broken.
* **It says something about the analysis, not just the run.**  Selection-model
  uncertainty contributes nothing to this line's `H0` error budget.  Whatever
  drives the interval, it is not the magnitude fit -- and the measured
  alternatives are on record: the per-pixel mask moves the median 2.5 sigma
  (S-3), and the `f_p` channel breaks the information identity by 5.5x
  (`field_level_plan/pr6a/DIAGNOSIS.md`).

## The number to quote if a sampled-theta posterior is ever needed

Sample `theta_sel` under a DELIBERATELY WIDENED prior -- the fit's covariance
inflated to whatever the modelling uncertainty in the Schechter/gaussian choice
is actually worth, which is a judgement about the luminosity function and not a
number the fit can supply.  The fit's own covariance measures how well 22.8M
galaxies pin two parameters of an ASSUMED form; it does not measure whether the
form is right.
