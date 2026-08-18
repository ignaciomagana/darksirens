# Tier C's overconfidence: what is established, and what is not (2026-08-18)

Tier C reports an `H0` median scattering ~2.5x more than the posterior's own
quoted `sigma`, in the latent arm AND in the no-field control.  PR-6a's
follow-up refuted the first hypothesis (the missing `b_gal` dispersion: it
moved the ratio by +0.04%, in the wrong direction, and an `s_b` scan showed the
response has the wrong SIGN).  This is the second pass, run against Essick &
Fishbach's DAG checklist.

## Established

**1. The mock generator is DAG-correct.**  Read, not assumed
(`scripts/mock_dark_sirens/generate_mock_data.py:872`): the threshold is
applied to each source's recorded `rho_obs`, and "that same measurement is
carried forward under `obs_*` keys so `_posterior_samples` conditions on the
very data the threshold saw".  It also stores REJECTED proposals with their own
`obs_rho`, which is what closes the loop on "detection is a function of the
data" -- and it exhibits the Malmquist scatter (detections whose TRUE SNR is
below threshold) that a noise-free statistic cannot produce.  Checklist rules 1
and 2 are satisfied by construction.

**2. The synthetic PE is not over-sharp.**  `pe_calibration.json`: residual
sd = **1.059**, against the **2.6** that would be needed to explain the
overconfidence on its own.  The most economical explanation is ruled out with
a number.

**3. The residual MEAN of +0.486 is expected, not a defect.**  Detection selects
upward SNR fluctuations, which under-estimate `dL`, so the truth sits
systematically above the PE mean.  That offset is precisely what the
hierarchical selection term exists to absorb; its presence is a sign the DAG is
right, not wrong.  (It is why the PP's KS p = 0.0055 fails: the distribution is
DISPLACED, not narrow.)

**4. The quoted `sigma` scales correctly.**  Over `N_obs` = 30 / 60 / 120 it
runs 12.20 / 8.13 / 5.94 -- ratios 1.50 and 1.37 against the 1.41 that
`1/sqrt(N)` predicts.  The likelihood's own width is behaving.

**5. The selection integral's `Neff` varies 2.86x across the scan** --
26,310 at `H0` = 45 falling monotonically to 9,198 at `H0` = 130
(`selection_coverage.json`, captured from the LIVE estimator by wrapping
`selection_log_correction`, not re-derived).  This is a structured variation,
not the collapse rule 6 gates on, but it is not flat either.

**6. The injection set is SHARED across realizations** (`tier_c.py:44` and
`nobs_scaling.py:67` both pass `reuse_injections`).  This matters for the
diagnosis: common noise in the selection integral displaces every realization
identically, so it CANNOT produce realization-to-realization scatter.  Whatever
drives the scatter has to vary with the event draw.

## NOT established -- and an over-read to correct

The `N_obs` scan tempts a strong reading.  The measured median scatter is
18.07 / 22.51 / 15.96 at `N_obs` = 30 / 60 / 120, which looks FLAT against a
quoted sigma that is falling as `1/sqrt(N)` -- the signature of a noise source
that never averages down, and exactly what checklist rule 2 warns about.

**That reading is not supported at the sample size it was measured at.**  Each
cell has n = 6 realizations, so the standard error on a standard deviation is
`sigma/sqrt(2(n-1))` ~ **32%**.  The three numbers are:

    N_obs=30   18.07 +- 5.7
    N_obs=60   22.51 +- 7.1
    N_obs=120  15.96 +- 5.0

against `1/sqrt(N)` predictions (anchored at N=30) of 18.07 / 12.78 / 9.03.
The `N_obs=120` cell is 1.4 sigma above its prediction.  Suggestive; not
significant.  The data are consistent with flat scatter AND with correct
`1/sqrt(N)` scaling, and the difference between those two is the whole
diagnosis.

**The measurement that decides it is running**: `nobs_scaling` at n = 40 per
cell over `N_obs` = 30/60/120/240, which takes the error on the scatter from
32% to ~11%, plus Tier C at n = 100.  Until those land, the honest statement is
that the source of the dispersion is NOT yet identified -- only that it is not
the field, not the PE width, and not the b_gal draw covariance.

## What would follow from each outcome

* **Scatter falls as `1/sqrt(N)`** -> the estimator is consistent and the
  "overconfidence" is a small-sample artifact of n = 24-50 realizations.  Tier C
  needs more realizations, not a repair.
* **Scatter is flat in `N`** -> there is a common-mode term that does not
  average down.  With the generator's DAG verified and the injections shared,
  the remaining candidate is the CATALOG realization: one fixed catalog is used
  for every realization in these tiers, so any information it carries about
  `H0` is common to all of them, and the tier would be measuring catalog
  variance with an event-variance error bar.  That would make Tier C's design,
  not the estimator, the thing to fix.
