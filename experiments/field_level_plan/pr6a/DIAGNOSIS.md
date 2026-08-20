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


---

# Pass 3 (2026-08-18): the selection-noise hypothesis, REFUTED

The hypothesis from pass 2's measurements was that the SHARED injection set's
Monte-Carlo noise adds a common, structured distortion to every realization's
`log mu(H0)` -- which would narrow every posterior while leaving the ensemble
centred, i.e. every symptom Tier C has.  The measured noise supported it: the
MC sd of the `-N_obs log mu` term runs 0.37-0.63 nats against a 0.279-nat
curvature over +-5 km/s.

**It is wrong.**  Rebuilt the injection set 18x larger with NOTHING else
changed -- same world (`make_mock.build` does `world = world or
W16.build_world()`, so both sets are matched to the same one), same seeds, same
catalog, same events -- and verified the two sets are statistically the same
population before using it:

| | original | 18x |
|---|---|---|
| Ndraw | 3,000,000 | 55,000,000 |
| detected | 65,791 | 1,201,469 |
| detected fraction | 0.021930 | 0.021845 |
| `log mu` proxy | +15.458649 | +15.454407 |
| **stamped Neff** | **1,953** | **12,618** |

Tier C at n = 24, the same seeds:

| arm | original | 18x injections |
|---|---|---|
| latent | 2.593 | **2.662** |
| latent_off | 2.259 | **2.413** |

A 6.5x improvement in `Neff` -- which should cut the MC noise by 2.5x -- moved
the overconfidence **up**, within noise.  The selection integral's Monte-Carlo
noise is not the cause.  Second hypothesis refuted with a controlled
experiment, after the `b_gal` draw covariance.

## What pass 3 did find: a rule-7 signature

Checklist rule 7 -- "a depth/selection knob must truncate the rows the
likelihood actually reads ... verify on the built artifact, never the builder's
intent" -- fires on this mock:

    catalog rows the prior reads: max z = 0.3847,  z_depth = 0.30
    9,384 of 192,757 galaxies (4.868% of count AND of weight) sit above it
    1,594 of 1,854 occupied rows contain at least one

and `z_depth = 0.3` is carried as an HDF5 **attribute**, which is precisely the
metadata-only pattern the rule names.  The precedent's cost was large
(+34...+57 terms and a fake non-monotone depth curve).

**This is NOT yet a diagnosis.**  The shipped code has a correction for exactly
this configuration (`redshift/prior.py`: `Nobs = Nobs * exp(kernels.log_depth_mass)`,
whose comment records that using the full row count with a depth-renormalised
kernel is "a double count that measured 2.6x on a 10-galaxy row with
z_depth=0.3").  So the 4.87% may be absorbed.  Whether it is, on this mock, is
the next controlled test -- and 4.87% of the catalog is a priori too small to
produce a 2.5x width error on its own, so the honest prior is that this is a
real DAG blemish worth fixing but probably not the cause either.

## Ruled out so far, each with a number

1. the latent field itself (the no-field control is equally affected)
2. the `b_gal` draw covariance (moved it +0.04%, and the `s_b` response has the
   wrong SIGN over three decades)
3. synthetic PE over-sharpness (residual sd 1.059 against the 2.6 required)
4. DAG violations in the event generator (rules 1 and 2 verified by reading:
   threshold on `rho_obs`, that measurement carried into the PE, rejections
   stored)
5. **selection-integral Monte-Carlo noise (this pass, 6.5x `Neff`, no change)**


---

# Two mundane explanations checked and eliminated (2026-08-18)

Before chasing further physics, the two ways a "2.5x overconfidence" can be an
artifact of how it is MEASURED rather than a real defect:

**Is the quoted `sigma` even comparable to the median scatter?**  It is.  The
posteriors are unimodal and near-Gaussian, so a 68%-interval half-width is a
fair standard deviation:

| | peaks | `sigma_moment / sigma_68` | skew |
|---|---|---|---|
| production `fp` | 1 | 1.017 | +0.16 |
| production `latent` | 1 | 1.017 | +0.15 |
| mock `latent_off` | 1 | 1.010 | +0.29 |
| mock `latent` | 1 | 1.012 | +0.28 |
| mock `table` | **6** | 1.065 | +0.49 |

(The `table` arm is multimodal, which is worth knowing on its own -- but it is
not the arm that fails.)

**Is the deficit driven by a few realizations with anomalously small widths?**
No.  The widths are stable across realizations (spread 18-19%), and the deficit
survives when each realization is scored against its OWN width:

| arm | `sd[(median - mean)/own sigma]` | coverage at nominal 90% |
|---|---|---|
| `latent_off` | **2.303** | **0.38** |
| `latent` | **2.858** | **0.38** |

Calibrated would be 1.000 and 0.90.  So the deficit is real, robust, and a
WIDTH problem: each posterior is individually about 2.3-2.9x too narrow for the
scatter its own estimator produces.

## The gate that has not been run, and would separate the two possibilities

Checklist gate 8(a): **machinery closure with delta-function PE at truth**.
Every test so far has varied something downstream of the likelihood's core.  A
delta-PE run removes the PE as a noise source entirely, so:

* correct coverage under delta-PE -> the likelihood is calibrated and the
  dispersion enters through the synthetic PE or the event construction;
* wrong coverage under delta-PE -> the width deficit is in the estimator's
  catalog/selection terms and is independent of the mock's PE.

That is the next test worth building, and it is the checklist's own FIRST gate
-- everything else is downstream of it.


---

# The variance split at 8x8 (2026-08-18)

The first split ran a 5x5 grid (25 cells).  This is 8x8 (64), which tightens
every number below by ~1.6x in error.

| | `latent_off` | `latent` |
|---|---|---|
| total spread | 17.41 | 19.18 |
| **overconfidence, total** | **2.360** | **2.753** |
| event spread at FIXED catalog | 15.72 | 16.63 |
| **overconfidence, events only** | **2.131** | **2.388** |
| catalog spread (debiased) | 7.93 | 10.13 |
| mean quoted `sigma` | 7.38 | 6.97 |
| fraction of variance from events | 0.816 | 0.752 |

*(Corrected 2026-08-18 against the harvested `variance_8x8.json` and an
independent recomputation from its 64 raw cells: the first four rows above were
transcribed from an intermediate and three of them were wrong -- event spread
15.57/16.49, events-only 2.110/2.367, and a fraction quoted as a range.  The
`latent` catalog spread was left blank and is 10.13.  Nothing in the reading
below changes.  The fraction is `var_within / var_total`; debiasing the
between-catalog variance for the finite 8 event sets per catalog instead gives
0.797 and 0.730.)*

Two things this settles.

**The deficit survives conditioning on the catalog.**  Holding the catalog
fixed and varying only the event draw still gives 2.13-2.39.  The analysis
treats the catalog as observed data, so its coverage ought to hold conditional
on that catalog -- and it does not.  Whatever is wrong is not "the tier varies
the catalog while the posterior conditions on it".

**There is a modest bias as well as a width deficit.**  Grand mean **64.10**
(`latent_off`) and **60.54** (`latent`) against `H0_true` = 67.74 -- i.e. -3.6
and -7.2 km/s, -0.5 and -1.0 of the quoted `sigma`.  The cell medians span 28.1
to 100.5 (`latent_off`) and 24.3 to 98.2 (`latent`); the worst single cell sits
6.7 and 11.2 of its OWN `sigma` from the truth.

This is what motivates running the checklist's first gate rather than a sixth
downstream variation.


---

# GATE 8(a) FAILS: the width deficit is in the ESTIMATOR, not the mock's PE

Ran Tier C with each event's PE samples collapsed onto that event's TRUE
parameters -- no parameter-estimation uncertainty at all.  Same seeds, same
arms, same catalog, `latent_off` (the no-field control, the simplest arm that
exhibits the deficit).  Construction verified on the artifact: max within-event
`dL` spread = 0.000e+00 exactly.

| | real PE | **delta-PE at truth** |
|---|---|---|
| median scatter | 17.49 | **17.61** |
| mean quoted `sigma` | 7.74 | **6.71** |
| **overconfidence** | **2.259** | **2.624** |
| coverage at nominal 90% | 0.38 | **0.42** |
| `cdf` mean | 0.493 | 0.524 |

**Removing the PE uncertainty entirely narrows each posterior (7.74 -> 6.71,
the expected direction, which confirms the intervention took effect) and leaves
the scatter untouched (17.49 -> 17.61).**  So the overconfidence gets slightly
WORSE, and the scatter is demonstrably not coming from the parameter
estimation.

By the gate's own reading, that settles the fork:

* the synthetic PE is **not** the source of the dispersion;
* the event construction is **not** the source (the events carry their exact
  true parameters here);
* **the width deficit lives in the estimator's own terms** -- the catalog term,
  the completion, or the selection integral -- and is independent of the mock's
  parameter estimation.

This redirects the investigation.  Five previous tests varied things downstream
of the likelihood core and found nothing; this one says why.

## What it does and does not license

It does NOT yet say "the shipped likelihood is miscalibrated", because the test
separates (estimator + catalog) from (PE + events), not estimator from catalog.
The mock's catalog is still the mock's -- nside 16, 1,854 occupied pixels,
192,757 galaxies, 60 events -- against production's nside 64, 30,470 pixels,
22.8M galaxies and 259 events.  A deficit that is a property of the small
configuration would not transfer.

It DOES mean the next test is a catalog-side one, not another data-side one,
and it means Tier C cannot be repaired by fixing the mock's PE or its event
draw.

## Open question this raises for the production result

If the estimator's intervals are systematically ~2.5x too narrow **on this
configuration**, the obvious question is whether the production 90% CI
([65.18, 79.38] for the latent arm) is too narrow as well.  Nothing measured
here answers that -- the scale difference is two orders of magnitude in
galaxies and a factor 4 in events -- and it must not be assumed in either
direction.  It is the reason the closure tiers matter, and it should be
resolved before the production interval is quoted as a credible interval.


---

# CORRECTION: the rule-7 call was wrong in mechanism

Pass 3 reported the mock's `z_depth` as "metadata-only", on the evidence that
the catalog rows run to z = 0.3847 against a stamped 0.30.  **That
characterisation is wrong and is withdrawn.**

`make_mock.py` DOES apply the cut to the rows:

    observed = ((complete["z"] <= Z_DEPTH)
                & (complete["app_mag"] <= survey_cfg.magnitude_limit)
                & (rng.uniform(size=total) < p_keep))

The cut is on the TRUE redshift; what the catalog STORES is `z_obs = z + zerr`.
So galaxies with true z just below the depth land above it once photo-z error
is applied.  Measured, and it is exactly that:

    above-depth galaxies      9,384
    median excess over 0.30   0.50 sigma      max excess  3.68 sigma
    stored dzgals (1 sigma)   0.0230          (the mock's photo-z width)
    n(z > 0.30 / .32 / .34 / .36 / .38) = 9,384 / 2,485 / 382 / 31 / 1

a Gaussian tail, not an untruncated parent.  Rule 7's actual signature -- "the
catalog's total row count/weight equals the uncut parent" -- does NOT hold
here; the count is the cut population, scattered.

**What IS different from production**, and it is the reverse of what I wrote:
production's catalog is hard-truncated at OBSERVED z = 0.3000 (zero galaxies
above), while the mock carries the physical photo-z tail.  The mock is the more
physical of the two on this point.

**What may still matter.**  The likelihood's `z_depth` convention is that
completeness is exactly zero beyond the depth -- every host there is missing.
A catalog that contains OBSERVED galaxies above the depth is inconsistent with
that convention regardless of how they got there, because the same galaxies are
then in the observed branch and inside the "everything is missing" region.  The
shipped code's `Nobs * exp(kernels.log_depth_mass)` rescaling exists for this.
So the truncation test remains worth running -- it measures sensitivity to the
boundary treatment -- but its motivation is "model/catalog convention
mismatch", not "the builder forgot to cut".

Recorded because the wrong version was already written into a commit message
and into pass 3 above.


---

# Pass 4: three more suspects eliminated, and where the width actually comes from

## The depth-boundary test: NULL

Tier C with the catalog truncated at `z_depth` (max z exactly 0.3000, matching
production), same seeds, nothing else changed:

| arm | untruncated | truncated |
|---|---|---|
| latent | 2.593 | **2.659** |
| latent_off | 2.259 | **2.399** |

No improvement.  The 4.87% of galaxies scattered above the depth by photo-z
error are not the cause.  (They remain a model/catalog convention mismatch
worth tidying -- see the correction above -- but they cost nothing here.)

## Rule 6's OTHER half: injection coverage is clean

The 18x injection test addressed "noisy".  Rule 6 also names "MIS-SLOPED", and
more draws from the same proposal would not fix that -- a distinction I had
conflated.  Tested directly: at every trial `H0` in the scan, no injection
falls outside the redshift grid.

| `H0` | grid `dL_hi` | injections beyond it | max injection z |
|---|---|---|---|
| 20 | 37,970 | **0** | 0.210 |
| 67.74 | 11,211 | **0** | 0.596 |
| 120 | 6,328 | **0** | 0.951 |
| 140 | 5,424 | **0** | 1.078 |

against a grid `zmax` of 1.500.  No coverage failure at any trial value.

(An earlier version of this check reported NaNs at `H0` = 140 and looked like a
coverage failure.  That was my own artifact: I ran the check at
`DARKSIRENS_ZMAX=1.0` while the mock pins **1.5**.  Corrected above.)

## The catalog KDE kernel is not too narrow

`sigma_eff = sqrt(sigma_cat^2 + sigma_kde^2)` with the mock's stored
`dzgals = 0.0230` and the run's `sigma_kde = 0.003` gives 0.0232 -- the stored
per-galaxy error dominates, so the kernel is the catalog's own, not the tiny
sampled floor.

## Where the width actually comes from: a 65% cancellation

Decomposing the `H0` curvature at the peak (`latent_off`, from the live
`log_mu` capture), `logL = sum_i(events) - N_obs log mu`:

    total      d2 logL/dH0^2 = -0.022314   -> sigma = 6.69 km/s
    events                   = -0.064303   (288% of the total)
    selection  -N log mu     = +0.041988   (-188% of the total)

**The posterior width is set by a near-cancellation**: the event sum supplies
-0.0643 of curvature and the selection term gives back +0.0420, leaving
-0.0223.  To match the measured event-only scatter of 15.57 km/s the net would
have to be -0.0041 -- i.e. the net is **5.4x too curved**, which corresponds to
a **~30-40% error in ONE of two large, mostly-cancelling terms** (the selection
term would need to be +0.060 instead of +0.042, or the event term -0.046
instead of -0.064).

That is the sharpest statement available about where to look, and it explains
why six downstream interventions all did nothing: none of them moves either
term by 30%.

## Eliminated so far, each with a number

1. the latent field (no-field control equally affected)
2. `b_gal` draw covariance (+0.04%, and the `s_b` response has the wrong SIGN)
3. PE over-sharpness (residual sd 1.059 vs the 2.6 required)
4. DAG rules 1-2 in the generator (verified by reading the source)
5. selection-integral MC **noise** (18x injections, Neff 1,953 -> 12,618)
6. selection-integral **coverage/mis-slope** (0 injections off-grid at any H0)
7. multimodality / outlier widths (posteriors near-Gaussian, widths stable)
8. the depth boundary (truncation test, null)
9. the catalog KDE width (dominated by the stored `dzgals`)

plus gate 8(a), which localises what remains to the estimator's own terms.


---

# An attempted shortcut that does NOT settle anything (recorded so it is not repeated)

The leading hypothesis after pass 4 is "the redshift prior is more informative
than the catalog warrants".  One tempting cheap version of that is: does the
selection MODEL's completeness match the mock's ACTUAL completeness?  If the
model thinks the survey is more complete than it is, the missing branch is too
small, the catalog spikes dominate, and the prior over-informs.

Attempted, and it does not survive its own assumptions.

    model  Cbar * <f_p>                = 0.6348
    actual, if n_complete is ALL-SKY   = 0.4566   -> model 1.39x too complete
    actual, if restricted to the 1,854
      occupied of 3,072 pixels         = 0.7565   -> model 0.84x, the OTHER WAY

The sign of the answer flips on whether `truth.json`'s `n_complete` counts the
full sky or the footprint, and the estimate additionally ignores the clustering
of the true field and the `(1+z)^delta` evolution in the number density.  A
quantity whose sign depends on an unresolved bookkeeping question is not
evidence, and the 1.39x version is NOT reported as support for the hypothesis
even though it is the right size.

`score_information.py` answers the same question -- does the event term carry
more information than its own curvature implies -- with none of these
assumptions, by measuring `J` and `H` from the same stored `logL(H0)` curves.
That is what the conclusion should rest on.


---

# The `N_obs` scaling at n=40: the deficit is a CONSTANT factor

Pass 2 flagged an unresolved question: at n=6 realisations per cell the median
scatter looked FLAT in `N_obs` (18.07 / 22.51 / 15.96), which would be a
common-mode term that never averages down -- but the 32% error on a standard
deviation at n=6 made that indistinguishable from correct `1/sqrt(N)` scaling,
and the difference between those two readings was the whole diagnosis.  Re-run
at **n = 40 per cell** (error ~11%):

| `N_obs` | n | scatter | mean `sigma` | overconfidence |
|---|---|---|---|---|
| 30 | 40 | 22.11 | 11.483 | 1.925 |
| 60 | 40 | 19.77 | 8.577 | 2.304 |
| 120 | 40 | 10.86 | 5.902 | 1.840 |

(`latent_off`; the `latent` arm gives 1.987 / 2.384 / 1.922.)

    d log(scatter)/d log N = -0.513   (latent_off),  -0.463 (latent)
    d log(sigma)  /d log N = -0.480   (latent_off),  -0.439 (latent)

**The scatter averages down as `1/sqrt(N)`.**  The flat reading was a
small-sample artifact, as pass 2 suspected; it is now excluded at 6.7x better
statistics.  There is no common-mode term that survives more events.

**And the quoted `sigma` scales the same way**, so the overconfidence ratio is
essentially CONSTANT in `N_obs` -- 1.93, 2.30, 1.84, no trend, scattered about
~2.0.

## Why that matters

A constant multiplicative deficit, independent of the number of events, is
exactly what a **fractional error in one of the two curvature terms** produces:
both the event sum and the selection term scale linearly with `N_obs`, so a
fixed fractional error in either leaves their ratio -- and therefore the ratio
of the naive width to the true scatter -- unchanged.  The `N_obs` lever arm
therefore CONFIRMS the cancellation framing and excludes mechanisms that would
scale differently: anything common-mode (`N^0`), and anything whose relative
size grows or shrinks with the event count.

Combined with gate 8(a) (the deficit is in the estimator, not the PE) and with
the selection term being a clean power law, the surviving description is: **a
fixed ~30% fractional error in the event/catalog term's `H0` curvature**, i.e.
a redshift prior more informative than the catalog warrants.


---

# The information identity, measured: `J/H = 6.03`

`H = -d^2 logL/dH0^2` (observed information) and `J = Var(dlogL/dH0)`
(expected information) must agree for a correctly specified model, and the MLE
variance is then `H^-1`.  When they disagree the correct variance is the
sandwich `H^-1 J H^-1`, and the naive interval is too narrow by `sqrt(J/H)`.
Both computed at `H0 = 68` from the SAME stored `logL(H0)` curves over 24
realisations (`latent_off`):

    J (expected, = Var of the score)  = 0.111133
    H (observed, = mean curvature)    = 0.018418
    J/H = 6.034     ->  intervals too narrow by  x2.456

**Tier C measured 2.1-2.9 directly from the realisation scatter.  These agree**
-- and they are independent routes to the same number, one from the spread of
medians, the other from the score and the curvature of individual likelihood
curves.

## What it localises, and one artifact ruled out

Because the injection set is shared across realisations, the selection term's
score is identical in every one and contributes **zero** variance.  `J` is
therefore a pure EVENT-term quantity, while `H` is the cancelled net.  So:

    J / |event curvature|  = 1.728      (1.0 if the event term were consistent)
    J / net curvature      = 6.034

The event term's score varies **1.73x more than its own curvature allows**, and
the 65% cancellation against the selection term amplifies that to 6.03 in the
net.  Both factors are real and they compound.

**Is `J` artificially small because the injections are held fixed?**  No.  If
they were redrawn per realisation the selection score would acquire a Monte
Carlo variance; from the measured variation of the `N log mu` MC error across
the scan (0.26 nats over ~100 km/s) that contributes `~7e-6`, against the
`0.093` needed to close the gap -- short by four orders of magnitude.  The
mismatch is not an artifact of the tier's design.

## The diagnosis, as far as the evidence carries it

Three independent measurements agree:

* **gate 8(a)** -- delta-PE at truth leaves the scatter unmoved: the deficit is
  in the estimator, not the parameter estimation or the event construction;
* **`N_obs` scaling at n=40** -- scatter and `sigma` both fall as `1/sqrt(N)`
  and the ratio is constant, which is what a fixed FRACTIONAL error in one
  curvature term produces and no common-mode term can;
* **the information identity** -- `J/H = 6.03`, matching the observed width
  deficit, with `J` a pure event-term quantity.

and the selection term is a clean power law with no structural defect.  The
surviving description is a **fixed fractional error in the event/catalog term**:
its curvature under-states the actual variability of its own score by 1.73x,
which the cancellation turns into a 2.5x interval deficit.  In physical terms,
the redshift prior is more informative than the catalog warrants.

## What is actionable now, without the root cause

`sqrt(J/H)` IS the correction.  A sandwich interval -- `H^-1 J H^-1` rather
than `H^-1` -- has the right coverage by construction whatever the underlying
misspecification, and both ingredients are computable from a single run's
likelihood curve plus a modest ensemble.  That is a defensible way to quote an
interval before the root cause is found, and it should be measured on the
production configuration rather than assumed to carry the mock's factor.


## `J/H` is not a single-node artifact

The identity was first reported at one node (`H0 = 68`).  Recomputed at every
node of the same stored curves:

| `H0` | `J` | `H` | `J/H` | `sqrt(J/H)` |
|---|---|---|---|---|
| 54 | 0.17621 | 0.02738 | 6.436 | 2.537 |
| 60 | 0.14300 | 0.02383 | 6.000 | 2.450 |
| 66 | 0.11813 | 0.01945 | 6.074 | 2.464 |
| 72 | 0.09792 | 0.01732 | 5.654 | 2.378 |
| 78 | 0.08180 | 0.01509 | 5.420 | 2.328 |
| 84 | 0.06953 | 0.01085 | 6.406 | 2.531 |

Over `H0` in [55, 85] -- the region that sets the width -- the median is
**6.074** (range 5.42-8.19), against the **6.034** reported at the single node.
`sqrt(J/H)` has median **2.464**, against 2.456.  The number is representative,
not a node chosen well.

Outside that window the ratio climbs (10.3 at `H0 = 42`, 19.2 at 102) because
`H` itself is collapsing there -- the likelihood is flat in the tails, so the
ratio is unstable and carries no information about the width.  That is expected
and is why the window is quoted.

At `n = 24` realisations the error on a variance is 29%, so `J/H = 6.03 +- 1.8`
-- far from the 1.0 a correctly specified model requires, and the conclusion
does not depend on the precision.


## Tier C at `n = 100`, on a disjoint seed block

The overconfidence ratio is the statistic the whole diagnosis turns on, and at
`n = 24` its error is 15%.  A third pass ran 100 realizations from a seed block
(7001 + 37k) that shares nothing with the earlier passes -- so this is an
independent replication, not an extension of the same sample.

| pass | `n` | seeds | `latent` | `latent_off` |
|---|---|---|---|---|
| second (`tier_c_v2`) | 50 | 90000 + 37k | 2.552 | 2.351 |
| big injections | 24 | as second | 2.662 | 2.413 |
| **third (`tier_c_n100`)** | **100** | **7001 + 37k** | **2.645** | **2.482** |

All three agree inside the ~7-15% precision of a variance ratio at these `n`
(the first pass predates the ratio being recorded; it reported the same coverage
failure, `frac_in_90` = 0.58 and 0.46).
The factor is settled: **`2.645` latent, `2.482` latent_off**, now to ~7%.

The `n = 100` pass in full (`tier_c_n100.json`; arms `latent` and `latent_off`,
`n0 = 5e-5`, 49 grid nodes over `H0` in [20, 140]):

| | `latent` | `latent_off` |
|---|---|---|
| spread of medians | 20.43 | 20.03 |
| mean quoted `sigma` | 7.73 | 8.07 |
| overconfidence | 2.645 | 2.482 |
| `frac_in_90` (nominal 0.90) | 0.45 | 0.49 |
| `frac_in_68` (nominal 0.68) | 0.29 | 0.31 |
| outside 99% | 35/100 | 33/100 |
| KS `p` | 5.1e-8 | 3.5e-5 |
| median bias | -0.223 `sigma` | -0.087 `sigma` |
| mean of medians | 64.07 | 66.57 |

Three details worth having on the record:

* **It is a width failure, not a bias.**  The median bias is 0.22 `sigma` and
  0.09 `sigma` while coverage is half its nominal value.  The mean of medians
  sits 3.7 (latent) and 1.2 (latent_off) km/s below `H0_true = 67.74`, which is
  the same -2.5 km/s paired field shift seen at every `n`.
* **No rail effects.**  Not one of the 200 medians lands within 1 km/s of either
  grid edge (range 24.0 to 107.4), so the spread is not a grid artifact.
* **It survives per-realization scoring.**  Scoring each realization by its OWN
  `sigma` rather than the ensemble mean gives `z` with sd 3.08 (latent) and 2.71
  (latent_off) -- larger, not smaller, than the ratio-of-means, because the
  quoted widths themselves scatter by 32% and do so in the wrong direction to
  help.

Nothing here changes the diagnosis; it removes the precision caveat from it.


## The PE offset across eight independent mocks

`pe_calibration.py` on the Tier-B mock had eliminated PE over-sharpness
(residual sd 1.059 against the 2.6 required) but left a residual MEAN of +0.486
with a KS `p` of 0.0055 unexplained.  Eight fresh mocks (seeds 8101-8108, same
injection set) were run through the same PP test; `pecal_multiseed.py`
aggregates them into `pecal_multiseed.json`.

| seed | `resid_mean` | `resid_sd` | KS `p` | `frac_in_90` | outside 99% |
|---|---|---|---|---|---|
| 8101 | +0.507 | 1.433 | 0.0004 | 0.783 | 4 |
| 8102 | +0.657 | 1.148 | 0.0001 | 0.800 | 2 |
| 8103 | +0.416 | 1.393 | 0.0789 | 0.800 | 2 |
| 8104 | +0.305 | 1.065 | 0.0034 | 0.900 | 2 |
| 8105 | +0.232 | 1.017 | 0.0737 | 0.900 | 0 |
| 8106 | +0.167 | 1.118 | 0.2163 | 0.917 | 2 |
| 8107 | +0.489 | 1.117 | 0.0007 | 0.833 | 1 |
| 8108 | +0.417 | 0.960 | 0.0028 | 0.883 | 0 |

**The width elimination now rests on eight realizations, not one.**  Mean
`resid_sd` = **1.156** with a between-seed spread of 0.170 and a range of
0.960-1.433, against the **2.6** Tier C requires.  The single realization's
1.059 was, if anything, on the low side of a distribution that never comes
within a factor 1.8 of what would be needed.

**The mean offset is one number, not noise.**  Positive in all eight, mean
**+0.399**, pooled standard error 0.053 -- 7.5 `sigma` from zero.  Its
between-seed scatter (0.159) is what the within-seed error alone predicts
(`1.156/sqrt(60) = 0.149`), so it is a constant feature of the construction
rather than something that varies from mock to mock.

**And the offset is the ENTIRE non-uniformity.**  Pooled over 480 events the raw
KS is decisive against uniformity (`p = 9.8e-17`, statistic 0.197).  Mapping
each quantile back through the Gaussian, removing the per-seed mean, and
re-testing gives statistic 0.043, **`p = 0.32`** -- uniform.  The shape is
right; only the centre is displaced.  (That re-test assumes near-Gaussian `dL`
posteriors, which the raw KS does not; it answers only the one question.)

**What this does NOT establish.**  The queue that generated these runs was set
up on the premise that an offset stable across realizations would be "a DAG
inconsistency, not noise".  That dichotomy is wrong and the conclusion does not
follow.  A positive offset is what detection PREDICTS -- selection keeps upward
SNR fluctuations, which under-estimate `dL`, so the truth sits above the PE mean
-- and it is exactly what the hierarchical likelihood's selection term exists to
absorb.  A working selection effect and a generator bug are BOTH stable across
seeds; only noise is not.  So stability establishes that the offset is real and
leaves its cause open.

Two things would separate the two, neither done here:

* the offset's dependence on SNR -- a selection offset is largest at threshold
  and vanishes for loud events, a construction bug need not be;
* whether the estimator's own bias stays small once the selection term is
  included, which is a statement about `H0`, not about `dL`.

On the second there is already indirect evidence: at `n = 100` the median bias
is 0.22 `sigma` and 0.09 `sigma`, i.e. the estimator is very nearly centred
while this +0.4 `sigma` offset sits in its input.  That is what absorption looks
like.  It is consistent with the offset being benign and it is not proof.

Either way this is a BIAS channel and Tier C's failure is a WIDTH failure, so
neither reading of the offset explains the 2.5x -- and gate 8(a) had already
closed that door from the other side, by collapsing the PE onto truth and
watching the scatter not move.


## The PE offset IS the selection effect, measured against its own prediction

The eight-mock aggregate left the ``+0.399 sigma`` residual mean open: stable
across realizations, therefore real, but stability does not separate "selection
working as designed" from a construction defect.  The named discriminator was
the offset's SNR dependence, and it is now run (`pe_offset_vs_snr.py`, 540
events over nine mocks -- the eight plus the Tier-B one).

Detection keeps upward SNR fluctuations, which under-estimate ``dL``, so the
truth sits above the PE mean -- but only near threshold; a loud event is
detected whatever its noise draw and must be centred.  From the mock's own
measurement model (``sigma_rho = 1``, ``rho_thr = 8``, ``dL = k/rho``), with
``n = rho_obs - rho_true`` and ``a = rho_thr - rho_true``,

    z ~ n + (n^2 - 1)/rho_true      =>     E[z | detected] ~ lambda(a) + a lambda(a)/rho_true

with ``lambda(a) = phi(a)/(1 - Phi(a))``.  That is parameter-free.  Against it:

| `rho_true` | events | measured `<z>` | predicted `<z>` |
|---|---|---|---|
| < 8 | 134 | +1.282 ± 0.092 | +1.591 |
| 8-9 | 100 | +0.437 ± 0.099 | +0.495 |
| 9-10 | 97 | +0.177 ± 0.102 | +0.129 |
| 10-11 | 60 | +0.038 ± 0.117 | +0.019 |
| 11-12 | 35 | +0.172 ± 0.208 | +0.001 |
| 12-15 | 60 | -0.183 ± 0.132 | +0.000 |
| > 15 | 54 | -0.175 ± 0.139 | +0.000 |

The offset falls from +1.28 below threshold to consistent with zero above
``rho_true ~ 10``, exactly where the prediction says it must, with a regression
slope of ``-0.0587 ± 0.0084`` (``p = 7e-12``).  Pooled, measured ``+0.408``
against a predicted ``+0.512``: the approximation overshoots by 20%, which is
the direction its own caveat gives -- it takes the ``dL`` posterior straight
from the ``rho`` posterior and ignores the mass and sky marginalisation that
widen it, and a wider ``sd_PE`` divides ``z`` down.

**So the offset is the Malmquist selection the hierarchical likelihood's
selection term exists to absorb, not a DAG defect.**  The events below
threshold in TRUE SNR are the same population whose presence already showed the
generator applies its threshold to the noisy statistic rather than the noiseless
one (checklist rules 1-2), and here they carry the largest offset, +1.28.

One residual worth recording rather than explaining away: above ``rho_true =
12`` the mean is ``-0.179 ± 0.095``, i.e. 1.9 sigma BELOW zero where the
prediction is 0.000.  Small, marginal, and in the opposite direction to
selection; the PE prior (flat in ``rho``, so ``p(dL) ~ dL^-2``) and the mass
marginalisation both enter at this order.  It is not the width deficit -- a
0.18-sigma centring error is a bias channel and Tier C fails on WIDTH by 2.5x --
but it is the one part of the PP test that the selection account does not
already explain.


## The OPG estimator of `J` is blind to the defect, and the mock proves it

Production has no ensemble, so its `J` was estimated from ONE dataset by the
outer product of per-event gradients (`desi_full259/production_opg.py`), which
assumes the per-event scores are independent: it forms `sum_i (u_i - ubar)^2`
and drops every `Cov(u_i, u_j)`.  That has always been quoted as a LOWER bound.
It was never measured how loose the bound is.  The mock can measure it, because
there the ensemble exists.

`opg_calibration.py` holds the catalog FIXED and varies only the event draw
(`make_mock.build(seed, event_seed=e)` re-seeds immediately before the events --
`variance_split.py`'s split), so both sides are the event-conditional variance
the quoted `sigma` claims to be:

| catalogs x event sets | `J` ensemble | mean `J_OPG` | `R` | `J_ens/H` | `J_OPG/H` |
|---|---|---|---|---|---|
| 3 x 16 (`opg_calibration.json`) | 0.0879 / 0.0945 / 0.1360 | 0.0183 / 0.0182 / 0.0136 | 4.80 / 5.20 / 9.96 | 5.80 / 3.92 / 8.23 | 1.21 / 0.75 / 0.83 |
| 2 x 12 (`opg_decomposition.json`) | 0.1559 / 0.0809 | 0.0213 / 0.0142 | 7.31 / 5.69 | 7.14 / 5.33 | 0.98 / 0.94 |

**`J_OPG` understates `J` by a factor 4.8-10 (median 5.2 and 6.5), and returns
`J/H ~ 1` on a configuration whose true `J/H` is 3.9-8.2.**  All 80 datasets
passed the ordering check exactly (0.0000%), so this is not a capture artifact.

**Where the missing variance is, measured rather than assumed.**  The second
run decomposes the score into the event term and the selection correction:

* the correction's derivative takes exactly TWO values across 24 datasets, one
  per catalog, with within-catalog variance `1.5e-27` -- it is bit-identical
  across event sets, as it must be, since `log mu` depends on `theta` and the
  SHARED injections and not on the events.  It contributes nothing.
* the event-term score carries all of it, and its ensemble variance is
  **7.31** and **5.69** times `N * E[Var_within(u_i)]` -- the i.i.d. prediction,
  which is what `J_OPG` computes.

So at fixed catalog the per-event scores are POSITIVELY CORRELATED, with an
implied mean pairwise correlation of `rho = 0.107` and `0.079` from
`R = 1 + (N-1) rho` at `N = 60`.  A ~9% average correlation is all it takes;
`J_OPG` assumes zero.

**What this does to the production numbers.**  Production's OPG gave
`J/H = 0.855` (fp) and `1.240` (nofp) after the curvature correction.  The mock
-- which has the defect, `J/H = 5-8` -- returns `0.75-1.21` through the SAME
estimator.  The production values are statistically indistinguishable from what
a defective configuration reports, so **the OPG route has no discriminating
power and production's `J/H ~ 1` is not evidence that production is clean.**
The withdrawal of the earlier `f_p` claim already followed from the curvature
check; this says the whole estimator, not just the curvature, cannot answer the
question.

**And the naive scaling is the wrong way to transfer it.**  What plausibly
carries over is `rho`, not `R`: at `N = 259` with `rho ~ 0.09` the inflation
would be `1 + 258 rho ~ 24`, not 5-6.  Whether `rho` itself survives the jump
from 1,854 pixels and 192,757 galaxies to 30,470 and 22.8M is exactly what is
not known -- a bigger, denser catalog could correlate events more (more shared
structure per event) or less (each event's hosts more locally determined).  No
number is quoted for production here.

**What it makes necessary.**  An ENSEMBLE of synthetic 259-event datasets on the
production catalog, which was already the stated requirement and is now the only
route: the shortcut has been measured and it does not work.


### The `J_OPG` gap is a dataset-level common mode, and it is not heavy tails

Two readings could give `R ~ 5-7` and they have different consequences, so both
were tested rather than argued.

**Not an outlier artifact.**  The dataset scores are light-tailed (kurtosis
-0.8 to -1.7), leave-one-out variance moves the ratio only from 4.80 to 4.39 and
from 7.30 to 6.27, and robust scales (IQR, MAD) bracket the plain estimate.  A
third pair of catalogs, run with per-event scores stored, gives `R = 6.70` and
`6.03` with `J_ens/H = 6.62` and `6.09` against `J_OPG/H = 0.99` and `1.01`.

**Not heavy per-event tails either, and the mechanism is a common mode.**  With
the per-event scores in hand (32 datasets x 60 events):

| | catalog 0 | catalog 1 |
|---|---|---|
| per-event `u`: kurtosis | +0.51 | -0.64 |
| max abs deviation / sd | 3.6 | 4.0 |
| dataset-mean `u`: sd | 0.00539 | 0.00490 |
| i.i.d. prediction for that sd | 0.00208 | 0.00200 |
| ratio (sd / variance) | 2.59 / 6.7 | 2.46 / 6.0 |
| pooled var / mean within-dataset var | 1.089 | 1.079 |

The per-event scores are near-Gaussian, so the sample variance inside one
dataset is a fair estimate of their spread.  What exceeds the i.i.d. prediction
is the DATASET MEAN: every event's score in a realization shifts together by
about `+-0.0054` against a within-dataset event-to-event spread of `0.0168`.
That 8-9% variance share is exactly the `rho` implied by `R = 1 + (N-1) rho`,
and it is what `J_OPG` removes when it centres on the dataset's own mean.

**It is not the events' redshifts.**  An event's own `z` explains only 2.1% of
the per-event score variance, and removing a smooth `g(z)` fitted on all pooled
events leaves the common mode untouched (variance ratio 6.56 and 5.78, against
6.70 and 6.03 before).  The correlation between the dataset mean and the
dataset's mean redshift is `-0.13` in one catalog and `+0.51` in the other --
inconsistent in sign, so not a redshift driver.

**What the common mode IS remains open.**  The catalog, the injections, the
pinned `theta` (`n0_calibration.json` and `selection_fit.json` are written at
the injected truth and are byte-identical across event seeds) and the selection
correction (variance `1.5e-27`) are all held fixed, and the events are drawn
i.i.d. from the same complete catalog.  Something shared by all 60 events of a
realization still moves them together.  That is a question about the estimator's
structure, not about this measurement, and it does not change what the
measurement licenses: **no single-dataset estimator that centres on its own mean
can see this term**, so the production `J` needs an ensemble.


## Catalog-side interventions: the deficit tracks the catalog branch's WEIGHT

Gate 8(a) put the deficit in the estimator; the information identity put it in
the event/catalog term.  Three catalog-side interventions, each Tier C's own
loop with one term perturbed, at `n = 24` from seed 7001 -- the first 24
realizations of the `n = 100` pass, so the baseline is matched seed for seed
(`tier_c_catalog_side.py`, `catalog_side_summary.py`):

| intervention | overconfidence | mean `sigma` | spread of medians | median bias |
|---|---|---|---|---|
| baseline (`latent_off`) | 2.397 | 8.125 | 19.48 | +0.436 |
| `kde_wide` (`sigma_kde` 0.003 -> 0.01) | 2.412 | 8.147 | 19.65 | +0.447 |
| `fp_scaled` (`f_p` x 0.8) | 1.893 | 7.794 | 14.75 | -0.002 |
| `per_pixel` (count-derived `C`) | 1.529 | 7.249 | 11.08 | +1.710 |

**The catalog kernel width is not the knob.**  Broadening the per-galaxy
redshift kernels by 3.3x moves the overconfidence by +0.015 and the quoted
`sigma` by 0.3%.  The redshift prior's informativeness does not come from how
sharply each galaxy is placed.

**Reducing the catalog branch's weight does move it, through the SCATTER.**
Scaling `f_p` down by 20% takes the overconfidence from 2.40 to 1.89 -- and it
does so by shrinking the realization-to-realization spread (19.5 to 14.8), not
by widening the interval (8.13 to 7.79).  **This is a lever, not a
mis-specification correction, and the mock says so unambiguously**: its
completeness model is EXACTLY right by construction (`world16.M0_HAT`,
`SIGMA_M`, `M_LIM` are the values `make_mock` draws magnitudes with, and
`c_sel_gaussian` reproduces the mock's own `Phi` to 1e-4), so there is no
completeness error to correct.  What the scan shows is that the excess scatter
is generated in proportion to how much the catalog branch carries.

**Swapping the completeness ESTIMATOR trades width for centring.**  The
per-pixel matched-kernel ratio cuts the overconfidence to 1.53 and the spread by
43%, but the median bias goes from +0.44 to **+1.71 sigma**.  It is not a fix;
it absorbs the observed angular clustering into `C`, which is exactly why the
production line does not use it.

So within the catalog side: not the kernels, not a wrong completeness curve, and
no intervention that keeps the estimator centred closes the gap.  The deficit
scales with the catalog branch's weight while the branch's own inputs are
correct -- which is the same statement the information identity makes, arrived
at from the other end.


### The common mode survives delta-PE, and that is an anomaly, not a mechanism

The obvious mechanism for a per-dataset common mode is the PE realization, so
gate 8(a)'s intervention was applied to the calibration itself
(`opg_calibration.py --delta-pe`, verified on the built artifact: max
within-event `dL` spread `0.000e+00` in the delta-PE trees, `8.7e+02` in the
ordinary one).  It does not go away:

| | ordinary PE | delta-PE at truth |
|---|---|---|
| `R` (catalog A / B) | 6.70 / 6.03 | 5.35 / 6.48 |
| `J_ens/H` | 6.62 / 6.09 | 4.64 / 6.14 |
| `J_OPG/H` | 0.99 / 1.01 | 0.87 / 0.95 |
| dataset-mean variance / i.i.d. | 6.7 / 6.0 | 5.4 / 6.5 |

**And a bootstrap makes the statement assumption-free.**  Pool a catalog's
`16 x 60` per-event scores, resample 16 synthetic datasets of 60, and take the
variance of their means: that is the i.i.d. null with no distributional
assumption at all.  Observed / null = **6.37, 5.81 (ordinary) and 5.31, 6.41
(delta-PE), each with `p < 5e-4`** over 2000 resamples.  Shuffling events across
datasets destroys the effect, so the per-event scores carry dataset identity.

**Here is why that is an anomaly.**  The events of one realization are drawn
independently from a FIXED catalog, with byte-identical injections, depth map
and pinned `theta` -- verified by hash, and the only file that differs between
event seeds beyond the events is `selection_fit.json`'s recorded `survey_path`
string.  If the per-event scores were i.i.d. -- which they must be if `u_i` is a
function of event `i`'s own parameters -- then `Var(sum_i u_i) = N Var(u_i)`
exactly.  It is 6x that.  The argument does not even need delta-PE: an
independent noise draw per event is still per-event.

**The capture is not the explanation.**  `capture_check.py` re-runs one dataset
at `pe_event_block` 4, 8 and 16, twice each: `logL` is identical to the last bit,
the captured multiset is identical (`max |sort(x) - sort(ref)| = 0`), and within
a fixed block size the ORDER is reproducible too.  Changing the block size
permutes the order, as vmap chunking should -- and the calibration holds the
block fixed across its three `H0` nodes, so its pairing is consistent.  (A
permuted capture would in any case INFLATE `J_OPG` by adding pairing noise, and
the discrepancy runs the other way.)

**It is not the events' redshifts.**  Their distributions are homogeneous across
datasets (Kruskal-Wallis `p = 0.82` and `0.037`) while the SCORE distributions
are not (`p < 1e-10` both), and a smooth `g(z)` fitted on all pooled events
reproduces only **8%** of the dataset-mean scatter.  Same redshifts, shifted
scores.

So one of two things is true and this campaign does not settle which:

* the mock's event draw does not produce independent events, despite drawing
  hosts i.i.d. from the catalog; or
* the per-event score carries a dataset-level dependence that is not in the
  event's own parameters -- a coupling in the estimator we have not found.

The second would be the more interesting, because it would be a property of the
LIKELIHOOD rather than of the mock, and Tier C's width deficit is exactly a
statement about the likelihood.  Worth noting alongside: `J_ens/H` = 4.6-8.2
here, and `sqrt` of that is 2.2-2.9, which is Tier C's measured overconfidence
range.  This may be the same object seen from a third direction.

**What does not depend on resolving it:** the operational conclusion.  A
per-dataset common mode is invisible to any single-dataset estimator that
centres on its own mean, whatever creates it, so production's `J` needs an
ensemble either way.


## Regrouping the SAME events: the common mode is the `f_p` channel

The anomaly's two explanations -- a non-i.i.d. event draw, or a coupling in the
estimator -- are separated by REGROUPING (`event_reshuffle.py`).  Pool the events
of 16 datasets that share one catalog and build new datasets of 60 drawn from
that pool, each spanning 15-16 of the source realizations.  The event population
is identical by construction; only the grouping changes.

| configuration | `R = J_ensemble / J_OPG` | `J_ens/H` | `J_OPG/H` |
|---|---|---|---|
| original grouping, `f_p` on | 6.70 / 6.03 | 6.62 / 6.09 | 0.99 / 1.01 |
| **regrouped**, `f_p` on (n=16) | **5.57** | 5.50 | 0.99 |
| **regrouped**, conditional sky weighting (n=8) | **6.70** | -- | -- |
| **regrouped**, `f_p` OFF (n=8) | **0.885** | -- | -- |

**The common mode is not the event draw.**  Regrouping the same events into new
datasets leaves `R = 5.57` -- if the generator's grouping had carried the
correlation, destroying it would have returned `R = 1`.

**It is not the field normalizer either.**  Under `catalog_sky_weighting =
conditional`, which drops the survey-global normalizer entirely, `R = 6.70`.

**It IS the per-pixel selection fraction.**  Drop `f_p` and `R = 0.885` -- the
identity holds, from `5.57` with it.  At `n = 8` a variance ratio carries ~50%
error, so `0.885` is `~1` and not a measurement of anything smaller, but the
separation from `5.57` is a factor 6 and far outside that.

That is the same channel the rest of this campaign keeps landing on: `f_p` moves
the production median by `4.31 sigma`, carries 97.2% of the ensemble's runtime,
and accounts for 91% of Tier B's failure -- and the catalog-side scan above found
the overconfidence falls monotonically as the `f_p` channel's weight is scaled
down (2.40 -> 1.89 -> 1.58 -> 1.40 at `f_p x 0.8/0.6/0.4`).  Removing it
entirely is the endpoint of that ladder, and it is where the information
identity comes back.

**One thing this does NOT say.**  The `f_p`-free arm is the S-3-EXPOSED
configuration -- it models unobserved sky as `Cbar`-complete, which is the defect
PR #406 fixes.  So "`R = 1` without `f_p`" is not "the correct model has `R = 1`";
it is "the channel that carries the common mode is the one `f_p` introduces".
Which of the two configurations is right is a separate question, already answered
in `f_p`'s favour on the modelling merits.

### A retracted check, and why the retraction matters

The regrouping was to be read alongside a per-event invariance check: does an
event's captured log-evidence change when it is moved into different company?
It appeared to fail decisively (max `|delta|` = 5.98 nats, none exact).

**That check is invalid and its result is withdrawn.**  A first control -- rebuild
source 0 from its own events, in its own order -- passed exactly (60/60,
`max |delta| = 0.000e+00`), which is why the failure looked real.  But that
control cannot see a reordering: the same 60 events reorder the same way.  A
second control rebuilds source 0 with its events REVERSED, and the capture then
matches NEITHER file order NOR the source's order, while `logL` is identical to
the last bit (`delta = 0.0`, as it must be for the same events).

So the capture's order is data-dependent, the `(source, event) -> capture slot`
mapping the invariance check assumed is wrong, and its 5.98 nats are
misattribution rather than coupling.  Nothing else here rests on it: `R` is
computed from the total `logL` and from the within-dataset spread about its own
mean, neither of which needs an event's identity -- only that the capture order
is stable across the three `H0` nodes of one dataset, which `capture_check.py`
verified.

The lesson is the control's, not the result's: an identity round trip validates
the round trip and nothing about the index mapping, and the two look the same
until something is permuted on purpose.


### CORRECTION: the `f_p` bisect was not like-for-like, and one number is withdrawn

The bisect above measured every arm's `R` at `H0 = 68`.  That is within half a
`sigma` of the `f_p`-on arm's peak (its median-of-medians is 70.9) and it is
**59 km/s from the `f_p`-off arm's**, whose median-of-medians is **126.9**.  The
information identity is a statement AT the peak; in the tail `H` collapses and
the ratio is unstable, which this file has already recorded once and which the
production arm comparison had to be corrected for (the arms were compared at
`H0 = 72` when `nofp` peaked at 90).  The same mistake, one experiment over.

**So `R = 0.885` without `f_p` is withdrawn as evidence about the `f_p`
channel.**  A like-for-like re-run at that arm's own peak is what the claim needs
and it is running; until it lands, the localisation is not established.

**What the same run DID establish, and it is not small.**  Tier C in the
`f_p`-off arm, 24 realizations on the seeds `tier_c_n100` uses:

| | `latent_off` (`f_p` on) | `latent_off_nofp` (`f_p` off) |
|---|---|---|
| overconfidence | 2.397 | **1.809** |
| spread of medians | 19.47 | 15.89 |
| mean quoted `sigma` | 8.13 | 8.78 |
| median bias | **+0.44 sigma** | **+5.11 sigma** |
| `frac_in_90` | 0.50 | **0.00** |
| median-of-medians | 70.9 | 126.9 |

Removing `f_p` does lower the overconfidence -- 2.40 to 1.81, continuing the
monotone trend the `f_p x 0.8/0.6/0.4` ladder traced -- and it destroys the
answer: every one of the 24 realizations rails, `H0` lands at 92-138 against a
truth of 67.74, and the truth is outside the 90% interval in 24 of 24.

That is the S-3 exposure measured on the mock, and it is the same trade the
`per_pixel` intervention showed: **every catalog-side lever that narrows the
dispersion buys it with bias.** `f_p` scaled down, `f_p` removed, and the
completeness estimator swapped all move along one line -- less weight on the
catalog branch, less scatter, more bias -- and none of them lands anywhere with
both.

And a tension worth recording rather than resolving prematurely: at `H0 = 68` the
`f_p`-off arm reported `J/H ~ 1` while its Tier C overconfidence is 1.81.  If
that `J/H` survives at the arm's true peak, then the identity and the coverage
disagree there, and `sqrt(J/H)` would not be the correction it is on the `f_p`-on
arm.  The off-peak evaluation is the more likely explanation and the re-run will
say.


### The like-for-like re-run: it IS the `f_p` channel, and it reproduces Tier C

Each arm re-measured at its OWN peak, 16 regrouped datasets, identity round-trip
control passing 60/60 exactly in each:

| arm | node | `H` | `J_ens/H` | `J_OPG/H` | `R` |
|---|---|---|---|---|---|
| `f_p` ON (peak 70.9) | 68 | 0.01476 | **5.50** | 0.99 | **5.57** |
| `f_p` OFF (peak 126.9) | 120 | 0.01963 | **0.75** | 0.76 | **0.98** |
| `f_p` OFF | 127 | 0.00083 | 18.77 | 16.33 | 1.15 |

The `H0 = 127` row is quoted and then set aside: the `f_p`-off arm rails, its
curvature there is `8e-4` -- twenty times smaller than at 120 -- and a ratio
against a curvature that is collapsing carries no information about a width.
It agrees on `R` and that is all it is used for.  The `H0 = 120` row is the
statement.

**With `f_p` the information identity is violated 5.5x and the single-dataset
estimator misses all of it** (`J_OPG/H = 0.99` against a truth of 5.50).
**Without `f_p` the identity holds and the estimator is tight** (0.75 and 0.76,
consistent with 1 at the ~35% error of a variance ratio at `n = 16`).  So the
withdrawn `0.885` was right for the wrong reason -- measured 59 km/s off-peak --
and the corrected measurement says the same thing where the identity is
meaningful.

**And it closes the loop on Tier C.**  `sqrt(J_ens/H)` is the interval inflation
the identity predicts:

| arm | `sqrt(J_ens/H)` predicted | Tier C overconfidence measured |
|---|---|---|
| `f_p` ON | **2.35** | **2.40** |
| `f_p` OFF | 0.87 | 1.81 |

On the `f_p`-on arm -- the deliverable's configuration -- the prediction and the
measurement agree to 2%, by two routes that share no arithmetic: one is the
variance of a score across an ensemble against a curvature, the other is the
scatter of 24 posterior medians against their own quoted widths.  **That is the
root-cause statement this campaign has been missing: Tier C's ~2.5x
overconfidence is the `f_p` channel breaking the information identity.**

The `f_p`-off row does NOT agree (0.87 predicted, 1.81 measured), and that is
expected rather than awkward: that arm is biased by `+5.11 sigma` with 24 of 24
realizations railing to 92-138, so its "scatter of medians" measures how far each
realization rails and not a sampling distribution around a peak.  A
Bernstein-von Mises quantity has nothing to predict there.

### Where that leaves the deliverable

Three things follow and they should not be run together:

1. **The mechanism is localised but not explained.**  `f_p` enters `C_p = f_p C`
   on both sides of the missing budget; why that should make the per-event
   scores share a per-dataset mode is not established here.  The natural
   suspicion is that `f_p` couples events through the survey-global budget --
   but conditional sky weighting, which removes the global normalizer, left
   `R = 6.70`, so that suspicion is already once refuted.
2. **`sqrt(J/H)` is a correction that now has a measured provenance.**  A
   sandwich interval on the `f_p`-on arm is right by construction, and its factor
   agrees with the directly measured coverage. What is still missing for
   production is `J` itself -- and this result says why the OPG shortcut cannot
   supply it: `J_OPG/H = 0.99` is exactly what the defective arm reports.
3. **The production number is unchanged and the caveat is unchanged.**  Nothing
   here moves a median. It says the interval on any `f_p`-bearing arm --
   `fp`, `latent`, and now `q_fp` -- is the one that needs the correction.


## The coupling, localized to `f_p` -- and one probe that disagrees

`R` needs no capture: it is `Var_D(total score) / J_OPG`, and the numerator comes
from total log-likelihoods.  But the INTERPRETATION -- that the excess is a
per-dataset common mode in the per-event scores -- rests on the capture, so it
was re-tested a capture-free way (`loo_coupling.py`).  For a dataset `D` and an
event `e`,

    u_e(D) = d/dH0 [ logL(D) - logL(D \ e) ]

is a difference of TOTALS, immune to whatever order the event reduction runs in.
Evaluate the SAME 12 events inside two datasets that share only those 12:

| arm | delta of centred `u_e` | within-dataset spread | ratio | corr(X, Y) |
|---|---|---|---|---|
| `f_p` ON | 0.0279 | 0.1558 | **0.179** | +0.997 |
| `f_p` OFF | **0.000000** | 0.0164 | **0.000** | **+1.000** |

**Without `f_p` an event's score is EXACTLY independent of its company** -- zero
to machine precision, correlation exactly 1.  With `f_p` it is not.  That is the
localisation stated at the level of a mechanism rather than a ratio: `f_p` is
what makes the per-event terms of this likelihood non-separable.

### The probe that disagrees, and what it does and does not overturn

Pushed further -- 8 datasets sharing 12 events, so the across-dataset variance of
those events' mean LOO score estimates `Var(c)` directly -- the answer comes back
NULL on the `f_p` arm:

    dataset means of u: sd 0.0237   within-dataset sd 0.1025
    Var(c) debiased = -3.2e-4  (i.e. consistent with ZERO)
    => R predicted at N = 60: 1.00,  against the measured 5.57

So the LOO route finds no common mode where the capture route finds a large one.
Both cannot be describing the same quantity, and they are not: the per-event
spread is **0.1025** by LOO against **0.0156** by capture, a factor 6.6.  LOO
carries the selection correction's non-linearity in `N` -- removing an event
changes the `60 -> 59` correction, which is precisely the term that broke the
subset estimator at 9.2% and is recorded in this file as structural.  So the LOO
`u` is a different decomposition of the same total, and its `Var(c)` does not
bound the capture's covariance.

**What survives regardless.**  `Sum_i u_i` equals the full score to 0.00% (the
ordering check), and `Sum_i (u_i - ubar)^2` is invariant to any permutation of
the capture -- so `J_OPG = 0.0146` is robust to the ordering defect that
invalidated the invariance check.  With `Var_D(total score) = 0.081` measured
from totals alone, `Var(S) > N Var(u)` is then arithmetic, and positive
covariance among the within-dataset scores is inescapable.

**What is now less settled than the previous section implied.**  The SIZE of the
common mode, and therefore the claim that ~9% pairwise correlation is the whole
story, rests on the capture-based decomposition alone; the independent probe of
that decomposition returns a null it cannot explain.  The identity violation
(`J_ens/H` = 5.50 with `f_p`, 0.75 without, `sqrt` matching Tier C's 2.40) is
untouched by this -- it needs no per-event decomposition at all -- but the
mechanism should be read as "`f_p` makes the per-event terms non-separable",
which the exact-zero LOO result establishes, and NOT as a measured correlation
coefficient.


### Four more mechanisms eliminated, and the coupling is still unexplained

The `f_p` arm's non-separability is not in doubt -- the control is that the SAME
test on the `f_p`-free arm returns **exactly zero**, and now over 8 datasets, not
two: the dataset-mean LOO score has sd **0.000000** there against 0.0237 with
`f_p`.  What carries it is another matter, and four candidates are now closed:

| candidate | test | result |
|---|---|---|
| the selection correction | its derivative across datasets | variance `1.5e-27` -- bit-identical |
| the survey-global normalizer | conditional sky weighting, which removes it | `R = 6.70`, unchanged |
| the reliability guard | `max_likelihood_variance` 1e6 -> 1e12 | `logL` **bit-identical** (`delta = 0.000e+00`), so the guard is already inert |
| the compact catalog's `f_p` gather | `f_p_rows[row]` vs the map at that row's pixel, across event sets with 177 / 177 / 174 rows | `max abs diff = 0.000e+00` on every set |

The guard test is the one worth dwelling on, because the guard was the only term
in this likelihood whose value is a function of the whole event SET, and it was
therefore the leading hypothesis.  Raising the budget a millionfold changes the
log-likelihood in no bit, so on this configuration the guard contributes nothing
at all -- neither a level nor a coupling.

The `f_p` gather test was checking for a defect rather than a subtlety: had the
per-row gather been misaligned when the compaction changed, every `f_p` run would
have been indexing the wrong completeness for some rows.  It is exact, including
on a set with a different row count, so there is no such bug.

**So the state is: `f_p` demonstrably makes the per-event terms non-separable,
and no shared term, no indexing path and no guard accounts for it.**  That is
less satisfying than a mechanism and more useful than a guess.  What it does not
touch, again, is the identity violation itself -- `J_ens/H` = 5.50 with `f_p` and
0.75 without, both at their own peaks, with `sqrt` = 2.35 against Tier C's
measured 2.40 -- which is computed from total log-likelihoods and needs none of
this decomposition.

Next probe, if this is picked up: the per-event term with `f_p` reads
`C_p = f_p(pix) C(z)`, and the only remaining route by which one event's row can
influence another's is through a reduction whose ORDER changes with the compact
row set -- i.e. float non-associativity in a sum that `f_p` makes
non-uniform. That predicts a coupling that shrinks with float64 accumulation and
vanishes at `f_p == 1`, both of which are cheap to test and neither of which has
been run.


## THE COUPLING IS A DEFECT IN THE `f_p` PATH, not a property of `f_p`

The additivity test settles it (`additivity.py`).  For DISJOINT event sets `A`
and `B`, the sum of per-event log-evidences must satisfy
`E(A u B) = E(A) + E(B)` if each term depends on its own event alone.  `E` is the
captured per-event values SUMMED, so it needs no ordering assumption -- a sum does
not care -- and it excludes the selection correction, the one term already known
not to decompose.

| arm | `E(A u B) - E(A) - E(B)` | relative to `|E|` |
|---|---|---|
| no `f_p` | **+0.000000e+00** (score residual 7.1e-15) | 0 |
| real `f_p` map | **-18.32 nat** | -4.1% |
| **`f_p == 1` everywhere** | **-55.27 nat** | **-10.5%** |

**The `f_p == 1` row is the finding.**  Multiplying the completeness by exactly
1.0 is the IEEE identity, so that arm's arithmetic is mathematically the no-`f_p`
arm's -- which is additive to the last bit.  It returns 55 nat of non-additivity.
That rules out every value-dependent explanation, including the float
non-associativity this file predicted as the next probe: 55 nat is not rounding,
and the residual gets LARGER, not smaller, as `f_p` becomes uniform.

**So the `f_p` code path breaks the additivity of the event sum.**  Not the
modelling of `C_p = f_p C`; the path.

### Where it is, and where it is not

* **Not the survey-global normalizer.**  Its two ingredients are bit-identical
  across event sets: `N_obs_total = 222107.85349099262` and the total missing
  curve `sum V = 2664415.0787219717` for `A`, `B` and `A u B` alike.
* **The per-event values themselves move.**  A multiset comparison
  (`ll_multiset.py`) -- order-free, so immune to the capture defect -- finds
  `sorted(ll(A) u ll(B))` differs from `sorted(ll(A u B))` in **all 24 of 24**
  values, by **+0.236 to +1.381 nat, mean +0.764**, every one the same sign.
* A uniform per-event offset of that size accounts for the whole residual
  (`24 x -0.764 = -18.3`), which is the signature of a NORMALIZER -- and yet the
  global normalizer is identical.  The compact row set is what differs
  (37 / 36 / 73 rows, disjoint: 37 + 36 = 73).

The remaining suspect is therefore a per-event normalizer that is assembled from
the COMPACT rows where it should be full-sky, on the `f_p` branch only.  Finding
it is a code-reading task, not a compute one, and it is the next thing to do.

### What this does to everything upstream

**The 5.5x information-identity violation on the `f_p` arm may BE this defect.**
The chain is: `J` is the variance of the total score across datasets; if each
event's term carries an offset that depends on which other events are present,
then the total score acquires exactly the kind of dataset-level common mode that
`J_ens` measures and `J_OPG` cannot see.  That is the same object this file spent
four passes describing as a mystery, and it now has a candidate cause that is a
bug rather than a property of the model.

**Nothing is retracted yet, and the numbers stand as measurements.**  Tier C's
2.4-2.6 overconfidence, `J_ens/H` = 5.50 with `f_p` against 0.75 without, and
`sqrt(J/H)` = 2.35 matching the measured 2.40 are all correctly measured
properties OF THE SHIPPED CODE.  What is now in question is their
INTERPRETATION: whether they describe the estimator's statistical behaviour or a
normalizer assembled over the wrong row set.

**Consequences to act on:**

1. **Do not build the production `J` ensemble yet.**  It would measure this
   defect at production scale and at production cost.  The ensemble was gated on
   finding the mechanism precisely so this could not happen.
2. **The production medians are NOT implicated by this test.**  A single H0 scan
   uses one fixed event set, so an event-set-dependent offset is a constant
   across the scan and cancels from the shape.  What it can move is anything
   compared ACROSS different event sets -- which is Tiers B and C, the `J`
   measurements, and the `N_obs` scaling, and not `fp`/`latent`/`q_fp`.
3. **S-3 stands.**  The mask's -2.49 sigma on the production Q line is a
   single-event-set comparison of two models on the same 259 events.


# THE BUG, FOUND AND FIXED -- and most of Tier C was it

`completion._row_C` indexes `f_p_rows` by CATALOG ROW.  On the flat
single-catalog UNION path those rows carry `union_unique_pixels`, but
`catalog_views` gathered `f_p_rows` with the per-view `unique_pixels_pe`.  When
the two differ the array is SHORTER than the row count -- **and JAX clamps an
out-of-bounds gather instead of raising**, so every row past the short array
silently took the LAST entry's completeness.  The per-view length depends on
which events are in the run, so an event's own `C_p` depended on its company.

On the mock the mismatch is brutal: 37 or 73 `f_p` entries against a 3,072-row
catalog, i.e. ~99% of rows clamped.

## What the fix moves

| quantity | before | after |
|---|---|---|
| per-row `N_miss` between event sets | 2.3% shift | fixed |
| event-sum additivity, real map | -18.32 nat | **-2.8e-14** |
| event-sum additivity, `f_p == 1` | -55.27 nat | **-2.8e-14** |
| **information identity `R = J_ens/J_OPG`** | **5.567** | **1.261** |
| **Tier C overconfidence** (latent / latent_off, n=24) | 2.65 / 2.40 | **1.570 / 1.579** |
| Tier C spread of medians | 19.5 | **10.2** |
| Tier C coverage at 90% | 0.45-0.50 | **0.62** |

`R = 1.261` is consistent with 1 at the ~35% error of a variance ratio at
`n = 16`.  **So the identity violation was the bug**, and the previously reported
agreement between `sqrt(J_ens/H) = 2.35` and Tier C's measured 2.40 was two
manifestations of ONE defect, not independent confirmation by two routes.  That
agreement was the strongest evidence in this file, and it is withdrawn AS
EVIDENCE even though both numbers were correctly measured.

## What did NOT go away

Tier C still fails: overconfidence **1.57**, coverage **0.62** against 0.90.  The
bug accounts for roughly `(2.4^2 - 1.57^2)/2.4^2 = 57%` of the variance excess; a
factor ~1.6 remains.  It is now essentially IDENTICAL in the two arms (1.570 vs
1.579, spreads 10.18 vs 10.26) where before they differed, and the residual bias
grew against the narrower intervals (+0.56 and +0.79 sigma).  So there is a
second, smaller effect: not the field, not the `f_p` gather, not yet identified.

## What this does to the production numbers

**Almost nothing, for a quantitative reason rather than a reassuring one.**  The
production PE view holds 49,143 rows against a 49,152-row catalog, so only ~9
rows were clamped -- 0.02%, against the mock's ~99%.  Re-run:

    fp arm, 259 events:   71.70 [65.0, 79.1]  ->  71.54 [64.6, 79.1]

a 0.16 km/s shift, 0.04 sigma.  **The production medians stand.**  An earlier
claim in this campaign that they were safe was right in conclusion and WRONG in
reasoning -- it argued that an event-set-dependent offset is constant across an
`H0` scan and cancels; the real reason is that production's compact view is
nearly complete, so the defect barely fires there.  Anything measured across
DIFFERENT event sets -- Tiers B and C, every `J` number, the `N_obs` scaling --
was fully exposed.

## The methodological lesson, which is the transferable part

Four passes of this file hunted a statistical explanation for a coding defect,
and every diagnostic that "cleared" a component did so honestly:

* the survey-global normalizer IS built from full-sky inputs and stayed
  bit-identical (`log Z` = 18.123588182747532 in all three views);
* `f_p_rows` DID match the map -- when checked against the per-view pixels, which
  is the wrong reference;
* the selection view was genuinely unaffected, because its pixel set already
  spanned the sky.

What broke it open was testing a property with no free parameters -- that the
event sum must be ADDITIVE across disjoint event sets -- and then the control that
`f_p == 1`, where the arithmetic is the no-`f_p` arm's by IEEE identity, still
broke additivity by 55 nat.  A silent out-of-bounds clamp is invisible to every
check that compares a quantity against itself; it is visible to one that compares
a whole against the sum of its parts.

The gather site now carries a fail-loud shape guard, and the regression pin
asserts the invariant (one `f_p` entry per catalog row, each equal to the map at
that row's own pixel) rather than the symptom.


## Post-fix re-measurement of everything the bug touched

Every tier that compares across event sets was exposed, so all were re-run.  The
picture is coherent in one place and contradictory in another, and both halves
matter.

**At FIXED catalog the estimator is now calibrated.**

| measurement | pre-fix | post-fix |
|---|---|---|
| variance split, events at fixed catalog | 2.110 | **1.018** |
| `N_obs` scaling at `N` = 30 / 60 / 120 (one catalog) | 1.93 / 2.30 / 1.84 | **0.46 / 0.50 / 0.99** |
| information identity `J_ens/H` (fixed catalog) | 5.50 | **1.266** |

The `N_obs` values are 6 event sets each, so they carry ~30% error and the
sub-unity reads as "1, or conservative" rather than a measured under-confidence.
What they are not any more is 2.

**The variance split's TOTAL is nearly calibrated too**, and its bias is gone:
overconfidence 1.141 (from 2.360), grand mean **67.48** against `H0_true` = 67.74
(from 64.10), with the catalog carrying 20% of the variance.

**But Tier C, on a different seed family, still reads 1.579.**  Same estimator,
same fix, same nominal quantity:

    variance_split (seeds 90000+, 6 x 6 = 36 cells)   overconfidence 1.141
    tier_c         (seeds 7001+,  24 realizations)    overconfidence 1.579

Those disagree at roughly 2.5 sigma and one of them is wrong about the total.
The discriminator -- `variance_split` re-run on TIER C's own seeds -- is running;
until it lands, **the honest statement is that post-fix total coverage is
somewhere between 1.14 and 1.58 and this campaign cannot yet say where.**

**Gate 8(a) survives the fix.**  Delta-PE at truth, post-fix, same seeds:
overconfidence **1.533** against the ordinary arm's 1.579 (both narrower: sigma
5.74 vs 6.50, scatter 8.80 vs 10.26).  So whatever the residual is, it is still
not the parameter estimation -- the one pre-fix conclusion in this file that never
depended on the corrupted `f_p` path.

**Tier B post-fix:** the field-only shift is **0.088 sigma** (was 0.136) and the
`f_p` channel's shift is **2.239 sigma** (was 1.674).  The `f_p` effect GREW when
the bug was fixed, which is the expected direction -- the defect was corrupting
the very channel it lived in -- and the ladder's headline is unchanged in kind:
the field does nothing measurable, `f_p` does everything.


## The residual, resolved into what it is and what it is not

The 1.14-vs-1.58 disagreement was the SEED FAMILY, and running the same
decomposition on Tier C's own seeds settles it:

| | seeds 90000+ | seeds 7001+ |
|---|---|---|
| variance_split, TOTAL | 1.141 | **1.323** |
| variance_split, events at FIXED catalog | 1.018 | **0.848** |
| tier_c, TOTAL (n = 47) | -- | **1.508** |
| grand mean vs `H0_true` = 67.74 | 67.48 | 70.73 / **72.58 +- 1.41** |

Two statements survive across both families and both estimators, and they are the
useful ones:

**1. Events at fixed catalog are calibrated.**  0.848 and 1.018 by the variance
split, 0.46-0.99 by the `N_obs` scaling, and `J_ens/H` = 1.266 by the information
identity -- three independent routes, all consistent with 1, where every one of
them read 2-5.5 before the fix.  **The per-event likelihood and the selection
term are no longer implicated at all.**

**2. What is left is the CATALOG term, plus a bias.**  The excess appears only
when the catalog is redrawn, and the 7001 family carries a grand mean of
**72.58 +- 1.41** against a truth of 67.74 -- **+3.4 sigma of the mean**, over 47
independent catalogs, so not a fluctuation.  The 90000 family shows none
(67.48).  Coverage is 0.64 at the 90% level and 0.43 at 68%.

That is a coherent and unsurprising place to land: **the analysis conditions on
its galaxy catalog as if the catalog were exact.**  Catalog-to-catalog variation
-- finite galaxy counts, photo-z scatter, which hosts happen to be catalogued --
is not propagated into the quoted interval, so redrawing the catalog produces
scatter the posterior never claimed to cover.  It is a modelling choice, not a
defect, and it is shared by every dark-siren analysis that treats the catalog as
data.

**What it does NOT explain** is the bias.  Unmodelled catalog noise inflates
scatter symmetrically; it does not move 47 independent realizations up by 4.8
km/s.  A seed-family-dependent bias of that size is the open item, and it is
plainly separate from the interval question.

### Status of the tiers

* **Tier C: still fails**, at 1.51 rather than 2.65, with the failure now
  attributable to the catalog term rather than to the estimator.
* **The interval caveat is much weaker than it was.**  `sqrt(J/H)` was proposed
  as a sandwich correction of 2.35-2.46; the honest post-fix number at fixed
  catalog is **1.13**, and production's own arm never showed a departure once its
  curvature was measured properly.
* **The production medians are unchanged** (`fp` 71.70 -> 71.54).


## Gate 8(c): the bias survives an IDEAL COMPLETE catalog

The residual bias had one modelling explanation left -- the completeness model.
Below the survey limit the analysis represents uncatalogued hosts by
`(1 - C(z)) dN_exp`, and that branch sits at HIGHER redshift than the catalogued
one, so misplacing it moves `H0` upward.  The checklist's gate for exactly this
is an end-to-end run on ideal inputs, and it had never been run here.

Ideal = the world's `f_p` set to 1 (which feeds BOTH the survey draw and the
depth map `make_mock` writes, so the survey is complete AND the analysis is told
so) and `m_lim = 99`, which makes the parametric `C(z)` ~1 everywhere.  The arm
is otherwise untouched, so the only difference from an ordinary Tier C run is
that there is nothing for the completeness model to correct.

    ordinary mock (n=49)   median u = 0.323   median-of-medians +2.25 km/s   oc 1.508
    IDEAL COMPLETE (n=24)  median u = 0.277   median-of-medians +3.00 km/s   oc 1.714

**Both survive.**  The completeness model is not the carrier of either the bias
or the residual dispersion.

(Setting `f_p_survey` alone would NOT have been the ideal test: `make_mock`
always writes the UNPERTURBED `world.f_p` into its depth map, so that route
makes the survey complete while telling the analysis it is masked -- Tier D-i's
deliberate mismatch, the opposite of what is wanted.  And switching to
`dark_sirens_complete` is not available here: the mock's footprint is 1,854 of
3,072 pixels, so off-footprint events have no catalog support and the first
attempt returned n = 0 surviving realizations.)

### Where that leaves the bias

Eliminated, each by measurement: the `f_p` gather (fixed, and the ideal run has
`f_p == 1` anyway), the parameter estimation (delta-PE at truth leaves it), the
event draw (events at fixed catalog are calibrated on three independent routes),
and now the completeness model.

The candidate that fits a BIAS specifically -- as opposed to the dispersion --
is the **selection integral**.  It is computed from ONE injection set shared by
every realization, so an error in it contributes no realization-to-realization
scatter and shifts every posterior the same way.  That is exactly the observed
shape: a bias of +0.3 sigma present in every configuration tried, alongside a
dispersion that the fixed-catalog tests say is fine.  It is also consistent with
the one thing the earlier injection work did NOT test: injections were rebuilt
18x and the SCATTER did not move, which measures noise, not bias.

The check is DAG rule 3 -- that `beta` uses the same detection statistic and
threshold as the event draw.  `make_mock` draws events by thresholding
`obs_rho` from `_detect_on_observation`, and builds injections with
`proposal="population+uniform"` through `_selection_injections`; whether the two
implied detection models agree to better than a few tenths of a sigma in `H0`
has not been measured.  That is the next test, and it is cheap: regenerate the
injection set from the SAME code path that draws the events and re-run.


## The bias is the SELECTION INTEGRAL's support, and it is measured

DAG rule 3 holds on the STATISTIC, and that part is settled by reading rather
than running: events threshold `obs_rho` from `_measure`, injections threshold
`rho_opt + N(0, sigma_rho)` in `_make_selection_kernel`, and both call the
identical `snr_ref (mc_det/30)^(5/6) (1000/dl)` with the same `sigma_rho` and the
same threshold.  Both also carry the `(1+z)^(gamma-1)` rate factor -- events by
rejection at line 964, injections inside `pdraw` at line 1191.

What rule 3 does NOT cover is the SOURCE distribution, and that is where the
mismatch is.  Events are drawn from CATALOG HOSTS; injections from a smooth
`dV_c/dz`.  Comparing the DETECTED redshift distributions, injections weighted by
`1/pdraw` -- which is exactly how `mu` weights them -- over 1,440 events from the
IDEAL-COMPLETE tree and 65,791 injections:

| quantile | events | injections |
|---|---|---|
| 0.10 | 0.0635 | 0.0282 |
| 0.25 | 0.0851 | 0.0630 |
| 0.50 | 0.1113 | 0.1289 |
| 0.75 | 0.1337 | **0.2149** |
| 0.90 | 0.1587 | **0.2959** |
| mean | 0.1107 | **0.1474 (+33.1%)** |

KS distance **0.310**.  `mu` expects detections 33% more distant in the mean than
the mock produced, and the direction is exactly right for the observed bias: a
host placed at higher `z` for the same `dL` implies a larger `H0`, so the fit
compensates upward.  The bias is `+0.32` in median `u`; this is a 33% error in
the selection integral's first moment.

**Two separable pieces, and only one is a defect.**

* **9.5% of `mu`'s weight sits ABOVE the catalog's depth** (`z > 0.30`, where the
  injections reach `z = 0.596` and the events cannot exist -- their max is 0.3003
  and 0.07% lie above).  That part is NOT an error: the analysis models
  above-depth hosts as uncatalogued, so `mu` is supposed to include them.
* **Restricted to `z <= 0.30`, where catalog hosts exist, the mismatch is still
  +13.5% in the mean** and the 90th percentile is 0.244 against the events' 0.159.
  That part is a genuine disagreement between the two source distributions inside
  the volume they share.

The residual `+13.5%` is what a smooth `dV_c/dz` proposal gives against a
CLUSTERED, depth-limited host catalog: at fixed `dL` the catalog's hosts are not
uniformly distributed in comoving volume, and the injections know nothing about
that.  This is the mock-data checklist's rule 6 in a form it does not state --
not "injections must cover every trial hyperparameter" but "injections must be
drawn from the same source population the events are", which for a dark-siren
mock means the CATALOG, not the volume.

**What this licenses, and what it does not.**  It identifies a specific,
quantified inconsistency of the right sign and plausible size, and it is the only
surviving candidate after the `f_p` gather, the PE, the event draw and the
completeness model were each eliminated by measurement.  It does NOT yet prove
causation: the step that would is rebuilding the injection set by drawing hosts
from the catalog (with the same detection rule) and re-running Tier C.  That is a
generator change, not an analysis change, so it belongs to the mock rather than
to the pipeline -- and it is worth stating plainly that **this is a defect in the
VALIDATION MOCK, not in the production analysis**, whose injections are the real
LVK campaign and whose events are real detections.


## RETRACTION: the selection integral's support is FINE, and my 33% was my own error

The previous section reported that `mu` expects detections 33% more distant than
the mock produced, and read that as the cause of the residual bias.  **That is
wrong and is withdrawn.**  It weighted the injections by `1/pdraw`, but the
selection integral weights them by `p_fid / pdraw`, where `p_fid` is the
population density `_mass_spin_pdf(m1src, q, chi) * p(z) / jac / 4pi` --
exactly as `generate_mock_data._selection_neff_at_fiducial` builds it.

The proposal is `population+uniform`, a MIXTURE containing a uniform-in-`m1det`
component.  Dropping `p_fid` therefore over-weights heavy systems, which are
detectable further away, and manufactures an excess of distant detections out of
nothing.  With the correct weight:

| quantile | events | injections (correct weight) | injections (`1/pdraw`, WRONG) |
|---|---|---|---|
| 0.10 | 0.0635 | 0.0593 | 0.0282 |
| 0.50 | 0.1113 | 0.1077 | 0.1289 |
| 0.90 | 0.1587 | 0.1571 | 0.2959 |
| mean | 0.1107 | **0.1086 (-1.9%)** | 0.1474 (+33.1%) |

`Neff` also rises from 1,953 to 37,111 under the correct weight, which is the
same statement seen from the variance side.

**So the selection integral's support matches the events to about 2%**, and the
residual `KS = 0.042` (`p = 0.012` at 1,440 events) is a small effect of the
**WRONG SIGN**: the injections are slightly CLOSER than the events, where an
upward `H0` bias needs `mu` to expect detections FURTHER away.  DAG rule 3 is
satisfied both on the statistic (by reading: identical `rho_opt` formula, `sigma_rho`,
threshold, and `(1+z)^(gamma-1)` factor on both sides) and now on the support.

**The residual bias is therefore still unexplained**, with the selection integral
eliminated alongside the `f_p` gather, the PE, the event draw and the completeness
model.  What remains untested is narrower than before: the catalog KERNEL (each
galaxy's photo-z is represented by a Gaussian of width `dz`, and a
mis-specified kernel SHAPE shifts the host redshift the analysis infers) and the
`z_depth` relaxation boundary (the mock's universe extends to `Z_UNIVERSE = 0.60`
while its catalog stops at 0.30, so 9.5% of `mu`'s weight is legitimately
above-depth -- correct by design, but the branch that handles it is the one place
the analysis must model hosts it cannot see).

The lesson is the same one this file has now recorded twice: **a weight is part of
a measurement's definition, and using the wrong one produces a confident number.**
The check that would have caught it immediately is the one that did catch it --
`Neff`, which the generator already computes with `p_fid/pdraw` and which was
sitting in `truth.json` all along at 1,953 for the wrong weight against 37,111
for the right one.


# THE BIAS IS THE PHOTO-z KERNEL, and both halves of Tier C now close

One knob, one answer.  Set the catalog's redshift-error floor to zero -- a
SPECTROSCOPIC catalog, `z_obs == z_true`, kernel at its 1e-4 numerical floor --
and hold everything else at the ideal-complete configuration, so `SIGMA_Z` is the
only difference:

| | photo-`z` ON | **spec-`z`** |
|---|---|---|
| median `u` (0.5 = unbiased) | 0.277 | **0.449** |
| median-of-medians | +3.00 km/s | **+0.42 km/s** |
| overconfidence | 1.714 | **1.224** |
| coverage at 90% | 0.62 | **0.88** |

**The bias essentially vanishes and the coverage essentially closes.**  So the
residual is the catalog's photo-`z` treatment -- and it explains BOTH halves at
once, which no earlier candidate did.

The mechanism is not a width mismatch: the stored `dz` is `0.02300`, exactly the
scatter applied, verified row by row.  It is that `SIGMA_Z = 0.023` at a median
host `z` of 0.11 is a **21% fractional** width, and at that width two things bite
which a matched width does not fix -- the kernel is not truncated at `z >= 0`
(129 galaxies per realization have `z_obs < 0`, and a Gaussian centred below zero
can only put mass at higher `z`), and a 21% kernel convolved against the steeply
rising volumetric prior is where the direction of the convolution matters at
exactly the few-percent level the bias sat at.

### A no-op run that had to be caught first

The first spec-`z` attempt patched `W16.SIGMA_Z` and came back **identical to
photo-`z` ON in all three statistics to three digits** -- median `u` 0.277,
+3.00 km/s, overconfidence 1.714.  That is not a null result, it is a run that
did nothing: the knob is `make_mock.SIGMA_Z_CAT` (its own module constant, which
becomes `SurveyConfig.redshift_error_floor`), and `W16.SIGMA_Z` only seeds it at
import.  The catalog's `dz` was still `0.02300`, which is what caught it.

**Three identical digits across three independent statistics is the signature of
an intervention that did not fire**, and it is worth naming as a check: any
"the knob does nothing" result should be confirmed by reading the artifact the
knob was supposed to change, not by the summary statistics.

### What this means for the scope of the whole diagnosis

The photo-`z` scatter is a property of the MOCK's catalog, and the production
catalog is DESI spectroscopy: the ingest line's own products carry
`sigma_kde = 0.003` against the mock's 0.023, an order of magnitude smaller, and
DESI redshifts are spectroscopic rather than photometric.  So this is the third
finding in a row whose scope is the validation mock rather than the production
analysis -- and taken together with the `f_p` gather bug (which barely fired on
production's near-complete compact view) it means **Tier C was, in the end,
measuring its own mock twice over**.


# THE SPEC-z REBUILD: the bias closes, the dispersion does not

The mock rebuilt with `SIGMA_Z_CAT = 1e-4` (DESI-like) and `world16.SIGMA_Z`
tracking it, so the shell response and the catalog agree.  Tier A and Tier C at
n = 50, against the photo-`z` post-fix run on the same seeds:

| | photo-`z` (n=47) | **spec-`z`** `latent_off` | **spec-`z`** `latent` |
|---|---|---|---|
| median `u` (0.5 = unbiased) | 0.323 | **0.401** | **0.474** |
| median-of-medians (truth 67.74) | 69.99 | **68.63** | **67.93** |
| overconfidence | 1.508 | 1.449 | 1.396 |
| coverage at 90% | 0.64 | **0.82** | **0.80** |
| coverage at 68% | 0.43 | 0.54 | 0.52 |

**Tier A passes** (slope 0.9984, representable 0.9997).

**The bias is closed.**  `u` moves from 0.323 to 0.401/0.474 and the
median-of-medians lands at 68.63 and **67.93** against a truth of 67.74 -- the
latent arm is within 0.2 km/s.  That is the prediction `gate_specz` made, now
confirmed on the full realistic survey rather than on an idealized one.

**The dispersion is not.**  Overconfidence stays at 1.40-1.45 and 90% coverage at
0.80-0.82 against a nominal 0.90, so `TIER_C` is still `false`.  Note what
changed underneath: the quoted `sigma` fell from 6.40 to **3.74** -- spectroscopic
redshifts make each posterior far tighter -- and the scatter fell with it, from
9.66 to 5.42.  The RATIO barely moved.  A residual that survives a 1.7x change in
the interval's own scale is not a mis-set width; it is the catalog-realization
variance the analysis does not propagate, which is exactly where the post-fix
variance split localized it.

## Two errors of mine on the way, both caught by measurement

**1. `SIGMA_Z_CAT = 0.003` was the wrong value, and I chose it for a bad reason.**
I set it to "the value production applies", but production applies `sigma_kde =
0.003` as the ANALYSIS's smoothing on top of a nearly exact catalog redshift.
Since `sig_eff = sqrt(dz^2 + sigma_kde^2)`, setting the catalog's own scatter to
0.003 makes the kernel `sqrt(2) x 0.003` -- **41% over-wide**, replacing one
mismatch with another.  Measured before I caught it: overconfidence 1.603 and 90%
coverage 0.53 at n = 17, no better than photo-`z`.  Kept as
`sz003_tier_c_partial.json`.

**2. I read stale results and reported them.**  `tier_c.py` writes its output
INCREMENTALLY to the same path, and I relaunched without deleting the killed
run's file.  A wait condition of "at least 20 rows" was satisfied instantly by
the dead run's leftovers, and I reported "spec-`z` at 1e-4 gives overconfidence
1.352, coverage 0.75" -- which was the 0.003 run's data.  The real 1e-4 numbers
are the table above.

**Both hazards are worth naming as procedure**: an incremental writer plus a
row-count wait condition is a stale-read trap, and the fix is to delete the
output before relaunching; and when a quantity enters in quadrature, matching the
mock to the ANALYSIS's total is not the same as matching it to the TRUTH.


## Tier D on the spec-`z` mock, and why its PASS/FAIL is not resolvable at n = 20

| variant (`latent`) | pre-fix | photo-`z` post-fix | **spec-`z`** |
|---|---|---|---|
| matched | -0.857 | +0.473 | **+0.636** |
| `fp_perturbed` | -0.709 | +0.388 | +0.551 |
| `fibre_5pc` | -0.842 | +0.498 | +0.645 |
| `ls_ang_2x` | -0.790 | +0.500 | +0.644 |
| `lognormal_tail` | -0.236 | +0.238 | +0.136 |
| **TIER_D** | false | **true** | **false** |

Read at face value that is a regression, and it is not one.  The tier's own
bootstrap standard error on the bias is **0.327 sigma** (photo-`z`) and **0.321
sigma** (spec-`z`), against a gate at **0.5 sigma**.  So +0.473 and +0.636 differ
by `0.16 +- 0.46` -- statistically indistinguishable, and both sit within one SE
of the threshold they are being tested against.  **At n = 20 per variant this gate
cannot resolve its own criterion**, and which side of 0.5 a run lands on is close
to a coin flip.  The pre-fix `-0.8` values were outside that noise; these are not.

A second reason the sigma-unit comparison misleads here: spec-`z` TIGHTENS every
posterior (Tier C's mean quoted `sigma` falls from 6.40 to 3.74, a factor 1.7), so
the same absolute bias in km/s becomes a larger fraction of `sigma`.  Tier C shows
that directly -- its median-of-medians moves from 69.99 (photo-`z`, +2.25 km/s) to
68.63 and **67.93** (spec-`z`, +0.89 and +0.19 km/s against a truth of 67.74).
The absolute bias shrank by 2.5x to 12x while the sigma-unit bias barely moved,
because the ruler shrank with it.

**That is worth stating as a property of the tier ladder rather than of this run:
gates expressed in units of the quoted `sigma` get HARDER as the analysis gets
better.** A tier that passes at 6.4 km/s intervals and fails at 3.7 km/s
intervals, with the same physical bias, is reporting the improvement as a
failure.


## The spec-`z` ladder, complete — and the residual decomposed

`gate_complete` re-run on the spec-`z` mock isolates the last piece.  Three
configurations, all at `SIGMA_Z_CAT = 1e-4`:

| configuration | median `u` | overconfidence | coverage 90% |
|---|---|---|---|
| photo-`z`, realistic survey | 0.323 | 1.508 | 0.64 |
| spec-`z`, realistic survey | 0.401 / 0.474 | 1.449 / 1.396 | 0.82 / 0.80 |
| **spec-`z`, IDEAL COMPLETE survey** | **0.451** | **1.224** | **0.88** |

(The ideal-complete row reproduces `gate_specz_v2`'s 0.449 / 1.224 / 0.88 to
three digits at `dz = 1e-4` rather than 0, which is the check that 1e-4 is
numerically "spectroscopic" as intended.)

So the post-fix residual splits cleanly into two pieces, neither of which is a
defect:

* **~1.22 is irreducible on this mock even with an ideal complete catalog and
  exact redshifts.**  What varies between those realizations is WHICH GALAXIES
  EXIST -- the catalog realization -- and the analysis conditions on its catalog
  as if exact.  That variance is not propagated into the quoted interval by
  construction, so redrawing the catalog produces scatter the posterior never
  claimed to cover.  It is the modelling choice every dark-siren analysis makes
  when it treats the catalog as data.
* **~0.2 more comes from survey incompleteness** (1.224 -> 1.40-1.45, coverage
  0.88 -> 0.80-0.82), i.e. from the completeness model working on a real
  magnitude limit and mask rather than on nothing.

## Tier B on the spec-`z` mock

    field only        (latent vs latent_off)        0.083 sigma
    f_p channel       (latent_off vs nofp)          5.366 sigma
    b_gal only        (latent_bgal vs latent)       0.0001 sigma
    latent vs table                                 5.003 sigma   gate: < 0.3
    ci90 width        latent 12.95   table 53.58
    TIER_B: false

The ladder's headline is unchanged and sharper than ever: **the field moves `H0`
by 0.083 sigma and the `f_p` channel by 5.37.**  With spectroscopic redshifts the
`f_p` channel's effect is larger still than the 2.24 sigma the photo-`z` mock
gave, because the intervals are 1.7x tighter.

`TIER_B` fails on its latent-vs-table gate, and that comparison is not
interpretable here: the table arm's 90% width is **53.6 km/s against latent's
13.0**, a factor 4.1.  The table arm carries a Q table built at
`--n-members 8` on this catalog, and a 53 km/s interval on a 60-event mock is
the Q ensemble's spread, not a cosmology measurement.  Comparing two arms whose
widths differ 4x through a "shift in units of sigma" gate is measuring the
denominator.


# A FINDING AGAINST S-3 ITSELF: `f_p` x Q table DOUBLE-COUNTS the mask

S-3 (PR #406) lifted the loader's refusal of `--per_pixel_completeness` alongside
a Q table, on the grounds that the only thing missing was the `f_p`-weighted
empty-pixel budget.  Tier B's `table_fp` arm -- the arm that refusal had made
impossible, and which I added to exercise it -- says that reasoning was
incomplete.

On the spec-`z` mock, one realization (seed 7001), truth `H0 = 67.74`:

| arm | median | 90% CI | width |
|---|---|---|---|
| `latent` | 69.65 | [62.5, 75.4] | 12.95 |
| `table` (`f_p` OFF) | 116.22 | [83.6, 137.1] | 53.58 |
| **`table_fp` (`f_p` ON)** | **41.24** | **[36.1, 46.3]** | **10.18** |

`table_fp` is **confidently wrong**: its 90% interval EXCLUDES the truth by
21 km/s, and it is the TIGHTEST arm in the run.  That is worse than `table`'s
wide-and-wrong 116, because a narrow interval in the wrong place is what a
consumer trusts.

**The mechanism, measured.**  The Q table is fit to the catalog's OBSERVED counts,
so it already carries the footprint.  On this catalog, at a low-`z` slice:

    mean Q on-footprint   1.624   (1,854 pixels)
    mean Q off-footprint  0.050   (1,218 pixels)      a 32x suppression
    corr(Q, f_p)         +0.410

So the missing-galaxy budget is multiplied by the mask TWICE -- once through
`Q_p(z)`, which learned it from the counts, and once through `C_p = f_p C`.  The
budget off-footprint is then suppressed by `~0.05 x 0` instead of `0`, and
on-footprint by `f_p` on top of a Q that already absorbed it.

**This is exactly the hazard the ORIGINAL refusal named, one term over.**  The
loader's docstring said a per-pixel count-derived `C` already contains the mask
loss, "multiplying would double-count it", and used that to require
`c_mode` in {aggregate, selection} -- where `C` is parametric, so multiplying IS
right.  What neither the docstring nor my change noticed is that `Q` is ALSO
count-derived.  Moving the mask out of `C` did not move it out of `Q`.

## What must change

1. **#406/#407's capability should be gated, not shipped as-is.**  Admitting
   `f_p` alongside a Q table is only safe for a Q table built to be `f_p`-aware --
   i.e. one whose builder divided the mask out, or was fit on mask-corrected
   counts.  Nothing stamps that property today, so the loader cannot check it.
   The honest options: (a) re-refuse the pairing unless the Q artifact carries an
   explicit `f_p_aware` stamp, (b) have `build_lognormal_completion` take the
   depth map and divide the mask out, stamping that it did, or (c) keep the
   pairing but require the operator to assert it via a flag.  (a) is the smallest
   correct change and (b) is the right one.
2. **The production `q_fp` number is affected.**  The S-3 production measurement
   reported `q_fp` = 80.61 against `q_nofp` = 89.90 and read the -2.49 sigma shift
   as "the mask matters on the Q line".  That shift is now partly or wholly the
   DOUBLE-COUNT, not the mask, so **80.61 must not be quoted** and the
   arm-selection question it was raising is void until (1) is settled.
   `desi_full259/data/s3_footprint/RESULT.md` needs this caveat.
3. **What is NOT affected**: the `f_p`-without-Q arms (`fp`, `latent`), because
   there is no second mask channel.  The production headline 71.54 stands, and so
   does the whole f_p-gather bug fix, which is independent.

## The lesson, and it is uncomfortable

I built the capability, wrote its tests, and pinned the arithmetic against a
brute-force reference -- and all of that was correct.  What I did not do was ask
whether the OTHER factor in the product already contained the thing I was
multiplying in.  The unit tests could not catch it: they pin
`Sum_empty Q_p (1 - f_p C)` against a hand-computed
`Sum_empty Q_p (1 - f_p C)`.  Checking a formula against itself proves it is
implemented, never that it is the right formula.  What caught it was running the
arm end to end and looking at where `H0` landed.


# THE PRODUCTION ANCHOR CARRIES THE PHOTO-`z` DEFECT TOO

Found by an independent review of the PR stack, and it is the same defect this
file spent a day chasing in the mock -- sitting in the SHIPPED production path.

`darksirens/cli/build_latent_field.py:447`:

    sigma_z = lambda z: 0.023 * np.ones_like(z)

Hard-coded, with **no CLI flag anywhere in the stack**.  That `sigma_z` is the
width of the photo-`z` convolution in `shell_response`, i.e. the radial kernel of
the count channel's forward model `W`.  The production line is DESI
**spectroscopy** (`SIGMA_Z_CAT = 1e-4`, analysis `sigma_kde = 0.003`).

**The arithmetic is damning on its own.**  With the shipped defaults
`--z-depth 0.30 --n-shells 12` the shell width is `0.30/12 = 0.025`, and
`sigma_z = 0.023` is **92% of a shell width**.  Each shell's response is therefore
smeared across roughly its two neighbours.  The reviewer computed the resulting
`W` directly: shell 0 covers `z in [0, 0.025]` but the shipped `W` places its
galaxies at a mean `z` of **0.0446** -- outside the shell entirely -- and every row
is ~3.2x too wide (radial std 0.0175-0.0238 against 0.0049-0.0073 at the truth).

**What it implicates.**  `experiments/field_level_plan/pr5/sbatch_anchor_v2.sh`
builds the production anchor with this CLI, and that artifact
(`latent_anchor_v2a.h5`) is what the 259-event latent arm consumes via
`--anchor`.  So the count-channel MAP `xi_hat` was fit with basis rows registered
to the wrong redshifts, and the field `Q(p,z)` the seam generates from it is
radially misregistered at exactly the redshifts the GW hosts occupy.

**And it changes how the ladder's headline should be read.**  Every arm comparison
says the field does nothing measurable -- 0.06 sigma on production, 0.083 sigma on
the spec-`z` mock.  That may be a statement about a MISREGISTERED field rather
than about the field.  The finding does not overturn "the field is not the
consequential part of this programme" -- `f_p` dominating by 50x is robust to it --
but it does mean the field's own number has not been measured on a correctly
registered basis.

**My part in missing it.**  When I fixed the mock's `sigma_z` I wrote, in
`world16.py`'s own comment, that "the PRODUCTION anchor is built by
`cli/build_latent_field.py` from its own depth map and is untouched" -- and
recorded that as reassurance.  It was the opposite: the production builder was
untouched BY THE FIX, carrying the same 0.023 whose effect I had just measured
to be the sole cause of the mock's bias.  I checked that changing the mock would
not disturb production, and never asked whether production needed the same
change.

**What to do**: add a `--sigma-z` flag defaulting to the survey's actual value
(and stamp it into the artifact's provenance, which already hashes the basis
geometry), then rebuild the production anchor and re-run the latent arm.  Until
then the latent arm's 71.95 should be quoted as "the field, on an anchor built
with a photometric kernel", not as the field's measured effect.


# RETRACTION: THE PRODUCTION CATALOG IS PHOTOMETRIC, SO THE PHOTO-`z` FINDING IS NOT MOCK-ONLY

Found by an independent review of PR #405, and it overturns the scope claim this
file has been closing on.

Everything above that says "the production line is DESI **spectroscopy**
(`SIGMA_Z_CAT = 1e-4`, analysis `sigma_kde = 0.003`)" is **wrong**.  I mistook
`sigma_kde` -- an analysis smoothing floor -- for the catalog's redshift
uncertainty.  Measured directly on the catalog the 259-event line actually loads
(`desi_ingest/data/pixelated_n64/catalog_pixelated_nside_64.h5`, 3.1M live
galaxy entries):

    median dz                    0.0238
    p75 / p95 / p99      0.045 / 0.069 / 0.089
    fraction dz < 1e-3 (true spec-z)   0.387
    fraction dz > 0.02                 0.532
    sigma_kde = 0.003 contributes      0.8% in quadrature

**The production catalog carries essentially the same 0.023 the mock did**, and
the ingest provenance says so itself (`zerr_range: [2.5e-7, 0.0999999]`).

## What this retracts

* **"Tier C was measuring its own mock, twice over"** -- withdrawn.  The `f_p`
  gather bug was genuinely mock-scale (production's compact view is 49,143 of
  49,152 rows), but the photo-`z` half is NOT: the production catalog has the same
  kernel width that, on the mock, was the sole cause of the residual bias.
* **The spec-`z` rebuild made the mock LESS faithful, not more.**  Setting
  `SIGMA_Z_CAT = 1e-4` gave the validation mock redshifts two orders of magnitude
  more precise than the survey it validates.  Its tiers now measure a
  configuration production does not have.  The correct mock carries production's
  `dz` DISTRIBUTION (median 0.024, p95 0.069), not a single spectroscopic value --
  and not the old single 0.023 either, since a third of production's galaxies
  really are spectroscopic.
* **`build_latent_field.py`'s hard-coded 0.023 was approximately RIGHT**, and my
  prescription to default `--sigma-z` to "the survey's actual value ~1e-4" would
  have under-smoothed the count channel's forward model by ~100x -- replacing the
  flagged defect with a worse one.  The flag is still right to exist (the value
  was unstated and unstampable); its default stays 0.023.

## What survives, and what it now means

The mechanism is unchanged and still measured: a kernel that is a large FRACTION
of the host redshift biases `H0` upward, and removing it took the mock's median
`u` from 0.277 to 0.449 with coverage 0.62 -> 0.88.  What changes is who it
applies to.  Production's fractional width is **~10%** (0.024 at a median host
`z` of 0.237) against the mock's **21%** (0.023 at 0.11) -- smaller, not absent.
So this is now a candidate systematic ON THE PRODUCTION LINE, of unknown size,
where I had just finished arguing it was a mock artifact.

The one part that genuinely does not transfer is the `z_obs < 0` truncation
asymmetry: at production's median `z` the kernel mass below zero is ~0.03%,
against the mock's 129 galaxies per realization.

## What to do

1. **Rebuild the mock at production's `dz` distribution** and re-run the tiers.
   That is the mock the line actually needs, and neither 0.023-flat nor 1e-4 is
   it.
2. **Quantify the production-side bias directly** -- the cheapest handle is the
   fractional-width scaling the mock now provides two points of (21% -> +0.32 in
   `u`; ~0% -> +0.05), evaluated at production's 10%.
3. **Do not quote "the photo-`z` effect is a mock property"** anywhere.

## The lesson

I checked what value the ANALYSIS applies (`sigma_kde = 0.003`) and never opened
the catalog to see what the DATA carries.  The number was one `h5py` call away
for the entire campaign, and the scope claim that rested on it -- the reassuring
half of two days' work -- was false.  When a conclusion turns on "our survey is
X", measure X on the artifact.
