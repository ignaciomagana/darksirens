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
| event spread at FIXED catalog | 15.57 | 16.49 |
| **overconfidence, events only** | **2.110** | **2.367** |
| catalog spread (debiased) | 7.93 | — |
| mean quoted `sigma` | 7.38 | 6.97 |
| fraction of variance from events | 0.72-0.82 | 0.67 |

Two things this settles.

**The deficit survives conditioning on the catalog.**  Holding the catalog
fixed and varying only the event draw still gives 2.11-2.37.  The analysis
treats the catalog as observed data, so its coverage ought to hold conditional
on that catalog -- and it does not.  Whatever is wrong is not "the tier varies
the catalog while the posterior conditions on it".

**There is a modest bias as well as a width deficit.**  Grand mean 64.10
against `H0_true` = 67.74, i.e. -3.6 km/s, about -0.5 of the quoted `sigma`.
The medians span 44.5 to 100.5 -- +-4 sigma excursions against a quoted 7.4.

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
