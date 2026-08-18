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
