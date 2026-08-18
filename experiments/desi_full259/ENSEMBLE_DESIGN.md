# The production `J` ensemble: what it takes, and the one thing that blocks it

**Status: DESIGNED, NOT BUILT. It needs an owner decision (§4) before it is
worth building.**

## 1. Why it is now the only route

`J/H` is the information identity behind Tier C's ~2.5x interval deficit: for a
correctly specified model the expected information `J = Var(dlogL/dH0)` equals
the observed information `H`, and a sandwich interval `H^-1 J H^-1` is correctly
covered whatever the misspecification.  The mock measures `J/H = 6.03` from its
ensemble.  Production has no ensemble, so `J` was estimated from ONE dataset by
the outer product of per-event gradients.

That estimator has now been calibrated against the truth on the mock
(`pr6a/opg_calibration.py`, `opg_structure.json`):

    J_OPG understates J by R = 4.8-10 (five catalogs, medians 5.2, 6.4, 6.5)
    and returns J/H = 0.75-1.21 where the truth is 3.9-8.2.

Production's OPG returned `J/H = 0.855` (fp) and `1.240` (nofp).  Those are
exactly what a configuration WITH the defect reports through this estimator.
The shortcut has no discriminating power, and the reason is structural: the
missing variance is a per-dataset COMMON MODE (8-9% of the per-event score
variance), and any single-dataset estimator that centres on its own mean
destroys it.  No cleverer single-dataset estimator fixes that.

So: an ensemble of synthetic 259-event datasets on the production catalog, at
fixed catalog and fixed injections, or no production `J`.

## 2. The construction, rule by rule

Essick & Fishbach's DAG checklist (`mock-data-dag`), applied to this build:

1. **Detection is a function of the data.**  The synthetic events must be a
   sample from `p(theta | detected)` under the model.  The clean way here is to
   RESAMPLE the shipped injection set -- 1,067,946 detections with `pdraw` --
   with weights `w_k = p_pop(theta_k) p_z(z_k, pix_k | catalog, H0*) / pdraw_k`.
   That realizes `p(theta|det)` exactly and, by construction, is consistent with
   the `beta` the likelihood computes from the SAME injection set (rule 3).
2. **One noise realization per event, used everywhere.**  This is the blocker;
   see §3.
3. **`beta` uses the same statistic and threshold as the event cut.**  Satisfied
   by construction if the events come from the injection set itself and the
   likelihood keeps using it.
4. **The synthetic analyst may know only data-derived quantities.**  The pinned
   `theta` (`log10n0`, `delta`, `theta_sel`) must be the SAME values the
   production scan pins -- not per-dataset refits.
5. **No posterior-derived event cuts.**  The 259-event line takes no cuts, so
   nothing to match.
6. **Injection coverage over every trial `H0`.**  Already the production
   scan's problem, unchanged: the ensemble is evaluated at three nodes around
   the peak, well inside the shipped coverage.
7. **Depth truncation must be in the rows the likelihood reads.**  The catalog
   is untouched by this build (it IS the production catalog, `max z = 0.3000`).
8. **Gate before quoting.**  See §5.

## 3. The blocker: the production injection set has no SNR

`selection_o3o4ab_allsky.h5` carries `m1det, m2det, dL, chieff, ra, dec,
redshift, pdraw` and `significance_type = 'far'`.  Detection was a FAR cut made
by the real pipeline.  There is no `rho_opt`, so the noise draw that decided an
injection's detection cannot be reconstructed -- and rule 2 says that same draw
must shape the event's PE.  The mock's own measurement makes the size of this
concrete: its detected events carry a residual mean of `+1.28 sigma` below
threshold in true SNR, falling to zero by `rho ~ 10`
(`pr6a/pe_offset_vs_snr.json`).  Synthetic PE drawn WITHOUT conditioning on
detection would have no such structure.

Three options, none free:

* **(a) Unconditioned PE.**  Resample injections, synthesize PE centred on a
  fresh noisy observation with no detection conditioning.  Cheapest; violates
  rule 2; the missing Malmquist structure is a per-event bias whose
  dataset-to-dataset variation enters `J` in an uncontrolled way.  Since `J` is
  a VARIANCE, a constant offset is harmless -- but that it is constant is an
  assumption, not a measurement.
* **(b) A calibrated SNR proxy, used consistently.**  Define `rho_opt(theta)` by
  a standard inspiral scaling calibrated so the injection set's detection
  fraction is reproduced, draw `n` truncated to `rho_opt + n > thr`, and use
  that same `n` in the PE.  Rule 2 is honoured -- but then rule 3 requires the
  synthetic analyses to use a `beta` built from the PROXY, so the ensemble
  characterises a neighbouring estimator rather than production's exactly.
* **(c) Hybrid.**  Production's `beta` and injection set, truths resampled from
  the injections, PE noise conditioned on detection through the proxy.
  Inconsistent in principle (two statistics), defensible in practice, and it
  must be stated in the result.

## 4. THE OWNER DECISION

Which of (a), (b), (c) -- or: is the production `J` worth this at all, versus
quoting the interval with the mock's factor as an explicit caveat?  The cost is
a day or two of careful construction and a few hours of GPU; the risk is that
the answer is conditional on the synthetic PE model either way.

A cheaper thing to do FIRST, and the reason this is not already built: **find
the common mode's mechanism on the mock.**  It is not the selection correction
(variance `1.5e-27`), not heavy tails (per-event scores are near-Gaussian), and
not the events' redshifts (`z` explains 2.1% of the per-event variance, and
removing a smooth `g(z)` leaves the ratio at 6.6/5.8).  If the mechanism turns
out to be something computable directly from the production likelihood, the
ensemble is unnecessary.  `pr6a/opg_calibration.py --delta-pe` is the running
test: under delta-function PE each per-event score is a pure function of that
event's own parameters, so i.i.d. events MUST give `R = 1` -- if the common mode
survives that, the events are not i.i.d.; if it vanishes, it lives in the PE
realization.

## 5. Gates, if it is built

Nothing is quoted until all four pass:

* **G1 closure** -- delta-function PE at truth on one synthetic dataset
  reproduces the analytic score to machine precision.
* **G2 recovery** -- the H0 posterior over the ensemble is centred on the value
  the datasets were generated at (a bias here invalidates `J` as much as it
  invalidates a median).
* **G3 the deliverable** -- `J_ensemble` vs `J_OPG` on the SAME datasets, i.e.
  production's own `R`, with the ordering and completeness checks
  (`production_opg.py`) passing on every dataset.
* **G4 robustness** -- `R` recomputed under a second PE model (different
  fractional `dL` error, different sky area).  If `R` moves by more than its own
  ~35% sampling error at `n = 16`, the number is a property of the synthetic PE
  and must not be transferred to the real events.
