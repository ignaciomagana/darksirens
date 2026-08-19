# Where to pick up (checkpoint 2026-08-19)

`js2h100` is idle; nothing is queued on PSC.  Three PRs are open and stacked.

## The result

**Tier C's ~2.5x overconfidence is the `f_p` channel breaking the information
identity.**  Two routes that share no arithmetic agree to 2% on the
deliverable's own configuration:

| arm | `sqrt(J_ens/H)` predicted | Tier C overconfidence measured |
|---|---|---|
| `f_p` ON | **2.35** | **2.40** |
| `f_p` OFF | 0.87 | 1.81 (biased +5.11 sigma; see below) |

Measured by regrouping the same events into new datasets (`event_reshuffle.py`),
each arm at its OWN peak, 16 regrouped datasets, identity round-trip control
60/60 exact.  `J_ens/H` = 5.50 with `f_p` and 0.75 without; `J_OPG/H` = 0.99 in
BOTH, which is why the single-dataset shortcut could never have found this.

Eliminated on the way, each with a number: the event draw (regrouping leaves
`R = 5.57`), the field normalizer (conditional weighting leaves `R = 6.70`), the
PE (survives delta-PE), the capture (block-invariant multiset), and the events'
redshifts (`g(z)` explains 8%).

## What is open, in the order I would take it

1. **The mechanism.**  `f_p` is localised, not explained: why `C_p = f_p C` on
   both sides of the budget makes per-event scores share a per-dataset mode is
   unknown, and the obvious suspicion -- coupling through the survey-global
   budget -- is already refuted by the conditional-weighting run.  This is the
   last unknown between here and a defensible interval.
2. **Then the production `J`.**  `desi_full259/ENSEMBLE_DESIGN.md` §4 is an
   owner decision (the injection set has no SNR column).  Note that #2 may not
   be needed if #1 yields something computable directly from the likelihood.
3. **Owner decisions unchanged**: which arm is production (`q_fp` 80.61 vs `fp`
   71.70, 8.9 km/s apart); whether PR-6a is acceptable; and the 259-event
   production run, still held.

## Two retractions on the record, both caught by their own controls

* the per-event **invariance check** -- the capture's order is data-dependent, so
  its `(source, event) -> slot` mapping was wrong.  The identity round-trip
  control passed and could not have caught it; the REVERSED-events control did.
* the first **`f_p` bisect** (`R = 0.885`) -- measured 59 km/s off that arm's
  peak.  Right answer, invalid measurement; the like-for-like re-run replaced it.

## PRs

* **#406** `feat/s3-footprint-mask` -- the S-3 library fix, stacked on #404.
* **#407** `feat/s3-followups` -- the Q ensemble x `f_p` pairing and the K>=2
  guard gap, stacked on #406.
* **#405** `feat/field-level-h100-production` -- the results branch.

The ladder #395-#404 is still unreviewed, and #406/#407 sit on top of it.
