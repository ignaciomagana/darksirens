# Where to pick up (checkpoint 2026-08-18, third)

Everything queued on 2026-08-18 has run.  `js2h100` is idle; nothing is pending
on PSC (the S-3 job was cancelled there and run on the H100 instead).

## What was measured, and what is now the owner's call

**1. S-3 is fixed and measured on production** (`845858d`, `3d52b86`).  The
Q-table line can carry a mask for the first time.  Four arms now exist:
`nofp` 90.25, `fp` 71.70, `q_nofp` 89.90, `q_fp` **80.61**.  The mask is worth
`-2.49 sigma` on the Q line, and Q and the mask INTERACT (Q alone: 0.35 km/s;
Q with the mask: +8.91).
**OWNER DECISION:** the ladder's headline arms (`fp`/`latent`, 71.7/72.0) carry
the mask and no Q table because that was the only runnable pairing; the shipped
scan's `selq_radial` is `q_nofp` and should now be `q_fp`.  They differ by 8.9
km/s.  See `desi_full259/data/s3_footprint/RESULT.md`.

**2. The OPG shortcut is dead** (`9b1c4b8`, `f16518a`).  `J_OPG` understates `J`
by 4.8-10x on the mock and returns `J/H ~ 1` where the truth is 3.9-8.2, so
production's `J/H = 0.855/1.240` is not evidence production is clean.  The
production `J` needs an ensemble: `desi_full259/ENSEMBLE_DESIGN.md`, whose §4 is
an owner decision (the injection set has no SNR column, so DAG rule 2 cannot be
honoured exactly).

**3. Tier C's catalog-side scan is complete** and closes nothing: the kernel
width has no leverage, the completeness WEIGHT is a monotone lever but the
mock's completeness model is exactly right by construction, and the `per_pixel`
estimator trades width for a `+1.71 sigma` bias.

**4. The mock's PE offset is the selection effect** (`200fe6f`), measured
against its own parameter-free prediction over 540 events.

## The one live thread

The per-dataset common mode in the score (`DIAGNOSIS.md`, last section) survives
delta-PE, a bootstrap null, and a verified capture, and contradicts what i.i.d.
events must give by a factor 6.  Either the mock's event draw is not i.i.d. or
the per-event score has a dataset-level dependence — and the second would be a
property of the LIKELIHOOD, with `sqrt(J_ens/H) = 2.2-2.9` sitting exactly on
Tier C's measured overconfidence.  That is the cheapest remaining route to the
root cause and it does not need the production ensemble.

Suggested next test: evaluate the per-event scores on datasets assembled by hand
from i.i.d. draws, bypassing `_draw_events_until_detected` entirely.  If `R = 1`
there, the generator's event draw is the culprit; if `R = 6`, the coupling is in
the estimator.

## Still true from before

The production 259-event run remains held for the owner's gate.  `variance_8x8`,
the `n = 100` Tier C pass and the eight PE-calibration mocks are all harvested.
