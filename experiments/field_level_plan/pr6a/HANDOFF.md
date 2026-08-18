# Where to pick up (checkpoint 2026-08-18)

## Jobs still running on js2h100 — DETACHED (PPID 1), survive any local reset

    tier_c_truncated.py   ~19/24   -> /media/volume/darksirens-data/darksirens-dev-data/runs/q/tier_c_trunc_n24.json
                                      log: /media/volume/darksirens-data/darksirens-dev-data/logs/q/tier_c_trunc.log
    nobs_scaling.py       ~95/160  -> /media/volume/darksirens-data/darksirens-dev-data/runs/q/nobs_scaling_n40.json
                                      log: /media/volume/darksirens-data/darksirens-dev-data/logs/q/nobs_n40.log

Check them with:
    ssh js2h100 "grep -cE '^\[tier C\]' /media/volume/darksirens-data/darksirens-dev-data/logs/q/tier_c_trunc.log"
    ssh js2h100 "grep -cE '^\[nobs\]'   /media/volume/darksirens-data/darksirens-dev-data/logs/q/nobs_n40.log"

## What each will answer

* **tier_c_trunc** — the rule-7 test: is the ~2.5x overconfidence caused by the
  mock's metadata-only depth (4.87% of galaxies above z_depth, where production
  has ZERO)?  Baseline to compare against, same seeds: latent 2.593 /
  latent_off 2.259.  Expected null; the mock still needs the fix for fidelity.
* **nobs_scaling n=40** — is the median scatter flat in N_obs or does it fall as
  1/sqrt(N)?  At n=6 per cell the error was 32% and the two were
  indistinguishable; n=40 takes it to ~11%.  Flat => a common-mode term that
  never averages down.

## Then

Gate 8(a) already localized the deficit to the ESTIMATOR (catalog term /
completion / selection integral), so **the next test is catalog-side, not
data-side**.  See memory `tier-c-dispersion` and
`experiments/field_level_plan/pr6a/DIAGNOSIS.md`.
