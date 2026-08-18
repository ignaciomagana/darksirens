# Where to pick up (checkpoint 2026-08-18, second)

## js2h100 is idle

The work queue (`queue_h100.sh`, copied here for provenance) ran to
`===== QUEUE DONE =====` at 08:57 UTC and the box has since rebooted.  Nothing
is running; the GPU is at 0%.

Everything it produced is now harvested into this directory:

| run | artifact | written up in |
|---|---|---|
| `tier_c_n100` | `tier_c_n100.json` | `DIAGNOSIS.md`, "Tier C at `n = 100`" |
| `mock_81xx` + `pecal_81xx` | `pecal_multiseed/`, `pecal_multiseed.json` | `DIAGNOSIS.md`, "The PE offset across eight independent mocks" |
| `variance_8x8` | still on the box at `runs/q/variance_8x8.json` | NOT harvested |

The `variance_8x8` rerun (8x8 cells, up from 5x5) reports a grand mean of 60.54
against the 64.10 recorded from the earlier split -- worth a look before it is
quoted anywhere.  Its log and json are at
`/media/volume/darksirens-data/darksirens-dev-data/{logs/q,runs/q}/variance_8x8.*`.

## What the harvest settled

* the overconfidence factor is `2.645` / `2.482`, replicated on 100 realizations
  from a seed block disjoint from every earlier pass, so the 15% precision
  caveat is gone;
* PE over-sharpness is eliminated across EIGHT mocks, not one (mean `resid_sd`
  1.156, never above 1.433, against the 2.6 required);
* the mock's PE carries a stable `+0.399 sigma` offset which is the entire
  non-uniformity of its PP test -- a bias channel, not the width failure, and
  its cause (selection working as designed vs a construction defect) is open.

## What is still open

Unchanged by this harvest: gate 8(a) localises the width deficit to the
ESTIMATOR (catalog term / completion / selection integral), so **the next test
is catalog-side**.  The production counterpart of `J/H` still needs an ensemble
of synthetic 259-event datasets on the production catalog; the OPG lower bound
is all there is, and the first attempt at a production `J/H` was withdrawn
(commit 122a2ba) because `H` from a +-2 km/s second difference does not
reproduce the posterior width.

See memory `tier-c-dispersion` and the tail of `DIAGNOSIS.md`.
