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
| `variance_8x8` | `variance_8x8.json` | `DIAGNOSIS.md`, "The variance split at 8x8" (table corrected against it) |

The `variance_8x8` "discrepancy" flagged at harvest was a misreading: 64.10 and
60.54 are the `latent_off` and `latent` grand means of the SAME run, not two
values of one number.  Harvesting it did turn up something real though -- three
cells of the 8x8 table in `DIAGNOSIS.md` had been transcribed from an
intermediate and disagreed with the run's own json, and are now corrected.

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
