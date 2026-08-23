# S-3 on the production line, measured (2026-08-18)

> **CAVEAT ADDED 2026-08-20 — `q_fp` = 80.61 IS WITHDRAWN.**  The `f_p` x Q-table
> pairing this run exercises DOUBLE-COUNTS the survey mask: the Q table is fit to
> the catalog's observed counts and therefore already carries the footprint
> (measured on the closure mock: mean `Q` 1.624 on-footprint vs **0.050** off, a
> 32x suppression, `corr(Q, f_p) = +0.41`), so applying `C_p = f_p C` on top
> applies the mask twice.  On the mock the same pairing puts `H0` at **41.24
> [36.1, 46.3]** against a truth of 67.74 — confidently wrong, and the tightest
> arm in that run.  So the `-2.49 sigma` shift below is partly or wholly the
> double-count rather than the mask, **80.61 must not be quoted**, and the
> arm-selection question it raises is void until the pairing is gated on an
> `f_p`-aware Q artifact.  Full record: `field_level_plan/pr6a/DIAGNOSIS.md`,
> section "A FINDING AGAINST S-3 ITSELF".
>
> **Unaffected**: the `f_p`-without-Q arms (`nofp` 90.25, `fp` 71.70 → 71.54 after
> the f_p-gather bugfix), which have no second mask channel.

The shipped `selq_radial` configuration -- `c_mode=selection` plus the radial
lognormal Q table -- had no legal masked form: the loader refused
`--per_pixel_completeness` alongside a Q table, because the field normalizer had
no `Sum_{p empty} f_p Q_p(z)` budget.  So every Q-table run on this
footprint-limited survey modelled its 38% of empty sky as `Cbar`-COMPLETE, and
there was no way to run it otherwise.  Commit 845858d built the missing budget.
This is the first measurement of what that was worth, on the 259-event line,
clean guard convention, `H0` in [20, 140] at 2 km/s.

| arm | Q table | mask | `H0` median | 90% CI | ms/eval | finite |
|---|---|---|---|---|---|---|
| `nofp` | -- | -- | 90.25 | [83.0, 96.5] | 1702 | 116/121 |
| `fp` | -- | yes | 71.70 | [65.0, 79.1] | 3215 | -- |
| `q_nofp` | yes | -- | **89.90** | [82.5, 96.5] | 1815 | 58/61 |
| `q_fp` | yes | yes | **80.61** | [73.9, 86.8] | 3302 | 61/61 |

(The first two rows are the Q-free arms already on record in
`field_level_plan/prod/h0_latent_scans.json`; the last two are this run.)

## What it says

**The mask matters on the Q-table line too: `-9.29` km/s, `-2.49 sigma` of the
masked arm's own width.**  S-3 is not a mock-scale artifact and it is not
confined to the Q-free configuration.  The exposed arm also fails to evaluate at
the three lowest nodes (`H0 = 20, 22, 24` are `-inf`) where the masked arm is
finite everywhere -- the unmasked budget is not merely displaced, it is
inconsistent enough to kill the likelihood at the edge of the prior.

**The Q table and the mask interact, and neither reading survives on its own.**
Without the mask, adding Q moves the median by `0.35` km/s -- nothing.  With the
mask, adding Q moves it by `+8.91` km/s.  A summary of the form "the Q table is
worth X" is therefore false as stated: what Q is worth depends entirely on
whether the completeness it modulates is the right one.  The natural reading is
that on the unmasked line the completeness error dominates and swamps the LSS
modulation, but that is an interpretation and this run does not separate it from
the alternative, that the f_p-weighted empty-pixel budget itself carries the
difference.  Both sides of the missing budget changed at once.

**Which arm is production is now an owner decision.**  The field-level ladder's
headline arms are `fp` and `latent` (71.70 and 71.95), which carry the mask and
no Q table -- the only combination that WAS runnable.  The shipped 259-event
scan's headline configuration is `selq_radial`, which is `q_nofp` as run to date
and should be `q_fp` from now on.  Those two answers differ by 8.9 km/s.
Nothing here says which model is right; it says the choice is now a choice
rather than a constraint.

## Caveats that ride with it

* The Q table is fixed at its build-time fiducials (`H0 = 67.74`, `Om0 =
  0.3075`, its own `n0` and bias), as it always has been.  Every arm above shares
  that, so the CONTRASTS are clean, but a Q-table arm's absolute median inherits
  the fiducial.
* The clean guard convention throughout (`selection_neff_soft_guard=False`,
  `max_likelihood_variance=1e6`), not the shipped scan's soft guard, for the
  reason PR-5b measured: the soft guard's wall responds to `Neff`, and a
  comparison run under it measures the guard.
* The interval widths are NOT to be read as uncertainties on these medians
  without the Tier C caveat: the mock says intervals on this estimator are ~2.5x
  too narrow, production has no ensemble to check it, and the OPG shortcut that
  looked like a check has been measured blind (`pr6a/DIAGNOSIS.md`).

---

## RESOLUTION (2026-08-23): the pairing is gated, the mask-free Q exists, and the number is replaced

The caveat above said the arm-selection question was void "until the pairing is
gated on an `f_p`-aware Q artifact".  Both halves now exist:

* the loader admits `f_p` x Q only on an EARNED `f_p_aware` stamp
  (`_verify_mask_free`: off-footprint `|logQ| <= 1e-6` exactly, and per-slice
  `|corr(Q, f_p) - corr(N/f_p, f_p)| <= 0.10` against the catalog's OWN
  density-depth coupling, which is +0.11..+0.24 on this catalog and so cannot
  be zero-anchored);
* `data/fits/q_v4_depthmap_prod.h5` earns it (worst delta 0.029; built with
  `--depth-map` + `--q-support-depth 0.30`, the catalog's measured truncation,
  and the wrap-padded truncated solve).

Closure first (rb spec-z world, truth 67.74): the admitted pairing moves
`table_fp` from 41.24 [36.1, 46.3] (truth excluded) to 76.77 [67.3, 87.4]
(truth at the 5.7th percentile); record
`field_level_plan/pr6a/sz_tier_b_tablefp_v4.json`.

Then the production line (this run, `data/s3_footprint_v4/`):

| arm | Q table | mask | `H0` median | 90% CI |
|---|---|---|---|---|
| `q_nofp` | v4 | -- | 90.22 | [82.6, 96.6] |
| `q_fp` | v4 | yes | **71.47** | [64.5, 79.2] |

**The Q-table line now AGREES with the Q-free line on the masked answer:
71.47 vs `fp` 71.54.**  The withdrawn 80.61 was the double-count; the true
mask shift on the Q line is -18.8 km/s (-4.40 sigma), the same story the
Q-free arms told.  The arm-selection question is no longer void: masked arms
agree at ~71.5 with or without Q.
