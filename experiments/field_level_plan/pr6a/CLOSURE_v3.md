# PR-6a mock closure, third pass — after the `f_p` gather bug

**Read this before `CLOSURE_v2.md`.**  That document's measurements are all
correct as measurements, and its central DIAGNOSTIC CONCLUSION is inverted by a
defect found afterwards.  Nothing in it is deleted; the sections this pass
supersedes are named below.

## The answer, first

`CLOSURE_v2.md` §V localized 82-92% of Tier C's excess variance to the GW-EVENT
channel, "with the catalog held byte-identical", and §F-4 concluded "the blocking
defect is in the GW-event channel of this mock, not in the galaxy channel".

**That is backwards, and the reason is a bug in the code that measurement ran
through.**  `catalog_views` gathered `f_p_rows` with the per-view pixel map while
`completion._row_C` indexes it by CATALOG ROW; on the union path the array was
shorter than the row count, and JAX clamps an out-of-bounds gather instead of
raising.  Every row past the short array silently took the last entry's
completeness, and because the per-view length depends on which events are in the
run, **an event's own `C_p` depended on which other events shared the run** --
precisely the "events at fixed catalog" quantity §V measured.

Fixed (`67ac782`), and re-measured:

| | `CLOSURE_v2` | post-fix |
|---|---|---|
| Tier C overconfidence, `latent` / `latent_off` | 2.65 / 2.40 | **1.51** (n=47) |
| variance split, events at FIXED catalog | 2.110 | **0.848 / 1.018** |
| `N_obs` scaling at 30 / 60 / 120 | 1.93 / 2.30 / 1.84 | **0.46 / 0.50 / 0.99** |
| information identity `J_ens / J_OPG` | 5.567 | **1.261** |
| Tier B, field-only shift | 0.136 sigma | **0.088 sigma** |
| Tier B, `f_p`-channel shift | 1.674 sigma | **2.239 sigma** |
| **Tier D** | **FAIL** | **PASS** |

Three independent routes now say the same thing: **at fixed catalog the estimator
is calibrated.**  The event channel is exonerated; what §V saw there was the bug.

## And the remainder is the mock's photo-`z`, not the pipeline

The post-fix residual (overconfidence ~1.5, bias +0.3 sigma) was chased through
five candidates, each eliminated by measurement -- the PE (delta-`PE` at truth
leaves it), the event draw (calibrated at fixed catalog), the completeness model
(gate 8(c): an IDEAL COMPLETE catalog keeps it), the selection integral (its
detected-`z` support matches the events to 2% under the correct `p_fid/pdraw`
weight, and the residual has the wrong sign), and the reliability guard
(`max_var` 1e6 -> 1e12 leaves `logL` bit-identical).

The sixth closes it.  A SPECTROSCOPIC catalog -- one knob, `SIGMA_Z_CAT = 0`,
everything else at the ideal-complete configuration:

| | photo-`z` ON | spec-`z` |
|---|---|---|
| median `u` (0.5 = unbiased) | 0.277 | **0.449** |
| median-of-medians | +3.00 km/s | **+0.42 km/s** |
| overconfidence | 1.714 | **1.224** |
| coverage at 90% | 0.62 | **0.88** |

`SIGMA_Z = 0.023` at a median host `z` of 0.11 is a **21% fractional** kernel
width.  The stored `dz` matches the applied scatter exactly (0.02300, row by row),
so this is not a width mismatch -- it is that at 21% the kernel is not truncated
at `z >= 0` (129 galaxies per realization have `z_obs < 0`) and interacts with the
rising volumetric prior at the few-percent level the bias sat at.

## Tier D post-fix: FAIL -> PASS

`tier_d.py`'s five robustness variants, median bias in units of the quoted
`sigma`, `latent` arm:

| variant | pre-fix | post-fix |
|---|---|---|
| matched | -0.857 | **+0.473** |
| `fp_perturbed` | -0.709 | **+0.388** |
| `fibre_5pc` | -0.842 | **+0.498** |
| `ls_ang_2x` | -0.790 | **+0.500** |
| `lognormal_tail` | -0.236 | **+0.238** |
| **TIER_D** | **false** | **true** |

Every variant moves by about +1.3 sigma and the tier passes.  The SIGN FLIP is
worth noting rather than glossing: the `f_p` gather defect pulled `H0` DOWN in
Tier D's configuration and UP in Tier C's, which is what an event-set-dependent
completeness offset does -- its direction depends on which rows the short array
happened to end at, so it is configuration-specific rather than systematic.  The
post-fix residual is the photo-`z` bias, positive and consistent across both
tiers (+0.24 to +0.50 sigma here, +0.3 in Tier C).

## §R revised — what this means for PR-6a

`CLOSURE_v2` §R's five points, each re-read against the post-fix numbers:

1. **"The object PR-6a adds still works" -- STANDS, and more cleanly.**  Tier A
   unchanged; the field-only Tier-B shift is now **0.088 sigma**, and the
   `f_p` channel's is **2.239 sigma**.  The `f_p` effect GREW when the bug was
   fixed, which is the expected direction for a defect that lived in that channel.
2. **"S-1 and S-2 are both real fixes" -- STANDS.**  Neither is touched by this.
3. **"Neither makes the mock closure pass, and the second was aimed at the wrong
   target" -- STANDS on the facts, and the reason is now known.**  `s_b` could not
   have closed the gap because the gap was a gather bug plus the mock's photo-`z`.
4. **"The failure is in the GW-event channel of this mock" -- WITHDRAWN.**  It was
   in the `f_p` code path, and the remainder is the mock's photo-`z` kernel.  The
   corrected statement: **the event channel is calibrated; what remains is a
   property of the mock's catalog, and the production catalog is DESI
   spectroscopy** (`sigma_kde = 0.003` against the mock's 0.023).
5. **"Nothing here licenses or blocks the production run" -- STANDS.**  Unchanged,
   and now better supported: the production `fp` median moved 71.70 -> **71.54**
   under the fix (0.16 km/s, 0.04 sigma), because production's compact view holds
   49,143 of 49,152 rows so the defect barely fired there.

**Net verdict on PR-6a: the case against it is weaker than `CLOSURE_v2` recorded,
and the case for the interval caveat is much weaker.**  `sqrt(J/H)` was proposed
there as a sandwich correction of 2.35-2.46; the honest post-fix number at fixed
catalog is **1.13**.  Tiers B and C still do not PASS as written, but their
failures are now attributed: a fixed code defect, and a mock catalog whose
photo-`z` the production line does not share.

## What a fourth pass would have to do

Not more diagnosis of this mock.  Either

* **rebuild the mock with spectroscopic redshifts** (`SIGMA_Z_CAT = 0.003`, DESI's
  own value) and re-run the tiers, which is the configuration the production line
  actually validates -- roughly a day of GPU; or
* **accept that the tiers as written measure the mock's photo-`z` treatment** and
  re-scope them, which is a decision rather than a computation.

Both are owner calls.  What should NOT happen is the production `J` ensemble: its
whole purpose was to measure an identity violation that the fix removed
(`J_ens/H` 5.50 -> 1.27).

## Files

Post-fix artifacts, all in this directory: `fix_tier_c_n24.json`,
`fix_tier_c_n50.json`, `fix_tier_c_deltape.json`, `fix_variance_split.json`,
`fix_varsplit_7001.json`, `fix_nobs_scaling.json`, `fix_tier_b.json`,
`fix_R_fp.json`, `fix_add_{fp,unitfp,nofp}.json`, `gate_complete.json`,
`gate_specz.json` (the no-op, kept as the record), `gate_specz_v2.json`,
`beta_consistency.json` (the wrong weight, kept) and `beta_consistency_v2.json`.

New tools: `additivity.py` (the test that found the bug), `capture_check.py`,
`event_reshuffle.py`, `loo_coupling.py`, `fp_row_alignment.py`, `row_terms.py`,
`ll_multiset.py`, `gate_complete.py`, `gate_specz.py`, `beta_consistency.py`.

The full narrative, in the order it was found, is `DIAGNOSIS.md`.
