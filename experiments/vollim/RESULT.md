# The volume-limited LOA+LS dark-siren analysis (z < 0.30): it selects one event

**The construction is complete and it does not measure `H0`.**  Requiring each
event's full `V_90` to lie inside the LOA+LS footprint and inside `z < 0.30` for
the whole `H0` prior leaves **1 of 259 events**, and one event carries 0.157 nat
about `H0` against a flat prior on [20, 140] -- a 90% interval of
**[22.5, 126.6]**, which is 87% of the prior range.

That is the result, and the rest of this note is the numbers behind it, because
the two criteria fail for different reasons and only one of them is about depth.

## 1. The boundary

Containment must hold for every `H0` in [20, 140].  At fixed `dL` the inferred
redshift rises with `H0`, so `H0 = 140` is the binding end:

    dL(z = 0.30; H0 = 140, Om0 = 0.3089)  =   774.9 Mpc     <- the cut
    dL(z = 0.30; H0 = 67.74)              =  1601.4 Mpc

The conservative choice is a factor **8.8 in volume** tighter than a
fiducial-cosmology cut.  It is not optional: a cut placed at the fiducial
distance would make the event list a function of the parameter being measured.

## 2. What each criterion costs

`V_90` is the sample-based 90% highest-density region on an (nside-32 cell) x
(dL bin) grid; a cell counts as inside the footprint only if all of its
nside-128 children clear `f_p >= 0.5`, so boundary cells are outside.

| criterion | events passing (of 259) |
|---|---|
| radial (`V_90` entirely inside 774.9 Mpc) | **12** |
| sky (`V_90` entirely inside the footprint) | **2** |
| both | **1** |

**The sky criterion is what kills the sample, not the depth.**  LOA+LS covers
26,702 deg^2 -- 65% of the sky -- yet only 2 events have a `V_90` that fits
inside it, because the median BBH localization is 1,645 deg^2 and the footprint's
boundary is long.  Of the 12 radially contained events, 8 have more than
three quarters of their `V_90` cells covered and still fail: they straddle an
edge.

The one survivor is **GW240413_022019**: `V_90` = 64 deg^2 spanning
243-733 Mpc, every cell covered, 42 Mpc of radial margin.

Loosening the coverage requirement does not rescue it -- the list is 4 events at
`f_p > 0`, 2 at `f_p >= 0.25`, 1 at `f_p >= 0.5`, and 0 at `f_p >= 0.75`.  Going
the other way, `f_p > 0` means counting a pixel that is 99% masked as complete,
which is exactly the assumption this analysis is supposed to avoid.

## 3. The selection function, and why it cannot be validated here

The likelihood's selection term must integrate the same two-stage criterion,
detection AND containment.  Injections carry true parameters but no posteriors,
so containment needs a proxy -- and the first thing the data say is that a
DETERMINISTIC proxy is impossible: `V_90` area correlates with `dL` at only
0.45, so at fixed (`dL`, position) containment is essentially a draw from the
localization-quality distribution.  A `q90` threshold proxy selects 0 of 259.

The proxy that is right for this purpose is a **probability**, since `beta`
needs `E[1_contained | theta]` and not a per-injection verdict.  For an
injection at `dL`, each of the 259 real events' (extent ratio, `V_90` radius)
pairs is rescaled to that distance by the fitted slopes and the contained
fraction is counted.  Applied back to the real events it gives:

* expected contained **0.22** against an observed **1**;
* the selected event sits at the **99.2nd percentile** of `P` -- so the proxy
  does rank the right event at the top;
* `P > 0` for 7 of 259 events, max `P` = 0.062.

**The validation is consistent but has almost no power.**  One observed event
against 0.22 expected is an ordinary Poisson outcome (`P(>=1) = 20%`), so this
constrains the selection function to a factor of a few and no better.  Everything
downstream inherits that.

Folded into the injection set (`pdraw / P`, `Ndraw` untouched, `P = 0` dropped
-- exact given `P`), the containment stage keeps an **effective 642 of 1,067,946
injections, a factor 1,663 reduction**, and:

    selection Neff = 3.6      against Vitale's 5 N_obs floor of 5.0

**The selection integral fails its own reliability criterion**, at `N_obs = 1`,
where that floor is as forgiving as it ever gets.

## 4. The posterior

`universe_model = dark_sirens_complete`: no missing-galaxy branch, no `C(z)`, no
luminosity function, no `Q`.  Inside the selected volume the catalog IS the
redshift prior.

| arm | `H0` median | 68% | 90% | KL from flat |
|---|---|---|---|---|
| complete (catalog) | 51.97 | [30.0, 100.9] | [22.5, 126.6] | **0.157 nat** |
| spectral (no catalog) | 59.92 | [30.3, 107.3] | [23.0, 128.8] | 0.080 nat |

Both rail to the low prior edge (`MAP` = 20).  The catalog adds **0.077 nat**
over the no-catalog control, and the catalog-vs-spectral log-ratio varies by
0.69 nat across the whole prior.  With one event there is no combined posterior
distinct from the single-event one, and no per-event comparison to make.

## 5. What this does and does not license

**It does not say the volume-limited idea is wrong.**  It says this
(survey, event set) pair cannot support it: BBH sky localizations are two to
three orders of magnitude larger than the footprint's boundary tolerance, and
the conservative `H0`-independent boundary removes 8.8x the volume a fiducial cut
would.  A survey covering the full sky, or events localized to a few deg^2,
would change the arithmetic entirely.

**It does not compare against the completeness-corrected line.**  That line uses
all 259 events and reports `H0` = 71.7-80.6 depending on the arm.  The
volume-limited result is not evidence for or against those numbers -- with 0.157
nat it is consistent with every value in the prior.

**The honest summary of the trade**: dropping the completeness model removes a
modelling assumption whose effect is measurable (the mask alone moves the
259-event median by 2.5 sigma) and pays for it with 258 of 259 events. On this
survey that is not a favourable exchange.

## Reproduce

    python select_events.py      # the event list + containment diagnostics
    python proxy.py              # the containment proxy + its validation
    python build_injections.py   # pdraw / P, and the Neff verdict
    python run_h0.py             # both arms

Artifacts: `data/selected_events.json` (per-event containment table),
`data/proxy.json`, `data/injections_contained.h5`, `data/h0/h0_vollim.json`.
