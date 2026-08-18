# PR-10's compaction experiment: measured, and not worth doing (2026-08-18)

PLAN section 7's PR-10 entry proposes a `z < z_depth` sample-compaction
experiment for the seam, with the premise:

> most GWTC-5 PE samples are above `z_depth`, so the gather may be largely
> skippable -- **measure, do not bank**

Measured on the production event and injection sets, `z_depth = 0.30`,
`Om0 = 0.3089`:

| `H0` | PE samples below depth | injections below depth |
|---|---|---|
| 20 | **89.90%** | **79.51%** |
| 40 | 62.56% | 46.35% |
| 67.74 (fid) | 40.20% | 24.92% |
| 90 | 27.65% | 17.07% |
| 120 | 16.63% | 11.47% |
| 140 | 12.31% | 9.25% |

**The premise is false at the low end of the prior.**  It is true at high `H0`
-- at 140 only 12% of PE samples are in support -- and only barely true at the
fiducial (40% below, so "most above" holds by 60/40).  At `H0 = 20`, **90% of
PE samples and 80% of injections are BELOW the depth**, i.e. in support, and
there is essentially nothing to skip.

That is the number that governs, because **the compaction has to be sized at
the worst `H0` in the prior**: the gather's shapes are static under JIT, so a
compacted buffer must hold the largest in-support count the scan will ever
produce, not the fiducial one.  At `H0 = 20` that leaves **20.5% of the
injection gather skippable**.

## Verdict: do not build it

The saving is 20.5% of the gather, in the worst case; the gather is one part of
the seam; and PR-6a measured the whole seam plus its 8-member ensemble at
**+1.39%** of the `f_p` baseline.  So the ceiling on this optimisation is a few
tenths of one percent of the likelihood, in exchange for a compaction index
that has to be rebuilt per proposal and a static buffer sized for the prior's
worst corner.  **Measured, and declined** -- which is exactly what "measure, do
not bank" asks for.

If the sampled-theta nested-sampling run is ever attempted (~1e6 calls, ~32
days on this H100 at 3265 ms/eval), the lever that matters is not this one.

## A consistency check that falls out

At the fiducial, 40.20% of PE samples are below the depth, and the DESI
footprint is 23,048 of 41,253 deg^2 = 55.9% of the sky.  Their product is
**22.5%**, against PR-0's independently measured in-support fraction of
**24.6%** -- the two agree to ~9% without being tuned to each other, which is
the expected relationship if in-support means "below the depth AND inside the
footprint".
