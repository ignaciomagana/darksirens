# PR-7 — multitracer K>=2, and the seam closed by deletion (2026-08-17)

Module: `darksirens/redshift/latent_counts.py` (`MultiTracerCountOperator`
stacking K tracers over ONE `xi`; per-tracer `b_k`, completeness, counts and
multinomial; per-tracer moment tables; per-tracer columns of `S`).  Pins in
`tests/test_latent_multitracer.py` and `tests/test_latent_multitracer_cli.py`.

## The campaign (20 realizations, nside 16, b_2/b_1 = 2)

Two DISJOINT tracers (OWNER DECISION 9) over one shared latent field, 180,000
galaxies each, 12 shells to `z_depth = 0.30`, rank `M = 40 x 8 = 320`.  Every
realization is fit twice on the SAME mock: once with the shared-`xi` likelihood
this PR ships, and once with an artificially DECOUPLED two-field variant.

## Tier E gates

| gate | value | threshold | verdict |
|---|---|---|---|
| **(i)** bias ratio within 2 sigma | **20/20**, pull +0.196 +- 0.724, median `r = 2.00156` (truth 2) | 20 realizations | **pass** |
| **(ii)** shared-`xi` tighter than two independent fits | `sigma` **0.00836 vs 1.07984** -> **129x tighter**, tighter in **20/20** | qualitative | **pass** |
| **(iii')** decoupled variant differs in the predicted direction | decoupled median `r = 1.456` against truth 2; shared recovers 2.0016 | v4 restatement | **pass** |
| convergence | 20/20 shared solves converged | | pass |
| shared/decoupled cross-tracer correlation | **0.999944** vs **0.000000** | structural | pass |

**(iii') is the gate that matters, and it is the v4 version.**  v3's gate --
"runs without `--allow_unverified_shared_lss_members`" -- was demoted by the
plan itself (§0.5 finding 12) to a statement of fact, because that flag and its
check live on the table-loader path (`inference/loaders.py:352-395`) that latent
mode DELETES: it would pass by deletion of the check rather than by
satisfaction of the property, which is structurally the same routing tautology
review caught in rev 1.  The substantive replacement is the decoupled-variant
comparison above, and it is what differentiator 2 rests on: the decoupled fit
is not merely wider, it is **biased to 1.456 against a truth of 2** while
carrying `sigma = 1.08`, i.e. it is simultaneously wrong and uninformative.
The shared field is what makes the bias ratio measurable at all.

## R14 — the overlap arm, and why OWNER DECISION 9 is load-bearing

OD9 specifies DISJOINT tracers.  That is not a convenience; it is a modelling
requirement, and the failure is fast:

| overlap phi | mean `r` | quoted `sigma_r` | pull | within 2 sigma |
|---|---|---|---|---|
| **0.00** | 1.99877 | 0.00861 | **-0.11 +- 0.69** | **8/8** |
| 0.05 | 1.91670 | 0.00813 | -10.51 +- 2.50 | **0/8** |
| 0.10 | 1.84065 | 0.00772 | -21.24 +- 5.12 | **0/8** |
| 0.25 | 1.65645 | 0.00684 | -51.61 +- 10.86 | **0/8** |
| 0.50 | 1.40941 | 0.00588 | -103.06 +- 19.66 | **0/8** |

**Five percent shared membership already breaks the bias-ratio recovery at
10 sigma**, and the quoted `sigma_r` does not grow to absorb it -- it SHRINKS,
because the estimator reads duplicated galaxies as extra independent
information.  So an overlapping pair does not merely lose accuracy; it becomes
confidently wrong, which is the more dangerous failure.  Any K>=2 application
must therefore establish disjointness as a property of the catalogs, not assume
it, and a future rung admitting overlap has to model the shared membership
rather than tolerate it.

## What this rung retires

`realization_set_id` ceases to have a referent.  One `xi` shared across K
tracers makes "member m of every catalog is the same realization" a **theorem**
rather than a stamped assertion to be verified (PLAN §4.4), so the three-way
compatibility matrix of joint builder x `c_mode` x K collapses to nothing.
That is differentiator 2 delivered structurally: on the table path the
configuration is not constructible at all (§0.4 -- the joint builder is
`per_pixel`-only, has no budget gauge fixing, and cannot resolve any physically
supportable correlation length), and the only way to run it is the flag that
marginalizes over an independent-fields product prior instead of the shared
field the estimator assumes.

## Scope

Mock scale, nside 16.  **No production 259-event H0 run was launched** -- it
remains held for the owner.  K=1 is bit-identical through every change here
(pinned); the P12 goldens stay 23/23 bit-exact.
