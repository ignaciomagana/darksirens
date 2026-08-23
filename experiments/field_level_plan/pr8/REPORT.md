# PR-8 — the `amp(z)` support, and what an assumption above `z_depth` costs `H0`

*Field-level ladder, rung 8. **Mock scale: nside 16, one realization
(`pr6a/data/rb`, seed 7001, 60 events, `DARKSIRENS_ZMAX = 1.5`).** The 259-event
production posterior is HELD by the owner and was not run.*

---

## The answer, first

**This table is an `H0` SHIFT UNDER AN ASSUMED `amp(z)`. It is not a measurement of
`amp(z)`, and no row of it is an LSS-marginalized posterior.** There are no counts above
`z_depth` — the fitted field's in-support fraction of the missing budget is 6e-5 (R1), and
on this mock 99.44% of the budget sits above the depth — so the width of every row is a
pure function of the number chosen for that row. Marginalizing over the rows would be
quoting a prior as a measurement (PLAN OWNER DECISION 7). What the table does is bound the
systematic PLAN §4.2 states and declines to fix: `Q == 1` above the depth assigns ZERO
variance where there is no data instead of the prior variance.

| assumed `amp(z > 0.30)` | `H0` median | 90% CI | 90% width | Δ median vs `amp = 0` | Δ width vs `amp = 0` |
|---|---|---|---|---|---|
| **0** (the shipped convention) | 80.006 | [69.700, 92.002] | 22.302 | — | — |
| **0.05** | 80.007 | [69.701, 91.996] | 22.295 | +0.0009 | −0.0073 |
| **0.10** | 80.008 | [69.702, 91.990] | 22.288 | +0.0018 | −0.0142 |
| **0.20** | 80.010 | [69.704, 91.978] | 22.274 | +0.0038 | −0.0281 |
| **0.40** | 80.015 | [69.708, 91.956] | 22.248 | +0.0087 | −0.0547 |
| 0.20, growth-shaped `amp_hi D(z)/D(z_depth)` | 80.010 | [69.704, 91.978] | 22.274 | +0.0037 | −0.0280 |
| *(reference)* no latent field at all | 80.950 | [70.354, 93.364] | 23.010 | +0.944 | +0.707 |

All numbers in km s⁻¹ Mpc⁻¹, flat `H0` prior on [20, 140], 121 grid nodes, `M_draw = 8`
members marginalized. Source: `scan_amp.json` (job 1136201, TWIG-GPU, 342 ms/eval).

**Read as a bounded systematic:** over the whole scanned range the assumption moves the
median by **< 0.01 km s⁻¹ Mpc⁻¹ (1.1e-4 of the median)** and the 90% width by **< 0.06
(2.5e-3 of the width)**. For scale, turning the *fitted* field on at all — the `amp = 0`
row against no field — moves the median by 0.94 and the width by 0.71. So on this
realization the assumption about 99.4% of the missing budget is worth about **1%** of what
the 0.6% of it that the counts actually constrain is worth. Both signs are also worth
naming: `amp > 0` moves `H0` UP and makes the posterior NARROWER, monotonically in
`amp_hi`, and the dependence is linear in `amp_hi` to the precision quoted (Δ median
0.0009 → 0.0087 across a factor 8 in `amp_hi`).

---

## §1 What `amp(z)` is, and where it is allowed to be anything but 1

PLAN §4.3 is the constraint the whole rung is built around: `(b, xi)` and `(amp, xi)` enter
the model only through `b*xi` and `b*amp`, so wherever the counts constrain the field there
is exactly ONE clustering amplitude and it is `b_gal`. A profile that touched the fitted
region would re-open that degeneracy. So `amp(z)` is pinned at **the literal float 1.0** for
every `z <= z_depth` — not "1 to double precision", the literal value, because
`x * 1.0 == x` in IEEE754 and that is what makes the fitted region *untouched* rather than
perturbed at the last bit (`tests/test_latent_amp.py::test_amp_is_bit_one_at_and_below_the_depth`).

Above the depth there are no counts at all, so `amp_hi` is an ASSUMPTION and the rung is a
scan. Two shapes ship: `step` (`amp_hi` everywhere above the depth — what the table quotes)
and `growth` (`amp_hi * D(z)/D(z_depth)`, the same assumed number carried by linear growth,
which decays and is therefore the conservative shape at fixed `amp_hi`).

**Where the profile is applied is the design decision.** `amp(z)` scales the REDSHIFT
FACTOR ROWS, `phi_z -> amp(z) phi_z`, which scales the field
`f(p, z) = row_fac[p] . phi_z[z]` by `amp(z)` exactly, at every pixel, with no change to
the node geometry, the kernels, or their Cholesky factors. The scalar `amp` argument that
`build_latent_basis` already had scales `K_sph` instead, and would NOT be bit-inert at
`amp = 1` because the per-factor jitter is absolute and does not scale with it.

**The support had to move with it.** Nodes stopped at `z_node_hi = z_depth`, and the RBF
rows decay to nothing within ~2 `ls_z` of the last node, so multiplying an assumed
amplitude into them would have scaled a field that is already zero. The scan's anchors put
`M_z = 11` nodes out to `z_node_hi = 1.5` (the top of the run's grid) against the pr6a
anchor's `M_z = 5` to `z = 0.30` — rank 704 against 320. Above the depth those extra modes
are unconstrained by construction, so the members carry the PRIOR there, which is exactly
the variance §4.2 says the `Q == 1` convention throws away.

---

## §2 The gates

**G1 — `amp(z)` reduces to a constant at the legacy value BIT-IDENTICALLY.** Pinned at
three levels, in `tests/test_latent_amp.py` (20 tests, all green):

* the profile returns the literal `1.0` on every below-depth node, for every `amp_hi` and
  both shapes;
* `amp_hi = 1` (a constant-1 profile over the whole grid) produces a basis whose
  `phi_sph`, `phi_z_out`, `phi_z_fine`, `proj_sph`, `L_sph`, `L_z` are `np.array_equal` to
  the no-profile basis;
* an `amp_hi = 0` artifact and a pre-PR-8 artifact of the SAME geometry load to plans whose
  `phi_z`, `below_depth`, `A`, `B` and `row_fac` are equal array-for-array.

**G1e2e — the same statement on the full likelihood, and its honest residual.** The scan's
`amp = 0` row and the `noprofile` control agree to `max |Δ log L| = 5.68e-14` over the 121
`H0` nodes and `Δ median = 2.8e-14`. It is NOT bit-zero, and the cause is not the amp
machinery: the two artifacts carry different numbers of consumption rows (1000 vs 287), and
the whitening triangular solve that builds `phi_z` is batched over those rows, so the
below-depth basis itself differs in the last bit — measured directly at `max rel 1.1e-15`
on `phi_z` below the depth. At FIXED row count the machinery is bit-identical, which is
what G1's third pin says. Quoting "bit-identical" for the e2e pair would have been quoting
someone else's ULP.

**G2 — K = 1 goldens.** `JAX_PLATFORMS=cpu DARKSIRENS_GOLDEN_EXACT=1 pytest
tests/test_unified_k1_golden.py -q` → **23 passed**, bit-exact, default `zMax`.

**G3 — table mode and pre-PR-8 latent mode are inert.** The 228-test latent suite
(`test_latent_{seam,seam_e2e,factory,cli,block_sizing,p11,p13,p17,guards,b_gal_dispersion,
solve_damping,field,counts,anchor}`) is green unchanged. The mechanism is a static
pytree-STRUCTURE branch: `EMCatalog.latent_support` is installed only by an anchor with
`amp_hi > 0`; on every table-mode run, every pre-PR-8 anchor, and every `amp_hi = 0` anchor
it stays `None` and the consumers compute `zgrid <= z_depth` exactly as they did.

**G4 — eq. (4) above the depth.** With the support extended, `rho` must be applied wherever
the field is nonzero or the seam injects an un-normalized monopole over 99.4% of the
missing budget — a change in the TOTAL missing count masquerading as placement. Pinned on
the real `sky_moments` of a real amp basis: `|sum_{p in F} Q_p / P_F - 1| < 1e-12` at every
`z` above the depth, at `C = 0` (which is the completeness the consumption uses there).

---

## §3 What was built, and why one solve serves every row

`amp` never enters the count channel (it is 1 below the depth, and the count operator's
fine grid stops at the depth), so the operator, the MAP, the Hessian, the sensitivities and
the Laplace draws are amp-INDEPENDENT. `build_anchor_amp.solve_once` does them once; each
row of the scan re-derives only the two objects the assumption touches — `phi_z` above the
depth and the eq. (2) moments `(A, B)`. Any spread across the rows is therefore the
assumption and cannot be a re-solve, a re-draw or a seed.

The one solve: rank 704, `grad_inf = 6.35e-10` (P6 passes at 1e-8), 1854 footprint pixels,
183,267 catalogued galaxies, 12 equal-comoving-volume shells, `s_b = 5.00e-02` with the
5% systematics floor binding, `M_draw = 8` members drawn from `H^-1 + s_b^2 v v^T` (S-2's
shipped covariance).

Injected modulation, measured on the anchors (`amp_diagnostics.json`) — the sd of the field
over footprint pixels and members at `z = 0.4`, which is linear in `amp_hi` as it must be:

| `amp_hi` | 0 | 0.05 | 0.10 | 0.20 | 0.40 |
|---|---|---|---|---|---|
| sd `f(p, z=0.4)` | 0.0 | 0.047 | 0.095 | 0.189 | 0.379 |
| member-to-member sd at fixed pixel | 0.0 | 0.031 | 0.061 | 0.122 | 0.244 |

For comparison the fitted field's sd at `z = 0.2` is 1.16, and PLAN §4.2's "factor of `e`"
statement is the `amp = 1` case.

---

## §4 Why the table moves the way it does

An assumed modulation above the depth can only reach the posterior through events whose
redshift prior reaches above the depth. On this realization (`amp_diagnostics.json`):

| | `H0 = 20` | `H0 = 67.74` | `H0 = 140` |
|---|---|---|---|
| fraction of PE samples above `z_depth` | 0.000 | 2.6e-4 | 0.347 |

So the assumption acts on the HIGH-`H0` end of the scan and essentially nowhere else, and
that is exactly what the likelihood shows: `Δ log L` against the `amp = 0` arm peaks at
`H0 = 138` (0.036, 0.080, 0.168, 0.362 for `amp_hi` = 0.05, 0.1, 0.2, 0.4) and is
`<= 5.9e-4` at `H0 = 68`. A one-sided penalty at the top of the range trims the upper tail:
hence a NARROWER 90% interval and a median that drifts slightly up as the lower tail
re-weights. The effect is a placement effect only — the survey-global budget is unchanged
by construction (eq. (4)), which §4.2 calls the gauge fixing "C and n0 own the budget, Q
owns placement".

---

## §5 What this does NOT say

1. **No row is "the right one".** The mock's above-depth galaxies are not clustered by this
   profile at any amplitude, so the scan measures the response to an injected assumption,
   not a recovery. Tier D is the only tier that tests misspecification and no tier can test
   the `z > z_depth` extrapolation at all (PLAN §6.1's stated limitation, R1-SEV2-9).
2. **The absolute `H0` of every row is off, identically.** `H0_true = 67.74` sits at
   `cdf = 0.018-0.024` in every arm. That is the Tier-B/C dispersion defect CLOSURE_v2 §V
   localizes to the EVENT draw at fixed catalog (82-92% of the H0 scatter, present in the
   no-field control at ×2.25) — a property of this mock's PE calibration, not of the latent
   design. The scan is DIFFERENTIAL across rows that share the defect exactly, which is why
   it is still informative.
3. **The numbers are mock scale.** nside 16, 60 events, `z_depth = 0.30`, grid to
   `z = 1.5`, 99.44% of the missing budget above the depth against R1's 99.994% at DESI
   scale. The production 259-event run is HELD and was not launched. Nothing here should be
   scaled to it by hand: the DESI-scale line has ~4.3× the events, a different footprint and
   a grid reaching `z = 6`, and the fraction of PE support above the depth — the quantity
   §4 shows the effect is carried by — is the thing that would change most.
4. **`amp_hi <= 0.4` is where the scan was run.** `amp = 1` (full prior variance above the
   depth) is not in the table; §3's linearity would extrapolate the median shift to ~0.02
   and the width to ~−0.14, but extrapolating a systematic is not measuring it.

---

## §6 Files and reproduction

```
experiments/field_level_plan/pr8/
  build_anchor_amp.py     one solve -> one anchor per assumed amp(z)   (~50 s, CPU)
  scan_amp.py             the 8 arms + the amp0-vs-noprofile gate
  amp_diagnostics.py      budget fraction, injected modulation, PE support
  sbatch_scan.sh          the GPU job (TWIG-GPU, 1136201)
  anchors/                7 artifacts + manifest.json  (gitignored, 20 MB each)
  scan_amp.json           every arm's full log-likelihood curve
  amp_diagnostics.json
```

```bash
cd experiments/field_level_plan/pr8
python build_anchor_amp.py --legacy-geometry --growth 0.2   # anchors/ + manifest.json
sbatch sbatch_scan.sh                                       # scan_amp.json
python amp_diagnostics.py                                   # amp_diagnostics.json
```

Shipped surface (all additive; `None`/absent is the pre-PR-8 path line for line):

* `redshift/latent_field.py` — `amp_profile`, `growth_factor`, and
  `build_latent_basis(..., amp_hi, amp_kind, amp_z_depth, amp_Om0)`; the profile is
  recorded in `basis_meta` ONLY when applied.
* `likelihood/latent_q.py` — the loader rebuilds the profile from `basis_meta` (no stored
  array, so builder and seam cannot carry two different `amp(z)`), generalizes the depth
  guards, and returns the SUPPORT in `LatentQPlan.below_depth`. It also exposes
  `z_fit_depth` (the last shell edge).
* `core/types.py`, `likelihood/core.py`, `likelihood/factory.py`,
  `redshift/completion.py`, `redshift/prior.py` — `EMCatalog.latent_support`, one accessor
  (`completion.latent_support_mask`), `rho` normalized against the consumed `C` (`C := 0`
  above the depth), and the Q-side depth relaxation dropped where the seam has a modelled
  value to say.
* `likelihood/factory.py` — the isotropy guard now takes the FITTED depth from
  `amp_z_depth`, else the last shell edge, else the node range (identical on every pre-PR-8
  artifact); the amp profile joins guard 1's content digest when present.
* `cli/build_latent_field.py` — `--z-node-hi`, `--amp-hi`, `--amp-kind`, four refusals
  around them, and the radial resolution guard moved onto the node range.

## §7 Tests

`tests/test_latent_amp.py` — 20 tests: the profile's below-depth bit-identity, the
`amp = 1` basis identity, the `amp = 0` loader identity, support extension and row scaling,
the loader's two refusals, the support leaf's structure branch, the isotropy guard's fitted
depth (both sources), the resolution guard on the node range, digest separation, and eq. (4)
above the depth.

Green on this branch: `test_latent_amp` (20), the 228-test latent suite, and
`test_unified_k1_golden` at 23/23 bit-exact.
