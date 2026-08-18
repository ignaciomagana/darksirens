# The production 259-event H0 scan, with the latent field (2026-08-18)

Run on a dedicated Jetstream H100 80 GB under the validated numerics stack
(jax 0.4.34 / numpy 1.26.4 / scipy 1.12.0 — see below).  259 GWTC-5 BBH events
x 4096 samples, 1,067,946 detected injections (Ndraw 9.44e8), the real DESI
union catalog (30,470 occupied nside-64 pixels, 22,787,566 galaxies), fixed
theta, 121 H0 nodes over [20, 140], and the **clean guard arm**
(`selection_neff_soft_guard=False`, `max_likelihood_variance=1e6`) — not the
shipped scan convention, for the reason in §4.

## The result

| arm | H0 median | 90% CI | width | finite | ms/eval |
|---|---|---|---|---|---|
| `nofp` — no `f_p`, no field | 90.25 | [82.95, 96.47] | 13.52 | **116**/121 | 1702 |
| `fp` — `f_p` on, no field | 71.70 | [64.97, 79.12] | 14.15 | 121/121 | 3215 |
| **`latent`** — `f_p` on, **field on**, 8 members | **71.95** | **[65.18, 79.38]** | 14.20 | 121/121 | 3265 |

## The decomposition, which is the point

| channel | shift | in sigma | width ratio |
|---|---|---|---|
| **`f_p` (PR-2)** | **-18.55** | **-4.31** | 1.046 |
| **the FIELD (PR-5/PR-6a)** | **+0.25** | **+0.06** | 1.004 |
| both | -18.30 | -4.24 | 1.051 |

**1.34% of the total shift is the field.**  The ladder's deliverable moves the
production H0 posterior by **0.06 sigma**; the per-pixel completeness map of
PR-2 moves it by **4.31 sigma**.

This must be read against `fp`, never against `nofp`.  Comparing the field
against the no-`f_p` baseline would credit it with PR-2's effect, and PR-6a
already measured the same conflation in the mocks (91% of its Tier-B shift,
97.2% of its runtime overhead) — this is that finding at production scale.

## What it means, stated plainly

**The field is a correctness result, not a signal result**, and every earlier
rung predicted exactly this:

* PR-0 measured `sigma_H` tiny — the field's GW content is its MAP/mean
  placement, not its ensemble spread.
* PR-3's K9 fired its BENIGN branch at 5.4e-4 nat, 300x under gate: the field
  barely responds to cosmology.
* PR-5b measured the member spread at 1.49e-2 nats at the anchor.
* PR-6a measured 91% of the mock shift and 97.2% of the cost as `f_p`.

A 0.06-sigma shift is the honest headline for the field itself.  What the
ladder buys is that the shift is now *computed correctly* — the missing-galaxy
budget is conserved exactly (P13 closes at 1e-15), the realisations are
marginalised through both the event terms and the selection normalisation, and
the 1.06 GB table is replaced by 7.8 MB of coefficients that can be rebuilt in
41 s.  It also buys the multitracer construction (PR-7), which is not
constructible at all on the table path.

Two corroborating details:

* **The field WIDENS the interval** (14.15 -> 14.20, 1.004x).  That is the
  correct direction for a marginalisation — it adds uncertainty rather than
  removing it — and it is the first time Tier B's "latent CI >= table CI" gate
  has been evaluated on a configuration where it is not vacuous.
* **`f_p` also repairs the support.**  `nofp` returns `-inf` at 5 of 121 nodes;
  both `f_p` arms are finite at all 121.

## `f_p` at production scale, and S-3

The -18.55 (-4.31 sigma) shift is the production-scale measurement of the
defect PR-6a logged as **S-3**: under `c_mode=selection`, off-footprint pixels
are modelled as `C_bar`-complete unless `f_p` supplies the zero.  The mock
campaign saw it rail H0 to 125-138 in 16 of 16 footprint runs; production is
less extreme but the direction and the size are unambiguous — **an 18.5-unit,
4.3-sigma bias in H0 from a modelling choice about pixels the survey never
observed.**

Because `inference/loaders.py:1021` refuses `f_p` alongside a Q table, every
Q-table configuration on a footprint-limited survey is exposed to this,
**including the shipped `selq_radial`**.  That is an owner-level decision and
it is not made here.

## The guard convention, and why it is not a detail

Quoted on PR-0's clean arm.  PR-5b measured that under the SHIPPED soft guard
the member spread reads 24-40 nats against the clean arm's 1.5e-2 — 610x to
34,206x larger and non-monotone in H0 — because the guard's wall responds to
member-dependent `Neff` rather than to the field.  A latent-vs-table comparison
run in that convention measures the guard.  The shipped scans' own 139 rail was
shaped by the same wall.

## Provenance

* anchor `runs/anchor/latent_anchor_h100.h5`, sha256
  `c9ac8dcfe8afe5e52639df4d8ffffdf28daad95064fcf103abf8a069e51f31fb`,
  `grad_inf = 9.82e-11`, 30,470 pixels / 22,787,566 galaxies / 12 shells,
  `s_b = 5.000e-2` (5% floor binding against a profile curvature of 6.35e-3),
  rank-1 `b_gal` inflation ON — member spread x1.0104 overall, **x8.17 along
  `v`**.  This anchor therefore carries the S-2 fix that the PSC `v2a` anchor
  predates, which is why its sha differs.
* Numerics: **jax 0.4.34**, pinned deliberately.  The machine was first given
  jax 0.10.2, under which
  `test_seam_is_bit_identical_to_the_table_evaluator` FAILS by one ulp
  (1.5e-16) because XLA re-associates the reduction differently.  Every PSC
  number was measured on 0.4.34 and the bit-identity pins are the ladder's core
  invariants, so a different stack makes the results incomparable.  Under the
  pinned stack the port is faithful: 275 passed, P12 goldens 23/23 bit-exact,
  marginalization regression 16/16.
* This is a GRID SCAN at fixed theta, which is what the 259-event line has
  always quoted.  It is **not** a sampled-theta posterior; that would be a
  ~1e6-call nested-sampling run, ~32 days on this GPU at the measured
  3265 ms/eval, and is not attempted here.
