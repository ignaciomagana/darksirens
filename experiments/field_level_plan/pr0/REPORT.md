# PR-0 — the decision rung: measurements and K0 evaluation (2026-08-13)

Base: darksirens master `8592b02` (post-review-campaign merge; the plan's
anchors were verified on `0c5b3db` — the soft-guard arms below reproduce the
2026-08-10 scan artifact node-for-node, so the pull did not move this line's
likelihood). Event/injection pair: the blessed Aug-10 gwcat-1.0 products
(confirmed current by the gwcat session 2026-08-13; carries the known inert
~0.28% endo3 pdraw caveat, a deliberate-continuity choice, not a defect).

## Item 2 — `sum_i phi_i` and the closed-form member spread

**`sum_i phi_i ~ 1.0-1.5`**, measured two independent ways:
- **0.993** through the production likelihood itself (`phi_results.json`):
  a killer Q table (`logQ = -60` on occupied-footprint pixels at `z <= 0.30`)
  deletes the in-support missing branch from both the event terms and the
  selection integral; with the selection shift removed via captured `log mu`
  (`Delta log mu = 2.02e-3`), the event-term delta gives
  `sum_i -log(1 - phi_i) = 0.993` — the exact code-path number.
- **1.47** from the sample-wise construction (uniform PE weights; the gap to
  0.993 is the population/prior reweighting the prediction skips).
24.6% of all PE samples are in-support; the mean in-support missing fraction
is ~2.3% (C_sel in [0.92, 1.00] over the footprint, clustering spikes
carrying the rest).

**K0 leg 1 does NOT fire** — the plan's own recorded expectation (that the
~1e-4 in-support budget fraction implies `sum phi < 1e-3`) conflated the
BUDGET share with the EVENT-WEIGHTED share. The missing budget in-support is
indeed tiny (below); the events, being localized exactly where the catalog
lives, still put O(1%) of per-event posterior mass on the missing branch.

**Predicted member spread (eq. 6): `sigma_H = 6.0e-4` nats**
(`||a||_2 = 0.156` Euclidean; the `H^{-1}` norm crushes it because the count
channel's Fisher at 22.8M galaxies has factor-eigenvalue products up to
~4e5 against the prior's identity). Consequences:
- P14/OD5: the marginalization-accuracy budget is trivially met; predicted
  `M_draw`-bias at `M_draw = 8` is `(e^{sigma^2}-1)/(2M) ~ 2e-8` nats.
  The 2.6-nat catastrophe scenario of PLAN §6.5 is excluded at the anchor.
- The Laplace ensemble members are nearly interchangeable in the GW
  likelihood: the field posterior is data-pinned, so LSS placement
  UNCERTAINTY (as opposed to placement itself) carries almost nothing.
Caveats: uniform PE-sample weights on the event side; unconditioned Poisson
Fisher (conservative — the multinomial Fisher is smaller, but by factors,
not the ~5 orders needed to change the conclusion); prediction to be
confirmed at PR-5b per the plan.

## Reconciled in-support budget fractions (definitions matter)

Computed on the shipped `C_sel` (gaussian family, theta_hat), calibrated
budget (`log10n0 = -2.398`, `delta = 0.940`), anchor cosmology, ZMAX = 6:

| estimand | value |
|---|---|
| in-support missing (occupied, z<=0.3) / TOTAL missing budget (all-sky, z<=6) | **1.32e-5** |
| in-support+empty-sky missing (z<=0.3) / total missing budget | 3.80e-4 |
| in-support missing fraction of the SELECTION integral mu (population-weighted) | 2.57e-3 |
| event-weighted missing-branch posterior share, `sum_i phi_i / 259` | 5.7e-3 |

The earlier 6e-5 / 2.7e-4 / 1.3e-4 / 7.9e-5 figures are all budget-type
estimands under slightly different denominators; the decision-relevant pair is
the last two rows — the field's lever on the DATA is 2-3 orders larger than
the budget share, because both the events and the injections are
population-weighted toward exactly the volume the survey covers.

## Item 3 — Q-on-at-anchor oscillation, guard-decomposed

The 2026-08-10 artifact's raw `osc_H0 = 2.1e5` nats (sel vs selq_radial,
soft guard) is **entirely the soft-guard wall**, not placement:

- Under the production QC guard in its hard form, `logL = -inf` at EVERY H0
  node in [25, 140], both arms: `pe_variance_sum = 0.2733` inflates the
  required selection Neff to ~92k while the line delivers ~31-36k. **The
  fixed-population 259-event grid convention never clears the GWTC-4/5
  variance criterion anywhere in the prior.**
- The soft-guard wall at these Neff values contributes ~-1e6 nats
  (gate ~115, reward_mag ~4.4e3) and dwarfs the likelihood: the shipped
  scans' H0 = 139 rail is guard-shaped, not likelihood-shaped —
  strengthening data/h0_scans/DIAGNOSTIC_ONLY.md beyond its own wording.
- Clean arms (Vitale floor only, variance cap lifted): see
  `osc_item3_results.json` — the honest placement oscillation is the
  `selq_nogv - sel_nogv` delta over the finite nodes, quoted below.

**osc_H0 [ logL(Q on) - logL(Q ≡ 1) ], clean arms: 10.85 nats** over the full
prior (Delta from -13.4 at H0 = 30 to -2.56 at H0 = 85). This is the upper
bound on everything the ladder can buy, and it is 100x the K0 kill threshold.
Structure of the delta: strongly H0-dependent below ~70, flat (-2.6 to -2.9)
above — the constant part is an evidence offset (the table redistributes mass
away from where these events sit at fixed population), the H0-varying part is
what a sampler can feel. Within the clean posterior bulk (H0 ~ 75-105) the
oscillation is ~0.3 nat, and the posterior medians differ by 0.3 km/s/Mpc
(89.7 -> 89.4, ~6% of the 68% width) — a real but small placement pull at the
K2 posterior level, measured here at fixed population only.
Caveats carried with the numbers: (i) by the lifted criterion itself,
`sigma(lnL) ~ 1.5` nats of MC noise everywhere on this line, so the 0.3-nat
posterior-core structure is beneath the estimator noise floor while the
10.85-nat prior-wide swing is well above it; (ii) fixed delta-function
population (the DIAGNOSTIC_ONLY convention) — the joint-sampled construction
is the production estimand.
**Side result: the clean fixed-population posterior peaks at H0 = 90 +/- 5,
not at the 139 rail — the shipped scans' rail is guard-shaped** (see above).

## Item 1 — cost baseline, three columns

See `cost_baseline.json` (production 259-event config, H100, value-only
path) and `profile_member_marg_cpu.md` (M-scaling of the factored member
path, synthetic fixture):

| arm | warm median | vs no-LSS |
|---|---|---|
| (i) no-LSS production baseline | **3027.1 ms** | — |
| (ii) table mode, deterministic Q (`q_radial.h5`) | 3029.3 ms | +0.1% |
| (iii) table mode, 8-member marginalization | 3022.9 ms | −0.1% (noise) |

(H100, value-only path, x64, 20 warm evals across the H0 prior; compile
~14 s/arm. `M_draw` in {32, 64}: no production-scale table exists; the
synthetic factored-path scaling in `profile_member_marg_cpu.md` gives
value-path ratios 1.0 / 1.4 / 2.7 / 4.2 at M = 1/8/32/64 on the member seam
alone.)

**The decisive fact: the production baseline on THIS line is 3.0 s/eval, not
the 27.5-49.3 ms constant PLAN §2 argues from** (that constant belongs to a
different configuration; the NS run's 3.3 s/call independently corroborates
3 s here). Every cost conclusion in the plan built on percentages of 27.5 ms
deflates by ~110x: the member seam (+3.3 ms at M_draw = 8, +69 ms at 64) is
+0.1% / +2.3% of the real baseline, the projected ~1 ms rung-1
`row_fac_shift` correction is +0.03%, and even the refused ~1 s per-proposal
re-solve would be +33%, not 36x. OWNER DECISION 5's budget is trivially
satisfiable at any `M_draw <= 64`; K4 cannot fire on cost. (The refusal of
the per-proposal re-solve should be re-argued from K6/smoothness and memory,
which survive, not from wall-clock, which does not.)

## Item 4 — novelty search

`NOVELTY_SEARCH.md` (committed alongside): ~22 papers assessed.
- Differentiator 1 (budget-preserving completion-field marginalization):
  **partially anticipated in disjoint pieces** (Dalang+ 2410.03275:
  deterministic mean-one renorm; Boruah+ 2503.07974: realization
  marginalization, event terms only, complete catalog; Cheng & Gair
  2603.13053: analytic free-monopole marginalization). The combination —
  budget-constrained stochastic realizations through BOTH event terms and the
  selection normalization — **not found**.
- Differentiator 2 (shared multi-tracer completion realizations): **novel;
  no prior implementation found** (closest: Mukherjee+ 2107.12787, power
  spectrum only).
- The owner's headline claim survives with the multi-clause qualifier;
  unqualified "first field-level dark sirens" would not. Scoop risk: Cosmic
  Cartography III (GLADE+ application) — monitor until submission.

## K0 evaluation

K0 fires only if `sum_i phi_i < 1e-3` AND `osc_H0 < 0.1` nat.
**`sum_i phi_i = 0.99 >> 1e-3` and `osc_H0 = 10.85 >> 0.1` nat -> neither
leg fires. K0 does NOT fire. The ladder proceeds.**

Findings that reshape (but do not stop) the ladder:
1. The guard, not the likelihood, owns the current production H0 scans; any
   K2/K8-style posterior-level comparison must be run on a configuration that
   clears (or honestly lifts) the variance criterion, or it is vacuous — this
   is now a measured fact, not a warning.
2. `sigma_H ~ 6e-4` nats predicts PR-5b will find member marginalization
   nearly free and K5/OD5 unbinding; the interesting object is the FIELD
   (MAP/mean placement), not the ensemble spread, on this data.
3. The event-side lever (`sum phi ~ 1.5`) is orders above the budget-share
   argument; the field CAN move event terms. Whether it moves H0 is exactly
   what the ladder's K2/K9 gates measure downstream.
