# Field-Level Latent Upgrade — Implementation Plan (v4, third adversarial round resolved)

**Supersedes** v3. Internal references to "rev 1" and "rev 2" below refer to the two *pre-promotion*
revisions; "v2" is the promoted revision; "v3" resolved the second adversarial round; **"v4" is this
document**, which resolves the twenty-eight findings of the third adversarial round (**§0.5** — read
that subsection first; where any sentence below still reads as v3, §0.5 supersedes it). Companion
files in this directory: `OWNER_CONTEXT.md` (the owner's external novelty analysis and the promoted
goal), `reviews.md` (adversarial findings, dispositions, and the `v2 (owner context)` section
recording what this revision changed), `MODEL.tex` (the compiling mathematical specification —
theorem/proposition numbers cited below resolve there).

**Base:** `master` @ `0c5b3db` (NOT `feat/stratified-q-base` @ `6f8e6ae`; that base is 30 commits
stale and every line citation below has been re-verified against `0c5b3db`).

**Scope:** stop exporting `Q` as a file. Replace the fixed-fiducial `logq_map` HDF5 table with a
low-rank whitened latent field `xi` carried inside the inference, **coupled to the cosmological
proposal**, so that every proposal changes the galaxy-field likelihood consistently. Primary
deliverable: a working **K=1** field-level likelihood; every interface multitracer-shaped by
construction. The compressed *theta-free* member ensemble that rev 2 recommended is retained — as
the intermediate rung, the control arm, and the shippable fallback — not as the deliverable.

**Production state (measured, `experiments/desi_full259/logs/ns_joint_sel_1119811.out`):**
`--c_mode selection --catalog_sky_weighting field --use_lss false`, no `--lss_completion`; 20 free
dimensions; 259 events × 4096 samples; 1,067,946 detected injections; nside 64; `z_depth = 0.3`;
static state 10.390 GiB of 72.7 GiB free; **256 concurrent evaluations**
(`replacement_chain_schedule = [1,4,16,64,256]`, banner `Peak model: value-only (tinyns, 256
concurrent evals)`). The 259-event production likelihood today has **no LSS at all**. The
field-level path would be the first LSS in production, which is why the bar is "provably inert when
off, provably correct when on."

---

## 0. What changed, and why

Two drivers have now acted on this document. **Adversarial review** (rev 1 → rev 2) fixed the
engineering. **The owner's external analysis** (`OWNER_CONTEXT.md`, rev 2 → v2) changed the
*question the plan is answering*, and therefore the deliverable.

### 0.0 The promotion

Rev 2 asked: *what is the cheapest correct way to get a defensible LSS marginalization into the
259-event likelihood?* Its answer, stated honestly in its own §0, was a **theta-independent
compressed member ensemble**: condition the counts on shell totals, freeze the shell-response
operator `W` at an anchor `theta_ref`, and `p(xi|d_gal)` loses all theta dependence *by
construction*.

The owner asks a different question: *what makes this a methods paper that Cosmic Cartography,
variance completion and harmonic cross-correlation cannot subsume?* That answer requires
`p(xi | theta, d_gal)` — "**no Q built at fixed n0, fixed cosmology, or theta_hat_selection; every
cosmological proposal changes the galaxy-field likelihood consistently**"
(`OWNER_CONTEXT.md:150-161`).

So rev 2's PR-9 "escape hatch" is the owner's headline, and rev 2's recommended OWNER DECISION 1 is
the thing that deletes it. **This is not a disagreement to resolve; it is a promotion to execute.**
v2 executes it, with three structural consequences:

1. **The theta-coupled latent is the headline target** (§0.2 rung 1, §1.7, PR-6b).
2. **The compressed theta-free ensemble is re-scoped**, not deleted: it is the intermediate rung,
   the control arm PR-6b must agree with in the benign limit, and the shippable fallback (PR-6a).
3. **A full per-proposal re-solve remains infeasible** — §3.4's own arithmetic is ~13 Newton
   iterations at tens of ms = ~1 s of linear algebra against a measured 27.5–49.3 ms baseline, a
   **~36x wall**. Promotion is affordable *only* through **linear response** (§1.7), which this plan
   already implements for one parameter under the name `bias_sensitivity` (§3.4). Read the owner's
   "solve/update the conditional posterior at each proposal" literally and it is not viable; the
   "or importance-sample a small fixed set of whitened latent draws" clause in the same paragraph
   (`OWNER_CONTEXT.md:184-185`) is the viable reading, and §1.7 is what it becomes.

Two corrections to rev 2's own framing that the promotion forces:

* Rev 2's §0 claimed the design delivers "**no fixed-theta provenance firewall**". Under shell-total
  conditioning that is **false**: guard 1 (§4.4) stamps `anchor theta_ref, b_gal` into the artifact
  sha256, so the firewall is *replaced by a fingerprint*, not abolished. The owner's line becomes
  literally true only at rung 1 and above.
* Rev 2's §1.1 item 3 stated a premise without its consequence. Shell-total conditioning is
  *precisely* the operation that deletes `H0` from the field posterior. The owner's demand is
  therefore **unsatisfiable for `H0` under conditioning, and only partly satisfiable without it** —
  see §1.8, which is the single most important scientific caveat in this document.

### 0.1 What review changed (rev 1 → rev 2, retained)

Eleven SEV1 findings and sixteen SEV2/3 landed on rev 1. Five of them reshape the design; the rest
are corrections. The reshaping, in one place:

| rev 1 said | rev 2 says | driver |
|---|---|---|
| the member weights are "exactly uniform" | **false** — the inner integral is importance sampling with weights `exp(ll_m)`; `M_draw` is now a *measured* parameter with a bias gate (§6.5) and the marginalization-accuracy budget is **OWNER DECISION 5** | R1-SEV1-1, R2-SEV1-5 |
| `f_p` is a bare area fraction used only in the count model | **`f_p = 1 - masked_frac_p` is a genuine per-pixel completeness and must appear on BOTH sides**: `C_p(z) = f_p C(z;theta)` in `dN_miss`, and `f_p` in `pi_pg`. Rev 1 had it on one side only — a sign-inverted completeness leaking into `Q` | R1-SEV1-2, R1-SEV1-3 |
| kernel = 50 Mpc radial, `ls_ang = 0.2` rad, `M = 8505` | the kernel is a 4:1 pancake (190 Mpc transverse vs 50 Mpc radial) and photo-z erases 79% of the 50 Mpc radial signal. **Go isotropic at ~190 Mpc: `M_z = 8-12`, `M = 2520-3780`** — 3x smaller, self-consistent with the data. **OWNER DECISION 3** | R1-SEV2-10, R1-SEV2-6 |
| `rho_m` (the budget renormalizer) is a per-proposal full-sky reduction — 86% of the added cost | **exactly two theta-free sky moments `A_m(z;b), B_m(z;b)` reduce `rho_m(z;theta)` and the field `log Z` to closed form.** The full-sky reduction leaves the per-proposal path entirely (§2.2). Added cost drops from a measured 25.6 ms to ~0 | R2-SEV1-2, R2-SEV2-14 |
| latent leaves built inside `body(coord, operands)` | leaves are **theta-free** and are therefore built in `make_likelihood` and `barrier()`-ed eagerly, per the documented convention (`factory.py:11-15`). No barrier conflict, no per-eval transient, no concurrency multiplier | R2-SEV1-6, R2-SEV1-3, R2-SEV1-4 |

The honest framing review demanded (R2-SEV2-14) survives, **scoped to rung 0**: under OWNER
DECISIONs 1, 4 and 8 *as rev 2 recommended*, `logQ(p,z; xi_m)` is theta-independent, so the latent
field enters the GW likelihood as a fixed set of `M_draw` Q-realizations — a compressed member
ensemble. That is the honest description of **PR-6a**. It is no longer the description of the
deliverable, because §1.7 restores the theta coupling at ~1 ms/proposal.

### 0.2 Novelty — what may and may not be claimed

Calibrated to `OWNER_CONTEXT.md:105-148` and not one notch further. **No literature search was
performed while writing this document.** The owner's instruction stands verbatim and is a gate on
PR-11, not a formality: *"do an exhaustive paper-by-paper novelty search before saying 'first'"*
(`OWNER_CONTEXT.md:138-139`).

**The forbidden sentence.** *"We introduce Bayesian LSS reconstruction for incomplete dark-siren
catalogs"* — the owner's own verdict is that this **"would not survive a literature review"**
(`OWNER_CONTEXT.md:129-130`). Leyde, Baker & Enzi (Cosmic Cartography) already do Bayesian
reconstruction of the galaxy field, propagate voxel-count uncertainty, and marginalize over
cosmological and bias parameters. On the theta-coupling axis Cosmic Cartography is **ahead of rev 2's
recommended design**, which is an independent reason the promotion is right. Nothing in this plan
may be written as if 3-D Bayesian density reconstruction for dark sirens were new.

The owner's scoring, transcribed, because every claim in the paper must be placed on it:

| claim | owner's score |
|---|---|
| homogeneous missing galaxies | none |
| `1 + b delta_g` correction | low |
| a multiplicative LSS `Q` | low |
| using clustering to reconstruct missing galaxies | not novel |
| Poisson/lognormal Bayesian density reconstruction | not novel broadly |
| **3-D Bayesian dark-siren catalog reconstruction** | **not novel broadly** (Cosmic Cartography) |
| low-rank sphere × z GP implementation | interesting implementation |
| posterior-mean rather than MAP `Q` | strong technical choice |
| per-z mean-one missing-budget constraint | interesting methodological contribution |
| empty pixels as Poisson information | important, not fundamentally new |
| `C_selection` separated from `Q` | quite interesting combination |
| marginalize `Q` members through a complete GW HBI | **potentially novel / strong differentiator** |
| joint multi-survey shared latent `Q` | **potentially very novel in dark sirens** |
| joint multi-survey + selection `C` + GW marginalization | **NOT IMPLEMENTED YET; strongest paper direction** |

#### Differentiator 1 — budget-preserving completion-field marginalization

`Q^(m)` posterior realizations of the clustered missing field, **each constrained to carry zero
missing-budget monopole**, marginalized through **both** the event terms and the detector-selection
normalization: `<Q^(m)>_{w,z} = 1` per realization, `logL = LSE_m logL(Q^(m)) - log M`.

*Delivered at PR-6a.* It does **not** require the field to be in the HBI. Say that plainly: the
cheap path is not worthless — it is precisely the path that lands differentiator 1.

**Three precisions that must ride with any abstract sentence:**

* **The budget identity is already exact in the shipped production completeness modes** —
  **[verified]** under `c_mode in {aggregate, selection}` the gp3d builder fits the whole sky
  (`fit = np.arange(n_pix)`, `build_lognormal_completion.py:714`) with **p-independent** weights
  (`w_budget = np.tile((1 - Cbar_fine) * dN_exp_density, (n_fit, 1))`, `:744`), so the mean-one
  condition reduces to `<Q>_sky = 1` and `sum_p (1-C) dN_exp Q_p` equals its `Q=1` value exactly,
  for every `z`, every member and every `theta`. Under radial `per_pixel` unfitted rows ship
  `logQ = 0` exactly (`:570-590`), so the total is conserved there too; the one genuine leak is the
  **gp3d `per_pixel` borrowing halo**, whose unrenormalized angular tail the builder's own docstring
  concedes (`:975-999`) — and `per_pixel` is refused in latent mode anyway (guard 6). So v2's eq. (4)
  is **not** a repair of a shipped defect. It is the requirement that the identity stay exact once
  §1.2 makes the consumption weights p-dependent (`f_p` inside `dN_miss`) — an internal-consistency
  obligation this plan creates for itself. `reviews.md` R1-SEV1-3 is re-dispositioned accordingly.
* **The marginalization is over the survey volume, not over the budget.** Under OD7, `Q == 1` with
  **zero variance** off-footprint (38% of the sky) and above `z_depth` (99.99% of the missing
  budget; risk R15). "We marginalize over the clustered missing-galaxy field" is true over the
  volume the survey sees and false over the volume that carries the budget. That belongs in the
  paper's scope sentence, not only in the risk table.
* **The 90x compression is engineering, not a methods claim.** Demote it from headline to enabling
  clause: *"budget-preserving member marginalization at a defensible `M_draw`, which only the
  compressed latent representation makes affordable."* The table member cube is
  `M_draw x 49,143 x 1086 x 4 B` = 1.71 GB at `M_draw=8` and 13.7 GB at 64; the latent row factors
  are 11.7 MB and 93.6 MB (`N_grid/M_z = 1086/12`).

**Literature this must be searched against before "first":** Dalang & Baker (variance completion)
and its implementation paper (the ratio-to-homogeneous packaging); Leyde, Baker & Enzi, Cosmic
Cartography I & II; Cheng & Gair (harmonic cross-correlation); the catalog-completeness lineage —
`gwcosmo` (Gray et al.), `DarkSirensStat`/`CHIMERA` (Finke et al., Borghi et al.),
`icarogw` (Mastrogiovanni et al.), GLADE+ completion (Dálya et al.); and anything enforcing a
mean-one completion field in a GW context.

#### Differentiator 2 — shared multi-tracer completion realizations, with selection `C`

One posterior over `delta` with per-tracer `(C_k, b_k)`, **matched-realization** marginalization in a
multitracer dark-siren HBI. The owner scores the *conjunction* — joint multi-survey **+** selection
`C` **+** GW marginalization — as **not implemented yet, strongest paper direction**.

*Delivered at PR-7,* and it delivers the matched-realization property **structurally**: one `xi`
shared across K tracers makes "member m of every catalog is the same realization" a **theorem**, so
`realization_set_id` / `member_content_sha256` matching is retired rather than satisfied (§4.4).

**The strongest design claim in this plan, currently under-stated: the latent path closes the
joint-builder seam by making the joint builder unnecessary.** There is no shared `realization_set_id`
to stamp because there is no exported per-catalog `Q` ensemble to match. That converts a three-way
compatibility matrix (joint builder × `c_mode` × K) into nothing. §0.4 is the evidence.

**Literature this must be searched against:** Cheng & Gair; the multitracer LSS lineage (Seljak
2009; McDonald & Seljak 2009; Abramo & Leonard); GW×LSS cross-correlation dark sirens (Mukherjee
et al.; Bera et al.; Diaz & Mukherjee; Ferri et al.); Cosmic Cartography II for the multi-probe
reconstruction.

#### What is retired as a novelty claim

* **"A correct conditional posterior for the field"** — dies as novelty (Cosmic Cartography), and its
  "no fixed-theta provenance firewall" half was false at rung 0 anyway (§0.0). What survives, stated
  narrowly, are two *supporting technical* claims about the **dark-siren missing-galaxy budget**, not
  about density reconstruction: (i) the **photo-z forward-convolved shell response `W`** (§1.4) —
  convolving the model rather than the data, keeping counts integer and the multinomial exact; and
  (ii) **per-pixel areal completeness entering both the count model and the missing budget** with a
  conserved budget identity (§1.2 + §4.2). DESI makes both first-order, and `OWNER_CONTEXT.md`'s
  sketch contains no photo-z treatment at all.
* **"Retirement of the provenance firewall by a change of likelihood"** — dies as a standalone claim;
  it is software maintainability. Under promotion it becomes *substantively* true ("no `Q` built at
  fixed cosmology") and earns exactly one sentence, phrased as the owner phrases it.
* **"The spurious `H0` standard-ruler channel is eliminated"** (rev 2 §1.1 item 3) — this was a
  *defensive* claim. Under promotion it inverts into the central scientific question. Restate it per
  §1.8; it cannot survive as written in either direction.

### 0.3 The three rungs of theta-coupling, and the criterion that separates them

| rung | design | which `theta` reach the field | per-proposal cost | ladder |
|---|---|---|---|---|
| **0** | shell-total–conditioned multinomial + **frozen** `W` | none | ~0 | **PR-6a** (control + fallback) |
| **1** | shell-total–conditioned multinomial + **theta-live** `pi_pg` | `Om0, w0, wa, delta`, `theta_sel` — **not `H0`** | **~1 ms** via linear response; ~1 s via re-solve | **PR-6b** (the deliverable) |
| **2** | **unconditional** Poisson (F3 restored) | the above **plus `H0`**, via the `n0 (c/H0)^3` budget | rung 1 + the Laplace log-det | **PR-6c** (optional, gated on OD6) |

Rev 2 jumped 0 → 2 and priced rung 2 at 8 days of the riskiest engineering in the document (PR-9).
**Rung 1 is the missing middle**, it is where the owner's sentence is satisfied for every parameter
the galaxy field can honestly constrain, and it costs ~5 days.

**What each rung buys:**

* **Rung 0** buys differentiator 1 in full, plus the shared-realization theorem at K>=2, plus the
  90x compression that makes `M_draw` affordable. It does **not** buy "no `Q` at fixed cosmology" —
  the fixed cosmology moves from an HDF5 attribute to an artifact fingerprint.
* **Rung 1** buys the owner's headline sentence, and it buys it twice over. (A) The *galaxy-side
  evidence* `log p(d_gal|theta)` becomes theta-dependent through the within-shell weighting — a new
  cosmological constraint from 22.79M galaxies. (B) The field realizations delivered to the GW
  likelihood become theta-dependent, so the missing-host placement responds to the proposal. **(A)
  and (B) must be reported separately** (§6.4), because (A) is a *shape* constraint driven by the
  assumed base and the photo-z forward model and is the more fragile of the two.
* **Rung 2** buys `H0` in the count channel — and `H0` enters there **only** as the volume–density
  amplitude `n0 (c/H0)^3`, exactly degenerate with `n0` (§1.8). It is therefore not a free win; it is
  a demand on the `n0` prior.

**The criterion, and it is measured, not chosen — and in v3 it is measured in nats, not in per-mode
posterior sd.** v2 gated the whole promoted direction on
`tau = max_theta ||xi_hat(theta) - xi_hat(theta_ref)||_H / sqrt(M)`, a **per-mode** norm. That
re-commits, at the scheduling level, the exact `sqrt(M)` error v2 congratulated itself for catching in
its own C2 (§6.3): what decides whether rung 0 *is* rung 1 is not whether the fitted field looks
different, it is whether the **likelihood** is different, and the likelihood error compounds over all
`M` modes exactly as the importance weights do. To first order
`Delta logL = grad_xi(ll) . Delta xi_hat`, and by Cauchy-Schwarz

```
|Delta logL|  <=  ||grad ll||_{H^-1} * ||Delta xi_hat||_H  =  sigma * tau * sqrt(M)     (0)
```

because `Var_m(ll_m) = grad(ll)^T H^-1 grad(ll)` to the same order — i.e. the **member spread `sigma`
of §6.5 is the same object** that converts `tau` into nats. At `M = 3780` (`sqrt(M) = 61.5`) and
§6.5's `sigma in [1, 2.6]`, `tau = 0.02` already admits **1.2-3.2 nats** — 12-32x the 0.1-nat P14
budget — and `tau = 0.3` admits 18-48 nats. The alignment-averaged value is `sigma * tau ~ 0.02-0.05`
nats and passes comfortably; the two readings differ by `sqrt(M) = 61.5` and **nothing in this plan
bounds the alignment** between `Delta xi_hat` (a radial re-weighting of shells) and `grad ll` (the
events' localization volumes), which live in the same radial direction and have every reason to
overlap. A criterion undetermined by a factor of 61.5 cannot gate a 33-day ladder.

**v3 therefore gates on the directly measured number, at zero extra solve cost.** New pin **P7c**
(§6.3): at the *same* 20 prior draws P7 already solves, evaluate the shipped member-marginalized GW
log-likelihood at the anchor field and at the exact theta-coupled field and report the
**theta-oscillation**

```
Delta(theta) = LSE_m ll_m(xi_hat_theta + L_H^-T g_m) - LSE_m ll_m(xi_hat_ref + L_H^-T g_m)
osc_theta Delta = max_theta Delta - min_theta Delta          [nats]                      (0b)
```

(the oscillation, not the level, because a constant offset is absorbed into the evidence —
`MODEL.tex` Rem. `rem:crn`). The thresholds, in the same units as every other budget in this document:

* `osc_theta Delta < 0.1 nat` — **rung 0 *is* rung 1** at the P14 budget. Ship PR-6a and quote the
  measured bound. Promotion adds nothing *and* effect (A) is negligible (see the mutual exclusivity
  below).
* `0.1 nat <= osc_theta Delta`, **and** the linear-response residual of P7b reduces the *residual*
  oscillation below 0.1 nat — **rung 1a is the design**: mean-shifted draws with common random
  numbers plus linear response (§1.7).
* the residual oscillation cannot be brought under 0.1 nat by linear response with a second-order
  term and multiple anchors — the linearization is inadequate, a re-solve is required, and the
  re-solve is infeasible at production scale. **Refuse promotion; ship PR-6a with the measured
  theta-dependence quoted as a systematic** (kill criterion K9).

`tau` is **retained and reported**, in two secondary roles it is correct for: as the
model-misspecification diagnostic ("does the fitted field look different"), and as the input to
`prop:esslaw` that makes the 1b trap visible (§6.3). It is no longer a gate. Equation (0) is the
**prediction** P7c confirms or refutes, in the same spirit as §6.5 item 5: a measured
`osc_theta Delta` far below `sigma * tau * sqrt(M)` means the field shift is nearly orthogonal to the
events' support, which is itself the most useful thing the measurement can say.

**[SUPERSEDED BY §0.5 D4 — the following paragraph's central claim is WITHDRAWN.]** The two
quantities are different contractions of the same residual and are **not** co-monotone: the evidence
responds to the full count residual, the field only to its `span(Phi)` projection damped by `H^{-1}`,
so "`tau` small with a large (A)" is the structurally expected case for a within-shell covariance.
Effect (A) has its own gate, **P7e** (§0.5 eq. 0d). Retained below for the record:

**Effect (A) and the benign branch are the same number, read two ways — and they are mutually
exclusive.** By `prop:cancel` *all* of rung 1's galaxy-side theta-dependence is the within-shell
covariance residual `eq:cancelresidual`, which is exactly what freezing `W` sets to zero and exactly
what P7 measures. So the advertised win (A) — "a new cosmological constraint from 22.79M galaxies"
(below) — and K9's benign branch ("promotion adds nothing") cannot both be live: if `tau` is small
enough for rung 0 to *be* rung 1, (A) is negligible; if (A) is a real constraint against 22.79M
galaxies, then `tau` is large, risk R19 is live and the linearization is at risk. v2 carried R21 and
the benign branch as co-equal possibilities without saying so. §6.4's channel split is the
instrument that decides it, and its acceptance rule is tightened accordingly.

Rung 1 → rung 2 is decided by **OWNER DECISION 6**, not by taste: rung 2 is admissible only if
`log10n0` carries an `H0`-aware prior (§1.8). Without that, rung 2 manufactures an `H0` constraint
out of the fiducial cosmology (kill criterion K10).

**One sub-variant distinction that decides whether rung 1 is affordable at all, and which neither
the owner context nor rev 2 draws.** At fixed `H`, a mean shift is exact:

* **1a — mean-shifted draws (CRN).** `xi_m(theta) = xi_hat(theta) + L_H^{-T} g_m` with `g_m` fixed
  across proposals is an **exact draw** from `N(xi_hat(theta), H^{-1})` for every `theta`, provided
  `H` is theta-independent. **No importance weights, no ESS penalty, no Jensen re-derivation.** The
  only approximations are the linearization of `xi_hat(theta)` (§1.7) and the theta-drift of `H`
  (second order).
* **1b — fixed draws + importance weights.** Keep the draws at `xi_hat(theta_ref)` and re-weight.
  Then `Var[log w] = ||xi_hat(theta) - xi_hat(theta_ref)||_H^2` **exactly** (Gaussian mode shift at
  fixed `H`; `MODEL.tex` Prop. `prop:esslaw`), so `ESS/M ~ exp(-||Delta||_H^2)`. At `M = 3780` a
  per-mode displacement of `tau = 0.1` gives `||Delta||_H^2 = 37.8` and **`ESS/M = 4e-17`**.

**Ship 1a. Never 1b.** This is also the correction to rev 2's P7 tolerance: `tau < 0.1` reads as
comfortably tight and would be catastrophic if anyone implemented 1b (§6.3).

### 0.4 The seam — which half is open (verified on `0c5b3db`, this session)

`OWNER_CONTEXT.md:140-148` says the joint multi-survey `Q` builder "has not caught up with selection
`C`" and that "[as of this analysis] the inference forbade `c_mode=selection` for K>=2", and calls
this "where I would work next". **One half of that is now stale and one half is understated.**

**Inference half — OPEN. The context is stale and now false.** K>=2 × `c_mode=selection` assembles
and evaluates end to end:

* `cli/inference.py:2365` `_selection_c_mode_by_catalog` resolves `c_mode` to a length-`n_catalogs`
  tuple; `inference/prior.py:676-685` already accepts a per-catalog `c_mode` sequence.
* `cli/inference.py:2378+` `_resolve_selection_fits` takes **one `--selection_fit` per catalog**,
  with an all-or-nothing anchoring rule, a **homogeneous luminosity-function family** requirement
  (`:2454-2465`) and one shared Schechter `M_faint_offset` (`:2522-2538`).
* `cli/inference.py:686-860` `_check_selection_qtable_theta` is explicitly **per catalog** — catalog
  k's table against catalog k's own fit.
* `tests/test_multitracer_selection.py` is a **K=2 × `c_mode=selection` end-to-end likelihood
  fixture**, 10 tests: finite likelihood with a live mixture weight, identical-catalog collapse to
  K=1 for any weight, per-catalog theta routing, per-catalog `K(z)` routing, two real fit JSONs
  through the resolver, homogeneous-Schechter K=2, bright-truncated offset.

**The correct narrow statement: inference forbids *stratified* selection fits at K>=2, not
`c_mode=selection` at K>=2.** `inference/parameters.py:460-470` and its pre-load twin
`cli/inference.py:1492-1502` refuse it, and the stated reason is data plumbing, not modelling: the
full-sky stratum map hangs off the **shared** data bundle while a K>=2 mixture builds each catalog's
views from its **own** bundle, so the map never reaches catalog k's `EMCatalog`. Two further narrow
K>=2 refusals survive: `--universe_model dark_sirens_complete` (`cli/inference.py:1590`), and
in-likelihood restrictions that are family-level rather than K-level (stratified selection is
gaussian-only; Schechter refuses a `K(z)` template — `redshift/completion.py:914-926`).

**Builder half — CLOSED, in four independent ways; the context names one.**

1. **No `c_mode` at all.** `cli/build_joint_lognormal_completion.py:180-181` calls
   `_assemble_gp3d_survey(path, cosmo=cosmo, survey=survey_k, z_s=z_s, edges_s=edges_s)` — no
   `c_mode` — taking the `per_pixel` default (`build_lognormal_completion.py:648`). **[verified]**
   grep counts in the joint builder: `c_mode` **0**, `selection_fit` **0**, `stratum` **0**.
   It also saves without `c_mode=` (`:206-209`, `:359-363`), so the file reads as legacy `per_pixel`
   and is **hard-rejected at load** by any `aggregate`/`selection` run (`catalogs/lss.py:33-48`).
2. **No budget gauge fixing.** `renormalize_q_mean_one` is **[verified]** never called in the joint
   builder (grep count **0**; `budget_renorm` count **0** against 15 in the single builder), so every
   joint file trips the legacy warning (`lognormal_completion.py:1191-1205`) and ships the full
   Jensen monopole — the measured **+55%** budget inflation the K=1 tables remove. The joint parity
   test concedes it by comparing against a `budget_renorm=False` single build
   (`tests/test_joint_lognormal_completion.py:107-112`).
3. **The rank cannot resolve any physical scale.** The joint inducing grid is hard-coded
   `M_SPH, M_Z = 32, 6` up to `Z_NODE_HI = 3.0` (`:73-74`), so the zeta node spacing is
   `log(4)/5 = 0.2773`. The **hard** radial guard then demands `ls_z >= 0.2773`, i.e.
   **`L_smooth >= 1.34 Gpc`** at `z_ref = 0.237` — every physically supportable length
   (50–190 Mpc → `ls_z = 0.010–0.039`) hard-fails, and the builder's own error message says the only
   remedy is raising `--lss-corr-length-mpc` (`:222-237`).
4. **The binding constraint the context misses.** The *single-catalog* builder **does** support
   `--c-mode selection` (`build_lognormal_completion.py:1148`) and `--selection-fit` (`:1160`) — but
   **[verified]** it has **no `--realization-set-id`** (grep count **0**, against 13 in the joint
   builder), so `save_lss_completion_hdf5` mints a fresh `uuid4().hex` per build
   (`lognormal_completion.py:1055-1056`). **The joint builder is the only producer of a shared
   `realization_set_id`, and it is `per_pixel`-only.**

**The precise seam statement: the consumer is ready; the producer cannot produce; and the one escape
hatch that lets the configuration run is exactly the physically-wrong one.** There is no route at any
K — joint, or K-times-single — to matched-member `c_mode=selection` `Q` ensembles. K>=2 × selection ×
`--lss_marginalize` is reachable today **only** by waiving the guarantee with
`--allow_unverified_shared_lss_members`, and the code itself states what that does: it marginalizes
over an **independent-fields product prior**, not the shared-field prior the estimator assumes
(`inference/loaders.py:352-395`). Restated once more, because it is the cleanest one-line motivation
for the whole promotion: **a K>=2 shared-latent LSS marginalization under selection `C` is not
refused — it is not constructible.**

**Where to work, therefore.** *Not* on the joint selection builder. PR-7 makes it unnecessary (§4.4).
One cheap, explicitly-interim exception is scheduled — passing `--c-mode` / `--selection-fit` /
`--realization-set-id` through the joint builder (~1 d) **solely** to produce the K>=2 selection
*table* baseline that OWNER DECISION 10's arm comparison needs and that cannot be built today. It is
**mock-scale only**: defect 3 above forbids a physically sensible DESI-scale build, and the DESI-scale
gp3d attempt already OOM'd at 21.7 GB (`logs/qbuild_gp3d_recal_1119087.err`).

### 0.5 The third adversarial round — twenty-eight findings, and the four that change the design

Two more independent reviews landed on v3 (`reviews.md` Part 3). Seven distinct SEV1 issues, seventeen
SEV2, four SEV3. **This subsection is the authoritative record of what v4 changed**; every other
section below is edited to agree with it, and where a downstream sentence still reads as v3 it is
superseded here.

Four findings change the design. The rest are corrections, and they are tabulated at the end.

#### D1 — The within-shell linearization is a stated premise, not an accident

*(fixes: the shell-collapsed gradient/Hessian; "`W` is a no-op unless the exponential is inside"; the
"exact gauge direction" claim; `P5`/`P6`'s reference objective; the two-stage cost model.)*

v3's eq. (1) put the exponential **inside** the shell sum, which makes
`eta_pg = log sum_n W_gn e^{b f(p,z_n)}` a log-sum-exp — **nonlinear in `xi`**. Three v3 claims are
false for that object: `Phi_g = Phi_s (x) phi_z[g]` (the `z` factor becomes `p`-dependent, so the
Kronecker separability collapses), eq. (3) (a nonlinear `eta` adds
`-sum_p (N_pg - T_g pi_pg) d^2 eta/d xi^2`, which is sign-indefinite), and §3.4's arithmetic (the
exact object needs `T[g,i,j,a,b]` = 3.7 GB and ~4.5e16 flop per Newton step — hours, not minutes, and
no offline `S` at all). The reviewer's dilemma is real and unavoidable: a `p`-independent
`sum_n W_gn` divides out of `pi_pg` identically, so `W` does something **only** if it acts on an
object that is `p`-dependent.

**The resolution is to move `W` onto the basis rather than around the exponential**, and to say out
loud what that costs. Exactly,

```
Lambda_pg = a_g(theta) f_p int_g omega_g(z;theta) exp[b s_p(z) - (b^2/2) sigma_p^2(z)] dz
log int_g omega_g e^{b s_p} dz = b <s_p>_g + (b^2/2) Var_g(s_p) + O(b^3 kappa_3)
```

where `<.>_g` and `Var_g` are the `omega_g`-weighted within-shell mean and variance. Keep the first
term and define the shell-response operator to act on the **radial basis rows**:

```
phi~_z[g,:] = sum_n W_gn phi_z[n,:],   W_gn = (int_g K(z|z_n) dz) omega_g(z_n;theta) Delta_n,
                                        rows normalized to 1
eta_pg = b (Phi_s[p] (x) phi~_z[g]) . xi                                          (1')
pi_pg  = f_p e^{eta_pg} / sum_p' f_p' e^{eta_p'g}
```

Then `<s_p>_g = (Phi_s[p] (x) phi~_z[g]).xi` **exactly**, `eta` is **exactly linear in `xi`**, and
therefore: `Phi_g = Phi_s (x) phi~_z[g]` is exact; eq. (3) is the **exact** Hessian of (1'); the
sky-constant subspace is an **exact** gauge direction of (1'); the separable two-stage contraction and
its ~1 s anchor-build arithmetic stand; and `W` is not a no-op, because it is the photo-z forward
convolution of the *basis* — the correct place for a forward model, and the reason the counts stay
integer.

**What is neglected, exactly.** The dropped term is
`(b^2/2)[Var_g(s_p) - <sigma_p^2>_g]` — the within-shell variance of the linear predictor, minus its
prior counterpart. It is `p`-dependent (that is why it does not divide out) and it is **the same
object as `prop:cancel`'s residual `eq:cancelresidual`**: both vanish identically iff `e^{b s_p(z)}`
is shell-constant. So one stated approximation — *the field is linear across a shell* — now buys all
three things v3 claimed separately and inconsistently: the theta-cancellation, the Kronecker
separability, and the exactness of eq. (3). The controlling scale is the shell width against the
photo-z-smoothed radial structure (§1.4), which is exactly the argument §1.4 was already making.

**Pins.** `P5`/`P6` reference **(1')**, not the exponential-inside object. New **P5b** measures the
neglected Jensen term directly at `xi_hat` on the real catalog:
`max_g,p (b^2/2)|Var_g(s_p) - <sigma_p^2>_g|` and the induced `Delta log p_count`. New **P5c**
compares (1') against an exact-quadrature objective **at reduced rank only** — the exact object is a
validation reference, never a shipped solve, and §3.4 now says so.

#### D2 — The galaxy-side evidence is the Laplace evidence, not the count log-likelihood

*(fixes: the envelope-theorem step in §1.7, flagged independently by both reviewers.)*

v3 §1.7 wrote `log p_count(theta) = sum_g sum_p N_pg log pi_pg(xi_hat_theta, theta)` and dropped
`d xi_hat/d theta` "by the envelope theorem". **That is wrong, and `MODEL.tex` says so in the opposite
direction.** With `J = 0.5||xi||^2 - l` and `l = sum N log pi`, stationarity gives
`d l/d xi = xi_hat`, **not** `0`. The envelope theorem removes `d xi_hat/d theta` from `J`, not from
`l`. The correct rung-1 galaxy-side term is `MODEL.tex` Prop. `prop:laplace`:

```
log L_gal(theta) = - J_theta(xi_hat_theta) - 0.5 log det H(theta)
                 = l(xi_hat_theta, theta) - 0.5 ||xi_hat_theta||^2 - 0.5 log det H(theta)   (5)
```

v3 shipped only the middle term. **Size of the omission:** with ~1015 data-constrained modes (§1.5)
`||xi_hat|| ~ 32`, and `||S.Dtheta||_2 <= ||S.Dtheta||_H = tau sqrt(M) = 6.2` at `tau = 0.1`, so the
dropped `d(0.5||xi_hat||^2)/dtheta = xi_hat^T S Dtheta` reaches **~2e2 nats worst case and ~3 nats at
random alignment**, against a 0.1-nat budget. This is not a diagnostic slip: §6.4 makes
`Delta log p_count` a mandatory reported channel and R21 says it may dominate 259 events.

**And it is free.** Under linear response `xi_hat_theta = xi_hat_ref + S.Dtheta`, so

```
0.5||xi_hat_theta||^2 = 0.5||xi_hat_ref||^2 + (S^T xi_hat_ref).Dtheta
                      + 0.5 Dtheta^T (S^T S) Dtheta
```

— one precomputed `n_theta`-vector and one `n_theta x n_theta` matrix, both offline, **zero online
cost**. The Occam term is the cheapest term in the rung.

**The log-det is now pinned, not assumed.** "With `H` frozen, `log det H` is constant" converts a
modelling approximation into a deleted term whose neglected motion enters multiplied by `M/2 = 1890`
(a 0.1% per-mode drift is 1.9 nats). New **P7d**: at the *same* 20 prior solves `P7` already performs,
report `osc_theta 0.5|log det H(theta) - log det H(theta_ref)|` — free, since the Cholesky is already
formed. If it exceeds 0.1 nat, the rung-1 evidence must carry the linear-response log-det term
`0.5 tr(H^{-1} dH/dtheta).Dtheta` (also an offline `n_theta`-vector), and the frozen-`H` claim of
§1.1 item 1 is scoped to the *draw covariance*, not to the evidence.

**Withdrawn:** §1.7's sentence "`S` is needed for the GW-side field shift, not for the evidence term."
`S` is needed for both.

**And the drawn covariance is measured too.** Drawing `xi_m` with `H_ref^{-1}` instead of `H_theta^{-1}`
shifts `logL` by `~0.5 a^T (H_ref^{-1} - H_theta^{-1}) a ~ 0.5 sigma^2 eps_H` — 0.11 nats at
`sigma = 1.5` and 0.34 nats at `sigma = 2.6` for a 10% drift, i.e. at or above the P14 budget. Folded
into **P7c'** (below), evaluated at the same 20 solves. v3 called this "a second-order neglect (a
covariance misspecification, not a mean error)" and measured it nowhere.

#### D3 — PR-0 becomes the decision rung: the two measurements that can kill the direction move to day 1

*(fixes: the inverted ladder; the literature search gated last.)*

The field modulates only `(1 - f_p C) dN_exp` **inside the footprint and below `z_depth`**: above
`z_depth` `Q == 1` (`completion.py:1295`, verified) and off-footprint `Q == 1`. Three independent
estimates of the in-support fraction of the missing budget now exist and agree at the same order:
R1's **6e-5**, a reviewer's **2.7e-4**, and an integral performed this session over the shipped
`C_sel` (`M0hat = -20.3098`, `sigma_M = 0.7144`, `m_lim = 21`, `delta = 0.9402`, `z` in `[0,6]`)
giving **1.3e-4 all-sky and 7.9e-5 over the occupied footprint**.

> **ORCHESTRATOR VERIFICATION (2026-08-10, session-owned number).** Recomputed
> independently from `data/selection_fit_union.json` (m_lim 21.0, M0hat −20.3098,
> sigma_M 0.7144) **including its cubic K(z) template**, with delta = 0.9405 from
> the 6.0-grid calibration, on the production `DARKSIRENS_ZMAX=6.0` grid:
> **in-support fraction = 2.2e-5 all-sky, 1.4e-5 restricted to the occupied
> footprint (f_sky 0.6199)**. This is BELOW all three earlier estimates because
> the K-correction drives C_sel down faster with z (C_sel = 0.9206 at z_depth,
> < 1e-3 by z = 1), moving still more of the missing budget above the depth where
> Q == 1. Use this number; the others were computed without the template and/or on
> a different grid. Cumulative budget below z: 0.000 at z <= 0.3, 0.025 at z <= 1,
> 0.31 at z <= 3.
 Whatever the exact figure, the field
is being asked to redistribute **~0.01% of the missing budget**, and R12 already records that the
production `H0` posterior is bit-identical with and without the `Q` table.

v3 scheduled the decisive measurements at PR-5b (day ~23) and PR-6a (day ~28), downstream of the
infrastructure whose value they determine. **They cost <= 1 day each against artifacts that exist
today.** PR-0 grows from 1 d to 3 d and becomes blocking on four items:

1. **`sum_i phi_i`** — the event-weighted fraction of each event's prior mass in the in-support
   missing branch — and, from the same objects, the closed-form member spread of §6.5 item 5.
2. **The Q-on-at-anchor oscillation.** Turn the *shipped* table on at the anchor and report
   `osc_H0 [ logL(Q on) - logL(Q == 1) ]` across the `H0` prior. This is an upper bound on everything
   the entire ladder can buy, measured with code that already runs.
3. **The in-support budget fraction**, reconciled across the three estimates above and published.
4. **The exhaustive paper-by-paper novelty search of §0.2** (1 d), moved from PR-11. v3 gated the
   search that decides whether 28 days of work has a paper at day ~50 while claiming a publishable
   differentiator at day ~28; OD11's own words ("the cheapest possible time to discover that a
   differentiator is already in the literature") contradict v3's own schedule.

New kill criterion **K0** (§9). If (1) and (2) both come back small — `sum_i phi_i < 1e-3` and
`osc < 0.1` nat — the field cannot move the 259-event posterior at all, the honest deliverable is the
bounded-systematic result §6.5 already names as the fallback, and it is reachable in ~3 days instead
of ~34. **This is the highest-value change in v4**, and it is decision-theoretic, not technical.

#### D4 — The promotion gate is a *directional* first-order oscillation, and the galaxy side gets its own

*(fixes: `tau`'s `sqrt(M)` indeterminacy; `P7c` existing only in §0.3 and being unevaluable where it
was placed; §6.5 item 5's wrong inner product; and §0.3's mutual-exclusivity claim, which is wrong.)*

v3 correctly retired `tau` (the per-mode norm) as a gate, because
`|Delta logL| <= sigma tau sqrt(M)` spans a factor of `sqrt(M) = 61.5` depending on an alignment
nothing bounds. But v3's replacement, `P7c`, appeared **only** in §0.3 — the pin table, the PR-3 gate
list, K9 and R19 all still gated on `tau` — and as written it required the shipped member-marginalized
seam (PR-5, day ~24) at a rung that ends on day ~14. Both defects have the same fix: **do not bound
the alignment, measure it.**

The gradient of the GW log-likelihood with respect to the field is available in closed form from
objects that exist before any seam is written (§6.5 item 5):

```
a = grad_xi(sum_i ll_i) = b_GW ( sum_i phi_i Phi_i  -  N_obs <Phi>_sel )                (6)
```

— the basis, the events' pixel/redshift distribution, the injection weights and `C(z;theta)`, and
nothing else. Two corrected consequences:

* **The member spread is an `H^{-1}` norm, not a Euclidean one.** Eq. (0)'s own derivation gives
  `Var_m(ll_m) = a^T H^{-1} a`, so
  `sigma = ||L_H^{-1} a||_2`, **not** `||a||_2`. §6.5 item 5's ingredient list omitted `H_chol`
  entirely; since `H >= I`, the Euclidean form systematically **over**-predicts `sigma`, and `sigma`
  drives an exponential `M_draw` requirement and OWNER DECISION 5. Corrected in §6.5.
* **The promotion gate is the measured inner product, not the Cauchy-Schwarz bound:**

  ```
  Delta logL_1(theta) = a . (xi_hat_theta - xi_hat_ref)
  osc_theta Delta logL_1 = max_theta - min_theta            [nats]                       (0c)
  ```

  Computable at **PR-3, day ~14**, from the same 20 prior solves `P7` already performs, with no seam,
  no moment tables and no `latent_q.py`. Thresholds unchanged in spirit and now in the right units:
  `< 0.1` nat -> rung 0 *is* rung 1 on the GW side; `>= 0.1` nat with `P7b`'s linear-response residual
  below 0.1 -> rung 1a; otherwise **K9**. `osc Delta logL_1` far below `sigma tau sqrt(M)` is itself
  the most useful thing the measurement can say: it means the field shift is nearly orthogonal to the
  events' support.

  **P7c'** is the nonlinear confirmation at PR-6a with the shipped seam
  (`LSE_m ll_m(xi_hat_theta + L_H^-T g_m) - LSE_m ll_m(xi_hat_ref + L_H^-T g_m)`), and it must agree
  with (0c) to the second-order term; it also carries the `H_ref`-vs-`H_theta` draw-covariance term of
  D2.

**§0.3's mutual-exclusivity claim is WITHDRAWN.** v3 asserted that effect (A) (the galaxy-side
evidence) and K9's benign branch "are the same number read two ways" and cannot both be live. They are
**different contractions of the same residual and are not co-monotone**:

```
d(log L_gal)/dtheta  <-  the full count residual r = N - T pi contracted with d eta/d theta
Delta xi_hat = -H^{-1} d(grad J)/d theta  <-  that residual PROJECTED onto span(Phi), damped by H^{-1}
```

A theta-direction lying **inside** `span(Phi)` is absorbed by the field: `tau` large, evidence flat,
no constraint. A theta-direction **orthogonal** to `span(Phi)` cannot be absorbed: `tau = 0`, evidence
constraint maximal. Since `eq:cancelresidual` makes the theta-direction exactly a *within-shell*
covariance — precisely the structure a shell-collapsed basis cannot represent — **"`tau` small with a
large (A)" is the structurally expected case, not an excluded one.** K9's benign branch therefore
licenses only "the GW-side field shift is negligible" and says nothing about (A).

**Effect (A) gets its own gate, free at the same 20 solves.** New **P7e**:

```
osc_theta [ l(xi_hat_theta, theta) - 0.5||xi_hat_theta||^2 - 0.5 log det H(theta) ]      (0d)
```

i.e. the oscillation of (5), the *correct* evidence of D2. R21 stays live independently of K9, and
§6.4's acceptance rule is tightened (below).

#### The other twenty-four findings, and where each is now carried

| # | finding | disposition in v4 |
|---|---|---|
| 1 | `P17` is stated against the **prior** marginalization while the shipped members are **posterior** draws; it also drops the selection subtraction, and §1.6's one-line derivation cannot produce the cross terms the bullets claim | **CORRECTED, and strengthened into the plan's only closed-form estimator test.** Two arms. (a) *counts-off* (`H = I`, `xi_hat = 0`): `LSE_m ll_m - log M -> (b^2/2) sum_ij K(x_i,x_j)`, the prior form, validated against truth. (b) *counts-on*, the shipped configuration: `LSE_m ll_m - log M - ll(xi_hat) -> 0.5 a^T H^{-1} a = 0.5 sigma^2` with `a` from (6) — the same `sigma` §6.5 item 5 predicts, so `P17` and the member-spread prediction become **one measurement**. The `- N_obs <Phi>_sel` subtraction is carried in `a` by construction. The correct Limit-III derivation is `log int dxi N(xi;0,I) e^{(a_GW + a_gal).xi} = 0.5||a_GW||^2 + a_GW . a_gal + 0.5||a_gal||^2`, which *does* produce event x event, event x voxel and voxel x voxel — v3's displayed line carried only the GW factor |
| 2 | The hierarchy is inconsistent whenever `b_GW != b_gal`: the catalogued-host branch is written field-free while the missing branch carries `e^{b_GW s}` | **CORRECTED — a model change.** Hosts trace `e^{b_GW s}` and galaxies trace `e^{b_gal s}`, so the host-per-*catalogued-galaxy* ratio is `e^{(b_GW - b_gal) s(x_j)}`. The spike weights become `w_j(Lambda) exp[(b_GW - b_gal)s(x_j) - ((b_GW^2 - b_gal^2)/2) sigma^2(x_j)]` (`MODEL.tex` eq:zprior-new amended). Limit I's "`xi` drops out identically" and `P16`'s "physics identity" are true **iff `b_GW = b_gal`**, and `P16` is restated at that condition; at `b_GW != b_gal` `P16` becomes a *non*-identity with a predicted size. OD8 is amended to recommend `b_GW == b_gal` in the headline, which makes Limit I exact and `P16` a genuine physics gate. The configurations where this bites are the ones actually scheduled (OD8's mock campaign, Tier E at `b_2/b_1 = 2`) |
| 3 | `prop:warmstart` bounds `J`, but the shipped rung-2 likelihood is `-J - 0.5 logdet H + LSE_m ll_m`; the member term is **first** order in `eps`, so "`eps < 0.14` buys 0.01 nat" is off by ~1.5 orders; and a saturating link can destroy `H >= I` | **CORRECTED.** The stopping rule is restated on the *likelihood*, not on `J`: `||grad J||_2 < eps` gives `||xi_* - xi_hat||_H <= eps`, so the member term moves by `<= sigma eps` and the budget is `eps < epsilon_budget / sigma` — **`eps < 3.8e-3`** at `sigma = 2.6` for 0.01 nat, not 0.14. The log-det term is bounded by evaluating `logdet H(xi_*)` *consistently* at the same stopping point (first order in the same `eps`). The convexity premise is restored by making **Fisher scoring normative**: `H := I + Fisher` is PSD for *any* link, including the smooth saturation, whereas the observed Hessian is not — §3.5 already said "fixed-trip Fisher scoring" and §10 now says why it is load-bearing. The multi-valued-likelihood worry does not apply under a deterministic anchored start with a deterministic stopping rule: `logL` is then a deterministic function of `theta` with a theta-*varying* residual, which is exactly what the `osc` budget governs. Binding at PR-6c only |
| 4 | `prop:compressexact` / `eq:requirefullsolve` is the exactness criterion for design **1b**, which §0.3 forbids; under 1a it can never fire, while the failure mode that *can* occur (theta-drift of `H`) has no pin | **CORRECTED.** `eq:requirefullsolve` is restated as **1b-only** and `MODEL.tex`'s online pseudo-code `S1/S2` no longer routes the FAST/FULL switch on `d2`. Under 1a the switch is governed by (0c) and (0d) plus `P7d`; the `H`-drift term is now measured (D2, `P7c'`). §1.6's citation of `prop:compressexact` for "which corner a given catalog/GW pairing sits in" is replaced by (0c)+(0d), which are the numbers that actually decide it |
| 5 | Factor **(F2) does not exist as a likelihood factor**: `magnitude_loglike_from_stats` is referenced nowhere outside its test; `theta_sel` enters as an **anchored Gaussian prior** | **CONFIRMED [verified this session] and CORRECTED.** `redshift/selection.py:341` is referenced only by `tests/test_selection_suffstats.py`; `inference/prior.py:1207` sets `kind_map[lbl] = ("normal", loc, scale)` from the offline fit's covariance, and `selection_fit_union.json` gives `cov[0][0] = 2.546e-8` -> `sigma(M0hat) = 1.60e-4` mag, matching §3.1's number. So the shipped hierarchy is **empirical Bayes on `theta_sel`**, not the joint of `eq:hierarchy`. §3.1 and `MODEL.tex` now say so. PR-2's work item is re-aimed: the question is not "re-derive F2 disjointness for Schechter" but **"may an anchored `theta_sel` prior coexist with a count likelihood whose base is `f_p C(z;theta_sel) Nbar`?"** — yes at rung 0 (`prop:cancel`: `pi_pg` carries zero `theta_sel` information), **bounded** at rung 1 (finding 6), no at rung 2. **K10 is extended from `n0` to `theta_sel`.** Whether to implement (F2) as a real likelihood factor is **OWNER DECISION 14** |
| 6 | Rung 1 re-opens a weakened form of the double-counting the plan proves fatal for rung 2: guard 5's premise ("the count channel carries zero information about `(n0, delta, theta_sel)` by construction") is exactly true at rung 0 and **false at rung 1** | **ACCEPTED and made measurable.** Guard 5 is restated **per rung**. At rung 1 the count channel carries exactly the within-shell residual information about `(delta, theta_sel)` — and that residual is precisely the quantity `P7e` measures. So the same number that gates the promotion also **bounds the overlap** with the calibration prior fitted to the same 22.79M counts. New rule: if `osc_theta` of (0d) restricted to the `(delta, theta_sel)` directions exceeds 0.1 nat, either the `delta` prior is widened by the measured overlap or rung 1's coupling set is restricted to `(Om0, w0, wa)`. No new machinery; one extra contraction of a solve that already runs |
| 7 | Effect (A) is degenerate with `delta`, with `sigma_z(z)`, and with the shell binning — all frozen inside `W` — so §6.4's acceptance rule is too weak | **ACCEPTED.** §6.4's rule is tightened to: **"(A) is reported as a diagnostic and never enters the headline posterior unless `W`'s own parameters (the within-shell profile `omega_g`, the photo-z kernel `sigma_z(z)`, the shell edges) are sampled or profiled."** Promoted to **OWNER DECISION 13**. (Its companion premise — that (A) and the benign branch are the same number — is withdrawn in D4) |
| 8 | The budget moments of eq. (2) are **all-sky**, while eq. (4) and the `Q == 1` off-footprint convention are **footprint-restricted**; the seam as written returns `Q = exp(-rho) != 1` off-footprint; the shipped code's own docstring warns against exactly the all-sky version | **CONFIRMED [verified] and CORRECTED.** `renormalize_q_mean_one`'s docstring requires the weights summed over "the FITTED FOOTPRINT ... never the full sky: out-of-footprint pixels' homogeneous budget must not absorb the footprint's monopole". Eq. (2) is restated over `F`: `A_m^F = sum_{p in F} e^{b f_m}`, `B_m^F = sum_{p in F} f_p e^{b f_m}`, `P_F = |F|`, `F_F = sum_{p in F} f_p`. Splitting eq. (4) at `f_p = 0` shows the off-footprint block (`weight 1`, `Q = 1`) is conserved trivially, so the constraint binds on the footprint block alone. **The seam needs an explicit index map**: the PE/injection pixel union is 49,143 / 49,152 rows against 30,470 footprint rows, so ~38% of gathered rows must route to `logQ = 0` exactly. New pin **P13b**: the seam returns bit-zero `logQ` on off-footprint rows |
| 9 | "Per-evaluation transient added: 0" is false at rung 1: `row_fac_shift` is used as `row_fac_shift[pix]`, so it is the `(n_rows x M_z)` row expansion — ~1.5 MB per evaluation, ~375 MB at 256 concurrency | **CORRECTED.** §2.4's categorical "0 transient" is scoped to **rung 0**. At rung 1 the shift is `(30470, 12)` f32 = **1.46 MB per evaluation, ~375 MB at 256 concurrency** against 72.7 GiB free — not fatal, but it must be *reserved*, so rung 1 routes through the **guarded transient branch** (`_slopes_and_fixed:708`, `batch_scale` at `:725`) rather than the static branch. The ~34 GB under-reservation precedent at `block_sizing.py:623` is the reason not to assume |
| 10 | The cost table divides a latent-minus-table delta by a baseline that has **no LSS and no member marginalization at all** | **ACCEPTED.** The production baseline (`--use_lss false`, no `--lss_completion`) contains zero member-dependent seam work, so the honest added cost of the deliverable is the **whole latent column**: 8.6 ms (**+31%**), 34.4 ms (**+125%**), 68.8 ms (**+250%**) at `M_draw = 8/32/64` against 27.5 ms. The delta column is the right number for a table-vs-latent comparison and is kept; **OWNER DECISION 5 is now taken on the baseline-relative column**, and PR-0 reports a three-column table (no-LSS baseline / table / latent) |
| 11 | §4.3 fixes `amp = 1`, which makes `b_gal` identified by the counts; §3.4 then calls `b_gal` and `amp` degenerate at K=1 and sets `s_b` to a 20% prior width | **ACCEPTED.** Both cannot hold. With `amp == 1` pinned, `b_gal` is the sole clustering amplitude and the count channel measures it. `s_b` becomes the **profile curvature** `s_b^2 = [-d^2 log p_count/db^2]^{-1}` at the anchor (one 1-D profile against the same `H_chol`), with a stated systematics floor; the 20% dial is retired. This also rescues Tier-B's "latent-on CI >= table CI", which with a free 20% dial could be made to pass or fail by choice |
| 12 | PR-7's Tier-E gate (iii) is vacuous: it requires a K=2 selection run "without `--allow_unverified_shared_lss_members`", but that flag and its check live on the table-loader path latent mode deletes | **ACCEPTED — a routing tautology, structurally the same defect as rev 1's theta-invariance pin.** (iii) is demoted from a gate to a **statement of fact** (the flag ceases to have a referent, §4.4). The gate becomes (i) bias-ratio recovery and (ii) shared-`xi` coupling demonstrably tighter than two independent fits, plus a new substantive (iii'): run the shared-`xi` likelihood **and** an artificially decoupled two-field variant on the same mock and show the bias-ratio credible region differs in the predicted direction — i.e. demonstrate the coupling the `--allow_unverified` flag throws away, rather than demonstrate that a deleted check does not fire |
| 13 | K8 at PR-2 is near-certain to fire, and the ladder's only response is "stop and re-plan" at day ~9; §1.2 also leads with the smaller of the two effects | **ACCEPTED.** The dominant term is **not** the p99 partial pixel: it is the **18,682 of 49,152 pixels with `f_p = 0`**, whose consumption weight moves from `(1-C)` to `1`. At `C ~ 0.5` the all-sky missing budget moves by roughly **+45%**, and the injection set is all-sky, so it lands directly in the selection integral. §1.2 is re-ordered to lead with it. K8 is made **non-terminal**: it is evaluated on the Tier-B/C closure and on a **non-railing** production configuration (the shipped posterior rails at `139.00 [138.3, 139.7]`, so "1 sigma" is undefined on the arm K8 names — the same vacuousness K2 was restated to avoid), and the `C`-side change ships as **its own deliverable with its own `H0` arm** rather than as a PR-2 side effect |
| 14 | §4.2's "exact there too" under stratified selection drops the theta caveat, and `prop:budgetexact(1)` does not cover the stratified branch | **CONFIRMED [verified] and CORRECTED.** `build_lognormal_completion.py:730-732` sets `w_budget = ((1.0 - Cfine_s) * dN_exp_density)[stratum_map[fit]]` — **p-dependent** weights at the build-time fit, exactly the two mechanisms `prop:budgetexact(2)` names. The stratified branch conserves the consumed budget **only at `theta = theta_ref`**; at any other `theta` it leaks at the order of the stratum-to-stratum spread in `C_sel`. §4.2's sentence is corrected; the claim survives for the production (non-stratified) configuration, which is what the surrounding argument needs |
| 15 | Colliding factor labels between `PLAN.md` (F1..F5) and `MODEL.tex` (F1..F4) | **CORRECTED.** `MODEL.tex`'s labelling is authoritative: **(F1)** prior, **(F2)** magnitudes, **(F3)** counts, **(F4)** GW. `PLAN.md`'s shell-total factor is renamed **(F3a) monopole** and its angular counts **(F3b) placement**, matching `eq:shellfactor` |
| 16 | §1.1 calls the shell-total factorization "exact", but `sum_p f_p e^{b s_p}` depends on `xi`, so the dropped factor carries `O(b^2 Var)` field information | **CORRECTED wording.** The *factorization* `p(N) = p(T) p(N|T)` is exact for any joint. Using only the second factor is a deliberate **partial likelihood**, and the discarded information is the shell monopole, which §4.1's projection removes to **linear** order and which retains `O(b^2 Var_sky(s)/2)` content. The word "exactly" is moved from the choice to the factorization |
| 17 | Limit III recovers Cheng & Gair for `eq:hierarchy` on paper, not for the estimator PR-6a/6b ships; the `Lambda_cat -> 0` trigger contradicts the `b_GW b_gal` cross term it produces; the expansion parameter is unwarranted | **ACCEPTED, all three.** (a) Limit III is presented as a property of **the hierarchy**, explicitly not of the rung-0 estimator, and the plan now states the consequence *in favour of the promotion*: conditioning on `d_gal` at **frozen theta** removes exactly the cosmology dependence of `C_l^{GWxg}` that a cross-correlation dark-siren measurement uses, and rung 1 is what restores it. (b) The trigger is corrected to *"the catalogued fraction of each event's prior mass tends to zero and the localization spans many correlation cells"* — the galaxy counts are **retained** as data (they are what carries `b_gal`), and the cross term arises from marginalizing the prior against (F3) **and** (F4) jointly, per finding 1's corrected derivation. (c) The expansion parameter is stated as `b_GW s` over the events' support and **measured** (`P20`), not asserted |
| 18 | `s_v ~ 5.6e-3` is an inherited, unverified number certifying three separate approximations, and it contradicts the plan's own `amp = 1` and rank conventions | **WITHDRAWN as a global certificate.** At `amp = 1` the *prior* sd of `s` is ~1, and §1.5 states only ~40% of coefficients are data-constrained, so on ~60% of modes `sigma_post = sigma_prior` by construction — a field with posterior sd `5.6e-3` everywhere would be the prior-collapse regime K3/Tier-A exists to refuse (measured slope 0.04). New pin **P20**: measure `s_v` on the real anchor and report its **distribution**, stratified interior / partial / off-footprint / above-`z_depth`. Where it survives is narrower and more defensible: the Laplace error is governed by the non-Gaussianity of the **count** term, which lives only on data-constrained voxels — where `sigma_post` is small *because* the counts are large — while the ~60% prior modes contribute exactly zero Laplace error because their posterior *is* the Gaussian prior. Limits II and III are restated as formal expansions in `b_GW s` with the regime of validity quoted from `P20`, not from an inherited scalar |
| 19 | The literature search that decides whether 28 days of work has a paper is gated at PR-11, last; and after removing what the owner already scores as not-novel, differentiator 1 reduces to a gauge convention plus a correctness requirement | **Half RESOLVED, half CONVERTED.** The search moves to PR-0 (D3 item 4). The framing objection is real and is **OWNER DECISION 12**: §4.2 itself calls the mean-one constraint a **gauge fixing** (`prop:gauge`) and `core.py:1372-1380` already enforces same-realization use, so neither is a measurement. The recommendation is to demote differentiator 1 from headline to **enabling clause** and to headline the **unification** (Limits I-III as one likelihood) plus the theta-coupled field — which is also what the owner's own promoted goal says |
| 20 | The critical-path day counts do not sum | **CORRECTED.** Summing the ladder's own durations (1+3+5+5+4+4+2+5 = 29 to PR-6a; +5 = 34 to PR-6b) — and PR-0 now costs 3 d, so **32 d to PR-6a and 37 d to PR-6b**, with K0 able to end it at day 3 |
| 21 | The cited evidence for "the production run loads no LSS table" is a banner that prints identically in runs that did load one | **CORRECTED.** The conclusion holds — `sbatch_ns_joint_sel.sh` carries `--use_lss false` and no `--lss_completion` — but the citation is replaced by the config, not the banner (`h0_scans_1119376.out` prints the same two lines inside the `selq_radial` arm that had just logged an LSS completion load) |
| 22-24 | Three verification notes with no design consequence: `P7c`'s absence from `reviews.md` (Part 3 now records it); the observation that every independently checkable number in v3 reproduced exactly (27.5/49.3 ms, 256 concurrent evals, 20 free dimensions, R12's Q-on deltas, the 47.6/21.7 GB dense-`Phi` sizes, the 190 Mpc kernel arithmetic, the sphere-guard WARN-only docstring); and the confirmation of §0.4's seam diagnosis on `0c5b3db` leg by leg | **NOTED** |

---

## 1. The decisions

### 1.1 Shell-total conditioning of the count likelihood (multinomial, not Poisson)

> **Terminology, fixed once, because the two source documents collide on it.** This plan says
> **"shell-total–conditioned"** for the operation below — conditioning the counts on the per-shell
> totals `T_g`, which is what *removes* `theta`. `OWNER_CONTEXT.md:182-183` says "conditional
> posterior `p(xi | theta, d_gal)`" for conditioning on the *data at a given theta*, which is what
> *restores* `theta`. These are incompatible senses of the same word and an implementer who conflates
> them will build the wrong object. "Conditional" alone is not used in this document.

Factor the count likelihood, per redshift shell `g` — the **factorization** is exact for any joint,
and using only the second factor is a deliberate **partial likelihood** (§0.5, finding 16):

```
p({N_pg}_p | xi, theta) = p(T_g | theta, xi) x p({N_pg}_p | T_g, xi),   T_g = sum_p N_pg   (F3a x F3b)
```

The discarded factor (F3a) is the shell monopole; §4.1's projection removes it to **linear** order,
and it retains `O(b^2 Var_sky(s)/2)` field content. Use only the second factor. Rev 1 wrote the conditional as if `base(z;theta)` cancelled
pointwise. It does not: the exact conditional is

```
pi_pg = f_p Lambda_pg / sum_p' f_p' Lambda_p'g,
Lambda_pg = int_{shell g} base(z;theta) exp(b_gal f(n^_p, z)) dz
```

and `base(z;theta)` does not factor out of the integral (R1-SEV2-4). Rev 2 makes the cancellation
*exact by construction* instead of approximate, by freezing the within-shell weighting into a
stamped operator:

```
pi_pg(xi) = f_p exp(eta_pg) / sum_p' f_p' exp(eta_p'g),
eta_pg    = b_gal * (Phi_s[p] (x) phi~_z[g]) . xi,   phi~_z[g,:] = sum_n W_gn phi_z[n,:]   (1')

log p_count(xi) = sum_g sum_p N_pg log pi_pg(xi)
```

**[v4, §0.5 D1]** Eq. (1') replaces v3's eq. (1), which put the exponential *inside* the shell sum and
therefore made `eta_pg` a log-sum-exp — nonlinear in `xi`, non-separable, and inconsistent with eq. (3)
and with §3.4's cost model. `W` now acts on the **radial basis rows**, so `eta` is exactly linear in
`xi`, `Phi_g = Phi_s (x) phi~_z[g]` is exact, and the neglected term is exactly the within-shell
Jensen residual `(b^2/2)[Var_g(s_p) - <sigma_p^2>_g]` — the same object as `prop:cancel`'s
`eq:cancelresidual`, vanishing iff `e^{b s_p(z)}` is shell-constant. Pins **P5b**/**P5c**.

`W` is the **shell-response operator**, `(G_s, N_fine)`, frozen at the anchor and covered by the
basis sha256. It carries three things at once:

* the within-shell weighting `base(z;theta_ref)` normalized to sum 1 per shell — which is what makes
  (1) *exactly* theta-free rather than approximately so;
* the **photo-z forward convolution** (§1.4) — resolving R1-SEV2-6;
* the shell indicator.

**In v2 the freezing is a switch, not a premise.** `W = W(theta)` is written with `theta` as an
argument from PR-3 onward; **rung 0 evaluates it once at `theta_ref`** and rung 1 evaluates its
*linear response* per proposal (§1.7). The residual `d pi_pg/d theta` at frozen `W` is measured,
bounded and gated (§6.3, pin **P7**) — and P7 is now the **feasibility gate for the promoted
direction**, not a bound on a residual we are ignoring. `f_p in [0,1]` is the per-pixel
*completeness*, not a bare area fraction (§1.2).

`MODEL.tex` Prop. `prop:cancel` gives the exact criterion: `pi_pg` is exactly theta-free **iff** the
base is p-separable within the shell (true whenever `C` is sky-uniform) **and** the field is
shell-constant; otherwise the residual is exactly a within-shell covariance, which vanishes
identically when `e^{b s_p(z)}` is shell-constant. So the frozen-`W` choice is not a modelling
preference — it is a **measurable residual whose size is set by the shell width, and the shell width
is set by the photo-z scatter** (§1.4). That is the cleanest statement of why §1.4 is load-bearing.

Consequences at **rung 0** (unchanged from rev 1 except where noted); what each becomes at rung 1 is
stated inline:

1. **`p(xi | d_gal)` is theta-independent by construction.** No per-proposal solve; `xi_hat` and `H`
   are computed once, offline; the Laplace log-determinant is a constant and drops out.
   *At rung 1:* `xi_hat` moves with `theta` (cheaply, §1.7) while **`H` stays frozen**, so the
   log-determinant is still constant and still drops out. That is not an accident — it is why rung 1
   is affordable and why rung 2 (which un-freezes `H` through the restored Poisson term) is the
   expensive one.
2. **The Q provenance firewall is replaced by a change of likelihood.** `_Q_CONDITIONED`
   (`inference/q_provenance.py:35-51`) exists because the table's base is frozen at build `theta`;
   **[verified]** it forbids sampling `log10n0`, `delta`, `b_miss`, `Om0`, `w0`, `wa` whenever a
   prebuilt `Q` is active, with `H0` alone exempt behind a loud warning and an explicit note that
   "the right fix is to interpolate Q over H0" (`:19-26`). **This is the sharpest code-level argument
   in the repository for the promotion:** today, the literal sentence "every cosmological proposal
   changes the galaxy-field likelihood consistently" is *forbidden by construction* for every
   cosmological parameter except the one it is least valid for. Eq. (1) has no base.
   *At rung 0 the firewall is replaced by a fingerprint* (guard 1 stamps `theta_ref`); only at
   rung 1 does the owner's line become literally true.
3. **The `H0` standard-ruler channel — restated, because rev 2 stated the premise and not the
   consequence.** Review confirmed that `H0` cancels from the count channel: `dV/dz =
   (c/H0)^3 x shape(z;Om0)` and `C_sel` is `H0`-free by the h-firewall
   (`redshift/selection.py:16-27`); `Om0/w0/wa/delta` do **not** cancel. Rev 2 concluded "the
   spurious `H0` ruler is eliminated" and stopped. The consequence it did not draw:
   **shell-total conditioning is precisely the operation that deletes `H0` from the field
   posterior.** The owner's demand is therefore *unsatisfiable for `H0`* at rungs 0 and 1, and at
   rung 2 the restored `H0` content is the volume–density amplitude `n0 (c/H0)^3`, **exactly
   degenerate with `n0`**. §1.8 states what that costs. `ls_z` remains a frozen basis constant at
   every rung (guard 2).
4. **Nested-sampling determinism** — no inner solve, no warm start, no convergence veto, no
   theta-discontinuous KKT pinning, no `logdet` cliff.
5. ~~"the weights are exactly uniform"~~ **WITHDRAWN — this was wrong.** The proposal
   `p(xi|d_gal)` is the target of the *outer* factorization, not of the inner integral
   `int p(d_GW|xi) p(xi|d_gal) dxi`. The inner weights are exactly `w_m = exp(ll_m)` and are
   non-uniform; `core.py:1274` is a plain log-mean-exp of non-uniform weights. This is now the
   single largest open quantitative risk and is handled in §6.5 and OWNER DECISION 5, not by
   assertion.
6. **Prior-draw IS collapse** (`ESS/M = 1e-1303`) remains irrelevant — the draws come from the
   conditional posterior, not the prior. That part of rev 1's claim survives; only the "uniform
   weights" corollary does not.
7. **`c_mode=per_pixel` is refused** (its `C_p` is a plug-in estimator of the same counts —
   circular). `aggregate` and `selection` are admitted *with* the per-pixel completeness of §1.2.

> **OWNER DECISION 1 — now four-way, not binary.** (a) shell-total–conditioned + **frozen** `W`:
> theta-free ensemble, PR-6a, the control arm and the fallback. (b) shell-total–conditioned +
> **linear response**: coupling for `Om0/w0/wa/delta/theta_sel`, **`H0` still absent**, ~+1 ms,
> PR-6b — **recommended, conditional on P7**. (c) unconditional Poisson + linear response: adds `H0`
> through the `n0 (c/H0)^3` budget, requires OD6 resolved and the degeneracy of §1.8 stated in the
> paper, PR-6c. (d) unconditional Poisson + **per-proposal re-solve**: **infeasible** (~1 s against a
> 27.5 ms baseline) — moved to §10 "do not attempt". Conditioning costs the `dN/dz` constraint on the
> budget, restored explicitly as a prior (OWNER DECISION 6), not as a by-product of the field.

### 1.2 Per-pixel selection enters BOTH the count model and the missing budget

Rev 1 put `f_p` in the count model's numerator and left the consumption side at
`dN_miss = (1 - C) dN_exp Q` with a sky-uniform `C` (`redshift/completion.py:1280`, verified
verbatim). Review showed this is a first-order, sign-inverting defect (R1-SEV1-2) and that the
budget-conservation identity rev 1 pinned is not the identity that is consumed (R1-SEV1-3).

**Measured, this session, from `experiments/desi_ingest/data/mth_map_nside128.h5`:**

| quantity | value |
|---|---|
| `mth_eff = min(21, median_m5)` over the occupied footprint | p1 = p50 = p99 = **21.000** |
| `median_m5` | p1 = 22.925, p50 = 23.775 — i.e. imaging depth is nowhere the limit |
| `stratum_edges` | `[19.025, 21, 21, 21, 21]` — the depth strata are **degenerate** |
| `masked_frac` over the occupied footprint | mean 0.1368, sd **0.1039**, p1 0.0470, p50 0.1126, p99 **0.6347** |
| occupied sky, nside 128 / nside 64 | 0.6473 / **0.6199** (30,470 of 49,152) |

So the per-pixel *magnitude depth* is uniform at the retention cut, and the entire per-pixel
selection variation is **areal masking**. That is a mercy: it is achromatic in `z` and in magnitude,
which makes the fix exact rather than a model.

Define, once, at nside 64 by area-weighted degradation of the nside-128 map:

```
f_p = 1 - masked_frac_p        in [0, 1],  f_p = 0 off-footprint
C_p(z; theta) = f_p * C(z; theta)          the per-pixel completeness
```

and use it on **both** sides:

* **count model** — `f_p` in the numerator and denominator of (1). `C(z;theta)` and `nbar(z;theta)`
  are `p`-independent and are absorbed by `W`, so (1) is unchanged in form. Off-footprint pixels
  have `f_p = 0`, `N_pg = 0` and contribute nothing — they are *excluded*, not fed to the likelihood
  as genuine zeros (this was risk R2 in rev 1; it is now structural).
* **consumption** — `dN_miss = (1 - f_p C(z;theta)) dN_exp(z;theta) Q_p(z)`. **[v4: re-ordered to
  lead with the dominant term, §0.5 finding 13.]** The dominant change is **not** the p99 partial
  pixel: it is the **18,682 of 49,152 pixels with `f_p = 0`** (38% of the sky), whose weight moves
  from `(1-C)` to `1` — *every* galaxy there is missing, because there is no catalog there. At
  `C ~ 0.5` that alone moves the all-sky missing budget by roughly **+45%**, and the injection set is
  all-sky, so it lands directly in the selection integral. The partial pixels are the second-order
  correction on top: at `masked_frac = 0.63`, 63% of the galaxies are missing at *all* magnitudes and
  today the code says `(1 - C)` there, understating that pixel's budget by up to ~2x. This is a
  `C`-side change of the missing-galaxy budget's sky distribution *even with the field off*, it is
  **expected to be large**, and it therefore ships behind its own flag (`--per_pixel_completeness`)
  with its own golden and — under v4's non-terminal K8 — potentially as **its own `H0` deliverable**.
  Latent mode **requires** it.
* **budget normalizer** — with exactly the consumption weights (§4.2).

**Sky-average preservation.** `<C_p(z)>_sky = <f_p> C(z;theta)`, so the survey-average completeness
is rescaled by the mean areal completeness, a single stamped number. "C says how much, Q says where"
survives: `Q` carries no completeness.

**Residual.** Any *unmodelled* `p`-dependent, `z`-dependent selection (fibre-assignment
incompleteness is the obvious one) is still absorbed into `xi` and re-enters `dN_miss` with the
wrong sign. Bounded, not eliminated: risk R2, Tier-D stress at the measured amplitude, and the
fallback of OWNER DECISION 4(b).

> **OWNER DECISION 4.** (a) Ship `f_p` on both sides with `C_p = f_p C(z)` — recommended, exact for
> areal masking, requires the depth map promoted out of `experiments/` and a `C`-side golden. Or
> (b) restrict the count channel to a depth- and mask-homogeneous sub-footprint
> (`masked_frac in [0.05, 0.20]`, ~70% of the occupied area), where the cancellation is exact
> without any `C`-side change — trivially correct, loses area, and is the safe fallback if the mask
> is not trusted. Do not ship (c) "ignore it": rev 1's version was (c).

### 1.3 The GW data never enter the `xi` solve — and that is exact

```
log p(d_GW, d_gal | Lambda, theta)
  = log p(d_gal | theta) + log int dxi p(xi | d_gal) p(d_GW | d_gal, xi, Lambda, theta)
```

Draws are needed from `p(xi | d_gal)` only — never `p(xi | d_gal, d_GW)`, and never a gradient of
the GW likelihood with respect to `xi`. This kills implicit differentiation through a Newton solve,
the `lax.optimization_barrier` conflict, and the member-leaf memory problem.

The only approximation is Gaussianity of `p(xi | d_gal)`. **[v4, §0.5 finding 18 — the inherited
`s_v ~ 5.6e-3` is WITHDRAWN as a global certificate.]** At `amp = 1` the *prior* sd of `s` is ~1, and
§1.5 states only ~40% of coefficients are data-constrained, so on ~60% of modes
`sigma_post = sigma_prior` by construction. The honest statement is narrower and stronger: the Laplace
error is governed by the non-Gaussianity of the **count** term, which lives only on data-constrained
voxels — where `sigma_post` is small *because* the counts are large — while the prior modes contribute
exactly zero Laplace error because their posterior *is* the Gaussian prior. Pin **P20** measures
`s_v`'s distribution on the real anchor, stratified interior / partial / off-footprint /
above-`z_depth`, and every downstream expansion (Limits II and III) quotes that distribution rather
than a scalar. This is *not* the estimator-bias question — that is §6.5 and is a different, larger
number.

### 1.4 Photo-z is inside `W`, not absorbed by shell width

Measured this session over 2.24M galaxies from
`experiments/desi_ingest/data/pixelated_n64/catalog_pixelated_nside_64.h5`:
`dz` p50 = **0.0227**, p75 = 0.0398, p99 = 0.0900; `dz/(1+z)` p50 = 0.0188; `z` p1/p50/p99 =
0.0604/0.2371/0.2990. At `z_ref = 0.237` (`H0 = 67.74`, `Om0 = 0.3089`), `dchi/dz = 3918 Mpc`, so
`sigma_chi = 89 Mpc`.

Rev 1's cheap option ("widen shells past the scatter") is **unavailable**: the current build already
has `Delta_z = 0.024 ~ sigma_z`, and widening drives `G_s <= 12` against `M_z` (R1-SEV2-6). Rev 2
does the correct thing instead and it costs one offline GEMM: **convolve the model, not the data.**

```
W[g, n] = ( int_{shell g} dz  K(z | z_n) ) * base(z_n; theta_ref) * Delta_n,   rows normalized to 1
K(z | z_n) = N(z; z_n, sigma_z(z_n))       population-average photo-z kernel
```

The counts stay integer (`np.histogram(zgals, bins=edges)`, verified at
`cli/build_lognormal_completion.py:750/762`), the multinomial stays exact, and the radial attenuation
`~ exp(-k^2 sigma_chi^2 / 2)` is in the **forward model**, so it does not bias `b_gal` and is not
absorbed into `xi`.

The attenuation is what forces OWNER DECISION 3: at a 50 Mpc radial kernel it is
`exp(-(89/50)^2/2) = 0.20` — the count channel would measure 20% of the radial signal it claims. At
190 Mpc it is `exp(-(89/190)^2/2) = 0.90`.

`sigma_z(z)` is a population average, not per-galaxy. Upgrade path (per-galaxy kernels) is gated by
a measured difference in `xi_hat` (pin P8); ship the average unless it fails.

### 1.5 The kernel must be isotropic at a scale the data can carry

`SurveyParams` defaults are `lss_corr_length_mpc = 50.0` and `lss_corr_length_ang = 0.2` chordal rad
(`core/types.py`). `ls_ang = 0.2 rad = 11.5 deg`; at `chi(0.237) ~ 950 Mpc` that is **190 Mpc
transverse** against **50 Mpc radial** — a 4:1 pancake the design was describing as "the 50 Mpc
correlation length" (R1-SEV2-10). Three self-consistent configurations:

| `L_smooth` | `ls_ang` (chordal) | `ls_z` (zeta) | `M_sph` (guard) | `M_z` (guard) | `M` | `H` (f64) | photo-z retention |
|---|---|---|---|---|---|---|---|
| 50 Mpc isotropic | 0.053 | 0.0103 | **4,470** | 27 | 120,690 | 117 GB | 0.20 |
| 190 Mpc / 50 Mpc (rev 1, anisotropic) | 0.200 | 0.0103 | 315 | 27 | 8,505 | 579 MB | 0.20 |
| **190 Mpc isotropic (rev 2)** | **0.200** | **0.039** | **315** | **8-12** | **2,520-3,780** | **51-114 MB** | **0.90** |

`ls_z` from `L/(dchi/dz)/(1+z_ref) = 190/3918/1.237 = 0.0392`; `M_z >= ceil(log1p(0.30)/ls_z) + 1 = 8`
(take 12 for margin); `M_sph >= 4pi/ls_ang^2 = 315`. Constrained-mode count at the rev-2 config:
`0.62 x 315 = 195` angular x `(log1p(0.299)-log1p(0.060))/0.0392 = 5.2` radial `~ 1,015` of 2,520,
i.e. **40% of the coefficients are data-constrained**, versus ~12% at rev 1's rank. The rest are
prior and remain prior — which is correct, and is what the Laplace covariance encodes.

Also: **the sphere side of `_gp3d_resolution_guard` only WARNS** — its docstring says so verbatim
(`cli/build_lognormal_completion.py:274-319`: "The sphere side only WARNS"). Rev 1 claimed it
"demands `M_sph >= 315`". It demands nothing. Latent mode installs a **hard** sphere guard (§4.4
guard 3).

> **OWNER DECISION 3.** Adopt the isotropic 190 Mpc kernel (recommended): it is the scale the
> angular rank can afford, the scale photo-z does not destroy, and it makes the "correlation length"
> in the writeup true. Cost: the analysis is explicitly a ~190 Mpc-smoothed density field, and the
> 50 Mpc fiducial in `SurveyParams` is retired for latent mode. Alternative: keep 50 Mpc radial and
> accept a 4:1 anisotropic kernel measuring 20% of its own radial signal — not recommended, and if
> chosen it must be stated as such in the paper.

### 1.6 The three limits — one hierarchy, three known analyses

Rev 2 contained **nothing** on this. It is the owner's "unification payoff"
(`OWNER_CONTEXT.md:163-176`) and it must be math, not a paragraph. Full treatment in `MODEL.tex`
§`sec:limits`; the operative statement here.

Write the per-event redshift/sky prior in this plan's own objects:

```
p(z, n^ | Lambda, theta, xi) ∝  sum_i w_i K(z|z_i) delta(n^ - n^_i)              [observed hosts]
                              + (1 - f_p C(z;theta)) dN_exp(z;theta) Q_p(z;xi)   [missing hosts]
   normalized by Z(theta, xi),   Q_p(z;xi) = exp(b_GW f(p,z;xi) - rho(z;c,b_GW)),
   f = Phi xi,   xi ~ N(0, I)
```

**Limit I — complete catalog** (`C -> 1`, `f_p -> 1` over the footprint). The missing branch vanishes,
the prior collapses to the weighted host spikes, and **`xi` drops out identically, for every `xi`**.
Therefore `logsumexp_m ll_m - log M = ll` *exactly*, member by member. Ordinary discrete-host dark
sirens. This is a physics identity, not a routing property — which is why it becomes a **stronger
inertness pin than P12** (new **P16**, §6.3).

**Limit II — the present architecture** (`0 < C < 1`). Both terms live; the field modulates only the
second; the marginalization is numerical because `b_GW f` is not uniformly small *inside the
selection integral*.

**Limit III — cross-correlation.** **[v4, corrected trigger and derivation — §0.5 finding 17.]** The
trigger is *not* `Lambda_cat -> 0` (an empty catalog has no `delta_g`, so the `b_GW b_gal` cross term
it is supposed to produce would vanish identically): it is **the catalogued fraction of each event's
prior mass tending to zero while the localization spans many correlation cells**. The galaxy counts
are *retained as data* — they are what carries `b_gal`. Then

```
ll_i(xi) = log int dz dn^  p_i(z,n^|d_i) w(z;theta) e^{b_GW (Phi xi)(n^,z)}  -  log Z(theta, xi)
```

Expand to first order in `b_GW Phi xi` — a **formal** expansion whose parameter is `b_GW s` over the
events' support, **measured by P20**, not certified by the inherited `s_v ~ 5.6e-3` (§0.5 finding 18
withdraws that number as a global certificate) — and marginalize `xi ~ N(0,I)` analytically **against
the count factor and the GW factor jointly**, which is what produces all three pairings:

```
a_GW = b_GW sum_i Phi_i,   a_gal = b_gal sum_{p,g} (N_pg - T_g pi_pg) Phi_pg

log int dxi N(xi;0,I) exp[ (a_GW + a_gal) . xi ]
     = (1/2)||a_GW||^2   +   a_GW . a_gal   +   (1/2)||a_gal||^2
```

**Marginalizing the Gaussian latent field replaces the field by the kernel.** Every pair of
field-coupled objects contributes `b_a b_b K(x_a, x_b)`:

* event × event → the GW auto-correlation;
* event × galaxy voxel → `sum_i sum_{p,g} N_pg K(x_i, x_{pg})` — **the GW–galaxy cross-correlation
  proper**, carrying the bias combination `b_GW b_gal`;
* voxel × voxel → the galaxy auto-correlation (i.e. the count likelihood itself).

In harmonic space `K_sph = sum_l (2l+1)/(4pi) C_l P_l(n^·n^')`, so with the per-event localization
windows `W^i_l` acting as tomographic bins the cross term is `sum_l (2l+1)/(4pi) W^i_l W^g_l
C_l^{GWxg}` — **Cheng & Gair's statistic, recovered as a limit of this likelihood rather than
imported as a separate method.**

**The unification statement, as math.** One generative hierarchy. Catalog sirens are the `C -> 1`
corner where only the spikes survive. Cross-correlation sirens are the `C -> 0`, weak-`b` corner
where analytic marginalization of `xi` leaves only the pairwise kernel sum. The present architecture
is the interior, where both terms are present and the marginalization is numerical. **Which corner a
given catalog/GW pairing sits in is decided by two measurable numbers** — `osc_theta Delta logL_1`
(§0.5 eq. 0c) and `osc_theta Delta log L_gal` (eq. 0d) — not by preference. *(v4: v3 cited
`prop:compressexact` here; that proposition is a statement about importance weights under design 1b,
which §0.3 forbids, so under the shipped design 1a it can never fire — §0.5 finding 4.)*

**Two scope statements that must ride with any unification claim.** (i) Limit III is a property of
**the hierarchy**, not of the estimator PR-6a/6b ships: the shipped architecture conditions on
`d_gal` first, so the galaxy counts are absorbed into `xi_hat` and `H` and the cross term is not a
separate object of the estimator. (ii) At **rung 0** that conditioning happens at *frozen theta*,
which removes exactly the cosmology dependence of `C_l^{GWxg}` that a cross-correlation dark-siren
measurement uses — **which is an argument for the promotion, not against the unification**: rung 1 is
what restores it.

**What this buys immediately: two pins the plan did not have** (§6.3): **P16**, the complete-catalog
limit (bit-identity per member at `C ≡ 1`), and **P17**, the Gaussian-marginalization limit — at
small `b_GW`, `logsumexp_m ll_m - log M` must reproduce the analytic `(b_GW^2/2) sum_{ij} K(x_i,x_j)`
within MC error. **P17 is the only proposed test in this document that validates the marginalization
estimator against a closed form**, i.e. against *truth* rather than against `M_draw = 256`. It
attacks §6.5 — the plan's largest open number — from an independent direction.

### 1.7 Linear response: the theta-coupled ensemble (the promoted design)

Everything here is built from objects this plan already defines. §3.4 already computes exactly this
derivative for **one** parameter, under the name `bias_sensitivity`. Rung 1 is that construction with
`b -> theta`.

**Offline, once (PR-3/PR-4):**

```
S = d xi_hat / d theta = - H^{-1} (d grad / d theta) |_(xi_hat_ref, theta_ref)   in R^{M x n_theta}
```

`n_theta` extra triangular solves against the **same** `H_chol` §3.4 already factors. Storage
`M x n_theta x 8 B` = **121 kB** at `M = 3780`, `n_theta = 4`.

**Per proposal, online:**

```
xi_m(theta) = xi_hat_ref + S·Δtheta + L_H^{-T} g_m                      (rung 1a, CRN)
```

The correction `S·Δtheta` is **member-independent**, and that single fact saves the entire cost
model:

* **`row_fac` stays theta-free and static.** §2.4's "0 transient" survives — the 256x concurrency
  multiplier still does not apply.
* One extra **member-independent** row factor `row_fac_shift(theta) = reshape(S·Δtheta,
  (M_sph, M_z))` is contracted through the same seam: **≈ one member's worth of the seam, ~1 ms**
  against §2.3's measured 2.65 → 4.30 ms (PE) / 2.66 → 4.28 ms (selection). **To be measured at PR-6b,
  not banked.**
* The sky moments `(A_m, B_m)` of §2.2 acquire theta dependence **only through the same projection**:

  ```
  A_m(z;b,theta) ≈ A_m^ref(z;b) + b (∂A_m/∂theta_j) Δtheta_j,
  ∂A_m/∂theta_j = sum_p e^{b f_m^ref(p,z)} (Phi_p · S_j)             (theta-free, precomputable)
  ```

  Storage `2 x M_draw x n_b x N_z_sub x n_theta x 8 B` = **1.1 MB** at `M_draw = 8, n_b = 33,
  N_z_sub = 64, n_theta = 4` (8.7 MB at `M_draw = 64`). **This is the load-bearing result: without
  the projection, promotion resurrects R2-SEV1-2 in full** — the 25.6 ms f64 / 13.1 ms f32 full-sky
  reduction returns to the per-proposal path at +52–93% of a 27.5–49.3 ms baseline.

**The galaxy-side evidence — CORRECTED IN v4 (§0.5 D2); v3's version of this paragraph was wrong.**
The rung-1 galaxy-side term is the **Laplace evidence**, not the count log-likelihood:

```
log L_gal(theta) = - J_theta(xi_hat_theta) - 0.5 log det H(theta)
                 = l(xi_hat_theta, theta) - 0.5 ||xi_hat_theta||^2 - 0.5 log det H(theta)     (5)
```

v3 shipped only `l` and dropped `∂xi_hat/∂theta` "by the envelope theorem". The envelope theorem
removes `∂xi_hat/∂theta` from `J`, **not** from `l`: stationarity gives `∂l/∂xi = xi_hat`, not `0`
(`MODEL.tex` `eq:laplacegrad` states the correct rule and v3 cited it for the wrong conclusion). The
omitted Occam term `xi_hat^T S Δtheta` reaches **~2e2 nats worst case, ~3 nats at random alignment**
against a 0.1-nat budget. Under linear response it is **free**:

```
0.5||xi_hat_theta||^2 = 0.5||xi_hat_ref||^2 + (S^T xi_hat_ref).Δtheta + 0.5 Δtheta^T (S^T S) Δtheta
```

— one offline `n_theta`-vector and one `n_theta x n_theta` matrix. **`S` is needed for both the
GW-side field shift and the evidence term.** The log-det is no longer assumed constant: **P7d**
measures its theta-oscillation at the same 20 solves, and if it exceeds 0.1 nat the rung carries the
linear-response term `0.5 tr(H^{-1} dH/dtheta).Δtheta` (also an offline `n_theta`-vector).

**Why linear response beats a re-solve on more than cost.** It is *analytically smooth in `theta`* —
no solver, no convergence tolerance, no iteration count, no history. **Kill criterion K6 (adjacent-
theta `ΔlogL` jump > 0.1 nat) becomes unfireable by construction**, whereas a per-proposal Newton
solve does not have that property and K6 exists precisely because it might fire.

**Where it is exact and where it is not.** Exact to first order in `Δtheta` across the prior;
`H(theta) ≈ H_ref` is a second-order neglect (a covariance misspecification, not a mean error).
**P7 measures exactly this** and the thresholds are in §0.3.

**Multitracer shaping — the owner's "K>=2 must be adding terms, not refactoring".** `S` is built from
the **stacked K-tracer objective from day one**, with each catalog's own selection theta
(`M0hat_c2`, `sigma_M_c2`, …) contributing its **own column**. Then K>=2 adds columns to `S`, not a
rewrite. Concretely: `latent_counts.bias_sensitivity` generalizes to
`sensitivity(xi_hat, H_chol, wrt=...)` returning columns of `S`, and `d xi_hat/d b` becomes one of
them (§3.4). **This interface is binding from PR-3 even though PR-7 is the rung that uses it.**

### 1.8 The `n0`–`H0` degeneracy — what rung 2 actually costs

Un-conditioning restores the shell-total factor, and the restored channel is

```
T_g(theta) ∝ n0 · (c/H0)^3 · int_shell C_sel(z;theta_sel) (1+z)^delta shape(z;Om0,w0,wa) dz
```

so **the entire `H0` content of the count channel is the volume–density amplitude `n0 H0^{-3}`,
which is exactly degenerate with `n0`.** It is identified only if `n0` carries an `H0`-independent
prior. It does not.

**[verified]** `experiments/desi_ingest/calibrate_n0.py:86` computes `n0 = w.sum() / (f_sky *
budget_int)` with `budget_int = np.trapz((csel * evo * dv)[zmask], zf[zmask])` — the comoving volume
element `dv` evaluated at the **fiducial cosmology**. `data/n0_calibration.json` records
`V_c_Mpc3 = 7.818e9`, `f_sky_occupied = 0.6199`, `sum_weights = 22,787,566`, `log10n0 = -2.3996`,
`delta = 0.9402`. `experiments/desi_full259/sbatch_ns_joint_sel.sh` passes that number through
`--fixed_parameter_values`.

**Therefore:** pinning `log10n0` and then sampling `H0` against an *unconditional* count likelihood
**manufactures a standard-density `H0` constraint out of the fiducial cosmology.** That is the same
class of defect as the `ls_z`-in-Mpc standard ruler that guard 2 (§4.4) exists to make
unrepresentable — and it is **larger**, because it acts against 22.79M galaxies rather than a kernel
width.

**The honest statement, which belongs verbatim in the paper and in any proposal:**

> The galaxy field constrains the shape parameters (`Om0, w0, wa, delta`) and the tracer biases. It
> constrains `H0` only through the volume–density amplitude, which is degenerate with the mean
> comoving number density; that degeneracy is why `n0` is a nuisance and why its prior must be
> `H0`-aware rather than a fiducial plug-in.

**Consequences.** OWNER DECISION 6 moves from *advisory* to **load-bearing**: under rung 2,
`log10n0` cannot be a plug-in at all, and the calibration prior must be re-derived as a function of
`H0` — or `n0` reparameterized to an `H0`-invariant combination such as `n0 h^{-3}`. Risk **R8** is
re-sized **medium-high → SEV1 under promotion**, and new kill criterion **K10** refuses the
unconditional arm rather than publish it.

### 1.9 What this work is a merge of (and what is structurally missing)

Four repo facts that reframe the effort and belong in the proposal's "what we do now" section.

* **An in-likelihood, sampled, whitened low-rank sphere × z GP already exists — on the GW side.**
  `_SphereZGPBase` (`darksirens/sky/models.py:273-379`) samples `M = M_sph x M_z = 192` whitened
  latents `sky_xi_i` with `prior_kind="normal"`, **plus three sampled hyperparameters**
  `sky_log_amp` / `sky_log_ls_sphere` / `sky_log_ls_z` (`:311-332`); `OverdensityGP3D` (`:395-422`)
  is already a comoving-volume-normalized 3-D over/under-density. The offline `Q` builder
  deliberately reuses that exact kernel and inducing geometry
  (`redshift/lognormal_completion.py:583-632`, `:610-632`). **The field-level upgrade is a merge of
  an existing sampled latent block with an existing Poisson-lognormal count term — not a new
  sampler.** Say it that way; it is both true and the strongest feasibility argument available.
* **The `Q` channel's own GP hyperparameters are structurally unsampleable.** `lss_corr_length_mpc`,
  `lss_sigma`, `lss_corr_length_ang` live on `SurveyParams` (`core/types.py:147-149`) but are
  **[verified]** absent from `SURVEY_PARAMS_FID_BY_NAME` (`core/constants.py:20-45`), the registry
  the decoder and the prior both address by name. Latent mode keeps them **frozen basis constants**
  (guard 2 forbids the Mpc spelling outright); sampling them is out of scope at every rung in this
  plan and would require a registry change, not a flag.
* **`b_miss` is the tracer-bias seam.** A `Q`-active catalog *drops* `b_miss` from the sampled block
  because `Q` **replaces** the local overdensity factor (`inference/loaders.py:229-247`,
  `cli/inference.py:2637-2676`, `inference/q_provenance.py:47-51`). Moving the latent in-likelihood
  brings `b_miss` back as the per-tracer bias — which is precisely the quantity the joint builder
  applies offline as `Phi'_k = b_k Phi_k` (`build_joint_lognormal_completion.py:239-267`). §4.3's
  `b_miss -> b_GW` inversion is therefore a restoration, not an invention.
* **Only the gp3d family can move in-likelihood.** The gp3d solve is a convex GLM in the whitened
  latents with an `M x M` Newton Hessian and Armijo backtracking
  (`lognormal_completion.py:660-864`) — one `M x M` solve, not a per-pixel loop. The radial builder
  is a **per-pixel L-BFGS-B with a 200,000-iteration cap** (`:246-415`) and cannot move
  in-likelihood at all. Since the production DESI table is **radial**
  (`experiments/desi_full259/data/fits/q_radial.h5`), adopting the latent path *is* a change of `Q`
  family — which is what OWNER DECISION 10 is about, now with a structural reason attached rather
  than a preference.

---

## 2. Cost and memory — measured, not modelled

Rev 1's cost section was wrong in every term. Review measured the corrections on this machine's
H100 NVL; rev 2 adopts them and then removes the dominant term by design.

### 2.1 Baseline

Rev 1 asserted "3-20 s/call (report 4)". The repo measures **27.5 ms/call** single-pass and
**49.3 ms/call** under a mis-chosen block plan, at the identical production dimensions
(`N_sel = 1,067,946`, `N_events = 259`, `n_samp = 4096`, `n_q = 200`) —
`docs/source/performance.md:104-116`. `factory.py` records "7.7 ms of 30.1 ms for the spectral
single pass, 13.8 of 51.3 ms at the production auto plan, 7.8 of 17.9 ms for a dark-siren mock."
**The 3-20 s figure is withdrawn.**

**PR-0 gate (new, blocking):** measure the dark-siren production baseline with
`scripts/profile_member_marginalization.py` at `M_draw in {1, 8, 32, 64}` before any latent code
lands. Every percentage below is provisional against 27.5-49.3 ms until PR-0 reports.

### 2.2 The full-sky reduction leaves the per-proposal path (exactly)

Rev 1's dominant term was `rho_m`, the per-shell budget renormalizer, measured by review at
**25.6 ms (f64) / 13.1 ms (f32)** for one step of five, essentially independent of `M_sph`. Rev 2
removes it in closed form.

The normalizer with the *consumption* weights of §1.2 is

```
rho_m(z; theta) = log[ sum_p w_p(z;theta) e^{b f_m(p,z)} / sum_p w_p(z;theta) ],
w_p(z;theta) = (1 - f_p C(z;theta)) * dN_exp(z;theta) * Omega_pix
```

`dN_exp` is `p`-independent and cancels. `C(z;theta)` enters only through the **scalar**
`c = C(z;theta) in [0,1]`. Expanding `w_p = 1 - f_p c` (up to the cancelled constant):

```
rho_m(z; c, b) = log[ (A_m(z;b) - c B_m(z;b)) / (P_F - c F_F) ]
A_m(z;b) = sum_{p in F} e^{b f_m(p,z)}    B_m(z;b) = sum_{p in F} f_p e^{b f_m(p,z)}
P_F = |F|                                  F_F = sum_{p in F} f_p                     (2)
```

**[v4, §0.5 finding 8 — the sums run over the FITTED FOOTPRINT `F`, not the full sky.]** v3 wrote
these as all-sky reductions, which contradicts both the `Q == 1` off-footprint convention and
`renormalize_q_mean_one`'s own docstring ("summed over the FITTED FOOTPRINT ... never the full sky:
out-of-footprint pixels' homogeneous budget must not absorb the footprint's monopole"). Splitting
eq. (4) at `f_p = 0` shows the off-footprint block (weight `1`, `Q = 1`) is conserved trivially, so
the constraint binds on the footprint block alone. **The seam therefore needs an explicit index map**
from the gathered pixel space (the PE/injection union, 49,143 of 49,152 rows) to footprint rows
(30,470), with off-footprint rows returning bit-zero `logQ` — pin **P13b**. Without it the seam
returns `Q = exp(-rho) != 1` off-footprint and P13 fails.

**`A_m` and `B_m` are theta-free.** They are two full-sky reductions, computed **once offline** on a
grid of `b_GW` nodes and stored; `c` enters in closed form online. The same decomposition collapses
the field-weighting global `log Z_k` (computed per row-class — occupied rows via the existing
row-wise path, empty rows via (2) over the empty subset; `redshift/completion.py:1792-1800` is the
existing empty-sum seam).

Storage: `2 x M_draw x n_b x N_z_sub x 8 B`. `N_z_sub` is the ~60-70 grid nodes with `z <= z_depth`
(above `z_depth`, `Q == 1` by the existing relaxation, `completion.py:1295`). At `M_draw = 64`,
`n_b = 33`: **2.4 MB**. Interpolation in `b_GW` only, on a Chebyshev grid, pinned to 1e-6 (P9).

**Consequence:** the 25.6 ms step and the "invisible at K=1 / recomputed at K>=2" contradiction
review flagged (R2-SEV2-14) both disappear, at K=1 *and* K>=2, whether or not `b_GW` is sampled.

### 2.3 What is left per proposal

Only the seam. Review measured the `prior.py` gather swap at production scale
(`N_PE = 1,060,864`, `N_SEL = 1,067,946`, `M_draw = 8`): `member_logq[pix,idx]` **2.65 ms ->
`row_fac[pix] @ phi_z[idx]` 4.30 ms** on the PE side and 2.66 -> 4.28 ms on the selection side, i.e.
**+3.3 ms combined, 1.6x on a step that is 5.3 ms of the baseline.**

`eval_dark_member_completion` and the per-event/selection reductions are the *only* member-vmapped
work — `core.py:960-971` documents that the population model, the `z(dL)` inversion, the proposal
reweighting, the sky factor and the observed-catalog KDE are hoisted and computed once. So `M_draw`
scales a small, well-identified slice of the likelihood.

| `M_draw` | member-dependent seam, table | member-dependent seam, latent | latent − table | **latent vs the no-LSS production baseline (27.5 ms)** |
|---|---|---|---|---|
| 8 | 5.3 ms | 8.6 ms | +3.3 ms = +12% | **+8.6 ms = +31%** |
| 32 | 21 ms | 34.4 ms | +13 ms = +47% | **+34.4 ms = +125%** |
| 64 | 42 ms (13.7 GB cube) | 68.8 ms (94 MB) | +26 ms = +95% | **+68.8 ms = +250%** |

**[v4, §0.5 finding 10.]** The production baseline (`--use_lss false`, no `--lss_completion`) contains
**zero** member-dependent seam work, so the deliverable is not "a table-marginalized run made latent",
it is "a non-marginalized production run made latent". The `latent − table` column is the right number
for a table-vs-latent comparison and is kept; **OWNER DECISION 5 and kill criterion K4 are taken on
the last column**, and PR-0 reports all three.

`M_z = 12` shortens the dot 2.25x versus the `M_z = 27` review measured; the step is gather-bound so
this is not banked. **The `M_draw` row is chosen by OWNER DECISION 5, and it is the dominant cost
lever in the whole plan.** Note the table column is not a real alternative above `M_draw = 8`: at
64 the cube is 13.7 GB against 72.7 GiB free with 10.4 GiB already static.

### 2.4 Memory — corrected

Every latent array is **theta-free**, so it is built in `make_likelihood`, `barrier()`-ed like every
other `EMCatalog` leaf (`factory.py:239-282`), and is **static, not per-evaluation**. The 256x
concurrency multiplier that review correctly applied to rev 1's design (R2-SEV1-3) does not apply to
rev 2's.

| array | shape | dtype | `M_draw=8` | `M_draw=64` |
|---|---|---|---|---|
| `phi_sph` | `(49152, 315)` | f64 | 124 MB | 124 MB |
| `phi_z_out` | `(N_z_sub, 12)` | f64 | 7 kB | 7 kB |
| `row_fac` (footprint rows only) | `(M_draw, 30470, 12)` | f32 | **11.7 MB** | **93.6 MB** |
| `A_m, B_m` tables | `(2, M_draw, n_b, N_z_sub)` | f64 | 0.3 MB | 2.4 MB |
| `latent_counts`, `f_p` | `(30470, G_s)`, `(49152,)` | f32 | 4.1 MB | 4.1 MB |
| **total resident added** | | | **~140 MB** | **~225 MB** |
| per-evaluation transient added, **rung 0** | | | **0** | **0** |
| per-evaluation transient added, **rung 1** (`row_fac_shift`, `(30470, M_z)` f32) | | | **1.46 MB** (~375 MB at 256 concurrency) | **1.46 MB** (~375 MB) |

**[v4, §0.5 finding 9.]** v3's categorical "per-evaluation transient added: 0" is true at rung 0 and
**false at rung 1**: `row_fac_shift` is consumed as `row_fac_shift[pix]`, so it is the row expansion
`Phi_sph · reshape(S·Δtheta, (M_sph, M_z))` of shape `(30470, 12)`, not the `(315, 12)` object §1.7
prices. 375 MB against 72.7 GiB free is not fatal, but it must be **reserved**: rung 1 routes through
the **guarded transient branch** (`_slopes_and_fixed:708`, `batch_scale` at `:725`), not the static
branch. The ~34 GB under-reservation precedent at `block_sizing.py:623` is why this is not assumed.

**The "-3.6 GB freed" claim is withdrawn.** The production run loads no LSS table — established by
its **configuration**, `sbatch_ns_joint_sel.sh` carrying `--use_lss false` and no `--lss_completion`,
**not** by the "Non-LSS run. Creating memory-efficient dummy (1, 1086) grid" banner v3 cited: that
banner belongs to the `--use_lss` overdensity field and prints identically inside the `selq_radial`
arm of `h0_scans_1119376.out` that had just logged an LSS completion load (§0.5 finding 21). So
there is nothing to free. The honest delta is **+140 to +225 MB static, +0 transient**. Rev 1's
comparison against the table cube was also not like-for-like: the existing member leaf is a *view*
of a resident constant (`core.py:988-999`), not a fresh allocation.

`block_sizing`: because the leaves are genuinely factory-static, `estimate_pending_static_bytes`
(`block_sizing.py:599`) and `measure_static_state_bytes` (`:649`) are the right functions — but the
routing must be justified, not assumed, so PR-5 also adds a **guarded transient branch** in
`_slopes_and_fixed` (`:708`, `batch_scale = concurrent_evals` at `:725`) that fires if anyone ever
enables a per-proposal latent recompute. Latent + recompute is refused until that branch is
measured. This is the direct answer to R2-SEV1-4 and to the ~34 GB under-reservation precedent
(`block_sizing.py:623`).

### 2.5 Cost under promotion — what survives, and what the two designs cost

**The entire cost case for promotion rests on linear response.** A reader must not be able to
conclude that theta-coupling is affordable in general. It is not.

| §2 claim | under a naive re-solve | under linear response (§1.7) |
|---|---|---|
| §2.2 `(A_m,B_m)` theta-free; the 25.6 ms full-sky reduction leaves the per-proposal path | **DIES** — the moments depend on `xi(theta)`; +25.6 ms f64 / +13.1 ms f32 = **+52–93%** on a 27.5–49.3 ms baseline | **SURVIVES** via the projected `∂A/∂theta`, `∂B/∂theta` tables (~1.1 MB at `M_draw=8`) |
| §2.4 latent leaves are static; per-evaluation transient = 0 | **DIES** — `row_fac` becomes per-evaluation; at 256 concurrency 11.7 MB → **3.0 GB** (`M_draw=8`) and 93.6 MB → **24 GB** (`M_draw=64`), against 72.7 GiB free with 10.4 GiB already static. **R2-SEV1-3 / -4 reinstated as live** | **SURVIVES** — the theta correction is member-independent and `(M_sph x M_z)`-sized |
| §2.4 the `block_sizing` **static** branch is the right target | **DIES** — the guarded transient branch (`_slopes_and_fixed:708`, `batch_scale` at `:725`) becomes the *primary* accounting path | **SURVIVES**, with the transient branch still shipped as the guard rail |
| §3.4 the anchor build is "minutes" | irrelevant: the *per-proposal* solve is ~1 s, a **~36x** wall against 27.5 ms | **SURVIVES** |
| OD5's `M_draw` arithmetic (+12% at 8 … +95% at 64) | **DIES** — the evaluation becomes memory-bound before it is compute-bound | **SURVIVES unchanged** |

**Two engineering items disappear from the promoted rung, verified.** Rev 2's PR-9 required
implicit-diff `custom_vjp` and pin **P15** (implicit-diff gradient vs finite differences) — the
single riskiest item in the document. The production sampler **does not differentiate the
likelihood**: `likelihood/block_sizing.py:294-302` states "*Only NumPyro NUTS does; dynesty and
tinyns are gradient-free*", and the production banner is `Peak model: value-only (tinyns, 256
concurrent evals)` (`logs/ns_joint_sel_1119811.out`). `custom_vjp` and P15 are **deleted** and
re-added only behind an explicit NUTS path.

**Rung 1's added per-proposal cost, to be measured at PR-6b:** one member-independent row-factor
contraction through the existing seam (~1 ms, i.e. ≈ `1/M_draw` of §2.3's member-dependent slice) plus
a rank-`n_theta` GEMV per moment table (negligible). Rung 1's added **static** memory over rung 0:
`S` (121 kB) + the projected moment derivatives (1.1 MB at `M_draw=8`, 8.7 MB at 64).

---

## 3. Model, algorithm, architecture

### 3.1 The joint likelihood

```
p(d_GW, d_gal | Lambda, theta)
  = int dxi  p(xi)                            (F1)  N(0, I_M) whitened prior
           * p({m_i} | {z_i}, theta_sel)      (F2)  magnitude channel  -- EXISTS
           * p({T_g} | theta)                 (F3)  shell totals -- DROPPED at rungs 0/1 (§1.1),
                                                    restored as a prior (OD6); RESTORED as a
                                                    likelihood factor only at rung 2 (PR-6c)
           * prod_k p({N_pg} | T_g, xi, b_k)  (F4)  angular counts -- NEW, eq. (1);
                                                    theta-free at rung 0, theta-live at rung 1
           * p(d_GW | d_gal, xi, Lambda, theta) (F5) the existing GW likelihood
```

**Double conditioning — why (F4) and (F5) may both be used without double-counting the same
galaxies.** This is the one structural objection a referee will raise, and it has an exact answer
(`MODEL.tex` Thm. `thm:double`). Conditional on `xi`, catalogued and missing galaxies are
**independent thinned Poisson processes**, so `p(U | C, xi) = p(U | xi)`: the chain rule
`p(C, d_GW | xi) = p(C | xi) · p(d_GW | C, xi)` is exact with no double use — **provided the missing
intensity is a function of `xi` alone and never of `C`**. Two named ways to break it, both live here:

1. **The empirical-Bayes plug-in `xi_hat(C)`** — i.e. today's `Q` table. It is safe *only* because
   the current code never multiplies by a count likelihood. **Adding (F4) without promoting `xi` to a
   latent variable is the most dangerous half-measure available in this plan** and must never be
   shipped as an intermediate state.
2. **The monopole, squared.** `experiments/desi_ingest/calibrate_n0.py` fits `(n0, delta)` to the
   *same counts* and pins them via `--fixed_parameter_values`. Keeping that calibration prior **and**
   an unconditional Poisson count likelihood (rung 2) double-counts the budget. Exactly two
   configurations are admissible: **shell-total–conditioned + calibration prior** (rungs 0/1), **or**
   **unconditional Poisson + a genuinely independent `n0` prior** (rung 2, and see §1.8). Guard 5
   (§4.4) enforces the first; kill criterion K10 refuses a rung-2 run that violates the second.

**[v4 — factor labels now follow `MODEL.tex`: (F1) prior, (F2) magnitudes, (F3) counts, (F4) GW. The
shell-total split above is (F3a) monopole / (F3b) placement. §0.5 finding 15.]**

**(F2) IS NOT A LIKELIHOOD FACTOR IN THE SHIPPED CODE — v4, §0.5 finding 5.** **[verified this
session]** `redshift/selection.py:341` `magnitude_loglike_from_stats` is referenced nowhere outside
`tests/test_selection_suffstats.py`; the likelihood never evaluates a magnitude term. `theta_sel`
enters through `inference/prior.py:1207`, `kind_map[lbl] = ("normal", loc, scale)`, anchored on the
offline fit's covariance (`selection_fit_union.json`: `cov[0][0] = 2.546e-8`, i.e.
`sigma(M0hat) = 1.60e-4` mag — the number §3.1 quotes). **The shipped hierarchy is empirical Bayes on
`theta_sel`, not the joint of `eq:hierarchy`,** and any writeup must say so. The real question PR-2
must answer is therefore not "re-derive F2's disjointness for Schechter" but *"may an anchored
`theta_sel` prior coexist with a count likelihood whose base is `f_p C(z;theta_sel) Nbar`?"* — yes at
rung 0, bounded at rung 1 (guard 5, §4.4), refused at rung 2 (K10, extended to `theta_sel`). Whether
to implement (F2) as a genuine likelihood factor and drop the anchored prior is **OWNER DECISION 14**.

**(F2) caveat (R1-SEV3-12).** Rev 1 declared F2 "EXISTS, no change" on the basis of
`_fit_gaussian_truncated` (`redshift/selection.py:675`), a clean conditional density
`p(Mhat_i | z_i, T_i)` — that part holds, and `sigma(M0hat) = 1.6e-4 mag` is confirmed. But `master`
adds `_fit_schechter_truncated` (`:725`), per-catalog fits, K>=2 homogeneous-Schechter mixtures and
`M_faint_offset`. **PR-2 must re-derive the disjointness argument for the Schechter family before
latent mode admits it**; until then latent mode refuses `selection_family=schechter`.

Rev 1 also used "disjoint sufficient statistics" loosely. Correct statement: the chain-rule
factorization `p({z},{pix},{m}) = p({z}) p({pix}|{z}) p({m}|{z},{pix})` — which is what the code
supports. Use that phrasing.

**(F5)** `dN_miss = (1 - f_p C(z;theta)) dN_exp(theta) Q(xi)` — the `f_p` is the §1.2 change;
`Q_eff = exp(clip(logQ, +-7))`, relaxed to 1 beyond `z_depth`, consumed at the two bracket nodes
(`redshift/prior.py:711-757`).

### 3.2 Objects

| symbol | shape | dtype |
|---|---|---|
| `xi` | `(M,)`, `M = M_sph M_z = 315 x 12 = 3780` | f64 |
| `Xi = reshape(xi, (M_sph, M_z))` | `(315, 12)` | f64 |
| `Phi_s` | `(49152, 315)` | f64 |
| `Phi_z_out` | `(N_z_sub, 12)` | f64 |
| `W` (shell response, frozen) | `(G_s, N_fine)` | f64 |
| `N_pg` | `(n_fit, G_s)` | f32 |
| `f_p` | `(49152,)` | f32 |
| `A_m, B_m` | `(M_draw, n_b, N_z_sub)` | f64 |
| `row_fac` | `(M_draw, n_fp_rows, M_z)` | f32 |

`Phi = Phi_s (x) Phi_z` **exactly**, never materialized. At `M = 3780` a dense `Phi` would be
`1,572,864 x 3780 x 8` = 47.6 GB (the recorded OOM at `M = 1728` was 21.7 GB,
`logs/qbuild_gp3d_recal_1119087.err`, `21,743,271,936 = 1572864*1728*8` — confirmed byte-for-byte).

### 3.3 Factored jitter — the convention, named

Rev 1 wrote `chol(k + j I)` for each factor and never defined `j_sph` or `j_z` (R2-SEV2-9), and
mis-sized the legacy delta (R2-SEV2-8). Review measured, at the production hyperparameters:

* `M_sph=64, M_z=27`: `max|Phi_kron - Phi_legacy| = 4.96e-5` (rev 1 said 5.4e-5 — fine)
* `M_sph=315, M_z=27`: **2.0e-3** — 40x rev 1's claim and 33x over rev 1's own `< 6e-5` gate
* the three natural conventions differ among themselves by 15x, and the `sqrt(jitter)` convention
  moves `max_v sum_i Phi[v,i]^2` — the quantity `prior_var_rows` returns and PR-1 pins — by 1.8%
* the Kronecker identity itself is exact under every convention (`9e-15` to `1.9e-14`)

**Named convention, stamped, fixed before PR-1:**

```
jitter_mode = "factored-v1"
j_sph = j_z = 1e-6          (absolute, amp-independent; amp == 1 by section 4.3)
```

chosen as the smallest value keeping `cond(K + jI) < 1e8` in f64 at `M_sph = 315` (measured
`cond = 4.3e4` at legacy jitter, so 1e-6 is comfortable). Consequences:

* PR-1's gate is **`max|Phi_s (x) Phi_z - chol-of-factored-K basis| < 1e-13`**, *set from
  measurement* (review measured 1.9e-14 over 12,800 rows) with a randomized 1e5-row spot check at
  production rank — not `1e-12` asserted, and **not** against `build_lowrank_operator`, whose dense
  `Phi` cannot be formed at production rank (R2-SEV2-11).
* legacy-vs-factored delta (2.0e-3 at `M_sph=315`) is **reported as a diagnostic, not gated**.
* PR-5's migration pin is against a rebuild **using the same convention** — otherwise it is
  unachievable by three orders of magnitude.

> **OWNER DECISION 2.** Adopt `factored-v1` with `j_sph = j_z = 1e-6` (recommended). Cost:
> latent-mode `Q` differs from any legacy `gp3d` table at the ~2e-3 level at the guard rank, so
> every migration pin references a rebuilt reference and the legacy delta is a reported number.

### 3.4 The offline solve

```
J(xi) = 0.5||xi||^2 - sum_g sum_p N_pg log pi_pg(xi)             eq. (1'), theta-free at rung 0
grad  = xi - b sum_g Phi_g^T (N_g - T_g pi_g)
H     = I + b^2 sum_g [ Phi_g^T diag(T_g pi_g) Phi_g - T_g u_g u_g^T ],   u_g = Phi_g^T pi_g   (3)
```

**[v4]** Eq. (3) is the **exact** Hessian of eq. (1'), because `eta_pg` is exactly linear in `xi` once
`W` acts on the basis rows (§0.5 D1); against v3's exponential-inside object it was neither exact nor
separable. It is also the **Fisher** information of the multinomial, hence `H >= I` for *any* link —
including PR-6c's smooth saturation, where the *observed* Hessian is indefinite. **Fisher scoring is
therefore normative, not a convenience**, and `prop:warmstart`'s premise survives PR-6c (§0.5
finding 3). The exact-quadrature objective (exponential inside) needs `T[g,i,j,a,b]` = 3.7 GB and
~4.5e16 flop/step; it exists **only** as a reduced-rank validation reference (P5c), never as a
shipped solve.

**The rank-1 term is the correction review found missing (R2-SEV2-10).** Rev 1 wrote only the
Poisson `I + b^2 Phi^T diag(lam) Phi`; dropping a PSD subtraction makes `H` too large, `H^{-1}` too
small, and `laplace_draws` **under-dispersed** — precisely the direction that worsens §6.5. It is
also separable, so it costs nothing: with `Phi_g = Phi_s (x) phi_z[g]`,

```
u_g = v_g (x) phi_z[g],   v_g = Phi_s^T pi_g in R^{M_sph}
T_g u_g u_g^T = T_g (v_g v_g^T) (x) (phi_z[g] phi_z[g]^T)
```

i.e. `G_s` rank-1 Kronecker outer products, `~G_s (M_sph^2 + M_z^2)` flops.

Separable two-stage contraction for the diagonal part (do **not** precompute `S[p,(i,j)]` — 39 GB):

```
stage 1: T[g,i,j] = sum_p lam[p,g] Phi_s[p,i] Phi_s[p,j]     2 G P M_sph^2 flops,  G M_sph^2 8 B
stage 2: B[(i,a),(j,b)] = sum_g T[g,i,j] phi_z[g,a] phi_z[g,b]   2 M_sph^2 G M_z^2 flops, M^2 8 B
```

At `M_sph=315, M_z=12, G_s=32`: stage 1 = 3.1e11 flop, `T` = 25 MB, `H` = 114 MB, Cholesky
`M^3/3` = 1.8e10 flop. **Tens of milliseconds per iteration, ~13 iterations, ~1 s of linear
algebra.** Rev 1's "one 24 h GPU build" (R2-SEV3-17) is withdrawn: the anchor build is
**minutes**, dominated by I/O and the count assembly, not by the solve. **The same arithmetic is the
refusal of the per-proposal re-solve**: ~1 s against a measured 27.5–49.3 ms baseline is a ~36x
wall, four orders above what a `~1e6`-call nested-sampling run can absorb. Rung 1 exists because of
this number.

**Sensitivities: one construction, `n_theta + 1` columns.** `b_gal` is **fixed at the anchor** in the
solve (`TracerCounts.bias: float`), and its uncertainty is carried by a **rank-1 inflation** of the
draw covariance (R1-SEV2-5):

```
Cov(xi) = H^{-1} + s_b^2 (d xi_hat / d b)(d xi_hat / d b)^T,
d xi_hat/d b = -H^{-1} (d grad / d b)      one extra triangular solve, from the IFT
```

**`d xi_hat/d b` is one column of `S = d xi_hat/d theta` (§1.7).** Both come from the identical
implicit-function-theorem construction against the identical `H_chol`, so the module ships
**one** entry point:

```
latent_counts.sensitivity(xi_hat, H_chol, *, wrt) -> (M, n_wrt)
    wrt = ("b_gal",)                      -> the rank-1 inflation of R1-SEV2-5
    wrt = ("Om0","w0","wa","delta", ...)  -> the columns of S used by rung 1
```

`bias_sensitivity` is retired as a separate name at PR-3. **K>=2 shaping is binding here, not at
PR-7:** `S` is assembled from the *stacked* K-tracer objective, so catalog k's own selection theta
(`M0hat_c2`, `sigma_M_c2`, …) and its own `b_k` each contribute their own column. K>=2 then **adds
columns**, which is the owner's "adding terms, not refactoring" requirement discharged at the level
of an interface rather than a promise.

**[v4, §0.5 finding 11 — v3's "`s_b` is a 20% prior width" is withdrawn.]** §4.3 fixes `amp = 1`, so
`b_gal` is the **sole** clustering amplitude and the count channel measures it — v3 could not
simultaneously claim that and call `(b_gal, amp)` degenerate at K=1. `s_b` is therefore the **profile
curvature** `s_b^2 = [-d^2 log p_count/db^2]^{-1}` at the anchor (a 1-D profile against the same
`H_chol`), with a stated systematics floor; at K>=2 the ratio `b_2/b_1` comes from its own 2x2 profile
curvature. Only then is Tier-B's "latent-on CI >= table CI" a real check: with a free 20% dial it
could be made to pass or fail by choice.

### 3.5 New modules

**`darksirens/redshift/latent_field.py`** — pure JAX.
`LatentBasis(phi_sph, phi_z_out, phi_z_fine, shell_response W, proj_sph, meta)`;
`build_latent_basis(...)`, `field_rows`, `row_factor`, `at_nodes`, `sky_moments` (eq. 2),
`prior_var_rows`, `sky_constant_coeffs`. Reuses `lowrank_inducing_nodes`
(`redshift/lognormal_completion.py:610`) and `_sphere_z_kernel` **unmodified** (the node-for-node
identity with `_SphereZGPBase` is pinned by `tests/test_lss_completion_gp3d.py:45`).

**`darksirens/redshift/latent_counts.py`** — the count channel.
`TracerCounts(pix, counts, completeness f_p, stratum, bias)`;
`shell_multinomial_logl(xi, theta)` — **`theta` is an argument from PR-3, not an afterthought**;
`count_map_solve(theta)` (fixed-trip Fisher scoring, smooth saturation, no host control flow) with
the offline anchor as the special case `theta = theta_ref`; `hessian_separable` (eq. 3, **including
the rank-1 term**); `sensitivity(xi_hat, H_chol, wrt=...)` returning the columns of `S` (§3.4);
`laplace_draws` (antithetic pairs, §6.5).

**`darksirens/likelihood/latent_q.py`** — the single seam.
`LatentQPlan(basis, Xi_m, row_fac, S, dA, dB, A_m, B_m, b_nodes, rows_pe, rows_sel, f_p, theta_ref,
n_draw: static)`; `latent_member_leaves(...)` returning `row_fac`;
`rho_from_moments(A, B, c, b)` in closed form; and, for rung 1,
`theta_shift(S, dtheta) -> row_fac_shift` (member-independent) plus
`moments_at(A, B, dA, dB, dtheta)`. **Rung 0 is the `dtheta = 0` branch of the same code path**, so
PR-6a and PR-6b share one implementation and the control arm is a flag, not a fork.

**`darksirens/cli/build_latent_field.py`** — the offline anchor builder, writing
`/latent_field/{xi_hat, H_chol, sensitivity_S, sensitivity_labels, dA_moments, dB_moments,
g_members, Xi_members, row_fac, A_moments, B_moments, basis_meta, shell_response, completeness,
counts, theta_ref, sha256}`. **`theta_ref` is a first-class field, not merely a fingerprint
ingredient** — at rung 1 it is the expansion point and must be readable, comparable and reported.

### 3.6 Modified modules (all anchors re-verified on `0c5b3db`)

| file | change |
|---|---|
| `likelihood/factory.py` | static `lss_field_mode in {"table","latent"}`; latent leaves built in `make_likelihood` and `barrier()`-ed with the others (`:239-282`), **not** inside `body()` — the module docstring `:11-15` says the barrier has no effect inside a JIT body |
| `likelihood/core.py` | latent branch in `_member_leaf_bundle` (`:999`) carrying `(row_fac_m, A_m, B_m, logZ_m)`; **member-ESS diagnostic** from the already-materialized `ll_members` at `:1273`; force `sel_has_members=True` in latent mode (`:1372-1380` is the existing all-or-none guard). The per-member `Neff`/variance guard is **not** new — `core.py:1250-1262` already calls `selection_log_correction(log_mu_m, Neff_m, ..., pe_variance_sum=...)` inside the member vmap |
| `redshift/prior.py` | `eval_dark_member_completion_latent`: `lq = b_gw*(row_fac[pix] @ phi_z[idx]) - rho(A,B,c,b)[idx]` replacing `member_logq[pix,idx]` (`:711-757`). `_materialize` (`:103-112`) and the `z_depth` relaxation untouched |
| `redshift/completion.py` | `f_p` into `dN_miss` (`:1280`) and the field normalizer (`:1792-1800`); latent variants of `build_field_lss_q_inputs` (`:659`) generating `Q` inside the existing chunked scan |
| `core/types.py` | `EMCatalog`: `latent_counts`, `latent_completeness`, `latent_fit_pixels`, `latent_z_edges`. `SurveyParams`: latent hypers become readable by the likelihood — the "never marginalised over" contract dies, pinned by `test_lss_completion_gp3d.py:528`, which must be scoped to table mode |
| `likelihood/block_sizing.py` | latent static branch in `estimate_pending_static_bytes` (`:599`) / `measure_static_state_bytes` (`:649`) **plus** the guarded transient branch in `_slopes_and_fixed` (`:708`, `batch_scale` at `:725`); `concurrent = max(1, chains, sched_max)` at `:324` is the multiplier that matters |
| `inference/q_provenance.py` | mode routing; latent bypasses `check_lss_completion_provenance` and installs §4.4 |
| `cli/inference.py` | new flags; `_check_selection_qtable_theta` (`:686`) bypass; `b_miss` rule inversion (`:2638-2673`); `settings.json` schema bump |

**The one architectural rule the promotion adds.** At rung 0 *every* latent array is theta-free, so
all of them are factory-static and `barrier()`-ed in `make_likelihood`. At rung 1 exactly **two**
objects become per-proposal — `row_fac_shift(theta) = reshape(S·Δtheta, (M_sph, M_z))` and the
moment correction `(∂A/∂theta)·Δtheta` — and they must therefore be computed **inside `body()` and
must not be barriered** (`factory.py:11-15`: the barrier has no effect inside a JIT body). Both are
**member-independent** and `O(M)`/`O(n_b N_z_sub)` in size, so they add nothing the 256x concurrency
multiplier can amplify — which is the entire reason §2.4's "0 transient" survives the promotion
(§2.5). `row_fac` itself, the `(M_draw, 30470, M_z)` leaf, stays static. Any implementation in which
`row_fac` acquires a `theta` index has silently become the re-solve design and is refused by the
`block_sizing` transient branch.
`redshift/prior.py`'s seam at rung 1 reads
`lq = b_gw * ((row_fac[pix] + row_fac_shift[pix]) @ phi_z[idx]) - rho(A(theta), B(theta), c, b)[idx]`
— one extra add on an existing gather, which is why §1.7 prices it at ~1 ms.

### 3.7 numpy-offline vs JAX; what is never traced

Untouched numpy/offline: the entire `radial` builder, `build_joint_lognormal_completion.py`,
`renormalize_q_mean_one` (`lognormal_completion.py:502`, still used by the table path), every HDF5
writer. Moved to JAX, one implementation used **both** offline and online (that identity is what
makes the migration pin meaningful): the basis, `W`, the multinomial objective/gradient/Hessian, the
separable contraction, the sky moments, `laplace_draws`.

Never traced: `z_depth` (a concrete float or `None`; a dozen Python-level branches assume it).
Static args: `c_mode`, `selection_family`, `catalog_sky_weighting`, `lss_marginalize`,
`lss_member_impl`, `n_catalogs`, `lss_field_mode`, `n_draw`.

---

## 4. Identifiability and conventions

### 4.1 The monopole

`{log n0, delta, log C_sel(z;theta)}` enter `log lam` as additive functions of `z` alone. Two
mechanisms, both applied: (i) the multinomial conditioning makes the sky-constant subspace an exact
gauge direction of (1); (ii) explicit coefficient projection `proj_sph = I - c^ c^^T`, `Phi_s c ~ 1`
over the footprint — theta-independent, Kronecker-preserving, removing `M_z` dofs.

### 4.2 The budget normalizer — pinned on the identity that is CONSUMED

Rev 1 pinned `sum_p f_p Q_p / sum_p f_p = 1` while consumption ran `sum_{all sky} (1-C) dN_exp Q`
with no `f_p` (R1-SEV1-3). Rev 2 states one identity and uses it in both places:

```
Q_p(z) = exp( b_GW f(p,z) - rho(z; c(z;theta), b_GW) )     for p in footprint, z <= z_depth
Q_p(z) = 1                                                  otherwise
sum_{p=1..n_pix} (1 - f_p C(z;theta)) Q_p(z)
   / sum_{p=1..n_pix} (1 - f_p C(z;theta))  ==  1      for every z, member, theta   (4)
```

with `rho` from eq. (2), which is exactly the `rho` that makes (4) hold. Properties:

* **Exact per-realization budget conservation** against the integral that is actually evaluated
  (`completion.py:1280` x trapezoid at `:1284/:1296`). The measured +55% Jensen inflation is removed
  by construction.
* **Scope correction, folded in from the repo reality check (and re-dispositioning R1-SEV1-3).**
  The identity is **already exact in the shipped production completeness modes**: under
  `c_mode in {aggregate, selection}` the builder fits the whole sky
  (`build_lognormal_completion.py:714`, `fit = np.arange(n_pix)`) with **p-independent** weights
  (`w_budget = np.tile((1 - Cbar_fine) * dN_exp_density, (n_fit, 1))`, `:744`), so mean-one reduces
  to `<Q>_sky = 1` and the consumed sum equals its `Q=1` value exactly for every `z`, member and
  `theta`; **under stratified selection the weights are `p`-dependent** —
  `w_budget = ((1.0 - Cfine_s) * dN_exp_density)[stratum_map[fit]]` (`:730-732`, **[verified]**) — so
  by `prop:budgetexact(2)`'s own two mechanisms the identity holds **only at `theta = theta_ref`** and
  leaks elsewhere at the order of the stratum-to-stratum spread in `C_sel` (v3 said "exact there too";
  that was wrong — §0.5 finding 14); under radial `per_pixel` unfitted rows ship
  `logQ = 0` exactly (`:570-590`). The one genuine leak is the **gp3d `per_pixel` borrowing halo**
  (`:975-999`), and `per_pixel` is refused in latent mode (guard 6). **Eq. (4) is therefore not a
  repair — it is the obligation this plan creates for itself by putting `f_p` inside the
  consumption (§1.2), which makes the weights p-dependent for the first time.** State it that way in
  any writeup; claiming to have fixed a shipped budget defect would not survive inspection.
* **The gauge, named.** The per-z monopole of `Q` is an *exact gauge direction* of the GW likelihood:
  `Q -> alpha(z) Q` together with `(1-C) nbar -> (1-C) nbar / alpha` leaves the entire GW likelihood
  invariant (`MODEL.tex` Prop. `prop:gauge`). "`C` and `n0` own the budget, `Q` owns placement" is a
  **gauge fixing, not an approximation** — the correct phrase for the paper.
* **Renormalizing each member does not commute with renormalizing the posterior mean**
  (`MODEL.tex` Rem. `rem:noncommute`): the ensemble mean and the deterministic table target
  different objects at `O(Var_sky(Q)/N_pix)`. "Members carry placement uncertainty but zero budget
  uncertainty" (`lognormal_completion.py:525-531`) is a **modelling choice**, not a theorem, and OD6
  is where the discarded budget uncertainty is restored.
* **`Q == 1` off-footprint and above `z_depth`** — the "no information, no modulation" convention.
  This is a *choice* and it under-disperses: it assigns zero variance where there is no data rather
  than the prior variance, which at `amp = 1` would be a factor of `e` over 38% of the sky and would
  inject fabricated variance into `-259 log mu`. Carried as a stated systematic, bounded by the
  `amp(z)` sensitivity scan (OWNER DECISION 7). Rev 1's identity silently did the opposite.
* **The offline/online inconsistency disappears**: the count likelihood (1) is exactly invariant to
  per-`z` monopoles, so applying `rho` downstream changes nothing upstream.
* The `-0.5 b^2 sigma2_vox` shift and the `prior_var - post_var` correction of `eval_logq_gp3d`
  (`lognormal_completion.py:889`) both disappear — subsumed and double-counting respectively.

### 4.3 Bias conventions

`(b, xi)` and `(amp, xi)` enter only through `b*xi` and `b*amp`. **Fix `amp = 1`; sample bias only.**
`b_gal` per tracer is fixed at the anchor in the solve with rank-1 uncertainty propagation (§3.4).
`b_GW` multiplies `f` in `dN_miss`; it is constrained only by 259 events and will be prior-dominated.
This inverts the existing `b_miss` guard (`cli/inference.py:2638-2673`); in latent mode `b_miss`
becomes `b_GW` and is genuinely identified. New explicit rule + `settings.json` schema bump; old and
new runs do not share a parameter space.

### 4.4 What replaces the provenance firewall

Retired for the latent path: `check_lss_completion_provenance`, the float whitelist
(`catalogs/lss.py`), `_check_selection_qtable_theta` (`cli/inference.py:686`), the `c_mode`
table-vs-run check, and `realization_set_id`/`member_content_sha256` matching.

**The shared-realization theorem — promoted from a parenthesis to a headline design claim.** One
`xi` shared across K tracers makes "**member m of every catalog is the same realization**" a
*theorem*, not a stamped assertion to be verified. That is differentiator 2 (§0.2) delivered
structurally, and §0.4 is why it matters operationally rather than aesthetically: today the joint
builder is the **only** producer of a shared `realization_set_id` and it is `per_pixel`-only, has no
budget gauge fixing, and cannot resolve any physically supportable correlation length — so a K>=2
shared-latent marginalization under selection `C` is *not constructible*, and the only way to run the
configuration is `--allow_unverified_shared_lss_members`, which
(`inference/loaders.py:352-395`, the code's own words) marginalizes over an **independent-fields
product prior** rather than the shared-field prior the estimator assumes. **The latent design closes
that seam by deleting the producer**, collapsing a three-way compatibility matrix (joint builder ×
`c_mode` × K) to nothing. `realization_set_id` is not satisfied; it ceases to have a referent.

Seven successors, all at likelihood-build time:

1. **Artifact fingerprint.** sha256 over `(M_sph, M_z, ls_ang, ls_z, amp, z_node_hi, jitter_mode,
   j_sph, j_z, nside, completeness-map hash, shell-response `W` hash, z_edges, counts hash,
   anchor theta_ref, b_gal)`. Closes a gap the *current* firewall has: Q-table stamps carry no
   catalog-content hash, so a table built from a different catalog at the same nside is undetectable
   today.
2. **`ls_z` units guard.** In latent mode `ls_z` **must** be supplied in `zeta = log1p(z)` units;
   supplying `lss_corr_length_mpc` is a hard error, because `ls_z = L/((1+z_ref) dchi/dz) ~ H0`
   varies by 7x over `H0 in [20,140]` and would make an assumed length a standard ruler against
   22.79M galaxies. Under decision 1.1 this cannot happen accidentally; the guard makes it
   unrepresentable.
3. **Resolution guard, both sides HARD.** Reuse `_gp3d_resolution_guard`
   (`cli/build_lognormal_completion.py:274-319`) at likelihood-build time and **promote the sphere
   side from WARN to ERROR in latent mode** — rev 1 assumed it already was. `d_sph = sqrt(4pi/M_sph)
   <= ls_ang` and `log1p(z_node_hi)/(M_z-1) <= ls_z`.
4. **Isotropy guard.** `|log( (ls_ang * chi(z_ref)) / (ls_z * (1+z_ref) * dchi/dz) )| < log(1.5)`
   in latent mode — refuses the 4:1 pancake that rev 1 shipped by accident.
5. **Budget-anchor guard — restated PER RUNG in v4 (§0.5 finding 6).** At **rung 0** the count channel
   carries **zero** information about `(n0, delta, theta_sel)` by construction (`prop:cancel`). At
   **rung 1** it carries exactly the within-shell residual information about `(delta, theta_sel)`,
   which is the same object `P7e` measures, so the overlap with the calibration prior fitted to the
   *same* 22.79M counts is bounded by a number the plan already computes: if `osc_theta` of eq. (0d)
   restricted to the `(delta, theta_sel)` directions exceeds 0.1 nat, either widen the `delta` prior
   by the measured overlap or restrict rung 1's coupling set to `(Om0, w0, wa)`. At **rung 2** the
   restored shell totals carry full monopole information and K10 applies — **extended in v4 from
   `n0` to `theta_sel`**, because the restored `T_g ∝ n0 (c/H0)^3 ∫ C_sel(z;theta_sel)...` draws
   `theta_sel` information from the same magnitudes that produced the `1.6e-4` mag anchored prior.
   A **flat** prior on `log10n0`/`delta` in latent mode is refused at every rung; the calibration
   prior (OWNER DECISION 6) or `--allow_unanchored_budget` is required.
6. **`c_mode` gate.** `latent + per_pixel` refused (circular). `latent + aggregate` and
   `latent + selection` admitted **only with `--per_pixel_completeness`** (§1.2).
   `latent + --lss_completion` refused. `latent + --use_lss` refused (inherits the existing
   `field_lss_q` vs `field_delta_g` exclusion, `redshift/completion.py:1698`).
7. **Occupancy guard.** Shell edges come from the artifact, never from `DARKSIRENS_ZMAX` (§5.2);
   every shell used by the count channel must have `>= 1e4` galaxies and `>= 500` occupied pixels.

### 4.5 Bit-identical in table mode

`_LOGQ_CLIP = 7.0`, `field_clip = 10.0`, the `Q`/`delta_g` exclusion, the `z_depth` relaxation, the
two-node bracket gather, the stratified empty-sum decomposition, `logsumexp(ll_m) - log M`, and
every golden cell.

---

## 5. Two configuration facts the plan must fix

### 5.1 The stale base

`master` is 30 commits ahead of the declared base, with `redshift/selection.py` +676 lines,
`cli/inference.py` +754, `inference/prior.py` +321, `inference/parameters.py` +199. **Rebase before
PR-1.** Every anchor in this document was re-verified on `0c5b3db`; the ones rev 1 got wrong (file
attribution for `_assemble_gp3d_survey`, `n_pix_total`, the `Q`/`delta_g` raise, `ndim=3`) are
corrected in §8. Do not propagate `inference/prior.py`'s stale docstring about the field being gated
to a plain galaxy-count host model.

### 5.2 The `zmax` split

`z_s = np.linspace(0, zgrid[-1], gp3d_nz_solve)` with `zgrid[-1] = DARKSIRENS_ZMAX`
(`cli/build_lognormal_completion.py:880`). `experiments/desi_ingest/run_qbuilds.sh` exports
`DARKSIRENS_ZMAX=0.75`; `experiments/desi_full259/sbatch_ns_joint_sel.sh` exports `6.0`. Rev 1
quoted both (`N_grid = 1086` *and* `Delta_z ~ 0.023`) and they are incompatible (R1-SEV2-7).
Harmless today because the production run carries no LSS; **not harmless under this plan**: at
`zmax = 6.0`, `linspace(0, 6, 32)` gives `Delta_z = 0.194` and the entire catalog (`z` p1/p99 =
0.060/0.299) lands in the first two shells, fitting `M_z` radial modes to ~2 data shells.

**Rule.** The count grid and the likelihood grid are **decoupled and separately stamped**:

* `z_count_edges` — `G_s + 1` edges over `[0, z_depth]`, stored in the artifact, **never** derived
  from `DARKSIRENS_ZMAX`. `phi_z_fine`/`W` live here.
* `zgrid` — the production `N_grid = 1086` grid to `DARKSIRENS_ZMAX = 6.0`. `phi_z_out` lives here,
  restricted to `z <= z_depth` (`N_z_sub ~ 60-70` nodes); `Q == 1` above.

Guard 7 (§4.4) enforces occupancy. `G_s` is set by the photo-z-limited radial resolution, not by
`gp3d_nz_solve`'s default of 32.

---

## 6. Validation

### 6.1 Testbed

`experiments/completeness_viz/generate_clustered_mock.py` draws the clustering truth from the same
low-rank GP family the builder fits and writes `truth.h5` with `xi_true`, `logq_truth_vox`,
`logq_truth_on_zgrid`, the complete catalog and the analytic selection ingredients. Three extensions
required: (i) write a **completeness map** `f_p` with the measured DESI `masked_frac` distribution
(mean 0.137, sd 0.104, p99 0.635) so PR-2 has real partial pixels; (ii) place GW hosts by the *true*
field including missing hosts; (iii) emit a matched injection set. Follow `mock-data-dag`: the
injection set comes from the same DAG as the events, `beta` uses the same statistic and threshold,
and no per-event constant is frozen at truth.

**Stated limitation (R1-SEV2-9).** Tiers A-C draw `xi_true` from the *fitted* family, so coverage is
guaranteed by construction in exactly the regime where there is no data. Tier D is the only tier
that tests misspecification, and no tier can test the `z > z_depth` extrapolation at all. That is why
PR-8 ships as a sensitivity scan, not a marginalization (OWNER DECISION 7).

### 6.2 Closure tiers

* **Tier A — field recovery, no GW** (CPU, nside 16). Draw `xi_true`, Poisson-sample counts, apply
  the completeness map, run `count_map_solve`, compare `logQ_fit` vs `logQ_truth` on voxels with
  `N >= 10`. *Accept:* slope `1.00 +- 0.05` (versus the 0.04 prior-collapse signature), Pearson
  `r > 0.90`, `||xi_hat - xi_true||/sqrt(M)` consistent with `sqrt(tr(H^-1)/M)` within 20%.
* **Tier B — single-realization H0 closure** (GPU, nside 16, ~60 events). **Four arms in v2:** latent
  off / table / **latent frozen (PR-6a)** / **latent theta-coupled (PR-6b)**. *Accept:* `H0_true`
  inside the 90% CI in every arm; latent-frozen and table agree within `0.3 sigma` at `theta_fid`;
  **latent-on CI width >= table CI width** — now a valid check because §3.4 propagates `b_gal`
  (rev 1's version compared against a point estimate and would have passed while under-dispersed);
  and **PR-6b vs PR-6a is the K2 comparison** (§9), reported with the §6.4 channel split so a shift
  driven by the galaxy-side evidence (effect A) is never mistaken for a dark-siren gain (effect B).
* **Tier C — coverage**, 50 realizations. *Accept:* PP-plot KS `p > 0.05`; median `H0` bias
  `< 0.2 sigma`; no realization outside the 99% band.
* **Tier D — misspecification, resized.** Rev 1's "mask wrong by 5%" was mis-sized against a true
  `delta` rms of 0.1-0.2. Rev 2 runs four stresses at the measured amplitudes: (i) completeness map
  perturbed by the measured `masked_frac` sd (0.104 in completeness, i.e. 50-100% of the signal);
  (ii) a `z`- and density-dependent unmodelled incompleteness at 5% amplitude (the
  fibre-assignment proxy of §1.2's residual); (iii) `ls_ang` wrong by 2x; (iv) a lognormal truth with
  a non-Gaussian tail. *Accept:* `H0` bias `< 0.5 sigma`; report each as a systematic.
* **Tier E — K=2** (PR-7), **elevated in v2 from a bias-ratio check to the gate on differentiator 2.**
  Two disjoint tracers, `b_2/b_1 = 2`. *Accept:* (i) ratio within `2 sigma` over 20 realizations;
  (ii) shared-`xi` coupling demonstrably tighter than two independent fits; (iii) **the seam-closure
  assertion — `K=2 × c_mode=selection × --lss_marginalize` runs without
  `--allow_unverified_shared_lss_members`.** (iii) is the whole claim: it is unreachable on the table
  path at any K today (§0.4), so passing it *is* the demonstration that the latent design closes the
  seam by deletion rather than by construction.

### 6.3 Numerical pins

| id | pin | tolerance |
|---|---|---|
| P1 | `Phi_s (x) Phi_z` vs chol-of-factored-K basis, small rank + 1e5-row spot check at production rank | `1e-13`, **set from measurement** (1.9e-14 observed) |
| P2 | `prior_var_rows` vs `sum(Phi**2, axis=1)` | `1e-12` (`factored-v1` only) |
| P3 | legacy-vs-factored `Phi` delta at `M_sph in {64,315}` | **reported, not gated** (4.96e-5 / 2.0e-3) |
| P4 | `shell_multinomial_logl` vs direct scipy | `1e-10` |
| P5 | separable `H` (eq. 3, **with** the rank-1 term) vs dense `H` of the **linearized (1') multinomial** | `1e-10` |
| **P5b** | **the within-shell Jensen residual** (§0.5 D1): `max_{g,p} (b^2/2)\|Var_g(s_p) - <sigma_p^2>_g\|` at `xi_hat` on the real catalog, and the induced `Delta log p_count` | reported; `Delta log p_count` **oscillation** across the 20 theta inside 0.1 nat, else the shells are too wide |
| **P5c** | **(1') vs exact quadrature** (exponential inside the shell integral), **reduced rank only** — the exact object is 3.7 GB / 4.5e16 flop per step and is never shipped | reported; the gate is P5b |
| P6 | `count_map_solve` vs a dense scipy Newton on the **same (1') objective**, plus `grad_inf(xi_hat) < 1e-8` | `1e-8` |
| P7 | **frozen-`W` misspecification diagnostic** (v4: **no longer the gate** — §0.5 D4). Solve the theta-dependent `pi_pg` at 20 theta drawn from the prior; report `tau = max_theta \|\|xi_hat(theta) - xi_hat(theta_ref)\|\|_H / sqrt(M)`. **Reported at PR-3, day ~14.** | reported. `tau` also feeds `prop:esslaw` so the 1b trap stays visible; the admissible 1b threshold is `tau < 0.025` at `M = 3780` |
| **P7c** | **THE PROMOTION GATE, GW SIDE** (§0.5 eq. 0c): `osc_theta [ a . (xi_hat_theta - xi_hat_ref) ]` in **nats**, with `a = b_GW(sum_i phi_i Phi_i - N_obs <Phi>_sel)` from eq. (6) — computable at **PR-3** from the same 20 solves, **no seam required** | `< 0.1` nat → rung 0 *is* rung 1 on the GW side; `>= 0.1` with P7b's residual below 0.1 → rung 1a; else **K9** |
| **P7c'** | **nonlinear confirmation** at PR-6a with the shipped seam: `osc_theta [ LSE_m ll_m(xi_hat_theta + L_H^-T g_m) - LSE_m ll_m(xi_hat_ref + L_H^-T g_m) ]`, **plus** the `H_ref`-vs-`H_theta` draw-covariance term | agrees with P7c to the second-order term; the covariance term inside 0.1 nat |
| **P7d** | **log-det drift**: `osc_theta 0.5\|log det H(theta) - log det H(theta_ref)\|` at the same 20 solves (free — the Cholesky is formed) | `< 0.1` nat, else the evidence carries `0.5 tr(H^{-1} dH/dtheta)·Δtheta` |
| **P7e** | **THE PROMOTION GATE, GALAXY SIDE** (§0.5 eq. 0d): `osc_theta [ l(xi_hat_theta,theta) - 0.5\|\|xi_hat_theta\|\|^2 - 0.5 log det H(theta) ]` — the **correct** evidence of eq. (5), free at the same 20 solves | reported; and its `(delta, theta_sel)` restriction gates guard 5's rung-1 overlap at 0.1 nat |
| P7b | **linear-response residual**: `max_theta \|\|xi_hat(theta) - xi_hat_ref - S·Δtheta\|\|_H / sqrt(M)` over the same 20 theta | `< 0.1` per-mode posterior sd, else add a second-order term or a small set of anchors across the prior |
| P8 | per-galaxy vs population-average photo-z kernel in `W`: `\|\|Delta xi_hat\|\|_H / sqrt(M)` | `< 0.1`, else ship per-galaxy |
| P9 | `rho` from the `(A,B)` moment table vs direct computation, 200 random `(b, c)` | `1e-6` |
| P10 | `log\|H\| = 2 sum log diag(H_chol)` vs `slogdet` | `1e-8` |
| P11 | latent `logQ` at `(theta_fid, xi_hat)` vs a rebuilt `logq_map`, same jitter convention, nside-16 first | `1e-10` |
| P12 | golden with latent off | **bit-identical**, `DARKSIRENS_GOLDEN_EXACT=1` |
| P13 | budget identity (4) — the **consumed** one, with the footprint/off-footprint split of §2.2 | `1e-12`, all `z`, all members, 5 theta |
| **P13b** | **off-footprint routing**: the seam returns **bit-zero** `logQ` on every gathered row outside the fitted footprint (~38% of the 49,143-row PE/injection union) | bit-identical zero; without it P13 fails and `Q = exp(-rho)` off-footprint |
| P14 | **`M_draw` convergence** (§6.5) | see below |
| ~~P15~~ | ~~implicit-diff gradient vs FD~~ | **DELETED** — the production sampler is gradient-free (`block_sizing.py:294-302`; banner `Peak model: value-only (tinyns, 256 concurrent evals)`). `custom_vjp` goes with it. Re-add only behind a NUTS path |
| **P16** | **complete-catalog limit** (§1.6 Limit I), **at `b_GW = b_gal`** (v4, §0.5 finding 2): set `C ≡ 1`, `f_p ≡ 1`; latent-on must equal latent-off **bit-identically, for every member**. At `b_GW != b_gal` it is deliberately a *non*-identity, and the predicted size of the excess-bias spike factor `e^{(b_GW-b_gal)s_j}` is reported instead | bit-identical at `b_GW = b_gal`, `DARKSIRENS_GOLDEN_EXACT=1`. Strictly stronger than P12: P12 tests the *flag* off, P16 tests the *physics* of the seam. Gate at **PR-5** |
| **P17** | **Gaussian-marginalization limit** (§1.6 Limit III), **restated in v4 in two arms** (§0.5 finding 1) — v3's single form pinned a posterior-covariance quantity against a prior-covariance formula and could pass only in the prior-collapse regime K3 exists to refuse. **(a) counts-off** (`H = I`, `xi_hat = 0`): `LSE_m ll_m - log M -> (b_GW^2/2) sum_{ij} K(x_i,x_j)`, validated against truth. **(b) counts-on** (the shipped configuration): `LSE_m ll_m - log M - ll(xi_hat) -> 0.5 a^T H^{-1} a = 0.5 sigma^2`, with `a` from eq. (6) carrying the `-N_obs <Phi>_sel` selection subtraction by construction | within the estimator's MC error. **Arm (b) makes P17 and §6.5 item 5's `sigma` prediction one measurement.** Gate at **PR-5b**, alongside P14 |
| **P20** | **posterior linear-predictor sd `s_v`**, measured on the real anchor and reported as a **distribution** stratified interior / partial / off-footprint / above-`z_depth` (v4: replaces the inherited scalar `5.6e-3`, §0.5 finding 18) | reported; it is the quoted regime of validity for Limits II and III and for the Laplace ledger row |
| **P18** | **rung-0 / rung-1 inertness**: PR-6b evaluated at `Δtheta = 0` reproduces PR-6a | bit-identical. Makes the control arm a flag rather than a fork |
| **P19** | **projected moment tables**: `(A_m, B_m)` from `A^ref + b (∂A/∂theta)·Δtheta` vs direct recomputation at the same 20 theta | same scale as P9 (`1e-6` relative), else the `shell_lognorm` step returns to the per-proposal path (§2.5) |

**P4/P5/P6 replace rev 1's tautological gate.** Rev 1 pinned `shell_multinomial_logl` for
theta-invariance across 20 theta "to prove decision 0.1" — but at rev 1 that function had no `theta`
argument, so the pin could only fail if someone passed one (R1-SEV2-4). In v2 it **does** take
`theta` (§3.5), so P7 is a real measurement of a real dependence rather than a check on an absence.
Likewise rev 1's solver pin was against `poisson_lognormal_gp3d_map`, the *unconditional* objective
§1.1 discards (R2-SEV2-10); P5/P6 pin the object that ships.

**P7's tolerance was wrong in rev 2, by a factor `sqrt(M)`, for one of its two roles.** The number
has two legitimate readings and rev 2 conflated them:

* **model misspecification** (how far the frozen-`W` model sits from the exact one) — a **per-mode**
  norm is defensible, and `tau < 0.1` is a reasonable *misspecification* tolerance;
* **estimator validity** under design **1b** (fixed draws, importance weights) — here only the
  **total** is admissible. `Var[log w] = ||Δxi_hat||_H^2` *exactly* (`MODEL.tex` Prop.
  `prop:esslaw`), so `ESS/M ≈ exp(-||Δ||_H^2)`. A per-mode `tau = 0.1` means
  `||Δ||_H^2 = 0.01 M = 37.8` at `M = 3780`, i.e. **`ESS/M = 4e-17` — from a pin that reads as
  comfortably tight.** The admissible 1b threshold is `||Δ||_H^2 < log 10 ≈ 2.3`, i.e.
  `tau < sqrt(2.3/M) = 0.025` at `M = 3780` (`0.030` at `M = 2520`).

**v2 ships 1a (mean-shifted draws), for which no ESS penalty exists at all** (§0.3) — which is
precisely why the distinction has to be written down: the trap is invisible from the pin's face
value, and an implementer who reaches for importance weights will pass P7 and produce nonsense.

### 6.4 Runtime diagnostics shipping with PR-6a/PR-6b

* **Member ESS** `exp(-sum_m p_m log p_m)`, `p_m = softmax_m(ll_m)`, from `core.py:1273`.
* **Per-member `Neff`/variance guard** — **already exists** (`core.py:1250-1262`); rev 1 listed it as
  new work (R1-SEV3-11). Nothing to build; assert it is on.
* **Determinism sweep.** 100 repeats bit-identical; a 1e4-point 1-D `H0` sweep with
  `max|Delta logL|` between adjacent points `< 10x` the median. **State its limits in the same
  breath:** under common random numbers the estimator is a *deterministic smooth function of
  `theta`*, so repeat-determinism and adjacent-theta smoothness pass **by construction** — and pass
  just as well on a badly distorted surrogate (`MODEL.tex` Rem. `rem:crn`). This sweep is a
  regression guard, never evidence of correctness. Only P14's **theta-varying** bias discriminates.
* **`M_draw` convergence trace** (§6.5) — the one that decides shippability.
* **Channel split — rung 1 only, and mandatory.** Report the galaxy-side evidence (effect **A**),
  **as eq. (5) and not as `Delta log p_count`** — the Occam and log-det terms are part of it (§0.5 D2)
  — **separately** from the GW-side member shift (effect **B**). They have different failure modes and
  different credibility, and (contra v3) they are **not** the same number: A responds to the full
  count residual, B only to its `span(Phi)` projection damped by `H^{-1}`. **The acceptance rule is
  tightened in v4 (§0.5 finding 7): A is reported as a diagnostic and NEVER enters the headline
  posterior unless `W`'s own parameters — the within-shell profile `omega_g`, the photo-z kernel
  `sigma_z(z)`, the shell edges — are sampled or profiled**, because A is a `dN/dz`-shape constraint
  exactly degenerate with `delta`, with `sigma_z(z)` and with the binning, all of which are frozen
  inside `W` and none of which are sampled. This is **OWNER DECISION 13**. If A dominates the posterior
  shift, the headline result is a galaxy-clustering measurement wearing a dark-siren hat, and that must
  be discovered here rather than in referee report 1.

### 6.5 The marginalization-accuracy problem — the plan's largest open number

`log Zhat = logsumexp_m(ll_m) - log M` is a Jensen-biased estimator of `log int p(d_GW|xi)p(xi|d_gal)`
whenever `ll_m` has spread. For lognormal weights with sd `sigma` nats: `E[ESS]/M ~ exp(-sigma^2)`.

**The bias, in its sharp form** (`MODEL.tex` Prop. `prop:jensen`) — rev 2 quoted the scaling
`M ~ exp(sigma^2)`; the usable statement is a number:

```
bias ≈ - Var(Zhat) / (2 Z^2) = - (e^{sigma^2} - 1) / (2 M)
  ==>  M  >  (e^{sigma^2} - 1) / (2 epsilon)     for an epsilon-nat bias
```

| `sigma` (nats) | `M_draw` for 0.1 nat | bias at `M_draw = 8` |
|---|---|---|
| 1.0 | **9** | 0.11 nat |
| 1.5 | **43** | 0.53 nat |
| 2.6 | **4.3e3** | **54 nats** |

Rev 1's own §1.5 estimated up to **2.6 nats** of member spread from the `-N_obs log mu` convexity at
259 events. If that number is right, `M_draw = 8` is an `ESS ~ 0.01` evaluation, K5 fires
immediately, and `M_draw = 32` cannot fix it. The table above is what PR-5b's measurement is compared
against, directly.

Rev 2 refuses to argue this and instead measures it. v2 keeps that and adds a **prediction**, so the
measurement can be right or wrong rather than merely reported:

1. **It is a measurement, gated before PR-6a ships.** New **PR-5b**: with latent leaves live but the
   likelihood otherwise unchanged, emit the `ll_m` vector at `M_draw = 256` at 33 `H0` nodes across
   `[20,140]`. Report `sigma(H0)`, `ESS(H0)`, and `log Zhat_{M} - log Zhat_{256}` for
   `M in {4,8,16,32,64,128}`.
2. **What matters is the theta-VARIATION of the bias, not its level.** With common random numbers
   (`g_m` fixed, which rev 1 already specified) `log Zhat(theta)` is a deterministic, smooth
   function and a constant bias is absorbed into the evidence. **P14: `max_H0 [ (log Zhat_M -
   log Zhat_256) - mean_H0(...) ] < 0.1 nat`** across the `H0` prior. That is the shippability
   criterion; the absolute bias is reported, not gated. Rev 1 had no such pin, and — because CRN
   makes the estimator deterministic — every diagnostic rev 1 listed (repeat-determinism, adjacent-
   theta cliffs) would have passed while the sampler explored a distorted surrogate.
3. **Antithetic draws.** `g_m` in `+-` pairs (`M_draw` even). Free; cancels the odd part of the
   response exactly and typically halves `sigma^2` for a near-linear `ll(xi)`.
4. **`M_draw` is affordable precisely because of the compression.** The table cube is 1.71 GB at
   `M_draw = 8` and 13.7 GB at 64; the latent row factors are 11.7 MB and 93.6 MB. The compute cost
   is real and linear (§2.3) and is what OWNER DECISION 5 is buying.
5. **Predict `sigma` in closed form before measuring it — a ~1 hour calculation, scheduled as
   PR-5b's first task.** The same first-order expansion that produces §1.6's Limit III gives the
   member spread analytically. At leading order

   ```
   a      =  b_GW · ( sum_i phi_i Phi_i  -  N_obs · <Phi>_sel )                       (6)
   sigma  =  || L_H^{-1} a ||_2        [the H^{-1} norm, NOT the Euclidean norm]
   ```

   where `phi_i` is event `i`'s missing-branch fraction of its total prior mass and `<Phi>_sel` is
   the selection-weighted basis average. **[v4, §0.5 D4 — corrected inner product.]** Eq. (0)'s own
   derivation gives `Var_m(ll_m) = a^T H^{-1} a`, so the norm is `H^{-1}`, not Euclidean; v3's
   ingredient list omitted `H_chol` entirely, and since `H >= I` the Euclidean form systematically
   **over**-predicts `sigma` — which feeds an *exponential* `M_draw` requirement and OWNER DECISION 5.
   The same `a` is the gate statistic of P7c and the closed-form target of P17 arm (b), so **one
   object serves three purposes**. **Every ingredient already exists**: the basis `Phi`, the
   event pixel/redshift distribution, the injection weights, `C(z;theta)`, and `H_chol`. It is
   computed at **PR-0**, not PR-5b (§0.5 D3). A discrepancy between
   the predicted and measured `sigma` is itself diagnostic — it means `b_GW f` is *not* small, which
   is exactly the regime in which the numerical marginalization is doing real work rather than
   reproducing a Gaussian. This converts §6.5 from "we will find out" into "we predict X and
   confirm it", which is a materially stronger position for both the plan and the paper.

If PR-5b reports `sigma > 1.5` nats and P14 cannot be met at `M_draw <= 128`, PR-6a does not ship as
a marginalization; the fallback is to quote the field as a **fixed-realization systematic** (run the
analysis at `xi_hat` and at `+-1 sigma` draws, report the `H0` spread) — which is a legitimate,
publishable result and does not require the estimator to converge.

> **OWNER DECISION 5.** Set the marginalization-accuracy budget. Recommended: P14 at **0.1 nat** of
> theta-varying bias across the `H0` prior, `M_draw` chosen by PR-5b to meet it, cost accepted per
> §2.3 (+12% at 8, +26-47% at 32, +53-95% at 64 against the PR-0 baseline). The alternative is the
> fixed-realization systematic above, at zero added cost and reduced scientific claim.

---

## 7. PR ladder

Legacy is bit-identical throughout, enforced by `tests/test_unified_k1_golden.py` +
`tests/golden/unified_k1_golden.json` at `rtol <= 1e-12` with `DARKSIRENS_GOLDEN_EXACT=1`.
**Regenerating that golden is forbidden at every rung.**

**PR-0 — THE DECISION RUNG: can the field move the answer at all?** (3 d, was 1 d — **§0.5 D3**)
Rebase onto `0c5b3db`. Then four blocking items, **all against artifacts that exist today**, none of
which requires a line of latent code:

1. **Cost baseline, three columns.** Measure the dark-siren production likelihood with
   `scripts/profile_member_marginalization.py` and `scripts/benchmark_block_sizes.py`: (i) the
   no-LSS production baseline, (ii) table-mode member marginalization at `M_draw in {1,8,32,64}`,
   (iii) the projected latent column. **OWNER DECISION 5 is taken on (i)-relative numbers** (§2.3).
2. **`sum_i phi_i` and the closed-form member spread.** The event-weighted fraction of prior mass in
   the **in-support** missing branch (footprint, `z <= z_depth`), and `sigma = ||L_H^{-1} a||_2` from
   eq. (6) (§6.5 item 5, corrected norm). ~1 h; every ingredient exists.
3. **The Q-on-at-anchor oscillation.** Turn the *shipped* table on at the anchor and report
   `osc_H0 [ logL(Q on) - logL(Q ≡ 1) ]` across the `H0` prior. **This is an upper bound on everything
   the entire ladder can buy**, measured with code that already runs.
4. **The exhaustive paper-by-paper novelty search of §0.2** (1 d), **moved here from PR-11** — v3
   gated the search that decides whether the work has a paper at day ~50 while claiming a publishable
   differentiator at day ~28 (§0.5 finding 19).

Also publish the reconciled **in-support budget fraction**: R1's 6e-5, a reviewer's 2.7e-4, and this
session's integral over the shipped `C_sel` giving 1.3e-4 all-sky / 7.9e-5 over the occupied
footprint**. All three agree that the field is being asked to redistribute ~0.01% of the missing
budget.

*Gates:* the four items above committed to this directory; **kill criterion K0 evaluated** (§9). **No
cost, novelty or feasibility claim in this plan is usable until this lands.**

**PR-1 — `latent_field.py`, offline only.** (3 d) The basis, `factored-v1` jitter with
`j_sph = j_z = 1e-6`, `shell_response` `W` (with the photo-z convolution), `sky_moments`,
`sky_constant_coeffs`, `prior_var_rows`. Rewire the builders to call it.
*Gates:* P1, P2, P3; inducing nodes still match `_SphereZGPBase` (`test_lss_completion_gp3d.py:45`
unchanged); builder outputs byte-identical. **Fast subset.**

**PR-2 — completeness map + counts as data.** (5 d, was 4) Promote
`experiments/desi_ingest/build_mth_map.py` out of `experiments/`; `f_p = 1 - masked_frac` degraded
to nside 64 by area weighting; new `EMCatalog` leaves; `--per_pixel_completeness` flag putting
`C_p = f_p C(z)` into `dN_miss` (`completion.py:1280`) and the field normalizer (`:1792-1800`) with
its **own golden**; re-derive the F2 disjointness argument for `_fit_schechter_truncated`
(`selection.py:725`) or refuse `selection_family=schechter` in latent mode.
*Gates:* counts round-trip against `np.histogram(zgals, bins=edges)` exactly; `sum_p f_p Omega_pix`
matches the published DESI area to 1%; a coverage report (occupied / in-footprint partial /
off-footprint) printed and stamped; **`--per_pixel_completeness` off is bit-identical**;
`--per_pixel_completeness` on changes `dN_miss` in the direction and by the magnitude predicted from
`masked_frac` (a signed, quantitative assertion, not a smoke test).

**PR-3 — the count channel, `theta`-shaped from the start.** (5 d, was 4) `shell_multinomial_logl(xi,
theta)` — **`theta` is an argument, and `base(z;theta)` lives inside the shell integral; the offline
anchor is the special case `theta = theta_ref`, not the only case**; `hessian_separable` **with the
rank-1 term**; `count_map_solve(theta)`; `sensitivity(xi_hat, H_chol, wrt=...)` returning the columns
of `S`, built from the **stacked K-tracer objective** so K>=2 adds columns (§3.4); `laplace_draws`
with antithetic pairs. Plus a standalone CLI diagnostic that doubles as the `dN/dz` PPD gate already
queued in `selection-channel-followups.md`.
*Gates:* P4, P5, **P5b**, **P5c**, P6, P8, and — **the promotion gates, reported here at day ~14** —
**P7c** (GW side, eq. 0c), **P7e** (galaxy side, eq. 0d), **P7d** (log-det drift), **P7b**
(linear-response residual), with **P7** (`tau`) reported as a misspecification diagnostic only.
**[v4: v3 gated this rung on `tau`, which §0.3 itself declared undetermined by a factor of 61.5, and
placed its replacement `P7c` only in §0.3 where it required a seam that does not exist until day ~24.
Eq. (6) makes all four gates computable here, from the same 20 prior solves, with no seam.]**
**This rung decides the promoted direction** (§0.5 D4, kill criterion K9): the feasibility of the
owner's headline is known ~20 days before the deliverable, not after it. **Fast subset**
(small-problem variants).

**PR-4 — the anchor artifact.** (4 d, was 3) `cli/build_latent_field.py`; the `(A, B)` moment tables
on the `b_GW` grid **plus the projected derivatives `(∂A/∂theta, ∂B/∂theta)`** (~1.1 MB at
`M_draw=8`); `S` and `sensitivity_labels`; **`theta_ref` as a first-class artifact field**, not
merely a fingerprint ingredient. The first guard-compliant `gp3d`-family artifact that exists.
*Gates:* build completes at `M = 3780` in **minutes** (rev 1's 24 h estimate is withdrawn — §3.4);
P9, **P19**, P10; `laplace_draws` mean matches the posterior mean (analogue of
`test_lss_completion_gp3d.py:197`); artifact sha256 reproducible across two runs at the same seed;
occupancy guard 7 satisfied on the real catalog.

**PR-5 — the seam, latent inert.** (4 d) `latent_q.py`; leaves built and barriered in
`make_likelihood`; static `lss_field_mode`; `eval_dark_member_completion_latent`; the `block_sizing`
static branch **and** the guarded transient branch; CLI plumbing. Default OFF.
*Gates:* P12 (every golden cell bit-identical with the flag off) plus an explicit "latent-off is
inert" assertion; `test_member_factoring_parity`, `test_member_pe_vectorization`,
`test_lss_marginalization` unchanged; **P11 at nside 16 against the `completeness_viz` gp3d rebuild**
(the DESI-scale `q_gp3d.h5` does not exist — R2-SEV3-16 — so the DESI-scale migration pin moves
behind PR-4 and is reported, not blocking); **P16** (the complete-catalog limit: at `C ≡ 1` latent-on
is bit-identical to latent-off *for every member* — a physics gate, not a routing gate);
`block_sizing` reserves within 10% of measured peak.

**PR-5b — member-spread measurement.** (2 d) §6.5 item 1, **preceded by §6.5 item 5's ~1 h
closed-form prediction of `sigma`**, so the measurement confirms or refutes a number rather than
producing one. **Gates:** the report exists; predicted-vs-measured `sigma` reconciled; **P17** (the
Gaussian-marginalization limit — the only closed-form validation of the estimator in this document);
OWNER DECISION 5 taken. No code ships to production from this rung.

**PR-6a — the compressed member ensemble (control arm + fallback).** (5 d) *This is rev 2's PR-6,
kept in full and re-scoped.* Turn the mode on at the frozen anchor: `Xi_m` drives the member axis;
`rho` from the moment tables; member-ESS diagnostic; provenance rewiring (§4.4) with mode routing
both ways; `b_miss -> b_GW` inversion + `settings.json` bump.
*Gates:* `test_marginalize_equals_logmeanexp_of_members` still holds; **P13** (the consumed budget
identity); **P14** (the `M_draw` bias gate); guards fire in table mode and are bypassed in latent
mode; latent mode permits `Om0/w0/wa/b_miss`, permits `log10n0/delta` **only** under the calibration
prior, and refuses `per_pixel`, `--lss_completion`, `--use_lss`, `schechter` (until PR-2 clears it),
and `--per_pixel_completeness=off`; **mock closure Tiers A-D**; determinism sweep; overhead within
the OWNER DECISION 5 budget.
**At the end of PR-6a, differentiator 1 exists and is publishable on its own** (§0.2). It is also
the arm PR-6b must agree with in the benign limit, and the fallback if K9 fires.

**PR-6b — THE THETA-COUPLED FIELD-LEVEL LIKELIHOOD (K=1).** (5 d) **The deliverable.** Linear
response per §1.7: `theta_shift(S, dtheta)` producing the member-independent `row_fac_shift`;
`moments_at(...)` applying the projected `(∂A/∂theta, ∂B/∂theta)`; **mean-shifted draws with common
random numbers (design 1a) — importance weights (1b) are forbidden and the reason is P7's second
role**; the channel-split diagnostic of §6.4.
*Gates:* **P18** (at `Δtheta = 0`, bit-identical to PR-6a — the control arm is a flag, not a fork);
**P7b** re-checked at production rank; **K2 restated as PR-6b vs PR-6a** at the posterior level;
**K6 must remain unfired** across a `1e4`-point sweep — and it is unfireable by construction under
linear response, so a firing means the implementation is not the design; measured per-proposal
overhead within §2.5's ~1 ms estimate; the `A`/`B` channel split reported.
**At the end of PR-6b the owner's headline exists: no `Q` built at fixed cosmology, and every
proposal in `Om0/w0/wa/delta/theta_sel` moves the galaxy-field likelihood.** `H0` does not, and §1.8
is why.

**PR-6c — optional: unconditional Poisson (was PR-9).** (5 d, was 8) Only under OWNER DECISION 1(c),
and only after OWNER DECISION 6 is resolved. Restores (F3) as a likelihood factor, adds the Laplace
log-det `-½ log det H(theta)`, the smooth saturation replacing `field_clip` + KKT pinning, and the
convergence veto. **`custom_vjp` and P15 are deleted** (§2.5). The `H0` content that this rung buys
is the volume–density amplitude and nothing else.
*Gates:* agrees with PR-6b where the conditioning is benign; **K10** (`log10n0` under an `H0`-aware
prior, or refuse); a gradient-norm stopping rule with `eps^2/2` inside the oscillation budget
(§10); no adjacent-theta `Delta logL` jump `> 0.1` nat over `1e4` points; veto never fires.

**PR-7 — multitracer K>=2, and the seam closed by deletion.** (4 d) Length-K tracer list; per-tracer
`b_k`, completeness, counts, multinomial; per-tracer columns of `S` (already shaped at PR-3);
**retire `realization_set_id`** (§4.4); overlap policy (OWNER DECISION 9); per-tracer moment tables
(the `Z_k` non-cancellation is closed-form, §2.2).
*Gates:* K=1 bit-identical to PR-6b; **Tier E**, whose gate is **(i)** bias-ratio recovery and
**(ii)** shared-`xi` coupling demonstrably tighter than two independent fits, plus **(iii')** the
substantive replacement introduced in v4 (§0.5 finding 12): run the shared-`xi` likelihood **and** an
artificially decoupled two-field variant on the same mock and show the bias-ratio credible region
differs in the predicted direction — i.e. *demonstrate the coupling the `--allow_unverified` flag
throws away*. **v3's gate (iii) — "runs without `--allow_unverified_shared_lss_members`" — is demoted
to a statement of fact**, because that flag and its check live on the table-loader path
(`inference/loaders.py:352-395`) which latent mode deletes: it would pass by deletion of the check
rather than by satisfaction of the property, structurally the same routing tautology R1-SEV2-4 caught
in rev 1.

**PR-7i — interim, mock-only: joint-builder passthrough.** (1 d, optional, independent of PR-7)
Pass `--c-mode` / `--selection-fit` / `--realization-set-id` through
`cli/build_joint_lognormal_completion.py` (`:180-181`, argparse `:375-422`) **solely** to produce the
K>=2 selection **table** baseline that OWNER DECISION 10's arm comparison needs and that cannot be
built today. Caveat it in the artifact banner: the joint builder still applies **no budget gauge
fixing** (`renormalize_q_mean_one` uncalled) and its inducing grid (`M_SPH,M_Z = 32,6` to `z=3`)
forces `L_smooth >= 1.34 Gpc`, so this is **mock-scale only** and is *not* a DESI baseline (the
DESI-scale gp3d build already OOM'd at 21.7 GB). Do not extend it; PR-7 makes it obsolete.

**PR-8 — support and amplitude profile.** (3 d) `amp(z)` growth profile, extended `z_node_hi`.
**Reframed:** this rung produces a **sensitivity scan**, never a marginalized posterior — there are
no counts above `z_depth`, so the width is a pure function of the assumed `amp(z)` (R1-SEV2-9).
*Gates:* `amp(z)` reduces to a constant at the legacy value bit-identically; the deliverable is a
table of `H0` median/width versus assumed `amp(z_>depth) in {0, 0.05, 0.1, 0.2, 0.4}`, quoted as
"H0 shift under an assumed amp(z)".

**PR-9 — DISSOLVED.** Rev 2's optional unconditional-Poisson rung is split: its useful content moves
to **PR-6c**, and its two hardest items (implicit-diff `custom_vjp`, pin P15) are **deleted outright**
because the production sampler does not differentiate the likelihood (§2.5). Its "never warm-start"
prohibition is **replaced by a gradient-norm stopping rule** (§10), which is the correct statement.

**PR-10 — performance and campaign.** (3 d) Chunk plans; f32 policy; a `z < z_depth` sample-compaction
experiment for the seam (most GWTC-5 PE samples are above `z_depth`, so the gather may be largely
skippable — measure, do not bank); `block_sizing` refinement; the production run.

**PR-11 — the proposal.** (3 d; **its novelty-search gate moved to PR-0, §0.5 D3**) The owner's stated
deliverable
(`OWNER_CONTEXT.md:202-204`) — a LaTeX/PDF proposal: *what we do now, the proposed work (mainly math
+ pseudo-code), proposed PRs*. Rev 2's ladder produced no such artifact. `MODEL.tex` in this
directory is the mathematical body (it compiles: 16 pages, 0 undefined references); PR-11 wraps it
with the "what we do now" section fed by §1.9 and §0.4, the proposed-PR section fed by §7, and the
novelty section fed by §0.2.
*Gates:* the PR-0 novelty search is cited and no "first" appears without it; the forbidden
sentence (§0.2) appears nowhere; the `H0` statement of §1.8 appears verbatim; the scope caveat about
`Q ≡ 1` off-footprint and above `z_depth` appears in the same paragraph as differentiator 1; **(F2)'s
empirical-Bayes status (§0.5 finding 5) and the `b_GW != b_gal` spike factor (finding 2) are stated
rather than elided.**

**Critical path, corrected in v4 (§0.5 finding 20 — v3's counts did not sum).** Summing the ladder's
own durations: PR-0 **3** + PR-1 3 + PR-2 5 + PR-3 5 + PR-4 4 + PR-5 4 + PR-5b 2 + PR-6a 5 = **31 d
to PR-6a**; + PR-6b 5 = **36 d to PR-6b**. (v3 quoted 28/33 against a ladder that summed to 29/34 with
a 1-day PR-0.) Two early exits: **K0 at day ~3** ends it before PR-1 if the field cannot move the
answer, and **K9 at day ~14** ships PR-6a instead of PR-6b. The promoted deliverable remains cheaper
than rev 2's 8-day optional PR-9, and PR-6a remains shippable at ~31 d.

**CI gap, closed in PR-1.** `tests/fast_subset.txt` contains **none** of `test_q_provenance_guard`,
`test_q_budget_renormalization`, `test_lss_completion*`, `test_lss_provenance`,
`test_lss_marginalization`, `test_unified_k1_golden`, `test_member_factoring_parity`,
`test_stratified_q_base`. "Legacy stays bit-identical" currently rests on manually-run Tier-0b tests.
Admit `test_q_provenance_guard` and every cheap new pin as it lands.

---

## 8. Risks

| # | risk | sev | mitigation | detection |
|---|---|---|---|---|
| R1 | **Support mismatch.** Nodes stop at `z = 0.30`; 99.994% of the missing budget lies above; measured in-support fraction 6e-5. | fatal to the science, not the code | K2 (§9); PR-8 as a sensitivity scan only; reframe as a bounded systematic | the posterior-level diagnostic of R12 |
| R2 | **Unmodelled `p`-dependent, `z`-dependent selection** (fibre assignment) absorbed into `xi`, re-entering `dN_miss` with the wrong sign. Areal masking is now handled exactly (§1.2) and depth is measured uniform (`mth_eff` p1=p50=p99=21.000), so this is the residual. | **high** | OWNER DECISION 4; Tier-D stress (ii) at 5% amplitude; fallback to the homogeneous sub-footprint | Tier D; a `masked_frac`-vs-`xi_hat` correlation report at PR-3 |
| R3 | **Member-estimator bias / ESS collapse.** `sigma` up to 2.6 nats would make `M_draw = 8` an `ESS ~ 0.01` evaluation with tens of nats of theta-dependent bias. | **high** | §6.5: PR-5b measurement, P14 gate, antithetic pairs, compression making large `M_draw` affordable, fixed-realization fallback | P14; member-ESS diagnostic |
| R4 | **Cost.** Baseline is 27.5-49.3 ms, not 3-20 s. `M_draw` scales the member-dependent slice linearly. | **high** | §2.2 removes the dominant term in closed form; PR-0 measures; OWNER DECISION 5 sets the budget; §2.5 prices the promoted rung | PR-0 table; PR-6a/PR-6b overhead gates |
| R5 | **Budget re-inflation / normalizer mismatch** (the measured +55%). | high | §4.2 pins the **consumed** identity (4), not rev 1's `f_p`-weighted one | P13 |
| R6 | **Golden drift** — any shared path changing a table-mode value. Now two flags to protect: `lss_field_mode` **and** `--per_pixel_completeness`. | high | static routing; purely additive branches; both goldens in CI from PR-2 | P12 + the PR-2 `C`-side golden |
| R7 | **`block_sizing` mis-routing** (~34 GB precedent at `:623`). Rev 1 routed per-eval transients into the static estimator. **Restated for v2:** under linear response the static branch stays correct because `S·Δtheta` is member-independent; under any *re-solve* variant the guarded transient branch becomes the **primary** accounting path and `row_fac` costs 3.0 GB (`M_draw=8`) to 24 GB (`M_draw=64`) at 256 concurrency (§2.5). | high | latent leaves are genuinely static (§2.4, §2.5); the transient branch ships alongside and gates the recompute path | measured-vs-reserved assertion at 256 concurrency |
| R8 | **Zero uncertainty on the radial budget.** `log10n0 = -2.3996`, `delta = 0.9402` are fitted to the *same* counts by `experiments/desi_ingest/calibrate_n0.py` over `[0.05, 0.3]` and passed as `--fixed_parameter_values`; §4.1's projection then makes the error unrecoverable. `delta` has absorbed whatever true radial monopole exists. **Re-sized: `medium-high` at rungs 0/1, `SEV1` at rung 2**, where the plug-in is not merely uncertain but carries the fiducial cosmology into an `H0` measurement (§1.8). | **medium-high → SEV1 under promotion** | OWNER DECISION 6: sample them under the calibration covariance, systematics-inflated; guard 5 refuses flat priors; at rung 2 an `H0`-aware prior or an `H0`-invariant reparameterization is **required**, not recommended | a budget-prior sensitivity arm in the H0 scan; kill criterion K10 |
| R9 | **Photo-z**: `sigma_z = 0.0227` (p50), `sigma_chi = 89 Mpc` — 1.8x the old radial kernel. | medium | forward-convolved into `W` (§1.4) + isotropic 190 Mpc kernel (OD3), giving 90% retention instead of 20% | P8; Tier D |
| R10 | **Resolution / anisotropy collapse** — the sphere guard only WARNS, so K4's "cut `M_sph`" escape hatch hard-fails nothing today. | medium | guards 3 and 4 (hard, both sides, plus isotropy) | Tier-A slope gate |
| R11 | **Frozen-`W` bias.** The exact `pi_pg` is theta-dependent through the within-shell weighting; `Om0/w0/wa/delta` do not cancel pointwise. | medium | `W` frozen and stamped; P7 measures the induced shift in `xi_hat` against the posterior sd | P7 |
| R12 | **The old `Delta logL` criterion is not discriminating.** Measured: turning the radial Q table on moves `logL` by **-37,620 nats at H0=60** (and -40,571 / -7,905 / -5,337 / -3,435 / -2,273 at H0 = 40/80/100/120/140) while the reported `H0` is *identical* in both arms — `139.00 [138.3, 139.7]`, both railing at the prior edge (`logs/h0_scans_1119376.out`). Rev 1's 0.05-nat K2 threshold is six orders below the ambient scale. | medium | K2 restated at the **posterior** level (§9) on a non-railing configuration | the mock closure + the h0-scan harness |
| R13 | **`zmax` split** (0.75 build vs 6.0 inference) silently collapsing the count shells. | medium | §5.2: decoupled, stamped grids; guard 7 | occupancy report |
| R14 | **Tracer overlap** at K>=2 (AGN subset of galaxies). | medium (K>=2) | OWNER DECISION 9: disjoint partition | Tier E on an overlapping mock |
| R15 | **Off-footprint / above-`z_depth` under-dispersion.** `Q == 1` with zero variance over 38% of the sky and 99.99% of the budget. | medium | stated systematic; PR-8 sensitivity scan; never quoted as a marginalization | PR-8 table |
| R16 | Legacy `gp3d` tables differ from `factored-v1` by **2.0e-3** at `M_sph = 315` (not 5e-5). | low | all pins against same-convention rebuilds; delta reported (P3) | P3 |
| R17 | `settings.json` parameter-space change breaks post-processing of old runs. | low | schema bump + reader shim | `test_run_provenance` |
| R18 | Rev 1 line citations were ~1/3 wrong against its own declared base. | low | rebase to `0c5b3db`; every anchor in this document re-verified this session | — |
| **R19** | **Linear-response inadequacy.** `xi_hat(theta)` is not well approximated by `xi_hat_ref + S·Δtheta` across the `H0`/`Om0` prior, so the promoted rung silently evaluates the wrong field. | **high (new)** | second-order term, or a small set of anchors spanning the prior with interpolation between them; failing that, ship PR-6a and quote the theta-dependence as a systematic | **P7c + P7b at PR-3, day ~14** (v4: `P7` alone is a misspecification diagnostic and does not detect this — §0.5 D4); kill criterion K9 |
| **R20** | **`n0`–`H0` degeneracy manufactured into an `H0` measurement** at rung 2 (§1.8): the pinned `log10n0` carries the fiducial comoving volume, so un-conditioning converts a calibration choice into a standard-density constraint against 22.79M galaxies. | **SEV1 at rung 2; absent at rungs 0/1** | rung 2 is admissible only under an `H0`-aware `n0` prior or an `H0`-invariant reparameterization (OD6); otherwise refuse the arm | budget-prior sensitivity arm; kill criterion K10 |
| **R21** | **The galaxy-side evidence becomes the result.** At rung 1 the count channel contributes `Delta log p_count(theta)` against 22.79M galaxies while the GW side carries 259 events; a `dN/dz`-shape constraint degenerate with `delta` and with the assumed photo-z kernel could dominate the posterior and be mis-read as a dark-siren measurement. | **medium-high (new)** | mandatory channel split in §6.4; `delta` free whenever A is used; A quoted separately in any writeup | the §6.4 A/B report at PR-6b |
| **R23** | **The field cannot move the answer.** The in-support missing budget is ~1e-4 of the total (three independent estimates: 6e-5, 1.3e-4, 2.7e-4), R12 records an identical production `H0` posterior with and without the table, and most GWTC-5 PE samples sit above `z_depth`. The entire ladder may be a no-op. | **SEV1 (new, v4)** | **K0 at PR-0, day ~3** — `sum_i phi_i` and the Q-on-at-anchor oscillation, both against artifacts that exist today; fallback is the bounded-systematic result | K0 |
| **R24** | **Within-shell nonlinearity.** Eq. (1')'s linear-in-shell premise fails, so the separable Hessian, the Kronecker factorization and the theta-cancellation degrade together — they share one approximation (§0.5 D1). | **medium (new, v4)** | narrower shells up to the photo-z floor; P5c's exact-quadrature reference at reduced rank | **P5b**, **P5c** at PR-3 |
| **R25** | **The galaxy-side evidence is mis-specified or mis-attributed.** v3 dropped the `0.5\|\|xi_hat\|\|^2` Occam term (up to ~2e2 nats) and froze `log det H` without measuring its drift (a 0.1% per-mode drift is 1.9 nats at `M/2 = 1890`). | **SEV1 at rung 1 (new, v4)** | eq. (5) is now the shipped form and both terms are closed-form under linear response | **P7d**, **P7e** at PR-3 |
| **R22** | **Importance weights instead of mean-shifted draws.** An implementer reads P7's per-mode tolerance as adequate and implements design 1b; `ESS/M = 4e-17` at `M = 3780` while every determinism diagnostic passes. | **medium (new, process)** | design 1a is normative in §1.7; §6.3 states the 1b threshold explicitly so the trap is visible; P18 and P17 both fail under 1b | member-ESS diagnostic; P17 |

**Retired from rev 1's risk table:** "no footprint mask" (now structural, §1.2) and "per-member
`Neff` guard missing" (it already exists, `core.py:1250-1262`).

---

## 9. Kill criteria

* **K0 (PR-0, new in v4 — §0.5 D3) — CAN THE FIELD MOVE THE ANSWER AT ALL?** If PR-0 reports
  `sum_i phi_i < 1e-3` (the event-weighted fraction of prior mass in the in-support missing branch)
  **and** `osc_H0 [ logL(Q on) - logL(Q ≡ 1) ] < 0.1` nat at the anchor, then no amount of field
  machinery can move the 259-event posterior: the in-support missing budget is ~1e-4 of the total,
  R12 already records an identical `H0` posterior with and without the table, and PR-10 already
  concedes that most GWTC-5 PE samples sit above `z_depth`. **Stop. The honest deliverable is the
  bounded-systematic result §6.5 already names as the fallback** — run at `xi_hat` and at `±1 sigma`
  draws, quote the `H0` spread — reachable in ~3 days instead of ~36. Do not start PR-1.
  *This criterion exists because v3 scheduled both of its inputs at days 23 and 28, downstream of the
  infrastructure whose value they determine, while its own recorded numbers predicted both would fire.*
* **K1 (PR-2).** If a defensible completeness map cannot be produced — `sum_p f_p Omega_pix`
  disagrees with the published DESI area by `> 10%`, or `> 5%` of zero-count pixels cannot be
  classified — **stop**; ship mock-only capability.
* **K2 (PR-6a/PR-6b) — restated at the posterior level, and the arm changed.** Rev 1 killed on
  `max_theta |Delta logL| < 0.05` nat. Withdrawn (R12). New, and now **two comparisons, not one**:
  (i) *latent-on vs latent-off* at PR-6a; (ii) **PR-6b vs PR-6a** — theta-coupled versus frozen
  anchor — which is the comparison that measures what the promotion actually bought. If either moves
  the `H0` **median by `< 0.1 sigma` and the 90% CI width by `< 5%`** on the Tier-B/C closure **and**
  the production `h0_scan`, **stop at that rung** and publish the result as a bounded systematic
  ("LSS placement uncertainty contributes `< X` to the `H0` posterior"), deferring the rest to a
  catalog/GW pairing whose depths overlap. The production arm must be run at a configuration where
  the posterior is not railing at the prior edge, or the test is vacuous.
* **K3 (Tier A).** Fitted-vs-truth `logQ` slope `< 0.8` at the guard-compliant rank — **stop**; the
  basis is collapsing.
* **K4 (PR-6a/PR-6b) — restated, with an explicit fallback ladder.** Rev 1's 25% wall was set against
  a baseline 60-400x too large. New: if added per-evaluation wall exceeds the OWNER DECISION 5 budget
  **and** P14 cannot be met at the affordable `M_draw`, descend the ladder one rung at a time:
  **PR-6b (theta-coupled) → PR-6a (frozen-anchor ensemble) → fixed-realization systematic.** Rev 2
  jumped straight to the weakest fallback; the middle rung is a publishable deliverable in its own
  right (differentiator 1, §0.2) and must be exhausted first.
* **K5 (PR-5b/PR-6a).** If member ESS `< 2` of `M_draw` at the production configuration and P14
  cannot be met at `M_draw <= 128` — do not ship it as a marginalization. Fixed-realization
  systematic instead.
* **K6 — promoted to a primary PR-6b gate (was PR-9 only).** Any adjacent-theta `Delta logL` jump
  `> 0.1` nat over a `1e4`-point sweep. Under linear response this is **unfireable by construction**
  — `xi_m(theta)` is an affine function of `theta` with no solver, no tolerance, no iteration count
  and no history — so **K6 firing at PR-6b means the implementation is not the design**, and the
  correct response is to fix the implementation, not to abandon the rung. K6 retains its original
  destructive meaning only at **PR-6c**, where a genuine per-proposal solve exists. K6 is therefore
  also the discriminator between linear response and a re-solve.
* **K7 (Tier C).** Coverage fails (KS `p < 0.01`, or median bias `> 0.5 sigma`) and the cause is not
  a mock bug within one week — **stop and revert to table mode**.
* **K8 (PR-2) — NON-TERMINAL in v4 (§0.5 finding 13).** The `--per_pixel_completeness` change is
  **expected** to be large, and the dominant term is not the p99 partial pixel: it is the **18,682 of
  49,152 pixels with `f_p = 0`**, whose consumption weight moves from `(1-C)` to `1`, moving the
  all-sky missing budget by roughly **+45%** at `C ~ 0.5` — straight into the all-sky injection set.
  K8 is therefore evaluated on the **Tier-B/C closure and on a non-railing production configuration**
  (the shipped posterior rails at `139.00 [138.3, 139.7]`, so "1 sigma" is undefined on the arm v3
  named — the same vacuousness K2 was restated to avoid). If the shift exceeds `1 sigma` there, the
  `C`-side change **ships as its own deliverable with its own `H0` arm and its own golden**, and the
  field ladder continues behind it. v3's only branch was "stop and re-plan" at day ~9, against an
  outcome its own numbers predicted.
* **K9 (PR-3) — PROMOTION FEASIBILITY, restated in v4 on the statistic that decides it.** If
  **P7c** reports `osc_theta [ a.(xi_hat_theta - xi_hat_ref) ] >= 0.1` nat **and** P7b's
  linear-response residual cannot be brought below 0.1 nat after a second-order term and multiple
  anchors, then linear response is inadequate and a re-solve is required — which §3.4's arithmetic
  makes infeasible at production scale (~1 s against 27.5 ms). **Refuse promotion; ship PR-6a and
  quote the measured theta-dependence as a systematic.** Decidable on **day ~14**, not day ~36.
  Symmetrically, if `osc_theta Delta logL_1 < 0.1` nat the **GW-side** field shift is negligible,
  PR-6b is a no-op on that channel, and PR-6a is the honest deliverable — report the bound and say so.
  **[v4: K9 no longer fires on `tau`, which §0.3 itself showed is undetermined by a factor of 61.5.
  And its benign branch no longer licenses any statement about effect (A): the two are different
  contractions of the same residual and are not co-monotone (§0.5 D4), so (A) is gated separately by
  P7e and R21 stays live independently.]**
* **K10 (PR-6c, new) — the `n0`–`H0` degeneracy.** If the unconditional arm is taken and `log10n0`
  is not sampled under an `H0`-aware prior (or reparameterized `H0`-invariantly), any `H0` shift is
  an artifact of the fiducial calibration (§1.8). **Refuse the arm rather than publish it.** This is
  not a soft gate: the defect is the same class as the `ls_z`-in-Mpc standard ruler guard 2 exists to
  make unrepresentable, and it is larger.

---

## 10. What the research says will not work — do not attempt

* **A full per-proposal Newton re-solve of the field** (the literal reading of
  `OWNER_CONTEXT.md:182-183`). ~13 iterations at tens of ms = **~1 s against a 27.5–49.3 ms
  baseline**, a ~36x wall, and it drags `row_fac` into the per-evaluation transient (3.0–24 GB at
  256 concurrency, §2.5). Linear response (§1.7) is the affordable form and is what the same
  paragraph's "importance-sample a small fixed set of whitened latent draws" clause becomes.
* **Reusing a fixed draw set with importance weights across `theta` (design 1b).** `Var[log w] =
  ||Δxi_hat||_H^2` exactly, so `ESS/M ≈ exp(-||Δ||_H^2)` — `4e-17` at `M = 3780` for a per-mode
  displacement that passes P7's misspecification reading (§6.3). Use mean-shifted draws (1a).
* **Adding the count likelihood (F4) while leaving `xi` a plug-in `xi_hat(C)`.** The double-use
  theorem (§3.1) holds only when the missing intensity is a function of `xi` alone; the plug-in is
  safe today *only* because no count likelihood multiplies it. This is the most dangerous
  intermediate state available in this plan and must never be a shipped rung.
* Prior-draw importance sampling (`ESS/M = 1e-1303`).
* ~~Warm-starting the solve across proposals~~ — corrected in v2, **and the operational constant
  corrected again in v4 (§0.5 finding 3), because v2's was off by ~1.5 orders of magnitude.**
  `prop:warmstart` bounds `J`, but the shipped rung-2 likelihood is
  `-J - 0.5 logdet H + LSE_m ll_m`, and only the first term enjoys the second-order protection. The
  stopping rule `||grad J||_2 < eps` gives `||xi_* - xi_hat||_H <= eps` (since `H ⪰ I`), so the
  **member term moves by `<= sigma·eps` — FIRST order** — and a 0.01-nat budget at `sigma = 2.6`
  requires **`eps < 3.8e-3`, not `eps < 0.14`**. The log-det term is bounded by evaluating
  `logdet H(xi_*)` *consistently* at the same stopping point. **`H ⪰ I` is preserved under PR-6c's
  smooth saturation only because Fisher scoring is normative** (§3.4): the *observed* Hessian of a
  saturating link is indefinite, the Fisher information is not. What remains genuinely inadmissible is
  a **fixed iteration count from a history-dependent point**, which leaves the residual unbounded and
  makes `logL` a functional of the sampler's traversal order, at which point the nested-sampling
  shrinkage argument `E[log X_i] = -i/n` no longer refers to a well-defined likelihood. Under a
  deterministic anchored start with a deterministic stopping rule, `logL` is a *deterministic* function
  of `theta` whose residual is theta-**varying**, so it is governed by the same `osc` budget as
  everything else. (Moot at rungs 0/1, which have no inner solve; binding at PR-6c.)
* Materializing `Phi` — 21.7 GB at `M = 1728`, 47.6 GB at `M = 3780`, 107 GB at `M = 8505`.
* `c_mode=per_pixel` with a count likelihood — circular by the builder's own docstring.
* Compacting to event rows to save memory — the PE/injection pixel union is 49,143 of 49,152.
* Sampling `xi` in the outer chain. The production NS run has **20** free dimensions
  (`ns_joint_sel_1119811.out:253`; rev 1's "ndim = 3" is the separate `ns_sampled_theta` run);
  adding `M >= 2520` is not feasible there.
* Widening shells past the photo-z scatter — collides with the radial resolution guard (R1-SEV2-6).
* **Building the joint selection `Q` builder as the way to close the seam.** PR-7 closes it by
  deletion (§4.4). PR-7i is a 1-day, mock-only, explicitly-terminal exception for OD10's baseline
  arm; anything beyond that is work whose product the deliverable makes unnecessary, and it would
  also inherit the joint builder's missing budget gauge fixing and its 1.34 Gpc rank floor (§0.4).

**One premise in `OWNER_CONTEXT.md` that must not be carried forward.** Line 186 — *"With only
`M ~ O(10^2)` whitened coefficients this is actually plausible"* — makes its feasibility argument at
a rank the resolution and isotropy guards **forbid**. This plan's guard-compliant rank is
`M = M_sph x M_z = 315 x (8-12) = 2520-3780` (§1.5), an order of magnitude larger; the GW-side
`_SphereZGPBase` runs at `M = 192` (§1.9) precisely because it has no such guard to satisfy. **Rung 1
is feasible because the per-proposal correction is member-independent and rank-`n_theta`, not because
`M` is small.** Any argument that reaches for a smaller `M` is reaching for the prior-collapse regime
the hard radial guard exists to refuse (measured fitted-vs-truth `logQ` slope 0.04).

---

## 11. OWNER DECISIONS

1. **Which rung?** *(Rewritten for v2; was binary, is now four-way — see §0.3 and §1.1.)*
   **(a)** shell-total–conditioned + frozen `W` → theta-free ensemble (**PR-6a**; the control arm and
   the fallback; delivers differentiator 1 in full).
   **(b)** shell-total–conditioned + **linear response** → coupling for `Om0/w0/wa/delta/theta_sel`,
   `H0` still absent (**PR-6b**). ***Recommended, conditional on P7 at PR-3.*** This is the owner's
   headline, it costs ~5 d, and it *replaces* rev 2's 8-d PR-9 rather than adding to it.
   **(c)** unconditional Poisson + linear response → adds `H0` through the `n0 (c/H0)^3` budget
   (**PR-6c**). Admissible **only** with OD6 resolved and §1.8's degeneracy stated in the paper.
   **(d)** unconditional Poisson + per-proposal re-solve → **infeasible**; moved to §10.
   Conditioning costs the `dN/dz` constraint on the budget, restored explicitly by OD6.
2. **`factored-v1` jitter, `j_sph = j_z = 1e-6`?** *Recommend yes.* Makes the Kronecker identity
   exact; costs a **2.0e-3** difference from legacy tables at `M_sph = 315`, so every migration pin
   references a same-convention rebuild and the delta is reported.
3. **Smoothing scale.** *Recommend the isotropic ~190 Mpc kernel* (`ls_ang = 0.2`, `ls_z = 0.039`,
   `M_sph = 315`, `M_z = 8-12`, `M = 2520-3780`). The 50 Mpc radial fiducial is a 4:1 pancake whose
   radial signal photo-z attenuates by 80%. Cost: the analysis is explicitly a 190 Mpc-smoothed
   field and `SurveyParams.lss_corr_length_mpc = 50.0` is retired for latent mode.
4. **Per-pixel selection.** *Recommend (a):* `f_p = 1 - masked_frac` on **both** sides, with
   `C_p(z) = f_p C(z;theta)` in `dN_miss`. This is a `C`-side change with its own flag and golden.
   Fallback (b): a mask-homogeneous sub-footprint, exact without touching `C`, losing ~30% of area.
   Rev 1 shipped (c) "ignore it" by accident.
5. **Marginalization-accuracy budget → `M_draw`.** *Recommend* P14 at 0.1 nat of theta-varying bias
   across the `H0` prior, `M_draw` set by PR-5b against §6.5's sharp table
   (`M > (e^{sigma^2}-1)/(2 epsilon)`) with `sigma = ||L_H^{-1} a||_2` in the **`H^{-1}` norm**
   (v4 correction, §0.5 D4). **Cost, corrected in v4 against the right denominator (§0.5 finding 10):
   the production baseline has no member marginalization at all, so the added cost is +31% at
   `M_draw = 8`, +125% at 32 and +250% at 64 — not the +12%/+47%/+95% v3 quoted, which is the
   latent-minus-table delta.** Alternative: the fixed-realization systematic, zero cost, reduced claim.
   **Conditional added in v2:** this arithmetic holds **only** under linear response. Under any
   re-solve variant `row_fac` becomes a per-evaluation transient — 3.0 GB at `M_draw=8` and 24 GB at
   64, at 256 concurrency — and the evaluation goes memory-bound before it goes compute-bound (§2.5).
6. **Budget uncertainty.** *Recommend sampling* `log10n0` and `delta` under the
   `data/n0_calibration.json` covariance, systematics-inflated, rather than the current
   `--fixed_parameter_values` plug-in from the same counts. Guard 5 refuses flat priors either way.
   **Promoted in v2 from *recommended* to *required* under OD1(c):** at rung 2 the plug-in is not
   merely under-dispersed, it carries the fiducial comoving volume into an `H0` measurement, so
   either the calibration prior is re-derived as a function of `H0` or `n0` is reparameterized to an
   `H0`-invariant combination such as `n0 h^{-3}` (§1.8, kill criterion K10).
7. **Above `z_depth`.** *Recommend* `Q == 1` with zero variance in the headline analysis, and PR-8
   shipped as an `amp(z)` **sensitivity scan** quoted as "H0 shift under an assumed amp(z)" — never
   as an LSS-marginalized posterior, because no gate in this plan constrains `amp(z)` there.
8. **Is `b_GW` sampled? — AMENDED IN v4, and it is now a model correctness question, not only a
   statistics one (§0.5 finding 2).** Conditional on `xi`, hosts trace `e^{b_GW s}` and galaxies trace
   `e^{b_gal s}`, so the probability that a **catalogued** galaxy `j` hosts the event carries an
   excess-bias factor `exp[(b_GW - b_gal)s(x_j) - ((b_GW^2 - b_gal^2)/2)sigma^2(x_j)]`. v3 wrote the
   catalogued branch field-free while the missing branch carried `e^{b_GW s}` — so Limit I's "`xi`
   drops out identically" and P16's "physics identity, not a routing property" were properties of the
   **omission**, not of the physics. *Recommend `b_GW == b_gal` in the headline DESI run*, which makes
   Limit I exact and P16 a genuine physics gate at zero cost; wherever `b_GW` is free (OD8's mock
   campaign, Tier E at `b_2/b_1 = 2`) the excess-bias factor **must** be written on the spike weights.
   With the `(A,B)` moment tables the online cost of sampling `b_GW` is ~0; the spike factor is one
   more row gather on objects that already exist. **Storage:** projected moments 1.1 MB at
   `M_draw = 8`, 8.7 MB at 64 (§1.7).
9. **Tracer overlap at K>=2.** *Recommend a disjoint partition* (AGN / non-AGN galaxies). Trivially
   correct; strengthens the bias-ratio measurement. The GW-side mixture is safe under overlap; the
   count-side product is not.
10. **Q family for the headline run, and keeping table mode.** The production DESI table is
    **radial** (`experiments/desi_full259/data/fits/q_radial.h5`), so adopting the latent path *is* a
    change of Q family — and §1.9 adds a structural reason: the radial builder is a per-pixel
    L-BFGS-B and **cannot** move in-likelihood at any rung, so the latent path is gp3d-family by
    necessity, not preference. *Recommend* landing PR-0..6a/6b, building the anchor, and running
    **four** arms of one comparison before choosing — radial-table / gp3d-table / **latent frozen
    (PR-6a)** / **latent theta-coupled (PR-6b)** — and keeping table mode **indefinitely** as the
    golden-pinned reference against which every parity claim here is stated.
    **Recorded for v2:** the K>=2 `c_mode=selection` **table** arm **cannot be built today** at any
    K (§0.4); PR-7i is the 1-day mock-only unblock, and it is not a DESI baseline.
11. **The proposal, and the novelty search — RESOLVED IN v4 by scheduling.** The search moves to
    **PR-0** (§0.5 D3 item 4), before PR-1, not to PR-11 after PR-10. v3's own words — "the cheapest
    possible time to discover that a differentiator is already in the literature" — contradicted v3's
    own schedule, which gated the search at day ~50 while claiming a publishable differentiator at
    day ~28. No decision remains; only execution.

12. **(New in v4) What is the headline?** A reviewer's argument, which must be answered before any
    abstract is written: strip what the owner has already scored as *not novel broadly* (3-D Bayesian
    dark-siren catalog reconstruction — Cosmic Cartography) and differentiator 1 reduces to (i) the
    per-realization mean-one budget constraint, which §4.2 itself calls a **gauge fixing**
    (`prop:gauge`) — and a convention that makes an estimand well-posed is a *definition*, not a
    measurement — plus (ii) using the same realization in the event terms and in the selection
    normalization, which is a **correctness requirement** (`core.py:1372-1380` already enforces
    all-or-none), not a contribution. With the mandatory scope caveats (`Q ≡ 1` over 38% of the sky
    and ~99.99% of the budget; the monopole deleted by hand; no field above `z_depth`), what remains
    is *"we marginalize over the angular placement of missing hosts inside the surveyed volume at
    fixed budget."*
    *Recommendation:* **demote differentiator 1 from headline to enabling clause** and headline the
    **unification** — Limits I–III as one likelihood, with the theta-coupled field as the object that
    makes the interpolation real — which is also what the owner's own promoted goal says. **(a)**
    headline the unification, differentiator 1 as the enabling result *(recommended)*; **(b)** headline
    differentiator 1 and accept that it may read as a methods note; **(c)** decide after PR-0's
    novelty search, which is now three days away.

13. **(New in v4) Does the galaxy-side evidence (effect A) enter the headline posterior?** At rung 1
    the count channel contributes eq. (5) against **22.79M galaxies** while the GW side carries 259
    events, and (A) is a `dN/dz`-shape constraint **exactly degenerate** with `delta`, with the
    population-average photo-z kernel `sigma_z(z)` and with the shell edges — all frozen inside `W`,
    none sampled. *Recommend* **(a): (A) is reported as a diagnostic and never enters the headline
    posterior unless `W`'s own parameters are sampled or profiled** (§6.4's tightened rule). **(b)**
    admit (A) into the headline and sample/profile `omega_g`, `sigma_z(z)` and the binning — real work,
    and the only honest way to quote a cosmological constraint from it. **(c)** admit (A) with `W`
    frozen — refused: it is a galaxy-clustering measurement whose systematic floor is an unsampled
    forward model, wearing a dark-siren hat.

14. **(New in v4) Is the magnitude channel (F2) a likelihood factor or an anchored prior?**
    **[verified]** today it is an **anchored Gaussian prior** (`inference/prior.py:1207`,
    `sigma(M0hat) = 1.60e-4` mag from `selection_fit_union.json`); `magnitude_loglike_from_stats` is
    referenced only by its own test. So `eq:hierarchy` as written is not the model the pipeline
    evaluates. *Recommend* **(a): keep empirical Bayes, state it plainly in the paper, and let guard 5
    and K10 (extended to `theta_sel`) carry the consequences** — zero work, honest. **(b)** implement
    (F2) as a likelihood factor and drop the anchored prior — closes the hierarchy exactly, permits
    rung 2 without the `theta_sel` double count, and costs a real PR.

---

## 12. Key file references (verified on `0c5b3db`, this session)

| purpose | path |
|---|---|
| gp3d basis, solver, Laplace, eval | `darksirens/redshift/lognormal_completion.py:610` (nodes), `:635` (operator, jitter at `:650`), `:660` (solve), `:889` (eval) |
| budget renorm (table path only) | `darksirens/redshift/lognormal_completion.py:502` |
| **per-voxel survey assembly, sky-uniform base** (rev 1 mis-attributed this to `lognormal_completion.py:714-745`) | `darksirens/cli/build_lognormal_completion.py:647`; `base_vox = np.tile(base_row, (n_fit,1))` at `:744`; `fit = np.arange(n_pix)` at `:451` and `:720` |
| `Q` consumption — the identity §4.2 must match | `darksirens/redshift/completion.py:1280` (`dN_miss = (1.0 - C)*grids.dN_exp*lss`), trapezoid at `:1284`/`:1296`, `z_depth` relaxation at `:1295` |
| field normalizer, empty-pixel sum | `darksirens/redshift/completion.py:1792-1800` |
| `field_lss_q` vs `field_delta_g` exclusion (rev 1 said `:1679-1685`) | `darksirens/redshift/completion.py:1698` |
| `n_pix_total` (rev 1 said `:959`) | `darksirens/redshift/completion.py:510`; the global-table guard at `:659-699` |
| two-node member gather (the seam) | `darksirens/redshift/prior.py:711-757`; `_materialize` at `:103-112`; the K=1 `log_Z_global` cancellation note at `:281-286` |
| member marginalization; hoisted member-independent work | `darksirens/likelihood/core.py:960-971` (docstring), `:999` (`_member_leaf_bundle`), `:1250-1262` (**per-member `Neff` guard, already present**), `:1274` (`logsumexp - log M`), `:1372-1380` (all-or-none guard) |
| barrier convention — leaves must be barriered in `make_likelihood` | `darksirens/likelihood/factory.py:11-15` (docstring), `:239-282` (application) |
| memory reservation | `darksirens/likelihood/block_sizing.py:324` (`concurrent = max(1, chains, sched_max)`), `:599` / `:649` (static), `:708`/`:725` (transient, `batch_scale`), `:623` (the ~34 GB defect) |
| provenance firewall (retired in latent mode) | `darksirens/inference/q_provenance.py:35-51`, `:130-260` |
| selection-table theta check (retired) | `darksirens/cli/inference.py:686` (rev 1's `:686-858` was right for `master`, wrong for its declared base) |
| `b_miss` rule | `darksirens/cli/inference.py:2638-2673` |
| resolution guard (sphere side **only WARNS**) | `darksirens/cli/build_lognormal_completion.py:274-319` |
| selection families — F2 must be re-derived for Schechter | `darksirens/redshift/selection.py:675` (gaussian), `:725` (schechter), h-firewall at `:16-27` |
| count histogram (integer counts) | `darksirens/cli/build_lognormal_completion.py:750`, `:762` |
| **online sampled sphere × z GP — the block this work merges with** | `darksirens/sky/models.py:273-379` (`_SphereZGPBase`, `M = M_sph x M_z = 192` whitened `sky_xi_i`), `:311-332` (the three sampled hyperparameters), `:395-422` (`OverdensityGP3D`); the builder reuses the same kernel/geometry at `redshift/lognormal_completion.py:583-632` |
| multi-tracer bias absorption (offline) | `darksirens/cli/build_joint_lognormal_completion.py:239-267` |
| completeness / depth map | `experiments/desi_ingest/build_mth_map.py`, `experiments/desi_ingest/data/mth_map_nside128.h5` |
| budget calibration (the plug-in of R8) | `experiments/desi_ingest/calibrate_n0.py`, `experiments/desi_ingest/data/n0_calibration.json` |
| closure mock generator | `experiments/completeness_viz/generate_clustered_mock.py` |
| validation harness | `experiments/completeness_viz/{run_validation.sh, fit_completeness.py, plot_completeness.py}` |
| production run config / evidence | `experiments/desi_full259/sbatch_ns_joint_sel.sh`, `logs/ns_joint_sel_1119811.out`, `logs/h0_scans_1119376.out` |
| baseline timings | `docs/source/performance.md:104-116`; `scripts/profile_member_marginalization.py`; `scripts/benchmark_block_sizes.py` |
| guard-failing gp3d build attempt | `experiments/desi_ingest/sbatch_qbuild_gp3d.sh`, `logs/qbuild_gp3d_recal_1119087.err` |
| golden (the bit-identity gate) | `tests/test_unified_k1_golden.py`, `tests/golden/unified_k1_golden.json` |
| CI gate (needs Q-firewall admissions) | `tests/fast_subset.txt`, `.github/workflows/ci.yml` |

**Anchors added in v2** (all verified on `0c5b3db` this session; the seam evidence of §0.4 and the
promotion evidence of §1.7–§1.9):

| purpose | path |
|---|---|
| **seam, inference half — OPEN**: per-catalog `c_mode` and per-catalog selection fits at K>=2 | `darksirens/cli/inference.py:2365` (`_selection_c_mode_by_catalog`), `:2378-2578` (`_resolve_selection_fits`), `:2454-2465` (one LF family), `:2522-2538` (one Schechter `M_faint_offset`), `:686-860` (per-catalog Q-table theta firewall), `darksirens/inference/prior.py:676-685` (per-catalog `c_mode` sequence) |
| **seam, the only surviving K>=2 selection refusal — STRATIFIED, and it is data plumbing** | `darksirens/inference/parameters.py:460-470`; pre-load twin `darksirens/cli/inference.py:1492-1502` |
| K=2 × `c_mode=selection` end-to-end likelihood fixture (10 tests) | `tests/test_multitracer_selection.py` |
| **seam, builder half — CLOSED (1) no `c_mode`** | `darksirens/cli/build_joint_lognormal_completion.py:180-181` (assembly call), `:375-422` (argparse); default at `darksirens/cli/build_lognormal_completion.py:648`; load-time hard error at `darksirens/catalogs/lss.py:33-48` |
| **seam, builder half — CLOSED (2) no budget gauge fixing** | `renormalize_q_mean_one` uncalled in the joint builder (grep 0) vs `build_lognormal_completion.py:582`, `:990`; legacy warning `redshift/lognormal_completion.py:1191-1205`; parity test concession `tests/test_joint_lognormal_completion.py:107-112` |
| **seam, builder half — CLOSED (3) rank cannot resolve any physical scale** | `build_joint_lognormal_completion.py:73-74` (`M_SPH, M_Z = 32, 6`, `Z_NODE_HI = 3.0` → `zeta` spacing `0.2773` → `L_smooth >= 1.34 Gpc`), guard re-anchor at `:222-237`, guard itself at `build_lognormal_completion.py:274-319` |
| **seam, builder half — CLOSED (4) the binding constraint: only the joint builder can stamp a shared id** | no `--realization-set-id` in `build_lognormal_completion.py` (grep 0) vs 13 in the joint builder; uuid mint at `redshift/lognormal_completion.py:1055-1056` |
| the physically-wrong escape hatch (`--allow_unverified_shared_lss_members`) | `darksirens/inference/loaders.py:352-395` ("INDEPENDENT-fields product prior, NOT the matched shared-field prior") |
| **Q provenance firewall — the code-level argument for the promotion** | `darksirens/inference/q_provenance.py:35-51` (`_Q_CONDITIONED` = `log10n0, delta, b_miss, Om0, w0, wa`), `:19-26` (`H0` exempt, "the right fix is to interpolate Q over H0"), `:196-242` |
| `b_miss` dropped for a Q-active catalog (the tracer-bias seam) | `darksirens/inference/loaders.py:229-247`, `darksirens/cli/inference.py:2637-2676`, `q_provenance.py:47-51` |
| Q-channel GP hypers exist on `SurveyParams` but not in the by-name registry | `darksirens/core/types.py:147-149` vs `darksirens/core/constants.py:20-45` |
| **the samplers are gradient-free** (deletes `custom_vjp` and P15) | `darksirens/likelihood/block_sizing.py:294-302`; production banner `Peak model: value-only (tinyns, 256 concurrent evals)` |
| budget identity already exact in the production `c_mode` (§4.2 scope correction) | `darksirens/cli/build_lognormal_completion.py:714` (`fit = np.arange(n_pix)`), `:744` (`w_budget = np.tile(...)`, p-independent), `:730-732` (stratified routing), `:570-590` / `:975-999` (renorm call sites and the halo caveat) |
| gp3d solve is a convex GLM (`M x M` Newton); radial is per-pixel L-BFGS-B and cannot move in-likelihood | `darksirens/redshift/lognormal_completion.py:660-864` vs `:246-415` |
| the `n0` calibration carries the fiducial cosmology (§1.8) | `experiments/desi_ingest/calibrate_n0.py:86`; `data/n0_calibration.json` (`V_c_Mpc3 = 7.818e9`, `log10n0 = -2.3996`, `delta = 0.9402`, `f_sky_occupied = 0.6199`, `sum_weights = 22,787,566`) |
| the mathematical specification (theorem/proposition labels cited above) | `experiments/field_level_plan/MODEL.tex` |
| the owner's external analysis (novelty scoring, promoted goal, directives) | `experiments/field_level_plan/OWNER_CONTEXT.md` |
