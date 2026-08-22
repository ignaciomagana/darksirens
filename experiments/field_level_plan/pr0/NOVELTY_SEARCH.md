# PR-0 item 4 — exhaustive paper-by-paper novelty search (2026-08-13)

Produced by a web-searching agent against the two differentiators of PLAN §0.2;
~22 papers individually assessed through 2026-08-13. Committed as the PR-0
gate artifact required before any "first" claim (PLAN §7 PR-0 item 4, §0.5 D3).

**Scope:** dark-siren / standard-siren cosmology, galaxy-catalog completion,
LSS-aware methods, cross-correlation and field-level approaches, 2019–2026
(emphasis 2024–2026). Search date: 2026-08-13.

**Key for the three tests applied to every paper:**
- **(a)** constrains the monopole / missing-galaxy budget of the completion field *per realization*
- **(b)** propagates field realizations through the GW detection-selection normalization (not just event terms)
- **(c)** matched multi-tracer realizations of one shared latent field

---

## 1. The Dalang–Baker "variance completion" line

### 1.1 Dalang & Baker, *The clustering of dark sirens' invisible host galaxies* — arXiv:2310.08991
Proposes clustering-based ("variance") completion for incomplete catalogs, shown to beat homogeneous and multiplicative completion when catalog structure is preserved. Conceptual/estimator paper; no GW inference run.
**(a)** no per-realization constraint (no realizations at all) **(b)** no **(c)** no.

### 1.2 Dalang, Fiorini & Baker, *Large scale structure prior knowledge in the dark siren method* — arXiv:2410.03275, JCAP 01 (2026) 034
Implementation paper: variance completion compressed into a ratio function R(z, n̂) applied to gwcosmo GLADE+ K-band lines of sight, including O3 events. **Critically, it is deterministic** — an optimization, not stochastic realizations: missing counts per voxel come from minimizing a cost function; there is no posterior over completion fields and no marginalization. **However, it explicitly enforces a budget/monopole constraint:** "the total number of missing galaxies in S averages to the total expected number … from homogeneous completion within 1%" and they "renormalize … to ensure that the ratio functions of a class S at fixed redshift average to one." This is a mean-one-per-redshift renormalization of a placement ratio — structurally the same *budget-preserving* idea as Differentiator 1, applied to a single deterministic field.
**(a) YES (deterministic analogue: mean-one per-redshift renormalization of the completion ratio)** **(b)** not discussed — the paper only modifies per-event LOS priors; no statement that the ratio enters the gwcosmo selection denominator, and no marginalization anywhere **(c)** no — single tracer (GLADE+; B-band-built ratio applied to K-band LOS, one unified system).

## 2. The Leyde–Baker–Enzi "Cosmic Cartography" line

### 2.1 Cosmic Cartography I — arXiv:2409.20531, JCAP 12 (2024) 013
Bayesian reconstruction of the true galaxy field from an incomplete catalog using a lognormal LSS prior; posterior quantifies per-voxel true number counts, marginalized over cosmology and bias. **No GW likelihood is run** — output is a proposed improved host prior.
**(a)** no — total counts float, controlled by an inferred amplitude parameter **(b)** no (no GW inference) **(c)** no.

### 2.2 Cosmic Cartography II — arXiv:2507.12171, JCAP (2026)
Extends I: lognormal dark-matter field + Gaussian-process (non-parametric) magnitude distribution, joint inference of spatial field and luminosity function; validated on Millennium mocks; explicitly "prepares the method for its application to GLADE+." **They do not draw completed-field realizations for downstream use** — "we approximate the realized true number count with the Poisson rate μ" — and no dark-siren H0 inference is performed. Survey selection enters the *galaxy* Poisson likelihood normalization, not any GW selection integral.
**(a)** no — μ_abs (overall rate) is a free inferred parameter, no per-realization budget pinning **(b)** no — no GW selection function anywhere **(c)** no — single simulated catalog.

**This is the group most likely to produce the same combination next** (their stated roadmap is GLADE+ application feeding GW inference), but as of 2026-08 no paper III exists (searched; confirmed absent).

## 3. Field-level / BORG-based lines

### 3.1 Boruah, Geshnizjani & Lavaux, *Inference of Hubble constant using standard sirens and reconstructed matter density field* — arXiv:2503.07974
**The closest prior work to the "marginalize field realizations through the GW likelihood" half of Differentiator 1.** Uses BORG posterior samples of the 2M++ density field; the single-event likelihood is written as an integral over the ensemble {δ} and marginalized "using the Monte Carlo samples supplied by BORG" (in practice reducing to the posterior-mean field because the likelihood is linear in δ). Applied to GW170817 + 2M++ and 500 mock events.
**(a)** no — they *assume a highly complete catalog*; there is no missing-galaxy completion field, hence no budget constraint **(b)** no — normalization is taken as α(H0) ∝ H0³ under the completeness assumption; the field does not enter the selection term **(c)** no — single survey (2M++).

### 3.2 Tsaprazi/Jasche, Aquila-consortium BORG applications
Searched; no BORG-based GW-host or dark-siren H0 paper beyond supernova-environment work. No competing entry found.

### 3.3 Ding / Seljak / Modi field-level papers including GW hosts
Searched multiple phrasings; **no such paper exists** in the indexed literature through 2026-08. Field-level inference papers by these authors concern galaxy surveys only.

## 4. Harmonic-space / cross-correlation lines

### 4.1 Cheng & Gair, *A unified harmonic framework for dark siren cosmology* — arXiv:2603.13053 (2026)
**The main theoretical competitor for framing.** A finite-event (mock-validated, not Fisher) likelihood unifying catalog, cross-correlation, and spectral-siren methods; the missing-galaxy field is **marginalized analytically** as a conditional multivariate Gaussian given the observed field — "we effectively marginalize over all possible realizations of the unknown galaxy field."
**(a)** no — the missing-field monopole **fluctuates freely** under the Gaussian prior; nothing pins the budget per realization (their Gaussian marginalization *includes* budget fluctuations rather than removing them — exactly the variance Differentiator 1's hard constraint eliminates) **(b)** no — "the cross-correlation method does not use any kind of information on the rate evolution of GW"; the detection term stays factorized from the field, entering only event terms **(c)** partial in principle — binning notation could host multiple tracers, but the paper treats one survey; no matched realizations (no realizations at all — analytic).

### 4.2 Mukherjee, Krolewski, Wandelt & Silk, *Cross-correlating dark sirens and galaxies: GWTC-3* — ApJ 975 (2024)
Angular cross-power between GW localization maps and photometric galaxies; marginalizes GW bias parameters. Power-spectrum statistic, galaxy density treated as fixed tracer; no field realizations, no field in selection normalization, single tracer, no monopole bookkeeping. **(a)(b)(c)** all no.

### 4.3 Mukherjee et al., *LVK synergy with DESI and SPHEREx* — arXiv:2107.12787
The canonical multi-tracer dark-siren forecast: cross-correlation of GW sources with DESI and SPHEREx separately/jointly, at the **power-spectrum level**. No latent field is instantiated, no matched realizations, no completion budget. **(a)(b)(c)** all no. Closest conceptual prior to Differentiator 2's *goal* (multi-survey dark sirens) but entirely different machinery.

### 4.4 Bera, Rana, More & Bose, *Incompleteness Matters Not* — arXiv:2007.04271
Angular BBH–galaxy clustering yields H0 "regardless of whether the host galaxies … are present in the galaxy catalog." Correlation-function method; no field realizations, no budget constraint, no field-coupled selection, single tracer. **(a)(b)(c)** all no. (No later Bera/Rana completion-realization paper found.)

### 4.5 Cross-Parkin, Howlett, Giani, Blake & Davis — arXiv:2605.06783 (2026)
Sensitivity of cross-correlation H0 to covariance, bias parametrization, binning, incompleteness. Notably argues selection effects can be absorbed into theory predictions "without explicit missing population modeling" — the opposite philosophy to Differentiator 1. **(a)(b)(c)** no.

### 4.6 Sala, Cuoco, Lesgourgues et al. — arXiv:2510.08699; multi-band CSST Fisher — arXiv:2606.15844
Tomographic C_ℓ forecasts (MCMC and Fisher respectively); incompleteness-immune by construction; single tracer each; no realizations, budgets, or field-coupled selection. **(a)(b)(c)** no.

## 5. gwcosmo line (Gray et al.) and LSS extensions

### 5.1 Gray et al., *Joint cosmological and GW population inference (LOS prior)* — arXiv:2308.02281, JCAP 12 (2023) 023
Introduces the per-pixel LOS redshift prior: in-catalog sum + analytic **homogeneous** out-of-catalog term. The completion is deterministic and homogeneous; the same LOS prior structure feeds the selection computation, so homogeneous completion does reach the denominator — but there are no clustered realizations and nothing stochastic to marginalize. **(a)** trivially (homogeneous completion preserves the budget by construction, deterministically) **(b)** only in the deterministic homogeneous sense **(c)** no.

### 5.2 Scalable gwcosmo (GPU) — arXiv:2605.23538; O4a H0 — arXiv:2603.20195
Computational acceleration and updated data; LOS prior remains homogeneous-completion; no LSS field machinery. The only LSS-aware gwcosmo extension in the literature is the deterministic Dalang et al. ratio (§1.2). **(a)(b)(c)** no.

### 5.3 Datrier & Hendry — arXiv:2502.14164
Statistical test of apparent-magnitude completeness limits for gwcosmo inputs. Orthogonal; no clustering, realizations, or selection coupling. **(a)(b)(c)** no.

## 6. DarkSirensStat / MICE / systematics lines

### 6.1 Finke, Foffa, Iacovelli, Maggiore & Mancarella — arXiv:2101.12660 (DarkSirensStat)
GLADE completion via homogeneous / **multiplicative** / mask-based interpolation; β(H0) selection term computed including the completed catalog (deterministic completion enters the denominator in their formalism — moderate confidence, from code options and paper structure). No stochastic realizations, no per-realization budget (multiplicative completion notoriously *violates* sensible budgets at low completeness, which they discuss), single compiled catalog. **(a)** no **(b)** deterministic-only **(c)** no.

### 6.2 Borghi, Moresco, Tagliazucchi & Cuomo, *Echoes from the dark* — arXiv:2509.18243, A&A 706 A199 (2026)
CHIMERA-pipeline study of incompleteness + host weighting on MICECATv2 mocks; **homogeneous completion only**; selection function ξ(λ) via MC injection integration (population-level, field-free). **(a)** deterministic homogeneous budget only **(b)** no field **(c)** no.

### 6.3 Kalomenopoulos, Barbieri, Khochfar, Gair & McGibbon — arXiv:2511.12334
Simulation study: clustering accelerates H0 convergence; 25–50% complete catalogs competitive. Diagnostic, not a method. **(a)(b)(c)** no.

### 6.4 VanWyngarden, Fishbach, Vijaykumar, Guerrero & Holz, *How Low Can You Go* — arXiv:2511.04786
Mock-catalog study of required completeness depth; clustering of faint galaxies enables unbiased H0. No completion realizations, budgets, field-selection coupling, or multi-tracer fields. **(a)(b)(c)** no.

### 6.5 Hanselman, Vijaykumar, Fishbach & Holz — arXiv:2405.14818
Galaxy-weighting systematics (mass/SFR weights, wrong-redshift-distribution and wrong-host biases). No completion-field machinery. **(a)(b)(c)** no. (No Hanselman/Vijaykumar clustered-completion paper exists.)

### 6.6 Also checked, all negative on (a)/(b)/(c):
- arXiv:2503.18887 (systematic bias in dark-siren statistical methods) — bias diagnosis, homogeneous completion.
- arXiv:2212.08694 (Gair et al., Hitchhiker's Guide) — formalism review; homogeneous out-of-catalog; selection denominator with deterministic completion.
- arXiv:2505.13568 (*Luminosity of the Darkness*, Schechter in dark sirens) — LF modeling of the out-of-catalog term, deterministic.
- arXiv:2505.11268 (bright galaxy subsets) — subset selection, not completion.
- arXiv:2311.13062 (GW190412 + DESI) — single-event DESI dark siren, standard completion; relevant as *the* DESI-catalog dark-siren precedent but methodologically standard.
- SPHEREx/DESI "field-level dark siren" forecasts beyond Mukherjee 2107.12787: none found.

---

## Verdicts

### Differentiator 1 — budget-preserving completion-field marginalization
**PARTIALLY ANTICIPATED — in disjoint pieces, never combined.** The per-redshift mean-one budget renormalization exists in *deterministic* form in Dalang, Fiorini & Baker (arXiv:2410.03275); marginalization of density-field posterior realizations through GW *event* likelihoods exists in Boruah, Geshnizjani & Lavaux (arXiv:2503.07974) — but with an assumed-complete catalog and α ∝ H0³; analytic (free-monopole) marginalization over missing-galaxy realizations exists in Cheng & Gair (arXiv:2603.13053). **No paper found propagates stochastic clustered completion realizations through the GW selection normalization, and none imposes a hard zero-missing-budget monopole per realization within a marginalized ensemble.** That specific construction (logsumexp over budget-constrained realizations in numerator *and* denominator) appears novel.
**Closest single prior work:** Dalang, Fiorini & Baker 2024/2026 (arXiv:2410.03275) — deterministic mean-one-per-redshift completion ratio in gwcosmo LOS priors; runner-up Boruah et al. (arXiv:2503.07974) for realization marginalization.

### Differentiator 2 — shared multi-tracer completion realizations
**NOVEL — no prior implementation found.** Multi-tracer dark sirens exist only at the two-point/harmonic level (Mukherjee et al. DESI+SPHEREx forecast, arXiv:2107.12787; multi-band Fisher forecasts), and compiled catalogs (GLADE+) are always treated as one tracer with one completeness. No paper reconstructs one latent density field jointly from K≥2 surveys with per-tracer bias b_k and selection C_k and marginalizes *matched* realizations through a dark-siren hierarchical likelihood. Cheng & Gair's framework could in principle be extended this way but does not do it and has no realizations.
**Closest single prior work:** Mukherjee et al. 2021 (arXiv:2107.12787) — multi-tracer in goal, power-spectrum in machinery; Cosmic Cartography II is the closest in machinery but single-tracer and GW-free.

### Phrase-level claim check
*"First joint finite-event multi-tracer field-level likelihood for dark-siren cosmology marginalizing survey selection, catalog incompleteness, LSS, and GW-host bias"* — **survivable, but only because of the "multi-tracer" qualifier, and referees will contest "field-level" and "first."** Expected objections: (i) Cheng & Gair is a finite-event likelihood that marginalizes the field (analytically) with bias — a referee may call it field-level; (ii) Cosmic Cartography I/II already do field-level reconstruction with survey selection and incompleteness (just not coupled to GW data); (iii) Boruah et al. marginalize field realizations in a finite-event GW likelihood. **Recommendation:** keep the claim but add a differentiating clause — e.g., "…via explicit budget-constrained posterior realizations propagated through both the event terms and the detection-selection normalization, and matched across tracers" — and cite Dalang+, Leyde+, Boruah+, and Cheng & Gair as the partial antecedents each lacking one leg. The unqualified word "first" applied to "field-level dark sirens" alone would likely not survive; the full multi-clause claim as restated would.

**Residual caveats:** Finke et al. β-term detail (§6.1) assessed from paper structure/code options, not verified line-by-line; Dalang et al. non-propagation into the gwcosmo selection denominator is inferred from silence in the paper (gwcosmo's denominator does consume LOS priors, so a referee could argue their ratio implicitly reaches it — worth one sentence of preemption in the paper). Literature checked through 2026-08-13; Cosmic Cartography III (GLADE+ application) is the obvious scoop risk and should be monitored until submission.
