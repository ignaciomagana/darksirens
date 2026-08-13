# Review findings on the field-level latent design, and their dispositions

**Two parts.** Part 1 (everything up to *Summary of dispositions*) records the two independent
adversarial reviews of **rev 1** and their resolution in **rev 2**. Part 2 — **`v2 (owner context)`**,
at the end of this file — records what the owner's external analysis (`OWNER_CONTEXT.md`) changed
about **rev 2 → v2**: which findings changed *role* under the promoted goal, which of rev 2's own
answers were wrong, how the two staleness claims in the owner context adjudicated against
`master @ 0c5b3db`, and eleven facts verified this session that neither reviewer had.

Two independent reviews were run against rev 1 of the design. Both returned `verdict: fixable`.
Part 1 records every finding **verbatim** (claim and argument as submitted) followed by the
architect's resolution and the section of `PLAN.md` that carries it.

Disposition codes:

* **RESOLVED IN DESIGN** — the design changed so the defect cannot occur.
* **CONVERTED TO OWNER DECISION** — the tradeoff is real and the owner must choose; stated with both
  arms and a recommendation.
* **ACKNOWLEDGED IN RISK TABLE** — accepted as a bounded, monitored residual with a detection
  mechanism.
* **CORRECTED** — a factual/numeric/citation error, fixed.
* **WITHDRAWN** — a rev-1 claim removed as false.
* **ROLE CHANGED** *(v2 only)* — finding and resolution both stand, but under the promoted goal the
  finding means something different.
* **CORRECTED IN v2** *(v2 only)* — rev 2's own answer was wrong or too strong; a self-correction,
  not a reviewer finding.

Independent verification performed this session is marked **[verified]** with the command output
that established it.

> **Note on section numbering.** Part 1 cites `PLAN.md` section numbers as they stood in rev 2.
> v2 inserted §0.0–§0.4, §1.6–§1.9 and §2.5, so a few of those cites now point one subsection
> earlier than the material they describe; the section *titles* quoted in each disposition are
> unchanged and remain the reliable pointer.

---

# Reviewer 1

## R1-SEV1-1 — the member weights are not uniform

**Claim (verbatim).** "§0.1(5) is wrong: the inner marginalization over xi is importance sampling
with NON-uniform weights, and the plan's own numbers put M_draw=8 below its own kill threshold.
There is no M_draw-convergence gate anywhere in §6.3/§6.4."

**Argument (verbatim).** "§0.1 item 5 states \"the proposal distribution *is* the target, so the
weights are exactly uniform and logsumexp(ll_m) - log M stays as-is.\" That is false. The proposal
p(xi|d_gal) is the target of the *outer* factorization, not of the inner integral ∫
p(d_GW|d_gal,xi,Λ) p(xi|d_gal) dxi. The weights are exactly w_m = exp(ll_m) — the per-member GW
likelihoods — which the plan itself sizes at \"up to ~2.6 nats\" of spread in §1.5 and §6.4.
Verified in code: darksirens/likelihood/core.py:1273 does `return logsumexp(ll_members) -
jnp.log(n_members)`, i.e. a plain log-mean-exp of non-uniform weights, and each ll_m carries its own
-N_obs log mu_m via selection_log_correction inside `_member_ll` (core.py:1249-1262). So §0.1(5)
contradicts §1.5/§6.4 in the same document. Quantitatively, for lognormal weights with sd sigma nats
and M draws: sigma=1.0 -> E[ESS]=2.9 of 8 and E[log Zhat]-log Z = -0.11 nats; sigma=1.5 ->
ESS=0.84, bias -0.53 nats; sigma=2.6 -> ESS=0.01, bias -54 nats. The plan's own §0.2 sets 0.01 nats
as \"the scale that matters at 259 events,\" and §7.4-K5 kills at ESS<2 of 8. Cross-checking with
the plan's other number (report-3 posterior linear-predictor sd s_v ~= 5.6e-3, §0.2):
Cov(Q,sens)/<sens> <~ 0.0056 -> sigma(log mu) x 259 ~= 1.45 nats -> ESS ~= 1.0 and bias ~= 0.5 nats.
The plan's own two numbers therefore jointly predict K5 fires and the estimator is biased ~50x above
the stated tolerance. R7's mitigation (\"raise M_draw to 16-32\") is inadequate: at sigma=2.6 you
need M ~ e^{sigma^2} ~= 860 draws for ESS~8. Worse, because g_m is fixed CRN the estimator is
*deterministic*, so every diagnostic in §6.4 (determinism sweep, 100 repeat evaluations
bit-identical, no adjacent-theta cliffs) will pass while the sampler explores a systematically
distorted surrogate target whose bias varies with H0 (the member spread runs through -N_obs log mu,
which is H0-dependent). Nothing in §6.3's pin table or §6.4's diagnostics compares logL at M_draw=8
against M_draw=64 as a function of H0 — the only test that can detect this. ESS is a proxy, not a
bound on the bias."

**Disposition: WITHDRAWN + RESOLVED IN DESIGN + CONVERTED TO OWNER DECISION 5.**

The uniform-weights claim is struck from the plan and replaced by an explicit statement of the
defect (`PLAN.md` §1.1 item 5). **[verified]** `core.py:1274` is `logsumexp(ll_members) -
jnp.log(n_members)`; `core.py:1250-1262` calls `selection_log_correction(log_mu_m, Neff_m, ...,
pe_variance_sum=...)` inside the member vmap. Four structural changes (`PLAN.md` §6.5):

1. **PR-5b**, a new blocking rung: emit `ll_m` at `M_draw = 256` across 33 `H0` nodes and report
   `sigma(H0)`, `ESS(H0)` and `log Zhat_M - log Zhat_256` for `M in {4,...,128}`. The number is
   measured before PR-6 ships, not argued.
2. **Pin P14**, the gate the reviewer says is missing: the shippability criterion is the
   **theta-variation** of the bias, not its level — `max_H0 [(log Zhat_M - log Zhat_256) -
   mean_H0(...)] < 0.1 nat`. The reviewer's own observation that CRN makes the estimator
   deterministic is what makes the level absorbable into the evidence and the variation the thing
   that matters.
3. **Antithetic draws** (`g_m` in `+-` pairs) as a free variance reduction.
4. **Large `M_draw` is now affordable** — the 90x compression (`N_grid/M_z`) takes the member
   payload from 13.7 GB to 94 MB at `M_draw = 64`.

If P14 cannot be met at `M_draw <= 128`, PR-6 does not ship as a marginalization; the fallback is a
**fixed-realization systematic** (run at `xi_hat` and `+-1 sigma` draws, quote the `H0` spread).
K5 is restated accordingly. OWNER DECISION 5 makes the accuracy budget and its cost explicit.

---

## R1-SEV1-2 — per-pixel incompleteness absorbed into xi with the wrong sign

**Claim (verbatim).** "The count channel absorbs unmodeled per-pixel incompleteness into xi, and
then applies it to the missing budget with the WRONG SIGN. §4.4 guard 5 refuses c_mode=per_pixel —
the only mode that could model it — precisely because it does not cancel."

**Argument (verbatim).** "Eq. (1) models N_pg ~ f_p exp(b f) with a sky-uniform completeness that
cancels. The production configuration has no sky-uniform completeness:
experiments/desi_ingest/data/selection_fit_union.json is a SINGLE stratum (\"all\", gaussian,
m_lim=21) for a footprint whose per-pixel selection varies strongly. Measured from
experiments/desi_ingest/data/mth_map_nside128.h5: masked_frac over the occupied footprint has mean
0.137, sd 0.104, 1st-99th percentile 0.047-0.635 — a factor-13 spread in per-pixel masked fraction.
The production stratum map (data/stratum_map_ns_nside64.h5) is a north/south DR9-vs-DR10 split, not
a depth split, and the headline fit does not use strata at all. Write the true intensity as f_p
C_p(z) nbar(z) e^{b delta_p}; eq. (1) fits f_p Cbar(z) nbar(z) e^{b f_p}. The solution absorbs b
f_hat = b delta_true + log(C_p/Cbar). Now the consumption side is verified at
darksirens/redshift/completion.py:1280: `dN_miss = (1.0 - C) * grids.dN_exp * lss`, with C
sky-uniform under c_mode=selection. In an under-complete pixel (C_p < Cbar) the fitted Q is
SUPPRESSED, while the true missing budget (1 - C_p) > (1 - Cbar) is ENHANCED. The
completeness-induced component of Q enters dN_miss with the opposite sign to the truth, and it is
compounded by (1-C) itself being wrong. f_p as specified is a bare area/mask fraction, not a
completeness, so it cannot absorb this. This directly violates the plan's own doctrine \"C says how
much, Q says where\": Q is now carrying a *completeness*, i.e. a how-much. The plan's mask gates
cannot see it — K1 (§7.4) tests total area to 10% and PR-2 tests sum_p f_p apix to 1%; both are
monopole tests, and §4.1's proj_sph = I - c c^T deliberately projects the monopole out. Tier D's
\"mask deliberately wrong by 5%\" is also mis-sized: the plan's own OWNER DECISION 5 puts the true
50-Mpc-smoothed delta rms at 0.1-0.2, so a 5-12% smooth selection error is 25-100% of the signal
being measured. This is not a plumbing fix: guard 5 refuses per_pixel on the grounds that C_p \"does
not cancel from pi_pg\", but the thing that does not cancel is exactly the thing that must be
modeled. The repair is to put the KNOWN per-pixel selection (mask x depth map, fixed data, not a
parameter) into pi_pg as an offset AND into (1-C_p) on the consumption side — which is a change to
decision 0.1, not to the plumbing."

**Disposition: RESOLVED IN DESIGN + CONVERTED TO OWNER DECISION 4 + ACKNOWLEDGED (R2).**

The finding is correct and the prescribed repair is adopted (`PLAN.md` §1.2). Independent
measurement narrows it usefully:

**[verified]** from `experiments/desi_ingest/data/mth_map_nside128.h5`:
`masked_frac` over occupied pixels mean **0.1368**, sd **0.1039**, p1 0.0470, p50 0.1126, p99
**0.6347** — the reviewer's numbers reproduce exactly. But also: `mth_eff = min(21, median_m5)` has
**p1 = p50 = p99 = 21.000** and `median_m5` has p1 = 22.925, and `stratum_edges` is
`[19.025, 21, 21, 21, 21]` — **degenerate**. The survey is limited by the `m_retention = 21` cut
everywhere, not by imaging depth. So the per-pixel *magnitude-depth* variation is negligible and the
entire per-pixel selection variation is **areal masking** — achromatic in `z` and in magnitude.

That makes the repair exact rather than a model:

```
f_p = 1 - masked_frac_p,   C_p(z;theta) = f_p * C(z;theta)
```

used on **both** sides — as the offset in `pi_pg` (where it already was) **and** as the consumed
completeness in `dN_miss` (`completion.py:1280`) and in the field normalizer (`:1792-1800`), where
rev 1 had a sky-uniform `C`. The sky-average completeness is preserved as `<f_p> C(z)`, so "C says
how much, Q says where" survives. This is a `C`-side change and ships behind its own flag
(`--per_pixel_completeness`) with its own golden (PR-2), plus kill criterion **K8**: if it moves the
production `H0` posterior by `> 1 sigma` with the field off, that is the headline result and the plan
re-plans.

Residual — unmodelled `p`- and `z`-dependent selection (fibre assignment) — is **risk R2**, with
Tier-D stress (ii) at 5% amplitude and a `masked_frac`-vs-`xi_hat` correlation report at PR-3.
**OWNER DECISION 4(b)** offers the exact fallback: restrict the count channel to a mask-homogeneous
sub-footprint, which is trivially correct without touching `C`. Tier D is resized per the reviewer's
point: the perturbation is now the measured `masked_frac` sd (0.104 in completeness), 50-100% of the
signal, not 5%.

---

## R1-SEV1-3 — the budget identity pinned is not the identity consumed

**Claim (verbatim).** "§4.2's budget-conservation identity is not the identity that is consumed, so
the PR-6 gate pins the wrong invariant and the +55% Jensen inflation is not actually removed."

**Argument (verbatim).** "§4.2 defines the in-likelihood renormalizer with w_p(z;theta) = f_p (1 -
C_p(z;theta)) dN_exp(z;theta) and claims \"Exact per-realization budget conservation: Sum_p w_p Q_p
/ Sum_p w_p = 1 identically\"; §6.3 and the PR-6 gate pin exactly that at 1e-12. But §1.2 (F5)
explicitly says the consumption is \"unchanged in form: dN_miss = (1 - C(theta)) dN_exp(theta)
Q(xi,theta) (completion.py:1280)\" — verified verbatim in darksirens/redshift/completion.py:1280,
with no f_p and with N_miss = trapezoid over the FULL grid at completion.py:1283/1294. The consumed
budget is therefore Sum_{all sky} (1-C) dN_exp Q, whose conservation requires normalizing with
weights that do NOT carry f_p, summed over the same pixel set the integral runs over. The gap is
exactly the covariance between f_p and Q. That covariance is large, not second-order: the DESI
footprint is f_sky = 0.62 (verified: 30,470 of 49,152 occupied pixels in
data/pixelated_n64/catalog_pixelated_nside_64.h5), so 38% of the sky has f_p = 0 and gets Q drawn
from the pure prior at amp = 1 (a factor of e), while the injection set is all-sky
(selection_o3o4ab_allsky.h5 in experiments/desi_full259/sbatch_ns_joint_sel.sh). Per-draw, the
all-sky missing budget is then inflated or deflated by an O(field sd) factor — reintroducing the
very Jensen-type budget error that renormalize_q_mean_one exists to kill (measured +55%,
lognormal_completion.py:502-512). Note also that the current convention is the opposite of the
plan's: under c_mode in {aggregate, selection} the builder sets `fit = np.arange(n_pix)`
(cli/build_lognormal_completion.py:722) so renormalize_q_mean_one already sums over the full sky
with no f_p. Inserting f_p into the normalizer while leaving consumption unchanged is a silent
change to the angular distribution of the missing budget — a C-side change riding in under a Q-side
flag, and it is not covered by any bit-identity pin because every golden cell in §4.5/§6.3 is stated
with latent mode OFF."

**Disposition: RESOLVED IN DESIGN.**

**[verified]** `completion.py:1280` = `dN_miss = (1.0 - C) * grids.dN_exp * lss`; trapezoid over the
full grid at `:1284`/`:1296`; `fit = np.arange(n_pix)` at `build_lognormal_completion.py:451` and
`:720`; occupied nside-64 pixels 30,470 of 49,152 (`f_sky = 0.6199`).

Rev 2 states **one** identity and uses it in both places (`PLAN.md` §4.2), and it is the consumed
one, now including the §1.2 per-pixel completeness:

```
sum_{p=1..n_pix} (1 - f_p C(z;theta)) Q_p(z)  /  sum_{p=1..n_pix} (1 - f_p C(z;theta))  ==  1
```

Pin **P13** is restated on exactly this expression, over exactly the pixel set the integral runs
over. The 38%-of-sky problem the reviewer identifies is closed by an explicit convention:
**`Q == 1` off-footprint and above `z_depth`** ("no information, no modulation"), so no pixel ever
draws `Q` from a pure `amp = 1` prior. That convention is itself a choice that *under-disperses* —
it assigns zero variance where there is no data — and it is carried as **risk R15** and bounded by
the PR-8 `amp(z)` sensitivity scan (OWNER DECISION 7), not hidden.

---

## R1-SEV2-4 — theta-freedom rests on an unstated shell-collapse approximation; the pin is a tautology

**Claim (verbatim).** "p(xi|d_gal) is theta-free only under an unstated shell-collapse
approximation, and the 1e-12 theta-invariance pin (the pin the plan says \"proves decision 0.1\") is
a tautology that cannot detect the error."

**Argument (verbatim).** "The exact conditional given the shell total is pi_pg = Lambda_pg / Sum_p'
Lambda_p'g with Lambda_pg = f_p integral_{shell g} base(z;theta) exp(b f(p,z)) dz. base(z;theta)
does not factor out of the integral, so it cancels only in the limit where f(p,.) is constant across
the shell. It is not: the plan's own ls_z = 0.0103 in zeta = log1p(z) units, and the shell width is
Delta_z = 0.024 at z~0.24 (z_s = np.linspace(0.0, zgrid[-1], gp3d_nz_solve=32),
cli/build_lognormal_completion.py:880, with the Q build at DARKSIRENS_ZMAX=0.75 per
experiments/desi_ingest/run_qbuilds.sh), i.e. Delta_zeta = 0.019 ~= 1.9 ls_z. The field varies by
O(1) sd across a shell, and base varies ~20-40% across a shell (dV/dz alone changes ~17% over
Delta_z=0.024 at z=0.24). So the within-shell weighting is a real, theta-dependent reweighting of
the field. Working out which parameters actually survive: H0 genuinely does cancel, because dV/dz =
(c/H0)^3 x f(z;Om0) is an H0-independent SHAPE times a constant, and C_sel is H0-free by the
h-firewall (redshift/selection.py:16-27) — so decision 0.1's claim #3 holds. But Om0, w0, wa and
delta do NOT cancel: they change the within-shell shape of base(z;theta). The plan retires the whole
_Q_CONDITIONED set (inference/q_provenance.py:35-42 = {log10n0, delta, b_miss, Om0, w0, wa}) as
\"provably unnecessary,\" and PR-6's gate explicitly requires that latent mode *permits* sampling
Om0/w0/wa. That permission is not established. Worse, PR-3's gate — \"theta-invariance pin:
identical to 1e-12 across 20 random theta (this is the pin that proves decision 0.1)\" — tests
conditional_multinomial_logl, which is theta-free by construction because eq. (1) has no theta
argument. The pin can only fail if someone passes theta into a function that has no theta parameter.
It proves nothing about the model."

**Disposition: RESOLVED IN DESIGN + ACKNOWLEDGED (R11).**

The approximation is converted from implicit to explicit, frozen and stamped (`PLAN.md` §1.1). The
within-shell weighting becomes the **shell-response operator** `W (G_s, N_fine)`, built once at the
anchor `theta_ref`, covered by the artifact sha256, and used identically offline and online:

```
pi_pg(xi) = f_p sum_n W_gn exp(b_gal f(p,z_n)) / sum_p' f_p' sum_n W_gn exp(b_gal f(p',z_n))
```

With `W` frozen, `pi_pg` is theta-free **by construction**, not in a limit. The residual is
`d pi_pg / d theta` at frozen `W`, which is measured by the pin that replaces the tautology:

**P7** — compute the *exact* theta-dependent `pi_pg` (base inside the shell integral) at 20 theta
drawn from the prior and require `max_theta ||xi_hat(theta) - xi_hat(theta_ref)||_H / sqrt(M) < 0.1`
posterior sd and `|Delta log p_count| < 0.05` nat. Risk **R11** carries what P7 does not remove.
The reviewer's `H0`-cancels / `Om0,w0,wa,delta`-do-not analysis is adopted verbatim into the plan's
statement of what guard retirement rests on.

`W` also carries the photo-z forward convolution, so one object resolves this finding and
R1-SEV2-6 together.

---

## R1-SEV2-5 — b_gal cannot be both sampled and fixed

**Claim (verbatim).** "b_gal cannot be both sampled (§4.3) and fixed in the offline solve (§2.1). If
sampled, decision 0.1's theta-independence fails; if fixed, the field-amplitude uncertainty is never
propagated and Xi_m is under-dispersed."

**Argument (verbatim).** "§2.1 specifies `xi_hat, H_chol = count_map_solve(basis, tracers, ...)` run
once offline, and §3.1's TracerCounts carries `bias: float` — a concrete Python float, so b_gal is
frozen at the offline solve. But §4.3 says \"Fix amp = 1 in the kernel; sample bias only\" and
\"b_gal (per tracer k): multiplies f in the count rate. Identified at K>=2 in the ratios b_k/b_1.\"
These are incompatible. p(xi | d_gal, b_gal) depends on b_gal — the count likelihood constrains the
product b_gal*f, so b_gal is exactly the parameter the count channel measures. If b_gal is sampled
in the outer chain, xi_hat and H_chol must be recomputed per proposal and the entire premise of
decision 0.1 (no per-proposal solve, no log-det, no firewall) collapses. If b_gal is fixed, then
§4.3's identifiability claim and PR-7's Tier-E gate (\"K=2 recovers b_2/b_1 within 2 sigma\")
describe a quantity that never enters the sampler, and — more consequentially — H^{-1} is a
covariance CONDITIONAL on b_gal being exactly right, so the Laplace draws Xi_m understate the field
uncertainty by the amount attributable to the amplitude. §6.2's Tier-B accept criterion
(\"latent-on CI width >= table CI width — marginalization cannot tighten\") is then not a valid
check: it compares against a point estimate, not against the correctly-dispersed marginal, and will
pass while the marginalization is under-dispersed. The plan needs to state which of the two it means
and, if fixed, carry the b_gal Laplace variance into H_chol (a rank-1 inflation along the direction
dxi_hat/db)."

**Disposition: RESOLVED IN DESIGN (the reviewer's own prescription).**

`PLAN.md` §3.4 states plainly: **`b_gal` is fixed at the anchor in the count solve**, and its
uncertainty is propagated by the rank-1 inflation the reviewer specifies:

```
Cov(xi) = H^{-1} + s_b^2 (d xi_hat/db)(d xi_hat/db)^T,   d xi_hat/db = -H^{-1} (d grad/db)
```

from the implicit function theorem — one extra triangular solve, a new `bias_sensitivity` entry
point in `latent_counts.py` and a new artifact field. At K=1, `b_gal` and `amp` are degenerate so
`s_b` is a prior width (recommend 20%); at K>=2 the `b_2/b_1` block is inflated from its own 2x2
profile curvature and Tier-E measures a quantity that now genuinely enters the ensemble. Tier-B's
"latent-on CI >= table CI" is restated as valid **because** of this change, with the reviewer's
objection recorded inline in §6.2.

---

## R1-SEV2-6 — photo-z is not absorbable by shell width, and the cheap fix collides with the guard

**Claim (verbatim).** "Photo-z is not a \"medium\" risk that shell width can absorb: the measured
DESI redshift errors are ~2x the assumed radial correlation length, and OWNER DECISION 4's proposed
fix collides head-on with resolution guard 3."

**Argument (verbatim).** "Measured directly from
data/pixelated_n64/catalog_pixelated_nside_64.h5 over a 4,000-pixel sample: median dzgals = 0.0238,
75th pct 0.0453, 99th pct 0.0890; median dz/(1+z) = 0.0197. At z ~ 0.24 that is a comoving smearing
of ~85-90 Mpc, versus the assumed ls_z = 50 Mpc (SurveyParams.lss_corr_length_mpc = 50.0,
core/types.py:147). The radial modes the count channel is supposed to measure are therefore smeared
BELOW the kernel scale, so the recovered radial field amplitude is attenuated by roughly exp(-k^2
sigma_chi^2/2) at the kernel scale — a large, systematic, scale-dependent suppression. Because b_gal
is frozen offline (see the prior finding), that attenuation is not absorbed anywhere; it propagates
straight into xi_hat, and then into dN_miss on the GW side where the field is treated as truth. R9
correctly notes that the count assembly histograms zgals and ignores dzgals (verified: `counts_s, _
= np.histogram(zs, bins=edges_s)` at cli/build_lognormal_completion.py:750/762 uses zgals only,
while the GW side builds per-galaxy kernels in redshift/completion.py:379-382). But OWNER DECISION
4's cheap option — \"widen shells to exceed the photo-z scatter\" — is not available: the current
build already has Delta_z = 0.024 ~= sigma_z, so widening past the scatter means G_s <~ 12 shells
over [0, 0.3], while §4.4 guard 3 (the hard radial resolution error,
cli/build_lognormal_completion.py:295-306) demands M_z >= 27. The plan's own table already lists
Phi_z_solve as (G_s, M_z) = (32, 27) — nearly saturated. Widening makes M_z > G_s and the radial
field entirely prior-driven, which is the exact prior-collapse failure mode guard 3 exists to
prevent (measured fitted-vs-truth slope 0.04). Only the expensive option (forward-convolve pi_pg
with the per-galaxy kernel) is actually consistent with the rest of the design."

**Disposition: RESOLVED IN DESIGN; rev 1's OWNER DECISION 4 RETIRED.**

**[verified]** over 2,238,680 galaxies from the production catalog: `dz` p50 = **0.0227**, p75 =
0.0398, p99 = 0.0900; `dz/(1+z)` p50 = 0.0188; `z` p1/p50/p99 = 0.0604/0.2371/0.2990. At
`z_ref = 0.237`, `dchi/dz = 3918 Mpc`, so `sigma_chi = 89 Mpc` — the reviewer's numbers.

The cheap option is struck from the plan and listed under §10 "do not attempt". Rev 2 takes the
expensive-but-correct option (`PLAN.md` §1.4) and pays for it once, offline: **convolve the model,
not the data.** The photo-z kernel is folded into the shell-response operator `W`, so counts stay
integer, the multinomial stays exact, and the radial attenuation lives in the *forward model* where
it neither biases `b_gal` nor is absorbed into `xi`. Pin **P8** gates the population-average kernel
against per-galaxy kernels (`||Delta xi_hat||_H / sqrt(M) < 0.1`, else ship per-galaxy).

The attenuation magnitude is what forces **OWNER DECISION 3**: at a 50 Mpc radial kernel the
retention is `exp(-(89/50)^2/2) = 0.20`; at 190 Mpc it is 0.90. The plan therefore also changes the
kernel (see R1-SEV2-10).

---

## R1-SEV2-7 — the shell grid inherits DARKSIRENS_ZMAX

**Claim (verbatim).** "The shell grid inherits DARKSIRENS_ZMAX, so the plan's own cost model is
internally inconsistent and at the production zmax the count channel has ~2 usable shells for 27
radial modes."

**Argument (verbatim).** "z_s is built as `np.linspace(0.0, float(zgrid_np[-1]),
int(gp3d_nz_solve))` with gp3d_nz_solve defaulting to 32 (cli/build_lognormal_completion.py:880,
:815, :1116), and zgrid[-1] = DARKSIRENS_ZMAX (redshift/grid.py:25-27). §2.4 fixes the production
configuration as \"N_grid = 1086 (DARKSIRENS_ZMAX=6.0); G_s = 32\", while R9 simultaneously says
\"coarse shells (G_s = 32, Delta_z ~= 0.023)\". Delta_z = 0.023 implies zmax ~= 0.74, not 6.0. Both
cannot hold. This is not a typo in the plan — it reflects a real split in the pipeline:
experiments/desi_ingest/run_qbuilds.sh exports DARKSIRENS_ZMAX=0.75 for the Q build, while
experiments/desi_full259/sbatch_ns_joint_sel.sh exports DARKSIRENS_ZMAX=6.0 for the inference. Today
that is harmless because the production run carries no LSS at all (--use_lss false, no
--lss_completion). Under the plan it is not: the latent basis and the count shells must be built on
the same zgrid the likelihood evaluates, so if the anchor artifact is built at the production
zmax=6.0, z_s = linspace(0, 6, 32) gives Delta_z = 0.194 and the entire catalog (verified z 1st-99th
pct = 0.061-0.299, z_depth = 0.3) falls in the first one or two shells. The multinomial then
conditions away essentially all radial information and M_z = 27 radial modes are fitted to ~2 data
shells. The plan must state which zmax the anchor is built at and add a gate on the number of shells
with non-trivial occupancy; neither appears in §6.3."

**Disposition: RESOLVED IN DESIGN.**

`PLAN.md` §5.2 makes the two grids **decoupled and separately stamped**: `z_count_edges` (`G_s + 1`
edges over `[0, z_depth]`, stored in the artifact, **never** derived from `DARKSIRENS_ZMAX`) carries
`phi_z_fine` and `W`; the production `zgrid` (`N_grid = 1086` to `zmax = 6.0`) carries `phi_z_out`
restricted to `z <= z_depth` (`N_z_sub ~ 60-70` nodes, `Q == 1` above). `G_s` is set by the
photo-z-limited radial resolution, not by `gp3d_nz_solve`'s default. New **guard 7** enforces
occupancy (`>= 1e4` galaxies and `>= 500` occupied pixels per shell) at likelihood-build time and is
a PR-4 gate. Risk **R13**. The `Delta_z = 0.023` / `N_grid = 1086` contradiction is removed from the
cost model.

---

## R1-SEV2-8 — zero uncertainty on the radial budget; the plug-in is unaccounted

**Claim (verbatim).** "The design assigns exactly zero uncertainty to the radial budget: the
monopole is projected out of the field AND n0/delta are plug-ins fitted to the same counts. §1.3's
\"each observable appears in exactly one density\" omits the plug-in."

**Argument (verbatim).** "§4.1 applies proj_sph = I - c c^T so the field cannot represent any
sky-constant mode, and §0.1 drops F3 = p({T_g}|theta) entirely. So the model asserts the observed
dN/dz is exactly the fiducial with no error bar. Meanwhile the fiducial itself is a fit to those
same counts: experiments/desi_ingest/calibrate_n0.py computes n0 = sum(w) / (f_sky_occ * C_eff *
V_c) and fits delta \"from the observed dN/dz shape against C_sel(z) (1+z)^delta dVc/dz over [0.05,
0.3]\", producing data/n0_calibration.json (log10n0 = -2.3996, delta = 0.9402, sum_weights =
22,787,566). experiments/desi_full259/sbatch_ns_joint_sel.sh then passes those two numbers via
--fixed_parameter_values, i.e. fixed at zero uncertainty. So T_g IS used in the analysis — as a hard
plug-in constraint — while the plan's §1.3 factorization argument accounts only for the three
densities F2/F4/F5 and never mentions the plug-in. That is not double counting of a density, but it
is a use of d_gal whose uncertainty is discarded, and it is precisely the uncertainty that the
projection then makes unrecoverable. Two consequences the plan should own: (i) §4.4 guard 4's
justification (\"the count channel carries ZERO information about (n0, delta, theta_sel) by
construction\") is correct but cuts the other way — the anchor is a point estimate from the same
data, so the budget error is uncontrolled, not protected; (ii) delta = 0.94 was fitted over [0.05,
0.3] as a residual to C_sel dVc/dz, so it has absorbed whatever true radial monopole exists in the
DESI volume, and the projection then guarantees the field cannot correct it. §7.4's kill criteria
contain no budget-uncertainty test."

**Disposition: CONVERTED TO OWNER DECISION 6 + ACKNOWLEDGED (R8).**

The plan now states the plug-in explicitly as a use of `d_gal` whose uncertainty is discarded
(`PLAN.md` §1.1 OD1 note, risk **R8**), and offers the fix: **sample `log10n0` and `delta` under the
`data/n0_calibration.json` covariance, systematics-inflated**, rather than passing them through
`--fixed_parameter_values`. Guard 5 is restated: latent mode refuses a **flat** prior on those
parameters (rev 1's blanket refusal cut the wrong way, as the reviewer notes) and **requires** the
calibration prior or `--allow_unanchored_budget`. A budget-prior sensitivity arm is added to the H0
scan as the detection mechanism. The reviewer's point (ii) — that `delta` has absorbed the true
radial monopole which the projection then makes unrecoverable — is recorded verbatim in R8 as the
reason the prior must be systematics-inflated rather than taken at its formal width.

---

## R1-SEV2-9 — above z_depth the marginalization is 100% prior and cannot be validated

**Claim (verbatim).** "Above z_depth the marginalization is 100% prior, and the plan's own mock
campaign is structurally incapable of validating it, so PR-8's \"LSS-marginalized H0\" would be an
assumption reported as a systematic."

**Argument (verbatim).** "R1 states the field's radial nodes stop at z = 0.30 with Phi = 0 above z ~
0.33 and 99.994% of the missing budget above — verified against the build (--gp3d-z-node-hi 0.30 in
experiments/desi_ingest/run_qbuilds.sh; catalog z_depth = 0.3, 99th pct z = 0.299) and against the
production inference range (DARKSIRENS_ZMAX=6.0). OWNER DECISION 5 / PR-8 propose extending the
field above z_depth with an amp(z) growth profile and reporting the result as an LSS-marginalized
H0. But there are no counts above z_depth, so above z_depth the field is drawn entirely from the
prior: the width of the resulting H0 posterior is a pure function of the assumed amp(z) and
ls_sph/ls_z, multiplied over 49,152 pixels and injected into -259 log mu. §6's validation cannot
test this. Tier A-C draw xi_true from generate_clustered_mock.py, which §6.1 states \"draws the
clustering truth from the same low-rank GP family the builder fits\" — so coverage is guaranteed by
construction in exactly the regime where there is no data. Tier D perturbs ls_sph and the tail
shape, not the amplitude profile in the un-sampled redshift range. A number produced this way is not
a marginalization over LSS; it is a sensitivity to a prior that no gate in the plan constrains. If
it ships, it must be quoted as \"H0 shift under an assumed amp(z)\", with the amp(z) sensitivity
scanned, not as an LSS-marginalized posterior."

**Disposition: RESOLVED IN DESIGN (PR-8 reframed) + CONVERTED TO OWNER DECISION 7 + ACKNOWLEDGED
(R15).**

PR-8 no longer produces a posterior. `PLAN.md` §7 states it delivers a **sensitivity scan** — a
table of `H0` median/width versus assumed `amp(z > z_depth) in {0, 0.05, 0.1, 0.2, 0.4}` — quoted in
the reviewer's own words as "H0 shift under an assumed amp(z)". OWNER DECISION 7 recommends `Q == 1`
with zero variance above `z_depth` for the headline analysis. §6.1 now states the campaign's
structural limitation explicitly, in the reviewer's terms: Tiers A-C draw `xi_true` from the fitted
family so coverage is guaranteed by construction where there is no data; Tier D is the only
misspecification tier; no tier can test the extrapolation. Risk **R15** carries the resulting
under-dispersion.

---

## R1-SEV2-10 — the kernel is a 4:1 pancake and the sphere guard warns rather than fails

**Claim (verbatim).** "The kernel is strongly anisotropic — 160 Mpc transverse vs 50 Mpc radial — so
most of the M = 8505 latent modes are prior-dominated even inside the survey volume, and the
\"field-level\" framing overstates what the counts constrain."

**Argument (verbatim).** "SurveyParams defaults are lss_corr_length_mpc = 50.0 and
lss_corr_length_ang = 0.2 (chordal radians), core/types.py:147-149. The resolution guard's sphere
test is d_sph = sqrt(4 pi / M_sph) <= ls_sph (cli/build_lognormal_completion.py:309-317), which is
where the plan's M_sph >= 315 comes from: 4 pi / 0.04 = 314.2. But ls_sph = 0.2 rad = 11.5 deg, and
at the count-weighted z_ref ~ 0.24 (comoving ~950 Mpc) that is ~190 Mpc transverse — roughly 4x the
50 Mpc radial length the plan repeatedly calls \"the correlation length.\" The kernel is therefore
not the isotropic 50 Mpc object the design describes; it is a pancake, and the angular structure it
can represent is smoothed on ~11 deg. Two consequences: (i) the number of angular modes actually
constrained over the f_sky = 0.62 footprint is ~ 0.62 x 4 pi / 0.04 ~= 195, so with ~a dozen usable
radial shells the count channel constrains at most a few thousand of the M = 8505 coefficients — the
rest of Xi_m is prior noise that nonetheless multiplies dN_miss and enters -259 log mu, feeding the
member-spread problem in the first finding; (ii) the sphere side of _gp3d_resolution_guard is only a
WARNING (the docstring says so explicitly: \"The sphere side only WARNS\"), so §4.4 guard 3's claim
that reusing it \"demands M_sph >= 315\" is wrong — it demands nothing. If M_sph is cut under K4's
performance escape hatch (\"revert ... or cut M_sph\"), nothing hard-fails."

**Disposition: CONVERTED TO OWNER DECISION 3 + RESOLVED IN DESIGN (guards) + CORRECTED.**

**[verified]** the docstring at `cli/build_lognormal_completion.py:274-319` states verbatim "The
sphere side only WARNS". Rev 1's claim that reusing the guard "demands `M_sph >= 315`" is
**CORRECTED** — it demands nothing. Latent mode installs a **hard** sphere guard (guard 3) plus a
new **isotropy guard** (guard 4) that refuses a kernel more than 1.5x anisotropic — which would have
refused rev 1's own configuration.

The anisotropy is elevated from an observation to **OWNER DECISION 3** with a three-row tradeoff
table (`PLAN.md` §1.5): 50 Mpc isotropic needs `M_sph = 4470` (`M = 120,690`, `H` = 117 GB —
infeasible); rev 1's 190/50 pancake is what the reviewer describes; **the recommendation is the
isotropic ~190 Mpc kernel** — `ls_ang = 0.2`, `ls_z = 0.039`, `M_sph = 315`, `M_z = 8-12`,
`M = 2520-3780`. That configuration also raises photo-z retention from 0.20 to 0.90 (R1-SEV2-6),
shrinks `H` from 579 MB to 51-114 MB, and lifts the data-constrained coefficient fraction from ~12%
to ~40% (`0.62 x 315 = 195` angular x 5.2 radial `~ 1,015` of 2,520). The reviewer's point (i) is
adopted as the argument for the change; the surviving prior-dominated fraction feeds R3 (member
spread) and is monitored there.

---

## R1-SEV3-11 — the per-member Neff guard already exists; R7's cost estimate is wrong

**Claim (verbatim).** "The \"per-member Neff / variance guard\" is presented as new machinery, but it
already exists and already runs per member; and R7's cost estimate for raising M_draw is off by
three orders of magnitude."

**Argument (verbatim).** "§3.2 lists \"**per-member** Neff/variance guard\" as a change to
likelihood/core.py and §6.4 asserts \"Applying it only to the member average lets a collapsed member
through.\" Verified false as written: in darksirens/likelihood/core.py:1249-1262, `_member_ll` calls
`selection_log_correction(log_mu_m, Neff_m, nEvents, ..., pe_variance_sum=jnp.sum(event_vars))`
INSIDE the member vmap, with per-member log_mu_m, Neff_m and per-member pe_variance_sum, whenever
sel_has_members is True — which the plan itself forces on in latent mode (§1.5). The Farr/Essick-Farr
wall (Neff > N_obs^2/max_likelihood_variance, likelihood/selection.py:231-267) is therefore already
per member. What is genuinely missing is the member-ESS diagnostic, which should be stated as the
single new item. Separately, R7 claims raising M_draw to 16-32 \"costs 110 kB of Xi_m\". That is the
Xi_m payload only. Per §2.4 the per-proposal cost is dominated by row_fac (85 MB at M_draw=8) and by
the rho_m full-sky shell reduction (2.5e10 of the 2.9e10 FLOP total), both of which scale linearly
in M_draw: M_draw=32 is 340 MB and ~1.2e11 FLOP, i.e. 4x the entire claimed overhead. Combined with
the first finding (M ~ e^{sigma^2} draws needed), R7's mitigation is neither free nor sufficient and
should be re-costed before K4's 25% wall is quoted."

**Disposition: CORRECTED.**

**[verified]** `core.py:1250-1262`. The plan now states (`PLAN.md` §3.6, §6.4, risk-table footer)
that the per-member `Neff`/variance guard **already exists and runs inside the member vmap**;
nothing is to be built, only asserted on. The **member-ESS diagnostic is the single new item**.
Rev 1's "110 kB" is struck: §2.3 re-costs `M_draw` against the measured member-dependent seam
(+3.3 ms at `M_draw = 8`, linear) and §2.4 gives the corrected memory (11.7 MB at 8, 93.6 MB at 64,
**static**), and K4's 25% wall is withdrawn and replaced by the OWNER DECISION 5 budget.

---

## R1-SEV3-12 — stale base, unverified Schechter factorization, loose terminology, bad anchors

**Claim (verbatim).** "The plan is written against a base that is 30 commits stale, including a
substantial rewrite of the one module §1.2 declares unchanged, and several cited anchors do not
resolve."

**Argument (verbatim).** "The plan states \"Repo state assumed: branch feat/stratified-q-base @
6f8e6ae.\" That commit is an ancestor of master, which is now at 0c5b3db — 30 commits ahead, with
darksirens/redshift/selection.py +676 lines, cli/inference.py +754, inference/prior.py +321,
inference/parameters.py +199. The changes land squarely on §1.2's F2, which the plan declares
\"EXISTS. **No change.**\": master adds a Schechter LF family (_fit_schechter_truncated),
per-catalog selection fits, K>=2 homogeneous-Schechter mixtures, and bright-truncated
M_faint_offset. The plan's F2 disjointness argument was checked against _fit_gaussian_truncated
(redshift/selection.py:675-722), which is a clean conditional density p(Mhat_i | z_i, T_i) — that
part holds, and sigma(M0hat) = 1.6e-4 mag is confirmed from data/selection_fit_union.json (cov[0,0]
= 2.5456e-08). But the argument has not been made for the Schechter family, which the module
docstring calls \"the real-catalog family\" and which takes z as an argument; whether it factors as
cleanly is unverified, and §1.3's whole \"disjoint sufficient statistics\" claim rests on it. Also
note §1.3 uses \"sufficient statistics\" loosely — neither {m_i} nor {N_pg} is a sufficient
statistic for anything here; the correct statement is the chain-rule factorization p({z},{pix},{m})
= p({z}) p({pix}|{z}) p({m}|{z},{pix}), which is what the code supports. Minor anchor errors to fix
before anyone follows the plan: §0.1 cites _assemble_gp3d_survey as \"lognormal_completion.py:714-745\"
but it lives at cli/build_lognormal_completion.py:647; §9's row for the same function has the same
problem. Separately, prior.py:271-272's docstring (\"field is gated to the plain galaxy-count host
model ... no marks, no Q_LSS ensemble/table\") is itself stale relative to prior.py:374-412, which
supports field x Q_LSS given the field_lss_q inputs — worth not propagating into the new design
docs."

**Disposition: CORRECTED + RESOLVED IN DESIGN.**

**[verified]** `master` = `0c5b3db`, 30 commits ahead of `6f8e6ae`, with `redshift/selection.py`
+676, `cli/inference.py` +754, `inference/prior.py` +321, `inference/parameters.py` +199;
`_fit_gaussian_truncated` at `selection.py:675`, `_fit_schechter_truncated` at `:725`;
`_assemble_gp3d_survey` at `cli/build_lognormal_completion.py:647`.

* The plan is **rebased onto `0c5b3db`** and **PR-0** makes the rebase a blocking rung.
* **PR-2 must re-derive the F2 disjointness argument for `_fit_schechter_truncated`**; until then
  latent mode **refuses `selection_family=schechter`** (a new mutual exclusion in guard 6 and a PR-6
  gate). The rev-1 "EXISTS. No change." is qualified accordingly (`PLAN.md` §3.1).
* "Disjoint sufficient statistics" is replaced by the reviewer's chain-rule phrasing, which is what
  the code supports.
* Anchors corrected (§12), and the plan explicitly warns not to propagate the stale
  `inference/prior.py` docstring.

---

# Reviewer 2

## R2-SEV1-1 — the baseline is unsupported and contradicted by the repo

**Claim (verbatim).** "The 'baseline likelihood 3–20 s/call' figure that makes the whole overhead
argument (0.05%–0.5%) work is unsupported and is contradicted by the repo's own measurements at
exactly the production dimensions by ~2 orders of magnitude."

**Argument (verbatim).** "§2.4 attributes '3–20 s/call (report 4 measurement at 259 events)' to a
document not in the repo. The repo measures the opposite. docs/source/performance.md:104-116 gives
per-call times for 'the real spectral run (N_sel = 1,067,946, N_events = 259, n_samp = 4096, n_q =
200)' — the identical dimension set the design quotes — at 27.5 ms/call single-pass and 49.3 ms/call
under the mis-chosen block plan. darksirens/likelihood/factory.py:~100 records 'MEASURED on the H100
NVL: 7.7 ms of 30.1 ms for the spectral single pass, 13.8 of 51.3 ms at the production auto plan,
7.8 of 17.9 ms for a dark-siren mock.' Nothing in the repo shows a multi-second call. Even granting
that the dark-siren catalog path is heavier than spectral, the design needs a 60×–400× gap to be
real. This matters because it is the denominator of every cost claim: PR-6's gate is 'overhead <
10%' and K4 kills at 25%. Against a 50 ms baseline, the design's own optimistic 2–15 ms addition is
4%–30% — straddling K4 before any of the measurement errors below are corrected. The plan should not
start until the baseline is measured on the actual production closure (the harness exists:
scripts/profile_member_marginalization.py, scripts/benchmark_block_sizes.py)."

**Disposition: WITHDRAWN + RESOLVED IN DESIGN (new blocking rung).**

**[verified]** `docs/source/performance.md:104-116` gives 27.5 ms/call single-pass and 49.3 ms/call
under the mis-chosen plan at the identical dimension set. The "3-20 s/call" figure is struck from the
plan (`PLAN.md` §2.1). **PR-0** is added as a blocking rung: measure the dark-siren production
baseline at `M_draw in {1,8,32,64}` with `scripts/profile_member_marginalization.py` and
`scripts/benchmark_block_sizes.py` before any latent code lands, and commit the table to this
directory. Every percentage in the plan is explicitly marked provisional against 27.5-49.3 ms until
PR-0 reports. K4's 25% wall is withdrawn and replaced by the OWNER DECISION 5 budget.

---

## R2-SEV1-2 — the added path costs 4-7x the claim, and shell_lognorm dominates

**Claim (verbatim).** "The measured cost of the added path is 4–7× the claimed '~2–5 ms'. I
benchmarked the dominant step (§2.1 step 2, `shell_lognorm`) at the exact production dimensions on
this machine's H100 NVL: 25.6 ms in the f64 the design specifies, 13.1 ms in f32 — for one step of
five, and it does not scale down at small M_sph as the design assumes."

**Argument (verbatim).** "I ran the reduction the design specifies (n_pix=49,152; N_grid=1086;
M_draw=8; M_z=27; chunk=4096; per-shell weighted logsumexp over the full sky) as a jitted lax.scan.
Results (min of 5, device shared with the user's live job 1119811, so these are upper bounds on
speed by ~2× — a calibration 4096³ f64 GEMM ran at 17.5 TFLOP/s against ~34 TFLOP/s peak): f32 13.08
ms at M_sph=64 and 13.06 ms at M_sph=315; f64 25.58 ms at both. §3.1 types `shell_ln` as f64.
Separately I timed the prior.py:745-748 seam swap (`member_logq[pix,idx]` two-node gather →
`row_fac[pix] @ phi_z[idx]` length-27 dot) at NPE=1,060,864 and NSEL=1,067,946 with M_draw=8:
2.65→4.30 ms and 2.66→4.28 ms, i.e. 1.6× and +3.3 ms combined. Uncontended totals are therefore ~17
ms (f64) or ~11 ms (f32) versus the claimed 2–5 ms. Two structural errors in §2.4 explain it: (a)
the effective throughput is 0.95–2.28 TFLOP/s, not the ~10 TFLOP/s implied — the kernel is bound by
427M f64 exp/logsumexp elements, not by GEMM FLOPs, so a FLOP-only cost table cannot size it; (b)
§2.4's claim that at M_sph=315 'R_m and rho_m scale by 5x in their first GEMM only → ~8–15 ms' is
wrong in both directions: the M_sph-dependent GEMM is 2·n_pix·M_sph·M_z = 8.4e8 while the
M_sph-independent contraction is 2·n_pix·M_z·N_grid = 2.9e9, so M_sph barely matters — measured
13.06 vs 13.08 ms."

**Disposition: RESOLVED IN DESIGN (the dominant step is removed, not optimized).**

The measurements are adopted. Rev 2 removes `shell_lognorm` from the per-proposal path **entirely
and exactly** (`PLAN.md` §2.2). With the per-pixel completeness of §1.2, the consumption weights are
`w_p(z;theta) = (1 - f_p C(z;theta)) dN_exp(z;theta)`; `dN_exp` is `p`-independent and cancels, and
`C(z;theta)` enters only through the **scalar** `c = C(z;theta)`, so

```
rho_m(z;c,b) = log[ (A_m(z;b) - c B_m(z;b)) / (P - c F) ],
A_m = sum_p e^{b f_m(p,z)},  B_m = sum_p f_p e^{b f_m(p,z)}
```

`A_m` and `B_m` are **theta-free**: two full-sky reductions computed once offline on a `b_GW` grid
(2.4 MB at `M_draw = 64`, `n_b = 33`), with `c` in closed form online. The same decomposition
closes the field-weighting global `log Z_k` at K>=2. The 25.6 ms step is gone at K=1 and K>=2,
sampled `b_GW` or not; pin **P9** bounds the `b_GW` interpolation at 1e-6.

What remains per proposal is only the seam, which the reviewer measured: **+3.3 ms at `M_draw = 8`**
(2.65 -> 4.30 ms PE, 2.66 -> 4.28 ms selection). §2.3 tabulates it against 27.5/49.3 ms and against
`M_draw`, and notes the FLOP-only table is replaced by the measurement. The rev-1 claim that `M_sph`
drives the reduction cost is struck.

---

## R2-SEV1-3 — the memory budget is wrong by two orders of magnitude

**Claim (verbatim).** "The memory budget in §3.4 ('≈130 MB, multiplied by concurrent_evals') is
wrong by roughly two orders of magnitude: it drops the member axis, uses the wrong dtype, and —
decisively — uses replacement_chains=1 when the production sampler's concurrency is 256."

**Argument (verbatim).** "block_sizing.sampler_block_sizing_profile (:294-325) sets `concurrent =
max(1, chains, sched_max)` where sched_max is the largest entry of `replacement_chain_schedule`. The
live production run's resolved config (experiments/desi_full259/logs/ns_joint_sel_1119811.out:158)
is `\"replacement_chains\": 1, \"replacement_chain_schedule\": [1, 4, 16, 64, 256]`, and the run
banner (:246) prints 'Peak model: value-only (tinyns, 256 concurrent evals)'. The design names
`replacement_chains` — the value that is 1 — as the multiplier. Then: `row_fac (M_draw, n_rows,
M_z)` at n_rows=49,143 is 42.5 MB f32 / 85 MB f64 per evaluation, so 10.9–21.8 GB at 256 concurrent.
The chunked reduction transient is quoted as '17.8 MB' = chunk(4096)·N_grid(1086)·4 B, which omits
the M_draw=8 axis the same step is batched over and the f64 dtype: the real per-eval chunk is 142 MB
(f32) to 284 MB (f64), i.e. 36–73 GB at 256 concurrent even before row_fac. The card reports 72.7
GiB free with 10.4 GiB static already committed (:245). This is not a 130 MB line item; it is the
dominant term in the peak model and it may force the member-leaf strategy to be redesigned (e.g.
scan members instead of batching them, or accept recompute)."

**Disposition: RESOLVED IN DESIGN (the arrays become static, so the multiplier does not apply).**

**[verified]** `block_sizing.py:324` = `concurrent = max(1, chains, sched_max)`; the production log
line 158 shows `replacement_chain_schedule: [1,4,16,64,256]` with `replacement_chains: 1`, and the
banner prints "Peak model: value-only (tinyns, 256 concurrent evals)"; static state 10.390 GiB, free
72.7 GiB.

Every latent array in rev 2 is **theta-free**, so it is built in `make_likelihood` and barriered
eagerly — resident, **not** per-evaluation, and the 256x multiplier does not apply. `PLAN.md` §2.4
gives the corrected table: `row_fac` is `(M_draw, 30470, M_z)` **f32 over footprint rows only**
= 11.7 MB at `M_draw = 8`, 93.6 MB at 64; the chunked reduction transient the reviewer sizes at
142-284 MB per evaluation is **0**, because the reduction it belonged to has left the per-proposal
path (R2-SEV1-2). Total added: **~140-225 MB static, 0 transient.** The reviewer's alternative
("scan members instead of batching them") is retained as the PR-10 escape hatch if PR-0/PR-5b
measurements contradict this.

---

## R2-SEV1-4 — block_sizing is routed to the wrong function

**Claim (verbatim).** "§3.2 routes the latent memory accounting into
`block_sizing.estimate_pending_static_bytes`, which is the wrong function: it models one-time
factory allocations, while every new latent array is a per-evaluation transient scaled by
concurrent_evals. R6 therefore reproduces, rather than avoids, the ~34 GB under-reservation
precedent it cites."

**Argument (verbatim).** "estimate_pending_static_bytes (block_sizing.py:600-646) is documented as
'what the likelihood FACTORY builds' — KDE caches, base_miss, Q-ensemble device copies — and its
output enters the resolver only as `static_state_bytes`, subtracted once from the budget. The
transient working set is a separate term: `_slopes_and_fixed` (:708-760) builds `sel_bpi`/`pe_bps`
scaled by `batch_scale = float(max(1, int(concurrent_evals)))`, and resolve_block_sizes forms
`TRUE_FIXED_VALUE_BYTES + static + max(sel_batch*sel_bpi, pe_block*n_samp*pe_bps)`. The latent
leaves (`row_fac`, `shell_ln`, the shell_lognorm chunk) are built inside `body(coord, operands)` per
proposal, are proportional to neither `sel_batch` nor `pe_block`, and are held once per concurrent
proposal. Putting them in the static term under-reserves by the concurrency factor and, worse, makes
them invisible to the block-size resolver, so blocking cannot relieve the pressure. The latent
branch belongs in `_slopes_and_fixed`'s fixed term (which is concurrency-scaled) or as a new
non-blockable transient; PR-5's gate 'block_sizing reserves within 10% of measured peak in latent
mode' cannot be met by patching the static estimator alone."

**Disposition: RESOLVED IN DESIGN.**

**[verified]** `estimate_pending_static_bytes` at `:599`, `measure_static_state_bytes` at `:649`,
`_slopes_and_fixed` at `:708` with `batch_scale = float(max(1, int(concurrent_evals)))` at `:725`.

The premise of the finding — that the latent arrays are per-evaluation transients — was true of
rev 1 and is false of rev 2, because the leaves moved to `make_likelihood` (R2-SEV1-6). So the
static estimator becomes the **correct** target. But the reviewer's structural point stands as a
guard rail, so PR-5 ships **both**: the static branch in `estimate_pending_static_bytes` /
`measure_static_state_bytes`, **and** a guarded transient branch in `_slopes_and_fixed` that fires if
anyone ever enables a per-proposal latent recompute — which is **refused** until that branch is
measured (`PLAN.md` §2.4, §3.6, risk R7). PR-5's 10%-of-measured-peak gate is run at 256 concurrency.

---

## R2-SEV1-5 — the deliverable trips its own K5 and M_draw 8→32 cannot fix it

**Claim (verbatim).** "The design's own member-spread estimate implies the deliverable trips its own
kill criterion K5 at PR-6, and the proposed remedy (M_draw 8→32) cannot fix it."

**Argument (verbatim).** "§1.5 estimates the selection-side field effect at '≲ 0.01 in log mu …
against -N_obs log mu with N_obs = 259 that is up to ~2.6 nats' of member-to-member spread in ll_m,
and §6.4 gates member ESS ≥ 4 of 8 with K5 killing below 2. For ll_m with log-scale spread σ, the
ESS of softmax weights is ≈ M·exp(-σ²). Even reading 2.6 nats as a full range (σ ≈ 1.3), ESS ≈
8·exp(-1.69) ≈ 1.5 — K5 fires. Read as an sd (σ = 2.6), ESS ≈ 8·exp(-6.8) ≈ 0.01. Restoring ESS ≥ 4
needs M ≈ 8·exp(σ²) ≈ 40 (σ=1.3) to ~7000 (σ=2.6), not 16–32; and §7.2 R7's cost accounting ('costs
110 kB of Xi_m') is the wrong quantity — raising M_draw scales `row_fac`, the shell reduction, and
the per-sample dots linearly, i.e. it multiplies the SEV1 cost and memory items above. The same
spread also means `logsumexp(ll_m) - log M` is a Jensen-biased estimator of the marginal by O(σ²/2)
≈ 0.8–3.4 nats, theta-dependently — the design nowhere states this, yet §0.2 justifies the Gaussian
approximation against a '0.01-nat scale that matters'. Either §1.5's 2.6-nat number is wrong (in
which case §6.4's headline risk evaporates and should be withdrawn) or PR-6 is unshippable as
specified; the plan cannot have it both ways."

**Disposition: RESOLVED IN DESIGN + CONVERTED TO OWNER DECISION 5** (jointly with R1-SEV1-1).

The plan stops having it both ways. `PLAN.md` §6.5 states the Jensen bias explicitly
(`~ -sigma^2/2`, theta-dependent), states that rev 1's `0.01`-nat justification applies to the
Gaussianity of `p(xi|d_gal)` and **not** to the estimator bias, and makes the number a measurement
(**PR-5b**) with a gate (**P14**) rather than an argument. The key reframing both reviewers imply:
under CRN the bias level is absorbed into the evidence, so the shippable quantity is its
**theta-variation**, gated at 0.1 nat across the `H0` prior. Antithetic pairs reduce `sigma^2` for
free. Rev 1's "110 kB" cost accounting is struck and replaced by §2.3's measured, linear scaling.
K5 is restated: below the gate, the field ships as a **fixed-realization systematic**, not as a
marginalization.

---

## R2-SEV1-6 — the barrier claim does not cover the new leaves

**Claim (verbatim).** "§3.4's claim that 'the barrier stays exactly where it is' and its
anti-recompute role is preserved does not cover the new leaves: the latent producers are built
inside the jit body, where the design's own chosen seam cannot barrier them, and they recreate
exactly the pattern the barrier exists to defeat."

**Argument (verbatim).** "There are two distinct barriers and the design conflates them. (1)
factory.py's module docstring (:11-14) states 'optimization_barrier MUST be applied before arrays
enter any JIT closure (i.e. in make_likelihood, not inside likelihood()). Inside a JIT body the
arrays are already abstract tracers and the barrier has no effect.' §3.2 instructs the opposite: 'in
latent mode build the leaves inside body(coord, operands) at the _body_with_tables seam and
EMCatalog._replace them'. Every other EMCatalog leaf is barriered eagerly (factory.py:390-396,
`out[field] = barrier(arr)`); the latent ones would not be. (2) prior._materialize =
lax.optimization_barrier (prior.py:103-112) exists because 'gathering state.dN_miss[pix, idx] per
sample invites XLA to recompute the producing (N_rows × N_grid) curves inside the sample loop —
measured ~10x slowdown of the one-shot prior at 1e5 samples.' The latent seam is structurally
identical: `row_fac[pix] @ phi_z[idx]` is gathered per sample from a producer `Phi_s[rows] @ Xi_m` of
shape (M_draw, 49143, M_z), but `row_fac` is carried in `_member_leaf_bundle` (core.py:999), not in
the barriered prior state, so it sits downstream of `_materialize` with no barrier. §3.4 asserts the
barrier's role is preserved while §3.2 places the new leaves outside its scope; one of the two must
change, and the design does not say which."

**Disposition: RESOLVED IN DESIGN — §3.2 changes; the barrier stays.**

**[verified]** `factory.py:11-15` states verbatim that the barrier "MUST be applied before arrays
enter any JIT closure (i.e. in make_likelihood, not inside likelihood()). Inside a JIT body the
arrays are already abstract tracers and the barrier has no effect."; leaves are barriered eagerly at
`factory.py:239-282`; `prior._materialize` at `:103-112`.

Rev 2 resolves the conflict in the direction the reviewer's point (1) demands: **the latent leaves
are theta-free, so they are built in `make_likelihood` and `barrier()`-ed with every other
`EMCatalog` leaf** (`PLAN.md` §3.6). `row_fac` is a barriered resident constant, not a producer
inside the trace, so the recompute pattern of point (2) cannot arise — there is nothing to
recompute. `prior._materialize` is untouched and NUTS keeps its existing barrier-off downgrade.
This is the same change that resolves R2-SEV1-3 and R2-SEV1-4.

---

## R2-SEV2-7 — the "-3.6 GB freed" has the wrong sign

**Claim (verbatim).** "The '−3.6 GB resident' benefit has the wrong sign for the run the design
names as production: that run carries no LSS at all, so there is no logq_members cube or logq_map to
free — the latent path is a pure addition."

**Argument (verbatim).** "The design states this itself in the preamble ('the 259-event production
likelihood today has no LSS at all') and then books the saving anyway in §2.4 ('Net ~3.6 GB freed'),
§3.4 ('Removes 3.85 GB of resident tables') and §7.1's PR-6 row ('−3.6 GB resident'). The production
log confirms no table is loaded: experiments/desi_full259/logs/ns_joint_sel_1119811.out:232 '-
Non-LSS run. Creating memory-efficient dummy (1, 1000) grid.' and :242 'δ_g field shape: (1, 1086)
(0.000 GB)'. The 3.42 GB / 427 MB figures are the size of the on-disk q_radial.h5 (3.84 GB,
experiments/desi_full259/data/fits/) which the production sbatch does not pass. Combined with the
SEV1 above, the honest production delta is roughly +11 to +73 GB of new device transient, not −3.6
GB. Separately, the comparison in §3.4 ('the per-member leaf becomes 10.6 MB instead of a (n_rows,
N_grid) slice of a 3.42 GB cube') is not like-for-like even when a table IS loaded: core.py:988-999
and completion.py:956-966 say the existing leaf is 'a VIEW of the resident data constant, so no (M,
N_rows, N_grid) copy is made', i.e. zero incremental allocation, whereas row_fac is freshly
materialized every evaluation."

**Disposition: WITHDRAWN + CORRECTED.**

**[verified]** `ns_joint_sel_1119811.out:232` "Non-LSS run. Creating memory-efficient dummy (1,
1086) grid" and ":242" `δ_g field shape: (1, 1086) (0.000 GB)`; `lss_completions []` in the resolved
config.

The "-3.6 GB freed" claim is **struck** from the plan in all three places (`PLAN.md` §2.4). The
honest delta is stated as **+140 to +225 MB static, +0 transient** — the transient figure being 0
rather than the reviewer's +11 to +73 GB because of the R2-SEV1-2 and R2-SEV1-6 redesigns. The
not-like-for-like comparison is also corrected: the plan now notes that the existing member leaf is
a **view** of a resident constant (`core.py:988-999`), not a fresh allocation, so the compression
argument is made on the *resident* payload (1.71 GB -> 11.7 MB at `M_draw = 8`; 13.7 GB -> 93.6 MB
at 64), which is the claim that survives and is what makes large `M_draw` feasible.

---

## R2-SEV2-8 — the legacy-vs-factored jitter delta is 40x larger at the target rank

**Claim (verbatim).** "The legacy-vs-factored jitter difference is ~40× larger than stated at the
guard-compliant rank the plan targets, so OWNER DECISION 2's stated cost is wrong and PR-1's own '<
6e-5' gate fails at M_sph=315. Measured, not argued."

**Argument (verbatim).** "I reproduced the construction with the production hyperparameters (amp=1,
ls_sph=0.2, ls_z=0.0103, z_node_hi=0.30, jitter = 1e-4·amp²+1e-9, node ordering and query flattening
as in lowrank_inducing_nodes:610-632 and build_lognormal_completion.py:931-932) and compared
Phi_s⊗Phi_z against build_lowrank_operator's full-K-jitter Phi. At M_sph=64, M_z=27 (M=1728, the OOM
configuration): max|ΔPhi| = 4.96e-5 — consistent with the design's '5.4e-5'. At M_sph=315, M_z=27
(M=8505, the rank PR-4 must build and _gp3d_resolution_guard demands): max|ΔPhi| = 2.0e-3, i.e. 40×
the claimed 5e-5 and 33× over PR-1's stated gate '< 6e-5 under legacy jitter'. The mechanism is
visible in the conditioning: cond(K_full+jI) rises from 1.5e2 to 4.3e4 between the two ranks because
the Fibonacci node spacing sqrt(4π/315)=0.20 equals ls_sph, so jitter placement stops being a
perturbation. Two consequences: R11 ('low' severity, '5e-5') is mis-sized, and the PR-5 migration
pin's tolerance (1e-10 against a rebuilt table) has to be stated against a rebuild that uses the
*same* jitter convention or it is unachievable by three orders of magnitude."

**Disposition: CORRECTED + RESOLVED IN DESIGN.**

Adopted verbatim. `PLAN.md` §3.3: the legacy-vs-factored delta is **2.0e-3 at `M_sph = 315`** (4.96e-5
at 64), it is **reported as a diagnostic (P3), not gated**, and rev 1's `< 6e-5` gate is struck.
Risk **R16** is re-sized from "low, 5e-5" to "low, 2.0e-3" with the correct mechanism (jitter
placement ceasing to be a perturbation as node spacing approaches `ls_sph`). Every migration pin —
P11 and PR-5's gate — is stated against a rebuild using the **same** jitter convention, as the
reviewer requires.

---

## R2-SEV2-9 — "factored jitter" is never specified

**Claim (verbatim).** "'Factored jitter' is never actually specified — the design never gives j_sph
and j_z — and the three natural conventions differ from each other by 15× in Phi and by 1.8% in the
Nyström prior variance that §3.1 exposes as a 1e-12 CI pin."

**Argument (verbatim).** "§0.3 writes `L_sph = chol(k_sph + j_sph I)`, `L_z = chol(k_z + j_z I)` and
never defines j_sph or j_z; the legacy scalar is jitter = 1e-4·amp²+1e-9 applied to the (M,M)
product kernel, and there is no canonical split. I measured all three obvious choices at M_sph=315,
M_z=27: max|Phi_kron − Phi_legacy| = 2.08e-3 (j_s=jitter, j_z=0), 3.17e-2 (j_s=j_z=sqrt(jitter), the
convention that preserves the product), 2.00e-3 (j_s=j_z=jitter). max_v sum_i Phi[v,i]² — the
quantity `prior_var_rows` returns and which PR-1 pins to 1e-12 — moves from 0.999880 (legacy) to
0.981607 under the sqrt convention, a 1.8% change. The Kronecker identity itself is exact under
every convention (I measured max|Phi_s⊗Phi_z − chol-of-factored-K basis| = 9e-15 to 1.9e-14), so the
math of §0.3 is sound; what is missing is the convention that fixes which Q the run actually uses.
Since PR-4 ships an anchor artifact whose sha256 covers `jitter_mode`, this has to be a named,
stamped constant before PR-1, not a mode string."

**Disposition: RESOLVED IN DESIGN.**

`PLAN.md` §3.3 names the convention before PR-1, as the reviewer requires:

```
jitter_mode = "factored-v1",  j_sph = j_z = 1e-6   (absolute, amp-independent; amp == 1)
```

chosen as the smallest value keeping `cond(K + jI) < 1e8` in f64 at `M_sph = 315` (the reviewer's
measured `cond = 4.3e4` at legacy jitter makes 1e-6 comfortable). Both constants are stamped into
the artifact sha256 (guard 1) alongside `jitter_mode`, not just the mode string. **P2**
(`prior_var_rows`) is scoped to `factored-v1` only, so the 1.8% sensitivity the reviewer measured
across conventions cannot be silently absorbed.

---

## R2-SEV2-10 — PR-3's gates never test the object PR-6 ships; the multinomial Hessian is wrong

**Claim (verbatim).** "PR-3's gates never test the object PR-6 ships: the only solver pin is against
the *unconditional* Poisson objective, and the only Hessian written down anywhere (§2.2/§2.3) is the
Poisson one, which is missing the multinomial's rank-per-shell correction — so the Laplace draws
that drive the member ensemble would be systematically under-dispersed."

**Argument (verbatim).** "Decision 0.1 makes the shipped objective the conditional multinomial of
eq. (1), whose Hessian is H = I + b²·Σ_g [Φ_gᵀ diag(T_g π_g) Φ_g − T_g u_g u_gᵀ] with u_g = Φ_gᵀ π_g.
§2.2 gives only `H(theta) = I + b² Phĩᵀ diag(lam) Phĩ` and §2.3's two-stage contraction implements
only the diag part; the −Σ_g T_g u_g u_gᵀ term (G=32 rank-1 PSD subtractions, cheap at ~32·M² flops)
appears nowhere, and dropping a PSD term from H makes `laplace_draws(xi_hat, H_chol, g)` too narrow
— the exact direction that worsens the member-ESS problem in the SEV1 above. The gates do not catch
it: §5 PR-3 pins 'separable Hessian == dense on a small problem to 1e-10' (separable-Poisson vs
dense-Poisson passes while both are the wrong Hessian) and 'count_map_solve reproduces
poisson_lognormal_gp3d_map on the *unconditional* objective' — poisson_lognormal_gp3d_map
(lognormal_completion.py:662-830) is precisely the objective decision 0.1 discards. `xi_hat` and
`H_chol` from the conditional solve — the two numbers the entire deliverable rests on — have no
stated numerical reference of any kind."

**Disposition: RESOLVED IN DESIGN.**

The correct Hessian is adopted verbatim (`PLAN.md` §3.4, eq. 3), including the rank-1 subtraction,
with the observation that it too is separable and therefore free:

```
u_g = v_g (x) phi_z[g],  v_g = Phi_s^T pi_g;   T_g u_g u_g^T = T_g (v_g v_g^T) (x) (phi_z[g] phi_z[g]^T)
```

i.e. `G_s` rank-1 Kronecker outer products at `~G_s (M_sph^2 + M_z^2)` flops. The gates are replaced:
**P5** pins the separable `H` (with the rank-1 term) against the dense `H` **of the conditional
multinomial**; **P6** pins `count_map_solve` against a dense scipy Newton on the **same conditional
objective** plus `grad_inf(xi_hat) < 1e-8`. Rev 1's `poisson_lognormal_gp3d_map` reference is
struck. `xi_hat` and `H_chol` now have the numerical references the reviewer says they lacked.

---

## R2-SEV2-11 — PR-1's headline gate cannot be executed at the rank it certifies

**Claim (verbatim).** "PR-1's headline gate (`max|Phi_s ⊗ Phi_z − build_lowrank_operator(...)| <
1e-12`) cannot be executed at the configuration it certifies, because running it requires
materializing the 107 GB Phi that §0.3 exists to forbid."

**Argument (verbatim).** "build_lowrank_operator returns Phi of shape (V, M) with V = n_fit·G_s =
1,572,864. At the guard-compliant M = 8505 that is 1,572,864·8505·8 = 107 GB — the design says so
itself in §0.3 and §7.3. The gate therefore only ever runs at a toy rank. That is not fatal on its
own (the identity is structural — I verified chol(A⊗B) = chol(A)⊗chol(B) numerically to 1.9e-14 at
M=8505 on a 400-pixel query set), but the plan presents it as the pin that licenses 'Phi is never
formed anywhere', and it should be restated as a small-rank identity plus a randomized-row spot
check at production rank. Note also that the absolute 1e-12 tolerance is already 50× tighter than my
measured 1.9e-14 max over 12,800 rows; the max over the real 1.57M rows will be larger by roughly
the ratio of extreme-value tails, so the number needs to be set from a measurement, not asserted."

**Disposition: RESOLVED IN DESIGN.**

**P1** is restated exactly as the reviewer prescribes (`PLAN.md` §3.3, §6.3): a **small-rank
identity** plus a **randomized 1e5-row spot check at production rank**, against the
**chol-of-factored-K basis** (not `build_lowrank_operator`, whose dense `Phi` cannot be formed at
production rank), with the tolerance **`1e-13`, set from measurement** (1.9e-14 observed over 12,800
rows) rather than asserted at 1e-12. The plan also corrects the dense-`Phi` size at the rev-2 rank:
47.6 GB at `M = 3780`.

---

## R2-SEV2-12 — the ΔlogL kill criterion is neither discriminating nor predictive

**Claim (verbatim).** "§6.4 nominates ΔlogL(latent on − off) as 'the number that decides whether the
upgrade is scientifically live for DESI', and K2 kills below 0.05 nat — but repo data show ΔlogL
from turning Q on is ~4e4 nats while the H0 result does not move at all, so the statistic is neither
discriminating nor predictive."

**Argument (verbatim).** "experiments/desi_full259/logs/h0_scans_1119376.out runs the production
configuration (1,067,946 injections, 259 events) with and without the radial Q table. Without Q
('sel'): logL = −1,395,525.93 at H0=60; with Q ('selq_radial'): −1,433,146.23 — ΔlogL = −37,620
nats, rising to −40,571 at H0=40 and falling to −2,273 at H0=140, i.e. strongly H0-dependent. Yet
the reported H0 is identical in both arms: 'sel: H0 = 139.00 [138.3, 139.7]' and 'selq_radial: H0 =
139.00 [138.3, 139.7]' (both railing at the prior edge 140). So a 4e4-nat, strongly H0-dependent
ΔlogL produced exactly zero posterior movement. K2's 0.05-nat threshold is six orders of magnitude
below the ambient scale and will pass trivially at PR-6 while telling the owner nothing; conversely
§7.4's premise that a small ΔlogL means 'scientifically inert' is not established. The kill
criterion should be stated on the posterior (H0 median/width shift, or the K7 coverage statistic),
which the same scan harness already produces."

**Disposition: RESOLVED IN DESIGN (K2 restated) + ACKNOWLEDGED (R12).**

**[verified]** `logs/h0_scans_1119376.out`: `sel` = -1,395,525.934 and `selq_radial` = -1,433,146.229
at H0=60 (Δ = -37,620); Δ = -40,571 / -7,905 / -5,337 / -3,435 / -2,273 at H0 = 40/80/100/120/140;
both arms report `H0 = 139.00 [138.3, 139.7]`, railing at the prior edge.

Rev 1's `ΔlogL` criterion is **struck**. `PLAN.md` §9 K2 is restated at the **posterior** level: kill
if latent-on versus latent-off moves the `H0` **median by `< 0.1 sigma` and the 90% CI width by
`< 5%`**, measured on the Tier-B/C closure **and** on the production `h0_scan` — with the explicit
requirement that the production arm run at a configuration where the posterior is **not railing at
the prior edge**, or the test is vacuous (the reviewer's example is exactly that failure). Risk
**R12** records the measured numbers as the reason.

---

## R2-SEV2-13 — the dominant cost term is self-refuted; state which situation the plan is in

**Claim (verbatim).** "The cost table's dominant term is self-refuted by §4.2: under the only two
c_modes §4.4 admits, the renormalizer's weights are theta-free, so the 2.5e10-FLOP full-sky
reduction that is 86% of the claimed added cost need not be recomputed per proposal at all — and
conversely, keeping it contradicts §1.5's claim that the expensive full-sky reduction 'is invisible
at K=1'."

**Argument (verbatim).** "§4.2 defines w_p(z;theta) = f_p·(1−C_p(z;theta))·dN_exp(z;theta) and then
states 'Under a sky-uniform base the weights cancel and the renormalizer is theta-free.' That is
correct and I verified the premise: build_lognormal_completion.py:713-745 shows c_mode ∈ {aggregate,
selection} builds ONE (G_s,) curve and tiles it (`base_vox = np.tile(base_row, (n_fit, 1))`), so C_p
and dN_exp are p-independent and cancel in the per-z normalized weight, leaving w_p = f_p. The only
surviving dependence is on b_GW. Under OWNER DECISION 6's own recommendation ('fix it at b_gal in
the headline DESI run'), rho_m is a pure constant and belongs in the PR-4 artifact — at which point
the online path reduces to a precomputed member cube, which is exactly the K4 fallback and which the
repo already supports via lss_completion_logq_members. Meanwhile §1.5 argues the K=1 log_Z_global
cancellation removes 'the single most expensive new full-sky reduction (over all 49,152 pixels ×
1086 nodes, per member)' — but §2.1 step 2 keeps an identically-shaped reduction unconditionally.
The plan should state plainly which of the two situations it is in, because it determines whether
PR-5+PR-6 (9 days) buy a new capability or a memory optimization."

**Disposition: RESOLVED IN DESIGN + the framing question ANSWERED EXPLICITLY.**

**[verified]** `base_vox = np.tile(base_row, (n_fit, 1))` at `cli/build_lognormal_completion.py:744`,
inside `_assemble_gp3d_survey` at `:647`.

The reviewer is right and the plan takes the win. `PLAN.md` §2.2 shows the theta-dependence of the
renormalizer is captured **exactly** by two theta-free sky moments `A_m(z;b)`, `B_m(z;b)` (with the
§1.2 per-pixel completeness, `C(z;theta)` enters only as a scalar `c` in closed form), so the
reduction leaves the per-proposal path at K=1 **and** K>=2, sampled `b_GW` or not — removing both the
25.6 ms cost and the internal contradiction the reviewer names.

The framing question is answered in `PLAN.md` §0, plainly: **under the recommended decisions
`logQ(p,z;xi_m)` is theta-independent, so the latent field enters the GW likelihood as a compressed
member ensemble.** What PR-5+PR-6 buy is therefore *not* a new theta-coupling but three things the
status quo cannot deliver: a correct conditional posterior (footprint, per-pixel completeness,
photo-z forward model, exact budget normalizer, no firewall); a marginalization with enough draws to
be defensible — which only the 90x compression makes affordable (1.71 GB -> 11.7 MB at `M_draw = 8`,
13.7 GB -> 93.6 MB at 64); and retirement of the provenance firewall by a change of likelihood.
OWNER DECISION 8 is reframed accordingly: sampling `b_GW` now costs ~0 online, so it is a statistics
decision rather than a cost decision.

---

## R2-SEV3-14 — a third of the load-bearing citations do not resolve

**Claim (verbatim).** "Roughly a third of the load-bearing line citations do not point at the code
they name, including both cli/inference.py references and the file attribution for the
sky-uniform-base claim."

**Argument (verbatim).** "Checked against `git show 6f8e6ae:<file>`, the branch/commit the design
declares. Wrong: (a) '_check_selection_qtable_theta (cli/inference.py:686-858, ~170 lines)' — the
function is at inference.py:554; :686-858 is `_universe_model_arg` and the argparse
`--marks`/`--pop_model` block. (b) 'b_miss rule inversion (cli/inference.py:2637-2677)' and 'the
existing b_miss guard (:2637-2677)' — that range is the results-saving block (samples.npy /
results.hdf5 / settings.json / corner plot); the real per-catalog b_miss rule is at :2106-2141 and
:2257. (c) '_assemble_gp3d_survey, lognormal_completion.py:714-745, builds ONE (G_s,) curve and
tiles it' — the function is in build_lognormal_completion.py:694-780; lognormal_completion.py:714-745
is the KKT clip projection inside poisson_lognormal_gp3d_map. (d) 'n_pix_total = round(4π/apix)
(completion.py:959)' — :956 is `_row_align_lss`. (e) 'the existing field_lss_q ⊥ field_delta_g
exclusion (completion.py:1679-1685)' — the raise is at completion.py:1632; :1676-1690 is the chunked
scan body. (f) 'Production NS runs ndim = 3' (§7.3) — the named production run
sbatch_ns_joint_sel.sh samples 20 dimensions (ns_joint_sel_1119811.out:253 'Parameter space built:
20 free dimensions'); ndim=3 is the separate ns_sampled_theta run. Many citations ARE correct
(prior.py:665-673 and :711-757, core.py:989-999/:1273/:1372-1380, q_provenance.py:35-51,
block_sizing.py:540-646 and its documented ~34 GB defect, sky/models.py:257-270/:294-306,
test_lss_completion_gp3d.py:45/146/184/197/528, and the OOM log — which I confirmed byte-for-byte:
'ran out of memory trying to allocate 21743271936 bytes' at _sphere_z_kernel, = 1,572,864·1728·8, in
qbuild_gp3d_recal_1119087.err). The mixture is the problem: an implementer following the pointers
will land in the wrong function often enough that the document cannot be used as a spec without
re-verification."

**Disposition: CORRECTED.**

Every anchor in rev 2 was re-verified this session against `master @ 0c5b3db` and is tabulated in
`PLAN.md` §12. Notes on the individual items:

* (a), (b) — **[verified]** on `master`: `_check_selection_qtable_theta` is at `inference.py:686` and
  the `b_miss` rule at `:2638-2673`. Rev 1's citations were **right for `master` and wrong for its
  own declared base**, which is itself the R1-SEV3-12 defect. Rebasing (PR-0) resolves both.
* (c) — **[verified]** correct as reported: `_assemble_gp3d_survey` is at
  `cli/build_lognormal_completion.py:647`, with the tile at `:744`. Corrected in §12 with an explicit
  "rev 1 mis-attributed this" note.
* (d) — **[verified]** `n_pix_total` is set at `completion.py:510`; the global-table guard is at
  `:659-699`. Corrected.
* (e) — **[verified]** the `field_lss_q` / `field_delta_g` raise is at `completion.py:1698` on
  `master`. Corrected.
* (f) — **[verified]** `ns_joint_sel_1119811.out` reports "Parameter space built: 20 free
  dimensions". Corrected in `PLAN.md` §10, with the note that `ndim = 3` is the separate
  `ns_sampled_theta` run.

Risk **R18** records the class of defect.

---

## R2-SEV3-15 — PR-5's migration pin depends on an artifact that does not exist

**Claim (verbatim).** "PR-5's migration pin and R11 are written against a DESI-scale `q_gp3d.h5`
that does not exist and has never been built, making the seam's only correctness gate depend on the
plan's riskiest step succeeding first."

**Argument (verbatim).** "The design says the migration pin should reference 'a freshly-rebuilt
logq_map … not the shipped q_gp3d.h5' and R11 calls it 'Legacy q_gp3d.h5 differs at the 5e-5 level'.
There is no q_gp3d.h5 at DESI scale: `find` returns only
experiments/completeness_viz/{output,output_agg,output_pp_s0,magpure_aggregate,magpure_selection}/fits/q_gp3d.h5
— nside=16 mock artifacts. experiments/desi_ingest/data/fits/ and experiments/desi_full259/data/fits/
contain only q_radial.h5 / q_radial_strat.h5, and the DESI gp3d build OOM'd
(qbuild_gp3d_recal_1119087). So PR-5's gate ('latent logQ at (theta_fid, xi_hat) equals a
freshly-rebuilt logq_map node-for-node to 1e-10') is blocked on PR-4 producing the first-ever
guard-compliant gp3d artifact — which the design itself flags as new, unproven work. The ladder
should either add a small-nside gp3d rebuild as the PR-5 reference (the completeness_viz artifacts
already provide one) or move the migration pin behind PR-4's success."

**Disposition: RESOLVED IN DESIGN (both of the reviewer's remedies, applied together).**

`PLAN.md` §7, PR-5: **P11 runs at nside 16 against the `experiments/completeness_viz` gp3d rebuild**
(the reviewer's first remedy), and the DESI-scale migration pin **moves behind PR-4** and is
**reported, not blocking** (the reviewer's second). Risk R16 is restated against the correct
artifact and the correct delta (2.0e-3 at `M_sph = 315`, per R2-SEV2-8).

---

## R2-SEV3-16 — supporting numbers in §1.5, §2.3 and §7.1 are internally inconsistent

**Claim (verbatim).** "Several supporting numbers in §1.5, §2.3 and §7.1 are internally
inconsistent: the injection-granularity argument uses the M_sph the guard forbids, the 'dense equiv'
FLOP column is off by up to 52× and does not scale with its own memory column, and §7.1's 24 h GPU
build contradicts §2.3's 20–30 ms per iteration by five orders of magnitude."

**Argument (verbatim).** "(a) §1.5: '1,067,946 detected injections over 4π/M_sph = 0.196 sr node
cells gives ~16,700 injections per angular mode, i.e. 0.8% MC error'. 4π/M_sph = 0.196 implies M_sph
= 64, but §4.4 guard 3 and PR-4 mandate M_sph = 315, giving 1,067,946/315 = 3,390 per mode and 1.7%
MC error — the argument is made at a rank the plan forbids. (The injection count itself is right: I
confirmed '1,067,946 detected injections' in ns_joint_sel_1119811.out:224.) (b) §2.3's 'dense equiv'
column: for Phi^T diag(lam) Phi the cost is 2·V·M² = 9.39e12 at M=1728 (claimed 2.9 TFLOP), 2.28e14
at M=8505 (claimed 4.4 TFLOP, 52× off) and 2.36e15 at M=27405 (claimed 45.8 TFLOP); the memory
column is right for rows 1–2 (21.7 GB, 107.0 GB) but row 3 is 344.8 GB, not 214 GB. The FLOP column
grows 1.5× while M grows 4.9×, which is impossible for any dense cost. The stage-1/T-mem/H-mem
columns all check out exactly, so it is only this column. (c) §2.3 gives '~20–30 ms/iteration on an
H100 fp64 plus ~15 ms for the M³/3 Cholesky' at M=8505 (I get stage-1 = 3.12e11 FLOP and chol =
2.05e11 FLOP, consistent with tens of ms), yet §7.1 budgets 'one 24 h GPU build at M=8505'. At the
measured cold trip count of 13 Newton iterations that is ~0.5 s of linear algebra. Either the 24 h
covers unmodelled work (the numpy radial builder, the process pool, HDF5 writes) — in which case say
so — or the iteration cost is wrong."

**Disposition: CORRECTED.**

(a) The injection-granularity argument is **removed** rather than rescaled: under rev 2 the
`M_sph = 315` figure gives 3,390 injections per angular mode and 1.7% MC error, which is the number
now used wherever granularity is discussed, and the `M_sph = 64` version is struck.

(b) The "dense equiv" FLOP column is **deleted** from `PLAN.md` §3.4. The stage-1/`T`-mem/`H`-mem
columns the reviewer verified are retained; the surviving dense-`Phi` figures are memory only, and
are restated at the rev-2 rank (47.6 GB at `M = 3780`, alongside the recorded 21.7 GB OOM at
`M = 1728` and 107 GB at `M = 8505`).

(c) The "24 h GPU build" is **withdrawn**. At the rev-2 rank (`M = 3780`) stage 1 is 3.1e11 flop,
the Cholesky 1.8e10 flop, and 13 iterations is ~1 s of linear algebra; `PLAN.md` §3.4 and PR-4's
gate state the anchor build takes **minutes**, dominated by I/O and the count assembly.

---

# Summary of dispositions

| finding | severity | disposition |
|---|---|---|
| R1-SEV1-1 non-uniform member weights | SEV1 | WITHDRAWN + RESOLVED (§6.5, P14, PR-5b) + OWNER DECISION 5 |
| R1-SEV1-2 per-pixel incompleteness, wrong sign | SEV1 | RESOLVED (§1.2) + OWNER DECISION 4 + risk R2 + kill K8 |
| R1-SEV1-3 budget identity mismatch | SEV1 | RESOLVED (§4.2, P13) + risk R15 |
| R1-SEV2-4 shell-collapse approximation, tautological pin | SEV2 | RESOLVED (frozen `W`, P7) + risk R11 |
| R1-SEV2-5 `b_gal` sampled vs fixed | SEV2 | RESOLVED (§3.4 rank-1 inflation) |
| R1-SEV2-6 photo-z | SEV2 | RESOLVED (forward convolution in `W`, P8); rev-1 OD4 retired |
| R1-SEV2-7 `zmax` split | SEV2 | RESOLVED (§5.2, guard 7) + risk R13 |
| R1-SEV2-8 budget plug-in uncertainty | SEV2 | OWNER DECISION 6 + risk R8 |
| R1-SEV2-9 above `z_depth` is 100% prior | SEV2 | RESOLVED (PR-8 reframed) + OWNER DECISION 7 + risk R15 |
| R1-SEV2-10 anisotropic kernel; sphere guard only warns | SEV2 | OWNER DECISION 3 + guards 3/4 + CORRECTED |
| R1-SEV3-11 per-member `Neff` guard already exists | SEV3 | CORRECTED |
| R1-SEV3-12 stale base, Schechter F2, anchors | SEV3 | CORRECTED + PR-0 rebase + schechter refused until PR-2 |
| R2-SEV1-1 baseline unsupported | SEV1 | WITHDRAWN + PR-0 blocking measurement |
| R2-SEV1-2 added cost 4-7x | SEV1 | RESOLVED (§2.2 moment decomposition removes the step) |
| R2-SEV1-3 memory off by 2 orders | SEV1 | RESOLVED (§2.4, leaves are static) |
| R2-SEV1-4 `block_sizing` mis-routed | SEV1 | RESOLVED (static + guarded transient branch) + risk R7 |
| R2-SEV1-5 K5 fires; `M_draw` 8→32 insufficient | SEV1 | RESOLVED (§6.5) + OWNER DECISION 5 |
| R2-SEV1-6 barrier conflict | SEV1 | RESOLVED (leaves built in `make_likelihood`) |
| R2-SEV2-7 "-3.6 GB" wrong sign | SEV2 | WITHDRAWN + CORRECTED |
| R2-SEV2-8 jitter delta 40x | SEV2 | CORRECTED (P3 reported, not gated) + risk R16 |
| R2-SEV2-9 jitter never specified | SEV2 | RESOLVED (`factored-v1`, `j = 1e-6`, stamped) |
| R2-SEV2-10 wrong Hessian, wrong reference | SEV2 | RESOLVED (rank-1 term; P5/P6) |
| R2-SEV2-11 PR-1 gate unexecutable | SEV2 | RESOLVED (P1 restated, tolerance from measurement) |
| R2-SEV2-12 `ΔlogL` criterion not discriminating | SEV2 | RESOLVED (K2 at posterior level) + risk R12 |
| R2-SEV2-13 dominant cost self-refuted; framing | SEV2 | RESOLVED (§2.2) + framing answered in §0 |
| R2-SEV3-14 one third of citations wrong | SEV3 | CORRECTED (§12 re-verified) + risk R18 |
| R2-SEV3-15 migration pin has no artifact | SEV3 | RESOLVED (nside-16 reference; DESI pin behind PR-4) |
| R2-SEV3-16 inconsistent supporting numbers | SEV3 | CORRECTED (granularity, FLOP column deleted, 24 h withdrawn) |

**Net effect on the ladder:** two new rungs (PR-0 baseline+rebase, PR-5b member-spread measurement),
PR-2 and PR-3 widened, PR-8 reframed from a posterior to a sensitivity scan, critical path 22 d ->
~27 d, and ten OWNER DECISIONs (up from eight, with three of the originals rewritten).

---
---

# v2 (owner context) — what the owner's external analysis changed, and why

**Date:** 2026-08-10. **Input:** `OWNER_CONTEXT.md` (the owner's novelty scoring of the `Q` machinery
against Dalang & Baker variance completion, Leyde/Baker/Enzi *Cosmic Cartography*, and Cheng & Gair
harmonic cross-correlation, plus the promoted goal and the session directives). **Verification base:**
`master` @ `0c5b3db`, re-checked this session.

**Nothing in this file is invalidated.** No finding is withdrawn, no disposition is reversed. What
changed is that **eight findings change role**, **four rev-2 answers are corrected**, and **eleven
facts verified this session are new** — none of which either reviewer had, because they were reviewing
a document that was answering a different question.

Two disposition codes are added for this section:

* **ROLE CHANGED** — the finding and its resolution both stand, but under the promoted goal the
  finding means something different (usually: a *defence* becomes a *specification*, or a *residual*
  becomes a *gate*).
* **CORRECTED IN v2** — rev 2's own answer was wrong or too strong, and v2 fixes it. These are
  self-corrections, not reviewer findings, and they are listed so the next reviewer does not have to
  find them again.

---

## v2-0 — The promotion, stated once

Rev 2 asked *what is the cheapest correct way to get a defensible LSS marginalization into the
259-event likelihood?* and answered honestly: a **theta-independent compressed member ensemble**
(condition on shell totals, freeze `W`, and `p(xi|d_gal)` loses all theta dependence by
construction). The owner asks *what makes this a methods paper that Cosmic Cartography, variance
completion and cross-correlation cannot subsume?*, and that answer requires `p(xi | theta, d_gal)`.

So rev 2's PR-9 escape hatch is the owner's headline, and rev 2's recommended OWNER DECISION 1 is
the thing that deletes it. **This is a promotion to execute, not a disagreement to resolve.** v2
executes it: the theta-coupled latent becomes the deliverable (PR-6b), the compressed ensemble is
re-scoped as the intermediate rung, the control arm and the fallback (PR-6a), PR-9 dissolves into
PR-6c with its two hardest items deleted, and the critical path *shrinks* relative to rev 2's own
optional path (PR-6b costs ~5 d; PR-9 cost 8 d).

---

## v2-1 — Findings whose ROLE changed

### R1-SEV2-4 (shell-collapse approximation) — **ROLE CHANGED: defence → specification**

Rev 2 adopted the review's algebra (`H0` cancels from the count channel because
`dV/dz = (c/H0)^3 x shape(z;Om0)` and `C_sel` is `H0`-free; `Om0/w0/wa/delta` do **not** cancel) as a
*defence*: freezing `W` closes the leak. Under promotion the same algebra is the **specification of
which parameters the galaxy field is allowed to couple to**. It says, exactly: rung 1 can couple
`Om0, w0, wa, delta, theta_sel`, and **cannot couple `H0` at all**. Carried in `PLAN.md` §1.1 item 3
and §0.3's rung table.

### R1-SEV2-4's pin, P7 — **ROLE CHANGED: measured residual → the feasibility gate for the whole direction**

Rev 2 reported P7 at PR-3 as a bound on a residual it was ignoring. In v2, P7 **is** the decision:
`tau < 0.02` means rung 0 *is* rung 1 and promotion adds nothing; `0.02–0.3` means linear response is
the design; `> 0.3` means refuse promotion (new kill criterion **K9**). **The owner's headline is
therefore decidable on day ~13 rather than day ~33** — the highest-value scheduling change in v2.

### R2-SEV1-3 and R2-SEV1-4 (memory off by two orders; `block_sizing` mis-routed) — **ROLE CHANGED: resolved → conditionally resolved**

Rev 2 resolved both by making every latent array theta-free and factory-static. That resolution
survives **only under linear response**. Under any per-proposal re-solve variant, `row_fac` becomes a
per-evaluation transient — **3.0 GB at `M_draw=8` and 24 GB at `M_draw=64` at 256 concurrency**,
against 72.7 GiB free with 10.4 GiB already static — and the guarded transient branch in
`_slopes_and_fixed` becomes the *primary* accounting path rather than a guard rail. Both findings are
therefore **reinstated as live risks for any design other than the one v2 recommends**, and that
conditionality is now written into risk R7, OWNER DECISION 5 and `PLAN.md` §2.5.

### R2-SEV1-2 (added cost 4–7x; `shell_lognorm` dominates) — **ROLE CHANGED: resolved → resolved by a mechanism that must be built**

Rev 2 removed the 25.6 ms f64 / 13.1 ms f32 full-sky reduction with the theta-free `(A_m, B_m)`
moment decomposition. Under promotion the moments are no longer theta-free, so the finding returns in
full (+52–93% on a 27.5–49.3 ms baseline) **unless** the projected derivatives
`∂A/∂theta, ∂B/∂theta` are built and shipped (~1.1 MB at `M_draw=8`). That projection is now a
first-class PR-4 deliverable with its own pin (**P19**), not an implementation detail.

### R2-SEV1-5 / R1-SEV1-1 (`M_draw`, member-estimator bias) — **ROLE CHANGED: measured → predicted, then measured**

Unchanged in substance. Two upgrades: the requirement is now stated sharply as
`M > (e^{sigma^2} - 1) / (2 epsilon)` rather than `M ~ exp(sigma^2)` (at `sigma = 2.6`: `M > 4.3e3`
for 0.1 nat, and `M_draw = 8` carries **54 nats**), and PR-5b is preceded by a ~1 h closed-form
**prediction** of `sigma`, so the measurement can confirm or refute a number rather than merely
produce one. A predicted-vs-measured discrepancy is itself diagnostic: it means `b_GW f` is not small,
which is the regime where the numerical marginalization is doing real work.

### R2-SEV2-12 / R12 (the `ΔlogL` criterion is not discriminating) — **ROLE CHANGED: risk → design property**

Rev 2 restated K2 at the posterior level. v2 adds the mechanism: under common random numbers the
estimator is a **deterministic smooth function of `theta`**, so repeat-determinism and adjacent-theta
smoothness pass *by construction* and pass just as well on a badly distorted surrogate. The
determinism sweep is now labelled a regression guard, never evidence of correctness, and only P14's
theta-*varying* bias discriminates. The same property makes **K6 unfireable by construction** under
linear response — so K6 firing at PR-6b means the implementation is not the design.

### R1-SEV2-9 / R15 (above `z_depth` is 100% prior; `Q ≡ 1` with zero variance) — **ROLE CHANGED: risk → scope sentence in the abstract**

The convention covers **38% of the sky and 99.99% of the missing budget**. Under rev 2 this was a
bounded systematic in a risk table. Under the owner's framing it is a **scope limit on differentiator
1** and must appear in the same paragraph as any claim to "marginalize over the clustered missing
field": the claim is true over the volume the survey sees and false over the volume that carries the
budget. Now a PR-11 gate.

---

## v2-2 — Rev 2's own answers, CORRECTED IN v2

### C1 — "no fixed-theta provenance firewall" was false at rung 0

Rev 2 §0 listed this as a deliverable. Under shell-total conditioning with a frozen `W`, guard 1
stamps `anchor theta_ref, b_gal` into the artifact sha256: **the firewall is replaced by a
fingerprint, not abolished.** The owner's line — "No `Q` built at fixed `n0`, fixed cosmology, or
`theta_hat_selection`" — becomes literally true only at rung 1. Corrected in `PLAN.md` §0.0 and §1.1
item 2.

### C2 — P7's tolerance was wrong by `sqrt(M)` for one of its two roles

Rev 2 wrote P7 as `max_theta ||xi_hat(theta) - xi_hat(theta_ref)||_H / sqrt(M) < 0.1` posterior sd.
That is defensible as a **model-misspecification** tolerance. It is catastrophic as an **estimator
validity** tolerance: for a fixed draw set, `Var[log w] = ||Δxi_hat||_H^2` *exactly* (Gaussian mode
shift at fixed `H`), so `ESS/M ≈ exp(-||Δ||_H^2)`; a per-mode `0.1` gives `||Δ||_H^2 = 0.01 M = 37.8`
at `M = 3780` and hence **`ESS/M = 4e-17` from a pin that reads as comfortably tight.** The
admissible fixed-draw threshold is `||Δ||_H^2 < log 10 ≈ 2.3`, i.e. `tau < 0.025` at `M = 3780`.

**The deeper fix, which neither reviewer nor the owner context states:** at fixed `H`, a *mean shift*
needs no importance weights at all. `xi_m(theta) = xi_hat(theta) + L_H^{-T} g_m` with `g_m` fixed is
an **exact draw** from `N(xi_hat(theta), H^{-1})`. v2 therefore ships **design 1a (mean-shifted
draws)** and forbids **1b (fixed draws + weights)**; §6.3 states 1b's threshold explicitly so the
trap is visible rather than latent, and new risk **R22** tracks the failure mode (an implementer
passes P7, passes every determinism diagnostic, and produces nonsense).

### C3 — the blanket prohibition on warm starts was too strong

Rev 2 §10 listed "warm-starting the solve across proposals — breaks nested sampling irreparably".
Since `H ⪰ I` under the whitened prior, stopping at `||grad J||_2 = eps` gives
`0 <= J(xi_stop) - J(xi_hat) <= eps^2/2`, so **any** start — warm included — yields a likelihood
history-independent to `eps^2/2` nats (`eps < 0.14` buys 0.01 nat). What is genuinely inadmissible is
a **fixed iteration count from a history-dependent point**, which leaves the residual unbounded and
makes `logL` a functional of the sampler's traversal order, at which point the NS shrinkage argument
`E[log X_i] = -i/n` no longer refers to a well-defined likelihood. **The rule is a gradient-norm
stopping criterion, not a prohibition.** Moot at rungs 0/1 (no inner solve); binding at PR-6c.

### C4 — R1-SEV1-3 is RE-DISPOSITIONED: real in `per_pixel`, vacuous in the production configuration

R1-SEV1-3's argument stands as written *against rev 1's proposal* (an `f_p`-weighted normalizer
against `f_p`-free consumption), and its own closing sentence already noticed the direction of the
problem. What neither rev 1 nor rev 2 established is the status of the **shipped** code, and it
changes what may be claimed:

**[verified]** under `c_mode in {aggregate, selection}` the gp3d builder fits the **whole sky**
(`build_lognormal_completion.py:714`, `fit = np.arange(n_pix)`) with **p-independent** weights
(`:744`, `w_budget = np.tile((1 - Cbar_fine) * dN_exp_density, (n_fit, 1))`), so the mean-one
condition reduces to `<Q>_sky = 1` and `sum_p (1-C) dN_exp Q_p` equals its `Q = 1` value **exactly**,
for every `z`, member and `theta`. Under stratified selection the weights are routed by the same
stratum map the likelihood consumes (`:730-732`). Under radial `per_pixel` unfitted rows ship
`logQ = 0` exactly (`:570-590`). The **only** genuine leak is the gp3d `per_pixel` borrowing halo,
which the builder's own docstring concedes (`:975-999`) — and `per_pixel` is refused in latent mode.

**Consequence for the writeup:** eq. (4) + P13 is **not a repair of a shipped budget defect**. It is
the obligation this plan creates for itself by putting `f_p` inside the consumption (§1.2), which
makes the weights p-dependent for the first time. Claiming to have fixed a shipped defect would not
survive inspection. `PLAN.md` §0.2 and §4.2 now say so.

### C5 — pin P15 and the `custom_vjp` requirement are DELETED

**[verified]** the production sampler does not differentiate the likelihood:
`likelihood/block_sizing.py:294-302` — "*Only NumPyro NUTS does; dynesty and tinyns are
gradient-free*" — and the production banner is `Peak model: value-only (tinyns, 256 concurrent
evals)`. Implicit-diff `custom_vjp` and its pin were the single riskiest engineering item in rev 2's
document, and they were required by a capability nothing in production uses. Both are removed and
re-added only behind an explicit NUTS path.

---

## v2-3 — The owner context's two staleness claims, adjudicated

Both were flagged in the task as "verify, do not trust either way". Verified on `0c5b3db`:

### Claim 1 — *"the inference forbade `c_mode=selection` for K>=2"* — **FALSE (stale)**

K>=2 × `c_mode=selection` assembles and evaluates end to end: `_selection_c_mode_by_catalog`
(`cli/inference.py:2365`) resolves `c_mode` per catalog; `_resolve_selection_fits` (`:2378-2578`)
takes one `--selection_fit` per catalog with an all-or-nothing anchoring rule, one luminosity-function
family (`:2454-2465`) and one shared Schechter `M_faint_offset` (`:2522-2538`);
`_check_selection_qtable_theta` (`:686-860`) is explicitly **per catalog**; `inference/prior.py:676-685`
already accepts a per-catalog `c_mode` sequence; and `tests/test_multitracer_selection.py` is a K=2 ×
selection end-to-end likelihood fixture with **10 tests**.

**The correct narrow statement: inference forbids *stratified* selection fits at K>=2, not
`c_mode=selection` at K>=2** (`inference/parameters.py:460-470` and its pre-load twin
`cli/inference.py:1492-1502`), and the stated reason is data plumbing — the full-sky stratum map hangs
off the shared bundle while a mixture builds per-catalog views — not modelling.

### Claim 2 — *"the joint multi-survey Q builder has not caught up with selection C"* — **TRUE, and understated by three defects**

**[verified]** grep counts in `cli/build_joint_lognormal_completion.py`: `c_mode` **0**,
`selection_fit` **0**, `stratum` **0**, `renormalize_q_mean_one` **0**, `budget_renorm` **0**;
`realization_set_id` **13**. Against `cli/build_lognormal_completion.py`: `c_mode` 46,
`selection_fit` 45, `budget_renorm` 15, and `realization_set_id` **0**.

1. **No `c_mode`.** `:180-181` calls `_assemble_gp3d_survey(...)` with no `c_mode`, taking the
   `per_pixel` default (`build_lognormal_completion.py:648`), and saves without `c_mode=`
   (`:206-209`, `:359-363`) so the file reads as legacy `per_pixel` and is **hard-rejected at load**
   by any `aggregate`/`selection` run (`catalogs/lss.py:33-48`).
2. **No budget gauge fixing.** `renormalize_q_mean_one` is never called, so every joint file ships the
   full Jensen monopole — the measured **+55%** inflation the K=1 tables remove. The joint parity
   test concedes it by comparing against a `budget_renorm=False` single build
   (`tests/test_joint_lognormal_completion.py:107-112`). **Neither reviewer had this.**
3. **The rank cannot resolve any physical scale.** `M_SPH, M_Z = 32, 6` up to `Z_NODE_HI = 3.0`
   (`:73-74`) gives a zeta node spacing of `log(4)/5 = 0.2773`; the **hard** radial guard then demands
   `ls_z >= 0.2773`, i.e. **`L_smooth >= 1.34 Gpc`** at `z_ref = 0.237`. Every physically supportable
   length (50–190 Mpc, `ls_z = 0.010–0.039`) hard-fails, and the builder's own re-anchored message
   says the only remedy is raising `--lss-corr-length-mpc` (`:222-237`). **Neither reviewer had this.**
4. **The binding constraint the owner context misses.** The *single-catalog* builder **does** support
   `--c-mode selection` (`:1148`) and `--selection-fit` (`:1160`), but has **no
   `--realization-set-id`**, so `save_lss_completion_hdf5` mints a fresh `uuid4().hex` per build
   (`redshift/lognormal_completion.py:1055-1056`). **The joint builder is the only producer of a
   shared `realization_set_id`, and it is `per_pixel`-only.**

**The precise seam statement, which is the single most valuable thing this update can say:** *the
consumer is ready; the producer cannot produce; and the one escape hatch that lets the configuration
run is exactly the physically-wrong one.* There is no route at any K — joint, or K-times-single — to
matched-member `c_mode=selection` `Q` ensembles; the only way to run it is
`--allow_unverified_shared_lss_members`, which the code itself says marginalizes over an
**independent-fields product prior** rather than the shared-field prior the estimator assumes
(`inference/loaders.py:352-395`). **A K>=2 shared-latent LSS marginalization under selection `C` is
not refused — it is not constructible.**

**Disposition:** do **not** build the joint selection builder. PR-7 closes the seam by making the
producer unnecessary. One 1-day, mock-only, explicitly-terminal exception (PR-7i) produces OWNER
DECISION 10's K>=2 selection *table* baseline, which cannot be built today; it inherits defects 2 and
3 and is not a DESI baseline.

---

## v2-4 — Facts verified this session that neither reviewer had

| # | fact | why it matters | anchor |
|---|---|---|---|
| V1 | The `Q` provenance firewall **already forbids** sampling `log10n0, delta, b_miss, Om0, w0, wa` whenever a prebuilt table is active; `H0` alone is exempt behind a warning that says the right fix is to interpolate `Q` over `H0`. | This is the sharpest **code-level** argument for the promotion: the owner's sentence is today forbidden by construction for every cosmological parameter except the one it is least valid for. | `inference/q_provenance.py:35-51`, `:19-26`, `:196-242` |
| V2 | An in-likelihood, **sampled**, whitened low-rank sphere × z GP already exists on the GW side — `M = 192` latents plus three sampled hyperparameters — and the offline builder deliberately reuses the same kernel and inducing geometry. | The field-level upgrade is a **merge of two existing pieces**, not a new sampler. Strongest feasibility argument available, and it belongs in the proposal. | `sky/models.py:273-379`, `:311-332`, `:395-422`; `redshift/lognormal_completion.py:583-632` |
| V3 | The `Q` channel's own GP hyperparameters are **structurally unsampleable**: they live on `SurveyParams` but are absent from `SURVEY_PARAMS_FID_BY_NAME`, the by-name registry the decoder and prior address. | Scopes the plan: latent mode keeps them frozen basis constants; sampling them is a registry change, not a flag. | `core/types.py:147-149` vs `core/constants.py:20-45` |
| V4 | `b_miss` is the tracer-bias seam — a `Q`-active catalog *drops* it because `Q` replaces the local overdensity factor. | §4.3's `b_miss -> b_GW` inversion is a **restoration**, not an invention; it is the same quantity the joint builder applies offline as `Phi'_k = b_k Phi_k`. | `inference/loaders.py:229-247`; `cli/inference.py:2637-2676`; `q_provenance.py:47-51` |
| V5 | Only the **gp3d** family can move in-likelihood: it is a convex GLM with an `M x M` Newton Hessian; the radial builder is a per-pixel L-BFGS-B with a 200,000-iteration cap. | The production DESI table is *radial*, so adopting the latent path is a change of `Q` family **by necessity**, not preference. Strengthens OWNER DECISION 10. | `lognormal_completion.py:660-864` vs `:246-415` |
| V6 | The `n0` calibration integrates `dV_c/dz` at the **fiducial cosmology**. | Any `H0` recovered by un-conditioning the count likelihood is exactly degenerate with `n0` and partly manufactured by the pin — the same class of defect as the `ls_z`-in-Mpc standard ruler guard 2 exists to forbid, but larger (22.79M galaxies). Promotes OWNER DECISION 6 from advisory to load-bearing; new risk **R20**, new kill **K10**. | `experiments/desi_ingest/calibrate_n0.py:86`; `data/n0_calibration.json` (`V_c_Mpc3 = 7.818e9`, `log10n0 = -2.3996`) |
| V7 | A **full per-proposal re-solve is infeasible by the plan's own arithmetic**: ~13 Newton iterations at tens of ms ≈ 1 s against a measured 27.5–49.3 ms baseline — a ~36x wall. | The owner context's "solve/update the conditional posterior at each proposal" is not viable read literally; its own "or importance-sample a small fixed set of whitened latent draws" clause is the viable reading, and linear response is what it becomes. | `PLAN.md` §3.4; `docs/source/performance.md:104-116` |
| V8 | The plan **already implements** the linear-response machinery for one parameter, under the name `bias_sensitivity` (`d xi_hat/d b = -H^{-1}(d grad/d b)`, from the implicit function theorem). | Generalizing `b -> theta` gives `S = d xi_hat/d theta` in `n_theta` extra triangular solves against the same `H_chol`, storage 121 kB. **The cheapest, highest-leverage edit in the document, and it lands at a rung that already exists.** | `PLAN.md` §3.4 → §1.7 |
| V9 | The two source documents **collide on the word "conditional"**: `PLAN.md` means *conditioned on shell totals* (which removes `theta`); `OWNER_CONTEXT.md:182-183` means *conditioned on the data at a given theta* (which restores it). | An implementer who conflates them builds the wrong object. `PLAN.md` now says "shell-total–conditioned" throughout and `conditional_multinomial_logl` is renamed `shell_multinomial_logl`. | `PLAN.md` §1.1 terminology box |
| V10 | The owner context's feasibility estimate — *"with only `M ~ O(10^2)` whitened coefficients this is actually plausible"* — is made at a rank the hard resolution and isotropy guards **forbid**. The guard-compliant rank is `M = 2520–3780`. | Feasibility at the real rank comes from the correction being **member-independent and rank-`n_theta`**, not from a small `M`. Any argument reaching for a smaller `M` is reaching for the prior-collapse regime (measured fitted-vs-truth slope 0.04). | `OWNER_CONTEXT.md:186` vs `PLAN.md` §1.5, §4.4 guards 3–4 |
| V11 | The owner's stated deliverable — a LaTeX/PDF proposal — **had no rung** in rev 2's PR-0..PR-10 ladder. | Added as **PR-11**, with `MODEL.tex` as its mathematical body and the exhaustive novelty search as a gate on it. | `OWNER_CONTEXT.md:202-204`; `PLAN.md` §7 |

---

## v2-5 — New material added to the plan, and why each is there

| addition | driver |
|---|---|
| §0.2 **Novelty**, calibrated to the owner's scoring table; the forbidden sentence written out; two differentiators with the literature each must be searched against | `OWNER_CONTEXT.md:105-148`; a paper cannot be planned around a claim the owner has already scored as *not novel broadly* |
| §0.3 **Three rungs + the criterion**, with the 1a/1b sub-variant distinction | the promotion needs a *decidable* structure, not a preference |
| §0.4 **The seam**, verified on both halves | the task's highest-value precision; also the cleanest one-line motivation for the promotion |
| §1.6 **The three limits** (`C -> 1` discrete hosts; the interior; `C -> 0` giving `(b^2/2) sum_ij K(x_i,x_j)` and the harmonic `sum_l (2l+1)/4pi W^i_l W^g_l C_l^{GWxg}`) | `OWNER_CONTEXT.md:163-176`; rev 2 contained nothing on the unification, and it buys two new pins (**P16**, **P17**) |
| §1.7 **Linear response** | the only affordable form of the promotion |
| §1.8 **The `n0`–`H0` degeneracy** | the consequence rev 2 stated a premise for and never drew; it is the central scientific caveat |
| §1.9 **What this is a merge of** | V2–V5; reframes the effort and feeds PR-11's "what we do now" |
| §2.5 **Cost under promotion** (survives/dies table) | so no reader concludes theta-coupling is affordable in general |
| §3.1 **Double conditioning** (`p(U \| C, xi) = p(U \| xi)` by thinning), with the two named failure modes | the structural objection a referee will raise; also names the most dangerous half-measure (counts without promoting `xi`) and the monopole-squared trap |
| §6.4 **Channel split** (A: galaxy-side evidence; B: GW-side member shift), mandatory at rung 1 | new risk **R21** — a `dN/dz`-shape constraint from 22.79M galaxies could dominate 259 events and be mis-read as a dark-siren measurement |
| **P16, P17, P18, P19, P7b** | Limits I and III; the control-arm inertness pin; the projected-moment pin; the linearization residual |
| **K9, K10**; **R19–R22** | promotion feasibility, the `n0`–`H0` artifact, linear-response inadequacy, the evidence-dominance risk, the 1b trap |
| **PR-6a / PR-6b / PR-6c / PR-7i / PR-11**; PR-9 dissolved | the ladder re-scoped around the promotion |
| **OWNER DECISION 1** four-way; **OD5** conditioned on linear response; **OD6** promoted to required under (c); **OD8** storage updated; **OD10** fourth arm + the K>=2 selection-table impossibility; **OD11** new | the decisions the promotion actually forces |

---

## v2-6 — Scoping statement that must survive into every downstream document

**No literature search was performed in producing v2.** The novelty section maps `PLAN.md`'s claims
onto the **owner's own** scoring table and is not an independent review of Dalang & Baker,
Leyde/Baker/Enzi or Cheng & Gair. The owner's instruction stands and is now a **gate on PR-11**:
*"do an exhaustive paper-by-paper novelty search before saying 'first'"* (`OWNER_CONTEXT.md:138-139`).

---

## Summary of v2 dispositions

| item | disposition |
|---|---|
| R1-SEV2-4 (`H0` cancels, shape parameters do not) | **ROLE CHANGED** — defence → specification of rung 1's coupling set |
| P7 (the pin) | **ROLE CHANGED** (residual → feasibility gate, K9) **+ CORRECTED IN v2** (tolerance wrong by `sqrt(M)` in its estimator-validity role) |
| R2-SEV1-3, R2-SEV1-4 | **ROLE CHANGED** — resolved *conditional on linear response*; reinstated as live for any re-solve design (risk R7, §2.5) |
| R2-SEV1-2 | **ROLE CHANGED** — resolved by a mechanism (projected moment derivatives) that must now be built and pinned (P19) |
| R2-SEV1-5, R1-SEV1-1 | **ROLE CHANGED** — sharp `M > (e^{sigma^2}-1)/(2 epsilon)` form; PR-5b now predicts before measuring |
| R2-SEV2-12 / R12 | **ROLE CHANGED** — CRN determinism is a design property, so the determinism sweep proves less than it appears; K6 unfireable under linear response |
| R1-SEV2-9 / R15 | **ROLE CHANGED** — risk → mandatory scope sentence beside differentiator 1 |
| R1-SEV1-3 | **RE-DISPOSITIONED** — real in `per_pixel`, **vacuous in the production `c_mode`**; eq. (4) is an obligation §1.2 creates, not a repair |
| rev 2's "no fixed-theta provenance firewall" | **CORRECTED IN v2** — false at rung 0 (guard 1 stamps `theta_ref`); true only at rung 1 |
| rev 2's blanket warm-start prohibition | **CORRECTED IN v2** — gradient-norm stopping rule, `eps^2/2` bound; the real defect is a fixed iteration count from a history-dependent point |
| pin P15 + `custom_vjp` | **DELETED** — the production samplers are gradient-free |
| owner staleness claim 1 (K>=2 selection forbidden) | **FALSE** — only *stratified* selection is refused at K>=2 |
| owner staleness claim 2 (joint builder behind) | **TRUE, understated by three defects** — no `c_mode`, no budget gauge fixing, 1.34 Gpc rank floor; plus the binding constraint the context misses (no `--realization-set-id` on the single builder) |
| owner's `M ~ O(10^2)` feasibility premise | **CORRECTED** — the guard-compliant rank is 2520–3780; feasibility comes from member-independence, not from a small `M` |
| owner's "solve at each proposal" | **CORRECTED** — infeasible read literally (~36x wall); linear response is the viable reading |

**Net effect on the ladder:** PR-6 splits into **PR-6a** (control/fallback) and **PR-6b** (the
deliverable); **PR-9 dissolves** into PR-6c minus its two hardest items; **PR-7i** and **PR-11** are
added; PR-3 and PR-4 widen by a day each to carry `theta` and `S`; PR-3 becomes the **decision rung**
(K9, day ~13). Critical path to the promoted deliverable **~33 d** — and PR-6a remains shippable at
~28 d if K9 fires. Eleven OWNER DECISIONs (up from ten, with four rewritten), twenty-two risks (up
from eighteen), ten kill criteria (up from eight), and nineteen numerical pins (P15 deleted; P7b,
P16–P19 added).

*(v4 note: the day counts in this paragraph were re-derived and corrected in v4 — see Part 3,
finding B14. The ladder's own durations sum to 29 d / 34 d as written here, and to 31 d / 36 d with
v4's 3-day PR-0.)*

---
---

# Part 3 — v4 (third adversarial round), 2026-08-10

**Input:** two further independent reviews of **v3**, both returning `verdict: fixable`; 28 findings
(7 distinct SEV1 issues — the envelope-theorem defect was raised by both reviewers — 17 SEV2, 4 SEV3).
**Verification base:** `master` @ `0c5b3db`. **Output:** `PLAN.md` v4, whose new **§0.5** is the
authoritative record of what changed and is written to supersede any v3 sentence that survives
elsewhere in that document.

**Nothing in Parts 1 and 2 is invalidated.** Four v3 claims are **withdrawn** (§0.3's
mutual-exclusivity of effect (A) and the benign branch; §1.7's envelope-theorem step; the inherited
`s_v ~ 5.6e-3` as a global certificate; §2.4's categorical "0 transient" at rung 1), one v3 equation is
**replaced** (eq. (1) by eq. (1')), one v2 operational constant is **corrected by ~1.5 orders of
magnitude** (the warm-start `eps`), and the ladder is **re-ordered** so the two measurements that can
kill the direction run on day 1 instead of days 23 and 28.

Reviewer labels below: **A1–A14** = first review, **B1–B14** = second.

## Part 3-0 — The four findings that changed the design

| id | finding | disposition | carried in |
|---|---|---|---|
| **A1, B1** | The `tau` thresholds re-commit the `sqrt(M)` error v3 congratulates itself for catching: at `tau = 0.02` the plan's own numbers admit 1.2–3.2 nats against a 0.1-nat budget, and `tau = 0.3` admits 18–48. v3's replacement pin `P7c` exists **only inside §0.3** — the pin table, PR-3's gate list, K9 and R19 all still gate on `tau` — and as placed it requires a seam that does not exist until day ~24 at a rung that ends day ~14. | **RESOLVED IN DESIGN.** Do not bound the alignment — **measure it**. `a = grad_xi(sum_i ll_i) = b_GW(sum_i phi_i Phi_i - N_obs <Phi>_sel)` is closed-form from PE samples, injections and the basis alone (no seam), so the gate becomes `osc_theta [ a·(xi_hat_theta - xi_hat_ref) ]` in **nats**, computable at PR-3 from the same 20 prior solves. `P7c` is now in the pin table, PR-3's gate list, K9 and R19; `P7c'` is the nonlinear confirmation at PR-6a; `tau` is retained as a misspecification diagnostic and as the 1b input only. | §0.5 **D4**, eq. (0c); pins P7, P7c, P7c'; K9; PR-3 |
| **A2, B3** | The rung-1 galaxy-side term is **mis-specified**: the envelope theorem is applied to `sum N log pi`, which is not the function it applies to. Stationarity gives `∂l/∂xi = xi_hat`, not `0`, so a `0.5\|\|xi_hat_theta\|\|^2` Occam term is dropped whose theta-variation is first order and reaches ~2e2 nats worst case / ~3 nats at random alignment. `MODEL.tex` `eq:laplacegrad` states the correct rule in the opposite direction, so the two documents disagreed. | **CORRECTED.** The shipped rung-1 term is the **Laplace evidence** `log L_gal = -J(xi_hat_theta) - 0.5 logdet H = l - 0.5\|\|xi_hat_theta\|\|^2 - 0.5 logdet H` (eq. 5). Under linear response the Occam term is **free** — one offline `n_theta`-vector `S^T xi_hat_ref` and one `n_theta x n_theta` matrix `S^T S`. v3's sentence "`S` is needed for the GW-side field shift, not for the evidence term" is **WITHDRAWN**: `S` is needed for both. The frozen-`logdet H` assumption becomes pin **P7d** (free at the same 20 solves; a 0.1% per-mode drift is 1.9 nats at `M/2 = 1890`), and the `H_ref`-vs-`H_theta` draw-covariance term (0.11–0.34 nats at a 10% drift) folds into **P7c'**. | §0.5 **D2**, eq. (5); §1.7; pins P7d, P7c'; risk R25 |
| **A3, A10** | The two measurements that can kill the whole direction cost ≤ 1 day each against artifacts that exist today, and are scheduled at days 23 and 28 of a 33-day plan — while the plan's own recorded numbers predict both will fire. The literature search that decides whether the work has a paper is gated at PR-11, day ~50, while a publishable differentiator is claimed at day ~28. | **RESOLVED IN DESIGN — the highest-value change of the round.** PR-0 grows 1 d → 3 d and becomes the **decision rung**: (1) the three-column cost baseline, (2) `sum_i phi_i` and the closed-form `sigma`, (3) the **Q-on-at-anchor oscillation** — an upper bound on everything the ladder can buy, measured with code that already runs — and (4) the exhaustive novelty search, moved from PR-11. New kill criterion **K0** ends the ladder at day ~3 if the field cannot move the posterior; the deliverable is then the bounded-systematic result at ~3 days instead of ~36. An independent integral this session over the shipped `C_sel` gives an in-support missing-budget fraction of **1.3e-4 all-sky / 7.9e-5 over the occupied footprint**, reconciling with R1's 6e-5 and the reviewer's 2.7e-4: all agree the field redistributes ~0.01% of the budget. | §0.5 **D3**; PR-0; **K0**; risk R23 |
| **B2** | §3.4's gradient and Hessian are those of a **shell-collapsed** model, not of eq. (1). With the exponential inside the shell sum, `eta_pg` is a log-sum-exp: `Phi_g = Phi_s ⊗ phi_z[g]` is false, the Kronecker separability collapses, eq. (3) omits a sign-indefinite second-derivative term, the exact object costs 3.7 GB / 4.5e16 flop per step, and P5/P6 pin an objective the plan does not ship. But a `p`-independent `sum_n W_gn` divides out of `pi_pg` identically, so `W` does something **only** if the exponential is inside — a sharp dilemma. | **RESOLVED IN DESIGN.** Move `W` onto the **basis rows** rather than around the exponential: `phi~_z[g,:] = sum_n W_gn phi_z[n,:]`, `eta_pg = b (Phi_s[p] ⊗ phi~_z[g])·xi` (eq. 1'). Then `eta` is exactly linear in `xi`, `Phi_g` factorizes exactly, eq. (3) is the **exact** Hessian, the sky-constant subspace is an **exact** gauge direction, the two-stage contraction stands, and `W` is not a no-op — it is the photo-z forward convolution *of the basis*. The single neglected term is `(b^2/2)[Var_g(s_p) - <sigma_p^2>_g]`, which is **the same object as `prop:cancel`'s residual**: one stated approximation now buys the theta-cancellation, the separability and eq. (3) together. New pins **P5b** (measure the Jensen residual) and **P5c** (exact quadrature at reduced rank, validation only). Fisher scoring is made normative, which also preserves `H ⪰ I` under PR-6c's saturating link. | §0.5 **D1**, eq. (1'); §3.4; pins P5, P5b, P5c; risk R24 |

## Part 3-1 — SEV1 findings not in the four above

| id | finding | disposition |
|---|---|---|
| **A4** | `s_v ~ 5.6e-3` is inherited and unverified, certifies three separate approximations (Laplace validity, the 1e-3-nat evidence bias, Limit III's expansion), and is contradicted by the plan's own `amp = 1` and its statement that ~60% of modes are unconstrained — where `sigma_post = sigma_prior ≈ 1` by construction. A field with posterior sd 5.6e-3 everywhere would be the prior-collapse regime K3 exists to refuse. | **WITHDRAWN as a global certificate + new pin.** **P20** measures `s_v`'s **distribution** on the real anchor, stratified interior / partial / off-footprint / above-`z_depth`. The claim survives in a narrower and stronger form: the Laplace error is governed by the non-Gaussianity of the **count** term, which lives only on data-constrained voxels — where `sigma_post` is small *because* the counts are large — while the prior modes contribute exactly zero Laplace error because their posterior *is* the Gaussian prior. Limits II and III are restated as formal expansions in `b_GW s` quoting P20, not a scalar. |
| **B4** | §0.3's claim that effect (A) and K9's benign branch "are the same number read two ways" and are mutually exclusive is **backwards**: `Delta(evidence)` is the full count residual contracted with `∂eta/∂theta`, while `Delta xi_hat` is that residual **projected onto `span(Phi)` and damped by `H^{-1}`**. A theta-direction inside `span(Phi)` is absorbed (tau large, evidence flat); one orthogonal to it is not (tau = 0, evidence maximal). Since `eq:cancelresidual` makes the theta-direction a *within-shell* covariance — exactly what a shell-collapsed basis cannot represent — "tau small with a large (A)" is the **structurally expected** case. Both branches of v3's day-13 decision were computed from GW-side quantities and were blind to (A). | **WITHDRAWN + RESOLVED.** §0.3's mutual-exclusivity paragraph is marked superseded and retained for the record. Effect (A) gets **its own gate, free at the same 20 solves**: **P7e** = `osc_theta` of eq. (5) — the *correct* evidence, including the Occam and log-det terms of A2. K9's benign branch now licenses only "the GW-side field shift is negligible" and says nothing about (A); **R21 stays live independently of K9**. The same statistic also bounds guard 5's rung-1 double-counting overlap (see B12). |

## Part 3-2 — SEV2 and SEV3 findings

| id | finding | disposition |
|---|---|---|
| **A5, B6** | **P17** is stated against the **prior** marginalization while the shipped members are Laplace **posterior** draws, so it can pass only when `xi_hat = 0` and `H = I` — i.e. with counts off, or in the prior-collapse regime K3 refuses. It also drops the selection subtraction, and §1.6's displayed one-line derivation carries only the GW factor and therefore cannot produce the event×voxel and voxel×voxel terms the following bullets claim; `MODEL.tex` `eq:limit3` gives a third version. | **CORRECTED, and strengthened.** Two arms: **(a) counts-off** (`H = I`, `xi_hat = 0`) reproduces the prior form `(b^2/2) sum_ij K(x_i,x_j)`, validating against truth; **(b) counts-on** (the shipped configuration) targets `LSE_m ll_m - log M - ll(xi_hat) -> 0.5 a^T H^{-1} a = 0.5 sigma^2`, with `a` from eq. (6) carrying the `-N_obs<Phi>_sel` subtraction by construction. **P17 and §6.5 item 5's `sigma` prediction become one measurement.** The correct Limit-III derivation is `log ∫dxi N(xi;0,I) e^{(a_GW + a_gal)·xi} = 0.5\|\|a_GW\|\|^2 + a_GW·a_gal + 0.5\|\|a_gal\|\|^2`, which *does* produce the three pairings. |
| **A6** | The hierarchy is internally inconsistent whenever `b_GW != b_gal`: the catalogued-host branch is written field-free while the missing branch carries `e^{b_GW s}`. The probability that a **catalogued** galaxy hosts the event is a ratio of intensities `∝ psi w_j exp[(b_GW - b_gal)s(x_j)]`. So Limit I's "`xi` drops out identically" and P16's promotion to "a physics identity, not a routing property" are properties of the **omission**, and P16 will pass by construction. The configurations where this bites are the ones actually scheduled (OD8's mock campaign; Tier E at `b_2/b_1 = 2`). | **CORRECTED — a model change**, and **CONVERTED TO OWNER DECISION 8 (amended)**. The spike weights carry `exp[(b_GW - b_gal)s(x_j) - ((b_GW^2 - b_gal^2)/2)sigma^2(x_j)]` (`MODEL.tex` `eq:zprior-new` amended). **P16 is restated at `b_GW = b_gal`**, where it is a genuine physics identity; at `b_GW != b_gal` it becomes a deliberate *non*-identity with a predicted size. OD8 now recommends `b_GW == b_gal` in the headline — which makes Limit I exact at zero cost — and requires the excess-bias factor wherever `b_GW` is free. |
| **A7** | `prop:warmstart` bounds `J`, but the shipped rung-2 likelihood is `-J - 0.5 logdet H + LSE_m ll_m`; the member term moves by `sigma·eps` — **first** order — so a 0.01-nat budget at `sigma = 2.6` needs `eps < 3.8e-3`, not `0.14`, a factor of ~37. The log-det motion is unbounded by anything in the document, and PR-6c's smooth saturation can make the observed Hessian indefinite, destroying the `H ⪰ I` premise of `eq:warmbound`, of `xi_hat`'s uniqueness and of the Laplace approximation. | **CORRECTED.** The stopping rule is restated on the **likelihood**: `eps < epsilon_budget / sigma`. The log-det is bounded by evaluating `logdet H(xi_*)` consistently at the same stopping point. **Fisher scoring is made normative** (§3.4): `H := I + Fisher` is PSD for *any* link including the smooth saturation, whereas the observed Hessian is not — so `H ⪰ I` survives PR-6c. The "stochastic likelihood" concern does not apply under a deterministic anchored start with a deterministic stopping rule: `logL` is then deterministic in `theta` with a theta-**varying** residual, governed by the same `osc` budget as everything else. |
| **A8** | `prop:compressexact` / `eq:requirefullsolve` is the exactness criterion for design **1b**, which §0.3 forbids; under the shipped 1a it can never fire, yet `MODEL.tex`'s online pseudo-code routes the FAST/FULL switch on exactly that `d2`. Meanwhile the one approximation 1a really makes — the theta-drift of `H` — is dismissed as "second-order" and measured nowhere, though it shifts `logL` by 0.11–0.34 nats for a 10% drift. | **CORRECTED.** `eq:requirefullsolve` is restated as **1b-only**; `MODEL.tex`'s `S1/S2` no longer routes on `d2`; §1.6's citation of `prop:compressexact` for "which corner a pairing sits in" is replaced by eqs. (0c)+(0d), the numbers that actually decide it. The `H`-drift is measured: **P7d** (log-det) and **P7c'** (draw covariance), both at the same 20 solves. |
| **A9** | Factor **(F2) does not exist as a likelihood factor**: `magnitude_loglike_from_stats` is referenced nowhere outside its test, and `theta_sel` enters as an **anchored Gaussian prior**, so `eq:hierarchy` is not the model the pipeline evaluates and rung 2 double-counts `theta_sel` exactly as it double-counts `n0`. | **CONFIRMED [verified this session] and CORRECTED + CONVERTED TO OWNER DECISION 14.** `redshift/selection.py:341` has exactly one non-test reference — its own definition; `inference/prior.py:1207` sets `kind_map[lbl] = ("normal", loc, scale)` and `selection_fit_union.json` gives `cov[0][0] = 2.546e-8` → `sigma(M0hat) = 1.60e-4` mag, matching §3.1's quoted number. The shipped hierarchy is **empirical Bayes on `theta_sel`**. PR-2's work item is re-aimed from "re-derive F2's disjointness" to "may the anchored prior coexist with a count likelihood whose base is `f_p C(z;theta_sel) Nbar`" — yes at rung 0, bounded at rung 1, refused at rung 2. **K10 is extended from `n0` to `theta_sel`.** |
| **A10** | (literature-search scheduling half) | **RESOLVED** — see A3; moved to PR-0. |
| **A10** | (framing half) After removing what the owner has already scored as not-novel, differentiator 1 reduces to a **gauge convention** (`prop:gauge`) plus a **correctness requirement** (`core.py:1372-1380` already enforces all-or-none) — and with the mandatory scope caveats, to "we marginalize over the angular placement of missing hosts inside the surveyed volume at fixed budget". | **CONVERTED TO OWNER DECISION 12.** The argument is not refuted; it is put to the owner with a recommendation: **demote differentiator 1 from headline to enabling clause and headline the unification** (Limits I–III as one likelihood, with the theta-coupled field making the interpolation real), which is also what the owner's promoted goal says. Final adjudication after PR-0's novelty search, now three days away. |
| **A11** | The unification recovers Cheng & Gair for `eq:hierarchy` **on paper, not for the estimator** PR-6a/6b ships (which conditions on `d_gal` first, absorbing the counts into `xi_hat` and `H`); `MODEL.tex`'s `Lambda_cat -> 0` trigger **contradicts** the `b_GW b_gal` cross term it is meant to produce; and the expansion parameter is asserted at a value 180x smaller than plausible. | **ACCEPTED, all three.** (a) Limit III is presented as a property of **the hierarchy**, explicitly not of the rung-0 estimator — and the plan now draws the consequence *in favour of the promotion*: conditioning at **frozen theta** removes exactly the cosmology dependence of `C_l^{GWxg}` that a cross-correlation measurement uses, and rung 1 restores it. (b) The trigger becomes "the catalogued fraction of each event's prior mass → 0 while the localization spans many correlation cells", with the counts **retained as data**. (c) The expansion parameter is `b_GW s` over the events' support and is **measured** (P20). |
| **A12** | Effect (A) cannot simultaneously be "a new cosmological constraint from 22.79M galaxies" and negligible under K9's benign branch; and (A) is degenerate with `delta`, with `sigma_z(z)` and with the shell binning, all frozen inside `W`, so §6.4's acceptance rule is too weak. | **SPLIT.** The **premise** (that they are the same number) is **superseded by B4**, which shows the two are different contractions and not co-monotone — the two reviewers disagree here and B4's argument is the correct one. The **consequence** is **ACCEPTED and promoted to OWNER DECISION 13**: §6.4's rule becomes "(A) is reported as a diagnostic and never enters the headline posterior unless `W`'s own parameters — `omega_g`, `sigma_z(z)`, the shell edges — are sampled or profiled." |
| **A13** | §4.2's "so it is exact there too" under stratified selection drops the theta caveat, and `prop:budgetexact(1)` does not cover the stratified branch it is cited for. | **CONFIRMED [verified] and CORRECTED.** `build_lognormal_completion.py:730-732` sets `w_budget = ((1.0 - Cfine_s) * dN_exp_density)[stratum_map[fit]]` — **p-dependent**, evaluated at the build-time fit: exactly `prop:budgetexact(2)`'s two mechanisms. The stratified branch conserves the consumed budget **only at `theta = theta_ref`** and leaks elsewhere at the order of the stratum-to-stratum `C_sel` spread. The surrounding claim survives for the production (non-stratified) configuration, which is what it needs. |
| **A14** | "Per-evaluation transient added: 0" is false at rung 1: `row_fac_shift` is consumed as `row_fac_shift[pix]`, so it is the `(n_rows × M_z)` row expansion — 1.46 MB per evaluation, ~375 MB at 256 concurrency — and the `block_sizing` routing decision turns on exactly that categorical claim. | **ACCEPTED and CORRECTED.** §2.4's "0 transient" is scoped to **rung 0**; rung 1's transient is tabulated at 1.46 MB / ~375 MB and **routed through the guarded transient branch** (`_slopes_and_fixed:708`, `batch_scale` at `:725`) rather than the static branch. Not fatal against 72.7 GiB free, but reserved rather than assumed — the ~34 GB precedent at `block_sizing.py:623` is the reason. |
| **B5** | The budget moments of eq. (2) are **all-sky** while eq. (4) and the `Q == 1` convention are footprint-restricted, so the seam returns `Q = exp(-rho) != 1` off-footprint and per-realization budget conservation — differentiator 1's defining property — fails; the shipped `renormalize_q_mean_one` docstring warns against exactly the all-sky version; and the gather index space is the 49,143-row PE/injection union, not 30,470, with no index map specified. | **CONFIRMED [verified] and CORRECTED.** Eq. (2) is restated over the fitted footprint `F` (`A_m^F`, `B_m^F`, `P_F`, `F_F`). Splitting eq. (4) at `f_p = 0` shows the off-footprint block is conserved trivially, so the constraint binds on the footprint block alone. New pin **P13b**: the seam returns **bit-zero** `logQ` on every gathered row outside `F` (~38% of the union), via an explicit index map from gathered pixel → footprint row. |
| **B7** | §6.5 item 5 predicts `sigma` with an unadorned (Euclidean) norm where eq. (0)'s own derivation requires the `H^{-1}` norm, and its ingredient list omits `H_chol`. Since `H ⪰ I` this systematically **over**-predicts `sigma`, which drives an exponential `M_draw` requirement and OWNER DECISION 5. | **CORRECTED.** `sigma = ||L_H^{-1} a||_2` with `a` from eq. (6); `H_chol` is added to the ingredient list; the same `a` now serves P7c's gate, P17 arm (b)'s closed form and §6.5's prediction, so **one object serves three purposes**. Computed at **PR-0**, not PR-5b. |
| **B8** | The cost table divides a latent-minus-table delta by a baseline that has **no LSS and no member marginalization at all**, so the deliverable costs ~2.6x more than the quoted percentages against the actual production configuration. | **ACCEPTED.** §2.3 now carries both columns; the baseline-relative numbers are **+31% / +125% / +250%** at `M_draw = 8 / 32 / 64` against 27.5 ms. **OWNER DECISION 5 and kill criterion K4 are taken on the baseline-relative column**, and PR-0 reports all three. |
| **B9** | §4.3 fixes `amp = 1`, which makes `b_gal` identified by the counts, while §3.4 calls them degenerate at K=1 and sets `s_b` to a 20% prior width — and `s_b` drives the rank-1 draw-covariance inflation, hence `sigma`, hence `M_draw`, hence OD5. It also makes Tier-B's "latent-on CI ≥ table CI" passable or failable by choice. | **ACCEPTED.** `s_b` becomes the **profile curvature** `s_b^2 = [-d^2 log p_count/db^2]^{-1}` at the anchor, computed against the same `H_chol`, with a stated systematics floor; the 20% dial is retired. |
| **B10** | PR-7's Tier-E gate (iii) is **vacuous**: it requires a K=2 selection run "without `--allow_unverified_shared_lss_members`", but that flag and its check live on the table-loader path latent mode deletes, so it passes by deletion of the check rather than by satisfaction of the property — the same routing tautology R1-SEV2-4 caught in rev 1. | **ACCEPTED.** (iii) is demoted to a **statement of fact**; the gate becomes (i) bias-ratio recovery, (ii) shared-`xi` coupling tighter than two independent fits, and a new substantive **(iii')**: run the shared-`xi` likelihood **and** an artificially decoupled two-field variant on the same mock and show the bias-ratio credible region differs in the predicted direction — demonstrating the coupling the flag throws away. (The reviewer independently re-verified every leg of §0.4's seam diagnosis on `0c5b3db`; that analysis stands.) |
| **B11** | K8 at PR-2 is **near-certain to fire**, and the ladder's only response is "stop and re-plan" at day ~9; §1.2 also leads with the smaller of the two effects — the dominant term is the 18,682 pixels at `f_p = 0`, worth roughly +45% of the all-sky missing budget, not the p99 partial pixel. The production posterior also rails, so "1 sigma" is undefined on the arm K8 names. | **ACCEPTED.** §1.2 is re-ordered to lead with the off-footprint term. **K8 becomes non-terminal**: evaluated on the Tier-B/C closure and on a **non-railing** production configuration, with the `C`-side change shipping as **its own deliverable with its own `H0` arm and golden** rather than as a PR-2 side effect. |
| **B12** | Rung 1 re-opens a weakened form of the double-counting the plan proves fatal for rung 2: guard 5's premise ("the count channel carries zero information about `(n0, delta, theta_sel)` by construction") is exactly true at rung 0 and **false at rung 1**, where `eq:cancelresidual` is precisely the channel through which `delta` and `theta_sel` re-enter — while OD6 requires a prior on those same parameters calibrated from the same 22.79M counts. | **ACCEPTED and made measurable.** Guard 5 is restated **per rung**. At rung 1 the overlap is exactly the within-shell residual, which is exactly what **P7e** measures — so the gate statistic and the double-counting bound are the same number. New rule: if `osc_theta` of eq. (0d) restricted to the `(delta, theta_sel)` directions exceeds 0.1 nat, widen the `delta` prior by the measured overlap or restrict rung 1's coupling set to `(Om0, w0, wa)`. |
| **B13** | Four internal-consistency defects: colliding factor labels (`PLAN.md` F1–F5 vs `MODEL.tex` F1–F4); `prop:compressexact` mis-cited in §1.6; the shell-total factorization called "exact" when the discarded factor carries `O(b^2 Var)` field information; and the "exact gauge direction" claim, which holds only under shell collapse. | **CORRECTED, all four.** `MODEL.tex`'s labelling is authoritative — (F1) prior, (F2) magnitudes, (F3) counts, (F4) GW — with the shell split written (F3a) monopole / (F3b) placement. The `prop:compressexact` citation is replaced by eqs. (0c)/(0d). The **factorization** is exact for any joint; using only the second factor is a **partial likelihood** whose discarded shell monopole retains `O(b^2 Var_sky(s)/2)` content after §4.1's linear-order projection. The gauge direction is **exact for eq. (1')**, which is D1's resolution. |
| **B14** | The critical-path day counts do not sum (the ladder's own durations give 29 d / 34 d, not 28 d / 33 d); and the cited evidence for "the production run loads no LSS table" is a banner that prints identically in runs that **did** load one. | **CORRECTED, both.** The path is restated as **31 d to PR-6a / 36 d to PR-6b** with v4's 3-day PR-0, with two early exits (K0 at day ~3, K9 at day ~14). The no-LSS claim is re-anchored on the **configuration** (`sbatch_ns_joint_sel.sh`: `--use_lss false`, no `--lss_completion`), not the banner. The reviewer also records that every independently checkable number in v3 reproduced exactly — 27.5/49.3 ms, 256 concurrent evals, 20 free dimensions, R12's Q-on deltas, the 47.6/21.7 GB dense-`Phi` sizes, the 190 Mpc kernel arithmetic, the sphere-guard WARN-only docstring. |

## Part 3-3 — Summary of v4 dispositions

| finding | severity | disposition |
|---|---|---|
| A1 / B1 `tau` gate + `P7c` orphaned and unschedulable | SEV1 | **RESOLVED IN DESIGN** (§0.5 D4; P7c/P7c' in the pin table, PR-3, K9, R19) |
| A2 / B3 envelope theorem; missing Occam term; unpinned log-det | SEV1 | **CORRECTED** (§0.5 D2, eq. 5; P7d, P7c'; risk R25) |
| A3 + A10 inverted ladder; search gated last | SEV1 | **RESOLVED IN DESIGN** (PR-0 = the decision rung; **K0**; risk R23) |
| B2 shell-collapsed gradient/Hessian vs eq. (1) | SEV1 | **RESOLVED IN DESIGN** (eq. 1'; P5b, P5c; Fisher scoring normative; risk R24) |
| A4 `s_v ~ 5.6e-3` | SEV1 | **WITHDRAWN** as a global certificate + **P20** |
| B4 (A) vs the benign branch are not co-monotone | SEV1 | **WITHDRAWN** (§0.3's claim) + **RESOLVED** (**P7e**) |
| A5 / B6 P17 prior-vs-posterior, selection subtraction, derivation | SEV2 | **CORRECTED** — two arms; unified with §6.5 item 5 |
| A6 `b_GW != b_gal` on the spike branch | SEV2 | **CORRECTED (model change)** + **OWNER DECISION 8 amended**; P16 restated |
| A7 warm start bounds `J`, not the likelihood | SEV2 | **CORRECTED** — `eps < budget/sigma`; Fisher scoring preserves `H ⪰ I` |
| A8 `eq:requirefullsolve` is 1b-only; `H` drift unpinned | SEV2 | **CORRECTED** — restated 1b-only; P7c'/P7d measure the drift |
| A9 (F2) is not a likelihood factor | SEV2 | **CONFIRMED [verified]** + **OWNER DECISION 14**; K10 extended to `theta_sel` |
| A10 differentiator 1 = gauge + correctness | SEV2 | **CONVERTED TO OWNER DECISION 12** |
| A11 unification: estimator vs hierarchy, trigger, expansion | SEV2 | **ACCEPTED**, all three; turned into an argument *for* the promotion |
| A12 effect (A) acceptance rule | SEV2 | premise **SUPERSEDED BY B4**; consequence → **OWNER DECISION 13** |
| A13 stratified budget exactness | SEV3 | **CONFIRMED [verified] + CORRECTED** |
| A14 rung-1 transient is not zero | SEV3 | **ACCEPTED** — 1.46 MB / ~375 MB, routed through the transient branch |
| B5 all-sky vs footprint moments; off-footprint `Q` | SEV2 | **CONFIRMED [verified] + CORRECTED** + **P13b** |
| B7 `sigma` in the wrong inner product | SEV2 | **CORRECTED** — `sigma = \|\|L_H^{-1} a\|\|_2` |
| B8 cost denominator | SEV2 | **ACCEPTED** — +31% / +125% / +250%; OD5 and K4 re-based |
| B9 `amp = 1` vs `s_b = 20%` | SEV2 | **ACCEPTED** — `s_b` from profile curvature |
| B10 Tier-E gate (iii) vacuous | SEV2 | **ACCEPTED** — demoted to fact; new substantive (iii') |
| B11 K8 near-certain; §1.2 mis-ordered | SEV2 | **ACCEPTED** — K8 non-terminal; §1.2 re-ordered |
| B12 rung-1 double counting | SEV2 | **ACCEPTED** — guard 5 per rung; bounded by P7e |
| B13 labels, mis-citation, "exactly", gauge direction | SEV3 | **CORRECTED**, all four |
| B14 day counts; banner citation | SEV3 | **CORRECTED**, both |

**Net effect on the ladder.** **PR-0 becomes the decision rung** (1 d → 3 d) and carries the two
measurements that can end the direction plus the novelty search; **K0** is added; **K8 becomes
non-terminal**; **K9 fires on nats, not on `tau`**; PR-3's gate list becomes P7c / P7d / P7e / P7b with
`tau` demoted to a diagnostic; PR-7's Tier-E gate is replaced by a substantive one. Eq. (1) is replaced
by eq. (1'); eq. (5) restores the Occam and log-det terms. **Fourteen OWNER DECISIONs** (up from
eleven), **twenty-five risks** (up from twenty-two), **eleven kill criteria** (up from ten), and
**twenty-six numerical pins** (P5b, P5c, P7c, P7c', P7d, P7e, P13b, P20 added). Critical path **31 d /
36 d**, with early exits at day ~3 and day ~14.

**Scoping statement, carried forward unchanged from v2-6:** no literature search has been performed in
producing v3 or v4. It is now a **PR-0** gate, not a PR-11 gate.
