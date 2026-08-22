# Owner-supplied external analysis (gpt56sol) on Q novelty and the field-level direction (2026-08-10)

[Verbatim from the owner. Math was pasted from a rendered source, so some
expressions are typographically mangled; the intent is recoverable and the
repo is the arbiter of what the code actually does.]

Yes. Q is now the more interesting part of the completeness machinery than
C, but the novelty story has to be stated carefully because the literature
has moved quickly.

The cleanest way to think about it is: dN_miss/dz(p,z) = [1-C(z)] dN_exp/dz
(how much is missing) x Q(p,z) (where the missing galaxies live), with Q
ideally constrained to have no monopole freedom of its own.

## 1. What was old
Before the explicit Q machinery, the missing branch was
dN_miss = (1-C) dN_exp max[1 + b_miss delta_g(p,z), 0] — a linear local
overdensity correction: take the observed galaxy overdensity field and
assume missing galaxies trace it with some bias. Useful but conceptually
primitive: no posterior over the unknown density field, no borrowing
between neighboring lines of sight, no nonlinear positivity-preserving
density model, no principled treatment of empty/poorly observed regions.
And the broader idea "missing galaxies should trace LSS rather than be
homogeneous" is definitely not new: Dalang & Baker's variance-completion
work was explicitly introduced to restore clustering of invisible hosts,
and the later implementation paper packages the correction as a ratio
relative to homogeneous completion. So the existence of a multiplicative
Q-like LSS correction is not the novelty.

## 2. The first real Q: radial Poisson-lognormal completion
s_p(chi) ~ N(0, K_chi); N_pv ~ Poisson[C_pv N_exp,v exp(b s_pv - b^2
sigma_s^2/2)]; Q_pv = exp(b s_pv - b^2 sigma_s^2 / 2). Lognormal form
guarantees positivity and E[Q]=1 under the prior. The radial
implementation solves a Poisson-lognormal MAP independently per HEALPix
line of sight with a correlated Gaussian prior along comoving distance.
A substantial improvement over 1 + b delta, but with the limitation
p(s_p1, s_p2) = p(s_p1) p(s_p2) for different sky pixels: a completely
empty pixel learns nothing from its neighbors. So radial Q is 1-D Bayesian
smoothing along each LOS, not a reconstructed 3-D density field. And under
the old per_pixel C it has a deeper identification problem: C and Q are
both learned from the same local counts — in that configuration Q is
mainly the sub-smoothing residual after per-pixel C has swallowed much of
the clustering signal. Useful, but not a novelty claim.

## 3. What is genuinely new in the current Q — five upgrades
A. Q is now a PLACEMENT field, not another completeness amplitude. The
code renormalizes Q at every z so that sum_p w_p(z) Q_p(z) / sum_p w_p(z)
= 1 with w_p(z) = [1 - C_p(z)] dN_exp(z), i.e. Q_p -> Q_p / <Q(z)>_w.
Thus sum_p dN_miss,p = sum_p [1-C_p] dN_exp regardless of Q: (C, n0)
determine the missing MONOPOLE; Q only REDISTRIBUTES it. Needed because
the posterior mean of a lognormal field has a Jensen contribution
(E[e^s] != e^{E[s]}) and spatially varying posterior variance otherwise
inflates the budget substantially; the renorm is applied separately to
every posterior realization. Excellent methodology — though mean-one
density normalization itself is not a new statistical invention; applying
it this explicitly to the dark-siren missing-host budget gives a very
clean estimand.

## 4. The major upgrade: gp3d
One continuous field s(nhat, z) instead of independent pixels: low-rank GP
s(x) = Phi(x) xi, xi ~ N(0, I), x = (nhat, zeta), zeta = log(1+z), kernel
K = K_sph(nhat, nhat') K_z(zeta, zeta'), finite Fibonacci-sphere x
redshift inducing basis. Q at an empty pixel is informed by neighbors; one
joint Poisson-lognormal problem instead of thousands of independent LOS
problems. AND the deterministic Q is the Laplace posterior MEAN, not the
MAP: log E[Q(x)] = b s_MAP(x) - (b^2/2)[sigma_prior^2(x) -
sigma_post^2(x)], so an unconstrained location has sigma_post -> sigma_prior
and Q -> 1: no information => homogeneous completion, rather than an
arbitrary MAP depression from the lognormal mean shift.

## 5. Empty pixels are now information
Under aggregate or selection C, Q is fitted over EVERY sky pixel; an empty
pixel has N_obs = 0 against a positive lambda_base = C(z) N_exp(z), so
zero is a measurement: 0 ~ Poisson[lambda_base Q] drives Q < 1 where the
surrounding field supports a void. Previously empty pixels were assigned
Q = 1. Important improvement, though not fundamentally new Bayesian logic.

## 6. selection C + Q is much cleaner than old C + Q
With C = C_sel(z; theta_LF) containing no count fluctuations, the Q
likelihood is really N_obs(p,z) ~ Poisson[C_sel(z) N_exp(z) Q(p,z)]:
C_sel is survey thinning, Q is density structure. Much closer to a
legitimate generative model than C_p = N_obs,p / N_exp followed by
reconstructing clustering from whatever is left.

## 7. Q now has a posterior, marginalized in the GW likelihood
For gp3d, p(xi | d_gal) ~ N(xi_MAP, H^-1); matched field realizations
xi^(m) -> Q^(m)(p,z); the gp3d ensemble uses the full low-rank latent
Hessian (sky-z correlations retained, unlike the radial FFT Hessian).
--lss_marginalize computes logL = logsumexp_m[logL_m] - log M, with each
member entering BOTH the GW event terms and the selection integral; a
factored implementation computes member-independent pieces once and vmaps
the cheap missing-density part across realizations. More than a posterior
predictive band: the field uncertainty enters the cosmological likelihood.

## 8. The multi-survey shared-Q machinery
The joint builder constructs ONE shared latent xi; catalog k sees it with
its own tracer bias: Q_k(x) = exp[b_k s(x) - ...]. Joint objective:
-log p(xi | {N_k}) = xi'xi/2 + sum_k sum_v [lambda_kv - N_kv log
lambda_kv]. Two surveys constrain the same realization; posterior member m
in catalog 1 IS the same underlying xi_m as member m in catalog 2 —
exactly what sum_m L(Q_1^m, ..., Q_K^m) needs. Randomly matching
independent posterior samples from separately reconstructed surveys would
be physically wrong.

## Novelty scoring table
- Homogeneous missing galaxies: none
- 1 + b delta_g correction: low
- A multiplicative LSS Q: low
- Using clustering to reconstruct missing galaxies: not novel
- Poisson/lognormal Bayesian density reconstruction: not novel broadly
- 3-D Bayesian dark-siren catalog reconstruction: not novel broadly
  (Leyde, Baker & Enzi "Cosmic Cartography" papers already do Bayesian
  reconstruction of the galaxy field, propagate voxel-count uncertainty,
  jointly infer aspects of the magnitude distribution; paper I describes
  marginalization over cosmological and bias parameters)
- Low-rank sphere x z GP implementation: interesting implementation
- Posterior-mean rather than MAP Q: strong technical choice
- Per-z mean-one missing-budget constraint: interesting methodological
  contribution
- Empty pixels as Poisson information: important, not fundamentally new
- C_selection separated from Q: quite interesting combination
- Marginalize Q members through complete GW HBI: potentially novel /
  strong differentiator
- Joint multi-survey shared latent Q: potentially very novel in dark
  sirens
- Joint multi-survey + selection C + GW marginalization: NOT IMPLEMENTED
  YET; strongest paper direction

A paper whose abstract says "We introduce Bayesian LSS reconstruction for
incomplete dark-siren catalogs" would not survive a literature review.

## Where the current code may genuinely differentiate
1. BUDGET-PRESERVING COMPLETION-FIELD MARGINALIZATION: posterior
   realizations of the clustered missing field, each constrained to carry
   zero missing-budget monopole, directly marginalized through the event
   likelihood and detector-selection normalization: Q^(m) with
   <Q^(m)>_{w,z} = 1 plus logL = LSE_m logL(Q^(m)) - log M. Not found in
   the dark-siren completion literature checked — but do an exhaustive
   paper-by-paper novelty search before saying "first".
2. SHARED MULTI-TRACER COMPLETION REALIZATIONS: one posterior over delta
   with per-tracer (C_k, b_k), matched-realization marginalization in a
   multitracer dark-siren HBI. Closer still to a real methods
   contribution. BUT: the joint multi-survey Q builder has not caught up
   with selection C (calls the common survey assembly without a nonlegacy
   c_mode; centered on the count-derived base), and [AS OF THIS ANALYSIS]
   the inference forbade c_mode=selection for K>=2. The two best ideas
   exist on opposite sides of a seam. That seam is where I would work
   next.

## What could be really novel
Stop treating Q as a file. Make Q a latent variable in the dark-siren
hierarchy: delta(x) ~ GP(0, K_psi) (or a physically motivated lognormal
density prior); for survey/tracer k: N_kv ~ Poisson[C_k(v; phi_k)
Nbar_k(v; n0_k, delta_k) exp(b_k delta_v - b_k^2 sigma_v^2 / 2)]; the
GW-host intensity conditioned on THAT SAME field: lambda_GW(v) ~
R(z|Lambda) exp[b_GW(z, Lambda) delta_v], or with tracer mixtures
lambda_GW(v) = sum_k f_k lambda_host,k(v). Then infer jointly
p(H0, Lambda, delta, {C_k}, {b_k}, b_GW, {f_k} | d_GW, d_gal,1..K).
No frozen Q. No Q provenance mismatch. No Q built at fixed n0, fixed
cosmology, or theta_hat_selection. Every cosmological proposal changes
the galaxy-field likelihood consistently.

## The unification payoff
This could unify catalog dark sirens and cross-correlation dark sirens.
Cheng & Gair's harmonic formulation shows the cross-correlation approach
can be understood as marginalizing over realizations of the unknown galaxy
field, incorporating galaxy-galaxy clustering, with survey selection at
the theoretical-field level. This version occupies the missing middle:
- complete catalog + excellent localization -> ordinary discrete-host
  dark sirens (sum of weighted host spikes);
- incomplete catalog + good localization -> observed spikes + latent
  clustered missing field (the present architecture);
- very incomplete / many diffuse events -> after marginalizing the latent
  field, approaches a correlation/statistical-field treatment.
One generative hierarchy interpolating catalog sirens <-> field-level
sirens <-> cross-correlation sirens, instead of separate methods.

## The minimum version worth doing
No BORG needed. Use the existing low-rank gp3d basis delta(x) = Phi(x) xi,
xi ~ N(0, I), but stop exporting xi -> Q.h5. At each cosmological/HBI
proposal: evaluate each survey's selection C_k; construct the multi-tracer
Poisson likelihood; solve/update the conditional posterior
p(xi | theta, d_gal); Laplace-marginalize xi, or importance-sample a small
fixed set of whitened latent draws; evaluate GW PE and GW selection using
the same latent realization; marginalize xi inside the global likelihood.
With only M ~ O(10^2) whitened coefficients this is actually plausible.
The methodological statement becomes: "A joint finite-event, multi-tracer,
field-level likelihood for dark siren cosmology that simultaneously
marginalizes survey selection, catalog incompleteness, large-scale
structure, and GW-host bias." Much harder for Cosmic Cartography, variance
completion, or ordinary cross-correlation papers to subsume. The current
repository already contains maybe 70% of the machinery: selection C,
shared gp3d latent representation, multi-survey joint solves, matched
field realizations, field-member GW marginalization, consistent
PE/selection propagation. The missing step is to abolish precomputed Q and
make the latent LSS field part of the actual HBI. That, rather than making
gp3d prettier, is the jump to make next.

## Owner directives layered on top (from the session)
- Single tracer FIRST, compatible with multitracer — "that's where the
  money is at". K>=2 must be adding terms, not refactoring.
- Deliverable of this update: revised PLAN.md + reviews.md + a LaTeX/PDF
  proposal for the owner's ingestion: what we do now, the proposed work
  (mainly math + pseudo-code), proposed PRs.
