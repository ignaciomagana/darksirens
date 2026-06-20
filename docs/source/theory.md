# Theory & methods

This page collects the mathematical framework that `darksirens` implements. It
is the reference the per-module API pages point back to: each
[reference page](reference/api.md) explains *where* a quantity is computed, while
this page explains *what* it is and *why*. Notation is shared throughout the
code and the docs.

```{contents}
:local:
:depth: 2
```

## Notation and conventions

| Symbol | Meaning |
|---|---|
| $\theta$ | population (mass/spin) hyper-parameters |
| $\Lambda = \{H_0, \Omega_{m,0}, w_0, w_a\}$ | cosmological parameters (CPL dark energy) |
| $m_1, q$ | source-frame primary mass and mass ratio $q=m_2/m_1$ |
| $m_1^{\rm det} = (1+z)\,m_1$ | detector-frame (redshifted) primary mass |
| $d_L$ | luminosity distance; $z = z(d_L; \Lambda)$ its redshift |
| $\chi_{\rm eff}$ | effective aligned spin |
| $\hat{n} = (n_x, n_y, n_z)$ | unit sky-direction vector |
| $R(\theta, z, \hat{n})$ | astrophysical merger-rate density |
| $p_{\rm pop}$ | normalised source-parameter population density |
| $\mathcal{N}$, $\mu$ | expected detected count / selection (sensitive) volume |

The **canonical sample basis** used inside the likelihood is $(m_1^{\rm det}, q,
d_L)$, because that is the space in which the GW parameter-estimation (PE)
samples and the injection set are drawn. Redshift is recovered from distance
through the cosmology, $z = z(d_L; \Lambda)$
([`utils.cosmology`](reference/utils.md)).

## Hierarchical Bayesian inference

`darksirens` infers $(\Lambda, \theta)$ from a catalog of $N_{\rm obs}$ GW
events $\{d_i\}$ under an inhomogeneous-Poisson population model. Marginalising
the Poisson rate normalisation with a scale-invariant prior gives the standard
selection-corrected hierarchical likelihood (Mandel, Farr & Gair 2019; Vitale et
al. 2022):

$$
\ln \mathcal{L}(\{d_i\} \mid \Lambda, \theta)
= \sum_{i=1}^{N_{\rm obs}} \ln \!\left[ \frac{1}{N_{{\rm s},i}}
  \sum_{k=1}^{N_{{\rm s},i}} w_{ik} \right]
\;-\; N_{\rm obs}\, \ln \mu(\Lambda, \theta).
$$

The first term is a Monte-Carlo estimate of the per-event marginal likelihood
over that event's PE samples; the second is the selection correction, with
$\mu$ the expected number of detections (up to the rate normalisation). Both
terms are built from the **same** per-sample weight $w$, which is the key to the
estimator's robustness — see
[`inference.likelihood_core`](reference/inference.md).

### The per-sample weight

For a sample drawn in $(m_1^{\rm det}, q, d_L)$ with PE prior weight
$\pi_{\rm PE}$, the importance weight is

$$
w = \frac{p_{\rm pop}\!\big(m_1, q, \chi_{\rm eff} \mid \theta\big)\,
          p(z \mid \hat n)\, \dfrac{\mathrm{d}N}{\mathrm{d}z}(z\mid\theta)\,
          \mathcal{J}}{\pi_{\rm PE}},
$$

where $\mathcal{J}$ is the Jacobian from the source frame $(m_1, q, z)$ to the
detector frame $(m_1^{\rm det}, q, d_L)$,

$$
\mathcal{J} = \left| \frac{\partial (m_1, z)}{\partial (m_1^{\rm det}, d_L)} \right|
= \frac{1}{1+z}\,\frac{1}{\mathrm{d}d_L/\mathrm{d}z},
$$

implemented in `inference.utils.log_jacobian_m1src_q_z_to_m1det_q_dL`. The
redshift law $p(z\mid\hat n)\,\mathrm{d}N/\mathrm{d}z$ is the **universe-model
prior** (below), and $p_{\rm pop}$ is the **population model**. Samples whose
$d_L$ falls outside the tabulated $z(d_L)$ support are assigned $w=0$
($\ln w = -\infty$), which propagates correctly through the log-sum-exp.

### Selection function

The sensitive volume is estimated from a set of $N_{\rm draw}$ detected software
injections with draw probability $p_{\rm draw}$ (Farr 2019):

$$
\mu = \frac{1}{N_{\rm draw}} \sum_{j \in \text{found}} w_j,
\qquad
\widehat{\sigma}^2_\mu = \frac{1}{N_{\rm draw}^2}\sum_j w_j^2 - \frac{\mu^2}{N_{\rm draw}}.
$$

`inference.selection._lse_to_log_mu_neff` returns
$(\ln\mu, N_{\rm eff}, \ln\widehat\sigma^2_\mu)$ in log-space, with the effective
sample size

$$
N_{\rm eff} = \frac{\mu^2}{\widehat\sigma^2_\mu}.
$$

The estimate is only trusted when $N_{\rm eff} > 4\,N_{\rm obs}$ (Vitale et al.
2022); otherwise `selection_log_correction` returns $-\infty$, vetoing that
proposal. The third return, $\ln\widehat\sigma^2_\mu$, is consumed by the
strong-lensing cluster combiner ([`lensing`](reference/lensing.md)); the
single-event likelihood ignores it.

A crucial property: because $w$ is identical in the PE term and the selection
term, any common multiplicative rate factor — e.g. the angular factor
$g(\hat n)$ or a marked-host efficiency $h(m\mid\eta)$ — is applied to both and
its overall normalisation cancels. This makes those factors *self-calibrating*
against an isotropically-drawn injection set.

## Cosmology

The background is a flat $w_0 w_a$CDM (CPL) model with

$$
E(z) = \sqrt{\Omega_{m,0}(1+z)^3 + (1-\Omega_{m,0})\,(1+z)^{3(1+w_0+w_a)} e^{-3 w_a z/(1+z)}},
$$

comoving distance $\chi(z) = (c/H_0)\!\int_0^z \mathrm{d}z'/E(z')$, luminosity
distance $d_L = (1+z)\chi$, and comoving volume element
$\mathrm{d}V_c/\mathrm{d}z = 4\pi (c/H_0)\,\chi^2/E(z)$. The inverse
$z(d_L)$ is tabulated and interpolated for speed
([`utils.cosmology`](reference/utils.md)); fiducials are Planck15.

## Universe models and the redshift prior

The `--universe_model` flag selects how the redshift law $p(z\mid\hat n)$ is
built ([`em.prior`](reference/em.md)):

- **`spectral_sirens`** — no catalog; a pure comoving-volume rate prior
  $p(z) \propto \frac{1}{1+z}\frac{\mathrm{d}V_c}{\mathrm{d}z}(1+z)^{\gamma}$.
- **`dark_sirens`** — incomplete galaxy catalog (below).
- **`dark_sirens_complete`** — assumes a 100%-complete catalog.
- **`bright_sirens`** — an EM counterpart pins the sky/redshift.
- **`spectral_sirens_wl`** — spectral sirens with weak-lensing magnification
  marginalisation ([`lensing`](reference/lensing.md)).

For the dark-siren models the per-pixel redshift prior is assembled from an
observed (catalog) term and a missing (incompleteness) term,

$$
p(z \mid \text{pix}) = \frac{N_{\rm obs}(\text{pix})\, p_{\rm cat}(z\mid\text{pix})
  + \mathrm{d}N_{\rm miss}(z\mid\text{pix})}
  {N_{\rm obs}(\text{pix}) + N_{\rm miss}(\text{pix})},
$$

where $p_{\rm cat}$ is a per-galaxy redshift-kernel sum and $N_{\rm obs}$ is the
catalog count in the pixel. To keep the JIT likelihood deterministic these
states are precomputed once per proposal by `prepare_redshift_prior_state` and
read by the traced `eval_redshift_prior_with_state`.

### Catalog completeness and the missing branch

The completeness is a matched-kernel ratio of the observed galaxy density to the
expected (homogeneous) density, clipped to $[0,1]$:

$$
C(p,z) = \min\!\left[\,\max\!\left(\frac{\mathrm{d}N_{\rm obs}/\mathrm{d}z}
{\mathrm{d}N_{\rm exp}/\mathrm{d}z},\,0\right),\,1\right],
$$

and the missing-galaxy density is the incomplete fraction of the expected count
modulated by a large-scale-structure factor,

$$
\frac{\mathrm{d}N_{\rm miss}}{\mathrm{d}z}
 = \big[1 - C(p,z)\big]\,\frac{\mathrm{d}N_{\rm exp}}{\mathrm{d}z}\,Q_{\rm LSS}(p,z).
$$

The default factor is the legacy local-overdensity model
$Q_{\rm LSS} \to \max(1 + b_{\rm eff}\,\delta_g(p,z),\,0)$. Optionally it is
replaced by a precomputed **LSS-conditioned lognormal completion field**
$Q_{\rm LSS}(p,z)$ (next section). See [`em.completion`](reference/em.md).

### LSS-conditioned lognormal completion

$Q_{\rm LSS}$ is a clustered, mean-one correction built **offline** so the
likelihood never samples a field. The latent log-overdensity $s$ is a Gaussian
field; the completion factor is the mean-one lognormal

$$
Q = \exp\!\big(b\,s - \tfrac12 b^2 \sigma_s^2\big),
$$

and the binned counts are Poisson with a clustered, completeness-modulated rate
$\lambda_v = C_v\,\mathrm{d}N_{{\rm exp},v}\,Q_v$. The maximum-a-posteriori field
minimises

$$
\mathcal{J}(s) = \tfrac12\, s^\top C^{-1} s
  + \sum_v \big[\lambda_v - N_{{\rm obs},v}\ln\lambda_v\big].
$$

**Radial mode** (`--mode radial`, the default) solves this *independently per
HEALPix pixel* along the line of sight, with a circulant Gaussian-correlation
prior diagonalised by the FFT, plus an FFT-diagonal Laplace ensemble for
uncertainty.

**3-D angular-coupling mode** (`--mode gp3d`) instead solves **one** low-rank
field over all occupied $(\text{pixel}\times z)$ voxels using the whitened
$(\text{sphere}\times z)$ Gaussian process shared with the sky models
([`sky`](reference/sky.md)). The field is linear in the $M\!\sim\!192$ whitened
latents $\xi$, $f = \Phi\,\xi$ with $\Phi = k(X,Z)\,L^{-\top}$ and $L =
\mathrm{chol}\,k(Z,Z)$, so the MAP is a single convex Newton solve. The
deterministic table written for inference is the **Laplace posterior mean**

$$
\mathbb{E}[Q](x) = \exp\!\Big(b\,f_{\rm MAP}(x)
  - \tfrac12 b^2\big[\mathrm{prior\_var}(x) - \mathrm{post\_var}(x)\big]\Big),
$$

with $\mathrm{prior\_var}(x) = k(x,Z)K^{-1}k(Z,x)$ (the Nyström variance) and
$\mathrm{post\_var}(x) = \phi(x)^\top H^{-1}\phi(x)$. Data-free voxels have
$\mathrm{post\_var}=\mathrm{prior\_var}$ so $Q\to 1$ (homogeneous); near data the
variance shrinks and empty pixels **borrow** from occupied neighbours. The
builder is [`em.lognormal_completion`](reference/em.md); both modes write the
same `Q_LSS(p,z)` HDF5 table.

## GW population models

The source-parameter density $p_{\rm pop}(m_1, q, \chi_{\rm eff}\mid\theta)$ and
the rate evolution $\mathrm{d}N/\mathrm{d}z \propto (1+z)^{\gamma}$ are composed
from a naming grammar ([`gw.populations`](reference/populations.md)). A name like
`powerlaw+peak` builds a stick-breaking mixture of component factors; e.g. the
primary mass

$$
p(m_1) \propto \Big[(1-\lambda)\, m_1^{-\alpha}\,\Theta(m_{\min}\!\le m_1\le m_{\max})
  + \lambda\, \mathcal{N}(m_1; \mu_m, \sigma_m)\Big]\, S(m_1),
$$

with smooth low-/high-mass tapers $S$. **Gaussian-process** population models
replace one or more factors by a whitened finite-rank GP prior,

$$
K = a^2\!\prod_{\rm axis}\! k_{\rm axis}(Z,Z) + \epsilon I,\quad
L = \mathrm{chol}(K),\quad
\alpha = L^{-\top}\xi,\quad
f(x_*) = \mu(x_*) + k(x_*, Z)\,\alpha,
$$

with $\xi\sim\mathcal N(0,I)$ — a genuine GP prior (not interpolation) over an
inducing grid $Z$, normalised only over the probability axes $\{m_1,q,\chi\}$.
`BinnedGPPopulation` is the piecewise-constant variant. See
[`gw.populations.gp`](reference/populations.md).

## Sky anisotropy

The rate factorises as $R(\theta, z, \hat n) = R_{\rm pop}(\theta,z)\,g(\hat
n, z)$ with a **mean-one** angular density $g$, so isotropy is exactly $g\equiv
1$ and the shape does not trade off against $R_0$
([`sky`](reference/sky.md)). The registry provides:

- `isotropic` — $g\equiv 1$ (bit-for-bit no-op);
- `dipole` — $g = 1 + \hat n\cdot \mathbf{d}$;
- `multipole` / `multipole_l3` — $g = 1 + \sum_{\ell\ge1}\sum_m a_{\ell m}
  Y_{\ell m}(\hat n)$, yielding the angular power spectrum $C_\ell =
  \sum_m a_{\ell m}^2$;
- `sphere_gp` — a log-Gaussian random field on $S^2$,
  $g = e^{f(\hat n)}/\langle e^f\rangle$;
- `sphere_gp_z` / `overdensity_gp` — a $(\text{sphere}\times z)$ GP with the
  product kernel $k = a^2\,e^{-\tfrac12\lVert\hat n-\hat n'\rVert^2/\ell_\Omega^2}\,
  e^{-\tfrac12(\zeta-\zeta')^2/\ell_z^2}$, $\zeta=\ln(1+z)$, normalised per
  $z$-shell or over the comoving volume respectively.

The factor enters the shared weight as $+\ln g(\hat n, z)$, so it is applied
identically to the PE and selection terms and the detector's own anisotropy
divides out.

## Marked-host model

When the catalog carries per-galaxy marks $m_g$ (stellar mass, sSFR,
metallicity, colour) a sampled BBH-host efficiency reweights each galaxy's
contribution to the redshift prior ([`marks`](reference/marks.md)):

$$
h(m_g \mid \eta) = \exp\!\Big(\textstyle\sum_k \eta_k\, \tilde m_{k,g}\Big),
$$

where $\tilde m_g = m_g - \mathbb{E}[m\mid z_g]$ are **redshift-centred** marks
so that $\eta$ measures host preference at fixed $z$ and does not mimic
$R(z)/H_0/\gamma$. The marked observed host density is $\sum_i w_i\,h(m_i\mid\eta)
\,K_i(z)$ and the missing branch is scaled by the expected efficiency of
unobserved galaxies $\mu_{\rm miss}(z\mid\eta)=\mathbb{E}_{\rm obs}[h\mid z]$.
With $\eta=0$ (or `--mark_model none`) this reduces exactly to the count-based
host model.

## Lensing

### Weak lensing (magnification)

For `spectral_sirens_wl` the per-event distance is magnified, $d_L^{\rm obs} =
d_L/\sqrt{\mu}$, and the PE integral is marginalised over the magnification
$\mu$ with a redshift-dependent PDF $p(\mu\mid z)$
([`lensing.wlmagnification`](reference/lensing.md)). The lognormal backend uses
$\ln\mu \sim \mathcal N(-\tfrac12 s^2, s^2)$ with variance $s^2(z) = a\,z^b$,
integrated by Gauss-Hermite quadrature in the standardised variable
$u = (\ln\mu - m(z))/s(z)$; a tabulated backend integrates an external
$p(\ln\mu\mid z)$ by Gauss-Legendre. The marginalisation is gated by a static
`wl_backend` code so all non-WL models are bit-identical.

### Strong lensing (clusters)

Multiply-imaged sirens behind a galaxy cluster are modelled as a marked Cox
process over image pairs ([`lensing`](reference/lensing.md)). The singly-isothermal-sphere (SIS) lens
gives an optical depth $\tau$, a source-position PDF $p(y)$, and image
magnifications $\mu_\pm = 1 \pm 1/y$; a pair KDE captures the joint distribution
of the two images' parameters. The cluster selection term carries its own
$\ln\widehat\sigma^2_\mu$ and is combined with the singleton selection via
`combined_selection_log_correction`.

## References

- R. Abbott et al. (LVK), population analyses (GWTC catalogs).
- W. Farr, *Accuracy requirements for empirically measured selection
  functions*, RNAAS 3, 66 (2019).
- I. Mandel, W. Farr, J. Gair, *Extracting distribution parameters from multiple
  uncertain observations*, MNRAS 486, 1086 (2019).
- S. Vitale et al., *Inferring the properties of a population of compact
  binaries*, in *Handbook of GW Astronomy* (2022).
- Essick et al. (2023); Isi, Farr & Varma (2023) — GW sky isotropy.
- Ray et al. (2023); Edelman et al. (2023) — GP / B-spline population priors.
