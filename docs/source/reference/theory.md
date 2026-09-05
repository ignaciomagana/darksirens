# Theory and methods

This page states what each quantity in `darksirens` is; the guide pages say
which flag configures it and the [API reference](api/index.md) where it is
computed.

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

The **canonical sample basis** inside the likelihood is $(m_1^{\rm det}, q,
d_L)$, the space in which the GW parameter-estimation (PE) samples and the
injection set are drawn. Redshift comes from distance through the cosmology,
$z = z(d_L; \Lambda)$ (`darksirens.utils.cosmology`).

## Inference

`darksirens` infers $(\Lambda, \theta)$ from $N_{\rm obs}$ GW events $\{d_i\}$
under an inhomogeneous-Poisson population model
([Inference](../guide/inference.md) has the flags). Marginalising the Poisson
rate normalisation with a scale-invariant prior gives the selection-corrected
hierarchical likelihood (Mandel, Farr & Gair 2019; Vitale et al. 2022):

$$ \ln \mathcal{L}(\{d_i\} \mid \Lambda, \theta) = \sum_{i=1}^{N_{\rm obs}} \ln \!\left[ \frac{1}{N_{{\rm s},i}} \sum_{k=1}^{N_{{\rm s},i}} w_{ik} \right] \;-\; N_{\rm obs}\, \ln \mu(\Lambda, \theta). $$

The first term is a Monte-Carlo estimate of the per-event marginal likelihood
over that event's PE samples; the second is the selection correction, with $\mu$
the expected number of detections up to the rate normalisation. Both are built
from the **same** per-sample weight $w$ (`darksirens.likelihood.core`).

### The per-sample weight

For a sample drawn in $(m_1^{\rm det}, q, d_L)$ with PE prior weight
$\pi_{\rm PE}$,

$$ w = \frac{p_{\rm pop}\!\big(m_1, q, \chi_{\rm eff} \mid \theta\big)\, p(z \mid \hat n)\, \dfrac{\mathrm{d}N}{\mathrm{d}z}(z\mid\theta)\, \mathcal{J}}{\pi_{\rm PE}}, \qquad \mathcal{J} = \frac{1}{1+z}\,\frac{1}{\mathrm{d}d_L/\mathrm{d}z}, $$

with $\mathcal{J}$ the Jacobian from the source frame $(m_1, q, z)$ to the
detector frame $(m_1^{\rm det}, q, d_L)$. The code carries its inverse,
$\ln(\mathrm{d}d_L/\mathrm{d}z) + \ln(1+z)$, in
`darksirens.inference.utils.log_jacobian_m1src_q_z_to_m1det_q_dL` and subtracts
it in `log_sample_weight`. The redshift law
$p(z\mid\hat n)\,\mathrm{d}N/\mathrm{d}z$ is the universe-model prior
(Catalogs, below) and $p_{\rm pop}$ the population model. A sample whose $d_L$
leaves the tabulated $z(d_L)$ support gets $w=0$, which propagates through the
log-sum-exp.

### Selection function

The sensitive volume is estimated from $N_{\rm draw}$ detected software
injections with draw probability $p_{\rm draw}$ (Farr 2019):

$$ \mu = \frac{1}{N_{\rm draw}} \sum_{j \in \text{found}} w_j, \qquad \widehat{\sigma}^2_\mu = \frac{1}{N_{\rm draw}^2}\sum_j w_j^2 - \frac{\mu^2}{N_{\rm draw}}, \qquad N_{\rm eff} = \frac{\mu^2}{\widehat\sigma^2_\mu}. $$

`darksirens.likelihood.selection._lse_to_log_mu_neff` returns those three in
log-space, and `selection_log_correction` returns

$$ -N_{\rm obs}\ln\mu \;+\; \frac{N_{\rm obs}(N_{\rm obs}+3)}{2 N_{\rm eff}}, $$

the leading Monte-Carlo uncertainty correction of Farr (2019, eq. 11). The $+3$
follows from marginalising $\mu$ under a scale-invariant prior
$p(\mu)\propto 1/\mu$ (a flat prior gives $N_{\rm obs}(N_{\rm obs}+1)$) and
matches `gwpopulation.vt.ResamplingVT.vt_factor`; it is not a higher-order
Taylor term.

### The total-variance guard

The estimate is trusted only when

$$ N_{\rm eff} > \max\!\left(5 N_{\rm obs},\; \frac{N_{\rm obs}^2}{\sigma^2_{\max} - \sum_i \sigma^2_i}\right), $$

and the correction is $-\infty$ otherwise. The first branch is the Vitale et
al. (2022) floor on the MEAN correction; the second bounds the variance of the
TOTAL log-likelihood estimator (Essick & Farr 2022; Talbot & Golomb 2023),

$$ \sigma^2_{\ln\mathcal{L}} = \sum_i \sigma^2_i + \frac{N_{\rm obs}^2}{N_{\rm eff}} \;\le\; \sigma^2_{\max}, $$

with $\sigma^2_i = \sum_j w_{ij}^2/(\sum_j w_{ij})^2 - 1/n_i$ and
$\sigma^2_{\max} = 1$ nat by default (`--max_likelihood_variance`, the
GWTC-4.0/5.0 threshold). Selection noise enters amplified by $N_{\rm obs}$
through $-N_{\rm obs}\ln\mu$, so the second branch is strictly stronger than
$5 N_{\rm obs}$ once $N_{\rm obs} > 5$. `--selection_neff_guard soft`
replaces the hard wall with a steep smooth penalty whose magnitude tracks the
$-N\ln\mu$ reward it guards, so gradient-based samplers are not
divergence-flagged on every trajectory that brushes it.

### The marked-Poisson generalisation

With both single events and $J=2$ image clusters the two channels are
independent Poisson processes with their own selection integrals:

$$ N_{\rm tot} = N_{\rm sing} + N_{\rm cl}, \quad \mu_{\rm tot} = \mu^{(1)}_{\rm sel} + \mu^{(2)}_{\rm sel}, \quad \sigma^2_{\rm tot} = \widehat\sigma^2\big(\mu^{(1)}\big) + \widehat\sigma^2\big(\mu^{(2)}\big), \quad N_{\rm eff, tot} = \frac{\mu_{\rm tot}^2}{\sigma^2_{\rm tot}}, $$

$$ \ln\mathcal{L}_{\rm sel} = -N_{\rm tot}\ln\mu_{\rm tot} + \frac{N_{\rm tot}(N_{\rm tot}+3)}{2 N_{\rm eff, tot}}. $$

`darksirens.likelihood.cluster_selection.combined_selection_log_correction`
forms these totals in log-space and delegates to `selection_log_correction`, so
the guard applies unchanged at $(N_{\rm tot}, N_{\rm eff, tot})$; with
$\ln\mu^{(2)} = -\infty$ it reduces to the singleton form. The pair channel's
threaded variance covers each branch's driving sample set only, so there the
budget is spent against a lower bound on the true variance.

Because $w$ is identical in the PE and selection terms, a common multiplicative
rate factor (the angular $g(\hat n)$, a marked-host efficiency $h(m\mid\eta)$)
applies to both and its normalisation cancels, which calibrates such factors
against an isotropically-drawn injection set.

## Cosmology

The background is a flat $w_0 w_a$CDM (CPL) model with

$$ E(z) = \sqrt{\Omega_{m,0}(1+z)^3 + (1-\Omega_{m,0})\,(1+z)^{3(1+w_0+w_a)} e^{-3 w_a z/(1+z)}}, $$

comoving distance $\chi(z) = (c/H_0)\!\int_0^z \mathrm{d}z'/E(z')$, luminosity
distance $d_L = (1+z)\chi$, and volume element
$\mathrm{d}V_c/\mathrm{d}z = 4\pi (c/H_0)\,\chi^2/E(z)$. The inverse $z(d_L)$
is tabulated and interpolated (`darksirens.utils.cosmology`); fiducials are
Planck15.

## Populations

The source-parameter density $p_{\rm pop}(m_1, q, \chi_{\rm eff}\mid\theta)$ and
the rate evolution $\mathrm{d}N/\mathrm{d}z \propto (1+z)^{\gamma}$ are composed
from a naming grammar (`darksirens.gw.populations`,
[Populations](../guide/populations.md)). A name like `powerlaw+peak` builds a
stick-breaking mixture of component factors; the primary mass reads

$$ p(m_1) \propto \Big[(1-\lambda)\, m_1^{-\alpha}\,\Theta(m_{\min}\!\le m_1\le m_{\max}) + \lambda\, \mathcal{N}(m_1; \mu_m, \sigma_m)\Big]\, S(m_1), $$

with smooth tapers $S$. Gaussian-process models replace one or more factors by
a whitened finite-rank GP prior over an inducing grid $Z$,

$$ K = a^2\!\prod_{\rm axis}\! k_{\rm axis}(Z,Z) + \epsilon I,\quad L = \mathrm{chol}(K),\quad \alpha = L^{-\top}\xi,\quad f(x_*) = \mu(x_*) + k(x_*, Z)\,\alpha, $$

with $\xi\sim\mathcal N(0,I)$, normalised only over the probability axes
$\{m_1,q,\chi\}$. `BinnedGPPopulation` is the piecewise-constant variant.

## Catalogs

`--universe_model` selects how the redshift law $p(z\mid\hat n)$ is built
(`darksirens.redshift.prior`): `spectral_sirens` uses the catalog-free
comoving-volume rate prior
$p(z) \propto \frac{1}{1+z}\frac{\mathrm{d}V_c}{\mathrm{d}z}(1+z)^{\gamma}$;
`dark_sirens` an incomplete galaxy catalog (below); `dark_sirens_complete` a
100%-complete catalog; `bright_sirens` an EM counterpart that pins the sky and
redshift; `spectral_sirens_wl` adds weak-lensing marginalisation and is owned by
`darksirens_inference_lensing` ([Lensing](../guide/lensing.md)).

For the dark-siren models the per-pixel prior is assembled from an observed
(catalog) term and a missing (incompleteness) term. The default `field`
estimand, the joint catalog host-density estimand, normalises the per-pixel
numerator by the survey-global budget; the `conditional` estimand
(`--catalog_sky_weighting conditional`, radial only) normalises each pixel by
its own:

$$ p(z \mid \text{pix}) = \frac{N_{\rm obs}(\text{pix})\, p_{\rm cat}(z\mid\text{pix}) + \mathrm{d}N_{\rm miss}(z\mid\text{pix})}{Z(\theta)}, \qquad Z(\theta) = \sum_{\text{pix}'} \big[N_{\rm obs}(\text{pix}') + N_{\rm miss}(\text{pix}')\big], $$

$$ p_{\rm cond}(z \mid \text{pix}) = \frac{N_{\rm obs}(\text{pix})\, p_{\rm cat}(z\mid\text{pix}) + \mathrm{d}N_{\rm miss}(z\mid\text{pix})} {N_{\rm obs}(\text{pix}) + N_{\rm miss}(\text{pix})}, $$

where $p_{\rm cat}$ is a per-galaxy redshift-kernel sum and $N_{\rm obs}$ is
the catalog count in the pixel. Under `field` the denominator is one
survey-wide constant, so the relative angular host density survives: a pixel
with 100 candidate hosts carries about $100\times$ the angular weight of a
pixel with one, and for a $K\!\ge\!2$ mixture each catalog's $Z_k(\theta)$
turns the mixture weight $f_{\mathrm{cat},k}$ into the host fraction. Under
`conditional` every pixel integrates to unit mass and that weighting is
discarded; the two differ only by a per-pixel constant in $z$. At $K\!=\!1$
$Z(\theta)$ cancels between the PE and selection terms, so the dedicated
number-density channel `log10n0`, which enters through $Z(\theta)$, is
degenerate there and marginalizes against its prior. These states are
precomputed once per proposal by `prepare_redshift_prior_state` and read by the
traced `eval_redshift_prior_with_state` ([Performance](../guide/performance.md)).

### Completeness and the missing branch

Completeness is a matched-kernel ratio of the observed galaxy density to the
expected (homogeneous) density, clipped to $[0,1]$, and the missing-galaxy density
is the incomplete fraction of that count modulated by an LSS factor:

$$ C(p,z) = \min\!\left[\,\max\!\left(\frac{\mathrm{d}N_{\rm obs}/\mathrm{d}z} {\mathrm{d}N_{\rm exp}/\mathrm{d}z},\,0\right),\,1\right], \qquad \frac{\mathrm{d}N_{\rm miss}}{\mathrm{d}z} = \big[1 - C(p,z)\big]\,\frac{\mathrm{d}N_{\rm exp}}{\mathrm{d}z}\,Q_{\rm LSS}(p,z). $$

The default factor is the local-overdensity model
$Q_{\rm LSS} \to \max(1 + b_{\rm eff}\,\delta_g(p,z),\,0)$; optionally it is a
precomputed lognormal completion field (`darksirens.redshift.completion`,
[Catalogs](../guide/catalogs.md)).

### LSS-conditioned lognormal completion

$Q_{\rm LSS}$ is a clustered, mean-one correction built offline, so the
likelihood never samples a field. The latent log-overdensity $s$ is a Gaussian
field, the completion factor is the mean-one lognormal
$Q = \exp\!\big(b\,s - \tfrac12 b^2 \sigma_s^2\big)$, binned counts are Poisson
with rate $\lambda_v = C_v\,\mathrm{d}N_{{\rm exp},v}\,Q_v$, and the
maximum-a-posteriori field minimises

$$ \mathcal{J}(s) = \tfrac12\, s^\top C^{-1} s + \sum_v \big[\lambda_v - N_{{\rm obs},v}\ln\lambda_v\big]. $$

`--mode radial` solves this independently per HEALPix pixel along the line of
sight, with a circulant Gaussian-correlation prior diagonalised by the FFT plus
an FFT-diagonal Laplace ensemble for uncertainty. `--mode gp3d` solves one
low-rank field over all occupied $(\text{pixel}\times z)$ voxels using the
whitened $(\text{sphere}\times z)$ GP shared with the sky models: it is linear
in the $M\!\sim\!192$ whitened latents, $f = \Phi\,\xi$ with
$\Phi = k(X,Z)\,L^{-\top}$ and $L = \mathrm{chol}\,k(Z,Z)$, so the MAP is one
convex Newton solve. Both modes write the same `Q_LSS(p,z)` table
(`darksirens.redshift.lognormal_completion`), holding the Laplace posterior mean

$$ \mathbb{E}[Q](x) = \exp\!\Big(b\,f_{\rm MAP}(x) - \tfrac12 b^2\big[\mathrm{prior\_var}(x) - \mathrm{post\_var}(x)\big]\Big), $$

with $\mathrm{prior\_var}(x) = k(x,Z)K^{-1}k(Z,x)$ (the Nystrom variance) and
$\mathrm{post\_var}(x) = \phi(x)^\top H^{-1}\phi(x)$. Data-free voxels have
$\mathrm{post\_var}=\mathrm{prior\_var}$ so $Q\to 1$; near data it shrinks and
empty pixels borrow from occupied neighbours.

### Marked hosts

When the catalog carries per-galaxy marks $m_g$ (stellar mass, sSFR,
metallicity, colour) a sampled BBH-host efficiency reweights each galaxy's
contribution to the redshift prior (`darksirens.marks`):

$$ h(m_g \mid \eta) = \exp\!\Big(\textstyle\sum_k \eta_k\, \tilde m_{k,g}\Big), $$

where $\tilde m_g = m_g - \mathbb{E}[m\mid z_g]$ are redshift-centred marks, so
$\eta$ measures host preference at fixed $z$ and does not mimic
$R(z)/H_0/\gamma$. The marked observed host density is
$\sum_i w_i\,h(m_i\mid\eta)\,K_i(z)$ and the missing branch is scaled by
$\mu_{\rm miss}(z\mid\eta)=\mathbb{E}_{\rm obs}[h\mid z]$, the expected
efficiency of unobserved galaxies. At $\eta=0$ (or `--mark_model none`) this
reduces exactly to the count-based host model.

### Sky anisotropy

The rate factorises as $R(\theta, z, \hat n) = R_{\rm pop}(\theta,z)\,g(\hat n,
z)$ with a mean-one angular density $g$, so isotropy is exactly $g\equiv 1$ and
the shape does not trade off against $R_0$ (`darksirens.sky`):

| `--sky_model` | $g$ |
|---|---|
| `isotropic` | $g\equiv 1$ (bit-for-bit no-op) |
| `dipole` | $g = 1 + \hat n\cdot \mathbf{d}$ |
| `multipole`, `multipole_l3` | $g = 1 + \sum_{\ell\ge1}\sum_m a_{\ell m} Y_{\ell m}(\hat n)$, with $C_\ell = \frac{1}{2\ell+1}\sum_m a_{\ell m}^2$ |
| `sphere_gp` | log-Gaussian random field on $S^2$, $g = e^{f(\hat n)}/\langle e^f\rangle$ |
| `sphere_gp_z`, `overdensity_gp` | $(\text{sphere}\times z)$ GP with $k = a^2\,e^{-\lVert\hat n-\hat n'\rVert^2/2\ell_\Omega^2} e^{-(\zeta-\zeta')^2/2\ell_z^2}$, $\zeta=\ln(1+z)$, normalised per $z$-shell or over the comoving volume |

The factor enters the shared weight as $+\ln g(\hat n, z)$, applied identically
to the PE and selection terms, so the detector's own anisotropy divides out.

## Lensing

### Weak lensing (magnification)

For `spectral_sirens_wl` the per-event distance is magnified,
$d_L^{\rm obs} = d_L/\sqrt{\mu}$, and the PE integral is marginalised over the
magnification with a redshift-dependent PDF $p_{\rm WL}(\mu\mid z)$
(`darksirens.lensing.wlmagnification`). The lognormal backend uses $\ln\mu \sim \mathcal N(-\tfrac12 s^2, s^2)$ with
$s^2(z) = a\,z^b$; a tabulated backend integrates an external
$p(\ln\mu\mid z)$ by Gauss-Legendre. The marginalisation is gated by a static
`wl_backend` code, so all non-WL models are bit-identical.

The lognormal backend integrates in the standardised variable
$u = (\ln\mu - m(z))/s(z)$ with $m(z) = -s^2(z)/2$, for which
$p_{\rm WL}(\mu\mid z)\,\mathrm{d}\mu = \mathcal N(u;0,1)\,\mathrm{d}u$
exactly, so fixed Gauss-Hermite nodes in $u$ carry the whole WL PDF as the
integration measure: each node contributes a $+\tfrac12\ln\mu$ physics Jacobian
and no extra $+\ln\mu$ from the substitution. Because the source redshift moves
with the magnification, $z_s(\mu) = z(d_L^{\rm obs}\sqrt{\mu})$, the nodes are
drawn from the proposal $q(\mu) = p_{\rm WL}(\mu\mid z_{\rm app})$ at the
apparent redshift rather than from the target
$p_{\rm WL}(\mu\mid z_s(\mu))$, and each node carries the importance ratio

$$ \ln\frac{p_{\rm WL}(\mu\mid z_s(\mu))}{p_{\rm WL}(\mu\mid z_{\rm app})} = \ln s_{\rm app} - \ln s_s - \frac{(\ln\mu - m_s)^2}{2 s_s^2} + \frac{u^2}{2}, $$

with $s_s = s(z_s(\mu))$ and $m_s = -s_s^2/2$; the $-\ln\mu$ terms cancel
between the two lognormals and the identity
$\ln\mu = m_{\rm app} + s_{\rm app}u$ turns the proposal's exponent into
$u^2/2$. At $a=0$ both widths vanish, every node collapses to $\mu=1$ and the
ratio is identically zero, recovering the unmarginalised weight. The exponent
grows like $(u^2/2)(1 - s_{\rm app}^2/s_s^2)$, positive for $u>0$ whenever
$b>0$, so the effective integrand is super-Gaussian and node-count convergence
is not guaranteed far above the calibrated $a \approx 4\times10^{-3}$;
`validate_wl_hermite_quadrature` checks the run's own $(a,b)$ against a dense
reference at startup.

### Strong lensing (clusters)

Multiply-imaged sirens behind a galaxy cluster are modelled as a marked Cox
process over image pairs (`darksirens.likelihood.cluster_likelihood`). The
singular-isothermal-sphere lens gives an optical depth $\tau_2(z_s)$ consumed
as a probability (clipped to $[0, 1-10^{-12}]$ so one parameter point cannot
carry two incompatible probabilities across channels), a source-position PDF
$p(y)$, and image magnifications $\mu_\pm = 1 \pm 1/y$; a pair KDE captures the
joint distribution of the two images' parameters. The pair channel carries its
own $\ln\widehat\sigma^2_\mu$ into the marked-Poisson correction above.

## References

- R. Abbott et al. (LVK), GWTC population analyses; GWTC-4.0, arXiv:2508.18083;
  GWTC-5.0, arXiv:2605.27226.
- W. Farr, *Accuracy requirements for empirically measured selection
  functions*, RNAAS 3, 66 (2019), arXiv:1904.10879.
- E. Thrane & C. Talbot, PASA 36, e010 (2019).
- I. Mandel, W. Farr, J. Gair, *Extracting distribution parameters from
  multiple uncertain observations*, MNRAS 486, 1086 (2019).
- S. Vitale et al., *Inferring the properties of a population of compact
  binaries*, in *Handbook of GW Astronomy* (2022), arXiv:2007.05579.
- R. Essick & W. Farr, arXiv:2204.00461 (2022); C. Talbot & J. Golomb, MNRAS
  526, 3495 (2023), arXiv:2304.06138.
- Essick et al. (2023) and Isi, Farr & Varma (2023) on GW sky isotropy; Ray et
  al. (2023) and Edelman et al. (2023) on GP and B-spline population priors.
