# Electromagnetic / catalog modules (`darksirens.em`)

The `em` subpackage turns a pixelated HEALPix galaxy survey into the redshift
prior $p(z\mid\text{pix})$ consumed by the dark-siren likelihood. It also hosts
the offline builder for the LSS-conditioned lognormal completion field
$Q_{\rm LSS}$. The mathematics is derived on the
[Theory & methods](../theory.md) page; this page documents the modules that
implement it.

## `darksirens.em`

The package `__init__` exposes the shared redshift grid `zgrid` and the
`get_redshift_prior` dispatcher that maps a `--universe_model` name onto the
corresponding one-shot prior function.

```{automodule} darksirens.em
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.em.utils`

Module-level utilities shared across the subpackage: the log-spaced redshift
grid `zgrid` (1000 points, $z\in[0,5]$, finer at low $z$ where the catalog is
densest) and `load_survey`, which reads the pixelated catalog datasets
(`zgals`, `ngals`, `dzgals`, `wgals`) and the `nside` attribute. The grid is
defined once so JAX can trace through it without recompilation. Optional marked
fields are loaded by `load_survey_marks`.

```{automodule} darksirens.em.utils
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.em.volume`

The comoving-volume rate prior used by the spectral-siren model. It precomputes
a volume grid from the cosmology and returns the (log) volume prior
$\log p(z) \propto \log\!\big[\tfrac{1}{1+z}\tfrac{\mathrm{d}V_c}{\mathrm{d}z}\big]$,
the homogeneous redshift law against which catalog completeness is measured.

```{automodule} darksirens.em.volume
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.em.prior`

Assembles the universe-model redshift priors. The dark-siren per-pixel prior is

$$
p(z \mid \text{pix}) = \frac{N_{\rm obs}\, p_{\rm cat}(z\mid\text{pix})
  + \mathrm{d}N_{\rm miss}(z\mid\text{pix})}{N_{\rm obs} + N_{\rm miss}},
$$

normalised to one per pixel. The module exposes both the historical one-shot
functions (`PRIOR_REGISTRY`, keyed by `spectral_sirens`, `dark_sirens`,
`dark_sirens_complete`, `bright_sirens`, and the `spectral_sirens_wl` alias) and
the two-phase **state API** used by the JIT likelihood: `prepare_redshift_prior_state`
runs the $O(N_{\rm rows}\times N_{\rm grid})$ precomputation once per proposal
(eager, concrete catalog), and `eval_redshift_prior_with_state` is the traced,
per-sample evaluator. `DarkSirenEnsemblePriorState` plus
`eval_redshift_prior_members_with_state` provide the Bayesian completion
diagnostic $p_{\rm Bayes}(z\mid p) = \tfrac1M\sum_m p_m(z\mid p)$ over a loaded
$Q_{\rm LSS}$ ensemble.

```{automodule} darksirens.em.prior
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.em.completion`

Implements the incomplete-catalog completion. The completeness is a
matched-kernel ratio clipped to $[0,1]$, $C(p,z) = \mathrm{clip}(\mathrm{d}N_{\rm
obs}/\mathrm{d}N_{\rm exp})$, built from a cached per-pixel KDE of the observed
galaxy redshifts (`build_pixel_kde_cache`). The missing-galaxy branch is
$\mathrm{d}N_{\rm miss} = (1-C)\,\mathrm{d}N_{\rm exp}\,Q_{\rm LSS}$, where
$Q_{\rm LSS}$ is either the legacy local-overdensity factor $\max(1 + b_{\rm
eff}\delta_g, 0)$ or a precomputed lognormal table. `completion_curves` is the
eager entry; the per-row work is `jax.vmap`-traced, so the optional $Q$ table is
resolved and row-aligned eagerly by `_resolve_lss_completion_row_tables`
(handling compact vs global HEALPix indexing, the $\pm7$ $\log Q$ clip, and the
grid-size check). Scalar/vmapped diagnostics (`catalog_completion`,
`completion_clip_diagnostics`) delegate to the same hot path.

```{automodule} darksirens.em.completion
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.em.lognormal_completion`

The **offline** builder for $Q_{\rm LSS}(p,z)$ — never imported by the
likelihood. The latent log-overdensity is Gaussian and the completion factor is
the mean-one lognormal $Q = \exp(b\,s - \tfrac12 b^2\sigma_s^2)$; the MAP
minimises $\tfrac12 s^\top C^{-1}s + \sum_v[\lambda_v - N_{{\rm obs},v}\ln\lambda_v]$
with $\lambda = C\,\mathrm{d}N_{\rm exp}\,Q$.

- **Radial mode** (`poisson_lognormal_map`): an independent 1-D field per pixel
  with a circulant Gaussian-correlation power spectrum
  (`gaussian_correlation_spectrum`) diagonalised by the FFT, plus an
  FFT-diagonal Laplace ensemble (`laplace_lognormal_members`).
- **3-D angular-coupling mode** (gp3d): one low-rank field over occupied
  $(\text{pixel}\times z)$ voxels using the whitened $(\text{sphere}\times z)$
  GP. `lowrank_inducing_nodes` reproduces the sky-model node geometry,
  `build_lowrank_operator` forms $\Phi = k(X,Z)L^{-\top}$,
  `poisson_lognormal_gp3d_map` is a single convex Newton solve over the $M$
  whitened latents, `laplace_lognormal_gp3d_members` draws from the MAP Hessian,
  and `eval_logq_gp3d` evaluates the Laplace posterior mean $\mathbb{E}[Q]$
  (so data-free pixels read $Q=1$ and neighbours borrow angularly).

`save_lss_completion_hdf5` / `load_lss_completion_hdf5` define the on-disk table
contract shared by both modes (see the
[`darksirens_build_lognormal_completion`](tools.md) CLI).

```{automodule} darksirens.em.lognormal_completion
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.em.catalog`

Builds the per-pixel catalog kernel state used by the dark-siren prior: the
per-galaxy redshift kernels $K_i(z) = \mathcal N(z; z_i, \sigma_i)\,g(z)/Z_i$
(normalised to one), the per-pixel weight sums, and the marked variant
(`marked_catalog_kernel_state`) that swaps the galaxy weights for marked
host-intensity weights $w_i\,h(m_i\mid\eta)$ (see [`marks`](marks.md)).

```{automodule} darksirens.em.catalog
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.em.checks`

Numerical self-consistency checks used in tests and dry-runs: normalisation of
the volume prior, the missing-probability branch, the catalog prior, and the
$[0,1]$ bounds on $C_{\rm eff}$. `run_all_checks` integrates each density on the
package grid and asserts it matches the analytic expectation within tolerance.

```{automodule} darksirens.em.checks
:members:
:undoc-members:
:show-inheritance:
```
