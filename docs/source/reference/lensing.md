# Lensing: weak magnification & strong-lensing clusters

This subsystem adds two gravitational-lensing capabilities, both opt-in and
inert by default: **weak-lensing magnification** marginalisation for the
spectral-siren PE integral (`darksirens.lensing` + `darksirens.likelihood.wl_weight`),
and a **strong-lensing cluster** likelihood for multiply-imaged sirens
(`darksirens.inference.cluster_*` + `darksirens.likelihood.likelihood_with_clusters`).
The physics is summarised on the [Theory & methods](../theory.md) page.

## `darksirens.lensing`

The package `__init__` re-exports the lensing parameter containers and factory
helpers.

```{automodule} darksirens.lensing
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.wlmagnification`

The weak-lensing magnification PDF $p(\mu\mid z)$. `WLParams` carries the backend
and its parameters. The lognormal backend models $\ln\mu\sim\mathcal N(-\tfrac12
s^2, s^2)$ with variance $s^2(z) = a\,z^b$ (so $\mathbb E[\mu]=1$); the tabulated
backend wraps an external $p(\ln\mu\mid z)$ grid (`make_tabulated_log_p_wl`).
`make_lognormal_wl_params` / `make_tabulated_wl_params` construct the containers
consumed by the likelihood.

```{automodule} darksirens.lensing.wlmagnification
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.grids`

Quadrature nodes for the lensing integrals: `make_hermite_u_grid`
(Gauss-Hermite in the standardised variable $u = (\ln\mu - m(z))/s(z)$, exact for
the lognormal), `make_log_mu_grid` (Gauss-Legendre in $\ln\mu$ for the tabulated
backend), and `make_y_grid` (source-position nodes for strong lensing).

```{automodule} darksirens.lensing.grids
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.slmarks`

Singular-isothermal-sphere (SIS) strong-lensing relations. `SISLensParams` holds
the lens model; `tau_2_SIS` / `tau_4_SIS` are the (doubly/quadruply imaged)
optical depths, `log_p_y_SIS` the source-position PDF $p(y)\propto y$, and
`mu_plus_minus_from_y` the image magnifications $\mu_\pm = 1 \pm 1/y$ (with the
inverse `y_from_mu_plus` and the time delay `delta_t_from_y`).

```{automodule} darksirens.lensing.slmarks
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.clusters`

The `ClusterSet` container and its HDF5 I/O (`make_cluster_set`,
`load_clusters`, `save_clusters`, `empty_cluster_set`) describing the
multiply-imaged systems associated with each event.

```{automodule} darksirens.lensing.clusters
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.lensing.lensed_injections`

The `LensedInjectionSet` container plus builders/loaders
(`make_lensed_injection_set`, `load_lensed_injections`, `save_lensed_injections`)
for the lensed selection injections used by the cluster selection term.

```{automodule} darksirens.lensing.lensed_injections
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.wl_weight`

The weak-lensing per-sample weights. `log_sample_weight_wl_lognormal_hermite`
marginalises the PE weight over magnification by Gauss-Hermite quadrature (the
redshift-prior Jacobian is evaluated **inside** the integral because $z$ depends
on $\mu$); `log_sample_weight_wl_or_standard` dispatches to the tabulated
quadrature or the standard (non-WL) weight. Both reduce exactly to
`log_sample_weight` as the lensing variance $\to 0$.

```{automodule} darksirens.likelihood.wl_weight
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.pair_kde`

A fixed-bandwidth Gaussian KDE over the image-pair parameters
$(m_1^{\rm det}, q, d_L^{\rm app}, \chi_{\rm eff})$ used by the cluster
likelihood. `make_pair_kde` builds a per-event KDE (Silverman diagonal
bandwidth), `stack_pair_kdes` batches them, and `log_eval_pair_kde` evaluates the
log-density inside the JIT.

```{automodule} darksirens.likelihood.pair_kde
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.cluster_likelihood`

The strong-lensing cluster likelihood for a lensed image pair: it combines the
SIS lens marks, the source-position PDF, and the pair KDE into the per-pair
contribution to the hierarchical likelihood.

```{automodule} darksirens.likelihood.cluster_likelihood
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.cluster_selection`

The cluster selection correction. `compute_cluster_selection_term` returns its
own $(\ln\mu, N_{\rm eff}, \ln\widehat\sigma^2_\mu)$ from the lensed injection
set, and `combined_selection_log_correction` merges the singleton and cluster
selection integrals (using the variance term from each) into one
Vitale-criterion-guarded correction.

```{automodule} darksirens.likelihood.cluster_selection
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.likelihood.likelihood_with_clusters`

The top-level driver `darksiren_log_likelihood_with_clusters` that splits events
into singletons and lensed pairs, evaluates each with the appropriate likelihood,
and applies the combined selection correction. It is invoked by the
[`darksirens_inference_lensing`](tools.md) CLI.

```{automodule} darksirens.likelihood.likelihood_with_clusters
:members:
:undoc-members:
:show-inheritance:
```
