# Core utilities (`darksirens.utils`)

Foundational types and numerics shared across the package: the container
pytrees that move arrays through the JIT likelihood, the cosmology layer, small
interpolation and math helpers, and the plotting style.

## `darksirens.core.types`

The JAX-pytree `NamedTuple` containers that define the shape contract between
data loading and the likelihood:

- `CosmoParams` — $(H_0, \Omega_{m,0}, w_0, w_a)$ (CPL dark energy).
- `SurveyParams` — completeness/selection parameters ($n_0$, $z_{50}$, $w$,
  $\delta$, $b_{\rm miss}$, $\alpha_{\rm miss}$, $\sigma_{\rm kde}$) plus the
  **fixed, never-sampled** offline-builder hyperparameters (`lss_corr_length_mpc`,
  `lss_sigma`, `lss_corr_length_ang`) and the optional weak-lensing `wl_params`.
- `EMCatalog` — the (compact or global) galaxy catalog arrays, the precomputed
  KDE cache, the optional LSS-completion table fields, and the optional
  per-galaxy mark fields.
- `GWEvent` — the per-sample GW arrays in the canonical $(m_1^{\rm det}, q, d_L)$
  basis plus the sky unit vector $(n_x, n_y, n_z)$ and a validity mask.

Optional array fields default to `None` (an empty pytree subtree) and metadata is
encoded as integer enums so every field stays JAX-traceable.

```{automodule} darksirens.core.types
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.utils.cosmology`

The flat $w_0 w_a$CDM (CPL) background. Provides $E(z)$, the comoving and
luminosity distances, the inverse $z(d_L)$ (tabulated and interpolated for the
hot path), the comoving-volume element $\mathrm{d}V_c/\mathrm{d}z$, and the
fiducial Planck15 constants (`H0Planck`, `Om0Planck`, `w0Fiducial`,
`waFiducial`). `jax_enable_x64` is set on import so the distance integrals carry
double precision.

```{automodule} darksirens.utils.cosmology
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.utils.interp2d`

Lightweight b/multi-linear interpolation on regular grids (`interp2d`,
`interpnd`, `CartesianGrid`) used where a small tabulated function is evaluated
inside the JIT.

```{automodule} darksirens.utils.interp2d
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.utils.utils`

Small numerical helpers, notably `logdiffexp(x, y)` — a numerically stable
$\ln(e^x - e^y)$ used by the selection-variance estimator and the cluster
combiner.

```{automodule} darksirens.utils.utils
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.utils.plotting`

Publication plotting style and helpers: `set_publication_style`, the corner-plot
builder `make_production_corner`, the GP-latent summary `make_latent_summary`,
and the label classifiers (`classify_label`, `latent_indices`,
`headline_indices`) that split a parameter vector into cosmology / population /
latent groups for figures.

```{automodule} darksirens.utils.plotting
:members:
:undoc-members:
:show-inheritance:
```
