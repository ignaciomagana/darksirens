# GW population models & data (`darksirens.gw`)

The `gw` subpackage defines the source-parameter population density
$p_{\rm pop}(m_1, q, \chi_{\rm eff}\mid\theta)$, the GW selection weights, and
the loaders for gravitational-wave PE and injection products. The population
mathematics is summarised on the [Theory & methods](../theory.md) page.

## `darksirens.gw`

Top-level package for gravitational-wave modelling (populations, selection, and
data loaders).

```{automodule} darksirens.gw
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.populations`

The populations package `__init__` re-exports the stable public API — the named
registry helpers (`pop_model_parser`, `pop_model_prior_parser`,
`get_fixed_population_params`, `get_model`, `register_model`) — so downstream
code imports from `darksirens.gw.populations` directly.

```{automodule} darksirens.gw.populations
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.populations.base`

The parameter-spec protocol every model exposes. `ParamSpec` carries a
parameter's name, prior bounds, LaTeX label, and prior family
(`prior_kind`/`prior_loc`/`prior_scale`); `pack_specs` collapses a list of specs
into the `(lows, highs, labels)` bound arrays. A population model is any object
exposing `param_specs`, `prior_bounds()`, and
`log_p_pop(m1, q, z, chieff, theta)` — the duck type the likelihood consumes.

```{automodule} darksirens.gw.populations.base
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.populations.components`

The building-block factors for the mass, mass-ratio, and spin distributions:
truncated power laws, Gaussian peaks, and the smooth low-/high-mass tapers. Each
component self-registers a declarative blueprint (parameter names, labels,
default bounds, fiducials) so the grammar can compose it by name.

```{automodule} darksirens.gw.populations.components
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.populations.grammar`

Parses a `--pop_model` string such as `powerlaw+peak` or
`brokenpowerlaw+2peaks` into a composition of components and assembles the
mixture generically. The `+` operator forms a stick-breaking mixture; the
assembled object normalises over the probability axes and exposes the standard
population duck type.

```{automodule} darksirens.gw.populations.grammar
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.populations.parametric`

The curated, physics-tuned parametric models (e.g. the GWTC fiducial
power-law + peak / broken-power-law + two-peaks families) with their standard
priors and fiducial parameter vectors.

```{automodule} darksirens.gw.populations.parametric
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.populations.gp`

Gaussian-process population models with a *genuine* GP prior via a whitened,
finite-rank (deterministic inducing-point) construction:

$$
K = a^2\!\prod_{\rm axis} k_{\rm axis}(Z,Z) + \epsilon I,\quad
L = \mathrm{chol}(K),\quad
\alpha = L^{-\top}\xi,\quad
f(x_*) = \mu(x_*) + k(x_*, Z)\,\alpha,
$$

with standard-normal latents $\xi$ (declared `prior_kind="normal"`) so the
sampler sees clean unit-scale geometry. The product ARD kernel
(`_build_kernel`) uses Matérn-3/2 on $\log m_1$ and RBF elsewhere; `_eval_field`
evaluates the field at the exact query points (no interpolation grid), and
`_broadcast_logp_inputs` broadcasts/flattens the query axes (supporting sparse
predictive meshes). Redshift is a **rate/conditioning** axis: the GP is
normalised only over the probability axes $\{m_1, q, \chi\}$ and the marginal
redshift evolution is the parametric $(1+z)^{\gamma-1}$ rate. `JointGPPopulation`
and `AdditiveGPPopulation` are the continuous models; `BinnedGPPopulation` is the
piecewise-constant (binned-GP) variant.

```{automodule} darksirens.gw.populations.gp
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.populations.registry`

The model registry and the stable named helpers. `register_model` records a
factory; `get_model` / `pop_model_parser` resolve a name to a model;
`pop_model_prior_parser` returns its `(lows, highs, labels, kinds, latex)`; and
`get_fixed_population_params` returns the fiducial parameter vector. The
`shared_beta` / `shared_spin` / `shared_gamma` switches control whether mixture
components share those parameters. Deprecated names emit a `DeprecationWarning`
and forward to the canonical model.

```{automodule} darksirens.gw.populations.registry
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.populations.utils`

Normalisation-grid utilities: `get_mass_grid` / `get_q_grid` / `get_chi_grid`,
the environment-configurable `configure_normalization_grids`, and the smooth
window functions `sfilter_low` / `sfilter_high` used by the component tapers.

```{automodule} darksirens.gw.populations.utils
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.selection`

GW detectability / selection helpers used when constructing or reweighting the
injection set.

```{automodule} darksirens.gw.selection
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.gw.utils`

Loaders for the gwcat-schema HDF5 products. `load_gw_samples` returns the
per-event PE arrays `(m1det, m2det, dL, chieff, ra, dec, p_pe, nEvents, nsamp)`
(flattened as `nEvents * nsamp` rows), and `load_selection_samples` returns the
injection arrays `(m1det, m2det, dL, chieff, ra, dec, p_draw, ndraw)`. Sky
angles are converted to the unit vector $\hat n$ downstream.

```{automodule} darksirens.gw.utils
:members:
:undoc-members:
:show-inheritance:
```
