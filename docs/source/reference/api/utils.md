# utils (`darksirens.utils`)

Numerical helpers shared across the package. Importing
`darksirens.utils.cosmology` builds interpolation grids, so the package
`__init__` deliberately re-exports nothing; import the submodules explicitly.

## `darksirens.utils.cosmology`

JAX-compatible flat-CPL distance and volume utilities, built on an import-time
interpolation table of comoving distance over `(z, Om0, w0, wa)` that runtime
functions rescale by `H0`.

```{eval-rst}
.. automodule:: darksirens.utils.cosmology
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.utils.interp2d`

The JAX multilinear interpolation helpers `interp2d`, `interpnd`,
`interpnd_scalar_head` and `CartesianGrid` used by the cosmology and prior
grids.

```{eval-rst}
.. automodule:: darksirens.utils.interp2d
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.utils.plotting`

Shared plotting helpers: the style entry point, a parameter-label classifier, a
high-dimensional-aware corner plot, and a compact Gaussian-process latent
summary.

```{eval-rst}
.. automodule:: darksirens.utils.plotting
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.utils.utils`

`logdiffexp` and `logsumexp_neginf_safe`, the log-space reductions used where a
`-inf` floor must survive differentiation.

```{eval-rst}
.. automodule:: darksirens.utils.utils
   :members:
   :undoc-members:
   :show-inheritance:
```
