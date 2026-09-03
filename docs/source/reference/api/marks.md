# marks (`darksirens.marks`)

The marked-host model: per-galaxy marks (log stellar mass, sSFR, metallicity,
colour) and a sampled BBH-host efficiency `h(m | eta)` that reweights the
catalog's contribution to the dark-siren redshift prior. Mirrors
`darksirens.sky`.

## `darksirens.marks.models`

The efficiency models themselves, including the null and loglinear forms, each
exposing `param_specs`, `prior_bounds()` and a per-galaxy `log_h` over the
z-centred mark fields.

```{eval-rst}
.. automodule:: darksirens.marks.models
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.marks.registry`

The public mark-model lookup API; `none` is the default null model with
`h = 1`, no sampled parameters and the plain galaxy-count host weighting.

```{eval-rst}
.. automodule:: darksirens.marks.registry
   :members:
   :undoc-members:
   :show-inheritance:
```
