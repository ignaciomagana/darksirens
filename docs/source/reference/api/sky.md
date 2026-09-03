# sky (`darksirens.sky`)

Angular models for the GW source rate. A sky model supplies a mean-one angular
density `g(n)` multiplying the population rate, so isotropy (`g = 1`) is the
null and the alternatives are compared to it by evidence.

## `darksirens.sky.analyze`

Post-processing for sky-anisotropy runs: the dipole amplitude and preferred
direction summary, and the `sphere_gp` posterior sky map.

```{eval-rst}
.. automodule:: darksirens.sky.analyze
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.sky.models`

The sky-distribution models: the angular or 3-D factor `g(n)` / `g(n, z)`, each
exposing `param_specs`, `prior_bounds()` and `log_g_sky`.

```{eval-rst}
.. automodule:: darksirens.sky.models
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.sky.registry`

The public sky-model lookup API: `get_sky_model`, `sky_model_parser`,
`sky_model_prior_parser`, `sky_log_prior_volume_correction` and
`get_fixed_sky_params`.

```{eval-rst}
.. automodule:: darksirens.sky.registry
   :members:
   :undoc-members:
   :show-inheritance:
```
