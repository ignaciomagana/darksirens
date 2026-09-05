# cli (`darksirens.cli`)

The console-script entry points. Every module exposes `main()`; the two
long-running drivers also expose `console_main`, the `run_cli` wrapper the
installed scripts use. Each is runnable as `python -m darksirens.cli.<module>`,
and every flag is documented in the [CLI reference](../cli.md).

## `darksirens.cli.analyze`

`darksirens_analyze`: post-processes and plots one or more inference run
directories.

```{eval-rst}
.. automodule:: darksirens.cli.analyze
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.build_joint_lognormal_completion`

`darksirens_build_joint_lognormal_completion`: offline builder of K matched
per-survey Q_LSS ensembles from one shared latent field.

```{eval-rst}
.. automodule:: darksirens.cli.build_joint_lognormal_completion
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.build_latent_field`

Builds the latent-field anchor artifact consumed by
`darksirens_inference --lss_field_mode latent`. No console script is installed;
run it as a module.

```{eval-rst}
.. automodule:: darksirens.cli.build_latent_field
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.build_lognormal_completion`

`darksirens_build_lognormal_completion`: offline builder of the LSS-conditioned
lognormal completion file `Q_LSS(p, z)`.

```{eval-rst}
.. automodule:: darksirens.cli.build_lognormal_completion
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.common`

Shared CLI helpers: the banner, section and status printing every script uses,
and the fixed dark-energy metadata block written into the run record.

```{eval-rst}
.. automodule:: darksirens.cli.common
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.diagnose_lognormal_completion`

`darksirens_diagnose_lognormal_completion`: per-pixel diagnostic plots of a
completion file.

```{eval-rst}
.. automodule:: darksirens.cli.diagnose_lognormal_completion
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.fit_selection`

`darksirens_fit_selection`: fits the parametric magnitude selection from a
pixelated survey's magnitudes and writes the selection JSON.

```{eval-rst}
.. automodule:: darksirens.cli.fit_selection
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.inference`

`darksirens_inference`: the dark-siren, spectral-siren and bright-siren
hierarchical inference entry point.

```{eval-rst}
.. automodule:: darksirens.cli.inference
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.inference_lensing`

`darksirens_inference_lensing`: the strong-lensing branch and the sole owner of
the weak-lensing universe model.

```{eval-rst}
.. automodule:: darksirens.cli.inference_lensing
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.pixelate`

`darksirens_pixelate`: bins a raw galaxy survey into HEALPix pixels and writes
the pixelated catalog.

```{eval-rst}
.. automodule:: darksirens.cli.pixelate
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.cli.skymaps_to_samples`

`darksirens_skymaps_to_samples`: converts 3-D LVK skymaps into a GW
importance-sample HDF5.

```{eval-rst}
.. automodule:: darksirens.cli.skymaps_to_samples
   :members:
   :undoc-members:
   :show-inheritance:
```
