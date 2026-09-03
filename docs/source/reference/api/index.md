# Python API

How the importable side of `darksirens` is organised, and the array conventions its
functions assume. See [Theory and methods](../theory.md) for the mathematics and the
[CLI reference](../cli.md) for the programs.

## Subpackages

| Subpackage | Contents |
|---|---|
| [`darksirens.core`](core.md) | Typed containers, constants, model groupings, JAX runtime setup |
| [`darksirens.gw`](gw.md) | gwcat PE and selection loading, population models, flow surrogates |
| [`darksirens.redshift`](redshift.md) | Redshift priors, completeness, LSS completion, latent field |
| [`darksirens.catalogs`](catalogs.md) | Survey catalog I/O, compaction, marks, depth maps |
| [`darksirens.inference`](inference.md) | Data staging, parameter spaces, priors, samplers, resume policy |
| [`darksirens.likelihood`](likelihood.md) | Hierarchical likelihood bodies, selection integral, lensing terms |
| [`darksirens.sky`](sky.md) | Angular source-rate models and their registry |
| [`darksirens.marks`](marks.md) | Marked-host efficiency models and their registry |
| [`darksirens.lensing`](lensing.md) | Weak-lensing and SIS strong-lensing forward model, file contracts |
| [`darksirens.io`](io.md) | Run settings and result writing |
| [`darksirens.utils`](utils.md) | Cosmology grids, interpolation, plotting |
| [`darksirens.cli`](cli.md) | The console-script entry points |

## Shape conventions

- Posterior-event arrays are flattened to `nEvents * nsamp` rows; event `i` is the
  slice `i * nsamp : (i + 1) * nsamp`. Selection arrays carry one row per injection,
  weighted by the draw density in the selection file.
- Catalog arrays are indexed by HEALPix pixel; compact views replace repeated
  sample pixels with `unique_pixels` plus `sample_to_unique_idx`, so one
  `EMCatalog` row is one unique inference pixel. Per-galaxy datasets are
  padded to `(N_catalog_rows, N_max_gals)`, with `ngals` real entries per row.
- Arrays crossing the likelihood boundary must be convertible to JAX arrays;
  loaders accept NumPy and convert before the JIT boundary.

## Stability notes

The stable import points are the registry lookups (`darksirens.gw.populations.get_model`,
`pop_model_parser`, `pop_model_prior_parser`, `get_fixed_population_params`,
`darksirens.sky.get_sky_model`, `darksirens.marks.get_mark_model`), the containers in
`darksirens.core.types`, and the loaders (`darksirens.gw.samples`, `darksirens.catalogs.io`,
`darksirens.redshift.prior.get_redshift_prior`). Names beginning with an underscore are
internal helpers: not shown here, and changed by memory-layout and performance work.

```{toctree}
:maxdepth: 1

core
gw
redshift
catalogs
inference
likelihood
sky
marks
lensing
io
utils
cli
```
