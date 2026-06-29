# Refactor migration guide

The package is being migrated incrementally to a clearer internal layout. This is a structure-only transition: scientific behavior, command names, CLI flags, output filenames, settings schema, and HDF5 datasets/attributes are not intentionally changing.

## Preferred new imports

Use the new package locations for new internal code:

- Core container types and constants: `darksirens.core.types`, `darksirens.core.constants`, and `darksirens.core.jax_config`.
- Redshift-prior, volume, and completion code: `darksirens.redshift.prior`, `darksirens.redshift.volume`, `darksirens.redshift.completion`, and `darksirens.redshift.lognormal_completion`.
- Likelihood implementation code: `darksirens.likelihood.factory`, `darksirens.likelihood.core`, `darksirens.likelihood.selection`, `darksirens.likelihood.cluster_selection`, `darksirens.likelihood.cluster_likelihood`, and `darksirens.likelihood.likelihood_with_clusters`.
- Command implementation modules: `darksirens.cli.*`.
- Catalog helpers: `darksirens.catalogs.*`, including `darksirens.catalogs.io` for survey-loading helpers.

## Breaking import-path cleanup

Compatibility wrappers for the staged package-layout refactor have been removed. Update downstream code to the new import paths below. Console command names did not change.

| Old import path | New import path | CLI command name changed? |
| --- | --- | --- |
| `darksirens.utils.containers` | `darksirens.core.types` | No |
| `darksirens.em.prior` | `darksirens.redshift.prior` | No |
| `darksirens.em.completion` | `darksirens.redshift.completion` | No |
| `darksirens.em.lognormal_completion` | `darksirens.redshift.lognormal_completion` | No |
| `darksirens.em.volume` | `darksirens.redshift.volume` | No |
| `darksirens.em.catalog` | `darksirens.redshift.catalog` | No |
| `darksirens.em.utils` | `darksirens.catalogs.io` | No |
| `darksirens.inference.likelihood` | `darksirens.likelihood.factory` | No |
| `darksirens.inference.likelihood_core` | `darksirens.likelihood.core` | No |
| `darksirens.inference.*` likelihood helper modules | `darksirens.likelihood.*` | No |
| `darksirens.tool.darksirens_*` | `darksirens.cli.*` | No |

No CLI command names changed as part of this migration.
