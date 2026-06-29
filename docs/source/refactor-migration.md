# Refactor migration guide

The package is being migrated incrementally to a clearer internal layout. This is a structure-only transition: scientific behavior, command names, CLI flags, output filenames, settings schema, and HDF5 datasets/attributes are not intentionally changing.

## Preferred new imports

Use the new package locations for new internal code:

- Core container types and constants: `darksirens.core.types`, `darksirens.core.constants`, and `darksirens.core.jax_config`.
- Redshift-prior, volume, and completion code: `darksirens.redshift.prior`, `darksirens.redshift.volume`, `darksirens.redshift.completion`, and `darksirens.redshift.lognormal_completion`.
- Likelihood implementation code: `darksirens.likelihood.factory`, `darksirens.likelihood.core`, `darksirens.likelihood.selection`, `darksirens.likelihood.cluster_selection`, `darksirens.likelihood.cluster_likelihood`, and `darksirens.likelihood.likelihood_with_clusters`.
- Command implementation modules: `darksirens.cli.*`.
- Catalog helpers: `darksirens.catalogs.*`, including `darksirens.catalogs.io` for survey-loading helpers.

## Temporary compatibility imports

The old import paths still work temporarily for downstream users and compatibility tests. They should be treated as compatibility paths, not preferred locations for new internal code.

- `darksirens.utils.containers` remains as a wrapper for `darksirens.core.types`.
- `darksirens.em` is being replaced by `darksirens.redshift` for redshift-prior and completion code.
- `darksirens.inference.likelihood*` is being replaced by `darksirens.likelihood.*`.
- `darksirens.tool` is now compatibility-only; command implementations live under `darksirens.cli`.

No CLI command names changed as part of this migration.
