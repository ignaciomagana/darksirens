# Data formats

The command-line tools exchange HDF5 files. This page documents the expected top-level structure used by the loaders and writers.

## GW posterior samples

`--gw_path` must point to a gwcat HDF5 export with `format_version="gwcat-1.0"`. Generate this file with `gwcat.GWCatalog.to_darksirens(...)` (the frozen v1 path, which takes no spin-basis argument), or with `GWCatalog.export(path, spin_basis="chieff")` if you want the versioned export — `export()` with no `spin_basis` produces a `component`-basis file, which darksirens rejects; darksirens no longer ingests raw PE files or performs catalog-specific coordinate conversions in `darksirens.gw.utils`.

The export must contain the datasets consumed by the likelihood: `ra`, `dec`, `m1det`, `m2det`, `dL`, `chieff`, `p_pe`, `m1src`, and `m2src`. It must also contain the attributes `nsamp`, `nobs`, `pe_cosmology_H0`, `pe_cosmology_Om0`, `chi_eff_in_p_pe`, and `chi_eff_amax`.

Before running inference, you can use `gwcat.validate_export(gw_path, selection_path)` to check that the posterior and selection files are mutually compatible.

## GW selection samples

`--gwselection_path` must point to a gwcat HDF5 export with `format_version="gwcat-selection-1.0"`. Generate it with `gwcat.SelectionSet.to_darksirens(...)` or `gwcat.CombinedSelectionSet.to_darksirens(...)`; darksirens no longer reads raw LVK injection files directly. The selection loader reads gwcat-preprocessed detected injections and their physical draw densities, then the likelihood computes an expected-detection correction.

For large injection sets, use:

```bash
--sel_batch_size 200000
```

Tune the value to fit your memory budget.

## Raw survey file for pixelation

`darksirens_pixelate` expects the input survey HDF5 file to contain these datasets:

| Dataset | Meaning | Units |
| --- | --- | --- |
| `TARGET_RA` | Right ascension | degrees |
| `TARGET_DEC` | Declination | degrees |
| `Z` | Redshift | dimensionless |
| `ZERR` | Redshift uncertainty | dimensionless |
| `WEIGHT` | Galaxy weight | arbitrary/non-negative |

## Pixelated survey output

The pixelation command writes `catalog_pixelated_nside_<nside>.h5` with:

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `zgals` | `(npix, max_galaxies_per_pixel)` | Galaxy redshifts per HEALPix pixel, padded with `100.0` |
| `dzgals` | `(npix, max_galaxies_per_pixel)` | Redshift uncertainties, padded with `1.0` |
| `wgals` | `(npix, max_galaxies_per_pixel)` | Galaxy weights, padded with `0.0` |
| `ngals` | `(npix,)` | Number of real galaxies in each pixel |

The file also stores `nside` as an HDF5 attribute.
