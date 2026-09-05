# Input files

This page lists every file the command-line tools read, with the datasets and
attributes each one must contain. All HDF5 requirements below are enforced at
load time, so a file that is missing a member fails immediately with a named
error rather than part-way through sampling.

## GW posterior samples (`--gw_path`)

A `gwcat` HDF5 export. Accepted `format_version` values:
`gwcat-1.0`, `gwcat-pe-2.0`, `gwcat-pe-2.1` (a `2.x` file must also carry
`spin_basis="chieff"`), and `observed-lensing-pe-1.0` for the lensing CLI.
Every dataset is 1-D with length `nobs * nsamp`; a shorter column is refused
because it would broadcast silently over the samples.

| Dataset | Shape | Units / type | Meaning |
| --- | --- | --- | --- |
| `ra` | `(nobs*nsamp,)` | radians, `[0, 2*pi)` | Right ascension per sample |
| `dec` | `(nobs*nsamp,)` | radians, `[-pi/2, pi/2]` | Declination per sample |
| `m1det`, `m2det` | `(nobs*nsamp,)` | solar masses, `> 0` | Detector-frame component masses, `m2det <= m1det` |
| `m1src`, `m2src` | `(nobs*nsamp,)` | solar masses, `> 0` | Source-frame component masses, `m2src <= m1src` |
| `dL` | `(nobs*nsamp,)` | Mpc, `> 0` | Luminosity distance |
| `chieff` | `(nobs*nsamp,)` | dimensionless, `[-1, 1]` | Effective aligned spin |
| `p_pe` | `(nobs*nsamp,)` | density, `>= 0` | PE proposal density in the canonical `(m1det, q, dL)` basis with `q = m2det / m1det` |

Required attributes: `nsamp`, `nobs` (both positive), `pe_cosmology_H0`,
`pe_cosmology_Om0`, `chi_eff_in_p_pe`, `chi_eff_amax`. A file that declares
`sky_position_available` with any `False` entry is refused, because those
campaigns carry NaN sky placeholders.

Produced by `gwcat`: `gwcat.GWCatalog.to_darksirens(...)`, or
`GWCatalog.export(path, spin_basis="chieff")`. `export()` with no
`spin_basis` writes a `component`-basis file, which the chi_eff likelihood
rejects. The mock generator in `scripts/mock_dark_sirens` writes a
`gwcat-1.0` file directly.

## GW selection injections (`--gwselection_path`)

A `gwcat` selection export whose `format_version` is one of
`gwcat-selection-1.0`, `gwcat-selection-2.0`, `gwcat-selection-2.1` (a `2.x`
file must carry `spin_basis="chieff"`). Every dataset is 1-D and all of them
share one length: the number of *detected* injections.

| Dataset | Shape | Units / type | Meaning |
| --- | --- | --- | --- |
| `m1det`, `m2det` | `(ndet,)` | solar masses, `> 0` | Detector-frame masses, `m2det <= m1det` |
| `m1src`, `m2src` | `(ndet,)` | solar masses, `> 0` | Source-frame masses |
| `dL` | `(ndet,)` | Mpc, `> 0` | Luminosity distance |
| `chieff` | `(ndet,)` | dimensionless, `[-1, 1]` | Effective aligned spin |
| `ra`, `dec` | `(ndet,)` | radians | Sky position |
| `pdraw` | `(ndet,)` | density, `> 0` | Physical injection draw density in the canonical `(m1det, q, dL)` basis, absolute scale retained |

Required attributes: `ndraw` (total proposed injections, at least the number
of detected rows) and `chi_eff_swap_applied` (whether `pdraw` already carries
the 1-D chi_eff draw density). If the file stamps
`injected_spin_uniform_isotropic` with a `False` entry the analytic chi_eff
swap is invalid and the load is refused unless
`--allow_invalid_spin_swap` is passed.

Produced by `gwcat.SelectionSet.to_darksirens(...)` or
`gwcat.CombinedSelectionSet.to_darksirens(...)`; the mock generator writes a
`gwcat-selection-1.0` file directly.

## Raw galaxy survey (`darksirens_pixelate --survey_path`)

A flat, table-like HDF5 file: one 1-D dataset per column, one row per galaxy.
Marks and galaxy properties are optional and are only read when the column is
present.

| Dataset | Shape | Units / type | Meaning |
| --- | --- | --- | --- |
| `TARGET_RA` | `(ngal,)` | degrees | Right ascension (converted to radians by the tool) |
| `TARGET_DEC` | `(ngal,)` | degrees | Declination |
| `Z` | `(ngal,)` | dimensionless | Observed redshift; must be finite, `> 0` and `< 100` (100 is the padding sentinel) |
| `ZERR` | `(ngal,)` | dimensionless | Redshift uncertainty; finite and `>= 0` |
| `WEIGHT` | `(ngal,)` | non-negative float | Galaxy weight; finite and strictly `> 0` |
| `LOGMSTAR` | `(ngal,)` | float, optional | Mark, stored as `mark_logmstar` |
| `LOGSSFR` | `(ngal,)` | float, optional | Mark, stored as `mark_logssfr` |
| `LOGZ` | `(ngal,)` | float, optional | Mark, stored as `mark_metallicity` |
| `GR_COLOR` | `(ngal,)` | float, optional | Mark, stored as `mark_color` |
| `APP_MAG` | `(ngal,)` | magnitudes, optional | Galaxy property, stored as `gal_app_mag` |
| `STRATUM` | `(ngal,)` | integer label, optional | Galaxy property, stored as `gal_stratum` |

Every mark and property value must be finite. Produced by your own survey
preparation; `scripts/mock_dark_sirens/generate_mock_data.py` writes one as
`mock_survey_raw.h5`.

## Pixelated catalog (`--survey_path`)

Written by `darksirens_pixelate` as
`catalog_pixelated_nside_<nside>.h5`, and read by
`darksirens.catalogs.io.load_survey`. Per-galaxy arrays are dense
`(npix, maxgals)` tables where `maxgals` is the largest pixel occupancy;
`ngals` gives the number of real galaxies per pixel and is the only valid
real-slot mask.

| Dataset / attribute | Shape | Units / type | Meaning |
| --- | --- | --- | --- |
| `zgals` | `(npix, maxgals)` | float64 | Galaxy redshifts; empty slots padded with `100.0` |
| `dzgals` | `(npix, maxgals)` | float64 | Redshift uncertainties; padded with `1.0` |
| `wgals` | `(npix, maxgals)` | float64 | Galaxy weights; padded with `0.0` |
| `ngals` | `(npix,)` | integer | Real galaxies in each HEALPix (RING) pixel |
| `mark_logmstar`, `mark_logssfr`, `mark_metallicity`, `mark_color` | `(npix, maxgals)` | float, optional | Per-galaxy marks; padded with `0.0` |
| `gal_app_mag`, `gal_stratum` | `(npix, maxgals)` | float, optional | Offline-only galaxy properties; padded with `0.0` |
| `nside` (attr) | scalar | integer | HEALPix NSIDE |
| `z_depth` (attr) | scalar | float, optional | Written by `--z_depth`; read as a completeness prior (completeness is zero above it) |

`load_survey` sorts each row's real-galaxy prefix by ascending redshift before
use; the mark and property loaders re-derive the same permutation, so all
per-galaxy tables stay co-indexed.

## Flow surrogate directory (`--gw_flows_path`)

A directory of per-event normalizing-flow checkpoints replacing stored PE
samples (`spectral_sirens` only, and mutually exclusive with `--gw_path`).

| Path | Contents | Meaning |
| --- | --- | --- |
| `<EVENT>/<EVENT>_flow.npz` | `arr_0 .. arr_N` plus a `config_json` string | The flow's array leaves in `eqx.partition` order, and the config needed to rebuild the skeleton |

The default glob is `*/*_flow.npz` (`--flows_pattern`). Checkpoints are read
with `allow_pickle=False` and structurally checked against the installed
`flowjax`; `--flows_on_mismatch` selects the policy on version drift. The
spectral-siren flows use columns
`("mass_1", "mass_2", "luminosity_distance", "chi_eff")`: detector-frame
masses in solar masses, `dL` in Mpc, dimensionless `chi_eff`. Trained
externally, one flow per event.

## LSS completion tables (`--lss_completion`)

Written by `darksirens_build_lognormal_completion` (and the joint builder) as
a single group `/lss_completion`.

| Dataset / attribute | Shape | Meaning |
| --- | --- | --- |
| `logq_map` | `(n_rows, n_grid)` | MAP log-Q table over pixel rows and the redshift grid |
| `logq_members` | `(n_members, n_rows, n_grid)` | Optional Laplace ensemble of log-Q realizations |
| `zgrid` | `(n_grid,)` | Redshift grid the tables are defined on |
| `indexing` (attr) | scalar | `"compact"` (catalog rows) or `"global"` (full-sky pixels) |
| `model`, `completion_kind`, `created_by`, `realization_set_id` (attrs) | scalar | Provenance; the realization id lets a K>=2 mixture verify matched ensembles |
| `c_mode`, `f_p_aware`, `q_support_depth`, `budget_renormalized`, `budget_monopole_logq` (attrs) | scalar / `(n_grid,)` | Which completeness estimator the fit was residual to, and how the budget was normalized; checked against the consuming run |
| `n_members`, `member_content_sha256`, `diagnostics` (attrs) | scalar | Ensemble size, content digest, JSON build diagnostics |

A file with any non-finite entry is never written. See
[Galaxy catalogs](../guide/catalogs.md) for how these tables enter the
missing-host budget.

## Lensing inputs

The lensing entry point `darksirens_inference_lensing` reads the same GW PE
and selection files as above, plus the files below. In the simulated study
they are produced by an external lensing simulation pipeline that is not
shipped in this repository. Usage is described in
[Lensing](../guide/lensing.md).

| File (flag) | Format | Required contents |
| --- | --- | --- |
| Observed catalog JSON (`--observed_catalog_path`) | `observed-lensing-catalog-1.0` | `event_indexing="global"`, `n_events`, and an `events` list of length `n_events`; each entry has a contiguous `event_index` (0..n_events-1), a unique non-empty `event_id`, and an optional finite `gps_time` |
| Pair metadata HDF5 (`--pair_metadata_path`) | HDF5 | attr `npairs`, then one `pair_<k>` group per pair carrying `event_index_image0` / `event_index_image1`, and, for `--pair_marks time`, `delta_t_obs` and `sigma_delta_t` (seconds) |
| Lensed injections (`--lensed_injections_path`) | HDF5 | Per-image 1-D datasets `source_id`, `image_id` (0 = mu_+, 1 = mu_-), `m1_src`, `q_src`, `z_src`, `chieff`, `y_source`, `mu`, `detected` (bool), `p_prop_src`, `p_prop_y`, all length `N_img`, plus attr `n_draw_sources`; optional `log_p_tag_per_source` / `p_tag_per_source`, `snr_image0`, `snr_image1`, `delta_t_obs`, `true_delta_t`, `log_sky_overlap`, `p_tag_true`, `tagged_pair` |
| Candidate pairs JSON (`--candidate_pairs_path`) | `candidate-pairs-1.0` | `n_events` and a `pairs` list; each pair has `i`, `j` (distinct, in range) and a finite `log_prior_odds`, with optional `marks` and a `label` that inference ignores. Required by `--partition_mode marginalize_exact` |
| Partition JSON (`--partition_path`) | JSON | `singleton_indices`, `pair_indices`, `n_singletons`, `n_pairs`. Required by `--partition_mode fixed` |

The observed-mode GW PE file additionally carries
`format_version="observed-lensing-pe-1.0"`, `event_indexing="global"`,
`n_events` and `nsamp`, and should carry `observed_catalog_path` /
`observed_catalog_sha256` for provenance. Selection inputs may also be
supplied as one consolidated `lensing-selection-inputs-1.0` file with
`unlensed` and/or `lensed` groups.
