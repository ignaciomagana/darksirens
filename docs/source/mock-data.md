# Mock data generation

The repository includes a configurable mock-data workflow for exercising the dark-sirens pipeline end to end.
The defaults keep the ingestibility check tractable, while `RUN_INFERENCE=1` now runs an uncapped, production-style Dynesty configuration unless you explicitly request local-debug caps.

## Generator

Run:

```bash
python scripts/mock_data/generate_mock_data.py --outdir data/mock_dark_sirens
```

By default the generator uses a realistic local galaxy-density normalization,
`--n0 1e-3` Mpc^-3, and a low-redshift generation range, `--zmax 0.08`, so the
fixture remains lightweight.  When `--n0` is set, the complete-catalog galaxy
count is derived from the full-sky comoving-volume range and
`--galaxy-density-delta`; pass `--n-galaxies` without `--n0` to request an
explicit catalog size instead.

Survey completeness and density-evolution parameters can be overridden with
`--survey-z50`, `--survey-width`, and `--galaxy-density-delta`; these are the
values the validation runner mirrors into the fixed inference survey JSON.

Mock GW posterior widths can be controlled with fractional PE-uncertainty
arguments, for example:

```bash
python scripts/mock_data/generate_mock_data.py \
  --outdir data/mock_dark_sirens \
  --dL-fractional-uncertainty 0.20 \
  --m1det-fractional-uncertainty 0.08 \
  --m2det-fractional-uncertainty 0.10 \
  --sky-uncertainty-deg 5.0
```

If `--dL-fractional-uncertainty` or `--sky-uncertainty-deg` is omitted, that
width falls back to the SNR-scaled heuristic.

Selection injections are drawn in vectorized NumPy chunks.  `--ndraw` is the
maximum number of proposed injections, and `--selection-batch-size` only controls
the chunk size used to reach that total.  Unless you explicitly pass
`--selection-target-detections` or `--selection-per-observation-factor`, the
generator exhausts all `--ndraw` proposals so changing `NDRAW` changes the
selection sample.  The two target options are mutually exclusive, and all batch
size/count arguments must be positive.  The logs report a detected-injection
proxy `Neff`, computed
from inverse proposal-density weights, as a conservative health check; for
production-like studies increase `--ndraw` until this proxy comfortably exceeds
the inference reliability threshold (`5 * Nobs`) with margin.

The generator writes files that can be consumed directly by `darksirens_inference`:

| File | Purpose |
| --- | --- |
| `mock_galaxy_catalog_complete.h5` | Complete galaxy catalog before EM incompleteness. |
| `mock_survey_raw.h5` | Raw survey-table format accepted by `darksirens_pixelate`. |
| `catalog_pixelated_nside_<nside>.h5` | Pixelated survey catalog accepted by `--survey_path`. |
| `mock_gw_events.h5` | Mock per-event GW posterior samples accepted by `--gw_path`. |
| `mock_gw_selection.h5` | Mock detected injection/selection samples accepted by `--gwselection_path`. |

The simulation is intentionally simple:

1. Draw galaxies isotropically on the sky and uniformly in comoving volume.
2. Apply an EM survey footprint, an apparent-magnitude cut, a redshift limit, and a smooth redshift-completeness curve.
3. Draw GW hosts from the complete pre-selection catalog.
4. Draw binary masses/spins from `powerlaw+peak_shared_beta_spin`: a power-law plus Gaussian peak mass model with one shared mass-ratio beta and one shared truncated-Gaussian `chi_eff` model.
5. Apply a semi-analytic network-SNR threshold to decide GW detectability.
6. Write posterior samples around detected truth values and detected injection samples with `p_draw` weights.

## Validation shell script

Run the default ingestibility validation with:

```bash
bash scripts/mock_data/run_mock_data_test.sh
```

The script creates a data set under `data/mock_dark_sirens_test` using `N0=1e-3` Mpc^-3 and calls `darksirens.inference.data.load_all_data` to verify that the generated HDF5 products are readable by the inference pipeline.

To also launch an optional production-style sampler run with free cosmology, use:

```bash
RUN_INFERENCE=1 bash scripts/mock_data/run_mock_data_test.sh
```

You can override the mock size without editing the script, for example:

```bash
NOBS=5 NSAMP=256 NDRAW=50000 NSIDE=16 bash scripts/mock_data/run_mock_data_test.sh
```

By default the validation script does not set a detected-injection stopping
target, so it consumes `NDRAW` proposed selection injections even when
`RUN_INFERENCE=1`.  If you need a fast cap for local debugging, set either
`SELECTION_TARGET_DETECTIONS` or `SELECTION_PER_OBSERVATION_FACTOR`; those caps
intentionally make `NDRAW` an upper bound rather than the exact number of
proposals.

The validation script pins common BLAS/OpenMP thread counts to one and disables
JAX preallocation unless the caller has already set those environment variables.
This keeps the small fixture responsive on shared CPU machines and avoids the
common fork-after-JAX runtime deadlock when a library creates worker processes
after JAX has initialized its thread pool.

The optional inference run fixes only the generated survey hyperparameters via
`--fixed_parameter_values`, including `log10n0 = -3`, and explicitly leaves
cosmology free (`--fix_cosmology False`) so both `H0` and `Om0` are sampled.  It
passes `--sel_batch_size` (default `INFERENCE_SEL_BATCH_SIZE=256`) for memory
safety, uses `INFERENCE_NLIVE=1000` and `INFERENCE_DLOGZ=0.1` by default, and
does not cap Dynesty likelihood calls unless you set a positive
`INFERENCE_MAX_SAMPLES` (the default `0` disables the cap).


## Bright-siren mock data

A separate bright-siren mock workflow is available under `scripts/mock_bright_sirens` so the dark-siren mock scripts remain unchanged. It generates a complete galaxy population, applies an EM survey selection, draws GW events from galaxies with detectable counterparts, fixes the PE sky samples to the counterpart positions, and writes joint GW+EM selection injections for multi-event bright-siren inference.

```bash
bash scripts/mock_bright_sirens/run_mock_bright_sirens_test.sh
```

The runner writes `mock_bright_gw_events.h5`, `mock_bright_gw_selection.h5`, and `bright_counterparts.json` under `data/mock_bright_sirens_test` by default. Set `RUN_INFERENCE=1` to run `darksirens_inference` with `--universe_model bright_sirens` and one `--counterpart RA DEC Z` triplet per generated event.
