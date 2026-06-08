# Configuration and parameters

## Boolean values

Boolean command-line options accept common true/false strings such as `true`, `false`, `1`, `0`, `yes`, and `no`.

## JSON options

`--prior_overrides` and `--fixed_parameter_values` must be JSON objects.

Prior override example:

```bash
--prior_overrides '{"H0": [60.0, 80.0], "Om0": [0.2, 0.4], "w0": [-1.2, -0.8], "wa": [-0.5, 0.5]}'
```

Fixed-parameter example:

```bash
--fixed_parameter_values '{"H0": 67.74, "Om0": 0.3075, "w0": -1.0, "wa": 0.0}'
```

Parameter labels must match the labels produced by the selected cosmology, population, and survey blocks. The inference command prints a parameter table at startup showing sampled, fixed, and overridden parameters.


## Bright-siren counterparts

For `--universe_model bright_sirens`, pass the electromagnetic counterpart as event metadata rather than as `--survey_path`:

```bash
--universe_model bright_sirens --counterpart RA1 DEC1 Z1 [RA2 DEC2 Z2 ...]
```

`RA` and `DEC` are in radians. Provide one triplet per GW event, in the same event order as the posterior samples in `--gw_path`. The inference loader turns the counterparts into a fixed synthetic catalog at `--counterpart_nside` with redshift width `--counterpart_dz`, so the survey/completion parameter block is fixed automatically for this model. Selection samples are still loaded from `--gwselection_path` in the standard way and should encode the joint GW+EM detection process for bright-siren analyses.

## Cosmology block

The standard cosmology block is a flat CPL dark-energy model and includes:

- `H0`: Hubble constant.
- `Om0`: matter density fraction.
- `w0`: present-day CPL dark-energy equation-of-state parameter.
- `wa`: CPL dark-energy evolution parameter.

The fiducial dark-energy values `w0=-1` and `wa=0` reproduce flat ΛCDM. Use `--fix_cosmology true` to remove all four cosmology labels from sampling, or `--fix_de true` to fix only `w0` and `wa` while still sampling `H0` and `Om0`. Individual cosmology labels can also be pinned with `--fixed_parameter_values`, for example `--fixed_parameter_values '{"w0": -1.0, "wa": 0.0}'`, and their prior ranges can be narrowed with `--prior_overrides`.

## Survey block

The dark-siren incompleteness model uses survey/completion parameters with these units and default prior ranges:

| Parameter | Meaning and units | Default prior | Recommended use |
| --- | --- | --- | --- |
| `log10n0` | Base-10 logarithm of the comoving galaxy number density `n0` in `Mpc^-3`. The completion model multiplies `n0` by the HEALPix pixel solid angle and `dV_c/dz` in `Mpc^3 sr^-1 dz^-1`. | `[-4, -1]` | Keep near measured catalog densities; override explicitly for unusual luminosity cuts. |
| `z50` | Redshift where the logistic survey rolloff is 50% complete. The completion grid covers `0 <= z <= 5`. | `[0.05, 4.5]` | Use a catalog-depth estimate when available. Avoid values at or beyond the grid edge. |
| `w` | Logistic rolloff width in redshift units. | `[0.02, 1.5]` | Use narrower ranges for surveys with a well-characterized depth transition. |
| `delta` | Power-law evolution of expected galaxy density, `n(z) = n0 (1+z)^delta`. | `[-3, 3]` | Broaden only with a catalog-specific justification. Merger-rate evolution is handled separately. |
| `b_miss` | Bias amplitude for the LSS-modulated missing-galaxy density. Dimensionless. | `[0, 3]` | Fix to `1` or narrow around it unless testing LSS systematics. |
| `alpha_miss` | Mixture between isotropic and LSS-modulated missing density. Dimensionless; `0` is isotropic, `1` is fully LSS-modulated. | `[0, 1]` | Use the full range for model uncertainty, or fix to `0` to disable LSS modulation. |

The default survey priors are intentionally narrower than earlier broad exploratory bounds, because extremely large density or evolution ranges can make `C_iso`, `C_eff`, or `rho_miss_eff` clip over much of the redshift grid. If a fit truly requires broader bounds, pass explicit `--prior_overrides` for the affected survey labels and record the catalog-density units used to justify them.

To validate a catalog/survey configuration without starting a sampler, run:

```bash
--validate_completion true --completion_validation_pixels 64
```

This dry run loads the survey, computes clipping fractions for `C_iso`, `C_eff`, and `rho_miss_eff` on the shared redshift grid, writes `completion_validation__*.json` under `--save_path`, and exits before likelihood construction.

## Population block

Population parameters depend on `--pop_model`. Use a small dry run to print the parameter table before committing compute time to a production job.

## Normalization-grid tuning

GW-population mass, mass-ratio, and spin components are normalized on cached trapezoid grids. The defaults (`--norm_nmass 500 --norm_nq 200 --norm_nchi 200`) are intended for development and moderate analyses. You can change individual dimensions from the command line or with the environment variables `DARKSIRENS_GW_N_MASS`, `DARKSIRENS_GW_N_Q`, and `DARKSIRENS_GW_N_CHI`; the active values are printed at startup and saved in `settings.json` as `normalization_grid`.

For production 500-event analyses, especially when priors allow minimum smoothing widths such as `\delta m_{\min}=0.01` or `\sigma_\chi=0.01`, use at least:

- `--norm_nmass 2000` for power-law and broken-power-law mass edges.
- `--norm_nq 1000` for mass-ratio normalizations with low-mass systems near the secondary-mass cutoff.
- `--norm_nchi 1000` for narrow effective-spin components.

If only one distribution has narrow features, increase only the corresponding grid rather than all three dimensions. For final evidence runs or sensitivity checks, compare against a higher-resolution rerun such as `--norm_nmass 5000 --norm_nq 3000 --norm_nchi 3000` and confirm posterior and evidence changes are negligible for the science target.

## Performance tuning

- Increase `--nlive` for more reliable nested-sampling evidences in high dimensions.
- Set `--sel_batch_size` if the injection file is too large to process at once.
- Reduce posterior-predictive grid sizes (`--nm`, `--nq`, `--nz`, `--nchi`) during analyzer smoke tests.
