# Concepts

## Spectral sirens

A spectral-siren analysis uses GW data alone. The redshift prior is the cosmological comoving-volume element, while the population model captures the mass, mass-ratio, spin, and redshift distribution of compact-binary mergers.

Use this mode with:

```bash
--universe_model spectral_sirens
```

## Dark sirens with a complete catalog

A complete-catalog dark-siren analysis assumes the electromagnetic catalog traces all possible hosts in the survey volume. The redshift prior is driven by the catalog density in each sky pixel.

Empty pixels are treated explicitly. The default `--complete_empty_pixel_policy zero` is the formal complete-catalog behavior: a sky pixel with `ngals == 0` has no possible host galaxies and contributes log-prior `-inf`. This check uses the catalog's real-galaxy count or mask, not whether the evaluated catalog KDE happened to be finite.

For sparse/high-resolution pixelations, `--complete_empty_pixel_policy volume` restores the historical fallback in which genuinely empty pixels use the comoving-volume redshift prior. This mode is a robustness approximation for sampler stability and sensitivity studies, not the strict complete-catalog likelihood. Non-empty pixels always use the catalog prior; numerical underflow in `p_cat` is not reinterpreted as an empty pixel.

Use this mode with:

```bash
--universe_model dark_sirens_complete
```

## Dark sirens with an incomplete catalog

The default dark-siren model combines catalog galaxies with a missing-galaxy completion term. The completeness curve changes with redshift and is controlled by survey parameters such as `z50`, `w`, and density/evolution parameters.

Use this mode with:

```bash
--universe_model dark_sirens
```

## Population models

Population models are selected by name with `--pop_model`. The code uses a registry internally, so documented names map to callable model implementations and prior blocks. Common examples include:

- `powerlaw+peak`
- `brokenpowerlaw+2peaks`
- `gwtc5_fiducial_brokenpowerlaw+2peaks` (GWTC-5 Table 5 fiducial BBH mass model; aliases: `gwtc5_brokenpowerlaw+2peaks`, `gwtc5_fiducial_bpl2peaks`)

## Selection effects

Selection effects are handled with the gwcat GW selection file supplied by `--gwselection_path`. The inference code computes a selection correction for each proposed parameter point, optionally batching the selection calculation with `--sel_batch_size` for memory control.
