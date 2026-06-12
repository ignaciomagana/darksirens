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

The default dark-siren model combines catalog galaxies with a missing-galaxy completion term, additively in galaxy number densities (the in/out-of-catalog decomposition of Gray et al. 2020, arXiv:1908.06050; Gair et al. 2023, AJ 166, 22):

```
p(z | pix) = [ N_obs(pix) * p_cat(z | pix) + dN_miss(z | pix) ] / [ N_obs(pix) + N_miss(pix) ]
```

Here `p_cat` is the normalised weighted-kernel catalog shape (each galaxy contributes a unit-mass volumetric photo-z posterior), `N_obs` is the observed galaxy count in the pixel, and `dN_miss = (1 - C(z)) * dN_exp(z) * max(1 + b_miss * delta_g, 0)` is the missing-galaxy density. Completeness `C(z)` is data-driven: the ratio of the boundary-corrected observed KDE to the expected counts `n0 * apix * dV/dz * (1+z)^delta`, both smoothed by the same matched kernel. There is no parametric roll-off (`z50`, `w` are inactive), and the prior integrates to 1 per pixel by construction. The catalog-vs-missing odds are therefore the count odds `N_obs : N_miss`, controlled by the sampled density normalisation `n0`.

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
