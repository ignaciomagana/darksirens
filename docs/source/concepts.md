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

Population models are selected by name with `--pop_model`, and the name itself
defines the mixture: `+`-separated component tokens, with optional digit count
prefixes and sharing suffixes:

```text
name        := composition [suffix]
composition := term ("+" term)*
term        := token | <digits><plural>      # "peak", "2peaks", "3powerlaws"
suffix      := _shared_beta | _shared_spin | _shared_beta_spin
```

Available mass tokens are `powerlaw`, `brokenpowerlaw`, and `peak` (a Gaussian
peak). Any composition works without code changes — `powerlaw+3peaks` builds a
power law plus three Gaussian peaks with blueprint-default priors. Curated
compositions (e.g. `powerlaw+peak`, `brokenpowerlaw+2peaks`,
`2powerlaws+peak`) additionally carry physics-tuned per-component priors and
fiducial values. Examples:

- `powerlaw+peak` — LVK POWER LAW + PEAK
- `brokenpowerlaw+2peaks+powerlaw` — BPL + two peaks + high-mass tail
- `2powerlaws+3peaks_shared_beta_spin` — five components, shared pairing/spin
- `gp_mass`, `gp_mass_pairing`, `gp_mass_pairing_joint` — Gaussian-process models
- `golomb_1g`, `golomb_1g+tail`, `gwtc5_fiducial_bpl2peaks` — bespoke
  (non-mixture) models registered explicitly

### Migration from the pre-grammar registry

Model names and parameter labels were regularized when the registry moved to
the grammar:

- `twopowerlaws+peak`, `twopowerlaws+2peaks`, and `twopowerlaws+3peaks` are
  now spelled `2powerlaws+peak`, `2powerlaws+2peaks`, `2powerlaws+3peaks`;
  `gwtc5_fiducial_brokenpowerlaw+2peaks` and `gwtc5_brokenpowerlaw+2peaks` are
  now `gwtc5_fiducial_bpl2peaks`. The old spellings still resolve (with a
  `DeprecationWarning`), so existing `settings.json` files and HDF5
  `pop_model` attributes keep working.
- Mass-component parameter labels in multi-component mixtures now always carry
  their component tag: `$\alpha$` became `$\alpha_{\rm PL}$`, `$\mu_1$` became
  `$\mu_{\rm G1}$`, `$m_{\min}$` became `$m_{\min,\rm BPL}$`, and so on. Tags
  are the component short name (`PL`, `BPL`, `G`) plus a 1-based index when a
  token appears more than once. Pairing, spin, weight (`$v_i$`), and
  `$\gamma$` labels are unchanged. **JSON keys passed via
  `--fixed_parameter_values` or `--prior_overrides` that used old mass labels
  must be updated.** Prior bounds, parameter ordering, and fiducial values are
  unchanged — only names and labels moved.

## Selection effects

Selection effects are handled with the gwcat GW selection file supplied by `--gwselection_path`. The inference code computes a selection correction for each proposed parameter point, optionally batching the selection calculation with `--sel_batch_size` for memory control.
