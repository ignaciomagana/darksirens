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

The default dark-siren model combines observed catalog galaxies with a missing-galaxy completion term. The current completion model is data-driven: for each catalog pixel it compares the smoothed observed galaxy redshift density to the smoothed expected density `n0 * dV_c/dz * (1 + z)^delta`. The same truncated, boundary-normalized Gaussian kernel is applied to both numerator and denominator, so a constant true completeness remains constant after smoothing, including near `z = 0`.

The inferred missing-host density is an additive count density, not a replacement probability:

```text
dN_miss/dz = (1 - C(z|pix)) * n0 * apix * dV_c/dz * (1 + z)^delta
             * max(1 + alpha_miss * b_miss * delta_g(pix,z), 0)
```

where `C(z|pix)` is the clipped matched-kernel ratio and `delta_g` is the optional observed large-scale-structure overdensity. `alpha_miss` and `b_miss` enter only through their exact product; by default the sampled amplitude is carried by `b_miss`; `alpha_miss` stays at its fiducial value of `1` unless explicitly fixed to a different value. The legacy `z50` and `w` survey parameters are retained for settings compatibility but no longer define a parametric logistic rolloff in the dark-siren completion likelihood. Use `--validate_completion true` before long runs to write clipping diagnostics for representative pixels and catch inconsistent survey-density assumptions.

Use this mode with:

```bash
--universe_model dark_sirens
```

## Population models

Population models are selected by name with `--pop_model`. For parametric
mixtures, the name itself is the mass-composition definition: `+`-separated
mass-component tokens with optional digit count prefixes. Pairing, spin, and
redshift-evolution sharing are separate CLI controls (`--shared_beta`,
`--shared_spin`, and `--shared_gamma`) and are not encoded as suffixes in
`--pop_model`:

```text
name        := composition
composition := term ("+" term)*
term        := token | <digits><plural>      # "peak", "2peaks", "3powerlaws"
```

Available grammar mass tokens are:

| Token | Plural for count prefixes | Meaning | Default mass parameters |
| --- | --- | --- | --- |
| `powerlaw` | `powerlaws` | Smoothed primary-mass power law | `alpha`, `m_min`, `m_max`, `dm_min`, `dm_max` |
| `brokenpowerlaw` | `brokenpowerlaws` | Two-slope primary-mass power law with a continuous break | `alpha1`, `alpha2`, `m_break`, `m_min`, `m_max`, `dm_min`, `dm_max` |
| `peak` | `peaks` | Gaussian primary-mass peak | `mu`, `sigma` |

Any grammar composition works without code changes: `powerlaw+3peaks` builds a
power law plus three Gaussian peaks with blueprint-default priors and fiducial
values. Curated compositions additionally carry physics-tuned per-component
priors, fiducials, and display names:

| Curated name | Display label | Notes |
| --- | --- | --- |
| `powerlaw+peak` | `PL+G` | Standard LVK-style power law plus peak. |
| `brokenpowerlaw+2peaks` | `BPL+2G` | Broken power law plus two Gaussian peaks. |
| `brokenpowerlaw+3peaks` | `BPL+3G` | Broken power law plus three Gaussian peaks. |
| `brokenpowerlaw+2peaks+powerlaw` | `BPL+2G+PL` | Adds a high-mass power-law tail. |
| `2powerlaws+peak`, `2powerlaws+2peaks`, `2powerlaws+3peaks` | `2PL+...` | Two power-law components plus one or more peaks. |

Examples:

- `powerlaw+peak` — LVK POWER LAW + PEAK
- `brokenpowerlaw+2peaks+powerlaw` — BPL + two peaks + high-mass tail
- `2powerlaws+3peaks` — five mass components; add `--shared_beta false`, `--shared_spin false`, or `--shared_gamma false` for per-component beta, spin, or redshift evolution
- `gp_mass`, `gp_mass_pairing`, `gp_mass_pairing_joint` — Gaussian-process models
- `golomb_1g`, `golomb_1g+tail`, `gwtc5_fiducial_bpl2peaks` — bespoke
  (non-mixture) models registered explicitly

### Parameter order, weights, and labels

The sampler-facing population parameter order is:

```text
v_weights -> mass components in composition order -> pairing -> spin -> gamma
```

Mixture weights are sampled as stick-breaking inputs `$v_i$`, not direct final fractions. A `k`-component mixture has `k - 1` sampled inputs in `[0, 1]`, and the final component receives the remaining stick by construction. This keeps all component weights non-negative and summing to one. If you are fixing a multi-component model by hand, convert desired final fractions with `v_i = w_i / (1 - w_1 - ... - w_{i-1})`; for two components, `v_1 = w_1`.

Mass-component labels are tagged whenever there is more than one mass slot. A single `powerlaw` uses base labels such as `$\alpha$`, while `powerlaw+peak` uses `$\alpha_{\rm PL}$`, `$m_{\min,\rm PL}$`, `$\mu_{\rm G}$`, and `$\sigma_{\rm G}$`. Repeated tokens receive 1-based indices such as `$\mu_{\rm G1}$` and `$\mu_{\rm G2}$`. Pairing, spin, and redshift-evolution parameters are shared by default, so their labels remain untagged (`$\beta$`, `$\mu_\chi$`, `$\sigma_\chi$`, `$\gamma$`) and `$\gamma$` is always last. Set `--shared_beta false`, `--shared_spin false`, or `--shared_gamma false` to sample per-component parameters tagged by mass slot (for example `$\beta_{\rm G2}$` or `$\gamma_{\rm PL}$`).

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
