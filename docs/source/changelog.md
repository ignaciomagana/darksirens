# Changelog

## Unreleased

- **`field` catalog sky-weighting is now the dark-siren default at every K.**
  The default estimand is the JOINT catalog host-density estimand: the per-pixel
  redshift-prior numerator is normalized by the survey-global `Z(theta)` instead
  of the per-pixel `Z[pix]`, so RELATIVE angular host density is preserved — a
  pixel with 100 candidate hosts now carries ~100x the angular weight of a pixel
  with one (the previous K=1 `conditional` default re-normalized every pixel to
  unit mass and silently discarded that contrast). `--catalog_sky_weighting
  field` at K=1 is no longer fatal; `conditional` stays selectable as the
  documented radial-only legacy estimand (bit-for-bit reproducible for older
  runs). At K=1 the `log10n0` number-density channel (which enters through the
  survey-global normalizer) cancels between the PE and selection terms, so
  `log10n0` is only weakly identified there and marginalizes against its prior —
  what field restores at K=1 is the angular host weighting. See
  `tests/test_field_sky_weighting_host_density.py`.
- **Multitracer unification: one likelihood path, full feature parity at any
  K.** The K=1 legacy body and the K>=2 mixture branch of
  `darksiren_log_likelihood` are unified — K=1 is a length-1 mixture with a
  static shortcut, verified **bit-exact** against pre-refactor golden values
  for the full K=1 feature matrix (`tests/test_unified_k1_golden.py`).
  Everything that worked at K=1 now composes with the K-catalog mixture:
  per-catalog `--lss_completion` Q_LSS tables, per-catalog `--use_LSS`
  overdensities (coupled to `b_miss_c{k}`), `--lss_marginalize` with one
  SHARED member index across catalogs (matched LSS realizations, equal M
  enforced), marked-host models with per-catalog `eta_<mark>_c{k}` blocks
  (a markless catalog runs `h = 1`), anisotropic `--sky_model` choices, and
  per-catalog `--validate_completion` diagnostics.  The FIELD-convention
  survey-global normalizer now carries the SAME modulated missing budget as
  the numerator (Q_LSS / delta_g / marked `mu_miss`, per ensemble member),
  so `fcat_k` stays a coherent host fraction with every mode active.
  `--catalog_sky_weighting` auto-resolves to `field` at every K (see the
  dedicated entry above) and explicit `conditional` at K>=2 stays fatal for
  `dark_sirens` (the railing z-shape-only `fcat`).  `dark_sirens_complete` keeps its pre-existing
  special-case rules (K>=2 requires field; no marks).  `--mark_model` now
  requires `--universe_model dark_sirens` at any K (other models silently
  sampled phantom flat eta dimensions).  `darksirens_analyze` reports the
  per-catalog host fractions `w_1..w_K` derived from the sampled sticks, and
  `pop_extractor` rebuilds K>=2 / Q-active parameter spaces correctly
  (`n_catalogs` / `lss_completion_active` are persisted in settings).
- **Binned Gaussian-process population models (`gppop`, `gppop_mz`).** Added the
  nonparametric binned-GP rate model of Ray et al. 2023 (arXiv:2304.08046) as
  standalone population models in `darksirens.gw.populations.gp`. The rate is
  piecewise-constant over lower-triangular `(ln m1, ln m2)` mass bins (optionally
  crossed with redshift bins) under the same whitened finite-rank GP prior as the
  existing GP models. `gppop` is mass-only with a parametric `(1+z)**(gamma-1)`
  rate; `gppop_mz` adds redshift bins carrying a free-form rate evolution. Both
  plug into the existing hierarchical likelihood and sampled cosmology with no
  framework changes; default bin edges are configurable via
  `DARKSIRENS_GPPOP_M_EDGES` / `DARKSIRENS_GPPOP_Z_EDGES`.
- Updated user-facing documentation for the current population-model grammar,
  CLI-level `--shared_beta` / `--shared_spin` / `--shared_gamma` controls,
  stick-breaking mixture weights, tagged CLI parameter labels, and the
  corrected matched-kernel catalog-completion model. The completion docs now
  clarify that `z50` and `w` are legacy compatibility parameters, while the
  likelihood uses a data-driven completeness ratio plus additive missing-galaxy
  density.
- **Population registry refactor (compositional grammar).** `--pop_model`
  names are now parsed directly into mixture compositions
  (`darksirens.gw.populations.grammar`); components self-register declarative
  blueprints (`darksirens.gw.populations.components`), so novel combinations
  such as `powerlaw+3peaks` work without code changes. Curated models keep
  their physics-tuned priors and fiducials through a pure-data table in the
  registry. Prior bounds, parameter ordering, fiducial vectors, and
  `log_p_pop` numerics for all previously registered models are unchanged
  (pinned by `tests/test_population_registry_golden.py`).
- **Breaking (names/labels only):** `twopowerlaws+peak/2peaks/3peaks` are
  renamed `2powerlaws+peak/2peaks/3peaks`, and
  `gwtc5_fiducial_brokenpowerlaw+2peaks` / `gwtc5_brokenpowerlaw+2peaks` are
  consolidated into `gwtc5_fiducial_bpl2peaks`; old spellings still resolve
  with a `DeprecationWarning`. Mass-parameter labels in multi-component
  mixtures now always carry their component tag (`$\mu_1$` →
  `$\mu_{\rm G1}$`, `$\alpha$` → `$\alpha_{\rm PL}$`); update
  `--fixed_parameter_values` / `--prior_overrides` JSON keys accordingly. See
  the migration notes in Concepts → Population models.
- Added hosted Sphinx documentation scaffolding for Read the Docs.
- Added user-guide pages for installation, quickstart, concepts, data formats, CLI usage, workflows, configuration, and troubleshooting.
- Added API reference pages for core modules.
