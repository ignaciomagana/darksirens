# Changelog

## Unreleased

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
