# Changelog

## Unreleased

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
