# Troubleshooting

## Import errors during inference

Make sure the runtime environment includes workflow-specific packages. For example, `dynesty` is needed only when `--sampler dynesty` is selected, `emcee` is needed only for `--sampler emcee`, and `numpyro` is needed only for `--sampler numpyro`.

## HDF5 dataset not found

Check that your file follows the expected dataset names. The raw survey pixelation command requires `TARGET_RA`, `TARGET_DEC`, `Z`, `ZERR`, and `WEIGHT`. Pixelated survey files must contain `zgals`, `dzgals`, `wgals`, and `ngals`.

## Out-of-memory errors

Try one or more of the following:

- Set `--sel_batch_size` for the inference command.
- Lower `--nlive` during test runs.
- Reduce analyzer grid sizes (`--nm`, `--nq`, `--nz`, `--nchi`).
- Keep GW-population normalization grids dimension-specific (`--norm_nmass`, `--norm_nq`, `--norm_nchi`) rather than raising every grid for one narrow feature.
- Run a fixed-parameter smoke test before a full production run.

## JAX preallocation issues

The inference command sets JAX memory environment variables before importing JAX. If your cluster enforces a different policy, set the desired variables in your job script before launching the command.

## Unknown population model name

`--pop_model` names are parsed as a composition grammar. If parsing fails, the error lists the known component tokens and suggests close matches for typos (`powrlaw` → `powerlaw`). Common pitfalls:

- Bare plurals are rejected: write `powerlaw+2peaks` or `powerlaw+peak`, never `powerlaw+peaks`.
- Sharing is not part of `--pop_model`; do not append `_shared_*` suffixes. Use `--shared_beta`, `--shared_spin`, and `--shared_gamma` instead.

A name that parses but is not curated (for example `powerlaw+4peaks`) is valid: it builds with blueprint-default priors and logs an informational message rather than failing.

## DeprecationWarning about a population model name

Old spellings such as `twopowerlaws+peak` or `gwtc5_fiducial_brokenpowerlaw+2peaks` still work but emit a `DeprecationWarning` pointing at the canonical name (`2powerlaws+peak`, `gwtc5_fiducial_bpl2peaks`). Update job scripts at your convenience; saved `settings.json` files and HDF5 `pop_model` attributes that use old names remain readable.

## Unexpected parameter names

Run the inference command with a small sampler configuration and inspect the printed parameter table. Use exactly those labels in JSON overrides or fixed-value dictionaries. Population labels follow the slot-tag convention described in [Concepts → Population models](concepts.md#population-models) — in multi-component mixtures every mass parameter carries its component tag (e.g. `$\alpha_{\rm PL}$`, `$\mu_{\rm G2}$`), so the pre-grammar labels `$\alpha$` or `$\mu_1$` no longer exist for those models.
