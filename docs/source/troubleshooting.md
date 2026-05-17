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

## Unexpected parameter names

Run the inference command with a small sampler configuration and inspect the printed parameter table. Use exactly those labels in JSON overrides or fixed-value dictionaries.
