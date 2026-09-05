# Changelog

## Unreleased

### Performance

- One in-repo benchmark, `scripts/benchmarks/bench_likelihood_call.py`, times
  a likelihood call end to end and per component and compares values between
  two builds.
- The per-sample catalog KDE window is sized from the bound catalog (never
  truncating within the accuracy contract), located with one binary search, and
  evaluated with three gathers per galaxy instead of four. Selection
  injections are sorted by pixel at build time.
- Population-only runs (cosmology and survey fixed) evaluate the redshift
  prior once at build time (`--freeze_redshift_prior`, on by default when
  admissible; verified in the graph on every call).
- Lensing: under `--partition_mode marginalize_exact` the cluster master
  likelihood is evaluated once per proposal and every partition is assembled
  from per-event and per-edge terms.

### Correctness

- Fixed `W = 1024` catalog window truncated the in-range galaxy block on
  photo-z-dense rows (measured 0.17 nats mean on a DESI-like mixed catalog).
- `sigma_kde` bound for the window was read from a coordinate placeholder when
  the label was sampled; the row-length cap of the one-sided window rule was
  unsound on ragged catalogs.
- Lensing: partition enumeration no longer recurses once per edge; the Hermite
  weak-lensing validator probes low redshift and refuses a negative `wl_a`;
  time marks at or beyond the observing window are refused; the pair branches
  normalise by valid PE rows; pinned lens-rate values under
  `--fix_lens_rate true` are honoured; a startup check reports the pair
  y-quadrature error.

### Documentation

- README and documentation rewritten around three entry points (getting
  started, user guide, reference) with every option verified against the
  command-line parsers.

## Earlier

- `field` catalog sky-weighting is the dark-siren default at every K; the
  per-pixel redshift-prior numerator is normalised by the survey-global
  `Z(theta)` so relative angular host density is preserved. `conditional` stays
  selectable as the radial-only legacy estimand.
- One likelihood path for K = 1 and K >= 2 catalogs, bit-exact against the K = 1
  golden values (`tests/test_unified_k1_golden.py`); every single-catalog mode
  composes with the K-catalog mixture.
- Binned Gaussian-process population models `gppop` and `gppop_mz`
  (Ray et al. 2023).
- Compositional `--pop_model` grammar with self-registering components and
  curated priors for standard names; `twopowerlaws+...` renamed to
  `2powerlaws+...`, `gwtc5_fiducial_brokenpowerlaw+2peaks` consolidated into
  `gwtc5_fiducial_bpl2peaks` (old spellings resolve with a
  `DeprecationWarning`); component-tagged mass labels (`$\alpha_{\rm PL}$`).
- Sphinx documentation on Read the Docs.
