# API reference

The public Python API is organized by scientific task.  Most users interact with
`darksirens` through the command-line programs, but the modules below are also
intended to be imported by tests, notebooks, and custom inference drivers.

## How to read the reference

- **Container types** in `darksirens.utils.containers` are lightweight named
  tuples that move JAX arrays through the likelihood without Python object
  mutation.  They define the shape contract between data loading, catalog view
  preparation, and the JIT-compiled likelihood core.
- **Population components** in `darksirens.gw.populations` expose three layers:
  component classes describe primary-mass, mass-ratio, and spin factors and
  self-register declarative blueprints (parameter names, labels, default
  bounds, and fiducials) in `components`; the `grammar` module parses
  `--pop_model` names into compositions and assembles them generically; and
  `registry` holds the curated physics tuning for standard models plus the
  stable named helpers (`get_model`, `pop_model_parser`,
  `pop_model_prior_parser`, `get_fixed_population_params`).
- **EM prior modules** in `darksirens.em` implement the redshift priors for
  spectral sirens, complete catalogs, incomplete catalogs, and bright-siren
  counterparts.  The inference code selects these functions with
  `get_redshift_prior` based on `--universe_model`.
- **Inference modules** in `darksirens.inference` load and compact data, build
  parameter tables and prior transforms, evaluate the selection correction, and
  expose sampler adapters for `jaxns`, `dynesty`, and `emcee`.

## Stability notes

The most stable import points are the named population registry functions, the
container classes, and the CLI-compatible data-loading helpers.  Private helpers
whose names start with an underscore may change when performance or memory
layout changes are made.  When writing downstream code, prefer passing complete
CLI-style option objects into the documented loaders rather than re-creating the
intermediate dictionaries by hand.

## Shape conventions

- Posterior-event arrays are flattened as `nEvents * nsamp` rows.  Event `i`
  occupies the slice `i * nsamp : (i + 1) * nsamp`.
- Selection/injection arrays use one row per injection and are weighted by the
  injection prior weight supplied in the selection file.
- Catalog arrays are indexed by HEALPix pixel.  Compact catalog views replace
  repeated sample pixels with `unique_pixels_*` plus `sample_to_unique_*` maps to
  reduce memory transfer into JAX.
- All likelihood-facing numerical arrays should be convertible to JAX arrays;
  NumPy arrays are accepted by loaders and compactors before the JIT boundary.

```{toctree}
:maxdepth: 2

modules
```
