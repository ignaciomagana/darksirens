# darksirens

`darksirens` is a Python package for joint gravitational-wave inference with large-scale galaxy surveys. It provides command-line tools for spectral-siren and dark-siren hierarchical inference, survey pixelation, and posterior-predictive analysis.

## Documentation

Hosted documentation can be built with Sphinx and published on Read the Docs using the included `.readthedocs.yaml` configuration.

Build the docs locally:

```bash
python -m pip install -r docs/requirements.txt
make docs-html
```

For a stricter pre-publish check that treats warnings as errors while continuing through all warnings, run:

```bash
make docs-strict
```

Start with the documentation source at [`docs/source/index.md`](docs/source/index.md), or see the quickstart guide at [`docs/source/quickstart.md`](docs/source/quickstart.md).


## Cosmology model

The inference cosmology block samples a flat CPL dark-energy model with labels `H0`, `Om0`, `w0`, and `wa`. The default dark-energy point `w0=-1` and `wa=0` recovers flat ΛCDM. Use `--fix_cosmology true` to hold all four cosmology labels fixed, `--fix_de true` to hold only `w0` and `wa` fixed, and `--prior_overrides`/`--fixed_parameter_values` to narrow or pin individual labels.

## Command-line tools

Installing the package exposes:

- `darksirens_pixelate` — convert a raw galaxy survey HDF5 file into a pixelated HEALPix catalog.
- `darksirens_inference` — run spectral-siren or dark-siren hierarchical inference.
- `darksirens_analyze` — analyze saved inference products and posterior-predictive distributions.
- `darksirens_skymaps_to_samples` — convert a directory of 3D skymap FITS files into GW posterior-like samples (`gwdata.h5`) with broad uninformative mass/spin surrogates for low-latency runs.
- `darksirens_build_lognormal_completion` — **offline** preprocessor that builds an LSS-conditioned lognormal completion field `Q_LSS(p,z)` (see below) from a pixelated survey catalog.
- `darksirens_diagnose_lognormal_completion` — per-pixel diagnostic plots of `Q_LSS`, the missing-galaxy density, and the redshift prior.

## Population and completion models

Population mixtures are selected with `--pop_model` using a compositional name grammar. Tokens such as `powerlaw`, `brokenpowerlaw`, and `peak` can be combined directly (`powerlaw+peak`, `brokenpowerlaw+2peaks`, `2powerlaws+3peaks`), with curated names receiving physics-tuned priors and arbitrary grammar combinations using blueprint defaults. Use `--shared_beta`, `--shared_spin`, and `--shared_gamma` to choose shared (default) versus per-component pairing, spin, and redshift-evolution parameters. Mixture weights are sampled as stick-breaking parameters labeled `$v_i$`; copy the printed startup parameter table when passing population labels to `--prior_overrides` or `--fixed_parameter_values`.

Incomplete-catalog dark-siren runs use a data-driven completion model rather than a parametric logistic rolloff: the observed per-pixel galaxy redshift KDE is divided by the identically smoothed expected `n0 * dV_c/dz * (1 + z)^delta` density, clipped to `[0, 1]`, and converted into an additive missing-galaxy density. `z50` and `w` remain in the survey parameter block for compatibility, but the current completion likelihood is controlled by `log10n0`, `delta`, `b_miss` (with fixed `alpha_miss = 1` unless overridden), and `sigma_kde`. Run `--validate_completion true` for dry-run clipping diagnostics before long dark-siren analyses.

### LSS-conditioned lognormal completion

By default the missing-galaxy branch is modulated by the local-overdensity factor `max(1 + b_eff * delta_g(p,z), 0)`. Optionally it can instead be multiplied by a **precomputed, LSS-conditioned lognormal completion field** `Q_LSS(p,z)`:

```
dN_miss(p,z) = [1 - C(p,z)] * dN_exp(z) * Q_LSS(p,z)
```

`Q_LSS` is a *clustered* missing-galaxy correction. It is built **offline** from a per-pixel, 1-D Poisson-lognormal model along the redshift grid (a latent Gaussian field correlated along comoving distance, with a built-in Gaussian-correlation power spectrum whose fixed hyperparameters — correlation length, amplitude, bias — come from `SurveyParams`/`CosmoParams` and are never marginalised). **The likelihood stays deterministic: it never samples a field or generates galaxies — it consumes fixed `Q` arrays.** With no completion file supplied, behaviour is unchanged (the legacy `delta_g` factor is used).

Build a completion file from a pixelated catalog and pass it to inference:

```bash
darksirens_build_lognormal_completion --catalog survey.h5 --out lss_completion.h5 --n-members 32
darksirens_inference --universe_model dark_sirens --sky ... --lss_completion lss_completion.h5 ...
darksirens_diagnose_lognormal_completion --catalog survey.h5 --lss-completion lss_completion.h5 --pixel 1234 --outdir figs
```

The builder produces a **MAP** estimate `Q_MAP` and, optionally, a fixed **Laplace/FFT-diagonal posterior ensemble** `{Q^(m)}` (an approximation around the MAP, not a full BORG sampler). Inference consumes only the deterministic MAP (or the posterior-mean when only an ensemble is present); the ensemble drives the Bayesian redshift-prior **diagnostic**

```
p_Bayes(z|p) = (1/M) * sum_m p_m(z|p),
```

where each member prior `p_m` is normalised individually. The target for a fully Bayesian marginalisation over completion realisations is

```
log L(Lambda) ≈ logsumexp_m log L(Lambda; Q_m) - log M,
```

which is **not** performed inside the GW likelihood in this implementation (member support is exposed through prior-state diagnostics).

**Scope & caveats (experimental — read before using as a science result).** This is a **radial, per-pixel** lognormal completion: each HEALPix line of sight is an independent 1-D field along comoving distance, so it does **not** borrow information angularly between neighbouring pixels (it is not a 3-D `P(k)`-conditioned reconstruction — a genuine angular-coupling upgrade is planned separately). The GW likelihood uses the **deterministic / posterior-mean** `Q` (not the fully-marginalised `logsumexp_m`), so it is not yet Bayesian over field uncertainty. The completeness `C = dN_obs/dN_exp` and the fitted `Q` are both derived from the **same** observed counts, so `Q` is the sub-smoothing **radial residual** rather than a separately-identifiable completeness; and the whole construction assumes **missing galaxies trace the observed clustering** along the line of sight — an assumption the data alone cannot validate. `Q` is built at **fixed fiducial** cosmology/survey parameters (printed + stored at load) while inference varies them. Treat results as exploratory.

## Minimal installation

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

`requirements.txt` installs `gwcat` from a pinned `ignaciomagana/gwcat` Git commit. Keep `gwcat` external rather than vendoring it into this repository: `gwcat` owns preprocessing of raw GW PE and selection/injection products, while `darksirens` consumes the resulting HDF5 catalogs for inference. Additional sampler-specific packages such as `dynesty` or `emcee` may be required for the workflows you choose.
