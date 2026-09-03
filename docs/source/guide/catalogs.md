# Dark sirens with galaxy catalogs

How to run `darksirens_inference` with one or more pixelated galaxy catalogs:
completeness models, the LSS-conditioned lognormal completion field, marks,
multitracer mixtures and sky anisotropy. The equations are in
[Theory & methods](../reference/theory.md), the file schemas in
[Input files](../getting-started/inputs.md).

## The incomplete-catalog model

`--universe_model dark_sirens` builds the per-pixel redshift prior from two
branches. The **observed** branch is the per-galaxy kernel sum
$N_{\rm obs}(p)\,p_{\rm cat}(z\mid p)$, each catalogued galaxy contributing a
truncated, boundary-normalised Gaussian of width $\sigma_{{\rm eff},i} =
\max(\sqrt{\sigma_{{\rm cat},i}^2 + \sigma_{\rm kde}^2},\,10^{-4})$: the sampled
`sigma_kde` is extra kernel broadening added in quadrature to each galaxy's own
redshift uncertainty. The **missing** branch is a count density,

$$
\frac{\mathrm{d}N_{\rm miss}}{\mathrm{d}z}(p,z)
 = \big[1 - C(p,z)\big]\,
   n_0\,a_{\rm pix}\,\frac{\mathrm{d}V_c}{\mathrm{d}z}\,(1+z)^{\delta}\,
   Q_{\rm LSS}(p,z),
$$

where `log10n0` is $\log_{10} n_0$ with $n_0$ in Mpc$^{-3}$ and `delta` is the
exponent of the $(1+z)^{\delta}$ density evolution. With `--use_lss` and no
completion table, $Q_{\rm LSS}$ is the legacy local-overdensity factor
$\max(1 + \alpha_{\rm miss} b_{\rm miss}\,\delta_g(p,z),\,0)$, whose amplitude
is the sampled `b_miss` (`alpha_miss` enters only through the exact product and
stays at 1); `--lss_floor conserve`, the default, renormalises that floored
factor to full-sky mean one at every redshift, and `legacy` does not.

Completeness is a matched-kernel ratio, not a parametric roll-off: the same
linear smoothing operator is applied to the observed per-galaxy density and to
the expected density $\mathrm{d}N_{\rm exp}/\mathrm{d}z$, and $C$ is their
clipped ratio in $[0,1]$ (`darksirens.redshift.completion`). `--c_mode
per_pixel` (the default) forms that ratio pixel by pixel; `aggregate` forms one
sky-aggregate $\bar C(z)$ per proposal and broadcasts it to every pixel, so $C$
carries only the radial budget and all angular structure is $Q$'s job;
`selection` replaces counts entirely with the parametric magnitude-limited
$C_{\rm sel}(z;\theta)$ (see
[`darksirens_fit_selection`](#parametric-magnitude-selection)).
`--survey_z_depth Z` declares the survey depth as a completeness prior:
completeness is exactly zero beyond `Z`, so every modelled host there is
uncatalogued rather than nonexistent, the observed kernel is zeroed there and
the completeness denominator is truncated at the depth. It applies to all
catalogs and overrides any per-catalog `z_depth` attribute from
`darksirens_pixelate --z_depth`. Run `--validate_completion true` first: it
writes a dry-run clipping diagnostic under `--save_path`
(`completion_validation__<ts>.json`, or `completion_validation_c{k}__<ts>.json`
per catalog) with the likelihood's own estimator and exits before the
likelihood is built; `--completion_validation_pixels` bounds the pixels it
inspects.

## The complete-catalog model

`--universe_model dark_sirens_complete` assumes the catalog traces every
possible host, so the prior is the catalog term alone with no missing branch.
Genuinely empty pixels are then a modelling choice:
`--complete_empty_pixel_policy zero` (the default) is the formal
complete-catalog behaviour and returns zero probability for a pixel holding no
real galaxies, while `volume` gives those pixels the comoving-volume prior as an
approximation for sparse or high-resolution pixelations. The test is the pixel's
real galaxy count, not whether the evaluated KDE was finite.

## The per-sample catalog KDE window

`p_cat` is evaluated on a window of the z-sorted galaxy row, not the whole row.
Unset, `--kde_window` is **sized from the data** at build time by
`darksirens.redshift.catalog.auto_kde_window`: the largest in-range block any
bound row holds at the widest `sigma_kde` the run can reach (its fixed value,
or the prior's upper bound when sampled), so the evaluator never truncates.
`--kde_window_nsigma X` (default 8) is the sizing multiplier: the window must
hold every galaxy within `X * max_row(sigma_eff)` of a sample. Pinning
`--kde_window W` below the data-sized value **truncates** the catalog prior and
warns at build time, since the window is centred on the sample's insertion
index and never repositioned to fit a block. `--kde_window 0` disables
windowing (the full-row path, for A/B validation).

## LSS-conditioned lognormal completion

$Q_{\rm LSS}(p,z)$ is a clustered, mean-one missing-galaxy modulation fitted
**offline** so the likelihood never samples a field: a Poisson-lognormal model
whose latent Gaussian field gives $Q = \exp(b s - b^2\sigma_s^2/2)$ against the
catalog's own observed counts. `darksirens_build_lognormal_completion --mode
radial` (the default) solves each HEALPix line of sight independently in 1-D
with an FFT-diagonalised circulant prior; `--mode gp3d` solves one low-rank
$(\text{sphere}\times z)$ GP field over occupied voxels, so empty pixels borrow
angularly from their neighbours. `--n-members M` (default 32) adds a Laplace
ensemble; `--log10n0` and `--delta` set the conditioning density, which must be
calibrated to the catalog ($N / (f_{\rm sky} V_c)$) because a mis-set $n_0$ is
absorbed into $Q$ as spurious redshift structure. The build's `--c-mode` must
match the run's `--c_mode` and, for `selection`, its `--selection-fit`; both
are stamped and hard-checked at load.

Pass the table with `--lss_completion PATH`, which replaces the legacy
overdensity factor. The table is conditioned on its build-time fiducials, so
the provenance guard refuses a run in which `log10n0`, `delta`, `b_miss`,
`Om0`, `w0` or `wa` is sampled or fixed to a value differing from the stamp
(`H0` is exempt; it is the measurand). Pin them with
`--fixed_parameter_values`, or rebuild the table at the values you intend.
`--lss_marginalize true` replaces the deterministic posterior-mean $Q$ with the
fully Bayesian ensemble marginalisation $\log L = \mathrm{logsumexp}_m \log
L(Q_m) - \log M$, and needs a file built with `--n-members M > 0`. For a
$K\ge2$ mixture the marginalisation uses **one shared member index**, so build
the per-catalog ensembles together with
`darksirens_build_joint_lognormal_completion`: it infers one latent field from
all $K$ catalogs jointly (`gp3d` only; `--mode radial` is rejected, having no
shared field) and stamps a single `realization_set_id` across the $K$ outputs.
Without matching ids the run aborts unless
`--allow_unverified_shared_lss_members` accepts the independent-fields
approximation. Inspect a table with
`darksirens_diagnose_lognormal_completion --pixel P`, which writes
`lss_completion_pixel<P>.pdf`: $Q=1$, the MAP field and the member band for
$Q_{\rm LSS}$, $\mathrm{d}N_{\rm miss}/\mathrm{d}z$ and $p(z\mid p)$.

```{warning}
Stated caveats, from the builder's module docstring
(`darksirens.redshift.lognormal_completion`): in `radial` mode the field is
independent per pixel with no angular coupling; $C$ and the fitted $Q$ come
from the **same** observed counts, so $Q$ is a sub-smoothing residual rather
than a separately identifiable completeness; the model assumes missing galaxies
trace the observed clustering, which the data alone cannot validate; and the
fit is built at fixed fiducial cosmology and survey parameters while the
inference varies them. Shipped tables are per-$z$ mean-one
renormalised under the $(1-C)\,\mathrm{d}N_{\rm exp}$ weights, so $Q$ places
the missing budget without rescaling it (`--no-budget-renorm` removes that
renormalisation; research ablations only).
```

## The marked-host model

`--mark_model loglinear` reweights each catalog galaxy by a sampled BBH-host
efficiency $h(m\mid\eta) = \exp(\sum_k \eta_k \tilde m_k)$ over per-galaxy
marks; the missing branch is scaled by the expected efficiency of unobserved
galaxies, $\mu_{\rm miss}(z\mid\eta) = \mathbb{E}_{\rm obs}[h\mid z]$.
`--marks LIST` selects a subset of `logmstar,logssfr,metallicity,color` and
defaults to every mark present in the catalog; provide them by adding the
columns `LOGMSTAR`, `LOGSSFR`, `LOGZ` and `GR_COLOR` to the raw catalog before
`darksirens_pixelate`, which pads them alongside `zgals`
(`MARK_INPUT_COLUMNS` in `darksirens.cli.pixelate`). Marks are **z-centred at
load** ($\tilde m = m - \mathbb{E}[m\mid z]$), so $\eta$ measures host
preference at fixed redshift and cannot mimic $R(z)$, $H_0$ or $\gamma$:
positive `eta_logmstar` means GW hosts prefer high-stellar-mass galaxies,
positive `eta_logssfr` means they prefer star-forming ones, and $\eta = 0$
(or `--mark_model none`, the default) recovers the galaxy-count host model
exactly.

## Multitracer catalog mixtures

Passing several files to `--survey_path a.h5 b.h5` runs the $K$-catalog mixture
for `dark_sirens`: the redshift prior becomes $\sum_k w_k\, p_k(z)$ with
per-catalog nside and pixelisation, per-catalog survey blocks labelled `_c{k}`
for $k \ge 2$ (`log10n0_c2`, `delta_c2`, `b_miss_c2`, `sigma_kde_c2`, and
`eta_<mark>_c2` under `--mark_model loglinear`), and sampled stick-breaking
weights `fcat_2..fcat_K`. Under
`--catalog_sky_weighting field` (the default at every $K$) the normaliser is
the survey-global budget, so relative angular host density survives and $w_k$
is the fraction of GW hosts drawn from catalog $k$'s tracer population;
`darksirens_analyze` derives $w_1..w_K$ from the sampled sticks, prints their
5/50/95% quantiles as host fractions and writes
`catalog_weights_<tag>.{pdf,npy}`. `--lss_completion` takes 0, 1 or $K$ paths
aligned with `--survey_path` and `--selection_fit` one comma-separated entry
per catalog in the same order (`""` as a placeholder in either); a single entry
is never broadcast across catalogs.

## Sky anisotropy

`--sky_model` multiplies the source rate by a mean-one angular density
$g(\hat n, z)$, so isotropy is exactly $g \equiv 1$ and the shape does not trade
against the rate normalisation. `isotropic` (the default) is the null; `dipole`
(Isi, Farr & Varma 2023) and `sphere_gp` (a log-Gaussian random field, Essick
et al. 2023) are angular; `sphere_gp_z` is a $(\text{sphere}\times z)$ GP
normalised per redshift shell and `overdensity_gp` the same field normalised
over the comoving volume (full 3-D clustering, to be used with `gamma` fixed);
`multipole` and `multipole_l3` are spherical-harmonic expansions $g = 1 + \sum
a_{\ell m} Y_{\ell m}$ to $\ell \le 2$ and $\ell \le 3$, giving the angular power
spectrum $C_\ell$. All are compared to isotropy by evidence, and one
population-level $g$ is shared across the catalogs of a mixture.

## Parametric magnitude selection

`darksirens_fit_selection` belongs to the `c_mode=selection` route: it reads a
pixelated survey's padded `gal_app_mag` dataset, fits the
truncated-luminosity-function selection at the survey's hard limit `--m_lim`
(`--family gaussian` for the mock program's truncated-normal magnitude model,
`--family schechter` for a real catalog's luminosity function, `--strata` for
one $\theta$ per stratum), and writes a JSON consumed twice: by
`darksirens_build_lognormal_completion --c-mode selection --selection-fit` as
the fixed base of the $Q$ fit, and by `darksirens_inference --selection_fit`,
where $\hat\theta$ and the marginal Laplace standard deviations become the
Gaussian prior on the sampled selection labels. It works in reference absolute
magnitudes $m - \mathrm{DM}(z; H_0 = 100)$, so it is independent of the true
$H_0$ and of the density field; run it with `JAX_PLATFORMS=cpu`.

## Complete commands

Build one catalog's table with a 32-member ensemble (`--n-members 0` writes
the MAP field alone):

```bash
darksirens_build_lognormal_completion \
  --catalog catalog_nside_32.h5 \
  --out q_radial_ens.h5 \
  --mode radial \
  --n-members 32 \
  --log10n0 -2.4 --delta 0.0 \
  --z-depth 0.3 \
  --c-mode per_pixel \
  --seed 1234
```

Build matched ensembles for a two-catalog mixture (one shared latent field):

```bash
darksirens_build_joint_lognormal_completion \
  --catalogs galaxies_nside_32.h5 agn_nside_32.h5 \
  --outs q_galaxies.h5 q_agn.h5 \
  --mode gp3d \
  --n-members 32 \
  --log10n0 -2.4 -4.0 \
  --delta 0.0 \
  --bias 1.0 1.5 \
  --seed 1234
```

Run the dark-siren analysis with the ensemble. The conditioning parameters are
pinned at the values the table was built with (`Om0`, `w0` and `wa` here are
the package fiducials in `darksirens.core.constants`):

```bash
darksirens_inference \
  --gw_path gw_events.h5 \
  --gwselection_path gw_selection.h5 \
  --survey_path catalog_nside_32.h5 \
  --universe_model dark_sirens \
  --survey_z_depth 0.3 \
  --lss_completion q_radial_ens.h5 \
  --lss_marginalize true \
  --mark_model loglinear \
  --marks logmstar,logssfr \
  --sky_model isotropic \
  --pop_model powerlaw+peak \
  --fix_population true \
  --fixed_parameter_values '{"log10n0": -2.4, "delta": 0.0, "Om0": 0.3075, "w0": -1.0, "wa": 0.0}' \
  --prior_overrides '{"H0": [40.0, 140.0]}' \
  --sampler tinyns \
  --nlive 1000 \
  --tinyns_sample rwalk \
  --tinyns_kernel jax \
  --checkpoint_interval 1800 \
  --resume auto \
  --save_path runs/darksirens
```

See [Running inference](inference.md) for the cosmology, survey and sampler
blocks, [Populations](populations.md) for `--pop_model`,
[Performance](performance.md) for KDE-window and memory sizing, and
[Analysis](analysis.md) for the run directory.
