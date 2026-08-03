# Mock data generation

The repository includes a configurable mock-data workflow for exercising the dark-sirens pipeline end to end.
The defaults keep the ingestibility check tractable, while `RUN_INFERENCE=1` now runs an uncapped, production-style Dynesty configuration unless you explicitly request local-debug caps.

## Generator

Run:

```bash
python scripts/mock_dark_sirens/generate_mock_data.py --outdir data/mock_dark_sirens
```

By default the generator uses a realistic local galaxy-density normalization,
`--n0 1e-3` Mpc^-3, and a low-redshift generation range, `--zmax 0.08`, so the
fixture remains lightweight.  When `--n0` is set, the complete-catalog galaxy
count is derived from the full-sky comoving-volume range and
`--galaxy-density-delta`; pass `--n-galaxies` without `--n0` to request an
explicit catalog size instead.

> **Known bug.** `--galaxy-density-delta` scales the catalog *count* through the
> density-weighted volume, but the redshifts are still drawn from the
> un-evolved comoving-volume CDF, so for `delta != 0` the catalog has the right
> total and the wrong `dN/dz`.  The default `delta = 0` is unaffected.

Survey completeness and density-evolution parameters can be overridden with
`--survey-z50`, `--survey-width`, and `--galaxy-density-delta`; these are the
values the validation runner mirrors into the fixed inference survey JSON.

Measurement widths are controlled by the coefficients of the measurement family
(see below).  Each is the width of its channel at `rho_obs = --snr-threshold`,
and scales as `1/rho_obs` from there:

```bash
python scripts/mock_dark_sirens/generate_mock_data.py \
  --outdir data/mock_dark_sirens \
  --snr-uncertainty 1.0 \
  --lnmc-uncertainty 0.08 \
  --lnq-uncertainty 0.60 \
  --chieff-uncertainty 0.20 \
  --sky-uncertainty-deg 5.0
```

There is no distance-width argument: `dL` is *derived* from `(Mc_det, rho)`, so
the distance precision follows from `--snr-uncertainty` and `--lnmc-uncertainty`
as `sigma_ln dL = sqrt((5/6 sigma_lnMc)^2 + (sigma_rho/rho)^2)`.  Omitting
`--sky-uncertainty-deg` derives the sky width from the recorded SNR;
passing it supplies a data-independent constant instead.

Selection injections are drawn in vectorized NumPy chunks.  `--ndraw` is the
maximum number of proposed injections, and `--selection-batch-size` only controls
the chunk size used to reach that total.  Unless you explicitly pass
`--selection-target-detections` or `--selection-per-observation-factor`, the
generator exhausts all `--ndraw` proposals so changing `NDRAW` changes the
selection sample.  The two target options are mutually exclusive, and all batch
size/count arguments must be positive.  The logs report a detected-injection
proxy `Neff`, computed
from inverse proposal-density weights, as a conservative health check; for
production-like studies increase `--ndraw` until this proxy comfortably exceeds
the inference reliability threshold (`5 * Nobs`) with margin.

The generator writes files that can be consumed directly by `darksirens_inference`:

| File | Purpose |
| --- | --- |
| `mock_galaxy_catalog_complete.h5` | Complete galaxy catalog before EM incompleteness. |
| `mock_survey_raw.h5` | Raw survey-table format accepted by `darksirens_pixelate`. |
| `catalog_pixelated_nside_<nside>.h5` | Pixelated survey catalog accepted by `--survey_path`. |
| `mock_gw_events.h5` | Mock per-event GW posterior samples accepted by `--gw_path`. |
| `mock_gw_selection.h5` | Mock detected gwcat selection samples accepted by `--gwselection_path`. |

The simulation is intentionally simple:

1. Draw galaxies isotropically on the sky and uniformly in comoving volume, each with a true redshift `z` and a realised photo-z `z_obs`.
2. Apply an EM survey footprint, an apparent-magnitude cut, a redshift limit, and a smooth redshift-completeness curve.  The survey block carries `z_obs`, never `z`.
3. Draw GW hosts from the complete pre-selection catalog, at their **true** redshifts.
4. Draw binary masses/spins from `powerlaw+peak` with the default shared beta/spin/gamma CLI settings: a power-law plus Gaussian peak mass model with shared mass-ratio beta, truncated-Gaussian `chi_eff`, and redshift-evolution gamma parameters.
5. Draw one measurement per source and threshold its **recorded** signal-to-noise ratio.
6. Write the exact flat-prior posterior of that same recorded measurement, plus detected gwcat-format selection samples with `pdraw` weights, drawn under the same rule.

## The measurement family

Every measurement width is a function of the recorded signal-to-noise ratio and
of nothing else.  In generative order:

```
rho_obs   = rho_opt(theta) + N(0, sigma_rho)          DETECTION: rho_obs >= rho_th
sigma_x   = a_x * (rho_th / rho_obs)                  for every other channel
ln Mc_obs = ln Mc_det + sigma_lnMc  * N(0,1)
ln q_obs  = ln q      + sigma_lnq   * N(0,1)
chi_obs   = chi_eff   + sigma_chieff* N(0,1)
dec_obs   = dec       + sigma_ang   * N(0,1)          declination FIRST
sigma_ra  = sigma_ang / max(cos dec_obs, 0.1)         from the RECORDED dec
ra_obs    = (ra + sigma_ra * N(0,1)) mod 2 pi
```

with `rho_opt(theta) = snr_ref (Mc_det/30)^(5/6) (1000 Mpc / dL)` and no
projection latent.

This is the family of Fishbach, Holz & Farr (2018),
[arXiv:1805.10270](https://arxiv.org/abs/1805.10270) eqs. 29–31, in the form
released as `GWMockCat` (Farah, Edelman, Zevin, Fishbach, Maria Ezquiaga, Farr
& Holz 2023, ApJ **955**, 107,
[arXiv:2301.00834](https://arxiv.org/abs/2301.00834), App. A;
[code](https://git.ligo.org/amanda.farah/GWMockCat), CC0), whose released
defaults are the generator's: `sigma_rho = 1`, `a_Mc = 0.08`, `a_chi = 0.2`,
`rho_th = 8`.  `a_q = 0.60` comes from the GW150914 anchor
(`q = 0.86 +0.14/-0.21` at network `rho ~ 24`;
[arXiv:1602.03840](https://arxiv.org/abs/1602.03840)), which brackets the
conversion of `GWMockCat`'s `sigma_eta = 0.022 (8/rho_obs)`.  Fishbach & Holz
(2020), [arXiv:1905.12669](https://arxiv.org/abs/1905.12669) App. B add a
`0.2 z/(1+z)` chirp-mass term; it is **not** adopted, because it is a function
of the source's latent redshift, which is exactly the defect this family exists
to remove.  `GWMockCat` drops it too.

Three structural consequences:

* **`dL` is derived, not measured.**  The measurement basis
  `(ln Mc_det, ln q, rho, chi_eff, ra, dec)` is a bijection of
  `(m1det, m2det, dL, chi_eff, ra, dec)`, with
  `dL = 1000 snr_ref (Mc_det/30)^(5/6) / rho`.  Recording `rho_obs` *and*
  measuring `dL` separately would leave `N(rho_obs; rho_opt(theta), sigma_rho)`
  in the true likelihood — a theta-dependent factor the consuming likelihood,
  which sees only samples and `p_pe`, cannot represent.  `GWMockCat` derives
  `dL` from `rho` for the same reason.
* **The mass channel is `(ln Mc, ln q)`, not independent components.**  The
  real degeneracy runs through the SNR (`rho ~ Mc_det^(5/6)/dL`), so a
  chirp-mass error *is* a distance error; independent 8 %/10 % component masses
  are an unphysically strong and uncorrelated mass measurement, and they put
  18.4 % of the PE samples in the `q > 1` region the population prior sets to
  zero.  Here `q <= 1` holds for every sample by construction.
* **Nothing recorded and nothing sampled is clipped.**  Clipping the data makes
  the measurement model censored — the likelihood acquires a theta-dependent
  normalisation `P(obs = boundary | theta) = 1 - Phi(...)` and the exact
  flat-prior posterior stops being a simple normal — and clipping a *sample*
  puts a point mass at the boundary, which is not a density at all.  Physical
  ranges (`q <= 1`, `rho > 0`, `|chi_eff| <= 1`, `|dec| <= pi/2`) are imposed on
  the PE **prior** by exact inverse-CDF truncation.  A recorded value is
  therefore allowed to lie outside the physical range.

The width is a function of *data*, which is the point.  A width scaled by the
latent truth, `N(obs; m, f m)`, carries a theta-dependent normalisation `1/(f m)`,
so its flat-prior posterior is skewed by construction; measured on a matched
mock with the exact per-event posterior of that family, the detected-set score
identity `E[C] = E[A]` was violated at **11.3 sigma**, against **1.4 sigma** for
this family.  The same mechanism in the sky channel alone was worth
**−0.49 ± 0.08 km/s/Mpc** in recovered `H0`.

`sigma` is treated as known per event on both the measurement and the posterior
side, which keeps the conjugacy exact; real parameter estimation infers the
width from the data.  This is the standard mock-PE idealisation (`GWMockCat`
makes it too).

### Sky-width scale

`sigma_ang = clip(35 deg / rho_sigma, 1, 12)` with
`rho_sigma = (11.5 / snr_ref) * rho_obs` — 11.5 is the reference the sky
convention has always used, and it is not necessarily the reference the
detection statistic uses.  At the default `--snr-ref 11.5` the factor is 1 and
the width is `clip(35/rho_obs, 1, 12)`; at `--snr-ref 6.278` it is 1.83165, i.e.
`clip(19.1069/rho_obs, 1, 12)`, which realises the same width distribution.  Any
port must carry that factor explicitly or the sky widths silently move by 1.83x.
(The `Delta Omega ~ rho^-2` scaling behind the convention: Fairhurst 2009,
[arXiv:0908.2356](https://arxiv.org/abs/0908.2356); Berry et al. 2015, ApJ
**804**, 114, [arXiv:1411.6934](https://arxiv.org/abs/1411.6934).)

### `--snr-ref` needs recalibrating

Detection thresholds `rho_obs`, and there is **no projection latent** — keeping
one would make the detection decision depend on a variable absent from the data,
which leaves an extra `P(det|theta)` inside each event's integral and is the
defect the family exists to remove.  Dropping it raises the detected fraction by
about **5.75x** at fixed `--snr-ref` relative to a true-parameter cut, so
`--snr-ref` must be recalibrated if a detected population is to stay comparable
to a mock generated before this change.  On the gws-agn deep mock,
`--snr-ref 6.278` reproduced the old detected fraction.

### `p_pe` is the PE prior in darksirens' canonical basis

`darksirens/inference/utils.py` fixes the basis:

> both posterior samples and detected injections are integrated in the same
> coordinates `(m1det, q, dL, chieff, sky pixel)`, not in `(m1det, m2det, dL)`.
> Any proposal density divided out by the likelihood (`p_pe` …) must be
> expressed per unit `m1det`, per unit `q`, per Mpc of `dL` … A density native
> to `(m1det, m2det, dL)` is converted to the canonical `(m1det, q, dL)` basis
> by multiplying by `|dm2det/dq| = m1det`.

With the PE prior flat in `(ln Mc_det, ln q, rho, chi_eff)`, the Jacobian to
that basis gives

```
p_pe  ~  rho / (dL m1det q)  ~  Mc_det^(5/6) / (dL^2 m1det q)
```

which is what the `p_pe` column holds, normalised to mean 1 per event
(darksirens renormalises per event, so only the shape matters).  An all-ones
column would declare a prior flat in the *stored* variables, which is a
different — and, in this basis, wrong — statement; the file records the basis in
the `p_pe_basis` attribute.

### The two redshift columns

`mock_galaxy_catalog_complete.h5` carries **both** `z` (true) and `z_obs`.  The
true redshift drives the host draw and the event truth; the survey products
(`mock_survey_raw.h5`'s `Z`/`ZERR` and the pixelated `zgals`/`dzgals`) carry
`z_obs` and the width of the model evaluated at it,
`dz = redshift_error_floor + redshift_error_slope (1 + z_obs)`.

Declaring an error without realising it is not conservative, it is an internal
inconsistency: the likelihood then smooths a comb that carries no error, so
darksirens' per-galaxy kernel `g(z) N(z; z_g, sigma_g)/Z(z_g)` is not the
Bayesian posterior for that host's true redshift.  Measured on a matched mock,
that was worth `+6.383e-4 ± 0.836e-4`, a **7.6 sigma** effect; realising the
error left `-5.49e-5 ± 9.19e-5`, 0.60 sigma.

`z_obs` may be negative for a galaxy at `z ~ 0`.  It is **not** clipped —
clipping re-introduces a censored observation — and the realised count is
recorded in `metadata_json` under `catalog_photoz`.

## Validation shell script

Run the default ingestibility validation with:

```bash
bash scripts/mock_dark_sirens/run_mock_data_test.sh
```

The script creates a data set under `data/mock_dark_sirens_test` using `N0=1e-3` Mpc^-3 and calls `darksirens.inference.data.load_all_data` to verify that the generated HDF5 products are readable by the inference pipeline.

To also launch an optional production-style sampler run with free cosmology, use:

```bash
RUN_INFERENCE=1 bash scripts/mock_dark_sirens/run_mock_data_test.sh
```

You can override the mock size without editing the script, for example:

```bash
NOBS=5 NSAMP=256 NDRAW=50000 NSIDE=16 bash scripts/mock_dark_sirens/run_mock_data_test.sh
```

By default the validation script does not set a detected-injection stopping
target, so it consumes `NDRAW` proposed selection injections even when
`RUN_INFERENCE=1`.  If you need a fast cap for local debugging, set either
`SELECTION_TARGET_DETECTIONS` or `SELECTION_PER_OBSERVATION_FACTOR`; those caps
intentionally make `NDRAW` an upper bound rather than the exact number of
proposals.

The validation script pins common BLAS/OpenMP thread counts to one and disables
JAX preallocation unless the caller has already set those environment variables.
This keeps the small fixture responsive on shared CPU machines and avoids the
common fork-after-JAX runtime deadlock when a library creates worker processes
after JAX has initialized its thread pool.

The optional inference run fixes only the generated survey hyperparameters via
`--fixed_parameter_values`, including `log10n0 = -3`, and explicitly leaves
cosmology free (`--fix_cosmology False`) so `H0`, `Om0`, `w0`, and `wa` are sampled. Use `--fix_de True` for a ΛCDM mock-analysis run that samples only `H0` and `Om0`. The mock generators accept `--w0` and `--wa` and store them in `metadata_json` alongside `H0` and `Om0`.  It
passes `--sel_batch_size` (default `INFERENCE_SEL_BATCH_SIZE=256`) for memory
safety, uses `INFERENCE_NLIVE=1000` and `INFERENCE_DLOGZ=0.1` by default, and
does not cap Dynesty likelihood calls unless you set a positive
`INFERENCE_MAX_SAMPLES` (the default `0` disables the cap).


## Bright-siren mock data

A separate bright-siren mock workflow is available under `scripts/mock_bright_sirens` so the dark-siren mock scripts remain unchanged. It generates a complete galaxy population, applies an EM survey selection, draws GW events from galaxies with detectable counterparts, fixes the PE sky samples to the counterpart positions, and writes joint GW+EM selection injections for multi-event bright-siren inference.

```bash
bash scripts/mock_bright_sirens/run_mock_bright_sirens_test.sh
```

The runner writes `mock_bright_gw_events.h5`, `mock_bright_gw_selection.h5`, and `bright_counterparts.json` under `data/mock_bright_sirens_test` by default. Set `RUN_INFERENCE=1` to run `darksirens_inference` with `--universe_model bright_sirens` and one `--counterpart RA DEC Z` triplet per generated event.
