# Population models

`--pop_model` names the compact-binary population: for parametric mixtures the
name itself is the mass-composition definition, and a few bespoke models are
registered under fixed names. This page covers that grammar, the curated names,
the sharing flags, the sampled labels, the GP variants and the flow surrogates.

## The name grammar

```text
composition := term ("+" term)*
term        := TOKEN | <digits><PLURAL>      # "peak", "2peaks", "3powerlaws"
```

So `brokenpowerlaw+2peaks+powerlaw` parses to the mass composition
`[BrokenPowerLaw, Gaussian, Gaussian, PowerLaw]`. The mass tokens are:

| Token | Plural | Slot tag | Mass parameters |
| --- | --- | --- | --- |
| `powerlaw` | `powerlaws` | `PL` | `alpha`, `m_min`, `m_max`, `dm_min`, `dm_max` |
| `brokenpowerlaw` | `brokenpowerlaws` | `BPL` | `alpha1`, `alpha2`, `m_break`, `m_min`, `m_max`, `dm_min`, `dm_max` |
| `peak` | `peaks` | `G` | `mu`, `sigma` |

Any composition works without code changes: an unseen one such as
`powerlaw+3peaks` builds from the blueprints' default priors and fiducials, with
uniform fiducial mixture weights and an informational log line, and is tuned per
run with `--prior_overrides`.

## Curated and bespoke names

Seven compositions are curated, carrying tuned per-component priors, fiducials
and a display label instead of the blueprint defaults: `powerlaw+peak` (`PL+G`),
`brokenpowerlaw+2peaks` (`BPL+2G`), `brokenpowerlaw+3peaks` (`BPL+3G`),
`brokenpowerlaw+2peaks+powerlaw` (`BPL+2G+PL`), `2powerlaws+peak` (`2PL+G`),
`2powerlaws+2peaks` (`2PL+2G`) and `2powerlaws+3peaks` (`2PL+3G`).

```{note}
`powerlaw+peak` is a curated test composition, not the published GWTC-3 model:
its fiducial puts $w_{\rm G} = 0.90$ in the Gaussian peak (GWTC-3 measures
$\lambda_{\rm peak} = 0.038$) and its peak is untapered on primary mass. Use
`gwtc3_fiducial_plpeak` for the published model.
```

Bespoke models bypass the mixture grammar: `gwtc3_fiducial_plpeak`
(arXiv:2111.03634 Table VI priors and Eqs. B4-B7),
`gwtc3_plpeak_component_spin` (the same masses with a component-basis spin block,
which needs a component-basis gwcat pair), `gwtc5_fiducial_bpl2peaks` (GWTC-5.0
release medians), `golomb_1g` and `golomb_1g+tail`. The spellings
`twopowerlaws+*`, `gwtc5_fiducial_brokenpowerlaw+2peaks`,
`gwtc5_brokenpowerlaw+2peaks`, `gwtc3_fiducial_powerlaw+peak` and
`gwtc3_powerlaw+peak` resolve to their canonical names with a
`DeprecationWarning`.

Under `--fix_population true`, `--population_fiducials` picks the curated vector
the block is pinned at: `legacy` (default), or `in_prior_v2`, which moves each
parameter violating its own declared prior to that prior's midpoint.
`2powerlaws+peak`, `2powerlaws+2peaks` and `2powerlaws+3peaks` are the entries
whose `legacy` fiducials sit outside their priors.

## Shared versus per-component blocks

Sharing is a separate CLI control, never a suffix in the name; each flag defaults
to `true`.

| Flag | `true` | `false` |
| --- | --- | --- |
| `--shared_beta` | one pairing distribution for the mixture | one `beta` per mass component |
| `--shared_spin` | one spin distribution | one spin distribution per mass component |
| `--shared_gamma` | one redshift-evolution `gamma` | one `gamma` per mass component |

## Sampled labels

The block is ordered `v_weights -> mass slots (composition order) -> pairing ->
spin -> gamma`, and the label rule is uniform:

- Mixture weights are stick-breaking inputs `$v_1$ ... $v_{k-1}$` in `[0, 1]` for
  a `k`-component mixture, with a Beta prior; the last component takes the
  remaining stick. To pin final fractions, convert with
  $v_i = w_i / (1 - w_1 - \dots - w_{i-1})$, so `$v_1$` is `w_1` at `k = 2`.
- Mass parameters use the base label for a single mass slot and the slot-tagged
  label otherwise (`$\alpha_{\rm PL}$`, `$m_{\min,\rm BPL}$`, `$\mu_{\rm G1}$`,
  `$\mu_{\rm G2}$`), indexed only when the token occurs more than once.
- Pairing and spin labels are bare when shared (`$\beta$`, `$\mu_\chi$`,
  `$\sigma_\chi$`) and slot-tagged otherwise (`$\beta_{\rm G2}$`), and
  `$\gamma$` is always last.

Use these spellings verbatim as JSON keys, escaping the backslashes the shell
would otherwise eat:

```bash
--prior_overrides        '{"$\\alpha_{\\rm PL}$": [0.0, 4.0]}'
--fixed_parameter_values '{"$\\mu_{\\rm G}$": 35.0, "$v_1$": 0.1}'
```

The bespoke models keep their own untagged labels (`gwtc5_fiducial_bpl2peaks`
samples `$\alpha_1$`, `$m_{\rm break}$`, `$\mu_1$`), so copy every key from the
startup table as described in [Running inference](inference.md).

## Gaussian-process variants

The GP family replaces a parametric mass, mass-ratio, spin or redshift shape with
a log-density Gaussian process, sampling kernel hyperparameters (`$\log A$`,
`$\log\ell_m$`, ...) plus one `$\xi_i$` latent per grid node, so these models are
high dimensional (92 labels for `gp2d_m1_q`, 26 for `gppop`). The registered
names are `gp1d_{m1,q,chi,z}`, the six `gp2d_*` pairs (`m1_q`, `m1_chi`, `m1_z`,
`q_chi`, `q_z`, `chi_z`), the four `gp3d_*` triples (`m1_q_chi`, `m1_q_z`,
`m1_chi_z`, `q_chi_z`), `gp4d`, `gp_separable` (independent 1-D GPs, no
interactions), `gp4d_additive` (functional-ANOVA mains plus interactions), and
the binned `gppop` and `gppop_mz`.

```{warning}
Evaluating a GP population imports `tinygp`, which ships in the `gp` extra
(`pip install "darksirens[gp]"`). Building the parameter space does not need it,
so a missing extra surfaces at the first likelihood call.
```

`darksirens_inference` writes `latents.pdf` and `darksirens_analyze` writes
`latents_<tag>.pdf` for these models.

## Normalizing-flow single-event surrogates

`--gw_flows_path <dir>` replaces the stored PE samples with one trained
normalizing flow per event (`<EVENT>/<EVENT>_flow.npz` flowjax checkpoints over
detector-frame `mass_1`, `mass_2`, `luminosity_distance`, `chi_eff`). It is
mutually exclusive with `--gw_path`, supports `--universe_model spectral_sirens`
only, and needs the `flows` extra (`pip install "darksirens[flows]"`, i.e.
`flowjax`, `paramax`, `equinox`).

Each likelihood call draws `--flows_nsamp` source-frame points per event from the
current population model and scores them with the event's flow `log_prob` divided
by the analytic PE prior, so the per-event evidence integral is a Monte Carlo
estimate instead of a sum over stored samples. Base uniforms are fixed by
`--flows_seed` (common random numbers), keeping the likelihood deterministic and
continuous in the hyperparameters.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--flows_nsamp` | `16384` | Population draws $J$ per event per likelihood call. Per-event ESS is typically 1-5% of $J$ on real events. |
| `--flows_seed` | `42` | Seed of the fixed base-uniform array. |
| `--flows_pattern` | `*/*_flow.npz` | Checkpoint glob relative to `--gw_flows_path`. |
| `--flows_on_mismatch` | `error` | Policy for checkpoints that do not match the installed flowjax: `error`, `skip`, `load`. |
| `--flows_chieff_amax` | `0.99` | $a_{\max}$ of the 1-D isotropic `chi_eff` PE prior. |
| `--flows_pe_cosmology` | `67.74,0.3089` | `H0,Om0` of the PE prior cosmology (UniformSourceFrame distance prior), matching the PE release. |
| `--flows_grid_nm`, `--flows_grid_nq` | `512`, `256` | Cells of the $(m_1, q)$ population sampling grid. |
| `--flows_support_margin` | `0.25` | Fractional per-side expansion of each event's sampled parameter range, corrected exactly in the proposal density. |
| `--flows_support_nsamples` | `4096` | Flow draws used to measure each event's support box. |
| `--flows_wfull` | `0.05` | Fraction of each event's draws taken from the full population support instead of its window. |

The support window is measured from finitely many flow samples and so does not
cover a spline flow's unbounded learned support. `--flows_wfull` therefore makes
the proposal a two-component mixture whose exact density enters every weight: any
value in `[0, 1)` is unbiased, while `0` restores the windowed-only estimator,
which truncates the event integral by a hyperparameter-dependent amount. The
residual Monte Carlo error is visible to `--max_likelihood_variance`, which
rejects undersampled settings.

Per-event ln-evidence offsets against a stored-PE run are
hyperparameter-independent PE-prior normalisation constants and do not move the
posterior. The selection integral has its own emulator (`--pdet_flow_path` and
its `--pdet_*` options), described in the [CLI reference](../reference/cli.md).
