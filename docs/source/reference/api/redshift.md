# redshift (`darksirens.redshift`)

The redshift priors and completeness models: the observed-galaxy catalog
kernel, the missing-galaxy budget, the offline LSS completion field, and the
latent-field basis that replaces it in latent mode. `get_redshift_prior`
dispatches on `--universe_model`; the modelling choices are discussed in
[Catalogs](../../guide/catalogs.md).

## `darksirens.redshift.catalog`

The EM-catalog redshift prior `p_cat(z | pix)`: each observed galaxy
contributes its normalised volumetric photo-z kernel, evaluated through a
windowed KDE over the z-sorted catalog rows.

```{eval-rst}
.. automodule:: darksirens.redshift.catalog
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.checks`

Non-JIT numerical checks of the redshift prior's normalisation, meant to run
once at startup or from tests.

```{eval-rst}
.. automodule:: darksirens.redshift.checks
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.completion`

The catalog completion model: the expected count `dN_exp/dz`, the completeness
estimator `C(z)` in its per-pixel, aggregate and parametric-selection forms,
and the `completion_curves` hot path plus its diagnostics.

```{eval-rst}
.. automodule:: darksirens.redshift.completion
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.grid`

The shared redshift grid (`zgrid`, `zMax`) used by every redshift prior and
completion model.

```{eval-rst}
.. automodule:: darksirens.redshift.grid
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.latent_counts`

The shell-total-conditioned angular count channel of the latent field: the
partial likelihood of the observed per-pixel, per-shell galaxy counts.

```{eval-rst}
.. automodule:: darksirens.redshift.latent_counts
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.latent_field`

The factored (sphere x z) latent-field basis, one construction shared by the
offline anchor builder and the in-likelihood latent seam.

```{eval-rst}
.. automodule:: darksirens.redshift.latent_field
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.lognormal_completion`

The offline builder of the LSS-conditioned lognormal completion field
`Q_LSS(p, z)`. Never imported by the GW likelihood, which consumes the fixed
arrays it writes.

```{eval-rst}
.. automodule:: darksirens.redshift.lognormal_completion
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.prior`

Redshift-prior assembly and the `get_redshift_prior` dispatch over the
spectral, complete-catalog, incomplete-catalog and bright-siren-counterpart
regimes, including the per-member Q evaluation.

```{eval-rst}
.. automodule:: darksirens.redshift.prior
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.selection`

Analytic magnitude-limited selection curves `C_sel(z; theta)` for the Gaussian
and Schechter luminosity functions, plus the offline maximum-likelihood fit
behind `darksirens_fit_selection`.

```{eval-rst}
.. automodule:: darksirens.redshift.selection
   :members:
   :undoc-members:
   :show-inheritance:
```

## `darksirens.redshift.volume`

The comoving-volume prior `p(z)` proportional to `dV_c/dz`, the agnostic
redshift prior of spectral-siren runs.

```{eval-rst}
.. automodule:: darksirens.redshift.volume
   :members:
   :undoc-members:
   :show-inheritance:
```
