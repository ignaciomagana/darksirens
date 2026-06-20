# Sky-anisotropy models (`darksirens.sky`)

The `sky` subpackage supplies the **mean-one** angular (or 3-D) factor
$g(\hat n, z)$ in the source rate $R(\theta, z, \hat n) = R_{\rm pop}(\theta,z)\,
g(\hat n, z)$, so isotropy/homogeneity is exactly $g\equiv 1$ and the shape does
not trade off against the overall rate $R_0$. Because $g$ enters the shared
per-sample weight, it is applied identically to the PE and selection terms and
the detector's own anisotropy divides out (see
[Theory & methods](../theory.md)).

## `darksirens.sky`

The package `__init__` exposes the `sky_model_parser` registry entry used by the
likelihood and the fiducial helpers.

```{automodule} darksirens.sky
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.sky.models`

The model classes, each implementing the protocol
`log_g_sky(nx, ny, nz, z, theta)` (purely-angular models ignore $z$):

- `IsotropicSky` — $g\equiv 1$, a bit-for-bit no-op.
- `DipoleSky` — $g = 1 + \hat n\cdot\mathbf d$, mean-one by construction; isotropy
  $\Leftrightarrow \mathbf d = 0$.
- `MultipoleSky` — $g = 1 + \sum_{\ell=1}^{\ell_{\max}}\sum_m a_{\ell m}
  Y_{\ell m}(\hat n)$ using orthonormal real Cartesian harmonics; mean-one is
  automatic ($\ell\ge1$ integrate to zero) and the angular power spectrum is
  $C_\ell = \sum_m a_{\ell m}^2$.
- `SphereGPSky` — a log-Gaussian random field on $S^2$,
  $g = e^{f(\hat n)}/\langle e^f\rangle$, with a whitened finite-rank GP
  (chordal-distance RBF kernel, Fibonacci-sphere inducing nodes); $\xi=0
  \Rightarrow f\equiv 0 \Rightarrow g\equiv1$.
- `SphereZGPSky` / `OverdensityGP3D` — a $(\text{sphere}\times z)$ GP sharing the
  product kernel
  $k = a^2\,e^{-\frac12\lVert\hat n-\hat n'\rVert^2/\ell_\Omega^2}\,
  e^{-\frac12(\zeta-\zeta')^2/\ell_z^2}$ with $\zeta=\ln(1+z)$ and
  Fibonacci-sphere $\times$ $z$-node inducing points; they differ only in the
  normalisation (per $z$-shell vs over the comoving volume). The whitened field
  $f(x_*) = k(x_*, Z)\,L^{-\top}\xi$ is the **same construction reused by the
  gp3d completion builder** ([`em.lognormal_completion`](em.md)).

```{automodule} darksirens.sky.models
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.sky.registry`

Maps `--sky_model` names to model factories (`SKY_MODEL_NAMES`,
`SKY_MODEL_LATEX`), provides `sky_model_parser` and the generic `sky_fiducial`
that zeroes all sky parameters so the isotropic fiducial is exact regardless of
prior centring.

```{automodule} darksirens.sky.registry
:members:
:undoc-members:
:show-inheritance:
```

## `darksirens.sky.analyze`

Post-processing for sky runs: `summarize_dipole_posterior` /
`plot_dipole_posterior` (amplitude + Mollweide direction),
`summarize_multipole_posterior` / `plot_multipole_cl` (the $C_\ell$ spectrum),
and `sphere_gp_posterior_map` / `plot_sphere_gp_map` (the posterior-mean
$g(\hat n, z_0)$ HEALPix map at chosen $z$-slices).

```{automodule} darksirens.sky.analyze
:members:
:undoc-members:
:show-inheritance:
```
