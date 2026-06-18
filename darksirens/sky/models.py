r"""
models.py
---------
Sky-distribution models: the angular (or 3-D) factor ``g(n̂)`` / ``g(n̂, z)`` in
the GW source rate ``R(θ, z, n̂) = R_pop(θ, z) · g(n̂, z)``.

Each model exposes the same duck type the likelihood consumes (mirroring the
population-model protocol in :mod:`darksirens.gw.populations`):

* ``param_specs`` — list of :class:`~darksirens.gw.populations.base.ParamSpec`.
* ``prior_bounds()`` — ``(lows, highs, labels)`` via ``pack_specs``.
* ``log_g_sky(nx, ny, nz, z, theta)`` — log of the density at sky direction
  ``n̂ = (nx, ny, nz)`` and redshift ``z`` (purely-angular models ignore ``z``),
  broadcasting over the sample axis, with ``theta`` the model's flat parameter
  sub-vector.

Models
~~~~~~
* ``isotropic`` / ``dipole`` / ``sphere_gp`` — purely angular ``g(n̂)``.
* ``sphere_gp_z`` (:class:`SphereZGPSky`) — a (sphere × z) GP, **normalised per
  z-shell** (``∫ g dΩ/4π = 1`` at every z): directional anisotropy that may
  evolve with distance, leaving the marginal redshift law untouched.
* ``overdensity_gp`` (:class:`OverdensityGP3D`) — the same (sphere × z) GP but
  **normalised over the comoving volume**: a full 3-D over/under-density
  (angular *and* radial clustering) of the source rate.

Normalisation convention
~~~~~~~~~~~~~~~~~~~~~~~~~~
``g`` is *mean-one*, so isotropy/homogeneity is exactly ``g ≡ 1`` and the shape
does not trade off with the overall rate ``R0`` (which the selection term
marginalises).  The ``dipole`` is mean-one by construction; the GP models divide
``exp(f)`` by its average — over the sphere (``sphere_gp``), per z-shell
(``sphere_gp_z``), or over the comoving volume (``overdensity_gp``) — estimated
on a fixed Fibonacci sphere quadrature (and a redshift grid for the 3-D models).

The GP fields use the whitened finite-rank construction of the population GP
(:mod:`darksirens.gw.populations.gp`): ``K = chol``, whitened ``xi``,
``f(x*) = k(x*, Z) alpha``.  Kernels are computed directly here (chordal-distance
RBF on the 3-D unit vectors × an RBF on ``ζ = log1p(z)``) — no ``tinygp``
dependency.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.scipy.special import logsumexp

from darksirens.gw.populations.base import ParamSpec, pack_specs

# Field clip mirrors the population GP: keeps exp(f) finite under wide xi.
_FIELD_CLIP = 10.0
_JITTER_REL = 1e-4
_JITTER_ABS = 1e-9   # absolute floor; product kernels are more degeneracy-prone
# Redshift normalisation grid for the 3-D models (cf. gp.py _ZNORM_N/_ZNORM_HI).
_ZNORM_N = 24
_ZNORM_HI = 3.0


def _sphere_rbf(A: jnp.ndarray, B: jnp.ndarray, amp: jnp.ndarray, ls: jnp.ndarray):
    """Chordal-distance RBF kernel between unit-vector sets ``A`` (Na,3) and
    ``B`` (Nb,3): ``amp^2 exp(-||n̂_A - n̂_B||^2 / 2ℓ^2)``.  For unit vectors the
    chordal distance squared is ``2(1 - n̂_A·n̂_B)``, so this is an isotropic
    (rotation-invariant) kernel on S² — the natural sphere analog of the RBF.
    """
    cos = A @ B.T
    d2 = jnp.clip(2.0 - 2.0 * cos, 0.0, 4.0)
    return amp ** 2 * jnp.exp(-0.5 * d2 / ls ** 2)


def _fibonacci_sphere(n: int) -> jnp.ndarray:
    """``(n, 3)`` near-uniform unit vectors on S² (golden-angle spiral).

    Used both for the GP inducing nodes and for the sphere-average quadrature
    that normalises ``g``.  Deterministic (no RNG), so it is JIT/trace-safe.
    """
    i = jnp.arange(n, dtype=jnp.float64)
    golden = jnp.pi * (3.0 - jnp.sqrt(5.0))
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = jnp.sqrt(jnp.clip(1.0 - z * z, 0.0, 1.0))
    phi = golden * i
    return jnp.stack([r * jnp.cos(phi), r * jnp.sin(phi), z], axis=-1)


class IsotropicSky:
    """Null model: ``g ≡ 1`` (no free parameters).  ``log_g_sky`` is identically
    zero, so an isotropic run is a bit-for-bit no-op relative to the legacy
    (sky-free) likelihood."""

    @property
    def param_specs(self):
        return []

    def prior_bounds(self):
        return [], [], []

    def log_g_sky(self, nx, ny, nz, z, theta):
        return jnp.zeros_like(nx)


class DipoleSky:
    r"""Pure dipole ``g(n̂) = 1 + n̂·d``, parameterised by the Cartesian dipole
    vector ``d = (d_x, d_y, d_z)`` (Isi, Farr & Varma 2023).  Isotropy ⟺ ``d=0``;
    the amplitude is ``|d|`` and the preferred direction is ``d/|d|``.  Mean-one
    by construction; positivity (``g ≥ 0``) holds for ``|d| ≤ 1`` and is enforced
    pointwise (``g ≤ 0`` ⇒ ``-inf``)."""

    @property
    def param_specs(self):
        return [
            ParamSpec(r"$d_x$", -1.0, 1.0, name="sky_dx"),
            ParamSpec(r"$d_y$", -1.0, 1.0, name="sky_dy"),
            ParamSpec(r"$d_z$", -1.0, 1.0, name="sky_dz"),
        ]

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_g_sky(self, nx, ny, nz, z, theta):
        dx, dy, dz = theta[0], theta[1], theta[2]
        g = 1.0 + nx * dx + ny * dy + nz * dz
        return jnp.where(g > 0.0, jnp.log(jnp.where(g > 0.0, g, 1.0)), -jnp.inf)


class SphereGPSky:
    r"""Log-Gaussian random field on the sphere (Essick et al. 2023):
    ``g(n̂) = exp(f(n̂)) / ⟨exp f⟩_{S²}`` with ``f`` a zero-mean GP whose RBF
    kernel acts on the 3-D unit vectors (chordal distance).  Whitened latents
    ``xi ~ N(0, I)`` carry ``prior_kind="normal"`` so the sampler sees clean
    unit-scale geometry; ``xi = 0`` ⇒ ``f ≡ 0`` ⇒ ``g ≡ 1`` (exact isotropy).

    The length scale is in chordal units (the distance between two unit vectors
    lies in ``[0, 2]``); a chordal length ``ℓ`` corresponds to an angular
    correlation length ``≈ 2 arcsin(ℓ/2)``.  ``xi = 0`` gives ``f ≡ 0`` ⇒
    ``g ≡ 1`` (exact isotropy), independent of the hyperparameters.
    """

    def __init__(
        self,
        n_inducing: int = 48,
        n_quad: int = 192,
        log_amp_bounds: tuple[float, float] = (float(jnp.log(0.1)), float(jnp.log(3.0))),
        log_ls_bounds: tuple[float, float] = (float(jnp.log(0.15)), float(jnp.log(2.0))),
    ):
        self._M = int(n_inducing)
        self._Z = _fibonacci_sphere(self._M)          # (M, 3) inducing nodes
        self._Zq = _fibonacci_sphere(int(n_quad))     # (Q, 3) sphere quadrature
        self._log_amp_bounds = log_amp_bounds
        self._log_ls_bounds = log_ls_bounds

    @property
    def param_specs(self):
        lo_a, hi_a = self._log_amp_bounds
        lo_l, hi_l = self._log_ls_bounds
        specs = [
            ParamSpec(r"$\log A_{\rm sky}$", lo_a, hi_a, name="sky_log_amp"),
            ParamSpec(r"$\log \ell_{\rm sky}$", lo_l, hi_l, name="sky_log_ls"),
        ]
        for i in range(self._M):
            specs.append(
                ParamSpec(
                    rf"$\xi^{{\rm sky}}_{{{i}}}$",
                    -10.0,
                    10.0,
                    name=f"sky_xi_{i}",
                    prior_kind="normal",
                    prior_loc=0.0,
                    prior_scale=1.0,
                )
            )
        return specs

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_g_sky(self, nx, ny, nz, z, theta):
        amp = jnp.exp(theta[0])
        ls = jnp.exp(theta[1])
        xi = theta[2:2 + self._M]

        # Whitened finite-rank GP (same construction as the population GP):
        #   K = k(Z, Z) + jitter I,  L = chol(K),  alpha = L^{-T} xi,
        #   f(x*) = k(x*, Z) @ alpha     (so f(Z) = L xi exactly).
        jitter = _JITTER_REL * amp ** 2
        K = _sphere_rbf(self._Z, self._Z, amp, ls) + jitter * jnp.eye(self._M)
        L = jnp.linalg.cholesky(K)
        alpha = jsl.solve_triangular(L, xi, lower=True, trans=1)

        coords = jnp.stack([nx, ny, nz], axis=-1)              # (N, 3)
        f = _sphere_rbf(coords, self._Z, amp, ls) @ alpha      # (N,)
        fq = _sphere_rbf(self._Zq, self._Z, amp, ls) @ alpha   # (Q,)

        f = jnp.clip(f, -_FIELD_CLIP, _FIELD_CLIP)
        fq = jnp.clip(fq, -_FIELD_CLIP, _FIELD_CLIP)

        # log ⟨exp f⟩ over the sphere ≈ log mean over uniform quadrature points,
        # so g = exp(f) / ⟨exp f⟩ integrates to 1 over the sphere (mean one).
        log_norm = logsumexp(fq) - jnp.log(fq.shape[0])
        return f - log_norm


def _sphere_z_kernel(A_n, A_z, B_n, B_z, amp, ls_sph, ls_z):
    """Product kernel ``amp² · k_Ω(n̂,n̂'; ls_sph) · k_z(ζ,ζ'; ls_z)`` over
    (sphere × redshift), with ``k_Ω`` the chordal-distance RBF on unit vectors
    and ``k_z`` a 1-D RBF on ``ζ = log1p(z)``.

    ``A_n`` (Na,3) / ``A_z`` (Na,) and ``B_n`` (Nb,3) / ``B_z`` (Nb,) are the
    paired sphere/redshift coordinates; returns ``(Na, Nb)``.
    """
    cos = A_n @ B_n.T
    d2 = jnp.clip(2.0 - 2.0 * cos, 0.0, 4.0)
    ksph = jnp.exp(-0.5 * d2 / ls_sph ** 2)
    dz = A_z[:, None] - B_z[None, :]
    kz = jnp.exp(-0.5 * dz ** 2 / ls_z ** 2)
    return amp ** 2 * ksph * kz


class _SphereZGPBase:
    r"""Shared (sphere × z) whitened finite-rank GP field ``f(n̂, z)``.

    The density is ``g(n̂, z) = exp(f(n̂, z)) / Z``; subclasses differ **only** in
    the normalisation ``Z`` (per z-shell vs. over the comoving volume).  ``f`` is
    a zero-mean GP with the product kernel :func:`_sphere_z_kernel` over inducing
    nodes = (Fibonacci sphere) × (redshift grid), whitened by standard-normal
    latents ``xi``; ``xi = 0`` ⇒ ``f ≡ 0`` ⇒ ``g ≡ 1`` (exact
    isotropy+homogeneity), independent of the hyperparameters.
    """

    def __init__(
        self,
        n_inducing_sphere: int = 32,
        n_inducing_z: int = 6,
        n_quad: int = 192,
        z_node_hi: float = 3.0,
        log_amp_bounds: tuple[float, float] = (float(jnp.log(0.1)), float(jnp.log(3.0))),
        log_ls_sphere_bounds: tuple[float, float] = (float(jnp.log(0.15)), float(jnp.log(2.0))),
        log_ls_z_bounds: tuple[float, float] = (float(jnp.log(0.05)), float(jnp.log(2.0))),
    ):
        self._M_sph = int(n_inducing_sphere)
        self._M_z = int(n_inducing_z)
        self._M = self._M_sph * self._M_z
        # Flattened product inducing nodes, ordering i = i_sph * M_z + i_z.
        Z_sph = _fibonacci_sphere(self._M_sph)                       # (M_sph, 3)
        zeta_nodes = jnp.linspace(0.0, float(jnp.log1p(z_node_hi)), self._M_z)
        self._Zn = jnp.repeat(Z_sph, self._M_z, axis=0)              # (M, 3)
        self._Zz = jnp.tile(zeta_nodes, self._M_sph)                 # (M,)
        self._Zq = _fibonacci_sphere(int(n_quad))                    # (Q, 3) sphere quad
        self._Q = int(n_quad)
        self._zg = jnp.linspace(0.0, _ZNORM_HI, _ZNORM_N)           # (Nzg,) physical z
        self._zeta_g = jnp.log1p(self._zg)                          # (Nzg,)
        self._log_amp_bounds = log_amp_bounds
        self._log_ls_sphere_bounds = log_ls_sphere_bounds
        self._log_ls_z_bounds = log_ls_z_bounds

    @property
    def param_specs(self):
        lo_a, hi_a = self._log_amp_bounds
        lo_s, hi_s = self._log_ls_sphere_bounds
        lo_z, hi_z = self._log_ls_z_bounds
        specs = [
            ParamSpec(r"$\log A_{\rm sky}$", lo_a, hi_a, name="sky_log_amp"),
            ParamSpec(r"$\log \ell^{\rm sky}_\Omega$", lo_s, hi_s, name="sky_log_ls_sphere"),
            ParamSpec(r"$\log \ell^{\rm sky}_z$", lo_z, hi_z, name="sky_log_ls_z"),
        ]
        for i in range(self._M):
            specs.append(
                ParamSpec(
                    rf"$\xi^{{\rm sky}}_{{{i}}}$",
                    -10.0,
                    10.0,
                    name=f"sky_xi_{i}",
                    prior_kind="normal",
                    prior_loc=0.0,
                    prior_scale=1.0,
                )
            )
        return specs

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def _field(self, nx, ny, nz, z, theta):
        """Return ``(f, fq)``: the field at the query points ``(N,)`` and the
        normalisation field ``fq`` on the (z-grid × sphere-quad) grid ``(Nzg, Q)``.

        ``fq`` is built from factored sphere/z kernels via an ``einsum`` so the
        ``(Nzg, Q, M)`` tensor is never materialised.
        """
        amp = jnp.exp(theta[0])
        ls_sph = jnp.exp(theta[1])
        ls_z = jnp.exp(theta[2])
        xi = theta[3:3 + self._M]

        jitter = _JITTER_REL * amp ** 2 + _JITTER_ABS
        K = _sphere_z_kernel(self._Zn, self._Zz, self._Zn, self._Zz, amp, ls_sph, ls_z)
        K = K + jitter * jnp.eye(self._M)
        L = jnp.linalg.cholesky(K)
        alpha = jsl.solve_triangular(L, xi, lower=True, trans=1)     # (M,) = L^{-T} xi

        # Field at the query samples.  clip(z, 0) keeps NaN (out-of-grid dL) as
        # NaN, which propagates to a non-finite log_g and is dropped by the
        # likelihood's ``valid & isfinite`` mask — do NOT nan_to_num here.
        zeta = jnp.log1p(jnp.clip(z, 0.0, None))                     # (N,)
        coords = jnp.stack([nx, ny, nz], axis=-1)                    # (N, 3)
        Kq = _sphere_z_kernel(coords, zeta, self._Zn, self._Zz, amp, ls_sph, ls_z)  # (N, M)
        f = jnp.clip(Kq @ alpha, -_FIELD_CLIP, _FIELD_CLIP)          # (N,)

        # Normalisation field on (z-grid × sphere-quad): factor the product
        # kernel and contract with alpha (never form (Nzg, Q, M)).
        cosq = self._Zq @ self._Zn.T                                 # (Q, M)
        ksph_q = jnp.exp(-0.5 * jnp.clip(2.0 - 2.0 * cosq, 0.0, 4.0) / ls_sph ** 2)
        dzg = self._zeta_g[:, None] - self._Zz[None, :]              # (Nzg, M)
        kz_g = jnp.exp(-0.5 * dzg ** 2 / ls_z ** 2)
        fq = amp ** 2 * jnp.einsum("qm,gm,m->gq", ksph_q, kz_g, alpha)  # (Nzg, Q)
        fq = jnp.clip(fq, -_FIELD_CLIP, _FIELD_CLIP)
        return f, fq

    def log_g_sky(self, nx, ny, nz, z, theta):
        f, fq = self._field(nx, ny, nz, z, theta)
        return f - self._log_norm(fq, z)

    def _log_norm(self, fq, z):
        raise NotImplementedError


class SphereZGPSky(_SphereZGPBase):
    r"""(sphere × z) GP normalised **per z-shell**: ``∫ g(n̂,z) dΩ/4π = 1`` for
    every z.  Captures directional anisotropy that may evolve with distance while
    leaving the marginal redshift distribution to the population + ``p(z|pix)``
    terms (orthogonal, identifiable).
    """

    def _log_norm(self, fq, z):
        # log ⟨exp f(·, z_g)⟩_sphere on the z-grid, interpolated to the query z
        # (the gp.py ``_znorm_interp`` idiom).  NaN z → NaN (passes through).
        log_norm_g = logsumexp(fq, axis=1) - jnp.log(self._Q)        # (Nzg,)
        return jnp.interp(z, self._zg, log_norm_g)                   # (N,)


class OverdensityGP3D(_SphereZGPBase):
    r"""(sphere × z) GP normalised over the **comoving volume**: a full 3-D
    over/under-density of the source rate (angular *and* radial clustering).

    The homogeneous reference is uniform in comoving volume, so ``exp(f)`` is
    averaged with a fiducial-cosmology ``dV_c/dz`` weight on the z-grid.  Because
    the sphere-average of ``g`` may then vary with z, this model can reshape the
    marginal redshift distribution and is therefore partially degenerate with the
    population redshift slope ``γ`` and the catalog ``p(z|pix)`` — use with ``γ``
    fixed for an identifiable clustering measurement.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Normalised log comoving-volume weights on the z-grid (the homogeneous
        # reference), using the code's fiducial cosmology.
        from darksirens.utils.cosmology import (
            dV_of_z, H0Planck, Om0Planck, w0Fiducial, waFiducial,
        )
        w = jnp.asarray(dV_of_z(self._zg, H0Planck, Om0Planck, w0Fiducial, waFiducial))
        w = jnp.maximum(w, 0.0)
        self._log_vol_w = jnp.log(w) - jnp.log(jnp.sum(w))          # (Nzg,)

    def _log_norm(self, fq, z):
        # Single scalar: log ⟨exp f⟩ over (sphere × volume-weighted z).  Broadcast
        # to all query samples.  xi=0 ⇒ fq=0 ⇒ this is log(Σ_g w_g) = 0.
        terms = fq + self._log_vol_w[:, None] - jnp.log(self._Q)     # (Nzg, Q)
        return logsumexp(terms)                                      # scalar
