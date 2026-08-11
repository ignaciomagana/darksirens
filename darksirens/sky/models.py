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

import math
import os

import numpy as np

import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.scipy.special import logsumexp

from darksirens.gw.populations.base import ParamSpec, pack_specs
from darksirens.redshift.grid import zMax as _ZMAX_GRID

# Field clip mirrors the population GP: keeps exp(f) finite under wide xi.
_FIELD_CLIP = 10.0
_JITTER_REL = 1e-4
_JITTER_ABS = 1e-9   # absolute floor; product kernels are more degeneracy-prone
# Redshift normalisation grid for the 3-D models.
#
# The normaliser is tabulated on [0, _ZNORM_HI] and evaluated with jnp.interp,
# which CLAMPS above the grid: any z > _ZNORM_HI silently reuses the norm at the
# edge, so the models' defining contract (the sphere -- or volume -- average of
# g is 1 at every z) is violated there.  This ceiling therefore has to track the
# SHARED analysis redshift grid, ``zMax`` from darksirens.redshift.grid (set by
# DARKSIRENS_ZMAX, default 5.0); the old hardcoded 3.0 silently froze z in
# (3, 5], which is reachable because likelihood/core.py evaluates log_g_sky at
# z_of_dL over the full grid and z_of_dL grows with H0 across its prior.
#
# This is the same defect and the same remedy as gw/populations/gp.py
# (library-review SEV-3); the fix was applied there and not here, and the
# "cf. gp.py" note that used to sit on these constants hid that.  ``_ZNORM_N``
# tracks ``_ZNORM_HI`` to hold node density fixed (8 per unit z: 40 nodes at the
# z=5 default, 24 at the 3.0 floor); for the (sphere x z) models it is only a
# FLOOR -- they size their own grid to the z-kernel's shortest prior length scale
# (see ``_SphereZGPBase.__init__``), which the 8-per-unit-z density does not
# resolve.
#: Sphere-average quadrature size for the mean-one normalisation of the GP sky
#: models.  192 nodes under-resolve a high-amplitude, short-length-scale field:
#: at the PRIOR CORNER (log_amp at its max, log_ls_sphere at its min) the sphere
#: average of g departed from 1 by 25%, breaking the models' defining contract
#: inside the sampled prior.  Convergence against an independent 8192-point
#: sphere is non-monotonic (Fibonacci-lattice aliasing against the field's own
#: structure) -- 192: 0.056, 384: 0.195, 768: 0.063, 1536: 0.0045, 3072: 0.0044
#: -- so 1536 is the first count that is reliably converged rather than merely
#: better.  Cost is one (Q, M) kernel matmul per proposal.
_SPHERE_NQ = 1536

if "DARKSIRENS_SKY_ZNORM_HI" in os.environ:
    _ZNORM_HI = float(os.environ["DARKSIRENS_SKY_ZNORM_HI"])
else:
    _ZNORM_HI = max(3.0, float(_ZMAX_GRID))
_ZNORM_N = max(24, int(round(8.0 * _ZNORM_HI)))


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
    by construction; positivity (``g ≥ 0`` everywhere on the sphere) requires
    ``|d| ≤ 1`` **globally**, so the prior lives on the unit BALL: the nested
    samplers map their cube onto it (``constraint_groups`` -> the ``ball3``
    cube map in ``make_prior_transform``), and ``log_g_sky`` rejects the
    whole parameter point for ``|d| > 1`` as the backstop.

    The old pointwise truncation (``g ≤ 0`` ⇒ ``-inf`` per direction, box
    prior on the components) silently replaced the mean-one dipole with a
    DIFFERENT model at ``|d| > 1``: the positive part of the dipole, whose
    spherical mean exceeds one (~1.077 at ``d = (1,1,1)``) and whose density
    is non-smooth on the nodal circle — ~21% of the sphere zeroed at that
    corner while the model still claimed to be a mean-one dipole."""

    @property
    def param_specs(self):
        return [
            ParamSpec(r"$d_x$", -1.0, 1.0, name="sky_dx"),
            ParamSpec(r"$d_y$", -1.0, 1.0, name="sky_dy"),
            ParamSpec(r"$d_z$", -1.0, 1.0, name="sky_dz"),
        ]

    @property
    def constraint_groups(self):
        return (("ball3", (r"$d_x$", r"$d_y$", r"$d_z$")),)

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_g_sky(self, nx, ny, nz, z, theta):
        dx, dy, dz = theta[0], theta[1], theta[2]
        g = 1.0 + nx * dx + ny * dy + nz * dz
        # Global validity: |d| <= 1 keeps g >= 0 on the whole sphere (equality
        # only at the antipode of d when |d| = 1). Outside the ball the WHOLE
        # point is invalid — pointwise positive-part clipping is a different,
        # non-mean-one model (see the class docstring).
        valid = (dx * dx + dy * dy + dz * dz) <= 1.0
        logg = jnp.where(g > 0.0, jnp.log(jnp.where(g > 0.0, g, 1.0)), -jnp.inf)
        return jnp.where(valid, logg, -jnp.inf)


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
        n_quad: int = _SPHERE_NQ,
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
        n_quad: int = _SPHERE_NQ,
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
        # Normalisation grid laid out in the FIELD's own coordinate zeta =
        # log1p(z) and resolved to the z-kernel.  A grid uniform in PHYSICAL z
        # (40 nodes over [0, 5], dz = 0.128) is ~2.5x coarser than the shortest
        # correlation length the prior allows (ls_z down to 0.05 in zeta units),
        # so between nodes the interpolated normaliser was arbitrarily wrong and
        # the defining contract (sphere -- or volume -- average of g is 1 at
        # every z) failed by up to 116% at the prior corner, measured against an
        # independent 8192-point sphere.  Unlike a constant error in g, a
        # z-DEPENDENT one does not cancel between the per-event term and beta:
        # it is a spurious grid-periodic modulation of the effective redshift
        # distribution the sampler can trade against real structure (H0, gamma).
        # Spacing d_zeta <= ls_z_min/3 brings the same corner to ~3%; cost is
        # linear in Nzg in the (Nzg, Q) einsum, which is not the bottleneck.
        zeta_hi = math.log1p(_ZNORM_HI)
        n_zg = max(
            _ZNORM_N,
            int(math.ceil(3.0 * zeta_hi / float(np.exp(log_ls_z_bounds[0])))) + 1,
        )
        self._zeta_g = jnp.linspace(0.0, zeta_hi, n_zg)              # (Nzg,)
        self._zg = jnp.expm1(self._zeta_g)                           # (Nzg,) physical z
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
        # Interpolated in zeta = log1p(z), where the nodes are uniform and the
        # z-kernel acts, so the interpolant resolves the field's own structure.
        log_norm_g = logsumexp(fq, axis=1) - jnp.log(self._Q)        # (Nzg,)
        zeta = jnp.log1p(jnp.clip(z, 0.0, None))                    # (N,)
        return jnp.interp(zeta, self._zeta_g, log_norm_g)            # (N,)


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
        # The nodes are uniform in zeta = log1p(z), not in z, so the Riemann sum
        # over the volume needs the dz/dzeta = (1 + z) Jacobian; without it the
        # low-z shells are over-weighted and the volume average of g drifts from
        # 1 by up to 7% (measured against a dense reference quadrature).
        w = w * (1.0 + self._zg)
        self._log_vol_w = jnp.log(w) - jnp.log(jnp.sum(w))          # (Nzg,)

    def _log_norm(self, fq, z):
        # Single scalar: log ⟨exp f⟩ over (sphere × volume-weighted z).  Broadcast
        # to all query samples.  xi=0 ⇒ fq=0 ⇒ this is log(Σ_g w_g) = 0.
        terms = fq + self._log_vol_w[:, None] - jnp.log(self._Q)     # (Nzg, Q)
        return logsumexp(terms)                                      # scalar


# ============================================================
# Low-ℓ spherical-harmonic multipole model
# ============================================================

def _real_ylm(nx, ny, nz, lmax):
    """Orthonormalised real spherical harmonics ``Y_lm(n̂)`` for ℓ=1..lmax,
    stacked as ``(..., n_coeff)`` with ``n_coeff = (lmax+1)² − 1``.

    Explicit Cartesian-polynomial forms in the unit-vector components
    ``n̂ = (nx, ny, nz)``; normalised so ``∫ Y_lm Y_l'm' dΩ = δ`` (hence
    ``C_ℓ = Σ_m a_lm² / (2ℓ+1)`` is the power per mode).  Column order matches
    ``[(ℓ, m) for ℓ in 1..lmax for m in -ℓ..ℓ]``.
    """
    pi = math.pi
    cols = []
    if lmax >= 1:
        c1 = math.sqrt(3.0 / (4.0 * pi))
        cols += [c1 * ny, c1 * nz, c1 * nx]                       # m = -1, 0, +1
    if lmax >= 2:
        c2a = math.sqrt(15.0 / (4.0 * pi))                        # xy, yz, xz
        cols += [
            c2a * nx * ny,                                        # m = -2
            c2a * ny * nz,                                        # m = -1
            math.sqrt(5.0 / (16.0 * pi)) * (3.0 * nz**2 - 1.0),   # m =  0
            c2a * nx * nz,                                        # m = +1
            math.sqrt(15.0 / (16.0 * pi)) * (nx**2 - ny**2),      # m = +2
        ]
    if lmax >= 3:
        cols += [
            0.25 * math.sqrt(35.0 / (2.0 * pi)) * ny * (3.0 * nx**2 - ny**2),  # -3
            0.5 * math.sqrt(105.0 / pi) * nx * ny * nz,                        # -2
            0.25 * math.sqrt(21.0 / (2.0 * pi)) * ny * (5.0 * nz**2 - 1.0),    # -1
            0.25 * math.sqrt(7.0 / pi) * nz * (5.0 * nz**2 - 3.0),             #  0
            0.25 * math.sqrt(21.0 / (2.0 * pi)) * nx * (5.0 * nz**2 - 1.0),    # +1
            0.25 * math.sqrt(105.0 / pi) * nz * (nx**2 - ny**2),               # +2
            0.25 * math.sqrt(35.0 / (2.0 * pi)) * nx * (nx**2 - 3.0 * ny**2),  # +3
        ]
    return jnp.stack(cols, axis=-1)


def multipole_lm_indices(lmax):
    """``[(ℓ, m), ...]`` in the column order of :func:`_real_ylm` / the params."""
    return [(l, m) for l in range(1, int(lmax) + 1) for m in range(-l, l + 1)]


class MultipoleSky:
    r"""Low-order spherical-harmonic anisotropy: ``g(n̂) = 1 + Σ_{ℓ=1}^{ℓmax}
    Σ_m a_lm Y_lm(n̂)`` — the perturbative basis for *small* departures from
    isotropy (generalises the dipole, which is exactly ℓ=1).

    Mean-one by construction (every ℓ≥1 harmonic integrates to zero), so
    ``a_lm = 0 ⇒ g ≡ 1``; positivity (``g ≥ 0``) is enforced GLOBALLY on a
    fixed Fibonacci-sphere quadrature grid: a coefficient point whose field
    dips negative anywhere on the grid is rejected as a whole (``-inf``),
    defining a constrained prior on the box.  The old pointwise truncation
    silently swapped in a different model — the positive part of the field,
    non-mean-one and non-smooth on its nodal lines — whenever a corner of
    the box produced negative regions.  With the orthonormal harmonics the
    angular power spectrum is ``C_ℓ = Σ_m a_lm² / (2ℓ+1)``.  Purely angular — the
    ``z`` argument is ignored.

    Prior-volume caveat for the isotropy Bayes factor
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The sampler is handed a UNIFORM BOX prior on the coefficients and the
    positivity gate returns ``-inf`` outside the valid region, so the effective
    prior is uniform on (box ∩ positivity) while its normalisation is that of the
    box.  The reported ``logZ`` therefore carries an offset ``log(valid
    fraction)`` — measured ``-0.51`` nats for ``lmax=2`` and ``-3.35`` for
    ``lmax=3`` at the default ``a_bound=1.0`` — which is an artifact of the
    arbitrary bound, not of the data: it penalises ``multipole_l3`` by ~3 nats
    against isotropy and makes the two multipole models mutually incomparable.
    :meth:`prior_volume_fraction` measures it so the comparison can subtract it
    (it is also the prior-draw efficiency: ~97% of ``lmax=3`` draws are
    rejected).
    """

    # Directions of the global-positivity quadrature. ~2e3 quasi-uniform
    # points resolve the smooth ℓ <= 3 harmonics' minima to ~1e-3 in g —
    # adequate for a validity test on fields whose shortest angular scale is
    # ~60 degrees.
    _POSITIVITY_GRID_N = 2048

    def __init__(self, lmax: int = 2, a_bound: float = 1.0):
        if int(lmax) not in (1, 2, 3):
            raise ValueError("MultipoleSky supports lmax in {1, 2, 3}.")
        self._lmax = int(lmax)
        self._a_bound = float(a_bound)
        self._lm = multipole_lm_indices(self._lmax)
        self._n_coeff = len(self._lm)                 # (lmax+1)^2 - 1
        # Fibonacci sphere (golden-angle spiral): deterministic quasi-uniform
        # directions, no healpy dependency at import time.
        n = self._POSITIVITY_GRID_N
        k = np.arange(n, dtype=float)
        z_grid = 1.0 - (2.0 * k + 1.0) / n
        phi = k * (np.pi * (3.0 - np.sqrt(5.0)))
        s = np.sqrt(np.maximum(1.0 - z_grid * z_grid, 0.0))
        self._Y_positivity = jnp.asarray(np.asarray(
            _real_ylm(
                jnp.asarray(s * np.cos(phi)),
                jnp.asarray(s * np.sin(phi)),
                jnp.asarray(z_grid),
                self._lmax,
            )
        ))

    @property
    def param_specs(self):
        b = self._a_bound
        return [
            ParamSpec(rf"$a_{{{l},{m}}}$", -b, b, name=f"sky_a_l{l}_m{m}")
            for (l, m) in self._lm
        ]

    def prior_volume_fraction(self, n_draws: int = 20000, seed: int = 0) -> float:
        """Fraction of the ``a_lm`` box that survives the global positivity gate.

        This is the offset the reported ``logZ`` carries relative to a properly
        normalised constrained prior (see the class docstring): subtract
        ``log(prior_volume_fraction())`` from ``logZ`` before forming the
        isotropy Bayes factor, or before comparing ``multipole`` with
        ``multipole_l3``.  Estimated by Monte Carlo on the model's OWN positivity
        grid, so it measures exactly the region ``log_g_sky`` accepts;
        deterministic in ``seed`` and cached for the default arguments.
        """
        cache = getattr(self, "_prior_volume_cache", None)
        if cache is not None and cache[0] == (int(n_draws), int(seed)):
            return cache[1]
        Y = np.asarray(self._Y_positivity)                    # (Ngrid, n_coeff)
        rng = np.random.default_rng(seed)
        n_valid = 0
        remaining = int(n_draws)
        while remaining > 0:                                  # chunked: the
            chunk = min(2048, remaining)                      # (draws, Ngrid)
            a = rng.uniform(-self._a_bound, self._a_bound,     # product would be
                            size=(chunk, self._n_coeff))       # hundreds of MB
            n_valid += int(np.count_nonzero(1.0 + (a @ Y.T).min(axis=1) >= 0.0))
            remaining -= chunk
        fraction = n_valid / float(n_draws)
        self._prior_volume_cache = ((int(n_draws), int(seed)), fraction)
        return fraction

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def log_g_sky(self, nx, ny, nz, z, theta):
        Y = _real_ylm(nx, ny, nz, self._lmax)         # (N, n_coeff)
        g = 1.0 + Y @ theta                           # (N,)
        # Global positivity on the quadrature grid: reject the WHOLE
        # coefficient point when the field dips negative anywhere, instead of
        # zeroing that region pointwise (a different, non-mean-one model).
        min_g = 1.0 + jnp.min(self._Y_positivity @ theta)
        logg = jnp.where(g > 0.0, jnp.log(jnp.where(g > 0.0, g, 1.0)), -jnp.inf)
        return jnp.where(min_g >= 0.0, logg, -jnp.inf)
