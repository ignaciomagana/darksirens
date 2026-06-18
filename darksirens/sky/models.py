r"""
models.py
---------
Sky-distribution models: the angular factor ``g(n̂)`` in the GW source rate
``R(θ, z, n̂) = R_pop(θ, z) · g(n̂)``.

Each model exposes the same duck type the likelihood consumes (mirroring the
population-model protocol in :mod:`darksirens.gw.populations`):

* ``param_specs`` — list of :class:`~darksirens.gw.populations.base.ParamSpec`.
* ``prior_bounds()`` — ``(lows, highs, labels)`` via ``pack_specs``.
* ``log_g_sky(nx, ny, nz, theta)`` — log of the angular density evaluated at the
  unit sky-direction components ``n̂ = (nx, ny, nz)`` (broadcasting over the
  sample axis), with ``theta`` the model's flat parameter sub-vector.

Normalisation convention
~~~~~~~~~~~~~~~~~~~~~~~~~~
``g`` is a *mean-one* density on the sphere, ``∫ g dΩ / 4π = 1``, so isotropy is
exactly ``g ≡ 1`` and the angular shape does not trade off with the overall
rate ``R0`` (which the selection term marginalises).  The ``dipole`` is mean-one
by construction (a pure ``ℓ=1`` harmonic integrates to zero); the ``sphere_gp``
divides ``exp(f)`` by its sphere average, estimated on a fixed Fibonacci
quadrature.

The ``sphere_gp`` field uses the same whitened finite-rank GP construction as
the population GP (:mod:`darksirens.gw.populations.gp`): ``K = chol``, whitened
``xi``, ``f(x*) = k(x*, Z) alpha``.  The kernel is the chordal-distance RBF on
the 3-D unit vectors (an isotropic kernel on S²), computed directly here — no
``tinygp`` dependency.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.scipy.linalg as jsl
from jax.scipy.special import logsumexp

from darksirens.gw.populations.base import ParamSpec, pack_specs

# Field clip mirrors the population GP: keeps exp(f) finite under wide xi.
_FIELD_CLIP = 10.0
_JITTER_REL = 1e-4


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

    def log_g_sky(self, nx, ny, nz, theta):
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

    def log_g_sky(self, nx, ny, nz, theta):
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

    def log_g_sky(self, nx, ny, nz, theta):
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
