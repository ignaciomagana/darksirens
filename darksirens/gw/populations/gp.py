r"""Gaussian-process population models (true-GP prior, mixture-decoupled).

This module provides *standalone* compact-binary population models whose
source-parameter density is a Gaussian-process prior over one or more of
``(m1, q, chi_eff, z)``.  They are registered with :func:`register_model` and
therefore bypass the stick-breaking ``MixtureModel`` grammar entirely: each
model exposes the population-model duck type consumed by the likelihood --
``param_specs``, ``prior_bounds()`` and ``log_p_pop(m1, q, z, chieff, theta)``.

Why this replaces the old node-conditioned classes
--------------------------------------------------
The previous GP classes sampled latent node *values* ``y`` from uniform box
priors and called ``tinygp.condition(y, ...)``.  With a flat prior on ``y`` and
no ``log N(y; mu, K)`` term, the kernel only governed interpolation between free
nodes -- it was *not* a GP prior.  These models instead impose a genuine GP
prior via a whitened (non-centred) parameterisation.

Construction (whitened, rank-M, no interpolation)
-------------------------------------------------
For the GP axes with inducing grid ``Z`` (in coordinate space) and standard
normal latent ``xi``::

    K     = amp^2 * prod_axis kernel_axis(Z, Z) + jitter
    L     = chol(K)
    alpha = L^{-T} xi                         # one triangular solve
    f(x*) = mu(x*) + k(x*, Z) @ alpha

This is the standard finite-dimensional ("deterministic inducing point") GP
prior, the same representation used by the binned-GP (Ray et al. 2023,
ApJ 957, 37) and B-spline (Edelman et al. 2023) population analyses: an exact
GP draw at the inducing points and a smooth RKHS conditional mean elsewhere,
evaluated at the exact query points with no interpolation grid.  Because
``xi ~ N(0, I)`` carries ``prior_kind="normal"``, NUTS samples the clean,
unit-scale Gaussian geometry the whitening is designed to expose.

Redshift convention
-------------------
``z`` is always a *conditioning / rate* axis, never a probability axis: the
GP is normalised only over the probability axes ``{m1, q, chi}`` that it spans
(conditional on ``z`` when ``z`` is a GP axis), and the marginal redshift
evolution is always the parametric ``(1 + z)**(gamma - 1)`` rate term.  A GP
on ``z`` therefore adds *conditional shape evolution* (e.g. a mass function
that evolves with redshift) on top of the parametric rate.

Import note
-----------
``tinygp`` is imported lazily inside the field evaluation so that importing
this module (for registration, parameter specs, or fiducials) never imports
``tinygp``.  Only ``log_p_pop`` requires it.
"""

from __future__ import annotations

import os
import math

from dataclasses import dataclass, field
from typing import Sequence

import jax.numpy as jnp
import jax.scipy.linalg as jsl

from darksirens.redshift.grid import zMax

from .base import ParamSpec, pack_specs
from .utils import (
    get_mass_grid,
    get_q_grid,
    get_chi_grid,
    normalization_grid_settings,
    sfilter_low,
    sfilter_high,
)


# ============================================================
# Axis coordinate transforms and default inducing geometry
# ============================================================

AXIS_ORDER = ("m1", "q", "chi", "z")          # canonical ordering
_PROB_AXES = ("m1", "q", "chi")               # normalised over these
_COND_AXES = ("z",)                           # conditioning / rate axis

_LOGSAFE = 1e-12
_FIELD_CLIP = 10.0
_JITTER_REL = 1e-4

# Normalisation grids.  When z is a GP axis the normalisation integral over the
# probability axes is a smooth 1-D function of z, so it is evaluated on a small
# z-grid and interpolated (cost independent of the number of query points).  For
# multi-axis (>=2) probability integrals under z-conditioning a dedicated coarse
# grid keeps the (z-grid x prob-grid) tensor tractable; single-axis and
# z-independent integrals use the full normalisation grids for accuracy.
_KD_N = {"m1": 48, "q": 24, "chi": 24}     # coarse per-axis sizes for k-D norm
_KD_SPAN = {"m1": (2.0, 100.0), "q": (0.02, 1.0), "chi": (-1.0, 1.0)}
_M1NORM_N = 64                             # log-m1 nodes for m1-conditional q-norm
# The z-normalisation interpolation grid is FROZEN above _ZNORM_HI: GP model
# evaluations at z > _ZNORM_HI silently reuse the norm at the grid edge
# (library-review SEV-3). Parametric models are unaffected (analytic in z).
# Coupling: the ceiling tracks the SHARED analysis redshift grid -- ``zMax`` from
# :mod:`darksirens.redshift.grid` (itself set by DARKSIRENS_ZMAX, default 5.0) --
# so the GP normaliser can never freeze below the redshifts the rest of the
# pipeline samples (the old default of 3.0 silently froze z in (3, 5]).
# DARKSIRENS_GP_ZNORM_HI still overrides explicitly; ``_ZNORM_N`` tracks
# ``_ZNORM_HI`` to hold node density fixed (8/z-unit: 40 nodes at the z=5
# default, 24 at the 3.0 floor).
if "DARKSIRENS_GP_ZNORM_HI" in os.environ:
    _ZNORM_HI = float(os.environ["DARKSIRENS_GP_ZNORM_HI"])
else:
    _ZNORM_HI = max(3.0, float(zMax))
_ZNORM_N = max(24, int(round(8.0 * _ZNORM_HI)))


def _coarse_axis_grid(axis: str):
    lo, hi = _KD_SPAN[axis]
    return jnp.linspace(lo, hi, _KD_N[axis])


def _m1norm_grid():
    """Log-spaced m1 nodes for the m1-conditional q-normalisation.

    The four ``q``-in-GP / ``m1``-not-in-GP models normalise the GP factor over
    ``q`` (and ``chi``) *conditional on m1*: the ``m2 = q*m1`` secondary cut ties
    the taper to the query m1, so the norm is a smooth 1-D function of m1 (like
    the z-conditioning idiom).  It is tabulated on these nodes once per proposal
    and interpolated in ``log m1`` per query (nodes are log-uniform).  The span
    matches the mass normalisation grid (floored at 2.0, the ``m_min`` prior
    lower bound, so every node can host a physical m2 = q*m1 >= m_min).
    """
    s = normalization_grid_settings()
    return jnp.exp(jnp.linspace(jnp.log(max(s.m_lo, 2.0)),
                                jnp.log(s.m_hi), _M1NORM_N))


def _interp2_log(ltab, zg, m1g, z_q, m1_q):
    """Bilinear, edge-clamped interpolation of a ``(Nz, Nm1)`` log-table.

    Evaluates ``exp(bilerp(ltab))`` at the *paired* queries ``(z_q, m1_q)``
    (both 1-D, same length) using ``jnp.searchsorted`` + clipped linear weights.
    Memory is O(N_query) -- the table is gathered at the bracketing nodes, never
    outer-producted against the queries -- and the map is differentiable exactly
    like ``jnp.interp`` (grad flows through the table values and the edge-clamped
    weights).  The m1 axis is interpolated in ``log m1`` because ``m1g`` is
    log-uniform.
    """
    lm1g = jnp.log(m1g)
    lm1q = jnp.log(jnp.clip(m1_q, _LOGSAFE, None))

    iz = jnp.clip(jnp.searchsorted(zg, z_q) - 1, 0, zg.shape[0] - 2)
    wz = jnp.clip((z_q - zg[iz]) / (zg[iz + 1] - zg[iz]), 0.0, 1.0)

    im = jnp.clip(jnp.searchsorted(lm1g, lm1q) - 1, 0, lm1g.shape[0] - 2)
    wm = jnp.clip((lm1q - lm1g[im]) / (lm1g[im + 1] - lm1g[im]), 0.0, 1.0)

    top = ltab[iz, im] * (1.0 - wm) + ltab[iz, im + 1] * wm
    bot = ltab[iz + 1, im] * (1.0 - wm) + ltab[iz + 1, im + 1] * wm
    return jnp.exp(top * (1.0 - wz) + bot * wz)


def _znorm_interp(eval_norm_fn, z_query):
    """Evaluate the prob-axis normalisation on a z-grid and interpolate (log-space).

    SEQUENTIAL (``lax.map``, not ``vmap``) over the z nodes.  ``eval_norm_fn``
    forms the explicit cross-kernel ``k(coords, Z)`` of shape (G, M) before its
    matvec; batching that with ``vmap`` materialises the whole (N_z, G, M) cube at
    once, which for the registered ``gp4d`` model (G = 48*24*24 = 27648 coarse
    normalisation nodes, M = 10*8*10*8 = 6400 inducing points, N_z = 40 at the
    default zMax = 5) is 40*27648*6400*8 B = 52.74 GiB of XLA scratch -- measured
    as ``temp_size_in_bytes`` for a query of only 8 points, i.e. before any event
    or injection data enters, so no ``--sel_batch_size``/auto-blocking setting can
    reduce it: an immediate OOM on any GPU in the fleet.  ``lax.map`` keeps one
    z node live (measured 2.64 GiB for gp4d, 20x less) and returns the same
    stacked table -- bit-identical values -- matching the memory-safe idiom
    AdditiveGPPopulation already uses for its own z grid.

    ``jax.checkpoint`` is REQUIRED on the mapped body, not cosmetic: a scan
    stores its per-iteration residuals for the transpose, so reverse mode
    (``--sampler numpyro``) would otherwise keep every z node's (G, M) kernel on
    the tape -- measured gp4d gradient scratch 756 GiB with a plain ``lax.map``
    (583 GiB with the old ``vmap``).  Rematerialising each node's forward pass
    instead brings it to 13.9 GiB, a 42x reduction against master, with the same
    forward values and gradients.
    """
    import jax
    zg = jnp.linspace(0.0, _ZNORM_HI, _ZNORM_N)
    norms = jax.lax.map(jax.checkpoint(eval_norm_fn), zg)
    return jnp.exp(jnp.interp(z_query, zg, jnp.log(jnp.where(norms > 0, norms, _LOGSAFE))))


def _to_coord(axis: str, m1, q, chi, z):
    """Map a physical value of ``axis`` to its GP coordinate."""
    if axis == "m1":
        return jnp.log(jnp.clip(m1, _LOGSAFE, None))
    if axis == "q":
        return jnp.asarray(q)
    if axis == "chi":
        return jnp.sin(0.5 * jnp.pi * jnp.clip(chi, -1.0, 1.0))   # boundary-warp
    if axis == "z":
        return jnp.log1p(jnp.asarray(z))
    raise ValueError(f"unknown axis {axis!r}")


def _axis_phys_grid(axis: str):
    """Normalisation grid (physical units) for a probability axis."""
    if axis == "m1":
        return get_mass_grid()
    if axis == "q":
        return get_q_grid()
    if axis == "chi":
        return get_chi_grid()
    raise ValueError(f"axis {axis!r} is not a probability axis")


@dataclass(frozen=True)
class AxisCfg:
    """Per-axis GP configuration."""
    name: str
    family: str          # "matern32" | "rbf"
    n_inducing: int
    node_lo: float       # inducing-node span (physical units)
    node_hi: float
    log_ls_lo: float     # log length-scale prior bounds (coordinate units)
    log_ls_hi: float

    def nodes_phys(self) -> jnp.ndarray:
        if self.name == "m1":
            return jnp.exp(jnp.linspace(jnp.log(self.node_lo),
                                        jnp.log(self.node_hi), self.n_inducing))
        if self.name == "z":
            # Uniform in the GP COORDINATE log1p(z), not in z: linear-in-z nodes
            # would leave coordinate gaps of 0.54 at low z (where every event is)
            # against a length-scale prior of [0.05, 0.8].
            return jnp.expm1(jnp.linspace(math.log1p(self.node_lo),
                                          math.log1p(self.node_hi),
                                          self.n_inducing))
        return jnp.linspace(self.node_lo, self.node_hi, self.n_inducing)

    def nodes_coord(self) -> jnp.ndarray:
        p = self.nodes_phys()
        return _to_coord(self.name, p, p, p, p)


# Default per-axis configurations (kernels: Matern-3/2 for m1, RBF elsewhere).
_DEFAULT_AXES = {
    "m1":  AxisCfg("m1",  "matern32", 10, 3.0,   100.0, math.log(0.15), math.log(2.0)),
    "q":   AxisCfg("q",   "rbf",       8, 0.10,    1.0, math.log(0.05), math.log(0.8)),
    "chi": AxisCfg("chi", "rbf",      10, -0.95,   0.95, math.log(0.05), math.log(1.0)),
    # The z inducing nodes must SPAN the shared analysis grid, exactly like
    # ``_ZNORM_HI`` above: the field is the RKHS conditional mean
    # ``mu + k(x*, Z) alpha``, so a query beyond the outermost node is suppressed
    # by exp(-d^2/2ls^2) and reverts to the prior mean (which is identically zero
    # on the z axis).  With nodes stopping at z = 1.5 (coordinate 0.916) against
    # zMax = 5 (1.792) the modulation decayed back to its zero-field value above
    # z ~ 2 -- 6.4e-5 of it survived at the geometric-median length scale --
    # silently reverting R(z) to the parametric (1+z)^(gamma-1) over most of the
    # analysis range.  The node COUNT is deliberately unchanged: M is the product
    # over axes, so gp4d would go from 6400 to 11200 inducing points.
    "z":   AxisCfg("z",   "rbf",       8, 0.0, max(1.5, float(zMax)),
                   math.log(0.05), math.log(0.8)),
}

# Shared prior bounds.
_B_MMIN  = (2.0, 10.0)
_B_DMMIN = (0.01, 10.0)
_B_MMAX  = (50.0, 100.0)
_B_DMMAX = (0.01, 20.0)
_B_LOGAMP = (math.log(0.1), math.log(3.0))
_B_ALPHA_GP = (-4.0, 12.0)     # m1 shape-mean slope
_B_BETA_GP  = (-4.0, 12.0)     # q  shape-mean slope
_B_ALPHA_MASS = (-4.0, 6.0)    # baseline power-law mass slope
_B_BETA_Q = (-2.0, 7.0)        # baseline pairing slope
_B_MUCHI = (-1.0, 1.0)
_B_SIGCHI = (0.01, 1.0)
_B_GAMMA = (-10.0, 10.0)


# ============================================================
# Kernel + whitened rank-M field
# ============================================================

def _build_kernel(amp: jnp.ndarray, ls: Sequence, families: Sequence[str]):
    """Product ARD kernel over axes: amp^2 * prod_i family_i(scale=ls_i) on axis i."""
    from tinygp import kernels, transforms   # lazy: tinygp only needed to evaluate

    fam_map = {"matern32": kernels.Matern32, "rbf": kernels.ExpSquared}
    k = None
    for i, (fam, l) in enumerate(zip(families, ls)):
        base = fam_map[fam](scale=l)
        sub = transforms.Subspace(i, base) if len(families) > 1 else base
        k = sub if k is None else k * sub
    return (amp ** 2) * k


def _field_factor(nodes, families, amp, ls, jitter):
    """Cholesky factor L of K(Z, Z) and the kernel object (reused for all evals)."""
    k = _build_kernel(amp, ls, families)
    K = k(nodes, nodes) + jitter * jnp.eye(nodes.shape[0])
    return jnp.linalg.cholesky(K), k


def _alpha_from_xi(L, xi):
    """alpha = L^{-T} xi  (so that f(Z) = mu(Z) + L xi exactly)."""
    return jsl.solve_triangular(L, xi, lower=True, trans=1)


def _eval_field(coords, kern, nodes, alpha, mean):
    """f(coords) = mean(coords) + k(coords, Z) @ alpha."""
    kqz = kern(coords, nodes)                      # (Nq, M)
    return mean + kqz @ alpha


def _eval_field_clipped(coords, kern, nodes, alpha, mean):
    """``mean + clip(GP deviation, -_FIELD_CLIP, _FIELD_CLIP)``.

    Only the DEVIATION is bounded.  Clipping the SUM would truncate the
    parametric mean, whose dynamic range is exactly what ``alpha_gp``/``beta_gp``
    are meant to control: at the shipped fiducial (alpha_gp = 4) the mean
    ``-alpha_gp log m1`` reaches -_FIELD_CLIP at m1 = exp(10/4) = 12.2 Msun, so
    every ``mean_mode="shape"`` model was FLAT in m1 above ~12 Msun instead of
    ~m^-4 (measured d log p / d log m1 = -0.07 between 20 and 40 Msun), with the
    same clip in the normaliser hiding it.  The mean is analytic and finite over
    the whole prior box (|mu| <= 12 log(200) + 12 log(1/0.01) ~ 119 nats), so it
    needs no guard; the deviation, which the whitened latent can drive anywhere,
    keeps one.
    """
    dev = kern(coords, nodes) @ alpha
    return mean + jnp.clip(dev, -_FIELD_CLIP, _FIELD_CLIP)


def _grid_integrate(values, grids):
    """Nested trapezoid of ``values`` over the listed 1-D ``grids`` (trailing axes)."""
    out = values
    for g in reversed(grids):
        out = jnp.trapezoid(out, g, axis=-1)
    return out


def _broadcast_logp_inputs(m1, q, z, chieff):
    """Broadcast population-model inputs and flatten the query axes for GP eval.

    Accepts scalars or arbitrarily broadcastable arrays (e.g. a sparse predictive
    mesh with one non-trivial axis each: ``m1`` ``(Nm,1,1,1)``, ``q``
    ``(1,Nq,1,1)``, …) and returns the four inputs broadcast to their common
    shape and raveled to 1-D, plus that common shape so callers can reshape the
    flat result back.  Scalars give ``out_shape == ()`` with length-1 flats.
    """
    m1, q, z, chi = jnp.broadcast_arrays(
        jnp.asarray(m1, dtype=float),
        jnp.asarray(q, dtype=float),
        jnp.asarray(z, dtype=float),
        jnp.asarray(chieff, dtype=float),
    )
    out_shape = m1.shape
    return m1.ravel(), q.ravel(), z.ravel(), chi.ravel(), out_shape


# ============================================================
# Joint GP population model
# ============================================================

@dataclass
class JointGPPopulation:
    """A population model with a joint GP source-shape over ``gp_axes``.

    ``p(m1, q, chi, z) = G(gp_axes) * Prod_{a not in gp_axes} B_a``, with the GP
    factor ``G`` normalised over its probability axes (m1, q, chi) -- conditional
    on z when z is a GP axis -- and the marginal redshift rate carried by the
    parametric ``(1+z)**(gamma-1)`` term.  Low-mass tapers and the ``m2 = q*m1``
    secondary cut are applied consistently to both the density and its
    normalisation integral.

    Parameter order (sliced by ``log_p_pop`` exactly as listed by ``param_specs``)
    ------------------------------------------------------------------------------
    m_min, dm_min, m_max, dm_max,                      # m1 taper (always)
    log_amp,                                           # GP amplitude
    log_ls_<axis> for axis in gp_axes,                 # ARD length scales
    [alpha_gp if m1 in gp_axes and mean_mode=="shape"],
    [beta_gp  if q  in gp_axes and mean_mode=="shape"],
    xi_0 .. xi_{M-1},                                  # whitened latent (normal)
    [alpha_mass if m1 not in gp_axes],                 # baseline mass slope
    [beta_q     if q  not in gp_axes],                 # baseline pairing slope
    [mu_chi, sigma_chi if chi not in gp_axes],         # baseline spin
    gamma                                              # redshift rate (always)
    """

    gp_axes: tuple
    mean_mode: str = "shape"            # "shape" | "zero"
    axes_cfg: dict = field(default_factory=lambda: dict(_DEFAULT_AXES))
    latex: str = "GP"

    def __post_init__(self):
        self.gp_axes = tuple(a for a in AXIS_ORDER if a in self.gp_axes)
        if not self.gp_axes:
            raise ValueError("JointGPPopulation needs at least one GP axis")
        self._cfgs = [self.axes_cfg[a] for a in self.gp_axes]
        self._families = [c.family for c in self._cfgs]
        node_axes = [c.nodes_coord() for c in self._cfgs]
        mesh = jnp.meshgrid(*node_axes, indexing="ij")
        self._Z = jnp.stack([m.ravel() for m in mesh], axis=-1)   # (M, d)
        self.M = int(self._Z.shape[0])
        self._prob_gp = [a for a in self.gp_axes if a in _PROB_AXES]
        self._z_in_gp = "z" in self.gp_axes
        self._jitter = _JITTER_REL

    # -- parameter specifications ------------------------------------------

    @property
    def param_specs(self):
        s = [
            ParamSpec(r"$m_{\min}$",        *_B_MMIN,  name="m_min"),
            ParamSpec(r"$\delta m_{\min}$", *_B_DMMIN, name="dm_min"),
            ParamSpec(r"$m_{\max}$",        *_B_MMAX,  name="m_max"),
            ParamSpec(r"$\delta m_{\max}$", *_B_DMMAX, name="dm_max"),
            ParamSpec(r"$\log A$",          *_B_LOGAMP, name="log_amp"),
        ]
        for a in self.gp_axes:
            c = self.axes_cfg[a]
            s.append(ParamSpec(rf"$\log\ell_{{{a}}}$", c.log_ls_lo, c.log_ls_hi,
                               name=f"log_ls_{a}"))
        if self.mean_mode == "shape":
            if "m1" in self.gp_axes:
                s.append(ParamSpec(r"$\alpha_{\rm GP}$", *_B_ALPHA_GP, name="alpha_gp"))
            if "q" in self.gp_axes:
                s.append(ParamSpec(r"$\beta_{\rm GP}$", *_B_BETA_GP, name="beta_gp"))
        for i in range(self.M):
            s.append(ParamSpec(rf"$\xi_{{{i}}}$", -6.0, 6.0, name=f"xi_{i}",
                               prior_kind="normal", prior_loc=0.0, prior_scale=1.0))
        if "m1" not in self.gp_axes:
            s.append(ParamSpec(r"$\alpha_m$", *_B_ALPHA_MASS, name="alpha_mass"))
        if "q" not in self.gp_axes:
            s.append(ParamSpec(r"$\beta_q$", *_B_BETA_Q, name="beta_q"))
        if "chi" not in self.gp_axes:
            s.append(ParamSpec(r"$\mu_\chi$", *_B_MUCHI, name="mu_chi"))
            s.append(ParamSpec(r"$\sigma_\chi$", *_B_SIGCHI, name="sigma_chi"))
        s.append(ParamSpec(r"$\gamma$", *_B_GAMMA, name="gamma"))
        return s

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def fiducial(self):
        """Fiducial vector: xi = 0 (field == mean), central hyperparameters."""
        return [0.0 if sp.name.startswith("xi_") else float((sp.low + sp.high) * 0.5)
                for sp in self.param_specs]

    # -- field mean (parametric shape on the GP axes) ----------------------

    def _mean_on_coords(self, coords, alpha_gp, beta_gp):
        """Additive parametric mean evaluated on GP coordinates ``coords`` (N, d)."""
        if self.mean_mode == "zero":
            return jnp.zeros(coords.shape[0])
        mu = jnp.zeros(coords.shape[0])
        for j, a in enumerate(self.gp_axes):
            if a == "m1":
                mu = mu - alpha_gp * coords[:, j]                 # coord = log m1
            elif a == "q":
                mu = mu + beta_gp * jnp.log(coords[:, j] + 0.01)  # coord = q
        return mu

    # -- evaluation --------------------------------------------------------

    def log_p_pop(self, m1, q, z, chieff, theta):
        # Flatten arbitrarily-broadcastable inputs (e.g. the WL path's
        # (Nsamp, Nnodes) mu-marginalisation mesh) to 1-D for the GP eval, then
        # reshape the result back.  The GP field / _mean_on_coords assume 1-D
        # coords, so a 2-D query previously crashed with a broadcast error.
        m1, q, z, chi, _out_shape = _broadcast_logp_inputs(m1, q, z, chieff)

        # ---- unpack theta in param_specs order ----
        m_min, dm_min, m_max, dm_max = theta[0], theta[1], theta[2], theta[3]
        i = 4
        amp = jnp.exp(theta[i]); i += 1
        ls = [jnp.exp(theta[i + j]) for j in range(len(self.gp_axes))]
        i += len(self.gp_axes)
        alpha_gp = beta_gp = 0.0
        if self.mean_mode == "shape":
            if "m1" in self.gp_axes:
                alpha_gp = theta[i]; i += 1
            if "q" in self.gp_axes:
                beta_gp = theta[i]; i += 1
        xi = theta[i:i + self.M]; i += self.M
        alpha_mass = beta_q = mu_chi = sig_chi = None
        if "m1" not in self.gp_axes:
            alpha_mass = theta[i]; i += 1
        if "q" not in self.gp_axes:
            beta_q = theta[i]; i += 1
        if "chi" not in self.gp_axes:
            mu_chi = theta[i]; sig_chi = theta[i + 1]; i += 2
        gamma = theta[i]

        # ---- GP factor ----
        L, kern = _field_factor(self._Z, self._families, amp, ls,
                                self._jitter * amp ** 2 + 1e-9)
        alpha = _alpha_from_xi(L, xi)

        coords_q = jnp.stack([_to_coord(a, m1, q, chi, z) for a in self.gp_axes], axis=-1)
        f_q = _eval_field_clipped(coords_q, kern, self._Z, alpha,
                                  self._mean_on_coords(coords_q, alpha_gp, beta_gp))
        p_gp_un = jnp.exp(f_q)
        p_gp_un = p_gp_un * self._taper_cut(m1, q, m_min, dm_min, m_max, dm_max)

        norm = self._normalise(kern, alpha, alpha_gp, beta_gp,
                               m1, z, m_min, dm_min, m_max, dm_max)
        p_gp = p_gp_un / jnp.where(norm > 0, norm, 1.0)

        # ---- baseline factors for non-GP probability axes ----
        p_base = jnp.ones_like(m1)
        if "m1" not in self.gp_axes:
            p_base = p_base * self._baseline_mass(m1, alpha_mass, m_min, dm_min, m_max, dm_max)
        if "q" not in self.gp_axes:
            p_base = p_base * self._baseline_pairing(m1, q, beta_q, m_min, dm_min)
        if "chi" not in self.gp_axes:
            p_base = p_base * self._baseline_spin(chi, mu_chi, sig_chi)

        p = p_gp * p_base
        log_p = jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)
        return (log_p + (gamma - 1.0) * jnp.log1p(z)).reshape(_out_shape)

    # -- tapers / cut, applied identically to density and normalisation ----

    def _taper_cut(self, m1, q, m_min, dm_min, m_max, dm_max):
        s = jnp.ones_like(m1)
        if "m1" in self.gp_axes:
            s = s * sfilter_low(m1, m_min, dm_min) * sfilter_high(m1, m_max, dm_max)
        if "q" in self.gp_axes:
            m2 = q * m1
            cut = jnp.nan_to_num(sfilter_low(m2, m_min, dm_min), nan=0.0)
            s = s * jnp.where(m2 < m_min, 0.0, cut)
        return s

    def _normalise(self, kern, alpha, alpha_gp, beta_gp,
                   m1, z, m_min, dm_min, m_max, dm_max):
        """Integrate exp(f) * tapers over the probability axes.

        When z is a GP axis the field is z-dependent; the integral is then a
        smooth function of z evaluated on a small z-grid and interpolated to the
        query redshifts (cost independent of the number of queries).  A coarse
        prob-axis grid is used for multi-axis integrals under z-conditioning;
        single-axis and z-independent integrals use the full grids.

        m1-conditioning
        ---------------
        When ``q`` is a probability axis but ``m1`` is *not* a GP axis the field
        is m1-independent, yet the ``m2 = q*m1`` secondary cut in ``_taper_cut``
        still ties the integrand to the query m1 (with m1 absent the norm would
        be evaluated at the placeholder ``phys["m1"] = 1``, cutting m2 = q <= 1 <
        m_min and zeroing the whole grid).  The norm is then a smooth function of
        m1 (and of z, when z is a GP axis): it is tabulated over an m1 grid and
        interpolated in ``log m1`` per query, exactly like the z-conditioning.
        The extra m1 axis multiplies only the taper broadcast and trapezoid --
        never the GP kernel evaluation (the field is computed once as ``(G,)``).
        """
        if not self._prob_gp:
            # Pure rate-axis GP (e.g. gp1d_z): no PROBABILITY axis to normalise
            # over (z is a rate/evolution axis, handled by the baseline
            # factors), so the prob-axis normalisation is unity. meshgrid() over
            # an empty axis list would otherwise IndexError at flat[0].shape[0].
            return jnp.ones_like(z) if self._z_in_gp else jnp.asarray(1.0)
        # Coarsen the probability-axis normalisation grid whenever the full
        # tensor grid would be intractable: any z-conditioned multi-prob-axis
        # model (the full grid is retabulated per z node), OR any >=3-prob-axis
        # model (e.g. gp3d_m1_q_chi: full 500x200x200 = 2e7 points x M nodes ~
        # 128 GB, an OOM/hang on the first likelihood call).  A 2-prob-axis,
        # z-free model (gp2d_q_chi, full 200x200 = 4e4) stays on the full grid,
        # so its normalisation is unchanged.
        coarse = (self._z_in_gp and len(self._prob_gp) >= 2) or len(self._prob_gp) >= 3
        if coarse:
            grids = [_coarse_axis_grid(a) for a in self._prob_gp]
        else:
            grids = [_axis_phys_grid(a) for a in self._prob_gp]
        mesh = jnp.meshgrid(*grids, indexing="ij")
        flat = [mm.ravel() for mm in mesh]
        G = flat[0].shape[0]
        phys = {a: jnp.ones(G) for a in AXIS_ORDER}
        for a, fv in zip(self._prob_gp, flat):
            phys[a] = fv

        m1_cond = ("q" in self._prob_gp) and ("m1" not in self.gp_axes)
        if m1_cond:
            m1g = _m1norm_grid()                             # (Nm1,), log-uniform
            grid_shape = [g.shape[0] for g in grids]

            def _eval_norm(zval):
                # Field is m1-independent on this branch: evaluate it once as
                # (G,), then broadcast against the m1-dependent m2 = q*m1 cut.
                cols = []
                for a in self.gp_axes:
                    if a == "z":
                        cols.append(_to_coord("z", None, None, None, jnp.full(G, zval)))
                    else:
                        cols.append(_to_coord(a, phys["m1"], phys["q"], phys["chi"], None))
                coords = jnp.stack(cols, axis=-1)
                fg = _eval_field_clipped(coords, kern, self._Z, alpha,
                                         self._mean_on_coords(coords, alpha_gp, beta_gp))
                exp_field = jnp.exp(fg)                                        # (G,)
                # m1 not in gp_axes -> _taper_cut skips its m1 branch and the q
                # branch computes m2 = q*m1 on the (Nm1, G) lattice.
                cut = self._taper_cut(m1g[:, None], phys["q"][None, :],
                                      m_min, dm_min, m_max, dm_max)            # (Nm1, G)
                val = exp_field[None, :] * cut                                 # (Nm1, G)
                val = val.reshape([m1g.shape[0], *grid_shape])
                return _grid_integrate(val, grids)                            # (Nm1,)

            if self._z_in_gp:
                import jax
                zg = jnp.linspace(0.0, _ZNORM_HI, _ZNORM_N)
                # lax.map + checkpoint, not vmap: batching the (Nm1, G) lattice
                # AND the (G, M) cross-kernel inside over all z nodes at once is
                # what made gp4d need 52.7 GiB of scratch (see _znorm_interp).
                tab = jax.lax.map(jax.checkpoint(_eval_norm), zg)              # (Nz, Nm1)
                ltab = jnp.log(jnp.where(tab > 0, tab, _LOGSAFE))
                return _interp2_log(ltab, zg, m1g, z, m1)                     # (Nquery,)
            tab = _eval_norm(0.0)                                             # (Nm1,)
            ltab = jnp.log(jnp.where(tab > 0, tab, _LOGSAFE))
            return jnp.exp(jnp.interp(
                jnp.log(jnp.clip(m1, _LOGSAFE, None)), jnp.log(m1g), ltab))   # (Nquery,)

        def _eval_norm(zval):
            cols = []
            for a in self.gp_axes:
                if a == "z":
                    cols.append(_to_coord("z", None, None, None, jnp.full(G, zval)))
                else:
                    cols.append(_to_coord(a, phys["m1"], phys["q"], phys["chi"], None))
            coords = jnp.stack(cols, axis=-1)
            fg = _eval_field_clipped(coords, kern, self._Z, alpha,
                                     self._mean_on_coords(coords, alpha_gp, beta_gp))
            val = jnp.exp(fg)
            val = val * self._taper_cut(phys["m1"], phys["q"],
                                        m_min, dm_min, m_max, dm_max)
            return _grid_integrate(val.reshape([g.shape[0] for g in grids]), grids)

        if self._z_in_gp:
            return _znorm_interp(_eval_norm, z)         # (Nquery,)
        return _eval_norm(0.0)                          # scalar

    # -- baseline parametric factors --------------------------------------

    def _baseline_mass(self, m1, alpha_mass, m_min, dm_min, m_max, dm_max):
        grid = get_mass_grid()
        un = (sfilter_low(grid, m_min, dm_min) * sfilter_high(grid, m_max, dm_max)
              * grid ** (-alpha_mass))
        norm = jnp.trapezoid(un, grid)
        p = (sfilter_low(m1, m_min, dm_min) * sfilter_high(m1, m_max, dm_max)
             * m1 ** (-alpha_mass))
        return p / jnp.where(norm > 0, norm, 1.0)

    def _baseline_pairing(self, m1, q, beta_q, m_min, dm_min):
        qg = get_q_grid()
        m2g = qg[None, :] * m1[..., None]
        # Safe pow (see PowerLawPairing._eval_unnorm): the q = 0 grid node
        # makes d/dbeta_q NaN for every beta_q < 0 without the double-where.
        qg_safe = jnp.where(qg > 0.0, qg, 1.0)
        raw = jnp.where(qg[None, :] > 0.0, qg_safe[None, :] ** beta_q, 0.0) * jnp.nan_to_num(
            sfilter_low(m2g, m_min, dm_min), nan=0.0)
        raw = jnp.where(m2g < m_min, 0.0, raw)
        norm = jnp.trapezoid(raw, qg, axis=-1)
        m2 = q * m1
        q_safe = jnp.where(q > 0.0, q, 1.0)
        p = jnp.where(q > 0.0, q_safe ** beta_q, 0.0) * jnp.nan_to_num(
            sfilter_low(m2, m_min, dm_min), nan=0.0)
        p = jnp.where(m2 < m_min, 0.0, p)
        return p / jnp.where(norm > 0, norm, 1.0)

    def _baseline_spin(self, chi, mu_chi, sig_chi):
        cg = get_chi_grid()
        un = jnp.where((cg >= -1) & (cg <= 1),
                       jnp.exp(-0.5 * ((cg - mu_chi) / sig_chi) ** 2), 0.0)
        norm = jnp.trapezoid(un, cg)
        p = jnp.where((chi >= -1) & (chi <= 1),
                      jnp.exp(-0.5 * ((chi - mu_chi) / sig_chi) ** 2), 0.0)
        return p / jnp.where(norm > 0, norm, 1.0)


# ============================================================
# Additive (functional-ANOVA) GP population model
# ============================================================

@dataclass
class AdditiveGPPopulation:
    """Additive decomposition: ``f = sum_terms f_term``.

    Probability axes are ``{m1, q, chi}``; ``z`` enters only through interaction
    terms that include it (conditioning).  Each term is an independent whitened
    rank-M GP with its own amplitude, length scales, and ``xi``, and is ZERO-MEAN
    (unlike :class:`JointGPPopulation`, this family has no parametric
    ``alpha_gp``/``beta_gp`` shape mean, so its prior is centred on a density that
    is flat in every probability axis -- the same convention the binned
    :class:`BinnedGPPopulation` uses).

    Identifiability: each term's grand mean over the coarse probability grid is
    subtracted before exponentiating, which keeps ``exp(F)`` at O(1) scale.  It is
    NOT the functional-ANOVA sum-to-zero projection and is not claimed to be: a
    per-term CONSTANT cancels identically against the normaliser ``Z(z)`` computed
    from the same centred field, so it has no effect on the likelihood.  The real
    degeneracy -- an interaction term such as ``f_{m1,q}`` spanning functions that
    are nearly constant in ``q`` and so duplicating ``f_{m1}`` -- is left to the
    N(0, 1) latent priors, which keep the posterior proper (long ridges in the
    latent space, not an improper posterior).  Projecting each interaction onto the
    orthogonal complement of the main effects' spans would remove it.
    """

    terms: tuple = (("m1",), ("q",), ("chi",), ("m1", "q"), ("m1", "z"), ("chi", "z"))
    axes_cfg: dict = field(default_factory=lambda: dict(_DEFAULT_AXES))
    latex: str = "GP add"

    def __post_init__(self):
        self.terms = tuple(tuple(a for a in AXIS_ORDER if a in t) for t in self.terms)
        self._term_meta = []
        for t in self.terms:
            cfgs = [self.axes_cfg[a] for a in t]
            node_axes = [c.nodes_coord() for c in cfgs]
            mesh = jnp.meshgrid(*node_axes, indexing="ij")
            Z = jnp.stack([m.ravel() for m in mesh], axis=-1)
            self._term_meta.append({"axes": t, "fam": [c.family for c in cfgs],
                                    "Z": Z, "M": int(Z.shape[0])})
        self._z_in_gp = any("z" in t for t in self.terms)
        self._jitter = _JITTER_REL

    @property
    def param_specs(self):
        s = [
            ParamSpec(r"$m_{\min}$",        *_B_MMIN,  name="m_min"),
            ParamSpec(r"$\delta m_{\min}$", *_B_DMMIN, name="dm_min"),
            ParamSpec(r"$m_{\max}$",        *_B_MMAX,  name="m_max"),
            ParamSpec(r"$\delta m_{\max}$", *_B_DMMAX, name="dm_max"),
        ]
        for t, meta in zip(self.terms, self._term_meta):
            tag = "".join(t)
            s.append(ParamSpec(rf"$\log A_{{{tag}}}$", *_B_LOGAMP, name=f"log_amp_{tag}"))
            for a in t:
                c = self.axes_cfg[a]
                s.append(ParamSpec(rf"$\log\ell_{{{tag},{a}}}$", c.log_ls_lo, c.log_ls_hi,
                                   name=f"log_ls_{tag}_{a}"))
            for i in range(meta["M"]):
                s.append(ParamSpec(rf"$\xi_{{{tag},{i}}}$", -6.0, 6.0,
                                   name=f"xi_{tag}_{i}",
                                   prior_kind="normal", prior_loc=0.0, prior_scale=1.0))
        s.append(ParamSpec(r"$\gamma$", *_B_GAMMA, name="gamma"))
        return s

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def fiducial(self):
        return [0.0 if sp.name.startswith("xi_") else float((sp.low + sp.high) * 0.5)
                for sp in self.param_specs]

    @staticmethod
    def _term_eval(meta, kern, alpha, phys, zval, n):
        """Evaluate one (uncentred) term field at ``n`` points; ``zval`` scalar or (n,)."""
        cols = []
        for a in meta["axes"]:
            if a == "z":
                cols.append(_to_coord("z", None, None, None, jnp.broadcast_to(zval, (n,))))
            else:
                cols.append(_to_coord(a, phys["m1"], phys["q"], phys["chi"], None))
        coords = jnp.stack(cols, axis=-1)
        return _eval_field(coords, kern, meta["Z"], alpha, jnp.zeros(n))

    def log_p_pop(self, m1, q, z, chieff, theta):
        import jax
        # Flatten broadcastable inputs (e.g. WL's (Nsamp, Nnodes) mesh) to 1-D;
        # the per-term GP eval assumes 1-D coords (N = m1.shape[0]).  Reshape
        # back at the end.  A 2-D query previously crashed with a broadcast error.
        m1, q, z, chi, _out_shape = _broadcast_logp_inputs(m1, q, z, chieff)
        N = m1.shape[0]

        m_min, dm_min, m_max, dm_max = theta[0], theta[1], theta[2], theta[3]
        i = 4
        # Build each term's kernel + alpha ONCE (z-independent).
        term_built = []
        for meta in self._term_meta:
            amp = jnp.exp(theta[i]); i += 1
            ls = [jnp.exp(theta[i + j]) for j in range(len(meta["axes"]))]
            i += len(meta["axes"])
            xi = theta[i:i + meta["M"]]; i += meta["M"]
            L, kern = _field_factor(meta["Z"], meta["fam"], amp, ls,
                                    self._jitter * amp ** 2 + 1e-9)
            term_built.append((meta, kern, _alpha_from_xi(L, xi)))
        gamma = theta[i]

        # Coarse probability grid (m1, q, chi) for centring + normalisation.
        grids = [_coarse_axis_grid("m1"), _coarse_axis_grid("q"), _coarse_axis_grid("chi")]
        mg, qg, cg = grids
        Mg, Qg, Cg = jnp.meshgrid(mg, qg, cg, indexing="ij")
        flatg = {"m1": Mg.ravel(), "q": Qg.ravel(), "chi": Cg.ravel()}
        G = flatg["m1"].shape[0]
        taper_grid = self._taper_cut(flatg["m1"], flatg["q"], m_min, dm_min, m_max, dm_max)

        # Per z-grid node: per-term centring constant c_term(z) and the prob-axis
        # normalisation Z(z).  Smooth in z -> interpolate to queries.
        def _grid_quant(zval):
            cs = []
            F = jnp.zeros(G)
            for meta, kern, alpha in term_built:
                fg = self._term_eval(meta, kern, alpha, flatg, zval, G)
                c = jnp.mean(fg)
                cs.append(c)
                F = F + (fg - c)
            pg = jnp.exp(jnp.clip(F, -_FIELD_CLIP, _FIELD_CLIP)) * taper_grid
            Z = _grid_integrate(pg.reshape([g.shape[0] for g in grids]), grids)
            return Z, jnp.stack(cs)

        if self._z_in_gp:
            zg = jnp.linspace(0.0, _ZNORM_HI, _ZNORM_N)
            # jax.checkpoint is REQUIRED, not cosmetic (see _znorm_interp): a scan
            # keeps its per-iteration residuals for the transpose, so reverse mode
            # would otherwise tape every z node's (G, M) cross-kernel for all six
            # terms.
            Zg, Cg_terms = jax.lax.map(jax.checkpoint(_grid_quant), zg)
            c_at = [jnp.interp(z, zg, Cg_terms[:, j]) for j in range(len(term_built))]
            norm = jnp.exp(jnp.interp(z, zg,
                                      jnp.log(jnp.where(Zg > 0, Zg, _LOGSAFE))))
        else:
            # No term carries z (e.g. gp_separable), so _grid_quant is constant in
            # z: evaluate it ONCE instead of mapping _ZNORM_N identical nodes and
            # interpolating a constant.
            Z0, C0 = _grid_quant(0.0)
            c_at = [C0[j] for j in range(len(term_built))]
            norm = jnp.where(Z0 > 0, Z0, 1.0)

        # Query points (direct evaluation; centring constants interpolated in z).
        qphys = {"m1": m1, "q": q, "chi": chi}
        F_q = jnp.zeros(N)
        for j, (meta, kern, alpha) in enumerate(term_built):
            fpt = self._term_eval(meta, kern, alpha, qphys, z, N)
            F_q = F_q + (fpt - c_at[j])
        pun = jnp.exp(jnp.clip(F_q, -_FIELD_CLIP, _FIELD_CLIP)) * \
            self._taper_cut(m1, q, m_min, dm_min, m_max, dm_max)
        # The _LOGSAFE operand exists only to keep the dead branch of the
        # where free of log(0) (both branches are differentiated); the
        # RETURNED value outside the taper support must be -inf, exactly as
        # the joint and binned GP models report it.  Returning log(_LOGSAFE)
        # (~ -690, finite) instead gave zero-support points a nonzero density
        # that importance reweighting happily integrates.
        log_p_src = (jnp.where(pun > 0, jnp.log(jnp.where(pun > 0, pun, _LOGSAFE)),
                               -jnp.inf)
                     - jnp.log(jnp.where(norm > 0, norm, 1.0)))
        return (log_p_src + (gamma - 1.0) * jnp.log1p(z)).reshape(_out_shape)

    def _taper_cut(self, m1, q, m_min, dm_min, m_max, dm_max):
        s = sfilter_low(m1, m_min, dm_min) * sfilter_high(m1, m_max, dm_max)
        m2 = q * m1
        cut = jnp.nan_to_num(sfilter_low(m2, m_min, dm_min), nan=0.0)
        return s * jnp.where(m2 < m_min, 0.0, cut)


# ============================================================
# Binned GP population model (gppop; Ray et al. 2023, ApJ 957, 37)
# ============================================================
#
# Same whitened finite-rank GP prior as ``JointGPPopulation``, but the field is
# *piecewise-constant* over bins: the inducing points are the bin centres and a
# query is gathered by bin lookup (no smooth interpolation).  Bins are
# lower-triangular in (ln m1, ln m2) (m2 <= m1) following gppop, optionally
# crossed with redshift bins.  The gppop rate density
#     p(m1, m2, z) = n_bin * (dV_c/dz)/(1+z) / (m1 m2)
# maps onto darksirens by splitting the (dV_c/dz)/(1+z) factor across three
# places: the redshift prior supplies dV_c/dz (volume.py), the likelihood's
# log1p Jacobian accounts for the m1src -> m1det mass change of variables (NOT
# time dilation; utils.py), and the 1/(1+z) source-time dilation is applied
# here in ``log_p_pop`` (see the z-bearing branch).  The (m1, m2) -> (m1, q)
# conversion via |dm2/dq| = m1 gives the source-frame shape  n_bin / (q m1).

# Default bin edges (coarse for tractability; override via env vars).
def _gppop_edges(env_name, default):
    import os
    raw = os.environ.get(env_name)
    if not raw:
        return tuple(float(x) for x in default)
    return tuple(float(x) for x in raw.replace(",", " ").split())


_GPPOP_M_EDGES = _gppop_edges(
    "DARKSIRENS_GPPOP_M_EDGES", (5.0, 10.0, 15.0, 25.0, 40.0, 65.0, 100.0))
_GPPOP_Z_EDGES = _gppop_edges(
    "DARKSIRENS_GPPOP_Z_EDGES", (0.0, 0.3, 0.7, 1.2))

# Log length-scale prior bounds (coordinate units: ln-mass for mass, linear z).
_B_GPPOP_LS_M = (math.log(0.1), math.log(3.0))
_B_GPPOP_LS_Z = (math.log(0.05), math.log(2.0))


@dataclass
class BinnedGPPopulation:
    """Binned (piecewise-constant) GP population over lower-triangular mass bins.

    ``mass_edges`` are the m1 = m2 bin edges (solar masses); ``z_edges`` (optional)
    adds redshift bins.  With ``z_edges is None`` (model ``gppop``) the marginal
    redshift rate is the parametric ``(1 + z)**(gamma - 1)`` term and the mass
    shape is normalised over the (m1, q) grid.  With ``z_edges`` set (model
    ``gppop_mz``) the (m1, m2, z) binning carries a *free-form* rate evolution
    (no gamma, no z-normalisation); the unnormalised z-dependence is the physical
    R(z), and the TOP z bin is open-ended (constant comoving rate above
    ``z_edges[-1]``, see :meth:`_binned_density`) so the model stays usable over
    the whole analysis redshift range.  A baseline truncated-Gaussian spin
    (mu_chi, sigma_chi) is always included.

    Parameter order (sliced by ``log_p_pop`` exactly as listed by ``param_specs``)
    ------------------------------------------------------------------------------
    log_amp, log_ls_m, [log_ls_z if z_edges],
    xi_0 .. xi_{M-1},                 # whitened latent (normal), M = #bins
    mu_chi, sigma_chi,
    [gamma if z_edges is None]
    """

    mass_edges: tuple
    z_edges: tuple | None = None
    latex: str = "gppop"

    def __post_init__(self):
        medges = [float(e) for e in self.mass_edges]
        if len(medges) < 2 or any(b <= a for a, b in zip(medges, medges[1:])):
            raise ValueError("mass_edges must be strictly increasing with >= 2 entries")
        lnm = jnp.log(jnp.asarray(medges))
        self._ln_m_edges = lnm
        n_m = len(medges) - 1
        self._n_m = n_m
        # Lower-triangular (j <= i, i.e. m2 <= m1) bin set + (i, j) -> tri index map.
        tril_map = [[-1] * n_m for _ in range(n_m)]
        centers_m = []
        t = 0
        for i in range(n_m):
            for j in range(n_m):
                if j <= i:
                    tril_map[i][j] = t
                    centers_m.append((0.5 * (lnm[i] + lnm[i + 1]),
                                      0.5 * (lnm[j] + lnm[j + 1])))
                    t += 1
        self._n_tril = t
        self._tril_map = jnp.asarray(tril_map, dtype=jnp.int32)

        if self.z_edges is None:
            self._has_z = False
            self._n_z = 1
            self._z_edges = None
            self._Z = jnp.asarray(centers_m)                       # (n_tril, 2)
            self._families = ["rbf", "rbf"]
        else:
            self._has_z = True
            zedges = [float(e) for e in self.z_edges]
            if len(zedges) < 2 or any(b <= a for a, b in zip(zedges, zedges[1:])):
                raise ValueError("z_edges must be strictly increasing with >= 2 entries")
            self._z_edges = jnp.asarray(zedges)
            n_z = len(zedges) - 1
            self._n_z = n_z
            zc = [0.5 * (zedges[k] + zedges[k + 1]) for k in range(n_z)]
            rows = [(a, b, zc[k]) for (a, b) in centers_m for k in range(n_z)]
            self._Z = jnp.asarray(rows)                            # (n_tril * n_z, 3)
            self._families = ["rbf", "rbf", "rbf"]
        self.M = int(self._Z.shape[0])
        self._jitter = _JITTER_REL

    # -- parameter specifications ------------------------------------------

    @property
    def param_specs(self):
        s = [
            ParamSpec(r"$\log A$", *_B_LOGAMP, name="log_amp"),
            ParamSpec(r"$\log\ell_m$", *_B_GPPOP_LS_M, name="log_ls_m"),
        ]
        if self._has_z:
            s.append(ParamSpec(r"$\log\ell_z$", *_B_GPPOP_LS_Z, name="log_ls_z"))
        for b in range(self.M):
            s.append(ParamSpec(rf"$\xi_{{{b}}}$", -6.0, 6.0, name=f"xi_{b}",
                               prior_kind="normal", prior_loc=0.0, prior_scale=1.0))
        s.append(ParamSpec(r"$\mu_\chi$", *_B_MUCHI, name="mu_chi"))
        s.append(ParamSpec(r"$\sigma_\chi$", *_B_SIGCHI, name="sigma_chi"))
        if not self._has_z:
            s.append(ParamSpec(r"$\gamma$", *_B_GAMMA, name="gamma"))
        return s

    def prior_bounds(self):
        return pack_specs(*self.param_specs)

    def fiducial(self):
        """Fiducial vector: xi = 0 (flat field), central hyperparameters."""
        return [0.0 if sp.name.startswith("xi_") else float((sp.low + sp.high) * 0.5)
                for sp in self.param_specs]

    # -- binned field + bin lookup ----------------------------------------

    def _bin_log_rates(self, amp, ls_vec, xi):
        """Per-bin log-rate logn = (L xi) at the bin centres (zero mean)."""
        L, _ = _field_factor(self._Z, self._families, amp, ls_vec,
                             self._jitter * amp ** 2 + 1e-9)
        return jnp.clip(L @ xi, -_FIELD_CLIP, _FIELD_CLIP)         # (M,)

    def _binned_density(self, m1, q, z, logn):
        """exp(logn[bin]) / (q m1) inside the bin grid, 0 outside (incl. m2 > m1)."""
        m2 = q * m1
        i = jnp.searchsorted(self._ln_m_edges,
                             jnp.log(jnp.clip(m1, _LOGSAFE, None)), side="right") - 1
        j = jnp.searchsorted(self._ln_m_edges,
                             jnp.log(jnp.clip(m2, _LOGSAFE, None)), side="right") - 1
        valid = (i >= 0) & (i < self._n_m) & (j >= 0) & (j < self._n_m) & (m2 <= m1)
        tril = self._tril_map[jnp.clip(i, 0, self._n_m - 1),
                              jnp.clip(j, 0, self._n_m - 1)]
        valid = valid & (tril >= 0)
        if self._has_z:
            # The TOP z bin is open-ended: a query above ``z_edges[-1]`` keeps the
            # last bin's rate (constant comoving-frame extrapolation) instead of
            # being declared invalid.  Truncating there asserted ZERO merger rate
            # above z = 1.2 (the default top edge) while the rest of the pipeline
            # runs to zMax = 5, so every PE sample and injection above it was
            # silently dropped -- and any event whose PE support sat entirely above
            # it made its log-likelihood -inf for EVERY proposal, which the
            # likelihood's isfinite guard turns into a rejection of the whole
            # parameter space (nested sampling cannot even initialise live points),
            # with nothing pointing at the z edges.  Unlike the mass edges -- where
            # zero density outside the binned range IS the model statement, as it is
            # for every tapered parametric mass model -- a hard zero in z is not a
            # statement anyone chose.  Override ``DARKSIRENS_GPPOP_Z_EDGES`` to
            # resolve the high-z rate instead of extrapolating it.
            k = jnp.searchsorted(self._z_edges, z, side="right") - 1
            valid = valid & (k >= 0)
            flat = jnp.clip(tril, 0, self._n_tril - 1) * self._n_z \
                + jnp.clip(k, 0, self._n_z - 1)
        else:
            flat = jnp.clip(tril, 0, self._n_tril - 1)
        # Double-where the 1/(q*m1) divide: the (m1,q) norm mesh includes the
        # q=0 grid node (get_q_grid()[0]==0) where valid is False, but a single
        # outer where still lets reverse-mode differentiate the
        # exp(logn)/0 = inf branch (0*inf = NaN cotangent), poisoning
        # d/d(logn) at the fiducial. Guard the denominator so the inf never
        # forms (cf. the safe-pow double-where at parametric.py:277).
        safe_qm1 = jnp.where(valid, q * m1, 1.0)
        val = jnp.exp(logn[jnp.clip(flat, 0, self.M - 1)]) / safe_qm1
        return jnp.where(valid, val, 0.0)

    def _mass_norm(self, logn):
        """(m1, q)-grid normalisation of the z-independent mass shape (scalar)."""
        mg = get_mass_grid()
        qg = get_q_grid()
        M1, Q = jnp.meshgrid(mg, qg, indexing="ij")
        pun = self._binned_density(M1.ravel(), Q.ravel(),
                                   jnp.zeros(M1.size), logn).reshape(M1.shape)
        return _grid_integrate(pun, [mg, qg])

    def _spin_density(self, chi, mu_chi, sig_chi):
        cg = get_chi_grid()
        un = jnp.where((cg >= -1) & (cg <= 1),
                       jnp.exp(-0.5 * ((cg - mu_chi) / sig_chi) ** 2), 0.0)
        norm = jnp.trapezoid(un, cg)
        p = jnp.where((chi >= -1) & (chi <= 1),
                      jnp.exp(-0.5 * ((chi - mu_chi) / sig_chi) ** 2), 0.0)
        return p / jnp.where(norm > 0, norm, 1.0)

    # -- evaluation --------------------------------------------------------

    def log_p_pop(self, m1, q, z, chieff, theta):
        # Same broadcast contract as JointGPPopulation / AdditiveGPPopulation:
        # any broadcastable combination of scalars and arrays (including the WL
        # path's (Nsamp, Nnodes) mesh), with the output reshaped back to the
        # common shape -- so an all-scalar query returns a SCALAR.  Forcing the
        # shape from m1 alone (``atleast_1d(m1)`` + ``broadcast_to(..., m1.shape)``)
        # made a scalar m1 against vector q/z/chi a hard broadcast error and
        # turned an all-scalar query into shape (1,).
        m1, q, z, chi, _out_shape = _broadcast_logp_inputs(m1, q, z, chieff)

        # ---- unpack theta in param_specs order ----
        i = 0
        amp = jnp.exp(theta[i]); i += 1
        if self._has_z:
            ls_m = jnp.exp(theta[i]); ls_z = jnp.exp(theta[i + 1]); i += 2
            ls_vec = [ls_m, ls_m, ls_z]
        else:
            ls_m = jnp.exp(theta[i]); i += 1
            ls_vec = [ls_m, ls_m]
        xi = theta[i:i + self.M]; i += self.M
        mu_chi = theta[i]; sig_chi = theta[i + 1]; i += 2
        gamma = None if self._has_z else theta[i]

        logn = self._bin_log_rates(amp, ls_vec, xi)
        p_un = self._binned_density(m1, q, z, logn)
        if not self._has_z:
            # Mass-only: normalise the z-independent mass shape over (m1, q).
            norm = self._mass_norm(logn)
            p_un = p_un / jnp.where(norm > 0, norm, 1.0)
        # gppop_mz: leave unnormalised over z -- the z-dependence is the free-form
        # rate R(z); its overall scale cancels with the rate in the likelihood.

        p = p_un * self._spin_density(chi, mu_chi, sig_chi)
        log_p = jnp.where(p > 0.0, jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny)), -jnp.inf)
        if self._has_z:
            # gppop_mz z-bins carry the source-frame R(z); apply the 1/(1+z)
            # observer/source time dilation here (the likelihood's log1p is the
            # detector-mass Jacobian, not dilation; cf. the parametric powerlaw
            # (gamma - 1) whose -1 IS this factor).
            return (log_p - jnp.log1p(z)).reshape(_out_shape)
        return (log_p + (gamma - 1.0) * jnp.log1p(z)).reshape(_out_shape)


# ============================================================
# Model menu + lazy factory (consumed by registry.register_model)
# ============================================================

def _joint(axes, latex):
    return lambda mode="shape": JointGPPopulation(tuple(axes), mean_mode=mode, latex=latex)


_GP_BUILDERS: dict = {}
GP_LATEX: dict = {}


def _reg(name, builder, latex):
    _GP_BUILDERS[name] = builder
    GP_LATEX[name] = latex


# 1-D
_reg("gp1d_m1",  _joint(("m1",),  "GP m1"),  "GP m1")
_reg("gp1d_q",   _joint(("q",),   "GP q"),   "GP q")
_reg("gp1d_chi", _joint(("chi",), "GP chi"), "GP chi")
_reg("gp1d_z",   _joint(("z",),   "GP z"),   "GP z")
# 2-D
_reg("gp2d_m1_q",   _joint(("m1", "q"),   "GP m1,q"),   "GP m1,q")
_reg("gp2d_m1_chi", _joint(("m1", "chi"), "GP m1,chi"), "GP m1,chi")
_reg("gp2d_m1_z",   _joint(("m1", "z"),   "GP m1,z"),   "GP m1,z")
_reg("gp2d_q_chi",  _joint(("q", "chi"),  "GP q,chi"),  "GP q,chi")
_reg("gp2d_q_z",    _joint(("q", "z"),    "GP q,z"),    "GP q,z")
_reg("gp2d_chi_z",  _joint(("chi", "z"),  "GP chi,z"),  "GP chi,z")
# 3-D
_reg("gp3d_m1_q_chi", _joint(("m1", "q", "chi"), "GP m1,q,chi"), "GP m1,q,chi")
_reg("gp3d_m1_q_z",   _joint(("m1", "q", "z"),   "GP m1,q,z"),   "GP m1,q,z")
_reg("gp3d_m1_chi_z", _joint(("m1", "chi", "z"), "GP m1,chi,z"), "GP m1,chi,z")
_reg("gp3d_q_chi_z",  _joint(("q", "chi", "z"),  "GP q,chi,z"),  "GP q,chi,z")
# 4-D joint
_reg("gp4d", _joint(("m1", "q", "chi", "z"), "GP 4D"), "GP 4D")
# Separable null: independent 1-D GP per probability axis, no interactions
# ``mode`` is accepted and ignored: the additive family is zero-mean, so
# ``@shape``/``@zero`` decorations name the same model.
_reg("gp_separable",
     lambda mode="shape": AdditiveGPPopulation(
         terms=(("m1",), ("q",), ("chi",)), latex="GP sep"),
     "GP sep")
# Additive (functional ANOVA): mains + m1q + m1z + chiz interactions
_reg("gp4d_additive",
     lambda mode="shape": AdditiveGPPopulation(latex="GP add"),
     "GP add")
# Binned GP (gppop): mass-only and full (m1, m2, z)
_reg("gppop",
     lambda mode="shape": BinnedGPPopulation(mass_edges=_GPPOP_M_EDGES),
     "gppop")
_reg("gppop_mz",
     lambda mode="shape": BinnedGPPopulation(mass_edges=_GPPOP_M_EDGES,
                                             z_edges=_GPPOP_Z_EDGES),
     "gppop m,z")


GP_MODEL_NAMES = tuple(_GP_BUILDERS.keys())


def build_gp_model(name: str):
    """Construct a GP population model by registry name (no tinygp import)."""
    base, _, suffix = name.partition("@")          # optional "@zero" / "@shape"
    mode = suffix or "shape"
    if base not in _GP_BUILDERS:
        raise KeyError(f"unknown GP model {name!r}; known: {sorted(_GP_BUILDERS)}")
    return _GP_BUILDERS[base](mode)


def gp_fiducial(name: str):
    """Fiducial vector for a GP model (xi = 0, central hyperparameters)."""
    return tuple(build_gp_model(name).fiducial())
