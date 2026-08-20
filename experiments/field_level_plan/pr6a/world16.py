"""The nside-16 mock world the PR-6a closure tiers (PLAN §6.2) run in.

One construction, shared by Tiers A-D, so that "the truth" means the same
object in every tier and a Tier-D stress is a one-argument change to the
same generator rather than a second pipeline.

Three things here are decisions, not conveniences, and each is argued where
it is defined:

1. **The completeness map is the real DESI map, degraded** (:func:`footprint`).
   PLAN §6.1 asks for "a completeness map ``f_p`` with the measured DESI
   ``masked_frac`` distribution (mean 0.137, sd 0.104, p99 0.635)".  Rather
   than sample that distribution we take the shipped
   ``experiments/desi_ingest/data/mth_map_nside128.h5`` -- the map the
   production line actually consumes -- and degrade it to nside 16.  Measured
   on that file at its native nside 64: 17009 of 49152 pixels have
   ``f_p = 0`` (34.6% of the sky), and over the 32143 non-zero pixels
   ``f_p`` has mean 0.8543 and sd 0.1087, i.e. ``masked_frac`` mean 0.1457
   and sd 0.1087 -- the plan's numbers, recovered from the artifact.
   Degrading to nside 16 averages 64 sub-pixels, so the surviving
   ``f_p`` sd is 0.045, NOT 0.109; the native scatter is what Tier D(i)
   re-injects, which is the correct place for it.

2. **The count draw is Poisson on a FINE z grid, then thinned into shells**
   (:func:`draw_counts`).  Drawing directly from the fit operator's own
   ``phi_shell`` would make the truth exactly representable and the tier
   would measure nothing but the linear algebra.  Instead the truth field is
   evaluated on the 400-node fine grid, the Poisson intensity is
   ``lambda_pn = n0_eff * base(z_n) * dz_n * f_p * exp(b_gal f(p, z_n))``,
   and each fine node is scattered into the shells by the SAME Gaussian
   photo-z mass ``W`` models.  Poisson is closed under thinning and
   superposition, so the shell counts are exactly
   ``N_pg ~ Poisson(sum_n lambda_pn mass[g, n])`` -- no galaxy loop needed,
   and the within-shell Jensen gap of PLAN §0.5 D1 ("the within-shell
   linearization is a stated premise") is LIVE: the truth carries
   ``sum_n w_gn e^{b f_pn}`` while the model carries ``e^{b sum_n w_gn f_pn}``.
   Tier A reports both this draw and the exactly-representable one so that a
   failure can be attributed.

3. **Truth is drawn on ``[0, z_depth]`` only.**  Galaxies beyond the survey
   depth scattering IN across the ``z_depth`` edge is an unmodelled
   contamination (``W``'s fine grid stops at ``z_depth``, so the shipped
   builder cannot see it).  That is a misspecification, and it belongs in
   Tier D, not in the estimator tier.  Scatter OUT is modelled correctly on
   both sides and is left in.

The basis geometry is the production convention scaled to nside 16:
``factored-v1`` jitter, ``ls_sph = 0.5`` with ``M_sph = 64``
(``sqrt(4 pi / 64) = 0.443 <= 0.5``, the builder's resolution guard),
``ls_z = 0.10`` with ``M_z = 5`` (``log1p(0.3)/4 = 0.0656 <= 0.10``).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: The grid this whole tier set is pinned to, exactly as
#: ``experiments/desi_full259/common.py`` pins its own.  The mock's detected
#: events reach ``dL = 1394 Mpc``, which at the top of the ``H0`` scan
#: (``H0 = 140``, ``Om0 = 0.3089``) is ``z ~ 0.48``; 1.5 clears it with margin
#: and still leaves 287 of the 1000 grid nodes below the ``z_depth = 0.30``
#: the field lives on.  It MUST agree between the mock build, the anchor
#: build and every inference call: ``latent_q.load_latent_plan`` refuses an
#: artifact whose ``z_sub`` is not the run grid's below-depth prefix.
ZMAX = "1.5"
os.environ.setdefault("DARKSIRENS_ZMAX", ZMAX)
if os.environ["DARKSIRENS_ZMAX"] != ZMAX:
    raise RuntimeError(
        f"DARKSIRENS_ZMAX={os.environ['DARKSIRENS_ZMAX']} conflicts with the "
        f"pr6a pin {ZMAX}; unset it or fix world16.py")

PR6A_DIR = Path(__file__).resolve().parent
PLAN_DIR = PR6A_DIR.parent
REPO_DIR = PLAN_DIR.parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.redshift.latent_field import (  # noqa: E402
    build_latent_basis, shell_response,
)
from darksirens.redshift.latent_counts import (  # noqa: E402
    TracerCounts, make_count_operator,
)

#: The shipped depth map the production line consumes.
MTH_MAP = REPO_DIR / "experiments" / "desi_ingest" / "data" / "mth_map_nside128.h5"

CLIGHT = 299792.458

# ------------------------------------------------------------------ geometry

NSIDE = 16
Z_DEPTH = 0.30
N_SHELLS = 12
M_SPH = 64
M_Z = 5
LS_SPH = 0.5
LS_Z = 0.10
N_FINE = 400
#: The catalog redshift uncertainty the WORLD assumes -- it becomes
#: ``world.meta["sigma_z"]`` and therefore the shell response ``W`` that
#: ``build_anchor16`` builds the count channel's forward model from.
#:
#: **1e-4, changed from 0.023 (2026-08-19), and it must track
#: ``make_mock.SIGMA_Z_CAT``** -- which is 1e-4 and not 0.003 because the
#: analysis adds ``sigma_kde = 0.003`` in quadrature, so a 0.003 catalog scatter
#: would make the kernel 41% over-wide (see that constant's note).  The photometric 0.023 was measured to be the
#: sole remaining cause of the closure tiers' bias and overconfidence (a 21%
#: fractional kernel at the median host `z`; see ``make_mock.SIGMA_Z_CAT`` and
#: ``gate_specz.py``).  Leaving THIS at 0.023 while the catalog moved to 0.003
#: would make the latent arm measure a photo-`z` MISMATCH between the shell
#: response and the catalog rather than the field -- so the two are changed
#: together, deliberately, and the anchors must be rebuilt.
#:
#: ``build_anchor16`` rebuilds ``W`` from ``world.meta`` on every call, so this
#: is safe for the mock.  The PRODUCTION anchor is built by
#: ``cli/build_latent_field.py`` from its own depth map and is untouched.
SIGMA_Z = 1.0e-4
B_GAL = 1.0

#: Mock survey magnitude limit.  With the mock LF ``M ~ N(-21, 1)`` the
#: distance modulus at ``z = 0.30`` is 40.95, so ``m_lim = 20.0`` puts
#: ``C(z_depth) ~ 0.5``: the selection roll-off bites well inside the depth,
#: which is what makes ``base(z)`` (and therefore ``W``) non-trivial.
M_LIM = 20.0
#: The mock's PHYSICAL luminosity-function mean (the number
#: ``rng.normal(M0_PHYS, SIGMA_M)`` draws absolute magnitudes from).
M0_PHYS = -21.0
SIGMA_M = 1.0
DELTA = 0.0
OM0 = 0.3075          # = Om0Planck, the background the selection fit stamps
H0_TRUE = 67.74
#: The h-SCALED zero point, which is what ``c_sel_gaussian`` and every
#: selection-fit JSON carry: ``selection.py:16-27`` defines the H0 firewall as
#: ``M0hat = M0 - 5 log10 h``, so at ``h = 0.6774`` the mock's ``M0 = -21``
#: is ``M0hat = -20.1542``.  Getting this wrong is not a convention quibble:
#: passing ``-21`` where ``M0hat`` is expected puts ``C(z_depth)`` at 0.795
#: instead of 0.491, i.e. it mis-states the missing budget at the survey edge
#: by 62%.  Verified against the mock's own selection: the empirical
#: ``Phi((m_lim - M0 - DM(z))/sigma)`` and ``c_sel_gaussian(z, m_lim, M0_HAT,
#: SIGMA_M, H0_TRUE, OM0)`` agree to 1e-4 at z = 0.15 and z = 0.30.
M0_HAT = M0_PHYS - 5.0 * np.log10(H0_TRUE / 100.0)


def _dc_grid(H0, Om0, z):
    E = np.sqrt(Om0 * (1 + z) ** 3 + (1 - Om0))
    dc = np.concatenate([[0.0], np.cumsum(
        0.5 * (1 / E[1:] + 1 / E[:-1]) * np.diff(z))]) * (CLIGHT / H0)
    return dc, E


def dvc_dz(H0, Om0, z):
    """``dV_c/dz`` in Mpc^3 (4 pi steradian)."""
    dc, E = _dc_grid(H0, Om0, z)
    return 4.0 * np.pi * dc ** 2 * (CLIGHT / H0) / E


def c_sel(z, *, H0=H0_TRUE, Om0=OM0, m_lim=M_LIM, M0hat=M0_HAT,
          sigma_M=SIGMA_M):
    """The magnitude-limited selection fraction, the SHIPPED function.

    Routed through :func:`darksirens.redshift.selection.c_sel_gaussian` so
    the mock's selection and the inference's selection are the same code,
    not two transcriptions of one formula (the mock LF is Gaussian and no
    K-correction is applied, which is exactly what ``c_sel_gaussian``
    assumes -- see the ``--survey-magnitude-limit`` note in
    ``scripts/mock_dark_sirens/generate_mock_data.py``).
    """
    from darksirens.redshift.selection import c_sel_gaussian
    return np.asarray(c_sel_gaussian(jnp.asarray(np.atleast_1d(z)), m_lim,
                                     M0hat, sigma_M, H0, Om0,
                                     k_corr_coeffs=None))


def base_curve(z, *, delta=DELTA, **kw):
    """``base(z; theta) = C(z) (1+z)^delta dV_c/dz`` -- the within-shell
    weight ``W`` normalizes away and the count intensity carries."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    return (c_sel(z, **kw) * (1.0 + z) ** delta
            * dvc_dz(kw.get("H0", H0_TRUE), kw.get("Om0", OM0), z) + 1e-300)


# ---------------------------------------------------------------- footprint

def footprint(nside=NSIDE, *, cov_min=0.9):
    """The real DESI depth map, degraded to ``nside``.

    Returns ``(pix, f_p)``: the footprint pixel ids (those whose sub-pixels
    are >= ``cov_min`` covered at nside 128) and the mean non-zero ``f_p``
    inside each.  Measured at nside 16, ``cov_min = 0.9``: 1854 of 3072
    pixels (60.4% of the sky), ``f_p`` mean 0.8722, sd 0.0450, min 0.6136.
    """
    import healpy as hp

    from darksirens.catalogs.depth_map import load_selection_fraction

    fp128 = np.asarray(load_selection_fraction(str(MTH_MAP), 128).f_p)
    occ = (fp128 > 0).astype(float)
    cov = hp.ud_grade(occ, nside, order_in="RING", order_out="RING")
    mean_all = hp.ud_grade(fp128, nside, order_in="RING", order_out="RING")
    f_p = np.where(cov > 0, mean_all / np.maximum(cov, 1e-12), 0.0)
    keep = np.where(cov >= cov_min)[0]
    return keep.astype(np.int64), np.clip(f_p[keep], 1e-3, 1.0)


# -------------------------------------------------------------------- world

@dataclass(frozen=True)
class World:
    nside: int
    pix: Any                 # (n_fit,) footprint pixel ids
    f_p: Any                 # (n_fit,) selection fraction
    basis: Any               # LatentBasis with footprint_rows = arange(n_fit)
    edges: Any               # (G+1,) shell edges in z
    z_fine: Any              # (N_fine,)
    W: Any                   # (G, N_fine) the frozen shell response
    mass: Any                # (G, N_fine) photo-z shell mass, UNweighted
    base: Any                # (N_fine,) base(z_fine)
    b_gal: float
    meta: dict


def build_world(*, nside=NSIDE, z_depth=Z_DEPTH, n_shells=N_SHELLS,
                m_sph=M_SPH, m_z=M_Z, ls_sph=LS_SPH, ls_z=LS_Z,
                sigma_z=SIGMA_Z, b_gal=B_GAL, delta=DELTA, cov_min=0.9,
                ls_sph_fit=None, full_sky=False) -> World:
    """Basis + footprint + shell response at one geometry.

    ``ls_sph_fit`` overrides the sphere length scale used to BUILD the basis
    while leaving the geometry otherwise fixed -- Tier D(iii) ("``ls_ang``
    wrong by 2x") is exactly a world built with the truth's ``ls_sph`` and
    fitted with another, so the two are separated here rather than in the
    tier script.
    """
    import healpy as hp

    from scipy.special import erf

    if full_sky:
        # The diagnostic ladder's rung 0: every pixel in the footprint and
        # ``f_p == 1``, i.e. a survey with no areal masking and no footprint
        # edge.  It exists so that a closure failure can be attributed to one
        # feature at a time instead of to "the mock".
        pix = np.arange(hp.nside2npix(nside), dtype=np.int64)
        f_p = np.ones(pix.size)
    else:
        pix, f_p = footprint(nside, cov_min=cov_min)
    d_sph = np.sqrt(4 * np.pi / m_sph)
    if d_sph > ls_sph:
        raise SystemExit(
            f"resolution guard (the builder's, cli/build_latent_field.py:101): "
            f"node spacing {d_sph:.3f} > ls_sph {ls_sph}")
    dz_node = np.log1p(z_depth) / (m_z - 1)
    if dz_node > ls_z:
        raise SystemExit(
            f"resolution guard: zeta node spacing {dz_node:.4f} > ls_z {ls_z}")

    edges = np.linspace(0.0, z_depth, n_shells + 1)
    z_fine = np.linspace(1e-4, z_depth, N_FINE)
    # ALL-SKY rows with the footprint as ``proj_sph``: the count channel and
    # the sky moments live on ``F`` (``proj_sph``), but the mock's galaxies --
    # and therefore the GW HOSTS, including the ones the survey never sees --
    # live everywhere, which is PLAN §6.1's extension (ii).
    vec = np.column_stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))

    from darksirens.redshift import zgrid
    z_sub = np.asarray(zgrid[zgrid <= z_depth])

    basis = build_latent_basis(
        vec, np.log1p(z_sub), n_inducing_sphere=m_sph, n_inducing_z=m_z,
        z_node_hi=z_depth, ls_sph=(ls_sph if ls_sph_fit is None else ls_sph_fit),
        ls_z=ls_z, zeta_fine=np.log1p(z_fine), footprint_rows=pix)

    base = base_curve(z_fine, delta=delta)
    sig = np.full_like(z_fine, float(sigma_z))
    lo = (edges[:-1, None] - z_fine[None, :]) / (np.sqrt(2.0) * sig[None, :])
    hi = (edges[1:, None] - z_fine[None, :]) / (np.sqrt(2.0) * sig[None, :])
    mass = 0.5 * (erf(hi) - erf(lo))                      # (G, N_fine)
    W = shell_response(edges, z_fine, lambda z: np.full_like(z, sigma_z),
                       lambda z: base_curve(z, delta=delta))

    return World(nside=nside, pix=pix, f_p=f_p, basis=basis, edges=edges,
                 z_fine=z_fine, W=np.asarray(W), mass=mass, base=base,
                 b_gal=float(b_gal),
                 meta=dict(nside=nside, z_depth=z_depth, n_shells=n_shells,
                           m_sph=m_sph, m_z=m_z, ls_sph=ls_sph,
                           ls_sph_fit=ls_sph_fit, ls_z=ls_z, sigma_z=sigma_z,
                           b_gal=b_gal, delta=delta, n_fit=int(pix.size),
                           z_sub_n=int(z_sub.size), rank=m_sph * m_z))


# ------------------------------------------------------------------- truth

def draw_xi_true(world: World, rng, *, lognormal_tail=0.0):
    """``xi_true ~ N(0, I_M)`` -- the prior the MAP is regularized by.

    ``lognormal_tail`` (Tier D-iv) mixes in a heavy, skewed component:
    ``xi -> (g + a (chi - 1)/sqrt 2) / sqrt(1 + a^2)`` with ``chi``
    chi-squared_1.  The renormalization is the point: it keeps the marginal
    variance at exactly 1 so the stress is PURE non-Gaussianity (skewness
    ``a^3 2 sqrt 2 / (1+a^2)^{3/2}``, excess kurtosis ``12 a^4/(1+a^2)^2``)
    rather than a field-amplitude change wearing a tail's clothes -- an
    un-normalized mixture at ``a = 0.5`` would also inflate the variance by
    25%, and the resulting ``H0`` shift could not be attributed.
    """
    M = world.meta["rank"]
    g = rng.standard_normal(M)
    if not lognormal_tail:
        return g
    a = float(lognormal_tail)
    c = rng.chisquare(1, M)
    return (g + a * (c - 1.0) / np.sqrt(2.0)) / np.sqrt(1.0 + a ** 2)


def field_fine(world: World, xi, *, all_sky=False):
    """``f(p, z_fine) = Phi_s Xi Phi_z_fine^T`` on the footprint (default) or
    on every pixel (``all_sky=True``, the host-placement rows)."""
    Xi = np.asarray(xi).reshape(world.meta["m_sph"], world.meta["m_z"])
    P = np.asarray(world.basis.phi_sph if all_sky else world.basis.proj_sph)
    return P @ Xi @ np.asarray(world.basis.phi_z_fine).T


def mean_one_shift(world: World, *, all_sky=False):
    """``-b^2 v(p, z)/2`` with ``v`` the Nystrom prior variance -- the
    mean-one lognormal convention this repo's own mock generator uses
    (``experiments/completeness_viz/generate_clustered_mock.py``: "mean-one
    lognormal shift -sigma^2/2").  With it, ``n0`` is the TRUE comoving
    density and the mock's ``log10n0`` calibration is exact by construction
    rather than off by ``E[e^{bf}] = e^{b^2 v/2}`` (= 1.44 at ``v ~ 0.85``).

    Measured on this basis, ``v`` runs 0.988-1.000 across pixels, so the shift
    is essentially a constant: the count channel's shell-total conditioning
    removes it exactly, which is why Tier A is unaffected either way.
    """
    ph = np.asarray(world.basis.phi_sph if all_sky else world.basis.proj_sph)
    vs = (ph ** 2).sum(axis=1)
    vz = (np.asarray(world.basis.phi_z_fine) ** 2).sum(axis=1)
    return -0.5 * (world.b_gal ** 2) * vs[:, None] * vz[None, :]


def draw_counts(world: World, xi_true, rng, *, n_target=1.1e6,
                f_p_draw=None, extra_incompleteness=None,
                representable=False):
    """Poisson shell counts (n_fit, G) from the truth field.

    ``f_p_draw`` is the completeness the TRUTH uses; the fit always uses
    ``world.f_p``.  Tier D(i) passes a perturbed map here and leaves the fit
    map alone -- that is what "the completeness map is wrong" means.
    ``extra_incompleteness(z, delta_field)`` is an additional multiplicative
    factor on the intensity (Tier D-ii's fibre-assignment proxy).

    ``representable=True`` bypasses the fine grid and draws from the FIT
    operator's own ``phi_shell``, i.e. from a truth the model can represent
    exactly.  It exists only so Tier A can attribute a failure to the
    within-shell linearization rather than to the estimator.
    """
    f_draw = world.f_p if f_p_draw is None else np.asarray(f_p_draw)
    dzn = np.gradient(world.z_fine)

    if representable:
        phi_shell = np.asarray(world.W) @ np.asarray(world.basis.phi_z_fine)
        Xi = np.asarray(xi_true).reshape(world.meta["m_sph"],
                                         world.meta["m_z"])
        eta = world.b_gal * (np.asarray(world.basis.proj_sph) @ Xi
                             @ phi_shell.T)                    # (n_fit, G)
        shell_w = (world.mass * (world.base * dzn)[None, :]).sum(1)  # (G,)
        mu = f_draw[:, None] * np.exp(eta) * shell_w[None, :]
    else:
        f = field_fine(world, xi_true)                         # (n_fit, N_fine)
        lam = (f_draw[:, None] * np.exp(world.b_gal * f)
               * (world.base * dzn)[None, :])                  # (n_fit, N_fine)
        if extra_incompleteness is not None:
            lam = lam * extra_incompleteness(world.z_fine, f)
        mu = lam @ world.mass.T                                # (n_fit, G)

    mu = mu * (float(n_target) / mu.sum())
    return rng.poisson(mu).astype(float), mu


def solve_damped(op, *, n_iter=60, tol=1e-9, max_backtrack=30, c1=1e-4):
    """Backtracking-damped Newton on the SAME ``J``/``grad``/``H`` as shipped.

    **This exists because the shipped solve diverges, and that is a finding,
    not a preference.**  ``latent_counts.count_map_solve`` takes a FIXED trip
    count of UNDAMPED Fisher steps and its docstring states "``H >= I`` makes
    the un-damped Fisher step globally well-posed for this objective in
    practice".  Measured here at nside 16 with 1.1e6 galaxies over 1854 x 12
    voxels, on ``xi_true`` drawn from the model's own prior: realization
    ``seed = 6100 + 977`` leaves ``xi = 0`` with a Newton step of length 41.5
    (against ``||xi_true|| = 16.9``), ``J`` RISES from 8.578e6 to 1.001e7 at
    trip 1, and the iteration runs away to ``|dx| ~ 1.4e6`` and settles into a
    period-2 limit cycle with ``grad_inf ~ 4.4e5``.  For the multinomial-logit
    the canonical link makes the Fisher information equal to the observed
    Hessian, so ``count_map_solve`` is EXACT Newton on a convex objective --
    and exact Newton on a convex objective is only locally convergent; global
    convergence needs a line search.  ``H >= I`` bounds the step, it does not
    make it a descent step of acceptable length.

    The failure is LOUD, not silent: ``cli/build_latent_field.py:170`` gates
    on ``grad_inf > 1e-8`` and raises "P6 gate: anchor solve did not converge",
    and the production anchor converged (``grad_inf = 1.09e-10``,
    ``latent_anchor_v2a.h5``).  So this is a robustness defect in the builder,
    not a correctness defect in any shipped artifact.

    Nothing else changes: the direction, the objective and the Hessian are the
    shipped functions, called unmodified.  Armijo backtracking with a fixed
    maximum keeps the solve a deterministic function of the inputs (PLAN §10's
    requirement: no history-dependent stopping).
    """
    from darksirens.redshift.latent_counts import (
        gradient, hessian_separable, objective)

    xi = jnp.zeros(op.rank)
    J = objective(xi, op)
    n_bt_total = 0
    for _ in range(int(n_iter)):
        g = gradient(xi, op)
        if float(jnp.max(jnp.abs(g))) < tol:
            break
        H = hessian_separable(xi, op)
        L = jnp.linalg.cholesky(H)
        dx = jax.scipy.linalg.cho_solve((L, True), g)
        slope = float(jnp.dot(g, dx))               # > 0: dx is a descent dir
        t = 1.0
        # The Armijo test needs an absolute slack: ``J ~ 8e6`` on this mock, so
        # near the optimum the true decrease (~1e-10) is BELOW the f64
        # resolution of ``J`` itself (eps |J| = 1.8e-9) and a strict test
        # backtracks to ``t = 2^-30`` while the gradient sits at 1e-6.
        # Measured: without this slack one realization in eight stalls at
        # ``grad_inf = 1.4e-6``, i.e. it fails P6 for a floating-point reason
        # and not a convergence one.
        slack = 1e-12 * abs(float(J))
        for _ in range(int(max_backtrack)):
            xn = xi - t * dx
            Jn = objective(xn, op)
            if float(Jn) <= float(J) - c1 * t * slope + slack:
                break
            t *= 0.5
            n_bt_total += 1
        xi, J = xn, Jn
    H = hessian_separable(xi, op)
    L = jnp.linalg.cholesky(H)
    g = gradient(xi, op)
    return dict(xi_hat=xi, H_chol=L, grad_inf=jnp.max(jnp.abs(g)),
                J=objective(xi, op), n_backtrack=n_bt_total)


def fit_operator(world: World, counts, *, f_p_fit=None, b_gal=None):
    """The count operator the solve consumes -- the SHIPPED constructor."""
    tracer = TracerCounts(
        pix=world.pix, counts=np.asarray(counts),
        completeness=(world.f_p if f_p_fit is None else np.asarray(f_p_fit)),
        bias=(world.b_gal if b_gal is None else float(b_gal)))
    return make_count_operator(world.basis.proj_sph,
                               world.basis.phi_z_fine, world.W, tracer)


__all__ = ["World", "build_world", "draw_xi_true", "draw_counts",
           "fit_operator", "solve_damped", "mean_one_shift",
           "field_fine", "footprint", "base_curve", "c_sel",
           "dvc_dz", "PR6A_DIR", "PLAN_DIR", "REPO_DIR",
           "NSIDE", "Z_DEPTH", "N_SHELLS", "M_SPH", "M_Z", "LS_SPH", "LS_Z",
           "SIGMA_Z", "B_GAL", "M_LIM", "M0_HAT", "M0_PHYS", "SIGMA_M", "OM0", "H0_TRUE"]
