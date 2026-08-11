"""PHY-3 + PHY-6: q-axis GP normalisation must be conditional on m1, and the
z-normalisation ceiling must track the shared analysis redshift grid.

PHY-3 -- ``JointGPPopulation._normalise`` filled absent probability axes with a
placeholder ``m1 = 1``.  For the four models with ``q`` in ``gp_axes`` but
``m1`` NOT (``gp1d_q``, ``gp2d_q_chi``, ``gp2d_q_z``, ``gp3d_q_chi_z``) the
``m2 = q*m1`` secondary cut then evaluated ``m2 = q <= 1 < m_min``, zeroing the
whole norm grid; the ``norm > 0`` guard silently divided the density by 1.  The
GP factor was therefore unnormalised AND acquired a spurious m1--q coupling (the
numerator's taper used the real query m1).  Measured pre-fix conditional
q-integrals were 0.108 / 0.207 / 0.210 at m1 = 10 / 20 / 50 (each should be 1).
The fix tabulates the norm on a log-m1 grid and interpolates it per query.

PHY-6 -- the z-normalisation interpolation ceiling ``_ZNORM_HI`` defaulted to
3.0 while the shared ``darksirens.redshift.grid.zMax`` defaults to 5.0, silently
freezing every z-conditioned GP normaliser on z in (3, 5].  The ceiling now
tracks ``zMax``.

Normalisation-isolation trick used throughout: ``exp(log_p_pop)`` factorises as
``p_gp(prob axes | m1, z) * Prod(baseline factors) * (1+z)**(gamma-1)``.
Dividing out the parametric baselines (mass / pairing / spin) and the rate term
leaves the GP factor, whose integral over its probability axes must be 1.  When
the pairing baseline (which carries an ``m2 = q*m1`` cut) would gate a region
where p(m1|z) is nonzero, we instead integrate over q as well -- ``pairing(q|m1)``
integrates to 1 over q at each m1, collapsing the double integral to
``int p(m1|z) dm1``.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

from darksirens.gw.populations.gp import build_gp_model, _ZNORM_HI
from darksirens.redshift.grid import zMax

# Some parametric test modules insert a tinygp STUB into sys.modules; GP models
# then cannot evaluate.  Follow the registry-golden convention: a stub (no
# __file__) counts as "tinygp unavailable" (see test_pop_gradient_nan_safety).
try:
    import tinygp  # noqa: F401
    HAVE_TINYGP = getattr(tinygp, "__file__", None) is not None
except Exception:  # pragma: no cover
    HAVE_TINYGP = False

_NEED_TINYGP = pytest.mark.skipif(
    not HAVE_TINYGP, reason="tinygp unavailable (or stubbed by a parametric test module)"
)

_Q_MODELS = ["gp1d_q", "gp2d_q_chi", "gp2d_q_z", "gp3d_q_chi_z"]


def _params(model, theta):
    idx = {sp.name: k for k, sp in enumerate(model.param_specs)}
    return lambda name: theta[idx[name]]


def _seed_theta(model, seed=0, scale=0.6, overrides=None):
    """Fiducial vector with the whitened GP latents ``xi_*`` seeded nonzero.

    xi = 0 makes the field equal its (mass/pairing-shaped) mean, which is
    already correctly normalised; seeding xi exercises a genuinely non-trivial
    GP shape so the normalisation is a real test.
    """
    theta = list(model.fiducial())
    names = [sp.name for sp in model.param_specs]
    for name, value in (overrides or {}).items():
        theta[names.index(name)] = float(value)
    rng = np.random.default_rng(seed)
    for k, sp in enumerate(model.param_specs):
        if sp.name.startswith("xi_"):
            theta[k] = float(scale * rng.standard_normal())
    return jnp.asarray(theta, dtype=float)


@_NEED_TINYGP
@pytest.mark.parametrize("name", _Q_MODELS)
def test_conditional_q_integral_is_unity(name):
    """Core PHY-3 regression: int p_gp(q[,chi] | m1[,z]) = 1 for the q-in-GP,
    m1-not-in-GP models, at off-node m1 (and, for z-models, off-node z)."""
    model = build_gp_model(name)
    theta = _seed_theta(model, seed=0, scale=0.6)
    P = _params(model, theta)
    chi_in = "chi" in model.gp_axes
    z_in = "z" in model.gp_axes

    qg = jnp.linspace(0.0, 1.0, 400)
    cg = jnp.linspace(-1.0, 1.0, 200)
    zs = (0.05, 1.0, 4.5) if z_in else (0.3,)

    for m1v in (8.3, 21.7, 55.1):
        mass = model._baseline_mass(m1v, P("alpha_mass"), P("m_min"),
                                    P("dm_min"), P("m_max"), P("dm_max"))
        for zv in zs:
            rate = (1.0 + zv) ** (P("gamma") - 1.0)
            if chi_in:
                Q, C = jnp.meshgrid(qg, cg, indexing="ij")
                dens = jnp.exp(model.log_p_pop(
                    jnp.full(Q.size, m1v), Q.ravel(),
                    jnp.full(Q.size, zv), C.ravel(), theta)).reshape(Q.shape)
                pgp = dens / (rate * mass)
                integ = jnp.trapezoid(jnp.trapezoid(pgp, cg, axis=-1), qg)
            else:
                chi0 = 0.1
                spin = model._baseline_spin(chi0, P("mu_chi"), P("sigma_chi"))
                dens = jnp.exp(model.log_p_pop(
                    jnp.full_like(qg, m1v), qg,
                    jnp.full_like(qg, zv), jnp.full_like(qg, chi0), theta))
                pgp = dens / (rate * mass * spin)
                integ = jnp.trapezoid(pgp, qg)

            # Taper-toe pinch: at the lowest m1 the norm curves steeply in
            # log-m1 as the q-support turns on, so the fixed 64-node log-m1
            # interpolation carries a few-% residual there; m1 >= 20 is <1%.
            rtol = 6e-2 if m1v < 12.0 else 2e-2
            assert abs(float(integ) - 1.0) < rtol, (name, m1v, zv, float(integ))


@_NEED_TINYGP
def test_shape_mean_slope_is_alpha_gp_over_the_whole_mass_range():
    """The parametric mean must never be clipped.

    At xi = 0 the field IS its mean ``-alpha_gp log m1``, so the GP factor is a
    pure power law and ``d log p_gp / d log m1 == -alpha_gp`` everywhere inside
    the taper plateau.  Clipping ``mean + deviation`` (instead of the deviation
    alone) truncated the mean at ``_FIELD_CLIP`` = 10 nats, i.e. above
    ``m1 = exp(10/alpha_gp)`` = 12.2 Msun at the fiducial alpha_gp = 4: measured
    slopes were -0.53 (12 -> 20 Msun) and -0.07 (20 -> 40 Msun) instead of -4.
    """
    model = build_gp_model("gp1d_m1")
    theta = jnp.asarray(model.fiducial(), dtype=float)
    P = _params(model, theta)
    alpha_gp = float(P("alpha_gp"))

    m1 = jnp.asarray([15.0, 20.0, 30.0, 45.0])
    q, chi, z = 0.9, 0.05, 0.3
    lp = np.asarray(model.log_p_pop(m1, jnp.full(m1.size, q), jnp.full(m1.size, z),
                                   jnp.full(m1.size, chi), theta))
    # Divide out the parametric baselines and the rate term (file-wide idiom) so
    # only the GP factor's m1 dependence is left.
    pair = np.asarray(model._baseline_pairing(m1, jnp.full(m1.size, q), P("beta_q"),
                                              P("m_min"), P("dm_min")))
    spin = float(model._baseline_spin(chi, P("mu_chi"), P("sigma_chi")))
    rate = (1.0 + z) ** (float(P("gamma")) - 1.0)
    log_gp = lp - np.log(pair * spin * rate)
    slopes = np.diff(log_gp) / np.diff(np.log(np.asarray(m1)))
    np.testing.assert_allclose(slopes, -alpha_gp, rtol=2e-2)


def test_znorm_hi_tracks_zmax():
    """PHY-6: the z-normalisation ceiling must never sit below the shared
    analysis redshift grid (no tinygp needed -- pure module constants)."""
    assert _ZNORM_HI >= zMax


@_NEED_TINYGP
def test_m1_norm_correct_at_high_z():
    """PHY-6 isolated: int p(m1 | z=4.5) dm1 = 1 for ``gp2d_m1_z``.  With a
    strong, long-length-scale z structure this integral collapses to
    norm(4.5)/norm(3.0) pre-fix (the frozen-at-3.0 ceiling) -- measured ~0.43 --
    and is unity once the ceiling tracks zMax."""
    model = build_gp_model("gp2d_m1_z")
    theta = _seed_theta(model, seed=2, scale=2.5,
                        overrides={"log_ls_z": math.log(0.8),
                                   "log_amp": math.log(2.5)})
    P = _params(model, theta)
    m1g = jnp.linspace(2.0, 150.0, 600)
    qg = jnp.linspace(0.0, 1.0, 400)
    M1, Q = jnp.meshgrid(m1g, qg, indexing="ij")
    zv, chi0 = 4.5, 0.1

    dens = jnp.exp(model.log_p_pop(
        M1.ravel(), Q.ravel(),
        jnp.full(M1.size, zv), jnp.full(M1.size, chi0), theta)).reshape(M1.shape)
    div = (1.0 + zv) ** (P("gamma") - 1.0) * model._baseline_spin(
        chi0, P("mu_chi"), P("sigma_chi"))
    # pairing(q|m1) integrates to 1 over q at each m1 -> double integral == int p(m1|z) dm1.
    integ = jnp.trapezoid(jnp.trapezoid(dens / div, qg, axis=-1), m1g)
    assert abs(float(integ) - 1.0) < 2e-2, float(integ)


@_NEED_TINYGP
def test_joint_m1_q_integral_unity_non_regression():
    """Non-regression: ``gp2d_m1_q`` (m1 IS a GP axis, so unaffected by the
    m1-conditioning branch) keeps its joint int p_gp(m1, q) = 1."""
    model = build_gp_model("gp2d_m1_q")
    theta = _seed_theta(model, seed=1, scale=0.6)
    P = _params(model, theta)
    m1g = jnp.linspace(2.0, 150.0, 600)
    qg = jnp.linspace(0.0, 1.0, 400)
    M1, Q = jnp.meshgrid(m1g, qg, indexing="ij")
    zv, chi0 = 0.3, 0.1

    dens = jnp.exp(model.log_p_pop(
        M1.ravel(), Q.ravel(),
        jnp.full(M1.size, zv), jnp.full(M1.size, chi0), theta)).reshape(M1.shape)
    div = (1.0 + zv) ** (P("gamma") - 1.0) * model._baseline_spin(
        chi0, P("mu_chi"), P("sigma_chi"))
    integ = jnp.trapezoid(jnp.trapezoid(dens / div, qg, axis=-1), m1g)
    assert abs(float(integ) - 1.0) < 2e-2, float(integ)


@_NEED_TINYGP
@pytest.mark.parametrize("name", ["gp1d_q", "gp3d_q_chi_z"])
def test_gradient_finite_including_taper_toe(name):
    """NUTS safety: jax.grad of the summed log_p_pop w.r.t. theta stays finite,
    including a query pinned inside the low-mass taper toe (m_min + 0.5 dm_min)
    and queries with exactly-zero density above/below the support."""
    model = build_gp_model(name)
    theta = _seed_theta(model, seed=0, scale=0.6)
    P = _params(model, theta)
    m_min = float(P("m_min"))
    dm_min = float(P("dm_min"))

    m1 = jnp.asarray([m_min + 0.5 * dm_min, 8.3, 21.7, 55.1, 1.2, 500.0])
    q = jnp.asarray([0.80, 0.50, 0.90, 0.30, 0.70, 0.50])
    z = jnp.asarray([0.30, 1.00, 4.50, 0.05, 0.30, 2.00])
    chi = jnp.asarray([0.00, 0.20, -0.30, 0.50, 0.00, 0.10])

    def total(th):
        lp = model.log_p_pop(m1, q, z, chi, th)
        lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
        return jax.scipy.special.logsumexp(lp)

    val = float(total(theta))
    grad = np.asarray(jax.grad(total)(theta))
    assert np.isfinite(val)
    assert np.all(np.isfinite(grad)), dict(enumerate(grad.tolist()))
