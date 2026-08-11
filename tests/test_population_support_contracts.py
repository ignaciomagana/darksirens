"""Population support contracts (review findings P1-12, P1-13, P1-14).

Three ways the population/sampling layer could quietly break the importance-
sampling contract "target support ⊆ proposal support, densities honest":

* P1-12 — the truncated-Normal DRAW collapsed to an atom at the window
  boundary once the window sat more than ~8.3 sigma from mu (both endpoint
  CDFs round to the same double), while its returned ``log_s`` still
  described a continuous distribution — an internally inconsistent
  importance estimator.  The inversion now happens in log-CDF space
  (``_ndtri_exp``: Newton on ``log_ndtr``), continuous arbitrarily far into
  the tail.
* P1-13 — the flow-path "full-support" mass grid started at m1 = 2 while
  Gaussian peaks (untruncated, normalized from the global M_LO = 1) and the
  generic pairing fallback keep finite target density on [1, 2): a region
  the proposal could never draw.
* P1-14 — the additive GP returned a finite density (log of the _LOGSAFE
  floor, ~ -690) below the hard mass taper where the joint and binned GP
  models correctly return -inf, giving zero-support points nonzero density.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
import jax.numpy as jnp

scipy_stats = pytest.importorskip("scipy.stats")

from darksirens.gw.populations.sampling import (
    truncnorm_logpdf,
    truncnorm_sample,
)


# ---------------------------------------------------------------------------
# P1-12: tail-stable truncated-Normal draws
# ---------------------------------------------------------------------------

UGRID = jnp.asarray(np.linspace(0.001, 0.999, 41))

WINDOWS = [
    ("central", 0.0, 1.0, -1.0, 1.0),
    ("review-repro-10-sigma", 0.0, 0.1, 1.0, 1.1),
    ("forty-sigma", 0.0, 0.1, 4.0, 4.1),
    ("seventy-sigma", 0.0, 0.1, 7.0, 7.1),
    ("lower-tail", 0.0, 0.1, -1.1, -1.0),
    ("mu-inside-narrow", 2.0, 0.1, -1.0, 1.0),
]


@pytest.mark.parametrize("name,mu,sig,lo,hi", WINDOWS)
def test_truncnorm_draw_matches_scipy_quantiles(name, mu, sig, lo, hi):
    out = truncnorm_sample(UGRID, jnp.asarray(mu), jnp.asarray(sig), lo, hi)
    x = np.asarray(out.x)
    ref = scipy_stats.truncnorm.ppf(
        np.asarray(UGRID), (lo - mu) / sig, (hi - mu) / sig, loc=mu, scale=sig
    )
    # The draw must be a real spread of quantiles, not an atom at a boundary.
    assert np.std(x) > 0.0, f"{name}: draws collapsed to a single value"
    # Absolute quantile agreement in units of the WINDOW width: far tails are
    # ill-conditioned in sigma units for any double-precision method (scipy
    # included), but the distribution over the window must match.
    width = hi - lo
    assert np.max(np.abs(x - ref)) / width < 5e-2, name
    # First and second moments against scipy's analytic values.
    m_ref, v_ref = scipy_stats.truncnorm.stats(
        (lo - mu) / sig, (hi - mu) / sig, loc=mu, scale=sig, moments="mv"
    )
    assert abs(np.mean(x) - float(m_ref)) < 0.05 * width, name


@pytest.mark.parametrize("name,mu,sig,lo,hi", WINDOWS)
def test_truncnorm_log_s_is_the_exact_density_at_the_draw(name, mu, sig, lo, hi):
    """The internal-consistency contract the old draw violated: log_s must be
    truncnorm_logpdf evaluated at x, exactly."""
    out = truncnorm_sample(UGRID, jnp.asarray(mu), jnp.asarray(sig), lo, hi)
    lp = truncnorm_logpdf(out.x, jnp.asarray(mu), jnp.asarray(sig), lo, hi)
    np.testing.assert_allclose(np.asarray(out.log_s), np.asarray(lp),
                               rtol=1e-12, atol=1e-12)


def test_truncnorm_endpoints_map_to_bounds():
    u = jnp.asarray([0.0, 1.0])
    out = truncnorm_sample(u, jnp.asarray(0.0), jnp.asarray(0.1), 1.0, 1.1)
    np.testing.assert_allclose(np.asarray(out.x), [1.0, 1.1], atol=1e-9)


def test_truncnorm_tail_draw_is_differentiable():
    """The sampler runs inside the jitted, differentiated likelihood body; the
    Newton refinement must carry finite gradients in the far tail too."""
    def f(mu):
        return jnp.sum(truncnorm_sample(
            jnp.asarray([0.25, 0.75]), mu, jnp.asarray(0.1), 1.0, 1.1
        ).x)

    g = jax.grad(f)(jnp.asarray(0.0))
    assert np.isfinite(float(g))


# ---------------------------------------------------------------------------
# P1-13: proposal grid covers the target's support
# ---------------------------------------------------------------------------

def test_mass_grid_lower_bound_reaches_global_normalization_floor():
    from darksirens.gw.populations.registry import get_model
    from darksirens.gw.populations.sampling import resolve_mass_grid_bounds
    from darksirens.gw.populations.utils import M_LO

    model = get_model("powerlaw+peak")
    m1_lo, m1_hi = resolve_mass_grid_bounds(model)
    assert m1_lo <= float(M_LO), (
        f"proposal grid starts at {m1_lo} but untruncated components are "
        f"normalized from M_LO = {M_LO}; target support below the grid "
        "violates the importance-sampling containment requirement"
    )
    assert m1_hi >= 200.0


# ---------------------------------------------------------------------------
# The mixture's low-mass edge is a MODEL parameter, not the quadrature floor
# ---------------------------------------------------------------------------

def test_peak_component_secondary_floor_is_the_sampled_m_min():
    """A mass component with no low-mass edge (the Gaussian ``peak``) must
    inherit the MIXTURE's sampled ``m_min`` for its ``m2 = q*m1`` cut.

    It used to fall back to the normalisation-grid floor ``M_LO`` = 1, so 90% of
    the fiducial ``powerlaw+peak`` population (w_G = 0.90) admitted 1 Msun
    secondaries: measured log p = -4.31 at (m1, q) = (30, 0.05), i.e. m2 = 1.5,
    and P(m2 < m_min = 5) = 0.019 at beta = 1 rising to 0.41 at beta = -1.
    """
    from darksirens.gw.populations.registry import (
        get_fixed_population_params,
        get_model,
    )

    model = get_model("powerlaw+peak")
    theta = jnp.asarray(get_fixed_population_params("powerlaw+peak"))
    m_min = float(theta[2])                                   # PL.m_min
    m1 = jnp.asarray([30.0, 35.0, 30.0])
    q = jnp.asarray([0.05, 0.04, 0.9 * m_min / 30.0])
    out = np.asarray(model.log_p_pop(m1, q, jnp.full(3, 0.3), jnp.zeros(3), theta))
    assert np.all(np.isneginf(out)), (
        f"secondaries below m_min = {m_min} kept finite density {out}"
    )

    # No probability leaks below the edge, and (m1, q) stays normalised.
    mt = model.mixture_theta(theta)
    mg = jnp.linspace(1.0, 200.0, 1201)
    qg = jnp.linspace(1.0e-4, 1.0, 601)
    dens = model.mixture.mass_q_density(mg[:, None], qg[None, :], mt)
    total = float(jnp.trapezoid(jnp.trapezoid(dens, qg, axis=1), mg))
    below = float(jnp.trapezoid(jnp.trapezoid(
        jnp.where(mg[:, None] * qg[None, :] < m_min, dens, 0.0), qg, axis=1), mg))
    assert abs(total - 1.0) < 5.0e-3, total
    assert below == 0.0, below


# ---------------------------------------------------------------------------
# P1-14: every registered population model returns -inf outside hard support
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _evict_tinygp_stub():
    # Combined-run guard copied from test_gp_oom_and_broadcast: other files
    # install a stub tinygp into sys.modules; evict it so the real package
    # imports for the GP constructions below.
    import sys

    for key in [k for k in sys.modules if k == "tinygp" or k.startswith("tinygp.")]:
        mod = sys.modules[key]
        if getattr(mod, "__file__", None) is None:
            del sys.modules[key]
    yield


def test_gp_models_agree_on_zero_support():
    """The additive GP used to return log(_LOGSAFE) ~ -690 (finite!) where
    the joint and binned GPs return -inf: below m_min, and at m2 < m_min."""
    pytest.importorskip("tinygp")
    from darksirens.gw.populations.gp import build_gp_model

    for name in ("gp4d_additive", "gp3d_m1_q_chi"):
        m = build_gp_model(name)
        th = jnp.asarray(m.fiducial())
        # Points far below any m_min draw of the taper (m1 = 0.5) and with
        # m2 = q*m1 far below m_min (m1 fine, q tiny).
        m1 = jnp.asarray([0.5, 30.0])
        q = jnp.asarray([0.9, 0.01])
        z = jnp.asarray([0.2, 0.2])
        chi = jnp.asarray([0.0, 0.0])
        out = np.asarray(m.log_p_pop(m1, q, z, chi, th))
        assert np.all(np.isneginf(out)), (
            f"{name}: zero-support points returned finite density {out}"
        )


def test_gp_additive_in_support_unchanged():
    """The -inf fix must not perturb in-support densities."""
    pytest.importorskip("tinygp")
    from darksirens.gw.populations.gp import build_gp_model

    m = build_gp_model("gp4d_additive")
    th = jnp.asarray(m.fiducial())
    out = np.asarray(m.log_p_pop(
        jnp.asarray([20.0, 35.0]), jnp.asarray([0.8, 0.9]),
        jnp.asarray([0.2, 0.4]), jnp.asarray([0.0, 0.1]), th,
    ))
    assert np.all(np.isfinite(out))
