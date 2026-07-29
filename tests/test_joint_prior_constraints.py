"""Constraint-preserving prior transforms for jointly-constrained models
(review finding P1-15).

The GWTC-5 fiducial BPL+2-peaks model enforces ``lambda0 + lambda1 <= 1``
and ``m2_low <= m1_low`` by likelihood-side rejection.  Each condition kills
half its square, so 75% of the nominal unit cube had zero likelihood: nested
proposals wasted 3 of 4 draws and the evidence carried a constant
``log(1/4) = -1.386`` offset relative to the intended normalized constrained
prior.  The model now declares its ``constraint_groups`` and
``make_prior_transform`` maps the cube ONTO the constrained region with
measure-preserving fold/sort maps; the likelihood-side ``valid`` mask remains
as a backstop (numpyro, overridden bounds).
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from darksirens.inference.prior import (
    build_parameter_space,
    make_prior_transform,
    resolve_joint_prior_constraints,
)

POP = "gwtc5_fiducial_bpl2peaks"

L0 = r"$\lambda_0$"
L1 = r"$\lambda_1$"
M1LOW = r"$m_{1,{\rm low}}$"
M2LOW = r"$m_{2,{\rm low}}$"


def _space(**kw):
    res = build_parameter_space(
        POP, False, True, True, fix_de=True, universe_model="spectral_sirens",
        **kw,
    )
    labels = list(res[0])
    return labels, np.asarray(res[1], float), np.asarray(res[2], float), res[11]


@pytest.fixture(scope="module")
def gwtc5_space():
    return _space()


def _rng_cube(n, d, seed=0):
    return jnp.asarray(np.random.default_rng(seed).uniform(size=(n, d)))


def test_resolver_finds_both_groups(gwtc5_space):
    labels, lower, upper, kinds = gwtc5_space
    jc = resolve_joint_prior_constraints(POP, labels, lower, upper, kinds)
    assert len(jc) == 2
    kinds_found = {k for k, _ in jc}
    assert kinds_found == {"simplex", "ordered_le"}
    by_kind = dict(jc)
    i, j = by_kind["simplex"]
    assert labels[i] == L0 and labels[j] == L1
    i, j = by_kind["ordered_le"]
    assert labels[i] == M2LOW and labels[j] == M1LOW


def test_transform_maps_the_whole_cube_into_the_constrained_region(gwtc5_space):
    labels, lower, upper, kinds = gwtc5_space
    jc = resolve_joint_prior_constraints(POP, labels, lower, upper, kinds)
    transform = make_prior_transform(lower, upper, kinds, joint_constraints=jc)

    u = _rng_cube(4000, len(labels))
    theta = np.asarray(jnp.stack([transform(u[k]) for k in range(u.shape[0])]))

    lam0 = theta[:, labels.index(L0)]
    lam1 = theta[:, labels.index(L1)]
    m1l = theta[:, labels.index(M1LOW)]
    m2l = theta[:, labels.index(M2LOW)]

    assert np.all(lam0 + lam1 <= 1.0 + 1e-12), "simplex constraint violated"
    assert np.all(m2l <= m1l + 1e-12), "ordering constraint violated"
    # Every draw is now inside the constrained region: zero wasted proposals.
    assert np.all((theta >= lower - 1e-9) & (theta <= upper + 1e-9))


def test_transform_is_measure_preserving(gwtc5_space):
    """The fold/sort maps must give the NORMALIZED uniform density on the
    constrained region — checked against the analytic moments of the uniform
    triangle: E[lam] = 1/3, and for the ordered pair on [a, b],
    E[min] = a + (b-a)/3, E[max] = a + 2(b-a)/3."""
    labels, lower, upper, kinds = gwtc5_space
    jc = resolve_joint_prior_constraints(POP, labels, lower, upper, kinds)
    transform = make_prior_transform(lower, upper, kinds, joint_constraints=jc)

    n = 20000
    u = _rng_cube(n, len(labels), seed=1)
    theta = np.asarray(jnp.stack([transform(u[k]) for k in range(n)]))

    lam0 = theta[:, labels.index(L0)]
    lam1 = theta[:, labels.index(L1)]
    np.testing.assert_allclose(np.mean(lam0), 1.0 / 3.0, atol=0.01)
    np.testing.assert_allclose(np.mean(lam1), 1.0 / 3.0, atol=0.01)

    i1, i2 = labels.index(M1LOW), labels.index(M2LOW)
    a, b = lower[i1], upper[i1]
    np.testing.assert_allclose(
        np.mean(theta[:, i2]), a + (b - a) / 3.0, atol=0.02 * (b - a))
    np.testing.assert_allclose(
        np.mean(theta[:, i1]), a + 2.0 * (b - a) / 3.0, atol=0.02 * (b - a))


def test_constant_likelihood_evidence_is_unbiased(gwtc5_space):
    """The review's evidence criterion: with L = const, logZ must be
    log(const) under a PROPER prior.  Monte-Carlo logZ = log mean L(theta(u))
    over the cube; the rejection prior gave log(const) + log(1/4)."""
    labels, lower, upper, kinds = gwtc5_space
    jc = resolve_joint_prior_constraints(POP, labels, lower, upper, kinds)
    transform = make_prior_transform(lower, upper, kinds, joint_constraints=jc)

    def toy_L(theta):
        lam0 = theta[labels.index(L0)]
        lam1 = theta[labels.index(L1)]
        m1l = theta[labels.index(M1LOW)]
        m2l = theta[labels.index(M2LOW)]
        valid = (lam0 + lam1 <= 1.0) & (m2l <= m1l)
        return np.where(valid, 1.0, 0.0)

    u = _rng_cube(4000, len(labels), seed=2)
    L = np.array([float(toy_L(np.asarray(transform(u[k])))) for k in range(4000)])
    # All mass lands in the valid region: logZ = 0 exactly, not log(1/4).
    np.testing.assert_allclose(np.log(np.mean(L)), 0.0, atol=1e-12)


def test_fixed_member_falls_back_to_rejection(gwtc5_space):
    labels, lower, upper, kinds = gwtc5_space
    # Drop lambda1 as if it were fixed: the simplex group cannot resolve.
    keep = [k for k, lab in enumerate(labels) if lab != L1]
    jc = resolve_joint_prior_constraints(
        POP, [labels[k] for k in keep], lower[keep], upper[keep],
        [kinds[k] for k in keep],
    )
    assert {k for k, _ in jc} == {"ordered_le"}


def test_overridden_bounds_fall_back_with_a_warning(gwtc5_space):
    labels, lower, upper, kinds = gwtc5_space
    lower2, upper2 = lower.copy(), upper.copy()
    upper2[labels.index(L1)] = 0.5           # simplex needs exact [0, 1]
    lower2[labels.index(M2LOW)] = 4.0        # ordering needs identical bounds
    with pytest.warns(RuntimeWarning, match="falling back to rejection"):
        jc = resolve_joint_prior_constraints(POP, labels, lower2, upper2, kinds)
    assert jc == []


def test_models_without_constraints_are_untouched():
    res = build_parameter_space(
        "powerlaw+peak", False, True, True, fix_de=True,
        universe_model="spectral_sirens",
    )
    jc = resolve_joint_prior_constraints(
        "powerlaw+peak", list(res[0]), res[1], res[2], res[11]
    )
    assert jc == []
    transform = make_prior_transform(res[1], res[2], res[11], joint_constraints=jc)
    u = jnp.asarray(np.full(len(res[0]), 0.25))
    ref = make_prior_transform(res[1], res[2], res[11])
    np.testing.assert_allclose(
        np.asarray(transform(u)), np.asarray(ref(u)), rtol=0, atol=0)
