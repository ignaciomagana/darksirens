"""Constraint-preserving prior transforms for jointly-constrained models
(review findings P1-15 and F-115).

The GWTC-5 fiducial BPL+2-peaks model enforces ``lambda0 + lambda1 <= 1``
and ``m2_low <= m1_low`` by likelihood-side rejection.  Each condition kills
half its square, so 75% of the nominal unit cube had zero likelihood: nested
proposals wasted 3 of 4 draws and the evidence carried a constant
``log(1/4) = -1.386`` offset relative to the intended normalized constrained
prior.  The model now declares its ``constraint_groups`` and
``make_prior_transform`` maps the cube ONTO the constrained region; the
likelihood-side ``valid`` mask remains as a backstop (numpyro, overridden
bounds).

The low-mass pair uses ``conditional_upper``, not ``ordered_le`` (F-115).
Both live on the ordered triangle ``{3 <= m2_low <= m1_low <= 10}``, so the
support tests below cannot tell them apart -- but a sort gives the UNIFORM
density there, while GWTC-5 Table 5 quotes the conditional
``m1_low ~ U(3, 10)``, ``m2_low ~ U(3, m1_low)``.  They differ by the factor
``1/(m1_low - 3)``: 2x the intended density at m1_low = 10 and 0.14x at 3.5,
a tilt that pushes the minimum BH mass up and with it the low-mass end of the
selection function.  So the density tests here are the ones with teeth.
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
    assert kinds_found == {"simplex", "conditional_upper"}
    by_kind = dict(jc)
    i, j = by_kind["simplex"]
    assert labels[i] == L0 and labels[j] == L1
    # Order matters for conditional_upper: i is the CONDITIONED parameter
    # (m2_low ~ U(3, m1_low)), j the one it is conditioned on.  Swapping them
    # would silently sample m1_low ~ U(3, m2_low) instead.
    i, j = by_kind["conditional_upper"]
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


def test_transform_reproduces_the_declared_prior_moments(gwtc5_space):
    """Each map must give the NORMALIZED density the model declares.

    Simplex: uniform on the triangle, E[lam] = 1/3 each.  Low-mass pair:
    Table 5's conditional, so m1_low keeps its FLAT U(a, b) marginal,
    E[m1_low] = a + (b-a)/2, and E[m2_low] = a + E[m1_low - a]/2 =
    a + (b-a)/4.  The uniform-triangle (``ordered_le``) answers would be
    a + 2(b-a)/3 and a + (b-a)/3 -- the numbers this test used to pin, and
    the reason it is worth pinning at all."""
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
        np.mean(theta[:, i1]), a + (b - a) / 2.0, atol=0.02 * (b - a))
    np.testing.assert_allclose(
        np.mean(theta[:, i2]), a + (b - a) / 4.0, atol=0.02 * (b - a))


def test_conditional_upper_is_exactly_the_1_over_m1_minus_3_reweighting(
    gwtc5_space
):
    """The whole content of F-115: conditional / ordered-triangle =
    (b - a) / (2 (m1_low - a)).

    Both maps push the SAME cube through the SAME affine bounds, so the ratio
    of their densities is the ratio of their Jacobians at matched points.
    Checked here on the marginal of the conditioning parameter, where the
    factor is visible in closed form: the sort gives p(m1) ~ (m1 - a) and the
    conditional gives p(m1) = const, so binning both and dividing must return
    (b - a) / (2 (m1 - a)) bin by bin.
    """
    labels, lower, upper, kinds = gwtc5_space
    i1 = labels.index(M1LOW)
    a, b = float(lower[i1]), float(upper[i1])

    cond = make_prior_transform(
        lower, upper, kinds,
        joint_constraints=[("conditional_upper", (labels.index(M2LOW), i1))],
    )
    sort = make_prior_transform(
        lower, upper, kinds,
        joint_constraints=[("ordered_le", (labels.index(M2LOW), i1))],
    )

    n = 200000
    u = _rng_cube(n, len(labels), seed=3)
    m1_cond = np.asarray(jnp.stack([cond(u[k]) for k in range(n)]))[:, i1]
    m1_sort = np.asarray(jnp.stack([sort(u[k]) for k in range(n)]))[:, i1]

    edges = np.linspace(a, b, 8)
    centres = 0.5 * (edges[:-1] + edges[1:])
    p_cond, _ = np.histogram(m1_cond, bins=edges, density=True)
    p_sort, _ = np.histogram(m1_sort, bins=edges, density=True)

    expected = (b - a) / (2.0 * (centres - a))
    np.testing.assert_allclose(p_cond / p_sort, expected, rtol=0.06)

    # The two numbers F-115 quotes, read off the same closed form the other
    # way round (sort density / Table 5 density = 2 (m1 - a) / (b - a)):
    # 2x at the top of the range, 0.14x at m1_low = 3.5.
    assert 2.0 * (b - a) / (b - a) == pytest.approx(2.0)
    assert 2.0 * (3.5 - a) / (b - a) == pytest.approx(0.1429, abs=1e-4)


def test_conditional_upper_marginal_is_the_flat_table5_prior(gwtc5_space):
    """m1_low ~ U(3, 10) exactly: the sort's ``p(m1_low) ~ (m1_low - 3)``
    tilt (which pushes the minimum BH mass up) must be gone."""
    labels, lower, upper, kinds = gwtc5_space
    jc = resolve_joint_prior_constraints(POP, labels, lower, upper, kinds)
    transform = make_prior_transform(lower, upper, kinds, joint_constraints=jc)

    i1, i2 = labels.index(M1LOW), labels.index(M2LOW)
    a, b = float(lower[i1]), float(upper[i1])

    n = 100000
    u = _rng_cube(n, len(labels), seed=4)
    theta = np.asarray(jnp.stack([transform(u[k]) for k in range(n)]))
    m1 = theta[:, i1]
    m2 = theta[:, i2]

    # Flat marginal: equal-width bins hold equal mass.
    counts, _ = np.histogram(m1, bins=np.linspace(a, b, 8))
    np.testing.assert_allclose(counts / counts.mean(), 1.0, atol=0.03)

    # Conditional: (m2 - 3) / (m1 - 3) ~ U(0, 1) independent of m1, so its
    # mean is 1/2 in every m1 slice.  The uniform triangle would instead give
    # a 1/2 that is only correct on AVERAGE, with the ratio's distribution
    # unchanged -- so also check the slice-wise flatness that separates them
    # from the m1 marginal above.
    ratio = (m2 - a) / (m1 - a)
    np.testing.assert_allclose(np.mean(ratio), 0.5, atol=0.01)
    np.testing.assert_allclose(np.std(ratio), 1.0 / np.sqrt(12.0), atol=0.01)


def test_conditional_upper_is_finite_at_the_degenerate_edge(gwtc5_space):
    """m1_low -> 3 is where the conditional density 1/(m1_low - 3) diverges.

    The map is written multiplicatively (u_i * u_j) precisely so that edge
    costs nothing: at u_j = 0 both parameters land on the shared lower bound
    instead of evaluating 0/0.  A NaN here would not be stopped by the
    likelihood's ``valid`` mask -- NaN fails every comparison, so the point is
    rejected but the poisoned value has already entered the arithmetic -- and
    would take out the whole nested iteration.
    """
    labels, lower, upper, kinds = gwtc5_space
    jc = resolve_joint_prior_constraints(POP, labels, lower, upper, kinds)
    transform = make_prior_transform(lower, upper, kinds, joint_constraints=jc)
    i1, i2 = labels.index(M1LOW), labels.index(M2LOW)
    a = float(lower[i1])

    for u_j in (0.0, 1e-300, 1e-12):
        for u_i in (0.0, 0.5, 1.0):
            u = np.full(len(labels), 0.5)
            u[i1] = u_j
            u[i2] = u_i
            theta = np.asarray(transform(jnp.asarray(u)))
            assert np.all(np.isfinite(theta)), (u_i, u_j)
            assert theta[i2] <= theta[i1] + 1e-12
            assert theta[i1] >= a and theta[i2] >= a
        # u_j = 0 is the exact corner: both collapse onto the lower bound.
        if u_j == 0.0:
            u = np.full(len(labels), 0.5)
            u[i1] = 0.0
            u[i2] = 1.0
            theta = np.asarray(transform(jnp.asarray(u)))
            assert theta[i1] == pytest.approx(a)
            assert theta[i2] == pytest.approx(a)


def test_conditional_upper_transform_is_differentiable_at_the_edge(gwtc5_space):
    """The nested transform is called inside jitted/differentiated code paths;
    a NaN gradient at the u_j -> 0 corner is as fatal as a NaN value."""
    import jax

    labels, lower, upper, kinds = gwtc5_space
    jc = resolve_joint_prior_constraints(POP, labels, lower, upper, kinds)
    transform = make_prior_transform(lower, upper, kinds, joint_constraints=jc)
    i1, i2 = labels.index(M1LOW), labels.index(M2LOW)

    def scalar(u):
        return jnp.sum(transform(u))

    u = jnp.asarray(np.full(len(labels), 0.5)).at[i1].set(0.0).at[i2].set(1.0)
    g = np.asarray(jax.grad(scalar)(u))
    assert np.all(np.isfinite(g))


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
    assert {k for k, _ in jc} == {"conditional_upper"}


def test_overridden_bounds_fall_back_with_a_warning(gwtc5_space):
    labels, lower, upper, kinds = gwtc5_space
    lower2, upper2 = lower.copy(), upper.copy()
    upper2[labels.index(L1)] = 0.5           # simplex needs exact [0, 1]
    # conditional_upper needs identical bounds too: the SHARED lower edge is
    # the conditional's floor, so a raised m2_low floor would sample a
    # different conditional than U(3, m1_low).
    lower2[labels.index(M2LOW)] = 4.0
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


def test_numpyro_reports_that_it_leaves_joint_constraints_to_rejection(capsys):
    """The numpyro model builds INDEPENDENT per-parameter sites, so the joint
    constraints the nested transform reparameterizes away are left to
    likelihood-side rejection there: same truncated measure, but NUTS integrates
    against a gradient-free -inf wall.  run_sampler must say so (the `_site`
    docstring used to claim measure parity with make_prior_transform)."""
    pytest.importorskip("numpyro")
    from types import SimpleNamespace

    from darksirens.inference.sampling import run_sampler

    labels = ["dx", "dy", "dz"]
    lower = np.full(3, -1.0)
    upper = np.full(3, 1.0)

    def loglike(theta):
        # Unit-ball indicator x a smooth core: the rejection wall NUTS sees.
        r2 = jnp.sum(jnp.asarray(theta) ** 2)
        return jnp.where(r2 <= 1.0, -0.5 * r2, -jnp.inf)

    opts = SimpleNamespace(
        seed=5, show_progress=False, nuts_warmup=2, nuts_samples=2,
        nuts_chains=1, nuts_target_accept=0.8, nuts_max_tree_depth=3,
        nuts_chain_method="sequential",
    )
    run_sampler("numpyro", loglike, make_prior_transform(lower, upper), labels,
                lower, upper, opts, joint_constraints=[("ball3", (0, 1, 2))])
    out = capsys.readouterr().out
    assert "joint prior constraints ball3(0, 1, 2)" in out
    assert "REJECTION" in out
