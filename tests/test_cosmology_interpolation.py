import jax
jax.config.update("jax_enable_x64", True)

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import jax.numpy as jnp
import numpy as np

from darksirens.utils.cosmology import dL_of_z, dV_of_z, r_of_z, z_of_dL


def test_cpl_fiducial_reproduces_flat_lambdacdm_distances_and_volume():
    H0 = 67.74
    Om0 = 0.3075
    z = jnp.array([0.0, 0.03, 0.1, 0.5, 1.0, 2.0])
    lcdm = FlatLambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=Om0)

    np.testing.assert_allclose(
        np.asarray(r_of_z(z, H0, Om0, w0=-1.0, wa=0.0)),
        lcdm.comoving_distance(np.asarray(z)).to_value(u.Mpc),
        rtol=5e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(dL_of_z(z, H0, Om0, w0=-1.0, wa=0.0)),
        lcdm.luminosity_distance(np.asarray(z)).to_value(u.Mpc),
        rtol=5e-5,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        np.asarray(dV_of_z(z, H0, Om0, w0=-1.0, wa=0.0)),
        lcdm.differential_comoving_volume(np.asarray(z)).to_value(u.Mpc**3 / u.sr),
        rtol=5e-5,
        atol=1e-8,
    )


def test_cpl_fiducial_inverse_distance_matches_lambdacdm_redshift():
    H0 = 67.74
    Om0 = 0.3075
    z = jnp.array([0.02, 0.1, 0.5, 1.0, 2.0])
    dL = dL_of_z(z, H0, Om0, w0=-1.0, wa=0.0)

    actual = z_of_dL(dL, H0, Om0, w0=-1.0, wa=0.0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(z), rtol=2e-4, atol=2e-6)


# ============================================================================
# Separable interpolation path (interpnd_scalar_head)
# ============================================================================
#
# `r_of_z` contracts the three scalar cosmology axes into a 1-D r(z) curve
# before the redshift lookup, instead of gathering 16 corners out of the full
# (Om0, w0, wa, z) table. Multilinear interpolation is a tensor product, so this
# is the same number up to floating-point reassociation — these tests pin that
# equivalence, including the NaN/out-of-range masking and the gradients (the
# masking exists to stop a NaN poisoning the backward pass, so it has to be
# checked on the backward pass too, not just the forward one).

import pytest

from darksirens.utils import cosmology as _cosmo
from darksirens.utils.interp2d import interpnd, interpnd_scalar_head

_GRIDS = (_cosmo.Om0grid, _cosmo.w0grid, _cosmo.wagrid, _cosmo.zgrid)


def _r_of_z_interpnd(z, H0, Om0, w0, wa):
    """The pre-optimisation general path, kept here as the reference."""
    return interpnd(
        (Om0, w0, wa, z), _GRIDS, _cosmo.rs, fill_value=jnp.nan
    ) * (_cosmo.H0Planck / H0)


def test_separable_matches_general_path_over_random_cosmologies():
    rng = np.random.default_rng(20260725)
    worst = 0.0
    for _ in range(40):
        Om0 = float(rng.uniform(0.16, 0.45))
        w0 = float(rng.uniform(-2.2, 0.2))
        wa = float(rng.uniform(-2.4, 2.4))
        H0 = float(rng.uniform(50.0, 110.0))
        z = jnp.asarray(rng.uniform(1e-3, 4.9, 4000))
        want = np.asarray(_r_of_z_interpnd(z, H0, Om0, w0, wa))
        got = np.asarray(r_of_z(z, H0, Om0, w0, wa))
        worst = max(worst, float(np.max(np.abs(got / want - 1.0))))
    assert worst < 1e-13, f"separable path drifted from interpnd: {worst:.2e}"


@pytest.mark.parametrize("Om0,w0,wa,label", [
    (0.05, -1.0, 0.0, "Om0 below grid"),
    (0.90, -1.0, 0.0, "Om0 above grid"),
    (0.31, -9.0, 0.0, "w0 below grid"),
    (0.31, 5.0, 0.0, "w0 above grid"),
    (0.31, -1.0, -9.0, "wa below grid"),
    (0.31, -1.0, 9.0, "wa above grid"),
    (0.31, -1.0, 0.0, "all in range"),
])
def test_separable_nan_masking_matches_per_axis(Om0, w0, wa, label):
    """Out-of-range on ANY axis -> NaN, identically on both paths."""
    z = jnp.array([0.5, -1.0, 99.0, jnp.nan])
    want = np.asarray(_r_of_z_interpnd(z, 70.0, Om0, w0, wa))
    got = np.asarray(r_of_z(z, 70.0, Om0, w0, wa))
    np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=label)
    finite = ~np.isnan(want)
    if finite.any():
        np.testing.assert_allclose(got[finite], want[finite], rtol=1e-13)


def test_separable_gradients_match_general_path():
    """Gradients w.r.t. all five arguments, not just the forward value."""
    z = jnp.asarray(np.linspace(0.05, 4.0, 128))
    f = lambda H0, Om0, w0, wa: jnp.sum(r_of_z(z, H0, Om0, w0, wa))
    g = lambda H0, Om0, w0, wa: jnp.sum(_r_of_z_interpnd(z, H0, Om0, w0, wa))
    got = jax.grad(f, argnums=(0, 1, 2, 3))(70.0, 0.31, -0.95, 0.1)
    want = jax.grad(g, argnums=(0, 1, 2, 3))(70.0, 0.31, -0.95, 0.1)
    for a, b in zip(got, want):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-11)


def test_separable_gradient_is_finite_for_out_of_range_queries():
    """The NaN-sanitisation contract: a masked query must not poison the grad."""
    z = jnp.array([0.5, 99.0, jnp.nan, 1.0])
    f = lambda Om0: jnp.sum(jnp.nan_to_num(r_of_z(z, 70.0, Om0, -1.0, 0.0)))
    assert np.isfinite(float(jax.grad(f)(0.31)))


def test_vmap_over_cosmology_samples_keeps_the_fast_path():
    """The inference pattern: one cosmology per posterior sample."""
    z = jnp.asarray(np.linspace(0.01, 4.5, 500))
    rng = np.random.default_rng(7)
    n = 64
    H0 = jnp.asarray(rng.uniform(60.0, 90.0, n))
    Om0 = jnp.asarray(rng.uniform(0.22, 0.40, n))
    w0 = jnp.asarray(rng.uniform(-1.8, -0.4, n))
    wa = jnp.asarray(rng.uniform(-1.5, 1.5, n))
    got = jax.vmap(r_of_z, in_axes=(None, 0, 0, 0, 0))(z, H0, Om0, w0, wa)
    want = jax.vmap(_r_of_z_interpnd, in_axes=(None, 0, 0, 0, 0))(z, H0, Om0, w0, wa)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13)


def test_array_valued_cosmology_falls_back_to_interpnd():
    """A caller that maps over Om0 directly still gets correct answers."""
    Om0 = jnp.asarray([0.28, 0.33, 0.36])
    z = jnp.asarray([1.0, 2.0, 3.0])
    got = np.asarray(r_of_z(z, 70.0, Om0, -1.0, 0.0))
    want = np.asarray(_r_of_z_interpnd(z, 70.0, Om0, -1.0, 0.0))
    np.testing.assert_allclose(got, want, rtol=1e-14)
    assert np.all(np.isfinite(got))


def test_interpnd_scalar_head_rejects_non_scalar_leading_axes():
    with pytest.raises(ValueError, match="scalar leading coordinates"):
        interpnd_scalar_head(
            (jnp.asarray([0.3, 0.31]), -1.0, 0.0, jnp.asarray([1.0, 2.0])),
            _GRIDS, _cosmo.rs,
        )


def test_interpnd_scalar_head_generalises_beyond_the_cosmology_table():
    """Same contract on a small hand-built table, any number of leading axes."""
    ax0 = jnp.asarray([0.0, 1.0, 2.0])
    ax1 = jnp.asarray([0.0, 0.5, 1.0, 1.5])
    tail = jnp.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(3)
    values = jnp.asarray(rng.normal(size=(ax0.size, ax1.size, tail.size)))
    q = jnp.asarray([0.25, 1.75, 3.5])
    got = interpnd_scalar_head((0.7, 1.1, q), (ax0, ax1, tail), values, fill_value=jnp.nan)
    want = interpnd((0.7, 1.1, q), (ax0, ax1, tail), values, fill_value=jnp.nan)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-13)


# ============================================================================
# Interpolation-grid node allocation
# ============================================================================
#
# Node counts are chosen from the MEASURED midpoint error per axis, not by
# habit. Multilinear interpolation is exact at nodes and worst at cell
# midpoints, so these tests probe midpoints — the on-node tests above would
# pass no matter how coarse the Om0/w0/wa axes were.

from astropy.cosmology import Flatw0waCDM

_Z_PROBE = np.array([0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.5])


def _dl_rel_error(Om0, w0, wa):
    ref = Flatw0waCDM(
        H0=_cosmo.H0Planck, Om0=Om0, w0=w0, wa=wa
    ).luminosity_distance(_Z_PROBE).value
    got = np.asarray(dL_of_z(jnp.asarray(_Z_PROBE), _cosmo.H0Planck, Om0, w0, wa))
    return float(np.max(np.abs(got / ref - 1.0)))


def _midpoint(grid, i):
    g = np.asarray(grid)
    return 0.5 * (g[i] + g[i + 1])


def test_fiducial_cosmology_sits_exactly_on_grid_nodes():
    """Why re-noding cannot perturb an H0-only run.

    With the fiducial on a node the interpolation weight is exactly 0, so the
    corner sum reduces to the tabulated column — which is built by the same
    trapezoid on the same zgrid regardless of how many Om0/w0/wa nodes there
    are. Fixed-cosmology results are therefore bitwise stable across changes
    to these node counts.
    """
    assert np.min(np.abs(np.asarray(_cosmo.Om0grid) - _cosmo.Om0Planck)) == 0.0
    assert np.min(np.abs(np.asarray(_cosmo.w0grid) - _cosmo.w0Fiducial)) == 0.0
    assert np.min(np.abs(np.asarray(_cosmo.wagrid) - _cosmo.waFiducial)) == 0.0


def test_midpoint_accuracy_budget_is_balanced_across_axes():
    """No axis may dominate the error budget, and none may be over-resolved.

    The z axis imposes a floor (linear-in-z interpolation of r(z)); an axis
    resolved far below that floor is spending table memory for nothing, which
    is what the Om0 axis was doing at 31 nodes (5.5e-5 midpoint against a
    5.4e-5 floor) while w0 sat at 5.1e-4.
    """
    on_node = _dl_rel_error(_cosmo.Om0Planck, _cosmo.w0Fiducial, _cosmo.waFiducial)
    om0_mid = _dl_rel_error(_midpoint(_cosmo.Om0grid, _cosmo.Om0grid.size // 2),
                            _cosmo.w0Fiducial, _cosmo.waFiducial)
    w0_mid = _dl_rel_error(_cosmo.Om0Planck,
                           _midpoint(_cosmo.w0grid, _cosmo.w0grid.size // 2),
                           _cosmo.waFiducial)
    wa_mid = _dl_rel_error(_cosmo.Om0Planck, _cosmo.w0Fiducial,
                           _midpoint(_cosmo.wagrid, _cosmo.wagrid.size // 2))

    assert on_node < 1e-4
    for name, err in (("Om0", om0_mid), ("w0", w0_mid), ("wa", wa_mid)):
        assert err < 2.0e-4, f"{name} midpoint error {err:.2e} exceeds the budget"
    # Balanced: no axis carries more than ~3x another. (Before re-noding the
    # w0/Om0 ratio was ~9.)
    worst, best = max(om0_mid, w0_mid, wa_mid), min(om0_mid, w0_mid, wa_mid)
    assert worst / best < 3.0, f"unbalanced budget: {om0_mid=} {w0_mid=} {wa_mid=}"


def test_worst_case_midpoint_error_over_the_physical_grid():
    """Cell midpoints across the physical region (w0 < -0.3; w0 near or above
    zero is unphysical and not worth budgeting accuracy for).

    Strided rather than exhaustive — the full midpoint set is ~17k astropy
    evaluations. The stride still visits both grid edges and the interior on
    every axis, which is where the error extremes live (widest cells and the
    strongest curvature in Om0 sit at the low-Om0/high-wa corner).
    """
    mids = lambda g: 0.5 * (np.asarray(g)[:-1] + np.asarray(g)[1:])
    worst, arg = 0.0, None
    for om in mids(_cosmo.Om0grid)[::3]:
        for w0 in mids(_cosmo.w0grid)[::3]:
            if w0 > -0.3:
                continue
            for wa in mids(_cosmo.wagrid)[::3]:
                err = _dl_rel_error(om, w0, wa)
                if err > worst:
                    worst, arg = err, (om, w0, wa)
    assert worst < 1.5e-3, f"worst-case midpoint error {worst:.2e} at {arg}"
