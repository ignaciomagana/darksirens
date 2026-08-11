"""
test_sky_models.py
------------------
Unit tests for the angular (sky-distribution) model layer:

* the registry contract (names, lookup, prior parser, fiducials),
* the ``log_g_sky`` math for each model (isotropy, dipole, sphere GP),
* the mean-one normalisation convention, and
* the parameter-space / decoder wiring (sky block appended, ``xi`` latents
  declared standard-normal, ``decode`` returns ``sky_params``).

The isotropic *no-op* property (an isotropic run reproduces the legacy,
sky-free likelihood bit-for-bit) is exercised by the existing likelihood /
sampler tests, which all default to ``sky_model="isotropic"``.
"""

import types

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from darksirens.sky import (
    SKY_MODEL_NAMES,
    get_sky_model,
    sky_model_parser,
    sky_model_prior_parser,
    get_fixed_sky_params,
)


def _random_unit_vectors(n, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return jnp.asarray(v[:, 0]), jnp.asarray(v[:, 1]), jnp.asarray(v[:, 2])


# --------------------------------------------------------------------------
# Registry contract
# --------------------------------------------------------------------------

def test_registry_names_and_unknown():
    assert set(SKY_MODEL_NAMES) == {
        "isotropic", "dipole", "sphere_gp", "sphere_gp_z", "overdensity_gp",
        "multipole", "multipole_l3",
    }
    for name in SKY_MODEL_NAMES:
        assert get_sky_model(name) is not None
    # cached: same object
    assert get_sky_model("dipole") is get_sky_model("dipole")
    with pytest.raises(ValueError):
        get_sky_model("does_not_exist")


def test_param_counts_and_kinds():
    # isotropic: no free parameters
    lows, highs, labels, kinds, _latex = sky_model_prior_parser("isotropic")
    assert labels == [] and lows == [] and highs == [] and kinds == []

    # dipole: three uniform parameters
    lows, highs, labels, kinds, _latex = sky_model_prior_parser("dipole")
    assert len(labels) == len(lows) == len(highs) == len(kinds) == 3
    assert all(k[0] == "uniform" for k in kinds)

    # sphere_gp: log_amp + log_ls (uniform) followed by M whitened normal latents
    model = get_sky_model("sphere_gp")
    M = model._M
    lows, highs, labels, kinds, _latex = sky_model_prior_parser("sphere_gp")
    assert len(labels) == 2 + M
    assert kinds[0][0] == "uniform" and kinds[1][0] == "uniform"
    assert all(k == ("normal", 0.0, 1.0) for k in kinds[2:])
    # the whitened latents carry ASCII names xi_* for the sampler channel
    names = [s.name for s in model.param_specs]
    assert sum(n.startswith("sky_xi_") for n in names) == M


# --------------------------------------------------------------------------
# log_g_sky math
# --------------------------------------------------------------------------

def test_isotropic_is_zero():
    nx, ny, nz = _random_unit_vectors(64)
    z = jnp.zeros_like(nx)
    log_g = sky_model_parser("isotropic")(nx, ny, nz, z, jnp.array([]))
    np.testing.assert_allclose(np.asarray(log_g), 0.0, atol=0.0)


def test_dipole_matches_closed_form_and_positivity():
    log_g_sky = sky_model_parser("dipole")
    nx, ny, nz = _random_unit_vectors(256, seed=1)
    z = jnp.zeros_like(nx)
    d = jnp.array([0.3, -0.2, 0.1])
    g = 1.0 + nx * d[0] + ny * d[1] + nz * d[2]
    np.testing.assert_allclose(
        np.asarray(log_g_sky(nx, ny, nz, z, d)), np.asarray(jnp.log(g)), rtol=1e-6
    )
    # antipodal point of a unit-amplitude dipole has g = 0 -> -inf
    dunit = jnp.array([1.0, 0.0, 0.0])
    log_g_anti = log_g_sky(jnp.array([-1.0]), jnp.array([0.0]), jnp.array([0.0]),
                           jnp.array([0.0]), dunit)
    assert not np.isfinite(np.asarray(log_g_anti)[0])


def test_dipole_is_mean_one_over_sphere():
    # Monte-Carlo average of g over the sphere is 1 (a pure dipole integrates
    # to zero); use many uniform directions.
    log_g_sky = sky_model_parser("dipole")
    nx, ny, nz = _random_unit_vectors(200_000, seed=2)
    d = jnp.array([0.4, 0.0, 0.0])
    g = jnp.exp(log_g_sky(nx, ny, nz, jnp.zeros_like(nx), d))
    assert abs(float(jnp.mean(g)) - 1.0) < 5e-3


def test_dipole_fiducial_is_isotropic():
    nx, ny, nz = _random_unit_vectors(32)
    theta = get_fixed_sky_params("dipole")
    np.testing.assert_allclose(np.asarray(theta), 0.0)
    log_g = sky_model_parser("dipole")(nx, ny, nz, jnp.zeros_like(nx), theta)
    np.testing.assert_allclose(np.asarray(log_g), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# Angular sphere GP
# --------------------------------------------------------------------------

def test_sphere_gp_fiducial_is_isotropic():
    nx, ny, nz = _random_unit_vectors(48, seed=3)
    theta = get_fixed_sky_params("sphere_gp")  # xi = 0 -> f = 0 -> g = 1
    log_g = sky_model_parser("sphere_gp")(nx, ny, nz, jnp.zeros_like(nx), theta)
    np.testing.assert_allclose(np.asarray(log_g), 0.0, atol=1e-6)


def test_sphere_gp_is_mean_one_on_quadrature():
    model = get_sky_model("sphere_gp")
    M = model._M
    rng = np.random.default_rng(4)
    # mid hyperparameters + random whitened latents
    specs = model.param_specs
    log_amp = 0.5 * (specs[0].low + specs[0].high)
    log_ls = 0.5 * (specs[1].low + specs[1].high)
    xi = rng.normal(size=M)
    theta = jnp.asarray(np.concatenate([[log_amp, log_ls], xi]))
    Zq = model._Zq
    z = jnp.zeros(Zq.shape[0])
    log_g = model.log_g_sky(Zq[:, 0], Zq[:, 1], Zq[:, 2], z, theta)
    # by construction g is normalised by its mean over exactly these points
    assert abs(float(jnp.mean(jnp.exp(log_g))) - 1.0) < 1e-6


# --------------------------------------------------------------------------
# 3-D (sphere × z) GP models: sphere_gp_z (per-shell) & overdensity_gp (global)
# --------------------------------------------------------------------------

def _gp3d_random_theta(model, seed):
    """Mid hyperparameters + random whitened latents for a (sphere × z) model."""
    rng = np.random.default_rng(seed)
    specs = model.param_specs
    hyper = [0.5 * (s.low + s.high) for s in specs[:3]]   # log_amp, log_ls_sphere, log_ls_z
    xi = rng.normal(size=model._M)
    return jnp.asarray(np.concatenate([hyper, xi]))


@pytest.mark.parametrize("name", ["sphere_gp_z", "overdensity_gp"])
def test_gp3d_param_contract(name):
    model = get_sky_model(name)
    M = model._M
    assert M == model._M_sph * model._M_z
    lows, highs, labels, kinds, _latex = sky_model_prior_parser(name)
    assert len(labels) == len(lows) == len(highs) == len(kinds) == 3 + M
    assert all(k[0] == "uniform" for k in kinds[:3])          # log_amp, ls_sphere, ls_z
    assert all(k == ("normal", 0.0, 1.0) for k in kinds[3:])  # whitened latents
    names = [s.name for s in model.param_specs]
    assert {"sky_log_amp", "sky_log_ls_sphere", "sky_log_ls_z"} <= set(names)
    assert sum(n.startswith("sky_xi_") for n in names) == M


@pytest.mark.parametrize("name", ["sphere_gp_z", "overdensity_gp"])
def test_gp3d_fiducial_is_isotropic_at_all_z(name):
    # xi = 0 -> f = 0 -> g = 1 for ANY redshift (incl. NaN passes through).
    nx, ny, nz = _random_unit_vectors(40, seed=5)
    theta = get_fixed_sky_params(name)
    log_g_sky = sky_model_parser(name)
    for z_val in (0.0, 0.3, 1.0, 2.5):
        z = jnp.full(nx.shape[0], z_val)
        log_g = log_g_sky(nx, ny, nz, z, theta)
        np.testing.assert_allclose(np.asarray(log_g), 0.0, atol=1e-6)
    # NaN z -> non-finite log_g (dropped downstream), never a bogus finite value
    z_nan = jnp.array([float("nan")] * nx.shape[0])
    log_g_nan = np.asarray(log_g_sky(nx, ny, nz, z_nan, theta))
    assert not np.any(np.isfinite(log_g_nan))


def test_sphere_gp_z_is_mean_one_per_shell():
    # Per-z-shell normalisation: at a z that is a normalisation-grid node, the
    # sphere mean of g is exactly 1 (interpolation returns the node value).
    model = get_sky_model("sphere_gp_z")
    theta = _gp3d_random_theta(model, seed=6)
    Zq = model._Zq
    for k in (1, len(model._zg) // 2, len(model._zg) - 2):
        z0 = float(model._zg[k])
        z = jnp.full(Zq.shape[0], z0)
        g = jnp.exp(model.log_g_sky(Zq[:, 0], Zq[:, 1], Zq[:, 2], z, theta))
        assert abs(float(jnp.mean(g)) - 1.0) < 1e-5


def test_overdensity_gp_is_mean_one_over_volume():
    # Global normalisation: the volume-weighted (sphere × z-grid) mean of g is 1.
    model = get_sky_model("overdensity_gp")
    theta = _gp3d_random_theta(model, seed=7)
    Zq = model._Zq
    w = np.exp(np.asarray(model._log_vol_w))            # (Nzg,) normalised z weights
    shell_means = []
    for k in range(len(model._zg)):
        z = jnp.full(Zq.shape[0], float(model._zg[k]))
        g = jnp.exp(model.log_g_sky(Zq[:, 0], Zq[:, 1], Zq[:, 2], z, theta))
        shell_means.append(float(jnp.mean(g)))
    vol_mean = float(np.sum(w * np.array(shell_means)))
    assert abs(vol_mean - 1.0) < 1e-4


# --------------------------------------------------------------------------
# Parameter-space / decoder wiring
# --------------------------------------------------------------------------

def test_build_parameter_space_appends_sky_block():
    from darksirens.inference.prior import build_parameter_space

    base = build_parameter_space("powerlaw+peak", False, False, False, sky_model="isotropic")
    labels_iso, sky_labels_iso = base[0], base[12]
    assert sky_labels_iso == []

    dip = build_parameter_space("powerlaw+peak", False, False, False, sky_model="dipole")
    assert len(dip[12]) == 3
    assert len(dip[0]) == len(labels_iso) + 3
    # the appended sky labels are last and uniform
    sky_kinds = dip[11][-3:]
    assert all(k[0] == "uniform" for k in sky_kinds)

    gp = build_parameter_space("powerlaw+peak", False, False, False, sky_model="sphere_gp")
    M = get_sky_model("sphere_gp")._M
    assert len(gp[12]) == 2 + M
    # whitened latents declared standard-normal to the sampler
    assert all(k == ("normal", 0.0, 1.0) for k in gp[11][-M:])


def _fake_opts(**overrides):
    base = dict(
        pop_model="powerlaw+peak",
        fix_population=False,
        fix_cosmology=False,
        fix_survey=False,
        fix_de=False,
        prior_overrides=None,
        universe_model="spectral_sirens",
        shared_beta=True,
        shared_spin=True,
        shared_gamma=True,
        sky_model="isotropic",
        complete_empty_pixel_policy="zero",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize(
    "sky_model,n_sky",
    [("isotropic", 0), ("dipole", 3)],
)
def test_decoder_returns_sky_params(sky_model, n_sky):
    from darksirens.inference.parameters import build_parameter_decoder
    from darksirens.gw.populations import get_fixed_population_params

    opts = _fake_opts(sky_model=sky_model)
    pop_fid = get_fixed_population_params("powerlaw+peak")
    decoder = build_parameter_decoder(opts, pop_fid)
    assert len(decoder.sky_labels) == n_sky

    coord = jnp.zeros(len(decoder.sampled_labels))
    cosmo, survey, pop_params, sky_params, _mark_params = decoder.decode(coord)
    assert int(sky_params.shape[0]) == n_sky


def test_summarize_dipole_posterior_recovers_direction():
    from darksirens.sky.analyze import summarize_dipole_posterior

    labels = ["H0", "$d_x$", "$d_y$", "$d_z$"]
    n = 2000
    samples = np.zeros((n, 4))
    samples[:, 0] = 70.0
    samples[:, 1] = 0.3  # d_x -> |d|=0.3, pointing at (ra, dec) = (0, 0)
    summ = summarize_dipole_posterior(samples, labels)
    assert abs(summ["amplitude_quantiles"][0.5] - 0.3) < 1e-6
    assert abs(summ["mean_direction_deg"]["ra"] - 0.0) < 1e-6
    assert abs(summ["mean_direction_deg"]["dec"] - 0.0) < 1e-6
    assert summ["P_amp_lt_0.05"] == 0.0


def test_sphere_gp_posterior_map_shape():
    hp = pytest.importorskip("healpy")
    from darksirens.sky.analyze import sphere_gp_posterior_map

    model = get_sky_model("sphere_gp")
    # saved run labels are the LaTeX spec labels, in param order
    labels = [s.label for s in model.param_specs]
    # fiducial (isotropic) sky parameters -> g ≡ 1 map
    theta = np.asarray(get_fixed_sky_params("sphere_gp"))
    samples = np.tile(theta, (5, 1))
    nside = 8
    m = sphere_gp_posterior_map(samples, labels, nside=nside, max_draws=5)
    assert m.shape[0] == hp.nside2npix(nside)
    np.testing.assert_allclose(m, 1.0, atol=1e-5)  # isotropic params -> g = 1


def test_gp3d_posterior_map_z_slice_shape():
    hp = pytest.importorskip("healpy")
    from darksirens.sky.analyze import sphere_gp_posterior_map

    model = get_sky_model("sphere_gp_z")
    labels = [s.label for s in model.param_specs]
    theta = np.asarray(get_fixed_sky_params("sphere_gp_z"))
    samples = np.tile(theta, (3, 1))
    nside = 8
    m = sphere_gp_posterior_map(samples, labels, nside=nside, max_draws=3,
                                sky_model="sphere_gp_z", z_slice=0.5)
    assert m.shape[0] == hp.nside2npix(nside)
    np.testing.assert_allclose(m, 1.0, atol=1e-5)  # fiducial -> g = 1 at any z


@pytest.mark.parametrize("universe_model", ["spectral_sirens", "dark_sirens", "dark_sirens_complete"])
def test_sky_block_present_for_all_universe_models(universe_model):
    """The sky model is orthogonal to the redshift-prior regime: the dipole
    block is added regardless of universe_model."""
    from darksirens.inference.prior import build_parameter_space

    res = build_parameter_space(
        "powerlaw+peak", False, False, False,
        universe_model=universe_model, sky_model="dipole",
    )
    assert len(res[12]) == 3


# --------------------------------------------------------------------------
# Low-ℓ spherical-harmonic multipole model
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name,lmax", [("multipole", 2), ("multipole_l3", 3)])
def test_multipole_param_contract(name, lmax):
    n_coeff = (lmax + 1) ** 2 - 1
    lows, highs, labels, kinds, _latex = sky_model_prior_parser(name)
    assert len(labels) == len(lows) == len(highs) == len(kinds) == n_coeff
    assert all(k[0] == "uniform" for k in kinds)
    names = [s.name for s in get_sky_model(name).param_specs]
    assert all(n.startswith("sky_a_") for n in names)


@pytest.mark.parametrize("name", ["multipole", "multipole_l3"])
def test_multipole_fiducial_is_isotropic(name):
    nx, ny, nz = _random_unit_vectors(40, seed=11)
    theta = get_fixed_sky_params(name)          # all a_lm = 0
    np.testing.assert_allclose(np.asarray(theta), 0.0)
    log_g = sky_model_parser(name)(nx, ny, nz, jnp.zeros_like(nx), theta)
    np.testing.assert_allclose(np.asarray(log_g), 0.0, atol=1e-12)


def test_multipole_is_mean_one_over_sphere():
    # Small random a_lm -> g>0 everywhere; sphere-mean of g is 1 (ℓ≥1 → 0).
    log_g_sky = sky_model_parser("multipole")
    nx, ny, nz = _random_unit_vectors(200_000, seed=12)
    rng = np.random.default_rng(12)
    a = jnp.asarray(rng.uniform(-0.1, 0.1, size=8))
    g = jnp.exp(log_g_sky(nx, ny, nz, jnp.zeros_like(nx), a))
    assert abs(float(jnp.mean(g)) - 1.0) < 5e-3


def test_real_ylm_orthonormality():
    from darksirens.sky.models import _real_ylm

    nx, ny, nz = _random_unit_vectors(500_000, seed=13)
    Y = np.asarray(_real_ylm(nx, ny, nz, 3))     # (N, 15)
    gram = 4.0 * np.pi * (Y.T @ Y) / Y.shape[0]  # ∫ Y_i Y_j dΩ ≈ δ_ij
    np.testing.assert_allclose(gram, np.eye(Y.shape[1]), atol=2e-2)


def test_multipole_l1_equals_dipole():
    from darksirens.sky.models import MultipoleSky

    nx, ny, nz = _random_unit_vectors(256, seed=14)
    z = jnp.zeros_like(nx)
    d = np.array([0.3, -0.2, 0.1])               # (d_x, d_y, d_z)
    # a_1m = sqrt(4π/3) * d_component, ordered (m=-1→y, m=0→z, m=+1→x)
    c = np.sqrt(4.0 * np.pi / 3.0)
    a = jnp.asarray([c * d[1], c * d[2], c * d[0]])
    log_g_mp = MultipoleSky(lmax=1).log_g_sky(nx, ny, nz, z, a)
    log_g_dip = sky_model_parser("dipole")(nx, ny, nz, z, jnp.asarray(d))
    np.testing.assert_allclose(np.asarray(log_g_mp), np.asarray(log_g_dip), rtol=1e-6)


def test_summarize_multipole_cl_matches_definition():
    """C_ℓ is the power PER MODE: Σ_m a_lm² / (2ℓ+1).

    The summary reported the TOTAL multipole power Σ_m a_lm² and labelled it
    C_ℓ, which is not the conventional normalisation the docstring, the plot
    axis and the docs all claim — and which is not comparable across ℓ (the
    total grows with the 2ℓ+1 mode count even for a scale-invariant field).
    """
    from darksirens.sky.analyze import summarize_multipole_posterior

    model = get_sky_model("multipole")
    labels = [s.label for s in model.param_specs]          # 8 coeffs (ℓ=1,2)
    a = np.arange(1, 9, dtype=float) * 0.05                 # distinct a_lm
    samples = np.tile(a, (4, 1))
    summ = summarize_multipole_posterior(samples, labels, "multipole")
    # C_1 = sum of first 3 squared / 3; C_2 = sum of next 5 squared / 5
    c1 = float(np.sum(a[:3] ** 2)) / 3.0
    c2 = float(np.sum(a[3:] ** 2)) / 5.0
    assert abs(summ["C_ell_quantiles"][1][0.5] - c1) < 1e-9
    assert abs(summ["C_ell_quantiles"][2][0.5] - c2) < 1e-9
    assert np.allclose(summ["C_ell_samples"][1], c1)


def test_summarize_multipole_cl_of_a_pure_dipole():
    """A pure dipole with known a_1m gives C_1 = Σ_m a_1m² / 3.

    Cross-check against the physical amplitude: for g = 1 + n̂·d the
    orthonormal-harmonic coefficients are a_1m = sqrt(4π/3) d_m, so
    C_1 = (4π/3)|d|²/3 — the mean-square anisotropy per mode.
    """
    from darksirens.sky.analyze import summarize_multipole_posterior

    model = get_sky_model("multipole")
    labels = [s.label for s in model.param_specs]
    d = np.array([0.3, -0.2, 0.1])
    c = np.sqrt(4.0 * np.pi / 3.0)
    a1 = np.array([c * d[1], c * d[2], c * d[0]])       # (m=-1, 0, +1)
    samples = np.zeros((3, len(labels)))
    samples[:, :3] = a1

    summ = summarize_multipole_posterior(samples, labels, "multipole")
    np.testing.assert_allclose(summ["C_ell_quantiles"][1][0.5],
                               float(np.sum(a1 ** 2)) / 3.0, rtol=1e-12)
    np.testing.assert_allclose(summ["C_ell_quantiles"][1][0.5],
                               (4.0 * np.pi / 3.0) * float(d @ d) / 3.0, rtol=1e-12)
    assert summ["C_ell_quantiles"][2][0.5] == 0.0


# ============================================================================
# Mean-one measured OFF the normaliser's own nodes
# ============================================================================
#
# test_sphere_gp_is_mean_one_on_quadrature and
# test_sphere_gp_z_is_mean_one_per_shell average g over exactly the quadrature
# directions (and, for the 3-D model, exactly the z-grid nodes) that _log_norm
# divides by, so the result is 1 by construction -- their own comments say so.
# They cannot see either approximation the normaliser actually makes: the finite
# Fibonacci sphere quadrature, and the jnp.interp in z between the z-nodes.
#
# These tests use an INDEPENDENT sphere sample and z values strictly BETWEEN the
# z-nodes, which is what the models' defining contract (<g> = 1 at every z)
# actually claims.

def _independent_sphere(n=8192):
    """Fibonacci sphere offset from the model's own quadrature lattice."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)], -1
    )


def _mid_hyper_theta(model, seed):
    """Prior-midpoint hyperparameters + random whitened latents."""
    specs = model.param_specs
    rng = np.random.default_rng(seed)
    th = rng.normal(size=len(specs))
    for i in range(3):
        th[i] = 0.5 * (specs[i].low + specs[i].high)
    return jnp.asarray(th)


def test_sphere_gp_z_mean_one_between_z_nodes_and_off_quadrature():
    """The real contract: <g> = 1 at every z, not just at the normaliser's nodes.

    Regression for the frozen-normaliser bug: _ZNORM_HI was hardcoded to 3.0
    while the analysis grid runs to zMax (default 5.0), and jnp.interp CLAMPS,
    so z > 3 silently reused the z=3 normaliser. Measured deviation there was up
    to 0.60 before the fix and <0.01 after. The on-node tests above are blind to
    it because they only probe z-grid nodes, all of which sat below 3.
    """
    import darksirens.sky.models as sky_models

    model = get_sky_model("sphere_gp_z")
    S = jnp.asarray(_independent_sphere())
    theta = _mid_hyper_theta(model, seed=11)

    zg = np.asarray(model._zg)
    # Strictly BETWEEN nodes, and spanning the full analysis range.
    z_probe = list(0.5 * (zg[:-1] + zg[1:])[::4]) + [3.5, 4.0, 4.5, 4.9]
    worst = 0.0
    for z0 in z_probe:
        zz = jnp.full(S.shape[0], float(z0))
        g = jnp.exp(model.log_g_sky(S[:, 0], S[:, 1], S[:, 2], zz, theta))
        worst = max(worst, abs(float(jnp.mean(g)) - 1.0))
    assert worst < 0.02, f"sphere mean of g deviates from 1 by {worst:.4f}"


def test_sphere_gp_z_mean_one_at_the_z_length_scale_prior_floor():
    """<g> = 1 between nodes must hold at the SHORTEST z length scale the prior
    allows, not only at mid-prior hyperparameters.

    Regression for the coarse-normaliser bug: the normaliser was tabulated on 40
    nodes uniform in physical z (dz = 0.128) while the field's coordinate is
    zeta = log1p(z) and log_ls_z reaches down to log(0.05) -- i.e. genuine
    structure ~2.5x finer than the node spacing.  Measured mean-one violation
    BETWEEN nodes was 4.8% at mid hyperparameters and 116% at the prior corner
    (shell mean ranging over [0.71, 2.16]); the on-node tests above are blind to
    it because linear interpolation is exact there.
    """
    model = get_sky_model("sphere_gp_z")
    specs = model.param_specs
    S = jnp.asarray(_independent_sphere())
    zeta_g = np.asarray(model._zeta_g)
    # Strictly BETWEEN nodes, in the coordinate the nodes are uniform in.
    z_probe = np.expm1(0.5 * (zeta_g[:-1] + zeta_g[1:]))[::7]

    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        th = rng.normal(size=len(specs))
        th[0] = specs[0].high      # max amplitude
        th[1] = specs[1].low       # min sphere length scale
        th[2] = specs[2].low       # min z length scale  <- the unpinned corner
        theta = jnp.asarray(th)
        worst = 0.0
        for z0 in z_probe:
            zz = jnp.full(S.shape[0], float(z0))
            g = jnp.exp(model.log_g_sky(S[:, 0], S[:, 1], S[:, 2], zz, theta))
            worst = max(worst, abs(float(jnp.mean(g)) - 1.0))
        assert worst < 0.10, (
            f"per-shell mean-one violated by {worst:.4f} at the prior corner "
            f"(seed {seed}); the z-normalisation grid under-resolves ls_z"
        )


def test_sky_z_normalisation_grid_resolves_the_z_kernel():
    """The normalisation nodes are uniform in zeta = log1p(z) (the coordinate the
    z-kernel acts on) and spaced at most a third of the shortest prior length
    scale, so the interpolated normaliser can follow the field."""
    model = get_sky_model("sphere_gp_z")
    zeta_g = np.asarray(model._zeta_g)
    d_zeta = np.diff(zeta_g)
    np.testing.assert_allclose(d_zeta, d_zeta[0], rtol=1e-9)
    ls_z_min = float(np.exp(model.param_specs[2].low))
    assert d_zeta[0] <= ls_z_min / 3.0
    np.testing.assert_allclose(np.asarray(model._zg), np.expm1(zeta_g), rtol=1e-12)


def test_sky_z_normalisation_grid_covers_the_analysis_grid():
    """The normaliser must not freeze below the redshifts the pipeline samples.

    likelihood/core.py evaluates log_g_sky at z_of_dL over the full distance
    range, and z_of_dL grows with H0 across its prior, so any ceiling below zMax
    is reachable in an ordinary run.
    """
    import darksirens.sky.models as sky_models
    from darksirens.redshift.grid import zMax

    assert sky_models._ZNORM_HI >= float(zMax), (
        f"sky z-normalisation ceiling {sky_models._ZNORM_HI} is below the "
        f"analysis grid zMax={zMax}; g would be mis-normalised above it"
    )
    # Node density is held fixed as the ceiling moves (8 per unit z).
    assert sky_models._ZNORM_N >= 8 * sky_models._ZNORM_HI - 1


def test_sphere_gp_mean_one_off_quadrature():
    """2-D model: independent sphere sample, prior-midpoint hyperparameters."""
    model = get_sky_model("sphere_gp")
    S = jnp.asarray(_independent_sphere())
    specs = model.param_specs
    rng = np.random.default_rng(12)
    th = rng.normal(size=len(specs))
    th[0] = 0.5 * (specs[0].low + specs[0].high)
    th[1] = 0.5 * (specs[1].low + specs[1].high)
    theta = jnp.asarray(th)
    z = jnp.zeros(S.shape[0])
    g = jnp.exp(model.log_g_sky(S[:, 0], S[:, 1], S[:, 2], z, theta))
    assert abs(float(jnp.mean(g)) - 1.0) < 0.02


def test_sphere_quadrature_resolves_the_prior_corner():
    """Mean-one must hold across the WHOLE sampled prior, not just its middle.

    A 192-node Fibonacci sphere under-resolves a high-amplitude,
    short-length-scale field: at log_amp = max, log_ls_sphere = min the sphere
    average of g was off by 25% -- inside the prior the sampler explores.
    Convergence is non-monotonic (lattice aliasing), so the node count is chosen
    to be reliably converged rather than merely better.
    """
    import darksirens.sky.models as sky_models

    model = get_sky_model("sphere_gp_z")
    specs = model.param_specs
    rng = np.random.default_rng(0)
    th = rng.normal(size=len(specs))
    th[0] = specs[0].high     # max amplitude
    th[1] = specs[1].low      # min sphere length scale
    th[2] = 0.0
    theta = jnp.asarray(th)

    S = jnp.asarray(_independent_sphere())
    worst = 0.0
    for z0 in np.linspace(0.3, 4.9, 9):
        zz = jnp.full(S.shape[0], float(z0))
        g = jnp.exp(model.log_g_sky(S[:, 0], S[:, 1], S[:, 2], zz, theta))
        worst = max(worst, abs(float(jnp.mean(g)) - 1.0))
    assert worst < 0.02, (
        f"mean-one violated by {worst:.4f} at the prior corner; the sphere "
        f"quadrature ({sky_models._SPHERE_NQ} nodes) under-resolves the field"
    )


# --------------------------------------------------------------------------
# Global positivity (review finding P1-18)
# --------------------------------------------------------------------------

def test_dipole_rejects_super_unity_amplitude_globally():
    """d = (1, 1, 1) has |d| = sqrt(3) > 1: ~21% of the sphere would go
    negative, and the old pointwise clip turned the model into a positive-
    part field with spherical mean ~1.077 while still claiming mean one.
    The whole parameter point is now invalid."""
    log_g_sky = sky_model_parser("dipole")
    nx, ny, nz = _random_unit_vectors(512, seed=21)
    d = jnp.array([1.0, 1.0, 1.0])
    out = np.asarray(log_g_sky(nx, ny, nz, jnp.zeros_like(nx), d))
    assert np.all(np.isneginf(out)), (
        "super-unity dipole must be globally invalid, not pointwise-clipped"
    )
    # Just inside the ball: valid everywhere, mean one.
    d_in = d / jnp.sqrt(3.0) * 0.99
    g = jnp.exp(log_g_sky(*_random_unit_vectors(200_000, seed=22),
                          jnp.zeros(200_000), d_in))
    assert np.all(np.isfinite(np.asarray(g)))
    assert abs(float(jnp.mean(g)) - 1.0) < 5e-3


def test_multipole_rejects_globally_negative_fields():
    """A coefficient corner whose field dips negative anywhere on the sphere
    is invalid as a whole (constrained prior), not silently replaced by its
    positive part."""
    log_g_sky = sky_model_parser("multipole")
    nx, ny, nz = _random_unit_vectors(64, seed=23)
    z = jnp.zeros_like(nx)
    a_corner = jnp.ones(8)  # box corner: g goes deeply negative in places
    out = np.asarray(log_g_sky(nx, ny, nz, z, a_corner))
    assert np.all(np.isneginf(out))
    # Small coefficients: positive everywhere, finite everywhere, mean one.
    rng = np.random.default_rng(24)
    a_small = jnp.asarray(rng.uniform(-0.05, 0.05, size=8))
    nx2, ny2, nz2 = _random_unit_vectors(200_000, seed=25)
    g = jnp.exp(log_g_sky(nx2, ny2, nz2, jnp.zeros_like(nx2), a_small))
    assert np.all(np.isfinite(np.asarray(g)))
    assert abs(float(jnp.mean(g)) - 1.0) < 5e-3


def test_dipole_ball_cube_map_covers_the_ball_uniformly():
    """The nested samplers reach the dipole through the ball3 cube map: every
    draw lands inside the unit ball (zero wasted proposals) with the uniform-
    ball radial law E[r] = 3/4."""
    from darksirens.inference.prior import (
        build_parameter_space,
        make_prior_transform,
        resolve_joint_prior_constraints,
    )

    res = build_parameter_space(
        "powerlaw+peak", True, True, True, fix_de=True,
        universe_model="spectral_sirens", sky_model="dipole",
    )
    labels, lower, upper, kinds = list(res[0]), res[1], res[2], res[11]
    assert labels == ["$d_x$", "$d_y$", "$d_z$"]
    jc = resolve_joint_prior_constraints(
        "powerlaw+peak", labels, lower, upper, kinds, sky_model="dipole"
    )
    assert jc == [("ball3", (0, 1, 2))]
    transform = make_prior_transform(lower, upper, kinds, joint_constraints=jc)

    u = jnp.asarray(np.random.default_rng(26).uniform(size=(5000, 3)))
    d = np.asarray(jnp.stack([transform(u[k]) for k in range(5000)]))
    r = np.linalg.norm(d, axis=1)
    assert np.all(r <= 1.0 + 1e-12)
    np.testing.assert_allclose(np.mean(r), 0.75, atol=0.01)
    # Isotropy of the direction: component means vanish.
    np.testing.assert_allclose(np.mean(d, axis=0), 0.0, atol=0.02)


# ============================================================================
# Registry construction is trace-safe (review finding F-079)
# ============================================================================

@pytest.mark.parametrize("name", [n for n in SKY_MODEL_NAMES if n != "isotropic"])
def test_registry_construction_inside_a_trace_is_safe(name):
    """The likelihood resolves the sky model INSIDE its jit body, so the first
    use of a model can build its geometry while a trace is active.

    Before the fix, ``sphere_gp`` cached TRACERS in the process-global registry
    (the first call returned a correct number, the next re-trace died with an
    UnexpectedTracerError) and ``sphere_gp_z`` / ``overdensity_gp`` /
    ``multipole*`` failed on a concretization inside their constructors.
    Production was protected only by the prior decoder happening to warm the
    registry first.
    """
    import jax

    from darksirens.sky import registry as sky_registry

    cached = sky_registry._SKY_REGISTRY.pop(name, None)
    try:
        @jax.jit
        def total(x):
            model = sky_registry.get_sky_model(name)
            log_g = sky_registry.sky_model_parser(name)(
                x, jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x),
                get_fixed_sky_params(name),
            )
            return jnp.sum(log_g) + jnp.sum(x) + float(len(model.param_specs))

        first = float(total(jnp.ones(3)))
        # Nothing cached may be a tracer, or it escapes into every later trace.
        for value in vars(sky_registry._SKY_REGISTRY[name]).values():
            assert not isinstance(value, jax.core.Tracer)
        # A re-trace (new batch shape) is where an escaped tracer would surface.
        second = float(total(jnp.ones(4)))
        assert np.isfinite(first) and np.isfinite(second)
    finally:
        if cached is not None:
            sky_registry._SKY_REGISTRY[name] = cached
