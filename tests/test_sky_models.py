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
    cosmo, survey, pop_params, sky_params = decoder.decode(coord)
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
    from darksirens.sky.analyze import summarize_multipole_posterior

    model = get_sky_model("multipole")
    labels = [s.label for s in model.param_specs]          # 8 coeffs (ℓ=1,2)
    a = np.arange(1, 9, dtype=float) * 0.05                 # distinct a_lm
    samples = np.tile(a, (4, 1))
    summ = summarize_multipole_posterior(samples, labels, "multipole")
    # C_1 = sum of first 3 squared; C_2 = sum of next 5 squared
    c1 = float(np.sum(a[:3] ** 2)); c2 = float(np.sum(a[3:] ** 2))
    assert abs(summ["C_ell_quantiles"][1][0.5] - c1) < 1e-9
    assert abs(summ["C_ell_quantiles"][2][0.5] - c2) < 1e-9
