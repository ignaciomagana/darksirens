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
    assert set(SKY_MODEL_NAMES) == {"isotropic", "dipole", "sphere_gp"}
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
    log_g = sky_model_parser("isotropic")(nx, ny, nz, jnp.array([]))
    np.testing.assert_allclose(np.asarray(log_g), 0.0, atol=0.0)


def test_dipole_matches_closed_form_and_positivity():
    log_g_sky = sky_model_parser("dipole")
    nx, ny, nz = _random_unit_vectors(256, seed=1)
    d = jnp.array([0.3, -0.2, 0.1])
    g = 1.0 + nx * d[0] + ny * d[1] + nz * d[2]
    np.testing.assert_allclose(
        np.asarray(log_g_sky(nx, ny, nz, d)), np.asarray(jnp.log(g)), rtol=1e-6
    )
    # antipodal point of a unit-amplitude dipole has g = 0 -> -inf
    dunit = jnp.array([1.0, 0.0, 0.0])
    log_g_anti = log_g_sky(jnp.array([-1.0]), jnp.array([0.0]), jnp.array([0.0]), dunit)
    assert not np.isfinite(np.asarray(log_g_anti)[0])


def test_dipole_is_mean_one_over_sphere():
    # Monte-Carlo average of g over the sphere is 1 (a pure dipole integrates
    # to zero); use many uniform directions.
    log_g_sky = sky_model_parser("dipole")
    nx, ny, nz = _random_unit_vectors(200_000, seed=2)
    d = jnp.array([0.4, 0.0, 0.0])
    g = jnp.exp(log_g_sky(nx, ny, nz, d))
    assert abs(float(jnp.mean(g)) - 1.0) < 5e-3


def test_dipole_fiducial_is_isotropic():
    nx, ny, nz = _random_unit_vectors(32)
    theta = get_fixed_sky_params("dipole")
    np.testing.assert_allclose(np.asarray(theta), 0.0)
    log_g = sky_model_parser("dipole")(nx, ny, nz, theta)
    np.testing.assert_allclose(np.asarray(log_g), 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# sphere GP (needs tinygp)
# --------------------------------------------------------------------------

def test_sphere_gp_fiducial_is_isotropic():
    nx, ny, nz = _random_unit_vectors(48, seed=3)
    theta = get_fixed_sky_params("sphere_gp")  # xi = 0 -> f = 0 -> g = 1
    log_g = sky_model_parser("sphere_gp")(nx, ny, nz, theta)
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
    log_g = model.log_g_sky(Zq[:, 0], Zq[:, 1], Zq[:, 2], theta)
    # by construction g is normalised by its mean over exactly these points
    assert abs(float(jnp.mean(jnp.exp(log_g))) - 1.0) < 1e-6


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
