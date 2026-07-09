"""Tests for the optional per-survey ``z_depth`` completion budget bound.

Covers (see the PR-2 plan):
  * ``darksirens.redshift.completion._assemble_curves``: ``z_depth=None`` is
    bit-identical to the pre-existing full-grid missing-galaxy budget, and a
    finite ``z_depth`` truncates ``N_miss``/``dN_miss`` to ``zgrid <= z_depth``.
  * ``darksirens.catalogs.io.load_survey``: optional ``f.attrs['z_depth']``
    round-trip.
  * ``darksirens.inference.parameters.build_parameter_decoder``: a resolved
    per-catalog ``z_depth`` lands on ``SurveyParams.z_depth``.

Follows the tiny in-memory fixture style of
``tests/test_complete_catalog_empty_pixel_policy.py`` and the x64 convention
in ``tests/conftest.py`` (jax x64 is enabled there before any darksirens
import; re-asserted here defensively as the other completion tests do).
"""
import warnings

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import h5py
import pytest

from darksirens.redshift import zgrid
from darksirens.redshift.completion import completion_curves
from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.catalogs.io import load_survey


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(z_depth=None):
    return SurveyParams(
        n0=1e-2,
        z50=1.0,
        w=0.5,
        delta=0.0,
        b_miss=1.0,
        alpha_miss=0.5,
        z_depth=z_depth,
    )


def _catalog():
    """A 2-row catalog using the on-the-fly KDE fallback (dN_obs_kde=None),
    modeled on tests/test_complete_catalog_empty_pixel_policy.py::_catalog."""
    return EMCatalog(
        apix=1.0,
        zgals=jnp.array([
            [0.0, 0.0],
            [0.2, 0.35],
        ]),
        dzgals=jnp.array([
            [1.0, 1.0],
            [0.02, 0.03],
        ]),
        wgals=jnp.array([
            [0.0, 0.0],
            [1.0, 0.5],
        ]),
        ngals=jnp.array([0, 2], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((2, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )


# ---------------------------------------------------------------------------
# completion.py: z_depth=None bit-identity and truncation behaviour
# ---------------------------------------------------------------------------

def test_z_depth_none_bit_identical():
    """z_depth=None must equal z_depth=zgrid[-1] EXACTLY (==, not isclose):
    at full depth the mask covers the whole grid, so it must be numerically
    inert -- proving the None branch is not merely "close" to the masked
    computation but takes the same values.

    The None branch additionally cannot even CONSTRUCT the depth mask: were
    it to attempt ``zgrid <= survey.z_depth`` with ``z_depth=None`` it would
    raise (comparing a JAX array with NoneType is a TypeError), so simply
    completing without error already exercises "no mask built for None".
    """
    cosmo = _cosmo()
    catalog = _catalog()

    curves_none = completion_curves(cosmo, _survey(z_depth=None), catalog)
    curves_full = completion_curves(cosmo, _survey(z_depth=float(zgrid[-1])), catalog)

    np.testing.assert_array_equal(
        np.asarray(curves_none.N_miss), np.asarray(curves_full.N_miss)
    )
    np.testing.assert_array_equal(
        np.asarray(curves_none.dN_miss), np.asarray(curves_full.dN_miss)
    )
    np.testing.assert_array_equal(
        np.asarray(curves_none.C_eff), np.asarray(curves_full.C_eff)
    )
    np.testing.assert_array_equal(
        np.asarray(curves_none.f), np.asarray(curves_full.f)
    )


def test_z_depth_truncates_missing_budget():
    """A finite z_depth < zMax truncates N_miss to an independent numpy
    trapezoid of dN_miss * (zgrid <= z_depth), and N_miss is monotone
    non-decreasing as z_depth grows (dN_miss >= 0 everywhere: C in [0, 1] and
    the LSS rate factor is clipped to be non-negative)."""
    cosmo = _cosmo()
    catalog = _catalog()
    zgrid_np = np.asarray(zgrid)

    # The FULL (unmasked) dN_miss, established bit-identical to z_depth=None
    # by test_z_depth_none_bit_identical, is the reference for the
    # independent numpy quadrature below.
    curves_full = completion_curves(cosmo, _survey(z_depth=None), catalog)
    dN_miss_full = np.asarray(curves_full.dN_miss)  # (N_rows, N_grid)
    assert np.all(dN_miss_full >= 0.0)

    z_depth = float(zgrid_np[len(zgrid_np) // 3])
    assert z_depth < zgrid_np[-1]

    curves_bounded = completion_curves(cosmo, _survey(z_depth=z_depth), catalog)
    N_miss_bounded = np.asarray(curves_bounded.N_miss)

    mask = zgrid_np <= z_depth
    expected = np.trapezoid(dN_miss_full * mask[None, :], zgrid_np, axis=-1)
    np.testing.assert_allclose(N_miss_bounded, expected, rtol=1e-12, atol=1e-300)

    # dN_miss itself must be exactly zeroed beyond z_depth.
    dN_miss_bounded = np.asarray(curves_bounded.dN_miss)
    np.testing.assert_array_equal(dN_miss_bounded[:, ~mask], 0.0)

    # Monotone non-decreasing in z_depth.
    depths = [
        float(zgrid_np[len(zgrid_np) // 10]),
        float(zgrid_np[len(zgrid_np) // 3]),
        float(zgrid_np[2 * len(zgrid_np) // 3]),
        float(zgrid_np[-1]),
    ]
    n_miss_row0 = []
    for d in depths:
        c = completion_curves(cosmo, _survey(z_depth=d), catalog)
        n_miss_row0.append(float(np.asarray(c.N_miss)[1]))  # row 1 has real galaxies
    assert all(a <= b + 1e-12 for a, b in zip(n_miss_row0, n_miss_row0[1:]))


def test_z_depth_clamped_above_zmax_warns():
    """z_depth > zMax warns and, once clamped by the CLI resolver, behaves as
    the full grid (the completion internals apply whatever z_depth they are
    given; clamping to zMax is the CLI's --survey_z_depth / file-attr
    resolution contract -- darksirens.cli.inference.resolve_survey_z_depth)."""
    from darksirens.cli.inference import resolve_survey_z_depth

    zmax = float(zgrid[-1])
    with pytest.warns(UserWarning, match="exceeds the redshift grid zMax"):
        resolved = resolve_survey_z_depth(zmax + 5.0, None, zmax=zmax)
    assert resolved == zmax

    # Clamped resolution behaves exactly like the (unclamped) full-grid
    # z_depth=zgrid[-1] case, which in turn is bit-identical to z_depth=None
    # (test_z_depth_none_bit_identical).
    cosmo = _cosmo()
    catalog = _catalog()
    curves_clamped = completion_curves(cosmo, _survey(z_depth=resolved), catalog)
    curves_full = completion_curves(cosmo, _survey(z_depth=zmax), catalog)
    np.testing.assert_array_equal(
        np.asarray(curves_clamped.N_miss), np.asarray(curves_full.N_miss)
    )

    # A z_depth within range is returned untouched, with no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved_ok = resolve_survey_z_depth(0.5 * zmax, None, zmax=zmax)
    assert resolved_ok == 0.5 * zmax


# ---------------------------------------------------------------------------
# catalogs/io.py: load_survey optional z_depth attribute
# ---------------------------------------------------------------------------

def _write_minimal_survey(path, z_depth=None):
    npix, maxgals = 2, 1
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 1
        f.create_dataset("zgals", data=np.zeros((npix, maxgals)))
        f.create_dataset("dzgals", data=np.ones((npix, maxgals)))
        f.create_dataset("wgals", data=np.zeros((npix, maxgals)))
        f.create_dataset("ngals", data=np.zeros(npix, dtype=np.int32))
        if z_depth is not None:
            f.attrs["z_depth"] = float(z_depth)


def test_load_survey_reads_z_depth(tmp_path):
    path_with = tmp_path / "survey_with_depth.h5"
    path_without = tmp_path / "survey_without_depth.h5"
    _write_minimal_survey(path_with, z_depth=1.25)
    _write_minimal_survey(path_without, z_depth=None)

    *_ignored, z_depth_with = load_survey(str(path_with))
    *_ignored, z_depth_without = load_survey(str(path_without))

    assert z_depth_with == pytest.approx(1.25)
    assert z_depth_without is None


# ---------------------------------------------------------------------------
# inference/parameters.py: resolved z_depth threads to SurveyParams
# ---------------------------------------------------------------------------

def _fixed_pop_fixture():
    """Fully block-fixed population fiducials for 'powerlaw+peak' (mirrors
    test_multitracer_likelihood.py::_pop_bits, minus the one sampled label)."""
    from darksirens.gw.populations import get_fixed_population_params
    return get_fixed_population_params("powerlaw+peak")


def _base_decoder_opts(**overrides):
    from types import SimpleNamespace
    kwargs = dict(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=True,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def test_survey_params_z_depth_threading():
    from darksirens.inference.parameters import build_parameter_decoder

    opts = _base_decoder_opts(resolved_survey_z_depths=[1.5])
    pop_params_fid = _fixed_pop_fixture()

    decoder = build_parameter_decoder(opts, pop_params_fid, fixed_parameter_values={})
    assert decoder.z_depths == (1.5,)

    coord = jnp.zeros((0,))
    _cosmo_out, survey, *_rest = decoder.decode(coord)
    assert survey.z_depth == 1.5


def test_survey_params_z_depth_defaults_to_none_when_unresolved():
    """A bare/legacy ``opts`` that never sets ``resolved_survey_z_depths``
    (e.g. existing tests/callers) must decode to ``z_depth=None`` -- the
    legacy full-grid budget, bit-identical to pre-existing behaviour."""
    from darksirens.inference.parameters import build_parameter_decoder

    opts = _base_decoder_opts()
    pop_params_fid = _fixed_pop_fixture()

    decoder = build_parameter_decoder(opts, pop_params_fid, fixed_parameter_values={})
    assert decoder.z_depths == ()

    coord = jnp.zeros((0,))
    _cosmo_out, survey, *_rest = decoder.decode(coord)
    assert survey.z_depth is None


def test_z_depth_rejects_nonfinite_and_nonpositive():
    """NaN passes every comparison-based guard (NaN > zmax is False) and would
    silently zero the whole missing-galaxy budget (depth_mask all-False);
    zero/negative likewise degenerate to 'fully complete'. The resolver must
    reject them loudly instead (adversarial-review SEV-2 for PR #204)."""
    from darksirens.cli.inference import resolve_survey_z_depth

    for bad in (float("nan"), float("inf"), float("-inf"), 0.0, -0.5):
        with pytest.raises(ValueError, match="finite positive redshift"):
            resolve_survey_z_depth(bad, None)
    # File-attr path is guarded identically.
    with pytest.raises(ValueError, match="finite positive redshift"):
        resolve_survey_z_depth(None, float("nan"))
