"""Tests for the per-survey ``z_depth`` completeness prior.

``z_depth`` encodes that the EM survey catalogs nothing past its depth:
completeness is exactly zero beyond it and every modeled host there is
uncatalogued (missing), NOT nonexistent.  Concretely, beyond ``z_depth`` the
missing density relaxes to the full expected count ``dN_miss = dN_exp`` (C := 0,
lss := 1) and the observed catalog kernel ``p_cat`` is zeroed (renormalised to
stay a proper density below the depth).

Covers:
  * ``darksirens.redshift.completion._assemble_curves``: ``z_depth=None`` is
    bit-identical to the pre-existing full-grid budget; a finite ``z_depth``
    relaxes ``dN_miss`` to ``dN_exp`` beyond it and integrates ``N_miss`` over
    the FULL grid.
  * ``darksirens.redshift.prior``: the assembled prior numerator above the depth
    is the volumetric x population shape ``dN_exp`` (empty pixel), and a
    cataloged galaxy whose photo-z leaks past the depth contributes nothing to
    the prior above it (observed-KDE truncation).
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

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import h5py
import pytest

from darksirens.redshift import zgrid
from darksirens.redshift.completion import completion_curves, _precompute_grids
from darksirens.redshift.prior import (
    prepare_redshift_prior_state,
    eval_redshift_prior_with_state,
)
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


def test_z_depth_relaxes_missing_budget_beyond_depth():
    """Beyond a finite z_depth < zMax the missing density relaxes to the full
    expected count (C := 0, lss := 1), so ``dN_miss == dN_exp`` there and
    ``C_eff == 0``; below the depth ``dN_miss`` follows the completion formula
    with the DEPTH-TRUNCATED denominator ``S @ (dN_exp · 1[z <= z_depth])``
    (S0a): far from the edge that matches the z_depth=None curve, while within
    a few smoothing sigmas of the edge the truncation can only REMOVE
    denominator mass, so C rises and ``dN_miss`` falls relative to the
    untruncated curve (the pre-fix behaviour was a C dip / dN_miss spike
    there).  ``N_miss`` is the FULL-grid quadrature of this relaxed density."""
    from darksirens.redshift.completion import _SIGMA_SMOOTH

    cosmo = _cosmo()
    catalog = _catalog()
    zgrid_np = np.asarray(zgrid)

    # The unbounded dN_miss (bit-identical to z_depth=None) is the below-depth
    # reference; dN_exp is the volumetric x population shape it relaxes to.
    curves_full = completion_curves(cosmo, _survey(z_depth=None), catalog)
    dN_miss_full = np.asarray(curves_full.dN_miss)  # (N_rows, N_grid)
    assert np.all(dN_miss_full >= 0.0)
    dN_exp = np.asarray(_precompute_grids(cosmo, _survey(z_depth=None), catalog).dN_exp)

    z_depth = float(zgrid_np[len(zgrid_np) // 3])
    assert z_depth < zgrid_np[-1]

    curves_bounded = completion_curves(cosmo, _survey(z_depth=z_depth), catalog)
    dN_miss_bounded = np.asarray(curves_bounded.dN_miss)
    C_eff_bounded = np.asarray(curves_bounded.C_eff)

    below = zgrid_np <= z_depth
    above = ~below

    # Below depth, FAR from the edge (> 10 sigma_smooth): the truncated
    # denominator loses only kernel mass ~exp(-0.5 (d/sigma)^2), far below the
    # f64 ulp of the retained sum, so the completion formula is unchanged.
    far_below = zgrid_np <= z_depth - 10.0 * _SIGMA_SMOOTH
    assert far_below.any()
    np.testing.assert_array_equal(
        dN_miss_bounded[:, far_below], dN_miss_full[:, far_below]
    )
    # Whole below-depth region: the truncation only REMOVES denominator mass,
    # so C >= C_untruncated and dN_miss <= the untruncated curve (up to
    # rounding) -- the depth edge can no longer SPIKE the missing density.
    np.testing.assert_array_less(
        dN_miss_bounded[:, below],
        dN_miss_full[:, below] * (1.0 + 1e-12) + 1e-300,
    )
    # Beyond depth: relaxed to dN_exp (C := 0, lss := 1), for EVERY row.
    for r in range(dN_miss_bounded.shape[0]):
        np.testing.assert_allclose(dN_miss_bounded[r, above], dN_exp[above], rtol=1e-12)
    # C_eff is exactly 0 wherever dN_miss == dN_exp (beyond the depth).
    np.testing.assert_allclose(C_eff_bounded[:, above], 0.0, atol=1e-12)

    # N_miss is the FULL-grid quadrature of the relaxed density (the relaxed
    # branch is dN_exp; the retained branch is the truncated-denominator
    # completion density asserted against dN_miss_full above).
    expected = _trapezoid(
        np.where(above[None, :], dN_exp[None, :], dN_miss_bounded), zgrid_np, axis=-1
    )
    np.testing.assert_allclose(
        np.asarray(curves_bounded.N_miss), expected, rtol=1e-12, atol=1e-300
    )

    # Monotone NON-INCREASING in z_depth: as the depth grows, more of the grid
    # switches from the full dN_exp to the (<= dN_exp) completion density, so the
    # retained-missing budget can only shrink.  (Row 1 has real galaxies; the
    # fixture's delta_g == 0 => lss == 1 <= any completeness-reduced factor.)
    depths = [
        float(zgrid_np[len(zgrid_np) // 10]),
        float(zgrid_np[len(zgrid_np) // 3]),
        float(zgrid_np[2 * len(zgrid_np) // 3]),
        float(zgrid_np[-1]),
    ]
    n_miss_row1 = [
        float(np.asarray(completion_curves(cosmo, _survey(z_depth=d), catalog).N_miss)[1])
        for d in depths
    ]
    assert all(a >= b - 1e-12 for a, b in zip(n_miss_row1, n_miss_row1[1:]))
    # At full depth the budget equals the z_depth=None (legacy) quadrature.
    np.testing.assert_allclose(
        n_miss_row1[-1], float(np.asarray(curves_full.N_miss)[1]), rtol=1e-12
    )


def test_z_depth_empty_pixel_prior_is_volumetric_beyond_depth():
    """Empty-pixel regression: with z_depth set, the assembled prior numerator
    above the depth is strictly positive and proportional to dN_exp (the
    volumetric x population shape), and below the depth it is unchanged from the
    completion formula (== dN_miss).  An empty pixel routes entirely to the
    missing branch (N_obs = 0), so the numerator IS dN_miss."""
    cosmo = _cosmo()
    catalog = _catalog()  # row 0 is empty (ngals == 0)
    zgrid_np = np.asarray(zgrid)
    z_depth = float(zgrid_np[len(zgrid_np) // 3])

    grids = _precompute_grids(cosmo, _survey(z_depth=z_depth), catalog)
    dN_exp = np.asarray(grids.dN_exp)

    state = prepare_redshift_prior_state(
        "dark_sirens", cosmo, _survey(z_depth=z_depth), catalog,
        materialize_state=False,
    )
    # Two samples straddling the depth, both in the empty pixel (row 0).
    below = zgrid_np[zgrid_np <= z_depth]
    above = zgrid_np[zgrid_np > z_depth]
    z_lo = float(below[len(below) // 2])
    z_hi = float(above[len(above) // 2])
    z = jnp.asarray([z_lo, z_hi])
    pix = jnp.asarray([0, 0], dtype=jnp.int32)

    lp = np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", state, z, pix, cosmo, _survey(z_depth=z_depth), catalog
    ))
    assert np.all(np.isfinite(lp)), lp  # strictly positive density both sides

    # The empty-pixel numerator is dN_miss / Z[pix]; above the depth dN_miss ==
    # dN_exp, so exp(lp_above) is proportional to dN_exp evaluated at the sample.
    # Compare the ratio of the two above-depth-vs-below-depth prior values with
    # the corresponding dN_miss ratio (Z[pix] cancels).
    dN_miss0 = np.asarray(state.dN_miss[0])
    p_lo = float(np.interp(z_lo, zgrid_np, dN_miss0))
    p_hi = float(np.interp(z_hi, zgrid_np, dN_miss0))
    # dN_miss above depth is exactly dN_exp.
    np.testing.assert_allclose(p_hi, float(np.interp(z_hi, zgrid_np, dN_exp)), rtol=1e-10)
    np.testing.assert_allclose(
        np.exp(lp[1] - lp[0]), p_hi / p_lo, rtol=1e-6
    )


def test_z_depth_observed_kde_contributes_nothing_beyond_depth():
    """Observed-KDE regression: a cataloged galaxy whose photo-z mass leaks past
    z_depth contributes nothing to the prior above the depth.  With N_miss
    forced to zero (a fixture with no expected missing density is impossible, so
    instead compare against a pure-missing baseline): concretely, above the
    depth the numerator equals the missing branch alone -- the observed catalog
    term p_cat is exactly zero there -- for a pixel whose single galaxy sits at
    the depth edge with a broad kernel that leaks well past it."""
    cosmo = _cosmo()
    zgrid_np = np.asarray(zgrid)
    z_depth = float(zgrid_np[len(zgrid_np) // 3])

    # One occupied pixel; its galaxy sits AT the depth with a broad photo-z so a
    # large fraction of its (untruncated) kernel mass lies beyond z_depth.
    leaky = EMCatalog(
        apix=1.0,
        zgals=jnp.array([[z_depth, 0.0]]),
        dzgals=jnp.array([[0.06, 1.0]]),
        wgals=jnp.array([[1.0, 0.0]]),
        ngals=jnp.array([1], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),
        dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    state = prepare_redshift_prior_state(
        "dark_sirens", cosmo, _survey(z_depth=z_depth), leaky,
        materialize_state=False,
    )

    above = zgrid_np[zgrid_np > z_depth]
    z_hi = float(above[len(above) // 2])
    pix = jnp.asarray([0], dtype=jnp.int32)

    # Prior above depth == missing-only branch (dN_miss / Z), i.e. the observed
    # catalog term N_obs * p_cat drops out entirely.
    lp = float(np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", state, jnp.asarray([z_hi]), pix, cosmo,
        _survey(z_depth=z_depth), leaky,
    ))[0])
    dN_miss0 = np.asarray(state.dN_miss[0])
    log_Z = float(state.log_Z[0])
    expected_missing_only = float(np.log(np.interp(z_hi, zgrid_np, dN_miss0))) - log_Z
    np.testing.assert_allclose(lp, expected_missing_only, rtol=1e-6)

    # Sanity: the galaxy's raw (untruncated) kernel genuinely leaks past the
    # depth, so the truncation is doing real work (not a no-op on this fixture).
    from darksirens.redshift.catalog import (
        catalog_kernel_state, eval_log_catalog_prior_state,
    )
    ker_untrunc = catalog_kernel_state(
        cosmo, _survey(z_depth=None), leaky, z_depth=None
    )
    p_cat_above = float(eval_log_catalog_prior_state(
        jnp.asarray(z_hi), jnp.asarray(0), ker_untrunc, leaky
    ))
    assert np.isfinite(p_cat_above)  # untruncated p_cat is nonzero above depth


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


def test_z_depth_edge_constant_completeness_no_dip_no_spike():
    """S0a depth-edge truncation: a constant-true-completeness survey with a
    concrete ``z_depth`` passes through the completeness ratio EXACTLY at the
    depth edge -- no C dip, no dN_miss spike within 2 smoothing sigmas below it.

    Mechanism: the observed-galaxy kernels renormalise on [0, zmax], so the
    numerator loses mass approaching the edge (no sources exist above the
    depth to smooth back down).  Pre-fix the denominator ``S @ dN_exp``
    integrated the full grid, so C dipped just below the edge and
    ``(1 - C) dN_exp`` spiked -- the closure experiment measured exactly this
    artifact.  With the truncated denominator ``S @ (dN_exp · 1[z <= z_depth])``
    both sides share the same support and the constant passes through.

    The synthetic survey is the infinite-galaxy limit of a constant-C0 thinning
    of the expected counts below the depth: its smoothed observed density is
    ``C0 · S @ (dN_exp · 1[z <= z_depth])`` (each galaxy kernel IS the
    operator's per-source kernel), injected via ``dN_obs_kde``.
    """
    from darksirens.redshift.completion import _SIGMA_SMOOTH, smoothing_operator

    cosmo = _cosmo()
    zgrid_np = np.asarray(zgrid)
    z_depth = float(zgrid_np[2 * len(zgrid_np) // 3])   # concrete, interior
    C0 = 0.7

    survey = _survey(z_depth=z_depth)
    catalog = EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((1, 2)),
        dzgals=jnp.full((1, 2), 0.02),
        wgals=jnp.ones((1, 2)),
        ngals=jnp.array([2], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 1)),   # lss == 1: pure completeness test
        dN_obs_kde=jnp.zeros((1, zgrid_np.size)),
        pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(cosmo, survey, catalog)
    dN_exp = np.asarray(grids.dN_exp)
    depth_mask = zgrid_np <= z_depth

    S = smoothing_operator()
    dN_obs = C0 * (S @ jnp.asarray(dN_exp * depth_mask))
    catalog = catalog._replace(dN_obs_kde=dN_obs[None, :])

    curves = completion_curves(cosmo, survey, catalog)
    C_eff = np.asarray(curves.C_eff)[0]        # == C below the depth (lss == 1)
    dN_miss = np.asarray(curves.dN_miss)[0]

    # Bands: within 2 sigma_smooth below the edge (the pre-fix artifact zone)
    # vs the same survey far from the edge (> 6 sigma below, away from z = 0).
    near = (zgrid_np > z_depth - 2.0 * _SIGMA_SMOOTH) & depth_mask
    far = (zgrid_np > 0.3) & (zgrid_np <= z_depth - 6.0 * _SIGMA_SMOOTH)
    assert near.any() and far.any()

    # No dip: C is the constant C0 right up to the edge, matching the
    # far-from-edge value node-for-node.
    np.testing.assert_allclose(C_eff[far], C0, rtol=1e-12)
    np.testing.assert_allclose(C_eff[near], C0, rtol=1e-12)
    np.testing.assert_allclose(
        C_eff[near], np.mean(C_eff[far]), rtol=1e-12
    )
    # No spike: the missing density stays at (1 - C0) dN_exp through the edge.
    np.testing.assert_allclose(
        dN_miss[near], (1.0 - C0) * dN_exp[near], rtol=1e-12
    )

    # Counterfactual teeth: the UNTRUNCATED denominator (the pre-fix recipe)
    # genuinely dips on this fixture -- at the edge the truncated smooth is
    # roughly half the full one -- so the assertions above are not vacuous.
    C_untrunc = np.clip(np.asarray(dN_obs) / np.asarray(S @ jnp.asarray(dN_exp)), 0.0, 1.0)
    assert C_untrunc[near].min() < 0.9 * C0


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
# builder: the completeness base is truncated at the catalog's own z_depth
# ---------------------------------------------------------------------------

def _depth_survey(path, z_depth=None, seed=0):
    """A load_survey catalog with galaxies well below ``z_depth``."""
    npix, maxg = 12, 6
    rng = np.random.default_rng(seed)
    zgals = np.full((npix, maxg), 100.0)
    dzgals = np.full((npix, maxg), 0.01)
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    for p in (0, 3, 7):
        zgals[p] = np.clip(0.25 + 0.05 * rng.standard_normal(maxg), 0.02, 0.45)
        wgals[p] = 1.0
        ngals[p] = maxg
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = 1
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
        f.create_dataset("ngals", data=ngals)
        if z_depth is not None:
            f.attrs["z_depth"] = float(z_depth)


def test_builder_truncates_completeness_base_at_catalog_z_depth(tmp_path, monkeypatch):
    """The offline Q builder must see the catalog's ``z_depth``.

    Both builders used to discard ``load_survey``'s depth, so Q was fit residual
    to an UNtruncated denominator while the likelihood uses the truncated one:
    C came out biased low (and the missing budget high) just below the edge --
    exactly the artifact the truncation was added to remove.  Compare the
    completeness the solver is handed with and without the attr, and check the
    provenance stamp.
    """
    import darksirens.cli.build_lognormal_completion as B

    Z_DEPTH = 0.6
    with_depth = str(tmp_path / "with_depth.h5")
    no_depth = str(tmp_path / "no_depth.h5")
    _depth_survey(with_depth, z_depth=Z_DEPTH)
    _depth_survey(no_depth, z_depth=None)

    captured = {}
    orig = B.poisson_lognormal_map

    def spy(N_obs, C, dN_exp, pk, **kw):
        captured.setdefault("C", []).append(np.asarray(C, dtype=float))
        return orig(N_obs, C, dN_exp, pk, **kw)

    monkeypatch.setattr(B, "poisson_lognormal_map", spy)
    _, _, diag_d = B.build_completion(with_depth, mode="radial", n_members=0,
                                      maxiter=3)
    _, _, diag_n = B.build_completion(no_depth, mode="radial", n_members=0,
                                      maxiter=3)
    C_depth, C_none = captured["C"]

    assert diag_d["fiducial_z_depth"] == pytest.approx(Z_DEPTH)
    assert "fiducial_z_depth" not in diag_n

    # Truncating the denominator RAISES C within ~2 sigma_smooth below the edge
    # (the untruncated denominator carries mass the numerator can never match).
    z = np.asarray(zgrid)
    from darksirens.redshift.completion import _SIGMA_SMOOTH
    edge = (z > Z_DEPTH - 2.0 * _SIGMA_SMOOTH) & (z <= Z_DEPTH)
    # The solver grid is uniform in chi, so compare on the same rows/columns.
    assert C_depth.shape == C_none.shape
    assert C_depth.sum() > C_none.sum()
    assert not np.allclose(C_depth, C_none)
    # Beyond the depth the truncated base is inert (C ~ 0 there).
    assert C_depth[:, edge].sum() >= C_none[:, edge].sum()


def test_builder_z_depth_cli_override_beats_catalog_attr(tmp_path):
    import darksirens.cli.build_lognormal_completion as B

    path = str(tmp_path / "with_depth.h5")
    _depth_survey(path, z_depth=0.6)
    _, _, diag = B.build_completion(path, mode="radial", n_members=0,
                                    maxiter=3, z_depth=1.1)
    assert diag["fiducial_z_depth"] == pytest.approx(1.1)


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


# ============================================================================
# z_depth must not double-count above-depth galaxies
# ============================================================================
#
# `_renorm_log_kw_below_depth` renormalises the catalog mixture to unit mass on
# [0, z_depth].  If the row's observed COUNT is left at the full N_obs, the host
# probability of every above-depth galaxy is relocated onto the below-depth
# redshifts -- while `_assemble_curves` counts those same galaxies again in the
# missing branch, which relaxes to the full dN_exp above the depth.
#
# The invariant: setting z_depth only moves above-depth hosts into the missing
# branch, so it must NOT change how much host probability sits below the depth.

def _depth_row_catalog(zs, dz=0.02):
    import jax.numpy as jnp

    from darksirens.core.types import EMCatalog
    from darksirens.redshift.grid import zgrid

    n = len(zs)
    nz = len(np.asarray(zgrid))
    return EMCatalog(
        apix=1.0,
        zgals=jnp.asarray(np.asarray(zs)[None, :]),
        dzgals=jnp.full((1, n), dz),
        wgals=jnp.ones((1, n)),
        ngals=jnp.asarray([n], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, nz)),
        dN_obs_kde=jnp.zeros((1, nz)),
        pixel_to_cache_idx=None,
    )


def _hosts_below(z_depth, zs, depth):
    """N_eff * (catalog mass below `depth`) -- the host count the prior places
    below the depth, i.e. exactly what multiplies p_cat in the numerator."""
    import jax.numpy as jnp

    from darksirens.core.types import CosmoParams, SurveyParams
    from darksirens.redshift.catalog import (
        catalog_kernel_state,
        eval_log_catalog_prior_state,
    )

    cat = _depth_row_catalog(zs)
    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)
    state = catalog_kernel_state(cosmo, survey, cat, z_depth=z_depth)
    zq = np.linspace(1e-4, depth, 4000)
    lp = np.array([
        float(eval_log_catalog_prior_state(jnp.asarray(z), 0, state, cat)) for z in zq
    ])
    shape_mass = _trapezoid(np.exp(lp), zq)
    n_eff = len(zs) * float(np.exp(np.asarray(state.log_depth_mass).ravel()[0]))
    return n_eff * shape_mass


def test_z_depth_does_not_inflate_below_depth_host_count():
    zs = [0.10, 0.15, 0.22, 0.28, 0.45, 0.55, 0.62, 0.75, 0.85, 0.95]
    depth = 0.3
    untruncated = _hosts_below(None, zs, depth)
    truncated = _hosts_below(depth, zs, depth)
    # 4 of the 10 galaxies are below the depth; kernels of width 0.02 leak a
    # little across it, so the reference is the untruncated value (~3.81), not 4.
    assert untruncated == pytest.approx(truncated, rel=1e-6), (
        f"z_depth changed the below-depth host count: {untruncated:.3f} -> "
        f"{truncated:.3f}. Pre-fix this was 10.00 (a 2.6x over-weighting) "
        "because the full N_obs was paired with a shape renormalised to unit "
        "mass below the depth."
    )
    assert 3.0 < truncated < 4.5


def test_depth_mass_is_unity_without_truncation():
    """No depth -> the amplitude correction must be an exact no-op."""
    import jax.numpy as jnp

    from darksirens.core.types import CosmoParams, SurveyParams
    from darksirens.redshift.catalog import catalog_kernel_state

    cat = _depth_row_catalog([0.1, 0.2, 0.3])
    state = catalog_kernel_state(
        CosmoParams(H0=67.74, Om0=0.3075),
        SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                     alpha_miss=0.0, sigma_kde=0.0),
        cat, z_depth=None,
    )
    assert np.allclose(np.asarray(state.log_depth_mass), 0.0)


def test_depth_above_all_galaxies_is_a_no_op():
    """A depth beyond every catalogued galaxy leaves the count untouched."""
    zs = [0.10, 0.15, 0.22, 0.28]
    assert _hosts_below(2.0, zs, 0.5) == pytest.approx(
        _hosts_below(None, zs, 0.5), rel=1e-6
    )


def test_deeply_truncated_kernel_norm_does_not_read_as_unit_mass():
    """A real galaxy far above the truncation has ~ZERO kernel mass below it.

    Both ``ndtr`` tails underflow to exactly 0 once the limit is >~39 sigma_eff
    away, so the direct span ``Phi(hi) - Phi(lo)`` is 0 and the pre-fix code
    fell through to the padding fallback, reporting log Z = 0 (UNIT mass) for a
    galaxy with essentially none.  Spectroscopic sigma_eff (the 1e-4 floor)
    reaches that regime for every galaxy a few thousandths above the depth, so
    the depth-mass consumer picked up their whole weight.
    """
    import jax.numpy as jnp

    from darksirens.redshift.catalog import _row_log_kernel_norms
    from darksirens.redshift.completion import log_galaxy_measure_grid
    from jax.scipy.special import log_ndtr

    cosmo = _cosmo()
    survey = _survey()
    log_g = log_galaxy_measure_grid(cosmo, survey)
    z_depth = 0.3
    zs = jnp.array([0.10, 0.28, 0.45, 0.95])
    real = jnp.array([True, True, True, True])

    for sig in (0.005, 1e-4):
        sig_eff = jnp.full(zs.shape, sig)
        log_Z_full = np.asarray(_row_log_kernel_norms(zs, sig_eff, real, log_g))
        log_Z_depth = np.asarray(
            _row_log_kernel_norms(zs, sig_eff, real, log_g, z_hi=z_depth)
        )
        assert np.all(np.isfinite(log_Z_depth))
        # Below-depth mass never exceeds the full-grid mass, and the galaxies
        # above the depth are suppressed by many hundreds of e-folds.
        assert np.all(log_Z_depth <= log_Z_full + 1e-12)
        assert np.all(log_Z_depth[2:] < log_Z_full[2:] - 100.0), log_Z_depth
        # Deep in the tail the mass is Phi((z_depth - z_gal)/sig) * g(z_depth).
        expect = float(log_ndtr((z_depth - zs[-1]) / sig)) + float(
            jnp.interp(z_depth, zgrid, log_g)
        )
        assert log_Z_depth[-1] == pytest.approx(expect, rel=1e-9)

    # Padding slots keep the inert 0.0 that keeps log(0) * 0 out of the
    # traced reduction.
    padded = np.asarray(_row_log_kernel_norms(
        jnp.array([0.10, 0.0]), jnp.array([0.01, 1e-4]),
        jnp.array([True, False]), log_g, z_hi=0.05))
    assert padded[1] == 0.0


# ---------------------------------------------------------------------------
# marked twin: z_depth on marked_catalog_kernel_state (349b717 follow-up)
# ---------------------------------------------------------------------------

def test_marked_depth_state_is_arrays_and_evaluates_under_jit():
    """Regression: 349b717 changed _renorm_log_kw_below_depth to return
    (log_kw, log_m) and updated the unmarked twin only; the marked row builder
    kept assigning the whole tuple to log_kw.  The state then built without
    error (tuples survive pytree flattening) and the first jitted evaluation
    died with TracerIntegerConversionError at state.log_kw[pix]."""
    import jax

    from darksirens.core.types import CosmoParams, SurveyParams
    from darksirens.redshift.catalog import (
        eval_log_catalog_prior_state,
        marked_catalog_kernel_state,
    )

    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)
    cat = _depth_row_catalog([0.10, 0.20, 0.45, 0.60])
    state, log_N_host = marked_catalog_kernel_state(
        cosmo, survey, cat, jnp.zeros_like(cat.zgals), z_depth=0.3
    )
    assert isinstance(state.log_kw, jnp.ndarray), type(state.log_kw)
    assert state.log_kw.shape == cat.zgals.shape
    assert np.asarray(state.log_depth_mass).shape == (1,)

    f = jax.jit(jax.vmap(
        lambda z, pix: eval_log_catalog_prior_state(z, pix, state, cat)
    ))
    lp = np.asarray(f(
        jnp.asarray([0.10, 0.25]), jnp.asarray([0, 0], dtype=jnp.int32)
    ))
    assert np.all(np.isfinite(lp)), lp


def test_marked_depth_amplitude_matches_unmarked_at_unit_marks():
    """With log_h == 0 and unit weights the marked twin must reproduce the
    unmarked depth treatment exactly: same renormalised kernels, same
    below-depth mass, and a marked amplitude exp(log_N_host + log_depth_mass)
    equal to the depth-scaled observed count.  Pre-fix the marked amplitude
    kept the FULL marked mass, so every above-depth galaxy was counted in the
    observed branch while _assemble_curves counted it again in the missing
    branch (the double count 349b717 removed from the unmarked path)."""
    from darksirens.core.types import CosmoParams, SurveyParams
    from darksirens.redshift.catalog import (
        catalog_kernel_state,
        marked_catalog_kernel_state,
    )

    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                          alpha_miss=0.0, sigma_kde=0.0)
    zs = [0.10, 0.15, 0.22, 0.28, 0.45, 0.55, 0.62, 0.75, 0.85, 0.95]
    cat = _depth_row_catalog(zs)
    depth = 0.3

    ref = catalog_kernel_state(cosmo, survey, cat, z_depth=depth)
    state, log_N_host = marked_catalog_kernel_state(
        cosmo, survey, cat, jnp.zeros_like(cat.zgals), z_depth=depth
    )
    np.testing.assert_allclose(
        np.asarray(state.log_kw), np.asarray(ref.log_kw), rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(state.log_depth_mass), np.asarray(ref.log_depth_mass),
        rtol=1e-12,
    )
    marked_amp = float(np.exp(np.asarray(log_N_host + state.log_depth_mass))[0])
    unmarked_amp = float(len(zs) * np.exp(np.asarray(ref.log_depth_mass))[0])
    assert marked_amp == pytest.approx(unmarked_amp, rel=1e-10)
    # 4 of 10 galaxies sit below the depth (0.02-wide kernels leak a little),
    # so the amplitude must be far below the raw count of 10.
    assert 3.0 < marked_amp < 4.5
