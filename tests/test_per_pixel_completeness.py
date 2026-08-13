"""Field-level PR-2: per-pixel selection fraction ``C_p(z) = f_p C(z)``.

Pins, per PLAN §7 PR-2:
* the depth-map degrade (area weighting, NaN masked_frac on uncovered pixels,
  the ``sum_p f_p Omega_pix`` area accounting);
* ``--per_pixel_completeness`` OFF is bit-identical (``f_p_rows = None``) and
  ``f_p ≡ 1`` is exactly the no-flag likelihood;
* the flag-on change to ``dN_miss`` is the SIGNED, QUANTITATIVE prediction
  ``Delta dN_miss = (1 - f_p) C(z) dN_exp(z)`` below the depth (not a smoke
  test);
* the field normalizer carries the same budget: occupied rows get
  ``f_p C``, empty pixels ``n_empty - C(z) sum_empty f_p``;
* the loader refuses the combinations whose budgets are not derived
  (per_pixel c_mode, K>=2, Q tables, strata).
"""
from __future__ import annotations

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import h5py
import jax.numpy as jnp  # noqa: E402

from darksirens.catalogs.depth_map import (  # noqa: E402
    load_selection_fraction,
)
from darksirens.core.types import (  # noqa: E402
    C_MODE_SELECTION_STRUCT,
    CosmoParams,
    EMCatalog,
    SurveyParams,
)
from darksirens.redshift.completion import (  # noqa: E402
    _field_missing_curve,
    build_field_normalization_inputs,
    completion_curves,
)
from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.redshift.selection import c_sel_gaussian  # noqa: E402

THETA = dict(m_lim=24.0, M0hat=-20.2, sigma_M=1.0)
Z_DEPTH = 0.35


def _survey(**kw):
    base = dict(n0=1e-4, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                alpha_miss=1.0, sigma_kde=0.0, z_depth=Z_DEPTH,
                c_mode=C_MODE_SELECTION_STRUCT, **THETA)
    base.update(kw)
    return SurveyParams(**base)


def _cosmo(h0=67.74):
    return CosmoParams(H0=h0, Om0=0.3075, w0=-1.0, wa=0.0)


def _em(n_pix=12, max_gals=3, seed=5, f_p_rows=None, **field_kw):
    rng = np.random.default_rng(seed)
    zg = np.full((n_pix, max_gals), 100.0)
    dz = np.ones((n_pix, max_gals))
    w = np.zeros((n_pix, max_gals))
    ng = np.zeros(n_pix, dtype=int)
    for p in range(n_pix):
        n = int(rng.integers(0, max_gals))
        zs = np.sort(rng.uniform(0.02, 0.4, n))
        zg[p, :n] = zs
        dz[p, :n] = 1e-3
        w[p, :n] = 1.0
        ng[p] = n
    import healpy as hp
    apix = hp.nside2pixarea(1)
    return EMCatalog(
        apix=float(apix), zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(w), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, int(zgrid.size))), dN_obs_kde=None,
        pixel_to_cache_idx=None,
        f_p_rows=(None if f_p_rows is None else jnp.asarray(f_p_rows)),
        **field_kw)


# ------------------------------------------------------------ depth map


def _write_mth_map(path, nside, masked_frac, counts):
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = nside
        f.attrs["ordering"] = "RING"
        f.create_dataset("masked_frac", data=masked_frac)
        f.create_dataset("counts", data=counts)


def test_degrade_area_and_nan_handling(tmp_path):
    import healpy as hp

    nside_in, nside_out = 4, 2
    npix = 12 * nside_in ** 2
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5, size=npix).astype(np.uint64)
    masked_frac = np.where(counts > 0, rng.uniform(0, 0.5, npix), np.nan)
    p = tmp_path / "mth.h5"
    _write_mth_map(p, nside_in, masked_frac, counts)

    sfm = load_selection_fraction(p, nside_out)
    assert sfm.f_p.shape == (12 * nside_out ** 2,)
    assert np.all(np.isfinite(sfm.f_p))
    assert np.all((sfm.f_p >= 0) & (sfm.f_p <= 1))

    # direct reference: mean over the 4 NESTED children, uncovered -> 0
    ring2nest = hp.ring2nest(nside_in, np.arange(npix))
    f_native = np.where(counts > 0,
                        np.clip(1 - np.nan_to_num(masked_frac, nan=1.0), 0, 1),
                        0.0)
    f_nest = np.zeros(npix)
    f_nest[ring2nest] = f_native
    ref_nest = f_nest.reshape(-1, 4).mean(1)
    ref = np.zeros_like(ref_nest)
    ref[hp.nest2ring(nside_out, np.arange(ref_nest.size))] = ref_nest
    np.testing.assert_allclose(sfm.f_p, ref, rtol=0, atol=1e-15)

    omega = hp.nside2pixarea(nside_out, degrees=True)
    np.testing.assert_allclose(sfm.area_deg2, sfm.f_p.sum() * omega)

    report = sfm.coverage_report(np.zeros(sfm.f_p.size, dtype=int))
    assert report["n_occupied"] == 0
    assert report["n_off_footprint"] == int((sfm.f_p == 0).sum())


def test_all_covered_unmasked_degrades_to_ones(tmp_path):
    nside_in = 4
    npix = 12 * nside_in ** 2
    _write_mth_map(tmp_path / "m.h5", nside_in,
                   np.zeros(npix), np.ones(npix, dtype=np.uint64))
    sfm = load_selection_fraction(tmp_path / "m.h5", 2)
    np.testing.assert_allclose(sfm.f_p, 1.0, rtol=0, atol=0)


# ------------------------------------------------------------ numerator


def test_fp_ones_is_bit_identical_and_none_is_off():
    sv = _survey()
    base = completion_curves(_cosmo(), sv, _em())
    ones = completion_curves(_cosmo(), sv, _em(f_p_rows=np.ones(12)))
    for a, b in zip(base[:4], ones[:4]):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_fp_dN_miss_signed_quantitative():
    sv = _survey()
    f_vals = np.linspace(0.4, 1.0, 12)
    base = completion_curves(_cosmo(), sv, _em())
    withf = completion_curves(_cosmo(), sv, _em(f_p_rows=f_vals))
    dN_miss_0 = np.asarray(base.dN_miss)
    dN_miss_f = np.asarray(withf.dN_miss)

    C = np.asarray(c_sel_gaussian(zgrid, THETA["m_lim"], THETA["M0hat"],
                                  THETA["sigma_M"], 67.74, 0.3075, -1.0, 0.0))
    from darksirens.redshift.completion import _precompute_grids
    below = np.asarray(zgrid) <= Z_DEPTH
    dN_exp = np.asarray(_precompute_grids(_cosmo(), sv, _em()).dN_exp)
    for p in range(12):
        delta = dN_miss_f[p] - dN_miss_0[p]
        pred = (1.0 - f_vals[p]) * C * dN_exp
        # signed: missing density can only GROW when f_p < 1, by exactly
        # (1 - f_p) C dN_exp, and only below the depth
        assert np.all(delta[below] >= -1e-12)
        np.testing.assert_allclose(delta[below], pred[below],
                                   rtol=1e-10, atol=1e-30)
        assert np.allclose(delta[~below], 0.0)


# ------------------------------------------------------------ normalizer


def test_field_missing_curve_fp_budget():
    n_pix = 12
    em0 = _em()
    field = build_field_normalization_inputs(
        em0.zgals, em0.wgals, em0.ngals)
    f_full = np.linspace(0.3, 1.0, n_pix)
    occ = np.asarray(field.occupied_pixels)
    f_occ = f_full[occ]
    f_empty_sum = float(f_full.sum() - f_occ.sum())

    kw = dict(field_dN_obs_s=field.dN_obs_s,
              field_n_empty=jnp.asarray(float(field.n_empty)),
              field_N_obs_total=jnp.asarray(float(field.N_obs_total)),
              field_occupied_pixels=jnp.asarray(occ))
    sv = _survey()
    em_nofp = _em(**kw)
    em_fp = _em(**kw, field_f_p_occ=jnp.asarray(f_occ),
                field_f_p_empty_sum=jnp.asarray(f_empty_sum))

    V0, dN_exp = _field_missing_curve(_cosmo(), sv, em_nofp)
    Vf, _ = _field_missing_curve(_cosmo(), sv, em_fp)
    V0, Vf = np.asarray(V0), np.asarray(Vf)

    C = np.asarray(c_sel_gaussian(zgrid, THETA["m_lim"], THETA["M0hat"],
                                  THETA["sigma_M"], 67.74, 0.3075, -1.0, 0.0))
    below = np.asarray(zgrid) <= Z_DEPTH
    n_occ = occ.size
    n_empty = n_pix - n_occ
    # direct references (selection mode: every pixel carries the survey curve)
    ref0 = n_occ * (1 - C) + n_empty * (1 - C)
    reff = (np.sum(1 - f_occ[:, None] * C[None, :], axis=0)
            + n_empty - C * f_empty_sum)
    np.testing.assert_allclose(V0[below], ref0[below], rtol=1e-10)
    np.testing.assert_allclose(Vf[below], reff[below], rtol=1e-10)
    # beyond depth both relax to the total pixel count
    np.testing.assert_allclose(V0[~below], n_pix, rtol=0, atol=1e-9)
    np.testing.assert_allclose(Vf[~below], n_pix, rtol=0, atol=1e-9)
    # signed: f_p < 1 can only increase the missing budget
    assert np.all(Vf[below] - V0[below] >= -1e-12)


def test_field_missing_curve_fp_refusals():
    em0 = _em()
    field = build_field_normalization_inputs(em0.zgals, em0.wgals, em0.ngals)
    occ = np.asarray(field.occupied_pixels)
    kw = dict(field_dN_obs_s=field.dN_obs_s,
              field_n_empty=jnp.asarray(float(field.n_empty)),
              field_N_obs_total=jnp.asarray(float(field.N_obs_total)),
              field_occupied_pixels=jnp.asarray(occ))
    # f_p without the empty sum
    em_bad = _em(**kw, field_f_p_occ=jnp.ones(occ.size))
    with pytest.raises(ValueError, match="field_f_p_empty_sum"):
        _field_missing_curve(_cosmo(), _survey(), em_bad)
    # f_p under a non-aggregate c_mode (per-pixel ratio)
    em_pp = _em(**kw, field_f_p_occ=jnp.ones(occ.size),
                field_f_p_empty_sum=jnp.asarray(0.0))
    with pytest.raises(NotImplementedError, match="aggregate/selection"):
        _field_missing_curve(_cosmo(), _survey(c_mode=0, m_lim=None,
                                               M0hat=None, sigma_M=None),
                             em_pp)


# ------------------------------------------------------------ loader


def test_loader_refusals(tmp_path):
    from types import SimpleNamespace

    from darksirens.inference.loaders import (
        attach_selection_fraction_inputs,
    )

    nside_in = 4
    npix = 12 * nside_in ** 2
    _write_mth_map(tmp_path / "m.h5", nside_in,
                   np.zeros(npix), np.ones(npix, dtype=np.uint64))
    path = str(tmp_path / "m.h5")

    data = dict(nside=2, ngals=np.zeros(48, dtype=int))
    ok = SimpleNamespace(per_pixel_completeness=path, c_mode="selection",
                         n_catalogs=1, lss_completion=None,
                         selection_strata_by_catalog=None, selection_fit=None)
    out = attach_selection_fraction_inputs(ok, dict(data))
    assert out["f_p_map"].shape == (48,)
    np.testing.assert_allclose(out["f_p_map"], 1.0)

    with pytest.raises(ValueError, match="c_mode"):
        attach_selection_fraction_inputs(
            SimpleNamespace(per_pixel_completeness=path, c_mode="per_pixel"),
            dict(data))
    with pytest.raises(NotImplementedError, match="K=1"):
        attach_selection_fraction_inputs(
            SimpleNamespace(per_pixel_completeness=path, c_mode="selection",
                            n_catalogs=2), dict(data))
    with pytest.raises(NotImplementedError, match="Q table"):
        attach_selection_fraction_inputs(
            SimpleNamespace(per_pixel_completeness=path, c_mode="selection",
                            n_catalogs=1, lss_completion="q.h5"), dict(data))
    with pytest.raises(NotImplementedError, match="strat"):
        attach_selection_fraction_inputs(
            SimpleNamespace(per_pixel_completeness=path, c_mode="selection",
                            n_catalogs=1, lss_completion=None,
                            selection_strata_by_catalog=[[(24.0, 0.0, 1.0)]]),
            dict(data))
