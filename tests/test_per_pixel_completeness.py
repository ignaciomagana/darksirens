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
    C_MODE_AGGREGATE_STRUCT,
    C_MODE_SELECTION_STRUCT,
    CosmoParams,
    EMCatalog,
    SurveyParams,
)
from darksirens.redshift.completion import (  # noqa: E402
    _aggregate_dN_obs_sum,
    _field_missing_curve,
    _precompute_grids,
    build_field_lss_q_fp_empty_sum,
    build_field_lss_q_fp_empty_sum_members,
    build_field_lss_q_inputs,
    build_field_lss_q_member_inputs,
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


# ------------------------------------------- aggregate Cbar sky normalization
#
# The aggregate estimator forms Cbar from OBSERVED COUNTS, so its denominator
# has to count the sky the survey could observe.  The numerator sums only
# footprint pixels; normalizing it by the full sphere makes Cbar the all-sky
# mean <f_p> * C_true, and the consumers' C_p = f_p * Cbar then applies the
# mask loss TWICE -- a fraction (1 - <f_p>) of the catalogued galaxies stays
# in the missing budget on top of its own observed counts, inflating the
# dark-host branch.  These pin the covered-sky denominator Sum_p f_p.

N_PIX_HALF = 12
COVERED = np.arange(N_PIX_HALF) < N_PIX_HALF // 2   # half-sky footprint
F_P_HALF = np.where(COVERED, 1.0, 0.0)              # f_p = 1 on-footprint


def _agg_survey(**kw):
    base = dict(n0=1e-8, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                alpha_miss=1.0, sigma_kde=0.0, z_depth=Z_DEPTH,
                c_mode=C_MODE_AGGREGATE_STRUCT,
                m_lim=None, M0hat=None, sigma_M=None)
    base.update(kw)
    return SurveyParams(**base)


_UNSET = object()


def _half_sky_em(f_p=None, f_p_total_sum=_UNSET, max_gals=6, seed=3, **kw):
    """Catalog whose galaxies live ONLY on the covered half of the sky."""
    import healpy as hp
    rng = np.random.default_rng(seed)
    zg = np.full((N_PIX_HALF, max_gals), 100.0)
    dz = np.ones((N_PIX_HALF, max_gals))
    w = np.zeros((N_PIX_HALF, max_gals))
    ng = np.zeros(N_PIX_HALF, dtype=int)
    for p in range(N_PIX_HALF):
        if not COVERED[p]:
            continue
        zg[p, :] = np.sort(rng.uniform(0.02, 0.3, max_gals))
        dz[p, :], w[p, :] = 1e-3, 1.0
        ng[p] = max_gals
    if f_p_total_sum is _UNSET:
        f_p_total_sum = None if f_p is None else float(np.sum(f_p))
    return EMCatalog(
        apix=float(hp.nside2pixarea(1)),
        zgals=jnp.asarray(zg), dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(w), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, int(zgrid.size))), dN_obs_kde=None,
        pixel_to_cache_idx=None,
        f_p_rows=(None if f_p is None else jnp.asarray(f_p)),
        f_p_total_sum=(None if f_p_total_sum is None
                       else jnp.asarray(float(f_p_total_sum))),
        **kw)


def _covered_half_C(grids):
    """C measured directly from the covered half: obs / (n_covered dN_exp_s)."""
    obs = np.asarray(_aggregate_dN_obs_sum(_half_sky_em()))
    smooth = np.asarray(grids.dN_exp_smooth)
    return obs / (int(COVERED.sum()) * np.where(smooth > 0.0, smooth, 1.0))


def _tuned_agg_survey(c_target=0.7):
    """Aggregate survey with n0 set so max Cbar below the depth is ``c_target``.

    Keeps the covered-half completeness inside [0, 1], where the consumers'
    clip is inert -- a clipped Cbar would hide the budget algebra below.
    """
    sv = _agg_survey()
    C = _covered_half_C(_precompute_grids(_cosmo(), sv, _half_sky_em()))
    below = np.asarray(zgrid) <= Z_DEPTH
    return sv._replace(n0=float(sv.n0) * float(C[below].max()) / c_target)


def test_aggregate_cbar_normalizes_by_the_covered_sky():
    """Half-sky footprint, f_p = 1 on it: Cbar is the COVERED-half C."""
    sv = _agg_survey()
    grids = _precompute_grids(_cosmo(), sv, _half_sky_em(f_p=F_P_HALF))
    Cbar = np.asarray(grids.C_bar_raw)
    C_covered = _covered_half_C(grids)
    below = np.asarray(zgrid) <= Z_DEPTH

    np.testing.assert_allclose(Cbar[below], C_covered[below], rtol=1e-12)
    # The full-sphere denominator is exactly half of the covered sky here, so
    # the double-count reads as Cbar = C_covered / 2 -- pinned as REFUSED.
    assert np.max(np.abs(Cbar[below] - 0.5 * C_covered[below])) > 1e-6

    # Without f_p the sky normalizer is the whole sphere (legacy, unchanged).
    Cbar_nofp = np.asarray(
        _precompute_grids(_cosmo(), sv, _half_sky_em()).C_bar_raw)
    np.testing.assert_allclose(Cbar_nofp[below], 0.5 * C_covered[below],
                               rtol=1e-12)


def test_aggregate_fp_field_budget_closes():
    """observed + modelled-missing == the full-sky expectation, under f_p.

    In the smoothed denominator Cbar is defined against, the field normalizer's
    budget is V(z) = N_pix - Cbar Sum_all f_p, so closure

        Sum_p dN_obs_s + V dN_exp_s == N_pix dN_exp_s

    holds iff Cbar = Sum_p dN_obs_s / (Sum_all f_p * dN_exp_s).  With the
    full-sphere denominator the left side overshoots by exactly
    (1 - <f_p>_allsky) x the observed sum -- half of it for this footprint.
    """
    sv = _tuned_agg_survey()
    em0 = _half_sky_em()
    field = build_field_normalization_inputs(em0.zgals, em0.wgals, em0.ngals)
    occ = np.asarray(field.occupied_pixels)
    f_occ = F_P_HALF[occ]
    f_empty_sum = float(F_P_HALF.sum() - f_occ.sum())
    em_fp = _half_sky_em(
        f_p=F_P_HALF,
        field_dN_obs_s=field.dN_obs_s,
        field_n_empty=jnp.asarray(float(field.n_empty)),
        field_N_obs_total=jnp.asarray(float(field.N_obs_total)),
        field_occupied_pixels=jnp.asarray(occ),
        field_f_p_occ=jnp.asarray(f_occ),
        field_f_p_empty_sum=jnp.asarray(f_empty_sum))

    grids = _precompute_grids(_cosmo(), sv, em_fp)
    V = np.asarray(_field_missing_curve(_cosmo(), sv, em_fp)[0])
    Cbar = np.asarray(grids.C_bar_raw)
    smooth = np.asarray(grids.dN_exp_smooth)
    obs = np.asarray(_aggregate_dN_obs_sum(em0))
    below = np.asarray(zgrid) <= Z_DEPTH
    assert Cbar[below].max() <= 1.0            # the clip is inert here

    np.testing.assert_allclose((obs + V * smooth)[below],
                               (N_PIX_HALF * smooth)[below], rtol=1e-6)
    # The normalizer side: V is the row-by-row budget with C_p = f_p Cbar.
    ref = (np.sum(1.0 - f_occ[:, None] * Cbar[None, :], axis=0)
           + float(field.n_empty) - Cbar * f_empty_sum)
    np.testing.assert_allclose(V[below], ref[below], rtol=1e-6)
    # ... and that budget equals N_pix - Cbar Sum_all f_p.
    np.testing.assert_allclose(
        V[below], (N_PIX_HALF - Cbar * float(F_P_HALF.sum()))[below],
        rtol=1e-6)


def test_aggregate_fp_without_the_covered_sky_sum_is_refused():
    """f_p under aggregate without ``f_p_total_sum`` must not fall back."""
    with pytest.raises(ValueError, match="f_p_total_sum"):
        _precompute_grids(_cosmo(), _agg_survey(),
                          _half_sky_em(f_p=F_P_HALF, f_p_total_sum=None))


def test_selection_curve_carries_no_sky_normalization():
    """c_mode=selection has no analogue of the aggregate double-count.

    Its C is the fitted truncated-LF detection probability ``P(m <= m_lim|z)``
    -- no observed counts and no pixel count enter it, so it is already the
    TRUE per-covered-sky completeness and ``C_p = f_p C_sel`` carries the mask
    loss exactly once.  Pinned by showing the curve is bit-identical to the
    bare selection function and blind to both the footprint and ``f_p``.
    """
    sv = _survey()
    ref = np.asarray(c_sel_gaussian(zgrid, THETA["m_lim"], THETA["M0hat"],
                                    THETA["sigma_M"], 67.74, 0.3075, -1.0, 0.0))
    C_nofp = np.asarray(
        _precompute_grids(_cosmo(), sv, _half_sky_em()).C_bar_raw)
    C_fp = np.asarray(
        _precompute_grids(_cosmo(), sv, _half_sky_em(f_p=F_P_HALF)).C_bar_raw)
    assert np.array_equal(C_nofp, ref)
    assert np.array_equal(C_fp, ref)


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
    # A Q table is REFUSED unless its artifact stamps f_p_aware: Q is fit to
    # observed counts, so it already carries the footprint and C_p = f_p C would
    # apply the survey mask TWICE (measured: H0 = 41.24 [36.1, 46.3] against a
    # truth of 67.74 on the closure mock, the tightest arm in that run).
    det = dict(data, lss_completion_logq=np.zeros((48, int(zgrid.size))))
    q_opts = lambda **kw: SimpleNamespace(  # noqa: E731
        per_pixel_completeness=path, c_mode="selection", n_catalogs=1,
        lss_completion="q.h5", selection_strata_by_catalog=None,
        selection_fit=None, **kw)
    with pytest.raises(NotImplementedError, match="DOUBLE-COUNTS"):
        attach_selection_fraction_inputs(q_opts(), dict(det))
    # ... admitted when the artifact asserts its builder removed the mask ...
    aware = dict(det, lss_completion_provenance={"f_p_aware": True})
    out_q = attach_selection_fraction_inputs(q_opts(), dict(aware))
    assert out_q["f_p_map"].shape == (48,)
    # ... or when the operator takes the exposed configuration deliberately.
    out_forced = attach_selection_fraction_inputs(
        q_opts(allow_double_counted_mask=True), dict(det))
    assert out_forced["f_p_map"].shape == (48,)
    # An ENSEMBLE is count-derived too, so the same gate applies to it, and it
    # passes once the artifact asserts the mask was removed.
    ens = dict(data,
               lss_completion_logq=np.zeros((48, int(zgrid.size))),
               lss_completion_logq_members=np.zeros((3, 48, int(zgrid.size))),
               lss_completion_provenance={"f_p_aware": True})
    out_e = attach_selection_fraction_inputs(q_opts(), dict(ens))
    assert out_e["f_p_map"].shape == (48,)
    with pytest.raises(NotImplementedError, match="strat"):
        attach_selection_fraction_inputs(
            SimpleNamespace(per_pixel_completeness=path, c_mode="selection",
                            n_catalogs=1, lss_completion=None,
                            selection_strata_by_catalog=[[(24.0, 0.0, 1.0)]]),
            dict(data))


@pytest.mark.parametrize("stamp", [None, False, True])
def test_f_p_aware_travels_from_the_artifact_to_the_gate(tmp_path, stamp):
    """The stamp is read off the FILE, not asserted by the caller.

    Written as a round trip because the gate is only as good as the plumbing
    under it: an artifact that never records ``f_p_aware`` (``stamp=None``, i.e.
    every Q table built so far) must reach the loader as False, not as a missing
    key that some later ``get`` defaults to True.
    """
    from types import SimpleNamespace

    from darksirens.catalogs.lss import maybe_load_lss_completion
    from darksirens.inference.loaders import attach_selection_fraction_inputs
    from darksirens.redshift.lognormal_completion import (
        load_lss_completion_hdf5, save_lss_completion_hdf5)

    nside_in = 4
    npix = 12 * nside_in ** 2
    _write_mth_map(tmp_path / "m.h5", nside_in,
                   np.zeros(npix), np.ones(npix, dtype=np.uint64))

    q_path = str(tmp_path / "q.h5")
    save_lss_completion_hdf5(
        q_path, logq_map=np.zeros((48, int(zgrid.size))),
        zgrid=np.asarray(zgrid), indexing="global", c_mode="selection",
        f_p_aware=stamp)
    assert load_lss_completion_hdf5(q_path)["f_p_aware"] is bool(stamp)

    opts = SimpleNamespace(
        lss_completion=q_path, survey_path=None, universe_model="dark_sirens",
        c_mode="selection", lss_marginalize=False, n_catalogs=1,
        per_pixel_completeness=str(tmp_path / "m.h5"),
        selection_strata_by_catalog=None, selection_fit=None)
    data = dict(nside=2, ngals=np.zeros(48, dtype=int))
    data.update(maybe_load_lss_completion(opts, zgrid=zgrid))
    assert data["lss_completion_provenance"]["f_p_aware"] is bool(stamp)

    if stamp:
        assert attach_selection_fraction_inputs(
            opts, dict(data))["f_p_map"].shape == (48,)
    else:
        with pytest.raises(NotImplementedError, match="DOUBLE-COUNTS"):
            attach_selection_fraction_inputs(opts, dict(data))


# ------------------------------------------------- S-3: f_p alongside a Q table


def _q_field_kw(n_pix=12, seed=11, f_full=None):
    """Field-normalizer inputs with a deterministic Q table, +/- f_p."""
    em0 = _em(n_pix=n_pix)
    field = build_field_normalization_inputs(em0.zgals, em0.wgals, em0.ngals)
    occ = np.asarray(field.occupied_pixels)
    rng = np.random.default_rng(seed)
    logq = rng.normal(0.0, 0.4, size=(n_pix, int(zgrid.size)))
    q_occ, q_empty_sum = build_field_lss_q_inputs(logq, occ, n_pix)
    kw = dict(field_dN_obs_s=field.dN_obs_s,
              field_n_empty=jnp.asarray(float(field.n_empty)),
              field_N_obs_total=jnp.asarray(float(field.N_obs_total)),
              field_occupied_pixels=jnp.asarray(occ),
              field_lss_q=q_occ, field_lss_q_empty_sum=q_empty_sum)
    return em0, field, occ, logq, kw


def test_fp_q_empty_budget_matches_brute_force():
    """``Sum_{p empty} f_p Q_p`` against an explicit loop, and the member twin."""
    n_pix = 12
    _, field, occ, logq, _ = _q_field_kw(n_pix)
    f_full = np.linspace(0.0, 1.0, n_pix)
    got = np.asarray(build_field_lss_q_fp_empty_sum(logq, occ, n_pix, f_full))
    empty = np.setdiff1d(np.arange(n_pix), occ)
    ref = (f_full[empty][:, None] * np.exp(logq[empty])).sum(axis=0)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)

    logq_m = np.stack([logq, logq + 0.1, logq - 0.2])
    got_m = np.asarray(
        build_field_lss_q_fp_empty_sum_members(logq_m, occ, n_pix, f_full))
    ref_m = np.stack([
        (f_full[empty][:, None] * np.exp(logq_m[m][empty])).sum(axis=0)
        for m in range(3)])
    assert got_m.shape == (3, int(zgrid.size))
    np.testing.assert_allclose(got_m, ref_m, rtol=0, atol=1e-12)


def test_fp_ones_with_a_q_table_recovers_the_no_fp_path():
    """f_p == 1 everywhere must reproduce the no-f_p Q path.

    To ROUNDING, not bit-exactly, and the distinction is the point: the f_p
    branch evaluates ``Sum_empty Q_p - C Sum_empty f_p Q_p`` where the shipped
    branch evaluates ``(1 - C) Sum_empty Q_p``, and ``a - Ca`` is not the same
    float as ``(1 - C)a``.  The shipped branch itself is untouched code, so it
    stays bit-exact -- that is what the golden suite pins; here the claim is
    only that the generalisation is the same function of the same inputs.
    """
    n_pix = 12
    _, field, occ, logq, kw = _q_field_kw(n_pix)
    fp_empty = build_field_lss_q_fp_empty_sum(
        logq, occ, n_pix, np.ones(n_pix))
    em_nofp = _em(n_pix=n_pix, **kw)
    em_fp = _em(n_pix=n_pix, **kw,
                field_f_p_occ=jnp.ones(occ.size, dtype=jnp.float32),
                field_f_p_empty_sum=jnp.asarray(float(n_pix - occ.size)),
                field_lss_q_fp_empty_sum=fp_empty)
    V0, _ = _field_missing_curve(_cosmo(), _survey(), em_nofp)
    Vf, _ = _field_missing_curve(_cosmo(), _survey(), em_fp)
    V0, Vf = np.asarray(V0), np.asarray(Vf)
    np.testing.assert_allclose(Vf, V0, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(Vf - V0)) < 1e-14 * max(np.max(np.abs(V0)), 1.0)


def test_fp_q_budget_is_the_q_weighted_formula():
    """The empty budget must weight f_p INSIDE the Q sum, not outside it."""
    n_pix = 12
    _, field, occ, logq, kw = _q_field_kw(n_pix)
    f_full = np.linspace(0.2, 1.0, n_pix)
    f_occ = f_full[occ]
    empty = np.setdiff1d(np.arange(n_pix), occ)
    fp_empty = build_field_lss_q_fp_empty_sum(logq, occ, n_pix, f_full)
    em_fp = _em(n_pix=n_pix, **kw,
                field_f_p_occ=jnp.asarray(f_occ.astype(np.float32)),
                field_f_p_empty_sum=jnp.asarray(float(f_full[empty].sum())),
                field_lss_q_fp_empty_sum=fp_empty)
    V, _ = _field_missing_curve(_cosmo(), _survey(), em_fp)
    V = np.asarray(V)

    C = np.asarray(c_sel_gaussian(zgrid, THETA["m_lim"], THETA["M0hat"],
                                  THETA["sigma_M"], 67.74, 0.3075, -1.0, 0.0))
    q = np.exp(logq)
    below = np.asarray(zgrid) <= Z_DEPTH
    ref = ((1.0 - f_occ[:, None] * C[None, :]) * q[occ]).sum(axis=0) \
        + q[empty].sum(axis=0) - C * (f_full[empty][:, None] * q[empty]).sum(axis=0)
    np.testing.assert_allclose(V[below], ref[below], rtol=1e-6)
    # and the WRONG pairing (f_p outside the Q sum) is measurably different
    wrong = ((1.0 - f_occ[:, None] * C[None, :]) * q[occ]).sum(axis=0) \
        + (n_pix - occ.size) - C * f_full[empty].sum()
    assert not np.allclose(V[below], wrong[below], rtol=1e-3)


def _fp_kw(kw, occ, logq, n_pix, lo=0.2):
    """``kw`` from :func:`_q_field_kw` plus the three f_p leaves."""
    f_full = np.linspace(lo, 1.0, n_pix)
    empty = np.setdiff1d(np.arange(n_pix), occ)
    return dict(
        kw,
        field_f_p_occ=jnp.asarray(f_full[occ].astype(np.float32)),
        field_f_p_empty_sum=jnp.asarray(float(f_full[empty].sum())),
        field_lss_q_fp_empty_sum=build_field_lss_q_fp_empty_sum(
            logq, occ, n_pix, f_full))


@pytest.mark.parametrize("with_fp", [False, True])
def test_folded_occupied_budget_matches_the_scan(monkeypatch, with_fp):
    """The folded occupied budget is the chunked scan, to rounding.

    Under an aggregate selection curve with a Q table the occupied rows carry
    ``(1 - f_p Cbar) Q_p`` with a constant ``Q_p``, so ``_field_missing_curve``
    sums the rows once instead of scanning them per proposal.  A/B against the
    scan itself (``_fold_occupied_rows`` monkeypatched off) rather than against
    a re-derivation, so the claim is old-code-vs-new-code on the same inputs.
    The two differ only in how the rounding falls (a different factorization
    of the same sum, not a different sum).
    """
    from darksirens.redshift import completion as completion_mod

    n_pix = 12
    _, _, occ, logq, kw = _q_field_kw(n_pix)
    if with_fp:
        kw = _fp_kw(kw, occ, logq, n_pix)
    em = _em(n_pix=n_pix, **kw)

    # the fold must actually FIRE on this configuration, or the A/B is vacuous
    taken = []
    real = completion_mod._fold_occupied_rows
    monkeypatch.setattr(
        completion_mod, "_fold_occupied_rows",
        lambda *a: (taken.append(real(*a)), taken[-1])[1])
    V_fold = np.asarray(_field_missing_curve(_cosmo(), _survey(), em)[0])
    assert taken and all(taken)

    monkeypatch.setattr(completion_mod, "_fold_occupied_rows",
                        lambda *a: False)
    V_scan = np.asarray(_field_missing_curve(_cosmo(), _survey(), em)[0])
    assert np.max(np.abs(V_scan)) > 0.0
    np.testing.assert_allclose(V_fold, V_scan, rtol=1e-13, atol=0.0)


def test_the_fold_is_only_taken_where_every_row_shares_one_C_and_a_fixed_Q():
    """The predicate, argument by argument: anything per-row keeps the scan."""
    from darksirens.redshift.completion import _fold_occupied_rows

    assert _fold_occupied_rows(True, True, False, False, False)
    assert not _fold_occupied_rows(False, True, False, False, False)  # per-pixel C
    assert not _fold_occupied_rows(True, False, False, False, False)  # no Q table
    assert not _fold_occupied_rows(True, True, True, False, False)    # latent Q
    assert not _fold_occupied_rows(True, True, False, True, False)    # stratified
    assert not _fold_occupied_rows(True, True, False, False, True)    # delta_g


def test_per_pixel_c_mode_with_a_q_table_is_bit_identical(monkeypatch):
    """The excluded configurations are untouched code, so they stay BIT-exact.

    ``C_p`` there is the row's own observed ratio, which the fold's algebra
    cannot factor out; the guard is that turning the hook off changes nothing
    at all, not merely nothing to rounding.
    """
    from darksirens.redshift import completion as completion_mod

    n_pix = 12
    _, _, occ, logq, kw = _q_field_kw(n_pix)
    em = _em(n_pix=n_pix, **kw)
    sur = _survey(c_mode=None)                      # legacy per-pixel C
    V0 = np.asarray(_field_missing_curve(_cosmo(), sur, em)[0])
    monkeypatch.setattr(completion_mod, "_fold_occupied_rows",
                        lambda *a: False)
    V1 = np.asarray(_field_missing_curve(_cosmo(), sur, em)[0])
    np.testing.assert_array_equal(V0, V1)


def test_folded_ensemble_member_curves_use_that_member_s_own_rows():
    """Member m's folded budget must equal a from-scratch member-m build.

    The occupied budget is folded from ``field_lss_q`` ITSELF, so the rows
    :func:`_replace_member_q` swaps in are the rows that get summed.  This pins
    that against the hazard a precomputed occupied-sum leaf would carry: a
    member normalizer silently reusing the deterministic (ensemble-mean)
    occupied budget while its numerator carries member m's Q -- exactly what
    ``field_lss_q_fp_empty_sum_members`` guards for the empty half.
    """
    from darksirens.redshift.completion import _replace_member_q

    n_pix = 12
    _, _, occ, logq, kw = _q_field_kw(n_pix)
    logq_m = np.stack([logq + 0.5, logq - 0.5, logq * 0.0])
    f_full = np.linspace(0.2, 1.0, n_pix)
    fp_m = np.asarray(
        build_field_lss_q_fp_empty_sum_members(logq_m, occ, n_pix, f_full))
    q_m, q_empty_m = build_field_lss_q_member_inputs(logq_m, occ, n_pix)
    em = _em(n_pix=n_pix, **_fp_kw(kw, occ, logq, n_pix),
             field_lss_q_members=q_m,
             field_lss_q_empty_sum_members=q_empty_m,
             field_lss_q_fp_empty_sum_members=jnp.asarray(fp_m))

    curves = []
    for m in range(3):
        cat_m = _replace_member_q(em, q_m[m], q_empty_m[m], None, fp_m[m])
        V_m = np.asarray(_field_missing_curve(_cosmo(), _survey(), cat_m)[0])
        # the same member built as a STANDALONE deterministic Q catalog
        q_occ, q_empty = build_field_lss_q_inputs(logq_m[m], occ, n_pix)
        kw_m = dict(field_dN_obs_s=kw["field_dN_obs_s"],
                    field_n_empty=kw["field_n_empty"],
                    field_N_obs_total=kw["field_N_obs_total"],
                    field_occupied_pixels=kw["field_occupied_pixels"],
                    field_lss_q=q_occ, field_lss_q_empty_sum=q_empty)
        em_m = _em(n_pix=n_pix, **_fp_kw(kw_m, occ, logq_m[m], n_pix))
        V_ref = np.asarray(_field_missing_curve(_cosmo(), _survey(), em_m)[0])
        np.testing.assert_allclose(V_m, V_ref, rtol=1e-12, atol=0.0)
        curves.append(V_m)
    # and the members are genuinely distinguishable, so the check has teeth
    assert not np.allclose(curves[0], curves[1])


@pytest.mark.parametrize("n_keep,with_fp", [(1, True), (-3, False)])
@pytest.mark.parametrize("c_mode", [C_MODE_SELECTION_STRUCT, None])
def test_a_misaligned_q_table_is_refused_not_summed(n_keep, with_fp, c_mode):
    """Row-count drift raises, on the folded branch exactly as on the scan.

    The chunked scan caught this only by accident -- its reshape raises -- and
    the fold has no such accident: ``jnp.sum(q_rows, axis=0)`` would total the
    wrong number of rows, and a single Q row would BROADCAST over all of them,
    both returning a silently corrupt (measurably negative, in the one-row
    case) missing budget.  So the check is unconditional and both c_modes are
    pinned here.
    """
    n_pix = 12
    _, _, occ, logq, kw = _q_field_kw(n_pix)
    if with_fp:
        kw = _fp_kw(kw, occ, logq, n_pix)
    q_bad = np.asarray(kw["field_lss_q"])[:n_keep]
    assert q_bad.shape[0] != occ.size
    em = _em(n_pix=n_pix, **dict(kw, field_lss_q=jnp.asarray(q_bad)))
    with pytest.raises(ValueError, match="occupied rows"):
        _field_missing_curve(_cosmo(), _survey(c_mode=c_mode), em)


def test_a_misaligned_f_p_column_is_refused_not_broadcast():
    """Same rule for ``field_f_p_occ``: one entry must not broadcast."""
    n_pix = 12
    _, _, occ, logq, kw = _q_field_kw(n_pix)
    kw = _fp_kw(kw, occ, logq, n_pix)
    f_bad = np.asarray(kw["field_f_p_occ"])[:1]
    em = _em(n_pix=n_pix, **dict(kw, field_f_p_occ=jnp.asarray(f_bad)))
    with pytest.raises(ValueError, match="field_f_p_occ"):
        _field_missing_curve(_cosmo(), _survey(), em)


def test_fp_q_refusals():
    n_pix = 12
    _, field, occ, logq, kw = _q_field_kw(n_pix)
    f_occ = np.linspace(0.2, 1.0, n_pix)[occ]
    # a Q table with f_p but no f_p-weighted empty budget
    em_bad = _em(n_pix=n_pix, **kw,
                 field_f_p_occ=jnp.asarray(f_occ.astype(np.float32)),
                 field_f_p_empty_sum=jnp.asarray(1.0))
    with pytest.raises(ValueError, match="field_lss_q_fp_empty_sum"):
        _field_missing_curve(_cosmo(), _survey(), em_bad)
    # a Q ENSEMBLE with f_p but WITHOUT the per-member twin: member m's
    # normalizer would reuse the deterministic f_p-weighted budget
    em_ens = _em(n_pix=n_pix, **kw,
                 field_f_p_occ=jnp.asarray(f_occ.astype(np.float32)),
                 field_f_p_empty_sum=jnp.asarray(1.0),
                 field_lss_q_fp_empty_sum=build_field_lss_q_fp_empty_sum(
                     logq, occ, n_pix, np.ones(n_pix)),
                 field_lss_q_members=jnp.asarray(
                     np.stack([np.asarray(kw["field_lss_q"])] * 2)))
    with pytest.raises(ValueError, match="fp_empty_sum_members"):
        _field_missing_curve(_cosmo(), _survey(), em_ens)


# ------------------------------------------------- S-3: the unmasked-footprint guard


def _guard_opts(**kw):
    from types import SimpleNamespace
    base = dict(per_pixel_completeness=None, c_mode="selection")
    base.update(kw)
    return SimpleNamespace(**base)


def test_unmasked_footprint_is_refused_and_the_flag_allows_it():
    from darksirens.inference.loaders import attach_selection_fraction_inputs

    # 3072 pixels, 40% empty, 60 galaxies per occupied pixel: Poisson would
    # predict essentially no empty pixels, so the emptiness is a footprint.
    rng = np.random.default_rng(0)
    ngals = rng.poisson(60, size=3072)
    ngals[:1200] = 0
    data = dict(nside=16, ngals_catalog=ngals)

    with pytest.raises(ValueError, match="FOOTPRINT"):
        attach_selection_fraction_inputs(_guard_opts(), dict(data))
    out = attach_selection_fraction_inputs(
        _guard_opts(allow_unmasked_footprint=True), dict(data))
    assert "f_p_map" not in out


def test_sparse_all_sky_catalog_passes_the_guard():
    """Sparsity is not a footprint: empties the mean count predicts are fine."""
    from darksirens.inference.loaders import attach_selection_fraction_inputs

    # lambda must be LARGE enough that the test discriminates: with the old
    # full-sky lambda the threshold was 10*exp(-lambda), which exceeds 1 for any
    # lambda <= ln(10) = 2.303, so a lambda = 0.7 catalog passed trivially and
    # demonstrated nothing. At lambda = 3 the bar is 0.498, so a catalog that is
    # ~5% empty by sparsity passes on its merits.
    rng = np.random.default_rng(1)
    ngals = rng.poisson(3.0, size=3072)
    assert 0.01 < (ngals == 0).mean() < 0.30
    data = dict(nside=16, ngals_catalog=ngals)
    attach_selection_fraction_inputs(_guard_opts(), dict(data))

    # and a per-pixel c_mode is out of scope for the guard entirely
    ngals_fp = rng.poisson(60, size=3072)
    ngals_fp[:1200] = 0
    attach_selection_fraction_inputs(
        _guard_opts(c_mode="per_pixel"), dict(nside=16, ngals_catalog=ngals_fp))


def test_fp_q_ensemble_members_get_their_own_empty_budget():
    """Each member's normalizer must consume ITS OWN Sum_empty f_p Q_p.

    The deterministic budget paired with member m's Q rows is the hazard the
    stratified path already guards one axis over: the numerator would carry
    member m's Q while the normalizer carried the ensemble mean's.  The test
    is that the per-member normalizers DIFFER, and differ in the way the
    per-member budgets do -- a run that reused the deterministic budget would
    still vary (the occupied rows differ) but by a different amount.
    """
    from darksirens.redshift.completion import (
        _replace_member_q, _member_fp_empty_rows)

    n_pix = 12
    _, field, occ, logq, kw = _q_field_kw(n_pix)
    rng = np.random.default_rng(7)
    logq_m = np.stack([logq + 0.5, logq - 0.5, logq * 0.0])
    f_full = np.linspace(0.2, 1.0, n_pix)
    empty = np.setdiff1d(np.arange(n_pix), occ)
    qm_occ, qm_empty = build_field_lss_q_inputs(logq_m[0], occ, n_pix), None
    fp_m = build_field_lss_q_fp_empty_sum_members(logq_m, occ, n_pix, f_full)
    assert np.asarray(fp_m).shape == (3, int(zgrid.size))
    # the three members' budgets are genuinely different
    assert not np.allclose(np.asarray(fp_m)[0], np.asarray(fp_m)[1])

    em = _em(n_pix=n_pix, **kw,
             field_f_p_occ=jnp.asarray(f_full[occ].astype(np.float32)),
             field_f_p_empty_sum=jnp.asarray(float(f_full[empty].sum())),
             field_lss_q_fp_empty_sum=build_field_lss_q_fp_empty_sum(
                 logq, occ, n_pix, f_full),
             field_lss_q_members=jnp.asarray(
                 np.stack([np.asarray(kw["field_lss_q"])] * 3)),
             field_lss_q_fp_empty_sum_members=fp_m)
    rows = _member_fp_empty_rows(em)
    assert rows is not None and np.asarray(rows).shape == (3, int(zgrid.size))

    # the substitution installs member m's budget, not the deterministic one
    for m in range(3):
        cat_m = _replace_member_q(em, em.field_lss_q_members[m],
                                  em.field_lss_q_empty_sum, None,
                                  np.asarray(rows)[m])
        np.testing.assert_allclose(
            np.asarray(cat_m.field_lss_q_fp_empty_sum),
            np.asarray(fp_m)[m], rtol=0, atol=0)
        V, _ = _field_missing_curve(_cosmo(), _survey(), cat_m)
        assert np.all(np.isfinite(np.asarray(V)))

    # and the per-member curves DIFFER, which the deterministic reuse would hide
    Vs = []
    for m in range(3):
        cat_m = _replace_member_q(em, em.field_lss_q_members[m],
                                  em.field_lss_q_empty_sum, None,
                                  np.asarray(rows)[m])
        Vs.append(np.asarray(_field_missing_curve(_cosmo(), _survey(), cat_m)[0]))
    assert not np.allclose(Vs[0], Vs[1])


def test_fp_without_a_q_ensemble_leaves_the_member_rows_none():
    """No f_p, or no ensemble: the accessor returns None so the vmap broadcasts."""
    from darksirens.redshift.completion import _member_fp_empty_rows

    n_pix = 12
    _, field, occ, logq, kw = _q_field_kw(n_pix)
    em_nofp = _em(n_pix=n_pix, **kw)
    assert _member_fp_empty_rows(em_nofp) is None
    # f_p and a DETERMINISTIC Q table, no ensemble: nothing to substitute
    em_fp_det = _em(n_pix=n_pix, **kw,
                    field_f_p_occ=jnp.ones(occ.size, dtype=jnp.float32),
                    field_f_p_empty_sum=jnp.asarray(0.0),
                    field_lss_q_fp_empty_sum=build_field_lss_q_fp_empty_sum(
                        logq, occ, n_pix, np.ones(n_pix)))
    assert _member_fp_empty_rows(em_fp_det) is None


def test_the_guard_fires_on_a_footprint_limited_survey():
    """A footprint-limited survey must trip the guard, and once did not.

    The Poisson reference was built from `ngals.sum()/n_pix`, which counts the
    mask's own zeros, so a bigger footprint hole gave a SMALLER lambda, a LARGER
    `exp(-lambda)` and a HIGHER bar -- the hole raised its own detection
    threshold. For any lambda <= ln(10) the bar exceeded 1 and the guard could
    not fire at any empty fraction.

    The fixture is chosen to DISCRIMINATE the two rules, which a deeper beam
    does not: at 20 gal/pixel over 12% of the sky the full-sky lambda is 2.56,
    bar 10 exp(-lambda) = 0.7734 < empty_frac 0.8802, so the BROKEN rule fires
    too and the test proves nothing. Here, 30% of the sky at Poisson(3) gives

      full-sky lambda 0.8818 -> bar 4.1402  (broken rule: never fires, the bar
                                             is above 1 at any empty fraction)
      occupied lambda 3.0889 -> bar 0.4555  (fixed rule: fires)

    against a measured empty fraction of 0.7145.
    """
    from darksirens.inference.loaders import guard_unmasked_footprint_counts
    from types import SimpleNamespace

    rng = np.random.default_rng(7)
    ngals = np.zeros(3072, dtype=int)
    beam = rng.choice(3072, size=int(0.30 * 3072), replace=False)
    ngals[beam] = rng.poisson(3, size=beam.size)       # shallow, 30% of sky
    empty_frac = float((ngals == 0).mean())
    lam_full = float(ngals.sum() / ngals.size)
    lam_occ = float(ngals[ngals > 0].mean())
    # the broken rule cannot fire on this sky; the fixed one must
    assert empty_frac < max(0.05, 10.0 * np.exp(-lam_full))
    assert empty_frac > max(0.05, 10.0 * np.exp(-lam_occ))

    with pytest.raises(ValueError, match="FOOTPRINT"):
        guard_unmasked_footprint_counts(
            SimpleNamespace(c_mode="selection"), ngals)


def test_the_footprint_guard_reaches_the_multitracer_path():
    """K >= 2 returns before the attach step, so the guard rides the bundle loader.

    Regression for the gap the S-3 PR documented and did not close: a
    multitracer mixture skipped `attach_selection_fraction_inputs` entirely, so
    a footprint-limited tracer under `c_mode=selection` was modelled as
    Cbar-complete off its own footprint with nothing said.  The guard is now
    called per BUNDLE, on that tracer's own full-sky counts before compaction --
    the compact view holds only the pixels this run's events touch, which is not
    a footprint.
    """
    from types import SimpleNamespace

    from darksirens.inference.loaders import guard_unmasked_footprint_counts

    rng = np.random.default_rng(4)
    ngals = rng.poisson(60, size=3072)
    ngals[:1200] = 0
    opts = SimpleNamespace(c_mode="selection")

    with pytest.raises(ValueError, match="FOOTPRINT"):
        guard_unmasked_footprint_counts(opts, ngals, label="tracer_B.h5")
    # the message names WHICH tracer, which is the point of the label for K >= 2
    try:
        guard_unmasked_footprint_counts(opts, ngals, label="tracer_B.h5")
    except ValueError as exc:
        assert "tracer_B.h5" in str(exc)

    guard_unmasked_footprint_counts(
        SimpleNamespace(c_mode="selection", allow_unmasked_footprint=True),
        ngals, label="tracer_B.h5")
    # sparse and per-pixel still pass, by the same core the K=1 path uses
    guard_unmasked_footprint_counts(opts, rng.poisson(0.7, size=3072))
    guard_unmasked_footprint_counts(SimpleNamespace(c_mode="per_pixel"), ngals)

