"""Per-z mean-one Q budget renormalization (S0b).

The GW likelihood forms the missing density as ``dN_miss = (1 - C) dN_exp Q``,
so a per-z monopole in ``Q`` RESCALES the total missing budget instead of only
redistributing it — and the Laplace posterior-mean ``E[Q]`` carries exactly
such a monopole because the posterior variance is largest where data are
sparse (spatially varying Jensen bias; measured +55% budget inflation for
radial tables).  The budget is C's and n0's job; ``Q`` only places it.

These tests pin the new convention end-to-end:

* :func:`renormalize_q_mean_one` enforces ``sum_p w_p Q_p == sum_p w_p`` per
  z-bin to float precision (MAP table and each member independently), leaving
  zero-budget bins untouched;
* the radial and gp3d builders apply it by default over the fitted footprint,
  so the total missing budget with Q equals the homogeneous budget per z-bin
  at the build fiducial — and a ``budget_renorm=False`` build genuinely
  violates that identity (the renormalization does real work);
* the HDF5 writer stamps the boolean + the removed monopole curve
  (fail-closed: ``True`` without the curve, or a non-finite curve, refuses to
  save), and the loader warns loudly on legacy tables without the stamp.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("jax")
import jax.numpy as jnp

from darksirens.redshift import zgrid
from darksirens.redshift.lognormal_completion import (
    renormalize_q_mean_one,
    save_lss_completion_hdf5,
    load_lss_completion_hdf5,
)

NG = int(zgrid.size)


# ------------------------------------------------------------
# The renormalization operator itself
# ------------------------------------------------------------

def test_renormalize_exact_mean_one_map_and_members():
    rng = np.random.default_rng(0)
    n_rows, n_grid = 5, 32
    logq = 0.7 * rng.standard_normal((n_rows, n_grid))
    w = rng.uniform(0.1, 3.0, size=(n_rows, n_grid))
    w[:, 3] = 0.0                      # a bin with no missing budget anywhere
    out, mono = renormalize_q_mean_one(logq, w)
    assert out.shape == logq.shape and mono.shape == (n_grid,)
    den = np.sum(w, axis=0)
    ok = den > 0
    # the weighted mean of Q is one wherever the weights carry budget ...
    np.testing.assert_allclose(
        np.sum(w * np.exp(out), axis=0)[ok] / den[ok], 1.0, rtol=1e-12)
    # ... and the zero-budget bin is left untouched (guard path)
    assert mono[3] == 0.0
    np.testing.assert_array_equal(out[:, 3], logq[:, 3])

    # member cube: each member individually mean-one (its own monopole) —
    # placement uncertainty survives, budget uncertainty does not
    cube = 0.5 * rng.standard_normal((4, n_rows, n_grid))
    outc, monoc = renormalize_q_mean_one(cube, w)
    assert outc.shape == cube.shape and monoc.shape == (4, n_grid)
    assert not np.allclose(monoc[0], monoc[1])   # genuinely per-member
    for m in range(4):
        np.testing.assert_allclose(
            np.sum(w * np.exp(outc[m]), axis=0)[ok] / den[ok], 1.0, rtol=1e-12)


def test_renormalize_rejects_bad_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        renormalize_q_mean_one(np.zeros((2, 4)), np.full((2, 4), -1.0))
    with pytest.raises(ValueError, match="weights shape"):
        renormalize_q_mean_one(np.zeros((2, 4)), np.zeros((3, 4)))
    with pytest.raises(ValueError, match="logq must be"):
        renormalize_q_mean_one(np.zeros(4), np.zeros((1, 4)))


# ------------------------------------------------------------
# End-to-end: builder output satisfies the budget identity
# ------------------------------------------------------------

def _write_noisy_survey(path, nside=2, n_occ=4, seed=3):
    """load_survey-schema catalog with NOISY per-pixel counts (different pixel
    occupations and redshifts), so the fitted Q genuinely varies across the
    footprint and the raw E[Q] monopole is nonzero."""
    import h5py
    import healpy as hp
    npix = hp.nside2npix(int(nside))
    maxg = 12
    zgals = np.full((npix, maxg), 100.0)
    dzgals = np.full((npix, maxg), 0.01)
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for p in range(int(n_occ)):
        n = int(rng.integers(3, maxg + 1))
        zgals[p, :n] = rng.uniform(0.2, 0.8, size=n)
        wgals[p, :n] = 1.0
        ngals[p] = n
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = int(nside)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return npix


def _budget_weights(cat_path):
    """Recompute the builder's fitted-footprint budget weights
    ``w_p(z) = (1 - C_p(z)) * dN_exp(z)`` at the build-time fiducial, step for
    step (same EMCatalog construction, same fine-grid completeness)."""
    import healpy as hp
    from darksirens.catalogs.io import load_survey
    from darksirens.core.types import EMCatalog
    from darksirens.redshift.completion import _precompute_grids, _kde_dndz_obs
    from darksirens.cli.build_lognormal_completion import _fiducial_cosmo_survey

    nside, ngals, zgals, dzgals, wgals, _z_depth = load_survey(cat_path)
    apix = float(hp.nside2pixarea(int(nside)))
    em = EMCatalog(
        apix=apix, zgals=jnp.asarray(zgals), dzgals=jnp.asarray(dzgals),
        wgals=jnp.asarray(wgals), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((1, NG)), dN_obs_kde=None,
        pixel_to_cache_idx=None,
    )
    cosmo, survey = _fiducial_cosmo_survey()
    grids = _precompute_grids(cosmo, survey, em)
    dN_exp = np.asarray(grids.dN_exp, dtype=float)
    smooth = np.asarray(grids.dN_exp_smooth, dtype=float)
    safe = np.where(smooth > 0.0, smooth, 1.0)
    occ = np.nonzero(np.asarray(ngals).astype(int) > 0)[0]
    w = np.empty((occ.size, NG), dtype=float)
    for i, r in enumerate(occ):
        kde = np.asarray(_kde_dndz_obs(int(r), em.zgals, ngals=em.ngals),
                         dtype=float)
        C = np.clip(kde / safe, 0.0, 1.0)
        w[i] = (1.0 - C) * dN_exp
    return occ, w


def test_radial_build_budget_matches_homogeneous_per_z(tmp_path):
    """The spec's budget-invariance gate: sum_p dN_miss(Q) == sum_p
    dN_miss(homog) per z-bin to float tolerance, for the MAP table AND for
    each member — and a budget_renorm=False build measurably violates it."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = str(tmp_path / "survey.h5")
    npix = _write_noisy_survey(cat)
    occ, w = _budget_weights(cat)
    homog = np.sum(w, axis=0)                      # sum_p (1 - C_p) dN_exp

    logq_map, logq_members, diag = build_completion(
        cat, mode="radial", n_members=2, seed=5, maxiter=60)
    assert diag["budget_renormalized"] is True
    mono = np.asarray(diag["budget_monopole_logq"], dtype=float)
    assert mono.shape == (NG,) and np.all(np.isfinite(mono))

    with_q = np.sum(w * np.exp(logq_map[occ]), axis=0)
    np.testing.assert_allclose(with_q, homog, rtol=1e-10)
    for m in range(logq_members.shape[0]):
        with_qm = np.sum(w * np.exp(logq_members[m][occ]), axis=0)
        np.testing.assert_allclose(with_qm, homog, rtol=1e-10)

    # empty (unfitted) rows stay exactly homogeneous — the footprint's
    # monopole must not leak onto the rest of the sky
    empty = np.setdiff1d(np.arange(npix), occ)
    assert np.all(logq_map[empty] == 0.0)
    assert np.all(logq_members[:, empty] == 0.0)

    # counterfactual: without the renormalization the identity fails (the
    # Laplace E[Q] monopole is real for these noisy counts)
    raw_map, _, raw_diag = build_completion(
        cat, mode="radial", n_members=0, maxiter=60, budget_renorm=False)
    assert raw_diag["budget_renormalized"] is False
    assert "budget_monopole_logq" not in raw_diag
    okbins = homog > 0
    raw_with_q = np.sum(w * np.exp(raw_map[occ]), axis=0)
    assert float(np.max(np.abs(raw_with_q[okbins] / homog[okbins] - 1.0))) > 1e-3


def test_gp3d_build_budget_matches_homogeneous_per_z(tmp_path):
    """Same identity for the gp3d builder, over its fitted footprint (the
    occupied rows; far pixels already read exactly Q = 1)."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = str(tmp_path / "survey.h5")
    _write_noisy_survey(cat)
    occ, w = _budget_weights(cat)
    homog = np.sum(w, axis=0)

    logq_map, logq_members, diag = build_completion(
        cat, mode="gp3d", n_members=2, seed=5, gp3d_nz_solve=16,
        gp3d_pix_chunk=8,
        lss_corr_length_mpc=3000.0)  # S0c: resolve the inducing grid (hard gate)
    assert diag["budget_renormalized"] is True
    with_q = np.sum(w * np.exp(logq_map[occ]), axis=0)
    np.testing.assert_allclose(with_q, homog, rtol=1e-10)
    for m in range(logq_members.shape[0]):
        with_qm = np.sum(w * np.exp(logq_members[m][occ]), axis=0)
        np.testing.assert_allclose(with_qm, homog, rtol=1e-10)


# ------------------------------------------------------------
# HDF5 provenance: stamped attrs, fail-closed writer, legacy warning
# ------------------------------------------------------------

def test_hdf5_budget_attrs_roundtrip_quietly(tmp_path):
    path = str(tmp_path / "q.h5")
    mono = 0.01 * np.ones(NG)
    save_lss_completion_hdf5(
        path, logq_map=np.zeros((3, NG)), zgrid=np.asarray(zgrid),
        budget_renormalized=True, budget_monopole_logq=mono)
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # a stamped table must load quietly
        d = load_lss_completion_hdf5(path)
    assert d["budget_renormalized"] is True
    np.testing.assert_allclose(d["budget_monopole_logq"], mono)


def test_stamped_false_is_deliberate_and_quiet(tmp_path):
    path = str(tmp_path / "q_off.h5")
    save_lss_completion_hdf5(
        path, logq_map=np.zeros((2, NG)), zgrid=np.asarray(zgrid),
        budget_renormalized=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        d = load_lss_completion_hdf5(path)
    assert d["budget_renormalized"] is False
    assert d["budget_monopole_logq"] is None


def test_writer_fail_closed_on_missing_or_nonfinite_monopole(tmp_path):
    # True without the removed curve: unauditable — refuse
    with pytest.raises(ValueError, match="budget_monopole_logq"):
        save_lss_completion_hdf5(
            str(tmp_path / "a.h5"), logq_map=np.zeros((2, NG)),
            budget_renormalized=True)
    # non-finite curve: the renormalization was fed a poisoned table — refuse
    bad = np.zeros(NG)
    bad[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        save_lss_completion_hdf5(
            str(tmp_path / "b.h5"), logq_map=np.zeros((2, NG)),
            budget_renormalized=True, budget_monopole_logq=bad)
    assert not (tmp_path / "a.h5").exists()
    assert not (tmp_path / "b.h5").exists()


def test_legacy_table_without_stamp_warns_loudly(tmp_path):
    """Spec: the loader tolerates legacy tables (no attr -> not renormalized)
    but emits a loud warning in the existing RuntimeWarning style."""
    path = str(tmp_path / "legacy.h5")
    save_lss_completion_hdf5(
        path, logq_map=np.zeros((2, NG)), zgrid=np.asarray(zgrid))
    with pytest.warns(RuntimeWarning, match="budget_renormalized"):
        d = load_lss_completion_hdf5(path)
    assert d["budget_renormalized"] is None
    assert d["budget_monopole_logq"] is None
    # the table itself still loads (tolerate, never reject)
    assert d["logq_map"].shape == (2, NG)
