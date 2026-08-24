"""The completion builders consume the shared batched KDE kernel.

``cli/build_lognormal_completion`` used to loop the scalar eager
``_kde_dndz_obs`` once per occupied pixel (four sites: the radial and gp3d
assemblies, each in both the per-pixel and the aggregate-Cbar branch) — an
O(n_occ) dispatch+sync round trip per build at ~30k occupied DESI pixels —
while ``completion._kde_rows`` already serves the module's ONE compiled
batched kernel (see the hoisting note at ``_batched_kde_dndz_obs``), and the
in-package field builder already routes through it.

These tests pin the substitution: the builder module binds ``_kde_rows`` and
never calls the scalar kernel from Python, and the batched assembly is
numerically inert (ulps at most) against the eager per-pixel loop it replaced.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("h5py")
import h5py
import healpy as hp

import darksirens.cli.build_lognormal_completion as B
import darksirens.redshift.completion as completion
from darksirens.redshift import zgrid

NG = int(zgrid.size)


def _write_survey(path, nside=1, n_occ=4, seed=3):
    """load_survey-schema catalog: the first ``n_occ`` pixels occupied with
    varying counts (mirrors the depth-map build fixture)."""
    npix = hp.nside2npix(int(nside))
    maxg = 9
    zgals = np.full((npix, maxg), 100.0)
    dzgals = np.full((npix, maxg), 0.01)
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for p in range(int(n_occ)):
        n = int(rng.integers(3, maxg + 1))
        zgals[p, :n] = np.sort(rng.uniform(0.2, 0.8, size=n))
        wgals[p, :n] = 1.0
        ngals[p] = n
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = int(nside)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return str(path)


def _coarse_grid(n=12):
    z_s = np.linspace(0.0, float(np.asarray(zgrid)[-1]), int(n))
    edges_s = np.concatenate([[z_s[0]], 0.5 * (z_s[:-1] + z_s[1:]), [z_s[-1]]])
    return z_s, edges_s


def _spies(monkeypatch):
    """Count batched-vs-scalar KDE entries into the builder module.

    ``raising=False`` keeps both patches valid on either side of the fix:
    pre-fix the module has no ``_kde_rows`` to intercept (the counter then
    stays 0 and the assertion fails) and the scalar spy counts its eager loop.
    """
    calls = {"rows": 0, "scalar": 0}
    real_rows = completion._kde_rows

    def rows_spy(pix_idx, zgals, wgals, ngals, batch_size):
        calls["rows"] += 1
        return real_rows(pix_idx, zgals, wgals, ngals, batch_size)

    def scalar_spy(*a, **k):
        calls["scalar"] += 1
        return completion._kde_dndz_obs(*a, **k)

    monkeypatch.setattr(B, "_kde_rows", rows_spy, raising=False)
    monkeypatch.setattr(B, "_kde_dndz_obs", scalar_spy, raising=False)
    return calls


def test_builder_binds_the_batched_rows_kernel():
    """The finding itself: the module consumes ``completion._kde_rows`` and no
    longer keeps the scalar kernel around to loop over."""
    assert B._kde_rows is completion._kde_rows
    assert not hasattr(B, "_kde_dndz_obs")


@pytest.mark.parametrize("c_mode", ["per_pixel", "aggregate"])
def test_gp3d_assembly_routes_the_kde_through_kde_rows(tmp_path, monkeypatch,
                                                       c_mode):
    """ONE ``_kde_rows`` call per assembly; ZERO Python-level scalar calls."""
    cat = _write_survey(tmp_path / "survey.h5", nside=2, n_occ=5)
    z_s, edges_s = _coarse_grid()
    cosmo, survey = B._fiducial_cosmo_survey()
    calls = _spies(monkeypatch)

    a = B._assemble_gp3d_survey(cat, cosmo=cosmo, survey=survey,
                                z_s=z_s, edges_s=edges_s, c_mode=c_mode)

    assert calls["scalar"] == 0, "the per-pixel eager KDE loop is back"
    assert calls["rows"] == 1
    assert np.all(np.isfinite(a.base_vox)) and float(np.max(a.base_vox)) > 0.0


@pytest.mark.parametrize("c_mode", ["per_pixel", "aggregate"])
def test_radial_build_routes_the_kde_through_kde_rows(tmp_path, monkeypatch,
                                                      c_mode):
    """Same contract for the radial builder (both completeness branches)."""
    cat = _write_survey(tmp_path / "survey.h5")
    calls = _spies(monkeypatch)

    logq, _members, _diag = B.build_completion(
        cat, mode="radial", n_members=0, maxiter=200, workers=1,
        c_mode=c_mode, budget_renorm=False)

    assert calls["scalar"] == 0, "the per-pixel eager KDE loop is back"
    assert calls["rows"] == 1
    assert logq.shape == (hp.nside2npix(1), NG)
    assert np.all(np.isfinite(logq))


@pytest.mark.parametrize("c_mode", ["per_pixel", "aggregate"])
def test_gp3d_assembly_is_numerically_inert_vs_the_eager_loop(tmp_path,
                                                              monkeypatch,
                                                              c_mode):
    """Swapping the batched kernel back for the scalar per-pixel loop moves the
    assembly by ulps at most (jit-fusion summation order — the verifier
    measured max rel ~1e-15): the substitution changes dispatch, not the
    science."""
    cat = _write_survey(tmp_path / "survey.h5", nside=2, n_occ=5)
    z_s, edges_s = _coarse_grid()
    cosmo, survey = B._fiducial_cosmo_survey()

    a_batched = B._assemble_gp3d_survey(cat, cosmo=cosmo, survey=survey,
                                        z_s=z_s, edges_s=edges_s, c_mode=c_mode)

    def eager_rows(pix_idx, zgals, wgals, ngals, batch_size):
        idx = np.asarray(pix_idx).reshape(-1)
        out = np.empty((idx.size, NG), dtype=np.float64)
        for i, r in enumerate(idx):
            out[i] = np.asarray(
                completion._kde_dndz_obs(int(r), zgals, wgals, ngals),
                dtype=float)
        return out

    monkeypatch.setattr(B, "_kde_rows", eager_rows, raising=False)
    a_eager = B._assemble_gp3d_survey(cat, cosmo=cosmo, survey=survey,
                                      z_s=z_s, edges_s=edges_s, c_mode=c_mode)

    np.testing.assert_allclose(a_eager.base_vox, a_batched.base_vox,
                               rtol=1e-12)
    np.testing.assert_allclose(a_eager.w_budget, a_batched.w_budget,
                               rtol=1e-12)
    np.testing.assert_array_equal(a_eager.N_obs_vox, a_batched.N_obs_vox)
