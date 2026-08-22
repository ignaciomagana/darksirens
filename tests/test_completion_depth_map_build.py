"""``--depth-map`` in the lognormal-completion builder: the f_p-aware Q table.

The EMPTY-row dedup.  Empty on-footprint pixels are deduplicated before the
L-BFGS-B solve because their inputs are identical -- but with ``--depth-map``
active each row's base is ``C_u = f_p Cbar``, and the ``N_obs = 0`` MAP moves
strongly with ``f_p``.  Collapsing them all into one group hands every empty
pixel one representative's ``logQ``, and those rows are precisely the
missing-host budget for unobserved sky (``field_lss_q_empty_sum`` /
``field_lss_q_fp_empty_sum``).
"""
import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("h5py")

NSIDE = 1
NPIX = 12 * NSIDE ** 2


def _write_survey(path, n_occ=4, seed=3):
    """load_survey-schema catalog: the first ``n_occ`` pixels occupied."""
    import h5py

    maxg = 12
    zgals = np.full((NPIX, maxg), 100.0)
    dzgals = np.full((NPIX, maxg), 0.01)
    wgals = np.zeros((NPIX, maxg))
    ngals = np.zeros(NPIX, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for p in range(int(n_occ)):
        n = int(rng.integers(3, maxg + 1))
        zgals[p, :n] = np.sort(rng.uniform(0.2, 0.8, size=n))
        wgals[p, :n] = 1.0
        ngals[p] = n
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = NSIDE
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return str(path)


def _write_depth_map(path, f_p):
    """build_mth_map-schema depth map giving exactly ``f_p = 1 - masked_frac``.

    Uncovered pixels (f_p = 0) carry counts = 0 and a NaN masked_frac, which is
    what the native builder writes and what ``load_selection_fraction`` reads as
    off-footprint.
    """
    import h5py

    f_p = np.asarray(f_p, dtype=float)
    counts = np.where(f_p > 0.0, 100.0, 0.0)
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = NSIDE
        f.attrs["ordering"] = "RING"
        f.create_dataset("counts", data=counts)
        f.create_dataset("masked_frac",
                         data=np.where(counts > 0.0, 1.0 - f_p, np.nan))
    return str(path)


# ---------------------------------------------------------------------------
# the empty-row dedup must key on f_p
# ---------------------------------------------------------------------------

def test_empty_covered_pixels_with_different_f_p_get_different_logq(tmp_path):
    """f_p 0.9 and f_p 0.4 empty pixels are NOT the same solve.

    They were: the dedup put every empty fitted row in group 1 and broadcast one
    representative's logQ over all of them, so the entire unobserved-sky budget
    took the f_p of whichever pixel happened to be first.  On this fixture the
    collapsed table gave rows 4-7 max|logQ| = 0.1666 alike; keyed on f_p the
    0.4 rows solve to 0.0795, a max|dlogQ| of 0.087 (larger on the production
    mock: 0.93 between f_p 1.0 and 0.3).
    """
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = _write_survey(tmp_path / "survey.h5")
    # occupied 0-3 fully unmasked; empty-but-COVERED 4,5 at 0.9 and 6,7 at 0.4;
    # 8-11 off footprint (f_p = 0, never fitted).
    f_p = np.array([1.0] * 4 + [0.9, 0.9, 0.4, 0.4] + [0.0] * 4)
    dmap = _write_depth_map(tmp_path / "depth.h5", f_p)

    logq, _members, diag = build_completion(
        cat, mode="radial", n_members=0, maxiter=200, workers=1,
        c_mode="aggregate", depth_map=dmap, budget_renorm=False)

    # pixels sharing an f_p still share a solve, exactly (that is the dedup)
    np.testing.assert_array_equal(logq[4], logq[5])
    np.testing.assert_array_equal(logq[6], logq[7])
    # ... and pixels with a different f_p do not
    assert not np.array_equal(logq[4], logq[6])
    assert float(np.abs(logq[4] - logq[6]).max()) > 1e-3
    # 4 occupied solves + one representative per unique empty f_p (0.9, 0.4)
    assert diag["n_solved_rows"] == 6
    assert diag["n_broadcast_duplicate_rows"] == 2
    # off-footprint rows are not fitted at all: mask-freedom needs logQ == 0
    assert np.all(logq[8:] == 0.0)


def test_without_a_depth_map_every_empty_pixel_still_shares_one_solve(tmp_path):
    """No f_p: the rows really are interchangeable, so the dedup must not split.

    The f_p keying is conditional; with ``f_p_map=None`` the aggregate build is
    unchanged, one representative for all eight empty pixels.
    """
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = _write_survey(tmp_path / "survey.h5")
    logq, _members, diag = build_completion(
        cat, mode="radial", n_members=0, maxiter=200, workers=1,
        c_mode="aggregate", budget_renorm=False)
    assert diag["n_solved_rows"] == 5
    assert diag["n_broadcast_duplicate_rows"] == 7
    for p in range(5, NPIX):
        np.testing.assert_array_equal(logq[p], logq[4])
