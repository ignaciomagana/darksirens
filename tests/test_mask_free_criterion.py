"""The mask-free criterion is anchored to the CATALOG, not to zero.

``measure_mask_free`` decides whether a Q table may be paired with
``--per_pixel_completeness``.  It asks two questions of the finished artifact:

* off-footprint (``f_p == 0``) pixels must carry ``logQ == 0`` EXACTLY.  That
  is the guard against the v1 failure -- an f_p-shaped Q with on/off mean
  contrast 1.62 vs 0.05, which put H0 at 41.24 against a truth of 67.74 -- and
  it is unchanged;
* on the covered sky, ``corr(Q, f_p)`` must match ``corr(N / f_p, f_p)``
  measured from the catalog in a band around that slice.

WHY THE SECOND ONE MOVED OFF ZERO.  The covered DESI sky genuinely correlates
density with depth: measured on the real pixelated catalog with the nside-128
depth map degraded to 64, ``corr(N/f_p, f_p)`` = +0.112 / +0.174 / +0.237 in
z in [0.10, 0.15] / [0.20, 0.25] / [0.27, 0.30], at 68-192 galaxies per covered
pixel (so not shrinkage noise).  A FAITHFUL Q reproduces that; the v3 build's
interior profile does, reaching +0.130 at z-slice 374 -- and the old
zero-anchored tolerance of 0.10 therefore REFUSED it for being right.  What the
zero anchor conflated is pinned apart here: mask SHAPE fails, catalog structure
passes, and off-footprint leakage fails either way.

The fixture below is built to make that distinction sharp.  A catalog whose
density were a PERFECT function of depth could not tell the two apart at all --
Pearson correlation is scale-free, so ``Q ~ f_p`` and ``N/f_p ~ f_p`` would both
read +1.  Real surveys are not like that (DESI's coupling is +0.11 to +0.24),
so the fixture puts the catalog at ~+0.2 with independent density scatter on
top, exactly where a mask-shaped Q (~+1.0) is separable from a faithful one.
"""
import numpy as np
import pytest

pytest.importorskip("healpy")
pytest.importorskip("h5py")

from test_completion_depth_map_build import (            # noqa: E402
    _write_depth_map,
    _write_survey,
)

NSIDE_T = 4                       # 192 pixels: enough for a correlation to mean something
NPIX_T = 12 * NSIDE_T ** 2
ZLO, ZHI = 0.01, 1.0
#: Target density-depth coupling, in the DESI ballpark (+0.11 to +0.24).  Built
#: as ``dens = 1 + A z(f_p) + B eps`` with independent eps, so the population
#: correlation is ``A / sqrt(A^2 + B^2)`` = 0.222.
_A_DEPTH, _B_SCATTER = 0.05, 0.22


def _fixture_f_p(seed=11):
    """Depths in [0.3, 1.0] on 2/3 of the sky, holes on the rest."""
    rng = np.random.default_rng(seed)
    fp = rng.uniform(0.3, 1.0, size=NPIX_T)
    fp[rng.permutation(NPIX_T)[: NPIX_T // 3]] = 0.0
    return fp


def _write_depth_map_nside(path, f_p, nside):
    """``build_mth_map`` schema at an arbitrary nside (the shared helper is
    pinned to the 12-pixel fixture)."""
    import h5py

    f_p = np.asarray(f_p, dtype=float)
    counts = np.where(f_p > 0.0, 100.0, 0.0)
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = int(nside)
        f.attrs["ordering"] = "RING"
        f.create_dataset("counts", data=counts)
        f.create_dataset("masked_frac",
                         data=np.where(counts > 0.0, 1.0 - f_p, np.nan))
    return str(path)


def _write_correlated_survey(path, f_p, seed=5):
    """Catalog whose covered-sky density correlates with depth at ~+0.22.

    ``N_p = round(6000 f_p dens_p)``: the counts carry the depth loss (the
    ``f_p`` factor, which ``N/f_p`` divides out) AND a true density-depth
    coupling on top of it, which is what ``corr(N/f_p, f_p)`` sees.

    Redshifts are laid out DETERMINISTICALLY across ``[ZLO, ZHI]`` (a linspace,
    not a draw), so every z band holds the same fixed fraction of each pixel's
    galaxies and the anchor is the same at every slice.  A random draw would
    make each narrow band a binomial thinning, attenuating ``corr_data``
    slice-to-slice by a noise level these tests are not about.
    """
    import h5py

    rng = np.random.default_rng(seed)
    on = f_p > 0.0
    zfp = np.zeros_like(f_p)
    zfp[on] = (f_p[on] - f_p[on].mean()) / f_p[on].std()
    dens = 1.0 + _A_DEPTH * zfp + _B_SCATTER * rng.standard_normal(NPIX_T)
    dens = np.clip(dens, 0.2, None)
    n_want = np.where(on, np.round(6000.0 * f_p * dens), 0).astype(int)

    maxg = int(n_want.max())
    zgals = np.full((NPIX_T, maxg), 100.0)
    dzgals = np.full((NPIX_T, maxg), 0.01)
    wgals = np.zeros((NPIX_T, maxg))
    for p in range(NPIX_T):
        n = int(n_want[p])
        if n:
            zgals[p, :n] = np.linspace(ZLO, ZHI, n)
            wgals[p, :n] = 1.0
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = NSIDE_T
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=n_want.astype(np.int32))
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return str(path), n_want


def _flat_logq(n_grid, per_pixel):
    """(NPIX_T, n_grid) logQ, constant in z, from a per-pixel value."""
    return np.repeat(np.asarray(per_pixel, dtype=float)[:, None], n_grid, axis=1)


@pytest.fixture
def catalog_and_map(tmp_path):
    """(catalog, depth map, f_p, per-pixel counts) for the fixture sky."""
    fp = _fixture_f_p()
    cat, n_want = _write_correlated_survey(tmp_path / "survey.h5", fp)
    dmap = _write_depth_map_nside(tmp_path / "depth.h5", fp, NSIDE_T)
    return cat, dmap, fp, n_want


def _faithful_q(fp, n_want):
    """Q = the catalog's own depth-corrected density: the faithful table."""
    on = fp > 0.0
    q = np.ones(NPIX_T)
    q[on] = n_want[on] / fp[on]
    q[on] /= q[on].mean()
    return q


# ---------------------------------------------------------------------------
# the anchor exists, and it is not zero
# ---------------------------------------------------------------------------

def test_the_catalog_anchor_is_measured_and_nonzero(catalog_and_map):
    """corr_data is read off the catalog, per slice -- not assumed to be 0.

    Without this the whole criterion collapses back to the zero-anchored one.
    """
    from darksirens.cli.build_lognormal_completion import measure_mask_free

    cat, dmap, fp, n_want = catalog_and_map
    zg = np.linspace(ZLO, ZHI, 60)

    # a constant-Q table has no varying slice at all, so nothing is compared
    res = measure_mask_free(_flat_logq(60, np.zeros(NPIX_T)), cat, dmap,
                            zgrid_nodes=zg)
    assert res["corr_q"] == {}

    res = measure_mask_free(_flat_logq(60, np.log(_faithful_q(fp, n_want))),
                            cat, dmap, zgrid_nodes=zg)
    assert res["corr_data"], "no catalog anchor was measured"
    # the fixture sky is coupled at ~+0.22, in the DESI ballpark -- comfortably
    # above the 0.10 the zero-anchored rule allowed
    assert all(0.10 < v < 0.40 for v in res["corr_data"].values()), res["corr_data"]
    assert res["band_half_nodes"] >= 1


# ---------------------------------------------------------------------------
# the real hazard still fails
# ---------------------------------------------------------------------------

def test_a_mask_shaped_q_fails_even_though_the_catalog_correlates(catalog_and_map):
    """Q carrying the mask's own shape: corr_Q ~ +1 >> corr_data ~ +0.22.

    This is the v1 failure mode expressed on the covered sky alone -- the
    off-footprint rows are left at bit-zero, so ONLY the correlation criterion
    can catch it.  If anchoring to the data had made the test permissive, this
    is what would have slipped through.
    """
    from darksirens.cli.build_lognormal_completion import measure_mask_free

    cat, dmap, fp, _n = catalog_and_map
    zg = np.linspace(ZLO, ZHI, 60)
    # the measured v1 contrast (mean 1.62 on-footprint against 0.05 off),
    # mapped onto f_p so Q IS the footprint's shape on the covered sky
    q = np.where(fp > 0.0, 0.05 + 1.57 * (fp / fp.max()), 1.0)
    res = measure_mask_free(_flat_logq(60, np.log(q)), cat, dmap, zgrid_nodes=zg)

    assert res["ok"] is False
    assert res["off_footprint"]["max_abs"] == 0.0     # not what caught it
    assert res["worst_abs_delta"] > 0.10
    assert res["worst_abs_corr_q"] > 0.9
    assert max(abs(v) for v in res["corr_data"].values()) < 0.4


def test_a_q_carrying_exactly_the_catalogs_correlation_passes(catalog_and_map):
    """The faithful table: Q built from the catalog's own N/f_p.

    Its correlation with f_p IS the catalog's, so the delta is ~0 and the stamp
    is earned.  Under the OLD zero-anchored rule this same table reads |corr|
    above 0.10 and is refused -- which is exactly what happened to the v3
    build's z-slice 374 (+0.130).
    """
    from darksirens.cli.build_lognormal_completion import _corr, measure_mask_free

    cat, dmap, fp, n_want = catalog_and_map
    zg = np.linspace(ZLO, ZHI, 60)
    q = _faithful_q(fp, n_want)
    res = measure_mask_free(_flat_logq(60, np.log(q)), cat, dmap, zgrid_nodes=zg)

    assert res["ok"] is True
    assert res["worst_abs_delta"] <= 0.10
    # ... and the zero anchor would have refused it
    on = fp > 0.0
    assert _corr(q[on], fp[on]) > 0.10
    assert res["worst_abs_corr_q"] > 0.10


def test_off_footprint_leakage_fails_whatever_the_correlations_do(catalog_and_map):
    """The unchanged half of the criterion, and it must dominate.

    Q here carries EXACTLY the catalog's correlation on the covered sky (delta
    ~ 0) plus one off-footprint pixel at logQ = 1e-3.  Those pixels hold no
    counts, so any structure there came from the prior or the budget
    renormalization -- the artifact is not mask-free and must not be stamped.
    """
    from darksirens.cli.build_lognormal_completion import measure_mask_free

    cat, dmap, fp, n_want = catalog_and_map
    zg = np.linspace(ZLO, ZHI, 60)
    lq = _flat_logq(60, np.log(_faithful_q(fp, n_want)))
    lq[np.nonzero(fp <= 0.0)[0][0], 3] = 1.0e-3          # 1000x the tolerance

    res = measure_mask_free(lq, cat, dmap, zgrid_nodes=zg)
    assert res["worst_abs_delta"] <= 0.10          # the correlations are innocent
    assert res["off_footprint"]["max_abs"] == pytest.approx(1.0e-3)
    assert res["ok"] is False


# ---------------------------------------------------------------------------
# plumbing: the builder and the measure script must share ONE implementation
# ---------------------------------------------------------------------------

def test_the_stamp_and_the_profiles_go_into_the_artifact(tmp_path):
    """Build a real (tiny) table and read both profiles back off the file.

    The measure script recomputes from the artifact with the SAME function, so a
    stamp that disagreed with a recomputation would mean the FILE changed -- not
    that two implementations drifted, which is what the shared function removes.
    """
    import json

    import h5py

    from darksirens.cli.build_lognormal_completion import main, measure_mask_free

    cat = _write_survey(tmp_path / "survey.h5")
    dmap = _write_depth_map(tmp_path / "depth.h5",
                            np.array([1.0] * 4 + [0.9, 0.9, 0.4, 0.4]
                                     + [0.0] * 4))
    out = tmp_path / "q.h5"
    main(["--catalog", cat, "--out", str(out), "--n-members", "0",
          "--maxiter", "200", "--mode", "radial", "--c-mode", "aggregate",
          "--allow-unconverged", "--depth-map", dmap])

    with h5py.File(str(out), "r") as f:
        g = f["lss_completion"]
        lq = np.asarray(g["logq_map"][...], dtype=float)
        zg = np.asarray(g["zgrid"][...], dtype=float)
        stamped = bool(g.attrs["f_p_aware"])
        diag = json.loads(g.attrs["diagnostics"])

    mf = diag["mask_free"]
    assert set(mf["corr_q"]) == set(mf["corr_data"]) == set(mf["corr_delta"])
    assert mf["corr_q"], "no slice was compared at all"
    assert mf["tolerances"]["corr_delta"] == pytest.approx(0.10)
    assert mf["ok"] is stamped

    # the recomputation route measure_maskfree_v2.py takes
    res = measure_mask_free(lq, cat, dmap, zgrid_nodes=zg)
    assert res["ok"] is stamped
    assert res["corr_q"] == mf["corr_q"]
    assert res["corr_data"] == mf["corr_data"]


def test_a_row_count_mismatch_is_skipped_not_crashed(catalog_and_map):
    """Compact-indexed tables cannot be checked; they are refused, not raised."""
    from darksirens.cli.build_lognormal_completion import measure_mask_free

    cat, dmap, _fp, _n = catalog_and_map
    res = measure_mask_free(np.zeros((3, 60)), cat, dmap,
                            zgrid_nodes=np.linspace(ZLO, ZHI, 60))
    assert res["skipped"] is True and res["ok"] is False


def test_the_band_is_a_ninth_of_the_fitted_range(catalog_and_map):
    """Band width follows the FITTED block, not the whole grid.

    Above a --q-support-depth cut every row is bit-zero, so the fitted block is
    read off the table itself; the bands must then tile the catalog's SUPPORT
    rather than a mostly-empty grid, or every slice would be anchored against
    counts from redshifts Q never modelled.
    """
    from darksirens.cli.build_lognormal_completion import measure_mask_free

    cat, dmap, fp, n_want = catalog_and_map
    zg = np.linspace(ZLO, ZHI, 90)
    lq = _flat_logq(90, np.log(_faithful_q(fp, n_want)))
    assert measure_mask_free(lq, cat, dmap, zgrid_nodes=zg)["band_half_nodes"] == 5

    lq[:, 54:] = 0.0                                   # a support cut at node 54
    res = measure_mask_free(lq, cat, dmap, zgrid_nodes=zg)
    assert res["n_fit_cols"] == 54
    assert res["band_half_nodes"] == 3                 # 54 / (2 * 9)
