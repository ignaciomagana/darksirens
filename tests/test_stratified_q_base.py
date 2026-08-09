"""Stratified Q-builder base (PR-A): per-pixel C_sel rows + provenance.

Pinned: (1) degenerate strata (identical thetas) build a radial table
identical to the pooled single-stratum base; (2) genuinely different strata
produce per-stratum-different logQ structure; (3) the stamps carry
n_strata, per-stratum thetas, and the stratum-map sha256; (4) the inference
firewall accepts only a matching (strata, map-hash) pair and is fatal in
every mismatch direction, including stratified-run x unstamped-table.
"""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.redshift.grid import zgrid  # noqa: E402
from darksirens.utils.cosmology import distance_modulus  # noqa: E402

NSIDE = 2
NPIX = 12 * NSIDE * NSIDE


def _write_catalog(path, rng, n=4000):
    """Tiny pixelated survey: galaxies over half the pixels, mags truncated."""
    import healpy as hp

    pix = rng.integers(0, NPIX // 2, size=n)          # half the sky occupied
    z = rng.uniform(0.05, 0.35, size=n)
    M = rng.normal(-21.0, 0.8, size=n)
    m = M + np.asarray(distance_modulus(jnp.asarray(z), 70.0))
    keep = m <= 21.0
    pix, z, m = pix[keep], z[keep], m[keep]
    counts = np.bincount(pix, minlength=NPIX)
    maxg = int(counts.max())
    zg = np.full((NPIX, maxg), 100.0)
    dz = np.ones((NPIX, maxg))
    w = np.zeros((NPIX, maxg))
    mag = np.zeros((NPIX, maxg))
    fill = np.zeros(NPIX, dtype=int)
    for p, zz, mm in zip(pix, z, m):
        i = fill[p]
        zg[p, i] = zz
        dz[p, i] = 0.01
        w[p, i] = 1.0
        mag[p, i] = mm
        fill[p] = i + 1
    order = np.argsort(zg, axis=1)
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = NSIDE
        f.attrs["z_depth"] = 0.35
        f.create_dataset("zgals", data=np.take_along_axis(zg, order, 1))
        f.create_dataset("dzgals", data=np.take_along_axis(dz, order, 1))
        f.create_dataset("wgals", data=np.take_along_axis(w, order, 1))
        f.create_dataset("ngals", data=counts)
        f.create_dataset("gal_app_mag", data=np.take_along_axis(mag, order, 1))
    return counts


def _write_fit(path, strata):
    payload = {
        "format_version": ("darksirens-selection-fit-1.0" if len(strata) == 1
                           else "darksirens-selection-fit-1.1"),
        "strata": [dict(family="gaussian", m_lim=ml, M0hat=m0, sigma_M=sg,
                        cov=[[1e-6, 0], [0, 1e-6]], n_gal=1000,
                        stratum=str(j), k_corr_coeffs=[])
                   for j, (ml, m0, sg) in enumerate(strata)],
    }
    path.write_text(json.dumps(payload))


def _write_map(path, labels):
    with h5py.File(path, "w") as f:
        f.create_dataset("stratum_map", data=np.asarray(labels, dtype=np.int32))


def _build(catalog, fit_json, stratum_map=None):
    from darksirens.cli.build_lognormal_completion import (
        _load_stratum_map,
        _stratum_map_sha256,
        build_completion,
    )
    from darksirens.redshift.selection import load_selection_fit_strata

    strata = load_selection_fit_strata(fit_json)
    kw = dict(selection_fit=strata[0])
    if len(strata) > 1:
        kw.update(selection_strata=strata,
                  stratum_map=_load_stratum_map(stratum_map, NSIDE, len(strata)),
                  stratum_map_sha=_stratum_map_sha256(stratum_map))
    return build_completion(
        str(catalog), mode="radial", n_members=0, seed=1, workers=1,
        log10n0=-2.5, delta=0.0, c_mode="selection", **kw)


THETA0 = (21.0, -20.8, 0.8)
THETA1 = (20.5, -20.5, 0.9)     # shallower + fainter second stratum


@pytest.fixture(scope="module")
def catalog(tmp_path_factory):
    d = tmp_path_factory.mktemp("strat_q")
    rng = np.random.default_rng(9)
    counts = _write_catalog(d / "cat.h5", rng)
    return d, counts


def test_degenerate_strata_table_matches_pooled(catalog):
    d, _ = catalog
    _write_fit(d / "fit1.json", [THETA0])
    _write_fit(d / "fit2.json", [THETA0, THETA0])
    _write_map(d / "map.h5", np.arange(NPIX) % 2)
    logq_pooled, _, diag_p = _build(d / "cat.h5", d / "fit1.json")
    logq_strat, _, diag_s = _build(d / "cat.h5", d / "fit2.json", d / "map.h5")
    np.testing.assert_allclose(logq_strat, logq_pooled, rtol=0, atol=1e-9)
    assert diag_s["selection_n_strata"] == 2.0
    assert "selection_stratum_map_sha256" in diag_s
    assert diag_s["selection_s1_m_lim"] == THETA0[0]
    assert "selection_n_strata" not in diag_p


def test_different_strata_change_the_base_per_pixel(catalog):
    d, _ = catalog
    _write_fit(d / "fit_ab.json", [THETA0, THETA1])
    _write_map(d / "map_ab.h5", np.arange(NPIX) % 2)
    _write_fit(d / "fit_a.json", [THETA0])
    logq_strat, _, diag = _build(d / "cat.h5", d / "fit_ab.json", d / "map_ab.h5")
    logq_pool, _, _ = _build(d / "cat.h5", d / "fit_a.json")
    assert not np.allclose(logq_strat, logq_pool, atol=1e-6)
    assert diag["selection_s0_M0hat"] == THETA0[1]
    assert diag["selection_s1_M0hat"] == THETA1[1]


def test_builder_pairing_errors(catalog):
    d, _ = catalog
    _write_fit(d / "fit_multi.json", [THETA0, THETA1])
    from darksirens.cli.build_lognormal_completion import main as bmain

    # multi-stratum fit without a map
    with pytest.raises((ValueError, SystemExit)):
        bmain(["--catalog", str(d / "cat.h5"), "--out", str(d / "q.h5"),
               "--c-mode", "selection",
               "--selection-fit", str(d / "fit_multi.json"),
               "--mode", "radial"])
    # wrong-size map
    _write_map(d / "map_bad.h5", np.zeros(7, dtype=np.int32))
    from darksirens.cli.build_lognormal_completion import _load_stratum_map

    with pytest.raises(ValueError, match="nside"):
        _load_stratum_map(d / "map_bad.h5", NSIDE, 2)


def _fid(n_strata=None, sha="abc", strata=None):
    f = {"path": "qt.h5", "selection_m_lim": 21.0,
         "selection_M0hat": -20.8, "selection_sigma_M": 0.8}
    if n_strata:
        f["selection_n_strata"] = float(n_strata)
        f["selection_stratum_map_sha256"] = sha
        for j, (ml, m0, sg) in enumerate(strata):
            f[f"selection_s{j}_m_lim"] = ml
            f[f"selection_s{j}_M0hat"] = m0
            f[f"selection_s{j}_sigma_M"] = sg
    return f


def test_firewall_strata_and_map_hash():
    from darksirens.cli import inference as cli

    def _opts(strata_fit, map_sha):
        # One per-catalog fit record per Q-table fiducial (catalog order).
        return SimpleNamespace(selection_fits=[{
            "catalog": 1, "path": "fit.json",
            "theta": {"m_lim": 21.0, "M0hat": -20.8, "sigma_M": 0.8},
            "k_corr_coeffs": [],
            "strata_fit": ([list(s) for s in strata_fit]
                           if strata_fit else None),
            "strata_struct": None, "stratum_map": None,
            "stratum_map_sha256": map_sha, "prior": {}}])

    Opts = _opts([THETA0, THETA1], "abc")

    good = _fid(2, "abc", [THETA0, THETA1])
    cli._check_selection_qtable_theta([good], Opts)          # passes

    with pytest.raises(SystemExit):                          # unstamped table
        cli._check_selection_qtable_theta([_fid()], Opts)
    with pytest.raises(SystemExit):                          # map hash mismatch
        cli._check_selection_qtable_theta(
            [_fid(2, "OTHER", [THETA0, THETA1])], Opts)
    with pytest.raises(SystemExit):                          # theta mismatch
        cli._check_selection_qtable_theta(
            [_fid(2, "abc", [THETA0, (20.5, -20.4, 0.9)])], Opts)
    with pytest.raises(SystemExit):                          # count mismatch
        cli._check_selection_qtable_theta(
            [_fid(3, "abc", [THETA0, THETA1, THETA1])], Opts)

    # Stamped (stratified) table against a single-stratum run.
    with pytest.raises(SystemExit):
        cli._check_selection_qtable_theta([good], _opts(None, None))
