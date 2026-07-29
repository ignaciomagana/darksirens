"""Units/normalisation guards for the RADIAL lognormal-completion builder.

Library review P0.1: the radial builder point-evaluated the per-unit-z
expected density at the uniform-chi nodes and multiplied by a bin width in
Mpc — missing the dz/dchi Jacobian (~c/H ~ 4e3) — so the homogeneous
expectation was ~1e3x the observed counts and every radial table came out
crushed (all-negative logQ, ~20% railed at the -7 clip, n_converged = 0).
These tests anchor the solver's base counts to the observed counts on a
catalog drawn EXACTLY from the fiducial expectation, where logQ must be ~0.

Radial finding 2: the deterministic table is now the Laplace posterior-mean
E[Q] (matching gp3d and the mean of the radial ensemble), so data-free bins
read Q = 1, not exp(-b^2 sigma^2 / 2).
"""
import healpy as hp
import h5py
import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

import darksirens.cli.build_lognormal_completion as B
from darksirens.redshift import zgrid


NSIDE = 8
N_OCC = 40
MEAN_GALS = 40


def _fiducial_drawn_catalog(path, log10n0_target_counts=MEAN_GALS, seed=7):
    """Write a load_survey catalog whose galaxies are drawn from the builder's
    own fiducial dN_exp, with log10n0 calibrated so the expected counts per
    occupied pixel equal the drawn counts. Returns (path, log10n0)."""
    import jax.numpy as jnp
    from darksirens.core.types import EMCatalog

    npix = hp.nside2npix(NSIDE)
    apix = hp.nside2pixarea(NSIDE)
    zg = np.asarray(zgrid, dtype=float)

    # Fiducial expectation at the default n0 (dN_exp is linear in n0).
    cosmo, survey = B._fiducial_cosmo_survey(log10n0=-2.0)
    em0 = EMCatalog(
        apix=apix, zgals=jnp.full((npix, 1), 100.0), dzgals=jnp.full((npix, 1), 0.01),
        wgals=jnp.zeros((npix, 1)), ngals=jnp.zeros(npix, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, zg.size)), dN_obs_kde=None, pixel_to_cache_idx=None,
    )
    grids = B._precompute_grids(cosmo, survey, em0)
    dN_exp = np.asarray(grids.dN_exp, dtype=float)
    T = float(_trapezoid(dN_exp, zg))
    log10n0 = -2.0 + np.log10(log10n0_target_counts / T)

    # Inverse-CDF draws from dN_exp.
    rng = np.random.default_rng(seed)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dN_exp[1:] + dN_exp[:-1]) * np.diff(zg))])
    cdf /= cdf[-1]

    maxg = int(MEAN_GALS * 3)
    zgals = np.full((npix, maxg), 100.0)
    dzgals = np.full((npix, maxg), 0.005)
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    occ = rng.choice(npix, size=N_OCC, replace=False)
    for p in occ:
        n = int(rng.poisson(MEAN_GALS))
        n = max(3, min(n, maxg))
        zs = np.interp(rng.uniform(size=n), cdf, zg)
        zgals[p, :n] = zs
        wgals[p, :n] = 1.0
        ngals[p] = n

    with h5py.File(path, "w") as f:
        f.attrs["nside"] = NSIDE
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return float(log10n0)


@pytest.fixture(scope="module")
def fiducial_catalog(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("radial") / "survey_fid.h5")
    log10n0 = _fiducial_drawn_catalog(path)
    return path, log10n0


def test_radial_base_counts_anchored_to_observed(fiducial_catalog, monkeypatch):
    """The solver's homogeneous expectation must be the SAME order as the
    observed counts on a fiducial-drawn catalog (pre-fix: ~1e3x larger)."""
    path, log10n0 = fiducial_catalog
    captured = {}
    orig = B.poisson_lognormal_map

    def spy(N_obs, C, dN_exp, pk, **kw):
        captured["base"] = np.asarray(C, dtype=float) * np.asarray(dN_exp, dtype=float)
        captured["N"] = np.asarray(N_obs, dtype=float)
        return orig(N_obs, C, dN_exp, pk, **kw)

    monkeypatch.setattr(B, "poisson_lognormal_map", spy)
    B.build_completion(path, mode="radial", n_members=0, log10n0=log10n0)
    ratio = float(captured["base"].sum() / captured["N"].sum())
    assert 1.0 / 3.0 < ratio < 3.0, f"base/observed = {ratio} (Jacobian bug regressed?)"


def test_radial_fiducial_catalog_gives_near_unity_Q(fiducial_catalog):
    """On a catalog drawn exactly from the fiducial expectation, logQ ~ 0:
    no crushed field, no clip railing (pre-fix: median ~ -0.5, 20% at -7)."""
    path, log10n0 = fiducial_catalog
    logq_map, _, diag = B.build_completion(path, mode="radial", n_members=0,
                                           log10n0=log10n0)
    occ_rows = np.abs(logq_map).max(axis=1) > 0
    vals = logq_map[occ_rows]
    assert np.isfinite(vals).all()
    assert float((np.abs(vals) >= diag["logq_clip"] - 1e-9).mean()) == 0.0
    assert abs(float(np.median(vals))) < 0.3, float(np.median(vals))


def test_radial_deterministic_matches_member_mean(fiducial_catalog):
    """The deterministic table is E[Q] under the same Laplace approximation
    that generates the members, so it must track log(mean_m Q_m)."""
    path, log10n0 = fiducial_catalog
    logq_map, logq_members, _ = B.build_completion(path, mode="radial",
                                                   n_members=32,
                                                   log10n0=log10n0, seed=3)
    occ_rows = np.abs(logq_map).max(axis=1) > 0
    det = logq_map[occ_rows]
    mem = np.log(np.mean(np.exp(logq_members[:, occ_rows, :]), axis=0))
    err = np.abs(mem - det)
    assert float(np.median(err)) < 0.3, float(np.median(err))


def test_parallel_row_solves_are_bitwise_identical_to_serial():
    """``workers > 1`` must be an exact wall-clock optimisation, not an
    approximation.

    The per-pixel MAP solves are independent, so farming contiguous row blocks
    out to a process pool cannot change a single bit of the result.  At DESI
    nside=128 the serial loop over ~69k occupied pixels is ~42 h against ~3 h
    16-way, so the parallel path is what makes a real Q table affordable -- and
    a silent numerical difference there would be indistinguishable from the
    under-relaxation the builder already warns about.

    Workers are SPAWNED, so this test relies on pytest's own entry point being
    ``__main__``-guarded (it is); see the ``workers`` note in
    ``poisson_lognormal_map``.
    """
    from darksirens.redshift.lognormal_completion import (
        gaussian_correlation_spectrum, poisson_lognormal_map)

    rng = np.random.default_rng(3)
    n_rows, n_grid = 6, 64
    pk = gaussian_correlation_spectrum(n_grid, 4.0, 1.0)
    dN_exp = np.abs(rng.normal(90.0, 30.0, n_grid)) + 1.0
    C = np.clip(rng.uniform(0.0, 1.0, (n_rows, n_grid)), 0.0, 1.0)
    N_obs = rng.poisson(C * dN_exp).astype(float)

    serial = poisson_lognormal_map(N_obs, C, dN_exp, pk, maxiter=200, workers=1)
    parallel = poisson_lognormal_map(N_obs, C, dN_exp, pk, maxiter=200, workers=3)

    for key in ("s_map", "logq_map", "q_map", "lambda_map"):
        assert np.array_equal(serial[key], parallel[key]), key
    assert (serial["diagnostics"]["n_converged"]
            == parallel["diagnostics"]["n_converged"])


def test_parallel_row_solves_handle_more_workers_than_rows():
    """``workers`` above the row count must not produce empty blocks or hang."""
    from darksirens.redshift.lognormal_completion import (
        gaussian_correlation_spectrum, poisson_lognormal_map)

    rng = np.random.default_rng(5)
    n_rows, n_grid = 2, 32
    pk = gaussian_correlation_spectrum(n_grid, 3.0, 1.0)
    dN_exp = np.abs(rng.normal(50.0, 10.0, n_grid)) + 1.0
    C = np.clip(rng.uniform(0.0, 1.0, (n_rows, n_grid)), 0.0, 1.0)
    N_obs = rng.poisson(C * dN_exp).astype(float)

    serial = poisson_lognormal_map(N_obs, C, dN_exp, pk, maxiter=200, workers=1)
    parallel = poisson_lognormal_map(N_obs, C, dN_exp, pk, maxiter=200, workers=8)
    assert np.array_equal(serial["logq_map"], parallel["logq_map"])
