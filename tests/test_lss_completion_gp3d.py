"""
Tests for the 3-D angular-coupling LSS completion builder (``mode="gp3d"``).

Exercises the low-rank Poisson-lognormal GP solver in
``darksirens.redshift.lognormal_completion`` and the ``--mode gp3d`` build path.  The
output keeps the SAME ``Q_LSS(p,z)`` HDF5 table contract as the radial builder,
so the inference side is unchanged (verified by feeding the table to
``completion_curves`` via the global-pixel resolver).
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.redshift import zgrid
from darksirens.sky.models import _SphereZGPBase, _fibonacci_sphere
from darksirens.redshift.lognormal_completion import (
    lowrank_inducing_nodes,
    build_lowrank_operator,
    poisson_lognormal_gp3d_map,
    laplace_lognormal_gp3d_members,
    eval_logq_gp3d,
    save_lss_completion_hdf5,
    load_lss_completion_hdf5,
)

NG = int(zgrid.size)

# S0c: gp3d builds hard-error when the radial inducing-node spacing in
# zeta = log1p(z) exceeds ls_z (the Mpc correlation length mapped to zeta at
# the count-weighted z_ref) — the 50 Mpc fiducial is ~30x under-resolved at
# the historical 6-node grid and collapses to the prior (Burt et al. 2019).
# End-to-end fixtures therefore build at a RESOLVED radial lengthscale; the
# guard itself is tested in the "resolution guard + knob plumbing" section.
RESOLVED_L_MPC = 3000.0


# ------------------------------------------------------------
# Kernel/node reuse + the mean-one / homogeneous limits
# ------------------------------------------------------------

def test_inducing_nodes_match_sky_gp():
    """The builder's inducing nodes MUST equal the online (sphere x z) GP's,
    node-for-node (same kernel + ordering i = i_sph*M_z + i_z) — the correctness
    guard for the direct-import reuse."""
    base = _SphereZGPBase(n_inducing_sphere=32, n_inducing_z=6, z_node_hi=3.0)
    Zn, Zz = lowrank_inducing_nodes(n_inducing_sphere=32, n_inducing_z=6, z_node_hi=3.0)
    assert Zn.shape == (192, 3)
    assert Zz.shape == (192,)
    assert np.allclose(np.asarray(Zn), np.asarray(base._Zn))
    assert np.allclose(np.asarray(Zz), np.asarray(base._Zz))


def _small_eval_grid(n_dir=24, n_z=8):
    n_hat = np.asarray(_fibonacci_sphere(n_dir))      # (n_dir, 3) unit vectors
    z_out = np.linspace(0.05, 2.0, n_z)
    return n_hat, z_out


def test_homogeneous_limit_is_unity():
    """Deterministic (posterior-mean) eval with xi=0 and H=I (no data shrinkage)
    -> logQ = 0 (Q = 1) everywhere: the homogeneous limit the GW likelihood must
    see for an empty sky.  Also checks the pixel-chunking is exact (chunk ∤ n)."""
    Zn, Zz = lowrank_inducing_nodes()
    M = int(Zn.shape[0])
    n_hat, z_out = _small_eval_grid()
    logq = eval_logq_gp3d(
        np.zeros(M), Zn, Zz, amp=1.0, ls_sph=0.4, ls_z=0.5,
        n_hat_out=n_hat, z_out=z_out, bias=1.0, H_chol=np.eye(M), pix_chunk=7,
    )
    assert logq.shape == (n_hat.shape[0], z_out.shape[0])
    assert float(np.max(np.abs(logq))) < 1e-9


def test_per_draw_xi0_is_mean_one_shift():
    """Per-draw eval (no H_chol) at xi=0 -> logQ = -0.5*bias^2*prior_var < 0 (the
    lognormal mean-one shift; the mean of Q over prior draws is 1, but a single
    draw at the prior mean sits below)."""
    Zn, Zz = lowrank_inducing_nodes()
    M = int(Zn.shape[0])
    n_hat, z_out = _small_eval_grid()
    logq = eval_logq_gp3d(
        np.zeros(M), Zn, Zz, amp=1.0, ls_sph=0.4, ls_z=0.5,
        n_hat_out=n_hat, z_out=z_out, bias=1.0,
    )
    assert np.all(logq <= 1e-9)
    assert float(np.min(logq)) < -1e-3


# ------------------------------------------------------------
# Solver + angular borrowing
# ------------------------------------------------------------

def _solve_overdense_pixel(amp=1.0, ls_sph=0.6, ls_z=0.6, bias=1.0):
    """Solve with ONE overdense occupied direction A; return the pieces to eval at
    A (overdense), B (near A), C (far)."""
    Zn, Zz = lowrank_inducing_nodes()
    A = np.array([0.0, 0.0, 1.0])
    th = np.deg2rad(15.0)
    B = np.array([np.sin(th), 0.0, np.cos(th)])       # ~15 deg from A
    C = np.array([0.0, 0.0, -1.0])                    # opposite pole (far)
    z_s = np.linspace(0.1, 1.5, 6)
    zeta_s = np.log1p(z_s)
    j_struct = 3
    base = np.full(z_s.size, 10.0)                    # homogeneous baseline rate
    N_obs = base.copy()
    N_obs[j_struct] = 80.0                            # overdensity at z_s[j_struct]
    X_n = np.repeat(A[None, :], z_s.size, axis=0)
    Phi, L = build_lowrank_operator(Zn, Zz, X_n, zeta_s, amp=amp, ls_sph=ls_sph, ls_z=ls_z)
    mp = poisson_lognormal_gp3d_map(N_obs, base, Phi, bias=bias)
    g = dict(A=A, B=B, C=C, z_s=z_s, j=j_struct, amp=amp, ls_sph=ls_sph,
             ls_z=ls_z, bias=bias)
    return Zn, Zz, L, mp, g


def test_solver_converges():
    Zn, Zz, L, mp, _ = _solve_overdense_pixel()
    d = mp["diagnostics"]
    assert d["converged"]
    assert d["grad_inf"] < 1e-6
    assert mp["H_chol"].shape == (d["M"], d["M"])
    assert np.all(np.isfinite(mp["H_chol"]))
    assert np.all(np.isfinite(mp["xi_map"]))


def test_angular_borrowing():
    """An overdense occupied pixel A makes a NEAR empty pixel B borrow (Q_B > 1 at
    the structure redshift) while a FAR empty pixel C stays homogeneous
    (Q_C ~ 1).  This is the whole point of the 3-D upgrade."""
    Zn, Zz, L, mp, g = _solve_overdense_pixel()
    n_hat = np.stack([g["A"], g["B"], g["C"]])
    logq = eval_logq_gp3d(
        mp["xi_map"], Zn, Zz, amp=g["amp"], ls_sph=g["ls_sph"], ls_z=g["ls_z"],
        n_hat_out=n_hat, z_out=g["z_s"], bias=g["bias"], L=L, H_chol=mp["H_chol"],
    )
    j = g["j"]
    qA, qB, qC = float(logq[0, j]), float(logq[1, j]), float(logq[2, j])
    assert qA > qB > qC                 # monotone with angular distance
    assert qB > 0.05                    # B genuinely borrows (Q_B > 1)
    assert abs(qC) < 0.05               # C is effectively homogeneous (Q_C ~ 1)


def test_solver_clip_saturation_no_stall():
    """A voxel demanding a field far beyond field_clip (base ~ 1e-300 with
    N_obs > 0 — the BGS z=0-node pathology) must saturate gracefully at the
    clip: with the clip-masked gradient/Hessian the line search stays
    consistent, the impossible voxel pins at +field_clip, and the rest of the
    field still converges (pre-fix: Armijo stalled at iteration 2 and the
    returned field was ~zero everywhere with grad_inf ~ 1e4)."""
    Zn, Zz = lowrank_inducing_nodes()
    z_s = np.linspace(0.1, 1.5, 6)
    A = np.array([0.0, 0.0, 1.0])
    X_n = np.repeat(A[None, :], z_s.size, axis=0)
    Phi, _ = build_lowrank_operator(Zn, Zz, X_n, np.log1p(z_s),
                                    amp=1.0, ls_sph=0.6, ls_z=0.6)
    base = np.full(z_s.size, 10.0)
    N_obs = base.copy()
    base[0] = 1e-300            # essentially zero expected counts ...
    N_obs[0] = 500.0            # ... but real galaxies counted in the bin
    # (the Poisson pull on the shared latents scales with N, so N must beat the
    # unit prior by >> field_clip to actually saturate, as the aggregated BGS
    # z=0-node counts did)
    mp = poisson_lognormal_gp3d_map(N_obs, base, Phi, bias=1.0)
    d = mp["diagnostics"]
    assert np.all(np.isfinite(mp["xi_map"]))
    fld = np.asarray(mp["f_solve"])
    assert float(fld[0]) >= 10.0 - 1e-6     # pinned at clip (within the
    # solver's clip_eps band; a kink minimum is reached from inside in floats)
    assert np.all(np.isfinite(fld))
    assert float(np.max(np.abs(fld[1:]))) < 5.0   # rest of the field stays fit
    # grad_inf is the residual AFTER absorbing the bounded KKT multipliers of
    # the pinned voxels; pre-fix it was ~1e4 (raw stalled gradient). Exact
    # stationarity at a kink needs equality-constrained steps the solver does
    # not take (the pin chatters), so `converged` is honestly False here and
    # only the relative residual is asserted; consistently-built inputs
    # (base ~ N wherever galaxies exist) never pin and do report converged.
    gscale = float(np.max(N_obs))
    assert d["grad_inf"] < 0.01 * gscale, d


def test_underdetermined_hessian_spd():
    """V < M (far fewer voxels than latents): the unit-Gaussian prior keeps H SPD
    and the solve well-posed (no special handling needed)."""
    Zn, Zz = lowrank_inducing_nodes()
    X_n = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    X_z = np.log1p(np.array([0.2, 0.8, 0.5]))
    Phi, L = build_lowrank_operator(Zn, Zz, X_n, X_z, amp=1.0, ls_sph=0.5, ls_z=0.5)
    mp = poisson_lognormal_gp3d_map(
        np.array([30.0, 5.0, 10.0]), np.array([10.0, 10.0, 10.0]), Phi, bias=1.0)
    assert mp["diagnostics"]["converged"]
    assert np.all(np.isfinite(mp["xi_map"]))


def test_members_mean_matches_posterior_mean():
    """The empirical mean of Q over Laplace members reproduces the analytic
    deterministic posterior-mean table (they are the same estimator)."""
    Zn, Zz, L, mp, g = _solve_overdense_pixel()
    n_hat = np.stack([g["A"], g["B"], g["C"]])
    det = eval_logq_gp3d(
        mp["xi_map"], Zn, Zz, amp=g["amp"], ls_sph=g["ls_sph"], ls_z=g["ls_z"],
        n_hat_out=n_hat, z_out=g["z_s"], bias=g["bias"], L=L, H_chol=mp["H_chol"],
    )
    xi_mem = laplace_lognormal_gp3d_members(mp["xi_map"], mp["H_chol"],
                                            n_members=3000, seed=0)
    assert xi_mem.shape == (3000, int(Zn.shape[0]))
    mem = eval_logq_gp3d(
        xi_mem, Zn, Zz, amp=g["amp"], ls_sph=g["ls_sph"], ls_z=g["ls_z"],
        n_hat_out=n_hat, z_out=g["z_s"], bias=g["bias"], L=L,   # per-draw
    )
    assert mem.shape == (3000, n_hat.shape[0], g["z_s"].size)
    emp_mean_logq = np.log(np.mean(np.exp(mem), axis=0))
    assert np.allclose(emp_mean_logq, det, atol=0.08)


# ------------------------------------------------------------
# End-to-end build + the inference contract
# ------------------------------------------------------------

def _write_tiny_survey(path, nside=2):
    """A minimal load_survey-schema catalog with two occupied pixels clustered
    near z ~ 0.5."""
    import h5py
    import healpy as hp
    npix = hp.nside2npix(int(nside))
    maxg = 5
    zgals = np.full((npix, maxg), 100.0)
    dzgals = np.full((npix, maxg), 0.01)
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    rng = np.random.default_rng(0)
    for p in (0, 1):
        zgals[p, :3] = 0.5 + 0.02 * rng.standard_normal(3)
        dzgals[p, :3] = 0.01
        wgals[p, :3] = 1.0
        ngals[p] = 3
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = int(nside)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return npix


def test_gp3d_build_smoke_and_contract(tmp_path):
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion
    cat = str(tmp_path / "survey.h5")
    npix = _write_tiny_survey(cat, nside=2)
    logq_map, logq_members, diag = build_completion(
        cat, mode="gp3d", n_members=4, seed=1, gp3d_nz_solve=16, gp3d_pix_chunk=8,
        lss_corr_length_mpc=RESOLVED_L_MPC,
    )
    assert logq_map.shape == (npix, NG)
    assert np.all(np.isfinite(logq_map))
    assert float(np.max(np.abs(logq_map))) <= 7.0 + 1e-9
    assert logq_members.shape == (4, npix, NG)
    assert diag["mode"] == "gp3d"
    assert diag["converged"]
    # the fiducial keys the inference loader prints must be present
    for k in ("fiducial_H0", "bias_b_miss", "lss_corr_length_mpc", "lss_sigma",
              "lss_corr_length_ang"):
        assert k in diag
    # S0c defaults: historical 32 x 6 inducing grid, but z_node_hi now spans
    # the PACKAGE zgrid (documented behavior change from the hardwired 3.0).
    assert diag["M_sph"] == 32 and diag["M_z"] == 6
    assert diag["gp3d_z_node_hi"] == float(np.asarray(zgrid)[-1])


def _write_lowz_survey(path, nside=2):
    """Galaxies at z ~ 0.05 in two pixels: the first coarse solve bin holds them
    all while dN_exp_density(z=0) ~ 0, so a base built by point-evaluating at
    the node (instead of integrating C * dN_exp over the bin) is ~1e-308
    against a real N_obs — the construction that saturated the BGS solve."""
    import h5py
    import healpy as hp
    npix = hp.nside2npix(int(nside))
    maxg = 5
    zgals = np.full((npix, maxg), 100.0)
    dzgals = np.full((npix, maxg), 0.01)
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    rng = np.random.default_rng(1)
    for p in (0, 1):
        zgals[p, :3] = 0.05 + 0.01 * rng.standard_normal(3)
        wgals[p, :3] = 1.0
        ngals[p] = 3
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = int(nside)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return npix


def test_gp3d_lowz_catalog_base_consistent(tmp_path):
    """End-to-end: a low-z catalog must build a converged gp3d table with no
    spurious low-z completion spike. Fails pre-fix (bin-integrated base): the
    z=0 node's point-evaluated base underflows, the solve saturates, and the
    table degenerates."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion
    cat = str(tmp_path / "survey_lowz.h5")
    npix = _write_lowz_survey(cat, nside=2)
    logq_map, _, diag = build_completion(cat, mode="gp3d", n_members=0,
                                         gp3d_nz_solve=16, gp3d_pix_chunk=8,
                                         lss_corr_length_mpc=RESOLVED_L_MPC)
    assert diag["converged"], diag
    assert logq_map.shape[0] == npix
    assert np.all(np.isfinite(logq_map))
    j_lo = int(np.searchsorted(np.asarray(zgrid), 0.15))
    # fixed build measures ~0.46 here; the point-evaluated-base regression
    # measures ~2.97 (module review of the fix, finding 2a)
    assert float(np.max(logq_map[:, :j_lo])) < 1.0


def test_gp3d_base_counts_anchored_to_histogram(tmp_path, monkeypatch):
    """Units guard on the bin-integrated base: over the bins that actually hold
    the galaxies, sum(base) must be the same order as sum(N_obs) — for these
    sparse fixtures C < 1 unclipped, so integral(C * dN_exp) ~ integral(dN_obs)
    ~ the galaxy count. Catches stray normalization factors (dropped/extra dz)
    that both end-to-end tests would miss (module review of the fix, finding
    2b: a x0.25 base error still converged and passed the spike assert)."""
    pytest.importorskip("healpy")
    import darksirens.cli.build_lognormal_completion as B
    cat = str(tmp_path / "survey_lowz.h5")
    _write_lowz_survey(cat, nside=2)
    captured = {}
    orig = B.poisson_lognormal_gp3d_map

    def spy(N_obs, base, Phi, **kw):
        captured["N"] = np.asarray(N_obs, dtype=float).copy()
        captured["base"] = np.asarray(base, dtype=float).copy()
        return orig(N_obs, base, Phi, **kw)

    monkeypatch.setattr(B, "poisson_lognormal_gp3d_map", spy)
    B.build_completion(cat, mode="gp3d", n_members=0,
                       gp3d_nz_solve=16, gp3d_pix_chunk=8,
                       lss_corr_length_mpc=RESOLVED_L_MPC)
    N, base = captured["N"], captured["base"]
    hold = N > 0
    assert hold.any()
    ratio = float(base[hold].sum() / N[hold].sum())
    assert 0.3 < ratio < 3.0, ratio


def test_gp3d_build_radial_dispatch_unchanged(tmp_path):
    """mode='radial' still returns the global (n_pix, NG) table; unknown mode
    raises."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion
    cat = str(tmp_path / "survey.h5")
    npix = _write_tiny_survey(cat, nside=2)
    logq_map, _, _ = build_completion(cat, mode="radial", n_members=0, maxiter=50)
    assert logq_map.shape == (npix, NG)
    with pytest.raises(ValueError, match="Unknown build mode"):
        build_completion(cat, mode="nope")


def test_gp3d_output_consumable_by_inference(tmp_path):
    """The global gp3d table round-trips through HDF5 and is consumed unchanged by
    the inference resolver (completion_curves) via unique_pixels -> rows."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion
    from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
    from darksirens.redshift.completion import completion_curves

    cat = str(tmp_path / "survey.h5")
    npix = _write_tiny_survey(cat, nside=2)
    logq_map, _, diag = build_completion(cat, mode="gp3d", n_members=0,
                                         gp3d_nz_solve=16, gp3d_pix_chunk=8,
                                         lss_corr_length_mpc=RESOLVED_L_MPC)
    out = str(tmp_path / "q_gp3d.h5")
    save_lss_completion_hdf5(out, logq_map=logq_map, zgrid=np.asarray(zgrid),
                             indexing="global", metadata=diag)
    d = load_lss_completion_hdf5(out)
    assert d["logq_map"].shape == (npix, NG)
    assert d["indexing"] == "global"
    assert np.allclose(np.asarray(d["zgrid"]), np.asarray(zgrid))

    cosmo = CosmoParams(H0=67.74, Om0=0.3075)
    survey = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0, b_miss=0.0, alpha_miss=1.0)
    zg = np.full((2, 3), 100.0); zg[0, :2] = [0.45, 0.5]; zg[1, :1] = [0.55]
    ng = np.array([2, 1], dtype=np.int32)
    em = EMCatalog(
        apix=1.0, zgals=jnp.asarray(zg), dzgals=jnp.asarray(np.full((2, 3), 0.01)),
        wgals=jnp.zeros((2, 3)), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, NG)), dN_obs_kde=None, pixel_to_cache_idx=None,
        unique_pixels=jnp.asarray([0, 1], jnp.int32),
        lss_completion_logq=jnp.asarray(d["logq_map"]), lss_completion_indexing=2,
    )
    curves = completion_curves(cosmo, survey, em)
    assert curves.dN_miss.shape == (2, NG)
    assert np.all(np.isfinite(np.asarray(curves.dN_miss)))


# ------------------------------------------------------------
# S0c: resolution guard + hyperparameter knob plumbing
# ------------------------------------------------------------

def test_resolution_guard_raises_at_underresolved_fiducial(tmp_path):
    """The shipped 50 Mpc fiducial maps to ls_z ~ 0.01 in zeta at z_ref ~ 0.5
    — ~30x below the 6-node spacing — so a default gp3d build must HARD ERROR
    (not silently collapse to the prior), stating both numbers and the minimum
    node count."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion
    cat = str(tmp_path / "survey.h5")
    _write_tiny_survey(cat, nside=2)
    with pytest.raises(ValueError, match="under-resolved") as ei:
        build_completion(cat, mode="gp3d", n_members=0,
                         gp3d_nz_solve=16, gp3d_pix_chunk=8)
    msg = str(ei.value)
    assert "ls_z" in msg and "zeta" in msg
    assert "--gp3d-nz-nodes >=" in msg


def test_resolution_guard_arithmetic(capsys):
    """Direct guard checks: the pass/fail boundary is spacing = ls_z, the
    suggested minimum node count actually resolves, and the sphere side only
    WARNS (never raises)."""
    from darksirens.cli.build_lognormal_completion import _gp3d_resolution_guard
    z_hi, n_z = 3.0, 6
    spacing = np.log1p(z_hi) / (n_z - 1)
    # exactly-resolved radial grid passes and returns the spacing
    got = _gp3d_resolution_guard(z_node_hi=z_hi, n_z_nodes=n_z, ls_z=spacing,
                                 n_sph_nodes=32, ls_sph=1.0)
    assert np.isclose(got, spacing)
    capsys.readouterr()
    # under-resolved raises, and the suggested minimum node count resolves
    with pytest.raises(ValueError, match="under-resolved") as ei:
        _gp3d_resolution_guard(z_node_hi=z_hi, n_z_nodes=n_z, ls_z=0.01,
                               n_sph_nodes=32, ls_sph=1.0)
    import re
    min_nodes = int(re.search(r"--gp3d-nz-nodes >= (\d+)", str(ei.value)).group(1))
    assert np.log1p(z_hi) / (min_nodes - 1) <= 0.01
    # fewer than 2 radial nodes is rejected outright
    with pytest.raises(ValueError, match="gp3d-nz-nodes"):
        _gp3d_resolution_guard(z_node_hi=z_hi, n_z_nodes=1, ls_z=1.0,
                               n_sph_nodes=32, ls_sph=1.0)
    # sphere side: sqrt(4pi/M_sph) > ls_sph warns but does not raise
    capsys.readouterr()
    _gp3d_resolution_guard(z_node_hi=z_hi, n_z_nodes=n_z, ls_z=spacing,
                           n_sph_nodes=32, ls_sph=0.2)
    assert "sphere" in capsys.readouterr().out


def test_gp3d_knob_plumbing_reaches_diagnostics(tmp_path):
    """Non-default knobs must reach the builder: the inducing-grid shape, the
    node range, and the overridden GP hyperparameters are all stamped in the
    diagnostics (and hence the HDF5 provenance)."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion
    cat = str(tmp_path / "survey.h5")
    _write_tiny_survey(cat, nside=2)
    _, _, diag = build_completion(
        cat, mode="gp3d", n_members=0, gp3d_nz_solve=12, gp3d_pix_chunk=8,
        lss_corr_length_mpc=3000.0, lss_sigma=0.8,
        gp3d_nz_nodes=5, gp3d_nsph_nodes=16, gp3d_z_node_hi=2.0,
    )
    assert diag["M_sph"] == 16 and diag["M_z"] == 5
    assert diag["M"] == 16 * 5
    assert diag["gp3d_z_node_hi"] == 2.0
    assert diag["lss_corr_length_mpc"] == 3000.0
    assert diag["lss_sigma"] == 0.8
    assert np.isclose(diag["zeta_node_spacing"], np.log1p(2.0) / 4)
    # ls_z is the mapped lengthscale actually used; it must resolve the grid
    assert diag["zeta_node_spacing"] <= diag["ls_z_zeta"]


def test_radial_knob_plumbing_reaches_builder(tmp_path):
    """--lss-corr-length-mpc / --lss-sigma reach the RADIAL builder too: the
    stamped hyperparameters follow the overrides and the uniform-chi grid
    length ell_grid scales linearly with the Mpc correlation length; defaults
    keep the SurveyParams fiducials (backward compatible)."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion
    cat = str(tmp_path / "survey.h5")
    _write_tiny_survey(cat, nside=2)
    kw = dict(mode="radial", n_members=0, maxiter=50)
    _, _, d0 = build_completion(cat, **kw)
    assert d0["lss_corr_length_mpc"] == 50.0   # SurveyParams fiducial
    assert d0["lss_sigma"] == 1.0
    _, _, d1 = build_completion(cat, lss_corr_length_mpc=100.0, lss_sigma=0.5, **kw)
    assert d1["lss_corr_length_mpc"] == 100.0
    assert d1["lss_sigma"] == 0.5
    assert np.isclose(d1["ell_grid_uniform_chi"] / d0["ell_grid_uniform_chi"], 2.0)


def test_cli_flags_reach_build_completion(tmp_path, monkeypatch):
    """The five S0c argparse flags must be forwarded verbatim into
    build_completion (spied; the build itself is not run)."""
    pytest.importorskip("healpy")
    import darksirens.cli.build_lognormal_completion as B
    captured = {}

    def spy(catalog_path, **kw):
        captured.update(kw)
        raise RuntimeError("spy stop")

    monkeypatch.setattr(B, "build_completion", spy)
    with pytest.raises(RuntimeError, match="spy stop"):
        B.main(["--catalog", "cat.h5", "--out", str(tmp_path / "q.h5"),
                "--mode", "gp3d",
                "--lss-corr-length-mpc", "1234.0", "--lss-sigma", "0.7",
                "--gp3d-nz-nodes", "17", "--gp3d-nsph-nodes", "9",
                "--gp3d-z-node-hi", "2.5"])
    assert captured["lss_corr_length_mpc"] == 1234.0
    assert captured["lss_sigma"] == 0.7
    assert captured["gp3d_nz_nodes"] == 17
    assert captured["gp3d_nsph_nodes"] == 9
    assert captured["gp3d_z_node_hi"] == 2.5
    # defaults are backward-compatible: no override unless the flag is passed
    captured.clear()
    with pytest.raises(RuntimeError, match="spy stop"):
        B.main(["--catalog", "cat.h5", "--out", str(tmp_path / "q.h5")])
    assert captured["lss_corr_length_mpc"] is None
    assert captured["lss_sigma"] is None
    assert captured["gp3d_nz_nodes"] == 6
    assert captured["gp3d_nsph_nodes"] == 32
    assert captured["gp3d_z_node_hi"] is None  # -> zgrid[-1] inside the builder


def test_lss_corr_length_ang_not_sampled():
    """The new fixed SurveyParams.lss_corr_length_ang (and the existing LSS
    hyperparameters) must NOT enter the sampled parameter space — they are
    offline-builder-only fiducials."""
    from darksirens.inference.prior import build_parameter_space
    out = build_parameter_space("powerlaw+peak", False, False, False,
                                universe_model="dark_sirens")
    labels, survey_labels = out[0], out[5]
    assert not any(("lss_corr" in l) or ("lss_sigma" in l) for l in labels)
    for fixed in ("lss_corr_length_ang", "lss_corr_length_mpc", "lss_sigma"):
        assert fixed not in survey_labels


def test_gp3d_empty_catalog_writes_unity_table(tmp_path):
    """An empty (or fully masked) catalog must write the honest Q = 1 table the
    n_occ == 0 shortcut builds.

    The shortcut returned diagnostics with no 'converged'/'grad_inf', so main()
    raised TypeError formatting None, and with the format fixed the fail-closed
    gate would then have declared the trivial solve unconverged (review F-112).
    """
    pytest.importorskip("healpy")
    import h5py
    import healpy as hp
    from darksirens.cli.build_lognormal_completion import main as build_main
    from darksirens.redshift.lognormal_completion import load_lss_completion_hdf5

    nside = 1
    npix = hp.nside2npix(nside)
    cat = str(tmp_path / "empty.h5")
    with h5py.File(cat, "w") as f:
        f.attrs["nside"] = nside
        f.create_dataset("zgals", data=np.full((npix, 1), 100.0))
        f.create_dataset("ngals", data=np.zeros(npix, dtype=np.int32))
        f.create_dataset("dzgals", data=np.full((npix, 1), 0.01))
        f.create_dataset("wgals", data=np.zeros((npix, 1)))
    out = str(tmp_path / "q_empty.h5")
    build_main(["--catalog", cat, "--out", out, "--mode", "gp3d",
                "--n-members", "2", "--gp3d-nz-solve", "8",
                "--lss-corr-length-mpc", str(RESOLVED_L_MPC)])
    d = load_lss_completion_hdf5(out)
    assert float(np.max(np.abs(np.asarray(d["logq_map"])))) == 0.0
    assert float(np.max(np.abs(np.asarray(d["logq_members"])))) == 0.0
    assert d["diagnostics"]["converged"] is True
    assert d["diagnostics"]["n_occupied"] == 0
    assert d["budget_renormalized"] is True


def test_gp3d_stamps_the_allsky_budget_residual(tmp_path):
    """The per-z mean-one renormalization covers the FITTED rows only, while the
    borrowing halo keeps Q != 1 against the FULL homogeneous budget weight
    (an unfitted pixel is empty, so C = 0). That residual must be auditable from
    the file rather than left unmeasured (review F-113)."""
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = str(tmp_path / "survey.h5")
    npix = _write_tiny_survey(cat, nside=2)
    logq_map, _members, diag = build_completion(
        cat, mode="gp3d", n_members=0, seed=1, gp3d_nz_solve=16,
        gp3d_pix_chunk=8, lss_corr_length_mpc=RESOLVED_L_MPC)

    assert diag["budget_renormalized"] is True
    assert diag["budget_residual_allsky_n_unfitted"] == npix - diag["n_occupied"]
    resid = diag["budget_residual_allsky_at_max"]
    dev = diag["budget_residual_allsky_max_abs_dev"]
    assert dev == pytest.approx(abs(resid - 1.0), rel=1e-12)
    assert 0.0 <= dev < 5.0
    assert 0.0 <= diag["budget_residual_allsky_z_at_max"] <= float(np.asarray(zgrid)[-1])
