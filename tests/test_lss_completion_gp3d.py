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
                                         gp3d_nz_solve=16, gp3d_pix_chunk=8)
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
