"""PR-0 item 2b: closed-form member-spread prediction sigma = ||L_H^{-1} a||_2.

PLAN eq. (6):   a = b_GW ( sum_i phi_i Phi_i - N_obs <Phi>_sel )
                sigma = || L_H^{-1} a ||_2          (H^{-1} norm)

with H = I + b^2 Phi^T diag(lambda) Phi the count-channel Laplace Hessian at
the anchor.  Because the field-mode missing base is sky-uniform over the
footprint (build_lognormal_completion.py:744, base_vox = tile(base_row)),
lambda_{p,z} = lambda_z for every occupied pixel and the Fisher factorizes
exactly over the separable kernel:  H = I + S_sph (x) S_z, diagonalized by the
factor eigenbases -- no M x M assembly.

Basis: the guard-compliant OD3 kernel (ls_sph = 0.2 chordal, ls_z = 0.039 in
zeta = log1p z, M_sph = 315, M_z = 8 over [0, log1p(0.30)]), factored-v1
jitter j_sph = j_z = 1e-6, amp = 1 -- the same construction as
sky/models._sphere_z_kernel and lognormal_completion.build_lowrank_operator.

Stated approximations (PR-0 is a prediction, confirmed/refuted at PR-5b):
  * event samples enter with uniform (posterior) weights, not the full
    population importance weights;
  * injection weights use log_target_density_base_and_z (population + pdraw +
    Jacobians) times this script's own catalog-prior densities, reproducing
    the field-weighting composition analytically rather than through the
    production evaluator;
  * the count-channel Fisher is the unconditioned Poisson one (the
    shell-total-conditioned multinomial Fisher of PLAN 1.1 is smaller, so
    sigma is if anything over-predicted -- conservative for OD5).

Run from experiments/desi_full259:  python ../field_level_plan/pr0/compute_sigma.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PR0_DIR = Path(__file__).resolve().parent
FULL259 = PR0_DIR.parent.parent / "desi_full259"
sys.path.insert(0, str(FULL259))

import common as C  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import h5py  # noqa: E402
import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ANCHOR_H0 = 67.74
B_GW = 1.0
LS_SPH = 0.2          # chordal
LS_Z = 0.039          # zeta = log1p z
M_SPH = 315
M_Z = 8
JITTER = 1e-6
NSIDE = 64
N_PIX = 12 * NSIDE ** 2
CLIGHT = 299792.458   # km/s


# ---------------------------------------------------------------- cosmology
def _dc_of_z(zgrid, H0, Om0):
    E = np.sqrt(Om0 * (1 + zgrid) ** 3 + (1 - Om0))
    integ = 1.0 / E
    dc = np.concatenate([[0.0], np.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * np.diff(zgrid))]) * CLIGHT / H0
    return dc, E


def main():
    from darksirens.redshift.selection import load_selection_fit_json, \
        c_sel_gaussian
    from darksirens.sky.models import _fibonacci_sphere

    sel = load_selection_fit_json(C.FIT_JSON)
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    n0 = 10.0 ** float(cal["log10n0"])
    delta = float(cal["delta"])
    kcorr = tuple(sel["k_corr_coeffs"])
    theta = dict(m_lim=float(sel["m_lim"]), M0hat=float(sel["M0hat"]),
                 sigma_M=float(sel["sigma_M"]))

    # fine z grid to z_depth
    zf = np.linspace(1e-4, C.Z_DEPTH, 400)
    dc, E = _dc_of_z(zf, ANCHOR_H0, C.OM0)
    dVdz = 4.0 * np.pi * (CLIGHT / ANCHOR_H0) * dc ** 2 / E        # Mpc^3
    Cz = np.asarray(c_sel_gaussian(
        jnp.asarray(zf), theta["m_lim"], theta["M0hat"], theta["sigma_M"],
        ANCHOR_H0, C.OM0, k_corr_coeffs=kcorr))
    dNexp_pix = n0 * (1 + zf) ** delta * dVdz / N_PIX              # per pixel
    miss_pix = (1.0 - Cz) * dNexp_pix                              # per pixel

    # ------------------------------------------------------------ survey
    with h5py.File(C.SURVEY_N64) as f:
        ngals = f["ngals"][...]
        zg = f["zgals"][...]
        wg = f["wgals"][...]
        dzg = f["dzgals"][...]
    occ = ngals > 0
    print(f"[survey] occupied {occ.sum()}/{N_PIX}")

    # ------------------------------------------------------------ basis
    Z_sph = np.asarray(_fibonacci_sphere(M_SPH))
    zeta_nodes = np.linspace(0.0, np.log1p(C.Z_DEPTH), M_Z)

    def chol_factor(K):
        return np.linalg.cholesky(K + JITTER * np.eye(K.shape[0]))

    cos = Z_sph @ Z_sph.T
    Ksph = np.exp(-0.5 * np.clip(2 - 2 * cos, 0, 4) / LS_SPH ** 2)
    Kzz = np.exp(-0.5 * (zeta_nodes[:, None] - zeta_nodes[None, :]) ** 2
                 / LS_Z ** 2)
    Lsph, Lz = chol_factor(Ksph), chol_factor(Kzz)

    pix_vec = np.column_stack(hp.pix2vec(NSIDE, np.arange(N_PIX)))

    from scipy.linalg import solve_triangular

    def _A(vec):
        c = np.clip(vec @ Z_sph.T, -1, 1)
        k = np.exp(-0.5 * np.clip(2 - 2 * c, 0, 4) / LS_SPH ** 2)
        return solve_triangular(Lsph, k.T, lower=True).T

    def _B(z):
        zeta = np.log1p(z)
        k = np.exp(-0.5 * (zeta[:, None] - zeta_nodes[None, :]) ** 2
                   / LS_Z ** 2)
        return solve_triangular(Lz, k.T, lower=True).T

    A_pix = _A(pix_vec)                                   # (N_PIX, M_SPH)

    # ------------------------------------------------------------ Fisher
    S_sph = A_pix[occ].T @ A_pix[occ]                     # (M_SPH, M_SPH)
    Bf = _B(zf)                                           # (400, M_Z)
    lam_z = Cz * dNexp_pix                                # counts model / pixel
    wtrap = np.gradient(zf)
    S_z = (Bf * (lam_z * wtrap)[:, None]).T @ Bf          # (M_Z, M_Z) per pixel
    # lambda identical across occupied pixels -> pixel sum lives in S_sph
    es, Us = np.linalg.eigh(S_sph)
    ez, Uz = np.linalg.eigh(S_z)
    denom = 1.0 + B_GW ** 2 * np.outer(es, ez)            # (M_SPH, M_Z)
    print(f"[fisher] S_sph tr={np.trace(S_sph):.3e}  S_z tr={np.trace(S_z):.3e}"
          f"  max eig product={denom.max() - 1:.3e}")

    # ------------------------------------------- per-pixel KDE missing frac
    sig_eff = np.maximum(np.sqrt(dzg ** 2 + 0.003 ** 2), 1e-4)

    def miss_frac(pix, z):
        """m_s = dN_miss / (dN_obs + dN_miss) at (pix, z); count units."""
        out = np.empty(z.shape)
        mp = np.interp(z, zf, miss_pix)
        for p in np.unique(pix):
            m = pix == p
            n = ngals[p]
            if n == 0:
                out[m] = 1.0
                continue
            zj, wj, sj = zg[p, :n], wg[p, :n], sig_eff[p, :n]
            wbar = wj.mean()
            d = (z[m][:, None] - zj[None, :]) / sj[None, :]
            dens = ((wj / (np.sqrt(2 * np.pi) * sj))[None, :]
                    * np.exp(-0.5 * d ** 2)).sum(1) / wbar
            out[m] = mp[m] / (dens + mp[m])
        return out

    # ------------------------------------------------------------ events
    with h5py.File(C.GW_259) as f:
        ra, dec, dL = f["ra"][...], f["dec"][...], f["dL"][...]
    nev, nsamp = 259, ra.size // 259
    zev_grid = np.linspace(1e-4, 6.0, 4000)
    dc6, E6 = _dc_of_z(zev_grid, ANCHOR_H0, C.OM0)
    dL_grid = dc6 * (1 + zev_grid)
    z_of = lambda d: np.interp(d, dL_grid, zev_grid)
    pix = hp.ang2pix(NSIDE, np.pi / 2 - dec, ra)
    z_s = z_of(dL)
    insupp = occ[pix] & (z_s <= C.Z_DEPTH)
    print(f"[events] {insupp.sum()}/{insupp.size} PE samples in-support "
          f"({100 * insupp.mean():.3f}%)")

    m_s = np.zeros(z_s.shape)
    if insupp.any():
        m_s[insupp] = miss_frac(pix[insupp], z_s[insupp])
    phi = (m_s * insupp).reshape(nev, nsamp).mean(1)      # (259,)

    T_ev = np.zeros((M_SPH, M_Z))
    idx = np.where(insupp)[0]
    if idx.size:
        Ax = _A(np.column_stack(hp.pix2vec(NSIDE, pix[idx])))
        Bx = _B(z_s[idx])
        T_ev = Ax.T @ ((m_s[idx] / nsamp)[:, None] * Bx)  # sum_i phi_i Phi_i

    # ------------------------------------------------------------ selection
    from darksirens.core.types import CosmoParams, SurveyParams
    from darksirens.gw.populations import pop_model_parser
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.utils import log_target_density_base_and_z

    with h5py.File(C.INJ_PLAIN) as f:
        i_ra, i_dec = f["ra"][...], f["dec"][...]
        i_m1, i_m2 = f["m1det"][...], f["m2det"][...]
        i_dL, i_ch, i_pd = f["dL"][...], f["chieff"][...], f["pdraw"][...]
    log_p_pop = pop_model_parser(pop_model="gwtc5_fiducial_bpl2peaks")
    pop_params = jnp.asarray(
        get_fixed_population_params("gwtc5_fiducial_bpl2peaks"))
    cosmo = CosmoParams(H0=ANCHOR_H0, Om0=C.OM0, w0=-1.0, wa=0.0)
    survey = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0,
                          b_miss=0.0, alpha_miss=1.0)
    base, zi = log_target_density_base_and_z(
        jnp.asarray(i_m1), jnp.asarray(i_m2 / np.maximum(i_m1, 1e-300)),
        jnp.asarray(i_dL), jnp.asarray(i_ch),
        jnp.zeros(i_m1.size, dtype=jnp.int32), jnp.asarray(i_pd),
        cosmo, survey, pop_params, None, log_p_pop, spin=None)
    base = np.asarray(base)
    zi = np.asarray(zi)
    ok = np.isfinite(base) & (i_pd > 0)
    wbase = np.where(ok, np.exp(base - np.nanmax(base[ok])), 0.0)

    i_pix = hp.ang2pix(NSIDE, np.pi / 2 - i_dec, i_ra)
    # total prior density (count units): obs spikes + missing everywhere
    # (Q = 1; off-footprint pixels carry the full homogeneous missing branch,
    #  in-footprint the (1 - C) one; above z_depth C = 0 -> full dN_exp).
    zf6 = np.linspace(1e-4, 6.0, 2000)
    dc_, E_ = _dc_of_z(zf6, ANCHOR_H0, C.OM0)
    dV6 = 4 * np.pi * (CLIGHT / ANCHOR_H0) * dc_ ** 2 / E_
    dN6_pix = n0 * (1 + zf6) ** delta * dV6 / N_PIX
    C6 = np.zeros(zf6.shape)
    mlo = zf6 <= C.Z_DEPTH
    from darksirens.redshift.selection import c_sel_gaussian as _cg
    C6[mlo] = np.asarray(_cg(jnp.asarray(zf6[mlo]), theta["m_lim"],
                             theta["M0hat"], theta["sigma_M"], ANCHOR_H0,
                             C.OM0, k_corr_coeffs=kcorr))
    miss_foot = np.interp(zi, zf6, (1 - C6) * dN6_pix)
    # occupied pixels see (1-C); empty ones see full dN_exp (C = 0 there)
    full_dN = np.interp(zi, zf6, dN6_pix)
    miss_dens = np.where(occ[i_pix], miss_foot, full_dN)

    obs_dens = np.zeros(zi.shape)
    sel_kde = occ[i_pix] & (zi <= C.Z_DEPTH + 0.05)
    if sel_kde.any():
        # obs spikes only matter near catalog support
        mfrac = miss_frac(i_pix[sel_kde], np.clip(zi[sel_kde], 0, C.Z_DEPTH))
        # recover obs density from the fraction: m = miss/(obs+miss)
        mp = np.interp(np.clip(zi[sel_kde], 0, C.Z_DEPTH), zf, miss_pix)
        obs_dens[sel_kde] = np.where(mfrac > 0, mp * (1 - mfrac) / np.maximum(mfrac, 1e-300), 0.0)
    w_tot = wbase * (obs_dens + miss_dens)
    ins_i = occ[i_pix] & (zi <= C.Z_DEPTH)
    w_miss = np.where(ins_i, wbase * miss_dens, 0.0)
    Z_tot = w_tot.sum()
    f_missing_sel = w_miss.sum() / Z_tot
    print(f"[selection] in-support missing fraction of mu: "
          f"{f_missing_sel:.6e}")

    T_sel = np.zeros((M_SPH, M_Z))
    j = np.where(w_miss > 0)[0]
    if j.size:
        Aj = _A(np.column_stack(hp.pix2vec(NSIDE, i_pix[j])))
        Bj = _B(zi[j])
        T_sel = Aj.T @ ((w_miss[j] / Z_tot)[:, None] * Bj)

    # ------------------------------------------------------------ sigma
    a_mat = B_GW * (T_ev - 259.0 * T_sel)                 # (M_SPH, M_Z)
    a_norm = np.linalg.norm(a_mat)
    at = Us.T @ a_mat @ Uz
    sigma = float(np.sqrt(np.sum(at ** 2 / denom)))
    out = dict(anchor_H0=ANCHOR_H0, b_GW=B_GW,
               kernel=dict(ls_sph=LS_SPH, ls_z=LS_Z, M_sph=M_SPH, M_z=M_Z,
                           jitter=JITTER, z_node_hi=C.Z_DEPTH),
               sum_phi=float(phi.sum()), phi_max=float(phi.max()),
               n_events_phi_pos=int((phi > 0).sum()),
               pe_insupport_sample_frac=float(insupp.mean()),
               f_missing_sel=float(f_missing_sel),
               a_euclid_norm=float(a_norm), sigma_Hnorm=sigma,
               git_sha=C.git_sha())
    with open(PR0_DIR / "sigma_results.json", "w") as f:
        json.dump(out, f, indent=1)
    np.savez(PR0_DIR / "a_vector.npz", a_mat=a_mat, phi=phi,
             S_sph=S_sph, S_z=S_z,
             kernel=np.array([LS_SPH, LS_Z, M_SPH, M_Z, JITTER]))
    print(f"[pr0-sigma] sum_i phi_i = {phi.sum():.6e}  (K0 threshold 1e-3)")
    print(f"[pr0-sigma] ||a||_2 = {a_norm:.6e}   sigma_H = {sigma:.6e} nats")
    print(f"[pr0-sigma] wrote {PR0_DIR / 'sigma_results.json'}")


if __name__ == "__main__":
    main()
