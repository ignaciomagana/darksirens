"""PR-5b task 1: the closed-form member-spread prediction, recomputed.

PLAN eq. (6) (§6.5 item 5, with the v4 §0.5-D4 correction to the inner
product):

    a      =  b_GW * ( sum_i phi_i Phi_i  -  N_obs * <Phi>_sel )          (6)
    sigma  =  || L_H^{-1} a ||_2          [the H^{-1} norm, NOT Euclidean]

``phi_i`` is event ``i``'s missing-branch fraction of its total prior mass
restricted to the in-support region (occupied footprint pixels, ``z <=
z_depth``); ``<Phi>_sel`` is the selection-weighted (mu-weighted) basis
average over the same region; ``L_H`` is the lower Cholesky of the count
channel's Laplace Fisher.  The members are ``xi_m = xi_hat + L_H^{-T} g_m``
(``latent_counts.laplace_draws``), so to first order in the field
``ll_m = ll(xi_hat) + a . L_H^{-T} g_m`` and ``Var_m(ll_m) = a^T H^{-1} a``
exactly — which is why the norm is the ``H^{-1}`` one.  ``H >= I`` (it is a
Fisher information plus the unit prior), so the Euclidean form
``||a||_2`` is an UPPER bound and systematically over-predicts sigma.  That
matters because sigma feeds an EXPONENTIAL ``M_draw`` requirement through
PLAN §6.5's bias table, so both norms and their ratio are reported here.

WHAT IS NEW RELATIVE TO PR-0 (experiments/field_level_plan/pr0/compute_sigma.py)
-------------------------------------------------------------------------------
PR-0 computed the same estimand but had no anchor artifact to lean on (PR-4
did not exist yet), so it SYNTHESIZED an approximate Fisher: the
unconditioned Poisson one, with a sky-uniform missing base, which factorizes
exactly as ``H = I + b^2 S_sph (x) S_z`` and can be inverted through the two
factor eigenbases without ever assembling an M x M matrix.  PR-0 flagged
this as conservative (the shell-total-conditioned MULTINOMIAL Fisher that
the code actually ships is smaller by the rank-1 subtraction of eq. (3), so
``H^{-1}`` is larger and sigma would grow).

This script keeps PR-0's event/injection side verbatim — same missing-branch
KDE, same uniform-PE-weight approximation, same population-weighted
injection composition — so that any difference in sigma is attributable to
ONE thing, and reproduces PR-0's factored arm as a control.  The headline
number instead uses the REAL ``H_chol`` from the PR-4/PR-5 anchor artifact:
the shell-total-conditioned multinomial Fisher of ``latent_counts``
eq. (3), at the real MAP ``xi_hat``, on the real 30470-pixel DESI footprint
with per-pixel ``f_p``, rank-1 term included.  Three arms are therefore
reported:

    sigma_eucl      ||a||_2                      (what PLAN v3 would have used)
    sigma_pr0       PR-0's factored Poisson H     (the reproduction control)
    sigma_anchor    the anchor's H_chol           (THE NUMBER)

The basis is not re-derived: it is rebuilt with
``latent_field.build_latent_basis`` at the anchor's own ``basis_meta``, and
the rebuild is VERIFIED against the artifact by re-evaluating
``latent_counts.gradient`` at the stored ``xi_hat`` with the stored ``W``,
``counts``, ``f_p`` and ``b_gal`` and comparing ``grad_inf`` to the value
the build stamped (the P6 gate residual, ~1e-10).  If the rebuilt basis
were off by so much as a whitening convention this check would blow up by
orders of magnitude, so it is a hard gate here.  With ``--verify-hessian``
the full eq. (3) Hessian is reassembled and compared to ``L L^T`` as well.

H0 DEPENDENCE.  ``a`` is theta-dependent (the events' redshifts, the
injections' weights and ``C(z; H0)`` all move with ``H0``), so sigma is a
function of ``H0`` and PLAN P14 gates the theta-VARIATION of the bias, not
its level.  ``H_chol`` is NOT re-solved per node: at rung 0 the shipped seam
consumes ONE anchor built at ``theta_ref`` (PLAN §1.7 — rung 1 is the
linear-response correction, and K9's benign branch retired it), and the
count channel is in any case H0-free by construction (conditioning on the
shell totals ``T_g`` deletes the monopole and with it ``dV/dz ~ (c/H0)^3``;
``latent_counts`` module docstring).  So the same ``L_H`` is correct at
every node and only ``a`` moves.

Stated approximations, inherited from PR-0 and unchanged so the comparison
is apples-to-apples:
  * event samples enter with uniform (posterior) weights, not the full
    population importance weights.  PR-0 measured the size of this: the
    sample-wise construction gives ``sum_i phi_i = 1.47`` where the
    production likelihood itself gives 0.993, so the event term is
    over-weighted by ~1.48x.  A rescaled arm is reported alongside.
  * injection weights use ``log_target_density_base_and_z`` (population +
    pdraw + Jacobians) times this script's own catalog-prior densities,
    reproducing the field-weighting composition analytically rather than
    through the production evaluator.
  * ``b_GW = 1`` (the anchor's ``b_gal``); sigma is exactly linear in
    ``b_GW``, so any other value rescales trivially.

Run from experiments/desi_full259:
    python ../field_level_plan/pr5b/predict_sigma.py [--h0-scan] [--verify-hessian]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PR5B_DIR = Path(__file__).resolve().parent
PLAN_DIR = PR5B_DIR.parent
FULL259 = PLAN_DIR.parent / "desi_full259"
sys.path.insert(0, str(FULL259))

import common as C  # noqa: E402   (pins DARKSIRENS_ZMAX=6.0; must be first)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import h5py  # noqa: E402
import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

ANCHOR_H0 = 67.74
B_GW = 1.0
N_OBS = 259
CLIGHT = 299792.458            # km/s
NSIDE = 64
N_PIX = 12 * NSIDE ** 2

#: PLAN §6.5 item 1 asks for the measurement at these M values.
M_LADDER = (4, 8, 16, 32, 64, 128, 256)

#: PR-0's two independent measurements of ``sum_i phi_i`` at the anchor: 0.993
#: through the production likelihood itself (the killer-Q table, exact code
#: path) and 1.467 from the sample-wise construction this script reuses.  The
#: gap is the population/prior reweighting the sample-wise form skips.  ``a``
#: is LINEAR in the event term, so rescaling it by this ratio is an exact
#: correction of that one contribution -- reported as a sensitivity arm, never
#: as the headline (PR-0 quoted the unrescaled number).
PR0_SUMPHI_LIKELIHOOD = 0.9934917605336928
PR0_SUMPHI_SAMPLEWISE = 1.4671212117641965

#: chunk for the (n_sample, M_sph) basis gathers -- 261k in-support PE samples
#: x 315 f64 is 0.66 GB in one shot, which is pointless when the reduction is
#: a plain accumulation.
CHUNK = 50_000

#: candidate anchor artifacts, best first (PR-5's post-eq.(4)-f32-fix rebuild,
#: then PR-4's original).  H_chol is untouched by that fix — it changed only
#: how the (A, B) sky moments are contracted — but provenance is recorded.
ANCHOR_CANDIDATES = (
    PLAN_DIR / "pr5" / "latent_anchor_v2a.h5",
    PLAN_DIR / "pr4" / "latent_anchor_a.h5",
)


# ------------------------------------------------------------------ cosmology
def _dc_of_z(zgrid, H0, Om0):
    """Comoving distance on ``zgrid`` by the same trapezoid PR-0 used."""
    E = np.sqrt(Om0 * (1 + zgrid) ** 3 + (1 - Om0))
    integ = 1.0 / E
    dc = np.concatenate([[0.0], np.cumsum(
        0.5 * (integ[1:] + integ[:-1]) * np.diff(zgrid))]) * CLIGHT / H0
    return dc, E


# ------------------------------------------------------------ the bias table
def bias_table(sigma: float, m_values=M_LADDER) -> list[dict]:
    """PLAN §6.5's sharp form: ``bias = -(e^{sigma^2} - 1) / (2 M)`` nats and
    ``E[ESS]/M ~ exp(-sigma^2)``.  ``expm1`` keeps this honest at the 1e-4
    sigma the anchor predicts, where ``e^{sigma^2} - 1`` underflows a naive
    subtraction entirely."""
    v = float(np.expm1(sigma ** 2))
    return [dict(M=int(m), bias_nats=-v / (2.0 * m)) for m in m_values]


def m_for_bias(sigma: float, eps: float) -> float:
    """``M > (e^{sigma^2} - 1) / (2 eps)`` for an ``eps``-nat bias."""
    return float(np.expm1(sigma ** 2) / (2.0 * eps))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchor", default=None,
                    help="latent anchor artifact (default: first existing of "
                         "pr5/latent_anchor_v2a.h5, pr4/latent_anchor_a.h5)")
    ap.add_argument("--h0-scan", action="store_true",
                    help="also predict sigma(H0) on a coarse node set, so the "
                         "THETA-VARIATION P14 gates is predicted too")
    ap.add_argument("--h0-nodes", default="plan33",
                    help="'plan33' = PLAN §6.5 item 1's 33 nodes across "
                         "[20, 140], or a comma-separated list")
    ap.add_argument("--verify-hessian", action="store_true",
                    help="reassemble eq. (3) and compare H to L L^T (slow)")
    ap.add_argument("--out", default=str(PR5B_DIR / "sigma_prediction.json"))
    args = ap.parse_args(argv)

    from darksirens.redshift.latent_counts import (
        CountOperator, gradient, hessian_separable)
    from darksirens.redshift.latent_field import build_latent_basis
    from darksirens.redshift.selection import (
        c_sel_gaussian, load_selection_fit_json)

    t_all = time.time()

    # ------------------------------------------------------------- anchor
    anchor_path = Path(args.anchor) if args.anchor else None
    if anchor_path is None:
        for cand in ANCHOR_CANDIDATES:
            if cand.exists():
                anchor_path = cand
                break
    if anchor_path is None or not anchor_path.exists():
        raise SystemExit(f"no anchor artifact found among "
                         f"{[str(p) for p in ANCHOR_CANDIDATES]}")
    with h5py.File(anchor_path) as f:
        g = f["latent_field"]
        L_H = g["H_chol"][...]                       # (M, M) lower
        xi_hat = g["xi_hat"][...]
        # NOTE (measured, PR-5b): the dataset NAMED ``g_members`` does NOT hold
        # the standard normals ``g_m`` that ``build_latent_field.py``'s header
        # advertises -- it holds ``draws``, i.e. the members ``xi_m`` themselves,
        # byte-identical to ``Xi_members`` reshaped (verified: per-row sd 2.52,
        # matching ``xi_hat``'s 2.46, not 1.0).  The build writes
        # ``create_dataset("g_members", data=draws)`` at build_latent_field.py:247.
        # Harmless for every current consumer (the seam reads ``row_fac`` /
        # ``Xi_members``), but a trap for anything that trusts the name, so this
        # script recovers the member OFFSETS the honest way,
        # ``xi_m - xi_hat = L_H^{-T} g_m``, which is antithetic as designed
        # (verified: rows 4..7 = -rows 0..3 exactly).
        xi_members = g["g_members"][...]             # (M_draw, M) = the draws
        fit_pix = g["fit_pixels"][...]
        f_p = g["completeness"][...]
        counts = g["counts"][...]
        W_ref = g["shell_response"][...]
        z_edges = g["z_count_edges"][...]
        basis_meta = json.loads(g.attrs["basis_meta"])
        theta_ref = json.loads(g.attrs["theta_ref"])
        anchor_sha = g.attrs["sha256"]
        anchor_gradinf = float(g.attrs["grad_inf"])
        b_gal = float(g.attrs["b_gal"])
        nside_a = int(g.attrs["nside"])
    M_SPH = int(basis_meta["M_sph"])
    M_Z = int(basis_meta["M_z"])
    LS_SPH = float(basis_meta["ls_sph"])
    LS_Z = float(basis_meta["ls_z"])
    Z_NODE_HI = float(basis_meta["z_node_hi"])
    JITTER = float(basis_meta["j_sph"])
    M_RANK = M_SPH * M_Z
    assert nside_a == NSIDE and L_H.shape == (M_RANK, M_RANK)
    print(f"[anchor] {anchor_path}")
    print(f"[anchor] sha256={anchor_sha}  M={M_RANK} ({M_SPH} x {M_Z})  "
          f"b_gal={b_gal}  n_fit={fit_pix.size}  grad_inf={anchor_gradinf:.2e}")
    print(f"[anchor] theta_ref={theta_ref}  basis={basis_meta}")

    # ------------------------------------------------------- basis (rebuilt)
    # All-sky sphere rows so events/injections can be evaluated at any pixel;
    # the footprint block is the anchor's own fit_pixels, in its own order.
    z_fine = np.linspace(1e-4, Z_NODE_HI, 400)       # W's grid (build-time)
    pix_vec_all = np.column_stack(hp.pix2vec(NSIDE, np.arange(N_PIX)))
    from darksirens.redshift.grid import zgrid as _zgrid
    z_sub = np.asarray(_zgrid[_zgrid <= Z_NODE_HI])
    basis = build_latent_basis(
        pix_vec_all, np.log1p(z_sub), n_inducing_sphere=M_SPH,
        n_inducing_z=M_Z, z_node_hi=Z_NODE_HI, ls_sph=LS_SPH, ls_z=LS_Z,
        zeta_fine=np.log1p(z_fine), footprint_rows=fit_pix)
    phi_sph_all = np.asarray(basis.phi_sph)          # (N_PIX, M_SPH)
    L_z = np.asarray(basis.L_z)
    zeta_nodes = np.linspace(0.0, np.log1p(Z_NODE_HI), M_Z)

    def phi_z_at(z):
        """The radial factor rows at arbitrary z — exactly ``_phi_z`` inside
        ``build_latent_basis`` (same L_z, same nodes, same kernel)."""
        zeta = np.log1p(np.asarray(z, dtype=float))
        k = np.exp(-0.5 * (zeta[:, None] - zeta_nodes[None, :]) ** 2
                   / LS_Z ** 2)
        return solve_triangular(L_z, k.T, lower=True).T

    # --- HARD GATE: the rebuilt basis must reproduce the anchor's own solve.
    op = CountOperator(proj_sph=jnp.asarray(basis.proj_sph),
                       phi_shell=jnp.asarray(W_ref) @ basis.phi_z_fine,
                       counts=jnp.asarray(counts),
                       log_fp=jnp.log(jnp.maximum(jnp.asarray(f_p), 1e-300)),
                       bias=b_gal)
    gi = float(jnp.max(jnp.abs(gradient(jnp.asarray(xi_hat), op))))
    print(f"[gate] rebuilt-basis grad_inf at stored xi_hat = {gi:.3e} "
          f"(anchor stamped {anchor_gradinf:.3e})")
    if not (gi < 1e-6):
        raise SystemExit("basis rebuild does not reproduce the anchor solve; "
                         "refusing to quote a prediction against a basis that "
                         "is not the artifact's.")
    hess_rel = None
    if args.verify_hessian:
        H_re = np.asarray(hessian_separable(jnp.asarray(xi_hat), op))
        H_ar = L_H @ L_H.T
        hess_rel = float(np.max(np.abs(H_re - H_ar))
                         / np.max(np.abs(H_ar)))
        print(f"[gate] max rel |H_rebuilt - L L^T| = {hess_rel:.3e}")
        del H_re, H_ar

    # H spectrum: this is WHY the H^{-1} norm crushes (or fails to crush) a.
    # The shell-total conditioning of eq. (1') deletes one direction per shell
    # from the count channel outright (that is what conditioning on T_g DOES),
    # and the footprint's sky coverage leaves further sphere modes with almost
    # no count leverage; in every such direction H = I and the H^{-1} norm buys
    # NOTHING.  The prediction therefore lives or dies on how much of ``a``
    # lands in the prior-dominated block, so it is reported explicitly.
    evH, VH = np.linalg.eigh(L_H @ L_H.T)
    EIG_BANDS = (1.01, 2.0, 10.0, 100.0, 1e4)
    print(f"[H] eigenvalues in [{evH.min():.9f}, {evH.max():.4e}]; "
          f"{int((evH < 1.01).sum())}/{M_RANK} modes essentially unconstrained "
          f"(eig < 1.01, i.e. prior-dominated)")

    # ----------------------------------------------------------- static data
    sel = load_selection_fit_json(C.FIT_JSON)
    cal = json.load(open(C.DATA_DIR / "n0_calibration.json"))
    n0 = 10.0 ** float(cal["log10n0"])
    delta = float(cal["delta"])
    kcorr = tuple(sel["k_corr_coeffs"])
    m_lim, M0hat, sigma_M = (float(sel["m_lim"]), float(sel["M0hat"]),
                             float(sel["sigma_M"]))

    with h5py.File(C.SURVEY_N64) as f:
        ngals = f["ngals"][...]
        zg = f["zgals"][...]
        wg = f["wgals"][...]
        dzg = f["dzgals"][...]
    occ = ngals > 0
    assert np.array_equal(np.where(occ)[0], np.asarray(fit_pix)), \
        "survey occupancy differs from the anchor's fit_pixels"
    sig_eff = np.maximum(np.sqrt(dzg ** 2 + 0.003 ** 2), 1e-4)

    with h5py.File(C.GW_259) as f:
        ev_ra, ev_dec, ev_dL = f["ra"][...], f["dec"][...], f["dL"][...]
    n_samp = ev_ra.size // N_OBS
    ev_pix = hp.ang2pix(NSIDE, np.pi / 2 - ev_dec, ev_ra)
    print(f"[events] {N_OBS} events x {n_samp} PE samples")

    with h5py.File(C.INJ_PLAIN) as f:
        i_ra, i_dec = f["ra"][...], f["dec"][...]
        i_m1, i_m2 = f["m1det"][...], f["m2det"][...]
        i_dL, i_ch, i_pd = f["dL"][...], f["chieff"][...], f["pdraw"][...]
    i_pix = hp.ang2pix(NSIDE, np.pi / 2 - i_dec, i_ra)
    print(f"[injections] {i_dL.size} detected injections")

    from darksirens.core.types import CosmoParams, SurveyParams
    from darksirens.gw.populations import pop_model_parser
    from darksirens.gw.populations.registry import get_fixed_population_params
    from darksirens.inference.utils import log_target_density_base_and_z
    log_p_pop = pop_model_parser(pop_model="gwtc5_fiducial_bpl2peaks")
    pop_params = jnp.asarray(
        get_fixed_population_params("gwtc5_fiducial_bpl2peaks"))
    survey = SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0,
                          b_miss=0.0, alpha_miss=1.0)

    # ------------------------------------------------ PR-0's factored Fisher
    # Reproduction control: the unconditioned Poisson Fisher with a sky-uniform
    # missing base, H = I + b^2 S_sph (x) S_z, diagonalized through the two
    # factor eigenbases (never assembled).  Built at the ANCHOR cosmology,
    # exactly as PR-0 did, and reused at every H0 node so the control is the
    # same object PR-0 quoted.
    A_fit = np.asarray(basis.proj_sph)
    S_sph = A_fit.T @ A_fit
    zf0 = np.linspace(1e-4, Z_NODE_HI, 400)
    dc0, E0 = _dc_of_z(zf0, ANCHOR_H0, C.OM0)
    dVdz0 = 4.0 * np.pi * (CLIGHT / ANCHOR_H0) * dc0 ** 2 / E0
    Cz0 = np.asarray(c_sel_gaussian(jnp.asarray(zf0), m_lim, M0hat, sigma_M,
                                    ANCHOR_H0, C.OM0, k_corr_coeffs=kcorr))
    dNexp0 = n0 * (1 + zf0) ** delta * dVdz0 / N_PIX
    Bf0 = phi_z_at(zf0)
    S_z = (Bf0 * (Cz0 * dNexp0 * np.gradient(zf0))[:, None]).T @ Bf0
    es, Us = np.linalg.eigh(S_sph)
    ez, Uz = np.linalg.eigh(S_z)
    denom_pr0 = 1.0 + B_GW ** 2 * np.outer(es, ez)
    print(f"[pr0-fisher] S_sph tr={np.trace(S_sph):.3e}  "
          f"S_z tr={np.trace(S_z):.3e}  max eig product={denom_pr0.max()-1:.3e}")

    # ---------------------------------------------------------------- core
    def predict_at(H0: float) -> dict:
        """Everything in eq. (6) at one H0, plus the three sigma arms."""
        t0 = time.time()
        # --- catalog-prior densities, per pixel, count units --------------
        zf = np.linspace(1e-4, Z_NODE_HI, 400)
        dc, E = _dc_of_z(zf, H0, C.OM0)
        dVdz = 4.0 * np.pi * (CLIGHT / H0) * dc ** 2 / E
        Cz = np.asarray(c_sel_gaussian(jnp.asarray(zf), m_lim, M0hat, sigma_M,
                                       H0, C.OM0, k_corr_coeffs=kcorr))
        dNexp_pix = n0 * (1 + zf) ** delta * dVdz / N_PIX
        miss_pix = (1.0 - Cz) * dNexp_pix

        def miss_frac(pix, z):
            """``m = dN_miss / (dN_obs + dN_miss)`` at ``(pix, z)``, count
            units, with the observed branch a per-pixel photo-z KDE of the
            catalog galaxies (PR-0's estimator, verbatim)."""
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

        # --- event term: sum_i phi_i Phi_i --------------------------------
        zev_grid = np.linspace(1e-4, 6.0, 4000)
        dc6, E6 = _dc_of_z(zev_grid, H0, C.OM0)
        dL_grid = dc6 * (1 + zev_grid)
        z_s = np.interp(ev_dL, dL_grid, zev_grid)
        insupp = occ[ev_pix] & (z_s <= Z_NODE_HI)
        m_s = np.zeros(z_s.shape)
        if insupp.any():
            m_s[insupp] = miss_frac(ev_pix[insupp], z_s[insupp])
        phi = (m_s * insupp).reshape(N_OBS, n_samp).mean(1)      # (259,)
        idx = np.where(insupp)[0]
        T_ev = np.zeros((M_SPH, M_Z))
        for k0 in range(0, idx.size, CHUNK):
            k = idx[k0:k0 + CHUNK]
            Ax = phi_sph_all[ev_pix[k]]
            Bx = phi_z_at(z_s[k])
            T_ev += Ax.T @ ((m_s[k] / n_samp)[:, None] * Bx)

        # --- selection term: <Phi>_sel ------------------------------------
        cosmo = CosmoParams(H0=H0, Om0=C.OM0, w0=-1.0, wa=0.0)
        base, zi = log_target_density_base_and_z(
            jnp.asarray(i_m1), jnp.asarray(i_m2 / np.maximum(i_m1, 1e-300)),
            jnp.asarray(i_dL), jnp.asarray(i_ch),
            jnp.zeros(i_m1.size, dtype=jnp.int32), jnp.asarray(i_pd),
            cosmo, survey, pop_params, None, log_p_pop, spin=None)
        base, zi = np.asarray(base), np.asarray(zi)
        ok = np.isfinite(base) & (i_pd > 0)
        wbase = np.where(ok, np.exp(base - np.nanmax(base[ok])), 0.0)

        # total prior density in count units: observed spikes + the missing
        # branch everywhere.  Off-footprint pixels carry the FULL homogeneous
        # missing branch, in-footprint the (1 - C) one, and above z_depth
        # C = 0 so the whole dN_exp is missing.
        zf6 = np.linspace(1e-4, 6.0, 2000)
        dc_, E_ = _dc_of_z(zf6, H0, C.OM0)
        dV6 = 4 * np.pi * (CLIGHT / H0) * dc_ ** 2 / E_
        dN6_pix = n0 * (1 + zf6) ** delta * dV6 / N_PIX
        C6 = np.zeros(zf6.shape)
        mlo = zf6 <= Z_NODE_HI
        C6[mlo] = np.asarray(c_sel_gaussian(
            jnp.asarray(zf6[mlo]), m_lim, M0hat, sigma_M, H0, C.OM0,
            k_corr_coeffs=kcorr))
        miss_foot = np.interp(zi, zf6, (1 - C6) * dN6_pix)
        full_dN = np.interp(zi, zf6, dN6_pix)
        miss_dens = np.where(occ[i_pix], miss_foot, full_dN)

        obs_dens = np.zeros(zi.shape)
        sel_kde = occ[i_pix] & (zi <= Z_NODE_HI + 0.05)
        if sel_kde.any():
            zc = np.clip(zi[sel_kde], 0, Z_NODE_HI)
            mfrac = miss_frac(i_pix[sel_kde], zc)
            mp = np.interp(zc, zf, miss_pix)
            obs_dens[sel_kde] = np.where(
                mfrac > 0, mp * (1 - mfrac) / np.maximum(mfrac, 1e-300), 0.0)
        w_tot = wbase * (obs_dens + miss_dens)
        ins_i = occ[i_pix] & (zi <= Z_NODE_HI)
        w_miss = np.where(ins_i, wbase * miss_dens, 0.0)
        Z_tot = w_tot.sum()
        f_missing_sel = w_miss.sum() / Z_tot

        T_sel = np.zeros((M_SPH, M_Z))
        jj = np.where(w_miss > 0)[0]
        for k0 in range(0, jj.size, CHUNK):
            k = jj[k0:k0 + CHUNK]
            Aj = phi_sph_all[i_pix[k]]
            Bj = phi_z_at(zi[k])
            T_sel += Aj.T @ ((w_miss[k] / Z_tot)[:, None] * Bj)

        # --- eq. (6) and the three norms ----------------------------------
        a_mat = B_GW * (T_ev - N_OBS * T_sel)                    # (M_SPH, M_Z)
        a_vec = a_mat.reshape(-1)     # i = i_sph * M_z + i_z (row_factor's own
                                      # flattening; latent_field.row_factor)
        sigma_eucl = float(np.linalg.norm(a_vec))
        u = solve_triangular(L_H, a_vec, lower=True)
        sigma_anchor = float(np.linalg.norm(u))
        # The SHIPPED members are xi_m = xi_hat + L_H^{-T} g_m with antithetic
        # g_m, so the first-order per-member offsets are known exactly, not
        # just their population sd:
        #     ll_m - ll(xi_hat) = a . (xi_m - xi_hat) = a . L_H^{-T} g_m .
        # These are the numbers PR-5b's measurement phase compares against
        # member-for-member; ``sigma`` above is their population sd, and
        # ``dll_sd`` is the sd actually realized by the 8 stored draws (an
        # M_draw = 8 antithetic estimate, so it is a 4-sample quantity and is
        # expected to scatter around ``sigma`` by tens of percent).
        dll = (xi_members - xi_hat[None, :]) @ a_vec              # (M_draw,)
        at = Us.T @ a_mat @ Uz
        sigma_pr0 = float(np.sqrt(np.sum(at ** 2 / denom_pr0)))

        # the event side alone, and the selection side alone, in the H^{-1}
        # norm -- how much of the (large) Euclidean a survives, and whether
        # the -N_obs <Phi>_sel subtraction is doing the cancelling.
        s_ev = float(np.linalg.norm(solve_triangular(
            L_H, (B_GW * T_ev).reshape(-1), lower=True)))
        s_sel = float(np.linalg.norm(solve_triangular(
            L_H, (B_GW * N_OBS * T_sel).reshape(-1), lower=True)))

        # sensitivity arm: the event term rescaled to the sum_phi the
        # PRODUCTION likelihood reports (PR-0 item 2a), assuming the ratio it
        # measured at the anchor carries to this node.  a is linear in T_ev.
        rw = PR0_SUMPHI_LIKELIHOOD / max(float(phi.sum()), 1e-300)
        sigma_rw = float(np.linalg.norm(solve_triangular(
            L_H, (B_GW * (rw * T_ev - N_OBS * T_sel)).reshape(-1), lower=True)))

        # where sigma^2 comes from, band by band in the H spectrum:
        # sigma^2 = sum_k (v_k . a)^2 / e_k.
        ck = VH.T @ a_vec
        s2 = ck ** 2 / evH
        bands = []
        lo = 0.0
        for hi in EIG_BANDS + (np.inf,):
            sel_b = (evH >= lo) & (evH < hi)
            bands.append(dict(eig_lo=float(lo),
                              eig_hi=(None if hi == np.inf else float(hi)),
                              n_modes=int(sel_b.sum()),
                              frac_a2=float(ck[sel_b] @ ck[sel_b]
                                            / (ck @ ck)),
                              frac_sigma2=float(s2[sel_b].sum() / s2.sum())))
            lo = hi

        dt = time.time() - t0
        print(f"[H0={H0:7.2f}] sum_phi={phi.sum():.6f}  "
              f"insupp={insupp.mean()*100:.3f}%  f_miss_sel={f_missing_sel:.4e}"
              f"  ||a||={sigma_eucl:.6e}  sigma_H={sigma_anchor:.6e}  "
              f"sigma_pr0={sigma_pr0:.6e}  [{dt:.1f}s]", flush=True)
        return dict(
            H0=float(H0), sum_phi=float(phi.sum()), phi_max=float(phi.max()),
            n_events_phi_pos=int((phi > 0).sum()),
            pe_insupport_sample_frac=float(insupp.mean()),
            f_missing_sel=float(f_missing_sel),
            sigma_euclid=sigma_eucl, sigma_anchor=sigma_anchor,
            sigma_pr0_factored=sigma_pr0,
            ratio_euclid_over_anchor=sigma_eucl / sigma_anchor,
            ratio_euclid_over_pr0=sigma_eucl / sigma_pr0,
            sigma_event_only=s_ev, sigma_sel_only=s_sel,
            pe_reweight_factor=float(rw), sigma_anchor_pe_reweighted=sigma_rw,
            eig_bands=bands, seconds=dt, a_mat=a_mat,
            dll_members=[float(x) for x in dll],
            dll_sd=float(np.std(dll)),
            # the M_draw = 8 estimator's own first-order bias, evaluated on the
            # stored draws rather than from the lognormal formula:
            # log(mean_m e^{dll_m}) - 0 , which is what LSE_m ll_m - log M
            # minus ll(xi_hat) will return at first order (PLAN P17 arm b).
            p17b_realized_nats=float(
                np.log(np.mean(np.exp(dll - dll.max()))) + dll.max()))

    # -------------------------------------------------------------- run it
    nodes = [ANCHOR_H0]
    if args.h0_scan:
        raw = (np.linspace(20.0, 140.0, 33) if args.h0_nodes == "plan33"
               else np.array([float(x) for x in args.h0_nodes.split(",")]))
        nodes = sorted(set(raw.tolist()) | {ANCHOR_H0})
    res = [predict_at(h) for h in nodes]
    anchor_res = [r for r in res if abs(r["H0"] - ANCHOR_H0) < 1e-9][0]
    np.savez(PR5B_DIR / "a_vector_anchor.npz",
             a_mat=anchor_res["a_mat"], H_eigvals=evH)
    for r in res:
        r.pop("a_mat")

    sig = anchor_res["sigma_anchor"]
    print("\n[bands] H-eigenvalue band | modes | frac ||a||^2 | frac sigma^2")
    for bd in anchor_res["eig_bands"]:
        print(f"    [{bd['eig_lo']:>8.2f}, "
              f"{'inf' if bd['eig_hi'] is None else format(bd['eig_hi'], '>10.2f')}"
              f") {bd['n_modes']:5d}   {bd['frac_a2']:.6f}   "
              f"{bd['frac_sigma2']:.6f}")
    tbl = bias_table(sig)
    ess = float(np.exp(-sig ** 2))
    print()
    print(f"[prediction] sigma_H(anchor) = {sig:.6e} nats")
    print(f"[prediction] E[ESS]/M = exp(-sigma^2) = {ess:.12f}")
    print(f"[prediction] M for 0.1-nat bias = {m_for_bias(sig, 0.1):.3e}")
    for row in tbl:
        print(f"    M = {row['M']:4d}   bias = {row['bias_nats']:+.4e} nats")

    # theta-variation of the bias (P14's actual gate quantity), if scanned
    p14 = p14_bulk = None
    if len(res) > 1:
        smax = max(r["sigma_anchor"] for r in res)
        smin = min(r["sigma_anchor"] for r in res)
        s_all = np.array([r["sigma_anchor"] for r in res])
        h_all = np.array([r["H0"] for r in res])
        # PR-0 item 3 measured the CLEAN (variance-cap-lifted, fixed-population)
        # posterior peaking at H0 = 90 +/- 5, so the prior-wide P14 and the
        # posterior-bulk P14 are different questions and both are predicted.
        bulk = (h_all >= 75.0) & (h_all <= 105.0)

        def _p14(sub):
            out = {}
            for m in M_LADDER:
                b = -np.expm1(s_all[sub] ** 2) / (2.0 * m)
                b256 = -np.expm1(s_all[sub] ** 2) / (2.0 * 256)
                d = b - b256
                out[str(m)] = float(np.max(np.abs(d - d.mean())))
            return out

        p14 = _p14(np.ones(s_all.size, bool))
        p14_bulk = _p14(bulk)
        print(f"\n[P14] sigma(H0) range over {len(res)} nodes: "
              f"[{smin:.4e}, {smax:.4e}] nats")
        print(f"[P14] predicted max |(bias_M - bias_256) - mean| over the FULL "
              f"prior [20, 140]:")
        print("      " + "  ".join(f"M={m}:{p14[str(m)]:.3e}" for m in M_LADDER))
        print(f"[P14] the same over the clean posterior bulk H0 in [75, 105] "
              f"({int(bulk.sum())} nodes):")
        print("      " + "  ".join(f"M={m}:{p14_bulk[str(m)]:.3e}"
                                   for m in M_LADDER))
        ok_full = [m for m in M_LADDER if p14[str(m)] < 0.1]
        ok_bulk = [m for m in M_LADDER if p14_bulk[str(m)] < 0.1]
        print(f"[P14] predicted smallest M_draw meeting 0.1 nat: "
              f"full prior {min(ok_full) if ok_full else '>256'}; "
              f"posterior bulk {min(ok_bulk) if ok_bulk else '>256'}")

    out = dict(
        script="experiments/field_level_plan/pr5b/predict_sigma.py",
        anchor_artifact=str(anchor_path), anchor_sha256=str(anchor_sha),
        anchor_is_pr5_v2a=("latent_anchor_v2a" in anchor_path.name),
        anchor_grad_inf_stamped=anchor_gradinf,
        rebuilt_basis_grad_inf=gi, hessian_rel_err=hess_rel,
        basis_meta=basis_meta, theta_ref=theta_ref, b_gal=b_gal, b_GW=B_GW,
        M_rank=M_RANK, n_fit=int(fit_pix.size),
        H_eig_min=float(evH.min()), H_eig_max=float(evH.max()),
        H_n_modes_below_1p01=int((evH < 1.01).sum()),
        anchor_H0=ANCHOR_H0, nodes=res,
        sigma_anchor=sig, sigma_euclid=anchor_res["sigma_euclid"],
        sigma_pr0_factored=anchor_res["sigma_pr0_factored"],
        ratio_euclid_over_anchor=anchor_res["ratio_euclid_over_anchor"],
        pr0_reference=dict(sigma_Hnorm=6.018525089540707e-4,
                           a_euclid_norm=0.15577360748965957,
                           sum_phi=1.4671212117641965),
        ess_over_M=ess, M_for_0p1_nat=m_for_bias(sig, 0.1),
        bias_table=tbl, p14_predicted_theta_variation=p14,
        p14_predicted_theta_variation_bulk_75_105=p14_bulk,
        # PLAN P17 arm (b): LSE_m ll_m - log M - ll(xi_hat) -> 0.5 sigma^2
        p17b_target_nats=0.5 * sig ** 2,
        sigma_anchor_pe_reweighted=anchor_res["sigma_anchor_pe_reweighted"],
        p17b_realized_nats=anchor_res["p17b_realized_nats"],
        dll_members_M8=anchor_res["dll_members"],
        dll_sd_M8=anchor_res["dll_sd"],
        bias_table_pe_reweighted=bias_table(
            anchor_res["sigma_anchor_pe_reweighted"]),
        total_seconds=time.time() - t_all,
        git_sha=C.git_sha(),
        DARKSIRENS_ZMAX=C.ZMAX)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n[P17b] predicted (LSE_m ll_m - log M) - ll(xi_hat) at the shipped "
          f"M_draw = 8 members: {anchor_res['p17b_realized_nats']:+.6e} nats "
          f"(asymptotic 0.5 sigma^2 = {0.5 * sig ** 2:.6e})")
    print(f"[P17b] predicted per-member ll_m - ll(xi_hat), M_draw = 8: "
          + ", ".join(f"{x:+.4f}" for x in anchor_res["dll_members"]))
    print(f"[P17b] realized sd over the 8 stored draws = "
          f"{anchor_res['dll_sd']:.6e} (population sigma {sig:.6e})")
    print(f"\n[pr5b] wrote {args.out}  ({out['total_seconds']:.1f} s total)")


if __name__ == "__main__":
    main()
