"""The nside-16 anchor artifact for the closure mocks.

A faithful adaptation of ``darksirens/cli/build_latent_field.py`` -- every
numerical step is the shipped function (``counts_from_catalog``,
``build_latent_basis``, ``shell_response``, ``count_map_solve``, ``gradient``,
``dgrad_db``, ``sensitivity``, ``laplace_draws``,
``sky_moments``, ``sky_constant_coeffs``), and the output is the same
``/latent_field`` group with the same ``format_version``, so
``likelihood/factory.py``'s guard 1 and ``latent_q.load_latent_plan`` consume
it unmodified.  Three differences, all forced by mock scale and all recorded
in the artifact's own metadata:

**1. Shells are equal-COMOVING-VOLUME, not equal-z.**  The CLI uses
``np.linspace(0, z_depth, n_shells+1)`` plus occupancy guard 7 ("each shell
>= 1e4 galaxies and >= 500 occupied pixels").  At ``z_depth = 0.30`` the
lowest linear shell holds ``(1/12)^3 = 5.8e-4`` of the volume, so guard 7
demands ~1.7e7 CATALOGUED galaxies before the first shell clears 1e4 -- more
than the DESI union itself.  Equal-volume edges give every shell a comparable
occupancy at a mock-affordable catalog size.  The guard is still EVALUATED
and its numbers are printed and stored; it is reported, not enforced.

**2. The solve is the damped one** (``world16.solve_damped``), for the reason
its docstring records: the shipped fixed-trip undamped Fisher iteration
diverges on 4 of 8 nside-16 realizations.  P6 (``grad_inf < 1e-8``) is
enforced here exactly as the CLI enforces it.

**3. ``sigma_z`` and the basis geometry come from ``world16``** rather than
from the CLI's hard-coded ``0.023`` / ``M_sph = 315`` defaults, so the anchor
and the mock share one geometry by construction rather than by coincidence.

Everything else -- the f32 ``row_fac`` that the moments are built from
(PLAN's eq. (4) fix, ``cli/build_latent_field.py:204-209``), the
``g_members``/``Xi_members`` distinction, the Chebyshev ``b`` nodes, the
sha256 over arrays + configuration -- is carried over verbatim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import world16 as W16


def build(*, survey, mth_map, out, world=None, z_depth=None, n_shells=12,
          b_gal=None, m_draw=8, n_b_nodes=33, b_max=4.0, seed=22,
          theta_ref=None, verbose=True, ls_sph_fit=None,
          b_gal_dispersion=False, s_b_floor_frac=None):
    """``b_gal_dispersion`` switches on PLAN §3.4's rank-1 draw-covariance
    inflation ``Cov(xi) = H^-1 + s_b^2 v v^T`` (S-2, landed in ``16e8195``).

    It is a KEYWORD with default ``False`` here and not in the shipped builder
    (where S-2 made it default ON) for one reason: this file's job in
    ``CLOSURE_v2.md`` is a BEFORE/AFTER comparison, and the "before" arm has to
    be reproducible from the same script.  ``False`` reproduces the ensemble the
    first closure pass measured; ``True`` is the shipped default.  ``s_b`` is
    never a free number on either path -- it is
    :func:`latent_counts.b_gal_profile_sigma`'s profile curvature at the anchor,
    maximised against the module's 5% systematics floor, and every component
    (statistical, floor, which one won, both curvatures) is stamped into the
    artifact so the two arms differ by a quantity that is on the record.
    """
    import h5py
    import healpy as hp
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from darksirens.catalogs.depth_map import load_selection_fraction
    from darksirens.redshift.grid import zgrid
    from darksirens.redshift.latent_counts import (
        B_GAL_SYSTEMATIC_FLOOR_FRAC, TracerCounts, b_gal_profile_sigma,
        counts_from_catalog, dgrad_db, gradient, laplace_draws,
        make_count_operator, sensitivity)
    from darksirens.redshift.latent_field import (
        shell_response, sky_constant_coeffs, sky_moments)

    world = world or W16.build_world(ls_sph_fit=ls_sph_fit)
    z_depth = float(z_depth if z_depth is not None else W16.Z_DEPTH)
    b_gal = float(world.b_gal if b_gal is None else b_gal)
    theta_ref = theta_ref or dict(M0hat=W16.M0_HAT, sigma_M=W16.SIGMA_M,
                                  delta=0.0, Om0=W16.OM0)

    with h5py.File(survey) as f:
        zg, ng = f["zgals"][...], f["ngals"][...]
        nside = int(f.attrs.get("nside", hp.npix2nside(zg.shape[0])))
    occ = np.where(ng > 0)[0]

    # Equal-comoving-volume shell edges (see the module docstring).
    zz = np.linspace(0.0, z_depth, 4001)
    dc = np.cumsum(np.concatenate(
        [[0.0], 0.5 * (1.0 / np.sqrt(W16.OM0 * (1 + zz[1:]) ** 3 + 1 - W16.OM0)
                       + 1.0 / np.sqrt(W16.OM0 * (1 + zz[:-1]) ** 3 + 1 - W16.OM0))
         * np.diff(zz)]))
    v = dc ** 3
    edges = np.interp(np.linspace(0.0, v[-1], n_shells + 1), v, zz)
    edges[0], edges[-1] = 0.0, z_depth

    counts = counts_from_catalog(zg, ng, occ, edges)
    gal_ok = counts.sum(0) >= 1e4
    pix_ok = (counts > 0).sum(0) >= 500
    guard7 = dict(shell_galaxies=counts.sum(0).astype(int).tolist(),
                  shell_occupied_pixels=(counts > 0).sum(0).astype(int).tolist(),
                  gal_ok=bool(gal_ok.all()), pix_ok=bool(pix_ok.all()))
    f_p = np.maximum(load_selection_fraction(str(mth_map), nside).f_p[occ], 1e-3)
    if verbose:
        print(f"[anchor] {occ.size} pixels, {int(counts.sum())} galaxies, "
              f"{n_shells} shells; guard7 gal_ok={guard7['gal_ok']} "
              f"pix_ok={guard7['pix_ok']} min shell "
              f"{int(counts.sum(0).min())}", flush=True)

    z_fine = np.asarray(world.z_fine)
    z_sub = np.asarray(zgrid[zgrid <= z_depth])
    # The fitted footprint is the survey's OCCUPIED pixels, exactly as
    # ``cli/build_latent_field.py:119`` defines it -- not the world's nominal
    # footprint.  At the mock's catalog size a handful of low-``f_p`` pixels
    # draw zero galaxies (4 of 1854 on realization c002), and an anchor built
    # on the nominal footprint would then index rows the catalog does not have.
    # So the basis is rebuilt here over ``occ``; the geometry (nodes, jitter,
    # length scales, z_node_hi) is the world's, only the row set differs.
    from darksirens.redshift.latent_field import build_latent_basis
    import healpy as _hp

    ls_sph_use = (world.meta["ls_sph"] if world.meta["ls_sph_fit"] is None
                  else world.meta["ls_sph_fit"])
    vec_all = np.column_stack(
        _hp.pix2vec(nside, np.arange(_hp.nside2npix(nside))))
    basis = build_latent_basis(
        vec_all, np.log1p(z_sub),
        n_inducing_sphere=world.meta["m_sph"], n_inducing_z=world.meta["m_z"],
        z_node_hi=z_depth, ls_sph=ls_sph_use, ls_z=world.meta["ls_z"],
        zeta_fine=np.log1p(z_fine), footprint_rows=occ)

    sigma_z = lambda z: np.full_like(np.asarray(z, float), world.meta["sigma_z"])

    def base_fn(theta):
        def _f(z):
            return W16.base_curve(z, delta=theta["delta"], Om0=theta["Om0"],
                                  M0hat=theta["M0hat"],
                                  sigma_M=theta["sigma_M"])
        return _f

    def op_at(theta):
        W = shell_response(edges, z_fine, sigma_z, base_fn(theta))
        tracer = TracerCounts(pix=occ, counts=counts, completeness=f_p,
                              bias=b_gal)
        return make_count_operator(basis.proj_sph, basis.phi_z_fine, W,
                                   tracer), W

    op, W_ref = op_at(theta_ref)
    sol = W16.solve_damped(op)
    xi_hat, L = sol["xi_hat"], sol["H_chol"]
    if verbose:
        print(f"[anchor] grad_inf={float(sol['grad_inf']):.2e}", flush=True)
    if float(sol["grad_inf"]) > 1e-8:
        raise SystemExit("P6 gate: anchor solve did not converge to 1e-8.")

    names = ["M0hat", "sigma_M", "delta", "Om0"]
    steps = dict(M0hat=1e-3, sigma_M=1e-3, delta=1e-2, Om0=1e-3)
    dgrad = np.zeros((op.rank, len(names) + 1))
    for j, nme in enumerate(names):
        tp = dict(theta_ref); tp[nme] += steps[nme]
        tm = dict(theta_ref); tm[nme] -= steps[nme]
        dgrad[:, j] = (np.asarray(gradient(xi_hat, op_at(tp)[0]))
                       - np.asarray(gradient(xi_hat, op_at(tm)[0]))) \
            / (2 * steps[nme])
    dgrad[:, -1] = np.asarray(dgrad_db(xi_hat, op))
    S = np.asarray(sensitivity(xi_hat, L, jnp.asarray(dgrad)))
    labels = names + ["b_gal"]

    # ---- PLAN §3.4's rank-1 b_gal inflation (S-2) -------------------------
    # ``v`` is read straight out of the ``sensitivity_S`` block that has just
    # been built against this same ``L`` -- the builder must not have two
    # constructions of ``d xi_hat/d b`` (``b_gal_profile_sigma`` accepts the
    # column for exactly that reason), and the artifact's ``sensitivity_S``
    # column and the draws' inflation direction are then the same array by
    # construction rather than by coincidence.
    v_b = S[:, labels.index("b_gal")]
    prof = b_gal_profile_sigma(
        xi_hat, op, dgrad_b=dgrad[:, -1], v_b=jnp.asarray(v_b),
        systematic_floor_frac=(B_GAL_SYSTEMATIC_FLOOR_FRAC
                               if s_b_floor_frac is None
                               else float(s_b_floor_frac)))
    s_b = float(prof["s_b"]) if b_gal_dispersion else 0.0
    if verbose:
        print(f"[anchor] s_b={s_b:.6e} (profile {prof['s_b_stat']:.6e}, "
              f"floor {prof['s_b_floor']:.6e}, "
              f"floor_active={prof['floor_active']}); "
              f"dispersion={'ON' if b_gal_dispersion else 'OFF'}", flush=True)

    m_sph, m_z = world.meta["m_sph"], world.meta["m_z"]
    draws, g_members, eps_members = laplace_draws(
        xi_hat, L, m_draw, jax.random.PRNGKey(seed), return_g=True,
        return_eps=True,
        s_b=(s_b if b_gal_dispersion else None),
        v_b=(jnp.asarray(v_b) if b_gal_dispersion else None))
    Xi_members = np.asarray(draws).reshape(m_draw, m_sph, m_z)
    proj_np = np.asarray(basis.proj_sph)
    row_fac = np.stack([
        (proj_np @ np.asarray(d).reshape(m_sph, m_z)).astype(np.float32)
        for d in draws])

    k = np.arange(n_b_nodes)
    b_nodes = 0.5 * b_max * (1 - np.cos(np.pi * k / (n_b_nodes - 1)))
    A, B = sky_moments(basis, np.asarray(draws), b_nodes, f_p, row_fac=row_fac)
    A, B = np.asarray(A), np.asarray(B)
    P_F, F_F = sky_constant_coeffs(f_p)

    n_th = S.shape[1]
    dA = np.zeros(A.shape + (n_th,), dtype=np.float32)
    dB = np.zeros_like(dA)
    phi_z_out = np.asarray(basis.phi_z_out)
    proj = np.asarray(basis.proj_sph)
    fprime = np.stack([(proj @ S[:, j].reshape(m_sph, m_z)) @ phi_z_out.T
                       for j in range(n_th)]).astype(np.float32)
    for m in range(m_draw):
        f_m = row_fac[m].astype(np.float64) @ phi_z_out.T
        for i, b in enumerate(b_nodes):
            e = np.exp(b * f_m)
            dA[m, i] = b * np.einsum("pz,jpz->zj", e, fprime)
            dB[m, i] = b * np.einsum("p,pz,jpz->zj", f_p, e, fprime)

    cfg = dict(survey=str(survey), mth_map=str(mth_map), z_depth=z_depth,
               n_shells=n_shells, b_gal=b_gal, m_draw=m_draw,
               n_b_nodes=n_b_nodes, b_max=b_max, seed=seed,
               theta_ref=theta_ref, labels=labels, jitter=dict(basis.meta),
               world=world.meta, shells="equal_comoving_volume",
               guard7=guard7,
               b_gal_dispersion=bool(b_gal_dispersion), s_b=s_b,
               s_b_profile=float(prof["s_b_stat"]),
               s_b_floor=float(prof["s_b_floor"]),
               s_b_floor_active=bool(prof["floor_active"]),
               b_gal_curvature_profile=float(prof["curvature_profile"]),
               b_gal_curvature_conditional=float(prof["curvature_conditional"]),
               draw_covariance=("H^-1 + s_b^2 v v^T" if b_gal_dispersion
                                else "H^-1"))
    sha = hashlib.sha256()
    for arr in (np.asarray(xi_hat), np.asarray(L), S, counts, f_p,
                np.asarray(W_ref), A, B, b_nodes, z_sub, edges):
        sha.update(np.ascontiguousarray(arr).tobytes())
    sha.update(json.dumps(cfg, sort_keys=True, default=str).encode())
    digest = sha.hexdigest()

    out = Path(out)
    with h5py.File(out, "w") as f:
        g = f.create_group("latent_field")
        g.create_dataset("xi_hat", data=np.asarray(xi_hat))
        g.create_dataset("H_chol", data=np.asarray(L))
        g.create_dataset("sensitivity_S", data=S)
        g.attrs["sensitivity_labels"] = json.dumps(labels)
        g.create_dataset("g_members", data=np.asarray(g_members))
        g.create_dataset("eps_members", data=np.asarray(eps_members))
        g.create_dataset("Xi_members", data=Xi_members)
        g.create_dataset("row_fac", data=row_fac)
        g.create_dataset("A_moments", data=A)
        g.create_dataset("B_moments", data=B)
        g.create_dataset("dA_moments", data=dA)
        g.create_dataset("dB_moments", data=dB)
        g.create_dataset("b_nodes", data=b_nodes)
        g.create_dataset("z_sub", data=z_sub)
        g.create_dataset("fit_pixels", data=occ.astype(np.int32))
        g.create_dataset("completeness", data=f_p)
        g.create_dataset("counts", data=counts)
        g.create_dataset("z_count_edges", data=edges)
        g.create_dataset("shell_response", data=np.asarray(W_ref))
        g.attrs["P_F"] = P_F
        g.attrs["F_F"] = F_F
        g.attrs["theta_ref"] = json.dumps(theta_ref)
        g.attrs["basis_meta"] = json.dumps(dict(basis.meta))
        g.attrs["nside"] = nside
        g.attrs["b_gal"] = b_gal
        g.attrs["seed"] = seed
        g.attrs["grad_inf"] = float(sol["grad_inf"])
        g.attrs["b_gal_dispersion"] = bool(b_gal_dispersion)
        g.attrs["s_b"] = s_b
        g.attrs["draw_covariance"] = cfg["draw_covariance"]
        g.attrs["sha256"] = digest
        g.attrs["format_version"] = "darksirens-latent-field-1.0"
        g.attrs["pr6a_config"] = json.dumps(cfg, default=str)
    if verbose:
        print(f"[anchor] wrote {out} sha256={digest[:16]}...", flush=True)
    return dict(path=str(out), sha256=digest, guard7=guard7,
                grad_inf=float(sol["grad_inf"]),
                b_gal_dispersion=bool(b_gal_dispersion), s_b=s_b,
                s_b_profile=float(prof["s_b_stat"]),
                s_b_floor=float(prof["s_b_floor"]),
                s_b_floor_active=bool(prof["floor_active"]),
                b_gal_curvature_profile=float(prof["curvature_profile"]),
                b_gal_curvature_conditional=float(prof["curvature_conditional"]),
                row_fac_sd=float(np.std(row_fac)),
                member_sd=float(np.mean(np.std(row_fac, axis=0))),
                xi_hat=np.asarray(xi_hat), H_chol=np.asarray(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--survey", required=True)
    p.add_argument("--mth-map", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-shells", type=int, default=12)
    p.add_argument("--m-draw", type=int, default=8)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--b-gal-dispersion", action="store_true",
                   help="PLAN §3.4 rank-1 inflation Cov = H^-1 + s_b^2 v v^T")
    p.add_argument("--s-b-floor-frac", type=float, default=None)
    a = p.parse_args(argv)
    build(survey=a.survey, mth_map=a.mth_map, out=a.out,
          n_shells=a.n_shells, m_draw=a.m_draw, seed=a.seed,
          b_gal_dispersion=a.b_gal_dispersion,
          s_b_floor_frac=a.s_b_floor_frac)


if __name__ == "__main__":
    main()
