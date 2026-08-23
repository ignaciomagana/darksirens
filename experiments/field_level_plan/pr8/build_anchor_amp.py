"""Extended-support anchors for the PR-8 amp(z) sensitivity scan (nside 16).

A copy of ``experiments/field_level_plan/pr6a/build_anchor16.py`` with exactly
two knobs added -- ``z_node_hi`` and the ``amp(z)`` profile -- and one property
its parent does not have: **one solve serves the whole scan.**

Why one solve is not a shortcut but the correct construction: PLAN §4.3 pins
``amp == 1`` at and below the fitted depth, so the count channel -- its
operator, its MAP, its Hessian, its Laplace draws -- is a function of the
geometry and the counts and NOT of ``amp_hi``.  Only the CONSUMPTION side
depends on it: ``phi_z`` above the depth, and through it the eq. (2) sky
moments ``(A, B)``.  So the scan's rows are built from one ``xi_hat``, one
``H_chol``, one member ensemble and one ``row_fac``, and differ in the two
arrays that the assumption actually touches.  Any spread across the rows is
therefore the assumption, and cannot be a re-solve, a re-draw or a seed.

The extended geometry is a REAL change against the pr6a anchor and is not
hidden: ``M_z`` goes from 5 to whatever the caller passes (11 for nodes to
z = 1.5), because the resolution guard binds on the node range, and the
below-depth basis is a different -- finer -- basis as a result.  That is why
``--amp-hi 0`` is built and scanned as its own row: it is the control for the
geometry, so the table's amp dependence is read off rows that share it.

Everything numerical is still the shipped function (``counts_from_catalog``,
``build_latent_basis``, ``shell_response``, ``make_count_operator``,
``gradient``, ``dgrad_db``, ``sensitivity``, ``b_gal_profile_sigma``,
``laplace_draws``, ``sky_moments``, ``sky_constant_coeffs``), the solve is
``world16.solve_damped`` (S-1), and the output is the same ``/latent_field``
group at the same ``format_version``, so ``likelihood/factory.py`` and
``latent_q.load_latent_plan`` consume it unmodified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pr6a"))

import world16 as W16                                          # noqa: E402


def solve_once(*, survey, mth_map, z_depth=None, z_node_hi=1.5, m_z=11,
               n_shells=12, b_gal=None, m_draw=8, seed=22, theta_ref=None,
               verbose=True, b_gal_dispersion=True):
    """The amp-INDEPENDENT half: counts, basis, solve, sensitivities, draws.

    Returns a dict the writer below turns into one artifact per ``amp_hi``.
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
        build_latent_basis, shell_response)

    z_depth = float(W16.Z_DEPTH if z_depth is None else z_depth)
    z_node_hi = float(z_node_hi)
    b_gal = float(W16.B_GAL if b_gal is None else b_gal)
    theta_ref = theta_ref or dict(M0hat=W16.M0_HAT, sigma_M=W16.SIGMA_M,
                                  delta=0.0, Om0=W16.OM0)

    with h5py.File(survey) as f:
        zg, ng = f["zgals"][...], f["ngals"][...]
        nside = int(f.attrs.get("nside", hp.npix2nside(zg.shape[0])))
    occ = np.where(ng > 0)[0]

    # Equal-comoving-volume shell edges -- pr6a's choice, for pr6a's reason
    # (guard 7 would demand ~1.7e7 catalogued galaxies under linear edges).
    zz = np.linspace(0.0, z_depth, 4001)
    dc = np.cumsum(np.concatenate(
        [[0.0], 0.5 * (1.0 / np.sqrt(W16.OM0 * (1 + zz[1:]) ** 3 + 1 - W16.OM0)
                       + 1.0 / np.sqrt(W16.OM0 * (1 + zz[:-1]) ** 3
                                       + 1 - W16.OM0))
         * np.diff(zz)]))
    v = dc ** 3
    edges = np.interp(np.linspace(0.0, v[-1], n_shells + 1), v, zz)
    edges[0], edges[-1] = 0.0, z_depth

    counts = counts_from_catalog(zg, ng, occ, edges)
    guard7 = dict(shell_galaxies=counts.sum(0).astype(int).tolist(),
                  shell_occupied_pixels=(counts > 0).sum(0).astype(int).tolist(),
                  gal_ok=bool((counts.sum(0) >= 1e4).all()),
                  pix_ok=bool(((counts > 0).sum(0) >= 500).all()))
    f_p = np.maximum(load_selection_fraction(str(mth_map), nside).f_p[occ], 1e-3)

    z_fine = np.linspace(1e-4, z_depth, W16.N_FINE)
    # The CONSUMPTION rows run to the top of the NODE range, not to the depth:
    # rho needs moments wherever the field can be nonzero (PLAN eq. 4).
    z_sub = np.asarray(zgrid[zgrid <= z_node_hi])
    vec_all = np.column_stack(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))

    # The count operator's basis: amp NEVER enters it (amp == 1 below the
    # depth, and z_fine stops at the depth), so it is built once, without a
    # profile, and every row of the scan solves against this operator.
    basis_fit = build_latent_basis(
        vec_all, np.log1p(z_sub), n_inducing_sphere=W16.M_SPH,
        n_inducing_z=m_z, z_node_hi=z_node_hi, ls_sph=W16.LS_SPH,
        ls_z=W16.LS_Z, zeta_fine=np.log1p(z_fine), footprint_rows=occ)

    sigma_z = lambda z: np.full_like(np.asarray(z, float), W16.SIGMA_Z)  # noqa: E731

    def base_fn(theta):
        return lambda z: W16.base_curve(
            z, delta=theta["delta"], Om0=theta["Om0"], M0hat=theta["M0hat"],
            sigma_M=theta["sigma_M"])

    def op_at(theta):
        W = shell_response(edges, z_fine, sigma_z, base_fn(theta))
        tracer = TracerCounts(pix=occ, counts=counts, completeness=f_p,
                              bias=b_gal)
        return make_count_operator(basis_fit.proj_sph, basis_fit.phi_z_fine, W,
                                   tracer), W

    op, W_ref = op_at(theta_ref)
    sol = W16.solve_damped(op)
    xi_hat, L = sol["xi_hat"], sol["H_chol"]
    if verbose:
        print(f"[solve] rank={op.rank} grad_inf={float(sol['grad_inf']):.3e} "
              f"({occ.size} pixels, {int(counts.sum())} galaxies, "
              f"{z_sub.size} consumption rows)", flush=True)
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

    v_b = S[:, labels.index("b_gal")]
    prof = b_gal_profile_sigma(
        xi_hat, op, dgrad_b=dgrad[:, -1], v_b=jnp.asarray(v_b),
        systematic_floor_frac=B_GAL_SYSTEMATIC_FLOOR_FRAC)
    s_b = float(prof["s_b"]) if b_gal_dispersion else 0.0
    draws, g_members, eps_members = laplace_draws(
        xi_hat, L, m_draw, jax.random.PRNGKey(seed), return_g=True,
        return_eps=True,
        s_b=(s_b if b_gal_dispersion else None),
        v_b=(jnp.asarray(v_b) if b_gal_dispersion else None))
    if verbose:
        print(f"[solve] s_b={s_b:.6e} (floor_active={prof['floor_active']}); "
              f"{m_draw} members", flush=True)

    return dict(nside=nside, occ=occ, counts=counts, f_p=f_p, edges=edges,
                z_fine=z_fine, z_sub=z_sub, z_depth=z_depth,
                z_node_hi=z_node_hi, m_z=int(m_z), b_gal=b_gal,
                theta_ref=theta_ref, W_ref=np.asarray(W_ref),
                xi_hat=np.asarray(xi_hat), H_chol=np.asarray(L), S=S,
                labels=labels, draws=np.asarray(draws),
                g_members=np.asarray(g_members),
                eps_members=np.asarray(eps_members), s_b=s_b, prof=prof,
                guard7=guard7, grad_inf=float(sol["grad_inf"]),
                vec_all=vec_all, m_draw=int(m_draw), seed=int(seed),
                n_shells=int(n_shells))


def write_amp_anchor(base, out, *, amp_hi, amp_kind="step", n_b_nodes=33,
                     b_max=4.0, verbose=True):
    """One row of the scan: the same solve, consumed under an assumed amp(z).

    ``amp_hi is None`` writes a LEGACY-geometry-free anchor (no profile at all,
    consumption rows truncated at the depth) -- the pre-PR-8 artifact shape.
    ``amp_hi = 0.0`` writes the same physics with the profile machinery
    switched on, which is the gate that the machinery is inert at the legacy
    value.
    """
    import h5py
    import jax

    jax.config.update("jax_enable_x64", True)

    from darksirens.redshift.latent_field import (
        build_latent_basis, sky_constant_coeffs, sky_moments)

    m_sph, m_z = W16.M_SPH, base["m_z"]
    z_sub = (base["z_sub"] if amp_hi is not None
             else base["z_sub"][base["z_sub"] <= base["z_depth"]])
    amp_kw = ({} if amp_hi is None else
              dict(amp_hi=float(amp_hi), amp_kind=str(amp_kind),
                   amp_z_depth=float(base["z_depth"]),
                   amp_Om0=float(W16.OM0)))
    basis = build_latent_basis(
        base["vec_all"], np.log1p(z_sub), n_inducing_sphere=m_sph,
        n_inducing_z=m_z, z_node_hi=base["z_node_hi"], ls_sph=W16.LS_SPH,
        ls_z=W16.LS_Z, zeta_fine=np.log1p(base["z_fine"]),
        footprint_rows=base["occ"], **amp_kw)

    draws = base["draws"]
    proj_np = np.asarray(basis.proj_sph)
    row_fac = np.stack([
        (proj_np @ np.asarray(d).reshape(m_sph, m_z)).astype(np.float32)
        for d in draws])
    k = np.arange(n_b_nodes)
    b_nodes = 0.5 * b_max * (1 - np.cos(np.pi * k / (n_b_nodes - 1)))
    A, B = sky_moments(basis, draws, b_nodes, base["f_p"], row_fac=row_fac)
    A, B = np.asarray(A), np.asarray(B)
    P_F, F_F = sky_constant_coeffs(base["f_p"])

    S = base["S"]
    n_th = S.shape[1]
    dA = np.zeros(A.shape + (n_th,), dtype=np.float32)
    dB = np.zeros_like(dA)
    phi_z_out = np.asarray(basis.phi_z_out)
    fprime = np.stack([(proj_np @ S[:, j].reshape(m_sph, m_z)) @ phi_z_out.T
                       for j in range(n_th)]).astype(np.float32)
    for m in range(base["m_draw"]):
        f_m = row_fac[m].astype(np.float64) @ phi_z_out.T
        for i, b in enumerate(b_nodes):
            e = np.exp(b * f_m)
            dA[m, i] = b * np.einsum("pz,jpz->zj", e, fprime)
            dB[m, i] = b * np.einsum("p,pz,jpz->zj", base["f_p"], e, fprime)

    cfg = dict(z_depth=base["z_depth"], z_node_hi=base["z_node_hi"],
               m_sph=m_sph, m_z=m_z, n_shells=base["n_shells"],
               b_gal=base["b_gal"], m_draw=base["m_draw"],
               n_b_nodes=n_b_nodes, b_max=b_max, seed=base["seed"],
               theta_ref=base["theta_ref"], labels=base["labels"],
               jitter=dict(basis.meta), guard7=base["guard7"],
               shells="equal_comoving_volume", s_b=base["s_b"],
               amp_hi=amp_hi, amp_kind=(None if amp_hi is None else amp_kind),
               b_gal_dispersion=True,
               draw_covariance="H^-1 + s_b^2 v v^T",
               pr8="sensitivity scan artifact: amp(z > z_depth) is ASSUMED, "
                   "not fitted (PLAN OWNER DECISION 7)")
    sha = hashlib.sha256()
    for arr in (base["xi_hat"], base["H_chol"], S, base["counts"], base["f_p"],
                base["W_ref"], A, B, b_nodes, z_sub, base["edges"]):
        sha.update(np.ascontiguousarray(arr).tobytes())
    sha.update(json.dumps(cfg, sort_keys=True, default=str).encode())
    digest = sha.hexdigest()

    out = Path(out)
    with h5py.File(out, "w") as f:
        g = f.create_group("latent_field")
        g.create_dataset("xi_hat", data=base["xi_hat"])
        g.create_dataset("H_chol", data=base["H_chol"])
        g.create_dataset("sensitivity_S", data=S)
        g.attrs["sensitivity_labels"] = json.dumps(base["labels"])
        g.create_dataset("g_members", data=base["g_members"])
        g.create_dataset("eps_members", data=base["eps_members"])
        g.create_dataset("Xi_members",
                         data=draws.reshape(base["m_draw"], m_sph, m_z))
        g.create_dataset("row_fac", data=row_fac)
        g.create_dataset("A_moments", data=A)
        g.create_dataset("B_moments", data=B)
        g.create_dataset("dA_moments", data=dA)
        g.create_dataset("dB_moments", data=dB)
        g.create_dataset("b_nodes", data=b_nodes)
        g.create_dataset("z_sub", data=z_sub)
        g.create_dataset("fit_pixels", data=base["occ"].astype(np.int32))
        g.create_dataset("completeness", data=base["f_p"])
        g.create_dataset("counts", data=base["counts"])
        g.create_dataset("z_count_edges", data=base["edges"])
        g.create_dataset("shell_response", data=base["W_ref"])
        g.attrs["P_F"] = P_F
        g.attrs["F_F"] = F_F
        g.attrs["theta_ref"] = json.dumps(base["theta_ref"])
        g.attrs["basis_meta"] = json.dumps(dict(basis.meta))
        g.attrs["nside"] = base["nside"]
        g.attrs["b_gal"] = base["b_gal"]
        g.attrs["seed"] = base["seed"]
        g.attrs["grad_inf"] = base["grad_inf"]
        g.attrs["b_gal_dispersion"] = True
        g.attrs["s_b"] = base["s_b"]
        g.attrs["draw_covariance"] = "H^-1 + s_b^2 v v^T"
        g.attrs["sha256"] = digest
        g.attrs["format_version"] = "darksirens-latent-field-1.0"
        g.attrs["pr8_config"] = json.dumps(cfg, default=str)
    if verbose:
        print(f"[amp {amp_hi}] wrote {out.name} sha={digest[:12]} "
              f"rows={z_sub.size} row_fac_sd={float(np.std(row_fac)):.4f}",
              flush=True)
    return dict(path=str(out), sha256=digest, amp_hi=amp_hi,
                amp_kind=(None if amp_hi is None else amp_kind),
                n_rows=int(z_sub.size),
                row_fac_sd=float(np.std(row_fac)),
                member_sd=float(np.mean(np.std(row_fac, axis=0))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(
        Path(__file__).resolve().parents[1] / "pr6a" / "data" / "rb"))
    p.add_argument("--outdir", default=str(Path(__file__).resolve().parent
                                          / "anchors"))
    p.add_argument("--z-node-hi", type=float, default=1.5)
    p.add_argument("--m-z", type=int, default=11)
    p.add_argument("--m-draw", type=int, default=8)
    p.add_argument("--seed", type=int, default=22)
    p.add_argument("--amp", type=float, action="append", default=None)
    p.add_argument("--growth", type=float, default=None,
                   help="also build a 'growth'-shaped anchor at this amp_hi")
    p.add_argument("--legacy-geometry", action="store_true",
                   help="also write the pre-PR-8 artifact shape (no profile, "
                        "rows truncated at the depth) on the SAME solve")
    a = p.parse_args(argv)

    d = Path(a.data)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    amps = [0.0, 0.05, 0.1, 0.2, 0.4] if a.amp is None else a.amp

    base = solve_once(survey=d / "catalog_pixelated_nside_16.h5",
                      mth_map=d / "mth_map_nside16.h5",
                      z_node_hi=a.z_node_hi, m_z=a.m_z, m_draw=a.m_draw,
                      seed=a.seed)
    manifest = []
    if a.legacy_geometry:
        manifest.append(write_amp_anchor(
            base, out / "anchor_noprofile.h5", amp_hi=None))
    for amp in amps:
        manifest.append(write_amp_anchor(
            base, out / f"anchor_amp{amp:g}.h5", amp_hi=float(amp)))
    if a.growth is not None:
        manifest.append(write_amp_anchor(
            base, out / f"anchor_growth{a.growth:g}.h5",
            amp_hi=float(a.growth), amp_kind="growth"))
    meta = dict(z_node_hi=a.z_node_hi, m_z=a.m_z, m_sph=W16.M_SPH,
                z_depth=base["z_depth"], rank=W16.M_SPH * a.m_z,
                grad_inf=base["grad_inf"], s_b=base["s_b"],
                guard7=base["guard7"], n_fit=int(base["occ"].size),
                n_rows_full=int(base["z_sub"].size), anchors=manifest)
    (out / "manifest.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
