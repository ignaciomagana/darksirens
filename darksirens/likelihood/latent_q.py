"""The latent-field seam (field-level PR-5) -- the ONE place ``Q`` is generated.

In table mode the missing-galaxy modulation ``Q_m(p, z)`` is a resident data
constant: an ``(M, N_rows, N_grid)`` log-Q table read from an HDF5 artifact and
gathered two nodes at a time by
:func:`darksirens.redshift.prior.eval_dark_member_completion`.  In LATENT mode
that table does not exist.  ``Q`` is *generated* from the compact anchor
artifact (``darksirens_build_latent_field``, PR-4) through the factored basis of
PLAN §3.5:

    logQ_m(p, z) = b_GW * (row_fac_m[p] . phi_z[z])  -  rho_m(z; c, b_GW)     (p in F, z <= z_depth)
    logQ_m(p, z) = 0                                                          (otherwise)

with ``row_fac_m = Phi_s[F] @ Xi_m`` an ``(n_fit, M_z)`` STATIC leaf (11.7 MB at
production rank, ``M_draw = 8``) and ``phi_z`` the ``(N_grid, M_z)`` redshift
factor.  The Kronecker basis is never materialized: a dense ``Phi`` at
``M = 3780`` would be 47.6 GB (PLAN §3.2).

``rho`` is the closed-form budget normalizer of PLAN eq. (2)/(4),

    rho_m(z; c, b) = log[ (A_m(z; b) - c(z) B_m(z; b)) / (P_F - c(z) F_F) ],

the unique per-``z`` monopole making the CONSUMED budget identity

    sum_p (1 - f_p c(z)) Q_p(z)  ==  sum_p (1 - f_p c(z))

hold exactly for every ``z``, member and ``theta``.  ``A_m`` / ``B_m`` are the
theta-free sky moments over the FITTED FOOTPRINT ``F`` only (never the full
sky), so the off-footprint block of the identity is conserved trivially with
``Q == 1`` -- which is why the seam must return **bit-zero** ``logQ`` on
off-footprint rows (pin P13b): the PE/injection pixel union is 49,143 rows
against 30,470 footprint rows, so ~38% of every gather routes there.

**Rung 0 is the ``dtheta = 0`` branch of this same code path** (PLAN §3.5), so
PR-6a and PR-6b share one implementation and the control arm is a flag, not a
fork.  :func:`theta_shift` and :func:`moments_at` are the rung-1 entry points;
at ``dtheta = 0`` they are exact identities, which is pin P18.

Nothing in this module is theta-dependent except ``rho``'s ``(c, b)``
arguments, both ``O(N_grid)``.  ``row_fac`` acquiring a ``theta`` index would
silently be the per-proposal re-solve design that PLAN §10 refuses; the
``block_sizing`` transient branch is what enforces that structurally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import jax.numpy as jnp

# NOTE: this module deliberately imports NOTHING from ``darksirens.redshift`` at
# module level.  ``redshift/prior.py`` imports the seam kernel from here, and
# ``darksirens.redshift.__init__`` imports ``prior``, so a module-level
# ``from darksirens.redshift.grid import zgrid`` closes a cycle.  The loader --
# the only function here that needs the grid or the basis builder -- imports
# them locally; it runs once, host-side, at likelihood build.

#: Sentinel row index for "this catalog row is OUTSIDE the fitted footprint".
#: The row-factor leaf carries one extra ZERO row at index ``n_fit`` so an
#: off-footprint gather is a normal (fast, non-branching) indexed read that
#: lands on zeros; the ``on_footprint`` mask then forces ``logQ = 0`` exactly,
#: including the ``rho`` term (pin P13b).
OFF_FOOTPRINT = -1


@dataclass(frozen=True)
class LatentQPlan:
    """Everything the seam consumes online, resolved once at likelihood build.

    Every array here is a FACTORY-STATIC leaf: theta-free, member-stacked where
    it needs to be, and sized so the 256x concurrency multiplier cannot
    amplify it (PLAN §2.4).  The per-proposal work is exactly ``rho`` --
    ``O(n_b + N_grid)`` per member -- plus, at rung 1, the two
    member-INDEPENDENT corrections of :func:`theta_shift` / :func:`moments_at`.

    Attributes
    ----------
    phi_z : (N_grid, M_z) f64
        The redshift factor rows on the FULL ``zgrid``, ZEROED above
        ``z_depth``.  Zeroing (rather than truncating) keeps the seam's gather
        an ordinary ``phi_z[idx]`` on the same index the table path uses, and
        makes the above-depth branch bit-exact rather than merely masked: a
        zero row gives ``logQ = 0`` gives ``Q_eff = 1``, which is precisely the
        ``z_depth`` relaxation the table path applies downstream.
    below_depth : (N_grid,) bool
        ``zgrid <= z_depth`` (all-True when ``z_depth is None``).
    row_fac : (M_draw, n_fit + 1, M_z) f32
        ``Phi_s[F] @ Xi_m`` per member, with a trailing ZERO pad row at index
        ``n_fit`` for off-footprint gathers.
    A, B : (M_draw, n_b, N_grid) f64
        PLAN eq. (2) sky moments over ``F``, padded from the artifact's
        below-depth ``z_sub`` block to the full grid (the pad is inert: it is
        only ever read above the depth, where ``below_depth`` masks it out).
    b_nodes : (n_b,) f64
        Chebyshev-Lobatto ``b_GW`` interpolation nodes.
    P_F, F_F : float
        ``|F|`` and ``sum_{p in F} f_p`` -- eq. (2)'s footprint constants.
    S, dA, dB : rung-1 (PR-6b) sensitivity blocks, or ``None``.
    """

    phi_z: Any
    below_depth: Any
    row_fac: Any
    A: Any
    B: Any
    b_nodes: Any
    P_F: float
    F_F: float
    m_sph: int
    m_z: int
    S: Any = None
    dA: Any = None
    dB: Any = None
    theta_ref: Mapping[str, float] = None
    labels: tuple = ()
    meta: Mapping[str, Any] = None

    @property
    def n_draw(self) -> int:
        return int(self.row_fac.shape[0])

    @property
    def n_fit(self) -> int:
        # The pad row is not part of the footprint.
        return int(self.row_fac.shape[1]) - 1


# ---------------------------------------------------------------- row mapping

def footprint_row_map(row_pixels, fit_pixels, n_fit: int) -> np.ndarray:
    """Map catalog rows -> footprint rows, with off-footprint -> the pad row.

    ``row_pixels`` is the catalog's row-to-global-HEALPix map (``N_rows,``);
    ``fit_pixels`` is the artifact's fitted footprint ``F`` (``n_fit,``).
    Returns an ``(N_rows,)`` int32 index into ``[0, n_fit]`` where ``n_fit`` is
    the ZERO pad row of :attr:`LatentQPlan.row_fac`.

    HOST-SIDE ONLY (called once at likelihood build): it uses ``searchsorted``
    on concrete arrays.  The companion ``on_footprint`` mask is what makes the
    off-footprint ``logQ`` bit-zero rather than merely small -- the pad row
    alone would zero the field term but leave ``-rho``.
    """
    row_pixels = np.asarray(row_pixels, dtype=np.int64)
    fit_pixels = np.asarray(fit_pixels, dtype=np.int64)
    order = np.argsort(fit_pixels)
    fit_sorted = fit_pixels[order]
    pos = np.searchsorted(fit_sorted, row_pixels)
    pos_c = np.clip(pos, 0, fit_sorted.size - 1)
    hit = fit_sorted[pos_c] == row_pixels
    rows = np.where(hit, order[pos_c], n_fit).astype(np.int32)
    return rows


def on_footprint_mask(row_map, n_fit: int) -> jnp.ndarray:
    """``(N_rows,)`` bool: ``True`` where the row is inside ``F``."""
    return jnp.asarray(np.asarray(row_map) != n_fit)


# ------------------------------------------------------------- b interpolation

def _bary_weights(n: int) -> np.ndarray:
    w = (-1.0) ** np.arange(n)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def interp_b(table, b_nodes, b):
    """Barycentric Chebyshev-Lobatto interpolation along the ``b`` axis.

    ``table`` is ``(..., n_b, N_grid)``; the result is ``(..., N_grid)`` at the
    scalar ``b``.  Exact at the nodes by the barycentric formula's pole
    handling (pin P9, 1e-6 against a direct rebuild).  ``b`` is a TRACED
    sampled parameter, so the node-coincidence branch is a ``where``, never a
    Python ``if``.
    """
    table = jnp.asarray(table)
    b_nodes = jnp.asarray(b_nodes, dtype=jnp.float64)
    n = b_nodes.shape[0]
    w = jnp.asarray(_bary_weights(n))
    d = jnp.asarray(b, dtype=jnp.float64) - b_nodes
    exact = jnp.any(d == 0.0)
    idx = jnp.argmin(jnp.abs(d))
    coef = w / jnp.where(d == 0.0, 1.0, d)
    coef = jnp.where(exact, jnp.zeros_like(coef).at[idx].set(1.0), coef)
    coef = coef / jnp.sum(coef)
    return jnp.tensordot(coef, table, axes=([0], [table.ndim - 2]))


# ------------------------------------------------------------ budget normalizer

def rho_from_moments(A_m, B_m, c, b, b_nodes, P_F: float, F_F: float,
                     below_depth=None):
    """The closed-form budget normalizer ``rho_m(z; c, b)`` -- PLAN eq. (2)/(4).

        rho = log[ (A_m(z; b) - c(z) B_m(z; b)) / (P_F - c(z) F_F) ]

    ``c`` is the SCALAR sky-aggregate completeness curve ``C(z; theta)`` on
    ``zgrid`` (the ``f_p`` structure is already inside ``B_m``; a stratified,
    pixel-varying ``c`` does not factor through the moments and is refused at
    the loader, PLAN guard 6).  ``b`` is the sampled ``b_GW``.

    Returns ``(N_grid,)``, ZEROED above the depth so that the seam's
    above-depth ``logQ`` is bit-zero and the ``z_depth`` relaxation is
    reproduced exactly rather than approximately.

    The denominator is the ``Q == 1`` value of the same sum, so ``rho`` is
    identically ``0`` when the field is ``0`` -- the property P16 turns into a
    physics gate.
    """
    A = interp_b(A_m, b_nodes, b)                       # (N_grid,)
    B = interp_b(B_m, b_nodes, b)                       # (N_grid,)
    c = jnp.asarray(c, dtype=A.dtype)
    num = A - c * B
    den = jnp.asarray(P_F, dtype=A.dtype) - c * jnp.asarray(F_F, dtype=A.dtype)
    # Both sides are sums of positive weights over the footprint; a non-positive
    # value can only come from c > 1 (clipped upstream) or a corrupt artifact.
    rho = jnp.log(jnp.maximum(num, 1e-300)) - jnp.log(jnp.maximum(den, 1e-300))
    if below_depth is not None:
        rho = jnp.where(below_depth, rho, 0.0)
    return rho


# --------------------------------------------------------------- the seam core

def member_row_fac(plan: LatentQPlan, m, row_map):
    """Member ``m``'s row factor gathered onto CATALOG rows: ``(N_rows, M_z)``.

    Off-footprint rows land on the zero pad row.  This is a gather of an
    ``(n_fit + 1, M_z)`` static leaf, NOT a materialization of anything
    ``(N_rows, N_grid)``-shaped.
    """
    return jnp.asarray(plan.row_fac)[m][jnp.asarray(row_map)]


def latent_logq_at(row_fac_rows, phi_z_nodes, rho_nodes, b_gw, on_fp):
    """``logQ`` at gathered ``(row, node)`` pairs -- the hot-path kernel.

        logQ = on_fp * [ b_GW * (row_fac[row] . phi_z[node]) - rho[node] ]

    ``row_fac_rows`` is ``(..., M_z)`` (already gathered by row),
    ``phi_z_nodes`` is ``(..., M_z)`` (already gathered by grid node),
    ``rho_nodes`` is ``(...,)``, ``on_fp`` is ``(...,)`` bool.  Broadcasting is
    ordinary, so this serves the two-node PE/selection gather and the full-row
    state-prep generation with one expression.

    The mask is applied to the WHOLE bracket, not just the field term: eq. (4)
    conserves the off-footprint block only because ``Q == 1`` there, and
    ``-rho`` alone would break it (pin P13b).
    """
    field = jnp.sum(row_fac_rows.astype(rho_nodes.dtype) * phi_z_nodes, axis=-1)
    lq = jnp.asarray(b_gw, dtype=field.dtype) * field - rho_nodes
    return jnp.where(on_fp, lq, 0.0)


def latent_logq_rows(plan: LatentQPlan, row_fac_rows, rho, b_gw, on_fp):
    """One member's FULL ``(N_rows, N_grid)`` log-Q block.

    This is the STATE-PREP path, not the hot path: it exists so the normalizer
    reductions (``member_N_miss_integrals``, the field-global budget) can run
    their EXISTING per-member ``lax.scan`` bodies unchanged -- one member's
    block is formed, integrated, and discarded, so the peak stays
    ``O(N_rows x N_grid)`` exactly as in table mode and no ``(M, N_rows,
    N_grid)`` cube is ever built.

    Above the depth ``plan.phi_z`` is zero and ``rho`` is zero, so the block is
    bit-zero there; off-footprint rows are bit-zero at every ``z``.
    """
    phi_z = jnp.asarray(plan.phi_z)                      # (N_grid, M_z)
    field = jnp.asarray(row_fac_rows, dtype=phi_z.dtype) @ phi_z.T   # (N_rows, N_grid)
    lq = jnp.asarray(b_gw, dtype=field.dtype) * field - rho[None, :]
    return jnp.where(jnp.asarray(on_fp)[:, None], lq, 0.0)


# ------------------------------------------------------------------ rung 1

def theta_shift(S, dtheta, m_sph: int, m_z: int):
    """``row_fac_shift(theta) = reshape(S . dtheta, (M_sph, M_z))`` (PLAN §1.7).

    MEMBER-INDEPENDENT and ``O(M)``, so it is computed inside ``body()`` and
    must NOT be barriered.  At ``dtheta = 0`` this returns exact zeros, which
    is the identity half of pin P18.
    """
    if S is None:
        return jnp.zeros((m_sph, m_z))
    return jnp.reshape(jnp.asarray(S) @ jnp.asarray(dtheta), (m_sph, m_z))


def moments_at(A, B, dA, dB, dtheta):
    """Linear-response correction of the sky moments (PLAN §1.7).

    ``A(theta) = A + (dA/dtheta) . dtheta`` and likewise for ``B``.  At
    ``dtheta = 0`` returns ``(A, B)`` unchanged -- bit-identically, since no
    arithmetic is performed on the ``None``/zero branch.
    """
    if dA is None or dB is None:
        return A, B
    d = jnp.asarray(dtheta)
    return (A + jnp.tensordot(jnp.asarray(dA), d, axes=([-1], [0])),
            B + jnp.tensordot(jnp.asarray(dB), d, axes=([-1], [0])))


# ------------------------------------------------------------------ the loader

def load_latent_plan(path, *, z_depth, expect_nside=None):
    """Read a PR-4 anchor artifact into a :class:`LatentQPlan` (HOST-SIDE).

    Called once at likelihood-build time.  Everything traced downstream is a
    concrete array by the time it leaves here; nothing in this function may run
    under a trace.

    The artifact's ``A``/``B`` moments and ``phi_z`` live on ``z_sub`` -- the
    below-depth PREFIX of ``zgrid`` -- so both are zero-padded to the full grid
    here.  The pad is provably inert: every consumer masks it with
    ``below_depth``.
    """
    import h5py

    from darksirens.redshift.grid import zgrid
    from darksirens.redshift.latent_field import build_latent_basis

    with h5py.File(path, "r") as f:
        g = f["latent_field"]
        meta = dict(g.attrs)
        basis_meta = _json_attr(g, "basis_meta")
        row_fac_fit = np.asarray(g["row_fac"][...], dtype=np.float32)
        A_sub = np.asarray(g["A_moments"][...], dtype=np.float64)
        B_sub = np.asarray(g["B_moments"][...], dtype=np.float64)
        b_nodes = np.asarray(g["b_nodes"][...], dtype=np.float64)
        z_sub = np.asarray(g["z_sub"][...], dtype=np.float64)
        fit_pixels = np.asarray(g["fit_pixels"][...], dtype=np.int64)
        P_F = float(g.attrs["P_F"])
        F_F = float(g.attrs["F_F"])
        S = np.asarray(g["sensitivity_S"][...], dtype=np.float64)
        dA = np.asarray(g["dA_moments"][...], dtype=np.float64)
        dB = np.asarray(g["dB_moments"][...], dtype=np.float64)
        labels = tuple(_json_attr(g, "sensitivity_labels") or ())
        theta_ref = _json_attr(g, "theta_ref")
        nside = int(g.attrs["nside"])

    if expect_nside is not None and int(expect_nside) != nside:
        raise ValueError(
            f"latent artifact was built at nside={nside} but the catalog is at "
            f"nside={int(expect_nside)}; the footprint row map would be "
            "meaningless. Rebuild the anchor at the run's nside.")

    zg = np.asarray(zgrid)
    n_grid = zg.size
    n_sub = z_sub.size
    if n_sub > n_grid or not np.allclose(zg[:n_sub], z_sub, rtol=0, atol=1e-12):
        raise ValueError(
            "latent artifact's z_sub is not the below-depth prefix of this "
            f"run's zgrid ({n_sub} nodes vs {n_grid}); the run's "
            "DARKSIRENS_ZMAX differs from the build's. Rebuild the anchor "
            "under the run's grid (PLAN §5.2: shell edges and the consumption "
            "grid come from the artifact, never from ZMAX).")

    below = np.ones(n_grid, dtype=bool) if z_depth is None else (zg <= float(z_depth))
    if z_depth is None:
        # ``below`` is all-True here, so a DEPTH-BUILT artifact would be read as
        # if its moments covered the whole grid.  Above ``z_sub`` the padded
        # ``A``/``B`` are exact zeros, so ``rho`` takes its ``1e-300`` floor
        # (``rho ~ -700``) instead of ``0`` and the seam returns ``logQ ~ +700``
        # on every footprint row -- clipped, that is a ~1e3x inflation of the
        # missing-host density above the build's depth.  Refuse instead.
        if n_sub != n_grid:
            raise ValueError(
                f"latent artifact was built with a depth ({n_sub} of {n_grid} "
                "zgrid nodes, z_sub ends at "
                f"z={float(z_sub[-1]):.4f}) but this run resolved NO survey "
                "z_depth, so the seam would treat the artifact's zero pad as "
                "real field above that node and rail logQ. Pass the "
                "--survey_z_depth the anchor was built at, or rebuild the "
                "anchor on the full grid.")
    elif int(below.sum()) != n_sub:
        raise ValueError(
            f"latent artifact covers {n_sub} below-depth nodes but the run's "
            f"z_depth={z_depth} selects {int(below.sum())}; the seam would "
            "silently truncate or extrapolate the field. Rebuild the anchor at "
            "the run's z_depth.")

    # phi_z on the FULL grid: the real factor rows below the depth, exact zeros
    # above (the z_depth relaxation, expressed in the basis rather than as a
    # downstream mask).
    m_sph = int(basis_meta["M_sph"])
    m_z = int(basis_meta["M_z"])
    sub_basis = build_latent_basis(
        np.zeros((1, 3)), np.log1p(z_sub),
        n_inducing_sphere=m_sph, n_inducing_z=m_z,
        z_node_hi=float(basis_meta["z_node_hi"]),
        amp=float(basis_meta["amp"]),
        ls_sph=float(basis_meta["ls_sph"]), ls_z=float(basis_meta["ls_z"]))
    phi_z = np.zeros((n_grid, m_z), dtype=np.float64)
    phi_z[:n_sub] = np.asarray(sub_basis.phi_z_out)

    def _pad_moments(t):
        out = np.zeros(t.shape[:2] + (n_grid,) + t.shape[3:], dtype=t.dtype)
        out[:, :, :n_sub] = t
        return out

    A = _pad_moments(A_sub)
    B = _pad_moments(B_sub)
    dA = _pad_moments(dA)
    dB = _pad_moments(dB)

    # The zero pad row that turns an off-footprint gather into zeros.
    pad = np.zeros((row_fac_fit.shape[0], 1, m_z), dtype=np.float32)
    row_fac = np.concatenate([row_fac_fit, pad], axis=1)

    plan = LatentQPlan(
        phi_z=jnp.asarray(phi_z), below_depth=jnp.asarray(below),
        row_fac=jnp.asarray(row_fac),
        A=jnp.asarray(A), B=jnp.asarray(B), b_nodes=jnp.asarray(b_nodes),
        P_F=P_F, F_F=F_F, m_sph=m_sph, m_z=m_z,
        S=jnp.asarray(S), dA=jnp.asarray(dA), dB=jnp.asarray(dB),
        theta_ref=theta_ref, labels=labels,
        meta=dict(basis_meta, nside=nside, fit_pixels=fit_pixels,
                  sha256=str(meta.get("sha256", "")),
                  format_version=str(meta.get("format_version", ""))),
    )
    return plan


def _json_attr(g, name):
    import json

    raw = g.attrs.get(name)
    if raw is None:
        return None
    return json.loads(raw) if isinstance(raw, str) else raw
