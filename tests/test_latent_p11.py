"""PR-5 pin P11 -- the MIGRATION pin: the seam's ``logQ`` vs a rebuilt ``logq_map``.

PLAN §6.3 P11 reads "latent ``logQ`` at ``(theta_fid, xi_hat)`` vs a rebuilt
``logq_map``, same jitter convention, nside-16 first", tolerance ``1e-10``.  Every
clause of that sentence is load-bearing:

**"a rebuilt ``logq_map``"** -- the reference here is built INDEPENDENTLY, in
numpy, from the materialized dense Kronecker basis ``Phi = Phi_sph (x) Phi_z``:
``f = Phi @ xi`` evaluated as a single ``(N_pix * N_z, M)`` matrix-vector
product, then ``logQ`` assembled straight from the definition of PLAN eq. (2).
The seam never forms that matrix -- it contracts ``row_fac_m[p] . phi_z[z]``,
which is the whole point of the factored basis (a dense ``Phi`` at production
rank ``M = 3780`` is 47.6 GB, PLAN §3.2).  At nside 16 with ``M = M_sph M_z =
144`` and a 48-node ``z`` subsample the dense form is **169.9 MB**, so the two
constructions can be compared voxel by voxel.

**"same jitter convention"** -- PLAN §3.3 / OWNER DECISION 2.  The reference is
rebuilt under ``jitter_mode = "factored-v1"``, ``j_sph = j_z = 1e-6``, the SAME
convention the seam consumes.  Comparing against a legacy joint-kernel ``gp3d``
table instead would fail by three orders of magnitude and would look like a bug
in the seam when it is not: the legacy-vs-factored basis delta is a REPORTED
diagnostic (pin P3), never a gate.  ``test_p11_reports_legacy_vs_factored_delta``
below measures it and asserts nothing about its size, exactly as PLAN §3.3
requires; it reproduces the plan's headline ``2.0e-3`` at the shipped production
rank.

**"nside-16 first"** -- and, at present, nside-16 ONLY.  The DESI-scale arm of
this pin cannot run: the DESI-scale ``q_gp3d.h5`` does not exist (finding
R2-SEV3-16) and the ``gp3d`` build OOM'd at 21.7 GB.  PLAN's PR-5 entry records
that the DESI-scale migration pin is **reported, not blocking**.  Its absence
from this file is that decision, not an oversight; do not "fix" it by weakening
the nside-16 arm.

MEASURED (this file, on the reference hardware): the seam and the dense rebuild
agree to ``max |dlogQ| = 9.7e-15`` over both members, on- and off-node ``b_GW``,
all 3072 pixels and all 48 reference ``z`` nodes -- five orders below the pin's
nominal ``1e-10``.  The gate is therefore set at ``1e-13``, the measurement with
one order of magnitude of headroom for GPU reduction-order drift, rather than at
the plan's nominal tolerance.  Loosening it back toward ``1e-10`` would hide a
real regression in the basis or the normalizer.

The f32 ``row_fac`` leaf that SHIPS (``LatentQPlan.row_fac`` is f32, 11.7 MB at
production rank) is a storage decision, not part of the basis identity, so it is
pinned in its own arm at its own measured size (``3.3e-8``) -- see
``test_p11_f32_row_factor_storage_delta``.
"""
from __future__ import annotations

import time

import healpy as hp
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.core.types import EMCatalog, SurveyParams
from darksirens.likelihood.latent_q import (
    LatentQPlan,
    footprint_row_map,
    latent_logq_at,
    latent_logq_rows,
    on_footprint_mask,
    rho_from_moments,
)
from darksirens.redshift.completion import latent_member_logq_rows
from darksirens.redshift.grid import zgrid
from darksirens.redshift.latent_field import (
    build_latent_basis,
    legacy_lowrank_operator,
    lowrank_inducing_nodes,
    sky_moments,
)

#: nside 16 -- 3072 pixels.  Small enough that the dense Kronecker basis fits in
#: memory, large enough that the footprint/off-footprint split is a real one.
NSIDE = 16
NPIX = hp.nside2npix(NSIDE)

Z_DEPTH = 0.30
M_SPH, M_Z = 24, 6
#: ``ls_sph`` must exceed the Fibonacci node spacing ``~2/sqrt(M_sph) = 0.41``
#: (the builder's own resolution guard, ``cli/build_latent_field.py:102``), and
#: ``ls_z`` the zeta node spacing ``log1p(0.30)/(M_z - 1) = 0.052``.
LS_SPH, LS_Z = 0.6, 0.12
#: The builder's shipped ``b_GW`` grid (``--n-b-nodes 33 --b-max 4``).
N_B, B_MAX = 33, 4.0
M_DRAW = 2
#: The ``z`` nodes at which the dense reference is formed.  The full below-depth
#: block is 379 nodes at ``DARKSIRENS_ZMAX=1.0``; 3072 x 379 x 144 would be
#: 1.34 GB of dense basis, and 48 nodes spanning the same interval is already a
#: far denser sampling of ``phi_z`` than the 6 basis functions can resolve.
N_Z_REF = 48

#: PLAN §6.3 P11's nominal tolerance.  The gate actually applied is
#: :data:`TOL_P11`, set from the measurement (see the module docstring).
TOL_P11_NOMINAL = 1e-10
TOL_P11 = 1e-13
#: The f32 ``row_fac`` storage arm: measured 3.3e-8, gated with 30x headroom.
TOL_F32_STORAGE = 1e-6


# --------------------------------------------------------------------- fixture

def _sky():
    """nside-16 pixel centres and a ~62% footprint (production is 61.5%).

    The footprint is a declination cut rather than a random subset so that it is
    a connected region with a real boundary, as the DESI footprint is; the
    complement is 38.5% of the sky, matching the production gather statistic
    (49,143 union rows vs 30,470 footprint rows) that pin P13b quotes.
    """
    vec = np.column_stack(hp.pix2vec(NSIDE, np.arange(NPIX)))
    fit_pixels = np.flatnonzero(vec[:, 2] > -0.24)
    return vec, fit_pixels


@pytest.fixture(scope="module")
def ref():
    """Basis, seam plan, and the dense-Kronecker reference -- built once.

    The two halves are deliberately built by DIFFERENT code: the plan by the
    shipped :func:`build_latent_basis` / :func:`sky_moments` (jax), the
    reference by an explicit ``np.einsum`` Kronecker product and a numpy
    matrix-vector product.  They share only ``phi_sph``, ``phi_z`` and ``xi`` --
    i.e. the basis rows themselves, which pin P1 owns.  What P11 tests is
    everything downstream of those rows: the factored contraction, the moment
    tables, the barycentric ``b`` interpolation, the closed-form ``rho``, the
    footprint routing and the depth relaxation.
    """
    t0 = time.time()
    zg = np.asarray(zgrid)
    n_grid = zg.size
    below = zg <= Z_DEPTH
    n_sub = int(below.sum())
    z_sub = zg[:n_sub]

    vec, fit_pixels = _sky()
    n_fit = int(fit_pixels.size)

    basis = build_latent_basis(
        vec, np.log1p(z_sub), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_DEPTH, ls_sph=LS_SPH, ls_z=LS_Z,
        footprint_rows=fit_pixels)

    rng = np.random.default_rng(20260817)          # seeded: the pin is fixed
    f_p = rng.uniform(0.5, 1.0, size=n_fit)
    xi = rng.normal(size=(M_DRAW, M_SPH * M_Z)) * 0.4

    phi_sph = np.asarray(basis.phi_sph)            # (NPIX, M_SPH)
    phi_z_sub = np.asarray(basis.phi_z_out)        # (n_sub, M_Z)

    # The seam's row factor, in f64.  The SHIPPED leaf is f32 (PLAN §3.5); that
    # truncation is a storage choice pinned separately below, and folding it in
    # here would turn a basis-identity pin into a float32 pin.
    row_fac_fit = np.stack([phi_sph[fit_pixels] @ x.reshape(M_SPH, M_Z)
                            for x in xi])          # (M_DRAW, n_fit, M_Z)

    k = np.arange(N_B)
    b_nodes = 0.5 * B_MAX * (1.0 - np.cos(np.pi * k / (N_B - 1)))
    # Exactly as the builder does it: the moments must see the SAME row factors
    # the seam consumes, or eq. (4) closes only to ~b|f| eps (sky_moments'
    # docstring measures 2.7e-7 at the production corner).
    A_sub, B_sub = sky_moments(basis, xi, b_nodes, f_p, row_fac=row_fac_fit)

    def _pad(t):
        out = np.zeros((M_DRAW, N_B, n_grid))
        out[:, :, :n_sub] = np.asarray(t)
        return out

    phi_z_full = np.zeros((n_grid, M_Z))
    phi_z_full[:n_sub] = phi_z_sub
    pad_row = np.zeros((M_DRAW, 1, M_Z))

    def _plan(row_fac_fit_arr, A_arr, B_arr):
        rf = np.concatenate([row_fac_fit_arr,
                             pad_row.astype(row_fac_fit_arr.dtype)], axis=1)
        return LatentQPlan(
            phi_z=jnp.asarray(phi_z_full), below_depth=jnp.asarray(below),
            row_fac=jnp.asarray(rf), A=jnp.asarray(A_arr), B=jnp.asarray(B_arr),
            b_nodes=jnp.asarray(b_nodes), P_F=float(n_fit),
            F_F=float(f_p.sum()), m_sph=M_SPH, m_z=M_Z)

    plan = _plan(row_fac_fit, _pad(A_sub), _pad(B_sub))

    # The SHIPPED f32 twin: f32 row factors AND moments rebuilt from them, which
    # is what ``build_latent_field`` writes to the artifact.
    row_fac32 = row_fac_fit.astype(np.float32)
    A32, B32 = sky_moments(basis, xi, b_nodes, f_p, row_fac=row_fac32)
    plan32 = _plan(row_fac32, _pad(A32), _pad(B32))

    row_map = footprint_row_map(np.arange(NPIX), fit_pixels, n_fit)
    on_fp = np.asarray(on_footprint_mask(row_map, n_fit))
    assert int(on_fp.sum()) == n_fit

    # ---- the independent reference: materialize Phi = Phi_sph (x) Phi_z ----
    z_idx = np.unique(np.linspace(0, n_sub - 1, N_Z_REF).astype(int))
    t_dense = time.time()
    Phi_dense = np.einsum("pi,ga->pgia", phi_sph, phi_z_sub[z_idx]).reshape(
        NPIX * z_idx.size, M_SPH * M_Z)
    dense_mb = Phi_dense.nbytes / 1e6
    t_dense = time.time() - t_dense

    # A smooth sky-aggregate completeness; only its below-depth values matter.
    C_full = 0.85 * np.exp(-(zg / 0.22) ** 2)

    def reference_logq(m, b):
        """``logQ_m(p, z)`` at the reference ``z`` nodes, straight from PLAN eq. (2).

        No seam code is touched: ``f`` comes from the dense ``Phi @ xi``,
        ``rho`` from the raw footprint sums (not from the ``(A, B)`` tables, not
        via the Chebyshev interpolation), and the footprint mask is applied to
        the whole bracket.
        """
        f = (Phi_dense @ xi[m]).reshape(NPIX, z_idx.size)
        e = np.exp(b * f[fit_pixels])
        c = C_full[z_idx]
        rho = np.log((e.sum(axis=0) - c * (f_p @ e)) / (n_fit - c * f_p.sum()))
        lq = b * f - rho[None, :]
        lq[~on_fp] = 0.0
        return lq, rho

    def seam_logq(m, b, *, p=None):
        """``logQ_m`` on the full grid through the SHIPPED seam path.

        ``latent_q.rho_from_moments`` + ``latent_q.latent_logq_rows`` is the
        object under test rather than ``completion.latent_member_logq_rows``,
        because the latter is the same arithmetic wrapped in an ``EMCatalog``
        accessor layer -- ``test_p11_completion_entry_point_is_the_same_seam``
        pins that the wrapper adds nothing, so the pin reads at the level where
        the algebra lives.
        """
        pl = plan if p is None else p
        rho = rho_from_moments(pl.A[m], pl.B[m], jnp.asarray(C_full), b,
                               pl.b_nodes, pl.P_F, pl.F_F, pl.below_depth)
        rows = jnp.asarray(pl.row_fac)[m][jnp.asarray(row_map)]
        return (np.asarray(latent_logq_rows(pl, rows, rho, b, jnp.asarray(on_fp))),
                np.asarray(rho))

    print(f"\n[P11] nside={NSIDE} npix={NPIX} n_fit={n_fit} "
          f"off-footprint={(1 - n_fit / NPIX):.3f} (production 0.38)")
    print(f"[P11] rank M = M_sph*M_z = {M_SPH}*{M_Z} = {M_SPH * M_Z}; "
          f"dense Phi {Phi_dense.shape} = {dense_mb:.1f} MB built in "
          f"{t_dense:.2f} s (47.6 GB at production rank M=3780)")
    print(f"[P11] fixture built in {time.time() - t0:.1f} s")

    return dict(basis=basis, plan=plan, plan32=plan32, xi=xi, f_p=f_p,
                fit_pixels=fit_pixels, n_fit=n_fit, row_map=row_map,
                on_fp=on_fp, b_nodes=b_nodes, C_full=C_full, z_idx=z_idx,
                n_sub=n_sub, below=below, phi_sph=phi_sph, phi_z_sub=phi_z_sub,
                reference_logq=reference_logq, seam_logq=seam_logq,
                Phi_dense=Phi_dense)


# ------------------------------------------------------------------- the pin

def test_p11_latent_logq_matches_dense_kronecker_rebuild(ref):
    """P11 proper: seam ``logQ`` == dense-rebuild ``logQ``, every member.

    ``b_GW`` sits exactly on a Chebyshev-Lobatto node here, so the barycentric
    interpolation of the moment tables is exact by its own pole handling and
    what is left is purely the migration: factored contraction vs dense
    ``Phi @ xi``, closed-form ``rho`` vs raw footprint sums.  The off-node case
    is the next test.
    """
    b = float(ref["b_nodes"][8])                    # 0.58579, an exact node
    worst = 0.0
    for m in range(M_DRAW):
        lq_ref, rho_ref = ref["reference_logq"](m, b)
        lq_seam, rho_seam = ref["seam_logq"](m, b)
        d_lq = np.abs(lq_seam[:, ref["z_idx"]] - lq_ref)
        d_rho = np.abs(rho_seam[ref["z_idx"]] - rho_ref)
        print(f"[P11] on-node  b={b:.5f} m={m}: max|dlogQ|={d_lq.max():.3e} "
              f"max|drho|={d_rho.max():.3e} (max|logQ|={np.abs(lq_ref).max():.3f})")
        worst = max(worst, float(d_lq.max()))
        assert d_lq.max() < TOL_P11
    print(f"[P11] worst on-node |dlogQ| = {worst:.3e}  "
          f"(gate {TOL_P11:.0e}, PLAN nominal {TOL_P11_NOMINAL:.0e})")


def test_p11_holds_off_the_b_interpolation_nodes(ref):
    """The same pin at a ``b_GW`` between nodes -- the production case.

    ``b_GW`` is sampled, so it almost never lands on a node.  Pin P9 owns the
    moment interpolation error in isolation (1e-6 over 200 random ``(b, c)``);
    what this test says is that at the shipped 33-node grid the interpolation
    contributes nothing measurable to P11 either -- the exponential moments are
    analytic in ``b`` and Chebyshev-Lobatto interpolation of an analytic
    function at 33 nodes is already at round-off.
    """
    b = 2.3171                                      # strictly between nodes
    assert not np.any(np.asarray(ref["b_nodes"]) == b)
    for m in range(M_DRAW):
        lq_ref, _ = ref["reference_logq"](m, b)
        lq_seam, _ = ref["seam_logq"](m, b)
        d = np.abs(lq_seam[:, ref["z_idx"]] - lq_ref)
        print(f"[P11] off-node b={b:.5f} m={m}: max|dlogQ|={d.max():.3e} "
              f"(max|logQ|={np.abs(lq_ref).max():.3f})")
        assert d.max() < TOL_P11


def test_p11_hot_path_two_node_kernel_agrees_with_the_rebuild(ref):
    """``latent_logq_at`` -- what the PE/selection gather actually calls.

    ``prior.eval_dark_member_completion_latent`` never forms a row block; it
    gathers two ``zgrid`` nodes per PE sample and calls
    :func:`latent_logq_at`.  P11 has to bind THAT expression to the rebuild,
    not only the state-prep row form, or the pin would certify a function the
    hot path does not use.
    """
    b = float(ref["b_nodes"][8])
    m = 1
    lq_ref, _ = ref["reference_logq"](m, b)
    plan = ref["plan"]
    rho = rho_from_moments(plan.A[m], plan.B[m], jnp.asarray(ref["C_full"]), b,
                           plan.b_nodes, plan.P_F, plan.F_F, plan.below_depth)

    rng = np.random.default_rng(3)
    rows = rng.integers(0, NPIX, size=4096)
    cols = rng.integers(0, ref["z_idx"].size, size=4096)
    nodes = ref["z_idx"][cols]

    got = np.asarray(latent_logq_at(
        jnp.asarray(plan.row_fac)[m][jnp.asarray(ref["row_map"])[rows]],
        jnp.asarray(plan.phi_z)[nodes],
        rho[nodes],
        b,
        jnp.asarray(ref["on_fp"][rows])))
    d = np.abs(got - lq_ref[rows, cols])
    print(f"[P11] hot-path latent_logq_at over 4096 gathered (row, node) pairs: "
          f"max|dlogQ|={d.max():.3e}")
    assert d.max() < TOL_P11


def test_p11_completion_entry_point_is_the_same_seam(ref):
    """``completion.latent_member_logq_rows`` == the ``latent_q`` row path.

    The completion-layer wrapper re-implements the seam expression inline over
    ``EMCatalog`` leaves (it does not call :func:`latent_logq_rows`), so "the
    wrapper adds nothing" is a claim that needs a pin rather than a reading.
    Bit-identity is the right bar: the two run the same ops on the same arrays
    in the same order.
    """
    b = float(ref["b_nodes"][8])
    plan = ref["plan"]
    n_rows = NPIX
    cat = EMCatalog(
        apix=1.0,
        zgals=jnp.zeros((n_rows, 1)), dzgals=jnp.ones((n_rows, 1)),
        wgals=jnp.zeros((n_rows, 1)), ngals=jnp.zeros(n_rows, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, zgrid.size)),
        dN_obs_kde=None, pixel_to_cache_idx=None,
        latent_row_fac=plan.row_fac, latent_phi_z=plan.phi_z,
        latent_row_map=jnp.asarray(ref["row_map"]),
        latent_on_fp=jnp.asarray(ref["on_fp"]),
        latent_A=plan.A, latent_B=plan.B, latent_b_nodes=plan.b_nodes,
        latent_P_F=plan.P_F, latent_F_F=plan.F_F)
    survey = SurveyParams(n0=1.0, z50=0.15, w=0.08, delta=0.0,
                          b_miss=b, alpha_miss=1.0, z_depth=Z_DEPTH)

    got = np.asarray(latent_member_logq_rows(
        cat, survey, jnp.asarray(ref["C_full"]), 1))
    want, _ = ref["seam_logq"](1, b)
    assert np.array_equal(got, want), (
        f"completion wrapper drifted from the seam: "
        f"max|d| = {np.abs(got - want).max():.3e}")


def test_p11_reference_and_seam_are_bit_zero_where_Q_is_one(ref):
    """The two conventions that make ``Q == 1`` regions exact, not approximate.

    Off-footprint rows (38.5% of the sky here) and nodes above ``z_depth`` must
    be BIT-zero in ``logQ``, in the rebuild and in the seam alike -- pin P13b
    for the sky half, PLAN §4.2's "no information, no modulation" for the depth
    half.  A tolerance-level agreement would not do: eq. (4) conserves the
    off-footprint budget block only because ``Q`` is exactly 1 there.
    """
    b = 2.3171
    off = ~ref["on_fp"]
    above = ~np.asarray(ref["below"])
    for m in range(M_DRAW):
        lq_ref, _ = ref["reference_logq"](m, b)
        lq_seam, _ = ref["seam_logq"](m, b)
        assert np.all(lq_ref[off] == 0.0)
        assert np.all(lq_seam[off, :] == 0.0)
        assert np.all(lq_seam[:, above] == 0.0)
    print(f"[P11] bit-zero: {int(off.sum())}/{NPIX} off-footprint rows, "
          f"{int(above.sum())}/{above.size} above-depth nodes")


def test_p11_f32_row_factor_storage_delta(ref):
    """The SHIPPED f32 ``row_fac`` leaf, measured against the same rebuild.

    ``LatentQPlan.row_fac`` is f32 (11.7 MB at production rank against 23.4 MB
    in f64, times the 256x concurrency multiplier PLAN §2.4 worries about), and
    the builder rebuilds ``(A, B)`` from those same f32 rows so that eq. (4)
    still closes exactly.  What that costs against an f64 rebuild is
    ``~ b |f| eps_f32``, MEASURED at 3.3e-8 here -- five orders above the
    basis-identity pin and seven below one nat, i.e. invisible to the
    likelihood.  Gated at 1e-6 (30x headroom) so a change of storage dtype or a
    lost moment-rebuild shows up as a failure rather than as drift.
    """
    b = float(ref["b_nodes"][8])
    worst = 0.0
    for m in range(M_DRAW):
        lq_ref, _ = ref["reference_logq"](m, b)
        lq_seam, _ = ref["seam_logq"](m, b, p=ref["plan32"])
        d = float(np.abs(lq_seam[:, ref["z_idx"]] - lq_ref).max())
        print(f"[P11] f32 row_fac m={m}: max|dlogQ|={d:.3e}")
        worst = max(worst, d)
        assert d < TOL_F32_STORAGE
    print(f"[P11] f32 storage cost = {worst:.3e} vs f64 basis identity "
          f"{TOL_P11:.0e} gate")


# -------------------------------------------------------------- P3, reported

@pytest.mark.parametrize("m_sph,m_z,ls_sph,ls_z,tag", [
    (M_SPH, M_Z, LS_SPH, LS_Z, "this test's rank"),
    (64, 8, 0.2, 0.039, "shipped hyperparameters, M_sph=64"),
    (315, 8, 0.2, 0.039, "SHIPPED production rank (cli defaults)"),
])
def test_p11_reports_legacy_vs_factored_delta(m_sph, m_z, ls_sph, ls_z, tag):
    """PLAN §3.3 / pin P3: the legacy-vs-factored basis delta, REPORTED.

    This is the number that makes P11's "same jitter convention" clause
    mandatory.  ``legacy_lowrank_operator`` builds ``chol(K_joint + (1e-4 amp^2
    + 1e-9) I)`` on the flattened nodes; ``build_latent_basis`` builds
    ``chol(K_sph + 1e-6 I) (x) chol(K_z + 1e-6 I)``.  The joint kernel is the
    same object either way (``K_sph (x) K_z == _sphere_z_kernel``), so the whole
    delta is where the jitter is put -- and it is NOT small.

    Nothing here is gated.  The only assertion is that the delta is finite and
    strictly positive: the two conventions genuinely differ, and a test that
    found them equal would mean the factored path had silently reverted to the
    legacy builder.  Sizing it is the point; policing it is pin P1's job, and
    P1 compares the factored basis against ITS OWN Cholesky, never against this.
    """
    vec, _ = _sky()
    sub_pix = np.arange(0, NPIX, 16)                # 192 pixels
    z_sub = np.linspace(0.0, Z_DEPTH, 12)

    basis = build_latent_basis(
        vec[sub_pix], np.log1p(z_sub), n_inducing_sphere=m_sph,
        n_inducing_z=m_z, z_node_hi=Z_DEPTH, ls_sph=ls_sph, ls_z=ls_z)
    Zn, Zz = lowrank_inducing_nodes(m_sph, m_z, Z_DEPTH)
    X_n = np.repeat(vec[sub_pix], z_sub.size, axis=0)
    X_z = np.tile(np.log1p(z_sub), sub_pix.size)
    t0 = time.time()
    Phi_leg = np.asarray(legacy_lowrank_operator(
        Zn, Zz, jnp.asarray(X_n), jnp.asarray(X_z),
        amp=1.0, ls_sph=ls_sph, ls_z=ls_z)[0])
    # The Kronecker basis on the SAME voxel ordering (v = p * N_z + g, node
    # ordering i = i_sph * M_z + i_z -- lowrank_inducing_nodes' convention).
    Phi_kron = np.einsum("pi,ga->pgia", np.asarray(basis.phi_sph),
                         np.asarray(basis.phi_z_out)).reshape(Phi_leg.shape)
    delta = float(np.abs(Phi_kron - Phi_leg).max())
    rel = delta / float(np.abs(Phi_leg).max())
    print(f"[P3] legacy-vs-factored |dPhi|: M_sph={m_sph:4d} M_z={m_z:2d} "
          f"ls=({ls_sph}, {ls_z})  max={delta:.3e}  rel={rel:.3e}  "
          f"[{tag}]  ({time.time() - t0:.1f} s)")
    assert np.isfinite(delta) and delta > 0.0
