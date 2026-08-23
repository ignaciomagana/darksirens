"""PR-8 pins: the ``amp(z)`` support, and the sense in which it is inert.

PR-8 adds ONE modelling knob and one refusal to be careless with it.

* ``amp(z)`` multiplies the redshift factor rows of the latent basis.  It is
  pinned at exactly ``1.0`` at and below the FITTED depth, because PLAN §4.3
  states that ``(b, xi)`` and ``(amp, xi)`` enter only through ``b*xi`` and
  ``b*amp``: where the counts constrain the field there is one clustering
  amplitude and it is ``b_gal``.  A profile that touched the fitted region
  would re-open that degeneracy, so the pins below check the below-depth
  branch BIT-wise, not to a tolerance.
* Above the depth there are no counts (R1: 99.994% of the missing budget lives
  there and the measured in-support fraction is 6e-5), so ``amp_hi`` is an
  ASSUMPTION.  ``amp_hi = 0`` is the shipped convention -- ``Q == 1`` above the
  depth, PLAN §4.2's stated under-dispersion -- and the gate this module exists
  for is that it reproduces that convention BIT-IDENTICALLY, so the ``amp = 0``
  row of the PR-8 table is the legacy analysis rather than a near-miss of it.

What is NOT tested here, because it cannot be: that any particular ``amp_hi``
is right.  Nothing in this pipeline constrains it, which is why PR-8 ships a
sensitivity scan (OWNER DECISION 7) and why the numbers live in
``experiments/field_level_plan/pr8/REPORT.md`` rather than in a gate.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("jax")
import h5py
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.likelihood.factory import (
    _latent_guard_isotropy,
    _latent_guard_resolution,
    _resolve_latent_leaves,
    latent_artifact_fingerprint,
)
from darksirens.likelihood.latent_q import load_latent_plan, rho_from_moments
from darksirens.redshift.grid import zgrid
from darksirens.redshift.latent_field import (
    amp_profile,
    build_latent_basis,
    growth_factor,
    sky_constant_coeffs,
    sky_moments,
)

_Z = np.asarray(zgrid)
N_GRID = int(_Z.size)

NSIDE = 4
Z_DEPTH = 0.30
Z_NODE_HI = 1.0
M_SPH, M_Z = 24, 12
M_DRAW, N_B = 2, 5
LS_SPH, LS_Z = 0.8, 0.11
FIT_PIXELS = np.arange(8, dtype=np.int64)
ROW_PIXELS = np.arange(0, 16, 2, dtype=np.int64)
F_P = np.linspace(0.55, 0.95, FIT_PIXELS.size)


# --------------------------------------------------------------- the profile

def test_amp_is_bit_one_at_and_below_the_depth():
    """PLAN §4.3's constraint, at the strongest reading available.

    Not "close to 1" and not "1 to double precision": the literal float 1.0, on
    every below-depth node, for every ``amp_hi`` and both profile shapes.  That
    is what makes the fitted region UNTOUCHED (``x * 1.0 == x`` exactly) rather
    than perturbed at the last bit -- and an ``amp`` that perturbed the fitted
    region at any level would be a second clustering amplitude next to
    ``b_gal``, which is exactly what §4.3 closes.
    """
    z = np.concatenate([_Z[_Z <= Z_DEPTH], [Z_DEPTH]])
    for amp_hi in (0.0, 0.05, 0.4, 1.0, 3.0):
        for kind in ("step", "growth"):
            a = amp_profile(z, z_depth=Z_DEPTH, amp_hi=amp_hi, kind=kind)
            assert np.all(a == 1.0), (amp_hi, kind)


def test_amp_is_the_assumed_constant_above_the_depth():
    z = _Z[_Z > Z_DEPTH]
    for amp_hi in (0.0, 0.05, 0.1, 0.2, 0.4):
        a = amp_profile(z, z_depth=Z_DEPTH, amp_hi=amp_hi, kind="step")
        assert np.all(a == amp_hi)


def test_growth_profile_decays_and_starts_at_amp_hi():
    """``growth`` is the same assumed number carried by ``D(z)/D(z_depth)``.

    Two properties are what make it the CONSERVATIVE shape at fixed ``amp_hi``:
    it starts at ``amp_hi`` just above the depth and it decreases with z, so it
    can never assume MORE clustering than the step profile anywhere.
    """
    z = _Z[_Z > Z_DEPTH]
    a = amp_profile(z, z_depth=Z_DEPTH, amp_hi=0.4, kind="growth", Om0=0.3075)
    assert np.all(np.diff(a) <= 0.0)
    assert a[0] == pytest.approx(0.4, rel=2e-3)
    assert np.all(a <= 0.4 + 1e-12)
    # D is normalized to 1 at z = 0 and falls like ~1/(1+z) in matter domination.
    D = growth_factor(np.array([0.0, 1.0]), 0.3075)
    assert D[0] == pytest.approx(1.0, rel=1e-6)
    assert 0.5 < D[1] < 0.65


def test_amp_profile_refuses_a_negative_amplitude():
    with pytest.raises(ValueError, match="must be >= 0"):
        amp_profile(_Z, z_depth=Z_DEPTH, amp_hi=-0.1)


def test_amp_profile_refuses_an_unknown_kind():
    with pytest.raises(ValueError, match="kind must be"):
        amp_profile(_Z, z_depth=Z_DEPTH, amp_hi=0.1, kind="linear")


# ------------------------------------------------- the basis: the legacy gate

def _basis(**kw):
    z_out = _Z[_Z <= Z_NODE_HI]
    z_fine = np.linspace(1e-4, Z_DEPTH, 64)
    return build_latent_basis(
        np.eye(3), np.log1p(z_out), n_inducing_sphere=M_SPH,
        n_inducing_z=M_Z, z_node_hi=Z_NODE_HI, ls_sph=LS_SPH, ls_z=LS_Z,
        zeta_fine=np.log1p(z_fine), **kw)


def test_amp_one_profile_is_bit_identical_to_no_profile():
    """THE PR-8 GATE: ``amp(z)`` reduces to a constant at the legacy value
    bit-identically.

    ``amp_hi = 1`` makes the profile the constant 1 over the WHOLE grid -- the
    legacy value PLAN §4.3 pins -- and the basis it produces is compared to the
    no-profile basis array by array with ``==``.  It holds because of where the
    profile is applied: a row scaling of ``phi_z`` by the literal 1.0, not a
    rescaling of the kernel (the scalar ``amp`` argument), whose absolute
    per-factor jitter does not scale with it and which therefore would NOT be
    bit-exact.
    """
    legacy = _basis()
    amped = _basis(amp_hi=1.0, amp_z_depth=Z_DEPTH, amp_kind="step")
    for name in ("phi_sph", "phi_z_out", "phi_z_fine", "proj_sph", "L_sph",
                 "L_z"):
        a = np.asarray(getattr(legacy, name))
        b = np.asarray(getattr(amped, name))
        assert a.shape == b.shape
        assert np.array_equal(a, b), f"{name} moved under amp == 1"


def test_amp_zero_zeroes_the_field_above_the_depth_and_nothing_below():
    legacy = _basis()
    amped = _basis(amp_hi=0.0, amp_z_depth=Z_DEPTH, amp_kind="step")
    z_out = _Z[_Z <= Z_NODE_HI]
    below, above = z_out <= Z_DEPTH, z_out > Z_DEPTH
    p0, p1 = np.asarray(legacy.phi_z_out), np.asarray(amped.phi_z_out)
    assert np.array_equal(p0[below], p1[below])
    assert np.all(p1[above] == 0.0)
    # ... and the legacy basis is NOT zero there, so the test above is a
    # statement about amp and not about the basis dying out on its own.
    assert np.any(np.abs(p0[above]) > 1e-3)


def test_amp_scales_the_redshift_factor_rows_exactly():
    """The field is ``row_fac . phi_z``, so a row scaling of ``phi_z`` by
    ``amp(z)`` scales the field by ``amp(z)`` at every pixel -- exactly, and
    with the SAME xi.  Checked on the rows themselves, where the statement is
    bit-exact (the contraction that follows re-associates the sum)."""
    legacy = _basis()
    amped = _basis(amp_hi=0.25, amp_z_depth=Z_DEPTH, amp_kind="step")
    z_out = _Z[_Z <= Z_NODE_HI]
    a = amp_profile(z_out, z_depth=Z_DEPTH, amp_hi=0.25)
    expect = np.asarray(legacy.phi_z_out) * a[:, None]
    assert np.array_equal(np.asarray(amped.phi_z_out), expect)


def test_amp_hi_without_a_depth_is_refused():
    with pytest.raises(ValueError, match="requires amp_z_depth"):
        _basis(amp_hi=0.2)


def test_meta_carries_the_profile_only_when_there_is_one():
    """``basis_meta`` is the loader's ONE test for 'does this artifact model
    the field above the depth?', and it is also hashed into guard 1.  So a
    build without a profile must carry no amp keys at all -- otherwise every
    pre-PR-8 artifact would differ from its own rebuild in metadata alone."""
    assert set(_basis().meta) == {
        "jitter_mode", "j_sph", "j_z", "amp", "ls_sph", "ls_z", "M_sph", "M_z",
        "z_node_hi"}
    m = _basis(amp_hi=0.2, amp_z_depth=Z_DEPTH).meta
    assert m["amp_hi"] == 0.2 and m["amp_z_depth"] == Z_DEPTH
    assert m["amp_kind"] == "step" and "amp_Om0" in m


# ------------------------------------------------------------- the artifact

def _write_artifact(path, *, amp_hi=None, amp_kind="step", amp_z_depth=None,
                    z_node_hi=Z_DEPTH, z_top=None, m_z=M_Z, f_p=F_P):
    """A tiny but structurally complete anchor, optionally with a profile.

    Mirrors ``tests/test_latent_factory._write_artifact`` (same datasets, same
    attrs); the difference is that ``z_sub`` may run ABOVE the depth, which is
    what a PR-8 anchor does so that ``rho`` has moments where the field is
    nonzero.
    """
    f_p = np.asarray(f_p, dtype=np.float64)
    n_fit = int(FIT_PIXELS.size)
    z_sub = _Z[_Z <= (Z_DEPTH if z_top is None else z_top)]
    n_sub = int(z_sub.size)
    rng = np.random.default_rng(11)
    n_th = 5
    basis_meta = dict(
        jitter_mode="factored-v1", j_sph=1e-6, j_z=1e-6, amp=1.0,
        ls_sph=float(LS_SPH), ls_z=float(LS_Z), M_sph=int(M_SPH),
        M_z=int(m_z), z_node_hi=float(z_node_hi))
    if amp_hi is not None:
        basis_meta.update(
            amp_hi=float(amp_hi), amp_kind=str(amp_kind),
            amp_z_depth=float(Z_DEPTH if amp_z_depth is None else amp_z_depth),
            amp_Om0=0.3075)
    with h5py.File(path, "w") as f:
        g = f.create_group("latent_field")
        g.create_dataset("row_fac", data=rng.normal(
            size=(M_DRAW, n_fit, m_z)).astype(np.float32))
        g.create_dataset("A_moments",
                         data=np.full((M_DRAW, N_B, n_sub), float(n_fit)))
        g.create_dataset("B_moments",
                         data=np.full((M_DRAW, N_B, n_sub), float(f_p.sum())))
        g.create_dataset("dA_moments",
                         data=np.zeros((M_DRAW, N_B, n_sub, n_th)))
        g.create_dataset("dB_moments",
                         data=np.zeros((M_DRAW, N_B, n_sub, n_th)))
        g.create_dataset("b_nodes", data=np.linspace(0.0, 4.0, N_B))
        g.create_dataset("z_sub", data=z_sub)
        g.create_dataset("fit_pixels", data=FIT_PIXELS.astype(np.int32))
        g.create_dataset("completeness", data=f_p)
        g.create_dataset("sensitivity_S", data=np.zeros((M_SPH * m_z, n_th)))
        g.attrs["sensitivity_labels"] = json.dumps(
            ["M0hat", "sigma_M", "delta", "Om0", "b_gal"])
        g.attrs["P_F"] = float(n_fit)
        g.attrs["F_F"] = float(np.maximum(f_p, 1e-3).sum())
        g.attrs["theta_ref"] = json.dumps(
            dict(M0hat=-20.3, sigma_M=0.6, delta=0.0, Om0=0.315))
        g.attrs["basis_meta"] = json.dumps(basis_meta)
        g.attrs["nside"] = int(NSIDE)
        g.attrs["sha256"] = "0" * 64
        g.attrs["format_version"] = "darksirens-latent-field-1.0"
        g.create_dataset("counts", data=np.full((3, n_fit), 7.0))
        g.create_dataset("z_count_edges", data=np.linspace(0.0, Z_DEPTH, 4))
        g.create_dataset("shell_response", data=np.eye(3, n_sub))
        g.attrs["b_gal"] = 1.4
    return str(path)


def test_amp_zero_artifact_loads_bit_identically_to_a_pre_pr8_one(tmp_path):
    """THE PR-8 GATE, at the loader: an ``amp_hi = 0`` anchor and a pre-PR-8
    anchor of the same geometry produce the SAME PLAN, array for array.

    This is the row of the sensitivity table that is quoted as "the shipped
    convention", so it has to BE the shipped convention and not a rebuild of it
    that agrees to plotting accuracy.
    """
    legacy = load_latent_plan(_write_artifact(tmp_path / "legacy.h5"),
                              z_depth=Z_DEPTH)
    amp0 = load_latent_plan(
        _write_artifact(tmp_path / "amp0.h5", amp_hi=0.0), z_depth=Z_DEPTH)
    assert np.array_equal(np.asarray(legacy.phi_z), np.asarray(amp0.phi_z))
    assert np.array_equal(np.asarray(legacy.below_depth),
                          np.asarray(amp0.below_depth))
    assert np.array_equal(np.asarray(legacy.A), np.asarray(amp0.A))
    assert np.array_equal(np.asarray(legacy.B), np.asarray(amp0.B))
    assert np.array_equal(np.asarray(legacy.row_fac), np.asarray(amp0.row_fac))


def test_amp_zero_over_an_extended_grid_still_has_below_depth_support(tmp_path):
    """An ``amp_hi = 0`` anchor may still carry consumption rows above the
    depth (they cost nothing and keep the scan's rows on ONE grid).  The
    support must still stop at the depth, because that is where the field
    stops -- the rows above it are multiplied by a literal zero."""
    plan = load_latent_plan(
        _write_artifact(tmp_path / "amp0x.h5", amp_hi=0.0,
                        z_node_hi=Z_NODE_HI, z_top=Z_NODE_HI),
        z_depth=Z_DEPTH)
    sup = np.asarray(plan.below_depth)
    assert np.array_equal(sup, _Z <= Z_DEPTH)
    assert np.all(np.asarray(plan.phi_z)[~sup] == 0.0)


def test_amp_positive_extends_the_support_and_scales_the_rows(tmp_path):
    """With ``amp_hi > 0`` the support reaches above the depth -- that IS the
    rung -- and the rows there are the unscaled basis times ``amp_hi``."""
    kw = dict(z_node_hi=Z_NODE_HI, z_top=Z_NODE_HI)
    p0 = load_latent_plan(
        _write_artifact(tmp_path / "a0.h5", amp_hi=0.0, **kw), z_depth=Z_DEPTH)
    p2 = load_latent_plan(
        _write_artifact(tmp_path / "a2.h5", amp_hi=0.2, **kw), z_depth=Z_DEPTH)
    sup = np.asarray(p2.below_depth)
    covered = _Z <= Z_NODE_HI
    assert np.array_equal(sup, covered)
    assert sup.sum() > int((_Z <= Z_DEPTH).sum())
    below = _Z <= Z_DEPTH
    phi0, phi2 = np.asarray(p0.phi_z), np.asarray(p2.phi_z)
    # below the depth the two anchors are the same field, bit for bit
    assert np.array_equal(phi0[below], phi2[below])
    # above it, amp_hi = 0 is zero and amp_hi = 0.2 is not
    assert np.all(phi0[covered & ~below] == 0.0)
    assert np.any(np.abs(phi2[covered & ~below]) > 0.0)
    # and beyond the artifact's own coverage BOTH are zero: the assumption is
    # bounded by the nodes, it does not extrapolate to the top of the grid.
    assert np.all(phi2[~covered] == 0.0)


def test_loader_refuses_a_profile_stepping_somewhere_else(tmp_path):
    art = _write_artifact(tmp_path / "bad.h5", amp_hi=0.2, amp_z_depth=0.5,
                          z_node_hi=Z_NODE_HI, z_top=Z_NODE_HI)
    with pytest.raises(ValueError, match="stepping at"):
        load_latent_plan(art, z_depth=Z_DEPTH)


def test_loader_refuses_an_amp_anchor_that_misses_part_of_the_fitted_region(
        tmp_path):
    """The anchor's rows stop at 0.30 while the run's depth is 0.5: the field
    would be unmodelled inside the region the counts constrain."""
    art = _write_artifact(tmp_path / "short.h5", amp_hi=0.2, amp_z_depth=0.5)
    with pytest.raises(ValueError, match="must at minimum cover"):
        load_latent_plan(art, z_depth=0.5)


# ------------------------------------------------------------ the guards

def _opts(**kw):
    base = dict(lss_field_mode="latent", lss_field_artifact=None,
                lss_completion=None, use_LSS=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _catalogs():
    up = jnp.asarray(ROW_PIXELS, dtype=jnp.int32)
    return SimpleNamespace(
        unique_pixels_pe=up, unique_pixels_sel=up,
        field_occupied_pixels=jnp.asarray(FIT_PIXELS, dtype=jnp.int32),
        field_f_p_occ=jnp.asarray(F_P.astype(np.float32)),
        f_p_rows_pe=None)


def _leaves(art):
    return _resolve_latent_leaves(
        _opts(lss_field_artifact=art), _catalogs(), Z_DEPTH, NSIDE,
        int(ROW_PIXELS.size), int(ROW_PIXELS.size))


def test_support_leaf_is_installed_only_by_an_amp_anchor(tmp_path):
    """The static pytree-STRUCTURE branch that keeps every other run on the
    pre-PR-8 code path.  ``amp_hi = 0`` is deliberately on the ``None`` side:
    its support IS ``zgrid <= z_depth``, so installing a leaf would only give
    the consumers a second way to compute the same mask."""
    _, pe0, sel0 = _leaves(_write_artifact(tmp_path / "legacy.h5"))
    assert pe0["latent_support"] is None and sel0["latent_support"] is None
    _, pez, _ = _leaves(_write_artifact(tmp_path / "amp0.h5", amp_hi=0.0,
                                        z_node_hi=Z_NODE_HI, z_top=Z_NODE_HI))
    assert pez["latent_support"] is None
    _, pe2, sel2 = _leaves(_write_artifact(
        tmp_path / "amp2.h5", amp_hi=0.2, z_node_hi=Z_NODE_HI,
        z_top=Z_NODE_HI))
    sup = np.asarray(pe2["latent_support"])
    assert sup.shape == (N_GRID,) and sup.dtype == bool
    assert np.array_equal(sup, _Z <= Z_NODE_HI)
    assert pe2["latent_support"] is sel2["latent_support"]


def test_isotropy_guard_uses_the_FITTED_depth_not_the_node_range(tmp_path):
    """PR-8 separates the node range from the fitted depth, and the aspect
    ratio has to be evaluated where the counts see the field.

    The anchor below is isotropic at the fitted depth's midpoint (z = 0.15) and
    would look like a 2:1 pancake at the NODE range's midpoint -- purely
    because ``chi(z)`` grows faster than ``(1+z) dchi/dz``.  Measuring it there
    would refuse an anchor whose fitted geometry is exactly the one the
    pre-PR-8 guard passed.
    """
    plan = load_latent_plan(
        _write_artifact(tmp_path / "iso.h5", amp_hi=0.2, z_node_hi=Z_NODE_HI,
                        z_top=Z_NODE_HI), z_depth=Z_DEPTH)
    assert plan.meta["z_fit_depth"] == pytest.approx(Z_DEPTH)
    _latent_guard_isotropy(plan)                      # fitted depth: passes
    # Both fitted-depth sources removed -> the guard falls back to the node
    # range and refuses, which is the failure mode PR-8 had to fix.
    hi = dict(plan.meta)
    hi.pop("amp_z_depth")
    hi.pop("z_fit_depth")
    with pytest.raises(ValueError, match="anisotropic"):
        _latent_guard_isotropy(SimpleNamespace(
            meta=hi, theta_ref=plan.theta_ref, m_sph=plan.m_sph,
            m_z=plan.m_z))


def test_shell_edges_give_the_fitted_depth_without_a_profile(tmp_path):
    """An anchor with extended NODES and no profile -- the control arm of the
    scan -- is still judged at the depth its COUNTS reach.

    The last shell edge is the counts' own upper limit and is already a guard-1
    fingerprint array, so it says where the field is constrained without the
    artifact having to carry a new number.  Without this the control arm would
    be refused by the isotropy guard while the anchors it controls for pass,
    which would make the scan's rows incomparable.
    """
    plan = load_latent_plan(
        _write_artifact(tmp_path / "ctl.h5", z_node_hi=Z_NODE_HI,
                        z_top=Z_DEPTH), z_depth=Z_DEPTH)
    assert plan.meta.get("amp_hi") is None
    assert plan.meta["z_fit_depth"] == pytest.approx(Z_DEPTH)
    _latent_guard_isotropy(plan)


def test_resolution_guard_binds_on_the_node_range(tmp_path):
    """The radial spacing guard is over the NODES, so extending them without
    raising ``M_z`` is refused -- the low-rank GP would collapse to the prior
    above the depth while reporting convergence, and PR-8 would then be
    scanning an amplitude on a field that carries no structure at all."""
    art = _write_artifact(tmp_path / "coarse.h5", amp_hi=0.2, m_z=4,
                          z_node_hi=Z_NODE_HI, z_top=Z_NODE_HI)
    plan = load_latent_plan(art, z_depth=Z_DEPTH)
    with pytest.raises(ValueError, match="under-resolved in REDSHIFT"):
        _latent_guard_resolution(plan)


def test_fingerprint_separates_two_amp_anchors(tmp_path):
    """Two anchors differing only in the assumed ``amp_hi`` generate different
    ``Q`` over 99.99% of the missing budget, so guard 1's CONTENT digest must
    tell them apart -- while a pre-PR-8 anchor keeps the digest it always had
    (no amp keys are hashed when there is no profile)."""
    kw = dict(z_node_hi=Z_NODE_HI, z_top=Z_NODE_HI)
    d0 = latent_artifact_fingerprint(
        _write_artifact(tmp_path / "f0.h5", amp_hi=0.0, **kw))["content"]
    d2 = latent_artifact_fingerprint(
        _write_artifact(tmp_path / "f2.h5", amp_hi=0.2, **kw))["content"]
    assert d0 != d2
    # Same arrays, same everything except the profile -> the difference is the
    # profile, which is the property being pinned.
    d2b = latent_artifact_fingerprint(
        _write_artifact(tmp_path / "f2b.h5", amp_hi=0.2, **kw))["content"]
    assert d2 == d2b


# -------------------------------------------- eq. (4) above the fitted depth

def test_budget_identity_closes_above_the_depth_under_amp():
    """PLAN eq. (4), where PR-8 newly needs it.

    Above the fitted depth the consumed completeness is ``C := 0`` (``base_miss``
    relaxes to ``dN_exp``), so the identity ``sum_p (1 - f_p C) Q_p == sum_p (1
    - f_p C)`` reduces to ``sum_{p in F} Q_p == P_F``.  If ``rho`` were left
    zeroed there -- the pre-PR-8 behaviour, which is correct only because the
    field is zero -- an amp anchor would inject an un-normalized monopole of
    ``exp(b*amp*f)`` over the 99.99% of the missing budget that lives above the
    depth: a spurious change in the TOTAL missing count, not a placement.

    Built from the real ``sky_moments`` on a real amp basis, so this pins the
    builder and the seam against each other and not a formula against itself.
    """
    n_fit = 32
    rng = np.random.default_rng(4)
    vec = rng.normal(size=(n_fit, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    z_out = _Z[_Z <= Z_NODE_HI]
    f_p = np.linspace(0.2, 0.95, n_fit)
    basis = build_latent_basis(
        vec, np.log1p(z_out), n_inducing_sphere=M_SPH, n_inducing_z=M_Z,
        z_node_hi=Z_NODE_HI, ls_sph=LS_SPH, ls_z=LS_Z,
        amp_hi=0.4, amp_z_depth=Z_DEPTH)
    xi = rng.normal(size=(1, M_SPH * M_Z))
    # Chebyshev-LOBATTO nodes, as the builder writes them: ``interp_b`` is
    # barycentric on that grid (pin P9), so an evenly spaced table would be
    # interpolated with the wrong weights and the residual below would measure
    # the test's own mistake instead of eq. (4).
    k = np.arange(9)
    b_nodes = 0.5 * 3.0 * (1.0 - np.cos(np.pi * k / 8))
    A, B = sky_moments(basis, xi, b_nodes, f_p)
    P_F, F_F = sky_constant_coeffs(f_p)

    b = float(b_nodes[5])
    rho = rho_from_moments(A[0], B[0], jnp.zeros(z_out.size), b,
                           jnp.asarray(b_nodes), P_F, F_F)
    field = np.asarray(basis.proj_sph @ xi[0].reshape(M_SPH, M_Z)
                       @ np.asarray(basis.phi_z_out).T)      # (n_fit, N_z)
    Q = np.exp(b * field - np.asarray(rho)[None, :])
    resid = np.abs(Q.sum(axis=0) / P_F - 1.0)
    above = np.asarray(z_out) > Z_DEPTH
    assert resid.max() < 1e-12
    assert above.sum() > 10 and resid[above].max() < 1e-12
