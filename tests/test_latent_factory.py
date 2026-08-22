"""PR-5 pins for the likelihood-BUILD half of the latent seam (factory.py).

``tests/test_latent_seam.py`` pins the kernel and ``tests/test_latent_seam_e2e.py``
pins the stack; this module pins the WIRING: that an anchor artifact becomes the
right ``EMCatalog.latent_*`` leaves, and that every way of pairing it with an
inconsistent run is refused HOST-SIDE, before the first likelihood evaluation.

The guards are the point.  Each one names a way the generated ``Q`` would still
evaluate --- no NaN, no crash, a perfectly plausible logL --- while conserving a
missing-galaxy budget nobody consumes:

guard 1  the run's depth map is not the anchor's, so ``F_F`` (and with it the
         eq. (4) identity) is wrong.
guard 2  an Mpc-valued radial correlation length, which maps to zeta like
         ``H0`` and would act as a standard ruler.
guard 3  an inducing grid coarser than its own lengthscale: the low-rank GP
         collapses to the prior while reporting convergence.
guard 4  a pancake field, fitting radial modes to angular structure.
guard 6  a second modulation (a loaded Q table, or ``--use_lss``'s delta_g)
         multiplying the generated one.

Everything here builds a TINY artifact in ``tmp_path`` (n_fit = 8 pixels at
nside = 4, M = 24 x 6) rather than reading the 64 MB production one.
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
    _resolve_latent_leaves,
    _resolve_lss_field_mode,
)
from darksirens.redshift.grid import zgrid

_Z = np.asarray(zgrid)
N_GRID = int(_Z.size)

NSIDE = 4
Z_DEPTH = 0.30
M_SPH, M_Z = 24, 6
M_DRAW, N_B = 2, 5
#: Footprint = pixels 0..7; the catalog rows below interleave them with pixels
#: that are NOT in the footprint, so every test exercises the pad-row branch
#: the production run spends ~38% of its gathers in (pin P13b).
FIT_PIXELS = np.arange(8, dtype=np.int64)
ROW_PIXELS = np.arange(0, 16, 2, dtype=np.int64)   # 0,2,4,...,14: 4 in F, 4 out

#: Isotropic-and-resolved defaults.  sqrt(4pi/24) = 0.72 <= ls_sph = 0.8 and
#: log1p(0.3)/5 = 0.0525 <= ls_z = 0.11, and at z_ref = 0.15 the two map to
#: ~495 Mpc transverse vs ~502 Mpc radial (aspect 0.99).
LS_SPH, LS_Z = 0.8, 0.11
F_P = np.linspace(0.55, 0.95, FIT_PIXELS.size)


# --------------------------------------------------------------------- fixture

def _write_artifact(path, *, f_p=F_P, fit_pixels=FIT_PIXELS, nside=NSIDE,
                    m_sph=M_SPH, m_z=M_Z, ls_sph=LS_SPH, ls_z=LS_Z,
                    z_depth=Z_DEPTH, f_f=None, p_f=None, theta_ref=None):
    """A structurally complete (but tiny) PR-4 anchor artifact.

    Only the datasets ``load_latent_plan`` reads are written; the seam never
    looks at ``xi_hat`` / ``H_chol`` / ``counts`` online.  The moment values are
    arbitrary (the factory guards read ``F_F``, ``P_F`` and ``basis_meta``, not
    the moments), which is exactly the separation of concerns this module
    tests -- the eq. (4) NUMBERS are pinned by tests/test_latent_seam.py.
    """
    f_p = np.asarray(f_p, dtype=np.float64)
    fit_pixels = np.asarray(fit_pixels, dtype=np.int64)
    n_fit = int(fit_pixels.size)
    z_sub = _Z[_Z <= z_depth]
    n_sub = int(z_sub.size)
    rng = np.random.default_rng(11)
    n_th = 5

    with h5py.File(path, "w") as f:
        g = f.create_group("latent_field")
        g.create_dataset("row_fac", data=rng.normal(
            size=(M_DRAW, n_fit, m_z)).astype(np.float32))
        g.create_dataset("A_moments", data=np.full((M_DRAW, N_B, n_sub),
                                                   float(n_fit)))
        g.create_dataset("B_moments", data=np.full((M_DRAW, N_B, n_sub),
                                                   float(f_p.sum())))
        g.create_dataset("dA_moments", data=np.zeros((M_DRAW, N_B, n_sub, n_th)))
        g.create_dataset("dB_moments", data=np.zeros((M_DRAW, N_B, n_sub, n_th)))
        g.create_dataset("b_nodes", data=np.linspace(0.0, 4.0, N_B))
        g.create_dataset("z_sub", data=z_sub)
        g.create_dataset("fit_pixels", data=fit_pixels.astype(np.int32))
        g.create_dataset("completeness", data=f_p)
        g.create_dataset("sensitivity_S", data=np.zeros((m_sph * m_z, n_th)))
        g.attrs["sensitivity_labels"] = json.dumps(
            ["M0hat", "sigma_M", "delta", "Om0", "b_gal"])
        # The builder floors f_p at 1e-3 BEFORE forming F_F; the guard has to
        # apply the same floor, which is what test_f_p_floor_matches_builder
        # pins.
        g.attrs["P_F"] = float(n_fit) if p_f is None else float(p_f)
        g.attrs["F_F"] = (float(np.maximum(f_p, 1e-3).sum()) if f_f is None
                          else float(f_f))
        g.attrs["theta_ref"] = json.dumps(
            theta_ref if theta_ref is not None
            else dict(M0hat=-20.3, sigma_M=0.6, delta=0.0, Om0=0.315))
        g.attrs["basis_meta"] = json.dumps(dict(
            jitter_mode="factored-v1", j_sph=1e-6, j_z=1e-6, amp=1.0,
            ls_sph=float(ls_sph), ls_z=float(ls_z), M_sph=int(m_sph),
            M_z=int(m_z), z_node_hi=float(z_depth)))
        g.attrs["nside"] = int(nside)
        g.attrs["sha256"] = "0" * 64
        g.attrs["format_version"] = "darksirens-latent-field-1.0"
        # The remaining guard-1 fingerprint ingredients (PLAN 4.4 successor 1:
        # completeness-map hash, shell-response W hash, z_edges, counts hash,
        # theta_ref, b_gal).  ``load_latent_plan`` never reads them -- the seam
        # is theta-free and count-free online -- but the fingerprint guard does,
        # and an artifact that cannot be fingerprinted is refused, so writing
        # them is what makes this fixture the "structurally complete (but tiny)
        # PR-4 anchor artifact" its docstring claims.
        g.create_dataset("counts", data=np.full((3, n_fit), 7.0))
        g.create_dataset("z_count_edges", data=np.linspace(0.0, z_depth, 4))
        g.create_dataset("shell_response", data=np.eye(3, n_sub))
        g.attrs["b_gal"] = 1.4
    return str(path)


def _opts(**kw):
    base = dict(lss_field_mode="latent", lss_field_artifact=None,
                lss_completion=None, use_LSS=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _catalogs(*, f_p_occ=F_P, occ_pixels=FIT_PIXELS, row_pixels=ROW_PIXELS,
              f_p_rows=None, shared_views=True):
    """A stand-in for the ``prepare_catalog_views`` result.

    The factory reads exactly five things off it in latent mode:
    ``unique_pixels_pe`` / ``_sel`` (the row -> global pixel maps),
    ``field_occupied_pixels`` / ``field_f_p_occ`` (the field rows and their
    selection fractions) and ``f_p_rows_pe`` (the catalog-row fallback).
    """
    up = None if row_pixels is None else jnp.asarray(row_pixels, dtype=jnp.int32)
    return SimpleNamespace(
        unique_pixels_pe=up,
        unique_pixels_sel=up if shared_views else (
            None if up is None else jnp.asarray(np.asarray(row_pixels),
                                                dtype=jnp.int32)),
        field_occupied_pixels=(None if occ_pixels is None
                               else jnp.asarray(occ_pixels, dtype=jnp.int32)),
        field_f_p_occ=(None if f_p_occ is None
                       else jnp.asarray(np.asarray(f_p_occ, dtype=np.float32))),
        f_p_rows_pe=(None if f_p_rows is None
                     else jnp.asarray(np.asarray(f_p_rows, dtype=np.float32))),
    )


def _resolve(opts, catalogs=None, n_rows=None):
    n = int(ROW_PIXELS.size) if n_rows is None else int(n_rows)
    return _resolve_latent_leaves(
        opts, catalogs if catalogs is not None else _catalogs(),
        Z_DEPTH, NSIDE, n, n)


# ------------------------------------------------------- table mode is inert

def test_table_mode_installs_nothing():
    """The default flag position adds NOTHING to either EMCatalog.

    The two constructions splat these dicts, so "empty" is the property that
    makes table mode textually and numerically the shipped path (PLAN §3.6:
    the latent branch must be a static structure branch, never a value one).
    """
    mode, pe, sel = _resolve(_opts(lss_field_mode="table"))
    assert mode == "table"
    assert pe == {} and sel == {}


def test_table_mode_defaults_when_opts_has_no_flag():
    mode, pe, sel = _resolve(SimpleNamespace())
    assert (mode, pe, sel) == ("table", {}, {})


def test_artifact_under_table_mode_is_refused(tmp_path):
    """A supplied artifact that is silently ignored is the failure this
    codebase refuses everywhere else (cf. the Q-table provenance checks)."""
    art = _write_artifact(tmp_path / "anchor.h5")
    with pytest.raises(ValueError, match="never reads it"):
        _resolve(_opts(lss_field_mode="table", lss_field_artifact=art))


def test_latent_mode_without_artifact_is_refused():
    with pytest.raises(ValueError, match="requires --lss_field_artifact"):
        _resolve(_opts())


def test_unknown_mode_string_is_refused():
    with pytest.raises(ValueError, match="must be 'table' or 'latent'"):
        _resolve_lss_field_mode(_opts(lss_field_mode="latent-ish"))


# ------------------------------------------------------------ the happy path

def test_latent_installs_leaves_on_both_catalogs(tmp_path):
    """Shapes, dtypes, the pad row, and the PE/selection ALIASING.

    The aliasing is not cosmetic: ``can_share_redshift_prior_state`` compares
    every EMCatalog field by ``is``, so two separately barriered copies of the
    same block would collapse the PE/selection prior-state sharing to False and
    double the per-member state precomputation.
    """
    art = _write_artifact(tmp_path / "anchor.h5")
    mode, pe, sel = _resolve(_opts(lss_field_artifact=art))
    assert mode == "latent"

    n_fit = int(FIT_PIXELS.size)
    n_rows = int(ROW_PIXELS.size)
    assert pe["latent_row_fac"].shape == (M_DRAW, n_fit + 1, M_Z)
    assert pe["latent_row_fac"].dtype == jnp.float32
    assert pe["latent_phi_z"].shape == (N_GRID, M_Z)
    assert pe["latent_A"].shape == (M_DRAW, N_B, N_GRID)
    assert pe["latent_B"].shape == (M_DRAW, N_B, N_GRID)
    assert pe["latent_b_nodes"].shape == (N_B,)
    assert pe["latent_P_F"] == float(n_fit)
    assert pe["latent_F_F"] == pytest.approx(float(F_P.sum()))

    # Catalog rows 0,2,4,6 are in F (footprint rows 0,2,4,6); 8,10,12,14 are
    # not and land on the ZERO pad row n_fit, masked off by latent_on_fp.
    row_map = np.asarray(pe["latent_row_map"])
    on_fp = np.asarray(pe["latent_on_fp"])
    assert row_map.shape == (n_rows,) and row_map.dtype == np.int32
    assert row_map.tolist() == [0, 2, 4, 6, n_fit, n_fit, n_fit, n_fit]
    assert on_fp.tolist() == [True] * 4 + [False] * 4
    assert np.all(np.asarray(pe["latent_row_fac"])[:, n_fit, :] == 0.0)

    # The field rows are the artifact's own footprint here, so every one of
    # them is inside F.
    assert np.asarray(pe["latent_field_row_map"]).tolist() == list(range(n_fit))
    assert bool(np.all(np.asarray(pe["latent_field_on_fp"])))

    for key in ("latent_row_fac", "latent_phi_z", "latent_A", "latent_B",
                "latent_b_nodes", "latent_field_row_map", "latent_field_on_fp"):
        assert pe[key] is sel[key], f"{key} must be ONE aliased object"
    assert pe["latent_row_map"] is sel["latent_row_map"]  # shared union views

    assert set(pe) == set(sel) == {
        "latent_row_fac", "latent_phi_z", "latent_row_map", "latent_on_fp",
        "latent_field_row_map", "latent_field_on_fp", "latent_A", "latent_B",
        "latent_b_nodes", "latent_P_F", "latent_F_F", "latent_support"}
    # PR-8's support leaf is INSTALLED only by an amp(z) anchor.  On this one
    # -- and on every anchor built before PR-8 -- it is None, which is the
    # static pytree-STRUCTURE branch that keeps the consumers on the
    # pre-PR-8 code path (they recompute ``zgrid <= z_depth`` themselves).
    assert pe["latent_support"] is None and sel["latent_support"] is None


def test_phi_z_is_zero_above_the_depth(tmp_path):
    """The depth relaxation is in the basis, not a downstream mask: above
    ``z_depth`` phi_z is BIT-zero, so logQ is bit-zero there."""
    art = _write_artifact(tmp_path / "anchor.h5")
    _, pe, _ = _resolve(_opts(lss_field_artifact=art))
    above = _Z > Z_DEPTH
    assert np.all(np.asarray(pe["latent_phi_z"])[above] == 0.0)
    assert np.any(np.asarray(pe["latent_phi_z"])[~above] != 0.0)


def test_row_map_falls_back_to_row_index_for_full_sky_catalogs(tmp_path):
    """``unique_pixels is None`` is the legacy full-sky catalog: row k IS
    pixel k, so the first n_fit rows are the footprint."""
    art = _write_artifact(tmp_path / "anchor.h5")
    npix = 12 * NSIDE ** 2
    cats = _catalogs(row_pixels=None)
    _, pe, _ = _resolve(_opts(lss_field_artifact=art), cats, n_rows=npix)
    row_map = np.asarray(pe["latent_row_map"])
    assert row_map.shape == (npix,)
    assert row_map[:8].tolist() == list(range(8))
    assert np.all(row_map[8:] == 8)


def test_field_leaves_are_none_without_field_rows(tmp_path):
    """Conditional sky weighting never builds the survey-global normalizer, so
    there are no field rows to map -- and the field leaves are correctly None."""
    art = _write_artifact(tmp_path / "anchor.h5")
    cats = _catalogs(occ_pixels=None, f_p_occ=None, f_p_rows=None)
    # The catalog-row f_p is then the only completeness source; give it the
    # footprint values on the on-footprint rows.
    f_p_rows = np.zeros(ROW_PIXELS.size)
    f_p_rows[:4] = F_P[[0, 2, 4, 6]]
    cats.f_p_rows_pe = jnp.asarray(f_p_rows.astype(np.float32))
    # ... and an artifact whose footprint is exactly those four pixels.
    art4 = _write_artifact(tmp_path / "anchor4.h5",
                           fit_pixels=ROW_PIXELS[:4], f_p=F_P[[0, 2, 4, 6]])
    _, pe, _ = _resolve(_opts(lss_field_artifact=art4), cats)
    assert pe["latent_field_row_map"] is None
    assert pe["latent_field_on_fp"] is None
    assert np.asarray(pe["latent_row_map"]).tolist() == [0, 1, 2, 3] + [4] * 4


# ------------------------------------------------------------------ guard 1

def test_f_p_mismatch_is_refused(tmp_path):
    """A depth map that is not the anchor's breaks the eq. (4) identity."""
    art = _write_artifact(tmp_path / "anchor.h5")
    bad = F_P.copy()
    bad[3] += 0.1                      # one pixel, 5e-3 of F_F: caught
    with pytest.raises(ValueError, match="disagrees with the run's sum of f_p"):
        _resolve(_opts(lss_field_artifact=art), _catalogs(f_p_occ=bad))


def test_f_p_agreement_survives_the_float32_round_trip(tmp_path):
    """MEASURED headroom: the run carries f_p as float32 while the builder
    summed float64; over the production 30,470 rows that moves F_F by
    <= 4.5e-10 relative, ~2200x inside the 1e-6 tolerance.  Here the fixture's
    f_p is float64 in the artifact and float32 on the view, and it passes --
    while test_f_p_mismatch_is_refused shows the same guard catching a single
    pixel that moved by 0.1."""
    art = _write_artifact(tmp_path / "anchor.h5")
    mode, pe, _ = _resolve(_opts(lss_field_artifact=art))
    assert mode == "latent"
    f32 = np.asarray(F_P, dtype=np.float32).astype(np.float64)
    rel = abs(f32.sum() - F_P.sum()) / F_P.sum()
    assert rel < 1e-6
    assert pe["latent_F_F"] == pytest.approx(float(F_P.sum()), rel=1e-6)


def test_f_p_floor_matches_the_builder(tmp_path):
    """The builder floors f_p at 1e-3; the guard must too, or every deep-mask
    pixel would read as a mismatch."""
    tiny = F_P.copy()
    tiny[:3] = [0.0, 1e-9, 5e-4]       # below the floor: builder sees 1e-3
    art = _write_artifact(tmp_path / "anchor.h5", f_p=tiny)
    mode, _, _ = _resolve(_opts(lss_field_artifact=art),
                          _catalogs(f_p_occ=tiny))
    assert mode == "latent"


def test_footprint_pixel_absent_from_the_run_is_refused(tmp_path):
    """The run's sky is not the anchor's sky: those rows would silently route
    to the off-footprint pad while the moments still count them."""
    art = _write_artifact(tmp_path / "anchor.h5")
    cats = _catalogs(occ_pixels=FIT_PIXELS[:-1], f_p_occ=F_P[:-1])
    with pytest.raises(ValueError, match="absent from the run's"):
        _resolve(_opts(lss_field_artifact=art), cats)


def test_missing_per_pixel_completeness_is_refused(tmp_path):
    art = _write_artifact(tmp_path / "anchor.h5")
    cats = _catalogs(f_p_occ=None, occ_pixels=None, f_p_rows=None)
    with pytest.raises(ValueError, match="per-pixel selection fraction"):
        _resolve(_opts(lss_field_artifact=art), cats)


# ------------------------------------------------------------------ guard 2

def test_mpc_correlation_length_is_a_hard_error(tmp_path):
    """ls_z = L/((1+z_ref) dchi/dz) scales like H0 (7x over [20, 140]), so an
    assumed Mpc length is a standard ruler against the galaxy catalog."""
    art = _write_artifact(tmp_path / "anchor.h5")
    with pytest.raises(ValueError, match="standard ruler"):
        _resolve(_opts(lss_field_artifact=art, lss_corr_length_mpc=50.0))


# ------------------------------------------------------------------ guard 3

def test_under_resolved_sphere_is_refused(tmp_path):
    """sqrt(4pi/M_sph) > ls_sph: the low-rank GP collapses to the prior while
    still reporting convergence (Burt et al. 2019).  HARD here, promoted from
    the gp3d builder's WARN."""
    art = _write_artifact(tmp_path / "anchor.h5", m_sph=8)   # 1.25 > 0.8
    with pytest.raises(ValueError, match="under-resolved on the SPHERE"):
        _resolve(_opts(lss_field_artifact=art))


def test_under_resolved_redshift_is_refused(tmp_path):
    """log1p(z_node_hi)/(M_z-1) > ls_z, with ls_sph raised so the sphere half
    passes and this is unambiguously the radial guard."""
    art = _write_artifact(tmp_path / "anchor.h5", ls_z=0.01, ls_sph=0.8,
                          f_f=None)
    with pytest.raises(ValueError, match="under-resolved in REDSHIFT"):
        _resolve(_opts(lss_field_artifact=art))


# ------------------------------------------------------------------ guard 4

def test_anisotropic_field_is_refused(tmp_path):
    """A 4:1 pancake: ls_z pushed far past the isotropic value while staying
    coarse enough to clear guard 3."""
    art = _write_artifact(tmp_path / "anchor.h5", ls_z=0.5)
    with pytest.raises(ValueError, match="anisotropic"):
        _resolve(_opts(lss_field_artifact=art))


def test_isotropy_is_evaluated_H0_free(tmp_path):
    """chi and dchi/dz both carry c/H0, so the aspect ratio depends only on
    Om0 -- the same reason guard 2 refuses an Mpc-valued ls_z.  Verified by
    re-deriving the ratio the guard accepts."""
    from darksirens.utils import cosmology

    z_ref = 0.5 * Z_DEPTH
    om0 = 0.315
    ratios = []
    for h0 in (30.0, 70.0, 140.0):
        chi = float(cosmology.r_of_z(z_ref, h0, om0))
        dchi = cosmology.speed_of_light / (h0 * float(cosmology.E(z_ref, om0)))
        ratios.append((LS_SPH * chi) / (LS_Z * (1 + z_ref) * dchi))
    assert max(ratios) - min(ratios) < 1e-12
    assert abs(np.log(ratios[0])) < np.log(1.5)


# ------------------------------------------------------------------ guard 6

def test_loaded_q_table_with_latent_is_refused(tmp_path):
    art = _write_artifact(tmp_path / "anchor.h5")
    with pytest.raises(ValueError, match="SECOND"):
        _resolve(_opts(lss_field_artifact=art, lss_completion="/tmp/q.h5"))


def test_use_lss_with_latent_is_refused(tmp_path):
    art = _write_artifact(tmp_path / "anchor.h5")
    with pytest.raises(ValueError, match="incompatible with --use_lss"):
        _resolve(_opts(lss_field_artifact=art, use_LSS=True))


# ------------------------------------------------- artifact/run disagreements

def test_nside_mismatch_is_refused(tmp_path):
    """The loader's own guard, reached through the factory: a footprint row map
    built at the wrong nside is meaningless."""
    art = _write_artifact(tmp_path / "anchor.h5", nside=16)
    with pytest.raises(ValueError, match="nside"):
        _resolve(_opts(lss_field_artifact=art))


def test_z_depth_mismatch_is_refused(tmp_path):
    """The artifact's below-depth block must be the run's."""
    art = _write_artifact(tmp_path / "anchor.h5")
    with pytest.raises(ValueError, match="below-depth"):
        _resolve_latent_leaves(
            _opts(lss_field_artifact=art), _catalogs(), 0.5, NSIDE,
            int(ROW_PIXELS.size), int(ROW_PIXELS.size))


def test_no_run_depth_against_a_depth_built_artifact_is_refused(tmp_path):
    """A run with NO ``--survey_z_depth`` may not load a depth-built anchor.

    This is the case ``z_depth is not None`` never reached.  ``below`` is
    all-True, so the artifact's ZERO PAD above ``z_sub`` is consumed as real
    field: ``A - C B`` is ``0`` there, ``rho`` takes its ``1e-300`` floor at
    about ``-693``, and the seam returns ``logQ = +693`` on every footprint row
    above the build's depth -- delivered as ``Q = e^7 = 1097`` by the clip, a
    ~1100x inflation of the missing-host density with no NaN and no crash.
    """
    art = _write_artifact(tmp_path / "anchor.h5")
    with pytest.raises(ValueError, match="built with a depth"):
        _resolve_latent_leaves(
            _opts(lss_field_artifact=art), _catalogs(), None, NSIDE,
            int(ROW_PIXELS.size), int(ROW_PIXELS.size))


def test_no_run_depth_against_a_full_grid_artifact_still_loads(tmp_path):
    """...and the legacy full-grid convention is NOT collateral damage.

    An anchor whose ``z_sub`` IS the whole zgrid has no pad to misread, so a
    depthless run loads it exactly as before.  ``m_z``/``ls_z`` are retuned
    here only to keep the resolution and isotropy guards satisfied out to
    ``z = 5``; the depth check is what the test is about.
    """
    art = _write_artifact(tmp_path / "anchor.h5", z_depth=float(_Z[-1]),
                          m_z=4, ls_z=1.0)
    _, pe, _ = _resolve_latent_leaves(
        _opts(lss_field_artifact=art), _catalogs(), None, NSIDE,
        int(ROW_PIXELS.size), int(ROW_PIXELS.size))
    # No zero pad anywhere: the last grid node carries real basis rows.
    assert np.any(np.asarray(pe["latent_phi_z"])[-1] != 0.0)
