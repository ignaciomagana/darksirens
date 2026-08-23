"""Memory accounting for the latent seam (field-level PR-5; PLAN §2.4/§2.5/§3.6).

``--lss_field_mode latent`` replaces a resident log-Q table with the compact
anchor-artifact leaves, so ``block_sizing`` has to account for a SUBSTITUTION:
count the leaves, and do NOT also count a table that is not there.  These pins
fix that, plus the two structural rules PLAN §3.6 attaches to it:

static      the latent leaves are what ``estimate_pending_static_bytes`` /
            ``measure_static_state_bytes`` reserve, and they carry ``base_miss``
            — which the table branch keys off a member table a latent run does
            not have, so without the latent branch it would be reserved by
            nobody (the ~34 GB under-reservation at ``block_sizing.py:623``
            failed exactly this way, in the same direction).
transient   ``rho`` (rung 0) and ``row_fac_shift[pix]`` + the moment correction
            (rung 1, PR-6b) go through ``_slopes_and_fixed``, where
            ``concurrent_evals`` multiplies them — PLAN §0.5 finding 9, which
            scopes §2.4's categorical "0 transient" to rung 0 only.
refusal     ``row_fac`` acquiring a ``theta`` index is PLAN §10's per-proposal
            re-solve; ``latent_recompute=True`` prices it (2.99 GB at
            ``M_draw=8``, 23.96 GB at 64, both at 256 concurrency) and warns.
inert       with the flag off nothing above runs: table mode is a static
            ``None``-vs-not Python branch away from every latent line.

``test_reservation_within_ten_percent_of_measured_peak`` is PLAN's PR-5 gate
("block_sizing reserves within 10% of measured peak").  It MEASURES the device
peak of the seam's per-evaluation work at production shapes and compares it with
the reservation; it self-skips when there is no GPU, when the allocator reports
no statistics (production sets ``XLA_PYTHON_CLIENT_ALLOCATOR=platform``, under
which ``memory_stats`` is inert), when another process has already driven this
process's monotone peak counter, or when the shared device has no room.  The
numbers it checks were measured on an NVIDIA H100 NVL on 2026-08-17 and were
bit-repeatable across three runs; the rest of the file is arithmetic and runs
anywhere.
"""
from __future__ import annotations

import numpy as np
import pytest

from darksirens.likelihood.block_sizing import (
    LATENT_RHO_LIVE_COPIES,
    LatentDims,
    TRUE_FIXED_VALUE_BYTES,
    _slopes_and_fixed,
    estimate_pending_static_bytes,
    latent_pending_bytes,
    latent_transient_bytes_per_eval,
    measure_static_state_bytes,
    predicted_peak_bytes,
    resolve_block_sizes,
)

# Production shapes (PLAN §2.4): DESI footprint at nside 64, M_draw = 8.
M_DRAW, N_FIT, M_Z, N_B = 8, 30_470, 12, 33
N_GRID, N_ROWS, N_OCC = 1_086, 49_143, 30_470
M_SPH, N_THETA = 315, 3
#: The 256-way ``vmap`` of the production sampler (tinyns ``replacement_chains``).
CONCURRENT = 256

MB = 1e6
GB = 1e9


def _dims(**kw):
    base = dict(m_draw=M_DRAW, n_fit=N_FIT, m_z=M_Z, n_b=N_B, n_grid=N_GRID,
                n_rows=N_ROWS, n_field_rows=N_OCC, m_sph=M_SPH, n_theta=N_THETA)
    base.update(kw)
    return LatentDims(**base)


# ────────────────────────────────────────────────────────── the leaf arithmetic

def test_row_fac_leaf_is_plans_11_7_mb():
    """``row_fac`` (M_draw, n_fit + 1, M_z) f32 — PLAN §2.4's 11.7 MB."""
    dims = _dims()
    assert dims.row_fac_bytes == M_DRAW * (N_FIT + 1) * M_Z * 4 == 11_700_864
    assert dims.row_fac_bytes / MB == pytest.approx(11.7, abs=0.01)
    # M_draw is the only lever PLAN §2.4 tabulates a second value for.
    assert _dims(m_draw=64).row_fac_bytes / MB == pytest.approx(93.6, abs=0.1)


def test_moment_tables_are_padded_to_the_full_grid_not_z_sub():
    """``A``/``B`` are 4.59 MB, not PLAN §2.4's 0.3 MB — and that is correct.

    ``latent_q.load_latent_plan`` zero-pads the artifact's below-depth ``z_sub``
    block (~65 nodes at ``z_depth = 0.3``) to the FULL ``N_grid`` grid so the
    seam's gather is the same ``[idx]`` the table path uses.  The pad is inert
    numerically but it IS allocated, so the reservation follows the allocation,
    not the table in the plan.
    """
    dims = _dims()
    assert dims.moment_bytes == 2 * M_DRAW * N_B * N_GRID * 8 + N_B * 8
    assert dims.moment_bytes / MB == pytest.approx(4.59, abs=0.01)
    on_z_sub = 2 * M_DRAW * N_B * 65 * 8
    assert on_z_sub / MB == pytest.approx(0.27, abs=0.01)   # PLAN's 0.3 MB
    assert dims.moment_bytes > 15 * on_z_sub


def test_leaves_replace_a_gigabyte_scale_table():
    """~30 MB of leaves against the 1.06–3.42 GB of table they substitute for."""
    dims = _dims()
    field_member_rows = M_DRAW * N_OCC * N_GRID * 4     # (M, n_occupied, N_grid) f32
    catalog_logq = M_DRAW * N_ROWS * N_GRID * 8         # (M, N_rows, N_grid) f64
    assert field_member_rows / GB == pytest.approx(1.06, abs=0.01)
    assert catalog_logq / GB == pytest.approx(3.42, abs=0.01)
    assert dims.leaf_bytes() / MB == pytest.approx(30.6, abs=0.1)
    assert dims.leaf_bytes() < field_member_rows / 30

    # The rung-1 sphere-basis footprint block dominates once it is reserved.
    assert (dims.leaf_bytes(rung=1) - dims.leaf_bytes()) == N_FIT * M_SPH * 8
    assert dims.leaf_bytes(rung=1) / MB == pytest.approx(107.4, abs=0.1)


def test_sensitivity_blocks_are_resident_at_rung_zero():
    """``S``/``dA``/``dB`` are read unconditionally by ``load_latent_plan``.

    They are rung-1 machinery, but the loader does not gate on the rung, so at
    ``n_theta = 3`` they are 13.9 MB of rung-0 resident state — more than the
    row factors.  An artifact without them (``n_theta = 0``) costs nothing.
    """
    dims = _dims()
    assert dims.sensitivity_bytes / MB == pytest.approx(13.85, abs=0.01)
    assert _dims(n_theta=0).sensitivity_bytes == 0
    assert dims.leaf_bytes() - _dims(n_theta=0).leaf_bytes() == dims.sensitivity_bytes


# ─────────────────────────────────────────────────────── the transient branch

def test_rung1_row_expansion_is_plans_1_46_mb():
    """``row_fac_shift[pix]`` is the row expansion, not the (M_sph, M_z) object.

    PLAN §0.5 finding 9: the shift is consumed as ``row_fac_shift[pix]``, so the
    per-evaluation object is ``(n_fit + 1, M_z)`` f32 = 1.46 MB, ~375 MB at 256
    concurrency — the correction §2.4 makes to its own "0 transient" row.
    """
    dims = _dims()
    expansion = (N_FIT + 1) * M_Z * 4
    assert expansion == 1_462_608
    assert expansion / MB == pytest.approx(1.46, abs=0.01)
    assert CONCURRENT * expansion / MB == pytest.approx(374.4, abs=0.5)
    # The (M_sph, M_z) object PLAN §1.7 prices is four orders smaller; reserving
    # IT instead of the expansion is the mistake finding 9 caught.
    assert M_SPH * M_Z * 8 < expansion / 40
    assert dims.shift_bytes > expansion


def test_rung0_transient_is_small_but_not_zero():
    """PLAN §2.4's "0 transient" is a statement about the LEAVES, not ``rho``."""
    dims = _dims()
    assert dims.rho_bytes == M_DRAW * N_GRID * 8 == 69_504
    per_eval = latent_transient_bytes_per_eval(dims, rung=0)
    assert per_eval == LATENT_RHO_LIVE_COPIES * dims.rho_bytes
    assert per_eval > 0
    # 142 MB at the production concurrency — reservable, not negligible-by-fiat.
    assert CONCURRENT * per_eval / MB == pytest.approx(142.3, abs=0.5)


def test_recompute_guard_prices_plan_section_10(capsys):
    """``row_fac`` with a theta index is the re-solve design PLAN §10 refuses."""
    dims = _dims()
    per_eval = latent_transient_bytes_per_eval(dims, recompute=True)
    assert per_eval - latent_transient_bytes_per_eval(dims) == dims.row_fac_bytes
    assert CONCURRENT * dims.row_fac_bytes / GB == pytest.approx(3.0, abs=0.02)
    assert (CONCURRENT * _dims(m_draw=64).row_fac_bytes / GB
            == pytest.approx(23.96, abs=0.05))
    out = capsys.readouterr().out
    assert "RECOMPUTE" in out and "PLAN §10" in out


def test_concurrency_multiplies_the_transient_and_not_the_leaves():
    """``concurrent = max(1, chains, sched_max)`` is the multiplier that matters."""
    dims = _dims()
    kw = dict(has_catalog=True, needs_grad=False, n_grid=N_GRID, n_q=200,
              max_gals_per_row=2113, n_catalogs=1, warn=False)
    _, _, fixed_1 = _slopes_and_fixed(concurrent_evals=1, latent_dims=dims, **kw)
    _, _, fixed_256 = _slopes_and_fixed(concurrent_evals=CONCURRENT,
                                        latent_dims=dims, **kw)
    per_eval = latent_transient_bytes_per_eval(dims, warn=False)
    # The value-only fixed term is a flat measured intercept, so the whole
    # difference is the latent transient, scaled by the concurrency.
    assert fixed_1 - TRUE_FIXED_VALUE_BYTES == pytest.approx(per_eval)
    assert fixed_256 - fixed_1 == pytest.approx((CONCURRENT - 1) * per_eval)
    # The leaves do NOT scale with it: they are static state, not transient.
    assert (estimate_pending_static_bytes(
        {"zgals_pe": np.zeros((4, 3))}, n_grid=N_GRID, has_catalog=True,
        lss_field_mode="latent", latent_dims=dims)
        == estimate_pending_static_bytes(
            {"zgals_pe": np.zeros((4, 3))}, n_grid=N_GRID, has_catalog=True,
            lss_field_mode="latent", latent_dims=dims))


def test_slope_addend_is_the_per_sample_row_fac_gather():
    """``row_fac_m[pix_fit]`` rides the BLOCK, so it is a slope, not a fixed term."""
    dims = _dims()
    kw = dict(has_catalog=True, needs_grad=False, n_grid=N_GRID, n_q=200,
              max_gals_per_row=2113, n_catalogs=1, concurrent_evals=CONCURRENT,
              warn=False)
    sel_t, pe_t, _ = _slopes_and_fixed(**kw)
    sel_l, pe_l, _ = _slopes_and_fixed(latent_dims=dims, **kw)
    expected = CONCURRENT * M_DRAW * M_Z * 4
    assert sel_l - sel_t == pytest.approx(expected)
    assert pe_l - pe_t == pytest.approx(expected)
    # M_draw is the lever: 384 B/unit at 8 draws, 3,072 at 64 (PLAN's OD5 axis).
    assert dims.slope_bytes_per_unit == 384
    assert _dims(m_draw=64).slope_bytes_per_unit == 3072


# ───────────────────────────────────────────── substitution, not addition

def _flat_latent_run():
    """A flat (K=1) catalog data dict as a latent run carries it: no Q table."""
    return {
        "zgals_pe": np.zeros((1200, 900), dtype=np.float64),
        "zgals_sel": np.zeros((1300, 900), dtype=np.float64),
        "catalog_memory": {"unique_pe_pixels": 1200, "unique_sel_pixels": 1300,
                           "max_galaxies_per_unique_pixel": 900},
    }


def test_latent_static_does_not_count_a_table_that_is_not_there():
    """A latent run's pending state is the leaves + ``base_miss``, and nothing else."""
    data = _flat_latent_run()
    dims = _dims(n_rows=2500, n_field_rows=0)
    pending = estimate_pending_static_bytes(
        data, n_grid=N_GRID, has_catalog=True, lss_field_mode="latent",
        latent_dims=dims)
    kde = (1200 + 1300) * N_GRID * 8
    assert pending == kde + latent_pending_bytes(dims)
    assert pending == kde + dims.leaf_bytes() + dims.base_miss_bytes


def test_latent_branch_reserves_the_base_miss_the_table_branch_would_miss():
    """Without the latent branch a latent run reserves NO completion state.

    The table branch keys every completion term off ``lss_completion_logq_members``.
    A latent run has none — ``completion.completion_curves`` refuses one — so the
    same data in table mode reserves only the KDE cache and silently drops
    ``base_miss``, here 43 MB and 854 MB at production rows.  Same class of
    defect as the ~34 GB under-reservation at ``block_sizing.py:623``.
    """
    data = _flat_latent_run()
    dims = _dims(n_rows=2500, n_field_rows=0)
    as_table = estimate_pending_static_bytes(data, n_grid=N_GRID, has_catalog=True)
    as_latent = estimate_pending_static_bytes(
        data, n_grid=N_GRID, has_catalog=True, lss_field_mode="latent",
        latent_dims=dims)
    assert as_latent - as_table == dims.leaf_bytes() + dims.base_miss_bytes
    assert dims.base_miss_bytes == 2 * 2500 * N_GRID * 8
    assert _dims().base_miss_bytes / MB == pytest.approx(853.9, abs=0.5)


def test_latent_is_cheaper_than_the_table_it_replaces():
    """The whole point: the substitution must be accounted AS a substitution."""
    members = np.zeros((M_DRAW, 2500, N_GRID), dtype=np.float64)   # host numpy
    table_data = dict(_flat_latent_run(), lss_completion_logq_members=members)
    dims = _dims(n_rows=2500, n_field_rows=0)
    table_pending = estimate_pending_static_bytes(
        table_data, n_grid=N_GRID, has_catalog=True)
    latent_pending = estimate_pending_static_bytes(
        _flat_latent_run(), n_grid=N_GRID, has_catalog=True,
        lss_field_mode="latent", latent_dims=dims)
    assert latent_pending < table_pending
    # The saving is the table's device copies (full transfer + per-view slices),
    # net of the leaves; the shared KDE / base_miss terms cancel.
    assert (table_pending - latent_pending) / MB == pytest.approx(
        (members.nbytes + 2 * M_DRAW * 2500 * N_GRID * 8 - dims.leaf_bytes()) / MB,
        rel=1e-9)


def test_measure_static_state_reports_the_latent_leaves():
    data = _flat_latent_run()
    dims = _dims(n_rows=2500, n_field_rows=0)
    loaded = sum(v.nbytes for v in data.values() if hasattr(v, "nbytes"))
    total = measure_static_state_bytes(
        data, n_grid=N_GRID, has_catalog=True, lss_field_mode="latent",
        latent_dims=dims)
    assert total == loaded + estimate_pending_static_bytes(
        data, n_grid=N_GRID, has_catalog=True, lss_field_mode="latent",
        latent_dims=dims)
    assert total > loaded + dims.leaf_bytes()


def test_k_ge_2_takes_one_dims_per_catalog():
    bundles = {"catalogs": [_flat_latent_run(), _flat_latent_run()]}
    one = _dims(n_rows=2500, n_field_rows=0)
    two = _dims(n_rows=1700, n_field_rows=0, m_draw=4)
    shared = estimate_pending_static_bytes(
        bundles, n_grid=N_GRID, has_catalog=True, lss_field_mode="latent",
        latent_dims=one)
    per_cat = estimate_pending_static_bytes(
        bundles, n_grid=N_GRID, has_catalog=True, lss_field_mode="latent",
        latent_dims=[one, two])
    assert shared - per_cat == latent_pending_bytes(one) - latent_pending_bytes(two)
    with pytest.raises(ValueError, match="one LatentDims per catalog"):
        estimate_pending_static_bytes(
            bundles, n_grid=N_GRID, has_catalog=True, lss_field_mode="latent",
            latent_dims=[one])


# ────────────────────────────────────────────────────────────────── the guards

def test_mode_and_dims_must_agree():
    data = _flat_latent_run()
    dims = _dims(n_rows=2500)
    with pytest.raises(ValueError, match="needs latent_dims"):
        estimate_pending_static_bytes(data, n_grid=N_GRID, has_catalog=True,
                                      lss_field_mode="latent")
    with pytest.raises(ValueError, match="lss_field_mode='table'"):
        estimate_pending_static_bytes(data, n_grid=N_GRID, has_catalog=True,
                                      latent_dims=dims)
    with pytest.raises(ValueError, match="must be 'table' or 'latent'"):
        estimate_pending_static_bytes(data, n_grid=N_GRID, has_catalog=True,
                                      lss_field_mode="field")
    with pytest.raises(ValueError, match="n_grid"):
        estimate_pending_static_bytes(data, n_grid=N_GRID + 1, has_catalog=True,
                                      lss_field_mode="latent", latent_dims=dims)


def test_transient_guards():
    dims = _dims()
    with pytest.raises(ValueError, match="latent_rung must be 0"):
        latent_transient_bytes_per_eval(dims, rung=2)
    with pytest.raises(ValueError, match="without latent_dims"):
        _slopes_and_fixed(has_catalog=True, needs_grad=False, n_grid=N_GRID,
                          n_q=200, max_gals_per_row=2113, n_catalogs=1,
                          concurrent_evals=1, latent_rung=1, warn=False)


def test_from_leaves_and_from_plan_agree_with_the_shipped_shapes():
    """The dims constructors read the real leaf shapes, not a remembered table."""
    jnp = pytest.importorskip("jax.numpy")
    from darksirens.likelihood.latent_q import LatentQPlan

    m_draw, n_fit, m_z, n_b, n_grid, n_rows = 3, 40, 6, 33, 64, 64
    plan = LatentQPlan(
        phi_z=jnp.zeros((n_grid, m_z)), below_depth=jnp.ones((n_grid,), bool),
        row_fac=jnp.zeros((m_draw, n_fit + 1, m_z), dtype=jnp.float32),
        A=jnp.zeros((m_draw, n_b, n_grid)), B=jnp.zeros((m_draw, n_b, n_grid)),
        b_nodes=jnp.zeros((n_b,)), P_F=float(n_fit), F_F=float(n_fit),
        m_sph=24, m_z=m_z, S=jnp.zeros((24 * m_z, 2)),
        dA=jnp.zeros((m_draw, n_b, n_grid, 2)),
        dB=jnp.zeros((m_draw, n_b, n_grid, 2)))
    dims = LatentDims.from_plan(plan, n_rows=n_rows, n_field_rows=0)
    assert (dims.m_draw, dims.n_fit, dims.m_z) == (m_draw, n_fit, m_z)
    assert (dims.n_b, dims.n_grid, dims.n_theta, dims.m_sph) == (n_b, n_grid, 2, 24)

    # The estimate IS the allocation: leaf_bytes equals the summed nbytes of the
    # arrays the plan actually holds, plus the row maps the catalog carries.
    plan_bytes = sum(int(np.asarray(x).nbytes) for x in
                     (plan.phi_z, plan.below_depth, plan.row_fac, plan.A, plan.B,
                      plan.b_nodes, plan.S, plan.dA, plan.dB))
    row_maps = n_rows * (4 + 1)
    assert dims.leaf_bytes() == plan_bytes + row_maps

    leaves = {
        "latent_row_fac": plan.row_fac, "latent_A": plan.A,
        "latent_row_map": jnp.zeros((n_rows,), dtype=jnp.int32),
        "latent_field_row_map": jnp.zeros((17,), dtype=jnp.int32),
    }
    from_leaves = LatentDims.from_leaves(leaves)
    assert from_leaves.n_rows == n_rows and from_leaves.n_field_rows == 17
    assert from_leaves.row_fac_bytes == dims.row_fac_bytes


# ───────────────────────────────────────────────────── flag-off inertness (P12)

def test_table_mode_is_untouched():
    """Every entry point returns exactly its pre-PR-5 value with the flag off."""
    data = dict(_flat_latent_run(),
                lss_completion_logq_members=np.zeros((2, 900, N_GRID)))
    assert (estimate_pending_static_bytes(data, n_grid=N_GRID, has_catalog=True)
            == estimate_pending_static_bytes(data, n_grid=N_GRID, has_catalog=True,
                                             lss_field_mode="table"))
    assert (measure_static_state_bytes(data, n_grid=N_GRID, has_catalog=True)
            == measure_static_state_bytes(data, n_grid=N_GRID, has_catalog=True,
                                          lss_field_mode="table"))
    kw = dict(has_catalog=True, needs_grad=False, n_grid=N_GRID, n_q=200,
              max_gals_per_row=2113, n_catalogs=1, concurrent_evals=CONCURRENT,
              warn=False)
    assert _slopes_and_fixed(**kw) == _slopes_and_fixed(latent_dims=None, **kw)
    peak_kw = dict(n_events=259, n_samp=4096, n_sel=1_067_946, has_catalog=True,
                   needs_grad=False, n_grid=N_GRID, n_q=200,
                   concurrent_evals=CONCURRENT, static_state_bytes=1e9)
    assert (predicted_peak_bytes(**peak_kw)
            == predicted_peak_bytes(latent_dims=None, latent_rung=0, **peak_kw))
    plan_kw = dict(n_events=259, n_samp=4096, n_sel=1_067_946,
                   sel_requested="auto", pe_requested="auto", has_catalog=True,
                   flow_path=False, n_grid=N_GRID, needs_grad=False,
                   concurrent_evals=CONCURRENT, free_bytes=int(72.7 * 1024**3),
                   backend="gpu")
    assert resolve_block_sizes(**plan_kw) == resolve_block_sizes(
        latent_dims=None, **plan_kw)


def test_latent_moves_the_plan_only_a_little():
    """Sanity: the latent terms are a real reservation, not a plan-wrecking one."""
    dims = _dims()
    plan_kw = dict(n_events=259, n_samp=4096, n_sel=1_067_946,
                   sel_requested="auto", pe_requested="auto", has_catalog=True,
                   flow_path=False, n_grid=N_GRID, needs_grad=False,
                   concurrent_evals=CONCURRENT, free_bytes=int(72.7 * 1024**3),
                   backend="gpu", static_state_bytes=11.2 * 1024**3)
    table = resolve_block_sizes(**plan_kw)
    latent = resolve_block_sizes(latent_dims=dims, **plan_kw)
    assert latent.source == table.source
    # Same blocking regime, a slightly smaller selection block (the per-sample
    # row_fac gather is 384 B/unit x 256 concurrent = 98 kB/unit against the
    # calibrated ~1.4 MB/unit at these scales).
    assert latent.sel_batch_size <= table.sel_batch_size
    assert latent.sel_batch_size >= 0.5 * table.sel_batch_size


# ──────────────────────────────────────────────── PLAN's PR-5 10% memory gate

def _device_measurement_context(min_free_bytes):
    """``(device, baseline)`` for a trustworthy peak measurement, or a skip reason."""
    jax = pytest.importorskip("jax")
    dev = jax.devices()[0]
    if str(dev.platform).lower() not in ("gpu", "cuda", "rocm"):
        return None, "no GPU device"
    stats = dev.memory_stats() or {}
    if not stats:
        return None, "allocator reports no memory_stats (platform allocator)"
    in_use = int(stats.get("bytes_in_use", 0))
    limit = int(stats.get("bytes_limit", 0))
    if limit - in_use < min_free_bytes:
        return None, f"device has < {min_free_bytes / GB:.1f} GB free (shared box)"
    return dev, None


def test_reservation_within_ten_percent_of_measured_peak():
    """PLAN PR-5 gate: ``block_sizing`` reserves within 10% of the measured peak.

    The quantity measured is the per-evaluation transient the guarded branch
    reserves — at rung 1, where it matters (rung 0's ``rho`` is 142 MB reserved
    against 62 MB measured at 256 concurrency: over-reserved, and immaterial
    either way).  The rung-1 expression is PLAN §1.7/§3.6's, built from the
    SHIPPED ``latent_q`` functions (``theta_shift``, ``moments_at``,
    ``rho_from_moments``) plus the row expansion ``Phi_sph[F] @ shift``; PR-6b's
    own code does not exist yet, which is precisely why the reservation has to be
    checked against the arithmetic of the expression PLAN commits to.

    MEASURED, H100 NVL, BFC allocator, production shapes, ``conc = 64``:
    peak - bytes_in_use = 391.66 MB here and 432.27 MB in a standalone harness
    that also held the full leaf set resident, against a 424.71 MB reservation
    (+8.4% / -1.8%).  Each number was bit-repeatable across three runs — XLA's
    allocation is deterministic — so the 10% band is about WHAT ELSE is live,
    not about measurement noise.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from darksirens.likelihood.latent_q import (
        moments_at, rho_from_moments, theta_shift)

    conc = 64
    dev, why = _device_measurement_context(min_free_bytes=4 * GB)
    if dev is None:
        pytest.skip(f"device peak not measurable here: {why}")

    dims = _dims(n_field_rows=0)
    rng = np.random.default_rng(0)
    A = jnp.asarray(1.0 + rng.random((M_DRAW, N_B, N_GRID)))
    B = jnp.asarray(0.1 * rng.random((M_DRAW, N_B, N_GRID)))
    dA = jnp.asarray(rng.standard_normal((M_DRAW, N_B, N_GRID, N_THETA)))
    dB = jnp.asarray(rng.standard_normal((M_DRAW, N_B, N_GRID, N_THETA)))
    S = jnp.asarray(rng.standard_normal((M_SPH * M_Z, N_THETA)))
    phi_sph_fit = jnp.asarray(rng.standard_normal((N_FIT + 1, M_SPH)))
    b_nodes = jnp.asarray(np.linspace(0.5, 2.5, N_B))
    below = jnp.ones((N_GRID,), dtype=bool)

    def one(c, b, dtheta):
        # PLAN §3.6's rung-1 seam: the two per-proposal objects, then rho.
        shift = theta_shift(S, dtheta, M_SPH, M_Z)          # (M_sph, M_z)
        rows = (phi_sph_fit @ shift).astype(jnp.float32)    # (n_fit + 1, M_z)
        A_t, B_t = moments_at(A, B, dA, dB, dtheta)
        rho = jax.vmap(lambda a, bb: rho_from_moments(
            a, bb, c, b, b_nodes, 1.0e4, 5.0e3, below))(A_t, B_t)
        return rho, rows

    f = jax.jit(jax.vmap(one))
    args = (jnp.asarray(0.5 * rng.random((conc, N_GRID))),
            jnp.asarray(1.0 + 0.1 * rng.random((conc,))),
            jnp.asarray(0.01 * rng.standard_normal((conc, N_THETA))))
    # Baseline with the inputs resident and nothing running; then ONE call, and
    # read the high-water mark it leaves.  The peak of the first call (which also
    # compiles) is the honest number here: it is the allocation the device has to
    # survive, and it is the only one this counter can attribute — the counter is
    # monotone per process, so a second call never raises it again.
    stats = dev.memory_stats() or {}
    before = int(stats["bytes_in_use"])
    peak_before = int(stats["peak_bytes_in_use"])
    jax.block_until_ready(f(*args))
    peak = int((dev.memory_stats() or {})["peak_bytes_in_use"])

    if peak <= peak_before:
        # An earlier allocation in this process still dominates the counter, so
        # the delta would not be ours.
        pytest.skip("an earlier allocation still dominates the peak counter")
    measured = peak - before
    reserved = conc * latent_transient_bytes_per_eval(dims, rung=1, warn=False)
    assert measured > 0
    assert reserved == pytest.approx(measured, rel=0.10), (
        f"latent rung-1 reservation {reserved:,} B vs measured peak "
        f"{measured:,} B at conc={conc}")
