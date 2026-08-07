"""Opt-in aggregate completeness mode (S1, ``SurveyParams.c_mode``).

The legacy per-pixel estimator divides a PER-PIXEL numerator (the observed
KDE) by an ISOTROPIC denominator, so C really estimates
``C_sel(z) * (1 + delta_obs(pix, z))`` clipped to [0, 1]: the observed angular
clustering is absorbed into the completeness and ``(1 - C)`` ANTI-tracks the
true missing density (the closure experiment measured r = -0.43).  The
aggregate mode replaces it with ONE sky-aggregate curve per proposal,

    Cbar(z) = clip( Sum_p dN_obs_s(z|p) / (N_pix_total dN_exp_smooth(z)), 0, 1 ),
    N_pix_total = round(4 pi / apix)   (occupied AND empty pixels),

broadcast to every pixel, so C carries only the radial budget and all angular
structure is the (mean-one) LSS factor's job.

These tests pin:

* the legacy default is bit-identical (``c_mode`` unset == ``c_mode=0`` ==
  the pre-existing per-row formula, node for node);
* on a clustered synthetic survey with CONSTANT true selection C0, the
  aggregate C is pixel-independent, matches the analytic C0, and dN_miss
  correlates POSITIVELY with the injected overdensity -- while the per-pixel
  mode on the same data measurably ANTI-correlates (the documented defect);
* the field-convention global normalizer uses the SAME Cbar curve (occupied
  and empty pixels alike), and the whole thing stays jit/grad-compatible
  (``c_mode`` is a trace-time branch);
* the builder fits empty pixels as N_obs = 0 rows in aggregate mode (voids
  read Q < 1) and stamps ``c_mode`` into the HDF5 attrs;
* the c_mode provenance check at Q load time hard-errors on a mismatch in
  BOTH directions (legacy unstamped tables count as per_pixel).

Fixture style follows tests/test_completion_depth.py: the observed density is
injected directly via ``dN_obs_kde`` in the infinite-galaxy KDE limit, so a
constant-C0 thinning of a z-independent overdensity field gives
``dN_obs_kde[p] = C0 (1 + delta_p) dN_exp_smooth`` exactly.
"""
import warnings
from types import SimpleNamespace

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

# numpy 1/2 compat: the validated env is numpy 1.26 (no np.trapezoid).
_trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
import pytest

from darksirens.core.constants import C_MODES
from darksirens.core.types import (
    AggregateCMode,
    C_MODE_AGGREGATE_STRUCT,
    CosmoParams,
    SurveyParams,
    EMCatalog,
)
from darksirens.redshift import zgrid
from darksirens.redshift.completion import (
    C_MODE_AGGREGATE,
    C_MODE_PER_PIXEL,
    _precompute_grids,
    completion_curves,
    field_global_log_Z,
)

NG = int(zgrid.size)
N_PIX = 8
APIX = 4.0 * np.pi / N_PIX

#: Mean-zero, z-independent injected overdensity (pairs negate exactly so the
#: sky mean vanishes to float precision); |delta| <= 0.2 keeps the per-pixel
#: ratio C0 (1 + delta) unclipped at C0 = 0.8.
DELTA_P = np.array([-0.2, -0.15, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2])
C0 = 0.8


def _cosmo():
    return CosmoParams(H0=67.74, Om0=0.3075, w0=-1.0, wa=0.0)


def _survey(c_mode=None):
    kwargs = {} if c_mode is None else {"c_mode": c_mode}
    return SurveyParams(
        n0=1e-2, z50=1.0, w=0.5, delta=0.0,
        b_miss=1.0, alpha_miss=1.0, **kwargs,
    )


def _clustered_catalog(delta_g="dummy"):
    """Full-sky (unique_pixels=None) catalog with the KDE-injected observed
    density ``C0 (1 + delta_p) dN_exp_smooth`` -- a constant-C0 thinning of a
    clustered galaxy field.  ``delta_g="real"`` attaches the matching
    per-pixel overdensity table (constant in z); ``"dummy"`` keeps lss == 1.
    Returns ``(catalog, dN_exp, dN_exp_smooth)`` (numpy)."""
    if delta_g == "real":
        dg = jnp.asarray(np.tile(DELTA_P[:, None], (1, NG)))
    else:
        dg = jnp.zeros((1, NG))
    cat = EMCatalog(
        apix=APIX,
        zgals=jnp.full((N_PIX, 2), 0.5),
        dzgals=jnp.full((N_PIX, 2), 0.02),
        wgals=jnp.ones((N_PIX, 2)),
        ngals=jnp.ones(N_PIX, dtype=jnp.int32),
        delta_g_pix_z=dg,
        dN_obs_kde=jnp.zeros((N_PIX, NG)),
        pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(_cosmo(), _survey(), cat)
    smooth = np.asarray(grids.dN_exp_smooth)
    kde = C0 * (1.0 + DELTA_P)[:, None] * smooth[None, :]
    return cat._replace(dN_obs_kde=jnp.asarray(kde)), np.asarray(grids.dN_exp), smooth


# ---------------------------------------------------------------------------
# (a) legacy default bit-identical
# ---------------------------------------------------------------------------

def test_c_mode_enum_and_default():
    assert C_MODES == {"per_pixel": 0, "aggregate": 1}
    assert C_MODE_PER_PIXEL == 0 and C_MODE_AGGREGATE == 1
    # the container default IS the legacy mode -- every existing caller that
    # never sets c_mode keeps the bit-identical per-pixel estimator.  It is
    # carried STRUCTURALLY as None (the z_depth pattern) so the trace-time
    # branch survives jit boundaries that trace int leaves.
    assert _survey().c_mode is None


def test_per_pixel_default_bit_identical():
    """``c_mode`` unset (the default) equals ``c_mode=0`` explicitly EXACTLY,
    and both equal the pre-existing per-row formula node for node."""
    cosmo = _cosmo()
    cat, dN_exp, smooth = _clustered_catalog(delta_g="dummy")

    curves_unset = completion_curves(cosmo, _survey(), cat)
    curves_zero = completion_curves(cosmo, _survey(c_mode=C_MODE_PER_PIXEL), cat)
    for field in ("f", "dN_miss", "C_eff", "N_miss"):
        np.testing.assert_array_equal(
            np.asarray(getattr(curves_unset, field)),
            np.asarray(getattr(curves_zero, field)),
        )

    # the legacy definition, computed independently: per-row matched-kernel
    # ratio, dN_miss = (1 - C) dN_exp (lss == 1 on the dummy overdensity) --
    # node-for-node BITWISE (same IEEE ops on the same values)
    safe = np.where(smooth > 0.0, smooth, 1.0)
    C_manual = np.clip(np.asarray(cat.dN_obs_kde) / safe[None, :], 0.0, 1.0)
    np.testing.assert_array_equal(
        np.asarray(curves_unset.dN_miss), (1.0 - C_manual) * dN_exp[None, :]
    )
    # C_eff recomposes through clip(1 - dN_miss / dN_exp), so it matches the
    # ratio to rounding, not bitwise
    np.testing.assert_allclose(
        np.asarray(curves_unset.C_eff), C_manual, rtol=1e-12
    )


# ---------------------------------------------------------------------------
# (b) aggregate mode: pixel-independent Cbar == analytic selection; sign test
# ---------------------------------------------------------------------------

def test_aggregate_cbar_pixel_independent_and_analytic():
    """Constant true selection C0 on a clustered sky: the aggregate numerator
    sums the clustering away (Sum_p (1 + delta_p) == N_pix exactly), so
    Cbar == C0 at every node and every pixel -- where the per-pixel mode reads
    C0 (1 + delta_p), absorbing the clustering into the budget."""
    cosmo = _cosmo()
    cat, dN_exp, _smooth = _clustered_catalog(delta_g="dummy")

    curves = completion_curves(cosmo, _survey(c_mode=C_MODE_AGGREGATE), cat)
    C_eff = np.asarray(curves.C_eff)      # == Cbar under lss == 1
    dN_miss = np.asarray(curves.dN_miss)

    # one curve for the whole sky ...
    for p in range(1, N_PIX):
        np.testing.assert_array_equal(C_eff[p], C_eff[0])
    # ... equal to the analytic selection, and the missing density is the
    # pixel-independent radial budget (1 - C0) dN_exp
    np.testing.assert_allclose(C_eff, C0, rtol=1e-12)
    np.testing.assert_allclose(
        dN_miss, np.tile((1.0 - C0) * dN_exp, (N_PIX, 1)), rtol=1e-12
    )

    # the per-pixel mode on the same data absorbs the clustering into C
    curves_pp = completion_curves(cosmo, _survey(), cat)
    np.testing.assert_allclose(
        np.asarray(curves_pp.C_eff),
        np.tile(C0 * (1.0 + DELTA_P)[:, None], (1, NG)), rtol=1e-12
    )


def test_aggregate_sign_positive_per_pixel_sign_negative():
    """THE sign test.  With the true missing density proportional to
    (1 - C0)(1 + delta_p), a correct budget must correlate POSITIVELY with the
    injected overdensity.  Aggregate: dN_miss = (1 - C0) dN_exp (1 + delta)
    -- exactly linear in delta, r -> +1.  Per-pixel: C absorbs the clustering,
    dN_miss propto (1 - C0(1 + delta))(1 + delta), decreasing in delta at
    C0 = 0.8 -- the structural anti-correlation (documented defect)."""
    cosmo = _cosmo()
    cat, _dN_exp, _smooth = _clustered_catalog(delta_g="real")

    N_agg = np.asarray(
        completion_curves(cosmo, _survey(c_mode=C_MODE_AGGREGATE), cat).N_miss)
    N_pp = np.asarray(completion_curves(cosmo, _survey(), cat).N_miss)

    r_agg = float(np.corrcoef(DELTA_P, N_agg)[0, 1])
    r_pp = float(np.corrcoef(DELTA_P, N_pp)[0, 1])
    assert r_agg > 0.99, f"aggregate dN_miss must track the overdensity (r={r_agg})"
    assert r_pp < -0.9, f"per-pixel dN_miss anti-correlates by construction (r={r_pp})"
    # and the aggregate budget is exactly proportional to (1 + delta)
    np.testing.assert_allclose(
        N_agg / N_agg[0], (1.0 + DELTA_P) / (1.0 + DELTA_P[0]), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# field-convention normalizer: same curve, occupied and empty pixels alike
# ---------------------------------------------------------------------------

def _field_catalog():
    """6 occupied rows + 2 empty pixels, with the field-normalizer inputs the
    production loader would build (f32 observed rows, empty-pixel count)."""
    delta6 = DELTA_P[:6]
    cat = EMCatalog(
        apix=APIX,
        zgals=jnp.full((6, 2), 0.5),
        dzgals=jnp.full((6, 2), 0.02),
        wgals=jnp.ones((6, 2)),
        ngals=jnp.ones(6, dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, NG)),
        dN_obs_kde=jnp.zeros((6, NG)),
        pixel_to_cache_idx=None,
    )
    grids = _precompute_grids(_cosmo(), _survey(), cat)
    smooth = np.asarray(grids.dN_exp_smooth)
    kde6 = C0 * (1.0 + delta6)[:, None] * smooth[None, :]
    return cat._replace(
        dN_obs_kde=jnp.asarray(kde6),
        field_dN_obs_s=jnp.asarray(kde6, dtype=jnp.float32),
        field_n_empty=2,
        field_N_obs_total=12.0,
    ), np.asarray(grids.dN_exp), smooth


def test_aggregate_field_normalizer_shares_the_curve():
    """field_global_log_Z in aggregate mode: every pixel (occupied AND empty)
    carries C == Cbar, so Z = N_obs + N_pix_total * integral (1 - Cbar) dN_exp
    -- the same single curve the per-row numerator uses (one formation site)."""
    cosmo = _cosmo()
    cat, dN_exp, smooth = _field_catalog()
    survey = _survey(c_mode=C_MODE_AGGREGATE)

    logZ = float(field_global_log_Z(cosmo, survey, cat))

    # manual, from the same (preferred) f32 field rows summed in f64
    obs_sum = np.sum(np.asarray(cat.field_dN_obs_s, dtype=np.float64), axis=0)
    safe = np.where(smooth > 0.0, smooth, 1.0)
    Cbar = np.clip(obs_sum / (N_PIX * safe), 0.0, 1.0)
    V = (1.0 - Cbar) * (6 + 2)          # occupied rows + empty pixels, same C
    Z_manual = 12.0 + _trapezoid(dN_exp * V, np.asarray(zgrid))
    np.testing.assert_allclose(logZ, np.log(Z_manual), rtol=1e-10)

    # the per-row curves consume the SAME Cbar (field rows preferred there too)
    curves = completion_curves(cosmo, survey, cat)
    np.testing.assert_allclose(np.asarray(curves.C_eff)[0], Cbar, rtol=1e-12)


def test_aggregate_is_jit_and_grad_compatible():
    """c_mode is a trace-time branch: the aggregate curves jit cleanly and are
    differentiable in n0 (the theta the Cbar denominator depends on)."""
    cosmo = _cosmo()
    cat, _dN_exp, _smooth = _field_catalog()
    survey = _survey(c_mode=C_MODE_AGGREGATE)

    def total_miss(n0):
        return jnp.sum(
            completion_curves(cosmo, survey._replace(n0=n0), cat).N_miss)

    val = jax.jit(total_miss)(1e-2)
    g = jax.grad(total_miss)(1e-2)
    assert np.isfinite(float(val)) and np.isfinite(float(g))

    def logZ(n0):
        return field_global_log_Z(cosmo, survey._replace(n0=n0), cat)

    val = jax.jit(logZ)(1e-2)
    g = jax.grad(logZ)(1e-2)
    assert np.isfinite(float(val)) and np.isfinite(float(g))


def test_c_mode_traced_int_leaf_hard_errors_not_silent_aggregate():
    """A concrete int c_mode on a survey passed THROUGH a jit boundary becomes
    a value-unreadable tracer.  Before the structural sentinel the tracer
    fallback silently evaluated as AGGREGATE -- so ``jit(f)(survey(c_mode=0))``
    computed different physics than the eager ``c_mode=0`` (measured 1.2e-2 in
    N_miss on a pixel-varying KDE).  Now BOTH traced ints are refused loudly
    with the normalisation contract in the message."""
    cosmo = _cosmo()
    cat, _dN_exp, _smooth = _clustered_catalog(delta_g="dummy")

    @jax.jit
    def total_miss(survey):
        return jnp.sum(completion_curves(cosmo, survey, cat).N_miss)

    for bad in (C_MODE_PER_PIXEL, C_MODE_AGGREGATE):
        with pytest.raises(ValueError, match="c_mode"):
            total_miss(_survey(c_mode=bad))


def test_c_mode_structural_conventions_survive_jit_boundary():
    """The decoder's conventions cross the survey-as-argument jit boundary as
    pytree STRUCTURE: the leaf-less sentinel selects aggregate (matching the
    eager int-1 curves), and None stays per-pixel (matching the eager
    default) -- on a clustered sky where the two modes measurably differ."""
    cosmo = _cosmo()
    cat, _dN_exp, _smooth = _clustered_catalog(delta_g="dummy")

    @jax.jit
    def n_miss(survey):
        return completion_curves(cosmo, survey, cat).N_miss

    agg_jit = np.asarray(n_miss(_survey(c_mode=C_MODE_AGGREGATE_STRUCT)))
    agg_eager = np.asarray(
        completion_curves(cosmo, _survey(c_mode=C_MODE_AGGREGATE), cat).N_miss)
    np.testing.assert_allclose(agg_jit, agg_eager, rtol=1e-12)
    # aggregate signature: one budget for the whole (constant-C0) sky
    np.testing.assert_allclose(agg_jit, agg_jit[0], rtol=1e-12)

    pp_jit = np.asarray(n_miss(_survey()))
    pp_eager = np.asarray(completion_curves(cosmo, _survey(), cat).N_miss)
    np.testing.assert_allclose(pp_jit, pp_eager, rtol=1e-12)
    # and the two modes really differ on this clustered fixture
    assert float(np.max(np.abs(pp_jit - agg_jit))) > 1e-3


def test_aggregate_compact_catalog_without_field_rows_refuses():
    """A compact catalog (unique_pixels set) without field_dN_obs_s covers only
    the union pixels, so the aggregate numerator would silently under-count --
    refused loudly at trace time."""
    cat, _dN_exp, _smooth = _clustered_catalog()
    cat = cat._replace(unique_pixels=jnp.arange(N_PIX, dtype=jnp.int32))
    with pytest.raises(ValueError, match="COMPACT catalog"):
        completion_curves(_cosmo(), _survey(c_mode=C_MODE_AGGREGATE), cat)


# ---------------------------------------------------------------------------
# decoder threading: the CLI string lands on SurveyParams as a concrete int
# ---------------------------------------------------------------------------

def test_decoder_threads_c_mode_onto_survey_params():
    from darksirens.gw.populations import get_fixed_population_params
    from darksirens.inference.parameters import build_parameter_decoder

    opts = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        fix_population=True, fix_cosmology=True, fix_survey=True,
        c_mode="aggregate",
    )
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params("powerlaw+peak"),
        fixed_parameter_values={})
    assert decoder.c_mode == C_MODE_AGGREGATE
    _cosmo_out, survey, *_ = decoder.decode(jnp.zeros((0,)))
    # aggregate lands on SurveyParams as the STRUCTURAL leaf-less sentinel
    # (never an int leaf, which would become a value-unreadable tracer at the
    # darksiren_log_likelihood jit boundary)
    assert isinstance(survey.c_mode, AggregateCMode)

    # bare/legacy opts (no c_mode attr) decode to the per-pixel default,
    # normalised to the STRUCTURAL None (survives jit boundaries)
    opts_legacy = SimpleNamespace(
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        fix_population=True, fix_cosmology=True, fix_survey=True,
    )
    decoder = build_parameter_decoder(
        opts_legacy, get_fixed_population_params("powerlaw+peak"),
        fixed_parameter_values={})
    assert decoder.c_mode == C_MODE_PER_PIXEL
    _cosmo_out, survey, *_ = decoder.decode(jnp.zeros((0,)))
    assert survey.c_mode is None


# ---------------------------------------------------------------------------
# builder: aggregate base fits empty pixels; c_mode stamped and roundtripped
# ---------------------------------------------------------------------------

def _write_clustered_survey(path, nside=1, n_occ=4, seed=3):
    """load_survey-schema catalog, nside=1 (12 pixels), first ``n_occ``
    occupied with varying counts (mirrors test_q_budget_renormalization)."""
    import h5py
    import healpy as hp
    npix = hp.nside2npix(int(nside))
    maxg = 12
    zgals = np.full((npix, maxg), 100.0)
    dzgals = np.full((npix, maxg), 0.01)
    wgals = np.zeros((npix, maxg))
    ngals = np.zeros(npix, dtype=np.int32)
    rng = np.random.default_rng(seed)
    for p in range(int(n_occ)):
        n = int(rng.integers(3, maxg + 1))
        zgals[p, :n] = rng.uniform(0.2, 0.8, size=n)
        wgals[p, :n] = 1.0
        ngals[p] = n
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = int(nside)
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("ngals", data=ngals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
    return npix


def test_radial_builder_aggregate_fits_empty_pixels(tmp_path):
    pytest.importorskip("healpy")
    from darksirens.cli.build_lognormal_completion import build_completion

    cat = str(tmp_path / "survey.h5")
    npix = _write_clustered_survey(cat)
    occ = np.arange(4)
    empty = np.arange(4, npix)

    # per-pixel base: empty pixels are never fit -> logQ exactly 0 there
    logq_pp, _m, diag_pp = build_completion(
        cat, mode="radial", n_members=0, maxiter=60)
    assert diag_pp["c_mode"] == "per_pixel"
    assert diag_pp["n_occupied"] == occ.size
    assert np.all(logq_pp[empty] == 0.0)

    # aggregate base: EVERY pixel is an N_obs row (empties at 0 counts against
    # the nonzero base Cbar dN_exp) -- the void information: empty pixels read
    # genuinely under-dense, not homogeneous
    logq_agg, _m, diag_agg = build_completion(
        cat, mode="radial", n_members=0, maxiter=60, c_mode="aggregate")
    assert diag_agg["c_mode"] == "aggregate"
    assert diag_agg["n_occupied"] == npix          # fitted-row count
    assert diag_agg["n_occupied_data"] == occ.size
    assert diag_agg["n_pix_total"] == npix
    assert not np.all(logq_agg[empty] == 0.0)
    assert (np.mean(logq_agg[empty]) < np.mean(logq_agg[occ]))


def test_c_mode_hdf5_attr_roundtrip(tmp_path):
    from darksirens.redshift.lognormal_completion import (
        load_lss_completion_hdf5,
        save_lss_completion_hdf5,
    )

    path = str(tmp_path / "q.h5")
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # fully-stamped file loads quietly
        save_lss_completion_hdf5(
            path, logq_map=np.zeros((N_PIX, NG)), zgrid=np.asarray(zgrid),
            indexing="global", budget_renormalized=True,
            budget_monopole_logq=np.zeros(NG), c_mode="aggregate")
        assert load_lss_completion_hdf5(path)["c_mode"] == "aggregate"

    # unstamped file reads c_mode=None (legacy per-pixel); bad value refused
    path2 = str(tmp_path / "q2.h5")
    save_lss_completion_hdf5(
        path2, logq_map=np.zeros((N_PIX, NG)), zgrid=np.asarray(zgrid),
        indexing="global", budget_renormalized=True,
        budget_monopole_logq=np.zeros(NG))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert load_lss_completion_hdf5(path2)["c_mode"] is None
    with pytest.raises(ValueError, match="c_mode must be"):
        save_lss_completion_hdf5(
            path, logq_map=np.zeros((N_PIX, NG)), c_mode="banana")


# ---------------------------------------------------------------------------
# (c) provenance: table c_mode vs survey c_mode hard-errors, both directions
# ---------------------------------------------------------------------------

def _write_q(tmp_path, name, c_mode):
    from darksirens.redshift.lognormal_completion import save_lss_completion_hdf5

    path = str(tmp_path / name)
    save_lss_completion_hdf5(
        path, logq_map=np.zeros((N_PIX, NG)), zgrid=np.asarray(zgrid),
        indexing="global", budget_renormalized=True,
        budget_monopole_logq=np.zeros(NG), c_mode=c_mode,
        metadata={"fiducial_n0": 1e-2, "fiducial_delta": 0.0},
    )
    return path


def _load_opts(path, c_mode):
    kwargs = {} if c_mode is None else {"c_mode": c_mode}
    return SimpleNamespace(
        lss_completion=path, survey_path=None,
        universe_model="dark_sirens", lss_marginalize=False, **kwargs)


def test_c_mode_provenance_mismatch_hard_errors_both_directions(tmp_path):
    from darksirens.catalogs.lss import maybe_load_lss_completion

    agg = _write_q(tmp_path, "agg.h5", "aggregate")
    pp = _write_q(tmp_path, "pp.h5", "per_pixel")
    legacy = _write_q(tmp_path, "legacy.h5", None)   # no attr

    # aggregate table consumed by a per-pixel survey: the table's Q carries
    # the full clustering; the per-pixel C absorbs it again -> double count
    with pytest.raises(ValueError, match="c_mode mismatch"):
        maybe_load_lss_completion(_load_opts(agg, "per_pixel"), zgrid=zgrid)

    # per-pixel-base table consumed in aggregate mode: the clustering signal
    # is dropped entirely
    with pytest.raises(ValueError, match="c_mode mismatch"):
        maybe_load_lss_completion(_load_opts(pp, "aggregate"), zgrid=zgrid)

    # an unstamped legacy table IS a per-pixel-base table
    with pytest.raises(ValueError, match="legacy"):
        maybe_load_lss_completion(_load_opts(legacy, "aggregate"), zgrid=zgrid)

    # matching modes load; bare/legacy opts (no c_mode attr) mean per_pixel
    out = maybe_load_lss_completion(_load_opts(agg, "aggregate"), zgrid=zgrid)
    assert out["lss_completion_logq"] is not None
    out = maybe_load_lss_completion(_load_opts(pp, None), zgrid=zgrid)
    assert out["lss_completion_logq"] is not None
    out = maybe_load_lss_completion(_load_opts(legacy, None), zgrid=zgrid)
    assert out["lss_completion_logq"] is not None
