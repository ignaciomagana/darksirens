"""Field-convention sky weighting combined with a Q_LSS completion table.

Two independent breakages of documented flag combinations, both under the
CLI-DEFAULT `--catalog_sky_weighting field`:

1. `make_likelihood`'s single-catalog path built its PE/selection EMCatalogs
   with an explicit field_* kwarg list that omitted field_lss_q_members /
   field_lss_q_empty_sum_members, even though prepare_catalog_views computes
   them. Every K=1 `--lss_marginalize` run therefore aborted in
   prepare_redshift_prior_state before the first likelihood evaluation.

2. `attach_lss_inputs` computed a real per-pixel delta_g whenever --use_lss was
   set, without checking whether a deterministic Q table was also loaded. Q
   REPLACES the local-overdensity factor, and field_global_log_Z enforces that
   as a hard invariant, so `--use_lss --lss_completion Q.h5` died during the jit
   trace with a message about an internal field-normalizer invariant.
"""
import re
import inspect
from types import SimpleNamespace

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.core.types import CosmoParams, SurveyParams, EMCatalog
from darksirens.redshift.grid import zgrid
from darksirens.redshift.prior import prepare_redshift_prior_state

NPIX, NG, NMEM = 12, 4, 3
NZ = len(np.asarray(zgrid))


def _catalog(with_member_rows: bool):
    rng = np.random.default_rng(0)
    zg = np.zeros((NPIX, NG)); wg = np.zeros((NPIX, NG))
    ng = np.zeros(NPIX, dtype=np.int32)
    for p in range(0, NPIX, 2):
        n = 3
        zg[p, :n] = rng.uniform(0.05, 0.4, n); wg[p, :n] = 1.0; ng[p] = n
    logq = np.zeros((NPIX, NZ))
    logq_m = rng.normal(scale=0.01, size=(NMEM, NPIX, NZ))
    occ = np.arange(NPIX)

    fq = fq_empty = None
    if with_member_rows:
        from darksirens.redshift.completion import build_field_lss_q_member_inputs
        fq, fq_empty = build_field_lss_q_member_inputs(
            jnp.asarray(logq_m), jnp.asarray(occ), NPIX
        )
    return EMCatalog(
        apix=4 * np.pi / NPIX,
        zgals=jnp.asarray(zg), dzgals=jnp.full((NPIX, NG), 0.02),
        wgals=jnp.asarray(wg), ngals=jnp.asarray(ng),
        delta_g_pix_z=jnp.zeros((1, NZ)),
        dN_obs_kde=jnp.zeros((NPIX, NZ)), pixel_to_cache_idx=None,
        lss_completion_logq=jnp.asarray(logq),
        lss_completion_logq_members=jnp.asarray(logq_m),
        field_dN_obs_s=jnp.zeros((NPIX, NZ)), field_n_empty=jnp.asarray(0.0),
        field_N_obs_total=jnp.asarray(float(ng.sum())),
        field_occupied_pixels=jnp.asarray(occ, dtype=jnp.int32),
        field_lss_q=jnp.exp(jnp.asarray(logq)),
        field_lss_q_empty_sum=jnp.zeros(NZ),
        field_lss_q_members=fq, field_lss_q_empty_sum_members=fq_empty,
    )


_COSMO = CosmoParams(H0=67.74, Om0=0.3075)
_SURVEY = SurveyParams(n0=1e-2, z50=1.0, w=0.5, delta=0.0, b_miss=1.0,
                       alpha_miss=0.0, sigma_kde=0.0)


def test_field_mode_with_q_ensemble_builds():
    """K=1 + field weighting + Q ensemble: the combination must work."""
    state = prepare_redshift_prior_state(
        "dark_sirens", _COSMO, _SURVEY, _catalog(True),
        catalog_sky_weighting="field",
    )
    assert state is not None


def test_field_mode_without_member_rows_is_the_documented_failure():
    """Guards the mechanism: dropping the member rows is what used to abort."""
    with pytest.raises(ValueError, match="per-member survey-global Q rows"):
        prepare_redshift_prior_state(
            "dark_sirens", _COSMO, _SURVEY, _catalog(False),
            catalog_sky_weighting="field",
        )


def test_every_field_mode_emcatalog_threads_the_member_rows():
    """Structural guard: no EMCatalog may set field_lss_q without the members.

    The K=1 path builds its EMCatalogs from an explicit kwarg list rather than
    by forwarding a view object, so a newly added field_* row is easy to miss
    on exactly one of the two (PE / selection) constructions.
    """
    import darksirens.likelihood.factory as factory

    src = inspect.getsource(factory)
    blocks = re.findall(r"EMCatalog\((?:[^()]|\([^()]*\))*\)", src, re.S)
    with_q = [b for b in blocks if "field_lss_q=" in b]
    assert with_q, "no EMCatalog sets field_lss_q -- test needs updating"
    missing = [b for b in with_q if "field_lss_q_members" not in b]
    assert not missing, (
        f"{len(missing)} of {len(with_q)} field-mode EMCatalog constructions "
        "omit field_lss_q_members / field_lss_q_empty_sum_members"
    )


# ---------------------------------------------------------------------------
# --use_lss must not build delta_g alongside a Q table
# ---------------------------------------------------------------------------

def test_use_lss_with_q_table_does_not_build_delta_g(monkeypatch):
    """Q replaces the overdensity factor, so delta_g must stay the dummy."""
    from darksirens.inference import loaders

    monkeypatch.setattr(
        loaders, "maybe_load_lss_completion",
        lambda opts, *, zgrid: {
            "lss_completion_logq": np.zeros((NPIX, NZ)),
            "lss_completion_logq_members": None,
            "lss_completion_fiducials": None,
            "lss_completion_indexing": 1,
            "lss_completion_provenance": None,
        },
    )
    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("compute_lss_overdensity must not run when Q is active")

    monkeypatch.setattr(loaders, "compute_lss_overdensity", _boom)

    opts = SimpleNamespace(universe_model="dark_sirens", use_LSS=True,
                           lss_completion="q.h5", survey_path="cat.h5",
                           lss_marginalize=False)
    data = {"nside": 1, "zgals": np.zeros((NPIX, NG)),
            "wgals": np.zeros((NPIX, NG)), "ngals_catalog": np.zeros(NPIX)}
    out = loaders.attach_lss_inputs(opts, data)

    assert called["n"] == 0
    # The dummy shape is what the "no overdensity" path produces.
    assert tuple(np.asarray(out["delta_g_pix_z"]).shape) == (1, NZ)
    assert np.all(np.asarray(out["delta_g_pix_z"]) == 0.0)


def test_use_lss_without_q_table_still_builds_delta_g(monkeypatch):
    """The plain --use_lss path is unchanged."""
    from darksirens.inference import loaders

    monkeypatch.setattr(
        loaders, "maybe_load_lss_completion",
        lambda opts, *, zgrid: {
            "lss_completion_logq": None, "lss_completion_logq_members": None,
            "lss_completion_fiducials": None, "lss_completion_indexing": 0,
            "lss_completion_provenance": None,
        },
    )
    sentinel = jnp.ones((NPIX, NZ))
    monkeypatch.setattr(loaders, "compute_lss_overdensity", lambda *a, **k: sentinel)

    opts = SimpleNamespace(universe_model="dark_sirens", use_LSS=True,
                           lss_completion=None, survey_path="cat.h5",
                           lss_marginalize=False)
    data = {"nside": 1, "zgals": np.zeros((NPIX, NG)),
            "wgals": np.zeros((NPIX, NG)), "ngals_catalog": np.zeros(NPIX)}
    out = loaders.attach_lss_inputs(opts, data)
    assert tuple(np.asarray(out["delta_g_pix_z"]).shape) == (NPIX, NZ)


# ---------------------------------------------------------------------------
# K>=2 field mixtures need one common pixelisation
# ---------------------------------------------------------------------------

def _opts(weighting, n_catalogs=2):
    return SimpleNamespace(n_catalogs=n_catalogs, catalog_sky_weighting=weighting,
                           mark_names_by_catalog=None, resolved_survey_z_depths=None)


def _bundles(*nsides):
    return {"catalogs": [{"nside": n} for n in nsides]}


def test_field_mixture_rejects_mixed_nside():
    """Per-pixel MASSES cannot be mixed across different pixel areas: the
    effective mixture weight becomes w_k * Omega_k and f_cat absorbs the ratio."""
    from darksirens.inference.validation import validate_multitracer_run

    with pytest.raises(ValueError, match="common HEALPix nside|share one HEALPix nside"):
        validate_multitracer_run(_opts("field"), _bundles(64, 256))


def test_field_mixture_accepts_common_nside():
    from darksirens.inference.validation import validate_multitracer_run

    validate_multitracer_run(_opts("field"), _bundles(64, 64))


def test_unset_weighting_defaults_to_field_and_is_guarded():
    """Unset auto-resolves to field, so it must be guarded too."""
    from darksirens.inference.validation import validate_multitracer_run

    with pytest.raises(ValueError, match="nside"):
        validate_multitracer_run(_opts(None), _bundles(64, 128))


def test_conditional_weighting_is_unaffected():
    """The legacy per-pixel-normalised estimand integrates each pixel to unit
    mass, so pixel area cancels and mixed nside is not a bias there."""
    from darksirens.inference.validation import validate_multitracer_run

    validate_multitracer_run(_opts("conditional"), _bundles(64, 256))


def test_single_catalog_is_unaffected():
    from darksirens.inference.validation import validate_multitracer_run

    validate_multitracer_run(_opts("field", n_catalogs=1), _bundles(64))


# ---------------------------------------------------------------------------
# The LINEAR Q table is not plumbed to the numerator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key", ["lss_completion_q", "lss_completion_q_members"])
def test_linear_q_table_is_rejected_by_the_view_builder(key):
    """prepare_catalog_views fed the linear table would build a Q-modulated
    FIELD normalizer while the numerator (which reads only the log forms) fell
    back to the legacy overdensity branch -- a silent budget mismatch.  It must
    be refused with the log-table hint instead."""
    from darksirens.likelihood.catalog_views import prepare_catalog_views

    data = {key: np.ones((NPIX, NZ))}
    opts = SimpleNamespace(catalog_sky_weighting="field")
    with pytest.raises(ValueError, match=key):
        prepare_catalog_views(opts, data, "dark_sirens", None)

# ===========================================================================
# Numerator vs global normalizer: ONE Q clip (review F-067 / F-080)
# ===========================================================================

def test_field_q_rows_carry_the_numerators_logq_clip():
    """The field normalizer's Q rows must be clipped exactly like the numerator's.

    ``_to_q`` / ``_member_q_eff_from_logq`` clip |logQ| <= _LOGQ_CLIP before
    exponentiating; the normalizer rows used a bare exp, and the shipped tables
    are NOT bounded by the builder's clip (it is applied BEFORE the per-z
    mean-one renorm, which then shifts each bin by its log-monopole).  On a
    railed cell the normalizer therefore integrated up to e^7 ~ 1.1e3 times the
    numerator's missing budget for that pixel.
    """
    from darksirens.redshift.completion import (
        _LOGQ_CLIP,
        _to_q,
        build_field_lss_q_inputs,
        build_field_lss_q_member_inputs,
    )

    rng = np.random.default_rng(3)
    logq = rng.uniform(-14.0, 14.0, size=(NPIX, NZ))     # rails both ways
    occ = np.arange(0, NPIX, 2)
    q_occ, q_empty = build_field_lss_q_inputs(jnp.asarray(logq), occ, NPIX)
    q_num = np.asarray(_to_q(jnp.asarray(logq), None))

    assert float(np.max(np.abs(logq))) > _LOGQ_CLIP      # the fixture rails
    np.testing.assert_allclose(np.asarray(q_occ, dtype=float),
                               q_num[occ].astype(np.float32), rtol=1e-6)
    empty = np.setdiff1d(np.arange(NPIX), occ)
    np.testing.assert_allclose(np.asarray(q_empty),
                               q_num[empty].sum(axis=0), rtol=1e-12)

    logq_m = rng.uniform(-14.0, 14.0, size=(NMEM, NPIX, NZ))
    qm_occ, qm_empty = build_field_lss_q_member_inputs(
        jnp.asarray(logq_m), occ, NPIX)
    q_num_m = np.exp(np.clip(logq_m, -_LOGQ_CLIP, _LOGQ_CLIP))
    np.testing.assert_allclose(np.asarray(qm_occ, dtype=float),
                               q_num_m[:, occ, :].astype(np.float32), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(qm_empty),
                               q_num_m[:, empty, :].sum(axis=1), rtol=1e-12)
