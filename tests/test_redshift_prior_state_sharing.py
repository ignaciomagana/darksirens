"""Sharing the prepared redshift-prior state across the PE / selection seams (E3).

``core.can_share_redshift_prior_state`` proves, by object identity on the
EMCatalog fields ``prepare_redshift_prior_state`` consumes, that the PE and
selection states would be identical -- so the state is built ONCE and reused for
both seams (trace-level dedup).  These tests pin:

  * the helper's verdicts (shared-arrays pair -> True; a pair differing in each
    of two representative consumed fields -> False; model guards for
    bright_sirens / WL / spectral / mismatched models -> False);
  * that dedup is exercised for a post-union K=2 bundle (one prepare per catalog,
    not two);
  * that a shared-state likelihood equals the unshared one BIT-FOR-BIT (same
    config with the helper forced off via monkeypatch + jax.clear_caches).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import darksirens.likelihood.core as core
from darksirens.core.types import EMCatalog
from darksirens.likelihood.core import can_share_redshift_prior_state
from darksirens.likelihood.factory import make_likelihood

from test_multitracer_likelihood import _base_opts, _mid_pop, _pop_bits
from test_multitracer_union_compaction import _k2_union_data, _union_bundle
import test_unified_k1_golden as golden


# ---------------------------------------------------------------------------
# Helper verdicts
# ---------------------------------------------------------------------------

def _make_cat(**overrides):
    z = jnp.array([[0.10, 0.20]])
    base = dict(
        apix=1.0,
        zgals=z,
        dzgals=jnp.array([[0.02, 0.02]]),
        wgals=jnp.ones((1, 2)),
        ngals=jnp.array([2], dtype=jnp.int32),
        delta_g_pix_z=jnp.zeros((1, 5)),
        dN_obs_kde=jnp.zeros((1, 5)),
        pixel_to_cache_idx=jnp.zeros((1,), dtype=jnp.int32),
        unique_pixels=jnp.array([0], dtype=jnp.int32),
        sample_to_unique_idx=jnp.zeros((3,), dtype=jnp.int32),
    )
    base.update(overrides)
    return EMCatalog(**base)


def test_share_true_when_all_consumed_fields_are_identical_objects():
    pe = _make_cat()
    # Selection differs ONLY in a field prepare never reads (the per-sample map).
    sel = pe._replace(sample_to_unique_idx=jnp.arange(3, dtype=jnp.int32))
    assert can_share_redshift_prior_state("dark_sirens", "dark_sirens", pe, sel)
    # dark_sirens_complete is likewise dedup-eligible.
    assert can_share_redshift_prior_state(
        "dark_sirens_complete", "dark_sirens_complete", pe, sel
    )


def test_share_false_when_zgals_differs():
    pe = _make_cat()
    # Same VALUES, distinct object -> not `is`-identical -> must not share.
    sel = pe._replace(zgals=jnp.array([[0.10, 0.20]]))
    assert not can_share_redshift_prior_state("dark_sirens", "dark_sirens", pe, sel)


def test_share_false_when_lss_completion_logq_differs():
    q_pe = jnp.zeros((1, 5))
    q_sel = jnp.zeros((1, 5))  # equal values, distinct object
    pe = _make_cat(lss_completion_logq=q_pe)
    sel = pe._replace(lss_completion_logq=q_sel)
    assert not can_share_redshift_prior_state("dark_sirens", "dark_sirens", pe, sel)


def test_share_false_when_stratum_map_differs():
    """``pixel_stratum_map`` IS read by the prepare tree (completion._row_C's
    stratified aggregate branch and _field_missing_curve), so a pair differing
    only in it must not share -- it fell out of the hand-maintained consumed
    list, which made the selection integral silently reuse the PE catalog's
    stratification."""
    smap_pe = jnp.zeros((4,), dtype=jnp.int32)
    smap_sel = jnp.zeros((4,), dtype=jnp.int32)   # equal values, distinct object
    pe = _make_cat(pixel_stratum_map=smap_pe)
    sel = pe._replace(pixel_stratum_map=smap_sel)
    assert not can_share_redshift_prior_state("dark_sirens", "dark_sirens", pe, sel)


def test_compared_fields_cover_every_consumed_leaf_and_nothing_is_unclassified():
    """Every EMCatalog leaf is either compared or explicitly excluded, and the
    documented consumed enumeration matches the compared set exactly.  A field
    added to EMCatalog therefore trips HERE (and defaults to NOT sharing) instead
    of silently falling out of the sharing proof."""
    compared = set(core._PREPARE_STATE_COMPARED_EMCATALOG_FIELDS)
    excluded = set(core._PREPARE_STATE_EXCLUDED_EMCATALOG_FIELDS)
    consumed = set(core._PREPARE_STATE_CONSUMED_EMCATALOG_FIELDS)
    assert compared | excluded == set(EMCatalog._fields)
    assert not (compared & excluded)
    assert consumed == compared, (
        "consumed enumeration out of sync: "
        f"missing={sorted(compared - consumed)}, extra={sorted(consumed - compared)}"
    )


def test_share_false_for_bright_and_wl_and_mismatched_models():
    pe = _make_cat()
    sel = pe._replace(sample_to_unique_idx=jnp.arange(3, dtype=jnp.int32))
    # bright_sirens: PE model differs from its spectral selection model.
    assert not can_share_redshift_prior_state(
        "bright_sirens", "spectral_sirens", pe, sel
    )
    # WL: both seams resolve to spectral_sirens, which is NOT dedup-eligible.
    assert not can_share_redshift_prior_state(
        "spectral_sirens", "spectral_sirens", pe, sel
    )
    # Any mismatched model pair.
    assert not can_share_redshift_prior_state(
        "dark_sirens", "dark_sirens_complete", pe, sel
    )


# ---------------------------------------------------------------------------
# Dedup is exercised, and it is value-invariant
# ---------------------------------------------------------------------------

def _k2_union_likelihood():
    _pl, _pu, _plabels, pop_fid, _sampled, fixed = _pop_bits()
    data = _k2_union_data(_union_bundle)
    opts = _base_opts(n_catalogs=2)
    return (
        make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed),
        jnp.asarray([_mid_pop(), 0.31]),
    )


def test_union_bundle_prepares_state_once_per_catalog(monkeypatch):
    """With sharing ON, a K=2 union bundle calls prepare_redshift_prior_state
    ONCE per catalog (2 total), not once per seam (4)."""
    calls = []
    orig = core.prepare_redshift_prior_state

    def _count(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(core, "prepare_redshift_prior_state", _count)
    jax.clear_caches()
    ll, coord = _k2_union_likelihood()
    val = float(ll(coord))
    assert np.isfinite(val)
    # K=2, one prepare per catalog because PE/selection states are shared.
    assert len(calls) == 2


def _flat_k1_likelihood(cell):
    """Flat (non-bundle) K=1 likelihood for one golden cell + its first coord."""
    opts, data = golden.CELLS[cell]()
    pop_fid, _overrides, fixed = golden._pop_bits()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    _labels, coords = golden._coords_for(opts)
    return ll, jnp.asarray(coords[0])


@pytest.mark.parametrize("cell", ["plain_full", "qdet", "ensemble_marg", "marks"])
def test_flat_k1_union_prepares_state_once(monkeypatch, cell):
    """The flat twin of the bundle test: prepare_catalog_views aliases the PE and
    selection views onto ONE union galaxy table, so a K=1 run must build the
    redshift-prior state ONCE (not once per seam) even when it carries a Q table,
    a Q ensemble or marks -- the factory re-sliced those per view, which broke
    the ``is``-identity the sharing verdict rests on."""
    calls = []
    orig = core.prepare_redshift_prior_state

    def _count(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(core, "prepare_redshift_prior_state", _count)
    jax.clear_caches()
    ll, coord = _flat_k1_likelihood(cell)
    assert np.isfinite(float(ll(coord)))
    assert len(calls) == 1


def test_shared_state_equals_unshared_bit_for_bit(monkeypatch):
    """Forcing the helper OFF (no sharing) must reproduce the shared value
    EXACTLY -- sharing reuses a provably-identical state object."""
    # Unshared: force the helper off and re-trace under a cleared cache.
    monkeypatch.setattr(
        core, "can_share_redshift_prior_state", lambda *a, **k: False
    )
    jax.clear_caches()
    ll_off, coord = _k2_union_likelihood()
    val_off = float(ll_off(coord))

    # Shared: restore the real helper and re-trace.
    monkeypatch.undo()
    jax.clear_caches()
    ll_on, coord2 = _k2_union_likelihood()
    val_on = float(ll_on(coord2))

    assert np.isfinite(val_on)
    assert val_on == val_off  # bit-for-bit
