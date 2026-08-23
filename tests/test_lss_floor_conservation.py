"""Number conservation of the legacy local-overdensity factor (PHY-02).

``compute_lss_overdensity`` centres ``delta_g`` on the FULL sky, so

    Sum_p (1 + b_eff delta_g_p(z)) == N_pix   for every z and every b_eff,

i.e. the missing-galaxy modulation is a pure REDISTRIBUTION of the budget that
``(1 - C)`` and ``n0`` set.  ``max(..., 0)`` breaks that identity wherever
``b_eff delta_g < -1``, which is reachable for ``b_miss > 1`` because
``delta_g >= -1`` by construction and ``b_miss`` samples over [0, 3].  The
review's example: two equal pixels with ``delta_g = (-1, +1)`` at ``b_eff = 2``
floor to ``(0, 3)``, mean 1.5 -- a 50% inflation of the TOTAL missing count,
not a spatial reshuffle, so it moves the catalog-versus-missing odds that
carry the dark-siren H0 information.

``SurveyParams.lss_floor is None`` (the default) renormalizes the floored
factor to full-sky mean one at every redshift; ``LSS_FLOOR_LEGACY_STRUCT``
restores the unrenormalized legacy factor.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.core.types import (
    LSS_FLOOR_LEGACY_STRUCT,
    CosmoParams,
    EMCatalog,
    SurveyParams,
)
from darksirens.redshift import zgrid
from darksirens.redshift.completion import (
    build_pixel_kde_cache,
    completion_curves,
    legacy_lss_floor_normalizer,
)

NG = int(zgrid.size)
COSMO = CosmoParams(H0=67.74, Om0=0.3075)


def _survey(b_miss, lss_floor=None, alpha_miss=1.0):
    return SurveyParams(n0=1e-2, z50=0.3, w=0.1, delta=0.0,
                        b_miss=b_miss, alpha_miss=alpha_miss,
                        lss_floor=lss_floor)


def _centred_delta_g(n_pix, seed=0):
    """A per-pixel ``delta_g`` with the construction's own two properties:
    exactly zero full-sky mean at every z, and ``delta_g >= -1``."""
    rng = np.random.default_rng(seed)
    # Lognormal-ish contrast, then centred exactly: (x - <x>)/<x> >= -1.
    x = rng.lognormal(mean=0.0, sigma=0.6, size=(n_pix, NG))
    mean = x.mean(axis=0, keepdims=True)
    dg = (x - mean) / mean
    np.testing.assert_allclose(dg.mean(axis=0), 0.0, atol=1e-13)
    assert dg.min() >= -1.0
    return jnp.asarray(dg)


def _catalog(delta_g, n_rows=None):
    """A full-sky catalog whose rows all carry the SAME galaxies.

    Identical rows give an identical per-pixel ``C``, which is what lets the
    budget identity be read off the row sum directly: with C constant,
    ``Sum_p dN_miss_p = (1 - C) dN_exp Sum_p lss_p``.
    """
    n_rows = int(delta_g.shape[0]) if n_rows is None else n_rows
    zg = jnp.tile(jnp.asarray([[0.10, 0.25, 0.40]]), (n_rows, 1))
    dz = jnp.full_like(zg, 0.02)
    w = jnp.ones_like(zg)
    ng = jnp.full((n_rows,), 3, dtype=jnp.int32)
    kde, _ = build_pixel_kde_cache(np.arange(n_rows, dtype=np.int32), zg,
                                   n_rows, ngals=ng)
    return EMCatalog(
        apix=1.0, zgals=zg, dzgals=dz, wgals=w, ngals=ng,
        delta_g_pix_z=delta_g, dN_obs_kde=kde, pixel_to_cache_idx=None,
    )


# ── the normalizer's own contract ─────────────────────────────────────────────

@pytest.mark.parametrize("b_miss", [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
def test_renormalized_factor_has_full_sky_mean_one_at_every_z(b_miss):
    """Exact conservation, per parameter draw AND per redshift slice."""
    dg = _centred_delta_g(64, seed=3)
    cat = _catalog(dg)
    norm = legacy_lss_floor_normalizer(_survey(b_miss), cat)
    lss = np.asarray(jnp.maximum(1.0 + b_miss * dg, 0.0)) / np.asarray(norm)
    np.testing.assert_allclose(lss.mean(axis=0), 1.0, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("b_miss", [0.0, 0.5, 1.0])
def test_below_the_floor_threshold_the_normalizer_is_exactly_one(b_miss):
    """``delta_g >= -1``, so ``b_eff <= 1`` can never floor: the renormalized
    path must then be BIT-identical to the legacy one."""
    dg = _centred_delta_g(64, seed=4)
    norm = np.asarray(legacy_lss_floor_normalizer(_survey(b_miss), _catalog(dg)))
    assert np.all(norm == 1.0)


def test_the_two_pixel_example_from_the_review():
    """delta_g = (-1, +1), b_eff = 2: floored (0, 3), mean 1.5 -- a 50% budget
    inflation; b_eff = 3 doubles it.  Both are exactly one after renormalizing."""
    dg = jnp.asarray([[-1.0] * NG, [1.0] * NG])
    cat = _catalog(dg)
    for b_eff, inflation in ((2.0, 1.5), (3.0, 2.0)):
        raw = np.asarray(jnp.maximum(1.0 + b_eff * dg, 0.0))
        np.testing.assert_allclose(raw.mean(axis=0), inflation)
        norm = np.asarray(legacy_lss_floor_normalizer(_survey(b_eff), cat))
        np.testing.assert_allclose(norm, inflation)
        np.testing.assert_allclose((raw / norm).mean(axis=0), 1.0, atol=1e-15)


def test_alpha_miss_enters_only_through_the_product():
    """``b_eff = alpha_miss * b_miss``: the normalizer must see the product."""
    dg = _centred_delta_g(32, seed=8)
    cat = _catalog(dg)
    a = legacy_lss_floor_normalizer(_survey(2.0, alpha_miss=1.0), cat)
    b = legacy_lss_floor_normalizer(_survey(1.0, alpha_miss=2.0), cat)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-14)


# ── the legacy escape hatch ───────────────────────────────────────────────────

def test_legacy_sentinel_disables_the_renormalization():
    dg = _centred_delta_g(64, seed=5)
    cat = _catalog(dg)
    assert legacy_lss_floor_normalizer(
        _survey(2.5, lss_floor=LSS_FLOOR_LEGACY_STRUCT), cat) is None
    assert legacy_lss_floor_normalizer(_survey(2.5, lss_floor="legacy"),
                                       cat) is None


def test_the_dummy_delta_g_is_left_alone():
    """The ``(1, N_grid)`` broadcast dummy is the package's spelling of "no
    per-pixel LSS"; every pixel shares one row, so the sky mean IS the factor
    and dividing would erase the field rather than conserve it."""
    cat = _catalog(jnp.zeros((1, NG)), n_rows=4)
    assert legacy_lss_floor_normalizer(_survey(2.5), cat) is None


def test_an_undecodable_lss_floor_is_a_hard_error():
    """A bool/int leaf crossing a jit boundary is value-unreadable; guessing
    would silently change the TOTAL missing budget."""
    dg = _centred_delta_g(8, seed=6)
    with pytest.raises(ValueError, match="lss_floor could not be decoded"):
        legacy_lss_floor_normalizer(_survey(2.0, lss_floor=object()),
                                    _catalog(dg))


# ── end to end through completion_curves ──────────────────────────────────────

def test_completion_curves_conserve_the_missing_budget():
    """With C identical across rows, ``Sum_p dN_miss_p == N_pix (1-C) dN_exp``.

    This is the identity the floor broke: it is a statement about the TOTAL,
    which is what sets the catalog-versus-missing odds, and it must hold at
    every z for every draw.
    """
    n_pix = 48
    dg = _centred_delta_g(n_pix, seed=11)
    cat = _catalog(dg)
    for b_miss in (0.0, 1.5, 2.5, 3.0):
        conserved = completion_curves(COSMO, _survey(b_miss), cat)
        legacy = completion_curves(
            COSMO, _survey(b_miss, lss_floor=LSS_FLOOR_LEGACY_STRUCT), cat)
        # The b_miss = 0 reference: lss == 1 everywhere, so this IS the
        # unmodulated (1 - C) dN_exp budget, summed over rows.
        flat = np.asarray(
            completion_curves(COSMO, _survey(0.0), cat).dN_miss).sum(axis=0)
        got = np.asarray(conserved.dN_miss).sum(axis=0)
        np.testing.assert_allclose(got, flat, rtol=1e-12)
        # ... and the legacy path is NOT conserved once the floor engages.
        raw = np.asarray(legacy.dN_miss).sum(axis=0)
        if b_miss > 1.0:
            assert np.max(np.abs(raw / np.where(flat > 0, flat, 1.0) - 1.0)) > 1e-3


def test_conserving_default_is_bit_identical_when_the_floor_never_engages():
    """b_eff <= 1 cannot floor, so the new default must not perturb a single
    bit of an existing run."""
    dg = _centred_delta_g(32, seed=13)
    cat = _catalog(dg)
    for b_miss in (0.0, 0.4, 1.0):
        a = completion_curves(COSMO, _survey(b_miss), cat)
        b = completion_curves(
            COSMO, _survey(b_miss, lss_floor=LSS_FLOOR_LEGACY_STRUCT), cat)
        np.testing.assert_array_equal(np.asarray(a.dN_miss),
                                      np.asarray(b.dN_miss))
        np.testing.assert_array_equal(np.asarray(a.N_miss),
                                      np.asarray(b.N_miss))


def test_renormalization_preserves_the_angular_pattern():
    """It is a budget fix, not a field fix: the ratio between any two pixels'
    modulation is untouched wherever both are above the floor."""
    dg = _centred_delta_g(16, seed=17)
    cat = _catalog(dg)
    b_miss = 2.5
    norm = np.asarray(legacy_lss_floor_normalizer(_survey(b_miss), cat))
    raw = np.asarray(jnp.maximum(1.0 + b_miss * dg, 0.0))
    scaled = raw / norm
    both = (raw[0] > 0.0) & (raw[1] > 0.0)
    assert both.any()
    np.testing.assert_allclose(
        scaled[0][both] / scaled[1][both], raw[0][both] / raw[1][both],
        rtol=1e-13)
