"""
test_catalog_kernel_pin.py
--------------------------
The build-time H0 pin of the catalog quadratures.

At fixed Om0, w0, wa, delta and sigma_kde the galaxy measure obeys
``g(z; H0) = (H0_ref/H0)^3 g(z; H0_ref)`` EXACTLY, so the per-galaxy kernel
normalisations are a build-time constant plus the scalar ``+3 ln(H0/H0_ref)`` on
``log_kw``, ``log_depth_mass`` is H0-invariant, and the survey-global observed
total ``Sum_i c_i Z_i^depth / Z_i^full`` is a run constant.  These tests pin
that identity, the gate that decides when it may be used, and the in-graph
probes that make a stale or mis-installed pin loud instead of plausible.
"""
import numpy as np
import pytest

pytest.importorskip("jax")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.core.types import CosmoParams, EMCatalog, GWEvent, SurveyParams
from darksirens.gw.populations import get_fixed_population_params
from darksirens.inference.parameters import ParameterDecoder
from darksirens.likelihood.core import darksiren_log_likelihood
from darksirens.likelihood.factory import _install_kernel_pin, kernel_pin_admissible
from darksirens.redshift import zgrid
from darksirens.redshift import catalog as C
from darksirens.redshift import completion as K
from darksirens.redshift.prior import (
    eval_redshift_prior_with_state,
    prepare_redshift_prior_state,
)

NG = int(zgrid.size)
H0_SCAN = (20.0, 40.0, 67.74, 100.0, 140.0)
REF = CosmoParams(H0=C.KERNEL_PIN_H0_REF, Om0=0.3075, w0=-1.0, wa=0.0)
Z_DEPTH = 0.30
POP = jnp.asarray(get_fixed_population_params("powerlaw+peak"))


def _survey(**kw):
    base = dict(n0=1.0, z50=0.15, w=0.08, delta=0.94, b_miss=0.0,
                alpha_miss=1.0, sigma_kde=0.003, z_depth=Z_DEPTH)
    base.update(kw)
    return SurveyParams(**base)


def _rows_catalog(seed=7, n_rows=6, n_max=12):
    """Multi-row catalog with an EMPTY pixel row (index 1) and z-sorted rows."""
    rng = np.random.default_rng(seed)
    ngals = np.array([5, 0, 12, 3, 1, 7], dtype=np.int32)[:n_rows]
    z = np.full((n_rows, n_max), 100.0)   # padding parks above the grid
    dz = np.ones((n_rows, n_max))
    w = np.zeros((n_rows, n_max))
    for r in range(n_rows):
        n = int(ngals[r])
        if not n:
            continue
        z[r, :n] = np.sort(rng.uniform(0.01, 0.45, n))
        dz[r, :n] = rng.uniform(1e-4, 0.03, n)
        w[r, :n] = rng.uniform(0.5, 2.0, n)
    kde, idx = K.build_pixel_kde_cache(
        np.arange(n_rows, dtype=np.int32), jnp.asarray(z), n_rows,
        ngals=jnp.asarray(ngals))
    return EMCatalog(
        apix=1.0, zgals=jnp.asarray(z), dzgals=jnp.asarray(dz),
        wgals=jnp.asarray(w), ngals=jnp.asarray(ngals),
        delta_g_pix_z=jnp.zeros((n_rows, NG)), dN_obs_kde=kde,
        pixel_to_cache_idx=idx, unique_pixels=None,
    )


def _flat_field_catalog(seed=11, n_gal=400):
    """Flat FULL-SKY depth inputs, the survey-global observed total's operand."""
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.01, 0.5, n_gal))
    dz = rng.uniform(1e-4, 0.03, n_gal)
    c = rng.uniform(0.5, 2.0, n_gal)
    return EMCatalog(
        apix=1.0, zgals=None, dzgals=None, wgals=None, ngals=None,
        delta_g_pix_z=None, dN_obs_kde=None, pixel_to_cache_idx=None,
        field_depth_z=jnp.asarray(z), field_depth_dz=jnp.asarray(dz),
        field_depth_c=jnp.asarray(c), field_N_obs_total=float(c.sum()),
    )


# ---------------------------------------------------------------------------
# (a) the pin reproduces the full quadrature
# ---------------------------------------------------------------------------

def test_pinned_state_matches_full_build_across_the_H0_prior():
    """max |dlog_kw| and max |dlog p_cat| stay at f64 rounding over H0 in
    [20, 140], with a depth truncation and an empty pixel row in the view."""
    cat, sur = _rows_catalog(), _survey()
    pin = C.build_pinned_kernel_quadrature(REF, sur, cat, z_depth=sur.z_depth)

    z_probe = jnp.asarray([0.05, 0.12, 0.20, 0.29])
    pix_probe = jnp.asarray([0, 1, 2, 5])
    evaluate = jax.vmap(C.eval_log_catalog_prior_state, in_axes=(0, 0, None, None))

    worst_kw = worst_pcat = worst_depth = 0.0
    for H0 in H0_SCAN:
        cosmo = REF._replace(H0=H0)
        full = C.catalog_kernel_state(cosmo, sur, cat, z_depth=sur.z_depth)
        pinned = C.catalog_kernel_state(cosmo, sur, cat, z_depth=sur.z_depth,
                                        pinned=pin)
        assert bool(pinned.pin_ok)

        a, b = np.asarray(full.log_kw), np.asarray(pinned.log_kw)
        real = a > -1e29
        # Padding is the -1e30 sentinel in both: -1e30 + 3 ln h is still -1e30.
        assert np.array_equal(a[~real], b[~real])
        worst_kw = max(worst_kw, float(np.max(np.abs(a[real] - b[real]))))
        worst_depth = max(worst_depth, float(np.max(np.abs(
            np.asarray(full.log_depth_mass) - np.asarray(pinned.log_depth_mass)))))

        # The theta-invariant leaves are returned unchanged, not recomputed.
        assert np.array_equal(np.asarray(full.sig_eff), np.asarray(pinned.sig_eff))
        assert np.array_equal(np.asarray(full.row_empty),
                              np.asarray(pinned.row_empty))
        assert np.array_equal(np.asarray(full.sig_eff_row_max),
                              np.asarray(pinned.sig_eff_row_max))

        p_full = np.asarray(evaluate(z_probe, pix_probe, full, cat))
        p_pin = np.asarray(evaluate(z_probe, pix_probe, pinned, cat))
        finite = np.isfinite(p_full)
        assert np.array_equal(finite, np.isfinite(p_pin))
        assert finite.any() and (~finite).any(), "need both an empty and a live row"
        worst_pcat = max(worst_pcat,
                         float(np.max(np.abs(p_full[finite] - p_pin[finite]))))

    assert worst_kw < 1e-13, worst_kw
    assert worst_depth < 1e-13, worst_depth
    assert worst_pcat < 1e-13, worst_pcat


def test_pinned_log_kw_carries_exactly_the_closed_form_shift():
    """``log_kw(H0) - log_kw(H0_ref) == 3 ln(H0/H0_ref)`` on every real galaxy."""
    cat, sur = _rows_catalog(), _survey()
    base = np.asarray(C.catalog_kernel_state(
        REF, sur, cat, z_depth=sur.z_depth).log_kw)
    real = base > -1e29
    for H0 in H0_SCAN:
        live = np.asarray(C.catalog_kernel_state(
            REF._replace(H0=H0), sur, cat, z_depth=sur.z_depth).log_kw)
        shift = 3.0 * np.log(H0 / C.KERNEL_PIN_H0_REF)
        assert np.max(np.abs(live[real] - base[real] - shift)) < 1e-13


# ---------------------------------------------------------------------------
# (b) the gate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("blocked", ["Om0", "w0", "wa", "delta", "sigma_kde",
                                     "Om0_c2", "sigma_kde_c3"])
def test_gate_refuses_every_premise_label(blocked):
    labels = ("H0", "log10n0", "M0hat", "sigma_M", blocked)
    assert not kernel_pin_admissible(labels, "dark_sirens", "none")


def test_gate_admits_the_production_sampled_set():
    labels = ("H0", "log10n0", "M0hat", "sigma_M", "alpha", "mu_g", "b_miss")
    assert kernel_pin_admissible(labels, "dark_sirens", "none")
    # ... and only for the plain galaxy-count dark-siren host model.
    assert not kernel_pin_admissible(labels, "dark_sirens", "logmstar")
    assert not kernel_pin_admissible(labels, "dark_sirens_complete", "none")
    assert not kernel_pin_admissible(labels, "spectral_sirens", "none")


def _decoder(sampled_labels):
    return ParameterDecoder(
        sampled_labels=tuple(sampled_labels),
        fixed_parameter_values={"Om0": 0.3075, "delta": 0.94,
                                "sigma_kde": 0.003},
        pop_labels=(), pop_params_fid=(),
        complete_empty_pixel_policy=0, z_depths=(Z_DEPTH,),
    )


def test_install_is_a_no_op_when_the_gate_is_off():
    """Om0 or sigma_kde sampled -> nothing is attached, so every consumer takes
    the pre-existing code path op for op (``pin_ok`` stays ``None``, and
    ``kernel_pin_poison`` then emits no node at all)."""
    cat = _rows_catalog()
    for blocked in ("Om0", "sigma_kde"):
        pe, sel = _install_kernel_pin(
            _decoder(("H0", blocked)), "dark_sirens", "none", "conditional",
            (cat,), (cat,))
        assert pe[0].pinned_kernels is None
        assert pe[0].field_depth_total_pinned is None
        assert sel[0].pinned_kernels is None
        state = C.catalog_kernel_state(REF, _survey(), pe[0], z_depth=Z_DEPTH,
                                       pinned=pe[0].pinned_kernels)
        assert state.pin_ok is None
        assert C.kernel_pin_poison(state) is None


def test_install_shares_one_pin_between_views_that_share_their_rows():
    """A union bundle's PE and selection views must carry the SAME pin objects,
    or ``can_share_redshift_prior_state`` (which compares leaves by identity)
    would stop deduplicating the prior-state build."""
    cat = _rows_catalog()
    pe, sel = _install_kernel_pin(
        _decoder(("H0", "log10n0")), "dark_sirens", "none", "conditional",
        (cat,), (cat,))
    assert pe[0].pinned_kernels is not None
    assert pe[0].pinned_kernels is sel[0].pinned_kernels


# ---------------------------------------------------------------------------
# (c) the in-graph probe
# ---------------------------------------------------------------------------

def _dark_gw(n_events, n_samp, seed, n_rows):
    rng = np.random.default_rng(seed)
    total = n_events * n_samp
    m1det = jnp.asarray(rng.uniform(20.0, 60.0, total))
    m2det = jnp.asarray(rng.uniform(8.0, 30.0, total))
    dL = jnp.asarray(rng.uniform(420.0, 1500.0, total))
    return GWEvent(
        m1det=m1det, m2det=m2det, dL=dL,
        chieff=jnp.asarray(rng.uniform(-0.2, 0.2, total)),
        prior_wt=jnp.asarray(rng.uniform(0.5, 1.5, total)),
        pixels=jnp.asarray(rng.integers(0, n_rows, total), dtype=jnp.int32),
        q=m2det / m1det, valid=jnp.ones(total, dtype=jnp.bool_),
    )


_N_EV, _N_SAMP, _N_SEL = 4, 64, 300
_GW_PE = _dark_gw(_N_EV, _N_SAMP, seed=0, n_rows=6)
_GW_SEL = _dark_gw(_N_SEL, 1, seed=10, n_rows=6)


def _log_likelihood(cat, cosmo, survey):
    return float(darksiren_log_likelihood(
        cosmo, survey, POP, _GW_PE, cat, _GW_SEL, cat,
        _N_EV, _N_SAMP, float(_N_SEL),
        pop_model="powerlaw+peak", universe_model="dark_sirens",
        sel_batch_size=None,
    ))


def _prior_log_p(cat, cosmo, survey, z, pix):
    state = prepare_redshift_prior_state("dark_sirens", cosmo, survey, cat)
    return np.asarray(eval_redshift_prior_with_state(
        "dark_sirens", state, z, pix, cosmo, survey, cat))


def test_likelihood_is_unchanged_by_the_pin():
    """End to end: the pinned graph reproduces the unpinned one to f64."""
    cat, sur = _rows_catalog(), _survey()
    pe, _ = _install_kernel_pin(
        _decoder(("H0", "log10n0")), "dark_sirens", "none", "conditional",
        (cat,), (cat,))
    for H0 in H0_SCAN:
        cosmo = REF._replace(H0=H0)
        ref = _log_likelihood(cat, cosmo, sur)
        got = _log_likelihood(pe[0], cosmo, sur)
        assert np.isfinite(ref)
        assert got == pytest.approx(ref, rel=1e-12, abs=1e-9), (H0, got, ref)


def test_corrupted_pin_poisons_the_prior_and_the_likelihood():
    """A hand-corrupted pin fails the probe: the redshift prior goes NaN and the
    likelihood stops being a number a sampler could accept.  ``log_p_cat`` alone
    could NOT carry this -- ``_eval_dark_scalar`` rewrites NaN there as -inf --
    which is why the verdict is spent on the normalizer."""
    cat, sur = _rows_catalog(), _survey()
    pin = C.build_pinned_kernel_quadrature(REF, sur, cat, z_depth=sur.z_depth)
    bad = cat._replace(pinned_kernels=pin._replace(log_kw=pin.log_kw + 1.0))

    state = C.catalog_kernel_state(REF, sur, bad, z_depth=sur.z_depth,
                                   pinned=bad.pinned_kernels)
    assert not bool(state.pin_ok)
    assert np.isnan(float(C.kernel_pin_poison(state)))

    z = jnp.asarray([0.05, 0.12, 0.20])
    pix = jnp.asarray([0, 2, 5])
    assert np.all(np.isfinite(_prior_log_p(cat, REF, sur, z, pix)))
    assert np.all(np.isnan(_prior_log_p(bad, REF, sur, z, pix)))
    # The likelihood core's final `where(isfinite(ll), ll, -inf)` turns that NaN
    # into -inf, so the poisoned run is loud rather than plausible.
    assert not np.isfinite(_log_likelihood(bad, REF, sur))
    assert np.isfinite(_log_likelihood(cat, REF, sur))


@pytest.mark.parametrize("violation", [
    {"Om0": 0.35},
    {"delta": 2.0},
    {"sigma_kde": 0.03},
    {"z_depth": 0.40},
])
def test_probe_fires_when_the_premise_is_violated_after_the_build(violation):
    """The replay the gate cannot prevent: a graph built under fixed
    Om0/delta/sigma_kde/z_depth, evaluated at a proposal that moved one of
    them."""
    cat, sur = _rows_catalog(), _survey()
    pin = C.build_pinned_kernel_quadrature(REF, sur, cat, z_depth=sur.z_depth)
    cosmo = REF._replace(**{k: v for k, v in violation.items() if k == "Om0"})
    survey = sur._replace(**{k: v for k, v in violation.items() if k != "Om0"})
    state = C.catalog_kernel_state(cosmo, survey, cat, z_depth=survey.z_depth,
                                   pinned=pin)
    assert not bool(state.pin_ok)
    assert np.isnan(float(C.kernel_pin_poison(state)))


# ---------------------------------------------------------------------------
# (d) the survey-global observed total
# ---------------------------------------------------------------------------

def test_field_observed_total_is_a_run_constant_the_pin_reproduces():
    """``Sum_i c_i Z_i^depth / Z_i^full`` is H0-INVARIANT (both quadratures
    carry the same (H0_ref/H0)^3), so the pin is a constant, not a shift."""
    cat, sur = _flat_field_catalog(), _survey()
    pin = K.build_pinned_field_observed_total(REF, sur, cat)
    pinned_cat = cat._replace(field_depth_total_pinned=pin)
    for H0 in H0_SCAN:
        cosmo = REF._replace(H0=H0)
        live = float(K.field_observed_global_total(cosmo, sur, cat))
        got = float(K.field_observed_global_total(cosmo, sur, pinned_cat))
        assert live > 0.0
        assert abs(got - live) / live < 1e-12, (H0, got, live)


@pytest.mark.parametrize("violation", [
    {"Om0": 0.35},
    {"delta": 2.0},
    {"sigma_kde": 0.03},
])
def test_field_observed_total_probe_fires_on_a_violated_premise(violation):
    """The field pin carries its OWN probe: at K>=2 this scalar does not cancel
    between the PE and selection seams, so a mis-installed one is a silent tilt
    on the sampled host fractions rather than a common offset."""
    cat, sur = _flat_field_catalog(), _survey()
    pinned_cat = cat._replace(
        field_depth_total_pinned=K.build_pinned_field_observed_total(REF, sur, cat))
    cosmo = REF._replace(**{k: v for k, v in violation.items() if k == "Om0"})
    survey = sur._replace(**{k: v for k, v in violation.items() if k != "Om0"})
    assert np.isfinite(float(K.field_observed_global_total(cosmo, survey, cat)))
    assert np.isnan(float(K.field_observed_global_total(cosmo, survey, pinned_cat)))
