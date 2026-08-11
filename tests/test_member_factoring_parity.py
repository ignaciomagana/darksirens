"""
test_member_factoring_parity.py
-------------------------------
A/B parity of the FACTORED LSS-completion member marginalization
(``lss_member_impl="factored"``, the default) against the REFERENCE
whole-likelihood vmap (``lss_member_impl="reference"``) inside
``darksiren_log_likelihood``.

The factored path precomputes the member-INDEPENDENT per-sample work ONCE (the
population model, the z(dL) inversion + Jacobians, the proposal reweighting, the
sky factor, and the O(N_max_gals) observed-catalog KDE) and vmaps ONLY the cheap
missing-galaxy completion over the M ensemble members; the reference re-runs the
entire per-member likelihood.  They must agree to floating-point re-association
on BOTH the value AND ``jax.grad`` -- ``rtol=1e-12`` -- across the K=1/K>=2 x
conditional/field x unmarked/marked feature matrix, selection batching, and an
ACTIVE selection variance / soft guard.  Both impls must also REFUSE the
asymmetric PE-members / selection-no-members structure (the per-member
normalizer does not cancel against a mean-Q mu).

Conditional cases build inputs at the direct ``darksiren_log_likelihood`` level
(as tests/test_lss_marginalization.py does); field cases reuse the validated
bundle fixtures of tests/test_marks_lss_marginalize.py through
``make_likelihood`` and toggle the implementation by wrapping the core call the
factory makes.
"""
import os
import sys

import numpy as np
import pytest

pytest.importorskip("jax")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
import test_lss_marginalization as L  # noqa: E402
import test_marks_lss_marginalize as MK  # noqa: E402
import darksirens.likelihood.factory as _factory  # noqa: E402
from darksirens.likelihood.core import darksiren_log_likelihood  # noqa: E402
from darksirens.redshift import zgrid  # noqa: E402

NG = int(zgrid.size)
RTOL = 1e-12
# Near-zero floor: some gradients are analytically 0 (e.g. an overall mixture-
# weight normalization cancels between the PE and selection terms), where both
# impls return only float64 reassociation noise ~1e-16 whose RELATIVE difference
# is meaningless.  1e-12 stays a tight absolute floor -- far below any real
# factored-vs-reference drift, which would show as a large relative diff on a
# non-zero value.
ATOL = 1e-12


# ===========================================================================
# Helpers
# ===========================================================================

def _assert_ab(vf, vr, gf, gr, tag):
    """Value + grad parity of factored (f) vs reference (r) at rtol=1e-12."""
    vf, vr = np.asarray(vf, float), np.asarray(vr, float)
    assert np.all(np.isfinite(vf)), (tag, "factored value not finite", vf)
    assert np.all(np.isfinite(vr)), (tag, "reference value not finite", vr)
    np.testing.assert_allclose(vf, vr, rtol=RTOL, atol=ATOL,
                               err_msg=f"{tag}: value drift")
    gf, gr = np.asarray(gf, float), np.asarray(gr, float)
    assert np.all(np.isfinite(gf)), (tag, "factored grad not finite", gf)
    assert np.all(np.isfinite(gr)), (tag, "reference grad not finite", gr)
    np.testing.assert_allclose(gf, gr, rtol=RTOL, atol=ATOL,
                               err_msg=f"{tag}: grad drift")
    # The margins actually observed (surfaced on -v runs).
    return (
        float(np.max(np.abs(vf - vr) / np.maximum(np.abs(vr), 1e-300))),
        float(np.max(np.abs(gf - gr) / np.maximum(np.abs(gr), 1e-300))),
    )


# --- Direct-core (conditional) A/B ----------------------------------------

def _direct_ab(build, theta0, tag):
    """``build(impl)`` returns a scalar function of one traced ``theta``; compare
    value and grad of the factored vs reference implementations at ``theta0``."""
    ff, fr = build("factored"), build("reference")
    vf, vr = ff(theta0), fr(theta0)
    gf = jax.grad(ff)(theta0)
    gr = jax.grad(fr)(theta0)
    return _assert_ab(vf, vr, gf, gr, tag)


# --- Factory (field) A/B via a core-call wrapper --------------------------

_REAL_CORE = _factory.darksiren_log_likelihood


def _force_impl(impl):
    def _wrapper(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["lss_member_impl"] = impl
        return _REAL_CORE(*args, **kwargs)

    _factory.darksiren_log_likelihood = _wrapper


def _restore_core():
    _factory.darksiren_log_likelihood = _REAL_CORE


def _factory_ab(ll, coord, tag):
    """A/B a ``make_likelihood`` closure by forcing lss_member_impl on the core
    call it makes (the factory itself takes no impl flag)."""
    coord = jnp.asarray(coord)
    try:
        _force_impl("factored")
        vf = float(ll(coord))
        gf = np.asarray(jax.grad(lambda c: ll(c))(coord))
        _force_impl("reference")
        vr = float(ll(coord))
        gr = np.asarray(jax.grad(lambda c: ll(c))(coord))
    finally:
        _restore_core()
    return _assert_ab(vf, vr, gf, gr, tag)


# ===========================================================================
# Conditional cases (direct core calls)
# ===========================================================================

def _core_k1(cat_pe, cat_sel, *, marginalize, mark_model="none", eta=None,
             sel_batch_size=None, soft_guard=False, max_var=None):
    """K=1 direct core call closure over pop_params[0] (population lever)."""
    def build(impl):
        def f(theta):
            pop = L.POP.at[0].set(theta)
            kw = dict(
                pop_model="powerlaw+peak", universe_model="dark_sirens",
                sel_batch_size=sel_batch_size, lss_marginalize=marginalize,
                lss_member_impl=impl, mark_model=mark_model,
                mark_params=(None if eta is None else jnp.asarray(eta)),
                mark_names=(() if mark_model == "none" else MK._MARK_NAMES),
                selection_neff_soft_guard=soft_guard,
            )
            if max_var is not None:
                kw["max_likelihood_variance"] = max_var
            return darksiren_log_likelihood(
                L.COSMO, L.SURVEY, pop, L._GW_PE, cat_pe, L._GW_SEL, cat_sel,
                L._N_EV, L._N_SAMP, float(L._N_SEL), **kw,
            )
        return f
    return build


def test_k1_conditional():
    cat = L._dark_catalog(logq_members=L._members_table())
    _direct_ab(_core_k1(cat, cat, marginalize=True), float(L.POP[0]),
               "k1_conditional")


def test_k1_conditional_batched_selection():
    cat = L._dark_catalog(logq_members=L._members_table())
    _direct_ab(_core_k1(cat, cat, marginalize=True, sel_batch_size=100),
               float(L.POP[0]), "k1_conditional_batched")


def test_k1_marked_conditional_grad_through_eta():
    """K=1 MARKED ensemble; grad taken w.r.t. eta so the reverse pass runs
    THROUGH the member completion (eta reshapes dN_miss_members)."""
    cat = MK._dark_catalog(logq_members=MK._members_table())

    def build(impl):
        def f(eta0):
            return darksiren_log_likelihood(
                MK.COSMO, MK.SURVEY, MK.POP, MK._GW_PE, cat, MK._GW_SEL, cat,
                MK._N_EV, MK._N_SAMP, float(MK._N_SEL),
                pop_model="powerlaw+peak", universe_model="dark_sirens",
                sel_batch_size=None, lss_marginalize=True, lss_member_impl=impl,
                mark_model="loglinear", mark_params=jnp.asarray([eta0]),
                mark_names=MK._MARK_NAMES,
                # eta reshapes dN_miss_members INSIDE the prior state, so the
                # reverse pass crosses the redshift-prior optimization barrier;
                # drop it (the numpyro/NUTS setting) so jax.grad is defined.
                materialize_redshift_prior_state=False,
            )
        return f

    _direct_ab(build, 1.3, "k1_marked_conditional")


def test_asymmetric_pe_members_sel_no_members_is_rejected():
    """PE catalog carries the ensemble, selection catalog does NOT: each member's
    numerator carries 1/Z_m, which cancels only against mu(Q_m), so pairing it
    with a posterior-mean-Q mu makes the member average a Z_m^{-N_obs}-weighted
    pick instead of a marginalization.  Both impls must refuse it (this used to
    be silently supported by hoisting the selection term out of the vmap)."""
    cat_pe = L._dark_catalog(logq_members=L._members_table())
    cat_sel = L._dark_catalog(logq=np.zeros((2, NG)))  # deterministic, no members
    for impl in ("factored", "reference"):
        with pytest.raises(ValueError, match="SELECTION catalog"):
            _core_k1(cat_pe, cat_sel, marginalize=True)(impl)(float(L.POP[0]))


def test_selection_soft_guard_active():
    """Selection variance/soft guard ACTIVE (tiny variance budget so the smooth
    wall bites): the guard is applied identically per member in both impls."""
    cat = L._dark_catalog(logq_members=L._members_table())
    build = _core_k1(cat, cat, marginalize=True, soft_guard=True, max_var=1e-3)
    # Confirm the guard actually engaged (penalized well below the un-guarded value).
    unguarded = _core_k1(cat, cat, marginalize=True)("factored")(float(L.POP[0]))
    guarded = build("factored")(float(L.POP[0]))
    assert float(guarded) < float(unguarded) - 1.0, (float(guarded), float(unguarded))
    _direct_ab(build, float(L.POP[0]), "soft_guard_active")


def test_selection_hard_guard_both_neg_inf():
    """Hard guard with an exhausted variance budget: both impls return -inf
    (parity at the wall)."""
    cat = L._dark_catalog(logq_members=L._members_table())
    build = _core_k1(cat, cat, marginalize=True, soft_guard=False, max_var=1e-6)
    vf = float(build("factored")(float(L.POP[0])))
    vr = float(build("reference")(float(L.POP[0])))
    assert vf == -np.inf and vr == -np.inf, (vf, vr)


# --- K=2 conditional (direct core, duplicated catalog) ---------------------

def _core_k2_conditional(cat_a, cat_b, *, marginalize, log_w):
    """K=2 direct core closure over mixture_log_weights[1] (mixture lever)."""
    pix2_pe = jnp.stack([L._GW_PE.pixels, L._GW_PE.pixels], axis=1)
    pix2_sel = jnp.stack([L._GW_SEL.pixels, L._GW_SEL.pixels], axis=1)
    gw_pe = L._GW_PE._replace(pixels=pix2_pe)
    gw_sel = L._GW_SEL._replace(pixels=pix2_sel)

    def build(impl):
        def f(logw1):
            mw = jnp.asarray([log_w[0], logw1])
            return darksiren_log_likelihood(
                L.COSMO, L.SURVEY, L.POP, gw_pe, cat_a, gw_sel, cat_a,
                L._N_EV, L._N_SAMP, float(L._N_SEL),
                pop_model="powerlaw+peak", universe_model="dark_sirens",
                sel_batch_size=None, lss_marginalize=marginalize,
                lss_member_impl=impl, n_catalogs=2,
                mixture_surveys=(L.SURVEY,),
                mixture_em_catalogs_pe=(cat_b,),
                mixture_em_catalogs_sel=(cat_b,),
                mixture_log_weights=mw,
            )
        return f
    return build


def test_k2_conditional_grad_through_mixture_weight():
    # DISTINCT per-catalog Q ensembles so the mixture-weight gradient is
    # genuinely non-zero (a duplicated catalog makes the weight an overall
    # normalization that cancels between PE and selection -> grad == 0).
    cat_a = L._dark_catalog(logq_members=L._members_table())
    cat_b = L._dark_catalog(logq_members=0.4 * L._members_table())
    log_w = np.log(np.array([0.6, 0.4]))
    _direct_ab(_core_k2_conditional(cat_a, cat_b, marginalize=True, log_w=log_w),
               float(log_w[1]), "k2_conditional")


# ===========================================================================
# Field cases (factory bundle fixtures; toggle impl on the core call)
# ===========================================================================

def test_k1_field():
    pop_fid, overrides, fixed, mid = MK._pop_bits()
    logq_m = MK._logq_members_table()
    ll = MK._field_likelihood(
        [MK._marked_field_bundle(logq_members=logq_m, mark_scale=1.0)],
        (("logmstar",),), fixed, pop_fid, overrides,
        lss_marginalize=True, barrier="off",
    )
    _factory_ab(ll, [mid, 0.7], "k1_field")


def test_k2_field():
    pop_fid, overrides, fixed, mid = MK._pop_bits()
    logq_m = MK._logq_members_table()
    fixed_k2 = dict(fixed)
    fixed_k2["eta_logmstar_c2"] = 0.8
    ll = MK._field_likelihood(
        [MK._marked_field_bundle(logq_members=logq_m, mark_scale=1.0),
         MK._marked_field_bundle(logq_members=logq_m, mark_scale=1.0)],
        (("logmstar",), ("logmstar",)), fixed_k2, pop_fid, overrides,
        lss_marginalize=True, barrier="off",
    )
    _factory_ab(ll, [mid, 0.37, 0.8], "k2_field")


def test_k2_one_marked_one_unmarked_field():
    """K=2 field, catalog 1 MARKED + catalog 2 UNMARKED, both ensembles,
    marginalized -> coord is [pop, fcat_2, eta_c1]."""
    pop_fid, overrides, fixed, mid = MK._pop_bits()
    logq_m = MK._logq_members_table()
    ll = MK._field_likelihood(
        [MK._marked_field_bundle(logq_members=logq_m, mark_scale=1.0),
         MK._marked_field_bundle(logq_members=logq_m, mark_scale=None)],
        (("logmstar",), ()), fixed, pop_fid, overrides,
        lss_marginalize=True, barrier="off",
    )
    _factory_ab(ll, [mid, 0.5, 0.6], "k2_mixed_field")


if __name__ == "__main__":  # pragma: no cover - manual margin dump
    for fn in [
        test_k1_conditional, test_k1_conditional_batched_selection,
        test_k1_marked_conditional_grad_through_eta,
        test_asymmetric_pe_members_sel_no_members_is_rejected,
        test_selection_soft_guard_active, test_selection_hard_guard_both_neg_inf,
        test_k2_conditional_grad_through_mixture_weight,
        test_k1_field, test_k2_field, test_k2_one_marked_one_unmarked_field,
    ]:
        fn()
        print("ok", fn.__name__)
