"""
test_member_pe_vectorization.py
-------------------------------
Block-vectorization parity for the FACTORED member path's per-event PE
precompute (``darksirens/likelihood/core.py::_factored_member_marginalization``).

Item A replaced the per-event ``lax.scan(_pe_precompute, ...)`` with the SAME
static block plan #252 used on the deterministic per-event reduction: a chunk of
``pe_event_block`` events is evaluated in ONE flattened ``(m*nsamp,)`` pass of the
member-INDEPENDENT kernels (``log_target_density_base_and_z``, the per-catalog
``eval_dark_obs_bracket``, the sky factor, masks) and reshaped to ``(m, nsamp,
...)``.  Those kernels are elementwise in the sample axis, so the stacked
``pe_pre`` pytree is identical (up to concatenation order) for every block size,
and the downstream ``_pe_member_terms`` consumer is unchanged.

The invariant checked here: on the factored path (``lss_marginalize=True``,
``lss_member_impl='factored'``) the marginalised log-likelihood and its gradient
are independent of ``pe_event_block``:

  * ``pe_event_block=None`` (all events in one flattened pass, no scan) MUST equal
  * ``pe_event_block=1`` (per-event scan, the historical shape) and
  * ``pe_event_block=3`` (remainder: 7 events -> chunks (3, 3, 1), exercising the
    n_full>1 scan AND the Python remainder chunk).

Because the concatenation happens at the ``pe_pre`` level (before the row
reduction), the vmap that reduces ``(nEvents, nsamp)`` sees ONE contiguous array
regardless of block size, so values are expected BIT-IDENTICAL; gradients agree
to ~1 ULP (the reverse pass through the flat-vs-scan precompute reassociates a
shared upstream cotangent).  Run with ``JAX_PLATFORMS=cpu``.
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
from darksirens.likelihood.core import darksiren_log_likelihood  # noqa: E402
from darksirens.redshift import zgrid  # noqa: E402

NG = int(zgrid.size)
# 7 events -> block=3 gives chunks (3, 3, 1): 2 full chunks (lax.scan, n_full>1)
# plus a 1-event Python remainder.  block=None is one flattened pass of all 7.
_N_EV7 = 7
_GW_PE7 = L._gw(_N_EV7, L._N_SAMP, seed=0)
_BLOCKS = [None, 1, 3]


# ---------------------------------------------------------------------------
# Closures over pe_event_block for the factored path.
# ---------------------------------------------------------------------------

def _k1_builder(cat_pe, cat_sel, gw_pe, n_ev, *, sel_batch_size=None,
                mark_model="none", eta=None):
    """K=1 factored-path closure over pop_params[0], parametrised by block."""
    def build(pe_block):
        def f(theta):
            pop = L.POP.at[0].set(theta)
            return darksiren_log_likelihood(
                L.COSMO, L.SURVEY, pop, gw_pe, cat_pe, L._GW_SEL, cat_sel,
                n_ev, L._N_SAMP, float(L._N_SEL),
                pop_model="powerlaw+peak", universe_model="dark_sirens",
                sel_batch_size=sel_batch_size, lss_marginalize=True,
                lss_member_impl="factored", pe_event_block=pe_block,
                mark_model=mark_model,
                mark_params=(None if eta is None else jnp.asarray(eta)),
                mark_names=(() if mark_model == "none" else MK._MARK_NAMES),
            )
        return f
    return build


def _k2_builder(cat_a, cat_b, gw_pe, n_ev, log_w):
    """K=2 factored-path closure over mixture_log_weights[1], by block."""
    pix2 = jnp.stack([gw_pe.pixels, gw_pe.pixels], axis=1)
    pix2_sel = jnp.stack([L._GW_SEL.pixels, L._GW_SEL.pixels], axis=1)
    gw_pe2 = gw_pe._replace(pixels=pix2)
    gw_sel2 = L._GW_SEL._replace(pixels=pix2_sel)

    def build(pe_block):
        def f(logw1):
            mw = jnp.asarray([log_w[0], logw1])
            return darksiren_log_likelihood(
                L.COSMO, L.SURVEY, L.POP, gw_pe2, cat_a, gw_sel2, cat_a,
                n_ev, L._N_SAMP, float(L._N_SEL),
                pop_model="powerlaw+peak", universe_model="dark_sirens",
                sel_batch_size=None, lss_marginalize=True,
                lss_member_impl="factored", pe_event_block=pe_block,
                n_catalogs=2, mixture_surveys=(L.SURVEY,),
                mixture_em_catalogs_pe=(cat_b,), mixture_em_catalogs_sel=(cat_b,),
                mixture_log_weights=mw,
            )
        return f
    return build


def _assert_block_invariant(build, theta0, tag, grad=True):
    """Values (and grads) at block None/1/3 must agree; report bitwise-ness."""
    vals = {b: float(build(b)(theta0)) for b in _BLOCKS}
    for b, v in vals.items():
        assert np.isfinite(v), (tag, b, v)
    v_none = vals[None]
    # None (one flattened pass) vs 1 (per-event scan): expected BIT-IDENTICAL.
    assert vals[1] == v_none, (tag, "None!=1", vals[1], v_none, vals[1] - v_none)
    # block=3 (remainder chunks) reduces the identical pe_pre in the identical
    # row order; only pe_pre's assembly reassociates -> tolerance well under the
    # golden pin.
    np.testing.assert_allclose(vals[3], v_none, rtol=1e-12, atol=0.0,
                               err_msg=f"{tag}: block=3 vs None value")
    margins = {"val_none_vs_3": abs(vals[3] - v_none)}
    if grad:
        grads = {b: np.asarray(jax.grad(build(b))(theta0)) for b in _BLOCKS}
        for b, g in grads.items():
            assert np.all(np.isfinite(g)), (tag, "grad not finite", b, g)
        gn = grads[None]
        # grad parity <= 1e-12 (report if bitwise): the forward is bit-identical;
        # the reverse pass reassociates a shared upstream cotangent (flat vs scan).
        for b in (1, 3):
            np.testing.assert_allclose(grads[b], gn, rtol=1e-12, atol=1e-12,
                                       err_msg=f"{tag}: grad block={b} vs None")
        margins["grad_none_vs_1_bitwise"] = bool(np.array_equal(grads[1], gn))
        margins["grad_none_vs_3_max"] = float(np.max(np.abs(grads[3] - gn)))
    margins["val_none_vs_1_bitwise"] = bool(vals[1] == v_none)
    return margins


# ---------------------------------------------------------------------------
# Cases.
# ---------------------------------------------------------------------------

def test_k1_conditional_block_invariant():
    cat = L._dark_catalog(logq_members=L._members_table())
    _assert_block_invariant(
        _k1_builder(cat, cat, _GW_PE7, _N_EV7), float(L.POP[0]), "k1_conditional")


def test_k1_conditional_batched_selection_block_invariant():
    """Selection batching uses a SEPARATE scan (sel_batch_size); pe_event_block
    must be independent of it."""
    cat = L._dark_catalog(logq_members=L._members_table())
    _assert_block_invariant(
        _k1_builder(cat, cat, _GW_PE7, _N_EV7, sel_batch_size=100),
        float(L.POP[0]), "k1_batched_sel")


def test_k1_marked_grad_through_completion_block_invariant():
    """MARKED ensemble; grad w.r.t. eta runs the reverse pass THROUGH the member
    completion (which reads the block-assembled pe_pre brackets)."""
    cat = MK._dark_catalog(logq_members=MK._members_table())
    gw_pe7 = MK._gw(_N_EV7, MK._N_SAMP, seed=0)

    def build(pe_block):
        def f(eta0):
            return darksiren_log_likelihood(
                MK.COSMO, MK.SURVEY, MK.POP, gw_pe7, cat, MK._GW_SEL, cat,
                _N_EV7, MK._N_SAMP, float(MK._N_SEL),
                pop_model="powerlaw+peak", universe_model="dark_sirens",
                sel_batch_size=None, lss_marginalize=True,
                lss_member_impl="factored", pe_event_block=pe_block,
                mark_model="loglinear", mark_params=jnp.asarray([eta0]),
                mark_names=MK._MARK_NAMES,
                materialize_redshift_prior_state=False,
            )
        return f
    _assert_block_invariant(build, 1.3, "k1_marked")


def test_k2_conditional_block_invariant():
    """K=2 mixture: the block plan carries a tuple-of-tuples pe_pre pytree
    (A_obs/idx/t/pixk per catalog) through the scan + concatenate."""
    cat_a = L._dark_catalog(logq_members=L._members_table())
    cat_b = L._dark_catalog(logq_members=0.4 * L._members_table())
    log_w = np.log(np.array([0.6, 0.4]))
    _assert_block_invariant(
        _k2_builder(cat_a, cat_b, _GW_PE7, _N_EV7, log_w), float(log_w[1]),
        "k2_conditional")


if __name__ == "__main__":  # pragma: no cover - manual margin dump
    cat = L._dark_catalog(logq_members=L._members_table())
    print(_assert_block_invariant(
        _k1_builder(cat, cat, _GW_PE7, _N_EV7), float(L.POP[0]), "k1"))
