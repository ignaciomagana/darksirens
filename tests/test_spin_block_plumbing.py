"""GWEvent.spin block plumbing (DS-07).

The chi_eff basis is d = 0: ``spin = None``, and the pytree structure --
hence every compiled likelihood -- is identical to a build without the
field.  A basis with extra spin coordinates carries ONE (N, d) block,
padded structurally and passed to the weight function as a trailing
``spin=`` keyword only when present, so 7-argument weight functions (and
test doubles) never see it.
"""
import sys
import types

_tqdm_stub = types.ModuleType("tqdm")
_tqdm_stub.tqdm = lambda iterable=None, *args, **kwargs: iterable
sys.modules.setdefault("tqdm", _tqdm_stub)

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from darksirens.likelihood.events import make_gw_event, pad_gw_event_to_multiple


def _event(n=10, d=None, seed=0):
    # Spin is drawn LAST so the base arrays are identical for d=None vs d>0
    # (the numerical-equivalence tests below rely on that).
    rng = np.random.default_rng(seed)
    spin_rng = np.random.default_rng(seed + 1000)
    spin = spin_rng.uniform(0.0, 0.9, size=(n, d)) if d else None
    return make_gw_event(
        m1det=rng.uniform(20.0, 60.0, n),
        m2det=rng.uniform(10.0, 20.0, n),
        dL=rng.uniform(300.0, 3000.0, n),
        chieff=rng.uniform(-0.5, 0.5, n),
        prior_wt=rng.uniform(0.5, 2.0, n),
        pixels=rng.integers(0, 12, n),
        spin=spin,
    )


def test_default_spin_is_none_and_pytree_structure_unchanged():
    ev = _event()
    assert ev.spin is None
    # None is an empty pytree subtree: leaf count identical to a spin-less
    # event, which is what keeps existing compiled functions byte-identical.
    leaves = jax.tree_util.tree_leaves(ev)
    assert len(leaves) == 11  # m1det m2det dL chieff prior_wt pixels q valid nx ny nz


def test_spin_block_carried_and_barriered():
    ev = _event(d=4)
    assert ev.spin.shape == (10, 4)
    assert ev.spin.dtype == jnp.float64


def test_spin_shape_validation():
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        make_gw_event(
            m1det=np.ones(4), m2det=np.ones(4), dL=np.ones(4),
            chieff=np.zeros(4), prior_wt=np.ones(4), pixels=np.zeros(4),
            spin=np.ones(4),  # 1-D: ambiguous, refused
        )
    with pytest.raises(ValueError, match=r"\(N, d\)"):
        make_gw_event(
            m1det=np.ones(4), m2det=np.ones(4), dL=np.ones(4),
            chieff=np.zeros(4), prior_wt=np.ones(4), pixels=np.zeros(4),
            spin=np.ones((5, 2)),  # wrong N
        )


def test_padding_pads_spin_block_and_masks_it():
    ev = _event(n=10, d=3)
    padded, pad = pad_gw_event_to_multiple(ev, 8)
    assert pad == 6
    assert padded.spin.shape == (16, 3)
    np.testing.assert_array_equal(np.asarray(padded.spin[:10]), np.asarray(ev.spin))
    assert np.all(np.asarray(padded.spin[10:]) == 0.0)
    assert not np.any(np.asarray(padded.valid[10:]))
    # d = 0 events keep spin=None through padding.
    padded0, _ = pad_gw_event_to_multiple(_event(n=10), 8)
    assert padded0.spin is None


def test_selection_term_passes_spin_only_when_present():
    from darksirens.likelihood.selection import compute_selection_term

    seen = {"spin": "unset", "legacy_calls": 0}

    def weight_no_spin(m1det, q, dL, chi, pix, pwt, cat):
        seen["legacy_calls"] += 1
        return -jnp.log(pwt)

    def weight_with_spin(m1det, q, dL, chi, pix, pwt, cat, spin=None):
        seen["spin"] = spin
        base = -jnp.log(pwt)
        if spin is not None:
            base = base + 0.0 * spin.sum(axis=-1)
        return base

    ev0 = _event(n=12)
    log_mu0, neff0, _ = compute_selection_term(
        ev0, None, weight_no_spin, Ndraw=100, nEvents=3)
    assert seen["legacy_calls"] == 1

    ev = _event(n=12, d=2)
    log_mu, neff, _ = compute_selection_term(
        ev, None, weight_with_spin, Ndraw=100, nEvents=3)
    assert np.asarray(seen["spin"]).shape == (12, 2)
    np.testing.assert_allclose(np.asarray(log_mu), np.asarray(log_mu0))

    # Batched path slices the spin block per scan chunk.
    log_mu_b, neff_b, _ = compute_selection_term(
        ev, None, weight_with_spin, Ndraw=100, nEvents=3, sel_batch_size=5)
    # Inside lax.scan the captured value is a tracer; its static shape is the
    # per-batch slice of the spin block.
    assert tuple(seen["spin"].shape) == (5, 2)
    np.testing.assert_allclose(np.asarray(log_mu_b), np.asarray(log_mu),
                               rtol=1e-12)


def test_target_density_forwards_spin_kwarg_only_when_present():
    from darksirens.core.types import CosmoParams
    from darksirens.inference.utils import log_sample_weight

    calls = {}

    def pop_no_spin(m1src, q, z, chieff, pop_params):
        calls["no_spin"] = True
        return jnp.zeros_like(m1src)

    def pop_with_spin(m1src, q, z, chieff, pop_params, spin=None):
        calls["spin"] = spin
        return jnp.zeros_like(m1src)

    cosmo = CosmoParams(H0=70.0, Om0=0.3, w0=-1.0, wa=0.0)
    args = dict(
        m1det=jnp.array([30.0]), q=jnp.array([0.8]), dL=jnp.array([1000.0]),
        chieff=jnp.array([0.1]), pix=jnp.array([0]),
        prior_wt=jnp.array([1.0]), cosmo=cosmo, survey=None,
        pop_params=jnp.zeros(1), catalog=None,
        log_prior_z_fn=lambda z, pix, cat: jnp.zeros_like(z),
    )
    log_sample_weight(log_p_pop_fn=pop_no_spin, **args)
    assert calls.get("no_spin")

    spin = jnp.ones((1, 4))
    log_sample_weight(log_p_pop_fn=pop_with_spin, spin=spin, **args)
    assert calls["spin"] is spin
