"""b_GW containment in the anchor's ``b_nodes`` (latent mode).

``rho_from_moments`` interpolates the eq. (2) sky moments in ``b`` by a global
barycentric Chebyshev-Lobatto polynomial that is exact ON the node interval
and EXTRAPOLATES off it with ~|T_{n-1}|-sized amplification -- so a sampled
``b_miss`` (= b_GW) prior wider than the anchor's ``--b-max``, or a narrow
anchor under the default [0, 3] prior, silently corrupts rho on exactly the
proposals the sampler visits (``A - c B`` goes negative and the 1e-300 floor
makes the garbage finite).  Two walls, both pinned here:

* build time: ``_latent_guard_b_range`` refuses a run whose resolved b_miss
  range (default prior, ``--prior_overrides``, or a fixed value) is not
  contained in ``[b_nodes[0], b_nodes[-1]]``;
* in kernel: ``rho_from_moments`` returns NaN (never a clamp, never a finite
  extrapolation) for ``b`` outside the node interval, so any future unguarded
  caller fails loudly through the finite guards instead of silently.

Fixture helpers are shared with tests/test_latent_factory.py (same tiny
nside = 4 anchor; the sibling-module import follows the
test_latent_solve_damping precedent).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("jax")
import h5py
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from darksirens.likelihood.factory import _resolve_latent_leaves
from darksirens.likelihood.latent_q import rho_from_moments
from test_latent_factory import (
    NSIDE,
    ROW_PIXELS,
    Z_DEPTH,
    _catalogs,
    _opts,
    _write_artifact,
)

N_ROWS = int(ROW_PIXELS.size)


# ------------------------------------------------------------------- kernel

def _tiny_moments(n_b=5, n_grid=7, b_hi=2.0):
    """Positive (n_b, n_grid) A/B moment stand-ins on nodes [0, b_hi]."""
    b_nodes = 0.5 * b_hi * (1.0 - np.cos(np.pi * np.arange(n_b) / (n_b - 1)))
    f = np.linspace(0.2, 0.8, n_grid)
    A = 8.0 * np.exp(b_nodes[:, None] * f[None, :])
    B = 0.75 * A
    return jnp.asarray(A), jnp.asarray(B), jnp.asarray(b_nodes)


def test_rho_is_nan_outside_the_node_interval():
    A, B, b_nodes = _tiny_moments()
    c = jnp.full(7, 0.5)
    kw = dict(c=c, b_nodes=b_nodes, P_F=8.0, F_F=6.0)

    # Inside (and at both closed endpoints): finite.
    for b in (0.0, 1.0, 2.0):
        rho = np.asarray(rho_from_moments(A, B, b=b, **kw))
        assert np.all(np.isfinite(rho)), f"rho not finite at in-range b={b}"

    # Outside, either side: NaN, never a silently floored finite value.
    for b in (-0.1, 2.5):
        rho = np.asarray(rho_from_moments(A, B, b=b, **kw))
        assert np.all(np.isnan(rho)), f"rho extrapolated silently at b={b}"


def test_rho_above_depth_stays_bit_zero_even_at_out_of_range_b():
    """Pin P13b's above-depth zero survives the NaN wall (mask applies last)."""
    A, B, b_nodes = _tiny_moments()
    below = jnp.asarray(np.array([True] * 5 + [False] * 2))
    rho = np.asarray(rho_from_moments(
        A, B, c=jnp.full(7, 0.5), b=2.5, b_nodes=b_nodes,
        P_F=8.0, F_F=6.0, below_depth=below))
    assert np.all(np.isnan(rho[:5]))
    assert np.all(rho[5:] == 0.0)


# ------------------------------------------------------------ build-time wall

def _resolve(opts, **kw):
    return _resolve_latent_leaves(
        opts, _catalogs(), Z_DEPTH, NSIDE, N_ROWS, N_ROWS, **kw)


def _narrow_anchor(tmp_path, b_hi=2.0):
    """The default fixture anchor with its b_nodes rewritten to [0, b_hi].

    b_nodes is not a guard-1 fingerprint ingredient (the digest identifies the
    FIELD, not the moment build), so the rewrite is exactly the freely-settable
    ``--b-max`` this guard exists to wall.
    """
    art = _write_artifact(tmp_path / "anchor.h5")
    with h5py.File(art, "r+") as f:
        g = f["latent_field"]
        n_b = int(g["b_nodes"].shape[0])
        del g["b_nodes"]
        g.create_dataset("b_nodes", data=np.linspace(0.0, b_hi, n_b))
    return art


def test_default_prior_inside_default_nodes_resolves(tmp_path):
    art = _write_artifact(tmp_path / "anchor.h5")   # b_nodes [0, 4] >= [0, 3]
    mode, pe, sel = _resolve(_opts(lss_field_artifact=art))
    assert mode == "latent" and pe


def test_prior_override_wider_than_nodes_is_refused(tmp_path):
    art = _write_artifact(tmp_path / "anchor.h5")
    with pytest.raises(ValueError, match="b_nodes span") as exc:
        _resolve(_opts(lss_field_artifact=art,
                       prior_overrides={"b_miss": [0.0, 5.0]}))
    assert "--b-max" in str(exc.value)              # the remedy is named


def test_narrow_anchor_under_the_default_prior_is_refused(tmp_path):
    """The headline case: --b-max 2 anchor, untouched [0, 3] default prior."""
    art = _narrow_anchor(tmp_path, b_hi=2.0)
    with pytest.raises(ValueError, match="b_nodes span"):
        _resolve(_opts(lss_field_artifact=art))


def test_fixed_b_miss_is_the_resolved_range(tmp_path):
    art = _write_artifact(tmp_path / "anchor.h5")   # b_nodes [0, 4]
    # A pin above the default prior but inside the nodes: legal (the fixed
    # value replaces the sampled range, per validate_fixed_parameter_overrides
    # this is a legitimate ablation device).
    mode, _, _ = _resolve(_opts(lss_field_artifact=art),
                          fixed_parameter_values={"b_miss": 3.5})
    assert mode == "latent"
    # A pin past the last node: refused.
    with pytest.raises(ValueError, match="fixed_parameter_values"):
        _resolve(_opts(lss_field_artifact=art),
                 fixed_parameter_values={"b_miss": 4.5})


def test_opts_carried_fixed_values_reach_the_guard(tmp_path):
    """Non-CLI callers stash fixed values on opts; the fallback must see them."""
    art = _write_artifact(tmp_path / "anchor.h5")
    with pytest.raises(ValueError, match="b_nodes span"):
        _resolve(_opts(lss_field_artifact=art,
                       fixed_parameter_values={"b_miss": 4.5}))
