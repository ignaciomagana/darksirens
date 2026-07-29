"""
test_flow_loader.py
-------------------
Checkpoint round-trip, structural version-drift guard, architecture grouping,
and batched ensemble evaluation for darksirens.gw.flows.

Builds tiny spline flows in the same npz format the trained event flows use
(np.savez of eqx.partition leaves + config_json), so the round-trip is tested
against the real serialization contract without shipping binary fixtures.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

flows_mod = pytest.importorskip("darksirens.gw.flows")

import equinox as eqx  # noqa: E402
import jax.tree_util as jtu  # noqa: E402


def _config(key, layers=2, knots=4):
    return {
        "base_dist": "Normal",
        "data_dim": 4,
        "type": "spline",
        "flow_layers": layers,
        "knots": knots,
        "key": key,
        "columns": list(flows_mod.SPECTRAL_COLUMNS),
        "Z_mean": [2.0, 3.5, 7.0, 0.0],
        "Z_std": [0.8, 0.2, 0.3, 0.2],
        "constraints": {
            "0": {"type": "ordered_positive", "dims": [0, 1]},
            "1": None,
            "2": {"type": "positive"},
            "3": {"type": "interval", "low": -1, "high": 1},
        },
    }


def _save_flow(path, flow, config):
    arrays, _ = eqx.partition(flow, eqx.is_array)
    leaves, _ = jtu.tree_flatten(arrays)
    np.savez(path, *[np.asarray(l) for l in leaves], config_json=json.dumps(config))


@pytest.fixture(scope="module")
def flows_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("flows")
    # Two flows share an architecture, one differs -> expect 2 groups.
    for name, cfg in [
        ("GW_TOY_A", _config(0, layers=2, knots=4)),
        ("GW_TOY_B", _config(1, layers=2, knots=4)),
        ("GW_TOY_C", _config(2, layers=3, knots=5)),
    ]:
        d = root / name
        d.mkdir()
        flow = flows_mod.create_flow_from_config(cfg)
        _save_flow(d / f"{name}_flow.npz", flow, cfg)
    # __MACOSX junk that must be ignored.
    junk = root / "__MACOSX" / "GW_TOY_A"
    junk.mkdir(parents=True)
    (junk / "GW_TOY_A_flow.npz").write_bytes(b"not-a-real-npz")
    return root


def test_load_ensemble_groups_and_names(flows_dir):
    ens = flows_mod.load_flow_ensemble(flows_dir)
    assert ens.names == ["GW_TOY_A", "GW_TOY_B", "GW_TOY_C"]
    assert ens.data_dim == 4
    assert ens.columns == flows_mod.SPECTRAL_COLUMNS
    assert len(ens.groups) == 2
    sizes = sorted(g.size for g in ens.groups)
    assert sizes == [1, 2]
    assert "architecture group" in ens.summary()


def test_ensemble_log_prob_matches_per_flow_loop(flows_dir):
    ens = flows_mod.load_flow_ensemble(flows_dir)
    eval_logflows = flows_mod.make_ensemble_log_prob(ens)

    rng = np.random.default_rng(1)
    X = jnp.asarray(
        np.column_stack(
            [
                rng.uniform(20, 60, 128),   # m1det
                rng.uniform(5, 19, 128),    # m2det < m1det
                rng.uniform(200, 3000, 128),
                rng.uniform(-0.9, 0.9, 128),
            ]
        )
    )
    got = eval_logflows(ens.group_params(), X)
    assert got.shape == (3, 128)

    # Reference: plain per-flow loop through load_flow.
    for i, p in enumerate(ens.paths):
        flow, _ = flows_mod.load_flow(p)
        ref = flow.log_prob(X)
        np.testing.assert_allclose(np.asarray(got[i]), np.asarray(ref), atol=1e-10)

    # And it must trace inside jit with the params as operands.
    jitted = jax.jit(eval_logflows)
    got_jit = jitted(ens.group_params(), X)
    np.testing.assert_allclose(np.asarray(got_jit), np.asarray(got), atol=1e-12)


def test_ensemble_sample_shapes_and_support(flows_dir):
    ens = flows_mod.load_flow_ensemble(flows_dir)
    s = np.asarray(flows_mod.ensemble_sample(ens, jax.random.key(0), 64))
    assert s.shape == (3, 64, 4)
    assert (s[..., 0] > s[..., 1]).all()          # ordered_positive: m1 > m2
    assert (s[..., 1] > 0).all()
    assert (s[..., 2] > 0).all()                  # positive dL
    assert (np.abs(s[..., 3]) <= 1).all()         # interval chi_eff


class _PickleBomb:
    """Payload whose *unpickling* deletes a canary file.

    ``__reduce__`` is evaluated when the object is written (harmless); the
    returned ``os.remove`` call only fires if something unpickles it.  The
    canary surviving a rejected load is the assertion that the loaders never
    reach numpy's object path.
    """

    def __init__(self, canary):
        self.canary = str(canary)

    def __reduce__(self):
        return (os.remove, (self.canary,))


def test_loader_refuses_pickled_checkpoint_entries(flows_dir, tmp_path):
    src = flows_dir / "GW_TOY_A" / "GW_TOY_A_flow.npz"
    with np.load(src, allow_pickle=False) as data:
        config = json.loads(str(data["config_json"]))
        n = sum(1 for k in data.files if k.startswith("arr_"))
        leaves = [data[f"arr_{i}"] for i in range(n)]

    canary = tmp_path / "canary.txt"
    canary.write_text("the payload never ran")

    # (a) config_json as a pickled object.
    bad_cfg = tmp_path / "bomb_config.npz"
    np.savez(bad_cfg, *leaves,
             config_json=np.array(_PickleBomb(canary), dtype=object))
    with pytest.raises(ValueError, match="pickled object"):
        flows_mod.load_flow(bad_cfg)
    with pytest.raises(ValueError, match="pickled object"):
        flows_mod.check_checkpoint_matches_skeleton(bad_cfg)

    # (b) a pickled object as an array leaf, with an honest config.
    bad_leaf = tmp_path / "bomb_leaf.npz"
    np.savez(bad_leaf, np.array(_PickleBomb(canary), dtype=object), *leaves[1:],
             config_json=json.dumps(config))
    with pytest.raises(ValueError, match="pickled object"):
        flows_mod.load_flow(bad_leaf)
    with pytest.raises(ValueError, match="pickled object"):
        flows_mod.check_checkpoint_matches_skeleton(bad_leaf)

    assert canary.exists(), "the checkpoint payload was unpickled"


def test_loader_rejects_non_string_config(flows_dir, tmp_path):
    src = flows_dir / "GW_TOY_A" / "GW_TOY_A_flow.npz"
    with np.load(src, allow_pickle=False) as data:
        n = sum(1 for k in data.files if k.startswith("arr_"))
        leaves = [data[f"arr_{i}"] for i in range(n)]
    bad = tmp_path / "numeric_config.npz"
    np.savez(bad, *leaves, config_json=np.array(3.0))
    with pytest.raises(ValueError, match="non-string config"):
        flows_mod.load_flow(bad)


def test_structural_check_rejects_drifted_checkpoint(flows_dir, tmp_path):
    src = flows_dir / "GW_TOY_A" / "GW_TOY_A_flow.npz"
    with np.load(src, allow_pickle=False) as data:
        config = json.loads(str(data["config_json"]))
        n = sum(1 for k in data.files if k.startswith("arr_"))
        leaves = [data[f"arr_{i}"] for i in range(n)]
    # Corrupt one leaf's shape (simulates flowjax version drift).
    leaves[3] = leaves[3][..., :-1]
    bad_dir = tmp_path / "GW_TOY_BAD"
    bad_dir.mkdir()
    bad = bad_dir / "GW_TOY_BAD_flow.npz"
    np.savez(bad, *leaves, config_json=json.dumps(config))

    with pytest.raises(flows_mod.CheckpointStructureError):
        flows_mod.check_checkpoint_matches_skeleton(bad)
    with pytest.raises(flows_mod.CheckpointStructureError):
        flows_mod.load_flow_ensemble(tmp_path, on_mismatch="error")
    # skip mode: nothing loadable left -> still CheckpointStructureError,
    # but with a healthy sibling it loads the rest.
    good_dir = tmp_path / "GW_TOY_GOOD"
    good_dir.mkdir()
    cfg = _config(7)
    _save_flow(good_dir / "GW_TOY_GOOD_flow.npz",
               flows_mod.create_flow_from_config(cfg), cfg)
    with pytest.warns(UserWarning, match="Skipped 1/2"):
        ens = flows_mod.load_flow_ensemble(tmp_path, on_mismatch="skip")
    assert ens.names == ["GW_TOY_GOOD"]
    assert len(ens.skipped) == 1


def test_unsupported_layouts_rejected(tmp_path):
    cfg = _config(0)
    cfg["columns"] = list(flows_mod.DARK_COLUMNS)
    d = tmp_path / "GW_DARK"
    d.mkdir()
    # data_dim stays 4 (config lies about columns) — the columns gate fires
    # before any shape use, which is the point of the scaffold.
    _save_flow(d / "GW_DARK_flow.npz", flows_mod.create_flow_from_config(cfg), cfg)
    with pytest.raises(NotImplementedError, match="dark-siren"):
        flows_mod.load_flow_ensemble(tmp_path)

    cfg2 = _config(1)
    cfg2["columns"] = ["mass_1", "mass_2"]
    d2 = tmp_path / "GW_WEIRD"
    d2.mkdir()
    _save_flow(d2 / "GW_WEIRD_flow.npz", flows_mod.create_flow_from_config(cfg2), cfg2)
    (d / "GW_DARK_flow.npz").unlink()
    with pytest.raises(ValueError, match="column layout"):
        flows_mod.load_flow_ensemble(tmp_path)


def test_data_dim_must_match_column_count(tmp_path):
    """A 5-D checkpoint still advertising the 4-column spectral layout.

    ``data_dim`` and ``columns`` each validated on their own, so only the PAIR
    exposes the mislabelled config -- and the likelihood builds its design
    matrix from ``columns`` while the flow consumes ``data_dim``.
    """
    cfg = _config(11)
    cfg["data_dim"] = 5
    cfg["Z_mean"] = [2.0, 3.5, 7.0, 0.0, 0.0]
    cfg["Z_std"] = [0.8, 0.2, 0.3, 0.2, 1.0]
    cfg["constraints"]["4"] = {"type": "real"}
    assert len(cfg["columns"]) == 4
    d = tmp_path / "GW_WIDE"
    d.mkdir()
    _save_flow(d / "GW_WIDE_flow.npz", flows_mod.create_flow_from_config(cfg), cfg)
    with pytest.raises(ValueError, match=r"data_dim=5 but lists 4 columns"):
        flows_mod.load_flow_ensemble(tmp_path)


def test_ordered_positive_inverse_rejects_reversed_and_equal():
    """Outside the open cone x[0] > ... > x[n-1] > 0 the point has no preimage.

    Flooring the differences to 1e-10 handed those points a FINITE z and a
    finite log|det|, i.e. a normalised-looking density where there is none.
    NaN is the convention flowjax's AbstractTransformed._log_prob consumes
    ("if log_prob is nan, we assume outside transform support") and what the
    Exp/Sigmoid constraints beside it already produce.
    """
    op = flows_mod.OrderedPositiveN(3)
    x = jnp.asarray([30.0, 12.0, 4.0])
    z, log_det = op.inverse_and_log_det(x)
    assert np.all(np.isfinite(np.asarray(z)))
    assert np.isfinite(float(log_det))
    # Still an exact inverse of the forward map on the support.
    x_back, log_det_fwd = op.transform_and_log_det(z)
    np.testing.assert_allclose(np.asarray(x_back), np.asarray(x), rtol=1e-12)
    assert float(log_det_fwd) == pytest.approx(-float(log_det), rel=1e-12)

    for bad in ([12.0, 30.0, 4.0], [12.0, 12.0, 4.0], [30.0, 12.0, 12.0],
                [30.0, 12.0, 0.0], [30.0, 12.0, -4.0], [30.0, -12.0, -40.0]):
        z_bad, ld_bad = op.inverse_and_log_det(jnp.asarray(bad))
        assert np.all(np.isnan(np.asarray(z_bad))), bad
        assert np.isnan(float(ld_bad)), bad


def test_out_of_support_masses_have_minus_inf_log_prob(flows_dir):
    ens = flows_mod.load_flow_ensemble(flows_dir)
    flow, _ = flows_mod.load_flow(ens.paths[0])
    X = jnp.asarray([
        [40.0, 20.0, 1000.0, 0.1],    # in support
        [20.0, 40.0, 1000.0, 0.1],    # m2 > m1: reversed
        [30.0, 30.0, 1000.0, 0.1],    # m2 == m1: on the boundary
        [40.0, -5.0, 1000.0, 0.1],    # m2 < 0
    ])
    lp = np.asarray(flow.log_prob(X))
    assert np.isfinite(lp[0])
    assert (lp[1:] == -np.inf).all()

    # The ensemble kernel (vmapped, jitted) agrees.
    eval_logflows = jax.jit(flows_mod.make_ensemble_log_prob(ens))
    lp_ens = np.asarray(eval_logflows(ens.group_params(), X))
    assert np.isfinite(lp_ens[:, 0]).all()
    assert (lp_ens[:, 1:] == -np.inf).all()

    # An in-support gradient must stay finite: the rejected branch injects its
    # NaN as a constant, so no NaN reaches the differentiable path.
    g = jax.grad(lambda m2: flow.log_prob(jnp.asarray([40.0, m2, 1000.0, 0.1])))
    assert np.isfinite(float(g(jnp.asarray(20.0))))


def test_unsupported_architecture_rejected():
    with pytest.raises(NotImplementedError):
        flows_mod.create_flow_from_config(
            {"base_dist": "Normal", "type": "MAF", "data_dim": 4, "key": 0}
        )
    with pytest.raises(NotImplementedError):
        flows_mod.create_flow_from_config(
            {"base_dist": "StudentT", "type": "spline", "data_dim": 4, "key": 0}
        )
