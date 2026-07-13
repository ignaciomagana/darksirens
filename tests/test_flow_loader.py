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


def test_structural_check_rejects_drifted_checkpoint(flows_dir, tmp_path):
    src = flows_dir / "GW_TOY_A" / "GW_TOY_A_flow.npz"
    with np.load(src, allow_pickle=True) as data:
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


def test_unsupported_architecture_rejected():
    with pytest.raises(NotImplementedError):
        flows_mod.create_flow_from_config(
            {"base_dist": "Normal", "type": "MAF", "data_dim": 4, "key": 0}
        )
    with pytest.raises(NotImplementedError):
        flows_mod.create_flow_from_config(
            {"base_dist": "StudentT", "type": "spline", "data_dim": 4, "key": 0}
        )
