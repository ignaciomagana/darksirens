"""
test_pdet_selection.py
----------------------
P_det emulator selection path (darksirens.gw.selection): vendored GWTC-4
injection-prior factors, checkpoint loading/validation, the
(m1src, m2src, z) -> (m1det, q, dL) Jacobian, pseudo-injection generation
invariants, and the Ndraw bookkeeping pin through compute_selection_term.

Mirrors test_flow_loader.py: tiny synthetic 13-dim checkpoints are built in
the real npz serialization format (np.savez of eqx.partition leaves +
config_json), so the loader contract is tested without binary fixtures.
The golden parity test against the shipped PDetO4NF.npz (constants generated
from the gw-nf reference implementation) is skipped when the checkpoint is
absent.
"""

import json
import math
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

from darksirens.gw import selection as sel  # noqa: E402

PDET_NPZ = PKG_ROOT / "PDetO4NF.npz"


# ── tiny checkpoint fixtures ────────────────────────────────────────────────


# The legacy transformations layout of the shipped PDetO4NF checkpoint.
_TRANSFORMATIONS = [
    "log", "log", "angle_pi", "angle_pi", "angle_pi", "angle_2pi",
    "angle_signed_pi", "angle_pi", "angle_signed_pi", "angle_signed_pi",
    "angle_signed_pi", "angle_2pi", "angle_2pi",
]


def _pdet_config(key=0, layers=2, knots=4, **overrides):
    """A miniature emulator config mirroring the PDetO4NF legacy schema:
    bare 13-dim spline flow leaves + transformations list + xi."""
    cfg = {
        "base_dist": "Normal",
        "data_dim": 13,
        "type": "spline",
        "flow_layers": layers,
        "knots": knots,
        "key": key,
        "columns": list(sel.PDET_COLUMNS),
        "transformations": list(_TRANSFORMATIONS),
        "xi": 0.00112459,
    }
    cfg.update(overrides)
    return cfg


def _save_flow(path, flow, config):
    arrays, _ = eqx.partition(flow, eqx.is_array)
    leaves, _ = jtu.tree_flatten(arrays)
    np.savez(path, *[np.asarray(l) for l in leaves], config_json=json.dumps(config))


@pytest.fixture(scope="module")
def tiny_pdet_npz(tmp_path_factory):
    root = tmp_path_factory.mktemp("pdet")
    cfg = _pdet_config(key=3)
    flow = flows_mod.create_flow_from_config(cfg)  # bare skeleton, like the real npz
    path = root / "tiny_pdet.npz"
    _save_flow(path, flow, cfg)
    return path


# ── loader: round-trip, drift guard, rejections ─────────────────────────────


def test_load_pdet_flow_roundtrip(tiny_pdet_npz):
    flow, config = sel.load_pdet_flow(tiny_pdet_npz)
    assert tuple(config["columns"]) == sel.PDET_COLUMNS
    assert float(config["xi"]) == pytest.approx(0.00112459)
    s = np.asarray(flow.sample(jax.random.key(0), (256,)))
    assert s.shape == (256, 13)
    # The wrapper puts samples in physical coordinates: fixed-range boxes.
    assert (s[:, 0] > 0).all() and (s[:, 1] > 0).all()          # log cols
    for i in (2, 3, 4, 7):                                       # angle_pi
        assert (s[:, i] > 0).all() and (s[:, i] < np.pi).all()
    for i in (5, 11, 12):                                        # angle_2pi
        assert (s[:, i] > 0).all() and (s[:, i] < 2 * np.pi).all()
    for i in (6, 8, 9, 10):                                      # signed_pi
        assert (np.abs(s[:, i]) < np.pi).all()
    # Round-trip fidelity: log_prob matches an independently rebuilt +
    # wrapped copy of the same leaves.
    import paramax
    from flowjax.distributions import Transformed

    orig = flows_mod.create_flow_from_config(_pdet_config(key=3))
    with np.load(tiny_pdet_npz, allow_pickle=True) as data:
        n = sum(1 for k in data.files if k.startswith("arr_"))
        leaves = [jnp.asarray(data[f"arr_{i}"]) for i in range(n)]
    arrays, static = eqx.partition(orig, eqx.is_array)
    _, treedef = jtu.tree_flatten(arrays)
    rebuilt = eqx.combine(jtu.tree_unflatten(treedef, leaves), static)
    rebuilt = Transformed(
        rebuilt,
        paramax.non_trainable(
            sel._build_transformations_bijection(_TRANSFORMATIONS, 13)
        ),
    )
    np.testing.assert_allclose(
        np.asarray(flow.log_prob(s)), np.asarray(rebuilt.log_prob(s)), atol=1e-12
    )


def test_transformations_bijection_semantics():
    """z = 0 maps to the interval midpoints / exp(0); unknown names raise."""
    bij = sel._build_transformations_bijection(_TRANSFORMATIONS, 13)
    x, _ = bij.transform_and_log_det(jnp.zeros(13))
    x = np.asarray(x)
    expected = {
        "log": 1.0,
        "angle_pi": np.pi / 2,
        "angle_2pi": np.pi,
        "angle_signed_pi": 0.0,
    }
    for i, name in enumerate(_TRANSFORMATIONS):
        assert x[i] == pytest.approx(expected[name], abs=1e-12)
    with pytest.raises(ValueError, match="Unknown transformation"):
        sel._build_transformations_bijection(["nope"] * 13, 13)
    with pytest.raises(ValueError, match="entries"):
        sel._build_transformations_bijection(["log"] * 12, 13)


def test_load_pdet_flow_rejects_structural_drift(tiny_pdet_npz, tmp_path):
    with np.load(tiny_pdet_npz, allow_pickle=True) as data:
        config = json.loads(str(data["config_json"]))
        n = sum(1 for k in data.files if k.startswith("arr_"))
        leaves = [data[f"arr_{i}"] for i in range(n)]
    leaves[2] = leaves[2][..., :-1]  # simulated flowjax version drift
    bad = tmp_path / "drifted.npz"
    np.savez(bad, *leaves, config_json=json.dumps(config))
    with pytest.raises(flows_mod.CheckpointStructureError):
        sel.load_pdet_flow(bad)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"columns": list(sel.PDET_COLUMNS[::-1])}, "columns"),
        ({"data_dim": 12}, "data_dim"),
        ({"xi": None}, "xi"),
        ({"xi": 1.5}, "xi="),
        ({"transformations": None}, "transformations"),
    ],
)
def test_load_pdet_flow_rejects_bad_config(tmp_path, overrides, match):
    cfg = _pdet_config(key=1)
    for key, val in list(overrides.items()):
        if val is None:
            del cfg[key]
            del overrides[key]
    cfg.update(overrides)
    if "data_dim" in overrides:
        # Keep columns consistent so the data_dim gate itself fires.
        cfg["columns"] = list(sel.PDET_COLUMNS)
    # Skeleton needs a *valid* dim to build; save leaves from the honest
    # config so only the config metadata is wrong.
    flow = flows_mod.create_flow_from_config(_pdet_config(key=1))
    path = tmp_path / "bad.npz"
    _save_flow(path, flow, cfg)
    with pytest.raises(ValueError, match=match):
        sel.load_pdet_flow(path)


def test_load_pdet_flow_rejects_newer_schema(tmp_path):
    cfg = _pdet_config(key=2, constraints={"0": {"type": "positive"}})
    flow = flows_mod.create_flow_from_config(_pdet_config(key=2))
    path = tmp_path / "newer.npz"
    _save_flow(path, flow, cfg)
    with pytest.raises(NotImplementedError, match="constraints"):
        sel.load_pdet_flow(path)


def test_load_pdet_flow_requires_x64(tiny_pdet_npz):
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="x64"):
            sel.load_pdet_flow(tiny_pdet_npz)
    finally:
        jax.config.update("jax_enable_x64", True)


# ── vendored injection-prior factors ────────────────────────────────────────


def test_m1_prior_normalized_continuous_and_bounded():
    m = np.logspace(0.0, 3.0, 200001)
    p = np.exp(sel._gwtc4_log_p_m1(m))
    assert np.trapezoid(p, m) == pytest.approx(1.0, rel=1e-6)
    for b in (3.0, 8.0, 50.0, 200.0):
        lo = np.exp(sel._gwtc4_log_p_m1(np.array([b * (1 - 1e-10)])))[0]
        hi = np.exp(sel._gwtc4_log_p_m1(np.array([b * (1 + 1e-10)])))[0]
        assert lo == pytest.approx(hi, rel=1e-8)
    outside = sel._gwtc4_log_p_m1(np.array([0.5, 1000.5, -3.0]))
    assert np.all(np.isneginf(outside))
    # Spot value: on the flat first segment p(m1) = 1/Z for m1 in [1, 3).
    Z = math.exp(sel._M1_LOG_NORM)
    assert np.exp(sel._gwtc4_log_p_m1(np.array([2.0])))[0] == pytest.approx(
        1.0 / Z, rel=1e-12
    )


def test_m2_prior_normalized_and_spot_value():
    m1 = 30.0
    m2 = np.linspace(1.0, m1, 100001)
    p = np.exp(sel._gwtc4_log_p_m2_given_m1(m2, np.full_like(m2, m1)))
    assert np.trapezoid(p, m2) == pytest.approx(1.0, rel=1e-6)
    # p(m2=2 | m1=10) = 2*2/(100-1) = 4/99
    got = np.exp(sel._gwtc4_log_p_m2_given_m1(np.array([2.0]), np.array([10.0])))
    assert got[0] == pytest.approx(4.0 / 99.0, rel=1e-12)
    assert np.isneginf(
        sel._gwtc4_log_p_m2_given_m1(np.array([11.0]), np.array([10.0]))
    )[0]


def test_spin_and_tilt_priors():
    a = np.linspace(0.0, 1.0, 100001)
    assert np.trapezoid(np.exp(sel._gwtc4_log_p_spin_mag(a)), a) == pytest.approx(
        1.0, rel=1e-6
    )
    # p(a=0) = 1/Z_a with Z_a = sqrt(pi)/(2 sqrt2) erf(sqrt2)
    Z_a = math.sqrt(math.pi) / (2.0 * math.sqrt(2.0)) * math.erf(math.sqrt(2.0))
    assert np.exp(sel._gwtc4_log_p_spin_mag(np.array([0.0])))[0] == pytest.approx(
        1.0 / Z_a, rel=1e-12
    )
    assert np.isneginf(sel._gwtc4_log_p_spin_mag(np.array([1.001])))[0]

    ct = np.linspace(-1.0, 1.0, 100001)
    assert np.trapezoid(np.exp(sel._gwtc4_log_p_cos_tilt(ct)), ct) == pytest.approx(
        1.0, rel=1e-6
    )
    assert np.exp(sel._gwtc4_log_p_cos_tilt(np.array([-1.0])))[0] == pytest.approx(
        0.35, rel=1e-12
    )
    assert np.exp(sel._gwtc4_log_p_cos_tilt(np.array([1.0])))[0] == pytest.approx(
        0.95, rel=1e-12
    )


def test_z_prior_normalized_and_zero_outside():
    z_grid, pdf = sel.build_injection_z_prior(n=20001)
    assert np.trapezoid(pdf, z_grid) == pytest.approx(1.0, rel=1e-8)
    logs = sel._gwtc4_log_p_z(np.array([-0.1, 3.5]), z_grid, pdf)
    assert np.all(np.isneginf(logs))


# ── Jacobian: finite-difference determinant test ────────────────────────────


def test_jacobian_matches_finite_differences():
    """|det d(m1det, q, dL)/d(m1src, m2src, z)| == (1+z) dL'(z) / m1src."""
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u

    H0, Om0 = sel.PDET_INJ_H0, sel.PDET_INJ_OM0
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)

    def forward(m1src, m2src, z):
        DC = cosmo.comoving_distance(z).to_value(u.Mpc)
        return np.array([m1src * (1 + z), m2src / m1src, (1 + z) * DC])

    for m1src, m2src, z in [(35.0, 20.0, 0.4), (8.0, 3.0, 1.5), (60.0, 55.0, 0.05)]:
        x0 = np.array([m1src, m2src, z])
        J = np.zeros((3, 3))
        for j in range(3):
            h = 1e-6 * max(abs(x0[j]), 1e-3)
            xp, xm = x0.copy(), x0.copy()
            xp[j] += h
            xm[j] -= h
            J[:, j] = (forward(*xp) - forward(*xm)) / (2 * h)
        det_fd = abs(np.linalg.det(J))

        DC = cosmo.comoving_distance(z).to_value(u.Mpc)
        E = np.sqrt(Om0 * (1 + z) ** 3 + (1 - Om0))
        dLprime = DC + (1 + z) * (299792.458 / H0) / E
        det_analytic = (1 + z) * dLprime / m1src
        assert det_fd == pytest.approx(det_analytic, rel=1e-6)


# ── pseudo-injection generation ─────────────────────────────────────────────


def _crafted_theta():
    """Hand-crafted 13-dim rows: 3 valid + one row per drop cause."""
    valid = [30.0, 20.0, 0.4, 0.3, 0.2, 1.0, 0.1, 1.0, 0.2, 0.5, -0.3, 1.0, 2.0]
    rows = [list(valid) for _ in range(3)]
    rows[1][0], rows[1][1] = 10.0, 9.0
    rows[2][2] = 1.2
    bad_a = list(valid); bad_a[3] = 0.995          # a1 > amax=0.99
    bad_m2 = list(valid); bad_m2[1] = 35.0         # m2 > m1
    bad_z = list(valid); bad_z[2] = 3.5            # z > 3
    bad_ct = list(valid); bad_ct[9] = 1.2          # |cos_tilt1| > 1
    bad_ang = list(valid); bad_ang[6] = -1.4       # |sin dec| > 1
    return np.array(rows + [bad_a, bad_m2, bad_z, bad_ct, bad_ang])


def test_generation_invariants_and_drop_accounting(tiny_pdet_npz, monkeypatch):
    theta = _crafted_theta()
    monkeypatch.setattr(
        sel, "sample_pdet_flow", lambda flow, nsamp, seed, batch=65536: theta
    )
    with pytest.warns(RuntimeWarning, match="outside the .* support"):
        (m1det, m2det, dL, chieff, ra, dec, pdraw, ndraw) = (
            sel.pseudo_injections_from_pdet_flow(
                tiny_pdet_npz, nsamp=len(theta), seed=0
            )
        )

    assert len(m1det) == 3                     # exactly the crafted valid rows
    assert ndraw == pytest.approx(len(theta) / 0.00112459)
    assert np.all(np.asarray(pdraw) > 0)
    assert np.all(np.asarray(m1det) >= np.asarray(m2det))
    assert np.all(np.asarray(dL) > 0)
    assert np.all(np.abs(np.asarray(chieff)) <= 0.99)
    assert np.all(np.abs(np.asarray(dec)) <= np.pi / 2)

    # Reference pdraw for the first crafted row, recomputed independently.
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    from gwcat.spin import chi_eff_prior_logprob

    m1s, m2s, z = 30.0, 20.0, 0.4
    a1, a2, ct1, ct2 = 0.3, 0.2, 0.5, -0.3
    cosmo = FlatLambdaCDM(H0=sel.PDET_INJ_H0, Om0=sel.PDET_INJ_OM0)
    DC = cosmo.comoving_distance(z).to_value(u.Mpc)
    E = np.sqrt(sel.PDET_INJ_OM0 * (1 + z) ** 3 + 1 - sel.PDET_INJ_OM0)
    dLp = DC + (1 + z) * (299792.458 / sel.PDET_INJ_H0) / E
    q = m2s / m1s
    chi = (a1 * ct1 + q * a2 * ct2) / (1 + q)
    z_grid, z_pdf = sel.build_injection_z_prior()
    log_ref = float(
        sel._gwtc4_log_p_m1(np.array([m1s]))[0]
        + sel._gwtc4_log_p_m2_given_m1(np.array([m2s]), np.array([m1s]))[0]
        + sel._gwtc4_log_p_z(np.array([z]), z_grid, z_pdf)[0]
        + np.log(m1s) - np.log1p(z) - np.log(dLp)
        + np.clip(chi_eff_prior_logprob(
            np.array([chi]), np.array([m1s]), np.array([m2s]), amax=0.99
        )[0], -50.0, None)
        + sel._gwtc4_log_p_spin_mag(np.array([a1]))[0]
        + sel._gwtc4_log_p_spin_mag(np.array([a2]))[0]
        + sel._gwtc4_log_p_cos_tilt(np.array([ct1]))[0]
        + sel._gwtc4_log_p_cos_tilt(np.array([ct2]))[0]
        + 2.0 * (np.log(0.99) + np.log(2.0))
    )
    assert float(pdraw[0]) == pytest.approx(math.exp(log_ref), rel=1e-10)
    # Detector-frame conversion of the same row.
    assert float(m1det[0]) == pytest.approx(m1s * (1 + z), rel=1e-12)
    assert float(dL[0]) == pytest.approx((1 + z) * DC, rel=1e-10)
    assert float(chieff[0]) == pytest.approx(chi, rel=1e-12)


@pytest.mark.skipif(not PDET_NPZ.exists(), reason="PDetO4NF.npz not present")
def test_generation_from_real_checkpoint():
    """End-to-end through the real flow.sample path (no monkeypatching).

    Uses the shipped checkpoint: an UNTRAINED tiny flow spreads its mass
    over the full fixed-range transform boxes, leaving essentially no draws
    inside the injection prior's support, so the hermetic fixtures only
    exercise the (monkeypatched) generation math above.
    """
    (m1det, m2det, dL, chieff, ra, dec, pdraw, ndraw) = (
        sel.pseudo_injections_from_pdet_flow(
            PDET_NPZ, nsamp=16384, seed=11, batch=8192
        )
    )
    assert ndraw == pytest.approx(16384 / 0.00112459)
    # The trained flow tracks the found-injection support: drops stay small.
    assert len(m1det) > 0.9 * 16384
    assert np.all(np.asarray(pdraw) > 0)
    assert np.all(np.asarray(m1det) >= np.asarray(m2det))
    assert np.all(np.asarray(dL) > 0)
    assert np.all(np.abs(np.asarray(chieff)) <= 0.99)


def test_ndraw_bookkeeping_pin(tiny_pdet_npz, monkeypatch):
    """With log_weight == log pdraw - log prior_wt == 0 per sample, the
    selection estimator must return exp(log_mu) == xi * n_kept / M exactly."""
    from darksirens.core.types import GWEvent
    from darksirens.likelihood.selection import compute_selection_term

    theta = _crafted_theta()
    monkeypatch.setattr(
        sel, "sample_pdet_flow", lambda flow, nsamp, seed, batch=65536: theta
    )
    with pytest.warns(RuntimeWarning):
        (m1det, m2det, dL, chieff, ra, dec, pdraw, ndraw) = (
            sel.pseudo_injections_from_pdet_flow(
                tiny_pdet_npz, nsamp=len(theta), seed=0
            )
        )
    n_kept = len(m1det)
    gw_sel = GWEvent(
        m1det=m1det,
        m2det=m2det,
        dL=dL,
        chieff=chieff,
        prior_wt=pdraw,
        pixels=jnp.zeros(n_kept, dtype=jnp.int32),
        q=m2det / m1det,
        valid=jnp.ones(n_kept, dtype=bool),
        nx=jnp.zeros(n_kept),
        ny=jnp.zeros(n_kept),
        nz=jnp.zeros(n_kept),
    )
    log_pdraw = jnp.log(pdraw)

    def log_weight_fn(m1det_b, q_b, dL_b, chi_b, pix_b, pwt_b, catalog):
        return log_pdraw - jnp.log(pwt_b)

    log_mu, Neff, _ = compute_selection_term(
        gw_sel, None, log_weight_fn, ndraw, nEvents=1
    )
    xi = 0.00112459
    assert float(jnp.exp(log_mu)) == pytest.approx(
        xi * n_kept / len(theta), rel=1e-12
    )
    # Neff = mu^2 / sigma^2 with sigma^2 = (s2 - mu^2/Ndraw); for unit
    # weights that is exactly n_kept / (1 - n_kept/Ndraw).
    assert float(Neff) == pytest.approx(n_kept / (1 - n_kept / ndraw), rel=1e-9)


# ── golden parity with gw-nf ────────────────────────────────────────────────

# Regression pins for the shipped PDetO4NF.npz, generated once by
# scripts/pdet_emulator_validation.py: log p_inj cross-checked against the
# gw-nf reference implementation (pdet.gwtc4_model.GWTC4Model), p_det from
# darksirens' wrapped reconstruction (gw-nf's CURRENT loader cannot rebuild
# this legacy-schema checkpoint faithfully — it drops the 'transformations'
# wrapper — so the flow factor is pinned against our validated
# reconstruction rather than upstream output).
GOLDEN_THETA = [
    [2.9604780798058318e+01, 2.7994786765056663e+01, 2.7029759427752609e-01,
     5.0095774985757269e-01, 2.4082919879122031e-01, 2.5503753817786351e-01,
     -8.8984094433097738e-02, 1.1050359133894516e+00, -5.1086850465979294e-01,
     -7.3348557188716423e-01, 7.4305593898843592e-01, 5.1688426312779328e+00,
     1.4085219517531580e+00],
    [4.7997343589034266e+01, 2.4115692587416124e+01, 8.4856915819702983e-01,
     5.4029047252352957e-01, 5.7782512783883688e-01, 5.2428961621347350e+00,
     -4.6239823409867142e-01, 5.3312607561440073e-01, 6.1439792998331955e-01,
     7.5988206928774593e-01, 4.5136302853976895e-01, 3.0846478098129566e+00,
     5.8815405613156244e+00],
    [5.4018103469974051e+01, 4.8658766390964047e+01, 4.9539079925277618e-01,
     2.8953993983378912e-01, 2.4559834471809991e-01, 3.5951653097232459e+00,
     7.3218908316213138e-02, 3.2416870403584169e-01, 3.9018182875097374e-01,
     -6.8045673554903896e-01, 1.8719143767191149e-01, 5.0114915855329585e+00,
     5.4298166712673801e+00],
    [5.8716626869758983e+01, 4.6610438047005118e+01, 1.6097479367471548e+00,
     3.8205081110662270e-01, 7.4292156184583169e-01, 6.2426219165228609e+00,
     -3.3296810214795647e-01, 2.5416561846756220e+00, -9.9216516766985352e-01,
     7.2012901253021067e-02, 9.8000035244733930e-01, 1.0870029180308600e+00,
     3.5865159884401181e+00],
    [4.2526660948565890e+01, 2.5952992110825321e+01, 7.6473822203925590e-01,
     3.2663050612878197e-01, 6.5290850725071925e-01, 4.2929588231255034e-01,
     8.0282488049542122e-01, 2.7733487915007142e+00, 8.0717221874445322e-01,
     2.2046287136181641e-01, 2.6688377142280029e-01, 2.1255411786361775e+00,
     1.7282882971442850e+00],
    [3.1750894167023748e+01, 2.0795374743332797e+01, 2.6125443182265390e-01,
     3.1897368069230009e-01, 3.8514635828364352e-01, 5.9550894074492762e+00,
     6.5214551695845291e-01, 3.0076751328168916e+00, 4.6444618565720841e-01,
     -3.7090852370106120e-01, 9.6406087701289955e-01, 3.9982992934779626e-01,
     5.2028979600649548e+00],
    [2.5285177859702440e+01, 1.5299892859450036e+01, 9.5983436385991006e-01,
     3.4101680658516575e-01, 2.4258493709898368e-01, 4.8568379898253413e+00,
     -9.5245232249831613e-02, 2.8667976111404725e+00, -7.2178247521107552e-01,
     9.9859711456804856e-01, 6.2147007432608259e-02, 3.8266780960560922e+00,
     3.8972736420589280e+00],
    [4.3148874722551255e+01, 4.0557464198835966e+01, 3.3726247510122104e-01,
     3.1458272539165627e-01, 1.8244516147300559e-02, 4.5242921116988444e-01,
     -3.9709636610878851e-01, 2.3905706141056196e+00, -9.0040727427956657e-02,
     6.0676851647625263e-03, 2.3313609766562493e-01, 3.3180636413399793e+00,
     4.7811491674376905e+00],
    [7.7522295163787271e+00, 6.6992961982663335e+00, 1.5282628262870035e-01,
     3.5724537595651357e-01, 4.3785126300493943e-02, 6.2679793359088123e+00,
     4.9154727181516300e-01, 2.5499681251487294e+00, -5.3157873170805958e-01,
     6.8054936454525938e-01, -9.4166774362229733e-01, 4.9497292198093863e+00,
     8.9636437406399971e-01],
    [4.5000000000000000e+01, 3.0000000000000000e+01, 2.0000000000000000e+00,
     2.9999999999999999e-01, 2.0000000000000001e-01, 1.0000000000000000e+00,
     1.0000000000000001e-01, 1.0000000000000000e+00, 2.0000000000000001e-01,
     5.0000000000000000e-01, -2.9999999999999999e-01, 1.0000000000000000e+00,
     2.0000000000000000e+00],
    [8.0000000000000000e+01, 6.0000000000000000e+01, 5.0000000000000000e-01,
     2.0000000000000001e-01, 1.0000000000000001e-01, 4.0000000000000000e+00,
     -5.0000000000000000e-01, 2.0000000000000000e+00, -4.0000000000000002e-01,
     -2.0000000000000001e-01, 5.9999999999999998e-01, 3.0000000000000000e+00,
     5.0000000000000000e+00],
    [3.5000000000000000e+01, 2.5000000000000000e+01, 2.9999999999999999e-01,
     9.4999999999999996e-01, 9.0000000000000002e-01, 2.0000000000000000e+00,
     5.9999999999999998e-01, 5.0000000000000000e-01, 8.0000000000000004e-01,
     9.0000000000000002e-01, 8.0000000000000004e-01, 5.0000000000000000e+00,
     1.0000000000000000e+00],
]
GOLDEN_LOG_PINJ = [
    -21.101004238648947, -21.18689396426054, -21.743938791060764,
    -21.357534657356958, -21.26396243618221, -21.29488079148473,
    -18.74808509919889, -21.29958221597647, -19.119656514617866,
    -19.916410241730134, -23.268518295663398, -23.39505555214776,
]
GOLDEN_PDET = [
    4.1633688689344744e-01, 5.7226979460217806e-02, 2.0261172788879078e-01,
    8.4016436435206718e-02, 1.7817439103537641e-01, 3.2413623608237407e-01,
    3.5280166620722098e-04, 1.6904695136485906e-01, 1.7448427805431541e-01,
    2.1045820400054278e-07, 2.6556625764685715e-01, 6.3249473996212391e-01,
]


@pytest.mark.skipif(not PDET_NPZ.exists(), reason="PDetO4NF.npz not present")
@pytest.mark.skipif(not GOLDEN_THETA, reason="golden constants not generated yet")
def test_golden_parity_with_gwnf():
    flow, config = sel.load_pdet_flow(PDET_NPZ)
    xi = float(config["xi"])
    theta = np.array(GOLDEN_THETA)

    z_grid, z_pdf = sel.build_injection_z_prior()
    c = {name: theta[:, i] for i, name in enumerate(sel.PDET_COLUMNS)}
    log_pinj = (
        sel._gwtc4_log_p_m1(c["m1_source"])
        + sel._gwtc4_log_p_m2_given_m1(c["m2_source"], c["m1_source"])
        + sel._gwtc4_log_p_z(c["redshift"], z_grid, z_pdf)
        + sel._gwtc4_log_p_spin_mag(c["a1"])
        + sel._gwtc4_log_p_spin_mag(c["a2"])
        + sel._gwtc4_log_p_cos_tilt(c["cos_tilt1"])
        + sel._gwtc4_log_p_cos_tilt(c["cos_tilt2"])
        - np.log(2 * np.pi)          # ra
        - np.log(2.0)                # sin dec
        - np.log(np.pi)              # psi
        - np.log(2.0)                # cos iota
        - 2.0 * np.log(2 * np.pi)    # phi1, phi2
    )
    np.testing.assert_allclose(log_pinj, GOLDEN_LOG_PINJ, rtol=1e-5)

    log_q = np.asarray(flow.log_prob(jnp.asarray(theta)))
    pdet = xi * np.exp(log_q - log_pinj)
    np.testing.assert_allclose(pdet, GOLDEN_PDET, rtol=1e-5)
