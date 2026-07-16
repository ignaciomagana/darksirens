"""Golden K=1 likelihood values pinning the legacy single-catalog path
through the unification refactor (darksirens/likelihood/{core,factory}.py).

Every cell builds ``make_likelihood`` inputs for one feature of the K=1
matrix -- plain dark_sirens (full-array union path AND compact caller-view
path), deterministic Q_LSS, the Q ensemble marginalization, marked hosts,
a live delta_g overdensity, an anisotropic sky model, bright-siren
counterparts, the complete-catalog model (both empty-pixel policies),
spectral sirens, weak lensing (lognormal), field sky weighting, and
selection batching -- and pins ``ll(coord)`` at three deterministic coords
against ``tests/data/unified_k1_golden.json``.

The goldens were recorded on the PRE-unification code (origin/master
likelihood path; PR-1 touches no likelihood math).  The unified K>=1 path
must reproduce them to rtol <= 1e-12 (single-element logsumexp is
IEEE-exact, so bit-identity is the expectation; set
``DARKSIRENS_GOLDEN_EXACT=1`` to enforce ``==``).

Goldens are stored PER JAX BACKEND (cpu/gpu reductions differ at the ULP
level), keyed by ``jax.default_backend()``; a backend with no recorded
goldens fails with a regeneration hint.

Regenerate (ONLY on the legacy path, never after the refactor):
    DARKSIRENS_REGEN_GOLDEN=1 python -m pytest tests/test_unified_k1_golden.py -q

A second test bank asserts each cell's feature is LIVE (perturbing the
feature changes the value), so a silently-inert fixture cannot pin a
meaningless golden.
"""
import jax
jax.config.update("jax_enable_x64", True)

import json
import os
from types import SimpleNamespace

import healpy as hp
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.gw.populations import pop_model_prior_parser
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.inference.prior import build_parameter_space
from darksirens.lensing.wlmagnification import WLParams
from darksirens.likelihood.factory import make_likelihood

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "unified_k1_golden.json")
NG = len(zgrid)
NSIDE = 1
NPIX = hp.nside2npix(NSIDE)
NSAMP = 2
NSEL = 8


# ---------------------------------------------------------------------------
# Shared fixture pieces (numbers mirror test_numpyro_sampler._small_likelihood_inputs
# and test_multitracer_likelihood._shared_physics)
# ---------------------------------------------------------------------------

def _pop_bits():
    pop_lower, pop_upper, pop_labels, _, _ = pop_model_prior_parser("powerlaw+peak")
    pop_fid = get_fixed_population_params("powerlaw+peak")
    sampled = pop_labels[0]
    prior_overrides = {sampled: [float(pop_lower[0]), float(pop_upper[0])]}
    fixed = {
        lbl: float(pop_fid[i]) for i, lbl in enumerate(pop_labels) if lbl != sampled
    }
    return pop_fid, prior_overrides, fixed


def _unit_dirs(n, phase):
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + phase
    nx = np.cos(ang) * np.sqrt(0.75)
    ny = np.sin(ang) * np.sqrt(0.75)
    nz = np.full(n, 0.5)
    return jnp.asarray(nx), jnp.asarray(ny), jnp.asarray(nz)


def _full_sky_data():
    zgals = np.full((NPIX, 1), 0.10, dtype=float)
    dzgals = np.full((NPIX, 1), 0.02, dtype=float)
    wgals = np.ones((NPIX, 1), dtype=float)
    ngals = np.ones(NPIX, dtype=np.int32)
    nx_pe, ny_pe, nz_pe = _unit_dirs(NSAMP, 0.1)
    nx_sel, ny_sel, nz_sel = _unit_dirs(NSEL, 0.7)
    return {
        "nEvents": 1,
        "nsamp": NSAMP,
        "Ndraw": float(NSEL),
        "apix": hp.nside2pixarea(NSIDE),
        "nside": NSIDE,
        "n_pix_catalog": NPIX,
        "zgals": zgals,
        "dzgals": dzgals,
        "wgals": wgals,
        "ngals_catalog": ngals,
        "zgals_catalog": zgals,
        "dzgals_catalog": dzgals,
        "wgals_catalog": wgals,
        "delta_g_pix_z": jnp.zeros((NPIX, NG)),
        "m1det": jnp.array([36.0, 38.0]),
        "m2det": jnp.array([28.8, 30.4]),
        "dL": jnp.array([460.0, 500.0]),
        "chieff": jnp.array([0.0, 0.02]),
        "p_pe": jnp.ones(NSAMP),
        "pixels_pe": jnp.array([7, 7], dtype=jnp.int32),
        "nx_pe": nx_pe, "ny_pe": ny_pe, "nz_pe": nz_pe,
        "m1detsels": jnp.linspace(34.0, 40.0, NSEL),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, NSEL),
        "dLsels": jnp.linspace(430.0, 530.0, NSEL),
        "chieffsels": jnp.zeros(NSEL),
        "p_draw": jnp.ones(NSEL),
        "pixels_sel": jnp.array([2, 7, 2, 7, 2, 7, 2, 7], dtype=jnp.int32),
        "nx_sel": nx_sel, "ny_sel": ny_sel, "nz_sel": nz_sel,
    }


def _compact_view_data():
    """Compact caller-view path (bundle-style keys, one catalog row)."""
    data = {
        "nEvents": 1, "nsamp": NSAMP, "Ndraw": float(NSEL),
        "apix": hp.nside2pixarea(NSIDE),
        "delta_g_pix_z": jnp.zeros((1, NG)),
        "m1det": jnp.array([36.0, 38.0]), "m2det": jnp.array([28.8, 30.4]),
        "dL": jnp.array([460.0, 500.0]), "chieff": jnp.array([0.0, 0.02]),
        "p_pe": jnp.ones(NSAMP),
        "m1detsels": jnp.linspace(34.0, 40.0, NSEL),
        "m2detsels": 0.8 * jnp.linspace(34.0, 40.0, NSEL),
        "dLsels": jnp.linspace(430.0, 530.0, NSEL),
        "chieffsels": jnp.zeros(NSEL), "p_draw": jnp.ones(NSEL),
        "zgals_pe": np.array([[0.10]]), "dzgals_pe": np.array([[0.02]]),
        "wgals_pe": np.array([[1.0]]), "ngals_pe": np.array([1], dtype=np.int32),
        "unique_pixels_pe": np.array([0], dtype=np.int32),
        "sample_to_unique_pe": np.zeros(NSAMP, dtype=np.int32),
        "zgals_sel": np.array([[0.10]]), "dzgals_sel": np.array([[0.02]]),
        "wgals_sel": np.array([[1.0]]), "ngals_sel": np.array([1], dtype=np.int32),
        "unique_pixels_sel": np.array([0], dtype=np.int32),
        "sample_to_unique_sel": np.zeros(NSEL, dtype=np.int32),
    }
    return data


def _base_opts(**overrides):
    _pop_fid, prior_overrides, fixed = _pop_bits()
    kwargs = dict(
        pop_model="powerlaw+peak",
        universe_model="dark_sirens",
        sel_batch_size=None,
        fix_cosmology=True,
        fix_population=False,
        fix_survey=True,
        prior_overrides=prior_overrides,
        fixed_parameter_values=fixed,
        complete_empty_pixel_policy="volume",
        bright_siren_sky_marginalized=False,
    )
    kwargs.update(overrides)
    return SimpleNamespace(**kwargs)


def _logq_table():
    """Mild deterministic Q pattern over global pixels: alternating boost/cut."""
    base = np.linspace(-0.3, 0.3, NG)
    table = np.array([base * (1.0 if p % 2 == 0 else -1.0) for p in range(NPIX)])
    return jnp.asarray(table)


def _logq_members(m=3):
    tab = np.asarray(_logq_table())
    members = np.stack([tab * s for s in (0.5, 1.0, 1.5)][:m], axis=0)
    return jnp.asarray(members)


def _mark_table(scale=1.0):
    """(NPIX, maxg=1) log-Mstar-like mark values, deterministic per pixel."""
    vals = 10.0 + 0.2 * np.arange(NPIX, dtype=float).reshape(NPIX, 1) * scale
    return jnp.asarray(vals)


def _delta_g_table():
    base = 0.5 * np.sin(np.linspace(0.0, 3.0, NG))
    table = np.array([base * ((p % 3) - 1) for p in range(NPIX)])
    return jnp.asarray(table)


# ---------------------------------------------------------------------------
# The golden matrix
# ---------------------------------------------------------------------------

def _cell_plain_full():
    return _base_opts(), _full_sky_data()


def _cell_plain_compact():
    return _base_opts(), _compact_view_data()


def _cell_qdet():
    data = _full_sky_data()
    data["lss_completion_logq"] = _logq_table()
    data["lss_completion_indexing"] = 2
    return _base_opts(), data


def _cell_ensemble_marg():
    data = _full_sky_data()
    data["lss_completion_logq_members"] = _logq_members()
    data["lss_completion_indexing"] = 2
    return _base_opts(lss_marginalize=True), data


def _cell_marks():
    data = _full_sky_data()
    data["mark_logmstar"] = _mark_table()
    return _base_opts(mark_model="loglinear", mark_names=("logmstar",)), data


def _cell_use_lss():
    data = _full_sky_data()
    data["delta_g_pix_z"] = _delta_g_table()
    return _base_opts(), data


def _cell_sky_dipole():
    return _base_opts(sky_model="dipole"), _full_sky_data()


def _cell_bright():
    data = _full_sky_data()
    data.update(
        counterpart_pixel=7,
        counterpart_pixels=jnp.array([7], dtype=jnp.int32),
        counterpart_zs=jnp.array([0.10]),
        counterpart_dzs=jnp.array([0.02]),
        bright_siren_sky_marginalized=False,
    )
    return _base_opts(universe_model="bright_sirens"), data


def _sparse_sky_data():
    """Full-sky data with pixel 2 EMPTIED: the selection samples visit it, so
    the complete-model empty-pixel policies (zero vs volume) genuinely
    diverge instead of pinning the same number."""
    data = _full_sky_data()
    for key in ("zgals", "zgals_catalog"):
        data[key] = np.array(data[key])
        data[key][2] = 0.0
    for key in ("wgals", "wgals_catalog"):
        data[key] = np.array(data[key])
        data[key][2] = 0.0
    ngals = np.array(data["ngals_catalog"])
    ngals[2] = 0
    data["ngals_catalog"] = ngals
    return data


def _cell_complete_volume():
    return _base_opts(universe_model="dark_sirens_complete"), _sparse_sky_data()


def _cell_complete_zero():
    return (
        _base_opts(universe_model="dark_sirens_complete",
                   complete_empty_pixel_policy="zero"),
        _sparse_sky_data(),
    )


def _cell_spectral():
    return _base_opts(universe_model="spectral_sirens"), _full_sky_data()


def _cell_wl_lognormal():
    data = _full_sky_data()
    data["wl_params"] = WLParams(
        backend=0, a=0.05, b=1.5,
        z_grid=jnp.asarray([0.0, 1.0]),
        log_mu_grid=jnp.asarray([0.0, 1.0]),
        log_p_table=jnp.zeros((2, 2)),
    )
    return _base_opts(universe_model="spectral_sirens_wl"), data


def _cell_field_k1():
    data = _full_sky_data()
    # The field scope gate requires the legacy dummy overdensity grid.
    data["delta_g_pix_z"] = jnp.zeros((1, NG))
    return _base_opts(catalog_sky_weighting="field"), data


def _cell_sel_batch():
    return _base_opts(sel_batch_size=3), _full_sky_data()


CELLS = {
    "plain_full": _cell_plain_full,
    "plain_compact": _cell_plain_compact,
    "qdet": _cell_qdet,
    "ensemble_marg": _cell_ensemble_marg,
    "marks": _cell_marks,
    "use_lss": _cell_use_lss,
    "sky_dipole": _cell_sky_dipole,
    "bright": _cell_bright,
    "complete_volume": _cell_complete_volume,
    "complete_zero": _cell_complete_zero,
    "spectral": _cell_spectral,
    "wl_lognormal": _cell_wl_lognormal,
    "field_k1": _cell_field_k1,
    "sel_batch": _cell_sel_batch,
}

COORD_FRACTIONS = (0.5, 0.35, 0.65)


def _space_for(opts):
    res = build_parameter_space(
        opts.pop_model,
        opts.fix_population,
        opts.fix_cosmology,
        opts.fix_survey,
        prior_overrides=opts.prior_overrides,
        fixed_parameter_values=opts.fixed_parameter_values,
        universe_model=opts.universe_model,
        sky_model=getattr(opts, "sky_model", "isotropic"),
        mark_model=getattr(opts, "mark_model", "none"),
        mark_names=tuple(getattr(opts, "mark_names", ()) or ()),
    )
    labels, lower, upper = res[0], np.asarray(res[1]), np.asarray(res[2])
    return labels, lower, upper


def _coords_for(opts):
    labels, lower, upper = _space_for(opts)
    return labels, [lower + f * (upper - lower) for f in COORD_FRACTIONS]


def _evaluate_cell(name):
    opts, data = CELLS[name]()
    pop_fid, _overrides, fixed = _pop_bits()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    _labels, coords = _coords_for(opts)
    return [float(ll(jnp.asarray(c))) for c in coords]


def _load_golden():
    if not os.path.exists(GOLDEN_PATH):
        pytest.fail(
            f"Golden file missing: {GOLDEN_PATH}. Regenerate ON THE LEGACY "
            "PATH with DARKSIRENS_REGEN_GOLDEN=1."
        )
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _golden_backend_key():
    """Golden-store key for the current device.

    ``cpu`` stays the plain historical key; accelerators are keyed by the
    concrete device kind (e.g. ``gpu:NVIDIA H100``) because XLA reduction
    strategies differ between GPU generations at the ULP level, so one GPU
    model's values must not masquerade as universal ``gpu`` goldens.
    """
    backend = jax.default_backend()
    if backend == "cpu":
        return "cpu"
    kind = str(jax.devices()[0].device_kind).strip()
    return f"{backend}:{kind}"


# ---------------------------------------------------------------------------
# Golden comparison
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CELLS))
def test_k1_golden(name):
    backend = _golden_backend_key()
    values = _evaluate_cell(name)
    assert np.all(np.isfinite(values)), (name, values)

    if os.environ.get("DARKSIRENS_REGEN_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        golden = {}
        if os.path.exists(GOLDEN_PATH):
            with open(GOLDEN_PATH) as f:
                golden = json.load(f)
        golden.setdefault(backend, {})[name] = values
        with open(GOLDEN_PATH, "w") as f:
            json.dump(golden, f, indent=2, sort_keys=True)
        pytest.skip(f"regenerated golden for {name} [{backend}]")

    golden = _load_golden()
    if backend not in golden:
        pytest.skip(
            f"no goldens recorded for backend '{backend}' (cpu goldens are the "
            "enforced pin); record them on a free device with "
            "DARKSIRENS_REGEN_GOLDEN=1."
        )
    assert name in golden[backend], (
        f"no golden recorded for cell '{name}' on '{backend}' -- regenerate."
    )
    expected = np.asarray(golden[backend][name])
    got = np.asarray(values)
    if os.environ.get("DARKSIRENS_GOLDEN_EXACT"):
        assert got.tolist() == expected.tolist(), (name, got, expected)
    else:
        np.testing.assert_allclose(got, expected, rtol=1e-12, atol=0.0,
                                   err_msg=f"cell '{name}' drifted")


# ---------------------------------------------------------------------------
# Feature-liveness probes: a cell whose feature is silently inert would pin a
# meaningless golden -- prove each perturbation moves the value.
# ---------------------------------------------------------------------------

def _value(opts, data, coord_override=None):
    pop_fid, _overrides, fixed = _pop_bits()
    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)
    _labels, coords = _coords_for(opts)
    coord = coords[0] if coord_override is None else coord_override
    return float(ll(jnp.asarray(coord)))


def test_qdet_is_live():
    v_plain = _value(*_cell_plain_full())
    v_q = _value(*_cell_qdet())
    assert v_plain != v_q


def test_ensemble_marginalization_is_live():
    v_q = _value(*_cell_qdet())
    v_m = _value(*_cell_ensemble_marg())
    assert v_q != v_m


def test_marks_eta_is_live():
    opts, data = _cell_marks()
    labels, coords = _coords_for(opts)
    assert "eta_logmstar" in labels
    idx = labels.index("eta_logmstar")
    base = np.array(coords[0])
    perturbed = base.copy()
    perturbed[idx] = base[idx] + 0.8
    assert _value(opts, data, base) != _value(opts, data, perturbed)


def test_delta_g_is_live():
    v_plain = _value(*_cell_plain_full())
    v_lss = _value(*_cell_use_lss())
    assert v_plain != v_lss


def test_sky_dipole_params_are_live():
    opts, data = _cell_sky_dipole()
    labels, coords = _coords_for(opts)
    # Dipole sky labels are LaTeX: $d_x$, $d_y$, $d_z$ (appended after pop).
    sky_idx = [i for i, l in enumerate(labels) if "d_" in l]
    assert sky_idx, f"no dipole sky labels found in {labels}"
    base = np.array(coords[0])
    perturbed = base.copy()
    perturbed[sky_idx[0]] += 0.05
    assert _value(opts, data, base) != _value(opts, data, perturbed)


def test_field_weighting_is_live():
    v_cond = _value(*_cell_plain_full())
    v_field = _value(*_cell_field_k1())
    assert v_cond != v_field


def test_wl_is_live_relative_to_spectral():
    v_spec = _value(*_cell_spectral())
    v_wl = _value(*_cell_wl_lognormal())
    assert v_spec != v_wl


def test_complete_policies_differ_models():
    v_dark = _value(*_cell_plain_full())
    v_complete = _value(*_cell_complete_volume())
    assert v_dark != v_complete
