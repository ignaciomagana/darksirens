"""Unit tests for the population model-name grammar and generic builder."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from darksirens.gw.populations import (
    get_fixed_population_params,
    pop_model_parser,
    pop_model_prior_parser,
)
from darksirens.gw.populations.base import PopulationModel
from darksirens.gw.populations.grammar import (
    ModelNameError,
    parse_model_name,
)
from darksirens.gw.populations.registry import MODEL_NAME_LATEX, get_model

ROOT = Path(__file__).resolve().parents[1]


# ── Parser ───────────────────────────────────────────────────────────────────

PARSE_CASES = [
    # name, tokens, tags, shared_beta, shared_spin, shared_gamma, canonical
    ("powerlaw+peak",
     ["powerlaw", "peak"], ["PL", "G"], True, True, True, "powerlaw+peak"),
    ("brokenpowerlaw+2peaks",
     ["brokenpowerlaw", "peak", "peak"], ["BPL", "G1", "G2"], True, True, True,
     "brokenpowerlaw+2peaks"),
    ("brokenpowerlaw+2peaks+powerlaw",
     ["brokenpowerlaw", "peak", "peak", "powerlaw"],
     ["BPL", "G1", "G2", "PL"], True, True, True, "brokenpowerlaw+2peaks+powerlaw"),
    ("2powerlaws+peak",
     ["powerlaw", "powerlaw", "peak"], ["PL1", "PL2", "G"], True, True, True,
     "2powerlaws+peak"),
    # respelled compositions collapse to the same canonical key
    ("powerlaw+powerlaw+peak",
     ["powerlaw", "powerlaw", "peak"], ["PL1", "PL2", "G"], True, True, True,
     "2powerlaws+peak"),
    ("brokenpowerlaw+peak+peak",
     ["brokenpowerlaw", "peak", "peak"], ["BPL", "G1", "G2"], True, True, True,
     "brokenpowerlaw+2peaks"),
]


@pytest.mark.parametrize("name,tokens,tags,sb,ss,sg,canonical", PARSE_CASES)
def test_parse_model_name(name, tokens, tags, sb, ss, sg, canonical):
    ir = parse_model_name(name)
    assert [s.token for s in ir.slots] == tokens
    assert [s.tag for s in ir.slots] == tags
    assert ir.shared_beta is sb
    assert ir.shared_spin is ss
    assert ir.shared_gamma is sg
    assert ir.canonical == canonical


def test_shared_suffixes_are_no_longer_part_of_model_names():
    for name in (
        "powerlaw+peak_shared_beta",
        "powerlaw+peak_shared_spin",
        "powerlaw+peak_shared_gamma",
        "powerlaw+peak_shared_beta_spin",
        "powerlaw+peak_shared_beta_shared_spin_shared_gamma",
    ):
        with pytest.raises(ValueError, match="Unknown component"):
            parse_model_name(name)


def test_respelled_curated_name_gets_curated_priors():
    """'powerlaw+powerlaw+peak' must resolve to the same curated priors as
    '2powerlaws+peak' (canonical aliasing), not blueprint defaults."""
    lo_a, hi_a, _, _, _ = pop_model_prior_parser("2powerlaws+peak")
    lo_b, hi_b, _, _, _ = pop_model_prior_parser("powerlaw+powerlaw+peak")
    assert lo_a == lo_b and hi_a == hi_b
    fid_a = np.asarray(get_fixed_population_params("2powerlaws+peak"))
    fid_b = np.asarray(get_fixed_population_params("powerlaw+powerlaw+peak"))
    assert np.array_equal(fid_a, fid_b)


# ── Errors ───────────────────────────────────────────────────────────────────

def test_bare_plural_rejected():
    with pytest.raises(ValueError, match=r"2peaks"):
        parse_model_name("powerlaw+peaks")


def test_typo_gets_suggestion():
    with pytest.raises(ValueError, match=r"powerlaw"):
        parse_model_name("powrlaw+peak")


def test_malformed_suffix_rejected():
    with pytest.raises(ValueError, match="Unknown component"):
        parse_model_name("powerlaw+peak_sharedbeta")


def test_per_component_gamma_suffix_rejected():
    with pytest.raises(ValueError, match="Unknown component"):
        parse_model_name("powerlaw+peak_per_component_gamma")


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError):
        get_model("not_a_model")


def test_unknown_model_error_lists_registered_models():
    with pytest.raises(ValueError) as err:
        get_model("not_a_model")

    message = str(err.value)
    assert "Registered population models:" in message
    assert "Mixture grammar:" in message
    assert "gp1d_m1" in message
    assert "gwtc5_fiducial_bpl2peaks" in message
    assert "golomb_1g" in message
    assert "powerlaw" in message
    assert "brokenpowerlaw" in message
    assert "peak" in message


def test_fixed_population_unknown_model_error_lists_registered_models():
    with pytest.raises(ValueError) as err:
        get_fixed_population_params("not_a_model")

    message = str(err.value)
    assert "Registered population models:" in message
    assert "gp4d" in message
    assert "gwtc5_brokenpowerlaw+2peaks" in message
    assert "golomb_1g+tail" in message


# ── Legacy aliases ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("legacy,canonical", [
    ("twopowerlaws+peak", "2powerlaws+peak"),
    ("gwtc5_fiducial_brokenpowerlaw+2peaks", "gwtc5_fiducial_bpl2peaks"),
    ("gwtc5_brokenpowerlaw+2peaks", "gwtc5_fiducial_bpl2peaks"),
])
def test_legacy_alias_warns_and_matches(legacy, canonical):
    canon_fid = np.asarray(get_fixed_population_params(canonical))
    with pytest.warns(DeprecationWarning, match="deprecated"):
        legacy_fid = np.asarray(get_fixed_population_params(legacy))
    assert np.array_equal(canon_fid, legacy_fid)
    # get_model may serve the legacy name from the registry cache (warm from
    # earlier tests) without re-warning; only equivalence is asserted here.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy_model = get_model(legacy)
        canon_model = get_model(canonical)
    assert [s.label for s in legacy_model.param_specs] == [
        s.label for s in canon_model.param_specs
    ]
    assert MODEL_NAME_LATEX[legacy] == MODEL_NAME_LATEX[canonical]


# ── Labels ───────────────────────────────────────────────────────────────────

def test_powerlaw_peak_labels():
    _, _, labels, _, latex = pop_model_prior_parser("powerlaw+peak")
    assert labels == [
        "$v_1$",
        r"$\alpha_{\rm PL}$", r"$m_{\min,\rm PL}$", r"$m_{\max,\rm PL}$",
        r"$\delta m_{\min,\rm PL}$", r"$\delta m_{\max,\rm PL}$",
        r"$\mu_{\rm G}$", r"$\sigma_{\rm G}$",
        r"$\beta$",
        r"$\mu_\chi$", r"$\sigma_\chi$",
        r"$\gamma$",
    ]
    assert latex == "PL+G"


def test_shared_gamma_cli_flag_builds_expected_parameters():
    _, _, labels, _, latex = pop_model_prior_parser(
        "powerlaw+peak", shared_beta=True, shared_spin=True, shared_gamma=True
    )
    assert labels[-1:] == [r"$\gamma$"]
    assert latex == "PL+G"

    fid = np.asarray(
        get_fixed_population_params(
            "powerlaw+peak", shared_beta=True, shared_spin=True, shared_gamma=True
        )
    )
    assert fid.shape == (len(labels),)
    assert fid[-1] == 0.0


def test_per_component_cli_flags_build_expected_labels():
    _, _, labels, _, latex = pop_model_prior_parser(
        "powerlaw+peak", shared_beta=False, shared_spin=False, shared_gamma=False
    )
    assert r"$\beta_{\rm PL}$" in labels
    assert r"$\beta_{\rm G}$" in labels
    assert r"$\mu_{\chi,\rm PL}$" in labels
    assert r"$\mu_{\chi,\rm G}$" in labels
    assert labels[-2:] == [r"$\gamma_{\rm PL}$", r"$\gamma_{\rm G}$"]
    assert latex == (
        r"PL+G (Per-component $\beta$, Per-component Spin, "
        r"Per-component $\gamma$)"
    )

    fid = np.asarray(
        get_fixed_population_params(
            "powerlaw+peak", shared_beta=False, shared_spin=False, shared_gamma=False
        )
    )
    assert fid.shape == (len(labels),)


def test_param_ascii_names():
    model = get_model("powerlaw+peak")
    names = [s.name for s in model.param_specs]
    assert names == [
        "v1",
        "PL.alpha", "PL.m_min", "PL.m_max", "PL.dm_min", "PL.dm_max",
        "G.mu", "G.sigma",
        "beta", "mu_chi", "sigma_chi",
        "gamma",
    ]


def test_single_component_labels_untagged():
    _, _, labels, _, _ = pop_model_prior_parser("gp1d_m1")
    assert labels[0] == r"$m_{\min}$"
    assert r"$\beta_q$" in labels and r"$\mu_\chi$" in labels


def test_per_component_gamma_specs_and_evaluation():
    shared_model = get_model("powerlaw+peak")
    model = PopulationModel(mixture=shared_model.mixture, shared_gamma=False)

    specs = model.param_specs
    assert [s.label for s in specs[-2:]] == [
        r"$\gamma_{\rm PL}$",
        r"$\gamma_{\rm G}$",
    ]
    assert [s.name for s in specs[-2:]] == ["PL.gamma", "G.gamma"]

    fid = jnp.asarray(get_fixed_population_params("powerlaw+peak"), dtype=jnp.float64)
    theta = jnp.concatenate([fid[:-1], jnp.asarray([-1.0, 2.0])])
    m1 = jnp.array([8.0, 35.0, 60.0])
    q = jnp.array([0.7, 0.9, 0.8])
    z = jnp.array([0.2, 0.3, 0.1])
    chi = jnp.array([0.0, 0.1, -0.1])

    comp = model.mixture.component_densities(m1, q, chi, fid[:-1])
    expected = jnp.log(
        jnp.sum(comp * (1.0 + z) ** jnp.asarray([[-2.0], [1.0]]), axis=0)
    )
    actual = model.log_p_pop(m1, q, z, chi, theta)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-10)


def test_shared_gamma_default_preserves_param_order_and_evaluation():
    model = get_model("powerlaw+peak")
    fid = jnp.asarray(get_fixed_population_params("powerlaw+peak"), dtype=jnp.float64)
    m1 = jnp.array([8.0, 35.0, 60.0])
    q = jnp.array([0.7, 0.9, 0.8])
    z = jnp.array([0.2, 0.3, 0.1])
    chi = jnp.array([0.0, 0.1, -0.1])

    assert model.shared_gamma is True
    assert model.param_specs[-1].label == r"$\gamma$"
    assert model.param_specs[-1].name == "gamma"

    p = model.mixture(m1, q, chi, fid[:-1])
    expected = jnp.log(p) + (fid[-1] - 1.0) * jnp.log1p(z)
    actual = model.log_p_pop(m1, q, z, chi, fid)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-10)


# ── Novel compositions ───────────────────────────────────────────────────────

NOVEL_NAMES = [
    "powerlaw+2peaks",
    "brokenpowerlaw+peak+powerlaw",
    "3powerlaws+peak",
]


@pytest.mark.parametrize("name", NOVEL_NAMES)
def test_novel_composition_builds_and_evaluates(name):
    lows, highs, labels, _, latex = pop_model_prior_parser(name)
    assert len(lows) == len(highs) == len(labels)
    assert len(set(labels)) == len(labels), "labels must be unique"

    fid = np.asarray(get_fixed_population_params(name), dtype=np.float64)
    assert fid.shape == (len(labels),)
    assert np.all(fid >= np.asarray(lows)) and np.all(fid <= np.asarray(highs))

    log_p_pop = pop_model_parser(name)
    m1 = jnp.array([8.0, 35.0, 60.0])
    q = jnp.array([0.7, 0.9, 0.8])
    z = jnp.array([0.2, 0.3, 0.1])
    chi = jnp.array([0.0, 0.1, -0.1])
    vals = np.asarray(log_p_pop(m1, q, z, chi, jnp.array(fid)))
    assert np.all(np.isfinite(vals)), f"non-finite log_p_pop for {name}: {vals}"


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_mixture_weight_prior_is_uniform_on_the_simplex(k):
    """The stick inputs must carry Beta(1, k-i), i.e. a uniform Dirichlet on the
    weights.  Sampling every v_i from U[0, 1] gave E[w] = (1/2, 1/4, 1/8, ...) --
    measured (0.500, 0.250, 0.250) for k = 3 against 1/3 each -- so the first-named
    component got half the population a priori."""
    from darksirens.gw.populations.base import _stick_breaking_weights
    from darksirens.inference.prior import make_prior_transform

    name = "+".join(["powerlaw"] * (k - 1) + ["peak"])
    lows, highs, labels, kinds, _ = pop_model_prior_parser(name)
    assert labels[:k - 1] == [rf"$v_{i + 1}$" for i in range(k - 1)]
    for i in range(k - 1):
        assert kinds[i] == ("beta", 1.0, float(k - 1 - i)), (i, kinds[i])

    transform = make_prior_transform(lows, highs, kinds)
    rng = np.random.default_rng(0)
    u = jnp.asarray(rng.uniform(size=(40_000, len(lows))))
    v = transform(u)[:, :k - 1]
    w = np.asarray(jax.vmap(_stick_breaking_weights)(v))
    np.testing.assert_allclose(w.mean(axis=0), 1.0 / k, atol=0.01)


def test_novel_latex_derived():
    _, _, _, _, latex = pop_model_prior_parser("3powerlaws+peak")
    assert latex == "3PL+G"


# ── Import laziness ──────────────────────────────────────────────────────────

def test_registry_import_does_not_import_tinygp():
    """Importing the registry must not import tinygp (it may be stubbed/absent).

    GP population models live in ``.gp``, which the registry now imports eagerly
    for registration -- but ``.gp`` imports ``tinygp`` lazily (only inside the
    field evaluation), so importing the registry never pulls ``tinygp``.
    """
    code = (
        "import sys; import darksirens.gw.populations.registry; "
        "assert 'tinygp' not in sys.modules, 'tinygp imported eagerly'; "
        "sys.stdout.write('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
