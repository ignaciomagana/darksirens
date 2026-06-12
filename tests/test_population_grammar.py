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
from darksirens.gw.populations.grammar import (
    ModelNameError,
    parse_model_name,
)
from darksirens.gw.populations.registry import MODEL_NAME_LATEX, get_model

ROOT = Path(__file__).resolve().parents[1]


# ── Parser ───────────────────────────────────────────────────────────────────

PARSE_CASES = [
    # name, tokens, tags, shared_beta, shared_spin, canonical
    ("powerlaw+peak",
     ["powerlaw", "peak"], ["PL", "G"], False, False, "powerlaw+peak"),
    ("brokenpowerlaw+2peaks",
     ["brokenpowerlaw", "peak", "peak"], ["BPL", "G1", "G2"], False, False,
     "brokenpowerlaw+2peaks"),
    ("brokenpowerlaw+2peaks+powerlaw",
     ["brokenpowerlaw", "peak", "peak", "powerlaw"],
     ["BPL", "G1", "G2", "PL"], False, False, "brokenpowerlaw+2peaks+powerlaw"),
    ("2powerlaws+peak",
     ["powerlaw", "powerlaw", "peak"], ["PL1", "PL2", "G"], False, False,
     "2powerlaws+peak"),
    ("2powerlaws+3peaks_shared_beta",
     ["powerlaw", "powerlaw", "peak", "peak", "peak"],
     ["PL1", "PL2", "G1", "G2", "G3"], True, False, "2powerlaws+3peaks"),
    ("powerlaw+peak_shared_beta_spin",
     ["powerlaw", "peak"], ["PL", "G"], True, True, "powerlaw+peak"),
    ("powerlaw+peak_shared_spin",
     ["powerlaw", "peak"], ["PL", "G"], False, True, "powerlaw+peak"),
    # respelled compositions collapse to the same canonical key
    ("powerlaw+powerlaw+peak",
     ["powerlaw", "powerlaw", "peak"], ["PL1", "PL2", "G"], False, False,
     "2powerlaws+peak"),
    ("brokenpowerlaw+peak+peak",
     ["brokenpowerlaw", "peak", "peak"], ["BPL", "G1", "G2"], False, False,
     "brokenpowerlaw+2peaks"),
]


@pytest.mark.parametrize("name,tokens,tags,sb,ss,canonical", PARSE_CASES)
def test_parse_model_name(name, tokens, tags, sb, ss, canonical):
    ir = parse_model_name(name)
    assert [s.token for s in ir.slots] == tokens
    assert [s.tag for s in ir.slots] == tags
    assert ir.shared_beta is sb
    assert ir.shared_spin is ss
    assert ir.canonical == canonical


def test_respelled_curated_name_gets_curated_priors():
    """'powerlaw+powerlaw+peak' must resolve to the same curated priors as
    '2powerlaws+peak' (canonical aliasing), not blueprint defaults."""
    lo_a, hi_a, _, _ = pop_model_prior_parser("2powerlaws+peak")
    lo_b, hi_b, _, _ = pop_model_prior_parser("powerlaw+powerlaw+peak")
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


def test_unknown_model_raises_value_error():
    with pytest.raises(ValueError):
        get_model("not_a_model")


# ── Legacy aliases ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("legacy,canonical", [
    ("twopowerlaws+peak", "2powerlaws+peak"),
    ("twopowerlaws+2peaks_shared_beta_spin", "2powerlaws+2peaks_shared_beta_spin"),
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
    _, _, labels, latex = pop_model_prior_parser("powerlaw+peak")
    assert labels == [
        "$v_1$",
        r"$\alpha_{\rm PL}$", r"$m_{\min,\rm PL}$", r"$m_{\max,\rm PL}$",
        r"$\delta m_{\min,\rm PL}$", r"$\delta m_{\max,\rm PL}$",
        r"$\mu_{\rm G}$", r"$\sigma_{\rm G}$",
        r"$\beta_{\rm PL}$", r"$\beta_{\rm G}$",
        r"$\mu_{\chi,\rm PL}$", r"$\sigma_{\chi,\rm PL}$",
        r"$\mu_{\chi,\rm G}$", r"$\sigma_{\chi,\rm G}$",
        r"$\gamma$",
    ]
    assert latex == "PL+G"


def test_param_ascii_names():
    model = get_model("powerlaw+peak_shared_beta_spin")
    names = [s.name for s in model.param_specs]
    assert names == [
        "v1",
        "PL.alpha", "PL.m_min", "PL.m_max", "PL.dm_min", "PL.dm_max",
        "G.mu", "G.sigma",
        "beta", "mu_chi", "sigma_chi",
        "gamma",
    ]


def test_single_component_labels_untagged():
    _, _, labels, _ = pop_model_prior_parser("gp_mass")
    assert labels[0] == r"$m_{\min}$"
    assert r"$\beta$" in labels and r"$\mu_\chi$" in labels


# ── Novel compositions ───────────────────────────────────────────────────────

NOVEL_NAMES = [
    "powerlaw+2peaks",
    "brokenpowerlaw+peak+powerlaw",
    "3powerlaws+peak",
    "powerlaw+2peaks_shared_beta_spin",
]


@pytest.mark.parametrize("name", NOVEL_NAMES)
def test_novel_composition_builds_and_evaluates(name):
    lows, highs, labels, latex = pop_model_prior_parser(name)
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


def test_novel_latex_derived():
    _, _, _, latex = pop_model_prior_parser("3powerlaws+peak")
    assert latex == "3PL+G"


# ── Import laziness ──────────────────────────────────────────────────────────

def test_registry_import_does_not_import_gp():
    """Importing the registry must not import .gp (tinygp may be stubbed)."""
    code = (
        "import sys; import darksirens.gw.populations.registry; "
        "assert 'darksirens.gw.populations.gp' not in sys.modules, 'gp imported eagerly'; "
        "sys.stdout.write('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
