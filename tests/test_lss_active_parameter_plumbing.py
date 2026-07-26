"""Per-catalog Q_LSS activity plumbing: sampler labels == decoder labels.

A loaded Q_LSS completion table REPLACES the local
``(1 + alpha_miss*b_miss*delta_g)`` factor for a catalog, so that catalog's
``b_miss`` no longer enters the likelihood and must be dropped from its sampled
survey block (else it is a phantom flat nuisance dimension).  The likelihood
decides Q-vs-``b_miss`` PER CATALOG (``completion.field_lss_q is not None``), so
a mixed K>=2 config (Q on some catalogs only) must keep ``b_miss_c{k}`` for the
Q-free catalogs.

This file pins the whole plumbing:
  * ``build_parameter_space`` gates b_miss per catalog (scalar broadcasts;
    sequence is per-catalog with a strict length check);
  * ``build_parameter_decoder`` re-derives the SAME sampled labels (the P0.1
    bug: the decoder ignored the flag and diverged from the sampler);
  * an explicit b_miss override/fixed value for a Q-active catalog fails early;
  * the ``expected_sampled_labels`` fail-fast net catches CLI/decoder drift;
  * an end-to-end K=2 mixture (Q on catalog 1 only) evaluates + differentiates;
  * ``make_pop_extractor`` rebuilds the same space from settings.json.

Fixtures reuse the tiny in-memory K=2 helpers from
``tests/test_multitracer_likelihood.py`` rather than inventing new data.
"""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from darksirens.redshift import zgrid
from darksirens.gw.populations.registry import get_fixed_population_params
from darksirens.inference.prior import build_parameter_space
from darksirens.inference.parameters import build_parameter_decoder
from darksirens.inference.pop_extractor import make_pop_extractor
from darksirens.likelihood.factory import make_likelihood

# Reuse the tiny in-memory multitracer fixture helpers.
from test_multitracer_likelihood import (
    _bundle,
    _shared_physics,
    _pop_bits,
    _base_opts,
    APIX1,
    Z_A,
    Z_B,
)

POP = "powerlaw+peak"


def _builder_labels(n_catalogs, lss_active, **overrides):
    """Survey-only sampled labels for the free-survey dark_sirens builder."""
    kwargs = dict(
        pop_model=POP,
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        universe_model="dark_sirens",
        n_catalogs=n_catalogs,
        lss_completion_active=lss_active,
    )
    kwargs.update(overrides)
    res = build_parameter_space(**kwargs)
    return list(res[0])


def _decoder_opts(n_catalogs, **attrs):
    base = dict(
        pop_model=POP,
        universe_model="dark_sirens",
        fix_population=True,
        fix_cosmology=True,
        fix_survey=False,
        n_catalogs=n_catalogs,
    )
    base.update(attrs)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# (a) Builder matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_catalogs, lss_active, present, absent",
    [
        (1, False, {"b_miss"}, set()),
        (1, True, set(), {"b_miss"}),
        (2, (True, True), set(), {"b_miss", "b_miss_c2"}),
        (2, (True, False), {"b_miss_c2"}, {"b_miss"}),
        (2, (False, True), {"b_miss"}, {"b_miss_c2"}),
    ],
)
def test_builder_matrix_per_catalog_b_miss_gating(n_catalogs, lss_active, present, absent):
    labels = set(_builder_labels(n_catalogs, lss_active))
    assert present <= labels, (present, labels)
    assert not (absent & labels), (absent, labels)


# ---------------------------------------------------------------------------
# (b) Scalar back-compat: a bare bool broadcasts to every catalog
# ---------------------------------------------------------------------------

def test_scalar_true_broadcasts_and_drops_both_b_miss_at_k2():
    labels = set(_builder_labels(2, True))
    assert "b_miss" not in labels
    assert "b_miss_c2" not in labels


def test_scalar_false_broadcasts_and_keeps_both_b_miss_at_k2():
    labels = set(_builder_labels(2, False))
    assert "b_miss" in labels
    assert "b_miss_c2" in labels


# ---------------------------------------------------------------------------
# (c) Wrong-length sequence raises, naming both lengths
# ---------------------------------------------------------------------------

def test_wrong_length_sequence_raises_naming_lengths():
    with pytest.raises(ValueError) as ei:
        _builder_labels(2, (True, False, True))
    msg = str(ei.value)
    assert "3" in msg and "2" in msg, msg


# ---------------------------------------------------------------------------
# (d) Decoder coherence: decoder sampled labels == builder labels, per row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_catalogs, lss_active",
    [
        (1, False),
        (1, True),
        (2, (True, True)),
        (2, (True, False)),
        (2, (False, True)),
    ],
)
def test_decoder_labels_match_builder_labels(n_catalogs, lss_active):
    builder = _builder_labels(n_catalogs, lss_active)
    opts = _decoder_opts(n_catalogs, lss_completion_active_by_catalog=lss_active)
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(POP), fixed_parameter_values={}
    )
    assert list(decoder.sampled_labels) == builder


# ---------------------------------------------------------------------------
# (e) Decoder fallback regression (the exact P0.1 bug): only the legacy scalar
# is set on opts, no by-catalog attribute -> the decoder must still drop b_miss.
# ---------------------------------------------------------------------------

def test_decoder_falls_back_to_legacy_scalar_flag():
    opts = _decoder_opts(1, lss_completion_active=True)  # NO by-catalog attr
    assert not hasattr(opts, "lss_completion_active_by_catalog")
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(POP), fixed_parameter_values={}
    )
    assert "b_miss" not in decoder.sampled_labels
    # sanity: without the flag b_miss is present
    opts_off = _decoder_opts(1, lss_completion_active=False)
    decoder_off = build_parameter_decoder(
        opts_off, get_fixed_population_params(POP), fixed_parameter_values={}
    )
    assert "b_miss" in decoder_off.sampled_labels


# ---------------------------------------------------------------------------
# (f) expected_sampled_labels mismatch -> ValueError containing both sequences
# ---------------------------------------------------------------------------

def test_expected_sampled_labels_mismatch_raises_with_both_sequences():
    opts = _decoder_opts(
        1,
        lss_completion_active_by_catalog=(False,),
        expected_sampled_labels=("H0", "definitely_wrong_label"),
    )
    with pytest.raises(ValueError) as ei:
        build_parameter_decoder(
            opts, get_fixed_population_params(POP), fixed_parameter_values={}
        )
    msg = str(ei.value)
    # the (wrong) CLI-resolved sequence and the (correct) decoder-resolved one
    assert "definitely_wrong_label" in msg
    assert "b_miss" in msg  # part of the real decoder-resolved survey block


def test_expected_sampled_labels_match_does_not_raise():
    labels = _builder_labels(2, (True, False))
    opts = _decoder_opts(
        2,
        lss_completion_active_by_catalog=(True, False),
        expected_sampled_labels=tuple(labels),
    )
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(POP), fixed_parameter_values={}
    )
    assert list(decoder.sampled_labels) == labels


# ---------------------------------------------------------------------------
# (g) Fail-early on an explicit b_miss override / fixed value for a Q-active
# catalog; still legal for a Q-free catalog.
# ---------------------------------------------------------------------------

def test_b_miss_override_under_k1_q_active_raises():
    with pytest.raises(ValueError, match=r"b_miss.*Q_LSS|Q_LSS.*b_miss"):
        _builder_labels(1, True, prior_overrides={"b_miss": [0.0, 1.0]})


def test_b_miss_fixed_under_k1_q_active_raises():
    with pytest.raises(ValueError, match="Q_LSS"):
        _builder_labels(1, True, fixed_parameter_values={"b_miss": 0.5})


def test_b_miss_c2_override_under_k2_true_false_is_allowed():
    # Catalog 2 is Q-FREE, so b_miss_c2 remains a legal sampled/overridable label.
    labels = _builder_labels(
        2, (True, False), prior_overrides={"b_miss_c2": [0.0, 1.0]}
    )
    assert "b_miss_c2" in labels
    assert "b_miss" not in labels  # catalog 1 is Q-active


def test_b_miss_c2_fixed_under_k2_false_true_raises():
    with pytest.raises(ValueError, match="Q_LSS"):
        _builder_labels(2, (False, True), fixed_parameter_values={"b_miss_c2": 0.5})


def test_block_fix_survey_is_not_affected_by_fail_early():
    # fix_survey=True fixes the WHOLE block, not a named b_miss label -> no raise
    # even with a Q-active catalog.
    res = build_parameter_space(
        pop_model=POP,
        fix_population=True,
        fix_cosmology=True,
        fix_survey=True,
        universe_model="dark_sirens",
        n_catalogs=1,
        lss_completion_active=True,
    )
    assert "b_miss" not in res[0]


# ---------------------------------------------------------------------------
# (h) End-to-end: K=2 mixture with a deterministic Q table on catalog 1 only.
# ---------------------------------------------------------------------------

def test_end_to_end_k2_q_on_catalog1_only():
    _pl, _pu, _plabels, pop_fid, _sampled, fixed = _pop_bits()

    lss_active = (True, False)
    opts = _base_opts(
        n_catalogs=2,
        fix_survey=False,
        lss_completion_active_by_catalog=lss_active,
        # Sampled survey params feed the Q-catalog completion, whose redshift
        # prior state is wrapped in lax.optimization_barrier under the default
        # "auto" mode; that barrier has no differentiation rule, so a
        # gradient-based sampler (NUTS) resolves it OFF.  Mirror that here so the
        # reverse-mode jax.grad check below exercises the real NUTS path.
        redshift_prior_barrier="off",
    )

    # Labels/bounds the CLI would resolve (and record on opts) for this config.
    res = build_parameter_space(
        pop_model=opts.pop_model,
        fix_population=opts.fix_population,
        fix_cosmology=opts.fix_cosmology,
        fix_survey=opts.fix_survey,
        prior_overrides=opts.prior_overrides,
        fixed_parameter_values=fixed,
        universe_model=opts.universe_model,
        n_catalogs=2,
        lss_completion_active=lss_active,
    )
    labels = list(res[0])
    lower = np.asarray(res[1], dtype=float)
    upper = np.asarray(res[2], dtype=float)

    # Catalog 1 Q-active -> b_miss dropped; catalog 2 Q-free -> b_miss_c2 kept.
    assert "b_miss" not in labels
    assert "b_miss_c2" in labels
    # Exercise the CLI fail-fast net end-to-end (decoder must agree).
    opts.expected_sampled_labels = tuple(labels)

    # Catalog 1 carries a deterministic (Q == 1) compact table; catalog 2 none.
    bundle_q = _bundle(APIX1, Z_A)
    bundle_q["lss_completion_logq"] = np.zeros((1, len(zgrid)), dtype=float)
    bundle_q["lss_completion_indexing"] = 1  # compact, used as-is
    data = dict(_shared_physics())
    data["apix"] = APIX1
    data["catalogs"] = [bundle_q, _bundle(APIX1, Z_B)]

    ll = make_likelihood(opts, data, pop_fid, fixed_parameter_values=fixed)

    coord = jnp.asarray(0.5 * (lower + upper))
    assert coord.shape[0] == len(labels)

    val = float(ll(coord))
    assert np.isfinite(val)

    grad = jax.grad(lambda c: ll(c))(coord)
    assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# (i) pop_extractor rebuilds the same per-catalog-gated space from settings.
# ---------------------------------------------------------------------------

def test_pop_extractor_prefers_by_catalog_list_over_scalar():
    settings = {
        "pop_model": POP,
        "universe_model": "dark_sirens",
        "n_catalogs": 2,
        # JSON round-trip yields a list of bools.
        "lss_completion_active_by_catalog": [True, False],
    }
    extractor = make_pop_extractor(settings)

    res = build_parameter_space(
        pop_model=POP,
        fix_population=False,
        fix_cosmology=False,
        fix_survey=False,
        universe_model="dark_sirens",
        n_catalogs=2,
        lss_completion_active=(True, False),
    )
    labels, pop_labels = list(res[0]), res[4]
    assert "b_miss" not in labels        # catalog 1 Q-active
    assert "b_miss_c2" in labels         # catalog 2 Q-free

    theta = jnp.arange(len(labels), dtype=jnp.float64)
    expected = np.array([float(theta[labels.index(l)]) for l in pop_labels])
    np.testing.assert_allclose(np.asarray(extractor(theta)), expected)


# ---------------------------------------------------------------------------
# (g) use_lss threading: the CLI-default configuration must not trip the net
# ---------------------------------------------------------------------------

def test_use_lss_off_builder_and_decoder_agree():
    """Regression (a439b98 follow-up): build_parameter_space grew a use_lss
    gate that drops b_miss when --use_lss is off, threaded at the CLI call
    site only.  build_parameter_decoder re-derived the space at the signature
    default use_lss=True, so the DEFAULT dark_sirens configuration (--use_lss
    off, --fix_survey off) tripped the expected_sampled_labels fail-fast net
    inside make_likelihood before sampling could start."""
    labels = _builder_labels(1, False, use_lss=False)
    assert "b_miss" not in labels

    opts = _decoder_opts(
        1,
        use_LSS=False,
        lss_completion_active_by_catalog=(False,),
        expected_sampled_labels=tuple(labels),
    )
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(POP), fixed_parameter_values={}
    )
    assert list(decoder.sampled_labels) == list(labels)
    assert "b_miss" not in decoder.sampled_labels


def test_use_lss_on_keeps_b_miss_in_decoder():
    """Complementary cell: --use_lss true keeps b_miss for a Q-free catalog,
    and builder/decoder still agree under the explicit flag."""
    labels = _builder_labels(1, False, use_lss=True)
    assert "b_miss" in labels

    opts = _decoder_opts(
        1,
        use_LSS=True,
        lss_completion_active_by_catalog=(False,),
        expected_sampled_labels=tuple(labels),
    )
    decoder = build_parameter_decoder(
        opts, get_fixed_population_params(POP), fixed_parameter_values={}
    )
    assert "b_miss" in decoder.sampled_labels
