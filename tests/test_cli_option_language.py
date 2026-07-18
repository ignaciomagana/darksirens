"""CLI option-language cleanup (PR 2): deprecated-spelling aliases, dest
stability, boolean standardization, lensing argument groups, and the TinyNS
resolved-config display keys.

These parse in-process via each CLI's ``build_parser()`` (no subprocess), so the
DeprecatedSpellingAction warnings land on captured stdout and argparse type
errors surface as ``SystemExit`` with code 2.
"""
from dataclasses import fields

import pytest

pytest.importorskip("jax")

from darksirens.cli.inference import (
    build_parser as build_inference_parser,
    _canonicalize_fixed_flags,
)
from darksirens.cli.inference_lensing import build_parser as build_lensing_parser
from darksirens.cli.common import _print_all_cli_options
from darksirens.inference.tinyns_config import (
    TINYNS_RESOLVED_DISPLAY_KEYS,
    TinyNSConfig,
)


# ── minimal required-arg prefixes ───────────────────────────────────────────────

_INF_REQ = ["--sampler", "tinyns"]
_LEN_REQ = ["--gw_path", "gw.h5", "--gwselection_path", "sel.h5", "--sampler", "tinyns"]


def _parse_inference(extra):
    return build_inference_parser().parse_args(_INF_REQ + extra)


def _parse_lensing(extra):
    return build_lensing_parser().parse_args(_LEN_REQ + extra)


# ── 1. alias equivalence + deprecation notices ──────────────────────────────────

@pytest.mark.parametrize(
    "dest, canonical, deprecated",
    [
        ("fix_cosmology", "--fix_cosmology", "--fixed_cosmology"),
        ("fix_de", "--fix_de", "--fixed_de"),
        ("use_LSS", "--use_lss", "--use_LSS"),
    ],
)
def test_alias_equivalence_and_notice(dest, canonical, deprecated, capsys):
    canon = _parse_inference([canonical, "true"])
    canon_out = capsys.readouterr().out
    assert getattr(canon, dest) is True
    # Canonical spelling is silent.
    assert "deprecated" not in canon_out

    depr = _parse_inference([deprecated, "true"])
    depr_out = capsys.readouterr().out
    # Same resulting opt.
    assert getattr(depr, dest) is True
    # One deprecation notice mapping old -> canonical.
    assert deprecated in depr_out
    assert "deprecated" in depr_out
    assert canonical in depr_out


def test_alias_false_value_maps_through_deprecated_spelling(capsys):
    depr = _parse_inference(["--fixed_cosmology=false"])
    out = capsys.readouterr().out
    assert depr.fix_cosmology is False
    assert "--fixed_cosmology is deprecated" in out


def test_canonical_spellings_together_are_silent(capsys):
    opts = _parse_inference(["--fix_cosmology", "true", "--fix_de", "true", "--use_lss", "true"])
    out = capsys.readouterr().out
    assert (opts.fix_cosmology, opts.fix_de, opts.use_LSS) == (True, True, True)
    assert "deprecated" not in out


# ── 2. dest-stability regression (persisted settings keys) ──────────────────────

def test_inference_dest_stability():
    # Representative command line exercising all three renamed flags via the
    # canonical spelling; the persisted dests (vars(opts) keys) must not change.
    opts = _parse_inference([
        "--fix_cosmology", "true",
        "--fix_de", "true",
        "--use_lss", "true",
        "--fix_population", "false",
    ])
    _canonicalize_fixed_flags(opts)  # sets the persisted fixed_cosmology/fixed_de
    keys = set(vars(opts))
    # argparse dests unchanged AND the canonical persisted names present.
    for required in ("fix_cosmology", "fixed_cosmology", "fix_de", "fixed_de",
                     "use_LSS", "fix_population"):
        assert required in keys, required


def test_lensing_dest_stability():
    opts = _parse_lensing([
        "--fix_cosmology", "true",
        "--fix_survey", "true",
        "--fix_population", "false",
        "--fix_lens_rate", "true",
        "--preflight_only", "false",
        "--allow_suspicious_time_marks", "false",
    ])
    keys = set(vars(opts))
    for required in ("fix_cosmology", "fix_survey", "fix_population",
                     "fix_lens_rate", "preflight_only", "show_progress",
                     "allow_suspicious_time_marks"):
        assert required in keys, required


# ── 3. boolean parsing matrix ───────────────────────────────────────────────────

def test_inference_optional_value_bool():
    # --allow_unverified_shared_lss_members: bare / value / default.
    assert _parse_inference(["--allow_unverified_shared_lss_members"]).allow_unverified_shared_lss_members is True
    assert _parse_inference(["--allow_unverified_shared_lss_members", "false"]).allow_unverified_shared_lss_members is False
    assert _parse_inference([]).allow_unverified_shared_lss_members is False
    # --show_progress requires a value on the main CLI (str_to_bool, no const).
    assert _parse_inference(["--show_progress", "false"]).show_progress is False
    assert _parse_inference([]).show_progress is True


def test_inference_garbage_bool_exits_2():
    with pytest.raises(SystemExit) as exc:
        _parse_inference(["--allow_unverified_shared_lss_members", "banana"])
    assert exc.value.code == 2


def test_lensing_show_progress_bool_matrix():
    assert _parse_lensing(["--show_progress"]).show_progress is True
    assert _parse_lensing(["--show_progress", "false"]).show_progress is False
    assert _parse_lensing([]).show_progress is False


def test_lensing_defaults_preserved():
    opts = _parse_lensing([])
    assert opts.fix_cosmology is True
    assert opts.fix_survey is True
    assert opts.fix_population is False
    assert opts.fix_lens_rate is True
    assert opts.preflight_only is False
    assert opts.allow_suspicious_time_marks is False


@pytest.mark.parametrize("flag", ["--fix_cosmology", "--show_progress", "--preflight_only"])
def test_lensing_garbage_bool_exits_2(flag):
    with pytest.raises(SystemExit) as exc:
        _parse_lensing([flag, "banana"])
    assert exc.value.code == 2


# ── 4. lensing argument groups + option table smoke ─────────────────────────────

def test_lensing_group_titles_present():
    parser = build_lensing_parser()
    titles = {g.title for g in parser._action_groups}
    for expected in ("Data", "Model", "Fixing", "Sampler", "Performance", "Output"):
        assert expected in titles, expected


def test_print_all_cli_options_without_normalization_grid(capsys):
    parser = build_lensing_parser()
    opts = parser.parse_args(_LEN_REQ)
    _print_all_cli_options(parser, opts)  # normalization_grid omitted
    out = capsys.readouterr().out
    assert "All CLI Options" in out
    assert "[Data]" in out and "[Fixing]" in out and "[Sampler]" in out
    # No GW-population normalization grid on this stack -> no [Derived] rows.
    assert "normalization_grid" not in out


# ── 5. TinyNS display keys are a subset of the config fields ─────────────────────

def test_tinyns_display_keys_subset_of_config_fields():
    field_names = {f.name for f in fields(TinyNSConfig)}
    assert set(TINYNS_RESOLVED_DISPLAY_KEYS) <= field_names
