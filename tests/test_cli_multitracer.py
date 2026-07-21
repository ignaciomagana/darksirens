"""CLI surface for the K-catalog multitracer mixture (--survey_path nargs='+'
and the post-parse guard block in darksirens/cli/inference.py).  Mirrors
test_cli_sigma_kernel_removed.py's subprocess style.

The guard block runs immediately after argparse, before any file is opened
(load_all_data is only reached much later, in the "Loading data" section), so
these tests use deliberately nonexistent dummy paths -- the process must exit
nonzero from the guard, never from a file-not-found error."""
import subprocess
import sys


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference"] + args,
        capture_output=True,
        text=True,
    )


def test_cli_help_mentions_multi_catalog_survey_path_usage():
    result = subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    # argparse nargs="+" usage marker for --survey_path.
    assert "--survey_path PATH [PATH ...]" in result.stdout
    # The help text documents the K-catalog mixture semantics.
    lowered = result.stdout.lower()
    assert "k-catalog" in lowered or "multiple paths" in lowered
    assert "fcat_2" in result.stdout


def test_cli_help_exposes_allow_unverified_shared_lss_members_flag():
    """The additive escape hatch for the K>=2 --lss_marginalize provenance guard
    must appear in --help.  It is now a str_to_bool flag with an OPTIONAL value
    (nargs='?', const=True), so the bare form still works and the usage line
    carries the ``[BOOL]`` metavar (BC7)."""
    result = subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--allow_unverified_shared_lss_members" in result.stdout
    # Optional-value BOOL flag: the usage line shows the [BOOL] metavar; the
    # legacy store_true-style VALUE metavar never appears.
    assert "--allow_unverified_shared_lss_members VALUE" not in result.stdout
    assert "--allow_unverified_shared_lss_members [BOOL]" in result.stdout
    # --lss_marginalize documents the matched-realizations / realization_set_id
    # requirement that the flag bypasses.
    assert "realization_set_id" in result.stdout


def test_two_survey_paths_with_non_dark_sirens_universe_model_exits_nonzero():
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--universe_model", "spectral_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "dark_sirens" in result.stdout
    # Must fail before any file access is attempted.
    assert "No such file" not in result.stdout
    assert "No such file" not in result.stderr


def test_two_survey_paths_with_one_lss_completion_entry_exits_nonzero():
    """K=2 requires --lss_completion to have 0 or exactly n_catalogs entries;
    a single entry (ambiguous which catalog it belongs to) must be rejected
    explicitly rather than silently broadcast."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--lss_completion", "/nonexistent/lssA.h5",
        "--universe_model", "dark_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "lss_completion" in result.stdout
    assert "No such file" not in result.stdout
    assert "No such file" not in result.stderr


def test_single_survey_path_is_unaffected_by_multi_catalog_guards():
    """A single --survey_path (the legacy single-catalog invocation) must NOT
    trip the K>=2 guards -- it should fail later, for the ordinary reason
    (missing GW/selection files), not on universe_model/lss_completion."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5",
        "--universe_model", "spectral_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "Multiple --survey_path catalogs" not in result.stdout


# ---------------------------------------------------------------------------
# Catalog sky-weighting coherence: auto-resolution by K and the two fatal
# degenerate explicit combinations (dark_sirens only).
# ---------------------------------------------------------------------------

def test_explicit_field_at_k1_dark_sirens_is_accepted():
    """field@K=1: now ACCEPTED for dark_sirens (field is the default estimand).
    The survey-global normalizer's log10n0 channel cancels at K=1, but field
    restores the relative angular host weighting the conditional estimand
    discards, so the run proceeds past the guard to data loading (failing there
    on the dummy paths, not in the sky-weighting guard)."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5",
        "--universe_model", "dark_sirens",
        "--catalog_sky_weighting", "field",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "degenerate with a single" not in result.stdout
    assert "field (explicit)" in result.stdout


def test_explicit_conditional_at_k2_dark_sirens_exits_nonzero():
    """conditional@K>=2: the per-pixel normalizer strips the number-density
    channel from fcat (measured railing pathology), so the explicit
    combination is fatal for dark_sirens."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--universe_model", "dark_sirens",
        "--catalog_sky_weighting", "conditional",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "not a host-fraction" in result.stdout
    assert "No such file" not in result.stdout
    assert "No such file" not in result.stderr


def test_unset_weighting_resolves_field_at_k1():
    """No --catalog_sky_weighting at K=1 resolves to field -- the joint
    catalog host-density estimand is now the default at every K (printed in the
    run configuration before data loading fails on the dummy paths)."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5",
        "--universe_model", "dark_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "field (auto)" in result.stdout


def test_unset_weighting_resolves_field_at_k2():
    """No --catalog_sky_weighting at K>=2 resolves to field -- the
    host-fraction estimand the mixture weight exists for."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--universe_model", "dark_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "field (auto)" in result.stdout


def test_unset_weighting_stays_inert_for_non_catalog_models():
    """spectral_sirens never evaluates the catalog prior, so the unset
    weighting must resolve to the inert legacy value rather than field --
    the field-scope validation would otherwise fatal every catalog-free run
    (regression: the field-default rollout broke spectral_sirens entirely)."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--universe_model", "spectral_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "field supports --universe_model" not in result.stdout


def test_unset_weighting_with_drop_full_catalog_resolves_conditional():
    """--drop_full_catalog is an explicit memory request incompatible with
    field's empty-pixel counting; the AUTO default honors it by resolving to
    conditional with a warning instead of a surprise fatal (an EXPLICIT
    field + drop_full_catalog request stays fatal)."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5",
        "--universe_model", "dark_sirens",
        "--drop_full_catalog", "true",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "conditional (auto)" in result.stdout
    assert "incompatible with --drop_full_catalog" not in result.stdout


def test_explicit_weighting_choices_remain_valid_in_coherent_regimes():
    """Explicit conditional@K=1 and explicit field@K=2 stay accepted (they are
    the coherent estimands) -- runs proceed to data loading and fail there."""
    k1 = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5",
        "--universe_model", "dark_sirens",
        "--catalog_sky_weighting", "conditional",
        "--sampler", "tinyns",
    ])
    assert k1.returncode != 0
    assert "conditional (explicit)" in k1.stdout
    k2 = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--universe_model", "dark_sirens",
        "--catalog_sky_weighting", "field",
        "--sampler", "tinyns",
    ])
    assert k2.returncode != 0
    assert "field (explicit)" in k2.stdout


def test_dark_sirens_complete_rules_unchanged():
    """dark_sirens_complete keeps its pre-existing special-case rules:
    explicit conditional@K>=2 trips the complete-specific fatal; unset
    weighting at K>=2 auto-resolves to field and passes that guard; explicit
    field@K=1 stays allowed (no n0 budget to lose)."""
    explicit_conditional = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--universe_model", "dark_sirens_complete",
        "--catalog_sky_weighting", "conditional",
        "--sampler", "tinyns",
    ])
    assert explicit_conditional.returncode != 0
    assert "incoherent estimand" in explicit_conditional.stdout

    auto_field = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--universe_model", "dark_sirens_complete",
        "--sampler", "tinyns",
    ])
    assert auto_field.returncode != 0
    assert "incoherent estimand" not in auto_field.stdout
    assert "field (auto)" in auto_field.stdout

    field_k1 = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5",
        "--universe_model", "dark_sirens_complete",
        "--catalog_sky_weighting", "field",
        "--sampler", "tinyns",
    ])
    assert field_k1.returncode != 0
    assert "degenerate with a single" not in field_k1.stdout
    assert "field (explicit)" in field_k1.stdout


def test_mark_model_requires_dark_sirens():
    """--mark_model with any other universe_model is fatal at any K: marks are
    inert outside the dark_sirens prior and would sample phantom flat eta
    dimensions."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5",
        "--universe_model", "dark_sirens_complete",
        "--mark_model", "loglinear",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "requires --universe_model" in result.stdout
    assert "phantom" in result.stdout
    assert "No such file" not in result.stdout
    assert "No such file" not in result.stderr


def test_k2_lss_completion_accepted_under_field_normalizer():
    """K>=2 + per-catalog --lss_completion is a supported combination: the
    field normalizer carries the Q-modulated missing budget, so validation
    passes and the run proceeds to data loading (failing there on the dummy
    paths, not in the guard block)."""
    result = _run([
        "--gw_path", "/nonexistent/gw.h5",
        "--gwselection_path", "/nonexistent/sel.h5",
        "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
        "--lss_completion", "/nonexistent/lssA.h5", "/nonexistent/lssB.h5",
        "--universe_model", "dark_sirens",
        "--sampler", "tinyns",
    ])
    assert result.returncode != 0
    assert "field (auto)" in result.stdout
    assert "is not supported with --lss_completion" not in result.stdout
    assert "FATAL: --lss_completion" not in result.stdout
