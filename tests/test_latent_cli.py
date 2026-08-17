"""CLI surface of ``--lss_field_mode latent`` (field-level PR-5, the seam).

Everything guard 6 (PLAN section 4.4) decides is PURE OPTS ARITHMETIC -- the
mode string, the artifact path, ``--c_mode``, ``--per_pixel_completeness``,
``--lss_completion``, ``--use_lss``, ``--stratum_map`` -- so these tests call
``_check_latent_field_mode`` directly on a parsed namespace instead of
spawning a process per refusal.  That is not only ~200x faster than the
subprocess style of test_cli_multitracer.py (measured: 0.9 s for this whole
file against ~4 s for a SINGLE interpreter start); it is also the honest test,
because the guard is deliberately placed pre-load in ``main`` and a subprocess
would prove only that SOME error fired somewhere.

Two things are pinned besides the refusals:

* **Table mode is untouched.**  A default namespace reaches exactly one
  ``getattr`` in the guard and returns; the b_miss rule, the Q-table theta
  firewall and the aggregate-requires-Q refusal all behave as they did.
* **The parameter spaces are separated.**  The run-fingerprint schema version
  is bumped, so a latent run cannot silently resume a table run's checkpoint
  even though the two carry a ``b_miss`` label with identical bounds.
"""
import pytest

import darksirens.cli.inference as inference
from darksirens.cli.inference import (
    _check_aggregate_requires_q,
    _check_latent_field_mode,
    _check_latent_selection_strata,
    _check_selection_qtable_theta,
    _normalize_multitracer_paths,
    build_parser,
    latent_field_mode,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _opts(*extra, artifact=None, latent=True):
    """A parsed namespace carrying the minimum for the pre-load guards.

    ``--sampler`` is the only argparse-required flag reached before the guard
    block, and the guard never opens ``--survey_path``, so a dummy path is
    enough.  ``_normalize_multitracer_paths`` is run because the guard reads
    ``opts.lss_completions`` (its normalized, per-catalog form).
    """
    argv = ["--sampler", "tinyns",
            "--universe_model", "dark_sirens",
            "--survey_path", "/nonexistent/survey.h5"]
    if latent:
        argv += ["--lss_field_mode", "latent"]
        argv += ["--lss_field_artifact", str(artifact)] if artifact else []
    argv += list(extra)
    opts = build_parser().parse_args(argv)
    _normalize_multitracer_paths(opts)
    return opts


@pytest.fixture
def artifact(tmp_path):
    """A stand-in for the PR-4 anchor artifact.

    The pre-load guard checks EXISTENCE only (``os.path.isfile``); the
    artifact's contents are validated by ``latent_q.load_latent_plan`` at
    likelihood-build time, against the run's nside and z_depth, which are not
    knowable here.  So an empty file is exactly the right stub: it separates
    "you forgot the artifact" from "your artifact is wrong".
    """
    path = tmp_path / "latent_field.h5"
    path.write_bytes(b"")
    return path


# ── The flags parse ────────────────────────────────────────────────────────────

def test_flags_default_to_table_mode():
    """Default is OFF: the shipped behaviour, and the predicate says so."""
    opts = build_parser().parse_args(["--sampler", "tinyns"])
    assert opts.lss_field_mode == "table"
    assert opts.lss_field_artifact is None
    assert latent_field_mode(opts) is False


def test_flags_parse_in_latent_mode(artifact):
    opts = _opts("--c_mode", "aggregate",
                 "--per_pixel_completeness", "/depth.h5",
                 artifact=artifact)
    assert opts.lss_field_mode == "latent"
    assert opts.lss_field_artifact == str(artifact)
    assert latent_field_mode(opts) is True
    # The admitted configuration: no refusal.
    _check_latent_field_mode(opts)


def test_latent_field_mode_predicate_defaults_for_namespaces_without_the_flag():
    """Every caller that hand-builds an opts namespace (the completion dry run,
    the lensing CLI's borrowed helpers, older tests) must stay on the table
    path rather than raise AttributeError."""
    class _Bare:
        pass

    assert latent_field_mode(_Bare()) is False


def test_help_documents_the_mode_and_the_parameter_space_change():
    """--help is where an operator learns that latent is a different parameter
    space, not a different Q backend."""
    text = build_parser().format_help()
    assert "--lss_field_mode" in text
    assert "--lss_field_artifact" in text
    assert "b_GW" in text


# ── Guard 6: the refusals ──────────────────────────────────────────────────────

def _fatal_message(opts, check=_check_latent_field_mode):
    with pytest.raises(SystemExit) as exc:
        check(opts)
    assert exc.value.code == 1
    return exc


def test_latent_without_artifact_refuses(capsys):
    opts = _opts("--c_mode", "aggregate",
                 "--per_pixel_completeness", "/depth.h5")
    _fatal_message(opts)
    out = capsys.readouterr().out
    assert "--lss_field_artifact" in out
    assert "darksirens_build_latent_field" in out


def test_artifact_without_latent_mode_refuses(capsys, artifact):
    """The orphan-flag direction: an artifact in table mode is never read, so a
    run that believes it exercised the seam would silently be a table run."""
    opts = _opts(latent=False)
    opts.lss_field_artifact = str(artifact)
    _fatal_message(opts)
    out = capsys.readouterr().out
    assert "without --lss_field_mode latent" in out


def test_latent_with_missing_artifact_file_refuses(capsys, tmp_path):
    opts = _opts("--c_mode", "aggregate",
                 "--per_pixel_completeness", "/depth.h5",
                 artifact=tmp_path / "absent.h5")
    _fatal_message(opts)
    assert "is not a file" in capsys.readouterr().out


def test_latent_with_per_pixel_c_mode_refuses_as_circular(capsys, artifact):
    """latent + per_pixel: (1 - C) already carries the angular clustering the
    field models, so the two completion models double-count each other."""
    opts = _opts("--c_mode", "per_pixel",
                 "--per_pixel_completeness", "/depth.h5",
                 artifact=artifact)
    _fatal_message(opts)
    out = capsys.readouterr().out
    assert "--c_mode per_pixel" in out
    assert "absorbs" in out and "clustering" in out


@pytest.mark.parametrize("c_mode", ["aggregate", "selection"])
def test_latent_without_per_pixel_completeness_refuses(capsys, artifact, c_mode):
    """The admitted c_modes are admitted ONLY with f_p: the artifact's sky
    moments B(z; b) = Sum_p f_p e^{b f} carry it, so a completion side without
    f_p conserves a different budget than rho normalizes."""
    opts = _opts("--c_mode", c_mode, artifact=artifact)
    _fatal_message(opts)
    out = capsys.readouterr().out
    assert "--per_pixel_completeness" in out
    assert "f_p" in out


def test_latent_with_lss_completion_refuses_two_completion_models(capsys, artifact):
    opts = _opts("--c_mode", "aggregate",
                 "--per_pixel_completeness", "/depth.h5",
                 "--lss_completion", "/q_table.h5",
                 artifact=artifact)
    _fatal_message(opts)
    out = capsys.readouterr().out
    assert "--lss_completion" in out
    assert "two different completion models" in out


def test_latent_with_use_lss_refuses(capsys, artifact):
    """Inherits the field_lss_q vs field_delta_g exclusion, and is what keeps
    the b_miss -> b_GW inversion unambiguous."""
    opts = _opts("--c_mode", "aggregate",
                 "--per_pixel_completeness", "/depth.h5",
                 "--use_lss", "true",
                 artifact=artifact)
    _fatal_message(opts)
    out = capsys.readouterr().out
    assert "--use_lss" in out
    assert "b_GW" in out


def test_latent_with_stratum_map_refuses(capsys, artifact):
    """Pre-load half of the stratified refusal: one (A, B) pair per artifact,
    but a stratified run needs one per stratum."""
    opts = _opts("--c_mode", "selection",
                 "--per_pixel_completeness", "/depth.h5",
                 "--selection_fit", "/fit.json",
                 "--stratum_map", "/strata.h5",
                 artifact=artifact)
    _fatal_message(opts)
    out = capsys.readouterr().out
    assert "--stratum_map" in out
    assert "(A, B)" in out or "(A_s, B_s)" in out


def test_latent_with_multi_stratum_fit_refuses_post_load(capsys, artifact):
    """Post-load half: a fit can carry several strata with no --stratum_map
    ever passed (the map only ROUTES them), and that is only knowable once
    _resolve_selection_fits has opened the JSON."""
    opts = _opts("--c_mode", "selection",
                 "--per_pixel_completeness", "/depth.h5",
                 artifact=artifact)
    opts.selection_strata_by_catalog = [[(20.0, -20.3, 1.1),
                                         (20.5, -20.4, 1.1)]]
    _fatal_message(opts, check=_check_latent_selection_strata)
    assert "selection strata" in capsys.readouterr().out


def test_latent_with_non_galaxy_aware_model_refuses(capsys, artifact):
    opts = _opts("--c_mode", "aggregate",
                 "--per_pixel_completeness", "/depth.h5",
                 "--universe_model", "spectral_sirens",
                 artifact=artifact)
    _fatal_message(opts)
    assert "galaxy-aware" in capsys.readouterr().out


# ── Latent supplies a Q, so the aggregate-requires-Q refusal must not fire ─────

def test_aggregate_requires_q_is_satisfied_by_latent_mode(artifact):
    """--c_mode aggregate refuses a run with no Q table because a mean-one
    delta_g cannot encode a footprint.  Latent mode HAS a Q -- generated, and
    explicitly footprint-shaped (bit-zero logQ off-footprint) -- so the
    premise does not hold and the admitted latent+aggregate pairing must
    survive."""
    opts = _opts("--c_mode", "aggregate",
                 "--per_pixel_completeness", "/depth.h5",
                 artifact=artifact)
    _check_aggregate_requires_q(opts, (False,))          # no Q table anywhere


def test_aggregate_requires_q_still_fires_in_table_mode(capsys):
    opts = _opts("--c_mode", "aggregate", latent=False)
    with pytest.raises(SystemExit):
        _check_aggregate_requires_q(opts, (False,))
    assert "--c_mode aggregate requires an --lss_completion table" in (
        capsys.readouterr().out)


# ── The Q-table theta firewall is bypassed in latent mode ONLY ────────────────

def _stale_gaussian_fiducials():
    """A c_mode=selection Q-table stamp whose theta contradicts the run's fit.

    In table mode this is the exact configuration the firewall exists to
    refuse: the table's FIXED C_sel base was built at M0hat = -20.31 while the
    run's --selection_fit centers theta at -19.00.
    """
    return [{
        "path": "/q_selection.h5",
        "selection_family": "gaussian",
        "selection_m_lim": 20.0,
        "selection_M0hat": -20.31,
        "selection_sigma_M": 1.10,
    }]


def _fit_at(m0hat):
    return [{
        "family": "gaussian",
        "theta": {"m_lim": 20.0, "M0hat": m0hat, "sigma_M": 1.10},
        "k_corr_coeffs": (),
        "strata_fit": None,
        "stratum_map_sha256": "",
    }]


def test_qtable_theta_firewall_still_fires_in_table_mode(capsys):
    """Not weakened for everyone: the table path keeps the fail-closed check."""
    opts = _opts(latent=False)
    opts.selection_fits = _fit_at(-19.00)
    with pytest.raises(SystemExit):
        _check_selection_qtable_theta(_stale_gaussian_fiducials(), opts)
    out = capsys.readouterr().out
    assert "M0hat" in out and "--selection_fit" in out


def test_qtable_theta_firewall_is_bypassed_in_latent_mode(capsys, artifact):
    """Latent mode has no Q table (guard 6 refuses both the flag and the
    in-catalog group), so a Q-table theta stamp has no referent.  Fed the very
    stamp that is fatal above, the check must return silently."""
    opts = _opts("--c_mode", "selection",
                 "--per_pixel_completeness", "/depth.h5",
                 artifact=artifact)
    opts.selection_fits = _fit_at(-19.00)
    _check_selection_qtable_theta(_stale_gaussian_fiducials(), opts)
    assert capsys.readouterr().out == ""


# ── Old and new runs do not share a parameter space ───────────────────────────

def test_fingerprint_schema_version_is_bumped_for_the_b_miss_inversion():
    """b_miss is dropped with --use_lss off in table mode and SAMPLED (as
    b_GW) in latent mode, with an identical label, bounds and prior family --
    so the sampled block alone cannot separate the two parameter spaces.  The
    schema bump is what makes a pre-bump table checkpoint refuse a latent
    resume with a diff an operator can read."""
    from darksirens.inference.run_fingerprint import FINGERPRINT_SCHEMA_VERSION

    assert FINGERPRINT_SCHEMA_VERSION >= 3
    assert inference.FINGERPRINT_SCHEMA_VERSION == FINGERPRINT_SCHEMA_VERSION


def test_b_miss_is_dropped_in_table_mode_and_sampled_as_b_gw_in_latent_mode():
    """The PLAN 4.3 rule inversion, at the seam where the CLI arms it.

    ``inference/prior.py:_b_miss_rule`` asks one question -- "is the
    missing-galaxy modulation a function of b_miss?" -- and the CLI answers it
    with the ``use_lss`` argument.  Table mode answers ``opts.use_LSS`` (b_miss
    multiplies the all-zero delta_g dummy, so it is dropped rather than sampled
    as a phantom flat dimension).  Latent mode answers YES unconditionally,
    because b_miss IS b_GW and multiplies the artifact's field.  Measured here
    on the real parameter-space builder, not asserted from the comment.
    """
    from darksirens.inference.prior import build_parameter_space

    def _labels(b_miss_identified):
        return build_parameter_space(
            "powerlaw+peak", False, False, False,
            universe_model="dark_sirens",
            lss_completion_active=(False,),
            use_lss=b_miss_identified,
            c_mode="aggregate",
            n_catalogs=1,
        )[0]

    assert "b_miss" not in _labels(False)    # table mode, --use_lss off
    assert "b_miss" in _labels(True)         # latent mode: b_miss = b_GW


def test_cli_arms_the_b_miss_inversion_from_the_latent_predicate():
    """The call site must not pass a bare ``opts.use_LSS`` any more: it passes
    ``use_lss or latent``, so latent mode keeps b_GW in the sampled block."""
    import inspect

    src = inspect.getsource(inference._build_and_report_parameter_space)
    assert "_b_miss_identified = bool(getattr(opts, \"use_LSS\", False)) or _latent" in src
    assert "use_lss                = _b_miss_identified," in src


def test_lss_field_mode_is_semantic_and_reaches_the_digest():
    """The mode must not be excluded as an operational knob: two runs that
    differ only in --lss_field_mode target different posteriors."""
    from darksirens.inference.run_fingerprint import _NON_SEMANTIC_KEYS

    assert "lss_field_mode" not in _NON_SEMANTIC_KEYS
    assert "lss_field_artifact" not in _NON_SEMANTIC_KEYS


# ── Table mode is completely unaffected ──────────────────────────────────────

def test_table_mode_guard_is_a_no_op():
    """A default table namespace passes the guard with no artifact, any
    c_mode, --use_lss on, a Q table and a stratum map -- i.e. the guard adds
    no constraint whatsoever to the shipped path."""
    opts = _opts("--c_mode", "per_pixel",
                 "--use_lss", "true",
                 "--lss_completion", "/q_table.h5",
                 "--selection_fit", "/fit.json",
                 "--stratum_map", "/strata.h5",
                 latent=False)
    _check_latent_field_mode(opts)
    _check_latent_selection_strata(opts)


def test_settings_schema_version_is_stamped_in_meta():
    """settings.json is the file a human reads to decide whether two archived
    runs are comparable, so the convention version belongs there too."""
    import inspect

    src = inspect.getsource(inference._base_meta)
    assert "settings_schema_version" in src
