"""Products of ``darksirens_skymaps_to_samples`` must not fit a population.

A 3D skymap is marginalised over masses and spins, so the converter fills those
coordinates with draws from a broad surrogate proposal and sets ``p_pe`` equal
to that proposal.  Its own module docstring has always said so -- "you cannot
infer the mass population from mass-marginalised skymaps", and selection
consistency requires the same fixed mass/spin model in the injection integral --
but the requirement lived only in prose: the main CLI defaults
``--fix_population false`` and no loader looked at the file.  A run started that
way fits the surrogate proposal, breaks the numerator/denominator cancellation
of the mass/spin factors, and biases the cosmology riding on them, with nothing
in the log to say so.

The converter now stamps ``requires_fixed_population = True`` and
``darksirens.cli.inference`` checks it during configuration validation -- before
the data load, the run directory, and sampling.

Everything here exercises the validation helpers directly against tiny HDF5
files; no inference runs.  ``_fatal`` prints and raises SystemExit, so its
message is assertable from captured stdout.
"""
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("jax")
h5py = pytest.importorskip("h5py")

from darksirens.cli.inference import (  # noqa: E402
    _gw_file_requires_fixed_population,
    _validate_run_config,
    _validate_skymap_surrogate_population,
    build_parser,
)
from darksirens.cli.skymaps_to_samples import SOURCE_TAG  # noqa: E402


# ── tiny PE-file fixtures ──────────────────────────────────────────────────────


def _write(path, **attrs):
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "gwcat-1.0"
        f.attrs["nobs"] = 1
        f.attrs["nsamp"] = 2
        for k, v in attrs.items():
            f.attrs[k] = v
        f.create_dataset("dL", data=np.array([100.0, 200.0]))
    return str(path)


@pytest.fixture
def skymap_file(tmp_path):
    """What the converter writes today: both markers."""
    return _write(
        tmp_path / "skymap_pe.h5",
        source=SOURCE_TAG,
        requires_fixed_population=True,
    )


@pytest.fixture
def legacy_skymap_file(tmp_path):
    """Written before requires_fixed_population existed: source marker only."""
    return _write(tmp_path / "legacy_pe.h5", source=SOURCE_TAG)


@pytest.fixture
def normal_file(tmp_path):
    """An ordinary PE file: neither marker."""
    return _write(tmp_path / "normal_pe.h5")


def _opts(gw_path, **kwargs):
    fields = {
        "gw_path": gw_path,
        "fix_population": False,
        "allow_skymap_population": False,
    }
    fields.update(kwargs)
    return SimpleNamespace(**fields)


# ── marker detection ───────────────────────────────────────────────────────────


def test_detects_explicit_attr(skymap_file):
    assert _gw_file_requires_fixed_population(skymap_file) is True


def test_detects_legacy_source_only_file(legacy_skymap_file):
    """Files converted before the attr existed must still be caught."""
    assert _gw_file_requires_fixed_population(legacy_skymap_file) is True


def test_ignores_ordinary_pe_file(normal_file):
    assert _gw_file_requires_fixed_population(normal_file) is False


def test_ignores_other_producers(tmp_path):
    path = _write(tmp_path / "mock.h5", source="darksirens_mock_generator")
    assert _gw_file_requires_fixed_population(path) is False


@pytest.mark.parametrize("name", ["missing.h5", "not_hdf5.h5"])
def test_unreadable_file_is_not_a_marker_hit(tmp_path, name):
    """The probe never becomes a second format check; the loader owns that."""
    path = tmp_path / name
    if name == "not_hdf5.h5":
        path.write_text("this is not HDF5")
    assert _gw_file_requires_fixed_population(str(path)) is False


# ── the guard itself ───────────────────────────────────────────────────────────


def test_free_population_on_skymap_product_is_fatal(skymap_file, capsys):
    with pytest.raises(SystemExit):
        _validate_skymap_surrogate_population(_opts(skymap_file))
    out = capsys.readouterr().out
    assert "requires_fixed_population" in out
    assert "--fix_population true" in out
    assert "--allow_skymap_population" in out


def test_legacy_file_is_also_fatal(legacy_skymap_file):
    with pytest.raises(SystemExit):
        _validate_skymap_surrogate_population(_opts(legacy_skymap_file))


def test_fixed_population_is_accepted(skymap_file, capsys):
    _validate_skymap_surrogate_population(_opts(skymap_file, fix_population=True))
    assert "fixed" in capsys.readouterr().out


def test_override_downgrades_to_a_loud_warning(skymap_file, capsys):
    _validate_skymap_surrogate_population(
        _opts(skymap_file, allow_skymap_population=True)
    )
    out = capsys.readouterr().out
    assert "--allow_skymap_population" in out
    assert "NOT a measurement" in out


def test_ordinary_file_is_silent(normal_file, capsys):
    _validate_skymap_surrogate_population(_opts(normal_file))
    assert capsys.readouterr().out == ""


def test_flow_runs_are_unaffected():
    """--gw_flows_path runs have no PE file to inspect."""
    _validate_skymap_surrogate_population(_opts(None))


# ── wiring: the guard is reached from _validate_run_config ─────────────────────


def _parse(gw_path, extra=()):
    argv = [
        "--gw_path", gw_path,
        "--gwselection_path", "sel.h5",
        "--sampler", "tinyns",
        *extra,
    ]
    return build_parser().parse_args(argv)


def test_validate_run_config_rejects_free_population(skymap_file, capsys):
    with pytest.raises(SystemExit):
        _validate_run_config(_parse(skymap_file))
    assert "darksirens_skymaps_to_samples" in capsys.readouterr().out


def test_validate_run_config_accepts_fixed_population(skymap_file, capsys):
    _validate_run_config(_parse(skymap_file, ["--fix_population", "true"]))
    assert "Configuration is valid." in capsys.readouterr().out


def test_validate_run_config_accepts_the_override(skymap_file, capsys):
    _validate_run_config(_parse(skymap_file, ["--allow_skymap_population"]))
    out = capsys.readouterr().out
    assert "Configuration is valid." in out
    assert "Proceeding anyway" in out


def test_validate_run_config_ignores_normal_files(normal_file, capsys):
    _validate_run_config(_parse(normal_file))
    out = capsys.readouterr().out
    assert "Configuration is valid." in out
    assert "skymap" not in out


def test_override_does_not_move_the_resume_fingerprint(normal_file):
    """The override is a start-time gate, not part of the run's target.

    Its effect on the target is carried entirely by ``fix_population`` (which is
    semantic), so folding it into the digest would only invalidate the
    checkpoints of every run that predates the flag.
    """
    from darksirens.inference.run_fingerprint import build_run_fingerprint

    common = dict(
        labels=["H0"], lower_bound=[20.0], upper_bound=[140.0],
        prior_kinds=[("uniform", None, None)],
    )
    off = build_run_fingerprint(_parse(normal_file), **common)
    on = build_run_fingerprint(
        _parse(normal_file, ["--allow_skymap_population"]), **common
    )
    assert off["digest"] == on["digest"]


def test_override_flag_defaults_to_false_and_takes_a_bool(normal_file):
    assert _parse(normal_file).allow_skymap_population is False
    assert _parse(normal_file, ["--allow_skymap_population"]).allow_skymap_population is True
    assert (
        _parse(normal_file, ["--allow_skymap_population", "true"])
        .allow_skymap_population
        is True
    )
