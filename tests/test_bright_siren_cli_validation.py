"""Parse-time gates on the bright-siren counterpart options.

Both the counterpart redshift (``--counterpart RA DEC Z``, parsed in
darksirens/cli/common.py) and its width (``--counterpart_dz``, checked in
darksirens/cli/inference.py) used to be guarded by a bare ``<= 0``, which NaN
and +-inf pass.  argparse's ``type=float`` accepts the strings 'nan'/'inf', so
both reached ``norm.logpdf(z, counterpart_z, counterpart_dz)`` in
darksirens/redshift/prior.py and turned the bright-siren term into NaN or an
all--inf likelihood -- hours into a run, not at submission.

Parsed in-process (no subprocess): ``_fatal`` prints to stdout and raises
SystemExit, so the message is assertable from captured output.
"""
import pytest

pytest.importorskip("jax")

from darksirens.cli.common import parse_counterpart_arg
from darksirens.cli.inference import build_parser, _validate_run_config

# Enough to clear the "exactly one of" gates that precede the counterpart
# checks; neither path is opened by validation.
_BASE = ["--gw_path", "gw.h5", "--gwselection_path", "sel.h5", "--sampler", "tinyns"]


@pytest.mark.parametrize("z", ["nan", "-nan", "inf", "-inf", "0", "-0.1"])
def test_counterpart_redshift_must_be_finite_and_positive(z, capsys):
    with pytest.raises(SystemExit):
        parse_counterpart_arg(["0.5", "0.2", z])
    assert "redshift Z must be a finite positive number" in capsys.readouterr().out


def test_counterpart_accepts_ordinary_triplets():
    assert parse_counterpart_arg(["0.5", "0.2", "0.1"]) == ((0.5, 0.2, 0.1),)
    assert len(parse_counterpart_arg(["0.5", "0.2", "0.1", "1.0", "-0.3", "0.05"])) == 2


@pytest.mark.parametrize("dz", ["nan", "inf", "-inf", "0", "-1e-4"])
def test_counterpart_dz_must_be_finite_and_positive(dz, capsys):
    # ``--opt=value`` form: argparse reads a bare '-inf' as an option name.
    opts = build_parser().parse_args(_BASE + [f"--counterpart_dz={dz}"])
    with pytest.raises(SystemExit):
        _validate_run_config(opts)
    assert "--counterpart_dz must be a finite positive number." in capsys.readouterr().out


def test_valid_counterpart_dz_clears_validation(capsys):
    opts = build_parser().parse_args(_BASE + ["--counterpart_dz", "1e-3"])
    _validate_run_config(opts)
    assert "counterpart_dz" not in capsys.readouterr().out
