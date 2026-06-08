import re
import sys
import types

import numpy as np
import pytest


if "tinygp" not in sys.modules:
    tinygp_stub = types.ModuleType("tinygp")

    class _GaussianProcessStub:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("tinygp is required to evaluate GP population models")

    class _KernelsStub:
        class Matern52:
            def __init__(self, *args, **kwargs):
                pass

            def __rmul__(self, other):
                return self

    tinygp_stub.GaussianProcess = _GaussianProcessStub
    tinygp_stub.kernels = _KernelsStub()
    sys.modules["tinygp"] = tinygp_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")

    def _tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []

    tqdm_stub.tqdm = _tqdm
    sys.modules["tqdm"] = tqdm_stub

if "gwdistributions" not in sys.modules:
    gwdistributions_stub = types.ModuleType("gwdistributions")
    distributions_stub = types.ModuleType("gwdistributions.distributions")
    spin_stub = types.ModuleType("gwdistributions.distributions.spin")

    class _SpinPriorStub:
        def _init_values(self, *args, **kwargs):
            pass

    spin_stub.IsotropicUniformMagnitudeChiEffGivenComponentMass = _SpinPriorStub
    sys.modules["gwdistributions"] = gwdistributions_stub
    sys.modules["gwdistributions.distributions"] = distributions_stub
    sys.modules["gwdistributions.distributions.spin"] = spin_stub


if "gwcat" not in sys.modules:
    gwcat_stub = types.ModuleType("gwcat")
    spin_stub = types.ModuleType("gwcat.spin")

    def _chi_eff_prior_logprob(*args, **kwargs):
        return 0.0

    spin_stub.chi_eff_prior_logprob = _chi_eff_prior_logprob
    sys.modules["gwcat"] = gwcat_stub
    sys.modules["gwcat.spin"] = spin_stub

if "seaborn" not in sys.modules:
    seaborn_stub = types.ModuleType("seaborn")

    def _color_palette(*args, **kwargs):
        return ["C0", "C1", "C2", "C3", "C4"]

    seaborn_stub.color_palette = _color_palette
    seaborn_stub.set_context = lambda *args, **kwargs: None
    seaborn_stub.set_style = lambda *args, **kwargs: None
    sys.modules["seaborn"] = seaborn_stub

from darksirens.tool.darksirens_inference import _print_parameter_table


def _render_table(
    capsys, *, fix_cosmology=False, fix_population=False, fix_survey=False
):
    _print_parameter_table(
        labels=["sampled"],
        lower_bound=[0.0],
        upper_bound=[1.0],
        fixed_parameter_values={},
        prior_overrides={},
        fixed_parameter_statuses={},
        fix_cosmology=fix_cosmology,
        fix_population=fix_population,
        fix_survey=fix_survey,
        pop_params_fid=np.asarray([1.25, -3.5, 7.0]),
        pop_labels_all=["pop_a", "pop_b", "pop_c"],
    )
    return capsys.readouterr().out


@pytest.mark.parametrize(
    ("fix_cosmology", "fix_population", "fix_survey", "expected"),
    [
        (False, False, False, None),
        (True, False, False, 4),
        (False, True, False, 3),
        (False, False, True, 6),
        (True, True, True, 13),
    ],
)
def test_parameter_table_block_fixed_count_logic(
    capsys, fix_cosmology, fix_population, fix_survey, expected
):
    output = _render_table(
        capsys,
        fix_cosmology=fix_cosmology,
        fix_population=fix_population,
        fix_survey=fix_survey,
    )

    match = re.search(r"Fixed \(block\)\s+(\d+)", output)
    if expected is None:
        assert match is None
    else:
        assert match is not None
        assert int(match.group(1)) == expected


def test_parameter_table_shows_block_fixed_fiducial_rows(capsys):
    output = _render_table(
        capsys,
        fix_cosmology=True,
        fix_population=True,
        fix_survey=True,
    )

    assert "Sampled parameters" in output
    assert "Individually fixed parameters" in output
    assert "Block-fixed parameters" in output

    assert "[cosmology]" in output
    assert "H0" in output and "67.74" in output
    assert "Om0" in output and "0.3075" in output
    assert "w0" in output and "-1" in output
    assert "wa" in output and "0" in output

    assert "[population]" in output
    pop_fiducials = {"pop_a": "1.25", "pop_b": "-3.5", "pop_c": "7"}
    for label, value in pop_fiducials.items():
        assert label in output
        assert value in output

    assert "[survey]" in output
    for label, value in {
        "log10n0": "-2",
        "z50": "1",
        "w": "0.5",
        "delta": "0",
        "b_miss": "1",
        "alpha_miss": "0.5",
    }.items():
        assert label in output
        assert value in output
