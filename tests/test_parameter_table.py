import json
import re
import sys
import types
from types import SimpleNamespace

import h5py
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


# Stub gwcat ONLY when the real package is absent: keying on
# ``"gwcat" not in sys.modules`` shadowed an INSTALLED gwcat for every later
# test module in the same process (test_pdet_selection's crafted-pdraw test
# then received the float-returning stub chi_eff_prior_logprob and died with
# "'float' object is not subscriptable" whenever this file ran first).
try:
    import gwcat.spin  # noqa: F401  -- real package preferred over the stub
except ImportError:
    gwcat_stub = types.ModuleType("gwcat")
    spin_stub = types.ModuleType("gwcat.spin")

    def _chi_eff_prior_logprob(*args, **kwargs):
        return 0.0

    class _ChiEffPrior:
        """Shape of gwcat's ``ChiEffPrior`` as far as ``darksirens.gw.utils``
        checks it at import: the ``support`` attribute is the GW-03 convention
        guard (a prior without it is refused as pre-``-inf``)."""
        support = (-1.0, 1.0)

    spin_stub.chi_eff_prior_logprob = _chi_eff_prior_logprob
    spin_stub.ChiEffPrior = _ChiEffPrior
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

from darksirens.cli.inference import (
    _format_fixed_dark_energy_summary,
    _print_parameter_table,
    save_results_hdf5,
    save_settings_json,
)


def _render_table(
    capsys, *, fix_cosmology=False, fix_de=False, fix_population=False, fix_survey=False
):
    _print_parameter_table(
        labels=["sampled"],
        lower_bound=[0.0],
        upper_bound=[1.0],
        fixed_parameter_values={},
        prior_overrides={},
        fixed_parameter_statuses={},
        fix_cosmology=fix_cosmology,
        fix_de=fix_de,
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
        # The survey block prints the 13 SURVEY_FID rows (log10n0, z50, w,
        # delta, b_miss, alpha_miss, sigma_kde, the gaussian selection block
        # m_lim/M0hat/sigma_M and the schechter one
        # Mstar_hat/alpha/M_faint_offset); the footer count matches.
        (False, False, True, 13),
        (True, True, True, 20),
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




def test_parameter_table_dark_energy_block_fixed_count(capsys):
    output = _render_table(capsys, fix_de=True)

    match = re.search(r"Fixed \(block\)\s+(\d+)", output)
    assert match is not None
    assert int(match.group(1)) == 2
    assert "[dark energy]" in output
    assert "w0" in output
    assert "wa" in output
    assert "H0" not in output
    assert "Om0" not in output


def test_parameter_table_full_cosmology_supersedes_dark_energy_count(capsys):
    output = _render_table(capsys, fix_cosmology=True, fix_de=True)

    match = re.search(r"Fixed \(block\)\s+(\d+)", output)
    assert match is not None
    assert int(match.group(1)) == 4
    assert "[cosmology]" in output
    assert "[dark energy]" not in output


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

def _serialization_opts(**overrides):
    values = dict(
        pop_model="powerlaw+peak",
        universe_model="spectral_sirens",
        complete_empty_pixel_policy="zero",
        sampler="dynesty",
        fix_cosmology=False,
        fix_de=False,
        fix_population=False,
        fix_survey=False,
        gw_path="gw.h5",
        gwselection_path="sel.h5",
        survey_path=None,
        counterpart=None,
        nlive=10,
        dlogz=0.1,
        nuts_warmup=0,
        nuts_samples=0,
        nuts_chains=0,
        nuts_target_accept=0.8,
        nuts_max_tree_depth=10,
        nuts_init_tries=1,
        nuts_init_seed_offset=0,
        seed=123,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _meta():
    return {
        "n_events": 1,
        "n_samp_per_event": 2,
        "n_draw": 3,
        "total_runtime": "0:00:01",
        "sampling_runtime": "0:00:01",
        "timestamp": "2026-06-08T00:00:00",
    }


def test_dark_energy_summary_reports_block_fixed_values():
    opts = _serialization_opts(fix_de=True)

    assert _format_fixed_dark_energy_summary(opts, {}) == "yes (w0=-1, wa=0)"


def test_dark_energy_summary_reports_individually_fixed_values():
    opts = _serialization_opts()

    assert (
        _format_fixed_dark_energy_summary(opts, {"w0": -0.9})
        == "partial (w0=-0.9)"
    )
    assert (
        _format_fixed_dark_energy_summary(opts, {"w0": -0.9, "wa": 0.2})
        == "yes (w0=-0.9, wa=0.2)"
    )


def test_result_serialization_records_fixed_dark_energy_state(tmp_path):
    opts = _serialization_opts(fix_de=True)
    results = {"samples": np.zeros((2, 1))}

    path = save_results_hdf5(
        results,
        str(tmp_path),
        labels=["H0"],
        lower_bound=[20.0],
        upper_bound=[120.0],
        fixed_parameter_values={},
        prior_overrides={},
        opts=opts,
        meta=_meta(),
    )

    with h5py.File(path, "r") as f:
        assert bool(f.attrs["fixed_dark_energy"])
        assert bool(f.attrs["w0_fixed"])
        assert bool(f.attrs["wa_fixed"])
        assert f.attrs["fixed_w0"] == -1.0
        assert f.attrs["fixed_wa"] == 0.0


def test_settings_serialization_records_fixed_dark_energy_values(tmp_path):
    opts = _serialization_opts()

    path = save_settings_json(
        opts,
        str(tmp_path),
        labels=["H0", "Om0"],
        lower_bound=[20.0, 0.1],
        upper_bound=[120.0, 0.5],
        fixed_parameter_values={"w0": -0.8, "wa": 0.1},
        prior_overrides={},
        meta=_meta(),
    )

    with open(path) as f:
        settings = json.load(f)

    assert settings["fixed_dark_energy"] is True
    assert settings["w0_fixed"] is True
    assert settings["wa_fixed"] is True
    assert settings["dark_energy_fixed_values"] == {"w0": -0.8, "wa": 0.1}
