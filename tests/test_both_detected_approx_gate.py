"""PHY-11: the both-detected pair approximation must not be an implicit default
when lensing RATES are being inferred.

``darksirens.likelihood.cluster_selection`` states it plainly: treating every
both-detected pair as identified is an UPPER BOUND on the true pair-detection
probability, it OVERestimates mu_sel^(2), and the inferred optical-depth
parameters are therefore BIASED LOW.  The API fallback is nevertheless
``log_p_tag = 0`` and the CLI default is ``--pair_tag_model constant
--pair_tag_constant 1.0``, so ``--fix_lens_rate false`` used to sample exactly
the quantities the approximation biases, silently.

The numerics are unchanged.  What changes is that the run is refused unless the
user either supplies a calibrated pair-ID efficiency or passes
``--allow_both_detected_approx true``, and that either way the resolved state is
stamped into settings.json and results.hdf5.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import darksirens.cli.inference_lensing as cli

from tests.test_lensing_cli_defects import _lensing_opts, _run_save_phase


class _FakeLensed:
    """Minimal stand-in for LensedInjectionSet: the resolver reads m1_src for
    the fallback length and log_p_tag_per_source for a legacy embedded table."""

    def __init__(self, n=4, log_p_tag=None):
        self.m1_src = np.full(n, 30.0)
        if log_p_tag is not None:
            self.log_p_tag_per_source = np.asarray(log_p_tag, dtype=float)


def _opts(fix_lens_rate, allow=False):
    extra = ["--fix_lens_rate", "false" if not fix_lens_rate else "true"]
    if allow:
        extra += ["--allow_both_detected_approx", "true"]
    opts = _lensing_opts(*extra)
    cli._resolve_lensing_run_config(opts)
    return opts


# ---------------------------------------------------------------------------
# refusal
# ---------------------------------------------------------------------------

def test_rate_inference_under_p_tag_one_is_refused():
    opts = _opts(fix_lens_rate=False)
    with pytest.raises(SystemExit) as exc:
        cli._pair_tag_log_probs_from_options(opts, _FakeLensed())
    msg = str(exc.value)
    assert "both-detected" in msg
    assert "BIASED LOW" in msg
    assert "--allow_both_detected_approx" in msg


def test_refusal_also_fires_on_an_embedded_all_ones_p_tag_table():
    """A campaign carrying p_tag_per_source = 1 is the same approximation
    wearing a dataset."""
    opts = _opts(fix_lens_rate=False)
    with pytest.raises(SystemExit, match="both-detected"):
        cli._pair_tag_log_probs_from_options(
            opts, _FakeLensed(log_p_tag=np.zeros(4))
        )


def test_refusal_fires_for_a_model_that_resolves_to_one_everywhere():
    """Gate on the RESOLVED values, not on the flags typed: constant=1.0 with a
    nonzero perturb_logit still routes through the model branch."""
    opts = _opts(fix_lens_rate=False)
    opts.pair_tag_model = "constant"
    opts.pair_tag_constant = 1.0
    opts.pair_tag_perturb_logit = 0.0

    resolved = []

    class _AllOnes:
        kind = "constant"
        required_fields = ()

        def log_probability(self, **fields):
            resolved.append(fields)
            return np.zeros(4)

    original = cli.make_pair_tag_selection_model
    cli.make_pair_tag_selection_model = lambda *a, **k: _AllOnes()
    try:
        # force the model branch rather than the legacy shortcut
        opts.pair_tag_constant = 0.999999
        with pytest.raises(SystemExit, match="both-detected"):
            cli._pair_tag_log_probs_from_options(opts, _FakeLensed())
    finally:
        cli.make_pair_tag_selection_model = original


# ---------------------------------------------------------------------------
# the ways through
# ---------------------------------------------------------------------------

def test_fixed_lens_rate_is_not_gated():
    """The bias lands on the rate parameters; with them fixed there is nothing
    to bias, and this is the CLI default arm."""
    opts = _opts(fix_lens_rate=True)
    out = cli._pair_tag_log_probs_from_options(opts, _FakeLensed())
    assert np.all(np.asarray(out) == 0.0)
    assert opts.pair_tag_both_detected_approx is True


def test_acknowledged_rate_inference_proceeds():
    opts = _opts(fix_lens_rate=False, allow=True)
    out = cli._pair_tag_log_probs_from_options(opts, _FakeLensed())
    assert np.all(np.asarray(out) == 0.0)
    assert opts.pair_tag_both_detected_approx is True
    assert opts.allow_both_detected_approx is True


def test_a_calibrated_p_tag_needs_no_acknowledgement():
    """The real way out: a campaign whose pair-ID efficiency is below one."""
    opts = _opts(fix_lens_rate=False)
    calibrated = np.log(np.array([0.9, 0.7, 0.95, 0.6]))
    out = cli._pair_tag_log_probs_from_options(
        opts, _FakeLensed(log_p_tag=calibrated)
    )
    np.testing.assert_allclose(np.asarray(out), calibrated)
    assert opts.pair_tag_both_detected_approx is False


def test_no_lensed_channel_is_not_gated():
    opts = _opts(fix_lens_rate=False)
    assert np.asarray(cli._pair_tag_log_probs_from_options(opts, None)).size == 0
    assert opts.pair_tag_both_detected_approx is False


# ---------------------------------------------------------------------------
# the stamp
# ---------------------------------------------------------------------------

def test_the_acknowledgement_is_stamped_into_settings_and_hdf5(tmp_path):
    opts = _opts(fix_lens_rate=False, allow=True)
    cli._pair_tag_log_probs_from_options(opts, _FakeLensed())
    attrs, settings, _lo, _hi = _run_save_phase(tmp_path, opts=opts)
    for record in (attrs, settings):
        assert bool(record["pair_tag_both_detected_approx"])
        assert bool(record["allow_both_detected_approx"])
        assert bool(record["lens_rate_inferred"])


def test_a_calibrated_run_is_stamped_as_not_approximate(tmp_path):
    opts = _opts(fix_lens_rate=False)
    cli._pair_tag_log_probs_from_options(
        opts, _FakeLensed(log_p_tag=np.log(np.full(4, 0.8)))
    )
    attrs, settings, _lo, _hi = _run_save_phase(tmp_path, opts=opts)
    for record in (attrs, settings):
        assert not bool(record["pair_tag_both_detected_approx"])
        assert not bool(record["allow_both_detected_approx"])
        assert bool(record["lens_rate_inferred"])


def test_the_flag_defaults_to_false_and_takes_a_bool():
    assert _lensing_opts().allow_both_detected_approx is False
    assert _lensing_opts(
        "--allow_both_detected_approx", "true"
    ).allow_both_detected_approx is True


# ---------------------------------------------------------------------------
# the workflows that sample the rate must still be runnable
# ---------------------------------------------------------------------------

def test_the_evidence_validation_rate_recovery_case_acknowledges(tmp_path):
    """That script's lens_rate_recovery case samples log10_tau_A under the
    default constant p_tag = 1; without the acknowledgement it would now die
    inside the CLI instead of validating anything."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rlev",
        HERE.parent / "scripts" / "mock_lensing"
        / "run_lensing_evidence_validation.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cfg = {"pe_max": 1, "n_universe": 1, "n_sing": 1, "n_pair": 1, "nsamp": 1,
           "n_unlensed_inj": 1, "n_lensed_inj": 1}
    sampled = mod._cli_cmd(
        tmp_path, tmp_path, cluster_mode="j2", partition=None, sampler="tinyns",
        nlive=8, dlogz=50.0, seed=1, cfg=cfg, fix_lens_rate=False,
    )
    assert "--allow_both_detected_approx" in sampled
    assert sampled[sampled.index("--allow_both_detected_approx") + 1] == "true"

    fixed = mod._cli_cmd(
        tmp_path, tmp_path, cluster_mode="j2", partition=None, sampler="tinyns",
        nlive=8, dlogz=50.0, seed=1, cfg=cfg, fix_lens_rate=True,
    )
    assert "--allow_both_detected_approx" not in fixed

    calibrated = mod._cli_cmd(
        tmp_path, tmp_path, cluster_mode="j2", partition=None, sampler="tinyns",
        nlive=8, dlogz=50.0, seed=1, cfg=cfg, fix_lens_rate=False,
        pair_tag_model="snr_sky",
    )
    assert "--allow_both_detected_approx" not in calibrated


def test_the_gate_helper_is_pure_bookkeeping_for_an_empty_channel():
    opts = SimpleNamespace(fix_lens_rate=False, allow_both_detected_approx=False)
    cli._gate_both_detected_approximation(opts, np.zeros(0))
    assert opts.pair_tag_both_detected_approx is False
