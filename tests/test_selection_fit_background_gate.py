"""The magnitude-selection fit x sampled-background gate (PHY-04).

``C_sel(z; theta)`` is built from ``m_lim - M0hat - DM(z; H0=100)``.  The
h-scaled zero point absorbs ``H0`` EXACTLY (``dL`` is exactly proportional to
``1/H0``), which is the whole H0 firewall.  It does NOT absorb ``Om0``/``w0``/
``wa``: those change the SHAPE of ``DM(z)`` and only its z-independent part is
absorbable by a fitted zero point.  ``darksirens/redshift/selection.py``
measures the residual at ~0.1 mag across a z = 0.05-0.5 catalog against a
Laplace fit sd of ~0.02 mag.

``_validate_fit_background`` already refuses a fit MEASURED away from the
package fiducial.  This gate covers the other direction: a fit measured at the
fiducial consumed by a run that SAMPLES the background.
"""

import types

import pytest

from darksirens.cli.inference import _check_selection_fit_background

_FIT = {"theta": {"M0hat": -20.3, "sigma_M": 0.5}, "family": "gaussian"}

_SAMPLED = ["H0", "Om0", "w0", "wa", "log10n0", "M0hat"]
_FIXED_BACKGROUND = ["H0", "log10n0", "M0hat"]


def _opts(**kw):
    base = dict(
        c_mode="selection",
        n_catalogs=1,
        selection_fits=[_FIT],
        allow_selection_fit_free_background=False,
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.mark.parametrize("label", ["Om0", "w0", "wa"])
def test_refuses_a_fit_while_any_background_parameter_is_sampled(label):
    """Fixing only w0/wa is not enough while Om0 is free, and vice versa."""
    with pytest.raises(SystemExit) as exc:
        _check_selection_fit_background(_opts(), ["H0", label, "log10n0"])
    assert exc.value.code == 1


def test_accepts_a_fit_at_a_fully_fixed_background():
    """--fix_cosmology leaves no Om0/w0/wa label: the anchored zero point and
    the run then share ONE background, which is the regime the fit is valid in."""
    assert _check_selection_fit_background(_opts(), _FIXED_BACKGROUND) is None


def test_no_fit_is_the_wide_open_ablation_and_is_not_gated():
    """Without a fit, theta samples flat inside its truncation bounds -- there
    is no imported zero point to be anchored at the wrong background."""
    assert _check_selection_fit_background(
        _opts(selection_fits=[None]), _SAMPLED) is None
    assert _check_selection_fit_background(
        _opts(selection_fits=None), _SAMPLED) is None


@pytest.mark.parametrize("c_mode", ["per_pixel", "aggregate"])
def test_other_completeness_modes_never_reach_the_gate(c_mode):
    """Only c_mode='selection' consumes the parametric magnitude curve."""
    assert _check_selection_fit_background(
        _opts(c_mode=c_mode), _SAMPLED) is None


def test_a_k2_mixture_is_gated_on_the_anchored_catalog(capsys):
    """One anchored catalog out of two is enough: its budget carries the shape."""
    with pytest.raises(SystemExit):
        _check_selection_fit_background(
            _opts(n_catalogs=2, selection_fits=[None, _FIT]), _SAMPLED)
    out = capsys.readouterr().out
    assert "catalog(s) [2]" in out, out


def test_the_refusal_names_the_measured_residual_and_the_way_out(capsys):
    with pytest.raises(SystemExit):
        _check_selection_fit_background(_opts(), _SAMPLED)
    out = capsys.readouterr().out
    assert "--fix_cosmology" in out
    assert "--fix_de" in out          # explicitly NOT sufficient
    assert "0.1 mag" in out
    assert "--allow_selection_fit_free_background" in out


def test_the_override_warns_loudly_and_proceeds(capsys):
    assert _check_selection_fit_background(
        _opts(allow_selection_fit_free_background=True), _SAMPLED) is None
    out = capsys.readouterr().out
    assert "allow_selection_fit_free_background" in out
    assert "artificial information" in out
