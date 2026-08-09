"""Per-catalog ``c_mode='selection'``: one magnitude fit per catalog, one
anchored theta prior per catalog, one theta-provenance comparison per catalog.
Pins the K=1 path bit-identical and the fail-closed edges of the new
all-or-nothing anchoring rule.
"""

import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from darksirens.inference.parameters import (
    ParameterDecoder,
    build_parameter_decoder,
)
from darksirens.inference.prior import build_parameter_space

_KW = dict(universe_model="dark_sirens", use_lss=False)
_ANCHOR1 = {"M0hat": (-20.19, 0.02), "sigma_M": (1.0, 0.015)}
_ANCHOR2 = {"M0hat_c2": (-21.30, 0.04), "sigma_M_c2": (0.7, 0.011)}


# ── prior / parameter space ────────────────────────────────────────


def test_k2_selection_emits_anchored_c2_labels():
    labels, _lo, _hi, *rest = build_parameter_space(
        "powerlaw+peak", True, True, False, n_catalogs=2, c_mode="selection",
        selection_prior={**_ANCHOR1, **_ANCHOR2}, **_KW)
    kinds = dict(zip(labels, rest[8]))
    assert {"M0hat", "sigma_M", "M0hat_c2", "sigma_M_c2"} <= set(labels)
    # Deliberately different values per catalog: proves no cross-wiring.
    assert kinds["M0hat"] == ("normal", -20.19, 0.02)
    assert kinds["M0hat_c2"] == ("normal", -21.30, 0.04)
    assert kinds["sigma_M"] == ("normal", 1.0, 0.015)
    assert kinds["sigma_M_c2"] == ("normal", 0.7, 0.011)
    # m_lim is a pinned truncation datum for every catalog, never sampled.
    assert "m_lim" not in labels and "m_lim_c2" not in labels


def test_k2_selection_missing_second_fit_is_fatal():
    with pytest.raises(ValueError, match="M0hat_c2") as exc:
        build_parameter_space("powerlaw+peak", True, True, False,
                              n_catalogs=2, c_mode="selection",
                              selection_prior=dict(_ANCHOR1), **_KW)
    assert "--selection_fit" in str(exc.value)


def test_k2_selection_no_fit_at_all_is_the_flat_ablation():
    labels, _lo, _hi, *rest = build_parameter_space(
        "powerlaw+peak", True, True, False, n_catalogs=2, c_mode="selection",
        selection_prior=None, **_KW)
    kinds = dict(zip(labels, rest[8]))
    for lbl in ("M0hat", "sigma_M", "M0hat_c2", "sigma_M_c2"):
        assert kinds[lbl] == ("uniform", None, None)


def test_k3_partial_anchor_names_every_unanchored_label():
    prior = {**_ANCHOR1,
             "M0hat_c3": (-21.0, 0.03), "sigma_M_c3": (0.8, 0.01)}
    with pytest.raises(ValueError) as exc:
        build_parameter_space("powerlaw+peak", True, True, False,
                              n_catalogs=3, c_mode="selection",
                              selection_prior=prior, **_KW)
    assert "M0hat_c2" in str(exc.value)
    assert "sigma_M_c2" in str(exc.value)


def test_mixed_c_mode_requires_fits_only_for_selection_catalogs():
    kw = dict(n_catalogs=2, c_mode=("selection", "aggregate"), **_KW)
    labels = build_parameter_space("powerlaw+peak", True, True, False,
                                   selection_prior=dict(_ANCHOR1), **kw)[0]
    assert "M0hat" in labels
    assert "M0hat_c2" not in labels
    with pytest.raises(ValueError, match="not sampled"):
        build_parameter_space("powerlaw+peak", True, True, False,
                              selection_prior={"M0hat_c2": (-21.0, 0.03)},
                              **kw)


def test_m_lim_pinned_per_catalog_is_silent_and_not_sampled():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labels, _lo, _hi, *_ = build_parameter_space(
            "powerlaw+peak", True, True, False, n_catalogs=2,
            c_mode="selection", selection_prior={**_ANCHOR1, **_ANCHOR2},
            fixed_parameter_values={"m_lim": 24.0, "m_lim_c2": 22.5}, **_KW)
    assert not caught
    assert not [lbl for lbl in labels if lbl.startswith("m_lim")]


def test_k1_selection_space_is_bit_identical():
    """The single-catalog call from test_completion_selection_mode is
    unchanged by the per-catalog plumbing."""
    res = build_parameter_space(
        "powerlaw+peak", True, True, False, c_mode="selection",
        selection_prior={"M0hat": (-20.19, 0.02), "sigma_M": (1.0, 0.015)},
        **_KW)
    kinds = dict(zip(res[0], res[11]))
    assert kinds["M0hat"] == ("normal", -20.19, 0.02)
    assert kinds["sigma_M"] == ("normal", 1.0, 0.015)
    # Same space as the flat call except for the two flipped kinds.
    flat = build_parameter_space("powerlaw+peak", True, True, False,
                                 c_mode="selection", **_KW)
    assert res[0] == flat[0]
    assert np.array_equal(res[1], flat[1])
    assert np.array_equal(res[2], flat[2])


# ── decoder ────────────────────────────────────────────────────────


def _decoder_opts(**extra):
    base = dict(pop_model="powerlaw+peak", fix_population=True,
                fix_cosmology=True, fix_survey=False, fix_de=True,
                universe_model="dark_sirens", use_LSS=False,
                n_catalogs=1, c_mode="selection")
    base.update(extra)
    return SimpleNamespace(**base)


def test_decoder_threads_per_catalog_kcorr_and_strata():
    dec = ParameterDecoder(
        sampled_labels=("M0hat", "M0hat_c2", "fcat_2"),
        fixed_parameter_values={},
        pop_labels=(), pop_params_fid=(),
        complete_empty_pixel_policy=0, c_mode=2, n_catalogs=2,
        k_corr_coeffs=((0.39, 0.11), None),
        selection_strata=(None, None),
    )
    surveys = dec.decode_mixture(np.array([-20.0, -21.0, 0.5]))[1]
    assert surveys[0].k_corr_coeffs == (0.39, 0.11)
    assert surveys[1].k_corr_coeffs is None


def test_decoder_rejects_legacy_flat_kcorr():
    with pytest.raises(ValueError, match="per catalog"):
        ParameterDecoder(
            sampled_labels=(), fixed_parameter_values={},
            pop_labels=(), pop_params_fid=(),
            complete_empty_pixel_policy=0, n_catalogs=2,
            k_corr_coeffs=(0.39, 0.11))


def test_decoder_rejects_legacy_flat_strata():
    with pytest.raises(ValueError, match="per catalog"):
        ParameterDecoder(
            sampled_labels=(), fixed_parameter_values={},
            pop_labels=(), pop_params_fid=(),
            complete_empty_pixel_policy=0, n_catalogs=2,
            selection_strata=((24.0, 0.0, 1.0), (23.0, 0.2, 1.1)))


@pytest.mark.parametrize("attr,value", [
    ("selection_fit_kcorr", (0.39,)),
    ("selection_strata_struct", ((24.0, 0.0, 1.0),)),
])
def test_build_decoder_rejects_retired_opts_attrs(attr, value):
    opts = _decoder_opts(**{attr: value})
    with pytest.raises(ValueError, match="retired"):
        build_parameter_decoder(opts, ())


def test_build_decoder_refuses_k2_stratified_selection():
    opts = _decoder_opts(
        n_catalogs=2,
        selection_strata_by_catalog=[((24.0, 0.0, 1.0), (23.0, 0.2, 1.1)),
                                     None])
    with pytest.raises(ValueError, match="single-catalog"):
        build_parameter_decoder(opts, ())


def test_decoder_k1_kcorr_reaches_survey_params():
    from darksirens.gw.populations import get_fixed_population_params

    opts = _decoder_opts(selection_kcorr_by_catalog=[(0.39, 0.11)])
    dec = build_parameter_decoder(
        opts, get_fixed_population_params("powerlaw+peak"))
    coord = np.zeros(len(dec.sampled_labels))
    assert dec.decode(coord)[1].k_corr_coeffs == (0.39, 0.11)


def test_percatalog_m_lim_reaches_each_survey_params():
    dec = ParameterDecoder(
        sampled_labels=("fcat_2",),
        fixed_parameter_values={"m_lim": 24.0, "m_lim_c2": 22.5},
        pop_labels=(), pop_params_fid=(),
        complete_empty_pixel_policy=0, c_mode=2, n_catalogs=2,
    )
    surveys = dec.decode_mixture(np.array([0.5]))[1]
    assert float(surveys[0].m_lim) == 24.0
    assert float(surveys[1].m_lim) == 22.5


# ── Q-table theta firewall (per catalog) ───────────────────────────


def _fit(catalog, m_lim, M0hat, sigma_M, kcorr=()):
    return {"catalog": catalog, "path": f"fit{catalog}.json",
            "theta": {"m_lim": m_lim, "M0hat": M0hat, "sigma_M": sigma_M},
            "k_corr_coeffs": list(kcorr), "strata_fit": None,
            "strata_struct": None, "stratum_map": None,
            "stratum_map_sha256": None, "prior": {}}


def _fid(path, m_lim, M0hat, sigma_M, kcorr=()):
    d = {"path": path, "selection_m_lim": m_lim, "selection_M0hat": M0hat,
         "selection_sigma_M": sigma_M}
    for j, c in enumerate(kcorr, start=1):
        d[f"selection_kcorr_c{j}"] = float(c)
    return d


def test_percatalog_firewall_catches_catalog2_mismatch():
    """Pre-PR-D both tables were compared against ONE fit, so a catalog-2
    table built on catalog 1's theta passed silently."""
    from darksirens.cli import inference as cli

    fits = [_fit(1, 21.0, -20.9, 0.9), _fit(2, 19.5, -21.4, 0.6)]
    fid1 = _fid("qt1.h5", 21.0, -20.9, 0.9)
    bad2 = _fid("qt2.h5", 21.0, -20.9, 0.9)     # catalog 1's theta
    with pytest.raises(SystemExit):
        cli._check_selection_qtable_theta(
            [fid1, bad2], SimpleNamespace(selection_fits=fits))

    good2 = _fid("qt2.h5", 19.5, -21.4, 0.6)
    cli._check_selection_qtable_theta(
        [fid1, good2], SimpleNamespace(selection_fits=fits))


def test_percatalog_firewall_kcorr_is_per_catalog():
    from darksirens.cli import inference as cli

    fits = [_fit(1, 21.0, -20.9, 0.9, (0.39, 0.11)),
            _fit(2, 19.5, -21.4, 0.6, (0.22,))]
    fid1 = _fid("qt1.h5", 21.0, -20.9, 0.9, (0.39, 0.11))
    bad2 = _fid("qt2.h5", 19.5, -21.4, 0.6, (0.39, 0.11))   # cat 1's template
    with pytest.raises(SystemExit):
        cli._check_selection_qtable_theta(
            [fid1, bad2], SimpleNamespace(selection_fits=fits))


def test_firewall_allows_q_on_one_catalog_only():
    from darksirens.cli import inference as cli

    fits = [_fit(1, 21.0, -20.9, 0.9), _fit(2, 19.5, -21.4, 0.6)]
    cli._check_selection_qtable_theta(
        [_fid("qt1.h5", 21.0, -20.9, 0.9), None],
        SimpleNamespace(selection_fits=fits))


def test_firewall_unanchored_catalog_warns_not_fatal(capsys):
    from darksirens.cli import inference as cli

    fits = [_fit(1, 21.0, -20.9, 0.9), None]
    cli._check_selection_qtable_theta(
        [_fid("qt1.h5", 21.0, -20.9, 0.9), _fid("qt2.h5", 19.5, -21.4, 0.6)],
        SimpleNamespace(selection_fits=fits))
    assert "catalog 2" in capsys.readouterr().out


def test_firewall_length_mismatch_is_fatal():
    from darksirens.cli import inference as cli

    with pytest.raises(SystemExit):
        cli._check_selection_qtable_theta(
            [_fid("qt1.h5", 21.0, -20.9, 0.9),
             _fid("qt2.h5", 19.5, -21.4, 0.6)],
            SimpleNamespace(selection_fits=[_fit(1, 21.0, -20.9, 0.9)]))


# ── CLI surface (pre-load guards, nonexistent paths) ───────────────


def _run(args):
    return subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference"] + args,
        capture_output=True, text=True)


_BASE_ARGS = [
    "--gw_path", "/nonexistent/gw.h5",
    "--gwselection_path", "/nonexistent/sel.h5",
    "--survey_path", "/nonexistent/catA.h5", "/nonexistent/catB.h5",
    "--c_mode", "selection",
    "--sampler", "tinyns",
]
_COUNT_GUARD = "comma-separated entries"


def test_cli_selection_fit_count_must_match_n_catalogs():
    r = _run(_BASE_ARGS + ["--selection_fit", "/nonexistent/fitA.json"])
    assert r.returncode != 0
    assert "selection_fit" in r.stdout
    assert _COUNT_GUARD in r.stdout
    assert "No such file" not in r.stdout and "No such file" not in r.stderr


def test_cli_two_selection_fits_pass_the_count_guard():
    r = _run(_BASE_ARGS + [
        "--selection_fit", "/nonexistent/fitA.json,/nonexistent/fitB.json"])
    assert r.returncode != 0          # the missing files bite later
    assert _COUNT_GUARD not in r.stdout


def test_cli_help_documents_per_catalog_selection_fit():
    r = subprocess.run(
        [sys.executable, "-m", "darksirens.cli.inference", "--help"],
        check=True, capture_output=True, text=True)
    assert "--selection_fit JSON[,JSON...]" in r.stdout
    assert "M0hat_c{k}" in r.stdout


def test_cli_stratum_map_refused_for_mixtures():
    r = _run(_BASE_ARGS + [
        "--selection_fit", "/nonexistent/fitA.json,/nonexistent/fitB.json",
        "--stratum_map", "/nonexistent/map.h5"])
    assert r.returncode != 0
    assert "single-catalog" in r.stdout
