"""Matched WL event/selection observation models (review finding P1-08).

The per-event (numerator) weak-lensing treatment comes from ``--wl_backend``;
the hierarchical selection normalization (denominator) from
``--wl_selection``.  The old defaults paired a lognormal event model with
standard selection, so every default lensing run normalized its hierarchy
under a different observation model than its numerator.  ``--wl_selection``
now defaults to 'auto' (resolve to the matched pair), an explicit mismatch is
fatal, and ``--allow_mismatched_wl_selection`` is the stamped ablation
override.
"""
import pytest

pytest.importorskip("jax")

from darksirens.cli import inference_lensing as lens_cli


def _opts(*extra):
    argv = [
        "--gw_path", "g.h5", "--gwselection_path", "s.h5",
        "--sampler", "tinyns", *extra,
    ]
    return lens_cli.build_parser().parse_args(argv)


def test_default_is_auto_and_resolves_to_the_matched_pair():
    opts = _opts()
    assert opts.wl_selection == "auto"
    assert opts.wl_backend == "lognormal"
    lens_cli._resolve_wl_selection(opts)
    assert opts.wl_selection == "wl_lognormal"
    assert opts.wl_selection_requested == "auto"


def test_auto_with_disabled_backend_resolves_to_standard():
    opts = _opts("--wl_backend", "disabled")
    lens_cli._resolve_wl_selection(opts)
    assert opts.wl_selection == "standard"


def test_auto_refuses_tabulated_backend():
    """No matched selection integral exists for the tabulated event model."""
    opts = _opts("--wl_backend", "tabulated",
                 "--lensing_wl_table_path", "/t.h5")
    with pytest.raises(SystemExit, match="no matched selection integral"):
        lens_cli._resolve_wl_selection(opts)


def test_explicit_standard_under_lognormal_events_is_fatal():
    opts = _opts("--wl_selection", "standard")  # wl_a default 4e-3 != 0
    with pytest.raises(SystemExit, match="different observation model"):
        lens_cli._resolve_wl_selection(opts)


def test_ablation_flag_downgrades_to_a_loud_warning():
    opts = _opts("--wl_selection", "standard",
                 "--allow_mismatched_wl_selection")
    with pytest.warns(RuntimeWarning, match="MISMATCHED WL"):
        lens_cli._resolve_wl_selection(opts)
    assert opts.wl_selection == "standard"
    assert opts.allow_mismatched_wl_selection is True  # -> settings.json


def test_wl_a_zero_makes_standard_exact_not_mismatched():
    """wl_a = 0 collapses the lognormal kernel to a delta function; standard
    selection is then identical, so no flag is needed."""
    opts = _opts("--wl_selection", "standard", "--lensing_wl_a", "0.0")
    lens_cli._resolve_wl_selection(opts)
    assert opts.wl_selection == "standard"


def test_wl_lognormal_needs_the_lognormal_backend():
    opts = _opts("--wl_selection", "wl_lognormal", "--wl_backend", "disabled")
    with pytest.raises(SystemExit, match="needs --wl_backend lognormal"):
        lens_cli._resolve_wl_selection(opts)


def test_matched_explicit_pairs_pass_untouched():
    opts = _opts("--wl_selection", "wl_lognormal")
    lens_cli._resolve_wl_selection(opts)
    assert opts.wl_selection == "wl_lognormal"

    opts = _opts("--wl_selection", "standard", "--wl_backend", "disabled")
    lens_cli._resolve_wl_selection(opts)
    assert opts.wl_selection == "standard"
