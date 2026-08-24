# test_ultra_fp_k2_refusal.py
"""The K >= 2 --per_pixel_completeness refusal must actually be reachable.

``load_all_data`` returns from its multitracer branch BEFORE
``attach_selection_fraction_inputs``, so the documented "K=1 only" refusal in
that attach step never fired for a K >= 2 mixture: the flag was accepted,
hashed into the run fingerprint and recorded in settings.json while the
likelihood ran entirely without f_p -- masked in provenance, unmasked in fact
(the CLI help promises "refuses ... K>=2").  Worse, the per-bundle S-3
footprint guard steered the operator who had already passed the map toward
--allow_unmasked_footprint.  These tests pin the refusal at every seam:
the data loader's multitracer branch, the CLI pre-load validation, and the
guard message's K-aware advice.
"""

from types import SimpleNamespace

import numpy as np
import pytest


def test_load_all_data_refuses_per_pixel_completeness_at_k2():
    # The refusal must fire BEFORE any file is opened: the nonexistent map
    # path proves the pre-load placement (any load attempt would die on the
    # missing GW/survey inputs first, with a different error).
    from darksirens.inference.data import load_all_data

    opts = SimpleNamespace(
        n_catalogs=2,
        per_pixel_completeness="/nonexistent/mth_map.h5",
        universe_model="dark_sirens",
    )
    with pytest.raises(NotImplementedError, match="K=1"):
        load_all_data(opts)


def test_multitracer_refusal_helper_is_inert_where_it_should_be():
    from darksirens.inference.loaders import (
        refuse_per_pixel_completeness_for_multitracer,
    )

    # K=1 with the flag: allowed (the attach step handles it downstream).
    refuse_per_pixel_completeness_for_multitracer(
        SimpleNamespace(n_catalogs=1, per_pixel_completeness="m.h5"))
    # K>=2 without the flag: nothing to refuse.
    refuse_per_pixel_completeness_for_multitracer(
        SimpleNamespace(n_catalogs=2, per_pixel_completeness=None))
    # Missing n_catalogs defaults to 1, like every consumer of the option.
    refuse_per_pixel_completeness_for_multitracer(
        SimpleNamespace(per_pixel_completeness="m.h5"))
    with pytest.raises(NotImplementedError, match="K=1"):
        refuse_per_pixel_completeness_for_multitracer(
            SimpleNamespace(n_catalogs=2, per_pixel_completeness="m.h5"))


def test_cli_validation_dies_before_the_load(capsys):
    # _validate_multitracer_config runs in main() before _load_and_report_data,
    # so a CLI run with the unsupported combination costs a second, not a
    # multi-catalog load -- and never writes a settings.json claiming a mask.
    from darksirens.cli.inference import _validate_multitracer_config

    opts = SimpleNamespace(
        n_catalogs=2,
        per_pixel_completeness="m.h5",
        universe_model="dark_sirens",
        catalog_sky_weighting="field",
        drop_full_catalog=False,
        mark_model="none",
    )
    with pytest.raises(SystemExit):
        _validate_multitracer_config(opts)
    assert "--per_pixel_completeness is K=1 only" in capsys.readouterr().out

    # ... and stays quiet for the same mixture without the flag.
    opts.per_pixel_completeness = None
    _validate_multitracer_config(opts)


def test_footprint_guard_does_not_advise_the_flag_at_k2():
    # The S-3 guard fires per bundle at K>=2; advising the operator to pass
    # --per_pixel_completeness on a path that refuses it steers them into
    # --allow_unmasked_footprint and an unmasked-in-fact run.
    from darksirens.inference.loaders import guard_unmasked_footprint_counts

    rng = np.random.default_rng(4)
    ngals = rng.poisson(60, size=3072)
    ngals[:1200] = 0

    with pytest.raises(ValueError) as e1:
        guard_unmasked_footprint_counts(
            SimpleNamespace(c_mode="selection", n_catalogs=1), ngals)
    assert "--per_pixel_completeness <mth_map.h5>" in str(e1.value)

    with pytest.raises(ValueError) as e2:
        guard_unmasked_footprint_counts(
            SimpleNamespace(c_mode="selection", n_catalogs=2), ngals,
            label="tracer_B.h5")
    msg = str(e2.value)
    assert "not supported for a K>=2 mixture" in msg
    assert "<mth_map.h5>" not in msg
    assert "--allow_unmasked_footprint" in msg
