"""Per-catalog --validate_completion for a K>=2 mixture run
(cli/inference.py run_completion_validation_multitracer).

The K>=2 dry run reuses the single-catalog diagnostic verbatim per catalog:
full survey rows re-read host-side, paired with the per-sample pixel indices
the bundle loader stashes when --validate_completion is set, and one
``completion_validation_c{k}__<ts>.json`` written per catalog."""
import json
import os
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from darksirens.cli.inference import (
    run_completion_validation,
    run_completion_validation_multitracer,
)

NSIDE = 1
NPIX = 12
MAXGALS = 4


def _write_survey(path, seed, weight=1.0):
    rng = np.random.default_rng(seed)
    zgals = np.zeros((NPIX, MAXGALS))
    dzgals = np.full((NPIX, MAXGALS), 0.02)
    wgals = np.zeros((NPIX, MAXGALS))
    ngals = np.zeros(NPIX, dtype=np.int32)
    for pix in range(0, NPIX, 2):  # populate every other pixel
        n = int(rng.integers(2, MAXGALS + 1))
        zgals[pix, :n] = rng.uniform(0.05, 0.4, size=n)
        wgals[pix, :n] = weight
        ngals[pix] = n
    with h5py.File(path, "w") as f:
        f.attrs["nside"] = NSIDE
        f.create_dataset("zgals", data=zgals)
        f.create_dataset("dzgals", data=dzgals)
        f.create_dataset("wgals", data=wgals)
        f.create_dataset("ngals", data=ngals)


def _bundle(pixels_pe, pixels_sel):
    import healpy as hp

    return dict(
        apix=hp.nside2pixarea(NSIDE),
        n_pix_catalog=NPIX,
        pixels_pe_full=np.asarray(pixels_pe, dtype=np.int32),
        pixels_sel_full=np.asarray(pixels_sel, dtype=np.int32),
    )


@pytest.fixture
def two_catalog_setup(tmp_path):
    # The two catalogs' GW samples land in DIFFERENT pixels (per-catalog
    # ang2pix at each catalog's own nside does this in production): the
    # diagnostics' per-pixel entries are the catalog-dependent payload.
    paths = []
    for k, seed in enumerate((101, 202), start=1):
        p = tmp_path / f"survey_c{k}.h5"
        _write_survey(p, seed)
        paths.append(str(p))
    opts = SimpleNamespace(
        survey_paths=paths,
        n_catalogs=2,
        save_path=str(tmp_path / "out"),
        completion_validation_pixels=8,
    )
    data = dict(catalogs=[
        _bundle([0, 2, 4, 6, 0, 2], [0, 2, 4, 8, 10]),
        _bundle([1, 3, 5, 7, 1, 3], [1, 3, 5, 9, 11]),
    ])
    return opts, data


def test_writes_one_json_per_catalog(two_catalog_setup):
    opts, data = two_catalog_setup
    paths = run_completion_validation_multitracer(opts, data, {}, {})
    assert len(paths) == 2
    for k, p in enumerate(paths, start=1):
        assert os.path.exists(p)
        assert f"completion_validation_c{k}__" in os.path.basename(p)
        with open(p) as f:
            diag = json.load(f)
        assert "survey_values" in diag and "cosmology_values" in diag


def test_catalogs_get_independent_diagnostics(two_catalog_setup):
    """Each catalog is validated on its OWN per-sample pixel indices: the
    per-pixel diagnostic entries must reflect each bundle's pixel set, not a
    shared/broadcast one."""
    opts, data = two_catalog_setup
    p1, p2 = run_completion_validation_multitracer(opts, data, {}, {})
    with open(p1) as f:
        d1 = json.load(f)
    with open(p2) as f:
        d2 = json.load(f)
    pix1 = [e["global_pixel"] for e in d1["per_pixel"]]
    pix2 = [e["global_pixel"] for e in d2["per_pixel"]]
    assert pix1 == sorted(set([0, 2, 4, 6, 8, 10]))[:8]
    assert pix2 == sorted(set([1, 3, 5, 7, 9, 11]))[:8]


def test_missing_stashed_pixels_is_fatal(two_catalog_setup):
    opts, data = two_catalog_setup
    del data["catalogs"][1]["pixels_pe_full"]
    with pytest.raises(SystemExit):
        run_completion_validation_multitracer(opts, data, {}, {})


def test_single_catalog_json_name_unchanged(tmp_path):
    """The K=1 filename contract (no suffix) is untouched."""
    p = tmp_path / "survey.h5"
    _write_survey(p, 303)
    import healpy as hp
    from darksirens.inference.loaders import load_survey

    _, ngals, zgals, dzgals, wgals, _ = load_survey(str(p), to_device=False)
    opts = SimpleNamespace(
        save_path=str(tmp_path / "out"), completion_validation_pixels=8
    )
    data = dict(
        zgals_catalog=zgals, dzgals_catalog=dzgals, wgals_catalog=wgals,
        ngals_catalog=ngals,
        pixels_pe=np.array([0, 2, 4], dtype=np.int32),
        pixels_sel=np.array([0, 2, 6], dtype=np.int32),
        apix=hp.nside2pixarea(NSIDE),
        n_pix_catalog=NPIX,
    )
    path = run_completion_validation(opts, data, {}, {})
    assert os.path.basename(path).startswith("completion_validation__")
