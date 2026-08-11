"""Per-galaxy property (gal_*) plumbing: pixelate -> load, co-indexed with zgals.

Galaxy properties (apparent magnitudes) ride the marks padding mechanism but
live in the separate GALPROP_DATASETS registry so they can never enroll a
catalog into the marked-host model.  The critical invariant is CO-INDEXING:
``load_survey`` z-sorts every row, and ``load_survey_galprops`` must apply the
identical permutation or property i no longer belongs to galaxy i.
"""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from darksirens.catalogs.io import (
    GALPROP_DATASETS,
    MARK_DATASETS,
    load_survey,
    load_survey_galprops,
    load_survey_marks,
)
from darksirens.cli import pixelate


def _write_survey(path, rng, n=400):
    """Raw survey file with a deterministic per-galaxy magnitude tag."""
    ra = rng.uniform(0.0, 360.0, n)
    dec = np.rad2deg(np.arcsin(rng.uniform(-1.0, 1.0, n)))
    z = rng.uniform(0.01, 0.4, n)
    # Bijective z -> mag map so co-indexing is checkable after any reordering.
    app_mag = 18.0 + 10.0 * z
    with h5py.File(path, "w") as f:
        f.create_dataset("TARGET_RA", data=ra)
        f.create_dataset("TARGET_DEC", data=dec)
        f.create_dataset("Z", data=z)
        f.create_dataset("ZERR", data=np.full(n, 1e-3))
        f.create_dataset("WEIGHT", data=np.ones(n))
        f.create_dataset("APP_MAG", data=app_mag)
    return z, app_mag


def test_pixelate_roundtrip_coindexed(tmp_path):
    rng = np.random.default_rng(11)
    raw = tmp_path / "survey.h5"
    _write_survey(raw, rng)
    pixelate.main(["--survey_path", str(raw), "--save_path", str(tmp_path),
                   "--nside", "4"])
    out = tmp_path / "catalog_pixelated_nside_4.h5"

    nside, ngals, zgals, dzgals, wgals, z_depth = load_survey(out)
    props = load_survey_galprops(out)
    assert set(props) == {"gal_app_mag"}
    mag = props["gal_app_mag"]
    assert mag.shape == np.asarray(zgals).shape

    zg = np.asarray(zgals)
    ng = np.asarray(ngals)
    real = np.arange(zg.shape[1])[None, :] < ng[:, None]
    # Real slots: the bijective z -> mag map must survive the z-sort
    # permutation slot-for-slot; padded slots keep the 0.0 fill.
    assert np.allclose(mag[real], 18.0 + 10.0 * zg[real], rtol=0, atol=1e-12)
    assert np.all(mag[~real] == 0.0)
    # Rows really are z-sorted (the permutation was applied, not skipped).
    padded_hi = np.where(real, zg, 1e9)  # finite sentinel: inf-inf would NaN
    assert np.all(np.diff(padded_hi, axis=1) >= 0)


def test_galprops_do_not_enroll_as_marks(tmp_path):
    rng = np.random.default_rng(12)
    raw = tmp_path / "survey.h5"
    _write_survey(raw, rng)
    pixelate.main(["--survey_path", str(raw), "--save_path", str(tmp_path),
                   "--nside", "4"])
    out = tmp_path / "catalog_pixelated_nside_4.h5"
    # The default mark loader must NOT pick up galaxy properties.
    assert load_survey_marks(out) == {}
    assert set(GALPROP_DATASETS).isdisjoint(MARK_DATASETS)


@pytest.mark.parametrize("column,index,value", [
    ("Z", 3, -1.0),          # survey failure sentinel
    ("Z", 5, 0.0),
    ("Z", 9, np.nan),
    ("Z", 11, 100.0),        # the padding sentinel this tool writes
    ("ZERR", 4, -1.0),
    ("ZERR", 6, np.nan),
])
def test_bad_redshift_or_zerr_rejected_at_boundary(tmp_path, column, index, value):
    rng = np.random.default_rng(17)
    raw = tmp_path / "survey.h5"
    _write_survey(raw, rng)
    with h5py.File(raw, "a") as f:
        col = np.asarray(f[column])
        col[index] = value
        del f[column]
        f.create_dataset(column, data=col)
    with pytest.raises(SystemExit):
        pixelate.main(["--survey_path", str(raw), "--save_path", str(tmp_path),
                       "--nside", "4"])


def test_nonfinite_magnitude_rejected_at_boundary(tmp_path):
    rng = np.random.default_rng(13)
    raw = tmp_path / "survey.h5"
    _write_survey(raw, rng)
    with h5py.File(raw, "a") as f:
        m = np.asarray(f["APP_MAG"])
        m[7] = np.nan
        del f["APP_MAG"]
        f.create_dataset("APP_MAG", data=m)
    with pytest.raises(SystemExit):
        pixelate.main(["--survey_path", str(raw), "--save_path", str(tmp_path),
                       "--nside", "4"])
