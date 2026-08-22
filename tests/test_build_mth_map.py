"""The depth-map builder's nside-consistency guard.

``scripts/build_mth_map.py`` is a script, not a package module, so it is loaded
here by path.  The guard it carries reads the nside out of the pixel column's
NAME (``HPX128_RING``) and refuses a mismatched ``--nside``, because the
out-of-range mask downstream would otherwise drop most of the sky silently.
"""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_mth_map.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("_build_mth_map", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_source(path, nside=8, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    npix = 12 * nside ** 2
    with h5py.File(path, "w") as f:
        f.create_dataset(f"HPX{nside}_RING",
                         data=rng.integers(0, npix, size=n).astype(np.int64))
        f.create_dataset("flux_ivar_r",
                         data=rng.uniform(1.0, 50.0, n).astype(np.float32))
        f.create_dataset("dered_mag_r",
                         data=rng.uniform(18.5, 21.0, n).astype(np.float32))
        f.create_dataset("maskbits",
                         data=(rng.random(n) < 0.1).astype(np.int32))
    return path


def test_the_builder_runs_at_the_columns_own_nside(tmp_path):
    """Every invocation used to die in the guard: `nside` was undefined there.

    The name existed only as ``args.nside`` in ``main()``, so the guard raised
    NameError before any source file was read -- the builder could not be run
    at ALL, matching or mismatched nside alike.
    """
    mod = _load_builder()
    src = _write_source(tmp_path / "src8.h5", nside=8)
    out = tmp_path / "mth8.h5"
    mod.main(["--sources", str(src), "--out", str(out), "--nside", "8",
              "--k-strata", "2"])
    with h5py.File(out, "r") as f:
        assert int(f.attrs["nside"]) == 8
        assert f["counts"].shape == (12 * 8 ** 2,)
        assert int(np.asarray(f["counts"]).sum()) == 2000


def test_a_mismatched_nside_raises_the_intended_value_error(tmp_path):
    mod = _load_builder()
    src = _write_source(tmp_path / "src8.h5", nside=8)
    with pytest.raises(ValueError, match="indexed at nside 8"):
        mod.main(["--sources", str(src), "--out", str(tmp_path / "mth4.h5"),
                  "--nside", "4", "--k-strata", "2"])
