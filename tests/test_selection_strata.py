"""Multi-stratum selection-fit plumbing (PR-2).

Pinned here: (1) gal_stratum rides the pixelate galprop registry next to
gal_app_mag; (2) darksirens_fit_selection --strata fits one theta per label
with per-stratum m_lim and writes the 1.1 multi-entry JSON (single stratum
stays 1.0); (3) load_selection_fit_strata reads both formats while
load_selection_fit_json keeps rejecting multi-stratum payloads (the K=1
consumers' contract).
"""

import json

import h5py
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.utils.cosmology import distance_modulus  # noqa: E402


def _write_raw_two_strata(path, rng, n=6000):
    """Two spatially-disjoint strata with different m_lim and M0."""
    z = rng.uniform(0.05, 0.4, size=n)
    stratum = (rng.uniform(size=n) < 0.5).astype(np.int8)
    m0_true = np.where(stratum == 0, -21.0, -20.6)
    M = rng.normal(m0_true, 0.8)
    dm = np.asarray(distance_modulus(jnp.asarray(z), 70.0))
    m = M + dm
    m_lim = np.where(stratum == 0, 20.0, 21.0)
    keep = m <= m_lim
    with h5py.File(path, "w") as f:
        f.create_dataset("TARGET_RA", data=rng.uniform(0, 360, keep.sum()))
        f.create_dataset("TARGET_DEC",
                         data=np.degrees(np.arcsin(rng.uniform(-1, 1,
                                                               keep.sum()))))
        f.create_dataset("Z", data=z[keep])
        f.create_dataset("ZERR", data=np.full(keep.sum(), 0.01))
        f.create_dataset("WEIGHT", data=np.ones(keep.sum()))
        f.create_dataset("APP_MAG", data=m[keep])
        f.create_dataset("STRATUM", data=stratum[keep])
    return int(keep.sum())


def _pixelate(raw, outdir):
    from darksirens.cli import pixelate

    pixelate.main(["--survey_path", str(raw), "--save_path", str(outdir),
                   "--nside", "8"])
    return outdir / "catalog_pixelated_nside_8.h5"


def test_gal_stratum_rides_the_galprop_registry(tmp_path):
    from darksirens.catalogs.io import GALPROP_DATASETS, load_survey_galprops
    from darksirens.cli.pixelate import GALPROP_INPUT_COLUMNS

    assert GALPROP_INPUT_COLUMNS["gal_stratum"] == "STRATUM"
    assert "gal_stratum" in GALPROP_DATASETS

    rng = np.random.default_rng(3)
    raw = tmp_path / "raw.h5"
    n = _write_raw_two_strata(raw, rng)
    surv = _pixelate(raw, tmp_path)
    props = load_survey_galprops(surv)
    assert set(props) >= {"gal_app_mag", "gal_stratum"}
    with h5py.File(surv, "r") as f:
        ngals = f["ngals"][...]
    real = np.arange(props["gal_stratum"].shape[1])[None, :] < ngals[:, None]
    labels = props["gal_stratum"][real]
    assert labels.size == n
    assert set(np.unique(labels)) == {0.0, 1.0}


def test_strata_cli_fits_per_stratum_and_writes_1_1(tmp_path):
    from darksirens.cli.fit_selection import main as fit_main
    from darksirens.redshift.selection import (
        load_selection_fit_json,
        load_selection_fit_strata,
    )

    rng = np.random.default_rng(5)
    raw = tmp_path / "raw.h5"
    _write_raw_two_strata(raw, rng, n=40000)
    surv = _pixelate(raw, tmp_path)
    out = tmp_path / "fit.json"
    fit_main(["--survey_path", str(surv), "--m_lim", "21.0",
              "--strata", "--m_lim_per_stratum", "0:20.0,1:21.0",
              "--out", str(out)])

    payload = json.load(open(out))
    assert payload["format_version"] == "darksirens-selection-fit-1.1"
    strata = load_selection_fit_strata(out)
    assert [s["stratum"] for s in strata] == ["0", "1"]
    assert [s["m_lim"] for s in strata] == [20.0, 21.0]
    m0hat_true = {"0": -21.0 - 5 * np.log10(0.70),
                  "1": -20.6 - 5 * np.log10(0.70)}
    for s in strata:
        sd = float(np.sqrt(np.asarray(s["cov"])[0, 0]))
        assert abs(s["M0hat"] - m0hat_true[s["stratum"]]) < 5 * sd

    # The K=1 single-stratum contract is unchanged: multi payloads rejected.
    with pytest.raises(NotImplementedError):
        load_selection_fit_json(out)


def test_single_stratum_output_stays_1_0(tmp_path):
    from darksirens.cli.fit_selection import main as fit_main
    from darksirens.redshift.selection import (
        load_selection_fit_json,
        load_selection_fit_strata,
    )

    rng = np.random.default_rng(7)
    raw = tmp_path / "raw.h5"
    _write_raw_two_strata(raw, rng, n=20000)
    surv = _pixelate(raw, tmp_path)
    out = tmp_path / "fit.json"
    fit_main(["--survey_path", str(surv), "--m_lim", "21.0",
              "--out", str(out)])
    payload = json.load(open(out))
    assert payload["format_version"] == "darksirens-selection-fit-1.0"
    # Both loaders accept it; the strata loader returns a length-1 list.
    assert load_selection_fit_json(out)["stratum"] == "all"
    assert len(load_selection_fit_strata(out)) == 1
