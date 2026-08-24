"""--m_lim_per_stratum entries naming absent strata are refused, not dropped.

Pinned here: darksirens_fit_selection validates the parsed
--m_lim_per_stratum labels against the strata actually present in
gal_stratum and _fatals on unknowns. Before the check, a typo'd label
('2:21.0' when the catalog carries {0, 1}) was silently ignored and that
stratum was fitted with the global --m_lim instead -- and m_lim is the
truncation DATUM of the fit (only m_lim - M0hat is identified), so the
wrong limit shifts the fitted M0hat one-for-one into the selection JSON.
"""

import h5py
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from darksirens.utils.cosmology import distance_modulus  # noqa: E402


def _write_raw_two_strata(path, rng, n=4000):
    """Two strata labeled {0, 1} with different m_lim and M0."""
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


def test_unknown_stratum_label_in_m_lim_per_stratum_is_fatal(tmp_path, capsys):
    from darksirens.cli import pixelate
    from darksirens.cli.fit_selection import main as fit_main

    rng = np.random.default_rng(17)
    raw = tmp_path / "raw.h5"
    _write_raw_two_strata(raw, rng)
    pixelate.main(["--survey_path", str(raw), "--save_path", str(tmp_path),
                   "--nside", "8"])
    surv = tmp_path / "catalog_pixelated_nside_8.h5"
    out = tmp_path / "fit.json"

    # Stratum 2 does not exist: refused up front, no JSON written.
    with pytest.raises(SystemExit):
        fit_main(["--survey_path", str(surv), "--m_lim", "21.0",
                  "--strata", "--m_lim_per_stratum", "0:20.0,2:21.0",
                  "--out", str(out)])
    msg = capsys.readouterr().out
    assert "not present" in msg and "2" in msg
    assert not out.exists()
