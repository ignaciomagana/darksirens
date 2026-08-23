"""Map the loa_rebuild flat catalog to the darksirens raw-survey schema.

Input  (read-only): rebuild_loa_faint_pixelate_input.h5
Output: data/desi_union_raw.h5 with the `darksirens_pixelate` datasets
        (TARGET_RA/TARGET_DEC/Z/ZERR/WEIGHT/APP_MAG) plus diagnostic extras
        the pixelator ignores (SURVEY_CODE, STRATUM, MAG_R).

Cut: isfinite(M_APP) & M_APP <= m_lim (21.0) — removes the 268-row unmatched
DESI-only tail (NaN M_APP) and the single spec row above the LS limit, so
every surviving row has a clean north/south assignment by source-row index.
"""

from __future__ import annotations

import numpy as np
import h5py

import common as C


def main() -> None:
    src = C.LOA_FLAT
    out = C.DATA_DIR / "desi_union_raw.h5"
    C.DATA_DIR.mkdir(exist_ok=True)

    with h5py.File(src, "r") as f:
        n = f["TARGET_RA"].shape[0]
        assert n == C.TOTAL_ROWS, f"source rows {n} != expected {C.TOTAL_ROWS}"
        cols = {k: f[k][...] for k in ("TARGET_RA", "TARGET_DEC", "Z", "ZERR", "WEIGHT", "MAG_R", "M_APP", "SURVEY_CODE")}
        src_attrs = dict(f.attrs)

    m_app = cols["M_APP"]
    keep = np.isfinite(m_app) & (m_app <= C.M_LIM_UNION)
    n_cut = int((~keep).sum())

    idx = np.nonzero(keep)[0]
    stratum = np.where(idx < C.SOUTH_END, 0, 1).astype(np.int8)  # 0=south, 1=north
    assert idx.max() < C.NORTH_END, "a kept row falls in the unmatched tail"

    code = cols["SURVEY_CODE"][keep].astype(np.int8)
    cut_codes = {int(c): int(m) for c, m in zip(*np.unique(cols["SURVEY_CODE"][~keep], return_counts=True))}

    with h5py.File(out, "w") as f:
        for name, key in (("TARGET_RA", "TARGET_RA"), ("TARGET_DEC", "TARGET_DEC"),
                          ("Z", "Z"), ("ZERR", "ZERR"), ("WEIGHT", "WEIGHT")):
            f.create_dataset(name, data=cols[key][keep].astype(np.float64), compression="gzip")
        f.create_dataset("APP_MAG", data=m_app[keep].astype(np.float64), compression="gzip")
        # extras (ignored by darksirens_pixelate today; STRATUM feeds Stage B)
        f.create_dataset("SURVEY_CODE", data=code, compression="gzip")
        f.create_dataset("STRATUM", data=stratum, compression="gzip")
        f.create_dataset("MAG_R", data=cols["MAG_R"][keep].astype(np.float64), compression="gzip")
        f.attrs["source"] = str(src)
        f.attrs["m_lim"] = C.M_LIM_UNION
        f.attrs["z_depth"] = C.Z_DEPTH
        f.attrs["n_rows"] = int(keep.sum())
        f.attrs["n_cut"] = n_cut

    z, zerr = cols["Z"][keep], cols["ZERR"][keep]
    prov = {
        "source": str(src),
        "source_sha256": C.sha256_of(src),
        "source_attrs": {k: str(v) for k, v in src_attrs.items()},
        "cut": f"isfinite(M_APP) & M_APP <= {C.M_LIM_UNION}",
        "n_source": int(n),
        "n_kept": int(keep.sum()),
        "n_cut": n_cut,
        "cut_rows_by_survey_code": cut_codes,
        "side_split_rows": {"south_end": C.SOUTH_END, "north_end": C.NORTH_END},
        "kept_by_stratum": {"south": int((stratum == 0).sum()), "north": int((stratum == 1).sum())},
        "kept_by_survey_code": {int(c): int(m) for c, m in zip(*np.unique(code, return_counts=True))},
        "z_range": [float(z.min()), float(z.max())],
        "zerr_range": [float(zerr.min()), float(zerr.max())],
        "app_mag_quantiles_50_99_999_max": [float(q) for q in np.percentile(m_app[keep], [50, 99, 99.9, 100])],
    }
    C.write_provenance(out, prov)
    print(f"wrote {out}: {prov['n_kept']:,} rows (cut {n_cut}; "
          f"south {prov['kept_by_stratum']['south']:,} / north {prov['kept_by_stratum']['north']:,})")


if __name__ == "__main__":
    main()
