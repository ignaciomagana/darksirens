"""Shared constants for the FULL-RANGE GWTC-5 259-event line (no z cuts).

Owner decision (2026-08-10): the final H0 construction uses the FULL
detected-event sample against the plain all-sky injection set, with NO
windows or z cuts anywhere -- "the point of completeness corrections is
not to have cuts".  This line therefore pins its OWN grid:

    z(max PE dL = 27118.9 Mpc; H0=140, Om0=0.3089) = 5.7353  ->  ZMAX = 6.0

(1086 log-spaced nodes; the log grid preserves low-z density, so the
catalog term keeps its resolution).  The desi_ingest 0.75-grid products
(44-event line, sampled-theta NS) are UNTOUCHED and incompatible with
this line -- the Q loader enforces the grid match.

The selection fit JSON and K(z) template are GRID-FREE and shared from
desi_ingest; n0 calibration and Q tables are rebuilt here on this grid.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ZMAX = "6.0"
os.environ.setdefault("DARKSIRENS_ZMAX", ZMAX)
if os.environ["DARKSIRENS_ZMAX"] != ZMAX:
    raise RuntimeError(
        f"DARKSIRENS_ZMAX={os.environ['DARKSIRENS_ZMAX']} conflicts with the "
        f"full-range pin {ZMAX}; unset it or fix common.py")

EXP_DIR = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parent.parent
DATA_DIR = EXP_DIR / "data"
INGEST_DATA = EXP_DIR.parent / "desi_ingest" / "data"

sys.path.insert(0, str(REPO_DIR))

# --- the owner-designated event/injection pair (2026-08-10) -----------------
#: The event/injection pair's directory.  Overridable by
#: ``DARKSIRENS_GWDATA_DIR`` so the same scripts run on a second machine
#: (the Jetstream H100) without a fork of this file: the PSC path below stays
#: the default, so nothing about the PSC line changes.
FULLNUTS = Path(os.environ.get(
    "DARKSIRENS_GWDATA_DIR",
    "/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/"
    "magana/GWTC5_S1_spin_fullnuts"))
GW_259 = FULLNUTS / "gwsamples_bbh_whitelist_all_events_final.h5"
INJ_PLAIN = FULLNUTS / "selection_o3o4ab_allsky.h5"

#: This product line's PE cosmology (differs from desi_ingest's 0.3075).
OM0 = 0.3089

# --- grid-free products shared from the ingest line -------------------------
SURVEY_N64 = INGEST_DATA / "pixelated_n64" / "catalog_pixelated_nside_64.h5"
FIT_JSON = INGEST_DATA / "selection_fit_union.json"
FIT_NS_JSON = INGEST_DATA / "selection_fit_ns.json"
STRATUM_MAP = INGEST_DATA / "stratum_map_ns_nside64.h5"
RAW_UNION = INGEST_DATA / "desi_union_raw.h5"

M_LIM_UNION = 21.0
Z_DEPTH = 0.30

# provenance helper shared verbatim
import json, subprocess  # noqa: E402


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True,
        text=True).stdout.strip()


def write_provenance(out_path: Path, payload: dict) -> None:
    from datetime import datetime, timezone

    payload = dict(payload)
    payload.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    payload.setdefault("darksirens_git_sha", git_sha())
    payload.setdefault("DARKSIRENS_ZMAX", os.environ["DARKSIRENS_ZMAX"])
    payload.setdefault("argv", sys.argv)
    prov = out_path.with_suffix(out_path.suffix + ".provenance.json")
    with open(prov, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    print(f"[provenance] {prov}")
