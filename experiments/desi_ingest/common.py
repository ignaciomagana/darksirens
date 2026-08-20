"""Shared constants and helpers for the DESI real-catalog ingestion experiment.

Import this FIRST in every script (before darksirens) so DARKSIRENS_ZMAX is
pinned before darksirens.redshift.grid reads it at import time.  A zgrid
mismatch between the selection fit, the Q-table build, and inference is fatal
(the Q loader enforces it); the pin here is the single source of truth.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# --- environment pins (set before importing darksirens) ---------------------
# z(max PE dL = 2239.4 Mpc; H0=140, Om0=0.3075) = 0.7314 over gwsamples_44.h5;
# 0.75 covers the full PE support at the top of the H0 scan grid.
ZMAX = "0.75"
os.environ.setdefault("DARKSIRENS_ZMAX", ZMAX)
if os.environ["DARKSIRENS_ZMAX"] != ZMAX:
    raise RuntimeError(
        f"DARKSIRENS_ZMAX={os.environ['DARKSIRENS_ZMAX']} conflicts with the "
        f"experiment pin {ZMAX}; unset it or fix common.py"
    )

EXP_DIR = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parent.parent
DATA_DIR = EXP_DIR / "data"
LOG_DIR = EXP_DIR / "logs"

sys.path.insert(0, str(REPO_DIR))

# --- read-only sources -------------------------------------------------------
DESI_REPO = Path("/hildafs/projects/phy230014p/magana/desi_darksirens_selection")
LOA_INPUTS = DESI_REPO / "final/experiments/experiment_loa_rebuild/inputs"
LOA_FLAT = LOA_INPUTS / "rebuild_loa_faint_pixelate_input.h5"
GW_SAMPLES_44 = LOA_INPUTS / "gwsamples_44.h5"
BETA_S0495 = LOA_INPUTS / "selection_betaS_v2_loaFaint_marg_s0495_noom.h5"
POP_FIDUCIAL_JSON = DESI_REPO / "production_final/configs/gwtc5_pop_fiducial.json"
REF_PIXELATED_N128 = LOA_INPUTS / "rebuild_loa_faint_pixelated/catalog_pixelated_nside_128.h5"
REF_JOINT42_SUMMARY = (
    DESI_REPO / "final/experiments/experiment_loa_rebuild/runs/loa_faint_joint42/h0_scan.summary.json"
)
KCORR_MODULE = DESI_REPO / "preprocess-kibo/calc_kcor.py"
LS_NORTH = Path("/hildafs/projects/phy220048p/share/legacysurvey/ls_dr9north_m21_v2.h5")
LS_SOUTH = Path("/hildafs/projects/phy220048p/share/legacysurvey/ls_dr10south_m21.h5")

# --- catalog facts (verified 2026-08-08 against the loa_rebuild build log) ---
M_LIM_UNION = 21.0            # LS backbone retention: dered r <= 21
Z_DEPTH = 0.30                # catalog completeness truncation (survey attr)
SOUTH_END = 17_907_592        # rows [0, SOUTH_END) are LS-DR10 south
NORTH_END = 22_787_567        # rows [SOUTH_END, NORTH_END) are LS-DR9 north
TOTAL_ROWS = 22_787_835       # rows >= NORTH_END are 268 unmatched DESI-only
MEDIAN_GR = 0.986             # median g-r used for the K(z) template

NSIDE_PRIMARY = 64
NSIDE_PARITY = 128


def sha256_of(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True
    ).stdout.strip()


def write_provenance(out_path: Path, payload: dict) -> None:
    """Write <out_path>.provenance.json next to a data product."""
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
