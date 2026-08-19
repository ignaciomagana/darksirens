"""Shared pins for the VOLUME-LIMITED LOA+LS dark-siren line (z < 0.30).

The construction, in one place because every script here depends on all of it:

* **The boundary is conservative by design.**  Containment must hold for the
  WHOLE ``H0`` prior [20, 140].  For a fixed ``dL`` the inferred redshift grows
  with ``H0``, so ``H0 = 140`` is the most restrictive end and
  ``dL(z = 0.30; H0 = 140) = 774.9 Mpc`` is the radial cut that guarantees
  ``z <= 0.30`` at every prior point.  At ``H0 = 67.74`` the same redshift sits
  at 1601 Mpc, so this is a factor ~2 in distance and ~8 in volume tighter than
  a fiducial-cosmology cut would be.  That is the price of a cut that does not
  move as ``H0`` is scanned -- and a cut that DID move would make the event
  list a function of the parameter being measured, which is the one thing a
  volume-limited selection cannot do.

* **The event list is frozen** once selected.  It is a function of the data and
  the survey, never of ``H0``.

* **No completeness model.**  Inside the selected volume LOA+LS is treated as
  complete: ``universe_model = dark_sirens_complete``, no missing-galaxy
  branch, no luminosity function, no ``C(z)``, no ``Q``.  What that assumes is
  recorded and measured in ``COMPLETENESS_ASSUMPTION.md`` rather than asserted.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ZMAX = "0.75"
os.environ.setdefault("DARKSIRENS_ZMAX", ZMAX)

EXP_DIR = Path(__file__).resolve().parent
REPO_DIR = EXP_DIR.parent.parent
DATA_DIR = EXP_DIR / "data"
INGEST_DATA = EXP_DIR.parent / "desi_ingest" / "data"
sys.path.insert(0, str(REPO_DIR))

#: The event/injection pair (the same one the 259-event line uses).
FULLNUTS = Path(os.environ.get(
    "DARKSIRENS_GWDATA_DIR",
    "/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/"
    "magana/GWTC5_S1_spin_fullnuts"))
GW_259 = FULLNUTS / "gwsamples_bbh_whitelist_all_events_final.h5"
INJ_PLAIN = FULLNUTS / "selection_o3o4ab_allsky.h5"

#: LOA+LS products.
SURVEY_N64 = INGEST_DATA / "pixelated_n64" / "catalog_pixelated_nside_64.h5"
MTH_MAP = INGEST_DATA / "mth_map_nside128.h5"

#: This line's cosmology pin: the PE cosmology of the pair.
OM0 = 0.3089

#: The volume definition.
Z_CUT = 0.30
H0_PRIOR = (20.0, 140.0)
#: dL(z=0.30; H0=140, Om0=0.3089), Mpc.  Recomputed and asserted by
#: ``select_events.py`` so this constant cannot drift from the cosmology code.
DL_CUT_MPC = 774.9

#: Credible level defining the localization volume.
CREDIBLE = 0.90

#: nside for the V_90 cell decomposition.  32 gives 3.36 deg^2 cells, so a
#: 300 deg^2 BBH localization spans ~90 cells and 4,096 samples give ~45 per
#: cell -- enough for a density ranking.  Containment is then tested by
#: requiring every nside-128 CHILD of a selected cell to be covered, which is
#: strictly conservative about footprint edges.
NSIDE_V90 = 32
NSIDE_MAP = 128

#: A cell counts as covered when its selection fraction clears this.  1.0 would
#: reject the whole survey (occupied mean f_p = 0.862); 0.0 would count a pixel
#: that is 99% masked.  The scan in ``select_events.py`` reports the event list
#: at several values so the choice is visible rather than buried.
FP_MIN = 0.5
