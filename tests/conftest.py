import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Enable x64 BEFORE any darksirens import: module-level grids (e.g.
# darksirens.redshift.grid.zgrid) are built at import time and freeze float32
# if x64 is enabled later, which makes tolerance tests import-order flaky.
try:
    import jax
    jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

# numpy<2 compatibility for the whole suite: several tests use np.trapezoid
# (added in numpy 2.0; np.trapz deprecated there). Production code uses
# jnp.trapezoid, which exists in the pinned jax.
import numpy as _np
if not hasattr(_np, "trapezoid"):
    _np.trapezoid = _np.trapz
