"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.pair_kde instead.
"""

import sys

from darksirens.likelihood import pair_kde as _module
from darksirens.likelihood.pair_kde import *  # noqa: F401,F403

sys.modules[__name__] = _module
