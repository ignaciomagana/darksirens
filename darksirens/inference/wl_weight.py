"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.wl_weight instead.
"""

import sys

from darksirens.likelihood import wl_weight as _module
from darksirens.likelihood.wl_weight import *  # noqa: F401,F403

sys.modules[__name__] = _module
