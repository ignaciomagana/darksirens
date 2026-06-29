"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.cluster_selection instead.
"""

import sys

from darksirens.likelihood import cluster_selection as _module
from darksirens.likelihood.cluster_selection import *  # noqa: F401,F403

sys.modules[__name__] = _module
