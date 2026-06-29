"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.cluster_likelihood instead.
"""

import sys

from darksirens.likelihood import cluster_likelihood as _module
from darksirens.likelihood.cluster_likelihood import *  # noqa: F401,F403

sys.modules[__name__] = _module
