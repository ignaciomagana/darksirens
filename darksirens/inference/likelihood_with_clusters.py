"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.likelihood_with_clusters instead.
"""

import sys

from darksirens.likelihood import likelihood_with_clusters as _module
from darksirens.likelihood.likelihood_with_clusters import *  # noqa: F401,F403

sys.modules[__name__] = _module
