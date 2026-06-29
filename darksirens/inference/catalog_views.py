"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.catalog_views instead.
"""

import sys

from darksirens.likelihood import catalog_views as _module
from darksirens.likelihood.catalog_views import *  # noqa: F401,F403

sys.modules[__name__] = _module
