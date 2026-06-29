"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.selection instead.
"""

import sys

from darksirens.likelihood import selection as _module
from darksirens.likelihood.selection import *  # noqa: F401,F403

sys.modules[__name__] = _module
