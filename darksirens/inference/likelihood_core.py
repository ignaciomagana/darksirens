"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.core instead.
"""

import sys

from darksirens.likelihood import core as _module
from darksirens.likelihood.core import *  # noqa: F401,F403

sys.modules[__name__] = _module
