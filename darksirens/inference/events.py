"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.events instead.
"""

import sys

from darksirens.likelihood import events as _module
from darksirens.likelihood.events import *  # noqa: F401,F403

sys.modules[__name__] = _module
