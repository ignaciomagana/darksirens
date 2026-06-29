"""Compatibility wrapper.

Deprecated: import from darksirens.likelihood.factory instead.
"""

import sys

from darksirens.likelihood import factory as _module
from darksirens.likelihood.factory import *  # noqa: F401,F403

sys.modules[__name__] = _module
