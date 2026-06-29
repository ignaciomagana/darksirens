"""Compatibility wrapper.

Deprecated: import from darksirens.cli.pixelate instead.
"""

from darksirens.cli.pixelate import *  # noqa: F401,F403
from darksirens.cli import pixelate as _impl

globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)
