"""Compatibility wrapper.

Deprecated: import from darksirens.cli.analyze instead.
"""

from darksirens.cli.analyze import *  # noqa: F401,F403
from darksirens.cli import analyze as _impl

globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)
