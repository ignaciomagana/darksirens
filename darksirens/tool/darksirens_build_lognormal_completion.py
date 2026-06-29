"""Compatibility wrapper.

Deprecated: import from darksirens.cli.build_lognormal_completion instead.
"""

from darksirens.cli.build_lognormal_completion import *  # noqa: F401,F403
from darksirens.cli import build_lognormal_completion as _impl

globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)
