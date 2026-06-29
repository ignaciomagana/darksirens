"""Compatibility wrapper.

Deprecated: import from darksirens.cli.skymaps_to_samples instead.
"""

from darksirens.cli.skymaps_to_samples import *  # noqa: F401,F403
from darksirens.cli import skymaps_to_samples as _impl

globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)
