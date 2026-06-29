"""Compatibility wrapper.

Deprecated: import from darksirens.cli.inference_lensing instead.
"""

from darksirens.cli.inference_lensing import *  # noqa: F401,F403
from darksirens.cli import inference_lensing as _impl

globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)
