"""Compatibility wrapper.

Deprecated: import from darksirens.cli.diagnose_lognormal_completion instead.
"""

from darksirens.cli.diagnose_lognormal_completion import *  # noqa: F401,F403
from darksirens.cli import diagnose_lognormal_completion as _impl

globals().update(
    {
        name: value
        for name, value in vars(_impl).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)
