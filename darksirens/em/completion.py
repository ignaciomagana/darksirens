"""Compatibility wrapper.

Deprecated: import from darksirens.redshift.completion instead.
"""

from darksirens.redshift.completion import *  # noqa: F401,F403
from darksirens.redshift import completion as _redshift_completion

globals().update(
    {
        _name: _value
        for _name, _value in vars(_redshift_completion).items()
        if not _name.startswith("__")
    }
)
