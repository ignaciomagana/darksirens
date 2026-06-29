"""Compatibility wrapper.

Deprecated: import from darksirens.redshift.volume instead.
"""

from darksirens.redshift.volume import *  # noqa: F401,F403
from darksirens.redshift import volume as _redshift_volume

globals().update(
    {
        _name: _value
        for _name, _value in vars(_redshift_volume).items()
        if not _name.startswith("__")
    }
)
