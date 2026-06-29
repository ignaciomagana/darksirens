"""Compatibility wrapper.

Deprecated: import from darksirens.redshift.lognormal_completion instead.
"""

from darksirens.redshift.lognormal_completion import *  # noqa: F401,F403
from darksirens.redshift import lognormal_completion as _redshift_lognormal_completion

globals().update(
    {
        _name: _value
        for _name, _value in vars(_redshift_lognormal_completion).items()
        if not _name.startswith("__")
    }
)
