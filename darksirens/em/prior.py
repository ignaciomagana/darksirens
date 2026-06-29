"""Compatibility wrapper.

Deprecated: import from darksirens.redshift.prior instead.
"""

from darksirens.redshift.prior import *  # noqa: F401,F403
from darksirens.redshift import prior as _redshift_prior

globals().update(
    {
        _name: _value
        for _name, _value in vars(_redshift_prior).items()
        if not _name.startswith("__")
    }
)
