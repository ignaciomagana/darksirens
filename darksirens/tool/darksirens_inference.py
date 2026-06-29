"""Compatibility wrapper for the inference CLI.

Deprecated: import from darksirens.cli.inference instead.
"""

from darksirens.cli.inference import *  # noqa: F401,F403
from darksirens.cli import inference as _inference

globals().update(
    {
        name: value
        for name, value in vars(_inference).items()
        if not (name.startswith("__") and name.endswith("__"))
    }
)

if __name__ == "__main__":
    main()
