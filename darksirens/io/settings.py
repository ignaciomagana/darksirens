import json
import os
import sys

import healpy as hp
import jax
import numpy as np

from darksirens.cli.common import _fixed_dark_energy_metadata
from darksirens.gw.populations.utils import normalization_grid_settings

def save_settings_json(
    opts,
    run_dir:                str,
    labels:                 list,
    lower_bound:            list,
    upper_bound:            list,
    fixed_parameter_values: dict,
    prior_overrides:        dict,
    meta:                   dict,
) -> str:
    """Human-readable settings.json for easy inspection and re-runs."""
    d: dict = {}

    for key, val in vars(opts).items():
        if key.startswith("_"):
            continue
        try:
            json.dumps(val)
            d[key] = val
        except (TypeError, ValueError):
            d[key] = str(val)

    # Emit None explicitly so it's obvious when not set — not an empty dict
    d["fixed_parameter_values"] = fixed_parameter_values if fixed_parameter_values else None
    d["prior_overrides"]        = prior_overrides        if prior_overrides        else None
    d["fixed_cosmology"]        = bool(getattr(opts, "fix_cosmology", False))
    d["fixed_de"]               = bool(getattr(opts, "fix_de", False))
    de_meta = _fixed_dark_energy_metadata(opts, fixed_parameter_values)
    d["fixed_dark_energy"]      = de_meta["fixed_dark_energy"]
    d["dark_energy_fixed_values"] = {
        label: value
        for label, value in (
            ("w0", de_meta["w0_value"]),
            ("wa", de_meta["wa_value"]),
        )
        if value is not None
    } or None
    d["w0_fixed"]               = de_meta["w0_fixed"]
    d["wa_fixed"]               = de_meta["wa_fixed"]

    d["labels"]      = list(labels)
    d["lower_bound"] = list(map(float, lower_bound))
    d["upper_bound"] = list(map(float, upper_bound))
    d.update(meta)
    d["normalization_grid"] = normalization_grid_settings().to_dict()

    d["environment"] = {
        "jax_version":    jax.__version__,
        "numpy_version":  np.__version__,
        "healpy_version": hp.__version__,
        "jax_backend":    jax.default_backend(),
        "jax_devices":    [str(dv) for dv in jax.devices()],
        "python_version": sys.version,
    }

    path = os.path.join(run_dir, "settings.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2, default=str)
    return path



