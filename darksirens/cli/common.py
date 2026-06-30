import json
import sys
from argparse import ArgumentParser

import numpy as np

from darksirens.core.constants import H0_FID, OM0_FID, W0_FID, WA_FID

# ── Formatting helpers ─────────────────────────────────────────────────────────

W = 72

def _banner(text: str):
    pad   = max(0, W - 4 - len(text))
    left  = pad // 2
    right = pad - left
    print(f"{'─' * W}")
    print(f"  {'·' * left} {text} {'·' * right}  ")
    print(f"{'─' * W}")

def _section(title: str):
    print()
    print(f"  ┌─ {title} {'─' * max(0, W - 6 - len(title))}┐")

def _row(label: str, value, width: int = 26):
    print(f"  │  {label:<{width}} {value}")

def _end():
    print(f"  └{'─' * (W - 3)}┘")

def _ok(msg: str):   print(f"  ✓  {msg}")
def _warn(msg: str): print(f"  ⚠  {msg}")
def _err(msg: str):  print(f"  ✗  {msg}")

def _fatal(msg: str):
    print()
    _err(f"FATAL: {msg}")
    print()
    sys.exit(1)


def _fixed_dark_energy_metadata(opts, fixed_parameter_values: dict | None) -> dict:
    """Return fixed-state metadata for CPL dark-energy parameters."""
    fixed_parameter_values = fixed_parameter_values or {}
    block_fixed = bool(
        getattr(opts, "fix_cosmology", False) or getattr(opts, "fix_de", False)
    )
    w0_fixed = block_fixed or "w0" in fixed_parameter_values
    wa_fixed = block_fixed or "wa" in fixed_parameter_values

    return {
        "fixed_dark_energy": bool(w0_fixed and wa_fixed),
        "w0_fixed": bool(w0_fixed),
        "wa_fixed": bool(wa_fixed),
        "w0_value": (
            float(fixed_parameter_values["w0"])
            if "w0" in fixed_parameter_values
            else float(W0_FID)
            if w0_fixed
            else None
        ),
        "wa_value": (
            float(fixed_parameter_values["wa"])
            if "wa" in fixed_parameter_values
            else float(WA_FID)
            if wa_fixed
            else None
        ),
    }


def _format_fixed_dark_energy_summary(opts, fixed_parameter_values: dict | None) -> str:
    """Format fixed CPL dark-energy state for human-readable summaries."""
    meta = _fixed_dark_energy_metadata(opts, fixed_parameter_values)
    if not (meta["w0_fixed"] or meta["wa_fixed"]):
        return "no"
    pieces = ["yes" if meta["fixed_dark_energy"] else "partial"]
    fixed_values = []
    if meta["w0_fixed"]:
        fixed_values.append(f"w0={meta['w0_value']:.6g}")
    if meta["wa_fixed"]:
        fixed_values.append(f"wa={meta['wa_value']:.6g}")
    if fixed_values:
        pieces.append(f"({', '.join(fixed_values)})")
    return " ".join(pieces)

def _format_option_value(value):
    """Format parsed CLI option values for human-readable config tables."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "none"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def _print_all_cli_options(optp: ArgumentParser, opts, *, normalization_grid: dict):
    """Print every parsed CLI option in argparse group order."""
    _section("All CLI Options")
    first_group = True
    seen: set[str] = set()

    for group in optp._action_groups:
        group_rows = []
        for action in group._group_actions:
            if action.dest == "help" or not hasattr(opts, action.dest):
                continue
            group_rows.append(action.dest)

        if not group_rows:
            continue

        if not first_group:
            print("  │")
        first_group = False

        _row(f"[{group.title}]", "")
        for dest in group_rows:
            seen.add(dest)
            _row(f"  {dest}", _format_option_value(getattr(opts, dest)))

    ungrouped = sorted(key for key in set(vars(opts)) - seen if not key.startswith("_"))
    if ungrouped:
        if not first_group:
            print("  │")
        _row("[Other]", "")
        for dest in ungrouped:
            _row(f"  {dest}", _format_option_value(getattr(opts, dest)))

    print("  │")
    _row("[Derived]", "")
    _row("  normalization_grid", _format_option_value(normalization_grid))
    _end()


# ── CLI helpers ────────────────────────────────────────────────────────────────

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in {"true", "t", "1", "yes", "y"}:
        return True
    if str(value).lower() in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse '{value}' as boolean.")


def parse_json_arg(value: str | None, argname: str) -> dict:
    if value is None:
        return {}
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("Expected a JSON object (dict).")
        return parsed
    except (json.JSONDecodeError, ValueError) as e:
        _fatal(f"--{argname} must be a valid JSON object. Error: {e}\n"
               f"  Example: --{argname} '{{\"H0\": [60, 80]}}'")


def parse_counterpart_arg(value: list[str] | None) -> tuple[tuple[float, float, float], ...] | None:
    """Parse one or more ``--counterpart RA DEC Z`` triplets into floats.

    Angles are expected in radians, matching the GW sample convention used by
    ``load_gw_samples`` and HEALPix indexing throughout the pipeline.  Multiple
    triplets are ordered by GW event, enabling multi-bright-siren analyses.
    """
    if value is None:
        return None
    if len(value) % 3 != 0:
        _fatal("--counterpart requires RA DEC Z triplets (angles in radians).")
    try:
        vals = [float(x) for x in value]
    except ValueError as e:
        _fatal(f"--counterpart values must be numeric RA DEC Z triplets. Error: {e}")
    out = []
    for i in range(0, len(vals), 3):
        ra, dec, z = vals[i : i + 3]
        if not (0.0 <= ra < 2.0 * np.pi):
            _fatal("--counterpart RA must be in radians with 0 <= RA < 2π.")
        if not (-0.5 * np.pi <= dec <= 0.5 * np.pi):
            _fatal("--counterpart Dec must be in radians with -π/2 <= Dec <= π/2.")
        if z <= 0.0:
            _fatal("--counterpart redshift Z must be positive.")
        out.append((ra, dec, z))
    return tuple(out)


