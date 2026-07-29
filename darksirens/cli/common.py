import argparse
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

from darksirens.core.constants import W0_FID, WA_FID

# ── Formatting helpers ─────────────────────────────────────────────────────────
#
# The CLIs print a light box-drawing UI (rounded frames, status glyphs).  ANSI
# colour is layered on ONLY when stdout is an interactive terminal and NO_COLOR
# is unset, so a redirected SLURM ``.out`` log stays plain UTF-8 text.  Colour is
# restricted to helper-exclusive elements — the horizontal frames, section
# titles and status glyphs — while the vertical ``│`` content walls are left
# uncoloured, so the many raw ``print("  │ ...")`` lines in the CLIs stay
# visually consistent with rows emitted through ``_row``.

W = 72

# ANSI SGR parameters (joined by ';' within one escape sequence).
_BOLD   = "1"
_DIM    = "2"
_RED    = "31"
_GREEN  = "32"
_YELLOW = "33"
_ACCENT = "38;2;215;119;87"   # Claude coral (#d77757); truecolour, gated on isatty


def _color_enabled() -> bool:
    """Whether it is safe to emit ANSI colour on stdout.

    Honours the ``NO_COLOR`` (disable) and ``FORCE_COLOR`` (enable) conventions
    and otherwise requires an interactive, non-dumb terminal.  Evaluated live
    per call so pytest's captured stdout and shell/SLURM redirection both yield
    plain text without caching a stale decision.
    """
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        if not sys.stdout.isatty():
            return False
    except (AttributeError, ValueError):
        # stdout is None, closed, or a stream without isatty() — colour is a
        # cosmetic add-on, so degrade to plain text rather than crash the CLI.
        return False
    return os.environ.get("TERM") != "dumb"


def _c(text: str, *codes: str) -> str:
    """Wrap ``text`` in ANSI SGR ``codes`` when colour is enabled, else return it."""
    if not codes or not _color_enabled():
        return text
    return f"\x1b[{';'.join(codes)}m{text}\x1b[0m"


def _banner(text: str):
    rule = _c("─" * W, _DIM)
    left = max(0, (W - len(text)) // 2)
    print(rule)
    print(f"{' ' * left}{_c(text, _BOLD, _ACCENT)}")
    print(rule, flush=True)

def _section(title: str):
    print()
    max_title = W - 8  # keep at least one "─" of fill before the "╮" corner
    if len(title) > max_title:
        title = title[: max_title - 1] + "…"
    fill = "─" * max(0, W - 6 - len(title))
    print("  " + _c("╭─ ", _DIM) + _c(title, _BOLD, _ACCENT) + " " + _c(fill + "╮", _DIM))

def _row(label: str, value, width: int = 26):
    print(f"  │  {label:<{width}} {value}")

def _end():
    # Flush at every frame close so redirected SLURM .out logs stay live
    # section-by-section on long runs (the glyph helpers flush line-by-line).
    print("  " + _c("╰" + "─" * (W - 3) + "╯", _DIM), flush=True)

def _ok(msg: str):   print(f"  {_c('✓', _GREEN)}  {msg}", flush=True)
def _warn(msg: str): print(f"  {_c('⚠', _YELLOW)}  {msg}", flush=True)
def _err(msg: str):  print(f"  {_c('✗', _RED, _BOLD)}  {msg}", flush=True)

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


def _print_all_cli_options(optp: ArgumentParser, opts, *, normalization_grid: dict | None = None):
    """Print every parsed CLI option in argparse group order.

    ``normalization_grid`` is optional: when None the ``[Derived]`` rows are
    omitted.  Both inference CLIs pass it -- the pairing/normalisation grids are
    process-global and env-configurable, so they are part of every run's
    provenance regardless of which CLI resolved them.
    """
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

    if normalization_grid is not None:
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


class DeprecatedSpellingAction(argparse.Action):
    """Store like the default ``store`` action, warning on a deprecated spelling.

    Register both the canonical and the deprecated option string(s) on ONE
    ``add_argument`` call (canonical first) and pass ``deprecated=[...]`` naming
    the backward-compatible spellings.  When argparse invokes the action through
    a deprecated option string it prints one ``_warn`` line mapping the old
    spelling to the canonical one; the canonical spelling is silent.  The value
    is stored on the (unchanged) ``dest`` with plain store semantics, so this
    composes with ``type=str_to_bool``, ``nargs="?"`` and ``const=True``.

    Detection is exact against the option string argparse passes.  argparse
    strips ``--flag=value`` to ``--flag`` and expands prefix abbreviations to the
    full option string before calling the action, so matching the passed
    ``option_string`` against the configured deprecated spellings covers the
    ``--flag value``, ``--flag=value`` and abbreviated forms alike.
    """

    def __init__(self, option_strings, dest, deprecated=None, **kwargs):
        self._deprecated = set(deprecated or ())
        # Canonical spelling: the first option string that is not deprecated.
        self._canonical = next(
            (opt for opt in option_strings if opt not in self._deprecated),
            option_strings[0],
        )
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self._deprecated:
            _warn(
                f"{option_string} is deprecated; use {self._canonical} "
                "(same effect, unchanged setting)."
            )
        setattr(namespace, self.dest, values)


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
        if not np.isfinite(z) or z <= 0.0:
            # NaN/inf slip past a bare ``z <= 0``: they reach
            # ``norm.logpdf(z, counterpart_z, counterpart_dz)`` in
            # redshift/prior.py and turn the bright-siren term into NaN or
            # an all--inf likelihood far downstream, after the run started.
            _fatal("--counterpart redshift Z must be a finite positive number.")
        out.append((ra, dec, z))
    return tuple(out)


# ── Shared main() resolvers ─────────────────────────────────────────────────────

def add_normalization_grid_arguments(group):
    """Attach the GW-population normalisation-grid flags to an argparse group.

    Shared by BOTH inference CLIs.  The pairing grid in particular is a
    process-global setting read from the environment at import of
    ``gw/populations/utils.py``, so a CLI that does not expose the flags still
    INHERITS ``DARKSIRENS_GW_PAIRING_M1_GRID`` and evaluates the same interpolated
    ``PairingModel.__call__`` branch -- it just cannot opt in or out, and used to
    skip the coverage guard entirely (PHY-5, one-twin fix).
    """
    group.add_argument(
        "--norm_nmass", type=int, default=None, metavar="N",
        help="Mass-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_MASS).")
    group.add_argument(
        "--norm_nq", type=int, default=None, metavar="N",
        help="Mass-ratio-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_Q).")
    group.add_argument(
        "--norm_nchi", type=int, default=None, metavar="N",
        help="Spin-grid size for GW-population normalisation (env: DARKSIRENS_GW_N_CHI).")
    group.add_argument(
        "--pairing_norm_grid", type=int, default=None, metavar="N",
        help="opt-in: interpolate the pairing model's per-sample "
             "q-normalization from an N-node log-spaced m1 grid "
             "instead of exact per-sample integration (~1.3-1.6x "
             "faster likelihood calls measured). The grid's upper "
             "bound is sized automatically to the selected model's "
             "primary-mass support (e.g. 300 Msun for "
             "gwtc5_fiducial_bpl2peaks), and N is scaled up with it so "
             "node density is preserved -- so it never clamps inside "
             "the support. Accuracy at N=2048: ~1e-8 relative TYPICAL, "
             "up to ~2e-4 WORST-CASE for samples right at the m_min "
             "support edge (the normaliser has a log-kink there); "
             "halves ~4x per grid doubling. NOT recommended when the "
             "run sits near the selection variance guard boundary "
             "(a ~1e-4 logL perturbation can flip the hard -inf "
             "wall). Default: exact "
             "(env: DARKSIRENS_GW_PAIRING_M1_GRID).")
    return group


def configure_normalization_grids_for_model(opts):
    """Configure the normalisation grids and size the opt-in pairing m1 grid.

    Shared by both inference CLIs so the PHY-5 guard cannot be applied to one
    twin only.  The opt-in pairing grid interpolates the q-normaliser on a static
    m1 grid and ``jnp.interp`` CLAMPS out-of-range m1, so the grid's upper bound
    must cover the selected model's primary-mass support or samples INSIDE the
    support silently reuse the endpoint normaliser (measured +0.25 nat in
    log p_pop at m1 = 250-299 for ``gwtc5_fiducial_bpl2peaks``, whose support
    reaches 300 while the default pairing ceiling is M_HI = 200).  Size the grid
    to the model, then assert coverage (defense in depth).

    A malformed ``--pop_model`` is reported later by the standard model-building
    path with its full registry-vocabulary message; don't pre-empt that here, so
    only size when the model resolves.  Coverage problems raise ``ValueError``,
    routed to ``_fatal``.
    """
    from darksirens.gw.populations import get_model, population_m1_support_max
    from darksirens.gw.populations.utils import (
        assert_pairing_grid_covers_support,
        configure_normalization_grids,
        normalization_grid_settings,
        size_pairing_grid_to_support,
    )

    try:
        configure_normalization_grids(
            n_mass=getattr(opts, "norm_nmass", None),
            n_q=getattr(opts, "norm_nq", None),
            n_chi=getattr(opts, "norm_nchi", None),
            pairing_m1_grid=getattr(opts, "pairing_norm_grid", None),
        )
        if normalization_grid_settings().pairing_m1_grid is not None:
            try:
                model = get_model(
                    opts.pop_model,
                    shared_beta=getattr(opts, "shared_beta", True),
                    shared_spin=getattr(opts, "shared_spin", True),
                    shared_gamma=getattr(opts, "shared_gamma", True),
                )
            except Exception:
                return
            support = population_m1_support_max(model)
            size_pairing_grid_to_support(support)
            assert_pairing_grid_covers_support(support, model_name=opts.pop_model)
    except ValueError as exc:
        _fatal(str(exc))


def resolve_redshift_prior_barrier(opts, *, flush=False):
    """Resolve the likelihood-internal redshift-prior optimization barrier.

    Sets ``opts.materialize_redshift_prior_state`` /
    ``opts.redshift_prior_barrier_resolved`` and prints the one-line
    'Disabling ...' notice when ``--redshift_prior_barrier auto`` resolves to a
    non-materialized (vmappable) state.  ``flush`` mirrors each caller's print
    (the lensing CLI flushes; the main CLI does not).
    """
    from darksirens.likelihood.factory import (
        _redshift_prior_materialization_reason,
        _resolve_redshift_prior_materialization,
    )

    opts.materialize_redshift_prior_state = _resolve_redshift_prior_materialization(opts)
    opts.redshift_prior_barrier_resolved = _redshift_prior_materialization_reason(
        opts, opts.materialize_redshift_prior_state
    )
    if (
        opts.redshift_prior_barrier == "auto"
        and not opts.materialize_redshift_prior_state
    ):
        print(
            "  [i] Disabling likelihood-internal redshift-prior optimization_barrier "
            f"({opts.redshift_prior_barrier_resolved}).",
            flush=flush,
        )


def resolve_selection_neff_guard(opts):
    """Resolve the sparse-selection Neff guard mode and variance cap.

    Sets ``opts.selection_neff_soft_guard`` and returns
    ``(guard_mode, max_likelihood_variance)``.  Each CLI keeps its own printed
    guard summary; only this resolution logic is shared.
    """
    from darksirens.likelihood.selection import DEFAULT_MAX_LIKELIHOOD_VARIANCE

    guard_mode = getattr(opts, "selection_neff_guard", "auto")
    opts.selection_neff_soft_guard = (
        guard_mode == "soft"
        or (guard_mode == "auto" and opts.sampler == "numpyro")
    )
    max_likelihood_variance = float(
        getattr(opts, "max_likelihood_variance", DEFAULT_MAX_LIKELIHOOD_VARIANCE)
    )
    return guard_mode, max_likelihood_variance


def run_cli(main_fn):
    """Run a CLI ``main`` and hard-exit on success.

    Skip interpreter teardown, which can hang for hours on shared GPUs when the
    XLA/CUDA runtime blocks in its exit handlers (observed as a finished run
    idling until the SLURM cgroup OOM-killed it).  All outputs are written and
    closed inside ``main``; exceptions still propagate normally and yield a
    nonzero exit.
    """
    main_fn()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
