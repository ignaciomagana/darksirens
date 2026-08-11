"""TinyNS CLI configuration resolution and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PRESETS = {
    "recommended": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=5,
        step_scale=0.1, min_accepts=1, replacement_chains=1,
        replacement_chain_schedule=None, bound="none", jax_block_size=32,
        jax_vectorized=False, vectorized=False, batch_size=128, max_attempts=None,
    ),
    "heavy_darksirens": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=80,
        step_scale=0.015, min_accepts=8, replacement_chains=16,
        replacement_chain_schedule=None, bound="none", jax_block_size=32,
        jax_vectorized=False, vectorized=False, batch_size=128, max_attempts=300000,
    ),
    "heavy_darksirens_strong": dict(
        sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=160,
        step_scale=0.01, min_accepts=12, replacement_chains=16,
        replacement_chain_schedule=None, bound="none", jax_block_size=32,
        jax_vectorized=False, vectorized=False, batch_size=128, max_attempts=300000,
    ),
    "conservative": dict(sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=5, replacement_chains=1, bound="none", jax_block_size=1),
    "python_debug": dict(sample="rwalk", kernel="python", rwalk_proposal="isotropic", walks=25, replacement_chains=1, replacement_chain_schedule=None, bound="none", jax_block_size=1),
    "prior": dict(sample="prior", kernel="python", vectorized=False, bound="none", jax_block_size=1, replacement_chains=1, replacement_chain_schedule=None),
    "batched_gpu": dict(sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=25, replacement_chains=16, replacement_chain_schedule=None, bound="none", jax_block_size=1),
    "adaptive_gpu": dict(sample="rwalk", kernel="jax", rwalk_proposal="isotropic", walks=25, replacement_chains=1, replacement_chain_schedule=(1, 4, 16, 64, 256), bound="none", jax_block_size=1),
    "bounded_single": dict(sample="rwalk", kernel="jax", bound="single", rwalk_seed="bound", bound_seed_kernel="python", rwalk_proposal="live-cov", walks=5, replacement_chains=16, jax_block_size=1),
    "bounded_multi": dict(sample="rwalk", kernel="jax", bound="multi", rwalk_seed="bound", bound_seed_kernel="python", rwalk_proposal="live-cov", walks=5, replacement_chains=16, jax_block_size=1),
    "fused_bounded_multi": dict(sample="rwalk", kernel="jax", bound="multi", rwalk_seed="bound", bound_seed_kernel="jax", fused_bound_rwalk=True, rwalk_proposal="live-cov", walks=5, replacement_chains=16, jax_block_size=1),
    "custom": {},
}

BASE_DEFAULTS = dict(
    sample="rwalk", kernel="jax", vectorized=False, max_attempts=None,
    walks=5, step_scale=0.1, batch_size=128, min_accepts=1,
    replacement_chains=1, replacement_chain_schedule=None,
    rwalk_proposal="isotropic",
    bound="none", bound_enlargement=1.25, bound_update_interval=100,
    bound_jitter=1e-12, bound_max_draws=None, multi_bound_max_ellipsoids=64,
    multi_bound_min_points=None, multi_bound_split_threshold=0.5,
    multi_bound_enlargement=None, multi_bound_overlap_correction=True,
    rwalk_seed="live", rwalk_seed_fallback=True, bound_seed_kernel="python",
    allow_unused_bound=False, fused_bound_rwalk=False,
    bound_rebuild_on_failure=True, bound_failure_rebuild_threshold=10,
    jax_vectorized=False, jax_block_size=32,
    checkpoint_path=None, checkpoint_interval=100, resume_from=None,
    checkpoint_path_out=None, progress_interval=100,
)
# checkpoint_interval MUST have a concrete default: tinyns' own signature is
# ``checkpoint_interval: int = 100`` and tinyns_run_kwargs forwards the value
# unconditionally, so leaving it None defeated that default and made
# tinyns/run.py's ``if checkpoint_path is not None and checkpoint_interval <= 0``
# raise ``TypeError: '<=' not supported between instances of 'NoneType' and
# 'int'`` -- i.e. every --tinyns_checkpoint_path / --tinyns_resume_from run that
# did not ALSO pass --tinyns_checkpoint_interval crashed after the data load,
# likelihood build and preflight, before writing a single checkpoint.  The unit
# is ITERATIONS (tinyns saves when iteration - last_saved >= interval), not
# seconds; the wall-clock --checkpoint_interval flag only toggles tinyns
# checkpointing on and places the file.

# Resolved-config keys the lensing CLI prints in its "TinyNS resolved config"
# summary.  Kept here (next to TinyNSConfig) so the display list and the config
# schema stay in one place; every key is a TinyNSConfig field (asserted in the
# tests).
TINYNS_RESOLVED_DISPLAY_KEYS = (
    "preset",
    "sample",
    "kernel",
    "rwalk_proposal",
    "walks",
    "step_scale",
    "min_accepts",
    "replacement_chains",
    "max_attempts",
    "jax_block_size",
)


def add_tinyns_arguments(parser_or_group, *, bool_type=None):
    """Register TinyNS CLI arguments with preset-safe defaults.

    Individual TinyNS options default to None so build_tinyns_config can
    distinguish omitted flags from explicit user overrides.
    """
    if bool_type is None:
        from darksirens.cli.common import str_to_bool as bool_type

    g = parser_or_group
    g.add_argument("--tinyns_preset", default="recommended", choices=list(PRESETS),
                   help="tinyns preset; explicit TinyNS options override preset defaults.")
    g.add_argument("--tinyns_sample", default=None, choices=["rwalk", "prior"],
                   help="tinyns sampler: current TinyNS supports rwalk or prior.")
    g.add_argument("--tinyns_kernel", default=None, choices=["jax", "python"],
                   help="tinyns proposal kernel.")
    g.add_argument("--tinyns_vectorized", type=bool_type, default=None, metavar="BOOL")
    g.add_argument("--tinyns_max_attempts", type=int, default=None)
    g.add_argument("--tinyns_walks", type=int, default=None)
    g.add_argument("--tinyns_step_scale", type=float, default=None)
    g.add_argument("--tinyns_batch_size", type=int, default=None)
    g.add_argument("--tinyns_min_accepts", type=int, default=None)
    g.add_argument("--tinyns_replacement_chains", type=int, default=None)
    g.add_argument("--tinyns_replacement_chain_schedule", type=str, default=None,
                   help="Comma-separated positive increasing schedule, e.g. '1,4,16,64'.")
    g.add_argument("--tinyns_rwalk_proposal", choices=["isotropic"], default=None)
    g.add_argument("--tinyns_bound", choices=["none", "single", "multi"], default=None)
    g.add_argument("--tinyns_bound_enlargement", type=float, default=None)
    g.add_argument("--tinyns_bound_update_interval", type=int, default=None)
    g.add_argument("--tinyns_bound_jitter", type=float, default=None)
    g.add_argument("--tinyns_bound_max_draws", type=int, default=None)
    g.add_argument("--tinyns_multi_bound_max_ellipsoids", type=int, default=None)
    g.add_argument("--tinyns_multi_bound_min_points", type=int, default=None)
    g.add_argument("--tinyns_multi_bound_split_threshold", type=float, default=None)
    g.add_argument("--tinyns_multi_bound_enlargement", type=float, default=None)
    g.add_argument("--tinyns_multi_bound_overlap_correction", type=bool_type, default=None, metavar="BOOL")
    g.add_argument("--tinyns_rwalk_seed", choices=["live", "bound"], default=None)
    g.add_argument("--tinyns_rwalk_seed_fallback", type=bool_type, default=None, metavar="BOOL")
    g.add_argument("--tinyns_bound_seed_kernel", choices=["python", "jax"], default=None)
    g.add_argument("--tinyns_allow_unused_bound", type=bool_type, default=None, metavar="BOOL")
    g.add_argument("--tinyns_fused_bound_rwalk", type=bool_type, default=None, metavar="BOOL")
    g.add_argument("--tinyns_bound_rebuild_on_failure", type=bool_type, default=None, metavar="BOOL")
    g.add_argument("--tinyns_bound_failure_rebuild_threshold", type=int, default=None)
    g.add_argument("--tinyns_jax_vectorized", type=bool_type, default=None, metavar="BOOL")
    g.add_argument("--tinyns_jax_block_size", type=int, default=None)
    g.add_argument("--tinyns_checkpoint_path", default=None,
                   help="Explicit tinyns checkpoint file; overrides the "
                        "<run_dir>/checkpoint.tinyns.npz that "
                        "--checkpoint_interval places automatically.")
    g.add_argument("--tinyns_checkpoint_interval", type=int, default=None,
                   help="tinyns checkpoint cadence in ITERATIONS (default 100). "
                        "tinyns saves per iteration, not per second, so this is "
                        "the tinyns cadence knob; --checkpoint_interval "
                        "(seconds) only enables/disables checkpointing here.")
    g.add_argument("--tinyns_resume_from", default=None,
                   help="Explicit tinyns checkpoint to resume; overrides --resume.")
    g.add_argument("--tinyns_checkpoint_path_out", default=None)
    g.add_argument("--tinyns_progress_interval", type=int, default=None)

@dataclass(frozen=True)
class TinyNSConfig:
    preset: str; nlive: int; dlogz: float; max_samples: int | None; seed: int; show_progress: bool
    sample: str; kernel: str; vectorized: bool; max_attempts: int
    walks: int; step_scale: float; batch_size: int; min_accepts: int
    replacement_chains: int; replacement_chain_schedule: tuple[int, ...] | None
    rwalk_proposal: str
    bound: str; bound_enlargement: float; bound_update_interval: int; bound_jitter: float; bound_max_draws: int | None
    multi_bound_max_ellipsoids: int; multi_bound_min_points: int | None; multi_bound_split_threshold: float; multi_bound_enlargement: float | None; multi_bound_overlap_correction: bool
    rwalk_seed: str; rwalk_seed_fallback: bool; bound_seed_kernel: str; allow_unused_bound: bool; fused_bound_rwalk: bool; bound_rebuild_on_failure: bool; bound_failure_rebuild_threshold: int
    jax_vectorized: bool; jax_block_size: int
    checkpoint_path: str | None; checkpoint_interval: int | None; resume_from: str | None; checkpoint_path_out: str | None; progress_interval: int
    explicit: tuple[str, ...] = ()

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self); d["replacement_chain_schedule"] = list(self.replacement_chain_schedule) if self.replacement_chain_schedule else None; d["explicit"] = list(self.explicit); return d


#: Resolved-config fields that do NOT change the statistical target: the
#: checkpoint/resume machinery and the progress printing.  ``build_run_fingerprint``
#: filters ``vars(opts)`` against ``run_fingerprint._NON_SEMANTIC_KEYS`` at the TOP
#: level only, so mirroring these onto ``opts.tinyns_resolved_config`` re-admitted
#: values the fingerprint had deliberately excluded (``show_progress``,
#: ``tinyns_checkpoint_interval``) through the nested dict -- a SLURM requeue that
#: changed the tinyns checkpoint cadence or toggled the progress bar then hit the
#: fatal resume-fingerprint mismatch instead of resuming.  Kept out of the mirror
#: (the dataclass still carries them; the flags themselves are recorded top-level
#: in settings.json).
_NON_SEMANTIC_RESOLVED_KEYS = (
    "checkpoint_path",
    "checkpoint_interval",
    "resume_from",
    "checkpoint_path_out",
    "show_progress",
    "progress_interval",
)


def parse_chain_schedule(raw):
    if raw in (None, ""):
        return None
    text = str(raw).strip().strip("()")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try: vals = tuple(int(p) for p in parts)
    except ValueError as e: raise ValueError("--tinyns_replacement_chain_schedule must contain only integers.") from e
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("--tinyns_replacement_chain_schedule must be a nonempty list of positive integers.")
    if any(b <= a for a, b in zip(vals, vals[1:])):
        raise ValueError("--tinyns_replacement_chain_schedule must be strictly increasing.")
    return vals


def tiny_ns_preset_defaults(preset):
    if preset not in PRESETS:
        raise ValueError(f"Unknown TinyNS preset {preset!r}.")
    d = dict(BASE_DEFAULTS); d.update(PRESETS["recommended"] if preset == "custom" else PRESETS[preset]); return d


def build_tinyns_config(opts):
    preset = getattr(opts, "tinyns_preset", "recommended")
    vals = tiny_ns_preset_defaults(preset)
    names = list(BASE_DEFAULTS)
    explicit = []
    for name in names:
        attr = f"tinyns_{name}"
        if hasattr(opts, attr) and getattr(opts, attr) is not None:
            val = getattr(opts, attr)
            vals[name] = parse_chain_schedule(val) if name == "replacement_chain_schedule" else val
            explicit.append(name)
    vals["replacement_chain_schedule"] = parse_chain_schedule(vals.get("replacement_chain_schedule"))
    max_active = max(vals["replacement_chain_schedule"]) if vals["replacement_chain_schedule"] else int(vals["replacement_chains"])
    if vals.get("max_attempts") is None:
        vals["max_attempts"] = max(10000, int(vals["walks"]) * max_active)
    cfg = TinyNSConfig(
        preset=preset, nlive=int(getattr(opts, "nlive", 1000)), dlogz=float(getattr(opts, "dlogz", 0.1)),
        max_samples=getattr(opts, "max_samples", None), seed=int(getattr(opts, "seed", 0)), show_progress=bool(getattr(opts, "show_progress", True)),
        explicit=tuple(explicit), **vals)
    validate_tinyns_config(cfg)
    # Keep individual CLI override attributes unchanged so repeated resolution
    # remains idempotent and omitted flags stay distinguishable from explicit
    # overrides. Store the resolved config separately for printing and result
    # metadata, MINUS the non-semantic checkpoint/progress fields: this dict is
    # fingerprinted wholesale by build_run_fingerprint (see
    # _NON_SEMANTIC_RESOLVED_KEYS).
    mirror = cfg.to_json_dict()
    for key in _NON_SEMANTIC_RESOLVED_KEYS:
        mirror.pop(key, None)
    # ``explicit`` records WHICH flags were passed, so a non-semantic flag's
    # mere presence there would change the digest just as its value did.
    mirror["explicit"] = [
        name for name in mirror["explicit"]
        if name not in _NON_SEMANTIC_RESOLVED_KEYS
    ]
    setattr(opts, "tinyns_resolved_config", mirror)
    return cfg


def _explicit(cfg, *names): return any(n in cfg.explicit for n in names)

def validate_tinyns_config(c):
    if c.sample not in {"rwalk", "prior"}: raise ValueError("TinyNS sample must be 'rwalk' or 'prior'.")
    if c.kernel not in {"python", "jax"}: raise ValueError("TinyNS kernel must be 'python' or 'jax'.")
    if c.nlive <= 0 or c.dlogz <= 0 or (c.max_samples is not None and c.max_samples < 0): raise ValueError("TinyNS requires nlive > 0, dlogz > 0, and max_samples >= 0.")
    if c.sample == "prior":
        if c.kernel == "jax": raise ValueError("TinyNS kernel='jax' is only valid with sample='rwalk'.")
        if c.replacement_chains != 1 or c.replacement_chain_schedule is not None or c.bound != "none" or c.jax_block_size != 1: raise ValueError("TinyNS sample='prior' requires replacement_chains=1, no chain schedule, bound='none', and jax_block_size=1.")
        if _explicit(c, "walks", "step_scale", "batch_size", "min_accepts", "rwalk_proposal"):
            raise ValueError("TinyNS rwalk-only options cannot be set with sample='prior'.")
    if c.kernel == "jax" and c.sample != "rwalk": raise ValueError("TinyNS kernel='jax' is only valid with sample='rwalk'.")
    if c.replacement_chains != 1 and not (c.sample == "rwalk" and c.kernel == "jax"): raise ValueError("replacement_chains != 1 requires sample='rwalk' and kernel='jax'.")
    if c.replacement_chain_schedule is not None and not (c.sample == "rwalk" and c.kernel == "jax"): raise ValueError("replacement_chain_schedule requires sample='rwalk' and kernel='jax'.")
    if c.replacement_chain_schedule is not None and c.replacement_chains != 1: raise ValueError("replacement_chain_schedule and replacement_chains != 1 are mutually exclusive.")
    if c.walks <= 0 or c.step_scale <= 0 or c.batch_size <= 0 or c.min_accepts <= 0: raise ValueError("TinyNS walks, step_scale, batch_size, and min_accepts must be positive.")
    max_active = max(c.replacement_chain_schedule) if c.replacement_chain_schedule else c.replacement_chains
    if c.max_attempts < c.walks * max_active: raise ValueError("TinyNS max_attempts must be >= walks * max_active_chains.")
    if c.bound not in {"none", "single", "multi"}: raise ValueError("TinyNS bound must be 'none', 'single', or 'multi'.")
    if c.bound != "none" and c.rwalk_seed == "live" and not c.allow_unused_bound: raise ValueError("TinyNS bounds are being built but not used for rwalk seeding; set --tinyns_rwalk_seed bound or --tinyns_allow_unused_bound true.")
    if c.fused_bound_rwalk and not (c.sample == "rwalk" and c.kernel == "jax" and c.bound in {"single", "multi"} and c.rwalk_seed == "bound"): raise ValueError("fused_bound_rwalk=True requires sample='rwalk', kernel='jax', bound single/multi, and rwalk_seed='bound'.")
    if c.jax_block_size <= 0: raise ValueError("jax_block_size must be positive.")
    block_ok = c.sample == "rwalk" and c.kernel == "jax" and (c.bound == "none" or (c.bound in {"single", "multi"} and c.rwalk_seed == "bound" and c.bound_seed_kernel == "jax" and c.fused_bound_rwalk))
    if c.jax_block_size > 1 and not block_ok: raise ValueError("jax_block_size > 1 requires unbounded rwalk+jax or fused bounded rwalk+jax with bound seeding.")
    if c.bound_seed_kernel == "jax" and not (c.sample == "rwalk" and c.kernel == "jax"): raise ValueError("bound_seed_kernel='jax' requires sample='rwalk' and kernel='jax'.")
    if c.bound != "multi" and _explicit(c, "multi_bound_max_ellipsoids", "multi_bound_min_points", "multi_bound_split_threshold", "multi_bound_enlargement", "multi_bound_overlap_correction"): raise ValueError("multi_bound_* options require --tinyns_bound multi.")
    if c.bound_update_interval <= 0 or c.bound_enlargement <= 0 or c.bound_jitter < 0: raise ValueError("bound_update_interval and bound_enlargement must be positive; bound_jitter must be >= 0.")
    if c.bound_failure_rebuild_threshold <= 0: raise ValueError("bound_failure_rebuild_threshold must be positive.")
    # Any checkpoint path at all means tinyns will evaluate
    # ``checkpoint_interval <= 0``, so demand a positive int here rather than
    # letting None/0 reach tinyns (None used to raise a bare TypeError).
    if (c.checkpoint_path or c.resume_from or c.checkpoint_path_out) and (
            c.checkpoint_interval is None or c.checkpoint_interval <= 0):
        raise ValueError("checkpoint_interval must be a positive number of iterations when checkpointing is enabled.")
    if c.checkpoint_path and c.resume_from and not c.checkpoint_path_out: raise ValueError("When both resume_from and checkpoint_path are set, also set checkpoint_path_out to clarify resumed checkpoint output.")


def tinyns_sampler_kwargs(c):
    return {k: getattr(c, k) for k in ["sample","kernel","vectorized","max_attempts","walks","step_scale","batch_size","min_accepts","replacement_chains","replacement_chain_schedule","rwalk_proposal","bound","bound_enlargement","bound_update_interval","bound_jitter","bound_max_draws","multi_bound_max_ellipsoids","multi_bound_min_points","multi_bound_split_threshold","multi_bound_enlargement","multi_bound_overlap_correction","rwalk_seed","rwalk_seed_fallback","bound_seed_kernel","allow_unused_bound","fused_bound_rwalk","bound_rebuild_on_failure","bound_failure_rebuild_threshold","jax_vectorized","jax_block_size"]}

def tinyns_run_kwargs(c):
    kwargs = dict(dlogz=c.dlogz, maxiter=None if c.max_samples is not None and c.max_samples <= 0 else c.max_samples, progress=c.show_progress, progress_interval=c.progress_interval)
    # Belt-and-braces with the concrete BASE_DEFAULTS value: never forward None,
    # which would override tinyns' own ``checkpoint_interval: int = 100`` default
    # with a value its ``checkpoint_interval <= 0`` guard cannot compare.
    if c.checkpoint_interval is not None:
        kwargs["checkpoint_interval"] = int(c.checkpoint_interval)
    return kwargs
