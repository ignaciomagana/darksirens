"""
flows.py
--------
Per-event normalizing-flow posterior surrogates: checkpoint loading and
batched ensemble evaluation.

The flows are trained externally (per event, hyperparameters tuned
independently) and shipped as ``<EVENT>/<EVENT>_flow.npz`` checkpoints:
``np.savez`` of the flow's array leaves (``arr_0..arr_N``, the order of
``eqx.partition``/``tree_flatten``) plus a ``config_json`` string holding
everything needed to rebuild the skeleton.  Loading therefore rebuilds the
skeleton from the config with the *installed* flowjax and assigns the stored
arrays positionally — a (shape, dtype) structural check guards against
silent corruption from flowjax/equinox version drift.

Each flow is a ``triangular_spline_flow`` over a standard-Normal base,
wrapped in ``Transformed(flow, non_trainable(Chain([Affine(Z_mean, Z_std),
constraint_transform])))`` so that ``flow.log_prob(X)`` is the PE-posterior
log density directly in physical coordinates.  The spectral-siren flows use
``columns = ("mass_1", "mass_2", "luminosity_distance", "chi_eff")``:
DETECTOR-frame masses in M_sun, dL in Mpc, dimensionless chi_eff.

Loader/reconstruction code in this module is adapted from the gw-nf package
(https://github.com/jgassert/gw-nf, MIT license, Julius Gassert): the
constraint bijections, ``create_flow_from_config``, the npz round-trip, and
the structure-signature grouping used for batched evaluation.

x64 note: the checkpoints were trained in float64; darksirens enables x64
globally (core/jax_config.py), and ``load_flow_ensemble`` asserts it.
"""

from __future__ import annotations

import glob as _glob
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

try:
    import equinox as eqx
    import paramax
    import flowjax.bijections as bij
    from flowjax.bijections import AbstractBijection, Affine, Chain
    from flowjax.distributions import Normal, Transformed
    from flowjax.flows import triangular_spline_flow
except ModuleNotFoundError as _e:  # pragma: no cover - environment-dependent
    raise ModuleNotFoundError(
        "darksirens.gw.flows requires the flow-surrogate extras "
        "(flowjax, paramax, equinox>=0.11). Install them with "
        "`pip install 'darksirens[flows]'` or "
        "`pip install 'flowjax>=17.1,<18' paramax 'equinox>=0.11,<0.13'` "
        "(jax>=0.6 environments need equinox>=0.12)."
    ) from _e


# Column layouts the likelihood knows how to assemble. The spectral-siren
# layout is fully supported; the dark-siren (sky-position) layout is
# recognised so its arrival fails loudly at the right place, not with a
# shape error deep inside the likelihood.
SPECTRAL_COLUMNS = ("mass_1", "mass_2", "luminosity_distance", "chi_eff")
DARK_COLUMNS = SPECTRAL_COLUMNS + ("ra", "dec")

DEFAULT_PATTERN = "*/*_flow.npz"


class CheckpointStructureError(RuntimeError):
    """A saved ``.npz`` does not match the skeleton rebuilt in this environment.

    ``load_flow`` assigns the stored arrays to the skeleton's leaves *by
    position*.  If the flowjax that wrote the checkpoint differs from the one
    rebuilding it, the leaf order or shapes no longer line up and arrays are
    silently mis-assigned.  This corrupts results without raising on its own,
    so it is detected and refused up front.
    """


# ── constraint bijections (vendored from gw-nf core/constrained_transform) ──


class OrderedPositiveN(AbstractBijection):
    """Maps z in R^n -> x[0] > x[1] > ... > x[n-1] > 0 via cumulative exp.

    The Jacobian is diagonal, so log|det J| = sum(z).
    """

    shape: tuple
    cond_shape: None = None

    def __init__(self, n: int):
        self.shape = (n,)

    def transform_and_log_det(self, z, condition=None):
        x = jnp.zeros_like(z)
        x = x.at[-1].set(jnp.exp(z[-1]))

        def body(carry, k):
            return carry.at[k].set(carry[k + 1] + jnp.exp(z[k])), None

        x, _ = jax.lax.scan(body, x, jnp.arange(z.shape[0] - 2, -1, -1))
        return x, jnp.sum(z)

    def inverse_and_log_det(self, x, condition=None):
        z = jnp.zeros_like(x)
        z = z.at[-1].set(jnp.log(x[-1]))

        def body(carry, k):
            return carry.at[k].set(jnp.log(jnp.maximum(x[k] - x[k + 1], 1e-10))), None

        z, _ = jax.lax.scan(body, z, jnp.arange(x.shape[0] - 2, -1, -1))
        return z, -jnp.sum(z)


def build_constrained_transform(d: int, constraints: dict) -> AbstractBijection:
    """Build a single bijection R^d -> constrained space from a spec dict.

    Keys are dimension indices; values are spec dicts
    (``{"type": "positive"}``, ``{"type": "interval", "low": a, "high": b}``,
    ``{"type": "ordered_positive", "dims": [i, j]}``) or None for dimensions
    consumed by a multi-dim constraint. Unlisted dimensions pass through.
    """
    handled: set[int] = set()
    indexed: list[AbstractBijection] = []

    for dim, spec in constraints.items():
        if spec is None or dim in handled:
            continue
        t = spec["type"]
        if t == "real":
            continue
        if t == "positive":
            indexed.append(bij.Indexed(bij.Exp(), dim, (d,)))
        elif t == "interval":
            lo, hi = float(spec["low"]), float(spec["high"])
            b = bij.Chain([bij.Sigmoid(), bij.Affine(loc=lo, scale=hi - lo)])
            indexed.append(bij.Indexed(b, dim, (d,)))
        elif t == "ordered_positive":
            dims = list(spec["dims"])
            idx = jnp.array(dims, dtype=jnp.int32)
            indexed.append(bij.Indexed(OrderedPositiveN(len(dims)), idx, (d,)))
            handled.update(dims)
        else:
            raise ValueError(f"Unknown constraint type: {t!r}")

    if not indexed:
        return bij.Identity((d,))
    if len(indexed) == 1:
        return indexed[0]
    return bij.Chain(indexed)


# ── checkpoint reconstruction ────────────────────────────────────────────────


def create_flow_from_config(config: dict) -> Transformed:
    """Rebuild the untrained flow skeleton described by a checkpoint config.

    Only the architecture used by the shipped event flows is supported:
    ``type="spline"`` (triangular spline flow) over a Normal base, optionally
    wrapped with the un-standardisation Affine and the constraint transform.
    """
    if config.get("base_dist") != "Normal":
        raise NotImplementedError(
            f"base_dist={config.get('base_dist')!r} not supported; the event "
            "flow checkpoints use 'Normal'."
        )
    if config.get("type") != "spline":
        raise NotImplementedError(
            f"flow type={config.get('type')!r} not supported; the event flow "
            "checkpoints use 'spline' (triangular_spline_flow)."
        )

    d = int(config["data_dim"])
    base = Normal(jnp.zeros(d), jnp.ones(d))
    flow = triangular_spline_flow(
        key=jax.random.key(config["key"]),
        base_dist=base,
        flow_layers=config["flow_layers"],
        knots=config["knots"],
    )

    bijs = []
    if "Z_mean" in config and "Z_std" in config:
        bijs.append(
            Affine(loc=jnp.array(config["Z_mean"]), scale=jnp.array(config["Z_std"]))
        )
    if "constraints" in config:
        # JSON round-trip turns int keys into strings; restore them.
        constraints = {int(k): v for k, v in config["constraints"].items()}
        bijs.append(build_constrained_transform(d, constraints))
    if bijs:
        full = bijs[0] if len(bijs) == 1 else Chain(bijs)
        flow = Transformed(flow, paramax.non_trainable(full))
    return flow


def check_checkpoint_matches_skeleton(
    path: str | Path,
    constructor_func: Callable[[dict], Any] = create_flow_from_config,
) -> None:
    """Verify a checkpoint's stored arrays match the rebuilt skeleton.

    Compares the count and per-leaf (shape, dtype) of the arrays stored in
    the ``.npz`` against the skeleton produced in the *current* environment;
    raises :class:`CheckpointStructureError` on any mismatch (almost always a
    flowjax/equinox version drift between training and now).
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        config = json.loads(str(data["config_json"]))
        n_file = sum(1 for k in data.files if k.startswith("arr_"))
        file_specs = [
            (tuple(data[f"arr_{i}"].shape), str(data[f"arr_{i}"].dtype))
            for i in range(n_file)
        ]

    skeleton = constructor_func(config)
    arrays, _ = eqx.partition(skeleton, eqx.is_array)
    leaves, _ = jtu.tree_flatten(arrays)
    skel_specs = [(tuple(l.shape), str(l.dtype)) for l in leaves]

    if file_specs == skel_specs:
        return

    diffs = []
    for i in range(max(len(file_specs), len(skel_specs))):
        f = file_specs[i] if i < len(file_specs) else None
        s = skel_specs[i] if i < len(skel_specs) else None
        if f != s:
            diffs.append(f"  arr_{i}: file={f}  skeleton={s}")
    raise CheckpointStructureError(
        f"Checkpoint '{path}' does not match the flow rebuilt from its config "
        f"in this environment: {len(file_specs)} stored arrays vs "
        f"{len(skel_specs)} skeleton leaves. load_flow assigns arrays "
        "positionally, so these would be silently mis-mapped. This is almost "
        "certainly a flowjax/equinox version mismatch between training and "
        "now. First differing leaves:\n" + "\n".join(diffs[:12])
    )


def load_flow(
    path: str | Path,
    constructor_func: Callable[[dict], Any] = create_flow_from_config,
) -> tuple[Any, dict]:
    """Load a serialized flow checkpoint and reconstruct the model."""
    with np.load(path, allow_pickle=True) as data:
        config = json.loads(str(data["config_json"]))
        skeleton = constructor_func(config)
        arrays, static = eqx.partition(skeleton, eqx.is_array)
        leaves, treedef = jtu.tree_flatten(arrays)
        loaded = [jnp.asarray(data[f"arr_{i}"]) for i in range(len(leaves))]
        flow = eqx.combine(jtu.tree_unflatten(treedef, loaded), static)
    return flow, config


# ── ensemble: grouping + stacked batched evaluation ─────────────────────────


def _structure_signature(flow: Any) -> tuple:
    """Hashable signature identifying flows that can be stacked & vmapped."""
    leaves, treedef = jtu.tree_flatten(flow)
    leaf_spec = tuple(
        (tuple(getattr(leaf, "shape", ())), str(getattr(leaf, "dtype", type(leaf).__name__)))
        for leaf in leaves
        if eqx.is_array(leaf)
    )
    return (treedef, leaf_spec)


@dataclass(frozen=True)
class FlowGroup:
    """A set of structurally-identical flows evaluated by one vmapped kernel.

    ``params`` is a pytree whose array leaves carry a leading (G,) model
    axis (traced jit operand); ``static`` is the shared non-array half
    (functions/metadata, closure-captured, never traced).
    """

    indices: tuple[int, ...]
    params: Any
    static: Any

    @property
    def size(self) -> int:
        return len(self.indices)


@dataclass
class FlowEnsemble:
    """Host-side container for the per-event flows of one inference run.

    Not a jit operand: pass :meth:`group_params` (arrays) into the jitted
    likelihood and evaluate with the callable from
    :func:`make_ensemble_log_prob`, which closure-captures the statics.
    """

    names: list[str]
    configs: list[dict]
    paths: list[Path]
    groups: list[FlowGroup]
    columns: tuple[str, ...]
    data_dim: int
    skipped: list = field(default_factory=list)

    @property
    def n_flows(self) -> int:
        return len(self.names)

    def group_params(self) -> tuple:
        """Stacked array-leaf pytrees, one per architecture group."""
        return tuple(g.params for g in self.groups)

    def summary(self) -> str:
        lines = [
            f"FlowEnsemble: {self.n_flows} flows, data_dim={self.data_dim}, "
            f"columns={list(self.columns)}, {len(self.groups)} architecture group(s)."
        ]
        for k, g in enumerate(self.groups):
            members = [self.names[i] for i in g.indices]
            preview = ", ".join(members[:5]) + (" ..." if len(members) > 5 else "")
            cfg = self.configs[g.indices[0]]
            lines.append(
                f"  group {k}: {g.size:3d} flow(s), layers={cfg.get('flow_layers')}, "
                f"knots={cfg.get('knots')}  [{preview}]"
            )
        if self.skipped:
            lines.append(f"  skipped {len(self.skipped)} mismatching checkpoint(s).")
        return "\n".join(lines)

    def __len__(self) -> int:
        return self.n_flows


def _event_name_from_path(p: Path) -> str:
    stem = p.name
    if stem.endswith("_flow.npz"):
        return stem[: -len("_flow.npz")]
    return p.stem


def _validate_columns(columns: tuple[str, ...]) -> None:
    if columns == SPECTRAL_COLUMNS:
        return
    if columns == DARK_COLUMNS:
        raise NotImplementedError(
            "6-D dark-siren flows (with ra/dec) are recognised but not yet "
            "supported: the likelihood-side pieces missing are the ra/dec "
            "input assembly (pixels/unit vectors), the 1/(4*pi) sky factor in "
            "the PE prior, and the per-pixel redshift-prior sampler. Only the "
            f"spectral layout {list(SPECTRAL_COLUMNS)} is implemented."
        )
    raise ValueError(
        f"Unrecognised flow column layout {list(columns)}; expected "
        f"{list(SPECTRAL_COLUMNS)} (spectral sirens)."
    )


def load_flow_ensemble(
    flows_dir: str | Path,
    pattern: str = DEFAULT_PATTERN,
    on_mismatch: str = "error",
) -> FlowEnsemble:
    """Discover and load every ``*_flow.npz`` under ``flows_dir``.

    Event order — and therefore the likelihood's event indexing — is the
    lexicographically sorted checkpoint order.  ``on_mismatch`` controls what
    happens when a checkpoint fails the structural check: ``"error"`` raises,
    ``"skip"`` warns and omits it, ``"load"`` skips the check (unsafe).
    """
    if on_mismatch not in ("error", "skip", "load"):
        raise ValueError("on_mismatch must be 'error', 'skip' or 'load'.")
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "Flow checkpoints require x64; call configure_jax_runtime() (or "
            "jax.config.update('jax_enable_x64', True)) before loading."
        )

    flows_dir = Path(flows_dir)
    paths = sorted(p for p in flows_dir.glob(pattern) if "__MACOSX" not in p.parts)
    if not paths:
        # Also accept an absolute / cwd-relative glob.
        paths = sorted(
            Path(p) for p in _glob.glob(str(pattern)) if "__MACOSX" not in Path(p).parts
        )
    if not paths:
        raise FileNotFoundError(
            f"No flow checkpoints matched '{pattern}' under '{flows_dir}'."
        )

    flows, names, configs, kept, skipped = [], [], [], [], []
    for p in paths:
        if on_mismatch != "load":
            try:
                check_checkpoint_matches_skeleton(p)
            except CheckpointStructureError:
                if on_mismatch == "error":
                    raise
                skipped.append(p)
                continue
        flow, config = load_flow(p)
        flows.append(flow)
        names.append(_event_name_from_path(p))
        configs.append(config)
        kept.append(p)

    if skipped:
        import warnings

        warnings.warn(
            f"Skipped {len(skipped)}/{len(paths)} checkpoints that do not match "
            f"the installed flow library (likely flowjax version drift). "
            f"First few: {[q.name for q in skipped[:3]]}"
        )
    if not flows:
        raise CheckpointStructureError(
            "Every checkpoint mismatched the rebuilt skeleton; none could be "
            "loaded. This is a flowjax/equinox version mismatch."
        )

    dims = {int(c.get("data_dim", -1)) for c in configs}
    if len(dims) != 1 or -1 in dims:
        raise ValueError(
            f"All flows must share one data_dim; found {sorted(dims)}."
        )
    col_sets = {tuple(c.get("columns", ())) for c in configs}
    if len(col_sets) != 1:
        raise ValueError(
            f"All flows must share one column layout; found {sorted(col_sets)}."
        )
    columns = col_sets.pop()
    _validate_columns(columns)

    # Group by exact pytree structure + array-leaf shapes so each group can
    # be stacked on a leading model axis and evaluated with one vmap kernel.
    buckets: dict[Any, list[int]] = {}
    order: list[Any] = []
    for i, flow in enumerate(flows):
        sig = _structure_signature(flow)
        if sig not in buckets:
            buckets[sig] = []
            order.append(sig)
        buckets[sig].append(i)

    groups = []
    for sig in order:
        idx = buckets[sig]
        member_params = [eqx.filter(flows[i], eqx.is_array) for i in idx]
        stacked = jtu.tree_map(lambda *xs: jnp.stack(xs), *member_params)
        _, static = eqx.partition(flows[idx[0]], eqx.is_array)
        groups.append(FlowGroup(indices=tuple(idx), params=stacked, static=static))

    ens = FlowEnsemble(
        names=names,
        configs=configs,
        paths=kept,
        groups=groups,
        columns=columns,
        data_dim=dims.pop(),
    )
    ens.skipped = skipped
    return ens


def make_ensemble_log_prob(ensemble: FlowEnsemble) -> Callable:
    """Return ``eval_logflows(group_params, X) -> (n_flows, N)``.

    The returned callable is pure and traceable inside an outer ``jax.jit``:
    the equinox static halves and group index arrays live in its closure;
    the stacked parameter pytrees are traced operands so callers can
    barrier-wrap them against constant folding.  Each architecture group
    costs one ``eqx.filter_vmap`` kernel; row i of the output is the log
    density of ``ensemble.names[i]``.
    """
    statics = tuple(g.static for g in ensemble.groups)
    index_arrays = tuple(np.asarray(g.indices, dtype=np.int32) for g in ensemble.groups)
    n_flows = ensemble.n_flows

    def eval_logflows(group_params: tuple, X: jnp.ndarray) -> jnp.ndarray:
        X = jnp.atleast_2d(X)
        out = jnp.zeros((n_flows, X.shape[0]), dtype=X.dtype)
        for params, static, idx in zip(group_params, statics, index_arrays):
            group_flow = eqx.combine(params, static)

            def _single(flow):
                return flow.log_prob(X)

            vals = eqx.filter_vmap(_single)(group_flow)  # (G, N)
            out = out.at[jnp.asarray(idx)].set(vals)
        return out

    return eval_logflows


def make_ensemble_log_prob_per_event(ensemble: FlowEnsemble) -> Callable:
    """Return ``eval_logflows(group_params, X) -> (n_flows, J)`` for PER-EVENT X.

    Like :func:`make_ensemble_log_prob` but each flow scores its OWN batch:
    ``X`` has shape ``(n_flows, J, D)`` and row i is evaluated only by flow i.
    Used by the event-windowed population proposal, where every event draws
    its own points.
    """
    statics = tuple(g.static for g in ensemble.groups)
    index_arrays = tuple(np.asarray(g.indices, dtype=np.int32) for g in ensemble.groups)
    n_flows = ensemble.n_flows

    def eval_logflows(group_params: tuple, X: jnp.ndarray) -> jnp.ndarray:
        if X.ndim != 3 or X.shape[0] != n_flows:
            raise ValueError(
                f"per-event X must be (n_flows, J, D); got {X.shape} for "
                f"{n_flows} flows."
            )
        out = jnp.zeros((n_flows, X.shape[1]), dtype=X.dtype)
        for params, static, idx in zip(group_params, statics, index_arrays):
            group_flow = eqx.combine(params, static)
            X_g = X[jnp.asarray(idx)]  # (G, J, D)

            def _single(flow, x):
                return flow.log_prob(x)

            vals = eqx.filter_vmap(_single)(group_flow, X_g)  # (G, J)
            out = out.at[jnp.asarray(idx)].set(vals)
        return out

    return eval_logflows


def compute_support_boxes(
    ensemble: FlowEnsemble,
    key=None,
    n: int = 4096,
    margin: float = 0.25,
) -> dict:
    """Per-event support windows for the event-windowed population proposal.

    Samples each flow and takes the per-dimension min/max, expanded by
    ``margin`` times the sampled range on each side (chi_eff clipped to
    [-1, 1], masses/distances floored at tiny positives).  The windows only
    shape the PROPOSAL — the truncated samplers fold the window normaliser
    into the exact proposal density, so any box covering the flow's support
    leaves the estimator unbiased; ``margin`` trades a little efficiency for
    tail coverage.

    Returns a dict of (n_flows, 2) arrays: ``m1det``, ``q``, ``dL``,
    ``chieff``.
    """
    if ensemble.columns != SPECTRAL_COLUMNS:
        raise NotImplementedError(
            "compute_support_boxes currently assumes the spectral 4-D layout."
        )
    key = jax.random.key(0) if key is None else key
    s = np.asarray(ensemble_sample(ensemble, key, n))  # (n_flows, n, 4)
    m1det, m2det, dL, chieff = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    q = m2det / m1det
    # Detector-frame chirp mass: well-measured events have razor-thin
    # constant-Mc ridges in (m1det, m2det); the likelihood samples m1 inside
    # this band (at the drawn q and z), which an axis-aligned box would miss.
    mc_det = (m1det * m2det) ** 0.6 / (m1det + m2det) ** 0.2

    # Robust ranges: spline flows have occasional far-tail samples (a handful
    # of draws can sit tens of sigma out), and a min/max box inflated by one
    # outlier silently disables the window (measured: a Mc box 8x wider than
    # the posterior collapses that event's ESS to ~1), and some flows carry
    # genuinely fat spline tails. The [0.5, 99.5] percentile range plus the
    # margin keeps >~99% of the posterior mass per dimension; the clipped
    # tail mass biases ln Z_i at the <~1e-2 level, far below the per-event
    # Monte-Carlo error, and the margin recovers most of it.
    q_lo_pct, q_hi_pct = 0.5, 99.5

    def _box(arr, lo_floor=None, hi_ceil=None):
        lo = np.percentile(arr, q_lo_pct, axis=1)
        hi = np.percentile(arr, q_hi_pct, axis=1)
        pad = margin * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if lo_floor is not None:
            lo = np.maximum(lo, lo_floor)
        if hi_ceil is not None:
            hi = np.minimum(hi, hi_ceil)
        return jnp.asarray(np.stack([lo, hi], axis=1))

    # (q, chi_eff) degeneracy: many events constrain a chi_eff(q) combination
    # far better than either alone; a per-draw chi_eff band around the
    # per-event linear fit chi ~ a + b q (residual range +/- margin) lets the
    # proposal ride that ridge.  Exactly corrected downstream like every
    # other window.
    qm = np.median(q, axis=1, keepdims=True)
    cm = np.median(chieff, axis=1, keepdims=True)
    qvar = np.median((q - qm) ** 2, axis=1)
    b = np.median((q - qm) * (chieff - cm), axis=1) / np.maximum(qvar, 1e-12)
    a = cm[:, 0] - b * qm[:, 0]
    resid = chieff - (a[:, None] + b[:, None] * q)
    r_lo = np.percentile(resid, 0.5, axis=1)
    r_hi = np.percentile(resid, 99.5, axis=1)
    pad = margin * (r_hi - r_lo)

    return {
        "m1det": _box(m1det, lo_floor=0.1),
        "q": _box(q, lo_floor=1e-4, hi_ceil=1.0),
        "dL": _box(dL, lo_floor=0.1),
        "chieff": _box(chieff, lo_floor=-1.0, hi_ceil=1.0),
        "mc_det": _box(mc_det, lo_floor=0.1),
        "chi_ab": jnp.asarray(np.stack([a, b], axis=1)),
        "chi_resid": jnp.asarray(np.stack([r_lo - pad, r_hi + pad], axis=1)),
    }


def ensemble_sample(
    ensemble: FlowEnsemble, key, n: int
) -> jnp.ndarray:
    """Draw ``n`` samples from every flow (diagnostics; host-side use).

    Returns an array of shape ``(n_flows, n, data_dim)`` in ensemble order.
    """
    keys = jax.random.split(key, ensemble.n_flows)
    out: list[Optional[jnp.ndarray]] = [None] * ensemble.n_flows
    for g in ensemble.groups:
        group_flow = eqx.combine(g.params, g.static)
        gkeys = jnp.stack([keys[i] for i in g.indices])

        def _single(flow, k):
            return flow.sample(k, (int(n),))

        vals = eqx.filter_vmap(_single)(group_flow, gkeys)  # (G, n, D)
        for local, gi in enumerate(g.indices):
            out[gi] = vals[local]
    return jnp.stack(out, axis=0)
