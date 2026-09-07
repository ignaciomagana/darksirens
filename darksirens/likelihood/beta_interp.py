"""
beta_interp.py
--------------
Tabulated selection integral ("beta interpolator").

Why this exists
~~~~~~~~~~~~~~~
The selection integral mu(Lambda) is a sky- and population-averaged
detectability normalisation.  Evaluated by importance sampling over the
injection set it carries the FULL per-sample weight, including the per-pixel
catalog redshift prior p_z(z | pix):

    w_i = p_pop(m1src, q, z, chi) * p_z(z | pix) / [ddL/dz (1+z)] / p_draw

The injections are drawn isotropically and smoothly in redshift, while
p_z(z | pix) is a sum of narrow galaxy kernels at discrete galaxy redshifts
(sigma_eff_floor ~ 1e-5 for spectroscopic data).  The weight therefore asks
"is there a galaxy at exactly this (pixel, z)?", which is almost always no --
a rigid comb.  A handful of injections land on galaxies and carry nearly all
the weight, so N_eff collapses: measured N_eff_sel = 29.9 out of 50,070
detections (0.06% efficiency), against a required N_eff of 2,826.

That loss is a Monte-Carlo artifact, not physics.  mu is an INTEGRAL over the
sky and the population; where an injection happens to fall relative to
individual galaxies should average out of it.  Raising Ndraw does not help:
N_eff grows linearly with Ndraw but the efficiency is fixed by the comb, so
reaching the threshold needs ~95x more injections.

This module instead evaluates log mu once per node of a coarse grid over the
sampled parameters that mu actually depends on, and interpolates in between.
The comb variance is then paid once per node -- where it can be beaten down
with a large Ndraw offline -- instead of being re-incurred, and re-guarded, at
every sampler draw.

What mu depends on
~~~~~~~~~~~~~~~~~~
mu is a function of the cosmology and the population, NOT of the sampled
survey block: the catalog prior enters mu only through the same p_z used by
the PE numerator, and it is that shared dependence which makes the estimator
self-calibrating (see ``selection_prior_model`` in ``likelihood/core.py`` and
commit 20d1f5c / library review P0.2).  Tabulating mu against a parameter that
is being SAMPLED but is not an axis of the table silently freezes it, which is
exactly the bias that review flagged.  So a table is admissible only when
every parameter mu depends on is either an axis of the table or held fixed.
:func:`check_axes_cover_sampled` enforces that and is not optional.

Uncertainty
~~~~~~~~~~~
The tabulated path reports the interpolation error and the per-node Monte-Carlo
error instead of a per-draw N_eff.  N_eff is itself only a proxy for
sigma(log mu); with a table the quantity of interest is available directly:

  * per-node MC error: sigma_node = 1/sqrt(N_eff_node), measured at build time
    with the large offline Ndraw;
  * interpolation error: estimated by comparing the interpolant against a
    held-out subset of nodes (leave-one-out on the build grid).

Both are recorded in the table file and surfaced at load, so a run states its
selection-term error budget up front rather than rejecting draws one at a time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

# Parameters the selection integral, mu, can depend on
MU_DEPENDENT_LABELS: tuple[str, ...] = (
    "H0",
    "Om0",
    "w0",
    "wa",
)


@dataclass(frozen=True)
class BetaTable:
    """A tabulated ``log mu`` on a rectilinear grid.

    Attributes
    ----------
    labels : parameter label per axis, in axis order.
    nodes : per-axis 1-D node coordinates (strictly increasing).
    log_mu : ``log mu`` at every node, shape ``tuple(len(n) for n in nodes)``.
    log_mu_err : per-node Monte-Carlo error on ``log mu`` (same shape), i.e.
        ``1/sqrt(N_eff_node)``.
    interp_err : scalar leave-one-out estimate of the interpolation error [nats].
    ndraw : injections used per node at build time.
    meta : free-form provenance (build settings, fixed parameter values).
    """

    labels: tuple[str, ...]
    nodes: tuple[np.ndarray, ...]
    log_mu: np.ndarray
    log_mu_err: np.ndarray
    interp_err: float
    ndraw: float
    meta: Mapping[str, object]

    @property
    def sigma_log_mu(self) -> float:
        """Total selection-term error budget [nats]: MC and interpolation."""
        return float(np.sqrt(float(np.max(self.log_mu_err)) ** 2 + self.interp_err ** 2))


def check_axes_cover_sampled(
    table_labels: Sequence[str],
    sampled_labels: Sequence[str],
) -> None:
    """Raise unless EVERY sampled label is an axis of the table."""
    axes = set(table_labels)
    missing = [lab for lab in sampled_labels if lab not in axes]
    if missing:
        raise ValueError(f"beta function does not depend on parameter(s) {missing}.")


def load_beta_table(path: str) -> BetaTable:
    """Read a beta table written by :func:`save_beta_table`."""
    import h5py

    with h5py.File(path, "r") as f:
        labels = tuple(
            s.decode() if isinstance(s, bytes) else str(s) for s in f["labels"][:]
        )
        nodes = tuple(np.asarray(f[f"nodes/{i}"][:], dtype=float) for i in range(len(labels)))
        log_mu = np.asarray(f["log_mu"][:], dtype=float)
        log_mu_err = np.asarray(f["log_mu_err"][:], dtype=float)
        interp_err = float(f.attrs["interp_err"])
        ndraw = float(f.attrs["ndraw"])
        meta = json.loads(f.attrs.get("meta", "{}"))

    expected = tuple(len(n) for n in nodes)
    if log_mu.shape != expected:
        raise ValueError(
            f"beta table {path}: log_mu shape {log_mu.shape} does not match the "
            f"node grid {expected}."
        )
    for lab, n in zip(labels, nodes):
        if n.size < 2 or not np.all(np.diff(n) > 0):
            raise ValueError(
                f"beta table {path}: axis {lab!r} must have >= 2 strictly "
                "increasing nodes."
            )
    return BetaTable(labels, nodes, log_mu, log_mu_err, interp_err, ndraw, meta)


def save_beta_table(path: str, table: BetaTable) -> None:
    """Write a beta table, including its error budget and provenance."""
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = "darksirens-beta-table-1.0"
        f.attrs["interp_err"] = float(table.interp_err)
        f.attrs["ndraw"] = float(table.ndraw)
        f.attrs["meta"] = json.dumps(dict(table.meta), default=str)
        f.create_dataset("labels", data=np.array([s.encode() for s in table.labels]))
        grp = f.create_group("nodes")
        for i, n in enumerate(table.nodes):
            grp.create_dataset(str(i), data=np.asarray(n, dtype=float))
        f.create_dataset("log_mu", data=np.asarray(table.log_mu, dtype=float))
        f.create_dataset("log_mu_err", data=np.asarray(table.log_mu_err, dtype=float))


def _axis_weights(jnp, x, nodes):
    """Clamped linear-interpolation index pair and weight along one axis.

    Clamped, not extrapolated: mu outside the build grid is not measured, and
    linear extrapolation of a log-integral drifts without bound. The caller is
    responsible for keeping the prior inside the grid; ``beta_table_covers_prior``
    checks that eagerly at load so a run fails fast rather than silently
    evaluating a clamped mu.
    """
    n = nodes.shape[0]
    i1 = jnp.clip(jnp.searchsorted(nodes, x, side="right"), 1, n - 1)
    i0 = i1 - 1
    x0, x1 = nodes[i0], nodes[i1]
    t = jnp.clip((x - x0) / jnp.where(x1 > x0, x1 - x0, 1.0), 0.0, 1.0)
    return i0, i1, t


def make_log_mu_interpolator(table: BetaTable):
    """Build a jit-friendly multilinear interpolator for ``log mu``.

    Returns ``f(values) -> log_mu`` where ``values`` is a sequence of scalars in
    the table's axis order. Multilinear in log mu (a smooth, slowly varying
    function of the cosmology), so the interpolant is exact at the nodes and
    the error between them is the ``interp_err`` recorded in the table.
    """
    import jax.numpy as jnp

    nodes = [jnp.asarray(n, dtype=jnp.float64) for n in table.nodes]
    grid = jnp.asarray(table.log_mu, dtype=jnp.float64)
    ndim = len(nodes)

    def log_mu_at(values):
        idx, wts = [], []
        for k in range(ndim):
            i0, i1, t = _axis_weights(jnp, jnp.asarray(values[k], dtype=jnp.float64),
                                      nodes[k])
            idx.append((i0, i1))
            wts.append((1.0 - t, t))
        # Sum the 2**ndim corners of the enclosing cell. ndim is small (<= 5)
        # and static, so this unrolls at trace time.
        acc = jnp.asarray(0.0, dtype=jnp.float64)
        for corner in range(1 << ndim):
            w = jnp.asarray(1.0, dtype=jnp.float64)
            sel = []
            for k in range(ndim):
                bit = (corner >> k) & 1
                w = w * wts[k][bit]
                sel.append(idx[k][bit])
            acc = acc + w * grid[tuple(sel)]
        return acc

    return log_mu_at


def make_likelihood_log_mu_fn(table: BetaTable):
    """Wrap a table as the ``log_mu_table_fn(cosmo, pop_params)`` the likelihood calls."""
    interp = make_log_mu_interpolator(table)
    getters = []
    for lab in table.labels:
        if lab == "H0":
            getters.append(lambda c: c.H0)
        elif lab == "Om0":
            getters.append(lambda c: c.Om0)
        elif lab == "w0":
            getters.append(lambda c: c.w0)
        elif lab == "wa":
            getters.append(lambda c: c.wa)
        else:
            raise ValueError(
                f"beta cannot depend upon parameter {lab!r}; "
                f"axes must be drawn from {MU_DEPENDENT_LABELS}."
            )

    def log_mu_table_fn(cosmo, pop_params=None):
        return interp([g(cosmo) for g in getters])

    return log_mu_table_fn


def beta_table_covers_prior(
    table: BetaTable,
    lower: Mapping[str, float],
    upper: Mapping[str, float],
) -> list[str]:
    """Return axes whose prior range escapes the table's node span.

    Interpolation is clamped at the grid edge, so a prior wider than the table
    would evaluate a constant mu out there and bias the posterior toward the
    edge. Reported per axis so the caller can widen the grid or narrow the prior.
    """
    bad = []
    for lab, n in zip(table.labels, table.nodes):
        lo, hi = lower.get(lab), upper.get(lab)
        if lo is None or hi is None:
            continue
        if lo < float(n[0]) or hi > float(n[-1]):
            bad.append(
                f"{lab}: prior [{lo:g}, {hi:g}] escapes table span "
                f"[{float(n[0]):g}, {float(n[-1]):g}]"
            )
    return bad