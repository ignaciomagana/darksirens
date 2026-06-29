"""
events.py
---------
Factory for constructing GWEvent containers with proper JAX barrier
wrapping and pre-computed mass ratio.

Why barriers on GW data?
~~~~~~~~~~~~~~~~~~~~~~~~
``lax.optimization_barrier`` tells XLA that an array is an opaque
runtime value, preventing constant-folding from materialising large
intermediate tensors in the HLO graph at compile time.

Without barriers, JAX sees the captured data arrays as compile-time
constants during JIT tracing and may attempt to evaluate operations
on them at compile time.  For O(200k) sample arrays this produces
enormous HLO graphs, slow compilation, and in the worst case an OOM
during the compile step — not the run step — which is notoriously
hard to diagnose.

The *correct* place to apply barriers is here, before the arrays are
captured in any JIT closure.  Applying them inside the likelihood body
is too late: JAX has already ingested the raw values during tracing.

Why q here?
~~~~~~~~~~~
``GWEvent.q = m2det / m1det`` is a derived quantity used in every
likelihood evaluation.  As a NamedTuple property it would be
recomputed on every access; stored explicitly it is computed once,
barrier-wrapped alongside the raw arrays, and reused cheaply.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import lax

from darksirens.core.types import GWEvent


def _barrier(arr: jnp.ndarray) -> jnp.ndarray:
    """
    Wrap a single array with ``lax.optimization_barrier``.

    This is the single canonical definition.  It was previously
    duplicated in ``likelihood.py``; it now lives only here.
    """
    return lax.optimization_barrier(jnp.asarray(arr))


def make_gw_event(
    m1det,
    m2det,
    dL,
    chieff,
    prior_wt,
    pixels,
    valid=None,
    nx=None,
    ny=None,
    nz=None,
) -> GWEvent:
    """
    Construct a ``GWEvent`` with barrier-wrapped arrays and pre-computed ``q``.

    Parameters
    ----------
    m1det, m2det : array-like
        Detector-frame masses [M_sun].
    dL : array-like
        Luminosity distance [Mpc].
    chieff : array-like
        Effective inspiral spin.
    prior_wt : array-like
        PE prior weights (normalised per event).
    pixels : array-like of int
        HEALPix pixel indices.
    valid : array-like of bool, optional
        Explicit structural mask.  Defaults to all True; padding helpers set
        padded entries to False so downstream log-sum-exp code can reject
        them without relying on physical sentinel values.
    nx, ny, nz : array-like, optional
        Sky-direction unit-vector components per sample (the angular analog of
        ``dL``).  Default to zeros when unspecified — harmless, because they are
        only read by an anisotropic sky model; an isotropic run never touches
        them.

    Returns
    -------
    GWEvent
        All floating-point fields are barrier-wrapped.  ``q`` is computed
        from the barrier-wrapped ``m2det / m1det`` so XLA cannot trace
        back through the division to the raw constants.
    """
    m1det_b   = _barrier(jnp.asarray(m1det,    dtype=jnp.float64))
    m2det_b   = _barrier(jnp.asarray(m2det,    dtype=jnp.float64))
    dL_b      = _barrier(jnp.asarray(dL,       dtype=jnp.float64))
    chieff_b  = _barrier(jnp.asarray(chieff,   dtype=jnp.float64))
    prior_wt_b = _barrier(jnp.asarray(prior_wt, dtype=jnp.float64))
    # Integer pixels: barrier still prevents constant-folding of the
    # ang2pix indexing chain, which can be large.
    pixels_b  = _barrier(jnp.asarray(pixels,   dtype=jnp.int32))
    if valid is None:
        valid = jnp.ones_like(dL_b, dtype=bool)
    valid_b   = _barrier(jnp.asarray(valid, dtype=bool))
    q_b       = _barrier(m2det_b / m1det_b)

    def _sky(v):
        if v is None:
            return _barrier(jnp.zeros_like(dL_b))
        return _barrier(jnp.asarray(v, dtype=jnp.float64))

    return GWEvent(
        m1det    = m1det_b,
        m2det    = m2det_b,
        dL       = dL_b,
        chieff   = chieff_b,
        prior_wt = prior_wt_b,
        pixels   = pixels_b,
        q        = q_b,
        valid    = valid_b,
        nx       = _sky(nx),
        ny       = _sky(ny),
        nz       = _sky(nz),
    )


def pad_gw_event_to_multiple(
    event: GWEvent,
    multiple: int,
    fill_prior_wt: float = 0.0,
) -> tuple[GWEvent, int]:
    """
    Pad a ``GWEvent`` so that its length is a multiple of ``multiple``.

    Used when ``sel_batch_size`` is set: ``lax.scan`` processes fixed-size
    chunks, so callers either need ceiling batches with a mask or an input
    length that divides evenly into batches.  This helper implements the
    latter by appending sentinel injections with ``valid == False``.
    Selection batching code treats this explicit mask as structural and
    converts those entries to ``-inf`` log weight, so padded rows contribute
    exactly zero to both first- and second-moment sums without relying on any
    high-distance/redshift/population heuristic.

    Parameters
    ----------
    event : GWEvent
    multiple : int
    fill_prior_wt : float
        Fill value for ``prior_wt`` on padded entries.  The explicit
        ``valid`` mask, not this value, identifies padded rows.

    Returns
    -------
    (GWEvent, int)
        Event with length rounded up to the nearest ``multiple`` and the
        number of padding entries added.
    """
    if multiple <= 0:
        raise ValueError("multiple must be positive")

    N = event.dL.shape[0]
    remainder = N % multiple
    if remainder == 0:
        return event, 0

    pad = multiple - remainder

    def _pad1d(arr, fill=0.0):
        return jnp.concatenate([arr, jnp.full(pad, fill, dtype=arr.dtype)])

    padded = make_gw_event(
        m1det    = _pad1d(event.m1det,    fill=30.0),
        m2det    = _pad1d(event.m2det,    fill=30.0),
        dL       = _pad1d(event.dL,       fill=event.dL[0]),
        chieff   = _pad1d(event.chieff,   fill=0.0),
        prior_wt = _pad1d(event.prior_wt, fill=fill_prior_wt),
        pixels   = _pad1d(event.pixels.astype(np.int32), fill=0),
        valid    = _pad1d(event.valid,    fill=False),
        # Sky direction: pad with a finite placeholder (masked out by valid).
        nx       = _pad1d(event.nx, fill=0.0) if event.nx is not None else None,
        ny       = _pad1d(event.ny, fill=0.0) if event.ny is not None else None,
        nz       = _pad1d(event.nz, fill=0.0) if event.nz is not None else None,
    )
    return padded, pad
