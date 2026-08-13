#!/usr/bin/env python
"""Subset a gwcat PE store to the events of a flow-checkpoint directory.

Accepts gwcat-1.0 / gwcat-pe-2.0 / gwcat-pe-2.1 stores (any other
format_version is refused rather than silently copied with stale metadata).
Every attr whose leading axis is (nobs,) -- spin_amax_*_per_event,
cosmology_*_per_event, the sample-set provenance arrays -- is reindexed to
the subset; scalar and space-level attrs (contract, contract_hash,
fit_columns, block_provenance) are invariant under an event subset and are
copied verbatim.

Produces a file whose events are exactly the flow ensemble's, in
the ensemble's (lexicographically sorted) order — the same order the
flow-surrogate likelihood uses — so a stored-PE baseline run and a flow run
index events identically.

Usage:
    python scripts/subset_pe_store.py \
        --store  <gwcat-1.0 file with event_names attr> \
        --flows  <dir of <EVENT>/<EVENT>_flow.npz> \
        --output <subset .h5>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _decode(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--flows", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--pattern", default="*/*_flow.npz")
    args = ap.parse_args()

    flow_paths = sorted(
        p for p in Path(args.flows).glob(args.pattern) if "__MACOSX" not in p.parts
    )
    flow_names = [p.name[: -len("_flow.npz")] for p in flow_paths]
    if not flow_names:
        raise SystemExit(f"No flow checkpoints under {args.flows}")

    _ACCEPTED_FORMATS = ("gwcat-1.0", "gwcat-pe-2.0", "gwcat-pe-2.1")

    with h5py.File(args.store, "r") as f:
        fmt = _decode(f.attrs.get("format_version", ""))
        if fmt not in _ACCEPTED_FORMATS:
            raise SystemExit(
                f"Refusing to subset format_version {fmt!r}: this tool knows "
                f"how to reindex {_ACCEPTED_FORMATS} only. Copying an unknown "
                "format would silently carry event-indexed metadata it "
                "cannot recognise."
            )
        names = [_decode(n) for n in f.attrs["event_names"]]
        nsamp = int(f.attrs["nsamp"])
        nobs = int(f.attrs["nobs"])
        index = {n: i for i, n in enumerate(names)}

        missing = [n for n in flow_names if n not in index]
        if missing:
            raise SystemExit(
                f"{len(missing)} flow event(s) not in the store: {missing[:5]} ..."
            )
        rows = [index[n] for n in flow_names]

        datasets = {}
        for key in f.keys():
            arr = f[key][...]
            if arr.shape[0] != nobs * nsamp:
                raise SystemExit(f"Dataset {key} has unexpected length {arr.shape}")
            blocks = arr.reshape(nobs, nsamp, *arr.shape[1:])
            datasets[key] = blocks[rows].reshape(len(rows) * nsamp, *arr.shape[1:])

        attrs = dict(f.attrs)

    # Update event-indexed attributes.
    attrs["nobs"] = len(rows)
    attrs["subset_of"] = str(args.store)
    attrs["subset_n_original_events"] = nobs
    attrs["event_names"] = np.asarray(flow_names, dtype=object)
    for key, val in list(attrs.items()):
        if key in ("nobs", "event_names"):
            continue
        if isinstance(val, np.ndarray) and val.shape[:1] == (nobs,):
            attrs[key] = val[rows]

    with h5py.File(args.output, "w") as out:
        for key, arr in datasets.items():
            out.create_dataset(key, data=arr)
        for key, val in attrs.items():
            try:
                out.attrs[key] = val
            except TypeError:
                out.attrs[key] = np.asarray(val, dtype="S")

    print(
        f"Wrote {args.output}: {len(rows)} events x {nsamp} samples "
        f"(from {nobs}-event store), order = sorted flow names."
    )


if __name__ == "__main__":
    main()
