#!/usr/bin/env python
"""Subset a gwcat-1.0 PE store to the events of a flow-checkpoint directory.

Produces a gwcat-1.0 file whose events are exactly the flow ensemble's, in
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

    with h5py.File(args.store, "r") as f:
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
