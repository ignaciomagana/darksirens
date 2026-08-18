#!/usr/bin/env bash
# Tier E, chunked.  Each chunk is its own process: the outer bias profile
# re-traces the inner solve on every trip, and ~500 XLA compilations in one
# process exhausts the JIT executable pool ("JIT session error: Cannot allocate
# memory", hit twice at realization 19/20).  Chunking changes no number -- the
# seeds are seed0 + step*(offset + k).
set -eu
cd "$(dirname "$0")/../../.."
export JAX_PLATFORMS=cpu DARKSIRENS_ZMAX=1.5 PYTHONPATH=.
D=experiments/field_level_plan/pr7
CH=${CHUNK:-4}
N=${NREAL:-20}
for ((o=0; o<N; o+=CH)); do
  python $D/tier_e.py --n-real $CH --seed-offset $o \
      --prior-sweep --sweep-n-real 0 --out $D/tier_e_chunk_$o.json \
      2>&1 | tee $D/tier_e_chunk_$o.log
done
# The prior sweep and the curvature verification, one process each.
# One process per prior width. --n-real must be >= --sweep-n-real: the sweep
# reuses the campaign's first seeds, so a zero-length seed list silently
# produces an EMPTY sweep (it did, on the first pass).
for s in 0.25 0.5 1.0 2.0 4.0; do
  python $D/tier_e.py --n-real 5 --prior-sweep $s --sweep-n-real 5 \
      --out $D/tier_e_sweep_$s.json 2>&1 | tee $D/tier_e_sweep_$s.log
done
python $D/tier_e.py --n-real 1 --verify --prior-sweep --sweep-n-real 0 \
    --out $D/tier_e_verify.json 2>&1 | tee $D/tier_e_verify.log
python $D/merge_tier_e.py --dir $D --out $D/tier_e.json
