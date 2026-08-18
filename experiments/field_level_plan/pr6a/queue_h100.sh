#!/bin/bash
# Continuous GPU work queue for the field-level ladder on the H100.
# Runs jobs back to back so the GPU never idles. Each job logs separately and
# a failure never stops the queue.
set -u
R=/media/volume/darksirens-data/darksirens-dev-data
source $R/env.sh
export DARKSIRENS_GWDATA_DIR=$R/data/gw
cd $R/repo/experiments/field_level_plan/pr6a
mkdir -p $R/runs/q $R/logs/q
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $R/logs/queue.log; }
run(){ n=$1; shift; log "START $n"; if "$@" > $R/logs/q/$n.log 2>&1; then log "  OK    $n"; else log "  FAIL  $n (see logs/q/$n.log)"; fi; }

log "===== QUEUE START ====="

# --- 1. Variance split at 8x8 (was 5x5): sharpen the event-vs-catalog split.
run variance_8x8 python variance_split.py --n-seed 8 --n-event 8 \
    --out $R/runs/q/variance_8x8.json

# --- 2. PE calibration across 8 INDEPENDENT mock realizations.  The single
#        realization measured resid_sd = 1.059 (PE is NOT over-sharp) but
#        resid_mean = +0.486 with KS p = 0.0055.  If that offset is stable
#        across realizations it is a DAG inconsistency, not noise.
for s in 8101 8102 8103 8104 8105 8106 8107 8108; do
  run mock_$s python make_mock.py --seed $s --outdir $R/runs/q/mock_$s \
      --reuse-injections data/rb/../injections.h5
  run pecal_$s python pe_calibration.py --gw $R/runs/q/mock_$s/gw_events.h5 \
      --out $R/runs/q/pecal_$s.json
done

# --- 3. Tier C at n=100 (the first pass ran 24, the second 50).  Doubling n
#        halves the error on the overconfidence ratio, which is the statistic
#        the whole diagnosis turns on.
run tier_c_n100 python tier_c.py --n-real 100 --seed0 7001 \
    --arms latent latent_off --out $R/runs/q/tier_c_n100.json

log "===== QUEUE DONE ====="
