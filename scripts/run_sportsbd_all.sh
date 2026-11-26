#!/bin/bash
# SportSBD on the 27 broadcasts (25 LaLiga + PL1, SA1): 6 processes over two GPUs.
# Match1 was timed alone (one process, SPORTSBD_BATCH=64): results_sbd/timing_match1/sportsbd.json.
cd "$(dirname "$0")/.."
source venv/bin/activate
export SPORTSBD_WEIGHTS=${SPORTSBD_WEIGHTS:-data/models/sportsbd_best.pt} OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
mkdir -p results_sbd/logs/sportsbd
M=($(seq -f "Match%g" 1 25) PL1 SA1)
for w in 0 1 2 3 4 5; do
  L=(); for i in "${!M[@]}"; do [ $((i % 6)) -eq $w ] && L+=("${M[$i]}"); done
  CUDA_VISIBLE_DEVICES=$((w % 2)) python scripts/run_sbd_baseline.py --method sportsbd --matches "${L[@]}" \
    > results_sbd/logs/sportsbd/worker$w.log 2>&1 &
done
wait
