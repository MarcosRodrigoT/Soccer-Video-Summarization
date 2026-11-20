#!/bin/bash
# Clean wall-clock timing of the three shot boundary detectors on Match1
# (run one after another on an otherwise idle machine; R1.2).
set -e
cd /home/mrt/Projects/Soccer-Video-Summarization
source venv/bin/activate
OUT=results_sbd/timing_match1
mkdir -p $OUT
python - <<'PY' > $OUT/ours.json
import json, time, glob
from soccer_summarization.config import PipelineConfig
from soccer_summarization.shot_segmentation import ShotSegmenter
cfg = PipelineConfig.from_yaml("config/default_hr.yaml")
v = glob.glob("/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos/Laliga_HR/Match1/*.mp4")[0]
t0 = time.perf_counter(); shots, _ = ShotSegmenter(cfg.shot_segmentation).run(v); t = time.perf_counter() - t0
print(json.dumps({"method": "ours", "seconds": t, "num_shots": len(shots), "device": "CPU (32 processes)"}))
PY
for m in transnetv2 pyscenedetect; do
  rm -rf $OUT/$m
  python scripts/run_sbd_baseline.py --method $m --matches Match1 --out-root $OUT/$m > /dev/null 2>&1
  cp $OUT/$m/$m/per_match/Match1/timing.json $OUT/$m.json
done
cat $OUT/*.json
