#!/bin/bash
# Orchestrate full experiment run for the Matlab-equivalent shot segmentation.
# Writes everything under results_hr_matlab/ — leaves results_hr/ untouched.

set -e
cd /home/mrt/Projects/Soccer-Video-Summarization
source venv/bin/activate

LOG=results_hr_matlab/pipeline.log
mkdir -p results_hr_matlab
date '+%F %T  STARTING pipeline run' > "$LOG"

# ── Stage 1: end-to-end pipeline (reuses cached YOLO/OCR via hardlinks) ───────
echo "" >> "$LOG"
date '+%F %T  Stage 1: run_all_matches' >> "$LOG"
python scripts/run_all_matches.py \
    --config config/default_hr_matlab.yaml \
    --matches-yaml config/matches_hr.yaml \
    --skip-video-export \
    >> "$LOG" 2>&1
date '+%F %T  Stage 1 complete' >> "$LOG"

# ── Stage 2: dashboards ───────────────────────────────────────────────────────
echo "" >> "$LOG"
date '+%F %T  Stage 2: generate_match_dashboard' >> "$LOG"
python scripts/generate_match_dashboard.py \
    --config config/default_hr_matlab.yaml \
    --matches-yaml config/matches_hr.yaml \
    --results-dir results_hr_matlab/per_match \
    --output-dir results_hr_matlab/dashboards \
    >> "$LOG" 2>&1
date '+%F %T  Stage 2 complete' >> "$LOG"

# ── Stage 3: goal clips video ─────────────────────────────────────────────────
echo "" >> "$LOG"
date '+%F %T  Stage 3: generate_goal_clips' >> "$LOG"
python scripts/generate_goal_clips.py \
    --config config/default_hr_matlab.yaml \
    --matches-yaml config/matches_hr.yaml \
    --results-dir results_hr_matlab/per_match \
    --output results_hr_matlab/goal_clips/all_goals.mp4 \
    --scale 0.5 --fps 25 \
    >> "$LOG" 2>&1
date '+%F %T  Stage 3 complete' >> "$LOG"

echo "" >> "$LOG"
date '+%F %T  ALL DONE' >> "$LOG"
