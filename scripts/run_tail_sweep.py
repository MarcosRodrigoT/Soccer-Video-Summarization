#!/usr/bin/env python3
"""Sensitivity of the end-to-end results to the clip length Delta_tail
(reviewer comment R1.1).

Re-runs only highlight assembly and evaluation for several values of
``goal_clip_max_duration``, reusing the cached shot boundaries, player
counts and score events of ``results_hr/per_match`` (no video is
re-processed). Both predictions and ground-truth clips are truncated to the
same Delta_tail, as in the paper. Delta_tail = 10 s and = inf must reproduce
results_hr_10s/ and results_hr/ exactly; the script checks it.

Usage:
    python scripts/run_tail_sweep.py \\
        --output results_hr_10s/aggregate/tail_sweep.json
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_single_match import load_match_info, run_pipeline  # noqa: E402
from soccer_summarization.config import PipelineConfig  # noqa: E402
from soccer_summarization.metrics_summary import summarize  # noqa: E402

CACHED = ["shot_boundaries.csv", "player_counts.csv", "frame_classification.csv",
          "score_events.csv", "scoreboard_region.json"]
TAILS = [5.0, 7.5, 10.0, 15.0, 20.0, None]  # None = full shot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_hr_10s.yaml")
    ap.add_argument("--cache-dir", default="results_hr/per_match")
    ap.add_argument("--output", default="results_hr_10s/aggregate/tail_sweep.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    matches = yaml.safe_load(open(Path(args.config).parent / "matches_hr.yaml"))["matches"]
    match_ids = sorted(matches, key=lambda m: int(m.replace("Match", "")))

    sweep = []
    with tempfile.TemporaryDirectory() as tmp:
        for tail in TAILS:
            cfg = PipelineConfig.from_yaml(args.config)
            cfg.highlight_assembler.goal_clip_max_duration = tail
            evals = []
            for match_id in match_ids:
                out = Path(tmp) / f"tail_{tail}" / match_id
                out.mkdir(parents=True)
                for name in CACHED:
                    src = Path(args.cache_dir) / match_id / name
                    if src.exists():
                        os.link(src, out / name)
                info = load_match_info(match_id, cfg, matches_yaml=str(
                    Path(args.config).parent / "matches_hr.yaml"))
                run_pipeline(video_path=info["video_path"], gt_path=info["gt_path"],
                             output_dir=str(out), config=cfg, match_id=match_id,
                             skip_video_export=True, goals_path=info.get("goals_path"))
                evals.append(json.load(open(out / "evaluation.json")))
            s = summarize(evals)
            s["tail_sec"] = tail
            sweep.append(s)
            print(f"tail={tail}: TP/FP/FN {s['tp']}/{s['fp']}/{s['fn']}  "
                  f"F1 {s['f1_mean']:.4f}±{s['f1_std']:.4f}  IoU {s['iou_mean']:.4f}  "
                  f"MAE {s['mae_start_mean']:.3f}/{s['mae_end_mean']:.3f}  "
                  f"compression {s['compression_mean']:.4f}")

    # Consistency with the canonical result directories used in the paper.
    for tail, ref_dir in ((10.0, "results_hr_10s"), (None, "results_hr")):
        ref = summarize([json.load(open(Path(ref_dir) / "per_match" / m / "evaluation.json"))
                         for m in match_ids])
        got = next(s for s in sweep if s["tail_sec"] == tail)
        diff = {k: (got[k], ref[k]) for k in ref if abs(got[k] - ref[k]) > 1e-9}
        if diff:
            sys.exit(f"tail={tail} does not reproduce {ref_dir}: {diff}")
        print(f"tail={tail} reproduces {ref_dir} exactly")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tails": sweep}, open(args.output, "w"), indent=2)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
