#!/usr/bin/env python3
"""Evaluate the cross-league matches against the frame-level ground truth
annotated with tools/annotator (annotations_v2/goals/<id>_goals.csv), with the
same protocol as the LaLiga corpus: only assembly + evaluation are re-run on
the cached outputs of results_other_leagues/per_match, for Delta_tail = 10 s
and full shot. Also reports the scoreboard-update delays.

    python scripts/evaluate_other_leagues.py --matches PL1 SA1
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_single_match import run_pipeline  # noqa: E402
from soccer_summarization.config import PipelineConfig  # noqa: E402
from soccer_summarization.metrics_summary import per_match_tp_mean, summarize  # noqa: E402

CACHED = ["shot_boundaries.csv", "player_counts.csv", "frame_classification.csv",
          "score_events.csv", "scoreboard_region.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", default=["PL1", "SA1"])
    ap.add_argument("--config", default="config/default_hr_10s.yaml")
    ap.add_argument("--output", default="results_other_leagues/evaluation_summary.json")
    args = ap.parse_args()
    meta = yaml.safe_load(open("config/matches_other_leagues.yaml"))["matches"]
    cfg0 = PipelineConfig.from_yaml(args.config)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for tail, name in ((10.0, "tail_10s"), (None, "full_shot")):
            cfg = PipelineConfig.from_yaml(args.config)
            cfg.highlight_assembler.goal_clip_max_duration = tail
            evals = []
            for m in args.matches:
                d = Path(tmp) / name / m
                d.mkdir(parents=True)
                for f in CACHED:
                    src = Path("results_other_leagues/per_match") / m / f
                    if src.exists():
                        os.link(src, d / f)
                gt = Path("annotations_v2/goals") / f"{m}_goals.csv"
                video = next((Path(cfg0.data_dir) / meta[m]["video_dir"]).glob("*.mp4"))
                dummy = d / "no_commentary_gt.csv"
                dummy.write_text("start,end,text,GoalGT1,GoalGT2\n")
                run_pipeline(video_path=str(video), gt_path=str(dummy), output_dir=str(d), config=cfg,
                             match_id=m, skip_video_export=True, goals_path=str(gt))
                e = json.load(open(d / "evaluation.json"))
                e["iou_tp_mean"] = per_match_tp_mean(e, "clip_iou_per_gt")
                e["mae_start_tp_mean"] = per_match_tp_mean(e, "clip_start_mae_per_gt")
                e["mae_end_tp_mean"] = per_match_tp_mean(e, "clip_end_mae_per_gt")
                evals.append(e)
            out[name] = {"summary": summarize(evals) if len(evals) > 1 else None,
                         "per_match": {e["match_id"]: {k: e[k] for k in (
                             "clip_aware_tp", "clip_aware_fp", "clip_aware_fn", "clip_aware_f1",
                             "true_positives", "false_positives", "false_negatives", "compression_ratio",
                             "iou_tp_mean", "mae_start_tp_mean", "mae_end_tp_mean", "clip_iou_per_gt")}
                             for e in evals}}
    delays = {}
    for m in args.matches:
        g = pd.read_csv(Path("annotations_v2/goals") / f"{m}_goals.csv")
        delays[m] = [round((r.end_frame_scoreboard - r.end_frame_shot) / 25, 2) for r in g.itertuples()]
    out["scoreboard_delay_sec"] = delays
    json.dump(out, open(args.output, "w"), indent=2)
    for name in ("tail_10s", "full_shot"):
        for m, r in out[name]["per_match"].items():
            print(f"{name:9s} {m}: TP/FP/FN {r['clip_aware_tp']}/{r['clip_aware_fp']}/{r['clip_aware_fn']} "
                  f"F1 {r['clip_aware_f1']:.3f} | OCR {r['true_positives']}/{r['false_positives']}/{r['false_negatives']} "
                  f"| IoU {r['iou_tp_mean']} MAEs {r['mae_start_tp_mean']} MAEe {r['mae_end_tp_mean']} "
                  f"| comp {r['compression_ratio']:.4f} | IoU per goal {[round(x, 2) if x is not None else None for x in r['clip_iou_per_gt']]}")
    print("delays", delays)


if __name__ == "__main__":
    main()
