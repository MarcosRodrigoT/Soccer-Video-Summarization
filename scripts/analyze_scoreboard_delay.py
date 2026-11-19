#!/usr/bin/env python3
"""Scoreboard-update delay analysis (reviewer comment R1.1).

For every ground-truth goal, the delay is measured between the end of the
live goal-leading shot (``end_frame_shot``, the cut to celebration/replay)
and the frame where the on-screen score changes (``end_frame_scoreboard``).
This is the gap the anchoring rule has to bridge. Since the goal itself
happens before the end of that shot, it is a lower bound of the delay with
respect to the goal instant.

For each goal it also reports the outcome under the primary protocol
(clip IoU from ``results_hr_10s``) and how many detected play long shots
(``highlights.csv``) end strictly between the end of the goal shot and the
score update, i.e. shots that could capture the anchor instead.

Usage:
    python scripts/analyze_scoreboard_delay.py \\
        --config config/default_hr_10s.yaml \\
        --output results_hr_10s/aggregate/scoreboard_delay.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FPS = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_hr_10s.yaml")
    ap.add_argument("--output", default="results_hr_10s/aggregate/scoreboard_delay.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_dir = Path(cfg["data_dir"])
    results_dir = Path(cfg["results_dir"])
    matches = yaml.safe_load(open(Path(args.config).parent / "matches_hr.yaml"))["matches"]
    lookback = cfg["highlight_assembler"]["goal_block_max_lookback"]

    goals = []
    for match_id in sorted(matches, key=lambda m: int(m.replace("Match", ""))):
        gdf = pd.read_csv(data_dir / matches[match_id]["goals_path"], encoding="latin-1")
        ev = json.load(open(results_dir / "per_match" / match_id / "evaluation.json"))
        hl = pd.read_csv(results_dir / "per_match" / match_id / "highlights.csv")
        for k, row in gdf.reset_index(drop=True).iterrows():
            shot_end = row["end_frame_shot"] / FPS
            score_t = row["end_frame_scoreboard"] / FPS
            between = int(((hl["end_time"] > shot_end + 0.5) & (hl["end_time"] < score_t)).sum())
            iou = ev["clip_iou_per_gt"][k]
            goals.append({
                "match": match_id,
                "goal": int(row["goal_id"]),
                "delay_sec": round(score_t - shot_end, 2),
                "play_long_shots_in_between": between,
                "clip_iou": None if iou is None else round(iou, 4),
                "end_to_end_tp": bool(iou is not None and iou >= 0.5),
                "note": "" if pd.isna(row["notes"]) else str(row["notes"]),
            })

    d = np.array([g["delay_sec"] for g in goals])
    out = {
        "definition": "delay = end_frame_scoreboard - end_frame_shot (seconds, 25 fps)",
        "num_goals": len(goals),
        "min": float(d.min()), "median": float(np.median(d)), "mean": float(d.mean()),
        "p90": float(np.percentile(d, 90)), "max": float(d.max()),
        "num_le_0": int((d <= 0).sum()),
        "num_le_10": int((d <= 10).sum()),
        "num_le_20": int((d <= 20).sum()),
        "num_gt_lookback": int((d > lookback).sum()),
        "lookback_sec": lookback,
        "max_excluding_gt_lookback": float(d[d <= lookback].max()),
        "num_with_play_long_shot_in_between": int(sum(g["play_long_shots_in_between"] > 0 for g in goals)),
        "goals": goals,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)

    print(f"{len(goals)} goals: delay min {d.min():.1f}s, median {np.median(d):.1f}s, "
          f"mean {d.mean():.1f}s, P90 {np.percentile(d, 90):.1f}s, max {d.max():.1f}s")
    print(f"<=0 s: {out['num_le_0']}, <=10 s: {out['num_le_10']}, <=20 s: {out['num_le_20']}, "
          f">{lookback:.0f} s: {out['num_gt_lookback']}")
    print(f"goals with a play long shot between shot end and score update: "
          f"{out['num_with_play_long_shot_in_between']}")
    for g in goals:
        if g["delay_sec"] > 20 or g["play_long_shots_in_between"] or not g["end_to_end_tp"]:
            print(" ", g["match"], g["goal"], g["delay_sec"], "between:", g["play_long_shots_in_between"],
                  "IoU:", g["clip_iou"], "TP:", g["end_to_end_tp"], g["note"][:70])


if __name__ == "__main__":
    main()
