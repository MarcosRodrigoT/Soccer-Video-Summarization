#!/usr/bin/env python3
"""Error analysis: categorize TP, FP, FN for each match.

For each goal event, determines:
- TP: Which component detected it (OCR score change? Play segment overlap?)
- FN: Why it was missed (OCR failure, no-play classification, short shot, etc.)
- FP: Why false alarm (OCR misread, replay, format change)

Additional analyses:
- Per-season breakdown
- Goal count vs F1 correlation
- Early vs late match goal detection

Usage:
    python scripts/error_analysis.py [--results-dir results/per_match]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator, GoalEvent
from soccer_summarization.utils import setup_logging


def categorize_events(
    match_id: str,
    match_dir: Path,
    gt_events: list[GoalEvent],
    predicted_goal_times: list[float],
    highlight_segments: list[tuple[float, float]],
    tolerance: float,
    total_duration: float,
) -> list[dict]:
    """Categorize each GT event and each prediction."""
    logger = logging.getLogger(__name__)
    categories = []

    # Load play segments for analysis
    play_path = match_dir / "play_segments.csv"
    play_df = pd.read_csv(play_path) if play_path.exists() else pd.DataFrame()

    # Load shot boundaries
    shot_path = match_dir / "shot_boundaries.csv"
    shot_df = pd.read_csv(shot_path) if shot_path.exists() else pd.DataFrame()

    # Load score events
    score_path = match_dir / "score_events.csv"
    try:
        score_df = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        score_df = pd.DataFrame()

    # Match GT events to predictions
    matched_gt = set()
    matched_pred = set()

    for i, pred_time in enumerate(predicted_goal_times):
        for j, gt_event in enumerate(gt_events):
            if gt_event.start_time - tolerance <= pred_time <= gt_event.end_time + tolerance:
                matched_gt.add(j)
                matched_pred.add(i)
                break

    # Categorize each GT event
    for j, event in enumerate(gt_events):
        event_mid = event.midpoint
        relative_time = event_mid / total_duration if total_duration > 0 else 0

        entry = {
            "match_id": match_id,
            "event_type": "goal",
            "gt_start": event.start_time,
            "gt_end": event.end_time,
            "gt_midpoint": event_mid,
            "team": event.team,
            "relative_time": relative_time,
            "half": "first" if relative_time < 0.5 else "second",
        }

        if j in matched_gt:
            # True Positive
            entry["category"] = "TP"

            # Determine which component detected it
            ocr_detected = False
            play_covered = False
            highlight_covered = False

            # Check OCR
            if not score_df.empty and "timestamp" in score_df.columns:
                for _, row in score_df.iterrows():
                    if event.start_time - tolerance <= row["timestamp"] <= event.end_time + tolerance:
                        ocr_detected = True
                        break

            # Check if in play segment
            if not play_df.empty:
                for _, row in play_df.iterrows():
                    if row["is_play"] and row["start_time"] <= event_mid <= row["end_time"]:
                        play_covered = True
                        break

            # Check if in highlight
            for hl_start, hl_end in highlight_segments:
                if hl_start <= event_mid <= hl_end:
                    highlight_covered = True
                    break

            entry["ocr_detected"] = ocr_detected
            entry["play_covered"] = play_covered
            entry["highlight_covered"] = highlight_covered
            entry["detection_source"] = []
            if ocr_detected:
                entry["detection_source"].append("ocr")
            if play_covered:
                entry["detection_source"].append("play")
        else:
            # False Negative
            entry["category"] = "FN"

            # Determine why it was missed
            reasons = []

            # Check OCR
            ocr_detected = False
            if not score_df.empty and "timestamp" in score_df.columns:
                for _, row in score_df.iterrows():
                    if event.start_time - tolerance <= row["timestamp"] <= event.end_time + tolerance:
                        ocr_detected = True
                        break

            if not ocr_detected:
                reasons.append("ocr_miss")

            # Check play classification
            play_covered = False
            if not play_df.empty:
                for _, row in play_df.iterrows():
                    if row["is_play"] and row["start_time"] <= event_mid <= row["end_time"]:
                        play_covered = True
                        break

            if not play_covered:
                reasons.append("classified_no_play")

            # Check shot filtering
            in_long_shot = False
            if not shot_df.empty:
                for _, row in shot_df.iterrows():
                    if (row["start_time"] <= event_mid <= row["end_time"]
                            and row["duration"] >= 4.0):
                        in_long_shot = True
                        break

            if not in_long_shot:
                reasons.append("short_shot")

            # Check highlight coverage
            in_highlight = False
            for hl_start, hl_end in highlight_segments:
                if hl_start <= event_mid <= hl_end:
                    in_highlight = True
                    break

            if not in_highlight:
                reasons.append("not_in_highlight")

            entry["fn_reasons"] = reasons
            entry["ocr_detected"] = ocr_detected
            entry["play_covered"] = play_covered

        categories.append(entry)

    # Categorize false positives (predicted goals not matching any GT)
    for i, pred_time in enumerate(predicted_goal_times):
        if i not in matched_pred:
            relative_time = pred_time / total_duration if total_duration > 0 else 0

            fp_entry = {
                "match_id": match_id,
                "event_type": "false_alarm",
                "category": "FP",
                "predicted_time": pred_time,
                "relative_time": relative_time,
                "half": "first" if relative_time < 0.5 else "second",
            }

            # Try to determine cause
            fp_reasons = []
            # Could be OCR misread, replay, etc.
            if not score_df.empty and "timestamp" in score_df.columns:
                close_events = score_df[
                    (score_df["timestamp"] - pred_time).abs() < 10
                ]
                if len(close_events) > 0:
                    fp_reasons.append("ocr_misread")

            fp_entry["fp_reasons"] = fp_reasons if fp_reasons else ["unknown"]
            categories.append(fp_entry)

    return categories


def main():
    parser = argparse.ArgumentParser(description="Run error analysis")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--output-dir", default="results/error_analysis")
    parser.add_argument("--matches", help="Comma-separated match IDs")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = PipelineConfig.from_yaml(args.config)
    evaluator = Evaluator(config.evaluation)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        all_matches = yaml.safe_load(f)["matches"]

    if args.matches:
        match_ids = [m.strip() for m in args.matches.split(",")]
    else:
        match_ids = [
            d.name for d in sorted(results_dir.iterdir())
            if d.is_dir() and d.name in all_matches
        ]

    all_events = []

    for match_id in match_ids:
        match_dir = results_dir / match_id
        if not match_dir.exists():
            continue

        info = all_matches[match_id]
        gt_path = str(Path(config.data_dir) / info["gt_path"])

        # Load GT
        gt_df = evaluator.load_ground_truth(gt_path)
        gt_events = evaluator.extract_goal_events(gt_df)

        # Load predictions
        score_path = match_dir / "score_events.csv"
        if score_path.exists():
            try:
                score_df = pd.read_csv(score_path)
                predicted_goals = score_df["timestamp"].tolist() if len(score_df) > 0 and "timestamp" in score_df.columns else []
            except (pd.errors.EmptyDataError, KeyError):
                predicted_goals = []
        else:
            predicted_goals = []

        # Load highlights
        hl_path = match_dir / "highlights.csv"
        if hl_path.exists():
            hl_df = pd.read_csv(hl_path)
            highlights = list(zip(hl_df["start_time"], hl_df["end_time"]))
        else:
            highlights = []

        # Load duration
        eval_json = match_dir / "evaluation.json"
        if eval_json.exists():
            with open(eval_json) as f:
                prev = json.load(f)
            total_duration = prev.get("total_video_duration", 0)
        else:
            total_duration = 0

        events = categorize_events(
            match_id, match_dir, gt_events, predicted_goals, highlights,
            config.evaluation.tolerance_window, total_duration,
        )

        # Add season info
        for e in events:
            e["season"] = info.get("season", "unknown")
            e["score"] = info.get("score", "?")

        all_events.extend(events)
        logger.info("%s: %d events categorized", match_id, len(events))

    # Save detailed events
    with open(output_dir / "event_categories.json", "w") as f:
        json.dump(all_events, f, indent=2, default=float)

    # Summary statistics
    tp_events = [e for e in all_events if e.get("category") == "TP"]
    fn_events = [e for e in all_events if e.get("category") == "FN"]
    fp_events = [e for e in all_events if e.get("category") == "FP"]

    summary = {
        "total_gt_events": len(tp_events) + len(fn_events),
        "total_tp": len(tp_events),
        "total_fn": len(fn_events),
        "total_fp": len(fp_events),
    }

    # TP analysis
    if tp_events:
        summary["tp_ocr_detected"] = sum(1 for e in tp_events if e.get("ocr_detected"))
        summary["tp_play_covered"] = sum(1 for e in tp_events if e.get("play_covered"))

    # FN reason breakdown
    if fn_events:
        fn_reasons = {}
        for e in fn_events:
            for reason in e.get("fn_reasons", ["unknown"]):
                fn_reasons[reason] = fn_reasons.get(reason, 0) + 1
        summary["fn_reason_breakdown"] = fn_reasons

    # FP reason breakdown
    if fp_events:
        fp_reasons = {}
        for e in fp_events:
            for reason in e.get("fp_reasons", ["unknown"]):
                fp_reasons[reason] = fp_reasons.get(reason, 0) + 1
        summary["fp_reason_breakdown"] = fp_reasons

    # Per-season breakdown
    seasons = {}
    for e in all_events:
        season = e.get("season", "unknown")
        if season not in seasons:
            seasons[season] = {"tp": 0, "fn": 0, "fp": 0}
        cat = e.get("category", "")
        if cat in seasons[season]:
            seasons[season][cat] += 1

    for season, counts in seasons.items():
        tp = counts["tp"]
        fn = counts["fn"]
        fp = counts["fp"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        counts["precision"] = p
        counts["recall"] = r
        counts["f1"] = f1

    summary["per_season"] = seasons

    # First half vs second half
    first_half = [e for e in all_events if e.get("half") == "first" and e.get("event_type") == "goal"]
    second_half = [e for e in all_events if e.get("half") == "second" and e.get("event_type") == "goal"]
    summary["first_half_tp"] = sum(1 for e in first_half if e.get("category") == "TP")
    summary["first_half_total"] = len(first_half)
    summary["second_half_tp"] = sum(1 for e in second_half if e.get("category") == "TP")
    summary["second_half_total"] = len(second_half)

    with open(output_dir / "error_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    # Print summary
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total GT events: {summary['total_gt_events']}")
    print(f"TP: {summary['total_tp']} | FN: {summary['total_fn']} | FP: {summary['total_fp']}")

    if fn_events:
        print(f"\nFN reasons:")
        for reason, count in sorted(summary.get("fn_reason_breakdown", {}).items(),
                                     key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print(f"\nPer-season:")
    for season, counts in sorted(seasons.items()):
        print(f"  {season}: TP={counts['tp']} FN={counts['fn']} FP={counts['fp']} "
              f"F1={counts['f1']:.3f}")

    print(f"\nFirst half: {summary['first_half_tp']}/{summary['first_half_total']} detected")
    print(f"Second half: {summary['second_half_tp']}/{summary['second_half_total']} detected")

    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
