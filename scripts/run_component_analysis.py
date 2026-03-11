#!/usr/bin/env python3
"""Component contribution analysis: systematically remove pipeline components.

Configurations:
    1. Full system (default)
    2. No OCR (skip OCR score detection)
    3. No play filter (play_threshold=0, include all segments)
    4. No shot filter (min_shot_duration=0, keep all shots)
    5. Play-only (treat video as one shot, no shot segmentation)

Usage:
    python scripts/run_component_analysis.py [--matches Match1,Match5]
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
from soccer_summarization.evaluation import Evaluator, EvaluationResult
from soccer_summarization.play_classifier import PlayClassifier, PlaySegment
from soccer_summarization.highlight_assembler import HighlightAssembler, HighlightSegment
from soccer_summarization.shot_segmentation import ShotBoundary
from soccer_summarization.utils import setup_logging


def load_cached_data(match_id, results_dir, config, matches):
    """Load all cached data for a match."""
    match_dir = results_dir / match_id
    info = matches[match_id]
    gt_path = str(Path(config.data_dir) / info["gt_path"])

    data = {"match_id": match_id, "gt_path": gt_path}

    eval_json = match_dir / "evaluation.json"
    if eval_json.exists():
        with open(eval_json) as f:
            prev = json.load(f)
        data["total_duration"] = prev["total_video_duration"]

    shot_path = match_dir / "shot_boundaries.csv"
    if shot_path.exists():
        shot_df = pd.read_csv(shot_path)
        data["shots"] = [
            ShotBoundary(
                shot_id=int(row["shot_id"]),
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                start_time=row["start_time"],
                end_time=row["end_time"],
                duration=row["duration"],
            )
            for _, row in shot_df.iterrows()
        ]

    player_path = match_dir / "player_counts.csv"
    if player_path.exists():
        data["player_counts"] = pd.read_csv(player_path)

    play_path = match_dir / "play_segments.csv"
    if play_path.exists():
        play_df = pd.read_csv(play_path)
        data["play_segments"] = [
            PlaySegment(
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                start_time=row["start_time"],
                end_time=row["end_time"],
                duration=row["duration"],
                is_play=bool(row["is_play"]),
                avg_player_count=row["avg_player_count"],
            )
            for _, row in play_df.iterrows()
        ]

    score_path = match_dir / "score_events.csv"
    if score_path.exists():
        try:
            df = pd.read_csv(score_path)
            data["predicted_goal_times"] = df["timestamp"].tolist() if len(df) > 0 and "timestamp" in df.columns else []
        except (pd.errors.EmptyDataError, KeyError):
            data["predicted_goal_times"] = []
    else:
        data["predicted_goal_times"] = []

    # Infer fps
    if data.get("shots") and len(data["shots"]) > 0:
        s = data["shots"][0]
        if s.duration > 0:
            data["fps"] = int(round((s.end_frame - s.start_frame) / s.duration))
        else:
            data["fps"] = 25
    else:
        data["fps"] = 25

    return data


def run_full_system(match_data, config, evaluator):
    """Config 1: Full system (default)."""
    assembler = HighlightAssembler(config.highlight_assembler)
    highlights = assembler.assemble(
        match_data["shots"], match_data["play_segments"], match_data["fps"]
    )
    highlight_segments = [(h.start_time, h.end_time) for h in highlights]

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    return evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=highlight_segments,
        predicted_goal_times=match_data.get("predicted_goal_times", []),
        gt_events=gt_events,
        total_video_duration=match_data["total_duration"],
    )


def run_no_ocr(match_data, config, evaluator):
    """Config 2: No OCR — same highlights, but no predicted goal times."""
    assembler = HighlightAssembler(config.highlight_assembler)
    highlights = assembler.assemble(
        match_data["shots"], match_data["play_segments"], match_data["fps"]
    )
    highlight_segments = [(h.start_time, h.end_time) for h in highlights]

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    return evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=highlight_segments,
        predicted_goal_times=[],  # No OCR predictions
        gt_events=gt_events,
        total_video_duration=match_data["total_duration"],
    )


def run_no_play_filter(match_data, config, evaluator):
    """Config 3: No play filter — set play_threshold=0 (all frames are play)."""
    from soccer_summarization.config import PlayClassifierConfig
    play_config = PlayClassifierConfig(
        play_threshold=0,
        smoothing_window=config.play_classifier.smoothing_window,
        min_shot_duration=config.play_classifier.min_shot_duration,
    )
    classifier = PlayClassifier(play_config)
    play_segments, _ = classifier.classify(match_data["player_counts"], match_data["fps"])

    assembler = HighlightAssembler(config.highlight_assembler)
    highlights = assembler.assemble(match_data["shots"], play_segments, match_data["fps"])
    highlight_segments = [(h.start_time, h.end_time) for h in highlights]

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    return evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=highlight_segments,
        predicted_goal_times=match_data.get("predicted_goal_times", []),
        gt_events=gt_events,
        total_video_duration=match_data["total_duration"],
    )


def run_no_shot_filter(match_data, config, evaluator):
    """Config 4: No shot filter — min_shot_duration=0."""
    from soccer_summarization.config import HighlightAssemblerConfig
    hl_config = HighlightAssemblerConfig(
        min_shot_duration=0.0,
        merge_gap_threshold=config.highlight_assembler.merge_gap_threshold,
        goal_context_before=config.highlight_assembler.goal_context_before,
        goal_context_after=config.highlight_assembler.goal_context_after,
    )
    assembler = HighlightAssembler(hl_config)
    highlights = assembler.assemble(
        match_data["shots"], match_data["play_segments"], match_data["fps"]
    )
    highlight_segments = [(h.start_time, h.end_time) for h in highlights]

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    return evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=highlight_segments,
        predicted_goal_times=match_data.get("predicted_goal_times", []),
        gt_events=gt_events,
        total_video_duration=match_data["total_duration"],
    )


def run_play_only(match_data, config, evaluator):
    """Config 5: Play-only — treat video as one shot, no shot segmentation."""
    fps = match_data["fps"]
    total_dur = match_data["total_duration"]
    total_frames = int(total_dur * fps)

    # Create a single shot spanning the entire video
    single_shot = [ShotBoundary(
        shot_id=1,
        start_frame=0,
        end_frame=total_frames,
        start_time=0.0,
        end_time=total_dur,
        duration=total_dur,
    )]

    assembler = HighlightAssembler(config.highlight_assembler)
    highlights = assembler.assemble(single_shot, match_data["play_segments"], fps)
    highlight_segments = [(h.start_time, h.end_time) for h in highlights]

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    return evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=highlight_segments,
        predicted_goal_times=[],  # No OCR in play-only mode
        gt_events=gt_events,
        total_video_duration=total_dur,
    )


CONFIGS = {
    "full_system": ("Full System", run_full_system),
    "no_ocr": ("No OCR", run_no_ocr),
    "no_play_filter": ("No Play Filter", run_no_play_filter),
    "no_shot_filter": ("No Shot Filter", run_no_shot_filter),
    "play_only": ("Play-Only", run_play_only),
}


def main():
    parser = argparse.ArgumentParser(description="Component contribution analysis")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--output-dir", default="results/component_analysis")
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

    logger.info("Running component analysis on %d matches", len(match_ids))

    all_results = {}

    for cfg_name, (display_name, cfg_func) in CONFIGS.items():
        logger.info("=" * 60)
        logger.info("Config: %s", display_name)
        logger.info("=" * 60)

        cfg_results = []
        for match_id in match_ids:
            if not (results_dir / match_id).exists():
                continue

            match_data = load_cached_data(match_id, results_dir, config, all_matches)

            # Check required data
            if "shots" not in match_data or "play_segments" not in match_data:
                logger.warning("Missing data for %s, skipping", match_id)
                continue

            result = cfg_func(match_data, config, evaluator)
            cfg_results.append(result)

        if cfg_results:
            aggregate = Evaluator.aggregate_results(cfg_results)
            all_results[cfg_name] = {
                "display_name": display_name,
                "aggregate": aggregate,
                "per_match": [
                    {
                        "match_id": r.match_id,
                        "precision": r.precision,
                        "recall": r.recall,
                        "f1": r.f1,
                        "compression_ratio": r.compression_ratio,
                        "goal_coverage": r.goal_coverage,
                        "mean_tiou": r.mean_tiou,
                        "num_segments": r.num_segments,
                    }
                    for r in cfg_results
                ],
            }

    # Save
    with open(output_dir / "component_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Print summary
    print("\n" + "=" * 80)
    print("COMPONENT CONTRIBUTION ANALYSIS")
    print("=" * 80)
    print(f"{'Config':<25} {'P':>6} {'R':>6} {'F1':>6} {'Comp%':>7} {'Cov%':>6} {'tIoU':>6} {'Segs':>6}")
    print("-" * 80)
    for cfg_name in CONFIGS:
        if cfg_name in all_results:
            a = all_results[cfg_name]["aggregate"]
            print(
                f"{all_results[cfg_name]['display_name']:<25} "
                f"{a['precision_mean']:>6.3f} {a['recall_mean']:>6.3f} {a['f1_mean']:>6.3f} "
                f"{a['compression_mean']*100:>6.1f} {a['goal_coverage_mean']*100:>5.1f} "
                f"{a.get('mean_tiou_mean', 0):>6.3f} {a.get('num_segments_mean', 0):>5.0f}"
            )

    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
