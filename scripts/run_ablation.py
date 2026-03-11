#!/usr/bin/env python3
"""Run ablation studies by varying one parameter at a time.

Reuses cached upstream stages where possible. Each ablation produces results
in results/ablation/{param_name}_{value}/MatchN/.

Usage:
    python scripts/run_ablation.py --ablation play_threshold [--matches Match1,Match5]
    python scripts/run_ablation.py --ablation all
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

from soccer_summarization.config import PipelineConfig, PlayClassifierConfig, HighlightAssemblerConfig
from soccer_summarization.evaluation import Evaluator, EvaluationResult
from soccer_summarization.play_classifier import PlayClassifier, PlaySegment
from soccer_summarization.highlight_assembler import HighlightAssembler
from soccer_summarization.shot_segmentation import ShotBoundary
from soccer_summarization.utils import setup_logging


# Ablation definitions: parameter -> (values, stages_to_rerun, subset_only)
ABLATIONS = {
    "play_threshold": {
        "values": [3, 4, 5, 6, 7],
        "default": 5,
        "rerun": ["play_classifier", "highlight_assembler", "evaluation"],
        "subset_only": False,
    },
    "smoothing_window": {
        "values": [25, 50, 75, 100],
        "default": 50,
        "rerun": ["play_classifier", "highlight_assembler", "evaluation"],
        "subset_only": False,
    },
    "min_shot_duration": {
        "values": [2.0, 3.0, 4.0, 5.0, 7.0],
        "default": 4.0,
        "rerun": ["highlight_assembler", "evaluation"],
        "subset_only": False,
    },
    "confidence_threshold": {
        "values": [0.3, 0.4, 0.5, 0.6, 0.7],
        "default": 0.5,
        "rerun": ["object_detection", "play_classifier", "highlight_assembler", "evaluation"],
        "subset_only": True,
    },
    "frame_step": {
        "values": [1, 2, 5, 10],
        "default": 5,
        "rerun": ["object_detection", "play_classifier", "highlight_assembler", "evaluation"],
        "subset_only": True,
    },
    "merge_gap_threshold": {
        "values": [0.0, 0.2, 0.5, 1.0, 2.0],
        "default": 0.5,
        "rerun": ["highlight_assembler", "evaluation"],
        "subset_only": False,
    },
    "tolerance_window": {
        "values": [10, 15, 20, 30, 45, 60],
        "default": 30,
        "rerun": ["evaluation"],
        "subset_only": False,
    },
}

SUBSET_MATCHES = ["Match1", "Match5", "Match10", "Match15", "Match22"]


def load_cached_data(match_id: str, results_dir: Path, config: PipelineConfig,
                     matches: dict) -> dict:
    """Load all cached pipeline outputs for a match."""
    match_dir = results_dir / match_id
    info = matches[match_id]
    gt_path = str(Path(config.data_dir) / info["gt_path"])

    data = {"match_id": match_id, "gt_path": gt_path}

    # Load evaluation.json for video duration and fps
    eval_json = match_dir / "evaluation.json"
    if eval_json.exists():
        with open(eval_json) as f:
            prev = json.load(f)
        data["total_duration"] = prev["total_video_duration"]

    # Load shot boundaries
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

    # Load player counts
    player_path = match_dir / "player_counts.csv"
    if player_path.exists():
        data["player_counts"] = pd.read_csv(player_path)

    # Load play segments
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

    # Load score events
    score_path = match_dir / "score_events.csv"
    if score_path.exists():
        try:
            df = pd.read_csv(score_path)
            data["predicted_goal_times"] = df["timestamp"].tolist() if len(df) > 0 and "timestamp" in df.columns else []
        except (pd.errors.EmptyDataError, KeyError):
            data["predicted_goal_times"] = []
    else:
        data["predicted_goal_times"] = []

    # Get fps from shot boundaries or default
    if data.get("shots") and len(data["shots"]) > 0:
        s = data["shots"][0]
        if s.duration > 0:
            data["fps"] = int(round((s.end_frame - s.start_frame) / s.duration))
        else:
            data["fps"] = 25
    else:
        data["fps"] = 25

    return data


def run_ablation_value(ablation_name: str, value, match_data: dict,
                       config: PipelineConfig, evaluator: Evaluator) -> EvaluationResult:
    """Run a single ablation value on a single match."""
    logger = logging.getLogger(__name__)
    stages = ABLATIONS[ablation_name]["rerun"]

    # Create modified config
    play_config = PlayClassifierConfig(
        play_threshold=config.play_classifier.play_threshold,
        smoothing_window=config.play_classifier.smoothing_window,
        min_shot_duration=config.play_classifier.min_shot_duration,
    )
    hl_config = HighlightAssemblerConfig(
        min_shot_duration=config.highlight_assembler.min_shot_duration,
        merge_gap_threshold=config.highlight_assembler.merge_gap_threshold,
        goal_context_before=config.highlight_assembler.goal_context_before,
        goal_context_after=config.highlight_assembler.goal_context_after,
    )

    # Apply the ablation parameter
    if ablation_name == "play_threshold":
        play_config.play_threshold = value
    elif ablation_name == "smoothing_window":
        play_config.smoothing_window = value
    elif ablation_name == "min_shot_duration":
        hl_config.min_shot_duration = value
    elif ablation_name == "merge_gap_threshold":
        hl_config.merge_gap_threshold = value
    elif ablation_name == "tolerance_window":
        eval_config = type(evaluator.config)(
            tolerance_window=value,
            goal_gap_threshold=evaluator.config.goal_gap_threshold,
        )
        evaluator = Evaluator(eval_config)

    # Re-run necessary stages
    play_segments = match_data.get("play_segments", [])
    shots = match_data.get("shots", [])
    fps = match_data.get("fps", 25)

    if "play_classifier" in stages and "player_counts" in match_data:
        classifier = PlayClassifier(play_config)
        play_segments, _ = classifier.classify(match_data["player_counts"], fps)

    if "highlight_assembler" in stages:
        assembler = HighlightAssembler(hl_config)
        highlights = assembler.assemble(shots, play_segments, fps)
        highlight_segments = [(h.start_time, h.end_time) for h in highlights]
    else:
        # Use cached highlights from the default pipeline run
        results_dir = Path("results/per_match") / match_data["match_id"]
        hl_path = results_dir / "highlights.csv"
        if hl_path.exists():
            hl_df = pd.read_csv(hl_path)
            highlight_segments = list(zip(hl_df["start_time"], hl_df["end_time"]))
        else:
            highlight_segments = []

    # Evaluation
    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    result = evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=highlight_segments,
        predicted_goal_times=match_data.get("predicted_goal_times", []),
        gt_events=gt_events,
        total_video_duration=match_data.get("total_duration", 0),
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--ablation", required=True,
                        help="Ablation name or 'all'")
    parser.add_argument("--matches", help="Comma-separated match IDs (override subset)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = PipelineConfig.from_yaml(args.config)

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        all_matches = yaml.safe_load(f)["matches"]

    if args.ablation == "all":
        ablation_names = list(ABLATIONS.keys())
    else:
        ablation_names = [a.strip() for a in args.ablation.split(",")]

    for abl_name in ablation_names:
        if abl_name not in ABLATIONS:
            logger.warning("Unknown ablation: %s", abl_name)
            continue

        abl_info = ABLATIONS[abl_name]
        logger.info("=" * 60)
        logger.info("Ablation: %s (values: %s, default: %s)",
                     abl_name, abl_info["values"], abl_info["default"])
        logger.info("=" * 60)

        # Determine match set
        if args.matches:
            match_ids = [m.strip() for m in args.matches.split(",")]
        elif abl_info["subset_only"]:
            match_ids = [m for m in SUBSET_MATCHES if (results_dir / m).exists()]
            logger.info("Using 5-match subset: %s", match_ids)
        else:
            match_ids = [
                d.name for d in sorted(results_dir.iterdir())
                if d.is_dir() and d.name in all_matches
            ]

        ablation_results = {}

        for value in abl_info["values"]:
            value_key = f"{abl_name}_{value}"
            logger.info("--- %s = %s ---", abl_name, value)

            evaluator = Evaluator(config.evaluation)
            value_results = []

            for match_id in match_ids:
                if not (results_dir / match_id).exists():
                    logger.warning("No cached data for %s, skipping", match_id)
                    continue

                match_data = load_cached_data(match_id, results_dir, config, all_matches)

                result = run_ablation_value(abl_name, value, match_data, config, evaluator)
                value_results.append(result)

                logger.info(
                    "  %s: P=%.2f R=%.2f F1=%.2f Comp=%.1f%%",
                    match_id, result.precision, result.recall, result.f1,
                    result.compression_ratio * 100,
                )

            if value_results:
                aggregate = Evaluator.aggregate_results(value_results)
                ablation_results[str(value)] = {
                    "value": value,
                    "is_default": value == abl_info["default"],
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
                        for r in value_results
                    ],
                }

                logger.info(
                    "  Aggregate: P=%.3f R=%.3f F1=%.3f Comp=%.1f%%",
                    aggregate["precision_mean"], aggregate["recall_mean"],
                    aggregate["f1_mean"], aggregate["compression_mean"] * 100,
                )

        # Save ablation results
        abl_output = output_dir / f"{abl_name}_results.json"
        with open(abl_output, "w") as f:
            json.dump(ablation_results, f, indent=2, default=float)
        logger.info("Saved to %s", abl_output)

        # Print summary
        print(f"\nAblation: {abl_name}")
        print(f"{'Value':<10} {'P':>6} {'R':>6} {'F1':>6} {'Comp%':>7} {'Cov%':>6} {'tIoU':>6}")
        print("-" * 50)
        for value in abl_info["values"]:
            key = str(value)
            if key in ablation_results:
                a = ablation_results[key]["aggregate"]
                default_marker = " *" if value == abl_info["default"] else ""
                print(
                    f"{str(value):<10} "
                    f"{a['precision_mean']:>6.3f} {a['recall_mean']:>6.3f} {a['f1_mean']:>6.3f} "
                    f"{a['compression_mean']*100:>6.1f} {a['goal_coverage_mean']*100:>5.1f} "
                    f"{a.get('mean_tiou_mean', 0):>6.3f}{default_marker}"
                )


if __name__ == "__main__":
    main()
