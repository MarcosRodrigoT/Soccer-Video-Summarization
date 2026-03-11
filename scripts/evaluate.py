#!/usr/bin/env python3
"""Run evaluation against ground truth for processed matches.

Reads saved pipeline outputs and computes metrics without re-running the pipeline.

Usage:
    python scripts/evaluate.py [--results-dir results/per_match] [--matches Match1,Match5]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator, EvaluationResult
from soccer_summarization.utils import setup_logging


def evaluate_match(
    match_id: str,
    match_dir: Path,
    gt_path: str,
    evaluator: Evaluator,
    total_duration: float,
) -> EvaluationResult:
    """Evaluate a single match from saved outputs."""
    logger = logging.getLogger(__name__)

    # Load GT
    gt_df = evaluator.load_ground_truth(gt_path)
    gt_events = evaluator.extract_goal_events(gt_df)

    # Load predicted goals from OCR
    score_path = match_dir / "score_events.csv"
    if score_path.exists():
        try:
            score_df = pd.read_csv(score_path)
            predicted_goals = score_df["timestamp"].tolist() if len(score_df) > 0 else []
        except (pd.errors.EmptyDataError, KeyError):
            predicted_goals = []
    else:
        predicted_goals = []
        logger.warning("No score_events.csv for %s", match_id)

    # Load highlights
    hl_path = match_dir / "highlights.csv"
    if hl_path.exists():
        hl_df = pd.read_csv(hl_path)
        highlight_segments = list(zip(hl_df["start_time"], hl_df["end_time"]))
    else:
        highlight_segments = []
        logger.warning("No highlights.csv for %s", match_id)

    return evaluator.evaluate_match(
        match_id=match_id,
        highlight_segments=highlight_segments,
        predicted_goal_times=predicted_goals,
        gt_events=gt_events,
        total_video_duration=total_duration,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate processed matches")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--matches", help="Comma-separated match IDs (default: all available)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    config = PipelineConfig.from_yaml(args.config)
    evaluator = Evaluator(config.evaluation)

    results_dir = Path(args.results_dir)
    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        all_matches = yaml.safe_load(f)["matches"]

    if args.matches:
        match_ids = [m.strip() for m in args.matches.split(",")]
    else:
        # Use all matches that have results
        match_ids = [
            d.name for d in sorted(results_dir.iterdir())
            if d.is_dir() and d.name in all_matches
        ]

    if not match_ids:
        logger.error("No processed matches found in %s", results_dir)
        return

    logger.info("Evaluating %d matches", len(match_ids))

    results = []
    for match_id in match_ids:
        match_dir = results_dir / match_id
        if not match_dir.exists():
            logger.warning("No results for %s, skipping", match_id)
            continue

        info = all_matches[match_id]
        gt_path = str(Path(config.data_dir) / info["gt_path"])

        # Get video duration from evaluation.json or re-compute
        eval_json = match_dir / "evaluation.json"
        if eval_json.exists():
            with open(eval_json) as f:
                prev = json.load(f)
            total_duration = prev.get("total_video_duration", 0)
        else:
            video_path = str(Path(config.data_dir) / info["video_path"])
            from soccer_summarization.utils import get_video_properties
            total_duration = get_video_properties(video_path)["duration"]

        result = evaluate_match(match_id, match_dir, gt_path, evaluator, total_duration)
        results.append(result)

        print(f"{match_id}: P={result.precision:.2f} R={result.recall:.2f} F1={result.f1:.2f} "
              f"| Comp={result.compression_ratio*100:.0f}% Cov={result.goal_coverage*100:.0f}% "
              f"| tIoU={result.mean_tiou:.3f} MAE={result.goal_timing_mae:.1f}s Segs={result.num_segments}")

    # Aggregate
    if results:
        aggregate = Evaluator.aggregate_results(results)

        print("\n" + "=" * 60)
        print(f"AGGREGATE ({len(results)} matches)")
        print("=" * 60)
        print(f"Goal Detection (macro): P={aggregate['precision_mean']:.3f} "
              f"R={aggregate['recall_mean']:.3f} F1={aggregate['f1_mean']:.3f}")
        print(f"Goal Detection (micro): P={aggregate['micro_precision']:.3f} "
              f"R={aggregate['micro_recall']:.3f} F1={aggregate['micro_f1']:.3f}")
        print(f"Compression: {aggregate['compression_mean']*100:.1f}% "
              f"+/- {aggregate['compression_std']*100:.1f}%")
        print(f"Goal Coverage: {aggregate['goal_coverage_mean']*100:.1f}% "
              f"+/- {aggregate['goal_coverage_std']*100:.1f}%")
        print(f"Mean tIoU: {aggregate.get('mean_tiou_mean', 0):.3f} "
              f"+/- {aggregate.get('mean_tiou_std', 0):.3f}")
        print(f"Goal Timing MAE: {aggregate.get('goal_timing_mae_mean', 0):.1f}s "
              f"+/- {aggregate.get('goal_timing_mae_std', 0):.1f}s")
        print(f"Segments: {aggregate.get('num_segments_mean', 0):.1f} "
              f"+/- {aggregate.get('num_segments_std', 0):.1f}")
        if "bootstrap_ci" in aggregate:
            ci = aggregate["bootstrap_ci"]
            print(f"\n95% Bootstrap CIs:")
            for metric in ["f1", "micro_f1", "compression_ratio", "goal_coverage", "mean_tiou"]:
                if metric in ci:
                    print(f"  {metric}: [{ci[metric]['lower']:.3f}, {ci[metric]['upper']:.3f}]")

        # Save
        agg_dir = Path(config.results_dir) / "aggregate"
        agg_dir.mkdir(parents=True, exist_ok=True)
        with open(agg_dir / "aggregate_metrics.json", "w") as f:
            json.dump(aggregate, f, indent=2, default=float)
        print(f"\nSaved to {agg_dir / 'aggregate_metrics.json'}")


if __name__ == "__main__":
    main()
