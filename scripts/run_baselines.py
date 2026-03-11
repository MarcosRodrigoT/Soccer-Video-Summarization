#!/usr/bin/env python3
"""Run internal baselines for comparison against the full system.

Baselines:
    1. Random Sampling (lower bound)
    2. Uniform Sampling
    3. Shot-Length Only (no YOLO/play classification)
    4. Play-Only (no shot structure)
    5. OCR-Only (score change detection only)
    6. Oracle Upper Bound (GT goal annotations)

Each baseline produces highlight segments evaluated with the same metrics.

Usage:
    python scripts/run_baselines.py [--matches Match1,Match5] [--baselines random,oracle]
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
from soccer_summarization.utils import setup_logging


def load_match_data(match_id: str, results_dir: Path, config: PipelineConfig, matches: dict) -> dict:
    """Load cached pipeline outputs for a match."""
    match_dir = results_dir / match_id
    info = matches[match_id]
    gt_path = str(Path(config.data_dir) / info["gt_path"])

    # Load evaluation.json for video duration
    eval_json = match_dir / "evaluation.json"
    if eval_json.exists():
        with open(eval_json) as f:
            prev = json.load(f)
        total_duration = prev["total_video_duration"]
    else:
        from soccer_summarization.utils import get_video_properties
        video_path = str(Path(config.data_dir) / info["video_path"])
        total_duration = get_video_properties(video_path)["duration"]

    data = {"match_id": match_id, "total_duration": total_duration, "gt_path": gt_path}

    # Load shot boundaries
    shot_path = match_dir / "shot_boundaries.csv"
    if shot_path.exists():
        data["shots"] = pd.read_csv(shot_path)

    # Load play segments
    play_path = match_dir / "play_segments.csv"
    if play_path.exists():
        data["play_segments"] = pd.read_csv(play_path)

    # Load score events
    score_path = match_dir / "score_events.csv"
    if score_path.exists():
        try:
            df = pd.read_csv(score_path)
            data["score_events"] = df if len(df) > 0 else pd.DataFrame()
        except (pd.errors.EmptyDataError, KeyError):
            data["score_events"] = pd.DataFrame()
    else:
        data["score_events"] = pd.DataFrame()

    # Load highlights (for matching compression ratio)
    hl_path = match_dir / "highlights.csv"
    if hl_path.exists():
        data["system_highlights"] = pd.read_csv(hl_path)

    return data


def baseline_random(match_data: dict, evaluator: Evaluator, n_seeds: int = 10) -> list[EvaluationResult]:
    """Baseline 1: Random segments totaling same duration as system output."""
    total_dur = match_data["total_duration"]
    if "system_highlights" not in match_data:
        return []

    system_hl = match_data["system_highlights"]
    target_duration = system_hl["duration"].sum()
    segment_dur = 30.0  # fixed segment duration

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    results = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        n_segments = max(1, int(target_duration / segment_dur))
        starts = rng.uniform(0, max(0, total_dur - segment_dur), size=n_segments)
        starts.sort()
        segments = [(s, s + segment_dur) for s in starts]

        result = evaluator.evaluate_match(
            match_id=match_data["match_id"],
            highlight_segments=segments,
            predicted_goal_times=[],
            gt_events=gt_events,
            total_video_duration=total_dur,
        )
        results.append(result)

    # Average over seeds
    avg = _average_results(results, match_data["match_id"])
    return [avg]


def baseline_uniform(match_data: dict, evaluator: Evaluator) -> list[EvaluationResult]:
    """Baseline 2: K evenly-spaced segments."""
    total_dur = match_data["total_duration"]
    if "system_highlights" not in match_data:
        return []

    system_hl = match_data["system_highlights"]
    target_duration = system_hl["duration"].sum()
    segment_dur = 30.0

    n_segments = max(1, int(target_duration / segment_dur))
    spacing = total_dur / (n_segments + 1)
    segments = []
    for i in range(1, n_segments + 1):
        center = i * spacing
        start = max(0, center - segment_dur / 2)
        end = min(total_dur, start + segment_dur)
        segments.append((start, end))

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    result = evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=segments,
        predicted_goal_times=[],
        gt_events=gt_events,
        total_video_duration=total_dur,
    )
    return [result]


def baseline_shot_only(match_data: dict, evaluator: Evaluator) -> list[EvaluationResult]:
    """Baseline 3: Keep all shots > T seconds, no play classification."""
    if "shots" not in match_data:
        return []

    shots_df = match_data["shots"]
    total_dur = match_data["total_duration"]

    # Target compression: match the system's output duration
    if "system_highlights" in match_data:
        target_duration = match_data["system_highlights"]["duration"].sum()
    else:
        target_duration = total_dur * 0.3

    # Find threshold T that gives closest compression ratio
    best_T = 4.0
    best_diff = float("inf")
    for T in np.arange(1.0, 60.0, 0.5):
        long = shots_df[shots_df["duration"] >= T]
        dur = long["duration"].sum()
        diff = abs(dur - target_duration)
        if diff < best_diff:
            best_diff = diff
            best_T = T

    long_shots = shots_df[shots_df["duration"] >= best_T]
    segments = list(zip(long_shots["start_time"], long_shots["end_time"]))

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    result = evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=segments,
        predicted_goal_times=[],
        gt_events=gt_events,
        total_video_duration=total_dur,
    )
    return [result]


def baseline_play_only(match_data: dict, evaluator: Evaluator) -> list[EvaluationResult]:
    """Baseline 4: Use play segments directly as highlights."""
    if "play_segments" not in match_data:
        return []

    play_df = match_data["play_segments"]
    play_segs = play_df[play_df["is_play"] == True]
    segments = list(zip(play_segs["start_time"], play_segs["end_time"]))

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    result = evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=segments,
        predicted_goal_times=[],
        gt_events=gt_events,
        total_video_duration=match_data["total_duration"],
    )
    return [result]


def baseline_ocr_only(match_data: dict, evaluator: Evaluator, context: float = 30.0) -> list[EvaluationResult]:
    """Baseline 5: ±context seconds around each OCR-detected goal."""
    total_dur = match_data["total_duration"]
    score_df = match_data.get("score_events", pd.DataFrame())

    if score_df.empty or "timestamp" not in score_df.columns:
        # No OCR detections — evaluate with empty highlights
        gt_df = evaluator.load_ground_truth(match_data["gt_path"])
        gt_events = evaluator.extract_goal_events(gt_df)
        result = evaluator.evaluate_match(
            match_id=match_data["match_id"],
            highlight_segments=[],
            predicted_goal_times=[],
            gt_events=gt_events,
            total_video_duration=total_dur,
        )
        return [result]

    predicted_goals = score_df["timestamp"].tolist()
    segments = []
    for t in predicted_goals:
        start = max(0, t - context)
        end = min(total_dur, t + context)
        segments.append((start, end))

    # Merge overlapping segments
    segments.sort()
    merged = [segments[0]]
    for s, e in segments[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    result = evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=merged,
        predicted_goal_times=predicted_goals,
        gt_events=gt_events,
        total_video_duration=total_dur,
    )
    return [result]


def baseline_oracle(match_data: dict, evaluator: Evaluator, context: float = 30.0) -> list[EvaluationResult]:
    """Baseline 6: Oracle — ±context seconds around each GT goal."""
    total_dur = match_data["total_duration"]

    gt_df = evaluator.load_ground_truth(match_data["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)

    segments = []
    for event in gt_events:
        start = max(0, event.start_time - context)
        end = min(total_dur, event.end_time + context)
        segments.append((start, end))

    # Merge overlapping
    if segments:
        segments.sort()
        merged = [segments[0]]
        for s, e in segments[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
    else:
        merged = []

    # Oracle "predicts" at GT midpoints
    predicted_goals = [e.midpoint for e in gt_events]

    result = evaluator.evaluate_match(
        match_id=match_data["match_id"],
        highlight_segments=merged,
        predicted_goal_times=predicted_goals,
        gt_events=gt_events,
        total_video_duration=total_dur,
    )
    return [result]


def _average_results(results: list[EvaluationResult], match_id: str) -> EvaluationResult:
    """Average multiple EvaluationResults (e.g., over random seeds)."""
    return EvaluationResult(
        match_id=match_id,
        true_positives=int(np.mean([r.true_positives for r in results])),
        false_positives=int(np.mean([r.false_positives for r in results])),
        false_negatives=int(np.mean([r.false_negatives for r in results])),
        precision=float(np.mean([r.precision for r in results])),
        recall=float(np.mean([r.recall for r in results])),
        f1=float(np.mean([r.f1 for r in results])),
        total_video_duration=results[0].total_video_duration,
        highlight_duration=float(np.mean([r.highlight_duration for r in results])),
        compression_ratio=float(np.mean([r.compression_ratio for r in results])),
        goal_coverage=float(np.mean([r.goal_coverage for r in results])),
        num_segments=int(np.mean([r.num_segments for r in results])),
        mean_segment_duration=float(np.mean([r.mean_segment_duration for r in results])),
        mean_tiou=float(np.mean([r.mean_tiou for r in results])),
        goal_timing_mae=float(np.mean([r.goal_timing_mae for r in results])),
        tiou_per_event=[],
        f1_at_tiou={
            k: float(np.mean([r.f1_at_tiou.get(k, 0) for r in results]))
            for k in ["0.1", "0.3", "0.5", "0.7"]
        },
    )


BASELINES = {
    "random": ("Random Sampling", baseline_random),
    "uniform": ("Uniform Sampling", baseline_uniform),
    "shot_only": ("Shot-Length Only", baseline_shot_only),
    "play_only": ("Play-Only", baseline_play_only),
    "ocr_only": ("OCR-Only", baseline_ocr_only),
    "oracle": ("Oracle Upper Bound", baseline_oracle),
}


def main():
    parser = argparse.ArgumentParser(description="Run internal baselines")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--matches", help="Comma-separated match IDs (default: all available)")
    parser.add_argument("--baselines", help="Comma-separated baseline names (default: all)",
                        default=",".join(BASELINES.keys()))
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

    baseline_names = [b.strip() for b in args.baselines.split(",")]

    logger.info("Running %d baselines on %d matches", len(baseline_names), len(match_ids))

    all_baseline_results = {}

    for bl_name in baseline_names:
        if bl_name not in BASELINES:
            logger.warning("Unknown baseline: %s", bl_name)
            continue

        display_name, bl_func = BASELINES[bl_name]
        logger.info("=" * 60)
        logger.info("Baseline: %s", display_name)
        logger.info("=" * 60)

        bl_results = []
        for match_id in match_ids:
            match_dir = results_dir / match_id
            if not match_dir.exists():
                logger.warning("No results for %s, skipping", match_id)
                continue

            match_data = load_match_data(match_id, results_dir, config, all_matches)
            match_results = bl_func(match_data, evaluator)
            bl_results.extend(match_results)

            if match_results:
                r = match_results[0]
                logger.info(
                    "  %s: P=%.2f R=%.2f F1=%.2f Comp=%.1f%%",
                    match_id, r.precision, r.recall, r.f1,
                    r.compression_ratio * 100,
                )

        if bl_results:
            aggregate = Evaluator.aggregate_results(bl_results)
            all_baseline_results[bl_name] = {
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
                        "goal_timing_mae": r.goal_timing_mae,
                        "num_segments": r.num_segments,
                    }
                    for r in bl_results
                ],
            }

            logger.info(
                "%s aggregate: P=%.3f R=%.3f F1=%.3f Comp=%.1f%%",
                display_name,
                aggregate["precision_mean"], aggregate["recall_mean"],
                aggregate["f1_mean"], aggregate["compression_mean"] * 100,
            )

    # Save all results
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump(all_baseline_results, f, indent=2, default=float)
    logger.info("Saved baseline results to %s", output_dir / "baseline_results.json")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Baseline':<25} {'P':>6} {'R':>6} {'F1':>6} {'Comp%':>7} {'Cov%':>6} {'tIoU':>6}")
    print("=" * 80)
    for bl_name in baseline_names:
        if bl_name in all_baseline_results:
            a = all_baseline_results[bl_name]["aggregate"]
            print(
                f"{all_baseline_results[bl_name]['display_name']:<25} "
                f"{a['precision_mean']:>6.3f} {a['recall_mean']:>6.3f} {a['f1_mean']:>6.3f} "
                f"{a['compression_mean']*100:>6.1f} {a['goal_coverage_mean']*100:>5.1f} "
                f"{a.get('mean_tiou_mean', 0):>6.3f}"
            )


if __name__ == "__main__":
    main()
