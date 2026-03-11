#!/usr/bin/env python3
"""Process a single match end-to-end through the full pipeline.

Usage:
    python scripts/run_single_match.py --video PATH --gt PATH --output-dir DIR [--config PATH]
    python scripts/run_single_match.py --match-id Match1 [--config PATH]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator
from soccer_summarization.highlight_assembler import HighlightAssembler
from soccer_summarization.object_detection import PlayerDetector
from soccer_summarization.ocr_scoreboard import ScoreboardDetector
from soccer_summarization.play_classifier import PlayClassifier
from soccer_summarization.shot_segmentation import ShotSegmenter
from soccer_summarization.utils import get_video_properties, setup_logging


def load_match_info(match_id: str, config: PipelineConfig) -> dict:
    """Load match video/GT paths from matches.yaml."""
    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        data = yaml.safe_load(f)

    if match_id not in data["matches"]:
        raise ValueError(f"Unknown match: {match_id}. Available: {list(data['matches'].keys())}")

    info = data["matches"][match_id]
    info["video_path"] = str(Path(config.data_dir) / info["video_path"])
    info["gt_path"] = str(Path(config.data_dir) / info["gt_path"])
    return info


def run_pipeline(
    video_path: str,
    gt_path: str,
    output_dir: str,
    config: PipelineConfig,
    match_id: str = "unknown",
    skip_ocr: bool = False,
    skip_video_export: bool = False,
):
    """Run the full pipeline on a single match."""
    logger = logging.getLogger(__name__)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    props = get_video_properties(video_path)
    total_duration = props["duration"]
    logger.info(
        "Match %s: %s (%.1f min, %d fps, %dx%d)",
        match_id, Path(video_path).name, total_duration / 60,
        props["fps"], props["width"], props["height"],
    )

    timings = {}

    # --- Stage 1: Shot segmentation ---
    shot_cache = out / "shot_boundaries.csv"
    if shot_cache.exists():
        logger.info("Loading cached shot boundaries")
        shot_df = pd.read_csv(shot_cache)
        from soccer_summarization.shot_segmentation import ShotBoundary
        shots = [
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
    else:
        logger.info("Stage 1: Shot segmentation...")
        t0 = time.time()
        segmenter = ShotSegmenter(config.shot_segmentation)
        shots, _diffs = segmenter.run(video_path)
        timings["shot_segmentation"] = time.time() - t0

        shot_df = pd.DataFrame([
            {"shot_id": s.shot_id, "start_frame": s.start_frame, "end_frame": s.end_frame,
             "start_time": s.start_time, "end_time": s.end_time, "duration": s.duration}
            for s in shots
        ])
        shot_df.to_csv(shot_cache, index=False)
        logger.info("  %d shots detected (saved to %s)", len(shots), shot_cache)

    # --- Stage 2: Object detection (YOLO) ---
    player_cache = out / "player_counts.csv"
    if player_cache.exists():
        logger.info("Loading cached player counts")
        player_df = pd.read_csv(player_cache)
    else:
        logger.info("Stage 2: YOLO object detection...")
        t0 = time.time()
        detector = PlayerDetector(config.object_detection)
        player_df = detector.run(video_path)
        timings["object_detection"] = time.time() - t0

        player_df.to_csv(player_cache, index=False)
        logger.info("  %d frames processed (saved to %s)", len(player_df), player_cache)

    # --- Stage 3: Play classification ---
    play_cache = out / "play_segments.csv"
    frame_cache = out / "frame_classification.csv"
    if play_cache.exists():
        logger.info("Loading cached play segments")
        play_df = pd.read_csv(play_cache)
        from soccer_summarization.play_classifier import PlaySegment
        play_segments = [
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
    else:
        logger.info("Stage 3: Play/no-play classification...")
        t0 = time.time()
        classifier = PlayClassifier(config.play_classifier)
        play_segments, frame_df = classifier.classify(player_df, props["fps"])
        timings["play_classifier"] = time.time() - t0

        play_seg_df = pd.DataFrame([
            {"start_frame": s.start_frame, "end_frame": s.end_frame,
             "start_time": s.start_time, "end_time": s.end_time,
             "duration": s.duration, "is_play": s.is_play,
             "avg_player_count": s.avg_player_count}
            for s in play_segments
        ])
        play_seg_df.to_csv(play_cache, index=False)
        frame_df.to_csv(frame_cache, index=False)
        logger.info("  %d segments (saved to %s)", len(play_segments), play_cache)

    # --- Stage 4: OCR scoreboard ---
    score_cache = out / "score_events.csv"
    score_events = []
    if skip_ocr:
        logger.info("Stage 4: OCR skipped")
    elif score_cache.exists():
        logger.info("Loading cached score events")
        try:
            score_df = pd.read_csv(score_cache)
            from soccer_summarization.ocr_scoreboard import ScoreEvent
            score_events = [
                ScoreEvent(**row) for _, row in score_df.iterrows()
            ]
        except (pd.errors.EmptyDataError, KeyError):
            logger.info("  Empty or invalid score_events.csv, treating as no detections")
            score_events = []
    else:
        logger.info("Stage 4: OCR scoreboard detection...")
        t0 = time.time()
        ocr_detector = ScoreboardDetector(config.ocr_scoreboard)
        region, score_events = ocr_detector.run(video_path)
        timings["ocr_scoreboard"] = time.time() - t0

        score_df = ScoreboardDetector.events_to_dataframe(score_events)
        score_df.to_csv(score_cache, index=False)
        if region:
            region_info = {"x": region.x, "y": region.y, "w": region.w, "h": region.h}
            with open(out / "scoreboard_region.json", "w") as f:
                json.dump(region_info, f)
        logger.info("  %d score events (saved to %s)", len(score_events), score_cache)

    # --- Stage 5: Highlight assembly ---
    highlight_cache = out / "highlights.csv"
    logger.info("Stage 5: Highlight assembly...")
    t0 = time.time()
    assembler = HighlightAssembler(config.highlight_assembler)

    highlights = assembler.assemble(
        shots=shots,
        play_segments=play_segments,
        fps=props["fps"],
    )
    timings["highlight_assembly"] = time.time() - t0

    hl_df = assembler.highlights_to_dataframe(highlights)
    hl_df.to_csv(highlight_cache, index=False)

    total_highlight = sum(h.duration for h in highlights)
    logger.info(
        "  %d highlight segments, %.1fs total (%.1f%% compression)",
        len(highlights), total_highlight,
        (1 - total_highlight / total_duration) * 100 if total_duration > 0 else 0,
    )

    # --- Stage 6: Export highlight video ---
    if not skip_video_export:
        highlight_video = out / "highlight_video.mp4"
        if not highlight_video.exists():
            logger.info("Stage 6: Exporting highlight video...")
            t0 = time.time()
            assembler.export_video(highlights, video_path, str(highlight_video))
            timings["video_export"] = time.time() - t0
            logger.info("  Saved to %s", highlight_video)
        else:
            logger.info("Stage 6: Highlight video already exists")

    # --- Stage 7: Evaluation ---
    logger.info("Stage 7: Evaluation against ground truth...")
    evaluator = Evaluator(config.evaluation)
    gt_df = evaluator.load_ground_truth(gt_path)
    gt_events = evaluator.extract_goal_events(gt_df)

    predicted_goal_times = [e.timestamp for e in score_events]
    highlight_segments = [(h.start_time, h.end_time) for h in highlights]

    result = evaluator.evaluate_match(
        match_id=match_id,
        highlight_segments=highlight_segments,
        predicted_goal_times=predicted_goal_times,
        gt_events=gt_events,
        total_video_duration=total_duration,
    )

    # Save evaluation results
    eval_results = {
        "match_id": match_id,
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "total_video_duration": result.total_video_duration,
        "highlight_duration": result.highlight_duration,
        "compression_ratio": result.compression_ratio,
        "goal_coverage": result.goal_coverage,
        "num_gt_events": len(gt_events),
        "num_predicted_goals": len(predicted_goal_times),
        "num_highlights": len(highlights),
        # Extended metrics
        "num_segments": result.num_segments,
        "mean_segment_duration": result.mean_segment_duration,
        "mean_tiou": result.mean_tiou,
        "goal_timing_mae": result.goal_timing_mae,
        "tiou_per_event": result.tiou_per_event,
        "f1_at_tiou": result.f1_at_tiou,
        "timings": timings,
    }
    def _json_default(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return float(obj)

    with open(out / "evaluation.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=_json_default)

    logger.info("Pipeline complete for %s", match_id)
    return result


def main():
    parser = argparse.ArgumentParser(description="Process a single match")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--match-id", help="Match ID from matches.yaml (e.g., Match1)")
    group.add_argument("--video", help="Path to video file")
    parser.add_argument("--gt", help="Path to ground truth CSV (required with --video)")
    parser.add_argument("--output-dir", help="Output directory (default: results/per_match/<match_id>)")
    parser.add_argument("--config", default="config/default.yaml", help="Pipeline config YAML")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR scoreboard detection")
    parser.add_argument("--skip-video-export", action="store_true", help="Skip highlight video export")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = PipelineConfig.from_yaml(args.config)

    if args.match_id:
        info = load_match_info(args.match_id, config)
        video_path = info["video_path"]
        gt_path = info["gt_path"]
        match_id = args.match_id
        output_dir = args.output_dir or f"{config.results_dir}/per_match/{match_id}"
    else:
        if not args.gt:
            parser.error("--gt is required when using --video")
        video_path = args.video
        gt_path = args.gt
        match_id = Path(video_path).stem[:20]
        output_dir = args.output_dir or f"{config.results_dir}/per_match/{match_id}"

    run_pipeline(
        video_path=video_path,
        gt_path=gt_path,
        output_dir=output_dir,
        config=config,
        match_id=match_id,
        skip_ocr=args.skip_ocr,
        skip_video_export=args.skip_video_export,
    )


if __name__ == "__main__":
    main()
