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
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator, GoalClip
from soccer_summarization.highlight_assembler import HighlightAssembler
from soccer_summarization.object_detection import PlayerDetector
from soccer_summarization.ocr_scoreboard import ScoreboardDetector
from soccer_summarization.play_classifier import PlayClassifier
from soccer_summarization.shot_segmentation import ShotSegmenter
from soccer_summarization.utils import StageTimer, get_video_properties, setup_logging


def load_match_info(match_id: str, config: PipelineConfig, matches_yaml: str = None) -> dict:
    """Load match video/GT paths from matches.yaml."""
    if matches_yaml is None:
        matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        data = yaml.safe_load(f)

    if match_id not in data["matches"]:
        raise ValueError(f"Unknown match: {match_id}. Available: {list(data['matches'].keys())}")

    info = data["matches"][match_id]
    info["video_path"] = str(Path(config.data_dir) / info["video_path"])
    info["gt_path"] = str(Path(config.data_dir) / info["gt_path"])
    if "goals_path" in info:
        info["goals_path"] = str(Path(config.data_dir) / info["goals_path"])
    return info


def run_pipeline(
    video_path: str,
    gt_path: str,
    output_dir: str,
    config: PipelineConfig,
    match_id: str = "unknown",
    skip_ocr: bool = False,
    skip_video_export: bool = False,
    goals_path: str = None,
    run_id: str = None,
):
    """Run the full pipeline on a single match.

    Per-stage timings are recorded with StageTimer and saved to
    `<results_dir>/timings/<run_id>/<match_id>.json` if run_id is provided.
    """
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

    # Per-stage wall-clock timer (stages mark cached=True when loading from disk)
    stage_timer = StageTimer()
    timings = {}  # kept for backward compat in evaluation.json

    # --- Stage 1: Shot segmentation ---
    shot_cache = out / "shot_boundaries.csv"
    if shot_cache.exists():
        logger.info("Loading cached shot boundaries")
        with stage_timer.time("shot_segmentation", cached=True):
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
        with stage_timer.time("shot_segmentation"):
            segmenter = ShotSegmenter(config.shot_segmentation)
            shots, _diffs = segmenter.run(video_path)
        timings["shot_segmentation"] = stage_timer.stages["shot_segmentation"]["duration_sec"]

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
        with stage_timer.time("object_detection", cached=True):
            player_df = pd.read_csv(player_cache)
    else:
        logger.info("Stage 2: YOLO object detection...")
        with stage_timer.time("object_detection"):
            detector = PlayerDetector(config.object_detection)
            player_df = detector.run(video_path)
        timings["object_detection"] = stage_timer.stages["object_detection"]["duration_sec"]

        player_df.to_csv(player_cache, index=False)
        logger.info("  %d frames processed (saved to %s)", len(player_df), player_cache)

    # --- Stage 3: Frame-level play classification (for player-count plots) ---
    # Produces frame_classification.csv only.  No play_segments — highlight
    # selection now uses per-shot avg player count directly from player_counts.
    frame_cache = out / "frame_classification.csv"
    if not frame_cache.exists():
        logger.info("Stage 3: Frame-level play classification...")
        with stage_timer.time("play_classifier"):
            classifier = PlayClassifier(config.play_classifier)
            _, frame_df = classifier.classify(player_df, props["fps"])
        timings["play_classifier"] = stage_timer.stages["play_classifier"]["duration_sec"]
        frame_df.to_csv(frame_cache, index=False)
        logger.info("  Frame classification saved to %s", frame_cache)
    else:
        logger.info("Loading cached frame classification")
        with stage_timer.time("play_classifier", cached=True):
            pass  # nothing actually loaded here

    # --- Stage 4: OCR scoreboard ---
    score_cache = out / "score_events.csv"
    score_events = []
    if score_cache.exists():
        logger.info("Loading cached score events")
        with stage_timer.time("ocr_scoreboard", cached=True):
            try:
                score_df = pd.read_csv(score_cache)
                from soccer_summarization.ocr_scoreboard import ScoreEvent
                score_events = [
                    ScoreEvent(**row) for _, row in score_df.iterrows()
                ]
            except (pd.errors.EmptyDataError, KeyError):
                logger.info("  Empty or invalid score_events.csv, treating as no detections")
                score_events = []
    elif not skip_ocr:
        logger.info("Stage 4: OCR scoreboard detection...")
        with stage_timer.time("ocr_scoreboard"):
            ocr_detector = ScoreboardDetector(config.ocr_scoreboard)
            region, score_events = ocr_detector.run(video_path)
        timings["ocr_scoreboard"] = stage_timer.stages["ocr_scoreboard"]["duration_sec"]

        score_df = ScoreboardDetector.events_to_dataframe(score_events)
        score_df.to_csv(score_cache, index=False)
        if region:
            # Save display bbox (edge-aware tightened) for diagnostics,
            # falling back to the OCR bbox if display_bbox isn't set.
            db = region.display_bbox or {"x": region.x, "y": region.y, "w": region.w, "h": region.h}
            region_info = {"x": int(db["x"]), "y": int(db["y"]), "w": int(db["w"]), "h": int(db["h"])}
            with open(out / "scoreboard_region.json", "w") as f:
                json.dump(region_info, f)
        logger.info("  %d score events (saved to %s)", len(score_events), score_cache)

    # --- Stage 5: Select play long shots as highlights ---
    # Highlights = long shots (>= min_shot_duration) whose avg per-frame
    # player count is >= play_threshold.  Each highlight is one full shot,
    # no intersection with frame-level play classification, no merging.
    highlight_cache = out / "highlights.csv"
    logger.info("Stage 5: Selecting play long shots...")
    assembler = HighlightAssembler(config.highlight_assembler)

    with stage_timer.time("highlight_assembly"):
        highlights = assembler.select_play_long_shots(
            shots=shots,
            player_counts=player_df,
            player_threshold=float(config.play_classifier.play_threshold),
        )
    timings["highlight_assembly"] = stage_timer.stages["highlight_assembly"]["duration_sec"]

    hl_df = assembler.highlights_to_dataframe(highlights)
    hl_df.to_csv(highlight_cache, index=False)

    total_highlight = sum(h.duration for h in highlights)
    logger.info(
        "  %d play long shots, %.1fs total (%.1f%% compression)",
        len(highlights), total_highlight,
        (1 - total_highlight / total_duration) * 100 if total_duration > 0 else 0,
    )

    # --- Stage 5b: Goal-shot selection ---
    # For each detected goal, pick the full play shot whose end aligns with the
    # goal moment.  Whole shots, no clipping/merging — the "moment of play
    # leading to the goal".
    goal_times = [e.timestamp for e in score_events]
    with stage_timer.time("goal_shot_selection"):
        goal_highlights = assembler.select_goal_shots(
            shots=shots,
            player_counts=player_df,
            goal_times=goal_times,
            fps=props["fps"],
            player_threshold=float(config.play_classifier.play_threshold),
        )
    timings["goal_shot_selection"] = stage_timer.stages["goal_shot_selection"]["duration_sec"]
    goal_hl_df = assembler.highlights_to_dataframe(goal_highlights)
    goal_hl_df.to_csv(out / "goal_highlights.csv", index=False)

    # --- Stage 6: Export highlight video ---
    if not skip_video_export:
        highlight_video = out / "highlight_video.mp4"
        if not highlight_video.exists():
            logger.info("Stage 6: Exporting highlight video...")
            with stage_timer.time("video_export"):
                assembler.export_video(highlights, video_path, str(highlight_video))
            timings["video_export"] = stage_timer.stages["video_export"]["duration_sec"]
            logger.info("  Saved to %s", highlight_video)
        else:
            logger.info("Stage 6: Highlight video already exists")

    # --- Stage 7: Evaluation ---
    # If goal_clip_max_duration is set (e.g. "last 10s" mode), the GT clips are
    # also truncated to that final window so the comparison is apples-to-apples.
    logger.info("Stage 7: Evaluation against ground truth...")
    gt_clip_max = config.highlight_assembler.goal_clip_max_duration
    with stage_timer.time("evaluation"):
        evaluator = Evaluator(config.evaluation)
        goal_clips: list[GoalClip] = []
        if goals_path:
            gt_events = evaluator.load_ground_truth_frame_level(goals_path, props["fps"])
            goal_clips = Evaluator.load_goal_clips(
                goals_path, props["fps"], max_duration=gt_clip_max,
            )
            logger.info(
                "  Using frame-level GT from %s (%d clips%s)",
                goals_path, len(goal_clips),
                f", truncated to last {gt_clip_max:.0f}s" if gt_clip_max is not None else "",
            )
        else:
            gt_df = evaluator.load_ground_truth(gt_path)
            gt_events = evaluator.extract_goal_events(gt_df)

        predicted_goal_times = [e.timestamp for e in score_events]
        # The "deliverable" is the goal-refined clips, so evaluate AGAINST those.
        goal_highlight_segments = [(h.start_time, h.end_time) for h in goal_highlights]

        result = evaluator.evaluate_match(
            match_id=match_id,
            highlight_segments=goal_highlight_segments,
            predicted_goal_times=predicted_goal_times,
            gt_events=gt_events,
            total_video_duration=total_duration,
        )

        # Per-goal clip-quality metrics (require frame-level GT from goals.csv)
        clip_metrics: dict = {}
        clip_aware: dict = {}
        if goal_clips:
            clip_metrics = evaluator.evaluate_clip_quality(
                goal_clips=goal_clips,
                goal_highlight_segments=goal_highlight_segments,
                predicted_goal_times=predicted_goal_times,
            )
            clip_aware = evaluator.evaluate_clip_aware_f1(
                goal_clips=goal_clips,
                predicted_goal_times=predicted_goal_times,
                goal_highlight_segments=goal_highlight_segments,
                iou_threshold=0.5,
            )
    timings["evaluation"] = stage_timer.stages["evaluation"]["duration_sec"]
    if goal_clips:
        logger.info(
            "Clip quality: IoU=%.3f Cov=%.3f Prec=%.3f StartMAE=%.1fs EndMAE=%.1fs",
            clip_metrics["clip_iou_mean"],
            clip_metrics["clip_coverage_mean"],
            clip_metrics["clip_precision_mean"],
            clip_metrics["clip_start_mae"],
            clip_metrics["clip_end_mae"],
        )
        logger.info(
            "Clip-aware F1 (IoU>=0.5): F1=%.3f  TP=%d FP=%d FN=%d",
            clip_aware["clip_aware_f1"],
            clip_aware["clip_aware_tp"],
            clip_aware["clip_aware_fp"],
            clip_aware["clip_aware_fn"],
        )

    # Save evaluation results
    eval_results = {
        "match_id": match_id,
        # Goal-detection metrics (OCR predictions vs GT timestamps)
        "true_positives": result.true_positives,
        "false_positives": result.false_positives,
        "false_negatives": result.false_negatives,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        # Summary stats — computed from goal_highlights, the actual deliverable
        "total_video_duration": result.total_video_duration,
        "highlight_duration": result.highlight_duration,
        "compression_ratio": result.compression_ratio,
        "goal_coverage": result.goal_coverage,
        "num_gt_events": len(gt_events),
        "num_predicted_goals": len(predicted_goal_times),
        "num_goal_clips": len(goal_highlights),
        "num_play_long_shots": len(highlights),
        # Segment stats
        "num_segments": result.num_segments,
        "mean_segment_duration": result.mean_segment_duration,
        "timings": timings,
        # Per-goal clip-quality metrics (predicted clip vs frame-level GT clip)
        **clip_metrics,
        # End-to-end clip-aware F1 (detection AND clip IoU >= 0.5)
        **clip_aware,
    }
    def _json_default(obj):
        if hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        return float(obj)

    with open(out / "evaluation.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=_json_default)

    # Save per-stage timings for this run (only when run_id provided)
    if run_id:
        timing_dir = Path(config.results_dir) / "timings" / run_id
        timing_dir.mkdir(parents=True, exist_ok=True)
        timing_record = {
            "match_id": match_id,
            "run_id": run_id,
            "video_path": str(video_path),
            "video_duration_sec": float(total_duration),
            "fps": float(props["fps"]),
            "width": int(props["width"]),
            "height": int(props["height"]),
            "stages": stage_timer.to_dict(),
        }
        with open(timing_dir / f"{match_id}.json", "w") as f:
            json.dump(timing_record, f, indent=2, default=_json_default)

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
    parser.add_argument("--run-id", default=None,
                        help="Tag for saving per-stage timings under "
                             "<results_dir>/timings/<run_id>/. Default: no timing log saved.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = PipelineConfig.from_yaml(args.config)

    if args.match_id:
        info = load_match_info(args.match_id, config)
        video_path = info["video_path"]
        gt_path = info["gt_path"]
        goals_path = info.get("goals_path")
        match_id = args.match_id
        output_dir = args.output_dir or f"{config.results_dir}/per_match/{match_id}"
    else:
        if not args.gt:
            parser.error("--gt is required when using --video")
        video_path = args.video
        gt_path = args.gt
        goals_path = None
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
        goals_path=goals_path,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
