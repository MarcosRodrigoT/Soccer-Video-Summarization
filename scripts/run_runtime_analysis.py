#!/usr/bin/env python3
"""Cold-cache substage-level runtime analysis on a single match.

Unlike `run_single_match.py`, this script:
  * Re-implements the orchestration of each module inline so we can wrap
    *substages* (not just top-level stages) in timing context managers.
  * Refuses to use any cache on disk — every stage actually re-computes.
  * Writes everything to a dedicated output directory so it does not
    clobber the canonical per-match results.

The output JSON is consumed by `generate_runtime_figure.py` to produce
the LaTeX table and PDF figure for Section 8.5 of the paper.

Usage:
    python scripts/run_runtime_analysis.py \\
        --match-id Match1 \\
        --config config/default_hr.yaml \\
        --output-dir results_hr/runtime_analysis/Match1
"""

import argparse
import json
import logging
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator, GoalClip
from soccer_summarization.highlight_assembler import HighlightAssembler
from soccer_summarization.object_detection import PlayerDetector
from soccer_summarization.ocr_scoreboard import ScoreboardDetector
from soccer_summarization.play_classifier import PlayClassifier
from soccer_summarization.shot_segmentation import ShotSegmenter, _build_shots
from soccer_summarization.utils import (
    format_time_detailed,
    get_video_properties,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ─── Substage timing ────────────────────────────────────────────────────────

class SubstageTimer:
    """Records wall-clock time for nested (stage, substage) pairs.

    Each substage is keyed by (stage, substage) and stores the elapsed
    seconds.  GPU work is synchronised on enter/exit so the recorded
    interval includes any pending CUDA kernels.
    """

    def __init__(self, sync_cuda: bool = True):
        self._records: list[dict] = []
        self._sync = sync_cuda and self._cuda_available()

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _maybe_sync(self):
        if not self._sync:
            return
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass

    def time(self, stage: str, substage: str):
        return _Span(self, stage, substage)

    def record(self, stage: str, substage: str, duration_sec: float):
        self._records.append({
            "stage": stage,
            "substage": substage,
            "duration_sec": float(duration_sec),
        })

    def to_list(self) -> list[dict]:
        return list(self._records)

    def total(self) -> float:
        return sum(r["duration_sec"] for r in self._records)


class _Span:
    def __init__(self, parent: SubstageTimer, stage: str, substage: str):
        self.parent = parent
        self.stage = stage
        self.substage = substage

    def __enter__(self):
        self.parent._maybe_sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.parent._maybe_sync()
        self.parent.record(
            self.stage, self.substage, time.perf_counter() - self._t0
        )
        return False


# ─── Hardware fingerprint ───────────────────────────────────────────────────

def detect_hardware() -> dict:
    """Best-effort hardware fingerprint for the runtime report."""
    info: dict = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }

    # CPU
    cpu_name = ""
    try:
        out = subprocess.check_output(
            ["lscpu"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if line.startswith("Model name:"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    info["cpu_name"] = cpu_name

    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        info["cpu_count"] = None

    # GPU
    gpus: list[dict] = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "name": parts[0],
                    "memory_mib": int(parts[1]),
                    "driver": parts[2],
                })
    except Exception:
        pass
    info["gpus"] = gpus

    # Torch + CUDA
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        info["torch_version"] = None
        info["cuda_version"] = None
        info["cuda_available"] = False

    # Memory
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    info["mem_total_kib"] = int(line.split()[1])
                    break
    except Exception:
        info["mem_total_kib"] = None

    return info


# ─── Match metadata ─────────────────────────────────────────────────────────

def load_match_info(match_id: str, config: PipelineConfig,
                    matches_yaml: Path) -> dict:
    with open(matches_yaml) as f:
        data = yaml.safe_load(f)
    if match_id not in data["matches"]:
        raise ValueError(
            f"Unknown match: {match_id}. "
            f"Available: {list(data['matches'].keys())}"
        )
    info = data["matches"][match_id]
    info["video_path"] = str(Path(config.data_dir) / info["video_path"])
    info["gt_path"] = str(Path(config.data_dir) / info["gt_path"])
    if "goals_path" in info:
        info["goals_path"] = str(Path(config.data_dir) / info["goals_path"])
    return info


# ─── Substage-instrumented OCR `track_scores` ───────────────────────────────

def track_scores_instrumented(
    detector: ScoreboardDetector,
    video_path: str,
    region,
    overlay_mask,
    fps: int,
    total_frames: int,
    timer: SubstageTimer,
) -> list:
    """Mirrors ScoreboardDetector.track_scores but with internal timing.

    Splits the work into three substages:
      4d. OCR sampling + recognition (the dominant cost)
      4e. Multi-strategy parsing
      4f. Robustness cascade (temporal median + monotonicity + refinement)
    """
    interval_frames = int(detector.config.score_sample_interval * fps)
    ocr = detector._get_ocr()
    pad = 15
    sb_x_min, sb_x_max = pad, pad + region.w
    sb_y_min, sb_y_max = pad, pad + region.h

    # ── 4d. OCR sampling + recognition ────────────────────────────────────
    all_readings: list[tuple] = []
    with timer.time("ocr_scoreboard", "ocr_sampling_recognition"):
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            if overlay_mask is not None:
                crop = detector._crop_and_mask(frame, region, overlay_mask, pad=pad)
            else:
                crop = detector._crop_region(frame, region, pad=pad)
            if crop.size == 0:
                frame_idx += interval_frames
                continue

            scale = 4
            upscaled = cv2.resize(
                crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            results = ocr.readtext(upscaled, text_threshold=0.5, low_text=0.3)

            for bbox, text, conf in results:
                cx = (bbox[0][0] + bbox[2][0]) / 2 / scale
                cy = (bbox[0][1] + bbox[2][1]) / 2 / scale
                if cx < sb_x_min or cx > sb_x_max or cy < sb_y_min or cy > sb_y_max:
                    continue
                all_readings.append(
                    (frame_idx, text.strip(), float(cx), float(cy), float(conf))
                )

            frame_idx += interval_frames
        cap.release()

    logger.info("Collected %d raw OCR readings", len(all_readings))

    if not all_readings:
        return []

    # ── 4e. Multi-strategy parsing ────────────────────────────────────────
    with timer.time("ocr_scoreboard", "parsing"):
        raw_scores = detector._parse_scores_multi_strategy(all_readings, fps)
    logger.info("Parsed %d raw score readings", len(raw_scores))

    if not raw_scores:
        return []

    # ── 4f. Robustness cascade ────────────────────────────────────────────
    with timer.time("ocr_scoreboard", "robustness_cascade"):
        events = detector._detect_score_changes(raw_scores, fps)
    logger.info("Detected %d score change events", len(events))

    return events


# ─── Main orchestration ─────────────────────────────────────────────────────

def run(match_id: str, config_path: Path, output_dir: Path) -> dict:
    config = PipelineConfig.from_yaml(str(config_path))

    matches_yaml = config_path.parent / "matches_hr.yaml"
    if not matches_yaml.exists():
        matches_yaml = config_path.parent / "matches.yaml"
    info = load_match_info(match_id, config, matches_yaml)

    video_path = info["video_path"]
    gt_path = info["gt_path"]
    goals_path = info.get("goals_path")

    # Fresh output dir → guarantees cold cache
    if output_dir.exists():
        logger.warning("Removing existing output dir: %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    props = get_video_properties(video_path)
    fps = props["fps"]
    total_frames = props["total_frames"]
    duration = props["duration"]
    logger.info(
        "%s: %s (%.1f min, %d fps, %dx%d, %d frames)",
        match_id, Path(video_path).name, duration / 60,
        fps, props["width"], props["height"], total_frames,
    )

    timer = SubstageTimer(sync_cuda=True)

    # ── Stage 1: Shot segmentation ────────────────────────────────────────
    logger.info("[1/6] Shot segmentation...")
    segmenter = ShotSegmenter(config.shot_segmentation)
    with timer.time("shot_segmentation", "frame_difference"):
        diffs = segmenter.compute_frame_differences(video_path)
    with timer.time("shot_segmentation", "abrupt_detection"):
        abrupt = segmenter.detect_abrupt_transitions(diffs)
    with timer.time("shot_segmentation", "gradual_detection"):
        gradual = segmenter.detect_gradual_transitions(diffs, abrupt)
    with timer.time("shot_segmentation", "shot_assembly"):
        shots = _build_shots(abrupt, gradual, total_frames, fps)
    logger.info(
        "  %d shots (%d abrupt + %d gradual)",
        len(shots), len(abrupt), len(gradual),
    )

    # ── Stage 2: YOLO object detection ────────────────────────────────────
    logger.info("[2/6] YOLO object detection...")
    detector = PlayerDetector(config.object_detection)
    with timer.time("object_detection", "model_load"):
        detector._load_model()
    with timer.time("object_detection", "inference_loop"):
        player_df = detector.run(video_path)
    logger.info("  %d frames processed", len(player_df))

    # ── Stage 3: Play classification ──────────────────────────────────────
    logger.info("[3/6] Play classification...")
    classifier = PlayClassifier(config.play_classifier)
    with timer.time("play_classifier", "smoothing_threshold"):
        _, frame_df = classifier.classify(player_df, props["fps"])

    # ── Stage 4: OCR scoreboard ───────────────────────────────────────────
    logger.info("[4/6] OCR scoreboard...")
    ocr_detector = ScoreboardDetector(config.ocr_scoreboard)
    with timer.time("ocr_scoreboard", "variability_map"):
        std_map = ocr_detector.find_static_overlays(video_path)
    with timer.time("ocr_scoreboard", "overlay_mask"):
        overlay_mask, raw_mask = ocr_detector.build_overlay_mask(std_map)
    with timer.time("ocr_scoreboard", "region_search"):
        region = ocr_detector.find_scoreboard_region(
            video_path, std_map, overlay_mask, raw_mask
        )
    if region is None:
        logger.error("Could not locate scoreboard - aborting OCR substages")
        score_events = []
    else:
        # Substages 4d/4e/4f are timed inside this helper
        score_events = track_scores_instrumented(
            ocr_detector, video_path, region, None, fps, total_frames, timer
        )

    # ── Stage 5: Highlight assembly ───────────────────────────────────────
    logger.info("[5/6] Highlight assembly...")
    assembler = HighlightAssembler(config.highlight_assembler)
    with timer.time("highlight_assembly", "play_long_shots"):
        highlights = assembler.select_play_long_shots(
            shots=shots,
            player_counts=player_df,
            player_threshold=float(config.play_classifier.play_threshold),
        )
    goal_times = [e.timestamp for e in score_events]
    with timer.time("highlight_assembly", "goal_shot_selection"):
        goal_highlights = assembler.select_goal_shots(
            shots=shots,
            player_counts=player_df,
            goal_times=goal_times,
            fps=props["fps"],
            player_threshold=float(config.play_classifier.play_threshold),
        )
    logger.info(
        "  %d play long shots, %d goal clips",
        len(highlights), len(goal_highlights),
    )

    # ── Stage 6: Evaluation ───────────────────────────────────────────────
    logger.info("[6/6] Evaluation...")
    evaluator = Evaluator(config.evaluation)
    gt_clip_max = config.highlight_assembler.goal_clip_max_duration

    with timer.time("evaluation", "load_ground_truth"):
        if goals_path:
            gt_events = evaluator.load_ground_truth_frame_level(goals_path, fps)
            goal_clips = Evaluator.load_goal_clips(
                goals_path, fps, max_duration=gt_clip_max,
            )
        else:
            gt_df = evaluator.load_ground_truth(gt_path)
            gt_events = evaluator.extract_goal_events(gt_df)
            goal_clips = []

    with timer.time("evaluation", "match_and_score"):
        goal_segments = [(h.start_time, h.end_time) for h in goal_highlights]
        result = evaluator.evaluate_match(
            match_id=match_id,
            highlight_segments=goal_segments,
            predicted_goal_times=[e.timestamp for e in score_events],
            gt_events=gt_events,
            total_video_duration=duration,
        )
        clip_aware: dict = {}
        if goal_clips:
            clip_aware = evaluator.evaluate_clip_aware_f1(
                goal_clips=goal_clips,
                predicted_goal_times=[e.timestamp for e in score_events],
                goal_highlight_segments=goal_segments,
                iou_threshold=0.5,
            )

    # ── Bundle results ────────────────────────────────────────────────────
    out = {
        "match_id": match_id,
        "video_path": video_path,
        "video_duration_sec": float(duration),
        "fps": int(fps),
        "total_frames": int(total_frames),
        "width": int(props["width"]),
        "height": int(props["height"]),
        "yolo_frame_step": int(config.object_detection.frame_step),
        "ocr_sample_interval_sec": float(config.ocr_scoreboard.score_sample_interval),
        "num_shots": len(shots),
        "num_score_events": len(score_events),
        "num_goal_clips": len(goal_highlights),
        "num_play_long_shots": len(highlights),
        "f1": float(getattr(result, "f1", 0.0)),
        "clip_aware_f1": float(clip_aware.get("clip_aware_f1", 0.0)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hardware": detect_hardware(),
        "substages": timer.to_list(),
        "total_compute_sec": timer.total(),
        "realtime_factor": timer.total() / duration if duration > 0 else None,
    }

    out_json = output_dir / "runtime_substages.json"
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", out_json)
    logger.info(
        "Total pipeline compute: %s  (real-time factor: %.3fx)",
        format_time_detailed(out["total_compute_sec"]),
        out["realtime_factor"] or 0.0,
    )

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match-id", default="Match1")
    ap.add_argument(
        "--config",
        default="config/default_hr.yaml",
        help="Pipeline config YAML (defaults to HR 1080p config).",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: <results_dir>/runtime_analysis/<match_id>)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    cfg_path = Path(args.config).resolve()
    config = PipelineConfig.from_yaml(str(cfg_path))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.results_dir) / "runtime_analysis" / args.match_id
    )
    output_dir = output_dir.resolve()

    run(args.match_id, cfg_path, output_dir)


if __name__ == "__main__":
    main()
