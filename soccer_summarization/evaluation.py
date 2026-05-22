"""Evaluation metrics for goal detection and highlight quality.

Bridges the gap between system output (highlight segments with optional goal
timestamps) and ground truth (commentary-level goal annotations in CSV).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import EvaluationConfig

logger = logging.getLogger(__name__)


@dataclass
class GoalEvent:
    """A discrete goal event extracted from GT or predictions."""
    start_time: float
    end_time: float
    team: str  # "team1", "team2", or "any"

    @property
    def midpoint(self) -> float:
        return (self.start_time + self.end_time) / 2.0


@dataclass
class GoalClip:
    """Frame-level GT clip for one goal (from goals.csv)."""
    start_time: float       # start_frame / fps  — start of the play action
    end_time: float         # end_frame_shot / fps — end of the goal celebration
    scoreboard_time: float  # end_frame_scoreboard / fps — scoreboard update


@dataclass
class EvaluationResult:
    """Evaluation metrics for a single match."""
    match_id: str
    # Goal detection
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    # Highlight quality
    total_video_duration: float
    highlight_duration: float
    compression_ratio: float
    goal_coverage: float  # fraction of goal-event time captured in highlights
    # Extended metrics
    num_segments: int = 0
    mean_segment_duration: float = 0.0
    mean_tiou: float = 0.0
    goal_timing_mae: float = 0.0  # mean abs error in seconds for TPs
    tiou_per_event: list = field(default_factory=list)
    f1_at_tiou: dict = field(default_factory=dict)
    # Clip-quality metrics (require frame-level GT from goals.csv)
    clip_iou_mean: float = 0.0        # tIoU between predicted highlight and GT clip
    clip_coverage_mean: float = 0.0   # recall: fraction of GT clip covered
    clip_precision_mean: float = 0.0  # precision: fraction of highlight that is GT clip
    clip_start_mae: float = 0.0       # MAE between predicted start and GT start_frame
    clip_end_mae: float = 0.0         # MAE between predicted end and GT end_frame_shot
    clip_iou_per_goal: list = field(default_factory=list)


def _compute_temporal_iou(seg_start: float, seg_end: float,
                          gt_start: float, gt_end: float) -> float:
    """Compute temporal IoU between a predicted segment and a GT segment."""
    intersection_start = max(seg_start, gt_start)
    intersection_end = min(seg_end, gt_end)
    intersection = max(0.0, intersection_end - intersection_start)

    union = (seg_end - seg_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _find_best_segment_for_event(
    highlight_segments: list[tuple[float, float]],
    event: GoalEvent,
    tolerance: float,
) -> tuple[float, float, Optional[tuple[float, float]]]:
    """Find the highlight segment with highest tIoU for a GT event.

    Returns (best_tiou, timing_error, best_segment_or_None).
    """
    best_tiou = 0.0
    best_seg = None

    # Expand event by tolerance for matching context
    expanded_start = event.start_time - tolerance
    expanded_end = event.end_time + tolerance

    for seg_start, seg_end in highlight_segments:
        tiou = _compute_temporal_iou(seg_start, seg_end,
                                     expanded_start, expanded_end)
        if tiou > best_tiou:
            best_tiou = tiou
            best_seg = (seg_start, seg_end)

    # Timing error: distance from segment midpoint to event midpoint
    if best_seg is not None:
        seg_mid = (best_seg[0] + best_seg[1]) / 2.0
        timing_error = abs(seg_mid - event.midpoint)
    else:
        timing_error = float("inf")

    return best_tiou, timing_error, best_seg


class Evaluator:
    """Evaluate highlight extraction against ground truth."""

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def load_ground_truth(self, gt_path: str) -> pd.DataFrame:
        """Load ground truth CSV.

        Tries utf-8-sig first, falls back to latin-1 for encoding issues.
        """
        for encoding in ("utf-8-sig", "latin-1", "utf-8"):
            try:
                df = pd.read_csv(gt_path, encoding=encoding)
                # Normalize column names
                df.columns = [c.strip().lower() for c in df.columns]
                if "goalgt1" in df.columns:
                    df = df.rename(columns={"goalgt1": "goal_team1", "goalgt2": "goal_team2"})
                return df
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError(f"Cannot read GT file: {gt_path}")

    def load_ground_truth_frame_level(self, goals_path: str, fps: float) -> list[GoalEvent]:
        """Load frame-level ground truth from goals.csv.

        Each row is one goal. Uses end_frame_scoreboard as the reference timestamp —
        the moment the broadcast scoreboard first reflects the new score.
        """
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(goals_path, encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            raise ValueError(f"Cannot read goals file: {goals_path}")
        events = []
        for _, row in df.iterrows():
            t = float(row["end_frame_scoreboard"]) / fps
            events.append(GoalEvent(start_time=t, end_time=t, team="any"))
        events.sort(key=lambda e: e.start_time)
        logger.info("Loaded %d frame-level GT events from %s", len(events), goals_path)
        return events

    @staticmethod
    def load_goal_clips(goals_path: str, fps: float) -> list[GoalClip]:
        """Load per-goal clip boundaries from goals.csv.

        Returns one GoalClip per goal with:
          start_time       = start_frame / fps  (first frame of the play action)
          end_time         = end_frame_shot / fps (last frame of the celebration)
          scoreboard_time  = end_frame_scoreboard / fps (scoreboard update)
        """
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(goals_path, encoding=encoding)
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            raise ValueError(f"Cannot read goals file: {goals_path}")
        clips = []
        for _, row in df.iterrows():
            clips.append(GoalClip(
                start_time=float(row["start_frame"]) / fps,
                end_time=float(row["end_frame_shot"]) / fps,
                scoreboard_time=float(row["end_frame_scoreboard"]) / fps,
            ))
        clips.sort(key=lambda c: c.scoreboard_time)
        return clips

    def evaluate_clip_quality(
        self,
        goal_clips: list[GoalClip],
        goal_highlight_segments: list[tuple[float, float]],
        predicted_goal_times: list[float],
    ) -> dict:
        """Compare predicted goal-highlight clips against GT clip boundaries.

        For each GT goal clip, finds the best-matching predicted highlight
        (the one with highest tIoU) and computes:
          - clip_iou:        temporal IoU between predicted highlight and GT clip
          - clip_coverage:   fraction of GT clip duration covered by prediction
          - clip_precision:  fraction of predicted clip that overlaps GT clip
          - clip_start_mae:  |predicted start - GT start_frame/fps|
          - clip_end_mae:    |predicted end   - GT end_frame_shot/fps|

        Only evaluates GT clips for which a predicted goal exists within
        the tolerance window (TPs only — undetected goals are skipped).
        """
        tolerance = self.config.tolerance_window
        iou_list, coverage_list, precision_list = [], [], []
        start_mae_list, end_mae_list = [], []

        # Per-GT-clip parallel lists (None for undetected goals, so that callers
        # can align with the GT goal index in dashboards / tables).
        iou_per_gt:       list = []
        coverage_per_gt:  list = []
        precision_per_gt: list = []
        start_mae_per_gt: list = []
        end_mae_per_gt:   list = []

        for clip in goal_clips:
            # Check if this goal was detected (TP)
            matched = any(
                abs(t - clip.scoreboard_time) <= tolerance
                for t in predicted_goal_times
            )
            if not matched:
                iou_per_gt.append(None)
                coverage_per_gt.append(None)
                precision_per_gt.append(None)
                start_mae_per_gt.append(None)
                end_mae_per_gt.append(None)
                continue  # FN — no predicted highlight to compare

            # Find the best-matching highlight segment for this clip
            gt_start, gt_end = clip.start_time, clip.end_time
            gt_dur = max(gt_end - gt_start, 1e-6)

            best_iou = 0.0
            best_cov = 0.0
            best_prec = 0.0
            best_start_err = float("inf")
            best_end_err = float("inf")

            for seg_start, seg_end in goal_highlight_segments:
                seg_dur = max(seg_end - seg_start, 1e-6)
                inter_s = max(seg_start, gt_start)
                inter_e = min(seg_end, gt_end)
                inter = max(0.0, inter_e - inter_s)
                if inter == 0:
                    continue
                union = seg_dur + gt_dur - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou
                    best_cov = inter / gt_dur
                    best_prec = inter / seg_dur
                    best_start_err = abs(seg_start - gt_start)
                    best_end_err = abs(seg_end - gt_end)

            iou_list.append(best_iou)
            coverage_list.append(best_cov)
            precision_list.append(best_prec)
            iou_per_gt.append(best_iou)
            coverage_per_gt.append(best_cov)
            precision_per_gt.append(best_prec)
            if best_start_err < float("inf"):
                start_mae_list.append(best_start_err)
                end_mae_list.append(best_end_err)
                start_mae_per_gt.append(best_start_err)
                end_mae_per_gt.append(best_end_err)
            else:
                start_mae_per_gt.append(None)
                end_mae_per_gt.append(None)

        def safe_mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        return {
            "clip_iou_mean": safe_mean(iou_list),
            "clip_coverage_mean": safe_mean(coverage_list),
            "clip_precision_mean": safe_mean(precision_list),
            "clip_start_mae": safe_mean(start_mae_list),
            "clip_end_mae": safe_mean(end_mae_list),
            # Per-GT-clip values (length == len(goal_clips); None for FN)
            "clip_iou_per_gt":       iou_per_gt,
            "clip_coverage_per_gt":  coverage_per_gt,
            "clip_precision_per_gt": precision_per_gt,
            "clip_start_mae_per_gt": start_mae_per_gt,
            "clip_end_mae_per_gt":   end_mae_per_gt,
        }

    def evaluate_clip_aware_f1(
        self,
        goal_clips: list[GoalClip],
        predicted_goal_times: list[float],
        goal_highlight_segments: list[tuple[float, float]],
        iou_threshold: float = 0.5,
    ) -> dict:
        """End-to-end precision / recall / F1 requiring BOTH detection AND a good clip.

        A GT goal counts as TP iff:
          (a) some predicted goal time falls within +-tolerance of its scoreboard time, AND
          (b) the best-matching highlight segment has IoU >= iou_threshold against
              the GT clip [start_frame/fps, end_frame_shot/fps].

        Predictions that don't match any GT goal in time are FPs.  Predictions that
        match in time but produce a poor clip do NOT count as TPs (they're FN on the
        GT side, FP on the prediction side).
        """
        tolerance = self.config.tolerance_window
        tp = 0
        fn = 0
        matched_pred_idx: set[int] = set()

        for clip in goal_clips:
            # (a) detection check: any prediction within tolerance of scoreboard_time
            best_match_idx = None
            best_match_dist = float("inf")
            for i, pt in enumerate(predicted_goal_times):
                d = abs(pt - clip.scoreboard_time)
                if d <= tolerance and d < best_match_dist:
                    best_match_idx = i
                    best_match_dist = d

            if best_match_idx is None:
                fn += 1
                continue

            # (b) boundary check: best clip IoU against GT clip
            gt_start, gt_end = clip.start_time, clip.end_time
            gt_dur = max(gt_end - gt_start, 1e-6)
            best_iou = 0.0
            for seg_start, seg_end in goal_highlight_segments:
                seg_dur = max(seg_end - seg_start, 1e-6)
                inter_s = max(seg_start, gt_start)
                inter_e = min(seg_end, gt_end)
                inter = max(0.0, inter_e - inter_s)
                if inter == 0:
                    continue
                union = seg_dur + gt_dur - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou

            if best_iou >= iou_threshold:
                tp += 1
                matched_pred_idx.add(best_match_idx)
            else:
                fn += 1  # detection succeeded but clip boundaries are too off

        fp = len(predicted_goal_times) - len(matched_pred_idx)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "clip_aware_tp": tp,
            "clip_aware_fp": fp,
            "clip_aware_fn": fn,
            "clip_aware_precision": precision,
            "clip_aware_recall": recall,
            "clip_aware_f1": f1,
            "clip_aware_iou_threshold": iou_threshold,
        }

    def extract_goal_events(self, gt_df: pd.DataFrame) -> list[GoalEvent]:
        """Group consecutive GoalGT=1 segments into discrete goal events.

        A new goal event starts when there's a gap > threshold between
        consecutive GoalGT=1 segments.
        """
        gap_threshold = self.config.goal_gap_threshold
        events = []

        for team_col, team_name in [("goal_team1", "team1"), ("goal_team2", "team2")]:
            if team_col not in gt_df.columns:
                continue

            goal_rows = gt_df[gt_df[team_col] == 1].copy()
            if len(goal_rows) == 0:
                continue

            goal_rows = goal_rows.sort_values("start").reset_index(drop=True)

            # Group consecutive segments
            current_start = goal_rows.iloc[0]["start"]
            current_end = goal_rows.iloc[0]["end"]

            for i in range(1, len(goal_rows)):
                row = goal_rows.iloc[i]
                if row["start"] - current_end > gap_threshold:
                    # New event
                    events.append(GoalEvent(
                        start_time=current_start,
                        end_time=current_end,
                        team=team_name,
                    ))
                    current_start = row["start"]
                    current_end = row["end"]
                else:
                    current_end = max(current_end, row["end"])

            # Last event
            events.append(GoalEvent(
                start_time=current_start,
                end_time=current_end,
                team=team_name,
            ))

        # Deduplicate events that overlap (both teams flagged for same goal)
        events = self._deduplicate_events(events)

        logger.info("Extracted %d goal events from GT", len(events))
        for e in events:
            logger.info(
                "  Goal at %.1fs-%.1fs (%s)",
                e.start_time, e.end_time, e.team,
            )
        return events

    def _deduplicate_events(self, events: list[GoalEvent]) -> list[GoalEvent]:
        """Merge overlapping events from different team columns."""
        if not events:
            return []

        events.sort(key=lambda e: e.start_time)
        merged = [events[0]]

        for e in events[1:]:
            prev = merged[-1]
            # If overlapping with previous, merge
            if e.start_time <= prev.end_time + self.config.tolerance_window:
                prev.end_time = max(prev.end_time, e.end_time)
                if prev.team != e.team:
                    prev.team = "any"
            else:
                merged.append(e)

        return merged

    def evaluate_goal_detection(
        self,
        predicted_goal_times: list[float],
        gt_events: list[GoalEvent],
    ) -> dict:
        """Evaluate goal detection precision, recall, F1.

        A prediction matches a GT event if it falls within
        [event.start - tolerance, event.end + tolerance].
        """
        tolerance = self.config.tolerance_window
        matched_gt = set()
        matched_pred = set()

        for i, pred_time in enumerate(predicted_goal_times):
            for j, gt_event in enumerate(gt_events):
                if (gt_event.start_time - tolerance <= pred_time <= gt_event.end_time + tolerance):
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break

        tp = len(matched_gt)
        fp = len(predicted_goal_times) - len(matched_pred)
        fn = len(gt_events) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def evaluate_highlight_quality(
        self,
        highlight_segments: list[tuple[float, float]],
        gt_events: list[GoalEvent],
        total_video_duration: float,
    ) -> dict:
        """Evaluate highlight segment quality.

        Args:
            highlight_segments: List of (start_time, end_time) tuples
            gt_events: Ground truth goal events
            total_video_duration: Total video duration in seconds
        """
        highlight_duration = sum(end - start for start, end in highlight_segments)
        compression_ratio = 1.0 - (highlight_duration / total_video_duration) if total_video_duration > 0 else 0.0

        # Goal coverage: what fraction of goal-event time is captured
        goal_time_covered = 0.0
        total_goal_time = 0.0

        for event in gt_events:
            event_duration = event.end_time - event.start_time
            total_goal_time += event_duration

            for hl_start, hl_end in highlight_segments:
                overlap_start = max(event.start_time, hl_start)
                overlap_end = min(event.end_time, hl_end)
                if overlap_start < overlap_end:
                    goal_time_covered += overlap_end - overlap_start

        goal_coverage = goal_time_covered / total_goal_time if total_goal_time > 0 else 0.0

        # Segment statistics
        num_segments = len(highlight_segments)
        mean_segment_duration = highlight_duration / num_segments if num_segments > 0 else 0.0

        return {
            "total_video_duration": total_video_duration,
            "highlight_duration": highlight_duration,
            "compression_ratio": compression_ratio,
            "goal_coverage": goal_coverage,
            "num_segments": num_segments,
            "mean_segment_duration": mean_segment_duration,
        }

    def evaluate_temporal_quality(
        self,
        highlight_segments: list[tuple[float, float]],
        predicted_goal_times: list[float],
        gt_events: list[GoalEvent],
    ) -> dict:
        """Compute temporal IoU, timing MAE, and F1 at tIoU thresholds."""
        tolerance = self.config.tolerance_window

        # Per-event tIoU: for each GT event, find best matching highlight segment
        tiou_per_event = []
        timing_errors = []

        for event in gt_events:
            best_tiou, timing_error, _ = _find_best_segment_for_event(
                highlight_segments, event, tolerance
            )
            tiou_per_event.append(best_tiou)
            if best_tiou > 0:
                timing_errors.append(timing_error)

        mean_tiou = float(np.mean(tiou_per_event)) if tiou_per_event else 0.0
        goal_timing_mae = float(np.mean(timing_errors)) if timing_errors else 0.0

        # F1 at tIoU thresholds
        f1_at_tiou = {}
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            tp = sum(1 for t in tiou_per_event if t >= threshold)
            fn = len(gt_events) - tp
            # For FP: count predicted goals not matched at this threshold
            fp = self._count_fp_at_tiou(
                highlight_segments, predicted_goal_times, gt_events, threshold, tolerance
            )
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            f1_at_tiou[str(threshold)] = f1

        return {
            "mean_tiou": mean_tiou,
            "goal_timing_mae": goal_timing_mae,
            "tiou_per_event": tiou_per_event,
            "f1_at_tiou": f1_at_tiou,
        }

    def _count_fp_at_tiou(
        self,
        highlight_segments: list[tuple[float, float]],
        predicted_goal_times: list[float],
        gt_events: list[GoalEvent],
        tiou_threshold: float,
        tolerance: float,
    ) -> int:
        """Count false positives at a given tIoU threshold."""
        fp = 0
        for pred_time in predicted_goal_times:
            matched = False
            for event in gt_events:
                expanded_start = event.start_time - tolerance
                expanded_end = event.end_time + tolerance
                if expanded_start <= pred_time <= expanded_end:
                    # Check if any highlight segment covers this well enough
                    best_tiou, _, _ = _find_best_segment_for_event(
                        highlight_segments, event, tolerance
                    )
                    if best_tiou >= tiou_threshold:
                        matched = True
                        break
            if not matched:
                fp += 1
        return fp

    def evaluate_match(
        self,
        match_id: str,
        highlight_segments: list[tuple[float, float]],
        predicted_goal_times: list[float],
        gt_events: list[GoalEvent],
        total_video_duration: float,
    ) -> EvaluationResult:
        """Full evaluation for a single match."""
        goal_metrics = self.evaluate_goal_detection(predicted_goal_times, gt_events)
        quality_metrics = self.evaluate_highlight_quality(
            highlight_segments, gt_events, total_video_duration
        )
        temporal_metrics = self.evaluate_temporal_quality(
            highlight_segments, predicted_goal_times, gt_events
        )

        result = EvaluationResult(
            match_id=match_id,
            **goal_metrics,
            **quality_metrics,
            **temporal_metrics,
        )

        logger.info(
            "Match %s: P=%.2f R=%.2f F1=%.2f | Compression=%.1f%% Coverage=%.1f%% | "
            "tIoU=%.3f MAE=%.1fs Segments=%d",
            match_id, result.precision, result.recall, result.f1,
            result.compression_ratio * 100, result.goal_coverage * 100,
            result.mean_tiou, result.goal_timing_mae, result.num_segments,
        )
        return result

    @staticmethod
    def aggregate_results(results: list[EvaluationResult]) -> dict:
        """Compute aggregate statistics across multiple matches."""
        if not results:
            return {}

        metrics = {
            "num_matches": len(results),
            "precision_mean": np.mean([r.precision for r in results]),
            "precision_std": np.std([r.precision for r in results]),
            "recall_mean": np.mean([r.recall for r in results]),
            "recall_std": np.std([r.recall for r in results]),
            "f1_mean": np.mean([r.f1 for r in results]),
            "f1_std": np.std([r.f1 for r in results]),
            "compression_mean": np.mean([r.compression_ratio for r in results]),
            "compression_std": np.std([r.compression_ratio for r in results]),
            "goal_coverage_mean": np.mean([r.goal_coverage for r in results]),
            "goal_coverage_std": np.std([r.goal_coverage for r in results]),
            "total_tp": sum(r.true_positives for r in results),
            "total_fp": sum(r.false_positives for r in results),
            "total_fn": sum(r.false_negatives for r in results),
        }

        # Micro-averaged P/R/F1
        tp, fp, fn = metrics["total_tp"], metrics["total_fp"], metrics["total_fn"]
        metrics["micro_precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics["micro_recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        p, r = metrics["micro_precision"], metrics["micro_recall"]
        metrics["micro_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        # Extended aggregate metrics
        metrics["mean_tiou_mean"] = float(np.mean([r.mean_tiou for r in results]))
        metrics["mean_tiou_std"] = float(np.std([r.mean_tiou for r in results]))
        metrics["goal_timing_mae_mean"] = float(np.mean([r.goal_timing_mae for r in results]))
        metrics["goal_timing_mae_std"] = float(np.std([r.goal_timing_mae for r in results]))
        metrics["num_segments_mean"] = float(np.mean([r.num_segments for r in results]))
        metrics["num_segments_std"] = float(np.std([r.num_segments for r in results]))
        metrics["mean_segment_duration_mean"] = float(np.mean([r.mean_segment_duration for r in results]))
        metrics["mean_segment_duration_std"] = float(np.std([r.mean_segment_duration for r in results]))

        # F1 at tIoU thresholds (average across matches)
        tiou_thresholds = ["0.1", "0.3", "0.5", "0.7"]
        for t in tiou_thresholds:
            vals = [r.f1_at_tiou.get(t, 0.0) for r in results if r.f1_at_tiou]
            if vals:
                metrics[f"f1_at_tiou_{t}_mean"] = float(np.mean(vals))
                metrics[f"f1_at_tiou_{t}_std"] = float(np.std(vals))

        # Bootstrap 95% CIs
        bootstrap_cis = Evaluator._bootstrap_confidence_intervals(results)
        metrics["bootstrap_ci"] = bootstrap_cis

        return metrics

    @staticmethod
    def _bootstrap_confidence_intervals(
        results: list[EvaluationResult],
        n_iterations: int = 1000,
        ci: float = 0.95,
    ) -> dict:
        """Compute bootstrap 95% CIs for key metrics."""
        if len(results) < 2:
            return {}

        rng = np.random.RandomState(42)
        n = len(results)
        alpha = (1 - ci) / 2

        metric_extractors = {
            "precision": lambda r: r.precision,
            "recall": lambda r: r.recall,
            "f1": lambda r: r.f1,
            "compression_ratio": lambda r: r.compression_ratio,
            "goal_coverage": lambda r: r.goal_coverage,
            "mean_tiou": lambda r: r.mean_tiou,
            "goal_timing_mae": lambda r: r.goal_timing_mae,
            "num_segments": lambda r: float(r.num_segments),
            "mean_segment_duration": lambda r: r.mean_segment_duration,
        }

        cis = {}
        for metric_name, extractor in metric_extractors.items():
            values = np.array([extractor(r) for r in results])
            bootstrap_means = np.array([
                np.mean(rng.choice(values, size=n, replace=True))
                for _ in range(n_iterations)
            ])
            lower = float(np.percentile(bootstrap_means, alpha * 100))
            upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))
            cis[metric_name] = {"lower": lower, "upper": upper, "mean": float(np.mean(values))}

        # Micro-averaged bootstrap
        tp_arr = np.array([r.true_positives for r in results])
        fp_arr = np.array([r.false_positives for r in results])
        fn_arr = np.array([r.false_negatives for r in results])

        micro_f1_boots = []
        for _ in range(n_iterations):
            idx = rng.choice(n, size=n, replace=True)
            tp_sum = tp_arr[idx].sum()
            fp_sum = fp_arr[idx].sum()
            fn_sum = fn_arr[idx].sum()
            p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
            r_val = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
            f1_val = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0
            micro_f1_boots.append(f1_val)

        micro_f1_boots = np.array(micro_f1_boots)
        cis["micro_f1"] = {
            "lower": float(np.percentile(micro_f1_boots, alpha * 100)),
            "upper": float(np.percentile(micro_f1_boots, (1 - alpha) * 100)),
            "mean": float(np.mean(micro_f1_boots)),
        }

        return cis
