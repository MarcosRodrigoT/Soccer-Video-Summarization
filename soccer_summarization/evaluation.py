"""Evaluation metrics for goal detection and highlight quality.

Bridges the gap between system output (highlight segments with optional goal
timestamps) and ground truth (commentary-level goal annotations in CSV).
"""

import logging
from dataclasses import dataclass
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

        return {
            "total_video_duration": total_video_duration,
            "highlight_duration": highlight_duration,
            "compression_ratio": compression_ratio,
            "goal_coverage": goal_coverage,
        }

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

        result = EvaluationResult(
            match_id=match_id,
            **goal_metrics,
            **quality_metrics,
        )

        logger.info(
            "Match %s: P=%.2f R=%.2f F1=%.2f | Compression=%.1f%% Coverage=%.1f%%",
            match_id, result.precision, result.recall, result.f1,
            result.compression_ratio * 100, result.goal_coverage * 100,
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

        return metrics
