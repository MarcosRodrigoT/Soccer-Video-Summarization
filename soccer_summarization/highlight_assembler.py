"""Highlight selection from shot boundaries and per-shot player counts.

A "highlight" is a long shot (duration >= min_shot_duration) whose average
per-frame player count is at least play_threshold.  Each highlight is one
full, unmodified shot — no intersection with frame-level play classification,
no merging across shot boundaries.

For goal moments, the dedicated select_goal_shots() picks one full play shot
per detected goal: the one whose end aligns with the OCR-detected goal time.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import HighlightAssemblerConfig
from .shot_segmentation import ShotBoundary
from .utils import format_time

logger = logging.getLogger(__name__)


@dataclass
class HighlightSegment:
    segment_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float


class HighlightAssembler:
    """Select highlight segments from shot boundaries and per-shot player counts."""

    def __init__(self, config: Optional[HighlightAssemblerConfig] = None):
        self.config = config or HighlightAssemblerConfig()

    def select_play_long_shots(
        self,
        shots: list[ShotBoundary],
        player_counts: pd.DataFrame,
        player_threshold: float = 5.0,
    ) -> list[HighlightSegment]:
        """Return long shots whose average per-frame player count >= threshold.

        A "long shot" is one with duration >= config.min_shot_duration.
        Each returned HighlightSegment matches one full shot exactly — no
        intersection, no merging.
        """
        min_dur = self.config.min_shot_duration
        out: list[HighlightSegment] = []
        n_long = 0

        for s in shots:
            if s.duration < min_dur:
                continue
            n_long += 1
            mask = (player_counts["frame"] >= s.start_frame) & \
                   (player_counts["frame"] <= s.end_frame)
            vals = player_counts.loc[mask, "player_count"]
            avg = float(vals.mean()) if len(vals) else 0.0
            if avg >= player_threshold:
                out.append(HighlightSegment(
                    segment_id=len(out) + 1,
                    start_frame=s.start_frame,
                    end_frame=s.end_frame,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    duration=s.duration,
                ))

        total_dur = sum(h.duration for h in out)
        logger.info(
            "Play long shots: %d / %d long shots (>=%.1fs), total %.1fs (%s)",
            len(out), n_long, min_dur, total_dur, format_time(total_dur),
        )
        return out

    def select_goal_shots(
        self,
        shots: list[ShotBoundary],
        player_counts: pd.DataFrame,
        goal_times: list[float],
        fps: int = 25,
        player_threshold: float = 5.0,
        max_lookback: float = 30.0,
        max_lookahead: float = 5.0,
    ) -> list[HighlightSegment]:
        """Select one full play shot per goal — the shot whose end aligns with the goal.

        For each goal time T, search for the play shot whose end_time is closest to T
        within the window [T - max_lookback, T + max_lookahead].  A "play shot" is one
        whose average per-frame player count is >= player_threshold (using the
        player_counts dataframe with 'frame' and 'player_count' columns).

        Falls back progressively if no clear match: containing-shot → nearest play
        shot → nearest shot of any kind.  Always returns one segment per goal.

        Returns a list of HighlightSegment with start/end equal to the chosen shot's
        full boundaries (no clipping, no merging).
        """
        if not goal_times or not shots:
            return []

        # Compute avg player count per shot
        pc = player_counts.copy()
        shot_avg: list[float] = []
        for s in shots:
            mask = (pc["frame"] >= s.start_frame) & (pc["frame"] <= s.end_frame)
            vals = pc.loc[mask, "player_count"]
            shot_avg.append(float(vals.mean()) if len(vals) else 0.0)

        is_play = [a >= player_threshold for a in shot_avg]
        play_shot_indices = [i for i, p in enumerate(is_play) if p]

        result: list[HighlightSegment] = []

        for g_idx, T in enumerate(goal_times):
            chosen: Optional[ShotBoundary] = None

            # Pass 1: play shot ending near T
            candidates = [
                i for i in play_shot_indices
                if (T - max_lookback) <= shots[i].end_time <= (T + max_lookahead)
            ]
            if candidates:
                chosen_i = min(candidates, key=lambda i: abs(shots[i].end_time - T))
                chosen = shots[chosen_i]
            else:
                # Pass 2: play shot CONTAINING T (goal mid-action)
                containing = [
                    i for i in play_shot_indices
                    if shots[i].start_time <= T <= shots[i].end_time
                ]
                if containing:
                    chosen = shots[containing[0]]
                else:
                    # Pass 3: nearest play shot in either direction (no window limit)
                    if play_shot_indices:
                        chosen_i = min(
                            play_shot_indices,
                            key=lambda i: abs(shots[i].end_time - T),
                        )
                        chosen = shots[chosen_i]
                    else:
                        # Pass 4 (degenerate): nearest shot of any kind
                        chosen_i = min(
                            range(len(shots)),
                            key=lambda i: abs(shots[i].end_time - T),
                        )
                        chosen = shots[chosen_i]

            assert chosen is not None
            result.append(HighlightSegment(
                segment_id=g_idx + 1,
                start_frame=chosen.start_frame,
                end_frame=chosen.end_frame,
                start_time=chosen.start_time,
                end_time=chosen.end_time,
                duration=chosen.duration,
            ))

        logger.info(
            "Selected %d goal shots from %d shots (%d play shots, threshold %.1f)",
            len(result), len(shots), len(play_shot_indices), player_threshold,
        )
        return result

    @staticmethod
    def export_video(
        highlights: list[HighlightSegment],
        input_video_path: str,
        output_video_path: str,
    ) -> None:
        """Export highlight segments as a single concatenated video."""
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        frames_written = 0
        for seg in highlights:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(seg.start_frame))
            for _ in range(int(seg.start_frame), int(seg.end_frame) + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1

        cap.release()
        out.release()
        logger.info("Exported %d frames to %s", frames_written, output_video_path)

    @staticmethod
    def highlights_to_dataframe(highlights: list[HighlightSegment]) -> pd.DataFrame:
        """Convert highlights to DataFrame."""
        return pd.DataFrame([
            {
                "segment_id": h.segment_id,
                "start_frame": h.start_frame,
                "end_frame": h.end_frame,
                "start_time": h.start_time,
                "end_time": h.end_time,
                "duration": h.duration,
                "start_formatted": format_time(h.start_time),
                "end_formatted": format_time(h.end_time),
            }
            for h in highlights
        ])
