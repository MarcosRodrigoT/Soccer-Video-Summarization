"""Highlight assembly: intersection of long shots and play moments.

Takes shot boundaries and play/no-play segments, filters short shots,
computes the intersection (long shot AND play moment), merges adjacent
segments, and exports the final highlight video.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import HighlightAssemblerConfig
from .play_classifier import PlaySegment
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
    """Combine shot boundaries and play classification into highlight segments."""

    def __init__(self, config: Optional[HighlightAssemblerConfig] = None):
        self.config = config or HighlightAssemblerConfig()

    def assemble(
        self,
        shots: list[ShotBoundary],
        play_segments: list[PlaySegment],
        fps: int = 25,
    ) -> list[HighlightSegment]:
        """Compute intersection of long shots and play moments.

        1. Filter shots by minimum duration
        2. Compute intersection with play segments
        3. Merge adjacent segments

        Returns list of HighlightSegment.
        """
        # Step 1: Filter short shots
        min_dur = self.config.min_shot_duration
        long_shots = [s for s in shots if s.duration >= min_dur]
        logger.info(
            "Shot filtering: %d/%d shots >= %.1fs",
            len(long_shots), len(shots), min_dur,
        )

        # Step 2: Get play-only segments
        play_only = [s for s in play_segments if s.is_play]
        logger.info("Play segments: %d", len(play_only))

        # Step 3: Compute intersection
        intersections = []
        for shot in long_shots:
            for play in play_only:
                overlap_start = max(shot.start_frame, play.start_frame)
                overlap_end = min(shot.end_frame, play.end_frame)

                if overlap_start < overlap_end:
                    intersections.append({
                        "start_frame": overlap_start,
                        "end_frame": overlap_end,
                        "start_time": overlap_start / fps,
                        "end_time": overlap_end / fps,
                        "duration": (overlap_end - overlap_start) / fps,
                    })

        if not intersections:
            logger.warning("No intersection segments found!")
            return []

        # Sort by start frame
        intersections.sort(key=lambda x: x["start_frame"])

        # Step 4: Merge adjacent segments
        merged = self._merge_adjacent(intersections, fps)

        highlights = [
            HighlightSegment(
                segment_id=i + 1,
                start_frame=seg["start_frame"],
                end_frame=seg["end_frame"],
                start_time=seg["start_time"],
                end_time=seg["end_time"],
                duration=seg["duration"],
            )
            for i, seg in enumerate(merged)
        ]

        total_dur = sum(h.duration for h in highlights)
        logger.info(
            "Highlights: %d segments, total %.1fs (%s)",
            len(highlights), total_dur, format_time(total_dur),
        )

        return highlights

    def _merge_adjacent(self, segments: list[dict], fps: int) -> list[dict]:
        """Merge segments with gap <= threshold."""
        if not segments:
            return []

        gap_threshold = self.config.merge_gap_threshold
        merged = [segments[0].copy()]

        for seg in segments[1:]:
            gap = (seg["start_frame"] - merged[-1]["end_frame"]) / fps
            if gap <= gap_threshold:
                merged[-1]["end_frame"] = seg["end_frame"]
                merged[-1]["end_time"] = seg["end_time"]
                merged[-1]["duration"] = merged[-1]["end_time"] - merged[-1]["start_time"]
            else:
                merged.append(seg.copy())

        if len(merged) < len(segments):
            logger.info("Merged %d -> %d segments", len(segments), len(merged))

        return merged

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
