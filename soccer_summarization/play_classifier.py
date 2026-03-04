"""Play/no-play classification based on smoothed player count.

Classifies each frame as "play" (active gameplay) or "no-play" (replays,
close-ups, transitions) based on whether the smoothed player count exceeds
a threshold. Uses a rolling average for temporal smoothing.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .config import PlayClassifierConfig
from .shot_segmentation import ShotBoundary

logger = logging.getLogger(__name__)


@dataclass
class PlaySegment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float
    is_play: bool
    avg_player_count: float


class PlayClassifier:
    """Classify frames as play/no-play based on player count."""

    def __init__(self, config: Optional[PlayClassifierConfig] = None):
        self.config = config or PlayClassifierConfig()

    def classify(
        self,
        player_counts: pd.DataFrame,
        video_fps: int = 25,
    ) -> tuple[list[PlaySegment], pd.DataFrame]:
        """Classify each frame as play or no-play.

        Args:
            player_counts: DataFrame with columns [frame, time_seconds, player_count]
            video_fps: Video frame rate (for interpolating to all frames)

        Returns:
            segments: List of PlaySegment objects (contiguous play/no-play regions)
            frame_df: DataFrame with per-frame classification including smoothed counts
        """
        df = player_counts.copy()

        # Interpolate to fill gaps (since YOLO processes every Nth frame)
        if len(df) > 1:
            frame_step = df["frame"].iloc[1] - df["frame"].iloc[0]
            if frame_step > 1:
                max_frame = df["frame"].max()
                all_frames = pd.DataFrame({"frame": range(0, max_frame + 1)})
                df = all_frames.merge(df, on="frame", how="left")
                df["player_count"] = df["player_count"].interpolate(method="linear").fillna(0)
                df["time_seconds"] = df["frame"] / video_fps

        # Apply rolling average smoothing
        window = self.config.smoothing_window
        df["player_count_smooth"] = (
            df["player_count"]
            .rolling(window=window, center=True, min_periods=1)
            .mean()
        )

        # Classify: play if smoothed count >= threshold
        threshold = self.config.play_threshold
        df["is_play"] = df["player_count_smooth"] >= threshold

        logger.info(
            "Classification: %.1f%% play frames (threshold=%d, window=%d)",
            df["is_play"].mean() * 100, threshold, window,
        )

        # Extract contiguous segments
        segments = self._extract_segments(df, video_fps)

        logger.info(
            "Found %d segments (%d play, %d no-play)",
            len(segments),
            sum(1 for s in segments if s.is_play),
            sum(1 for s in segments if not s.is_play),
        )

        return segments, df

    def _extract_segments(self, df: pd.DataFrame, fps: int) -> list[PlaySegment]:
        """Extract contiguous play/no-play segments from classified frames."""
        if len(df) == 0:
            return []

        segments = []
        current_is_play = bool(df.iloc[0]["is_play"])
        segment_start = 0

        for i in range(1, len(df)):
            frame_is_play = bool(df.iloc[i]["is_play"])
            if frame_is_play != current_is_play:
                # Segment boundary
                segment_df = df.iloc[segment_start:i]
                segments.append(PlaySegment(
                    start_frame=int(segment_df.iloc[0]["frame"]),
                    end_frame=int(segment_df.iloc[-1]["frame"]),
                    start_time=float(segment_df.iloc[0]["frame"]) / fps,
                    end_time=float(segment_df.iloc[-1]["frame"]) / fps,
                    duration=(float(segment_df.iloc[-1]["frame"]) - float(segment_df.iloc[0]["frame"])) / fps,
                    is_play=current_is_play,
                    avg_player_count=float(segment_df["player_count_smooth"].mean()),
                ))
                segment_start = i
                current_is_play = frame_is_play

        # Final segment
        segment_df = df.iloc[segment_start:]
        segments.append(PlaySegment(
            start_frame=int(segment_df.iloc[0]["frame"]),
            end_frame=int(segment_df.iloc[-1]["frame"]),
            start_time=float(segment_df.iloc[0]["frame"]) / fps,
            end_time=float(segment_df.iloc[-1]["frame"]) / fps,
            duration=(float(segment_df.iloc[-1]["frame"]) - float(segment_df.iloc[0]["frame"])) / fps,
            is_play=current_is_play,
            avg_player_count=float(segment_df["player_count_smooth"].mean()),
        ))

        return segments

    @staticmethod
    def segments_to_dataframe(segments: list[PlaySegment]) -> pd.DataFrame:
        """Convert segment list to DataFrame."""
        return pd.DataFrame([
            {
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
                "is_play": s.is_play,
                "avg_player_count": s.avg_player_count,
            }
            for s in segments
        ])
