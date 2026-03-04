"""Shot boundary detection via frame differencing with adaptive threshold.

Detects camera cuts (shot boundaries) by computing the mean absolute difference
between consecutive grayscale frames. Uses an adaptive threshold:
    tau = max(fixed_threshold, mean + k * std)
with local maximum detection and minimum shot length constraint.
"""

import logging
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import ShotSegmentationConfig
from .utils import get_video_properties

logger = logging.getLogger(__name__)


@dataclass
class ShotBoundary:
    shot_id: int
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    duration: float


class ShotSegmenter:
    """Detect shot boundaries in a video using frame differencing."""

    def __init__(self, config: Optional[ShotSegmentationConfig] = None):
        self.config = config or ShotSegmentationConfig()

    @staticmethod
    def _process_frame_batch(args: tuple) -> list[tuple[int, float]]:
        """Process a batch of frames in a separate process.

        Returns list of (frame_index, mean_abs_difference) tuples.
        """
        start_frame, end_frame, video_path, total_frames = args

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        differences = []
        prev_gray = None

        for i in range(start_frame, min(end_frame, total_frames - 1)):
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                differences.append((i, float(np.mean(diff))))

            prev_gray = gray

        cap.release()
        return differences

    def compute_frame_differences(self, video_path: str) -> np.ndarray:
        """Compute per-frame differences using multiprocessing."""
        props = get_video_properties(video_path)
        total_frames = props["total_frames"]
        num_processes = self.config.num_processes or cpu_count()

        batch_size = total_frames // num_processes
        process_args = []
        for i in range(num_processes):
            start = i * batch_size
            end = (i + 1) * batch_size if i < num_processes - 1 else total_frames
            process_args.append((start, end, video_path, total_frames))

        logger.info("Computing frame differences with %d processes...", num_processes)
        with Pool(num_processes) as pool:
            results = pool.map(ShotSegmenter._process_frame_batch, process_args)

        all_diffs = []
        for batch in results:
            all_diffs.extend(batch)
        all_diffs.sort(key=lambda x: x[0])

        return np.array([d for _, d in all_diffs])

    def detect_boundaries(self, differences: np.ndarray) -> list[int]:
        """Detect shot boundary frames from difference signal.

        Uses adaptive threshold: tau = max(fixed, mean + k * std)
        with local maximum detection and minimum shot length constraint.
        """
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        adaptive_threshold = mean_diff + self.config.adaptive_k * std_diff
        threshold = max(self.config.threshold, adaptive_threshold)

        logger.info(
            "Threshold: %.2f (fixed=%.1f, adaptive=%.2f, mean=%.2f, std=%.2f)",
            threshold, self.config.threshold, adaptive_threshold, mean_diff, std_diff,
        )

        boundaries = []
        for i in range(1, len(differences) - 1):
            if (
                differences[i] > threshold
                and differences[i] > differences[i - 1]
                and differences[i] > differences[i + 1]
            ):
                if not boundaries or i - boundaries[-1] >= self.config.min_shot_length_frames:
                    boundaries.append(i)

        return boundaries

    def run(self, video_path: str) -> tuple[list[ShotBoundary], np.ndarray]:
        """Run shot segmentation on a video.

        Returns:
            shots: List of ShotBoundary objects
            differences: Raw frame difference signal (for visualization)
        """
        props = get_video_properties(video_path)
        fps = props["fps"]
        total_frames = props["total_frames"]

        differences = self.compute_frame_differences(video_path)
        boundary_frames = self.detect_boundaries(differences)

        # Convert boundary frames to shot segments
        shots = []
        all_boundaries = [0] + boundary_frames + [total_frames - 1]

        for i in range(len(all_boundaries) - 1):
            start_frame = all_boundaries[i]
            end_frame = all_boundaries[i + 1]
            shots.append(ShotBoundary(
                shot_id=i + 1,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_frame / fps,
                end_time=end_frame / fps,
                duration=(end_frame - start_frame) / fps,
            ))

        logger.info("Detected %d shots (%d boundaries)", len(shots), len(boundary_frames))
        return shots, differences

    @staticmethod
    def shots_to_dataframe(shots: list[ShotBoundary]) -> pd.DataFrame:
        """Convert shot list to a DataFrame."""
        return pd.DataFrame([
            {
                "shot_id": s.shot_id,
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_time": s.start_time,
                "end_time": s.end_time,
                "duration": s.duration,
            }
            for s in shots
        ])
