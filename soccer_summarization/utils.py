"""Utility functions for time formatting and I/O helpers."""

import os
import logging
import time
from pathlib import Path

import cv2
import pandas as pd


logger = logging.getLogger(__name__)


# ─── Per-stage compute-time tracking ─────────────────────────────────────────

class StageTimer:
    """Records wall-clock duration of named pipeline stages.

    Use as a context manager::

        timer = StageTimer()
        with timer.time("shot_segmentation"):
            shots = segmenter.run(video_path)

    Each stage records its `duration_sec` and a `cached` flag.  Cached stages
    represent file-load time only (typically negligible); their `duration_sec`
    is recorded but should be excluded when reporting compute time, since the
    actual compute happened on a prior run.
    """

    def __init__(self):
        self.stages: dict[str, dict] = {}

    def time(self, name: str, cached: bool = False):
        return _StageTimer(self, name, cached)

    def record(self, name: str, duration_sec: float, cached: bool = False):
        self.stages[name] = {"duration_sec": float(duration_sec), "cached": bool(cached)}

    def to_dict(self) -> dict:
        return dict(self.stages)


class _StageTimer:
    def __init__(self, parent: StageTimer, name: str, cached: bool):
        self.parent = parent
        self.name = name
        self.cached = cached

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.parent.record(self.name, time.perf_counter() - self._t0, self.cached)
        return False


def format_time(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_time_detailed(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.ms format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes:02d}:{secs:05.2f}"


def get_video_properties(video_path: str) -> dict:
    """Get video properties (fps, width, height, total_frames, duration)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    props = {
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    props["duration"] = props["total_frames"] / props["fps"] if props["fps"] > 0 else 0
    cap.release()
    return props


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_csv(path: str, **kwargs) -> pd.DataFrame:
    """Load CSV with common error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, **kwargs)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the package."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
