"""Automatic scoreboard localization and OCR-based score tracking.

Uses pixel variance analysis to find static broadcast overlays (scoreboard,
logos, clock), then PaddleOCR to read score values and detect goal events.

The key insight: the scoreboard is a persistent overlay that stays in the
same position with nearly identical pixels across most of the match, while
the game action underneath changes constantly. Low per-pixel variance
identifies these static regions.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import OCRScoreboardConfig
from .utils import get_video_properties

logger = logging.getLogger(__name__)


@dataclass
class ScoreEvent:
    """A detected score change (goal event)."""
    timestamp: float  # seconds
    frame: int
    home_score: int
    away_score: int
    team: str  # "home" or "away"


@dataclass
class ScoreboardRegion:
    """Detected scoreboard bounding box."""
    x: int
    y: int
    w: int
    h: int


class ScoreboardDetector:
    """Detect and read broadcast scoreboards via variance analysis + OCR."""

    def __init__(self, config: Optional[OCRScoreboardConfig] = None):
        self.config = config or OCRScoreboardConfig()
        self._ocr = None

    def _get_ocr(self):
        """Lazy-load EasyOCR reader."""
        if self._ocr is None:
            import easyocr
            self._ocr = easyocr.Reader(["en"], gpu=True, verbose=False)
        return self._ocr

    def find_static_overlays(self, video_path: str) -> np.ndarray:
        """Build a variance map to identify static overlay regions.

        Samples frames evenly across the video, computes per-pixel variance
        in grayscale. Low-variance regions are static overlays.

        Returns:
            variance_map: 2D array of per-pixel variance values
        """
        props = get_video_properties(video_path)
        total_frames = props["total_frames"]
        n_samples = self.config.num_sample_frames

        cap = cv2.VideoCapture(video_path)
        sample_indices = np.linspace(0, total_frames - 1, n_samples, dtype=int)

        frames_gray = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                frames_gray.append(gray)
        cap.release()

        if len(frames_gray) < 10:
            raise ValueError(f"Could only read {len(frames_gray)} frames, need at least 10")

        stack = np.stack(frames_gray, axis=0)
        variance_map = np.var(stack, axis=0)

        logger.info(
            "Variance map: shape=%s, range=[%.1f, %.1f], mean=%.1f",
            variance_map.shape, variance_map.min(), variance_map.max(), variance_map.mean(),
        )
        return variance_map

    def find_scoreboard_region(
        self, video_path: str, variance_map: np.ndarray
    ) -> Optional[ScoreboardRegion]:
        """Find the scoreboard among static overlay regions.

        1. Threshold variance map to get low-variance (static) regions
        2. Extract connected components
        3. Run OCR on each candidate to find score pattern
        """
        # Threshold: low variance = static overlay
        threshold = np.percentile(variance_map, self.config.variance_threshold_percentile)
        binary = (variance_map < threshold).astype(np.uint8) * 255

        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        # Filter candidates by area
        candidates = []
        for i in range(1, num_labels):  # skip background
            area = stats[i, cv2.CC_STAT_AREA]
            if self.config.min_region_area <= area <= self.config.max_region_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                candidates.append(ScoreboardRegion(x=x, y=y, w=w, h=h))

        logger.info("Found %d candidate overlay regions", len(candidates))

        if not candidates:
            return None

        # Read a frame to test OCR on each candidate
        cap = cv2.VideoCapture(video_path)
        props = get_video_properties(video_path)
        # Sample a frame from middle of the match
        cap.set(cv2.CAP_PROP_POS_FRAMES, props["total_frames"] // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        ocr = self._get_ocr()
        score_pattern = re.compile(self.config.score_pattern)

        for region in candidates:
            # Crop region with some padding
            pad = 10
            y1 = max(0, region.y - pad)
            y2 = min(frame.shape[0], region.y + region.h + pad)
            x1 = max(0, region.x - pad)
            x2 = min(frame.shape[1], region.x + region.w + pad)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # Run OCR (EasyOCR returns [(bbox, text, conf), ...])
            results = ocr.readtext(crop)
            if not results:
                continue

            # Check each detected text for score pattern
            for (bbox, text, conf) in results:
                if score_pattern.search(text):
                    logger.info(
                        "Scoreboard found at (%d,%d,%d,%d): '%s' (conf=%.2f)",
                        region.x, region.y, region.w, region.h, text, conf,
                    )
                    return region

        logger.warning("No scoreboard region matched score pattern among %d candidates", len(candidates))
        return None

    def track_scores(
        self,
        video_path: str,
        region: ScoreboardRegion,
    ) -> list[ScoreEvent]:
        """Track score changes over time by sampling frames and running OCR.

        Returns list of ScoreEvent (goal detections).
        """
        props = get_video_properties(video_path)
        fps = props["fps"]
        total_frames = props["total_frames"]
        interval_frames = int(self.config.score_sample_interval * fps)

        ocr = self._get_ocr()
        score_pattern = re.compile(self.config.score_pattern)

        cap = cv2.VideoCapture(video_path)
        pad = 10

        raw_scores = []  # (frame, home, away)
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Crop scoreboard region
            y1 = max(0, region.y - pad)
            y2 = min(frame.shape[0], region.y + region.h + pad)
            x1 = max(0, region.x - pad)
            x2 = min(frame.shape[1], region.x + region.w + pad)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                frame_idx += interval_frames
                continue

            results = ocr.readtext(crop)
            if results:
                for (bbox, text, conf) in results:
                    match = score_pattern.search(text)
                    if match:
                        score_text = match.group()
                        parts = re.split(r"[-:]", score_text)
                        if len(parts) == 2:
                            try:
                                home = int(parts[0].strip())
                                away = int(parts[1].strip())
                                raw_scores.append((frame_idx, home, away))
                            except ValueError:
                                pass
                        break

            frame_idx += interval_frames

        cap.release()

        if not raw_scores:
            logger.warning("No scores detected via OCR")
            return []

        # Apply temporal majority voting and detect changes
        events = self._detect_score_changes(raw_scores, fps)
        logger.info("Detected %d score change events", len(events))
        return events

    def _detect_score_changes(
        self, raw_scores: list[tuple[int, int, int]], fps: int
    ) -> list[ScoreEvent]:
        """Apply majority voting and detect when score changes.

        Validates that score only increases by 1 at a time.
        """
        if not raw_scores:
            return []

        window = self.config.majority_window

        # Apply majority voting for each reading
        stable_scores = []
        for i in range(len(raw_scores)):
            start = max(0, i - window // 2)
            end = min(len(raw_scores), i + window // 2 + 1)
            window_scores = raw_scores[start:end]

            # Most common (home, away) pair in window
            from collections import Counter
            counts = Counter((h, a) for _, h, a in window_scores)
            (home, away), _ = counts.most_common(1)[0]
            stable_scores.append((raw_scores[i][0], home, away))

        # Detect changes
        events = []
        prev_home, prev_away = stable_scores[0][1], stable_scores[0][2]

        for frame_idx, home, away in stable_scores[1:]:
            if home != prev_home or away != prev_away:
                # Validate: score should increase by exactly 1
                home_diff = home - prev_home
                away_diff = away - prev_away

                if (home_diff == 1 and away_diff == 0) or (home_diff == 0 and away_diff == 1):
                    team = "home" if home_diff == 1 else "away"
                    events.append(ScoreEvent(
                        timestamp=frame_idx / fps,
                        frame=frame_idx,
                        home_score=home,
                        away_score=away,
                        team=team,
                    ))
                    prev_home, prev_away = home, away
                elif home_diff >= 0 and away_diff >= 0:
                    # Multiple goals somehow - still record
                    for _ in range(home_diff):
                        events.append(ScoreEvent(
                            timestamp=frame_idx / fps,
                            frame=frame_idx,
                            home_score=home,
                            away_score=away,
                            team="home",
                        ))
                    for _ in range(away_diff):
                        events.append(ScoreEvent(
                            timestamp=frame_idx / fps,
                            frame=frame_idx,
                            home_score=home,
                            away_score=away,
                            team="away",
                        ))
                    prev_home, prev_away = home, away
                # else: likely OCR error, ignore

        return events

    def run(self, video_path: str) -> tuple[Optional[ScoreboardRegion], list[ScoreEvent]]:
        """Full pipeline: find scoreboard, track scores, detect goals.

        Returns:
            region: Detected scoreboard region (or None)
            events: List of score change events
        """
        logger.info("Building variance map...")
        variance_map = self.find_static_overlays(video_path)

        # Save variance map for debugging
        logger.info("Finding scoreboard region...")
        region = self.find_scoreboard_region(video_path, variance_map)

        if region is None:
            logger.warning("Could not locate scoreboard")
            return None, []

        logger.info("Tracking scores...")
        events = self.track_scores(video_path, region)

        return region, events

    @staticmethod
    def events_to_dataframe(events: list[ScoreEvent]) -> pd.DataFrame:
        """Convert score events to DataFrame."""
        return pd.DataFrame([
            {
                "timestamp": e.timestamp,
                "frame": e.frame,
                "home_score": e.home_score,
                "away_score": e.away_score,
                "team": e.team,
            }
            for e in events
        ])
