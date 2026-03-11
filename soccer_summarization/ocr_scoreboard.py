"""Automatic scoreboard localization and OCR-based score tracking.

Uses pixel variance analysis to find static broadcast overlays (scoreboard,
logos, clock), then EasyOCR to read score values and detect goal events.

The key insight: the scoreboard is a persistent overlay that stays in the
same position with nearly identical pixels across most of the match, while
the game action underneath changes constantly. Low per-pixel variance
identifies these static regions.

Handles multiple La Liga broadcast formats:
- Classic: "TEAM1 X-Y TEAM2" (2013-2018 era)
- Modern: "TEAM1 X  Y TEAM2" or vertical stacking (2020+ era)

Uses temporal stability to distinguish score digits from clock digits:
the clock changes every sample while scores change only at goal events.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .config import OCRScoreboardConfig
from .utils import get_video_properties

logger = logging.getLogger(__name__)

# Known La Liga team abbreviations for scoreboard validation
TEAM_PATTERNS = re.compile(
    r"\b(BAR|RMA|ATM|SEV|BET|VAL|VIL|CEL|SOC|ATH|GIR|ALM|CAD|GET|GRA|"
    r"MAL|MAJ|OSA|RAY|ELC|ESP|LPA|VAL|FCB|RM|ATL|VCF|RCD|RCE|LEV|"
    r"LALIGA|LA LIGA)\b",
    re.IGNORECASE,
)

# Clock pattern: MM:SS or MM.SS
CLOCK_PATTERN = re.compile(r"\b(\d{1,3})[:.]\s*(\d{2})\b")

# Score with separator: "X - Y" or "X : Y"
SCORE_WITH_SEP = re.compile(r"\b(\d{1,2})\s*[-]\s*(\d{1,2})\b")


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

        Strategy:
        1. Restrict to top 30% of frame (scoreboards are always at top)
        2. Threshold variance map to get low-variance (static) regions
        3. Merge nearby static regions into scoreboard candidates
        4. Test OCR on multiple frames to find regions with text
        5. Score candidates by presence of team names/digits
        """
        h, w = variance_map.shape
        top_fraction = 0.30
        top_h = int(h * top_fraction)

        # Work only on top portion
        var_top = variance_map[:top_h, :]

        # Use adaptive thresholding — try increasing percentiles until we get candidates
        for percentile in [10.0, 15.0, 20.0, 25.0]:
            threshold = np.percentile(var_top, percentile)
            binary = (var_top < threshold).astype(np.uint8) * 255

            # Gentle morphological operations to clean noise without over-merging
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )

            # Filter candidates by area
            min_area = 100
            max_area = top_h * w * 0.4
            candidates = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if min_area <= area <= max_area:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    cw = stats[i, cv2.CC_STAT_WIDTH]
                    ch = stats[i, cv2.CC_STAT_HEIGHT]
                    candidates.append(ScoreboardRegion(x=x, y=y, w=cw, h=ch))

            if len(candidates) >= 2:
                logger.info(
                    "Found %d candidate regions at percentile %.0f",
                    len(candidates), percentile,
                )
                break
        else:
            logger.warning("No candidate regions found at any percentile")
            return None

        # Merge overlapping/nearby candidates into a single scoreboard region
        merged = self._merge_regions(candidates)
        logger.info("Merged into %d candidate regions", len(merged))

        # Test OCR on candidates using multiple frames
        props = get_video_properties(video_path)
        cap = cv2.VideoCapture(video_path)

        # Sample 3 frames from different parts of the match
        test_frames = []
        for frac in [0.25, 0.5, 0.75]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(props["total_frames"] * frac))
            ret, frame = cap.read()
            if ret:
                test_frames.append(frame)
        cap.release()

        if not test_frames:
            return None

        ocr = self._get_ocr()
        best_region = None
        best_score = -1

        for region in merged:
            region_score = 0
            for frame in test_frames:
                crop = self._crop_region(frame, region, pad=15)
                if crop.size == 0:
                    continue

                # Upscale 3x for better OCR on small text
                upscaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                results = ocr.readtext(upscaled)

                for bbox, text, conf in results:
                    text_upper = text.upper().strip()
                    # Score points for team names
                    if TEAM_PATTERNS.search(text_upper):
                        region_score += 3
                    # Score points for any digit (potential score)
                    if re.search(r"\d", text):
                        region_score += 1
                    # Score points for clock pattern (indicates broadcast overlay)
                    if CLOCK_PATTERN.search(text):
                        region_score += 2

            if region_score > best_score:
                best_score = region_score
                best_region = region

        if best_region and best_score >= 3:
            logger.info(
                "Scoreboard at (%d,%d,%d,%d) score=%d",
                best_region.x, best_region.y, best_region.w, best_region.h, best_score,
            )
            return best_region

        # Fallback: if no region scored well, use the largest region in top-left
        # quadrant (broadcast scoreboards are typically top-left)
        top_left = [r for r in merged if r.x < w // 2]
        if top_left:
            largest = max(top_left, key=lambda r: r.w * r.h)
            logger.info(
                "Fallback: using largest top-left region (%d,%d,%d,%d)",
                largest.x, largest.y, largest.w, largest.h,
            )
            return largest

        if merged:
            largest = max(merged, key=lambda r: r.w * r.h)
            logger.info(
                "Fallback: using largest region (%d,%d,%d,%d)",
                largest.x, largest.y, largest.w, largest.h,
            )
            return largest

        logger.warning("No scoreboard region found")
        return None

    def _merge_regions(self, regions: list[ScoreboardRegion]) -> list[ScoreboardRegion]:
        """Merge overlapping or nearby regions into larger candidates."""
        if not regions:
            return []

        # Convert to (x1, y1, x2, y2) format
        boxes = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in regions]
        merged = []
        used = [False] * len(boxes)
        gap = 20  # merge regions within 20px of each other

        for i in range(len(boxes)):
            if used[i]:
                continue
            x1, y1, x2, y2 = boxes[i]
            used[i] = True
            changed = True
            while changed:
                changed = False
                for j in range(len(boxes)):
                    if used[j]:
                        continue
                    jx1, jy1, jx2, jy2 = boxes[j]
                    # Check if regions are close enough to merge
                    if (jx1 <= x2 + gap and jx2 >= x1 - gap and
                            jy1 <= y2 + gap and jy2 >= y1 - gap):
                        x1 = min(x1, jx1)
                        y1 = min(y1, jy1)
                        x2 = max(x2, jx2)
                        y2 = max(y2, jy2)
                        used[j] = True
                        changed = True
            merged.append(ScoreboardRegion(x=x1, y=y1, w=x2 - x1, h=y2 - y1))

        return merged

    def _crop_region(
        self, frame: np.ndarray, region: ScoreboardRegion, pad: int = 10
    ) -> np.ndarray:
        """Crop a region from a frame with padding."""
        y1 = max(0, region.y - pad)
        y2 = min(frame.shape[0], region.y + region.h + pad)
        x1 = max(0, region.x - pad)
        x2 = min(frame.shape[1], region.x + region.w + pad)
        return frame[y1:y2, x1:x2]

    def track_scores(
        self,
        video_path: str,
        region: ScoreboardRegion,
    ) -> list[ScoreEvent]:
        """Track score changes over time by sampling frames and running OCR.

        Uses a two-phase approach:
        1. Collect all OCR readings with spatial information
        2. Separate clock readings from score readings using temporal stability
        3. Detect score changes from stable readings

        Returns list of ScoreEvent (goal detections).
        """
        props = get_video_properties(video_path)
        fps = props["fps"]
        total_frames = props["total_frames"]
        interval_frames = int(self.config.score_sample_interval * fps)

        ocr = self._get_ocr()
        cap = cv2.VideoCapture(video_path)

        # Phase 1: Collect all OCR readings with position info
        all_readings = []  # (frame_idx, text, bbox_center_x, bbox_center_y, conf)
        frame_idx = 0

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            crop = self._crop_region(frame, region, pad=15)
            if crop.size == 0:
                frame_idx += interval_frames
                continue

            # Upscale for better OCR
            upscaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(upscaled)

            for bbox, text, conf in results:
                # Compute center of bbox (in upscaled coordinates, convert back)
                cx = (bbox[0][0] + bbox[2][0]) / 2 / 3
                cy = (bbox[0][1] + bbox[2][1]) / 2 / 3
                all_readings.append((frame_idx, text.strip(), float(cx), float(cy), float(conf)))

            frame_idx += interval_frames

        cap.release()

        if not all_readings:
            logger.warning("No OCR readings collected")
            return []

        logger.info("Collected %d raw OCR readings", len(all_readings))

        # Phase 2: Parse scores using multiple strategies
        raw_scores = self._parse_scores_multi_strategy(all_readings, fps)

        if not raw_scores:
            logger.warning("No valid scores parsed from OCR readings")
            return []

        logger.info("Parsed %d raw score readings", len(raw_scores))

        # Phase 3: Apply temporal filtering and detect changes
        events = self._detect_score_changes(raw_scores, fps)
        logger.info("Detected %d score change events", len(events))
        return events

    def _parse_scores_multi_strategy(
        self, readings: list[tuple], fps: int
    ) -> list[tuple[int, int, int]]:
        """Parse score readings using multiple strategies.

        Strategy 1: Find "X-Y" pattern (classic format with separator)
        Strategy 2: Position-cluster tracking (modern format with separate digits)

        Returns: list of (frame_idx, home_score, away_score)
        """
        # Group readings by frame
        by_frame = {}
        for frame_idx, text, cx, cy, conf in readings:
            if frame_idx not in by_frame:
                by_frame[frame_idx] = []
            by_frame[frame_idx].append((text, cx, cy, conf))

        # Strategy 1: Try "X-Y" or "X - Y" pattern (classic broadcast format)
        strategy1_scores = []
        for frame_idx, frame_readings in sorted(by_frame.items()):
            for text, cx, cy, conf in frame_readings:
                m = SCORE_WITH_SEP.search(text)
                if m:
                    home = int(m.group(1))
                    away = int(m.group(2))
                    # Filter out clock readings (values > 12 are likely clock minutes)
                    if home <= 12 and away <= 12:
                        strategy1_scores.append((frame_idx, home, away))

        if len(strategy1_scores) >= 10:
            logger.info("Strategy 1 (X-Y pattern): %d readings", len(strategy1_scores))
            return strategy1_scores

        # Strategy 2: Position-cluster tracking
        # Collect ALL single-digit readings across ALL frames, cluster by 2D
        # position, then use the two most prominent clusters as home/away scores.
        # This works even when only 1 digit is detected per frame (common in
        # modern broadcast formats where EasyOCR misses the second digit).
        strategy2_scores = self._parse_by_position_clusters(by_frame)
        if len(strategy2_scores) >= 10:
            logger.info("Strategy 2 (position clusters): %d readings", len(strategy2_scores))
            return strategy2_scores

        return strategy1_scores or strategy2_scores or []

    def _parse_by_position_clusters(
        self, by_frame: dict[int, list[tuple]]
    ) -> list[tuple[int, int, int]]:
        """Track score digits by clustering OCR readings at consistent positions.

        Modern La Liga scoreboards display each score as a separate digit at a
        fixed position. EasyOCR often only detects one digit per frame, but over
        time both positions accumulate enough readings to build a complete
        timeline. This method:

        1. Collects all single-digit OCR readings (filtering out clock/team text)
        2. Clusters them by 2D spatial position
        3. Identifies the two score-digit clusters (most readings, stable values)
        4. Builds per-frame score pairs, filling gaps from the last known value
        """
        # Collect all single-digit readings with positions
        digit_readings = []  # (frame_idx, digit, cx, cy)
        clock_positions = []  # (cx, cy) of clock readings for filtering

        for frame_idx, frame_texts in sorted(by_frame.items()):
            for text, cx, cy, conf in frame_texts:
                text_clean = text.strip()
                if CLOCK_PATTERN.search(text_clean):
                    clock_positions.append((cx, cy))
                    continue
                if TEAM_PATTERNS.search(text_clean.upper()):
                    continue
                if re.fullmatch(r"\d", text_clean) and conf >= 0.3:
                    digit_readings.append((frame_idx, int(text_clean), cx, cy))

        if len(digit_readings) < 20:
            logger.info("Too few digit readings for position clustering: %d", len(digit_readings))
            return []

        # Compute median clock position to filter nearby digits
        clock_cx = np.median([cx for cx, cy in clock_positions]) if clock_positions else None
        clock_cy = np.median([cy for cx, cy in clock_positions]) if clock_positions else None

        # Filter out digits very close to clock position
        if clock_cx is not None:
            digit_readings = [
                (f, d, cx, cy) for f, d, cx, cy in digit_readings
                if not (abs(cx - clock_cx) < 15 and abs(cy - clock_cy) < 15)
            ]

        if len(digit_readings) < 20:
            return []

        # 2D spatial clustering using union-find approach
        clusters = self._cluster_by_position(digit_readings, radius=12)

        # Sort clusters by number of readings (most populated first)
        clusters.sort(key=lambda c: len(c[2]), reverse=True)

        logger.info(
            "Position clusters: %d total, top sizes: %s",
            len(clusters),
            [len(c[2]) for c in clusters[:5]],
        )

        # Filter out non-score digit positions:
        # 1. Clock/timer positions cycle through all 10 digits (low concentration)
        # 2. Static overlay elements show a single non-zero value (e.g., "4" from
        #    La Liga branding, jersey numbers, or channel logos)
        # Score clusters have concentrated distributions dominated by real score
        # values and must contain "0" readings (since every match starts 0-0).
        score_clusters = []
        for cx, cy, members in clusters:
            if len(members) < 10:
                continue
            digits = [d for _, d in members]
            digit_counts = Counter(digits)
            most_common_digit, most_common_count = digit_counts.most_common(1)[0]
            concentration = most_common_count / len(members)

            # Filter 1: Low concentration = clock digit (uniform distribution)
            if concentration < 0.20:
                continue

            # Filter 2: Static non-zero element (same non-zero digit throughout).
            # A cluster dominated by a single non-zero value with very few "0"
            # readings is a static overlay element, not a score position.
            zero_count = digit_counts.get(0, 0)
            if (most_common_digit != 0
                    and most_common_count / len(members) > 0.90
                    and zero_count / len(members) < 0.05):
                logger.info(
                    "Filtering static element at (%.0f,%.0f): digit=%d, n=%d",
                    cx, cy, most_common_digit, len(members),
                )
                continue

            score_clusters.append((cx, cy, members))

        logger.info(
            "Score clusters after filtering: %d of %d",
            len(score_clusters), len(clusters),
        )

        if len(score_clusters) < 2:
            return []

        # Take the two largest score clusters as home and away
        score_clusters.sort(key=lambda c: len(c[2]), reverse=True)
        cluster_a = score_clusters[0]  # (cx, cy, [(frame, digit), ...])
        cluster_b = score_clusters[1]

        # Determine which is home (top or left) and which is away
        # Compare relative spread in x vs y to robustly detect layout
        dx = abs(cluster_a[0] - cluster_b[0])
        dy = abs(cluster_a[1] - cluster_b[1])

        if dy > dx:
            # Vertical layout: top = home, bottom = away
            if cluster_a[1] < cluster_b[1]:
                home_cluster, away_cluster = cluster_a, cluster_b
            else:
                home_cluster, away_cluster = cluster_b, cluster_a
            logger.info("Vertical layout: home@y=%.0f, away@y=%.0f",
                        home_cluster[1], away_cluster[1])
        else:
            # Horizontal layout: left = home, right = away
            if cluster_a[0] < cluster_b[0]:
                home_cluster, away_cluster = cluster_a, cluster_b
            else:
                home_cluster, away_cluster = cluster_b, cluster_a
            logger.info("Horizontal layout: home@x=%.0f, away@x=%.0f",
                        home_cluster[0], away_cluster[0])

        # Build per-frame readings for each cluster
        home_by_frame = {}
        for f, d in home_cluster[2]:
            home_by_frame[f] = d
        away_by_frame = {}
        for f, d in away_cluster[2]:
            away_by_frame[f] = d

        logger.info(
            "Home cluster: %d readings, Away cluster: %d readings",
            len(home_by_frame), len(away_by_frame),
        )

        # Apply temporal median filter to each cluster independently.
        # This removes transient stat overlay spikes (possession, throws, etc.)
        # that briefly replace score digits for a few seconds.
        home_by_frame = self._temporal_median_filter(home_by_frame, window=15)
        away_by_frame = self._temporal_median_filter(away_by_frame, window=15)

        # Build combined timeline using last-known-value for missing frames
        all_frames = sorted(set(home_by_frame.keys()) | set(away_by_frame.keys()))
        if not all_frames:
            return []

        raw_scores = []
        last_home = home_by_frame.get(all_frames[0], 0)
        last_away = away_by_frame.get(all_frames[0], 0)

        for frame_idx in all_frames:
            home = home_by_frame.get(frame_idx, last_home)
            away = away_by_frame.get(frame_idx, last_away)
            if home <= 12 and away <= 12:
                raw_scores.append((frame_idx, home, away))
                last_home = home
                last_away = away

        return raw_scores

    def _temporal_median_filter(
        self, readings_by_frame: dict[int, int], window: int = 15
    ) -> dict[int, int]:
        """Apply temporal median filter to remove transient stat overlay spikes.

        Broadcast scoreboards sometimes temporarily show statistics (possession,
        throws, shots on target, etc.) in place of the goal score. These stat
        overlays last a few seconds and produce digit values that differ from the
        actual score. The median filter replaces each reading with the median of
        its temporal neighborhood, effectively removing brief outlier spikes while
        preserving genuine score changes that persist over time.

        Args:
            readings_by_frame: {frame_idx: digit_value} for a single position cluster
            window: Number of neighboring readings to consider (must be odd for
                    true median; even values are rounded up internally)
        """
        if not readings_by_frame:
            return {}

        frames = sorted(readings_by_frame.keys())
        values = [readings_by_frame[f] for f in frames]
        filtered = {}
        half_w = window // 2

        for i, f in enumerate(frames):
            start = max(0, i - half_w)
            end = min(len(values), i + half_w + 1)
            filtered[f] = int(np.median(values[start:end]))

        return filtered

    def _cluster_by_position(
        self, readings: list[tuple[int, int, float, float]], radius: float
    ) -> list[tuple[float, float, list[tuple[int, int]]]]:
        """Cluster digit readings by 2D spatial position.

        Returns: list of (centroid_x, centroid_y, [(frame, digit), ...])
        """
        if not readings:
            return []

        # Simple greedy clustering
        clusters = []  # (sum_x, sum_y, count, [(frame, digit), ...])
        for frame_idx, digit, cx, cy in readings:
            assigned = False
            for i, (sx, sy, cnt, members) in enumerate(clusters):
                mean_x = sx / cnt
                mean_y = sy / cnt
                if abs(cx - mean_x) < radius and abs(cy - mean_y) < radius:
                    clusters[i] = (sx + cx, sy + cy, cnt + 1, members + [(frame_idx, digit)])
                    assigned = True
                    break
            if not assigned:
                clusters.append((cx, cy, 1, [(frame_idx, digit)]))

        return [(sx / cnt, sy / cnt, members) for sx, sy, cnt, members in clusters]

    def _detect_score_changes(
        self, raw_scores: list[tuple[int, int, int]], fps: int
    ) -> list[ScoreEvent]:
        """Apply majority voting and detect when score changes.

        Uses a large voting window to smooth OCR noise, then validates each
        detected transition with a forward persistence check to filter out
        remaining stat overlay artifacts.

        Only accepts transitions where the total score increases by exactly 1.
        Jumps >1 are treated as stat overlay contamination (not missed goals),
        since the temporal median filter and majority voting should capture all
        real single-step transitions.
        """
        if not raw_scores:
            return []

        # Use larger window for stability (at 3s intervals, 11 readings = ~33s)
        window = max(self.config.majority_window, 11)

        # Apply majority voting for each reading
        stable_scores = []
        for i in range(len(raw_scores)):
            start = max(0, i - window // 2)
            end = min(len(raw_scores), i + window // 2 + 1)
            window_scores = raw_scores[start:end]

            # Most common (home, away) pair in window
            counts = Counter((h, a) for _, h, a in window_scores)
            (home, away), _ = counts.most_common(1)[0]
            stable_scores.append((raw_scores[i][0], home, away))

        # Detect transitions in the stable timeline
        events = []
        prev_home, prev_away = stable_scores[0][1], stable_scores[0][2]

        # Minimum number of forward readings that must match the new score
        # to confirm the change is permanent (not a stat overlay).
        # At 3s intervals, 8 readings = ~24 seconds.
        min_persist = 8

        for idx, (frame_idx, home, away) in enumerate(stable_scores[1:], 1):
            if home == prev_home and away == prev_away:
                continue

            home_diff = home - prev_home
            away_diff = away - prev_away

            # Both scores must be non-decreasing (goals can't be un-scored)
            if home_diff < 0 or away_diff < 0:
                continue

            total_diff = home_diff + away_diff
            # Reject large jumps — these are stat overlay artifacts (possession,
            # throws, etc.) that survived median filtering and majority voting.
            # Allow small jumps (≤2 total) for rapid successive goals that the
            # temporal smoothing might merge into a single transition.
            if total_diff == 0 or total_diff > 2:
                continue

            # Forward persistence check: verify the new score persists
            # for at least min_persist readings after the transition.
            # This filters out transient spikes from stat overlays that
            # survived the median filter and majority voting.
            lookahead = stable_scores[idx:idx + min_persist]
            if len(lookahead) >= min_persist:
                match_count = sum(1 for _, h, a in lookahead if h == home and a == away)
                if match_count < min_persist * 0.6:
                    # New score doesn't persist — likely stat overlay
                    continue

            # Generate one event per goal in the transition
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

        return events

    def run(self, video_path: str) -> tuple[Optional[ScoreboardRegion], list[ScoreEvent]]:
        """Full pipeline: find scoreboard, track scores, detect goals.

        Returns:
            region: Detected scoreboard region (or None)
            events: List of score change events
        """
        logger.info("Building variance map...")
        variance_map = self.find_static_overlays(video_path)

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
