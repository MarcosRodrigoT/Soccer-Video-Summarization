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

import bisect
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
# Note: LALIGA/LA LIGA excluded — those appear on the league logo, not the scoreboard
TEAM_PATTERNS = re.compile(
    r"\b(BAR|RMA|ATM|SEV|BET|VAL|VIL|CEL|SOC|ATH|GIR|ALM|CAD|GET|GRA|"
    r"MAL|MAJ|OSA|RAY|ELC|ESP|LPA|FCB|RM|ATL|VCF|RCD|RCE|LEV)\b",
    re.IGNORECASE,
)

# League branding — penalizes regions that are logos, not scoreboards
LEAGUE_PATTERNS = re.compile(r"\b(LALIGA|LA\s*LIGA|EA\s*SPORTS)\b", re.IGNORECASE)

# Clock pattern: MM:SS or MM.SS
CLOCK_PATTERN = re.compile(r"\b(\d{1,3})[:.]\s*(\d{2})\b")

# Score with separator: "X - Y" (dash only; colon matches clock text)
SCORE_WITH_SEP = re.compile(r"\b(\d{1,2})\s*[-]\s*(\d{1,2})\b")

# Common OCR confusions: letter → digit mapping for scoreboard text.
# EasyOCR often misreads thin "1" as "l"/"I"/"|", round "0" as "O", etc.
_CHAR_TO_DIGIT = {
    'l': '1', 'I': '1', '|': '1', 'i': '1',
    'O': '0', 'o': '0', 'D': '0',
    'S': '5', 's': '5',
    'B': '8', 'b': '6',
    'g': '9', 'q': '9',
    'Z': '2', 'z': '2',
}


def _normalize_ocr_digits(text: str) -> str:
    """Replace common OCR letter-digit confusions with digit equivalents."""
    return ''.join(_CHAR_TO_DIGIT.get(c, c) for c in text)


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
    display_bbox: dict = None  # edge-aware tightened bbox for visualization


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
        """Build a per-pixel variability map to identify static overlays.

        Samples frames evenly across the video and computes a robust
        variability measure for each pixel. Uses the interquartile range
        (P75 - P25) instead of standard deviation, because std is sensitive
        to outlier frames (halftime, replays, intros) where the scoreboard
        isn't visible. Even 5% of such frames can make white scoreboard
        pixels appear "high variance" with std, but IQR ignores those
        outliers and correctly identifies them as static.

        Returns:
            variability_map: 2D array (same units as pixel intensity, 0-255).
                Static overlays have values near 0.
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

        # IQR (interquartile range) is robust to outlier frames where
        # the scoreboard isn't present (halftime, replays, intros).
        p75 = np.percentile(stack, 75, axis=0)
        p25 = np.percentile(stack, 25, axis=0)
        variability_map = p75 - p25

        logger.info(
            "Variability map (IQR): shape=%s, range=[%.1f, %.1f], mean=%.1f, "
            "%d samples",
            variability_map.shape, variability_map.min(), variability_map.max(),
            variability_map.mean(), len(frames_gray),
        )
        return variability_map

    def build_overlay_mask(self, std_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build a binary mask of static broadcast overlay regions.

        Uses a low percentile of the std map to find truly static pixels,
        then applies morphological closing to bridge gaps at digit/clock
        positions (which change occasionally). The result is a mask where
        overlay regions (scoreboard, clock, logo) are white and game action
        is black.

        Returns:
            overlay_mask: Binary mask after morphological ops (0 or 255)
            raw_mask: Binary mask before closing (used for splitting wide regions)
        """
        threshold = np.percentile(std_map, self.config.std_percentile)
        raw_mask = (std_map < threshold).astype(np.uint8) * 255

        logger.info("Overlay mask: std threshold=%.1f (P%.0f), %d static pixels (%.1f%%)",
                     threshold, self.config.std_percentile,
                     np.sum(raw_mask > 0), np.sum(raw_mask > 0) / raw_mask.size * 100)

        # Closing fills gaps at digit positions (which change value occasionally)
        kw, kh = self.config.closing_kernel
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # Opening removes small noise specks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        return mask, raw_mask

    def find_scoreboard_region(
        self, video_path: str, std_map: np.ndarray,
        overlay_mask: np.ndarray = None,
        raw_mask: np.ndarray = None,
    ) -> Optional[ScoreboardRegion]:
        """Find the scoreboard among static overlay regions.

        Broadcast overlays (scoreboard, logo, clock) appear as distinct
        low-variance regions, typically at the top of the frame. The std map
        makes them clearly visible. This method:

        1. Extracts connected components from the overlay mask
        2. Splits over-wide regions using column gaps in the raw mask
        3. Tests each candidate with OCR, heavily favoring score patterns
        4. Returns the region most likely to be the scoreboard
        """
        h, w = std_map.shape

        if overlay_mask is None:
            overlay_mask, raw_mask = self.build_overlay_mask(std_map)

        # Extract connected components from the overlay mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            overlay_mask, connectivity=8
        )

        min_area = self.config.min_region_area
        max_width = w * 0.4  # scoreboard is never >40% of frame width
        candidates = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]

            # Tighten bounding box to actual content in raw mask.
            # Morphological closing expands regions by half the kernel size,
            # which can push bounding boxes to frame edges.
            if raw_mask is not None:
                x, y, cw, ch = self._tighten_to_raw(raw_mask, x, y, cw, ch)
                if cw * ch < min_area:
                    continue

            if cw > max_width:
                # Region too wide — likely merged UI elements. Split by
                # finding vertical gaps in the raw (pre-closing) mask.
                split_mask = raw_mask if raw_mask is not None else overlay_mask
                sub_regions = self._split_wide_region(
                    split_mask, x, y, cw, ch, min_area
                )
                candidates.extend(sub_regions)
            else:
                candidates.append(ScoreboardRegion(x=x, y=y, w=cw, h=ch))

        if not candidates:
            logger.warning("No candidate overlay regions found")
            return None

        # Only merge components that truly overlap (gap=5px) — don't merge
        # separate UI elements (scoreboard vs logo) into one huge region
        merged = self._merge_regions(candidates, gap=5)
        logger.info("Found %d overlay regions (merged from %d components)",
                     len(merged), len(candidates))
        for r in merged:
            logger.info("  Region: (%d,%d) %dx%d", r.x, r.y, r.w, r.h)

        # Test OCR on candidates using a mid-match frame
        props = get_video_properties(video_path)
        cap = cv2.VideoCapture(video_path)
        test_frames = []
        for frac in [0.3, 0.5, 0.7]:
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
                # Use plain crop (no masking) — masking can black out digits
                crop = self._crop_region(frame, region, pad=5)
                if crop.size == 0:
                    continue

                upscaled = cv2.resize(crop, None, fx=4, fy=4,
                                      interpolation=cv2.INTER_CUBIC)
                results = ocr.readtext(upscaled)

                for bbox, text, conf in results:
                    text_upper = text.upper().strip()
                    # League branding = logo, not scoreboard
                    if LEAGUE_PATTERNS.search(text_upper):
                        region_score -= 10
                        continue
                    # Score pattern is the strongest signal (e.g. "2-1", "0:0")
                    if SCORE_WITH_SEP.search(text):
                        region_score += 10
                    # Team abbreviations are strong evidence
                    if TEAM_PATTERNS.search(text_upper):
                        region_score += 5
                    # Clock pattern (MM:SS) is good but less decisive
                    if CLOCK_PATTERN.search(text):
                        region_score += 3
                    # Any digit is weak evidence
                    if re.search(r"\d", text):
                        region_score += 1

            logger.info("  Region (%d,%d,%d,%d): OCR score=%d",
                        region.x, region.y, region.w, region.h, region_score)

            if region_score > best_score:
                best_score = region_score
                best_region = region

        if best_region and best_score >= 3:
            logger.info(
                "Scoreboard at (%d,%d,%d,%d) score=%d",
                best_region.x, best_region.y, best_region.w, best_region.h,
                best_score,
            )
            return best_region

        # Fallback: prefer the largest region in the top 30% of the frame
        top_regions = [r for r in merged if r.y < h * 0.3]
        if top_regions:
            largest = max(top_regions, key=lambda r: r.w * r.h)
            logger.info(
                "Fallback: largest top region (%d,%d,%d,%d)",
                largest.x, largest.y, largest.w, largest.h,
            )
            return largest

        logger.warning("No scoreboard region found")
        return None

    def _merge_regions(self, regions: list[ScoreboardRegion],
                       gap: int = 5) -> list[ScoreboardRegion]:
        """Merge only truly overlapping or adjacent regions.

        Uses a small gap (default 5px) to avoid merging distinct UI elements
        (e.g. scoreboard and channel logo) that are spatially separated.
        """
        if not regions:
            return []

        # Convert to (x1, y1, x2, y2) format
        boxes = [(r.x, r.y, r.x + r.w, r.y + r.h) for r in regions]
        merged = []
        used = [False] * len(boxes)

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

    def _split_wide_region(
        self, mask: np.ndarray, x: int, y: int, w: int, h: int,
        min_area: int,
    ) -> list[ScoreboardRegion]:
        """Split a too-wide overlay region by finding significant vertical gaps.

        When the morphological closing bridges the scoreboard and logo into
        one wide band, this method finds major column gaps (30+ consecutive
        low-density columns) in the raw mask to split it into separate regions.
        Small gaps within the scoreboard (digit positions, etc.) are ignored.
        """
        region_mask = mask[y:y + h, x:x + w]
        col_sums = np.sum(region_mask > 0, axis=0)

        # Classify each column as "filled" or "empty"
        gap_threshold = h * 0.15  # <15% of height filled = empty column
        is_filled = col_sums > gap_threshold

        # Find runs of empty columns that are wide enough to be real gaps
        # between separate UI elements (not just digit gaps within scoreboard)
        min_gap_width = 30  # minimum gap to split on
        regions = []
        start_col = None

        col = 0
        while col < len(is_filled):
            if is_filled[col]:
                if start_col is None:
                    start_col = col
                col += 1
            else:
                # Found an empty column — measure the gap width
                gap_start = col
                while col < len(is_filled) and not is_filled[col]:
                    col += 1
                gap_width = col - gap_start

                if start_col is not None and gap_width >= min_gap_width:
                    # Significant gap — end current region
                    rw = gap_start - start_col
                    if rw * h >= min_area:
                        regions.append(ScoreboardRegion(
                            x=int(x + start_col), y=int(y), w=int(rw), h=int(h)
                        ))
                    start_col = None
                # If gap is small, keep scanning (don't split)

        # Close last region
        if start_col is not None:
            rw = len(is_filled) - start_col
            if rw * h >= min_area:
                regions.append(ScoreboardRegion(
                    x=int(x + start_col), y=int(y), w=int(rw), h=int(h)
                ))

        if len(regions) > 1:
            logger.info("Split wide region (%d,%d,%d,%d) into %d sub-regions: %s",
                        x, y, w, h, len(regions),
                        [(r.x, r.y, r.w, r.h) for r in regions])
        elif not regions:
            # Can't split — return as-is
            regions = [ScoreboardRegion(x=x, y=y, w=w, h=h)]

        return regions

    @staticmethod
    def _tighten_to_raw(
        raw_mask: np.ndarray, x: int, y: int, w: int, h: int,
        ignore_edges: bool = False,
    ) -> tuple[int, int, int, int]:
        """Shrink a bounding box to the actual content in the raw mask.

        Morphological closing expands regions by half the kernel width/height,
        which can push bounding boxes to the frame edge even when the actual
        static pixels don't extend that far. This trims the box to the
        tightest fit around raw mask content.

        When ignore_edges=True, two additional filters are applied:
        1. Frame-edge pixels (first/last 2 rows and columns) are excluded.
           Broadcasts often have a 1-pixel static black strip at the edges
           that the IQR detects as "static".
        2. A 5% density threshold replaces np.any — a column/row must have
           at least 5% of its pixels active to count as content. This
           filters out sparse stray static pixels (typically <2% density)
           that morphological closing bridges to the scoreboard, inflating
           the bbox well beyond the actual overlay.
        """
        region = raw_mask[y:y + h, x:x + w]

        if ignore_edges:
            min_density = 0.05
            col_has_content = np.mean(region > 0, axis=0) >= min_density
            row_has_content = np.mean(region > 0, axis=1) >= min_density
            frame_h, frame_w = raw_mask.shape
            border_px = 2
            for dy in range(h):
                abs_y = y + dy
                if abs_y < border_px or abs_y >= frame_h - border_px:
                    row_has_content[dy] = False
            for dx in range(w):
                abs_x = x + dx
                if abs_x < border_px or abs_x >= frame_w - border_px:
                    col_has_content[dx] = False
        else:
            col_has_content = np.any(region > 0, axis=0)
            row_has_content = np.any(region > 0, axis=1)

        cols = np.where(col_has_content)[0]
        rows = np.where(row_has_content)[0]

        if len(cols) == 0 or len(rows) == 0:
            return x, y, w, h  # no content — keep original

        new_x = x + int(cols[0])
        new_y = y + int(rows[0])
        new_w = int(cols[-1] - cols[0] + 1)
        new_h = int(rows[-1] - rows[0] + 1)
        return new_x, new_y, new_w, new_h

    def _crop_region(
        self, frame: np.ndarray, region: ScoreboardRegion, pad: int = 10
    ) -> np.ndarray:
        """Crop a region from a frame with padding."""
        y1 = max(0, region.y - pad)
        y2 = min(frame.shape[0], region.y + region.h + pad)
        x1 = max(0, region.x - pad)
        x2 = min(frame.shape[1], region.x + region.w + pad)
        return frame[y1:y2, x1:x2]

    def _crop_and_mask(
        self, frame: np.ndarray, region: ScoreboardRegion,
        overlay_mask: np.ndarray, pad: int = 10,
    ) -> np.ndarray:
        """Crop a region and black out pixels outside the overlay mask.

        The overlay mask (from build_overlay_mask) covers static broadcast
        elements. Pixels outside this mask are game action that should not
        be fed to OCR. Setting them to black prevents false digit detections
        from jersey numbers, advertising boards, etc.
        """
        y1 = max(0, region.y - pad)
        y2 = min(frame.shape[0], region.y + region.h + pad)
        x1 = max(0, region.x - pad)
        x2 = min(frame.shape[1], region.x + region.w + pad)

        crop = frame[y1:y2, x1:x2].copy()
        mask_crop = overlay_mask[y1:y2, x1:x2]

        # Black out non-overlay pixels
        crop[mask_crop == 0] = 0
        return crop

    @staticmethod
    def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """Preprocess an upscaled crop for better OCR accuracy.

        Converts to grayscale, enhances local contrast with CLAHE,
        removes compression artifacts with bilateral filtering, and
        sharpens digit edges with an unsharp mask.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # CLAHE: normalize local contrast for semi-transparent overlays
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter: smooth compression artifacts, preserve edges
        denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)

        # Unsharp mask: sharpen digit edges softened by upscale
        blurred = cv2.GaussianBlur(denoised, (0, 0), 2.0)
        sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

        return sharpened

    def track_scores(
        self,
        video_path: str,
        region: ScoreboardRegion,
        overlay_mask: np.ndarray = None,
    ) -> list[ScoreEvent]:
        """Track score changes over time by sampling frames and running OCR.

        Uses a two-phase approach:
        1. Collect all OCR readings with spatial information
        2. Separate clock readings from score readings using temporal stability
        3. Detect score changes from stable readings

        If overlay_mask is provided, non-overlay pixels in the crop are blacked
        out before OCR, preventing false readings from game action background.

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
        pad = 15

        # Scoreboard bounds within the padded crop (cx/cy are crop-relative)
        sb_x_min = pad
        sb_x_max = pad + region.w
        sb_y_min = pad
        sb_y_max = pad + region.h

        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            if overlay_mask is not None:
                crop = self._crop_and_mask(frame, region, overlay_mask, pad=pad)
            else:
                crop = self._crop_region(frame, region, pad=pad)
            if crop.size == 0:
                frame_idx += interval_frames
                continue

            # Upscale 4x and preprocess for better OCR on small digits
            scale = 4
            upscaled = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            # Lower detection thresholds to catch thin "1" strokes that
            # EasyOCR's CRAFT detector misses at default sensitivity
            results = ocr.readtext(upscaled, text_threshold=0.5, low_text=0.3)

            for bbox, text, conf in results:
                # Compute center of bbox (in upscaled coordinates, convert back)
                cx = (bbox[0][0] + bbox[2][0]) / 2 / scale
                cy = (bbox[0][1] + bbox[2][1]) / 2 / scale
                # Discard readings whose center falls outside the scoreboard
                # region (the padding area can contain jersey numbers, ads, etc.)
                if cx < sb_x_min or cx > sb_x_max or cy < sb_y_min or cy > sb_y_max:
                    continue
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
                # Normalize dots between single digits to dashes (OCR misreads
                # tiny dashes as dots in classic broadcasts, e.g. "2.3" → "2-3")
                # Only single-digit.single-digit to avoid normalizing clock "45.30"
                text_norm = _normalize_ocr_digits(text)
                text_norm = re.sub(r"(?<!\d)(\d)\s*\.\s*(\d)(?!\d)", r"\1-\2", text_norm)
                m = SCORE_WITH_SEP.search(text_norm)
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
                text_normalized = _normalize_ocr_digits(text_clean)
                if re.fullmatch(r"\d", text_normalized) and conf >= 0.15:
                    digit_readings.append((frame_idx, int(text_normalized), cx, cy))

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

        # Ghost digit inference: infer "1" when one position detected but other missing
        ghost_home, ghost_away = self._infer_ghost_ones(
            home_cluster, away_cluster, by_frame)
        logger.info("Ghost inference: %d home, %d away ghost '1' readings",
                    len(ghost_home), len(ghost_away))

        # Build per-frame readings for each cluster (including ghost readings)
        home_by_frame = {}
        for f, d in home_cluster[2]:
            home_by_frame[f] = d
        for f, d in ghost_home:
            if f not in home_by_frame:
                home_by_frame[f] = d

        away_by_frame = {}
        for f, d in away_cluster[2]:
            away_by_frame[f] = d
        for f, d in ghost_away:
            if f not in away_by_frame:
                away_by_frame[f] = d

        logger.info(
            "Home cluster: %d readings (%d ghost), Away cluster: %d readings (%d ghost)",
            len(home_by_frame), len(ghost_home),
            len(away_by_frame), len(ghost_away),
        )

        # Apply temporal median filter to each cluster independently.
        # This removes transient stat overlay spikes (possession, throws, etc.)
        # that briefly replace score digits for a few seconds.
        home_by_frame = self._temporal_median_filter(home_by_frame, window=15)
        away_by_frame = self._temporal_median_filter(away_by_frame, window=15)

        # Enforce monotonic non-decreasing scores: goals can't be un-scored.
        # After median filtering, any remaining dips are OCR errors (e.g., thin
        # "1" digit missed → reads as "0"). Clamp each value to the running max.
        home_by_frame = self._enforce_monotonic(home_by_frame)
        away_by_frame = self._enforce_monotonic(away_by_frame)

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

    @staticmethod
    def _enforce_monotonic(readings_by_frame: dict[int, int]) -> dict[int, int]:
        """Enforce monotonic non-decreasing values over time.

        Goals cannot be un-scored, so the score at any point must be >= all
        previous scores. This clamps transient dips (e.g., EasyOCR missing a
        thin "1" digit and reading "0" instead) to the running maximum.
        """
        if not readings_by_frame:
            return {}
        result = {}
        running_max = 0
        for f in sorted(readings_by_frame.keys()):
            val = readings_by_frame[f]
            running_max = max(running_max, val)
            result[f] = running_max
        return result

    @staticmethod
    def _infer_ghost_ones(
        home_cluster: tuple,
        away_cluster: tuple,
        by_frame: dict[int, list[tuple]],
        nearby_window: int = 10,
    ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        """Infer '1' at score positions where OCR detected nothing.

        The digit '1' is thin enough that EasyOCR's CRAFT detector often
        produces no bounding box at all.  When one score position has a
        detection but the other doesn't, and the missing position's
        nearby readings are predominantly '1', the gap is filled with a
        ghost '1' reading.

        The nearby-context requirement prevents false ghosts: at score 0
        (nearby readings are 0) or score 3 (nearby readings are 3), no
        ghost is added.  Only when nearby readings confirm the score is
        currently '1' does the ghost fire.

        Args:
            nearby_window: number of nearest real readings (before and
                after) to consult when deciding the local score value.

        Returns (ghost_home, ghost_away) lists of (frame_idx, 1) tuples.
        """
        home_frames = {f for f, d in home_cluster[2]}
        away_frames = {f for f, d in away_cluster[2]}
        visible_frames = set(by_frame.keys())

        # Sort cluster members for binary-search of nearby readings
        home_sorted = sorted(home_cluster[2])
        away_sorted = sorted(away_cluster[2])

        def _local_mode(sorted_members, frame_idx, k):
            """Most common digit among the k nearest readings to frame_idx."""
            if not sorted_members:
                return None
            # Binary search for insertion point
            frames = [f for f, d in sorted_members]
            pos = bisect.bisect_left(frames, frame_idx)
            # Gather up to k nearest readings
            neighbors = []
            lo, hi = pos - 1, pos
            while len(neighbors) < k and (lo >= 0 or hi < len(sorted_members)):
                if lo >= 0 and hi < len(sorted_members):
                    if frame_idx - sorted_members[lo][0] <= sorted_members[hi][0] - frame_idx:
                        neighbors.append(sorted_members[lo][1])
                        lo -= 1
                    else:
                        neighbors.append(sorted_members[hi][1])
                        hi += 1
                elif lo >= 0:
                    neighbors.append(sorted_members[lo][1])
                    lo -= 1
                else:
                    neighbors.append(sorted_members[hi][1])
                    hi += 1
            if not neighbors:
                return None
            return Counter(neighbors).most_common(1)[0][0]

        ghost_home: list[tuple[int, int]] = []
        ghost_away: list[tuple[int, int]] = []
        for frame_idx in visible_frames:
            has_home = frame_idx in home_frames
            has_away = frame_idx in away_frames
            if has_home and not has_away:
                if _local_mode(away_sorted, frame_idx, nearby_window) == 1:
                    ghost_away.append((frame_idx, 1))
            elif has_away and not has_home:
                if _local_mode(home_sorted, frame_idx, nearby_window) == 1:
                    ghost_home.append((frame_idx, 1))
        return ghost_home, ghost_away

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

            # Backtrack to find the first raw reading with the new score.
            # Majority voting delays detection by ~half the window; the actual
            # score change happened earlier in the raw data.
            event_frame = frame_idx
            for j in range(idx - 1, -1, -1):
                _, rh, ra = raw_scores[j]
                if rh == home and ra == away:
                    event_frame = raw_scores[j][0]
                else:
                    break

            # Generate one event per goal in the transition
            for _ in range(home_diff):
                events.append(ScoreEvent(
                    timestamp=event_frame / fps,
                    frame=event_frame,
                    home_score=home,
                    away_score=away,
                    team="home",
                ))
            for _ in range(away_diff):
                events.append(ScoreEvent(
                    timestamp=event_frame / fps,
                    frame=event_frame,
                    home_score=home,
                    away_score=away,
                    team="away",
                ))
            prev_home, prev_away = home, away

        return events

    def run(self, video_path: str) -> tuple[Optional[ScoreboardRegion], list[ScoreEvent]]:
        """Full pipeline: find scoreboard, track scores, detect goals.

        Returns:
            region: Detected scoreboard region (or None).
                The region has a ``display_bbox`` attribute with edge-aware
                tightened coordinates for visualization (excludes frame-edge
                static pixels that inflate the bbox but don't affect OCR).
            events: List of score change events
        """
        logger.info("Building std map...")
        std_map = self.find_static_overlays(video_path)

        logger.info("Building overlay mask...")
        overlay_mask, raw_mask = self.build_overlay_mask(std_map)

        logger.info("Finding scoreboard region...")
        region = self.find_scoreboard_region(video_path, std_map, overlay_mask, raw_mask)

        if region is None:
            logger.warning("Could not locate scoreboard")
            return None, []

        # Compute display bbox: edge-aware tightened version that excludes
        # frame-edge static pixels (black broadcast borders) for cleaner
        # visualization. The OCR uses the original (wider) region.
        dx, dy, dw, dh = self._tighten_to_raw(
            raw_mask, region.x, region.y, region.w, region.h,
            ignore_edges=True,
        )
        region.display_bbox = {"x": dx, "y": dy, "w": dw, "h": dh}

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
