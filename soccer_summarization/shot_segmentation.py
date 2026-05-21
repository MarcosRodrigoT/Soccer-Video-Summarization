"""Shot boundary detection — faithful Python port of the original Matlab implementation.

Algorithm (matching Matlab exactly):
  1. Compute per-frame L1 distance on normalized 256-bin grayscale histograms.
  2. Detect abrupt transitions (cuts) with a fixed threshold + 15-frame window validation.
  3. Detect gradual transitions (dissolves/fades) via peak detection on a smoothed signal.
  4. Build shots from the gaps between the union of forbidden segments.

Reference Matlab files:
  calculateHistogramDifferences.m  → compute_frame_differences()
  detectAbruptTransitions.m        → detect_abrupt_transitions()
  detectGradualTransitionsByPeak.m → detect_gradual_transitions()
  Main.m (shot-building section)   → _build_shots()
"""

import logging
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

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


# ── frame histogram computation ───────────────────────────────────────────────

def _histogram_diff_batch(args: tuple) -> list[tuple[int, float]]:
    """Compute normalized-histogram L1 distances for a batch of frames.

    Returns (frame_index, l1_distance) pairs where frame_index is the index
    of the SECOND frame in each consecutive pair (same convention as Matlab's
    allHistDifferences indexing, converted to 0-based).

    For non-first batches, read_start < actual_start so that the first frame
    read initialises prev_hist without generating a diff entry — this ensures
    cross-batch boundary diffs are computed correctly.
    """
    read_start, actual_start, end_frame, video_path, total_frames = args

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, read_start)

    diffs = []
    prev_hist = None

    for i in range(read_start, min(end_frame, total_frames - 1)):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        # Fast 256-bin histogram for uint8 images (equivalent to Matlab imhist)
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        total = hist.sum()
        if total > 0:
            hist /= total

        if prev_hist is not None and i >= actual_start:
            l1 = float(np.sum(np.abs(hist - prev_hist)))
            diffs.append((i, l1))  # index i = diff between frame i-1 and frame i

        prev_hist = hist

    cap.release()
    return diffs


# ── abrupt transition detection ───────────────────────────────────────────────

def _detect_abrupt(
    diffs: np.ndarray,
    threshold: float,
    window_size: int,
    neighborhood_factor: float,
) -> list[int]:
    """Detect abrupt cuts (Matlab detectAbruptTransitions).

    A peak at index i is an abrupt transition iff:
      1. diffs[i] > threshold
      2. diffs[i] is the maximum in a centred window of size window_size
      3. All other values in that window are < diffs[i] * neighborhood_factor

    Returns list of 0-based diff indices where abrupt cuts occur.
    """
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2

    # Zero-pad both ends (matching Matlab's paddedHistDifferences)
    padded = np.concatenate([np.zeros(half), diffs, np.zeros(half)])

    abrupt_indices = []
    n = len(diffs)

    for i in range(n):
        val = diffs[i]
        if val <= threshold:
            continue

        # Window in padded array centred on position i
        w_start = i          # i + half (padded offset) - half = i
        w_end   = i + window_size  # inclusive slice end is w_end (exclusive)
        window = padded[w_start:w_end]

        peak_val = window.max()
        if peak_val != val:
            continue  # not the maximum in the window

        # All non-peak values must be < val * neighborhood_factor
        non_peak = window[window != val]
        if len(non_peak) > 0 and np.any(non_peak >= val * neighborhood_factor):
            continue

        abrupt_indices.append(i)

    return abrupt_indices


# ── gradual transition detection ──────────────────────────────────────────────

def _detect_gradual(
    diffs: np.ndarray,
    abrupt_indices: list[int],
    filter_window: int,
    min_peak_height: float,
    min_peak_prominence: float,
    local_threshold_factor: float,
    local_window_ratio: int,
    min_duration: int,
    min_distance_from_abrupt: int,
    min_contrast_factor: float,
) -> list[tuple[int, int]]:
    """Detect gradual transitions (Matlab detectGradualTransitionsByPeak).

    Returns list of (start_frame, end_frame) 0-based frame ranges that are
    gradual transition regions (inclusive).  These correspond to Matlab's
    consolidatedGraduals converted from 1-based to 0-based.
    """
    n = len(diffs)

    # Step 1: smooth with moving average (Matlab movmean with 'Endpoints','fill')
    filtered = (
        pd.Series(diffs)
        .rolling(window=filter_window, center=True, min_periods=1)
        .mean()
        .values
    )

    # Step 2: find peaks
    peaks, _ = find_peaks(
        filtered,
        height=min_peak_height,
        prominence=min_peak_prominence,
    )

    # Convert abrupt diff indices to "incoming frame" indices for distance check.
    # Matlab uses: abs((peakIdx + 1) - abruptFrames) where peakIdx is 1-based diff
    # index and abruptFrames = abruptTransitions[:,2] (1-based incoming frame).
    # In Python 0-based: abrupt at diff j → incoming frame j+1.
    # Peak at diff index p (0-based) → "incoming frame" p+1.
    # Distance = |(p+1) - (j+1)| = |p - j|
    abrupt_set = set(abrupt_indices)

    gradual_candidates = []

    exclusion_half = filter_window // 2 + 1
    estimated_peak_width = min_duration
    local_win = max(filter_window, max(3, round(estimated_peak_width * local_window_ratio)))
    half_local = local_win // 2

    for p in peaks:
        peak_val = filtered[p]

        # Discard if too close to an abrupt transition
        if any(abs(p - j) < min_distance_from_abrupt for j in abrupt_set):
            continue

        # Compute local baseline excluding the peak region
        l_start = max(0, p - half_local)
        l_end   = min(n - 1, p + half_local)

        base_indices = [
            k for k in range(l_start, l_end + 1)
            if k < (p - exclusion_half) or k > (p + exclusion_half)
        ]
        if not base_indices:
            continue

        local_base = float(np.mean(filtered[base_indices]))

        # Local drop threshold
        threshold_to_drop = local_base * local_threshold_factor
        threshold_to_drop = min(threshold_to_drop, peak_val * 0.9)
        threshold_to_drop = max(threshold_to_drop, min_peak_height * 0.5)

        # Discard low-contrast peaks: (peak - base) must exceed base * factor
        if (peak_val - local_base) <= local_base * min_contrast_factor:
            continue

        # Expand backward from peak
        start_idx = p
        for prev_i in range(p, -1, -1):
            if filtered[prev_i] < threshold_to_drop or prev_i == 0:
                start_idx = prev_i
                break
            start_idx = prev_i

        # Expand forward from peak
        end_idx = p
        for next_i in range(p, n):
            if filtered[next_i] < threshold_to_drop or next_i == n - 1:
                end_idx = next_i
                break
            end_idx = next_i

        # Convert diff indices to frame numbers (0-based).
        # Matlab: candidateStartFrame = startIdx + 1 (1-based) → Python: startIdx (0-based diff)
        # Forbidden frame range = [start_idx, end_idx] in diff-index space,
        # which maps to frame range [start_idx + 1, end_idx + 1] in 0-based frame numbers.
        cand_start_frame = start_idx + 1   # 0-based frame
        cand_end_frame   = end_idx   + 1   # 0-based frame

        if (cand_end_frame - cand_start_frame + 1) < min_duration:
            continue

        # Reject if any abrupt transition falls inside this candidate
        overlaps_abrupt = any(
            cand_start_frame <= (j + 1) <= cand_end_frame
            for j in abrupt_set
        )
        if overlaps_abrupt:
            continue

        gradual_candidates.append((cand_start_frame, cand_end_frame))
        logger.debug(
            "Gradual: frames [%d, %d] dur=%d  peak=%.4f  base=%.4f",
            cand_start_frame, cand_end_frame,
            cand_end_frame - cand_start_frame + 1,
            peak_val, local_base,
        )

    # Post-consolidation: merge adjacent/overlapping graduales
    if not gradual_candidates:
        return []

    gradual_candidates = sorted(set(gradual_candidates), key=lambda x: x[0])
    consolidated = []
    cur_s, cur_e = gradual_candidates[0]

    for s, e in gradual_candidates[1:]:
        if s <= cur_e + min_duration:
            cur_e = max(cur_e, e)
        else:
            consolidated.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    consolidated.append((cur_s, cur_e))

    return consolidated


# ── shot construction ─────────────────────────────────────────────────────────

def _build_shots(
    abrupt_indices: list[int],
    gradual_regions: list[tuple[int, int]],
    total_frames: int,
    fps: float,
) -> list[ShotBoundary]:
    """Build shot list from forbidden segments (Matlab Main.m shot-building section).

    Forbidden segments (0-based frame ranges, inclusive):
      - Abrupt at diff index j   → forbidden frame j+1
      - Gradual region [fs, fe]  → those frames are already in frame space

    Shots are the gaps between merged forbidden segments.
    """
    forbidden: list[tuple[int, int]] = []
    for j in abrupt_indices:
        forbidden.append((j + 1, j + 1))
    for fs, fe in gradual_regions:
        forbidden.append((fs, fe))

    if not forbidden:
        return [ShotBoundary(
            shot_id=1,
            start_frame=0,
            end_frame=total_frames - 1,
            start_time=0.0,
            end_time=(total_frames - 1) / fps,
            duration=(total_frames - 1) / fps,
        )]

    # Sort and merge overlapping / adjacent forbidden segments
    forbidden.sort(key=lambda x: x[0])
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = forbidden[0]
    for s, e in forbidden[1:]:
        if s <= cur_e + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # Fill gaps with shots
    shots: list[ShotBoundary] = []
    shot_id = 1
    current_start = 0

    for f_start, f_end in merged:
        shot_end = f_start - 1
        if current_start <= shot_end:
            shots.append(ShotBoundary(
                shot_id=shot_id,
                start_frame=current_start,
                end_frame=shot_end,
                start_time=current_start / fps,
                end_time=shot_end / fps,
                duration=(shot_end - current_start) / fps,
            ))
            shot_id += 1
        current_start = f_end + 1

    if current_start <= total_frames - 1:
        shots.append(ShotBoundary(
            shot_id=shot_id,
            start_frame=current_start,
            end_frame=total_frames - 1,
            start_time=current_start / fps,
            end_time=(total_frames - 1) / fps,
            duration=(total_frames - 1 - current_start) / fps,
        ))

    return shots


# ── public class ──────────────────────────────────────────────────────────────

class ShotSegmenter:
    """Detect shot boundaries in a video using the original Matlab algorithm."""

    def __init__(self, config: Optional[ShotSegmentationConfig] = None):
        self.config = config or ShotSegmentationConfig()

    def compute_frame_differences(self, video_path: str) -> np.ndarray:
        """Compute per-frame normalized histogram L1 distances using multiprocessing.

        Returns array of length total_frames-1 where diffs[j] = L1 distance
        between frames j and j+1.
        """
        props = get_video_properties(video_path)
        total_frames = props["total_frames"]
        num_processes = self.config.num_processes or cpu_count()
        num_processes = min(num_processes, total_frames // 2)

        batch_size = total_frames // num_processes
        process_args = []
        for i in range(num_processes):
            actual_start = i * batch_size
            end = (i + 1) * batch_size if i < num_processes - 1 else total_frames
            # Non-first batches read one extra frame before their start so the
            # cross-batch boundary diff (between frames actual_start-1 and actual_start)
            # is correctly computed.
            read_start = max(0, actual_start - 1) if i > 0 else 0
            process_args.append((read_start, actual_start, end, video_path, total_frames))

        logger.info("Computing histogram differences with %d processes...", num_processes)
        with Pool(num_processes) as pool:
            results = pool.map(_histogram_diff_batch, process_args)

        all_diffs: list[tuple[int, float]] = []
        for batch in results:
            all_diffs.extend(batch)
        all_diffs.sort(key=lambda x: x[0])

        return np.array([d for _, d in all_diffs])

    def detect_abrupt_transitions(self, diffs: np.ndarray) -> list[int]:
        """Detect abrupt cuts. Returns 0-based diff indices."""
        c = self.config
        boundaries = _detect_abrupt(
            diffs,
            threshold=c.abrupt_threshold,
            window_size=c.abrupt_window_size,
            neighborhood_factor=c.abrupt_neighborhood_factor,
        )
        logger.info("Abrupt transitions: %d", len(boundaries))
        return boundaries

    def detect_gradual_transitions(
        self, diffs: np.ndarray, abrupt_indices: list[int]
    ) -> list[tuple[int, int]]:
        """Detect gradual transitions. Returns (start_frame, end_frame) 0-based pairs."""
        c = self.config
        regions = _detect_gradual(
            diffs,
            abrupt_indices=abrupt_indices,
            filter_window=c.gradual_filter_window,
            min_peak_height=c.gradual_min_peak_height,
            min_peak_prominence=c.gradual_min_peak_prominence,
            local_threshold_factor=c.gradual_local_threshold_factor,
            local_window_ratio=c.gradual_local_window_ratio,
            min_duration=c.gradual_min_duration,
            min_distance_from_abrupt=c.gradual_min_distance_from_abrupt,
            min_contrast_factor=c.gradual_min_contrast_factor,
        )
        logger.info("Gradual transitions: %d regions", len(regions))
        return regions

    def run(self, video_path: str) -> tuple[list[ShotBoundary], np.ndarray]:
        """Run full shot segmentation pipeline.

        Returns:
            shots: List of ShotBoundary objects
            diffs: Raw per-frame L1 histogram difference signal
        """
        props = get_video_properties(video_path)
        fps = props["fps"]
        total_frames = props["total_frames"]

        diffs = self.compute_frame_differences(video_path)
        abrupt = self.detect_abrupt_transitions(diffs)
        gradual = self.detect_gradual_transitions(diffs, abrupt)
        shots = _build_shots(abrupt, gradual, total_frames, fps)

        logger.info(
            "Shot segmentation: %d shots from %d abrupt + %d gradual transitions",
            len(shots), len(abrupt), len(gradual),
        )
        return shots, diffs

    @staticmethod
    def shots_to_dataframe(shots: list[ShotBoundary]) -> pd.DataFrame:
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
