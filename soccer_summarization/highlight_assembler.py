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
        """Select one play BLOCK per goal — the block whose end aligns with the goal.

        Definitions:
          * "play long shot" = duration >= config.min_shot_duration AND average
            per-frame player count >= player_threshold (same as select_play_long_shots).
          * "play block" = maximal run of consecutive play long shots whose
            gaps are <= config.goal_block_max_gap seconds.  This captures
            multi-angle coverage of a single play action where the shot
            segmenter inserts brief transition frames between camera angles.

        For each goal time T, the block whose end_time is closest to T within
        the window [T - max_lookback, T + max_lookahead] is returned in full.
        Falls back progressively: containing-block → nearest block → nearest
        shot of any kind.  Always returns one segment per goal.
        """
        if not goal_times or not shots:
            return []

        min_dur     = self.config.min_shot_duration
        block_gap   = self.config.goal_block_max_gap
        block_back  = self.config.goal_block_max_lookback

        # Compute avg player count per shot
        pc = player_counts.copy()
        shot_avg: list[float] = []
        for s in shots:
            mask = (pc["frame"] >= s.start_frame) & (pc["frame"] <= s.end_frame)
            vals = pc.loc[mask, "player_count"]
            shot_avg.append(float(vals.mean()) if len(vals) else 0.0)

        # A play long shot must satisfy BOTH duration AND player-count thresholds —
        # this matches select_play_long_shots() and prevents 1-2s "transition"
        # shots between the play action and the goal celebration from winning the
        # nearest-end tiebreak.
        is_play = [
            (a >= player_threshold) and (shots[i].duration >= min_dur)
            for i, a in enumerate(shot_avg)
        ]
        play_shot_indices = sorted(i for i, p in enumerate(is_play) if p)

        result: list[HighlightSegment] = []
        n_anchors_extended = 0

        for g_idx, T in enumerate(goal_times):
            # Step 1 — pick an anchor: the single play long shot best matching T.
            anchor_idx: Optional[int] = None
            cands = [
                i for i in play_shot_indices
                if (T - max_lookback) <= shots[i].end_time <= (T + max_lookahead)
            ]
            if cands:
                anchor_idx = min(cands, key=lambda i: abs(shots[i].end_time - T))
            else:
                containing = [
                    i for i in play_shot_indices
                    if shots[i].start_time <= T <= shots[i].end_time
                ]
                if containing:
                    anchor_idx = containing[0]
                elif play_shot_indices:
                    anchor_idx = min(play_shot_indices, key=lambda i: abs(shots[i].end_time - T))
                else:
                    anchor_idx = min(range(len(shots)), key=lambda i: abs(shots[i].end_time - T))

            assert anchor_idx is not None
            a = shots[anchor_idx]
            sf, ef = a.start_frame, a.end_frame
            st, et = a.start_time, a.end_time

            # Step 2 — walk BACKWARDS from the anchor, merging immediately
            # adjacent play long shots that look like multi-angle coverage of
            # the same play action.  Two guards prevent over-merging:
            #   (a) gap between the candidate's end and the current block start
            #       must be <= goal_block_max_gap (small inter-camera cut),
            #   (b) T − candidate.start must stay <= goal_block_max_lookback
            #       (otherwise we'd be pulling in an earlier, unrelated play).
            if anchor_idx in set(play_shot_indices):
                # Iterate play long shots ending strictly before the anchor
                earlier = [i for i in play_shot_indices if i < anchor_idx]
                for i in reversed(earlier):
                    cand = shots[i]
                    if (st - cand.end_time) > block_gap:
                        break
                    if (T - cand.start_time) > block_back:
                        break
                    sf = cand.start_frame
                    st = cand.start_time
                    n_anchors_extended += 1

            # Optional: keep only the last N seconds of the selected play block
            # (e.g. "last 10s" variant).  None = keep the full block.
            max_dur = self.config.goal_clip_max_duration
            if max_dur is not None and (et - st) > max_dur:
                new_sf = ef - int(round(max_dur * fps))
                new_sf = max(new_sf, sf)
                new_st = new_sf / float(fps)
                result.append(HighlightSegment(
                    segment_id=g_idx + 1,
                    start_frame=new_sf,
                    end_frame=ef,
                    start_time=new_st,
                    end_time=et,
                    duration=et - new_st,
                ))
            else:
                result.append(HighlightSegment(
                    segment_id=g_idx + 1,
                    start_frame=sf,
                    end_frame=ef,
                    start_time=st,
                    end_time=et,
                    duration=et - st,
                ))

        suffix = (f" — truncated to last {self.config.goal_clip_max_duration:.0f}s"
                  if self.config.goal_clip_max_duration is not None else "")
        logger.info(
            "Selected %d goal clips from %d shots "
            "(%d play long shots; anchors extended back %d times "
            "with gap<=%.1fs, lookback<=%.1fs, thr=%.1f, min_dur=%.1fs)%s",
            len(result), len(shots), len(play_shot_indices),
            n_anchors_extended, block_gap, block_back,
            player_threshold, min_dur, suffix,
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
