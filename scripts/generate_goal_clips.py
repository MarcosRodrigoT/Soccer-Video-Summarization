#!/usr/bin/env python3
"""Extract and compile moment-of-play clips for each detected goal.

For each match and each detected goal (from score_events.csv), this script
finds the last play-shot segment that immediately precedes the goal
(from highlights.csv and shot_boundaries.csv), extracts that clip from the
video at a reduced resolution, and concatenates all clips across all matches
into a single output video.

The "moment of play" is defined as: the last highlight segment (intersection of
a long shot with play activity) whose mid-point falls within the goal context
window [goal_time - context_before, goal_time].  This is the play action that
directly led to the goal.

Usage:
    python scripts/generate_goal_clips.py \
        --config config/default_hr.yaml \
        --matches-yaml config/matches_hr.yaml \
        --results-dir results_hr/per_match \
        --output results_hr/goal_clips/all_goals.mp4 \
        [--scale 0.5] [--fps 25] [--context-before 40] [--context-after 10]
"""

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator
from soccer_summarization.utils import get_video_properties, setup_logging

logger = logging.getLogger(__name__)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _find_moment_of_play(
    goal_time: float,
    highlights_df: pd.DataFrame,
    context_before: float = 40.0,
    context_after: float = 10.0,
) -> tuple[float, float]:
    """Return (start, end) of the best highlight clip for this goal.

    Searches for the last highlight segment whose end_time falls within
    [goal_time - context_before, goal_time + context_after].  If multiple
    segments qualify, returns the one whose end_time is closest to goal_time.
    Falls back to a raw context window if no highlight segment is found.
    """
    if highlights_df is None or highlights_df.empty:
        return (max(0.0, goal_time - context_before), goal_time + context_after)

    window_start = goal_time - context_before
    window_end   = goal_time + context_after

    candidates = highlights_df[
        (highlights_df["end_time"] >= window_start) &
        (highlights_df["start_time"] <= window_end)
    ].copy()

    if candidates.empty:
        return (max(0.0, goal_time - context_before), goal_time + context_after)

    # Pick the segment whose end_time is nearest to goal_time
    candidates["dist"] = abs(candidates["end_time"] - goal_time)
    best = candidates.sort_values("dist").iloc[0]
    return (float(best["start_time"]), float(best["end_time"]))


def _trim_trailing_short_shot(
    clip_start: float,
    clip_end: float,
    shots_df: pd.DataFrame,
    max_trailing_shot: float = 4.0,
    min_main_fraction: float = 0.20,
) -> tuple[float, float]:
    """Remove a brief trailing shot that bleeds in at the clip end.

    The highlight assembler merges adjacent play segments with up to a 0.5 s
    gap, which can attach a brief shot (camera cut to net/keeper/crowd) right
    after the main play action.  We trim the trailing shot only when it is
    clearly incidental — i.e. it is short in absolute terms AND it makes up
    less than min_main_fraction of the total clip duration.  This prevents
    trimming legitimate short goal shots that happen to appear at the end.
    """
    if shots_df is None or shots_df.empty:
        return clip_start, clip_end

    clip_shots = shots_df[
        (shots_df["end_time"] > clip_start) &
        (shots_df["start_time"] < clip_end)
    ].copy()

    if len(clip_shots) < 2:
        return clip_start, clip_end  # single shot — nothing to trim

    last = clip_shots.iloc[-1]
    last_start = float(last["start_time"])
    last_dur = clip_end - max(last_start, clip_start)  # duration within clip
    clip_dur = clip_end - clip_start

    # Only trim when:
    #  1. The trailing shot is short in absolute terms, AND
    #  2. It makes up less than min_main_fraction of the clip
    #     (so we don't remove a 5 s goal shot from a 6 s clip)
    if last_dur < max_trailing_shot and last_dur < min_main_fraction * clip_dur:
        new_end = max(last_start, clip_start + 1.0)
        if new_end > clip_start:
            return clip_start, new_end

    return clip_start, clip_end


def _extract_clip(
    video_path: str,
    start: float,
    end: float,
    output_path: str,
    scale: float = 0.5,
    target_fps: int = 25,
) -> bool:
    """Extract [start, end] from video_path at reduced resolution.

    Uses ffmpeg with stream-copy where possible, falls back to re-encode.
    Returns True on success.
    """
    duration = end - start
    if duration <= 0:
        return False

    # Determine output size: scale the source resolution
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-vf", f"scale=iw*{scale}:ih*{scale}",
        "-r", str(target_fps),
        "-c:v", "libx264", "-crf", "28", "-preset", "fast",
        "-an",                # no audio
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("ffmpeg failed for %s: %s", output_path,
                       result.stderr.decode("utf-8", errors="replace"))
        return False
    return True


def _add_text_overlay(
    input_path: str,
    output_path: str,
    text: str,
    font_size: int = 28,
) -> bool:
    """Burn a title text into the video using ffmpeg drawtext."""
    safe_text = text.replace("'", "\\'").replace(":", "\\:")
    filter_str = (
        f"drawtext=text='{safe_text}'"
        f":fontsize={font_size}"
        f":fontcolor=white"
        f":x=10:y=10"
        f":box=1:boxcolor=black@0.6:boxborderw=5"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", filter_str,
        "-c:v", "libx264", "-crf", "28", "-preset", "fast",
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def _concatenate_clips(clip_paths: list[str], output_path: str) -> bool:
    """Concatenate clips using ffmpeg concat demuxer."""
    if not clip_paths:
        return False

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in clip_paths:
            f.write(f"file '{Path(p).resolve()}'\n")
        list_file = f.name

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_file,
        "-c:v", "libx264", "-crf", "28", "-preset", "fast",
        "-an",
        "-loglevel", "error",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    Path(list_file).unlink(missing_ok=True)
    return result.returncode == 0


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract moment-of-play goal clips for all matches")
    parser.add_argument("--config", default="config/default_hr.yaml")
    parser.add_argument("--matches-yaml", default="config/matches_hr.yaml")
    parser.add_argument("--results-dir", default="results_hr/per_match",
                        help="Base dir containing per_match results")
    parser.add_argument("--output", default="results_hr/goal_clips/all_goals.mp4",
                        help="Output compilation video path")
    parser.add_argument("--matches", default=None,
                        help="Comma-separated match IDs (default: all)")
    parser.add_argument("--scale", type=float, default=0.5,
                        help="Resolution scale factor (0.5 = half resolution)")
    parser.add_argument("--fps", type=int, default=25,
                        help="Output frames per second")
    parser.add_argument("--context-before", type=float, default=40.0,
                        help="Seconds before goal to include in context window")
    parser.add_argument("--context-after", type=float, default=10.0,
                        help="Seconds after goal to include in context window")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = PipelineConfig.from_yaml(args.config)
    evaluator = Evaluator(config.evaluation)

    with open(args.matches_yaml) as f:
        all_matches = yaml.safe_load(f)["matches"]

    if args.matches:
        match_ids = [m.strip() for m in args.matches.split(",")]
    else:
        match_ids = sorted(all_matches.keys(),
                           key=lambda mid: int(re.search(r'\d+', mid).group()))

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Temp dir for individual clips
    clips_dir = output_path.parent / "clips_tmp"
    clips_dir.mkdir(exist_ok=True)

    all_clip_paths: list[str] = []
    total_goals = 0
    matched_goals = 0

    for match_id in match_ids:
        info = all_matches.get(match_id)
        if info is None:
            logger.warning("Match %s not in matches yaml, skipping", match_id)
            continue

        match_dir = results_dir / match_id
        video_path = str(Path(config.data_dir) / info["video_path"])
        goals_path = str(Path(config.data_dir) / info["goals_path"]) \
            if "goals_path" in info else None

        # ── Load detected goals ─────────────────────────────────────────────
        score_path = match_dir / "score_events.csv"
        if not score_path.exists():
            logger.warning("%s: no score_events.csv, skipping", match_id)
            continue
        try:
            score_df = pd.read_csv(score_path)
            if score_df.empty or "timestamp" not in score_df.columns:
                logger.warning("%s: empty score_events.csv", match_id)
                continue
        except Exception:
            continue

        # ── Load highlights and shot boundaries ─────────────────────────────
        hl_path = match_dir / "highlights.csv"
        highlights_df = pd.read_csv(hl_path) if hl_path.exists() else None

        shots_path = match_dir / "shot_boundaries.csv"
        shots_df = pd.read_csv(shots_path) if shots_path.exists() else None

        # ── Load GT goal clips for overlay info ─────────────────────────────
        gt_clips = []
        if goals_path and Path(goals_path).exists():
            try:
                gt_clips = Evaluator.load_goal_clips(goals_path, 25.0)
            except Exception:
                pass

        # ── Load match metadata ─────────────────────────────────────────────
        home = info.get("home_team", "Home")
        away = info.get("away_team", "Away")
        match_score = info.get("score", "?-?")

        logger.info("Processing %s (%s %s %s) — %d detected goals",
                    match_id, home, match_score, away, len(score_df))

        for i, row in score_df.iterrows():
            goal_time = float(row["timestamp"])
            home_score = int(row.get("home_score", 0))
            away_score = int(row.get("away_score", 0))
            total_goals += 1

            # Find the moment-of-play clip (best highlight segment)
            clip_start, clip_end = _find_moment_of_play(
                goal_time, highlights_df,
                context_before=args.context_before,
                context_after=args.context_after,
            )

            # Trim any trailing shot that bleeds in at the end.
            # The highlight assembler merges adjacent shots with <=0.5 s gap;
            # this can attach a short trailing shot (camera cut to net/keeper)
            # to the end of the main play shot.  Trim shots shorter than 6 s
            # at the clip end — visible as "1-2 seconds of the next shot".
            clip_start, clip_end = _trim_trailing_short_shot(
                clip_start, clip_end, shots_df)

            # Clamp to video duration if we know it
            try:
                props = get_video_properties(video_path)
                clip_end = min(clip_end, props["duration"])
            except Exception:
                pass

            if clip_end <= clip_start:
                logger.warning("%s goal %d: empty clip, skipping", match_id, i + 1)
                continue

            # Extract raw clip at reduced resolution
            raw_clip = str(clips_dir / f"{match_id}_goal{i+1:02d}_raw.mp4")
            if not _extract_clip(video_path, clip_start, clip_end, raw_clip,
                                  scale=args.scale, target_fps=args.fps):
                logger.warning("%s goal %d: extraction failed", match_id, i + 1)
                continue

            # Build overlay text
            mins = int(goal_time // 60)
            secs = int(goal_time % 60)
            label = (f"{match_id}: {home} {home_score}-{away_score} {away}  "
                     f"[Goal at {mins}:{secs:02d}]")

            # Burn text overlay
            labeled_clip = str(clips_dir / f"{match_id}_goal{i+1:02d}.mp4")
            if _add_text_overlay(raw_clip, labeled_clip, label):
                Path(raw_clip).unlink(missing_ok=True)
                all_clip_paths.append(labeled_clip)
                matched_goals += 1
                logger.info("  Goal %d @ %dm%ds  clip [%.1f→%.1f] OK",
                            i + 1, mins, secs, clip_start, clip_end)
            else:
                # Fall back to unlabeled clip
                all_clip_paths.append(raw_clip)
                matched_goals += 1

    # ── Concatenate all clips (already in match/goal order from the loop) ───
    if not all_clip_paths:
        logger.error("No clips extracted — nothing to concatenate")
        return

    # Sort numerically Match1→Match25, goal01→goalN to guarantee order
    def _clip_sort_key(p):
        m = re.search(r'Match(\d+)_goal(\d+)', Path(p).stem)
        return (int(m.group(1)), int(m.group(2))) if m else (9999, 9999)
    all_clip_paths.sort(key=_clip_sort_key)

    logger.info("Concatenating %d clips → %s", len(all_clip_paths), output_path)
    if not _concatenate_clips([str(p) for p in all_clip_paths], str(output_path)):
        logger.error("Concatenation failed")
        return

    # Clean up temp clips
    for p in all_clip_paths:
        Path(p).unlink(missing_ok=True)
    try:
        clips_dir.rmdir()
    except OSError:
        pass

    logger.info(
        "Done. Extracted %d/%d goal clips → %s",
        matched_goals, total_goals, output_path,
    )
    print(f"\nOutput: {output_path}")
    print(f"Goals extracted: {matched_goals}/{total_goals}")


if __name__ == "__main__":
    main()
