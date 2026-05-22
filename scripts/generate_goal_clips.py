#!/usr/bin/env python3
"""Extract and compile moment-of-play clips for each detected goal.

For each match and each detected goal (from score_events.csv), this script
selects the full play shot whose end aligns with the goal moment (from
shot_boundaries.csv + player_counts.csv), extracts that whole shot at a
reduced resolution, and concatenates all clips across all matches into a
single output video.

A "play shot" is a shot whose average per-frame player count is above the
PlayClassifier threshold.  Among play shots ending within
[goal - 30s, goal + 5s], the one whose end_time is closest to the goal is
selected and returned with its full, unchopped boundaries.

Usage:
    python scripts/generate_goal_clips.py \
        --config config/default_hr.yaml \
        --matches-yaml config/matches_hr.yaml \
        --results-dir results_hr/per_match \
        --output results_hr/goal_clips/all_goals.mp4 \
        [--scale 0.5] [--fps 25]
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

def _select_goal_shot(
    goal_time: float,
    shots_df: pd.DataFrame,
    player_counts_df: pd.DataFrame,
    player_threshold: float = 5.0,
    min_shot_duration: float = 4.0,
    block_max_gap: float = 2.0,
    block_max_lookback: float = 30.0,
    max_lookback: float = 30.0,
    max_lookahead: float = 5.0,
) -> tuple[float, float]:
    """Pick a play long shot anchor near the goal, then walk BACK over its
    immediately-adjacent play long shots (multi-angle coverage of the same
    play action) as long as the gap is small AND the block start stays within
    a bounded lookback window of the goal moment.

    A "play long shot" is one with duration >= min_shot_duration AND average
    per-frame player count >= player_threshold.  Falls back to "nearest shot
    of any kind" only if no play long shots exist.
    """
    shots = shots_df.copy().sort_values("start_time").reset_index(drop=True)
    avgs = []
    for _, s in shots.iterrows():
        mask = (player_counts_df["frame"] >= s["start_frame"]) & \
               (player_counts_df["frame"] <= s["end_frame"])
        vals = player_counts_df.loc[mask, "player_count"]
        avgs.append(float(vals.mean()) if len(vals) else 0.0)
    shots["avg_players"] = avgs
    play_mask = (shots["avg_players"] >= player_threshold) & \
                (shots["duration"] >= min_shot_duration)
    play_idx = list(shots[play_mask].index)

    # Step 1: anchor selection
    anchor: int = -1
    if play_idx:
        in_window = [i for i in play_idx
                     if (goal_time - max_lookback) <= shots.at[i, "end_time"] <= (goal_time + max_lookahead)]
        if in_window:
            anchor = min(in_window, key=lambda i: abs(shots.at[i, "end_time"] - goal_time))
        else:
            containing = [i for i in play_idx
                          if shots.at[i, "start_time"] <= goal_time <= shots.at[i, "end_time"]]
            if containing:
                anchor = containing[0]
            else:
                anchor = min(play_idx, key=lambda i: abs(shots.at[i, "end_time"] - goal_time))
    else:
        # No play long shots at all — fall back to nearest shot of any kind
        nearest = (shots.assign(dist=(shots["end_time"] - goal_time).abs())
                          .sort_values("dist").iloc[0])
        return float(nearest["start_time"]), float(nearest["end_time"])

    st = float(shots.at[anchor, "start_time"])
    et = float(shots.at[anchor, "end_time"])

    # Step 2: walk back over adjacent play long shots
    play_idx_sorted = sorted(play_idx)
    pos = play_idx_sorted.index(anchor)
    for prev_pos in range(pos - 1, -1, -1):
        prev_i = play_idx_sorted[prev_pos]
        prev_et = float(shots.at[prev_i, "end_time"])
        prev_st = float(shots.at[prev_i, "start_time"])
        if (st - prev_et) > block_max_gap:
            break
        if (goal_time - prev_st) > block_max_lookback:
            break
        st = prev_st  # extend the block backwards

    return st, et


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
    parser.add_argument(
        "--clip-max-duration", type=float, default=None,
        help="If set, keep only the last N seconds of each selected play shot "
             "(e.g. 10 for the 'last 10s' variant). Default: full shot.",
    )
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

        # ── Load shot boundaries and per-frame player counts ────────────────
        shots_path = match_dir / "shot_boundaries.csv"
        shots_df = pd.read_csv(shots_path) if shots_path.exists() else None

        pc_path = match_dir / "player_counts.csv"
        player_counts_df = pd.read_csv(pc_path) if pc_path.exists() else None

        if shots_df is None or shots_df.empty or player_counts_df is None:
            logger.warning("%s: missing shots or player_counts, skipping", match_id)
            continue

        player_threshold = float(config.play_classifier.play_threshold)
        min_shot_dur     = float(config.highlight_assembler.min_shot_duration)
        block_max_gap    = float(config.highlight_assembler.goal_block_max_gap)
        block_max_back   = float(config.highlight_assembler.goal_block_max_lookback)

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

            # Select the full play shot whose end aligns with the goal moment.
            # This returns whole shot boundaries — no clipping, no merging.
            clip_start, clip_end = _select_goal_shot(
                goal_time=goal_time,
                shots_df=shots_df,
                player_counts_df=player_counts_df,
                player_threshold=player_threshold,
                min_shot_duration=min_shot_dur,
                block_max_gap=block_max_gap,
                block_max_lookback=block_max_back,
            )

            # Optional: truncate to last N seconds of the selected shot
            if args.clip_max_duration is not None and (clip_end - clip_start) > args.clip_max_duration:
                clip_start = clip_end - args.clip_max_duration

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
