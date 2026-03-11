#!/usr/bin/env python3
"""Generate visual debug outputs for each pipeline component.

Produces human-inspectable images and short video clips so you can verify
each stage works correctly.

Usage:
    python scripts/generate_debug_visuals.py --match-id Match1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator
from soccer_summarization.utils import format_time, get_video_properties

logger = logging.getLogger(__name__)


def load_match_info(match_id: str, config: PipelineConfig) -> dict:
    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        data = yaml.safe_load(f)
    info = data["matches"][match_id]
    info["video_path"] = str(Path(config.data_dir) / info["video_path"])
    info["gt_path"] = str(Path(config.data_dir) / info["gt_path"])
    return info


def extract_frame(cap, frame_idx):
    """Extract a single frame from video capture."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# ─── 1. Shot Boundary Detection ─────────────────────────────────────────

def viz_shot_boundary_grid(video_path, shots_df, debug_dir, fps, max_shots=40):
    """Grid of first frames at each shot boundary."""
    logger.info("Generating shot boundary grid...")
    cap = cv2.VideoCapture(video_path)

    # Sample evenly if too many shots
    if len(shots_df) > max_shots:
        indices = np.linspace(0, len(shots_df) - 1, max_shots, dtype=int)
        sample = shots_df.iloc[indices]
    else:
        sample = shots_df

    n = len(sample)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2))
    axes = np.array(axes).flatten()

    for idx, (_, shot) in enumerate(sample.iterrows()):
        frame = extract_frame(cap, shot["start_frame"])
        ax = axes[idx]
        if frame is not None:
            ax.imshow(frame)
        ax.set_title(f'Shot {int(shot["shot_id"])}\n{format_time(shot["start_time"])}',
                     fontsize=7)
        ax.axis('off')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    cap.release()
    fig.suptitle(f'Shot Boundaries ({len(shots_df)} total, showing {n})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(debug_dir / "shot_boundary_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved shot_boundary_grid.png")


def viz_shot_diff_signal(shots_df, debug_dir, video_path):
    """Frame difference signal with detected boundaries and threshold."""
    from soccer_summarization.shot_segmentation import ShotSegmenter
    from soccer_summarization.config import ShotSegmentationConfig

    logger.info("Generating frame difference signal plot...")
    segmenter = ShotSegmenter(ShotSegmentationConfig())
    diffs = segmenter.compute_frame_differences(video_path)

    props = get_video_properties(video_path)
    fps = props["fps"]
    times = np.arange(len(diffs)) / fps

    # Compute threshold
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)
    adaptive = mean_diff + 2.0 * std_diff
    threshold = max(30.0, adaptive)

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.plot(times / 60, diffs, linewidth=0.3, color='#2C3E50', alpha=0.7)
    ax.axhline(threshold, color='#E74C3C', linestyle='--', linewidth=1.5,
               label=f'Threshold = {threshold:.1f}')

    # Mark boundaries
    for _, shot in shots_df.iterrows():
        if shot["start_frame"] > 0:
            ax.axvline(shot["start_time"] / 60, color='#E74C3C', alpha=0.15, linewidth=0.5)

    ax.set_xlabel('Time (minutes)', fontsize=11)
    ax.set_ylabel('Frame Difference (MAD)', fontsize=11)
    ax.set_title(f'Frame Difference Signal — {len(shots_df)} shots detected', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_xlim(0, times[-1] / 60)
    plt.tight_layout()
    plt.savefig(debug_dir / "frame_diff_signal.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved frame_diff_signal.png")


# ─── 2. YOLO Object Detection ───────────────────────────────────────────

def viz_yolo_detection_grid(video_path, player_df, debug_dir, config):
    """Grid of frames with YOLO detection boxes drawn."""
    logger.info("Generating YOLO detection grid...")
    from ultralytics import YOLO

    model = YOLO(config.object_detection.model_path)
    cap = cv2.VideoCapture(video_path)
    props = get_video_properties(video_path)
    fps = props["fps"]

    # Pick 8 representative frames: 2 high-count, 2 medium, 2 low, 2 zero
    sorted_df = player_df.sort_values("player_count")
    n = len(sorted_df)
    picks = []
    # 2 from bottom (low/zero count)
    picks.extend(sorted_df.iloc[:2].index.tolist())
    # 2 from ~25th percentile
    picks.extend(sorted_df.iloc[n // 4: n // 4 + 2].index.tolist())
    # 2 from ~75th percentile
    picks.extend(sorted_df.iloc[3 * n // 4: 3 * n // 4 + 2].index.tolist())
    # 2 from top (highest count)
    picks.extend(sorted_df.iloc[-2:].index.tolist())

    CLASS_COLORS = {0: (255, 165, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0)}
    CLASS_NAMES = {0: "Ball", 1: "GK", 2: "Player", 3: "Referee"}

    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for i, pick_idx in enumerate(picks[:8]):
        row = player_df.iloc[pick_idx]
        frame_idx = int(row["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Run YOLO on this frame
        results = model.predict(frame, conf=config.object_detection.confidence_threshold,
                                verbose=False)
        annotated = frame.copy()

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_NAMES.get(cls, '?')} {conf:.1f}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        axes[i].imshow(annotated_rgb)
        axes[i].set_title(f'{format_time(row["time_seconds"])} — {int(row["player_count"])} players',
                         fontsize=9)
        axes[i].axis('off')

    cap.release()

    # Legend
    legend_patches = [mpatches.Patch(color=np.array(c) / 255, label=CLASS_NAMES[k])
                      for k, c in CLASS_COLORS.items()]
    fig.legend(handles=legend_patches, loc='lower center', ncol=4, fontsize=10)
    fig.suptitle('YOLO Detections — Representative Frames', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(debug_dir / "yolo_detection_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved yolo_detection_grid.png")


def viz_yolo_detection_video(video_path, debug_dir, config, start_sec=900, duration_sec=30):
    """Short video clip with YOLO detection overlays."""
    logger.info("Generating YOLO detection video clip (%.0fs from %.0fs)...", duration_sec, start_sec)
    from ultralytics import YOLO

    model = YOLO(config.object_detection.model_path)
    cap = cv2.VideoCapture(video_path)
    props = get_video_properties(video_path)
    fps = props["fps"]

    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = str(debug_dir / "yolo_detection_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    CLASS_COLORS = {0: (0, 165, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 255, 255)}
    CLASS_NAMES = {0: "Ball", 1: "GK", 2: "Player", 3: "Referee"}

    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO every frame for the short clip
        results = model.predict(frame, conf=config.object_detection.confidence_threshold,
                                verbose=False)
        people_count = 0
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = CLASS_COLORS.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                if cls in {1, 2, 3}:
                    people_count += 1

        # Overlay info
        t = frame_idx / fps
        text = f"t={format_time(t)} | Players: {people_count}"
        cv2.putText(frame, text, (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    logger.info("  Saved yolo_detection_clip.mp4")


# ─── 3. Play/No-Play Classification ─────────────────────────────────────

def viz_play_timeline(player_df, play_df, gt_events, debug_dir, total_duration):
    """Full-match timeline with player count, play/no-play, and GT goals."""
    logger.info("Generating play/no-play timeline...")

    fig, axes = plt.subplots(3, 1, figsize=(22, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1, 0.5], 'hspace': 0.05})

    # Panel 1: Player count + play/no-play shading
    ax1 = axes[0]
    for _, seg in play_df.iterrows():
        color = '#C8E6C9' if seg['is_play'] else '#FFCDD2'
        ax1.axvspan(seg['start_time'] / 60, seg['end_time'] / 60, alpha=0.4, color=color)

    ax1.plot(player_df['time_seconds'] / 60, player_df['player_count'],
             linewidth=0.3, color='#2C3E50', alpha=0.5)

    # Smoothed
    window = min(50, len(player_df))
    smoothed = player_df['player_count'].rolling(window=window, center=True, min_periods=1).mean()
    ax1.plot(player_df['time_seconds'] / 60, smoothed, linewidth=1.2, color='#E74C3C')
    ax1.axhline(5, color='#F39C12', linestyle='--', linewidth=1, alpha=0.8)

    ax1.set_ylabel('Player Count', fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.5, 25)
    ax1.legend(handles=[
        mpatches.Patch(color='#C8E6C9', alpha=0.5, label='Play'),
        mpatches.Patch(color='#FFCDD2', alpha=0.5, label='No-play'),
        plt.Line2D([0], [0], color='#E74C3C', lw=1.2, label='Smoothed count'),
        plt.Line2D([0], [0], color='#F39C12', lw=1, ls='--', label='Threshold (5)'),
    ], loc='upper right', fontsize=8)

    # Panel 2: Play segments as blocks
    ax2 = axes[1]
    for _, seg in play_df.iterrows():
        if seg['is_play']:
            ax2.axvspan(seg['start_time'] / 60, seg['end_time'] / 60,
                       alpha=0.7, color='#5BA55B')
    ax2.set_ylabel('Play', fontsize=10, fontweight='bold')
    ax2.set_yticks([])

    # Panel 3: GT goals
    ax3 = axes[2]
    for ev in gt_events:
        ax3.axvspan(ev.start_time / 60, ev.end_time / 60, alpha=0.7, color='#E74C3C')
    ax3.set_ylabel('GT Goals', fontsize=10, fontweight='bold')
    ax3.set_yticks([])
    ax3.set_xlabel('Time (minutes)', fontsize=11)
    ax3.set_xlim(0, total_duration / 60)

    fig.suptitle('Play/No-Play Classification — Full Match',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(debug_dir / "play_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved play_timeline.png")


def viz_play_video_clip(video_path, play_df, player_df, debug_dir, fps,
                        start_sec=900, duration_sec=60):
    """Short video with PLAY/NO-PLAY overlay."""
    logger.info("Generating play/no-play video clip...")
    cap = cv2.VideoCapture(video_path)
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = str(debug_dir / "play_classification_clip.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Precompute play status per frame
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        # Find which play segment this frame belongs to
        is_play = False
        for _, seg in play_df.iterrows():
            if seg['start_time'] <= t <= seg['end_time']:
                is_play = bool(seg['is_play'])
                break

        # Find nearest player count
        closest = player_df.iloc[(player_df['time_seconds'] - t).abs().argsort()[:1]]
        pcount = int(closest['player_count'].values[0]) if len(closest) > 0 else 0

        # Overlay
        label = "PLAY" if is_play else "NO-PLAY"
        color = (0, 200, 0) if is_play else (0, 0, 200)
        cv2.rectangle(frame, (5, 5), (250, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"{label} | Players: {pcount}", (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, format_time(t), (width - 100, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    logger.info("  Saved play_classification_clip.mp4")


# ─── 4. OCR Scoreboard Detection ────────────────────────────────────────

def viz_scoreboard_detection(video_path, score_events_df, debug_dir, config, match_dir):
    """Variance map, detected region, and score timeline."""
    from soccer_summarization.ocr_scoreboard import ScoreboardDetector

    logger.info("Generating scoreboard detection visuals...")
    detector = ScoreboardDetector(config.ocr_scoreboard)

    # Build variance map
    variance_map = detector.find_static_overlays(video_path)

    # Load scoreboard region if saved
    region_file = match_dir / "scoreboard_region.json"
    region = None
    if region_file.exists():
        with open(region_file) as f:
            r = json.load(f)
        from soccer_summarization.ocr_scoreboard import ScoreboardRegion
        region = ScoreboardRegion(**r)

    # Get a sample frame
    props = get_video_properties(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, props["total_frames"] // 2)
    ret, frame = cap.read()
    cap.release()
    sample_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

    # Figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Panel 1: Sample frame with scoreboard region
    if sample_rgb is not None:
        axes[0].imshow(sample_rgb)
        if region:
            import matplotlib.patches as patches
            rect = patches.Rectangle((region.x, region.y), region.w, region.h,
                                     linewidth=3, edgecolor='red', facecolor='none')
            axes[0].add_patch(rect)
            axes[0].set_title(f'Detected Scoreboard at ({region.x},{region.y},{region.w}x{region.h})',
                             fontsize=10, fontweight='bold')
        else:
            axes[0].set_title('Sample Frame (no scoreboard detected)', fontsize=10)
        axes[0].axis('off')

    # Panel 2: Variance map
    var_display = np.log1p(variance_map)
    im = axes[1].imshow(var_display, cmap='hot')
    axes[1].set_title('Per-Pixel Variance Map (log scale)', fontsize=10, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Score timeline
    ax3 = axes[2]
    if len(score_events_df) > 0:
        times = score_events_df['timestamp'] / 60
        ax3.step(times, score_events_df['home_score'], where='post',
                 label='Home', color='#4A90D9', linewidth=2)
        ax3.step(times, score_events_df['away_score'], where='post',
                 label='Away', color='#E74C3C', linewidth=2)
        for _, ev in score_events_df.iterrows():
            ax3.axvline(ev['timestamp'] / 60, color='#F39C12', alpha=0.5, linestyle='--')
        ax3.legend(fontsize=9)
        ax3.set_title(f'Score Timeline — {len(score_events_df)} events detected',
                     fontsize=10, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No score events detected', ha='center', va='center',
                fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Score Timeline', fontsize=10, fontweight='bold')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Score')

    fig.suptitle('OCR Scoreboard Detection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(debug_dir / "scoreboard_detection.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved scoreboard_detection.png")


# ─── 5. Highlight Assembly ───────────────────────────────────────────────

def viz_full_match_timeline(shots_df, player_df, play_df, highlights_df, gt_events,
                            debug_dir, total_duration):
    """Full-match overview: shots + play + highlights + GT goals."""
    logger.info("Generating full-match timeline...")

    fig, axes = plt.subplots(4, 1, figsize=(22, 10), sharex=True,
                              gridspec_kw={'height_ratios': [1, 2.5, 1, 0.5], 'hspace': 0.06})

    t_max = total_duration / 60

    # Panel 1: Shots
    ax1 = axes[0]
    for _, shot in shots_df.iterrows():
        color = '#4A90D9' if int(shot['shot_id']) % 2 == 0 else '#7DB8E0'
        ax1.axvspan(shot['start_time'] / 60, shot['end_time'] / 60,
                    alpha=0.5, color=color, linewidth=0)
    ax1.set_ylabel('Shots', fontsize=11, fontweight='bold')
    ax1.set_yticks([])
    ax1.text(t_max * 1.01, 0.5, f'{len(shots_df)} shots', va='center', fontsize=8,
             transform=ax1.get_yaxis_transform())

    # Panel 2: Player count
    ax2 = axes[1]
    for _, seg in play_df.iterrows():
        color = '#C8E6C9' if seg['is_play'] else '#FFCDD2'
        ax2.axvspan(seg['start_time'] / 60, seg['end_time'] / 60, alpha=0.35, color=color)

    ax2.plot(player_df['time_seconds'] / 60, player_df['player_count'],
             linewidth=0.3, color='#2C3E50', alpha=0.4)
    window = min(50, len(player_df))
    smoothed = player_df['player_count'].rolling(window=window, center=True, min_periods=1).mean()
    ax2.plot(player_df['time_seconds'] / 60, smoothed, linewidth=1, color='#E74C3C')
    ax2.axhline(5, color='#F39C12', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Players', fontsize=11, fontweight='bold')
    ax2.set_ylim(-0.5, 25)
    ax2.legend(handles=[
        mpatches.Patch(color='#C8E6C9', alpha=0.5, label='Play'),
        mpatches.Patch(color='#FFCDD2', alpha=0.5, label='No-play'),
    ], loc='upper right', fontsize=8)

    # Panel 3: Highlights
    ax3 = axes[2]
    for _, hl in highlights_df.iterrows():
        ax3.axvspan(hl['start_time'] / 60, hl['end_time'] / 60,
                    alpha=0.7, color='#8E44AD')
    ax3.set_ylabel('Highlights', fontsize=11, fontweight='bold')
    ax3.set_yticks([])
    total_hl = highlights_df['duration'].sum()
    ax3.text(t_max * 1.01, 0.5,
             f'{len(highlights_df)} segs\n{total_hl:.0f}s',
             va='center', fontsize=7, transform=ax3.get_yaxis_transform())

    # Panel 4: GT goals
    ax4 = axes[3]
    for ev in gt_events:
        ax4.axvspan(ev.start_time / 60, ev.end_time / 60, alpha=0.7, color='#E74C3C')
    ax4.set_ylabel('GT Goals', fontsize=11, fontweight='bold')
    ax4.set_yticks([])
    ax4.set_xlabel('Time (minutes)', fontsize=12)
    ax4.set_xlim(0, t_max)

    fig.suptitle('Full Match Overview — Pipeline Components',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(debug_dir / "full_match_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved full_match_timeline.png")


# ─── 6. Scoreboard Crop Close-Up ────────────────────────────────────────

def viz_scoreboard_closeup(video_path, match_dir, debug_dir):
    """Show cropped scoreboard region at several timestamps."""
    region_file = match_dir / "scoreboard_region.json"
    if not region_file.exists():
        logger.warning("No scoreboard_region.json, skipping close-up")
        return

    with open(region_file) as f:
        r = json.load(f)

    logger.info("Generating scoreboard close-up grid...")
    cap = cv2.VideoCapture(video_path)
    props = get_video_properties(video_path)
    total_frames = props["total_frames"]
    fps = props["fps"]

    # Sample 12 frames evenly across the match
    sample_frames = np.linspace(100, total_frames - 100, 12, dtype=int)
    pad = 15

    fig, axes = plt.subplots(2, 6, figsize=(18, 4))
    axes = axes.flatten()

    for i, fidx in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
        ret, frame = cap.read()
        if not ret:
            continue

        y1 = max(0, r['y'] - pad)
        y2 = min(frame.shape[0], r['y'] + r['h'] + pad)
        x1 = max(0, r['x'] - pad)
        x2 = min(frame.shape[1], r['x'] + r['w'] + pad)
        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        axes[i].imshow(crop_rgb)
        axes[i].set_title(f'{format_time(fidx / fps)}', fontsize=8)
        axes[i].axis('off')

    cap.release()
    fig.suptitle('Scoreboard Crops Over Time', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(debug_dir / "scoreboard_closeup_grid.png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("  Saved scoreboard_closeup_grid.png")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate debug visual outputs")
    parser.add_argument("--match-id", required=True, help="Match ID (e.g., Match1)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--yolo-start", type=float, default=None,
                        help="Start time (seconds) for YOLO video clip")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    config = PipelineConfig.from_yaml(args.config)
    info = load_match_info(args.match_id, config)
    video_path = info["video_path"]
    gt_path = info["gt_path"]

    match_dir = Path(config.results_dir) / "per_match" / args.match_id
    debug_dir = match_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Check intermediates exist
    required = ["shot_boundaries.csv", "player_counts.csv", "play_segments.csv"]
    for f in required:
        if not (match_dir / f).exists():
            logger.error("Missing %s — run the pipeline first: python scripts/run_single_match.py --match-id %s",
                         f, args.match_id)
            return

    # Load data
    shots_df = pd.read_csv(match_dir / "shot_boundaries.csv")
    player_df = pd.read_csv(match_dir / "player_counts.csv")
    play_df = pd.read_csv(match_dir / "play_segments.csv")
    highlights_df = pd.read_csv(match_dir / "highlights.csv")
    score_path = match_dir / "score_events.csv"
    try:
        score_events_df = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        score_events_df = pd.DataFrame()

    props = get_video_properties(video_path)
    total_duration = props["duration"]
    fps = props["fps"]

    # Load GT goal events
    evaluator = Evaluator(config.evaluation)
    gt_df = evaluator.load_ground_truth(gt_path)
    gt_events = evaluator.extract_goal_events(gt_df)

    logger.info("Match: %s (%s vs %s, %s)", args.match_id,
                info["home_team"], info["away_team"], info["score"])
    logger.info("Duration: %.1f min, %d shots, %d GT goals",
                total_duration / 60, len(shots_df), len(gt_events))
    logger.info("Generating debug visuals to %s", debug_dir)

    # Generate all visualizations
    viz_shot_boundary_grid(video_path, shots_df, debug_dir, fps)
    viz_shot_diff_signal(shots_df, debug_dir, video_path)
    viz_yolo_detection_grid(video_path, player_df, debug_dir, config)

    # YOLO video clip — start near first goal if possible
    yolo_start = args.yolo_start
    if yolo_start is None and gt_events:
        yolo_start = max(0, gt_events[0].start_time - 15)
    elif yolo_start is None:
        yolo_start = total_duration / 3
    viz_yolo_detection_video(video_path, debug_dir, config, start_sec=yolo_start, duration_sec=30)

    viz_play_timeline(player_df, play_df, gt_events, debug_dir, total_duration)

    # Play video clip near first goal transition
    play_start = yolo_start - 30 if yolo_start > 30 else 0
    viz_play_video_clip(video_path, play_df, player_df, debug_dir, fps,
                        start_sec=play_start, duration_sec=60)

    viz_scoreboard_detection(video_path, score_events_df, debug_dir, config, match_dir)
    viz_scoreboard_closeup(video_path, match_dir, debug_dir)
    viz_full_match_timeline(shots_df, player_df, play_df, highlights_df, gt_events,
                            debug_dir, total_duration)

    logger.info("All debug visuals generated in %s", debug_dir)
    logger.info("Files:")
    for f in sorted(debug_dir.iterdir()):
        logger.info("  %s (%.1f KB)", f.name, f.stat().st_size / 1024)


if __name__ == "__main__":
    main()
