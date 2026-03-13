#!/usr/bin/env python3
"""Visualize OCR scoreboard diagnostics for human verification.

For each match, generates a multi-panel figure showing:
  - Panel 1: Std map with overlay mask and detected scoreboard region
  - Panels 2-N: Sample frames (10 by default) with scoreboard highlighted + OCR annotations
  - Panel N+1: Raw digit readings per position cluster over time (before filtering)
  - Panel N+2: Filtered readings after temporal median filter + majority voting
  - Panel N+3: Detected score timeline with GT goal events

Usage:
    python scripts/visualize_ocr_diagnostics.py                    # All matches
    python scripts/visualize_ocr_diagnostics.py --matches Match1,Match6
    python scripts/visualize_ocr_diagnostics.py --matches Match1 --annotated-frames 5
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
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
from soccer_summarization.ocr_scoreboard import (
    CLOCK_PATTERN, SCORE_WITH_SEP, TEAM_PATTERNS,
    ScoreboardDetector, ScoreboardRegion,
)
from soccer_summarization.utils import get_video_properties

logger = logging.getLogger(__name__)


def load_match_info(match_id: str, config: PipelineConfig) -> dict:
    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        data = yaml.safe_load(f)
    info = data["matches"][match_id]
    info["video_path"] = str(Path(config.data_dir) / info["video_path"])
    info["gt_path"] = str(Path(config.data_dir) / info["gt_path"])
    return info


def collect_ocr_readings(video_path: str, region: ScoreboardRegion, config):
    """Re-run OCR collection and return all raw readings with positions."""
    detector = ScoreboardDetector(config.ocr_scoreboard)
    props = get_video_properties(video_path)
    fps = props["fps"]
    total_frames = props["total_frames"]
    interval_frames = int(config.ocr_scoreboard.score_sample_interval * fps)

    ocr = detector._get_ocr()
    cap = cv2.VideoCapture(video_path)

    all_readings = []
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        crop = detector._crop_region(frame, region, pad=15)
        if crop.size == 0:
            frame_idx += interval_frames
            continue
        scale = 4
        upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        results = ocr.readtext(upscaled)
        for bbox, text, conf in results:
            cx = (bbox[0][0] + bbox[2][0]) / 2 / scale
            cy = (bbox[0][1] + bbox[2][1]) / 2 / scale
            all_readings.append((frame_idx, text.strip(), float(cx), float(cy), float(conf)))
        frame_idx += interval_frames

    cap.release()
    return all_readings, fps


def extract_digit_clusters(readings, fps):
    """Extract digit clusters and return raw + filtered per-frame readings."""
    by_frame = {}
    for frame_idx, text, cx, cy, conf in readings:
        if frame_idx not in by_frame:
            by_frame[frame_idx] = []
        by_frame[frame_idx].append((text, cx, cy, conf))

    # Collect single-digit readings
    digit_readings = []
    clock_positions = []
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
        return None

    # Filter near clock
    clock_cx = np.median([cx for cx, cy in clock_positions]) if clock_positions else None
    clock_cy = np.median([cy for cx, cy in clock_positions]) if clock_positions else None
    if clock_cx is not None:
        digit_readings = [
            (f, d, cx, cy) for f, d, cx, cy in digit_readings
            if not (abs(cx - clock_cx) < 15 and abs(cy - clock_cy) < 15)
        ]

    # Cluster
    detector = ScoreboardDetector()
    clusters = detector._cluster_by_position(digit_readings, radius=12)
    clusters.sort(key=lambda c: len(c[2]), reverse=True)

    # Filter clusters (same logic as production code)
    score_clusters = []
    for cx, cy, members in clusters:
        if len(members) < 10:
            continue
        digits = [d for _, d in members]
        digit_counts = Counter(digits)
        most_common_digit, most_common_count = digit_counts.most_common(1)[0]
        concentration = most_common_count / len(members)
        if concentration < 0.20:
            continue
        zero_count = digit_counts.get(0, 0)
        if (most_common_digit != 0
                and most_common_count / len(members) > 0.90
                and zero_count / len(members) < 0.05):
            continue
        score_clusters.append((cx, cy, members))

    if len(score_clusters) < 2:
        return None

    # Determine home/away
    cluster_a = score_clusters[0]
    cluster_b = score_clusters[1]
    dx = abs(cluster_a[0] - cluster_b[0])
    dy = abs(cluster_a[1] - cluster_b[1])

    if dy > dx:
        if cluster_a[1] < cluster_b[1]:
            home_cluster, away_cluster = cluster_a, cluster_b
        else:
            home_cluster, away_cluster = cluster_b, cluster_a
        layout = "vertical"
    else:
        if cluster_a[0] < cluster_b[0]:
            home_cluster, away_cluster = cluster_a, cluster_b
        else:
            home_cluster, away_cluster = cluster_b, cluster_a
        layout = "horizontal"

    # Raw readings
    home_raw = {f: d for f, d in home_cluster[2]}
    away_raw = {f: d for f, d in away_cluster[2]}

    # Filtered readings
    home_filtered = detector._temporal_median_filter(home_raw, window=15)
    away_filtered = detector._temporal_median_filter(away_raw, window=15)

    return {
        "home_raw": home_raw,
        "away_raw": away_raw,
        "home_filtered": home_filtered,
        "away_filtered": away_filtered,
        "home_pos": (home_cluster[0], home_cluster[1]),
        "away_pos": (away_cluster[0], away_cluster[1]),
        "layout": layout,
        "all_clusters": clusters,
        "score_clusters": score_clusters,
    }


def get_annotated_frames(video_path, region, n_frames=3):
    """Extract sample frames with scoreboard region highlighted and OCR annotations."""
    props = get_video_properties(video_path)
    total_frames = props["total_frames"]
    detector = ScoreboardDetector()
    ocr = detector._get_ocr()

    # Sample frames at 25%, 50%, 75% of video
    sample_positions = [int(total_frames * p) for p in np.linspace(0.25, 0.75, n_frames)]

    frames_out = []
    cap = cv2.VideoCapture(video_path)
    for frame_idx in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Draw scoreboard region on full frame
        annotated = frame.copy()
        x, y, w, h = int(region.x), int(region.y), int(region.w), int(region.h)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop and OCR the scoreboard
        crop = detector._crop_region(frame, region, pad=15)
        scale = 4
        upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        results = ocr.readtext(upscaled)

        # Draw OCR results on the crop (at upscaled size)
        crop_annotated = upscaled.copy()
        for bbox, text, conf in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(crop_annotated, [pts], True, (0, 255, 0), 2)
            cv2.putText(crop_annotated, f"{text} ({conf:.2f})",
                        (int(pts[0][0]), int(pts[0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        timestamp = frame_idx / props["fps"]
        frames_out.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "full_frame": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            "crop": cv2.cvtColor(crop_annotated, cv2.COLOR_BGR2RGB),
            "ocr_results": [(text, conf) for _, text, conf in results],
        })

    cap.release()
    return frames_out


def generate_diagnostic_figure(match_id, match_info, config, output_dir,
                               n_annotated_frames=10, reuse_readings=True):
    """Generate the full diagnostic figure for one match."""
    video_path = match_info["video_path"]
    gt_path = match_info["gt_path"]
    score = match_info["score"]
    results_dir = Path(f"results/per_match/{match_id}")

    # Load scoreboard region
    region_path = results_dir / "scoreboard_region.json"
    if not region_path.exists():
        print(f"  {match_id}: No scoreboard_region.json, skipping")
        return
    with open(region_path) as f:
        rj = json.load(f)
    region = ScoreboardRegion(x=rj["x"], y=rj["y"], w=rj["w"], h=rj["h"])

    # Load GT
    evaluator = Evaluator(config.evaluation)
    gt_df = evaluator.load_ground_truth(gt_path)
    gt_events = evaluator.extract_goal_events(gt_df)
    gt_times = [(e.start_time, e.end_time) for e in gt_events]

    # Load detected events
    score_events_path = results_dir / "score_events.csv"
    detected_events = []
    if score_events_path.exists():
        try:
            se_df = pd.read_csv(score_events_path)
            detected_events = list(se_df.itertuples(index=False))
        except (pd.errors.EmptyDataError, KeyError):
            pass

    props = get_video_properties(video_path)
    fps = props["fps"]
    total_duration = props["duration"]

    # Compute std map and overlay mask for visualization
    print(f"  Computing std map for {match_id}...")
    detector = ScoreboardDetector(config.ocr_scoreboard)
    std_map = detector.find_static_overlays(video_path)
    overlay_mask, raw_mask = detector.build_overlay_mask(std_map)

    print(f"  Collecting OCR readings for {match_id}...")
    readings, _ = collect_ocr_readings(video_path, region, config)
    cluster_data = extract_digit_clusters(readings, fps)

    print(f"  Extracting {n_annotated_frames} annotated frames...")
    annotated_frames = get_annotated_frames(video_path, region, n_annotated_frames)

    # --- Build figure ---
    n_frame_panels = len(annotated_frames)
    # 1 std map row + frame rows + 3 chart rows
    fig = plt.figure(figsize=(20, 6 + 2.5 * n_frame_panels + 9))

    # Title
    home_team = match_info.get("home_team", "Home")
    away_team = match_info.get("away_team", "Away")
    fig.suptitle(f"{match_id}: {home_team} {score} {away_team}",
                 fontsize=16, fontweight="bold", y=0.99)

    n_rows = 1 + n_frame_panels + 3  # std_map + frames + raw + filtered + timeline
    gs = fig.add_gridspec(n_rows, 2, hspace=0.4, wspace=0.3,
                          height_ratios=[3] + [2.5]*n_frame_panels + [2, 2, 2])

    # --- Panel 0: Std map + overlay mask ---
    ax_std = fig.add_subplot(gs[0, 0])
    ax_std.imshow(std_map, cmap="hot", vmin=0, vmax=np.percentile(std_map, 95))
    ax_std.set_title("Per-pixel IQR map (hot = high variability)", fontsize=10)
    ax_std.axis("off")
    # Draw detected scoreboard region
    from matplotlib.patches import Rectangle
    rect = Rectangle((region.x, region.y), region.w, region.h,
                      linewidth=2, edgecolor="cyan", facecolor="none")
    ax_std.add_patch(rect)

    ax_mask = fig.add_subplot(gs[0, 1])
    # Show overlay mask with scoreboard region highlighted
    mask_rgb = np.stack([overlay_mask]*3, axis=-1)
    # Highlight scoreboard region in green
    x, y, w, h = region.x, region.y, region.w, region.h
    mask_rgb[y:y+h, x:x+2, 1] = 255
    mask_rgb[y:y+h, x+w-2:x+w, 1] = 255
    mask_rgb[y:y+2, x:x+w, 1] = 255
    mask_rgb[y+h-2:y+h, x:x+w, 1] = 255
    ax_mask.imshow(mask_rgb)
    ax_mask.set_title(f"Overlay mask (P{config.ocr_scoreboard.std_percentile:.0f} threshold, "
                      f"closing {config.ocr_scoreboard.closing_kernel})", fontsize=10)
    ax_mask.axis("off")

    # --- Frame panels ---
    for i, fdata in enumerate(annotated_frames):
        ax_full = fig.add_subplot(gs[1 + i, 0])
        ax_full.imshow(fdata["full_frame"])
        ax_full.set_title(f"Frame {fdata['frame_idx']} ({fdata['timestamp']:.0f}s)",
                          fontsize=10)
        ax_full.axis("off")

        ax_crop = fig.add_subplot(gs[1 + i, 1])
        ax_crop.imshow(fdata["crop"])
        ocr_text = ", ".join([f'"{t}" ({c:.2f})' for t, c in fdata["ocr_results"]])
        ax_crop.set_title(f"Scoreboard crop (4x) — OCR: {ocr_text}", fontsize=9)
        ax_crop.axis("off")

    chart_start = 1 + n_frame_panels  # after std map + frame panels

    if cluster_data is None:
        # No clusters found — still show what we can
        ax_msg = fig.add_subplot(gs[chart_start:, :])
        ax_msg.text(0.5, 0.5, "No score digit clusters detected\n(too few digit readings)",
                    ha="center", va="center", fontsize=14, color="red")
        ax_msg.axis("off")
    else:
        # --- Panel: Raw readings ---
        ax_raw = fig.add_subplot(gs[chart_start, :])
        home_raw = cluster_data["home_raw"]
        away_raw = cluster_data["away_raw"]

        hr_t = [f / fps / 60 for f in sorted(home_raw.keys())]
        hr_v = [home_raw[f] for f in sorted(home_raw.keys())]
        ar_t = [f / fps / 60 for f in sorted(away_raw.keys())]
        ar_v = [away_raw[f] for f in sorted(away_raw.keys())]

        ax_raw.scatter(hr_t, hr_v, s=3, alpha=0.4, c="blue", label="Home raw")
        ax_raw.scatter(ar_t, ar_v, s=3, alpha=0.4, c="red", label="Away raw")

        # Shade GT events
        for start, end in gt_times:
            ax_raw.axvspan(start / 60, end / 60, alpha=0.15, color="green")

        ax_raw.set_ylabel("Digit value")
        ax_raw.set_title("Raw OCR digit readings per cluster (before filtering)", fontsize=11)
        ax_raw.legend(loc="upper left", fontsize=8)
        ax_raw.set_ylim(-0.5, max(max(hr_v + [0]), max(ar_v + [0])) + 1)
        ax_raw.set_xlim(0, total_duration / 60)
        ax_raw.grid(True, alpha=0.3)

        # --- Panel: Filtered readings ---
        ax_filt = fig.add_subplot(gs[chart_start + 1, :])
        home_filt = cluster_data["home_filtered"]
        away_filt = cluster_data["away_filtered"]

        hf_t = [f / fps / 60 for f in sorted(home_filt.keys())]
        hf_v = [home_filt[f] for f in sorted(home_filt.keys())]
        af_t = [f / fps / 60 for f in sorted(away_filt.keys())]
        af_v = [away_filt[f] for f in sorted(away_filt.keys())]

        ax_filt.plot(hf_t, hf_v, linewidth=1, c="blue", alpha=0.7, label="Home filtered")
        ax_filt.plot(af_t, af_v, linewidth=1, c="red", alpha=0.7, label="Away filtered")

        for start, end in gt_times:
            ax_filt.axvspan(start / 60, end / 60, alpha=0.15, color="green")

        ax_filt.set_ylabel("Score value")
        ax_filt.set_title("After temporal median filter (window=15)", fontsize=11)
        ax_filt.legend(loc="upper left", fontsize=8)
        ax_filt.set_ylim(-0.5, max(max(hf_v + [0]), max(af_v + [0])) + 1)
        ax_filt.set_xlim(0, total_duration / 60)
        ax_filt.grid(True, alpha=0.3)

        # --- Panel: Final score timeline ---
        ax_score = fig.add_subplot(gs[chart_start + 2, :])

        # Build stable score timeline from detected events
        event_times = []
        event_labels = []
        for ev in detected_events:
            t_min = ev.timestamp / 60
            event_times.append(t_min)
            event_labels.append(f"{ev.home_score}-{ev.away_score}")
            ax_score.axvline(t_min, color="purple", linewidth=2, alpha=0.7)
            ax_score.annotate(f"{ev.home_score}-{ev.away_score}",
                              (t_min, 0.85), xycoords=("data", "axes fraction"),
                              fontsize=9, color="purple", fontweight="bold",
                              ha="center", rotation=45)

        # GT events
        for i_gt, (start, end) in enumerate(gt_times):
            ax_score.axvspan(start / 60, end / 60, alpha=0.25, color="green",
                             label="GT goal" if i_gt == 0 else None)

        # Detected score line from filtered data
        ax_score.plot(hf_t, hf_v, linewidth=1.5, c="blue", alpha=0.5, label="Home")
        ax_score.plot(af_t, af_v, linewidth=1.5, c="red", alpha=0.5, label="Away")

        ax_score.set_xlabel("Time (minutes)")
        ax_score.set_ylabel("Score")
        ax_score.set_title(
            f"Detected events (purple lines) vs GT goals (green shading) — "
            f"Final: {score}, Detected: {len(detected_events)} events, GT: {len(gt_events)} goals",
            fontsize=11,
        )
        ax_score.legend(loc="upper left", fontsize=8)
        ax_score.set_ylim(-0.5, max(max(hf_v + [0]), max(af_v + [0])) + 1)
        ax_score.set_xlim(0, total_duration / 60)
        ax_score.grid(True, alpha=0.3)

    out_path = output_dir / f"{match_id}_ocr_diagnostics.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize OCR diagnostics")
    parser.add_argument("--matches", help="Comma-separated match IDs (default: all)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="results/ocr_diagnostics")
    parser.add_argument("--annotated-frames", type=int, default=10,
                        help="Number of annotated sample frames per match")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = PipelineConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        all_matches = list(yaml.safe_load(f)["matches"].keys())

    if args.matches:
        match_ids = [m.strip() for m in args.matches.split(",")]
    else:
        match_ids = all_matches

    for match_id in match_ids:
        print(f"\n{'='*60}")
        print(f"Processing {match_id}")
        print(f"{'='*60}")
        info = load_match_info(match_id, config)
        try:
            generate_diagnostic_figure(
                match_id, info, config, output_dir,
                n_annotated_frames=args.annotated_frames,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
