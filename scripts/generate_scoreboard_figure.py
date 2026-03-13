#!/usr/bin/env python3
"""Generate the scoreboard detection figure for the README.

Creates a 3-panel figure showing:
  1. Sample mid-match frame with detected scoreboard bbox
  2. IQR variability map (log scale) with overlay mask contour
  3. Static overlays mask + detected scoreboard highlighted

Usage:
    python scripts/generate_scoreboard_figure.py
    python scripts/generate_scoreboard_figure.py --match-id Match10
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.ocr_scoreboard import ScoreboardDetector
from soccer_summarization.utils import get_video_properties


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", default="Match10",
                        help="Match to use for the figure")
    parser.add_argument("--output", default="docs/figures/scoreboard_detection.png")
    args = parser.parse_args()

    config = PipelineConfig.from_yaml("config/default.yaml")

    matches_yaml = Path("config/matches.yaml")
    with open(matches_yaml) as f:
        data = yaml.safe_load(f)
    match_info = data["matches"][args.match_id]
    video_path = str(Path(config.data_dir) / match_info["video_path"])
    score = match_info["score"]
    home = match_info["home_team"]
    away = match_info["away_team"]
    title = f"{home} {score} {away}"

    # Run detection
    det = ScoreboardDetector(config.ocr_scoreboard)
    iqr_map = det.find_static_overlays(video_path)
    overlay_mask, raw_mask = det.build_overlay_mask(iqr_map)
    region = det.find_scoreboard_region(video_path, iqr_map, overlay_mask, raw_mask)

    if region is None:
        print(f"No scoreboard found for {args.match_id}")
        return

    # Compute display bbox
    dx, dy, dw, dh = det._tighten_to_raw(
        raw_mask, region.x, region.y, region.w, region.h, ignore_edges=True
    )

    # Get a mid-match frame
    props = get_video_properties(video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(props["total_frames"] * 0.5))
    ret, frame = cap.read()
    cap.release()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Create figure ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.0))
    fig.suptitle(f"Automatic Scoreboard Localization via IQR Variability Analysis",
                 fontsize=13, fontweight="bold", y=0.98)

    # Panel 1: Sample frame with bbox
    ax = axes[0]
    ax.imshow(frame_rgb)
    rect = mpatches.Rectangle((dx, dy), dw, dh, linewidth=2,
                               edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_title("Sample Frame (mid-match)", fontsize=10)
    ax.axis("off")

    # Panel 2: IQR map (log scale)
    ax = axes[1]
    iqr_display = iqr_map.copy().astype(float)
    iqr_display[iqr_display < 1] = 1  # avoid log(0)
    im = ax.imshow(np.log10(iqr_display), cmap="hot", aspect="auto")
    # Draw overlay mask contour
    contours, _ = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_squeezed = cnt.squeeze()
        if cnt_squeezed.ndim == 2 and len(cnt_squeezed) > 2:
            polygon = plt.Polygon(cnt_squeezed, fill=False,
                                   edgecolor="cyan", linewidth=1.0, linestyle="--")
            ax.add_patch(polygon)
    # Draw detected scoreboard bbox
    rect = mpatches.Rectangle((dx, dy), dw, dh, linewidth=2,
                               edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_title("Per-Pixel IQR Map (log scale)", fontsize=10)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log₁₀(IQR)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Panel 3: Overlay mask + detected scoreboard
    ax = axes[2]
    # Show frame dimmed with overlay mask highlighted
    frame_dimmed = (frame_rgb * 0.3).astype(np.uint8)
    overlay_colored = frame_dimmed.copy()
    overlay_colored[overlay_mask > 0] = frame_rgb[overlay_mask > 0]
    ax.imshow(overlay_colored)
    rect = mpatches.Rectangle((dx, dy), dw, dh, linewidth=2,
                               edgecolor="lime", facecolor="none")
    ax.add_patch(rect)
    ax.set_title("Static Overlays + Detected Scoreboard", fontsize=10)
    ax.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(edgecolor="lime", facecolor="none", linewidth=2,
                       label="Detected Scoreboard"),
    ]
    axes[2].legend(handles=legend_elements, loc="lower right", fontsize=7,
                   framealpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
