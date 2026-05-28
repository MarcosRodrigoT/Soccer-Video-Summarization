"""Figure 13 — Qualitative YOLOv8 player detection on Match1.

The figure illustrates the *asymmetric* response of the detector that
specialises in long-range (small-silhouette) players: it fires densely on
wide-angle live-action frames and stays essentially silent on close-ups.
A 2x2 grid is rendered: the top row shows two long-range play frames
where the detector recovers many small-silhouette players, and the bottom
row shows two foreground/close-up frames (replays, dugout, post-goal
celebrations) where the same detector returns few or no detections.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from _style import save_figure, set_paper_style

PROJECT = Path(__file__).resolve().parents[2]
MATCH_DIR = PROJECT / "results_hr" / "per_match" / "Match1"
VIDEO = Path("/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos/Laliga_HR/Match1/"
             "Valencia CF - FC Barcelona (2-3) LALIGA 2013⧸2014 FULL MATCH "
             "[Qzrg3QDmQ4k].mp4")
MODEL_PATH = PROJECT / "data" / "models" / "best.pt"

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
CLASS_COLORS = {
    "ball":       "#E91E63",
    "goalkeeper": "#FF9800",
    "player":     "#2196F3",
    "referee":    "#9C27B0",
}


def _segment_center(seg) -> int:
    return int((seg["start_frame"] + seg["end_frame"]) / 2)


def pick_frames() -> tuple[list[int], list[int]]:
    """Return (long_shots, close_ups) frame indices for the 2x2 grid."""
    segs = pd.read_csv(MATCH_DIR / "play_segments.csv")
    segs["dur"] = segs["end_time"] - segs["start_time"]

    play = segs[(segs["is_play"] == True) & (segs["dur"] > 10.0)] \
            .sort_values("avg_player_count", ascending=False)
    long_shots = [_segment_center(play.iloc[i]) for i in (0, 3)]

    no_play = segs[(segs["is_play"] == False) &
                   (segs["avg_player_count"] < 1.0) &
                   (segs["dur"] > 5.0)].sort_values("dur", ascending=False)
    close_ups = [_segment_center(no_play.iloc[i]) for i in (0, 1)]

    return long_shots, close_ups


def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def _draw_detections(ax, frame_rgb, results) -> dict[str, int]:
    ax.imshow(frame_rgb)
    counts = {n: 0 for n in CLASS_NAMES.values()}
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        cls = results.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c in zip(xyxy, cls):
            name = CLASS_NAMES.get(int(c), "unknown")
            color = CLASS_COLORS.get(name, "white")
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor=color,
                                   linewidth=1.3))
            counts[name] = counts.get(name, 0) + 1
    ax.set_xticks([]); ax.set_yticks([])
    return counts


def _panel_title(ax, label: str, counts: dict[str, int]) -> None:
    people = counts.get("player", 0) + counts.get("goalkeeper", 0) + counts.get("referee", 0)
    sub = f"$n(t)={people}$"
    ax.set_title(f"{label}   ({sub})", loc="left",
                 fontsize=9.5, fontweight="bold")


def build_figure():
    set_paper_style()
    fig = plt.figure(figsize=(9.0, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.06)

    axes = [[fig.add_subplot(gs[r, c]) for c in range(2)] for r in range(2)]

    if not VIDEO.exists() or not MODEL_PATH.exists():
        for r in range(2):
            for c in range(2):
                axes[r][c].text(0.5, 0.5, "asset unavailable",
                                ha="center", transform=axes[r][c].transAxes)
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
        return fig

    long_shots, close_ups = pick_frames()

    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))

    cap = cv2.VideoCapture(str(VIDEO))

    panel_labels = [
        ["(a) long range, play",   "(b) long range, play"],
        ["(c) foreground, no-play", "(d) foreground, no-play"],
    ]
    frame_sets = [long_shots, close_ups]

    for r, frames in enumerate(frame_sets):
        for c, fidx in enumerate(frames):
            frame_rgb = _read_frame(cap, fidx)
            if frame_rgb is None:
                axes[r][c].text(0.5, 0.5, "frame unavailable",
                                ha="center", transform=axes[r][c].transAxes)
                axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
                continue
            res = model.predict(source=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                                conf=0.5, verbose=False)[0]
            counts = _draw_detections(axes[r][c], frame_rgb, res)
            _panel_title(axes[r][c], panel_labels[r][c], counts)

    cap.release()

    # Compact legend at the bottom
    handles = [Patch(facecolor="none", edgecolor=c, label=n)
               for n, c in CLASS_COLORS.items()]
    fig.legend(handles=handles,
               loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.01),
               fontsize=9, frameon=False,
               handlelength=2.0, columnspacing=2.0)

    fig.subplots_adjust(bottom=0.06)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "qualitative_player_detection")
