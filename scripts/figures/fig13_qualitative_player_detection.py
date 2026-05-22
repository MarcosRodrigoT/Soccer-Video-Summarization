"""Figure 13 — Qualitative YOLOv8 player detection on a real Match1 frame.

Runs the trained model on one frame chosen from inside a play segment and
renders the detections with per-class colors.
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

# YOLOv8 model trained with these class IDs:
CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
CLASS_COLORS = {
    "ball":       "#E91E63",
    "goalkeeper": "#FF9800",
    "player":     "#2196F3",
    "referee":    "#9C27B0",
}


def pick_play_frame() -> int:
    """Pick a representative play frame — longest play segment after t = 60 s."""
    pc = pd.read_csv(MATCH_DIR / "play_segments.csv")
    play = pc[(pc["is_play"] == True) & (pc["start_time"] > 60.0)].copy()
    if len(play) == 0:
        return 50000
    play["dur"] = play["end_time"] - play["start_time"]
    seg = play.sort_values("dur", ascending=False).iloc[0]
    return int((seg["start_frame"] + seg["end_frame"]) / 2)


def build_figure():
    set_paper_style()
    fig = plt.figure(figsize=(8.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.20)

    ax_img = fig.add_subplot(gs[0])
    ax_stat = fig.add_subplot(gs[1])

    if not VIDEO.exists() or not MODEL_PATH.exists():
        ax_img.text(0.5, 0.5, "video / model unavailable",
                    ha="center", transform=ax_img.transAxes, fontsize=11)
        ax_img.set_xticks([]); ax_img.set_yticks([])
        return fig

    cap = cv2.VideoCapture(str(VIDEO))
    frame_idx = pick_play_frame()
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        ax_img.text(0.5, 0.5, "could not read frame",
                    ha="center", transform=ax_img.transAxes, fontsize=11)
        return fig

    # Lazy-import to avoid YOLO cost at module import time
    from ultralytics import YOLO
    model = YOLO(str(MODEL_PATH))
    results = model.predict(source=frame_bgr, conf=0.5, verbose=False)
    res = results[0]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ax_img.imshow(frame_rgb)
    counts = {n: 0 for n in CLASS_NAMES.values()}
    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c, conf in zip(xyxy, cls, confs):
            name = CLASS_NAMES.get(c, "unknown")
            color = CLASS_COLORS.get(name, "white")
            ax_img.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       fill=False, edgecolor=color,
                                       linewidth=1.4))
            counts[name] = counts.get(name, 0) + 1

    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f"YOLOv8 detections on Match 1, frame {frame_idx}",
                     loc="left", fontsize=10, fontweight="bold")

    # Right panel: counts and legend
    ax_stat.axis("off")
    handles = [Patch(facecolor="none", edgecolor=c, label=f"{n}  ×{counts.get(n, 0)}")
               for n, c in CLASS_COLORS.items()]
    ax_stat.legend(handles=handles, loc="upper center", fontsize=10,
                   frameon=True, framealpha=1.0,
                   edgecolor="#BDBDBD", handlelength=2.0,
                   title="detected class · count",
                   title_fontsize=10)

    info = (f"Frame index: {frame_idx}\n"
            f"Time:        {frame_idx / 25:.1f} s\n"
            f"Confidence τ: 0.50\n"
            f"Model:       YOLOv8 (best.pt)\n"
            f"mAP@50:      0.929")
    ax_stat.text(0.05, 0.30, info, va="top", ha="left",
                 fontsize=9, family="monospace", color="#212121",
                 transform=ax_stat.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                           edgecolor="#9E9E9E", linewidth=0.7))

    fig.suptitle("Qualitative player detection — Match 1, play phase",
                 fontsize=12, fontweight="bold", y=0.97)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "qualitative_player_detection")
