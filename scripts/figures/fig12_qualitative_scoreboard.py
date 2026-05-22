"""Figure 12 — Qualitative scoreboard pipeline on a real Match1 frame.

Composite figure with 5 panels:
  (a) raw video frame (with localized scoreboard bbox overlaid)
  (b) per-pixel IQR map
  (c) binary overlay mask (post threshold + morphology)
  (d) cropped scoreboard region (4× upscaled for OCR)
  (e) detected score events on the match timeline
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from _style import save_figure, set_paper_style

PROJECT = Path(__file__).resolve().parents[2]
MATCH_DIR = PROJECT / "results_hr" / "per_match" / "Match1"
VIDEO = Path("/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos/Laliga_HR/Match1/"
             "Valencia CF - FC Barcelona (2-3) LALIGA 2013⧸2014 FULL MATCH "
             "[Qzrg3QDmQ4k].mp4")


def grab_frame(path: Path, frame_idx: int = 12000) -> np.ndarray | None:
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def build_figure():
    set_paper_style()

    # Load region (480p coordinates — scale ×2.25 for 1080p)
    with open(MATCH_DIR / "scoreboard_region.json") as f:
        bbox480 = json.load(f)
    scale = 1080 / 480
    bx = int(bbox480["x"] * scale)
    by = int(bbox480["y"] * scale)
    bw = int(bbox480["w"] * scale)
    bh = int(bbox480["h"] * scale)

    # IQR map (1080p)
    iqr = np.load(MATCH_DIR / "iqr_map.npy")

    # binary mask: P8 threshold on IQR
    thr = np.percentile(iqr, 8.0)
    binary = (iqr < thr).astype(np.uint8) * 255
    # morphology
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              np.ones((51, 21), np.uint8))
    overlay_mask = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                                    np.ones((9, 9), np.uint8))

    # frame
    frame = grab_frame(VIDEO, frame_idx=12000)

    # score events (timeline)
    score_df = pd.read_csv(MATCH_DIR / "score_events.csv")

    fig = plt.figure(figsize=(9.5, 7.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.55],
                          hspace=0.30, wspace=0.20)

    # (a) raw frame with bbox
    ax_a = fig.add_subplot(gs[0, :2])
    if frame is not None:
        ax_a.imshow(frame)
        ax_a.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                                 edgecolor="#FF1744", linewidth=2.0))
        ax_a.text(bx, by - 20, "localized scoreboard",
                  color="#FF1744", fontsize=8.5, fontweight="bold")
    else:
        ax_a.text(0.5, 0.5, "video frame unavailable",
                  ha="center", transform=ax_a.transAxes, fontsize=10)
    ax_a.set_title("(a)  Match1 frame  ·  bbox from scoreboard_region.json",
                   loc="left", fontsize=9.5, fontweight="bold")
    ax_a.set_xticks([]); ax_a.set_yticks([])

    # (d) cropped scoreboard region — small inset on top-right
    ax_d = fig.add_subplot(gs[0, 2])
    if frame is not None:
        crop = frame[by:by + bh, bx:bx + bw]
        crop_up = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4),
                             interpolation=cv2.INTER_CUBIC)
        ax_d.imshow(crop_up)
    ax_d.set_title("(d)  Cropped region (4× upscaled\nfor OCR)",
                   loc="left", fontsize=9.5, fontweight="bold")
    ax_d.set_xticks([]); ax_d.set_yticks([])

    # (b) IQR heat map
    ax_b = fig.add_subplot(gs[1, 0])
    im = ax_b.imshow(iqr, cmap="magma", vmin=0, vmax=80)
    ax_b.set_title("(b)  Per-pixel IQR map",
                   loc="left", fontsize=9.5, fontweight="bold")
    ax_b.set_xticks([]); ax_b.set_yticks([])
    plt.colorbar(im, ax=ax_b, fraction=0.04)

    # (c) binary overlay mask after morphology
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.imshow(overlay_mask, cmap="gray_r")
    ax_c.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                             edgecolor="#FF1744", linewidth=1.5))
    ax_c.set_title(f"(c)  Binary mask (IQR < P{8})  ·  close + open",
                   loc="left", fontsize=9.5, fontweight="bold")
    ax_c.set_xticks([]); ax_c.set_yticks([])

    # Mini decision panel: the algorithm summary
    ax_x = fig.add_subplot(gs[1, 2])
    ax_x.axis("off")
    txt = ("Pipeline\n"
           "1. sample 200 frames\n"
           "2. compute IQR per pixel\n"
           "3. threshold IQR < P₈\n"
           "4. close(51,21) + open(9,9)\n"
           "5. connected components\n"
           "6. width filter, split, score\n"
           "7. tighten bbox\n"
           f"\nResult bbox  (1080p)\n"
           f"  x = {bx}, y = {by}\n"
           f"  w = {bw}, h = {bh}")
    ax_x.text(0.0, 1.0, txt, va="top", ha="left", fontsize=8.5,
              family="monospace", color="#212121")

    # (e) detected score events timeline
    ax_e = fig.add_subplot(gs[2, :])
    match_duration = 6800.0  # ≈ 113 min including stoppage
    for _, row in score_df.iterrows():
        t = row["timestamp"]
        ax_e.axvline(t, color="#E65100", linewidth=1.5)
        ax_e.text(t, 0.85, f"{int(row['home_score'])}-{int(row['away_score'])}",
                  rotation=0, ha="center", va="bottom",
                  fontsize=8, color="#E65100", fontweight="bold")
    ax_e.set_xlim(0, match_duration)
    ax_e.set_ylim(0, 1.0)
    ax_e.set_yticks([])
    ax_e.set_xlabel("time into match (s)")
    ax_e.set_title("(e)  Detected score events (Match 1, final 2-3)",
                   loc="left", fontsize=9.5, fontweight="bold")
    ax_e.spines["top"].set_visible(False)
    ax_e.spines["right"].set_visible(False)
    ax_e.spines["left"].set_visible(False)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "qualitative_scoreboard")
