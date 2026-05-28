"""Figure 12 — Qualitative scoreboard pipeline on Match1.

Compact, single-grid layout:
  Row 1 (three equal panels):  (a) frame with localised bbox,
                              (b) per-pixel IQR map,
                              (c) binary mask after morphology.
  Row 2:                       (d) the cropped scoreboard (4x upscale)
                                  side-by-side with a horizontal timeline
                                  of detected score events.
No floating boxes or summary text; every panel sits in its own grid cell.
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

PANEL_TITLE = dict(loc="left", fontsize=9.5, fontweight="bold")
BBOX_COLOR = "#FF1744"


def _read_frame(idx: int = 12000) -> np.ndarray | None:
    if not VIDEO.exists():
        return None
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


def _binary_mask(iqr: np.ndarray) -> np.ndarray:
    thr = np.percentile(iqr, 8.0)
    raw = (iqr < thr).astype(np.uint8) * 255
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE,
                              np.ones((51, 21), np.uint8))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                              np.ones((9, 9), np.uint8))
    return opened


def build_figure():
    set_paper_style()

    with open(MATCH_DIR / "scoreboard_region.json") as f:
        bbox = json.load(f)
    bx, by = int(bbox["x"]), int(bbox["y"])
    bw, bh = int(bbox["w"]), int(bbox["h"])

    iqr = np.load(MATCH_DIR / "iqr_map.npy")
    mask = _binary_mask(iqr)
    frame = _read_frame()

    score_df = pd.read_csv(MATCH_DIR / "score_events.csv")

    # Layout:
    #   row 0 (three equal image panels) ........ (a) (b) (c)
    #   row 1 (one wide panel split in two)    ... (d) crop  |  (e) timeline
    fig = plt.figure(figsize=(10.5, 5.1))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 0.45],
        hspace=0.28, wspace=0.10,
    )

    # ---- Row 1: three image panels ----------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    if frame is not None:
        ax_a.imshow(frame)
        ax_a.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                                 edgecolor=BBOX_COLOR, linewidth=1.8))
    else:
        ax_a.text(0.5, 0.5, "frame unavailable",
                  ha="center", transform=ax_a.transAxes)
    ax_a.set_title("(a) broadcast frame  ·  localised bbox", **PANEL_TITLE)
    ax_a.set_xticks([]); ax_a.set_yticks([])

    ax_b = fig.add_subplot(gs[0, 1])
    im = ax_b.imshow(iqr, cmap="magma", vmin=0, vmax=80)
    ax_b.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                             edgecolor=BBOX_COLOR, linewidth=1.2))
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    ax_b.set_title(r"(b) per-pixel IQR map", **PANEL_TITLE)
    ax_b.set_xticks([]); ax_b.set_yticks([])

    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.imshow(mask, cmap="gray_r")
    ax_c.add_patch(Rectangle((bx, by), bw, bh, fill=False,
                             edgecolor=BBOX_COLOR, linewidth=1.2))
    ax_c.set_title(r"(c) binary mask after closing/opening", **PANEL_TITLE)
    ax_c.set_xticks([]); ax_c.set_yticks([])

    # ---- Row 2: cropped scoreboard | timeline ------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    if frame is not None:
        crop = frame[by:by + bh, bx:bx + bw]
        crop_up = cv2.resize(crop, (crop.shape[1] * 4, crop.shape[0] * 4),
                             interpolation=cv2.INTER_CUBIC)
        ax_d.imshow(crop_up)
    ax_d.set_title(r"(d) cropped region (4$\times$ upscale, OCR input)",
                   **PANEL_TITLE)
    ax_d.set_xticks([]); ax_d.set_yticks([])

    ax_e = fig.add_subplot(gs[1, 1:])
    match_duration = 6800.0  # ~113 min with stoppage
    ax_e.set_xlim(0, match_duration)
    ax_e.set_ylim(0, 1.0)
    ax_e.set_yticks([])
    ax_e.set_xlabel("time into match (s)", fontsize=8.5)
    # Faint baseline
    ax_e.axhline(0.4, color="#BDBDBD", linewidth=0.8, zorder=1)
    for _, row in score_df.iterrows():
        t = float(row["timestamp"])
        ax_e.plot([t, t], [0.40, 0.75],
                  color="#E65100", linewidth=1.6, zorder=2)
        ax_e.scatter([t], [0.40], s=14, color="#E65100", zorder=3)
        ax_e.text(t, 0.80,
                  f"{int(row['home_score'])}-{int(row['away_score'])}",
                  ha="center", va="bottom",
                  fontsize=8, color="#E65100", fontweight="bold")
    for spine in ("top", "right", "left"):
        ax_e.spines[spine].set_visible(False)
    ax_e.tick_params(axis="x", labelsize=7)
    ax_e.set_title(r"(e) detected score events  ·  converging on final 2--3",
                   **PANEL_TITLE)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "qualitative_scoreboard")
