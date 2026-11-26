"""Figure 4 — Player detection detail.

Diagrams the YOLOv5-based per-frame detection module:
  - frame sub-sampling at 5 fps (every 5th frame at 25 fps)
  - YOLOv5 inference with confidence threshold
  - 4-class output (ball, goalkeeper, player, referee)
  - aggregation into player_counts.csv with avg bbox diagonal
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from _style import (
    COLORS, arrow, clean_axes, page_size, rounded_box,
    save_figure, set_paper_style,
)


def build_figure():
    set_paper_style()
    fig = plt.figure(figsize=page_size(8.5, 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.7], hspace=0.30)

    # =================================================== Block diagram (top)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 6)
    clean_axes(ax)

    rounded_box(ax, 0.3, 2.3, 3.2, 1.4,
                "Video frames",
                subtitle="25 fps · 1080p",
                color="input", fontsize=9.5)
    rounded_box(ax, 4.1, 2.3, 3.6, 1.4,
                "Frame sub-sampling",
                subtitle="every 5th frame ⇒ 5 fps",
                color="visual", fontsize=9.5)
    rounded_box(ax, 8.3, 2.3, 3.8, 1.4,
                "YOLOv5 inference",
                subtitle="best.pt · GPU",
                color="visual", fontsize=9.5)
    rounded_box(ax, 12.7, 2.3, 3.8, 1.4,
                "Confidence filter",
                subtitle="$\\tau_{conf} = 0.5$",
                color="visual", fontsize=9.5)
    rounded_box(ax, 17.0, 2.3, 2.8, 1.4,
                "player_counts.csv",
                subtitle="frame · count · diag",
                color="neutral", fontsize=9.0,
                fontweight="normal")

    for x_start, x_end in [(3.5, 4.1), (7.7, 8.3), (12.1, 12.7), (16.5, 17.0)]:
        arrow(ax, (x_start, 3.0), (x_end, 3.0))

    # Four classes legend (below pipeline)
    classes = [
        ("ball",       "#E91E63"),
        ("goalkeeper", "#FF9800"),
        ("player",     "#2196F3"),
        ("referee",    "#9C27B0"),
    ]
    legend_x = 1.0
    ax.text(legend_x, 1.0, "Classes detected:", fontsize=9,
            fontweight="bold", color="#212121")
    x0 = legend_x + 3.0
    for name, col in classes:
        ax.add_patch(Rectangle((x0, 0.85), 0.4, 0.3,
                               facecolor=col, edgecolor="black",
                               linewidth=0.5))
        ax.text(x0 + 0.55, 1.0, name, fontsize=9, color="#212121",
                ha="left", va="center")
        x0 += 2.7

    ax.text(10, 5.4, "(a) YOLOv5 player-detection sub-pipeline",
            ha="center", fontsize=10, fontweight="bold")

    # =================================================== Synthetic frame (bottom)
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 16)
    ax2.set_ylim(0, 9)
    clean_axes(ax2)

    # Pitch background (green)
    ax2.add_patch(Rectangle((0, 0), 16, 9, facecolor="#A5D6A7",
                            edgecolor="black", linewidth=1.0, zorder=0))
    # Pitch line markings
    ax2.plot([8, 8], [0.5, 8.5], color="white", linewidth=1.0)
    ax2.add_patch(plt.Circle((8, 4.5), 1.2, fill=False,
                             edgecolor="white", linewidth=1.0))
    # Goal areas
    for x_pos in [0.5, 13.5]:
        ax2.add_patch(Rectangle((x_pos, 3.0), 2.0, 3.0,
                                fill=False, edgecolor="white", linewidth=1.0))

    # Detected players (random-ish positions)
    rng = np.random.default_rng(7)
    n_players = 18
    px = rng.uniform(1.5, 14.5, n_players)
    py = rng.uniform(1.0, 8.0, n_players)
    for x, y in zip(px, py):
        ax2.add_patch(Rectangle((x - 0.18, y - 0.45), 0.36, 0.9,
                                fill=False, edgecolor="#2196F3", linewidth=1.0))
    # Goalkeepers
    for x in [0.8, 14.8]:
        y = 4.5
        ax2.add_patch(Rectangle((x - 0.18, y - 0.45), 0.36, 0.9,
                                fill=False, edgecolor="#FF9800", linewidth=1.2))
    # Referee
    ax2.add_patch(Rectangle((8 - 0.18, 5.5 - 0.45), 0.36, 0.9,
                            fill=False, edgecolor="#9C27B0", linewidth=1.2))
    # Ball
    ax2.add_patch(plt.Circle((9.0, 4.5), 0.18,
                             fill=False, edgecolor="#E91E63", linewidth=1.2))

    ax2.text(8, 8.6, "Detected: 18 players · 2 goalkeepers · 1 referee · 1 ball",
             ha="center", fontsize=9, color="#212121", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#9E9E9E", linewidth=0.8))
    ax2.set_title("(b) Schematic of YOLOv5 output on a play-phase frame",
                  fontsize=10, pad=4, loc="center")

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "player_detection_detail")
