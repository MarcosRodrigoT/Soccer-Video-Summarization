"""Figure 1 — End-to-end system architecture.

The canonical paper figure: full pipeline showing all six stages, both
processing branches (visual + scoreboard), inputs, outputs, and evaluation.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    arrow,
    clean_axes,
    page_size,
    rounded_box,
    save_figure,
    set_paper_style,
)

# Canvas dimensions (logical units, not inches)
W = 20.0
H = 16.0
# Standard stage block dimensions
BW = 4.6   # block width
BH = 1.6   # block height


def build_figure():
    set_paper_style()
    fig, ax = plt.subplots(figsize=page_size(9.5, 7.5))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    clean_axes(ax)

    # Column x-centers
    col_left  = 3.4
    col_mid   = 9.6
    col_right = 16.0

    # ============================================================ Input row
    rounded_box(
        ax, col_mid - BW / 2, 14.2, BW, 1.3,
        "Input Video",
        subtitle="full La Liga match · 25 fps · ≈100 min",
        color="input", fontsize=10,
    )

    # Branch labels  — one span across both visual columns, one on scoreboard
    visual_center = (col_left + col_mid) / 2
    ax.text(visual_center, 12.6, "Visual analysis branch",
            ha="center", fontsize=10, fontweight="bold",
            style="italic", color="#1B5E20")
    ax.text(col_right, 12.6, "Scoreboard branch",
            ha="center", fontsize=10, fontweight="bold",
            style="italic", color="#E65100")
    # Separator lines for the branch headers
    ax.plot([col_left - BW / 2, col_mid + BW / 2],
            [12.35, 12.35], color="#388E3C",
            linewidth=0.6, alpha=0.5, zorder=1)
    ax.plot([col_right - BW / 2, col_right + BW / 2],
            [12.35, 12.35], color="#F57C00",
            linewidth=0.6, alpha=0.5, zorder=1)

    # ============================================================ Stage 1 row
    rounded_box(
        ax, col_left - BW / 2, 10.6, BW, BH,
        "Shot Boundary Detection",
        subtitle="frame differencing · adaptive threshold",
        color="visual",
    )
    rounded_box(
        ax, col_mid - BW / 2, 10.6, BW, BH,
        "Player Detection",
        subtitle="YOLOv5 · 4 classes · sampled @ 5 fps",
        color="visual",
    )
    rounded_box(
        ax, col_right - BW / 2, 10.6, BW, BH,
        "Scoreboard Localization",
        subtitle="per-pixel IQR · morphology · CC",
        color="scoreboard",
    )

    # ============================================================ Stage 2 row
    rounded_box(
        ax, col_left - BW / 2, 8.0, BW, BH,
        "Shot Filtering",
        subtitle="discard shots < 4 s",
        color="visual",
    )
    rounded_box(
        ax, col_mid - BW / 2, 8.0, BW, BH,
        "Play / No-Play Classification",
        subtitle="smoothed count ≥ 5 ⇒ play",
        color="visual",
    )
    rounded_box(
        ax, col_right - BW / 2, 8.0, BW, BH,
        "Goal Event Detection",
        subtitle="EasyOCR · multi-filter · score Δ",
        color="scoreboard",
    )

    # ============================================================ Fusion
    fuse_x = 3.0
    fuse_w = 14.0
    rounded_box(
        ax, fuse_x, 5.2, fuse_w, 1.7,
        "Highlight Assembly",
        subtitle="long shots ∩ play moments · merge gaps · expand goal-context windows",
        color="intersection", fontsize=11,
    )

    # ============================================================ Output row (centered)
    rounded_box(
        ax, 5.7, 3.3, 8.6, 1.4,
        "Highlight Summary Video",
        subtitle="highlights.csv · goal_highlights.csv · concatenated MP4",
        color="output", fontsize=10.5,
    )

    # ============================================================ Eval + GT row (below output)
    rounded_box(
        ax, 3.6, 1.0, 5.2, 1.4,
        "Evaluation",
        subtitle="P · R · F1 · tIoU",
        color="evaluation", fontsize=10,
    )
    rounded_box(
        ax, 10.8, 1.0, 5.2, 1.4,
        "Ground Truth",
        subtitle="frame · sentence-level",
        color="ground_truth", fontsize=10,
    )

    # ====================================================== Arrows: Input → Stage 1
    # End arrows at row 12.2 (just above the stage-1 row) and start from
    # bottom edge of the Input box (y=14.2) at three different x positions.
    arrow(ax, (col_mid - 1.6, 14.2), (col_left,  12.25), rad=-0.18)
    arrow(ax, (col_mid,       14.2), (col_mid,   12.25))
    arrow(ax, (col_mid + 1.6, 14.2), (col_right, 12.25), rad=0.18)

    # ====================================================== Stage 1 → Stage 2
    arrow(ax, (col_left,  10.6), (col_left,  9.6))
    arrow(ax, (col_mid,   10.6), (col_mid,   9.6))
    arrow(ax, (col_right, 10.6), (col_right, 9.6))

    # Cross-branch: shot bounds → play classification (Stage 2 row).
    # Label placed above both rows in the inter-row gap.
    arrow(ax, (col_left + BW / 2, 8.8), (col_mid - BW / 2, 8.8),
          color="#666666")
    ax.text((col_left + col_mid) / 2, 9.85, "shot bounds",
            ha="center", va="center", fontsize=7.5,
            style="italic", color="#666666", zorder=3)

    # ====================================================== Stage 2 → Fusion
    arrow(ax, (col_left,  8.0), (5.6,  6.9), rad=0.12)
    arrow(ax, (col_mid,   8.0), (col_mid, 6.9))
    arrow(ax, (col_right, 8.0), (14.6, 6.9), rad=-0.12,
          label="score events", label_pos=0.55, label_offset=(0.6, 0.3),
          color="#666666")

    # ====================================================== Fusion → Output
    arrow(ax, (10.0, 5.2), (10.0, 4.7))

    # ====================================================== Output → Evaluation (downward)
    arrow(ax, (6.2, 3.3), (6.2, 2.4),
          label="predictions", label_pos=0.5, label_offset=(-1.1, 0.0),
          color="#666666", label_fontsize=7.5)

    # Ground Truth → Evaluation (right-to-left)
    arrow(ax, (10.8, 1.7), (8.8, 1.7),
          label="reference", label_pos=0.5, label_offset=(0, 0.25),
          color="#666666", label_fontsize=7.5)

    # ====================================================== Title
    ax.text(
        W / 2, 15.55,
        "Soccer Video Summarization — End-to-End Pipeline",
        ha="center", va="center", fontsize=12.5, fontweight="bold",
    )

    # (Color legend removed — branch labels above each column already serve
    # the same purpose and the per-row stage colors are self-explanatory.)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "system_architecture")
