"""Figure 1 (horizontal) — End-to-end system architecture, landscape layout.

Wide variant of the canonical paper figure, intended to span the full text
width of a journal page. Same content as `fig01_system_architecture.py` but
laid out left-to-right with three horizontal processing lanes that converge
into the Highlight Assembly stage.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from _style import (
    arrow,
    clean_axes,
    page_size,
    rounded_box,
    save_figure,
    set_paper_style,
)

# Canvas dimensions (logical units)
W = 24.0
H = 9.6
BW = 3.6   # standard stage block width
BH = 1.3   # standard stage block height


def build_figure():
    set_paper_style()
    fig, ax = plt.subplots(figsize=page_size(13.8, 5.5))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    clean_axes(ax)

    # ---------------------------- layout coordinates --------------------------
    # Row y-centers (top = shot lane, mid = player lane, bot = scoreboard lane)
    row_top = 7.0
    row_mid = 5.0
    row_bot = 3.0

    # Column x-centers
    col_input = 2.05
    col_s1    = 7.00   # Stage 1
    col_s2    = 11.50  # Stage 2
    col_fuse  = 16.00  # Highlight Assembly
    col_out   = 20.30  # Highlight Summary Video

    # Highlight Assembly box dims (slightly wider, taller than stage boxes)
    fuse_w_box = 3.6
    fuse_h_box = 1.5

    # Output box dims
    out_w = 3.6
    out_h = 1.5

    # Eval / GT row
    eval_w = 3.4
    eval_h = 1.3
    eval_y = 1.0
    col_eval = 17.10
    col_gt   = 21.50

    # ---------------------------- branch-grouping bands -----------------------
    # Subtle pastel bands behind each branch's two stage columns to make the
    # visual / scoreboard grouping immediately obvious without text-heavy labels.
    band_left  = col_s1 - BW / 2 - 0.45
    band_right = col_s2 + BW / 2 + 0.45
    band_w     = band_right - band_left

    # Visual band spans rows top + mid
    vb_y_bot = row_mid - BH / 2 - 0.22
    vb_y_top = row_top + BH / 2 + 0.30
    visual_band = FancyBboxPatch(
        (band_left, vb_y_bot), band_w, vb_y_top - vb_y_bot,
        boxstyle="round,pad=0,rounding_size=0.18",
        facecolor="#F1F8F2", edgecolor="#CFE3D0",
        linewidth=0.8, zorder=0.5,
    )
    ax.add_patch(visual_band)

    # Scoreboard band spans row bot only
    sb_y_bot = row_bot - BH / 2 - 0.30
    sb_y_top = row_bot + BH / 2 + 0.22
    scoreboard_band = FancyBboxPatch(
        (band_left, sb_y_bot), band_w, sb_y_top - sb_y_bot,
        boxstyle="round,pad=0,rounding_size=0.18",
        facecolor="#FDF6EE", edgecolor="#F4DBBE",
        linewidth=0.8, zorder=0.5,
    )
    ax.add_patch(scoreboard_band)

    # Branch labels (floating just above their bands)
    ax.text(
        (band_left + band_right) / 2, vb_y_top + 0.30,
        "Visual analysis branch",
        ha="center", va="bottom", fontsize=10,
        fontweight="bold", style="italic", color="#1B5E20", zorder=2,
    )
    ax.text(
        (band_left + band_right) / 2, sb_y_top + 0.05,
        "Scoreboard branch",
        ha="center", va="bottom", fontsize=10,
        fontweight="bold", style="italic", color="#E65100", zorder=2,
    )

    # ---------------------------- Input ---------------------------------------
    input_h = 1.5
    rounded_box(
        ax, col_input - BW / 2, row_mid - input_h / 2, BW, input_h,
        "Input Video",
        subtitle="La Liga match · 25 fps · ≈100 min",
        color="input", fontsize=10,
    )

    # ---------------------------- Stage 1 -------------------------------------
    rounded_box(
        ax, col_s1 - BW / 2, row_top - BH / 2, BW, BH,
        "Shot Boundary Detection",
        subtitle="frame differencing · adaptive threshold",
        color="visual",
    )
    rounded_box(
        ax, col_s1 - BW / 2, row_mid - BH / 2, BW, BH,
        "Player Detection",
        subtitle="YOLOv8 · 4 classes · sampled @ 5 fps",
        color="visual",
    )
    rounded_box(
        ax, col_s1 - BW / 2, row_bot - BH / 2, BW, BH,
        "Scoreboard Localization",
        subtitle="per-pixel IQR · morphology · CC",
        color="scoreboard",
    )

    # ---------------------------- Stage 2 -------------------------------------
    rounded_box(
        ax, col_s2 - BW / 2, row_top - BH / 2, BW, BH,
        "Shot Filtering",
        subtitle="discard shots < 4 s",
        color="visual",
    )
    rounded_box(
        ax, col_s2 - BW / 2, row_mid - BH / 2, BW, BH,
        "Play / No-Play Classification",
        subtitle="smoothed count ≥ 5 ⇒ play",
        color="visual",
    )
    rounded_box(
        ax, col_s2 - BW / 2, row_bot - BH / 2, BW, BH,
        "Goal Event Detection",
        subtitle="EasyOCR · multi-filter · score Δ",
        color="scoreboard",
    )

    # ---------------------------- Highlight Assembly --------------------------
    rounded_box(
        ax, col_fuse - fuse_w_box / 2, row_mid - fuse_h_box / 2,
        fuse_w_box, fuse_h_box,
        "Highlight Assembly",
        subtitle="long shots ∩ play\nmerge gaps · goal context",
        color="intersection", fontsize=10.5,
    )

    # ---------------------------- Output --------------------------------------
    rounded_box(
        ax, col_out - out_w / 2, row_mid - out_h / 2, out_w, out_h,
        "Highlight Summary Video",
        subtitle="highlights.csv · goals.csv · MP4",
        color="output", fontsize=10,
    )

    # ---------------------------- Evaluation + Ground Truth -------------------
    rounded_box(
        ax, col_eval - eval_w / 2, eval_y - eval_h / 2, eval_w, eval_h,
        "Evaluation",
        subtitle="P · R · F1 · tIoU",
        color="evaluation", fontsize=10,
    )
    rounded_box(
        ax, col_gt - eval_w / 2, eval_y - eval_h / 2, eval_w, eval_h,
        "Ground Truth",
        subtitle="frame · sentence-level",
        color="ground_truth", fontsize=10,
    )

    # ============================================================ Arrows
    # ----- Input → Stage 1 (three lanes diverging) -----
    arrow(ax, (col_input + BW / 2, row_mid + 0.30),
          (col_s1 - BW / 2,        row_top), rad=0.22)
    arrow(ax, (col_input + BW / 2, row_mid),
          (col_s1 - BW / 2,        row_mid))
    arrow(ax, (col_input + BW / 2, row_mid - 0.30),
          (col_s1 - BW / 2,        row_bot), rad=-0.22)

    # ----- Stage 1 → Stage 2 (one horizontal per lane) -----
    arrow(ax, (col_s1 + BW / 2, row_top), (col_s2 - BW / 2, row_top))
    arrow(ax, (col_s1 + BW / 2, row_mid), (col_s2 - BW / 2, row_mid))
    arrow(ax, (col_s1 + BW / 2, row_bot), (col_s2 - BW / 2, row_bot))

    # ----- Cross-lane: Shot Filtering (top) → Play Classification (mid) -----
    arrow(ax, (col_s2, row_top - BH / 2),
          (col_s2, row_mid + BH / 2), color="#666666")
    ax.text(col_s2 + 0.28, (row_top + row_mid) / 2,
            "shot bounds",
            ha="left", va="center", fontsize=7.5,
            style="italic", color="#666666", zorder=3)

    # ----- Stage 2 → Highlight Assembly (three lanes converging) -----
    fuse_left = col_fuse - fuse_w_box / 2
    arrow(ax, (col_s2 + BW / 2, row_top),
          (fuse_left,           row_mid + 0.30), rad=-0.15)
    arrow(ax, (col_s2 + BW / 2, row_mid),
          (fuse_left,           row_mid))
    arrow(ax, (col_s2 + BW / 2, row_bot),
          (fuse_left,           row_mid - 0.30), rad=0.15, color="#666666")
    # Label sits in the gap between the two band backgrounds
    ax.text((col_s2 + BW / 2 + fuse_left) / 2, (sb_y_top + vb_y_bot) / 2,
            "score events",
            ha="center", va="center", fontsize=7.5,
            style="italic", color="#666666", zorder=3)

    # ----- Highlight Assembly → Output -----
    arrow(ax, (col_fuse + fuse_w_box / 2, row_mid),
          (col_out - out_w / 2,           row_mid))

    # ----- Output → Evaluation (downward, slight curve) -----
    arrow(ax, (col_out - 0.55, row_mid - out_h / 2),
          (col_eval + 0.20,    eval_y + eval_h / 2),
          rad=0.18, color="#666666")
    ax.text((col_out - 0.55 + col_eval + 0.20) / 2 - 0.55,
            (row_mid - out_h / 2 + eval_y + eval_h / 2) / 2,
            "predictions",
            ha="center", va="center", fontsize=7.5,
            style="italic", color="#666666", zorder=3)

    # ----- Ground Truth → Evaluation (right-to-left) -----
    arrow(ax, (col_gt - eval_w / 2, eval_y),
          (col_eval + eval_w / 2,   eval_y),
          color="#666666",
          label="reference", label_pos=0.5,
          label_offset=(0, 0.22), label_fontsize=7.5)

    # ============================================================ Title
    ax.text(
        W / 2, H - 0.42,
        "Soccer Video Summarization — End-to-End Pipeline",
        ha="center", va="center", fontsize=13.5, fontweight="bold",
    )

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "system_architecture_horizontal")
