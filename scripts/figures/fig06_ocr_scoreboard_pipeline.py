"""Figure 6 — OCR scoreboard sub-pipeline.

The complete chain that detects, localizes and reads the scoreboard.
Phase A localizes the region once per match; Phase B reads scores from
sampled frames.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    arrow, clean_axes, page_size, rounded_box,
    save_figure, set_paper_style,
)


def build_figure():
    set_paper_style()
    fig, ax = plt.subplots(figsize=page_size(9.5, 8.0))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 18)
    clean_axes(ax)

    # ----- Title --------------------------------------------------------
    ax.text(12, 17.4,
            "Scoreboard localization & OCR sub-pipeline",
            ha="center", fontsize=12, fontweight="bold")

    # ===================================================================
    # Phase A: Region localization
    # ===================================================================
    ax.text(12, 16.4,
            "Phase A — Region localization (run once per match)",
            ha="center", fontsize=10.5, fontweight="bold", style="italic",
            color="#E65100")

    # Row A1: 4 boxes, each 4.8 wide, gap 0.4 → 4*4.8 + 3*0.4 = 20.4, centred at 12
    row1 = [
        ("Sample 200 frames",   "uniform · grayscale"),
        ("Per-pixel IQR",       "$P_{75} - P_{25}$"),
        ("Percentile threshold","$IQR < P_{8}$"),
        ("Morphology",          "close(51,21) · open(9,9)"),
    ]
    bx = 1.8
    bw = 4.8
    gap = 0.4
    for i, (lbl, sub) in enumerate(row1):
        x = bx + i * (bw + gap)
        rounded_box(ax, x, 13.6, bw, 1.4, lbl, subtitle=sub,
                    color="scoreboard", fontsize=9.5)
        if i > 0:
            arrow(ax, (x - gap + 0.05, 14.3), (x, 14.3))

    # Row A2: 4 boxes
    row2 = [
        ("Connected components", "candidate regions"),
        ("Width filter",         "≤ 40 % frame width"),
        ("Split wide regions",   "via raw-mask gaps"),
        ("Region score · best",  "+10 SCORE · −10 LALIGA"),
    ]
    for i, (lbl, sub) in enumerate(row2):
        x = bx + i * (bw + gap)
        rounded_box(ax, x, 10.6, bw, 1.4, lbl, subtitle=sub,
                    color="scoreboard", fontsize=9.5)
        if i > 0:
            arrow(ax, (x - gap + 0.05, 11.3), (x, 11.3))

    # Wrap from end of row1 to start of row2
    arrow(ax, (bx + 3 * (bw + gap) + bw, 13.6),
          (bx, 12.0), rad=-0.25)

    # Tighten bbox + output
    rounded_box(ax, bx, 7.6, bw, 1.4, "Tighten bbox",
                subtitle="trim to raw mask",
                color="scoreboard", fontsize=9.5)
    arrow(ax, (bx + 3 * (bw + gap) + bw, 10.6),
          (bx + bw, 8.3), rad=-0.25)

    rounded_box(ax, 9.0, 7.6, 6.0, 1.4, "scoreboard_region.json",
                subtitle="{ x, y, w, h }",
                color="neutral", fontsize=9.5, fontweight="normal")
    arrow(ax, (bx + bw, 8.3), (9.0, 8.3))

    # ===================================================================
    # Phase B: Per-frame OCR
    # ===================================================================
    ax.text(12, 6.6,
            "Phase B — Per-frame score reading (every 3 s)",
            ha="center", fontsize=10.5, fontweight="bold", style="italic",
            color="#E65100")

    row_b = [
        ("Sample frames", "every 3 s"),
        ("Crop region",   "from json"),
        ("Upscale 4×",    "+ mask out background"),
        ("EasyOCR",       "digit + text confidence"),
        ("Parse score",   "X–Y · position clusters"),
    ]
    bx_b = 0.6
    bw_b = 4.4
    gap_b = 0.3
    for i, (lbl, sub) in enumerate(row_b):
        x = bx_b + i * (bw_b + gap_b)
        rounded_box(ax, x, 3.8, bw_b, 1.4, lbl, subtitle=sub,
                    color="scoreboard", fontsize=9.5)
        if i > 0:
            arrow(ax, (x - gap_b + 0.02, 4.5), (x, 4.5))

    # Filter chain (red)
    rounded_box(ax, 3.0, 0.8, 14.0, 1.6,
                "Robustness filters",
                subtitle=("temporal median  ·  cluster concentration  ·  "
                          "static-element  ·  persistence  ·  max-jump"),
                color="intersection", fontsize=10)

    arrow(ax, (bx_b + 4 * (bw_b + gap_b) + bw_b / 2, 3.8),
          (17.0, 2.4), rad=-0.15)

    # Score events output (right edge)
    rounded_box(ax, 18.0, 0.8, 5.4, 1.6,
                "score_events.csv",
                subtitle="timestamp · home · away · team",
                color="accent", fontsize=9.5, fontweight="normal")
    arrow(ax, (17.0, 1.6), (18.0, 1.6))

    # cross-phase: region.json drives Phase B "Crop region"
    arrow(ax, (12.0, 7.6), (7.6, 5.2), rad=0.20,
          color="#666666", linewidth=0.9)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "ocr_scoreboard_pipeline")
