"""Figure 15 — Error taxonomy.

A taxonomy chart of true-positive, false-positive and false-negative cases,
organized by which component is responsible.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    arrow, clean_axes, page_size, rounded_box,
    save_figure, set_paper_style,
)


def build_figure():
    set_paper_style()
    fig, ax = plt.subplots(figsize=page_size(9.5, 6.8))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    clean_axes(ax)

    ax.text(10, 13.4,
            "Error taxonomy — categorizing TP, FP and FN by failure mode",
            ha="center", fontsize=12, fontweight="bold")

    # Root
    rounded_box(ax, 7.5, 11.3, 5.0, 1.4,
                "Highlight prediction outcome",
                color="evaluation", fontsize=10.5)

    # Three branches: TP, FP, FN — wider boxes with shorter subtitles
    rounded_box(ax, 0.2, 9.0, 6.0, 1.4, "True positive",
                subtitle="overlap with GT ≥ τ",
                color="visual", fontsize=10)
    rounded_box(ax, 7.0, 9.0, 6.0, 1.4, "False positive",
                subtitle="pred ∉ GT",
                color="intersection", fontsize=10)
    rounded_box(ax, 13.8, 9.0, 6.0, 1.4, "False negative",
                subtitle="GT ∉ pred",
                color="scoreboard", fontsize=10)

    arrow(ax, (10.0, 11.3), (3.2, 10.4), rad=-0.15)
    arrow(ax, (10.0, 11.3), (10.0, 10.4))
    arrow(ax, (10.0, 11.3), (16.8, 10.4), rad=0.15)

    # Subcategories
    tp_sub = [
        "Goal action correctly covered",
        "Long-shot play correctly kept",
        "Goal-context window aligned",
    ]
    fp_sub = [
        "Stat overlay misread as score",
        "Long shot during warm-up / replay",
        "Crowd close-up classified as play",
        "Goal-context window over-extended",
    ]
    fn_sub = [
        "Goal occurs in short shot (≤ 4 s)",
        "Score event missed by OCR",
        "Play classifier under-counts players",
        "Highlight ends before goal celebration",
    ]

    def draw_list(items, x, color):
        y0 = 7.3
        for txt in items:
            rounded_box(ax, x, y0, 6.0, 0.9, txt,
                        color=color, fontsize=8.5,
                        fontweight="normal", rounding=0.07,
                        linewidth=0.7)
            y0 -= 1.15
        # arrow from parent to first entry
        arrow(ax, (x + 3.0, 9.0), (x + 3.0, 8.2), color="#9E9E9E",
              linewidth=0.7)

    draw_list(tp_sub, 0.2,  "visual")
    draw_list(fp_sub, 7.0,  "intersection")
    draw_list(fn_sub, 13.8, "scoreboard")

    # bottom annotation — relate to scripts
    ax.text(10, 0.5,
            "FP/FN categories are scored automatically by  "
            "scripts/error_analysis.py  (per-match and aggregated).",
            ha="center", fontsize=8.5, style="italic", color="#616161")

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "error_taxonomy")
