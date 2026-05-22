"""Figure 9 — Evaluation framework.

Predictions ↔ ground truth matching at the chosen tolerance window;
computation of Precision / Recall / F1, compression ratio, tIoU, goal-time MAE.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    arrow, clean_axes, page_size, rounded_box,
    save_figure, set_paper_style,
)


def build_figure():
    set_paper_style()
    fig = plt.figure(figsize=page_size(8.5, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.7, 1.0], hspace=0.45)

    # ========================================================== TOP: schema
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 6)
    clean_axes(ax)

    rounded_box(ax, 0.4, 2.3, 3.8, 1.6,
                "Predictions",
                subtitle="highlights.csv",
                color="output", fontsize=10)
    rounded_box(ax, 0.4, 0.3, 3.8, 1.6,
                "Ground truth",
                subtitle="frame-level / sentence-level",
                color="ground_truth", fontsize=10)

    rounded_box(ax, 5.4, 1.5, 4.4, 2.0,
                "Match by overlap",
                subtitle="tolerance window = 30 s",
                color="evaluation", fontsize=10)

    rounded_box(ax, 11.0, 4.2, 2.8, 1.4, "TP", subtitle="overlap ≥ τ",
                color="visual", fontsize=10)
    rounded_box(ax, 11.0, 2.3, 2.8, 1.4, "FP", subtitle="pred ∉ GT",
                color="intersection", fontsize=10)
    rounded_box(ax, 11.0, 0.4, 2.8, 1.4, "FN", subtitle="GT ∉ pred",
                color="scoreboard", fontsize=10)

    rounded_box(ax, 14.8, 2.3, 5.0, 2.0,
                "Aggregate metrics",
                subtitle=("Precision · Recall · F1\n"
                          "compression · mean tIoU\n"
                          "goal timing MAE"),
                color="accent", fontsize=10, fontweight="normal")

    arrow(ax, (4.2, 3.1), (5.4, 2.7), rad=-0.1)
    arrow(ax, (4.2, 1.1), (5.4, 2.1), rad=0.1)
    arrow(ax, (9.8, 2.95), (11.0, 4.5), rad=0.15)
    arrow(ax, (9.8, 2.5), (11.0, 2.9), rad=0.0)
    arrow(ax, (9.8, 2.05), (11.0, 1.0), rad=-0.15)
    arrow(ax, (13.8, 4.0), (14.8, 3.6), rad=-0.1)
    arrow(ax, (13.8, 2.85), (14.8, 3.1), rad=0.0)
    arrow(ax, (13.8, 1.1), (14.8, 2.6), rad=0.1)

    ax.text(10, 5.5, "(a) Prediction–GT matching and metrics",
            ha="center", fontsize=10, fontweight="bold")

    # ========================================================== BOTTOM: viz
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 200)
    ax2.set_ylim(-0.3, 2.7)
    ax2.set_yticks([0.5, 1.5])
    ax2.set_yticklabels(["GT", "Pred"])
    ax2.set_xlabel("time (s)")

    gt = [(20, 50), (70, 110), (135, 165), (180, 195)]
    preds = [(25, 55), (90, 120), (140, 150)]

    # Draw GT
    for s, e in gt:
        ax2.add_patch(plt.Rectangle((s, 0.25), e - s, 0.5,
                                    facecolor="#CFD8DC",
                                    edgecolor="#546E7A", linewidth=1.0))

    # Tolerance windows around each GT
    tol = 30
    for s, e in gt:
        ax2.add_patch(plt.Rectangle((s - tol, 0.05), (e + tol) - (s - tol), 0.9,
                                    facecolor="#FFE0B2", alpha=0.20,
                                    edgecolor="#F57C00", linewidth=0.6,
                                    linestyle="--"))

    # Draw predictions (color them by category)
    def matches(pred, gt_intervals, tol=30):
        ps, pe = pred
        for gs_, ge in gt_intervals:
            # Overlap inside tolerance-expanded GT
            if not (pe < gs_ - tol or ps > ge + tol):
                return True
        return False

    matched_gt = [False] * len(gt)
    for ps, pe in preds:
        is_tp = False
        for i, (gs_, ge) in enumerate(gt):
            if not (pe < gs_ - tol or ps > ge + tol):
                matched_gt[i] = True
                is_tp = True
        col = ("#C8E6C9", "#388E3C") if is_tp else ("#FFCDD2", "#C62828")
        ax2.add_patch(plt.Rectangle((ps, 1.25), pe - ps, 0.5,
                                    facecolor=col[0], edgecolor=col[1],
                                    linewidth=1.0))
        # label
        ax2.text((ps + pe) / 2, 1.95, "TP" if is_tp else "FP",
                 ha="center", fontsize=8, color=col[1], fontweight="bold")

    # FN markers
    for ok, (gs_, ge) in zip(matched_gt, gt):
        if not ok:
            ax2.text((gs_ + ge) / 2, -0.05, "FN",
                     ha="center", fontsize=8, color="#F57C00", fontweight="bold")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("(b) Worked example — TP / FP / FN with 30 s tolerance window",
                  fontsize=10, pad=4)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "evaluation_framework")
