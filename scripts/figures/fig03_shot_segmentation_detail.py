"""Figure 3 — Shot segmentation detail.

Decomposes the shot-boundary detection stage:
  - histogram differencing
  - abrupt (cut) detection via adaptive threshold + windowed maximum
  - gradual transition detection via smoothed-signal peaks
  - shot construction + minimum-duration filtering
A small synthetic histogram-difference signal illustrates the detection.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import (
    COLORS, arrow, clean_axes, page_size, rounded_box,
    save_figure, set_paper_style,
)


def build_figure():
    set_paper_style()
    fig = plt.figure(figsize=page_size(8.5, 6.5))

    # Two-row layout: top = block diagram, bottom = signal illustration
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.30)

    # =================================================== Block diagram (top)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 5.5)
    clean_axes(ax)

    rounded_box(ax, 0.2, 2.0, 3.2, 1.5,
                "Video frames",
                subtitle="grayscale",
                color="input", fontsize=9.5)
    rounded_box(ax, 3.9, 2.0, 3.4, 1.5,
                "256-bin histogram",
                subtitle="per frame",
                color="visual", fontsize=9.5)
    rounded_box(ax, 7.8, 2.0, 3.4, 1.5,
                "L1 histogram diff",
                subtitle="$d_i = \\|h_i - h_{i-1}\\|_1$",
                color="visual", fontsize=9.5)

    # Two parallel branches: abrupt and gradual
    rounded_box(ax, 11.7, 3.4, 4.4, 1.3,
                "Abrupt cuts",
                subtitle="threshold + window max",
                color="visual", fontsize=9.5)
    rounded_box(ax, 11.7, 0.7, 4.4, 1.3,
                "Gradual transitions",
                subtitle="smoothed-peak detection",
                color="visual", fontsize=9.5)

    rounded_box(ax, 16.6, 2.0, 3.2, 1.5,
                "Shot list",
                subtitle="min duration filter",
                color="visual", fontsize=9.5)

    arrow(ax, (3.4, 2.75), (3.9, 2.75))
    arrow(ax, (7.3, 2.75), (7.8, 2.75))
    arrow(ax, (11.2, 3.05), (11.7, 4.0), rad=0.15,
          color="#666666")
    arrow(ax, (11.2, 2.45), (11.7, 1.4), rad=-0.15,
          color="#666666")
    arrow(ax, (16.1, 4.0), (16.6, 2.95), rad=-0.15)
    arrow(ax, (16.1, 1.4), (16.6, 2.55), rad=0.15)

    ax.text(10, 5.1, "(a) Shot boundary detection sub-pipeline",
            ha="center", fontsize=10, fontweight="bold")

    # =================================================== Signal plot (bottom)
    ax2 = fig.add_subplot(gs[1])

    # Synthetic histogram-difference signal with two cuts and one dissolve
    rng = np.random.default_rng(42)
    n = 600
    t = np.arange(n)
    base = 0.04 + 0.015 * rng.standard_normal(n)
    base = np.abs(base)

    # inject two abrupt cuts (single-frame spikes)
    base[120] = 0.85
    base[400] = 0.72
    # inject a gradual transition around t=250
    grad_center, grad_w = 260, 14
    for k, idx in enumerate(range(grad_center - grad_w, grad_center + grad_w + 1)):
        base[idx] += 0.25 * np.exp(-((k - grad_w) ** 2) / (2 * 6.0 ** 2))

    ax2.plot(t, base, color="#388E3C", linewidth=0.7, label="$d_i$ (L1 histogram diff)")
    ax2.axhline(0.30, color="#C62828", linestyle="--", linewidth=1.0,
                label="abrupt-cut threshold τ")
    ax2.plot([120, 400], [base[120], base[400]], "o", color="#C62828",
             markersize=8, markeredgecolor="white", markeredgewidth=1.0,
             label="detected cut")
    ax2.plot([grad_center], [base[grad_center]], "s", color="#F57C00",
             markersize=8, markeredgecolor="white", markeredgewidth=1.0,
             label="dissolve peak")

    # annotate shots between cuts
    shot_bounds = [0, 120, grad_center, 400, n]
    for i in range(len(shot_bounds) - 1):
        s, e = shot_bounds[i], shot_bounds[i + 1]
        ax2.axvspan(s, e, alpha=0.06,
                    color=(COLORS["visual"][1] if i % 2 == 0 else "#FFC107"),
                    zorder=0)
        ax2.text((s + e) / 2, -0.10, f"shot {i + 1}",
                 ha="center", fontsize=8, color="#1B5E20",
                 transform=ax2.get_xaxis_transform())

    ax2.set_xlim(0, n)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("frame index", fontsize=9)
    ax2.set_ylabel("L1 histogram distance", fontsize=9)
    ax2.legend(loc="upper right", fontsize=7.5, ncol=2, frameon=True,
               framealpha=0.95, edgecolor="#BDBDBD")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("(b) Synthetic example — frame-to-frame histogram distance "
                  "with detected boundaries",
                  fontsize=10, pad=4, loc="center")

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "shot_segmentation_detail")
