"""Figure 5 — Play classification detail.

Player counts → smoothing window → threshold ⇒ contiguous play/no-play segments.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import arrow, clean_axes, page_size, rounded_box, save_figure, set_paper_style


def build_figure():
    set_paper_style()
    fig = plt.figure(figsize=page_size(8.5, 6.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.6, 1.0], hspace=0.35)

    # ============================================ top: block diagram
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4)
    clean_axes(ax)

    rounded_box(ax, 0.2,  1.2, 3.4, 1.5,
                "player_counts.csv",
                subtitle="raw count @ 5 fps",
                color="neutral", fontsize=9, fontweight="normal")
    rounded_box(ax, 4.2,  1.2, 3.6, 1.5,
                "Smoothing",
                subtitle="rolling mean · w=50",
                color="visual", fontsize=9.5)
    rounded_box(ax, 8.4,  1.2, 3.6, 1.5,
                "Threshold",
                subtitle="count ≥ 5 ⇒ play",
                color="visual", fontsize=9.5)
    rounded_box(ax, 12.6, 1.2, 3.6, 1.5,
                "Segmentation",
                subtitle="contiguous regions",
                color="visual", fontsize=9.5)
    rounded_box(ax, 16.8, 1.2, 3.0, 1.5,
                "play_segments.csv",
                subtitle="start · end · is_play",
                color="neutral", fontsize=9, fontweight="normal")

    for x0, x1 in [(3.6, 4.2), (7.8, 8.4), (12.0, 12.6), (16.2, 16.8)]:
        arrow(ax, (x0, 1.95), (x1, 1.95))

    ax.text(10, 3.5, "(a) Play-classification sub-pipeline",
            ha="center", fontsize=10, fontweight="bold")

    # ============================================ bottom: signal plot
    ax2 = fig.add_subplot(gs[1])

    rng = np.random.default_rng(1)
    duration_s = 240.0
    fps_sampled = 5
    t = np.arange(0, int(duration_s * fps_sampled)) / fps_sampled

    # Synthetic ground-truth play intervals (in seconds)
    gt_play = [(8, 40), (55, 90), (110, 175), (200, 225)]

    raw = np.zeros_like(t)
    for s, e in gt_play:
        mask = (t >= s) & (t <= e)
        # In a play segment, around 18 ± 4 players visible
        raw[mask] = 18 + 3.5 * rng.standard_normal(mask.sum())
    # Outside play: 0-3 (sparse)
    not_play_mask = raw == 0
    raw[not_play_mask] = np.clip(2 + 2.5 * rng.standard_normal(not_play_mask.sum()),
                                  0, None)
    # Inject occasional spurious detections during no-play (close-ups)
    spurious_idx = rng.integers(0, len(t), 20)
    raw[spurious_idx] += rng.uniform(2, 8, 20)

    # Smooth (window=50 samples = 10 s at 5 fps)
    win = 50
    smoothed = np.convolve(raw, np.ones(win) / win, mode="same")
    thr = 5.0
    is_play = smoothed >= thr

    ax2.plot(t, raw, color="#9E9E9E", linewidth=0.5, label="raw count",
             alpha=0.8)
    ax2.plot(t, smoothed, color="#388E3C", linewidth=1.7, label="smoothed")
    ax2.axhline(thr, color="#C62828", linestyle="--", linewidth=1.1,
                label=f"threshold = {int(thr)}")

    # Shade play / no-play
    in_segment = False
    seg_start = None
    for i, val in enumerate(is_play):
        if val and not in_segment:
            seg_start = t[i]
            in_segment = True
        elif not val and in_segment:
            ax2.axvspan(seg_start, t[i], alpha=0.18, color="#388E3C", zorder=0)
            in_segment = False
    if in_segment:
        ax2.axvspan(seg_start, t[-1], alpha=0.18, color="#388E3C", zorder=0)
    # Inverse: no-play
    in_segment = False
    seg_start = t[0]
    for i, val in enumerate(is_play):
        if not val and not in_segment:
            seg_start = t[i]
            in_segment = True
        elif val and in_segment:
            ax2.axvspan(seg_start, t[i], alpha=0.15, color="#C62828", zorder=0)
            in_segment = False
    if in_segment:
        ax2.axvspan(seg_start, t[-1], alpha=0.15, color="#C62828", zorder=0)

    ax2.set_xlabel("time (s)", fontsize=9)
    ax2.set_ylabel("player count", fontsize=9)
    ax2.set_xlim(0, duration_s)
    ax2.set_ylim(0, 30)

    # Legend manual (also show shading)
    from matplotlib.patches import Patch
    handles, labels = ax2.get_legend_handles_labels()
    handles.append(Patch(facecolor="#388E3C", alpha=0.3, label="play"))
    handles.append(Patch(facecolor="#C62828", alpha=0.25, label="no-play"))
    ax2.legend(handles=handles, loc="upper right", fontsize=7.5, ncol=2,
               frameon=True, framealpha=0.95, edgecolor="#BDBDBD")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("(b) Player count signal · smoothing · threshold ⇒ play / no-play segments",
                  fontsize=10, pad=4)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "play_classification_detail")
