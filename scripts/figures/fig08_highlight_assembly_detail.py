"""Figure 8 — Highlight assembly detail.

Shows the intersection of long shots ∩ play moments, then the optional
goal-context expansion. A small synthetic timeline at the bottom makes it
concrete.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import arrow, clean_axes, page_size, rounded_box, save_figure, set_paper_style


def build_figure():
    set_paper_style()
    fig = plt.figure(figsize=page_size(8.5, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.55, 1.0], hspace=0.35)

    # ============================================ block diagram (top)
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 4)
    clean_axes(ax)

    rounded_box(ax, 0.4, 2.5, 3.6, 1.3,
                "shot_boundaries", subtitle="filtered ≥ 4 s",
                color="neutral", fontsize=9, fontweight="normal")
    rounded_box(ax, 0.4, 0.6, 3.6, 1.3,
                "play_segments", subtitle="per-frame ⇒ intervals",
                color="neutral", fontsize=9, fontweight="normal")

    rounded_box(ax, 5.2, 1.6, 4.0, 1.6,
                "Intersection",
                subtitle="long shot ∩ play",
                color="intersection", fontsize=10)
    rounded_box(ax, 10.0, 1.6, 4.0, 1.6,
                "Gap merging",
                subtitle="bridge gaps < 0.5 s",
                color="intersection", fontsize=10)
    rounded_box(ax, 14.8, 2.5, 4.6, 1.3,
                "highlights.csv",
                subtitle="standard highlight set",
                color="output", fontsize=9.5)
    rounded_box(ax, 14.8, 0.6, 4.6, 1.3,
                "goal_highlights.csv",
                subtitle="expanded ±goal context",
                color="output", fontsize=9.5)

    arrow(ax, (4.0, 3.15), (5.2, 2.6), rad=-0.1)
    arrow(ax, (4.0, 1.25), (5.2, 2.0), rad=0.1)
    arrow(ax, (9.2, 2.4), (10.0, 2.4))
    arrow(ax, (14.0, 2.4), (14.8, 3.0))
    arrow(ax, (14.0, 2.4), (14.8, 1.4))

    # score events influence the goal-highlights path
    ax.text(12.0, 0.25, "score_events  ⇒  ±30 s before / +15 s after",
            ha="center", fontsize=8, color="#E65100", style="italic")

    ax.text(10, 3.7, "(a) Highlight-assembly logic",
            ha="center", fontsize=10, fontweight="bold")

    # ============================================ timeline (bottom)
    ax2 = fig.add_subplot(gs[1])

    duration = 200.0
    ax2.set_xlim(0, duration)
    ax2.set_ylim(0, 6.0)
    ax2.set_yticks([0.5, 2.0, 3.5, 5.0])
    ax2.set_yticklabels(["highlights", "play segments", "long shots", "score events"])
    ax2.set_xlabel("time (s)")

    # Synthetic long shots
    long_shots = [(5, 35), (40, 80), (95, 140), (155, 195)]
    for s, e in long_shots:
        ax2.add_patch(plt.Rectangle((s, 3.2), e - s, 0.6,
                                    facecolor="#C8E6C9", edgecolor="#388E3C",
                                    linewidth=0.8))

    # Synthetic play segments
    play = [(8, 30), (45, 78), (100, 135), (165, 188)]
    for s, e in play:
        ax2.add_patch(plt.Rectangle((s, 1.7), e - s, 0.6,
                                    facecolor="#C8E6C9", edgecolor="#388E3C",
                                    linewidth=0.8))

    # Score events
    score_events = [60, 125]
    for s in score_events:
        ax2.plot([s], [5.3], "v", color="#E65100", markersize=10, zorder=5)
        ax2.axvline(s, color="#E65100", alpha=0.3, linewidth=0.6,
                    linestyle="--", zorder=0)

    # Intersection of long-shots ∩ play
    def intersect(a, b):
        out = []
        for s1, e1 in a:
            for s2, e2 in b:
                s, e = max(s1, s2), min(e1, e2)
                if e > s:
                    out.append((s, e))
        return out
    intr = intersect(long_shots, play)

    # Expand intersections that fall in goal context (±30s before, +15 after)
    def in_goal_ctx(seg, events, pre=30, post=15):
        s, e = seg
        return any((s <= ev + post) and (e >= ev - pre) for ev in events)
    final = []
    for s, e in intr:
        if in_goal_ctx((s, e), score_events):
            # expand
            for ev in score_events:
                if s <= ev + 15 and e >= ev - 30:
                    s = min(s, ev - 30)
                    e = max(e, ev + 15)
            final.append((s, e))
        else:
            final.append((s, e))

    # Draw highlights (final)
    for s, e in final:
        ax2.add_patch(plt.Rectangle((s, 0.2), e - s, 0.6,
                                    facecolor="#E1BEE7", edgecolor="#7B1FA2",
                                    linewidth=1.0, zorder=2))

    # mark goal-context windows
    for ev in score_events:
        ax2.add_patch(plt.Rectangle((ev - 30, 0.0), 45, 1.0,
                                    facecolor="#FFE0B2", alpha=0.25,
                                    edgecolor="#F57C00", linewidth=0.5,
                                    linestyle="--", zorder=1))

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("(b) Synthetic timeline — final highlights (purple) "
                  "result from long-shots ∩ play, expanded around goals",
                  fontsize=10, pad=4)

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "highlight_assembly_detail")
