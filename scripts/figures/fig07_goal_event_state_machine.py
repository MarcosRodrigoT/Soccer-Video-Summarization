"""Figure 7 — Goal-event detection: filter chain that turns raw OCR readings
into authoritative score-change events.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _style import (
    arrow, clean_axes, page_size, rounded_box,
    save_figure, set_paper_style,
)


def build_figure():
    set_paper_style()
    fig, ax = plt.subplots(figsize=page_size(8.5, 7.5))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 18)
    clean_axes(ax)

    ax.text(10, 17.3,
            "Goal-event detection — multi-stage OCR-reading filter",
            ha="center", fontsize=12, fontweight="bold")

    # Stages flow top-to-bottom
    stages = [
        (15.4, "Raw OCR readings",
         "per-frame parsed scores  (home, away)",
         "input"),
        (13.4, "1.  Temporal median filter",
         "window = 15 frames — kills single-frame stat-overlay digits",
         "scoreboard"),
        (11.4, "2.  Cluster concentration filter",
         "remove digit-positions used <20% of the time (clocks)",
         "scoreboard"),
        (9.4,  "3.  Static-element filter",
         ">90% single non-zero digit + <5% zeros ⇒ branding, discard",
         "scoreboard"),
        (7.4,  "4.  Monotonicity enforcement",
         "scores can only increase — clip downward transients",
         "scoreboard"),
        (5.4,  "5.  Forward-persistence check",
         "new value must hold ≥ 8 consecutive readings",
         "scoreboard"),
        (3.4,  "6.  Max-jump constraint",
         "Δ(home + away) ≤ 2 between adjacent score states",
         "scoreboard"),
        (1.4,  "Score-event timeline",
         "anchored to last '0→1' transition (refine_transition_timing)",
         "intersection"),
    ]

    for y, lbl, sub, color in stages:
        rounded_box(ax, 2.0, y, 16.0, 1.6, lbl, subtitle=sub,
                    color=color, fontsize=10)

    for i in range(len(stages) - 1):
        y_top = stages[i][0]
        y_bot = stages[i + 1][0]
        arrow(ax, (10.0, y_top), (10.0, y_bot + 1.6))

    # Right-side annotations of what each filter removes
    annotations = [
        (13.4, "stat-overlay\nspike digits"),
        (11.4, "clock 00:23 ⇒\nposition 4–5"),
        (9.4,  "stale\n'1' brand mark"),
        (7.4,  "downward\nblips"),
        (5.4,  "transient\nmisreads"),
        (3.4,  "score jumps\n>2 in 1 frame"),
    ]
    for y, txt in annotations:
        ax.text(19.0, y + 0.8, txt,
                ha="center", va="center", fontsize=7.5,
                color="#C62828",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="#FFEBEE",
                          edgecolor="#C62828", linewidth=0.6))
        arrow(ax, (18.5, y + 0.8), (18.05, y + 0.8),
              color="#C62828", linewidth=0.6, style="-|>")

    # Left-side caption
    ax.text(1.0, 9.4, "discard:",
            ha="center", va="center", fontsize=7.5,
            rotation=90, color="#C62828", fontweight="bold")

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "goal_event_state_machine")
