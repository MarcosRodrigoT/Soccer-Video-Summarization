"""Figure 14 — Qualitative timeline for Match 1.

Vector rebuild of the existing timeline_visualization.png, using real
cached artifacts. Stacks shots, player count + play/no-play, highlights,
and ground-truth goal regions over a chosen window of the match.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _style import save_figure, set_paper_style

PROJECT = Path(__file__).resolve().parents[2]
MATCH_DIR = PROJECT / "results_hr" / "per_match" / "Match1"


def build_figure():
    set_paper_style()

    shots = pd.read_csv(MATCH_DIR / "shot_boundaries.csv")
    fc = pd.read_csv(MATCH_DIR / "frame_classification.csv")
    play_seg = pd.read_csv(MATCH_DIR / "play_segments.csv")
    highlights = pd.read_csv(MATCH_DIR / "highlights.csv")
    goal_hl = pd.read_csv(MATCH_DIR / "goal_highlights.csv")
    score_ev = pd.read_csv(MATCH_DIR / "score_events.csv")

    # Window: 60 s around the first detected goal (frame ~24450, t≈978s)
    if len(score_ev) > 0:
        t_goal = float(score_ev.iloc[0]["timestamp"])
    else:
        t_goal = 1000.0
    win_lo, win_hi = max(0, t_goal - 120), t_goal + 60.0

    fig, axes = plt.subplots(
        nrows=5, figsize=(9.5, 6.2), sharex=True,
        gridspec_kw=dict(hspace=0.32, height_ratios=[0.5, 1.6, 0.5, 0.6, 0.4]),
    )

    # ---- Row 1: shots
    ax = axes[0]
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel("shots", fontsize=8, rotation=0, ha="right", va="center")
    win_shots = shots[(shots["end_time"] >= win_lo) & (shots["start_time"] <= win_hi)]
    for _, row in win_shots.iterrows():
        ax.axvspan(max(row["start_time"], win_lo),
                   min(row["end_time"],   win_hi),
                   alpha=0.55, color="#64B5F6",
                   edgecolor="#1976D2", linewidth=0.3)
    ax.text(win_hi + 1, 0.5, f"{len(win_shots)} shots", fontsize=7.5,
            va="center", color="#1976D2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ---- Row 2: player count smoothed + play overlay
    ax = axes[1]
    win_fc = fc[(fc["time_seconds"] >= win_lo) & (fc["time_seconds"] <= win_hi)]
    ax.plot(win_fc["time_seconds"], win_fc["player_count"],
            color="#9E9E9E", linewidth=0.5, label="raw count", alpha=0.7)
    if "player_count_smooth" in win_fc.columns:
        smooth = win_fc["player_count_smooth"]
    else:
        smooth = win_fc["player_count"].rolling(50, min_periods=1).mean()
    ax.plot(win_fc["time_seconds"], smooth,
            color="#C62828", linewidth=1.4, label="smoothed")
    ax.axhline(5, color="#F57C00", linestyle="--", linewidth=0.8,
               label="threshold")

    win_play = play_seg[(play_seg["end_time"] >= win_lo) &
                        (play_seg["start_time"] <= win_hi)]
    for _, row in win_play.iterrows():
        col = "#C8E6C9" if row["is_play"] else "#FFCDD2"
        ax.axvspan(max(row["start_time"], win_lo),
                   min(row["end_time"],   win_hi),
                   color=col, alpha=0.45, zorder=0)
    ax.set_ylabel("player count", fontsize=8)
    ax.legend(loc="upper right", fontsize=7, ncol=3,
              frameon=True, framealpha=0.95, edgecolor="#BDBDBD")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ---- Row 3: standard highlights
    ax = axes[2]
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel("highlights", fontsize=8, rotation=0, ha="right", va="center")
    win_hl = highlights[(highlights["end_time"] >= win_lo) &
                        (highlights["start_time"] <= win_hi)]
    for _, row in win_hl.iterrows():
        ax.axvspan(max(row["start_time"], win_lo),
                   min(row["end_time"],   win_hi),
                   color="#E1BEE7",
                   edgecolor="#7B1FA2", linewidth=0.6)
    ax.text(win_hi + 1, 0.5, f"{len(win_hl)} hl", fontsize=7.5,
            va="center", color="#7B1FA2")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ---- Row 4: goal highlights
    ax = axes[3]
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel("goal\nhighlights", fontsize=8, rotation=0,
                  ha="right", va="center")
    win_ghl = goal_hl[(goal_hl["end_time"] >= win_lo) &
                      (goal_hl["start_time"] <= win_hi)]
    for _, row in win_ghl.iterrows():
        ax.axvspan(max(row["start_time"], win_lo),
                   min(row["end_time"],   win_hi),
                   color="#FFCC80", edgecolor="#E65100", linewidth=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # ---- Row 5: score events
    ax = axes[4]
    ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_ylabel("score\nevents", fontsize=8, rotation=0,
                  ha="right", va="center")
    for _, row in score_ev.iterrows():
        t = row["timestamp"]
        if win_lo <= t <= win_hi:
            ax.axvline(t, color="#E65100", linewidth=2.0)
            ax.text(t, 0.5,
                    f"{int(row['home_score'])}-{int(row['away_score'])}",
                    rotation=0, ha="center", va="center", fontsize=8,
                    color="#E65100", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white",
                              edgecolor="#E65100", linewidth=0.6))
    ax.set_xlabel("time (s)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_xlim(win_lo, win_hi)

    fig.suptitle(f"Pipeline output timeline — Match 1, window around first goal "
                 f"({win_lo:.0f}–{win_hi:.0f} s)",
                 fontsize=12, fontweight="bold", y=0.99)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "qualitative_timeline")
