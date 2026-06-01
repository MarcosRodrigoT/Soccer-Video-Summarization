"""Figure: goal-centred clip assembly timeline for Section 7.

Illustrates the two-line decision used by the highlight assembly stage
to extract a goal-centred clip from a detected scoreboard transition.
A short window around the first goal of a representative match is shown,
with the following stacked lanes:

  1. detected shot boundaries (vertical ticks),
  2. shot-level play/no-play classification (play long shots highlighted),
  3. OCR-derived score transition at the detected goal time,
  4. resulting highlight clip (final tail of the anchor shot).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT / "scripts" / "figures"))

from _style import save_figure, set_paper_style  # noqa: E402

MATCH_ID = "Match1"
MATCH_DIR = PROJECT / "results_hr_10s" / "per_match" / MATCH_ID
PLAY_DIR = PROJECT / "results_hr" / "per_match" / MATCH_ID

WINDOW_BEFORE_S = 60.0
WINDOW_AFTER_S = 15.0


def _mmss(t_s: float, _pos) -> str:
    t_s = max(0.0, t_s)
    return f"{int(t_s) // 60:02d}:{int(t_s) % 60:02d}"


def build_figure() -> None:
    set_paper_style()

    shots = pd.read_csv(MATCH_DIR / "shot_boundaries.csv")
    fc = pd.read_csv(MATCH_DIR / "frame_classification.csv")
    play_seg = pd.read_csv(PLAY_DIR / "play_segments.csv")
    highlights = pd.read_csv(MATCH_DIR / "highlights.csv")
    goal_hl = pd.read_csv(MATCH_DIR / "goal_highlights.csv")
    score_ev = pd.read_csv(MATCH_DIR / "score_events.csv")

    # Centre window on the first detected goal event
    t_goal = float(score_ev.iloc[0]["timestamp"])
    win_lo = max(0.0, t_goal - WINDOW_BEFORE_S)
    win_hi = t_goal + WINDOW_AFTER_S

    fig, axes = plt.subplots(
        nrows=4, sharex=True, figsize=(9.5, 4.0),
        gridspec_kw=dict(hspace=0.20,
                         height_ratios=[0.5, 1.5, 0.5, 0.5]),
    )

    # ---- Lane 1: shot boundaries (vertical ticks) -------------------------
    ax = axes[0]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("shots", rotation=0, ha="right", va="center", fontsize=8)
    win_shots = shots[(shots["end_time"] >= win_lo)
                      & (shots["start_time"] <= win_hi)]
    for _, row in win_shots.iterrows():
        ax.axvline(row["start_time"], color="#1565C0", lw=0.7, alpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- Lane 2: smoothed visible-people count + play overlay -------------
    ax = axes[1]
    win_fc = fc[(fc["time_seconds"] >= win_lo) & (fc["time_seconds"] <= win_hi)]
    ax.plot(win_fc["time_seconds"], win_fc["player_count_smooth"],
            color="#1B5E20", lw=1.2, label=r"$\widetilde{n}(t)$")
    ax.axhline(5, color="#9E9E9E", ls="--", lw=0.8,
               label=r"$\tau_{\mathrm{play}}=5$")
    # Play long shots: shade only the ones that overlap the window
    win_play = play_seg[(play_seg["end_time"] >= win_lo)
                        & (play_seg["start_time"] <= win_hi)
                        & (play_seg["is_play"] == True)]
    for _, row in win_play.iterrows():
        ax.axvspan(max(row["start_time"], win_lo),
                   min(row["end_time"], win_hi),
                   ymin=0, ymax=1, color="#C8E6C9", alpha=0.55, zorder=0)
    ax.set_ylabel("people", fontsize=8)
    ax.set_ylim(0, max(25, win_fc["player_count_smooth"].max() + 2))
    ax.legend(loc="upper left", fontsize=7, ncols=2)
    ax.spines[["top", "right"]].set_visible(False)

    # ---- Lane 3: OCR-derived score transition -----------------------------
    ax = axes[2]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("score event", rotation=0, ha="right", va="center",
                  fontsize=8)
    ax.axvline(t_goal, color="#7b1fa2", lw=1.6, alpha=0.95)
    ax.annotate(r"detected $\hat t_m$",
                xy=(t_goal, 0.5), xytext=(t_goal + 1.5, 0.5),
                fontsize=8, color="#7b1fa2", va="center")
    ax.spines[["top", "right"]].set_visible(False)

    # ---- Lane 4: goal-centred highlight clip ------------------------------
    ax = axes[3]
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(r"$\mathcal{H}_{\mathrm{goal}}$",
                  rotation=0, ha="right", va="center", fontsize=8)
    # Pick the goal_highlight clip whose midpoint is closest to t_goal
    if len(goal_hl) > 0:
        mid = (goal_hl["start_time"] + goal_hl["end_time"]) / 2
        idx = (mid - t_goal).abs().idxmin()
        clip = goal_hl.loc[idx]
        ax.axvspan(clip["start_time"], clip["end_time"],
                   color="#FFB74D", alpha=0.85, zorder=2)
        ax.text(clip["start_time"] - 0.5, 0.5,
                f"{clip['end_time'] - clip['start_time']:.0f} s",
                fontsize=7.5, ha="right", va="center", color="#E65100")
    ax.spines[["top", "right"]].set_visible(False)

    # X axis formatting
    from matplotlib.ticker import FuncFormatter
    axes[-1].set_xlim(win_lo, win_hi)
    axes[-1].set_xlabel("Time (mm:ss)")
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_mmss))

    save_figure(fig, "goal_assembly_timeline", also_png=True)


if __name__ == "__main__":
    build_figure()
