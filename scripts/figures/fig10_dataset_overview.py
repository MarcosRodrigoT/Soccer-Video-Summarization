"""Figure 10 — Dataset overview (vector rebuild of existing PNG).

Loads match metadata from config/matches.yaml and produces:
  (a) per-match goal bar chart
  (b) goal distribution histogram
  (c) season pie chart
  (d) team appearance frequency
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from _style import COLORS, save_figure, set_paper_style

PROJECT = Path(__file__).resolve().parents[2]


def load_matches():
    with open(PROJECT / "config" / "matches.yaml") as f:
        cfg = yaml.safe_load(f)
    rows = []
    for mid, m in cfg["matches"].items():
        home, away = m["score"].split("-")
        rows.append({
            "match": mid,
            "home_team": m["home_team"],
            "away_team": m["away_team"],
            "home_goals": int(home),
            "away_goals": int(away),
            "season": m["season"],
            "match_num": int(mid.replace("Match", "")),
        })
    rows.sort(key=lambda r: r["match_num"])
    return rows


def build_figure():
    set_paper_style()
    rows = load_matches()
    n = len(rows)
    total_goals = sum(r["home_goals"] + r["away_goals"] for r in rows)

    fig = plt.figure(figsize=(9.5, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          width_ratios=[1.3, 1.0, 1.0], wspace=0.32, hspace=0.40)

    # ----- (a) per-match goal bars ------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(n)
    home_goals = np.array([r["home_goals"] for r in rows])
    away_goals = np.array([r["away_goals"] for r in rows])
    ax1.bar(x, home_goals, color="#64B5F6", label="Home goals",
            edgecolor="white", linewidth=0.4)
    ax1.bar(x, away_goals, bottom=home_goals, color="#EF9A9A",
            label="Away goals", edgecolor="white", linewidth=0.4)
    mean_g = (home_goals + away_goals).mean()
    ax1.axhline(mean_g, color="#F57C00", linestyle="--", linewidth=1.0)
    ax1.text(n - 0.5, mean_g + 0.15, f"avg = {mean_g:.1f}",
             fontsize=8, color="#F57C00", ha="right", va="bottom")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"M{r['match_num']}" for r in rows],
                        fontsize=7, rotation=0)
    ax1.set_ylabel("goals")
    ax1.set_title("(a)  Goals per match", loc="left", fontsize=10,
                  fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(0, max(home_goals + away_goals) + 1)

    # ----- (b) goal-total histogram -----------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    totals = home_goals + away_goals
    bins = np.arange(totals.min(), totals.max() + 2) - 0.5
    ax2.hist(totals, bins=bins, color="#A5D6A7", edgecolor="#388E3C")
    ax2.set_xlabel("total goals")
    ax2.set_ylabel("matches")
    ax2.set_title("(b)  Total-goals distribution",
                  loc="left", fontsize=10, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ----- (c) season pie ---------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    seasons = Counter(r["season"] for r in rows)
    season_order = sorted(seasons.keys())
    sizes = [seasons[s] for s in season_order]
    colors = ["#FFF59D", "#CFD8DC", "#90CAF9", "#A5D6A7"][:len(season_order)]
    wedges, texts, autotexts = ax3.pie(sizes, labels=season_order,
                                       colors=colors,
                                       autopct="%.0f%%",
                                       textprops={"fontsize": 8},
                                       wedgeprops={"edgecolor": "white",
                                                   "linewidth": 1.0})
    ax3.set_title("(c)  Matches by season",
                  loc="left", fontsize=10, fontweight="bold")

    # ----- (d) team frequency ----------------------------------------------
    ax4 = fig.add_subplot(gs[1, 2])
    team_counts: Counter = Counter()
    for r in rows:
        team_counts[r["home_team"]] += 1
        team_counts[r["away_team"]] += 1
    top = team_counts.most_common(8)
    names, counts = zip(*top)
    short = [n.replace("FC Barcelona", "Barcelona")
              .replace("Real Madrid", "R. Madrid")
              .replace("Atlético de Madrid", "Atl. Madrid")
              .replace("Real Betis", "R. Betis")
              .replace("Rayo Vallecano", "Rayo Vall.")
              .replace("Villarreal CF", "Villarreal")
              .replace("Girona FC", "Girona")
              .replace("CA Osasuna", "Osasuna")
              .replace("Real Sociedad", "R. Sociedad")
            for n in names]
    ax4.barh(range(len(short)), counts[::-1], color="#B39DDB",
             edgecolor="#7B1FA2")
    ax4.set_yticks(range(len(short)))
    ax4.set_yticklabels(short[::-1], fontsize=8)
    ax4.set_xlabel("appearances")
    ax4.set_title("(d)  Top teams (matches feat.)",
                  loc="left", fontsize=10, fontweight="bold")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    fig.suptitle(f"Dataset overview — {n} La Liga matches · {total_goals} goals",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.subplots_adjust(top=0.92)
    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "dataset_overview_v2")
