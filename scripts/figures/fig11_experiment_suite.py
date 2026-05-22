"""Figure 11 — Experiment suite map.

Shows how the core pipeline (centre) is wrapped by the suite of scripts that
produce the paper's quantitative results: baselines, ablations, component
analysis, error analysis, statistical tests, and figure/table generation.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from _style import arrow, clean_axes, page_size, rounded_box, save_figure, set_paper_style


def build_figure():
    set_paper_style()
    fig, ax = plt.subplots(figsize=page_size(8.5, 7.5))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-9, 9)
    clean_axes(ax)

    # Central pipeline node
    rounded_box(ax, -3.0, -1.2, 6.0, 2.4,
                "Core pipeline",
                subtitle="run_single_match.py",
                color="intersection", fontsize=11)

    satellites = [
        # (angle, label, subtitle, color, file)
        (90,  "Baselines",
         "uniform · shot-only · play-only · OCR-only · oracle",
         "visual", "run_baselines.py"),
        (40,  "Ablations",
         "param sweeps · 7 dimensions",
         "visual", "run_ablation.py"),
        (-20, "Component analysis",
         "drop one component at a time",
         "visual", "run_component_analysis.py"),
        (-70, "Error analysis",
         "FP / FN categorization",
         "scoreboard", "error_analysis.py"),
        (-130,"Statistical tests",
         "Wilcoxon · Cliff's δ",
         "scoreboard", "statistical_analysis.py"),
        (170, "Figures + tables",
         "10 plots · LaTeX tables",
         "output", "generate_figures.py · generate_tables.py"),
        (130, "Match dashboards",
         "7-panel per-match diagnostic",
         "output", "generate_match_dashboard.py"),
    ]

    for ang, lbl, sub, col, fname in satellites:
        rad = 6.5
        x_c = rad * np.cos(np.deg2rad(ang))
        y_c = rad * np.sin(np.deg2rad(ang))
        bx = x_c - 2.4
        by = y_c - 0.85
        rounded_box(ax, bx, by, 4.8, 1.7, lbl, subtitle=sub,
                    color=col, fontsize=9.5)
        # arrow from center to satellite
        # start from edge of central box, end at edge of satellite
        sx = 0.0 + 1.5 * np.cos(np.deg2rad(ang))
        sy = 0.0 + 0.9 * np.sin(np.deg2rad(ang))
        ex = x_c - 1.6 * np.cos(np.deg2rad(ang))
        ey = y_c - 0.65 * np.sin(np.deg2rad(ang))
        arrow(ax, (sx, sy), (ex, ey), color="#9E9E9E", linewidth=0.9,
              style="-|>")
        # filename below box in italic small
        ax.text(x_c, by - 0.35, fname,
                ha="center", fontsize=7, style="italic", color="#616161")

    ax.text(0, 8.3,
            "Experiment suite — scripts that wrap the core pipeline",
            ha="center", fontsize=12, fontweight="bold")
    ax.text(0, -8.3,
            "All scripts cache intermediate stages; ablations reuse upstream "
            "artifacts so only the varied parameter is re-run.",
            ha="center", fontsize=8.5, color="#616161", style="italic")

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "experiment_suite")
