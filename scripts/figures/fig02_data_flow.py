"""Figure 2 — Data flow / artifact dependency graph.

Two-lane diagram: stages on the left, the CSV/JSON artifacts they emit and
consume on the right. Read top-to-bottom.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _style import arrow, clean_axes, page_size, rounded_box, save_figure, set_paper_style


def build_figure():
    set_paper_style()
    fig, ax = plt.subplots(figsize=page_size(8.5, 9.5))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 22)
    clean_axes(ax)

    # ---------------------- Top: external inputs ----------------------------
    rounded_box(ax, 0.8,  20.0, 4.5, 1.2,
                "Video file (.mp4)",
                subtitle="1080p · 25 fps · full match",
                color="input", fontsize=10)
    rounded_box(ax, 7.5,  20.0, 5.0, 1.2,
                "config/default.yaml",
                subtitle="thresholds, paths, parameters",
                color="accent", fontsize=10, fontweight="normal")
    rounded_box(ax, 14.7, 20.0, 4.5, 1.2,
                "Ground Truth CSV",
                subtitle="GoalGT1 · GoalGT2 · sentences",
                color="ground_truth", fontsize=10)

    # ---------------------- Stages and their artifacts ----------------------
    # (stage_label, stage_color, y, artifact_specs)
    # artifact_specs is a list of (filename, optional sub)
    stages = [
        ("Shot Segmentation",  "visual",       17.6,
            [("shot_boundaries.csv", "shot_id · start · end · duration")]),
        ("Player Detection",   "visual",       15.4,
            [("player_counts.csv", "frame · player_count · box_diag")]),
        ("Play Classification","visual",       13.2,
            [("frame_classification.csv", "per-frame play / no-play"),
             ("play_segments.csv",        "contiguous play intervals")]),
        ("Scoreboard OCR",     "scoreboard",   10.6,
            [("scoreboard_region.json", "auto-detected bbox"),
             ("score_events.csv",       "timestamp · home · away · team")]),
        ("Highlight Assembly", "intersection",  7.4,
            [("highlights.csv",        "long-shot ∩ play segments"),
             ("goal_highlights.csv",   "expanded around score events")]),
        ("Evaluation",         "evaluation",    4.0,
            [("evaluation.json",       "P · R · F1 · tIoU · MAE")]),
    ]

    stage_x = 1.2
    stage_w = 5.4
    stage_h = 1.5
    art_x = 9.2
    art_w = 7.6
    art_h = 0.9

    stage_anchors = {}   # stage label → (right-side x, mid-y)
    art_anchors = {}     # artifact filename → (left-side x, mid-y)

    for label, color, y, arts in stages:
        rounded_box(ax, stage_x, y, stage_w, stage_h, label,
                    color=color, fontsize=10.5)
        stage_anchors[label] = (stage_x + stage_w, y + stage_h / 2)

        # Place artifacts to the right, vertically centered around the stage row.
        n = len(arts)
        for i, (fname, sub) in enumerate(arts):
            if n == 1:
                ay = y + stage_h / 2 - art_h / 2
            else:
                ay = y + stage_h - art_h - i * (art_h + 0.15) + 0.05
            rounded_box(ax, art_x, ay, art_w, art_h, fname,
                        subtitle=sub, color="neutral",
                        fontsize=8.5, fontweight="normal", rounding=0.05,
                        linewidth=0.9)
            art_anchors[fname] = (art_x, ay + art_h / 2)

            # produces arrow: stage → artifact
            arrow(ax, (stage_x + stage_w, y + stage_h / 2),
                  (art_x, ay + art_h / 2),
                  color="#388E3C", linewidth=1.0)

    # ---------------------- Vertical flow between stages --------------------
    for i in range(len(stages) - 1):
        _, _, y_top, _ = stages[i]
        _, _, y_bot, _ = stages[i + 1]
        arrow(ax, (stage_x + stage_w / 2, y_top),
              (stage_x + stage_w / 2, y_bot + stage_h),
              color="#424242", linewidth=1.2)

    # ---------------------- Consumer arrows (artifact → stage) --------------
    consumes = [
        ("shot_boundaries.csv",  "Highlight Assembly"),
        ("player_counts.csv",    "Play Classification"),
        ("play_segments.csv",    "Highlight Assembly"),
        ("score_events.csv",     "Highlight Assembly"),
        ("highlights.csv",       "Evaluation"),
        ("goal_highlights.csv",  "Evaluation"),
    ]
    for art_name, stage_name in consumes:
        ax, _ = ax, None
        ax_pos = art_anchors[art_name]
        sx, sy = stage_anchors[stage_name]
        # Curve right around the artifact column and feed into the stage.
        arrow_x = art_x + art_w + 0.6
        arrow(plt.gca(), ax_pos, (arrow_x, ax_pos[1]),
              color="#C62828", linewidth=0.8, style="-")
        arrow(plt.gca(), (arrow_x, ax_pos[1]), (arrow_x, sy),
              color="#C62828", linewidth=0.8, style="-")
        arrow(plt.gca(), (arrow_x, sy), (sx + 0.3, sy),
              color="#C62828", linewidth=0.8)

    # ---------------------- External input → stage arrows -------------------
    # Video → shot seg, player det, scoreboard ocr
    for stage_name in ["Shot Segmentation", "Player Detection", "Scoreboard OCR"]:
        sx, sy = stage_anchors[stage_name]
        arrow(plt.gca(), (3.0, 20.0), (stage_x + stage_w / 2, sy + stage_h / 2),
              color="#1976D2", linewidth=1.0, rad=-0.15)

    # Ground Truth → Evaluation
    sx, sy = stage_anchors["Evaluation"]
    arrow(plt.gca(), (16.95, 20.0), (stage_x + stage_w, sy),
          color="#546E7A", linewidth=1.0, rad=0.2)

    # Config → all stages (dashed)
    for stage_name in [s[0] for s in stages]:
        sx, sy = stage_anchors[stage_name]
        arrow(plt.gca(), (10.0, 20.0), (stage_x + stage_w, sy + 0.4),
              color="#9E9E9E", linewidth=0.6, linestyle="--", rad=-0.05)

    # ---------------------- Title and legend --------------------------------
    ax.text(10, 21.5, "Data Flow — Per-Match Artifact Dependencies",
            ha="center", fontsize=12.5, fontweight="bold")

    leg_y = 0.6
    items = [
        ("produces",        "#388E3C", "—▶"),
        ("consumes",        "#C62828", "—▶"),
        ("video input",     "#1976D2", "—▶"),
        ("ground truth",    "#546E7A", "—▶"),
        ("config (params)", "#9E9E9E", "··▶"),
    ]
    x0 = 0.8
    for lbl, col, sym in items:
        ax.text(x0, leg_y, sym, color=col, fontsize=10, fontweight="bold",
                ha="left", va="center")
        ax.text(x0 + 1.0, leg_y, lbl, fontsize=8.5, color="#212121",
                ha="left", va="center")
        x0 += 4.0

    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "data_flow")
