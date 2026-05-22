#!/usr/bin/env python3
"""Generate per-match evaluation dashboards from cached CSV/JSON data.

Fast (no video access needed). Produces a comprehensive single-image
overview per match for quick visual assessment of all pipeline components.

Usage:
    python scripts/generate_match_dashboard.py                    # All matches
    python scripts/generate_match_dashboard.py --matches Match1,Match6
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator
from soccer_summarization.utils import setup_logging

logger = logging.getLogger(__name__)


def format_mm_ss(seconds):
    """Format seconds as M:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def time_formatter(x, _):
    """Matplotlib axis formatter for MM:SS."""
    return format_mm_ss(x)


def load_match_data(match_dir: Path):
    """Load all cached pipeline outputs for a match."""
    data = {}

    # Shot boundaries
    p = match_dir / "shot_boundaries.csv"
    if p.exists():
        data["shots"] = pd.read_csv(p)

    # Per-frame player counts (raw, every 5th frame) — for per-shot avg bars
    p = match_dir / "player_counts.csv"
    if p.exists():
        data["player_counts"] = pd.read_csv(p)

    # Per-frame play classification (smoothed) — for the player-count line plot
    p = match_dir / "frame_classification.csv"
    if p.exists():
        data["frames"] = pd.read_csv(p)

    # Score events
    p = match_dir / "score_events.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            if len(df) > 0:
                data["score_events"] = df
        except pd.errors.EmptyDataError:
            pass

    # Highlights
    p = match_dir / "highlights.csv"
    if p.exists():
        data["highlights"] = pd.read_csv(p)

    # Goal-refined highlights
    p = match_dir / "goal_highlights.csv"
    if p.exists():
        data["goal_highlights"] = pd.read_csv(p)

    # Evaluation
    p = match_dir / "evaluation.json"
    if p.exists():
        with open(p) as f:
            data["eval"] = json.load(f)

    return data


def generate_dashboard(
    match_id: str,
    match_info: dict,
    match_dir: Path,
    gt_events: list,
    output_path: Path,
    config: PipelineConfig,
    goal_clips: list = None,
):
    """Generate a comprehensive dashboard PNG for one match."""
    data = load_match_data(match_dir)
    if "eval" not in data:
        logger.warning("No evaluation.json for %s, skipping", match_id)
        return

    ev = data["eval"]
    total_dur = ev["total_video_duration"]

    # Inject goal clips for timeline visualization if provided
    if goal_clips:
        data["goal_clips"] = [
            {"start_time": c.start_time, "end_time": c.end_time}
            for c in goal_clips
        ]

    # --- Figure layout: header / timeline / players / scores / goal-table ---
    fig = plt.figure(figsize=(24, 17), dpi=150)
    gs = gridspec.GridSpec(
        5, 2,
        figure=fig,
        height_ratios=[1.2, 1.5, 1.2, 1.2, 1.5],
        hspace=0.35,
        wspace=0.25,
        left=0.05, right=0.95, top=0.96, bottom=0.03,
    )

    colors = {
        "play": "#2196F3",
        "no_play": "#FFCDD2",
        "highlight": "#4CAF50",
        "goal_highlight": "#FF9800",
        "gt_goal": "#F44336",
        "pred_goal": "#2196F3",
        "tp": "#4CAF50",
        "fp": "#FF9800",
        "fn": "#F44336",
        "shot_long": "#90CAF9",
        "shot_short": "#E0E0E0",
    }

    # =============================================
    # Row 0: Header with match info and metrics
    # =============================================
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.axis("off")

    from matplotlib.patches import FancyBboxPatch

    home = match_info.get("home_team", "Home")
    away = match_info.get("away_team", "Away")
    score = match_info.get("score", "?-?")
    season = match_info.get("season", "?")

    title = f"{match_id}:  {home}  {score}  {away}  ({season})"
    ax_header.text(
        0.5, 0.94, title,
        transform=ax_header.transAxes, fontsize=18, fontweight="bold",
        ha="center", va="center",
    )

    # ── Pull metrics from evaluation.json ──────────────────────────────────────
    n_gt   = ev.get("num_gt_events", 0)
    n_pred = ev.get("num_predicted_goals", 0)
    tp, fp, fn = ev["true_positives"], ev["false_positives"], ev["false_negatives"]
    p, r, f1   = ev["precision"], ev["recall"], ev["f1"]

    ca_tp = ev.get("clip_aware_tp", 0)
    ca_fp = ev.get("clip_aware_fp", 0)
    ca_fn = ev.get("clip_aware_fn", 0)
    ca_p  = ev.get("clip_aware_precision", 0.0)
    ca_r  = ev.get("clip_aware_recall", 0.0)
    ca_f1 = ev.get("clip_aware_f1", 0.0)
    ca_thr = ev.get("clip_aware_iou_threshold", 0.5)

    has_clip = "clip_iou_mean" in ev
    ciou  = ev.get("clip_iou_mean", 0.0)
    ccov  = ev.get("clip_coverage_mean", 0.0)
    cpre  = ev.get("clip_precision_mean", 0.0)
    cs_mae = ev.get("clip_start_mae", 0.0)
    ce_mae = ev.get("clip_end_mae", 0.0)

    comp    = ev["compression_ratio"]
    hl_dur  = ev["highlight_duration"]
    n_clips = ev.get("num_goal_clips", 0)

    # Helpers for value colouring
    def q_color(v, good=0.9, ok=0.7):
        return "#2E7D32" if v >= good else "#EF6C00" if v >= ok else "#C62828"

    def err_color(v, good=2.0, ok=5.0):
        return "#2E7D32" if v <= good else "#EF6C00" if v <= ok else "#C62828"

    DARK = "#212121"
    NA   = "#9E9E9E"

    # ── Four cards: section title + body of (label, value, value_colour) rows ──
    cards = [
        {
            "title":    "GOAL DETECTION",
            "subtitle": "OCR timestamps vs GT",
            "color":    "#1565C0",
            "rows": [
                ("F1",          f"{f1:.3f}",          q_color(f1)),
                ("Precision",   f"{p:.3f}",           q_color(p)),
                ("Recall",      f"{r:.3f}",           q_color(r)),
                ("TP / FP / FN", f"{tp} / {fp} / {fn}", DARK),
                ("GT / OCR",    f"{n_gt} / {n_pred}", DARK),
            ],
        },
        {
            "title":    "CLIP QUALITY",
            "subtitle": "mean across detected goals (TPs)",
            "color":    "#6A1B9A",
            "rows": (
                [
                    ("Clip IoU",  f"{ciou:.3f}",   q_color(ciou, 0.7, 0.5)),
                    ("Coverage",  f"{ccov:.3f}",   q_color(ccov)),
                    ("Precision", f"{cpre:.3f}",   q_color(cpre)),
                    ("Start MAE", f"{cs_mae:.1f}s", err_color(cs_mae)),
                    ("End MAE",   f"{ce_mae:.1f}s", err_color(ce_mae)),
                ] if has_clip else [("—", "no frame-level GT", NA)]
            ),
        },
        {
            "title":    f"END-TO-END  (IoU ≥ {ca_thr:.1f})",
            "subtitle": "detection AND good clip",
            "color":    "#2E7D32",
            "rows": (
                [
                    ("F1",          f"{ca_f1:.3f}",                 q_color(ca_f1)),
                    ("Precision",   f"{ca_p:.3f}",                  q_color(ca_p)),
                    ("Recall",      f"{ca_r:.3f}",                  q_color(ca_r)),
                    ("TP / FP / FN", f"{ca_tp} / {ca_fp} / {ca_fn}", DARK),
                ] if has_clip else [("—", "no frame-level GT", NA)]
            ),
        },
        {
            "title":    "SUMMARY",
            "subtitle": "output size & duration",
            "color":    "#37474F",
            "rows": [
                ("Compression",        f"{comp*100:.1f}%",          q_color(comp, 0.9, 0.7)),
                ("Goal clips",         f"{n_clips}",                DARK),
                ("Total clip duration", f"{format_mm_ss(hl_dur)}",  DARK),
                ("Match duration",     f"{format_mm_ss(total_dur)}", DARK),
            ],
        },
    ]

    # Layout: 4 cards across the header subplot
    n_cards   = len(cards)
    x0, x1    = 0.02, 0.98
    gap       = 0.012
    card_w    = (x1 - x0 - (n_cards - 1) * gap) / n_cards
    card_top  = 0.83
    card_bot  = 0.04
    band_h    = 0.18

    for i, card in enumerate(cards):
        xc    = x0 + i * (card_w + gap)
        color = card["color"]

        # Body
        ax_header.add_patch(FancyBboxPatch(
            (xc, card_bot), card_w, card_top - card_bot,
            boxstyle="round,pad=0.0,rounding_size=0.008",
            facecolor="#FAFAFA", edgecolor=color, linewidth=1.5,
            transform=ax_header.transAxes, zorder=1, mutation_aspect=0.3,
        ))
        # Header band
        ax_header.add_patch(FancyBboxPatch(
            (xc, card_top - band_h), card_w, band_h,
            boxstyle="round,pad=0.0,rounding_size=0.008",
            facecolor=color, edgecolor=color, linewidth=0,
            transform=ax_header.transAxes, zorder=2, mutation_aspect=0.3,
        ))
        # Title
        ax_header.text(
            xc + card_w / 2, card_top - band_h * 0.32, card["title"],
            transform=ax_header.transAxes, fontsize=11, fontweight="bold",
            color="white", ha="center", va="center", zorder=3,
        )
        # Subtitle
        ax_header.text(
            xc + card_w / 2, card_top - band_h * 0.78, card["subtitle"],
            transform=ax_header.transAxes, fontsize=8, fontstyle="italic",
            color="white", ha="center", va="center", alpha=0.9, zorder=3,
        )
        # Rows
        rows = card["rows"]
        body_top = card_top - band_h - 0.03
        body_bot = card_bot + 0.03
        n_rows   = max(len(rows), 1)
        row_h    = (body_top - body_bot) / n_rows
        for j, (label, value, val_color) in enumerate(rows):
            y_line = body_top - (j + 0.5) * row_h
            ax_header.text(
                xc + 0.010, y_line, label,
                transform=ax_header.transAxes, fontsize=10,
                color="#424242", ha="left", va="center", zorder=3,
            )
            ax_header.text(
                xc + card_w - 0.010, y_line, value,
                transform=ax_header.transAxes, fontsize=11, fontweight="bold",
                color=val_color, ha="right", va="center", zorder=3,
            )

    # =============================================
    # Row 1: Full match timeline (4 lanes)
    # =============================================
    ax_timeline = fig.add_subplot(gs[1, :])
    _draw_timeline(ax_timeline, data, gt_events, total_dur, colors, config)

    # =============================================
    # Row 2: Player count time series
    # =============================================
    ax_players = fig.add_subplot(gs[2, :])
    _draw_player_counts(ax_players, data, gt_events, total_dur, colors, config)

    # =============================================
    # Row 3: Score tracking panel
    # =============================================
    ax_scores = fig.add_subplot(gs[3, :])
    _draw_score_tracking(ax_scores, data, gt_events, total_dur, colors, match_info)

    # =============================================
    # Row 4: Goal event comparison table
    # =============================================
    ax_table = fig.add_subplot(gs[4, :])
    _draw_goal_table(ax_table, data, gt_events, ev, colors, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Dashboard saved: %s", output_path)


def _draw_timeline(ax, data, gt_events, total_dur, colors, config):
    """4-lane timeline: shots, avg players (long shots), OCR + GT goals, highlights."""
    import matplotlib.ticker as mticker

    lanes = ["Shots", "Avg Players\n(long shots)", "Goals (GT + Pred)", "Highlights"]
    lane_y = {name: i for i, name in enumerate(lanes)}
    bar_h = 0.7

    min_dur = config.highlight_assembler.min_shot_duration
    play_thr = float(config.play_classifier.play_threshold)

    # Lane 0: Shot boundaries
    if "shots" in data:
        for _, row in data["shots"].iterrows():
            color = colors["shot_long"] if row["duration"] >= min_dur else colors["shot_short"]
            ax.barh(
                lane_y["Shots"], row["duration"], left=row["start_time"],
                height=bar_h, color=color, edgecolor="none", linewidth=0,
            )

    # Lane 1: Per-long-shot avg player count as vertical bars.
    # Bar height is proportional to avg players (normalized by the play threshold
    # so that bars >= threshold reach the top half of the lane).  Color encodes
    # play (avg>=threshold) vs no-play (avg<threshold).
    if "shots" in data and "player_counts" in data:
        pc = data["player_counts"]
        lane_baseline = lane_y["Avg Players\n(long shots)"] - 0.45
        lane_height = 0.9

        # Compute per-shot avg for ALL long shots
        bars = []
        for _, s in data["shots"].iterrows():
            if s["duration"] < min_dur:
                continue
            mask = (pc["frame"] >= s["start_frame"]) & (pc["frame"] <= s["end_frame"])
            vals = pc.loc[mask, "player_count"]
            if len(vals) == 0:
                continue
            avg = float(vals.mean())
            bars.append((s["start_time"], s["end_time"], avg))

        if bars:
            # Normalize: 2× threshold maps to full lane height
            max_avg = max(2.0 * play_thr, max(a for *_, a in bars))
            for st, en, avg in bars:
                h = min(avg / max_avg, 1.0) * lane_height
                color = colors["play"] if avg >= play_thr else colors["no_play"]
                ax.bar(
                    x=(st + en) / 2.0, height=h, width=en - st,
                    bottom=lane_baseline,
                    color=color, alpha=0.75, edgecolor="none", linewidth=0,
                )

    # Lane 2: GT goals (scoreboard events) + clip windows + predicted goals
    for evt in gt_events:
        # Scoreboard update marker (vertical tick)
        ax.axvline(evt.start_time, ymin=lane_y["Goals (GT + Pred)"] / len(lanes),
                   ymax=(lane_y["Goals (GT + Pred)"] + 1) / len(lanes),
                   color=colors["gt_goal"], linewidth=2, alpha=0.8, zorder=4)
    # GT clip windows [start_frame→end_frame_shot] if available
    if "goal_clips" in data:
        for clip in data["goal_clips"]:
            dur = clip["end_time"] - clip["start_time"]
            ax.barh(
                lane_y["Goals (GT + Pred)"], dur, left=clip["start_time"],
                height=bar_h, color="#E91E63", alpha=0.3,
                edgecolor="#E91E63", linewidth=1, label="_nolegend_",
            )
    if "score_events" in data:
        for _, row in data["score_events"].iterrows():
            ax.plot(
                row["timestamp"], lane_y["Goals (GT + Pred)"],
                marker="v", color=colors["pred_goal"], markersize=8, zorder=5,
            )

    # Lane 3: Highlights
    if "highlights" in data:
        for _, row in data["highlights"].iterrows():
            ax.barh(
                lane_y["Highlights"], row["duration"], left=row["start_time"],
                height=bar_h * 0.5, color=colors["highlight"], edgecolor="none",
                alpha=0.6,
            )
    # Goal-refined highlights (thinner, on top)
    if "goal_highlights" in data:
        for _, row in data["goal_highlights"].iterrows():
            ax.barh(
                lane_y["Highlights"] - 0.25, row["duration"], left=row["start_time"],
                height=bar_h * 0.4, color=colors["goal_highlight"], edgecolor="none",
                alpha=0.8,
            )

    ax.set_yticks(range(len(lanes)))
    ax.set_yticklabels(lanes, fontsize=10)
    ax.set_xlim(0, total_dur)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
    ax.set_xlabel("Match Time")
    ax.set_title("Full Match Timeline", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Legend
    legend_items = [
        mpatches.Patch(color=colors["shot_long"], label=f"Long shot (≥{config.highlight_assembler.min_shot_duration}s)"),
        mpatches.Patch(color=colors["shot_short"], label="Short shot"),
        mpatches.Patch(color=colors["play"], label=f"Avg players ≥{play_thr:.0f} (play)"),
        mpatches.Patch(color=colors["no_play"], label=f"Avg players <{play_thr:.0f} (no-play)"),
        mpatches.Patch(color=colors["gt_goal"], alpha=0.4, label="GT goal"),
        plt.Line2D([0], [0], marker="v", color=colors["pred_goal"], linestyle="", label="Predicted goal"),
        mpatches.Patch(color=colors["highlight"], alpha=0.6, label="Highlights (play long shots)"),
        mpatches.Patch(color=colors["goal_highlight"], alpha=0.8, label="Goal-refined"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8, ncol=4)


def _draw_player_counts(ax, data, gt_events, total_dur, colors, config):
    """Player count time series with play threshold and GT goal markers."""
    import matplotlib.ticker as mticker

    if "frames" not in data:
        ax.text(0.5, 0.5, "No frame classification data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Player Count Over Time", fontsize=12, fontweight="bold")
        return

    df = data["frames"]
    # Downsample for performance (every 25th frame = 1 per second)
    step = max(1, len(df) // 6000)
    df_ds = df.iloc[::step]

    # Plot raw and smoothed
    ax.plot(df_ds["time_seconds"], df_ds["player_count"], color="#BBDEFB",
            alpha=0.4, linewidth=0.5, label="Raw count")
    if "player_count_smooth" in df_ds.columns:
        ax.plot(df_ds["time_seconds"], df_ds["player_count_smooth"], color="#1565C0",
                linewidth=1.0, label="Smoothed")

    # Threshold line
    threshold = config.play_classifier.play_threshold
    ax.axhline(threshold, color="#F44336", linestyle="--", linewidth=1, alpha=0.7,
               label=f"Threshold ({threshold})")

    # GT goal markers
    for evt in gt_events:
        ax.axvline(evt.midpoint, color=colors["gt_goal"], linestyle=":", linewidth=1, alpha=0.6)

    ax.set_xlim(0, total_dur)
    ax.set_ylim(0, min(30, df["player_count"].max() * 1.1) if len(df) > 0 else 30)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
    ax.set_ylabel("Player Count")
    ax.set_title("Player Count & Play Classification", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)


def _draw_score_tracking(ax, data, gt_events, total_dur, colors, match_info):
    """Score over time: predicted events vs GT goals."""
    import matplotlib.ticker as mticker

    # GT goal shading
    for evt in gt_events:
        ax.axvspan(evt.start_time, evt.end_time, alpha=0.15, color=colors["gt_goal"], linewidth=0)
        ax.axvline(evt.midpoint, color=colors["gt_goal"], linestyle=":", linewidth=1, alpha=0.5)

    if "score_events" not in data:
        ax.text(0.5, 0.5, "No score events detected (OCR found no goals)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Score Tracking", fontsize=12, fontweight="bold")
        ax.set_xlim(0, total_dur)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
        return

    se = data["score_events"]

    # Build step function for home and away scores
    times_h = [0] + se["timestamp"].tolist() + [total_dur]
    scores_h = [0] + se["home_score"].tolist() + [se["home_score"].iloc[-1]]
    scores_a = [0] + se["away_score"].tolist() + [se["away_score"].iloc[-1]]

    ax.step(times_h, scores_h, where="post", color="#1565C0", linewidth=2,
            label=f"Home ({match_info.get('home_team', 'Home')})")
    ax.step(times_h, scores_a, where="post", color="#C62828", linewidth=2,
            label=f"Away ({match_info.get('away_team', 'Away')})")

    # Mark each detected event
    for _, row in se.iterrows():
        marker_color = "#1565C0" if row["team"] == "home" else "#C62828"
        score_val = row["home_score"] if row["team"] == "home" else row["away_score"]
        ax.plot(row["timestamp"], score_val, "o", color=marker_color, markersize=8, zorder=5)
        ax.annotate(
            f"{format_mm_ss(row['timestamp'])}",
            (row["timestamp"], score_val),
            textcoords="offset points", xytext=(5, 8), fontsize=7,
            color=marker_color, fontweight="bold",
        )

    # Parse expected final score
    expected = match_info.get("score", "0-0")
    try:
        exp_h, exp_a = [int(x) for x in expected.split("-")]
        max_score = max(exp_h, exp_a, 1)
    except ValueError:
        max_score = max(se["home_score"].max(), se["away_score"].max(), 1)

    ax.set_xlim(0, total_dur)
    ax.set_ylim(-0.3, max_score + 0.8)
    ax.set_yticks(range(max_score + 1))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
    ax.set_ylabel("Score")
    ax.set_title("Score Tracking (OCR Events vs GT Goals)", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)


def _draw_goal_table(ax, data, gt_events, ev, colors, config):
    """Per-goal table: detection + clip-quality breakdown per GT goal."""
    ax.axis("off")

    tolerance = config.evaluation.tolerance_window

    score_events = data.get("score_events", pd.DataFrame())
    pred_times = score_events["timestamp"].tolist() if len(score_events) > 0 else []

    # Per-GT-clip clip-quality arrays (None for undetected goals)
    iou_pg   = ev.get("clip_iou_per_gt",       [None] * len(gt_events))
    cov_pg   = ev.get("clip_coverage_per_gt",  [None] * len(gt_events))
    pre_pg   = ev.get("clip_precision_per_gt", [None] * len(gt_events))
    s_mae_pg = ev.get("clip_start_mae_per_gt", [None] * len(gt_events))
    e_mae_pg = ev.get("clip_end_mae_per_gt",   [None] * len(gt_events))
    iou_thr  = ev.get("clip_aware_iou_threshold", 0.5)

    def f(v, fmt="{:.3f}"):
        return "-" if v is None else fmt.format(v)

    rows = []
    matched_preds = set()

    for i, evt in enumerate(gt_events):
        # Find the closest predicted goal within tolerance
        best_pred, best_dist, best_idx = None, float("inf"), -1
        for j, pt in enumerate(pred_times):
            dist = abs(pt - evt.midpoint)
            if dist < best_dist and dist <= tolerance:
                best_dist, best_pred, best_idx = dist, pt, j

        # End-to-end status: TP requires detection AND clip IoU >= threshold
        iou_v = iou_pg[i] if i < len(iou_pg) else None
        if best_pred is not None and best_idx not in matched_preds:
            matched_preds.add(best_idx)
            if iou_v is not None and iou_v >= iou_thr:
                status   = "TP"
                pred_str = format_mm_ss(best_pred)
                err_str  = f"{best_dist:.1f}s"
            else:
                status   = "FN (clip)"  # detected but boundaries too off
                pred_str = format_mm_ss(best_pred)
                err_str  = f"{best_dist:.1f}s"
        else:
            status, pred_str, err_str = "FN", "-", "-"

        rows.append([
            f"Goal {i+1}",
            format_mm_ss(evt.midpoint),
            pred_str,
            err_str,
            f(iou_v),
            f(cov_pg[i] if i < len(cov_pg) else None),
            f(pre_pg[i] if i < len(pre_pg) else None),
            f(s_mae_pg[i] if i < len(s_mae_pg) else None, "{:.1f}s"),
            f(e_mae_pg[i] if i < len(e_mae_pg) else None, "{:.1f}s"),
            status,
        ])

    # FP predictions (no matching GT)
    for j, pt in enumerate(pred_times):
        if j not in matched_preds:
            rows.append([
                f"FP {j+1}", "-", format_mm_ss(pt), "-", "-", "-", "-", "-", "-", "FP",
            ])

    if not rows:
        ax.text(0.5, 0.5, "No goal events in this match",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Per-Goal Breakdown", fontsize=12, fontweight="bold")
        return

    columns = [
        "Event", "GT Mid", "Predicted", "Time Err",
        "Clip IoU", "Coverage", "Precision", "Start MAE", "End MAE",
        "Status",
    ]

    table = ax.table(
        cellText=rows, colLabels=columns,
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.9)

    # Status-row coloring
    status_colors = {
        "TP":         "#C8E6C9",
        "FP":         "#FFE0B2",
        "FN":         "#FFCDD2",
        "FN (clip)":  "#FFF59D",  # detected but bad clip — yellow
    }
    for i, row in enumerate(rows):
        bg = status_colors.get(row[-1], "#FFFFFF")
        for j in range(len(columns)):
            table[i + 1, j].set_facecolor(bg)

    # Header styling
    for j in range(len(columns)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title(
        f"Per-Goal Breakdown   (TP = detected AND Clip IoU ≥ {iou_thr:.1f})",
        fontsize=12, fontweight="bold", pad=20,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate per-match evaluation dashboards")
    parser.add_argument("--matches", help="Comma-separated match IDs (default: all)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--matches-yaml", default=None,
                        help="Path to matches YAML (default: config/matches.yaml)")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--output-dir", default="results/dashboards")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = PipelineConfig.from_yaml(args.config)
    evaluator = Evaluator(config.evaluation)

    matches_yaml_path = Path(args.matches_yaml) if args.matches_yaml else \
        Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml_path) as f:
        all_matches = yaml.safe_load(f)["matches"]

    if args.matches:
        match_ids = [m.strip() for m in args.matches.split(",")]
    else:
        match_ids = list(all_matches.keys())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for match_id in match_ids:
        if match_id not in all_matches:
            logger.warning("Unknown match: %s", match_id)
            continue

        info = all_matches[match_id]
        match_dir = Path(args.results_dir) / match_id

        if not match_dir.exists():
            logger.warning("No results for %s", match_id)
            continue

        # Load GT events — prefer frame-level goals.csv when available
        gt_path = str(Path(config.data_dir) / info["gt_path"])
        goals_path = str(Path(config.data_dir) / info["goals_path"]) \
            if "goals_path" in info else None
        goal_clips = []
        # If the config sets goal_clip_max_duration (last-10s mode), the GT
        # clips are truncated so the dashboard's GT-window visualisation
        # matches the metric numbers.
        gt_clip_max = config.highlight_assembler.goal_clip_max_duration
        try:
            if goals_path and Path(goals_path).exists():
                gt_events = evaluator.load_ground_truth_frame_level(goals_path, 25.0)
                goal_clips = Evaluator.load_goal_clips(
                    goals_path, 25.0, max_duration=gt_clip_max,
                )
            else:
                gt_df = evaluator.load_ground_truth(gt_path)
                gt_events = evaluator.extract_goal_events(gt_df)
        except Exception as e:
            logger.warning("Cannot load GT for %s: %s", match_id, e)
            gt_events = []

        output_path = output_dir / f"{match_id}_dashboard.png"
        logger.info("Generating dashboard for %s...", match_id)

        try:
            generate_dashboard(
                match_id=match_id,
                match_info=info,
                match_dir=match_dir,
                gt_events=gt_events,
                output_path=output_path,
                config=config,
                goal_clips=goal_clips,
            )
        except Exception as e:
            logger.error("Failed for %s: %s", match_id, e, exc_info=True)

    logger.info("Done. Dashboards saved to %s", output_dir)


if __name__ == "__main__":
    main()
