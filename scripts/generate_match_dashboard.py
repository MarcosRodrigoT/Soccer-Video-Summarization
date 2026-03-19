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

    # Player counts (frame-level)
    p = match_dir / "frame_classification.csv"
    if p.exists():
        data["frames"] = pd.read_csv(p)

    # Play segments
    p = match_dir / "play_segments.csv"
    if p.exists():
        data["play_segments"] = pd.read_csv(p)

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
):
    """Generate a comprehensive dashboard PNG for one match."""
    data = load_match_data(match_dir)
    if "eval" not in data:
        logger.warning("No evaluation.json for %s, skipping", match_id)
        return

    ev = data["eval"]
    total_dur = ev["total_video_duration"]

    # --- Figure layout ---
    fig = plt.figure(figsize=(24, 20), dpi=150)
    gs = gridspec.GridSpec(
        7, 2,
        figure=fig,
        height_ratios=[0.6, 1.5, 1.2, 1.2, 1.2, 0.8, 1.5],
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

    home = match_info.get("home_team", "Home")
    away = match_info.get("away_team", "Away")
    score = match_info.get("score", "?-?")
    season = match_info.get("season", "?")

    title = f"{match_id}:  {home}  {score}  {away}  ({season})"
    ax_header.text(
        0.5, 0.75, title,
        transform=ax_header.transAxes, fontsize=18, fontweight="bold",
        ha="center", va="center",
    )

    # Metrics bar
    f1 = ev["f1"]
    tp, fp, fn = ev["true_positives"], ev["false_positives"], ev["false_negatives"]
    comp = ev["compression_ratio"]
    tiou = ev["mean_tiou"]
    mae = ev["goal_timing_mae"]
    n_gt = ev["num_gt_events"]
    n_pred = ev["num_predicted_goals"]

    f1_color = "#4CAF50" if f1 >= 1.0 else "#FF9800" if f1 >= 0.8 else "#F44336"
    metrics_text = (
        f"F1 = {f1:.3f}   |   "
        f"TP={tp}  FP={fp}  FN={fn}   |   "
        f"GT goals={n_gt}  Predicted={n_pred}   |   "
        f"Compression={comp*100:.1f}%   |   "
        f"tIoU={tiou:.3f}   |   "
        f"MAE={mae:.1f}s   |   "
        f"Duration={format_mm_ss(total_dur)}"
    )
    ax_header.text(
        0.5, 0.25, metrics_text,
        transform=ax_header.transAxes, fontsize=11, ha="center", va="center",
        color=f1_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F5F5F5", edgecolor=f1_color, alpha=0.8),
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
    # Row 4: Highlight coverage heatmap
    # =============================================
    ax_coverage = fig.add_subplot(gs[4, :])
    _draw_highlight_coverage(ax_coverage, data, gt_events, total_dur, colors)

    # =============================================
    # Row 5: Minute-by-minute coverage bar
    # =============================================
    ax_minute = fig.add_subplot(gs[5, :])
    _draw_minute_coverage(ax_minute, data, gt_events, total_dur, colors)

    # =============================================
    # Row 6: Goal event comparison table
    # =============================================
    ax_table = fig.add_subplot(gs[6, :])
    _draw_goal_table(ax_table, data, gt_events, ev, colors, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Dashboard saved: %s", output_path)


def _draw_timeline(ax, data, gt_events, total_dur, colors, config):
    """4-lane timeline: shots, play/no-play, OCR events + GT, highlights."""
    import matplotlib.ticker as mticker

    lanes = ["Shots", "Play/No-Play", "Goals (GT + Pred)", "Highlights"]
    lane_y = {name: i for i, name in enumerate(lanes)}
    bar_h = 0.7

    # Lane 0: Shot boundaries
    if "shots" in data:
        min_dur = config.highlight_assembler.min_shot_duration
        for _, row in data["shots"].iterrows():
            color = colors["shot_long"] if row["duration"] >= min_dur else colors["shot_short"]
            ax.barh(
                lane_y["Shots"], row["duration"], left=row["start_time"],
                height=bar_h, color=color, edgecolor="none", linewidth=0,
            )

    # Lane 1: Play segments
    if "play_segments" in data:
        for _, row in data["play_segments"].iterrows():
            color = colors["play"] if row["is_play"] else colors["no_play"]
            ax.barh(
                lane_y["Play/No-Play"], row["duration"], left=row["start_time"],
                height=bar_h, color=color, edgecolor="none", linewidth=0,
            )

    # Lane 2: GT goals + predicted goals
    for evt in gt_events:
        ax.barh(
            lane_y["Goals (GT + Pred)"], evt.end_time - evt.start_time,
            left=evt.start_time, height=bar_h, color=colors["gt_goal"],
            alpha=0.4, edgecolor=colors["gt_goal"], linewidth=1,
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
        mpatches.Patch(color=colors["play"], label="Play"),
        mpatches.Patch(color=colors["no_play"], label="No-play"),
        mpatches.Patch(color=colors["gt_goal"], alpha=0.4, label="GT goal"),
        plt.Line2D([0], [0], marker="v", color=colors["pred_goal"], linestyle="", label="Predicted goal"),
        mpatches.Patch(color=colors["highlight"], alpha=0.6, label="Highlights"),
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

    # Shade play/no-play regions
    if "play_segments" in data:
        for _, row in data["play_segments"].iterrows():
            color = colors["play"] if row["is_play"] else colors["no_play"]
            ax.axvspan(row["start_time"], row["end_time"], alpha=0.15, color=color, linewidth=0)

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


def _draw_highlight_coverage(ax, data, gt_events, total_dur, colors):
    """Highlight segments as blocks with GT goals overlaid."""
    import matplotlib.ticker as mticker

    # GT goal shading
    for evt in gt_events:
        ax.axvspan(evt.start_time, evt.end_time, alpha=0.2, color=colors["gt_goal"], linewidth=0)

    # Standard highlights
    if "highlights" in data:
        for _, row in data["highlights"].iterrows():
            ax.barh(0, row["duration"], left=row["start_time"], height=0.6,
                    color=colors["highlight"], alpha=0.5, edgecolor="none")

    # Goal-refined highlights
    if "goal_highlights" in data:
        for _, row in data["goal_highlights"].iterrows():
            ax.barh(1, row["duration"], left=row["start_time"], height=0.6,
                    color=colors["goal_highlight"], alpha=0.7, edgecolor="none")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Standard\nHighlights", "Goal-Refined\nHighlights"], fontsize=9)
    ax.set_xlim(0, total_dur)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(time_formatter))
    ax.set_title("Highlight Coverage (red shading = GT goal segments)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()


def _draw_minute_coverage(ax, data, gt_events, total_dur, colors):
    """Minute-by-minute highlight coverage density bar."""
    n_minutes = int(np.ceil(total_dur / 60))
    coverage = np.zeros(n_minutes)

    if "highlights" in data:
        for _, row in data["highlights"].iterrows():
            start_min = int(row["start_time"] / 60)
            end_min = min(int(row["end_time"] / 60), n_minutes - 1)
            for m in range(start_min, end_min + 1):
                min_start = m * 60
                min_end = (m + 1) * 60
                overlap = max(0, min(row["end_time"], min_end) - max(row["start_time"], min_start))
                coverage[m] += overlap / 60.0

    coverage = np.clip(coverage, 0, 1)

    # Color bars by coverage level
    bar_colors = []
    for c in coverage:
        if c > 0.8:
            bar_colors.append(colors["highlight"])
        elif c > 0.3:
            bar_colors.append("#81C784")
        elif c > 0:
            bar_colors.append("#C8E6C9")
        else:
            bar_colors.append("#F5F5F5")

    ax.bar(range(n_minutes), coverage, color=bar_colors, edgecolor="#E0E0E0", linewidth=0.5)

    # Mark GT goal minutes
    for evt in gt_events:
        goal_min = int(evt.midpoint / 60)
        if 0 <= goal_min < n_minutes:
            ax.axvline(goal_min, color=colors["gt_goal"], linestyle="-", linewidth=2, alpha=0.6)

    ax.set_xlim(-0.5, n_minutes - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Match Minute")
    ax.set_ylabel("Coverage")
    ax.set_title("Per-Minute Highlight Density (red lines = GT goal minutes)", fontsize=11, fontweight="bold")


def _draw_goal_table(ax, data, gt_events, ev, colors, config):
    """Goal event comparison table: GT vs predicted, TP/FP/FN classification."""
    ax.axis("off")

    tolerance = config.evaluation.tolerance_window

    # Classify each GT event
    score_events = data.get("score_events", pd.DataFrame())
    pred_times = score_events["timestamp"].tolist() if len(score_events) > 0 else []

    rows = []
    matched_preds = set()

    for i, evt in enumerate(gt_events):
        best_pred = None
        best_dist = float("inf")
        best_idx = -1
        for j, pt in enumerate(pred_times):
            dist = abs(pt - evt.midpoint)
            if dist < best_dist and dist <= tolerance:
                best_dist = dist
                best_pred = pt
                best_idx = j

        if best_pred is not None and best_idx not in matched_preds:
            matched_preds.add(best_idx)
            status = "TP"
            pred_str = format_mm_ss(best_pred)
            error_str = f"{best_dist:.1f}s"
            tiou_str = f"{ev['tiou_per_event'][i]:.3f}" if i < len(ev.get("tiou_per_event", [])) else "-"
        else:
            status = "FN"
            pred_str = "-"
            error_str = "-"
            tiou_str = f"{ev['tiou_per_event'][i]:.3f}" if i < len(ev.get("tiou_per_event", [])) else "-"

        rows.append([
            f"Goal {i+1}",
            f"{format_mm_ss(evt.start_time)} - {format_mm_ss(evt.end_time)}",
            format_mm_ss(evt.midpoint),
            pred_str,
            error_str,
            tiou_str,
            status,
        ])

    # Add FP predictions
    for j, pt in enumerate(pred_times):
        if j not in matched_preds:
            rows.append([
                f"FP {j+1}",
                "-",
                "-",
                format_mm_ss(pt),
                "-",
                "-",
                "FP",
            ])

    if not rows:
        ax.text(0.5, 0.5, "No goal events in this match",
                transform=ax.transAxes, ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Goal Event Comparison", fontsize=12, fontweight="bold")
        return

    columns = ["Event", "GT Window", "GT Mid", "Predicted", "Error", "tIoU", "Status"]

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Color cells by status
    status_colors = {"TP": "#C8E6C9", "FP": "#FFE0B2", "FN": "#FFCDD2"}
    for i, row in enumerate(rows):
        status = row[-1]
        bg = status_colors.get(status, "#FFFFFF")
        for j in range(len(columns)):
            table[i + 1, j].set_facecolor(bg)

    # Header styling
    for j in range(len(columns)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Goal Event Comparison (GT vs Predicted)", fontsize=12, fontweight="bold", pad=20)


def main():
    parser = argparse.ArgumentParser(description="Generate per-match evaluation dashboards")
    parser.add_argument("--matches", help="Comma-separated match IDs (default: all)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--output-dir", default="results/dashboards")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)

    config = PipelineConfig.from_yaml(args.config)
    evaluator = Evaluator(config.evaluation)

    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
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

        # Load GT events
        gt_path = str(Path(config.data_dir) / info["gt_path"])
        try:
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
            )
        except Exception as e:
            logger.error("Failed for %s: %s", match_id, e, exc_info=True)

    logger.info("Done. Dashboards saved to %s", output_dir)


if __name__ == "__main__":
    main()
