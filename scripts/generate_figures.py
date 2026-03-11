#!/usr/bin/env python3
"""Generate all paper figures using matplotlib.

Figures:
    1. Pipeline architecture (external, polish only)
    2. Dataset overview — goals distribution, seasons
    3. Scoreboard detection example (external, polish only)
    4. Full-match timeline visualization
    5. Play threshold ablation plot
    6. Smoothing window ablation plot
    7. Speed-accuracy tradeoff (frame_step)
    8. Tolerance window sensitivity
    9. Error analysis examples (requires frame extraction)
   10. Player count distribution

Usage:
    python scripts/generate_figures.py [--figures all] [--output-dir results/figures]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.utils import setup_logging

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Journal-quality defaults
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.figsize": (7, 4),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def fig_dataset_overview(matches_yaml_path: Path, results_dir: Path, output_path: Path):
    """Figure 2: Dataset overview — goals per match, season distribution."""
    import yaml
    with open(matches_yaml_path) as f:
        matches = yaml.safe_load(f)["matches"]

    match_ids = sorted(matches.keys(), key=lambda x: int(x.replace("Match", "")))
    goals = []
    seasons = []
    for mid in match_ids:
        info = matches[mid]
        score = info.get("score", "0-0")
        parts = score.replace(" ", "").split("-")
        try:
            total = int(parts[0]) + int(parts[1])
        except (ValueError, IndexError):
            total = 0
        goals.append(total)
        seasons.append(info.get("season", "unknown"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Goals per match
    ax = axes[0]
    colors = ["#2196F3" if s != "2013/14" else "#FF9800" for s in seasons]
    bars = ax.bar(range(len(goals)), goals, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Match")
    ax.set_ylabel("Number of Goals")
    ax.set_title("Goals per Match")
    ax.set_xticks(range(0, len(goals), 5))
    ax.set_xticklabels([match_ids[i] for i in range(0, len(goals), 5)], rotation=45, ha="right")
    ax.axhline(y=np.mean(goals), color="red", linestyle="--", alpha=0.7, label=f"Mean={np.mean(goals):.1f}")
    ax.legend()

    # Season distribution
    ax = axes[1]
    season_counts = {}
    for s in seasons:
        season_counts[s] = season_counts.get(s, 0) + 1
    sorted_seasons = sorted(season_counts.keys())
    ax.bar(range(len(sorted_seasons)), [season_counts[s] for s in sorted_seasons],
           color="#4CAF50", edgecolor="white")
    ax.set_xticks(range(len(sorted_seasons)))
    ax.set_xticklabels(sorted_seasons, rotation=45, ha="right")
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of Matches")
    ax.set_title("Matches per Season")

    fig.suptitle(f"Dataset Overview: {len(match_ids)} La Liga Matches, {sum(goals)} Total Goals", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig_match_timeline(results_dir: Path, match_id: str, output_path: Path):
    """Figure 4: Full-match timeline visualization."""
    match_dir = results_dir / match_id

    # Load data
    play_df = pd.read_csv(match_dir / "play_segments.csv")
    shot_df = pd.read_csv(match_dir / "shot_boundaries.csv")
    hl_df = pd.read_csv(match_dir / "highlights.csv")

    score_path = match_dir / "score_events.csv"
    try:
        score_df = pd.read_csv(score_path) if score_path.exists() else pd.DataFrame()
    except pd.errors.EmptyDataError:
        score_df = pd.DataFrame()

    eval_json = match_dir / "evaluation.json"
    with open(eval_json) as f:
        eval_data = json.load(f)
    total_dur = eval_data["total_video_duration"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 1, 1.5]})

    time_min = np.arange(0, total_dur) / 60  # Convert to minutes

    # Row 1: Shot boundaries
    ax = axes[0]
    for _, row in shot_df.iterrows():
        ax.axvspan(row["start_time"]/60, row["end_time"]/60,
                   alpha=0.3, color="#2196F3", linewidth=0)
    ax.set_ylabel("Shots")
    ax.set_yticks([])
    ax.set_title(f"Match Timeline: {match_id}", fontsize=11)

    # Row 2: Play/no-play segments
    ax = axes[1]
    for _, row in play_df.iterrows():
        color = "#4CAF50" if row["is_play"] else "#F44336"
        alpha = 0.6 if row["is_play"] else 0.2
        ax.axvspan(row["start_time"]/60, row["end_time"]/60,
                   alpha=alpha, color=color, linewidth=0)
    ax.set_ylabel("Play")
    ax.set_yticks([])
    play_patch = mpatches.Patch(color="#4CAF50", alpha=0.6, label="Play")
    noplay_patch = mpatches.Patch(color="#F44336", alpha=0.2, label="No-play")
    ax.legend(handles=[play_patch, noplay_patch], loc="upper right", ncol=2)

    # Row 3: Score events (OCR)
    ax = axes[2]
    if not score_df.empty and "timestamp" in score_df.columns:
        for _, row in score_df.iterrows():
            ax.axvline(row["timestamp"]/60, color="#FF9800", linewidth=2, alpha=0.8)
    ax.set_ylabel("OCR Goals")
    ax.set_yticks([])

    # Row 4: Highlight segments
    ax = axes[3]
    for _, row in hl_df.iterrows():
        ax.axvspan(row["start_time"]/60, row["end_time"]/60,
                   alpha=0.5, color="#9C27B0", linewidth=0)
    ax.set_ylabel("Highlights")
    ax.set_yticks([])
    ax.set_xlabel("Time (minutes)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig_ablation_plot(ablation_file: Path, param_name: str, output_path: Path):
    """Figures 5-6: Ablation line plots."""
    with open(ablation_file) as f:
        data = json.load(f)

    values = []
    f1s, precisions, recalls = [], [], []
    compressions, coverages = [], []

    for key in sorted(data.keys(), key=lambda k: float(data[k]["value"])):
        val = data[key]
        values.append(val["value"])
        agg = val["aggregate"]
        f1s.append(agg["f1_mean"])
        precisions.append(agg["precision_mean"])
        recalls.append(agg["recall_mean"])
        compressions.append(agg["compression_mean"] * 100)
        coverages.append(agg["goal_coverage_mean"] * 100)

    # Find default
    default_idx = None
    for i, key in enumerate(sorted(data.keys(), key=lambda k: float(data[k]["value"]))):
        if data[key].get("is_default"):
            default_idx = i
            break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: P/R/F1
    ax1.plot(values, f1s, "o-", color="#2196F3", label="F1", linewidth=2, markersize=6)
    ax1.plot(values, precisions, "s--", color="#4CAF50", label="Precision", linewidth=1.5, markersize=5)
    ax1.plot(values, recalls, "^--", color="#FF9800", label="Recall", linewidth=1.5, markersize=5)
    if default_idx is not None:
        ax1.axvline(values[default_idx], color="gray", linestyle=":", alpha=0.5, label="Default")
    ax1.set_xlabel(param_name.replace("_", " ").title())
    ax1.set_ylabel("Score")
    ax1.set_title("Goal Detection Performance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Compression and Coverage
    ax2.plot(values, compressions, "o-", color="#9C27B0", label="Compression %", linewidth=2, markersize=6)
    ax2.plot(values, coverages, "s-", color="#E91E63", label="Goal Coverage %", linewidth=2, markersize=6)
    if default_idx is not None:
        ax2.axvline(values[default_idx], color="gray", linestyle=":", alpha=0.5, label="Default")
    ax2.set_xlabel(param_name.replace("_", " ").title())
    ax2.set_ylabel("Percentage")
    ax2.set_title("Highlight Quality")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"Ablation: {param_name.replace('_', ' ').title()}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig_speed_accuracy(ablation_file: Path, output_path: Path):
    """Figure 7: Speed-accuracy tradeoff (frame_step)."""
    with open(ablation_file) as f:
        data = json.load(f)

    values, f1s = [], []
    for key in sorted(data.keys(), key=lambda k: float(data[k]["value"])):
        val = data[key]
        values.append(val["value"])
        f1s.append(val["aggregate"]["f1_mean"])

    # Speed is inversely proportional to frame_step
    relative_speeds = [v / min(values) for v in values]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    color1 = "#2196F3"
    ax1.plot(values, f1s, "o-", color=color1, linewidth=2, markersize=8, label="F1 Score")
    ax1.set_xlabel("Frame Step (every Nth frame)")
    ax1.set_ylabel("F1 Score", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "#FF9800"
    ax2.plot(values, relative_speeds, "s--", color=color2, linewidth=2, markersize=8, label="Relative Speed")
    ax2.set_ylabel("Relative Speed (×)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Speed-Accuracy Tradeoff (YOLO Frame Sampling)", fontsize=12)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig_tolerance_sensitivity(ablation_file: Path, output_path: Path):
    """Figure 8: Tolerance window sensitivity."""
    with open(ablation_file) as f:
        data = json.load(f)

    values, f1s = [], []
    for key in sorted(data.keys(), key=lambda k: float(data[k]["value"])):
        val = data[key]
        values.append(val["value"])
        f1s.append(val["aggregate"]["f1_mean"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(values, f1s, "o-", color="#2196F3", linewidth=2, markersize=8)

    # Mark default
    default_val = 30
    if default_val in values:
        idx = values.index(default_val)
        ax.plot(default_val, f1s[idx], "o", color="#F44336", markersize=12, zorder=5,
                label=f"Default ({default_val}s)")

    ax.set_xlabel("Tolerance Window (seconds)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Evaluation Protocol Sensitivity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig_player_count_distribution(results_dir: Path, match_id: str, output_path: Path):
    """Figure 10: Player count distribution with bimodal pattern."""
    match_dir = results_dir / match_id

    frame_path = match_dir / "frame_classification.csv"
    if not frame_path.exists():
        print(f"  Skipped (no frame_classification.csv for {match_id})")
        return

    df = pd.read_csv(frame_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Histogram of smoothed counts
    if "smoothed_count" in df.columns:
        counts = df["smoothed_count"].dropna()
    elif "player_count" in df.columns:
        counts = df["player_count"].dropna()
    else:
        print(f"  Skipped (no count column for {match_id})")
        return

    ax1.hist(counts, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    ax1.axvline(x=5, color="#F44336", linewidth=2, linestyle="--", label="Threshold (5)")
    ax1.set_xlabel("Player Count (smoothed)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Player Count Distribution")
    ax1.legend()

    # Time series
    time_minutes = np.arange(len(counts)) / (25 / 5) / 60  # Assuming 5fps effective
    ax2.plot(time_minutes, counts, color="#2196F3", alpha=0.3, linewidth=0.5)
    ax2.axhline(y=5, color="#F44336", linewidth=2, linestyle="--", label="Threshold (5)")
    ax2.set_xlabel("Time (minutes)")
    ax2.set_ylabel("Player Count (smoothed)")
    ax2.set_title("Player Count Over Time")
    ax2.legend()

    fig.suptitle(f"Player Count Analysis: {match_id}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def fig_baselines_comparison(baseline_file: Path, output_path: Path):
    """Generate baseline comparison bar chart."""
    with open(baseline_file) as f:
        baselines = json.load(f)

    names = []
    f1s, compressions = [], []
    for bl_name in ["random", "uniform", "shot_only", "play_only", "ocr_only", "oracle"]:
        if bl_name in baselines:
            names.append(baselines[bl_name]["display_name"])
            f1s.append(baselines[bl_name]["aggregate"]["f1_mean"])
            compressions.append(baselines[bl_name]["aggregate"]["compression_mean"] * 100)

    if not names:
        print("  Skipped (no baseline results)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(names))
    width = 0.6

    ax1.bar(x, f1s, width, color="#2196F3", edgecolor="white")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("Goal Detection F1")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, compressions, width, color="#4CAF50", edgecolor="white")
    ax2.set_ylabel("Compression (%)")
    ax2.set_title("Video Compression")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Baseline Comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", default="results/per_match")
    parser.add_argument("--ablation-dir", default="results/ablation")
    parser.add_argument("--baseline-file", default="results/baselines/baseline_results.json")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--figures", default="all",
                        help="Comma-separated figure numbers or 'all'")
    parser.add_argument("--timeline-match", default="Match1",
                        help="Match ID for timeline figure")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path(args.results_dir)
    ablation_dir = Path(args.ablation_dir)
    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"

    if args.figures == "all":
        figures = list(range(1, 11))
    else:
        figures = [int(f.strip()) for f in args.figures.split(",")]

    print("Generating figures...")

    if 2 in figures:
        print("\nFigure 2: Dataset Overview")
        fig_dataset_overview(matches_yaml, results_dir,
                            output_dir / "fig2_dataset_overview.pdf")

    if 4 in figures:
        print(f"\nFigure 4: Match Timeline ({args.timeline_match})")
        match_dir = results_dir / args.timeline_match
        if match_dir.exists():
            fig_match_timeline(results_dir, args.timeline_match,
                             output_dir / "fig4_match_timeline.pdf")
        else:
            print(f"  Skipped (no results for {args.timeline_match})")

    if 5 in figures:
        print("\nFigure 5: Play Threshold Ablation")
        abl_file = ablation_dir / "play_threshold_results.json"
        if abl_file.exists():
            fig_ablation_plot(abl_file, "play_threshold",
                            output_dir / "fig5_play_threshold_ablation.pdf")
        else:
            print("  Skipped (no ablation results)")

    if 6 in figures:
        print("\nFigure 6: Smoothing Window Ablation")
        abl_file = ablation_dir / "smoothing_window_results.json"
        if abl_file.exists():
            fig_ablation_plot(abl_file, "smoothing_window",
                            output_dir / "fig6_smoothing_window_ablation.pdf")
        else:
            print("  Skipped (no ablation results)")

    if 7 in figures:
        print("\nFigure 7: Speed-Accuracy Tradeoff")
        abl_file = ablation_dir / "frame_step_results.json"
        if abl_file.exists():
            fig_speed_accuracy(abl_file, output_dir / "fig7_speed_accuracy.pdf")
        else:
            print("  Skipped (no frame_step ablation results)")

    if 8 in figures:
        print("\nFigure 8: Tolerance Window Sensitivity")
        abl_file = ablation_dir / "tolerance_window_results.json"
        if abl_file.exists():
            fig_tolerance_sensitivity(abl_file,
                                    output_dir / "fig8_tolerance_sensitivity.pdf")
        else:
            print("  Skipped (no tolerance_window ablation results)")

    if 10 in figures:
        print(f"\nFigure 10: Player Count Distribution ({args.timeline_match})")
        fig_player_count_distribution(results_dir, args.timeline_match,
                                     output_dir / "fig10_player_distribution.pdf")

    # Baseline comparison (bonus)
    baseline_file = Path(args.baseline_file)
    if baseline_file.exists():
        print("\nBaseline Comparison Figure")
        fig_baselines_comparison(baseline_file,
                               output_dir / "fig_baselines_comparison.pdf")

    print("\nDone generating figures.")


if __name__ == "__main__":
    main()
