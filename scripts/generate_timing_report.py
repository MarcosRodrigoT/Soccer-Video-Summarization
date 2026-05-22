#!/usr/bin/env python3
"""Aggregate per-stage timings from a pipeline run and produce a report.

Reads per-match timing JSONs that run_single_match.py / run_all_matches.py
write under `<results_dir>/timings/<run_id>/<MatchN>.json` when invoked with
`--run-id`.  Produces:

    <results_dir>/timings/<run_id>/summary.json   (aggregate stats)
    <results_dir>/timings/<run_id>/breakdown.png  (per-stage bar chart)
    <results_dir>/timings/<run_id>/per_match.png  (per-match stacked bar)
    <results_dir>/timings/<run_id>/README.md      (markdown summary)

Cached stages (loaded from disk, not actually recomputed) are flagged but
their durations are excluded from compute-time aggregates.

Usage:
    python scripts/generate_timing_report.py --timing-dir results_hr/timings/<run_id>
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from soccer_summarization.utils import format_time_detailed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# Stage ordering and display labels
STAGE_ORDER = [
    "shot_segmentation",
    "object_detection",
    "play_classifier",
    "ocr_scoreboard",
    "highlight_assembly",
    "goal_shot_selection",
    "video_export",
    "evaluation",
]
STAGE_LABEL = {
    "shot_segmentation":   "Shot segmentation",
    "object_detection":    "YOLO detection",
    "play_classifier":     "Play classification",
    "ocr_scoreboard":      "OCR scoreboard",
    "highlight_assembly":  "Highlight assembly",
    "goal_shot_selection": "Goal-shot selection",
    "video_export":        "Video export",
    "evaluation":          "Evaluation",
}
STAGE_COLOR = {
    "shot_segmentation":   "#1565C0",
    "object_detection":    "#6A1B9A",
    "play_classifier":     "#2E7D32",
    "ocr_scoreboard":      "#EF6C00",
    "highlight_assembly":  "#00838F",
    "goal_shot_selection": "#5D4037",
    "video_export":        "#37474F",
    "evaluation":          "#C62828",
}


def _match_sort_key(s: str) -> int:
    m = re.search(r"\d+", s)
    return int(m.group()) if m else 9999


def load_run(timing_dir: Path) -> list[dict]:
    """Load all per-match JSON records from a run directory."""
    records: list[dict] = []
    for p in sorted(timing_dir.glob("Match*.json"), key=lambda p: _match_sort_key(p.stem)):
        with open(p) as f:
            records.append(json.load(f))
    return records


def aggregate(records: list[dict]) -> dict:
    """Compute per-stage mean / median / total compute time across matches.

    Cached stages are excluded — their time reflects disk-load only, not the
    actual compute cost of the stage.
    """
    by_stage: dict[str, list[float]] = {name: [] for name in STAGE_ORDER}
    cached_by_stage: dict[str, int] = {name: 0 for name in STAGE_ORDER}

    for rec in records:
        for name, info in rec.get("stages", {}).items():
            if name not in by_stage:
                by_stage[name] = []
                cached_by_stage[name] = 0
            if info.get("cached"):
                cached_by_stage[name] += 1
            else:
                by_stage[name].append(float(info["duration_sec"]))

    summary = {
        "num_matches": len(records),
        "stages": {},
    }
    for name in STAGE_ORDER + [n for n in by_stage if n not in STAGE_ORDER]:
        vals = by_stage.get(name, [])
        summary["stages"][name] = {
            "n_compute":   len(vals),
            "n_cached":    cached_by_stage.get(name, 0),
            "total_sec":   float(sum(vals)),
            "mean_sec":    float(mean(vals)) if vals else 0.0,
            "median_sec":  float(median(vals)) if vals else 0.0,
            "min_sec":     float(min(vals)) if vals else 0.0,
            "max_sec":     float(max(vals)) if vals else 0.0,
        }

    # Compute total wall-clock per match (sum of all non-cached stages)
    per_match_totals = []
    per_match_durations = []
    for rec in records:
        t = sum(
            float(s["duration_sec"]) for s in rec.get("stages", {}).values()
            if not s.get("cached")
        )
        per_match_totals.append(t)
        per_match_durations.append(float(rec.get("video_duration_sec", 0.0)))

    if per_match_totals:
        summary["total_pipeline_sec"] = {
            "mean":   float(mean(per_match_totals)),
            "median": float(median(per_match_totals)),
            "min":    float(min(per_match_totals)),
            "max":    float(max(per_match_totals)),
            "sum":    float(sum(per_match_totals)),
        }
    if per_match_durations:
        summary["mean_video_duration_sec"] = float(mean(per_match_durations))
        if mean(per_match_durations) > 0 and per_match_totals:
            summary["realtime_factor_mean"] = float(
                mean(per_match_totals) / mean(per_match_durations)
            )

    return summary


def plot_breakdown(summary: dict, output_path: Path):
    """Horizontal bar chart: mean compute time per stage."""
    stage_names = [n for n in STAGE_ORDER if summary["stages"][n]["n_compute"] > 0]
    means     = [summary["stages"][n]["mean_sec"] for n in stage_names]
    medians   = [summary["stages"][n]["median_sec"] for n in stage_names]
    n_compute = [summary["stages"][n]["n_compute"] for n in stage_names]
    n_cached  = [summary["stages"][n]["n_cached"] for n in stage_names]

    labels = [STAGE_LABEL.get(n, n) for n in stage_names]
    colors = [STAGE_COLOR.get(n, "#666666") for n in stage_names]

    fig, ax = plt.subplots(figsize=(11, 0.6 * len(stage_names) + 2), dpi=140)
    y = np.arange(len(stage_names))

    bars = ax.barh(y, means, color=colors, alpha=0.85, edgecolor="#212121", linewidth=0.7)

    # Median tick markers
    for i, m in enumerate(medians):
        ax.plot([m, m], [i - 0.25, i + 0.25], color="black", linewidth=1.5)

    # Right-side annotations
    max_mean = max(means) if means else 1
    for i, (mu, n_c, n_k) in enumerate(zip(means, n_compute, n_cached)):
        ann = f"  {mu:.1f}s  (n={n_c}"
        if n_k:
            ann += f", {n_k} cached"
        ann += ")"
        ax.text(mu + max_mean * 0.01, i, ann, va="center", fontsize=9, color="#212121")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean compute time per match (seconds)", fontsize=11)
    ax.set_title(
        f"Per-stage compute cost — {summary['num_matches']} matches "
        f"(black tick = median)", fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_xlim(0, max_mean * 1.30 if means else 1)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_per_match(records: list[dict], output_path: Path):
    """Stacked bar chart: per-match total time broken down by stage."""
    records = sorted(records, key=lambda r: _match_sort_key(r["match_id"]))
    match_ids = [r["match_id"] for r in records]

    stage_names = [n for n in STAGE_ORDER]
    matrix = np.zeros((len(stage_names), len(match_ids)))
    for j, rec in enumerate(records):
        for i, name in enumerate(stage_names):
            info = rec.get("stages", {}).get(name, None)
            if info and not info.get("cached"):
                matrix[i, j] = float(info["duration_sec"])

    fig, ax = plt.subplots(figsize=(max(12, 0.4 * len(match_ids) + 4), 6), dpi=140)
    x = np.arange(len(match_ids))
    bottoms = np.zeros(len(match_ids))
    for i, name in enumerate(stage_names):
        if matrix[i].sum() == 0:
            continue
        ax.bar(
            x, matrix[i], bottom=bottoms,
            color=STAGE_COLOR.get(name, "#666666"),
            label=STAGE_LABEL.get(name, name),
            edgecolor="white", linewidth=0.4, alpha=0.9,
        )
        bottoms += matrix[i]

    ax.set_xticks(x)
    ax.set_xticklabels(match_ids, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Compute time (seconds)", fontsize=11)
    ax.set_title(
        f"Per-match compute cost breakdown ({len(match_ids)} matches)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def write_markdown(summary: dict, run_id: str, output_path: Path):
    """Compact markdown report."""
    lines = [
        f"# Timing report — run `{run_id}`",
        "",
        f"- Matches: **{summary['num_matches']}**",
    ]
    if "mean_video_duration_sec" in summary:
        lines.append(
            f"- Mean video duration: **{format_time_detailed(summary['mean_video_duration_sec'])}**"
        )
    if "total_pipeline_sec" in summary:
        t = summary["total_pipeline_sec"]
        lines.append(
            f"- Mean total pipeline time per match: **{format_time_detailed(t['mean'])}** "
            f"(median {format_time_detailed(t['median'])}, "
            f"min {format_time_detailed(t['min'])}, max {format_time_detailed(t['max'])})"
        )
        lines.append(f"- Sum over all matches: **{format_time_detailed(t['sum'])}**")
    if "realtime_factor_mean" in summary:
        rf = summary["realtime_factor_mean"]
        lines.append(
            f"- Mean real-time factor: **{rf:.3f}×** "
            f"(processing time / video duration; <1 means faster than real-time)"
        )
    lines += ["", "## Per-stage compute cost", ""]
    lines.append("| Stage | n compute | n cached | mean (s) | median (s) | min (s) | max (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in STAGE_ORDER + [n for n in summary["stages"] if n not in STAGE_ORDER]:
        s = summary["stages"].get(name)
        if not s or (s["n_compute"] == 0 and s["n_cached"] == 0):
            continue
        lines.append(
            f"| {STAGE_LABEL.get(name, name)} | {s['n_compute']} | {s['n_cached']} | "
            f"{s['mean_sec']:.2f} | {s['median_sec']:.2f} | "
            f"{s['min_sec']:.2f} | {s['max_sec']:.2f} |"
        )
    lines += [
        "",
        "*Cached* counts indicate stages that were skipped because their output "
        "was already on disk; they're excluded from the time aggregates.",
        "",
        "## Files in this run",
        "",
        "- `Match*.json` — per-match per-stage timings (raw).",
        "- `summary.json` — aggregate stats.",
        "- `breakdown.png` — per-stage mean compute time (bar chart).",
        "- `per_match.png` — per-match total time, stacked by stage.",
    ]
    output_path.write_text("\n".join(lines))
    logger.info("Saved %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Aggregate timings + generate plots")
    parser.add_argument(
        "--timing-dir", required=True,
        help="Directory holding per-match timing JSONs "
             "(e.g. results_hr/timings/20260523_120000)",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    timing_dir = Path(args.timing_dir).resolve()
    if not timing_dir.exists():
        parser.error(f"Timing dir not found: {timing_dir}")

    records = load_run(timing_dir)
    if not records:
        parser.error(f"No Match*.json files in {timing_dir}")

    logger.info("Loaded %d match records from %s", len(records), timing_dir)
    summary = aggregate(records)

    # Write outputs alongside the per-match JSONs
    (timing_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Saved %s", timing_dir / "summary.json")

    plot_breakdown(summary, timing_dir / "breakdown.png")
    plot_per_match(records, timing_dir / "per_match.png")
    write_markdown(summary, timing_dir.name, timing_dir / "README.md")

    if "total_pipeline_sec" in summary:
        t = summary["total_pipeline_sec"]
        print(f"\nMean total pipeline time per match: {format_time_detailed(t['mean'])}")
        print(f"Per-match range: {format_time_detailed(t['min'])} → {format_time_detailed(t['max'])}")
    if "realtime_factor_mean" in summary:
        print(f"Mean real-time factor: {summary['realtime_factor_mean']:.3f}× video duration")


if __name__ == "__main__":
    main()
