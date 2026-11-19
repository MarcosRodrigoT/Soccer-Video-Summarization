#!/usr/bin/env python3
"""Generate the runtime-analysis figure + LaTeX table from a substage JSON.

Reads `runtime_substages.json` produced by `run_runtime_analysis.py` and writes:
  * `<paper>/resources/runtime_breakdown.pdf` — horizontal bar chart of
    per-substage wall-clock, with sub-bars grouped under their parent stage.
  * `<paper>/resources/runtime_breakdown.svg` — vector version.
  * `<json-dir>/runtime_table.tex`            — LaTeX table snippet ready
    to paste into the paper.
  * `<json-dir>/runtime_summary.md`           — markdown report (debugging /
    grep-able alternative to the JSON).

Usage:
    python scripts/generate_runtime_figure.py \\
        --json results_hr/runtime_analysis/Match1/runtime_substages.json \\
        --paper-resources ../Soccer-Paper/Soccer-Summarization/resources
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from soccer_summarization.utils import format_time_detailed, setup_logging  # noqa: E402

logger = logging.getLogger(__name__)


# Display order + labels for the report.  Substages not in this map are
# rendered in the order they were recorded, which mirrors the pipeline.
STAGE_ORDER = [
    "shot_segmentation",
    "object_detection",
    "play_classifier",
    "ocr_scoreboard",
    "highlight_assembly",
    "evaluation",
]
STAGE_LABEL = {
    "shot_segmentation":  "Shot segmentation",
    "object_detection":   "Player detection (YOLOv8)",
    "play_classifier":    "Play classification",
    "ocr_scoreboard":     "Scoreboard branch",
    "highlight_assembly": "Highlight assembly",
    "evaluation":         "Evaluation",
}
STAGE_COLOR = {
    "shot_segmentation":  "#388E3C",
    "object_detection":   "#1976D2",
    "play_classifier":    "#7B1FA2",
    "ocr_scoreboard":     "#F57C00",
    "highlight_assembly": "#00838F",
    "evaluation":         "#37474F",
}

SUBSTAGE_LABEL = {
    # shot_segmentation
    "frame_difference":           "Per-frame histogram differences",
    "abrupt_detection":           "Abrupt-cut detection",
    "gradual_detection":          "Gradual-transition detection",
    "shot_assembly":              "Shot assembly",
    # object_detection
    "model_load":                 "YOLOv8 model load",
    "inference_loop":             "Frame extraction + inference loop",
    "frame_decode":               "Video decoding of sampled frames",
    "yolo_inference":             "YOLOv8 inference + post-processing",
    # play_classifier
    "smoothing_threshold":        "Rolling smoothing + thresholding",
    # ocr_scoreboard
    "variability_map":            "IQR variability map",
    "overlay_mask":               "Overlay mask construction",
    "region_search":              "Scoreboard region search",
    "ocr_sampling_recognition":   "OCR sampling + EasyOCR inference",
    "parsing":                    "Multi-strategy parsing",
    "robustness_cascade":         "Robustness cascade",
    # highlight_assembly
    "play_long_shots":            "Play long-shot selection",
    "goal_shot_selection":        "Goal-shot anchoring",
    # evaluation
    "load_ground_truth":          "Load ground-truth annotations",
    "match_and_score":            "Matching + scoring",
}


def latex_escape(s: str) -> str:
    """Minimal LaTeX escaping for substage labels (we keep them safe by design)."""
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def load_substages(data: dict) -> tuple[list[dict], float]:
    """Return (records, total) grouped by stage in canonical order."""
    records = list(data["substages"])
    seen_stages = []
    for r in records:
        if r["stage"] not in seen_stages:
            seen_stages.append(r["stage"])

    ordered = []
    # Stages in canonical order first
    for stage in STAGE_ORDER:
        ordered.extend([r for r in records if r["stage"] == stage])
    # Append any non-canonical stages (defensive)
    for stage in seen_stages:
        if stage not in STAGE_ORDER:
            ordered.extend([r for r in records if r["stage"] == stage])

    total = sum(r["duration_sec"] for r in ordered)
    return ordered, total


def format_secs(s: float) -> str:
    if s >= 60:
        return format_time_detailed(s)
    return f"{s:.2f} s"


def _ms_per_frame(seconds: float, total_frames: int) -> float:
    """Convert a wall-clock duration to per-frame milliseconds."""
    return (seconds / total_frames * 1000.0) if total_frames > 0 else 0.0


def _fmt_ms(ms: float) -> str:
    """Compact, tiered formatter for per-frame ms values."""
    if ms >= 10:
        return f"{ms:.1f} ms"
    if ms >= 1:
        return f"{ms:.2f} ms"
    if ms >= 0.01:
        return f"{ms:.3f} ms"
    return "< 0.01 ms"


def plot_breakdown(data: dict, records: list[dict], total: float,
                   output_paths: list[Path]) -> None:
    """Horizontal bar chart of per-frame compute time, grouped visually by stage."""
    total_frames = int(data.get("total_frames", 0))

    n = len(records)
    fig_h = max(4.5, 0.35 * n + 1.4)
    fig, ax = plt.subplots(figsize=(8.6, fig_h), dpi=140)

    y = np.arange(n)[::-1]  # top-down
    ms_vals = [_ms_per_frame(r["duration_sec"], total_frames) for r in records]
    colors = [STAGE_COLOR.get(r["stage"], "#666666") for r in records]
    labels = [
        SUBSTAGE_LABEL.get(r["substage"], r["substage"].replace("_", " "))
        for r in records
    ]

    ax.barh(
        y, ms_vals, color=colors, edgecolor="#212121", linewidth=0.6, alpha=0.9,
    )

    xmax = max(ms_vals) if ms_vals else 1.0
    for yi, (val, rec) in enumerate(zip(ms_vals, records)):
        pct = 100.0 * rec["duration_sec"] / total if total > 0 else 0.0
        ax.text(
            val + xmax * 0.012, y[yi], f"  {_fmt_ms(val)}   ({pct:4.1f}%)",
            va="center", fontsize=8.5, color="#212121",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)

    # Group separators: thin horizontal lines between adjacent stages
    prev_stage = None
    for i, r in enumerate(records):
        if r["stage"] != prev_stage and prev_stage is not None:
            ax.axhline(y[i] + 0.5, color="#BDBDBD", linewidth=0.6, linestyle="-")
        prev_stage = r["stage"]

    # Stage legend (manual)
    handles = []
    used_stages = []
    for r in records:
        if r["stage"] not in used_stages:
            used_stages.append(r["stage"])
    for stage in used_stages:
        handles.append(
            plt.Line2D([0], [0], marker="s", linestyle="",
                       markersize=10, markeredgecolor="#212121",
                       markeredgewidth=0.6,
                       markerfacecolor=STAGE_COLOR.get(stage, "#666666"),
                       label=STAGE_LABEL.get(stage, stage))
        )
    ax.legend(handles=handles, loc="lower right", fontsize=8,
              frameon=True, framealpha=0.95, ncol=2)

    ax.set_xlabel("Per-frame compute time (ms)", fontsize=10)
    duration = data.get("video_duration_sec", 0.0)
    match_disp = data['match_id'].replace("Match", "Match ")
    title = (
        f"Substage-level per-frame compute cost on {match_disp} "
        f"(broadcast {format_time_detailed(duration)}, "
        f"total compute {format_time_detailed(total)})"
    )
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, xmax * 1.40)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for p in output_paths:
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", p)
    plt.close(fig)


# ── Simplified four-bar figure ──────────────────────────────────────────────

HIGH_VOLUME_KEYS = [
    # (stage, substage, display label, color)
    ("shot_segmentation",  "frame_difference",
     "Shot dissimilarity signal (Sec. 4)",        "#388E3C"),
    ("object_detection",   "inference_loop",
     "YOLOv8 player detection (Sec. 5)",          "#1976D2"),
    ("object_detection",   "yolo_inference",
     "YOLOv8 player detection (Sec. 5)",          "#1976D2"),
    ("object_detection",   "frame_decode",
     "Video decoding for detection (Sec. 5)",     "#64B5F6"),
    ("ocr_scoreboard",     "ocr_sampling_recognition",
     "Scoreboard OCR tracking (Sec. 6)",          "#F57C00"),
]
OTHERS_LABEL = "All other components"
OTHERS_COLOR = "#37474F"


def plot_simplified_breakdown(data: dict, records: list[dict], total: float,
                              output_paths: list[Path]) -> None:
    """Four-bar variant: three dominant substages + everything else lumped.

    Labels link each bar to its section in the paper so the reader can
    map the runtime cost back to the methodological description.
    """
    total_frames = int(data.get("total_frames", 0))

    by_key = {(r["stage"], r["substage"]): r for r in records}
    bars: list[tuple[str, float, str, float]] = []  # (label, ms, color, sec)
    captured_sec = 0.0

    for stage, substage, label, color in HIGH_VOLUME_KEYS:
        rec = by_key.get((stage, substage))
        if rec is None:
            continue
        sec = float(rec["duration_sec"])
        captured_sec += sec
        bars.append((label, _ms_per_frame(sec, total_frames), color, sec))

    others_sec = total - captured_sec
    bars.append((
        OTHERS_LABEL,
        _ms_per_frame(others_sec, total_frames),
        OTHERS_COLOR,
        others_sec,
    ))

    fig, ax = plt.subplots(figsize=(8.6, 3.6), dpi=140)
    y = np.arange(len(bars))[::-1]
    ms_vals = [b[1] for b in bars]
    labels = [b[0] for b in bars]
    colors = [b[2] for b in bars]
    secs = [b[3] for b in bars]

    ax.barh(y, ms_vals, color=colors, edgecolor="#212121", linewidth=0.6, alpha=0.9)

    xmax = max(ms_vals) if ms_vals else 1.0
    for yi, (val, s) in enumerate(zip(ms_vals, secs)):
        pct = 100.0 * s / total if total > 0 else 0.0
        ax.text(
            val + xmax * 0.012, y[yi], f"  {_fmt_ms(val)}   ({pct:4.1f}%)",
            va="center", fontsize=9.5, color="#212121",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Per-frame compute time (ms)", fontsize=10)
    ax.set_xlim(0, xmax * 1.40)
    ax.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # No title inside the figure: the journal asks for captions only.
    fig.tight_layout()
    for p in output_paths:
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        logger.info("Saved %s", p)
    plt.close(fig)


def _fmt_thousands_math(value: float, decimals: int = 1) -> str:
    """Format a number with LaTeX-safe thousands separators inside math mode."""
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "{,}")


def write_latex_table(data: dict, records: list[dict], total: float,
                      out_path: Path) -> None:
    """Booktabs table grouped by stage, with stage subtotals + grand total."""
    # Substages below this floor are reported as "$< 0.1$~s" / "$\sim 0$".
    TIME_FLOOR = 0.1   # seconds

    rf = float(data.get("realtime_factor", 0.0))
    overall_rt = 1.0 / rf if rf > 0 else float("nan")

    lines = [
        r"\begin{table}[!tbp]",
        r"\centering",
        r"\caption{Substage-level wall-clock runtime on Match~1 "
        r"(Valencia~CF~vs.~FC~Barcelona, 2013/14; $1\,\mathrm{h}\,44\,\mathrm{min}$ at $1920\times 1080$, $25$~fps). "
        r"Wall-clock time is reported per substage with its share of the "
        r"total pipeline cost; stage subtotals are bolded. Substages "
        r"whose wall-clock time falls below the $0.1$~s noise floor are "
        r"reported as such. All numbers come from a single cold-cache "
        r"run on the hardware described in the text; the overall "
        rf"pipeline runs at a real-time factor of ${rf:.2f}$ "
        rf"(${overall_rt:.2f}\times$ faster than real-time)." + "}",
        r"\label{tab:runtime}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"\textbf{Stage} & \textbf{Substage} & "
        r"\textbf{Time (s)} & \textbf{Share (\%)} \\",
        r"\midrule",
    ]

    stages_seen = []
    for r in records:
        if r["stage"] not in stages_seen:
            stages_seen.append(r["stage"])

    for stage in stages_seen:
        rows = [r for r in records if r["stage"] == stage]
        stage_total = sum(r["duration_sec"] for r in rows)
        stage_label_disp = latex_escape(STAGE_LABEL.get(stage, stage))
        for i, r in enumerate(rows):
            t = r["duration_sec"]
            stage_cell = stage_label_disp if i == 0 else ""
            sub_label = latex_escape(
                SUBSTAGE_LABEL.get(r["substage"], r["substage"].replace("_", " "))
            )
            if t < TIME_FLOOR:
                time_cell = r"$< 0.1$"
                share_cell = r"$\sim 0$"
            else:
                pct = 100.0 * t / total if total > 0 else 0.0
                time_cell = f"${_fmt_thousands_math(t, 1)}$"
                share_cell = f"${pct:.1f}$"
            lines.append(
                f"{stage_cell} & {sub_label} & {time_cell} & {share_cell} \\\\"
            )
        # Subtotal
        sub_pct = 100.0 * stage_total / total if total > 0 else 0.0
        lines.append(r"\cmidrule(lr){2-4}")
        if stage_total < TIME_FLOOR:
            sub_time = r"$\mathbf{< 0.1}$"
            sub_share = r"$\mathbf{\sim 0}$"
        else:
            sub_time = f"$\\mathbf{{{_fmt_thousands_math(stage_total, 1)}}}$"
            sub_share = f"$\\mathbf{{{sub_pct:.1f}}}$"
        lines.append(
            f" & \\textit{{Subtotal}} & {sub_time} & {sub_share} \\\\"
        )
        lines.append(r"\midrule")

    # Grand total
    lines.append(
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Total pipeline}}}} & "
        f"$\\mathbf{{{_fmt_thousands_math(total, 1)}}}$ & "
        f"$\\mathbf{{100.0}}$ \\\\"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    text = "\n".join(lines) + "\n"
    out_path.write_text(text)
    logger.info("Saved %s", out_path)


def write_markdown_summary(data: dict, records: list[dict], total: float,
                           out_path: Path) -> None:
    duration = float(data.get("video_duration_sec", 0.0))
    rf = float(data.get("realtime_factor", 0.0))
    hw = data.get("hardware", {}) or {}
    gpus = hw.get("gpus") or []
    gpu_str = ", ".join(
        f"{g['name']} ({g['memory_mib']} MiB)" for g in gpus
    ) or "n/a"
    lines = [
        f"# Runtime analysis — {data['match_id']}",
        "",
        f"- Broadcast duration: **{format_time_detailed(duration)}** "
        f"({data.get('total_frames'):,} frames at {data.get('fps')} fps, "
        f"{data.get('width')}×{data.get('height')})",
        f"- Total pipeline compute: **{format_time_detailed(total)}**",
        f"- Real-time factor: **{rf:.3f}×** "
        f"(compute / video duration; <1 means faster than real-time)",
        f"- Hardware: **CPU** {hw.get('cpu_name', 'n/a')} "
        f"({hw.get('cpu_count')} threads), **GPU** {gpu_str}, "
        f"torch {hw.get('torch_version')}/CUDA {hw.get('cuda_version')}",
        f"- Generated: {data.get('timestamp', 'n/a')}",
        "",
        "## Per-substage compute time",
        "",
        "| Stage | Substage | Time (s) | Share (%) | Throughput |",
        "|---|---|---:|---:|---:|",
    ]
    for r in records:
        t = r["duration_sec"]
        pct = 100.0 * t / total if total > 0 else 0.0
        rt = duration / t if t > 0 else float("nan")
        lines.append(
            f"| {STAGE_LABEL.get(r['stage'], r['stage'])} "
            f"| {SUBSTAGE_LABEL.get(r['substage'], r['substage'])} "
            f"| {t:.2f} | {pct:.1f} | {rt:.1f}× |"
        )
    out_path.write_text("\n".join(lines) + "\n")
    logger.info("Saved %s", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default="results_hr/runtime_analysis/Match1/runtime_substages.json",
    )
    ap.add_argument(
        "--paper-resources",
        default="../Soccer-Paper/Soccer-Summarization/resources",
        help="Path to the paper's resources/ directory (PDF figure target).",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    setup_logging(args.log_level)

    json_path = Path(args.json).resolve()
    if not json_path.exists():
        ap.error(f"JSON not found: {json_path}")
    data = json.loads(json_path.read_text())

    records, total = load_substages(data)
    if not records:
        ap.error("No substage records in JSON")

    out_dir = json_path.parent

    # ── Figures ──────────────────────────────────────────────────────────
    paper_res = Path(args.paper_resources).resolve()
    paper_res.mkdir(parents=True, exist_ok=True)

    detailed_targets = [
        paper_res / "runtime_breakdown.pdf",
        paper_res / "runtime_breakdown.svg",
        out_dir / "runtime_breakdown.png",
    ]
    plot_breakdown(data, records, total, detailed_targets)

    simplified_targets = [
        paper_res / "runtime_breakdown_simplified.pdf",
        paper_res / "runtime_breakdown_simplified.svg",
        out_dir / "runtime_breakdown_simplified.png",
    ]
    plot_simplified_breakdown(data, records, total, simplified_targets)

    # ── LaTeX table + markdown summary ───────────────────────────────────
    write_latex_table(data, records, total, out_dir / "runtime_table.tex")
    write_markdown_summary(data, records, total, out_dir / "runtime_summary.md")

    print(f"\nTotal pipeline compute: {format_time_detailed(total)}")
    print(f"Real-time factor: {data['realtime_factor']:.3f}x")
    print(f"Wrote detailed figure   → {detailed_targets[0]}")
    print(f"Wrote simplified figure → {simplified_targets[0]}")
    print(f"Wrote table             → {out_dir / 'runtime_table.tex'}")


if __name__ == "__main__":
    main()
