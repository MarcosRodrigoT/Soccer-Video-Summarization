"""Figure: score robustness cascade for the paper.

Two-panel timeline figure illustrating how the OCR-based score-tracking
stage converts a noisy raw digit stream into a clean goal-event timeline.
The upper panel shows the raw per-digit OCR readings for the home and
away score positions; the lower panel shows the home/away score
timelines recovered by the robustness cascade, with detected goal events
overlaid against the ground-truth scoreboard updates.

Uses the cached OCR readings pickle produced by the existing diagnostics
pipeline (``results_hr/per_match/<match>/ocr_readings_figure.pkl``), so
this script does not re-run OCR.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import yaml
from matplotlib.ticker import FuncFormatter

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from soccer_summarization.config import PipelineConfig  # noqa: E402

from _style import save_figure, set_paper_style  # noqa: E402
from visualize_ocr_diagnostics import (  # noqa: E402
    extract_digit_clusters, extract_xy_pattern_scores, load_match_info,
)

import os
MATCH_ID = os.environ.get("CASCADE_MATCH", "Match1")
RESULTS_DIR = PROJECT / "results_hr" / "per_match" / MATCH_ID
TOLERANCE_S = 30.0


def _mmss(x_minutes: float, _pos) -> str:
    total = int(round(x_minutes * 60))
    return f"{total // 60:02d}:{total % 60:02d}"


def _load_gt_times(match_info: dict, fps: float) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) for each GT goal event."""
    config = PipelineConfig.from_yaml(str(PROJECT / "config" / "default.yaml"))
    from soccer_summarization.evaluation import Evaluator
    evaluator = Evaluator(config.evaluation)
    gt_df = evaluator.load_ground_truth(match_info["gt_path"])
    gt_events = evaluator.extract_goal_events(gt_df)
    return [(e.start_time, e.end_time) for e in gt_events]


def build_figure() -> None:
    set_paper_style()

    # Load cached OCR readings
    readings_path = RESULTS_DIR / "ocr_readings_figure.pkl"
    with readings_path.open("rb") as f:
        readings = pickle.load(f)

    # Resolve fps + duration
    info = load_match_info(
        MATCH_ID, PipelineConfig.from_yaml(str(PROJECT / "config" / "default.yaml")),
        matches_yaml=str(PROJECT / "config" / "matches.yaml"),
    )
    # Read duration from frame_classification.csv (last timestamp)
    fc = pd.read_csv(RESULTS_DIR / "frame_classification.csv")
    total_dur_s = float(fc["time_seconds"].max())
    fps = 25.0
    total_min = total_dur_s / 60.0

    # Extract clusters; fall back to X-Y pattern parser
    cluster_data = extract_digit_clusters(readings, fps)
    if cluster_data is None:
        cluster_data = extract_xy_pattern_scores(readings, fps)
    if cluster_data is None:
        raise RuntimeError(f"No score readings recovered for {MATCH_ID}")

    home_raw = cluster_data["home_raw"]
    away_raw = cluster_data["away_raw"]
    home_filt = cluster_data["home_filtered"]
    away_filt = cluster_data["away_filtered"]

    # GT events + detected events
    gt_times = _load_gt_times(info, fps)
    se_df = pd.read_csv(RESULTS_DIR / "score_events.csv")
    detected_times = se_df["timestamp"].tolist() if len(se_df) else []

    # ---- Figure ----------------------------------------------------------
    fig, (ax_raw, ax_filt) = plt.subplots(
        nrows=2, sharex=True, figsize=(9.5, 5.2),
        gridspec_kw=dict(hspace=0.42, height_ratios=[1.0, 1.0]),
    )
    TITLE_PAD = 14.0

    tol_min = TOLERANCE_S / 60.0

    # ===== Top panel: raw OCR readings =====
    hr_t = [f / fps / 60.0 for f in sorted(home_raw.keys())]
    hr_v = [home_raw[f] for f in sorted(home_raw.keys())]
    ar_t = [f / fps / 60.0 for f in sorted(away_raw.keys())]
    ar_v = [away_raw[f] for f in sorted(away_raw.keys())]

    ax_raw.scatter(hr_t, hr_v, s=4, alpha=0.45, c="#1565C0",
                   label="Home raw")
    ax_raw.scatter(ar_t, ar_v, s=4, alpha=0.45, c="#C62828",
                   label="Away raw")

    # GT overlay (tolerance band + dashed line + triangle marker)
    top_trans_raw = mtransforms.blended_transform_factory(
        ax_raw.transData, ax_raw.transAxes)
    for i, (start, _end) in enumerate(gt_times):
        t_min = start / 60.0
        ax_raw.axvspan(t_min - tol_min, t_min + tol_min,
                       color="#4caf50", alpha=0.12, zorder=1)
        ax_raw.axvline(t_min, color="#1b5e20", ls="--", lw=1.2,
                       alpha=0.9, zorder=2,
                       label="GT scoreboard update" if i == 0 else None)
        ax_raw.plot(t_min, 1.04, marker="v", color="#1b5e20", markersize=7,
                    transform=top_trans_raw, clip_on=False, zorder=6)

    y_max_raw = max(max(hr_v + [0]), max(ar_v + [0])) + 1
    ax_raw.set_ylim(-0.5, y_max_raw)
    ax_raw.set_xlim(0, total_min)
    ax_raw.set_ylabel("Digit value")
    ax_raw.set_title("Raw OCR readings (per-digit, full broadcast)",
                     fontsize=10, pad=TITLE_PAD)
    ax_raw.legend(loc="upper left", fontsize=8)
    ax_raw.grid(True, alpha=0.3)

    # ===== Bottom panel: filtered score timelines =====
    hf_t = [f / fps / 60.0 for f in sorted(home_filt.keys())]
    hf_v = [home_filt[f] for f in sorted(home_filt.keys())]
    af_t = [f / fps / 60.0 for f in sorted(away_filt.keys())]
    af_v = [away_filt[f] for f in sorted(away_filt.keys())]

    ax_filt.plot(hf_t, hf_v, c="#1565C0", lw=1.8, alpha=0.85,
                 label="Home (filtered)")
    ax_filt.plot(af_t, af_v, c="#C62828", lw=1.8, alpha=0.85,
                 label="Away (filtered)")

    top_trans_f = mtransforms.blended_transform_factory(
        ax_filt.transData, ax_filt.transAxes)
    for i, (start, _end) in enumerate(gt_times):
        t_min = start / 60.0
        ax_filt.axvspan(t_min - tol_min, t_min + tol_min,
                        color="#4caf50", alpha=0.12, zorder=1,
                        label="GT $\\pm$ tolerance" if i == 0 else None)
        ax_filt.axvline(t_min, color="#1b5e20", ls="--", lw=1.4,
                        alpha=0.9, zorder=2,
                        label="GT scoreboard update" if i == 0 else None)
        ax_filt.plot(t_min, 1.04, marker="v", color="#1b5e20", markersize=7,
                     transform=top_trans_f, clip_on=False, zorder=6)
    for i, t in enumerate(detected_times):
        ax_filt.axvline(t / 60.0, color="#7b1fa2", lw=1.6, alpha=0.85,
                        zorder=3,
                        label="Detected goal event" if i == 0 else None)

    y_max_f = max(max(hf_v + [0]), max(af_v + [0])) + 1
    ax_filt.set_ylim(-0.5, y_max_f)
    ax_filt.set_xlim(0, total_min)
    ax_filt.set_ylabel("Score")
    ax_filt.set_xlabel("Time (mm:ss)")
    ax_filt.set_title("Filtered score timelines after the robustness cascade",
                      fontsize=10, pad=TITLE_PAD)
    ax_filt.legend(loc="upper left", fontsize=8, ncols=2)
    ax_filt.grid(True, alpha=0.3)
    ax_filt.xaxis.set_major_formatter(FuncFormatter(_mmss))

    out_name = "score_cascade" if MATCH_ID == "Match1" else f"score_cascade_{MATCH_ID}"
    save_figure(fig, out_name, also_png=True)


if __name__ == "__main__":
    build_figure()
