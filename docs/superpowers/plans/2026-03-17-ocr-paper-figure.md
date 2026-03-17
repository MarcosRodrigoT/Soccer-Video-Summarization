# OCR Paper Figure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a script that generates per-match paper-quality figures showing OCR scoreboard detection with before/at/after frames around each goal event, connected to a score timeline via dashed lines.

**Architecture:** Single script (`scripts/generate_ocr_paper_figure.py`) that reuses existing OCR diagnostics functions for data collection and adds new figure layout logic with nested GridSpec. Events are classified as TP/FN/FP by replicating the evaluation module's matching logic to get per-event indices.

**Tech Stack:** matplotlib (GridSpec, ConnectionPatch), OpenCV (frame extraction, crop annotation), EasyOCR (via ScoreboardDetector), pandas, numpy

**Spec:** `docs/superpowers/specs/2026-03-17-ocr-paper-figure-design.md`

---

## Chunk 1: Core Script Structure

### Task 1: Create script skeleton with CLI and imports

**Files:**
- Create: `scripts/generate_ocr_paper_figure.py`

- [ ] **Step 1: Write script skeleton**

```python
#!/usr/bin/env python3
"""Generate paper-quality OCR scoreboard figures for each match.

For each match, generates a horizontal figure with:
  - Top row: IQR map, overlay mask, sample frame with scoreboard bbox
  - Middle rows: Before/at/after frames around each event (TP/FN/FP)
  - Bottom: Score timeline with dashed connectors from frames

Usage:
    python scripts/generate_ocr_paper_figure.py
    python scripts/generate_ocr_paper_figure.py --matches Match1,Match5
    python scripts/generate_ocr_paper_figure.py --offset 30
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from soccer_summarization.config import PipelineConfig
from soccer_summarization.evaluation import Evaluator
from soccer_summarization.ocr_scoreboard import ScoreboardDetector, ScoreboardRegion
from soccer_summarization.utils import get_video_properties

# Reuse from diagnostics script
from scripts.visualize_ocr_diagnostics import (
    load_match_info,
    collect_ocr_readings,
    extract_digit_clusters,
    extract_xy_pattern_scores,
    _mmss_formatter,
)

logger = logging.getLogger(__name__)

# Journal-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

# Event type colors
COLOR_TP = "#2e7d32"   # green
COLOR_FN = "#e65100"   # orange
COLOR_FP = "#c62828"   # red

MAX_EVENTS = 12
EVENTS_PER_ROW = 4
```

- [ ] **Step 2: Verify imports work**

Run: `source venv/bin/activate && python -c "import scripts.generate_ocr_paper_figure"`

If the `scripts/` import path doesn't work (no `__init__.py`), the script uses `sys.path.insert` so run:
```bash
cd /home/mrt/Projects/Soccer-Video-Summarization && python scripts/generate_ocr_paper_figure.py --help
```
Expected: no import errors (will fail on missing `main()` — that's fine at this stage).

Note: the import `from scripts.visualize_ocr_diagnostics import ...` may need adjustment. If it fails, change to:
```python
sys.path.insert(0, str(Path(__file__).parent))
from visualize_ocr_diagnostics import (
    load_match_info,
    collect_ocr_readings,
    extract_digit_clusters,
    extract_xy_pattern_scores,
    _mmss_formatter,
)
```

- [ ] **Step 3: Commit skeleton**

```bash
git add scripts/generate_ocr_paper_figure.py
git commit -m "Add skeleton for OCR paper figure generation script"
```

---

### Task 2: Event classification (TP/FN/FP)

**Files:**
- Modify: `scripts/generate_ocr_paper_figure.py`

- [ ] **Step 1: Add classify_events function**

Append after the constants:

```python
def classify_events(detected_events, gt_events, tolerance=30.0):
    """Classify events as TP, FN, or FP using same logic as Evaluator.

    Returns (events_list, original_count) where events_list contains dicts:
        {"type": "tp"|"fn"|"fp", "time": float, "label": str,
         "gt_event": GoalEvent|None, "detected": namedtuple|None}
    """
    predicted_times = [ev.timestamp for ev in detected_events]

    # Match using same algorithm as evaluation.py (no one-to-one guard on GT,
    # matching the exact behavior of Evaluator.evaluate_goal_detection)
    matched_gt = set()
    matched_pred = set()
    pred_to_gt = {}
    for i, pred_time in enumerate(predicted_times):
        for j, gt_event in enumerate(gt_events):
            if gt_event.start_time - tolerance <= pred_time <= gt_event.end_time + tolerance:
                matched_gt.add(j)
                matched_pred.add(i)
                pred_to_gt[i] = j
                break

    events = []

    # TPs: detected events that matched a GT event
    for i, ev in enumerate(detected_events):
        if i in matched_pred:
            j = pred_to_gt[i]
            events.append({
                "type": "tp",
                "time": ev.timestamp,
                "label": f"Goal {ev.home_score}-{ev.away_score} (TP)",
                "gt_event": gt_events[j],
                "detected": ev,
            })

    # FNs: GT events with no matching detection
    for j, gt_ev in enumerate(gt_events):
        if j not in matched_gt:
            t = gt_ev.midpoint
            ts = f"{int(t // 60)}:{int(t % 60):02d}"
            events.append({
                "type": "fn",
                "time": t,
                "label": f"GT Goal @{ts} (FN)",
                "gt_event": gt_ev,
                "detected": None,
            })

    # FPs: detections with no matching GT
    for i, ev in enumerate(detected_events):
        if i not in matched_pred:
            events.append({
                "type": "fp",
                "time": ev.timestamp,
                "label": f"FP {ev.home_score}-{ev.away_score}",
                "gt_event": None,
                "detected": ev,
            })

    events.sort(key=lambda e: e["time"])

    original_count = len(events)
    if len(events) > MAX_EVENTS:
        events = events[:MAX_EVENTS]

    return events, original_count
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "from scripts.generate_ocr_paper_figure import classify_events; print('OK')"` or equivalent via:
```bash
python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from generate_ocr_paper_figure import classify_events
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_ocr_paper_figure.py
git commit -m "Add event classification (TP/FN/FP) for paper figure"
```

---

### Task 3: Frame extraction around events

**Files:**
- Modify: `scripts/generate_ocr_paper_figure.py`

- [ ] **Step 1: Add extract_event_frames function**

```python
def extract_event_frames(video_path, region, events, fps, total_frames, offset=30):
    """Extract before/at/after frames for each event with scoreboard crops.

    Args:
        video_path: Path to video file
        region: ScoreboardRegion
        events: List of event dicts from classify_events()
        fps: Video FPS
        total_frames: Total video frames
        offset: Seconds before/after event center

    Returns:
        List of event dicts augmented with "frames" key containing 3 frame dicts.
    """
    detector = ScoreboardDetector()
    ocr = detector._get_ocr()
    cap = cv2.VideoCapture(video_path)
    pad = 15
    scale = 4

    sb_x_min = pad * scale
    sb_x_max = (pad + region.w) * scale
    sb_y_min = pad * scale
    sb_y_max = (pad + region.h) * scale

    for event in events:
        center_sec = event["time"]
        frame_times = {
            "Before": center_sec - offset,
            "At event": center_sec,
            "After": center_sec + offset,
        }

        event["frames"] = []
        for label, t_sec in frame_times.items():
            frame_idx = int(t_sec * fps)
            clamped = max(0, min(frame_idx, total_frames - 1))
            is_clamped = clamped != frame_idx
            frame_idx = clamped

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Full frame with bbox
            annotated = frame.copy()
            evt_color_bgr = _event_color_bgr(event["type"])
            x, y, w, h = int(region.x), int(region.y), int(region.w), int(region.h)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), evt_color_bgr, 2)

            # Scoreboard crop with OCR annotations
            crop = detector._crop_region(frame, region, pad=pad)
            upscaled = cv2.resize(crop, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            results = ocr.readtext(upscaled)

            crop_annotated = upscaled.copy()
            for bbox, text, conf in results:
                cx = (bbox[0][0] + bbox[2][0]) / 2
                cy = (bbox[0][1] + bbox[2][1]) / 2
                inside = sb_x_min <= cx <= sb_x_max and sb_y_min <= cy <= sb_y_max
                pts = np.array(bbox, dtype=np.int32)
                color = (0, 255, 0) if inside else (128, 128, 128)
                cv2.polylines(crop_annotated, [pts], True, color, 2)
                cv2.putText(crop_annotated, f"{text} ({conf:.2f})",
                            (int(pts[0][0]), int(pts[0][1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            ts_str = f"{int(t_sec // 60)}:{int(t_sec % 60):02d}"
            title = f"{label} ({ts_str})"
            if is_clamped:
                title += " (clamped)"

            event["frames"].append({
                "label": title,
                "full_frame": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                "crop": cv2.cvtColor(crop_annotated, cv2.COLOR_BGR2RGB),
                "time_sec": max(0.0, t_sec),
            })

    cap.release()
    return events


def _event_color_bgr(event_type):
    """Return BGR color tuple for OpenCV drawing."""
    if event_type == "tp":
        return (50, 180, 50)
    elif event_type == "fn":
        return (0, 130, 230)
    else:
        return (50, 50, 200)
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from generate_ocr_paper_figure import extract_event_frames
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_ocr_paper_figure.py
git commit -m "Add event frame extraction with scoreboard crops"
```

---

## Chunk 2: Figure Generation and Main Loop

### Task 4: Build the figure layout

**Files:**
- Modify: `scripts/generate_ocr_paper_figure.py`

- [ ] **Step 1: Add generate_paper_figure function**

This is the main figure generation function. It builds the nested GridSpec layout.

```python
def generate_paper_figure(match_id, match_info, config, output_dir, offset=30):
    """Generate the paper-quality OCR figure for one match."""
    video_path = match_info["video_path"]
    gt_path = match_info["gt_path"]
    score = match_info["score"]
    home_team = match_info.get("home_team", "Home")
    away_team = match_info.get("away_team", "Away")
    results_dir = Path(f"results/per_match/{match_id}")

    # Load scoreboard region
    region_path = results_dir / "scoreboard_region.json"
    if not region_path.exists():
        print(f"  {match_id}: No scoreboard_region.json, skipping")
        return
    with open(region_path) as f:
        rj = json.load(f)
    region = ScoreboardRegion(x=rj["x"], y=rj["y"], w=rj["w"], h=rj["h"])

    # Load GT events
    evaluator = Evaluator(config.evaluation)
    gt_df = evaluator.load_ground_truth(gt_path)
    gt_events = evaluator.extract_goal_events(gt_df)
    gt_times = [(e.start_time, e.end_time) for e in gt_events]

    # Load detected events
    score_events_path = results_dir / "score_events.csv"
    detected_events = []
    if score_events_path.exists():
        try:
            se_df = pd.read_csv(score_events_path)
            detected_events = list(se_df.itertuples(index=False))
        except (pd.errors.EmptyDataError, KeyError):
            pass

    # Video properties
    props = get_video_properties(video_path)
    fps = props["fps"]
    total_frames = props["total_frames"]
    total_duration = props["duration"]

    # Classify events
    events, original_event_count = classify_events(
        detected_events, gt_events,
        tolerance=config.evaluation.tolerance_window)

    n_events = len(events)
    has_events = n_events > 0

    # Extract frames around events
    if has_events:
        print(f"  Extracting frames for {n_events} events...")
        events = extract_event_frames(video_path, region, events, fps,
                                      total_frames, offset=offset)

    # Compute IQR map and overlay mask
    print(f"  Computing IQR map...")
    detector = ScoreboardDetector(config.ocr_scoreboard)
    std_map = detector.find_static_overlays(video_path)
    overlay_mask, raw_mask = detector.build_overlay_mask(std_map)

    # Get sample frame for top row
    print(f"  Extracting sample frame...")
    mid_frame_idx = total_frames // 2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, sample_frame = cap.read()
    cap.release()
    if ret:
        sample_annotated = sample_frame.copy()
        x, y, w, h = int(region.x), int(region.y), int(region.w), int(region.h)
        cv2.rectangle(sample_annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        sample_frame_rgb = cv2.cvtColor(sample_annotated, cv2.COLOR_BGR2RGB)
    else:
        sample_frame_rgb = np.zeros((480, 854, 3), dtype=np.uint8)

    # Collect OCR readings for timeline
    print(f"  Collecting OCR readings...")
    readings, _ = collect_ocr_readings(video_path, region, config)
    cluster_data = extract_digit_clusters(readings, fps)
    if cluster_data is None:
        cluster_data = extract_xy_pattern_scores(readings, fps)

    # --- Compute layout dimensions ---
    n_event_rows = (n_events + EVENTS_PER_ROW - 1) // EVENTS_PER_ROW if has_events else 0
    events_in_widest_row = min(n_events, EVENTS_PER_ROW) if has_events else 0
    fig_width = max(14, events_in_widest_row * 4.5)
    fig_height = 2.5 + 4.5 * n_event_rows + 3.0

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Title
    title = f"{match_id}: {home_team} {score} {away_team}"
    if original_event_count > MAX_EVENTS:
        title += f" (showing {MAX_EVENTS}/{original_event_count} events)"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)

    # --- Outer grid: top_row | event_rows... | timeline ---
    n_outer_rows = 1 + n_event_rows + 1
    height_ratios = [2.5] + [4.5] * n_event_rows + [3.0]
    outer_gs = fig.add_gridspec(n_outer_rows, 1, hspace=0.3,
                                height_ratios=height_ratios,
                                top=0.96, bottom=0.04)

    # === TOP ROW: IQR map | Overlay mask | Sample frame ===
    top_gs = outer_gs[0].subgridspec(1, 3, wspace=0.08)

    ax_iqr = fig.add_subplot(top_gs[0])
    ax_iqr.imshow(std_map, cmap="hot", vmin=0, vmax=np.percentile(std_map, 95))
    rect = Rectangle((region.x, region.y), region.w, region.h,
                      linewidth=2, edgecolor="cyan", facecolor="none")
    ax_iqr.add_patch(rect)
    ax_iqr.set_title("IQR variability map", fontsize=10)
    ax_iqr.axis("off")

    ax_mask = fig.add_subplot(top_gs[1])
    ax_mask.imshow(overlay_mask, cmap="gray")
    rect_mask = Rectangle((region.x, region.y), region.w, region.h,
                           linewidth=2, edgecolor="cyan", facecolor="none")
    ax_mask.add_patch(rect_mask)
    ax_mask.set_title("Overlay mask", fontsize=10)
    ax_mask.axis("off")

    ax_sample = fig.add_subplot(top_gs[2])
    ax_sample.imshow(sample_frame_rgb)
    ax_sample.set_title("Sample frame with scoreboard", fontsize=10)
    ax_sample.axis("off")

    # === EVENT ROWS ===
    crop_axes_for_connectors = []  # (ax, time_sec, event_type) for drawing connectors

    for row_idx in range(n_event_rows):
        row_start = row_idx * EVENTS_PER_ROW
        row_end = min(row_start + EVENTS_PER_ROW, n_events)
        row_events = events[row_start:row_end]
        n_in_row = len(row_events)

        # Each event has 3 columns (before, at, after)
        n_cols = n_in_row * 3
        event_gs = outer_gs[1 + row_idx].subgridspec(
            2, n_cols, wspace=0.05, hspace=0.08)

        for evt_idx, event in enumerate(row_events):
            evt_color = _event_color_hex(event["type"])
            col_base = evt_idx * 3

            for frame_idx, fdata in enumerate(event.get("frames", [])):
                col = col_base + frame_idx

                # Full frame
                ax_frame = fig.add_subplot(event_gs[0, col])
                ax_frame.imshow(fdata["full_frame"])
                ax_frame.set_title(fdata["label"], fontsize=7, color=evt_color,
                                   fontweight="bold")
                ax_frame.axis("off")
                # Color border
                for spine in ax_frame.spines.values():
                    spine.set_edgecolor(evt_color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)

                # Crop
                ax_crop = fig.add_subplot(event_gs[1, col])
                ax_crop.imshow(fdata["crop"])
                if frame_idx == 1:  # "At event" — show event label
                    ax_crop.set_title(event["label"], fontsize=7,
                                      color=evt_color, fontweight="bold")
                ax_crop.axis("off")
                for spine in ax_crop.spines.values():
                    spine.set_edgecolor(evt_color)
                    spine.set_linewidth(2)
                    spine.set_visible(True)

                # Track for connector drawing
                crop_axes_for_connectors.append(
                    (ax_crop, fdata["time_sec"], event["type"]))

    # === TIMELINE ===
    timeline_gs = outer_gs[-1].subgridspec(1, 1)
    ax_timeline = fig.add_subplot(timeline_gs[0])

    # Plot filtered score curves if available
    if cluster_data is not None:
        home_filt = cluster_data["home_filtered"]
        away_filt = cluster_data["away_filtered"]
        hf_t = [f / fps / 60 for f in sorted(home_filt.keys())]
        hf_v = [home_filt[f] for f in sorted(home_filt.keys())]
        af_t = [f / fps / 60 for f in sorted(away_filt.keys())]
        af_v = [away_filt[f] for f in sorted(away_filt.keys())]

        ax_timeline.plot(hf_t, hf_v, linewidth=1.5, c="blue", alpha=0.7,
                         label="Home")
        ax_timeline.plot(af_t, af_v, linewidth=1.5, c="red", alpha=0.7,
                         label="Away")
        y_max = max(max(hf_v + [0]), max(af_v + [0])) + 1
    else:
        y_max = 5

    # GT event shading
    for i_gt, (start, end) in enumerate(gt_times):
        ax_timeline.axvspan(start / 60, end / 60, alpha=0.25, color="green",
                            label="GT goal" if i_gt == 0 else None)

    # Detected event lines
    for i_det, ev in enumerate(detected_events):
        ax_timeline.axvline(ev.timestamp / 60, color="purple", linewidth=2,
                            alpha=0.7, label="Detected" if i_det == 0 else None)

    ax_timeline.set_xlabel("Time (m:ss)")
    ax_timeline.set_ylabel("Score")
    ax_timeline.set_ylim(-0.5, y_max)
    ax_timeline.set_xlim(0, total_duration / 60)
    ax_timeline.xaxis.set_major_formatter(FuncFormatter(_mmss_formatter))
    ax_timeline.legend(loc="upper left", fontsize=8)
    ax_timeline.grid(True, alpha=0.3)

    # === DASHED CONNECTORS ===
    fig.canvas.draw()  # Finalize positions before drawing connectors

    for ax_crop, time_sec, event_type in crop_axes_for_connectors:
        evt_color = _event_color_hex(event_type)
        time_min = time_sec / 60

        # Only draw if time is within timeline range
        if time_min < 0 or time_min > total_duration / 60:
            continue

        con = ConnectionPatch(
            xyA=(0.5, 0.0), coordsA=ax_crop.transAxes,
            xyB=(time_min, y_max), coordsB=ax_timeline.transData,
            color=evt_color, linewidth=0.8, linestyle="--", alpha=0.5,
        )
        fig.add_artist(con)

    # Save
    out_path = output_dir / f"{match_id}_ocr_paper_figure.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def _event_color_hex(event_type):
    """Return hex color for matplotlib."""
    if event_type == "tp":
        return COLOR_TP
    elif event_type == "fn":
        return COLOR_FN
    else:
        return COLOR_FP
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from generate_ocr_paper_figure import generate_paper_figure
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_ocr_paper_figure.py
git commit -m "Add paper figure generation with nested GridSpec layout"
```

---

### Task 5: Add main() and CLI

**Files:**
- Modify: `scripts/generate_ocr_paper_figure.py`

- [ ] **Step 1: Add main function**

```python
def main():
    parser = argparse.ArgumentParser(
        description="Generate paper-quality OCR scoreboard figures")
    parser.add_argument("--matches", help="Comma-separated match IDs (default: all)")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output-dir", default="results/ocr_paper_figures")
    parser.add_argument("--offset", type=int, default=30,
                        help="Seconds before/after event for frame extraction")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = PipelineConfig.from_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matches_yaml = Path(__file__).parent.parent / "config" / "matches.yaml"
    with open(matches_yaml) as f:
        all_matches = list(yaml.safe_load(f)["matches"].keys())

    if args.matches:
        match_ids = [m.strip() for m in args.matches.split(",")]
    else:
        match_ids = all_matches

    for match_id in match_ids:
        print(f"\n{'=' * 60}")
        print(f"Processing {match_id}")
        print(f"{'=' * 60}")
        info = load_match_info(match_id, config)
        try:
            generate_paper_figure(match_id, info, config, output_dir,
                                  offset=args.offset)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test on a single match**

```bash
source venv/bin/activate
python scripts/generate_ocr_paper_figure.py --matches Match5
```

Expected: Generates `results/ocr_paper_figures/Match5_ocr_paper_figure.png`. Open it and visually verify:
- Top row shows IQR map, overlay mask, sample frame
- Event rows show before/at/after frames with crops
- Bottom shows score timeline
- Dashed connectors link crops to timeline
- Event borders are color-coded (green=TP, orange=FN, red=FP)

- [ ] **Step 3: Fix any visual issues found during review**

Common issues to watch for:
- Connector lines misaligned → adjust `ConnectionPatch` coordinates
- Event rows too cramped → increase `fig_height` or adjust height ratios
- Crop images too small → increase event row height ratio
- Title overlap → adjust `y` parameter of `suptitle`

- [ ] **Step 4: Test on a match with many events (e.g., Match8 with FPs)**

```bash
python scripts/generate_ocr_paper_figure.py --matches Match8
```

Verify the wrapping works correctly for >4 events.

- [ ] **Step 5: Test on a match with 0 events edge case**

If any match has 0 events, verify the figure renders without event rows.

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_ocr_paper_figure.py
git commit -m "Complete OCR paper figure script with CLI and event visualization"
```

---

### Task 6: Run on all 25 matches and verify

- [ ] **Step 1: Generate all figures**

```bash
python scripts/generate_ocr_paper_figure.py
```

This will take a while (OCR reading collection per match). Run in a screen session if needed:
```bash
screen -S paper_fig
source venv/bin/activate
python scripts/generate_ocr_paper_figure.py 2>&1 | tee results/ocr_paper_fig.log
```

- [ ] **Step 2: Spot-check 3-5 figures visually**

Open figures for Match1 (Strategy 1), Match5 (Strategy 2), Match8 (known FPs), Match10 (perfect match), and verify they all look reasonable.

- [ ] **Step 3: Final commit**

```bash
git add scripts/generate_ocr_paper_figure.py
git commit -m "Finalize OCR paper figure script after full 25-match validation"
```
