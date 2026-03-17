# OCR Paper Figure — Design Spec

## Goal

Create a script that generates a per-match paper-quality figure showing the OCR scoreboard detection pipeline in action: static overlay analysis at the top, before/at/after frames around each goal event in the middle, and the filtered score timeline at the bottom, with dashed connectors linking frames to their timeline positions.

## Output

- One PNG per match in `results/ocr_paper_figures/`
- Named `{match_id}_ocr_paper_figure.png`

## Figure Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Title: "Match5: FC Barcelona 1-3 Real Madrid"              │
├──────────────────┬──────────────────┬───────────────────────┤
│   IQR map        │  Overlay mask    │ Sample frame + bbox   │
├──────────────────┴──────────────────┴───────────────────────┤
│  Event row 1 (up to 4 events, chronological):               │
│  ┌── Event 1 (TP) ──┐  ┌── Event 2 (FN) ──┐  ...          │
│  │ Before  At  After │  │ Before  At  After │               │
│  │ [img] [img] [img] │  │ [img] [img] [img] │               │
│  │ [crp] [crp] [crp] │  │ [crp] [crp] [crp] │               │
│  └───╎─────╎────╎────┘  └───╎─────╎────╎────┘               │
│  Event row 2 (overflow if >4 events):                       │
│  ...                                                        │
├─────────────────────────────────────────────────────────────┤
│  Score timeline (full width):                               │
│  Blue=home, Red=away filtered curves                        │
│  Green shading=GT, Purple lines=detected, dashed connectors │
└─────────────────────────────────────────────────────────────┘
```

### Top row: 3 panels side-by-side

1. **IQR variability map** — hot colormap, cyan rectangle on detected scoreboard region
2. **Overlay mask** — binary mask, cyan rectangle on scoreboard region
3. **Sample frame** — mid-match frame with green rectangle on scoreboard bbox

### Event rows: up to 4 events per row, wrapping

Each event occupies a 3-column × 2-row block:
- Top: 3 full frames (before, at event, after) with scoreboard bbox drawn
- Bottom: 3 corresponding scoreboard crops (4x upscaled, OCR annotations: green=inside bbox, gray=outside)

Color-coded by event type (border and title text):
- **Green** — TP (detected event matched to GT)
- **Orange** — FN (GT event with no matching detection)
- **Red** — FP (detection with no matching GT)

Maximum 12 events per figure. If the union exceeds 12, truncate with a note in the title.

### Bottom: Score timeline

Full-width plot showing:
- Blue/red filtered score curves (with monotonic enforcement)
- Green shading for GT event windows
- Purple vertical lines for detected events
- Dashed connector lines from each frame down to its x-position on the timeline, colored by event type

If `cluster_data` is None (no score curves available), show only GT shading and detected event lines.

### Zero events case

If a match has 0 GT events and 0 detected events, skip the event rows entirely. The figure shows only the top row and the timeline (with score curves but no event markers).

## Event Selection

**Union of GT events and detected events.**

Matching uses `Evaluator.evaluate_goal_detection()` from `soccer_summarization/evaluation.py` to ensure TP/FN/FP classifications are identical to the paper's reported metrics. The method iterates predictions in order and matches each to the first GT event whose window `[start - tolerance, end + tolerance]` contains it. One-to-one matching.

Classification:
- **TP**: detected event matched to a GT event
- **FN**: GT event with no matching detection
- **FP**: detected event with no matching GT

Events sorted chronologically.

## Frame Selection

For each event, 3 frames are extracted at specific timestamps (in seconds):
- **Before**: event center − offset (default 30s)
- **At**: event center
- **After**: event center + offset (default 30s)

Event center (in seconds):
- TP: detected event `timestamp` field (seconds)
- FN: GT event midpoint `(start_time + end_time) / 2` (seconds)
- FP: detected event `timestamp` field (seconds)

Convert to frame index: `frame_idx = int(center_seconds * fps)`.

Frames clamped to valid video range `[0, total_frames - 1]`. If clamping causes two frames to be identical (e.g., event near video start), annotate the duplicate with "(clamped)" in the title.

For each frame, extract:
1. Full frame with scoreboard bbox drawn as a colored rectangle (event-type color)
2. Scoreboard crop using `ScoreboardDetector._crop_region(frame, region, pad=15)`, upscaled 4x, with OCR annotations drawn (green for inside-bbox readings, gray for outside) — same logic as `get_annotated_frames()` in the diagnostics script.

## Dashed Connectors

Thin dashed lines from the bottom edge of each crop image to the corresponding x-position on the timeline axis. Colored by event type (green/orange/red). Drawn using `ConnectionPatch` with `coordsA="axes fraction"` and `coordsB="data"` after calling `fig.canvas.draw()` to get final axis positions and avoid misalignment from `bbox_inches="tight"`.

## Figure Sizing

- Width: `max(14, n_events_per_row * 4.5)` inches
- Height: 2.5 (top) + 4.5 per event row + 3 (timeline) inches. For 0 event rows: 2.5 + 3 = 5.5 inches.
- DPI: 150 for output PNG

## Layout Implementation

Use nested `GridSpec`:
- Outer grid: 3 sections (top row, event area, timeline) with variable height ratios
- Top row: inner 1×3 grid
- Each event row: inner 2×(n_events×3) grid (frames row + crops row, 3 columns per event)
- Timeline: single full-width axes

This avoids the complexity of a single flat grid with 12+ columns.

## CLI

Script: `scripts/generate_ocr_paper_figure.py`

```
--matches       Comma-separated match IDs (default: all)
--config        Config YAML (default: config/default.yaml)
--output-dir    Output directory (default: results/ocr_paper_figures/)
--offset        Seconds before/after event for frames (default: 30)
```

## Code Reuse

Imported from `scripts/visualize_ocr_diagnostics.py`:
- `load_match_info()`
- `collect_ocr_readings()`
- `extract_digit_clusters()`
- `extract_xy_pattern_scores()`
- `_mmss_formatter()`

Imported from core package:
- `ScoreboardDetector`, `ScoreboardRegion`
- `Evaluator` (for GT loading, event extraction, and TP/FN/FP matching)
- `get_video_properties()`

Detected events loaded from `results/per_match/{match_id}/score_events.csv` via `pd.read_csv()`. Expected columns: `timestamp`, `frame`, `home_score`, `away_score`, `team`. Wrapped in try/except for `EmptyDataError` and missing files (returns empty list).

New code:
- Event union construction using `Evaluator.evaluate_goal_detection()` for matching
- Frame extraction at event ± offset with scoreboard crop + OCR annotation
- Multi-panel figure layout with nested gridspec and auto-wrapping (max 4 per row)
- Dashed connector drawing with `ConnectionPatch`

## Files to Create

- `scripts/generate_ocr_paper_figure.py` — the script (single file)
