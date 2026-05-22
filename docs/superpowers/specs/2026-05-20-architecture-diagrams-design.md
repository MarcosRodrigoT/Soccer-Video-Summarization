# Architecture Diagrams for Soccer Video Summarization Paper

## Context

The project lacks publication-grade vector diagrams. Existing PNGs in `docs/figures/` (pipeline_architecture.png, dataset_overview.png, timeline_visualization.png, scoreboard_detection.png) are useful internal references but are raster, somewhat informal, and only cover the high-level pipeline. The user is preparing a journal paper and needs a comprehensive vector-graphics catalog (SVG + PDF) that covers the full system: top-level architecture, per-stage internals, data flow, evaluation, dataset, and qualitative examples on real frames.

## Goals

- Vector outputs (SVG + PDF) for every figure
- Clean academic style: white background, subtle pastel fills, thin black borders, rounded corners, sans-serif labels
- Color identity inherited from the existing diagram (blue=input, green=visual branch, yellow=scoreboard branch, red=intersection, purple=output, dark-blue=evaluation)
- Single-column journal sizing (default 6.5in usable width)
- Reproducible via Python scripts under `scripts/figures/`
- Both schematic block diagrams and qualitative examples grounded in real Match1 data

## Output

All figures written to `docs/figures/` (alongside existing PNGs, not replacing them). Each figure: `<name>.svg` and `<name>.pdf`.

## Figure Catalog

| # | Filename stem | Type | Purpose |
|---|---|---|---|
| 1 | `system_architecture` | Block diagram | The canonical Figure 1: full end-to-end pipeline showing all 6 stages, two branches (visual + scoreboard), inputs, outputs, and ground truth. |
| 2 | `data_flow` | Dependency graph | Artifact-level view: which CSV/JSON files each stage produces and consumes. Read complements (1). |
| 3 | `shot_segmentation_detail` | Sub-block diagram | Frame differencing → adaptive threshold → minimum-duration filtering. |
| 4 | `player_detection_detail` | Sub-block diagram | YOLOv8 inference, 4-class detection, frame sub-sampling at 5fps. |
| 5 | `play_classification_detail` | Sub-block + signal plot | Player count → smoothing window → threshold → contiguous play/no-play segments. |
| 6 | `ocr_scoreboard_pipeline` | Sub-block diagram | Per-pixel IQR map → percentile threshold → morphology → CC → bbox tightening → EasyOCR. |
| 7 | `goal_event_state_machine` | State diagram | Filtering chain that converts raw OCR readings into authoritative score events (temporal median, cluster concentration, static-element filter, persistence, max-jump). |
| 8 | `highlight_assembly_detail` | Sub-block diagram | Intersection of long shots ∩ play segments + goal-context window merging. |
| 9 | `evaluation_framework` | Diagram | Predictions vs. GT matching at tolerance window, P/R/F1, compression ratio, mean tIoU, goal timing MAE. |
| 10 | `dataset_overview` | Three-panel statistical | Per-match goal bars, goal distribution histogram, season pie. Vector remake of existing PNG. |
| 11 | `experiment_suite` | Map diagram | Shows how baselines, ablations, component analysis, error analysis, and statistical tests relate to the core pipeline. |
| 12 | `qualitative_scoreboard` | Real-frame composite | Match1 representative frame → cropped scoreboard region → IQR heatmap → binary mask → tightened bbox → OCR text overlay. |
| 13 | `qualitative_player_detection` | Real-frame composite | Match1 frame with YOLOv8 detections rendered (ball/goalkeeper/player/referee). |
| 14 | `qualitative_timeline` | Stacked timeline | Vector remake of existing `timeline_visualization.png` (Match1 2-min clip with shots, players, plays, GT, highlights). |
| 15 | `error_taxonomy` | Schematic | TP/FP/FN categorization: missed goals, stat overlay false positives, spurious cuts, etc. |

## Structure

```
scripts/figures/
├── _style.py                              # Shared matplotlib styling (colors, sizes, save helper)
├── fig01_system_architecture.py
├── fig02_data_flow.py
├── fig03_shot_segmentation_detail.py
├── fig04_player_detection_detail.py
├── fig05_play_classification_detail.py
├── fig06_ocr_scoreboard_pipeline.py
├── fig07_goal_event_state_machine.py
├── fig08_highlight_assembly_detail.py
├── fig09_evaluation_framework.py
├── fig10_dataset_overview.py
├── fig11_experiment_suite.py
├── fig12_qualitative_scoreboard.py
├── fig13_qualitative_player_detection.py
├── fig14_qualitative_timeline.py
├── fig15_error_taxonomy.py
└── render_all.py                          # Runs all figure scripts; reports any failures
```

`_style.py` exports:
- `COLORS` dict (input, visual, scoreboard, intersection, output, evaluation, ground_truth, neutral_fill, neutral_edge)
- `set_paper_style()` — applies rcParams (font family, sizes, axes spines)
- `save_figure(fig, name)` — writes both `docs/figures/<name>.svg` and `docs/figures/<name>.pdf`
- Helper `rounded_box(ax, x, y, w, h, label, color)` to draw consistent stage blocks

## Data Sources for Qualitative Figures (12–14)

- Match1 video at `Laliga_HR/Match1/Valencia CF - FC Barcelona (2-3) ... [Qzrg3QDmQ4k].mp4`
- Cached artifacts under `results_hr/per_match/Match1/`:
  - `iqr_map.npy` — pre-computed per-pixel IQR map for Figure 12
  - `scoreboard_region.json` — bbox {x:120, y:83, w:373, h:46} (480p coordinates; scale ×2.25 for 1080p)
  - `score_events.csv` — 5 goal events confirming detection
  - `shot_boundaries.csv`, `player_counts.csv`, `play_segments.csv`, `highlights.csv` — for Figure 14 timeline

For Figure 13 (player detection), the script will load the YOLO model from `data/models/best.pt` and run on a single frame sampled around a play segment.

## Verification

- After each script writes its outputs, confirm both `.svg` and `.pdf` exist with reasonable file size (>5KB schematic, >50KB for raster-embedded qualitative figures).
- Visually inspect at least one figure (the main system architecture) by reading the rendered SVG into Claude.
- Re-running `python scripts/figures/render_all.py` from a fresh shell should regenerate every figure deterministically.
