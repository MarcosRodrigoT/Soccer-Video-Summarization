# Automatic Soccer Video Summarization

A modular pipeline for automatic highlight extraction from full-length soccer broadcast footage. The system combines shot boundary detection, YOLOv8-based play/no-play classification, and OCR-based scoreboard reading to produce concise video summaries centered around goal events.

## Background

This project originates from a Master's thesis (TFM) by Lixing Fan at Universidad Politecnica de Madrid, titled *"Development of a Deep Learning Model for Video Summarization of Soccer Matches"*. The thesis explored multiple approaches (multimodal fusion, CLIP embeddings, transformer-based methods, WhisperX audio transcription) before converging on a pragmatic pipeline that combines visual analysis with scoreboard OCR. The original codebase was scattered across many experimental files. This repository is a clean, modular rewrite of the working components, designed to scale evaluation across 25 La Liga matches and produce journal-quality results.

## Pipeline Architecture

The system operates in two parallel branches that converge at the highlight assembly stage:

```
                        Input Video
                       /           \
              Visual Analysis    Scoreboard Analysis
              /          \              |
    Shot Boundary    YOLO Object     OCR Score
    Detection        Detection       Reading
         |               |              |
    Shot Filtering   Play/No-Play   Goal Event
    (min duration)   Classification  Detection
              \          /              |
           Intersection:               |
        Long Shots AND Play Moments    |
                       \              /
                     Highlight Assembly
                    (backward lookup from goals)
                            |
                      Summary Video
```

### Module Descriptions

| Module | Description | Method |
|--------|-------------|--------|
| **Shot Segmentation** | Detects shot boundaries (camera cuts) | Frame differencing with adaptive threshold: `tau = max(fixed, mu + k*sigma)`, local maximum detection, minimum shot length constraint |
| **Object Detection** | Counts players visible per frame | YOLOv8 Medium trained on custom 45,080-image dataset (2 classes: Ball, Players). mAP@50 = 0.929, F1 = 0.92 |
| **Play/No-Play Classifier** | Classifies time segments as active play or dead time | Smoothed player count (50-frame rolling average) thresholded at >= 5 players |
| **OCR Scoreboard** | Reads the broadcast scoreboard to detect score changes | Automatic scoreboard localization via pixel variance analysis + PaddleOCR |
| **Highlight Assembler** | Produces the final summary video | Intersection of (long shots) AND (play moments), merge adjacent segments, backward lookup from goal timestamps |

## Project Structure

```
soccer_summarization/          # Core Python package
    __init__.py
    config.py                  # YAML-based configuration loader
    shot_segmentation.py       # Shot boundary detection with multiprocessing
    object_detection.py        # YOLOv8 inference and per-frame player counting
    play_classifier.py         # Play/no-play classification with smoothing
    ocr_scoreboard.py          # Adaptive scoreboard localization and OCR
    highlight_assembler.py     # Shot filtering, intersection, segment merging
    evaluation.py              # Metrics computation (P/R/F1, compression, coverage)
    utils.py                   # Time formatting, I/O helpers

scripts/                       # Entry points
    run_single_match.py        # Process one match end-to-end
    run_all_matches.py         # Batch processing with intermediate caching
    evaluate.py                # Run evaluation against ground truth
    generate_tables.py         # Generate LaTeX-ready result tables

config/                        # Configuration files
    default.yaml               # Default hyperparameters and thresholds
    matches.yaml               # Per-match metadata (video path, GT, teams, score)

results/                       # Auto-populated by the pipeline
    per_match/                 # One directory per match with all intermediates
    aggregate/                 # Summary tables, plots, aggregate metrics
```

## Dataset

- **25 full La Liga matches** (Match1 through Match25), spanning seasons 2013/14 to 2023/24
- **Ground truth**: 25 CSV files with commentary-level annotations. Each row is a commentary segment with columns: `start`, `end`, `text`, `GoalGT1`, `GoalGT2` (binary flags indicating whether the segment overlaps with a goal event for each team)
- **Trained model**: YOLOv8 Medium (`best.pt`) trained on a consolidated Roboflow dataset of 45,080 images with 2 classes (Ball=0, Players=1)

## Implementation Roadmap

### Phase 1: Clean Project Structure & Modular Pipeline

Extract and refactor the four core modules from the original codebase into clean, configurable classes:

- `ShotSegmentation.py` &rarr; `shot_segmentation.py` — Parallel frame analysis with adaptive thresholding
- `YOLOpredictBox.py` &rarr; `object_detection.py` — YOLO inference with correct 2-class mapping, configurable frame sampling
- `ShotsegAndPlaynoplay.py` &rarr; `play_classifier.py` — Dual-criteria analysis (shot-level average + frame-level smoothed count)
- `ShotSegmentationAndPlayNoplayBoundary.py` &rarr; `highlight_assembler.py` — Duration filtering, intersection computation, segment merging, video export

All hardcoded paths and thresholds are moved to YAML configuration files.

### Phase 2: Adaptive OCR Scoreboard Detection

Replace the original hardcoded pixel coordinates with automatic scoreboard localization using **pixel variance analysis**:

1. **Build static overlay mask** — Sample ~50-100 frames across the match and compute per-pixel variance. Static overlays (scoreboard, channel logo) have low variance while the game action changes constantly
2. **Identify scoreboard** — Run PaddleOCR on each low-variance connected component, match score patterns via regex (`\d+\s*[-:]\s*\d+`)
3. **Track score changes** — Sample 1 frame every 2-5 seconds, crop only the cached scoreboard region, detect when the score increments

This approach works regardless of scoreboard position (top, bottom, corner) and handles different broadcast layouts automatically. As a fallback, WhisperX transcriptions can be searched for goal-related keywords ("gol", "golazo").

### Phase 3: Evaluation Framework

Bridge the gap between system output (highlight segments) and ground truth (commentary-level goal annotations):

- **Goal event extraction**: Group consecutive GoalGT=1 segments into discrete goal events (new event when gap > 30s)
- **Goal detection metrics**: Precision, Recall, F1 with configurable tolerance window (default +/-30s)
- **Highlight quality metrics**: Compression ratio, goal time coverage, false inclusion rate
- **Ablation study**: Sweep over player count threshold (3-7), min shot duration (3-7s), smoothing window (25-100 frames)

### Phase 4: Batch Processing

- Build a match registry mapping each match to its video file, GT file, and metadata
- Process all 25 matches with intermediate result caching at each stage (shot boundaries, YOLO detections, play segments, OCR scores, highlights)
- Speed optimizations: YOLO inference every 5th frame (5 fps is sufficient for player counting), OCR sampling every 125 frames

### Phase 5: Baselines & Comparisons

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Uniform sampling | Random segments of equal total duration | Lower bound |
| Shot-length-only | Keep shots > K seconds, no play/no-play filtering | Isolates YOLO contribution |
| Commentary keyword | Search WhisperX transcriptions for "gol" | Text-only baseline |
| System ablations | With/without OCR, varied thresholds | Component analysis |

### Phase 6: Paper-Ready Output

- Result tables (LaTeX): YOLO detection performance, shot segmentation statistics, per-match and aggregate goal detection P/R/F1
- Visualizations: Pipeline architecture diagram, shot+play timeline plots, precision-recall curves, ablation bar charts
- Statistical significance: Confidence intervals and Wilcoxon signed-rank tests across 25 matches

## Verification Plan

1. Run the clean pipeline on a single test match and compare against original results to ensure no regression
2. Validate GT parsing by manually checking 2-3 matches (compare extracted goal count against known match scores)
3. Run full batch on all 25 matches and inspect per-match results for outliers
4. Generate evaluation tables and verify metrics are reasonable

## Requirements

- Python 3.10+
- ultralytics (YOLOv8)
- opencv-python
- paddleocr
- pandas, numpy, matplotlib
- PyYAML
- tqdm

## Original Thesis

Fan, L. (2024). *Development of a Deep Learning Model for Video Summarization of Soccer Matches*. Master's Thesis, Universidad Politecnica de Madrid.
