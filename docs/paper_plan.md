# Plan: Paper-Ready Experiments for Soccer Video Summarization

## Context

The clean modular pipeline is complete and verified on a test clip. Now we need to run all experiments, generate all figures/tables, and produce everything needed for a journal paper. The critical constraint: **no external baselines exist** (no public system or dataset for this exact task), so the paper must be rigorous through internal baselines, thorough ablation, component analysis, statistical testing, and error analysis.

**Target venues**: Multimedia Tools and Applications, Signal Processing: Image Communication, IEEE Access — systems papers that value strong methodology over SOTA comparisons.

---

## Phase 1: Full Pipeline on All 25 Matches

**Goal**: Get the main results table.

### Step 1: Run full pipeline (all 25 matches, default config)
```bash
python scripts/run_all_matches.py --skip-video-export
```
- Cached per stage: shot boundaries, YOLO counts, play segments, OCR scores, highlights
- Produces `results/per_match/MatchN/evaluation.json` for each match
- Estimated time: ~15-25h GPU (YOLO dominates at ~20-40min/match)

### Step 2: Review results
- Run `python scripts/evaluate.py` and `python scripts/generate_tables.py`
- Sanity-check: expected ~139 GT goals across 25 matches. Verify GT parsing gives correct goal counts per match.
- Flag any broken matches (video issues, encoding problems, OCR failures)

---

## Phase 2: Extend Evaluation Metrics

**Files to modify**: `soccer_summarization/evaluation.py`

Add these metrics to `EvaluationResult` and `evaluate_match()`:

| New Metric | Definition | Why |
|---|---|---|
| **Temporal IoU (tIoU)** | IoU between each predicted highlight segment and its closest GT goal segment | Standard in temporal localization |
| **Mean tIoU** | Average tIoU across all matched goal events | Aggregate quality measure |
| **Goal timing MAE** | For TPs: mean absolute error between predicted goal time and GT goal midpoint (seconds) | Shows temporal precision |
| **Segment count** | Number of highlight segments produced | Fewer = better viewing experience |
| **Mean segment duration** | Average highlight segment length (seconds) | Longer = more coherent |
| **Highlight F1 at tIoU thresholds** | F1 at tIoU = {0.1, 0.3, 0.5, 0.7} | Like mAP in object detection |

Also extend `aggregate_results()` to compute bootstrap 95% CIs for all metrics (1000 iterations).

---

## Phase 3: Internal Baselines

**New file**: `scripts/run_baselines.py`

Each baseline produces highlight segments in the same format → evaluated with same metrics.

### Baseline 1: Random Sampling (lower bound)
Select random segments totaling the same duration as the system output. Average over 10 random seeds.

### Baseline 2: Uniform Sampling
K evenly-spaced segments of fixed duration. Tune K to match the system's compression ratio.

### Baseline 3: Shot-Length Only (no YOLO)
Keep all shots > T seconds. No play/no-play filtering. Sweep T to match compression ratio. Shows YOLO's contribution.

### Baseline 4: Play-Only (no shot structure)
Use play classification directly as highlights (ignore shot boundaries). Shows shot segmentation's contribution.

### Baseline 5: OCR-Only
Include ±30s context around each OCR-detected goal. No visual analysis. Isolates OCR value.

### Baseline 6: Oracle Upper Bound
Use GT goal annotations directly. Include ±30s around each GT goal. Shows theoretical ceiling.

---

## Phase 4: Ablation Studies

**New file**: `scripts/run_ablation.py` — varies one parameter, reuses cached upstream stages.

Output structure: `results/ablation/{param_name}_{value}/MatchN/`

### Ablation A: play_threshold ← MOST CRITICAL
- **Values**: [3, 4, **5**, 6, 7]
- **Re-runs**: PlayClassifier + HighlightAssembler + Evaluation (YOLO cached)
- **Time**: ~minutes per match per value

### Ablation B: smoothing_window
- **Values**: [25, **50**, 75, 100]
- **Re-runs**: PlayClassifier + HighlightAssembler + Evaluation

### Ablation C: min_shot_duration
- **Values**: [2.0, 3.0, **4.0**, 5.0, 7.0]
- **Re-runs**: HighlightAssembler + Evaluation only (play segments cached)

### Ablation D: YOLO confidence_threshold
- **Values**: [0.3, 0.4, **0.5**, 0.6, 0.7]
- **Re-runs**: Full YOLO re-run needed
- **Run on 5 representative matches only** (Match1, Match5, Match10, Match15, Match22) to save time

### Ablation E: frame_step (YOLO sampling rate)
- **Values**: [1, 2, **5**, 10]
- **Re-runs**: Full YOLO re-run
- **5-match subset**. Also produces speed-accuracy tradeoff plot.

### Ablation F: merge_gap_threshold
- **Values**: [0.0, 0.2, **0.5**, 1.0, 2.0]
- **Re-runs**: HighlightAssembler + Evaluation only

### Ablation G: tolerance_window (evaluation protocol sensitivity)
- **Values**: [10, 15, 20, **30**, 45, 60]
- **Re-runs**: Evaluation only (no pipeline change). Shows metrics aren't cherry-picked.

Defaults in **bold**.

---

## Phase 5: Component Contribution Analysis

Run the pipeline with systematic component removal:

| Config | Shot Seg | YOLO+Play | OCR | How |
|---|---|---|---|---|
| **Full system** | ✓ | ✓ | ✓ | Default |
| **No OCR** | ✓ | ✓ | ✗ | `--skip-ocr` |
| **No play filter** | ✓ | ✗ | ✓ | Set `play_threshold=0` |
| **No shot filter** | ✓ | ✓ | ✓ | Set `min_shot_duration=0` |
| **Play-only** | ✗ | ✓ | ✗ | Treat video as one shot |

---

## Phase 6: Error Analysis

**New file**: `scripts/error_analysis.py`

For each match, categorize every goal event:
- **TP**: Which component detected it? (OCR score change? Play segment overlap?)
- **FN**: Why was it missed? Categories: (a) OCR failed to read scoreboard, (b) goal moment classified as no-play, (c) shot too short and filtered, (d) scoreboard not visible
- **FP**: Why the false alarm? Categories: (a) OCR misread, (b) score displayed during replay, (c) scoreboard format change

Additional analyses:
- **Per-season breakdown**: Do 2013/14 matches (different broadcasters) perform differently from 2023/24?
- **Goal count correlation**: Scatter plot of F1 vs number of goals per match
- **Temporal analysis**: Are early-match goals detected better than late-match goals?

---

## Phase 7: Statistical Analysis

**New file**: `scripts/statistical_analysis.py`

- **Bootstrap 95% CIs** for all aggregate metrics (1000 iterations, sampling with replacement from 25 matches)
- **Paired Wilcoxon signed-rank tests** comparing each ablation variant vs default, each baseline vs full system
- **Bonferroni correction** for multiple comparisons within each ablation group
- **Effect size** (Cliff's delta) for key comparisons

---

## Phase 8: Figures and Tables

**New file**: `scripts/generate_figures.py`

### Tables (8)

1. **Dataset statistics**: Match ID, teams, score, season, GT goals, duration
2. **YOLOv8 model performance**: Per-class AP, mAP@50, training details
3. **Main per-match results**: P/R/F1, compression, goal coverage, tIoU (all 25 matches)
4. **Aggregate results with CIs**: Macro/micro P/R/F1, compression, coverage ± CI
5. **Component contribution**: Full vs no-OCR vs no-play vs no-shot
6. **Baselines comparison**: Random, uniform, shot-only, play-only, OCR-only, oracle
7. **Ablation results**: One sub-table per key parameter (play_threshold, smoothing_window, min_shot_duration)
8. **Runtime analysis**: Per-stage timing breakdown, total per match

### Figures (10)

1. Pipeline architecture (exists, polish for journal)
2. Dataset overview — goals distribution, seasons (exists, polish)
3. Scoreboard detection — variance map, thresholding, OCR result (exists, polish)
4. **Full-match timeline** — extend existing 2-min visualization to one complete match
5. **Play threshold ablation** — line plot: X=threshold, Y=F1/P/R, second panel for compression
6. **Smoothing window ablation** — same format
7. **Speed-accuracy tradeoff** — X=frame_step, Y=F1, second axis=runtime
8. **Tolerance window sensitivity** — X=tolerance, Y=F1 (shows evaluation isn't gamed)
9. **Error analysis examples** — 2x2 grid: TP, FP, FN, OCR misread (actual frames)
10. **Player count distribution** — histogram of smoothed counts, bimodal (play vs no-play), threshold line

---

## Phase 9: Runtime Analysis

Report wall-clock time per stage for all 25 matches:
- Shot segmentation (multiprocessing)
- YOLO object detection (GPU)
- Play classification (CPU, fast)
- OCR scoreboard detection (GPU)
- Highlight assembly (CPU, instant)
- Total pipeline time vs video duration (real-time factor)

Already partially collected in `evaluation.json` timings dict.

---

## Execution Order & Dependencies

```
Phase 1 (full run, 25 matches) ─────────────────────────── Days 1-3
    │
Phase 2 (extend metrics) ───────────────────────────────── Day 3-4
    │
    ├── Phase 3 (baselines) ─────────────────────────────── Days 4-5
    ├── Phase 4 (ablations, cheap ones first) ───────────── Days 4-7
    └── Phase 5 (component contribution) ────────────────── Day 5
         │
    Phase 6 (error analysis) ────────────────────────────── Day 7-8
    Phase 7 (statistical analysis) ──────────────────────── Day 8
    Phase 8 (figures & tables) ──────────────────────────── Days 8-10
    Phase 9 (runtime analysis) ──────────────────────────── Day 9
```

Phases 3/4/5 can run in parallel once Phase 2 is done.

---

## Files to Create

| File | Purpose |
|---|---|
| `scripts/run_ablation.py` | Sweep one parameter, reuse caches, collect results |
| `scripts/run_baselines.py` | All 6 internal baselines |
| `scripts/error_analysis.py` | TP/FP/FN categorization per match |
| `scripts/statistical_analysis.py` | Bootstrap CIs, Wilcoxon tests |
| `scripts/generate_figures.py` | All matplotlib paper figures |

## Files to Modify

| File | Changes |
|---|---|
| `soccer_summarization/evaluation.py` | Add tIoU, MAE, segment stats, bootstrap CIs |
| `scripts/generate_tables.py` | Extend for ablation/baseline/component tables |
| `scripts/run_single_match.py` | Support reading cached intermediates from alternate directories (for ablations) |

---

## Verification

After each phase:
1. **Phase 1**: Check GT goal counts match known scores (e.g., Match1 Valencia 2-3 Barcelona = 5 goals)
2. **Phase 2**: Verify new metrics on test clip produce sensible values
3. **Phase 3**: Random baseline should have near-zero goal detection; oracle should have perfect recall
4. **Phase 4**: Check that default parameter always appears in ablation results and matches Phase 1
5. **Phase 6**: Total TP+FN per match should equal GT goal count
6. **Phase 7**: CIs should be tighter for metrics with low variance, wider for high-variance
7. **Phase 8**: LaTeX tables compile, figures render at journal resolution (300+ dpi)
