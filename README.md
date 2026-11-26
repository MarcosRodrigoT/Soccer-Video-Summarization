# Goal-Centered Soccer Highlights from Full-Length Broadcasts

Code, ground truth and results of the paper *"Broadcast-State-Guided Generation of Goal-Centered Soccer Highlights from Full-Length Match Videos"* (M. Rodrigo, C. Cuevas, N. García; Universidad Politécnica de Madrid).

Given a full-length broadcast, the pipeline returns one short clip per goal: the final seconds of the live play that leads to it. It needs no event-level training data. Goals are located by reading the on-screen scoreboard, and the clip is taken from the wide-angle play shot that precedes the score update.

## Pipeline

Two branches run on the input video and meet at the highlight assembly stage.

**Visual branch**
1. **Shot boundary detection** (`shot_segmentation.py`). Histogram differencing between consecutive frames, with separate detectors for abrupt cuts and gradual transitions (dissolves, wipes). It needs no training data and runs on the CPU.
2. **Player detection** (`object_detection.py`). A YOLOv5l6u detector with four classes (ball, goalkeeper, player, referee) runs on every 5th frame (5 fps). The three person classes are counted as visible people.
3. **Shot characterization** (`highlight_assembler.py`). Each shot is described by its mean people count over its sampled frames. A shot is a *play long shot* when it lasts at least 4 s and its mean count is at least 5. Wide-angle live play shows many people; replays, close-ups and crowd shots show few.

**Scoreboard branch** (`ocr_scoreboard.py`)
4. **Scoreboard localization**. A per-pixel interquartile-range map over 200 frames finds the static overlays. EasyOCR then picks the overlay that reads as a score. No template or position prior is used.
5. **Score tracking**. The scoreboard crop is read every 3 s, and a cascade of filters removes OCR errors and stat overlays (temporal median, monotonic scores, persistence, maximum jump). Each accepted score change is a goal event with its time and scoring team.

**Highlight assembly** (`highlight_assembler.py`)
6. For each goal, the anchor is the play long shot whose end is closest to the score update among those ending between 30 s before and 5 s after it. Consecutive play long shots separated by at most 2 s (two camera angles of the same move) are joined. The clip is the last 10 s of the anchor.

All parameters are in `config/default_hr_10s.yaml` and are the same for every match, season and league.

## Detector weights

The detector is the YOLOv5l6u model released by the open-source [Football Analysis](https://github.com/abdullahtarek/football_analysis) project ("Trained Yolo v5" link in its README). It was trained for 100 epochs on the Roboflow [football-players-detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) dataset and is used here without further training. Place it at `data/models/best.pt` in the repository root (or set `object_detection.model_path` in the config). Its validation metrics, read from the checkpoint, are in `results_hr/detector_validation.json` (precision 0.96, recall 0.74, mAP@50 0.83).

## Data

The broadcasts are not redistributed. [`dataset/`](dataset/README.md) contains the ground truth and the YouTube identifier of each broadcast:

- 27 broadcasts: 25 LaLiga matches (seasons 2013/14 to 2023/24, 139 goals) and two matches from other leagues (Premier League and Serie A, 16 goals);
- frame-level goal annotations (`dataset/goals/`);
- shot-level annotations of a 5-min window of each broadcast: every cut and gradual transition, and a play / no-play label per shot (`dataset/shots/`).

`config/matches_hr.yaml` and `config/matches_other_leagues.yaml` give the expected location of each video under `data_dir` (set in the config files). Videos are decoded at 25 fps.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install ultralytics opencv-python easyocr pandas numpy matplotlib pyyaml tqdm scipy
```

A CUDA GPU is recommended for the detector and EasyOCR. `docs/env/` records the exact package versions used.

The shot boundary baselines need `transnetv2-pytorch`, `scenedetect` and `sportsbd` 0.1.2 (`pip install --no-deps sportsbd==0.1.2`, since it pins an older NumPy; its released weights go to `data/models/sportsbd_best.pt`, or set `SPORTSBD_WEIGHTS`).

## Reproducing the paper

```bash
# Full pipeline on the 25 LaLiga matches (writes results_hr_10s/per_match/)
python scripts/run_all_matches.py --config config/default_hr_10s.yaml \
    --matches-yaml config/matches_hr.yaml --skip-video-export

# End-to-end numbers, per-match table and bootstrap CIs
python scripts/summarize_end_to_end.py

# Scoreboard-update delay and clip-length sensitivity
python scripts/analyze_scoreboard_delay.py
python scripts/run_tail_sweep.py

# Component-level evaluation on the annotated windows
python scripts/run_sbd_baseline.py --method transnetv2      # also: pyscenedetect, sportsbd
python scripts/annotations/evaluate_carlos_sbd.py
python scripts/run_sbd_end_to_end.py --methods ours transnetv2 pyscenedetect sportsbd

# Premier League and Serie A matches
python scripts/run_other_leagues.py --matches PL1 SA1
python scripts/evaluate_other_leagues.py --matches PL1 SA1

# Runtime on Match 1
python scripts/run_runtime_analysis.py --match-id Match1 --config config/default_hr.yaml
```

`scripts/run_single_match.py --match-id MatchN --config config/default_hr_10s.yaml` processes a single match. Intermediate results are cached, so re-running a match skips completed stages. `scripts/generate_match_dashboard.py` draws a per-match diagnostic dashboard from the cached results.

## Results

End-to-end, 25 LaLiga matches (`results_hr_10s/aggregate/end_to_end_summary.json`). A goal is correct when the predicted time is within 30 s of the annotated goal and the clip overlaps the annotated clip with IoU ≥ 0.5.

| Metric | Value |
|---|---|
| Goals found by the scoreboard branch | 138 of 139 (1 false positive) |
| End-to-end TP / FP / FN | 134 / 5 / 5 |
| End-to-end F1 (mean over matches) | 0.97 (95% bootstrap CI 0.93–0.99) |
| Matches with F1 = 1 | 21 of 25 |
| Mean clip IoU | 0.98 |
| Compression | > 99% |

Component level, 27 annotated 5-min windows (`annotations_v2/results_sbd_play.json`; 765 transitions, 790 shots):

| Shot boundary detector | Precision | Recall | F1 | Recall (cuts) | Recall (gradual) |
|---|---|---|---|---|---|
| Ours | 0.94 | 0.98 | 0.96 | 1.00 | 0.95 |
| TransNetV2 | 1.00 | 0.80 | 0.89 | 0.98 | 0.42 |
| SportSBD* | 0.98 | 0.91 | 0.95 | 0.97 | 0.79 |
| PySceneDetect | 0.75 | 0.85 | 0.80 | 0.94 | 0.67 |

*SportSBD reports its boundaries on a 4-frame grid, 0–3 frames before the cut, so it is matched within ±3 frames; the other detectors within ±2 frames.

The play / no-play decision reaches a precision of 0.93 and a recall of 1.00 over all annotated shots. On the Premier League and Serie A matches, run with the same parameters, all 16 goals are found (`results_other_leagues/evaluation_summary.json`). The Bundesliga and Ligue 1 broadcasts in `results_other_leagues/` were processed but have no ground truth.

## Repository layout

```
soccer_summarization/    pipeline package (one module per stage, evaluation, metrics)
scripts/                 entry points, evaluation and figure scripts
config/                  pipeline parameters and match lists
dataset/                 public ground truth
annotations_v2/          raw annotation sessions and their evaluation
tools/annotator/         web tool used for the shot and goal annotations
results_hr_10s/          per-match evaluation and aggregate results reported in the paper
results_hr/              cached intermediate outputs, full-shot variant, runtime analysis
results_sbd/             TransNetV2, SportSBD and PySceneDetect outputs and their end-to-end results
results_other_leagues/   cross-league runs and evaluation
docs/                    documentation figures and package versions
```
