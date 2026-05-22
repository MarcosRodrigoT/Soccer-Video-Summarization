# Timing report — run `20260522_144051`

- Matches: **25**
- Mean video duration: **01:48:50.20**
- Mean total pipeline time per match: **00:00.16** (median 00:00.16, min 00:00.13, max 00:00.21)
- Sum over all matches: **00:04.06**
- Mean real-time factor: **0.000×** (processing time / video duration; <1 means faster than real-time)

## Per-stage compute cost

| Stage | n compute | n cached | mean (s) | median (s) | min (s) | max (s) |
|---|---:|---:|---:|---:|---:|---:|
| Shot segmentation | 0 | 25 | 0.00 | 0.00 | 0.00 | 0.00 |
| YOLO detection | 0 | 25 | 0.00 | 0.00 | 0.00 | 0.00 |
| Play classification | 0 | 25 | 0.00 | 0.00 | 0.00 | 0.00 |
| OCR scoreboard | 0 | 25 | 0.00 | 0.00 | 0.00 | 0.00 |
| Highlight assembly | 25 | 0 | 0.06 | 0.06 | 0.05 | 0.07 |
| Goal-shot selection | 25 | 0 | 0.10 | 0.10 | 0.08 | 0.13 |
| Evaluation | 25 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |

*Cached* counts indicate stages that were skipped because their output was already on disk; they're excluded from the time aggregates.

## Files in this run

- `Match*.json` — per-match per-stage timings (raw).
- `summary.json` — aggregate stats.
- `breakdown.png` — per-stage mean compute time (bar chart).
- `per_match.png` — per-match total time, stacked by stage.