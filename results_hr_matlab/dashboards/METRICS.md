# Dashboard Metrics

Every dashboard image has the same structure: a header with four colored
metric cards, several timeline plots, and a per-goal breakdown table at the
bottom. This document explains exactly what each metric means.

Symbols:
- `T_pred` = a predicted goal timestamp (from OCR scoreboard reading)
- `T_gt`   = a ground-truth goal timestamp (`end_frame_scoreboard / fps`)
- `tolerance` = 30 s (configured in `evaluation.tolerance_window`)

A predicted goal **matches** a GT goal if `|T_pred − T_gt| ≤ tolerance`.

---

## Card 1 — GOAL DETECTION (OCR timestamps vs GT)

Only the timestamps are compared. Clip boundaries do **not** affect this card.

| Metric | Definition |
|---|---|
| **TP** | `# GT goals` with at least one matching predicted timestamp |
| **FP** | `# predicted goals` that don't match any GT goal |
| **FN** | `# GT goals` with no matching predicted timestamp |
| **Precision** | `TP / (TP + FP)` |
| **Recall** | `TP / (TP + FN)` |
| **F1** | `2 · P · R / (P + R)` |
| **GT / OCR** | counts: `len(gt_events) / len(predicted_goal_times)` |

> *Reads*: "Did the OCR notice a goal? It doesn't say whether the clip is well-cut."

---

## Card 2 — CLIP QUALITY (mean across detected goals)

For each detected (TP) goal, the predicted goal clip
`[start_pred, end_pred]` is compared against the frame-level GT clip
`[start_frame / fps, end_frame_shot / fps]` from `goals.csv`. The card shows
the **mean** of each metric across all TPs (see the per-goal table at the
bottom for individual values).

Let `I = max(0, min(end_pred, end_gt) − max(start_pred, start_gt))` be the
overlap and `U = (end_pred − start_pred) + (end_gt − start_gt) − I` the union.

| Metric | Definition |
|---|---|
| **Clip IoU** | `I / U` — temporal Intersection-over-Union. Range `[0, 1]`. |
| **Coverage** | `I / (end_gt − start_gt)` — fraction of the GT clip captured by the prediction (= per-clip recall). |
| **Precision** | `I / (end_pred − start_pred)` — fraction of the predicted clip that lies inside the GT clip (= per-clip precision). |
| **Start MAE** | `mean(|start_pred − start_gt|)` in seconds. |
| **End MAE** | `mean(|end_pred − end_gt|)` in seconds. |

> *Reads*: "When we did detect a goal, how cleanly did we cut its clip?"

Undetected (FN) goals are **excluded** from these means — they get the dedicated
End-to-End card instead.

---

## Card 3 — END-TO-END (IoU ≥ 0.5)

A stricter F1 that requires both detection **and** a well-cut clip.

For each GT goal:
- It is a **TP** iff *(a)* some predicted timestamp falls within `tolerance` of
  `T_gt` **AND** *(b)* the best-matching predicted clip has `Clip IoU ≥ 0.5`.
- Otherwise it is **FN** (split in the table into `FN` if undetected and
  `FN (clip)` if detected but with a poor clip).

Predictions that don't match any GT in time are **FP**.

| Metric | Definition |
|---|---|
| **TP / FP / FN** | as above |
| **Precision** | `TP / (TP + FP)` |
| **Recall** | `TP / (TP + FN)` |
| **F1** | `2 · P · R / (P + R)` |

> *Reads*: "How many goals did we get **right end-to-end**?" This is the
> headline number for the system as a whole.

The IoU threshold is configurable (`evaluate_clip_aware_f1(iou_threshold=…)`)
and is shown in the card title.

---

## Card 4 — SUMMARY

| Metric | Definition |
|---|---|
| **Compression** | `1 − total_clip_duration / match_duration`. Higher = shorter summary. |
| **Goal clips** | Number of clips in the final output (one per detected goal). |
| **Total clip duration** | Sum of all goal-clip lengths (mm:ss). |
| **Match duration** | Original video length (hh:mm:ss). |

> *Reads*: "What does the actual deliverable look like in absolute terms?"

---

## Color coding

Values inside the cards are colored to give a quick read at a glance.

| Color | Score-like (F1, IoU, Coverage, Precision, Compression) | Error-like (Start MAE, End MAE) |
|---|---|---|
| 🟢 Green  | `≥ 0.9` (Clip IoU: `≥ 0.7`) | `≤ 2 s` |
| 🟠 Orange | `≥ 0.7` (Clip IoU: `≥ 0.5`) | `≤ 5 s` |
| 🔴 Red    | below the orange threshold   | above 5 s |

Counts (TP / FP / FN, GT / OCR, Goal clips, match duration) are always shown
in neutral dark gray.

---

## Per-goal table (bottom of the dashboard)

Same columns as Card 2, but **one row per GT goal** — so you can spot the
goal that's dragging down the mean. The `Status` column uses the **End-to-End
(Card 3)** classification:

| Status | Meaning | Row color |
|---|---|---|
| **TP** | Detected and `Clip IoU ≥ 0.5` | 🟢 green |
| **FN (clip)** | Detected but `Clip IoU < 0.5` | 🟡 yellow |
| **FN** | Not detected at all | 🔴 red |
| **FP** | Predicted, but no matching GT goal | 🟠 orange |
