#!/usr/bin/env python3
"""Evaluate the count-based play/no-play decision against the reviewed labels
(reviewer comment R1.3).

Ground truth: annotations/play_noplay/labels.json (human review). Items not
reviewed yet fall back to the suggested label and the output is flagged as
preliminary. "mixed" and "unsure" items are excluded.

Writes annotations/play_noplay/results.json.
"""

import collections
import json
from pathlib import Path

TASK = Path("annotations/play_noplay")


def prf(tp, fp, fn, tn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "accuracy": (tp + tn) / max(1, tp + fp + fn + tn)}


def main():
    items = json.load(open(TASK / "items.json"))["items"]
    pred = json.load(open(TASK / "pipeline_predictions.json"))
    labels = json.load(open(TASK / "labels.json")) if (TASK / "labels.json").exists() else {}
    reviewed = sum(1 for it in items if it["id"] in labels)
    per = collections.defaultdict(lambda: collections.Counter())
    errors = collections.Counter()
    classes = collections.Counter()
    excluded = 0
    for it in items:
        lab = labels.get(it["id"], {}).get("value", it["prelabel"])
        if lab in (None, "mixed", "unsure"):
            excluded += 1
            continue
        classes[lab] += 1
        gt = lab == "play"
        pp = pred[it["id"]]["pipeline_play"]
        key = ("tp" if pp else "fn") if gt else ("fp" if pp else "tn")
        per[it["meta"]["match"]][key] += 1
        per["all"][key] += 1
        if key == "fp":
            errors["fp_" + lab] += 1
    out = {"preliminary": reviewed < len(items), "reviewed": reviewed, "total": len(items),
           "excluded_mixed_unsure": excluded, "classes": dict(classes), "false_positive_classes": dict(errors)}
    for k, c in per.items():
        out[k] = prf(c["tp"], c["fp"], c["fn"], c["tn"])
    json.dump(out, open(TASK / "results.json", "w"), indent=2)
    a = out["all"]
    print(f"{'PRELIMINARY ' if out['preliminary'] else ''}reviewed {reviewed}/{len(items)}; "
          f"TP {a['tp']} FP {a['fp']} FN {a['fn']} TN {a['tn']}  P {a['precision']:.3f} R {a['recall']:.3f} "
          f"F1 {a['f1']:.3f} acc {a['accuracy']:.3f}; FP classes {dict(errors)}")


if __name__ == "__main__":
    main()
