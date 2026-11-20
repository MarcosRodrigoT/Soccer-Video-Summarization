#!/usr/bin/env python3
"""Precision / recall / F1 of each shot boundary detector against the
reviewed transitions (reviewer comment R1.2).

Ground truth: candidates of annotations/sbd_transitions labelled
"transition" (pooled over the three detectors, see prepare_sbd_task.py).
A detected transition is a true positive if its frame interval overlaps a
ground-truth transition after a +-TOL frames dilation (each ground-truth
transition is matched at most once). Unreviewed candidates fall back to the
suggestion; candidates without suggestion and without review are excluded,
and the output is flagged as preliminary.

Writes annotations/sbd_transitions/results.json.
"""

import json
from pathlib import Path

TASK = Path("annotations/sbd_transitions")
TOL = 5


def overlap(a, b):
    return a[0] - TOL <= b[1] + TOL and b[0] - TOL <= a[1] + TOL


def main():
    items = json.load(open(TASK / "items.json"))["items"]
    det = json.load(open(TASK / "detections.json"))
    labels = json.load(open(TASK / "labels.json")) if (TASK / "labels.json").exists() else {}
    reviewed = sum(1 for it in items if it["id"] in labels)
    gt, unknown = {}, []
    for it in items:
        lab = labels.get(it["id"], {}).get("value", it["prelabel"])
        if lab is None or lab == "unsure":
            unknown.append(it)
            continue
        if lab == "transition":
            gt.setdefault(it["meta"]["match"], []).append((it["meta"]["a"], it["meta"]["b"]))
    unknown_spans = {}
    for it in unknown:
        unknown_spans.setdefault(it["meta"]["match"], []).append((it["meta"]["a"], it["meta"]["b"]))
    methods = [k for k in next(iter(det.values())) if k != "segment_frames"]
    out = {"preliminary": reviewed < len(items), "reviewed": reviewed, "total": len(items),
           "excluded_unknown": len(unknown), "gt_transitions": sum(len(v) for v in gt.values()), "methods": {}}
    for meth in methods:
        tp = fp = fn = 0
        for match, d in det.items():
            g = list(gt.get(match, []))
            used = [False] * len(g)
            for t in d[meth]:
                if any(overlap(t, u) for u in unknown_spans.get(match, [])):
                    continue  # undecided candidate: neither TP nor FP
                hit = next((i for i, u in enumerate(g) if not used[i] and overlap(t, u)), None)
                if hit is None:
                    fp += 1
                else:
                    used[hit] = True
                    tp += 1
            fn += used.count(False)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out["methods"][meth] = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r,
                                "f1": 2 * p * r / (p + r) if p + r else 0.0}
        print(f"{'PRELIM ' if out['preliminary'] else ''}{meth:14s} TP {tp:4d} FP {fp:4d} FN {fn:4d}  "
              f"P {p:.3f} R {r:.3f} F1 {out['methods'][meth]['f1']:.3f}")
    json.dump(out, open(TASK / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
