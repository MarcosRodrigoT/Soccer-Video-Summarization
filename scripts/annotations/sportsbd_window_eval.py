#!/usr/bin/env python3
"""Where SportSBD places its boundaries, on the annotated windows.

Reads results_sbd/sportsbd/annotated_windows/stride<N>/*.json
(scripts/run_sportsbd_windows.py) and writes
results_sbd/sportsbd/annotated_windows/window_eval.json with
  * offset_cuts: histogram of (closest detection - annotated cut frame) for
    every annotated cut with a detection within 6 frames, per stride;
  * by_tolerance: P / R / F1 with the matching of evaluate_carlos_sbd.py
    (gradual hits ignored when duplicated) for tol = 0, 1, 2, 3, 5 (stride 4).
With stride 4 the detections lie 0-3 frames before the cut (4-frame grid);
with stride 1 they still lie 1-3 frames before it, because the NMS keeps the
first of several windows with saturated confidence.

    python scripts/annotations/sportsbd_window_eval.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_carlos_sbd import gt_from_session, match, prf  # noqa: E402

W = ROOT / "results_sbd" / "sportsbd" / "annotated_windows"


def load(stride):
    out = {}
    for f in sorted((ROOT / "annotations_v2/sbd").glob("*.json")):
        d = json.load(open(f))
        p = W / f"stride{stride}" / f"{d['match_id']}.json"
        if p.exists():
            out[d["match_id"]] = (d, [x["frame_idx"] for x in json.load(open(p))["detections"]])
    return out


def offsets(runs):
    c = Counter()
    for d, det in runs.values():
        _, _, gts, _ = gt_from_session(d)
        for kind, s, _e in gts:
            if kind == "cut" and det:
                b = min(det, key=lambda x: abs(x - s))
                if abs(b - s) <= 6:
                    c[b - s] += 1
    return {str(k): c[k] for k in sorted(c)}


def main():
    res = {"offset_cuts": {}, "by_tolerance": {}}
    for stride in (4, 1):
        runs = load(stride)
        res["offset_cuts"][f"stride{stride}"] = {"windows": sorted(runs), "offset_counts": offsets(runs)}
    runs = load(4)
    for tol in (0, 1, 2, 3, 5):
        a = {"tp": 0, "fp": 0, "fn": 0, "cut_hit": 0, "n_cut": 0, "grad_hit": 0, "n_gradual": 0}
        for d, det in runs.values():
            S, E, gts, _ = gt_from_session(d)
            r = match(gts, [(x, x) for x in sorted(det) if S <= x <= E - 1], tol, True)
            for k in ("tp", "fp", "fn", "n_cut", "n_gradual"):
                a[k] += r[k]
            a["cut_hit"] += round((r["recall_cut"] or 0) * r["n_cut"])
            a["grad_hit"] += round((r["recall_gradual"] or 0) * r["n_gradual"])
        p, rc, f1 = prf(a)
        res["by_tolerance"][f"tol{tol}_dupok"] = a | {"precision": p, "recall": rc, "f1": f1,
                                                      "recall_cut": a["cut_hit"] / a["n_cut"],
                                                      "recall_gradual": a["grad_hit"] / a["n_gradual"]}
    json.dump(res, open(W / "window_eval.json", "w"), indent=1)
    for k, v in res["offset_cuts"].items():
        print(k, len(v["windows"]), "windows, offsets", v["offset_counts"])
    for k, v in res["by_tolerance"].items():
        print(f"{k:11s} P {v['precision']:.3f} R {v['recall']:.3f} F1 {v['f1']:.3f} "
              f"Rcut {v['recall_cut']:.3f} Rgrad {v['recall_gradual']:.3f} TP {v['tp']} FP {v['fp']} FN {v['fn']}")


if __name__ == "__main__":
    main()
