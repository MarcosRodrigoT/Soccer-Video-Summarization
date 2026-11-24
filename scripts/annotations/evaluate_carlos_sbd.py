#!/usr/bin/env python3
"""Evaluate shot boundary detectors and the play/no-play criterion against the
manual annotations made with tools/annotator (annotations_v2/sbd/*.json).

Ground truth per session (window [S, E)):
  * cut at frame f      -> new shot starts at f (transition interval [f, f])
  * gradual [s, e]      -> transition frames s..e
  * shot_labels         -> play / no_play per shot (keyed by first frame)

Detections: each detector's shot list (results_hr/per_match for ours,
results_sbd/<method>/per_match for TransNetV2 and PySceneDetect) converted to
transition intervals: between consecutive shots, [end_prev + 1, start_next]
(a hard cut gives [f, f]). Only detections whose interval intersects the
window are evaluated.

Matching (one-to-one, greedy by distance):
  * GT cut f matches a detection whose interval is within +-tol frames of f.
  * GT gradual [s, e] matches any detection overlapping [s - tol, e + tol]
    (agreed rule: any detection inside a gradual transition is a hit).
  Unmatched detections are false positives; with --dup-ok, extra detections
  that fall inside an already-matched gradual are ignored instead.

Play/no-play: for each GT shot, average people count of the pipeline
(results_hr/per_match/<m>/player_counts.csv, every 5th frame) over the shot;
predicted play if >= tau_play (5). Reported for all shots and for shots of at
least D_shot (4 s), the only ones the assembly stage classifies, plus a
frame-level agreement of the pipeline's own play long shots with GT play.

    python scripts/annotations/evaluate_carlos_sbd.py --out annotations_v2/results_sbd_play.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

METHODS = {"ours": "results_hr/per_match", "transnetv2": "results_sbd/transnetv2/per_match",
           "pyscenedetect": "results_sbd/pyscenedetect/per_match"}
OTHER_LEAGUES = ("PL1", "SA1")  # pipeline outputs live in results_other_leagues/per_match


def pipeline_dir(match_id):
    root = "results_other_leagues/per_match" if match_id in OTHER_LEAGUES else "results_hr/per_match"
    return Path(root) / match_id


def shots_csv(method, match_id):
    if method == "ours":
        return pipeline_dir(match_id) / "shot_boundaries.csv"
    return Path(METHODS[method]) / match_id / "shot_boundaries.csv"
FPS, TAU_PLAY, D_SHOT = 25, 5.0, 4.0


def gt_from_session(d):
    S, E = d["start_frame"], d["end_frame"]
    tr = sorted(d["transitions"], key=lambda t: t["frame"] if t["type"] == "cut" else t["start"])
    gts, starts, ends = [], [S], []
    for t in tr:
        if t["type"] == "cut":
            gts.append(("cut", t["frame"], t["frame"]))
            ends.append(t["frame"] - 1); starts.append(t["frame"])
        else:
            gts.append(("gradual", t["start"], t["end"]))
            ends.append(t["start"] - 1); starts.append(t["end"] + 1)
    ends.append(E - 1)
    shots = [(s, e, d["shot_labels"].get(str(s))) for s, e in zip(starts, ends)]
    return S, E, gts, shots


def detections(shots_csv, S, E):
    sh = pd.read_csv(shots_csv).sort_values("start_frame")
    out = []
    for (_, a), (_, b) in zip(sh.iloc[:-1].iterrows(), sh.iloc[1:].iterrows()):
        lo, hi = int(a["end_frame"]) + 1, int(b["start_frame"])
        if hi >= S and lo <= E - 1:
            out.append((lo, hi))
    return out, sh


def match(gts, dets, tol, dup_ok):
    pairs = []
    for gi, (kind, s, e) in enumerate(gts):
        for di, (a, b) in enumerate(dets):
            if kind == "cut":
                dist = 0 if a <= s <= b else min(abs(a - s), abs(b - s))
                if dist <= tol:
                    pairs.append((dist, gi, di))
            elif a <= e + tol and b >= s - tol:
                pairs.append((0, gi, di))
    pairs.sort()
    gm, dm = {}, {}
    for dist, gi, di in pairs:
        if gi not in gm and di not in dm:
            gm[gi] = di; dm[di] = gi
    fp_idx = [di for di in range(len(dets)) if di not in dm]
    if dup_ok:  # extra detections inside an already-matched gradual are not errors
        grads = [gts[gi] for gi in gm if gts[gi][0] == "gradual"]
        fp_idx = [di for di in fp_idx
                  if not any(dets[di][0] <= e + tol and dets[di][1] >= s - tol for _, s, e in grads)]
    tp = len(gm)
    res = {"tp": tp, "fp": len(fp_idx), "fn": len(gts) - tp}
    for kind in ("cut", "gradual"):
        n = sum(1 for g in gts if g[0] == kind)
        res[f"recall_{kind}"] = sum(1 for gi in gm if gts[gi][0] == kind) / n if n else None
        res[f"n_{kind}"] = n
    res["fp_frames"] = [dets[di] for di in fp_idx]
    res["fn_items"] = [gts[gi] for gi in range(len(gts)) if gi not in gm]
    return res


def prf(r):
    p = r["tp"] / (r["tp"] + r["fp"]) if r["tp"] + r["fp"] else 0.0
    rc = r["tp"] / (r["tp"] + r["fn"]) if r["tp"] + r["fn"] else 0.0
    return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="annotations_v2/sbd")
    ap.add_argument("--out", default="annotations_v2/results_sbd_play.json")
    args = ap.parse_args()
    sessions = [json.load(open(f)) for f in sorted(Path(args.sessions).glob("*.json"))]
    # Only sessions whose detector outputs all exist (TransNetV2 / PySceneDetect may still be running).
    sessions = [d for d in sessions if all(shots_csv(m, d["match_id"]).exists() for m in METHODS)]
    out = {"sessions": [d["session_id"] for d in sessions], "sbd": {}, "play": {}}

    for tol in (0, 1, 2, 5):
        for dup_ok in (False, True):
            key = f"tol{tol}{'_dupok' if dup_ok else ''}"
            out["sbd"][key] = {}
            for m, root in METHODS.items():
                tot = {"tp": 0, "fp": 0, "fn": 0, "cut_hit": 0, "n_cut": 0, "grad_hit": 0, "n_gradual": 0}
                per = {}
                for d in sessions:
                    S, E, gts, _ = gt_from_session(d)
                    dets, _ = detections(shots_csv(m, d["match_id"]), S, E)
                    r = match(gts, dets, tol, dup_ok)
                    per[d["match_id"]] = {k: r[k] for k in ("tp", "fp", "fn")} | {
                        "fp_frames": r["fp_frames"], "fn_items": r["fn_items"]}
                    for k in ("tp", "fp", "fn"):
                        tot[k] += r[k]
                    tot["cut_hit"] += round((r["recall_cut"] or 0) * r["n_cut"]); tot["n_cut"] += r["n_cut"]
                    tot["grad_hit"] += round((r["recall_gradual"] or 0) * r["n_gradual"]); tot["n_gradual"] += r["n_gradual"]
                p, rc, f1 = prf(tot)
                out["sbd"][key][m] = {**tot, "precision": p, "recall": rc, "f1": f1,
                                      "recall_cut": tot["cut_hit"] / tot["n_cut"],
                                      "recall_gradual": tot["grad_hit"] / tot["n_gradual"], "per_match": per}

    # Play / no-play
    allshots, frame_stats = [], {"agree": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for d in sessions:
        S, E, _, shots = gt_from_session(d)
        res = pipeline_dir(d["match_id"])
        pc = pd.read_csv(res / "player_counts.csv")
        for s, e, lab in shots:
            m = (pc["frame"] >= s) & (pc["frame"] <= e)
            avg = float(pc.loc[m, "player_count"].mean()) if m.any() else 0.0
            allshots.append({"match": d["match_id"], "start": s, "end": e, "dur": (e - s + 1) / FPS,
                             "label": lab, "avg": avg, "pred_play": avg >= TAU_PLAY})
        # frame-level: pipeline play long shots (its own shots) vs GT play frames
        hl = pd.read_csv(res / "highlights.csv")
        pipe = set()
        for _, h in hl.iterrows():
            pipe.update(range(max(S, int(h["start_frame"])), min(E, int(h["end_frame"]) + 1)))
        gtp = set()
        for s, e, lab in shots:
            if lab == "play":
                gtp.update(range(s, e + 1))
        for f in range(S, E):
            a, b = f in gtp, f in pipe
            frame_stats["total"] += 1
            frame_stats["agree"] += a == b
            frame_stats["tp" if a and b else "fp" if b else "fn" if a else "tn"] += 1

    def conf(rows):
        tp = sum(r["label"] == "play" and r["pred_play"] for r in rows)
        fp = sum(r["label"] == "no_play" and r["pred_play"] for r in rows)
        fn = sum(r["label"] == "play" and not r["pred_play"] for r in rows)
        tn = sum(r["label"] == "no_play" and not r["pred_play"] for r in rows)
        p = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": rc,
                "f1": 2 * p * rc / (p + rc) if p + rc else 0.0, "accuracy": (tp + tn) / len(rows)}

    out["play"]["all_shots"] = conf(allshots)
    out["play"]["long_shots"] = conf([r for r in allshots if r["dur"] >= D_SHOT])
    out["play"]["frame_level_pipeline_vs_gt"] = frame_stats | {
        "accuracy": frame_stats["agree"] / frame_stats["total"]}
    out["play"]["errors"] = [r for r in allshots if (r["label"] == "play") != r["pred_play"]]
    out["play"]["shots"] = allshots
    json.dump(out, open(args.out, "w"), indent=1)

    for key in ("tol2", "tol2_dupok", "tol0", "tol5"):
        print(f"--- SBD {key}")
        for m, r in out["sbd"][key].items():
            print(f"  {m:14s} TP {r['tp']:3d} FP {r['fp']:3d} FN {r['fn']:3d}  P {r['precision']:.3f} "
                  f"R {r['recall']:.3f} F1 {r['f1']:.3f}  R_cut {r['recall_cut']:.3f} R_grad {r['recall_gradual']:.3f}")
    for k in ("all_shots", "long_shots"):
        r = out["play"][k]
        print(f"--- play {k}: n {r['n']} TP {r['tp']} FP {r['fp']} FN {r['fn']} TN {r['tn']}  "
              f"P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f} acc {r['accuracy']:.3f}")
    fs = out["play"]["frame_level_pipeline_vs_gt"]
    print(f"--- frame-level pipeline play long shots vs GT play: acc {fs['accuracy']:.3f} "
          f"(TP {fs['tp']} FP {fs['fp']} FN {fs['fn']} TN {fs['tn']})")


if __name__ == "__main__":
    main()
