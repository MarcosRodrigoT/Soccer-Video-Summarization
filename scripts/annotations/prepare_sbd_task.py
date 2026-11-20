#!/usr/bin/env python3
"""Prepare the shot-transition review task (reviewer comment R1.2).

Within a 10-minute segment of four matches (one per season), the transitions
proposed by any of the three detectors (ours, TransNetV2, PySceneDetect) are
pooled: transitions whose frame intervals overlap (after a +-TOL frames
dilation) become one candidate. The reviewer confirms whether each candidate
is a real shot transition (cut or gradual). The confirmed candidates form the
ground truth used by scripts/annotations/evaluate_sbd.py; transitions missed
by all three detectors are not represented (pooled annotation).

The suggestion shown is "transition" when at least two detectors agree; which
detectors fired is not shown.

    python scripts/annotations/prepare_sbd_task.py
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_single_match import load_match_info  # noqa: E402
from soccer_summarization.config import PipelineConfig  # noqa: E402

SEGMENTS = {  # match -> (start, end) in seconds of video time
    "Match1": (1500, 2100),   # 2013/14
    "Match4": (1500, 2100),   # 2018/19
    "Match3": (1500, 2100),   # 2020/21
    "Match6": (1500, 2100),   # 2023/24
}
METHODS = {"ours": "results_hr/per_match", "transnetv2": "results_sbd/transnetv2/per_match",
           "pyscenedetect": "results_sbd/pyscenedetect/per_match"}
TOL = 5  # frames
OPTIONS = [
    {"value": "transition", "label": "Real shot transition (cut, fade, wipe, logo)"},
    {"value": "none", "label": "No transition (same shot continues)"},
    {"value": "unsure", "label": "Unsure"},
]
INSTRUCTIONS = (
    "Each item is a <b>candidate shot transition</b>. The six frames go from before (left) to after (right); "
    "the middle two are the first and last frame of the proposed transition. Answer <b>transition</b> if the "
    "camera shot changes there (hard cut, dissolve, fade, wipe or logo transition, including transitions into "
    "or out of replays), and <b>none</b> if it is the same continuous shot (camera motion, zoom, flash, "
    "on-screen graphics appearing). The suggestion is a detector consensus, so please check every item."
)


def transitions(shots_csv):
    """Transition intervals [end_of_shot + 1, start_of_next - 1] (hard cut: a == b)."""
    s = pd.read_csv(shots_csv).sort_values("start_frame")
    out = []
    for (_, a), (_, b) in zip(s.iloc[:-1].iterrows(), s.iloc[1:].iterrows()):
        lo, hi = int(a["end_frame"]) + 1, int(b["start_frame"]) - 1
        out.append((min(lo, hi), max(lo, hi)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default_hr.yaml")
    ap.add_argument("--task", default="sbd_transitions")
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args()
    cfg = PipelineConfig.from_yaml(args.config)
    task_dir = Path("annotations") / args.task
    (task_dir / "img").mkdir(parents=True, exist_ok=True)
    items, detections = [], {}

    for match_id, (t0, t1) in SEGMENTS.items():
        info = load_match_info(match_id, cfg, matches_yaml=str(Path(args.config).parent / "matches_hr.yaml"))
        fps = 25
        f0, f1 = t0 * fps, t1 * fps
        per_method = {}
        for name, root in METHODS.items():
            tr = [t for t in transitions(Path(root) / match_id / "shot_boundaries.csv") if f0 <= t[0] <= f1]
            per_method[name] = tr
        detections[match_id] = {"segment_frames": [f0, f1], **per_method}
        # Pool overlapping intervals.
        pool = sorted((a, b, name) for name, tr in per_method.items() for a, b in tr)
        groups = []
        for a, b, name in pool:
            if groups and a - TOL <= groups[-1]["b"] + TOL:
                g = groups[-1]
                g["a"], g["b"] = min(g["a"], a), max(g["b"], b)
                g["methods"].add(name)
            else:
                groups.append({"a": a, "b": b, "methods": {name}})
        cap = cv2.VideoCapture(info["video_path"])
        for k, g in enumerate(groups):
            a, b = g["a"], g["b"]
            frames = [a - 12, a - 4, a, b, b + 4, b + 12]
            imgs = []
            for j, f in enumerate(frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, im = cap.read()
                if not ok:
                    continue
                h, w = im.shape[:2]
                im = cv2.resize(im, (args.width, int(h * args.width / w)), interpolation=cv2.INTER_AREA)
                name = f"img/{match_id}_c{k:04d}_{j}.jpg"
                cv2.imwrite(str(task_dir / name), im, [cv2.IMWRITE_JPEG_QUALITY, 80])
                imgs.append(name)
            mm, ss = divmod(a // fps, 60)
            kind = "hard cut" if b - a <= 1 else f"{b - a + 1}-frame interval"
            items.append({
                "id": f"{match_id}_c{k:04d}", "images": imgs,
                "image_labels": ["−12 fr", "−4 fr", "start", "end", "+4 fr", "+12 fr"],
                "caption": f"{match_id} · {mm}:{ss:02d} · proposed {kind}",
                "prelabel": "transition" if len(g["methods"]) >= 2 else None,
                "meta": {"match": match_id, "a": a, "b": b, "methods": sorted(g["methods"])},
            })
        cap.release()
        print(match_id, {k: len(v) for k, v in per_method.items()}, "->", len(groups), "candidates", flush=True)

    json.dump({"title": "R1.2 · Shot transitions (4 × 10 min)", "instructions": INSTRUCTIONS,
               "options": OPTIONS, "columns": 6, "items": items},
              open(task_dir / "items.json", "w"), indent=1)
    json.dump(detections, open(task_dir / "detections.json", "w"), indent=1)
    print("wrote", len(items), "items")


if __name__ == "__main__":
    main()
