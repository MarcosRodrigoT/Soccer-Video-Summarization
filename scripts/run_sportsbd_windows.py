#!/usr/bin/env python3
"""SportSBD (sportsbd 0.1.2, released weights) on the annotated 5-min windows only.

Same defaults as sportsbd.inference.run_video_inference (16-frame windows,
stride aligned to frame 0, 112x112 PIL bilinear resize, Kinetics
normalisation, boundary at the centre of windows whose top class is a
transition with probability >= 0.7, NMS over 25 frames), but only the
annotated window plus 2 s of context on each side is decoded. Within the
window it gives the same detections as the full-match run of
run_sbd_baseline.py --method sportsbd.

Used to study where SportSBD places its boundaries
(scripts/annotations/sportsbd_window_eval.py). SPORTSBD_STRIDE=1 runs it
with stride 1; that run was stopped after 2 windows (Match10, Match11).

    python scripts/run_sportsbd_windows.py                     # stride 4 (default)
    SPORTSBD_STRIDE=1 python scripts/run_sportsbd_windows.py
Weights: $SPORTSBD_WEIGHTS (default data/models/sportsbd_best.pt).
Output: results_sbd/sportsbd/annotated_windows/stride<N>/<match>.json
"""
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from sportsbd.config import DEFAULT_CONFIG
from sportsbd.inference import _apply_nms
from sportsbd.model import load_model
from sportsbd.transforms import get_frame_transform

ROOT = Path(__file__).resolve().parents[1]
STRIDE = int(os.environ.get("SPORTSBD_STRIDE", 4))  # sportsbd default: 4
OUT = ROOT / "results_sbd" / "sportsbd" / "annotated_windows" / f"stride{STRIDE}"
T_FRAMES, THRESH, FPS, CTX, BATCH = 16, 0.7, 25, 50, 64


def video_of(match_id):
    cfg = yaml.safe_load(open(ROOT / "config/default_hr.yaml"))
    data = Path(cfg["data_dir"])
    laliga = yaml.safe_load(open(ROOT / "config/matches_hr.yaml"))["matches"]
    if match_id in laliga:
        return data / laliga[match_id]["video_path"]
    other = yaml.safe_load(open(ROOT / "config/matches_other_leagues.yaml"))["matches"]
    return next((data / other[match_id]["video_dir"]).glob("*.mp4"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(os.environ.get("SPORTSBD_WEIGHTS", str(ROOT / "data/models/sportsbd_best.pt")), device=dev)
    tf = get_frame_transform()
    for f in sorted((ROOT / "annotations_v2/sbd").glob("*.json")):
        d = json.load(open(f))
        mid, S, E = d["match_id"], d["start_frame"], d["end_frame"]
        out_f = OUT / f"{mid}.json"
        if out_f.exists():
            continue
        f0 = max(0, (S - CTX) // STRIDE * STRIDE)  # same window grid as a run from frame 0
        f1 = E + CTX
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(str(video_of(mid)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        frames = []
        for _ in range(f1 - f0):
            ok, im = cap.read()
            if not ok:
                break
            frames.append(tf(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))))
        bank = torch.stack(frames)
        starts = list(range(0, len(frames) - T_FRAMES + 1, STRIDE))
        dets = []
        with torch.no_grad():
            for i in range(0, len(starts), BATCH):
                b = torch.stack([bank[s:s + T_FRAMES] for s in starts[i:i + BATCH]]).permute(0, 2, 1, 3, 4).to(dev)
                probs = torch.softmax(model(b), dim=1).cpu().numpy()
                for s, p in zip(starts[i:i + BATCH], probs):
                    k = int(np.argmax(p))
                    if k < 3 and float(p.max()) >= THRESH:  # class 3 = no boundary
                        dets.append({"frame_idx": f0 + s + T_FRAMES // 2, "confidence": float(p.max()),
                                     "predicted_class": DEFAULT_CONFIG.class_names[k]})
        dets = _apply_nms(dets, nms_threshold_frames=FPS)
        el = time.perf_counter() - t0
        json.dump({"match_id": mid, "window": [S, E], "context_frames": [f0, f1], "stride": STRIDE,
                   "seconds": el, "detections": dets}, open(out_f, "w"), indent=1)
        print(mid, len(dets), "detections", f"{el:.0f}s", flush=True)


if __name__ == "__main__":
    main()
