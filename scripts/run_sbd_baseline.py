#!/usr/bin/env python3
"""Run an off-the-shelf shot boundary detector on full matches (reviewer R1.2).

Writes, per match, a shot_boundaries.csv in the pipeline's own format, so the
rest of the pipeline can be re-run with the alternative shot list:

    results_sbd/<method>/per_match/<match>/shot_boundaries.csv
    results_sbd/<method>/per_match/<match>/timing.json
    results_sbd/<method>/per_match/<match>/transnetv2_predictions.npy  (TransNetV2 only)

Methods (default settings of each tool):
  transnetv2     TransNetV2 (Soucek & Lokoc, 2020), PyTorch port with the
                 original weights; threshold 0.5 on the single-frame output.
  pyscenedetect  PySceneDetect ContentDetector (threshold 27, min. scene
                 length 15 frames, automatic downscaling).

Usage:
    python scripts/run_sbd_baseline.py --method transnetv2 --matches Match1 Match2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_single_match import load_match_info  # noqa: E402
from soccer_summarization.config import PipelineConfig  # noqa: E402
from soccer_summarization.utils import get_video_properties  # noqa: E402


def scenes_to_df(scenes, fps):
    """[(start, end_inclusive), ...] -> shot_boundaries.csv rows."""
    rows = []
    for i, (s, e) in enumerate(scenes, 1):
        rows.append({"shot_id": i, "start_frame": int(s), "end_frame": int(e),
                     "start_time": s / fps, "end_time": e / fps, "duration": (e - s) / fps})
    return pd.DataFrame(rows)


def run_transnetv2(video_path, out_dir):
    import torch
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    t0 = time.perf_counter()
    _frames, single, _many = model.predict_video(video_path, quiet=True)
    elapsed = time.perf_counter() - t0
    single = single.cpu().numpy()
    np.save(out_dir / "transnetv2_predictions.npy", single.astype(np.float16))
    scenes = TransNetV2.predictions_to_scenes(single, threshold=0.5)
    return [(int(s), int(e)) for s, e in scenes], elapsed, len(single)


def run_pyscenedetect(video_path, out_dir):
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(video_path, backend="opencv")
    manager = SceneManager()
    manager.auto_downscale = True
    manager.add_detector(ContentDetector())
    t0 = time.perf_counter()
    manager.detect_scenes(video, show_progress=False)
    elapsed = time.perf_counter() - t0
    scenes = manager.get_scene_list(start_in_scene=True)
    # PySceneDetect end frames are exclusive.
    out = [(s.get_frames(), e.get_frames() - 1) for s, e in scenes]
    return out, elapsed, int(video.frame_number)


METHODS = {"transnetv2": run_transnetv2, "pyscenedetect": run_pyscenedetect}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--config", default="config/default_hr.yaml")
    ap.add_argument("--out-root", default="results_sbd")
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    for match_id in args.matches:
        out_dir = Path(args.out_root) / args.method / "per_match" / match_id
        if (out_dir / "shot_boundaries.csv").exists():
            print(match_id, "already done", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        info = load_match_info(match_id, cfg, matches_yaml=str(Path(args.config).parent / "matches_hr.yaml"))
        props = get_video_properties(info["video_path"])
        scenes, elapsed, n_frames = METHODS[args.method](info["video_path"], out_dir)
        scenes_to_df(scenes, props["fps"]).to_csv(out_dir / "shot_boundaries.csv", index=False)
        json.dump({"method": args.method, "seconds": elapsed, "frames_processed": n_frames,
                   "frames_opencv": props["total_frames"], "num_shots": len(scenes)},
                  open(out_dir / "timing.json", "w"), indent=2)
        print(match_id, args.method, f"{len(scenes)} shots, {elapsed:.0f}s, "
              f"frames {n_frames} vs {props['total_frames']}", flush=True)


if __name__ == "__main__":
    main()
