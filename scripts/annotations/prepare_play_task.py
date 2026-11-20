#!/usr/bin/env python3
"""Prepare the play/no-play review task (reviewer comment R1.3).

Items are the long shots (>= D_shot = 4 s, the only ones the assembly stage
classifies) detected by the pipeline in the selected matches. Each item shows
three frames (at 20%, 50% and 80% of the shot). The pipeline's own decision
(average people count >= tau_play) is written to a separate file and is NOT
shown to the reviewer, so the labels are independent of it.

    python scripts/annotations/prepare_play_task.py --matches Match1 Match6
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

OPTIONS = [
    {"value": "play", "label": "Live play (wide-angle game action)"},
    {"value": "replay", "label": "Replay (any framing)"},
    {"value": "other", "label": "Other (close-up, crowd, bench, graphics...)"},
    {"value": "mixed", "label": "Mixed (shot contains a missed cut)"},
    {"value": "unsure", "label": "Unsure"},
]
INSTRUCTIONS = (
    "Each item is one shot detected by the pipeline (only shots of 4 s or more). "
    "Decide what the shot shows: <b>live play</b> is wide-angle footage of the match as it happens; "
    "<b>replays</b> (often with a logo transition and without the scoreboard) are not live play even if wide; "
    "<b>other</b> covers close-ups, crowd, bench, coaches and graphics. Use <b>mixed</b> if the three frames "
    "clearly belong to different shots. The suggestion comes from a visual pre-labelling, not from the pipeline."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", default=["Match1", "Match6"])
    ap.add_argument("--config", default="config/default_hr.yaml")
    ap.add_argument("--task", default="play_noplay")
    ap.add_argument("--width", type=int, default=640)
    args = ap.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    min_dur = cfg.play_classifier.min_shot_duration
    thr = cfg.play_classifier.play_threshold
    task_dir = Path("annotations") / args.task
    (task_dir / "img").mkdir(parents=True, exist_ok=True)

    items, pipeline = [], {}
    for match_id in args.matches:
        info = load_match_info(match_id, cfg, matches_yaml=str(Path(args.config).parent / "matches_hr.yaml"))
        res = Path(cfg.results_dir) / "per_match" / match_id
        shots = pd.read_csv(res / "shot_boundaries.csv")
        counts = pd.read_csv(res / "player_counts.csv")
        cap = cv2.VideoCapture(info["video_path"])
        for _, s in shots.iterrows():
            if s["duration"] < min_dur:
                continue
            sid = f"{match_id}_shot{int(s['shot_id']):04d}"
            m = (counts["frame"] >= s["start_frame"]) & (counts["frame"] <= s["end_frame"])
            avg = float(counts.loc[m, "player_count"].mean()) if m.any() else 0.0
            pipeline[sid] = {"avg_people": avg, "pipeline_play": avg >= thr}
            imgs = []
            for k, frac in enumerate((0.2, 0.5, 0.8)):
                f = int(s["start_frame"] + frac * (s["end_frame"] - s["start_frame"]))
                cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (args.width, int(h * args.width / w)), interpolation=cv2.INTER_AREA)
                name = f"img/{sid}_{k}.jpg"
                cv2.imwrite(str(task_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
                imgs.append(name)
            mm, ss = divmod(int(s["start_time"]), 60)
            items.append({
                "id": sid, "images": imgs, "image_labels": ["20%", "50%", "80%"],
                "caption": f"{match_id} · shot {int(s['shot_id'])} · starts {mm}:{ss:02d} · {s['duration']:.1f} s",
                "prelabel": None,
                "meta": {"match": match_id, "shot_id": int(s["shot_id"]), "start_frame": int(s["start_frame"]),
                         "end_frame": int(s["end_frame"]), "duration": float(s["duration"])},
            })
        cap.release()
        print(match_id, sum(1 for i in items if i["meta"]["match"] == match_id), "long shots", flush=True)

    json.dump({"title": "R1.3 · Play / no-play of long shots", "instructions": INSTRUCTIONS,
               "options": OPTIONS, "columns": 3, "items": items},
              open(task_dir / "items.json", "w"), indent=1)
    json.dump(pipeline, open(task_dir / "pipeline_predictions.json", "w"), indent=1)
    print("wrote", task_dir / "items.json", len(items), "items")


if __name__ == "__main__":
    main()
