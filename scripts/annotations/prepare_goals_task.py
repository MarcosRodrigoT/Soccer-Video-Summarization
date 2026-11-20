#!/usr/bin/env python3
"""Prepare the review of the cross-league results (reviewer comment R1.5).

One item per goal event detected by the pipeline on the cross-league
matches. The reviewer sees the localized scoreboard before and after the
detected score change, and six frames of the produced 10-s clip (plus one
frame 2 s after it), and decides whether the event is a real goal and
whether the clip shows the live play that ends in that goal.

Goals missed by the pipeline are counted from the final score of each match
(scripts/annotations/evaluate_goals.py).

    python scripts/annotations/prepare_goals_task.py --matches PL1 SA1 BL1 L1
"""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
import yaml

OPTIONS = [
    {"value": "goal_clip_ok", "label": "Real goal — clip shows the live play ending in the goal"},
    {"value": "goal_clip_wrong", "label": "Real goal — clip shows something else (replay, other play...)"},
    {"value": "not_goal", "label": "Not a goal (score did not really change, or goal disallowed)"},
    {"value": "unsure", "label": "Unsure"},
]
INSTRUCTIONS = (
    "Each item is a <b>goal detected</b> by the pipeline on a match from another league. The first two images "
    "show the scoreboard found by the pipeline a few seconds <b>before</b> and <b>after</b> the detected score "
    "change. The next images are frames of the produced <b>10-s goal clip</b> (start → end) and one frame 2 s "
    "after it. Answer <b>clip OK</b> if the clip is the live wide-angle play that ends with the goal (the same "
    "kind of clip as in the LaLiga evaluation), <b>clip wrong</b> if the goal is real but the clip shows a replay, "
    "a celebration or another play, and <b>not a goal</b> if the score change is spurious."
)


def grab(cap, frame, width=None, crop=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame)))
    ok, im = cap.read()
    if not ok:
        return None
    if crop:
        x, y, w, h = crop
        pad = 20
        im = im[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        im = cv2.resize(im, (im.shape[1] * 2, im.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    elif width:
        h, w = im.shape[:2]
        im = cv2.resize(im, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", default=["PL1", "SA1", "BL1", "L1"])
    ap.add_argument("--matches-yaml", default="config/matches_other_leagues.yaml")
    ap.add_argument("--data-dir", default="/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos")
    ap.add_argument("--results", default="results_other_leagues/per_match")
    ap.add_argument("--task", default="other_leagues_goals")
    args = ap.parse_args()
    meta_all = yaml.safe_load(open(args.matches_yaml))["matches"]
    task_dir = Path("annotations") / args.task
    (task_dir / "img").mkdir(parents=True, exist_ok=True)
    items = []
    for m in args.matches:
        meta = meta_all[m]
        res = Path(args.results) / m
        video = next((Path(args.data_dir) / meta["video_dir"]).glob("*.mp4"))
        try:
            ev = pd.read_csv(res / "score_events.csv")
        except pd.errors.EmptyDataError:  # no scoreboard found: no events to review
            ev = pd.DataFrame()
        try:
            clips = pd.read_csv(res / "goal_highlights.csv")
        except (FileNotFoundError, pd.errors.EmptyDataError):
            clips = pd.DataFrame()
        region = json.load(open(res / "scoreboard_region.json")) if (res / "scoreboard_region.json").exists() else None
        crop = (region["x"], region["y"], region["w"], region["h"]) if region else None
        cap = cv2.VideoCapture(str(video))
        prev = (0, 0)
        for k, e in ev.reset_index(drop=True).iterrows():
            fps = 25
            f = int(e["frame"])
            imgs, labels = [], []
            for tag, ff in (("scoreboard −6 s", f - 6 * fps), ("scoreboard +6 s", f + 6 * fps)):
                im = grab(cap, ff, crop=crop) if crop else grab(cap, ff, width=640)
                if im is not None:
                    name = f"img/{m}_g{k:02d}_{len(imgs)}.jpg"
                    cv2.imwrite(str(task_dir / name), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    imgs.append(name); labels.append(tag)
            if k < len(clips):
                c = clips.iloc[k]
                s0, s1 = int(c["start_frame"]), int(c["end_frame"])
                for j, ff in enumerate([s0 + (s1 - s0) * q / 5 for q in range(6)] + [s1 + 2 * fps]):
                    im = grab(cap, ff, width=480)
                    if im is not None:
                        name = f"img/{m}_g{k:02d}_{len(imgs)}.jpg"
                        cv2.imwrite(str(task_dir / name), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
                        imgs.append(name)
                        labels.append("clip end +2 s" if j == 6 else f"clip {j * 2} s")
                clip_txt = f"clip {c['start_time']:.1f}–{c['end_time']:.1f} s"
            else:
                clip_txt = "no clip"
            mm, ss = divmod(int(e["timestamp"]), 60)
            new = (int(e["home_score"]), int(e["away_score"]))
            items.append({
                "id": f"{m}_g{k:02d}", "images": imgs, "image_labels": labels, "columns": 4,
                "caption": f"{meta['league']} · {meta['home_team']} {meta['score']} {meta['away_team']} · "
                           f"detected {prev[0]}-{prev[1]} → {new[0]}-{new[1]} at {mm}:{ss:02d}",
                "info": clip_txt, "prelabel": None,
                "meta": {"match": m, "event": int(k), "frame": f, "score": list(new)},
            })
            prev = new
        cap.release()
        print(m, len(ev), "events", flush=True)
    json.dump({"title": "R1.5 · Goals on other leagues", "instructions": INSTRUCTIONS,
               "options": OPTIONS, "items": items}, open(task_dir / "items.json", "w"), indent=1)
    print("wrote", len(items), "items")


if __name__ == "__main__":
    main()
