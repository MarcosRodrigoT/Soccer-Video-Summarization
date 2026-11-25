#!/usr/bin/env python3
"""Export the ground truth of the paper to dataset/ (public release).

    python scripts/export_dataset.py

dataset/
  matches.csv                       27 broadcasts: id, league, season, teams, score, YouTube id/URL
  goals/<id>_goals.csv              frame-level goal annotations (25 LaLiga + PL1, SA1):
                                    goal_id, start_frame, end_frame_shot, end_frame_scoreboard
  commentary/<id>_goal_flags.csv    commentary segments (start, end in s) with binary goal flags,
                                    without the transcript text (LaLiga matches)
  shots/<id>_shots.json             shot-level annotation of a 5-min window per broadcast:
                                    window, transitions (cut frame / gradual start-end), shots with
                                    play / no_play label
  shots/<id>_transitions.csv, shots/<id>_shots.csv   the same in CSV
Frame indices are 0-based decoded frames at 25 fps (OpenCV order).
"""

import csv
import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dataset"


def read_csv_any(path):
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def main():
    cfg = yaml.safe_load(open(ROOT / "config/default_hr.yaml"))
    data_dir = Path(cfg["data_dir"])
    laliga = yaml.safe_load(open(ROOT / "config/matches_hr.yaml"))["matches"]
    other = yaml.safe_load(open(ROOT / "config/matches_other_leagues.yaml"))["matches"]
    urls = read_csv_any(ROOT / "config/match_urls.csv").set_index("match_id")
    for sub in ("goals", "commentary", "shots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    ids = sorted(laliga, key=lambda m: int(m[5:])) + [m for m in other if other[m].get("annotate_goals")]
    for mid in ids:
        if mid in laliga:
            m = laliga[mid]
            yt = urls.loc[mid, "full_match_youtube_id"]
            rows.append({"match_id": mid, "league": "LaLiga", "season": m["season"], "home_team": m["home_team"],
                         "away_team": m["away_team"], "score": m["score"], "youtube_id": yt,
                         "url": f"https://www.youtube.com/watch?v={yt}"})
            g = read_csv_any(data_dir / m["goals_path"])
            c = read_csv_any(data_dir / m["gt_path"])
            c.columns = [x.strip() for x in c.columns]
            c[["start", "end", "GoalGT1", "GoalGT2"]].to_csv(OUT / "commentary" / f"{mid}_goal_flags.csv", index=False)
        else:
            m = other[mid]
            rows.append({"match_id": mid, "league": m["league"], "season": m["season"], "home_team": m["home_team"],
                         "away_team": m["away_team"], "score": m["score"], "youtube_id": m["youtube_id"],
                         "url": f"https://www.youtube.com/watch?v={m['youtube_id']}"})
            g = read_csv_any(ROOT / "annotations_v2/goals" / f"{mid}_goals.csv")
        g = g[["goal_id", "start_frame", "end_frame_shot", "end_frame_scoreboard"]]
        g.to_csv(OUT / "goals" / f"{mid}_goals.csv", index=False)

        sess = sorted((ROOT / "annotations_v2/sbd").glob(f"{mid}_*.json"))
        for f in sess:
            d = json.load(open(f))
            S, E = d["start_frame"], d["end_frame"]
            tr = sorted(d["transitions"], key=lambda t: t["frame"] if t["type"] == "cut" else t["start"])
            trs, starts, ends = [], [S], []
            for t in tr:
                if t["type"] == "cut":
                    trs.append({"type": "cut", "start_frame": t["frame"], "end_frame": t["frame"]})
                    ends.append(t["frame"] - 1); starts.append(t["frame"])
                else:
                    trs.append({"type": "gradual", "start_frame": t["start"], "end_frame": t["end"]})
                    ends.append(t["start"] - 1); starts.append(t["end"] + 1)
            ends.append(E - 1)
            shots = [{"start_frame": s, "end_frame": e, "label": d["shot_labels"][str(s)]}
                     for s, e in zip(starts, ends) if e >= s]
            doc = {"match_id": mid, "fps": 25, "window": {"start_frame": S, "end_frame_exclusive": E},
                   "note": "cut: start_frame is the first frame of the new shot; gradual: first and last "
                           "frame of the transition (inclusive). Shot labels: play = live action, "
                           "no_play = replays, close-ups, crowd, bench, graphics, stoppages.",
                   "transitions": trs, "shots": shots}
            json.dump(doc, open(OUT / "shots" / f"{mid}_shots.json", "w"), indent=1)
            pd.DataFrame(trs).to_csv(OUT / "shots" / f"{mid}_transitions.csv", index=False)
            pd.DataFrame(shots).to_csv(OUT / "shots" / f"{mid}_shots.csv", index=False)

    with open(OUT / "matches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"exported {len(rows)} matches to {OUT}")


if __name__ == "__main__":
    main()
