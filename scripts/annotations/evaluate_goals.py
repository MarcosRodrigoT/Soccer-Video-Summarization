#!/usr/bin/env python3
"""Summarize the cross-league review (reviewer comment R1.5).

Per match: goals in the match (from its final score), detected events that
are real goals, false events, and real goals whose clip shows the live play
ending in the goal. Unreviewed items fall back to the suggested label and the
output is flagged as preliminary.

Writes annotations/other_leagues_goals/results.json.
"""

import json
from pathlib import Path

import yaml

TASK = Path("annotations/other_leagues_goals")


def main():
    items = json.load(open(TASK / "items.json"))["items"]
    labels = json.load(open(TASK / "labels.json")) if (TASK / "labels.json").exists() else {}
    meta = yaml.safe_load(open("config/matches_other_leagues.yaml"))["matches"]
    out = {"preliminary": any(it["id"] not in labels for it in items), "reviewed": len(labels),
           "total": len(items), "matches": {}}
    for m, info in meta.items():
        goals = sum(int(x) for x in str(info["score"]).split("-"))
        its = [it for it in items if it["meta"]["match"] == m]
        lab = [labels.get(it["id"], {}).get("value", it["prelabel"]) for it in its]
        real = sum(v in ("goal_clip_ok", "goal_clip_wrong") for v in lab)
        out["matches"][m] = {
            "league": info["league"], "season": info["season"], "score": info["score"], "goals": goals,
            "events": len(its), "real_goals_detected": real,
            "false_events": sum(v == "not_goal" for v in lab),
            "clips_ok": sum(v == "goal_clip_ok" for v in lab),
            "unsure": sum(v in (None, "unsure") for v in lab),
            "scoreboard_found": (Path("results_other_leagues/per_match") / m / "scoreboard_region.json").exists(),
        }
        r = out["matches"][m]
        print(f"{'PRELIM ' if out['preliminary'] else ''}{m} {info['league']:15s} goals {goals}  detected "
              f"{real}  false {r['false_events']}  clips OK {r['clips_ok']}  scoreboard {r['scoreboard_found']}")
    json.dump(out, open(TASK / "results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
