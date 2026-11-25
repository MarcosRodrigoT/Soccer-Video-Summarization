# Ground truth

Annotations used in the paper "Broadcast-State-Guided Generation of Goal-Centered Soccer Highlights from Full-Length Match Videos". The broadcasts themselves are not redistributed; `matches.csv` gives the YouTube identifier of each one.

Frame indices are 0-based decoded frames at 25 fps, in the order returned by OpenCV.

| Path | Content |
|---|---|
| `matches.csv` | 27 broadcasts: 25 LaLiga matches (2013/14–2023/24), plus one Premier League match (PL1) and one Serie A match (SA1) |
| `goals/<id>_goals.csv` | One row per goal. `start_frame` and `end_frame_shot` are the first and last frames of the live shot in which the goal is scored. `end_frame_scoreboard` is the first frame in which the on-screen score shows the new result. |
| `commentary/<id>_goal_flags.csv` | Commentary segments (`start` and `end` in seconds), with binary goal flags for the home team (`GoalGT1`) and the away team (`GoalGT2`). Available for the LaLiga matches only. |
| `shots/<id>_shots.json` (and `_transitions.csv`, `_shots.csv`) | Shot-level annotation of a 5-min window of each broadcast (frames 30000–37499), described below |

The shot-level annotation of each window contains:
- every transition: for cuts, `start_frame` = `end_frame` = the first frame of the new shot; for gradual transitions, their first and last frames (inclusive);
- every resulting shot, labelled `play` (live action) or `no_play` (replays, close-ups, crowd, bench, graphics and stoppages).

The annotations may be freely reused for non-commercial research.
