"""Figure 16 — Scoreboard localization across LaLiga broadcasters and seasons.

Three example matches spanning eleven seasons of LaLiga, each rendered with
the auto-localized scoreboard bounding box (from scoreboard_region.json) in
red. The same IQR-based localization pipeline, with the same parameters,
recovers very different overlays: a long thin top scoreboard in early
seasons and a narrower, taller boxy scoreboard in the modern LALIGA EA
SPORTS era. The figure is meant to be referenced from the Scoreboard
localization section of the paper as evidence of broadcaster-/season-agnostic
operation.

Coordinates in scoreboard_region.json are in 480p (the resolution at which
the IQR map is computed); the broadcast videos are 1080p, so the bboxes are
scaled by 1080/480 = 2.25 before overlaying.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from _style import save_figure, set_paper_style

PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results_hr" / "per_match"
VIDEOS_ROOT = Path("/mnt/Data-fast/mrt/lf/SoccerNetClip10Videos/Laliga_HR")

# bbox coordinates in scoreboard_region.json are already in full 1080p
# resolution (same as the IQR map), so no rescaling is needed.
COORD_SCALE = 1.0
# Crop the top fraction of each 1080p frame so the figure stays compact while
# still showing some context around the scoreboard. Must be tall enough to
# cover the bottom edge of every selected bbox.
CROP_TOP_FRACTION = 0.28

# Three matches chosen to span the three visually distinct LaLiga scoreboard
# layouts present in the corpus: 2013/14 (thin top-left scoreboard with lower
# y position), 2020/21 (thin top scoreboard, classic layout), 2023/24 (modern
# narrower-and-taller LALIGA EA SPORTS scoreboard).
MATCHES = [
    {
        "match": "Match1",
        "video": "Valencia CF - FC Barcelona (2-3) LALIGA 2013⧸" "2014 FULL MATCH "
                 "[Qzrg3QDmQ4k].mp4",
        "label": "LaLiga 2013/14 — Valencia CF vs. FC Barcelona",
        "frame_idx": 50000,
    },
    {
        "match": "Match5",
        "video": "FC Barcelona 1 - 3 Real Madrid ｜ PARTIDO COMPLETO ｜ "
                 "LALIGA EA SPORTS 2020⧸" "21 [J6HGKSJi448].mp4",
        "label": "LaLiga 2020/21 — FC Barcelona vs. Real Madrid",
        "frame_idx": 50000,
    },
    {
        "match": "Match10",
        "video": "FC Barcelona 5-0 Real Betis ｜ PARTIDO COMPLETO ｜ "
                 "LALIGA EA SPORTS 2023⧸" "24 [JaDHm8QNKMg].mp4",
        "label": "LaLiga 2023/24 — FC Barcelona vs. Real Betis",
        "frame_idx": 50000,
    },
]


def grab_frame(video_path: Path, frame_idx: int):
    if not video_path.exists():
        return None
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_bbox(match_dir: Path):
    with open(match_dir / "scoreboard_region.json") as f:
        b = json.load(f)
    return {
        "x": int(b["x"] * COORD_SCALE),
        "y": int(b["y"] * COORD_SCALE),
        "w": int(b["w"] * COORD_SCALE),
        "h": int(b["h"] * COORD_SCALE),
    }


def build_figure():
    set_paper_style()
    fig, axes = plt.subplots(len(MATCHES), 1, figsize=(7.2, 5.4),
                             gridspec_kw={"hspace": 0.55})

    for ax, spec in zip(axes, MATCHES):
        match_dir = RESULTS / spec["match"]
        video_path = VIDEOS_ROOT / spec["match"] / spec["video"]
        bbox = load_bbox(match_dir)
        frame = grab_frame(video_path, spec["frame_idx"])

        if frame is None:
            ax.text(0.5, 0.5, f"frame unavailable: {spec['match']}",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        h_full = frame.shape[0]
        # Ensure the crop is tall enough to include the entire bbox + margin
        crop_h = max(int(h_full * CROP_TOP_FRACTION),
                     bbox["y"] + bbox["h"] + 40)
        crop = frame[:crop_h]

        ax.imshow(crop)
        ax.add_patch(Rectangle(
            (bbox["x"], bbox["y"]), bbox["w"], bbox["h"],
            fill=False, edgecolor="#E53935", linewidth=2.0,
        ))
        ax.set_title(
            f"{spec['label']}   |   bbox = ({bbox['x']},{bbox['y']}) "
            f"· {bbox['w']}×{bbox['h']} px",
            fontsize=9, loc="left", pad=2.0,
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        "Automatic scoreboard localization across LaLiga broadcasters and seasons",
        fontsize=11, fontweight="bold", y=0.995,
    )
    return fig


if __name__ == "__main__":
    fig = build_figure()
    save_figure(fig, "scoreboard_diversity")
