"""Match catalogue for the annotator: videos, metadata and pipeline hints.

Two views:

* ``goals``: matches of ``config/matches_other_leagues.yaml`` with
  ``annotate_goals: true`` (video = the single ``*.mp4`` in
  ``data_dir/video_dir``; hints in ``results_other_leagues/per_match/<id>/``).
* ``sbd``: the LaLiga matches of ``config/matches_hr.yaml``
  (video = ``data_dir/video_path``; hints in ``results_hr/per_match/<id>/``).

``data_dir`` comes from ``config/default_hr.yaml``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

VIEWS = ("goals", "sbd")


class UnknownMatch(KeyError):
    """Raised for a match id that is not in the catalogue."""


@dataclass
class Match:
    id: str
    view: str
    label: str
    league: str
    season: str
    home: str
    away: str
    score: str
    video_path: Path | None
    hints_dir: Path
    meta: dict = field(default_factory=dict)
    extra_views: tuple = ()          # other views where the match also appears
    view_labels: dict = field(default_factory=dict)  # label per view, if different

    def in_view(self, view: str) -> bool:
        return view == self.view or view in self.extra_views


def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f)]


def _to_int(v):
    if v is None or v == "":
        return None
    return int(round(float(v)))


class Catalog:
    def __init__(self, repo_root: Path | str = REPO_ROOT, cache_dir: Path | str | None = None):
        self.repo_root = Path(repo_root)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._lock = threading.Lock()
        self._meta_lock = threading.Lock()
        self._meta_disk: dict | None = None
        self._matches: dict[str, Match] = {}
        self._load()

    # ------------------------------------------------------------------ config
    def _load(self):
        root = self.repo_root
        with open(root / "config" / "default_hr.yaml", encoding="utf-8") as f:
            data_dir = Path(yaml.safe_load(f)["data_dir"])
        self.data_dir = data_dir

        matches: dict[str, Match] = {}

        with open(root / "config" / "matches_other_leagues.yaml", encoding="utf-8") as f:
            other = yaml.safe_load(f).get("matches", {}) or {}
        for mid, m in other.items():
            if not m.get("annotate_goals", False):
                continue
            vdir = data_dir / m["video_dir"]
            vids = sorted(vdir.glob("*.mp4")) if vdir.is_dir() else []
            video = vids[0] if len(vids) == 1 else None
            home, away = m.get("home_team", ""), m.get("away_team", "")
            score, season, league = str(m.get("score", "")), str(m.get("season", "")), m.get("league", "")
            label = f"{mid} · {home} {score} {away} ({league} {season})"
            # Also offered in the shot/play view, numbered after the LaLiga matches (sbd_number).
            extra, labels = (), {}
            if m.get("sbd_number"):
                extra = ("sbd",)
                labels["sbd"] = f"M{m['sbd_number']} · {home} {score} {away} ({league} {season})"
            matches[mid] = Match(mid, "goals", label, league, season, home, away, score, video,
                                 root / "results_other_leagues" / "per_match" / mid,
                                 extra_views=extra, view_labels=labels)

        with open(root / "config" / "matches_hr.yaml", encoding="utf-8") as f:
            laliga = yaml.safe_load(f).get("matches", {}) or {}
        for mid, m in laliga.items():
            home, away = m.get("home_team", ""), m.get("away_team", "")
            score, season = str(m.get("score", "")), str(m.get("season", ""))
            short = "M" + mid[len("Match"):] if mid.startswith("Match") else mid
            label = f"{short} · {home} {score} {away} ({season})"
            matches[mid] = Match(mid, "sbd", label, "LaLiga", season, home, away, score,
                                 data_dir / m["video_path"], root / "results_hr" / "per_match" / mid)

        self._matches = matches

    # ------------------------------------------------------------------ access
    def ids(self, view: str | None = None) -> list[str]:
        ms = [m for m in self._matches.values() if view is None or m.in_view(view)]
        # LaLiga first (Match1..Match25), then the other leagues by their sbd number.
        return [m.id for m in sorted(ms, key=lambda m: (m.view != "sbd", _natural_key(m.view_labels.get("sbd", m.id))))]

    def get(self, match_id: str) -> Match:
        try:
            return self._matches[match_id]
        except KeyError:
            raise UnknownMatch(match_id) from None

    def video_path(self, match_id: str) -> Path:
        m = self.get(match_id)
        if m.video_path is None or not m.video_path.is_file():
            raise FileNotFoundError(f"No se encuentra el vídeo del partido {match_id}.")
        return m.video_path

    # ---------------------------------------------------------------- metadata
    def _meta_cache_file(self) -> Path | None:
        return self.cache_dir / "video_meta.json" if self.cache_dir else None

    def _load_meta_disk(self) -> dict:
        if self._meta_disk is None:
            self._meta_disk = {}
            f = self._meta_cache_file()
            if f and f.is_file():
                try:
                    self._meta_disk = json.loads(f.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self._meta_disk = {}
        return self._meta_disk

    def _save_meta_disk(self):
        f = self._meta_cache_file()
        if not f:
            return
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f".{f.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(self._meta_disk, indent=1), encoding="utf-8")
            os.replace(tmp, f)
        except OSError:
            pass

    def meta(self, match_id: str) -> dict:
        """fps, total_frames, width, height of the match video (cached)."""
        m = self.get(match_id)
        if m.meta:
            return m.meta
        with self._meta_lock:
            if m.meta:
                return m.meta
            path = self.video_path(match_id)
            st = path.stat()
            key = f"{path}|{st.st_size}|{int(st.st_mtime)}"
            disk = self._load_meta_disk()
            if key in disk:
                m.meta = disk[key]
                return m.meta
            m.meta = probe_video(path)
            disk[key] = m.meta
            self._save_meta_disk()
            return m.meta

    def info(self, match_id: str, view: str | None = None) -> dict:
        m = self.get(match_id)
        meta = self.meta(match_id)
        return {
            "id": m.id, "label": m.view_labels.get(view, m.label), "view": m.view,
            "views": [m.view, *m.extra_views], "league": m.league, "season": m.season,
            "home": m.home, "away": m.away, "score": m.score,
            "fps": meta["fps"], "total_frames": meta["total_frames"],
            "duration_sec": round(meta["total_frames"] / meta["fps"], 3) if meta["fps"] else None,
            "width": meta["width"], "height": meta["height"],
        }

    def total_frames(self, match_id: str) -> int:
        return self.meta(match_id)["total_frames"]

    def fps(self, match_id: str) -> float:
        return self.meta(match_id)["fps"]

    # ------------------------------------------------------------------- hints
    def hints(self, match_id: str) -> dict:
        d = self.get(match_id).hints_dir
        region = None
        f = d / "scoreboard_region.json"
        if f.is_file():
            try:
                r = json.loads(f.read_text(encoding="utf-8"))
                region = {k: int(r[k]) for k in ("x", "y", "w", "h")}
            except (OSError, ValueError, KeyError, TypeError):
                region = None

        events = []
        for row in _read_csv(d / "score_events.csv"):
            try:
                events.append({"frame": _to_int(row["frame"]), "home": _to_int(row["home_score"]),
                               "away": _to_int(row["away_score"]), "team": row.get("team") or None})
            except (KeyError, ValueError, TypeError):
                continue
        events.sort(key=lambda e: e["frame"])

        starts = set()
        for row in _read_csv(d / "shot_boundaries.csv"):
            try:
                starts.add(_to_int(row["start_frame"]))
            except (KeyError, ValueError, TypeError):
                continue
        starts.discard(None)

        clips = []
        for row in _read_csv(d / "goal_highlights.csv"):
            try:
                clips.append({"start_frame": _to_int(row["start_frame"]), "end_frame": _to_int(row["end_frame"])})
            except (KeyError, ValueError, TypeError):
                continue
        clips.sort(key=lambda c: c["start_frame"])

        return {"scoreboard_region": region, "score_events": events,
                "shot_starts": sorted(starts), "goal_clips": clips}


def probe_video(path: Path) -> dict:
    """Read fps / frame count / size with OpenCV and check the last frame decodes."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"OpenCV no puede abrir el vídeo {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # The container's frame count can exceed what actually decodes: make
        # sure the last advertised frame exists (step back a few if not).
        total = n
        for back in range(0, 30):
            k = n - 1 - back
            if k < 0:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok = cap.grab()
            if ok:
                total = k + 1
                break
        return {"fps": round(fps, 6), "total_frames": total, "width": w, "height": h,
                "container_frames": n}
    finally:
        cap.release()
