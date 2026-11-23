"""Persistence of the annotations (goals documents and shot/play sessions).

Layout under ``data_root``::

    goals/<match_id>.json          goals document (atomic writes)
    goals/<match_id>_goals.csv     export of the complete goals
    sbd/<session_id>.json          shot boundary / play session
    history/goals_<id>.jsonl       every saved document, one JSON per line
    history/sbd_<session_id>.jsonl (append-only, never truncated)
    trash/                         deleted sessions (moved, never removed)

Every save checks ``revision`` (409 on mismatch), validates the document
(400 with a Spanish message), appends it to the history and replaces the
JSON file atomically (tmp + fsync + os.replace + directory fsync).
Unknown keys are preserved; keys starting with ``_`` are dropped.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

SCHEMA = 1
MAX_GOALS = 200
MAX_TRANSITIONS = 20_000
MAX_NOTES = 10_000
SESSION_RE = re.compile(r"^([A-Za-z0-9]+)_(\d+)_(\d+)$")
MATCH_RE = re.compile(r"^[A-Za-z0-9]+$")
CSV_HEADER = ["goal_id", "start_frame", "end_frame_shot", "end_frame_scoreboard", "notes"]
LABELS = ("play", "no_play")


class StoreError(Exception):
    def __init__(self, status: int, message: str, current: dict | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.current = current


def _invalid(msg: str):
    raise StoreError(400, msg)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _fsync_dir(d: Path):
    try:
        fd = os.open(str(d), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write(path: Path, data: bytes):
    """Write ``data`` to ``path`` atomically: a crash leaves old or new content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def append_line(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _frame_or_none(v, name: str, total: int, where: str):
    if v is None:
        return None
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if not _is_int(v):
        _invalid(f"{where}: «{name}» debe ser un número de fotograma entero o null.")
    if not (0 <= v < total):
        _invalid(f"{where}: «{name}» = {v} está fuera del vídeo (0 a {total - 1}).")
    return v


def _strip_private(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if not str(k).startswith("_")}


class Store:
    """``match_info(match_id) -> {"total_frames": int, "fps": float, "view": str}``
    (raises KeyError for an unknown match)."""

    def __init__(self, root: Path | str, match_info):
        self.root = Path(root)
        self.match_info = match_info
        for d in ("goals", "sbd", "history", "trash", "cache"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.RLock] = {}
        self._locks_lock = threading.Lock()

    def lock(self, name: str) -> threading.RLock:
        with self._locks_lock:
            lk = self._locks.get(name)
            if lk is None:
                lk = self._locks[name] = threading.RLock()
            return lk

    # ----------------------------------------------------------------- helpers
    def _info(self, match_id: str) -> dict:
        if not isinstance(match_id, str) or not MATCH_RE.match(match_id):
            raise StoreError(404, f"Partido desconocido: {match_id}")
        try:
            return self.match_info(match_id)
        except KeyError:
            raise StoreError(404, f"Partido desconocido: {match_id}") from None
        except FileNotFoundError as e:
            raise StoreError(404, str(e)) from None

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            # Never overwrite a file we cannot read: refuse instead.
            raise StoreError(500, f"No se puede leer {path.name} ({e}). No se ha modificado nada; "
                                  f"revisa el historial en history/.") from None

    def _save(self, path: Path, history: Path, doc: dict):
        data = json.dumps(doc, ensure_ascii=False, indent=1).encode("utf-8")
        atomic_write(path, data)
        append_line(history, doc)          # one line per successful save

    def _check_revision(self, body: dict, current: dict | None, initial: dict):
        rev = body.get("revision")
        if not _is_int(rev):
            _invalid("El documento no tiene un número de revisión («revision») válido.")
        stored = current["revision"] if current else 0
        if rev != stored:
            cur = current if current is not None else initial
            raise StoreError(409, "El documento ha cambiado en otra pestaña o ventana "
                                  f"(revisión guardada {stored}, recibida {rev}). Recarga para continuar.",
                             current=cur)
        return stored

    # =================================================================== goals
    def goals_path(self, match_id: str) -> Path:
        return self.root / "goals" / f"{match_id}.json"

    def goals_csv_path(self, match_id: str) -> Path:
        return self.root / "goals" / f"{match_id}_goals.csv"

    def initial_goals(self, match_id: str) -> dict:
        return {"match_id": match_id, "schema": SCHEMA, "updated": None, "revision": 0,
                "complete": False, "goals": []}

    def get_goals(self, match_id: str) -> dict:
        self._info(match_id)
        with self.lock(f"goals:{match_id}"):
            doc = self._read_json(self.goals_path(match_id))
        return doc if doc is not None else self.initial_goals(match_id)

    def validate_goals(self, match_id: str, body, total: int) -> dict:
        if not isinstance(body, dict):
            _invalid("El documento de goles debe ser un objeto JSON.")
        doc = _strip_private(copy.deepcopy(body))
        if doc.get("match_id", match_id) != match_id:
            _invalid(f"El documento es del partido «{doc.get('match_id')}», no de «{match_id}».")
        if doc.get("schema", SCHEMA) != SCHEMA:
            _invalid(f"Versión de formato no soportada: {doc.get('schema')}.")
        complete = doc.get("complete", False)
        if not isinstance(complete, bool):
            _invalid("«complete» debe ser verdadero o falso.")
        goals = doc.get("goals", [])
        if not isinstance(goals, list):
            _invalid("«goals» debe ser una lista.")
        if len(goals) > MAX_GOALS:
            _invalid(f"Demasiados goles ({len(goals)}).")
        seen = set()
        clean = []
        for i, g in enumerate(goals, 1):
            where = f"Gol {i}"
            if not isinstance(g, dict):
                _invalid(f"{where}: debe ser un objeto.")
            g = dict(g)
            gid = g.get("id")
            if not isinstance(gid, str) or not gid.strip():
                _invalid(f"{where}: falta el identificador («id»).")
            if gid in seen:
                _invalid(f"{where}: identificador repetido «{gid}».")
            seen.add(gid)
            for k in ("start_frame", "end_frame_shot", "end_frame_scoreboard"):
                g[k] = _frame_or_none(g.get(k), k, total, where)
            s, e = g["start_frame"], g["end_frame_shot"]
            if s is not None and e is not None and s > e:
                _invalid(f"{where}: el fin del plano ({e}) es anterior al inicio del plano ({s}).")
            notes = g.get("notes", "")
            if notes is None:
                notes = ""
            if not isinstance(notes, str):
                _invalid(f"{where}: las notas deben ser texto.")
            if len(notes) > MAX_NOTES:
                _invalid(f"{where}: las notas son demasiado largas.")
            g["notes"] = notes
            skipped = g.get("skipped", False)
            if not isinstance(skipped, bool):
                _invalid(f"{where}: «skipped» debe ser verdadero o falso.")
            g["skipped"] = skipped
            clean.append(g)
        doc.update({"match_id": match_id, "schema": SCHEMA, "complete": complete, "goals": clean})
        return doc

    @staticmethod
    def goal_is_complete(g: dict) -> bool:
        return (not g.get("skipped")) and all(
            g.get(k) is not None for k in ("start_frame", "end_frame_shot", "end_frame_scoreboard"))

    def goals_csv(self, doc: dict) -> str:
        goals = sorted((g for g in doc["goals"] if self.goal_is_complete(g)), key=lambda g: g["start_frame"])
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(CSV_HEADER)
        for i, g in enumerate(goals, 1):
            w.writerow([i, g["start_frame"], g["end_frame_shot"], g["end_frame_scoreboard"], g.get("notes", "")])
        return buf.getvalue()

    def put_goals(self, match_id: str, body) -> dict:
        info = self._info(match_id)
        with self.lock(f"goals:{match_id}"):
            current = self._read_json(self.goals_path(match_id))
            if not isinstance(body, dict):
                _invalid("El documento de goles debe ser un objeto JSON.")
            stored = self._check_revision(body, current, self.initial_goals(match_id))
            doc = self.validate_goals(match_id, body, info["total_frames"])
            doc["revision"] = stored + 1
            doc["updated"] = now_iso()
            self._save(self.goals_path(match_id), self.root / "history" / f"goals_{match_id}.jsonl", doc)
            atomic_write(self.goals_csv_path(match_id), self.goals_csv(doc).encode("utf-8"))
            return doc

    # ===================================================================== sbd
    def sbd_path(self, session_id: str) -> Path:
        return self.root / "sbd" / f"{session_id}.json"

    def _parse_session_id(self, session_id: str):
        m = SESSION_RE.match(session_id or "")
        if not m:
            raise StoreError(404, f"Identificador de sesión no válido: {session_id}")
        return m.group(1), int(m.group(2)), int(m.group(3))

    @staticmethod
    def shot_starts(doc: dict) -> list[int]:
        """First frame of every (non-empty) shot of a validated session."""
        S, E = doc["start_frame"], doc["end_frame"]
        trs = sorted(doc["transitions"], key=_tr_order)
        starts = []
        cur = S
        for t in trs:
            first = _tr_first(t)
            if first > cur:
                starts.append(cur)
            cur = t["frame"] if t["type"] == "cut" else t["end"] + 1
        if cur < E:
            starts.append(cur)
        return starts

    def session_summary(self, doc: dict) -> dict:
        starts = self.shot_starts(doc)
        labels = doc.get("shot_labels", {})
        return {"session_id": doc["session_id"], "match_id": doc["match_id"],
                "start_frame": doc["start_frame"], "end_frame": doc["end_frame"],
                "n_transitions": len(doc["transitions"]), "n_shots": len(starts),
                "n_labelled": sum(1 for s in starts if str(s) in labels),
                "complete": bool(doc.get("complete")), "updated": doc.get("updated"),
                "revision": doc.get("revision")}

    def list_sessions(self) -> list[dict]:
        out = []
        for p in (self.root / "sbd").glob("*.json"):
            if not SESSION_RE.match(p.stem):
                continue
            try:
                with self.lock(f"sbd:{p.stem}"):
                    doc = json.loads(p.read_text(encoding="utf-8"))
                out.append(self.session_summary(doc))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        out.sort(key=lambda s: s.get("updated") or "", reverse=True)
        return out

    def create_session(self, body) -> tuple[dict, bool]:
        if not isinstance(body, dict):
            _invalid("La petición debe ser un objeto JSON con match_id, start_frame y end_frame.")
        mid = body.get("match_id")
        info = self._info(mid)
        if info.get("view") not in (None, "sbd"):
            _invalid(f"El partido {mid} no pertenece a la vista de planos.")
        S, E = body.get("start_frame"), body.get("end_frame")
        if not _is_int(S) or not _is_int(E):
            _invalid("El inicio y el fin de la ventana deben ser números de fotograma enteros.")
        total, fps = info["total_frames"], info.get("fps") or 25
        if S < 0 or E > total:
            _invalid(f"La ventana {S}–{E} se sale del vídeo (0 a {total}).")
        if E - S < round(fps):
            _invalid("La ventana debe durar al menos 1 segundo.")
        if E - S > round(60 * 60 * fps):
            _invalid("La ventana no puede durar más de 60 minutos.")
        sid = f"{mid}_{S}_{E}"
        with self.lock(f"sbd:{sid}"):
            cur = self._read_json(self.sbd_path(sid))
            if cur is not None:
                return cur, False
            doc = {"session_id": sid, "match_id": mid, "start_frame": S, "end_frame": E,
                   "schema": SCHEMA, "updated": now_iso(), "revision": 1, "complete": False,
                   "transitions": [], "shot_labels": {}}
            self._save(self.sbd_path(sid), self.root / "history" / f"sbd_{sid}.jsonl", doc)
            return doc, True

    def get_session(self, session_id: str) -> dict:
        self._parse_session_id(session_id)
        with self.lock(f"sbd:{session_id}"):
            doc = self._read_json(self.sbd_path(session_id))
        if doc is None:
            raise StoreError(404, f"La sesión {session_id} no existe.")
        return doc

    def validate_session(self, current: dict, body) -> tuple[dict, list[str]]:
        if not isinstance(body, dict):
            _invalid("El documento de la sesión debe ser un objeto JSON.")
        doc = _strip_private(copy.deepcopy(body))
        for k in ("session_id", "match_id", "start_frame", "end_frame"):
            if k in doc and doc[k] != current[k]:
                _invalid(f"No se puede cambiar «{k}» de una sesión existente "
                         f"({current[k]} → {doc[k]}); crea otra sesión.")
            doc[k] = current[k]
        if doc.get("schema", SCHEMA) != SCHEMA:
            _invalid(f"Versión de formato no soportada: {doc.get('schema')}.")
        doc["schema"] = SCHEMA
        S, E = current["start_frame"], current["end_frame"]
        complete = doc.get("complete", False)
        if not isinstance(complete, bool):
            _invalid("«complete» debe ser verdadero o falso.")
        trs = doc.get("transitions", [])
        if not isinstance(trs, list):
            _invalid("«transitions» debe ser una lista.")
        if len(trs) > MAX_TRANSITIONS:
            _invalid("Demasiadas transiciones.")
        seen = set()
        clean = []
        for i, t in enumerate(trs, 1):
            if not isinstance(t, dict):
                _invalid(f"Transición {i}: debe ser un objeto.")
            t = dict(t)
            tid = t.get("id")
            if not isinstance(tid, str) or not tid.strip():
                _invalid(f"Transición {i}: falta el identificador («id»).")
            if tid in seen:
                _invalid(f"Transición {i}: identificador repetido «{tid}».")
            seen.add(tid)
            typ = t.get("type")
            for k in ("frame", "start", "end"):
                if isinstance(t.get(k), float) and t[k].is_integer():
                    t[k] = int(t[k])
            if typ == "cut":
                f = t.get("frame")
                if not _is_int(f):
                    _invalid(f"Corte {tid}: falta el fotograma del corte («frame»).")
                if not (S < f < E):
                    _invalid(f"Corte {tid} en el fotograma {f}: debe estar dentro de la ventana "
                             f"({S + 1} a {E - 1}).")
                t.pop("start", None)
                t.pop("end", None)
            elif typ == "gradual":
                s, e = t.get("start"), t.get("end")
                if not _is_int(s) or not _is_int(e):
                    _invalid(f"Transición gradual {tid}: faltan el inicio y/o el fin "
                             f"(una gradual abierta no se puede guardar).")
                if s > e:
                    _invalid(f"Transición gradual {tid}: el fin ({e}) es anterior al inicio ({s}).")
                if not (S <= s and e < E):
                    _invalid(f"Transición gradual {tid} ({s}–{e}): debe estar dentro de la ventana "
                             f"({S} a {E - 1}).")
                t.pop("frame", None)
            else:
                _invalid(f"Transición {tid}: tipo desconocido «{typ}» (debe ser «cut» o «gradual»).")
            clean.append(t)
        # No overlaps: every shot between two consecutive transitions must be non-empty.
        order = sorted(clean, key=_tr_order)
        # (sorted by first frame, it is enough to check consecutive pairs).
        for a, b in zip(order, order[1:]):
            next_start = a["frame"] if a["type"] == "cut" else a["end"] + 1
            fb = _tr_first(b)
            if fb < next_start:
                _invalid(f"Las transiciones {_tr_desc(a)} y {_tr_desc(b)} se solapan.")
            if fb == next_start and a["type"] == b["type"] == "cut":
                _invalid(f"Hay dos cortes en el mismo fotograma ({fb}).")
            if fb == next_start:
                _invalid(f"Las transiciones {_tr_desc(a)} y {_tr_desc(b)} están pegadas: entre ellas "
                         f"debe quedar al menos un fotograma de plano.")
        doc["transitions"] = clean

        labels = doc.get("shot_labels", {})
        if labels is None:
            labels = {}
        if not isinstance(labels, dict):
            _invalid("«shot_labels» debe ser un objeto {fotograma: etiqueta}.")
        for k, v in labels.items():
            if not (isinstance(k, str) and k.isdigit()):
                _invalid(f"Etiqueta de plano con clave no válida «{k}» (debe ser el primer fotograma del plano).")
            if v not in LABELS:
                _invalid(f"Etiqueta no válida «{v}» para el plano {k} (debe ser «play» o «no_play»).")
        doc["shot_labels"] = labels
        starts = self.shot_starts(doc)
        valid = {str(s) for s in starts}
        pruned = sorted((k for k in labels if k not in valid), key=int)
        doc["shot_labels"] = {str(s): labels[str(s)] for s in starts if str(s) in labels}
        if complete:
            missing = len(starts) - len(doc["shot_labels"])
            if missing:
                _invalid(f"No se puede marcar la sesión como terminada: faltan {missing} "
                         f"plano(s) por etiquetar.")
        doc["complete"] = complete
        return doc, pruned

    def put_session(self, session_id: str, body) -> dict:
        self._parse_session_id(session_id)
        with self.lock(f"sbd:{session_id}"):
            current = self._read_json(self.sbd_path(session_id))
            if current is None:
                raise StoreError(404, f"La sesión {session_id} no existe (¿se eliminó?).")
            if not isinstance(body, dict):
                _invalid("El documento de la sesión debe ser un objeto JSON.")
            stored = self._check_revision(body, current, current)
            doc, pruned = self.validate_session(current, body)
            doc["revision"] = stored + 1
            doc["updated"] = now_iso()
            self._save(self.sbd_path(session_id), self.root / "history" / f"sbd_{session_id}.jsonl", doc)
            out = dict(doc)
            out["_pruned_labels"] = pruned
            return out

    def delete_session(self, session_id: str) -> dict:
        self._parse_session_id(session_id)
        with self.lock(f"sbd:{session_id}"):
            p = self.sbd_path(session_id)
            doc = self._read_json(p)
            if doc is None:
                raise StoreError(404, f"La sesión {session_id} no existe.")
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            dest = self.root / "trash" / f"{session_id}.{stamp}.json"
            hist = dict(doc)
            hist["_deleted_at"] = now_iso()
            hist["_trash"] = str(dest.relative_to(self.root))
            append_line(self.root / "history" / f"sbd_{session_id}.jsonl", hist)
            os.replace(p, dest)
            _fsync_dir(p.parent)
            _fsync_dir(dest.parent)
            return {"ok": True, "session_id": session_id, "trash": str(dest.relative_to(self.root))}


def _tr_first(t: dict) -> int:
    return t["frame"] if t["type"] == "cut" else t["start"]


def _tr_order(t: dict):
    return (_tr_first(t), 0 if t["type"] == "gradual" else 1)


def _tr_desc(t: dict) -> str:
    if t["type"] == "cut":
        return f"«corte en {t['frame']}»"
    return f"«gradual {t['start']}–{t['end']}»"
