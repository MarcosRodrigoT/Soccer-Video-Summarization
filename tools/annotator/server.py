#!/usr/bin/env python3
"""Local web annotator (goals of other leagues; shots and play/no-play).

    source venv/bin/activate
    python tools/annotator/server.py                 # http://127.0.0.1:8765
    python tools/annotator/server.py --port 9000 --data-root /tmp/ann

Listens on 127.0.0.1 only (reach it through an SSH tunnel). Annotations are
stored under --data-root (default: annotations_v2/ at the repo root).
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from catalog import REPO_ROOT, VIEWS, Catalog, UnknownMatch  # type: ignore
    from frames import FrameError, FrameService, compute_diff_curve  # type: ignore
    from store import Store, StoreError, atomic_write  # type: ignore
else:
    from .catalog import REPO_ROOT, VIEWS, Catalog, UnknownMatch
    from .frames import FrameError, FrameService, compute_diff_curve
    from .store import Store, StoreError, atomic_write

VERSION = "1.0"
STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_DATA_ROOT = REPO_ROOT / "annotations_v2"
MAX_BODY = 8 * 2**20
QUIET_PREFIXES = ("/api/frame/", "/api/scoreboard/")
CTYPES = {".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".html": "text/html; charset=utf-8",
          ".json": "application/json; charset=utf-8", ".svg": "image/svg+xml",
          ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon",
          ".woff2": "font/woff2", ".map": "application/json"}


class ApiError(Exception):
    def __init__(self, status: int, message: str, extra: dict | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.extra = extra


class App:
    """Everything the handler needs: catalogue, frames, store, diff cache."""

    def __init__(self, data_root: Path | str = DEFAULT_DATA_ROOT, repo_root: Path | str = REPO_ROOT):
        self.data_root = Path(data_root).resolve()
        self.store = Store(self.data_root, self._match_info)
        self.catalog = Catalog(repo_root, cache_dir=self.data_root / "cache")
        self.frames = FrameService(self.catalog)
        self._diff_locks: dict[str, threading.Lock] = {}
        self._diff_locks_lock = threading.Lock()

    def _match_info(self, match_id: str) -> dict:
        m = self.catalog.get(match_id)              # UnknownMatch is a KeyError
        meta = self.catalog.meta(match_id)
        return {"total_frames": meta["total_frames"], "fps": meta["fps"], "view": m.view,
                "views": [m.view, *m.extra_views]}

    def match(self, match_id: str):
        try:
            return self.catalog.get(match_id)
        except UnknownMatch:
            raise ApiError(404, f"Partido desconocido: {match_id}") from None

    def info(self, match_id: str, view: str | None = None) -> dict:
        self.match(match_id)
        try:
            return self.catalog.info(match_id, view)
        except FileNotFoundError as e:
            raise ApiError(404, str(e)) from None
        except OSError as e:
            raise ApiError(500, f"No se puede leer el vídeo del partido {match_id}: {e}") from None

    def diff_curve(self, match_id: str, f0: int, f1: int) -> dict:
        info = self.info(match_id)
        total, fps = info["total_frames"], info["fps"] or 25
        if not (0 <= f0 < f1 <= total):
            raise ApiError(400, f"Rango no válido: {f0}–{f1} (el vídeo tiene {total} fotogramas).")
        if f1 - f0 > round(30 * 60 * fps):
            raise ApiError(400, "El rango de la curva no puede superar 30 minutos.")
        path = self.data_root / "cache" / f"diff_{match_id}_{f0}_{f1}.npy"
        key = path.name
        with self._diff_locks_lock:
            lk = self._diff_locks.setdefault(key, threading.Lock())
        with lk:
            raw = None
            if path.is_file():
                try:
                    raw = np.load(path)
                    if len(raw) != f1 - f0:
                        raw = None
                except (OSError, ValueError):
                    raw = None
            if raw is None:
                raw = compute_diff_curve(self.catalog.video_path(match_id), f0, f1).astype(np.float32)

                buf = io.BytesIO()
                np.save(buf, raw)
                atomic_write(path, buf.getvalue())
        mx = float(raw.max()) if len(raw) else 0.0
        vals = raw / mx if mx > 0 else raw
        return {"start": f0, "end": f1, "values": [round(float(v), 4) for v in vals],
                "raw_max": round(mx, 6),
                "note": "values = |diferencia| media en grises a 160x90 entre fotogramas consecutivos, "
                        "normalizada por el máximo de la ventana (raw_max, en unidades 0..1)."}


def _int_param(q: dict, name: str, default=None, lo=None, hi=None) -> int | None:
    v = q.get(name, [None])[0]
    if v is None or v == "":
        return default
    try:
        x = int(v)
    except ValueError:
        raise ApiError(400, f"El parámetro «{name}» debe ser un número entero.") from None
    if (lo is not None and x < lo) or (hi is not None and x > hi):
        raise ApiError(400, f"El parámetro «{name}» debe estar entre {lo} y {hi}.")
    return x


def _float_param(q: dict, name: str, default: float, lo: float, hi: float) -> float:
    v = q.get(name, [None])[0]
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except ValueError:
        raise ApiError(400, f"El parámetro «{name}» debe ser un número.") from None
    if not (lo <= x <= hi):
        raise ApiError(400, f"El parámetro «{name}» debe estar entre {lo:g} y {hi:g}.")
    return x


def _frame_index(s: str) -> int:
    if not re.fullmatch(r"-?\d+", s):
        raise ApiError(400, f"Número de fotograma no válido: {s}")
    return int(s)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True   # headers and body go in separate writes: avoid 40 ms stalls
    server_version = "Anotador/" + VERSION
    app: App = None  # set by make_server
    log_requests = True

    # ---------------------------------------------------------------- output
    def log_message(self, fmt, *args):
        pass

    def _log(self, code: int, t0: float):
        if not self.log_requests:
            return
        path = urlparse(self.path).path
        if path.startswith(QUIET_PREFIXES) and code < 400:
            return
        sys.stderr.write(f"{time.strftime('%H:%M:%S')} {self.command} {self.path} {code} "
                         f"{(time.perf_counter() - t0) * 1000:.0f} ms\n")

    def _send(self, code: int, body, ctype: str = "application/json; charset=utf-8",
              headers: dict | None = None):
        if isinstance(body, (bytes, bytearray)):
            data = bytes(body)
        else:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        hdrs = {"Cache-Control": "no-store"}
        hdrs.update(headers or {})
        for k, v in hdrs.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        self._code = code

    def _error(self, code: int, message: str, extra: dict | None = None):
        body = dict(extra or {})
        body["error"] = message
        self._send(code, body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ApiError(413, "El documento es demasiado grande.")
        raw = self.rfile.read(n) if n else b""
        try:
            return json.loads(raw.decode("utf-8")) if raw else None
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "El cuerpo de la petición no es JSON válido.") from None

    # -------------------------------------------------------------- dispatch
    def _handle(self, method: str):
        t0 = time.perf_counter()
        self._code = 0
        try:
            try:
                self._route(method)
            except ApiError as e:
                self._error(e.status, e.message, e.extra)
            except StoreError as e:
                extra = None
                if e.status == 409 and e.current is not None:
                    extra = dict(e.current)
                    extra["current"] = e.current
                self._error(e.status, e.message, extra)
            except FrameError as e:
                self._error(e.status, e.message)
            except UnknownMatch as e:
                self._error(404, f"Partido desconocido: {e.args[0]}")
            except FileNotFoundError as e:
                self._error(404, str(e))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
            return
        except Exception as e:  # never let a bad request kill the server
            traceback.print_exc(file=sys.stderr)
            try:
                self._error(500, f"Error interno del servidor: {type(e).__name__}: {e}")
            except Exception:
                self.close_connection = True
        self._log(self._code or 500, t0)

    def do_GET(self):
        self._handle("GET")

    def do_HEAD(self):
        self._handle("GET")

    def do_PUT(self):
        self._handle("PUT")

    def do_POST(self):
        self._handle("POST")

    def do_DELETE(self):
        self._handle("DELETE")

    def _route(self, method: str):
        url = urlparse(self.path)
        path = unquote(url.path)
        q = parse_qs(url.query)
        parts = [p for p in path.split("/") if p]
        app = self.app

        if not parts or parts[0] != "api":
            if method != "GET":
                raise ApiError(405, "Método no permitido.")
            return self._static(path)

        api = parts[1:]
        if not api:
            raise ApiError(404, "Ruta no encontrada.")
        head = api[0]

        if head == "health" and method == "GET":
            return self._send(200, {"ok": True, "version": VERSION})

        if head == "matches" and method == "GET" and len(api) == 1:
            view = (q.get("view") or ["goals"])[0]
            if view not in VIEWS:
                raise ApiError(400, f"Vista desconocida «{view}» (debe ser «goals» o «sbd»).")
            out = []
            for mid in app.catalog.ids(view):
                try:
                    out.append(app.info(mid, view))
                except ApiError as e:
                    m = app.catalog.get(mid)
                    out.append({"id": mid, "label": m.label, "league": m.league, "season": m.season,
                                "home": m.home, "away": m.away, "score": m.score, "error": e.message})
            return self._send(200, out)

        if head == "match" and method == "GET" and len(api) == 2:
            info = app.info(api[1])
            info["hints"] = app.catalog.hints(api[1])
            return self._send(200, info)

        if head == "frame" and method == "GET" and len(api) == 3:
            mid, n = api[1], _frame_index(api[2])
            info = app.info(mid)
            if not (0 <= n < info["total_frames"]):
                raise ApiError(404, f"Fotograma fuera de rango: {n} (0 a {info['total_frames'] - 1}).")
            w = _int_param(q, "w", 960, 120, 1920)
            qual = _int_param(q, "q", None, 30, 100)
            data = app.frames.jpeg(mid, n, w, qual)
            return self._send(200, data, "image/jpeg",
                              {"X-Frame-Index": str(n), "Cache-Control": "max-age=86400",
                               "Access-Control-Expose-Headers": "X-Frame-Index"})

        if head == "scoreboard" and method == "GET" and len(api) == 3:
            mid, n = api[1], _frame_index(api[2])
            info = app.info(mid)
            if not (0 <= n < info["total_frames"]):
                raise ApiError(404, f"Fotograma fuera de rango: {n} (0 a {info['total_frames'] - 1}).")
            region = app.catalog.hints(mid)["scoreboard_region"]
            if region is None:
                raise ApiError(404, f"No hay región de marcador para el partido {mid}.")
            scale = _float_param(q, "scale", 2.0, 0.5, 4.0)
            data = app.frames.scoreboard_jpeg(mid, n, region, scale)
            return self._send(200, data, "image/jpeg",
                              {"X-Frame-Index": str(n), "Cache-Control": "max-age=86400",
                               "Access-Control-Expose-Headers": "X-Frame-Index"})

        if head == "diffcurve" and method == "GET" and len(api) == 2:
            mid = api[1]
            info = app.info(mid)
            f0 = _int_param(q, "start", 0)
            f1 = _int_param(q, "end", info["total_frames"])
            return self._send(200, app.diff_curve(mid, f0, f1))

        if head == "goals" and len(api) == 2:
            mid = api[1]
            app.match(mid)
            if method == "GET":
                return self._send(200, app.store.get_goals(mid))
            if method == "PUT":
                return self._send(200, app.store.put_goals(mid, self._body()))
            raise ApiError(405, "Método no permitido.")

        if head == "sbd":
            if len(api) == 2 and api[1] == "sessions":
                if method == "GET":
                    return self._send(200, app.store.list_sessions())
                if method == "POST":
                    body = self._body()
                    if isinstance(body, dict) and isinstance(body.get("match_id"), str):
                        app.match(body["match_id"])
                    doc, created = app.store.create_session(body)
                    return self._send(201 if created else 200, doc)
                raise ApiError(405, "Método no permitido.")
            if len(api) == 2:
                sid = api[1]
                if method == "GET":
                    return self._send(200, app.store.get_session(sid))
                if method == "PUT":
                    return self._send(200, app.store.put_session(sid, self._body()))
                if method == "DELETE":
                    return self._send(200, app.store.delete_session(sid))
                raise ApiError(405, "Método no permitido.")

        raise ApiError(404, "Ruta no encontrada.")

    def _static(self, path: str):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        root = STATIC.resolve()
        f = (root / rel).resolve()
        if root not in f.parents or not f.is_file():
            raise ApiError(404, "Fichero no encontrado.")
        ctype = CTYPES.get(f.suffix.lower()) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        self._send(200, f.read_bytes(), ctype, {"Cache-Control": "no-cache, no-store, must-revalidate"})


def make_server(port: int = 8765, data_root: Path | str = DEFAULT_DATA_ROOT, host: str = "127.0.0.1",
                log: bool = True):
    app = App(data_root)
    handler = type("BoundHandler", (Handler,), {"app": app, "log_requests": log})
    srv = ThreadingHTTPServer((host, port), handler)
    srv.daemon_threads = True
    srv.app = app
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description="Anotador web local (goles; planos y juego).")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT),
                    help="carpeta de las anotaciones (por defecto annotations_v2/ en la raíz del repo)")
    args = ap.parse_args(argv)
    srv = make_server(args.port, args.data_root)

    def warm():  # read the video metadata of every match once (cached on disk)
        for mid in srv.app.catalog.ids():
            try:
                srv.app.catalog.meta(mid)
            except Exception as e:
                print(f"Aviso: {mid}: {e}", file=sys.stderr)

    threading.Thread(target=warm, daemon=True).start()
    print(f"Anotador {VERSION} en http://127.0.0.1:{args.port}  (datos: {srv.app.data_root})", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        srv.app.frames.close()


if __name__ == "__main__":
    main()
