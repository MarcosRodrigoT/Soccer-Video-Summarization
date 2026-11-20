#!/usr/bin/env python3
"""Local review app for the manual annotations of the revision.

Serves the tasks under annotations/<task>/ (items.json + images) and stores
the reviewer's decisions in annotations/<task>/labels.json after every key
press. Only the Python standard library is used.

    python tools/review_app/server.py            # http://localhost:8765
    python tools/review_app/server.py --port 9000

From VS Code (Remote-SSH), run it in the integrated terminal and open the
forwarded port when VS Code offers it (or Ports tab -> Forward 8765).
"""

import argparse
import json
import mimetypes
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "annotations"
STATIC = Path(__file__).resolve().parent / "static"
LOCK = threading.Lock()


def task_dirs():
    return sorted(p for p in ANN.iterdir() if (p / "items.json").exists())


def load_labels(task_dir):
    f = task_dir / "labels.json"
    return json.loads(f.read_text()) if f.exists() else {}


def save_labels(task_dir, labels):
    # Atomic write so a crash never leaves a truncated file.
    fd, tmp = tempfile.mkstemp(dir=task_dir, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(labels, f, indent=1, sort_keys=True)
    os.replace(tmp, task_dir / "labels.json")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep the terminal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path):
        if not path.is_file():
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def do_GET(self):
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        if not parts:
            return self._file(STATIC / "index.html")
        if parts[0] == "static":
            return self._file(STATIC.joinpath(*parts[1:]))
        if parts[:2] == ["api", "tasks"]:
            out = []
            for d in task_dirs():
                items = json.loads((d / "items.json").read_text())
                labels = load_labels(d)
                out.append({"name": d.name, "title": items.get("title", d.name),
                            "total": len(items["items"]), "done": len(labels)})
            return self._send(200, out)
        if parts[:2] == ["api", "task"] and len(parts) == 3:
            d = ANN / parts[2]
            items = json.loads((d / "items.json").read_text())
            items["labels"] = load_labels(d)
            return self._send(200, items)
        if parts[0] == "media" and len(parts) >= 3:
            p = (ANN / parts[1]).joinpath(*parts[2:]).resolve()
            if ANN.resolve() not in p.parents:
                return self._send(403, {"error": "forbidden"})
            return self._file(p)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/") if p]
        if parts[:2] != ["api", "label"] or len(parts) != 3:
            return self._send(404, {"error": "not found"})
        d = ANN / parts[2]
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with LOCK:
            labels = load_labels(d)
            if body.get("value") is None:
                labels.pop(body["id"], None)
            else:
                labels[body["id"]] = {"value": body["value"], "note": body.get("note", ""),
                                      "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            save_labels(d, labels)
        self._send(200, {"ok": True, "done": len(labels)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Review app running at http://localhost:{args.port}  (Ctrl+C to stop)")
    print("Tasks:", ", ".join(d.name for d in task_dirs()) or "none")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped. Labels are saved in annotations/<task>/labels.json")


if __name__ == "__main__":
    main()
