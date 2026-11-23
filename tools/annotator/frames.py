"""Frame service: exact OpenCV frame indexing with capture pools and caches.

Frame ``n`` is the n-th frame that ``cv2.VideoCapture.read()`` returns when
reading the video sequentially from the start (0-based), i.e. the convention
of the pipeline and the existing ground truth.

Each video keeps a small pool of captures with their current position:

* ``n == pos``            -> ``read()`` (sequential fast path)
* ``0 < n - pos <= 50``   -> ``grab()`` forward, then ``read()``
* otherwise               -> seek with ``CAP_PROP_POS_FRAMES`` (verified to be
  frame-exact on these files, see tests/test_backend.py), then ``read()``.

When the request is a short step backwards from a recently served frame
(the annotator pressing <-), the reader seeks ~1 s earlier and decodes
forward, caching every frame so the next backward steps are cache hits.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

GRAB_LIMIT = 50          # forward jumps up to this many frames use grab()
BACKFILL = 24            # frames decoded before n on a short backward step
BACKFILL_AHEAD = 8       # ... and after n (up to the recent frame), for filmstrips
BACK_WINDOW = 60         # "short backward step": n within this of a recent frame
POOL_SIZE = 4            # captures per video
MAX_OPEN_VIDEOS = 4      # readers kept open (LRU)
JPEG_BUDGET = 300 * 2**20
DECODED_BUDGET = 1024 * 2**20
SCOREBOARD_MARGIN = 12


class FrameError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class LRUBytes:
    """Thread-safe LRU keyed cache with a total size budget in bytes."""

    def __init__(self, budget: int):
        self.budget = budget
        self.size = 0
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    @staticmethod
    def _nbytes(v) -> int:
        return v.nbytes if isinstance(v, np.ndarray) else len(v)

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return v

    def put(self, key, value):
        nb = self._nbytes(value)
        if nb > self.budget:
            return
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self.size -= self._nbytes(old)
            self._d[key] = value
            self.size += nb
            while self.size > self.budget and self._d:
                _, v = self._d.popitem(last=False)
                self.size -= self._nbytes(v)

    def clear(self):
        with self._lock:
            self._d.clear()
            self.size = 0


class _Capture:
    """One cv2.VideoCapture with its position (index of the next frame read)."""

    def __init__(self, path: Path):
        self.path = path
        self.cap = None
        self.pos: int | None = None
        self.lock = threading.Lock()
        self.last_used = 0.0

    def _open(self):
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise FrameError(500, f"No se puede abrir el vídeo {self.path.name}")
        self.cap = cap
        self.pos = 0

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = None
        self.pos = None

    def cost(self, n: int) -> int:
        if self.cap is None or self.pos is None:
            return 10_000
        d = n - self.pos
        return d if 0 <= d <= GRAB_LIMIT else 1_000

    def _position_at(self, n: int):
        """Leave the capture so that the next read() returns frame n."""
        if self.cap is None:
            self._open()
        d = n - self.pos if self.pos is not None else -1
        if 0 <= d <= GRAB_LIMIT:
            for _ in range(d):
                if not self.cap.grab():
                    self.pos = None
                    raise FrameError(404, f"No se pudo decodificar el fotograma {n}")
                self.pos += 1
        else:
            self.pos = None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            self.pos = n

    def read_range(self, a: int, b: int) -> list[np.ndarray]:
        """Decode frames a..b (inclusive). Caller holds self.lock."""
        self.last_used = time.monotonic()
        try:
            self._position_at(a)
            out = []
            for k in range(a, b + 1):
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.pos = None
                    raise FrameError(404, f"No se pudo decodificar el fotograma {k}")
                self.pos = k + 1
                out.append(frame)
            return out
        except FrameError:
            raise
        except Exception as e:  # OpenCV error: drop the capture, reopen next time
            self.close()
            raise FrameError(500, f"Error al decodificar el vídeo: {e}") from e


class VideoReader:
    """Pool of captures for one video.

    ``cache_get(n)`` / ``cache_put(n, frame)`` (optional) connect the reader to
    the decoded-frame cache, so that frames decoded by a backfill are visible to
    the requests that waited for it.
    """

    def __init__(self, path: Path, total_frames: int, pool_size: int = POOL_SIZE,
                 cache_get=None, cache_put=None):
        self.path = Path(path)
        self.total_frames = total_frames
        self.caps = [_Capture(self.path) for _ in range(pool_size)]
        self._pool_lock = threading.Lock()
        self._recent: deque = deque(maxlen=16)
        self._pending: list[tuple[int, int, threading.Event]] = []
        self.cache_get = cache_get
        self.cache_put = cache_put
        self.stats = {"read": 0, "grab": 0, "seek": 0, "backfill": 0, "waited": 0}

    def close(self):
        for c in self.caps:
            with c.lock:
                c.close()

    def _acquire(self, n: int) -> _Capture:
        with self._pool_lock:
            order = sorted(self.caps, key=lambda c: (c.cost(n), c.last_used))
        for c in order:
            if c.lock.acquire(blocking=False):
                return c
        c = order[0]
        c.lock.acquire()
        return c

    def _plan(self, n: int, allow_backfill: bool):
        """Called with the pool lock held: ("single",) or ("range", a, b)."""
        if not allow_backfill or n == 0:
            return ("single",)
        if any(c.cost(n) <= GRAB_LIMIT for c in self.caps):
            return ("single",)
        later = [m for m in self._recent if n < m <= n + BACK_WINDOW]
        if not later:
            return ("single",)
        a = max(0, n - BACKFILL)
        b = min(n + BACKFILL_AHEAD, min(later) - 1, self.total_frames - 1)
        return ("range", a, b)

    def read(self, n: int, allow_backfill: bool = True) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        """Return (frame n, extra decoded frames {k: frame}) — extras may be cached."""
        if not (0 <= n < self.total_frames):
            raise FrameError(404, f"Fotograma fuera de rango: {n} (0..{self.total_frames - 1})")
        while True:
            with self._pool_lock:
                ev = next((e for a, b, e in self._pending if a <= n <= b), None)
                if ev is None:
                    plan = self._plan(n, allow_backfill)
                    if plan[0] == "range":
                        mine = (plan[1], plan[2], threading.Event())
                        self._pending.append(mine)
                    break
            # another request is decoding a range that contains n: wait for it
            self.stats["waited"] += 1
            ev.wait(timeout=30)
            if self.cache_get is not None:
                f = self.cache_get(n)
                if f is not None:
                    return f, {}

        extras: dict[int, np.ndarray] = {}
        if plan[0] == "range":
            a, b, ev = mine
            try:
                c = self._acquire(a)
                try:
                    self.stats["backfill"] += 1
                    frames = c.read_range(a, b)
                finally:
                    c.lock.release()
                frame = frames[n - a]
                extras = {a + i: f for i, f in enumerate(frames) if a + i != n}
                if self.cache_put is not None:
                    for k, f in extras.items():
                        self.cache_put(k, f)
                    self.cache_put(n, frame)
            finally:
                with self._pool_lock:
                    self._pending.remove(mine)
                ev.set()
        else:
            c = self._acquire(n)
            try:
                cost = c.cost(n)
                self.stats["read" if cost == 0 else "grab" if cost <= GRAB_LIMIT else "seek"] += 1
                frame = c.read_range(n, n)[0]
            finally:
                c.lock.release()
        with self._pool_lock:
            self._recent.append(n)
        return frame, extras

    def read_direct_seek(self, n: int) -> np.ndarray:
        """Independent strategy for verification: fresh capture, seek, read."""
        cap = cv2.VideoCapture(str(self.path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok, f = cap.read()
            if not ok:
                raise FrameError(404, f"No se pudo decodificar el fotograma {n}")
            return f
        finally:
            cap.release()


def resize_to_width(frame: np.ndarray, w: int) -> np.ndarray:
    h0, w0 = frame.shape[:2]
    if w == w0:
        return frame
    h = max(1, int(round(h0 * w / w0)))
    interp = cv2.INTER_AREA if w < w0 else cv2.INTER_CUBIC
    return cv2.resize(frame, (w, h), interpolation=interp)


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise FrameError(500, "No se pudo codificar la imagen JPEG")
    return buf.tobytes()


class FrameService:
    """Frames of the catalogue's matches, as arrays or JPEGs, with caches."""

    def __init__(self, catalog, jpeg_budget: int = JPEG_BUDGET, decoded_budget: int = DECODED_BUDGET,
                 max_open_videos: int = MAX_OPEN_VIDEOS, pool_size: int = POOL_SIZE):
        self.catalog = catalog
        self.jpegs = LRUBytes(jpeg_budget)
        self.decoded = LRUBytes(decoded_budget)
        self.max_open_videos = max_open_videos
        self.pool_size = pool_size
        self._readers: OrderedDict[str, VideoReader] = OrderedDict()
        self._lock = threading.Lock()
        self._inflight: dict = {}
        self._inflight_lock = threading.Lock()

    def reader(self, match_id: str) -> VideoReader:
        with self._lock:
            r = self._readers.get(match_id)
            if r is not None:
                self._readers.move_to_end(match_id)
                return r
        path = self.catalog.video_path(match_id)       # may raise
        total = self.catalog.total_frames(match_id)
        evicted = []
        with self._lock:
            r = self._readers.get(match_id)
            if r is None:
                r = VideoReader(path, total, self.pool_size,
                                cache_get=lambda k, m=match_id: self.decoded.get((m, k)),
                                cache_put=lambda k, f, m=match_id: self._put_decoded(m, k, f))
                self._readers[match_id] = r
                while len(self._readers) > self.max_open_videos:
                    evicted.append(self._readers.popitem(last=False)[1])
            self._readers.move_to_end(match_id)
        for old in evicted:
            old.close()
        return r

    def close(self):
        with self._lock:
            readers = list(self._readers.values())
            self._readers.clear()
        for r in readers:
            r.close()

    # ------------------------------------------------------------ raw frames
    def _put_decoded(self, match_id: str, n: int, f: np.ndarray):
        f.setflags(write=False)
        self.decoded.put((match_id, n), f)

    def frame(self, match_id: str, n: int) -> np.ndarray:
        """Decoded BGR frame n (full resolution). Do not modify the result."""
        key = (match_id, n)
        f = self.decoded.get(key)
        if f is not None:
            return f
        # Single flight: concurrent requests for the same frame decode it once.
        with self._inflight_lock:
            ev = self._inflight.get(key)
            owner = ev is None
            if owner:
                ev = threading.Event()
                self._inflight[key] = ev
        if not owner:
            ev.wait(timeout=60)
            f = self.decoded.get(key)
            if f is not None:
                return f
        try:
            r = self.reader(match_id)
            f, extras = r.read(n)
            for k, e in extras.items():
                self._put_decoded(match_id, k, e)
            self._put_decoded(match_id, n, f)
            return f
        finally:
            if owner:
                with self._inflight_lock:
                    self._inflight.pop(key, None)
                ev.set()

    # ----------------------------------------------------------------- JPEGs
    def jpeg(self, match_id: str, n: int, width: int = 960, quality: int | None = None) -> bytes:
        if quality is None:
            quality = 85 if width >= 480 else 80
        key = (match_id, n, width, quality)
        data = self.jpegs.get(key)
        if data is not None:
            return data
        img = resize_to_width(self.frame(match_id, n), width)
        data = encode_jpeg(img, quality)
        self.jpegs.put(key, data)
        return data

    def scoreboard_jpeg(self, match_id: str, n: int, region: dict, scale: float = 2.0,
                        quality: int = 92) -> bytes:
        key = (match_id, n, "sb", float(scale), quality)
        data = self.jpegs.get(key)
        if data is not None:
            return data
        f = self.frame(match_id, n)
        H, W = f.shape[:2]
        m = SCOREBOARD_MARGIN
        x0 = max(0, int(region["x"]) - m)
        y0 = max(0, int(region["y"]) - m)
        x1 = min(W, int(region["x"]) + int(region["w"]) + m)
        y1 = min(H, int(region["y"]) + int(region["h"]) + m)
        if x1 <= x0 or y1 <= y0:
            raise FrameError(404, "La región del marcador está fuera de la imagen")
        crop = f[y0:y1, x0:x1]
        if scale != 1:
            crop = cv2.resize(crop, (max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale)))),
                              interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        data = encode_jpeg(crop, quality)
        self.jpegs.put(key, data)
        return data


# ---------------------------------------------------------------- diff curve
DIFF_SIZE = (160, 90)
DIFF_CHUNK = 2500        # frames per worker when the window is long
DIFF_WORKERS = 6


def _diff_chunk(path: str, a: int, b: int) -> np.ndarray:
    """Raw mean |gray(k) - gray(k-1)| / 255 at 160x90 for k in [a, b)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FrameError(500, "No se puede abrir el vídeo")
    try:
        first = a - 1 if a > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        prev = None
        out = np.zeros(b - a, dtype=np.float32)
        for k in range(first, b):
            ok, f = cap.read()
            if not ok:
                raise FrameError(500, f"No se pudo decodificar el fotograma {k}")
            g = cv2.cvtColor(cv2.resize(f, DIFF_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            g = g.astype(np.int16)
            if k >= a and prev is not None:
                out[k - a] = float(np.abs(g - prev).mean()) / 255.0
            prev = g
        return out
    finally:
        cap.release()


def compute_diff_curve(path: Path, f0: int, f1: int) -> np.ndarray:
    """values[i] = change between frames f0+i-1 and f0+i (values[0] = 0), i < f1-f0."""
    n = f1 - f0
    bounds = list(range(f0, f1, DIFF_CHUNK)) + [f1]
    parts = list(zip(bounds[:-1], bounds[1:]))
    if len(parts) == 1:
        vals = _diff_chunk(str(path), f0, f1)
    else:
        with ThreadPoolExecutor(max_workers=min(DIFF_WORKERS, len(parts))) as ex:
            vals = np.concatenate(list(ex.map(lambda p: _diff_chunk(str(path), *p), parts)))
    assert len(vals) == n
    vals[0] = 0.0
    return vals
