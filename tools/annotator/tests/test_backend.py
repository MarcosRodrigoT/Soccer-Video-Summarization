"""Backend tests of the annotator.

    source venv/bin/activate
    python -m unittest discover -s tools/annotator/tests -v

Uses the real PL1 / SA1 / Match1 videos (paths from the catalogue) and a
temporary data root. Takes a few minutes (sequential decoding references).
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import frames  # noqa: E402
import server  # noqa: E402
import store as store_mod  # noqa: E402
from frames import FrameError, FrameService, VideoReader  # noqa: E402

GOAL_MATCHES = ("PL1", "SA1")
N_REF = 700            # frames read sequentially from 0 as the reference
DEEP_FRAME = 12_000    # a far position checked against grab() from frame 0

STATE: dict = {}


def md5(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def setUpModule():
    tmp = Path(tempfile.mkdtemp(prefix="annotator_test_"))
    srv = server.make_server(0, tmp / "data", log=False)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    STATE.update(tmp=tmp, srv=srv, app=srv.app, port=srv.server_address[1], refs={})


def tearDownModule():
    STATE["srv"].shutdown()
    STATE["srv"].server_close()
    STATE["app"].frames.close()
    shutil.rmtree(STATE["tmp"], ignore_errors=True)


def reference_hashes(mid: str) -> list[str]:
    """md5 of frames 0..N_REF-1 read sequentially with a fresh capture."""
    refs = STATE["refs"]
    if mid not in refs:
        cap = cv2.VideoCapture(str(STATE["app"].catalog.video_path(mid)))
        hs = []
        for _ in range(N_REF):
            ok, f = cap.read()
            assert ok
            hs.append(md5(f))
        cap.release()
        refs[mid] = hs
    return refs[mid]


def frame_via_preroll(path: Path, n: int, preroll: int = 300) -> np.ndarray:
    """Independent strategy: fresh capture, seek to n-preroll, grab forward, read."""
    cap = cv2.VideoCapture(str(path))
    a = max(0, n - preroll)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    for _ in range(n - a):
        assert cap.grab()
    ok, f = cap.read()
    cap.release()
    assert ok, n
    return f


# ---------------------------------------------------------------- HTTP utils
def http(method: str, path: str, body=None, raw: bool = False):
    url = f"http://127.0.0.1:{STATE['port']}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = r.read()
            return r.status, (payload if raw else json.loads(payload)), r.headers
    except urllib.error.HTTPError as e:
        payload = e.read()
        try:
            payload = json.loads(payload)
        except ValueError:
            pass
        return e.code, payload, e.headers


# ======================================================= frame exactness tests
class FrameExactnessMixin:
    mid = None

    @classmethod
    def setUpClass(cls):
        app = STATE["app"]
        cls.path = app.catalog.video_path(cls.mid)
        cls.total = app.catalog.total_frames(cls.mid)
        cls.ref = reference_hashes(cls.mid)
        # Fresh service (own caches) so that earlier tests do not mask errors.
        cls.svc = FrameService(app.catalog)

    @classmethod
    def tearDownClass(cls):
        cls.svc.close()

    def check(self, n, frame=None):
        f = self.svc.frame(self.mid, n) if frame is None else frame
        self.assertEqual(md5(f), self.ref[n], f"{self.mid}: el fotograma {n} no coincide con la lectura secuencial")

    def test_01_first_frame(self):
        self.check(0)

    def test_02_sequential(self):
        for n in range(150, 181):           # seek to 150, then read() fast path
            self.check(n)
        r = self.svc.reader(self.mid)
        self.assertGreater(r.stats["read"], 20)

    def test_03_backward_steps(self):
        self.svc.frame(self.mid, 420)
        for n in range(419, 360, -1):        # <- pressed repeatedly (backfill + cache)
            self.check(n)
        self.assertGreater(self.svc.reader(self.mid).stats["backfill"], 0)

    def test_04_short_forward_jumps(self):
        for n in (500, 501, 525, 550, 600, 651, 652, 699):   # grab() path and seeks
            self.check(n)

    def test_05_random_jumps_near_start(self):
        rnd = random.Random(1234)
        for n in rnd.sample(range(N_REF), 40):
            self.check(n)

    def test_06_raw_reader_no_cache(self):
        r = VideoReader(self.path, self.total, pool_size=2)
        try:
            rnd = random.Random(99)
            seq = [0, 1, 2, 3, 699, 698, 10, 60, 61, 111, 112, 5] + rnd.sample(range(N_REF), 20)
            for n in seq:
                f, _ = r.read(n, allow_backfill=False)
                self.assertEqual(md5(f), self.ref[n], f"{self.mid}: lector sin caché, fotograma {n}")
        finally:
            r.close()

    def test_07_far_random_positions(self):
        rnd = random.Random(7 if self.mid == "PL1" else 8)
        positions = sorted(rnd.sample(range(1000, self.total - 1), 16))
        rnd.shuffle(positions)
        for n in positions:
            a = self.svc.frame(self.mid, n)
            b = frame_via_preroll(self.path, n)
            self.assertTrue(np.array_equal(a, b), f"{self.mid}: fotograma {n}, seek directo != preroll")
            # neighbours must differ from n somewhere in a window (otherwise the check is weak)
        # stepping around a far position: n-1, n+1, n+2 consistent with preroll strategy
        n = positions[0]
        for k in (n - 1, n + 1, n + 2, n - 2):
            self.assertTrue(np.array_equal(self.svc.frame(self.mid, k), frame_via_preroll(self.path, k)))

    def test_08_deep_sequential_reference(self):
        # Trusted reference: grab() from frame 0 up to DEEP_FRAME, then read a few frames.
        cap = cv2.VideoCapture(str(self.path))
        for _ in range(DEEP_FRAME):
            self.assertTrue(cap.grab())
        want = []
        for _ in range(5):
            ok, f = cap.read()
            self.assertTrue(ok)
            want.append(md5(f))
        cap.release()
        svc = FrameService(STATE["app"].catalog)
        try:
            for i in (2, 0, 4, 1, 3):
                self.assertEqual(md5(svc.frame(self.mid, DEEP_FRAME + i)), want[i],
                                 f"{self.mid}: fotograma {DEEP_FRAME + i} tras seek")
        finally:
            svc.close()

    def test_09_last_frame_and_out_of_range(self):
        last = self.total - 1
        self.assertTrue(np.array_equal(self.svc.frame(self.mid, last), frame_via_preroll(self.path, last)))
        cap = cv2.VideoCapture(str(self.path))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.total)
        cap.release()
        for bad in (-1, self.total, self.total + 1000):
            with self.assertRaises(FrameError) as cm:
                self.svc.frame(self.mid, bad)
            self.assertEqual(cm.exception.status, 404)

    def test_10_concurrent_requests(self):
        rnd = random.Random(5)
        jobs = [rnd.randrange(N_REF) for _ in range(60)]
        errors = []

        def worker(ns):
            for n in ns:
                try:
                    f = self.svc.frame(self.mid, n)
                    if md5(f) != self.ref[n]:
                        errors.append(n)
                except Exception as e:  # pragma: no cover
                    errors.append((n, repr(e)))

        self.svc.decoded.clear()
        ths = [threading.Thread(target=worker, args=(jobs[i::6],)) for i in range(6)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        self.assertEqual(errors, [])

    def test_11_filmstrip_pattern(self):
        # UI pattern: main frame n, then n-4..n+4 concurrently (backfill + waiting
        # requests), stepping forwards and backwards.
        svc = FrameService(STATE["app"].catalog)
        errors = []

        def get(k):
            try:
                if md5(svc.frame(self.mid, k)) != self.ref[k]:
                    errors.append(k)
            except Exception as e:  # pragma: no cover
                errors.append((k, repr(e)))

        try:
            for n in [300, 301, 302, 301, 300, 299, 298, 250, 275, 274, 600, 560, 520, 521, 100, 99, 98]:
                get(n)
                ths = [threading.Thread(target=get, args=(n + j,)) for j in range(-4, 5)]
                for t in ths:
                    t.start()
                for t in ths:
                    t.join()
            self.assertEqual(errors, [])
            self.assertGreater(svc.reader(self.mid).stats["backfill"], 0)
        finally:
            svc.close()


class FrameExactnessPL1(FrameExactnessMixin, unittest.TestCase):
    mid = "PL1"


class FrameExactnessSA1(FrameExactnessMixin, unittest.TestCase):
    mid = "SA1"


# ================================================================ HTTP tests
class HttpTest(unittest.TestCase):
    def test_health(self):
        code, body, _ = http("GET", "/api/health")
        self.assertEqual(code, 200)
        self.assertTrue(body["ok"])
        self.assertIn("version", body)

    def test_matches(self):
        code, body, _ = http("GET", "/api/matches?view=goals")
        self.assertEqual(code, 200)
        self.assertEqual([m["id"] for m in body], ["PL1", "SA1"])
        for m in body:
            for k in ("id", "label", "league", "season", "home", "away", "score", "fps",
                      "total_frames", "duration_sec", "width", "height"):
                self.assertIn(k, m)
            self.assertEqual(m["fps"], 25.0)
        self.assertEqual(body[0]["total_frames"], 153717)
        self.assertEqual(body[1]["total_frames"], 163543)
        code, body, _ = http("GET", "/api/matches?view=sbd")
        self.assertEqual(code, 200)
        self.assertEqual(len(body), 25)
        self.assertEqual(body[0]["id"], "Match1")
        self.assertEqual(body[0]["label"], "M1 · Valencia CF 2-3 FC Barcelona (2013/14)")
        self.assertEqual(body[-1]["id"], "Match25")
        self.assertTrue(all("error" not in m for m in body))
        code, body, _ = http("GET", "/api/matches?view=nada")
        self.assertEqual(code, 400)
        self.assertIn("error", body)

    def test_match_hints(self):
        code, body, _ = http("GET", "/api/match/PL1")
        self.assertEqual(code, 200)
        h = body["hints"]
        self.assertEqual(len(h["score_events"]), 8)
        self.assertEqual(h["score_events"][0], {"frame": 27975, "home": 0, "away": 1, "team": "away"})
        self.assertEqual(h["scoreboard_region"], {"x": 80, "y": 69, "w": 660, "h": 161})
        self.assertGreater(len(h["shot_starts"]), 100)
        self.assertEqual(h["shot_starts"], sorted(h["shot_starts"]))
        self.assertEqual(h["shot_starts"][0], 0)
        self.assertTrue(all(set(c) == {"start_frame", "end_frame"} for c in h["goal_clips"]))
        code, body, _ = http("GET", "/api/match/Match1")
        self.assertEqual(code, 200)
        self.assertIsNotNone(body["hints"]["scoreboard_region"])
        code, body, _ = http("GET", "/api/match/XX9")
        self.assertEqual(code, 404)
        self.assertIn("desconocido", body["error"])

    def test_frame_jpeg(self):
        code, data, hdr = http("GET", "/api/frame/PL1/1234?w=640", raw=True)
        self.assertEqual(code, 200)
        self.assertEqual(hdr["Content-Type"], "image/jpeg")
        self.assertEqual(hdr["X-Frame-Index"], "1234")
        self.assertIn("max-age=86400", hdr["Cache-Control"])
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(img.shape, (360, 640, 3))
        code, data, _ = http("GET", "/api/frame/SA1/0?w=160", raw=True)
        self.assertEqual(code, 200)
        self.assertEqual(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape, (90, 160, 3))
        # full resolution, high quality: the JPEG is closest to frame n, not its neighbours
        ref = reference_hashes("PL1")  # noqa: F841  (ensures the video is readable)
        path = STATE["app"].catalog.video_path("PL1")
        cap = cv2.VideoCapture(str(path))
        seq = [cap.read()[1] for _ in range(N_REF)]
        cap.release()
        for n in (0, 333, 600):
            code, data, _ = http("GET", f"/api/frame/PL1/{n}?w=1920&q=100", raw=True)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
            errs = {k: float(np.abs(img - seq[k]).mean()) for k in range(max(0, n - 3), n + 4)
                    if not np.array_equal(seq[k], seq[n])}
            err_n = float(np.abs(img - seq[n]).mean())
            self.assertLess(err_n, 2.0)
            for k, e in errs.items():
                self.assertLess(err_n, e, f"JPEG del fotograma {n} se parece más al {k}")

    def test_frame_errors(self):
        code, body, _ = http("GET", "/api/frame/PL1/153717")
        self.assertEqual(code, 404)
        self.assertIn("fuera de rango", body["error"])
        code, body, _ = http("GET", "/api/frame/PL1/-1")
        self.assertEqual(code, 404)
        code, body, _ = http("GET", "/api/frame/PL1/abc")
        self.assertEqual(code, 400)
        code, body, _ = http("GET", "/api/frame/PL1/10?w=50")
        self.assertEqual(code, 400)
        self.assertIn("«w»", body["error"])
        code, body, _ = http("GET", "/api/frame/NOPE/10")
        self.assertEqual(code, 404)
        code, body, _ = http("GET", "/api/nada")
        self.assertEqual(code, 404)
        code, body, _ = http("GET", "/static/../server.py")
        self.assertEqual(code, 404)

    def test_scoreboard(self):
        code, data, hdr = http("GET", "/api/scoreboard/PL1/28000?scale=2", raw=True)
        self.assertEqual(code, 200)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        # region 80,69,660x161 + 12 px margin each side, x2
        self.assertEqual(img.shape[:2], ((161 + 24) * 2, (660 + 24) * 2))
        code, data, _ = http("GET", "/api/scoreboard/SA1/28000", raw=True)
        self.assertEqual(code, 200)
        code, body, _ = http("GET", "/api/scoreboard/PL1/28000?scale=9", raw=False)
        self.assertEqual(code, 400)

    def test_diffcurve(self):
        code, body, _ = http("GET", "/api/diffcurve/PL1?start=100&end=400")
        self.assertEqual(code, 200)
        self.assertEqual((body["start"], body["end"]), (100, 400))
        vals = body["values"]
        self.assertEqual(len(vals), 300)
        self.assertEqual(vals[0], 0)
        self.assertAlmostEqual(max(vals), 1.0, places=3)
        self.assertTrue(all(0 <= v <= 1 for v in vals))
        cache = STATE["app"].data_root / "cache" / "diff_PL1_100_400.npy"
        self.assertTrue(cache.is_file())
        raw = np.load(cache)
        # independent sequential computation
        cap = cv2.VideoCapture(str(STATE["app"].catalog.video_path("PL1")))
        grays = []
        for k in range(400):
            ok, f = cap.read()
            if k >= 99:
                g = cv2.cvtColor(cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
                grays.append(g.astype(np.int16))
        cap.release()
        want = [0.0] + [float(np.abs(grays[i] - grays[i - 1]).mean()) / 255 for i in range(2, len(grays))]
        self.assertEqual(len(want), 300)
        np.testing.assert_allclose(raw, np.array(want, np.float32), atol=1e-6)
        # chunked (parallel) computation equals the single-chunk one
        with mock.patch.object(frames, "DIFF_CHUNK", 70):
            chunked = frames.compute_diff_curve(STATE["app"].catalog.video_path("PL1"), 100, 400)
        np.testing.assert_allclose(chunked, raw, atol=1e-6)
        code, body, _ = http("GET", "/api/diffcurve/PL1?start=400&end=100")
        self.assertEqual(code, 400)
        code, body, _ = http("GET", "/api/diffcurve/PL1?start=0&end=60000")
        self.assertEqual(code, 400)

    def test_static(self):
        static = Path(server.STATIC)
        if (static / "index.html").is_file():
            code, data, hdr = http("GET", "/", raw=True)
            self.assertEqual(code, 200)
            self.assertIn("no-cache", hdr["Cache-Control"])
        else:
            code, _, _ = http("GET", "/")
            self.assertEqual(code, 404)


# =============================================================== goals tests
class GoalsTest(unittest.TestCase):
    mid = "SA1"

    def setUp(self):
        st = STATE["app"].store
        for p in (st.goals_path(self.mid), st.goals_csv_path(self.mid),
                  st.root / "history" / f"goals_{self.mid}.jsonl"):
            if p.exists():
                p.unlink()

    def history_lines(self):
        p = STATE["app"].store.root / "history" / f"goals_{self.mid}.jsonl"
        return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []

    def test_roundtrip_revision_conflict_csv_history(self):
        code, doc, _ = http("GET", f"/api/goals/{self.mid}")
        self.assertEqual(code, 200)
        self.assertEqual(doc["revision"], 0)
        self.assertEqual(doc["goals"], [])
        self.assertFalse(STATE["app"].store.goals_path(self.mid).exists())   # lazy creation

        doc["goals"] = [
            {"id": "g2", "start_frame": 50000, "end_frame_shot": 50300, "end_frame_scoreboard": 50310,
             "notes": "remate, de cabeza \"raro\"", "skipped": False, "hint_frame": 50500},
            {"id": "g1", "start_frame": 20000, "end_frame_shot": 20100, "end_frame_scoreboard": 20200,
             "notes": "", "skipped": False},
            {"id": "g3", "start_frame": 70000, "end_frame_shot": None, "end_frame_scoreboard": None,
             "notes": "", "skipped": False},
            {"id": "g4", "start_frame": 90000, "end_frame_shot": 90010, "end_frame_scoreboard": 90020,
             "notes": "anulado", "skipped": True},
            {"id": "g5", "start_frame": None, "end_frame_shot": None, "end_frame_scoreboard": None,
             "notes": "", "skipped": False},
        ]
        code, saved, _ = http("PUT", f"/api/goals/{self.mid}", doc)
        self.assertEqual(code, 200, saved)
        self.assertEqual(saved["revision"], 1)
        self.assertIsNotNone(saved["updated"])
        self.assertEqual(saved["goals"], doc["goals"])            # unknown keys preserved
        code, again, _ = http("GET", f"/api/goals/{self.mid}")
        self.assertEqual(again, saved)

        csv_text = STATE["app"].store.goals_csv_path(self.mid).read_text()
        self.assertEqual(csv_text,
                         "goal_id,start_frame,end_frame_shot,end_frame_scoreboard,notes\n"
                         "1,20000,20100,20200,\n"
                         "2,50000,50300,50310,\"remate, de cabeza \"\"raro\"\"\"\n")

        saved["complete"] = True
        code, saved2, _ = http("PUT", f"/api/goals/{self.mid}", saved)
        self.assertEqual((code, saved2["revision"]), (200, 2))

        code, body, _ = http("PUT", f"/api/goals/{self.mid}", saved)    # stale revision 1
        self.assertEqual(code, 409)
        self.assertIn("error", body)
        self.assertEqual(body["revision"], 2)
        self.assertEqual(body["current"]["revision"], 2)
        self.assertTrue(body["current"]["complete"])

        hist = self.history_lines()
        self.assertEqual([h["revision"] for h in hist], [1, 2])
        self.assertEqual(hist[-1], saved2)

    def test_new_doc_requires_revision_0(self):
        doc = STATE["app"].store.initial_goals(self.mid)
        doc["revision"] = 3
        code, body, _ = http("PUT", f"/api/goals/{self.mid}", doc)
        self.assertEqual(code, 409)
        self.assertEqual(body["current"]["revision"], 0)

    def test_validation(self):
        base = STATE["app"].store.initial_goals(self.mid)
        cases = [
            ([{"id": "g1", "start_frame": 500, "end_frame_shot": 400}], "anterior"),
            ([{"id": "g1", "start_frame": 10 ** 7}], "fuera del vídeo"),
            ([{"id": "g1", "start_frame": -5}], "fuera del vídeo"),
            ([{"id": "g1", "start_frame": "12"}], "entero"),
            ([{"id": "g1", "start_frame": True}], "entero"),
            ([{"start_frame": 1}], "identificador"),
            ([{"id": "g1"}, {"id": "g1"}], "repetido"),
            ([{"id": "g1", "notes": 3}], "notas"),
            ("nada", "lista"),
        ]
        for goals, msg in cases:
            d = dict(base, goals=goals)
            code, body, _ = http("PUT", f"/api/goals/{self.mid}", d)
            self.assertEqual(code, 400, (goals, body))
            self.assertIn(msg, body["error"])
        code, body, _ = http("PUT", f"/api/goals/{self.mid}", dict(base, match_id="PL1"))
        self.assertEqual(code, 400)
        code, body, _ = http("PUT", f"/api/goals/{self.mid}", dict(base, revision="0"))
        self.assertEqual(code, 400)
        req = urllib.request.Request(f"http://127.0.0.1:{STATE['port']}/api/goals/{self.mid}",
                                     data=b"{no json", method="PUT")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)
        self.assertIn("JSON", json.loads(cm.exception.read())["error"])
        self.assertFalse(STATE["app"].store.goals_path(self.mid).exists())
        code, body, _ = http("GET", "/api/goals/XX1")
        self.assertEqual(code, 404)

    def test_crash_during_write_keeps_old_content(self):
        st = STATE["app"].store
        doc = st.initial_goals(self.mid)
        doc["goals"] = [{"id": "g1", "start_frame": 100, "end_frame_shot": 200, "end_frame_scoreboard": 210,
                         "notes": "", "skipped": False}]
        saved = st.put_goals(self.mid, doc)
        before = st.goals_path(self.mid).read_bytes()
        saved["goals"][0]["start_frame"] = 150

        def boom(*a, **k):
            raise OSError("simulated crash before rename")

        with mock.patch.object(store_mod.os, "replace", boom):
            with self.assertRaises(OSError):
                st.put_goals(self.mid, saved)
        self.assertEqual(st.goals_path(self.mid).read_bytes(), before)
        self.assertEqual(st.get_goals(self.mid)["goals"][0]["start_frame"], 100)
        self.assertEqual(list(st.goals_path(self.mid).parent.glob(".*.tmp")), [])
        self.assertEqual(len(self.history_lines()), 1)
        # A leftover tmp file from a hard crash (no cleanup) does not affect reads.
        (st.goals_path(self.mid).parent / f".{self.mid}.json.999.1.tmp").write_text("{trunc")
        self.assertEqual(st.get_goals(self.mid)["revision"], 1)
        again = st.put_goals(self.mid, saved)
        self.assertEqual(again["revision"], 2)
        self.assertEqual(again["goals"][0]["start_frame"], 150)

    def test_concurrent_puts(self):
        mid = self.mid
        rounds, nthreads = 5, 8
        for r in range(rounds):
            code, cur, _ = http("GET", f"/api/goals/{mid}")
            results = []
            barrier = threading.Barrier(nthreads)

            def worker(i):
                d = json.loads(json.dumps(cur))
                d["goals"] = [{"id": f"g{i}", "start_frame": 1000 * i + r, "end_frame_shot": None,
                               "end_frame_scoreboard": None, "notes": f"hilo {i}", "skipped": False}]
                barrier.wait()
                code, body, _ = http("PUT", f"/api/goals/{mid}", d)
                results.append((code, i))

            ths = [threading.Thread(target=worker, args=(i,)) for i in range(nthreads)]
            for t in ths:
                t.start()
            for t in ths:
                t.join()
            codes = sorted(c for c, _ in results)
            self.assertEqual(codes, [200] + [409] * (nthreads - 1), results)
            winner = [i for c, i in results if c == 200][0]
            code, now, _ = http("GET", f"/api/goals/{mid}")
            self.assertEqual(now["revision"], cur["revision"] + 1)
            self.assertEqual(now["goals"][0]["id"], f"g{winner}")
        hist = self.history_lines()
        self.assertEqual([h["revision"] for h in hist], list(range(1, rounds + 1)))
        json.loads(STATE["app"].store.goals_path(mid).read_text())


# ================================================================= sbd tests
class SbdTest(unittest.TestCase):
    S, E = 30000, 37500

    def setUp(self):
        st = STATE["app"].store
        for d in ("sbd", "trash"):
            for p in (st.root / d).glob("*.json"):
                p.unlink()

    def create(self, mid="Match1", s=None, e=None):
        return http("POST", "/api/sbd/sessions",
                    {"match_id": mid, "start_frame": self.S if s is None else s, "end_frame": self.E if e is None else e})

    def put(self, doc, **changes):
        d = json.loads(json.dumps(doc))
        d.update(changes)
        return http("PUT", f"/api/sbd/{doc['session_id']}", d)

    def test_create_idempotent_and_window_checks(self):
        code, doc, _ = self.create()
        self.assertEqual(code, 201)
        self.assertEqual(doc["session_id"], "Match1_30000_37500")
        self.assertEqual((doc["revision"], doc["transitions"], doc["shot_labels"]), (1, [], {}))
        code, again, _ = self.create()
        self.assertEqual(code, 200)
        self.assertEqual(again, doc)
        for mid, s, e, status in [("Match1", 100, 110, 400),          # < 1 s
                                  ("Match1", 0, 90001, 400),          # > 60 min
                                  ("Match1", 150000, 160000, 400),    # beyond the video
                                  ("Match1", -10, 1000, 400),
                                  ("PL1", 1000, 2000, 400),           # not a LaLiga match
                                  ("Nope", 1000, 2000, 404)]:
            code, body, _ = self.create(mid, s, e)
            self.assertEqual(code, status, (mid, s, e, body))
            self.assertIn("error", body)
        code, body, _ = http("POST", "/api/sbd/sessions", {"match_id": "Match1", "start_frame": "1"})
        self.assertEqual(code, 400)
        code, body, _ = self.create("Match1", 0, 90000)                 # exactly 60 min is fine
        self.assertEqual(code, 201)

    def test_transitions_labels_pruning_and_list(self):
        code, doc, _ = self.create()
        trs = [{"id": "t1", "type": "cut", "frame": 30120},
               {"id": "t2", "type": "gradual", "start": 30500, "end": 30515},
               {"id": "t3", "type": "cut", "frame": 31000}]
        labels = {"30000": "play", "30120": "no_play", "30516": "play", "31000": "no_play"}
        code, saved, _ = self.put(doc, transitions=trs, shot_labels=labels)
        self.assertEqual(code, 200, saved)
        self.assertEqual(saved["revision"], 2)
        self.assertEqual(saved["shot_labels"], labels)
        self.assertEqual(saved["_pruned_labels"], [])
        self.assertEqual(STATE["app"].store.shot_starts(saved), [30000, 30120, 30516, 31000])

        code, got, _ = http("GET", f"/api/sbd/{doc['session_id']}")
        self.assertNotIn("_pruned_labels", got)
        self.assertEqual(got["transitions"], trs)

        code, lst, _ = http("GET", "/api/sbd/sessions")
        self.assertEqual(len(lst), 1)
        s = lst[0]
        self.assertEqual((s["session_id"], s["n_transitions"], s["n_shots"], s["n_labelled"], s["complete"]),
                         ("Match1_30000_37500", 3, 4, 4, False))

        # removing the cut at 31000 invalidates its label
        got["transitions"] = trs[:2]
        code, saved, _ = http("PUT", f"/api/sbd/{doc['session_id']}", got)
        self.assertEqual(code, 200, saved)
        self.assertEqual(saved["_pruned_labels"], ["31000"])
        self.assertNotIn("31000", saved["shot_labels"])
        code, got, _ = http("GET", f"/api/sbd/{doc['session_id']}")
        self.assertEqual(set(got["shot_labels"]), {"30000", "30120", "30516"})

        # complete requires every shot labelled
        code, body, _ = self.put(got, complete=True, shot_labels={"30000": "play"})
        self.assertEqual(code, 400)
        self.assertIn("faltan 2", body["error"])
        code, body, _ = self.put(got, complete=True)
        self.assertEqual(code, 200, body)
        self.assertTrue(body["complete"])

        # gradual at the window start / end leaves no empty shot
        code, body, _ = http("GET", f"/api/sbd/{doc['session_id']}")
        code, body, _ = self.put(body, complete=False, shot_labels={},
                                 transitions=[{"id": "a", "type": "gradual", "start": 30000, "end": 30010},
                                              {"id": "b", "type": "gradual", "start": 37490, "end": 37499}])
        self.assertEqual(code, 200, body)
        self.assertEqual(STATE["app"].store.shot_starts(body), [30011])

    def test_invalid_transitions(self):
        code, doc, _ = self.create()
        bad = [
            ([{"id": "t1", "type": "cut", "frame": 30000}], "dentro de la ventana"),
            ([{"id": "t1", "type": "cut", "frame": 37500}], "dentro de la ventana"),
            ([{"id": "t1", "type": "cut", "frame": 29000}], "dentro de la ventana"),
            ([{"id": "t1", "type": "gradual", "start": 37400, "end": 37500}], "dentro de la ventana"),
            ([{"id": "t1", "type": "gradual", "start": 29990, "end": 30010}], "dentro de la ventana"),
            ([{"id": "t1", "type": "gradual", "start": 30600, "end": 30500}], "anterior"),
            ([{"id": "t1", "type": "gradual", "start": 30600, "end": None}], "abierta"),
            ([{"id": "t1", "type": "cut", "frame": 30100}, {"id": "t2", "type": "cut", "frame": 30100}],
             "mismo fotograma"),
            ([{"id": "t1", "type": "gradual", "start": 30500, "end": 30515},
              {"id": "t2", "type": "cut", "frame": 30505}], "solapan"),
            ([{"id": "t1", "type": "gradual", "start": 30500, "end": 30515},
              {"id": "t2", "type": "cut", "frame": 30500}], "solapan"),
            ([{"id": "t2", "type": "cut", "frame": 30500},
              {"id": "t1", "type": "gradual", "start": 30500, "end": 30515}], "solapan"),
            ([{"id": "t1", "type": "gradual", "start": 30500, "end": 30515},
              {"id": "t2", "type": "cut", "frame": 30516}], "pegadas"),
            ([{"id": "t1", "type": "gradual", "start": 30500, "end": 30515},
              {"id": "t2", "type": "gradual", "start": 30510, "end": 30530}], "solapan"),
            ([{"id": "t1", "type": "gradual", "start": 30500, "end": 30600},
              {"id": "t2", "type": "gradual", "start": 30520, "end": 30530},
              {"id": "t3", "type": "cut", "frame": 30700}], "solapan"),
            ([{"id": "t1", "type": "fade", "frame": 30500}], "tipo desconocido"),
            ([{"id": "t1", "type": "cut", "frame": 30500}, {"id": "t1", "type": "cut", "frame": 30600}],
             "repetido"),
            ([{"type": "cut", "frame": 30500}], "identificador"),
            ([{"id": "t1", "type": "cut", "frame": "30500"}], "fotograma"),
        ]
        for trs, msg in bad:
            code, body, _ = self.put(doc, transitions=trs)
            self.assertEqual(code, 400, (trs, body))
            self.assertIn(msg, body["error"], trs)
        for labels, msg in [({"30000": "juego"}, "no válida"), ({"abc": "play"}, "clave"), ([], "objeto")]:
            code, body, _ = self.put(doc, shot_labels=labels)
            self.assertEqual(code, 400, labels)
            self.assertIn(msg, body["error"])
        code, body, _ = self.put(doc, start_frame=1000)
        self.assertEqual(code, 400)
        self.assertIn("No se puede cambiar", body["error"])
        # adjacent but valid: one-frame shots between consecutive cuts
        code, body, _ = self.put(doc, transitions=[{"id": "a", "type": "cut", "frame": 30100},
                                                   {"id": "b", "type": "cut", "frame": 30101},
                                                   {"id": "c", "type": "gradual", "start": 30102, "end": 30110},
                                                   {"id": "d", "type": "cut", "frame": 30112}])
        self.assertEqual(code, 200, body)
        code, cur, _ = http("GET", f"/api/sbd/{doc['session_id']}")
        self.assertEqual(cur["revision"], 2)
        code, body, _ = self.put(doc)                                     # stale revision
        self.assertEqual(code, 409)
        self.assertEqual(body["current"]["revision"], 2)

    def test_delete_moves_to_trash(self):
        code, doc, _ = self.create()
        sid = doc["session_id"]
        code, body, _ = http("DELETE", f"/api/sbd/{sid}")
        self.assertEqual(code, 200, body)
        st = STATE["app"].store
        self.assertFalse(st.sbd_path(sid).exists())
        trashed = list((st.root / "trash").glob(f"{sid}.*.json"))
        self.assertEqual(len(trashed), 1)
        self.assertEqual(json.loads(trashed[0].read_text())["session_id"], sid)
        code, body, _ = http("GET", f"/api/sbd/{sid}")
        self.assertEqual(code, 404)
        code, lst, _ = http("GET", "/api/sbd/sessions")
        self.assertEqual(lst, [])
        code, body, _ = http("PUT", f"/api/sbd/{sid}", doc)
        self.assertEqual(code, 404)
        code, body, _ = http("DELETE", f"/api/sbd/{sid}")
        self.assertEqual(code, 404)
        code, body, _ = http("GET", "/api/sbd/..%2F..%2Fetc")
        self.assertEqual(code, 404)


if __name__ == "__main__":
    unittest.main()
