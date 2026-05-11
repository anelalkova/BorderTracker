"""
snapshot_scheduler.py
=====================
Two jobs in one process:

  1. CameraWorker threads  (one per crossing)
     – Opens the HLS stream headlessly (no OpenCV window)
     – Runs YOLOv8 + BotSort tracking
     – Writes to `vehicle_crossings` per vehicle
     – Writes to `snapshots` every --interval minutes
     – Auto-reconnects on stream drop with exponential back-off

  2. Prediction scheduler  (main thread)
     – Reads the freshest snapshot for each crossing
     – Calls wait_estimator_v3.estimate_wait()
     – Saves result to `wait_time_estimates`
     – Runs every --pred-interval minutes (default 5)

Prerequisites:
    pip install psycopg2-binary sqlalchemy scikit-learn joblib numpy pandas ultralytics opencv-python-headless

Usage:
    python snapshot_scheduler.py                          # all crossings
    python snapshot_scheduler.py --crossing bogorodica   # single crossing
    python snapshot_scheduler.py --once                  # one prediction pass then exit (cameras keep running until Ctrl-C)
    python snapshot_scheduler.py --interval 5            # snapshot DB write interval (minutes)
    python snapshot_scheduler.py --pred-interval 5       # prediction interval (minutes)
    python snapshot_scheduler.py --model yolov8s         # lighter model
    python snapshot_scheduler.py --conf 0.30             # YOLO confidence threshold
    python snapshot_scheduler.py --verbose               # per-lane prediction detail
    python snapshot_scheduler.py --no-camera             # skip camera workers (prediction-only, original behaviour)
"""

import argparse
import json
import time
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import psycopg2.extras

# ── Reuse wait estimator logic ───────────────────────────────────────────────
try:
    from wait_estimator_v3 import (
        get_conn,
        get_engine,
        get_crossing_id,
        load_multiplier,
        estimate_wait,
        MODEL_DIR,
        CROSSINGS,
    )
    _imported_estimator = True
except ImportError:
    _imported_estimator = False

if not _imported_estimator:
    import joblib
    from sqlalchemy import create_engine as _create_engine

    DB_CONFIG = {
        "host":     "localhost",
        "port":     5432,
        "dbname":   "border_crossing",
        "user":     "postgres",
        "password": "postgres",
    }
    MODEL_DIR = Path("models_v3")
    CROSSINGS = [
        "bogorodica", "blace", "tabanovce",
        "deve_bair",  "kafasan", "medzitlija",
    ]
    DEFAULT_MULTIPLIER = 4.0

    def get_conn():
        return psycopg2.connect(**DB_CONFIG)

    def get_engine():
        c = DB_CONFIG
        url = (f"postgresql+psycopg2://{c['user']}:{c['password']}"
               f"@{c['host']}:{c['port']}/{c['dbname']}")
        return _create_engine(url)

    def get_crossing_id(conn, name):
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM crossings WHERE name = %s", (name,))
            row = cur.fetchone()
            return row[0] if row else None

    def load_multiplier(conn, crossing_id):
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT multiplier, notes FROM crossing_queue_multipliers
                WHERE crossing_id = %s
            """, (crossing_id,))
            row = cur.fetchone()
        if not row:
            return {"global": DEFAULT_MULTIPLIER, "tod": {}}
        tod = {}
        notes = row["notes"] or ""
        if "tod=" in notes:
            try:
                tod = json.loads(notes.split("tod=")[1])
            except Exception:
                pass
        return {"global": float(row["multiplier"]), "tod": tod}

    def estimate_wait(conn, crossing_id, multipliers, model_path=None, verbose=False):
        import wait_estimator_v3 as _wev3
        return _wev3.estimate_wait(conn, crossing_id, multipliers,
                                   model_path=model_path, verbose=verbose)

# ── Crossing definitions (lanes + stream URLs) ────────────────────────────────
# Imported from border_crossings.py if available, otherwise defined inline.
try:
    from border_crossings import CROSSINGS as CROSSINGS_CFG, VEHICLE_CLASSES, MIN_FRAMES, MAX_DURATION
except ImportError:
    CROSSINGS_CFG = {
        "bogorodica": {
            "display_name": "Bogorodica (МК–ГР)",
            "neighbor": "Greece",
            "lanes": {
                "Bogorodica L1": [(0.32,0.12),(0.39,0.14),(0.00,0.53),(0.00,0.25)],
                "Bogorodica L2": [(0.39,0.16),(0.43,0.16),(0.05,0.94),(0.00,0.68)],
                "Bogorodica L3": [(0.43,0.16),(0.46,0.16),(0.26,0.94),(0.05,0.94)],
                "Bogorodica L4": [(0.50,0.16),(0.54,0.16),(0.93,0.94),(0.73,0.94)],
                "Bogorodica L5": [(0.54,0.16),(0.59,0.16),(1.00,0.70),(0.93,0.94)],
            },
        },
        "blace": {
            "display_name": "Blace (МК–КС)",
            "neighbor": "Kosovo",
            "lanes": {
                "Blace L1": [(0.480,0.135),(0.435,0.211),(0.368,0.343),(0.308,0.474),(0.250,0.625),(0.196,0.769),(0.140,0.918),(0.117,0.993),(0.003,0.997),(0.000,0.720),(0.100,0.542),(0.193,0.394),(0.264,0.301),(0.347,0.214),(0.438,0.123)],
                "Blace L2": [(0.488,0.133),(0.519,0.135),(0.523,0.207),(0.551,0.510),(0.573,0.745),(0.596,0.993),(0.204,0.992),(0.270,0.731),(0.345,0.483),(0.410,0.291),(0.458,0.185)],
                "Blace L3": [(0.579,0.132),(0.910,0.995),(0.625,0.989),(0.548,0.263),(0.536,0.137)],
            },
        },
        "tabanovce": {
            "display_name": "Tabanovce (МК–СР)",
            "neighbor": "Serbia",
            "lanes": {
                "Tabanovce L1": [(0.516,0.177),(0.494,0.137),(0.356,0.161),(0.217,0.215),(0.006,0.389),(0.003,0.641),(0.203,0.384),(0.311,0.297)],
                "Tabanovce L2": [(0.003,0.684),(0.233,0.416),(0.346,0.324),(0.377,0.309),(0.414,0.368),(0.308,0.523),(0.221,0.666),(0.084,0.995),(0.001,0.991)],
                "Tabanovce L3": [(0.145,0.993),(0.421,0.997),(0.490,0.368),(0.415,0.376),(0.353,0.449)],
            },
        },
        "deve_bair": {
            "display_name": "Deve Bair (МК–БГ)",
            "neighbor": "Bulgaria",
            "lanes": {
                "DeveBair L1": [(0.406,0.168),(0.396,0.234),(0.345,0.340),(0.062,0.992),(0.423,0.996),(0.494,0.353),(0.507,0.168)],
                "DeveBair L2": [(0.573,0.179),(0.578,0.342),(0.645,0.996),(0.995,0.994),(0.997,0.658),(0.840,0.360),(0.814,0.290),(0.782,0.170)],
            },
        },
        "kafasan": {
            "display_name": "Kafasan (МК–АЛ)",
            "neighbor": "Albania",
            "lanes": {
                "Kafasan L1": [(0.511,0.243),(0.508,0.341),(0.233,0.994),(0.006,0.994),(0.004,0.716),(0.414,0.246)],
                "Kafasan L2": [(0.519,0.243),(0.523,0.338),(0.557,0.516),(0.612,0.670),(0.740,0.997),(0.998,0.996),(0.991,0.706),(0.602,0.244)],
            },
        },
        "medzitlija": {
            "display_name": "Megjitlija (МК–ГР)",
            "neighbor": "Greece",
            "lanes": {
                "Medzitlija L1": [(0.366,0.220),(0.002,0.506),(0.000,0.332),(0.236,0.222),(0.333,0.193)],
                "Medzitlija L2": [(-0.001,0.533),(0.723,0.262),(0.956,0.345),(0.995,0.547),(0.999,0.995),(0.000,0.995)],
            },
        },
    }
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    MIN_FRAMES   = 15
    MAX_DURATION = 7200

STREAM_BASE = "https://streaming1.neotel.net.mk/stream/{name}.m3u8"

# ── Scheduler constants ───────────────────────────────────────────────────────
MAX_SNAPSHOT_AGE_MINUTES = 15
MODEL_VERSION            = "v3-scheduler"


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _point_in_polygon(px, py, poly):
    inside, n, j = False, len(poly), len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def _find_lane(cx_f, cy_f, lanes_cfg):
    for lane_name, poly in lanes_cfg.items():
        if _point_in_polygon(cx_f, cy_f, poly):
            return lane_name
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers shared by both workers and the prediction loop
# ─────────────────────────────────────────────────────────────────────────────

def _db_connect():
    return get_conn()

def _ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE wait_time_estimates
                ADD COLUMN IF NOT EXISTS context_json JSONB;
        """)
    conn.commit()

def _latest_snapshot_age(conn, crossing_id):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT captured_at FROM snapshots
            WHERE crossing_id = %s
            ORDER BY captured_at DESC LIMIT 1
        """, (crossing_id,))
        row = cur.fetchone()
    if not row:
        return None
    ts = row[0]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - ts).total_seconds() / 60, 1)

def _save_estimate(conn, crossing_id, result):
    cw         = result["crossing_wait"]
    wait_range = cw["wait_high_min"] - cw["wait_low_min"]
    confidence = max(0.0, min(1.0, 1.0 - wait_range / 60.0))
    context    = {
        "lanes":          result["lanes"],
        "snapshot_at":    result["snapshot_at"],
        "multiplier":     result["multiplier"],
        "hour":           result["hour"],
        "total_vehicles": result["total_vehicles"],
        "crossing_wait":  cw,
    }
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO wait_time_estimates
                (crossing_id, estimated_at, estimated_wait_minutes,
                 confidence, model_version, context_json)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            crossing_id,
            datetime.now(timezone.utc),
            cw["weighted_wait_min"],
            round(confidence, 3),
            MODEL_VERSION,
            psycopg2.extras.Json(context),
        ))
        row_id = cur.fetchone()[0]
    conn.commit()
    return row_id


# ─────────────────────────────────────────────────────────────────────────────
# CameraWorker — headless stream reader + YOLO tracker
# ─────────────────────────────────────────────────────────────────────────────

class CameraWorker(threading.Thread):
    """
    Background thread for one border crossing.
    Reads HLS stream headlessly, tracks vehicles with BotSort,
    writes to `vehicle_crossings` and `snapshots`.
    """

    def __init__(self, crossing_name, model_name="yolov8m",
                 conf=0.30, interval_minutes=5, stop_event=None):
        super().__init__(name=f"cam-{crossing_name}", daemon=True)
        self.crossing_name   = crossing_name
        self.model_name      = model_name
        self.conf            = conf
        self.interval_sec    = interval_minutes * 60
        self.stop_event      = stop_event or threading.Event()

        cfg = CROSSINGS_CFG.get(crossing_name, {})
        self.display_name = cfg.get("display_name", crossing_name)
        self.lanes_cfg    = cfg.get("lanes", {})
        self.stream_url   = STREAM_BASE.format(name=crossing_name)

        self._conn         = None
        self._crossing_id  = None
        self._total_saved  = 0
        self._total_skip   = 0

    # ── logging helpers ──────────────────────────────────────────────────

    def _log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [cam:{self.crossing_name}] {ts}  {msg}")

    # ── DB ───────────────────────────────────────────────────────────────

    def _init_db(self):
        self._conn = _db_connect()
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO crossings (name, display_name, neighbor)
                VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING
            """, (
                self.crossing_name,
                CROSSINGS_CFG[self.crossing_name]["display_name"],
                CROSSINGS_CFG[self.crossing_name]["neighbor"],
            ))
        self._conn.commit()
        self._crossing_id = get_crossing_id(self._conn, self.crossing_name)

    def _db_ok(self):
        """Ping the connection; reconnect if dead."""
        try:
            self._conn.cursor().execute("SELECT 1")
            return True
        except Exception:
            self._log("DB connection lost — reconnecting …")
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = _db_connect()
            return True

    def _save_snapshot(self, lane_counts, fps):
        if not self._db_ok():
            return
        totals     = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        grand      = 0
        for lc in lane_counts.values():
            grand += lc.get("total", 0)
            for vtype, cnt in lc.get("by_type", {}).items():
                totals[vtype] = totals.get(vtype, 0) + cnt
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO snapshots
                        (crossing_id, captured_at, interval_minutes,
                         total_vehicles, cars, motorcycles, buses, trucks,
                         lane_breakdown, fps)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self._crossing_id,
                    datetime.now(timezone.utc),
                    int(self.interval_sec / 60),
                    grand,
                    totals["car"], totals["motorcycle"],
                    totals["bus"], totals["truck"],
                    psycopg2.extras.Json(lane_counts),
                    round(fps, 2),
                ))
            self._conn.commit()
            self._log(f"Snapshot saved  queue={grand}  fps={fps:.1f}")
        except Exception as e:
            self._log(f"Snapshot save error: {e}")
            try:
                self._conn.rollback()
            except Exception:
                pass

    def _save_track(self, track):
        if not self._db_ok():
            return False
        duration = (track["exited_at"] - track["entered_at"]).total_seconds()
        avg_conf = (
            track["confidence_sum"] / track["frame_count"]
            if track["frame_count"] > 0 else None
        )
        try:
            with self._conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO vehicle_crossings
                        (crossing_id, track_id, vehicle_type, lane,
                         entered_at, exited_at, duration_sec,
                         frame_count, avg_confidence, notes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    self._crossing_id,
                    track["track_id"],
                    track["vehicle_type"],
                    track["lane"],
                    track["entered_at"],
                    track["exited_at"],
                    round(duration, 2),
                    track["frame_count"],
                    round(avg_conf, 3) if avg_conf else None,
                    track.get("notes"),
                ))
            self._conn.commit()
            return True
        except Exception as e:
            self._log(f"Track save error (id={track['track_id']}): {e}")
            try:
                self._conn.rollback()
            except Exception:
                pass
            return False

    # ── stream ───────────────────────────────────────────────────────────

    def _open_stream(self):
        cap = cv2.VideoCapture(self.stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,  5_000)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open stream: {self.stream_url}")
        return cap

    # ── main loop ────────────────────────────────────────────────────────

    def run(self):
        self._log("Starting …")

        try:
            self._init_db()
        except Exception as e:
            self._log(f"DB init failed: {e}")
            return

        # Load YOLO
        try:
            from ultralytics import YOLO
        except ImportError:
            self._log("ultralytics not installed — camera worker disabled.")
            return

        model_file = self.model_name if self.model_name.endswith(".pt") else f"{self.model_name}.pt"
        self._log(f"Loading {model_file} …")
        try:
            model = YOLO(model_file)
        except Exception as e:
            self._log(f"Failed to load model: {e}")
            return

        self._log("Model ready. Opening stream …")

        cap             = None
        active_vehicles = {}
        lane_counts     = {n: {"total": 0, "by_type": {}} for n in self.lanes_cfg}
        last_snap_t     = time.time()
        frame_times     = []
        fps_display     = 0.0

        def _try_open():
            retry = 0
            while not self.stop_event.is_set():
                retry += 1
                wait = min(2 ** retry, 60)
                self._log(f"Reconnect attempt {retry} (wait {wait}s) …")
                time.sleep(wait)
                try:
                    c = self._open_stream()
                    self._log(f"Stream connected (attempt {retry}).")
                    return c
                except RuntimeError as e:
                    self._log(f"  → {e}")
            return None

        # Initial open
        try:
            cap = self._open_stream()
            self._log("Stream connected.")
        except RuntimeError as e:
            self._log(f"Initial open failed: {e} — entering retry loop.")
            cap = _try_open()
            if cap is None:
                self._log("Stop requested during reconnect. Exiting.")
                return

        while not self.stop_event.is_set():
            # ── periodic snapshot ─────────────────────────────────────
            if time.time() - last_snap_t >= self.interval_sec:
                self._save_snapshot(lane_counts, fps_display)
                last_snap_t = time.time()
                # Reset per-interval lane counts after saving
                lane_counts = {n: {"total": 0, "by_type": {}} for n in self.lanes_cfg}

            # ── read frame ────────────────────────────────────────────
            ret, frame = cap.read()
            # Drain a second frame to avoid buffer build-up on HLS
            cap.read()

            if not ret or frame is None:
                self._log("Stream lost.")
                cap.release()
                cap = _try_open()
                if cap is None:
                    break
                continue

            h, w = frame.shape[:2]

            # ── YOLO tracking ─────────────────────────────────────────
            try:
                results = model.track(
                    frame,
                    conf=self.conf,
                    persist=True,
                    tracker="botsort.yaml",
                    verbose=False,
                    iou=0.30,
                )[0]
            except Exception as e:
                self._log(f"model.track error: {e}")
                continue

            now_dt      = datetime.now(timezone.utc)
            current_ids = set()
            frame_lane  = {n: {"total": 0, "by_type": {}} for n in self.lanes_cfg}

            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                track_id = int(box.id[0]) if box.id is not None else None
                label    = VEHICLE_CLASSES[cls_id]
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                cx_f = ((x1 + x2) / 2) / w
                cy_f = ((y1 + y2) / 2) / h

                lane_hit = _find_lane(cx_f, cy_f, self.lanes_cfg)
                if lane_hit:
                    frame_lane[lane_hit]["total"] += 1
                    frame_lane[lane_hit]["by_type"][label] = (
                        frame_lane[lane_hit]["by_type"].get(label, 0) + 1
                    )

                if track_id is None:
                    continue

                current_ids.add(track_id)

                if track_id not in active_vehicles:
                    active_vehicles[track_id] = {
                        "entered_at":     now_dt,
                        "last_seen_at":   now_dt,
                        "lane":           lane_hit,
                        "vehicle_type":   label,
                        "frame_count":    1,
                        "confidence_sum": conf_val,
                        "notes":          None,
                        "lane_history":   [lane_hit] if lane_hit else [],
                    }
                else:
                    v = active_vehicles[track_id]
                    v["last_seen_at"]    = now_dt
                    v["frame_count"]    += 1
                    v["confidence_sum"] += conf_val
                    if lane_hit and (not v["lane_history"] or v["lane_history"][-1] != lane_hit):
                        v["lane_history"].append(lane_hit)
                        if len(v["lane_history"]) > 1:
                            v["notes"] = f"lane_switch:{'>'.join(v['lane_history'])}"
                    if v["lane"] is None and lane_hit:
                        v["lane"] = lane_hit

            # Use the running maximum for the interval snapshot
            for ln in self.lanes_cfg:
                if frame_lane[ln]["total"] > lane_counts[ln]["total"]:
                    lane_counts[ln] = frame_lane[ln].copy()

            # ── flush vehicles that left frame ────────────────────────
            for tid in set(active_vehicles.keys()) - current_ids:
                v        = active_vehicles.pop(tid)
                duration = (v["last_seen_at"] - v["entered_at"]).total_seconds()
                if v["frame_count"] < MIN_FRAMES or duration > MAX_DURATION:
                    self._total_skip += 1
                    continue
                ok = self._save_track({
                    "track_id":       tid,
                    "vehicle_type":   v["vehicle_type"],
                    "lane":           v["lane"],
                    "entered_at":     v["entered_at"],
                    "exited_at":      v["last_seen_at"],
                    "frame_count":    v["frame_count"],
                    "confidence_sum": v["confidence_sum"],
                    "notes":          v.get("notes"),
                })
                if ok:
                    self._total_saved += 1
                else:
                    self._total_skip  += 1

            # ── FPS ───────────────────────────────────────────────────
            frame_times.append(time.time())
            frame_times = [t for t in frame_times if time.time() - t < 2.0]
            fps_display = len(frame_times) / 2.0

        # ── shutdown: flush remaining tracks ─────────────────────────
        self._log(f"Flushing {len(active_vehicles)} in-flight tracks …")
        flush_dt = datetime.now(timezone.utc)
        for tid, v in active_vehicles.items():
            duration = (flush_dt - v["entered_at"]).total_seconds()
            if v["frame_count"] < MIN_FRAMES or duration > MAX_DURATION:
                continue
            self._save_track({
                "track_id":       tid,
                "vehicle_type":   v["vehicle_type"],
                "lane":           v["lane"],
                "entered_at":     v["entered_at"],
                "exited_at":      flush_dt,
                "frame_count":    v["frame_count"],
                "confidence_sum": v["confidence_sum"],
                "notes":          (v.get("notes") or "") + " [shutdown]",
            })

        if cap:
            cap.release()
        if self._conn:
            self._conn.close()
        self._log(f"Stopped. saved={self._total_saved}  skipped={self._total_skip}")


# ─────────────────────────────────────────────────────────────────────────────
# Prediction scheduler (runs on main thread)
# ─────────────────────────────────────────────────────────────────────────────

def _run_prediction(conn, name, verbose):
    cid = get_crossing_id(conn, name)
    if not cid:
        print(f"  [{name}] Not found in DB — skipping.")
        return None

    age = _latest_snapshot_age(conn, cid)
    if age is None:
        print(f"  [{name}] No snapshots yet — skipping.")
        return None
    if age > MAX_SNAPSHOT_AGE_MINUTES:
        print(f"  [{name}] Snapshot is {age:.1f} min old (limit {MAX_SNAPSHOT_AGE_MINUTES}) — skipping.")
        return None

    model_path = MODEL_DIR / f"{name}_proc_time.joblib"
    if not model_path.exists():
        print(f"  [{name}] No trained model at {model_path} — using fallback.")
        model_path = None

    multipliers = load_multiplier(conn, cid)

    try:
        result = estimate_wait(conn, cid, multipliers,
                               model_path=model_path, verbose=verbose)
    except Exception as e:
        print(f"  [{name}] estimate_wait error: {e}")
        if verbose:
            traceback.print_exc()
        return None

    row_id = _save_estimate(conn, cid, result)
    cw     = result["crossing_wait"]
    ts     = datetime.now().strftime("%H:%M:%S")
    print(
        f"  [{name}]  {ts}  "
        f"wait={cw['weighted_wait_min']:.0f}min "
        f"({cw['wait_low_min']:.0f}–{cw['wait_high_min']:.0f})  "
        f"[{cw['congestion'].upper()}]  "
        f"best={cw['best_lane']}  "
        f"snap_age={age:.1f}min  "
        f"row_id={row_id}"
    )
    return result


def _run_all_predictions(conn, targets, verbose):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'─'*60}")
    print(f"  Prediction pass at {ts}  ({len(targets)} crossing(s))")
    print(f"{'─'*60}")
    ok, skipped, errors = 0, 0, 0
    for name in targets:
        try:
            r = _run_prediction(conn, name, verbose)
            if r is None:
                skipped += 1
            else:
                ok += 1
        except Exception as e:
            errors += 1
            print(f"  [{name}] Unhandled error: {e}")
            if verbose:
                traceback.print_exc()
            try:
                conn.rollback()
            except Exception:
                pass
    print(f"\n  Done — saved={ok}  skipped={skipped}  errors={errors}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Integrated border crossing: camera capture + wait-time prediction"
    )
    parser.add_argument("--crossing",      choices=CROSSINGS, default=None,
                        help="Single crossing (default: all)")
    parser.add_argument("--once",          action="store_true",
                        help="Run one prediction pass then keep cameras alive (Ctrl-C to stop)")
    parser.add_argument("--interval",      type=int, default=5,
                        help="Snapshot DB write interval in minutes (default: 5)")
    parser.add_argument("--pred-interval", type=int, default=5,
                        help="Wait-time prediction interval in minutes (default: 5)")
    parser.add_argument("--model",         default="yolov8m",
                        help="YOLO model: yolov8n/s/m/l/x (default: yolov8m)")
    parser.add_argument("--conf",          type=float, default=0.30,
                        help="YOLO confidence threshold (default: 0.30)")
    parser.add_argument("--verbose",       action="store_true",
                        help="Per-lane prediction detail")
    parser.add_argument("--no-camera",     action="store_true",
                        help="Skip camera workers (prediction-only mode)")
    args = parser.parse_args()

    targets = [args.crossing] if args.crossing else CROSSINGS

    print(f"\n{'='*65}")
    print(f"  Border Crossing Integrated Scheduler")
    print(f"  Crossings      : {', '.join(targets)}")
    print(f"  Snapshot every : {args.interval} min")
    print(f"  Predict every  : {args.pred_interval} min")
    print(f"  YOLO model     : {args.model}  (conf ≥ {args.conf})")
    print(f"  Model dir      : {MODEL_DIR.resolve()}")
    print(f"  Camera workers : {'disabled (--no-camera)' if args.no_camera else 'enabled'}")
    print(f"{'='*65}\n")

    # ── DB connection for prediction loop ─────────────────────────────
    pred_conn = _db_connect()
    _ensure_schema(pred_conn)
    print("  PostgreSQL connected (prediction loop).\n")

    # ── Start camera workers ───────────────────────────────────────────
    stop_event = threading.Event()
    workers    = []

    if not args.no_camera:
        for name in targets:
            if name not in CROSSINGS_CFG:
                print(f"  [{name}] No camera config — skipping camera worker.")
                continue
            w = CameraWorker(
                crossing_name    = name,
                model_name       = args.model,
                conf             = args.conf,
                interval_minutes = args.interval,
                stop_event       = stop_event,
            )
            w.start()
            workers.append(w)
        print(f"  {len(workers)} camera worker(s) started.\n")
    else:
        print("  Camera workers skipped (--no-camera).\n")

    # ── Give streams ~30 s to produce their first snapshot ────────────
    if workers:
        print("  Waiting 30 s for initial snapshots …")
        time.sleep(30)

    # ── Initial prediction pass ────────────────────────────────────────
    _run_all_predictions(pred_conn, targets, args.verbose)

    if args.once:
        print("\n  --once: stopping camera workers …")
        stop_event.set()
        for w in workers:
            w.join(timeout=10)
        pred_conn.close()
        print("  Exiting.")
        return

    # ── Prediction loop ────────────────────────────────────────────────
    pred_interval_sec = args.pred_interval * 60
    print(f"\n  Scheduler running — prediction every {args.pred_interval} min.  Ctrl-C to stop.\n")

    try:
        while True:
            time.sleep(pred_interval_sec)

            # Reconnect pred_conn if it dropped during sleep
            try:
                pred_conn.cursor().execute("SELECT 1")
            except Exception:
                print("  Prediction DB connection lost — reconnecting …")
                try:
                    pred_conn.close()
                except Exception:
                    pass
                pred_conn = _db_connect()
                _ensure_schema(pred_conn)

            _run_all_predictions(pred_conn, targets, args.verbose)

    except KeyboardInterrupt:
        print("\n\n  Ctrl-C received — shutting down …")

    finally:
        stop_event.set()
        print(f"  Stopping {len(workers)} camera worker(s) …")
        for w in workers:
            w.join(timeout=15)
        pred_conn.close()
        print("  All done.")


if __name__ == "__main__":
    main()