"""
Macedonia Border Crossings — Per-Vehicle Tracking Edition
=========================================================
Streams live HLS video from any of the 6 Macedonian border crossing cameras,
runs YOLOv8 vehicle detection with ByteTrack, and saves each individual
vehicle's full lifetime (entered_at / exited_at / lane / type) to PostgreSQL.

Key changes vs original:
  - Uses yolov8m by default (better edge-lane detection; free open-source weights)
  - Tracks every individual vehicle with a stable ID via BotSort
  - Saves each vehicle to `vehicle_crossings` when it leaves the frame
  - Periodic snapshots still saved to `snapshots` for aggregate queries
  - Confidence and frame count recorded per vehicle for quality filtering

Usage:
    python border_crossings.py                        # default: bogorodica
    python border_crossings.py --crossing blace
    python border_crossings.py --model yolov8s        # faster, slightly less accurate
    python border_crossings.py --model yolov8l        # more accurate, slower
    python border_crossings.py --conf 0.30            # lower = catch more at edges
    python border_crossings.py --interval 5           # snapshot interval in minutes
    python border_crossings.py --save                 # record to MP4
    python border_crossings.py --calibrate            # raw frame for lane tuning
    python border_crossings.py --list                 # show all crossings

Controls:
    Q / ESC  →  quit
    P        →  pause / unpause
    S        →  screenshot
    L        →  toggle lane overlay
    D        →  force a DB snapshot now
    T        →  print active vehicle track count
"""

import argparse
import contextlib
import os
import sys
import tempfile
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import cv2
import numpy as np
import psycopg2
import psycopg2.extras
import requests
from config import DB_CONFIG, DEFAULT_SNAPSHOT_INTERVAL_MIN, STREAM_BASE_URL
from crossings_db import ensure_schema, get_crossing_id, load_crossings

WAIT_MODEL_DIR = Path(__file__).resolve().parent / "models_v3"

VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

LANE_COLORS = [
    (0,   200, 255),
    (0,   255, 100),
    (255, 180,   0),
    (180,   0, 255),
    (0,   180, 255),
    (255,  50, 150),
]

# ---------------------------------------------------------------------------
# Minimum frames a vehicle must be tracked before we save it.
# At ~10fps, 15 frames = ~1.5 seconds — filters out false positives.
# ---------------------------------------------------------------------------
MIN_FRAMES   = 15
# Maximum realistic wait time in seconds (2 hours)
MAX_DURATION = 7200

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def init_db() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(**DB_CONFIG)
    ensure_schema(conn)
    return conn


def save_snapshot(conn, crossing_name: str, lane_counts: dict,
                  fps: float, interval_minutes: int, stream_ok: bool) -> dict | None:
    if not stream_ok:
        print("[DB] Skipping snapshot — stream not live.")
        return None

    crossing_id = get_crossing_id(conn, crossing_name)
    if not crossing_id:
        return None

    totals = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    grand_total = 0
    for lc in lane_counts.values():
        grand_total += lc.get("total", 0)
        for vtype, cnt in lc.get("by_type", {}).items():
            totals[vtype] = totals.get(vtype, 0) + cnt

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO snapshots
                (crossing_id, captured_at, interval_minutes,
                 total_vehicles, cars, motorcycles, buses, trucks,
                 lane_breakdown, fps)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id, captured_at
        """, (
            crossing_id,
            datetime.now(timezone.utc),
            interval_minutes,
            grand_total,
            totals["car"], totals["motorcycle"],
            totals["bus"], totals["truck"],
            psycopg2.extras.Json(lane_counts),
            round(fps, 2),
        ))
        snapshot_id, captured_at = cur.fetchone()
    conn.commit()
    return {
        "snapshot_id": int(snapshot_id),
        "captured_at": captured_at,
        "queue": grand_total,
    }


def save_vehicle_track(conn, crossing_name: str, track: dict) -> bool:
    crossing_id = get_crossing_id(conn, crossing_name)
    if not crossing_id:
        return False

    duration = (track["exited_at"] - track["entered_at"]).total_seconds()
    avg_conf = (
        track["confidence_sum"] / track["frame_count"]
        if track["frame_count"] > 0 else None
    )

    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO vehicle_crossings
                    (crossing_id, track_id, vehicle_type, lane,
                     entered_at, exited_at, duration_sec,
                     frame_count, avg_confidence, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                crossing_id,
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
        conn.commit()
        return True
    except Exception as e:
        print(f"[DB ERROR] Failed to save track {track['track_id']}: {e}")
        conn.rollback()   # <-- critical: without this, connection stays in error state
        return False


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def point_in_polygon(px, py, poly):
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def scale_poly(poly, w, h):
    return [(int(x * w), int(y * h)) for x, y in poly]


def find_lane(cx_f: float, cy_f: float, lanes_cfg: dict) -> str | None:
    for lane_name, poly in lanes_cfg.items():
        if point_in_polygon(cx_f, cy_f, poly):
            return lane_name
    return None

# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

def build_url(crossing_name: str) -> str:
    return STREAM_BASE_URL.format(name=crossing_name)


def console_text(value: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return value.encode(encoding, errors="replace").decode(encoding)


@contextlib.contextmanager
def no_proxy_env():
    names = [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
        "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY",
    ]
    saved = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value


class SegmentPollingCapture:
    """
    Fallback capture for HLS streams when OpenCV/FFmpeg cannot open the HTTPS
    manifest directly. It polls the playlist, downloads the next TS segment
    locally, and lets OpenCV decode that local file.
    """
    def __init__(self, playlist_url: str):
        self.playlist_url = playlist_url
        self.session = requests.Session()
        self.session.trust_env = False
        self.cap = None
        self.temp_path = None
        self.current_segment = None
        self.opened = self._open_next_segment(initial=True)

    def isOpened(self):
        return self.opened

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.unlink(self.temp_path)
            except OSError:
                pass
            self.temp_path = None

    def set(self, *_args, **_kwargs):
        return False

    def get(self, prop_id):
        if self.cap is None:
            return 0
        return self.cap.get(prop_id)

    def read(self):
        while True:
            if self.cap is None:
                if not self._open_next_segment():
                    return False, None

            ret, frame = self.cap.read()
            if ret and frame is not None:
                return True, frame

            self.release()
            self.cap = None
            time.sleep(0.2)

    def _fetch_segment_candidates(self):
        response = self.session.get(self.playlist_url, timeout=10)
        response.raise_for_status()
        lines = [
            line.strip()
            for line in response.text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        return [urljoin(self.playlist_url, line) for line in lines]

    def _open_next_segment(self, initial: bool = False):
        attempts = 0
        while attempts < 4:
            attempts += 1
            try:
                candidates = self._fetch_segment_candidates()
            except Exception:
                time.sleep(1)
                continue

            if not candidates:
                time.sleep(1)
                continue

            ordered = list(reversed(candidates))
            if self.current_segment:
                ordered = [u for u in ordered if u != self.current_segment] + [self.current_segment]

            for seg_url in ordered:
                try:
                    data = self.session.get(seg_url, timeout=15).content
                except Exception:
                    continue
                if len(data) < 1024:
                    continue

                self.release()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ts")
                tmp.write(data)
                tmp.close()
                self.temp_path = tmp.name

                cap = cv2.VideoCapture(self.temp_path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.cap = cap
                        self.current_segment = seg_url
                        return True
                cap.release()
                if self.temp_path and os.path.exists(self.temp_path):
                    os.unlink(self.temp_path)
                    self.temp_path = None

            if initial:
                time.sleep(1)

        return False


def open_stream(url: str):
    with no_proxy_env():
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5_000)
        if cap.isOpened():
            return cap
        cap.release()

        fallback = SegmentPollingCapture(url)
        if fallback.isOpened():
            print("[STREAM] Using TS segment polling fallback.")
            return fallback

    raise RuntimeError(
        f"Could not open stream:\n  {url}\n"
        "Check your internet connection or try the URL in VLC."
    )

# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
def annotate_frame(frame, detections, lane_counts, lanes_cfg,
                   active_vehicles, show_lanes, fps_display,
                   display_name, next_snap_in):
    h, w = frame.shape[:2]
    out  = frame.copy()

    if show_lanes:
        for idx, (name, poly) in enumerate(lanes_cfg.items()):
            color = LANE_COLORS[idx % len(LANE_COLORS)]
            pts   = np.array(scale_poly(poly, w, h), dtype=np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)
            cv2.polylines(out, [pts], True, color, 2)
            cx = int(sum(p[0] for p in pts) / len(pts))
            cy = int(sum(p[1] for p in pts) / len(pts))
            count = lane_counts.get(name, {}).get("total", 0)
            cv2.putText(out, name, (cx - 36, cy - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            cv2.putText(out, f"{count} veh.", (cx - 28, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

    for (x1, y1, x2, y2, label, conf, cx, cy, track_id) in detections:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        id_str = f"#{track_id} {label} {conf:.2f}" if track_id is not None else f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(id_str, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 140, 0), -1)
        cv2.putText(out, id_str, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(out, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    total     = sum(lc.get("total", 0) for lc in lane_counts.values())
    active    = len(active_vehicles)
    ts        = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    banner    = (f"  {display_name}  |  {ts}  |"
                 f"  Queue: {total}  |  Tracked: {active}"
                 f"  |  FPS: {fps_display:.1f}"
                 f"  |  DB save in: {next_snap_in}s")
    cv2.rectangle(out, (0, 0), (w, 34), (20, 20, 20), -1)
    cv2.putText(out, banner, (6, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    hints = "  Q/ESC: quit   P: pause   S: screenshot   L: lanes   D: force DB save   T: track stats"
    cv2.rectangle(out, (0, h - 24), (w, h), (20, 20, 20), -1)
    cv2.putText(out, hints, (6, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    return out

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Macedonia border crossing — per-vehicle tracker")
    parser.add_argument("--crossing",  default="bogorodica")
    parser.add_argument("--list",      action="store_true")
    parser.add_argument("--url",       default=None)
    parser.add_argument("--model",     default="yolov8m",
                        help="yolov8n / yolov8s / yolov8m / yolov8l / yolov8x  (default: m)")
    parser.add_argument("--conf",      type=float, default=0.30,
                        help="YOLO confidence threshold — lower catches more at frame edges")
    parser.add_argument("--interval",  type=int,   default=DEFAULT_SNAPSHOT_INTERVAL_MIN,
                        help="Snapshot DB interval in minutes")
    parser.add_argument("--save",      action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.list:
        try:
            conn = init_db()
        except Exception as e:
            print(f"DB connection failed: {e}")
            sys.exit(1)
        print("\nAvailable border crossings:")
        for name, cfg in load_crossings(conn).items():
            print(f"  {name:<15} {console_text(cfg['display_name'])}")
        print()
        conn.close()
        return

    preload_conn = init_db()
    try:
        cfg = load_crossings(preload_conn).get(args.crossing)
    finally:
        preload_conn.close()

    if not cfg:
        print(f"Unknown crossing '{args.crossing}'. Use --list to see available crossings.")
        sys.exit(1)

    display_name = cfg["display_name"]
    lanes_cfg = cfg["lane_config"]
    stream_url = args.url or build_url(args.crossing)

    print(f"\n{'='*65}")
    print(f"  Crossing  : {console_text(display_name)}")
    print(f"  Stream    : {stream_url}")
    print(f"  DB        : {DB_CONFIG['dbname']} on {DB_CONFIG['host']}")
    print(f"  Model     : {args.model}  (conf ≥ {args.conf})")
    print(f"  Interval  : every {args.interval} min")
    print(f"{'='*65}\n")

    # ── Database ──────────────────────────────────────────────────────────
    try:
        conn = init_db()
        print("PostgreSQL connected.\n")
    except Exception as e:
        print(f"DB connection failed: {e}")
        sys.exit(1)

    crossings = load_crossings(conn)
    cfg_db = crossings.get(args.crossing)
    if cfg_db:
        display_name = cfg_db["display_name"]
        lanes_cfg = cfg_db["lane_config"]
    else:
        print(f"Crossing '{args.crossing}' was not found in the database metadata.")
        conn.close()
        sys.exit(1)

    # ── YOLO ──────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    model_name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    print("Model ready. Connecting to stream ...\n")

    # ── Open stream ───────────────────────────────────────────────────────
    try:
        cap = open_stream(stream_url)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1280
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

    writer = None
    if args.save:
        out_path = Path(f"{args.crossing}_out.mp4")
        writer   = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (fw, fh))
        print(f"Recording → {out_path}")

    win_name = console_text(f"MK Border - {display_name}")
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(fw, 1280), min(fh, 720))

    # ── State ─────────────────────────────────────────────────────────────
    show_lanes       = True
    paused           = False
    stream_ok        = False
    fps_display      = 0.0
    frame_times      = []
    screenshot_dir   = Path("screenshots")
    last_frame       = None
    last_lane_counts = {n: {"total": 0, "by_type": {}} for n in lanes_cfg}

    # active_vehicles: track_id → dict with per-frame accumulated data
    # {
    #   "entered_at":       datetime,
    #   "last_seen_at":     datetime,
    #   "lane":             str | None,   (lane on entry)
    #   "vehicle_type":     str,
    #   "frame_count":      int,
    #   "confidence_sum":   float,
    #   "notes":            str | None,
    #   "lane_history":     list[str],    (for detecting lane switches)
    # }
    active_vehicles: dict[int, dict] = {}

    # Stats
    total_saved   = 0
    total_skipped = 0

    interval_sec = args.interval * 60
    last_snap_t  = time.time()

    print("Stream open. Press Q or ESC to quit.\n")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused." if paused else "Resumed.")
        elif key == ord('l'):
            show_lanes = not show_lanes
        elif key == ord('s'):
            screenshot_dir.mkdir(exist_ok=True)
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
            spath = screenshot_dir / f"{args.crossing}_{ts}.jpg"
            if last_frame is not None:
                cv2.imwrite(str(spath), last_frame)
                print(f"Screenshot saved → {spath}")
        elif key == ord('d'):
            snap = save_snapshot(conn, args.crossing, last_lane_counts,
                              fps_display, args.interval, stream_ok)
            last_snap_t = time.time()
            print(f"[DB] Manual snapshot saved  (queue={snap['queue'] if snap else None})")
        elif key == ord('t'):
            print(f"\n[TRACKS] Active: {len(active_vehicles)}  "
                  f"Saved: {total_saved}  Skipped: {total_skipped}")
            for tid, v in list(active_vehicles.items())[:10]:
                age = (datetime.now(timezone.utc) - v["entered_at"]).total_seconds()
                print(f"  ID {tid:>5}  type={v['vehicle_type']:<12}"
                      f"  lane={str(v['lane']):<18}  frames={v['frame_count']:>4}"
                      f"  age={age:.0f}s")
            print()

        # ── Periodic auto-snapshot ─────────────────────────────────────
        now = time.time()
        if not paused and (now - last_snap_t) >= interval_sec:
            snap = save_snapshot(conn, args.crossing, last_lane_counts,
                              fps_display, args.interval, stream_ok)
            last_snap_t = now
            if snap is not None:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[DB] {ts}  Snapshot  crossing={args.crossing}  queue={snap['queue']}"
                      f"  active_tracks={len(active_vehicles)}")

        next_snap_in = max(0, int(interval_sec - (time.time() - last_snap_t)))

        if paused:
            if last_frame is not None:
                disp = last_frame.copy()
                cv2.putText(disp, "  PAUSED  ", (fw // 2 - 80, fh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 60, 255), 3, cv2.LINE_AA)
                cv2.imshow(win_name, disp)
            time.sleep(0.05)
            continue

        # ── Read frame ────────────────────────────────────────────────
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret or frame is None:
            stream_ok = False
            cap.release()
            retry = 0
            while True:
                # Let the user quit even while reconnecting
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    print("\nQuitting during reconnect …")
                    conn.close()
                    cv2.destroyAllWindows()
                    sys.exit(0)

                retry += 1
                wait = min(2 ** retry, 60)  # exponential back-off, cap at 60 s
                print(f"Stream lost. Reconnect attempt {retry} (waiting {wait}s) …")

                # Show a "reconnecting" overlay so the window doesn't freeze
                if last_frame is not None:
                    disp = last_frame.copy()
                else:
                    disp = np.zeros((fh, fw, 3), dtype=np.uint8)
                msg = f"Stream lost — reconnect attempt {retry} …  (Q to quit)"
                cv2.putText(disp, msg, (20, fh // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 255), 2, cv2.LINE_AA)
                cv2.imshow(win_name, disp)

                time.sleep(wait)
                try:
                    cap = open_stream(stream_url)
                    stream_ok = True
                    print(f"Reconnected after {retry} attempt(s).\n")
                    break
                except RuntimeError as e:
                    print(f"  → Failed: {e}")
            continue

        stream_ok = True

        if args.calibrate:
            cv2.imshow(win_name, frame)
            continue

        # ── YOLO tracking (BotSort gives better re-ID across occlusions) ─
        h, w    = frame.shape[:2]
        results = model.track(
            frame,
            conf=args.conf,
            persist=True,
            tracker="botsort.yaml",   # better re-identification than bytetrack
            verbose=False,
            iou=0.30,
        )[0]

        detections   = []
        lane_counts  = {n: {"total": 0, "by_type": {}} for n in lanes_cfg}
        now_dt       = datetime.now(timezone.utc)
        current_ids  = set()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            label    = VEHICLE_CLASSES[cls_id]
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2
            cx_f, cy_f = cx / w, cy / h

            detections.append((x1, y1, x2, y2, label, conf, cx, cy, track_id))

            # ── Lane counting ─────────────────────────────────────────
            lane_hit = find_lane(cx_f, cy_f, lanes_cfg)
            if lane_hit:
                lane_counts[lane_hit]["total"] += 1
                lane_counts[lane_hit]["by_type"][label] = (
                    lane_counts[lane_hit]["by_type"].get(label, 0) + 1
                )

            # ── Per-vehicle tracking ───────────────────────────────────
            if track_id is None:
                continue

            current_ids.add(track_id)

            if track_id not in active_vehicles:
                # New vehicle — record its entry
                active_vehicles[track_id] = {
                    "entered_at":     now_dt,
                    "last_seen_at":   now_dt,
                    "lane":           lane_hit,
                    "vehicle_type":   label,
                    "frame_count":    1,
                    "confidence_sum": conf,
                    "notes":          None,
                    "lane_history":   [lane_hit] if lane_hit else [],
                }
                print(f"[NEW]  ID {track_id:>5}  {label:<12}  lane={lane_hit}")
            else:
                v = active_vehicles[track_id]
                v["last_seen_at"]   = now_dt
                v["frame_count"]   += 1
                v["confidence_sum"] += conf

                # Detect lane switches
                if lane_hit and (not v["lane_history"] or v["lane_history"][-1] != lane_hit):
                    v["lane_history"].append(lane_hit)
                    if len(v["lane_history"]) > 1:
                        v["notes"] = f"lane_switch:{'>'.join(v['lane_history'])}"

                # Set lane to whichever lane was most occupied (use first detected)
                if v["lane"] is None and lane_hit:
                    v["lane"] = lane_hit

        # ── Flush vehicles that left the frame ─────────────────────────
        disappeared = set(active_vehicles.keys()) - current_ids
        for tid in disappeared:
            v        = active_vehicles.pop(tid)
            duration = (v["last_seen_at"] - v["entered_at"]).total_seconds()

            # Quality gate: must have enough frames and a plausible duration
            if v["frame_count"] < MIN_FRAMES:
                total_skipped += 1
                continue
            if duration > MAX_DURATION:
                total_skipped += 1
                continue


            track = {
                "track_id":       tid,
                "vehicle_type":   v["vehicle_type"],
                "lane":           v["lane"],
                "entered_at":     v["entered_at"],
                "exited_at":      v["last_seen_at"],
                "frame_count":    v["frame_count"],
                "confidence_sum": v["confidence_sum"],
                "notes":          v.get("notes"),
            }

            ok = save_vehicle_track(conn, args.crossing, track)
            if ok:
                total_saved += 1
                print(f"[SAVED] ID {tid:>5}  {v['vehicle_type']:<12}"
                      f"  lane={str(v['lane']):<18}"
                      f"  frames={v['frame_count']:>4}"
                      f"  duration={duration:.0f}s"
                      f"  notes={v.get('notes') or '-'}")
            else:
                total_skipped += 1

        # ── FPS ───────────────────────────────────────────────────────
        frame_times.append(time.time())
        frame_times  = [t for t in frame_times if time.time() - t < 2.0]
        fps_display  = len(frame_times) / 2.0

        annotated = annotate_frame(
            frame, detections, lane_counts, lanes_cfg,
            active_vehicles, show_lanes, fps_display,
            display_name, next_snap_in,
        )
        last_frame       = annotated
        last_lane_counts = lane_counts

        cv2.imshow(win_name, annotated)
        if writer:
            writer.write(annotated)

    # ── Shutdown: save any still-active tracks ─────────────────────────
    print(f"\nFlushing {len(active_vehicles)} still-active tracks …")
    flush_dt = datetime.now(timezone.utc)
    for tid, v in active_vehicles.items():
        duration = (flush_dt - v["entered_at"]).total_seconds()
        if v["frame_count"] < MIN_FRAMES or duration > MAX_DURATION:
            continue


        save_vehicle_track(conn, args.crossing, {
            "track_id":       tid,
            "vehicle_type":   v["vehicle_type"],
            "lane":           v["lane"],
            "entered_at":     v["entered_at"],
            "exited_at":      flush_dt,
            "frame_count":    v["frame_count"],
            "confidence_sum": v["confidence_sum"],
            "notes":          (v.get("notes") or "") + " [stream_end]",
        })

    # ── Cleanup ───────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
        print(f"Video saved → {args.crossing}_out.mp4")
    conn.close()
    cv2.destroyAllWindows()
    print(f"\nDone. Saved: {total_saved}  Skipped: {total_skipped}")


if __name__ == "__main__":
    main()
