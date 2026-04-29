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
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# PostgreSQL connection config
# ---------------------------------------------------------------------------
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "border_crossing",
    "user":     "postgres",
    "password": "postgres",
}

STREAM_BASE = "https://streaming1.neotel.net.mk/stream/{name}.m3u8"

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

CROSSINGS = {
    "bogorodica": {
        "display_name": "Bogorodica (МК–ГР)",
        "neighbor": "Greece",
        "lanes": {
            "Bogorodica L1": [(0.32, 0.12), (0.39, 0.14), (0.00, 0.53), (0.00, 0.25)],
            "Bogorodica L2": [(0.39, 0.16), (0.43, 0.16), (0.05, 0.94), (0.00, 0.68)],
            "Bogorodica L3": [(0.43, 0.16), (0.46, 0.16), (0.26, 0.94), (0.05, 0.94)],
            "Bogorodica L4": [(0.50, 0.16), (0.54, 0.16), (0.93, 0.94), (0.73, 0.94)],
            "Bogorodica L5": [(0.54, 0.16), (0.59, 0.16), (1.00, 0.7), (0.93, 0.94)],
        },
    },
    "blace": {
        "display_name": "Blace (МК–КС)",
        "neighbor": "Kosovo",
        "lanes": {
            "Blace L1": [(0.487, 0.13), (0.440, 0.13), (0.000, 0.686), (0.22, 0.95)],
            "Blace L2": [(0.53, 0.13), (0.487, 0.13), (0.22, 0.95), (0.58, 0.95)],
            "Blace L3": [(0.56, 0.13), (0.53, 0.13), (0.58, 0.95), (0.95, 0.95)],
        },
    },
    "tabanovce": {
        "display_name": "Tabanovce (МК–СР)",
        "neighbor": "Serbia",
        "lanes": {
            "Tabanovce L1": [(0.55, 0.12), (0.58, 0.12), (0.00, 0.60), (0.00, 0.32)],
            "Tabanovce L2": [(0.60, 0.20), (0.66, 0.20), (0.14, 0.93), (0.00, 0.79)],
            "Tabanovce L3": [(0.52, 0.10), (0.68, 0.10), (0.88, 0.95), (0.48, 0.95)],
        },
    },
    "deve_bair": {
        "display_name": "Deve Bair (МК–БГ)",
        "neighbor": "Bulgaria",
        "lanes": {
            "DeveBair L1": [(0.32, 0.10), (0.51, 0.10), (0.43, 0.93), (0.00, 0.93)],
            "DeveBair L2": [(0.57, 0.10), (0.75, 0.10), (0.99, 0.64), (0.65, 0.93)],
        },
    },
    "kafasan": {
        "display_name": "Kafasan (МК–АЛ)",
        "neighbor": "Albania",
        "lanes": {
            "Kafasan L1": [(0.43, 0.2), (0.49, 0.2), (0.00, 0.81), (0.02, 0.52)],
            "Kafasan L2": [(0.49, 0.2), (0.55, 0.2), (0.30, 1), (0.00, 0.81)],
            "Kafasan L3": [(0.55, 0.2), (0.58, 0.2), (1.00, 1), (0.72, 1)],
        },
    },
    "medzitlija": {
        "display_name": "Megjitlija (МК–ГР)",
        "neighbor": "Greece",
        "lanes": {
            "Medzitlija L1": [(0.22, 0.10), (0.42, 0.10), (0.08, 0.92), (0.00, 0.92)],
            "Medzitlija L2": [(0.42, 0.10), (0.60, 0.10), (0.55, 0.92), (0.08, 0.92)],
            "Medzitlija L3": [(0.60, 0.10), (0.78, 0.10), (0.95, 0.92), (0.55, 0.92)],
        },
    },
}

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
    with conn.cursor() as cur:
        for name, cfg in CROSSINGS.items():
            cur.execute("""
                INSERT INTO crossings (name, display_name, neighbor)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO NOTHING
            """, (name, cfg["display_name"], cfg["neighbor"]))
    conn.commit()
    return conn


def get_crossing_id(conn, crossing_name: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (crossing_name,))
        row = cur.fetchone()
        return row[0] if row else None


def save_snapshot(conn, crossing_name: str, lane_counts: dict,
                  fps: float, interval_minutes: int, stream_ok: bool) -> int | None:
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
    conn.commit()
    return grand_total


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
    return STREAM_BASE.format(name=crossing_name)


def open_stream(url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10_000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC,  5_000)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open stream:\n  {url}\n"
            "Check your internet connection or try the URL in VLC."
        )
    return cap

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
    parser.add_argument("--crossing",  default="bogorodica",
                        choices=list(CROSSINGS.keys()))
    parser.add_argument("--list",      action="store_true")
    parser.add_argument("--url",       default=None)
    parser.add_argument("--model",     default="yolov8m",
                        help="yolov8n / yolov8s / yolov8m / yolov8l / yolov8x  (default: m)")
    parser.add_argument("--conf",      type=float, default=0.30,
                        help="YOLO confidence threshold — lower catches more at frame edges")
    parser.add_argument("--interval",  type=int,   default=5,
                        help="Snapshot DB interval in minutes")
    parser.add_argument("--save",      action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable border crossings:")
        for k, v in CROSSINGS.items():
            print(f"  {k:<15} {v['display_name']}")
        print()
        return

    cfg          = CROSSINGS[args.crossing]
    display_name = cfg["display_name"]
    lanes_cfg    = cfg["lanes"]
    stream_url   = args.url or build_url(args.crossing)

    print(f"\n{'='*65}")
    print(f"  Crossing  : {display_name}")
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

    # ── YOLO ──────────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    model_name = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    print(f"Loading {model_name} …")
    model = YOLO(model_name)
    print("Model ready. Connecting to stream …\n")

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

    win_name = f"MK Border – {display_name}"
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
            n = save_snapshot(conn, args.crossing, last_lane_counts,
                              fps_display, args.interval, stream_ok)
            last_snap_t = time.time()
            print(f"[DB] Manual snapshot saved  (queue={n})")
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
            n = save_snapshot(conn, args.crossing, last_lane_counts,
                              fps_display, args.interval, stream_ok)
            last_snap_t = now
            if n is not None:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[DB] {ts}  Snapshot  crossing={args.crossing}  queue={n}"
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
        if not ret or frame is None:
            print("Stream lost. Reconnecting …")
            stream_ok = False
            cap.release()
            time.sleep(2)
            try:
                cap       = open_stream(stream_url)
                stream_ok = True
            except RuntimeError as e:
                print(f"Reconnect failed: {e}")
                break
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