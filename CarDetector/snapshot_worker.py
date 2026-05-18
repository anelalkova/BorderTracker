"""
Headless snapshot worker — all border crossings
================================================
Grabs one frame from each crossing's HLS stream, runs YOLOv8 vehicle
detection (no tracking, no GUI), counts vehicles per lane, and saves a
snapshot row to PostgreSQL — exactly the same schema used by border_crossings.py.

Usage:
    python snapshot_worker.py                        # all crossings, 5-min interval
    python snapshot_worker.py --interval 2           # every 2 minutes
    python snapshot_worker.py --crossing blace       # single crossing only
    python snapshot_worker.py --once                 # one round then exit
    python snapshot_worker.py --model yolov8s        # faster model
    python snapshot_worker.py --conf 0.30
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import cv2

from border_crossings import (
    CROSSINGS,
    VEHICLE_CLASSES,
    init_db,
    open_stream,
    find_lane,
    save_snapshot,
    build_url,
)


# ---------------------------------------------------------------------------
# Grab a single usable frame from an HLS stream, then close it immediately.
# Reads a few frames to let the decoder warm up before sampling.
# ---------------------------------------------------------------------------
WARMUP_FRAMES  = 5
MAX_RETRIES    = 5          # give up after this many failed attempts per crossing
RETRY_BASE_SEC = 5          # first retry wait; doubles each time (5, 10, 20, 40, 60)
RETRY_CAP_SEC  = 60         # backoff ceiling


def grab_frame(stream_url: str) -> "cv2.Mat | None":
    """
    Open the stream, discard WARMUP_FRAMES, return the next good frame.
    On failure retries up to MAX_RETRIES times with exponential backoff.
    Returns None only after all retries are exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        cap = None
        try:
            cap = open_stream(stream_url)
        except RuntimeError as exc:
            print(f"    [STREAM] Attempt {attempt}/{MAX_RETRIES} — could not open: {exc}")
        else:
            frame = None
            try:
                for _ in range(WARMUP_FRAMES + 1):
                    ret, f = cap.read()
                    if ret and f is not None:
                        frame = f
            except Exception as exc:
                print(f"    [STREAM] Attempt {attempt}/{MAX_RETRIES} — read error: {exc}")
            finally:
                cap.release()

            if frame is not None:
                if attempt > 1:
                    print(f"    [STREAM] Reconnected on attempt {attempt}.")
                return frame

            print(f"    [STREAM] Attempt {attempt}/{MAX_RETRIES} — stream opened but no frame returned.")

        if attempt < MAX_RETRIES:
            wait = min(RETRY_BASE_SEC * (2 ** (attempt - 1)), RETRY_CAP_SEC)
            print(f"    [STREAM] Retrying in {wait}s …")
            time.sleep(wait)

    print(f"    [STREAM] Gave up after {MAX_RETRIES} attempts.")
    return None


# ---------------------------------------------------------------------------
# Run YOLO on one frame, return lane_counts dict compatible with save_snapshot.
# ---------------------------------------------------------------------------

def detect_vehicles(model, frame, lanes_cfg: dict, conf_threshold: float) -> dict:
    """Returns {lane_name: {"total": int, "by_type": {vehicle_type: count}}}."""
    h, w = frame.shape[:2]
    lane_counts = {name: {"total": 0, "by_type": {}} for name in lanes_cfg}

    results = model.predict(
        frame,
        conf=conf_threshold,
        verbose=False,
        iou=0.30,
    )[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in VEHICLE_CLASSES:
            continue

        label = VEHICLE_CLASSES[cls_id]
        x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
        cx_f = ((x1 + x2) / 2) / w
        cy_f = ((y1 + y2) / 2) / h

        lane_hit = find_lane(cx_f, cy_f, lanes_cfg)
        if lane_hit:
            lane_counts[lane_hit]["total"] += 1
            lane_counts[lane_hit]["by_type"][label] = (
                lane_counts[lane_hit]["by_type"].get(label, 0) + 1
            )

    return lane_counts


# ---------------------------------------------------------------------------
# Process one crossing: grab frame → detect → save snapshot.
# ---------------------------------------------------------------------------

def process_crossing(conn, model, crossing_name: str, conf: float, interval_minutes: int) -> bool:
    cfg         = CROSSINGS[crossing_name]
    stream_url  = build_url(crossing_name)
    lanes_cfg   = cfg["lanes"]

    print(f"  [{crossing_name}] Grabbing frame …")
    frame = grab_frame(stream_url)
    if frame is None:
        print(f"  [{crossing_name}] No frame — skipping.")
        return False

    lane_counts = detect_vehicles(model, frame, lanes_cfg, conf)

    total = sum(lc["total"] for lc in lane_counts.values())
    print(f"  [{crossing_name}] Detected {total} vehicles across {len(lanes_cfg)} lanes.")

    snap = save_snapshot(
        conn,
        crossing_name=crossing_name,
        lane_counts=lane_counts,
        fps=0.0,                    # not meaningful for single-frame snapshots
        interval_minutes=interval_minutes,
        stream_ok=True,
    )

    if snap:
        print(f"  [{crossing_name}] Snapshot #{snap['snapshot_id']} saved  "
              f"(queue={snap['queue']}  at={snap['captured_at'].strftime('%H:%M:%S')})")
        return True
    else:
        print(f"  [{crossing_name}] save_snapshot returned None.")
        return False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(crossing_names: list[str], interval_minutes: int, model_name: str,
        conf: float, run_once: bool) -> None:

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)

    mname = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
    print(f"Loading {mname} …")
    model = YOLO(mname)
    print("Model ready.\n")

    conn = init_db()
    print("PostgreSQL connected.\n")

    interval_sec = interval_minutes * 60

    try:
        while True:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{ts}] Starting snapshot round ({len(crossing_names)} crossings) …")

            for name in crossing_names:
                try:
                    process_crossing(conn, model, name, conf, interval_minutes)
                except Exception as exc:
                    conn.rollback()
                    print(f"  [{name}] ERROR: {exc}")

            print()

            if run_once:
                break

            next_ts = datetime.fromtimestamp(time.time() + interval_sec).strftime("%H:%M:%S")
            print(f"Next round at {next_ts}  (sleeping {interval_minutes} min …)\n")
            time.sleep(interval_sec)

    finally:
        conn.close()
        print("DB connection closed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Headless YOLOv8 snapshot worker for all MK border crossings")
    parser.add_argument("--crossing",  choices=list(CROSSINGS.keys()),
                        help="Single crossing (default: all)")
    parser.add_argument("--interval",  type=int, default=5,
                        help="Snapshot interval in minutes (default: 5)")
    parser.add_argument("--model",     default="yolov8m",
                        help="YOLO model: yolov8n/s/m/l/x  (default: yolov8m)")
    parser.add_argument("--conf",      type=float, default=0.30,
                        help="YOLO confidence threshold (default: 0.30)")
    parser.add_argument("--once",      action="store_true",
                        help="Run one round then exit")
    return parser.parse_args()


def main():
    args = parse_args()
    crossing_names = [args.crossing] if args.crossing else list(CROSSINGS.keys())

    print(f"\n{'='*60}")
    print(f"  Crossings : {', '.join(crossing_names)}")
    print(f"  Interval  : every {args.interval} min")
    print(f"  Model     : {args.model}  (conf >= {args.conf})")
    print(f"  Mode      : {'once' if args.once else 'continuous'}")
    print(f"{'='*60}\n")

    run(
        crossing_names=crossing_names,
        interval_minutes=args.interval,
        model_name=args.model,
        conf=args.conf,
        run_once=args.once,
    )


if __name__ == "__main__":
    main()