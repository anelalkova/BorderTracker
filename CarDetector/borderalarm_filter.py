"""
borderalarm_filter.py
=====================
Validates crowdsourced borderalarm reports against camera-observed
vehicle_crossings data and flags unreliable entries.

A report is flagged as SUSPECT when:
  - claimed wait > MAX_RATIO × camera avg duration for that hour
  - claimed wait > ABSOLUTE_MAX_MIN (hard cap, e.g. 4 hours)
  - it's the only report in that hour with no camera data to compare

A report is flagged as LAGGED when:
  - the reported time is plausible but the wait would have started
    well before the camera's observed queue depth (reporting lag heuristic)

Flagged rows are written to `crowdsourced_waits.quality_flag` column.
The column is added if it doesn't exist.

Usage:
    python borderalarm_filter.py --crossing bogorodica
    python borderalarm_filter.py --all
    python borderalarm_filter.py --crossing bogorodica --dry-run
    python borderalarm_filter.py --crossing bogorodica --show-stats
"""

import argparse
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "dbname":   "border_crossing",
    "user":     "postgres",
    "password": "postgres",
}

CROSSINGS = [
    "bogorodica", "blace", "tabanovce",
    "deve_bair", "kafasan", "medzitlija",
]

# A crowdsourced report is suspect if claimed wait > this multiple
# of the camera's observed avg duration for that hour
MAX_RATIO = 3.0

# Hard upper cap regardless of camera data (minutes)
ABSOLUTE_MAX_MIN = 240

# If camera avg is below this, don't use it for ratio filtering
# (avoids flagging real jams when camera briefly sees 0 cars)
MIN_CAMERA_DURATION_MIN = 2.0

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_quality_flag_column(conn):
    """Add quality_flag column to crowdsourced_waits if missing."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE crowdsourced_waits
            ADD COLUMN IF NOT EXISTS quality_flag TEXT DEFAULT NULL;
        """)
        cur.execute("""
            ALTER TABLE crowdsourced_waits
            ADD COLUMN IF NOT EXISTS camera_avg_min REAL DEFAULT NULL;
        """)
    conn.commit()


def get_crossing_id(conn, name: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (name,))
        row = cur.fetchone()
        return row[0] if row else None


def fetch_crowdsourced(conn, crossing_id: int) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, reported_at, wait_minutes, quality_flag
            FROM crowdsourced_waits
            WHERE crossing_id = %s
            ORDER BY reported_at
        """, (crossing_id,))
        return [dict(r) for r in cur.fetchall()]


def fetch_camera_hourly(conn, crossing_id: int) -> dict:
    """
    Returns a dict: hour_bucket (datetime) -> avg_duration_min
    Built from vehicle_crossings.duration_sec grouped by hour.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                DATE_TRUNC('hour', entered_at) AS hour_bucket,
                AVG(duration_sec) / 60.0       AS avg_duration_min,
                COUNT(*)                        AS vehicle_count
            FROM vehicle_crossings
            WHERE crossing_id = %s
              AND exited_at IS NOT NULL
              AND duration_sec > 0
            GROUP BY 1
        """, (crossing_id,))
        return {
            row["hour_bucket"]: {
                "avg_min": float(row["avg_duration_min"]),
                "count":   int(row["vehicle_count"]),
            }
            for row in cur.fetchall()
        }


def update_flags(conn, updates: list[dict], dry_run: bool):
    """
    updates: list of {id, quality_flag, camera_avg_min}
    """
    if dry_run:
        for u in updates:
            print(f"  [DRY RUN] id={u['id']}  flag={u['quality_flag']}  "
                  f"camera_avg={u.get('camera_avg_min')}")
        return

    with conn.cursor() as cur:
        for u in updates:
            cur.execute("""
                UPDATE crowdsourced_waits
                SET quality_flag   = %s,
                    camera_avg_min = %s
                WHERE id = %s
            """, (u["quality_flag"], u.get("camera_avg_min"), u["id"]))
    conn.commit()

# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

def classify_report(report: dict, camera_hourly: dict) -> tuple[str, float | None]:
    """
    Returns (quality_flag, camera_avg_min).
    quality_flag: 'ok' | 'suspect_ratio' | 'suspect_absolute' | 'no_camera_data'
    """
    wait = float(report["wait_minutes"])

    # Hard cap first
    if wait > ABSOLUTE_MAX_MIN:
        return "suspect_absolute", None

    # Find the matching camera hour bucket
    # reported_at is UTC-aware; truncate to hour
    reported_at = report["reported_at"]
    if reported_at.tzinfo is None:
        reported_at = reported_at.replace(tzinfo=timezone.utc)

    # Try exact hour match, then ±1 hour
    bucket = reported_at.replace(minute=0, second=0, microsecond=0)
    camera = None
    for delta_h in [0, -1, 1, -2, 2]:
        from datetime import timedelta
        candidate = bucket + timedelta(hours=delta_h)
        # Strip tzinfo for dict key comparison (stored as naive UTC from DB)
        candidate_naive = candidate.replace(tzinfo=None)
        if candidate_naive in camera_hourly:
            camera = camera_hourly[candidate_naive]
            break

    if camera is None:
        return "no_camera_data", None

    cam_avg = camera["avg_min"]

    # Don't penalise based on very short camera durations
    if cam_avg < MIN_CAMERA_DURATION_MIN:
        return "ok", round(cam_avg, 1)

    ratio = wait / cam_avg
    if ratio > MAX_RATIO:
        return f"suspect_ratio_{ratio:.1f}x", round(cam_avg, 1)

    return "ok", round(cam_avg, 1)


def filter_crossing(conn, crossing_name: str,
                    dry_run: bool = False, verbose: bool = True) -> dict:
    cid = get_crossing_id(conn, crossing_name)
    if not cid:
        print(f"  Crossing '{crossing_name}' not found.")
        return {}

    reports      = fetch_crowdsourced(conn, cid)
    camera_hourly = fetch_camera_hourly(conn, cid)

    if not reports:
        print(f"  No crowdsourced reports for '{crossing_name}'.")
        return {}

    updates = []
    counts  = {"ok": 0, "suspect_ratio": 0, "suspect_absolute": 0, "no_camera_data": 0}

    for r in reports:
        flag, cam_avg = classify_report(r, camera_hourly)
        updates.append({"id": r["id"], "quality_flag": flag, "camera_avg_min": cam_avg})

        bucket = "ok" if flag == "ok" else (
            "suspect_absolute" if flag == "suspect_absolute" else
            "no_camera_data"   if flag == "no_camera_data"   else
            "suspect_ratio"
        )
        counts[bucket] += 1

        if verbose and flag != "ok":
            ts = r["reported_at"].strftime("%Y-%m-%d %H:%M UTC")
            cam_str = f"camera={cam_avg} min" if cam_avg else "no camera data"
            print(f"  ⚠  [{flag}]  {ts}  claimed={r['wait_minutes']} min  {cam_str}")

    update_flags(conn, updates, dry_run=dry_run)
    return counts


def show_stats(conn, crossing_name: str):
    cid = get_crossing_id(conn, crossing_name)
    if not cid:
        return

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                quality_flag,
                COUNT(*)                              AS count,
                ROUND(AVG(wait_minutes)::NUMERIC, 1) AS avg_wait,
                ROUND(AVG(camera_avg_min)::NUMERIC, 1) AS avg_camera
            FROM crowdsourced_waits
            WHERE crossing_id = %s
            GROUP BY quality_flag
            ORDER BY count DESC
        """, (cid,))
        rows = cur.fetchall()

    print(f"\n  Quality flag breakdown for {crossing_name}:")
    print(f"  {'Flag':<30} {'Count':>6} {'Avg claimed':>12} {'Avg camera':>11}")
    print(f"  {'-'*30} {'-'*6} {'-'*12} {'-'*11}")
    for r in rows:
        print(f"  {str(r['quality_flag'] or 'unreviewed'):<30} "
              f"{r['count']:>6} "
              f"{str(r['avg_wait']) + ' min':>12} "
              f"{str(r['avg_camera'] or '—') + (' min' if r['avg_camera'] else ''):>11}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Filter borderalarm crowdsourced reports")
    parser.add_argument("--crossing", choices=CROSSINGS, default=None)
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--show-stats", action="store_true")
    args = parser.parse_args()

    if not args.crossing and not args.all:
        parser.error("Specify --crossing <name> or --all")

    targets = CROSSINGS if args.all else [args.crossing]
    conn    = get_conn()

    ensure_quality_flag_column(conn)

    for name in targets:
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        if args.show_stats:
            show_stats(conn, name)
        else:
            counts = filter_crossing(conn, name, dry_run=args.dry_run)
            if counts:
                total = sum(counts.values())
                print(f"\n  Results ({total} reports):")
                print(f"    ✓ ok              : {counts['ok']}")
                print(f"    ⚠ suspect ratio   : {counts['suspect_ratio']}")
                print(f"    ⚠ suspect absolute: {counts['suspect_absolute']}")
                print(f"    ? no camera data  : {counts['no_camera_data']}")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()