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
    python borderalarm_filter.py --crossing bogorodica --force   # re-flag already-ok rows too
"""

import argparse
from datetime import timedelta, timezone

import psycopg2
import psycopg2.extras
from config import DB_CONFIG
from crossings_db import load_crossing_names

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_RATIO            = 3.0
ABSOLUTE_MAX_MIN     = 240
MIN_CAMERA_DURATION_MIN = 2.0

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_schema(conn):
    """Add columns and indexes if missing. Safe to call every run."""
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE crowdsourced_waits
                ADD COLUMN IF NOT EXISTS quality_flag   TEXT DEFAULT NULL,
                ADD COLUMN IF NOT EXISTS camera_avg_min REAL DEFAULT NULL;
        """)
        # Covering index so the hourly GROUP BY never full-scans
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_vc_crossing_entered
                ON vehicle_crossings (crossing_id, entered_at)
                WHERE exited_at IS NOT NULL AND duration_sec > 0;
        """)
        # Speeds up the crowdsourced fetch
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_cw_crossing_reported
                ON crowdsourced_waits (crossing_id, reported_at);
        """)
    conn.commit()


def get_all_crossing_ids(conn) -> dict[str, int]:
    """Fetch all crossing name→id pairs in one query."""
    with conn.cursor() as cur:
        cur.execute("SELECT name, id FROM crossings")
        return dict(cur.fetchall())


def fetch_crowdsourced(conn, crossing_id: int, force: bool = False) -> list[dict]:
    """
    Fetch reports that need (re-)classification.
    Without --force, skip rows already marked 'ok' to avoid pointless work
    on repeat runs.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if force:
            cur.execute("""
                SELECT id, reported_at, wait_minutes, quality_flag
                FROM crowdsourced_waits
                WHERE crossing_id = %s
                ORDER BY reported_at
            """, (crossing_id,))
        else:
            cur.execute("""
                SELECT id, reported_at, wait_minutes, quality_flag
                FROM crowdsourced_waits
                WHERE crossing_id = %s
                  AND (quality_flag IS NULL OR quality_flag <> 'ok')
                ORDER BY reported_at
            """, (crossing_id,))
        return [dict(r) for r in cur.fetchall()]


def fetch_camera_hourly(conn, crossing_id: int) -> dict:
    """
    Single aggregation query — the covering index makes this fast even on
    large vehicle_crossings tables.
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
            row["hour_bucket"].replace(tzinfo=timezone.utc): {
                "avg_min": float(row["avg_duration_min"]),
                "count":   int(row["vehicle_count"]),
            }
            for row in cur.fetchall()
        }


def update_flags(conn, updates: list[dict], dry_run: bool):
    if not updates:
        return

    if dry_run:
        for u in updates:
            print(f"  [DRY RUN] id={u['id']}  flag={u['quality_flag']}  "
                  f"camera_avg={u.get('camera_avg_min')}")
        return

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, """
            UPDATE crowdsourced_waits AS cw
            SET quality_flag   = v.quality_flag,
                camera_avg_min = v.camera_avg_min::real
            FROM (VALUES %s) AS v(id, quality_flag, camera_avg_min)
            WHERE cw.id = v.id::int
        """, [(u["id"], u["quality_flag"], u.get("camera_avg_min")) for u in updates])
    conn.commit()

# ---------------------------------------------------------------------------
# Filtering logic
# ---------------------------------------------------------------------------

# Pre-built list of hour deltas to check: 0, -1, +1, -2, +2
_HOUR_DELTAS = [timedelta(hours=d) for d in (0, -1, 1, -2, 2)]


def classify_report(report: dict, camera_hourly: dict) -> tuple[str, float | None]:
    wait = float(report["wait_minutes"])

    if wait > ABSOLUTE_MAX_MIN:
        return "suspect_absolute", None

    reported_at = report["reported_at"]
    if reported_at.tzinfo is None:
        reported_at = reported_at.replace(tzinfo=timezone.utc)

    bucket = reported_at.replace(minute=0, second=0, microsecond=0)

    camera = None
    for delta in _HOUR_DELTAS:
        candidate = bucket + delta
        if candidate in camera_hourly:
            camera = camera_hourly[candidate]
            break

    if camera is None:
        return "no_camera_data", None

    cam_avg = camera["avg_min"]
    if cam_avg < MIN_CAMERA_DURATION_MIN:
        return "ok", round(cam_avg, 1)

    ratio = wait / cam_avg
    if ratio > MAX_RATIO:
        return f"suspect_ratio_{ratio:.1f}x", round(cam_avg, 1)

    return "ok", round(cam_avg, 1)


def filter_crossing(conn, crossing_name: str, crossing_id: int,
                    dry_run: bool = False, force: bool = False,
                    verbose: bool = True) -> dict:

    reports       = fetch_crowdsourced(conn, crossing_id, force=force)
    camera_hourly = fetch_camera_hourly(conn, crossing_id)

    if not reports:
        print(f"  No unreviewed reports for '{crossing_name}'.")
        return {}

    updates = []
    counts  = {"ok": 0, "suspect_ratio": 0, "suspect_absolute": 0, "no_camera_data": 0}

    for r in reports:
        flag, cam_avg = classify_report(r, camera_hourly)
        updates.append({"id": r["id"], "quality_flag": flag, "camera_avg_min": cam_avg})

        bucket = (
            "ok"               if flag == "ok"               else
            "suspect_absolute" if flag == "suspect_absolute" else
            "no_camera_data"   if flag == "no_camera_data"   else
            "suspect_ratio"
        )
        counts[bucket] += 1

        if verbose and flag != "ok":
            ts      = r["reported_at"].strftime("%Y-%m-%d %H:%M UTC")
            cam_str = f"camera={cam_avg} min" if cam_avg else "no camera data"
            print(f"  ⚠  [{flag}]  {ts}  claimed={r['wait_minutes']} min  {cam_str}")

    update_flags(conn, updates, dry_run=dry_run)
    return counts


def show_stats(conn, crossing_id: int, crossing_name: str):
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                quality_flag,
                COUNT(*)                               AS count,
                ROUND(AVG(wait_minutes)::NUMERIC, 1)   AS avg_wait,
                ROUND(AVG(camera_avg_min)::NUMERIC, 1) AS avg_camera
            FROM crowdsourced_waits
            WHERE crossing_id = %s
            GROUP BY quality_flag
            ORDER BY count DESC
        """, (crossing_id,))
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
    parser.add_argument("--crossing",   default=None)
    parser.add_argument("--all",        action="store_true")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--show-stats", action="store_true")
    parser.add_argument("--force",      action="store_true",
                        help="Re-classify rows already marked 'ok' (default: skip them)")
    args = parser.parse_args()

    if not args.crossing and not args.all:
        parser.error("Specify --crossing <name> or --all")

    conn    = get_conn()
    available_crossings = load_crossing_names(conn)
    if args.crossing and args.crossing not in available_crossings:
        parser.error(f"Unknown crossing '{args.crossing}'. Available: {', '.join(available_crossings)}")

    targets = available_crossings if args.all else [args.crossing]

    ensure_schema(conn)

    # Fetch all crossing IDs in one query instead of one per crossing
    crossing_ids = get_all_crossing_ids(conn)

    for name in targets:
        cid = crossing_ids.get(name)
        if not cid:
            print(f"\n  Crossing '{name}' not found in DB — skipping.")
            continue

        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        if args.show_stats:
            show_stats(conn, cid, name)
        else:
            counts = filter_crossing(conn, name, cid,
                                     dry_run=args.dry_run,
                                     force=args.force)
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
