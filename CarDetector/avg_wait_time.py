"""
wait_time_from_crossings.py
============================
Estimates current border crossing wait times directly from the
vehicle_crossings table — no crowdsourced data or ML model needed.

The estimate is built from three signals, combined in order of recency:

  1. ACTIVE VEHICLES  — cars currently in lane (no exited_at yet).
                        Their elapsed time is a guaranteed minimum wait.

  2. RECENT COMPLETED — vehicles that finished crossing in the last N minutes.
                        Exponentially weighted so the freshest exits dominate.

  3. HOURLY BASELINE  — rolling average for this hour-of-day + day-of-week,
                        used when recent data is sparse (night, low traffic).

Usage:
    python wait_time_from_crossings.py --crossing bogorodica
    python wait_time_from_crossings.py --crossing bogorodica --window 30
    python wait_time_from_crossings.py --all-crossings
    python wait_time_from_crossings.py --crossing bogorodica --history
"""

import argparse
from datetime import datetime, timezone, timedelta

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

# How many minutes back to look for "recent" completed crossings
DEFAULT_WINDOW_MIN = 45

# Minimum number of completed crossings needed before we trust the estimate
MIN_SAMPLES_FOR_CONFIDENCE = 5

# Vehicle types considered "heavy" (slower processing at booth)
HEAVY_TYPES = {"bus", "truck"}

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def fetch_active_vehicles(conn, crossing_id: int) -> list[dict]:
    """Vehicles currently visible in a lane (entered but not yet exited)."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                track_id,
                vehicle_type,
                lane,
                entered_at,
                EXTRACT(EPOCH FROM (NOW() - entered_at)) AS elapsed_sec
            FROM vehicle_crossings
            WHERE crossing_id = %s
              AND exited_at IS NULL
              AND entered_at > NOW() - INTERVAL '3 hours'
            ORDER BY entered_at
        """, (crossing_id,))
        return [dict(r) for r in cur.fetchall()]


def fetch_recent_completed(conn, crossing_id: int,
                            window_min: int = DEFAULT_WINDOW_MIN) -> list[dict]:
    """Completed crossings within the last window_min minutes."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                track_id,
                vehicle_type,
                lane,
                entered_at,
                exited_at,
                duration_sec,
                EXTRACT(EPOCH FROM (NOW() - exited_at)) AS seconds_ago
            FROM vehicle_crossings
            WHERE crossing_id = %s
              AND exited_at IS NOT NULL
              AND duration_sec IS NOT NULL
              AND duration_sec > 0
              AND exited_at > NOW() - INTERVAL '%s minutes'
                AND lane IS NOT NULL
            ORDER BY exited_at DESC
        """, (crossing_id, window_min))
        return [dict(r) for r in cur.fetchall()]


def fetch_hourly_baseline(conn, crossing_id: int) -> dict | None:
    """
    Historical average duration for this hour-of-day and day-of-week.
    Used as a fallback when recent data is sparse.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                ROUND(AVG(duration_sec)::NUMERIC, 1)   AS avg_duration_sec,
                ROUND(STDDEV(duration_sec)::NUMERIC, 1) AS std_duration_sec,
                COUNT(*)                                AS sample_count
            FROM vehicle_crossings
            WHERE crossing_id = %s
              AND exited_at IS NOT NULL
              AND duration_sec IS NOT NULL
              AND duration_sec > 0
              AND EXTRACT(hour FROM entered_at AT TIME ZONE 'UTC') =
                  EXTRACT(hour FROM NOW() AT TIME ZONE 'UTC')
              AND EXTRACT(dow  FROM entered_at AT TIME ZONE 'UTC') =
                  EXTRACT(dow  FROM NOW() AT TIME ZONE 'UTC')
        """, (crossing_id,))
        row = cur.fetchone()
        if row and row["sample_count"] and row["sample_count"] > 0:
            return dict(row)
        return None


def fetch_lane_breakdown(conn, crossing_id: int,
                          window_min: int = DEFAULT_WINDOW_MIN) -> list[dict]:
    """Per-lane stats for the recent window."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                lane,
                COUNT(*)                                   AS completed,
                ROUND(AVG(duration_sec)::NUMERIC, 1)       AS avg_sec,
                ROUND(MIN(duration_sec)::NUMERIC, 1)       AS min_sec,
                ROUND(MAX(duration_sec)::NUMERIC, 1)       AS max_sec,
                COUNT(*) FILTER (WHERE exited_at IS NULL)  AS still_active
            FROM vehicle_crossings
            WHERE crossing_id = %s
              AND entered_at > NOW() - INTERVAL '%s minutes'
              AND duration_sec > 0
            GROUP BY lane
            ORDER BY avg_sec DESC NULLS LAST
        """, (crossing_id, window_min))
        return [dict(r) for r in cur.fetchall()]


def fetch_crossing_id(conn, name: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (name,))
        row = cur.fetchone()
        return row[0] if row else None

# ---------------------------------------------------------------------------
# Estimation logic
# ---------------------------------------------------------------------------

def exponential_weight(seconds_ago: float, half_life_sec: float = 900) -> float:
    """
    Weight that decays by half every `half_life_sec` seconds.
    Default half-life = 15 minutes — a crossing from 15 min ago
    counts half as much as one from right now.
    """
    import math
    return math.exp(-0.693 * seconds_ago / half_life_sec)


def estimate_wait(
    active: list[dict],
    completed: list[dict],
    baseline: dict | None,
    window_min: int = DEFAULT_WINDOW_MIN,
) -> dict:
    """
    Combine the three signals into a single wait time estimate.

    Returns a dict with:
        wait_minutes        — best estimate
        confidence          — 'high' / 'medium' / 'low'
        method              — which signal(s) drove the estimate
        active_count        — vehicles currently in lane
        completed_count     — completed crossings used
        min_wait_minutes    — lower bound (fastest lane right now)
        max_wait_minutes    — upper bound (slowest active vehicle)
        details             — human-readable breakdown
    """

    # --- Signal 1: active vehicles (lower bound) ---
    active_elapsed = [v["elapsed_sec"] for v in active if v["elapsed_sec"]]
    current_min_wait = max(active_elapsed) / 60.0 if active_elapsed else None
    # The longest-waiting active vehicle sets the floor:
    # anyone joining the queue now will wait at least that long
    # PLUS the time for that vehicle to finish.

    # --- Signal 2: recent completed (exponentially weighted mean) ---
    weighted_sum = 0.0
    weight_total = 0.0
    for v in completed:
        w = exponential_weight(float(v["seconds_ago"]))
        weighted_sum += w * float(v["duration_sec"])
        weight_total += w

    recent_avg_sec = (weighted_sum / weight_total) if weight_total > 0 else None
    recent_avg_min = recent_avg_sec / 60.0 if recent_avg_sec is not None else None

    # --- Signal 3: hourly baseline ---
    baseline_min = (
        float(baseline["avg_duration_sec"]) / 60.0
        if baseline and baseline["avg_duration_sec"]
        else None
    )

    # --- Combine ---
    n_completed = len(completed)

    if n_completed >= MIN_SAMPLES_FOR_CONFIDENCE:
        # Enough recent data — trust the weighted average
        wait_min = recent_avg_min
        method = f"recent {n_completed} crossings (last {window_min} min), exp-weighted"
        confidence = "high" if n_completed >= 15 else "medium"

    elif n_completed > 0 and baseline_min is not None:
        # Blend recent with baseline (weight by sample count)
        blend = (n_completed * recent_avg_min + 3 * baseline_min) / (n_completed + 3)
        wait_min = blend
        method = f"blend of {n_completed} recent crossings + historical baseline"
        confidence = "medium"

    elif baseline_min is not None:
        wait_min = baseline_min
        method = "historical baseline for this hour/weekday only"
        confidence = "low"

    elif current_min_wait is not None:
        # Only active vehicles, no completions at all
        wait_min = current_min_wait
        method = "active vehicle elapsed time only (lower bound)"
        confidence = "low"

    else:
        return {
            "wait_minutes":     None,
            "confidence":       "none",
            "method":           "no data available",
            "active_count":     len(active),
            "completed_count":  0,
            "min_wait_minutes": None,
            "max_wait_minutes": None,
            "details":          "No vehicle data in the selected window.",
        }

    # Bounds
    all_durations = [float(v["duration_sec"]) for v in completed if v["duration_sec"]]
    min_wait = min(all_durations) / 60.0 if all_durations else None
    max_wait = max(all_durations) / 60.0 if all_durations else None

    # If active vehicles imply a longer wait, raise the floor
    if current_min_wait and current_min_wait > (wait_min or 0):
        wait_min = current_min_wait
        method += " [raised by active vehicle floor]"

    return {
        "wait_minutes":     round(wait_min, 1),
        "confidence":       confidence,
        "method":           method,
        "active_count":     len(active),
        "completed_count":  n_completed,
        "min_wait_minutes": round(min_wait, 1) if min_wait else None,
        "max_wait_minutes": round(max_wait, 1) if max_wait else None,
    }

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_estimate(crossing_name: str, result: dict, lanes: list[dict],
                   window_min: int):
    bar_width = 30

    print(f"\n{'═' * 55}")
    print(f"  {crossing_name.upper()}  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═' * 55}")

    wait = result["wait_minutes"]
    if wait is None:
        print("  ⚠  No estimate available.")
        print(f"  {result['details']}")
        return

    # Visual bar
    max_display = 60
    filled = int(min(wait, max_display) / max_display * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\n  Estimated wait : {wait:.1f} min  [{bar}]")

    conf_icon = {"high": "●", "medium": "◑", "low": "○", "none": "✗"}
    print(f"  Confidence     : {conf_icon.get(result['confidence'], '?')} "
          f"{result['confidence'].upper()}")
    print(f"  Method         : {result['method']}")

    if result["min_wait_minutes"] and result["max_wait_minutes"]:
        print(f"  Range          : {result['min_wait_minutes']}–"
              f"{result['max_wait_minutes']} min "
              f"(last {window_min} min window)")

    print(f"\n  Active in lane : {result['active_count']} vehicle(s)")
    print(f"  Completed      : {result['completed_count']} crossing(s) "
          f"in last {window_min} min")

    if lanes:
        print(f"\n  ── Lane breakdown ──────────────────────────────")
        for lane in lanes:
            avg_s = lane.get("avg_sec") or 0
            print(f"  Lane {str(lane['lane']):>6}  "
                  f"avg {float(avg_s)/60:.1f} min  "
                  f"({lane['completed']} completed, "
                  f"{lane['still_active']} active)")


def print_history(conn, crossing_id: int, crossing_name: str,
                  days: int = 7):
    """Print hourly average wait times for the past N days."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                DATE_TRUNC('hour', entered_at) AS hour_bucket,
                COUNT(*)                        AS vehicles,
                ROUND(AVG(duration_sec)::NUMERIC / 60, 1) AS avg_wait_min,
                ROUND(MIN(duration_sec)::NUMERIC / 60, 1) AS min_wait_min,
                ROUND(MAX(duration_sec)::NUMERIC / 60, 1) AS max_wait_min
            FROM vehicle_crossings
            WHERE crossing_id = %s
              AND exited_at IS NOT NULL
              AND duration_sec > 0
              AND entered_at > NOW() - INTERVAL '%s days'
            GROUP BY 1
            ORDER BY 1 DESC
        """, (crossing_id, days))
        rows = cur.fetchall()

    if not rows:
        print(f"\n  No completed crossings in the last {days} days.")
        return

    print(f"\n{'═' * 55}")
    print(f"  {crossing_name.upper()} — hourly history (last {days} days)")
    print(f"{'═' * 55}")
    print(f"  {'Hour (UTC)':<22} {'Vehicles':>8} {'Avg':>6} {'Min':>6} {'Max':>6}")
    print(f"  {'-'*22} {'-'*8} {'-'*6} {'-'*6} {'-'*6}")
    for r in rows:
        print(f"  {str(r['hour_bucket']):<22} "
              f"{r['vehicles']:>8} "
              f"{r['avg_wait_min']:>5} m "
              f"{r['min_wait_min']:>5} m "
              f"{r['max_wait_min']:>5} m")


def load_multiplier(conn, crossing_id: int) -> float:
    """Get the best available multiplier for current hour."""
    import json
    with conn.cursor() as cur:
        cur.execute("""
                    SELECT multiplier, notes
                    FROM crossing_queue_multipliers
                    WHERE crossing_id = %s
                    """, (crossing_id,))
        row = cur.fetchone()
    if not row:
        return 1.0

    multiplier, notes = row
    tod = {}
    if notes and "tod=" in notes:
        try:
            tod = json.loads(notes.split("tod=")[1])
        except Exception:
            pass

    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 6 and tod.get("overnight"):
        return tod["overnight"]
    elif 6 <= hour < 12 and tod.get("morning"):
        return tod["morning"]
    elif 12 <= hour < 18 and tod.get("afternoon"):
        return tod["afternoon"]
    elif 18 <= hour < 24 and tod.get("evening"):
        return tod["evening"]
    return float(multiplier)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate wait times from vehicle_crossings data"
    )
    parser.add_argument("--crossing",      choices=CROSSINGS, default=None)
    parser.add_argument("--all-crossings", action="store_true")
    parser.add_argument("--window",        type=int, default=DEFAULT_WINDOW_MIN,
                        help="Minutes to look back for recent crossings (default: 45)")
    parser.add_argument("--history",       action="store_true",
                        help="Print hourly history for the past 7 days")
    parser.add_argument("--history-days",  type=int, default=7)
    args = parser.parse_args()

    if not args.crossing and not args.all_crossings:
        parser.error("Specify --crossing <name> or --all-crossings")

    targets = CROSSINGS if args.all_crossings else [args.crossing]

    conn = get_conn()

    for name in targets:
        cid = fetch_crossing_id(conn, name)
        if cid is None:
            print(f"\n  Crossing '{name}' not found in DB.")
            continue

        if args.history:
            print_history(conn, cid, name, days=args.history_days)
            continue

        active    = fetch_active_vehicles(conn, cid)
        completed = fetch_recent_completed(conn, cid, window_min=args.window)
        baseline  = fetch_hourly_baseline(conn, cid)
        lanes     = fetch_lane_breakdown(conn, cid, window_min=args.window)

        result = estimate_wait(active, completed, baseline, window_min=args.window)

        result = estimate_wait(active, completed, baseline, window_min=args.window)

        if result["wait_minutes"] is not None:
            m = load_multiplier(conn, cid)
            result["wait_minutes"] = round(result["wait_minutes"] * m, 1)
            if result["min_wait_minutes"]:
                result["min_wait_minutes"] = round(result["min_wait_minutes"] * m, 1)
            if result["max_wait_minutes"]:
                result["max_wait_minutes"] = round(result["max_wait_minutes"] * m, 1)

        print_estimate(name, result, lanes, window_min=args.window)

    conn.close()
    print()


if __name__ == "__main__":
    main()