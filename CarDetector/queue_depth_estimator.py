"""
queue_depth_estimator.py
=========================
Learns a "queue depth multiplier" — the ratio between what borderalarm
users report as their total wait and what the camera directly observes
as crossing duration.

If the camera sees avg 5 min/vehicle but borderalarm reports 20 min,
the multiplier is 4x — meaning the invisible queue (beyond camera frame)
adds ~3× the camera-visible portion.

This multiplier is then used to scale up camera-based predictions
to account for the full queue, not just the visible portion.

The multiplier is stored per crossing in the DB and used by
wait_time_from_crossings.py and wait_time_model_v2.py at prediction time.

It only uses borderalarm reports flagged as 'ok' by borderalarm_filter.py.
Run borderalarm_filter.py first.

Usage:
    python queue_depth_estimator.py --crossing bogorodica
    python queue_depth_estimator.py --all
    python queue_depth_estimator.py --crossing bogorodica --apply
    python queue_depth_estimator.py --crossing bogorodica --show
"""

import argparse
import json
from datetime import datetime, timezone, timedelta

import numpy as np
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

# Maximum hour offset when matching borderalarm report to camera hour
MAX_HOUR_OFFSET = 1

# Multiplier is clamped to this range to prevent nonsense values
MIN_MULTIPLIER = 1.0
MAX_MULTIPLIER = 8.0

# Minimum matched pairs needed before we trust the multiplier
MIN_PAIRS = 3

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def ensure_multiplier_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS crossing_queue_multipliers (
                id              SERIAL PRIMARY KEY,
                crossing_id     INTEGER NOT NULL REFERENCES crossings(id),
                multiplier      REAL    NOT NULL,
                confidence      TEXT    NOT NULL,
                matched_pairs   INTEGER NOT NULL,
                computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                notes           TEXT,
                UNIQUE (crossing_id)
            );
        """)
    conn.commit()


def get_crossing_id(conn, name: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (name,))
        row = cur.fetchone()
        return row[0] if row else None


def fetch_camera_hourly(conn, crossing_id: int) -> dict:
    """hour_bucket (naive UTC) -> avg_duration_min"""
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


def fetch_borderalarm_ok(conn, crossing_id: int) -> list[dict]:
    """Only quality_flag='ok' reports."""
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT reported_at, wait_minutes
                FROM crowdsourced_waits
                WHERE crossing_id = %s
                  AND quality_flag = 'ok'
                ORDER BY reported_at
            """, (crossing_id,))
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        # quality_flag column may not exist yet
        return []


def save_multiplier(conn, crossing_id: int, multiplier: float,
                    confidence: str, pairs: int, notes: str):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO crossing_queue_multipliers
                (crossing_id, multiplier, confidence, matched_pairs,
                 computed_at, notes)
            VALUES (%s, %s, %s, %s, NOW(), %s)
            ON CONFLICT (crossing_id) DO UPDATE SET
                multiplier    = EXCLUDED.multiplier,
                confidence    = EXCLUDED.confidence,
                matched_pairs = EXCLUDED.matched_pairs,
                computed_at   = EXCLUDED.computed_at,
                notes         = EXCLUDED.notes
        """, (crossing_id, multiplier, confidence, pairs, notes))
    conn.commit()


def load_multiplier(conn, crossing_id: int) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT multiplier, confidence, matched_pairs, computed_at, notes
            FROM crossing_queue_multipliers
            WHERE crossing_id = %s
        """, (crossing_id,))
        row = cur.fetchone()
        return dict(row) if row else None

# ---------------------------------------------------------------------------
# Estimation logic
# ---------------------------------------------------------------------------

def match_pairs(camera_hourly: dict, ba_reports: list[dict]) -> list[dict]:
    """
    Match each borderalarm report to the closest camera hour bucket.
    Returns list of {ba_wait_min, cam_avg_min, ratio, reported_at, hour_offset}.
    """
    pairs = []

    for r in ba_reports:
        reported_at = r["reported_at"]
        if reported_at.tzinfo is None:
            reported_at = reported_at.replace(tzinfo=timezone.utc)

        bucket = reported_at.replace(minute=0, second=0, microsecond=0)

        best       = None
        best_delta = None

        for delta_h in range(-MAX_HOUR_OFFSET, MAX_HOUR_OFFSET + 1):
            candidate      = bucket + timedelta(hours=delta_h)
            candidate_naive = candidate.replace(tzinfo=None)
            if candidate_naive in camera_hourly:
                cam = camera_hourly[candidate_naive]
                if cam["avg_min"] > 0 and (best_delta is None or
                                            abs(delta_h) < abs(best_delta)):
                    best       = cam
                    best_delta = delta_h

        if best is None:
            continue

        ba_wait = float(r["wait_minutes"])
        cam_avg = best["avg_min"]

        # Skip if camera avg is near zero (momentary gap, not a real reading)
        if cam_avg < 1.0:
            continue

        ratio = ba_wait / cam_avg
        pairs.append({
            "reported_at":  reported_at,
            "ba_wait_min":  ba_wait,
            "cam_avg_min":  round(cam_avg, 2),
            "ratio":        round(ratio, 3),
            "hour_offset":  best_delta,
            "cam_count":    best["count"],
        })

    return pairs


def compute_multiplier(pairs: list[dict]) -> dict:
    """
    Robust estimate of the queue depth multiplier.
    Uses median (resistant to outliers) weighted by camera sample count.
    """
    if not pairs:
        return {
            "multiplier": 1.0,
            "confidence": "none",
            "pairs":      0,
            "notes":      "No matched pairs available.",
        }

    ratios  = np.array([p["ratio"] for p in pairs])
    weights = np.array([p["cam_count"] for p in pairs], dtype=float)

    # Weighted median
    sort_idx       = np.argsort(ratios)
    ratios_sorted  = ratios[sort_idx]
    weights_sorted = weights[sort_idx]
    cum_weights    = np.cumsum(weights_sorted)
    midpoint       = cum_weights[-1] / 2.0
    median_ratio   = float(ratios_sorted[cum_weights >= midpoint][0])

    # Clamp
    multiplier = float(np.clip(median_ratio, MIN_MULTIPLIER, MAX_MULTIPLIER))

    n = len(pairs)
    confidence = "high" if n >= 10 else "medium" if n >= MIN_PAIRS else "low"

    iqr = float(np.percentile(ratios, 75) - np.percentile(ratios, 25))
    notes = (
        f"median_ratio={median_ratio:.2f}, "
        f"IQR={iqr:.2f}, "
        f"min={ratios.min():.2f}, max={ratios.max():.2f}, "
        f"n={n}"
    )

    return {
        "multiplier": round(multiplier, 3),
        "confidence": confidence,
        "pairs":      n,
        "notes":      notes,
    }

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(crossing_name: str, pairs: list[dict], result: dict):
    print(f"\n{'═' * 55}")
    print(f"  {crossing_name.upper()} — Queue Depth Multiplier")
    print(f"{'═' * 55}")
    print(f"  Matched pairs  : {result['pairs']}")
    print(f"  Multiplier     : {result['multiplier']}×")
    print(f"  Confidence     : {result['confidence'].upper()}")
    print(f"  Notes          : {result['notes']}")

    if pairs:
        print(f"\n  ── Matched pairs ───────────────────────────────")
        print(f"  {'Reported':>20}  {'BA wait':>8}  {'Cam avg':>8}  {'Ratio':>6}")
        print(f"  {'-'*20}  {'-'*8}  {'-'*8}  {'-'*6}")
        for p in sorted(pairs, key=lambda x: x["reported_at"]):
            ts = p["reported_at"].strftime("%m-%d %H:%M")
            print(f"  {ts:>20}  "
                  f"{p['ba_wait_min']:>6.0f}m  "
                  f"{p['cam_avg_min']:>6.1f}m  "
                  f"{p['ratio']:>6.2f}×")

    print(f"\n  Interpretation:")
    m = result["multiplier"]
    if m <= 1.2:
        print(f"  Camera sees essentially the full queue (×{m}).")
    elif m <= 2.5:
        print(f"  Invisible queue adds ~{m-1:.1f}× the camera-visible portion (×{m}).")
        print(f"  Multiply camera estimates by {m} for total wait.")
    else:
        print(f"  Large invisible queue — camera sees only 1/{m:.1f} of total (×{m}).")
        print(f"  Camera estimates need significant upward correction.")


def show_current(conn, crossing_name: str, crossing_id: int):
    m = load_multiplier(conn, crossing_id)
    if not m:
        print(f"  No multiplier stored for '{crossing_name}'. Run without --show first.")
        return
    print(f"\n  Stored multiplier for {crossing_name}:")
    print(f"    Value      : {m['multiplier']}×")
    print(f"    Confidence : {m['confidence']}")
    print(f"    Pairs used : {m['matched_pairs']}")
    print(f"    Computed   : {m['computed_at']}")
    print(f"    Notes      : {m['notes']}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Queue depth multiplier estimator")
    parser.add_argument("--crossing", choices=CROSSINGS, default=None)
    parser.add_argument("--all",      action="store_true")
    parser.add_argument("--apply",    action="store_true",
                        help="Save multiplier to DB (default: just print)")
    parser.add_argument("--show",     action="store_true",
                        help="Show stored multiplier without recomputing")
    args = parser.parse_args()

    if not args.crossing and not args.all:
        parser.error("Specify --crossing <n> or --all")

    targets = CROSSINGS if args.all else [args.crossing]
    conn    = get_conn()
    ensure_multiplier_table(conn)

    for name in targets:
        cid = get_crossing_id(conn, name)
        if not cid:
            print(f"\n  '{name}' not found in DB.")
            continue

        if args.show:
            show_current(conn, name, cid)
            continue

        camera  = fetch_camera_hourly(conn, cid)
        ba_rpts = fetch_borderalarm_ok(conn, cid)

        if not camera:
            print(f"\n  No camera data for '{name}'.")
            continue

        if not ba_rpts:
            print(f"\n  No quality-filtered borderalarm reports for '{name}'.")
            print(f"  Run borderalarm_filter.py first, then re-run this script.")
            continue

        pairs  = match_pairs(camera, ba_rpts)
        result = compute_multiplier(pairs)

        print_report(name, pairs, result)

        if args.apply:
            save_multiplier(
                conn, cid,
                multiplier=result["multiplier"],
                confidence=result["confidence"],
                pairs=result["pairs"],
                notes=result["notes"],
            )
            print(f"\n  ✓ Multiplier saved to DB.")
        else:
            print(f"\n  (Run with --apply to save to DB)")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()