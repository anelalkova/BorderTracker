"""
Background worker that watches saved snapshots and persists wait_estimator_v3
results into a dedicated v3 results table.

Usage:
    python snapshot_predict_worker.py --crossing blace
    python snapshot_predict_worker.py --all-crossings
    python snapshot_predict_worker.py --all-crossings --poll-seconds 10
    python snapshot_predict_worker.py --all-crossings --once
"""

import argparse
import os
import time
from pathlib import Path

import psycopg2
import psycopg2.extras

from border_crossings import CROSSINGS, WAIT_MODEL_DIR, init_db
from wait_estimator_v3 import estimate_and_save_v3_result


def get_crossing_id(conn, name: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (name,))
        row = cur.fetchone()
        return row[0] if row else None


def find_pending_snapshots(conn, crossing_names: list[str] | None = None, limit: int = 20) -> list[dict]:
    params = {"limit": limit}
    sql = """
        SELECT
            s.id           AS snapshot_id,
            s.crossing_id  AS crossing_id,
            c.name         AS crossing_name,
            s.captured_at  AS captured_at
        FROM snapshots s
        JOIN crossings c ON c.id = s.crossing_id
        LEFT JOIN wait_estimator_v3_results r ON r.snapshot_id = s.id
        WHERE r.id IS NULL
    """
    if crossing_names:
        sql += " AND c.name = ANY(%(crossing_names)s)"
        params["crossing_names"] = crossing_names
    sql += """
        ORDER BY s.captured_at ASC
        LIMIT %(limit)s
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def process_snapshot(conn, crossing_name: str, crossing_id: int, snapshot_id: int) -> int | None:
    model_path = WAIT_MODEL_DIR / f"{crossing_name}_proc_time.joblib"
    if not model_path.exists():
        print(f"[ML] No v3 model for {crossing_name} at {model_path.name}; marking as skipped.")
        _mark_snapshot_failed(conn, snapshot_id, f"no model: {model_path.name}")
        return None

    result = estimate_and_save_v3_result(
        conn,
        crossing_id=crossing_id,
        snapshot_id=snapshot_id,
        model_path=model_path,
    )
    print(
        f"[ML] Saved v3 result #{result['estimate_id']}  "
        f"crossing={crossing_name}  wait={result['estimated_wait_minutes']} min  "
        f"snapshot_id={result['snapshot_id']}"
    )
    return result["estimate_id"]

def run_worker(crossing_names: list[str] | None, poll_seconds: int, run_once: bool) -> None:
    conn = init_db()
    print("[WORKER] Snapshot predict worker started.")
    try:
        while True:
            pending = find_pending_snapshots(conn, crossing_names=crossing_names)
            if not pending:
                if run_once:
                    print("[WORKER] No pending snapshots found.")
                    return
                time.sleep(poll_seconds)
                continue

            for row in pending:
                try:
                    process_snapshot(
                        conn,
                        crossing_name=row["crossing_name"],
                        crossing_id=row["crossing_id"],
                        snapshot_id=row["snapshot_id"],
                    )
                except Exception as exc:
                    conn.rollback()
                    print(
                        f"[ML] Failed snapshot {row['snapshot_id']} for "
                        f"{row['crossing_name']}: {exc}"
                    )
                    # Insert a tombstone so this snapshot isn't retried forever
                    _mark_snapshot_failed(conn, row["snapshot_id"], str(exc))

            if run_once:
                return

            time.sleep(poll_seconds)  # <-- was missing, hammers DB with no delay on success
    finally:
        conn.close()


def _mark_snapshot_failed(conn, snapshot_id: int, reason: str) -> None:
    """Insert a failed placeholder so the snapshot is excluded from future polling."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO wait_estimator_v3_results
                    (crossing_id, snapshot_id, estimated_at, model_version, result_json)
                SELECT
                    s.crossing_id,
                    s.id,
                    s.captured_at,
                    'failed',
                    %s::jsonb
                FROM snapshots s
                WHERE s.id = %s
                ON CONFLICT DO NOTHING
            """, (psycopg2.extras.Json({"error": reason}), snapshot_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[ML] Could not mark snapshot {snapshot_id} as failed: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Background snapshot -> v3 prediction worker")
    parser.add_argument("--crossing", choices=list(CROSSINGS.keys()))
    parser.add_argument("--all-crossings", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.crossing and not args.all_crossings:
        args.all_crossings = True

    crossing_names = None if args.all_crossings else [args.crossing]
    run_worker(crossing_names=crossing_names, poll_seconds=args.poll_seconds, run_once=args.once)


if __name__ == "__main__":
    main()
