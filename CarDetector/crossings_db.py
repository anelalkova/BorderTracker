from __future__ import annotations

import json
from pathlib import Path

import psycopg2.extras


SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def ensure_schema(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


def load_crossings(conn, require_borderalarm_slug: bool = False) -> dict[str, dict]:
    ensure_schema(conn)
    sql = """
        SELECT
            id,
            name,
            display_name,
            neighbor,
            borderalarm_slug,
            lane_config
        FROM crossings
    """
    if require_borderalarm_slug:
        sql += " WHERE borderalarm_slug IS NOT NULL AND borderalarm_slug <> ''"
    sql += " ORDER BY id"

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        rows = [dict(row) for row in cur.fetchall()]

    crossings: dict[str, dict] = {}
    for row in rows:
        lane_config = row.get("lane_config") or {}
        if isinstance(lane_config, str):
            lane_config = json.loads(lane_config)
        row["lane_config"] = lane_config
        crossings[row["name"]] = row
    return crossings


def load_crossing(conn, crossing_name: str) -> dict | None:
    return load_crossings(conn).get(crossing_name)


def load_crossing_names(conn, require_borderalarm_slug: bool = False) -> list[str]:
    return list(load_crossings(conn, require_borderalarm_slug=require_borderalarm_slug).keys())


def get_crossing_id(conn, crossing_name: str) -> int | None:
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (crossing_name,))
        row = cur.fetchone()
        return row[0] if row else None


def get_lane_count(conn, crossing_id: int) -> int:
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(jsonb_object_length(lane_config), 0)
            FROM crossings
            WHERE id = %s
            """,
            (crossing_id,),
        )
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0
