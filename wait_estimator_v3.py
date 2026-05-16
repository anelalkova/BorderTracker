"""
wait_estimator_v3.py
====================
Camera-only wait time estimator. No borderalarm required.

Architecture:
    Total wait = (estimated_full_queue) / (throughput_capacity_vpm)

Where:
    estimated_full_queue     = visible_queue × multiplier
    throughput_capacity_vpm  = 60 / avg_processing_sec
                               (capacity: how fast CAN this lane process,
                                not how many vehicles happened to arrive)

The GBR sub-model predicts avg_processing_sec using time-of-day,
day-of-week, and vehicle mix. Used when live data is sparse.

Key fixes vs v3 initial:
  - Target variable cleaned: per-lane 95th-percentile cap on duration_sec
    to remove tracker artifacts (stuck vehicles, late re-acquisition).
  - vpm derived from processing time (capacity), not wall-time count.
    Wall-time count measures utilisation, not lane capacity — they diverge
    badly during low-traffic hours.
  - Three-tier fallback for sparse lanes:
      1. live median (>= MIN_LIVE_VEHICLES tracked this window)
      2. crossing-wide median from other lanes (same booth policy/staffing)
      3. GBR model prediction

Usage:
    python wait_estimator_v3.py --crossing bogorodica --train
    python wait_estimator_v3.py --crossing bogorodica --predict
    python wait_estimator_v3.py --crossing bogorodica --train --predict
    python wait_estimator_v3.py --all-crossings --train
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

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

MODEL_DIR = Path("models_v3")

CROSSINGS = [
    "bogorodica", "blace", "tabanovce",
    "deve_bair", "kafasan", "medzitlija",
]

# How many hours of live vehicle_crossings to use for throughput estimate
THROUGHPUT_LOOKBACK_HOURS = 2

# Minimum vehicles tracked in the lookback window before we trust live data.
# Below this, fall back to crossing median or GBR model.
MIN_LIVE_VEHICLES = 10

# Multiplier fallback if not stored in DB
DEFAULT_MULTIPLIER = 4.0

# Minimum rows needed to train
MIN_TRAIN_ROWS = 20

# Hard floor/ceiling on duration_sec used for training and throughput.
# Floor: anything under 10s is a tracker flicker, not a real crossing.
# Ceiling: hard cap at 10 min before per-lane percentile trim.
MIN_DURATION_SEC      = 10
MAX_DURATION_SEC_HARD = 600   # 10 min hard cap

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_engine():
    c = DB_CONFIG
    url = (
        f"postgresql+psycopg2://{c['user']}:{c['password']}"
        f"@{c['host']}:{c['port']}/{c['dbname']}"
    )
    return create_engine(url)


def get_crossing_id(conn, name: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (name,))
        row = cur.fetchone()
        return row[0] if row else None


# ---------------------------------------------------------------------------
# Multiplier loading
# ---------------------------------------------------------------------------

def load_multiplier(conn, crossing_id: int) -> dict:
    """
    Returns { "global": float, "tod": { "overnight": float, ... } }
    Falls back to DEFAULT_MULTIPLIER if not stored.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT multiplier, notes
            FROM crossing_queue_multipliers
            WHERE crossing_id = %s
        """, (crossing_id,))
        row = cur.fetchone()

    if not row:
        return {"global": DEFAULT_MULTIPLIER, "tod": {}}

    tod   = {}
    notes = row["notes"] or ""
    if "tod=" in notes:
        try:
            tod = json.loads(notes.split("tod=")[1])
        except Exception:
            pass

    return {"global": float(row["multiplier"]), "tod": tod}


def get_multiplier_for_hour(hour: int, multipliers: dict) -> float:
    tod = multipliers.get("tod", {})
    if   0  <= hour < 6  and tod.get("overnight"): return tod["overnight"]
    elif 6  <= hour < 12 and tod.get("morning"):   return tod["morning"]
    elif 12 <= hour < 18 and tod.get("afternoon"): return tod["afternoon"]
    elif 18 <= hour < 24 and tod.get("evening"):   return tod["evening"]
    return multipliers.get("global", DEFAULT_MULTIPLIER)


# ---------------------------------------------------------------------------
# GBR sub-model: predicts avg_processing_sec per lane per hour
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "lane_idx",
    "avg_vehicles", "peak_vehicles",
    "avg_cars", "avg_buses", "avg_trucks", "heavy_ratio",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "is_weekend", "is_morning_rush", "is_evening_rush", "is_night",
]

TARGET_COL = "avg_processing_sec"


def load_training_data(engine, crossing_id: int) -> pd.DataFrame | None:
    """
    One row per (lane, hour).
    Target = avg_duration_sec, cleaned per-lane at the 95th percentile.

    We pull raw vehicle-level data so we can apply the percentile trim
    per lane before aggregating — SQL-side percentile_cont would trim
    globally, masking that L1 has a different natural range than L4.
    """
    sql_raw = """
        SELECT
            vc.lane                             AS lane_name,
            DATE_TRUNC('hour', vc.entered_at)   AS hour_utc,
            vc.duration_sec,
            vc.vehicle_type,
            vc.avg_confidence
        FROM vehicle_crossings vc
        WHERE vc.crossing_id = %(cid)s
          AND vc.exited_at IS NOT NULL
          AND vc.duration_sec BETWEEN %(min_dur)s AND %(max_dur)s
        ORDER BY vc.lane, vc.entered_at
    """

    sql_snaps = """
        SELECT
            DATE_TRUNC('hour', s.captured_at)                AS hour_utc,
            kv.key                                           AS lane_name,
            AVG((kv.value->>'total')::numeric)               AS snap_avg_vehicles,
            MAX((kv.value->>'total')::int)                   AS snap_peak_vehicles,
            AVG(COALESCE((kv.value->'by_type'->>'car')::numeric,   0)) AS snap_avg_cars,
            AVG(COALESCE((kv.value->'by_type'->>'bus')::numeric,   0)) AS snap_avg_buses,
            AVG(COALESCE((kv.value->'by_type'->>'truck')::numeric, 0)) AS snap_avg_trucks
        FROM snapshots s,
             jsonb_each(s.lane_breakdown) AS kv(key, value)
        WHERE s.crossing_id = %(cid)s
        GROUP BY DATE_TRUNC('hour', s.captured_at), kv.key
    """

    raw = pd.read_sql(sql_raw, engine, params={
        "cid":     crossing_id,
        "min_dur": MIN_DURATION_SEC,
        "max_dur": MAX_DURATION_SEC_HARD,
    })

    if raw.empty:
        return None

    # ── Per-lane 95th-percentile trim ────────────────────────────────────
    # Removes tracker artifacts without discarding legitimately long waits.
    cleaned_rows = []
    for lane, grp in raw.groupby("lane_name"):
        p95     = grp["duration_sec"].quantile(0.95)
        trimmed = grp[grp["duration_sec"] <= p95].copy()
        n_dropped = len(grp) - len(trimmed)
        if n_dropped:
            print(f"    [{lane}] trimmed {n_dropped} rows above p95={p95:.0f}s")
        cleaned_rows.append(trimmed)

    raw = pd.concat(cleaned_rows, ignore_index=True)

    # ── Aggregate to hourly per-lane ──────────────────────────────────────
    agg = (
        raw.groupby(["lane_name", "hour_utc"])
        .agg(
            vehicle_count      = ("duration_sec", "count"),
            avg_processing_sec = ("duration_sec", "mean"),
            avg_cars           = ("vehicle_type", lambda s: (s == "car").sum()),
            avg_buses          = ("vehicle_type", lambda s: (s == "bus").sum()),
            avg_trucks         = ("vehicle_type", lambda s: (s == "truck").sum()),
        )
        .reset_index()
    )
    # Require at least 3 vehicles per lane-hour for a reliable mean
    agg = agg[agg["vehicle_count"] >= 3].copy()
    agg["avg_vehicles"]  = agg["vehicle_count"].astype(float)
    agg["peak_vehicles"] = agg["avg_vehicles"]   # overwritten by snaps below

    agg["hour_utc"] = pd.to_datetime(agg["hour_utc"], utc=True)

    # ── Merge snapshot vehicle counts (better visible-queue proxy) ────────
    try:
        snaps = pd.read_sql(sql_snaps, engine, params={"cid": crossing_id})
        snaps["hour_utc"] = pd.to_datetime(snaps["hour_utc"], utc=True)
        df = agg.merge(snaps, on=["hour_utc", "lane_name"], how="left")
        for col, snap_col in [
            ("avg_vehicles",  "snap_avg_vehicles"),
            ("peak_vehicles", "snap_peak_vehicles"),
            ("avg_cars",      "snap_avg_cars"),
            ("avg_buses",     "snap_avg_buses"),
            ("avg_trucks",    "snap_avg_trucks"),
        ]:
            if snap_col in df.columns:
                df[col] = df[snap_col].fillna(df[col])
    except Exception:
        df = agg.copy()

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_utc"]    = pd.to_datetime(df["hour_utc"], utc=True)
    df["hour_of_day"] = df["hour_utc"].dt.hour.astype(float)
    df["day_of_week"] = df["hour_utc"].dt.dayofweek.astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(float)
    df["is_morning_rush"] = df["hour_of_day"].between(7, 10).astype(float)
    df["is_evening_rush"] = df["hour_of_day"].between(15, 20).astype(float)
    df["is_night"]        = (
        df["hour_of_day"].between(0, 5) | df["hour_of_day"].between(22, 23)
    ).astype(float)

    for col in ["avg_vehicles", "peak_vehicles", "avg_cars", "avg_buses", "avg_trucks"]:
        df[col] = df.get(col, pd.Series(0, index=df.index)).fillna(0)

    total = df["avg_vehicles"].replace(0, 1)
    df["heavy_ratio"] = (df["avg_buses"] + df["avg_trucks"]) / total

    if "lane_name" in df.columns:
        lane_names = sorted(df["lane_name"].dropna().unique())
        lane_map   = {n: i for i, n in enumerate(lane_names)}
        df["lane_idx"] = df["lane_name"].map(lane_map).fillna(0).astype(float)
    else:
        df["lane_idx"] = 0.0

    df["sample_weight"] = np.log1p(
        df.get("vehicle_count", pd.Series(1, index=df.index)).fillna(1)
    )

    return df


def train(conn, engine, crossing_name: str, crossing_id: int,
          verbose: bool = True) -> Path | None:

    print(f"  Loading and cleaning training data …")
    raw = load_training_data(engine, crossing_id)
    if raw is None or raw.empty:
        print(f"  No camera data for '{crossing_name}'.")
        return None

    df = engineer_features(raw)
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)

    if len(df) < MIN_TRAIN_ROWS:
        print(f"  Only {len(df)} rows — need {MIN_TRAIN_ROWS}. Collect more data.")
        return None

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    w = df["sample_weight"].values

    if verbose:
        print(f"\n  Training on {len(df)} lane-hour rows (after outlier trim)")
        print(f"  Target: avg_processing_sec  "
              f"range={y.min():.1f}–{y.max():.1f}s  avg={y.mean():.1f}s")
        if "lane_name" in df.columns:
            for lane, grp in df.groupby("lane_name"):
                yl = grp[TARGET_COL].values
                cap = 60.0 / (yl.mean() / 60.0)
                print(f"    {lane:<22}  n={len(grp):>4}  "
                      f"{yl.min():.0f}–{yl.max():.0f}s  avg={yl.mean():.0f}s  "
                      f"(≈{yl.mean()/60:.2f} min/veh  →  cap {cap:.2f} vpm)")

    gbr = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.04,
        max_depth=4, subsample=0.8,
        min_samples_leaf=2, random_state=42,
    )
    model = Pipeline([("scaler", StandardScaler()), ("model", gbr)])

    # Hold-out eval on last 20% (time-ordered = a future test)
    split  = max(1, int(len(X) * 0.8))
    model.fit(X[:split], y[:split], model__sample_weight=w[:split])
    y_pred = model.predict(X[split:])
    mae    = mean_absolute_error(y[split:], y_pred)
    r2     = r2_score(y[split:], y_pred)

    if verbose:
        print(f"\n  Hold-out eval (last 20%):")
        print(f"    MAE : {mae:.1f}s  ({mae/60:.2f} min/veh)")
        print(f"    R²  : {r2:.3f}")
        if len(X) >= 20:
            cv = cross_val_score(
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", GradientBoostingRegressor(
                        n_estimators=300, learning_rate=0.04,
                        max_depth=4, subsample=0.8,
                        min_samples_leaf=2, random_state=42,
                    ))
                ]),
                X, y, cv=5, scoring="neg_mean_absolute_error",
            )
            print(f"    CV-5 MAE: {-cv.mean():.1f}s ± {cv.std():.1f}s  "
                  f"({-cv.mean()/60:.2f} min/veh)")

    # Final fit on all data
    model.fit(X, y, model__sample_weight=w)

    lane_map = {}
    if "lane_name" in df.columns:
        lane_names = sorted(df["lane_name"].dropna().unique())
        lane_map   = {n: i for i, n in enumerate(lane_names)}

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{crossing_name}_proc_time.joblib"
    joblib.dump({
        "model": model,
        "meta": {
            "crossing":     crossing_name,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
            "n_samples":    len(X),
            "target":       "avg_processing_sec",
            "feature_cols": FEATURE_COLS,
            "lane_map":     lane_map,
            "y_mean":       float(y.mean()),
            "y_std":        float(y.std()),
        }
    }, str(path))

    if verbose:
        print(f"\n  Model saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Throughput: live capacity from recent vehicle_crossings
# ---------------------------------------------------------------------------

def get_live_throughput(conn, crossing_id: int,
                        lookback_hours: int = THROUGHPUT_LOOKBACK_HOURS) -> dict:
    """
    Returns per-lane processing stats from the last N hours.

    We use MEDIAN processing time, not mean — more robust to single
    stuck vehicles inflating the window average.

    NOTE: we do NOT compute vpm as COUNT/minutes here because that
    measures utilisation (how busy the lane was), not capacity (how
    fast it CAN process). A nearly-empty lane still processes each
    vehicle in ~30s even if only 5 vehicles arrived in 2 hours.
    vpm is derived later as 60 / median_proc_sec.
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT
                lane,
                COUNT(*)                AS vehicles,
                AVG(duration_sec)       AS avg_proc_sec,
                PERCENTILE_CONT(0.5)
                    WITHIN GROUP (ORDER BY duration_sec) AS median_proc_sec,
                STDDEV(duration_sec)    AS std_proc_sec
            FROM vehicle_crossings
            WHERE crossing_id = %(cid)s
              AND entered_at > NOW() - INTERVAL '1 hour' * %(hours)s
              AND duration_sec BETWEEN %(min_dur)s AND %(max_dur)s
              AND exited_at IS NOT NULL
            GROUP BY lane
        """, {
            "cid":     crossing_id,
            "hours":   lookback_hours,
            "min_dur": MIN_DURATION_SEC,
            "max_dur": MAX_DURATION_SEC_HARD,
        })
        rows = cur.fetchall()

    result = {}
    for r in rows:
        if not r["lane"]:
            continue
        result[r["lane"]] = {
            "vehicles":        int(r["vehicles"]),
            "avg_proc_sec":    float(r["avg_proc_sec"]),
            "median_proc_sec": float(r["median_proc_sec"] or r["avg_proc_sec"]),
            "std_proc_sec":    float(r["std_proc_sec"] or 0),
        }
    return result


def get_latest_snapshot(conn, crossing_id: int) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT lane_breakdown, captured_at, total_vehicles
            FROM snapshots
            WHERE crossing_id = %s
            ORDER BY captured_at DESC LIMIT 1
        """, (crossing_id,))
        row = cur.fetchone()

    if not row:
        return {}

    result = {
        "_captured_at":    row["captured_at"],
        "_total_vehicles": int(row["total_vehicles"] or 0),
    }

    details = row["lane_breakdown"]
    if isinstance(details, str):
        details = json.loads(details)

    for lane_name, lane_data in (details or {}).items():
        by_type = lane_data.get("by_type", {})
        result[lane_name] = {
            "total":  int(lane_data.get("total", 0)),
            "cars":   int(by_type.get("car",   0)),
            "buses":  int(by_type.get("bus",   0)),
            "trucks": int(by_type.get("truck", 0)),
        }

    return result


def predict_proc_time_from_model(model_path: Path, lane_name: str,
                                  lane_counts: dict, ts: datetime) -> float | None:
    """Use GBR to predict avg_processing_sec when live data is sparse."""
    if not model_path or not model_path.exists():
        return None

    bundle   = joblib.load(str(model_path))
    model    = bundle["model"]
    meta     = bundle["meta"]
    lane_map = meta.get("lane_map", {})

    hour   = ts.hour
    dow    = ts.weekday()
    total  = float(lane_counts.get("total",  0))
    cars   = float(lane_counts.get("cars",   0))
    buses  = float(lane_counts.get("buses",  0))
    trucks = float(lane_counts.get("trucks", 0))
    heavy  = (buses + trucks) / max(total, 1)

    row = {
        "lane_idx":        float(lane_map.get(lane_name, 0)),
        "avg_vehicles":    total,
        "peak_vehicles":   total,
        "avg_cars":        cars,
        "avg_buses":       buses,
        "avg_trucks":      trucks,
        "heavy_ratio":     heavy,
        "hour_sin":        np.sin(2 * np.pi * hour / 24),
        "hour_cos":        np.cos(2 * np.pi * hour / 24),
        "dow_sin":         np.sin(2 * np.pi * dow  / 7),
        "dow_cos":         np.cos(2 * np.pi * dow  / 7),
        "is_weekend":      float(dow >= 5),
        "is_morning_rush": float(7 <= hour <= 10),
        "is_evening_rush": float(15 <= hour <= 20),
        "is_night":        float(hour <= 5 or hour >= 22),
    }

    X    = np.array([[row[f] for f in FEATURE_COLS]])
    pred = float(model.predict(X)[0])
    return max(pred, float(MIN_DURATION_SEC))


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

def estimate_wait(conn, crossing_id: int, multipliers: dict,
                  model_path: Path | None = None,
                  verbose: bool = False) -> dict:
    """
    Physics-based wait estimator.

    For each lane:
        full_queue   = visible_queue × multiplier
        proc_sec     = median processing time per vehicle
        capacity_vpm = 60 / proc_sec
        wait_minutes = full_queue / capacity_vpm
                     = (full_queue × proc_sec) / 60

    Three-tier fallback for proc_sec:
        1. Live median from recent vehicle_crossings (>= MIN_LIVE_VEHICLES)
        2. Crossing-wide median from other lanes (same booth policy)
        3. GBR model prediction for this lane/hour/conditions
    """
    snap        = get_latest_snapshot(conn, crossing_id)
    captured_at = snap.pop("_captured_at", datetime.now(timezone.utc))
    total_veh   = snap.pop("_total_vehicles", 0)
    throughput  = get_live_throughput(conn, crossing_id)

    ts   = captured_at
    hour = ts.hour if hasattr(ts, "hour") else datetime.now(timezone.utc).hour
    mult = get_multiplier_for_hour(hour, multipliers)

    # Crossing-wide median: lanes at the same crossing share booth staffing
    # and policy, so a lane with sparse data can borrow from its neighbours.
    all_live_proc = [
        v["median_proc_sec"]
        for v in throughput.values()
        if v["vehicles"] >= MIN_LIVE_VEHICLES
    ]
    crossing_median_proc = float(np.median(all_live_proc)) if all_live_proc else None

    results = {}

    for lane_name, counts in snap.items():
        visible_q = counts.get("total", 0)
        full_q    = visible_q * mult

        live_tp     = throughput.get(lane_name)
        data_source = "live"

        if live_tp and live_tp["vehicles"] >= MIN_LIVE_VEHICLES:
            proc_sec = live_tp["median_proc_sec"]

        elif crossing_median_proc is not None:
            proc_sec    = crossing_median_proc
            data_source = "crossing_median"

        else:
            data_source = "model"
            proc_sec    = predict_proc_time_from_model(
                model_path, lane_name, counts, ts
            )
            if proc_sec is None:
                # Absolute last resort: trained mean stored in model metadata
                if model_path and model_path.exists():
                    meta     = joblib.load(str(model_path))["meta"]
                    proc_sec = meta.get("y_mean", 90.0)
                else:
                    proc_sec = 90.0

        proc_sec     = max(proc_sec, float(MIN_DURATION_SEC))
        capacity_vpm = 60.0 / proc_sec
        wait_min     = (full_q / capacity_vpm) if full_q > 0 else 0.0

        results[lane_name] = {
            "visible_queue":   visible_q,
            "full_queue":      round(full_q, 1),
            "multiplier_used": round(mult, 2),
            "proc_sec":        round(proc_sec, 1),
            "capacity_vpm":    round(capacity_vpm, 3),
            "wait_minutes":    round(wait_min, 1),
            "data_source":     data_source,
            "live_vehicles":   live_tp["vehicles"] if live_tp else 0,
        }

        if verbose:
            print(f"  {lane_name:<22}  Q={visible_q}→{full_q:.0f}  "
                  f"proc={proc_sec:.0f}s  cap={capacity_vpm:.3f}vpm  "
                  f"wait={wait_min:.1f}min  [{data_source}]")

    # ── Crossing-level summary ────────────────────────────────────────────
    # Weighted average: lanes with more vehicles contribute more to the
    # crossing-level wait than empty lanes.
    total_visible = sum(d["visible_queue"] for d in results.values())

    if total_visible > 0:
        weighted_wait = sum(
            d["wait_minutes"] * d["visible_queue"]
            for d in results.values()
        ) / total_visible
    else:
        # No visible queue — crossing is essentially clear.
        # Use processing time of best lane as a floor (you still have to
        # drive through the booth even with nobody ahead of you).
        best_proc = min(
            (d["proc_sec"] for d in results.values()),
            default=30.0
        )
        weighted_wait = best_proc / 60.0

    # Confidence interval from multiplier uncertainty.
    # We know the median ratio (mult) but the IQR tells us how wide the
    # true multiplier distribution is. We use ±0.5×IQR around the median
    # as a reasonable 50% confidence interval, then scale to ~80% by using
    # the full IQR. Without storing IQR in the DB we approximate from the
    # known spread: overnight is tighter, afternoon/evening are wilder.
    # If tod multipliers are available we derive spread from their range.
    tod = multipliers.get("tod", {})
    tod_vals = [v for v in tod.values() if v is not None]
    if len(tod_vals) >= 2:
        # Spread across time-of-day buckets gives us a sense of variability
        mult_spread = (max(tod_vals) - min(tod_vals)) / 2.0
    else:
        # Fallback: assume ±40% of the multiplier as uncertainty
        mult_spread = mult * 0.4

    # Scale the wait estimate by (mult ± spread) / mult
    wait_low  = round(weighted_wait * max(1.0, mult - mult_spread) / mult, 1)
    wait_high = round(weighted_wait * (mult + mult_spread) / mult, 1)
    wait_low  = max(wait_low, 0.0)

    # Congestion label for human-readable output
    if weighted_wait < 5:
        congestion = "clear"
    elif weighted_wait < 15:
        congestion = "light"
    elif weighted_wait < 30:
        congestion = "moderate"
    elif weighted_wait < 60:
        congestion = "heavy"
    else:
        congestion = "severe"

    best_lane = min(results.items(), key=lambda x: x[1]["wait_minutes"])

    return {
        "lanes":          results,
        "snapshot_at":    str(ts),
        "total_vehicles": total_veh,
        "multiplier":     mult,
        "hour":           hour,
        # Crossing-level summary — this is the main output
        "crossing_wait": {
            "weighted_wait_min": round(weighted_wait, 1),
            "wait_low_min":      wait_low,
            "wait_high_min":     wait_high,
            "congestion":        congestion,
            "best_lane":         best_lane[0],
            "best_lane_wait":    best_lane[1]["wait_minutes"],
            "total_visible":     total_visible,
        },
    }


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------

def print_predict(result: dict, crossing_name: str):
    lanes = result["lanes"]
    cw    = result["crossing_wait"]

    # ── Crossing-level summary (the main answer) ──────────────────────────
    print(f"\n  {'═'*52}")
    print(f"  {crossing_name.upper()} — Expected Wait")
    print(f"  {'═'*52}")
    print(f"  {cw['weighted_wait_min']:.0f} min  "
          f"(range: {cw['wait_low_min']:.0f}–{cw['wait_high_min']:.0f} min)  "
          f"[{cw['congestion'].upper()}]")
    print(f"  Best lane : {cw['best_lane']}  ({cw['best_lane_wait']:.0f} min)")
    print(f"  {'─'*52}")
    print(f"  Snapshot       : {result['snapshot_at']}")
    print(f"  Visible queue  : {cw['total_visible']} vehicles  "
          f"(est. full queue: {cw['total_visible'] * result['multiplier']:.0f})")
    print(f"  Multiplier     : {result['multiplier']}×  (hour={result['hour']})")

    # ── Per-lane breakdown ────────────────────────────────────────────────
    print(f"\n  {'Lane':<22}  {'Visible':>7}  {'Full Q':>7}  "
          f"{'Proc(s)':>8}  {'Wait':>8}  Source")
    print(f"  {'-'*72}")

    for lane_name, d in sorted(lanes.items()):
        src  = f"{d['data_source']} ({d['live_vehicles']}v)"
        flag = " ◄ best" if lane_name == cw["best_lane"] else ""
        print(
            f"  {lane_name:<22}  "
            f"{d['visible_queue']:>7}  "
            f"{d['full_queue']:>7.0f}  "
            f"{d['proc_sec']:>8.1f}  "
            f"{d['wait_minutes']:>6.1f}min  "
            f"{src}{flag}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Camera-only wait estimator (physics + GBR processing time)"
    )
    parser.add_argument("--crossing",      choices=CROSSINGS, default=None)
    parser.add_argument("--all-crossings", action="store_true")
    parser.add_argument("--train",         action="store_true")
    parser.add_argument("--predict",       action="store_true")
    parser.add_argument("--verbose",       action="store_true")
    args = parser.parse_args()

    if not args.crossing and not args.all_crossings:
        parser.error("Specify --crossing <name> or --all-crossings")

    targets = CROSSINGS if args.all_crossings else [args.crossing]
    conn    = get_conn()
    engine  = get_engine()

    for name in targets:
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        cid = get_crossing_id(conn, name)
        if not cid:
            print(f"  Crossing not found in DB.")
            continue

        model_path = MODEL_DIR / f"{name}_proc_time.joblib"

        if args.train:
            print(f"  Training processing-time sub-model …\n")
            train(conn, engine, name, cid, verbose=True)

        if args.predict:
            multipliers = load_multiplier(conn, cid)
            result      = estimate_wait(
                conn, cid, multipliers,
                model_path=model_path if model_path.exists() else None,
                verbose=args.verbose,
            )
            print_predict(result, name)

    conn.close()
    engine.dispose()
    print("\nDone.")


if __name__ == "__main__":
    main()