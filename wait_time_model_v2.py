"""
wait_time_model_v2.py
=====================
Trains a gradient-boosted regression model using vehicle_crossings.duration_sec
as the primary ground truth — no crowdsourced data required.

Operates at the LANE level: each lane gets its own feature rows during training,
and predictions are returned per-lane using the lane_details JSON from snapshots.

One model is trained per crossing (with lane_idx as a feature) rather than one
model per lane, so sparse data is shared across lanes of the same crossing.

Target variable:
    avg_duration_min  — mean crossing duration per hour per lane

Features:
    - lane index (ordinal, 0-based)
    - vehicle counts per lane (avg, peak, cars, buses, trucks, heavy ratio)
    - time-of-day (cyclical hour + binary rush/night flags)
    - day-of-week (cyclical)
    - queue depth (peak_vehicles as proxy for invisible queue)

Usage:
    python wait_time_model_v2.py --crossing bogorodica --train
    python wait_time_model_v2.py --crossing bogorodica --train --eval
    python wait_time_model_v2.py --crossing bogorodica --predict
    python wait_time_model_v2.py --crossing bogorodica --feature-importance
    python wait_time_model_v2.py --all-crossings --train
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json

import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
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

MODEL_DIR = Path("models_v2")

CROSSINGS = [
    "bogorodica", "blace", "tabanovce",
    "deve_bair", "kafasan", "medzitlija",
]

# Weight given to borderalarm 'ok' reports relative to camera rows
BORDERALARM_BLEND_WEIGHT = 0.3

# Minimum completed vehicle crossings needed to attempt training
MIN_CAMERA_ROWS = 20

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
# Data loading — lane-level
# ---------------------------------------------------------------------------

def load_camera_hourly_by_lane(engine, crossing_id: int) -> pd.DataFrame:
    """
    Aggregate vehicle_crossings into hourly rows grouped by lane.
    Filters on confidence >= MIN_CONFIDENCE.
    Returns one row per (lane_id, lane_name, hour_utc).
    """
    sql = """
        SELECT
            vc.lane                                      AS lane_name,
            DATE_TRUNC('hour', vc.entered_at)           AS hour_utc,
            COUNT(*)                                     AS vehicle_count,
            ROUND(AVG(vc.duration_sec)::NUMERIC, 2)     AS avg_duration_sec,
            ROUND(STDDEV(vc.duration_sec)::NUMERIC, 2)  AS std_duration_sec,
            ROUND(AVG(CASE WHEN vc.vehicle_type = 'car'   THEN 1.0 ELSE 0.0 END)
                  * COUNT(*), 1)                         AS avg_cars,
            ROUND(AVG(CASE WHEN vc.vehicle_type = 'bus'   THEN 1.0 ELSE 0.0 END)
                  * COUNT(*), 1)                         AS avg_buses,
            ROUND(AVG(CASE WHEN vc.vehicle_type = 'truck' THEN 1.0 ELSE 0.0 END)
                  * COUNT(*), 1)                         AS avg_trucks,
            COUNT(*)                                     AS avg_vehicles,
            MAX(COUNT(*)) OVER (
                PARTITION BY vc.lane, DATE_TRUNC('hour', vc.entered_at)
            )                                            AS peak_vehicles
        FROM vehicle_crossings vc
        WHERE vc.crossing_id = %(cid)s
          AND vc.exited_at IS NOT NULL
          AND vc.duration_sec > 0
        GROUP BY vc.lane, DATE_TRUNC('hour', vc.entered_at)
        ORDER BY vc.lane, DATE_TRUNC('hour', vc.entered_at)
    """
    return pd.read_sql(sql, engine, params={"cid": crossing_id})


def load_snapshot_hourly_by_lane(engine, crossing_id: int) -> pd.DataFrame:
    """
    Parse lane_details JSONB from snapshots into per-lane hourly rows.
    Returns columns: hour_utc, lane_name, snap_total, snap_cars, snap_buses, snap_trucks.
    """
    sql = """
        SELECT
            DATE_TRUNC('hour', s.captured_at)               AS hour_utc,
            kv.key                                           AS lane_name,
            ROUND(AVG((kv.value->>'total')::numeric), 1)    AS snap_avg_vehicles,
            MAX((kv.value->>'total')::int)                   AS snap_peak_vehicles,
            ROUND(AVG(COALESCE((kv.value->'by_type'->>'car')::numeric,   0)), 1) AS snap_avg_cars,
            ROUND(AVG(COALESCE((kv.value->'by_type'->>'bus')::numeric,   0)), 1) AS snap_avg_buses,
            ROUND(AVG(COALESCE((kv.value->'by_type'->>'truck')::numeric, 0)), 1) AS snap_avg_trucks
        FROM snapshots s,
             jsonb_each(s.lane_breakdown) AS kv(key, value)
        WHERE s.crossing_id = %(cid)s
        GROUP BY DATE_TRUNC('hour', s.captured_at), kv.key
        ORDER BY hour_utc, lane_name
    """
    try:
        return pd.read_sql(sql, engine, params={"cid": crossing_id})
    except Exception:
        return pd.DataFrame()


def load_borderalarm_ok(engine, crossing_id: int) -> pd.DataFrame:
    """Load quality_flag='ok' borderalarm reports aggregated by hour."""
    try:
        return pd.read_sql("""
            SELECT
                DATE_TRUNC('hour', reported_at) AS hour_utc,
                AVG(wait_minutes)               AS ba_avg_wait_min,
                COUNT(*)                        AS ba_reports
            FROM crowdsourced_waits
            WHERE crossing_id = %(cid)s
              AND (quality_flag = 'ok' OR quality_flag IS NULL)
            GROUP BY 1
            ORDER BY 1
        """, engine, params={"cid": crossing_id})
    except Exception:
        return pd.DataFrame()


def load_latest_snapshot_by_lane(conn, crossing_id: int) -> dict:
    """
    Fetch the most recent snapshot and parse lane_details into a dict:
        { "Bogorodica L1": {"total": 2, "cars": 0, "buses": 0, "trucks": 2}, ... }
    """
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
        "_captured_at": row["captured_at"],
        "_total_vehicles": int(row["total_vehicles"] or 0),
    }

    try:
        details = row["lane_breakdown"]
        if isinstance(details, str):
            details = json.loads(details)
        for lane_name, lane_data in details.items():
            by_type = lane_data.get("by_type", {})
            result[lane_name] = {
                "total":  int(lane_data.get("total", 0)),
                "cars":   int(by_type.get("car",   0)),
                "buses":  int(by_type.get("bus",   0)),
                "trucks": int(by_type.get("truck", 0)),
            }
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Multipliers
# ---------------------------------------------------------------------------

def load_multiplier(conn, crossing_id: int) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT multiplier, notes
            FROM crossing_queue_multipliers
            WHERE crossing_id = %s
        """, (crossing_id,))
        row = cur.fetchone()

    if not row:
        return {"global": 1.0, "tod": {}}

    tod = {}
    if row["notes"] and "tod=" in row["notes"]:
        try:
            tod = json.loads(row["notes"].split("tod=")[1])
        except Exception:
            pass

    return {"global": float(row["multiplier"]), "tod": tod}


def get_multiplier_for_hour(hour: int, multipliers: dict) -> float:
    tod = multipliers.get("tod", {})
    if 0 <= hour < 6 and tod.get("overnight"):
        return tod["overnight"]
    elif 6 <= hour < 12 and tod.get("morning"):
        return tod["morning"]
    elif 12 <= hour < 18 and tod.get("afternoon"):
        return tod["afternoon"]
    elif 18 <= hour < 24 and tod.get("evening"):
        return tod["evening"]
    return multipliers.get("global", 1.0)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "lane_idx",
    "avg_vehicles", "peak_vehicles",
    "avg_cars", "avg_buses", "avg_trucks", "heavy_ratio",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "is_weekend", "is_morning_rush", "is_evening_rush", "is_night",
]

TARGET_COL = "target_wait_min"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour_utc"]    = pd.to_datetime(df["hour_utc"], utc=True)
    df["hour_of_day"] = df["hour_utc"].dt.hour.astype(float)
    df["day_of_week"] = df["hour_utc"].dt.dayofweek.astype(float)

    # Cyclical time encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Binary time flags
    df["is_weekend"]      = (df["day_of_week"] >= 5).astype(float)
    df["is_morning_rush"] = df["hour_of_day"].between(7, 10).astype(float)
    df["is_evening_rush"] = df["hour_of_day"].between(15, 20).astype(float)
    df["is_night"]        = (
        df["hour_of_day"].between(0, 5) | df["hour_of_day"].between(22, 23)
    ).astype(float)

    # Vehicle counts — prefer snapshot columns, fall back to camera counts
    def _col(df, *names):
        for n in names:
            if n in df.columns:
                return df[n].fillna(0)
        return pd.Series(0, index=df.index)

    df["avg_vehicles"]  = _col(df, "snap_avg_vehicles",  "avg_vehicles")
    df["peak_vehicles"] = _col(df, "snap_peak_vehicles", "peak_vehicles")
    df["avg_cars"]      = _col(df, "snap_avg_cars",      "avg_cars")
    df["avg_buses"]     = _col(df, "snap_avg_buses",     "avg_buses")
    df["avg_trucks"]    = _col(df, "snap_avg_trucks",    "avg_trucks")

    total = df["avg_vehicles"].replace(0, 1)
    df["heavy_ratio"] = (df["avg_buses"] + df["avg_trucks"]) / total

    # Lane index: encode lane_name as a stable integer
    if "lane_name" in df.columns:
        lane_names = sorted(df["lane_name"].dropna().unique())
        lane_map   = {n: i for i, n in enumerate(lane_names)}
        df["lane_idx"] = df["lane_name"].map(lane_map).fillna(0).astype(float)
    else:
        df["lane_idx"] = 0.0

    # Sample weight
    df["sample_weight"] = np.log1p(
        df.get("vehicle_count", pd.Series(1, index=df.index)).fillna(1)
    )

    return df


# ---------------------------------------------------------------------------
# Build training dataset
# ---------------------------------------------------------------------------

def build_training_data(conn, engine, crossing_id: int,
                         blend_borderalarm: bool = True) -> pd.DataFrame | None:
    camera    = load_camera_hourly_by_lane(engine, crossing_id)
    snapshots = load_snapshot_hourly_by_lane(engine, crossing_id)

    if camera.empty:
        return None

    multipliers = load_multiplier(conn, crossing_id)
    camera["hour_utc"] = pd.to_datetime(camera["hour_utc"], utc=True)
    camera["target_wait_min"] = camera.apply(
        lambda r: (r["avg_duration_sec"] / 60.0),
        axis=1
    )

    # Merge per-lane snapshot features
    if not snapshots.empty:
        snapshots["hour_utc"] = pd.to_datetime(snapshots["hour_utc"], utc=True)
        df = camera.merge(snapshots, on=["hour_utc", "lane_name"], how="left")
    else:
        df = camera.copy()

    # Blend borderalarm (crossing-level, applied to all lanes equally)
    if blend_borderalarm:
        ba = load_borderalarm_ok(engine, crossing_id)
        if not ba.empty:
            ba["hour_utc"] = pd.to_datetime(ba["hour_utc"], utc=True)
            df = df.merge(ba, on="hour_utc", how="left")
            mask = df["ba_avg_wait_min"].notna()
            if mask.any():
                cam_w = 1.0 - BORDERALARM_BLEND_WEIGHT
                ba_w  = BORDERALARM_BLEND_WEIGHT
                df["vehicle_count"] = df["vehicle_count"].astype(float)
                df.loc[mask, "target_wait_min"] = (
                    cam_w * df.loc[mask, "target_wait_min"] +
                    ba_w  * df.loc[mask, "ba_avg_wait_min"]
                )
                df.loc[mask, "vehicle_count"] = (
                    df.loc[mask, "vehicle_count"].fillna(1) * 1.5
                )

    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(model_type: str = "gbr"):
    if model_type == "gbr":
        est = GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.04,
            max_depth=4, subsample=0.8,
            min_samples_leaf=2, random_state=42,
        )
    elif model_type == "rf":
        est = RandomForestRegressor(
            n_estimators=300, max_depth=8,
            min_samples_leaf=2, random_state=42,
        )
    elif model_type == "ridge":
        est = Ridge(alpha=10.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([("scaler", StandardScaler()), ("model", est)])


def train(conn, engine, crossing_name: str, crossing_id: int,
          model_type: str = "gbr", evaluate: bool = True,
          verbose: bool = True) -> Path | None:

    raw = build_training_data(conn, engine, crossing_id, blend_borderalarm=True)

    if raw is None or raw.empty:
        print(f"  No camera data for '{crossing_name}'.")
        return None

    df = engineer_features(raw)
    df = df.dropna(subset=[TARGET_COL] + FEATURE_COLS)

    if len(df) < MIN_CAMERA_ROWS:
        print(f"  Only {len(df)} rows — need {MIN_CAMERA_ROWS}. "
              f"Collect more camera data.")
        return None

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    w = df["sample_weight"].values

    # Report per-lane stats
    if verbose:
        print(f"  Total lane-hour rows : {len(df)}")
        if "lane_name" in df.columns:
            for lane, grp in df.groupby("lane_name"):
                y_l = grp[TARGET_COL].values
                print(f"    {lane:<20} n={len(grp):>4}  "
                      f"wait {y_l.min():.1f}–{y_l.max():.1f} min  "
                      f"avg {y_l.mean():.1f} min")
        print(f"  Overall wait range   : {y.min():.1f}–{y.max():.1f} min")
        print(f"  Overall avg wait     : {y.mean():.1f} min")

    model = build_model(model_type)

    if evaluate and len(X) >= 10:
        split  = max(1, int(len(X) * 0.8))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        w_tr       = w[:split]

        model.fit(X_tr, y_tr, model__sample_weight=w_tr)
        y_pred = model.predict(X_te)

        mae = mean_absolute_error(y_te, y_pred)
        r2  = r2_score(y_te, y_pred)

        if verbose:
            print(f"\n  Hold-out eval (20%):")
            print(f"    MAE : {mae:.2f} min")
            print(f"    R²  : {r2:.3f}")
            if len(X) >= 20:
                cv = cross_val_score(
                    build_model(model_type), X, y,
                    cv=5, scoring="neg_mean_absolute_error",
                )
                print(f"    CV-5 MAE: {-cv.mean():.2f} ± {cv.std():.2f} min")

    # Final fit on all data
    model.fit(X, y, model__sample_weight=w)

    # Store the lane name→index mapping in metadata
    lane_map = {}
    if "lane_name" in df.columns:
        lane_names = sorted(df["lane_name"].dropna().unique())
        lane_map   = {n: i for i, n in enumerate(lane_names)}

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{crossing_name}_{model_type}.joblib"
    meta = {
        "crossing":      crossing_name,
        "model_type":    model_type,
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "n_samples":     len(X),
        "target":        "avg_duration_min_from_camera_per_lane",
        "feature_cols":  FEATURE_COLS,
        "lane_map":      lane_map,
    }
    joblib.dump({"model": model, "meta": meta}, str(path))

    if verbose:
        print(f"\n  Model saved → {path}")
    return path


# ---------------------------------------------------------------------------
# Prediction — per lane
# ---------------------------------------------------------------------------

def predict_now(conn, crossing_id: int, model_path: Path) -> dict:
    """
    Predict wait time for each lane using the latest snapshot's lane_details.
    Returns a dict keyed by lane name, plus crossing-level totals.
    """
    bundle = joblib.load(str(model_path))
    model  = bundle["model"]
    meta   = bundle["meta"]
    lane_map: dict = meta.get("lane_map", {})

    snap_by_lane = load_latest_snapshot_by_lane(conn, crossing_id)

    captured_at    = snap_by_lane.pop("_captured_at", datetime.now(timezone.utc))
    total_vehicles = snap_by_lane.pop("_total_vehicles", 0)

    ts   = captured_at
    hour = ts.hour   if hasattr(ts, "hour")    else datetime.now(timezone.utc).hour
    dow  = ts.weekday() if hasattr(ts, "weekday") else datetime.now(timezone.utc).weekday()

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin  = np.sin(2 * np.pi * dow  / 7)
    dow_cos  = np.cos(2 * np.pi * dow  / 7)
    is_weekend      = float(dow >= 5)
    is_morning_rush = float(7 <= hour <= 10)
    is_evening_rush = float(15 <= hour <= 20)
    is_night        = float(hour <= 5 or hour >= 22)

    lane_results = {}

    if not snap_by_lane:
        # No lane data — predict crossing-level with lane_idx=0
        row = {
            "lane_idx": 0.0,
            "avg_vehicles": 0.0, "peak_vehicles": 0.0,
            "avg_cars": 0.0, "avg_buses": 0.0, "avg_trucks": 0.0,
            "heavy_ratio": 0.0,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "dow_sin": dow_sin,   "dow_cos": dow_cos,
            "is_weekend": is_weekend, "is_morning_rush": is_morning_rush,
            "is_evening_rush": is_evening_rush, "is_night": is_night,
            "sample_weight": 1.0,
        }
        X = np.array([[row[f] for f in FEATURE_COLS]])
        y = float(model.predict(X)[0])
        lane_results["unknown"] = {
            "wait_minutes": 0.0,  # queue_size is 0, so no wait
            "queue_size": 0,
            "per_vehicle_min": round(max(y, 0), 1),
        }
    else:
        for lane_name, counts in snap_by_lane.items():
            total  = float(counts["total"])
            cars   = float(counts["cars"])
            buses  = float(counts["buses"])
            trucks = float(counts["trucks"])
            heavy  = (buses + trucks) / max(total, 1)

            lane_idx = float(lane_map.get(lane_name, 0))

            row = {
                "lane_idx":      lane_idx,
                "avg_vehicles":  total,
                "peak_vehicles": total,
                "avg_cars":      cars,
                "avg_buses":     buses,
                "avg_trucks":    trucks,
                "heavy_ratio":   heavy,
                "hour_sin": hour_sin, "hour_cos": hour_cos,
                "dow_sin":  dow_sin,  "dow_cos":  dow_cos,
                "is_weekend":      is_weekend,
                "is_morning_rush": is_morning_rush,
                "is_evening_rush": is_evening_rush,
                "is_night":        is_night,
                "sample_weight":   1.0,
            }

            X = np.array([[row[f] for f in FEATURE_COLS]])
            y = float(model.predict(X)[0])  # per-vehicle processing time in minutes

            lane_results[lane_name] = {
                "wait_minutes": round(max(y * max(total, 1), 0), 1),  # ← multiply by queue
                "queue_size": int(total),
                "per_vehicle_min": round(max(y, 0), 1),  # useful to keep for debugging
            }

    return {
        "lanes":        lane_results,
        "snapshot_at":  str(ts),
        "total_vehicles": total_vehicles,
        "model_type":   meta["model_type"],
        "trained_at":   meta["trained_at"],
        "n_train":      meta["n_samples"],
    }


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def print_feature_importance(model_path: Path):
    bundle    = joblib.load(str(model_path))
    estimator = bundle["model"].named_steps["model"]
    meta      = bundle["meta"]

    if hasattr(estimator, "feature_importances_"):
        imps = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        imps = np.abs(estimator.coef_)
    else:
        print("Model doesn't expose importances.")
        return

    pairs = sorted(zip(FEATURE_COLS, imps), key=lambda x: -x[1])
    print(f"\nFeature importances ({meta['model_type']}, "
          f"trained on {meta['n_samples']} samples):")
    max_imp = max(imps)
    for feat, imp in pairs:
        bar = "█" * int(imp * 40 / max_imp)
        print(f"  {feat:<30} {imp:.4f}  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Wait time model v2 — per-lane, camera-first"
    )
    parser.add_argument("--crossing",           choices=CROSSINGS, default=None)
    parser.add_argument("--all-crossings",      action="store_true")
    parser.add_argument("--train",              action="store_true")
    parser.add_argument("--predict",            action="store_true")
    parser.add_argument("--eval",               action="store_true")
    parser.add_argument("--feature-importance", action="store_true")
    parser.add_argument("--model-type",         default="gbr",
                        choices=["gbr", "rf", "ridge"])
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

        if args.train:
            print(f"  Training {args.model_type} model (per-lane, camera-first) …\n")
            train(conn, engine, name, cid,
                  model_type=args.model_type,
                  evaluate=args.eval or True)

        if args.predict:
            path = MODEL_DIR / f"{name}_{args.model_type}.joblib"
            if not path.exists():
                print(f"  No model at {path}. Run --train first.")
                continue
            result = predict_now(conn, cid, path)
            print(f"  Snapshot at      : {result['snapshot_at']}")
            print(f"  Total vehicles   : {result['total_vehicles']}")
            print(f"  Model            : {result['model_type']} "
                  f"(trained on {result['n_train']} lane-hours)\n")
            print(f"  {'Lane':<22} {'Queue':>6}  {'Wait':>8}")
            print(f"  {'-'*40}")
            for lane_name, lane_data in sorted(result["lanes"].items()):
                print(f"  {lane_name:<22} {lane_data['queue_size']:>6}  "
                      f"{lane_data['wait_minutes']:>6.1f} min")

        if args.feature_importance:
            path = MODEL_DIR / f"{name}_{args.model_type}.joblib"
            if path.exists():
                print_feature_importance(path)
            else:
                print(f"  No model at {path}.")

    conn.close()
    engine.dispose()
    print("\nDone.")


if __name__ == "__main__":
    main()