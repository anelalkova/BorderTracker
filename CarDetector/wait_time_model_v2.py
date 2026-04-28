"""
wait_time_model_v2.py
=====================
Trains a gradient-boosted regression model using vehicle_crossings.duration_sec
as the primary ground truth — no crowdsourced data required.

Optionally blends in borderalarm reports that pass the quality filter
(quality_flag = 'ok') as a soft correction signal.

This replaces wait_time_model.py for the common case where you have
camera data but sparse / unreliable crowdsourced reports.

Target variable:
    avg_duration_min  — mean crossing duration per hour from vehicle_crossings

Features:
    - vehicle counts (avg, peak, cars, buses, trucks, heavy ratio)
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

import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
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
# 0.0 = ignore borderalarm entirely, 1.0 = treat equally
BORDERALARM_BLEND_WEIGHT = 0.3

# Minimum completed vehicle crossings needed to attempt training
MIN_CAMERA_ROWS = 20

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def get_crossing_id(conn, name: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM crossings WHERE name = %s", (name,))
        row = cur.fetchone()
        return row[0] if row else None


def load_camera_hourly(conn, crossing_id: int) -> pd.DataFrame:
    """
    Aggregate vehicle_crossings into hourly rows.
    Each row = one hour bucket with avg duration + vehicle counts.
    """
    return pd.read_sql("""
        SELECT
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
                PARTITION BY DATE_TRUNC('hour', vc.entered_at)
            )                                            AS peak_vehicles
        FROM vehicle_crossings vc
        WHERE vc.crossing_id = %(cid)s
          AND vc.exited_at IS NOT NULL
          AND vc.duration_sec > 0
        GROUP BY 1
        ORDER BY 1
    """, conn, params={"cid": crossing_id})


def load_snapshot_hourly(conn, crossing_id: int) -> pd.DataFrame:
    """
    Hourly snapshot aggregates for vehicle count features.
    Joined with camera data to enrich features where available.
    """
    return pd.read_sql("""
        SELECT
            DATE_TRUNC('hour', s.captured_at) AS hour_utc,
            ROUND(AVG(s.total_vehicles), 1)   AS snap_avg_vehicles,
            MAX(s.total_vehicles)              AS snap_peak_vehicles,
            ROUND(AVG(s.cars), 1)             AS snap_avg_cars,
            ROUND(AVG(s.buses), 1)            AS snap_avg_buses,
            ROUND(AVG(s.trucks), 1)           AS snap_avg_trucks
        FROM snapshots s
        WHERE s.crossing_id = %(cid)s
        GROUP BY 1
        ORDER BY 1
    """, conn, params={"cid": crossing_id})


def load_borderalarm_ok(conn, crossing_id: int) -> pd.DataFrame:
    """
    Load only quality_flag = 'ok' borderalarm reports, aggregated by hour.
    Falls back gracefully if quality_flag column doesn't exist.
    """
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
        """, conn, params={"cid": crossing_id})
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse hour/dow from the hour_utc bucket
    df["hour_utc"]    = pd.to_datetime(df["hour_utc"], utc=True)
    df["hour_of_day"] = df["hour_utc"].dt.hour.astype(float)
    df["day_of_week"] = df["hour_utc"].dt.dayofweek.astype(float)  # 0=Mon

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
    df["avg_vehicles"]  = df.get("snap_avg_vehicles",  df.get("avg_vehicles",  pd.Series(0, index=df.index))).fillna(0)
    df["peak_vehicles"] = df.get("snap_peak_vehicles", df.get("peak_vehicles", pd.Series(0, index=df.index))).fillna(0)
    df["avg_cars"]      = df.get("snap_avg_cars",      df.get("avg_cars",      pd.Series(0, index=df.index))).fillna(0)
    df["avg_buses"]     = df.get("snap_avg_buses",     df.get("avg_buses",     pd.Series(0, index=df.index))).fillna(0)
    df["avg_trucks"]    = df.get("snap_avg_trucks",    df.get("avg_trucks",    pd.Series(0, index=df.index))).fillna(0)

    # Heavy vehicle ratio
    total = df["avg_vehicles"].replace(0, 1)
    df["heavy_ratio"] = (df["avg_buses"] + df["avg_trucks"]) / total

    # Sample weight: more vehicles in hour = more reliable label
    df["sample_weight"] = np.log1p(df.get("vehicle_count", pd.Series(1, index=df.index)).fillna(1))

    return df


FEATURE_COLS = [
    "avg_vehicles", "peak_vehicles",
    "avg_cars", "avg_buses", "avg_trucks", "heavy_ratio",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "is_weekend", "is_morning_rush", "is_evening_rush", "is_night",
]

TARGET_COL = "target_wait_min"

# ---------------------------------------------------------------------------
# Build training dataset
# ---------------------------------------------------------------------------

def build_training_data(conn, crossing_id: int,
                         blend_borderalarm: bool = True) -> pd.DataFrame | None:
    """
    Merge camera hourly data with snapshot features and optionally
    blend in quality-filtered borderalarm reports.
    """
    camera   = load_camera_hourly(conn, crossing_id)
    snapshots = load_snapshot_hourly(conn, crossing_id)

    if camera.empty:
        return None

    # Primary target: avg duration from camera
    camera["hour_utc"] = pd.to_datetime(camera["hour_utc"], utc=True)
    camera["target_wait_min"] = camera["avg_duration_sec"].astype(float) / 60.0

    # Merge snapshot features if available
    if not snapshots.empty:
        snapshots["hour_utc"] = pd.to_datetime(snapshots["hour_utc"], utc=True)
        df = camera.merge(snapshots, on="hour_utc", how="left")
    else:
        df = camera.copy()

    # Optionally blend borderalarm 'ok' reports
    if blend_borderalarm:
        ba = load_borderalarm_ok(conn, crossing_id)
        if not ba.empty:
            ba["hour_utc"] = pd.to_datetime(ba["hour_utc"], utc=True)
            df = df.merge(ba, on="hour_utc", how="left")

            # Where we have a verified borderalarm reading, blend it in
            mask = df["ba_avg_wait_min"].notna()
            if mask.any():
                cam_w = 1.0 - BORDERALARM_BLEND_WEIGHT
                ba_w  = BORDERALARM_BLEND_WEIGHT
                df.loc[mask, "target_wait_min"] = (
                    cam_w * df.loc[mask, "target_wait_min"] +
                    ba_w  * df.loc[mask, "ba_avg_wait_min"]
                )
                # Increase sample weight for blended rows
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


def train(conn, crossing_name: str, crossing_id: int,
          model_type: str = "gbr", evaluate: bool = True,
          verbose: bool = True) -> Path | None:

    raw = build_training_data(conn, crossing_id, blend_borderalarm=True)

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

    if verbose:
        print(f"  Camera rows    : {len(df)}")
        print(f"  Wait range     : {y.min():.1f}–{y.max():.1f} min")
        print(f"  Avg wait       : {y.mean():.1f} min")

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

    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{crossing_name}_{model_type}.joblib"
    meta = {
        "crossing":      crossing_name,
        "model_type":    model_type,
        "trained_at":    datetime.now(timezone.utc).isoformat(),
        "n_samples":     len(X),
        "target":        "avg_duration_min_from_camera",
        "feature_cols":  FEATURE_COLS,
    }
    joblib.dump({"model": model, "meta": meta}, str(path))

    if verbose:
        print(f"\n  Model saved → {path}")
    return path


def predict_now(conn, crossing_id: int, model_path: Path) -> dict:
    """Predict wait time using the latest snapshot."""
    bundle  = joblib.load(str(model_path))
    model   = bundle["model"]
    meta    = bundle["meta"]

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT total_vehicles, cars, buses, trucks, captured_at
            FROM snapshots
            WHERE crossing_id = %s
            ORDER BY captured_at DESC LIMIT 1
        """, (crossing_id,))
        snap = cur.fetchone()

    now  = datetime.now(timezone.utc)
    ts   = dict(snap)["captured_at"] if snap else now
    hour = ts.hour if hasattr(ts, "hour") else now.hour
    dow  = ts.weekday() if hasattr(ts, "weekday") else now.weekday()

    total  = float(snap["total_vehicles"]) if snap else 0
    cars   = float(snap["cars"])   if snap else 0
    buses  = float(snap["buses"])  if snap else 0
    trucks = float(snap["trucks"]) if snap else 0
    heavy  = (buses + trucks) / max(total, 1)

    row = {
        "avg_vehicles":  total, "peak_vehicles": total,
        "avg_cars":      cars,  "avg_buses":     buses,
        "avg_trucks":    trucks,"heavy_ratio":   heavy,
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dow  / 7),
        "dow_cos":  np.cos(2 * np.pi * dow  / 7),
        "is_weekend":      float(dow >= 5),
        "is_morning_rush": float(7 <= hour <= 10),
        "is_evening_rush": float(15 <= hour <= 20),
        "is_night":        float(hour <= 5 or hour >= 22),
        "sample_weight":   1.0,
    }

    X = np.array([[row[f] for f in FEATURE_COLS]])
    y = float(model.predict(X)[0])

    return {
        "wait_minutes": round(max(y, 0), 1),
        "model_type":   meta["model_type"],
        "trained_at":   meta["trained_at"],
        "n_train":      meta["n_samples"],
        "snapshot_at":  str(ts),
        "queue_size":   int(total),
    }


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
    parser = argparse.ArgumentParser(description="Wait time model v2 (camera-first)")
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
        parser.error("Specify --crossing <n> or --all-crossings")

    targets = CROSSINGS if args.all_crossings else [args.crossing]
    conn    = get_conn()

    for name in targets:
        print(f"\n{'='*55}")
        print(f"  {name}")
        print(f"{'='*55}")

        cid = get_crossing_id(conn, name)
        if not cid:
            print(f"  Crossing not found in DB.")
            continue

        if args.train:
            print(f"  Training {args.model_type} model (camera-first) …\n")
            train(conn, name, cid,
                  model_type=args.model_type,
                  evaluate=args.eval or True)

        if args.predict:
            path = MODEL_DIR / f"{name}_{args.model_type}.joblib"
            if not path.exists():
                print(f"  No model at {path}. Run --train first.")
                continue
            result = predict_now(conn, cid, path)
            print(f"  Snapshot at    : {result['snapshot_at']}")
            print(f"  Queue size     : {result['queue_size']} vehicles")
            print(f"  Predicted wait : {result['wait_minutes']} min")
            print(f"  Model          : {result['model_type']} "
                  f"(trained on {result['n_train']} camera-hours)")

        if args.feature_importance:
            path = MODEL_DIR / f"{name}_{args.model_type}.joblib"
            if path.exists():
                print_feature_importance(path)
            else:
                print(f"  No model at {path}.")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()