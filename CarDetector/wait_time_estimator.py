"""
wait_time_model.py
==================
Trains a gradient-boosted regression model to predict border crossing
wait times from camera-observed vehicle counts and time-of-day features.

Ground truth comes from the `crowdsourced_waits` table (borderalarm.com data).
Camera features come from `snapshots` and `vehicle_crossings`.

The trained model is saved as a .joblib file that the LLM estimator
(wait_time_estimator.py) can load as a fast fallback or cross-check.

Usage:
    python wait_time_estimator.py --crossing bogorodica --train
    python wait_time_estimator.py --crossing bogorodica --train --eval
    python wait_time_estimator.py --crossing bogorodica --predict  # use latest snapshot
    python wait_time_estimator.py --all-crossings --train          # train one model per crossing
    python wait_time_estimator.py --crossing bogorodica --feature-importance

Requirements:
    pip install scikit-learn pandas joblib psycopg2-binary
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
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

MODEL_DIR = Path("models")

CROSSINGS = [
    "bogorodica", "blace", "tabanovce",
    "deve_bair", "kafasan", "medzitlija",
]

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_conn():
    return psycopg2.connect(**DB_CONFIG)


def load_training_data(conn, crossing_name: str) -> pd.DataFrame:
    """
    Load the v_training_data view for a crossing.
    Rows where ground_truth_wait_minutes IS NULL are excluded by the view.
    """
    df = pd.read_sql("""
        SELECT *
        FROM v_training_data
        WHERE crossing = %(crossing)s
        ORDER BY hour_utc
    """, conn, params={"crossing": crossing_name})
    return df


def load_latest_snapshot(conn, crossing_name: str) -> dict | None:
    """Return the most recent snapshot row for live prediction."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT s.*, c.name AS crossing_name
            FROM snapshots s
            JOIN crossings c ON s.crossing_id = c.id
            WHERE c.name = %s
            ORDER BY s.captured_at DESC
            LIMIT 1
        """, (crossing_name,))
        row = cur.fetchone()
        return dict(row) if row else None


def load_crowdsourced_stats(conn, crossing_name: str) -> dict:
    """Return basic stats about available crowdsourced data."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COUNT(*), MIN(reported_at), MAX(reported_at),
                   ROUND(AVG(wait_minutes)::NUMERIC, 1),
                   ROUND(STDDEV(wait_minutes)::NUMERIC, 1)
            FROM crowdsourced_waits cw
            JOIN crossings c ON cw.crossing_id = c.id
            WHERE c.name = %s
        """, (crossing_name,))
        row = cur.fetchone()
        return {
            "count":    row[0],
            "min_date": row[1],
            "max_date": row[2],
            "avg_wait": row[3],
            "std_wait": row[4],
        }

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from raw training data.
    All columns in the returned df are numeric and ready for sklearn.
    """
    df = df.copy()

    # Time features
    df["hour_of_day"]    = df["hour_of_day"].astype(float)
    df["day_of_week"]    = df["day_of_week"].astype(float)
    df["is_weekend"]     = (df["day_of_week"] >= 5).astype(float)
    df["is_morning_rush"]= df["hour_of_day"].between(7, 10).astype(float)
    df["is_evening_rush"]= df["hour_of_day"].between(15, 20).astype(float)
    df["is_night"]       = (df["hour_of_day"].between(0, 5) |
                             df["hour_of_day"].between(22, 23)).astype(float)

    # Cyclical encoding of hour (captures continuity between 23:00 and 00:00)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Vehicle counts
    df["avg_vehicles"]     = df["avg_vehicles"].fillna(0)
    df["peak_vehicles"]    = df["peak_vehicles"].fillna(0)
    df["avg_cars"]         = df["avg_cars"].fillna(0)
    df["avg_buses"]        = df["avg_buses"].fillna(0)
    df["avg_trucks"]       = df["avg_trucks"].fillna(0)

    # Bus/truck ratio (slow-processing vehicles)
    total = df["avg_vehicles"].replace(0, 1)
    df["heavy_ratio"] = (df["avg_buses"] + df["avg_trucks"]) / total

    # Crossing duration from tracker (may be null if no tracks yet)
    df["cam_avg_crossing_min"] = (
        df["cam_avg_crossing_sec"].fillna(0) / 60.0
    )
    df["tracked_vehicles"] = df["tracked_vehicles"].fillna(0)

    # Sample size weight (more reports = more trustworthy label)
    df["label_weight"] = np.log1p(df["ground_truth_sample_size"].fillna(1))

    return df


FEATURE_COLS = [
    "avg_vehicles", "peak_vehicles",
    "avg_cars", "avg_buses", "avg_trucks",
    "heavy_ratio",
    "cam_avg_crossing_min", "tracked_vehicles",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "is_weekend", "is_morning_rush", "is_evening_rush", "is_night",
]

TARGET_COL = "ground_truth_wait_minutes"

# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def build_model(model_type: str = "gbr"):
    """
    Return a sklearn pipeline (scaler + estimator).
    GBR is the default — it handles non-linearities well for small datasets.
    """
    if model_type == "gbr":
        estimator = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=3,
            random_state=42,
        )
    elif model_type == "rf":
        estimator = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
        )
    elif model_type == "ridge":
        estimator = Ridge(alpha=10.0)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  estimator),
    ])


def train(conn, crossing_name: str, model_type: str = "gbr",
          evaluate: bool = True, verbose: bool = True) -> Path | None:
    """
    Train and save a wait-time model for the given crossing.
    Returns the path to the saved .joblib file, or None on failure.
    """
    raw = load_training_data(conn, crossing_name)
    if raw.empty:
        print(f"  No training data for '{crossing_name}'. "
              f"Collect more snapshots + crowdsourced reports first.")
        return None

    if verbose:
        print(f"  Loaded {len(raw)} training rows  "
              f"(wait range: {raw[TARGET_COL].min():.0f}–{raw[TARGET_COL].max():.0f} min)")

    df     = engineer_features(raw)
    X      = df[FEATURE_COLS].values
    y      = df[TARGET_COL].values
    w      = df["label_weight"].values

    if len(X) < 10:
        print(f"  Too few rows ({len(X)}) to train. Need at least 10.")
        return None

    model = build_model(model_type)

    if evaluate:
        # Cross-validation (leave-out-last-20% as a hold-out)
        split  = max(1, int(len(X) * 0.8))
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        w_tr       = w[:split]

        model.fit(X_tr, y_tr, model__sample_weight=w_tr)
        y_pred = model.predict(X_te)

        mae = mean_absolute_error(y_te, y_pred)
        r2  = r2_score(y_te, y_pred)

        if verbose:
            print(f"  Evaluation (hold-out 20%):")
            print(f"    MAE : {mae:.1f} min")
            print(f"    R²  : {r2:.3f}")
            if len(X) >= 20:
                cv_scores = cross_val_score(
                    build_model(model_type), X, y,
                    cv=5, scoring="neg_mean_absolute_error",
                )
                print(f"    CV-5 MAE: {-cv_scores.mean():.1f} ± {cv_scores.std():.1f} min")

    # Re-train on all data
    model.fit(X, y, model__sample_weight=w)

    MODEL_DIR.mkdir(exist_ok=True)
    out_path = MODEL_DIR / f"{crossing_name}_{model_type}.joblib"
    meta = {
        "crossing":     crossing_name,
        "model_type":   model_type,
        "trained_at":   datetime.now(timezone.utc).isoformat(),
        "n_samples":    len(X),
        "feature_cols": FEATURE_COLS,
    }
    joblib.dump({"model": model, "meta": meta}, str(out_path))

    if verbose:
        print(f"  Model saved → {out_path}")

    return out_path


def predict_from_snapshot(snapshot: dict, model_path: Path) -> dict:
    """
    Make a prediction from a single raw snapshot row.
    Returns {"wait_minutes": float, "model_type": str, "trained_at": str}.
    """
    bundle = joblib.load(str(model_path))
    model  = bundle["model"]
    meta   = bundle["meta"]

    now = snapshot.get("captured_at", datetime.now(timezone.utc))
    if hasattr(now, "hour"):
        hour = now.hour
        dow  = now.weekday()   # 0=Mon, 6=Sun
    else:
        now  = datetime.now()
        hour = now.hour
        dow  = now.weekday()

    row = {
        "avg_vehicles":             snapshot.get("total_vehicles", 0),
        "peak_vehicles":            snapshot.get("total_vehicles", 0),
        "avg_cars":                 snapshot.get("cars", 0),
        "avg_buses":                snapshot.get("buses", 0),
        "avg_trucks":               snapshot.get("trucks", 0),
        "cam_avg_crossing_sec":     None,
        "tracked_vehicles":         None,
        "ground_truth_sample_size": None,
        "ground_truth_wait_minutes": 0,  # placeholder, not used
        "hour_of_day":              hour,
        "day_of_week":              dow,
    }
    df = pd.DataFrame([row])
    df = engineer_features(df)
    X  = df[FEATURE_COLS].values
    y  = model.predict(X)[0]

    return {
        "wait_minutes": round(float(y), 1),
        "model_type":   meta["model_type"],
        "trained_at":   meta["trained_at"],
        "n_train":      meta["n_samples"],
    }


def print_feature_importance(model_path: Path):
    bundle     = joblib.load(str(model_path))
    model      = bundle["model"]
    estimator  = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        imps = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        imps = np.abs(estimator.coef_)
    else:
        print("Model doesn't expose feature importances.")
        return

    pairs = sorted(zip(FEATURE_COLS, imps), key=lambda x: -x[1])
    print(f"\nFeature importances ({bundle['meta']['model_type']}):")
    for feat, imp in pairs:
        bar = "█" * int(imp * 50 / max(imps))
        print(f"  {feat:<30} {imp:.4f}  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wait time ML model trainer")
    parser.add_argument("--crossing",           default=None,
                        choices=CROSSINGS)
    parser.add_argument("--all-crossings",      action="store_true")
    parser.add_argument("--train",              action="store_true")
    parser.add_argument("--predict",            action="store_true",
                        help="Predict wait time from the latest snapshot")
    parser.add_argument("--eval",               action="store_true",
                        help="Show evaluation metrics during training")
    parser.add_argument("--feature-importance", action="store_true")
    parser.add_argument("--model-type",         default="gbr",
                        choices=["gbr", "rf", "ridge"],
                        help="gbr=GradientBoosting, rf=RandomForest, ridge=Ridge")
    args = parser.parse_args()

    if not args.crossing and not args.all_crossings:
        parser.error("Specify --crossing <n> or --all-crossings")

    targets = CROSSINGS if args.all_crossings else [args.crossing]

    conn = get_conn()
    print(f"PostgreSQL connected.\n")

    for name in targets:
        print(f"\n{'='*55}")
        print(f"  Crossing: {name}")
        print(f"{'='*55}")

        stats = load_crowdsourced_stats(conn, name)
        print(f"  Crowdsourced reports: {stats['count']}")
        if stats["count"]:
            print(f"  Date range   : {stats['min_date']} → {stats['max_date']}")
            print(f"  Avg wait     : {stats['avg_wait']} min  (σ={stats['std_wait']})")

        if args.train:
            print(f"\n  Training {args.model_type} model …")
            model_path = train(conn, name, model_type=args.model_type,
                               evaluate=args.eval or True)
            if not model_path:
                continue

        if args.predict:
            model_path = MODEL_DIR / f"{name}_{args.model_type}.joblib"
            if not model_path.exists():
                print(f"  No trained model found at {model_path}. Run --train first.")
                continue
            snapshot = load_latest_snapshot(conn, name)
            if not snapshot:
                print(f"  No snapshots found for '{name}'.")
                continue
            result = predict_from_snapshot(snapshot, model_path)
            print(f"\n  Latest snapshot   : {snapshot.get('captured_at')}")
            print(f"  Queue size        : {snapshot.get('total_vehicles')} vehicles")
            print(f"  Predicted wait    : {result['wait_minutes']} min")
            print(f"  Model             : {result['model_type']}"
                  f"  (trained on {result['n_train']} samples)")

        if args.feature_importance:
            model_path = MODEL_DIR / f"{name}_{args.model_type}.joblib"
            if model_path.exists():
                print_feature_importance(model_path)
            else:
                print(f"  No model found at {model_path}.")

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()