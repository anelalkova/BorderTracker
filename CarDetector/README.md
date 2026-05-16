# Macedonia Border Crossing – Vehicle Detection & Wait Time Estimator

Real-time YOLO vehicle detection on all 6 Macedonian border crossing cameras,
with periodic DB snapshots, crowdsourced wait time ingestion, and an ML model
for per-lane wait time prediction.

---

## Files

| File | Purpose |
|---|---|
| `border_crossings.py` | Main detector – streams video, runs YOLO, saves snapshots to DB |
| `wait_time_model_v2.py` | Trains a per-lane GBR model using camera `duration_sec` as ground truth; predicts current wait per lane |
| `borderalarm_scraper.py` | Scrapes crowdsourced wait reports from borderalarm.com and saves to DB |
| `borderalarm_filter.py` | Validates crowdsourced reports against camera data; flags suspect entries |
| `queue_depth_estimator.py` | Computes a queue depth multiplier per crossing by comparing borderalarm reports to camera durations |
| `schema.sql` | Database schema reference (auto-created by the detector) |

---

## Quick Start

### 1. Install dependencies

```bash
pip install ultralytics opencv-python numpy anthropic \
            psycopg2-binary sqlalchemy scikit-learn joblib \
            pandas requests beautifulsoup4
```

### 2. Set up the database

Ensure PostgreSQL is running locally with a `border_crossing` database:

```bash
psql -U postgres -c "CREATE DATABASE border_crossing;"
```

The schema is created automatically when you first run `border_crossings.py`.

### 3. Run the camera detector

```bash
# Run on Bogorodica (default), saving a snapshot every 5 minutes
python border_crossings.py

# Run on a specific crossing with a custom snapshot interval
python border_crossings.py --crossing tabanovce --interval 10

# List all available crossings
python border_crossings.py --list

# Calibrate lane polygons (shows raw frame, no detection)
python border_crossings.py --crossing blace --calibrate
```

### 4. Ingest crowdsourced wait times (optional but recommended)

```bash
# Scrape borderalarm.com once for a crossing
python borderalarm_scraper.py --crossing bogorodica --once

# Poll every 15 minutes (default)
python borderalarm_scraper.py --crossing bogorodica

# Scrape all configured crossings
python borderalarm_scraper.py --all
```

### 5. Filter crowdsourced reports

Run this after scraping to flag unreliable borderalarm reports before they
are used in model training:

```bash
python borderalarm_filter.py --crossing bogorodica
python borderalarm_filter.py --all

# Preview changes without writing to DB
python borderalarm_filter.py --crossing bogorodica --dry-run

# Show quality flag breakdown
python borderalarm_filter.py --crossing bogorodica --show-stats
```

### 6. Compute queue depth multipliers (optional)

Estimates how much of the queue is invisible to the camera and saves a
per-crossing multiplier to DB. Requires filtered borderalarm data from step 5.

```bash
python queue_depth_estimator.py --crossing bogorodica

# Save the multiplier to DB for use during prediction
python queue_depth_estimator.py --crossing bogorodica --apply

# Show the currently stored multiplier
python queue_depth_estimator.py --crossing bogorodica --show
```

### 7. Train the wait time model

Trains a gradient-boosted regression model per crossing, using camera
`duration_sec` as ground truth. Run after collecting at least a few hours
of camera data (minimum 20 lane-hour rows).

```bash
python wait_time_model_v2.py --crossing bogorodica --train

# Train and evaluate (hold-out + 5-fold CV)
python wait_time_model_v2.py --crossing bogorodica --train --eval

# Train all crossings at once
python wait_time_model_v2.py --all-crossings --train

# Choose a different model type (gbr | rf | ridge)
python wait_time_model_v2.py --crossing bogorodica --train --model-type rf
```

### 8. Predict current wait times

```bash
python wait_time_model_v2.py --crossing bogorodica --predict
```

Example output:

```
  Snapshot at      : 2026-04-29 14:00:24+02:00
  Total vehicles   : 4
  Model            : gbr (trained on 116 lane-hours)

  Lane                    Queue      Wait
  ----------------------------------------
  Bogorodica L1               0      3.0 min
  Bogorodica L5               4      6.4 min   (4 cars × 1.6 min/vehicle)
```

### 9. Inspect feature importances

```bash
python wait_time_model_v2.py --crossing bogorodica --feature-importance
```

---

## Crossings & Stream URLs

| Key | Display Name | Neighbours |
|---|---|---|
| `bogorodica` | Bogorodica | МК–ГР (Greece) |
| `blace` | Blace | МК–КС (Kosovo) |
| `tabanovce` | Tabanovce | МК–СР (Serbia) |
| `deve_bair` | Deve Bair | МК–БГ (Bulgaria) |
| `kafasan` | Kafasan | МК–АЛ (Albania) |
| `medzitlija` | Megjitlija | МК–ГР (Greece) |

Stream URLs are hardcoded in `border_crossings.py` as `.m3u8` HLS streams.

**TODO:** Replace hardcoded URLs with a scraper that parses
`https://roads.org.mk/patna-mreza/video-kameri/` — the page loads streams
via JavaScript, so use `playwright` or `selenium` to get the rendered DOM,
then extract `<source src="...">` or the JS config object.

Crossings with borderalarm.com pages: `bogorodica`, `tabanovce`, `blace`, `medzitlija`.
`deve_bair` and `kafasan` are not yet configured — add slugs to `BORDERALARM_SLUGS`
in `borderalarm_scraper.py` if pages become available.

---

## Lane Calibration

Lane polygons are fractional (0.0–1.0 of frame size) and are rough defaults.
Each camera has a different angle and zoom level.

1. Run `python border_crossings.py --crossing <name> --calibrate`
2. A clean frame appears – take a screenshot (press `S`)
3. Open the screenshot and note pixel coords of lane boundaries
4. Divide by frame width/height to get fractional coords
5. Update `CROSSINGS[<name>]["lanes"]` in `border_crossings.py`

---

## Database

PostgreSQL database: `border_crossing`

### Tables

| Table | Description |
|---|---|
| `crossings` | Static metadata for each crossing |
| `snapshots` | YOLO vehicle counts every N minutes, with per-lane JSONB breakdown |
| `vehicle_crossings` | Individual vehicle records with entry/exit timestamps and `duration_sec` |
| `crowdsourced_waits` | Scraped borderalarm reports with `quality_flag` and `camera_avg_min` |
| `crossing_queue_multipliers` | Per-crossing queue depth multipliers computed from borderalarm vs camera ratio |
| `wait_time_estimates` | LLM-generated wait estimates (legacy, from `wait_time_estimator.py`) |

### Recommended snapshot interval

5 minutes — short enough to track queue build-up, long enough to smooth
per-frame YOLO noise. Yields ~288 rows/day/crossing.

---

## Typical Workflow

```
border_crossings.py          ← collect camera data continuously
        ↓
borderalarm_scraper.py       ← collect crowdsourced reports (every 15 min)
        ↓
borderalarm_filter.py        ← flag suspect reports
        ↓
queue_depth_estimator.py     ← compute queue multipliers (optional)
        ↓
wait_time_model_v2.py --train   ← train model (retrain periodically)
        ↓
wait_time_model_v2.py --predict ← predict current per-lane wait
```