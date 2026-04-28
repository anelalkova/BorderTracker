# Macedonia Border Crossing – Vehicle Detection & Wait Time Estimator

Real-time YOLO vehicle detection on all 6 Macedonian border crossing cameras,
with periodic DB snapshots and LLM-powered wait time estimation.

---

## Files

| File | Purpose |
|---|---|
| `border_crossings.py` | Main detector – streams video, runs YOLO, saves snapshots |
| `wait_time_estimator.py` | Queries DB history + calls Claude API to estimate wait time |
| `schema.sql` | Database schema reference (auto-created by the detector) |

---

## Quick Start

```bash
pip install ultralytics opencv-python numpy anthropic

# Run detector on Bogorodica (default)
python border_crossings.py

# Run on Tabanovce, saving every 10 min
python border_crossings.py --crossing tabanovce --interval 10

# List all crossings
python border_crossings.py --list

# Calibrate lane polygons (shows raw frame, no detection)
python border_crossings.py --crossing blace --calibrate
```

---

## Crossings & Stream URLs (hardcoded)

| Key | Display Name | URL |
|---|---|---|
| `bogorodica` | Bogorodica (МК–ГР) | `…/stream/bogorodica.m3u8` |
| `blace` | Blace (МК–КС) | `…/stream/blace.m3u8` |
| `tabanovce` | Tabanovce (МК–СР) | `…/stream/tabanovce.m3u8` |
| `deve_bair` | Deve Bair (МК–БГ) | `…/stream/deve_bair.m3u8` |
| `kafasan` | Kafasan (МК–АЛ) | `…/stream/kafasan.m3u8` |
| `medzitlija` | Megjitlija (МК–ГР) | `…/stream/medzitlija.m3u8` |

**TODO (future):** Replace hardcoded URLs with a scraper that parses
`https://roads.org.mk/patna-mreza/video-kameri/` — the page loads streams
via JavaScript, so use `playwright` or `selenium` to get the rendered DOM,
then extract `<source src="...">` or the JS config object.

---

## Lane Calibration

Lane polygons are fractional (0.0–1.0 of frame size) and are rough defaults.
Each camera has a different angle and zoom level.

**To calibrate:**
1. Run `python border_crossings.py --crossing <name> --calibrate`
2. A clean frame appears – take a screenshot (press `S`)
3. Open the screenshot and note pixel coords of lane boundaries
4. Divide by frame width/height to get fractional coords
5. Update `CROSSINGS[<name>]["lanes"]` in `border_crossings.py`

---

## Database

SQLite file: `border_data.sqlite` (created automatically)

### Tables

**`crossings`** – static metadata for each crossing

**`snapshots`** – vehicle counts every N minutes
```
captured_at | total_vehicles | cars | motorcycles | buses | trucks | lane_breakdown (JSON) | fps
```

**`wait_time_estimates`** – LLM-generated estimates
```
estimated_at | estimated_wait_minutes | confidence | model_version | context_json
```

### Why 5 minutes?

- Short enough to track real queue build-up (queues form/clear over 15–60 min)
- Long enough to smooth out per-frame YOLO noise
- ~288 rows/day/crossing → ~630 K rows/year for all 6 → trivial for SQLite

---

## Wait Time Estimation

```bash
export ANTHROPIC_API_KEY=sk-...

# Estimate current wait at Tabanovce using last 12 snapshots (1 hour of data)
python wait_time_estimator.py --crossing tabanovce

# Use a wider 3-hour window
python wait_time_estimator.py --crossing bogorodica --window 36

# View recent snapshots without estimating
python wait_time_estimator.py --crossing blace --show-history
```

### How it works

1. The last `--window` snapshots are pulled from the DB (~1 h at 5-min cadence)
2. A JSON context is built: vehicle counts, time of day, day of week, vehicle mix
3. Claude receives a system prompt with border crossing processing rate knowledge
4. Claude returns: `estimated_wait_minutes`, `confidence`, `reasoning`, `queue_trend`
5. The estimate is saved to `wait_time_estimates` for historical tracking

### Future improvements

- Feed `v_hourly_averages` to give the LLM day-of-week / hour-of-day baseline patterns
- Add an `actual_wait_minutes` column to `wait_time_estimates` (manually logged or 
  from user-reported data) so you can measure LLM accuracy over time
- Run estimator on a cron every 15–30 minutes per crossing
- Build a simple web dashboard showing live counts + estimated wait per crossing
